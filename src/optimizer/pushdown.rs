//! Predicate pushdown.
//!
//! Filters are split into conjuncts and pushed as far toward the scans as
//! semantics allow:
//!
//! * through projections (column references are replaced by the projected
//!   expressions) and through sorts without a fetch limit;
//! * below an aggregate when they only reference grouping keys;
//! * into the matching side of an inner join; a conjunct `l = r` that relates
//!   the two sides becomes an equi-join key, anything else spanning both sides
//!   becomes the join's residual filter;
//! * for a LEFT join, only into the preserved (left) side from above, and only
//!   into the right side from the join's own ON condition. A WHERE predicate
//!   that rejects NULLs from the right side turns the LEFT join into an INNER
//!   join first;
//! * finally into the scan itself, where it runs before data leaves the
//!   operator (and can prune Parquet row groups).

use std::collections::HashSet;

use super::OptimizerRule;
use crate::error::Result;
use crate::plan::expr::{conjunction, split_conjunction, BinaryOp, ColId, Expr, ScalarFunc};
use crate::plan::logical::{JoinType, LogicalPlan};

pub struct PredicatePushdown;

impl OptimizerRule for PredicatePushdown {
    fn name(&self) -> &'static str {
        "predicate_pushdown"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        push(plan, Vec::new())
    }
}

fn with_filter(plan: LogicalPlan, preds: Vec<Expr>) -> LogicalPlan {
    match conjunction(preds) {
        Some(p) => LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: p,
        },
        None => plan,
    }
}

fn subset(e: &Expr, ids: &HashSet<ColId>) -> bool {
    e.column_ids().is_subset(ids)
}

/// `Some((l, r))` when `e` is `l = r` with `l` over `left` and `r` over `right`
/// (either orientation), both sides referencing at least one column.
pub(crate) fn as_equi_key(
    e: &Expr,
    left: &HashSet<ColId>,
    right: &HashSet<ColId>,
) -> Option<(Expr, Expr)> {
    let Expr::Binary {
        left: a,
        op: BinaryOp::Eq,
        right: b,
    } = e
    else {
        return None;
    };
    let (ac, bc) = (a.column_ids(), b.column_ids());
    if ac.is_empty() || bc.is_empty() {
        return None;
    }
    if ac.is_subset(left) && bc.is_subset(right) {
        Some((*a.clone(), *b.clone()))
    } else if ac.is_subset(right) && bc.is_subset(left) {
        Some((*b.clone(), *a.clone()))
    } else {
        None
    }
}

fn push(plan: LogicalPlan, mut preds: Vec<Expr>) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Filter { input, predicate } => {
            preds.extend(split_conjunction(predicate));
            push(*input, preds)
        }
        LogicalPlan::Project { input, exprs } => {
            let pushed: Vec<Expr> = preds
                .into_iter()
                .map(|p| {
                    p.substitute(&|c| {
                        exprs
                            .iter()
                            .find(|(_, def)| def.id == c.id)
                            .map(|(e, _)| e.clone())
                    })
                })
                .collect();
            Ok(LogicalPlan::Project {
                input: Box::new(push(*input, pushed)?),
                exprs,
            })
        }
        LogicalPlan::Sort {
            input,
            keys,
            fetch: None,
        } => Ok(LogicalPlan::Sort {
            input: Box::new(push(*input, preds)?),
            keys,
            fetch: None,
        }),
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggs,
        } => {
            let group_ids: HashSet<ColId> = group_by.iter().map(|(_, c)| c.id).collect();
            let (down, keep): (Vec<Expr>, Vec<Expr>) = preds.into_iter().partition(|p| {
                !group_by.is_empty() && !p.column_ids().is_empty() && subset(p, &group_ids)
            });
            let down = down
                .into_iter()
                .map(|p| {
                    p.substitute(&|c| {
                        group_by
                            .iter()
                            .find(|(_, def)| def.id == c.id)
                            .map(|(e, _)| e.clone())
                    })
                })
                .collect();
            let node = LogicalPlan::Aggregate {
                input: Box::new(push(*input, down)?),
                group_by,
                aggs,
            };
            Ok(with_filter(node, keep))
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => push_join(*left, *right, join_type, on, filter, preds),
        LogicalPlan::Scan(mut s) => {
            s.filters.extend(preds);
            Ok(LogicalPlan::Scan(s))
        }
        // Sort with fetch (top-k), Limit and Empty are barriers.
        other => {
            let node = other.map_children(&mut |c| push(c, Vec::new()))?;
            Ok(with_filter(node, preds))
        }
    }
}

fn push_join(
    left: LogicalPlan,
    right: LogicalPlan,
    mut join_type: JoinType,
    mut on: Vec<(Expr, Expr)>,
    filter: Option<Expr>,
    preds: Vec<Expr>,
) -> Result<LogicalPlan> {
    let lids = left.output_ids();
    let rids = right.output_ids();
    if join_type == JoinType::Left && preds.iter().any(|p| null_rejecting(p, &rids)) {
        join_type = JoinType::Inner;
    }
    let mut left_preds = Vec::new();
    let mut right_preds = Vec::new();
    let mut residual = Vec::new();
    let mut above = Vec::new();
    let own: Vec<Expr> = filter.map(split_conjunction).unwrap_or_default();
    match join_type {
        JoinType::Inner => {
            for p in preds.into_iter().chain(own) {
                let ids = p.column_ids();
                if ids.is_subset(&lids) {
                    left_preds.push(p);
                } else if ids.is_subset(&rids) {
                    right_preds.push(p);
                } else if let Some(k) = as_equi_key(&p, &lids, &rids) {
                    on.push(k);
                } else {
                    residual.push(p);
                }
            }
        }
        JoinType::Left => {
            for p in preds {
                let ids = p.column_ids();
                if !ids.is_empty() && ids.is_subset(&lids) {
                    left_preds.push(p);
                } else {
                    above.push(p);
                }
            }
            for p in own {
                let ids = p.column_ids();
                if !ids.is_empty() && ids.is_subset(&rids) {
                    right_preds.push(p);
                } else if let Some(k) = as_equi_key(&p, &lids, &rids) {
                    on.push(k);
                } else {
                    residual.push(p);
                }
            }
        }
    }
    let node = LogicalPlan::Join {
        left: Box::new(push(left, left_preds)?),
        right: Box::new(push(right, right_preds)?),
        join_type,
        on,
        filter: conjunction(residual),
    };
    Ok(with_filter(node, above))
}

/// Is `p` guaranteed to be NULL or FALSE when every column in `side` is NULL?
/// Such a WHERE predicate discards all NULL-extended rows of an outer join.
pub fn null_rejecting(p: &Expr, side: &HashSet<ColId>) -> bool {
    match p {
        Expr::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => null_rejecting(left, side) || null_rejecting(right, side),
        Expr::Binary {
            left,
            op: BinaryOp::Or,
            right,
        } => null_rejecting(left, side) && null_rejecting(right, side),
        Expr::Not(inner) => strict_over(inner, side),
        _ => strict_over(p, side),
    }
}

/// True when `e` is NULL whenever any referenced `side` column is NULL and it
/// references at least one such column.
fn strict_over(e: &Expr, side: &HashSet<ColId>) -> bool {
    match e {
        Expr::Column(c) => side.contains(&c.id),
        Expr::Literal(_) => false,
        Expr::Binary { left, op, right } => {
            if matches!(op, BinaryOp::And | BinaryOp::Or) {
                return false;
            }
            strict_over(left, side) || strict_over(right, side)
        }
        Expr::Negative(x) | Expr::Cast { expr: x, .. } => strict_over(x, side),
        Expr::Like { expr, pattern, .. } => strict_over(expr, side) || strict_over(pattern, side),
        Expr::InList { expr, .. } => strict_over(expr, side),
        Expr::Function { func, args, .. } => {
            *func != ScalarFunc::Coalesce && args.iter().any(|a| strict_over(a, side))
        }
        Expr::Not(_) | Expr::IsNull(_) | Expr::IsNotNull(_) | Expr::Case { .. } => false,
    }
}
