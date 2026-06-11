//! Projection pushdown (column pruning).
//!
//! Walks the plan top-down carrying the set of column ids the parent needs.
//! Scans read only required columns (plus columns their own filters use),
//! unused projection items and aggregates are dropped, and after a join a
//! narrowing projection removes columns (typically join keys) that nothing
//! above needs, so later joins and aggregates move less data.

use std::collections::HashSet;

use super::OptimizerRule;
use crate::error::Result;
use crate::plan::expr::{ColId, Expr};
use crate::plan::logical::LogicalPlan;

pub struct ProjectionPushdown;

impl OptimizerRule for ProjectionPushdown {
    fn name(&self) -> &'static str {
        "projection_pushdown"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        let required = plan.output_ids();
        prune(plan, &required)?.transform_up(&mut |p| Ok(merge_projections(p)))
    }
}

/// Collapse `Project(columns) -> Project(exprs)` into one projection and drop
/// projections that return their input unchanged.
fn merge_projections(plan: LogicalPlan) -> LogicalPlan {
    let LogicalPlan::Project { input, exprs } = plan else {
        return plan;
    };
    let all_columns = exprs.iter().all(|(e, _)| matches!(e, Expr::Column(_)));
    match *input {
        LogicalPlan::Project {
            input: inner_input,
            exprs: inner,
        } if all_columns => {
            let merged = exprs
                .into_iter()
                .map(|(e, def)| {
                    let Expr::Column(c) = &e else {
                        unreachable!("checked above")
                    };
                    let source = inner
                        .iter()
                        .find(|(_, d)| d.id == c.id)
                        .map(|(ie, _)| ie.clone())
                        .unwrap_or(e);
                    (source, def)
                })
                .collect();
            merge_projections(LogicalPlan::Project {
                input: inner_input,
                exprs: merged,
            })
        }
        other => {
            let schema = other.schema();
            let identity = all_columns
                && exprs.len() == schema.len()
                && exprs.iter().zip(&schema).all(|((e, def), c)| {
                    matches!(e, Expr::Column(r) if r.id == c.id)
                        && def.id == c.id
                        && def.name == c.name
                });
            if identity {
                other
            } else {
                LogicalPlan::Project {
                    input: Box::new(other),
                    exprs,
                }
            }
        }
    }
}

fn refs<'a>(exprs: impl IntoIterator<Item = &'a Expr>, out: &mut HashSet<ColId>) {
    for e in exprs {
        e.collect_columns(out);
    }
}

fn prune(plan: LogicalPlan, required: &HashSet<ColId>) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Project { input, exprs } => {
            let kept: Vec<_> = exprs
                .into_iter()
                .filter(|(_, c)| required.contains(&c.id))
                .collect();
            let mut need = HashSet::new();
            refs(kept.iter().map(|(e, _)| e), &mut need);
            Ok(LogicalPlan::Project {
                input: Box::new(prune(*input, &need)?),
                exprs: kept,
            })
        }
        LogicalPlan::Filter { input, predicate } => {
            let mut need = required.clone();
            predicate.collect_columns(&mut need);
            Ok(LogicalPlan::Filter {
                input: Box::new(prune(*input, &need)?),
                predicate,
            })
        }
        LogicalPlan::Sort { input, keys, fetch } => {
            let mut need = required.clone();
            refs(keys.iter().map(|k| &k.expr), &mut need);
            Ok(LogicalPlan::Sort {
                input: Box::new(prune(*input, &need)?),
                keys,
                fetch,
            })
        }
        LogicalPlan::Limit { input, skip, fetch } => Ok(LogicalPlan::Limit {
            input: Box::new(prune(*input, required)?),
            skip,
            fetch,
        }),
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggs,
        } => {
            let aggs: Vec<_> = aggs
                .into_iter()
                .filter(|a| required.contains(&a.output.id))
                .collect();
            let mut need = HashSet::new();
            refs(group_by.iter().map(|(e, _)| e), &mut need);
            refs(aggs.iter().filter_map(|a| a.arg.as_ref()), &mut need);
            Ok(LogicalPlan::Aggregate {
                input: Box::new(prune(*input, &need)?),
                group_by,
                aggs,
            })
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let mut need = required.clone();
            for (l, r) in &on {
                l.collect_columns(&mut need);
                r.collect_columns(&mut need);
            }
            if let Some(f) = &filter {
                f.collect_columns(&mut need);
            }
            let lneed: HashSet<ColId> = left.output_ids().intersection(&need).copied().collect();
            let rneed: HashSet<ColId> = right.output_ids().intersection(&need).copied().collect();
            let join = LogicalPlan::Join {
                left: Box::new(prune(*left, &lneed)?),
                right: Box::new(prune(*right, &rneed)?),
                join_type,
                on,
                filter,
            };
            let schema = join.schema();
            if schema.iter().all(|c| required.contains(&c.id)) {
                return Ok(join);
            }
            let exprs = schema
                .into_iter()
                .filter(|c| required.contains(&c.id))
                .map(|c| (c.to_expr(), c))
                .collect();
            Ok(LogicalPlan::Project {
                input: Box::new(join),
                exprs,
            })
        }
        LogicalPlan::Scan(mut s) => {
            let mut need = required.clone();
            refs(&s.filters, &mut need);
            let mut keep: Vec<usize> = (0..s.columns.len())
                .filter(|&i| need.contains(&s.columns[i].id))
                .collect();
            if keep.is_empty() && !s.columns.is_empty() {
                // Keep one column so the scan still reports row counts (count(*)).
                keep.push(0);
            }
            s.columns = keep.iter().map(|&i| s.columns[i].clone()).collect();
            s.source_indices = keep.iter().map(|&i| s.source_indices[i]).collect();
            Ok(LogicalPlan::Scan(s))
        }
        empty @ LogicalPlan::Empty { .. } => Ok(empty),
    }
}
