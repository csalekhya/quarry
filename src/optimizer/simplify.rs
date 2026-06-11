//! Expression simplification: constant folding, boolean identities and
//! factoring common conjuncts out of disjunctions.
//!
//! Constant sub-expressions are folded by running them through the *same*
//! vectorized evaluator used at execution time on a one-row batch, so folding
//! can never disagree with runtime semantics. An expression whose evaluation
//! fails (say, an overflow) is left in place and fails at runtime only if a
//! row actually reaches it.

use super::OptimizerRule;
use crate::error::Result;
use crate::physical::expr::evaluate_constant;
use crate::plan::expr::{
    conjunction, disjunction, split_conjunction, split_disjunction, BinaryOp, Expr,
};
use crate::plan::logical::LogicalPlan;
use crate::types::ScalarValue;

/// Rule: simplify every expression and remove trivially true/false filters.
pub struct SimplifyExpressions;

impl OptimizerRule for SimplifyExpressions {
    fn name(&self) -> &'static str {
        "simplify_expressions"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        plan.transform_up(&mut |node| {
            let node = node.map_exprs(&mut |e| Ok(simplify(e)))?;
            Ok(prune_trivial_filters(node))
        })
    }
}

fn is_true(e: &Expr) -> bool {
    matches!(e, Expr::Literal(ScalarValue::Boolean(Some(true))))
}

/// FALSE or NULL: a filter on it keeps nothing.
fn is_false_or_null(e: &Expr) -> bool {
    matches!(e, Expr::Literal(v) if v.is_null() || *v == ScalarValue::Boolean(Some(false)))
}

fn prune_trivial_filters(node: LogicalPlan) -> LogicalPlan {
    match node {
        LogicalPlan::Filter { input, predicate } if is_true(&predicate) => *input,
        LogicalPlan::Filter { input, predicate } if is_false_or_null(&predicate) => {
            LogicalPlan::Empty {
                columns: input.schema(),
                one_row: false,
            }
        }
        LogicalPlan::Scan(mut s) => {
            if s.filters.iter().any(is_false_or_null) {
                return LogicalPlan::Empty {
                    columns: s.columns,
                    one_row: false,
                };
            }
            s.filters.retain(|f| !is_true(f));
            LogicalPlan::Scan(s)
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter: filter.filter(|f| !is_true(f)),
        },
        other => other,
    }
}

/// Simplify an expression bottom-up.
pub fn simplify(e: Expr) -> Expr {
    e.transform_up(&mut |x| Ok(simplify_node(x)))
        .unwrap_or_else(|_| unreachable!("simplify_node is infallible"))
}

fn simplify_node(e: Expr) -> Expr {
    if !matches!(e, Expr::Literal(_)) && e.is_constant() {
        if let Ok(v) = evaluate_constant(&e) {
            return Expr::Literal(v);
        }
        return e;
    }
    match e {
        Expr::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            if is_true(&left) {
                *right
            } else if is_true(&right) {
                *left
            } else if matches!(*left, Expr::Literal(ScalarValue::Boolean(Some(false))))
                || matches!(*right, Expr::Literal(ScalarValue::Boolean(Some(false))))
            {
                Expr::Literal(ScalarValue::Boolean(Some(false)))
            } else {
                Expr::Binary {
                    left,
                    op: BinaryOp::And,
                    right,
                }
            }
        }
        Expr::Binary {
            left,
            op: BinaryOp::Or,
            right,
        } => {
            if is_true(&left) || is_true(&right) {
                Expr::Literal(ScalarValue::Boolean(Some(true)))
            } else if matches!(*left, Expr::Literal(ScalarValue::Boolean(Some(false)))) {
                *right
            } else if matches!(*right, Expr::Literal(ScalarValue::Boolean(Some(false)))) {
                *left
            } else {
                factor_or(Expr::Binary {
                    left,
                    op: BinaryOp::Or,
                    right,
                })
            }
        }
        Expr::Not(inner) => match *inner {
            Expr::Not(x) => *x,
            other => Expr::Not(Box::new(other)),
        },
        Expr::Cast { expr, to } if expr.data_type() == to => *expr,
        other => other,
    }
}

/// `(a AND b) OR (a AND c)` → `a AND (b OR c)`, and `a OR (a AND b)` → `a`.
///
/// Both identities hold in SQL's three-valued logic. This matters for
/// queries like TPC-H Q19, whose join condition is repeated inside every
/// branch of an OR: after factoring it becomes an ordinary equi-join key.
fn factor_or(e: Expr) -> Expr {
    let branches: Vec<Vec<Expr>> = split_disjunction(e.clone())
        .into_iter()
        .map(split_conjunction)
        .collect();
    let common: Vec<Expr> = branches[0]
        .iter()
        .filter(|c| branches[1..].iter().all(|b| b.contains(c)))
        .fold(Vec::new(), |mut acc, c| {
            if !acc.contains(c) {
                acc.push(c.clone());
            }
            acc
        });
    if common.is_empty() {
        return e;
    }
    let rests: Vec<Vec<Expr>> = branches
        .into_iter()
        .map(|b| b.into_iter().filter(|c| !common.contains(c)).collect())
        .collect();
    let mut parts = common;
    if rests.iter().all(|r| !r.is_empty()) {
        let ors: Vec<Expr> = rests.into_iter().filter_map(conjunction).collect();
        if let Some(d) = disjunction(ors) {
            parts.push(d);
        }
    }
    conjunction(parts).unwrap_or(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::expr::ColumnRef;
    use crate::types::DataType;

    fn col(id: usize, name: &str) -> Expr {
        Expr::Column(ColumnRef {
            id,
            name: name.into(),
            data_type: DataType::Int64,
            nullable: true,
        })
    }

    fn int(v: i64) -> Expr {
        Expr::Literal(ScalarValue::Int64(Some(v)))
    }

    fn eq(a: Expr, b: Expr) -> Expr {
        Expr::binary(a, BinaryOp::Eq, b)
    }

    #[test]
    fn folds_constants_and_identities() {
        let e = Expr::binary(
            col(1, "a"),
            BinaryOp::Gt,
            Expr::binary(int(2), BinaryOp::Multiply, int(5)),
        );
        assert_eq!(simplify(e).to_string(), "a > 10");
        let t = Expr::Literal(ScalarValue::Boolean(Some(true)));
        let f = Expr::Literal(ScalarValue::Boolean(Some(false)));
        assert_eq!(
            simplify(eq(col(1, "a"), int(1)).and(t.clone())).to_string(),
            "a = 1"
        );
        assert_eq!(
            simplify(eq(col(1, "a"), int(1)).and(f.clone())).to_string(),
            "FALSE"
        );
        assert_eq!(
            simplify(Expr::binary(eq(col(1, "a"), int(1)), BinaryOp::Or, t)).to_string(),
            "TRUE"
        );
        assert_eq!(
            simplify(Expr::Not(Box::new(Expr::Not(Box::new(eq(
                col(1, "a"),
                int(1)
            ))))))
            .to_string(),
            "a = 1"
        );
        // Overflow is not folded; it stays for runtime.
        let o = Expr::binary(int(i64::MAX), BinaryOp::Plus, int(1));
        assert_eq!(simplify(o.clone()), o);
    }

    #[test]
    fn factors_common_conjuncts_out_of_or() {
        let k = eq(col(1, "a"), col(2, "b"));
        let e = Expr::binary(
            k.clone().and(eq(col(3, "c"), int(1))),
            BinaryOp::Or,
            k.clone().and(eq(col(3, "c"), int(2))),
        );
        assert_eq!(simplify(e).to_string(), "a = b AND (c = 1 OR c = 2)");
        let absorb = Expr::binary(
            k.clone(),
            BinaryOp::Or,
            k.clone().and(eq(col(3, "c"), int(2))),
        );
        assert_eq!(simplify(absorb).to_string(), "a = b");
    }
}
