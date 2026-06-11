//! Cardinality estimation from catalog statistics.
//!
//! Classic System-R style estimates under the usual independence and
//! containment assumptions:
//!
//! * `col = literal` → `1 / ndv(col)`; ranges interpolate between min and max;
//! * `a = b` (join or filter) → `1 / max(ndv(a), ndv(b))`;
//! * `AND` multiplies, `OR` is `s1 + s2 - s1*s2`, `NOT` is `1 - s`;
//! * `LIKE` 0.1, anything unrecognised 0.25.
//!
//! Distinct counts are capped by the estimated row count as they flow up.

use std::collections::HashMap;

use crate::plan::expr::{BinaryOp, ColId, Expr};
use crate::plan::logical::{JoinType, LogicalPlan};

const DEFAULT_SELECTIVITY: f64 = 0.25;
const LIKE_SELECTIVITY: f64 = 0.1;
const RANGE_SELECTIVITY: f64 = 1.0 / 3.0;

/// Statistics for one column of an intermediate result.
#[derive(Debug, Clone, PartialEq)]
pub struct ColStat {
    pub ndv: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub null_frac: f64,
}

/// Estimated output of a plan node.
#[derive(Debug, Clone, Default)]
pub struct Estimate {
    pub rows: f64,
    pub cols: HashMap<ColId, ColStat>,
}

impl Estimate {
    fn cap_ndv(&mut self) {
        let rows = self.rows.max(1.0);
        for s in self.cols.values_mut() {
            s.ndv = s.ndv.min(rows).max(1.0);
        }
    }

    /// Distinct-value estimate for an expression (columns looked up through casts).
    pub fn ndv(&self, e: &Expr) -> f64 {
        match column_of(e).and_then(|id| self.cols.get(&id)) {
            Some(s) => s.ndv.max(1.0),
            None => self.rows.max(1.0),
        }
    }
}

/// The column an expression is a (possibly cast) reference to.
fn column_of(e: &Expr) -> Option<ColId> {
    match e {
        Expr::Column(c) => Some(c.id),
        Expr::Cast { expr, .. } => column_of(expr),
        _ => None,
    }
}

/// Estimate the output of `plan`.
pub fn estimate(plan: &LogicalPlan) -> Estimate {
    let mut est = match plan {
        LogicalPlan::Scan(s) => {
            let stats = &s.table.stats;
            let rows = stats.row_count as f64;
            let mut cols = HashMap::new();
            for (def, &idx) in s.columns.iter().zip(&s.source_indices) {
                if let Some(cs) = stats.columns.get(idx) {
                    cols.insert(
                        def.id,
                        ColStat {
                            ndv: (cs.ndv as f64).max(1.0),
                            min: cs.min.as_ref().and_then(|v| v.as_f64()),
                            max: cs.max.as_ref().and_then(|v| v.as_f64()),
                            null_frac: if rows > 0.0 {
                                cs.null_count as f64 / rows
                            } else {
                                0.0
                            },
                        },
                    );
                }
            }
            let mut est = Estimate { rows, cols };
            let sel: f64 = s.filters.iter().map(|f| selectivity(f, &est)).product();
            est.rows *= sel;
            est
        }
        LogicalPlan::Filter { input, predicate } => {
            let mut est = estimate(input);
            est.rows *= selectivity(predicate, &est);
            est
        }
        LogicalPlan::Project { input, exprs } => {
            let inner = estimate(input);
            let mut cols = HashMap::new();
            for (e, def) in exprs {
                let stat = match column_of(e).and_then(|id| inner.cols.get(&id)) {
                    Some(s) => s.clone(),
                    None => ColStat {
                        ndv: inner.rows.max(1.0),
                        min: None,
                        max: None,
                        null_frac: 0.0,
                    },
                };
                cols.insert(def.id, stat);
            }
            Estimate {
                rows: inner.rows,
                cols,
            }
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let l = estimate(left);
            let r = estimate(right);
            let mut cols = l.cols.clone();
            cols.extend(r.cols.clone());
            let mut est = Estimate {
                rows: l.rows * r.rows,
                cols,
            };
            for (a, b) in on {
                est.rows /= l.ndv(a).max(r.ndv(b));
            }
            if let Some(f) = filter {
                est.rows *= selectivity(f, &est);
            }
            if *join_type == JoinType::Left {
                // Containment: a fraction ndv(r)/ndv(l) of left keys find a match;
                // the rest appear once, NULL-extended.
                let matched = on
                    .first()
                    .map(|(a, b)| (r.ndv(b) / l.ndv(a)).min(1.0))
                    .unwrap_or(1.0);
                let unmatched = l.rows * (1.0 - matched);
                est.rows = (est.rows + unmatched).max(l.rows);
                let null_frac = if est.rows > 0.0 {
                    unmatched / est.rows
                } else {
                    0.0
                };
                for id in r.cols.keys() {
                    if let Some(s) = est.cols.get_mut(id) {
                        s.null_frac = s.null_frac.max(null_frac);
                    }
                }
            }
            est
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggs,
        } => {
            let inner = estimate(input);
            let rows = if group_by.is_empty() {
                1.0
            } else {
                group_by
                    .iter()
                    .map(|(e, _)| inner.ndv(e))
                    .product::<f64>()
                    .min(inner.rows.max(1.0))
            };
            let mut cols = HashMap::new();
            for (e, def) in group_by {
                let mut stat = column_of(e)
                    .and_then(|id| inner.cols.get(&id).cloned())
                    .unwrap_or(ColStat {
                        ndv: rows,
                        min: None,
                        max: None,
                        null_frac: 0.0,
                    });
                stat.ndv = stat.ndv.min(rows);
                cols.insert(def.id, stat);
            }
            for a in aggs {
                cols.insert(
                    a.output.id,
                    ColStat {
                        ndv: rows,
                        min: None,
                        max: None,
                        null_frac: 0.0,
                    },
                );
            }
            Estimate { rows, cols }
        }
        LogicalPlan::Sort { input, fetch, .. } => {
            let mut est = estimate(input);
            if let Some(f) = fetch {
                est.rows = est.rows.min(*f as f64);
            }
            est
        }
        LogicalPlan::Limit { input, skip, fetch } => {
            let mut est = estimate(input);
            est.rows = (est.rows - *skip as f64).max(0.0);
            if let Some(f) = fetch {
                est.rows = est.rows.min(*f as f64);
            }
            est
        }
        LogicalPlan::Empty { one_row, .. } => Estimate {
            rows: if *one_row { 1.0 } else { 0.0 },
            cols: HashMap::new(),
        },
    };
    est.cap_ndv();
    est
}

/// Estimated fraction of rows for which `pred` is true.
pub fn selectivity(pred: &Expr, est: &Estimate) -> f64 {
    let s = match pred {
        Expr::Literal(v) => match v {
            crate::types::ScalarValue::Boolean(Some(true)) => 1.0,
            _ => 0.0,
        },
        Expr::Binary { left, op, right } => match op {
            BinaryOp::And => selectivity(left, est) * selectivity(right, est),
            BinaryOp::Or => {
                let (a, b) = (selectivity(left, est), selectivity(right, est));
                a + b - a * b
            }
            o if o.is_comparison() => comparison_selectivity(left, *o, right, est),
            _ => DEFAULT_SELECTIVITY,
        },
        Expr::Not(e) => 1.0 - selectivity(e, est),
        Expr::IsNull(e) => column_of(e)
            .and_then(|id| est.cols.get(&id))
            .map(|s| s.null_frac)
            .unwrap_or(0.1),
        Expr::IsNotNull(e) => column_of(e)
            .and_then(|id| est.cols.get(&id))
            .map(|s| 1.0 - s.null_frac)
            .unwrap_or(0.9),
        Expr::Like { negated, .. } => {
            if *negated {
                1.0 - LIKE_SELECTIVITY
            } else {
                LIKE_SELECTIVITY
            }
        }
        Expr::InList {
            expr,
            list,
            negated,
        } => {
            let s = (list.len() as f64 / est.ndv(expr)).min(1.0);
            if *negated {
                1.0 - s
            } else {
                s
            }
        }
        _ => DEFAULT_SELECTIVITY,
    };
    s.clamp(0.0, 1.0)
}

fn comparison_selectivity(left: &Expr, op: BinaryOp, right: &Expr, est: &Estimate) -> f64 {
    let (col, lit, op) = match (column_of(left), column_of(right)) {
        (Some(_), Some(_)) => {
            return match op {
                BinaryOp::Eq => 1.0 / est.ndv(left).max(est.ndv(right)),
                BinaryOp::NotEq => 1.0 - 1.0 / est.ndv(left).max(est.ndv(right)),
                _ => RANGE_SELECTIVITY,
            }
        }
        (Some(c), None) => (c, right, op),
        (None, Some(c)) => (c, left, op.commute().unwrap_or(op)),
        (None, None) => return DEFAULT_SELECTIVITY,
    };
    let stat = est.cols.get(&col);
    let ndv = stat.map(|s| s.ndv).unwrap_or(est.rows).max(1.0);
    match op {
        BinaryOp::Eq => 1.0 / ndv,
        BinaryOp::NotEq => 1.0 - 1.0 / ndv,
        _ => {
            let value = match lit {
                Expr::Literal(v) => v.as_f64(),
                _ => None,
            };
            match (value, stat.and_then(|s| s.min), stat.and_then(|s| s.max)) {
                (Some(v), Some(lo), Some(hi)) if hi > lo => {
                    let below = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
                    match op {
                        BinaryOp::Lt | BinaryOp::LtEq => below,
                        _ => 1.0 - below,
                    }
                }
                _ => RANGE_SELECTIVITY,
            }
        }
    }
}
