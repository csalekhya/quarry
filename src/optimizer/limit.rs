//! Limit pushdown: `LIMIT k OFFSET m` over a sort becomes a top-k sort that
//! only keeps `k + m` rows (a partial sort instead of a full one).

use super::OptimizerRule;
use crate::error::Result;
use crate::plan::logical::LogicalPlan;

pub struct LimitPushdown;

impl OptimizerRule for LimitPushdown {
    fn name(&self) -> &'static str {
        "limit_pushdown"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        plan.transform_up(&mut |node| match node {
            LogicalPlan::Limit {
                input,
                skip,
                fetch: Some(f),
            } => Ok(LogicalPlan::Limit {
                input: Box::new(push_fetch(*input, skip.saturating_add(f))),
                skip,
                fetch: Some(f),
            }),
            other => Ok(other),
        })
    }
}

/// Give the nearest sort below (through projections) a fetch bound.
fn push_fetch(plan: LogicalPlan, n: usize) -> LogicalPlan {
    match plan {
        LogicalPlan::Sort { input, keys, fetch } => LogicalPlan::Sort {
            input,
            keys,
            fetch: Some(fetch.map_or(n, |f| f.min(n))),
        },
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: Box::new(push_fetch(*input, n)),
            exprs,
        },
        other => other,
    }
}
