//! Rule-based logical optimizer.
//!
//! Rules run once each, in a fixed order chosen so every rule sees the
//! output it depends on:
//!
//! 1. [`simplify::SimplifyExpressions`] — constant folding, boolean
//!    identities, factoring common conjuncts out of ORs;
//! 2. [`pushdown::PredicatePushdown`] — filters to scans, WHERE equalities
//!    to join keys, LEFT→INNER join conversion;
//! 3. [`join_order::JoinReorder`] — DP/greedy join ordering on estimated
//!    cardinalities (needs pushed filters for good estimates);
//! 4. [`projection::ProjectionPushdown`] — column pruning;
//! 5. [`limit::LimitPushdown`] — top-k sorts.

pub mod cardinality;
pub mod join_order;
pub mod limit;
pub mod projection;
pub mod pushdown;
pub mod simplify;

use crate::error::{QuarryError, Result};
use crate::plan::logical::LogicalPlan;

/// A logical plan rewrite.
pub trait OptimizerRule {
    /// Stable identifier, used to disable a rule and in EXPLAIN VERBOSE.
    fn name(&self) -> &'static str;
    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan>;
}

/// Names of all rules in execution order.
pub const RULE_NAMES: [&str; 5] = [
    "simplify_expressions",
    "predicate_pushdown",
    "join_reorder",
    "projection_pushdown",
    "limit_pushdown",
];

/// An ordered list of rules.
pub struct Optimizer {
    rules: Vec<Box<dyn OptimizerRule>>,
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer {
            rules: vec![
                Box::new(simplify::SimplifyExpressions),
                Box::new(pushdown::PredicatePushdown),
                Box::new(join_order::JoinReorder),
                Box::new(projection::ProjectionPushdown),
                Box::new(limit::LimitPushdown),
            ],
        }
    }
}

impl Optimizer {
    /// The default rule set minus the named rules.
    pub fn without(disabled: &[String]) -> Result<Self> {
        for d in disabled {
            if !RULE_NAMES.contains(&d.as_str()) {
                return Err(QuarryError::Plan(format!(
                    "unknown optimizer rule '{d}' (known: {})",
                    RULE_NAMES.join(", ")
                )));
            }
        }
        let mut o = Optimizer::default();
        o.rules.retain(|r| !disabled.iter().any(|d| d == r.name()));
        Ok(o)
    }

    pub fn rule_names(&self) -> Vec<&'static str> {
        self.rules.iter().map(|r| r.name()).collect()
    }

    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        self.rules.iter().try_fold(plan, |p, r| r.optimize(p))
    }

    /// Optimize and record the plan after every rule (for EXPLAIN VERBOSE).
    pub fn optimize_traced(
        &self,
        plan: LogicalPlan,
    ) -> Result<(LogicalPlan, Vec<(&'static str, LogicalPlan)>)> {
        let mut trace = Vec::new();
        let mut p = plan;
        for r in &self.rules {
            p = r.optimize(p)?;
            trace.push((r.name(), p.clone()));
        }
        Ok((p, trace))
    }
}
