//! Cost-based join reordering for clusters of inner joins.
//!
//! A maximal tree of inner joins is flattened into its leaves (the inputs
//! that are not themselves inner joins) and a set of predicates. Then:
//!
//! * for up to [`DP_LIMIT`] leaves, dynamic programming over subsets
//!   (DPsub) finds the bushy join tree minimizing the sum of intermediate
//!   result sizes (the C_out cost model), considering only splits connected
//!   by a predicate; cross products are allowed only if the join graph is
//!   disconnected;
//! * above that, a greedy pass repeatedly joins the pair with the smallest
//!   estimated result.
//!
//! The cardinality of a subset is the product of its leaves' cardinalities
//! times the selectivity of every predicate it fully contains, so it does not
//! depend on join order. At each join the smaller input becomes the right
//! (build) side of the hash join. DPsub is O(3^n) time and O(2^n) space.

use std::collections::{HashMap, HashSet};

use super::cardinality::{estimate, selectivity, Estimate};
use super::pushdown::as_equi_key;
use super::OptimizerRule;
use crate::error::Result;
use crate::plan::expr::{conjunction, split_conjunction, BinaryOp, ColId, Expr};
use crate::plan::logical::{JoinType, LogicalPlan};

/// Largest cluster solved exactly by dynamic programming.
pub const DP_LIMIT: usize = 10;

/// Largest cluster reordered at all (leaf sets are 64-bit masks).
const MAX_LEAVES: usize = 63;

pub struct JoinReorder;

impl OptimizerRule for JoinReorder {
    fn name(&self) -> &'static str {
        "join_reorder"
    }

    fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        reorder(plan)
    }
}

fn is_inner_join(p: &LogicalPlan) -> bool {
    matches!(
        p,
        LogicalPlan::Join {
            join_type: JoinType::Inner,
            ..
        }
    )
}

/// Number of leaves in the inner-join cluster rooted at `p`.
fn cluster_size(p: &LogicalPlan) -> usize {
    match p {
        LogicalPlan::Join { left, right, .. } if is_inner_join(p) => {
            cluster_size(left) + cluster_size(right)
        }
        _ => 1,
    }
}

fn reorder(plan: LogicalPlan) -> Result<LogicalPlan> {
    // Subsets are bitmasks; clusters wider than that keep their written order.
    if !is_inner_join(&plan) || cluster_size(&plan) > MAX_LEAVES {
        return plan.map_children(&mut reorder);
    }
    let mut leaves = Vec::new();
    let mut preds = Vec::new();
    flatten(plan, &mut leaves, &mut preds);
    let leaves = leaves
        .into_iter()
        .map(reorder)
        .collect::<Result<Vec<_>>>()?;
    Ok(solve(leaves, preds))
}

fn flatten(plan: LogicalPlan, leaves: &mut Vec<LogicalPlan>, preds: &mut Vec<Expr>) {
    match plan {
        LogicalPlan::Join {
            left,
            right,
            join_type: JoinType::Inner,
            on,
            filter,
        } => {
            for (l, r) in on {
                preds.push(Expr::binary(l, BinaryOp::Eq, r));
            }
            if let Some(f) = filter {
                preds.extend(split_conjunction(f));
            }
            flatten(*left, leaves, preds);
            flatten(*right, leaves, preds);
        }
        other => leaves.push(other),
    }
}

type Mask = u64;

#[derive(Clone)]
struct Pred {
    expr: Expr,
    mask: Mask,
    selectivity: f64,
}

#[derive(Clone)]
enum Tree {
    Leaf(usize),
    Join(Box<Tree>, Box<Tree>),
}

#[derive(Clone)]
struct Best {
    tree: Tree,
    cost: f64,
}

struct Problem {
    leaf_rows: Vec<f64>,
    preds: Vec<Pred>,
}

impl Problem {
    fn rows(&self, s: Mask) -> f64 {
        let mut r: f64 = (0..self.leaf_rows.len())
            .filter(|i| s & (1 << i) != 0)
            .map(|i| self.leaf_rows[i].max(1.0))
            .product();
        for p in &self.preds {
            if p.mask & s == p.mask && p.mask.count_ones() >= 2 {
                r *= p.selectivity;
            }
        }
        r.max(1.0)
    }

    fn connected(&self, a: Mask, b: Mask) -> bool {
        self.preds
            .iter()
            .any(|p| p.mask & a != 0 && p.mask & b != 0 && p.mask & !(a | b) == 0)
    }
}

fn solve(leaves: Vec<LogicalPlan>, preds: Vec<Expr>) -> LogicalPlan {
    let n = leaves.len();
    let estimates: Vec<Estimate> = leaves.iter().map(estimate).collect();
    let mut combined = Estimate::default();
    for e in &estimates {
        combined.cols.extend(e.cols.clone());
    }
    combined.rows = estimates.iter().map(|e| e.rows).fold(1.0, f64::max);
    let leaf_ids: Vec<HashSet<ColId>> = leaves.iter().map(|l| l.output_ids()).collect();
    let mut constant = Vec::new();
    let mut problem = Problem {
        leaf_rows: estimates.iter().map(|e| e.rows).collect(),
        preds: Vec::new(),
    };
    for p in preds {
        let ids = p.column_ids();
        let mask = (0..n)
            .filter(|&i| ids.iter().any(|id| leaf_ids[i].contains(id)))
            .fold(0, |m, i| m | (1 << i));
        if mask == 0 {
            constant.push(p);
            continue;
        }
        let selectivity = selectivity(&p, &combined);
        problem.preds.push(Pred {
            expr: p,
            mask,
            selectivity,
        });
    }
    let tree = if n <= DP_LIMIT {
        dp(&problem, n)
    } else {
        greedy(&problem, n)
    };
    let mut slots: Vec<Option<LogicalPlan>> = leaves.into_iter().map(Some).collect();
    let mut remaining: Vec<Option<Pred>> = problem.preds.iter().cloned().map(Some).collect();
    let (plan, _) = build(&tree, &mut slots, &mut remaining, &problem, &leaf_ids);
    // Predicates over a single leaf (pushdown disabled) or constants go on top.
    let leftover: Vec<Expr> = remaining
        .into_iter()
        .flatten()
        .map(|p| p.expr)
        .chain(constant)
        .collect();
    match conjunction(leftover) {
        Some(f) => LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: f,
        },
        None => plan,
    }
}

fn dp(problem: &Problem, n: usize) -> Tree {
    let full: Mask = (1 << n) - 1;
    for allow_cross in [false, true] {
        let mut best: HashMap<Mask, Best> = HashMap::new();
        for i in 0..n {
            best.insert(
                1 << i,
                Best {
                    tree: Tree::Leaf(i),
                    cost: 0.0,
                },
            );
        }
        // Subsets in increasing numeric order: every proper subset comes first.
        for s in 1..=full {
            if s.count_ones() < 2 {
                continue;
            }
            let low = s & s.wrapping_neg();
            let rows = problem.rows(s);
            let mut candidate: Option<Best> = None;
            // Enumerate splits (s1, s - s1) with s1 containing the lowest bit.
            let mut s1 = (s - 1) & s;
            while s1 > 0 {
                if s1 & low != 0 {
                    let s2 = s & !s1;
                    if let (Some(a), Some(b)) = (best.get(&s1), best.get(&s2)) {
                        if allow_cross || problem.connected(s1, s2) {
                            let cost = a.cost + b.cost + rows;
                            if candidate.as_ref().is_none_or(|c| cost < c.cost) {
                                candidate = Some(Best {
                                    tree: Tree::Join(
                                        Box::new(a.tree.clone()),
                                        Box::new(b.tree.clone()),
                                    ),
                                    cost,
                                });
                            }
                        }
                    }
                }
                s1 = (s1 - 1) & s;
            }
            if let Some(c) = candidate {
                best.insert(s, c);
            }
        }
        if let Some(b) = best.remove(&full) {
            return b.tree;
        }
    }
    unreachable!("cross products make every subset reachable")
}

fn greedy(problem: &Problem, n: usize) -> Tree {
    let mut parts: Vec<(Mask, Tree)> = (0..n).map(|i| (1 << i, Tree::Leaf(i))).collect();
    while parts.len() > 1 {
        let mut pick: Option<(usize, usize, bool, f64)> = None;
        for i in 0..parts.len() {
            for j in i + 1..parts.len() {
                let (a, b) = (parts[i].0, parts[j].0);
                let connected = problem.connected(a, b);
                let rows = problem.rows(a | b);
                let better = match pick {
                    None => true,
                    Some((_, _, pc, pr)) => (connected && !pc) || (connected == pc && rows < pr),
                };
                if better {
                    pick = Some((i, j, connected, rows));
                }
            }
        }
        let (i, j, _, _) = pick.unwrap_or_else(|| unreachable!("at least two parts"));
        let (mb, tb) = parts.remove(j);
        let (ma, ta) = parts.remove(i);
        parts.push((ma | mb, Tree::Join(Box::new(ta), Box::new(tb))));
    }
    parts
        .pop()
        .map(|(_, t)| t)
        .unwrap_or_else(|| unreachable!("non-empty"))
}

/// Materialize a join tree, attaching each predicate at the lowest join that
/// covers it. Returns the plan and its leaf mask.
fn build(
    tree: &Tree,
    slots: &mut [Option<LogicalPlan>],
    preds: &mut [Option<Pred>],
    problem: &Problem,
    leaf_ids: &[HashSet<ColId>],
) -> (LogicalPlan, Mask) {
    match tree {
        Tree::Leaf(i) => (
            slots[*i]
                .take()
                .unwrap_or_else(|| unreachable!("each leaf used once")),
            1 << i,
        ),
        Tree::Join(a, b) => {
            let (pa, ma) = build(a, slots, preds, problem, leaf_ids);
            let (pb, mb) = build(b, slots, preds, problem, leaf_ids);
            // Smaller input becomes the build (right) side.
            let ((left, lm), (right, rm)) = if problem.rows(ma) >= problem.rows(mb) {
                ((pa, ma), (pb, mb))
            } else {
                ((pb, mb), (pa, ma))
            };
            let ids_of = |m: Mask| -> HashSet<ColId> {
                (0..leaf_ids.len())
                    .filter(|i| m & (1 << i) != 0)
                    .flat_map(|i| leaf_ids[i].iter().copied())
                    .collect()
            };
            let (lids, rids) = (ids_of(lm), ids_of(rm));
            let mut on = Vec::new();
            let mut filters = Vec::new();
            for slot in preds.iter_mut() {
                let covered = slot.as_ref().is_some_and(|p| {
                    p.mask & (lm | rm) == p.mask && p.mask & lm != 0 && p.mask & rm != 0
                });
                if !covered {
                    continue;
                }
                let p = slot.take().unwrap_or_else(|| unreachable!("checked above"));
                match as_equi_key(&p.expr, &lids, &rids) {
                    Some(k) => on.push(k),
                    None => filters.push(p.expr),
                }
            }
            (
                LogicalPlan::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    join_type: JoinType::Inner,
                    on,
                    filter: conjunction(filters),
                },
                lm | rm,
            )
        }
    }
}
