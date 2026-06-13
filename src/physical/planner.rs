//! Lowers an optimized [`LogicalPlan`] into a tree of [`ExecNode`]s.

use arrow::compute::SortOptions;

use super::aggregate::{create_accumulator, HashAggregateExec, PhysicalAggregate};
use super::basic::{EmptyExec, FilterExec, LimitExec, PhysicalSortKey, ProjectExec, SortExec};
use super::expr::{column_index, create_physical_expr};
use super::join::{HashJoinExec, NestedLoopJoinExec};
use super::scan::{PruningPredicate, ScanExec};
use super::{arrow_schema, ExecConfig, ExecNode};
use crate::error::Result;
use crate::plan::expr::{conjunction, Expr};
use crate::plan::logical::{fmt_sort_key, JoinType, LogicalPlan, ScanNode};

/// Build the executable operator tree for `plan`.
pub fn create_physical_plan(plan: &LogicalPlan, config: &ExecConfig) -> Result<ExecNode> {
    let schema = arrow_schema(&plan.schema());
    let node = match plan {
        LogicalPlan::Scan(scan) => {
            let filter = match conjunction(scan.filters.clone()) {
                Some(f) => Some(create_physical_expr(&f, &scan.columns)?),
                None => None,
            };
            let text = scan.filters.iter().map(|f| f.to_string()).collect();
            ExecNode::new(
                Box::new(ScanExec::new(
                    scan.table.clone(),
                    scan.source_indices.clone(),
                    config.batch_size,
                    filter,
                    text,
                    pruning_predicates(scan),
                )),
                schema,
            )
        }
        LogicalPlan::Filter { input, predicate } => {
            let child = create_physical_plan(input, config)?;
            let pred = create_physical_expr(predicate, &input.schema())?;
            ExecNode::new(
                Box::new(FilterExec::new(child, pred, predicate.to_string())),
                schema,
            )
        }
        LogicalPlan::Project { input, exprs } => {
            let child = create_physical_plan(input, config)?;
            let in_schema = input.schema();
            let phys = exprs
                .iter()
                .map(|(e, _)| create_physical_expr(e, &in_schema))
                .collect::<Result<Vec<_>>>()?;
            let text = exprs
                .iter()
                .map(|(e, c)| {
                    let s = e.to_string();
                    if s == c.name {
                        s
                    } else {
                        format!("{s} AS {}", c.name)
                    }
                })
                .collect();
            ExecNode::new(
                Box::new(ProjectExec::new(child, phys, schema.clone(), text)),
                schema,
            )
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let l = create_physical_plan(left, config)?;
            let r = create_physical_plan(right, config)?;
            let joined = plan.schema();
            let phys_filter = match filter {
                Some(f) => Some(create_physical_expr(f, &joined)?),
                None => None,
            };
            let kind = match join_type {
                JoinType::Inner => "inner",
                JoinType::Left => "left",
            };
            let mut text = kind.to_string();
            if !on.is_empty() {
                let keys: Vec<String> = on.iter().map(|(a, b)| format!("{a} = {b}")).collect();
                text.push_str(&format!(" on=[{}]", keys.join(", ")));
            }
            if let Some(f) = filter {
                text.push_str(&format!(" filter={f}"));
            }
            if on.is_empty() {
                ExecNode::new(
                    Box::new(NestedLoopJoinExec::new(
                        l,
                        r,
                        *join_type,
                        phys_filter,
                        schema.clone(),
                        config.batch_size,
                        text,
                    )),
                    schema,
                )
            } else {
                let (ls, rs) = (left.schema(), right.schema());
                let left_keys = on
                    .iter()
                    .map(|(a, _)| create_physical_expr(a, &ls))
                    .collect::<Result<Vec<_>>>()?;
                let right_keys = on
                    .iter()
                    .map(|(_, b)| create_physical_expr(b, &rs))
                    .collect::<Result<Vec<_>>>()?;
                let key_types = on.iter().map(|(a, _)| a.data_type().to_arrow()).collect();
                ExecNode::new(
                    Box::new(HashJoinExec::try_new(
                        l,
                        r,
                        *join_type,
                        left_keys,
                        right_keys,
                        key_types,
                        phys_filter,
                        schema.clone(),
                        text,
                    )?),
                    schema,
                )
            }
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggs,
        } => {
            let child = create_physical_plan(input, config)?;
            let in_schema = input.schema();
            let group_exprs = group_by
                .iter()
                .map(|(e, _)| create_physical_expr(e, &in_schema))
                .collect::<Result<Vec<_>>>()?;
            let group_types = group_by
                .iter()
                .map(|(e, _)| e.data_type().to_arrow())
                .collect();
            let mut phys_aggs = Vec::new();
            for a in aggs {
                let arg = match &a.arg {
                    Some(e) => Some(create_physical_expr(e, &in_schema)?),
                    None => None,
                };
                let input_type = a.arg.as_ref().map(|e| e.data_type().to_arrow());
                phys_aggs.push(PhysicalAggregate {
                    func: a.func,
                    arg,
                    acc: create_accumulator(a.func, input_type.as_ref(), a.distinct)?,
                });
            }
            let g: Vec<String> = group_by.iter().map(|(e, _)| e.to_string()).collect();
            let ag: Vec<String> = aggs.iter().map(|a| a.to_string()).collect();
            let text = format!("group=[{}] aggs=[{}]", g.join(", "), ag.join(", "));
            ExecNode::new(
                Box::new(HashAggregateExec::new(
                    child,
                    group_exprs,
                    group_types,
                    phys_aggs,
                    schema.clone(),
                    config.batch_size,
                    text,
                )),
                schema,
            )
        }
        LogicalPlan::Sort { input, keys, fetch } => {
            let child = create_physical_plan(input, config)?;
            let in_schema = input.schema();
            let phys = keys
                .iter()
                .map(|k| {
                    Ok(PhysicalSortKey {
                        expr: create_physical_expr(&k.expr, &in_schema)?,
                        options: SortOptions {
                            descending: !k.asc,
                            nulls_first: k.nulls_first,
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let text = keys.iter().map(fmt_sort_key).collect::<Vec<_>>().join(", ");
            ExecNode::new(
                Box::new(SortExec::new(child, phys, *fetch, config.batch_size, text)),
                schema,
            )
        }
        LogicalPlan::Limit { input, skip, fetch } => {
            let child = create_physical_plan(input, config)?;
            ExecNode::new(Box::new(LimitExec::new(child, *skip, *fetch)), schema)
        }
        LogicalPlan::Empty { one_row, .. } => {
            ExecNode::new(Box::new(EmptyExec::new(schema.clone(), *one_row)), schema)
        }
    };
    Ok(node)
}

/// Extract `column <op> literal` filters usable for Parquet row-group pruning.
fn pruning_predicates(scan: &ScanNode) -> Vec<PruningPredicate> {
    let mut out = Vec::new();
    for f in &scan.filters {
        let Expr::Binary { left, op, right } = f else {
            continue;
        };
        if !op.is_comparison() {
            continue;
        }
        let (col, op, lit) = match (left.as_ref(), right.as_ref()) {
            (Expr::Column(c), Expr::Literal(v)) => (c, *op, v),
            (Expr::Literal(v), Expr::Column(c)) => match op.commute() {
                Some(o) => (c, o, v),
                None => continue,
            },
            _ => continue,
        };
        if lit.is_null() || lit.data_type() != col.data_type {
            continue;
        }
        if let Some(pos) = column_index(col.id, &scan.columns) {
            out.push(PruningPredicate {
                column: scan.source_indices[pos],
                op,
                value: lit.clone(),
            });
        }
    }
    out
}
