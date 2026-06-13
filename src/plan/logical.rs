//! The logical plan: a tree of relational operators over [`Expr`]s.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::sync::Arc;

use super::expr::{AggFunc, ColId, ColumnRef, Expr};
use crate::catalog::TableEntry;
use crate::error::Result;
use crate::types::DataType;

/// One output column of a plan node.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub id: ColId,
    pub name: String,
    /// Table alias the column is visible under (`None` for computed columns).
    pub qualifier: Option<String>,
    pub data_type: DataType,
    pub nullable: bool,
}

impl ColumnDef {
    pub fn to_ref(&self) -> ColumnRef {
        ColumnRef {
            id: self.id,
            name: self.display_name(),
            data_type: self.data_type,
            nullable: self.nullable,
        }
    }

    pub fn to_expr(&self) -> Expr {
        Expr::Column(self.to_ref())
    }

    /// `qualifier.name` when qualified, else `name`.
    pub fn display_name(&self) -> String {
        match &self.qualifier {
            Some(q) => format!("{q}.{}", self.name),
            None => self.name.clone(),
        }
    }
}

/// An aggregate computed by an [`LogicalPlan::Aggregate`] node.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateExpr {
    pub func: AggFunc,
    /// `None` only for `count(*)`.
    pub arg: Option<Expr>,
    pub distinct: bool,
    pub output: ColumnDef,
}

impl std::fmt::Display for AggregateExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.arg {
            None => write!(f, "count(*)"),
            Some(a) => write!(
                f,
                "{}({}{a})",
                self.func,
                if self.distinct { "DISTINCT " } else { "" }
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SortKey {
    pub expr: Expr,
    pub asc: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    /// Left outer join: every left row appears at least once.
    Left,
}

/// A base-table scan. Filters and the column subset are filled in by the
/// optimizer's pushdown rules.
#[derive(Debug, Clone)]
pub struct ScanNode {
    pub table: Arc<TableEntry>,
    /// Alias the table was referenced by (or its name).
    pub alias: String,
    /// Output columns, parallel to `source_indices`.
    pub columns: Vec<ColumnDef>,
    /// Indices into the table schema of the columns actually read.
    pub source_indices: Vec<usize>,
    /// Conjunctive predicates evaluated inside the scan.
    pub filters: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub enum LogicalPlan {
    Scan(ScanNode),
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    Project {
        input: Box<LogicalPlan>,
        exprs: Vec<(Expr, ColumnDef)>,
    },
    /// Equi-join on `on` pairs (left expr, right expr) plus an optional residual
    /// `filter`. With no `on` pairs it is a nested-loop / cross join.
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<(Expr, ColumnDef)>,
        aggs: Vec<AggregateExpr>,
    },
    Sort {
        input: Box<LogicalPlan>,
        keys: Vec<SortKey>,
        /// Top-k: only the first `fetch` rows are needed.
        fetch: Option<usize>,
    },
    Limit {
        input: Box<LogicalPlan>,
        skip: usize,
        fetch: Option<usize>,
    },
    /// A relation with a fixed schema and either no rows or exactly one
    /// zero-column row (`SELECT 1` without `FROM`).
    Empty {
        columns: Vec<ColumnDef>,
        one_row: bool,
    },
}

impl LogicalPlan {
    /// Output columns of this node.
    pub fn schema(&self) -> Vec<ColumnDef> {
        match self {
            LogicalPlan::Scan(s) => s.columns.clone(),
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => input.schema(),
            LogicalPlan::Project { exprs, .. } => exprs.iter().map(|(_, c)| c.clone()).collect(),
            LogicalPlan::Join {
                left,
                right,
                join_type,
                ..
            } => {
                let mut cols = left.schema();
                let mut r = right.schema();
                if *join_type == JoinType::Left {
                    for c in &mut r {
                        c.nullable = true;
                    }
                }
                cols.extend(r);
                cols
            }
            LogicalPlan::Aggregate { group_by, aggs, .. } => group_by
                .iter()
                .map(|(_, c)| c.clone())
                .chain(aggs.iter().map(|a| a.output.clone()))
                .collect(),
            LogicalPlan::Empty { columns, .. } => columns.clone(),
        }
    }

    pub fn output_ids(&self) -> HashSet<ColId> {
        self.schema().iter().map(|c| c.id).collect()
    }

    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan(_) | LogicalPlan::Empty { .. } => vec![],
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
        }
    }

    /// Short operator name used in EXPLAIN.
    pub fn name(&self) -> &'static str {
        match self {
            LogicalPlan::Scan(_) => "Scan",
            LogicalPlan::Filter { .. } => "Filter",
            LogicalPlan::Project { .. } => "Projection",
            LogicalPlan::Join { on, filter, .. } if on.is_empty() && filter.is_none() => {
                "CrossJoin"
            }
            LogicalPlan::Join { .. } => "Join",
            LogicalPlan::Aggregate { .. } => "Aggregate",
            LogicalPlan::Sort { .. } => "Sort",
            LogicalPlan::Limit { .. } => "Limit",
            LogicalPlan::Empty { .. } => "Empty",
        }
    }

    /// One-line description of this node (without children).
    pub fn describe(&self) -> String {
        let list = |v: Vec<String>| v.join(", ");
        match self {
            LogicalPlan::Scan(s) => {
                let mut out = if s.alias == s.table.name {
                    format!("Scan: {}", s.table.name)
                } else {
                    format!("Scan: {} AS {}", s.table.name, s.alias)
                };
                let cols: Vec<String> = s.columns.iter().map(|c| c.name.clone()).collect();
                let _ = write!(out, " [{}]", list(cols));
                if !s.filters.is_empty() {
                    let f: Vec<String> = s.filters.iter().map(|e| e.to_string()).collect();
                    let _ = write!(out, " filters=[{}]", list(f));
                }
                out
            }
            LogicalPlan::Filter { predicate, .. } => format!("Filter: {predicate}"),
            LogicalPlan::Project { exprs, .. } => {
                let items: Vec<String> = exprs
                    .iter()
                    .map(|(e, c)| {
                        let shown = e.to_string();
                        if shown == c.display_name() || shown == c.name {
                            shown
                        } else {
                            format!("{shown} AS {}", c.name)
                        }
                    })
                    .collect();
                format!("Projection: {}", list(items))
            }
            LogicalPlan::Join {
                join_type,
                on,
                filter,
                ..
            } => {
                if on.is_empty() && filter.is_none() {
                    return "CrossJoin".to_string();
                }
                let kind = match join_type {
                    JoinType::Inner => "Inner",
                    JoinType::Left => "Left",
                };
                let mut out = format!("{kind}Join:");
                if !on.is_empty() {
                    let keys: Vec<String> = on.iter().map(|(l, r)| format!("{l} = {r}")).collect();
                    let _ = write!(out, " on=[{}]", list(keys));
                }
                if let Some(f) = filter {
                    let _ = write!(out, " filter={f}");
                }
                out
            }
            LogicalPlan::Aggregate { group_by, aggs, .. } => {
                let g: Vec<String> = group_by.iter().map(|(e, _)| e.to_string()).collect();
                let a: Vec<String> = aggs
                    .iter()
                    .map(|a| format!("{a} AS {}", a.output.name))
                    .collect();
                format!("Aggregate: group=[{}] aggs=[{}]", list(g), list(a))
            }
            LogicalPlan::Sort { keys, fetch, .. } => {
                let k: Vec<String> = keys.iter().map(fmt_sort_key).collect();
                match fetch {
                    Some(n) => format!("TopK: fetch={n} keys=[{}]", list(k)),
                    None => format!("Sort: {}", list(k)),
                }
            }
            LogicalPlan::Limit { skip, fetch, .. } => match fetch {
                Some(n) => format!("Limit: skip={skip} fetch={n}"),
                None => format!("Limit: skip={skip}"),
            },
            LogicalPlan::Empty { one_row, .. } => {
                format!("Empty: rows={}", if *one_row { 1 } else { 0 })
            }
        }
    }

    /// Indented tree rendering; `annotate` may append text per node.
    pub fn display_tree(&self, annotate: &dyn Fn(&LogicalPlan) -> String) -> String {
        let mut out = String::new();
        self.write_tree(&mut out, 0, annotate);
        out
    }

    fn write_tree(
        &self,
        out: &mut String,
        depth: usize,
        annotate: &dyn Fn(&LogicalPlan) -> String,
    ) {
        let _ = writeln!(
            out,
            "{}{}{}",
            "  ".repeat(depth),
            self.describe(),
            annotate(self)
        );
        for c in self.children() {
            c.write_tree(out, depth + 1, annotate);
        }
    }
}

/// `expr ASC|DESC [NULLS FIRST|LAST]`, omitting NULLS when it matches the default.
pub fn fmt_sort_key(k: &SortKey) -> String {
    let dir = if k.asc { "ASC" } else { "DESC" };
    // Default: NULLs are the smallest value (first for ASC, last for DESC).
    let default_nulls_first = k.asc;
    if k.nulls_first == default_nulls_first {
        format!("{} {dir}", k.expr)
    } else {
        format!(
            "{} {dir} NULLS {}",
            k.expr,
            if k.nulls_first { "FIRST" } else { "LAST" }
        )
    }
}

impl std::fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.display_tree(&|_| String::new()))
    }
}

impl LogicalPlan {
    /// Rebuild this node with each direct child transformed by `f`.
    pub fn map_children(
        self,
        f: &mut dyn FnMut(LogicalPlan) -> Result<LogicalPlan>,
    ) -> Result<LogicalPlan> {
        let mut b = |p: Box<LogicalPlan>| f(*p).map(Box::new);
        Ok(match self {
            LogicalPlan::Scan(_) | LogicalPlan::Empty { .. } => self,
            LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
                input: b(input)?,
                predicate,
            },
            LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
                input: b(input)?,
                exprs,
            },
            LogicalPlan::Join {
                left,
                right,
                join_type,
                on,
                filter,
            } => LogicalPlan::Join {
                left: b(left)?,
                right: b(right)?,
                join_type,
                on,
                filter,
            },
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggs,
            } => LogicalPlan::Aggregate {
                input: b(input)?,
                group_by,
                aggs,
            },
            LogicalPlan::Sort { input, keys, fetch } => LogicalPlan::Sort {
                input: b(input)?,
                keys,
                fetch,
            },
            LogicalPlan::Limit { input, skip, fetch } => LogicalPlan::Limit {
                input: b(input)?,
                skip,
                fetch,
            },
        })
    }

    /// Rebuild this node with every expression it owns (not its children's)
    /// transformed by `f`.
    pub fn map_exprs(self, f: &mut dyn FnMut(Expr) -> Result<Expr>) -> Result<LogicalPlan> {
        Ok(match self {
            LogicalPlan::Scan(mut s) => {
                s.filters = s.filters.into_iter().map(&mut *f).collect::<Result<_>>()?;
                LogicalPlan::Scan(s)
            }
            LogicalPlan::Empty { .. } => self,
            LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
                input,
                predicate: f(predicate)?,
            },
            LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
                input,
                exprs: exprs
                    .into_iter()
                    .map(|(e, c)| Ok((f(e)?, c)))
                    .collect::<Result<_>>()?,
            },
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
                on: on
                    .into_iter()
                    .map(|(l, r)| Ok((f(l)?, f(r)?)))
                    .collect::<Result<_>>()?,
                filter: filter.map(&mut *f).transpose()?,
            },
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggs,
            } => LogicalPlan::Aggregate {
                input,
                group_by: group_by
                    .into_iter()
                    .map(|(e, c)| Ok((f(e)?, c)))
                    .collect::<Result<_>>()?,
                aggs: aggs
                    .into_iter()
                    .map(|mut a| {
                        a.arg = a.arg.map(&mut *f).transpose()?;
                        Ok(a)
                    })
                    .collect::<Result<_>>()?,
            },
            LogicalPlan::Sort { input, keys, fetch } => LogicalPlan::Sort {
                input,
                keys: keys
                    .into_iter()
                    .map(|mut k| {
                        k.expr = f(k.expr)?;
                        Ok(k)
                    })
                    .collect::<Result<_>>()?,
                fetch,
            },
            LogicalPlan::Limit { .. } => self,
        })
    }

    /// Apply `f` to every node bottom-up.
    pub fn transform_up(
        self,
        f: &mut dyn FnMut(LogicalPlan) -> Result<LogicalPlan>,
    ) -> Result<LogicalPlan> {
        let rebuilt = self.map_children(&mut |c| c.transform_up(f))?;
        f(rebuilt)
    }

    /// Visit every expression in the tree (pre-order).
    pub fn visit_exprs(&self, f: &mut dyn FnMut(&Expr)) {
        match self {
            LogicalPlan::Scan(s) => s.filters.iter().for_each(&mut *f),
            LogicalPlan::Empty { .. } | LogicalPlan::Limit { .. } => {}
            LogicalPlan::Filter { predicate, .. } => f(predicate),
            LogicalPlan::Project { exprs, .. } => exprs.iter().for_each(|(e, _)| f(e)),
            LogicalPlan::Join { on, filter, .. } => {
                for (l, r) in on {
                    f(l);
                    f(r);
                }
                if let Some(x) = filter {
                    f(x);
                }
            }
            LogicalPlan::Aggregate { group_by, aggs, .. } => {
                group_by.iter().for_each(|(e, _)| f(e));
                aggs.iter().filter_map(|a| a.arg.as_ref()).for_each(&mut *f);
            }
            LogicalPlan::Sort { keys, .. } => keys.iter().for_each(|k| f(&k.expr)),
        }
        for c in self.children() {
            c.visit_exprs(f);
        }
    }
}

/// Shorten display names: a column reference is shown unqualified unless its
/// bare name is shared by columns with different ids somewhere in the plan.
pub fn simplify_display_names(plan: LogicalPlan) -> Result<LogicalPlan> {
    let mut by_name: HashMap<String, HashSet<ColId>> = HashMap::new();
    fn bare(n: &str) -> &str {
        n.rsplit('.').next().unwrap_or(n)
    }
    plan.visit_exprs(&mut |e| {
        let mut stack = vec![e];
        while let Some(x) = stack.pop() {
            if let Expr::Column(c) = x {
                by_name
                    .entry(bare(&c.name).to_string())
                    .or_default()
                    .insert(c.id);
            }
            stack.extend(x.children());
        }
    });
    fn collect_scans(p: &LogicalPlan, by_name: &mut HashMap<String, HashSet<ColId>>) {
        if let LogicalPlan::Scan(s) = p {
            for c in &s.columns {
                by_name.entry(c.name.clone()).or_default().insert(c.id);
            }
        }
        for ch in p.children() {
            collect_scans(ch, by_name);
        }
    }
    collect_scans(&plan, &mut by_name);
    let unique: HashSet<String> = by_name
        .into_iter()
        .filter(|(_, ids)| ids.len() == 1)
        .map(|(n, _)| n)
        .collect();
    plan.transform_up(&mut |node| {
        let node = match node {
            LogicalPlan::Scan(mut s) => {
                for c in &mut s.columns {
                    if unique.contains(&c.name) {
                        c.qualifier = None;
                    }
                }
                LogicalPlan::Scan(s)
            }
            other => other,
        };
        node.map_exprs(&mut |e| {
            e.transform_up(&mut |x| {
                Ok(match x {
                    Expr::Column(mut c) => {
                        let b = bare(&c.name).to_string();
                        if unique.contains(&b) {
                            c.name = b;
                        }
                        Expr::Column(c)
                    }
                    other => other,
                })
            })
        })
    })
}
