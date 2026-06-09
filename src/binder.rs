//! The binder: resolves names, checks types and lowers the AST into a
//! [`LogicalPlan`].
//!
//! Responsibilities:
//! * name resolution with SQL scoping (aliases, qualified names, ambiguity),
//! * CTE expansion (each reference re-binds the CTE body, so a CTE used twice
//!   gets distinct column ids and can be self-joined),
//! * type checking and implicit coercion (`INT` vs `DOUBLE`, string literals
//!   compared with dates, untyped `NULL`),
//! * aggregate analysis (`GROUP BY` / `HAVING`, aggregates in `ORDER BY`),
//! * `ORDER BY` on aliases, ordinals and hidden (non-selected) expressions.

use std::rc::Rc;

use crate::catalog::Catalog;
use crate::error::{bind_err, type_err, QuarryError, Result};
use crate::plan::expr::{AggFunc, BinaryOp, ColId, Expr, ScalarFunc};
use crate::plan::logical::{AggregateExpr, ColumnDef, JoinType, LogicalPlan, ScanNode, SortKey};
use crate::sql::ast;
use crate::types::{parse_date, DataType, ScalarValue};

/// Qualifier used for internal aggregate placeholders. It contains a NUL byte,
/// which the lexer can never produce, so user SQL cannot collide with it.
const AGG_QUALIFIER: &str = "\u{0}agg";

#[derive(Debug, Clone)]
struct ScopeColumn {
    qualifier: Option<String>,
    name: String,
    def: ColumnDef,
}

/// The set of columns visible to an expression.
#[derive(Debug, Clone, Default)]
struct Scope {
    columns: Vec<ScopeColumn>,
}

impl Scope {
    fn extend(&mut self, other: Scope) {
        self.columns.extend(other.columns);
    }

    fn qualifiers(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .columns
            .iter()
            .filter_map(|c| c.qualifier.as_deref())
            .collect();
        v.dedup();
        v
    }

    fn resolve(&self, parts: &[String]) -> Result<ColumnDef> {
        let (qual, name) = match parts {
            [name] => (None, name),
            [q, name] => (Some(q), name),
            _ => return bind_err(format!("invalid identifier {}", parts.join("."))),
        };
        let mut found: Option<&ScopeColumn> = None;
        for c in &self.columns {
            if &c.name != name {
                continue;
            }
            if let Some(q) = qual {
                if c.qualifier.as_ref() != Some(q) {
                    continue;
                }
            }
            if let Some(prev) = found {
                if prev.def.id != c.def.id {
                    return bind_err(format!(
                        "column reference '{}' is ambiguous",
                        parts.join(".")
                    ));
                }
            }
            found = Some(c);
        }
        match found {
            Some(c) => Ok(c.def.clone()),
            None => {
                if let Some(q) = qual {
                    if !self.columns.iter().any(|c| c.qualifier.as_ref() == Some(q)) {
                        return bind_err(format!("unknown table or alias '{q}'"));
                    }
                }
                bind_err(format!("column '{}' not found", parts.join(".")))
            }
        }
    }
}

#[derive(Debug)]
struct CteDef {
    query: ast::Query,
    columns: Vec<String>,
    /// CTEs visible where this one was defined (earlier siblings and outer CTEs).
    env: Vec<(String, Rc<CteDef>)>,
}

/// State for binding the select list / HAVING / ORDER BY of an aggregate query.
struct AggContext {
    base: Scope,
    group: Vec<(Expr, ColumnDef)>,
    aggs: Vec<AggregateExpr>,
}

/// Converts parsed SQL into logical plans against a catalog.
pub struct Binder<'a> {
    catalog: &'a Catalog,
    next_id: ColId,
    ctes: Vec<(String, Rc<CteDef>)>,
}

impl<'a> Binder<'a> {
    pub fn new(catalog: &'a Catalog) -> Self {
        Binder {
            catalog,
            next_id: 0,
            ctes: Vec::new(),
        }
    }

    fn fresh_id(&mut self) -> ColId {
        self.next_id += 1;
        self.next_id
    }

    /// Bind a top-level query and tidy column display names.
    pub fn bind(&mut self, q: &ast::Query) -> Result<LogicalPlan> {
        let plan = self.bind_query(q)?;
        crate::plan::logical::simplify_display_names(plan)
    }

    /// Bind a (possibly nested) query.
    fn bind_query(&mut self, q: &ast::Query) -> Result<LogicalPlan> {
        let saved = self.ctes.len();
        for cte in &q.ctes {
            if q.ctes.iter().filter(|c| c.name == cte.name).count() > 1 {
                return bind_err(format!(
                    "WITH query name '{}' specified more than once",
                    cte.name
                ));
            }
            let def = CteDef {
                query: (*cte.query).clone(),
                columns: cte.columns.clone(),
                env: self.ctes.clone(),
            };
            self.ctes.push((cte.name.clone(), Rc::new(def)));
        }
        let result = self.bind_select(&q.body, &q.order_by, q.limit, q.offset);
        self.ctes.truncate(saved);
        result
    }

    fn bind_select(
        &mut self,
        select: &ast::Select,
        order_by: &[ast::OrderByExpr],
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> Result<LogicalPlan> {
        let (mut plan, scope) = self.bind_from(&select.from)?;

        if let Some(w) = &select.selection {
            if contains_aggregate(w) {
                return bind_err("aggregate functions are not allowed in WHERE");
            }
            let pred = self.bind_expr(w, &scope)?;
            let pred = expect_boolean(pred, "WHERE")?;
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicate: pred,
            };
        }

        let is_aggregate = !select.group_by.is_empty()
            || select.having.is_some()
            || select.projection.iter().any(|item| {
                matches!(item, ast::SelectItem::Expr { expr, .. } if contains_aggregate(expr))
            })
            || order_by.iter().any(|o| contains_aggregate(&o.expr));

        // Expand the select list into (ast expr, output name) pairs.
        let items = self.expand_select_items(&select.projection, &scope)?;

        let mut agg_ctx = None;
        let mut projections: Vec<(Expr, String)> = Vec::new();
        if is_aggregate {
            let mut ctx = AggContext {
                base: scope.clone(),
                group: Vec::new(),
                aggs: Vec::new(),
            };
            for g in &select.group_by {
                let ge = self.bind_group_expr(g, &items, &scope)?;
                if ctx.group.iter().any(|(e, _)| e == &ge) {
                    continue;
                }
                let def = match &ge {
                    Expr::Column(c) => ColumnDef {
                        id: c.id,
                        name: column_name_of(&ge),
                        qualifier: None,
                        data_type: c.data_type,
                        nullable: c.nullable,
                    },
                    other => ColumnDef {
                        id: self.fresh_id(),
                        name: other.to_string(),
                        qualifier: None,
                        data_type: other.data_type(),
                        nullable: other.nullable(),
                    },
                };
                ctx.group.push((ge, def));
            }
            for (ast_expr, name) in &items {
                let e = self.bind_post_aggregate(ast_expr, &mut ctx)?;
                projections.push((e, name.clone()));
            }
            let having = match &select.having {
                Some(h) => Some(expect_boolean(
                    self.bind_post_aggregate(h, &mut ctx)?,
                    "HAVING",
                )?),
                None => None,
            };
            agg_ctx = Some((ctx, having));
        } else {
            for (ast_expr, name) in &items {
                projections.push((self.bind_expr(ast_expr, &scope)?, name.clone()));
            }
        }

        // ORDER BY keys are resolved before the projection is built because
        // they may add hidden columns to it.
        let mut sort_targets = Vec::new();
        for o in order_by {
            let target = self.bind_order_target(
                o,
                &items,
                &projections,
                &scope,
                agg_ctx.as_mut().map(|(c, _)| c),
            )?;
            sort_targets.push((target, o.asc, o.nulls_first.unwrap_or(o.asc)));
        }

        if let Some((ctx, having)) = agg_ctx {
            plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group_by: ctx.group,
                aggs: ctx.aggs,
            };
            if let Some(h) = having {
                plan = LogicalPlan::Filter {
                    input: Box::new(plan),
                    predicate: h,
                };
            }
        }

        let visible = projections.len();
        let mut hidden = 0;
        let mut key_indices = Vec::new();
        for (target, _, _) in &sort_targets {
            match target {
                OrderTarget::Output(i) => key_indices.push(*i),
                OrderTarget::Hidden(e) => {
                    if select.distinct {
                        return bind_err(
                            "for SELECT DISTINCT, ORDER BY expressions must appear in the select list",
                        );
                    }
                    let pos = projections
                        .iter()
                        .position(|(p, _)| p == e)
                        .unwrap_or_else(|| {
                            projections.push((e.clone(), format!("__order{hidden}")));
                            hidden += 1;
                            projections.len() - 1
                        });
                    key_indices.push(pos);
                }
            }
        }

        let exprs = self.make_projection(projections);
        let mut out_cols: Vec<ColumnDef> = exprs.iter().map(|(_, c)| c.clone()).collect();
        plan = LogicalPlan::Project {
            input: Box::new(plan),
            exprs,
        };

        if select.distinct {
            let group_by: Vec<(Expr, ColumnDef)> = out_cols
                .iter()
                .map(|c| {
                    let def = ColumnDef {
                        id: self.fresh_id(),
                        ..c.clone()
                    };
                    (c.to_expr(), def)
                })
                .collect();
            out_cols = group_by.iter().map(|(_, c)| c.clone()).collect();
            plan = LogicalPlan::Aggregate {
                input: Box::new(plan),
                group_by,
                aggs: Vec::new(),
            };
        }

        if !sort_targets.is_empty() {
            let keys = sort_targets
                .iter()
                .zip(&key_indices)
                .map(|((_, asc, nulls_first), &i)| SortKey {
                    expr: out_cols[i].to_expr(),
                    asc: *asc,
                    nulls_first: *nulls_first,
                })
                .collect();
            plan = LogicalPlan::Sort {
                input: Box::new(plan),
                keys,
                fetch: None,
            };
        }

        if limit.is_some() || offset.is_some() {
            plan = LogicalPlan::Limit {
                input: Box::new(plan),
                skip: offset.unwrap_or(0) as usize,
                fetch: limit.map(|l| l as usize),
            };
        }

        if hidden > 0 {
            let exprs = out_cols[..visible]
                .iter()
                .map(|c| (c.to_expr(), c.clone()))
                .collect();
            plan = LogicalPlan::Project {
                input: Box::new(plan),
                exprs,
            };
        }
        Ok(plan)
    }

    /// Give each projected expression an output column. Bare column references
    /// keep their id (unless already used in this projection).
    fn make_projection(&mut self, items: Vec<(Expr, String)>) -> Vec<(Expr, ColumnDef)> {
        let mut used = std::collections::HashSet::new();
        items
            .into_iter()
            .map(|(e, name)| {
                let id = match &e {
                    Expr::Column(c) if !used.contains(&c.id) => c.id,
                    _ => self.fresh_id(),
                };
                used.insert(id);
                let def = ColumnDef {
                    id,
                    name,
                    qualifier: None,
                    data_type: e.data_type(),
                    nullable: e.nullable(),
                };
                (e, def)
            })
            .collect()
    }

    fn expand_select_items(
        &mut self,
        items: &[ast::SelectItem],
        scope: &Scope,
    ) -> Result<Vec<(ast::Expr, String)>> {
        let mut out = Vec::new();
        for item in items {
            match item {
                ast::SelectItem::Wildcard => {
                    if scope.columns.is_empty() {
                        return bind_err("SELECT * with no tables specified");
                    }
                    for c in &scope.columns {
                        out.push((scope_ident(c), c.name.clone()));
                    }
                }
                ast::SelectItem::QualifiedWildcard(q) => {
                    let cols: Vec<&ScopeColumn> = scope
                        .columns
                        .iter()
                        .filter(|c| c.qualifier.as_deref() == Some(q.as_str()))
                        .collect();
                    if cols.is_empty() {
                        return bind_err(format!("unknown table or alias '{q}'"));
                    }
                    for c in cols {
                        out.push((scope_ident(c), c.name.clone()));
                    }
                }
                ast::SelectItem::Expr { expr, alias } => {
                    let name = alias.clone().unwrap_or_else(|| default_name(expr));
                    out.push((expr.clone(), name));
                }
            }
        }
        Ok(out)
    }

    /// GROUP BY items may be expressions, select-list ordinals or select aliases.
    fn bind_group_expr(
        &mut self,
        g: &ast::Expr,
        items: &[(ast::Expr, String)],
        scope: &Scope,
    ) -> Result<Expr> {
        if contains_aggregate(g) {
            return bind_err("aggregate functions are not allowed in GROUP BY");
        }
        if let ast::Expr::Literal(ast::Literal::Integer(k)) = g {
            let idx = usize::try_from(*k)
                .ok()
                .filter(|i| *i >= 1 && *i <= items.len());
            return match idx {
                Some(i) => {
                    let item = &items[i - 1].0;
                    if contains_aggregate(item) {
                        return bind_err(format!("GROUP BY position {k} refers to an aggregate"));
                    }
                    self.bind_expr(item, scope)
                }
                None => bind_err(format!("GROUP BY position {k} is not in the select list")),
            };
        }
        match self.bind_expr(g, scope) {
            Ok(e) => Ok(e),
            Err(err) => {
                if let ast::Expr::Identifier(parts) = g {
                    if parts.len() == 1 {
                        if let Some((item, _)) = items.iter().find(|(_, n)| n == &parts[0]) {
                            if !contains_aggregate(item) {
                                return self.bind_expr(item, scope);
                            }
                        }
                    }
                }
                Err(err)
            }
        }
    }

    fn bind_order_target(
        &mut self,
        o: &ast::OrderByExpr,
        items: &[(ast::Expr, String)],
        projections: &[(Expr, String)],
        scope: &Scope,
        agg: Option<&mut AggContext>,
    ) -> Result<OrderTarget> {
        if let ast::Expr::Literal(ast::Literal::Integer(k)) = &o.expr {
            return match usize::try_from(*k)
                .ok()
                .filter(|i| *i >= 1 && *i <= items.len())
            {
                Some(i) => Ok(OrderTarget::Output(i - 1)),
                None => bind_err(format!("ORDER BY position {k} is not in the select list")),
            };
        }
        if let ast::Expr::Identifier(parts) = &o.expr {
            if parts.len() == 1 {
                let matches: Vec<usize> = items
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, n))| n == &parts[0])
                    .map(|(i, _)| i)
                    .collect();
                if let [i] = matches[..] {
                    return Ok(OrderTarget::Output(i));
                }
                if matches.len() > 1 {
                    let first = &projections[matches[0]].0;
                    if matches.iter().all(|&i| &projections[i].0 == first) {
                        return Ok(OrderTarget::Output(matches[0]));
                    }
                    return bind_err(format!("ORDER BY '{}' is ambiguous", parts[0]));
                }
            }
        }
        let bound = match agg {
            Some(ctx) => self.bind_post_aggregate(&o.expr, ctx)?,
            None => self.bind_expr(&o.expr, scope)?,
        };
        match projections.iter().position(|(p, _)| p == &bound) {
            Some(i) => Ok(OrderTarget::Output(i)),
            None => Ok(OrderTarget::Hidden(bound)),
        }
    }

    /// Bind an expression evaluated after aggregation: aggregate calls become
    /// references to aggregate outputs and grouped expressions become
    /// references to group keys. Anything else is an error.
    fn bind_post_aggregate(&mut self, e: &ast::Expr, ctx: &mut AggContext) -> Result<Expr> {
        let mut placeholders = Scope::default();
        let rewritten = self.extract_aggregates(e, ctx, &mut placeholders)?;
        let mut scope = ctx.base.clone();
        scope.extend(placeholders);
        let bound = self.bind_expr(&rewritten, &scope)?;
        let replaced = replace_group_exprs(bound, &ctx.group);
        let allowed: std::collections::HashSet<ColId> = ctx
            .group
            .iter()
            .map(|(_, c)| c.id)
            .chain(ctx.aggs.iter().map(|a| a.output.id))
            .collect();
        let mut bad = None;
        check_columns(&replaced, &allowed, &mut bad);
        if let Some(name) = bad {
            return bind_err(format!(
                "column '{name}' must appear in the GROUP BY clause or be used in an aggregate function"
            ));
        }
        Ok(replaced)
    }

    /// Replace aggregate calls in `e` with placeholder identifiers, registering
    /// each distinct aggregate in `ctx`.
    fn extract_aggregates(
        &mut self,
        e: &ast::Expr,
        ctx: &mut AggContext,
        placeholders: &mut Scope,
    ) -> Result<ast::Expr> {
        map_ast(e, &mut |node| {
            let ast::Expr::Function {
                name,
                args,
                distinct,
                star,
            } = node
            else {
                return Ok(None);
            };
            let Some(func) = aggregate_func(name, *star) else {
                return Ok(None);
            };
            if args.iter().any(contains_aggregate) {
                return bind_err("aggregate function calls cannot be nested");
            }
            let agg = self.bind_aggregate(func, args, *distinct, &ctx.base)?;
            let existing = ctx
                .aggs
                .iter()
                .find(|a| a.func == agg.func && a.arg == agg.arg && a.distinct == agg.distinct)
                .map(|a| a.output.clone());
            let output = match existing {
                Some(o) => o,
                None => {
                    let output = ColumnDef {
                        id: self.fresh_id(),
                        name: node.to_string(),
                        qualifier: None,
                        data_type: agg.ret,
                        nullable: agg.func != AggFunc::Count && agg.func != AggFunc::CountStar,
                    };
                    ctx.aggs.push(AggregateExpr {
                        func: agg.func,
                        arg: agg.arg,
                        distinct: agg.distinct,
                        output: output.clone(),
                    });
                    output
                }
            };
            let label = output.id.to_string();
            placeholders.columns.push(ScopeColumn {
                qualifier: Some(AGG_QUALIFIER.to_string()),
                name: label.clone(),
                def: output,
            });
            Ok(Some(ast::Expr::Identifier(vec![
                AGG_QUALIFIER.to_string(),
                label,
            ])))
        })
    }

    fn bind_aggregate(
        &mut self,
        func: AggFunc,
        args: &[ast::Expr],
        distinct: bool,
        scope: &Scope,
    ) -> Result<BoundAggregate> {
        if func == AggFunc::CountStar {
            return Ok(BoundAggregate {
                func,
                arg: None,
                distinct: false,
                ret: DataType::Int64,
            });
        }
        let [arg] = args else {
            return bind_err(format!("{func}() takes exactly one argument"));
        };
        let arg = self.bind_expr(arg, scope)?;
        let t = arg.data_type();
        let (arg, ret) = match func {
            AggFunc::Count => (arg, DataType::Int64),
            AggFunc::Sum => match t {
                DataType::Int64 | DataType::Float64 => (arg, t),
                DataType::Null => (cast_to(arg, DataType::Int64)?, DataType::Int64),
                _ => return type_err(format!("sum() requires a numeric argument, got {t}")),
            },
            AggFunc::Avg => match t {
                DataType::Int64 | DataType::Float64 | DataType::Null => {
                    (cast_to(arg, DataType::Float64)?, DataType::Float64)
                }
                _ => return type_err(format!("avg() requires a numeric argument, got {t}")),
            },
            AggFunc::Min | AggFunc::Max => {
                if t == DataType::Null {
                    (cast_to(arg, DataType::Int64)?, DataType::Int64)
                } else {
                    (arg, t)
                }
            }
            AggFunc::CountStar => unreachable!("handled above"),
        };
        Ok(BoundAggregate {
            func,
            arg: Some(arg),
            distinct,
            ret,
        })
    }

    // ----- FROM -----------------------------------------------------------

    fn bind_from(&mut self, from: &[ast::TableWithJoins]) -> Result<(LogicalPlan, Scope)> {
        if from.is_empty() {
            return Ok((
                LogicalPlan::Empty {
                    columns: vec![],
                    one_row: true,
                },
                Scope::default(),
            ));
        }
        let mut acc: Option<(LogicalPlan, Scope)> = None;
        for item in from {
            let (plan, scope) = self.bind_table_with_joins(item)?;
            acc = Some(match acc {
                None => (plan, scope),
                Some((lp, ls)) => {
                    let merged = merge_scopes(ls, scope)?;
                    (cross_join(lp, plan), merged)
                }
            });
        }
        Ok(acc.unwrap_or_else(|| unreachable!("from is non-empty")))
    }

    fn bind_table_with_joins(&mut self, t: &ast::TableWithJoins) -> Result<(LogicalPlan, Scope)> {
        let (mut plan, mut scope) = self.bind_factor(&t.relation)?;
        for j in &t.joins {
            let (rplan, rscope) = self.bind_factor(&j.relation)?;
            scope = merge_scopes(scope, rscope)?;
            plan = match j.kind {
                ast::JoinKind::Cross => cross_join(plan, rplan),
                kind => {
                    let on = j
                        .on
                        .as_ref()
                        .ok_or_else(|| QuarryError::Bind("JOIN requires an ON condition".into()))?;
                    if contains_aggregate(on) {
                        return bind_err("aggregate functions are not allowed in JOIN conditions");
                    }
                    let cond = expect_boolean(self.bind_expr(on, &scope)?, "JOIN ... ON")?;
                    LogicalPlan::Join {
                        left: Box::new(plan),
                        right: Box::new(rplan),
                        join_type: if kind == ast::JoinKind::Left {
                            JoinType::Left
                        } else {
                            JoinType::Inner
                        },
                        on: vec![],
                        filter: Some(cond),
                    }
                }
            };
        }
        Ok((plan, scope))
    }

    fn bind_factor(&mut self, f: &ast::TableFactor) -> Result<(LogicalPlan, Scope)> {
        match f {
            ast::TableFactor::Table { name, alias } => {
                let qualifier = alias.clone().unwrap_or_else(|| name.clone());
                if let Some(def) = self.lookup_cte(name) {
                    let plan = self.bind_cte(&def)?;
                    let names = if def.columns.is_empty() {
                        plan.schema().iter().map(|c| c.name.clone()).collect()
                    } else {
                        def.columns.clone()
                    };
                    let scope = derived_scope(&plan, &qualifier, &names, name)?;
                    return Ok((plan, scope));
                }
                let table = self
                    .catalog
                    .get(name)
                    .ok_or_else(|| QuarryError::Bind(format!("table '{name}' does not exist")))?;
                let mut columns = Vec::new();
                let mut scope = Scope::default();
                for (i, field) in table.schema.fields().iter().enumerate() {
                    let def = ColumnDef {
                        id: self.fresh_id(),
                        name: field.name().clone(),
                        qualifier: Some(qualifier.clone()),
                        data_type: table.column_type(i),
                        nullable: true,
                    };
                    scope.columns.push(ScopeColumn {
                        qualifier: Some(qualifier.clone()),
                        name: field.name().clone(),
                        def: def.clone(),
                    });
                    columns.push(def);
                }
                let source_indices = (0..columns.len()).collect();
                let plan = LogicalPlan::Scan(ScanNode {
                    table,
                    alias: qualifier,
                    columns,
                    source_indices,
                    filters: vec![],
                });
                Ok((plan, scope))
            }
            ast::TableFactor::Derived {
                subquery,
                alias,
                columns,
            } => {
                // Derived tables are uncorrelated: bind with no outer scope.
                let plan = self.bind_query(subquery)?;
                let names = if columns.is_empty() {
                    plan.schema().iter().map(|c| c.name.clone()).collect()
                } else {
                    columns.clone()
                };
                let scope = derived_scope(&plan, alias, &names, alias)?;
                Ok((plan, scope))
            }
        }
    }

    fn lookup_cte(&self, name: &str) -> Option<Rc<CteDef>> {
        self.ctes
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, d)| Rc::clone(d))
    }

    fn bind_cte(&mut self, def: &CteDef) -> Result<LogicalPlan> {
        let saved = std::mem::replace(&mut self.ctes, def.env.clone());
        let result = self.bind_query(&def.query);
        self.ctes = saved;
        result
    }

    // ----- expressions ----------------------------------------------------

    fn bind_expr(&mut self, e: &ast::Expr, scope: &Scope) -> Result<Expr> {
        use ast::Expr as A;
        match e {
            A::Identifier(parts) => Ok(scope.resolve(parts)?.to_expr()),
            A::Literal(l) => bind_literal(l),
            A::Unary { op, expr } => {
                let inner = self.bind_expr(expr, scope)?;
                match op {
                    ast::UnaryOp::Not => Ok(Expr::Not(Box::new(expect_boolean(inner, "NOT")?))),
                    ast::UnaryOp::Neg => {
                        let t = inner.data_type();
                        if !t.is_numeric() && t != DataType::Null {
                            return type_err(format!("cannot negate a value of type {t}"));
                        }
                        Ok(Expr::Negative(Box::new(inner)))
                    }
                }
            }
            A::Binary { left, op, right } => self.bind_binary(left, *op, right, scope),
            A::IsNull { expr, negated } => {
                let inner = Box::new(self.bind_expr(expr, scope)?);
                Ok(if *negated {
                    Expr::IsNotNull(inner)
                } else {
                    Expr::IsNull(inner)
                })
            }
            A::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let x = self.bind_expr(expr, scope)?;
                let lo = self.bind_expr(low, scope)?;
                let hi = self.bind_expr(high, scope)?;
                let (op_lo, op_hi, join) = if *negated {
                    (BinaryOp::Lt, BinaryOp::Gt, BinaryOp::Or)
                } else {
                    (BinaryOp::GtEq, BinaryOp::LtEq, BinaryOp::And)
                };
                let a = comparison(x.clone(), op_lo, lo)?;
                let b = comparison(x, op_hi, hi)?;
                Ok(Expr::binary(a, join, b))
            }
            A::InList {
                expr,
                list,
                negated,
            } => {
                let x = self.bind_expr(expr, scope)?;
                let items = list
                    .iter()
                    .map(|i| self.bind_expr(i, scope))
                    .collect::<Result<Vec<_>>>()?;
                let mut t = x.data_type();
                for i in &items {
                    t = common_type(t, i.data_type()).ok_or_else(|| {
                        QuarryError::Type(format!("IN list item {i} is incompatible with {t}"))
                    })?;
                }
                Ok(Expr::InList {
                    expr: Box::new(cast_to(x, t)?),
                    list: items
                        .into_iter()
                        .map(|i| cast_to(i, t))
                        .collect::<Result<_>>()?,
                    negated: *negated,
                })
            }
            A::Like {
                expr,
                pattern,
                negated,
            } => {
                let x = expect_type(self.bind_expr(expr, scope)?, DataType::Utf8, "LIKE")?;
                let p = expect_type(
                    self.bind_expr(pattern, scope)?,
                    DataType::Utf8,
                    "LIKE pattern",
                )?;
                Ok(Expr::Like {
                    expr: Box::new(x),
                    pattern: Box::new(p),
                    negated: *negated,
                })
            }
            A::Case {
                operand,
                branches,
                else_expr,
            } => {
                let operand = match operand {
                    Some(o) => Some(self.bind_expr(o, scope)?),
                    None => None,
                };
                let mut conds = Vec::new();
                let mut results = Vec::new();
                for (w, t) in branches {
                    let w = self.bind_expr(w, scope)?;
                    let cond = match &operand {
                        Some(o) => comparison(o.clone(), BinaryOp::Eq, w)?,
                        None => expect_boolean(w, "CASE WHEN")?,
                    };
                    conds.push(cond);
                    results.push(self.bind_expr(t, scope)?);
                }
                let else_bound = match else_expr {
                    Some(e) => Some(self.bind_expr(e, scope)?),
                    None => None,
                };
                let mut t = DataType::Null;
                for r in results.iter().chain(else_bound.iter()) {
                    t = common_type(t, r.data_type()).ok_or_else(|| {
                        QuarryError::Type(format!(
                            "CASE branches have incompatible types {t} and {}",
                            r.data_type()
                        ))
                    })?;
                }
                let branches = conds
                    .into_iter()
                    .zip(results)
                    .map(|(c, r)| Ok((c, cast_to(r, t)?)))
                    .collect::<Result<_>>()?;
                let else_expr = match else_bound {
                    Some(e) => Some(Box::new(cast_to(e, t)?)),
                    None => None,
                };
                Ok(Expr::Case {
                    branches,
                    else_expr,
                    data_type: t,
                })
            }
            A::Cast { expr, data_type } => {
                let inner = self.bind_expr(expr, scope)?;
                let to = match data_type {
                    ast::TypeName::Boolean => DataType::Boolean,
                    ast::TypeName::BigInt => DataType::Int64,
                    ast::TypeName::Double => DataType::Float64,
                    ast::TypeName::Varchar => DataType::Utf8,
                    ast::TypeName::Date => DataType::Date32,
                };
                explicit_cast(inner, to)
            }
            A::Extract { field, expr } => {
                let inner = expect_type(self.bind_expr(expr, scope)?, DataType::Date32, "EXTRACT")?;
                let func = match field {
                    ast::DateField::Year => ScalarFunc::ExtractYear,
                    ast::DateField::Month => ScalarFunc::ExtractMonth,
                    ast::DateField::Day => ScalarFunc::ExtractDay,
                };
                Ok(Expr::Function {
                    func,
                    args: vec![inner],
                    data_type: DataType::Int64,
                })
            }
            A::Function {
                name,
                args,
                distinct,
                star,
            } => {
                if aggregate_func(name, *star).is_some() {
                    return bind_err(format!("aggregate function {name}() is not allowed here"));
                }
                if *star || *distinct {
                    return bind_err(format!("{name}() does not accept * or DISTINCT"));
                }
                self.bind_scalar_function(name, args, scope)
            }
        }
    }

    fn bind_binary(
        &mut self,
        left: &ast::Expr,
        op: BinaryOp,
        right: &ast::Expr,
        scope: &Scope,
    ) -> Result<Expr> {
        // date +/- INTERVAL is rewritten into a calendar function.
        if matches!(op, BinaryOp::Plus | BinaryOp::Minus) {
            let interval = |e: &ast::Expr| match e {
                ast::Expr::Literal(ast::Literal::Interval { value, unit }) => Some((*value, *unit)),
                _ => None,
            };
            let (date_side, iv) = match (interval(left), interval(right)) {
                (None, Some(iv)) => (left, iv),
                (Some(iv), None) if op == BinaryOp::Plus => (right, iv),
                (Some(_), _) => return type_err("INTERVAL must be added to a DATE"),
                (None, None) => (left, (0, ast::IntervalUnit::Day)),
            };
            if interval(left).is_some() || interval(right).is_some() {
                let d = self.bind_expr(date_side, scope)?;
                let d = expect_type(d, DataType::Date32, "date arithmetic")?;
                let sign = if op == BinaryOp::Minus { -1 } else { 1 };
                let v = i32::try_from(iv.0 * sign)
                    .map_err(|_| QuarryError::Type("interval value out of range".into()))?;
                let (months, days) = match iv.1 {
                    ast::IntervalUnit::Day => (0, v),
                    ast::IntervalUnit::Month => (v, 0),
                    ast::IntervalUnit::Year => (v.saturating_mul(12), 0),
                };
                return Ok(Expr::Function {
                    func: ScalarFunc::DateAdd { months, days },
                    args: vec![d],
                    data_type: DataType::Date32,
                });
            }
        }
        let l = self.bind_expr(left, scope)?;
        let r = self.bind_expr(right, scope)?;
        match op {
            BinaryOp::And | BinaryOp::Or => {
                let l = expect_boolean(l, "AND/OR")?;
                let r = expect_boolean(r, "AND/OR")?;
                Ok(Expr::binary(l, op, r))
            }
            BinaryOp::Concat => Ok(Expr::binary(
                explicit_cast(l, DataType::Utf8)?,
                op,
                explicit_cast(r, DataType::Utf8)?,
            )),
            o if o.is_comparison() => comparison(l, o, r),
            _ => {
                let (lt, rt) = (l.data_type(), r.data_type());
                let ok = |t: DataType| t.is_numeric() || t == DataType::Null;
                if !ok(lt) || !ok(rt) {
                    return type_err(format!("operator {op} is not defined for {lt} and {rt}"));
                }
                let t = match (lt, rt) {
                    (DataType::Null, DataType::Null) => DataType::Int64,
                    _ => common_type(lt, rt).unwrap_or(DataType::Float64),
                };
                Ok(Expr::binary(cast_to(l, t)?, op, cast_to(r, t)?))
            }
        }
    }

    fn bind_scalar_function(
        &mut self,
        name: &str,
        args: &[ast::Expr],
        scope: &Scope,
    ) -> Result<Expr> {
        let bound = args
            .iter()
            .map(|a| self.bind_expr(a, scope))
            .collect::<Result<Vec<_>>>()?;
        let arity = |lo: usize, hi: usize| -> Result<()> {
            if bound.len() < lo || bound.len() > hi {
                return bind_err(format!("wrong number of arguments to {name}()"));
            }
            Ok(())
        };
        let (func, args, ret) = match name {
            "abs" => {
                arity(1, 1)?;
                let t = bound[0].data_type();
                if !t.is_numeric() {
                    return type_err(format!("abs() requires a numeric argument, got {t}"));
                }
                (ScalarFunc::Abs, bound, t)
            }
            "lower" | "upper" | "length" | "char_length" => {
                arity(1, 1)?;
                let a = expect_type(
                    bound.into_iter().next().unwrap_or_else(|| unreachable!()),
                    DataType::Utf8,
                    name,
                )?;
                let (func, ret) = match name {
                    "lower" => (ScalarFunc::Lower, DataType::Utf8),
                    "upper" => (ScalarFunc::Upper, DataType::Utf8),
                    _ => (ScalarFunc::Length, DataType::Int64),
                };
                (func, vec![a], ret)
            }
            "substr" | "substring" => {
                arity(2, 3)?;
                let mut it = bound.into_iter();
                let mut args = vec![expect_type(
                    it.next().unwrap_or_else(|| unreachable!()),
                    DataType::Utf8,
                    name,
                )?];
                for a in it {
                    args.push(expect_type(a, DataType::Int64, name)?);
                }
                (ScalarFunc::Substr, args, DataType::Utf8)
            }
            "coalesce" => {
                if bound.is_empty() {
                    return bind_err("coalesce() requires at least one argument");
                }
                let mut t = DataType::Null;
                for a in &bound {
                    t = common_type(t, a.data_type()).ok_or_else(|| {
                        QuarryError::Type(format!(
                            "coalesce() arguments have incompatible types {t} and {}",
                            a.data_type()
                        ))
                    })?;
                }
                let args = bound
                    .into_iter()
                    .map(|a| cast_to(a, t))
                    .collect::<Result<_>>()?;
                (ScalarFunc::Coalesce, args, t)
            }
            "round" => {
                arity(1, 2)?;
                let mut it = bound.into_iter();
                let x = it.next().unwrap_or_else(|| unreachable!());
                if !x.data_type().is_numeric() {
                    return type_err("round() requires a numeric argument");
                }
                let mut args = vec![cast_to(x, DataType::Float64)?];
                for a in it {
                    args.push(expect_type(a, DataType::Int64, "round digits")?);
                }
                (ScalarFunc::Round, args, DataType::Float64)
            }
            _ => return bind_err(format!("unknown function {name}()")),
        };
        Ok(Expr::Function {
            func,
            args,
            data_type: ret,
        })
    }
}

enum OrderTarget {
    /// Sort by an existing select-list column.
    Output(usize),
    /// Sort by an expression that must be computed as a hidden column.
    Hidden(Expr),
}

struct BoundAggregate {
    func: AggFunc,
    arg: Option<Expr>,
    distinct: bool,
    ret: DataType,
}

fn aggregate_func(name: &str, star: bool) -> Option<AggFunc> {
    Some(match name {
        "count" if star => AggFunc::CountStar,
        "count" => AggFunc::Count,
        "sum" => AggFunc::Sum,
        "avg" => AggFunc::Avg,
        "min" => AggFunc::Min,
        "max" => AggFunc::Max,
        _ => return None,
    })
}

/// Does the AST contain an aggregate function call?
pub fn contains_aggregate(e: &ast::Expr) -> bool {
    let mut found = false;
    let _ = map_ast(e, &mut |node| {
        if let ast::Expr::Function { name, star, .. } = node {
            if aggregate_func(name, *star).is_some() {
                found = true;
            }
        }
        Ok(None)
    });
    found
}

/// Top-down AST rewrite: `f` may replace a node (children are then not visited).
fn map_ast(
    e: &ast::Expr,
    f: &mut dyn FnMut(&ast::Expr) -> Result<Option<ast::Expr>>,
) -> Result<ast::Expr> {
    use ast::Expr as A;
    if let Some(r) = f(e)? {
        return Ok(r);
    }
    let mut m = |x: &ast::Expr| map_ast(x, f).map(Box::new);
    Ok(match e {
        A::Identifier(_) | A::Literal(_) => e.clone(),
        A::Unary { op, expr } => A::Unary {
            op: *op,
            expr: m(expr)?,
        },
        A::Binary { left, op, right } => A::Binary {
            left: m(left)?,
            op: *op,
            right: m(right)?,
        },
        A::IsNull { expr, negated } => A::IsNull {
            expr: m(expr)?,
            negated: *negated,
        },
        A::Between {
            expr,
            low,
            high,
            negated,
        } => A::Between {
            expr: m(expr)?,
            low: m(low)?,
            high: m(high)?,
            negated: *negated,
        },
        A::InList {
            expr,
            list,
            negated,
        } => {
            let expr = m(expr)?;
            let list = list
                .iter()
                .map(|x| m(x).map(|b| *b))
                .collect::<Result<_>>()?;
            A::InList {
                expr,
                list,
                negated: *negated,
            }
        }
        A::Like {
            expr,
            pattern,
            negated,
        } => A::Like {
            expr: m(expr)?,
            pattern: m(pattern)?,
            negated: *negated,
        },
        A::Case {
            operand,
            branches,
            else_expr,
        } => {
            let operand = match operand {
                Some(o) => Some(m(o)?),
                None => None,
            };
            let branches = branches
                .iter()
                .map(|(w, t)| Ok((*m(w)?, *m(t)?)))
                .collect::<Result<_>>()?;
            let else_expr = match else_expr {
                Some(x) => Some(m(x)?),
                None => None,
            };
            A::Case {
                operand,
                branches,
                else_expr,
            }
        }
        A::Cast { expr, data_type } => A::Cast {
            expr: m(expr)?,
            data_type: *data_type,
        },
        A::Extract { field, expr } => A::Extract {
            field: *field,
            expr: m(expr)?,
        },
        A::Function {
            name,
            args,
            distinct,
            star,
        } => A::Function {
            name: name.clone(),
            args: args
                .iter()
                .map(|x| m(x).map(|b| *b))
                .collect::<Result<_>>()?,
            distinct: *distinct,
            star: *star,
        },
    })
}

/// Replace (top-down, largest first) sub-expressions equal to a GROUP BY key.
fn replace_group_exprs(e: Expr, group: &[(Expr, ColumnDef)]) -> Expr {
    if let Some((_, def)) = group.iter().find(|(g, _)| g == &e) {
        return def.to_expr();
    }
    match e {
        Expr::Column(_) | Expr::Literal(_) => e,
        Expr::Binary { left, op, right } => Expr::Binary {
            left: Box::new(replace_group_exprs(*left, group)),
            op,
            right: Box::new(replace_group_exprs(*right, group)),
        },
        Expr::Not(x) => Expr::Not(Box::new(replace_group_exprs(*x, group))),
        Expr::Negative(x) => Expr::Negative(Box::new(replace_group_exprs(*x, group))),
        Expr::IsNull(x) => Expr::IsNull(Box::new(replace_group_exprs(*x, group))),
        Expr::IsNotNull(x) => Expr::IsNotNull(Box::new(replace_group_exprs(*x, group))),
        Expr::Cast { expr, to } => Expr::Cast {
            expr: Box::new(replace_group_exprs(*expr, group)),
            to,
        },
        Expr::Like {
            expr,
            pattern,
            negated,
        } => Expr::Like {
            expr: Box::new(replace_group_exprs(*expr, group)),
            pattern: Box::new(replace_group_exprs(*pattern, group)),
            negated,
        },
        Expr::InList {
            expr,
            list,
            negated,
        } => Expr::InList {
            expr: Box::new(replace_group_exprs(*expr, group)),
            list: list
                .into_iter()
                .map(|x| replace_group_exprs(x, group))
                .collect(),
            negated,
        },
        Expr::Case {
            branches,
            else_expr,
            data_type,
        } => Expr::Case {
            branches: branches
                .into_iter()
                .map(|(w, t)| (replace_group_exprs(w, group), replace_group_exprs(t, group)))
                .collect(),
            else_expr: else_expr.map(|x| Box::new(replace_group_exprs(*x, group))),
            data_type,
        },
        Expr::Function {
            func,
            args,
            data_type,
        } => Expr::Function {
            func,
            args: args
                .into_iter()
                .map(|x| replace_group_exprs(x, group))
                .collect(),
            data_type,
        },
    }
}

fn check_columns(e: &Expr, allowed: &std::collections::HashSet<ColId>, bad: &mut Option<String>) {
    if bad.is_some() {
        return;
    }
    if let Expr::Column(c) = e {
        if !allowed.contains(&c.id) {
            *bad = Some(c.name.clone());
        }
        return;
    }
    for child in e.children() {
        check_columns(child, allowed, bad);
    }
}

fn scope_ident(c: &ScopeColumn) -> ast::Expr {
    match &c.qualifier {
        Some(q) => ast::Expr::Identifier(vec![q.clone(), c.name.clone()]),
        None => ast::Expr::Identifier(vec![c.name.clone()]),
    }
}

fn merge_scopes(mut left: Scope, right: Scope) -> Result<Scope> {
    let lq: Vec<String> = left.qualifiers().into_iter().map(String::from).collect();
    for q in right.qualifiers() {
        if lq.iter().any(|x| x == q) {
            return bind_err(format!(
                "table name '{q}' specified more than once (use an alias)"
            ));
        }
    }
    left.extend(right);
    Ok(left)
}

fn derived_scope(
    plan: &LogicalPlan,
    qualifier: &str,
    names: &[String],
    what: &str,
) -> Result<Scope> {
    let schema = plan.schema();
    if names.len() != schema.len() {
        return bind_err(format!(
            "'{what}' has {} columns but {} column names were given",
            schema.len(),
            names.len()
        ));
    }
    Ok(Scope {
        columns: schema
            .into_iter()
            .zip(names)
            .map(|(def, name)| ScopeColumn {
                qualifier: Some(qualifier.to_string()),
                name: name.clone(),
                def: ColumnDef {
                    name: name.clone(),
                    qualifier: Some(qualifier.to_string()),
                    ..def
                },
            })
            .collect(),
    })
}

fn cross_join(left: LogicalPlan, right: LogicalPlan) -> LogicalPlan {
    LogicalPlan::Join {
        left: Box::new(left),
        right: Box::new(right),
        join_type: JoinType::Inner,
        on: vec![],
        filter: None,
    }
}

fn default_name(e: &ast::Expr) -> String {
    match e {
        ast::Expr::Identifier(parts) => parts.last().cloned().unwrap_or_default(),
        other => other.to_string(),
    }
}

fn column_name_of(e: &Expr) -> String {
    match e {
        Expr::Column(c) => c.name.rsplit('.').next().unwrap_or(&c.name).to_string(),
        other => other.to_string(),
    }
}

fn bind_literal(l: &ast::Literal) -> Result<Expr> {
    Ok(Expr::Literal(match l {
        ast::Literal::Integer(i) => ScalarValue::Int64(Some(*i)),
        ast::Literal::Float(f) => ScalarValue::Float64(Some(*f)),
        ast::Literal::String(s) => ScalarValue::Utf8(Some(s.clone())),
        ast::Literal::Boolean(b) => ScalarValue::Boolean(Some(*b)),
        ast::Literal::Null => ScalarValue::Null,
        ast::Literal::Date(s) => match parse_date(s) {
            Some(d) => ScalarValue::Date32(Some(d)),
            None => return type_err(format!("invalid date literal '{s}' (expected YYYY-MM-DD)")),
        },
        ast::Literal::Interval { .. } => {
            return type_err("INTERVAL literals are only supported in date +/- interval")
        }
    }))
}

/// The common supertype two values can be coerced to, if any.
pub fn common_type(a: DataType, b: DataType) -> Option<DataType> {
    use DataType::*;
    Some(match (a, b) {
        (x, y) if x == y => x,
        (Null, x) | (x, Null) => x,
        (Int64, Float64) | (Float64, Int64) => Float64,
        (Utf8, Date32) | (Date32, Utf8) => Date32,
        _ => return None,
    })
}

/// Implicit coercion. Literals are converted eagerly so plans stay readable.
fn cast_to(e: Expr, to: DataType) -> Result<Expr> {
    let from = e.data_type();
    if from == to || to == DataType::Null {
        return Ok(e);
    }
    if let Expr::Literal(v) = &e {
        let converted = match (v, to) {
            (v, t) if v.is_null() => Some(ScalarValue::null_of(t)),
            (ScalarValue::Int64(Some(i)), DataType::Float64) => {
                Some(ScalarValue::Float64(Some(*i as f64)))
            }
            (ScalarValue::Utf8(Some(s)), DataType::Date32) => match parse_date(s) {
                Some(d) => Some(ScalarValue::Date32(Some(d))),
                None => return type_err(format!("cannot interpret '{s}' as a DATE")),
            },
            _ => None,
        };
        if let Some(c) = converted {
            return Ok(Expr::Literal(c));
        }
    }
    Ok(Expr::Cast {
        expr: Box::new(e),
        to,
    })
}

fn explicit_cast(e: Expr, to: DataType) -> Result<Expr> {
    let from = e.data_type();
    if from == DataType::Boolean && to == DataType::Date32
        || from == DataType::Date32 && to.is_numeric()
    {
        return type_err(format!("cannot cast {from} to {to}"));
    }
    cast_to(e, to)
}

fn comparison(l: Expr, op: BinaryOp, r: Expr) -> Result<Expr> {
    let (lt, rt) = (l.data_type(), r.data_type());
    let t = common_type(lt, rt)
        .ok_or_else(|| QuarryError::Type(format!("cannot compare {lt} with {rt}")))?;
    let t = if t == DataType::Null {
        DataType::Int64
    } else {
        t
    };
    Ok(Expr::binary(cast_to(l, t)?, op, cast_to(r, t)?))
}

fn expect_boolean(e: Expr, ctx: &str) -> Result<Expr> {
    expect_type(e, DataType::Boolean, ctx)
}

fn expect_type(e: Expr, t: DataType, ctx: &str) -> Result<Expr> {
    match e.data_type() {
        x if x == t => Ok(e),
        DataType::Null => cast_to(e, t),
        // Strings are accepted where a DATE is expected (and parsed).
        DataType::Utf8 if t == DataType::Date32 => cast_to(e, t),
        other => type_err(format!("{ctx} expects {t}, got {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{StatsCollector, TableEntry, TableSource};
    use crate::sql::{ast::Statement, parse_statement};
    use arrow::datatypes::{DataType as ArrowType, Field, Schema};
    use std::sync::Arc;

    fn catalog() -> Catalog {
        let mut c = Catalog::new();
        for (name, cols) in [
            (
                "t",
                vec![
                    ("a", ArrowType::Int64),
                    ("b", ArrowType::Float64),
                    ("s", ArrowType::Utf8),
                    ("d", ArrowType::Date32),
                ],
            ),
            ("u", vec![("a", ArrowType::Int64), ("c", ArrowType::Utf8)]),
        ] {
            let schema = Arc::new(Schema::new(
                cols.into_iter()
                    .map(|(n, t)| Field::new(n, t, true))
                    .collect::<Vec<_>>(),
            ));
            c.insert(TableEntry {
                name: name.into(),
                schema: schema.clone(),
                source: TableSource::Memory(vec![]),
                stats: StatsCollector::new(&schema).finish(),
            });
        }
        c
    }

    fn bind(sql: &str) -> Result<LogicalPlan> {
        let cat = catalog();
        let Statement::Query(q) = parse_statement(sql)? else {
            panic!("not a query")
        };
        Binder::new(&cat).bind(&q)
    }

    fn plan(sql: &str) -> String {
        bind(sql).unwrap().to_string()
    }

    fn err(sql: &str) -> String {
        bind(sql).unwrap_err().to_string()
    }

    #[test]
    fn simple_select_shape() {
        assert_eq!(
            plan("SELECT a, b * 2 AS bb FROM t WHERE s = 'x' ORDER BY bb DESC LIMIT 3"),
            "Limit: skip=0 fetch=3\n  Sort: bb DESC\n    Projection: a, b * 2.0 AS bb\n      Filter: s = 'x'\n        Scan: t [a, b, s, d]\n"
        );
    }

    #[test]
    fn coercions() {
        let p = plan("SELECT a + b, a / 2, d > '1995-01-01', a IN (1, 2.5) FROM t");
        assert!(p.contains("CAST(a AS DOUBLE) + b"), "{p}");
        assert!(p.contains("a / 2"), "{p}");
        assert!(p.contains("d > DATE '1995-01-01'"), "{p}");
        assert!(p.contains("CAST(a AS DOUBLE) IN (1.0, 2.5)"), "{p}");
        let p = plan("SELECT d + INTERVAL '1' YEAR, d - INTERVAL '3' DAY FROM t");
        assert!(p.contains("date_add[12mo,0d](d)"), "{p}");
        assert!(p.contains("date_add[0mo,-3d](d)"), "{p}");
    }

    #[test]
    fn aggregates_and_grouping() {
        let p = plan("SELECT s, sum(a) AS total, count(*) FROM t GROUP BY s HAVING sum(a) > 1 ORDER BY count(*)");
        assert!(
            p.contains("Aggregate: group=[s] aggs=[sum(a) AS sum(a), count(*) AS count(*)]"),
            "{p}"
        );
        assert!(p.contains("Filter: sum(a) > 1"), "{p}");
        assert!(p.contains("Sort: count(*) ASC"), "{p}");
        // GROUP BY ordinal and alias.
        assert!(plan("SELECT s AS k, count(*) FROM t GROUP BY 1").contains("group=[s]"));
        assert!(
            plan("SELECT upper(s) AS k, count(*) FROM t GROUP BY k").contains("group=[upper(s)]")
        );
        // Expressions over group keys are allowed.
        assert!(bind("SELECT s || 'x', count(*) FROM t GROUP BY s").is_ok());
        // Global aggregate.
        assert!(plan("SELECT count(*), avg(a) FROM t")
            .contains("group=[] aggs=[count(*) AS count(*), avg(CAST(a AS DOUBLE))"));
    }

    #[test]
    fn hidden_order_by_columns_are_projected_away() {
        let p = plan("SELECT a FROM t ORDER BY b");
        assert!(
            p.starts_with("Projection: a\n  Sort: __order0 ASC\n    Projection: a, b AS __order0"),
            "{p}"
        );
    }

    #[test]
    fn joins_ctes_and_derived_tables() {
        let p = plan("SELECT t.a, u.c FROM t JOIN u ON t.a = u.a LEFT JOIN u AS u2 ON u2.a = t.a");
        assert!(p.contains("LeftJoin: filter=u2.a = t.a"), "{p}");
        assert!(p.contains("InnerJoin: filter=t.a = u.a"), "{p}");
        let p = plan("WITH x AS (SELECT a FROM t), y AS (SELECT a FROM x) SELECT * FROM x, y AS y2 WHERE x.a = y2.a");
        assert_eq!(p.matches("Scan: t").count(), 2, "{p}");
        assert!(bind("SELECT k FROM (SELECT a FROM t) AS d (k)").is_ok());
        assert!(bind("SELECT d.* FROM (SELECT a, s FROM t) d").is_ok());
        assert!(bind("SELECT 1 + 1").is_ok());
    }

    #[test]
    fn distinct_becomes_group_by() {
        let p = plan("SELECT DISTINCT s FROM t ORDER BY s");
        assert!(p.contains("Aggregate: group=[s] aggs=[]"), "{p}");
        assert!(err("SELECT DISTINCT s FROM t ORDER BY a").contains("SELECT DISTINCT"));
    }

    #[test]
    fn bind_errors() {
        assert!(err("SELECT x FROM t").contains("column 'x' not found"));
        assert!(err("SELECT a FROM t, u").contains("ambiguous"));
        assert!(err("SELECT * FROM nope").contains("does not exist"));
        assert!(err("SELECT a, count(*) FROM t").contains("GROUP BY"));
        assert!(err("SELECT * FROM t WHERE sum(a) > 1").contains("WHERE"));
        assert!(err("SELECT sum(count(a)) FROM t").contains("nested"));
        assert!(err("SELECT * FROM t, t").contains("more than once"));
        assert!(err("SELECT a FROM t ORDER BY 5").contains("position 5"));
        assert!(err("SELECT a FROM t GROUP BY 9").contains("position 9"));
        assert!(err("SELECT q.a FROM t").contains("unknown table or alias 'q'"));
        assert!(err("SELECT nosuch(a) FROM t").contains("unknown function"));
        assert!(err("SELECT * FROM (SELECT a FROM t) d (x, y)").contains("column names"));
        assert!(err("WITH c AS (SELECT * FROM c) SELECT * FROM c").contains("does not exist"));
        assert!(err("SELECT *").contains("no tables"));
        assert!(err("SELECT abs(a) FROM t GROUP BY abs(a) ORDER BY b").contains("GROUP BY"));
    }

    #[test]
    fn type_errors() {
        assert!(err("SELECT a + s FROM t").contains("not defined"));
        assert!(err("SELECT * FROM t WHERE a").contains("WHERE expects BOOLEAN"));
        assert!(err("SELECT * FROM t WHERE s = 1").contains("cannot compare"));
        assert!(err("SELECT sum(s) FROM t").contains("numeric"));
        assert!(err("SELECT a LIKE 'x' FROM t").contains("LIKE"));
        assert!(err("SELECT DATE '1995-02-30'").contains("invalid date"));
        assert!(err("SELECT INTERVAL '1' DAY").contains("INTERVAL"));
        assert!(err("SELECT CASE WHEN a > 1 THEN 's' ELSE 1 END FROM t").contains("incompatible"));
        assert!(err("SELECT -s FROM t").contains("negate"));
        assert!(err("SELECT CAST(d AS BIGINT) FROM t").contains("cannot cast"));
    }
}
