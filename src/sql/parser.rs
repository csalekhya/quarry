//! Recursive-descent parser for the documented SQL subset.
//!
//! Statements and clauses are parsed by plain recursive descent; expressions
//! use precedence climbing (a Pratt parser) with these binding powers, lowest
//! first:
//!
//! | bp | operators                                        |
//! |----|--------------------------------------------------|
//! | 1  | `OR`                                             |
//! | 2  | `AND`                                            |
//! | 3  | prefix `NOT`                                     |
//! | 4  | `= <> < <= > >=`, `IS [NOT] NULL`, `[NOT] LIKE/IN/BETWEEN` |
//! | 5  | `\|\|`                                           |
//! | 6  | `+ -`                                            |
//! | 7  | `* / %`                                          |
//! | 8  | prefix `-` / `+`                                 |
//!
//! The grammar is listed in `docs/SQL.md`.

use super::ast::*;
use super::lexer::{tokenize, Spanned, Token};
use crate::error::{QuarryError, Result};

/// Words that may never be used as an implicit alias or bare column name.
const RESERVED: &[&str] = &[
    "select",
    "from",
    "where",
    "group",
    "having",
    "order",
    "limit",
    "offset",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "outer",
    "cross",
    "on",
    "as",
    "and",
    "or",
    "not",
    "union",
    "with",
    "by",
    "asc",
    "desc",
    "nulls",
    "is",
    "in",
    "like",
    "between",
    "case",
    "when",
    "then",
    "else",
    "end",
    "distinct",
    "null",
    "true",
    "false",
    "cast",
    "extract",
    "interval",
    "exists",
    "all",
    "using",
    "natural",
    "except",
    "intersect",
];

const BP_OR: u8 = 1;
const BP_AND: u8 = 2;
const BP_NOT: u8 = 3;
const BP_CMP: u8 = 4;
const BP_CONCAT: u8 = 5;
const BP_ADD: u8 = 6;
const BP_MUL: u8 = 7;
const BP_UNARY: u8 = 8;

/// Parse exactly one statement (a trailing `;` is allowed).
pub fn parse_statement(sql: &str) -> Result<Statement> {
    let mut p = Parser {
        tokens: tokenize(sql)?,
        pos: 0,
    };
    let stmt = p.statement()?;
    p.consume(&Token::Semicolon);
    if !p.at(&Token::Eof) {
        return p.unexpected("end of statement");
    }
    Ok(stmt)
}

/// Parse a standalone expression (used by tests and tooling).
pub fn parse_expression(sql: &str) -> Result<Expr> {
    let mut p = Parser {
        tokens: tokenize(sql)?,
        pos: 0,
    };
    let e = p.expr()?;
    if !p.at(&Token::Eof) {
        return p.unexpected("end of expression");
    }
    Ok(e)
}

struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    // ----- token helpers -------------------------------------------------

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn peek_nth(&self, n: usize) -> &Token {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        &self.tokens[i].token
    }

    fn advance(&mut self) -> Token {
        let t = self.tokens[self.pos].token.clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn at(&self, t: &Token) -> bool {
        self.peek() == t
    }

    fn consume(&mut self, t: &Token) -> bool {
        if self.at(t) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, t: &Token) -> Result<()> {
        if self.consume(t) {
            Ok(())
        } else {
            self.unexpected(&t.describe())
        }
    }

    fn is_kw_at(&self, n: usize, kw: &str) -> bool {
        matches!(self.peek_nth(n), Token::Ident { value, quoted: false } if value == kw)
    }

    fn is_kw(&self, kw: &str) -> bool {
        self.is_kw_at(0, kw)
    }

    fn consume_kw(&mut self, kw: &str) -> bool {
        if self.is_kw(kw) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        if self.consume_kw(kw) {
            Ok(())
        } else {
            self.unexpected(&kw.to_uppercase())
        }
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T> {
        let sp = &self.tokens[self.pos];
        Err(QuarryError::Parse {
            message: message.into(),
            line: sp.line,
            column: sp.column,
        })
    }

    fn unexpected<T>(&self, expected: &str) -> Result<T> {
        self.error(format!(
            "expected {expected}, found {}",
            self.peek().describe()
        ))
    }

    /// An identifier that is not a reserved word (quoted identifiers always qualify).
    fn identifier(&mut self) -> Result<String> {
        match self.peek().clone() {
            Token::Ident { value, quoted } if quoted || !RESERVED.contains(&value.as_str()) => {
                self.advance();
                Ok(value)
            }
            _ => self.unexpected("identifier"),
        }
    }

    fn at_plain_identifier(&self) -> bool {
        matches!(self.peek(), Token::Ident { value, quoted } if *quoted || !RESERVED.contains(&value.as_str()))
    }

    fn unsigned_integer(&mut self, what: &str) -> Result<u64> {
        match self.peek().clone() {
            Token::Number(n) => match n.parse::<u64>() {
                Ok(v) => {
                    self.advance();
                    Ok(v)
                }
                Err(_) => self.error(format!("{what} must be a non-negative integer, found {n}")),
            },
            _ => self.unexpected(what),
        }
    }

    // ----- statements ----------------------------------------------------

    fn statement(&mut self) -> Result<Statement> {
        if self.consume_kw("explain") {
            let analyze = self.consume_kw("analyze");
            let verbose = self.consume_kw("verbose");
            let query = Box::new(self.query()?);
            return Ok(Statement::Explain {
                analyze,
                verbose,
                query,
            });
        }
        if !self.is_kw("select") && !self.is_kw("with") {
            return self.unexpected("SELECT, WITH or EXPLAIN");
        }
        Ok(Statement::Query(Box::new(self.query()?)))
    }

    fn query(&mut self) -> Result<Query> {
        let mut ctes = Vec::new();
        if self.consume_kw("with") {
            if self.is_kw("recursive") {
                return Err(QuarryError::Unsupported("WITH RECURSIVE".into()));
            }
            loop {
                let name = self.identifier()?;
                let columns = if self.at(&Token::LParen) {
                    self.column_list()?
                } else {
                    Vec::new()
                };
                self.expect_kw("as")?;
                self.expect(&Token::LParen)?;
                let query = Box::new(self.query()?);
                self.expect(&Token::RParen)?;
                ctes.push(Cte {
                    name,
                    columns,
                    query,
                });
                if !self.consume(&Token::Comma) {
                    break;
                }
            }
        }
        let body = self.select()?;
        if self.is_kw("union") || self.is_kw("except") || self.is_kw("intersect") {
            return Err(QuarryError::Unsupported(
                "set operations (UNION/EXCEPT/INTERSECT)".into(),
            ));
        }
        let mut order_by = Vec::new();
        if self.consume_kw("order") {
            self.expect_kw("by")?;
            loop {
                order_by.push(self.order_by_expr()?);
                if !self.consume(&Token::Comma) {
                    break;
                }
            }
        }
        let mut limit = None;
        let mut offset = None;
        if self.consume_kw("limit") {
            limit = Some(self.unsigned_integer("LIMIT count")?);
        }
        if self.consume_kw("offset") {
            offset = Some(self.unsigned_integer("OFFSET count")?);
        }
        Ok(Query {
            ctes,
            body,
            order_by,
            limit,
            offset,
        })
    }

    fn column_list(&mut self) -> Result<Vec<String>> {
        self.expect(&Token::LParen)?;
        let mut cols = vec![self.identifier()?];
        while self.consume(&Token::Comma) {
            cols.push(self.identifier()?);
        }
        self.expect(&Token::RParen)?;
        Ok(cols)
    }

    fn order_by_expr(&mut self) -> Result<OrderByExpr> {
        let expr = self.expr()?;
        let asc = if self.consume_kw("desc") {
            false
        } else {
            self.consume_kw("asc");
            true
        };
        let nulls_first = if self.consume_kw("nulls") {
            if self.consume_kw("first") {
                Some(true)
            } else if self.consume_kw("last") {
                Some(false)
            } else {
                return self.unexpected("FIRST or LAST");
            }
        } else {
            None
        };
        Ok(OrderByExpr {
            expr,
            asc,
            nulls_first,
        })
    }

    fn select(&mut self) -> Result<Select> {
        self.expect_kw("select")?;
        let distinct = if self.consume_kw("distinct") {
            true
        } else {
            self.consume_kw("all");
            false
        };
        let mut projection = vec![self.select_item()?];
        while self.consume(&Token::Comma) {
            projection.push(self.select_item()?);
        }
        let mut from = Vec::new();
        if self.consume_kw("from") {
            loop {
                from.push(self.table_with_joins()?);
                if !self.consume(&Token::Comma) {
                    break;
                }
            }
        }
        let selection = if self.consume_kw("where") {
            Some(self.expr()?)
        } else {
            None
        };
        let mut group_by = Vec::new();
        if self.consume_kw("group") {
            self.expect_kw("by")?;
            loop {
                group_by.push(self.expr()?);
                if !self.consume(&Token::Comma) {
                    break;
                }
            }
        }
        let having = if self.consume_kw("having") {
            Some(self.expr()?)
        } else {
            None
        };
        Ok(Select {
            distinct,
            projection,
            from,
            selection,
            group_by,
            having,
        })
    }

    fn select_item(&mut self) -> Result<SelectItem> {
        if self.consume(&Token::Star) {
            return Ok(SelectItem::Wildcard);
        }
        if self.at_plain_identifier()
            && self.peek_nth(1) == &Token::Dot
            && self.peek_nth(2) == &Token::Star
        {
            let q = self.identifier()?;
            self.advance();
            self.advance();
            return Ok(SelectItem::QualifiedWildcard(q));
        }
        let expr = self.expr()?;
        let alias = self.optional_alias()?;
        Ok(SelectItem::Expr { expr, alias })
    }

    fn optional_alias(&mut self) -> Result<Option<String>> {
        if self.consume_kw("as") {
            return Ok(Some(self.identifier()?));
        }
        if self.at_plain_identifier() {
            return Ok(Some(self.identifier()?));
        }
        Ok(None)
    }

    fn table_with_joins(&mut self) -> Result<TableWithJoins> {
        let relation = self.table_factor()?;
        let mut joins = Vec::new();
        loop {
            let kind = if self.consume_kw("cross") {
                self.expect_kw("join")?;
                JoinKind::Cross
            } else if self.consume_kw("left") {
                self.consume_kw("outer");
                self.expect_kw("join")?;
                JoinKind::Left
            } else if self.consume_kw("inner") {
                self.expect_kw("join")?;
                JoinKind::Inner
            } else if self.consume_kw("join") {
                JoinKind::Inner
            } else if self.is_kw("right") || self.is_kw("full") || self.is_kw("natural") {
                return Err(QuarryError::Unsupported(format!(
                    "{} JOIN",
                    self.peek().describe().to_uppercase()
                )));
            } else {
                break;
            };
            let relation = self.table_factor()?;
            let on = if kind == JoinKind::Cross {
                None
            } else if self.consume_kw("on") {
                Some(self.expr()?)
            } else if self.is_kw("using") {
                return Err(QuarryError::Unsupported("JOIN ... USING".into()));
            } else {
                return self.unexpected("ON");
            };
            joins.push(Join { relation, kind, on });
        }
        Ok(TableWithJoins { relation, joins })
    }

    fn table_factor(&mut self) -> Result<TableFactor> {
        if self.consume(&Token::LParen) {
            if !(self.is_kw("select") || self.is_kw("with")) {
                return Err(QuarryError::Unsupported(
                    "parenthesized joins in FROM (only subqueries may be parenthesized)".into(),
                ));
            }
            let subquery = Box::new(self.query()?);
            self.expect(&Token::RParen)?;
            self.consume_kw("as");
            let alias = match self.identifier() {
                Ok(a) => a,
                Err(_) => return self.error("a subquery in FROM must have an alias"),
            };
            let columns = if self.at(&Token::LParen) {
                self.column_list()?
            } else {
                Vec::new()
            };
            return Ok(TableFactor::Derived {
                subquery,
                alias,
                columns,
            });
        }
        let name = self.identifier()?;
        let alias = self.optional_alias()?;
        Ok(TableFactor::Table { name, alias })
    }

    // ----- expressions ---------------------------------------------------

    fn expr(&mut self) -> Result<Expr> {
        self.expr_bp(0)
    }

    fn expr_bp(&mut self, min_bp: u8) -> Result<Expr> {
        let mut lhs = self.prefix()?;
        loop {
            // Postfix / special infix forms at comparison precedence.
            let negated_form = self.is_kw("not")
                && (self.is_kw_at(1, "between")
                    || self.is_kw_at(1, "in")
                    || self.is_kw_at(1, "like"));
            if (negated_form
                || self.is_kw("is")
                || self.is_kw("between")
                || self.is_kw("in")
                || self.is_kw("like"))
                && BP_CMP > min_bp
            {
                lhs = self.comparison_postfix(lhs)?;
                continue;
            }
            let (op, bp) = match self.peek() {
                Token::Ident {
                    value,
                    quoted: false,
                } if value == "or" => (BinaryOp::Or, BP_OR),
                Token::Ident {
                    value,
                    quoted: false,
                } if value == "and" => (BinaryOp::And, BP_AND),
                Token::Eq => (BinaryOp::Eq, BP_CMP),
                Token::NotEq => (BinaryOp::NotEq, BP_CMP),
                Token::Lt => (BinaryOp::Lt, BP_CMP),
                Token::LtEq => (BinaryOp::LtEq, BP_CMP),
                Token::Gt => (BinaryOp::Gt, BP_CMP),
                Token::GtEq => (BinaryOp::GtEq, BP_CMP),
                Token::Concat => (BinaryOp::Concat, BP_CONCAT),
                Token::Plus => (BinaryOp::Plus, BP_ADD),
                Token::Minus => (BinaryOp::Minus, BP_ADD),
                Token::Star => (BinaryOp::Multiply, BP_MUL),
                Token::Slash => (BinaryOp::Divide, BP_MUL),
                Token::Percent => (BinaryOp::Modulo, BP_MUL),
                _ => break,
            };
            if bp <= min_bp {
                break;
            }
            self.advance();
            let rhs = self.expr_bp(bp)?;
            lhs = Expr::Binary {
                left: Box::new(lhs),
                op,
                right: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn comparison_postfix(&mut self, lhs: Expr) -> Result<Expr> {
        if self.consume_kw("is") {
            let negated = self.consume_kw("not");
            self.expect_kw("null")?;
            return Ok(Expr::IsNull {
                expr: Box::new(lhs),
                negated,
            });
        }
        let negated = self.consume_kw("not");
        if self.consume_kw("between") {
            let low = self.expr_bp(BP_CMP)?;
            self.expect_kw("and")?;
            let high = self.expr_bp(BP_CMP)?;
            return Ok(Expr::Between {
                expr: Box::new(lhs),
                low: Box::new(low),
                high: Box::new(high),
                negated,
            });
        }
        if self.consume_kw("in") {
            self.expect(&Token::LParen)?;
            if self.is_kw("select") || self.is_kw("with") {
                return Err(QuarryError::Unsupported(
                    "IN (subquery); rewrite it as a join against a derived table".into(),
                ));
            }
            let mut list = vec![self.expr()?];
            while self.consume(&Token::Comma) {
                list.push(self.expr()?);
            }
            self.expect(&Token::RParen)?;
            return Ok(Expr::InList {
                expr: Box::new(lhs),
                list,
                negated,
            });
        }
        self.expect_kw("like")?;
        let pattern = self.expr_bp(BP_CMP)?;
        Ok(Expr::Like {
            expr: Box::new(lhs),
            pattern: Box::new(pattern),
            negated,
        })
    }

    fn prefix(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Token::Number(n) => {
                self.advance();
                number_literal(&n)
                    .map(Expr::Literal)
                    .or_else(|m| self.error(m))
            }
            Token::Str(s) => {
                self.advance();
                Ok(Expr::Literal(Literal::String(s)))
            }
            Token::LParen => {
                self.advance();
                if self.is_kw("select") || self.is_kw("with") {
                    return Err(QuarryError::Unsupported(
                        "scalar subqueries; use a derived table in FROM instead".into(),
                    ));
                }
                let e = self.expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            Token::Minus => {
                self.advance();
                let e = self.expr_bp(BP_UNARY)?;
                Ok(match e {
                    Expr::Literal(Literal::Integer(i)) => Expr::Literal(Literal::Integer(-i)),
                    Expr::Literal(Literal::Float(f)) => Expr::Literal(Literal::Float(-f)),
                    other => Expr::Unary {
                        op: UnaryOp::Neg,
                        expr: Box::new(other),
                    },
                })
            }
            Token::Plus => {
                self.advance();
                self.expr_bp(BP_UNARY)
            }
            Token::Ident {
                value,
                quoted: false,
            } => self.keyword_or_identifier(&value),
            Token::Ident { .. } => self.identifier_expr(),
            _ => self.unexpected("an expression"),
        }
    }

    fn keyword_or_identifier(&mut self, word: &str) -> Result<Expr> {
        match word {
            "null" => {
                self.advance();
                Ok(Expr::Literal(Literal::Null))
            }
            "true" | "false" => {
                self.advance();
                Ok(Expr::Literal(Literal::Boolean(word == "true")))
            }
            "not" => {
                self.advance();
                let e = self.expr_bp(BP_NOT)?;
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(e),
                })
            }
            "case" => self.case_expr(),
            "cast" => {
                self.advance();
                self.expect(&Token::LParen)?;
                let e = self.expr()?;
                self.expect_kw("as")?;
                let data_type = self.type_name()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Cast {
                    expr: Box::new(e),
                    data_type,
                })
            }
            "extract" => {
                self.advance();
                self.expect(&Token::LParen)?;
                let field = if self.consume_kw("year") {
                    DateField::Year
                } else if self.consume_kw("month") {
                    DateField::Month
                } else if self.consume_kw("day") {
                    DateField::Day
                } else {
                    return self.unexpected("YEAR, MONTH or DAY");
                };
                self.expect_kw("from")?;
                let e = self.expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Extract {
                    field,
                    expr: Box::new(e),
                })
            }
            "date" if matches!(self.peek_nth(1), Token::Str(_)) => {
                self.advance();
                let Token::Str(s) = self.advance() else {
                    unreachable!("peeked a string literal")
                };
                Ok(Expr::Literal(Literal::Date(s)))
            }
            "interval" => self.interval(),
            "exists" => Err(QuarryError::Unsupported(
                "EXISTS subqueries; rewrite as a join".into(),
            )),
            w if RESERVED.contains(&w) => self.unexpected("an expression"),
            _ => self.identifier_expr(),
        }
    }

    fn interval(&mut self) -> Result<Expr> {
        self.advance();
        let text = match self.advance() {
            Token::Str(s) => s,
            Token::Number(n) => n,
            _ => return self.error("expected interval value such as INTERVAL '3' MONTH"),
        };
        // Accept both INTERVAL '3' MONTH and INTERVAL '3 month'.
        let mut parts = text.split_whitespace();
        let value: i64 = match parts.next().and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => return self.error(format!("invalid interval value '{text}'")),
        };
        let unit_word = match parts.next() {
            Some(u) => u.to_ascii_lowercase(),
            None => match self.advance() {
                Token::Ident {
                    value,
                    quoted: false,
                } => value,
                _ => return self.error("expected interval unit DAY, MONTH or YEAR"),
            },
        };
        let unit = match unit_word.trim_end_matches('s') {
            "day" => IntervalUnit::Day,
            "month" => IntervalUnit::Month,
            "year" => IntervalUnit::Year,
            other => return self.error(format!("unsupported interval unit '{other}'")),
        };
        Ok(Expr::Literal(Literal::Interval { value, unit }))
    }

    fn case_expr(&mut self) -> Result<Expr> {
        self.advance();
        let operand = if self.is_kw("when") {
            None
        } else {
            Some(Box::new(self.expr()?))
        };
        let mut branches = Vec::new();
        while self.consume_kw("when") {
            let w = self.expr()?;
            self.expect_kw("then")?;
            let t = self.expr()?;
            branches.push((w, t));
        }
        if branches.is_empty() {
            return self.unexpected("WHEN");
        }
        let else_expr = if self.consume_kw("else") {
            Some(Box::new(self.expr()?))
        } else {
            None
        };
        self.expect_kw("end")?;
        Ok(Expr::Case {
            operand,
            branches,
            else_expr,
        })
    }

    fn type_name(&mut self) -> Result<TypeName> {
        let word = match self.advance() {
            Token::Ident {
                value,
                quoted: false,
            } => value,
            _ => return self.error("expected a type name"),
        };
        let t = match word.as_str() {
            "boolean" | "bool" => TypeName::Boolean,
            "bigint" | "int" | "integer" | "smallint" | "tinyint" | "int8" | "int4" => {
                TypeName::BigInt
            }
            "double" => {
                self.consume_kw("precision");
                TypeName::Double
            }
            "float" | "real" | "decimal" | "numeric" => TypeName::Double,
            "varchar" | "text" | "string" | "char" => TypeName::Varchar,
            "date" => TypeName::Date,
            other => return Err(QuarryError::Unsupported(format!("type {other}"))),
        };
        // Swallow precision/length modifiers such as DECIMAL(12, 2) or VARCHAR(25).
        if self.consume(&Token::LParen) {
            self.unsigned_integer("type modifier")?;
            if self.consume(&Token::Comma) {
                self.unsigned_integer("type modifier")?;
            }
            self.expect(&Token::RParen)?;
        }
        Ok(t)
    }

    fn identifier_expr(&mut self) -> Result<Expr> {
        let first = self.identifier()?;
        if self.at(&Token::LParen) {
            return self.function_call(first);
        }
        let mut parts = vec![first];
        while self.at(&Token::Dot) {
            self.advance();
            parts.push(self.identifier()?);
        }
        if parts.len() > 2 {
            return self.error(format!(
                "identifier '{}' has too many parts",
                parts.join(".")
            ));
        }
        Ok(Expr::Identifier(parts))
    }

    fn function_call(&mut self, name: String) -> Result<Expr> {
        self.expect(&Token::LParen)?;
        if self.consume(&Token::Star) {
            self.expect(&Token::RParen)?;
            return Ok(Expr::Function {
                name,
                args: Vec::new(),
                distinct: false,
                star: true,
            });
        }
        let distinct = self.consume_kw("distinct");
        let mut args = Vec::new();
        if !self.at(&Token::RParen) {
            args.push(self.expr()?);
            if (name == "substring" || name == "substr") && self.consume_kw("from") {
                // SUBSTRING(s FROM start [FOR len])
                args.push(self.expr()?);
                if self.consume_kw("for") {
                    args.push(self.expr()?);
                }
            } else {
                while self.consume(&Token::Comma) {
                    args.push(self.expr()?);
                }
            }
        }
        self.expect(&Token::RParen)?;
        Ok(Expr::Function {
            name,
            args,
            distinct,
            star: false,
        })
    }
}

fn number_literal(text: &str) -> std::result::Result<Literal, String> {
    if !text.contains(['.', 'e']) {
        if let Ok(i) = text.parse::<i64>() {
            return Ok(Literal::Integer(i));
        }
    }
    text.parse::<f64>()
        .map(Literal::Float)
        .map_err(|_| format!("invalid number '{text}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(sql: &str) -> Query {
        match parse_statement(sql).unwrap() {
            Statement::Query(q) => *q,
            other => panic!("not a query: {other:?}"),
        }
    }

    fn expr(sql: &str) -> String {
        parse_expression(sql).unwrap().to_string()
    }

    #[test]
    fn precedence_climbing() {
        assert_eq!(expr("1 + 2 * 3"), "1 + 2 * 3");
        assert_eq!(expr("(1 + 2) * 3"), "(1 + 2) * 3");
        assert_eq!(expr("1 - 2 - 3"), "1 - 2 - 3");
        assert_eq!(expr("1 - (2 - 3)"), "1 - (2 - 3)");
        assert_eq!(expr("a OR b AND c"), "a OR b AND c");
        assert_eq!(expr("(a OR b) AND c"), "(a OR b) AND c");
        assert_eq!(expr("NOT a = b AND c"), "NOT a = b AND c");
        assert_eq!(expr("NOT (a AND b)"), "NOT (a AND b)");
        assert_eq!(expr("-a * b"), "-a * b");
        assert_eq!(expr("-(a * b)"), "-(a * b)");
        assert_eq!(expr("a || b = c"), "a || b = c");
        assert_eq!(expr("x BETWEEN 1 AND 2 AND y"), "x BETWEEN 1 AND 2 AND y");
        assert_eq!(
            expr("x NOT IN (1, 2) OR y IS NOT NULL"),
            "x NOT IN (1, 2) OR y IS NOT NULL"
        );
        assert_eq!(expr("s NOT LIKE 'a%'"), "s NOT LIKE 'a%'");
        assert_eq!(expr("-5"), "-5");
        assert_eq!(expr("+5"), "5");
    }

    #[test]
    fn special_forms() {
        assert_eq!(
            expr("CASE WHEN a > 1 THEN 'x' ELSE 'y' END"),
            "CASE WHEN a > 1 THEN 'x' ELSE 'y' END"
        );
        assert_eq!(expr("CASE a WHEN 1 THEN 2 END"), "CASE a WHEN 1 THEN 2 END");
        assert_eq!(expr("CAST(x AS DECIMAL(12,2))"), "CAST(x AS Double)");
        assert_eq!(expr("EXTRACT(YEAR FROM d)"), "EXTRACT(Year FROM d)");
        assert_eq!(
            expr("DATE '1995-01-01' + INTERVAL '3' MONTH"),
            "DATE '1995-01-01' + INTERVAL '3' MONTH"
        );
        assert_eq!(expr("INTERVAL '1 year'"), "INTERVAL '1' YEAR");
        assert_eq!(expr("count(DISTINCT a)"), "count(DISTINCT a)");
        assert_eq!(expr("count(*)"), "count(*)");
        assert_eq!(expr("substring(s FROM 1 FOR 2)"), "substring(s, 1, 2)");
        assert_eq!(expr("t.a"), "t.a");
        assert_eq!(expr("1.5e2"), "150");
    }

    #[test]
    fn full_query_shape() {
        let q = query(
            "WITH r(k) AS (SELECT 1) SELECT DISTINCT a AS x, b y, t.* FROM t AS t1 \
             JOIN u ON t1.a = u.a LEFT OUTER JOIN v ON u.b = v.b CROSS JOIN w, \
             (SELECT 1 AS one) AS d (o) WHERE a > 1 GROUP BY a, b HAVING count(*) > 2 \
             ORDER BY 1 DESC NULLS LAST, b LIMIT 10 OFFSET 5;",
        );
        assert_eq!(q.ctes.len(), 1);
        assert_eq!(q.ctes[0].columns, vec!["k"]);
        assert!(q.body.distinct);
        assert_eq!(q.body.projection.len(), 3);
        assert_eq!(q.body.from.len(), 2);
        let joins = &q.body.from[0].joins;
        assert_eq!(
            joins.iter().map(|j| j.kind).collect::<Vec<_>>(),
            vec![JoinKind::Inner, JoinKind::Left, JoinKind::Cross]
        );
        assert!(
            matches!(&q.body.from[1].relation, TableFactor::Derived { alias, columns, .. } if alias == "d" && columns == &vec!["o".to_string()])
        );
        assert_eq!(q.body.group_by.len(), 2);
        assert!(q.body.having.is_some());
        assert_eq!(q.order_by.len(), 2);
        assert!(!q.order_by[0].asc);
        assert_eq!(q.order_by[0].nulls_first, Some(false));
        assert_eq!((q.limit, q.offset), (Some(10), Some(5)));
    }

    #[test]
    fn explain_variants() {
        assert!(matches!(
            parse_statement("EXPLAIN ANALYZE SELECT 1").unwrap(),
            Statement::Explain {
                analyze: true,
                verbose: false,
                ..
            }
        ));
        assert!(matches!(
            parse_statement("explain verbose select 1").unwrap(),
            Statement::Explain {
                analyze: false,
                verbose: true,
                ..
            }
        ));
    }

    #[test]
    fn errors_carry_positions() {
        match parse_statement("SELECT a FROM t WHERE") {
            Err(QuarryError::Parse {
                message,
                line,
                column,
            }) => {
                assert_eq!((line, column), (1, 22));
                assert!(message.contains("expected an expression"), "{message}");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            parse_statement("SELECT 1 2"),
            Err(QuarryError::Parse { .. })
        ));
        assert!(matches!(
            parse_statement("SELECT FROM t"),
            Err(QuarryError::Parse { .. })
        ));
        assert!(matches!(
            parse_statement("UPDATE t SET a = 1"),
            Err(QuarryError::Parse { .. })
        ));
        assert!(matches!(
            parse_statement("SELECT * FROM (SELECT 1)"),
            Err(QuarryError::Parse { .. })
        ));
        assert!(matches!(
            parse_statement("SELECT * FROM t LIMIT -1"),
            Err(QuarryError::Parse { .. })
        ));
        assert!(matches!(
            parse_statement("SELECT a.b.c FROM t"),
            Err(QuarryError::Parse { .. })
        ));
    }

    #[test]
    fn unsupported_features_are_named() {
        for sql in [
            "SELECT * FROM t WHERE a IN (SELECT b FROM u)",
            "SELECT (SELECT 1)",
            "SELECT * FROM t RIGHT JOIN u ON t.a = u.a",
            "SELECT 1 UNION SELECT 2",
            "WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r",
            "SELECT * FROM t JOIN u USING (a)",
            "SELECT * FROM t WHERE EXISTS (SELECT 1)",
            "SELECT CAST(a AS BLOB) FROM t",
        ] {
            assert!(
                matches!(parse_statement(sql), Err(QuarryError::Unsupported(_))),
                "{sql}"
            );
        }
    }
}
