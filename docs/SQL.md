# Supported SQL

The parser (`src/sql/parser.rs`) accepts exactly this grammar. Keywords and
unquoted identifiers are case-insensitive (folded to lower case);
`"Quoted"` identifiers keep their case. `--` and `/* */` comments are allowed.

```ebnf
statement     = [ "EXPLAIN" [ "ANALYZE" ] [ "VERBOSE" ] ] query [ ";" ] ;
query         = [ "WITH" cte { "," cte } ] select
                [ "ORDER BY" order_item { "," order_item } ]
                [ "LIMIT" integer ] [ "OFFSET" integer ] ;
cte           = name [ "(" name { "," name } ")" ] "AS" "(" query ")" ;
select        = "SELECT" [ "DISTINCT" | "ALL" ] select_item { "," select_item }
                [ "FROM" from_item { "," from_item } ]
                [ "WHERE" expr ]
                [ "GROUP BY" expr { "," expr } ]
                [ "HAVING" expr ] ;
select_item   = "*" | name "." "*" | expr [ [ "AS" ] name ] ;
from_item     = factor { join } ;
join          = ( [ "INNER" ] "JOIN" | "LEFT" [ "OUTER" ] "JOIN" ) factor "ON" expr
              | "CROSS JOIN" factor ;
factor        = name [ [ "AS" ] name ]
              | "(" query ")" [ "AS" ] name [ "(" name { "," name } ")" ] ;
order_item    = expr [ "ASC" | "DESC" ] [ "NULLS" ( "FIRST" | "LAST" ) ] ;

expr          = literal | column | "(" expr ")"
              | "-" expr | "+" expr | "NOT" expr
              | expr binop expr
              | expr "IS" [ "NOT" ] "NULL"
              | expr [ "NOT" ] "BETWEEN" expr "AND" expr
              | expr [ "NOT" ] "IN" "(" expr { "," expr } ")"
              | expr [ "NOT" ] "LIKE" expr
              | "CASE" [ expr ] ( "WHEN" expr "THEN" expr )+ [ "ELSE" expr ] "END"
              | "CAST" "(" expr "AS" type ")"
              | "EXTRACT" "(" ( "YEAR" | "MONTH" | "DAY" ) "FROM" expr ")"
              | function "(" [ "DISTINCT" ] [ expr { "," expr } ] ")"
              | "count" "(" "*" ")"
              | "substring" "(" expr "FROM" expr [ "FOR" expr ] ")" ;
binop         = "OR" | "AND" | "=" | "==" | "<>" | "!=" | "<" | "<=" | ">" | ">="
              | "||" | "+" | "-" | "*" | "/" | "%" ;
column        = name | name "." name ;
literal       = integer | decimal | "'string'" | "TRUE" | "FALSE" | "NULL"
              | "DATE" "'YYYY-MM-DD'" | "INTERVAL" "'n'" ( "DAY" | "MONTH" | "YEAR" ) ;
type          = "BOOLEAN" | "BIGINT" | "INT" | "INTEGER" | "DOUBLE" [ "PRECISION" ]
              | "FLOAT" | "REAL" | "DECIMAL" [ "(" p [ "," s ] ")" ]
              | "VARCHAR" [ "(" n ")" ] | "TEXT" | "DATE" ;
```

Operator precedence, loosest first: `OR`; `AND`; `NOT`; comparisons, `IS`,
`LIKE`, `IN`, `BETWEEN`; `||`; `+ -`; `* / %`; unary minus.

## Types

| SQL type | Arrow storage | Notes |
|---|---|---|
| BOOLEAN | Boolean | |
| BIGINT (INT, INTEGER) | Int64 | all integer source columns are widened to Int64 |
| DOUBLE (FLOAT, REAL, DECIMAL) | Float64 | DECIMAL is approximated |
| VARCHAR (TEXT) | Utf8 | |
| DATE | Date32 | `'YYYY-MM-DD'` strings compared with dates are coerced |

Implicit coercions: BIGINT with DOUBLE to DOUBLE; VARCHAR literal with DATE to
DATE; untyped `NULL` to the other operand's type.

## Functions

| Function | Result |
|---|---|
| `count(*)`, `count([DISTINCT] x)` | BIGINT, never NULL |
| `sum([DISTINCT] x)` | BIGINT for BIGINT input, else DOUBLE; NULL on no input |
| `avg(x)` | DOUBLE |
| `min(x)`, `max(x)` | input type (numbers, strings, dates, booleans) |
| `abs(x)` | input type |
| `round(x [, digits])` | DOUBLE, half away from zero |
| `lower(s)`, `upper(s)` | VARCHAR |
| `length(s)`, `char_length(s)` | BIGINT, in characters |
| `substr(s, start [, len])`, `substring(s FROM start [FOR len])` | VARCHAR, 1-based |
| `coalesce(a, b, ...)` | common type of the arguments |
| `EXTRACT(YEAR\|MONTH\|DAY FROM d)` | BIGINT |
| `d + INTERVAL 'n' DAY\|MONTH\|YEAR`, `d - INTERVAL ...` | DATE (month arithmetic clamps to month end) |

## NULL and arithmetic semantics

- Three-valued logic everywhere (`NULL AND FALSE` is FALSE, `NULL OR TRUE` is TRUE).
- `x / 0` and `x % 0` are NULL. Integer division truncates toward zero.
- Integer overflow is an error. Float operations follow IEEE 754.
- NULLs sort first in ascending order and last in descending order.
- `LIKE` is case-sensitive; `%` matches any run, `_` one character.

## Not supported

Rejected with an error (an `unsupported:` error naming the feature for the
subquery, set-operation and join forms; a syntax or bind error otherwise):
`IN (SELECT ...)`, `EXISTS`, scalar
subqueries, correlated subqueries, `UNION/EXCEPT/INTERSECT`, `RIGHT/FULL/NATURAL
JOIN`, `JOIN ... USING`, `WITH RECURSIVE`, window functions, DDL and DML.
Parenthesized joins in FROM are not accepted (only parenthesized subqueries).
