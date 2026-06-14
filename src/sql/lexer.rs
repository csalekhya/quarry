//! Hand-written SQL tokenizer.
//!
//! Unquoted identifiers are folded to lower case; `"quoted"` identifiers keep
//! their spelling. Keywords are not distinguished here: the parser recognises
//! them by comparing unquoted identifiers, which keeps the token set tiny.

use crate::error::{QuarryError, Result};

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// An identifier or keyword. `quoted` identifiers are never keywords.
    Ident {
        value: String,
        quoted: bool,
    },
    /// A numeric literal, kept as text so the parser can pick int vs float.
    Number(String),
    /// A single-quoted string literal with `''` escapes resolved.
    Str(String),
    Comma,
    Dot,
    LParen,
    RParen,
    Semicolon,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Concat,
    Eof,
}

impl Token {
    /// Human-readable rendering for error messages.
    pub fn describe(&self) -> String {
        match self {
            Token::Ident {
                value,
                quoted: true,
            } => format!("\"{value}\""),
            Token::Ident { value, .. } => value.clone(),
            Token::Number(n) => n.clone(),
            Token::Str(s) => format!("'{s}'"),
            Token::Comma => ",".into(),
            Token::Dot => ".".into(),
            Token::LParen => "(".into(),
            Token::RParen => ")".into(),
            Token::Semicolon => ";".into(),
            Token::Plus => "+".into(),
            Token::Minus => "-".into(),
            Token::Star => "*".into(),
            Token::Slash => "/".into(),
            Token::Percent => "%".into(),
            Token::Eq => "=".into(),
            Token::NotEq => "<>".into(),
            Token::Lt => "<".into(),
            Token::LtEq => "<=".into(),
            Token::Gt => ">".into(),
            Token::GtEq => ">=".into(),
            Token::Concat => "||".into(),
            Token::Eof => "end of input".into(),
        }
    }
}

/// A token plus the 1-based position where it starts.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned {
    pub token: Token,
    pub line: usize,
    pub column: usize,
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
}

/// Tokenize a SQL string. The result always ends with [`Token::Eof`].
pub fn tokenize(sql: &str) -> Result<Vec<Spanned>> {
    let mut lx = Lexer {
        chars: sql.chars().collect(),
        pos: 0,
        line: 1,
        column: 1,
    };
    let mut out = Vec::new();
    loop {
        lx.skip_trivia()?;
        let (line, column) = (lx.line, lx.column);
        let token = match lx.peek() {
            None => {
                out.push(Spanned {
                    token: Token::Eof,
                    line,
                    column,
                });
                return Ok(out);
            }
            Some(c) => lx.next_token(c)?,
        };
        out.push(Spanned {
            token,
            line,
            column,
        });
    }
}

impl Lexer {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(QuarryError::Parse {
            message: message.into(),
            line: self.line,
            column: self.column,
        })
    }

    fn skip_trivia(&mut self) -> Result<()> {
        loop {
            match (self.peek(), self.peek_at(1)) {
                (Some(c), _) if c.is_whitespace() => {
                    self.bump();
                }
                (Some('-'), Some('-')) => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.bump();
                    }
                }
                (Some('/'), Some('*')) => {
                    self.bump();
                    self.bump();
                    loop {
                        match (self.peek(), self.peek_at(1)) {
                            (Some('*'), Some('/')) => {
                                self.bump();
                                self.bump();
                                break;
                            }
                            (Some(_), _) => {
                                self.bump();
                            }
                            (None, _) => return self.error("unterminated block comment"),
                        }
                    }
                }
                _ => return Ok(()),
            }
        }
    }

    fn next_token(&mut self, c: char) -> Result<Token> {
        if c.is_ascii_alphabetic() || c == '_' {
            let mut s = String::new();
            while let Some(c) = self.peek() {
                if c.is_ascii_alphanumeric() || c == '_' || c == '$' {
                    s.push(c.to_ascii_lowercase());
                    self.bump();
                } else {
                    break;
                }
            }
            return Ok(Token::Ident {
                value: s,
                quoted: false,
            });
        }
        if c.is_ascii_digit() || (c == '.' && self.peek_at(1).is_some_and(|d| d.is_ascii_digit())) {
            return self.number();
        }
        self.bump();
        Ok(match c {
            '\'' => Token::Str(self.quoted('\'', "string literal")?),
            '"' => {
                let value = self.quoted('"', "quoted identifier")?;
                if value.is_empty() {
                    return self.error("empty quoted identifier");
                }
                Token::Ident {
                    value,
                    quoted: true,
                }
            }
            ',' => Token::Comma,
            '.' => Token::Dot,
            '(' => Token::LParen,
            ')' => Token::RParen,
            ';' => Token::Semicolon,
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '%' => Token::Percent,
            '=' => {
                if self.peek() == Some('=') {
                    self.bump();
                }
                Token::Eq
            }
            '!' if self.peek() == Some('=') => {
                self.bump();
                Token::NotEq
            }
            '<' => match self.peek() {
                Some('=') => {
                    self.bump();
                    Token::LtEq
                }
                Some('>') => {
                    self.bump();
                    Token::NotEq
                }
                _ => Token::Lt,
            },
            '>' => {
                if self.peek() == Some('=') {
                    self.bump();
                    Token::GtEq
                } else {
                    Token::Gt
                }
            }
            '|' if self.peek() == Some('|') => {
                self.bump();
                Token::Concat
            }
            other => {
                // Report at the offending character, which we already consumed.
                return Err(QuarryError::Parse {
                    message: format!("unexpected character '{other}'"),
                    line: self.line,
                    column: self.column.saturating_sub(1),
                });
            }
        })
    }

    fn number(&mut self) -> Result<Token> {
        let mut s = String::new();
        let mut seen_dot = false;
        let mut seen_exp = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
            } else if c == '.' && !seen_dot && !seen_exp {
                seen_dot = true;
                s.push(c);
            } else if (c == 'e' || c == 'E') && !seen_exp {
                let next = self.peek_at(1);
                let next2 = self.peek_at(2);
                let digit_follows = next.is_some_and(|d| d.is_ascii_digit())
                    || (matches!(next, Some('+') | Some('-'))
                        && next2.is_some_and(|d| d.is_ascii_digit()));
                if !digit_follows {
                    break;
                }
                seen_exp = true;
                s.push('e');
                self.bump();
                if let Some(sign @ ('+' | '-')) = self.peek() {
                    s.push(sign);
                    self.bump();
                }
                continue;
            } else {
                break;
            }
            self.bump();
        }
        if self
            .peek()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        {
            return self.error(format!(
                "invalid numeric literal '{s}{}'",
                self.peek().unwrap_or(' ')
            ));
        }
        Ok(Token::Number(s))
    }

    /// Read a quoted run, the opening quote already consumed; a doubled quote escapes itself.
    fn quoted(&mut self, quote: char, what: &str) -> Result<String> {
        let mut s = String::new();
        loop {
            match self.bump() {
                Some(c) if c == quote => {
                    if self.peek() == Some(quote) {
                        self.bump();
                        s.push(quote);
                    } else {
                        return Ok(s);
                    }
                }
                Some(c) => s.push(c),
                None => return self.error(format!("unterminated {what}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(sql: &str) -> Vec<Token> {
        tokenize(sql)
            .unwrap()
            .into_iter()
            .map(|s| s.token)
            .collect()
    }

    fn ident(s: &str) -> Token {
        Token::Ident {
            value: s.into(),
            quoted: false,
        }
    }

    #[test]
    fn keywords_fold_to_lower_case() {
        assert_eq!(
            kinds("SELECT a FROM T"),
            vec![
                ident("select"),
                ident("a"),
                ident("from"),
                ident("t"),
                Token::Eof
            ]
        );
    }

    #[test]
    fn literals_and_operators() {
        assert_eq!(
            kinds("1 2.5 .5 1e3 'it''s' <> != <= >= || == \"Col\""),
            vec![
                Token::Number("1".into()),
                Token::Number("2.5".into()),
                Token::Number(".5".into()),
                Token::Number("1e3".into()),
                Token::Str("it's".into()),
                Token::NotEq,
                Token::NotEq,
                Token::LtEq,
                Token::GtEq,
                Token::Concat,
                Token::Eq,
                Token::Ident {
                    value: "Col".into(),
                    quoted: true
                },
                Token::Eof
            ]
        );
    }

    #[test]
    fn comments_are_skipped_and_positions_tracked() {
        let toks = tokenize("-- hello\nselect /* x\n y */ a").unwrap();
        assert_eq!(toks[0].token, ident("select"));
        assert_eq!((toks[0].line, toks[0].column), (2, 1));
        assert_eq!((toks[1].line, toks[1].column), (3, 7));
    }

    #[test]
    fn lexical_errors() {
        assert!(matches!(tokenize("'abc"), Err(QuarryError::Parse { .. })));
        assert!(matches!(
            tokenize("/* never closed"),
            Err(QuarryError::Parse { .. })
        ));
        assert!(matches!(
            tokenize("select #"),
            Err(QuarryError::Parse { column: 8, .. })
        ));
        assert!(matches!(tokenize("12abc"), Err(QuarryError::Parse { .. })));
        assert!(matches!(tokenize("\"\""), Err(QuarryError::Parse { .. })));
    }
}
