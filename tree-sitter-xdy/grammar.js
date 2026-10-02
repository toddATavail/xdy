/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name: "xdy",

  // Tokens may be separated by spaces, horizontal tabs, line feeds, and
  // carriage returns, but by no other whitespace, exactly as the reference nom
  // parser separates them (see parser::combinators::is_token_space).
  extras: ($) => [/[ \t\r\n]/],

  rules: {
    //////////////////////////// Grammatical rules. ////////////////////////////

    // The main entry point of the grammar, representing a complete dice
    // function definition.
    function: ($) =>
      seq(
        field("parameters", optional($.parameters)),
        field("body", $._expression),
      ),

    // The formal parameters of a dice function.
    parameters: ($) =>
      seq(
        field("parameter", $.parameter),
        repeat(seq(",", field("parameter", $.parameter))),
        ":",
      ),

    // A single formal parameter.
    parameter: ($) =>
      seq($._open_brace, field("identifier", $.identifier), $._close_brace),

    // A dice expression.
    _expression: ($) =>
      choice(
        $.group,
        $.constant,
        $.variable,
        $.binding,
        $.range,
        $._dice,
        $._arithmetic,
      ),

    // A parenthesized expression.
    group: ($) => seq("(", field("expression", $._expression), ")"),

    // A variable, representing a parameter, a local binding, or an external
    // variable.
    variable: ($) =>
      seq($._open_brace, field("identifier", $.identifier), $._close_brace),

    // A local binding, naming the integer result of a subexpression so it can
    // be referenced as a variable at later lexical positions.
    binding: ($) =>
      seq(
        $._open_brace,
        field("name", $.identifier),
        $._close_brace,
        "@",
        "(",
        field("expression", $._expression),
        ")",
      ),

    // A range expression.
    range: ($) =>
      seq(
        "[",
        field("start", $._expression),
        ":",
        field("end", $._expression),
        "]",
      ),

    // A dice expression.
    _dice: ($) =>
      choice($.standard_dice, $.custom_dice, $.drop_lowest, $.drop_highest),

    // A standard dice expression, i.e., the faces of the dice are numbered
    // from 1 to N.
    standard_dice: ($) =>
      seq(
        field("count", $._dice_count),
        $._d,
        field("faces", $._standard_faces),
      ),

    // A custom dice expression, i.e., the faces of the dice are arbitrary.
    custom_dice: ($) =>
      seq(field("count", $._dice_count), $._d, field("faces", $.custom_faces)),

    // An expression that represents the number of dice to roll.
    _dice_count: ($) => choice($.constant, $.variable, $.binding, $.group),

    // An expression that represents the faces of a standard die, which may be
    // a signed integer, as in `3D-6`.
    _standard_faces: ($) => choice($.integer, $.variable, $.binding, $.group),

    // The faces of a custom die.
    custom_faces: ($) =>
      seq(
        "[",
        field("face", $.integer),
        repeat(seq(",", field("face", $.integer))),
        "]",
      ),

    // A dice expression that drops its N lowest dice, or one if N is absent.
    drop_lowest: ($) =>
      prec.left(
        5,
        seq(
          field("dice", $._dice),
          "drop",
          "lowest",
          field("drop", optional($._drop_expression)),
        ),
      ),

    // A dice expression that drops its N highest dice, or one if N is absent.
    drop_highest: ($) =>
      prec.left(
        5,
        seq(
          field("dice", $._dice),
          "drop",
          "highest",
          field("drop", optional($._drop_expression)),
        ),
      ),

    // An expression that represents the number of dice to drop.
    _drop_expression: ($) => choice($.constant, $.variable, $.binding, $.group),

    // An arithmetic expression.
    _arithmetic: ($) => choice($.add, $.sub, $.mul, $.div, $.mod, $.exp, $.neg),

    // An addition expression.
    add: ($) =>
      prec.left(
        1,
        seq(field("op1", $._expression), "+", field("op2", $._expression)),
      ),

    // A subtraction expression.
    sub: ($) =>
      prec.left(
        1,
        seq(field("op1", $._expression), "-", field("op2", $._expression)),
      ),

    // A multiplication expression.
    mul: ($) =>
      prec.left(
        2,
        seq(field("op1", $._expression), /[*×]/, field("op2", $._expression)),
      ),

    // A division expression.
    div: ($) =>
      prec.left(
        2,
        seq(field("op1", $._expression), /[/÷]/, field("op2", $._expression)),
      ),

    // A modulo expression.
    mod: ($) =>
      prec.left(
        2,
        seq(field("op1", $._expression), "%", field("op2", $._expression)),
      ),

    // An exponentiation expression.
    exp: ($) =>
      prec.right(
        4,
        seq(field("op1", $._expression), "^", field("op2", $._expression)),
      ),

    // A negation expression.
    neg: ($) => prec(3, seq("-", field("op", $._expression))),

    ////////////////////////////// Lexical rules. //////////////////////////////

    // An unsigned constant, CONSTANT in the reference nom parser's EBNF. A
    // minus before one is negation, so `-3D6` is `-(3D6)`.
    constant: ($) => /\d+/,

    // A signed integer, INTEGER in the reference nom parser's EBNF, for the
    // faces of dice only, as in `3D-6` and `1D[-1, 0, 1]`.
    integer: ($) => /-?\d+/,

    // The dice operator.
    _d: ($) => /[dD]/,

    // The braces of a name, with any whitespace (White_Space, per
    // char::is_whitespace) just inside them, which is not part of the name, as
    // the reference nom parser allows (see parser::combinators::braced_name).
    // Between tokens, only the extras may separate them. The extras already
    // skip a space, tab, line feed, or carriage return before the closing
    // brace, so its token begins with any other whitespace, lest it begin with
    // an extra and claim the whitespace that follows a name.
    _open_brace: ($) => token(seq("{", /\p{White_Space}*/u)),

    _close_brace: ($) =>
      token(
        choice(
          "}",
          seq(
            /[\u000B\u000C\u0085\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]/,
            /\p{White_Space}*/u,
            "}",
          ),
        ),
      ),

    // An identifier, the name of a parameter, a local binding, or an external
    // variable, always between braces. The character set is exact with the
    // reference nom parser (see parser::combinators::is_identifier_char): any
    // code point except the braces, the control characters (Cc, per
    // char::is_control) other than whitespace, the bidirectional formatting
    // controls, and the other invisible characters enumerated below.
    // Whitespace of any kind (White_Space, per char::is_whitespace) may occur
    // within an identifier, but not at either end: the whitespace around it,
    // inside its braces, is not part of it. The nom parser collapses each run
    // of whitespace within a name to a single space (see
    // parser::combinators::canonical_name).
    identifier: ($) =>
      /[^{}\p{Cc}\p{White_Space}\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069\u00AD\u115F\u1160\u180E\u200B\u2060-\u2064\u3164\uFEFF\uFFA0](\p{White_Space}*[^{}\p{Cc}\p{White_Space}\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069\u00AD\u115F\u1160\u180E\u200B\u2060-\u2064\u3164\uFEFF\uFFA0])*/u,
  },
});
