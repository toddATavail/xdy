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
        $.highest,
        $.lowest,
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

    // A call that answers the greatest of the values of its arguments, two or
    // more full expressions. As a dice count, faces, the count of a clause, or
    // an operand of a reroll or explode, a call must be grouped.
    highest: ($) =>
      seq(
        "highest",
        "(",
        field("argument", $._expression),
        repeat1(seq(",", field("argument", $._expression))),
        ")",
      ),

    // A call that answers the least of the values of its arguments, two or more
    // full expressions. As a dice count, faces, the count of a clause, or an
    // operand of a reroll or explode, a call must be grouped.
    lowest: ($) =>
      seq(
        "lowest",
        "(",
        field("argument", $._expression),
        repeat1(seq(",", field("argument", $._expression))),
        ")",
      ),

    // A dice expression.
    _dice: ($) =>
      choice(
        $.standard_dice,
        $.custom_dice,
        $.drop_lowest,
        $.drop_highest,
        $.keep_lowest,
        $.keep_highest,
        $.reroll,
        $.explode,
      ),

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

    // A dice expression that keeps the N lowest of the dice that the clauses
    // before it left kept, or one if N is absent.
    keep_lowest: ($) =>
      prec.left(
        5,
        seq(
          field("dice", $._dice),
          "keep",
          "lowest",
          field("keep", optional($._drop_expression)),
        ),
      ),

    // A dice expression that keeps the N highest of the dice that the clauses
    // before it left kept, or one if N is absent.
    keep_highest: ($) =>
      prec.left(
        5,
        seq(
          field("dice", $._dice),
          "keep",
          "highest",
          field("keep", optional($._drop_expression)),
        ),
      ),

    // A dice expression that rerolls each of the dice that the clauses before
    // it left kept that matches its predicate, once, or as many times as its
    // limit says.
    reroll: ($) =>
      prec.left(
        5,
        seq(
          field("dice", $._dice),
          "reroll",
          field("predicate", $._predicate),
          field("limit", optional($._limit)),
        ),
      ),

    // A dice expression that explodes each of the dice that the clauses before
    // it left kept that matches its predicate, rolling another die to extend
    // its chain, once, or as many times as its limit says. A die that extends
    // a chain may itself match, extending the chain again.
    explode: ($) =>
      prec.left(
        5,
        seq(
          field("dice", $._dice),
          "explode",
          field("predicate", $._predicate),
          field("limit", optional($._limit)),
        ),
      ),

    // The predicate of a reroll or explode, which decides which results it
    // rerolls or explodes.
    _predicate: ($) =>
      choice(
        $.eq,
        $.ne,
        $.lt,
        $.le,
        $.gt,
        $.ge,
        $.in_list,
        $.in_range,
        $.in_range_inclusive,
      ),

    // A predicate that matches a result equal to its threshold.
    eq: ($) => seq("=", field("threshold", $._standard_faces)),

    // A predicate that matches a result unequal to its threshold.
    ne: ($) => seq(choice("!=", "≠"), field("threshold", $._standard_faces)),

    // A predicate that matches a result less than its threshold.
    lt: ($) => seq("<", field("threshold", $._standard_faces)),

    // A predicate that matches a result less than or equal to its threshold.
    le: ($) => seq(choice("<=", "≤"), field("threshold", $._standard_faces)),

    // A predicate that matches a result greater than its threshold.
    gt: ($) => seq(">", field("threshold", $._standard_faces)),

    // A predicate that matches a result greater than or equal to its
    // threshold.
    ge: ($) => seq(choice(">=", "≥"), field("threshold", $._standard_faces)),

    // A predicate that matches a result among its literals, of which there may
    // be none.
    in_list: ($) =>
      seq(
        "in",
        "[",
        optional(
          seq(
            field("value", $.integer),
            repeat(seq(",", field("value", $.integer))),
          ),
        ),
        "]",
      ),

    // A predicate that matches a result within the half-open range
    // `start..end`, which is empty if its end does not exceed its start.
    in_range: ($) =>
      seq(
        "in",
        field("start", $._standard_faces),
        "..",
        field("end", $._standard_faces),
      ),

    // A predicate that matches a result within the inclusive range
    // `start..=end`, which is empty if its end is less than its start.
    in_range_inclusive: ($) =>
      seq(
        "in",
        field("start", $._standard_faces),
        "..=",
        field("end", $._standard_faces),
      ),

    // The limit of a reroll or explode, which bounds how many times it rerolls
    // or explodes each die.
    _limit: ($) => choice($.times, $.unlimited),

    // A limit that rerolls or explodes each die at most N times. `time` and
    // `times` read alike.
    times: ($) =>
      seq(field("count", $._drop_expression), choice("time", "times")),

    // A limit that rerolls or explodes each die until it no longer matches.
    // `∞` and `unlimited` read alike, as do `time` and `times`.
    unlimited: ($) => seq(choice("unlimited", "∞"), choice("time", "times")),

    // An expression that represents the number of dice to drop or keep, or the
    // number of times to reroll or explode each die.
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
