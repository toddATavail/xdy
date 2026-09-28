/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name: "xdy",

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
    parameter: ($) => seq("{", field("identifier", $.identifier), "}"),

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
    variable: ($) => seq("{", field("identifier", $.identifier), "}"),

    // A local binding, naming the integer result of a subexpression so it can
    // be referenced as a variable at later lexical positions.
    binding: ($) =>
      seq(
        "{",
        field("name", $.identifier),
        "}",
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
    // Has higher precedence than `standard_dice` to resolve the single token
    // lookahead ambiguity.
    custom_dice: ($) =>
      seq(field("count", $._dice_count), $._d, field("faces", $.custom_faces)),

    // An expression that represents the number of dice to roll.
    _dice_count: ($) => choice($.constant, $.variable, $.binding, $.group),

    // An expression that represents the faces of a standard die.
    _standard_faces: ($) => choice($.constant, $.variable, $.binding, $.group),

    // The faces of a custom die.
    custom_faces: ($) =>
      seq(
        "[",
        field("face", $._custom_face),
        repeat(seq(",", field("face", $._custom_face))),
        "]",
      ),

    // A custom face of a custom die, which can be a constant or a negative
    // constant.
    _custom_face: ($) => choice($.constant, $.negative_constant),

    // A dice expression that keeps the highest N-1 dice.
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

    // A dice expression that keeps the lowest N-1 dice.
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

    // A negative constant, for custom faces only.
    negative_constant: ($) => /\-\d+/,

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

    // A constant integer.
    constant: ($) => /\d+/,

    // The dice operator.
    _d: ($) => /[dD]/,

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
