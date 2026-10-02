/-!
# Primitives

The arithmetic primitives of the xDy dice language, mirroring
`xdy/src/primitives.rs` case by case. The Rust primitives operate on `i32`;
here every value is an `Int` that lies within the `i32` range, and each
primitive answers the clamp of its exact mathematical result into that range.
For operands within the range, this is exactly what Rust's saturating
operations compute, so the two agree on every input.

Keeping values as `Int` rather than as a bounded type keeps the executable
oracle simple and leaves the range invariant to be stated and proved
alongside the specification.
-/

namespace Xdy

/-! ## The `i32` range -/

/-- The least `i32`, `-2³¹`. -/
def i32Min : Int := -2147483648

/-- The greatest `i32`, `2³¹ - 1`. -/
def i32Max : Int := 2147483647

/--
Clamp an integer into the `i32` range, which is how every primitive saturates.

# Parameters
- `x`: The exact result of an operation.

# Returns
`x` if it lies within the `i32` range, otherwise the nearer bound.

# Examples
```lean
#guard clamp 7 == 7
#guard clamp (i32Max + 1) == i32Max
#guard clamp (i32Min - 1) == i32Min
```
-/
def clamp (x : Int) : Int := max i32Min (min i32Max x)

/-! ## Arithmetic -/

/--
Compute the sum of the two operands, saturating on overflow. Mirrors `add` in
`primitives.rs`.

# Parameters
- `op1`: The augend.
- `op2`: The addend.

# Returns
The sum of the operands.

# Examples
```lean
#guard add 1 2 == 3
#guard add i32Max 1 == i32Max
#guard add i32Min (-1) == i32Min
```
-/
def add (op1 op2 : Int) : Int := clamp (op1 + op2)

/--
Compute the difference of the two operands, saturating on overflow. Mirrors
`sub` in `primitives.rs`.

# Parameters
- `op1`: The minuend.
- `op2`: The subtrahend.

# Returns
The difference of the operands.

# Examples
```lean
#guard sub 1 2 == -1
#guard sub i32Max (-1) == i32Max
#guard sub i32Min 1 == i32Min
```
-/
def sub (op1 op2 : Int) : Int := clamp (op1 - op2)

/--
Compute the product of the two operands, saturating on overflow. Mirrors `mul`
in `primitives.rs`.

# Parameters
- `op1`: The multiplicand.
- `op2`: The multiplier.

# Returns
The product of the operands.

# Examples
```lean
#guard mul 2 3 == 6
#guard mul i32Max 2 == i32Max
#guard mul i32Min 2 == i32Min
#guard mul i32Max (-1) == i32Min + 1
```
-/
def mul (op1 op2 : Int) : Int := clamp (op1 * op2)

/--
Compute the quotient of the two operands, rounding toward zero and saturating
on overflow. Division by zero answers zero. Mirrors `div` in `primitives.rs`.

# Parameters
- `op1`: The dividend.
- `op2`: The divisor.

# Returns
The quotient of the operands, or `0` if the divisor is `0`.

# Notes
Rust's `/` rounds toward zero, which is `Int.tdiv`, not Lean's `/` on `Int`.
The only quotient of two `i32`s that overflows is `i32Min / -1`, which
saturates to `i32Max`.

# Examples
```lean
#guard div 6 2 == 3
#guard div 6 0 == 0
#guard div (-7) 2 == -3
#guard div i32Min (-1) == i32Max
```
-/
def div (op1 op2 : Int) : Int :=
  if op2 == 0 then 0 else clamp (op1.tdiv op2)

/--
Compute the remainder of dividing the first operand by the second, taking the
sign of the dividend. Division by zero answers zero. Mirrors `mod` in
`primitives.rs`.

# Parameters
- `op1`: The dividend.
- `op2`: The divisor.

# Returns
The remainder, or `0` if the divisor is `0`.

# Notes
Rust's `%` takes the sign of the dividend, which is `Int.tmod`, not Lean's `%`
on `Int`. Rust special-cases `i32Min % -1`, which would otherwise overflow in
the machine; the exact remainder is `0` anyway, so no special case is needed
here.

# Examples
```lean
#guard «mod» 6 4 == 2
#guard «mod» 6 0 == 0
#guard «mod» (-7) 2 == -1
#guard «mod» i32Min (-1) == 0
```
-/
def «mod» (op1 op2 : Int) : Int :=
  if op2 == 0 then 0 else op1.tmod op2

/--
Raise the first operand to the power of the second, saturating on overflow.
Mirrors `exp` in `primitives.rs`, including its special cases: `0 ^ 0` is
`1`; `0` to any other power is `0`, even a negative one; `1` to any power is
`1`; `-1` alternates by the parity of the exponent, even a negative one; and
any other base to a negative power is `0`.

# Parameters
- `op1`: The base.
- `op2`: The exponent.

# Returns
The power.

# Notes
The exact power can be astronomically large, e.g., `2 ^ i32Max`, so it is not
computed outright. A base of magnitude at least `2` raised to a power of at
least `32` has magnitude at least `2³²`, beyond the `i32` range, so it
saturates toward its sign: `i32Min` for a negative base and an odd exponent,
otherwise `i32Max`. Any smaller power is computed exactly and clamped.

# Examples
```lean
#guard exp 2 3 == 8
#guard exp 0 0 == 1
#guard exp 0 (-1) == 0
#guard exp 1 (-1) == 1
#guard exp (-1) (-3) == -1
#guard exp 2 (-1) == 0
#guard exp 10 20 == i32Max
#guard exp (-10) 21 == i32Min
```
-/
def exp (op1 op2 : Int) : Int :=
  if op1 == 0 then
    if op2 == 0 then 1 else 0
  else if op1 == 1 then 1
  else if op1 == -1 then
    if op2.tmod 2 == 0 then 1 else -1
  else if op2 < 0 then 0
  else if op2 >= 32 then
    if op1 < 0 && op2.tmod 2 != 0 then i32Min else i32Max
  else clamp (op1 ^ op2.toNat)

/--
Compute the negation of the operand, saturating on overflow. Mirrors `neg` in
`primitives.rs`.

# Parameters
- `op`: The operand.

# Returns
The negation of the operand.

# Examples
```lean
#guard neg 1 == -1
#guard neg i32Max == i32Min + 1
#guard neg i32Min == i32Max
```
-/
def neg (op : Int) : Int := clamp (-op)

/--
Compute the greater of the two operands. Mirrors `max` in `primitives.rs`,
which the optimizer uses to clamp values that the language clamps implicitly.

# Parameters
- `op1`: The first operand.
- `op2`: The second operand.

# Returns
The greater operand.

# Examples
```lean
#guard «max» 1 2 == 2
#guard «max» 0 (-3) == 0
#guard «max» i32Min i32Max == i32Max
```
-/
def «max» (op1 op2 : Int) : Int := Max.max op1 op2

end Xdy
