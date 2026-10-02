import Xdy.Saturation
import XdySpec.Semantics

/-!
# Independent lifts and convolution powers

Lemmas 1 and 2 of the design: the law of a binary operation over independent
values, and the law of the sum of many independent dice.

Lemma 1 says that lifting a binary operation over independent laws weighs the
result of each pair of outcomes by the product of their probabilities, and
adds the weights of pairs with equal results. That is what `pairwise` in
`xdy/src/distribution/propagation.rs` computes. A saturating sum or difference
of two values is their true sum or difference, clamped, so its lift is the
convolution of the operands, clamped once, as `saturating_sum` computes it.

Lemma 2 says that the `n`th convolution power of a die is the law of the sum
of `n` independent draws from it, as `rollDice` records them, and that the
powers add: `convPow d (m + n)` is `convPow d m` convolved with `convPow d n`.
That justifies both `convolution_power` in
`xdy/src/distribution/propagation/record.rs`, which squares, and `Powers`
beside it, which steps up from the greatest power that it already knows.

Case 1 of the design puts lemmas 2 and 3 together: a roll without drops, of
dice whose faces pass the test of `folds_to_clamp` in `record.rs`, has the
law of the die's convolution power, clamped once, which is what the Rust
builds for it rather than folding. Lemma 3 speaks of sorted results, and the
specification's records hold theirs in roll order, so it applies through
`Record.toOracle` and the invariance of the true sum under reordering.

Mathlib has no convolution of `PMF`s, so this module defines one, as the lift
of addition.
-/

namespace Xdy.Spec

/-! ## Independent lifts -/

/--
The law of a binary operation over independent values.

# Parameters
- `op`: The operation.
- `p`: The law of the first operand.
- `q`: The law of the second operand, independent of the first.

# Returns
The law of `op x y`, where `x` is drawn from `p` and `y` from `q`.

# Notes
The operands of an instruction are independent unless some random value
reaches both. That is a property of the joint law of the machine state, not of
two separate laws, so it is lemma 6's, in `XdySpec/Conditioning.lean`, where
the Rust conditions on the values that reach both.

# Examples
```lean
example : lift (· * ·) (PMF.pure 2) (PMF.pure 3) = PMF.pure 6 := by
  simp [lift, PMF.pure_map]
```
-/
noncomputable def lift {α β γ : Type*} (op : α → β → γ) (p : PMF α)
    (q : PMF β) : PMF γ :=
  p.bind fun x => q.map (op x)

/--
Lemma 1: lifting a binary operation over independent laws equals the pairwise
sum, which weighs the result of each pair of outcomes by the product of their
probabilities. Mirrors `pairwise` in `propagation.rs`.

# Parameters
- `op`: The operation.
- `p`: The law of the first operand.
- `q`: The law of the second operand.
- `c`: A result.
-/
theorem lift_apply {α β γ : Type*} [DecidableEq γ] (op : α → β → γ)
    (p : PMF α) (q : PMF β) (c : γ) :
    lift op p q c = ∑' x, ∑' y, if c = op x y then p x * q y else 0 := by
  simp only [lift, PMF.bind_apply, PMF.map_apply]
  congr 1; ext x
  rw [← ENNReal.tsum_mul_left]
  congr 1; ext y
  split_ifs <;> simp

/--
Applying a function to the result of a lift lifts the composite operation.

# Parameters
- `op`: The operation.
- `f`: The function to apply to its result.
- `p`: The law of the first operand.
- `q`: The law of the second operand.
-/
theorem map_lift {α β γ δ : Type*} (op : α → β → γ) (f : γ → δ)
    (p : PMF α) (q : PMF β) :
    (lift op p q).map f = lift (fun x y => f (op x y)) p q := by
  simp [lift, PMF.map_bind, PMF.map_comp, Function.comp_def]

/-! ## Convolution -/

/--
The convolution of two laws: the law of the sum of independent values.

# Parameters
- `p`: The law of the first value.
- `q`: The law of the second value.

# Returns
The law of `x + y`, where `x` is drawn from `p` and `y` from `q`. The sum is
exact, not saturating, as in `convolve` in `record.rs`, which sums in `i64`.

# Examples
```lean
example : conv (PMF.pure 1) (PMF.pure 2) = PMF.pure 3 := by
  simp [conv, lift, PMF.pure_map]
```
-/
noncomputable def conv (p q : PMF Int) : PMF Int := lift (· + ·) p q

/--
The lift of saturating addition is the convolution, clamped once. Mirrors
`saturating_sum` in `propagation.rs`, when it adds.

# Parameters
- `p`: The law of the first operand.
- `q`: The law of the second operand.
-/
theorem lift_add (p q : PMF Int) : lift add p q = (conv p q).map clamp := by
  rw [conv, map_lift]; rfl

/--
The lift of saturating subtraction is the convolution with the negated second
operand, clamped once. Mirrors `saturating_sum` in `propagation.rs`, when it
subtracts.

# Parameters
- `p`: The law of the first operand.
- `q`: The law of the second operand.
-/
theorem lift_sub (p q : PMF Int) :
    lift sub p q = (conv p (q.map (-·))).map clamp := by
  rw [conv, map_lift]
  simp only [lift, PMF.map_comp, Function.comp_def]
  rfl

/--
Convolution is commutative.

# Parameters
- `p`, `q`: The laws.
-/
theorem conv_comm (p q : PMF Int) : conv p q = conv q p := by
  simp only [conv, lift, ← PMF.bind_pure_comp, Function.comp_def]
  rw [PMF.bind_comm]
  simp [Int.add_comm]

/--
Convolution is associative.

# Parameters
- `p`, `q`, `r`: The laws.
-/
theorem conv_assoc (p q r : PMF Int) :
    conv (conv p q) r = conv p (conv q r) := by
  simp only [conv, lift, ← PMF.bind_pure_comp, Function.comp_def,
    PMF.bind_bind, PMF.pure_bind, Int.add_assoc]

/--
A certain `0` is a right identity of convolution.

# Parameters
- `p`: The law.
-/
theorem conv_zero (p : PMF Int) : conv p (PMF.pure 0) = p := by
  simp [conv, lift, PMF.pure_map]

/--
A certain `0` is a left identity of convolution.

# Parameters
- `p`: The law.
-/
theorem zero_conv (p : PMF Int) : conv (PMF.pure 0) p = p := by
  rw [conv_comm, conv_zero]

/-! ## Convolution powers -/

/--
The convolution power of a law: the law of the sum of `n` independent draws
from it.

# Parameters
- `d`: The law of one draw.
- `n`: The number of draws.

# Returns
The law of the sum, which is a certain `0` if there are no draws. The sum is
exact, not saturating.

# Examples
```lean
example (d : PMF Int) : convPow d 0 = PMF.pure 0 := rfl
example (d : PMF Int) : convPow d 1 = d := by simp [convPow, zero_conv]
```
-/
noncomputable def convPow (d : PMF Int) : Nat → PMF Int
  | 0 => PMF.pure 0
  | n + 1 => conv (convPow d n) d

/--
Lemma 2, the law of addition of powers: the power of a sum of counts is the
convolution of the powers of the counts. So `convolution_power` in
`record.rs` may square, since `convPow d (n + n)` is `convPow d n` convolved
with itself, and `Powers` may step up from the power of `k ≤ n` dice, since
`convPow d n` is `convPow d k` convolved with `convPow d (n - k)`.

# Parameters
- `d`: The law of one draw.
- `m`, `n`: The counts.

# Examples
```lean
example (d : PMF Int) (n : Nat) :
    convPow d (n + n) = conv (convPow d n) (convPow d n) :=
  convPow_add d n n
```
-/
theorem convPow_add (d : PMF Int) (m n : Nat) :
    convPow d (m + n) = conv (convPow d m) (convPow d n) := by
  induction n with
  | zero => simp [convPow, conv_zero]
  | succ n ih => rw [← Nat.add_assoc, convPow, ih, convPow, conv_assoc]

/--
Lemma 2, the law of the sum of dice: the true sum of the results of a set of
dice, as `rollDice` records them, has the law of the convolution power of one
die.

# Parameters
- `n`: The number of dice. A count of `0` or less rolls no dice.
- `die`: The law of one die.

# Notes
This is the true sum of every result, in the order rolled. `Record.sum`
instead folds the kept results in ascending order, saturating; when there are
no drops and the fold equals one clamp of the true sum, as lemma 3 gives, its
law is this one, clamped: see `rollDice_saturating_sum`.

# Examples
```lean
example (die : PMF Int) :
    (rollDice 2 die).map (·.results.sum) = conv die die := by
  rw [rollDice_sum]; simp [convPow, zero_conv]
```
-/
theorem rollDice_sum (n : Int) (die : PMF Int) :
    (rollDice n die).map (·.results.sum) = convPow die n.toNat := by
  unfold rollDice
  induction n.toNat with
  | zero => simp [Nat.repeat, convPow, PMF.pure_map]
  | succ k ih =>
    rw [Nat.repeat, convPow, ← ih]
    simp [conv, lift, PMF.map_bind, PMF.bind_map, PMF.map_comp,
      Function.comp_def, Record.push]

/-! ## Case 1: sums without drops -/

/--
The sum of a record without drops equals one clamp of the true sum of its
results, under the test of `folds_to_clamp` in
`xdy/src/distribution/propagation/record.rs`. The specification's form of
`Xdy.Record.sum_eq_clamp`, which needs sorted results: it applies to the
record sorted, whose true sum is the same.

# Parameters
- `r`: The record, whose results are in roll order.
- `least`: A lower bound on the results, such as the least face of the die.
- `greatest`: An upper bound on the results, such as the greatest face of the
  die.

# Hypotheses
- `lowest`, `highest`: The record drops none of its results.
- `hleast`: Every result is at least `least`.
- `hgreatest`: Every result is at most `greatest`.
- `test`: The test of `folds_to_clamp` passes: `least` is nonnegative,
  `greatest` is nonpositive, or the number of results times `least` is no
  less than `i32Min`.
-/
theorem Record.sum_eq_clamp (r : Record) (lowest : r.lowest = 0)
    (highest : r.highest = 0) (least greatest : Int)
    (hleast : ∀ x ∈ r.results, least ≤ x)
    (hgreatest : ∀ x ∈ r.results, x ≤ greatest)
    (test : 0 ≤ least ∨ greatest ≤ 0 ∨ i32Min ≤ r.results.length * least) :
    r.sum = clamp r.results.sum := by
  have perm : r.sorted.Perm r.results := List.mergeSort_perm _ _
  rw [← Record.sum_toOracle, Xdy.Record.sum_eq_clamp r.toOracle
    (Record.sorted_pairwise r) lowest highest least greatest
    (fun x hx => hleast x (perm.mem_iff.mp hx))
    (fun x hx => hgreatest x (perm.mem_iff.mp hx))
    (by simpa [Record.toOracle, Record.sorted] using test)]
  exact congrArg clamp perm.sum_eq

/--
Every possible record of `m` dice drops nothing, and holds `m` results, each
a possible face of the die.

# Parameters
- `die`: The law of one die.
- `m`: The number of dice.
- `r`: The record.

# Hypotheses
- `hr`: The record is possible.
-/
private theorem mem_support_repeat {die : PMF Int} (m : Nat) {r : Record}
    (hr : r ∈ (m.repeat (fun d => d.bind fun r => die.map r.push)
      (PMF.pure {})).support) :
    r.lowest = 0 ∧ r.highest = 0 ∧ r.results.length = m ∧
      ∀ x ∈ r.results, x ∈ die.support := by
  induction m generalizing r with
  | zero =>
    simp only [Nat.repeat, PMF.mem_support_pure_iff] at hr
    subst hr
    simp
  | succ m ih =>
    simp only [Nat.repeat, PMF.mem_support_bind_iff,
      PMF.mem_support_map_iff] at hr
    obtain ⟨r', hr', x, hx, rfl⟩ := hr
    obtain ⟨h1, h2, h3, h4⟩ := ih hr'
    refine ⟨h1, h2, by simp [Record.push, h3], fun y hy => ?_⟩
    simp only [Record.push, List.mem_append, List.mem_singleton] at hy
    rcases hy with hy | rfl
    · exact h4 y hy
    · exact hx

/--
Every possible record of a set of dice drops nothing, and holds one result
per die, each a possible face of the die.

# Parameters
- `n`: The number of dice.
- `die`: The law of one die.
- `r`: The record.

# Hypotheses
- `hr`: The record is possible.
-/
theorem mem_support_rollDice {n : Int} {die : PMF Int} {r : Record}
    (hr : r ∈ (rollDice n die).support) :
    r.lowest = 0 ∧ r.highest = 0 ∧ r.results.length = n.toNat ∧
      ∀ x ∈ r.results, x ∈ die.support :=
  mem_support_repeat n.toNat hr

/--
Transformations that agree on the support of a law transform it alike.

# Parameters
- `p`: The law.
- `f`, `g`: The transformations.

# Hypotheses
- `h`: `f` and `g` agree on every possible outcome.
-/
theorem map_congr_support {α β : Type} {p : PMF α} {f g : α → β}
    (h : ∀ a ∈ p.support, f a = g a) : p.map f = p.map g := by
  ext b
  simp only [PMF.map_apply]
  refine tsum_congr fun a => ?_
  by_cases ha : a ∈ p.support
  · rw [h a ha]
  · rw [PMF.mem_support_iff, not_not] at ha
    simp [ha]

/--
Case 1, end to end: the saturating sum of a roll without drops has the law
of the die's convolution power, clamped once, if the die's faces pass the
test of `folds_to_clamp` in `xdy/src/distribution/propagation/record.rs`.
That is what the Rust builds for such a roll, rather than folding, so this
proves it sound.

# Parameters
- `n`: The number of dice.
- `die`: The law of one die.
- `least`: A lower bound on the die's faces, such as its least face.
- `greatest`: An upper bound on the die's faces, such as its greatest face.

# Hypotheses
- `hdie`: Every possible face lies between `least` and `greatest`.
- `test`: The test of `folds_to_clamp` passes: `least` is nonnegative,
  `greatest` is nonpositive, or `n` copies of `least` sum to no less than
  `i32Min`.

# Notes
Lemma 2, `rollDice_sum`, gives the law of the true sum, and lemma 3,
`Record.sum_eq_clamp`, that every possible roll's saturating sum is its true
sum clamped once.

# Examples
```lean
-- `3D6`: the faces are positive.
example : (rollDice 3 (standardDie 6)).map (·.sum)
    = (convPow (standardDie 6) 3).map clamp :=
  rollDice_saturating_sum 3 _ 1 6 (fun x hx => by
    simp [standardDie] at hx; omega)
    (.inl (by decide))
```
-/
theorem rollDice_saturating_sum (n : Int) (die : PMF Int)
    (least greatest : Int)
    (hdie : ∀ x ∈ die.support, least ≤ x ∧ x ≤ greatest)
    (test : 0 ≤ least ∨ greatest ≤ 0 ∨ i32Min ≤ n.toNat * least) :
    (rollDice n die).map (·.sum) = (convPow die n.toNat).map clamp := by
  rw [← rollDice_sum, PMF.map_comp]
  apply map_congr_support
  intro r hr
  obtain ⟨hl, hh, hlen, hmem⟩ := mem_support_rollDice hr
  exact Record.sum_eq_clamp r hl hh least greatest
    (fun x hx => (hdie x (hmem x hx)).1) (fun x hx => (hdie x (hmem x hx)).2)
    (by rw [hlen]; exact test)

end Xdy.Spec
