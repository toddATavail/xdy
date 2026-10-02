import Std.Data.HashMap

/-!
# Finite distributions

A finite distribution assigns a positive natural weight to each of finitely
many outcomes. The probability of an outcome is its weight over the total of
all the weights, so the weights are an exact distribution over one total, and
no probability is ever rounded. Compiled Lean backs `Nat` with GMP, so the
weights may grow as large as they need to.

Equal outcomes always merge, adding their weights. Merging is what keeps the
oracle's state space small: after a draw, machine states that became equal
are carried as one.

## Why not a monad

`bind` needs to merge equal outcomes, so the outcomes must support `BEq` and
`Hashable`, and a `Monad` instance cannot demand that of its type argument.
`pure` and `bind` are therefore ordinary functions.
-/

namespace Xdy

open Std (HashMap)

/-- A finite distribution over outcomes of type `α`, as positive weights over
their sum. -/
structure Dist (α : Type) [BEq α] [Hashable α] where
  /-- The weight of each possible outcome. Every weight is positive; an
  impossible outcome is absent. -/
  weights : HashMap α Nat := {}

namespace Dist

variable {α β : Type} [BEq α] [Hashable α] [BEq β] [Hashable β]

/-- Two distributions are equal when they give every outcome the same weight.
Distributions with equal probabilities but different totals are unequal; see
`lowestTerms`. -/
instance : BEq (Dist α) where
  beq d e := d.weights == e.weights

/--
Answer the weight of an outcome.

# Parameters
- `d`: The distribution.
- `a`: The outcome.

# Returns
The weight of `a`, or `0` if `a` is impossible.
-/
def weight (d : Dist α) (a : α) : Nat := d.weights.getD a 0

/--
Answer the total of the weights, the denominator of every probability.

# Parameters
- `d`: The distribution.

# Returns
The sum of the weights of every outcome.

# Examples
```lean
#guard (Dist.uniform [1, 2, 3]).total == 3
```
-/
def total (d : Dist α) : Nat := d.weights.fold (fun t _ w => t + w) 0

/--
Add weight to an outcome, merging it with any weight already there.

# Parameters
- `d`: The distribution.
- `a`: The outcome.
- `w`: The weight to add.

# Returns
The distribution with the weight of `a` increased by `w`.

# Examples
```lean
#guard ((Dist.pure 7).add 7 2).weight 7 == 3
#guard ((Dist.pure 7).add 8 2).total == 3
```
-/
def add (d : Dist α) (a : α) (w : Nat) : Dist α :=
  ⟨d.weights.insert a (d.weight a + w)⟩

/--
The distribution that is certain of one outcome.

# Parameters
- `a`: The outcome.

# Returns
The distribution giving `a` weight `1`.

# Examples
```lean
#guard (Dist.pure 7).weight 7 == 1
#guard (Dist.pure 7).total == 1
```
-/
def pure (a : α) : Dist α := ({} : Dist α).add a 1

/--
The distribution that draws one of the listed outcomes, each position equally
likely. An outcome listed more than once is correspondingly more likely.

# Parameters
- `as`: The outcomes, which must not be empty.

# Returns
The distribution giving each outcome weight equal to the number of times it
is listed.

# Notes
An empty list yields the empty distribution, which describes nothing; the
interpreter never asks for one.

# Examples
```lean
#guard (Dist.uniform [1, 2, 2]).weight 2 == 2
#guard (Dist.uniform [1, 2, 2]).total == 3
```
-/
def uniform (as : List α) : Dist α := as.foldl (fun d a => d.add a 1) {}

/--
Transform every outcome, merging outcomes that the transformation makes
equal.

# Parameters
- `f`: The transformation.
- `d`: The distribution.

# Returns
The distribution of `f a`, for `a` drawn from `d`, over the same total.

# Examples
```lean
#guard ((Dist.uniform [1, 2, 3]).map (· % 2)).weight 1 == 2
```
-/
def map (f : α → β) (d : Dist α) : Dist β :=
  d.weights.fold (fun e a w => e.add (f a) w) {}

/--
Draw an outcome, then continue with a distribution that depends on it.

# Parameters
- `d`: The distribution of the first draw.
- `k`: The continuation, from each outcome of `d` to a distribution of
  final outcomes. It must never answer the empty distribution.

# Returns
The distribution of final outcomes, over the total of `d` times the least
common multiple of the totals of the continuations.

# Notes
The continuations can answer distributions over different totals, e.g., when
the number of faces of a die depends on an earlier roll. To keep one total,
each continuation's weights are scaled up to the least common multiple `l` of
their totals: an outcome `a` of weight `w`, whose continuation has total `t`,
contributes each of its final outcomes with its weight times `w * (l / t)`.
The probability of each final outcome is then exact, and the scaling is
undone, if it can be, by `lowestTerms`.

# Examples
```lean
-- Roll a die whose faces are 1D2: half the time a 1D1, half the time a 1D2.
#guard ((Dist.uniform [1, 2]).bind fun n => Dist.uniform (List.range' 1 n))
  == Dist.uniform [1, 1, 1, 2]
```
-/
def bind (d : Dist α) (k : α → Dist β) : Dist β :=
  let branches := d.weights.fold (fun bs a w => (w, k a) :: bs) []
  let l := branches.foldl (fun l (_, e) => l.lcm e.total) 1
  branches.foldl
    (fun acc (w, e) =>
      let scale := w * (l / e.total)
      e.weights.fold (fun acc b v => acc.add b (scale * v)) acc)
    {}

/--
Reduce the weights to lowest terms, dividing each by their greatest common
divisor. The probabilities are unchanged, and distributions with equal
probabilities become equal.

# Parameters
- `d`: The distribution.

# Returns
The distribution with the same probabilities over the least possible total.

# Examples
```lean
#guard (Dist.uniform [1, 1, 2, 2]).lowestTerms == Dist.uniform [1, 2]
```
-/
def lowestTerms (d : Dist α) : Dist α :=
  let g := d.weights.fold (fun g _ w => g.gcd w) 0
  ⟨d.weights.map fun _ w => w / g⟩

/--
List the outcomes and their weights in ascending order of outcome.

# Parameters
- `d`: The distribution.

# Returns
Each possible outcome with its weight, least outcome first.

# Examples
```lean
#guard (Dist.uniform [3, 1, 3]).toList == [(1, 1), (3, 2)]
```
-/
def toList [Ord α] (d : Dist α) : List (α × Nat) :=
  d.weights.toList.mergeSort fun (a, _) (b, _) => compare a b |>.isLE

end Dist

end Xdy
