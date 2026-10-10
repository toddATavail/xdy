import XdySpec.Outcomes

/-!
# Multisets of chains

The forward pass in `xdy/src/distribution/propagation.rs` keeps the chains of
a record whole only where a later reroll or explode reads them, and once a
drop has made its chains unlike, or a split has fixed them, it keeps the
multisets of its chains still kept (`Roll::Stacks` in
`propagation/chain.rs`). This module mirrors the operations by which the pass
reaches those multisets, which the explodes and rerolls of chains then
transform:

- `multisets` mirrors `multisets` in `propagation/record.rs`, which answers
  every multiset of `n` independent draws of a die of faces of any type, its
  results or its chains, as `Outcomes.outcomeStates` grows them; over their
  weights, they are the multiset of `n` independent draws
  (`Outcomes.weighted_outcomeStates`).
- `slice` mirrors `slice`, which answers the chains still kept of a multiset of
  chains after its drops: each chain counts as one die, worth its total, since
  drops and keeps order the chains by `Xdy.Chain.le`, their totals first.
  `keptPool` mirrors `kept_pool`, which slices every multiset of a pool.
- `transformPool` mirrors `transform_pool`, which slices every multiset of a
  pool and draws each copy of each chain still kept from the law of what it
  becomes, padding each multiset to the most kept dice of any;
  `transformStacks` mirrors `transform_stacks` on stacks whose chains are still
  themselves, which defers the law of what each chain becomes instead; and
  `joined` mirrors `joined`, which draws each copy of each chain of stacks from
  the law of what it has become.
- `Roll.stacks` mirrors `Roll::stacks`, which answers every outcome of a roll
  as multisets of chains: the multisets of the law of i.i.d. chains, the
  joined stacks, or the outcomes of any other roll, each result the chain of
  one die.

The Rust keeps the distinct values of each multiset in ascending order, which
`slice` reads by position. A law of chains here need not ascend, so the
multisets of chains are unordered: `slice` sorts the chains of a multiset by
`Xdy.Chain.le` before it takes the window, and multisets join by
concatenation. The Rust merges the multisets that coincide, adding their
weights; this lists them as they come, which weighs every multiset alike.

`XdySpec/Stacks.lean` proves that these multisets are the laws of the chains
still kept of the records that a roll fills.
-/

namespace Xdy.Spec

open Xdy (Chain)

/--
Every multiset of `n` independent draws of a die, weighed by the number of
branches of the draws that reach it. Mirrors `multisets` in
`propagation/record.rs`.

# Type parameters
- `α`: The type of the faces: results, or chains.

# Parameters
- `die`: The distinct faces of the die, each with its weight.
- `n`: The number of draws.

# Returns
The multisets, each of the distinct faces drawn, in the order of the die,
with their positive numbers of copies, and its weight. Over their weights,
they are the multiset of `n` independent draws of the die
(`Outcomes.weighted_outcomeStates`).

# Examples
```lean
#guard multisets [((1 : Int), 1), (2, 2)] 2 ==
  [([(1, 2)], 1), ([(1, 1), (2, 1)], 4), ([(2, 2)], 4)]
```
-/
def multisets {α : Type} (die : List (α × Nat)) (n : Nat) :
    List (List (α × Nat) × Nat) :=
  Outcomes.outcomeStates die n

/--
Each distinct chain of a list, with the number of times that it appears.

# Parameters
- `cs`: The chains, possibly repeated.

# Returns
The distinct chains, each with its positive count, which repeat to the chains
in some order.

# Examples
```lean
#guard group [.single 2, .single 1, .single 2] ==
  [(.single 1, 1), (.single 2, 2)]
```
-/
def group (cs : List Chain) : List (Chain × Nat) :=
  cs.dedup.map fun c => (c, cs.count c)

/--
The chains still kept of a multiset of chains, after its lowest `lowest` and
its highest `highest` are dropped, each chain counted as one die worth its
total. Mirrors `slice` in `propagation/chain.rs`.

# Parameters
- `chains`: The distinct chains, each with its number of copies.
- `lowest`: The number of lowest chains dropped.
- `highest`: The number of highest chains dropped.

# Returns
The distinct chains still kept, each with its positive number of copies.

# Notes
The Rust reads the kept window by position from distinct chains in ascending
order. These chains need not ascend, so this sorts them by `Xdy.Chain.le`, as
`Record.keptChains` does, takes the window, and counts the copies of each
chain in it.

# Examples
```lean
#guard slice [(.single 3, 1), (.single 1, 2), (.single 2, 1)] 1 1 ==
  [(.single 1, 1), (.single 2, 1)]
```
-/
def slice (chains : List (Chain × Nat)) (lowest highest : Nat) :
    List (Chain × Nat) :=
  let sorted := (Roll.expand chains).mergeSort Chain.le
  group ((sorted.drop lowest).take (sorted.length - lowest - highest))

/--
The multisets of the chains still kept of multisets of chains, after their
lowest `lowest` and their highest `highest` are dropped, as the pool of stacks
whose chains are still themselves. Mirrors `kept_pool` in
`propagation/chain.rs`.

# Parameters
- `outcomes`: The multisets of the chains, each with its branches.
- `lowest`: The number of lowest chains dropped.
- `highest`: The number of highest chains dropped.

# Returns
The multisets of the chains still kept, each with its branches.

# Notes
The Rust merges the multisets that coincide, adding their branches; this
lists them as they come.

# Examples
```lean
#guard keptPool [([(.single 1, 1), (.single 4, 1)], 2)] 1 0 ==
  [([(.single 4, 1)], 2)]
```
-/
def keptPool (outcomes : List (List (Chain × Nat) × Nat))
    (lowest highest : Nat) : List (List (Chain × Nat) × Nat) :=
  outcomes.map fun e => (slice e.1 lowest highest, e.2)

/--
Transform each multiset of chains, after its lowest `lowest` and its highest
`highest` chains are dropped, by the law of what each chain still kept
becomes, and answer the multisets of what the chains still kept become.
Mirrors `transform_pool` in `propagation/chain.rs`: each group of `c` copies
of a chain becomes each multiset of `c` independent draws of its law, and the
multisets of the groups join; a multiset whose chains hold `d` kept dice in
all is padded to the most of any, `D`, by `G^(D − d)`, so that every multiset
weighs the same in all.

# Type parameters
- `T`: The type of what a chain becomes: a chain, or a total.

# Parameters
- `outcomes`: The multisets of the chains, each with its branches.
- `lowest`: The number of lowest chains dropped.
- `highest`: The number of highest chains dropped.
- `padding`: The branches that a reroll adds to each kept die, `G`, or one
  for an explode.
- `becomes`: The law of what a chain becomes, each value with its branches,
  and the number of kept dice of the chain by which a reroll pads it.

# Returns
The new multisets, each with its branches.

# Notes
The Rust joins multisets as sorted maps, and merges the multisets that
coincide; this joins them by concatenation, and lists them as they come.

# Examples
```lean
-- Each kept chain of one die becomes itself or a chain of two.
#guard transformPool [([(.single 1, 1), (.single 2, 1)], 1)] 1 0 1
    (fun ch => ([(ch, 1), (.live [6] ch.last, 1)], 0)) ==
  [([(.single 2, 1)], 1), ([(.live [6] 2, 1)], 1)]
```
-/
def transformPool {T : Type} (outcomes : List (List (Chain × Nat) × Nat))
    (lowest highest padding : Nat)
    (becomes : Chain → List (T × Nat) × Nat) :
    List (List (T × Nat) × Nat) :=
  let kept := outcomes.map fun e =>
    let groups := slice e.1 lowest highest
    (groups, (groups.map fun g => (becomes g.1).2 * g.2).sum, e.2)
  let most := (kept.map (·.2.1)).foldl Max.max 0
  kept.flatMap fun k =>
    k.1.foldl
      (fun soFar g => soFar.flatMap fun e =>
        (multisets (becomes g.1).1 g.2).map fun o => (e.1 ++ o.1, e.2 * o.2))
      [([], k.2.2 * padding ^ (most - k.2.1))]

/--
Transform stacks whose chains are still themselves by the law of what each
chain becomes, chain by chain, without joining them, into stacks that hold
the law of what each distinct chain of the pool has become. Mirrors
`transform_stacks` in `propagation/chain.rs` on stacks without laws, such as
`kept_pool` leaves: a multiset whose chains hold `d` kept dice in all is
padded to the most of any, `D`, by `G^(D − d)`.

# Parameters
- `pool`: The multisets of the stacks, each with its branches.
- `padding`: The branches that a reroll adds to each kept die, `G`, or one
  for an explode.
- `becomes`: The law of what a chain becomes, each chain with its branches,
  and the number of kept dice of the chain by which a reroll pads it.

# Returns
The new stacks.

# Notes
The Rust keeps the laws in a map ordered by chain; this lists each distinct
chain of the pool once. The stacks of a pool whose chains have already become
something are left to later proofs.

# Examples
```lean
#guard transformStacks [([(.single 2, 1)], 3)] 1
    (fun ch => ([(ch, 1), (.live [6] ch.last, 1)], 0)) ==
  .stacked [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])]
```
-/
def transformStacks (pool : List (List (Chain × Nat) × Nat)) (padding : Nat)
    (becomes : Chain → List (Chain × Nat) × Nat) : Roll :=
  let dice := fun (chains : List (Chain × Nat)) =>
    (chains.map fun g => (becomes g.1).2 * g.2).sum
  let most := (pool.map (dice ·.1)).foldl Max.max 0
  .stacked (pool.map fun e => (e.1, e.2 * padding ^ (most - dice e.1)))
    ((pool.flatMap fun e => e.1.map (·.1)).dedup.map fun ch =>
      (ch, (becomes ch).1))

/--
Join stacks: each multiset of the chains that they hold, with its branches,
each copy of each chain of each multiset of the pool drawn independently from
the law of what it has become. Mirrors `joined` in `propagation/chain.rs`.

# Parameters
- `pool`: The multisets of the stacks, each with its branches.
- `laws`: The law of what each distinct chain of the pool has become, or none
  for a chain that is still itself.

# Returns
The multisets, each with its branches: the pool itself if every chain is still
itself.

# Examples
```lean
#guard joined [([(.single 2, 1)], 3)] [] == [([(.single 2, 1)], 3)]
#guard joined [([(.single 2, 1)], 3)]
    [(.single 2, [(.single 2, 1), (.live [6] 2, 1)])] ==
  [([(.single 2, 1)], 3), ([(.live [6] 2, 1)], 3)]
```
-/
def joined (pool : List (List (Chain × Nat) × Nat))
    (laws : List (Chain × List (Chain × Nat))) :
    List (List (Chain × Nat) × Nat) :=
  if laws.isEmpty then pool
  else
    transformPool pool 0 0 1 fun ch =>
      match laws.lookup ch with
      | some law => (law, 0)
      | none => ([(ch, 1)], 0)

/--
Every outcome of a roll as multisets of chains: each multiset of the chains of
its records, weighed by the number of branches of the roll that reach it.
Mirrors `Roll::stacks` in `propagation/chain.rs`.

# Parameters
- `ρ`: The roll.

# Returns
The multisets of `count` draws of the law of i.i.d. chains, or none if
`count` is not positive; the joined stacks; or the outcomes of any other roll,
each result the chain of one die.

# Examples
```lean
#guard (Roll.range 1 2).stacks ==
  [([(.single 1, 1)], 1), ([(.single 2, 1)], 1)]
#guard (Roll.chains 1 [(.live [6] 2, 1)]).stacks == [([(.live [6] 2, 1)], 1)]
```
-/
def Roll.stacks : Roll → List (List (Chain × Nat) × Nat)
  | .chains count law => multisets law count.toNat
  | .stacked ms laws => joined ms laws
  | ρ => ρ.outcomes.map fun e => (e.1.map fun p => (.single p.1, p.2), e.2)

end Xdy.Spec
