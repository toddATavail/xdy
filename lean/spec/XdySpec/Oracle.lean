import XdySpec.Law
import XdySpec.Semantics

/-!
# The oracle meets the specification

The executable oracle, `Xdy.Function.run`, answers exactly the specification's
law of a function's result wherever it answers: `run_eq` says that the
oracle's weights over its total are the specification's probabilities, and
that each fails, with the same message, exactly when the other does, unless
the oracle refuses an explode that may draw without bound, whose law the
specification gives, with infinitely many outcomes.

The two differ in how they hold rolling records. The specification keeps
results in the order rolled, as Rust does; the oracle sorts them as they
arrive, so that states differing only in the order of their rolls merge. So
the proof relates each oracle distribution of states to the specification's
law of states with every record sorted, read by `State.toOracle`. That
relation holds of the initial state, whose records are empty; each
instruction keeps it, since sorting commutes with rolling, dropping, keeping,
rerolling, exploding and summing, by `hasLaw_step`, wherever the oracle does
not refuse it; so it holds of the final states, unless the oracle refuses, by
`folds_foldlM`; and the returned operand reads only registers, which sorting
leaves alone. A record that holds chains sorts its chains, as the oracle keeps
them, and a reroll or explode reads them only through its kept chains, which
sorting leaves alone (`Record.keptChains_toOracle`).

Each instruction keeps the relation because `Dist.hasLaw_bind` needs the law
of each step only at the sorted reading of a state, and the oracle's step
from the sorted state has the law of the specification's step from the
unsorted one, sorted: the order of a record's results changes nothing that
any instruction can observe.
-/

namespace Xdy.Spec

/-! ## Reading states sorted -/

namespace State

/--
Sorting leaves every operand's value alone.

# Parameters
- `s`: The state.
- `op`: The operand.
-/
theorem value_toOracle (s : State) (op : AddressingMode) :
    s.toOracle.value op = s.value op := by
  cases op <;> rfl

/--
Reading a record of the sorted state reads the record sorted.

# Parameters
- `s`: The state.
- `r`: The rolling record.
-/
theorem record_toOracle (s : State) (r : Nat) :
    s.toOracle.record r = (s.record r).toOracle := by
  simp only [Xdy.State.record, record, toOracle, Array.getD_eq_getD_getElem?,
    Array.getElem?_map]
  cases s.records[r]? <;> simp [Record.toOracle_empty]

/--
Sorting leaves alone whether an instruction may explode without bound, which
reads a record only through its kept chains and its die.

# Parameters
- `s`: The state.
- `i`: The instruction.
-/
theorem unbounded_toOracle (s : State) (i : Instruction) :
    Xdy.unbounded s.toOracle i = unbounded s i := by
  cases i <;> try rfl
  case explode d p l =>
    simp only [Xdy.unbounded, unbounded, record_toOracle,
      Record.keptChains_toOracle,
      show s.toOracle.value = s.value from funext (value_toOracle s)]
    rfl

/--
Sorting commutes with replacing a record.

# Parameters
- `s`: The state.
- `r`: The rolling record.
- `rec`: Its new contents.
-/
theorem setRecord_toOracle (s : State) (r : Nat) (rec : Record) :
    (s.setRecord r rec).toOracle = s.toOracle.setRecord r rec.toOracle := by
  simp only [toOracle, setRecord, Xdy.State.setRecord, Xdy.State.mk.injEq,
    true_and]
  apply Array.ext
  · simp
  · intro i _ _
    simp only [Array.getElem_map, Array.getElem_modify]
    split <;> rfl

/--
Sorting commutes with writing a register.

# Parameters
- `s`: The state.
- `r`: The register.
- `v`: The value.
-/
theorem setRegister_toOracle (s : State) (r : Nat) (v : Int) :
    (s.setRegister r v).toOracle = s.toOracle.setRegister r v := rfl

/--
Sorting the specification's reading of an oracle state gives it back, if its
records are sorted, as the oracle keeps them.

# Parameters
- `s`: The oracle's state.

# Hypotheses
- `sorted`: The results of every record are sorted in ascending order, and its
  chains by `Xdy.Chain.le`.
-/
theorem toOracle_ofOracle (s : Xdy.State)
    (sorted : ∀ r ∈ s.records, r.results.Pairwise (· ≤ ·) ∧
      r.chains.Pairwise (fun a b => a.le b = true)) :
    (ofOracle s).toOracle = s := by
  cases s with
  | mk registers records =>
  simp only [toOracle, ofOracle, Xdy.State.mk.injEq, true_and]
  apply Array.ext
  · simp
  · intro i _ _
    simp only [Array.getElem_map]
    exact Record.toOracle_ofOracle _ (sorted _ (Array.getElem_mem _)).1
      (sorted _ (Array.getElem_mem _)).2

end State

/-! ## Draws -/

/--
The integers from `a` to `b`, as `Finset.Icc` holds them, are those that the
oracle lists: `a` plus each natural below the width of the range.

# Parameters
- `a`: The least integer.
- `b`: The greatest integer.
-/
theorem icc_val (a b : ℤ) :
    (Finset.Icc a b).val =
      ((List.range (b - a + 1).toNat).map fun (i : ℕ) => a + i : List ℤ) := by
  rw [Int.Icc_eq_finset_map, Finset.map_val, Finset.range_val, Multiset.range,
    Multiset.map_coe]
  congr 2
  rw [show b + 1 - a = b - a + 1 by omega]

/--
The oracle's range has the law of the specification's, sorted.

# Parameters
- `a`: The least result.
- `b`: The greatest result.
-/
theorem hasLaw_rollRange (a b : Int) :
    (Xdy.rollRange a b).HasLaw ((rollRange a b).map Record.toOracle) := by
  unfold Xdy.rollRange rollRange
  by_cases h : b < a
  · simp only [h, ↓reduceIte, ↓reduceDIte, PMF.pure_map]
    convert Dist.hasLaw_pure _ <;>
      first | infer_instance | simp [Record.toOracle, Record.sorted]
  · simp only [h, ↓reduceIte, ↓reduceDIte, PMF.map_comp]
    have hne :
        ((List.range (b - a + 1).toNat).map fun (i : ℕ) => a + i) ≠ [] := by
      simp; omega
    convert Dist.hasLaw_map _ (Dist.hasLaw_uniform _ hne) using 2
    all_goals first
      | infer_instance
      | (congr 1; exact icc_val a b)
      | (funext x; simp [Record.toOracle, Record.sorted])

/--
The oracle's standard die has the specification's law.

# Parameters
- `n`: The number of faces.
-/
theorem hasLaw_standardDie (n : Int) :
    (Xdy.standardDie n).HasLaw (standardDie n) := by
  unfold Xdy.standardDie standardDie
  by_cases h : n ≤ 0
  · simp only [h, ↓reduceIte, ↓reduceDIte]
    exact Dist.hasLaw_pure _
  · simp only [h, ↓reduceIte, ↓reduceDIte]
    have hne :
        ((List.range n.toNat).map fun (i : ℕ) => (i : Int) + 1) ≠ [] := by
      simp; omega
    convert Dist.hasLaw_uniform _ hne using 2
    rw [icc_val, show n - 1 + 1 = n by omega]
    congr 2
    funext i; omega

/--
The oracle's custom die has the specification's law.

# Parameters
- `faces`: The faces.
-/
theorem hasLaw_customDie (faces : List Int) :
    (Xdy.customDie faces).HasLaw (customDie faces) := by
  unfold Xdy.customDie customDie
  by_cases h : faces = []
  · subst h; exact Dist.hasLaw_pure _
  · simp only [h, ↓reduceIte, ↓reduceDIte, List.isEmpty_eq_false_iff.mpr h,
      Bool.false_eq_true]
    exact Dist.hasLaw_uniform _ h

/--
The oracle's set of dice has the law of the specification's, sorted: rolling
in order and sorting at the end is sorting as each die arrives.

# Parameters
- `c`: The number of dice.
- `die`: The oracle's die.
- `pdie`: The specification's die.

# Hypotheses
- `h`: The oracle's die has the specification's law.
-/
theorem hasLaw_rollDice (c : Int) (die : Unit → Dist Int) (pdie : PMF Int)
    (h : (die ()).HasLaw pdie) :
    (Xdy.rollDice c die).HasLaw ((rollDice c pdie).map Record.toOracle) := by
  unfold Xdy.rollDice rollDice
  by_cases hc : c ≤ 0
  · have : c.toNat = 0 := by omega
    simp only [hc, ↓reduceIte, this, Nat.repeat, PMF.pure_map,
      Record.toOracle_empty]
    exact Dist.hasLaw_pure _
  · simp only [hc, ↓reduceIte]
    generalize c.toNat = n
    induction n with
    | zero =>
      simp only [Nat.repeat, PMF.pure_map, Record.toOracle_empty]
      exact Dist.hasLaw_pure _
    | succ n ih =>
      simp only [Nat.repeat]
      refine Dist.hasLaw_bind ih fun r _ => ?_
      rw [PMF.map_comp, show Record.toOracle ∘ r.push = r.toOracle.push from
        funext (Record.toOracle_push r)]
      exact Dist.hasLaw_map _ h

/-! ## Rerolls -/

/--
The oracle's redraw has the specification's law, if the die and the result
before it do.

# Parameters
- `c`: The criterion.
- `die`: The oracle's die.
- `pdie`: The specification's die.
- `d`: The oracle's result before the redraw.
- `p`: The specification's result before the redraw.

# Hypotheses
- `hdie`: The oracle's die has the specification's law.
- `h`: The oracle's result has the specification's law.
-/
theorem hasLaw_redraw (c : Criterion) {die : Dist Int} {pdie : PMF Int}
    {d : Dist Int} {p : PMF Int} (hdie : die.HasLaw pdie) (h : d.HasLaw p) :
    (Xdy.redraw c die d).HasLaw (redraw c pdie p) := by
  have := Dist.hasLaw_bind (g := id) (h := id) (q := fun x =>
      if c.matches x then pdie else PMF.pure x)
    (k := fun x => if c.matches x then die else .pure x)
    (by rw [PMF.map_id]; exact h) fun x _ => by
      rw [PMF.map_id]
      by_cases hm : c.matches x
      · simpa [hm] using hdie
      · simpa [hm] using Dist.hasLaw_pure x
  rw [PMF.map_id] at this
  exact this

/--
The oracle's limited replacement has the specification's law, if the die
does.

# Parameters
- `c`: The criterion.
- `die`: The oracle's die.
- `pdie`: The specification's die.
- `limit`: The number of times that each die may be rerolled.

# Hypotheses
- `hdie`: The oracle's die has the specification's law.
-/
theorem hasLaw_limitedReplacement (c : Criterion) {die : Dist Int}
    {pdie : PMF Int} (limit : Nat) (hdie : die.HasLaw pdie) :
    (Xdy.limitedReplacement c die limit).HasLaw
      (limitedReplacement c pdie limit) := by
  unfold Xdy.limitedReplacement limitedReplacement
  induction limit - 1 with
  | zero => exact hdie
  | succ n ih => exact hasLaw_redraw c hdie ih

/--
The oracle's reroll of kept results has the law of the specification's,
sorted, from any records before them that do: rerolling in place and sorting
at the end is sorting as each result arrives.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `rep`: The oracle's replacement.
- `prep`: The specification's replacement.
- `kept`: The results to reroll.
- `d`: The oracle's records before them.
- `p`: The specification's records before them.

# Hypotheses
- `hrep`: The oracle's replacement has the specification's law.
- `h`: `d` has the law of `p`, sorted.
-/
theorem hasLaw_foldl_rerollKept (rerolled : Int → Bool)
    {rep : Thunk (Dist Int)} {prep : PMF Int} (hrep : rep.get.HasLaw prep)
    (kept : List Int) {d : Dist Xdy.Record} {p : PMF Record}
    (h : d.HasLaw (p.map Record.toOracle)) :
    (kept.foldl
        (fun d x =>
          if rerolled x then d.bind fun r => rep.get.map r.push
          else d.map (·.push x)) d).HasLaw
      ((kept.foldl
        (fun p x =>
          if rerolled x then p.bind fun r => prep.map r.push
          else p.map (·.push x)) p).map Record.toOracle) := by
  induction kept generalizing d p with
  | nil => exact h
  | cons x kept ih =>
    simp only [List.foldl_cons]
    apply ih
    by_cases hx : rerolled x
    · simp only [hx, ↓reduceIte]
      refine Dist.hasLaw_bind h fun r _ => ?_
      rw [PMF.map_comp, show Record.toOracle ∘ r.push = r.toOracle.push from
        funext (Record.toOracle_push r)]
      exact Dist.hasLaw_map _ hrep
    · simp only [hx, Bool.false_eq_true, ↓reduceIte]
      rw [PMF.map_comp, show Record.toOracle ∘ (·.push x) =
          (·.push x) ∘ Record.toOracle from
        funext fun r => Record.toOracle_push r x, ← PMF.map_comp]
      exact Dist.hasLaw_map _ h

/--
The oracle's reroll of the kept results of a record has the law of the
specification's, sorted.

# Parameters
- `die`: The die of the record.
- `kept`: The kept results.
- `rerolled`: Whether the reroll rerolls a result.
- `rep`: The oracle's replacement.
- `prep`: The specification's replacement.

# Hypotheses
- `hrep`: The oracle's replacement has the specification's law.
-/
theorem hasLaw_rerollKept (die : Die) (kept : List Int)
    (rerolled : Int → Bool) {rep : Thunk (Dist Int)} {prep : PMF Int}
    (hrep : rep.get.HasLaw prep) :
    (Xdy.rerollKept die kept rerolled rep).HasLaw
      ((rerollKept die kept rerolled prep).map Record.toOracle) := by
  refine hasLaw_foldl_rerollKept rerolled hrep kept ?_
  rw [PMF.pure_map, show Record.toOracle { die } = { die } by
    simp [Record.toOracle, Record.sorted]]
  exact Dist.hasLaw_pure _

/--
Binding has the law of binding, without transforming the outcomes: the plain
law of `bind`.

# Parameters
- `d`: The oracle's first draw.
- `k`: The oracle's continuation.
- `p`: The specification's first draw.
- `q`: The specification's continuation.

# Hypotheses
- `hd`: `d` has law `p`.
- `hk`: For each possible outcome `a` of `p`, `k a` has law `q a`.
-/
private theorem hasLaw_bind_id {α β : Type} [BEq α] [LawfulBEq α]
    [Hashable α] [BEq β] [LawfulBEq β] [Hashable β] {d : Dist α}
    {k : α → Dist β} {p : PMF α} {q : α → PMF β} (hd : d.HasLaw p)
    (hk : ∀ a ∈ p.support, (k a).HasLaw (q a)) :
    (d.bind k).HasLaw (p.bind q) := by
  have := Dist.hasLaw_bind (g := id) (h := id) (by rwa [PMF.map_id]) fun a ha =>
    by rw [PMF.map_id]; exact hk a ha
  rwa [PMF.map_id] at this

/--
A fold whose steps each keep a law keeps it: if the oracle's and the
specification's steps have the same laws for each element, then so do their
folds, from starts of the same law.

# Parameters
- `l`: The list folded.
- `f`: The oracle's step.
- `g`: The specification's step.
- `d`: The oracle's start.
- `p`: The specification's start.

# Hypotheses
- `hfg`: Each step, of an element of `l`, keeps the law.
- `h`: The starts have the same law.
-/
private theorem hasLaw_foldl_id {γ β : Type} [BEq β] [LawfulBEq β]
    [Hashable β]
    (l : List γ) {f : Dist β → γ → Dist β} {g : PMF β → γ → PMF β}
    (hfg : ∀ d p x, x ∈ l → d.HasLaw p → (f d x).HasLaw (g p x))
    {d : Dist β} {p : PMF β} (h : d.HasLaw p) :
    (l.foldl f d).HasLaw (l.foldl g p) := by
  induction l generalizing d p with
  | nil => exact h
  | cons x l ih =>
    exact ih (fun d p y hy => hfg d p y (List.mem_cons_of_mem x hy))
      (hfg d p x List.mem_cons_self h)

/--
The oracle's reroll of the dice of a chain has the specification's law.

# Parameters
- `rerolled`: Whether the reroll rerolls a result.
- `rep`: The oracle's replacement.
- `prep`: The specification's replacement.
- `c`: The chain.

# Hypotheses
- `hrep`: The oracle's replacement has the specification's law.
-/
theorem hasLaw_rerollChain (rerolled : Int → Bool) {rep : Thunk (Dist Int)}
    {prep : PMF Int} (hrep : rep.get.HasLaw prep) (c : Chain) :
    (Xdy.rerollChain rerolled rep c).HasLaw (rerollChain rerolled prep c) := by
  have hdraw (x : Int) :
      (if rerolled x then rep.get else Dist.pure x).HasLaw
        (if rerolled x then prep else PMF.pure x) := by
    split
    · exact hrep
    · exact Dist.hasLaw_pure x
  cases c with
  | saturated v => exact Dist.hasLaw_pure _
  | live body last =>
    simp only [Xdy.rerollChain, rerollChain]
    apply hasLaw_bind_id (α := List Int) (β := Chain)
    · apply hasLaw_foldl_id (β := List Int) body _ (Dist.hasLaw_pure _)
      intro _ _ x _ h
      apply hasLaw_bind_id (α := List Int) (β := List Int) h
      intro _ _
      exact Dist.hasLaw_map _ (hdraw x)
    · intro _ _
      exact Dist.hasLaw_map _ (hdraw last)

/--
The oracle's reroll of the kept chains of a record has the law of the
specification's, sorted.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains.
- `rerolled`: Whether the reroll rerolls a result.
- `rep`: The oracle's replacement.
- `prep`: The specification's replacement.

# Hypotheses
- `hrep`: The oracle's replacement has the specification's law.
-/
theorem hasLaw_rerollChains (die : Die) (kept : List Chain)
    (rerolled : Int → Bool) {rep : Thunk (Dist Int)} {prep : PMF Int}
    (hrep : rep.get.HasLaw prep) :
    (Xdy.rerollChains die kept rerolled rep).HasLaw
      ((rerollChains die kept rerolled prep).map Record.toOracle) := by
  unfold Xdy.rerollChains rerollChains
  rw [PMF.map_comp, show Record.toOracle ∘ Record.ofChains die =
    Xdy.Record.ofChains die from funext (Record.toOracle_ofChains die)]
  apply Dist.hasLaw_map
  apply hasLaw_foldl_id (β := List Chain) kept _ (Dist.hasLaw_pure _)
  intro _ _ c _ h
  apply hasLaw_bind_id (α := List Chain) (β := List Chain) h
  intro _ _
  exact Dist.hasLaw_map _ (hasLaw_rerollChain _ hrep c)

/--
The oracle's reroll of each kept die of the sorted record has the law of the
specification's, sorted.

# Parameters
- `r`: The specification's record.
- `rerolled`: Whether the reroll rerolls a result.
- `rep`: The oracle's replacement.
- `prep`: The specification's replacement.

# Hypotheses
- `hrep`: The oracle's replacement has the specification's law.
-/
theorem hasLaw_rerollEach (r : Record) (rerolled : Int → Bool)
    {rep : Thunk (Dist Int)} {prep : PMF Int} (hrep : rep.get.HasLaw prep) :
    (Xdy.rerollEach r.toOracle rerolled rep).HasLaw
      ((rerollEach r rerolled prep).map Record.toOracle) := by
  unfold Xdy.rerollEach rerollEach
  rw [Record.chains_isEmpty_toOracle]
  split
  · rw [Record.kept_toOracle]
    exact hasLaw_rerollKept _ _ _ hrep
  · rw [Record.keptChains_toOracle]
    exact hasLaw_rerollChains _ _ _ hrep

/--
The oracle's reroll of the sorted record has the law of the specification's,
sorted.

# Parameters
- `r`: The specification's record.
- `c`: The criterion.
- `limit`: The number of times that each die may be rerolled, or `none` for
  no limit.
-/
theorem hasLaw_reroll (r : Record) (c : Criterion) (limit : Option Nat) :
    (Xdy.reroll r.toOracle c limit).HasLaw
      ((reroll r c limit).map Record.toOracle) := by
  have hdie : r.toOracle.die = r.die := rfl
  cases limit with
  | none =>
    unfold Xdy.reroll reroll
    rw [hdie]
    by_cases h : r.die.facesWithin c
    · simp only [h, ↓reduceIte, ↓reduceDIte, Record.chains_isEmpty_toOracle]
      split
      · have hs : (r.kept.filter (!c.matches ·)).mergeSort
              (fun a b => decide (a ≤ b)) = r.kept.filter (!c.matches ·) :=
          List.mergeSort_of_pairwise
            (by simpa using (Record.kept_pairwise r).filter _)
        rw [Record.kept_toOracle, PMF.pure_map]
        simp only [Record.toOracle, Record.sorted, hs, List.mergeSort_nil]
        exact Dist.hasLaw_pure _
      · rw [Record.keptChains_toOracle, PMF.pure_map, Record.toOracle_ofChains]
        exact Dist.hasLaw_pure _
    · simp only [h, Bool.false_eq_true, ↓reduceIte, ↓reduceDIte]
      refine hasLaw_rerollEach _ _ ?_
      exact Dist.hasLaw_uniform _ (by
        simpa using unmatched_ne_zero h)
  | some limit =>
    unfold Xdy.reroll reroll
    rw [hdie]
    exact hasLaw_rerollEach _ _
      (hasLaw_limitedReplacement c limit (hasLaw_customDie _))

/--
The oracle's extension of a chain has the specification's law, if the die
does.

# Parameters
- `c`: The criterion.
- `die`: The oracle's die.
- `pdie`: The specification's die.
- `limit`: The number of times that each chain may be exploded.

# Hypotheses
- `hdie`: The oracle's die has the specification's law.
-/
theorem hasLaw_extension (c : Criterion) {die : Dist Int} {pdie : PMF Int}
    (limit : Nat) (hdie : die.HasLaw pdie) :
    (Xdy.extension c die limit).HasLaw (extension c pdie limit) := by
  unfold Xdy.extension extension
  induction limit - 1 with
  | zero => exact Dist.hasLaw_map _ hdie
  | succ n ih =>
    refine hasLaw_bind_id ih fun ⟨drawn, x⟩ _ => ?_
    by_cases hm : c.matches x
    · simpa [hm] using Dist.hasLaw_map _ hdie
    · simpa [hm] using Dist.hasLaw_pure (drawn, x)

/--
The oracle's explode of the kept chains of a record has the law of the
specification's, sorted, if their extensions have the same law wherever some
chain grows.

# Parameters
- `die`: The die of the record.
- `kept`: The kept chains.
- `grows`: Whether the explode grows a chain.
- `ext`: The oracle's extension.
- `pext`: The specification's extension.

# Hypotheses
- `hext`: If some kept chain grows, the oracle's extension has the
  specification's law.
-/
theorem hasLaw_explodeChains (die : Die) (kept : List Chain)
    (grows : Chain → Bool) {ext : Thunk (Dist (List Int × Int))}
    {pext : PMF (List Int × Int)}
    (hext : (∃ c ∈ kept, grows c = true) → ext.get.HasLaw pext) :
    (Xdy.explodeChains die kept grows ext).HasLaw
      ((explodeChains die kept grows pext).map Record.toOracle) := by
  unfold Xdy.explodeChains explodeChains
  rw [PMF.map_comp, show Record.toOracle ∘ Record.ofChains die =
    Xdy.Record.ofChains die from funext (Record.toOracle_ofChains die)]
  refine Dist.hasLaw_map _ (hasLaw_foldl_id kept (fun d p c hc h => ?_)
    (Dist.hasLaw_pure _))
  by_cases hg : grows c
  · simp only [hg, ↓reduceIte]
    exact hasLaw_bind_id h fun _ _ => Dist.hasLaw_map _ (hext ⟨c, hc, hg⟩)
  · simp only [hg, Bool.false_eq_true, ↓reduceIte]
    exact Dist.hasLaw_map _ h

/--
The oracle's explode of the sorted record has the law of the specification's,
sorted, unless the explode may grow a chain without bound, which the oracle
refuses.

# Parameters
- `r`: The specification's record.
- `c`: The criterion.
- `limit`: The number of times that each chain may be exploded, or `none` for
  no limit.

# Hypotheses
- `hb`: If the explode is unlimited, and does not match every face of the
  die, then it grows no kept chain.
-/
theorem hasLaw_explode (r : Record) (c : Criterion) (limit : Option Nat)
    (hb : limit = none → ¬r.die.facesWithin c = true →
      ∀ ch ∈ r.keptChains, ch.grows c = false) :
    (Xdy.explode r.toOracle c limit).HasLaw
      ((explode r c limit).map Record.toOracle) := by
  have hdie : r.toOracle.die = r.die := rfl
  cases limit with
  | none =>
    unfold Xdy.explode explode
    rw [hdie, Record.keptChains_toOracle]
    by_cases h : r.die.facesWithin c
    · simp only [h, ↓reduceIte, ↓reduceDIte, PMF.pure_map,
        Record.toOracle_ofChains]
      exact Dist.hasLaw_pure _
    · simp only [h, Bool.false_eq_true, ↓reduceIte, ↓reduceDIte]
      rw [explodeChains_congr _ _ (grows' := fun _ => false)
        fun ch hch => hb rfl h ch hch]
      exact hasLaw_explodeChains _ _ _ fun ⟨_, _, hg⟩ => absurd hg (by simp)
  | some limit =>
    unfold Xdy.explode explode
    rw [hdie, Record.keptChains_toOracle]
    exact hasLaw_explodeChains _ _ _ fun _ =>
      hasLaw_extension c limit (hasLaw_customDie _)

/-! ## Instructions -/

/--
A roll into a record of the sorted state has the law of the specification's
roll, sorted, if the rolls themselves do, and the records that they fill are
finished alike, as the dice are by recording their die.

# Parameters
- `s`: The specification's state.
- `d`: The rolling record.
- `oracle`: The oracle's roll.
- `spec`: The specification's roll.
- `g`: How the oracle finishes each record that it rolls.
- `f`: How the specification finishes each record that it rolls.

# Hypotheses
- `hfg`: Finishing a record and then sorting it is sorting it and then
  finishing it.
- `h`: The oracle's roll has the law of the specification's, sorted.
-/
private theorem hasLaw_roll (s : State) (d : Nat) (oracle : Dist Xdy.Record)
    (spec : PMF Record) (g : Xdy.Record → Xdy.Record) (f : Record → Record)
    (hfg : ∀ r, (f r).toOracle = g r.toOracle)
    (h : oracle.HasLaw (spec.map Record.toOracle)) :
    (oracle.map fun r => s.toOracle.setRecord d (g r)).HasLaw
      ((spec.map fun r => s.setRecord d (f r)).map State.toOracle) := by
  rw [PMF.map_comp, show State.toOracle ∘ (fun r => s.setRecord d (f r)) =
      (fun r => s.toOracle.setRecord d (g r)) ∘ Record.toOracle from
    funext fun r => by
      simp only [Function.comp_apply, State.setRecord_toOracle, hfg],
    ← PMF.map_comp]
  exact Dist.hasLaw_map _ h

/--
The oracle's step from the sorted state has the law of the specification's
step, sorted, unless the instruction is an explode that may draw without
bound, whose law has infinitely many outcomes, and which the oracle refuses.

# Parameters
- `s`: The specification's state.
- `i`: The instruction.

# Hypotheses
- `hb`: `i` may not explode without bound in `s`.
-/
theorem hasLaw_step (s : State) (i : Instruction)
    (hb : unbounded s i = false) :
    (Xdy.step s.toOracle i).HasLaw ((step s i).map State.toOracle) := by
  cases i with
  | rollRange d a b =>
    simp only [Xdy.step, step, State.value_toOracle]
    exact hasLaw_roll s d _ _ id id (fun _ => rfl) (hasLaw_rollRange _ _)
  | rollStandardDice d c n =>
    simp only [Xdy.step, step, State.value_toOracle]
    exact hasLaw_roll s d _ _ _ _ (fun _ => rfl)
      (hasLaw_rollDice _ _ _ (hasLaw_standardDie _))
  | rollCustomDice d c faces =>
    simp only [Xdy.step, step, State.value_toOracle]
    exact hasLaw_roll s d _ _ _ _ (fun _ => rfl)
      (hasLaw_rollDice _ _ _ (hasLaw_customDie _))
  | dropLowest d c =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRecord_toOracle,
      Record.toOracle_dropLowest, State.record_toOracle, State.value_toOracle]
    exact Dist.hasLaw_pure _
  | dropHighest d c =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRecord_toOracle,
      Record.toOracle_dropHighest, State.record_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | keepLowest d c =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRecord_toOracle,
      Record.toOracle_keepLowest, State.record_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | keepHighest d c =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRecord_toOracle,
      Record.toOracle_keepHighest, State.record_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | reroll d p l =>
    simp only [Xdy.step, step, State.record_toOracle,
      show s.toOracle.value = s.value from funext (State.value_toOracle s)]
    exact hasLaw_roll s d _ _ id id (fun _ => rfl) (hasLaw_reroll _ _ _)
  | explode d p l =>
    simp only [Xdy.step, step, State.record_toOracle,
      show s.toOracle.value = s.value from funext (State.value_toOracle s)]
    refine hasLaw_roll s d _ _ id id (fun _ => rfl) (hasLaw_explode _ _ _ ?_)
    intro hl hf ch hch
    simp only [unbounded, hl, beq_self_eq_true, hf, Bool.not_false,
      Bool.true_and, List.any_eq_false] at hb
    simpa using hb ch hch
  | sumRollingRecord d r =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRegister_toOracle,
      State.record_toOracle, Record.sum_toOracle]
    exact Dist.hasLaw_pure _
  | binary op d a b =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRegister_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | neg d a =>
    simp only [Xdy.step, step, PMF.pure_map, State.setRegister_toOracle,
      State.value_toOracle]
    exact Dist.hasLaw_pure _
  | «return» _ =>
    simp only [Xdy.step, step, PMF.pure_map]
    exact Dist.hasLaw_pure _

/--
How the oracle's run of instructions over a distribution of states agrees with
the specification's law of states after them: it answers a distribution that
has the law, sorted, or it refuses with `Xdy.unboundedMessage`.

# Parameters
- `e`: The oracle's run.
- `q`: The specification's law of states.
-/
private def Folds : Except String (Dist Xdy.State) → PMF State → Prop
  | .ok d, q => d.HasLaw (q.map State.toOracle)
  | .error m, _ => m = Xdy.unboundedMessage

/--
Running instructions keeps the relation: if the oracle's states have the law
of the specification's, sorted, then so do they after the instructions,
unless the oracle refuses an explode that may draw without bound.

# Parameters
- `is`: The instructions.
- `d`: The oracle's distribution of states before them.
- `p`: The specification's law of states before them.

# Hypotheses
- `h`: `d` has the law of `p`, sorted.
-/
private theorem folds_foldlM (is : List Instruction) (d : Dist Xdy.State)
    (p : PMF State) (h : d.HasLaw (p.map State.toOracle)) :
    Folds (is.foldlM Xdy.stepAll d)
      (is.foldl (fun p i => p.bind (step · i)) p) := by
  induction is generalizing d p with
  | nil => exact h
  | cons i is ih =>
    rw [List.foldlM_cons, List.foldl_cons]
    unfold Xdy.stepAll
    split
    · rfl
    · rename_i hany
      refine ih _ _ (Dist.hasLaw_bind h fun t ht => hasLaw_step t i ?_)
      rw [← State.unbounded_toOracle]
      exact Dist.any_eq_false_of_hasLaw (P := (Xdy.unbounded · i)) h
        (by simpa using hany) ht

/-! ## Functions -/

/--
A property that every step of a fold in `Except` keeps holds of the fold's
result, if it succeeds.

# Parameters
- `P`: The property.
- `f`: The step.
- `l`: The list folded.
- `b`: The start.
- `b'`: The result.

# Hypotheses
- `hf`: Each step that succeeds keeps `P`.
- `hb`: The start satisfies `P`.
- `h`: The fold succeeds with `b'`.
-/
private theorem foldlM_inv {α β ε : Type} (P : β → Prop)
    (f : β → α → Except ε β) (hf : ∀ b a b', P b → f b a = .ok b' → P b') :
    ∀ (l : List α) (b b' : β), P b → l.foldlM f b = .ok b' → P b'
  | [], b, b', hb, h => by
    simp only [List.foldlM_nil] at h; cases h; exact hb
  | a :: l, b, b', hb, h => by
    rw [List.foldlM_cons] at h
    cases hfa : f b a with
    | error e => rw [hfa] at h; cases h
    | ok c => rw [hfa] at h; exact foldlM_inv P f hf l c b' (hf b a c hb hfa) h

/--
Every rolling record of the initial state is empty, since binding arguments
and external variables writes only registers.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `s`: The initial state.

# Hypotheses
- `h`: `s` is the initial state.
-/
theorem initialState_records {f : Function} {args : List Int}
    {externals : List (String × Int)} {s : Xdy.State}
    (h : f.initialState args externals = .ok s) :
    s.records = .replicate f.rollingRecordCount {} := by
  unfold Xdy.Function.initialState at h
  simp only [bind, Except.bind] at h
  split at h
  · split at h
    · cases h
    · rename_i v hv
      refine foldlM_inv (fun s : Xdy.State => s.records = _) _ ?_ _ v s
        (foldlM_inv (fun s : Xdy.State => s.records = _) _ ?_ _ _ v rfl hv) h
      all_goals
        intro b x b' hb hx
        repeat' split at hx
        all_goals first
          | (cases hx; done)
          | (cases hx; exact hb)
  · cases h

/--
A well-formed function ends with a `return`, its only one.

# Parameters
- `f`: The function.

# Hypotheses
- `hv`: The function is well formed.
-/
theorem back?_of_validate {f : Function} (hv : f.validate = .ok ()) :
    ∃ src, f.instructions.back? = some (.return src) := by
  unfold Xdy.Function.validate at hv
  simp only [bind, Except.bind] at hv
  repeat' split at hv
  all_goals first | (cases hv; done) | skip
  all_goals
    rename_i i hfind hlast
    have hsize : i + 1 = f.instructions.size := by simpa using hlast
    obtain ⟨hlt, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.mp hfind
    rw [Array.back?_eq_getElem?, show f.instructions.size - 1 = i by omega,
      Array.getElem?_eq_getElem hlt]
    revert hp
    cases f.instructions[i] <;> simp

/--
The oracle's run and the specification's agree: both fail with the same
message, or both succeed, the oracle's distribution having the
specification's law, or the oracle refuses an explode that may draw without
bound, whose law the specification answers.
-/
private def RunsAgree : Except String (Dist Int) → Except String (PMF Int) →
    Prop
  | .ok d, .ok p => d.HasLaw p
  | .error e, .error e' => e = e'
  | .error e, .ok _ => e = Xdy.unboundedMessage
  | _, _ => False

/--
The oracle and the specification agree on every run.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
-/
private theorem runsAgree (f : Function) (args : List Int)
    (externals : List (String × Int)) :
    RunsAgree (Xdy.Function.run f args externals) (run f args externals) := by
  unfold run Xdy.Function.run
  simp only [bind, Except.bind]
  split
  · rfl
  · rename_i v hv
    obtain ⟨src, hsrc⟩ := back?_of_validate (f := f) (by cases v; exact hv)
    split
    · rfl
    · rename_i s hs
      have sorted : ∀ r ∈ s.records, r.results.Pairwise (· ≤ ·) ∧
          r.chains.Pairwise (fun a b => a.le b = true) := by
        rw [initialState_records hs]
        intro r hr
        rw [(Array.mem_replicate.mp hr).2]
        exact ⟨.nil, .nil⟩
      have hfold := folds_foldlM f.instructions.toList (Dist.pure s)
        (PMF.pure (.ofOracle s)) (by
          rw [PMF.pure_map, State.toOracle_ofOracle s sorted]
          exact Dist.hasLaw_pure s)
      rw [Array.foldlM_toList, Array.foldl_toList] at hfold
      revert hfold
      simp only [hsrc]
      cases f.instructions.foldlM Xdy.stepAll (Dist.pure s) with
      | error m => exact id
      | ok final =>
        intro hlaw
        have := Dist.hasLaw_map (·.value src) hlaw
        rwa [PMF.map_comp, show (·.value src) ∘ State.toOracle =
          (·.value src) from funext fun s => State.value_toOracle s src]
          at this

/--
If the oracle answers a distribution, then the specification answers a law,
and the distribution has it.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.
- `d`: The oracle's distribution.

# Hypotheses
- `h`: The oracle answers `d`.
-/
theorem run_hasLaw {f : Function} {args : List Int}
    {externals : List (String × Int)} {d : Dist Int}
    (h : Xdy.Function.run f args externals = .ok d) :
    ∃ p, run f args externals = .ok p ∧ d.HasLaw p := by
  have := runsAgree f args externals
  rw [h] at this
  cases hr : run f args externals with
  | error e => rw [hr] at this; cases this
  | ok p => rw [hr] at this; exact ⟨p, rfl, this⟩

/--
The oracle meets the specification wherever it answers: its weights over its
total are the specification's probabilities, and it fails, with the same
message, exactly when the specification does, unless it refuses an explode
that may draw without bound.

# Parameters
- `f`: The function.
- `args`: The arguments.
- `externals`: The external variables.

# Hypotheses
- `h`: The oracle does not refuse the function as unbounded.

# Notes
A `PMF` is determined by its probabilities, so this determines the
specification's law from the oracle's distribution. `render` in
`Xdy/Oracle.lean` reduces the weights to lowest terms, which leaves every
probability unchanged. The specification gives an unlimited explode that grows
a chain a law of infinitely many outcomes, which the oracle refuses to
enumerate, as the Rust forward pass refuses to build it.
-/
theorem run_eq (f : Function) (args : List Int)
    (externals : List (String × Int))
    (h : Xdy.Function.run f args externals ≠ .error Xdy.unboundedMessage) :
    (run f args externals).map (⇑) =
      (Xdy.Function.run f args externals).map Dist.prob := by
  have := runsAgree f args externals
  cases ho : Xdy.Function.run f args externals with
  | error e =>
    cases hr : run f args externals with
    | error e' => rw [ho, hr] at this; cases this; rfl
    | ok p => rw [ho, hr] at this; cases this; exact absurd ho h
  | ok d =>
    cases hr : run f args externals with
    | error e' => rw [ho, hr] at this; cases this
    | ok p =>
      rw [ho, hr] at this
      simp only [Except.map]
      congr 1
      funext x
      exact this.2.2 x

end Xdy.Spec
