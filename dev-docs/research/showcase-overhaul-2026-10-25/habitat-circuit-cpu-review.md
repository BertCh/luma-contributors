# Independent CPU review of circuit residual drift

This review was requested after wave16's strong-resistance, no-water-barrier, four-neighbor
64 × 64 habitat circuit reported a current-conservation residual of `2.24e-4` at configured
budget 1,024 and threshold `1e-4`. The first four mild-resistance configurations had passed.
No GPU or browser was launched for this investigation. No runtime source was edited by this
reviewer.

## Mechanism

`GPUJacobiPCG` uses a recursively updated residual and gates later updates after its norm
passes the internal threshold. The sparse resistance recipe sets that internal squared
threshold to `tolerance² × 0.01`, then independently recomputes the CSR residual and certifies
maximum current imbalance, including the grounded sink. Increasing the configured budget
cannot repair a residual gap after the iterative updates have been gated off.

A CPU Float32 recurrence on the actual uploaded conductance contract reproduces the observed
scale: the failing configuration stops after 468 updates with recursive norm `9.51635e-6`,
but its freshly recomputed maximum residual is `2.25067e-4`. One warm-start restart recomputes
`b − A x`, takes 29 more updates, and reduces the certificate to `3.81470e-5`. A Float64 solve
rounded to Float32 certifies `4.10378e-5`, so the requested threshold is attainable; this is
not evidence for relaxing it.

The CPU model uses sequential per-row Float32 products/sums and NumPy Float32 dot products.
It is a mechanism diagnostic, **not a bit-exact GPU emulation**. GPU reduction order, fused
operations and compiler choices may differ. A production rerun remains necessary.

The primary numerical-analysis basis is the gap between recursively updated and explicitly
computed residuals described by H. A. van der Vorst and Qiang Ye in
[Residual Replacement Strategies for Krylov Subspace Iterative Methods for the Convergence
of True Residuals](https://epubs.siam.org/doi/10.1137/S1064827599353865).
Their work studies controlled residual replacement using error bounds. The proposed bounded
warm-start restart uses the same motivation; it does **not** claim to implement their adaptive
replacement strategy or guarantee convergence for arbitrary ill-conditioned systems.

## Budget allocation evidence

The durable matrix covers eight actual crop configurations: mild/strong resistance,
water barrier off/on, four/eight neighbors. Each was evaluated with seven allocations,
for **56 CPU experiments**. The external certificate threshold is unchanged at `1e-4`.

| Total budget and allocation | Result on these eight configurations |
| --- | --- |
| 128, uninterrupted | All remain nonconverged; withholding is correct |
| 512, uninterrupted | Four mild cases pass; four strong cases fail true-residual certificate |
| 1,024, uninterrupted | Same as 512 because recursive convergence gates later updates |
| 512 = 256 + 256 | Regresses mild four-neighbor cases to about `4.72e-4`; strong four-neighbor cases about `0.385` |
| 1,024 = 512 + 512 | All eight certify |
| 512 = 448 + 64 | All eight certify |
| 1,024 = 960 + 64 | All eight certify |

Short fixed restart blocks can discard useful Krylov progress before the slow modes converge.
The recommended bounded change therefore preserves a long first phase: for budgets at least
512, reserve 64 for one existing `GPUJacobiPCG` warm-start call; leave smaller budgets
unchanged. The maximum certificate among the tested reserve-64 cases is `8.50708e-5`.
The final certificate and nonconverged-output withholding must remain authoritative.

The recipe currently does not expose or consume a solver `getResult()` value. Its public
outputs/status remain the same physical resources. Status word 1 means configured total
budget, not executed iterations; the two phase budgets must sum to that declared total.
The existing first solver's identity should be preserved where feasible, and the restart
must depend on its final potential writes without clearing or reinitializing the solution.
The generic PCG contributor should remain unchanged.

## Durable evidence and reproduction

- `habitat-circuit-classes.json`: source class codes, with crop and source hash in
  `habitat-circuit-classes.provenance.json`.
- `habitat-circuit-float32-review.py` and `.json`: original failing-configuration diagnostic,
  true-residual restart, and rounded Float64 comparison.
- `habitat-circuit-budget-matrix-review.py` and `.json`: 56 CPU budget experiments, including
  exact update counts and both CSR and independently assembled edge-current residuals.

Run from the repository checkout with the prepared NumPy/SciPy environment:

```sh
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/habitat-circuit-float32-review.py
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/habitat-circuit-budget-matrix-review.py
```

The parent approved the reserve-64 recipe correction after this review. Earth owns its
implementation and focused regression. Source/test review and production acceptance are
pending at this record's initial writing; these CPU experiments alone do not establish a
GPU pass.

## Independent implementation and test review

Reviewed Earth's final recipe patch in
`modules/graph/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance.ts` and the focused
`modules/graph/test/gpu-spatial-analysis/urban-ecology/resistance-residual-drift.spec.ts`.
The recipe's first phase keeps the original solver ID and reserves exactly 64 updates only
when the declared budget is at least 512. The second solver uses the same CSR, RHS and potential
views, with no solution clearing between passes. Its existing initialization computes a fresh
residual from the previous solution. Both phase budgets sum to the unchanged configured total.
The final certificate, currents masking and status meanings are unchanged. Budgets 128 and
256 retain a single PCG call. No generic PCG code was changed for this correction; the existing
unrelated SpMV-strategy working-tree change was already present before this review.

The final focused regression covers 128 and 256 as honestly withheld, and 512 and 1,024 as
strictly certified, with the original `1e-4` threshold. It also checks independent resistance
and energy and explicit phase accounting. The fixture decodes to exactly 4,096 source class
values, matching the independently extracted 64 × 64 crop and source PNG hash. Its Float64
effective-resistance reference differs from this review's independent sparse direct solve by
only `3.46e-9`. It is a full 64 × 64 fixture, not the 16 × 16 scene comparison subset.
The four-neighbor raster has 8,192 edge slots, of which 128 are zero-conductance boundary
placeholders, leaving 8,064 actual edges.

Status and diagnostic taps use exact 16-byte `uint32x4` and `float32x4` views of their existing
imported buffers. Status fields are convergence, configured total budget, reachability and
invalid-edge count. Diagnostic fields are effective resistance (NaN when uncertified), maximum
current residual, energy and residual norm. The regression checks the relevant documented
contract without treating configured budget as an executed-iteration count.

**No source/test review blockers remain. GPU regression execution is still pending and must
not be inferred from the CPU evidence or this review.** The parent was notified that the patch
is ready for its coordinated wave17 build and test slot.

## Wave17 GPU result and accumulation-order follow-up

The reserve-64 change alone did **not** pass the GPU gate. The first focused run failed the
512 and 1,024 certificates. A diagnostic rerun certified 512 but its resistance error was
`0.22507693`, above the unchanged `0.2` assertion. The 1,024 case reported maximum residual
`0.0001147821822`, residual norm `0.0007908833213`, and energy `1633.1060791`.
The 128 and 256 cases correctly withheld uncertified results. These results supersede any
inference of numerical acceptance from the earlier CPU-only budget matrix.

The adaptive SpMV planner selects subgroup-row for these nine-entry average CSR rows on a
subgroup-capable adapter. PCG therefore recomputes its restart residual using a subgroup
reduction, while the recipe's final certificate sums each CSR row sequentially. Duplicate
diagonal entries intentionally preserve separate incident edges. Grounded-sink balance is
reconstructed from the sum of nongrounded residuals, so small per-row errors can accumulate.

`habitat-circuit-accumulation-review.py` and its JSON report preserve 18 CPU experiments:
three first-phase accumulation methods × three correction methods × two total budgets.
The methods model sequential Float32 products/sums, a balanced subgroup-like reduction,
and sequential fused multiply-add. They are sensitivity experiments, **not a bit-exact GPU
emulator**; NumPy dot products and compiler fusion can differ from the actual adapter.
The script also independently evaluates Float64 edge currents using the final Float32
potentials and rounded Float32 conductances.

At total budget 1,024, sequential/sequential accumulation gives a `3.81470e-5` CSR certificate
and `3.35693e-5` Float64 physical maximum current residual. Tree/tree gives `1.14724e-4`
and `1.34770e-4`, respectively. Tree-first/sequential-correction still gives `1.09345e-4`
and `1.32844e-4`. The physical residual failure shows that the modeled discrepancy is not
only certificate rounding, and correction-only scalar accumulation is not sufficient in
this model. Fused variants also show sensitivity; actual GPU execution remains necessary.

Reproduce the 18 cases from the checkout:

```sh
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/habitat-circuit-accumulation-review.py
```

The existing primary numerical-analysis justification remains residual-gap control:
[van der Vorst and Ye, Residual Replacement Strategies](https://epubs.siam.org/doi/10.1137/S1064827599353865).
The fixed restart used here is not that paper's adaptive error-bound algorithm. Matching
row accumulation order is a separate implementation-contract decision supported by the
source review and sensitivity experiments above.

## Wave18 bounded source review — GPU acceptance pending

Independently reviewed the final recipe and focused test after Earth's scalar-row change.
For budgets at least 512, **both** existing PCG phases now use `spmvStrategy: 'scalar-row'`.
The initial phase remains total minus 64 and the correction remains 64, sharing the same
CSR, RHS and potential views without reinitialization. Budgets 128 and 256 retain the
original adaptive strategy and a single solve. Provenance records the strategy and exact
phase budgets. No generic PCG implementation change was introduced by this correction.
The strict `1e-4` final certificate and independent `0.2` resistance/energy assertions remain
unchanged; a small residual alone does not establish a tight potential-error bound.

The 1,024 regression now captures all 4,096 final potentials and all 8,192 raster conductance
slots before its assertions. This permits a separate Float64 physical-current audit of actual
GPU results. The existing status and diagnostic views remain exact 16-byte readbacks.
No source-review blocker remains for the coordinated wave18 build. **Neither the focused
GPU regression nor the eight actual scene configurations are accepted by this review.**

## Wave18 actual GPU capture — correction experiment rejected

The actual scalar-row GPU run still failed at 1,024 with the same reported diagnostics as
wave17. The raw capture is preserved as `habitat-circuit-wave18-gpu-capture.json`.
`habitat-circuit-wave18-physical-audit.py` independently assembles edge currents from all
4,096 captured potentials and 8,192 captured conductances. Every conductance matches the
reference Float32 conductance exactly. The actual physical maximum residual is `2.48359138e-4`
at grounded sink 2,084, above `1e-4`. Effective resistance is `1633.33740234`, versus the
independent Float64 `1633.56235721` (error `-0.22495486`); physical energy is `1633.11250960`.
The final certificate's `1.14782182e-4` is therefore not merely a rounding artifact that masks
an otherwise physically valid solution. Both reported and independently evaluated residuals fail.

**Correction to the earlier causal interpretation:** subgroup selection was not established
for this actual device. The focused fixture requests `getWebGPUTestDevice('core')` without
optional features. `getRequiredWebGPUFeatures('core', [])` requests no optional subgroup
feature. Thus the prior tree-reduction experiments are sensitivity studies, not evidence
that this actual run selected subgroup-row. The source explicitly requests scalar-row in
wave18, yet the observed failure is unchanged.

A second eight-case CPU sensitivity study is preserved in
`habitat-circuit-fused-update-review.py` and `.json`: sequential versus fused matrix actions,
NumPy versus portable 256-element tree dot reductions, and separate versus fused vector
updates. Fused matrix-action models closely reproduce the actual resistance bias, whereas
sequential separately rounded products do not. For portable tree reductions and fused
vector updates, fused matrix action gives resistance `1633.33508301` and physical maximum
residual near `2.50e-4`. This implicates cancellation in matrix action on nearly constant
potential modes, but is not a bit-exact proof of compiler fusion on the actual device.

The requested zero-initialized correction equation was tested as a CPU diagnostic. It
eliminates many lost small updates: an extra 64-step warm solve from the captured final
potential loses 123,409 nonzero component updates, while the separate correction loses only
one. Nonetheless its physical sink error remains approximately `2.44e-4` and resistance
error approximately `-0.2231`. From the fused CPU first-phase model, an in-budget 64-step
separate correction likewise fails. These experiments do **not** justify another runtime
change on the assumption that separate correction alone repairs the solve.
[Carson and Higham's iterative-refinement analysis](https://epubs.siam.org/doi/10.1137/17M1140819)
explains the correction-equation framework and the distinct roles of residual and correction
precision; it does not guarantee convergence for this same-precision, fixed-budget example.

The scope recommendation is to restore the original single-solve recipe behavior, retaining
strict certification and nonconverged-output withholding. Remove only this experiment's
restart/strategy changes; preserve unrelated preexisting generic solver work. The showcase
should explicitly present nonconvergence as a result. All eight actual scene configurations
must still be checked: accepted cases against independent references, and withheld cases
for absent accepted scalar/map/chart evidence, including display and rebuild transitions.
Do not predeclare that all four strong cases withhold until that exact source is executed.
No generic solver redesign, tolerance relaxation, or extra iteration budget is recommended.

Reproduce the actual-array and arithmetic-sensitivity audits:

```sh
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/habitat-circuit-wave18-physical-audit.py
/tmp/showcase-statistics-reference-env/bin/python dev-docs/research/showcase-overhaul-2026-10-25/habitat-circuit-fused-update-review.py
```

## Final rollback and eight-case verifier review

The experimental recipe change has been removed. `git diff --exit-code --
modules/graph/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance.ts` succeeds:
the recipe exactly matches HEAD, including its original single PCG call with the full
configured budget, strict certificate, and NaN currents/effective resistance when rejected.
The scene additionally hides both potential and current layers when the main certificate
fails; residuals remain explicitly diagnostic, and accepted energy is not displayed.

Generic solver files were unchanged across the rollback, verified by SHA256:

- `gpu-pcg.ts`: `59799f2e7e17824412f7cb4901cdd68889b8fc0cb5c5a2e204a2dda72b0189e2`
- `gpu-adaptive-spmv.ts`: `943eabc5c010d785157c9b81db4ca4abe9e1e2084738808e45d5e1b821370165`
- `gpu-spmv-strategy.ts`: `f64ad2e72bfcf7a2e2525b8fee04473c2aedc9e86b6abcf119f2ef0f6507cbc7`

Independently reviewed `check-habitat-circuit-browser.mjs`; `node --check` passes.
All four mild configurations must remain accepted, while each strong configuration may
accept only under the existing gate or explicitly withhold. Accepted resistance and energy
retain the original independent-reference tolerance of `0.1% + 0.003`; this was not widened
for the failed experiment. The separate 16 × 16 comparison remains separately checked and
does not certify the full 64 × 64 result.

For each withheld case, the verifier checks absence of accepted resistance/energy claims,
absence of circuit charts, and identical isolated analysis-canvas pixels across potential
and current products at opacity 0.2 and 1. Accepted cases provide the positive control:
each product must respond to opacity. Screenshot-only visibility styling excludes HTML
legends and the MapLibre canvas from the comparison. Returning to mild contrast must
recover an accepted map and independent scalar agreement. Partial reports are preserved,
and served build evidence is compared before and after the run.

These are source-review conclusions only. The eight-case GPU run remains pending. For
withheld cases the suite verifies publication policy and the rendered certificate threshold;
it does not claim independent physical accuracy for the rejected solution. The archived
wave18 capture retains the independent physical-failure evidence described above.
