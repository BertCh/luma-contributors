<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
-->

# BIXI spatial interaction and hypothetical service locations

## Question and support

These two connected stories use BIXI Montréal's August 2024 trip history. The interaction study asks
how fitted station-to-station counts change under four gravity-model margin constraints. The siting
study asks how hypothetical maintenance or support locations would change under different location
objectives. In both, a station's observed August departures plus arrivals is an **activity weight**,
not a measurement of crew workload or a staffing requirement.

The stories share a deterministic induced sample of 16 stations with the largest monthly departures
plus arrivals. Their IDs, names, 225 observed directed pairs, 15 missing off-diagonal pairs and 16
analytically excluded self-pairs are listed in
[`bixi-models-reference.json`](./bixi-models-reference.json). Self-pairs are structurally excluded
because this analysis studies inter-station trips; this does not assert that BIXI recorded no
same-station rides. The showcase builder (`scripts/data/bixi-flows/build.py`, `PAIR_MIN = 3`) retains monthly pairs with at least three rides from the original BIXI trip records. This threshold and exclusion of same-station rows are showcase aggregation choices, not an official BIXI publication or privacy rule. A pair
missing from this table is censored or otherwise unobserved, not an observed zero. The hourly file
uses a different ten-ride pair threshold and sentinel residual rows, so it is not used to fit the
monthly gravity model.

Source bundle metadata credits BIXI Montréal, describes dropped short rides and pair thresholds, and
states the August 2024 local-time window. The bundled showcase catalog lists the Montreal open data
portal's CC BY attribution for the trip-history listing; BIXI's own open-data page does not itself
publish a licence statement. The source data README retains this distinction and its BIXI credit.

## Methods

### Interaction experiment

`GPUSpatialInteraction` fits a one-covariate Poisson gravity model using straight-line station
distance in kilometres. Its coefficient is per kilometre. The nine-point grid spans −1.5 to 0;
bounded profile search uses the same closed interval. Every fit explicitly marks each candidate OD
cell as observed, analytically structural zero, or missing. Missing cells are excluded from
calibration, margins, deviance and prediction display; diagonal cells are structural zeros. All
reported margins are conditional on observed pairs in this induced support.

The unconstrained, origin-constrained, destination-constrained and doubly constrained fits are
different conditional models. In particular, “doubly constrained” uses iterative proportional
fitting over the available observed support. It does not restore the missing pairs or the complete
station margins. Poisson deviance here is an in-sample descriptive fit statistic; no uncertainty,
causal effect, travel-time effect or out-of-sample prediction is claimed.

The independent CPU implementation reconstructs the support from raw monthly binary columns,
profiles the same coefficient grid, balances the same margins, and separately uses a dense scan as
an oracle for the bounded coefficient. The scan is only a reference; the GPU contributor uses its
bounded golden-section profile and reports its own bracket, balancing and status outputs.

### Hypothetical support-site experiment

The same activity-weighted station sample supplies 16 demand points. Eight real BIXI stations form
the candidate set: highest activity first, then deterministic farthest-point additions in the local
metric frame. They are candidate locations for a **hypothetical** support-site exercise, not
asserted facilities, available parcels or recommended operating bases. Costs are straight-line
distance in kilometres, a map-scale proxy rather than cycle-network distance or travel time.

The story runs `p-median`, `maximum-coverage` and `set-covering` with a configurable 0.5–3 km
coverage radius (1.5 km initially), a three-site budget for the first two, and swap improvement for
the applicable greedy objectives. It compares the full demand-major dense cost matrix with a sparse
CSR matrix that truncates costs beyond 3 km; the cutoff is fixed and explicitly reported. It also
runs exact bounded `p-center` over all \(\binom{8}{3}=56\) candidate subsets and
`capacitated-p-median` using an unsplit demand-order greedy assignment. Capacity is 0.4 times the
sampled total activity at each candidate (1.2 times total over all eight candidate capacities); it
is only an invented activity-proxy scenario. The station-ID order and activity-descending order are
explicit comparisons. Neither bounded capacity output is a global assignment optimum or an
infeasibility certificate.

The independent CPU placement oracle exhaustively evaluates all size-three candidate subsets for
p-median, maximum coverage and p-center, searches increasing candidate-set sizes for set covering,
and separately reproduces each bounded capacity demand order. Exact subset enumeration is a small
reference for this reduced example; it does not certify the production greedy contributors' general
optimality.

For allocation, the oracle mirrors the production input contract: station coordinates are projected
around `[-73.58, 45.52]` with the showcase `LocalMetricProjection` refinement, rounded into float32
positions, and converted to float32 kilometre costs before evaluating candidates. It then exhaustively
enumerates all 56 three-site subsets and computes weighted objectives from the float32 activity inputs.
The oracle uses the same metric inputs without using GPU solver outputs. A small known-value fixture
exercises the same objective-winner helper used by enumeration: candidates with objective values 2 and
7 select 2 for minimization and 7 for maximization.

## Research sources

- BIXI Montréal, [open data and 2024 trip history](https://bixi.com/en/open-data/).
- Données Québec, [BIXI trip-history listing and CC BY 4.0 metadata](https://www.donneesquebec.ca/recherche/dataset/vmtl-bixi-historique-des-deplacements).
- PySAL `spint`, [gravity model constraint forms and Poisson calibration](https://github.com/pysal/spint). This implementation's grid and bounded-profile calibration with contributor IPF is not claimed to reproduce PySAL's IRLS implementation.
- PySAL `spopt`, [p-median weighted distance objective](https://pysal.org/spopt/generated/spopt.locate.PMedian.html).
- PySAL `spopt`, [LSCP and MCLP model API](https://pysal.org/spopt/api.html).
- Sa, Kang and Lee, [A review of location models](https://www.theoj.org/joss-papers/joss.03330/10.21105.joss.03330.pdf).

## Coverage and verification

| Planned functionality | Delivered source | Current evidence |
| --- | --- | --- |
| Four spatial-interaction constraints | `bixi-demand-model.compute.ts` | CPU grid reference for all four; GPU/browser not run |
| Grid and bounded coefficient calibration | same | CPU dense-scan reference; GPU bracket output is wired; GPU/browser not run |
| Observed, structural-zero and missing pair support | `bixi-model-data.ts` | CPU asserts 225 + 15 + 16 = 256 cells; missing predictions stay NaN |
| Predictions, residual margins, deviance, coefficient, balancing and status | interaction compute and scene | GPU outputs feed map/readouts/charts; no GPU acceptance claim |
| p-median, maximum coverage, set covering and swap state | `bixi-service-locations.compute.ts` | Exhaustive CPU reduced-instance objective oracle; GPU/browser not run |
| Exact bounded p-center and capacity-constrained p-median | same | Independent exhaustive CPU reference for 56 subsets; capacity order outputs differ |
| Demand-major dense and radius-truncated sparse costs | same | CPU reference: 76 of 128 facility-demand costs lie within 3 km; no demand row is isolated |
| Assignment spokes, selected candidate points, uncovered demand and objective/status | same | GPU buffers directly feed map and readouts; GPU/browser not run |
| Source provenance, sample rules, units and limits | this file and source README | Written; CC BY attribution preserved |

Independent reference command:

```sh
node examples/deck/spatial-analysis-showcase/scripts/check-bixi-models-reference.mjs
```

It writes the reproducible values to `bixi-models-reference.json`. Scoped formatting and CPU
reference checks passed. `check-story-steps.mjs` passes both BIXI scenes. The flow/network static
cartography command reports 11 findings in other scenes; both BIXI scenes pass. The showcase
typecheck reports no BIXI diagnostics but remains red on independently owned
`terrain-boundaries.scene.ts` and `habitat-connectivity.compute.ts` errors. The browser harness at
[`bixi-models-browser-harness.mjs`](./bixi-models-browser-harness.mjs) passed all 15 numerical cases on wave 15 bundle `5c13c77a7d5dd844146b724ae5c70d49272c63080f028da1228e99f5e22fac65`, with zero console/page errors. The accepted report is `/tmp/luma-showcase-overhaul-20261025/movement-wave15/bixi-retry/report.json`. P-center is checked against the exact CPU result rounded to the public two-decimal kilometre readout, rather than comparing a rounded display with a tighter raw-value tolerance.

The focused durable verifier `scripts/check-bixi-selected-labels.mjs` subsequently passed on wave 16 bundle `31e545703f8e9c2990ab47eb3c8ec879c9e3532b9a386c16da552c59782e7eaf`: selected A, B and H labels are visible, categorical spokes and legend agree, and the subdued Montréal map context is present after tile loading. Earlier failed reports are retained. `movement-acceptance-final.json` contains exact report hashes, commands, methods, tolerances and limitations for both scenes and the visual check.
