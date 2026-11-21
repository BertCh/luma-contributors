# Geographical detector, REDCAP, spatial entropy and the contiguity-order correlogram

STX-08, first group. Import from `@luma-contributors/analysis/gpu-spatial-analysis`. Each method names its reference and what was
run against it. Where a reference package could not be installed, a binary64 CPU evaluation of the published definition stands in, and the section says so.

## Geographical detector: `GPUGeographicalDetector`

The label and response arrays support up to 16,777,216 rows at a 128 MiB storage binding limit.
The partition stage also caps `labelCapacity` at 33,554,431 at that binding size.

The factor detector of Wang et al. (2010): `q = 1 - SSW / SST`, the share of the variance of a response `values` explained by a
stratification `labels` (a classification, an administrative code, a zoning). It reads the sums of squares, strata sizes and strata means
from `GPURegionPartitionEvaluation` (which gained an optional `output.regionMeans` for this), then reduces the strata means in one tiled pass.
No float atomics; results are bitwise reproducible.

This interface is the factor detector. It does not return interaction classifications, risk
comparisons between strata or ecological comparisons between explanatory stratifications.
Those models have no input/result contract in this API. Separate single/joint stratifications
can each be passed to the factor detector to obtain q, without implying those additional tests.
The contributor reads declared construction fields once (including inherited getters) and
captures the output map. Later changes to caller label capacities or view maps cannot redirect
its graph; the original value/label buffers remain live inputs.

`output.summary` is laid out by `GPU_GEOGRAPHICAL_DETECTOR_LAYOUT`: `q`, `fStatistic`
`F = ((N - L) / (L - 1)) q / (1 - q)`, `nonCentrality`, `strataCount` `L`, `observationCount` `N`, the two degrees of freedom and
`totalSumOfSquares`. Rows labelled `0xffffffff` or at or above `labelCapacity`, and rows whose value is NaN or infinite, are left out. A constant response gives NaN.

For solver-produced strata, supply the optional scalar uint32 `labelsComplete` together with
`output.labelsComplete` (one uint32 word). The output copies nonzero to 1 and zero to 0 on each
submission. Zero marks a provisional partition; all summary statistics remain available and
describe the supplied incumbent labels. Ordinary categorical or administrative labels need no
certificate: omit both ports. Neither the labels nor the certificate are a row count.

q uses the partition's directly accumulated `SSB/SST`, retaining small explained variation when
`SST-SSW` would round to zero. These independently reduced float32 SSDs can leave q a few ulps
below one even when the exact ratio rounds to one; q is not forced to one for nearly perfect
splits with positive SSW. A defined perfect partition with zero SSW and positive SSB publishes
exactly q = 1. For strata `[0]` and `[1,1,1,1]`, SSB = SST = 4/5, q = 1 and F = +Infinity.
A constant response still has undefined q and F. These numerical conditions are independent of
the optional partition-completeness certificate. For positive SSW, F uses the partition's exponent-safe pseudo-F
`((N-L)/(L-1)) * SSB/SSW`, with mantissas and exponents combined before scaling. It avoids
an intermediate ratio overflow when the final F is representable and remains finite when a
nearly perfect split rounds q to 1. Infinity represents zero within-
stratum SS with positive between-stratum SS. For `[0,0,0,1]` and `[10000,10000,10000,10001]`,
F is 800000000 and the packed-summary noncentral-F tail is about 1.52051e-24.
When positive within-stratum SS gives a finite mathematical F outside the float32 range,
F and its derived probability are NaN rather than a false exactly-perfect zero tail.
The small-signal regression has `SSB=2^-25`, `SSW=8`, `F=2.2351741790771484e-8` and
p about 0.9998855591. An extreme seven-stratum regression has `F=2^127` and p about
9.0780105121473e-20, despite its overflowing unscaled SS ratio.
Stratum-mean partials merge in parallel groups above 4,096 tiles, keeping the final summary
loop bounded. Up to 4,096 tiles retain the original accumulation order. The sparse-label
regression also exercises 4,097 tiles with just eight observations.

Before evaluating moments, responses whose largest usable magnitude is below `2^-32` receive
one common power-of-two normalization on the GPU. Integer bit classification retains tiny
response values before squaring and distinguishes zero from a subnormal word. Ratios and
noncentrality use the normalized moments; `totalSumOfSquares` is restored to response-squared
units with nearest-even float32 rounding, including subnormal results. Its stored value can
therefore round to zero while the dimensionless ratios remain defined. For strata `[0,2h]`
and `[4h,8h]`, SSW=`10h²`, SSB=`25h²`, q=`5/7`, F=`5` and noncentrality=`10/7`, both at
`h=1` and `h=2^-76`; underflow of an unnormalized within SS cannot certify a perfect split.

The p-value is computed on the host, from the read-back summary: `getGeographicalDetectorPValue(summary)` is the upper tail of the
non-central F distribution, summed as a Poisson mixture of incomplete beta functions in binary64 (`getNonCentralFUpperTail` is the
distribution tail itself). Both beta arguments are formed directly, retaining the smaller argument
even when the other rounds to one. Stable log-beta normalization handles disparate degrees of
freedom; weighted survival terms are accumulated in log space and rounded only after summation,
including subnormal tails. For finite positive `F`, supported non-centrality is at most 10000;
larger or infinite non-centrality throws an error. With valid positive degrees of freedom and finite
non-centrality, `F = Infinity` returns zero and `F <= 0` returns one before applying that mixture
limit. This includes an exactly perfect detector (`SSW = 0`, positive `SST`) with non-centrality above 10000.
NaN statistics return NaN. Each mixture direction
and beta continued fraction is bounded by 100000 iterations; a convergence failure throws an error.
It is not computed on the GPU.

For consumers that can display a summary without inference, `readGeographicalDetectorPValue(summary)`
returns `{pValue, unavailableReason}`. A supported result has a numeric `pValue` and null reason;
an undefined statistic, unsupported range or numerical convergence failure has null `pValue` and
an explanatory reason. Unrelated read errors still throw. The strict helper above keeps its
throwing contract. Kepler preserves all summary rows, publishes null in the `pValue` row,
sets `summary.pValueStatus = 'unavailable'` and reports a
`geographical-detector-p-value-unavailable` warning with the reason. Supported probabilities have
`pValueStatus = 'available'`. The website readout likewise retains q and F and labels an
unavailable probability with its reason. For example, strata `[A,B,B,B]` with responses
`[1000,1001,1002,1003]` retain `q = 0.6` and `F = 3` even though their non-centrality exceeds 10000.

Reference: `geodetector` 0.2.0 (MIT) `q_statistic` and `q_significance_test`, whose formulas match the R `GD` package; its `_stats.py` ran in
`stx08-spatial-methods.py`. Differences from the reference, stated as they are:

- The package drops strata with a single row. This contributor keeps them (they add nothing to SSW but count in `L` and `N`). Compare on data whose strata have at least two rows.
- `nonCentrality` is the published formula `(sum_h mean_h^2 - (sum_h sqrt(N_h) mean_h)^2 / N) / sigma^2` with the population variance. It is not invariant to shifting `values`, and it differs from the textbook `sum_h N_h (mean_h - mean)^2 / sigma^2`. It was implemented as the reference computes it, not corrected.
- float32 accuracy: `q` agrees to about `2e-5` in the fixture. `nonCentrality` subtracts nearly equal sums when the strata means dwarf their spread (the fixture's offset case, means near 50), and keeps one to two digits there; centre and scale `values` first when that matters, bearing in mind the formula is not shift invariant.

{/* lens:GPUGeographicalDetector */}

## REDCAP single linkage: `GPURedcapSingleLinkage`

Contiguity-constrained single-linkage clustering, cut at a region count that is a per-frame parameter. It is the tree of first-order
single-linkage REDCAP (Guo 2008) with a different partition step, so it does not return the REDCAP regions: REDCAP partitions the tree by removing,
at each step, the edge whose removal lowers the within-region sum of squared deviations the most, which for this tree is what
`GPUSpatialWeightsMinimumSpanningTree` followed by `GPUSkaterRegions` computes; this class removes the heaviest edges. On a path with values
`[0, 1, 2.1, 3.3, 4.6, 6.0]` and two regions it returns `{0..4}, {5}` (within-region sum of squares 13.26) where the SSD partition is
`{0, 1, 2}, {3, 4, 5}` (5.85). Single linkage chains, so expect single outlying rows to be split off first. Use SKATER when REDCAP regions are
wanted. The minimum spanning tree of the contiguity graph under squared Euclidean dissimilarity of the standardised attributes holds every
single-linkage merge, so removing its `k - c` heaviest edges (`c` trees in the forest) leaves `k` contiguous regions. It composes
`GPUSpatialWeightsMinimumSpanningTree` (edges and dissimilarities), a stable `GPUSort` on `(cost, edge slot)`, a masked copy of the weights and the weak
components of what remains (`createMaskedWeakComponentNodes`). Region `k` refines region `k - 1`.

The SSD distinction is covered by `redcap-audit-b.oracle.py`, which exhaustively evaluates all
five cuts of that path after float32 input rounding. With two additional islands and a target of
four regions, the heaviest-edge cut labels are `[0,0,0,0,0,5,6,7]` and its SSD is
`13.259999456405648`; MST followed by SKATER labels `[0,0,0,3,3,3,6,7]` and its SSD is
`5.853333263397228`. SKATER's cut log records slot 4 and a gain of `19.439999828338618`.
The node oracle checks the direct contributor, lowered GPU operation, CPU function and CPU
operation against those independent values. This verifies the SSD objective and the composition;
it does not verify the unavailable Guo primary paper or supply the other linkage tree builders.

`parameters` holds the region count (`getGPURedcapSingleLinkageParameterValues`). Outputs: `labels` (the smallest row index of each region),
and optionally `regionCount` and `status` with `converged`, `iterationLimitReached` and `invalidCount` (the tree's non-finite entry count; an edge that touches a non-finite row costs `+Infinity`
and is cut first). The independent optional `status.iterationCount` receives the number of executed
label-propagation rounds: always `componentIterations`, because all scheduled rounds run even after
labels stabilize. It excludes the spanning-tree stage and is written even for a partition with no
retained edges; requesting it does not require either convergence output.
`regionCount`, `status.converged`, `status.iterationLimitReached` and `status.invalidCount`
require packed `uint32` views with at least one word, validated at
construction. Region count, convergence/iteration-limit flags and invalid count write only their
first word; spare scalar capacity remains untouched on ordinary, suppressed and recovery frames.

A supplied nonzero `weights.status.overflow` or `candidateOverflow` suppresses the partition:
every label is `0xffffffff`, `regionCount` is 0, and both convergence/iteration-limit flags are 0.
The executed iteration count and the attribute invalid count retain their own meanings. When the
producer restores complete weights and clears the loss flag, the next submission publishes the
ordinary partition again; a partial forest is never presented as a valid region assignment.
REDCAP passes the original CSR and both supplied loss words directly to its minimum-spanning-tree
stage. That stage suppresses its tree flags and edge count on the GPU; REDCAP also suppresses the
dependent partition outputs. Clean status words produce the same partition as weights without
status, and changing either word requires no graph recompilation.
Supply the producer's final `overflow` with any `requiredCount`. For producers reporting
`complete`, `converged`, `iterationLimitReached` or `unresolvedCount`, verify completeness
before construction and omit those views from the REDCAP input.
REDCAP snapshots its declared weights, loss views and output maps before validation. Replacing
the caller's maps after construction cannot disconnect a validated loss port; change buffer
contents to resubmit with the same bindings.

{/* lens:GPURedcapSingleLinkage */}

Limits and non-goals:

- Accepts 1–32 attribute columns. The attribute, row, neighbor and edge scratch allocations must
  each fit one storage binding (33,554,432 32-bit entries at 128 MiB). The estimated graph is
  limited to 32,768 command nodes.
- The label propagation runs `componentIterations` iterations (default 32, at most 256) and `output.status.converged` reports whether that sufficed. A 300-row path with regions of 100 rows converged in 32; a region that is a much longer thin path may need more, and an unconverged labelling is not valid.
- No minimum region size or bound (the GeoDa and `spopt` floor), and no full-order, average or complete linkage variants (they need the tree rebuilt per region). Weights must be symmetric.
- Ties among edge costs are broken by the lower edge slot; other libraries tie-break differently, so compare on distinct costs.
- float32: costs are float32 squared distances, so two edges whose true costs differ by about one part in `1e7` can swap order.

Reference: no REDCAP package was run. The reference is the constrained single-linkage definition,
evaluated in binary64 with SciPy (minimum spanning forest, heaviest-edge removal, connected
components). For a connected graph with distinct costs, this agrees with
`sklearn.cluster.AgglomerativeClustering(linkage='single', connectivity=W)` on the same
preprocessed attribute table only if preprocessing and numerical distance handling preserve the
same edge order. Scikit-learn 1.6.1 replaces zero distances by machine epsilon during its
single-linkage MST construction. For the connected cycle `0–1–2–3–0`, raw values
`[0,0,1e-18,3e-18]` and three regions, its labels canonicalized to each cluster's smallest row
are `[0,1,1,3]`, while this contributor keeps the
exact zero-cost edge and returns `[0,0,2,3]`. Distinct costs alone do not ensure equivalence.
Scikit-learn 1.6.1 was run on this cycle and on connected and disconnected eight-row
fixtures in `redcap-audit-b.oracle.py`. Its `_fix_connectivity` completes a disconnected graph,
whereas this contributor never adds links: a six-row path and two islands requested as one region
remain three regions here, while scikit-learn joins them into one. Therefore that reference
equivalence requires connected inputs and matching numerical edge ordering. Existing SciPy fixtures also include a 7 by 7 and
a 9 by 5 lattice and a two-component graph, at region counts from 1 to above the row count.

{/* parameters: GPURedcapSingleLinkage generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

### `GPURedcapSingleLinkage` parameters

Import from `@luma-contributors/analysis/gpu-spatial-analysis`: `GPURedcapSingleLinkage`.

- Packer: `getGPURedcapSingleLinkageParameterValues(settings)`, writes `GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH` (1) `uint32` values.

Settings of `getGPURedcapSingleLinkageParameterValues`:

| Field | Type | Description |
|---|---|---|
| `regionCount` | `number` | Target region count, at least 1. A value at or below the number of trees of the spanning forest (the connected components of `weights`) cuts nothing; a value above the row count cuts every edge. |

{/* /parameters */}

## Spatial entropy: `GPUSpatialEntropy` and `GPUCategoryPairCounts`

Both contributors capture category counts, category/CSR view maps, status/provenance metadata
and output maps before validation. Declared inherited getters are read once. Replacing those
maps or metadata later does not alter the validated graph; original GPU buffers remain live.
No distinct O'Neill estimator/mode is exposed: the available measures are the explicitly defined
Shannon, ordered-pair and endpoint-category quantities below.

`GPUCategoryPairCounts` counts, over the links of a weights graph, the ordered category pairs and the rows per category, with exact integer atomics
(at most 32 categories). `GPUSpatialEntropy` builds on it: the Shannon entropy of the categories, the entropy of the origin and destination category of a link, the
co-occurrence entropy of the ordered pair (Leibovici 2009), and the endpoint-category mutual information
`H(origin) + H(destination) - H(pair)`, with its share of the destination entropy. All in nats. Every active slot (positive weight, another row: the shared neighbor rule) is a link;
self slots and zero-weight slots are not, and the size of a positive weight is not used: use
contiguity weights, or separate distance-band weights to obtain ordered-pair entropies per band
for comparison. These runs do not compute Altieri's decomposition. `output.summary` is laid out
by `GPU_SPATIAL_ENTROPY_LAYOUT`.

Category row counts and Shannon entropy depend only on the category rows, independently of
adjacency. With a complete graph containing no links, the link count is 0 and every link entropy
and mutual information is NaN; row counts and Shannon entropy retain their category-only values.
Both contributors require a verified complete CSR. Omit upstream loss, requirement, completeness,
iteration and unresolved-work status views once you have verified the graph.

Entropies are float32 from exact counts, with fixed pairwise accumulation of the entropy terms.
Two invocations compute integer totals and four parallel invocations each reduce one entropy,
visiting at most 32² counters
(`GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT = 32`). Eight publication invocations then
read those reduced values. Independent uniform 16- and
26-category tables have joint entropies `ln(256)` and `ln(676)` and exactly zero mutual
information. Device verification of their absolute error near zero is pending; no measured
near-zero tolerance is claimed here.
Altieri, Cocchi and Roli's decomposition uses mutual information between unordered category pairs
and distance classes; it is not the endpoint-category mutual information returned here. That
decomposition and Batty's area-weighted entropy require distance-class/area inputs absent from
this interface. Separate distance-band graphs provide ordered pair entropies for comparison.
Reference: independent binary64 NumPy/SciPy evaluations of the definitions above
(`stx08-spatial-methods.py` and `categorical-association-audit.oracle.py`); no pinned `SpatEntropy` run.

`GPUCategoryPairCounts` and `GPUSpatialEntropy` reject any supplied `weights.status` view for
`overflow`, `candidateOverflow`, `requiredCount`, `complete`, `converged`, `iterationLimitReached`
or `unresolvedCount` at construction. Their output layouts have no field for upstream loss.
Before passing a complete CSR without those views, check all supplied producer evidence. Require
zero overflow flags, a required count within the retained link count and slot capacity, affirmative
completeness and convergence flags, and zero iteration-limit and unresolved-work indicators, wherever
the producer supplies them. Repeat these checks each time the producer writes a new CSR, or use an
immutable verified copy.

{/* lens:GPUCategoryPairCounts */}

{/* lens:GPUSpatialEntropy */}

## Correlogram over contiguity orders: `addContiguityOrderCorrelogramRecipe`

Global Moran's I with analytic inference for the neighbors at exactly 1 to `maximumOrder` steps: per order `GPUSpatialWeightsAlgebra` `higherOrder`,
`GPUSpatialWeightsTransform` `'row'` and `GPUGlobalSpatialStatistics`. `outputs.correlogram` holds `GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS.length` floats per order (I, expectation,
both variances, z and p under normality and randomization, `n`, island count). `outputs.incompleteOrders` is one flag per order; it includes upstream final/candidate loss and either loss introduced by that order. A flagged order publishes NaN Moran statistics and inference. Clearing input loss and resubmitting the same graph restores defined results when the order fits.
Capacities grow with the order: the defaults (`slotCapacity` and `workCapacity`) hold a square lattice, and memory is `maximumOrder` times the order capacity.
`intermediates.orderWeights` contains the algebra's described, row-standardized results. For
orders above one it omits `status.requiredCount`, which can describe retained intermediate work
after candidate loss; the raw algebra diagnostic remains available on the contributor's destination.
The public intermediate keeps `candidateOverflow` so downstream inference still suppresses loss.

Reference: a binary64 breadth-first ring construction and Moran's I with the `esda.Moran` expectation and variances, in `stx08-spatial-methods.py`; libpysal and esda
were not installed. The fixture has an 8 by 8 and a 10 by 6 lattice for orders 1 to 5. The reference excludes rows with no neighbor at an order, as `islands: 'exclude'` does; no fixture exercises an island order.

{/* lens:addContiguityOrderCorrelogramRecipe */}
