# Bounded remote-sensing contributors

These contributors are exported by `@luma-contributors/analysis/raster-analysis`. `GPURasterTexture`, `GPURasterRegistration` and `GPURasterRegionGraph` accept `grid: GPURasterMetadata`; image indexing uses its width and height, and registration shifts remain in column/row pixels. They produce native WebGPU graph nodes with caller-owned output views. They never read observations to the CPU or submit work themselves. Scalars use packed `float32`/`uint32` views; lengths, grid dimensions and model options are graph topology. Observation values and QA buffers can change between executions. Float outputs with no defined answer are NaN, accompanied by counts or status words.

## Conservative remapping (RS03)

`GPURasterConservativeRemapping` takes axis-aligned Cartesian source and destination grids, each with positive spacing and a lower-left corner origin. Source values are row-major. Exact rectangular overlap areas are generated on the GPU into bounded graph scratch and accumulated per destination cell. Grid-origin differences are formed in host double precision before local float32 overlap arithmetic. Each destination may examine at most 256 source candidates; unsupported ratios fail at construction.

`normalization` is required:

- `destination-area`: divide the valid overlap-weighted sum by the full destination cell area. Uncovered and invalid source area contribute zero to the numerator.
- `covered-area`: divide by valid covered area, preserving a constant intensive field on partially covered cells.
- `extensive`: treat values as source-cell totals and redistribute the fraction of each source cell's area.

`output.values`, `output.coveredArea` and `output.status` have one row per destination cell. Status is 0 for a finite result, 1 for no valid overlap, or 2 for nonfinite arithmetic. A dynamic `validity` buffer and nonfinite input values exclude observations from both numerator and covered area. By default geometry weights are graph-owned scratch rebuilt each execution. To retain them, supply `preparedWeights: {indices, areas}` with packed uint32/float32 views, each of length `destination.width * destination.height * stride`, where `stride = (ceil(destination.spacing[0]/source.spacing[0])+1) * (ceil(destination.spacing[1]/source.spacing[1])+1)`. `weightMode: 'prepare'` writes them and applies the current values. After that graph submission completes, import the same caller-owned buffers into another graph and use `weightMode: 'reuse'`: this emits only the sparse accumulation kernel. Both grids, ordering, stride and area units must remain identical; rebuild after geometry changes. Observation values, finite filtering and masks are reread on every application, so QA changes do not invalidate geometry weights. No hidden cache, preparation submission or CPU readback occurs. Curvilinear grids, spherical areas and geographic reprojection are outside this operator.

## GLCM texture (RS04)

`GPURasterTexture` processes one supplied integer `offset` direction at a time. It quantizes to 2–32 levels using a supplied finite increasing `range`, clamps out-of-range values, and builds a private histogram for a square window with radius 1–8. Both endpoints must be inside the clipped window and pass per-cell validity and finite-value checks. Symmetric mode inserts the reverse pair too, including diagonal pairs, and `pairCount` includes those reverse insertions.

`features` has seven floats per cell: contrast, dissimilarity, homogeneity, angular second moment, energy, correlation and natural-log entropy. All properties use normalized pair frequencies. Correlation is 1 when either marginal variance is effectively zero; no pairs produces seven NaNs. `requiredHalo` equals the radius because no pair endpoint outside the window is read. Evaluate on a halo-expanded source and retain only core outputs to avoid tile seams. Multiple directions are separate calls; there is no implicit direction averaging.

## Temporal stacks (RS06)

`GPURasterTemporal` accepts a cell-major regular-time stack of up to 256 slices, with optional **per-observation** QA. Nonfinite observations or zero QA are omitted without compressing original date indices.

`operation` selects `mean`, `minimum`, `maximum`, `median` or `savitzky-golay`. Composites write one result per cell. Extrema use the earliest original slice on ties. The median is the middle stable-sorted observation for odd counts, or the average of the two middle observations for even counts. `chosenSlice` reports the actual selected acquisition; means, even medians and smoothing write `0xffffffff`. `counts` always reports the valid support.

Savitzky–Golay writes one result for every original observation. `windowLength` is odd, 3–15, no longer than the stack; `polynomialOrder` is 0–3 and less than the window length. `edgeMode: 'interp'` (default) shifts a full-width window inward at edges and evaluates the fitted polynomial at the target date; `'truncate'` clips the centered window. Missing observations trigger an actual QR refit on the remaining original time coordinates. Signed coefficients are never renormalized. Status is 0 for a fit, 1 for fewer than degree+1 valid samples, or 2 for rank/nonfinite failure. For composites, status is 0 or 1 for empty support. There is no irregular-time interpretation or implicit resampling.

## Translation registration (RS07)

`GPURasterRegistration` defaults to `mode: 'periodic'`, computing unmasked phase correlation between finite scalar images. Both dimensions are powers of two from 2 to 2048. Graph-native FFT2D produces the spectra; normalized cross-power rejects DC and products at or below `max(1e-20, 1e-6 * maximumProductMagnitude)` so numerically empty spectral bins do not become arbitrary unit phases. An inverse FFT locates the integer peak. A separable localized inverse DFT refines the shift with spacing `1 / upsampleFactor` (integer 1–32, default 20).

`output.shift` is one `float32x2` **(column, row)** translation to apply to the moving image to align it with the reference. For example, an image shifted right by 1.25 pixels should return a column shift near -1.25. `output.peak` is the normalized refined correlation. Status is 0 for an accepted shift, 1 for flat/nonfinite input or an unusable common spectrum (including arithmetic overflow), or 2 for a nonadjacent competing peak within relative `ambiguityTolerance` (default `1e-4`); rejected shifts and peaks are NaN. Neighbor comparisons use periodic distances. Wrap-equivalent shifts differ by the image dimensions.

`mode: 'masked-linear'` accepts arbitrary dimensions 2–512 and optional `referenceValidity` / `movingValidity` views. Nonfinite observations are also treated as missing. It pads each dimension to the next power of two covering the full linear correlation. Six FFT correlations compute paired count, each side's sum and sum of squares, and cross-products. The score is centered normalized cross-correlation over paired-valid support. Full-image valid means are subtracted before transformation to reduce cancellation. Local variances must exceed both `1e-12` and `1e-6` times the corresponding squared-value sum.

`maximumShift` bounds the integer column/row search, defaulting to the full linear range. `minimumOverlapCount` defaults to 3; `minimumOverlapFraction` defaults to 0.3 times the smaller whole-image valid count. Both support requirements must be met. `minimumCorrelation` defaults to zero. Nonadjacent score competition uses the same relative ambiguity tolerance. Optional `output.overlap` records the accepted refined pair count, or zero on failure. Masked status is 0 for accepted, 2 for ambiguity, and 3 for insufficient overlap, constant local support, or no score meeting the minimum. Nonfinite pixels are missing rather than invalidating an entire masked image.

Masked refinement evaluates local shifts at spacing `1 / upsampleFactor` using **bilinear interpolation of moving values and centered paired moments**. Every interpolation corner with positive weight must be in the image and pass its mask; reference cells must pass theirs. There is no wrap, missing-weight renormalization or cross-gap interpolation. Subpixel support can therefore differ from integer support. `upsampleFactor: 1` returns the integer NCC maximum without refinement. Unlike the periodic mode's band-limited DFT refinement, masked refinement maximizes this explicit bilinear NCC objective; the two modes do not promise identical subpixel accuracy. Refined candidates remain inside `maximumShift`.

Window apodization and nonrigid alignment are not provided. The ambiguity diagnostic is an explicit peak rule, not a statistical confidence probability. Accuracy depends on image bandwidth, noise and repeated texture; refinement spacing alone is not a universal error guarantee.

## Region adjacency and merging (RS08)

`GPURasterRegionGraph` consumes complete-domain row-major labels and scalar observations, with up to 256 region IDs `0..regionCount-1`; all other IDs are background. It counts shared four-neighbor boundaries into a symmetric dense `regionCount²` `adjacency` matrix, computes each original region's finite-value mean, and writes absolute mean differences into `weights`. Absent edges and edges touching regions without valid values receive NaN weights.

`mergeMode: 'fixed'` (default) uses `mergeThreshold` to select static edges whose weights are no greater than the threshold. `maximumIterations` (1–256) bounds minimum-source-label propagation along these edges. `mapping` preserves the original region index and contains its smallest reachable source ID after the configured rounds. The single-word `status` is 1 if a further propagation round would change any mapping, otherwise 0. Graph scratch is ping-ponged between dispatches. `mergeMode: 'adaptive'` instead performs one global closest eligible pair merge per round. Component means are recomputed from original finite sample counts and means after each merge, and adjacency is the quotient of original boundaries under the current source mapping. Equal distances choose the lexicographically smallest pair of source roots; the smaller root survives. `maximumIterations` is therefore a merge budget in adaptive mode, with at most `regionCount - 1` meaningful merges. No eligible pair leaves the mapping unchanged. The final `status` checks whether another eligible adaptive merge remains; final `weights` describe current component-mean differences at original source-edge positions and are NaN for edges whose endpoints have merged. `adjacency` retains original boundary counts and `mapping` retains stable original-region indexing.

Adaptive distances use observation-count-weighted scalar means, matching this declared graph objective. Custom merge callbacks, vector color metrics and arbitrary linkage definitions are outside the API. Cross-tile labels and seam adjacency must be reconciled before this complete-domain call.

`addRasterRegionGraphTiles(graph, props)` explicitly assembles a complete partition of disjoint tile cores on the GPU before applying the same region operation. Each tile provides `grid`, integer global `origin`, `labels`, `values`, and optional `sourceRegionIds` mapping local labels to stable global IDs. Core rectangles must cover the destination exactly without overlaps; halos must be excluded. Out-of-range local labels map to background. Supply equivalence IDs from cross-tile components or caller topology: this recipe does not infer them from radiometry. Repeated source IDs combine observations before mean calculation, and seam edges are counted once in the assembled domain. Direct scikit-image hierarchical partitions and tiled/untiled edge counts are tested.

## Spectral unmixing (RS10)

`GPURasterSpectralUnmixing` accepts pixel-major spectra and **endmember-major** supplied spectra, with at most 256 bands and 8 endmembers, and at least as many bands as endmembers. A band is omitted when the pixel observation, any corresponding endmember coefficient, or the optional pixel-band QA is invalid. No calibration or endmember discovery is inferred.

Band and endmember limits are joint: with `b` bands and `m` endmembers, unconstrained fits require `max(b*(m+1)+m+64, b*m+m*m+2*m+32) <= 2048`; constrained fits require `b*(m+2)+4*m+64 <= 2048`. These conservative limits keep explicit working arrays and scalar locals within an 8 KiB per-function budget, not a guarantee about total compiler-generated private memory. Thus 256-band inputs support six unconstrained or five constrained endmembers; eight endmembers support 219 unconstrained or 195 constrained bands. The complete OTB fixture (224 bands, five endmembers) fits within these limits.

`constraint` selects:

- `unconstrained` (default): reorthogonalized modified Gram–Schmidt QR and triangular solve.
- `nonnegative`: projected-gradient NNLS with a conservative Frobenius-norm step bound.
- `simplex`: projected-gradient least squares with exact Euclidean projection onto nonnegative abundances summing to one.

`maximumIterations` is 1–4096 (default 256), and positive `tolerance` defaults to `1e-5`. Convergence compares the infinity norm of the projected-gradient mapping with tolerance times the larger of one and the initial unconstrained gradient norm. `abundances` is pixel-major; `residual` is the sum of squared errors over valid bands; `iterationCounts` is zero for direct QR. Status is 0 for solved, 1 for insufficient bands, 2 for rank/nonfinite failure, or 4 for exhausted iteration budget. Exhausted constrained fits retain feasible abundances and residuals. There is no condition-number certificate or automatic regularization; nearly collinear endmembers may fail QR or exhaust constrained iterations.

## Seasonal decomposition, breaks and monitoring (RS11)

`GPURasterSeasonalBreak` preserves the harmonic default: intercept, normalized linear time, one cosine and sine with `period > 2` slices, comparing no break against one level jump at each eligible acquisition boundary. `seasonality: 'dummy'` instead fits phase indicators 1 through `period-1`, with phase zero as reference and integer periods 3–12. Every acquisition keeps its original regular-time slot. Each break side requires `minimumSegmentLength >= 3` elapsed slots **and** supported training observations.

The BIC score is `n * log(max(SSE, max(1e-20, observationEnergy * 1e-12)) / n) + k * log(n)`. With `p` baseline coefficients, `k=p` for no break and `k=p+2` for a jump plus its selected date. The best strict improvement above `minimumBICGain` wins; exact ties retain the earliest boundary. This float32 stabilization floor is part of the model.

`fitMethod: 'joint'` uses reorthogonalized QR. `fitMethod: 'iterative'` performs cyclic least-squares updates of trend, seasonal terms and jump, starting from zero and recomputing residuals after every sweep. A separate QR rank check rejects unidentifiable models. `maximumIterations` is 1–256 (default 128), and `tolerance` defaults to `1e-5`. Convergence requires the largest coefficient update to be at most `tolerance * (1 + largest absolute coefficient)`. Every eligible candidate is fitted; any valid but exhausted candidate makes the overall result incomplete, even when the selected incumbent converged. The iteration count reports the largest sweep count across candidates.

`missingPolicy` is `omit` by default, `reject` for any missing training observation, or `interpolate` for linear interpolation of interior training gaps. Interpolation never extrapolates leading/trailing gaps or uses observations beyond the training window. Filled observations participate in regression support and BIC; their reported residuals remain NaN to preserve acquisition QA. Interpolation across seasonal transitions changes the fitted model and is explicitly selected by the caller. Irregular acquisition dates require upstream placement on regular slots or a different model; timestamps are never compressed.

Optional `monitoring: {historyLength, threshold, minimumConsecutive}` fits only the initial history, then scores subsequent observations against that fixed historical model. The scale is the selected historical residual standard deviation, using the BIC residual floor and at least one residual degree of freedom. The first run of `minimumConsecutive` same-sign standardized residuals at or beyond `threshold` produces an alarm at the run's **first original acquisition index**. Missing observations reset persistence; later observations never update or interpolate historical data. Re-encode an extended supplied stack to update monitoring explicitly. This is a deterministic residual monitor, not a calibrated BFAST significance boundary or a hidden stateful stream.

`coefficients` has five floats per harmonic cell (intercept, slope per `sliceCount` slots, cosine, sine, jump), or `period+2` for dummy seasons (intercept, slope, phase 1 through phase `period-1`, jump). `components` retains interleaved trend including jump, seasonality and observed residual. `breakIndex` and `improvement` describe the **historical/offline** BIC break, not the monitoring alarm. The optional `diagnostics` output is required for iterative or monitoring modes and has six floats per cell: maximum iterations, convergence flag, supported training count, historical residual scale, monitoring alarm index (`-1` absent), maximum absolute monitoring score. Exhausted fits retain finite incumbent coefficients/components but cannot certify complete breakpoint selection.

Status is 0 for fit, 1 for insufficient support, 2 for rank/nonfinite failure, 4 for a rejected training gap, or 8 for exhausted iterations. The stack limit is 256 slots. With `h` training slots and `c` coefficients including jump, `h*c + c*c + 2*c + 8 <= 2048` bounds private QR storage, and `h*c + h + ceil(h/32) + c + 64 <= 2048` bounds the main fit workspace, each within an 8 KiB explicit working-array budget per kernel function (for example 128 training slots and 12 dummy phases). Iterative work additionally requires `h*c*(h-2*minimumSegmentLength+2)*maximumIterations <= 16,777,216`. Compiler temporaries, helper arguments and total driver private memory are hardware dependent. The equivalent pinned NumPy references verify these selected models; multiple trend/seasonal break counts, automatic period estimation, calibrated inferential p-values and full BFAST equivalence are not claimed.

## Evidence and scaling limits

The focused remote-sensing specs check independent analytic conservation, exact GLCM counts, polynomial reproduction through QA gaps, original-date provenance, known fractional periodic translations, independent masked linear interpolation/NCC fixtures, ambiguous/flat images, incomplete merge propagation, adaptive mean-updating merges, constrained/unconstrained abundances and known seasonal jumps. Pinned NumPy 2.2.6 / SciPy 1.15.3 fixtures independently verify Savitzky–Golay interpolation, least squares, NNLS and simplex-constrained optimization; regenerate with `modules/analysis/test/oracles/raster-remote-sensing.py`. Independent SciPy 1.15.3 `map_coordinates` plus direct centered correlation and NetworkX 3.5 quotient-graph fixtures cover masked registration and adaptive region updates; regenerate with `modules/analysis/test/oracles/raster-registration-regions.py`. Static graph tests parse all shader variants and audit WGSL identifiers/bindings.

`raster-round-two.py` directly executes scikit-image 0.25.2 `graycomatrix` / `graycoprops`, `phase_cross_correlation` and `rag_mean_color` / `merge_hierarchical`; SciPy entropy supplies the seventh texture measure. Its independent NumPy least-squares model selects seasonal breaks using the exact documented BIC floor and penalty, including noisy no-break series and QA gaps. `raster-terra.R` directly executes terra 1.9.50 focalPairs/focalReg and temporal mean/min/max/median, pinned with R 4.4.3. `raster-esmf.py` directly executes ESMPy/ESMF 8.9.1 Cartesian `CONSERVE` with `DSTAREA` and `FRACAREA`: nine coarsened/masked/partial/disjoint cases each reuse the reference route for a second date. Extensive totals use explicit source-density and destination-area conversion. The GPU comparison checks both prepared and reused weights, values, covered areas and absent-overlap status. `raster-seasonal-extended.py` pins NumPy 2.2.6 and independently solves dummy/harmonic regressions, BIC candidates and historical residual monitors; GPU cases additionally verify cyclic convergence against the joint solution, exhausted incumbents, rank-deficient phases, QA gaps and future-data isolation. These fixtures are consumed by GPU tests, not only generated.

| Executed acceptance workload | Verified invariant |
| --- | --- |
| 16,384 exact zonal observations | Weighted mean, support and inverse-CDF endpoints/median; 256 partials and a fixed reduction tree limit float32 accumulation drift. |
| 32,768 categorical IDs | Every class area/proportion, majority/minority ties and conservation. The allocation limit is 65,534 classes subject to the 65,535 total-group budget. |
| 128², 256² and 512² remapping sources | Persistent weights imported into a second graph, three changing dates/QA masks, mass conservation, identical retained weights, one apply kernel per reuse. |
| 128² texture, 32 levels, radius 8 | Exact checkerboard properties and paired counts at maximum documented histogram/window settings. |
| Three masked texture directions, including diagonal symmetry | Whole-image versus halo-expanded tile cores and direct scikit-image properties. |
| Four region tiles with local label maps | Whole-domain/scikit-image hierarchical partition and seam edges; repeated source-region identities across horizontal seams. |
| 128² raster with 64 regions | Every adjacency entry, stable source mapping and bounded convergence. |

These are tested workload sizes, not hardware-independent throughput promises. Region means scan the raster per region; seasonal fitting scans eligible dates per pixel; local GLCM uses private levels-squared storage. Automatic equivalence discovery uses the separate cross-tile component API. Curvilinear/spherical remapping, irregular-time resampling, nonrigid registration, endmember discovery, and full multi-break BFAST inference remain separate models; bounded dummy-season decomposition and fixed-history monitoring are available above. Direct ESMF 8.9.1 Cartesian parity is verified for the documented bounded normalization modes. Direct OTB execution is unavailable on this native Apple Silicon host. RS10 additionally compares every abundance against the published OTB 9.1.1 UCLS golden image, as detailed below; supplied-index formulas and constrained-unmixing checks use pinned Spyndex and SciPy.

### Published OTB unmixing reference

`raster-otb-unmixing.py` downloads and SHA256-verifies the three original OTB 9.1.1 Git LFS objects at commit `154726b379fdb39b481ee8fd793fbd3d584884e6`: the synthetic `hsi_cube.tif`, `endmembers.tif`, and published `apTvHyHyperspectralUnmixing_UCLS.tif` baseline. The [pinned upstream test](https://github.com/orfeotoolbox/OTB/blob/154726b379fdb39b481ee8fd793fbd3d584884e6/Modules/Hyperspectral/Applications/test/CMakeLists.txt) runs `HyperspectralUnmixing -ua ucls` and writes double abundances. The checked-in fixture retains all 256 pixels, all 224 bands and all five endmembers. Inputs are converted to float32 for normal GPU upload; expected coefficients retain upstream float64 precision. The GPU test compares all 1,280 abundances with absolute tolerance `2e-5`, including negative coefficients and noisy residuals. It does not regenerate expected abundances with the implementation under test.

This is parity with **published OTB output**, not a claim that OTB was executed locally. The generator records source hashes and encoding, and preserves upstream Apache-2.0 license and notice files. NNLS and simplex constraints continue to use the independent SciPy references; they are not claimed equivalent to OTB's finite-iteration ISRA or MDMD-NMF algorithms. Separate GPU tests exercise both ends of the joint workspace bounds for unconstrained, nonnegative and simplex solves, including 256-band inputs, eight-endmember inputs and changed signed coefficients. Static audits cover the boundary and rejection immediately beyond it for five through eight endmembers. Each pixel uses private QR or projected-gradient working arrays; work grows with band count and endmember count, and the limit does not promise constant throughput.

## Executable GPU examples

Each lens executes its public contributor on explicit synthetic inputs and exposes real graph and output captures. The seasonal lens fits six historical dummy-season cycles and exposes the first persistent monitoring alarm, with historical fit and monitoring diagnostics separate.

### GPURasterConservativeRemapping

{/* lens:GPURasterConservativeRemapping */}

::lens{scenario=gpu-raster.raster-conservative-remapping rung=graph}

### GPURasterTexture

{/* lens:GPURasterTexture */}

::lens{scenario=gpu-raster.raster-texture rung=graph}

### GPURasterTemporal

{/* lens:GPURasterTemporal */}

::lens{scenario=gpu-raster.raster-temporal rung=graph}

### GPURasterRegistration

{/* lens:GPURasterRegistration */}

::lens{scenario=gpu-raster.raster-registration rung=graph}

### GPURasterRegionGraph

This example assembles two tile cores with different local label numbering through `addRasterRegionGraphTiles`, then captures actual adaptive-merge results and assembly kernels.

{/* lens:GPURasterRegionGraph */}

::lens{scenario=gpu-raster.raster-region-graph rung=graph}

### GPURasterSpectralUnmixing

{/* lens:GPURasterSpectralUnmixing */}

::lens{scenario=gpu-raster.raster-spectral-unmixing rung=graph}

### GPURasterSeasonalBreak

{/* lens:GPURasterSeasonalBreak */}

::lens{scenario=gpu-raster.raster-seasonal-break rung=graph}

### Measured execution phases

The opt-in raster phase harness (`modules/analysis/test/gpu-raster/remote-sensing/raster-performance-README.md`) measures small and larger RS02–RS04/RS06–RS11 workloads, including both mean and Savitzky–Golay temporal operations. It separates preparation, upload, compilation, first execution, resident repetitions, changed-input repetitions and readback; GPU timestamp runs are recorded separately when available. Graph physical transient bytes are measured from compiled allocations. Independent CPU and analytic checks validate initial, repeated and updated outputs. These bounded structured workloads do not certify every option or worst-case performance. The recorded native run retains all twenty measured cases, raw samples and source hashes. Earlier RS01/RS05 measurements remain in the tranche-one record.
