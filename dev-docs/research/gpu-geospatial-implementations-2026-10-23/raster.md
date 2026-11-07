# Raster and remote-sensing GPU prior art: RS01–RS11

Research date: 2026-10-23. Read-only working-tree inspection and primary upstream source review; no kernels were implemented, upstream packages executed, formatting run, or correctness/performance certification performed. Recommendations and numerical targets below are proposed acceptance criteria.

## Reconciled implementation baseline

[modules/graph/src/gpu-raster/index.ts](../../../modules/graph/src/gpu-raster/index.ts) already exports nodata-aware filters, NDVI/band math, histograms, reprojection, fractional coverage/statistics, components, cross-tile components, watershed segmentation, region measurements, and tile halos/cache. Do not reopen these as missing capabilities. [gpu-raster-fractional-coverage.ts](../../../modules/graph/src/gpu-raster/gpu-raster-fractional-coverage.ts) already emits compact `(zoneId, cellId, fraction)` tuples with count/required/overflow and feature validity; its current zonal summary supplies total, mean, covered area and observed area. RS01–RS02 add weighted order statistics and categorical summaries to that structure.

The temporal baseline is substantive: [change-detection/gpu-change-detection.ts](../../../modules/graph/src/gpu-raster/change-detection/gpu-change-detection.ts) supplies difference/log ratio/percent change, Welch tests, Theil–Sen and Mann–Kendall; NaN slices are skipped and the optional mask is per-cell. Its cell-major stack is a useful layout precedent, but per-observation QA needs an explicit additional contract. [gpu-dataframe/temporal-reduction/gpu-temporal-reduction.ts](../../../modules/graph/src/gpu-dataframe/temporal-reduction/gpu-temporal-reduction.ts) already supplies timestamp buckets, counts, extrema and first/last observations, including exact two-word times.

**Roadmap correction:** [modules/graph/src/gpu-core/gpu-fft2d.ts](../../../modules/graph/src/gpu-core/gpu-fft2d.ts) is already graph-native, batched, chunk-aware and uses graph scratch; [gpu-fft1d.ts](../../../modules/graph/src/gpu-core/gpu-fft1d.ts) is also graph-native. The older raster roadmap's encoder-only FFT description is stale. Current supported dimensions remain radix-2, 2–2048; registration should reuse this implementation rather than require a speculative LuFFT migration.

## Eight useful implementation patterns

### 1. GPU keyed weighted accumulation: RS01–RS02

**Actual GPU implementation:** CuPy's [histogram/bincount kernels](https://github.com/cupy/cupy/blob/9aac5cf188c24adf2e6c5051278a1491d0101a1f/cupy/_statistics/histogram.py) include weighted `atomicAdd` accumulation. It is a reusable GPU primitive, not a polygon-zonal algorithm or weighted-quantile implementation. [exactextract operations](https://isciences.github.io/exactextract/operations.html) are the CPU semantic oracle distinguishing cell fraction, optional observation weights, population variance and class proportions.

Adapt existing tuple gathers into lexicographically sorted `(zone, category)` or `(zone, value)` runs. Reuse [gpu-dataframe/group-statistics/group-statistics-sort.ts](../../../modules/graph/src/gpu-dataframe/group-statistics/group-statistics-sort.ts), [group-statistics-order.ts](../../../modules/graph/src/gpu-dataframe/group-statistics/group-statistics-order.ts) and [group-statistics-moments.ts](../../../modules/graph/src/gpu-dataframe/group-statistics/group-statistics-moments.ts); existing unweighted percentiles/mode do not establish weighted equivalence. Add segmented coverage prefixes and a documented weighted-CDF interpolation rule. Accumulate class area separately from counts; expose missing support and effective sample size, with the precise weight interpretation documented.

WGSL baseline cannot assume CUDA floating-point atomics. Prefer sorted segmented reduction or bounded workgroup partials; avoid global compare-exchange float-sum contention. Never average tile quantiles. Benchmark 1M/10M tuples, one giant zone versus many tiny zones, tied classes, zero weights and 50% missing support; require partition-equivalent CDFs and class-area conservation.

### 2. Prepared sparse remapping: RS03

**Actual GPU implementation:** [CuPy CSR multiplication](https://github.com/cupy/cupy/blob/9aac5cf188c24adf2e6c5051278a1491d0101a1f/cupyx/scipy/sparse/_csr.py) dispatches GPU sparse-vector/matrix operations through cuSPARSE. This demonstrates resident sparse application, not GPU generation of geospatial overlap weights. [xESMF's official comparison](https://xesmf.readthedocs.io/en/stable/notebooks/Compare_algorithms.html) documents destination-area versus covered-destination-area normalization; it is an ESMF-backed CPU semantic reference.

Prepare rectilinear overlap weights once, then reuse CSR across dates/bands. Reuse [modules/gpgpu/src/gpu-data/gpu-sparse-matrix.ts](../../../modules/gpgpu/src/gpu-data/gpu-sparse-matrix.ts) data contracts; reuse the existing graph-native [gpu-adaptive-spmv.ts](../../../modules/graph/src/gpu-core/gpu-adaptive-spmv.ts) multiplication contributor; the sparse-matrix container alone is not the execution API. Existing [gpu-raster-reprojection.ts](../../../modules/graph/src/gpu-raster/gpu-raster-reprojection.ts) and [gpu-raster-tile-reprojection.ts](../../../modules/graph/src/gpu-raster/gpu-raster-tile-reprojection.ts) supply grid/validity integration, not conservative remapping parity.

Separate extensive redistribution from intensive averages. Static geometry/mask changes invalidate prepared weights; dynamic observation masks require numerator/support recomputation. WGSL adaptations need bounded 32-bit offsets, local-coordinate overlap calculations and explicit scratch ownership. Benchmark prepared versus rebuild cost on 512²→1024² and irregular-ratio grids across 100 dates. Proposed gate: constant-field and total-mass relative error ≤1e-5 on well-conditioned float32 fixtures, separately checking partially uncovered cells.

### 3. Window-parallel texture extraction: RS04

**Actual CUDA implementation:** HaraliCU's [computeFeatures kernel](https://github.com/andreatangherloni/HaraliCU/blob/a7fce6372228e9fa740291d5d8b3e0d946ed04db/GPU/src/CudaFunctions.cu) assigns autonomous windows to threads; [WindowFeatureComputer](https://github.com/andreatangherloni/HaraliCU/blob/a7fce6372228e9fa740291d5d8b3e0d946ed04db/GPU/src/WindowFeatureComputer.cu) delegates pair/feature computation. This is stronger evidence than a GLCM speedup paper, but its allocation/object machinery is not WGSL-ready.

Use bounded quantization, directions and active windows; compare per-invocation pair lists against workgroup-private histograms rather than allocate `levels² × every pixel`. Reuse the calibrated cooperative tile load in [gpu-raster-neighborhood.ts](../../../modules/graph/src/gpu-raster/gpu-raster-neighborhood.ts) and [gpu-raster-tile-halo.ts](../../../modules/graph/src/gpu-raster/gpu-raster-tile-halo.ts), adding pair-validity logic. Both pair endpoints must be valid; define whether offsets that leave a window are excluded. Halo width must include the actual pair support, not merely the nominal radius.

Benchmark 1024² images at 16/32 levels, 7×7/15×15 windows, four offsets, constant/checkerboard/noisy input and masked seams. Validate raw integer pair counts before normalized Haralick properties against pinned scikit-image CPU calls. Proposed property tolerance: 1e-5 for bounded normalized fixtures; define constant-window correlation explicitly.

### 4. Paired masked stencil moments: RS05

**Actual GPU primitive:** [CuPy ndimage filters](https://github.com/cupy/cupy/blob/9aac5cf188c24adf2e6c5051278a1491d0101a1f/cupyx/scipy/ndimage/_filters.py) implement GPU correlation and separable uniform filtering. They are building blocks, not a claimed implementation of terra focal regression.

Form paired-valid channels `(count, x, y, x², xy, y²)` once, then apply identical window support to every channel. Derive covariance/correlation and simple regression from those sums. Reuse [gpu-raster-neighborhood.ts](../../../modules/graph/src/gpu-raster/gpu-raster-neighborhood.ts), but its `ignore-renormalize` rejects negative coefficients; it cannot directly encode arbitrary derivative or Savitzky–Golay weights. Separate support accumulation from coefficient arithmetic. Reuse small-solve/status conventions from [gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares.ts), without assuming that its public shape is a batched local solver.

Large offsets with tiny variance expose cancellation in `E[x²]−E[x]²`: compare centered/two-pass moments, not just random data. Benchmark paired masks at 0/10/50% gaps, singular neighborhoods, clamp/reflect/nodata borders and whole-domain versus 256² tiles. Measure 7×7 and 15×15 windows; publish covariance error against float64 terra/NumPy fixtures alongside throughput.

### 5. Time-axis convolution with explicit edge fitting: RS06

**Actual GPU implementation:** [CuPy Savitzky–Golay](https://github.com/cupy/cupy/blob/9aac5cf188c24adf2e6c5051278a1491d0101a1f/cupyx/scipy/signal/_savitzky_golay.py) constructs coefficients through least squares, performs `convolve1d`, and handles `mode='interp'` with separate polynomial edge fitting. Thus interpolation edges are real additional work, not equivalent to reflect padding.

Precompute coefficients for complete regular windows; use a bounded masked least-squares fit or an explicit rejection rule for incomplete windows. Renormalizing surviving signed SG coefficients does not preserve polynomial reproduction. Temporal composites should reuse existing reductions where semantics match and add chosen-date/QA provenance rather than reimplement counts/extrema.

Benchmark 256²/1024² cells × 24/120 dates, polynomial degrees 2/3, windows 7/15, missing observations and repeated updates. Tile spatial dimensions while keeping the required temporal window. Gate complete-window polynomial reproduction against SciPy; test edge modes separately, and mark insufficient observations/rank failures invalid. Include transpose/repacking cost when comparing cell-major and time-major layouts. Irregular acquisition dates require explicit resampling or a different fit.

### 6. FFT correlation plus local subpixel refinement: RS07

**Actual GPU implementation:** cuCIM's [phase correlation](https://github.com/rapidsai/cucim/blob/59bd1bdb287ad07416820fab7aaf05761992c101/python/cucim/src/cucim/skimage/registration/_phase_cross_correlation.py) uses CuPy arrays/FFTs and localized upsampled DFT refinement. Its [masked implementation](https://github.com/rapidsai/cucim/blob/59bd1bdb287ad07416820fab7aaf05761992c101/python/cucim/src/cucim/skimage/registration/_masked_phase_cross_correlation.py) is a separate algorithm; do not claim that its mask path automatically preserves ordinary subpixel behavior.

Reuse current graph-native FFT2D, add real/complex packing, normalized cross-power, peak reduction and a small local DFT. Preserve all resources and decision/status buffers in the graph; upstream Python scalar decisions are not a WebGPU command-recording design. Padding and support maps must distinguish periodic wrap from linear translations.

Benchmark 256²–2048² pairs, known fractional shifts, repeated texture, cloud masks and 10–90% overlap. Proposed unmasked gate: ≤0.05-pixel error for clean band-limited synthetic translations with 20× refinement. Masked accuracy needs its own scoped gate. Return ambiguity/insufficient-overlap status for flat or repeated patterns; verify row/column sign convention and independent tile alignment discontinuities.

### 7. Batched harmonic monitoring fits: RS11

**Actual OpenCL implementation:** [diku-dk/bfast](https://github.com/diku-dk/bfast) includes readable [Futhark equations](https://github.com/diku-dk/bfast/blob/4fd0c1727ec6ac309966d0ecfeef67e3ef02776b/bfast/monitor/opencl/futhark/bfastfinal.fut) and generated [OpenCL runtime/kernels](https://github.com/diku-dk/bfast/blob/4fd0c1727ec6ac309966d0ecfeef67e3ef02776b/bfast/monitor/opencl/bfastfinal.py). It builds trend/harmonic designs, fits histories with missing values, predicts residuals and computes monitoring statistics/break outputs. This is BFASTmonitor, not general offline iterative BFAST decomposition or every segmented-fit criterion in RS11.

Reuse per-pixel stack traversal and diagnostics from existing change detection, sharing a small masked-design solver with RS05/RS10. Prefer rank-aware QR over blindly porting normal-equation inversion. Keep a documented regular-time model, minimum history/segment counts and bounded iterations; map compressed valid observations back to original acquisition dates.

Benchmark 65,536 pixels ×120/240 dates with known seasonal harmonics, trends, injected break dates and 0/30/70% gaps. Gate no-break/break classification and date/magnitude recovery separately; compare only the matching monitor model against this implementation. Existing Mann–Kendall and Sen slopes remain distinct reusable outputs.

### 8. Reusable small QR solves, with constraints separate: RS10

**Actual GPU numerical building block:** PyTorch's [CUDA linear-algebra implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp) contains cuSOLVER QR and orthogonal-transform application; [lstsq documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) identifies CUDA's full-rank `gels` assumption. This is neither NNLS nor a complete spectral-unmixing implementation.

For supplied endmembers, reuse the endmember factorization across pixels only when observation masks and weighting permit. Define unconstrained least squares, nonnegative solves and sum-to-one solves as separate contracts. A bounded active-set/projected method needs residual, feasibility and convergence outputs; clipping negative unconstrained coefficients is not NNLS.

Benchmark 256² pixels with 6/12 bands and 3/6 endmembers, highly correlated spectra, missing bands and forced iteration exhaustion. Compare unconstrained residuals with float64 least squares and constrained objectives with a pinned OTB/SciPy NNLS oracle. Proposed well-conditioned gates: relative residual discrepancy ≤1e-4 and abundance feasibility within 1e-5, with conditioning/status reported.

## Remaining compositions and existing scaling work

RS08 should begin with boundary-pair extraction from existing component labels, normalized `(minLabel,maxLabel)` keys, shared grouping for edge weights and explicit merge rounds. Cross-tile equivalence and source-label mapping matter more than importing a general graph package. This review did not establish a directly reusable GPU implementation of scikit-image hierarchical RAG merging.

RS09 is a small calibrated recipe layer over existing band math: name band wavelengths, scale/offset application, QA and denominator rules for NDWI/NBR/EVI/SAVI. It does not justify another generic GPU math system.

The root agent's source inspection found existing watershed segmentation scanning all cells in a single invocation for serial steps. Researching a parallel priority frontier with the **same FIFO/tie semantics** is valuable existing scaling work, adjacent to RS08 rather than a missing operator. Treat its currently observed tiny-grid timings as diagnostic evidence only, not a WebGPU scaling claim.

## Licensing, portability and measurement gates

CuPy's [repository license](https://github.com/cupy/cupy/blob/9aac5cf188c24adf2e6c5051278a1491d0101a1f/LICENSE) is MIT; preserve embedded upstream attribution, including SciPy-derived routines. cuCIM's registration file explicitly declares **Apache-2.0 AND BSD-3-Clause**, not simply its repository Apache license. HaraliCU's [LICENSE](https://github.com/andreatangherloni/HaraliCU/blob/a7fce6372228e9fa740291d5d8b3e0d946ed04db/LICENSE) is LGPL v3; BFAST's [LICENSE](https://github.com/diku-dk/bfast/blob/4fd0c1727ec6ac309966d0ecfeef67e3ef02776b/LICENSE) is GPL v3. Use their algorithmic patterns as research; direct copying requires a deliberate license-compatible decision. PyTorch has a [BSD-style license with third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE); cuSOLVER itself is not portable WGSL source.

All ports need explicit graph hazards and multi-dispatch synchronization: workgroup barriers do not synchronize separate workgroups. Keep invalidity separate from valid zero, calibrate once, exclude halo outputs, preserve chunk ownership, and bound storage/bindings/dispatch. Do not assume portable float64, floating atomics, CUDA warps or unlimited scratch. Report compile, upload, GPU execution, optional readback and peak scratch separately; include CPU crossovers and repeated prepared-input updates. No cited CUDA/OpenCL timing establishes browser WebGPU speedup.
