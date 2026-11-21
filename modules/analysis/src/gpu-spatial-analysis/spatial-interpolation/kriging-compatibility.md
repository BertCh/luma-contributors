<!--
SPDX-License-Identifier: MIT
SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
-->

# GPUKriging and GPULocalKriging compatibility (DUP-01)

The two contributors must remain separate until a compatibility adapter can preserve the contracts
below. Sharing the analysis-core LU factorization does not make their statistical models or outputs
interchangeable.

| Contract | GPUKriging | GPULocalKriging |
| --- | --- | --- |
| Query/output shape | Raster cells and optional sample leave-one-out diagnostics | Packed or chunked support quadrature; point, block, drift and multivariate predictions |
| Model | Ordinary semivariogram model, with process nugget and normalized total sill | Covariance/coregionalization model plus independent observation-error variances |
| Duplicate observations | Coalesces identical positions among retained neighbors by their mean; duplicates occupy neighbor slots before coalescing | Preserves independent observations; zero-error duplicate systems can be singular |
| Exact hits | Returns the coalesced observation mean and variance zero before solving | Solves the covariance system; measurement error can smooth an exact hit |
| Coordinates | Raster/index-relative planar coordinates and supported geographic metrics | Native planar coordinates only |
| Selection | Indexed nearest neighbors ordered by squared distance and source row; distinct-position minimum | Supplied candidates, bounded scan or shared sample grid; Euclidean-distance ties by source row |
| Minimum neighborhood | Default `minimumNeighborCount = 3` distinct positions after duplicate coalescing (the exact-hit branch precedes this check) | One usable sample for simple/ordinary (simple allows zero only with `emptyNeighborhood: 'prior'`); universal needs at least 3 for its affine design, or `driftCount` supplied columns; external drift needs `driftCount`; known-mean co-kriging needs one usable sample of any variable; ordinary co-kriging needs at least its active variable constraints, including the target; each design must also have full rank |
| Conditioning | Pivots and pivot ratios tested on the normalized semivariogram augmented system, before any equilibration (none is applied) | Four symmetric Ruiz equilibration passes precede pivot and pivot-ratio tests on the covariance/design system; the same numeric ratio threshold therefore tests a different scaled matrix |
| Diagnostics | Raster validity and invalid/uncertain counts, plus optional cross-validation | Per-query success, insufficient-neighborhood or rank-failure status; retained count and pivot ratio; optional weights |

A future adapter should construct raster-center supports without CPU readback, retain the raster
contributor's duplicate and exact-hit policy as an explicit compatibility mode, translate the
semivariogram/nugget convention into covariance without changing measurement-error semantics, and
preserve both status layouts. Geographic distances require a separate supported metric route; simply
projecting callers' coordinates changes their contract. Tie handling, duplicate truncation and
minimum-neighbor rules need independent oracle cases before selecting a common default. Existing
exact-hit, small-lag, duplicate and conditioning specs remain the acceptance tests.

The private-system neighbor limits remain 16 for GPUKriging and 32 for GPULocalKriging. Local kriging's
64 support points bound the separate quadratic quadrature calculation. Raising the local neighbor
limit requires a storage-backed or cooperative factorization with bounded query batches, device-size
prechecks before declarations, and one shader/pipeline across batches. Raising the shared private
solver capacity alone would increase spilling and cubic invocation work, and is insufficient.

IN05's degree-above-one local RBF route already exists: degree 2 uses six centered/scaled polynomial
columns and is the default for the quintic kernel. This does not provide a global dense RBF fit.
STD-10's shared LU adoption is already present through `equilibrated-solve-wgsl.ts` for local kriging
and RBF, and directly in the raster kriging kernel. This pass retains their conditioning policies.

Automatic RBF indexing still needs a reusable index of usable samples. Indexing raw positions and
taking only the nearest `neighborCount` rows changes results: with a nonfinite nearest value, the
scan skips it and retains a farther usable sample. Per-sample smoothing has the same exclusion rule.
An index must filter these records on the GPU, preserve `(distance(), source row)` ordering even at
squared-distance rounding ties, and reuse one resident build across all query chunks. The current
supplied-neighborhood route remains available; replacing the scan with raw top-k candidates would
violate the existing numerical contract.
