// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisOperator} from '../spatial-analysis-ontology';

/** Qualified graph and array forms. Source documentation is retained as contract notes;
 * unknown notes must not be interpreted as proof of semantic parity. */
export const SPATIAL_OPERATION_FORMS: Readonly<Record<string, GPUAnalysisOperator['forms']>> = {
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPettittTest': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUPettittTestOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUPettittTestOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePettittTestOnCPU',
        signature:
          'function computePettittTestOnCPU( values: Float32Array, options: CPUPettittTestOptions ): CPUPettittTestResult',
        supportedInputs: ['values: Float32Array', 'options: CPUPettittTestOptions'],
        differences: [
          "/**\n * Computes Pettitt's statistic from average ranks, with the contributor's first-maximum rule.\n * Non-finite values are skipped. A constant series of length >= 2 returns its second finite slice.\n *\n * What differs from the GPU contributor:\n * - Uses float64 ranks and the p-value formula, rounding p-values once; compare within 2e-6 absolute.\n * - Replaces quadratic pair comparisons by sorting and rank prefixes.\n * - Accepts up to 131071 slices instead of 512; the retained limit protects the uint32 statistic.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocationChangePoints': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPULocationChangePointsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULocationChangePointsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeLocationChangePointsOnCPU',
        signature:
          'function computeLocationChangePointsOnCPU( values: Float32Array, parameters: Float32Array, options: CPULocationChangePointsOptions ): CPULocationChangePointsResult',
        supportedInputs: [
          'values: Float32Array',
          'parameters: Float32Array',
          'options: CPULocationChangePointsOptions'
        ],
        differences: [
          "/**\n * Exact unpruned optimal partitioning of each finite series, using float64 running central\n * moments, true slice positions and the contributor's relative variance floor. Starts are visited\n * in ascending order; a strict improvement keeps the smallest preceding start on exact ties.\n * The earliest changes are kept on overflow, while required counts and optimum remain unclamped.\n *\n * What differs from the GPU contributor:\n * - Costs and partition comparisons use float64 with one output rounding. Near ties may choose\n *   different partitions; compare well-separated optima with relative cost tolerance 3e-4.\n * - The GPU's 512-slice, 256-slope-slice and 128-change capacity limits are dropped; uint32 shape\n *   and index limits remain. The full quadratic dynamic program is still evaluated.\n * - Normal costs use central moments in data units rather than float32 standardized moments.\n *   Poisson partitioning uses centered deviance and a float64 entropy series without the GPU's\n *   degree cap; the common likelihood constant is restored only for optimalCost. The relative\n *   variance floor and constant-series special cases are preserved. Both forms evaluate the\n *   first segment without subtracting and re-adding the penalty.\n * - Negative Poisson observations (outside the documented nonnegative-count domain) yield no\n *   changes and NaN cost. Invalid packed penalty/segment-length parameters throw naming parameters.\n * - Subnormals are retained, and finite float64 intermediate costs may exceed float32 range;\n *   only the published optimalCost is rounded, potentially to infinity.\n *\n * @throws If values does not exactly match the cube dimensions or parameters is malformed.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPointToCell': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUPointToCellOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUPointToCellOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePointToCellOnCPU',
        signature:
          'function computePointToCellOnCPU( positions: Float32Array, options: CPUPointToCellOptions, mask?: Uint32Array ): CPUPointToCellResult',
        supportedInputs: [
          'positions: Float32Array',
          'options: CPUPointToCellOptions',
          'mask?: Uint32Array'
        ],
        differences: [
          "/**\n * Converts packed float32 longitude/latitude rows to the contributor's global grid keys.\n *\n * What differs from the GPU contributor:\n * - S2 uses float64 trigonometry and projection, so keys near cell edges can differ, especially\n *   at levels 26–30. Exact quadrant reduction preserves poles and equivalent ±180 longitudes.\n * - H3 uses binary64 projection for every precision option; it does not emulate float32 or\n *   double-float intermediates. Cell-edge decisions can differ from either GPU precision.\n * - Nothing differs for Quadbin, quadkey or geohash keys: their integer-defined cell methods\n *   are retained. Quadbin/quadkey fixed-point arithmetic uses exact BigInt intermediates.\n * - Empty arrays and zero-row named graph operations return empty cells and validity arrays.\n * - Non-finite rows produce zero keys. H3 also returns zero for magnitudes at least 3e38,\n *   matching its kernel guard. Other finite S2/H3 longitudes are reduced modulo 360;\n *   S2 clamps latitude to ±90, H3 treats it as a periodic angle. Subnormals are retained.\n *\n * @throws Error naming positions or mask when packed array lengths are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCellMeasures': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUCellMeasuresOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUCellMeasuresOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeCellMeasuresOnCPU',
        signature:
          "function computeCellMeasuresOnCPU( cells: Uint32Array, options: CPUCellMeasuresOptions = {family: 'h3'} ): CPUCellMeasuresResult",
        supportedInputs: ['cells: Uint32Array', "options: CPUCellMeasuresOptions = {family: 'h3'}"],
        differences: [
          "/**\n * Decodes exact integer H3, S2 and Quadbin keys and measures their boundaries in float64, rounding only\n * the returned rows. Invalid keys produce zeros; edgeCounts is never truncated by the stride.\n *\n * What differs from the GPU contributor:\n * - Every precision option computes in binary64 here; 'double-single' (aliases 'fp64', 'extended') selects hi/lo float32 arithmetic only on the GPU.\n * - All boundaries use binary64 face projection, including pentagons and face crossings at\n *   resolutions 11–15 where the GPU's float32 fallback is unreliable. Near-zero lengths and\n *   areas still have binary64 coordinate error; no float32 intermediate is emulated.\n * - S2 uses binary64 unit-vector differences rather than the GPU local-increment evaluation.\n * - Empty cells arrays are accepted by the pure function; graph operations require rowCount >= 1.\n * - Nothing differs in invalid-key handling, units, boundary order, padding or stride truncation.\n *\n * @throws Error naming cells if it does not contain packed uint32x2 rows.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUS2CapCovering': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUS2CapCoveringOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUS2CapCoveringOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeS2CapCoveringOnCPU',
        signature:
          'function computeS2CapCoveringOnCPU( caps: Float32Array, options: CPUS2CapCoveringOptions ): CPUS2CapCoveringResult',
        supportedInputs: ['caps: Float32Array', 'options: CPUS2CapCoveringOptions'],
        differences: [
          "/**\n * Exact hierarchy traversal for the contributor's fixed-level spherical-cap cover. Cells are\n * ordered by ascending ID within each cap; unused output rows are zero.\n *\n * What differs from the GPU contributor:\n * - Float64 geometry replaces float32; classifications within GPU rounding error may differ.\n *   The exterior expansion/interior shrink tolerance and inclusive bounds are retained.\n * - Omitted frontierCapacity retains complete adaptive traversal, including level-synchronous\n *   requests. An explicit frontierCapacity honors the GPU refinement bound and pruning slack;\n *   frontier exhaustion writes zero cells/counts and status bit 4. The GPU defaults to a bound.\n * - Levels up to 30 are accepted, dropping both GPU level caps. The uint32 output-count layout\n *   remains; a complete count above 0xffffffff throws instead of wrapping.\n * - Empty arrays are accepted by the pure function; graph operations require at least one cap.\n *   Invalid cap rows return status 1 and zero outputs. The radius upper bound is the GPU's\n *   float32 pi bound, so a float32 encoding of pi remains valid.\n *\n * @throws Error naming caps for incomplete packed rows, or requiredCount for unrepresentable counts.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUS2PolygonCovering': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUS2PolygonCoveringOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUS2PolygonCoveringOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeS2PolygonCoveringOnCPU',
        signature:
          'function computeS2PolygonCoveringOnCPU( input: CPUS2PolygonCoveringInput, options: CPUS2PolygonCoveringOptions ): CPUS2PolygonCoveringResult',
        supportedInputs: [
          'input: CPUS2PolygonCoveringInput',
          'options: CPUS2PolygonCoveringOptions'
        ],
        differences: [
          '/**\n * Adaptive S2 cover, in ascending cell-ID order, with orientation-independent shell and\n * hole subtraction, inclusive tolerance bands and invalid-ring status.\n *\n * What differs from the GPU contributor:\n * - Predicates and geometry use float64; cell-edge decisions within GPU rounding error can differ.\n * - Only an explicit maximumVisitedCells bounds traversal. Omission retains a complete CPU\n *   traversal; the GPU defaults to 65,536. Serial exhaustion retains its emitted prefix;\n *   level-synchronous exhaustion clears cells/counts. Both set bit 4. Frontier overflow counts\n *   only cells already visited; classification exhaustion reports the explicit visit budget.\n * - Levels up to 30 are accepted (the S2 key layout limit), dropping the GPU level-15 precision\n *   cap and fixed stack. The uint32 counts/visitedCount layout limits still apply.\n * - Malformed ring ranges, non-unit/non-finite vertices and degenerate rings return status 1,\n *   as on the GPU. Inconsistent packed array lengths throw. Unreferenced vertices are ignored.\n *\n * @throws Error naming positions or ringOffsets for inconsistent packed arrays, or naming the\n * uint32 output whose complete count cannot be represented.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUDelaunayTessellation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUDelaunayTessellationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUDelaunayTessellationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeDelaunayTessellationOnCPU',
        signature:
          'function computeDelaunayTessellationOnCPU( input: CPUDelaunayTessellationInput, options: CPUDelaunayTessellationOptions ): CPUDelaunayTessellationResult',
        supportedInputs: [
          'input: CPUDelaunayTessellationInput',
          'options: CPUDelaunayTessellationOptions'
        ],
        differences: [
          "/**\n * Exact Delaunay construction by sequential Bowyer-Watson cavities with symbolic vertices at\n * infinity and the contributor's row-order lifted perturbation. No finite super triangle loses\n * thin hull triangles. Duplicates belong to the first row; triangles are canonical and grouped.\n *\n * What differs from the GPU contributor:\n * - Sequential cavities replace parallel insertion/Lawson rounds. No GPU step, scratch or\n *   command-node limit applies; iterationLimitReached is always zero. iterationCount counts\n *   distinct finite sites inserted into valid groups, rather than GPU insertion/flip rounds.\n * - Exact arbitrary-precision fallbacks accept every finite float32 exponent span; the GPU\n *   rejects spans above 55. Subnormals retain their exact values.\n * - Nonfinite sites cancel their group (or the whole ungrouped result), as on the GPU. Global\n *   invalidCount counts nonfinite rows only for ungrouped input; grouped validity is per group.\n * - Unused output capacity is zeroed. The GPU leaves unused triangle rows unwritten.\n *\n * @throws Error naming positions, groupIds, groupCount or capacity for inconsistent inputs.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTransitionDiagnostics': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTransitionDiagnosticsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTransitionDiagnosticsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTransitionDiagnosticsOnCPU',
        signature:
          'function computeTransitionDiagnosticsOnCPU( counts: Uint32Array, options: CPUTransitionDiagnosticsOptions ): CPUTransitionDiagnosticsResult',
        supportedInputs: ['counts: Uint32Array', 'options: CPUTransitionDiagnosticsOptions'],
        differences: [
          "/**\n * Computes homogeneity, stationary distributions and sparsity in the GPU's scalar row layouts.\n * Stationarity checks strong connectivity on the count support, then uses float64 GTH elimination.\n * Empty chains return NaNs without clearing `converged`; any failed nonempty chain clears it.\n * Extra counts after conditionCount * classCount² are unused capacity, as on the GPU.\n *\n * What differs from the GPU contributor:\n * - Uses float64 sums and elimination, rounding each float output once; ordinary inputs should\n *   agree within 1e-5 relative for statistics and 2e-6 absolute for stationary probabilities.\n * - Uses the public float64 chi-square survival function at df=1 or 2 and a finite integer-df\n *   recurrence, without the GPU's incomplete-gamma iteration cap or large-df normal approximation.\n *   GPU p-values allow 2e-4 absolute error (including its large-df approximation).\n * - Drops the 64-class private-array limit; output layout imposes no corresponding class limit.\n * - Aggregated counts use float64 rather than float32; individual input and output words stay uint32.\n * - minimumTransitions is not capped at uint32, unlike the GPU's integer sparsity diagnostic.\n * - Rescales stationary weights during back-substitution, staying finite where the GPU's unscaled\n *   float32 weights overflow. A non-finite or non-positive normalization leaves the chain NaN and\n *   sets converged to 0; the GPU also clears converged if its unscaled weights overflow.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURankMarkov': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPURankMarkovOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPURankMarkovOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeRankMarkovOnCPU',
        signature:
          'function computeRankMarkovOnCPU( values: Float32Array, options: CPURankMarkovOptions, mask?: Uint32Array ): CPURankMarkovResult',
        supportedInputs: [
          'values: Float32Array',
          'options: CPURankMarkovOptions',
          'mask?: Uint32Array'
        ],
        differences: [
          '/**\n * Complete-case ordinal-rank Markov transitions, including all unconditional diagnostics.\n *\n * What differs from the GPU contributor:\n * - Arithmetic uses float64 and rounds once at the outputs; use 2e-6 relative tolerance for floats.\n * - GPU pairwise ranking and unconditional diagnostics are limited to 64 locations; sorted ranking supports up to 65535 logical locations subject to dense output bindings. The 65536-period cap is dropped by the CPU. Dense output sizes and total\n *   transition opportunities must fit uint32.\n * - Sorting replaces pairwise ranking. Tie order and missing ranks are unchanged.\n * - Diagnostics call the public float64 computeTransitionDiagnosticsOnCPU, including its\n *   rescaled GTH stationary solver. Empty chains keep converged=1 and NaN steady states.\n *\n * @throws If values is not a balanced panel or mask does not have exactly one word per location.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalFrameProjection': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPULocalFrameProjectionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULocalFrameProjectionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeLocalFrameProjectionOnCPU',
        signature:
          'function computeLocalFrameProjectionOnCPU( positions: Float32Array | Uint32Array, frame: GPULocalFrame, bounds: ProjectionBounds ): CPULocalFrameProjectionResult',
        supportedInputs: [
          'positions: Float32Array | Uint32Array',
          'frame: GPULocalFrame',
          'bounds: ProjectionBounds'
        ],
        differences: [
          "/**\n * Calls the frame's public binary64 projectArray and writes the same absolute frame rows as\n * GPULocalFrameProjection. Uint32 inputs encode little-endian Float64Array word pairs.\n *\n * What differs from the GPU contributor:\n * - Calls projectArray directly, avoiding compiled polynomial patches and their approximation\n *   error. tolerance only controls the GPU patch compiler and is unused by the CPU calculation.\n * - Bounds rejection remains inclusive. Non-finite source or projected rows are rejected with\n *   zero output and validity = 0, including singularities. The CPU accepts domains the GPU\n *   patch compiler may refuse before execution.\n * - Supports only one packed graph chunk, in float32x2 or raw-binary64 uint32x4 format.\n * - Rounds absolute frame meters once at the output, rather than rounding polynomial terms.\n *\n * @throws If positions contains an incomplete row or bounds is inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGeodesicPairs': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUGeodesicPairsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUGeodesicPairsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeGeodesicPairsOnCPU',
        signature:
          'function computeGeodesicPairsOnCPU( input: CPUGeodesicPairsInput, spatialContext: GPUSpatialContext, maximumIterations?: number ): CPUGeodesicPairsResult',
        supportedInputs: [
          'input: CPUGeodesicPairsInput',
          'spatialContext: GPUSpatialContext',
          'maximumIterations?: number'
        ],
        differences: [
          "/**\n * Measures packed pairs in float64. Spherical central angle, bearings, seam wrapping and slerp\n * follow geodesic-wgsl.ts step by step; ellipsoidal inverse calls public math.gl Karney geodesy.\n *\n * What differs from the GPU contributor:\n * - Without an explicit budget, ellipsoidal distance/bearings use EllipsoidGeodesicExact,\n *   including near antipodes, and valid ellipsoidal rows have converged=1. An explicit\n *   maximumIterations bounds auxiliary-longitude convergence with float32 updates and the\n *   GPU stopping criterion; exhaustion uses spherical distance/bearings\n *   and reports converged=0, approximation=1, iterationLimitReached=1.\n * - Rhumb stretched latitudes use the full float64 closed form, without the short-line/series\n *   approximations or the 0.999999 atanh clamp. The latitude clamp is the GPU's float32 value.\n * - Except for the explicit-budget convergence gate, arithmetic is float64 until output.\n *   Bearings that round to -180 are normalized to +180;\n *   zero bearings are positive zero. Longitude half-turn ties use round-to-even.\n * - Nonfinite coordinates or latitudes outside [-90, 90] emit NaN and converged=0; GPU results\n *   for these inputs are unspecified. Exactly antipodal spherical midpoints are nonunique;\n *   midpoints within 1e-6 radians of an antipode are NaN on both paths.\n * - approximation marks spherical bearing/midpoint conditioning near antipodes on both paths,\n *   and the spherical fallback when an explicit inverse budget does not converge.\n * - Midpoints remain spherical for the ellipsoidal model, as on the GPU. Rhumb midpoint latitude\n *   is the original (unclamped) mean, matching the current kernel.\n * - For (20,90) -> (20,90), the CPU midpoint is [0,90], choosing longitude zero via JavaScript\n *   atan2(0, 0). Pole midpoint longitudes follow JavaScript's signed-zero atan2 convention.\n *   Both paths guard zero-axis atan2 and select longitude zero for the pole midpoint.\n *\n * @throws If origins/targets are not aligned packed float32x2 rows or the context is unsupported.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUHealpixPointToCell': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUHealpixPointToCellOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUHealpixPointToCellOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeHealpixPointToCellOnCPU',
        signature:
          'function computeHealpixPointToCellOnCPU( positions: Float32Array, options: {order: number; scheme?: GPUHealpixIndexScheme; mask?: Uint32Array} ): CPUHealpixPointToCellResult',
        supportedInputs: [
          'positions: Float32Array',
          'options: {order: number; scheme?: GPUHealpixIndexScheme; mask?: Uint32Array}'
        ],
        differences: [
          "/**\n * Projects packed longitude/latitude degrees to nested or UNIQ keys, little-endian low/high\n * words, with validity zero for masked or non-finite coordinates. Latitude clamps to [-90,90].\n *\n * What differs from the GPU contributor:\n * - Projection arithmetic uses float64. Keys near cell boundaries can differ from the GPU's\n *   float32 projection, especially at high orders; input float32 quantization remains.\n * - Longitude reduction uses remainder, avoiding cancellation for very large finite longitudes.\n * - The order limit 29 is retained for the HEALPix 64-bit key layout. Empty arrays are accepted\n *   by the pure function and named graph forms; the direct GPU contributor requires at least one row.\n *\n * @throws Error naming positions or mask when packed arrays have inconsistent lengths.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULabelPoint': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPULabelPointOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULabelPointOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeLabelPointOnCPU',
        signature: 'function computeLabelPointOnCPU(input: CPUPolygonRows): CPULabelPointResult',
        supportedInputs: ['input: CPUPolygonRows'],
        differences: [
          "/**\n * Finds polygon poles with unbounded best-first subdivision in binary64.\n * Emptiness and bounds use the GPU's contiguous feature interval, while signed distances and\n * scanline seeds walk individual rings. This distinction also applies to descending offsets.\n *\n * What differs from the GPU contributor:\n * - Searches every cell whose distance bound can improve the answer, rather than a capped set\n *   of coarse grid lobes. Stops at 1e-6 of the feature's maximum extent (binary64 resolution\n *   is the lower limit), independent of all GPU grid and refinement settings.\n * - Equal distances retain the existing winner. The first vertex and GPU scanline seeds keep\n *   their order; subsequent candidates use bounds-center/quadtree order instead of grid order.\n * - Non-finite features return NaN points/distances and degenerate = 1; GPU label arithmetic\n *   does not define a non-finite status. Empty features have that same output.\n * - Collapsed-axis features return the first vertex and zero distance without subdivision.\n * - Empty position storage is accepted for empty features; the GPU requires at least one row.\n * - Computes in binary64 and rounds each output once; no float32 intermediate rounding.\n *\n * @throws If positions or any offsets array is inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineSimplification': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPULineSimplificationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULineSimplificationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeLineSimplificationOnCPU',
        signature:
          'function computeLineSimplificationOnCPU( input: CPULineSimplificationInput, options: CPULineSimplificationOptions = {} ): CPULineSimplificationResult',
        supportedInputs: [
          'input: CPULineSimplificationInput',
          'options: CPULineSimplificationOptions = {}'
        ],
        differences: [
          "/**\n * Computes a complete Douglas-Peucker split tree or sequential Visvalingam-Whyatt effective\n * areas, then selects rows using the packed tolerance. Intermediates stay in float64, including\n * comparisons to tolerance; the returned importance column is rounded only after selection.\n *\n * What differs from the GPU contributor:\n * - An explicit maximumIterations binds the CPU rounds; omission leaves the CPU uncapped. Douglas-Peucker is\n *   complete; Visvalingam uses global smallest-(area,row) sequential removal per line, with the\n *   effective-area floor propagated to surviving neighbors, rather than a parallel approximation.\n * - Distances/areas use float64 (ordinary division and square root), not the GPU's defined f32\n *   floor operations. Near ties and tolerance boundaries can therefore select different rows.\n * - At the explicit limit, unresolved Visvalingam rows have infinite importance; unresolved\n *   Douglas-Peucker rows retain the parent split bound. converged is 0.\n *   iterationCount is the logical split-round count for\n *   Douglas-Peucker (honoring finishSpanLimit), or the maximum sequential removals in one line\n *   for Visvalingam (at least 1). The latter is not the GPU's parallel round count.\n * - Unused ids slots are zero; the GPU leaves them unwritten. Only the first count slots are\n *   meaningful. Per-line counts/starts are unclamped, as on the GPU.\n * - Nonfinite positions or timestamps used by the metric throw an Error naming the argument. GPU results for them\n *   and subnormal arithmetic are unspecified; finite subnormals are evaluated in float64 here.\n * - Finite inputs whose GPU intermediates overflow are evaluated in float64. If the final\n *   importance exceeds the float32 range, it rounds to positive Infinity.\n *\n * @throws For inconsistent arrays, nonmonotone/out-of-range offsets, or unsupported options.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocationAllocation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPULocationAllocationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULocationAllocationGraphOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeLocationAllocationOnCPU',
        signature:
          'function computeLocationAllocationOnCPU( input: CPULocationAllocationInput, options: CPULocationAllocationOptions ): CPULocationAllocationResult',
        supportedInputs: [
          'input: CPULocationAllocationInput',
          'options: CPULocationAllocationOptions'
        ],
        differences: [
          "/**\n * Greedy location-allocation followed, when enabled, by best improving single-facility swaps.\n * Reachable weight precedes reachable cost in p-median; ties select the lowest facility, and\n * swaps the lowest removed facility then lowest candidate. Set covering counts rows, including\n * zero-weight demand. Every finite nonnegative cost is reachable; positive infinity is unreachable.\n *\n * What differs from the GPU contributor:\n * - The explicit or default swap budget binds both forms; zero disables swaps. The CPU drops\n *   the GPU's 1000-iteration admission cap, command-node and swap-pair storage limits.\n * - Swaps accept every strictly positive lexicographic improvement, without the GPU's 1e-6\n *   noise tolerance. This can change selected facilities, swap counts and convergence evidence.\n * - Scores, gains and objectives use float64 with one float32 output rounding. Near ties may\n *   differ. Nonfinite demand weights use IEEE\n *   arithmetic and may produce NaN objectives.\n * - Inconsistent lengths throw. Invalid costs, weights, thresholds and sparse structure retain\n *   invalidCount status instead of throwing. Sparse row validation walks clamped row ranges;\n *   GPU merge-path traversal requires monotonic offsets, so its extra row-error count is not\n *   specified for decreasing offsets. The CPU reports each offending row entry once per range.\n *\n * @throws Error naming arrays whose lengths do not match the dimensions.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMapColoring': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUMapColoringOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUMapColoringOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeMapColoringOnCPU',
        signature:
          "function computeMapColoringOnCPU( offsets: Uint32Array, neighbors: Uint32Array, parameters: Uint32Array, priority: GPUMapColoringPriority = 'largest-degree-first', maximumIterations?: number ): CPUMapColoringResult",
        supportedInputs: [
          'offsets: Uint32Array',
          'neighbors: Uint32Array',
          'parameters: Uint32Array',
          "priority: GPUMapColoringPriority = 'largest-degree-first'",
          'maximumIterations?: number'
        ],
        differences: [
          '/**\n * Colors an unchunked CSR in descending (degree, seeded hash), then ascending row order.\n * Self slots are ignored in color selection but count toward the degree, as on the GPU.\n *\n * What differs from the GPU contributor:\n * - Without an explicit round budget, completes one priority-ordered greedy sweep.\n * - With a budget, visits rows in ascending order per round, retaining uncolored rows when\n *   a higher-priority neighbor is still uncolored. GPU scheduling can finish different rows\n *   within a round; both forms bind the supplied maximum and report actual convergence.\n * - The 1024-round GPU limit is dropped. Chunked CSR is not supported by either operation.\n * - On asymmetric adjacency the unbudgeted CPU sweep follows priority order; GPU scheduling can\n *   produce different colors and conflicts. Symmetric adjacency gives the same complete coloring.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMinimumBounds': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUMinimumBoundsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUMinimumBoundsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeMinimumBoundsOnCPU',
        signature:
          'function computeMinimumBoundsOnCPU(input: CPUMinimumBoundsInput): CPUMinimumBoundsResult',
        supportedInputs: ['input: CPUMinimumBoundsInput'],
        differences: [
          "/**\n * Computes group hulls and their bounding shapes in float64, rounding each output once.\n * Hulls use the documented global power-of-two lattice, exact integer orientation and lowest-row\n * representatives. Rectangle ties retain the first edge within the documented 1e-6 relative band.\n *\n * What differs from the GPU contributor:\n * - No hull budget: every hull is computed, overflow and groupOverflow are always zero, and maximumHullVertices on\n *   the operation is accepted for interchangeability but does not truncate the CPU result.\n * - Lattice extent/subtraction and shape arithmetic use float64. Near lattice boundaries hulls\n *   can differ from float32 quantization; diameter ties use float64 squared distances.\n * - Rectangle and diameter search exhaust all edges/pairs instead of using calipers.\n * - One-/two-vertex hull rectangles copy their corners directly and have exactly zero height\n *   and area. GPU projection/reconstruction can leave float32 cancellation residuals in these\n *   quantities and in other mathematically zero computed coordinates.\n * - Rectangle angles are range-normalized after float32 rounding to stay in (-pi, pi].\n * - Circles use the same Philox shuffle, but float64 containment slack (16 epsilon) replaces\n *   4e-6, and only exactly collinear triples use a diametral circle (no near-collinear cutoff).\n * - Nonfinite positions and out-of-range labels are excluded, as on the GPU. Empty groups emit NaN.\n * - The GPU's row-count cap is dropped; uint32 group/output indexing limits remain.\n *\n * @throws If positions, labels or any offset array has an inconsistent length or range.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMultipleTestingAdjustment': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUMultipleTestingAdjustmentOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUMultipleTestingAdjustmentOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeMultipleTestingAdjustmentOnCPU',
        signature:
          'function computeMultipleTestingAdjustmentOnCPU( pValues: Float32Array, method: GPUMultipleTestingMethod, minimumFamilySize = 0 ): CPUMultipleTestingAdjustmentResult',
        supportedInputs: [
          'pValues: Float32Array',
          'method: GPUMultipleTestingMethod',
          'minimumFamilySize = 0'
        ],
        differences: [
          '/**\n * Calls the public float64 adjustment and rounds only the output. testedCount is the family size,\n * including minimumFamilySize, even for method none.\n *\n * What differs from the GPU contributor:\n * - Float64 arithmetic replaces float32 intermediate arithmetic; compare with relative tolerance 2e-6.\n * - Benjamini-Yekutieli uses a binary64 Euler-Maclaurin expansion above 64 terms.\n * - The GPU sort row and minimumFamilySize limits at 2^31 are dropped; the uint32 family-size\n *   layout limit remains. The logical family size does not allocate virtual rows: only supplied\n *   p-values are stored and sorted, and the harmonic factor takes bounded work.\n * - Subnormal p-values may be flushed to zero on the GPU and are kept by the twin.\n */',
          'CPUMultipleTestingAdjustmentResult contains adjustedPValues and one-word uint32 counts: finiteTestCount counts finite p-values in [0,1]; effectiveFamilySize = max(finiteTestCount, minimumFamilySize). observedTestCount is the deprecated finiteTestCount spelling; familySize and testedCount are deprecated effectiveFamilySize spellings (including padding, even for method none). Named operations expose the same optional count destinations; disagreeing aliases reject.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNeighborhoodSummary': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUNeighborhoodSummaryOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUNeighborhoodSummaryOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeNeighborhoodSummaryOnCPU',
        signature:
          'function computeNeighborhoodSummaryOnCPU( input: CPUNeighborhoodSummaryInput, options: CPUNeighborhoodSummaryOptions = {} ): CPUNeighborhoodSummaryResult',
        supportedInputs: [
          'input: CPUNeighborhoodSummaryInput',
          'options: CPUNeighborhoodSummaryOptions = {}'
        ],
        differences: [
          "/**\n * Computes numeric summaries by sorting and categorical summaries with a frequency map. Members\n * and row-standardization follow the kernel, including zero-weight focal members. CSR offsets\n * are clamped to capacity and reversed ranges are empty, as explicitly defined by the kernel.\n *\n * What differs from the GPU contributor:\n * - Arithmetic is float64 with one output rounding (relative tolerance 3e-5 for conditioned\n *   positive outputs). Uint32 values first undergo the contributor's float32 conversion.\n * - Medians have no member cap; maximumNeighbors is ignored and output overflow is always zero.\n *   Incomplete weights invalidate included rows. The helper reads `status.overflow`,\n *   `status.candidateOverflow`, `status.requiredCount`, and `status.complete`. A nonzero\n *   `status.iterationLimitReached` or zero `status.converged` also invalidates included rows.\n *   Supply aggregate stopping flags at matrix scope. The existing `weightsIterationLimitReached`\n *   and `weightsConverged` inputs also report matrix-level loss. The helper rejects\n *   `status.unresolvedCount`, `status.invalidCount` and `status.approximation`.\n *   Masked rows still have count zero.\n * - Weighted sums retain finite float64 intermediates where GPU float32 may overflow.\n * - Category 0xffffffff is reserved for missing categories/no mode and excluded from categorical\n *   membership. Ties use the lowest category. Subnormal values and weights are retained.\n *\n * @throws If aligned arrays disagree in length, or a required numeric column is absent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConstrainedDelaunayTriangulation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUConstrainedDelaunayTriangulationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUConstrainedDelaunayTriangulationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeConstrainedDelaunayTriangulationOnCPU',
        signature:
          'function computeConstrainedDelaunayTriangulationOnCPU( input: CPUConstrainedDelaunayTriangulationInput, options: CPUConstrainedDelaunayTriangulationOptions ): CPUConstrainedDelaunayTriangulationResult',
        supportedInputs: [
          'input: CPUConstrainedDelaunayTriangulationInput',
          'options: CPUConstrainedDelaunayTriangulationOptions'
        ],
        differences: [
          '/**\n * Triangulates polygon slices and flips strictly illegal interior edges to completion. The first\n * illegal pair/side in slice order is flipped, as in the GPU kernel; cocircular edges stay put.\n *\n * What differs from the GPU contributor:\n * - maximumFlips is ignored: Lawson flipping continues until no strictly illegal convex\n *   quadrilateral remains. Every valid polygon then has converged=1; failed polygons have 0.\n * - The initial triangulation uses budgeted un-hashed Earcut and the selected monotone route,\n *   as documented by computePolygonTriangulationOnCPU. GPU hashing can change the initial slices.\n * - Arbitrary-precision predicates accept all finite float32 exponent spans, without the GPU\n *   incircle fallback above span 55. Midpoints/crossings in the initial Earcut use float64.\n * - Inconsistent arrays throw with the argument name. Nonfinite live rings fail; subnormals\n *   retain their exact values. Output slices, padding and allocation limits are retained.\n *\n * @throws Error naming the inconsistent geometry array or indexCapacity.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPolygonTriangulation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUPolygonTriangulationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUPolygonTriangulationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePolygonTriangulationOnCPU',
        signature:
          'function computePolygonTriangulationOnCPU( input: CPUPolygonTriangulationInput, options: CPUPolygonTriangulationOptions ): CPUPolygonTriangulationResult',
        supportedInputs: [
          'input: CPUPolygonTriangulationInput',
          'options: CPUPolygonTriangulationOptions'
        ],
        differences: [
          "/**\n * Triangulates each live shell and its holes with the contributor's Earcut/monotone routing.\n * Ring classification, winding, hole order, bridge ties, ear order, failure padding and polygon\n * slice offsets follow the current kernels, including uint32 layout prefixes and signed int32\n * slot arithmetic for clamped ring ends. Degenerate shells are empty, including their holes.\n *\n * What differs from the GPU contributor:\n * - Earcut honors maximumWork (default 4M steps) per polygon, with no GPU scratch limit.\n *   This reference uses un-hashed Earcut; its work count can differ from GPU hashing.\n *   The selected monotone route and retry honor the same raw-vertex capacity and threshold.\n * - Midpoint placement and crossing arithmetic use float64. Exact orientation has an arbitrary\n *   precision fallback, so finite exponent spans that the GPU cannot certify are accepted.\n *   Diagonal choices at float32 midpoint rounding boundaries can differ.\n * - Non-finite coordinates in live rings fail the polygon; subnormal coordinates are treated\n *   as their exact values. Invalid polygons are not repaired; no tiling is promised for them.\n * - Decreasing offsets and out-of-range feature/polygon offsets throw with the argument name;\n *   the GPU does not validate them. Ring ends beyond positions are clamped for reads, while\n *   slice capacities use the original offsets, as on the GPU. Output allocation limits remain.\n *\n * @throws Error naming positions, featureOffsets, polygonOffsets, ringOffsets or indexCapacity\n * for inconsistent arrays or an invalid allocation.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUEnsembleScores': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUEnsembleScoresOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUEnsembleScoresOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeEnsembleScoresOnCPU',
        signature:
          'function computeEnsembleScoresOnCPU( input: CPUEnsembleScoresInput, options: CPUEnsembleScoresOptions ): CPUEnsembleScoresResult',
        supportedInputs: ['input: CPUEnsembleScoresInput', 'options: CPUEnsembleScoresOptions'],
        differences: [
          '/**\n * Scores a member-major ensemble in float64. The sorted gap identity computes the CRPS pair\n * term in O(m log m) per site; energy uses unordered member pairs and variograms use listed pairs.\n *\n * What differs from the GPU contributor:\n * - All arithmetic uses float64 with one float32 output rounding; relative tolerance 3e-5 is\n *   appropriate for well-conditioned scores. Near cancellation may need an absolute error bound.\n * - GPU workgroup/invocation limits (65535 members and 67108863 sites) are dropped; uint32 array\n *   indexing limits remain. Unrequested scores are not computed.\n * - Non-finite ensemble members give NaN CRPS/energy; non-finite observations follow IEEE\n *   arithmetic. Variograms propagate NaN, including out-of-range pairs, as the kernel does.\n * - Subnormals and finite intermediates beyond float32 range are retained in float64. The plain\n *   numeric variogramOrder prop is evaluated in float64, rather than rounded in a shader literal.\n *\n * @throws If ensemble/observations are too short, pairs are incomplete, or weights are too short.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSinkhornDivergence': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSinkhornDivergenceOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSinkhornDivergenceOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSinkhornDivergenceOnCPU',
        signature:
          'function computeSinkhornDivergenceOnCPU( source: Float32Array, target: Float32Array, parameters: Float32Array, options: CPUSinkhornDivergenceOptions ): CPUSinkhornDivergenceResult',
        supportedInputs: [
          'source: Float32Array',
          'target: Float32Array',
          'parameters: Float32Array',
          'options: CPUSinkhornDivergenceOptions'
        ],
        differences: [
          "/**\n * Runs the full-kernel log-domain Sinkhorn method in float64, including the symmetric self\n * updates, epsilon schedule, all-three-plan marginal stopping test, and the final half-update on a\n * budget-limited run. The explicit maximumIterations is a solver budget, with no GPU unroll cap.\n *\n * What differs from the GPU contributor:\n * - Arithmetic and normalization use float64 with one output rounding. Stopping near tolerance\n *   can change iterationCount and convergence flags; no universal relative bound exists near\n *   zero divergence or marginal error. Well-conditioned values agree within 3e-4 relative.\n * - Omitting transportRadius evaluates full support; an explicit transportRadius bounds each axis.\n *   No finite masked-log cutoff discards a term. The explicit solver\n *   budget is retained, including iterationLimitReached; its GPU maximum of 1000 is dropped.\n * - The GPU's grid-side, row-count and storage-binding limits are dropped; uint32 grid dimensions\n *   and diagnostic counts remain. A 1x1 grid is accepted by both routes.\n * - Subnormals and intermediates outside float32 range are retained. Invalid masses are counted\n *   and zeroed as on the GPU; zero total mass or invalid packed parameters gives NaN values.\n *\n * @throws If source/target do not match the grid or parameters is shorter than its packed layout.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpaceTimeHawkes': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpaceTimeHawkesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpaceTimeHawkesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpaceTimeHawkesOnCPU',
        signature:
          'function computeSpaceTimeHawkesOnCPU( input: CPUSpaceTimeHawkesInput, options: CPUSpaceTimeHawkesOptions = {} ): CPUSpaceTimeHawkesResult',
        supportedInputs: [
          'input: CPUSpaceTimeHawkesInput',
          'options: CPUSpaceTimeHawkesOptions = {}'
        ],
        differences: [
          'The finite-window compensator derivative enters the safeguarded decay root; branching ratio is refit for each candidate. A likelihood decrease beyond rounding slack restores the entering fit. logLikelihood describes the entering iterate; fittedLogLikelihood describes the published fit.',
          "/**\n * Fits the documented exponential-Gaussian Hawkes update until its relative-change tolerance or\n * iteration budget is reached, using every earlier parent inside the inclusive lag/distance window.\n * Equal-time events do not trigger each other. The convergence comparison uses spatial variance,\n * as the kernel does.\n * logLikelihood is evaluated at the iterate entering the final update; probabilities use the\n * published estimates, as does fittedLogLikelihood. The finite-window temporal mass is evaluated\n * with expm1 in float64.\n *\n * What differs from the GPU contributor:\n * - No parent capacity; candidateOverflow is always zero. The iteration budget defaults\n *   to 100 in both forms. An explicit maximumIterations is honored by both\n *   operations; CPU accepts budgets above the GPU's 1000-iteration unrolling cap, up to uint32.\n *   At the budget, the latest fit is published with converged zero and iterationLimitReached one\n *   unless that update converged. This also terminates floating-point rounding cycles.\n * - Float64 log-domain evaluation and sums replace float32 tiles. Compare fitted values and\n *   probabilities at relative 5e-4; convergence counts can differ near the tolerance threshold.\n * - The exact -expm1(-beta * remainingTime) replaces the GPU's second-order small-argument mass.\n * - The finite-window decay root takes up to 64 bracketed Newton refinements, versus the GPU's\n *   eight; both start at the entering decay and use the same likelihood-safeguard slack.\n * - The GPU's log-rate floors, branching-ratio floor and 1e-6 triggered-mass update guard are\n *   retained. Spatial/temporal cutoff approximation remains, so approximation is still one.\n * - The GPU-only 2^24 event cap is dropped; the uint32 iteration/status layout limit remains.\n * - Nonfinite coordinates/times are excluded. Nonfinite iterates publish NaNs and stop with\n *   converged zero; subnormal float32 outputs are retained. These are outside GPU comparisons.\n *\n * @throws If positions/times lengths disagree, parameters has fewer than nine values, or\n * maximumIterations is not an integer in [1, 2^32 - 1].\n */",
          'logLikelihood: finite-window log likelihood at the iterate entering the final update.',
          'fittedLogLikelihood: finite-window log likelihood at the published estimates.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUShapeDescriptors': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUShapeDescriptorsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUShapeDescriptorsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeShapeDescriptorsOnCPU',
        signature:
          'function computeShapeDescriptorsOnCPU( input: CPUPolygonRows, parameters: Float32Array ): CPUShapeDescriptorsResult',
        supportedInputs: ['input: CPUPolygonRows', 'parameters: Float32Array'],
        differences: [
          "/**\n * Integrates shell/hole moments and computes every shape descriptor in binary64.\n * `parameters` is the getGPUShapeDescriptorsParameterValues packed layout.\n * Measurements treat zero signed area as nonnegative when weighting centroid numerators;\n * central moments retain a zero sign for those rings. Hull membership follows the GPU's\n * contiguous feature interval, even when descending offsets leave nonempty individual rings.\n *\n * What differs from the GPU contributor:\n * - All measures, centroids, central moments and ratios stay binary64 until each output is\n *   rounded once. Close-to-threshold sliver decisions can therefore differ from float32.\n * - Convexity always uses a monotone-chain hull with exact float32-coordinate turn predicates;\n *   convexityMethod and maximumVerticesPerFeature only steer GPU execution and are unused.\n * - No GPU sorting, storage-binding or per-invocation hull limits apply.\n * - Empty position storage is accepted for empty features; the GPU requires at least one row.\n *\n * @throws If positions, offsets or parameters arrays are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMantelTest': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUMantelTestOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUMantelTestOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeMantelTestOnCPU',
        signature:
          'function computeMantelTestOnCPU( input: CPUMantelTestInput, options: CPUMantelTestOptions ): CPUMantelTestResult',
        supportedInputs: ['input: CPUMantelTestInput', 'options: CPUMantelTestOptions'],
        differences: [
          '/**\n * Mantel correlation of unique listed unordered pairs, with the registered Philox swap-or-not\n * stream and optional Besag-Clifford stopping in permutation order.\n *\n * What differs from the GPU contributor:\n * - Correlations are computed in float64, then rounded to published float32 precision before\n *   threshold comparisons and summary moments, so hidden precision does not split published ties.\n *   Correlations usually agree within 1e-4 absolute. Near ties can change counts and stopping.\n * - GPU-only row and permutation caps are dropped; uint32 indexing/output lengths remain.\n * - Scaled centered vectors avoid intermediate overflow for finite transformed data.\n * - Nonfinite times, selected distances or transformed pair values throw an Error naming the\n *   argument. These inputs have no defined GPU result. Invalid neighbor IDs, self entries and\n *   duplicates are ignored as in the GPU pair mask; malformed CSR offsets throw.\n * - Transform props use their supplied binary64 numbers, rather than WGSL float32 literals.\n *\n * @throws For inconsistent arrays, invalid offsets or undefined numeric input as described above.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOPTICS': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUOPTICSOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUOPTICSOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeOPTICSOnCPU',
        signature:
          'function computeOPTICSOnCPU( positions: Float32Array, options: CPUOPTICSOptions, times?: Float32Array ): CPUOPTICSResult',
        supportedInputs: [
          'positions: Float32Array',
          'options: CPUOPTICSOptions',
          'times?: Float32Array'
        ],
        differences: [
          '/**\n * Computes self-inclusive OPTICS core distances, deterministic (reachability, row) ordering and\n * DBSCAN or xi extraction. All numeric work, including xi comparisons, stays float64 until output.\n * Invalid positions/times are excluded; labels and unused ordering/predecessor/interval slots\n * are 0xffffffff. Distances of roots, invalid and non-core rows are infinity as appropriate.\n *\n * What differs from the GPU contributor:\n * - Uses float64 Euclidean/haversine distances and xi comparisons, rounding only output distances.\n *   Ordinary planar distances allow 2e-5 relative error, spherical distances 2e-4 relative.\n *   Near distance ties, radius boundaries or xi thresholds, float32 arithmetic can change discrete\n *   ordering, predecessors, intervals and labels; the strict tie and inclusive bound rules match.\n * - Honors maximumIterations as a visited-row budget. An incomplete order withholds all labels\n *   and xi intervals. frontierBatchSize, neighborIndex, orderingMethod and neighborCapacity do\n *   not affect the exhaustive CPU route; candidateOverflow is always 0.\n * - Drops the GPU row limits (4096/65536), minimumSamples limit (256), batch and command budgets.\n *   The implementation is exhaustive and intended for small collections, with O(n² log n) work.\n * - Xi still has rowCount interval capacity, required by the output layout. Summary is the\n *   clamped interval count and overflow flag; overflow clears all labels, as on the GPU.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConditionalGaussianSimulation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUConditionalGaussianSimulationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUConditionalGaussianSimulationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeConditionalGaussianSimulationOnCPU',
        signature:
          'function computeConditionalGaussianSimulationOnCPU( input: CPUConditionalGaussianSimulationInput, options: CPUConditionalGaussianSimulationOptions ): CPUConditionalGaussianSimulationResult',
        supportedInputs: [
          'input: CPUConditionalGaussianSimulationInput',
          'options: CPUConditionalGaussianSimulationOptions'
        ],
        differences: [
          "/**\n * Conditions a joint Gaussian distribution and draws its conditional query vector using the\n * contributor's Philox purpose and (replicateOffset + replicate, latent query index, 0, 0) counters.\n * All calculations use float64, with a single rounding into each returned float32 array.\n *\n * What differs from the GPU contributor:\n * - The 4096 joint-row and GPU storage-binding limits are dropped; this still allocates a dense\n *   quadratic factor. Uint32 draw identities and output lengths remain limited by their layouts.\n * - Positive observation pivots are accepted below the GPU's 1e-6 covariance-scale cutoff.\n *   Query roundoff is tested at 64 * Number.EPSILON * joint size * covariance scale instead.\n * - Exact coordinate aliases have exactly zero new variance; noiseless observed queries copy\n *   the observation. The cooperative GPU path can retain positive cancellation residue.\n * - Full closed-form Matern correlations replace truncated series; the matern smoothness alias\n *   is accepted. Plain numeric covariance and mean props keep their float64 values.\n * - Philox words and their order match. Box-Muller and its half-bin angle are evaluated in\n *   float64; GPU transcendental and angle rounding differ. Compare ordinary moments at relative\n *   1e-4 and realizations at 2e-4 of max(conditional standard deviation, absolute mean).\n * - The pure function and CPU operation recompute preparation on each call.\n * - Invalid included observations or queries give status 1; singular observations give status 2.\n *   Subnormal outputs are retained; finite float64 results outside float32 range round to infinity\n *   without setting status 2. These extremes are outside GPU comparisons.\n *\n * @throws If positions, observations or queries have inconsistent lengths or options are invalid.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaFisherScoring': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUVecchiaFisherScoringOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUVecchiaFisherScoringOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeVecchiaFisherScoringOnCPU',
        signature:
          'function computeVecchiaFisherScoringOnCPU( input: CPUVecchiaFisherScoringInput ): CPUVecchiaFisherScoringResult',
        supportedInputs: ['input: CPUVecchiaFisherScoringInput'],
        differences: [
          "/**\n * Evaluates the Gaussian conditional likelihood, log-parameter score and expected Fisher\n * information. Parameter order is log sill, log effective range, log nugget; information is\n * I00, I01, I02, I11, I12, I22. Measurement-error variances are fixed, not differentiated.\n *\n * What differs from the GPU contributor:\n * - All usable CSR slots are used, without neighborCount truncation or the 32-neighbor GPU limit.\n * - Float64 arithmetic replaces intermediate float32; equilibrated reliability thresholds are\n *   retained, so near-threshold statuses can differ. Compare ordinary outputs at relative 1e-4.\n * - Matern correlations use the full closed form instead of the GPU's short-lag Taylor series.\n * - Totals sum unrounded float64 site cells and split into high/low float32 words. The GPU sums\n *   rounded site cells with a fixed-point accumulator. Compare decoded totals at relative 1e-4.\n * - Malformed CSR spans throw an Error naming neighborhoodOffsets; the GPU has no explicit check.\n * - Nonfinite site coordinates fail with status 2. Float64-finite results beyond float32 range\n *   publish infinities on rounding, and subnormal outputs are retained; these are not compared.\n *\n * @throws If positions, observations, neighborhoodOffsets or parameters have inconsistent lengths.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaLogLikelihood': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUVecchiaLogLikelihoodOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUVecchiaLogLikelihoodOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeVecchiaLogLikelihoodOnCPU',
        signature:
          'function computeVecchiaLogLikelihoodOnCPU( input: CPUVecchiaLogLikelihoodInput ): CPUVecchiaLogLikelihoodResult',
        supportedInputs: ['input: CPUVecchiaLogLikelihoodInput'],
        differences: [
          "/**\n * Evaluates zero-mean Gaussian conditional predictions and likelihood in float64, rounding each\n * published array once. Conditioning errors affect the solve diagonal; the site's error is added\n * only to its likelihood variance. Excluded sites still receive conditional prediction rows.\n *\n * What differs from the GPU contributor:\n * - All usable conditioning slots are solved, without the GPU's neighborCount truncation or\n *   32-neighbor private-array limit; stable distance order is preserved.\n * - Float64 solves keep the GPU's equilibrated pivot diagnostic and reliability thresholds.\n *   Near a threshold, rounding can change a status. Compare ordinary outputs at relative 1e-4.\n *   For conditional means, cancellation can amplify relative error: comparisons also allow an\n *   absolute error of 2 * 2^-24 * sum(abs(weight * observation)), using independent oracle weights.\n * - Matern correlations use the full closed form instead of a short-lag truncated series.\n * - Totals sum unrounded float64 site terms, then split into float32 high/low words; the GPU sums\n *   float32 site terms with a fixed-point accumulator. Compare decoded totals at relative 1e-4.\n * - Nonfinite coordinates are rejected by the conditional solve; an empty CSR site's likelihood\n *   still follows the GPU's marginal branch. Float64-finite results outside float32 range publish\n *   infinities on rounding; subnormal outputs are retained. These are outside GPU comparisons.\n * - Malformed CSR spans retain the spatial kernel's conditionalStatus 1 result.\n *\n * @throws If positions, observations, parameters or neighborhoodOffsets have inconsistent lengths.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaSpaceTimeLogLikelihood': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUVecchiaSpaceTimeLogLikelihoodOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUVecchiaSpaceTimeLogLikelihoodOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeVecchiaSpaceTimeLogLikelihoodOnCPU',
        signature:
          'function computeVecchiaSpaceTimeLogLikelihoodOnCPU( input: CPUVecchiaSpaceTimeLogLikelihoodInput ): CPUVecchiaSpaceTimeLogLikelihoodResult',
        supportedInputs: ['input: CPUVecchiaSpaceTimeLogLikelihoodInput'],
        differences: [
          "/**\n * Evaluates zero-mean Gaussian conditional predictions and likelihood in float64, rounding each\n * published array once. Conditioning errors affect the solve diagonal; the site's error is added\n * only to its likelihood variance. Excluded sites still receive conditional prediction rows.\n * Finite negative structure codes in hand-written buffers select spherical covariance, following\n * the GPU's clamped u32 conversion; the normal parameter packer does not produce these codes.\n *\n * What differs from the GPU contributor:\n * - All usable conditioning slots are solved, without the GPU's neighborCount truncation or\n *   32-neighbor private-array limit; CSR visit order is preserved.\n * - Float64 solves keep the GPU's equilibrated pivot diagnostic and reliability thresholds.\n *   Near a threshold, rounding can change a status. Compare ordinary outputs at relative 1e-4.\n * - Matern correlations use the full closed form instead of a short-lag truncated series.\n * - Totals sum unrounded float64 site terms, then split into float32 high/low words; the GPU sums\n *   float32 site terms with a fixed-point accumulator. Compare decoded totals at relative 1e-4.\n * - Nonfinite coordinates are rejected by the conditional solve; an empty CSR site's likelihood\n *   still follows the GPU's marginal branch. Float64-finite results outside float32 range publish\n *   infinities on rounding; subnormal outputs are retained. These are outside GPU comparisons.\n * - Malformed CSR spans throw an Error naming neighborhoodOffsets (the GPU has no explicit check).\n * - A NaN structure code selects the Matern 5/2 default branch on CPU; WGSL's NaN-to-u32 result\n *   is indeterminate, so such parameter buffers are outside GPU comparisons.\n *\n * @throws If positions, observations, times, parameters or neighborhoodOffsets have inconsistent lengths.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUWardRegions': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUWardRegionsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUWardRegionsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeWardRegionsOnCPU',
        signature:
          'function computeWardRegionsOnCPU( input: { offsets: Uint32Array; neighbors: Uint32Array; values: Float32Array; parameters: Uint32Array; status?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array}; }, options: CPUWardRegionsOptions ): CPUWardRegionsResult',
        supportedInputs: [
          'input: {\n    offsets: Uint32Array;\n    neighbors: Uint32Array;\n    values: Float32Array;\n    parameters: Uint32Array;\n    status?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array};\n  }',
          'options: CPUWardRegionsOptions'
        ],
        differences: [
          '/**\n * Completes spatial Ward agglomeration, using the undirected union of the CSR pattern, the\n * seed-keyed total pair order, and either sequential or mutual-nearest merging. Non-finite\n * feature entries and out-of-range neighbor slots each add one to invalidCount and prevent merges.\n * Status contains the four GPU_WARD_STATUS_LAYOUT words, optionally followed by the mutual-nearest round\n * count. A disconnected final round counts as an attempted round. The fifth word is unused for\n * sequential schedules and is returned as zero; callers should use four words in that mode.\n * Optional input.status loss scalars suppress the partition as on GPU: labels are 0xffffffff,\n * objective is NaN and iteration/convergence fields are zero. Loss is not an invalid feature.\n *\n * What differs from the GPU contributor:\n * - Uses float64 column standardization, region sums, costs and objective, with one output rounding.\n *   Objective comparisons allow 1e-4 relative error; near-tied costs can change the partition.\n * - Honors maximumIterations; omitted in the pure function it defaults to rows minus the target.\n *   iterationCount is the completed merge count; the budget counts rounds for either schedule.\n * - Drops the eight-column private-array limit and GPU storage/command scheduling limits.\n * - Finite inputs whose GPU merge costs overflow remain mergeable; the float32 objective can\n *   overflow when the completed float64 objective is rounded at the output.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSkaterRegions': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSkaterRegionsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSkaterRegionsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSkaterRegionsOnCPU',
        signature:
          'function computeSkaterRegionsOnCPU( input: CPUSkaterRegionsInput, options: CPUSkaterRegionsOptions ): CPUSkaterRegionsResult',
        supportedInputs: ['input: CPUSkaterRegionsInput', 'options: CPUSkaterRegionsOptions'],
        differences: [
          "/**\n * Greedy minimum-spanning-forest cuts maximizing the between-side sum of squares. Each candidate\n * traverses its current region afresh; labels remain the vertex closest to the original root.\n * Zero-gain cuts are allowed, minimum sizes are inclusive, and exact ties use the lowest edge\n * slot (the lesser of its two CSR slots). The parameter layout is the two uint32 words packed\n * by getGPUSkaterRegionsParameterValues: regionCount, minimumRegionSize.\n * Either nonzero input.status loss word suppresses labels/cut IDs to 0xffffffff, regionCount\n * to zero/unavailable and cut gains to NaN, including when the upstream forest is suppressed.\n *\n * What differs from the GPU contributor:\n * - Direct float64 side sums replace compensated float32 tour sums and repeated subtraction.\n *   Gains round once at output; near ties can change the cut sequence, since GPU gain keys are\n *   float32. A finite float64 gain above float32 range is eligible and publishes Infinity.\n * - The 16-column and GPU storage-binding limits are dropped. maximumRegionCount remains for\n *   the output log shape and target clamp, rather than an additional CPU iteration budget.\n *   Empty input slot arrays are accepted.\n * - Malformed CSR arrays, non-forest tree flags and inconsistent componentLabels throw errors\n *   naming the input. The GPU assumes a symmetric CSR and valid upstream forest outputs.\n * - Nonfinite values without treeStatus suppress cuts in their whole initial tree, as the GPU's\n *   nonfinite region sums do. With nonzero treeStatus.invalidCount no tree is cut.\n *\n * @throws Error naming inconsistent arrays or a malformed forest.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUAZPRegions': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUAZPRegionsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUAZPRegionsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeAZPRegionsOnCPU',
        signature:
          'function computeAZPRegionsOnCPU( input: CPUAZPRegionsInput, options: CPUAZPRegionsOptions ): CPUAZPRegionsResult',
        supportedInputs: ['input: CPUAZPRegionsInput', 'options: CPUAZPRegionsOptions'],
        differences: [
          "/**\n * Greedy AZP with independent batched moves, two-hop source-region removal tests, seeded row\n * ties and lowest destination-label ties. Retains the GPU's explicit relative gain threshold\n * (gain > 4e-6 times the four squared-moment terms), minimum size and extensive constraints.\n *\n * What differs from the GPU contributor:\n * - Float64 standardization, centered sums and stable removal/insertion gains replace float32\n *   moments; objective rounds only at output. Relative objective tolerance is 1e-4; near ties\n *   and gain-threshold boundaries can change moves, labels and iterationCount.\n * - Preserves two-hop admissibility, dropping only the 128-row and 64-degree neighborhood caps.\n * - Extensive sums and departing-value subtraction also use float64. An extensive-threshold\n *   decision can differ when a small remainder is lost from the GPU's float32 region sum.\n *   For [1, 16777216] at threshold 1, the twin permits the second row to leave; the GPU may not.\n * - Honors maximumIterations (default 256 for the pure function). A final successful batch at\n *   the limit is not proof of convergence; a no-move round must execute to set converged.\n * - Drops GPU column, storage and command-node caps. Empty slot arrays are accepted. Invalid\n *   labels, nonfinite features, negative/nonfinite extensive values or threshold and bad neighbor\n *   IDs keep labels, publish zero objective and set invalidCount, including extra initial components.\n * - Finite float64 gains remain eligible above float32 range; objective can round to Infinity.\n *   Full initial connectivity has no GPU hook-round cutoff, so it never reports an unfinished\n *   component search as invalid input.\n *\n * Nonzero input.status overflow or candidateOverflow suppresses both forms: missing labels\n * (0xffffffff), NaN objective and zero status words.\n *\n * @throws Error naming inconsistent arrays, including malformed CSR offsets.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMaxPRegions': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUMaxPRegionsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUMaxPRegionsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeMaxPRegionsOnCPU',
        signature:
          'function computeMaxPRegionsOnCPU( input: CPUMaxPRegionsInput, options: CPUMaxPRegionsOptions = {} ): CPUMaxPRegionsResult',
        supportedInputs: ['input: CPUMaxPRegionsInput', 'options: CPUMaxPRegionsOptions = {}'],
        differences: [
          "/**\n * Runs budgeted seeded parallel Max-P growth and enclave assignment, selects by infeasible\n * regions, region count, grown-region SSD and start index, and optionally refines with two-hop AZP.\n * This is the documented multistart heuristic, not a global maximum-P optimizer. Seed rows and\n * ties use the contributor's registered Philox stream and bijective order. Enclave means stay\n * frozen at the end of growth; the published objective is recomputed from the final labels.\n *\n * What differs from the GPU contributor:\n * - Float64 standardization, means, distances, growth scores and objective round only at output.\n *   Compare objectives with relative tolerance 1e-4; near ties or extensive-sum threshold\n *   boundaries can change labels and selectedStart.\n * - Growth and enclaves honor maximumGrowthSteps independently, as on GPU. The expected-size cap,\n *   63-neighbor priority cap and truncated float32 distance keys are removed; priorities then\n *   exact distances then the same seeded ties order offers. Wave arithmetic also uses float64.\n * - Drops GPU column, start-count, storage and command-node limits; uint32 layouts remain.\n *   Empty neighbor-slot arrays are accepted.\n * - Positive maximumIterations bounds AZP refinement rounds with two-hop admissibility.\n *   improvementIterationCount is the number of successful batched rounds, as on GPU.\n *   Its neighborhood storage/degree caps are dropped. Refinement extensive sums and departing-\n *   value subtraction use float64, so near-threshold feasibility decisions can differ as in AZP.\n * - Finite distances above float32 range remain eligible. Final objective may round to Infinity.\n *   Nonfinite features, negative/nonfinite extensive values or threshold, and bad neighbor IDs\n *   follow the GPU invalid status path (singleton labels, zero objective and regionCount).\n *\n * Nonzero input.status overflow or candidateOverflow suppresses both forms: missing labels\n * (0xffffffff), NaN objective and zero status words.\n * Unfinished enclave remainders retain each connected component together. All pre-enclave\n * growth scores contribute once, even for fully absorbed regions. Final feasibility is\n * recomputed after refinement from the returned memberships.\n *\n * @throws Error naming inconsistent arrays, including malformed CSR offsets.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPanelErrorMaximumLikelihood': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialPanelErrorMaximumLikelihoodOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialPanelErrorMaximumLikelihoodOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialPanelErrorMaximumLikelihoodOnCPU',
        signature:
          'function computeSpatialPanelErrorMaximumLikelihoodOnCPU( input: CPUSpatialPanelErrorMaximumLikelihoodInput, options: CPUSpatialPanelErrorMaximumLikelihoodOptions ): CPUSpatialPanelErrorMaximumLikelihoodResult',
        supportedInputs: [
          'input: CPUSpatialPanelErrorMaximumLikelihoodInput',
          'options: CPUSpatialPanelErrorMaximumLikelihoodOptions'
        ],
        differences: [
          "/**\n * Profiled balanced fixed-effects error ML, divisor N*T, with conditional expected-information\n * standard errors and pooled OLS LM diagnostics. All arithmetic is float64 until publication.\n *\n * What differs from the GPU contributor:\n * - Float64 within transforms and QR replace float32 tiled transforms and Cholesky; rank uses\n *   the shared QR floor. GPU-only row/predictor/power/period-contrast limits are dropped.\n * - The same 1024-interval stationary-crossing search uses direct filtered residuals and\n *   48 bisections instead of float32 moment polynomials and 32 bisections. It can miss extrema\n *   narrower than one interval, as can the GPU search. No unbounded optimization is claimed.\n * - Initial rank deficiency returns status 1; no filtered rescue is attempted. No mask or\n *   missing panels: any nonfinite observation returns status 3. Overflow suppresses the fit\n *   and diagnostics (status 3 and 1 respectively). Per-frame values remain ordinary arrays.\n * - Shared float64 chi-square survival and conditioning helpers retain their own conventions.\n * - An exact original-data certificate rejects a zero-variance pooled OLS fit with diagnostic\n *   status 2; roundoff in a QR residual cannot supply the positive variance required by LM tests.\n *   The Gram certificate also covers multivariate/effect-space exact fits; the GPU's original-\n *   input witness covers y=a+b*x_j on normal/zero coordinates certified by integer collinearity.\n * - Trace standard errors remain available when exact-fit rejection returns status 3: they\n *   describe W and its probes independently of whether y has a positive-variance fit.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOrdinaryLeastSquares': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUOrdinaryLeastSquaresOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUOrdinaryLeastSquaresOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeOrdinaryLeastSquaresOnCPU',
        signature:
          'function computeOrdinaryLeastSquaresOnCPU( input: CPUOrdinaryLeastSquaresInput, predictorCount: number ): CPUOrdinaryLeastSquaresResult',
        supportedInputs: ['input: CPUOrdinaryLeastSquaresInput', 'predictorCount: number'],
        differences: [
          "/**\n * Centered OLS (or slope-only ridge) and all 22 report slots in float64, rounded once on return.\n * Non-finite predictor/response rows and mask=0 rows are excluded, as in the contributor.\n * Predictions still cover those rows when their predictors are finite; fitted values remain NaN.\n *\n * What differs from the GPU contributor:\n * - Float64 moments, solves, residuals and diagnostics replace float32 tile reductions and mean\n *   corrections. tileRowCount is ignored. Comparisons on well-conditioned inputs use 5e-3 relative\n *   tolerance for diagnostics and 5e-4 for coefficients and row outputs.\n * - The shader-only 15-predictor cap is dropped. Numerical singularity uses a squared Cholesky\n *   pivot floor of 64 * Number.EPSILON relative to that column's correlation diagonal, instead of\n *   the GPU's pivot < 1e-3 rejection; some GPU status-1 fits succeed here.\n * - Conditioning uses one-sided Jacobi singular values until float64 orthogonality or numerical\n *   stagnation, with a separate bounded host sweep count. A singular unpenalized design in a\n *   successful ridge fit reports Belsley 3e38,\n *   matching the GPU sentinel. The precision-warning threshold still describes the GPU matrix.\n * - The public float64 chi-square function uses a converged shape-dependent host evaluation;\n *   the GPU retains its 200-iteration budget for at most 15 degrees of freedom.\n * - Subnormals are retained. Non-finite parameter lambda means no ridge, as on the GPU.\n *\n * @throws If predictors, mask or parameters have inconsistent lengths, naming the argument.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURestrictedRegimeRegression': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPURestrictedRegimeRegressionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPURestrictedRegimeRegressionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeRestrictedRegimeRegressionOnCPU',
        signature:
          'function computeRestrictedRegimeRegressionOnCPU( input: CPURestrictedRegimeRegressionInput, options: CPURestrictedRegimeRegressionOptions ): CPURestrictedRegimeRegressionResult',
        supportedInputs: [
          'input: CPURestrictedRegimeRegressionInput',
          'options: CPURestrictedRegimeRegressionOptions'
        ],
        differences: [
          "/**\n * Fits both nested models by norm-scaled Householder QR in float64. The coefficient order is\n * source-column-major, shared once or repeated by regime. Returns pooled restricted covariance,\n * Chow [F, numerator df, denominator df], and per-column plus joint Wald [chi-square, df, p].\n *\n * What differs from the GPU contributor:\n * - Float64 QR replaces shifted float32 normal equations; response origins are per regime when\n *   intercepts vary, and pooled otherwise. The unrestricted model always uses per-regime origins.\n *   Compare well-conditioned outputs at 5e-3 relative tolerance (covariances at 5e-4\n *   of their diagonal geometric mean). Fitted rows are evaluated before output rounding.\n * - Shader-only predictor, regime, 16-coefficient and 2^24-row limits are dropped. Numerical rank\n *   uses a norm-scaled QR pivot floor of 64 * Number.EPSILON, rather than GPU Cholesky 1e-3;\n *   some GPU status-2/3 fits succeed here. Empty regimes still give status 3 when the restricted\n *   model fits, with its coefficients/covariance/fitted retained and classical Chow/stability\n *   unavailable. Optional robust inference needs only the restricted fit and n > restricted k.\n * - Wald covariance is factored through a QR square root, avoiding raw covariance cancellation.\n *   The public float64 chi-square survival function supplies p-values (shared 200-iteration cap).\n *   A statistic exceeding float32 range rounds to Infinity; its float64 p-value may round to zero.\n * - Subnormals are retained; non-finite observations are excluded exactly as by the contributor.\n * - Optional robust accepts the same white/HC1 or HAC option with CPU kernel arrays. Scores and\n *   contrasts use float64 QR roots rather than the GPU's shifted inverse matrix; status meanings,\n *   original-row masking, unit kernel diagonal and suppression of invalid inference are identical.\n *   Signed contrast variances must exceed 64 * Number.EPSILON * n times the absolute score-product\n *   accumulation scale before equilibration; the GPU uses its float32 tile/CSR error budget.\n *\n * @throws If predictors, regimes or mask lengths disagree with response, naming the argument.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSeeminglyUnrelatedRegression': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSeeminglyUnrelatedRegressionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSeeminglyUnrelatedRegressionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSeeminglyUnrelatedRegressionOnCPU',
        signature:
          'function computeSeeminglyUnrelatedRegressionOnCPU( input: CPUSeeminglyUnrelatedRegressionInput, options: CPUSeeminglyUnrelatedRegressionOptions ): CPUSeeminglyUnrelatedRegressionResult',
        supportedInputs: [
          'input: CPUSeeminglyUnrelatedRegressionInput',
          'options: CPUSeeminglyUnrelatedRegressionOptions'
        ],
        differences: [
          "/**\n * Feasible GLS: equation OLS, joint residual covariance (divisor complete-case n),\n * Cholesky whitening and a Householder QR system fit. All work is float64 until output.\n * Optional iteration repeats covariance and GLS with exactly the caller's budget and tolerance.\n * The first GLS counts as round 1; status 4 retains the last fit on budget exhaustion. Both\n * covariance outputs are those used by the last round, not a post-fit covariance refresh.\n *\n * What differs from the GPU contributor:\n * - GPU tolerance must round to a normal positive float32, avoiding zero after rounding or\n *   subnormal flushing. CPU convenience tolerance accepts every positive finite JavaScript\n *   number, including smaller tolerances and values above the float32 range, in float64.\n *   Optional per-frame parameters retain the shared GPU float32 validity/layout convention.\n * - Spatial mode retains the contributor's OLS -> Kelejian-Prucha GM -> filtered joint GLS ->\n *   restored intercept/structural-residual sequence, entirely in float64. Its profiled quartic\n *   moment objective is minimized over both variance branches using all stationary points and\n *   branch boundaries, without the GPU's 1024-interval scan or 40-step refinement cap. Ties select\n *   the lowest lambda. Invalid spatial weights retain equation status 4 and NaN parameters.\n * - Float64 centered QR replaces float32 tile reductions and normal-equation solves. Residuals\n *   use centered coefficients before rounding. Compare well-conditioned outputs at 5e-3 relative\n *   tolerance; covariance error is scaled by the diagonal geometric mean.\n * - GPU equation, predictor, 16-system-coefficient and float32 row-count envelopes are dropped.\n *   Design rank uses norm-scaled QR pivots > 64 * Number.EPSILON; normalized error covariance\n *   squared Cholesky pivots use 64 * Number.EPSILON times the corresponding diagonal. GPU 1e-3\n *   Cholesky-pivot failures may therefore succeed on the CPU.\n * - Subnormals are retained. In nonspatial mode a non-finite value in any equation, including\n *   supplied initial residuals, excludes the aligned row. Spatial non-finite equation data returns\n *   equation status 3, except where insufficient-row status 2 or invalid-weight status 4 takes\n *   precedence. Error covariance is retained on one-step joint status 3; an iterated singular\n *   covariance clears numeric outputs to NaN, preserving the completed-round count.\n * - Spatial CSR shape mismatches throw by argument name. Invalid row offsets, links, ordering,\n *   or row sums covered by the GPU's weight-status check return equation status 4 instead.\n *   Status-bearing spatial weights are refused because these forms cannot consume loss ports.\n *\n * @throws If predictors, responses, mask, initialResiduals or spatial CSR arrays are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialLagImpacts': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialLagImpactsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialLagImpactsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialLagImpactsOnCPU',
        signature:
          'function computeSpatialLagImpactsOnCPU( input: CPUSpatialLagImpactsInput, options: CPUSpatialLagImpactsOptions ): {impacts: Float32Array}',
        supportedInputs: [
          'input: CPUSpatialLagImpactsInput',
          'options: CPUSpatialLagImpactsOptions'
        ],
        differences: [
          '/**\n * Evaluates the documented truncated impacts with the public getSpatialLagImpactsFromTraces\n * helper, in float64, and writes [direct, indirect, total] per predictor in float32.\n *\n * What differs from the GPU contributor:\n * - Float64 intermediate arithmetic, rounded once at output; compare with relative tolerance 3e-6.\n * - The shader-only maximumPower <= 256 limit is dropped; the requested truncation is retained.\n * - rhoBound must be finite. Its float32 representation is used for the same strict bound as\n *   the shader constant, but rho * infinityNorm is evaluated in float64 at that bound.\n * - Non-finite coefficients and traces propagate through the public helper; subnormals are kept.\n *\n * @throws If an input array cannot hold the rows the GPU reads, naming that argument.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialRegressionDiagnostics': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialRegressionDiagnosticsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialRegressionDiagnosticsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialRegressionDiagnosticsOnCPU',
        signature:
          'function computeSpatialRegressionDiagnosticsOnCPU( input: CPUSpatialRegressionDiagnosticsInput, options: CPUSpatialRegressionDiagnosticsOptions ): CPUSpatialRegressionDiagnosticsResult',
        supportedInputs: [
          'input: CPUSpatialRegressionDiagnosticsInput',
          'options: CPUSpatialRegressionDiagnosticsOptions'
        ],
        differences: [
          "/**\n * Spatial LM tests and residual Moran inference in float64, using the supplied residuals. Sparse\n * products of an orthonormal design give the low-rank trace corrections without an n-by-n matrix.\n *\n * What differs from the GPU contributor:\n * - Float64 QR and sparse products replace float32 centered Gram solves and reductions.\n *   Both forms compute information from squared projection residuals of W(y-e); the CPU uses\n *   an orthonormal QR basis and the GPU uses its scaled Gram solve. Well-conditioned comparisons use\n *   relative tolerance 5e-3, with row counts and test degrees compared exactly.\n * - Shader-only 15-predictor and device binding-size limits are dropped; tileRowCount is ignored.\n *   Design rank uses norm-scaled QR pivots > 64 * Number.EPSILON, so some GPU status-1 designs\n *   succeed. Failure precedence and all status values/NaN layouts are retained.\n * - P-values call the public float64 chi-square survival helper instead of the shader erfc\n *   approximation. Its shared 200-iteration cap remains. Subnormals are retained.\n * - Sorted unique CSR neighbors are checked on the CPU. Invalid neighbor IDs are dropped,\n *   unused capacity is ignored, and non-finite active weights produce status 4. The pure function\n *   accepts empty slot arrays, whereas the GPU needs positive slot capacity.\n * - As in the kernel, Moran's trace correction assumes zero diagonal weights; self-links should\n *   not be supplied (the GPUSpatialWeights self-join invariant).\n *\n * @throws If weights, predictors, response, residuals or mask arrays are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialWeightsTraces': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialWeightsTracesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialWeightsTracesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialWeightsTracesOnCPU',
        signature:
          'function computeSpatialWeightsTracesOnCPU( input: CPUSpatialWeightsTracesInput, options: CPUSpatialWeightsTracesOptions ): CPUSpatialWeightsTracesResult',
        supportedInputs: [
          'input: CPUSpatialWeightsTracesInput',
          'options: CPUSpatialWeightsTracesOptions'
        ],
        differences: [
          '/**\n * Sparse power chains in float64. Exact mode sums unit-probe diagonals; stochastic mode replays\n * the registered Philox Rademacher probes in the GPU order. Sums always use the ones chain.\n * Optional Gram moments are in original weight units, including per-slot square sums.\n * For requested cubic and quartic Gram probes, the GPU checks a conservative exponent bound.\n * Inside the bound it uses scalar normalized intermediates and restores original weight units.\n * Outside the bound it uses raw weights and carries separate exponents through the reduction.\n *\n * What differs from the GPU contributor:\n * - Float64 chains/reductions replace float32 arithmetic, with one rounding per output. Compare\n *   nonzero outputs with relative tolerance 2e-4; integer approximation and exact-mode errors agree\n *   exactly. Stochastic standard errors use float64 online centered moments.\n * - The shader power cap and explicit-exact row cap are dropped. The same mode default switches\n *   to stochastic above 4096 rows. tileRowCount and probeBatchSize have no computational effect.\n * - Whole and chunked CSR retain separate arrays; no implicit packing occurs. As on the GPU,\n *   Gram traces require whole CSR. Unused metadata ports are omitted.\n * - A finite positive infinity norm is used as the scale. Empty weights scale by 1.\n *   A non-finite active weight makes the norm NaN and all floating\n *   results NaN; the GPU does not define portable NaN-reduction behavior. Subnormals are retained.\n * - Empty slot arrays are accepted by the pure function; GPU weights require positive capacity.\n *\n * @throws If weights.offsets, weights.weights or parameters are inconsistent, naming the argument.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMultiscaleGeographicallyWeightedRegression':
    {
      cpu: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'CPUMultiscaleGeographicallyWeightedRegressionOperation'
      },
      gpu: {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUMultiscaleGeographicallyWeightedRegressionOperation'
      },
      arrays: [
        {
          package: '@luma-contributors/analysis/gpu-spatial-analysis',
          exportName: 'computeMultiscaleGeographicallyWeightedRegressionOnCPU',
          signature:
            'function computeMultiscaleGeographicallyWeightedRegressionOnCPU( input: CPUMultiscaleGeographicallyWeightedRegressionInput, options: CPUMultiscaleGeographicallyWeightedRegressionOptions ): CPUMultiscaleGeographicallyWeightedRegressionResult',
          supportedInputs: [
            'input: CPUMultiscaleGeographicallyWeightedRegressionInput',
            'options: CPUMultiscaleGeographicallyWeightedRegressionOptions'
          ],
          differences: [
            "/**\n * Cyclic additive backfitting from global OLS, with per-component leave-one-out bandwidth\n * selection (finite-grid or continuous golden-section, LOO or AICc), known-variance conditional errors and full response-smoother inference. Each component is replaced at all\n * included foci before the next component starts; exclusions have code 1 and NaN rows.\n *\n * What differs from the GPU contributor:\n * - Float64 QR initializes global coefficients; float64 distances, weighted sums, CV scores and\n *   contributions are rounded once at output. Compare well-conditioned row outputs with 5e-3\n *   relative tolerance. Near-tied CV scores and tolerance boundaries can change selected\n *   bandwidths or completed sweep counts; exact score ties still choose the first candidate.\n * - AICc selection uses each scalar partial-response smoother trace, as mgwr multi_bw;\n *   full-model trace(S) is the sum of the coupled component operator traces (Li and Fotheringham\n *   2020), as reported by inference.summary[0]. Golden-section uses GWR's float32 score tolerance\n *   and also probes the upper endpoint to retain a feasible bandwidth when interiors are invalid.\n * - Shader-only predictor, row, candidate and backfit iteration caps are dropped. maximumIterations\n *   bounds completed sweeps (default 32), as on GPU; finite limit iterates have code 3 and publish\n *   coefficients and fitted values. Conditional and full inference require convergence.\n *   Coefficients and predictions must be representable in float32; output overflow invalidates\n *   the entire coupled fit with code 4 and NaN outputs. Component search-limit flags accumulate\n *   independently of backfit convergence and reset on each call.\n * - The CPU visits all complete-case rows in input order; indexGridSize has no computational\n *   effect. Gaussian cutoff is retained; bisquare weights are zero at or above the bandwidth.\n * - Great-circle distances call the shared float64 transcription of the GPU geodesic helper,\n *   including seam wrapping. Subnormals are retained; non-finite data are excluded as on the GPU.\n * - Full inference propagates the same selected bandwidth history from zero initialization in\n *   float64; its shader row/work caps are dropped. Variance uses float64 fitted values, rather\n *   than re-reading rounded float32 predictions. Conditional errors still use supplied variance.\n * - No getGPU parameter packer exists for this contributor; parameters retain its documented\n *   [contribution tolerance, intercept bandwidth, predictor bandwidths...] layout.\n *\n * @throws If positions, predictors, mask, parameters, bandwidthCandidates or noiseVariance lengths disagree.\n */"
          ]
        }
      ]
    },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialErrorGM': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialErrorGMOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialErrorGMOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialErrorGMOnCPU',
        signature:
          'function computeSpatialErrorGMOnCPU( input: CPUSpatialErrorGMInput, options: CPUSpatialErrorGMOptions ): CPUSpatialErrorGMResult',
        supportedInputs: ['input: CPUSpatialErrorGMInput', 'options: CPUSpatialErrorGMOptions'],
        differences: [
          "/**\n * Fits both spatial error GM estimators in float64, retaining the homoskedastic first-step betas\n * and residuals while evaluating their covariance at the second-step lambda.\n *\n * What differs from the GPU contributor:\n * - Float64 centered QR, residuals, moments and traces replace float32 reductions/normal solves;\n *   only final arrays are rounded. Well-conditioned comparisons use 5e-3 relative tolerance.\n * - Both piecewise-quartic moment objectives are minimized over every stationary point and\n *   variance-zero boundary, to float64 resolution. The GPU's 1024-interval scan and 40 bisections\n *   can miss narrow basins. Exact ties still choose the lowest lambda.\n * - Shader-only predictor and explicit-exact trace row caps are dropped; tileRowCount is ignored.\n *   QR rank uses 64 * Number.EPSILON, so some GPU status-1 fits succeed here. Conditioning uses\n *   converged float64 singular values, without a GPU sweep cap.\n * - Stochastic traces retain the GPU's default mode, registered Philox seed/probe order and\n *   approximation flag. They are not silently replaced with exact traces. Kelejian-Prucha\n *   ignores traceMode and traceProbeCount, including otherwise invalid settings.\n * - P-values call the public float64 chi-square survival helper (its 200-iteration cap remains).\n *   Subnormals are retained; non-finite active weights or data produce failure codes. The pure\n *   function accepts empty slot arrays; GPU weights require positive capacity.\n * - As in the kernels, residuals are the first filtered fit's rows even if the final report fails;\n *   the finish kernel clears table/summary only. Whole CSR is supported; chunked weights are omitted.\n *\n * @throws If weights, predictors, response or traceParameters arrays are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUProbitRegression': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUProbitRegressionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUProbitRegressionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeProbitRegressionOnCPU',
        signature:
          'function computeProbitRegressionOnCPU( input: CPUProbitRegressionInput, options: CPUProbitRegressionOptions ): CPUProbitRegressionResult',
        supportedInputs: ['input: CPUProbitRegressionInput', 'options: CPUProbitRegressionOptions'],
        differences: [
          "/**\n * Binary probit by centered/scaled Fisher scoring and monotone likelihood backtracking.\n * Only status 0 publishes expected-information errors. Status 3 publishes the last iterate;\n * statuses 1, 2 and 4 leave coefficients, predictions and effects NaN. Complete cases are shared\n * by all outputs; nonbinary complete responses return status 4 before recording a row count.\n *\n * What differs from the GPU contributor:\n * - Float64 means, information, centered predictions, densities and diagnostics are rounded once\n *   on output. Well-conditioned comparisons use 5e-4 relative tolerance (iterations are discrete).\n * - Shader row/predictor/iteration caps and the 24-halving cap are dropped. The requested iteration\n *   budget and tolerance remain; backtracking stops when the candidate no longer changes.\n * - Stable float64 log tails and inverse Mills ratios remove the wrong-side bound of 12 and the\n *   float32 likelihood noise allowance. A bound-pinned GPU status 3 can converge here.\n * - Singular pivots use the shared float64 floor, 64 * Number.EPSILON of the information diagonal,\n *   instead of the GPU's 1e-3 pivot limit. Some GPU status 2 fits succeed here.\n * - Single-column separation is tested over the entire class supports before iteration. General\n *   Fisher directions (including the GPU's cleaned candidate) are certified with exact dyadic\n *   arithmetic on raw inputs. Certificates may be found sooner; an unclassified, nearly separating\n *   direction blocks status 0. The candidate search is still incomplete for multiple columns.\n * - Diagnostic probabilities stay float64 until output, with the same [1e-7, 1-1e-7] clamp.\n *   P-values call the public chi-square survival helper. Subnormals are retained.\n * - Optional spatialLag uses the same weighted probability-residual linearization, unweighted\n *   instruments, complete-case induced graph, status codes and HC0 structural-residual sandwich.\n *   Its two stages retain the GPU's 1e-3 factor-pivot limit; intermediates remain float64.\n *\n * @throws If predictors, mask or diagnostic arrays have inconsistent lengths, naming the argument.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialTwoStageLeastSquares': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialTwoStageLeastSquaresOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialTwoStageLeastSquaresOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialTwoStageLeastSquaresOnCPU',
        signature:
          'function computeSpatialTwoStageLeastSquaresOnCPU( input: CPUSpatialTwoStageLeastSquaresInput, options: CPUSpatialTwoStageLeastSquaresOptions ): CPUSpatialTwoStageLeastSquaresResult',
        supportedInputs: [
          'input: CPUSpatialTwoStageLeastSquaresInput',
          'options: CPUSpatialTwoStageLeastSquaresOptions'
        ],
        differences: [
          "/**\n * Spatial lag IV with instruments [1,X,WX(,W²X)], using centered Householder QR projections.\n * The structural residuals and every diagnostic use the float64 estimate before publication.\n * Moran's I is NaN at zero total weight; the cancelled Anselin-Kelejian statistic remains defined.\n *\n * What differs from the GPU contributor:\n * - Float64 sparse lags, centered QR projection, solves and residuals replace float32 lags,\n *   normal equations and one refinement step. Rounding occurs once per returned output. Compare\n *   ordinary well-conditioned fits at 5e-3 relative tolerance (5e-4 for table/residuals).\n * - Shader predictor/instrument caps, tile budget and float32 row-count limits are dropped.\n *   Instrument order remains 1 or 2. Rank uses norm-scaled QR pivots > 64 * Number.EPSILON.\n * - The ordinary comparison tolerances do not bound ill-conditioned instrument systems. For\n *   adjacent-pair weights and x=[-1,-1+epsilon,1,1,-1,-1,1,1-epsilon], epsilon=2^-24, the\n *   instrument correlation condition number is about 2.25e15. The GPU accepted the fit with\n *   status 0 but its intercept differed by 2.63% from the exact IV intercept (4.5 after rounding).\n *   The CPU retains the float64 QR estimate; GPU status 0 with a precision warning does not\n *   imply agreement of coefficients, covariance, residuals or derived diagnostics in this case.\n * - Conditioning uses the shared float64 Jacobi singular values without the GPU sweep cap.\n *   Probability calculations call the public chi-square survival helper, retaining its cap.\n * - HC0, HC1 and HAC use float64 influence rows and structural residuals, avoiding the GPU's\n *   second rounding of raw lag columns and residuals before robust covariance. Kernel symmetry\n *   and diagonal tolerances remain 1e-5; PSD remains a caller precondition.\n * - HAC loss flags may be supplied as flat robust arrays or kernel.status arrays. CPU loss\n *   arrays hold exactly one word; GPU status views may have spare capacity. Both consume word 0.\n * - HC2/HC3 leverage and the shifted PSD test use float64 with the same 1e-6 and 1e-5\n *   tolerances as the GPU. CLUSTER sums scores in input order, rather than the GPU segment tree.\n * - Non-finite active data/weights return solver status 1, matching the current kernel's failed\n *   solve branch (despite its status-3 documentation); skipped neighbor sentinels ignore weights.\n *   Subnormals are retained. Inconsistent model CSR arrays throw rather than reading outside\n *   storage; malformed diagnostic/HAC row bounds return their defined failure statuses.\n * - Insufficient-row fits return NaN residuals. The GPU publishes its residual pass before\n *   selecting status 2 and does not clear those rows; a full-rank saturated fit can therefore\n *   leave finite residuals (zero to rounding for y=X) despite its failed table and summary.\n *\n * @throws If predictors, weights or robust kernel arrays are inconsistent, naming the argument.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialErrorMaximumLikelihood': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialErrorMaximumLikelihoodOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialErrorMaximumLikelihoodOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialErrorMaximumLikelihoodOnCPU',
        signature:
          'function computeSpatialErrorMaximumLikelihoodOnCPU( input: CPUSpatialErrorMaximumLikelihoodInput, options: CPUSpatialErrorMaximumLikelihoodOptions ): CPUSpatialErrorMaximumLikelihoodResult',
        supportedInputs: [
          'input: CPUSpatialErrorMaximumLikelihoodInput',
          'options: CPUSpatialErrorMaximumLikelihoodOptions'
        ],
        differences: [
          "/**\n * Fits y = X b + u, u = lambda W u + e, using centered float64 QR and direct residuals.\n * The likelihood is the requested trace-series (including optional geometric tail) or supplied\n * eigenvalue likelihood. Its rational profile derivative is isolated without a sampling grid.\n *\n * What differs from the GPU contributor:\n * - Float64 centered QR, residuals and probe chains replace float32 moments/Cholesky; arrays round\n *   once at output. Well-conditioned GPU comparisons use 5e-3 relative tolerance. Residuals and\n *   pseudo R squared use the centered fit; squared correlation uses ratios, avoiding raw\n *   intercept cancellation and products of large sums of squares.\n *   Predictions come directly from the centered slopes; zero prediction variance gives NaN.\n *   Exact zero-moment certificates suppress QR noise for an orthogonal response.\n * - Integer polynomial determinants of the float64 filtered rows and Bernstein root isolation\n *   find every positive-to-negative stationary crossing to binary64 resolution, without the\n *   GPU's 1024-interval scan or 32-bisection limit. Equal objectives choose the lowest lambda;\n *   an identically flat profile chooses the lower boundary. localMaximumCount can exceed the GPU's.\n *   For exact log determinant with structurally certified W=cI, both forms choose that lower\n *   boundary, report one candidate and suppress the unidentifiable spatial SE/z/p regardless\n *   of the inverse-series length. Coefficient inference remains defined.\n *   Weights are normalized before multiplication so a scalar identity retains its flat profile;\n *   the GPU's float32 search can choose another point on that profile.\n * - GPU-only predictor, power and exact-row caps are dropped; tileRowCount has no numerical effect.\n *   QR rank uses 64 * Number.EPSILON; some GPU status-1 fits succeed. Conditioning uses converged\n *   float64 singular values. Polynomial isolation is intended for small reference-tier fits.\n * - The requested finite trace series, geometric tail model, public tail correction and Q-term\n *   inverse series for the variance are retained, including on the eigenvalue path. The tail\n *   correction calls the public helper with its existing stopping rule. Stochastic mode retains\n *   the default mode threshold, Philox seed/probe order, standard errors and approximation flag.\n * - P-values call the public float64 chi-square survival helper (its iteration cap remains).\n *   Subnormals are retained. Non-finite data/eigenvalues give status 3 (status 2 takes precedence\n *   for too few rows); non-finite active weights make stochastic trace errors NaN (exact-mode\n *   trace errors remain zero). Inactive capacity and out-of-range neighbor sentinels are ignored.\n *   Empty slot arrays are accepted by the pure\n *   function. Whole CSR is supported; optional chunked weights are omitted from the operation.\n * - Finite float64 norms and moments are retained when the GPU's float32 norm/moments overflow;\n *   some GPU status-3 fits succeed here. Final float32 outputs may underflow or overflow.\n * - An exact augmented-Gram certificate rejects an exactly fitted response with status 3 and\n *   NaN fit outputs: zero variance has no finite Gaussian maximum. The GPU can instead publish\n *   a tiny positive variance created by rounding. No small nonzero residual is thresholded away.\n *\n * @throws If weights, predictors, response, eigenvalues or traceParameters arrays are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialLagMaximumLikelihood': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialLagMaximumLikelihoodOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialLagMaximumLikelihoodOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialLagMaximumLikelihoodOnCPU',
        signature:
          'function computeSpatialLagMaximumLikelihoodOnCPU( input: CPUSpatialLagMaximumLikelihoodInput, options: CPUSpatialLagMaximumLikelihoodOptions ): CPUSpatialLagMaximumLikelihoodResult',
        supportedInputs: [
          'input: CPUSpatialLagMaximumLikelihoodInput',
          'options: CPUSpatialLagMaximumLikelihoodOptions'
        ],
        differences: [
          "/**\n * Fits y = rho W y + X b + e with centered float64 QR. The finite trace-series likelihood\n * (or the supplied eigenvalue likelihood) is maximized over the contributor's norm-scaled bound.\n * The information matrix, residuals, trace outputs and all 22 summary fields use that same fit.\n * Pseudo-R squared uses the contributor's residual moments, including NaN for zero fitted variance.\n *\n * What differs from the GPU contributor:\n * - Float64 centered QR, direct residuals, sparse chains and scaled information replace float32\n *   moments/Cholesky; outputs round once. Ordinary GPU comparisons use 5e-3 relative tolerance.\n *   QR rank uses 64 * Number.EPSILON, so some GPU status-1 designs succeed. Conditioning uses\n *   converged float64 singular values. Subnormals are retained.\n * - The existing integer-polynomial/Bernstein helper isolates all stationary maxima to binary64\n *   resolution, removing the 1024-interval scan and 32-step bisection caps. Equal objectives choose\n *   the lowest rho; an identically flat profile chooses the lower boundary. The local maximum\n *   count can exceed the GPU's. This exact search is intended for small reference-tier fits.\n * - GPU-only predictor, power and exact-row limits are dropped; tileRowCount has no numerical\n *   effect. Whole CSR is supported; optional chunked weights are omitted from the operation.\n * - The requested Q-term determinant/trace series, geometric tail model, public tail correction\n *   stopping rule and Q-term inverse series for prediction and variance are retained. The default\n *   mode threshold, Philox seed/probe order, sampling errors and approximation flag are retained.\n *   P-values use the public float64 chi-square survival helper, including its stopping rule.\n * - Stationarity is checked from unrounded float64 residuals with a binary64 spacing allowance.\n *   A finite rejected iterate is published with status 3 and converged 0; the GPU clears its table\n *   and summary in this branch, retaining only residuals. A zero-RSS pole has no finite maximizer\n *   and returns status 3 with NaN fit rows. S = 0 outside the bound and e1'e1 = 0 remain admissible.\n *   An exact integer rank certificate on the original float32 data distinguishes true poles\n *   from positive residual minima, even when the QR leaves a positive rounding residual.\n * - Non-finite data/eigenvalues/active weights return status 3 (too few rows takes precedence).\n *   Non-finite active weights give NaN trace outputs. Unused capacity and out-of-range neighbor\n *   sentinels are ignored; empty slot arrays are accepted by the pure function. Zero-link weights\n *   return status 3, while their trace infinityNorm is 1, matching the trace contributor fallback.\n *   The GPU and CPU traces retain positive norms throughout the finite float32 range.\n *   The CPU still normalizes in float64 when its published infinityNorm rounds to infinity.\n *   The bundled GPU trace call shares the power-trace normalization and requests powerTraces,\n *   powerSums, infinityNorm and optional powerTraceStandardErrors. It omits gramTraces,\n *   so the bundle does not activate Gram protection.\n *\n * @throws If weights, predictors, response, eigenvalues or traceParameters arrays are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPanelLagMaximumLikelihood': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialPanelLagMaximumLikelihoodOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialPanelLagMaximumLikelihoodOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialPanelLagMaximumLikelihoodOnCPU',
        signature:
          'function computeSpatialPanelLagMaximumLikelihoodOnCPU( input: CPUSpatialPanelLagMaximumLikelihoodInput, options: CPUSpatialPanelLagMaximumLikelihoodOptions ): CPUSpatialPanelLagMaximumLikelihoodResult',
        supportedInputs: [
          'input: CPUSpatialPanelLagMaximumLikelihoodInput',
          'options: CPUSpatialPanelLagMaximumLikelihoodOptions'
        ],
        differences: [
          "/**\n * Balanced random/unit-effects lag ML with centered float64 QR, direct residual squares and\n * dense log determinants, or the supplied entity spectrum for tiled unit effects.\n * Conditional covariance and spreg's reported likelihood convention\n * (without N log(phi), and with the extra FE constant) follow the current contributor.\n *\n * What differs from the GPU contributor:\n * - Float64 centering, entity means, QR, lags and residuals replace float32 packed moments and\n *   Cholesky. Residual-square totals use compensated float64 accumulation. Outputs round once.\n *   Ordinary comparisons use 5e-3 relative error, covariance\n *   scaled by diagonal geometric means; near-zero fitted/residual rows use a response-scale floor.\n * - Entity, period and predictor shader caps are dropped. Numerical rank uses the shared\n *   norm-scaled QR / correlation Cholesky float64 floors instead of the GPU's 1e-3 pivot floor.\n * - Explicit maximumIterations and maximumSearchIterations bind the CPU fit. Exhaustion returns\n *   status 3 with its iterate and bracket width, withholding covariance. When budgets are omitted,\n *   golden section continues to binary64 resolution including interval endpoints, and alternation\n *   continues to the same relative phi tolerance or a repeated complete state (status 3).\n *   Sweep counts, phi change and bracket width can differ. The CPU compares float64 profile\n *   values; GPU fine brackets use the analytic likelihood derivative after eight coarse golden\n *   steps. Neither bounded profile method guarantees a global maximum on a multimodal profile.\n * - The rho interval, phi floor 1e-8, boundary bits, and status-3 publication at that floor are\n *   retained. Covariance is withheld for status 3. Variance that rounds to nonpositive or\n *   nonfinite float32 is status 2, with covariance/fitted/residuals withheld. Hausman keeps\n *   RE-minus-FE covariance and df=predictorCount+2, and uses the public chi-square helper.\n *   On representability failures the CPU retains its float64 iterate and likelihood before\n *   output rounding; the GPU's earlier float32 overflow/underflow can change those diagnostics.\n * - A float64 profile score/curvature check rejects nonmaximal interior candidates as status 3\n *   with the iterate retained. Exact response copies of a design column are detected independently\n *   of QR. Other zero-variance poles with admissible rho are detected when the augmented QR\n *   estimate passes an exact dyadic residual certificate; this is not a complete symbolic rank\n *   test. An additional original-input BigInt certificate checks within differences with\n *   24 nonzero-preserving rounded slope/rho proposals plus 24 zero proposals, including\n *   the original float32 tuple and exact products of rho, weights and response.\n *   Both forms withhold every floating result for certified poles (status 2; Hausman status 1).\n *   The CPU certificate retains subnormals and arbitrary exponent spans, where the GPU's\n *   256-bit certificate supports normal/zero operands and a 168-bit product-exponent span.\n * - Weight normalization acceptance uses serial float32 addition and the shader's float32\n *   1e-5 predicate. Statistical lags and fitting still use the original float32 weights in float64.\n * - Non-finite observations and invalid active CSR rows return status 1; malformed row bounds\n *   also return status 1, as on the GPU. Array shape mismatches throw. Empty CSR slot arrays\n *   are accepted by the pure function and produce status 1. Subnormals are retained.\n * - Optional weights distances and status arrays retain shape validation; information metadata\n *   retains contributor validation. The first overflow or candidateOverflow word invalidates\n *   the fit. A supplied complete=0 invalidates the fit. Written count must agree with\n *   offsets[entityCount]-offsets[0]; requiredCount must fit that accessible span and slot capacity. Other status values and distances have no numerical effect.\n * - A supplied spectrum follows the GPU mode selection and validation: one finite value per\n *   entity with absolute value at most float32(1.00001). Invalid values return status 1;\n *   a wrong length throws. The caller is responsible for matching the spectrum to W.\n *\n * @throws If predictors, response, weights, eigenvalue lengths, overflow or fitMode are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUContiguityWeights': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUContiguityWeightsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUContiguityWeightsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeContiguityWeightsOnCPU',
        signature:
          'function computeContiguityWeightsOnCPU( input: CPUContiguityWeightsInput, options: { criterion: GPUContiguityCriterion; capacity: number; snapTolerance?: number; pairCapacity?: number; } ): CPUContiguityWeightsResult',
        supportedInputs: [
          'input: CPUContiguityWeightsInput',
          'options: {\n    criterion: GPUContiguityCriterion;\n    capacity: number;\n    snapTolerance?: number;\n    pairCapacity?: number;\n  }'
        ],
        differences: [
          "/**\n * Constructs exact queen/rook adjacency using shared point/edge keys, including holes and parts.\n * Directed slots are sorted by (feature, neighbor), then truncated to capacity. As on the GPU,\n * offsets are clamped and the count is the terminal offset; there is no separate count port.\n * Snapping is anchored at ringOffsets[0], falling back to zero for empty or non-finite input.\n *\n * What differs from the GPU contributor:\n * - An explicit pairCapacity binds candidate emission before deduplication; candidateOverflow\n *   reports loss and requiredCount is then a lower bound. Without an explicit budget the CPU\n *   uses no bounded pair scratch. Under candidate loss the retained prefix can differ because\n *   GPU float32 key ordering and CPU point-key insertion order can differ. Final overflow is\n *   separate and depends only on retained distinct links exceeding final capacity.\n * - Positive-tolerance snap keys use float64 round-half-up, without the GPU's float32 rounding,\n *   2^23-step shortcut or signed-integer clamp. Cells near float32 rounding boundaries may differ.\n * - The GPU snap-cell range limit is dropped. Zero tolerance still compares exact input float32\n *   values, normalizes signed zero, and ignores non-finite vertices.\n * - Unused output slots are zero-filled; the GPU leaves them unspecified. Integer used rows and\n *   binary weights agree exactly when snap cell identities agree and GPU candidates fit.\n *\n * @throws If positions is not paired, an offset array is inconsistent, or options are invalid.\n */"
        ]
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePolygonContiguityOnCPU',
        signature:
          'function computePolygonContiguityOnCPU( topology: PreparedPolygonContiguityTopology, options: {criterion: GPUContiguityCriterion; capacity: number} ): CPUContiguityWeightsResult',
        supportedInputs: [
          'topology: PreparedPolygonContiguityTopology',
          'options: {criterion: GPUContiguityCriterion; capacity: number}'
        ],
        differences: [
          '/**\n * Optional CPU execution over the same exact-source equality tokens used by GPUContiguityWeights.\n * Always uses zero snap tolerance: tokens are identities, never metric coordinates. The CPU twin\n * has no candidate-pair overflow and reports an exact requiredCount; final capacity still truncates.\n * Empty and invalid rows remain isolated and source aligned via geometryMask.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialWeightsValidation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialWeightsValidationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialWeightsValidationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialWeightsValidationOnCPU',
        signature:
          'function computeSpatialWeightsValidationOnCPU( offsets: Uint32Array, neighbors: Uint32Array, weights: Float32Array, options: SpatialWeightsValidationOptions = {} ): CPUSpatialWeightsValidationResult',
        supportedInputs: [
          'offsets: Uint32Array',
          'neighbors: Uint32Array',
          'weights: Float32Array',
          'options: SpatialWeightsValidationOptions = {}'
        ],
        differences: [
          "/**\n * Checks whole CSR and returns the contributor's exact per-row violation bits and invalid count.\n * Only slots in [begin, min(end, capacity)) are inspected, even for malformed offsets. Offset\n * violations are attributed to row 0, the decreasing row, or the final row exactly as on the GPU.\n *\n * What differs from the GPU contributor:\n * - Zero neighbor capacity is accepted: offsets=[0,0] with empty neighbors and weights returns\n *   flags=[0] and invalidCount=[0]. The GPU contributor requires positive neighbor capacity.\n * - Chunked CSR is omitted from the operation form.\n *\n * @throws If offsets has fewer than two entries or weights and neighbors have different lengths.\n * Invalid CSR contents are reported in flags, not thrown.\n */",
          'SpatialWeightsValidationOptions applies to direct and named forms: columnCount is the source-column count (default target-row count), permitting rectangular CSR. diagonal accepts forbid (default), allow or require-unit. require-unit requires columnCount >= rowCount and one weight=1 diagonal per target row. CPUSpatialWeightsValidationResult contains flags in the target-row domain and invalidCount. Invalid CSR contents set flags; invalid option values reject with accepted values.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialWeightsAlgebra': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialWeightsAlgebraOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialWeightsAlgebraOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialWeightsAlgebraOnCPU',
        signature:
          'function computeSpatialWeightsAlgebraOnCPU( input: CPUSpatialWeightsAlgebraInput, options: CPUSpatialWeightsAlgebraOptions ): CPUSpatialWeightsAlgebraResult',
        supportedInputs: [
          'input: CPUSpatialWeightsAlgebraInput',
          'options: CPUSpatialWeightsAlgebraOptions'
        ],
        differences: [
          '/**\n * Computes sparse set algebra, shortest-path neighborhoods and space-time products in float64,\n * rounding weights and distances once on output. Higher-order expansion follows ascending\n * frontier order; a capacity-limited final frontier is truncated before sorting, as on GPU.\n *\n * What differs from the GPU contributor:\n * - Higher-order searches use dynamic frontiers when workCapacity is omitted. An explicit\n *   workCapacity clamps each intermediate matrix in row order, exactly as on GPU, and sets\n *   candidateOverflow. The order-32 GPU cap is dropped. After loss requiredCount counts the\n *   retained computation and is not a bound on the complete exact-order result.\n * - The GPU limit of 16 temporal lags is dropped. Index and output sizes remain uint32.\n *   Empty input slot arrays are accepted; output capacity remains at least one.\n * - Arithmetic is float64 with one output rounding; near rounding boundaries products/sums\n *   may differ. Nonfinite per-row self weights are copied, as the GPU does for that view.\n * - Malformed CSR offsets, unsorted/duplicate neighbors and inconsistent array lengths throw.\n *   Unused output capacity is zero; GPU unused capacity is unspecified.\n *\n * @throws Error naming an inconsistent input array or invalid option.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrackSimilarity': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrackSimilarityOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrackSimilarityOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrackSimilarityOnCPU',
        signature:
          'function computeTrackSimilarityOnCPU( input: CPUTrackSimilarityInput, options: CPUTrackSimilarityOptions = {} ): CPUTrackSimilarityResult',
        supportedInputs: [
          'input: CPUTrackSimilarityInput',
          'options: CPUTrackSimilarityOptions = {}'
        ],
        differences: [
          "/**\n * Computes the requested trajectory measures, with the contributor's densification, inclusive\n * matching thresholds, widened Sakoe-Chiba bands and original-vertex SSPD/maximum distance.\n *\n * What differs from the GPU contributor:\n * - Displayed arithmetic uses float64 and rounds once to float32 output; compare noninteger distances with relative tolerance 3e-5.\n *   Numeric match tolerances and the ERP gap point are converted to float32 as GPU constants are.\n * - The shorter-track, subdivision, pair-dispatch and band-index GPU limits are dropped.\n *   An explicit maximumFrechetVertexCount bounds dynamic-program measures, setting\n *   frechetCapExceeded and leaving their outputs unavailable when exceeded.\n *   Measures that densify reject expanded tracks above 2^32 - 1 vertices before allocation.\n * - Dynamic programs compute the complete documented method without GPU strips or early exits.\n *   DTW costs are not saturated at the largest finite float32; output overflow becomes Infinity.\n * - Inconsistent arrays and decreasing/out-of-range lineOffsets throw an Error naming the array.\n *   Nonfinite coordinates or geographic latitudes outside [-90, 90] leave distances NaN and\n *   frechetWithin zero, with invalidGeometry evidence. Invalid epsilon affects only decision validity.\n *\n * @throws Error naming an argument when array layouts or options are inconsistent.\n */",
          'Planar radius membership uses inclusive Euclidean distance on stored coordinates, retaining the comparison residual rather than deciding from a rounded norm. This rule applies to track matching, location connectivity and spatial-validation buffers. Genuine arithmetic differences remain for displayed distances, interpolated/densified geometry, great-circle arithmetic and non-buffer clustering/objective ties.',
          'A displayed Frechet distance can equal the radius with membership false. Invalid geometry makes affected outputs unavailable with pair bit 8 invalidGeometry; invalid epsilon makes decision outputs unavailable with bit 16 invalidDecisionThreshold. Decode pairStatus with decodeGPUTrackSimilarityPairStatus; metric validity remains separate from rounded display.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectorySplit': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectorySplitOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectorySplitOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectorySplitOnCPU',
        signature:
          'function computeTrajectorySplitOnCPU( input: CPUTrajectorySplitInput, options: CPUTrajectorySplitOptions ): CPUTrajectorySplitResult',
        supportedInputs: ['input: CPUTrajectorySplitInput', 'options: CPUTrajectorySplitOptions'],
        differences: [
          "/**\n * Splits each track into maximal runs, drops runs shorter than two rows and repeats the first\n * row of the next run in value-change mode. Segment and row capacities are independent, exactly\n * as in the contributor (including its terminal offset when segment capacity overflows).\n *\n * What differs from the GPU contributor:\n * - Time differences use float64; Int64 words are subtracted exactly before conversion. A gap\n *   decision at a float32 rounding boundary can differ.\n * - Decreasing trackOffsets after the GPU's row-count clamp throw an Error.\n * - Unwritten rowIds and copied geometry capacity rows are zero. The GPU leaves them untouched.\n * - Nonfinite timestamps follow JavaScript comparison rules; GPU behavior is unspecified there.\n *\n * @throws Error naming an inconsistent positions, timestamps, trackOffsets, parameters,\n * staypointIds or values argument.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryThinning': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectoryThinningOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectoryThinningOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectoryThinningOnCPU',
        signature:
          'function computeTrajectoryThinningOnCPU( input: CPUTrajectoryThinningInput, options: CPUTrajectoryThinningOptions ): CPUTrajectoryThinningResult',
        supportedInputs: [
          'input: CPUTrajectoryThinningInput',
          'options: CPUTrajectoryThinningOptions'
        ],
        differences: [
          '/**\n * Follows the first later fix at or beyond the tolerance from each kept fix; always keeps both\n * track endpoints. Validity describes the full selection even when output capacity overflows.\n *\n * What differs from the GPU contributor:\n * - Omitted maximumSearchRows leaves CPU searches uncapped; explicit caps split the chain and\n *   set candidateOverflow exactly as on the GPU.\n * - Distances and time differences use float64 with one output rounding; Int64 subtraction is\n *   exact before conversion. Threshold decisions at float32 rounding boundaries can differ.\n * - Decreasing trackOffsets after clamping throw an Error.\n * - Unwritten copied geometry capacity rows are zero; the GPU leaves them untouched.\n * - Nonfinite distances or times fail the tolerance comparison when NaN; track endpoints are\n *   still kept. GPU results for nonfinite coordinates or times are unspecified.\n *\n * @throws Error naming inconsistent positions, timestamps, trackOffsets or parameters.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUBrownianBridgeUtilization': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUBrownianBridgeUtilizationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUBrownianBridgeUtilizationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeBrownianBridgeUtilizationOnCPU',
        signature:
          'function computeBrownianBridgeUtilizationOnCPU( input: CPUBrownianBridgeUtilizationInput, options: { gridSize: readonly [number, number]; subStepCount?: number; tailPolicy?: BrownianBridgeTailPolicy; } ): CPUBrownianBridgeUtilizationResult',
        supportedInputs: [
          'input: CPUBrownianBridgeUtilizationInput',
          'options: {\n    gridSize: readonly [number, number];\n    subStepCount?: number;\n    tailPolicy?: BrownianBridgeTailPolicy;\n  }'
        ],
        differences: [
          '/**\n * Computes time-weighted Gaussian bridge density at each row-major grid cell centre. Positive\n * duration segments in each track contribute to the normalization, including zero-variance\n * segments; non-positive durations are skipped. Negative motion variances clamp to zero.\n *\n * What differs from the GPU contributor:\n * - Omitted tailPolicy keeps the full Gaussian tail (GPU default: truncate-at-seven-sigma).\n *   Explicit policies agree: truncation skips a segment outside its conservative seven-sigma box.\n * - Arithmetic uses float64 with one float32 output rounding; Int64 differences are exact before\n *   float64 conversion. Comparisons use absolute/relative tolerances, rather than bitwise equality.\n * - Fix-count and subStepCount GPU maxima are dropped. The requested positive integer midpoint\n *   count still defines the integration rule; this does not replace it with continuous quadrature.\n * - All forms preserve the requested grid shape for zero or one fix, returning zero density.\n * - Offsets are clamped to the row count, then decreasing offsets throw. The GPU does not\n *   validate ordering; decreasing offsets can produce overlapping track ranges there.\n * - Non-finite motion variances are not defined by the GPU contributor. The twin clamps\n *   -Infinity to zero; NaN and +Infinity contribute no density while retaining their positive\n *   segment duration in the denominator. For two stationary fixes at times [0, 1], motion\n *   variances [NaN, 0], location errors [1, 1], one midpoint and a cell centred at zero, the twin\n *   returns density 0; the GPU returned 0.31830987334251404 (measured on an Apple GPU, 2026-11-11).\n *\n * @throws Error naming any input array whose packed length or offsets are inconsistent.\n */',
          'tailPolicy accepts full or truncate-at-seven-sigma. The pure CPU and CPU named operation default to full; GPU direct and named forms default to truncate-at-seven-sigma. Both explicit policies are supported by every form. subStepCount defaults to 15 midpoint steps and controls integration independently of the tail rule.',
          'All direct and named forms accept zero or one fix and preserve the requested grid shape with zero density. CPU input.storage defaults to resident-capacity (clamp offsets then reject decreases); compact requires initial zero, nondecreasing offsets within rowCount and exact terminal rowCount. GPU resident views do not validate offset ordering.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMobilityIndicators': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUMobilityIndicatorsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUMobilityIndicatorsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeMobilityIndicatorsOnCPU',
        signature:
          'function computeMobilityIndicatorsOnCPU( input: CPUMobilityIndicatorsInput, options: CPUMobilityIndicatorsOptions ): CPUMobilityIndicatorsResult',
        supportedInputs: [
          'input: CPUMobilityIndicatorsInput',
          'options: CPUMobilityIndicatorsOptions'
        ],
        differences: [
          '/**\n * Computes staypoint mobility measures with float64 distances, maps for location shares/durations\n * and an uncapped Lempel-Ziv estimator. Radius of gyration uses spherical centering for great-circle contexts by default; coordinate-mean remains explicit. Missing and unsettled location keys are excluded from location indicators and reported in excludedLocationCount. Tracking quality\n * uses positive endpoint time differences, not the staypoint duration column; overlaps are summed.\n *\n * What differs from the GPU contributor:\n * - Arithmetic uses float64 and rounds each output once. Int64 timestamps are subtracted exactly\n *   before conversion. Distances, entropies and duration-ranking ties near rounding can differ.\n * - Real entropy uses the full sequence unless the maximumEntropySequenceRows option is given;\n *   then, as on the GPU, a longer track gets NaN and sets candidateOverflow. The operation twin\n *   passes the cap only when its props set one, so the GPU default cap of 1024 is not applied.\n *   candidateOverflow also reports supplied staypoint loss. includeRealEntropy=false skips that\n *   unused calculation and ignores its cap. An explicit cap is validated only when entropy is\n *   requested; the CPU operation validates that prop at construction.\n * - Inconsistent columns, invalid row indices and unordered active track IDs throw an Error.\n * - Nonfinite coordinates/durations/times use JavaScript arithmetic and comparisons; GPU results\n *   for such inputs are unspecified.\n *\n * @throws Error naming an inconsistent timestamps, staypoints or triplegs argument.\n */',
          'The great-circle default gyrationCenter is spherical-mean; explicit coordinate-mean retains arithmetic longitude/latitude centering. Planar contexts default to coordinate-mean and reject spherical-mean. A mean unit-vector norm <= 1e-6 is indeterminate (including cancelling antipodes): radiusOfGyration is NaN. For sphereRadius=6371000 meters, rotating [-1,1] to [179,-179] preserves radius 111194.93 meters (R*pi/180). Source centroid coordinates are longitude/latitude degrees; distance outputs follow spatialContext units.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUModalSplit': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUModalSplitOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUModalSplitOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeModalSplitOnCPU',
        signature:
          'function computeModalSplitOnCPU( input: CPUModalSplitInput, options: CPUModalSplitOptions ): CPUModalSplitResult',
        supportedInputs: ['input: CPUModalSplitInput', 'options: CPUModalSplitOptions'],
        differences: [
          '/**\n * Classifies average tripleg speeds and accumulates count, path length or endpoint duration by\n * track and mode. Missing modes are excluded; other out-of-range keys enter the last mode. Speed\n * class bounds are exclusive and the final class is unbounded. As in the GPU literal builder,\n * nonfinite speed breaks become the largest finite float32.\n *\n * What differs from the GPU contributor:\n * - Lengths, speeds and sums use float64, with one rounding at the metric output. Near speed\n *   thresholds and for paths whose float32 length overflows, classifications can differ.\n * - Int64 timestamps are subtracted exactly before conversion to float64.\n * - Decreasing clamped trackOffsets and inconsistent, unordered or out-of-track tripleg arrays\n *   throw an Error. Empty triplegs are accepted for supplied-mode count/distance aggregation;\n *   prediction and duration aggregation still reject them because endRows minus one does not\n *   identify a member of an empty tripleg.\n * - Prediction omits the optional input modes port, which the contributor otherwise writes in\n *   place. predictedModes without prediction is omitted from the operation outputs.\n * - Nonfinite coordinates/times use JavaScript arithmetic; nonfinite predicted speeds are missing.\n *   The GPU does not specify nonfinite trajectory input behavior.\n *\n * @throws Error naming the inconsistent input array or missing modes.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryRecursions': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectoryRecursionsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectoryRecursionsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectoryRecursionsOnCPU',
        signature:
          'function computeTrajectoryRecursionsOnCPU( input: CPUTrajectoryRows & {radii: Float32Array}, options: {maximumTrackSegments?: number} = {} ): CPUTrajectoryRecursionsResult',
        supportedInputs: [
          'input: CPUTrajectoryRows & {radii: Float32Array}',
          'options: {maximumTrackSegments?: number} = {}'
        ],
        differences: [
          "/**\n * Counts closed-disc visits, integrates residence times and finds the forward first exit,\n * using linear interpolation between fixes of the same track. Non-positive radii produce zero\n * times/counts and censored=1; rows outside all tracks do likewise. Non-positive segment\n * durations add no residence time, matching the GPU definition.\n *\n * What differs from the GPU contributor:\n * - Omitted maximumTrackSegments examines whole tracks. An explicit cap uses the GPU's centred\n *   segment windows, reports candidateOverflow and censors forward passage at the window end.\n * - The GPU radius-count maximum is dropped; the fix-major output layout is unchanged.\n * - Circle intersections and time sums use float64, rounded once on output. The intersection\n *   calculation uses projection onto the segment's supporting line to avoid subtracting large\n *   quadratic discriminant terms. Near-boundary decisions can differ from float32 GPU decisions.\n * - Int64 timestamps are subtracted exactly before conversion to float64.\n * - The pure function accepts empty trajectories or radii and returns empty row outputs. The CPU\n *   graph operation retains the GPU contributor's minima of one fix and one radius.\n * - Offsets are clamped to the row count, then decreasing offsets throw. The GPU does not\n *   validate ordering; decreasing offsets can produce overlapping track ranges there.\n * - Non-finite positions are not defined by the GPU contributor. In the twin, segments touching\n *   NaN coordinates have no inside interval; an outside endpoint that ends passage on such a\n *   segment uses its start time. For positions [(0, 0), (NaN, 0), (2, 0)], times [0, 1, 2], one\n *   track and radius 1, the twin returns revisits [0, 0, 0], residence [0, 0, 0], passage [1, 1, 0]\n *   and censored [0, 1, 1]. The device comparison returned GPU [2, 2, 2] where the twin returned\n *   [0, 0, 0] (measured on an Apple GPU, 2026-11-11); that undefined GPU result is not matched.\n *\n * @param options Optional positive safe-integer maximumTrackSegments; omission examines whole tracks.\n * @throws Error naming maximumTrackSegments, positions, timestamps or trackOffsets for invalid input.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryStaypoints': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectoryStaypointsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectoryStaypointsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectoryStaypointsOnCPU',
        signature:
          "function computeTrajectoryStaypointsOnCPU(input: CPUTrajectoryStaypointsInput, options: { capacity: number; spatialContext?: GPUSpatialContext; maximumSearchRows?: number; missingFixPolicy?: 'break'; }): CPUTrajectoryStaypointsResult",
        supportedInputs: [
          'input: CPUTrajectoryStaypointsInput',
          "options: { capacity: number; spatialContext?: GPUSpatialContext; maximumSearchRows?: number; missingFixPolicy?: 'break'; }"
        ],
        differences: [
          "/**\n * Sequential sliding-anchor staypoints, with the contributor's inclusive distance/time tests,\n * strict gap test, distinct-coordinate centroid and exclusive end rows.\n *\n * What differs from the GPU contributor:\n * - Omitted maximumSearchRows leaves CPU searches uncapped; explicit caps split windows,\n *   report candidateOverflow and write NaN durations and centroids on the cut track, as on the GPU.\n * - Arithmetic uses float64, rounded once to float32 outputs; threshold decisions near rounding\n *   boundaries can differ. Int64 timestamps are subtracted exactly before conversion to float64.\n * - No GPU row-count or search-budget limit is imposed. Output capacity still clamps count.\n * - Compact offsets must terminate at the row count; storage: 'resident-capacity' opts into\n *   GPU-compatible clamping. The GPU does not validate ordering.\n *\n * @throws Error naming positions, timestamps, trackOffsets or parameters for inconsistent arrays.\n */",
          'Geographic identity gives 180, -180 and 540 one longitude vote; longitude does not distinguish exact poles. Canonical representatives are used for distinct votes and center offsets only. Preserve planar equality, source coordinates and source row IDs. Substituting any seam spelling in [177, 180, 180] at latitude zero gives the same distinct-coordinate center 178.5 degrees.',
          'The float32 longitudes 179.99998474121094 and -180.00001525878906 share one vote in both forms. With 177 degrees as the other distinct vote at latitude zero, the center before float32 rounding is 178.49999237060547 degrees.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryLocations': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectoryLocationsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectoryLocationsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectoryLocationsOnCPU',
        signature:
          'function computeTrajectoryLocationsOnCPU( input: CPUTrajectoryLocationsInput, options: {spatialContext?: GPUSpatialContext; maximumIterations?: number} = {} ): CPUTrajectoryLocationsResult',
        supportedInputs: [
          'input: CPUTrajectoryLocationsInput',
          'options: {spatialContext?: GPUSpatialContext; maximumIterations?: number} = {}'
        ],
        differences: [
          "/**\n * Finds same-track radius-connected components with exact union-find or explicitly bounded\n * hook/compress rounds, labels them in first-visit order, and averages distinct coordinates\n * with the GPU's anchor-relative longitude convention.\n * Active count is clamped to input capacity; epsilon is inclusive.\n *\n * What differs from the GPU contributor:\n * - Without an explicit maximumIterations, components are exact. With an explicit budget,\n *   bounded hook/compress rounds and the final convergence check match the GPU algorithm.\n *   Unsettled labels publish only sentinel IDs and a zero location count in both forms.\n * - Distances and centroid sums use float64, rounded once for centers; near-boundary centers can differ. Wrapped longitude is rewrapped after float32 rounding.\n * - Unordered active tracks throw an Error. Unwritten center capacity rows are zero; the GPU\n *   leaves them untouched.\n * - NaN coordinates have no neighbors and each NaN coordinate is distinct for centroid voting;\n *   GPU behavior for nonfinite coordinates is unspecified.\n *\n * @throws Error naming inconsistent centroids, tracks, count or parameters.\n */",
          'Planar radius membership uses inclusive Euclidean distance on stored coordinates, retaining the comparison residual rather than deciding from a rounded norm. This rule applies to track matching, location connectivity and spatial-validation buffers. Genuine arithmetic differences remain for displayed distances, interpolated/densified geometry, great-circle arithmetic and non-buffer clustering/objective ties.',
          'Geographic identity gives 180, -180 and 540 one longitude vote; longitude does not distinguish exact poles. Canonical representatives are used for distinct votes and center offsets only. Preserve planar equality, source coordinates and source row IDs. Substituting any seam spelling in [177, 180, 180] at latitude zero gives the same distinct-coordinate center 178.5 degrees.',
          'The float32 longitudes 179.99998474121094 and -180.00001525878906 share one vote in both forms. With 177 degrees as the other distinct vote at latitude zero, the center before float32 rounding is 178.49999237060547 degrees.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectorySegments': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectorySegmentsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectorySegmentsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectorySegmentsOnCPU',
        signature:
          'function computeTrajectorySegmentsOnCPU( input: CPUTrajectorySegmentsInput, options: CPUTrajectorySegmentsOptions ): CPUTrajectorySegmentsResult',
        supportedInputs: [
          'input: CPUTrajectorySegmentsInput',
          'options: CPUTrajectorySegmentsOptions'
        ],
        differences: [
          '/**\n * Constructs moving triplegs and the trackintel activity/gap event state machine. Trips consume\n * only the capacity-clamped tripleg table, as on GPU, so upstream tripleg overflow is kept separate\n * from trip overflow. Activity and gap comparisons are strict; staypoints win equal-start ties.\n *\n * What differs from the GPU contributor:\n * - Time differences use float64, with exact Int64 subtraction before conversion. Decisions at\n *   float32 rounding boundaries can differ.\n * - Decreasing clamped trackOffsets and inconsistent or unordered staypoint columns throw an Error.\n * - Nonfinite coordinates compare with JavaScript rules (NaN differs from itself); nonfinite\n *   timestamps use JavaScript comparisons. GPU results for these inputs are unspecified.\n *\n * @throws Error naming an inconsistent input array or missing staypoints for trip construction.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMaximumInscribedCircle': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUMaximumInscribedCircleOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUMaximumInscribedCircleOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeMaximumInscribedCircleOnCPU',
        signature:
          'function computeMaximumInscribedCircleOnCPU( input: CPUPolygonRows, tolerance: number, options: CircleWorkOptions = {} ): CPUMaximumInscribedCircleResult',
        supportedInputs: [
          'input: CPUPolygonRows',
          'tolerance: number',
          'options: CircleWorkOptions = {}'
        ],
        differences: [
          "/**\n * Best-first even-odd polygon circle search with dynamic live-cell storage in binary64.\n *\n * What differs from the GPU contributor:\n * - Cell storage is dynamic, but explicit cellCapacity and maximumIterations bind before\n *   subdivision at the same point as the contributor. Omitted limits allow search completion.\n * - The remaining-error field bounds the completed binary64 search, stopping at tolerance or\n *   binary64 resolution. Status 1 means a work/storage budget or binary64 resolution\n *   prevented subdivision before the requested tolerance was reached.\n * - With no explicit work limits, collapsed-axis polygons have an exact zero-radius result\n *   and zero remaining error. Explicit limits retain the contributor's subdivision/error rules.\n * - Float32 arithmetic overflow does not set status 3; only non-finite input does. Finite float32\n *   coordinates are evaluated in binary64 and outputs are rounded once.\n * - Live-slot winner ties, child order, last-vertex fallback and first closest-segment witness\n *   follow the GPU search. Different arithmetic can still resolve near ties differently.\n *\n * @throws If tolerance is not positive and finite, or input arrays are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNormalizeGeometry': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUNormalizeGeometryOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUNormalizeGeometryOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeNormalizeGeometryOnCPU',
        signature:
          'function computeNormalizeGeometryOnCPU( input: CPUNormalizeGeometryInput ): CPUNormalizeGeometryResult',
        supportedInputs: ['input: CPUNormalizeGeometryInput'],
        differences: [
          '/**\n * Normalizes finite leaves and stably orders collections in descending (kind, canonical contents)\n * order. Polygon exteriors stay first; holes sort after them. Leaf comparison is vertex-count\n * first; collection comparison is child-sequence first and count last. stableSourceId is copied\n * only through the unchanged nodes input and does not break ties, matching the kernel.\n *\n * What differs from the GPU contributor:\n * - Postorder evaluation and an explicit comparison stack replace dependency rounds. There is\n *   no nesting cap, so CPU status 2 never means depth exhaustion. The operation accepts but\n *   ignores maximumNestingDepth, including values over 64.\n * - Winding is an unbounded exact BigInt predicate on float32 coordinates, so uncertain GPU\n *   orientations succeed on the CPU. Finite subnormals are supported.\n * - Overlapping leaf vertex ranges or collection child-slot ranges throw an Error naming nodes;\n *   the GPU has conflicting writes and does not define their result. Nonfinite leaves retain\n *   their input coordinates with status 1; bad kind/range/child order and invalid parents get 2.\n * - All other output is copied exactly (no arithmetic rounding). Signed zero is preserved.\n *   Closed lines use the shell winding rule; zero-winding rings retain direction; duplicate\n *   minimum vertices retain the first minimum, as in the current kernel.\n *\n * @throws If arrays are not packed or node-owned ranges overlap. Individual invalid node ranges\n * and child references are reported with status 2 instead of throwing, as on the GPU.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNormalizePolygons': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUNormalizePolygonsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUNormalizePolygonsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeNormalizePolygonsOnCPU',
        signature:
          'function computeNormalizePolygonsOnCPU( input: CPUNormalizePolygonsInput ): CPUNormalizePolygonsResult',
        supportedInputs: ['input: CPUNormalizePolygonsInput'],
        differences: [
          "/**\n * Orients exterior rings clockwise and holes counterclockwise, selects the smallest complete\n * cyclic sequence, then stably sorts holes and polygon parts in descending canonical order.\n * Ring comparison starts with total vertex count, including a closing coordinate. Zero winding\n * retains its original direction; closure, duplicates, empty rings/parts and feature order remain.\n *\n * What differs from the GPU contributor:\n * - Winding is an unbounded exact BigInt predicate on float32 coordinates. The GPU's uncertain\n *   orientation (200 binary orders of product exponents) leaves the original direction alone;\n *   the CPU still orients such a ring. Finite subnormals are supported on the CPU.\n * - Sorting replaces repeated exhaustive ranking, retaining source order on equal comparisons.\n * - Slots not owned by a covered feature are zero; the GPU leaves these slots unwritten.\n * - Nonfinite coordinates throw an Error naming positions; the GPU requires finite inputs and\n *   does not define a canonical result for them. Negative zero becomes positive zero on output.\n *\n * @throws If positions are malformed/nonfinite or an offset table is inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSegmentSnap': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSegmentSnapOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSegmentSnapOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSegmentSnapOnCPU',
        signature:
          'function computeSegmentSnapOnCPU( input: CPUSegmentSnapInput, options: CPUSegmentSnapOptions ): CPUSegmentSnapResult',
        supportedInputs: ['input: CPUSegmentSnapInput', 'options: CPUSegmentSnapOptions'],
        differences: [
          "/**\n * Snaps endpoints and inserts unique nearby references in (projection fraction,reference row)\n * order. Consecutive equal coordinates are suppressed. Counts cover the complete result even\n * when only a prefix fits the fixed slots for a line.\n *\n * What differs from the GPU contributor:\n * - Distances and projection fractions use float64; near tolerance boundaries or projection ties\n *   can differ. The numeric tolerance is rounded to the GPU's float32 literal before use.\n * - Candidates are sorted once per segment instead of repeatedly scanning for the next winner.\n *   Reference duplicates retain their first row, exactly as on the GPU.\n * - Unused positions/sourceEdges slots are zero; the GPU leaves them unwritten. Only each line's\n *   written prefix is meaningful. Out-of-range offsets are clamped and reversed intervals are\n *   empty, as explicitly defined by the current kernel.\n * - Nothing else: nonfinite references invalidate every line, including empty lines; nonfinite\n *   input vertices invalidate their line. An empty line has validity=1; a collapsed line has\n *   issueCode bit 1 and validity=0.\n *\n * @throws If arrays are not packed, lineOffsets is too short, options are invalid, or the\n * uint32 required-count/output-indexing bounds would be exceeded.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConvexStraightSkeleton': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUConvexStraightSkeletonOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUConvexStraightSkeletonOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeConvexStraightSkeletonOnCPU',
        signature:
          'function computeConvexStraightSkeletonOnCPU( input: CPUConvexStraightSkeletonInput, options: CPUConvexStraightSkeletonOptions ): CPUConvexStraightSkeletonResult',
        supportedInputs: [
          'input: CPUConvexStraightSkeletonInput',
          'options: CPUConvexStraightSkeletonOptions'
        ],
        differences: [
          "/**\n * Advances strictly convex planar rings through simultaneous edge collapses, retaining the GPU's\n * vertex order, source edge IDs, tolerance grouping, strict offset-before-event test and ridge ties.\n * Ring offsets are clamped to positions length; decreasing clamped endpoints report input flag 1\n * for that ring, without preventing subsequent rings from being processed.\n *\n * What differs from the GPU contributor:\n * - Arithmetic is float64, rounded once at output; near event/tolerance boundaries topology can differ.\n * - Collapse events run to completion when maximumIterations is omitted; an explicit budget\n *   binds at the same point as the contributor, after offset capture and before arc emission.\n * - The GPU's 64-vertex ceiling and 3e38 event-time sentinel are removed. maximumVertices still\n *   determines output block size; larger rings receive input flag 1, as on the GPU.\n * - Unused output rows are zeroed. The contributor leaves those rows untouched on reused buffers.\n * - Nonfinite coordinates set numerical flag 8 (possibly with input flag 1); undefined nonfinite\n *   intermediate intersections also set flag 8 instead of emitting nonfinite arcs.\n *\n * @throws Error naming positions, ringOffsets or an option for inconsistent input.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineLoopRemoval': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPULineLoopRemovalOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULineLoopRemovalOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeLineLoopRemovalOnCPU',
        signature:
          'function computeLineLoopRemovalOnCPU( input: CPULineLoopRemovalInput, options: CPULineLoopRemovalOptions ): CPULineLoopRemovalResult',
        supportedInputs: ['input: CPULineLoopRemovalInput', 'options: CPULineLoopRemovalOptions'],
        differences: [
          "/**\n * Nodes open planar curves at crossings, touches and overlap endpoints, then erases loops in\n * traversal order using exact visit keys. A repeat replaces the retained vertex's outgoing source\n * edge. The final vertex inherits its predecessor's source edge, matching the contributor. Lines\n * shorter than two retained vertices disappear; overflow drops whole lines in input order, and a\n * later shorter line can still fit.\n *\n * What differs from the GPU contributor:\n * - An explicit intersectionCapacity binds the noding pair list. Exhaustion suppresses CPU\n *   geometry and sets incompleteNoding=1; the GPU can keep an incomplete prefix instead.\n *   Omitted capacity is uncapped. GPU intermediate binding and successor-scan limits are dropped.\n * - Predicates and visit identities use exact rationals of float32 input coordinates. Coordinates\n *   use float64 and round once at output; the GPU rounds crossings before loop erasure. Distinct\n *   exact crossings that round to the same float32 point remain distinct in the twin. The twin\n *   never reports uncertain predicates.\n * - Unused positions/sourceEdges are zeroed; the GPU leaves them untouched on reused buffers.\n * - Empty coordinate arrays with valid zero offsets are accepted and return zero-filled outputs\n *   with all-zero status; the GPU line-split delegate requires at least one coordinate row.\n * - Inconsistent offsets throw instead of relying on out-of-bounds shader reads. Nonfinite\n *   vertices set incompleteNoding=1, their edges are not intersected, and their original values\n *   are retained in the traversal (NaNs never count as repeated visits).\n *\n * @throws Error naming lines.positions, lines.lineOffsets or vertexCapacity for inconsistent input.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOffsetCurveComponents': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUOffsetCurveComponentsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUOffsetCurveComponentsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeOffsetCurveComponentsOnCPU',
        signature:
          'function computeOffsetCurveComponentsOnCPU( input: CPUOffsetCurveComponentsInput, options: CPUOffsetCurveComponentsOptions ): CPUOffsetCurveComponentsResult',
        supportedInputs: [
          'input: CPUOffsetCurveComponentsInput',
          'options: CPUOffsetCurveComponentsOptions'
        ],
        differences: [
          "/**\n * Nodes raw planar offsets, trims their source-distance bands, cancels duplicate/opposite fragments,\n * and chains components in the contributor's seed/successor order. Preserves its geometric slack,\n * first-match ties, whole-component metadata publication and partial writes.\n *\n * What differs from the GPU contributor:\n * - Arithmetic is float64, rounded once at output; distance-band, coincidence and cut decisions\n *   near tolerance boundaries can differ.\n * - Noding uses exact rational predicates/visit keys on float32 input edges. Optional segmentCapacity\n *   retains a traversal-order prefix; optional intersectionCapacity suppresses geometry when its\n *   pair list is exhausted, since the GPU's spatial-index-dependent partial prefix is not replayed.\n *   Either loss sets incompleteNoding=1 without setting final-output overflow. Omitted budgets are\n *   uncapped. No uncertain-predicate status, 1024-successor scan or GPU binding limit is imposed.\n * - Intersections remain float64 through clipping instead of being rounded in intermediate buffers.\n * - Unused positions/sourceEdges are zeroed; the GPU leaves them untouched on reused buffers.\n * - Empty coordinate arrays with valid zero offsets are accepted. Empty raw arrays return\n *   zero-filled outputs with all-zero status; the GPU noding delegate requires a raw coordinate row.\n * - Inconsistent offsets throw instead of relying on out-of-bounds shader reads. Nonfinite raw\n *   vertices set incompleteNoding=1 and their edges are skipped. Nonfinite original edges are\n *   ignored by distance tests; the GPU does not define their geometric result.\n *\n * @throws Error naming original, raw or an option for inconsistent arrays/properties.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURatioConcaveHull': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPURatioConcaveHullOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPURatioConcaveHullOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeRatioConcaveHullOnCPU',
        signature:
          'function computeRatioConcaveHullOnCPU( positions: Float32Array, options: CPURatioConcaveHullOptions ): CPURatioConcaveHullResult',
        supportedInputs: ['positions: Float32Array', 'options: CPURatioConcaveHullOptions'],
        differences: [
          "/**\n * Erodes the exact Delaunay mesh in simultaneous local-priority rounds. Border candidates have\n * one exposed edge and an interior opposite vertex; optional hole candidates have three interior\n * vertices and are considered only when no border candidate remains. Eligible candidates maximize\n * length at every incident vertex; lowest triangle row wins ties. This is the GPU's definition.\n *\n * What differs from the GPU contributor:\n * - An explicit maximumIterations limits the number of triangle removals, including an entire\n *   simultaneous round or none of it. Omitted budgets allow erosion to finish.\n * - Edge lengths, extrema and thresholds use float64. The ratio is first rounded to float32,\n *   matching the GPU override. Near threshold/tie boundaries, float32 GPU lengths can select\n *   different triangles. Distances that overflow float32 remain finite on the CPU.\n * - Initial triangles come from computeDelaunayTessellationOnCPU, with the same row-order\n *   cocircular rule, no exponent-span cap and exact subnormals. Nonfinite sites cancel the hull.\n * - Unused triangle capacity is zeroed; GPU Delaunay leaves unused triangle rows unwritten.\n *\n * @throws Error naming positions or ratio for inconsistent rows or an invalid ratio.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVectorStraightSkeleton': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUVectorStraightSkeletonOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUVectorStraightSkeletonOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeVectorStraightSkeletonOnCPU',
        signature:
          'function computeVectorStraightSkeletonOnCPU( input: CPUVectorStraightSkeletonInput, options: CPUVectorStraightSkeletonOptions ): CPUVectorStraightSkeletonResult',
        supportedInputs: [
          'input: CPUVectorStraightSkeletonInput',
          'options: CPUVectorStraightSkeletonOptions'
        ],
        differences: [
          "/**\n * Advances linked planar wavefronts through edge collapses, reflex splits and hole merges.\n * Original slot order, shell/hole winding, source position IDs, inclusive offset-before-event\n * capture, tolerance tests, terminal ridges and per-polygon capacity status follow the contributor.\n *\n * What differs from the GPU contributor:\n * - Arithmetic is float64 with one float32 rounding at output. Near tolerance/event ties, topology\n *   and therefore slot IDs and arc order can differ; the GPU's tolerance definitions are retained.\n * - maximumEvents sizes the fixed offset-slot layout and limits event work at the same point\n *   as the contributor, after offset capture and before the next arc/event emission.\n * - A repeated active wavefront at the same event time terminates with numerical flag 8,\n *   retaining emitted arcs and the captured offset. The GPU has no repeat-state check and can\n *   instead stop at its work budget with flag 4.\n * - The 1024-vertex and 4096-event ceilings and the 3e38 event-time sentinel are removed.\n *   maximumVertices still bounds initial vertices because it participates in the output layout.\n * - Unwritten float/source rows are zeroed; the GPU leaves them untouched on reused buffers.\n * - Nonfinite input coordinates set input flag 1. Nonfinite motion sets numerical flag 8;\n *   the GPU's result for nonfinite intermediate arithmetic is undefined.\n * - Malformed offset hierarchies throw an Error naming the array; the GPU does not validate them.\n *\n * @throws Error naming an argument when arrays or options are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPopulationSupportAlignment': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUPopulationSupportAlignmentOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUPopulationSupportAlignmentOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePopulationSupportAlignmentOnCPU',
        signature:
          'function computePopulationSupportAlignmentOnCPU( input: CPUPopulationSupportAlignmentInput ): CPUPopulationSupportAlignmentResult',
        supportedInputs: ['input: CPUPopulationSupportAlignmentInput'],
        differences: [
          '/**\n * Exact ID alignment in O(sourceCount + targetCount) expected time using maps. Population values\n * are copied, including NaNs, negative values and infinities; only IDs determine alignment.\n * All uint32 IDs are valid. Duplicate source IDs invalidate any target naming them. A repeated\n * target with no source has code 1; all other ambiguous targets have code 2. Excluded source\n * rows are counted individually, including duplicates whose ID never appears among the targets.\n *\n * What differs from the GPU contributor:\n * - GPU storage-binding limits and sorting scratch do not apply to CPU maps.\n * - The function checks source/population lengths before joining. Otherwise outputs, status\n *   words and row order are identical; no floating-point arithmetic is needed.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUContinuousSpatialSample': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUContinuousSpatialSampleOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUContinuousSpatialSampleOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeContinuousSpatialSampleOnCPU',
        signature:
          'function computeContinuousSpatialSampleOnCPU( input: CPUContinuousSpatialSampleInput, options: CPUContinuousSpatialSampleOptions ): CPUContinuousSpatialSampleResult',
        supportedInputs: [
          'input: CPUContinuousSpatialSampleInput',
          'options: CPUContinuousSpatialSampleOptions'
        ],
        differences: [
          "/**\n * Enumerates complete random-shift line or square lattices in float64. Triangle edges are\n * inclusive, first triangle wins ties, upper area bounds and the line's final arc length are\n * excluded. Legacy sites appear first, carry zero weight and exclude candidates strictly closer\n * than spacing in the frame's metric. Invalid geometry or legacy sites set flag 1 and emit no\n * rows; capacity truncation sets flag 2 while preserving the full requirement. Unused slots are\n * zero with source/ordinal/role 0xffffffff. A resident replicate overflow invalidates that wave.\n *\n * What differs from the GPU contributor:\n * - Float64 geometry, phases and interpolation round once at output; compare coordinates with\n *   absolute tolerance 2e-5 on unit-scale frames. Design weights match the GPU's float32 literals\n *   exactly for corresponding slots.\n *   Near triangle edges, segment boundaries or legacy exclusion radii, classifications may differ.\n * - All lattice candidates are examined: maximumCandidates is ignored and flag 4 is never set.\n * - GPU-only frame, wave, capacity and legacy-site caps and the coordinate/spacing precision\n *   guard are removed. uint32 slot, candidate-ordinal and required-count layout limits remain.\n * - Finite positive spacing with a finite float64 square and finite bounds are accepted even\n *   outside float32 arithmetic limits; output values may round to zero or infinity. Non-finite\n *   frame or legacy coordinates still yield invalid flag 1, as on GPU.\n *\n * @throws If positions, legacySites or parameters lengths are inconsistent, or a lattice cannot\n * be represented by the uint32 provenance/count layout.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialBalancedSample': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialBalancedSampleOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialBalancedSampleOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialBalancedSampleOnCPU',
        signature:
          'function computeSpatialBalancedSampleOnCPU( input: CPUSpatialBalancedSampleInput, options: CPUSpatialBalancedSampleOptions ): CPUSpatialBalancedSampleResult',
        supportedInputs: [
          'input: CPUSpatialBalancedSampleInput',
          'options: CPUSpatialBalancedSampleOptions'
        ],
        differences: [
          "/**\n * Computes the stratified randomized-quadtree circular-systematic design in float64. All rows\n * must be finite, enclosed by bounds and assigned to an allocation; otherwise every stratum is\n * invalidated. Invalid ticket totals invalidate only their own stratum. Missing output IDs are\n * 0xffffffff and their probabilities and weights are NaN. IDs need not be unique for execution;\n * uniqueness is the caller's design responsibility, as on GPU.\n *\n * What differs from the GPU contributor:\n * - Float64 normalization can assign a different hierarchy cell at a float32 cell boundary;\n *   Philox words, rotations, row-index tie breaking and integer selection are otherwise identical.\n * - Float64 probabilities and reciprocal weights are rounded independently once at output;\n *   compare positive floats with relative tolerance 2e-6.\n * - The GPU-only 4096-row and 256-stratum caps and 128-draw rejection budget are removed.\n *   Rejection continues until accepted, so RNG-exhaustion status 2 is never emitted.\n * - Bounds still round to float32, and the 24-bit stratum IDs and inclusion-ticket total remain\n *   limits of the design. Non-finite positions produce invalid-frame status 1, as on GPU.\n * - Integer sample-rank arithmetic does not wrap at 2^32 on larger CPU frames.\n *\n * @throws If positions, frameIds, strata, inclusionTickets or parameters lengths disagree.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPointPatternEnvelope': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUPointPatternEnvelopeOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUPointPatternEnvelopeOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePointPatternEnvelopeOnCPU',
        signature:
          'function computePointPatternEnvelopeOnCPU( input: CPUPointPatternEnvelopeInput, options: CPUPointPatternEnvelopeOptions ): CPUPointPatternEnvelopeResult',
        supportedInputs: [
          'input: CPUPointPatternEnvelopeInput',
          'options: CPUPointPatternEnvelopeOptions'
        ],
        differences: [
          "/**\n * Pointwise simulation envelopes and global deviation or exchangeable extreme-midrank tests.\n * All curves participate, ties are conservative, and no invalid replicate is discarded.\n *\n * What differs from the GPU contributor:\n * - Both forms compare maximum and integrated deviations with exact dyadic\n *   integer arithmetic on the stored float32 inputs. Displayed statistics remain\n *   float32; use 2e-6 relative tolerance for ordinary CPU/GPU display comparisons.\n * - GPU-only 4096-row, 4096-simulation and quadratic rank-comparison caps are dropped;\n *   packed simulation length must still fit uint32. Sorting computes exact order statistics.\n * - Finite float64 statistics that overflow float32 round to Infinity with status=0, whereas\n *   the GPU's float32 intermediate overflow invalidates every inferential result.\n * - Extreme-rank mode ignores reference and integrationWeights values, exactly as the kernel\n *   does. Maximum deviation ignores integrationWeights; integrated deviation requires positive finite weights.\n *\n * @throws If simulations, reference or integrationWeights have inconsistent lengths.\n */",
          'extreme-rank needs observed/simulations only. maximum-deviation also requires reference and ignores integrationWeights (including zero or absent weights). integrated-squared-deviation also requires positive finite integrationWeights. An omitted simulationCount is inferred only from an exact quotient simulations.length / observed.length of at least two; an explicit count must match. Shape/ownership/binding checks apply only to used vectors; nullModel and fittedParameters remain required.',
          'CPU maximum-deviation and integrated-squared-deviation use exact dyadic arithmetic for exceedance decisions on the stored float32 inputs. The displayed statistic is rounded to float32. With observed [1, 2^-13], nineteen simulations [1, 0], zero reference and unit weights, the exact integrated statistic is 1+2^-26, with zero exceedances and p=0.05. The displayed statistic rounds to 1.',
          'GPU maximum-deviation and integrated-squared-deviation use exact dyadic arithmetic for exceedance decisions on the stored float32 inputs when the float32 intermediates remain finite. The displayed statistic uses float32 arithmetic. With observed [-1], reference [2^-60] and seven simulations [1], the displayed deviations round to 1, with zero exceedances and p=0.125.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryKalman': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectoryKalmanOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectoryKalmanOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectoryKalmanOnCPU',
        signature:
          'function computeTrajectoryKalmanOnCPU( input: CPUTrajectoryKalmanInput, options: CPUTrajectoryKalmanOptions = {} ): CPUTrajectoryKalmanResult',
        supportedInputs: [
          'input: CPUTrajectoryKalmanInput',
          'options: CPUTrajectoryKalmanOptions = {}'
        ],
        differences: [
          '/**\n * Runs constant-velocity Kalman filtering and optional RTS smoothing, retaining binary64 states\n * and covariances throughout both passes. Resets, missing measurements, gap boundaries, covariance\n * layouts and status bits follow GPUTrajectoryKalman, including forward values at singular smooths.\n *\n * What differs from the GPU contributor:\n * - Arithmetic is float64 until the final float32 output, including filtered values used by the\n *   smoother. Compare well-conditioned outputs with relative tolerance 3e-4 (absolute 3e-5 at zero).\n * - Scaled covariance solves avoid GPU intermediate overflow. Invalid/smoothingSingular are set\n *   for failures in float64, so GPU float32 failures can instead produce valid CPU estimates.\n *   A finite binary64 estimate outside float32 range rounds to Infinity at output.\n * - Int64 times are differenced exactly before conversion to float64, rather than float32.\n *   maximumGap=0 means unlimited even for finite time differences above the GPU float32 maximum.\n * - Offsets are clamped to the observation count as on GPU; decreasing clamped offsets throw an\n *   Error naming trackOffsets. Untouched rows outside tracks are zero, as in fresh graph outputs.\n * - No GPU-only limits exist to drop. Smoothing-singular rows retain their forward values, as the\n *   kernels do; an invalid parameter value returns status 4 rather than throwing.\n *\n * @throws Error naming positions, timestamps, trackOffsets or parameters for inconsistent arrays.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryRestriction': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUTrajectoryRestrictionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUTrajectoryRestrictionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeTrajectoryRestrictionOnCPU',
        signature:
          'function computeTrajectoryRestrictionOnCPU( input: CPUTrajectoryRestrictionInput, options: CPUTrajectoryRestrictionOptions = {} ): CPUTrajectoryRestrictionResult',
        supportedInputs: [
          'input: CPUTrajectoryRestrictionInput',
          'options: CPUTrajectoryRestrictionOptions = {}'
        ],
        differences: [
          "/**\n * Restricts planar interpolated tracks to a box, even-odd polygon or continuous pairwise distance.\n * Preserves slot order, endpoint policies, last-fix singletons, gap support, relationship counts,\n * stable interval union and the contributor's three-word compact summary.\n *\n * What differs from the GPU contributor:\n * - Arithmetic and compaction use float64, rounded once at publication. Threshold decisions,\n *   root endpoint inclusion and very narrow polygon pieces can differ near rounding boundaries.\n * - Polygon vertices, static pairs and interval capacities have no GPU shader/array maxima.\n *   Caller-selected slot capacities still drop excess intervals and set capacityExceeded.\n * - Exact orientation uses the existing arbitrary-precision CPU predicate, without a GPU\n *   exponent-span limit. Subnormal coordinates retain their values.\n * - Int64 times are subtracted exactly before conversion to float64, including relative origin\n *   times; the GPU's intermediate float32 word-time table is not reproduced.\n * - Unowned observation rows are initialized to NaN intervals/positions and zero status; the\n *   GPU leaves those rows unwritten. Empty inputs and zero compact capacity are supported.\n *   Offsets are clamped independently; reversed track ranges are empty, as on the GPU.\n *   Overlapping nonempty geometry ranges throw an Error naming trackOffsets, because their GPU\n *   invocations race on shared output slots. Pairwise ranges may overlap.\n * - Nonfinite fixes/parameters follow the GPU invalid/gap branches. Finite values whose\n *   intermediate arithmetic overflows float32 can remain valid on the CPU.\n *\n * @throws Error naming the argument for inconsistent packed arrays or invalid options.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialCrossValidation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialCrossValidationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialCrossValidationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialCrossValidationOnCPU',
        signature:
          'function computeSpatialCrossValidationOnCPU( input: CPUSpatialCrossValidationInput, options: CPUSpatialCrossValidationOptions ): CPUSpatialCrossValidationResult',
        supportedInputs: [
          'input: CPUSpatialCrossValidationInput',
          'options: CPUSpatialCrossValidationOptions'
        ],
        differences: [
          "/**\n * Computes seeded block folds or the groups, environmental, NNDM and kNNDM designs, then excludes\n * training rows at distance <= the buffer radius of a testing row. Only finite positions with a\n * nonzero mask are usable. Environmental rows additionally require complete finite features;\n * NNDM/kNNDM require every input and prediction row usable, otherwise design status is 1.\n * Diagnostics retain [status, iterations, objective, selected candidate] and NNDM radii. As on\n * GPU, groups and NNDM leave the objective/candidate NaN, environmental leaves candidate NaN,\n * and failed designs publish missing folds. Ties use the first row/cluster/candidate.\n *\n * What differs from the GPU contributor:\n * - Float64 distances, standardization and objectives round only when diagnostics are written.\n *   Near cluster ties, fold choices can differ; diagnostics use relative tolerance\n *   2e-5 (plus 2e-6 absolute for zero objectives).\n * - GPU-only row, fold, feature, prediction, cluster and candidate caps are removed; uint32\n *   output-length limits and a 2^24 candidate-count limit remain so float32 diagnostics preserve\n *   every selectable index exactly. Candidate count still defines the supplied/generated ladder.\n * - Environmental clustering honors explicit maximumIterations; exhausted budgets emit design\n *   status 3 and missing folds. Without a budget, CPU clustering retains convergence as its\n *   default, including generated kNNDM restarts; the GPU has a bounded default of 100 steps.\n * - Float64 accepts finite distances/features beyond the GPU's arithmetic range and removes\n *   the GPU's 1e18 buffer/maximum-distance bound. Non-finite rows keep the GPU's missing rules.\n * - An exhausted NNDM neighborhood uses the GPU's finite no-distance sentinel in diagnostics;\n *   internal comparisons use infinity rather than clamping finite distances to that sentinel.\n * - Failed NNDM designs return deterministic zero radii. The GPU returns before writing those\n *   fields, so a reused diagnostics view can retain radii from an earlier successful run.\n *\n * @throws If positions, blocks, mask, features, predictionPositions or candidates are inconsistent.\n */",
          'Planar radius membership uses inclusive Euclidean distance on stored coordinates, retaining the comparison residual rather than deciding from a rounded norm. This rule applies to track matching, location connectivity and spatial-validation buffers. Genuine arithmetic differences remain for displayed distances, interpolated/densified geometry, great-circle arithmetic and non-buffer clustering/objective ties.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNestedInterpolationValidation': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUNestedInterpolationValidationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUNestedInterpolationValidationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeNestedInterpolationValidationOnCPU',
        signature:
          'function computeNestedInterpolationValidationOnCPU( input: CPUNestedInterpolationValidationInput, options: CPUNestedInterpolationValidationOptions ): CPUNestedInterpolationValidationResult',
        supportedInputs: [
          'input: CPUNestedInterpolationValidationInput',
          'options: CPUNestedInterpolationValidationOptions'
        ],
        differences: [
          "/**\n * Selects the first candidate with minimum complete inner held-out MSE, refits on the complete\n * outer training mask, and publishes only outer testing predictions. Overlap rejects the entire\n * outer fold. Scores are [MSE, scored, missing]; metrics are [scored, missing, MSE, MAE, bias, R²].\n * Status bits are 1 overlap, 2 no selected candidate, 4 missing held-out predictions, 8 empty\n * held-out set, 16 nonfinite metric arithmetic. R² is NaN for constant observed values.\n *\n * What differs from the GPU contributor:\n * - All fits, scores and metrics use float64, rounding only the five returned arrays. Near score,\n *   neighbor or rank thresholds selection/status can differ; compare finite outputs with relative\n *   tolerance 5e-4 and absolute tolerance 2e-5 for values near zero.\n * - GPU work caps (4096 tiled / 256 reference rows, 16 outer folds/candidates, 8 inner folds, 32 neighbors) are removed.\n *   The neighborCount hyperparameter still selects the nearest requested number, with row-index\n *   ties. uint32 output-length limits and the 2^24 exact float32 count limit remain.\n * - Matern uses the exact closed form rather than the GPU's short-lag truncated Taylor series;\n *   thin-plate-spline uses log(r) without the GPU's 1e-30 floor. The GPU's equilibrated pivot\n *   and variance rejection tolerances are retained.\n * - Float64 permits scores/distances beyond float32 range, without the GPU's finite best-score\n *   sentinel or 1e18 buffer cap. Status 16 concerns float64 arithmetic; final float32 overflow\n *   can produce infinity. Nonfinite coordinates/responses are omitted as on GPU; a nonfinite\n *   solve produces a missing prediction. Subnormals are preserved rather than flushed to zero.\n *\n * @throws If positions, response, training, testing or innerBlocks have inconsistent lengths,\n * or a numeric option or candidate model is invalid.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialInteraction': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSpatialInteractionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSpatialInteractionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialInteractionOnCPU',
        signature:
          'function computeSpatialInteractionOnCPU( input: CPUSpatialInteractionInput, options: CPUSpatialInteractionOptions ): CPUSpatialInteractionResult',
        supportedInputs: [
          'input: CPUSpatialInteractionInput',
          'options: CPUSpatialInteractionOptions'
        ],
        differences: [
          "/**\n * Calibrates the four Poisson gravity constraints, with log-domain proportional fitting and\n * float64 candidate selection. Missing pairs are excluded before reading IDs/flow/covariate;\n * structural-zero covariates are ignored, as on the GPU.\n *\n * What differs from the GPU contributor:\n * - Float64 log-sum-exp balancing removes the GPU's exponent-span >80 rejection (candidate code 4\n *   is never set). Predictions outside the float64 exponential range become zero internally;\n *   their finite log means still contribute to deviance. Final float32 rounding can underflow\n *   additional predictions. Subnormals are retained internally.\n * - Pair/group/candidate caps and GPU upper limits on iteration budgets are dropped. Explicit\n *   maximumIterations and maximumCalibrationIterations still bind the solve. Without a budget,\n *   balancing continues to tolerance or float64 stagnation, and ordinary bounded calibration\n *   continues to coefficientTolerance or an unrepresentable narrower bracket. Recovery when\n *   both probes fail preserves the interval and defaults to 32 exploratory rounds; an explicit\n *   maximumCalibrationIterations overrides that limit. Candidate code 1/status 3\n *   describe balancing exhaustion or stagnation; status 4 describes a coefficient bracket that\n *   did not converge. Iteration counts may differ from the GPU at tolerance boundaries.\n * - Plain numeric props stay float64. Coefficients need not be representable as float32 shader\n *   literals. A non-finite float64 coefficient-times-covariate product reports invalid input.\n * - Output arithmetic is rounded once. Compare predictions/deviance with 5e-4 relative tolerance;\n *   marginal errors/residuals near numerical zero need absolute tolerance relative to mass.\n *   Grid ties choose the first candidate; ordinary bounded ties retain the lower bracket and\n *   exploratory ties retain the incumbent, as on the GPU.\n * - A candidate with finite float64 deviance remains successful even if publishing that deviance\n *   as float32 gives Infinity. The GPU's finite-deviance gate may instead reject it (candidate\n *   code 1/status 3); the target-device outcome is covered by a pending headless case.\n * - The pure function accepts empty pair arrays (status 1); an operation needs positive pairCount.\n *\n * @throws If origins, destinations, flows, covariate or pairState lengths disagree.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSphericalArcOverlay': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'CPUSphericalArcOverlayOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPUSphericalArcOverlayOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSphericalArcOverlayOnCPU',
        signature:
          'function computeSphericalArcOverlayOnCPU( input: CPUSphericalArcOverlayInput, options: CPUSphericalArcOverlayOptions ): CPUSphericalArcOverlayResult',
        supportedInputs: [
          'input: CPUSphericalArcOverlayInput',
          'options: CPUSphericalArcOverlayOptions'
        ],
        differences: [
          "/**\n * Nodes minor great-circle arcs, classifies every piece on both sides, then chases oriented rings\n * with the result interior on the left. Input and ring iteration order and maximum-left-turn\n * junction selection match the contributor. Failure empties all geometry, retaining the required\n * segment count accumulated before failure and the Boolean reference membership.\n *\n * What differs from the GPU contributor:\n * - Geometry and predicates use float64, rounded only when output positions are stored. Results\n *   near tolerance thresholds or tied junctions can differ from float32; compare positions with\n *   absolute tolerance 3e-6 on well-separated inputs. Integer fields agree on those inputs.\n * - The GPU-only 128-input, 130-split-point and 1024-output limits are dropped. Both forms retain\n *   every unique segment before output capacity is checked, so counts[2] is independent of the\n *   caller's output capacity, including on coincident boundaries.\n * - Tolerance-based invalid/unresolved rules and the seven reference-anchored containment paths\n *   are retained as the contributor's definition of numerical separation. Status bits 1, 2 and 4\n *   retain their meanings; a capacity failure publishes zero vertices and rings, not a prefix.\n * - Malformed ring ranges and non-unit/non-finite coordinates return status 1 as on GPU;\n *   inconsistent packed positions or missing offset arrays throw an Error naming the argument.\n * - Unknown runtime operation values throw; the GPU contributor falls through to symmetric-difference.\n *\n * @throws For inconsistent packed arrays, invalid options, or an unrepresentable uint32 count.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineLengthPerPolygon': {
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeLineLengthPerPolygonOnCPU',
        signature:
          'function computeLineLengthPerPolygonOnCPU( input: { lines: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; lineWeights?: Float32Array; }, options: CPULineLengthPerPolygonOptions )',
        supportedInputs: [
          'input: {\n    lines: CPUSpatialJoinGeometry;\n    polygons: CPUSpatialJoinGeometry;\n    lineWeights?: Float32Array;\n  }',
          'options: CPULineLengthPerPolygonOptions'
        ],
        differences: [
          '/**\n * Sums all clipped segment lengths of every line part, weighted once by the owning feature.\n * What differs from the GPU contributor:\n * - Planar float64 clipping and accumulation, rounded once at output. Near-edge cut fractions\n *   and totals can differ from float32. Both forms include boundary-coincident pieces.\n * - Offset hierarchies are checked on CPU; malformed layouts throw. Nonfinite line edges skip.\n * - Empty coordinate arrays with valid offsets are accepted, without GPU storage-binding limits.\n * - Pair and crossing budgets are preserved; overflow reports partial results, never completeness.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addPointsInPolygonsChoroplethRecipe': {
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePointsInPolygonsChoroplethOnCPU',
        signature:
          'function computePointsInPolygonsChoroplethOnCPU( input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; values?: Float32Array}, options: CPUPointsInPolygonsChoroplethOptions )',
        supportedInputs: [
          'input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; values?: Float32Array}',
          'options: CPUPointsInPolygonsChoroplethOptions'
        ],
        differences: [
          '/**\n * Aggregates one value per feature in every matched polygon for multipart input, and uses the\n * lowest polygon row for ordinary single-point assignments, preserving recipe behavior.\n * What differs from the GPU recipe:\n * - Packed planar aggregation only, before optional class-break/color and longitude-latitude\n *   preparation stages; those remain composable GPU contributors, not silently omitted options.\n * - Uses the exact CPU point join and float64 sums/means, rounded once at output. CPU orientation\n *   has no dynamic-range uncertainty; malformed polygon hierarchies throw (see the CPU join).\n * - On candidate loss the retained prefix uses unsorted polygon rows. The recipe may spatially\n *   sort larger polygon sets; either form flags incomplete counts/statistics with overflow=1.\n * - Single-point zonal and group backends share the same CPU aggregation arithmetic.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGaussianPredictiveScores': {
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeGaussianPredictiveScoresOnCPU',
        signature:
          'function computeGaussianPredictiveScoresOnCPU( input: CPUGaussianPredictiveScoresInput, options: CPUGaussianPredictiveScoresOptions = {} ): CPUGaussianPredictiveScoresResult',
        supportedInputs: [
          'input: CPUGaussianPredictiveScoresInput',
          'options: CPUGaussianPredictiveScoresOptions = {}'
        ],
        differences: [
          "/**\n * Binary64 Gaussian score evaluation with one float32 output rounding. Compile-time\n * alpha and threshold use the contributor's float32 settings. Subnormal input values\n * may flush on GPU; the CPU retains them. No GPU dispatch-size cap applies on CPU.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPredicateJoin': {
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeSpatialPredicateJoinOnCPU',
        signature:
          'function computeSpatialPredicateJoinOnCPU( left: CPUSpatialJoinGeometry, right: CPUSpatialJoinGeometry, options: CPUFeatureJoinOptions )',
        supportedInputs: [
          'left: CPUSpatialJoinGeometry',
          'right: CPUSpatialJoinGeometry',
          'options: CPUFeatureJoinOptions'
        ],
        differences: [
          '/**\n * Planar feature-level predicate join. Candidate slots count logical feature pairs in row order.\n * What differs from the GPU contributor: rational CPU topology has no orientation uncertainty;\n * distances compute in float64. Empty features have no candidates. In anti mode leftRows are\n * unmatched left rows and rightRows retain the missing-row sentinel.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNearestFeatureJoin': {
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeNearestFeatureJoinOnCPU',
        signature:
          'function computeNearestFeatureJoinOnCPU( queries: CPUSpatialJoinGeometry, targets: CPUSpatialJoinGeometry )',
        supportedInputs: ['queries: CPUSpatialJoinGeometry', 'targets: CPUSpatialJoinGeometry'],
        differences: [
          '/**\n * Exact one-neighbor planar feature search, with minimum distance over all usable parts.\n * Nonfinite point parts and edges are skipped; no usable edge means a missing neighbor.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUBufferSelection': {
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computeBufferSelectionOnCPU',
        signature:
          "function computeBufferSelectionOnCPU( input: CPUBufferSelectionInput, options: CPUBufferSelectionOptions ): Record<keyof BufferSelectionOperationProps['output'], Uint32Array | Float32Array>",
        supportedInputs: ['input: CPUBufferSelectionInput', 'options: CPUBufferSelectionOptions'],
        differences: [
          '/**\n * Selects a feature when any point part is within the inclusive distance of a reference point.\n * What differs from the GPU contributor:\n * - Distances use float64 internally, rounded once at output; last-ulp radius decisions can differ.\n * - This packed planar adapter supports point and segment references; use GPUBufferSelection directly for great-circle references.\n * - Unused compact ID slots are zero. GPU callers must only read the prefix selected by count.\n * - Invalid offset layouts throw on CPU. Nonfinite points are ignored, like GPU query probes.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPointInPolygonJoin': {
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePointInPolygonJoinOnCPU',
        signature:
          "function computePointInPolygonJoinOnCPU( input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry}, options: CPUPointInPolygonJoinOptions ): Record<keyof PointInPolygonJoinOperationProps['output'], Uint32Array>",
        supportedInputs: [
          'input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry}',
          'options: CPUPointInPolygonJoinOptions'
        ],
        differences: [
          '/**\n * Joins packed points/multipoints to polygons using exact binary64 integer orientation signs.\n * What differs from the GPU contributor:\n * - Exact CPU predicates have no dynamic-range uncertainty; GPU may\n *   refuse a candidate whose edge orientation spans more than 200 binary orders of magnitude.\n * - Malformed layouts throw. Nonfinite polygon candidates increment uncertainCount and do not\n *   match; results for other polygons survive, as on the GPU.\n * - Unused compact slots are zero, and are outside the readable count prefix.\n * - Unprepared planar geometry only; sorted/prepared GPU traversals remain contributor options.\n * Part candidates are visited in input-part order, then polygon row order, exactly as the\n * unsorted complete binary BVH. A cutoff flags the owner even when an earlier part matched.\n */'
        ]
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'computePointInPolygonCountsOnCPU',
        signature:
          'function computePointInPolygonCountsOnCPU(input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry}, options: CPUPointInPolygonJoinOptions): CPUPointInPolygonCountsResult',
        supportedInputs: ['packed points/multipoints and polygons', 'CPUPointInPolygonJoinOptions'],
        differences: [
          'Counts and assignments from sparse matches without a feature-by-polygon membership matrix. Uses the same exact binary64 predicates and capacity/loss rules as computePointInPolygonJoinOnCPU; omits featurePolygonMembership.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkTimeRouting': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkTimeRoutingOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkTimeRoutingOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkTimeRoutingOnCPU',
        signature:
          'function computeNetworkTimeRoutingOnCPU(input: CPUNetworkTimeRoutingInput): CPUNetworkTimeRoutingResult',
        supportedInputs: ['input: CPUNetworkTimeRoutingInput'],
        differences: [
          '/**\n * Computes FIFO earliest arrivals and predecessor transitions in float64, then rounds once.\n * Summary is [valid, converged, targetState, float32 arrival bits]; missing keys are 0xffffffff.\n *\n * What differs from the GPU contributor:\n * - GPU state, transition and sample caps are dropped. maximumRelaxations bounds transition\n *   sweeps (default state count), including the final unchanged sweep that proves convergence.\n * - Float64 arithmetic can distinguish routes tied in float32 and accept arrivals that overflow\n *   shader arithmetic. Float32 output overflow becomes Infinity without clearing valid.\n * - Strict improvement in transition scan order and lowest-state target ties match the GPU.\n *\n * @throws If transitions, costs, endpoints or departure have inconsistent array lengths.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkPointRouting': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkPointRoutingOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkPointRoutingOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkPointRoutingOnCPU',
        signature:
          "function computeNetworkPointRoutingOnCPU( input: CPUNetworkPointRoutingInput, options: {mode?: 'reference' | 'scalable'; maximumRelaxations?: number} = {} ): CPUNetworkPointRoutingResult",
        supportedInputs: [
          'input: CPUNetworkPointRoutingInput',
          "options: {mode?: 'reference' | 'scalable'; maximumRelaxations?: number} = {}"
        ],
        differences: [
          "/**\n * Routes between positions on directed edges. Turn order resolves equal-cost discoveries, with\n * strict improvements only, as in the contributor's explicit `mode: 'reference'` route. Same-edge forward travel competes\n * with legal returns, including when the target fraction precedes the source fraction.\n *\n * What differs from the GPU contributor:\n * - The default GPU mode is scalable CSR: full-edge predecessors use reachability's smallest-parent,\n *   acyclic plateau rule and terminal partial-edge ties use the smallest cycle-safe parent.\n *   The CPU retains turn-order ties for either mode; pass `mode: 'reference'` to the GPU operation\n *   when comparing those ties. With an explicit budget, mode selects turn scans or synchronous hops.\n * - Float64 turn relaxation and partial-edge arithmetic precede one float32 output rounding;\n *   near ties, subnormals and paths overflowing float32 may produce different costs or parents.\n * - A full-edge sum overflowing float32 does not trigger the GPU's status-1 early return. The\n *   CPU continues in float64 and can publish a finite partial-target cost with status 0 and\n *   predecessors where the GPU leaves Infinity and status 1. Scalable mode clears invalid-query\n *   predecessors; reference mode can retain partially written predecessors.\n * - Without an explicit budget, relaxation continues to a fixpoint. An explicit maximumRelaxations\n *   binds reference turn scans or scalable rounds of up to 64 synchronous hops. Exhaustion sets\n *   status 2 and publishes the incumbent; zero permits only direct same-edge travel.\n * - Negative or non-finite base costs close that edge: turns touching it are skipped and queries\n *   with a closed endpoint get status 1. Other queries remain usable.\n * - Invalid edge IDs, fractions and turn endpoints retain the GPU's per-query status result;\n *   inconsistent packed array lengths throw instead of being interpreted as partial rows.\n *\n * @throws Error naming turns, costs, queries or fractions for an inconsistent packed array.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkFlow': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkFlowOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkFlowOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkFlowOnCPU',
        signature:
          'function computeNetworkFlowOnCPU( input: CPUNetworkFlowInput, options: CPUNetworkFlowOptions ): CPUNetworkFlowResult',
        supportedInputs: ['input: CPUNetworkFlowInput', 'options: CPUNetworkFlowOptions'],
        differences: [
          '/**\n * Integral flow CPU twin. Reference and successive-shortest-path solves scan residual arcs in\n * input order (forward then reverse per edge), with strict-improvement predecessor updates.\n * Parallel phase solves use bounded preflow discharges and excess returns. Parallel, antiparallel\n * and self-loop edges retain their slots. complete=1 means the solve finished, including an\n * infeasible finite request (unserved > 0). cutCertified is set only on residual exhaustion.\n * Invalid endpoints/source/sink, capacity totals or supplied CSR connectivity return zero outputs\n * and valid=0. CSR slots must match the edge-list source row and destination at every slot.\n * Optional pathCapacity enables final-flow decomposition: FIFO BFS, ascending outgoing edge rows,\n * first predecessor wins, one bottleneck volume per path. Paths retain expanded super arcs.\n *\n * What differs from the GPU contributor:\n * - Solver selection matches the contributor. Reference and costed successive-shortest-path use\n *   maximumAugmentations; uncosted parallel uses maximumRelabels; primal-dual uses maximumCostPhases.\n *   Parallel phases use sink-rooted labels, bounded preflow discharges and excess returns, then a\n *   separate source-rooted cut search. CPU discharges scan input edge rows; GPU atomic ordering may\n *   choose another equal-cost branch. Explicit maximumRelaxationRounds bounds synchronous searches,\n *   including the final unchanged sweep. Omitted CPU budgets are uncapped. GPU private-array caps\n *   are dropped for the solve; optional decomposition keeps the 128-node/1024-edge layout limits.\n *   Decomposition and padding agree exactly for the same final edge flow.\n * - Total capacity may reach 0xffffffff (the uint32 output limit), rather than 0x00ffffff;\n *   original costs may use the entire uint32 range rather than 0..32767.\n * - Integral distances and the objective use exact integers (BigInt), avoiding signed-distance\n *   overflow. The summary cost is rounded once directly to float32; exactCost is [low32, high32].\n *   The GPU accumulates a float32 summary cost, compared with relative tolerance 2e-5.\n *   Aggregate value and objective agree when both complete within their numerical limits; parallel\n *   path allocation, phase progress and diagnostics can differ with GPU scheduling.\n * - Both operation forms and the pure function support zero edges and empty edge-flow outputs.\n *\n * @throws If edges/capacities/costs/parameters lengths are inconsistent or nodeCount is invalid.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkLouvain': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkLouvainOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkLouvainOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkLouvainOnCPU',
        signature:
          'function computeNetworkLouvainOnCPU( input: CPUNetworkLouvainInput, options: CPUNetworkLouvainOptions ): CPUNetworkLouvainResult & {status: ReturnType<typeof decodeNetworkLouvainStatus>}',
        supportedInputs: ['input: CPUNetworkLouvainInput', 'options: CPUNetworkLouvainOptions'],
        differences: [
          "/**\n * Sequential Louvain local moves and implicit coarsening, with full float64 objective evaluation.\n * Only existing neighboring communities are candidates, visited by ascending community ID. A\n * move must improve the best objective by more than the supplied tolerance. A seeded visit order\n * draws the same Fisher-Yates permutation once from the contributor's registered Philox stream.\n * Invalid parameters/weights return identity labels, NaN trace and valid=0.\n *\n * What differs from the GPU contributor:\n * - Sparse CSR uses the GPU's parallel-proposal/reservation schedule in float64. It drops the\n *   GPU pair-key/scratch/node-budget caps; near-tie labels and stopping rounds can differ.\n *   Sparse seed uses XOR priority, whereas the legacy dense path uses Fisher-Yates visitation.\n *   Float64 reciprocal sums can also differ near the 1e-6 undirected symmetry threshold.\n * - Float64 objective evaluation can choose different moves at near ties or tolerance boundaries.\n *   Well-separated partitions compare exactly; their objective traces use relative tolerance 2e-5.\n *   The GPU can leave a finite cancellation residual where the float64 objective is zero. For\n *   gamma=1, the coarsening comparison allows absolute error 8 * 2^-23 at zero: eight float32\n *   spacings at unit magnitude, the scale of the normalized internal-weight and penalty terms.\n *   Nonzero trace entries retain the relative tolerance; the CPU keeps its exact zero.\n * - GPU-only 32-node, 32-level and 64-sweep caps are dropped. The requested maximumLevels and\n *   maximumIterations (defaults 8 and 16) remain because they define the trace length and the\n *   documented nonconvergence status. This remains heuristic Louvain, not global optimization.\n * - Float32 intermediate overflow does not invalidate finite float64 work; output overflow becomes\n *   Infinity. Subnormal weights are retained.\n *\n * @throws If weights/parameters lengths are inconsistent or shape/seed/budgets are invalid.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkRationalAccess': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkRationalAccessOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkRationalAccessOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkRationalAccessOnCPU',
        signature:
          'function computeNetworkRationalAccessOnCPU( input: CPUNetworkRationalAccessInput, options: CPUNetworkRationalAccessOptions = {} ): CPUNetworkRationalAccessResult',
        supportedInputs: [
          'input: CPUNetworkRationalAccessInput',
          'options: CPUNetworkRationalAccessOptions = {}'
        ],
        differences: [
          "/**\n * Runs simultaneous rational-agent equalization, retaining earliest-facility ties, integral\n * flooring and the early inflow cap's ties-to-even rounding. Unreachable or zero-demand origins\n * retain NaN scores. Invalid numeric data returns valid=0 and the partial initialization/results\n * reached before rejection, exactly in the contributor's order.\n *\n * What differs from the GPU contributor:\n * - Arithmetic uses float64 and rounds once per output. Near integer-shift or facility-cost ties,\n *   allocations and convergence cycles can differ. Well-conditioned scores/gaps use relative\n *   tolerance 2e-5; integral assignments compare exactly when the selected moves agree.\n * - GPU-only 64-origin, 32-facility and 4096-cycle caps are dropped. maximumCycles (default 150)\n *   and initialLimitCycles (default 20) remain algorithm controls, including nonconverged stalled\n *   states; the total-demand precision limit of 2^24 is retained for the float32 assignment layout.\n * - Float32 intermediate overflow/underflow does not invalidate otherwise finite float64 work;\n *   output overflow becomes Infinity. Subnormal values are retained.\n *\n * @throws If costs or parameters have inconsistent lengths, demand/supply are empty, or cycle\n * controls are not nonnegative integers. Invalid numeric rows instead return valid=0.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUAlternativeRoutes': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUAlternativeRoutesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUAlternativeRoutesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeAlternativeRoutesOnCPU',
        signature:
          'function computeAlternativeRoutesOnCPU( input: CPUAlternativeRoutesInput, options: CPUAlternativeRoutesOptions ): CPUAlternativeRoutesResult',
        supportedInputs: [
          'input: CPUAlternativeRoutesInput',
          'options: CPUAlternativeRoutesOptions'
        ],
        differences: [
          "/**\n * Finds the exact top-k simple edge paths with an unbounded binary min heap. Repeated vertices\n * are prohibited, parallel edges are distinct, and equal costs use lexicographic CSR edge IDs.\n *\n * What differs from the GPU contributor:\n * - Float64 path costs are rounded once on output; near ties, subnormals and paths whose float32\n *   sums overflow can change route order and membership.\n * - Explicit expansion/frontier budgets bind as on GPU, retaining already published paths and\n *   setting complete=0 on exhaustion. Omitted budgets are unbounded. GPU node/route/device\n *   shape caps are dropped.\n * - This exact search remains exponential in the worst case and can require substantial CPU memory.\n * - Invalid endpoints are empty certified queries; invalid neighbors/weights are impassable,\n *   and offset values retain the GPU's per-row clamping interpretation.\n *\n * @throws Error naming topology, sources or targets for inconsistent array lengths.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPURadiationFlowModel': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPURadiationFlowModelOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPURadiationFlowModelOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeRadiationFlowModelOnCPU',
        signature:
          'function computeRadiationFlowModelOnCPU(input: CPURadiationFlowModelInput): CPURadiationFlowModelResult',
        supportedInputs: ['input: CPURadiationFlowModelInput'],
        differences: [
          '/**\n * Computes radiation fluxes using one distance sort and grouped population prefix per origin.\n * Returns float32 flows at row origin * zoneCount + destination, with a zero diagonal.\n * Equal distances are processed together, preserving the strict intervening-distance rule.\n *\n * What differs from the GPU contributor:\n * - Distances, population sums and fluxes use float64; each output is rounded once. Near ties\n *   may therefore differ from float32 distance ties on the GPU.\n * - The GPU-only 1024-zone work cap is dropped; sorting costs O(zones squared log zones).\n *\n * @throws If positions, populations or outflows have inconsistent lengths.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUMapMatching': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUMapMatchingOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUMapMatchingOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeMapMatchingOnCPU',
        signature:
          'function computeMapMatchingOnCPU( input: CPUMapMatchingInput, options: CPUMapMatchingOptions ): CPUMapMatchingResult',
        supportedInputs: ['input: CPUMapMatchingInput', 'options: CPUMapMatchingOptions'],
        differences: [
          "/**\n * Newson-Krumm matching with exact geometric candidates, optionally bounded A* and per-track\n * Viterbi decoding. Distances and likelihoods remain float64 until output. Equal candidate\n * distances prefer the lowest edge row; equal Viterbi scores prefer the earliest candidate.\n * The finite score sentinel and cutoff of the contributor are retained: only scores strictly\n * above its float32 -1e29 boundary are normalized and eligible for backward selection.\n *\n * What differs from the GPU contributor:\n * - Float64 intermediates and one float32 output rounding replace float32 arithmetic. Near ties,\n *   radius/bound thresholds and extreme magnitudes can change selected paths. Compare ordinary\n *   metric outputs with relative tolerance 5e-5 (absolute 2e-5 near zero); integers compare exactly.\n *   Geographic snapped positions are compared in spatial units using endpoint/output float32\n *   ULPs and a segment-length projection-error allowance, not absolute longitude magnitudes.\n * - All valid edges are searched, without the grid's 16-cell span or entryCapacity truncation.\n *   The searchRadius clamp to twice the cell size in metric units is retained. CPU candidateCount\n *   may exceed 8, but the requested candidateCount and default 8 still define the model.\n * - An explicit routeNodeBudget limits fallback and recovery searches as on GPU, including\n *   candidateOverflow, routeOverflow and budget breaks (2). Omitted, CPU searches remain uncapped.\n *   A complete routeTable within its cost limit certifies scoring transitions without a search.\n *   Grid candidate loss is absent on CPU. Fixed matched outputs always have overflow=0.\n * - Traversed counts remain unclamped, as on GPU; traversedOverflow reports output truncation\n *   only; traversedCandidateOverflow reports incomplete recovery and inserts a NONE separator\n *   before the destination edge. Unwritten slots are zero on CPU (GPU leaves them untouched).\n *   This inherited layout is a section 8 contract departure, not compact output: counts is the\n *   per-track requirement and no separate requiredCount exists. Correcting it needs a coordinated\n *   contributor/twin API change; clamping only the twin would violate the same-graph-values rule.\n * - Malformed packed lengths, decreasing CSR offsets and out-of-range track offsets throw.\n *   Monotonic network offsets are clamped to edge capacity as on GPU; invalid targets and unused\n *   edge slots are ignored. Points outside the track ranges are deterministically unmatched.\n * - Nonfinite coordinates or parameters throw; their GPU results are undefined. Subnormal and\n *   very large finite inputs are evaluated in float64; final float32 outputs may overflow.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkCostMatrix': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkCostMatrixOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkCostMatrixOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkCostMatrixOnCPU',
        signature:
          'function computeNetworkCostMatrixOnCPU( input: CPUNetworkCostMatrixInput, options: CPUNetworkCostMatrixOptions ): CPUNetworkCostMatrixResult',
        supportedInputs: [
          'input: CPUNetworkCostMatrixInput',
          'options: CPUNetworkCostMatrixOptions'
        ],
        differences: [
          "/**\n * Computes a multi-start shortest-path search per row, gathering targets when requested.\n *\n * What differs from the GPU contributor:\n * - Float64 paths are rounded only at output; near ties, cutoffs, subnormals and float32-overflow\n *   paths can differ. NaN costLimit disables the cutoff, matching the GPU comparison rule.\n * - Omitted maximumIterations searches to completion. An explicit round budget permits at most\n *   maximumIterations rounds with localIterations hops and a 256-slot local queue per chunk\n *   (localIterations defaults to 16). Overflow waits for the next round. GPU workgroup\n *   ordering can change partial labels with competing discoveries. Converged is 1 only\n *   when every row's frontier empties; iterationLimitReached is its complement.\n * - GPU lane, command-node and device limits are dropped.\n * - CPU memory is one network search plus the output, without lane-expanded scratch.\n *\n * @throws Error naming inconsistent startCosts, startRows, targetNodes, costLimit or topology arrays.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkPathExtraction': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkPathExtractionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkPathExtractionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkPathExtractionOnCPU',
        signature:
          'function computeNetworkPathExtractionOnCPU( input: CPUNetworkPathExtractionInput ): CPUNetworkPathExtractionResult',
        supportedInputs: ['input: CPUNetworkPathExtractionInput'],
        differences: [
          '/**\n * Extracts complete root-to-target paths with explicit cycle detection, packs them in target\n * order, and optionally resolves links to CSR slots or stable IDs. Invalid/inactive/unreached\n * targets and broken or cyclic chains have empty paths. A NONE root requires complete predecessors.\n * Edge extraction requires predecessorEdges from the search and resolves them through validated\n * CSR slots. Counts clamp to capacity; offsets remain unclamped.\n *\n * What differs from the GPU contributor:\n * - Cycles are detected exactly instead of waiting for the walk budget; they report candidate loss.\n *   maximumPathLength binds both forms (default 1024); CPU accepts budgets above the GPU 65536 cap.\n *   Only final output capacity sets overflow; candidateOverflow reports bounded walk loss.\n * - The safe bound is activeTargetUpperBound × min(nodeCount, maximumPathLength). Larger\n *   batches are admitted but actual totals above 4294967295 reject before publication.\n * - Unused node ID capacity is zero-filled; the GPU leaves rows beyond count unwritten. CPU edge\n *   padding is 0xffffffff for nonempty target tables and zero for empty tables; the GPU initializes\n *   edge padding only for nonempty target tables. Only rows before count are compact results.\n *\n * @throws If costs, nodeIds, targetCount, predecessorsComplete or edge arrays are inconsistent.\n */',
          'The safe bound is activeTargetUpperBound * min(nodeCount, maximumPathLength). Larger bounds are admitted with checked actual totals above 4294967295 rejected before publication. Numeric targetCount must lie in [0, target capacity] and rejects outside that range; the live uint32 targetCount word is clamped to target capacity. Target capacity is targets.length for the array form and targetCapacity for the named form. An omitted count uses the allocated capacity as the active target prefix; a graph-value count name selects the live prefix and uses targetCapacity as its construction-time upper bound.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkServiceAreas': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkServiceAreasOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkServiceAreasOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkServiceAreasOnCPU',
        signature:
          'function computeNetworkServiceAreasOnCPU( input: CPUNetworkServiceAreasInput, options: CPUNetworkServiceAreasOptions = {} ): CPUNetworkServiceAreasResult',
        supportedInputs: [
          'input: CPUNetworkServiceAreasInput',
          'options: CPUNetworkServiceAreasOptions = {}'
        ],
        differences: [
          '/**\n * Assigns minimum-cost facilities, breaking ties by facility row, and builds same-label predecessors.\n * Overlapping CSR rows reuse slot-indexed tight/tie flags exactly as on the GPU; assignments and\n * predecessors can then follow edges that are not tight for that source or have a different label.\n *\n * What differs from the GPU contributor:\n * - Float64 shortest paths, tight-edge tests and facility sums precede one float32 output rounding.\n *   Near ties, cutoffs, subnormals and float32-overflow paths can change assignments and parents.\n * - Explicit cost, label and tie budgets bound synchronous CPU frontiers; activeIterations\n *   clamps cost and label rounds. Omitted budgets keep exact traversal. GPU scheduling can\n *   change partial labels near a budget boundary; device limits are dropped.\n * - Unused facility aggregates are zero, including a zero-length facility input when aggregates\n *   are omitted from the graph operation. NaN costLimit disables the cutoff, like the GPU.\n *\n * @throws Error naming inconsistent topology, facilities, facilityCosts, facilityCount or scalar arrays.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkCentrality': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkCentralityOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkCentralityOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkCentralityOnCPU',
        signature:
          'function computeNetworkCentralityOnCPU( input: CPUNetworkCentralityInput, options: CPUNetworkCentralityOptions = {} ): CPUNetworkCentralityResult',
        supportedInputs: [
          'input: CPUNetworkCentralityInput',
          'options: CPUNetworkCentralityOptions = {}'
        ],
        differences: [
          '/**\n * Computes outward radius-bounded centrality using Dijkstra and topological Brandes frontiers.\n * Repeated pivots and parallel slots count separately; endpoints and self-loops are excluded.\n * Negative/NaN weights and invalid neighbors are impassable. A negative radius rejects even the\n * pivot; a NaN radius imposes no bound, matching the GPU comparison. Offset bounds are\n * clamped to the edge capacity; a reversed interval is empty, as on the GPU.\n *\n * What differs from the GPU contributor:\n * - Distances, tight-edge tests and dependencies use float64, rounded only at output. Small\n *   positive edges absorbed by float32 can therefore participate in shortest paths on CPU.\n *   Compare well-conditioned centralities with relative tolerance 2e-5.\n * - Explicit maximumIterations bounds search rounds (localIterations defaults to 16, with\n *   256-entry local queues), and independently bounds each dependency phase to one edge per\n *   round. Omitted maximumIterations runs to completion. GPU scheduling can change partial\n *   results for competing discoveries; command-node and lane-storage limits are dropped.\n * - Path counts use scaled float64 mantissas and exponents to avoid overflow; sums are independent\n *   of laneCount. Zero-cycle partial node and edge dependencies retain the GPU frontier convention.\n * - Float32 distance/sum overflow is avoided internally. Subnormals are retained; undefined GPU\n *   arithmetic at extreme magnitudes can differ.\n *\n * @throws If topology arrays, reverseTopology node count or costLimit length are inconsistent.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkCostTable': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkCostTableOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkCostTableOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkCostTableOnCPU',
        signature:
          'function computeNetworkCostTableOnCPU( input: CPUNetworkCostTableInput, options: CPUNetworkCostTableOptions ): CPUNetworkCostTableResult',
        supportedInputs: ['input: CPUNetworkCostTableInput', 'options: CPUNetworkCostTableOptions'],
        differences: [
          '/**\n * Builds cutoff all-pairs CSR rows with ascending targets, including finite diagonal zeros.\n *\n * What differs from the GPU contributor:\n * - Each path addition rounds to float32, as on the GPU, before the inclusive cutoff test. This\n *   deliberately preserves table membership for mixed CPU-table/GPU-matcher graphs rather than\n *   using float64 paths. NaN costLimit disables the cutoff, as in the GPU comparisons.\n * - An explicit round budget applies; omitting maximumIterations searches to completion.\n *   localIterations defaults to 16. CPU workgroups visit\n *   source rows independently in a deterministic order; GPU scheduling and lane packing can change\n *   partial incumbents and convergence near the budget boundary. Completed paths agree up to rounding.\n *   GPU lane, command-node and device limits are dropped. Capacity still truncates rows.\n * - Unused target/cost capacity is zero-initialized; the GPU does not write unused capacity.\n *\n * @throws Error naming an inconsistent topology or costLimit argument.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkReachability': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkReachabilityOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkReachabilityOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkReachabilityOnCPU',
        signature:
          'function computeNetworkReachabilityOnCPU( input: CPUNetworkReachabilityInput, options: CPUNetworkReachabilityOptions = {} ): CPUNetworkReachabilityResult',
        supportedInputs: [
          'input: CPUNetworkReachabilityInput',
          'options: CPUNetworkReachabilityOptions = {}'
        ],
        differences: [
          "/**\n * Computes shortest paths and the contributor's cycle-safe predecessor and band rules.\n * The CPU predecessor reader keeps tie witnesses specific to their source, including when\n * clamped CSR rows overlap. An edge tight for one source cannot supply another source's parent.\n *\n * What differs from the GPU contributor:\n * - costPrecision defaults to float64 costs/tight-edge tests and one output rounding.\n *   float32-per-edge rounds every addition before cutoff and predecessor tests. Subnormal\n *   weights and overflowing sums can still differ from GPU arithmetic.\n * - With explicit round budgets or activeIterations, a synchronous CPU frontier consumes at most\n *   localIterations hops per round (default 32). GPU workgroup scheduling can finish fewer hops.\n *   Without explicit budgets, searches and tie levels run to completion. Device limits are dropped.\n * - iterationCount counts bounded frontier rounds, or 1 for an uncapped CPU search.\n *   relaxationCount counts CPU edge visits (bounded frontiers can revisit nodes).\n *   reachedEdgeCount counts in-range edges\n *   from finite float64 labels; labels that round to float32 infinity can therefore contribute.\n * - Negative/NaN weights and starts are ignored; NaN costLimit disables the cutoff, like the GPU.\n *   Out-of-range or decreasing offsets retain the GPU's per-row clamping interpretation.\n *\n * @throws Error naming an argument whose array length is inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkStructure': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkStructureOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkStructureOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkStructureOnCPU',
        signature:
          "function computeNetworkStructureOnCPU( input: CPUNetworkStructureInput | {topology: Omit<CPUNetworkStructureInput, 'reverseEdgeSlots'>; reverseEdgeSlots?: Uint32Array}, options: {maximumIterations?: number} = {} ): CPUNetworkStructureResult",
        supportedInputs: [
          "input: CPUNetworkStructureInput | {topology: Omit<CPUNetworkStructureInput, 'reverseEdgeSlots'>; reverseEdgeSlots?: Uint32Array}",
          'options: {maximumIterations?: number} = {}'
        ],
        differences: [
          '/**\n * Finds SCCs and, with reverse slots, undirected resilience using iterative Tarjan traversals.\n * SCC labels are the lowest vertex IDs; block labels are the lowest CSR slots. Invalid neighbor\n * IDs are ignored. Without reverse slots only strongComponents is meaningful.\n *\n * What differs from the GPU contributor:\n * - Exact SCC traversal takes O(vertices + slots) time and memory rather than mutual-reachability BFS.\n * - GPU-only 1024-vertex and 65536-slot caps are dropped. Integer output layouts are unchanged.\n * - Malformed CSR offsets throw an Error on CPU; the GPU clamps offset bounds in its kernels.\n *\n * Supplying maximumIterations selects the budgeted trim/colour/backward/finalize SCC state machine.\n * Exhaustion invalidates every SCC label while leaving resilience results valid. Within each phase,\n * CPU vertex order is deterministic; GPU scheduling may change the number of propagation steps.\n * Omit the budget for an exact Tarjan traversal and status [converged=1, iterationCount=1, limit=0].\n *\n * @throws If offsets or reverseEdgeSlots have inconsistent lengths or offsets are not valid CSR.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkTravelTimePercentiles': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUNetworkTravelTimePercentilesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUNetworkTravelTimePercentilesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeNetworkTravelTimePercentilesOnCPU',
        signature:
          'function computeNetworkTravelTimePercentilesOnCPU( input: CPUNetworkTravelTimePercentilesInput ): CPUNetworkTravelTimePercentilesResult',
        supportedInputs: ['input: CPUNetworkTravelTimePercentilesInput'],
        differences: [
          "/**\n * Routes every window departure within its budget, then selects vertex-major nearest-rank travel\n * times. Unreachable or invalid departures count as Infinity. Optional graph outputs have the\n * same layouts as these always-returned reachedCounts and per-departure [valid, converged] rows.\n * Ranks are ceil(percentile * departureCount / 100), evaluated exactly for the percentile's\n * shortest round-trip decimal (String(percentile)) without an epsilon, so 99.9% of 1,000\n * departures is rank 999; the same host helper serves the GPU contributor.\n *\n * What differs from the GPU contributor:\n * - Routing state, transition, sample and round caps are dropped, as are the 60-departure and\n *   16-percentile caps. Explicit maximumRelaxations still bounds each departure's sweeps.\n * - Departures, routing labels and travel-time subtraction stay float64 until final float32\n *   output; the GPU rounds at its intermediate buffers and shader operations.\n * - Float32 output overflow does not clear valid. A budget exit reports [1, 0]; invalid input\n *   contents retain the GPU's [0, 0] routing status.\n *\n * @throws If initialState, window, transitions or costs have inconsistent array lengths.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-network:GPUKShortestPaths': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'CPUKShortestPathsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-network',
      exportName: 'GPUKShortestPathsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'computeKShortestPathsOnCPU',
        signature:
          'function computeKShortestPathsOnCPU( input: CPUKShortestPathsInput, options: CPUKShortestPathsOptions ): CPUKShortestPathsResult',
        supportedInputs: ['input: CPUKShortestPathsInput', 'options: CPUKShortestPathsOptions'],
        differences: [
          "/**\n * Yen's algorithm using the shared float64 routing and acyclic predecessor helpers.\n *\n * What differs from the GPU contributor:\n * - Float64 costs, rounded once on output, can change near-tie membership and overflow behavior.\n * - GPU node, route, command and device limits are dropped. An explicit maximumIterations binds\n *   each shortest search and tie phase to 64 hops per round on a synchronous CPU frontier.\n *   Without an explicit budget, shortest searches run to convergence.\n * - Equal-cost paths use the same predecessor policy but float64 tightness, so ties may differ.\n *\n * @throws Error naming topology, sources or targets for malformed packed arrays.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainBranchingFlow': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainBranchingFlowOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainBranchingFlowOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainBranchingFlowOnCPU',
        signature:
          'function computeTerrainBranchingFlowOnCPU( input: CPUTerrainBranchingFlowInput, options: CPUTerrainBranchingFlowOptions ): CPUTerrainBranchingFlowResult',
        supportedInputs: [
          'input: CPUTerrainBranchingFlowInput',
          'options: CPUTerrainBranchingFlowOptions'
        ],
        differences: [
          '/**\n * Resolves transport or downstream branching recurrences with a topological dependency queue.\n * Fractions are not renormalized. Invalid local inputs have status 0; dependencies on invalid\n * cells, cycles and invalid used lengths have status 2. unresolvedCount counts both 0 and 2.\n *\n * What differs from the GPU contributor:\n * - Float64 edge lengths, fraction sums and recurrences round once at output; compare resolved\n *   values with relative tolerance 2e-5. Rows at the fraction-sum tolerance can classify differently.\n * - No sweep or successor-walk budget is used. Every finite DAG recurrence completes; status 2\n *   never means an exhausted budget. maximumIterations and maximumWalkLength are ignored.\n * - Float64 intermediates can resolve paths that overflow on GPU. A final value outside float32\n *   is published as NaN/status 2; it may still contribute to a representable dependent result.\n * - Subnormal inputs are retained; GPU arithmetic may flush them to zero.\n *\n * @throws If fractions, values, edgeLengths or validity have inconsistent lengths.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainBreach': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainBreachOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainBreachOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainBreachOnCPU',
        signature:
          'function computeTerrainBreachOnCPU( input: CPUTerrainBreachInput, options: CPUTerrainBreachOptions ): CPUTerrainBreachResult',
        supportedInputs: ['input: CPUTerrainBreachInput', 'options: CPUTerrainBreachOptions'],
        differences: [
          "/**\n * Finds minimum-total-excavation paths on the bounded step-layered D8 graph. Ties choose fewer\n * steps, then the lowest outlet ID; predecessor ties choose the lowest cell ID. All pits search\n * the original surface. Their minimum-elevation proposals combine simultaneously before burning.\n * Paths remain published if the final strict-descent check changes pitStatus from 1 to 2.\n *\n * What differs from the GPU contributor:\n * - Levels, costs, cuts and burns use float64, rounded once at output. Compare finite nonzero\n *   elevations/depths with relative tolerance 2e-5 away from near ties and feasibility bounds;\n *   near ties can choose different integer paths or modification flags.\n * - No GPU scratch-word or storage-binding cap is imposed: one pit's search is held at a time.\n *   maximumSteps and maximumCutDepth remain the definition of the bounded optimization problem.\n * - Positive finite decrements need not be representable in float32 or change the float32 pit\n *   elevation. The search uses float64; final paths must descend strictly in the rounded output,\n *   so collapsed output steps have status 2. A decrement too small to change float64 base has\n *   status 0. GPU-only representability rejection can therefore become a CPU search result.\n * - The pure function supports no pits (burn-only); the operation form requires pitCount above\n *   zero because GPUContributorOperation requires nonempty output vectors.\n * - Subnormals are retained; GPU arithmetic may flush them to zero.\n *\n * @throws If elevation, validity or burnDepth does not contain exactly one value per grid cell.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainCatchmentSummary': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainCatchmentSummaryOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainCatchmentSummaryOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainCatchmentSummaryOnCPU',
        signature:
          'function computeTerrainCatchmentSummaryOnCPU( input: CPUTerrainCatchmentSummaryInput, options: CPUTerrainCatchmentSummaryOptions ): CPUTerrainCatchmentSummaryResult',
        supportedInputs: [
          'input: CPUTerrainCatchmentSummaryInput',
          'options: CPUTerrainCatchmentSummaryOptions'
        ],
        differences: [
          "/**\n * Aggregates basin statistics and cumulative upper-edge histograms, then memoizes stream paths.\n * Each edge belongs to its source basin, even across a basin boundary. Profiles traverse through\n * excluded basin IDs; only the starting cell's basin determines whether its output is excluded.\n * maximumSteps is retained: the number of stream edges must be strictly less than it.\n *\n * What differs from the GPU contributor:\n * - Float64 ground measures, compensated elevation sums, means, thresholds and profiles round\n *   once at output, instead of fixed-point sums of float32 terms and float32 pointer-jumping.\n *   Compare nonzero finite arithmetic results with relative tolerance 2e-5. Copied elevations\n *   and extrema agree exactly; integer statuses agree away from threshold/overflow boundaries.\n *   Values close to a hypsometry edge can enter a different bin.\n * - Histograms accumulate each cell once and prefix-sum bins; profile walks have no GPU round\n *   budget. The documented exclusive maximumSteps bound and cycle/failure statuses are kept.\n * - A final distance outside float32 becomes NaN/status 2; finite float64 intermediates can\n *   resolve a path that overflows on GPU. Subnormals are retained rather than flushed to zero.\n *\n * @throws If any input array does not contain exactly one value per grid cell.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainDepressionHierarchy': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainDepressionHierarchyOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainDepressionHierarchyOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainDepressionHierarchyOnCPU',
        signature:
          'function computeTerrainDepressionHierarchyOnCPU( input: CPUTerrainDepressionHierarchyInput, options: CPUTerrainDepressionHierarchyOptions ): CPUTerrainDepressionHierarchyResult',
        supportedInputs: [
          'input: CPUTerrainDepressionHierarchyInput',
          'options: CPUTerrainDepressionHierarchyOptions'
        ],
        differences: [
          "/**\n * Builds the stable sublevel-set merge tree (elevation then cell ID), retaining each cell's\n * activation node. New nodes represent isolated births or merges of distinct active components;\n * a single adjacent component absorbs the cell. Curves include a node's descendants at every\n * query stage, without clipping to spill levels. Outlets are explicit; invalid cells are barriers.\n *\n * What differs from the GPU contributor:\n * - Ground areas, depths, products and curve sums use float64, rounded once at output, instead\n *   of fixed-point sums of float32 terms. Compare nonzero finite curves with relative tolerance\n *   2e-5. Node IDs, stable ties, birth/spill levels, ownership and status ports agree exactly.\n * - The CPU has no GPU watchdog or storage-binding scratch limit. Overflow still invalidates\n *   every published node/curve, while cellNodes retains the full internal ownership IDs.\n * - The pure function accepts an empty stages array. The operation form requires stageCount\n *   above zero because GPUContributorOperation requires nonempty output vectors.\n * - Subnormal elevations are retained; GPU arithmetic may flush them to zero.\n *\n * @throws If elevation, outlets or validity does not hold exactly one value per grid cell.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainFlowDistance': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainFlowDistanceOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainFlowDistanceOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainFlowDistanceOnCPU',
        signature:
          'function computeTerrainFlowDistanceOnCPU( flowDirections: Uint32Array, options: CPUTerrainFlowDistanceOptions, streams?: Uint32Array, transitionCosts?: Float32Array ): CPUTerrainFlowDistanceResult',
        supportedInputs: [
          'flowDirections: Uint32Array',
          'options: CPUTerrainFlowDistanceOptions',
          'streams?: Uint32Array',
          'transitionCosts?: Float32Array'
        ],
        differences: [
          '/**\n * Resolves downstream distances and terminal cells with memoized float64 path sums. A terminal\n * contributes one cell to maximumSteps, including when a stream stops the path. Bad directions\n * take precedence over stream stops, as in the GPU seed kernel.\n *\n * What differs from the GPU contributor:\n * - Edge lengths and sums use float64 and round once at the output, rather than float32\n *   pointer-doubling sums. Resolved distances agree within float32 accumulation error.\n * - No command-node or pointer-round limit is needed. maximumSteps is retained as the documented\n *   path-length definition; cycles, overlong paths, bad costs and unknown inputs stay unresolved.\n * - Finite float64 totals that overflow float32 are invalidated at output; GPU intermediate\n *   overflow can invalidate additional paths. All terminal and missing codes otherwise match.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainStreamOrder': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainStreamOrderOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainStreamOrderOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainStreamOrderOnCPU',
        signature:
          'function computeTerrainStreamOrderOnCPU( flowDirections: Uint32Array, streams: Uint32Array, options: {grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding} ): CPUTerrainStreamOrderResult',
        supportedInputs: [
          'flowDirections: Uint32Array',
          'streams: Uint32Array',
          'options: {grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding}'
        ],
        differences: [
          "/**\n * Computes Strahler orders with a donor queue. Unknown direction neighbors seed unknown order,\n * which propagates downstream. Invalid/non-stream cells are zero. Rootless stream components\n * (including tributaries into receiver cycles) are unresolved, matching the GPU Euler-tour path.\n * The streams input uses 0 for nonstream, 1..0xfffffffe for membership and 0xffffffff for\n * unknown. Translate imported unresolved 0xfffffffe to 0xffffffff, as on the GPU port; this\n * preserves the existing meaning of 0xfffffffe membership. Direction sentinels are separate.\n *\n * What differs from the GPU contributor:\n * - Completes the donor queue without maximumIterations, the 1024-round pull cap, or tour scratch.\n * - iterationCount is 0 (no GPU rounds); iterationLimitReached is always 0. converged is 0 only\n *   for rootless receiver-cycle components, matching the Euler-tour path's status definition.\n * - Rootless tributaries match the Euler-tour output; the GPU pull fallback can resolve their\n *   upstream orders before it stalls at a cycle. Unknown-input statuses follow the tour path too.\n * - Resolved orders are exact integers; there is no numeric tolerance difference.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainWatersheds': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainWatershedsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainWatershedsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainWatershedsOnCPU',
        signature:
          'function computeTerrainWatershedsOnCPU( flowDirections: Uint32Array, options: { grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding; maximumIterations?: number; }, pourPoints?: Uint32Array ): CPUTerrainWatershedsResult',
        supportedInputs: [
          'flowDirections: Uint32Array',
          'options: {\n    grid: GPURasterMetadata;\n    directionEncoding?: GPUTerrainFlowDirectionEncoding;\n    maximumIterations?: number;\n  }',
          'pourPoints?: Uint32Array'
        ],
        differences: [
          '/**\n * Labels each path at its nearest inclusive pour point, or at its terminal cell without points.\n * Memoized walks preserve duplicate-point priority, invalid words and unknown destinations.\n *\n * What differs from the GPU contributor:\n * - Without maximumIterations, walks complete and iterationCount is 0 (no pointer rounds).\n *   An explicit budget instead bounds synchronous pointer rounds, including confirmation;\n *   iterationLimitReached is one if the last permitted round still changed a pointer.\n *   Explicit budgets are in [1,1024], as on the GPU contributor.\n * - Both forms reject cycles and unmarked unresolved directions, even when their pointers\n *   are stable. An inclusive pour point can intercept either path and restore convergence.\n * - No float outputs or tolerance differences; resolved labels and point ties are exact integers.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainFlow': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainFlowOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainFlowOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainFlowOnCPU',
        signature:
          'function computeTerrainFlowOnCPU( input: CPUTerrainFlowInput, options: CPUTerrainFlowOptions ): CPUTerrainFlowResult',
        supportedInputs: ['input: CPUTerrainFlowInput', 'options: CPUTerrainFlowOptions'],
        differences: [
          "/**\n * Fills by a multi-source minimax priority flood, resolves Barnes flats by breadth-first hop\n * distances, and accumulates the complete receiver DAG in topological order. Boundary means the\n * tile edge or any of eight neighbors missing. Strict D8/facet/mask comparisons retain the\n * contributor's first-direction tie rule. Nonfinite/negative runoff contributes zero.\n *\n * What differs from the GPU contributor:\n * - Calibration, fill increments, slopes, fractions, areas and accumulation use float64, with\n *   one float32 rounding at output. Use relative tolerance 3e-5 on well-conditioned float outputs;\n *   near slope/facet ties, rounded flat heights and stream thresholds can change integer outputs.\n *   GPU float32 intermediate overflow can invalidate additional elevations or routings; the twin\n *   keeps finite float64 intermediates even when their published elevation overflows float32.\n * - Zero-epsilon spill filling and DAG accumulation complete without relaxation budgets.\n *   Without explicit budgets, positive-epsilon filling and flat resolution run to completion.\n * - An explicit maximumFillIterations bounds positive-epsilon tiled filling, including unchanged\n *   confirmation. Exhaustion clears fillConverged/flatsConverged and suppresses dependent outputs.\n * - An explicit maximumFlatIterations bounds each of the three tiled flat stages, including\n *   unchanged confirmation. Plain tile halos read a round snapshot; after 96 rounds both bounded\n *   fields cycle tiled/right/left/down/up, publishing each swept tile to the next on its line.\n *   GPU workgroup scheduling can advance other lines sooner. Exhaustion clears flatsConverged and suppresses\n *   dependent directions, classes, accumulation and streams with unresolved values.\n * - Iteration-count words are 0 except explicit fill/flat budgets, which count executed stage rounds.\n *   Accumulation convergence describes DAG completion, not the GPU's conservative last-round flag.\n * - Explicit flat budgets must be in [1,1024]. Other GPU-only upper iteration limits are dropped.\n *   Optional persistent flatContinuation and texture bands have no operation ports.\n * - Nonfinite calibrated elevations are missing; subnormals are retained. NaN slopes do not\n *   route, and non-positive/nonfinite epsilon and exponent use the contributor's defaults.\n *   A float64 accumulation that overflows float32 is published as infinity and its stream word\n *   is unresolved, as for any nonfinite GPU accumulation.\n * - Both forms write requested flat statuses even on filledElevation-only requests.\n *\n * No-data metadata follows the scalar format: uint32 and sint32 require an integer in their\n * respective ranges; float32 accepts NaN or a value that stays finite after float32 rounding.\n * Invalid metadata (including uint32 -1 or 0.5, sint32 NaN and float32 Infinity) throws an Error\n * naming elevation.noDataValue, both at operation construction and in this function.\n *\n * @throws Error naming inconsistent elevation.values, elevation.validity, parameters or runoff,\n * or invalid elevation.noDataValue metadata.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUProfilePeaks': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUProfilePeaksOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUProfilePeaksOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeProfilePeaksOnCPU',
        signature:
          'function computeProfilePeaksOnCPU( values: Float32Array, offsets: Uint32Array, parameters: Float32Array, options: CPUProfilePeaksOptions, validity?: Uint32Array ): CPUProfilePeaksResult',
        supportedInputs: [
          'values: Float32Array',
          'offsets: Uint32Array',
          'parameters: Float32Array',
          'options: CPUProfilePeaksOptions',
          'validity?: Uint32Array'
        ],
        differences: [
          '/**\n * Finds window-prominent maxima, refines them parabolically and completes ranked greedy suppression.\n * Parameters use the layout of getGPUProfilePeaksParameterValues (minimumProminence).\n *\n * What differs from the GPU contributor:\n * - Prominence rounds to float32 before inclusive minimum acceptance and candidate ranking,\n *   matching the reported precision. Parabolic refinement uses float64 and rounds on output;\n *   near the inclusive suppression boundary, float32 GPU position decisions can differ.\n * - Suppression honors the round budget (default 16). Within a round, decisions are visited in\n *   source-row order; GPU invocation scheduling can decide a different subset at a small budget.\n * - GPU radius limits are dropped; window and radius may exceed 4096.\n * - Unused compact ID rows are zero-filled; only the first count rows are defined by the GPU.\n * - As on the GPU, overflow reports capacity alone. candidateOverflow reports undecided peaks;\n *   a compact-only operation must forward that port or a convergence status to prove completion.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainPeakProminence': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUTerrainPeakProminenceOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUTerrainPeakProminenceOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeTerrainPeakProminenceOnCPU',
        signature:
          'function computeTerrainPeakProminenceOnCPU( elevation: CPURasterOperationBand, grid: GPURasterMetadata ): CPUTerrainPeakProminenceResult',
        supportedInputs: ['elevation: CPURasterOperationBand', 'grid: GPURasterMetadata'],
        differences: [
          "/**\n * Builds the upper-excursion elder join tree by stable descending-saddle edge unions. Higher\n * peaks survive; equal heights favor the lowest cell index. Validity, raw nodata and calibration\n * use the shared raster-band helper. Disconnected elders have infinite prominence.\n *\n * What differs from the GPU contributor:\n * - Buffer bands only; texture storage is excluded from both operation forms.\n * - Calibration, saddle ordering and height differences use float64, rounding only final float\n *   outputs. GPU float32 calibration can collapse distinct heights, changing elder and parent ties,\n *   or overflow samples that float64 retains. Native float32 nodata is rounded to its input format.\n * - Serial union-find replaces atomic parent splicing and ancestor doubling, retaining stable\n *   cell/E/SE/S/SW edge order and exact integer parent ties when calibrated heights agree.\n * - The GPU's five-tree-nodes-per-cell storage-binding limit is dropped; the CPU needs no GPU scratch.\n * - Zero saddles are always +0 in the twin; GPU source normalizes them too, but the test GPU (Apple, measured 2026-11-11) publishes -0.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUSolarPosition': {
    cpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'CPUSolarPositionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/gpu-terrain',
      exportName: 'GPUSolarPositionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'computeSolarPositionOnCPU',
        signature:
          'function computeSolarPositionOnCPU( positions: Float32Array, parameters: Float32Array, options: {refraction?: boolean} = {} ): CPUSolarPositionResult',
        supportedInputs: [
          'positions: Float32Array',
          'parameters: Float32Array',
          'options: {refraction?: boolean} = {}'
        ],
        differences: [
          "/**\n * Evaluates getSolarPosition for packed longitude/latitude rows using the packed time and\n * daylight threshold from getGPUSolarPositionParameterValues.\n *\n * What differs from the GPU contributor:\n * - Calls the public float64 NOAA helper and rounds outputs once. The GPU's split float32\n *   orbital-angle evaluation differs increasingly far from J2000 and near zenith/nadir.\n * - Daylight can differ for altitudes within the GPU angular error of the strict threshold.\n * - A final float32 azimuth of 360 is canonicalized to 0. The GPU's angular error can place its\n *   answer on either side of this seam, so compare angles circularly and check the range separately.\n * - Longitude arithmetic remains float64, including finite inputs for which the GPU's\n *   multiplication by 4 overflows float32; actual GPU behavior there is unverified.\n * - Nothing else: invalid coordinates yield NaN angles and daylight 0; refraction affects only altitude.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUClassBreaks': {
    cpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'CPUClassBreaksOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPUClassBreaksOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'computeClassBreaksOnCPU',
        signature:
          'function computeClassBreaksOnCPU( values: Float32Array | Uint32Array | Int32Array, parameters: Float32Array, options: CPUClassBreaksOptions, mask?: Uint32Array ): CPUClassBreaksResult',
        supportedInputs: [
          'values: Float32Array | Uint32Array | Int32Array',
          'parameters: Float32Array',
          'options: CPUClassBreaksOptions',
          'mask?: Uint32Array'
        ],
        differences: [
          "/**\n * Computes all class-break methods with float64 moments and exact Fisher-Jenks partitions.\n * Outputs retain the lower-edge classification rule, repeated edges, end-class infinities and\n * NaN tails. Maximum-gap distinctness and class assignment use numeric float32 comparison keys:\n * -0 equals +0 and both enter the upper class at a zero break. Raw key ordering preserves extrema bits.\n * Integer input columns undergo the contributor's initial float32 conversion.\n *\n * What differs from the GPU contributor:\n * - Arithmetic uses float64 with one float32 edge rounding. Quantile ranks and head thresholds\n *   can differ near a float32 boundary. Compare conditioned arithmetic edges within 3e-5 relative;\n *   counts can differ when a computed edge crosses a source value.\n * - Natural breaks always find the global float64 Fisher-Jenks optimum over distinct-value\n *   boundaries, using all source multiplicities, for approximate, exact and monotone GPU modes.\n *   There is no binning, quantization, local refinement or normalized float32 recurrence. Exact\n *   cost ties retain the smallest predecessor, as GPU exact mode does; near ties may differ.\n * - Maximum-break gaps and midpoints use float64; float32 gap overflow or rounded gap ties no\n *   longer affect their order. Equal exact gaps still favor the lower sorted position.\n * - The GPU's 64-class, 2^24-row, bin-count and exact/monotone storage limits are dropped;\n *   uint32 counts and edge-array lengths remain. Natural-break execution/storage settings are\n *   ignored by the CPU operation. Its DP costs O(k u^2) time and O(k u) memory for u distinct values.\n * - Subnormals and finite float64 intermediates beyond float32 range are retained. Non-custom\n *   computed inner edges are clamped to the finite data range before publishing, as on the GPU.\n *\n * @throws If mask length disagrees with values or parameters is shorter than its packed layout.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUColumnQuantiles': {
    cpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'CPUColumnQuantilesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPUColumnQuantilesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'computeColumnQuantilesOnCPU',
        signature:
          'function computeColumnQuantilesOnCPU( input: CPUColumnQuantilesInput, quantileCount: number ): CPUColumnQuantilesResult',
        supportedInputs: ['input: CPUColumnQuantilesInput', 'quantileCount: number'],
        differences: [
          "/**\n * Sorts the finite, unmasked float32 sample and computes order statistics in float64.\n * The parameter header and probability pairs have the contributor packers' layouts.\n *\n * What differs from the GPU contributor:\n * - At most 2^24 valid rows: packed float ranks and filter products round to float32 before selecting rows, as on GPU. For CPU samples above 2^24 valid rows, rank products retain float64 precision.\n *   Interpolation uses float64 with one float32 output rounding; rational ranks use exact integers.\n * - The GPU's 64-quantile and float32 row-count limits are dropped; uint32 output counts remain.\n * - Non-finite values are missing and zero is canonicalized to +0, as on the GPU. Subnormal values\n *   are retained; devices may flush them to zero. Unknown interpolation codes use lower, as the kernel does.\n *\n * @throws If mask, parameters or rationalProbabilities have inconsistent lengths.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUInequality': {
    cpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'CPUInequalityOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPUInequalityOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'computeInequalityOnCPU',
        signature:
          'function computeInequalityOnCPU(input: CPUInequalityInput): CPUInequalityResult',
        supportedInputs: ['input: CPUInequalityInput'],
        differences: [
          '/**\n * Computes weighted zone indices, Lorenz knots and pooled Theil decomposition in float64.\n * Filters and undefined outputs match GPUInequality, including NaN knots for zero income.\n *\n * What differs from the GPU contributor:\n * - Float64 sums and transcendental functions replace float32; compare ordinary inputs at 2e-5.\n * - Atkinson uses log-sum-exp, so finite results survive intermediate powers that overflow on GPU.\n * - The GPU scratch-binding zone limit and 1025-knot cap are dropped; uint32 layout limits remain.\n * - Equal incomes retain input order; this cannot change the exact weighted statistics.\n * - Empty values and zoneIds are accepted: counts and pooled totals are zero; indices, mean and\n *   Lorenz knots are NaN. The GPU contributor requires at least one input row.\n * - Subnormal incomes and weights may be flushed to zero on the GPU and are kept by the twin.\n *\n * @throws If zoneIds, weights or mask lengths disagree with values, parameters is too short,\n * or the zone or knot count is invalid. Argument names are included in errors.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUVariogramModelFit': {
    cpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'CPUVariogramModelFitOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPUVariogramModelFitOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'computeVariogramModelFitOnCPU',
        signature:
          'function computeVariogramModelFitOnCPU( input: VariogramModelInput, options: CPUVariogramModelFitOptions = {} ): CPUVariogramModelFitResult',
        supportedInputs: [
          'input: VariogramModelInput',
          'options: CPUVariogramModelFitOptions = {}'
        ],
        differences: [
          '/**\n * Fits each model family to empirical variogram bins with `fitVariogramModel` in float64 and\n * returns the rows `GPUVariogramModelFit` writes, rounded to float32.\n *\n * What differs from the GPU fit:\n * - Status 2 is set only when a fitted value is not finite. The GPU fit also refuses inputs whose\n *   float32 objective cannot be evaluated; float64 fits those.\n * - CPU defaults to 60 refinement steps, GPU to 32. Explicit rangeSteps and maximumIterations\n *   bind both searches. CPU refines up to its budget in float64; GPU may stop at bracket 1e-5.\n * - The bracket diagnostic is the measured final log-range bracket. Both forms set status 3\n *   for an endpoint ladder optimum or a final bracket wider than 1e-5.\n * - Cressie weights have no floor other than the smallest float64.\n *\n * @param input Empirical bins. Bins with no pairs, a non-finite value or a non-positive lag are\n *   skipped.\n * @param options Model families, weighting and search budgets.\n * @returns One row per model.\n * @throws If bin arrays have unequal lengths, a model family or the weighting is unknown.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/dataframe-analysis:GPURipleyEnvelope': {
    cpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'CPURipleyEnvelopeOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPURipleyEnvelopeOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'computeRipleyEnvelopeOnCPU',
        signature:
          'function computeRipleyEnvelopeOnCPU( input: CPURipleyEnvelopeInput, options: CPURipleyEnvelopeOptions ): CPURipleyEnvelopeResult',
        supportedInputs: ['input: CPURipleyEnvelopeInput', 'options: CPURipleyEnvelopeOptions'],
        differences: [
          "/**\n * Binomial CSR rectangle simulations, Ripley K estimation and pointwise/global MAD envelopes.\n * Draws use the contributor's registered Philox purpose and counter order.\n *\n * What differs from the GPU contributor:\n * - This operation form omits the optional prepared polygon window; only rectangles are supported.\n * - Coordinates, distances, K estimates, deviations and inference use float64, rounded only at\n *   final outputs. Near distance thresholds or deviation ties can change counts and p-values.\n * - Isotropic weights retain the documented cap of 100 but omit GPU fixed-point quantization.\n *   Ordinary curves allow 2e-4 relative error, with 2e-5 absolute for L-minus-r cancellation.\n * - GPU-only row, radius, simulation and command-node caps are dropped. Pair searches are exact\n *   all-pairs, and gridSize/maximumAttempts are unused for rectangles.\n * - Invalid lattice bounds/nonpositive maximumDistance exclude all observations. Finite values\n *   whose GPU intermediate width, squared radius or area overflows remain usable in float64.\n * - Nonfinite coordinates are excluded, as on the GPU; NaN observed K ordinates are skipped.\n *\n * @throws If positions is not float32x2 rows, mask length differs, observedK has fewer than\n * radiusCount ordinates, or parameters does not contain the packer's full layout.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUSpaceTimeDiggleTest': {
    cpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'CPUSpaceTimeDiggleTestOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPUSpaceTimeDiggleTestOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'computeSpaceTimeDiggleTestOnCPU',
        signature:
          'function computeSpaceTimeDiggleTestOnCPU( input: CPUSpaceTimeDiggleTestInput, options: CPUSpaceTimeDiggleTestOptions ): CPUSpaceTimeDiggleTestResult',
        supportedInputs: [
          'input: CPUSpaceTimeDiggleTestInput',
          'options: CPUSpaceTimeDiggleTestOptions'
        ],
        differences: [
          "/**\n * Diggle's sum of D=K-K_s K_t under random time relabelling, with the current splancs rules:\n * largest distance/lag excluded, interior class bounds included, temporal edge ties weighted 2.\n *\n * What differs from the GPU contributor:\n * - All calculations and inference comparisons use float64, rounding only final output arrays.\n *   Ordinary K/marginal/standard-deviation values allow 2e-4 relative error; signed residuals\n *   and sums allow 2e-4 of the contributing K scale. Near ties can change permutation counts.\n * - The registered Philox shuffle is unchanged, over included rows in original row order.\n * - GPU-only 64-class, 999-permutation, histogram, binding-size and command-node caps are\n *   dropped; uint32 output lengths remain. All ordered pairs are visited without a grid index.\n * - Isotropic weights retain the estimator's cap of 100; fixed-point quantization is omitted.\n *   Status 3 (GPU integer accumulator overflow) is never set; 0 and 1 keep their meanings.\n * - Nonfinite/out-of-window positions or times are excluded. Finite bounds, squared search\n *   radii and areas that overflow float32 remain usable when the float64 result is finite.\n *\n * @throws If positions/times/mask lengths disagree or either parameter layout is incomplete.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUCoregionalizationFit': {
    cpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'CPUCoregionalizationFitOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/dataframe-analysis',
      exportName: 'GPUCoregionalizationFitOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'computeCoregionalizationFitOnCPU',
        signature:
          'function computeCoregionalizationFitOnCPU( input: CPUCoregionalizationFitInput, options: CoregionalizationFitOptions ): CPUCoregionalizationFitResult',
        supportedInputs: [
          'input: CPUCoregionalizationFitInput',
          'options: CoregionalizationFitOptions'
        ],
        differences: [
          '/**\n * Goulard–Voltz fitting from zero: sequential conditional least squares and PSD projection.\n *\n * What differs from the GPU contributor:\n * - Float64 sufficient statistics, eigendecompositions and coefficients; one rounding at output.\n * - GPU-only bin and sweep caps are dropped; variable/structure bounds keep the kriging layout.\n * - Convergence iteration counts and eigenvector signs/bases can differ at float32 rounding\n *   thresholds or repeated eigenvalues. Compare B and L L^T, not individual factor entries.\n * - Finite float64 intermediate arithmetic can fit bins the GPU rejects on float32 overflow.\n *\n * Empty/invalid bins skip the complete lag matrix. Malformed array lengths throw. Invalid model\n * parameters or a zero-energy structure return validity 0 and NaN coefficients. Budget exits\n * return the current PSD matrices, including the zero model for a zero budget.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPUCostDistancePath': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPUCostDistancePathOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPUCostDistancePathOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeCostDistancePathOnCPU',
        signature:
          'function computeCostDistancePathOnCPU( backLinks: Uint32Array, target: Uint32Array, grid: GPURasterMetadata, capacity: number ): CPUCostDistancePathResult',
        supportedInputs: [
          'backLinks: Uint32Array',
          'target: Uint32Array',
          'grid: GPURasterMetadata',
          'capacity: number'
        ],
        differences: [
          '/**\n * Follows D8 predecessor codes from one target, publishing only a walk that reaches a source.\n * Invalid walks have zero count even if their scratch prefix has been written to `ids`.\n *\n * What differs from the GPU contributor:\n * - Unwritten rows of `ids` are zero; the GPU leaves rows outside its written prefix unspecified.\n * - Nothing else: integer codes, the cell-count cycle bound and all status words match exactly.\n *\n * @throws If `backLinks` or `target` has an inconsistent length, or grid/capacity is invalid.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterWalkingCost': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterWalkingCostOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterWalkingCostOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterWalkingCostOnCPU',
        signature:
          'function computeRasterWalkingCostOnCPU( input: CPURasterWalkingCostInput, options: CPURasterWalkingCostOptions ): CPURasterWalkingCostResult',
        supportedInputs: [
          'input: CPURasterWalkingCostInput',
          'options: CPURasterWalkingCostOptions'
        ],
        differences: [
          "/**\n * Computes the directed least-cost fixed point, with the contributor's Tobler/GRASS transition,\n * midpoint-latitude ground lengths, impassable transitions and model-mismatch status. Sources\n * are a per-cell nonzero mask, not a list of cell indices. Nonfinite parameters follow the same\n * tests as the kernel: invalid model words move nothing; other nonfinite transition costs are\n * discarded. Band calibration is the shared raster operation calibration. Only the first\n * grid.width * grid.height sources rows are read; surplus source rows are ignored.\n *\n * What differs from the GPU contributor:\n * - Explicit maximumIterations permits at most maximumIterations + 1 synchronous rounds, just\n *   as on the GPU, publishing partial costs and the stopping flags. With no explicit budget the\n *   CPU retains its complete fixed-point solve. The GPU command-node cap does not apply.\n * - Calibration, ground scales, transitions and costs use float64 before one float32 output\n *   rounding. Compare costs with relative tolerance 3e-5; near-equal path choices and the number\n *   of rounds can differ. A float64-finite transition that overflows float32 remains usable.\n * - iterationCount counts synchronous float64 rounds including the first unchanged round, when\n *   reached. invalidCount keeps the GPU's model-layout mismatch meaning.\n * - Subnormal samples and costs are retained; GPU implementations may flush them to zero.\n *\n * @throws Error naming an input whose array length is inconsistent, or an invalid option.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPUCostDistance': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPUCostDistanceOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPUCostDistanceOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeCostDistanceOnCPU',
        signature:
          'function computeCostDistanceOnCPU( input: CPUCostDistanceInput, options: CPUCostDistanceOptions ): CPUCostDistanceResult',
        supportedInputs: ['input: CPUCostDistanceInput', 'options: CPUCostDistanceOptions'],
        differences: [
          "/**\n * Computes least costs with float64 Dijkstra when unbounded, or bounded tiled relaxation,\n * and cycle-safe plateau levels. The pure function returns back-links, so it applies the tie budget;\n * the operation omits that phase when back-links are not requested.\n * Strict predecessors minimize candidate cost, with smallest D8 direction on ties; plateau\n * links use the smallest direction one level nearer a strict entry or surviving source.\n *\n * What differs from the GPU contributor:\n * - Explicit cost/tie budgets use 16x16 tiled rounds, with round-start snapshots between\n *   workgroups and ordered writes within sweep lines. The final unchanged round consumes budget.\n *   Without an explicit budget the CPU computes the full fixed point. Stateful continuation is omitted.\n * - settledAt is the minimum hop count of a float64 least-cost path from a surviving seed,\n *   or zero for unreached cells. iterationCount is the largest such count plus one (the first\n *   unchanged synchronous round). With an explicit cost budget these become tiled last-decrease\n *   stamps and the executed tiled round count. GPU scheduling can give different diagnostics.\n * - Calibration, distances and path sums use float64, rounded once at output. Compare costs\n *   with relative tolerance 3e-5. Near ties, threshold bands and back-links can differ.\n * - Strict tightness uses the GPU's 4-ulp relative slack in float64. Equal-cost edges use its\n *   addition test; subnormal and float64-finite costs that exceed float32 are retained internally.\n *   A cost that rounds to infinity has no published back-link or band and is not counted.\n * - Nonfinite or negative transition arithmetic produces no usable transitions; invalid seeds\n *   are ignored. Negative packed cell sizes are outside the documented GPU parameter layout.\n *\n * @throws Error naming an inconsistent input array or invalid grid.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPUStreamlines': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPUStreamlinesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPUStreamlinesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeStreamlinesOnCPU',
        signature:
          'function computeStreamlinesOnCPU( input: CPUStreamlinesInput, options: CPUStreamlinesOptions ): CPUStreamlinesResult',
        supportedInputs: ['input: CPUStreamlinesInput', 'options: CPUStreamlinesOptions'],
        differences: [
          "/**\n * Traces jittered seeds using normalized RK2 and bilinear cell-centre sampling, then resolves\n * occupancy greedily in descending Philox priority. Publishes complete lines in ascending seed\n * order, stopping at the first line that exceeds either capacity. Only active rows are defined.\n *\n * What differs from the GPU contributor:\n * - Pruning honors maximumIterations (default 32); unresolved seeds are dropped and reported by\n *   candidateOverflow and iterationLimitReached. Final overflow reports capacity alone.\n *   Candidate spans contain traced backward/forward counts, then final kept counts for accepted\n *   seeds and zeros for rejected seeds; GPU rejected seeds can retain intermediate kept counts.\n * - Seed placement, interpolation, normalization and RK2 arithmetic use float64,\n *   rounded once to float32 positions. Compare positions with absolute tolerance 3e-5 times\n *   placement magnitude; near field/grid boundaries integer selections can differ.\n *   Seeds and steps whose final float32 position has no field support or is outside the grid\n *   are rejected/truncated. Supported below-minimumSpeed terminal landings are retained;\n *   they cannot start a following step, matching the GPU streamline contributor.\n *   Occupancy uses published float32 coordinates with the GPU's correctly rounded subtraction\n *   and reciprocal multiplication, so the exclusive bound is checked in that representation.\n * - The 65535 seed limit remains because priorities reserve 16 bits for seed index. GPU dimension\n *   caps are dropped; field rows still fit uint32 and occupancy indices must be safe integers.\n *   Unused output positions are zero-initialized; unused streamline slots have repeated terminal offsets, preserving monotonicity through the full offsets allocation.\n * - NaN field corners invalidate a sample even at zero interpolation weight. Nonfinite\n *   directions/positions terminate tracing; subnormal finite velocities remain usable.\n *\n * @throws Error naming inconsistent arrays or invalid dimensions/capacities.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPUIsolines': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPUIsolinesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPUIsolinesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeIsolinesOnCPU',
        signature:
          'function computeIsolinesOnCPU( input: CPUIsolinesInput, options: CPUIsolinesOptions ): CPUIsolinesResult',
        supportedInputs: ['input: CPUIsolinesInput', 'options: CPUIsolinesOptions'],
        differences: [
          '/**\n * Marching squares in cell/level/exit-edge order, followed by directed edge-chain walks.\n * Keeps high values on the left, uses the inclusive asymptotic saddle decider, retains raw\n * zero-length bridges, and checks complete-chain capacity before collapsing duplicate vertices.\n * Only rows below count/vertexCount and offsets through polylineCount are defined outputs.\n *\n * What differs from the GPU contributor:\n * - Interpolation and the saddle determinant use float64, with one rounding per output coordinate.\n *   Compare coordinates with absolute tolerance 2e-6 times the extent magnitude. Saddle decisions\n *   near a float32 determinant tie and duplicate collapse near rounding boundaries can differ.\n * - Chains are walked directly rather than pointer-jumped; no GPU round or command budget applies.\n * - Finite input differences that overflow float32 are evaluated in float64. In particular,\n *   opposing maximum-float32 samples at level zero cross at the midpoint (x = 1 on a unit\n *   2-by-2 placement). GPU interpolation after denominator overflow is outside the comparison\n *   domain; its device/compiler-dependent result is not assumed here.\n * - Nonfinite active levels emit no segments; nonfinite placement parameters propagate to vertices.\n *   Subnormal values are retained. Unused output capacity is zero-initialized on the CPU.\n *\n * @throws Error naming arrays that are too short or invalid dimensions/capacities.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterFractionalZonalDistribution': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterFractionalZonalDistributionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterFractionalZonalDistributionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterFractionalZonalDistributionOnCPU',
        signature:
          'function computeRasterFractionalZonalDistributionOnCPU( input: CPURasterFractionalZonalDistributionInput, options: CPURasterFractionalZonalDistributionOptions ): CPURasterFractionalZonalDistributionResult',
        supportedInputs: [
          'input: CPURasterFractionalZonalDistributionInput',
          'options: CPURasterFractionalZonalDistributionOptions'
        ],
        differences: [
          "/**\n * Computes coverage/reliability weighted moments and exactextract distinct-value interpolated\n * quantiles (or inverse weighted CDF) in the contributor's zone-major output layout. Non-uniform\n * cellSizeMode uses getRasterRowAreaScale, with the row edges and the area determinant from\n * parameters when supplied and from grid.affine otherwise, just as the contributor does.\n *\n * What differs from the GPU contributor:\n * - Calibration, weighted moments and quantile interpolation use float64; final arrays round once.\n *   NON_FINITE_RESULT checks final float32 representability rather than float32 intermediates.\n *   Calibrated observations beyond float32 range can contribute instead of becoming GPU nodata.\n * - Ground-area row scales use the existing float64 helper before final area output rounding.\n * - GPU tuple, quantile and zone caps are dropped; no tuple is approximated or discarded.\n * - The twin requires only a finite nonzero grid.affine determinant. The contributor also\n *   refuses a determinant that rounds to zero in float32 or whose absolute value times the\n *   coverage capacity is not finite in float32; the twin accepts both and reports any area that\n *   overflows float32 as NON_FINITE_RESULT.\n *\n * @throws If any input array has an inconsistent length, or grid/options are invalid.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterFractionalZonalStatistics': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterFractionalZonalStatisticsOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterFractionalZonalStatisticsOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterFractionalZonalStatisticsOnCPU',
        signature:
          'function computeRasterFractionalZonalStatisticsOnCPU( input: CPURasterFractionalZonalStatisticsInput, options: CPURasterFractionalZonalStatisticsOptions ): CPURasterFractionalZonalStatisticsResult',
        supportedInputs: [
          'input: CPURasterFractionalZonalStatisticsInput',
          'options: CPURasterFractionalZonalStatisticsOptions'
        ],
        differences: [
          "/**\n * Aggregates all zones in one tuple pass, preserving the contributor's count clamp, ignored\n * out-of-range IDs and completeness rule (matching in-capacity counts, overflow and optional\n * candidateOverflow zero, zoneValidity nonzero). Fractions are used as supplied,\n * including negative and nonfinite fractions; they are not silently sanitized. Coverage cellIds\n * and fractions may have surplus rows beyond zoneIds.length, zoneValidity beyond zoneCount,\n * and scalar coverage statuses beyond their first row; these tails are ignored as on the GPU.\n *\n * What differs from the GPU contributor:\n * - Band calibration, sums and division use float64, with one rounding at each final output;\n *   compare ordinary finite outputs with relative tolerance 3e-6. Float32 intermediate overflow\n *   and underflow can therefore produce different values and validity near representability limits.\n * - A calibrated observation beyond float32 range remains usable when finite in float64.\n * - Subnormal inputs and intermediate quantities are retained; GPU implementations may flush them.\n * - Operation outputs require at least one zone, as required by GPUContributorOperation.\n *\n * @throws Error naming an argument when its array lengths or options are inconsistent.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterRegionGraph': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterRegionGraphOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterRegionGraphOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterRegionGraphOnCPU',
        signature:
          'function computeRasterRegionGraphOnCPU(labels: Uint32Array, values: Float32Array, options: CPURasterRegionGraphOptions | RasterRegionGraphOperationProps): CPURasterRegionGraphResult',
        supportedInputs: [
          'labels: Uint32Array',
          'values: Float32Array',
          'options: CPURasterRegionGraphOptions | RasterRegionGraphOperationProps'
        ],
        differences: [
          '/**\n * Builds four-neighbour adjacency and merges regions within the iteration budget. Fixed mode uses original\n * mean differences; adaptive mode repeatedly merges the closest adjacent component pair, with\n * lexicographic source-root ties and observation-count-weighted means.\n *\n * What differs from the GPU contributor:\n * - Accumulation, means and edge comparisons use float64, rounded only in output weights;\n *   threshold decisions near float32 rounding boundaries may differ.\n * - Both modes honor `maximumIterations`, including dense propagation and Boruvka phases.\n *   Auto uses dense propagation for small budgets and Boruvka when the budget guarantees closure.\n *   The CPU has no device-dependent sparse-scratch fallback; select dense explicitly to match it.\n * - The 256-region and 256-round GPU limits are dropped; the uint32 adjacency cell bound remains.\n *\n * @throws If labels/values do not cover the grid, or options are invalid.\n */',
          'The pure CPU function preserves the no-certificate RasterRegionGraphOperationProps compatibility branch. Supplied labelsComplete must be exactly one Uint32Array word, not a graph-value name.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterGeometricMedianComposite': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterGeometricMedianCompositeOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterGeometricMedianCompositeOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterGeometricMedianCompositeOnCPU',
        signature:
          'function computeRasterGeometricMedianCompositeOnCPU( values: Float32Array, parameters: Float32Array, options: CPURasterGeometricMedianCompositeOptions, mask?: Uint32Array ): CPURasterGeometricMedianCompositeResult',
        supportedInputs: [
          'values: Float32Array',
          'parameters: Float32Array',
          'options: CPURasterGeometricMedianCompositeOptions',
          'mask?: Uint32Array'
        ],
        differences: [
          "/**\n * Runs exactly `iterationCount` epsilon-clamped Weiszfeld passes from the valid-observation mean,\n * with the contributor's packed `[distanceEpsilon]` per-frame parameters and dense output layout.\n * A zero observation mask or any nonfinite band rejects the whole slice.\n *\n * What differs from the GPU contributor:\n * - Means, weights and updates use float64, rounded once in composite and lastStep outputs.\n *   The requested iteration count and epsilon clamp are retained as the documented method.\n * - GPU limits of 256 slices, 16 bands and 64 iterations are dropped; uint32 shape limits remain.\n *\n * @throws If values, mask or parameters are too short, or dimensions/iterationCount invalid.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterMedoidComposite': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterMedoidCompositeOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterMedoidCompositeOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterMedoidCompositeOnCPU',
        signature:
          'function computeRasterMedoidCompositeOnCPU( values: Float32Array, dimensions: CPUCompositeDimensions, mask?: Uint32Array ): CPURasterMedoidCompositeResult',
        supportedInputs: [
          'values: Float32Array',
          'dimensions: CPUCompositeDimensions',
          'mask?: Uint32Array'
        ],
        differences: [
          '/**\n * Selects the real observation with minimum total multiband Euclidean distance to all valid\n * observations, with earliest-slice ties. A zero mask or any nonfinite band rejects the whole slice.\n *\n * What differs from the GPU contributor:\n * - Distances and sums use float64; near ties and float32 distance overflow can select a different\n *   medoid. Output values are copied from the selected float32 observation.\n * - GPU limits of 64 slices and 16 bands are dropped; uint32 shape/indexing limits remain.\n *\n * @throws If values/mask are too short or dimensions are invalid.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterWatershedSegmentation': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterWatershedSegmentationOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterWatershedSegmentationOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterWatershedSegmentationOnCPU',
        signature:
          'function computeRasterWatershedSegmentationOnCPU( input: CPURasterOperationBand, markers: Uint32Array, options: CPURasterWatershedSegmentationOptions ): CPURasterWatershedSegmentationResult',
        supportedInputs: [
          'input: CPURasterOperationBand',
          'markers: Uint32Array',
          'options: CPURasterWatershedSegmentationOptions'
        ],
        differences: [
          "/**\n * Marker-controlled FIFO minimax priority flooding, or a rooted minimum-spanning forest. FIFO\n * ties follow discovery age, starting with row-major markers and row-major neighbour offsets.\n * Forest edges order by maximum endpoint elevation and then canonical edge ID.\n *\n * What differs from the GPU contributor:\n * - Calibration and costs use float64 without intermediate float32 rounding; nearly equal\n *   calibrated elevations can therefore order differently. Calibrated elevations beyond float32\n *   range can participate instead of becoming GPU nodata.\n * - Explicit `maximumSteps` bounds FIFO pops and `maximumIterations` bounds synchronous rooted\n *   Boruvka contractions, matching the GPU budgets. An incomplete run marks usable non-seeds\n *   unresolved (0xfffffffe), with `converged=0` and `iterationLimitReached=1`.\n * - GPU FIFO cell and forest round limits are dropped. The forest's uint32 edge-ID cell bound\n *   remains so canonical edge identities retain their documented representation.\n *\n * @throws If band values/validity do not exactly match the grid, markers do not cover it, or options are invalid.\n */"
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterReconstruction': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterReconstructionOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterReconstructionOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterReconstructionOnCPU',
        signature:
          'function computeRasterReconstructionOnCPU( marker: Float32Array, mask: Float32Array, options: CPURasterReconstructionOptions ): CPURasterReconstructionResult',
        supportedInputs: [
          'marker: Float32Array',
          'mask: Float32Array',
          'options: CPURasterReconstructionOptions'
        ],
        differences: [
          '/**\n * Computes bounded geodesic grayscale reconstruction. Invalid marker/mask pairs\n * (nonfinite, marker > mask for dilation, marker < mask for erosion) stay NaN and are barriers.\n * Only grid.width * grid.height marker and mask rows are read; surplus rows are ignored.\n *\n * What differs from the GPU contributor:\n * - The GPU command-node budget is not used. maximumIterations (default 64) still bounds\n *   output propagation. iterationCount includes the final verification pass; that pass does\n *   not advance the published values. Finite min/max results are exact float32 input values.\n * - Subnormal inputs are retained; GPU implementations may flush them to zero.\n *\n * @throws Error naming marker or mask for an inconsistent length, or an invalid option.\n */'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterFillHoles': {
    cpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'CPURasterFillHolesOperation'
    },
    gpu: {
      package: '@luma-contributors/analysis/raster-analysis',
      exportName: 'GPURasterFillHolesOperation'
    },
    arrays: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'computeRasterFillHolesOnCPU',
        signature:
          'function computeRasterFillHolesOnCPU( input: Uint32Array, options: CPURasterFillHolesOptions ): CPURasterFillHolesResult',
        supportedInputs: ['input: Uint32Array', 'options: CPURasterFillHolesOptions'],
        differences: [
          "/**\n * Floods zero-valued background from the boundary and fills every unreached component. All\n * nonzero input values are foreground and become 1, as in the contributor's reconstruction.\n * Only grid.width * grid.height input rows are read; any surplus rows are ignored.\n * The queue contains only cells within maximumIterations edges of the boundary. Frontier cells\n * inspect their immediate neighbors for the verification pass but never enqueue farther cells.\n *\n * What differs from the GPU contributor:\n * - The CPU uses shortest boundary-background path lengths instead of synchronous rounds.\n *   maximumIterations (default 64) bounds the flood distance just as on the GPU;\n *   iterationCount includes the verification pass, which does not advance the published flood.\n * - The GPU command-node budget is not used.\n *\n * @throws Error naming input for an inconsistent length, or an invalid option.\n */"
        ]
      }
    ]
  }
};
