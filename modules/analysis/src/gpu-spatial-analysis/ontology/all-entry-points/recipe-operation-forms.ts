// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUAnalysisOperator} from '../spatial-analysis-ontology';
/** Qualified analysis recipe stages with source conditions and invocation contract notes. */
export const RECIPE_OPERATION_FORMS: Readonly<Record<string, GPUAnalysisOperator['forms']>> = {
  '@luma-contributors/analysis/gpu-spatial-analysis:addHotSpotAnalysisRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULonLatPolygonPreparation',
        appliesWhen: "source.kind === 'points' && props.prepare"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellCover',
        appliesWhen: "source.kind === 'points'"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellAggregation',
        appliesWhen:
          "source.kind === 'points' OR source.kind === 'points' && source.family === 'quadbin' OR source.kind === 'points' && !(source.family === 'quadbin')"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointToCell',
        appliesWhen: "source.kind === 'points' && !(source.family === 'quadbin')"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellGeometry',
        appliesWhen: "source.kind === 'points'"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellMeasures',
        appliesWhen: "source.kind === 'points' && analyze === 'density' && source.family === 'h3'"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUNeighborSearch',
        appliesWhen: "source.kind === 'points' && distanceBand"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellTopology',
        appliesWhen: "source.kind === 'points' && !(distanceBand)"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULatticeWeights',
        appliesWhen: "!(source.kind === 'points')"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUHotSpotAnalysis',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULocalPermutationTest',
        appliesWhen: 'permutation'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUClassBreaks',
        appliesWhen: 'color'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUColorScale',
        appliesWhen: 'color'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addHotSpotAnalysisRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUHotSpotAnalysisRecipeProps ): GPUHotSpotAnalysisRecipeResult',
      '/**\n * Hot spot analysis recipe: points on a study area, or a raster, to Getis-Ord Gi* hot and cold\n * spots, optionally confirmed by conditional permutation and colored by class breaks.\n *\n * With `prepare: true`, `GPULonLatPolygonPreparation` precedes the study-area cover and adds\n * `study-area-preparation` diagnostics to `status.stages`.\n *\n * Chain (points source): `GPUCellCover` (study-area cells) -> `GPUCellAggregation` (sorted unique\n * study table) and `GPUPointToCell` (H3 only) -> `GPUCellAggregation` (sorted point table) -> a\n * join kernel (binary search of each study key in the point table; empty cells get 0) -> area\n * normalization (`GPUCellMeasures` for H3, the exact spherical rectangle for Quadbin) ->\n * `GPUCellTopology` disk -> binary-searched CSR weights -> `GPUHotSpotAnalysis` ->\n * `invalidCount` tally -> `GPULocalPermutationTest` -> `GPUClassBreaks` -> `GPUColorScale`.\n * A lattice source replaces the study and aggregation stages with `GPULatticeWeights` over the\n * dense raster.\n *\n * Statistical contract:\n * - Every study-area cell is a row, zeros included, so the global mean and deviation describe the\n *   whole area and clusters of empty cells can be cold spots. Weighted-point sum saturation\n *   publishes the exact aggregation sumOverflow view on the point stage and joined table,\n *   suppressing occupied sums with NaN while retaining counts and the study inclusion mask.\n *   Arithmetic saturation or upstream capacity loss excludes all rows from dependent inference,\n *   including fixed-moment inference. Joined candidateOverflow reports upstream table/cover loss\n *   separately from the destination overflow and sumOverflow words.\n * - Neighbors are topological (grid disk), never a metric radius in degrees.\n * - Rows with fewer than `minimumNeighbors` (default 8) neighbors are NaN, not a plain\n *   standardized value, and are outside the multiple-testing family.\n * - `multipleTesting` defaults to Benjamini-Hochberg. Bins `+-1/2/3` always mean 90/95/99%\n *   confidence; with correction, BH at q = 0.10/0.05/0.01. `result.inference` records this.\n *\n * Rows are study cells (`cellCapacity`) or lattice cells. Everything except the output views is\n * graph-owned; pass caller-owned views for the results you read back.\n *\n * Status: `status.stages` lists, in order, `study-area` (count, overflow, candidate overflow,\n * required count, rejected and latitude-limited features), `study-table` and `point-aggregation`\n * (count, overflow, required count, plus point sumOverflow for weighted points) for a points source;\n * `weights` (overflow, plus the unclamped neighbor total for a lattice or distance band and the dropped rows of a distance band);\n * `minimum-neighbors`; and with `permutation`, `permutation`.\n */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addRateClusterMapRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUEmpiricalBayesRates',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUContiguityWeights',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialWeightsTransform',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULocalMoran',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULocalPermutationTest',
        appliesWhen: 'props.permutation'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addRateClusterMapRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPURateClusterMapRecipeProps ): GPURateClusterMapRecipeResult',
      "/**\n * Rate cluster map recipe: counts and populations to significant local Moran clusters of\n * empirical-Bayes rates, colored by quadrant (esda `Moran_Local_Rate`).\n *\n * Chain: `GPUEmpiricalBayesRates` -> `GPUContiguityWeights` -> `GPUSpatialWeightsTransform`\n * (`'row'`, in place) -> `GPULocalMoran` -> `GPULocalPermutationTest` -> quadrant colors. The last\n * stage maps quadrants to palette colors. With `permutation`, `GPULocalMoran` runs with\n * `quadrantGating: 'none'` and the quadrant is kept only where the permutation `significant` mask\n * is nonzero; without it the analytic gate applies. Excluded rows (NaN rate, or islands) get\n * quadrant 0 and a NaN p-value: they are not part of the tested family, so they are never counted\n * as tested but not significant.\n *\n * Defaults: `multipleTesting` is `'benjamini-hochberg'` over all tested polygons and the\n * alternative is `'two-sided'`. Analytic inference (no `permutation`) is a preview; permutation\n * inference (999 permutations recommended) is the reference. The palette meaning is fixed (index 0\n * not significant, 1..4 the quadrants) whatever the correction; `result.inference` describes it.\n * `analyze: 'smoothed'` uses the clamped empirical-Bayes prior variance (`clampPriorVariance`).\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addSpatialRegressionRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUOrdinaryLeastSquares',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialRegressionDiagnostics',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULocalMoran',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUGeographicallyWeightedRegression',
        appliesWhen: 'local'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addSpatialRegressionRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUSpatialRegressionRecipeProps ): GPUSpatialRegressionRecipeResult',
      "/**\n * Spatial regression recipe: OLS, spatial dependence diagnostics on its residuals, a GPU-written\n * specification decision, a residual local Moran map, and optionally geographically weighted local\n * fits.\n *\n * Chain: `GPUOrdinaryLeastSquares` -> `GPUSpatialRegressionDiagnostics` (LM tests, residual\n * Moran) -> specification search (Anselin's rule, see {@link getSpatialRegressionSpecification}:\n * 0 OLS adequate, 1 spatial lag, 2 spatial error, 3 both robust tests significant, 4 undetermined,\n * at `significanceLevel`) and `GPULocalMoran` of the OLS residuals;\n * `GPUGeographicallyWeightedRegression` runs on the same inputs. A `mask` (or non-finite rows)\n * is handled end to end: the diagnostics analyze the fitted rows with `W` restricted to them\n * (`restandardizeRows` rescales the remaining weights), and `diagnostics.summary` reports the\n * excluded row count. The recipe stops at the decision and does not estimate the lag or error\n * model; follow the decision with `GPUSpatialTwoStageLeastSquares` or `GPUSpatialErrorGM`.\n *\n * Status: stages `ordinary-least-squares` and `spatial-diagnostics` publish the contributor's\n * `code` word as the stage's `code`. It is 0 on success and otherwise the contributor's failure\n * code, not a number of rows: read scalar `code` against 0. The compatibility alias\n * `status.invalidCount` retains the caller's original view, which may be longer than one word;\n * `code` exposes only word zero. Both refer to the same solver word at the same buffer offset,\n * but need not be the same view or length. Summing the alias does not count excluded rows;\n * use `diagnostics.summary` for that count. When local fits are requested, stage\n * `geographically-weighted-regression` additionally publishes `rowCodes` (one GWR `localStatus`\n * code per observation), with no scalar stage `code` or count alias.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addSpaceTimeHotSpotsRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUCalendarBuckets',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUPeriodIndex',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen: '!(!useGroupStatistics)'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUEmergingHotSpots',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUColorScale',
        appliesWhen: 'props.color'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addSpaceTimeHotSpotsRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUSpaceTimeHotSpotsRecipeProps ): GPUSpaceTimeHotSpotsRecipeResult',
      '/**\n * Space-time hot spot recipe: timestamped events to an ArcGIS-style emerging hot spot map.\n *\n * Chain: `GPUCalendarBuckets` decodes event times, `GPUPeriodIndex` turns the calendar columns into\n * a slice per event, an adapter kernel bins every event into\n * `cell * sliceCount + slice` of a zeroed dense `cell x slice` cube (`uint32`, the layout\n * `GPUEmergingHotSpots` reads) with an integer `atomicAdd`, `GPUEmergingHotSpots` classifies every\n * cell, and `GPUColorScale` (ordinal) colors the categories.\n *\n * Slices are absolute time steps (`slices.step`, `slices.interval`, `slices.origin`), so the\n * Mann-Kendall trend runs over a monotone time axis. Events outside the slice range are skipped\n * and counted in the `slice-range` stage `invalidCount`. `multipleTesting`, `finalSlice` and the\n * other options of `GPUEmergingHotSpots` apply as documented there, and the result carries a plain\n * `inference` object (`GPUInferenceMetadata`).\n *\n * Passing any of `scratch.keys`, `scratch.keyMask`, `scratch.groupKeys`, `scratch.groupCounts`,\n * `scratch.groupCount` or `outputs.status.groupOverflow` selects the sorted path instead: an adapter\n * writes the key per event, `GPUGroupStatistics` counts the events per key, and an adapter\n * scatters the sparse counts into the cube. Only that path fills those views, adds the\n * `GPUGroupStatistics` contributor and reports the `cube-counts` status stage. Both paths produce\n * the same cube, bit for bit. With `occupiedCellsOnly: true`, parallel event and cell passes derive\n * an ever-occupied mask directly from eligible events across every slice and intersect it with `cellMask`.\n * It is rebuilt each submission. Zero bins in those cells remain in the population, neighborhoods\n * and multiple-testing family; never-occupied and caller-excluded cells do not. Sorted group\n * truncation remains visible in cube-counts and excludes all cells from inference until a\n * complete submission; it does not alter the independently derived occupancy mask.\n *\n * The roadmap chain names `GPUCellAggregation` for the per-slice counts. Its keys must be valid\n * Quadbin or H3 cells and cannot carry a slice index, so the recipe counts with\n * `GPUGroupStatistics` over a lattice or caller-supplied cell index instead; the\n * `GPUPointToCell` + `GPUCellAggregation` route to a cube would need a cell-key to row lookup.\n */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addPeriodComparisonRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellAggregation',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUCellTableCompare',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUClassBreaks',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUColorScale',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addPeriodComparisonRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUPeriodComparisonRecipeProps ): GPUPeriodComparisonRecipeResult',
      "/**\n * Period comparison recipe: two periods of events to a classified, colored change map.\n *\n * Chain: two `GPUCellAggregation` tables -> `GPUCellTableCompare` (outer join with delta, ratio,\n * percent change and z-score) -> `GPUClassBreaks` of the chosen column -> `GPUColorScale`. The\n * compare node's `mask` output marks the unused tail of the capacity-bounded union table so it\n * never reaches the classification (the compare columns of the tail are undefined).\n *\n * Undefined values: `ratio` and `percentChange` are NaN where the earlier value is 0. The\n * `zScore` column is a descriptive Poisson-difference (or standardized-delta) score with no null\n * distribution or multiple-testing correction; it ranks change magnitude and must not be read as\n * significance. A zero score for a zero denominator is written by `GPUCellTableCompare` (see its\n * documentation); with `measure: 'count'` every union cell has `before + after > 0`, so the\n * Poisson denominator is never zero.\n *\n * Status: stages `before-aggregation`, `after-aggregation` and `comparison`, each with count,\n * overflow, sumOverflow and the unclamped required count. Sum saturation suppresses numeric\n * comparisons (NaN), so those rows receive the no-data color.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addPointsInPolygonsChoroplethRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULonLatPolygonPreparation',
        appliesWhen: 'props.prepare'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointInPolygonJoin',
        appliesWhen: "usesFeatureJoin OR !(usesFeatureJoin) && !(backend === 'zonal')"
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen:
          "usesFeatureJoin && !(statistic !== 'count' && queryCount === 0) && statistic !== 'count' && membership OR !(usesFeatureJoin) && !(backend === 'zonal')"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUZonalStatistics',
        appliesWhen: "!(usesFeatureJoin) && backend === 'zonal'"
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUClassBreaks',
        appliesWhen: 'color'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUColorScale',
        appliesWhen: 'color'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addPointsInPolygonsChoroplethRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUPointsInPolygonsChoroplethRecipeProps ): GPUPointsInPolygonsChoroplethRecipeResult',
      "/**\n * Points-in-polygons choropleth recipe: points to a per-polygon statistic to class breaks to\n * colors.\n *\n * With `prepare: true`, `GPULonLatPolygonPreparation` precedes containment, preserving feature\n * rows and publishing `polygon-preparation` diagnostics in `status.stages`.\n *\n * Chain: `GPUZonalStatistics` (or `GPUPointInPolygonJoin` -> dense `GPUGroupStatistics`) ->\n * `GPUClassBreaks` -> `GPUColorScale`. Features with no points keep NaN (mean, minimum, maximum)\n * or 0 (count, sum); NaN rows are no-data in the color scale and are skipped by the class breaks.\n * Counts feed the classification as the `uint32` column they are, with no cast.\n *\n * Spatial context: planar by default; `longitude-latitude` is accepted because counts and value\n * statistics are topological (see `spatialContext`). Density is not mapped, so no polygon area is\n * involved.\n *\n * Status: stage `zonal-statistics` (overflow, uncertain count), or `point-in-polygon` (overflow,\n * uncertain count) and `group-statistics` (count, overflow, required count) with `backend: 'group'`.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addClusterAndOutlineRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUDBSCAN',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUGroupGeometry',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUGroupConvexHull',
        appliesWhen: "props.outline?.kind !== 'alpha'"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUGeometryMeasures',
        appliesWhen: "props.outline?.kind !== 'alpha'"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUGroupConcaveHull',
        appliesWhen: "!(props.outline?.kind !== 'alpha')"
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addClusterAndOutlineRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUClusterAndOutlineRecipeProps ): GPUClusterAndOutlineRecipeResult',
      "/**\n * Cluster and outline recipe: DBSCAN clusters, per-cluster summaries, convex hulls and their\n * area, perimeter and centroid.\n *\n * Chain: `GPUDBSCAN` -> `GPUGroupGeometry` (counts, bounds, mean centers) and\n * `GPUGroupConvexHull` -> `GPUGeometryMeasures` with each hull as one single-ring polygon feature\n * (the hull's `offsets` is the ring offset list; feature and polygon offsets are the identity). With `minimumSamples = 1` the\n * clusters are PostGIS `ST_ClusterWithin` groups.\n *\n * Status: `status.stages` reports `clustering`, `convex-hull` and `hull-measures` (`status.invalidCount`\n * of rejected hull polygons).\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addChangeOfSupportRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULonLatPolygonPreparation',
        appliesWhen: 'props.prepare'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPUPolygonRasterization',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUArealInterpolation',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialLag',
        appliesWhen: 'props.sourceValues'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addChangeOfSupportRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUChangeOfSupportRecipeProps ): GPUChangeOfSupportRecipeResult',
      "/**\n * Change of support recipe: transfer values between two polygon zone systems by area.\n *\n * Chain: `GPUPolygonRasterization` (source and target to one raster) -> `GPUArealInterpolation`\n * (overlap areas, extensive and intensive weights) -> `GPUSpatialLag` over each weights set (the\n * lag treats the target-by-source weights as a cross matrix). Rasterization overflow flags are\n * reported separately in `status.rasterOverflow` and `status.rasterRequiredCount` (stages `source-rasterization` and\n * `target-rasterization`).\n *\n * Accuracy is bounded by the raster resolution (cell centers decide membership). Mass that a\n * source loses (it lies partly outside the target system, or it is smaller than a cell) is\n * reported per source in `sourceTransferredFractions` and `sourceAreas`, and sources with no\n * raster area are counted in `status.invalidCount` (stage `areal-interpolation`), unless\n * `sourceAnchorCells` rescues them. Default change: intensive weights now use the `'overlap'`\n * denominator, so a target only partly covered by sources is the mean of what covers it.\n *\n * With `prepare: true`, each zone system is prepared before rasterization; feature rows are\n * preserved and `source-preparation` / `target-preparation` diagnostics join `status.stages`.\n * Spatial context: `spatialContext` is resolved once and handed to the areal stage; see\n * `GPUChangeOfSupportRecipeProps.spatialContext`.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addStraightLineCatchmentsRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPUDistanceField',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterZonalStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addStraightLineCatchmentsRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUStraightLineCatchmentsRecipeProps ): GPUStraightLineCatchmentsRecipeResult',
      "/**\n * Straight-line catchments recipe: nearest-facility zones on a raster and what falls in each.\n *\n * Chain: `GPUDistanceField` (Euclidean allocation, i.e. raster Voronoi) ->\n * `GPURasterZonalStatistics` over a value raster, with the allocation as the zone raster. Cells\n * with no facility within `maximumDistance` carry `0xffffffff` and are ignored by the statistics.\n *\n * Spatial context: planar only (see `GPUStraightLineCatchmentsRecipeProps.spatialContext`).\n * Distances are in the context's length unit, meters for a `createGPULocalFrame` context.\n *\n * Status: stage `zonal-statistics` reports `overflow` (a zone id past the zone capacity).\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addFleetDwellRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUTrajectoryMetrics',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointInPolygonJoin',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addFleetDwellRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUFleetDwellRecipeProps ): GPUFleetDwellRecipeResult',
      "/**\n * Fleet dwell recipe (stops variant): where do vehicles stop and for how long, per zone.\n *\n * Chain: `GPUTrajectoryMetrics` stops -> adapter (mask and join positions restricted to the\n * GPU-written stop count) -> `GPUPointInPolygonJoin` of the active stop centroids against the zones ->\n * dense `GPUGroupStatistics` (`keyCount` = zone rows) of stop duration keyed by zone, one row per\n * zone. Stops outside every zone carry the reserved no-zone key and are skipped.\n *\n * Spatial context: the recipe resolves one planar `spatialContext` (default native planar; a\n * `createGPULocalFrame` context for meters) and passes the same object to the metrics, join and\n * zone stages. Previously it always emitted a native planar context to the metrics stage regardless\n * of the caller's data. Stop durations use the timestamps' unit; the metrics stage speed threshold\n * uses context length units per time unit. Longitude-latitude throws.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addFleetDwellZoneEventsRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUZoneEvents',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addFleetDwellZoneEventsRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUFleetDwellZoneEventsRecipeProps ): GPUFleetDwellZoneEventsRecipeResult',
      "/**\n * Fleet dwell recipe (zone events variant): enter and exit events of tracks against zones, dwell\n * per (track, zone) and a per-zone roll-up.\n *\n * Chain: `GPUZoneEvents` -> adapter (zone key and visited mask per `(track, zone)` cell) ->\n * dense `GPUGroupStatistics` of dwell time keyed by zone, one row per zone. Unlike the stops variant this counts any time\n * spent inside a zone, moving or not, and needs no point-in-polygon join. `table.counts` is the\n * number of tracks that visited the zone. `events.eventPositions` and `visitTable` are optional\n * pass-through outputs of `GPUZoneEvents` (crossing positions and a sparse `(track, zone)` table).\n *\n * Status: stages `zone-events` (event count, unclamped `requiredCount`, event overflow and\n * `candidateOverflow`; the unclamped candidate count is `status.candidateCount`),\n * `visit-table` (with `scratch.visitTable`) and `zone-statistics`. The statistics stage's\n * `candidateOverflow` reports candidate scratch loss only: its dwell statistics are incomplete\n * then. Per-track and final event-list truncation leave dwell statistics exact.\n * The optional visit-table stage publishes its supplied candidate-loss flag, or the same scratch\n * flag when omitted, independently of visit-table output capacity.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addZonalDistributionRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointInPolygonJoin',
        appliesWhen: "source.kind === 'points'"
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addZonalDistributionRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUZonalDistributionRecipeProps ): GPUZonalDistributionRecipeResult',
      "/**\n * Assigns each point to one feature (the join's deterministic ownership rule), then computes\n * unweighted zone distributions. Prepared zone rows support raster cells or already expanded\n * overlapping memberships. Empty zones retain their ID, zero counts and NaN order statistics.\n * Raster callers must calibrate and apply validity before supplying values.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addInterpolationDiagnosticsRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUKriging',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addInterpolationDiagnosticsRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUInterpolationDiagnosticsRecipeProps ): GPURecipeResult<GPUInterpolationDiagnosticsOutput>',
      '/** Ordinary kriging and leave-one-out diagnostics, with no hidden fitting or readback. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addRasterZonalDistributionRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointInPolygonJoin',
        appliesWhen: "source.kind === 'points'"
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addRasterZonalDistributionRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPURasterZonalDistributionRecipeProps )',
      '/** Whole-cell, unweighted distributions, with native-band validity and calibration applied first. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addExposureByZoneRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterZonalStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addExposureByZoneRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUExposureByZoneRecipeProps )',
      '/** Denominators retain population with missing hazards; observed coverage is returned separately. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addRegionalRasterChangeRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPUChangeDetection',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterZonalStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addRegionalRasterChangeRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPURegionalRasterChangeRecipeProps )',
      '/** Two-period raster differences over common support, with independently reported coverage. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addSurfacePreparationRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterFillNoData',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterRankFilter',
        appliesWhen: 'props.filter'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      "export function addSurfacePreparationRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUSurfacePreparationRecipeProps ): GPURecipeResult<GPUSurfacePreparationOutput> & {band: GPURasterBufferBand<'float32'>}",
      '/** Continuous-surface repair with original validity and independent fill/filter provenance. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addDrainageNetworkRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'GPUTerrainFlow',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'GPUTerrainStreamOrder',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'GPUTerrainWatersheds',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'GPUTerrainFlowDistance',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'GPUTerrainStreamNetwork',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addDrainageNetworkRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUDrainageNetworkRecipeProps )',
      "/** Builds a drainage network on one complete D8 domain. Downstream distances, reach lengths and mean downstream distances use affine units by default and ground metres when `flow.cellSizeMode` is 'geographic' or 'web-mercator'. Graph-owned outputs can feed rendering or explicit readback copies. */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addLocalVariabilityRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULocalSpatialHeteroscedasticity',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPULocalMoran',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addLocalVariabilityRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPULocalVariabilityRecipeProps ): GPULocalVariabilityRecipeResult',
      "/**\n * Compare local mean association with local dispersion on the same explicit graph and complete\n * input mask. LOSH is descriptive; Moran's analytic-normal p-values and quadrants remain separate.\n * Both exclude islands. No combined significance, dispersion class or permutation p-value is\n * inferred. Upstream overflow and stage-specific invalid counts are retained, with per-row codes.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addCellFlowTopologyRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUH3CellIdentities',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addCellFlowTopologyRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUCellFlowTopologyRecipeProps )',
      '/** R11: preserves the original OD table and separately aggregates actual neighboring-cell flows.\n * Cell totals include every valid OD pair (including self and nonneighbor pairs). Edge totals\n * include only immediate H3 neighbors; edge sums can therefore be smaller. Each accumulator\n * sums valid rows in input order with sequential float32 rounding. Regrouping by origin or\n * destination changes rounding, so sums of the cell totals need not equal each other or the\n * valid-pair sum exactly. No route inference is performed. Each direct-scan invocation examines\n * at most GPU_CELL_FLOW_TOPOLOGY_MAXIMUM_PAIRS_PER_PASS = 4096 pairs per pass, per cell or edge\n * slot. Larger tables use ordered passes carrying the previous float32 accumulator, with no\n * total pair cutoff. Aligned pair subviews let full passes share a shader, with separate output\n * initialization. Work remains O(cellCount * pairCount), including six edge-slot scans per\n * cell; preaggregate pairs when that cost is too large. H3 keys are exact;\n * shared-boundary coordinates inherit the existing f32 H3 geometry precision limits.\n */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addDasymetricRedistributionRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterFractionalCoverage',
        appliesWhen: "props.source.kind === 'polygons'"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUArealInterpolation',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialLag',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addDasymetricRedistributionRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUDasymetricRedistributionRecipeProps )',
      '/**\n * Fractional source support -> ancillary eligibility -> weighted areal interpolation -> mass ledger.\n * Targets own complete raster cells: independent source/target fractions are never multiplied.\n * Source features may overlap: their totals are independent and targetSupportWeights add their\n * measures, not their geometric union. Prepared coverage must contain unique (source, cell) rows,\n * with no repeated halo cells. Uniqueness is a caller precondition, not certified by complete. Concatenate disjoint partition cores BEFORE normalizing; independently\n * normalized tile totals cannot be added. Source totals and all input buffers stay caller-owned.\n *\n * Zero eligible support assigns zero and retains the entire source total. Missing/negative ancillary\n * samples carry no support and remain visible as NaN in intermediates.cellWeights. Incomplete\n * coverage, invalid tuples/source totals, arithmetic overflow or pair overflow invalidate all published numerical outputs;\n * the original per-stage evidence remains available. No CPU readback or fallback is performed.\n */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addExcessRiskRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUEmpiricalBayesRates',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      "export function addExcessRiskRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUExcessRiskRecipeProps ): GPURecipeResult<{excessRisk: GraphDataView<'float32'>}>",
      '/**\n * Appends observed/expected excess risk E_i / (P_i * sum(E) / sum(P)).\n * Uses the existing empirical-Bayes reductions without changing its other outputs. Zero or\n * negative expected counts produce NaN; events must be finite and nonnegative and populations\n * finite and positive. The optional mask excludes rows from both totals and outputs.\n * No rate scale applies: the result is dimensionless. Inputs are packed graph views, not vectors\n * to concatenate; caller owns imported chunks. No submission or readback occurs.\n */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addSpatialRateRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialEmpiricalBayesRates',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      "export function addSpatialRateRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUSpatialRateRecipeProps ): GPURecipeResult<{spatialRates: GraphDataView<'float32'>}, {pooledRates: GraphDataView<'float32'>}>",
      "/**\n * Appends scale * (E_i + sum neighbors E) / (P_i + sum neighbors P), never an average of rates.\n * Included neighbors have finite nonnegative events, finite positive populations, nonzero mask\n * and positive stored weight (magnitude is ignored); self edges\n * are skipped because self is already included once. Duplicate nonself slots count repeatedly.\n * Islands (no included nonself neighbor) produce NaN by default, matching GeoDa's undefined flag.\n * The underlying spatial EB contributor retains its own-rate island behavior unchanged.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addContiguityOrderCorrelogramRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialWeightsAlgebra',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialWeightsTransform',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUGlobalSpatialStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addContiguityOrderCorrelogramRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUContiguityOrderCorrelogramRecipeProps ): GPUContiguityOrderCorrelogramRecipeResult',
      "/**\n * Spatial correlogram over contiguity orders: global Moran's I with analytic inference for the\n * neighbors at exactly 1, 2, ..., `maximumOrder` steps, each with row-standardised weights\n * (`libpysal` `higher_order` rings and `esda.Moran`). It composes `GPUSpatialWeightsAlgebra`\n * (`higherOrder`), `GPUSpatialWeightsTransform` (row standardisation) and\n * `GPUGlobalSpatialStatistics` once per order, so every limit of those applies: the algebra's\n * frontier work grows with the squared degree, and an order whose ring needs more than `slotCapacity`\n * slots is reported in `outputs.incompleteOrders` and its row is not valid.\n *\n * Rows with no neighbor at an order are islands and are excluded from that order's `n`.\n * Reference: a binary64 breadth-first ring construction and Moran's I with the same island rule\n * (libpysal and esda were not installed for this item).\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addStreetProfilesRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addStreetProfilesRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUStreetProfilesProps )',
      '/** Side-aware first-building profiles; missing sides remain range-censored and never become observed walls. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addBuildingAdjacencyRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPolygonOverlay',
        appliesWhen: 'props.structures'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addBuildingAdjacencyRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUBuildingAdjacencyProps )',
      '/** Shared-wall adjacency and axial alignment from the existing shared-path topology contributor. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addIsovistsRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addIsovistsRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUIsovistsProps )',
      '/** Angularly sampled obstacle isovist polygons, counterclockwise from +X, with an explicit circular range boundary. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addVisibilityGraphRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addVisibilityGraphRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUVisibilityGraphProps | GPUVisibilityGraphAdjacencyCSRProps )'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addBuildingTessellationRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPolygonTriangulation',
        appliesWhen:
          "props.mode === 'local' || ('candidateRadius' in props && props.mode !== 'reference') && props.enclosureMode === 'simple'"
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPolygonOverlay',
        appliesWhen: 'always OR simpleEnclosure'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addBuildingTessellationRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUBuildingTessellationProps | GPUBuildingTessellationLocalProps )'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addBuildingTessellationReferenceRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPolygonOverlay',
        appliesWhen: 'always OR simpleEnclosure'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addBuildingTessellationReferenceRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUBuildingTessellationProps )',
      "/**\n * Boundary-sampled reference Voronoi fragments dissolved per building, with an optional simple\n * enclosure clip. This route accepts at most 64 buildings, 32 total enclosure vertices and 128\n * total boundary samples per call. Every overlay must also fit the caller device's storage and\n * vertex limits; the complete declaration is checked before the caller graph is mutated.\n * The legacy dispatcher selects this same typed reference entry.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addLocalBuildingTessellationRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPolygonTriangulation',
        appliesWhen: "props.enclosureMode === 'simple'"
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addLocalBuildingTessellationRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUBuildingTessellationLocalProps )',
      '/**\n * Bounded, building-parallel sampled Voronoi tessellation for planar coordinates and convex\n * or simple (possibly holed) enclosures. Simple enclosures are triangulated once, and each\n * sample cell is intersected with every bounded enclosure triangle. One clipping dispatch handles\n * every building, regardless of building count.\n * Per-frame: input contents and optional parameters (spacing, radius, tolerance). Compile-time: lengths and capacities.\n * The legacy overlay recipe remains the reference for dissolved shell/hole rings.\n * Per call: at most 1,000,000 buildings, 16,777,216 total samples, 32 total enclosure vertices\n * and 32 enclosure rings. Per building: at most 32 own samples and 256 candidate samples.\n * Per convex piece: at most 128 vertices. Each indexed query attempt visits at most 8,192 rows.\n * Device storage limits may lower these bounds. The named entry and legacy local overload share\n * this implementation and the sample-cell-union output representation.\n * The legacy dispatcher selects this same typed local entry.\n */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addLandscapeMetricsRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterPatchMetrics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addLandscapeMetricsRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPULandscapeMetricsProps )',
      '/** Named class/landscape metrics from a complete, already labelled categorical raster. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addPatchProximityRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addPatchProximityRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUPatchProximityProps | GPUPatchProximityBoundaryProps )'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addResistanceConnectivityRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addResistanceConnectivityRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUResistanceConnectivityProps )',
      "/**\n * Bounded small-network grounded-Laplacian CG solve, with independently recomputed current residual.\n *\n * @deprecated Use {@link addSparseResistanceConnectivityRecipe}. This recipe solves the whole\n * system in one GPU thread, so it is limited to 256 vertices and 65,536 edges. The sparse recipe takes the same edges, conductances,\n * source and sink and also reports per-vertex currents; it is parallel, and its limits are 1,048,576\n * vertices and 2,097,152 edges. Its default tolerance is 1e-4 (this recipe's is 1e-5) and its\n * iterations are capped by `maximumIterations` (default 1,024; this recipe's is 256).\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addSparseResistanceConnectivityRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addSparseResistanceConnectivityRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUSparseResistanceConnectivityProps )',
      '/**\n * Sparse grounded-Laplacian Jacobi-PCG resistance solve with unit current injection, component\n * masking, an independent current-conservation residual and per-edge and per-vertex currents.\n *\n * Outputs: `potentials` (vertexCount), `currents` (edgeCount, signed first to second endpoint),\n * `vertexCurrents` (vertexCount, Circuitscape-style node current) and `diagnostics`\n * `[effectiveResistance, maximumTrueResidual, dissipatedEnergy, recursiveResidualNorm]`.\n * `status` is `[converged, iterations, reachable, invalidEdges]`, where `reachable` is 1 when the\n * source and sink share a component, 0 when they provably do not and 2 when label propagation did\n * not converge within `connectivityIterations`.\n */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addSegregationDecompositionRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addSegregationDecompositionRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUSegregationDecompositionProps )',
      '/** Two-factor Shapley decomposition of the dissimilarity-index change, without randomization inference. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addSegregationCounterfactualRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addSegregationCounterfactualRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUSegregationCounterfactualProps )',
      '/** Generates both replacement orders before the existing Shapley decomposition, without host readback. */'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addPopulationSupportAlignmentRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPopulationSupportAlignment',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addPopulationSupportAlignmentRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUPopulationSupportAlignmentRecipeProps )',
      "/**\n * Exact ID alignment. Prefer the CPU operation when support data is already on the host.\n * The alignment stage certifies a one-to-one support match with no excluded sources.\n * The missing-targets, ambiguous-targets and excluded-sources stages report invalidCount.\n * Each scalar aliases its word in the contributor's status without additional commands.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addRasterResistanceConnectivityRecipe': {
    recipeStages: [],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addRasterResistanceConnectivityRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPURasterResistanceConnectivityProps ): GPURasterResistanceConnectivityRecipeResult',
      "/**\n * Circuit connectivity across a resistance raster, for one source and one sink. Each cell is a\n * vertex; each pair of neighbouring cells with data is an edge with conductance\n * `2 / (R(a) + R(b)) / edgeLength`, where `edgeLength` is the ground distance between the cell\n * centres (the diagonal length for diagonal neighbours) under `cellSizeMode`. Resistance is\n * therefore per unit ground length, and conductance is per unit ground length too. The edge list\n * is built on the GPU, one invocation per edge slot, then solved by\n * {@link addSparseResistanceConnectivityRecipe} with unit current from `source` to `sink`.\n *\n * Relation to Circuitscape's raster pairwise mode: this is not the same convention. Circuitscape\n * (average-resistance connection) uses `1 / mean(R)` for cardinal neighbours and\n * `1 / (sqrt(2) * mean(R))` for diagonal ones and never divides by the cell size. On a planar grid\n * of square cells of size `s` (`'uniform'` mode) every conductance here is Circuitscape's divided\n * by `s`: the current map is the same, and potentials and effective resistance are `s` times\n * Circuitscape's (two cells with `R = 1` and `s = 30` give 30, Circuitscape gives 1). With\n * rectangular cells, and in `'geographic'` and `'web-mercator'` mode where the edge length changes\n * with the row, the conductance ratios differ and so does the current map.\n *\n * Edge slots have a fixed layout, `cell * slotsPerCell + direction`, with directions east and south\n * (plus south-east and south-west for `neighborhood: 8`); a slot whose neighbour is off the grid is\n * a self edge of zero conductance, and a slot touching a no-data cell has zero conductance.\n *\n * No-data: zero validity, the band's no-data value or a non-finite resistance (including\n * +Infinity) removes the cell; its potential and current are NaN. Zero or negative resistance with\n * data is invalid: such cells are counted in stage `resistance` (`invalidCount`), each edge they\n * share with a cell with data gets a NaN conductance, and the sparse solve then reports those\n * edges in stage `solve` (`invalidCount`) and publishes no potentials. A positive resistance is\n * invalid in the same way when an edge between it and another positive resistance has a\n * conductance that is not finite in float32 (for example `R = 1e-35` on both sides of a `1e-4`\n * cell): the edge gets a NaN conductance and both cells are counted, so `invalidCount` of stage\n * `resistance` is nonzero whenever stage `solve` reports invalid edges. Clamp resistances to a\n * positive minimum (or mark them no-data) before calling.\n *\n * Limits: at most 1,048,576 cells and at most 2,097,152 edge slots (524,288 cells with eight\n * neighbours), the limits of the sparse solve. Compile-time: the grid size, `neighborhood`,\n * `cellSizeMode`, `source`, `sink`, `tolerance` and the iteration budgets. Per-frame: `parameters`\n * (the grid georeference) and the resistance values.\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-network:addDriveTimeCatchmentRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkSnapping',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkServiceAreas',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkIsochrones',
        appliesWhen: 'isochrones'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointInPolygonJoin',
        appliesWhen: 'isochrones?.joinDemand && polygons'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUGroupStatistics',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addDriveTimeCatchmentRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUDriveTimeCatchmentRecipeProps ): GPUDriveTimeCatchmentRecipeOutput',
      "/**\n * Drive-time catchment recipe: which demand is within each drive-time band of the facilities.\n *\n * Chain: `GPUNetworkSnapping` (facilities) -> start-row adapter -> `GPUNetworkServiceAreas`\n * (multi-source costs) -> optional `GPUNetworkIsochrones` (isoband triangles, or cell-outline\n * rings) -> `GPUNetworkSnapping` (demand) -> adapter kernels (demand time, then the band) ->\n * `GPUGroupStatistics` keyed by band.\n *\n * Roads and direction: a snapped point lies at fraction `f` of the straight road between the two\n * end nodes of its edge. Driving that road from one end to the other costs the weight of the\n * cheapest passable edge going that way, and is not possible when the topology has no such edge.\n * So a facility starts at the edge target with cost `(1 - f) * w(source -> target)` and at the\n * edge source with cost `f * w(target -> source)`, and a one-way edge gives one interior start row; an endpoint coincident with the facility\n * is always seeded at zero cost. The\n * demand time is the smallest of `cost[source] + f * w(source -> target)`,\n * `cost[target] + (1 - f) * w(target -> source)`, and, for every facility snapped to the same\n * road, the way straight along it: `(f - f_facility) * w(source -> target)` when the demand is\n * ahead of the facility and `(f_facility - f) * w(target -> source)` when it is behind. On a\n * two-way road of one cost that last term is `abs(f - f_facility) * w`. The facilities are\n * sorted by road (the published `GPUSort`, keyed by the smallest edge row between the two end\n * nodes) and each demand point binary-searches its own road, so it is compared only with the\n * facilities snapped to that road: no demand point scans every facility, and the facility count\n * needs no cap. A demand point coincident with an endpoint uses that node cost without\n * traversing a direction, even if the opposite road direction is unavailable.\n *\n * Bands are classified in network space (exact at the snapped edge, no polygon approximation):\n * isoband output has a GPU-written, data-dependent triangle count that `GPUPointInPolygonJoin`\n * cannot take as polygons. The cell path produces closed rings (`GPUSegmentRingAssembly`) whose\n * polygon layout does fit the join: its offsets are GPU-written over fixed-length views, so\n * `isochrones.joinDemand` joins the demand points against the isochrone at `cellCostLimit` in the\n * coordinates of the facilities and demand (longitude/latitude). That polygon is the union of\n * cells holding reached nodes, a coarser answer than the network-space bands. The ring edges are\n * axis-aligned cell edges in degrees, so the join evaluates containment in the degree plane exactly\n * and is constructed with the native planar context whatever `spatialContext` says.\n *\n * Spatial context: see `GPUDriveTimeCatchmentRecipeProps.spatialContext`. Costs are in the units\n * of `topology.weights`; distances (`maximumSnapDistance`) are in the context's length unit.\n *\n * Status: `status.stages` reports `snap-facilities` and `snap-demand` (candidate overflow, with\n * `candidateCapacity`), `service-areas` (convergence), the isochrone stages for the caller-owned\n * isochrone outputs that carry status (`isochrone-bands`, `isochrone-band-edges`,\n * `isochrone-band-rings` on the raster path; `isochrone-cells`, `isochrone-outline`,\n * `isochrone-rings` on the cell path), `band-statistics` (count, overflow, required count) and,\n * with a demand join, `isochrone-demand`. The service-area `assignments` are handed to\n * `GPUNetworkIsochrones`, so `raster.output.pixelFacilities`, `triangleFacilities` and\n * `cellOutline.byFacility` work through the recipe; facility rows there are start rows\n * (`2 * facility + endpoint`).\n */"
    ]
  },
  '@luma-contributors/analysis/gpu-network:addNetworkDisruptionRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkCostMatrix',
        appliesWhen: '!props.baseline OR always'
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkAccessibility',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addNetworkDisruptionRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUNetworkDisruptionRecipeProps ): GPUNetworkDisruptionRecipeResult',
      '/** Compare directed routing scenarios without compaction or Infinity - Infinity arithmetic. */'
    ]
  },
  '@luma-contributors/analysis/gpu-network:addFacilityScenarioRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkAccessibility',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addFacilityScenarioRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUFacilityScenarioRecipeProps ): GPURecipeResult<GPUFacilityScenarioOutput>',
      '/** Evaluate one live facility scenario against a reusable matrix. Selection is a separate objective. */'
    ]
  },
  '@luma-contributors/analysis/gpu-network:addTrackQualityRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUSpatialJoinPrepared',
        appliesWhen:
          'view instanceof GPUSpatialJoinPrepared OR view instanceof GPUSpatialJoinPrepared && view instanceof GPUSpatialJoinPrepared'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUTrajectoryMetrics',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUTrackSimilarity',
        appliesWhen: 'props.similarity && similarity'
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUMapMatching',
        appliesWhen: 'props.mapMatching && matching'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addTrackQualityRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUTrackQualityRecipeProps )',
      '/** Raw quality flags, optional route distances and optional map-matching completeness. */'
    ]
  },
  '@luma-contributors/analysis/gpu-network:addVisibilityGraphRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkCentrality',
        appliesWhen: "props.mode === 'csr'"
      },
      {
        package: '@luma-contributors/analysis/gpu-network',
        exportName: 'GPUNetworkReachability',
        appliesWhen: "props.mode === 'csr'"
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addVisibilityGraphRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUVisibilityGraphProps | GPUNetworkVisibilityGraphCSRProps )'
    ]
  },
  '@luma-contributors/analysis/raster-analysis:addRasterRegionGraphTiles': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPURasterRegionGraph',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addRasterRegionGraphTiles<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPURasterRegionGraphTilesProps ): GPURasterRegionGraph & {readonly estimatedRecipeCommandNodeCount: number}',
      '/** Explicit GPU assembly of a complete, nonoverlapping core partition followed by region merging.\n * The caller supplies equivalence IDs (e.g. from cross-tile components). Shared source IDs merge\n * observations before computing means; all inter-core boundary edges are counted exactly once.\n * No host readback or implicit label reconciliation occurs. */',
      'estimatedRecipeCommandNodeCount = inner plan.estimatedCommandNodeCount + tiles.length assembly commands + (certificateCount > 0 ? 1 + certificateCount : 0) certificate commands. The returned readonly recipe estimate includes assembly and certificate work; the inner contributor estimate covers only region-graph work.'
    ]
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:addStopValueCubeRecipe': {
    recipeStages: [
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUPeriodIndex',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'GPUPointToCell',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUKeyLookup',
        appliesWhen: 'always'
      },
      {
        package: '@luma-contributors/analysis/dataframe-analysis',
        exportName: 'GPUValueCube',
        appliesWhen: 'always'
      }
    ],
    recipeImplementation:
      'Analysis contributor stages reachable through first-party function calls; conditions are source expressions in their declaring function, not evaluated caller predicates. Inline kernels and upstream primitives are not separate analysis identities.',
    contractNotes: [
      'export function addStopValueCubeRecipe<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPUStopValueCubeRecipeProps )',
      '/** Bins live stops against a stable dictionary. Empty observed bins are zero. Loss suppresses the cube. */',
      'GPUPeriodIndex bins signed Int64 source timestamps selected by start/finish rows, GPUPointToCell indexes the live stop prefix, GPUKeyLookup maps cell keys to dictionary row order, and GPUValueCube counts cell-major periods. Inline glue selects timestamps and combines stop, slice and weights loss; loss suppresses counts. These glue kernels are not separate contributor identities. Generic preflight prepared-handle copying cannot apply to this recipe and is not a stop-cube stage.'
    ]
  }
};
