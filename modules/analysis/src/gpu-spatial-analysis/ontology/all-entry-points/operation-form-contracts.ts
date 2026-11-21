// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUAnalysisFormContract} from '../spatial-analysis-ontology';
/** Source invocation contracts. Unknown is undisclosed, never proof that options/limits are absent.
 * Expressions are scoped to their source declarations and are not executable admission plans. */
export const OPERATION_FORM_CONTRACTS: Readonly<Record<string, GPUAnalysisFormContract>> = {
  '@luma-contributors/analysis/gpu-spatial-analysis:computePettittTestOnCPU': {
    signature:
      'export function computePettittTestOnCPU( values: Float32Array, options: CPUPettittTestOptions ): CPUPettittTestResult',
    supportedInputs: ['values: Float32Array', 'options: CPUPettittTestOptions'],
    defaults: 'unknown',
    limits: [
      '- Accepts up to 131071 slices instead of 512; the retained limit protects the uint32 statistic.',
      "validateCapacity('computePettittTestOnCPU', 'cellCount', cellCount)",
      "validateCapacity( 'computePettittTestOnCPU', 'sliceCount', sliceCount, 1, CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT )",
      'values.length !== cellCount * sliceCount'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUPettittTestOperation': {
    signature: 'new CPUPettittTestOperation(props: PettittTestOperationProps)',
    supportedInputs: ['props: PettittTestOperationProps'],
    defaults: ["props.id ?? 'pettitt-test'"],
    limits: [
      "/** CPU twin; the GPU's 512-slice private-array limit is replaced by the uint32 output limit. */",
      "validateCapacity(id, 'cellCount', props.cellCount)",
      "validateCapacity(id, 'sliceCount', props.sliceCount, 1, CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT)"
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPettittTestOperation': {
    signature: 'new GPUPettittTestOperation(props: PettittTestOperationProps)',
    supportedInputs: ['props: PettittTestOperationProps'],
    defaults: ["props.id ?? 'pettitt-test'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeLocationChangePointsOnCPU': {
    signature:
      'export function computeLocationChangePointsOnCPU( values: Float32Array, parameters: Float32Array, options: CPULocationChangePointsOptions ): CPULocationChangePointsResult',
    supportedInputs: [
      'values: Float32Array',
      'parameters: Float32Array',
      'options: CPULocationChangePointsOptions'
    ],
    defaults: 'unknown',
    limits: [
      "- The GPU's 512-slice, 256-slope-slice and 128-change capacity limits are dropped; uint32 shape",
      'values.length !== cellCount * sliceCount',
      'parameters.length < GPU_LOCATION_CHANGE_POINTS_PARAMETER_LENGTH',
      '!Number.isFinite(penalty) || penalty < 0 || !Number.isFinite(packedMinimumLength)'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPULocationChangePointsOperation': {
    signature: 'new CPULocationChangePointsOperation(props: LocationChangePointsOperationProps)',
    supportedInputs: ['props: LocationChangePointsOperationProps'],
    defaults: ["props.id ?? 'location-change-points'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocationChangePointsOperation': {
    signature: 'new GPULocationChangePointsOperation(props: LocationChangePointsOperationProps)',
    supportedInputs: ['props: LocationChangePointsOperationProps'],
    defaults: ["props.id ?? 'location-change-points'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePointToCellOnCPU': {
    signature:
      'export function computePointToCellOnCPU( positions: Float32Array, options: CPUPointToCellOptions, mask?: Uint32Array ): CPUPointToCellResult',
    supportedInputs: [
      'positions: Float32Array',
      'options: CPUPointToCellOptions',
      'mask?: Uint32Array'
    ],
    defaults: [
      '- Non-finite rows produce zero keys. With family-default, H3 also returns zero for magnitudes at least 3e38,',
      "/** Prefix for generated node IDs. Defaults to `'point-to-cell'`. */",
      '/** H3 arithmetic: float32 by default, or double-single (hi/lo float32 pairs).',
      "/** Longitude policy, default 'family-default'; 'wrap' uses [-180, 180) degrees for every family. */",
      'Coordinate contract of `positions`. Defaults to, and must be, longitude/latitude without a',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: ['positions.length % 2 !== 0', 'mask !== undefined && mask.length !== rowCount'],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Nothing differs for Quadbin, quadkey or geohash keys: their integer-defined cell methods'
    ],
    source: 'modules/analysis/src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUPointToCellOperation': {
    signature: 'new CPUPointToCellOperation(props: PointToCellOperationProps)',
    supportedInputs: ['props: PointToCellOperationProps'],
    defaults: [
      "/** Prefix for generated node IDs. Defaults to `'point-to-cell'`. */",
      '/** H3 arithmetic: float32 by default, or double-single (hi/lo float32 pairs).',
      "/** Longitude policy, default 'family-default'; 'wrap' uses [-180, 180) degrees for every family. */",
      'Coordinate contract of `positions`. Defaults to, and must be, longitude/latitude without a',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      'positions.length !== props.rowCount * 2'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPointToCellOperation': {
    signature: 'new GPUPointToCellOperation(props: PointToCellOperationProps)',
    supportedInputs: ['props: PointToCellOperationProps'],
    defaults: [
      "props.id ?? 'point-to-cell'",
      'props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT',
      "/** Prefix for generated node IDs. Defaults to `'point-to-cell'`. */",
      '/** H3 arithmetic: float32 by default, or double-single (hi/lo float32 pairs).',
      "/** Longitude policy, default 'family-default'; 'wrap' uses [-180, 180) degrees for every family. */",
      'Coordinate contract of `positions`. Defaults to, and must be, longitude/latitude without a',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: ["validateCapacity(id, 'rowCount', props.rowCount, 0)"],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/cell-indexing/gpu-point-to-cell-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeCellMeasuresOnCPU': {
    signature:
      "export function computeCellMeasuresOnCPU( cells: Uint32Array, options: CPUCellMeasuresOptions = {family: 'h3'} ): CPUCellMeasuresResult",
    supportedInputs: ['cells: Uint32Array', "options: CPUCellMeasuresOptions = {family: 'h3'}"],
    defaults: [
      "options.maximumEdgeCount ?? (options.family === 'h3' ? 6 : 4)",
      "options.units ?? 'meters'",
      "/** Prefix for generated node IDs. Defaults to `'cell-measures'`. */",
      "/** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */",
      'Defaults to 6 for H3 and 4 otherwise. H3 distortion vertices beyond the stride are truncated.',
      "Unit of `areas`, `perimeters` and `edgeLengths`: `'meters'` (default; square meters for areas)",
      "Geometry arithmetic: 'float32' (default) or the published 'double-single' module ('fp64' and 'extended' are deprecated aliases).",
      'One `float32` area per cell: square meters by default, square kilometers with `units:'
    ],
    limits: [
      'cells.length % 2',
      '`rows * maximumEdgeCount` `float32` edge lengths in meters (or kilometers). Cell `row` owns',
      'entries `[row * maximumEdgeCount, (row + 1) * maximumEdgeCount)`; entry `i` joins vertex `i` to'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Nothing differs in invalid-key handling, units, boundary order, padding or stride truncation.'
    ],
    source: 'modules/analysis/src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUCellMeasuresOperation': {
    signature: 'new CPUCellMeasuresOperation(props: CellMeasuresOperationProps)',
    supportedInputs: ['props: CellMeasuresOperationProps'],
    defaults: [
      "/** Prefix for generated node IDs. Defaults to `'cell-measures'`. */",
      "/** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */",
      'Defaults to 6 for H3 and 4 otherwise. H3 distortion vertices beyond the stride are truncated.',
      "Unit of `areas`, `perimeters` and `edgeLengths`: `'meters'` (default; square meters for areas)",
      "Geometry arithmetic: 'float32' (default) or the published 'double-single' module ('fp64' and 'extended' are deprecated aliases).",
      'One `float32` area per cell: square meters by default, square kilometers with `units:'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement( id, Object.values(props.output).some(value => value !== undefined), 'output must contain at least one output' )",
      'cells.length !== props.rowCount * 2',
      '`rows * maximumEdgeCount` `float32` edge lengths in meters (or kilometers). Cell `row` owns',
      'entries `[row * maximumEdgeCount, (row + 1) * maximumEdgeCount)`; entry `i` joins vertex `i` to'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/cell-indexing/cpu-cell-measures-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUCellMeasuresOperation': {
    signature: 'new GPUCellMeasuresOperation(props: CellMeasuresOperationProps)',
    supportedInputs: ['props: CellMeasuresOperationProps'],
    defaults: [
      "props.id ?? 'cell-measures'",
      "props.maximumEdgeCount ?? (props.family === 'h3' ? 6 : 4)",
      "/** Prefix for generated node IDs. Defaults to `'cell-measures'`. */",
      "/** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */",
      'Defaults to 6 for H3 and 4 otherwise. H3 distortion vertices beyond the stride are truncated.',
      "Unit of `areas`, `perimeters` and `edgeLengths`: `'meters'` (default; square meters for areas)",
      "Geometry arithmetic: 'float32' (default) or the published 'double-single' module ('fp64' and 'extended' are deprecated aliases).",
      'One `float32` area per cell: square meters by default, square kilometers with `units:'
    ],
    limits: [
      "validateRequirement( id, (['areas', 'perimeters', 'edgeLengths', 'edgeCounts', 'pentagons'] as const).some( port => props.output[port] !== undefined ), 'output must contain at least one terminal measure: areas, perimeters, edgeLengths, edgeCounts or pentagons' )",
      '`rows * maximumEdgeCount` `float32` edge lengths in meters (or kilometers). Cell `row` owns',
      'entries `[row * maximumEdgeCount, (row + 1) * maximumEdgeCount)`; entry `i` joins vertex `i` to'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeS2CapCoveringOnCPU': {
    signature:
      'export function computeS2CapCoveringOnCPU( caps: Float32Array, options: CPUS2CapCoveringOptions ): CPUS2CapCoveringResult',
    supportedInputs: ['caps: Float32Array', 'options: CPUS2CapCoveringOptions'],
    defaults: [
      'frontier exhaustion writes zero cells/counts and status bit 4. The GPU defaults to a bound.',
      'options.tolerance ?? 1e-5',
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      '/** Exterior expands / interior shrinks by this angular safety margin, default 1e-5. */',
      "Enumeration. Defaults to `'uniform'`, the reference tier whose outputs are pinned by recorded",
      'final cell count, not the final count itself. Default `8 * capacity + 64`, which holds a cover',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'caps.length % 4 !== 0',
      "validateMaximumCount('s2-cap-covering', 'caps times capacity', capCount * capacity, 0xffffffff)",
      "validateMaximumCount( 's2-cap-covering', 'requiredCount', required + descendantCount, 0xffffffff )",
      "- `'uniform'`: each cap's invocation tests every cell at `level` (level at most"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUS2CapCoveringOperation': {
    signature: 'new CPUS2CapCoveringOperation(props: S2CapCoveringOperationProps)',
    supportedInputs: ['props: S2CapCoveringOperationProps'],
    defaults: [
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      '/** Exterior expands / interior shrinks by this angular safety margin, default 1e-5. */',
      "Enumeration. Defaults to `'uniform'`, the reference tier whose outputs are pinned by recorded",
      'final cell count, not the final count itself. Default `8 * capacity + 64`, which holds a cover',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'capCount', props.capCount)",
      "validateMaximumCount( id, 'capCount times capacity', props.capCount * props.capacity, 0xffffffff )",
      'caps.length !== props.capCount * 4',
      "- `'uniform'`: each cap's invocation tests every cell at `level` (level at most"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/cell-topology/cpu-s2-cap-covering-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeS2PolygonCoveringOnCPU': {
    signature:
      'export function computeS2PolygonCoveringOnCPU( input: CPUS2PolygonCoveringInput, options: CPUS2PolygonCoveringOptions ): CPUS2PolygonCoveringResult',
    supportedInputs: ['input: CPUS2PolygonCoveringInput', 'options: CPUS2PolygonCoveringOptions'],
    defaults: [
      'traversal; the GPU defaults to 65,536. Serial exhaustion retains its emitted prefix;',
      'options.tolerance ?? 1e-6',
      'options.hemisphereCenter ?? (sum as [number, number, number])',
      'options.maximumVisitedCells ?? Infinity',
      'options.minimumLevel ?? 0',
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      'Maximum visited hierarchy cells. Exhaustion is explicit; default',
      '{@link GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS}. In the level-synchronous traversal it',
      "Traversal. Defaults to `'serial'`, whose outputs, including partial results on exhaustion, are",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- Levels up to 30 are accepted (the S2 key layout limit), dropping the GPU level-15 precision',
      'positions.length % 3 !== 0',
      'ringOffsets.length < 2',
      "validateMaximumCount('s2-polygon-covering', 'visitedCount', visited, 0xffffffff)",
      "validateMaximumCount('s2-polygon-covering', 'requiredCount', required, 0xffffffff)",
      '/** Maximum output S2 cells, across all levels, in either traversal route. */',
      '`maximumLevel + 1` rounds classifies the frontier, then a scan and a write build the next one.'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'as on the GPU. Inconsistent packed array lengths throw. Unreferenced vertices are ignored.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUS2PolygonCoveringOperation': {
    signature: 'new CPUS2PolygonCoveringOperation(props: S2PolygonCoveringOperationProps)',
    supportedInputs: ['props: S2PolygonCoveringOperationProps'],
    defaults: [
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      'Maximum visited hierarchy cells. Exhaustion is explicit; default',
      '{@link GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS}. In the level-synchronous traversal it',
      "Traversal. Defaults to `'serial'`, whose outputs, including partial results on exhaustion, are",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'positionCount', props.positionCount, 0)",
      "validateCapacity(id, 'ringOffsetCount', props.ringOffsetCount, 2)",
      'positions.length !== props.positionCount * 3',
      'ringOffsets.length !== props.ringOffsetCount',
      '/** Maximum output S2 cells, across all levels, in either traversal route. */',
      '`maximumLevel + 1` rounds classifies the frontier, then a scan and a write build the next one.'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUS2CapCoveringOperation': {
    signature: 'new GPUS2CapCoveringOperation(props: S2CapCoveringOperationProps)',
    supportedInputs: ['props: S2CapCoveringOperationProps'],
    defaults: [
      "props.id ?? 's2-cap-covering'",
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      '/** Exterior expands / interior shrinks by this angular safety margin, default 1e-5. */',
      "Enumeration. Defaults to `'uniform'`, the reference tier whose outputs are pinned by recorded",
      'final cell count, not the final count itself. Default `8 * capacity + 64`, which holds a cover',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: ["- `'uniform'`: each cap's invocation tests every cell at `level` (level at most"],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUS2PolygonCoveringOperation': {
    signature: 'new GPUS2PolygonCoveringOperation(props: S2PolygonCoveringOperationProps)',
    supportedInputs: ['props: S2PolygonCoveringOperationProps'],
    defaults: [
      "props.id ?? 's2-polygon-covering'",
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      'Maximum visited hierarchy cells. Exhaustion is explicit; default',
      '{@link GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS}. In the level-synchronous traversal it',
      "Traversal. Defaults to `'serial'`, whose outputs, including partial results on exhaustion, are",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'positionCount', props.positionCount, 0)",
      "validateCapacity(id, 'ringOffsetCount', props.ringOffsetCount, 2)",
      'validateRequirement(id, positions.length === props.positionCount, `positionCount must equal declared ${props.positionCount}; received positions.length ${positions.length}`)',
      'validateRequirement(id, ringOffsets.length === props.ringOffsetCount, `ringOffsetCount must equal declared ${props.ringOffsetCount}; received ringOffsets.length ${ringOffsets.length}`)',
      '/** Maximum output S2 cells, across all levels, in either traversal route. */',
      '`maximumLevel + 1` rounds classifies the frontier, then a scan and a write build the next one.'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeDelaunayTessellationOnCPU': {
    signature:
      'export function computeDelaunayTessellationOnCPU( input: CPUDelaunayTessellationInput, options: CPUDelaunayTessellationOptions ): CPUDelaunayTessellationResult',
    supportedInputs: [
      'input: CPUDelaunayTessellationInput',
      'options: CPUDelaunayTessellationOptions'
    ],
    defaults: [
      'options.groupCount ?? 1',
      'options.groupCount ?? 0',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'positions.length % 2 !== 0',
      "validateCapacity(id, 'capacity', options.capacity, 0)",
      "validateCapacity(id, 'groupCount', options.groupCount)",
      '(groupIds !== undefined) !== (options.groupCount !== undefined)',
      'groupIds && groupIds.length !== positions.length / 2',
      "validateCapacity(id, 'requiredCount', requiredCount, 0)"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUDelaunayTessellationOperation': {
    signature: 'new CPUDelaunayTessellationOperation(props: DelaunayTessellationOperationProps)',
    supportedInputs: ['props: DelaunayTessellationOperationProps'],
    defaults: [
      "/** Prefix for generated nodes and transients. Defaults to `'delaunay-tessellation'`. */",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'capacity', props.capacity, 0)",
      "validateCapacity(id, 'groups.count', props.groups.count)",
      "validateRequirement( id, Boolean(props.groups) === Boolean(props.output.groups), 'groups and output.groups must be supplied together' )",
      'capacity = 0 is a count-only request. output.status.count, output.status.requiredCount and output.status.overflow must each name a nonempty graph value, including at zero capacity. count is min(requiredCount, capacity); overflow is one when requiredCount exceeds capacity.'
    ],
    optionHandling: [
      'triangulation. Points whose label is at least `count` are ignored. Output triangles are then',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUDelaunayTessellationOperation': {
    signature: 'new GPUDelaunayTessellationOperation(props: DelaunayTessellationOperationProps)',
    supportedInputs: ['props: DelaunayTessellationOperationProps'],
    defaults: [
      "props.id ?? 'delaunay-tessellation'",
      "/** Prefix for generated nodes and transients. Defaults to `'delaunay-tessellation'`. */",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( props.id ?? 'delaunay-tessellation', typeof value === 'string' && value.length > 0, `output.status.${port} must name a graph value; received ${String(value)}` )",
      'capacity = 0 is a count-only request. output.status.count, output.status.requiredCount and output.status.overflow must each name a nonempty graph value, including at zero capacity. count is min(requiredCount, capacity); overflow is one when requiredCount exceeds capacity.'
    ],
    optionHandling: [
      'triangulation. Points whose label is at least `count` are ignored. Output triangles are then',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTransitionDiagnosticsOnCPU': {
    signature:
      'export function computeTransitionDiagnosticsOnCPU( counts: Uint32Array, options: CPUTransitionDiagnosticsOptions ): CPUTransitionDiagnosticsResult',
    supportedInputs: ['counts: Uint32Array', 'options: CPUTransitionDiagnosticsOptions'],
    defaults: [
      'options.conditionCount ?? 1',
      'options.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS'
    ],
    limits: [
      "- minimumTransitions is not capped at uint32, unlike the GPU's integer sparsity diagnostic.",
      "validateCapacity('computeTransitionDiagnosticsOnCPU', 'classCount', classCount)",
      "validateCapacity('computeTransitionDiagnosticsOnCPU', 'conditionCount', conditionCount)",
      "validateCapacity( 'computeTransitionDiagnosticsOnCPU', 'minimumTransitions', minimumTransitions, 1, Number.MAX_SAFE_INTEGER )",
      'counts.length < conditionCount * classCount * classCount'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTransitionDiagnosticsOperation': {
    signature: 'new CPUTransitionDiagnosticsOperation(props: TransitionDiagnosticsOperationProps)',
    supportedInputs: ['props: TransitionDiagnosticsOperationProps'],
    defaults: [
      "props.id ?? 'transition-diagnostics'",
      'props.conditionCount ?? 1',
      'props.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS'
    ],
    limits: [
      '/** CPU twin. Class count is not capped at the GPU private-array size of 64. */',
      "validateCapacity(id, 'classCount', props.classCount)",
      "validateCapacity(id, 'conditionCount', conditionCount)",
      "validateCapacity(id, 'minimumTransitions', minimumTransitions, 1, Number.MAX_SAFE_INTEGER)",
      "validateRequirement( id, !output.homogeneity || conditionCount >= 2, 'output.homogeneity needs conditionCount of at least 2' )",
      "validateRequirement( id, !output.status?.converged || Boolean(output.steadyStates), 'output.status.converged needs output.steadyStates' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTransitionDiagnosticsOperation': {
    signature: 'new GPUTransitionDiagnosticsOperation(props: TransitionDiagnosticsOperationProps)',
    supportedInputs: ['props: TransitionDiagnosticsOperationProps'],
    defaults: [
      "props.id ?? 'transition-diagnostics'",
      'props.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS'
    ],
    limits: [
      "validateCapacity( props.id ?? 'transition-diagnostics', 'minimumTransitions', props.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS, 1 )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeRankMarkovOnCPU': {
    signature:
      'export function computeRankMarkovOnCPU( values: Float32Array, options: CPURankMarkovOptions, mask?: Uint32Array ): CPURankMarkovResult',
    supportedInputs: [
      'values: Float32Array',
      'options: CPURankMarkovOptions',
      'mask?: Uint32Array'
    ],
    defaults: 'unknown',
    limits: ['!Number.isInteger(rowCount) || rowCount < 1', 'mask && mask.length !== rowCount'],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPURankMarkovOperation': {
    signature: 'new CPURankMarkovOperation(props: RankMarkovOperationProps)',
    supportedInputs: ['props: RankMarkovOperationProps'],
    defaults: 'unknown',
    limits: [
      "validateRequirement( id, props.output.status === undefined || props.output.steadyStates !== undefined, 'output.status.converged needs output.steadyStates' )",
      'values.length !== props.rowCount * props.periodCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURankMarkovOperation': {
    signature: 'new GPURankMarkovOperation(props: RankMarkovOperationProps)',
    supportedInputs: ['props: RankMarkovOperationProps'],
    defaults: ["props.id ?? 'rank-markov'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeLocalFrameProjectionOnCPU': {
    signature:
      'export function computeLocalFrameProjectionOnCPU( positions: Float32Array | Uint32Array, frame: GPULocalFrame, bounds: ProjectionBounds ): CPULocalFrameProjectionResult',
    supportedInputs: [
      'positions: Float32Array | Uint32Array',
      'frame: GPULocalFrame',
      'bounds: ProjectionBounds'
    ],
    defaults: [
      'Projects rows of `[longitude, latitude, ...]` with `stride` numbers per row (default 2) to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: ['positions.length % components !== 0'],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-measures/cpu-local-frame-projection-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPULocalFrameProjectionOperation': {
    signature: 'new CPULocalFrameProjectionOperation(props: LocalFrameProjectionOperationProps)',
    supportedInputs: ['props: LocalFrameProjectionOperationProps'],
    defaults: [
      'props.tolerance ?? 0.01',
      "/** Prefix of generated graph nodes. Defaults to `'gpu-local-frame-projection'`. */",
      'relative to the center of `bounds`. Defaults to `0.01`. It does not bound the float32 rounding',
      'Projects rows of `[longitude, latitude, ...]` with `stride` numbers per row (default 2) to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 1)",
      "validateRequirement(id, (props.tolerance ?? 0.01) > 0, 'tolerance must be greater than 0')",
      "validateRequirement( id, props.positionsFormat === undefined || props.positionsFormat === 'float32x2' || props.positionsFormat === 'uint32x4', 'positionsFormat must be float32x2 or uint32x4' )",
      "validateRequirement( id, positions.length === props.rowCount * (props.positionsFormat === 'uint32x4' ? 4 : 2), 'positions length must match rowCount' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-measures/cpu-local-frame-projection-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocalFrameProjectionOperation': {
    signature: 'new GPULocalFrameProjectionOperation(props: LocalFrameProjectionOperationProps)',
    supportedInputs: ['props: LocalFrameProjectionOperationProps'],
    defaults: [
      "props.id ?? 'gpu-local-frame-projection'",
      "/** Prefix of generated graph nodes. Defaults to `'gpu-local-frame-projection'`. */",
      'relative to the center of `bounds`. Defaults to `0.01`. It does not bound the float32 rounding',
      'Projects rows of `[longitude, latitude, ...]` with `stride` numbers per row (default 2) to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeGeodesicPairsOnCPU': {
    signature:
      'export function computeGeodesicPairsOnCPU( input: CPUGeodesicPairsInput, spatialContext: GPUSpatialContext, maximumIterations?: number ): CPUGeodesicPairsResult',
    supportedInputs: [
      'input: CPUGeodesicPairsInput',
      'spatialContext: GPUSpatialContext',
      'maximumIterations?: number'
    ],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'maximumIterations bounds auxiliary-longitude convergence with float32 updates and the',
      'and reports converged=0, approximation=1, iterationLimitReached=1.',
      "validateCapacity( 'computeGeodesicPairsOnCPU', 'maximumIterations', maximumIterations, 1, GPU_GEODESIC_MAXIMUM_ITERATIONS )",
      'input.origins.length % 2 !== 0',
      'input.targets.length !== input.origins.length'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '@throws If origins/targets are not aligned packed float32x2 rows or the context is unsupported.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUGeodesicPairsOperation': {
    signature: 'new CPUGeodesicPairsOperation(props: GeodesicPairsOperationProps)',
    supportedInputs: ['props: GeodesicPairsOperationProps'],
    defaults: [
      "props.id ?? 'geodesic-pairs'",
      'props.maximumIterations ?? 16',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'pairCount', props.pairCount)",
      "validateCapacity( id, 'maximumIterations', props.maximumIterations ?? 16, 1, GPU_GEODESIC_MAXIMUM_ITERATIONS )",
      "validateRequirement( id, Object.values(props.output).some(value => typeof value === 'string') || props.output.status?.converged !== undefined || props.output.status?.approximation !== undefined || props.output.status?.iterationLimitReached !== undefined, 'output requires at least one column' )",
      "validateRequirement( id, props.output.status?.converged === undefined || model === 'wgs84', \"output.status.converged requires metric 'ellipsoidal'\" )",
      'origins.length !== props.pairCount * 2'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUGeodesicPairsOperation': {
    signature: 'new GPUGeodesicPairsOperation(props: GeodesicPairsOperationProps)',
    supportedInputs: ['props: GeodesicPairsOperationProps'],
    defaults: [
      "props.id ?? 'geodesic-pairs'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeHealpixPointToCellOnCPU': {
    signature:
      'export function computeHealpixPointToCellOnCPU( positions: Float32Array, options: {order: number; scheme?: GPUHealpixIndexScheme; mask?: Uint32Array} ): CPUHealpixPointToCellResult',
    supportedInputs: [
      'positions: Float32Array',
      'options: {order: number; scheme?: GPUHealpixIndexScheme; mask?: Uint32Array}'
    ],
    defaults: 'unknown',
    limits: [
      '- The order limit 29 is retained for the HEALPix 64-bit key layout. Empty arrays are accepted',
      '!Number.isInteger(rowCount)',
      'options.mask && options.mask.length !== rowCount',
      "validateCapacity('healpix-point-to-cell', 'order', options.order, 0, HEALPIX_MAXIMUM_ORDER)"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUHealpixPointToCellOperation': {
    signature: 'new CPUHealpixPointToCellOperation(props: HealpixPointToCellOperationProps)',
    supportedInputs: ['props: HealpixPointToCellOperationProps'],
    defaults: [
      'props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT',
      "/** Prefix for generated node IDs. Defaults to `'healpix-point-to-cell'`. */",
      "/** Key scheme. Defaults to `'nested'`. */",
      '/** Coordinate contract of `positions`: longitude/latitude without a metric (the default). */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'order', props.order, 0, HEALPIX_MAXIMUM_ORDER)",
      'positions.length !== props.rowCount * 2'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUHealpixPointToCellOperation': {
    signature: 'new GPUHealpixPointToCellOperation(props: HealpixPointToCellOperationProps)',
    supportedInputs: ['props: HealpixPointToCellOperationProps'],
    defaults: [
      "props.id ?? 'healpix-point-to-cell'",
      'props.spatialContext ?? GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT',
      "/** Prefix for generated node IDs. Defaults to `'healpix-point-to-cell'`. */",
      "/** Key scheme. Defaults to `'nested'`. */",
      '/** Coordinate contract of `positions`: longitude/latitude without a metric (the default). */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'order', props.order, 0, HEALPIX_MAXIMUM_ORDER)",
      "validateCapacity(id, 'rowCount', props.rowCount, 0)"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeLabelPointOnCPU': {
    signature: 'export function computeLabelPointOnCPU(input: CPUPolygonRows): CPULabelPointResult',
    supportedInputs: ['input: CPUPolygonRows'],
    defaults: 'unknown',
    limits: [
      '- Searches every cell whose distance bound can improve the answer, rather than a capped set',
      "of coarse grid lobes. Stops at 1e-6 of the feature's maximum extent (binary64 resolution"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-spatial-analysis/label-point/cpu-label-point-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPULabelPointOperation': {
    signature: 'new CPULabelPointOperation(props: LabelPointOperationProps)',
    supportedInputs: ['props: LabelPointOperationProps'],
    defaults: [
      "/** Prefix for generated node IDs. Defaults to `'label-point'`. */",
      "/** Samples per axis of the first grid over the feature's bounds. Default 16. */",
      '/** Samples per axis of every refinement grid. Default 8. */',
      '/** Refinement rounds. Default 6. */',
      'cannot beat the best result so far are skipped, so typical shapes refine one or two. Default 4.',
      'stays proportional to `vertices / 64`. Default 32, the measured crossover.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'featureCount', props.featureCount, 1)",
      "validateRequirement( id, input.featureOffsets.length === props.featureCount + 1, 'featureOffsets length must equal featureCount + 1' )",
      'Features with at most this many vertices are searched by one thread each, which is cheapest',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/label-point/cpu-label-point-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULabelPointOperation': {
    signature: 'new GPULabelPointOperation(props: LabelPointOperationProps)',
    supportedInputs: ['props: LabelPointOperationProps'],
    defaults: [
      "props.id ?? 'label-point'",
      "/** Prefix for generated node IDs. Defaults to `'label-point'`. */",
      "/** Samples per axis of the first grid over the feature's bounds. Default 16. */",
      '/** Samples per axis of every refinement grid. Default 8. */',
      '/** Refinement rounds. Default 6. */',
      'cannot beat the best result so far are skipped, so typical shapes refine one or two. Default 4.',
      'stays proportional to `vertices / 64`. Default 32, the measured crossover.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'Features with at most this many vertices are searched by one thread each, which is cheapest',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/label-point/gpu-label-point-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeLineLengthPerPolygonOnCPU': {
    signature:
      'export function computeLineLengthPerPolygonOnCPU( input: { lines: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; lineWeights?: Float32Array; }, options: CPULineLengthPerPolygonOptions )',
    supportedInputs: [
      'input: { lines: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; lineWeights?: Float32Array; }',
      'options: CPULineLengthPerPolygonOptions'
    ],
    defaults: ['options.maximumCrossingCount ?? 64'],
    limits: [
      "validateRequirement( id, lines.kind === 'lines' && lines.positions.length % 2 === 0 && Boolean(lines.lineOffsets), 'lines needs XY positions and lineOffsets' )",
      "validateRequirement( id, !lineWeights || lineWeights.length === featureOffsets.length - 1, 'lineWeights needs one row per line feature' )",
      "validateCapacity(id, 'pairCapacity', options.pairCapacity)",
      "validateCapacity( id, 'maximumCrossingCount', maximumCrossingCount, 1, LINE_LENGTH_MAXIMUM_CROSSING_COUNT )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/line-density/cpu-line-length-per-polygon-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeLineSimplificationOnCPU': {
    signature:
      'export function computeLineSimplificationOnCPU( input: CPULineSimplificationInput, options: CPULineSimplificationOptions = {} ): CPULineSimplificationResult',
    supportedInputs: [
      'input: CPULineSimplificationInput',
      'options: CPULineSimplificationOptions = {}'
    ],
    defaults: [
      'options.capacity ?? rowCount',
      'options.maximumIterations ?? Number.MAX_SAFE_INTEGER',
      'options.finishSpanLimit ?? 32',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- An explicit maximumIterations binds the CPU rounds; omission leaves the CPU uncapped. Douglas-Peucker is',
      '!Number.isInteger(rowCount)',
      '!input.positions.every(Number.isFinite)',
      'lineOffsets must contain at least one terminal offset. One offset describes zero lines.',
      'With storage omitted, lineOffsets must be nondecreasing and in [0, rowCount].',
      "With storage 'compact', lineOffsets must be nondecreasing and in [0, rowCount], start at 0, and end at rowCount.",
      "With storage 'resident-capacity', lineOffsets must be nondecreasing after clamping each entry to rowCount.",
      'input.timestamps && input.timestamps.length !== rowCount',
      "options.metric === 'time-ratio' && options.computeImportance !== false && (!input.timestamps || !input.timestamps.every(Number.isFinite))",
      'options.computeImportance === false && (!input.importance || input.importance.length !== rowCount)',
      'input.parameters && input.parameters.length < GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH',
      "validateCapacity('computeLineSimplificationOnCPU', 'capacity', capacity, 0)",
      "validateCapacity( 'computeLineSimplificationOnCPU', 'maximumIterations', maximumIterations, 1, Number.MAX_SAFE_INTEGER )"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '@throws For inconsistent arrays, offsets invalid for the chosen storage, or unsupported options.',
      '/** Packed line rows; importance is required in selection-only mode. */'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPULineSimplificationOperation': {
    signature: 'new CPULineSimplificationOperation(props: LineSimplificationOperationProps)',
    supportedInputs: ['props: LineSimplificationOperationProps'],
    defaults: [
      "props.id ?? 'line-simplification'",
      'props.capacity ?? props.rowCount',
      'props.maximumIterations ?? 64',
      'props.neighborhoodRadius ?? 3',
      "/** Prefix for generated node and transient IDs. Defaults to `'line-simplification'`. */",
      "Native planar coordinates (default), or `{coordinateSpace: 'longitude-latitude', metric:",
      '(mean Earth radius in meters or kilometers by default), and chords are straight in',
      "Simplification algorithm. Default `'douglas-peucker'`. With `'visvalingam'` the importance of a",
      'Default 3. Larger values follow the sequential smallest-area-first order more closely and need',
      "Distance used by the importance pass. `'segment'` (default) is the Euclidean distance from a",
      'Whether this contributor computes `importance`. Default `true`. Set to `false` in a per-frame graph',
      'Compile-time cap on level-synchronous iterations (rounds), in `[1, 1024]`. Default 64 (256 for',
      'default {@link GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT}; `0` runs rounds only.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      "Simplification algorithm. `'douglas-peucker'` (default) is exact recursive Douglas-Peucker (or"
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'lineCount', props.lineCount)",
      "validateCapacity(id, 'capacity', props.capacity ?? props.rowCount, 0)",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations ?? 64)",
      "validateCapacity(id, 'neighborhoodRadius', props.neighborhoodRadius ?? 3)",
      "validateRequirement( id, props.computeImportance !== false || props.output?.ids !== undefined, 'computeImportance=false requires output.ids' )",
      "validateRequirement( id, props.computeImportance !== false || (props.output?.status?.converged === undefined && props.output?.status?.iterationCount === undefined), 'output convergence diagnostics require computeImportance' )",
      "validateRequirement( id, props.output?.ids === undefined || (props.parameters !== undefined && props.output.status?.count !== undefined && props.output.status.overflow !== undefined), 'output.ids requires parameters and status.count/overflow' )",
      "validateRequirement( id, props.metric !== 'time-ratio' || props.computeImportance === false || props.timestamps !== undefined, 'metric time-ratio requires timestamps' )",
      "validateRequirement( id, props.output?.ids !== undefined || [ props.output?.outputMask, props.output?.lineCounts, props.output?.lineStarts, props.output?.status?.count, props.output?.status?.overflow, props.output?.status?.candidateOverflow, props.output?.status?.requiredCount ].every(value => value === undefined), 'selection columns require output.ids' )",
      'positions.length !== props.rowCount * 2',
      'lineOffsets.length !== props.lineCount + 1',
      'Douglas-Peucker only. An open interval with at most this many interior rows is solved',
      'default {@link GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT}; `0` runs rounds only.',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional',
      '/** One-row scalar: `1` when every row was decided within `maximumIterations`, otherwise `0`. */'
    ],
    optionHandling: [
      "Packed float32 times aligned with `positions`. Required for `metric: 'time-ratio'`, ignored",
      "`'segment'` metric applies, and `timestamps` are ignored."
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineSimplificationOperation': {
    signature: 'new GPULineSimplificationOperation(props: LineSimplificationOperationProps)',
    supportedInputs: ['props: LineSimplificationOperationProps'],
    defaults: [
      'props.capacity ?? props.rowCount',
      "props.id ?? 'line-simplification'",
      "/** Prefix for generated node and transient IDs. Defaults to `'line-simplification'`. */",
      "Native planar coordinates (default), or `{coordinateSpace: 'longitude-latitude', metric:",
      '(mean Earth radius in meters or kilometers by default), and chords are straight in',
      "Simplification algorithm. Default `'douglas-peucker'`. With `'visvalingam'` the importance of a",
      'Default 3. Larger values follow the sequential smallest-area-first order more closely and need',
      "Distance used by the importance pass. `'segment'` (default) is the Euclidean distance from a",
      'Whether this contributor computes `importance`. Default `true`. Set to `false` in a per-frame graph',
      'Compile-time cap on level-synchronous iterations (rounds), in `[1, 1024]`. Default 64 (256 for',
      'default {@link GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT}; `0` runs rounds only.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      "Simplification algorithm. `'douglas-peucker'` (default) is exact recursive Douglas-Peucker (or"
    ],
    limits: [
      'Douglas-Peucker only. An open interval with at most this many interior rows is solved',
      'default {@link GPU_LINE_SIMPLIFICATION_DEFAULT_FINISH_SPAN_LIMIT}; `0` runs rounds only.',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional',
      '/** One-row scalar: `1` when every row was decided within `maximumIterations`, otherwise `0`. */'
    ],
    optionHandling: [
      "Packed float32 times aligned with `positions`. Required for `metric: 'time-ratio'`, ignored",
      "`'segment'` metric applies, and `timestamps` are ignored."
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/line-simplification/gpu-line-simplification-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeLocationAllocationOnCPU': {
    signature:
      'export function computeLocationAllocationOnCPU( input: CPULocationAllocationInput, options: CPULocationAllocationOptions ): CPULocationAllocationResult',
    supportedInputs: ['input: CPULocationAllocationInput', 'options: CPULocationAllocationOptions'],
    defaults: [
      '- The explicit or default swap budget binds both forms; zero disables swaps. The CPU drops',
      "options.id ?? 'location-allocation'",
      'options.maximumSwapIterations ?? GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS'
    ],
    limits: [
      "validateRequirement( id, !(Number(input.costs !== undefined) + Number(input.sparseCosts !== undefined) !== 1), 'costs and sparseCosts require exactly one representation' )",
      "validateRequirement( id, !(input.costs && input.costs.length !== demandCount * facilityCount), 'costs length must equal demandCount * facilityCount' )",
      "validateRequirement( id, !(input.demandWeights && input.demandWeights.length !== demandCount), 'demandWeights length must equal demandCount' )",
      "validateRequirement( id, !(operation !== 'p-median' && !input.coverageDistance?.length), 'coverageDistance must hold at least one value' )",
      "validateRequirement( id, !(input.costsComplete && input.costsComplete.length !== 1), 'costsComplete must contain exactly one row' )",
      "validateRequirement( id, !(input.costLimit && input.costLimit.length !== 1), 'costLimit must contain exactly one row' )",
      "validateRequirement( id, !(sparse.offsets.length !== demandCount + 1), 'sparseCosts.offsets length must equal demandCount + 1' )",
      "validateRequirement( id, !(!sparse.costs.length || sparse.facilities.length !== sparse.costs.length), 'sparseCosts.costs and facilities must have equal positive lengths' )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPULocationAllocationOperation': {
    signature: 'new CPULocationAllocationOperation(props: LocationAllocationOperationProps)',
    supportedInputs: ['props: LocationAllocationOperationProps'],
    defaults: ["props.id ?? 'location-allocation'"],
    limits: [
      "validateRequirement( id, Number(props.costs !== undefined) + Number(props.sparseCosts !== undefined) === 1, 'costs and sparseCosts require exactly one representation' )",
      "validateRequirement( id, props.operation === 'p-median' || props.coverageDistance !== undefined, 'coverageDistance is required for coverage operations' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/location-allocation/cpu-location-allocation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULocationAllocationGraphOperation': {
    signature: 'new GPULocationAllocationGraphOperation(props: LocationAllocationOperationProps)',
    supportedInputs: ['props: LocationAllocationOperationProps'],
    defaults: ["props.id ?? 'location-allocation'"],
    limits: [
      "validateRequirement( id, costs.length === props.demandCount * props.facilityCount, 'costs length must equal demandCount * facilityCount' )",
      "validateRequirement( id, sparseCosts.offsets.length === props.demandCount + 1, 'sparseCosts.offsets length must equal demandCount + 1' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/location-allocation/gpu-location-allocation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMapColoringOnCPU': {
    signature:
      "export function computeMapColoringOnCPU( offsets: Uint32Array, neighbors: Uint32Array, parameters: Uint32Array, priority: GPUMapColoringPriority = 'largest-degree-first', maximumIterations?: number ): CPUMapColoringResult",
    supportedInputs: [
      'offsets: Uint32Array',
      'neighbors: Uint32Array',
      'parameters: Uint32Array',
      "priority: GPUMapColoringPriority = 'largest-degree-first'",
      'maximumIterations?: number'
    ],
    defaults: 'unknown',
    limits: [
      'rowCount < 1 || offsets[0] !== 0',
      'parameters.length < GPU_MAP_COLORING_PARAMETER_LENGTH',
      "validateRequirement( 'computeMapColoringOnCPU', ['largest-degree-first', 'random'].includes(priority), \"priority must be 'largest-degree-first' or 'random'\" )",
      "validateCapacity('computeMapColoringOnCPU', 'maximumIterations', maximumIterations)",
      'offsets[row] > offsets[row + 1] || offsets[row + 1] > neighbors.length',
      'neighbors[slot] >= rowCount'
    ],
    optionHandling: [
      'Self slots are ignored in color selection but count toward the degree, as on the GPU.',
      'What differs from the GPU contributor:',
      '- The 1024-round GPU limit is dropped. Chunked CSR is not supported by either operation.'
    ],
    source: 'modules/analysis/src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMapColoringOperation': {
    signature: 'new CPUMapColoringOperation(props: MapColoringOperationProps)',
    supportedInputs: ['props: MapColoringOperationProps'],
    defaults: ["props.id ?? 'map-coloring'", "props.priority ?? 'largest-degree-first'"],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations)",
      "validateRequirement( id, ['largest-degree-first', 'random'].includes(priority), \"priority must be 'largest-degree-first' or 'random'\" )",
      'offsets.length !== rowCount + 1',
      'weights.length !== neighbors.length'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMapColoringOperation': {
    signature: 'new GPUMapColoringOperation(props: MapColoringOperationProps)',
    supportedInputs: ['props: MapColoringOperationProps'],
    defaults: ["props.id ?? 'map-coloring'", "props.priority ?? 'largest-degree-first'"],
    limits: [
      "validateRequirement( id, ['largest-degree-first', 'random'].includes(props.priority ?? 'largest-degree-first'), \"priority must be 'largest-degree-first' or 'random'\" )"
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMinimumBoundsOnCPU': {
    signature:
      'export function computeMinimumBoundsOnCPU(input: CPUMinimumBoundsInput): CPUMinimumBoundsResult',
    supportedInputs: ['input: CPUMinimumBoundsInput'],
    defaults: 'unknown',
    limits: [
      'input.positions.length % 2 !== 0',
      "'labels' in input",
      "validateCapacity('computeMinimumBoundsOnCPU', 'groupCount', groupCount)",
      'input.labels.length !== rowCount'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMinimumBoundsOperation': {
    signature: 'new CPUMinimumBoundsOperation(props: MinimumBoundsOperationProps)',
    supportedInputs: ['props: MinimumBoundsOperationProps'],
    defaults: [
      "props.id ?? 'minimum-bounds'",
      'props.maximumHullVertices ?? DEFAULT_MAXIMUM_HULL_VERTICES',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity( id, props.labels !== undefined ? 'groupCount' : 'featureCount', props.labels !== undefined ? props.groupCount : props.featureCount )",
      "validateCapacity( id, 'maximumHullVertices', props.maximumHullVertices ?? DEFAULT_MAXIMUM_HULL_VERTICES, 3 )",
      "validateRequirement( id, ['rectangleCorners', 'rectangleSizes', 'circles', 'longestLines', 'diameters', 'bounds'].some( port => props.output[port as keyof typeof props.output] !== undefined ), 'output requires at least one shape column' )",
      'positions.length !== props.rowCount * 2',
      "'featureOffsets' in input && input.featureOffsets.length !== props.featureCount! + 1"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMinimumBoundsOperation': {
    signature: 'new GPUMinimumBoundsOperation(props: MinimumBoundsOperationProps)',
    supportedInputs: ['props: MinimumBoundsOperationProps'],
    defaults: [
      "props.id ?? 'minimum-bounds'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'validateRequirement( id, positions.length === props.rowCount, `positions length must equal rowCount (${props.rowCount})` )',
      'validateRequirement( id, featureOffsets.length === props.featureCount + 1, `featureOffsets length must equal featureCount + 1 (${props.featureCount + 1})` )'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMultipleTestingAdjustmentOnCPU': {
    signature:
      'export function computeMultipleTestingAdjustmentOnCPU( pValues: Float32Array, method: GPUMultipleTestingMethod, minimumFamilySize = 0 ): CPUMultipleTestingAdjustmentResult',
    supportedInputs: [
      'pValues: Float32Array',
      'method: GPUMultipleTestingMethod',
      'minimumFamilySize = 0'
    ],
    defaults: 'unknown',
    limits: [
      '- The GPU sort row and minimumFamilySize limits at 2^31 are dropped; the uint32 family-size',
      "validateCapacity( 'computeMultipleTestingAdjustmentOnCPU', 'minimumFamilySize', minimumFamilySize, 0 )"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'CPUMultipleTestingAdjustmentResult contains adjustedPValues and one-word uint32 counts: finiteTestCount counts finite p-values in [0,1]; effectiveFamilySize = max(finiteTestCount, minimumFamilySize). observedTestCount is the deprecated finiteTestCount spelling; familySize and testedCount are deprecated effectiveFamilySize spellings (including padding, even for method none). Named operations expose the same optional count destinations; disagreeing aliases reject.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMultipleTestingAdjustmentOperation': {
    signature:
      'new CPUMultipleTestingAdjustmentOperation(props: MultipleTestingAdjustmentOperationProps)',
    supportedInputs: ['props: MultipleTestingAdjustmentOperationProps'],
    defaults: ["props.id ?? 'multiple-testing-adjustment'", 'props.minimumFamilySize ?? 0'],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'minimumFamilySize', props.minimumFamilySize ?? 0, 0)",
      'pValues.length !== props.rowCount',
      'method is required; accepted values: none, bonferroni, holm, benjamini-hochberg, benjamini-yekutieli. Omission and unknown values reject at construction.'
    ],
    optionHandling: [
      'CPUMultipleTestingAdjustmentResult contains adjustedPValues and one-word uint32 counts: finiteTestCount counts finite p-values in [0,1]; effectiveFamilySize = max(finiteTestCount, minimumFamilySize). observedTestCount is the deprecated finiteTestCount spelling; familySize and testedCount are deprecated effectiveFamilySize spellings (including padding, even for method none). Named operations expose the same optional count destinations; disagreeing aliases reject.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMultipleTestingAdjustmentOperation': {
    signature:
      'new GPUMultipleTestingAdjustmentOperation(props: MultipleTestingAdjustmentOperationProps)',
    supportedInputs: ['props: MultipleTestingAdjustmentOperationProps'],
    defaults: ["props.id ?? 'multiple-testing-adjustment'", 'props.minimumFamilySize ?? 0'],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 1, GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1)",
      "validateCapacity( id, 'minimumFamilySize', props.minimumFamilySize ?? 0, 0, GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1 )",
      'method is required; accepted values: none, bonferroni, holm, benjamini-hochberg, benjamini-yekutieli. Omission and unknown values reject at construction.'
    ],
    optionHandling: [
      'CPUMultipleTestingAdjustmentResult contains adjustedPValues and one-word uint32 counts: finiteTestCount counts finite p-values in [0,1]; effectiveFamilySize = max(finiteTestCount, minimumFamilySize). observedTestCount is the deprecated finiteTestCount spelling; familySize and testedCount are deprecated effectiveFamilySize spellings (including padding, even for method none). Named operations expose the same optional count destinations; disagreeing aliases reject.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeNeighborhoodSummaryOnCPU': {
    signature:
      'export function computeNeighborhoodSummaryOnCPU( input: CPUNeighborhoodSummaryInput, options: CPUNeighborhoodSummaryOptions = {} ): CPUNeighborhoodSummaryResult',
    supportedInputs: [
      'input: CPUNeighborhoodSummaryInput',
      'options: CPUNeighborhoodSummaryOptions = {}'
    ],
    defaults: ['options.statistics ?? []', 'options.focalWeight ?? 1'],
    limits: [
      'offsets.length < 2',
      'weights.length !== neighbors.length',
      'array && array.length !== rows',
      'input[name] && input[name].length < 1',
      'statistics.length && !values'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Medians have no member cap; maximumNeighbors is ignored and output overflow is always zero.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUNeighborhoodSummaryOperation': {
    signature: 'new CPUNeighborhoodSummaryOperation(props: NeighborhoodSummaryOperationProps)',
    supportedInputs: ['props: NeighborhoodSummaryOperationProps'],
    defaults: [
      "props.id ?? 'neighborhood-summary'",
      'props.statistics ?? []',
      'props.maximumNeighbors ?? 32'
    ],
    limits: [
      '/** CPU twin with uncapped medians. Optional unused spatial-weights ports are omitted. */',
      "validateCapacity(id, 'rowCount', props.rowCount, 1)",
      "validateCapacity(id, 'maximumNeighbors', props.maximumNeighbors ?? 32, 1)",
      "validateCapacity(id, 'rowCount * statistics.length', props.rowCount * statistics.length, 0)",
      "validateRequirement(id, numeric || categorical, 'output must request a statistic')",
      "validateRequirement( id, numeric === (props.values !== undefined) && numeric === (props.output.table !== undefined), 'statistics requires values and output.table' )",
      "validateRequirement( id, !categorical || props.categories !== undefined, 'categories is required for modes or entropy' )",
      "validateRequirement( id, !statistics.includes('median') || props.output.status !== undefined, 'median requires output.status.overflow' )",
      "validateRequirement( id, props.valuesFormat === undefined || ['float32', 'uint32'].includes(props.valuesFormat), 'valuesFormat must be float32 or uint32' )",
      "validateRequirement( id, props.weights.information.rowCount === props.rowCount, 'weights.information.rowCount must equal rowCount' )",
      'offsets.length !== props.rowCount + 1'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNeighborhoodSummaryOperation': {
    signature: 'new GPUNeighborhoodSummaryOperation(props: NeighborhoodSummaryOperationProps)',
    supportedInputs: ['props: NeighborhoodSummaryOperationProps'],
    defaults: ['props.statistics ?? []', "props.id ?? 'neighborhood-summary'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeConstrainedDelaunayTriangulationOnCPU': {
    signature:
      'export function computeConstrainedDelaunayTriangulationOnCPU( input: CPUConstrainedDelaunayTriangulationInput, options: CPUConstrainedDelaunayTriangulationOptions ): CPUConstrainedDelaunayTriangulationResult',
    supportedInputs: [
      'input: CPUConstrainedDelaunayTriangulationInput',
      'options: CPUConstrainedDelaunayTriangulationOptions'
    ],
    defaults: [
      "/** Positive elementary-step budget per polygon; defaults to the contributor's 4M steps. */",
      "/** Routing method, as on the contributor; defaults to 'monotone'. */",
      '/** Vertex threshold for the first monotone route; defaults to 4096. */',
      '/** Total raw vertex capacity for routed polygons; defaults to all positions. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: [
      'What differs from the GPU contributor:',
      '- maximumFlips is ignored: Lawson flipping continues until no strictly illegal convex'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUConstrainedDelaunayTriangulationOperation': {
    signature:
      'new CPUConstrainedDelaunayTriangulationOperation(props: CPUConstrainedDelaunayTriangulationOperationProps)',
    supportedInputs: ['props: CPUConstrainedDelaunayTriangulationOperationProps'],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'polygon-triangulation'`. */",
      'Planar only (default: native planar coordinates). A longitude-latitude context throws at',
      'case of ear clipping. Defaults to {@link GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK},',
      '(default `true`). Ring nodes are kept in a second list sorted by Morton code, so an ear test',
      "What happens to polygons that per-invocation Earcut cannot finish (default `'monotone'`).",
      'and skip Earcut (default 4096). Earcut with its z-order hash is near-linear for typical',
      'Capacity of the monotone path in cleaned vertices (default: all `positions` rows). Scratch,',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      "/** Positive elementary-step budget per polygon; defaults to the contributor's 4M steps. */",
      "/** Routing method, as on the contributor; defaults to 'monotone'. */",
      '/** Vertex threshold for the first monotone route; defaults to 4096. */',
      '/** Total raw vertex capacity for routed polygons; defaults to all positions. */',
      'storage defaults to resident-capacity; compact is an explicit host-validation option.'
    ],
    limits: [
      "validateCapacity(id, 'polygonCount', props.polygonCount)",
      "validateCapacity(id, 'indexCapacity', props.indexCapacity)",
      'polygonOffsets.length !== props.polygonCount + 1',
      "- `'none'`: today's behavior, a polygon over `maximumWork` is flagged `valid = 0` with a",
      'fit skip the monotone path and are given to Earcut within `maximumWork` (`valid = 0` with a',
      'degenerate slice if that fails too). At most',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'storage accepts compact or resident-capacity. compact requires initial zero, nondecreasing offsets and exact terminal coverage at every offsets level; resident-capacity preserves the GPU-compatible ring clamping. Unknown storage values reject at construction.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConstrainedDelaunayTriangulationOperation': {
    signature:
      'new GPUConstrainedDelaunayTriangulationOperation(props: ConstrainedDelaunayTriangulationOperationProps)',
    supportedInputs: ['props: ConstrainedDelaunayTriangulationOperationProps'],
    defaults: [
      "props.id ?? 'constrained-delaunay'",
      "/** Prefix for generated node and transient IDs. Defaults to `'polygon-triangulation'`. */",
      'Planar only (default: native planar coordinates). A longitude-latitude context throws at',
      'case of ear clipping. Defaults to {@link GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK},',
      '(default `true`). Ring nodes are kept in a second list sorted by Morton code, so an ear test',
      "What happens to polygons that per-invocation Earcut cannot finish (default `'monotone'`).",
      'and skip Earcut (default 4096). Earcut with its z-order hash is near-linear for typical',
      'Capacity of the monotone path in cleaned vertices (default: all `positions` rows). Scratch,',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "- `'none'`: today's behavior, a polygon over `maximumWork` is flagged `valid = 0` with a",
      'fit skip the monotone path and are given to Earcut within `maximumWork` (`valid = 0` with a',
      'degenerate slice if that fails too). At most',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePolygonTriangulationOnCPU': {
    signature:
      'export function computePolygonTriangulationOnCPU( input: CPUPolygonTriangulationInput, options: CPUPolygonTriangulationOptions ): CPUPolygonTriangulationResult',
    supportedInputs: [
      'input: CPUPolygonTriangulationInput',
      'options: CPUPolygonTriangulationOptions'
    ],
    defaults: [
      '- Earcut honors maximumWork (default 4M steps) per polygon, with no GPU scratch limit.',
      'options.maximumWork ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK',
      "options.largePolygonMethod ?? 'monotone'",
      'options.monotoneMinimumVertices ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES',
      'options.monotoneMaximumVertices ?? Math.max(1, vertexCount)',
      "/** Positive elementary-step budget per polygon; defaults to the contributor's 4M steps. */",
      "/** Routing method, as on the contributor; defaults to 'monotone'. */",
      '/** Vertex threshold for the first monotone route; defaults to 4096. */',
      '/** Total raw vertex capacity for routed polygons; defaults to all positions. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- Earcut honors maximumWork (default 4M steps) per polygon, with no GPU scratch limit.',
      'positions.length % 2 !== 0',
      "validateCapacity('polygon-triangulation', 'maximumWork', maximumWork)",
      "validateRequirement( 'polygon-triangulation', options.largePolygonMethod === undefined || ['none', 'monotone'].includes(options.largePolygonMethod), \"largePolygonMethod must be 'monotone' or 'none'\" )",
      "validateCapacity('polygon-triangulation', 'monotoneMinimumVertices', monotoneMinimumVertices)",
      "validateCapacity('polygon-triangulation', 'monotoneMaximumVertices', monotoneCapacity)",
      "validateCapacity('polygon-triangulation', 'indexCapacity', options.indexCapacity, 0)",
      "validateRequirement( 'polygon-triangulation', options.indexCapacity >= minimumIndexCount, `indexCapacity must be at least ${minimumIndexCount}` )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUPolygonTriangulationOperation': {
    signature: 'new CPUPolygonTriangulationOperation(props: CPUPolygonTriangulationOperationProps)',
    supportedInputs: ['props: CPUPolygonTriangulationOperationProps'],
    defaults: [
      "props.id ?? 'polygon-triangulation'",
      "/** Prefix for generated node and transient IDs. Defaults to `'polygon-triangulation'`. */",
      'Planar only (default: native planar coordinates). A longitude-latitude context throws at',
      'case of ear clipping. Defaults to {@link GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK},',
      '(default `true`). Ring nodes are kept in a second list sorted by Morton code, so an ear test',
      "What happens to polygons that per-invocation Earcut cannot finish (default `'monotone'`).",
      'and skip Earcut (default 4096). Earcut with its z-order hash is near-linear for typical',
      'Capacity of the monotone path in cleaned vertices (default: all `positions` rows). Scratch,',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      "/** Positive elementary-step budget per polygon; defaults to the contributor's 4M steps. */",
      "/** Routing method, as on the contributor; defaults to 'monotone'. */",
      '/** Vertex threshold for the first monotone route; defaults to 4096. */',
      '/** Total raw vertex capacity for routed polygons; defaults to all positions. */',
      'storage defaults to resident-capacity; compact is an explicit host-validation option.'
    ],
    limits: [
      "validateCapacity(id, 'polygonCount', props.polygonCount, 1)",
      "validateCapacity(id, 'indexCapacity', props.indexCapacity, 1)",
      'polygonOffsets.length !== props.polygonCount + 1',
      "- `'none'`: today's behavior, a polygon over `maximumWork` is flagged `valid = 0` with a",
      'fit skip the monotone path and are given to Earcut within `maximumWork` (`valid = 0` with a',
      'degenerate slice if that fails too). At most',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'storage accepts compact or resident-capacity. compact requires initial zero, nondecreasing offsets and exact terminal coverage at every offsets level; resident-capacity preserves the GPU-compatible ring clamping. Unknown storage values reject at construction.'
    ],
    source:
      'modules/analysis/src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPolygonTriangulationOperation': {
    signature: 'new GPUPolygonTriangulationOperation(props: PolygonTriangulationOperationProps)',
    supportedInputs: ['props: PolygonTriangulationOperationProps'],
    defaults: [
      "props.id ?? 'polygon-triangulation'",
      "/** Prefix for generated node and transient IDs. Defaults to `'polygon-triangulation'`. */",
      'Planar only (default: native planar coordinates). A longitude-latitude context throws at',
      'case of ear clipping. Defaults to {@link GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK},',
      '(default `true`). Ring nodes are kept in a second list sorted by Morton code, so an ear test',
      "What happens to polygons that per-invocation Earcut cannot finish (default `'monotone'`).",
      'and skip Earcut (default 4096). Earcut with its z-order hash is near-linear for typical',
      'Capacity of the monotone path in cleaned vertices (default: all `positions` rows). Scratch,',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'maximumWork', props.maximumWork)",
      "- `'none'`: today's behavior, a polygon over `maximumWork` is flagged `valid = 0` with a",
      'fit skip the monotone path and are given to Earcut within `maximumWork` (`valid = 0` with a',
      'degenerate slice if that fails too). At most',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePointsInPolygonsChoroplethOnCPU': {
    signature:
      'export function computePointsInPolygonsChoroplethOnCPU( input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; values?: Float32Array}, options: CPUPointsInPolygonsChoroplethOptions )',
    supportedInputs: [
      'input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry; values?: Float32Array}',
      'options: CPUPointsInPolygonsChoroplethOptions'
    ],
    defaults: ["options.statistic ?? 'count'"],
    limits: [
      "validateRequirement( id, ['count', 'sum', 'mean', 'minimum', 'maximum'].includes(statistic), 'statistic must be count, sum, mean, minimum or maximum' )",
      "validateRequirement( id, options.backend === undefined || options.backend === 'group' || options.backend === 'zonal', 'backend must be group or zonal' )",
      "validateRequirement( id, statistic === 'count' || Boolean(input.values), 'non-count statistic requires values' )",
      "validateRequirement( id, !input.values || input.values.length === queryCount, 'values needs one row per point feature' )"
    ],
    optionHandling: ['What differs from the GPU recipe:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/recipes/cpu-points-in-polygons-choropleth-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeEnsembleScoresOnCPU': {
    signature:
      'export function computeEnsembleScoresOnCPU( input: CPUEnsembleScoresInput, options: CPUEnsembleScoresOptions ): CPUEnsembleScoresResult',
    supportedInputs: ['input: CPUEnsembleScoresInput', 'options: CPUEnsembleScoresOptions'],
    defaults: [
      "options.scores ?? [ 'crps', 'energyScore', ...(variogramPairs ? ['variogramScore' as const] : []) ]",
      'options.variogramOrder ?? 0.5',
      '/** Defaults to CRPS, energy, and (when pairs exist) variogram. Only requested work is done. */'
    ],
    limits: [
      '- GPU workgroup/invocation limits (65535 members and 67108863 sites) are dropped; uint32 array',
      'ensemble.length < memberCount * dimension',
      'observations.length < dimension',
      'variogramPairs && variogramPairs.length % 2 !== 0',
      'variogramWeights && (!variogramPairs || variogramWeights.length < variogramPairs.length / 2)',
      "scores.includes('variogramScore') && (!variogramPairs || variogramPairs.length === 0)"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUEnsembleScoresOperation': {
    signature: 'new CPUEnsembleScoresOperation(props: EnsembleScoresOperationProps)',
    supportedInputs: ['props: EnsembleScoresOperationProps'],
    defaults: ["props.id ?? 'ensemble-scores'"],
    limits: [
      "validateRequirement( id, (props.output.variogramScore !== undefined) === (props.variogramPairs !== undefined), 'output.variogramScore and variogramPairs must go together' )",
      "validateRequirement( id, props.variogramWeights === undefined || props.variogramPairs !== undefined, 'variogramWeights requires variogramPairs' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUEnsembleScoresOperation': {
    signature: 'new GPUEnsembleScoresOperation(props: EnsembleScoresOperationProps)',
    supportedInputs: ['props: EnsembleScoresOperationProps'],
    defaults: ["props.id ?? 'ensemble-scores'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeGaussianPredictiveScoresOnCPU': {
    signature:
      'export function computeGaussianPredictiveScoresOnCPU( input: CPUGaussianPredictiveScoresInput, options: CPUGaussianPredictiveScoresOptions = {} ): CPUGaussianPredictiveScoresResult',
    supportedInputs: [
      'input: CPUGaussianPredictiveScoresInput',
      'options: CPUGaussianPredictiveScoresOptions = {}'
    ],
    defaults: [
      'options.scores ?? []',
      '/** Two-tailed noncoverage probability of the central interval; defaults to 0.05. */'
    ],
    limits: [
      "validateCapacity(id, 'means.length', count, 1)",
      "validateRequirement( id, input.variances.length >= count && input.observations.length >= count, 'variances and observations need one row per mean' )",
      "validateRequirement( id, scores.every(score => ['logScore', 'brierScore', 'intervalScore'].includes(score)), 'scores must name logScore, brierScore or intervalScore' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/scoring-rules/cpu-gaussian-predictive-scores-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSinkhornDivergenceOnCPU': {
    signature:
      'export function computeSinkhornDivergenceOnCPU( source: Float32Array, target: Float32Array, parameters: Float32Array, options: CPUSinkhornDivergenceOptions ): CPUSinkhornDivergenceResult',
    supportedInputs: [
      'source: Float32Array',
      'target: Float32Array',
      'parameters: Float32Array',
      'options: CPUSinkhornDivergenceOptions'
    ],
    defaults: [
      'options.transportRadius ?? options.blurRadius ?? Math.max(width, height) - 1',
      'options.transportRadius ?? options.blurRadius',
      'options.maximumIterations ?? GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'budget is retained, including iterationLimitReached; its GPU maximum of 1000 is dropped.',
      "- The GPU's grid-side, row-count and storage-binding limits are dropped; uint32 grid dimensions",
      'source.length !== count',
      'target.length !== count',
      'parameters.length < GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSinkhornDivergenceOperation': {
    signature: 'new CPUSinkhornDivergenceOperation(props: SinkhornDivergenceOperationProps)',
    supportedInputs: ['props: SinkhornDivergenceOperationProps'],
    defaults: [
      "props.id ?? 'sinkhorn-divergence'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSinkhornDivergenceOperation': {
    signature: 'new GPUSinkhornDivergenceOperation(props: SinkhornDivergenceOperationProps)',
    supportedInputs: ['props: SinkhornDivergenceOperationProps'],
    defaults: [
      "props.id ?? 'sinkhorn-divergence'",
      'props.output.status ?? {}',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpaceTimeHawkesOnCPU': {
    signature:
      'export function computeSpaceTimeHawkesOnCPU( input: CPUSpaceTimeHawkesInput, options: CPUSpaceTimeHawkesOptions = {} ): CPUSpaceTimeHawkesResult',
    supportedInputs: ['input: CPUSpaceTimeHawkesInput', 'options: CPUSpaceTimeHawkesOptions = {}'],
    defaults: [
      '100 default iterations in CPU and GPU forms; maximumIterations is the budget, iterationCount is performed updates; converged distinguishes termination.',
      'An explicit maximumIterations is honored by both forms',
      'options.maximumIterations ?? CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS'
    ],
    limits: [
      'maximumIterations is not an integer in [1, 2^32 - 1].',
      "validateCapacity('computeSpaceTimeHawkesOnCPU', 'maximumIterations', maximumIterations)",
      'positions.length !== times.length * 2',
      'parameters.length < GPU_SPACE_TIME_HAWKES_PARAMETER_LENGTH'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'logLikelihood: finite-window log likelihood at the iterate entering the final update.',
      'fittedLogLikelihood: finite-window log likelihood at the published estimates.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpaceTimeHawkesOperation': {
    signature: 'new CPUSpaceTimeHawkesOperation(props: SpaceTimeHawkesOperationProps)',
    supportedInputs: ['props: SpaceTimeHawkesOperationProps'],
    defaults: [
      "props.id ?? 'space-time-hawkes'",
      '100 default iterations on CPU and GPU; maximumIterations is the budget; iterationCount counts updates; converged reports tolerance termination',
      "/** Prefix for generated node and transient IDs. Defaults to `'space-time-hawkes'`. */",
      '{@link GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS}. Defaults to',
      '{@link GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS}.',
      '`output.status.candidateOverflow`. Defaults to',
      '{@link GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS}.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 2)",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations)",
      "validateCapacity(id, 'maximumParents', props.maximumParents)",
      'times.length !== props.rowCount',
      '100 default iterations on CPU and GPU; maximumIterations is the budget; iterationCount counts updates; converged reports tolerance termination',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'logLikelihood: finite-window log likelihood at the iterate entering the final update.',
      'fittedLogLikelihood: finite-window log likelihood at the published estimates.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpaceTimeHawkesOperation': {
    signature: 'new GPUSpaceTimeHawkesOperation(props: SpaceTimeHawkesOperationProps)',
    supportedInputs: ['props: SpaceTimeHawkesOperationProps'],
    defaults: [
      "props.id ?? 'space-time-hawkes'",
      'props.output.status ?? {}',
      '100 default iterations on CPU and GPU; maximumIterations is the budget; iterationCount counts updates; converged reports tolerance termination',
      "/** Prefix for generated node and transient IDs. Defaults to `'space-time-hawkes'`. */",
      '{@link GPU_SPACE_TIME_HAWKES_MAXIMUM_ITERATIONS}. Defaults to',
      '{@link GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS}.',
      '`output.status.candidateOverflow`. Defaults to',
      '{@link GPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_PARENTS}.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '100 default iterations on CPU and GPU; maximumIterations is the budget; iterationCount counts updates; converged reports tolerance termination',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'logLikelihood: finite-window log likelihood at the iterate entering the final update.',
      'fittedLogLikelihood: finite-window log likelihood at the published estimates.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeShapeDescriptorsOnCPU': {
    signature:
      'export function computeShapeDescriptorsOnCPU( input: CPUPolygonRows, parameters: Float32Array ): CPUShapeDescriptorsResult',
    supportedInputs: ['input: CPUPolygonRows', 'parameters: Float32Array'],
    defaults: 'unknown',
    limits: ['parameters.length < GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH'],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUShapeDescriptorsOperation': {
    signature: 'new CPUShapeDescriptorsOperation(props: ShapeDescriptorsOperationProps)',
    supportedInputs: ['props: ShapeDescriptorsOperationProps'],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'shape-descriptors'`. */",
      "/** Hull algorithm behind `output.convexity`. Default `'auto'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'featureCount', props.featureCount, 1)",
      "validateRequirement( id, Object.keys(operation.outputs).length > 0, 'output must name at least one column' )",
      "validateRequirement( id, input.featureOffsets.length === props.featureCount + 1, 'featureOffsets length must equal featureCount + 1' )",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUShapeDescriptorsOperation': {
    signature: 'new GPUShapeDescriptorsOperation(props: ShapeDescriptorsOperationProps)',
    supportedInputs: ['props: ShapeDescriptorsOperationProps'],
    defaults: [
      "props.id ?? 'shape-descriptors'",
      "/** Prefix for generated node and transient IDs. Defaults to `'shape-descriptors'`. */",
      "/** Hull algorithm behind `output.convexity`. Default `'auto'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMantelTestOnCPU': {
    signature:
      'export function computeMantelTestOnCPU( input: CPUMantelTestInput, options: CPUMantelTestOptions ): CPUMantelTestResult',
    supportedInputs: ['input: CPUMantelTestInput', 'options: CPUMantelTestOptions'],
    defaults: 'unknown',
    limits: [
      "validateCapacity('mantel-test', `status.${name}.length`, flag.length, 1)",
      'offsets.length !== rows + 1 || offsets[0] !== 0',
      'weights.length !== neighbors.length',
      'distances.length !== neighbors.length',
      'parameters.length < GPU_SPACE_TIME_PARAMETER_LENGTH',
      '!times.every(Number.isFinite)',
      'offsets[row + 1] < offsets[row] || offsets[row + 1] > neighbors.length',
      '!Number.isFinite(distances[entry]) || !Number.isFinite(distance)',
      '!temporal.every(Number.isFinite)'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'duplicates are ignored as in the GPU pair mask; malformed CSR offsets throw.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMantelTestOperation': {
    signature: 'new CPUMantelTestOperation(props: MantelTestOperationProps)',
    supportedInputs: ['props: MantelTestOperationProps'],
    defaults: 'unknown',
    limits: [
      "validateRequirement( id, (props.stopAfterExceedances !== undefined) === (props.output.permutationCount !== undefined), 'output.permutationCount is required exactly when stopAfterExceedances is supplied' )",
      'times.length !== props.rowCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMantelTestOperation': {
    signature: 'new GPUMantelTestOperation(props: MantelTestOperationProps)',
    supportedInputs: ['props: MantelTestOperationProps'],
    defaults: ["props.id ?? 'mantel-test'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeOPTICSOnCPU': {
    signature:
      'export function computeOPTICSOnCPU( positions: Float32Array, options: CPUOPTICSOptions, times?: Float32Array ): CPUOPTICSResult',
    supportedInputs: [
      'positions: Float32Array',
      'options: CPUOPTICSOptions',
      'times?: Float32Array'
    ],
    defaults: [
      "options.id ?? 'computeOPTICSOnCPU'",
      'distances[options.minimumSamples - 1] ?? Infinity',
      'options.maximumIterations ?? rowCount',
      '/** Defaults to planar native distances. Geographic positions are lon/lat degrees with',
      '/** Default dbscan. Xi retains nested hierarchy and labels only disjoint innermost clusters. */',
      'Sequential visits per dispatch, default 256. Ignored by parallel ordering.',
      '/** Maximum visited rows, default input length. Incomplete extraction labels stay noise. */',
      'How the reachability step finds rows within `maximumDistance`. Compile-time, default',
      "Compile-time, default 'sequential' for compatibility and its smaller graph.",
      'default batched sequential ordering for a complete million-row ordering.',
      'directions). Compile-time, default `rows * min(rows - 1, 64)`. Too small sets',
      "The default can exceed the device's storage binding limit for large inputs; choose",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '- Drops the GPU row limits (4096/65536), minimumSamples limit (256), batch and command budgets.',
      "validateRequirement(id, positions.length % 2 === 0, 'positions must contain float32x2 rows')",
      "validateRequirement( id, !times || times.length === rowCount, 'times length must equal positions row count' )",
      '`maximumDistance < 2 * eps1`. A row with a non-finite time is invalid like a non-finite',
      "reachability and predecessors. `'brute-force'` accepts at most",
      "{@link GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS} rows; `'grid'` at most",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'Sequential visits per dispatch, default 256. Ignored by parallel ordering.'
    ],
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUOPTICSOperation': {
    signature: 'new CPUOPTICSOperation(props: OPTICSOperationProps)',
    supportedInputs: ['props: OPTICSOperationProps'],
    defaults: [
      "props.id ?? 'optics'",
      "props.neighborIndex ?? 'auto'",
      '/** Defaults to planar native distances. Geographic positions are lon/lat degrees with',
      '/** Default dbscan. Xi retains nested hierarchy and labels only disjoint innermost clusters. */',
      'Sequential visits per dispatch, default 256. Ignored by parallel ordering.',
      '/** Maximum visited rows, default input length. Incomplete extraction labels stay noise. */',
      'How the reachability step finds rows within `maximumDistance`. Compile-time, default',
      "Compile-time, default 'sequential' for compatibility and its smaller graph.",
      'default batched sequential ordering for a complete million-row ordering.',
      'directions). Compile-time, default `rows * min(rows - 1, 64)`. Too small sets',
      "The default can exceed the device's storage binding limit for large inputs; choose",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateRequirement( id, !(output.clusterIntervals || output.clusterSummary) || props.extractionMethod === 'xi', \"clusterIntervals and clusterSummary need extractionMethod 'xi'\" )",
      'validateRequirement( id, grid || (props.neighborCapacity === undefined && output.ports?.status.candidateOverflow === undefined), "neighborCapacity and output.ports.status.candidateOverflow need neighborIndex \'grid\'" )',
      'validateRequirement( id, !(positions.length !== props.rowCount * 2), `positions length must equal rowCount * 2` )',
      '`maximumDistance < 2 * eps1`. A row with a non-finite time is invalid like a non-finite',
      "reachability and predecessors. `'brute-force'` accepts at most",
      "{@link GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS} rows; `'grid'` at most",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['Sequential visits per dispatch, default 256. Ignored by parallel ordering.'],
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOPTICSOperation': {
    signature: 'new GPUOPTICSOperation(props: OPTICSOperationProps)',
    supportedInputs: ['props: OPTICSOperationProps'],
    defaults: [
      "props.id ?? 'optics'",
      '/** Defaults to planar native distances. Geographic positions are lon/lat degrees with',
      '/** Default dbscan. Xi retains nested hierarchy and labels only disjoint innermost clusters. */',
      'Sequential visits per dispatch, default 256. Ignored by parallel ordering.',
      '/** Maximum visited rows, default input length. Incomplete extraction labels stay noise. */',
      'How the reachability step finds rows within `maximumDistance`. Compile-time, default',
      "Compile-time, default 'sequential' for compatibility and its smaller graph.",
      'default batched sequential ordering for a complete million-row ordering.',
      'directions). Compile-time, default `rows * min(rows - 1, 64)`. Too small sets',
      "The default can exceed the device's storage binding limit for large inputs; choose",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '`maximumDistance < 2 * eps1`. A row with a non-finite time is invalid like a non-finite',
      "reachability and predecessors. `'brute-force'` accepts at most",
      "{@link GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS} rows; `'grid'` at most",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['Sequential visits per dispatch, default 256. Ignored by parallel ordering.'],
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-optics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeConditionalGaussianSimulationOnCPU': {
    signature:
      'export function computeConditionalGaussianSimulationOnCPU( input: CPUConditionalGaussianSimulationInput, options: CPUConditionalGaussianSimulationOptions ): CPUConditionalGaussianSimulationResult',
    supportedInputs: [
      'input: CPUConditionalGaussianSimulationInput',
      'options: CPUConditionalGaussianSimulationOptions'
    ],
    defaults: [
      'options.mean ?? 0',
      'options.covariance.nugget ?? 0',
      'options.replicateOffset ?? 0'
    ],
    limits: [
      "validateRequirement( 'computeConditionalGaussianSimulationOnCPU', !(input.positions.length % 2 !== 0), 'positions length must be a multiple of two' )",
      "validateRequirement( 'computeConditionalGaussianSimulationOnCPU', !(input.observations.length !== input.positions.length * 2), 'observations length must equal positions row count * four' )",
      "validateRequirement( 'computeConditionalGaussianSimulationOnCPU', !(input.queries.length % 2 !== 0 || input.queries.length === 0), 'queries must contain one or more two-component rows' )",
      "validateCapacity(id, 'queryCount * replicateCount', queryCount * options.replicateCount)"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-conditional-gaussian-simulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUConditionalGaussianSimulationOperation': {
    signature:
      'new CPUConditionalGaussianSimulationOperation(props: ConditionalGaussianSimulationOperationProps)',
    supportedInputs: ['props: ConditionalGaussianSimulationOperationProps'],
    defaults: [
      "props.id ?? 'conditional-gaussian-simulation'",
      '/** Stable replicate identity across explicit batches; defaults 0. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'observationCount', props.observationCount, 0)",
      "validateCapacity(id, 'queryCount', props.queryCount)",
      "validateCapacity(id, 'queryCount * replicateCount', props.queryCount * props.replicateCount)",
      "validateRequirement( id, observations.length === props.observationCount * 4, 'observations length must equal observationCount * 4' )",
      'validateRequirement( id, !(positions.length !== props.observationCount * 2), `positions length must equal observationCount * 2` )',
      'validateRequirement( id, !(queries.length !== props.queryCount * 2), `queries length must equal queryCount * 2` )'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-conditional-gaussian-simulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeVecchiaFisherScoringOnCPU': {
    signature:
      'export function computeVecchiaFisherScoringOnCPU( input: CPUVecchiaFisherScoringInput ): CPUVecchiaFisherScoringResult',
    supportedInputs: ['input: CPUVecchiaFisherScoringInput'],
    defaults: 'unknown',
    limits: 'unknown',
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUVecchiaFisherScoringOperation': {
    signature: 'new CPUVecchiaFisherScoringOperation(props: VecchiaFisherScoringOperationProps)',
    supportedInputs: ['props: VecchiaFisherScoringOperationProps'],
    defaults: [
      "props.id ?? 'vecchia-fisher-scoring'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'validateRequirement( id, !(positions.length !== props.rowCount * 2), `positions length must equal rowCount * 2` )'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-fisher-scoring-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeVecchiaLogLikelihoodOnCPU': {
    signature:
      'export function computeVecchiaLogLikelihoodOnCPU( input: CPUVecchiaLogLikelihoodInput ): CPUVecchiaLogLikelihoodResult',
    supportedInputs: ['input: CPUVecchiaLogLikelihoodInput'],
    defaults: 'unknown',
    limits: 'unknown',
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUVecchiaLogLikelihoodOperation': {
    signature: 'new CPUVecchiaLogLikelihoodOperation(props: VecchiaLogLikelihoodOperationProps)',
    supportedInputs: ['props: VecchiaLogLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'vecchia-log-likelihood'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'validateRequirement( id, !(positions.length !== props.rowCount * 2), `positions length must equal rowCount * 2` )'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeVecchiaSpaceTimeLogLikelihoodOnCPU': {
    signature:
      'export function computeVecchiaSpaceTimeLogLikelihoodOnCPU( input: CPUVecchiaSpaceTimeLogLikelihoodInput ): CPUVecchiaSpaceTimeLogLikelihoodResult',
    supportedInputs: ['input: CPUVecchiaSpaceTimeLogLikelihoodInput'],
    defaults: [
      "- A NaN structure code selects the Matern 5/2 default branch on CPU; WGSL's NaN-to-u32 result"
    ],
    limits: 'unknown',
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUVecchiaSpaceTimeLogLikelihoodOperation': {
    signature:
      'new CPUVecchiaSpaceTimeLogLikelihoodOperation(props: VecchiaSpaceTimeLogLikelihoodOperationProps)',
    supportedInputs: ['props: VecchiaSpaceTimeLogLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'vecchia-space-time-log-likelihood'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'validateRequirement( id, !(positions.length !== props.rowCount * 2), `positions length must equal rowCount * 2` )',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-space-time-log-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConditionalGaussianSimulationOperation': {
    signature:
      'new GPUConditionalGaussianSimulationOperation(props: ConditionalGaussianSimulationOperationProps)',
    supportedInputs: ['props: ConditionalGaussianSimulationOperationProps'],
    defaults: [
      "props.id ?? 'conditional-gaussian-simulation'",
      '/** Stable replicate identity across explicit batches; defaults 0. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'observationCount', props.observationCount, 0)",
      "validateCapacity(id, 'queryCount', props.queryCount, 1)",
      "validateCapacity(id, 'replicateCount', props.replicateCount, 1)",
      "validateRequirement( id, positions.length === props.observationCount, 'positions.length must equal observationCount' )",
      "validateRequirement( id, observations.length === props.observationCount, 'observations.length must equal observationCount' )",
      "validateRequirement( id, queries.length === props.queryCount, 'queries.length must equal queryCount' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaFisherScoringOperation': {
    signature: 'new GPUVecchiaFisherScoringOperation(props: VecchiaFisherScoringOperationProps)',
    supportedInputs: ['props: VecchiaFisherScoringOperationProps'],
    defaults: [
      "props.id ?? 'vecchia-fisher-scoring'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaLogLikelihoodOperation': {
    signature: 'new GPUVecchiaLogLikelihoodOperation(props: VecchiaLogLikelihoodOperationProps)',
    supportedInputs: ['props: VecchiaLogLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'vecchia-log-likelihood'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVecchiaSpaceTimeLogLikelihoodOperation': {
    signature:
      'new GPUVecchiaSpaceTimeLogLikelihoodOperation(props: VecchiaSpaceTimeLogLikelihoodOperationProps)',
    supportedInputs: ['props: VecchiaSpaceTimeLogLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'vecchia-space-time-log-likelihood'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialJoinPrepared': {
    signature: 'new GPUSpatialJoinPrepared(props: GPUSpatialJoinPreparedProps<Parameters>)',
    supportedInputs: ['props: GPUSpatialJoinPreparedProps<Parameters>'],
    defaults: [
      "props.id ?? 'spatial-join-prepared'",
      'props.edgeSlabs ?? false',
      'props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1))',
      'props.spatialSort ?? getDefaultSpatialSort(this.featureCount)',
      "/** Prefix for generated node, buffer and transient IDs. Defaults to `'spatial-join-prepared'`. */",
      '/** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */',
      'source-row order after probing. Defaults on from 256 features.',
      "preparation. A join given this handle defaults its `edgeSlabs` to the handle's value.",
      'Default false: other join kinds do not read it.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity( this.id, 'geometry view length', view.length, 0, SPATIAL_JOIN_PREPARED_MAXIMUM_ROWS )",
      'this.partitioning',
      'this.partitioning.length !== this.featureCount',
      "this.edgeSlabs && props.geometry.kind !== 'polygons'",
      '!this.preparedStorage',
      '!view || view.length < length',
      'validateCapacity( this.id, `storage.${name}.length`, view.length, length, SPATIAL_JOIN_PREPARED_MAXIMUM_ROWS )',
      'validateRequirement( this.id, view.length === length, `storage.${name}.length must equal ${length} rows for the published BVH layout` )',
      'this.graph',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'three vertices (shorter rings are ignored). Ring 0 of a polygon is its shell and later rings are'
    ],
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialPredicateJoinOnCPU': {
    signature:
      'export function computeSpatialPredicateJoinOnCPU( left: CPUSpatialJoinGeometry, right: CPUSpatialJoinGeometry, options: CPUFeatureJoinOptions )',
    supportedInputs: [
      'left: CPUSpatialJoinGeometry',
      'right: CPUSpatialJoinGeometry',
      'options: CPUFeatureJoinOptions'
    ],
    defaults: [
      '/** Materialize packed matrices for retained inner pairs. Defaults to false (an empty array). */'
    ],
    limits: 'unknown',
    optionHandling: [
      'What differs from the GPU contributor: rational CPU topology has no orientation uncertainty;'
    ],
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-join/cpu-feature-join.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeNearestFeatureJoinOnCPU': {
    signature:
      'export function computeNearestFeatureJoinOnCPU( queries: CPUSpatialJoinGeometry, targets: CPUSpatialJoinGeometry )',
    supportedInputs: ['queries: CPUSpatialJoinGeometry', 'targets: CPUSpatialJoinGeometry'],
    defaults: 'unknown',
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-spatial-analysis/spatial-join/cpu-feature-join.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeBufferSelectionOnCPU': {
    signature:
      "export function computeBufferSelectionOnCPU( input: CPUBufferSelectionInput, options: CPUBufferSelectionOptions ): Record<keyof BufferSelectionOperationProps['output'], Uint32Array | Float32Array>",
    supportedInputs: ['input: CPUBufferSelectionInput', 'options: CPUBufferSelectionOptions'],
    defaults: ['options.leafCapacity ?? 2 ** Math.ceil(Math.log2(Math.max(1, featureCount)))'],
    limits: [
      "validateRequirement( id, (input.features.kind === 'segments' || (input.features.kind === 'points' && !input.features.featureOffsets)) && starts.length % 2 === 0 && ends.length === starts.length, 'features must contain equal-length starts/ends XY segment arrays or individual point rows' )",
      "validateCapacity(id, 'candidateCapacity', options.candidateCapacity)",
      "validateCapacity(id, 'matchCapacity', options.matchCapacity)",
      "validateCapacity(id, 'point feature count', queryCount)"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Invalid offset layouts throw on CPU. Nonfinite points are ignored, like GPU query probes.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePointInPolygonJoinOnCPU': {
    signature:
      "export function computePointInPolygonJoinOnCPU( input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry}, options: CPUPointInPolygonJoinOptions ): Record<keyof PointInPolygonJoinOperationProps['output'], Uint32Array>",
    supportedInputs: [
      'input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry}',
      'options: CPUPointInPolygonJoinOptions'
    ],
    defaults: 'unknown',
    limits: 'unknown',
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePointInPolygonCountsOnCPU': {
    signature:
      'export function computePointInPolygonCountsOnCPU(input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry}, options: CPUPointInPolygonJoinOptions): CPUPointInPolygonCountsResult',
    supportedInputs: [
      'input: {points: CPUSpatialJoinGeometry; polygons: CPUSpatialJoinGeometry}',
      'options: CPUPointInPolygonJoinOptions'
    ],
    defaults: 'unknown',
    limits: 'unknown',
    optionHandling: [
      'Sparse counts and assignments without featurePolygonMembership; shares the join predicates and capacity/loss checks.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeWardRegionsOnCPU': {
    signature:
      'export function computeWardRegionsOnCPU( input: { offsets: Uint32Array; neighbors: Uint32Array; values: Float32Array; parameters: Uint32Array; status?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array}; }, options: CPUWardRegionsOptions ): CPUWardRegionsResult',
    supportedInputs: [
      'input: { offsets: Uint32Array; neighbors: Uint32Array; values: Float32Array; parameters: Uint32Array; status?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array}; }',
      'options: CPUWardRegionsOptions'
    ],
    defaults: [
      '- Honors maximumIterations; omitted in the pure function it defaults to rows minus the target.',
      "options.id ?? 'computeWardRegionsOnCPU'",
      'options.columnCount ?? 1',
      "options.mergeSchedule ?? 'sequential'",
      'options.statusLength ?? GPU_WARD_STATUS_LAYOUT.length',
      'options.maximumIterations ?? rowCount - options.targetRegionCount',
      '/** Status words: four by default, or five to include the mutual-nearest round count. */'
    ],
    limits: [
      "validateCapacity(id, 'maximumIterations', maximumIterations, 0)",
      "validateCapacity(id, 'rowCount', rowCount)",
      "validateCapacity(id, 'columnCount', columnCount)",
      "validateCapacity( id, 'statusLength', statusLength, GPU_WARD_STATUS_LAYOUT.length, GPU_WARD_ROUND_COUNT_STATUS_WORD + 1 )",
      "validateCapacity(id, 'targetRegionCount', options.targetRegionCount, 1, rowCount)",
      "validateRequirement( id, ['sequential', 'mutual-nearest'].includes(mergeSchedule), \"mergeSchedule must be 'sequential' or 'mutual-nearest'\", { code: 'INVALID_OPTION', path: 'mergeSchedule', received: mergeSchedule, expected: ['sequential', 'mutual-nearest'] } )",
      "validateRequirement( id, false, `values length must equal rowCount * columnCount; received values.length=${values.length}`, { code: 'VIEW_LENGTH', path: 'values.length', received: values.length, expected: 'values length must equal rowCount * columnCount' } )",
      "validateRequirement( id, false, `parameters is too short; received parameters.length=${parameters.length}`, { code: 'VIEW_LENGTH', path: 'parameters.length', received: parameters.length, expected: 'parameters is too short' } )",
      "validateRequirement(id, offsets[0] === 0, 'offsets must start at 0', { code: 'REQUIREMENT', path: 'offsets[0]', received: offsets[0], expected: 0 })",
      "validateRequirement( id, offsets[row] <= offsets[row + 1] && offsets[row + 1] <= neighbors.length, 'offsets must be monotone and fit neighbors length', { code: 'REQUIREMENT', path: `offsets[${row + 1}]`, received: offsets[row + 1], expected: {minimum: offsets[row], maximum: neighbors.length} } )",
      'validateCapacity(id, `status.${name}.length`, values.length, 1)'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUWardRegionsOperation': {
    signature: 'new CPUWardRegionsOperation(props: WardRegionsOperationProps)',
    supportedInputs: ['props: WardRegionsOperationProps'],
    defaults: [
      "props.id ?? 'ward-regions'",
      'props.columnCount ?? 1',
      'props.statusLength ?? GPU_WARD_STATUS_LAYOUT.length',
      "props.mergeSchedule ?? 'sequential'",
      '/** Status words: four by default, or five to include the mutual-nearest round count. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'columnCount', props.columnCount ?? 1)",
      "validateCapacity( id, 'statusLength', props.statusLength ?? GPU_WARD_STATUS_LAYOUT.length, GPU_WARD_STATUS_LAYOUT.length, GPU_WARD_ROUND_COUNT_STATUS_WORD + 1 )",
      "validateCapacity(id, 'targetRegionCount', props.targetRegionCount, 1, props.rowCount)",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations, 0)",
      "validateRequirement( id, ['sequential', 'mutual-nearest'].includes(props.mergeSchedule ?? 'sequential'), \"mergeSchedule must be 'sequential' or 'mutual-nearest'\", { code: 'INVALID_OPTION', path: 'mergeSchedule', received: props.mergeSchedule, expected: ['sequential', 'mutual-nearest'] } )",
      "validateRequirement( id, false, `offsets length must equal rowCount + 1; received offsets.length=${offsets.length}`, { code: 'VIEW_LENGTH', path: 'offsets.length', received: offsets.length, expected: 'offsets length must equal rowCount + 1' } )",
      "validateRequirement( id, false, `weights length must equal neighbors length; received weights.length=${weights.length}, neighbors.length=${neighbors.length}`, { code: 'VIEW_LENGTH', path: 'weights.length', received: weights.length, expected: 'weights length must equal neighbors length' } )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUWardRegionsOperation': {
    signature: 'new GPUWardRegionsOperation(props: WardRegionsOperationProps)',
    supportedInputs: ['props: WardRegionsOperationProps'],
    defaults: [
      "props.id ?? 'ward-regions'",
      'props.statusLength ?? GPU_WARD_STATUS_LAYOUT.length',
      '/** Status words: four by default, or five to include the mutual-nearest round count. */'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSkaterRegionsOnCPU': {
    signature:
      'export function computeSkaterRegionsOnCPU( input: CPUSkaterRegionsInput, options: CPUSkaterRegionsOptions ): CPUSkaterRegionsResult',
    supportedInputs: ['input: CPUSkaterRegionsInput', 'options: CPUSkaterRegionsOptions'],
    defaults: ["options.id ?? 'computeSkaterRegionsOnCPU'", 'options.columnCount ?? 1'],
    limits: [
      "validateRequirement( id, false, `treeStatus must hold GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length words; received input.treeStatus.length=${input.treeStatus.length}, GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length=${GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length}`, { code: 'VIEW_LENGTH', path: 'treeStatus.length', received: input.treeStatus.length, expected: 'treeStatus must hold GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length words' } )",
      'validateCapacity(id, `weights.status.${name} length`, input.status[name]!.length, 1)',
      "validateCapacity(id, 'rowCount', rowCount)",
      "validateRequirement(id, offsets[0] === 0, 'offsets must start at zero', { code: 'REQUIREMENT', path: 'offsets[0]', received: offsets[0], expected: 0 })",
      "validateRequirement( id, offsets[row] >= offsets[row - 1] && offsets[row] <= neighbors.length, 'offsets must be nondecreasing CSR offsets within neighbors length', { code: 'REQUIREMENT', path: `offsets[${row}]`, received: offsets[row], expected: {minimum: offsets[row - 1], maximum: neighbors.length} } )",
      "validateRequirement( id, false, `weights length must equal neighbors length; received input.weights.length=${input.weights.length}, neighbors.length=${neighbors.length}`, { code: 'VIEW_LENGTH', path: 'weights.length', received: input.weights.length, expected: 'weights length must equal neighbors length' } )",
      "validateRequirement( id, false, `distances length must equal neighbors length; received input.distances.length=${input.distances.length}, neighbors.length=${neighbors.length}`, { code: 'VIEW_LENGTH', path: 'distances.length', received: input.distances.length, expected: 'distances length must equal neighbors length' } )",
      "validateRequirement( id, false, `treeEdgeFlags length must equal neighbors length; received treeEdgeFlags.length=${treeEdgeFlags.length}, neighbors.length=${neighbors.length}`, { code: 'VIEW_LENGTH', path: 'treeEdgeFlags.length', received: treeEdgeFlags.length, expected: 'treeEdgeFlags length must equal neighbors length' } )",
      "validateRequirement( id, false, `componentLabels length must equal row count; received componentLabels.length=${componentLabels.length}`, { code: 'VIEW_LENGTH', path: 'componentLabels.length', received: componentLabels.length, expected: 'componentLabels length must equal row count' } )",
      "validateRequirement( id, false, `values length must equal row count * columnCount; received values.length=${values.length}`, { code: 'VIEW_LENGTH', path: 'values.length', received: values.length, expected: 'values length must equal row count * columnCount' } )",
      "validateRequirement( id, false, `parameters must hold the getGPUSkaterRegionsParameterValues layout; received parameters.length=${parameters.length}`, { code: 'VIEW_LENGTH', path: 'parameters.length', received: parameters.length, expected: 'parameters must hold the getGPUSkaterRegionsParameterValues layout' } )",
      "validateRequirement( id, false, `neighbors must be valid strictly ascending row IDs; received rowCount=${rowCount}, neighbor=${neighbor}, row=${row}`, { code: 'REQUIREMENT', path: `neighbors[${slot}]`, received: neighbor, expected: 'neighbors must be valid strictly ascending row IDs' } )",
      "validateRequirement( id, false, `neighbors must list reciprocal links; received reverse=${reverse}`, { code: 'VIEW_LENGTH', path: 'reverse', received: reverse, expected: 'neighbors must list reciprocal links' } )",
      "validateRequirement( id, false, `treeEdgeFlags must not select self links; received neighbor=${neighbor}, row=${row}`, { code: 'REQUIREMENT', path: `treeEdgeFlags[${slot}]`, received: treeEdgeFlags[slot], expected: 'treeEdgeFlags must not select self links' } )",
      "validateRequirement( id, false, `componentLabels must name the smallest row of each flagged tree; received row=${row}`, { code: 'REQUIREMENT', path: `componentLabels[${row}]`, received: componentLabels[row], expected: 'componentLabels must name the smallest row of each flagged tree' } )",
      "validateRequirement( id, false, `treeEdgeFlags must select an acyclic forest; received neighbor=${neighbor}`, { code: 'REQUIREMENT', path: `treeEdgeFlags[${edge.slot}]`, received: treeEdgeFlags[edge.slot], expected: 'treeEdgeFlags must select an acyclic forest' } )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSkaterRegionsOperation': {
    signature: 'new CPUSkaterRegionsOperation(props: SkaterRegionsOperationProps)',
    supportedInputs: ['props: SkaterRegionsOperationProps'],
    defaults: [
      "props.id ?? 'skater-regions'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'edgeCount', props.edgeCount, 0)",
      "validateRequirement( id, false, `offsets length must equal rowCount + 1; received offsets.length=${offsets.length}`, { code: 'VIEW_LENGTH', path: 'offsets.length', received: offsets.length, expected: 'offsets length must equal rowCount + 1' } )",
      "validateRequirement( id, false, `neighbors length must equal edgeCount; received neighbors.length=${neighbors.length}`, { code: 'VIEW_LENGTH', path: 'neighbors.length', received: neighbors.length, expected: 'neighbors length must equal edgeCount' } )"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSkaterRegionsOperation': {
    signature: 'new GPUSkaterRegionsOperation(props: SkaterRegionsOperationProps)',
    supportedInputs: ['props: SkaterRegionsOperationProps'],
    defaults: [
      "props.id ?? 'skater-regions'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeAZPRegionsOnCPU': {
    signature:
      'export function computeAZPRegionsOnCPU( input: CPUAZPRegionsInput, options: CPUAZPRegionsOptions ): CPUAZPRegionsResult',
    supportedInputs: ['input: CPUAZPRegionsInput', 'options: CPUAZPRegionsOptions'],
    defaults: [
      '- Honors maximumIterations (default 256 for the pure function). A final successful batch at',
      "options.id ?? 'computeAZPRegionsOnCPU'",
      'options.columnCount ?? 1',
      'options.minimumRegionSize ?? 1',
      'options.maximumIterations ?? 256',
      '/** Explicit CPU round budget; defaults to 256 when the pure function is called without one. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- Honors maximumIterations (default 256 for the pure function). A final successful batch at',
      "validateRequirement( id, false, `initialLabels length must equal rowCount; received input.initialLabels.length=${input.initialLabels.length}`, { code: 'VIEW_LENGTH', path: 'initialLabels.length', received: input.initialLabels.length, expected: 'initialLabels length must equal rowCount' } )",
      "validateRequirement(id, false, 'extensiveValues and extensiveThreshold are required together', { code: 'MISSING_PROP', path: 'extensiveValues', received: { extensiveValues: input.extensiveValues, extensiveThreshold: input.extensiveThreshold }, expected: 'both views or neither' })",
      "validateRequirement( id, false, `extensiveValues length must equal rowCount; received input.extensiveValues.length=${input.extensiveValues.length}`, { code: 'VIEW_LENGTH', path: 'extensiveValues.length', received: input.extensiveValues.length, expected: 'extensiveValues length must equal rowCount' } )",
      "validateRequirement( id, false, `extensiveThreshold must hold one value; received input.extensiveThreshold.length=${input.extensiveThreshold.length}`, { code: 'VIEW_LENGTH', path: 'extensiveThreshold.length', received: input.extensiveThreshold.length, expected: 'extensiveThreshold must hold one value' } )"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUAZPRegionsOperation': {
    signature: 'new CPUAZPRegionsOperation(props: AZPRegionsOperationProps)',
    supportedInputs: ['props: AZPRegionsOperationProps'],
    defaults: [
      "props.id ?? 'azp-regions'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 2)",
      "validateRequirement( id, typeof props.maximumIterations === 'number', 'maximumIterations must be a positive integer' )",
      "validateRequirement( id, (props.extensiveValues === undefined) === (props.extensiveThreshold === undefined), 'extensiveValues and extensiveThreshold are required together' )",
      "validateRequirement( id, false, `offsets length must equal rowCount + 1; received offsets.length=${offsets.length}`, { code: 'VIEW_LENGTH', path: 'offsets.length', received: offsets.length, expected: 'offsets length must equal rowCount + 1' } )"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMaxPRegionsOnCPU': {
    signature:
      'export function computeMaxPRegionsOnCPU( input: CPUMaxPRegionsInput, options: CPUMaxPRegionsOptions = {} ): CPUMaxPRegionsResult & {evidence: Uint32Array}',
    supportedInputs: ['input: CPUMaxPRegionsInput', 'options: CPUMaxPRegionsOptions = {}'],
    defaults: [
      "options.id ?? 'computeMaxPRegionsOnCPU'",
      'options.columnCount ?? 1',
      'options.maximumIterations ?? 0',
      'options.startCount ?? GPU_MAX_P_DEFAULT_START_COUNT',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- Drops GPU column, start-count, storage and command-node limits; uint32 layouts remain.',
      "validateRequirement( id, (options.maximumIterations ?? 0) === 0 || rowCount >= 2, 'maximumIterations needs at least 2 rows', {code: 'INTEGER_RANGE', path: 'rowCount', received: rowCount, expected: {minimum: 2}} )",
      "validateRequirement( id, false, `extensiveValues length must equal rowCount; received input.extensiveValues.length=${input.extensiveValues.length}`, { code: 'VIEW_LENGTH', path: 'extensiveValues.length', received: input.extensiveValues.length, expected: 'extensiveValues length must equal rowCount' } )",
      "validateRequirement( id, false, `threshold must hold one value; received input.threshold.length=${input.threshold.length}`, { code: 'VIEW_LENGTH', path: 'threshold.length', received: input.threshold.length, expected: 'threshold must hold one value' } )"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMaxPRegionsOperation': {
    signature: 'new CPUMaxPRegionsOperation(props: MaxPRegionsOperationProps)',
    supportedInputs: ['props: MaxPRegionsOperationProps'],
    defaults: [
      "props.id ?? 'max-p-regions'",
      'props.maximumIterations ?? 0',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement( id, (props.maximumIterations ?? 0) === 0 || props.rowCount >= 2, 'maximumIterations needs at least 2 rows' )",
      "validateRequirement( id, false, `offsets length must equal rowCount + 1; received offsets.length=${offsets.length}`, { code: 'VIEW_LENGTH', path: 'offsets.length', received: offsets.length, expected: 'offsets length must equal rowCount + 1' } )"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUAZPRegionsOperation': {
    signature: 'new GPUAZPRegionsOperation(props: AZPRegionsOperationProps)',
    supportedInputs: ['props: AZPRegionsOperationProps'],
    defaults: [
      "props.id ?? 'azp-regions'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMaxPRegionsOperation': {
    signature: 'new GPUMaxPRegionsOperation(props: MaxPRegionsOperationProps)',
    supportedInputs: ['props: MaxPRegionsOperationProps'],
    defaults: [
      "props.id ?? 'max-p-regions'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPanelErrorMaximumLikelihoodOperation':
    {
      signature:
        'new GPUSpatialPanelErrorMaximumLikelihoodOperation(props: SpatialPanelErrorMaximumLikelihoodOperationProps)',
      supportedInputs: ['props: SpatialPanelErrorMaximumLikelihoodOperationProps'],
      defaults: [
        "props.id ?? 'spatial-panel-error-maximum-likelihood'",
        'props.maximumPower ?? 128',
        "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-maximum-likelihood'`. */",
        '/** Series length `Q` of the log-determinant and trace series, an integer in `[2, 256]`. Defaults to 128. */',
        "`'exact'` or `'stochastic'` (Hutchinson) traces. Defaults to exact up to 4096 rows and",
        'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
        "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
        '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
        'Rows reduced by one invocation of the accumulation passes. Defaults to',
        'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
        "/** Fixed effects to absorb; defaults to 'unit'. Time/twoWay require predictorCount + periodCount - 1 <= 15. Random effects are refused because their independent-unit covariance needs a second determinant, unsupported by this trace-series fit. */"
      ],
      limits: [
        "validateRequirement( id, typeof value === 'string' && value.length > 0, `weights.status.${name} must name a non-empty graph value` )",
        "validateRequirement( id, getView('offsets').length === props.entityCount + 1, 'entityCount must equal weights.offsets.length - 1' )",
        '`GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
        '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
        'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
      ],
      optionHandling: [
        'outputs must select the fields they actually write and reject unsupported requests before',
        "/** Fixed effects to absorb; defaults to 'unit'. Time/twoWay require predictorCount + periodCount - 1 <= 15. Random effects are refused because their independent-unit covariance needs a second determinant, unsupported by this trace-series fit. */"
      ],
      source:
        'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation.ts'
    },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialPanelErrorMaximumLikelihoodOperation':
    {
      signature:
        'new CPUSpatialPanelErrorMaximumLikelihoodOperation(props: SpatialPanelErrorMaximumLikelihoodOperationProps)',
      supportedInputs: ['props: SpatialPanelErrorMaximumLikelihoodOperationProps'],
      defaults: [
        "props.id ?? 'spatial-panel-error-maximum-likelihood'",
        "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-maximum-likelihood'`. */",
        '/** Series length `Q` of the log-determinant and trace series, an integer in `[2, 256]`. Defaults to 128. */',
        "`'exact'` or `'stochastic'` (Hutchinson) traces. Defaults to exact up to 4096 rows and",
        'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
        "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
        '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
        'Rows reduced by one invocation of the accumulation passes. Defaults to',
        'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
        "/** Fixed effects to absorb; defaults to 'unit'. Time/twoWay require predictorCount + periodCount - 1 <= 15. Random effects are refused because their independent-unit covariance needs a second determinant, unsupported by this trace-series fit. */"
      ],
      limits: [
        "validateCapacity(id, 'entityCount', props.entityCount)",
        "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
        "validateRequirement( id, (props.logDeterminant === 'exact') === (props.eigenvalues !== undefined), 'eigenvalues are required exactly when logDeterminant is exact' )",
        "validateRequirement( id, getCPUOperationUint32Rows(id, inputs, 'offsets').length === props.entityCount + 1, 'entityCount must equal weights.offsets.length - 1' )",
        "validateRequirement( id, result.residuals.length === props.entityCount * props.periodCount, 'entityCount must match weights row count' )",
        '`GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
        '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
        'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
      ],
      optionHandling: [
        'outputs must select the fields they actually write and reject unsupported requests before',
        "/** Fixed effects to absorb; defaults to 'unit'. Time/twoWay require predictorCount + periodCount - 1 <= 15. Random effects are refused because their independent-unit covariance needs a second determinant, unsupported by this trace-series fit. */"
      ],
      source:
        'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation.ts'
    },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialPanelErrorMaximumLikelihoodOnCPU':
    {
      signature:
        'export function computeSpatialPanelErrorMaximumLikelihoodOnCPU( input: CPUSpatialPanelErrorMaximumLikelihoodInput, options: CPUSpatialPanelErrorMaximumLikelihoodOptions ): CPUSpatialPanelErrorMaximumLikelihoodResult',
      supportedInputs: [
        'input: CPUSpatialPanelErrorMaximumLikelihoodInput',
        'options: CPUSpatialPanelErrorMaximumLikelihoodOptions'
      ],
      defaults: [
        'options.maximumPower ?? 128',
        "options.traceMode ?? (rows <= 4096 ? 'exact' : 'stochastic')",
        'options.traceProbeCount ?? 64',
        "options.effects ?? 'unit'",
        "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-maximum-likelihood'`. */",
        '/** Series length `Q` of the log-determinant and trace series, an integer in `[2, 256]`. Defaults to 128. */',
        "`'exact'` or `'stochastic'` (Hutchinson) traces. Defaults to exact up to 4096 rows and",
        'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
        "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
        '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
        'Rows reduced by one invocation of the accumulation passes. Defaults to',
        'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
        "/** Fixed effects to absorb; defaults to 'unit'. Time/twoWay require predictorCount + periodCount - 1 <= 15. Random effects are refused because their independent-unit covariance needs a second determinant, unsupported by this trace-series fit. */"
      ],
      limits: [
        "validateRequirement( id, input.response.length === rows, 'response length must equal entities * periodCount' )",
        "validateRequirement( id, input.predictors.length === rows * predictorCount, 'predictors length must equal response length * predictorCount' )",
        "validateRequirement( id, exact === (input.eigenvalues !== undefined), 'eigenvalues are required exactly when logDeterminant is exact' )",
        'validateRequirement( id, input.eigenvalues.length === entities || input.eigenvalues.length === rows, `eigenvalues length ${input.eigenvalues.length} must equal entityCount ${entities} or total panel rows ${rows}` )',
        "validateRequirement( id, input.traceParameters.length >= 1, 'traceParameters must hold the packed seed' )",
        "validateRequirement( id, input.weights.status.overflow.length >= 1, 'weights.status.overflow must hold one word' )",
        "validateRequirement( id, input.weights.status.candidateOverflow.length >= 1, 'weights.status.candidateOverflow must hold one word' )",
        '`GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
        '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
        'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
      ],
      optionHandling: [
        'What differs from the GPU contributor:',
        'outputs must select the fields they actually write and reject unsupported requests before',
        "/** Fixed effects to absorb; defaults to 'unit'. Time/twoWay require predictorCount + periodCount - 1 <= 15. Random effects are refused because their independent-unit covariance needs a second determinant, unsupported by this trace-series fit. */"
      ],
      source:
        'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation.ts'
    },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeOrdinaryLeastSquaresOnCPU': {
    signature:
      'export function computeOrdinaryLeastSquaresOnCPU( input: CPUOrdinaryLeastSquaresInput, predictorCount: number ): CPUOrdinaryLeastSquaresResult',
    supportedInputs: ['input: CPUOrdinaryLeastSquaresInput', 'predictorCount: number'],
    defaults: 'unknown',
    limits: [
      'the GPU retains its 200-iteration budget for at most 15 degrees of freedom.',
      "validateCapacity(id, 'predictorCount', predictorCount)",
      'input.predictors.length !== rowCount * predictorCount',
      'input.mask && input.mask.length !== rowCount',
      'input.parameters && input.parameters.length < GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'corrections. tileRowCount is ignored. Comparisons on well-conditioned inputs use 5e-3 relative'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUOrdinaryLeastSquaresOperation': {
    signature: 'new CPUOrdinaryLeastSquaresOperation(props: OrdinaryLeastSquaresOperationProps)',
    supportedInputs: ['props: OrdinaryLeastSquaresOperationProps'],
    defaults: [
      "props.id ?? 'ordinary-least-squares'",
      "/** Prefix for generated node and transient IDs. Defaults to `'ordinary-least-squares'`. */",
      'Rows reduced by one invocation in the accumulation passes. Defaults to'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'predictorCount', props.predictorCount)",
      "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
      "validateRequirement( id, props.predictorCount + 1 <= 0xffffffff, 'predictorCount + 1 must fit uint32' )",
      'response.length !== props.rowCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeRestrictedRegimeRegressionOnCPU': {
    signature:
      'export function computeRestrictedRegimeRegressionOnCPU( input: CPURestrictedRegimeRegressionInput, options: CPURestrictedRegimeRegressionOptions ): CPURestrictedRegimeRegressionResult',
    supportedInputs: [
      'input: CPURestrictedRegimeRegressionInput',
      'options: CPURestrictedRegimeRegressionOptions'
    ],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, robust.type === 'white' || robust.type === 'hac', \"robust.type must be 'white' or 'hac'\" )",
      "validateRequirement( id, robust.smallSample === undefined || typeof robust.smallSample === 'boolean', 'robust.smallSample must be boolean' )",
      "validateRequirement( id, robust.kernelWeights.offsets.length === rowCount + 1, 'robust.kernelWeights.offsets length must equal response length + 1' )",
      "validateRequirement( id, robust.kernelWeights.neighbors.length === robust.kernelWeights.weights.length, 'robust.kernelWeights.weights length must equal neighbors length' )",
      "validateRequirement( id, values.length === 1, 'robust.kernelWeights.status ports must have length 1' )",
      'input.predictors.length !== rowCount * predictorCount',
      'input.regimes.length !== rowCount',
      'input.mask && input.mask.length !== rowCount'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPURestrictedRegimeRegressionOperation': {
    signature:
      'new CPURestrictedRegimeRegressionOperation(props: RestrictedRegimeRegressionOperationProps)',
    supportedInputs: ['props: RestrictedRegimeRegressionOperationProps'],
    defaults: [
      'props.robust.kernelWeights.status ?? {}',
      "props.id ?? 'restricted-regime-regression'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      'response.length !== props.rowCount'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-restricted-regime-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSeeminglyUnrelatedRegressionOnCPU': {
    signature:
      'export function computeSeeminglyUnrelatedRegressionOnCPU( input: CPUSeeminglyUnrelatedRegressionInput, options: CPUSeeminglyUnrelatedRegressionOptions ): CPUSeeminglyUnrelatedRegressionResult',
    supportedInputs: [
      'input: CPUSeeminglyUnrelatedRegressionInput',
      'options: CPUSeeminglyUnrelatedRegressionOptions'
    ],
    defaults: [
      'Stops when max abs(new-old)/max(1,abs(old)) < tolerance (default 1e-5).',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '/** Opt into iterated FGLS. The budget includes the first GLS fit; maximum 256 rounds.'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSeeminglyUnrelatedRegressionOperation': {
    signature:
      'new CPUSeeminglyUnrelatedRegressionOperation(props: SeeminglyUnrelatedRegressionOperationProps)',
    supportedInputs: ['props: SeeminglyUnrelatedRegressionOperationProps'],
    defaults: [
      "props.id ?? 'seemingly-unrelated-regression'",
      'Stops when max abs(new-old)/max(1,abs(old)) < tolerance (default 1e-5).',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, !props.parameters || Boolean(props.iteration), 'parameters requires iteration; omit parameters for one-step SUR' )",
      "validateRequirement( id, !props.spatial || (!props.mask && !props.initialResiduals), 'spatial requires no mask or initialResiduals' )",
      '/** Opt into iterated FGLS. The budget includes the first GLS fit; maximum 256 rounds.'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-seemingly-unrelated-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialLagImpactsOnCPU': {
    signature:
      'export function computeSpatialLagImpactsOnCPU( input: CPUSpatialLagImpactsInput, options: CPUSpatialLagImpactsOptions ): {impacts: Float32Array}',
    supportedInputs: ['input: CPUSpatialLagImpactsInput', 'options: CPUSpatialLagImpactsOptions'],
    defaults: [
      'options.firstPredictorIndex ?? 1',
      'options.rhoBound ?? 1',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-impacts'`. */",
      '/** Row of the first predictor (the intercept is excluded). Defaults to 1. */',
      'larger values give NaN. The series needs it below 1, which is the default and is correct for'
    ],
    limits: [
      '- The shader-only maximumPower <= 256 limit is dropped; the requested truncation is retained.',
      'input.coefficients.length < lastRow * options.coefficientStride + 1',
      'input[name].length < options.maximumPower',
      'input.infinityNorm.length < 1',
      'input.status && input.status.length < 1',
      '/** Series length `Q` used, at most the length of each traces view. */'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-impacts-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialLagImpactsOperation': {
    signature: 'new CPUSpatialLagImpactsOperation(props: SpatialLagImpactsOperationProps)',
    supportedInputs: ['props: SpatialLagImpactsOperationProps'],
    defaults: [
      "props.id ?? 'spatial-lag-impacts'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-impacts'`. */",
      '/** Row of the first predictor (the intercept is excluded). Defaults to 1. */',
      'larger values give NaN. The series needs it below 1, which is the default and is correct for'
    ],
    limits: [
      "validateRequirement( id, ![props.coefficients, ...Object.values(props.traces), props.status].includes( props.output.impacts ), 'output.impacts must not name an input graph value' )",
      '/** Series length `Q` used, at most the length of each traces view. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-impacts-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialRegressionDiagnosticsOnCPU': {
    signature:
      'export function computeSpatialRegressionDiagnosticsOnCPU( input: CPUSpatialRegressionDiagnosticsInput, options: CPUSpatialRegressionDiagnosticsOptions ): CPUSpatialRegressionDiagnosticsResult',
    supportedInputs: [
      'input: CPUSpatialRegressionDiagnosticsInput',
      'options: CPUSpatialRegressionDiagnosticsOptions'
    ],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-regression-diagnostics'`. */",
      "1, which is `libpysal.weights.w_subset` followed by `transform = 'r'`. Default false: the",
      '{@link getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount} (2^25 on a default device).',
      'Rows reduced by one invocation of the accumulation pass. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'predictorCount', options.predictorCount)",
      'input.response.length !== rowCount',
      'input.residuals.length !== rowCount',
      'input.predictors.length !== rowCount * predictorCount',
      'input.mask && input.mask.length !== rowCount',
      'neighbors[slot] <= neighbors[slot - 1]',
      '`rows * (predictorCount + 3)` is at most',
      '{@link getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount} (2^25 on a default device).',
      'table holds at most {@link getGPUSpatialRegressionDiagnosticsMaximumTileMomentCount} values.'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Shader-only 15-predictor and device binding-size limits are dropped; tileRowCount is ignored.',
      'unused capacity is ignored, and non-finite active weights produce status 4. The pure function',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialRegressionDiagnosticsOperation': {
    signature:
      'new CPUSpatialRegressionDiagnosticsOperation(props: SpatialRegressionDiagnosticsOperationProps)',
    supportedInputs: ['props: SpatialRegressionDiagnosticsOperationProps'],
    defaults: [
      "props.id ?? 'spatial-regression-diagnostics'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-regression-diagnostics'`. */",
      "1, which is `libpysal.weights.w_subset` followed by `transform = 'r'`. Default false: the",
      '{@link getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount} (2^25 on a default device).',
      'Rows reduced by one invocation of the accumulation pass. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'predictorCount', props.predictorCount)",
      "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
      'response.length !== props.rowCount',
      '`rows * (predictorCount + 3)` is at most',
      '{@link getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount} (2^25 on a default device).',
      'table holds at most {@link getGPUSpatialRegressionDiagnosticsMaximumTileMomentCount} values.'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialWeightsTracesOnCPU': {
    signature:
      'export function computeSpatialWeightsTracesOnCPU( input: CPUSpatialWeightsTracesInput, options: CPUSpatialWeightsTracesOptions ): CPUSpatialWeightsTracesResult',
    supportedInputs: [
      'input: CPUSpatialWeightsTracesInput',
      'options: CPUSpatialWeightsTracesOptions'
    ],
    defaults: [
      '- The shader power cap and explicit-exact row cap are dropped. The same mode default switches',
      "options.mode ?? (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS ? 'exact' : 'stochastic')",
      'options.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-traces'`. */",
      "and `'stochastic'` runs Hutchinson estimation with `probeCount` Rademacher probes. Defaults to",
      '/** Probes of the stochastic mode. Defaults to 64. */',
      '/** Also compute `output.gramTraces`. Defaults to whether `output.gramTraces` is present. */',
      'Rows reduced by one invocation of the dot-product passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'weights.rowCount', rowCount, 0)",
      "validateRequirement( id, !chunked || !options.includeGramTraces, 'weights chunks do not support Gram traces' )",
      'validateCapacity(id, `weights.chunks[${index}].rowCount`, chunk.rowCount, 0)',
      'validateRequirement( id, chunk.rowStart === nextRow, `weights.chunks[${index}].rowStart must equal ${nextRow}` )',
      'validateRequirement( id, validateCPURegressionWeights(`${id} weights.chunks[${index}]`, chunk.weights) === chunk.rowCount, `weights.chunks[${index}].offsets length must equal rowCount + 1` )',
      "validateRequirement(id, nextRow === rowCount, 'weights.chunks must cover weights.rowCount rows')",
      'input.parameters && input.parameters.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH',
      "`'exact'` when `rows <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS` (4096) and to",
      'batch is also limited by the 64 MiB scratch budget, so this can only shrink it. Results are',
      '/** At least `maximumPower` float32 values: `tr((W / s)^q)` at index `q - 1`, `s = infinityNorm`. */',
      'Optional, at least `maximumPower` float32 values: the standard error of the mean of the',
      "/** At least `maximumPower` float32 values: `1' (W / s)^q 1` at index `q - 1` (always exact). */"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialWeightsTracesOperation': {
    signature: 'new CPUSpatialWeightsTracesOperation(props: SpatialWeightsTracesOperationProps)',
    supportedInputs: ['props: SpatialWeightsTracesOperationProps'],
    defaults: [
      "props.id ?? 'spatial-weights-traces'",
      'props.includeGramTraces ?? props.output.gramTraces !== undefined',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-traces'`. */",
      "and `'stochastic'` runs Hutchinson estimation with `probeCount` Rademacher probes. Defaults to",
      '/** Probes of the stochastic mode. Defaults to 64. */',
      '/** Also compute `output.gramTraces`. Defaults to whether `output.gramTraces` is present. */',
      'Rows reduced by one invocation of the dot-product passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement(id, !includeGramTraces, 'weights chunks do not support Gram traces')",
      'validateCapacity(id, `weights.chunks[${index}].rowCount`, chunk.rowCount)',
      'validateRequirement( id, chunk.rowStart === nextRow, `weights.chunks[${index}].rowStart must equal ${nextRow}` )',
      "validateRequirement( id, nextRow === props.rowCount && nextRow === props.weights.rowCount, 'weights.chunks must cover rowCount rows' )",
      "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
      "validateCapacity(id, 'probeBatchSize', props.probeBatchSize)",
      "validateRequirement( id, includeGramTraces === (props.output.gramTraces !== undefined), 'output.gramTraces is required exactly when includeGramTraces is true' )",
      "('chunks' in weights ? weights.rowCount : weights.offsets.length - 1) !== props.rowCount",
      "`'exact'` when `rows <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS` (4096) and to",
      'batch is also limited by the 64 MiB scratch budget, so this can only shrink it. Results are',
      '/** At least `maximumPower` float32 values: `tr((W / s)^q)` at index `q - 1`, `s = infinityNorm`. */',
      'Optional, at least `maximumPower` float32 values: the standard error of the mean of the',
      "/** At least `maximumPower` float32 values: `1' (W / s)^q 1` at index `q - 1` (always exact). */"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOrdinaryLeastSquaresOperation': {
    signature: 'new GPUOrdinaryLeastSquaresOperation(props: OrdinaryLeastSquaresOperationProps)',
    supportedInputs: ['props: OrdinaryLeastSquaresOperationProps'],
    defaults: [
      "props.id ?? 'ordinary-least-squares'",
      "/** Prefix for generated node and transient IDs. Defaults to `'ordinary-least-squares'`. */",
      'Rows reduced by one invocation in the accumulation passes. Defaults to'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement( id, response.length === props.rowCount, 'response length must equal rowCount' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURestrictedRegimeRegressionOperation': {
    signature:
      'new GPURestrictedRegimeRegressionOperation(props: RestrictedRegimeRegressionOperationProps)',
    supportedInputs: ['props: RestrictedRegimeRegressionOperationProps'],
    defaults: [
      'props.robust.kernelWeights.status ?? {}',
      "props.id ?? 'restricted-regime-regression'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, Boolean(props.robust) === Boolean(props.output.robust), 'robust and output.robust must be supplied together' )",
      "validateRequirement( id, props.robust.type === 'white' || props.robust.type === 'hac', \"robust.type must be 'white' or 'hac'\" )",
      "validateRequirement( id, props.robust.smallSample === undefined || typeof props.robust.smallSample === 'boolean', 'robust.smallSample must be boolean' )"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-restricted-regime-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSeeminglyUnrelatedRegressionOperation': {
    signature:
      'new GPUSeeminglyUnrelatedRegressionOperation(props: SeeminglyUnrelatedRegressionOperationProps)',
    supportedInputs: ['props: SeeminglyUnrelatedRegressionOperationProps'],
    defaults: [
      "props.id ?? 'seemingly-unrelated-regression'",
      'Stops when max abs(new-old)/max(1,abs(old)) < tolerance (default 1e-5).',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '/** Opt into iterated FGLS. The budget includes the first GLS fit; maximum 256 rounds.'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialLagImpactsOperation': {
    signature: 'new GPUSpatialLagImpactsOperation(props: SpatialLagImpactsOperationProps)',
    supportedInputs: ['props: SpatialLagImpactsOperationProps'],
    defaults: [
      "props.id ?? 'spatial-lag-impacts'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-impacts'`. */",
      '/** Row of the first predictor (the intercept is excluded). Defaults to 1. */',
      'larger values give NaN. The series needs it below 1, which is the default and is correct for'
    ],
    limits: ['/** Series length `Q` used, at most the length of each traces view. */'],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialRegressionDiagnosticsOperation': {
    signature:
      'new GPUSpatialRegressionDiagnosticsOperation(props: SpatialRegressionDiagnosticsOperationProps)',
    supportedInputs: ['props: SpatialRegressionDiagnosticsOperationProps'],
    defaults: [
      "props.id ?? 'spatial-regression-diagnostics'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-regression-diagnostics'`. */",
      "1, which is `libpysal.weights.w_subset` followed by `transform = 'r'`. Default false: the",
      '{@link getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount} (2^25 on a default device).',
      'Rows reduced by one invocation of the accumulation pass. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, typeof value === 'string' && value.length > 0, `weights.status.${name} must name a non-empty graph value` )",
      "validateRequirement( id, getView('response').length === props.rowCount, 'response length must equal rowCount' )",
      '`rows * (predictorCount + 3)` is at most',
      '{@link getGPUSpatialRegressionDiagnosticsMaximumRowTableValueCount} (2^25 on a default device).',
      'table holds at most {@link getGPUSpatialRegressionDiagnosticsMaximumTileMomentCount} values.'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialWeightsTracesOperation': {
    signature: 'new GPUSpatialWeightsTracesOperation(props: SpatialWeightsTracesOperationProps)',
    supportedInputs: ['props: SpatialWeightsTracesOperationProps'],
    defaults: [
      "props.id ?? 'spatial-weights-traces'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-traces'`. */",
      "and `'stochastic'` runs Hutchinson estimation with `probeCount` Rademacher probes. Defaults to",
      '/** Probes of the stochastic mode. Defaults to 64. */',
      '/** Also compute `output.gramTraces`. Defaults to whether `output.gramTraces` is present. */',
      'Rows reduced by one invocation of the dot-product passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "`'exact'` when `rows <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS` (4096) and to",
      'batch is also limited by the 64 MiB scratch budget, so this can only shrink it. Results are',
      '/** At least `maximumPower` float32 values: `tr((W / s)^q)` at index `q - 1`, `s = infinityNorm`. */',
      'Optional, at least `maximumPower` float32 values: the standard error of the mean of the',
      "/** At least `maximumPower` float32 values: `1' (W / s)^q 1` at index `q - 1` (always exact). */"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMultiscaleGeographicallyWeightedRegressionOnCPU':
    {
      signature:
        'export function computeMultiscaleGeographicallyWeightedRegressionOnCPU( input: CPUMultiscaleGeographicallyWeightedRegressionInput, options: CPUMultiscaleGeographicallyWeightedRegressionOptions ): CPUMultiscaleGeographicallyWeightedRegressionResult',
      supportedInputs: [
        'input: CPUMultiscaleGeographicallyWeightedRegressionInput',
        'options: CPUMultiscaleGeographicallyWeightedRegressionOptions'
      ],
      defaults: [
        'bounds completed sweeps (default 32), as on GPU; finite limit iterates have code 3 and publish',
        'options.maximumIterations ?? 32',
        'options.bandwidthSelection.maximumSearchIterations ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_DEFAULT_MAXIMUM_SEARCH_ITERATIONS',
        "Planar (default), or `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` for",
        '/** Additive linear-predictor offset: log exposure for Poisson. Defaults to zero. */',
        '/** Defaults to fixed distance. Adaptive counts complete-case rows, including the focus. */',
        '/** Compile-time adaptive selection capacity, 1–128; defaults to 128. */',
        '/** Compile-time search budget, 1–64 (default 24). */',
        'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
      ],
      limits: [
        'bounds completed sweeps (default 32), as on GPU; finite limit iterates have code 3 and publish',
        'input.positions.length !== count * 2',
        'input.predictors.length !== count * options.predictorCount',
        'input.mask && input.mask.length !== count',
        'input.parameters.length < width + COEFFICIENT_BANDWIDTH_START',
        'input.noiseVariance && input.noiseVariance.length !== 1',
        'input.bandwidthCandidates && (!options.bandwidthCandidateCount || input.bandwidthCandidates.length !== width * options.bandwidthCandidateCount)',
        '!input.bandwidthCandidates && options.bandwidthCandidateCount !== undefined',
        'subject to device binding limits. With an explicit grid, untruncated Gaussian accepts at most',
        '(fixed or adaptive bandwidth). The omitted weight of a row is at most `exp(-gaussianCutoff^2 / 2)`',
        '3 iteration limit (finite iterate published), 4 numerical failure,'
      ],
      optionHandling: [
        'What differs from the GPU contributor:',
        'undefined is the exact unbounded Gaussian, which still visits every row. Ignored by the'
      ],
      source:
        'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation.ts'
    },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMultiscaleGeographicallyWeightedRegressionOperation':
    {
      signature:
        'new CPUMultiscaleGeographicallyWeightedRegressionOperation(props: MultiscaleGeographicallyWeightedRegressionOperationProps)',
      supportedInputs: ['props: MultiscaleGeographicallyWeightedRegressionOperationProps'],
      defaults: [
        'props.maximumIterations ?? 32',
        "props.id ?? 'multiscale-geographically-weighted-regression'",
        "Planar (default), or `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` for",
        '/** Additive linear-predictor offset: log exposure for Poisson. Defaults to zero. */',
        '/** Defaults to fixed distance. Adaptive counts complete-case rows, including the focus. */',
        '/** Compile-time adaptive selection capacity, 1–128; defaults to 128. */',
        '/** Compile-time search budget, 1–64 (default 24). */',
        'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
      ],
      limits: [
        "validateCapacity(id, 'rowCount', props.rowCount)",
        "validateRequirement( id, !props.output.conditionalStandardErrors || props.noiseVariance !== undefined, 'output.conditionalStandardErrors requires noiseVariance' )",
        "validateRequirement( id, (props.bandwidthCandidates !== undefined) === (props.bandwidthCandidateCount !== undefined), 'bandwidthCandidates and bandwidthCandidateCount must be supplied together' )",
        'response.length !== props.rowCount',
        'subject to device binding limits. With an explicit grid, untruncated Gaussian accepts at most',
        '(fixed or adaptive bandwidth). The omitted weight of a row is at most `exp(-gaussianCutoff^2 / 2)`',
        '3 iteration limit (finite iterate published), 4 numerical failure,'
      ],
      optionHandling: [
        'undefined is the exact unbounded Gaussian, which still visits every row. Ignored by the'
      ],
      source:
        'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-multiscale-geographically-weighted-regression-operation.ts'
    },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialErrorGMOnCPU': {
    signature:
      'export function computeSpatialErrorGMOnCPU( input: CPUSpatialErrorGMInput, options: CPUSpatialErrorGMOptions ): CPUSpatialErrorGMResult',
    supportedInputs: ['input: CPUSpatialErrorGMInput', 'options: CPUSpatialErrorGMOptions'],
    defaults: [
      "- Stochastic traces retain the GPU's default mode, registered Philox seed/probe order and",
      'options.estimator ?? GPU_SPATIAL_ERROR_GM_DEFAULT_ESTIMATOR',
      "options.traceMode ?? (count <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS ? 'exact' : 'stochastic')",
      'options.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-gm'`. */",
      "Estimator. `'homoskedastic'` (the default, changed from the previous Kelejian-Prucha default)",
      'see `GPUSpatialWeightsTraces`. Defaults to exact up to 4096 rows and stochastic above, where',
      '/** Hutchinson probe count of the stochastic traces. Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'input.response.length !== count',
      'input.predictors.length !== count * predictorCount',
      'input.traceParameters && input.traceParameters.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Shader-only predictor and explicit-exact trace row caps are dropped; tileRowCount is ignored.',
      'Ignored by `kelejian-prucha`.',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialErrorGMOperation': {
    signature: 'new CPUSpatialErrorGMOperation(props: SpatialErrorGMOperationProps)',
    supportedInputs: ['props: SpatialErrorGMOperationProps'],
    defaults: [
      "props.id ?? 'spatial-error-gm'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-gm'`. */",
      "Estimator. `'homoskedastic'` (the default, changed from the previous Kelejian-Prucha default)",
      'see `GPUSpatialWeightsTraces`. Defaults to exact up to 4096 rows and stochastic above, where',
      '/** Hutchinson probe count of the stochastic traces. Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
      'response.length !== props.rowCount'
    ],
    optionHandling: [
      '/** CPU whole-CSR twin, without shader predictor/trace-size caps; tileRowCount is ignored. */',
      'Ignored by `kelejian-prucha`.',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMultiscaleGeographicallyWeightedRegressionOperation':
    {
      signature:
        'new GPUMultiscaleGeographicallyWeightedRegressionOperation(props: MultiscaleGeographicallyWeightedRegressionOperationProps)',
      supportedInputs: ['props: MultiscaleGeographicallyWeightedRegressionOperationProps'],
      defaults: [
        'props.maximumIterations ?? 32',
        "props.id ?? 'multiscale-geographically-weighted-regression'",
        "Planar (default), or `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` for",
        '/** Additive linear-predictor offset: log exposure for Poisson. Defaults to zero. */',
        '/** Defaults to fixed distance. Adaptive counts complete-case rows, including the focus. */',
        '/** Compile-time adaptive selection capacity, 1–128; defaults to 128. */',
        '/** Compile-time search budget, 1–64 (default 24). */',
        'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
      ],
      limits: [
        'subject to device binding limits. With an explicit grid, untruncated Gaussian accepts at most',
        '(fixed or adaptive bandwidth). The omitted weight of a row is at most `exp(-gaussianCutoff^2 / 2)`',
        '3 iteration limit (finite iterate published), 4 numerical failure,'
      ],
      optionHandling: [
        'undefined is the exact unbounded Gaussian, which still visits every row. Ignored by the'
      ],
      source:
        'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression-operation.ts'
    },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialErrorGMOperation': {
    signature: 'new GPUSpatialErrorGMOperation(props: SpatialErrorGMOperationProps)',
    supportedInputs: ['props: SpatialErrorGMOperationProps'],
    defaults: [
      "props.id ?? 'spatial-error-gm'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-gm'`. */",
      "Estimator. `'homoskedastic'` (the default, changed from the previous Kelejian-Prucha default)",
      'see `GPUSpatialWeightsTraces`. Defaults to exact up to 4096 rows and stochastic above, where',
      '/** Hutchinson probe count of the stochastic traces. Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, getView('response').length === props.rowCount, 'response length must equal rowCount' )"
    ],
    optionHandling: [
      'Ignored by `kelejian-prucha`.',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeProbitRegressionOnCPU': {
    signature:
      'export function computeProbitRegressionOnCPU( input: CPUProbitRegressionInput, options: CPUProbitRegressionOptions ): CPUProbitRegressionResult',
    supportedInputs: ['input: CPUProbitRegressionInput', 'options: CPUProbitRegressionOptions'],
    defaults: [
      'options.maximumIterations ?? 64',
      'options.tolerance ?? 1e-5',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "instead of the GPU's 1e-3 pivot limit. Some GPU status 2 fits succeed here.",
      "Its two stages retain the GPU's 1e-3 factor-pivot limit; intermediates remain float64.",
      'input.predictors.length !== rowCount * predictorCount',
      'input.mask && input.mask.length !== rowCount',
      'spatial.weights.offsets.length !== rowCount + 1',
      'spatial.weights.weights.length !== spatial.weights.neighbors.length',
      'spatial.overflow && spatial.overflow.length !== 1',
      '!values || values.length !== 1'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUProbitRegressionOperation': {
    signature: 'new CPUProbitRegressionOperation(props: ProbitRegressionOperationProps)',
    supportedInputs: ['props: ProbitRegressionOperationProps'],
    defaults: [
      "props.id ?? 'probit-regression'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      'response.length !== props.rowCount'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialTwoStageLeastSquaresOnCPU': {
    signature:
      'export function computeSpatialTwoStageLeastSquaresOnCPU( input: CPUSpatialTwoStageLeastSquaresInput, options: CPUSpatialTwoStageLeastSquaresOptions ): CPUSpatialTwoStageLeastSquaresResult',
    supportedInputs: [
      'input: CPUSpatialTwoStageLeastSquaresInput',
      'options: CPUSpatialTwoStageLeastSquaresOptions'
    ],
    defaults: [
      'options.instrumentOrder ?? 1',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-two-stage-least-squares'`. */",
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Instrument order (spreg `w_lags`), compile-time: 1 (the default) instruments `W y` with',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'validateCPURegressionWeights(id, input.weights) !== rowCount',
      'input.predictors.length !== rowCount * predictorCount',
      'input.robust',
      'input.robust.clusters && input.robust.clusters.length !== rowCount',
      'input.robust.kernel && input.robust.kernel.offsets.length !== rowCount + 1',
      'input.robust.kernel && input.robust.kernel.weights.length !== input.robust.kernel.neighbors.length',
      'view && view.length !== 1',
      '/** Capacity for dense cluster keys, at most the row count. Empty slots do not count in CR1. */'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialTwoStageLeastSquaresOperation': {
    signature:
      'new CPUSpatialTwoStageLeastSquaresOperation(props: SpatialTwoStageLeastSquaresOperationProps)',
    supportedInputs: ['props: SpatialTwoStageLeastSquaresOperationProps'],
    defaults: [
      "props.id ?? 'spatial-two-stage-least-squares'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-two-stage-least-squares'`. */",
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Instrument order (spreg `w_lags`), compile-time: 1 (the default) instruments `W y` with',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
      'response.length !== props.rowCount',
      '/** Capacity for dense cluster keys, at most the row count. Empty slots do not count in CR1. */'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUProbitRegressionOperation': {
    signature: 'new GPUProbitRegressionOperation(props: ProbitRegressionOperationProps)',
    supportedInputs: ['props: ProbitRegressionOperationProps'],
    defaults: [
      "props.id ?? 'probit-regression'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'validateRequirement( id, !spatial?.weights.overflow || !spatial.weights.status?.overflow || spatial.weights.overflow === spatial.weights.status.overflow, `${name}.weights.overflow and ${name}.weights.status.overflow must name the same value; distinct flags cannot be merged, use the canonical status.overflow only` )'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialTwoStageLeastSquaresOperation': {
    signature:
      'new GPUSpatialTwoStageLeastSquaresOperation(props: SpatialTwoStageLeastSquaresOperationProps)',
    supportedInputs: ['props: SpatialTwoStageLeastSquaresOperationProps'],
    defaults: [
      "props.id ?? 'spatial-two-stage-least-squares'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-two-stage-least-squares'`. */",
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Instrument order (spreg `w_lags`), compile-time: 1 (the default) instruments `W y` with',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, typeof value === 'string' && value.length > 0, `robust.kernel.${name} must name a non-empty graph value` )",
      "validateRequirement( id, getView('response').length === props.rowCount, 'response length must equal rowCount' )",
      '/** Capacity for dense cluster keys, at most the row count. Empty slots do not count in CR1. */'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialErrorMaximumLikelihoodOnCPU': {
    signature:
      'export function computeSpatialErrorMaximumLikelihoodOnCPU( input: CPUSpatialErrorMaximumLikelihoodInput, options: CPUSpatialErrorMaximumLikelihoodOptions ): CPUSpatialErrorMaximumLikelihoodResult',
    supportedInputs: [
      'input: CPUSpatialErrorMaximumLikelihoodInput',
      'options: CPUSpatialErrorMaximumLikelihoodOptions'
    ],
    defaults: [
      'the default mode threshold, Philox seed/probe order, standard errors and approximation flag.',
      'options.maximumPower ?? likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER',
      "options.traceMode ?? (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS ? 'exact' : 'stochastic')",
      'options.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-maximum-likelihood'`. */",
      '/** Series length `Q` of the log-determinant and trace series, an integer in `[2, 256]`. Defaults to 128. */',
      "`'exact'` or `'stochastic'` (Hutchinson) traces. Defaults to exact up to 4096 rows and",
      'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
      "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
      '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'input.response.length !== rowCount',
      'input.predictors.length !== rowCount * predictorCount',
      'exact !== (input.eigenvalues !== undefined)',
      'input.eigenvalues && input.eigenvalues.length !== rowCount',
      'input.traceParameters && input.traceParameters.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH',
      '`GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
      '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
      'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'trace errors remain zero). Inactive capacity and out-of-range neighbor sentinels are ignored.',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialErrorMaximumLikelihoodOperation': {
    signature:
      'new CPUSpatialErrorMaximumLikelihoodOperation(props: SpatialErrorMaximumLikelihoodOperationProps)',
    supportedInputs: ['props: SpatialErrorMaximumLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'spatial-error-maximum-likelihood'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-maximum-likelihood'`. */",
      '/** Series length `Q` of the log-determinant and trace series, an integer in `[2, 256]`. Defaults to 128. */',
      "`'exact'` or `'stochastic'` (Hutchinson) traces. Defaults to exact up to 4096 rows and",
      'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
      "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
      '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
      "validateRequirement( id, (props.logDeterminant === 'exact') === (props.eigenvalues !== undefined), 'eigenvalues are required exactly when logDeterminant is exact' )",
      'response.length !== props.rowCount',
      '`GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
      '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
      'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialErrorMaximumLikelihoodOperation': {
    signature:
      'new GPUSpatialErrorMaximumLikelihoodOperation(props: SpatialErrorMaximumLikelihoodOperationProps)',
    supportedInputs: ['props: SpatialErrorMaximumLikelihoodOperationProps'],
    defaults: [
      'Main regression tiles default to max(64, ceil(rowCount / 4096)) rows: 4096 is a target.',
      "props.id ?? 'spatial-error-maximum-likelihood'",
      'props.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-error-maximum-likelihood'`. */",
      '/** Series length `Q` of the log-determinant and trace series, an integer in `[2, 256]`. Defaults to 128. */',
      "`'exact'` or `'stochastic'` (Hutchinson) traces. Defaults to exact up to 4096 rows and",
      'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
      "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
      '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'Main and nested trace layouts enforce at most 16,777,216 (4096 squared) tiles, subject to',
      'device binding limits and the independent command budget. Above 4096 tiles a group merge',
      'chunk. Traces widen toward the 4096-tile target when at most 4096 chunks are nonempty;',
      '`GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
      '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
      'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialLagMaximumLikelihoodOnCPU': {
    signature:
      'export function computeSpatialLagMaximumLikelihoodOnCPU( input: CPUSpatialLagMaximumLikelihoodInput, options: CPUSpatialLagMaximumLikelihoodOptions ): CPUSpatialLagMaximumLikelihoodResult',
    supportedInputs: [
      'input: CPUSpatialLagMaximumLikelihoodInput',
      'options: CPUSpatialLagMaximumLikelihoodOptions'
    ],
    defaults: [
      'stopping rule and Q-term inverse series for prediction and variance are retained. The default',
      'options.maximumPower ?? likelihood.GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER',
      "options.traceMode ?? (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS ? 'exact' : 'stochastic')",
      'options.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-maximum-likelihood'`. */",
      '`[2, 256]`. Defaults to 128. See the class documentation for the truncation bound.',
      "`'exact'` (unit probes, no sampling error) or `'stochastic'` (Hutchinson). Defaults to exact",
      'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
      "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
      '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'input.response.length !== rowCount',
      'input.predictors.length !== rowCount * predictorCount',
      'exact !== (input.eigenvalues !== undefined)',
      'input.eigenvalues && input.eigenvalues.length !== rowCount',
      'input.traceParameters && input.traceParameters.length < GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH',
      'error and `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
      '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
      'Optional `maximumPower` float32 values `tr((W/s)^q)` the fit used, so that a downstream',
      "/** Optional `maximumPower` float32 values `1' (W/s)^q 1`, see `powerTraces`. */",
      'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'sentinels are ignored; empty slot arrays are accepted by the pure function. Zero-link weights',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialLagMaximumLikelihoodOperation': {
    signature:
      'new CPUSpatialLagMaximumLikelihoodOperation(props: SpatialLagMaximumLikelihoodOperationProps)',
    supportedInputs: ['props: SpatialLagMaximumLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'spatial-lag-maximum-likelihood'",
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-maximum-likelihood'`. */",
      '`[2, 256]`. Defaults to 128. See the class documentation for the truncation bound.',
      "`'exact'` (unit probes, no sampling error) or `'stochastic'` (Hutchinson). Defaults to exact",
      'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
      "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
      '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'tileRowCount', props.tileRowCount)",
      "validateRequirement( id, (props.logDeterminant === 'exact') === (props.eigenvalues !== undefined), 'eigenvalues are required exactly when logDeterminant is exact' )",
      "validateRequirement( id, (props.output.powerTraces === undefined) === (props.output.powerSums === undefined) && (props.output.powerTraces === undefined) === (props.output.infinityNorm === undefined), 'output.powerTraces, output.powerSums and output.infinityNorm must be provided together' )",
      'response.length !== props.rowCount',
      'error and `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
      '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
      'Optional `maximumPower` float32 values `tr((W/s)^q)` the fit used, so that a downstream',
      "/** Optional `maximumPower` float32 values `1' (W/s)^q 1`, see `powerTraces`. */",
      'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialLagMaximumLikelihoodOperation': {
    signature:
      'new GPUSpatialLagMaximumLikelihoodOperation(props: SpatialLagMaximumLikelihoodOperationProps)',
    supportedInputs: ['props: SpatialLagMaximumLikelihoodOperationProps'],
    defaults: [
      'Main regression tiles default to max(64, ceil(rowCount / 4096)) rows: 4096 is a target.',
      "props.id ?? 'spatial-lag-maximum-likelihood'",
      'props.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER',
      "/** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-maximum-likelihood'`. */",
      '`[2, 256]`. Defaults to 128. See the class documentation for the truncation bound.',
      "`'exact'` (unit probes, no sampling error) or `'stochastic'` (Hutchinson). Defaults to exact",
      'traces oscillating. With stochastic traces the tail inherits the probe noise. Defaults to',
      "`'series'` (default) evaluates `ln|I - rho W|` and the traces from the truncated power series.",
      '/** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */',
      'Rows reduced by one invocation of the accumulation passes. Defaults to',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'Main and nested trace layouts enforce at most 16,777,216 (4096 squared) tiles, subject to',
      'device binding limits and the independent command budget. Above 4096 tiles a group merge',
      'chunk. Traces widen toward the 4096-tile target when at most 4096 chunks are nonempty;',
      'error and `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.',
      '`GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT` (1024) rows because the eigenvalues come',
      'Optional `maximumPower` float32 values `tr((W/s)^q)` the fit used, so that a downstream',
      "/** Optional `maximumPower` float32 values `1' (W/s)^q 1`, see `powerTraces`. */",
      'Optional `maximumPower` float32 values: the standard error of the mean of each `tr((W/s)^q)`'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialPanelLagMaximumLikelihoodOnCPU': {
    signature:
      'export function computeSpatialPanelLagMaximumLikelihoodOnCPU( input: CPUSpatialPanelLagMaximumLikelihoodInput, options: CPUSpatialPanelLagMaximumLikelihoodOptions ): CPUSpatialPanelLagMaximumLikelihoodResult',
    supportedInputs: [
      'input: CPUSpatialPanelLagMaximumLikelihoodInput',
      'options: CPUSpatialPanelLagMaximumLikelihoodOptions'
    ],
    defaults: [
      'options.tolerance ?? 1e-5',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'entity with absolute value at most float32(1.00001). Invalid values return status 1;',
      "validateCapacity(id, 'weights.offsets entityCount', entityCount, 2)",
      'input.eigenvalues && input.eigenvalues.length !== entityCount',
      'input.response.length !== rowCount',
      'input.predictors.length !== rowCount * predictorCount',
      'input.weights.neighbors.length !== input.weights.weights.length',
      'input.weights.distances && input.weights.distances.length !== input.weights.neighbors.length',
      'input.overflow && input.overflow.length < 1',
      'values && values.length < 1'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '/** Period-major complete panel and whole entity CSR. Unused slot capacity is ignored. */',
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialPanelLagMaximumLikelihoodOperation': {
    signature:
      'new CPUSpatialPanelLagMaximumLikelihoodOperation(props: SpatialPanelLagMaximumLikelihoodOperationProps)',
    supportedInputs: ['props: SpatialPanelLagMaximumLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'spatial-panel-lag-maximum-likelihood'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'entityCount', props.entityCount, 2)",
      'offsets.length !== props.entityCount + 1',
      'eigenvalues && eigenvalues.length !== props.entityCount'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialPanelLagMaximumLikelihoodOperation': {
    signature:
      'new GPUSpatialPanelLagMaximumLikelihoodOperation(props: SpatialPanelLagMaximumLikelihoodOperationProps)',
    supportedInputs: ['props: SpatialPanelLagMaximumLikelihoodOperationProps'],
    defaults: [
      "props.id ?? 'spatial-panel-lag-maximum-likelihood'",
      'props.weights.status ?? {}',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUChunkedSpatialWeightsValidation': {
    signature:
      'new GPUChunkedSpatialWeightsValidation(props: GPUChunkedSpatialWeightsValidationProps)',
    supportedInputs: ['props: GPUChunkedSpatialWeightsValidationProps'],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'chunked-spatial-weights-validation'`. */",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-validation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUChunkedSpatialWeightsTranspose': {
    signature:
      'new GPUChunkedSpatialWeightsTranspose(props: GPUChunkedSpatialWeightsTransposeProps<Output>)',
    supportedInputs: ['props: GPUChunkedSpatialWeightsTransposeProps<Output>'],
    defaults: [
      'getGPUSpatialWeightsTransposeParameterValues (numeric defaults use the same GPUParameterBuffer layout). Packed output preserves spatial context and',
      "props.id ?? 'chunked-spatial-weights-transpose'",
      'props.symmetryTolerance ?? 0',
      'props.columnCount ?? this.rowCount',
      "/** Prefix for generated node and transient IDs. Defaults to 'chunked-spatial-weights-transpose'. */",
      '/** Number of columns of W. Defaults to the total input row count. */',
      '/** Per-frame absolute symmetry tolerance: numeric default or one-float parameter view. Defaults to 0. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity( this.id, 'symmetryTolerance.length', tolerance.length, GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH )",
      'validateRequirement( id, !chunk.weights.status?.overflow && !chunk.weights.status?.candidateOverflow, `input chunk ${index} status loss ports must be aggregated on outer weights.status` )',
      "validateCapacity(id, 'rowCount', this.rowCount)",
      "validateCapacity(id, 'columnCount', this.columnCount, 1, MAXIMUM_SLOT_COUNT - 1)",
      "validateCapacity( id, 'summed input capacity', capacity, 1, output.asymmetricSlots ? Math.floor(MAXIMUM_SLOT_COUNT / 2) : MAXIMUM_SLOT_COUNT )",
      "validateCapacity( id, 'input chunk capacity', chunk.weights.neighbors.length, 1, MAXIMUM_SORT_SLOT_COUNT )",
      "validateRequirement( id, output.asymmetricSlots.length === 1, 'output.asymmetricSlots must hold one uint32 row' )",
      "validateRequirement( id, this.rowCount === this.columnCount, 'output.asymmetricSlots requires square weights' )",
      'validateCommandNodeBudget(id, this.estimatedCommandNodeCount)',
      "validateRequirement( id, !output.status || Object.values(output.status).every(view => !view), 'output.status is unsupported for packed transpose because input status is preserved; omit output.status or use chunked output for destination diagnostics' )",
      "validateRequirement( id, !output.distances || weights.chunks.every(chunk => Boolean(chunk.weights.distances)), 'output.distances requires distances on every input chunk; omit output.distances otherwise' )",
      "validateCapacity(id, 'output chunk count', output.chunks.length)",
      "validateRequirement( id, !view || ['count', 'requiredCount', 'overflow', 'candidateOverflow'].includes(field), `${name}.status.${field} is unsupported because transpose does not compute it; omit it and run the matching diagnostic contributor afterward` )",
      'validateRequirement(id, chunk.rowStart === nextRow, `${name}.rowStart must equal ${nextRow}`)',
      'validateCapacity(id, `${name}.rowCount`, chunk.rowCount, 1, this.columnCount - nextRow)',
      'validateRequirement( id, validateGPUSpatialWeights(id, chunk.weights, name) === chunk.rowCount, `${name}.offsets must hold rowCount + 1 entries` )',
      'validateRequirement(id, Boolean(view), `${name}.status.${field} is required`)',
      'validateRequirement( id, view!.length === 1, `${name}.status.${field} must hold one uint32 row` )',
      'validateRequirement( id, !chunk.weights.distances || weights.chunks.every(input => Boolean(input.weights.distances)), `${name}.distances requires distances on every input chunk` )',
      "validateRequirement( id, nextRow === this.columnCount, 'output chunks must cover columnCount rows' )",
      "validateRequirement( id, new Set(outputViews.filter(view => view).map(view => view!.buffer)).size === outputViews.filter(view => view).length, 'output views must use separate buffers' )",
      "validateRequirement( id, this.getPortLayout(props) === this.portLayout, 'corrected input/output ports and shapes must match construction; restore the declared views or rebuild the transpose' )"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUChunkedSpatialWeightsTransform': {
    signature:
      'new GPUChunkedSpatialWeightsTransform(props: GPUChunkedSpatialWeightsTransformProps)',
    supportedInputs: ['props: GPUChunkedSpatialWeightsTransformProps'],
    defaults: [
      "props.id ?? 'chunked-spatial-weights-transform'",
      'props.output ?? []',
      "/** Prefix for generated node IDs. Defaults to `'chunked-spatial-weights-transform'`. */",
      '/** Numeric default, adaptive mode, or a shared per-frame fixed-bandwidth parameter view. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'output && output.length !== weights.chunks.length',
      "props.operation === 'average-reciprocal' && this.chunkTransforms.some(t => t.isInPlace())"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUChunkedSpatialWeightsAlgebra': {
    signature: 'new GPUChunkedSpatialWeightsAlgebra(props: GPUChunkedSpatialWeightsAlgebraProps)',
    supportedInputs: ['props: GPUChunkedSpatialWeightsAlgebraProps'],
    defaults: [
      "props.id ?? 'chunked-spatial-weights-algebra'",
      "props.weightRule ?? 'left'",
      "/** Prefix for generated node IDs. Defaults to `'chunked-spatial-weights-algebra'`. */",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      "- `'left'` (default): the weight of `left`."
    ],
    limits: [
      'validateGPUChunkedSpatialWeights(id, operand) !== rows',
      'props.output.chunks.length !== operands[0].chunks.length',
      'chunk.rowStart !== outputChunk.rowStart || chunk.rowCount !== outputChunk.rowCount',
      'validateGPUSpatialWeightsOutput(id, outputChunk.weights) !== outputChunk.rowCount',
      'validateRequirement( id, Boolean(props.left.chunks[index].weights.distances), `output chunk ${index}.distances requires left chunk distances; omit output.distances otherwise` )',
      'validateRequirement( id, Boolean(props.right.chunks[index].weights.distances), `output chunk ${index}.distances requires right chunk distances; omit output.distances otherwise` )',
      'validateRequirement( id, Boolean(props.weights.chunks[index].weights.distances), `output chunk ${index}.distances requires input chunk distances; omit output.distances otherwise` )',
      "props.operation === 'selfWeight' && typeof props.selfWeight !== 'number'",
      'props.selfWeight.length !== rows',
      "props.operation === 'selfWeight'",
      '!(Number.isFinite(selfWeight) && selfWeight >= 0)',
      "props.operation === 'subgraph'",
      'props.mask.length !== rows',
      "'weightRule' in props && props.weightRule && !(props.weightRule in COMBINE_RULES)"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeContiguityWeightsOnCPU': {
    signature:
      'export function computeContiguityWeightsOnCPU( input: CPUContiguityWeightsInput, options: { criterion: GPUContiguityCriterion; capacity: number; snapTolerance?: number; pairCapacity?: number; } ): CPUContiguityWeightsResult',
    supportedInputs: [
      'input: CPUContiguityWeightsInput',
      'options: { criterion: GPUContiguityCriterion; capacity: number; snapTolerance?: number; pairCapacity?: number; }'
    ],
    defaults: ['options.snapTolerance ?? 0', 'options.pairCapacity ?? Infinity'],
    limits: [
      '- The GPU snap-cell range limit is dropped. Zero tolerance still compares exact input float32',
      "validateCapacity('computeContiguityWeightsOnCPU', 'capacity', capacity, 0)",
      "validateCapacity('computeContiguityWeightsOnCPU', 'pairCapacity', options.pairCapacity)",
      "validateRequirement( 'computeContiguityWeightsOnCPU', criterion === 'queen' || criterion === 'rook', \"criterion must be 'queen' or 'rook'\" )",
      "validateRequirement( 'computeContiguityWeightsOnCPU', Number.isFinite(tolerance) && tolerance >= 0, 'snapTolerance must be finite and >= 0' )",
      'positions.length % 2 !== 0',
      'requiredCount > 0xffffffff'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUContiguityWeightsOperation': {
    signature: 'new CPUContiguityWeightsOperation(props: ContiguityWeightsOperationProps)',
    supportedInputs: ['props: ContiguityWeightsOperationProps'],
    defaults: ["props.id ?? 'contiguity-weights'"],
    limits: ['featureOffsets.length !== props.featureCount + 1'],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialWeightsValidationOnCPU': {
    signature:
      'export function computeSpatialWeightsValidationOnCPU( offsets: Uint32Array, neighbors: Uint32Array, weights: Float32Array, options: SpatialWeightsValidationOptions = {} ): CPUSpatialWeightsValidationResult',
    supportedInputs: [
      'offsets: Uint32Array',
      'neighbors: Uint32Array',
      'weights: Float32Array',
      'options: SpatialWeightsValidationOptions = {}'
    ],
    defaults: [
      '/** Source-column count; neighbor rows must be below it. Defaults to the target-row count. */',
      '/** Defaults to forbid. require-unit requires one weight=1 diagonal slot per target row. */'
    ],
    limits: ['offsets.length < 2', 'weights.length !== neighbors.length'],
    optionHandling: [
      'What differs from the GPU contributor:',
      'SpatialWeightsValidationOptions applies to direct and named forms: columnCount is the source-column count (default target-row count), permitting rectangular CSR. diagonal accepts forbid (default), allow or require-unit. require-unit requires columnCount >= rowCount and one weight=1 diagonal per target row. CPUSpatialWeightsValidationResult contains flags in the target-row domain and invalidCount. Invalid CSR contents set flags; invalid option values reject with accepted values.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialWeightsValidationOperation': {
    signature:
      'new CPUSpatialWeightsValidationOperation(props: SpatialWeightsValidationOperationProps)',
    supportedInputs: ['props: SpatialWeightsValidationOperationProps'],
    defaults: [
      "props.id ?? 'spatial-weights-validation'",
      '/** Source-column count; neighbor rows must be below it. Defaults to the target-row count. */',
      '/** Defaults to forbid. require-unit requires one weight=1 diagonal slot per target row. */'
    ],
    limits: ['offsets.length !== rowCount + 1'],
    optionHandling: [
      'SpatialWeightsValidationOptions applies to direct and named forms: columnCount is the source-column count (default target-row count), permitting rectangular CSR. diagonal accepts forbid (default), allow or require-unit. require-unit requires columnCount >= rowCount and one weight=1 diagonal per target row. CPUSpatialWeightsValidationResult contains flags in the target-row domain and invalidCount. Invalid CSR contents set flags; invalid option values reject with accepted values.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUContiguityWeightsOperation': {
    signature: 'new GPUContiguityWeightsOperation(props: ContiguityWeightsOperationProps)',
    supportedInputs: ['props: ContiguityWeightsOperationProps'],
    defaults: ["props.id ?? 'contiguity-weights'"],
    limits: [
      "validateRequirement( id, Boolean(props.output.status.candidateOverflow), 'output.status.candidateOverflow is required for bounded contiguity candidates' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialWeightsValidationOperation': {
    signature:
      'new GPUSpatialWeightsValidationOperation(props: SpatialWeightsValidationOperationProps)',
    supportedInputs: ['props: SpatialWeightsValidationOperationProps'],
    defaults: [
      "props.id ?? 'spatial-weights-validation'",
      '/** Source-column count; neighbor rows must be below it. Defaults to the target-row count. */',
      '/** Defaults to forbid. require-unit requires one weight=1 diagonal slot per target row. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', rowCount, 1, 0xfffffffe)",
      "validateRequirement( id, offsets.length === rowCount + 1, 'offsets length must equal rowCount + 1' )"
    ],
    optionHandling: [
      'Operation form for whole CSR only. Chunked CSR is unsupported by the mixed executor.',
      'SpatialWeightsValidationOptions applies to direct and named forms: columnCount is the source-column count (default target-row count), permitting rectangular CSR. diagonal accepts forbid (default), allow or require-unit. require-unit requires columnCount >= rowCount and one weight=1 diagonal per target row. CPUSpatialWeightsValidationResult contains flags in the target-row domain and invalidCount. Invalid CSR contents set flags; invalid option values reject with accepted values.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialWeightsAlgebraOnCPU': {
    signature:
      'export function computeSpatialWeightsAlgebraOnCPU( input: CPUSpatialWeightsAlgebraInput, options: CPUSpatialWeightsAlgebraOptions ): CPUSpatialWeightsAlgebraResult',
    supportedInputs: [
      'input: CPUSpatialWeightsAlgebraInput',
      'options: CPUSpatialWeightsAlgebraOptions'
    ],
    defaults: [
      'options.cumulative ?? false',
      "options.weightRule ?? 'left'",
      'options.selfWeight ?? input.selfWeights![row]',
      "- `'left'` (default): the weight of `left`.",
      '/** Temporal weight `T[t, t - lag]`, positive and finite. Defaults to 1. */'
    ],
    limits: [
      '- The GPU limit of 16 temporal lags is dropped. Index and output sizes remain uint32.',
      'right && right.offsets.length !== left!.offsets.length',
      "options.operation === 'block' && !input.groupIds",
      "options.operation === 'subgraph' && input.mask?.length !== baseRowCount",
      "options.operation === 'selfWeight' && options.selfWeight === undefined && input.selfWeights?.length !== baseRowCount",
      'options.distances',
      '!(left ?? source)?.distances',
      "(options.operation === 'union' || options.operation === 'symmetric-difference') && !right?.distances",
      "validateCapacity('spatial-weights-algebra', 'rowCount', rowCount)"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialWeightsAlgebraOperation': {
    signature: 'new CPUSpatialWeightsAlgebraOperation(props: SpatialWeightsAlgebraOperationProps)',
    supportedInputs: ['props: SpatialWeightsAlgebraOperationProps'],
    defaults: [
      "props.id ?? 'spatial-weights-algebra'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      "- `'left'` (default): the weight of `left`.",
      '/** Temporal weight `T[t, t - lag]`, positive and finite. Defaults to 1. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement( id, props.left.distances !== undefined, 'output.distances requires left.distances' )",
      "validateRequirement( id, props.right.distances !== undefined, 'output.distances requires right.distances' )",
      "validateRequirement( id, props.weights.distances !== undefined, 'output.distances requires weights.distances' )",
      "validateRequirement( id, props.rowCount % props.periodCount === 0, 'rowCount must be a multiple of periodCount' )",
      'result.offsets.length !== props.rowCount + 1'
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialWeightsAlgebraOperation': {
    signature: 'new GPUSpatialWeightsAlgebraOperation(props: SpatialWeightsAlgebraOperationProps)',
    supportedInputs: ['props: SpatialWeightsAlgebraOperationProps'],
    defaults: [
      "props.id ?? 'spatial-weights-algebra'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      "- `'left'` (default): the weight of `left`.",
      '/** Temporal weight `T[t, t - lag]`, positive and finite. Defaults to 1. */'
    ],
    limits: [
      "validateRequirement( props.id ?? 'spatial-weights-algebra', Boolean(props.output.status.candidateOverflow) || !( (props.operation === 'higherOrder' && props.order > 1) || operands.some(operand => operand.status?.overflow || operand.status?.candidateOverflow) ), 'output.status.candidateOverflow is required for operand loss or bounded intermediate work' )"
    ],
    optionHandling: [
      'outputs must select the fields they actually write and reject unsupported requests before'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:preparePolygonContiguityTopology': {
    signature:
      'export function preparePolygonContiguityTopology( source: PreparedPolygonTopologySource ): PreparedPolygonContiguityTopology',
    supportedInputs: ['source: PreparedPolygonTopologySource'],
    defaults: 'unknown',
    limits: [
      "validateCapacity(id, 'source row capacity', rowCount, 0, POLYGON_CONTIGUITY_MAXIMUM_ROWS)",
      "validateMaximumCount( id, 'source batch capacity', source.batches.length, POLYGON_CONTIGUITY_MAXIMUM_ROWS )",
      '!Number.isSafeInteger(batch.rowCount) || batch.rowCount < 0 || batch.sourceRowOffset !== expectedOffset || batch.polygons.length !== batch.rowCount',
      'expectedOffset > rowCount',
      'manifestBytes > MAXIMUM_MANIFEST_BYTES',
      'expectedOffset !== rowCount',
      'source.metadata.invalidRows.length > rowCount',
      '!Number.isSafeInteger(diagnostic.sourceRowIndex) || diagnostic.sourceRowIndex < 0 || diagnostic.sourceRowIndex >= rowCount',
      "validateMaximumCount( id, 'Float32 equality-token vertex capacity', POLYGON_CONTIGUITY_MAXIMUM_VERTICES, 2 ** 24 )"
    ],
    optionHandling: [
      'antimeridian normalization or geometric repair is performed. Z is ignored. Holes and polygon'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePolygonContiguityOnCPU': {
    signature:
      'export function computePolygonContiguityOnCPU( topology: PreparedPolygonContiguityTopology, options: {criterion: GPUContiguityCriterion; capacity: number} ): CPUContiguityWeightsResult',
    supportedInputs: [
      'topology: PreparedPolygonContiguityTopology',
      'options: {criterion: GPUContiguityCriterion; capacity: number}'
    ],
    defaults: 'unknown',
    limits: [
      "validateRequirement( 'computePolygonContiguityOnCPU', topology.topologyEncoding === 'source-xy-key-ids' && topology.equality === 'exact-source-xy', 'topology must use exact-source-xy equality tokens' )",
      "validateCapacity('computePolygonContiguityOnCPU', 'capacity', options.capacity, 0)",
      "validateRequirement( 'computePolygonContiguityOnCPU', options.criterion === 'queen' || options.criterion === 'rook', \"criterion must be 'queen' or 'rook'\" )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrackSimilarityOnCPU': {
    signature:
      'export function computeTrackSimilarityOnCPU( input: CPUTrackSimilarityInput, options: CPUTrackSimilarityOptions = {} ): CPUTrackSimilarityResult',
    supportedInputs: ['input: CPUTrackSimilarityInput', 'options: CPUTrackSimilarityOptions = {}'],
    defaults: [
      "options.measures ?? Object.keys(result).filter( port => port !== 'pairStatus' && (port !== 'frechetWithin' || input.frechetEpsilon !== undefined) )",
      '/** Defaults to every measure (frechetWithin only when frechetEpsilon is supplied). */',
      "/** Prefix for generated node IDs. Defaults to `'track-similarity'`. */",
      'Coordinate space of both vertex sets. Native planar (the default): distances are Euclidean in',
      '`sphereRadius` (default',
      '/** Second set of tracks. Defaults to the first set (similarity inside one set). */',
      '1 to 2048 (workgroup memory holds one row of it). Defaults to 256. The longer track is',
      "units. Defaults to 1. A metric ball, like `lcssTolerance` (Chen's original uses a per-axis box).",
      '`[longitude, latitude]` degrees). Defaults to the origin. Chen and Ng use a fixed point such as',
      '/** Compile-time nonnegative distance tolerance for LCSS, in spatial-context units. Defaults to 1. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateRequirement( 'computeTrackSimilarityOnCPU', input.pairB.length === input.pairA.length, `pairB.length must equal pairA.length ${input.pairA.length}; received ${input.pairB.length}` )",
      'input.activePairCount?.length === 0',
      'input.frechetEpsilon?.length === 0',
      "!(measure in result) || measure === 'pairStatus'",
      "measures.has('frechetWithin') && !input.frechetEpsilon",
      "validateCapacity( 'computeTrackSimilarityOnCPU', 'densifiedVertexCount', (track.length - 1) * subdivisions + 1, 1, MAXIMUM_DENSIFIED_VERTEX_COUNT )",
      'Track index into the first set for each pair. The length is the pair capacity, at most',
      'unlimited: it is swept in strips of 256 lanes. Pairs whose shorter track exceeds the cap report',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional',
      'Optional per-pair Frechet decision: 1 when the discrete Frechet distance is at most',
      'into the other, where two vertices match, at cost 0, when their distance is at most',
      'capped; the sums are fixed-order. NaN when not computed.'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrackSimilarityOperation': {
    signature: 'new CPUTrackSimilarityOperation(props: TrackSimilarityOperationProps)',
    supportedInputs: ['props: TrackSimilarityOperationProps'],
    defaults: [
      "props.id ?? 'track-similarity'",
      "/** Prefix for generated node IDs. Defaults to `'track-similarity'`. */",
      'Coordinate space of both vertex sets. Native planar (the default): distances are Euclidean in',
      '`sphereRadius` (default',
      '/** Second set of tracks. Defaults to the first set (similarity inside one set). */',
      '1 to 2048 (workgroup memory holds one row of it). Defaults to 256. The longer track is',
      "units. Defaults to 1. A metric ball, like `lcssTolerance` (Chen's original uses a per-axis box).",
      '`[longitude, latitude]` degrees). Defaults to the origin. Chen and Ng use a fixed point such as',
      '/** Compile-time nonnegative distance tolerance for LCSS, in spatial-context units. Defaults to 1. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'pairCount', props.pairCount, 0)",
      "validateRequirement( id, Object.entries(props.output).some( ([port, value]) => port !== 'pairStatus' && value !== undefined ), 'output must request at least one measure' )",
      "validateRequirement( id, (props.output.frechetWithin !== undefined) === (props.frechetEpsilon !== undefined), 'output.frechetWithin and frechetEpsilon must be given together' )",
      'pairA.length !== props.pairCount',
      'Track index into the first set for each pair. The length is the pair capacity, at most',
      'unlimited: it is swept in strips of 256 lanes. Pairs whose shorter track exceeds the cap report',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional',
      'Optional per-pair Frechet decision: 1 when the discrete Frechet distance is at most',
      'into the other, where two vertices match, at cost 0, when their distance is at most',
      'capped; the sums are fixed-order. NaN when not computed.'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/track-similarity/cpu-track-similarity-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrackSimilarityOperation': {
    signature: 'new GPUTrackSimilarityOperation(props: TrackSimilarityOperationProps)',
    supportedInputs: ['props: TrackSimilarityOperationProps'],
    defaults: [
      "props.id ?? 'track-similarity'",
      "/** Prefix for generated node IDs. Defaults to `'track-similarity'`. */",
      'Coordinate space of both vertex sets. Native planar (the default): distances are Euclidean in',
      '`sphereRadius` (default',
      '/** Second set of tracks. Defaults to the first set (similarity inside one set). */',
      '1 to 2048 (workgroup memory holds one row of it). Defaults to 256. The longer track is',
      "units. Defaults to 1. A metric ball, like `lcssTolerance` (Chen's original uses a per-axis box).",
      '`[longitude, latitude]` degrees). Defaults to the origin. Chen and Ng use a fixed point such as',
      '/** Compile-time nonnegative distance tolerance for LCSS, in spatial-context units. Defaults to 1. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'runs when the graph resolves its views. CPUTrackSimilarityOperation is the uncapped CPU twin.',
      'Track index into the first set for each pair. The length is the pair capacity, at most',
      'unlimited: it is swept in strips of 256 lanes. Pairs whose shorter track exceeds the cap report',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional',
      'Optional per-pair Frechet decision: 1 when the discrete Frechet distance is at most',
      'into the other, where two vertices match, at cost 0, when their distance is at most',
      'capped; the sums are fixed-order. NaN when not computed.'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectorySplitOnCPU': {
    signature:
      'export function computeTrajectorySplitOnCPU( input: CPUTrajectorySplitInput, options: CPUTrajectorySplitOptions ): CPUTrajectorySplitResult',
    supportedInputs: ['input: CPUTrajectorySplitInput', 'options: CPUTrajectorySplitOptions'],
    defaults: [
      '/** Explicit resident-capacity storage retains GPU offset clamping. Defaults to compact. */'
    ],
    limits: [
      "mode === 'gap' && (!input.parameters || input.parameters.length < GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH)",
      'input.staypointIds && input.staypointIds.length !== rowCount',
      'input.values && input.values.length !== rowCount',
      "mode === 'stop' && !input.staypointIds",
      "mode === 'value-change' && !input.values"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectorySplitOperation': {
    signature: 'new CPUTrajectorySplitOperation(props: TrajectorySplitOperationProps)',
    supportedInputs: ['props: TrajectorySplitOperationProps'],
    defaults: [
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-split'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateRequirement( id, props.timestampFormat === undefined || ['float32', 'uint32x2'].includes(props.timestampFormat), 'timestampFormat must be float32 or uint32x2' )",
      "validateRequirement( id, props.mode !== 'gap' || props.parameters !== undefined, \"mode 'gap' requires parameters\" )",
      "validateRequirement( id, props.mode !== 'stop' || props.staypointIds !== undefined, \"mode 'stop' requires staypointIds\" )",
      "validateRequirement( id, props.mode !== 'value-change' || props.values !== undefined, \"mode 'value-change' requires values\" )",
      'positions.length !== props.rowCount * 2',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectoryThinningOnCPU': {
    signature:
      'export function computeTrajectoryThinningOnCPU( input: CPUTrajectoryThinningInput, options: CPUTrajectoryThinningOptions ): CPUTrajectoryThinningResult',
    supportedInputs: ['input: CPUTrajectoryThinningInput', 'options: CPUTrajectoryThinningOptions'],
    defaults: [
      'options.maximumSearchRows ?? Number.MAX_SAFE_INTEGER',
      '/** Explicit resident-capacity storage retains GPU offset clamping. Defaults to compact. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- Omitted maximumSearchRows leaves CPU searches uncapped; explicit caps split the chain and',
      "validateCapacity(id, 'capacity', options.capacity)",
      "validateCapacity(id, 'maximumSearchRows', maximumSearchRows, 1, Number.MAX_SAFE_INTEGER)",
      "validateRequirement( id, ['minimum-distance', 'minimum-time'].includes(options.mode), \"mode must be 'minimum-distance' or 'minimum-time'\" )",
      'input.parameters.length < GPU_TRAJECTORY_THINNING_PARAMETER_LENGTH'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectoryThinningOperation': {
    signature: 'new CPUTrajectoryThinningOperation(props: TrajectoryThinningOperationProps)',
    supportedInputs: ['props: TrajectoryThinningOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-thinning'",
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-thinning'`. */",
      '/** Forward-search cap per row (default 4096, at most 16,777,216). */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'output.status.candidateOverflow is required (one uint32 row)',
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'trackCount', props.trackCount, 0)",
      "validateCapacity(id, 'capacity', props.capacity)",
      "validateCapacity( id, 'maximumSearchRows', props.maximumSearchRows, 1, Number.MAX_SAFE_INTEGER )",
      "validateRequirement( id, ['minimum-distance', 'minimum-time'].includes(props.mode), \"mode must be 'minimum-distance' or 'minimum-time'\" )",
      "validateRequirement( id, props.timestampFormat === undefined || ['float32', 'uint32x2'].includes(props.timestampFormat), 'timestampFormat must be float32 or uint32x2' )",
      'positions.length !== props.rowCount * 2',
      'trackOffsets.length !== props.trackCount + 1',
      '/** Same graph values for the GPU contributor and uncapped CPU sequential walk. */',
      '/** Forward-search cap per row (default 4096, at most 16,777,216). */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectorySplitOperation': {
    signature: 'new GPUTrajectorySplitOperation(props: TrajectorySplitOperationProps)',
    supportedInputs: ['props: TrajectorySplitOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-split'",
      "props.timestampFormat ?? 'float32'",
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-split'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryThinningOperation': {
    signature: 'new GPUTrajectoryThinningOperation(props: TrajectoryThinningOperationProps)',
    supportedInputs: ['props: TrajectoryThinningOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-thinning'",
      "props.timestampFormat ?? 'float32'",
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-thinning'`. */",
      '/** Forward-search cap per row (default 4096, at most 16,777,216). */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'output.status.candidateOverflow is required (one uint32 row)',
      '/** Same graph values for the GPU contributor and uncapped CPU sequential walk. */',
      '/** Forward-search cap per row (default 4096, at most 16,777,216). */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeBrownianBridgeUtilizationOnCPU': {
    signature:
      'export function computeBrownianBridgeUtilizationOnCPU( input: CPUBrownianBridgeUtilizationInput, options: { gridSize: readonly [number, number]; subStepCount?: number; tailPolicy?: BrownianBridgeTailPolicy; } ): CPUBrownianBridgeUtilizationResult',
    supportedInputs: [
      'input: CPUBrownianBridgeUtilizationInput',
      'options: { gridSize: readonly [number, number]; subStepCount?: number; tailPolicy?: BrownianBridgeTailPolicy; }'
    ],
    defaults: [
      '- Omitted tailPolicy keeps the full Gaussian tail (GPU default: truncate-at-seven-sigma).',
      'options.subStepCount ?? GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT',
      'input.storage defaults to resident-capacity.'
    ],
    limits: [
      'motionVariances.length !== rowCount',
      'locationErrors.length !== rowCount',
      'parameters.length < GPU_BROWNIAN_BRIDGE_UTILIZATION_PARAMETER_LENGTH',
      "validateCapacity('brownian-bridge-utilization', 'gridSize columns', columns)",
      "validateCapacity('brownian-bridge-utilization', 'gridSize rows', lines)",
      "validateCapacity('brownian-bridge-utilization', 'subStepCount', subStepCount)"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'tailPolicy accepts full or truncate-at-seven-sigma; CPU default full, GPU default truncate-at-seven-sigma. All forms accept zero or one fix and emit a zero raster of the requested gridSize. The midpoint subStepCount controls integration independently of tailPolicy. CPU input.storage defaults to resident-capacity; compact validates exact offset coverage.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUBrownianBridgeUtilizationOperation': {
    signature:
      'new CPUBrownianBridgeUtilizationOperation(props: BrownianBridgeUtilizationOperationProps)',
    supportedInputs: ['props: BrownianBridgeUtilizationOperationProps'],
    defaults: [
      'props.subStepCount ?? GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT',
      "/** Prefix for generated node and transient IDs. Defaults to `'brownian-bridge-utilization'`. */",
      '/** Midpoint-rule steps per segment (1 to 256, default 15). Compile-time. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'gridSize columns', props.gridSize[0])",
      "validateCapacity(id, 'gridSize rows', props.gridSize[1])",
      "validateCapacity( id, 'subStepCount', props.subStepCount ?? GPU_BROWNIAN_BRIDGE_DEFAULT_SUB_STEP_COUNT )",
      "validateRequirement( id, props.timestampFormat === undefined || props.timestampFormat === 'float32' || props.timestampFormat === 'uint32x2', 'timestampFormat must be float32 or uint32x2' )",
      'positions.length !== props.rowCount * 2',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'tailPolicy accepts full or truncate-at-seven-sigma; CPU default full, GPU default truncate-at-seven-sigma. All forms accept zero or one fix and emit a zero raster of the requested gridSize. The midpoint subStepCount controls integration independently of tailPolicy. Named forms expose no storage selector; the CPU named form constructs its own resident-capacity input.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUBrownianBridgeUtilizationOperation': {
    signature:
      'new GPUBrownianBridgeUtilizationOperation(props: BrownianBridgeUtilizationOperationProps)',
    supportedInputs: ['props: BrownianBridgeUtilizationOperationProps'],
    defaults: [
      "props.id ?? 'brownian-bridge-utilization'",
      "/** Prefix for generated node and transient IDs. Defaults to `'brownian-bridge-utilization'`. */",
      '/** Midpoint-rule steps per segment (1 to 256, default 15). Compile-time. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, props.timestampFormat === undefined || props.timestampFormat === 'float32' || props.timestampFormat === 'uint32x2', `timestampFormat must be float32 or uint32x2; received ${props.timestampFormat}` )",
      'validateRequirement( id, positions.length === props.rowCount, `positions length must equal declared rowCount ${props.rowCount}; received ${positions.length}` )',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'tailPolicy accepts full or truncate-at-seven-sigma; CPU default full, GPU default truncate-at-seven-sigma. All forms accept zero or one fix and emit a zero raster of the requested gridSize. The midpoint subStepCount controls integration independently of tailPolicy. Named forms expose no storage selector; the CPU named form constructs its own resident-capacity input.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMobilityIndicatorsOnCPU': {
    signature:
      'export function computeMobilityIndicatorsOnCPU( input: CPUMobilityIndicatorsInput, options: CPUMobilityIndicatorsOptions ): CPUMobilityIndicatorsResult',
    supportedInputs: ['input: CPUMobilityIndicatorsInput', 'options: CPUMobilityIndicatorsOptions'],
    defaults: [
      'passes the cap only when its props set one, so the GPU default cap of 1024 is not applied.',
      '/** Real entropy can be skipped when unused; the default computes every indicator. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      'gyrationCenter defaults to spherical-mean for great-circle contexts and coordinate-mean for planar contexts.'
    ],
    limits: [
      "validateCapacity(id, 'trackCount', trackCount)",
      "validateCapacity(id, 'maximumEntropySequenceRows', maximumEntropySequenceRows)",
      '!Number.isInteger(rowCount)',
      'staypoints.centroids.length !== capacity * 2',
      'staypoints[column].length !== capacity',
      'staypoints.count.length < 1',
      'triplegs',
      'triplegs.startRows.length !== triplegs.tracks.length || triplegs.endRows.length !== triplegs.tracks.length || triplegs.staypointFinishRows.length !== capacity',
      'triplegs.count.length < 1',
      'flag && flag.length < 1',
      'staypoints.locationIterationLimitReached && staypoints.locationIterationLimitReached.length < 1',
      'table.count.length < 1',
      'table.tracks[row] >= trackCount || (row > 0 && table.tracks[row] < table.tracks[row - 1])',
      'staypoints.startRows[stay] >= rowCount',
      'triplegs.startRows.length !== triplegCapacity',
      'triplegs.endRows.length !== triplegCapacity',
      'triplegs.staypointFinishRows.length !== capacity',
      'triplegs.staypointFinishRows[stay] >= rowCount',
      'triplegs.startRows[tripleg] >= rowCount',
      'triplegs.endRows[tripleg] === 0 || triplegs.endRows[tripleg] > rowCount || triplegs.endRows[tripleg] <= triplegs.startRows[tripleg]',
      "validateRequirement( id, gyrationCenter !== 'spherical-mean' || radius !== undefined, 'gyrationCenter spherical-mean requires a great-circle spatialContext; use coordinate-mean for planar positions' )",
      'stayFlag && stayFlag.length < 1',
      "validateRequirement( id, !locationLoss || locationLoss.length >= 1, 'staypoints.locationIterationLimitReached must contain at least one row' )"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'gyrationCenter accepts coordinate-mean or spherical-mean; spherical-mean requires a great-circle spatialContext. Indeterminate spherical mean (mean unit-vector norm <= 1e-6) produces NaN radiusOfGyration.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMobilityIndicatorsOperation': {
    signature: 'new CPUMobilityIndicatorsOperation(props: MobilityIndicatorsOperationProps)',
    supportedInputs: ['props: MobilityIndicatorsOperationProps'],
    defaults: [
      "props.id ?? 'mobility-indicators'",
      "/** Prefix for generated node and transient IDs. Defaults to `'mobility-indicators'`. */",
      '/** Cap on usable location keys per track for `realEntropy` (default 1024, at most 16384). */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      'gyrationCenter defaults to spherical-mean for great-circle contexts and coordinate-mean for planar contexts.'
    ],
    limits: [
      "validateCapacity(id, 'trackCount', props.trackCount)",
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'staypointCapacity', props.staypointCapacity, 0)",
      "validateCapacity(id, 'maximumEntropySequenceRows', props.maximumEntropySequenceRows)",
      "validateCapacity(id, 'triplegCapacity', props.triplegCapacity!, 0)",
      "validateRequirement( id, Object.entries(props.output).some( ([port, value]) => port !== 'status' && value !== undefined ), 'requires at least one output' )",
      "validateRequirement( id, props.output.trackingQuality === undefined || Boolean(props.triplegs), 'trackingQuality requires triplegs' )",
      "validateRequirement( id, props.timestampFormat === undefined || ['float32', 'uint32x2'].includes(props.timestampFormat), 'timestampFormat must be float32 or uint32x2' )",
      "timestamps.length !== props.rowCount * (props.timestampFormat === 'uint32x2' ? 2 : 1)",
      'tracks.length !== props.staypointCapacity',
      'triplegTracks && triplegTracks.length !== props.triplegCapacity',
      '/** Cap on usable location keys per track for `realEntropy` (default 1024, at most 16384). */'
    ],
    optionHandling: [
      'gyrationCenter accepts coordinate-mean or spherical-mean; spherical-mean requires a great-circle spatialContext. Indeterminate spherical mean (mean unit-vector norm <= 1e-6) produces NaN radiusOfGyration.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeModalSplitOnCPU': {
    signature:
      'export function computeModalSplitOnCPU( input: CPUModalSplitInput, options: CPUModalSplitOptions ): CPUModalSplitResult',
    supportedInputs: ['input: CPUModalSplitInput', 'options: CPUModalSplitOptions'],
    defaults: [
      '/** Explicit resident-capacity storage retains GPU offset clamping. Defaults to compact. */',
      '/** Unknown supplied IDs enter the last bin by default; the missing key is always excluded. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'flag && flag.length < 1',
      'triplegs.startRows.length !== capacity',
      'triplegs.endRows.length !== capacity',
      'triplegs.count.length < 1',
      'input.modes && input.modes.length !== capacity',
      '!speedBounds && !input.modes',
      'track >= trackCount || (tripleg > 0 && track < triplegs.tracks[tripleg - 1])',
      'start < trackOffsets[track] || start > rowCount || start > end',
      'end > trackOffsets[track + 1]',
      'needsDuration && start === end'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUModalSplitOperation': {
    signature: 'new CPUModalSplitOperation(props: ModalSplitOperationProps)',
    supportedInputs: ['props: ModalSplitOperationProps'],
    defaults: [
      "props.id ?? 'modal-split'",
      "/** Prefix for node and transient IDs. Defaults to `'modal-split'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      '/** Unknown supplied IDs enter the last bin by default; the missing key is always excluded. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'trackCount', props.trackCount, 0)",
      "validateCapacity(id, 'triplegCapacity', props.triplegCapacity, 0)",
      "validateCapacity(id, 'trackCount * modeCount', props.trackCount * props.modeCount, 0)",
      "validateRequirement( id, Boolean(props.modeSpeedUpperBoundsFloat32) || props.modes !== undefined, 'modes or modeSpeedUpperBoundsFloat32 is required' )",
      "validateRequirement( id, props.timestampFormat === undefined || ['float32', 'uint32x2'].includes(props.timestampFormat), 'timestampFormat must be float32 or uint32x2' )",
      'positions.length !== props.rowCount * 2',
      'trackOffsets.length !== props.trackCount + 1',
      'tracks.length !== props.triplegCapacity',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/cpu-modal-split-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMobilityIndicatorsOperation': {
    signature: 'new GPUMobilityIndicatorsOperation(props: MobilityIndicatorsOperationProps)',
    supportedInputs: ['props: MobilityIndicatorsOperationProps'],
    defaults: [
      "props.id ?? 'mobility-indicators'",
      "/** Prefix for generated node and transient IDs. Defaults to `'mobility-indicators'`. */",
      '/** Cap on usable location keys per track for `realEntropy` (default 1024, at most 16384). */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      'gyrationCenter defaults to spherical-mean for great-circle contexts and coordinate-mean for planar contexts.'
    ],
    limits: [
      "validateRequirement(id, Object.keys(outputs).length > 0, 'requires at least one output')",
      '/** Cap on usable location keys per track for `realEntropy` (default 1024, at most 16384). */'
    ],
    optionHandling: [
      'gyrationCenter accepts coordinate-mean or spherical-mean; spherical-mean requires a great-circle spatialContext. Indeterminate spherical mean (mean unit-vector norm <= 1e-6) produces NaN radiusOfGyration.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/gpu-mobility-indicators-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUModalSplitOperation': {
    signature: 'new GPUModalSplitOperation(props: ModalSplitOperationProps)',
    supportedInputs: ['props: ModalSplitOperationProps'],
    defaults: [
      "props.id ?? 'modal-split'",
      "/** Prefix for node and transient IDs. Defaults to `'modal-split'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      '/** Unknown supplied IDs enter the last bin by default; the missing key is always excluded. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-mobility/gpu-modal-split-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectoryRecursionsOnCPU': {
    signature:
      'export function computeTrajectoryRecursionsOnCPU( input: CPUTrajectoryRows & {radii: Float32Array}, options: {maximumTrackSegments?: number} = {} ): CPUTrajectoryRecursionsResult',
    supportedInputs: [
      'input: CPUTrajectoryRows & {radii: Float32Array}',
      'options: {maximumTrackSegments?: number} = {}'
    ],
    defaults: [
      'options.maximumTrackSegments ?? Number.MAX_SAFE_INTEGER',
      '/** Explicit resident-capacity storage retains GPU offset clamping. Defaults to compact. */'
    ],
    limits: [
      "validateCapacity( 'trajectory-recursions', 'maximumTrackSegments', maximumTrackSegments, 1, Number.MAX_SAFE_INTEGER )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectoryRecursionsOperation': {
    signature: 'new CPUTrajectoryRecursionsOperation(props: TrajectoryRecursionsOperationProps)',
    supportedInputs: ['props: TrajectoryRecursionsOperationProps'],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'trajectory-recursions'`. */",
      'Segments examined per row (default 4096, at most 65,536). A longer track is examined in a',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', rowCount, 0)",
      "validateCapacity(id, 'radiusCount', radiusCount)",
      "validateCapacity(id, 'output rows', rowCount * radiusCount, 0)",
      "validateCapacity( id, 'maximumTrackSegments', maximumTrackSegments, 1, Number.MAX_SAFE_INTEGER )",
      "validateRequirement( id, Boolean( props.output.revisits || props.output.residenceTimes || props.output.firstPassageTimes ), 'output must request revisits, residenceTimes or firstPassageTimes' )",
      "validateRequirement( id, !props.output.firstPassageCensored || Boolean(props.output.firstPassageTimes), 'output.firstPassageCensored requires output.firstPassageTimes' )",
      "validateRequirement( id, timestampFormat === undefined || timestampFormat === 'float32' || timestampFormat === 'uint32x2', 'timestampFormat must be float32 or uint32x2' )",
      'positions.length !== rowCount * 2',
      'radii.length !== radiusCount',
      'at most {@link GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT}). A radius that is not positive',
      'Segments examined per row (default 4096, at most 65,536). A longer track is examined in a',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-recursion/cpu-trajectory-recursions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryRecursionsOperation': {
    signature: 'new GPUTrajectoryRecursionsOperation(props: TrajectoryRecursionsOperationProps)',
    supportedInputs: ['props: TrajectoryRecursionsOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-recursions'",
      "/** Prefix for generated node and transient IDs. Defaults to `'trajectory-recursions'`. */",
      'Segments examined per row (default 4096, at most 65,536). A longer track is examined in a',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'at most {@link GPU_TRAJECTORY_RECURSIONS_MAXIMUM_RADIUS_COUNT}). A radius that is not positive',
      'Segments examined per row (default 4096, at most 65,536). A longer track is examined in a',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectoryStaypointsOnCPU': {
    signature:
      "export function computeTrajectoryStaypointsOnCPU(input: CPUTrajectoryStaypointsInput, options: { capacity: number; spatialContext?: GPUSpatialContext; maximumSearchRows?: number; missingFixPolicy?: 'break'; }): CPUTrajectoryStaypointsResult",
    supportedInputs: [
      'input: CPUTrajectoryStaypointsInput',
      "options: { capacity: number; spatialContext?: GPUSpatialContext; maximumSearchRows?: number; missingFixPolicy?: 'break'; }"
    ],
    defaults: [
      'options.maximumSearchRows ?? Number.MAX_SAFE_INTEGER',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- Omitted maximumSearchRows leaves CPU searches uncapped; explicit caps split windows,',
      "validateRequirement('trajectory-staypoints', options.missingFixPolicy === undefined || options.missingFixPolicy === 'break', 'missingFixPolicy must be break')",
      "validateRequirement('trajectory-staypoints', !mask || mask.length === positions.length / 2, 'mask must match positions row count')",
      'parameters.length < GPU_TRAJECTORY_STAYPOINTS_PARAMETER_LENGTH',
      "validateCapacity('trajectory-staypoints', 'capacity', options.capacity)",
      "validateCapacity( 'trajectory-staypoints', 'maximumSearchRows', maximumSearchRows, 1, Number.MAX_SAFE_INTEGER )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectoryStaypointsOperation': {
    signature: 'new CPUTrajectoryStaypointsOperation(props: TrajectoryStaypointsOperationProps)',
    supportedInputs: ['props: TrajectoryStaypointsOperationProps'],
    defaults: [
      '/** Encoding of the timestamp graph value; defaults to float32. */',
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-staypoints'`. */",
      '/** Planar (default) or longitude/latitude great-circle (trackintel `haversine`, set `sphereRadius` 6371000 to match). */',
      '/** Forward-search cap per anchor in rows (default 4096, at most 16,777,216). */',
      '/** Missing fixes end dwell at the previous observed timestamp. Defaults to break. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'output.status.candidateOverflow is required (one uint32 row)',
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'capacity', props.capacity)",
      "validateCapacity( id, 'maximumSearchRows', props.maximumSearchRows, 1, Number.MAX_SAFE_INTEGER )",
      "validateRequirement( id, props.timestampFormat === undefined || props.timestampFormat === 'float32' || props.timestampFormat === 'uint32x2', 'timestampFormat must be float32 or uint32x2' )",
      'positions.length !== props.rowCount * 2',
      '/** Forward-search cap per anchor in rows (default 4096, at most 16,777,216). */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryStaypointsOperation': {
    signature: 'new GPUTrajectoryStaypointsOperation(props: TrajectoryStaypointsOperationProps)',
    supportedInputs: ['props: TrajectoryStaypointsOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-staypoints'",
      '/** Encoding of the timestamp graph value; defaults to float32. */',
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-staypoints'`. */",
      '/** Planar (default) or longitude/latitude great-circle (trackintel `haversine`, set `sphereRadius` 6371000 to match). */',
      '/** Forward-search cap per anchor in rows (default 4096, at most 16,777,216). */',
      '/** Missing fixes end dwell at the previous observed timestamp. Defaults to break. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'output.status.candidateOverflow is required (one uint32 row)',
      '/** Forward-search cap per anchor in rows (default 4096, at most 16,777,216). */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectoryLocationsOnCPU': {
    signature:
      'export function computeTrajectoryLocationsOnCPU( input: CPUTrajectoryLocationsInput, options: {spatialContext?: GPUSpatialContext; maximumIterations?: number} = {} ): CPUTrajectoryLocationsResult',
    supportedInputs: [
      'input: CPUTrajectoryLocationsInput',
      'options: {spatialContext?: GPUSpatialContext; maximumIterations?: number} = {}'
    ],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '!Number.isInteger(capacity)',
      "validateCapacity('trajectory-locations', 'centroids capacity', capacity)",
      "validateCapacity( 'trajectory-locations', 'maximumIterations', maximumIterations, 0, GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS )",
      'tracks.length !== capacity',
      'count.length < 1',
      'parameters.length < GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH',
      'flag && flag.length < 1',
      'tracks[row] < tracks[row - 1]'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectoryLocationsOperation': {
    signature: 'new CPUTrajectoryLocationsOperation(props: TrajectoryLocationsOperationProps)',
    supportedInputs: ['props: TrajectoryLocationsOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-locations'",
      "/** Prefix for generated node and transient IDs. Defaults to `'trajectory-locations'`. */",
      '/** Hook-and-compress rounds, compile-time (default 32, at most 256). */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'output.status.iterationLimitReached is required (one uint32 row)',
      "validateCapacity(id, 'capacity', capacity)",
      "validateCapacity( id, 'maximumIterations', maximumIterations, 0, GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS )",
      'centroids.length !== capacity * 2',
      '/** Hook-and-compress rounds, compile-time (default 32, at most 256). */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectorySegmentsOnCPU': {
    signature:
      'export function computeTrajectorySegmentsOnCPU( input: CPUTrajectorySegmentsInput, options: CPUTrajectorySegmentsOptions ): CPUTrajectorySegmentsResult',
    supportedInputs: ['input: CPUTrajectorySegmentsInput', 'options: CPUTrajectorySegmentsOptions'],
    defaults: [
      'options.tripCapacity ?? 0',
      '/** Explicit resident-capacity storage retains GPU offset clamping. Defaults to compact. */'
    ],
    limits: [
      "validateCapacity(id, 'triplegCapacity', options.triplegCapacity)",
      "validateCapacity(id, 'tripCapacity', options.tripCapacity)",
      'input.staypointIds.length !== rowCount',
      'input.parameters.length < GPU_TRAJECTORY_SEGMENTS_PARAMETER_LENGTH',
      'options.tripCapacity !== undefined && !input.staypoints',
      'staypoints',
      'staypoints[field] && staypoints[field]!.length === 0',
      'staypoints[column].length !== capacity',
      'staypoints.count.length < 1',
      'track >= trackCount || (stay > 0 && track < staypoints.tracks[stay - 1])',
      'start < trackOffsets[track] || start >= trackOffsets[track + 1] || (stay > 0 && track === staypoints.tracks[stay - 1] && start < staypoints.startRows[stay - 1])',
      'finish < start || finish >= trackOffsets[track + 1]'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Nonfinite coordinates compare with JavaScript rules (NaN differs from itself); nonfinite'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectorySegmentsOperation': {
    signature: 'new CPUTrajectorySegmentsOperation(props: TrajectorySegmentsOperationProps)',
    supportedInputs: ['props: TrajectorySegmentsOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-segments'",
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-segments'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'triplegCapacity', props.triplegCapacity)",
      "validateCapacity(id, 'tripCapacity', props.tripCapacity!)",
      "validateRequirement(id, Boolean(props.staypoints), 'staypoints are required for trips')",
      "validateRequirement( id, props.timestampFormat === undefined || ['float32', 'uint32x2'].includes(props.timestampFormat), 'timestampFormat must be float32 or uint32x2' )",
      'positions.length !== props.rowCount * 2',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-segments-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryLocationsOperation': {
    signature: 'new GPUTrajectoryLocationsOperation(props: TrajectoryLocationsOperationProps)',
    supportedInputs: ['props: TrajectoryLocationsOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-locations'",
      "/** Prefix for generated node and transient IDs. Defaults to `'trajectory-locations'`. */",
      '/** Hook-and-compress rounds, compile-time (default 32, at most 256). */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'output.status.iterationLimitReached is required (one uint32 row)',
      "validateCapacity( id, 'maximumIterations', props.maximumIterations, 0, GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS )",
      '/** Hook-and-compress rounds, compile-time (default 32, at most 256). */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-locations-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectorySegmentsOperation': {
    signature: 'new GPUTrajectorySegmentsOperation(props: TrajectorySegmentsOperationProps)',
    supportedInputs: ['props: TrajectorySegmentsOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-segments'",
      "/** Prefix for node and transient IDs. Defaults to `'trajectory-segments'`. */",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-segments-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeMaximumInscribedCircleOnCPU': {
    signature:
      'export function computeMaximumInscribedCircleOnCPU( input: CPUPolygonRows, tolerance: number, options: CircleWorkOptions = {} ): CPUMaximumInscribedCircleResult',
    supportedInputs: [
      'input: CPUPolygonRows',
      'tolerance: number',
      'options: CircleWorkOptions = {}'
    ],
    defaults: [
      '/** Maximum live quadtree cells per feature, default 256. */',
      '/** Maximum cell splits per feature, default 1024. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity('maximum-inscribed-circle', 'maximumIterations', options.maximumIterations, 0)",
      "validateCapacity('maximum-inscribed-circle', 'cellCapacity', options.cellCapacity, 1)",
      '!(Number.isFinite(tolerance) && tolerance > 0)',
      '/** Maximum live quadtree cells per feature, default 256. */',
      '/** Maximum cell splits per feature, default 1024. */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUMaximumInscribedCircleOperation': {
    signature:
      'new CPUMaximumInscribedCircleOperation(props: MaximumInscribedCircleOperationProps)',
    supportedInputs: ['props: MaximumInscribedCircleOperationProps'],
    defaults: [
      '/** Maximum live quadtree cells per feature, default 256. */',
      '/** Maximum cell splits per feature, default 1024. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'featureCount', props.featureCount, 0)",
      "validateRequirement(id, props.tolerance > 0, 'tolerance must be greater than 0')",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations, 0)",
      "validateCapacity(id, 'cellCapacity', props.cellCapacity, 1)",
      "validateRequirement( id, input.featureOffsets.length === props.featureCount + 1, 'featureOffsets length must equal featureCount + 1' )",
      '/** Maximum live quadtree cells per feature, default 256. */',
      '/** Maximum cell splits per feature, default 1024. */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUMaximumInscribedCircleOperation': {
    signature:
      'new GPUMaximumInscribedCircleOperation(props: MaximumInscribedCircleOperationProps)',
    supportedInputs: ['props: MaximumInscribedCircleOperationProps'],
    defaults: [
      "props.id ?? 'maximum-inscribed-circle'",
      '/** Maximum live quadtree cells per feature, default 256. */',
      '/** Maximum cell splits per feature, default 1024. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '/** Maximum live quadtree cells per feature, default 256. */',
      '/** Maximum cell splits per feature, default 1024. */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeNormalizeGeometryOnCPU': {
    signature:
      'export function computeNormalizeGeometryOnCPU( input: CPUNormalizeGeometryInput ): CPUNormalizeGeometryResult',
    supportedInputs: ['input: CPUNormalizeGeometryInput'],
    defaults: 'unknown',
    limits: [
      'ignores maximumNestingDepth, including values over 64.',
      'input.positions.length % 2 !== 0',
      'input.nodes.length % 4 !== 0',
      'ranges[index].first < ranges[index - 1].last'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUNormalizeGeometryOperation': {
    signature: 'new CPUNormalizeGeometryOperation(props: NormalizeGeometryOperationProps)',
    supportedInputs: ['props: NormalizeGeometryOperationProps'],
    defaults: ["props.id ?? 'normalize-geometry'", 'props.maximumNestingDepth ?? 64'],
    limits: [
      'validateCapacity(id, name, props[name], 0)',
      "validateCapacity(id, 'childCount', props.childCount, 0)",
      "validateCapacity(id, 'maximumNestingDepth', props.maximumNestingDepth ?? 64)",
      'input[name].length !== expected'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeNormalizePolygonsOnCPU': {
    signature:
      'export function computeNormalizePolygonsOnCPU( input: CPUNormalizePolygonsInput ): CPUNormalizePolygonsResult',
    supportedInputs: ['input: CPUNormalizePolygonsInput'],
    defaults: 'unknown',
    limits: ['input.positions.length % 2 !== 0 || !input.positions.every(Number.isFinite)'],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-polygons-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUNormalizePolygonsOperation': {
    signature: 'new CPUNormalizePolygonsOperation(props: NormalizePolygonsOperationProps)',
    supportedInputs: ['props: NormalizePolygonsOperationProps'],
    defaults: ["props.id ?? 'normalize-polygons'"],
    limits: ['validateCapacity(id, name, props[name], 0)', 'input[name].length !== expected'],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-polygons-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSegmentSnapOnCPU': {
    signature:
      'export function computeSegmentSnapOnCPU( input: CPUSegmentSnapInput, options: CPUSegmentSnapOptions ): CPUSegmentSnapResult',
    supportedInputs: ['input: CPUSegmentSnapInput', 'options: CPUSegmentSnapOptions'],
    defaults: 'unknown',
    limits: [
      "validateCapacity(id, 'maximumVerticesPerLine', options.maximumVerticesPerLine)",
      'input.positions.length % 2 !== 0',
      'input.referencePositions.length % 2 !== 0',
      'input.lineOffsets.length < 1',
      "validateCapacity( id, 'positions.length * (referencePositions.length + 1)', rowCount * (referenceCount + 1), 0 )",
      "validateCapacity(id, 'positions.length', rowCount, 0, MAXIMUM_POSITION_COUNT)",
      "validateCapacity(id, 'referencePositions.length', referenceCount, 0, MAXIMUM_POSITION_COUNT)",
      "validateCapacity(id, 'lineCount', lineCount, 0, MAXIMUM_LINE_COUNT)",
      "validateCapacity( id, 'lineCount * maximumVerticesPerLine', lineCount * capacity, 0, MAXIMUM_POSITION_COUNT )",
      'validateRequirement( id, canExhaustivelySnap(rowCount, referenceCount, lineCount), `(positions.length + lineCount) * (referencePositions.length + 1)^3 must not exceed MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK (${MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK})` )'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSegmentSnapOperation': {
    signature: 'new CPUSegmentSnapOperation(props: SegmentSnapOperationProps)',
    supportedInputs: ['props: SegmentSnapOperationProps'],
    defaults: [
      "props.id ?? 'segment-snap'",
      'props.output.status?.candidateOverflow ?? props.output.candidateOverflow',
      '/** Default grid route reuses bounded candidate batches. Explicit brute-force is work limited.',
      '/** Reference IDs retained per line. Default 1024. Overflow is never silently truncated. */',
      '/** Default bounded-brute-force: exact fallback within the named work limit, otherwise incomplete.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'referenceCount', props.referenceCount, 0)",
      "validateCapacity(id, 'lineCount', props.lineCount, 0)",
      "validateCapacity(id, 'maximumVerticesPerLine', props.maximumVerticesPerLine)",
      "validateCapacity( id, 'rowCount * (referenceCount + 1)', props.rowCount * (props.referenceCount + 1), 0 )",
      "validateCapacity( id, 'lineCount * maximumVerticesPerLine', props.lineCount * props.maximumVerticesPerLine, 0, MAXIMUM_POSITION_COUNT )",
      'validateRequirement( id, canExhaustivelySnap(props.rowCount, props.referenceCount, props.lineCount), `(rowCount + lineCount) * (referenceCount + 1)^3 must not exceed MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK (${MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK})` )',
      "validateCapacity(id, 'rowCount', props.rowCount, 0, MAXIMUM_POSITION_COUNT)",
      "validateCapacity(id, 'referenceCount', props.referenceCount, 0, MAXIMUM_POSITION_COUNT)",
      "validateCapacity(id, 'lineCount', props.lineCount, 0, MAXIMUM_LINE_COUNT)",
      'positions.length !== props.rowCount * 2',
      'referencePositions.length !== props.referenceCount * 2',
      'lineOffsets.length !== props.lineCount + 1'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNormalizeGeometryOperation': {
    signature: 'new GPUNormalizeGeometryOperation(props: NormalizeGeometryOperationProps)',
    supportedInputs: ['props: NormalizeGeometryOperationProps'],
    defaults: ["props.id ?? 'normalize-geometry'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-geometry-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNormalizePolygonsOperation': {
    signature: 'new GPUNormalizePolygonsOperation(props: NormalizePolygonsOperationProps)',
    supportedInputs: ['props: NormalizePolygonsOperationProps'],
    defaults: ["props.id ?? 'normalize-polygons'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSegmentSnapOperation': {
    signature: 'new GPUSegmentSnapOperation(props: SegmentSnapOperationProps)',
    supportedInputs: ['props: SegmentSnapOperationProps'],
    defaults: [
      "props.id ?? 'segment-snap'",
      'props.output.status?.candidateOverflow ?? props.output.candidateOverflow',
      '/** Default grid route reuses bounded candidate batches. Explicit brute-force is work limited.',
      '/** Reference IDs retained per line. Default 1024. Overflow is never silently truncated. */',
      '/** Default bounded-brute-force: exact fallback within the named work limit, otherwise incomplete.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'referenceCount', props.referenceCount, 0)",
      "validateCapacity(id, 'lineCount', props.lineCount, 0)",
      "validateCapacity(id, 'maximumVerticesPerLine', props.maximumVerticesPerLine, 1, 0x7fffffff)",
      "validateRequirement( id, props.output.status?.candidateOverflow === undefined || props.output.candidateOverflow === undefined || props.output.status.candidateOverflow === props.output.candidateOverflow, 'output.status.candidateOverflow and its deprecated alias must name the same output' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeConvexStraightSkeletonOnCPU': {
    signature:
      'export function computeConvexStraightSkeletonOnCPU( input: CPUConvexStraightSkeletonInput, options: CPUConvexStraightSkeletonOptions ): CPUConvexStraightSkeletonResult',
    supportedInputs: [
      'input: CPUConvexStraightSkeletonInput',
      'options: CPUConvexStraightSkeletonOptions'
    ],
    defaults: [
      'options.maximumVertices ?? 32',
      'options.segmentCapacity ?? maximumVertices * 2',
      'options.eventTolerance ?? 1e-5',
      'options.maximumIterations ?? Infinity',
      '/** Relative simultaneous-event tolerance, default 1e-5. */',
      '/** Number of skeleton arcs reserved per polygon; default twice maximumVertices. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: ['positions.length % 2 !== 0', 'ringOffsets.length < 1'],
    optionHandling: [
      'What differs from the GPU contributor:',
      '/** [writtenArcs, requiredArcs, offsetVertexCount, flags]: unsupported=1, capacity=2, budget=4, numerical=8. */'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUConvexStraightSkeletonOperation': {
    signature:
      'new CPUConvexStraightSkeletonOperation(props: ConvexStraightSkeletonOperationProps)',
    supportedInputs: ['props: ConvexStraightSkeletonOperationProps'],
    defaults: [
      "props.id ?? 'convex-straight-skeleton'",
      'props.maximumIterations ?? props.maximumVertices ?? 32',
      'props.maximumIterations ?? props.maximumVertices',
      '/** Relative simultaneous-event tolerance, default 1e-5. */',
      '/** Number of skeleton arcs reserved per polygon; default twice maximumVertices. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'polygonCount', props.polygonCount)",
      "validateCapacity( id, 'maximumIterations', props.maximumIterations ?? props.maximumVertices ?? 32, 0 )",
      'ringOffsets.length !== props.polygonCount + 1'
    ],
    optionHandling: [
      '/** [writtenArcs, requiredArcs, offsetVertexCount, flags]: unsupported=1, capacity=2, budget=4, numerical=8. */'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeLineLoopRemovalOnCPU': {
    signature:
      'export function computeLineLoopRemovalOnCPU( input: CPULineLoopRemovalInput, options: CPULineLoopRemovalOptions ): CPULineLoopRemovalResult',
    supportedInputs: ['input: CPULineLoopRemovalInput', 'options: CPULineLoopRemovalOptions'],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'Omitted capacity is uncapped. GPU intermediate binding and successor-scan limits are dropped.',
      "validateCapacity('line-loop-removal', 'vertexCapacity', options.vertexCapacity, 0)",
      "validateCapacity('line-loop-removal', 'intersectionCapacity', options.intersectionCapacity)",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPULineLoopRemovalOperation': {
    signature: 'new CPULineLoopRemovalOperation(props: LineLoopRemovalOperationProps)',
    supportedInputs: ['props: LineLoopRemovalOperationProps'],
    defaults: [
      "props.id ?? 'line-loop-removal'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'lineCount', props.lineCount)",
      "validateCapacity(id, 'vertexCapacity', props.vertexCapacity, 0)",
      "validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity)",
      'lineOffsets.length !== props.lineCount + 1',
      'featureOffsets && featureOffsets.length !== props.featureCount! + 1',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeOffsetCurveComponentsOnCPU': {
    signature:
      'export function computeOffsetCurveComponentsOnCPU( input: CPUOffsetCurveComponentsInput, options: CPUOffsetCurveComponentsOptions ): CPUOffsetCurveComponentsResult',
    supportedInputs: [
      'input: CPUOffsetCurveComponentsInput',
      'options: CPUOffsetCurveComponentsOptions'
    ],
    defaults: [
      'options.rowsPerVertex ?? 1',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'uncapped. No uncertain-predicate status, 1024-successor scan or GPU binding limit is imposed.',
      'input.original.lineOffsets.length !== input.raw.lineOffsets.length',
      'featureOffsets || rawFeatureOffsets',
      '!featureOffsets || !rawFeatureOffsets || featureOffsets.length !== rawFeatureOffsets.length || featureOffsets.length < 1',
      'featureOffsets[0] !== 0 || featureOffsets[featureOffsets.length - 1] !== input.original.lineOffsets.length - 1',
      'featureOffsets[row] !== rawFeatureOffsets[row] || (row > 0 && featureOffsets[row] < featureOffsets[row - 1])',
      '/** Output dimensions, geometric parameters and optional noding budgets (uncapped when omitted). */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'ignored by distance tests; the GPU does not define their geometric result.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUOffsetCurveComponentsOperation': {
    signature: 'new CPUOffsetCurveComponentsOperation(props: OffsetCurveComponentsOperationProps)',
    supportedInputs: ['props: OffsetCurveComponentsOperationProps'],
    defaults: [
      "props.id ?? 'offset-curve-components'",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'lineCount', props.lineCount)",
      "validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity)",
      "validateCapacity(id, 'segmentCapacity', props.segmentCapacity)",
      'originalOffsets.length !== props.lineCount + 1 || rawOffsets.length !== props.lineCount + 1',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUConvexStraightSkeletonOperation': {
    signature:
      'new GPUConvexStraightSkeletonOperation(props: ConvexStraightSkeletonOperationProps)',
    supportedInputs: ['props: ConvexStraightSkeletonOperationProps'],
    defaults: [
      'props.maximumVertices ?? 32',
      'props.segmentCapacity ?? maximumVertices * 2',
      "props.id ?? 'convex-straight-skeleton'",
      '/** Relative simultaneous-event tolerance, default 1e-5. */',
      '/** Number of skeleton arcs reserved per polygon; default twice maximumVertices. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: [
      '/** [writtenArcs, requiredArcs, offsetVertexCount, flags]: unsupported=1, capacity=2, budget=4, numerical=8. */'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-convex-straight-skeleton-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPULineLoopRemovalOperation': {
    signature: 'new GPULineLoopRemovalOperation(props: LineLoopRemovalOperationProps)',
    supportedInputs: ['props: LineLoopRemovalOperationProps'],
    defaults: [
      "props.id ?? 'line-loop-removal'",
      'props.featureCount ?? Number.NaN',
      'props.featureCount ?? props.lineCount',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity( props.id ?? 'line-loop-removal', 'featureCount', props.featureCount ?? Number.NaN, 0 )",
      "validateRequirement( props.id ?? 'line-loop-removal', props.featureCount === props.lineCount, 'featureCount must equal lineCount without input grouping' )",
      "validateCapacity( props.id ?? 'line-loop-removal', 'lines.featureOffsets.length', features.length, props.featureCount! + 1, props.featureCount! + 1 )",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUOffsetCurveComponentsOperation': {
    signature: 'new GPUOffsetCurveComponentsOperation(props: OffsetCurveComponentsOperationProps)',
    supportedInputs: ['props: OffsetCurveComponentsOperationProps'],
    defaults: [
      "props.id ?? 'offset-curve-components'",
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateRequirement( props.id ?? 'offset-curve-components', Boolean(props.original.featureOffsets) === Boolean(props.raw.featureOffsets), 'original.featureOffsets and raw.featureOffsets must both be absent or both name identical feature-to-part grouping; provide both input values' )",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeRatioConcaveHullOnCPU': {
    signature:
      'export function computeRatioConcaveHullOnCPU( positions: Float32Array, options: CPURatioConcaveHullOptions ): CPURatioConcaveHullResult',
    supportedInputs: ['positions: Float32Array', 'options: CPURatioConcaveHullOptions'],
    defaults: [
      'options.maximumIterations ?? Infinity',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'positions.length % 2 !== 0',
      "validateCapacity('ratio-concave-hull', 'maximumIterations', options.maximumIterations, 0)",
      "validateRequirement( 'ratio-concave-hull', options.ratio >= 0 && options.ratio <= 1, 'ratio must be in [0, 1]' )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPURatioConcaveHullOperation': {
    signature: 'new CPURatioConcaveHullOperation(props: RatioConcaveHullOperationProps)',
    supportedInputs: ['props: RatioConcaveHullOperationProps'],
    defaults: [
      "props.id ?? 'ratio-concave-hull'",
      '/** Allow interior triangle erosion to start holes. Default false. */',
      '/** Maximum triangle removals; default the triangle capacity. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'pointCount', props.pointCount, 3)",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations, 0)",
      "validateRequirement(id, props.ratio >= 0 && props.ratio <= 1, 'ratio must be in [0, 1]')",
      'positions.length !== props.pointCount * 2'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPURatioConcaveHullOperation': {
    signature: 'new GPURatioConcaveHullOperation(props: RatioConcaveHullOperationProps)',
    supportedInputs: ['props: RatioConcaveHullOperationProps'],
    defaults: [
      "props.id ?? 'ratio-concave-hull'",
      '/** Allow interior triangle erosion to start holes. Default false. */',
      '/** Maximum triangle removals; default the triangle capacity. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeVectorStraightSkeletonOnCPU': {
    signature:
      'export function computeVectorStraightSkeletonOnCPU( input: CPUVectorStraightSkeletonInput, options: CPUVectorStraightSkeletonOptions ): CPUVectorStraightSkeletonResult',
    supportedInputs: [
      'input: CPUVectorStraightSkeletonInput',
      'options: CPUVectorStraightSkeletonOptions'
    ],
    defaults: [
      'options.maximumVertices ?? 128',
      'options.maximumEvents ?? 128',
      'options.eventTolerance ?? 1e-5',
      'options.maximumEvents ?? Infinity',
      '/** Original vertices per polygon, excluding repeated closing vertices; default 128. */',
      '/** Edge collapse / reflex split events per polygon; default 128. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'positions.length % 2 !== 0',
      '/** float32x2, polygonCount * (maximumVertices + 2 * maximumEvents) rows. */',
      '/** Fixed polygon blocks of maximumVertices+2*maximumEvents slots. Inactive slots are identified by links. */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUVectorStraightSkeletonOperation': {
    signature:
      'new CPUVectorStraightSkeletonOperation(props: VectorStraightSkeletonOperationProps)',
    supportedInputs: ['props: VectorStraightSkeletonOperationProps'],
    defaults: [
      "props.id ?? 'vector-straight-skeleton'",
      '/** Original vertices per polygon, excluding repeated closing vertices; default 128. */',
      '/** Edge collapse / reflex split events per polygon; default 128. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'polygonCount', props.polygonCount, 0)",
      'polygonOffsets.length !== props.polygonCount + 1',
      '/** float32x2, polygonCount * (maximumVertices + 2 * maximumEvents) rows. */',
      '/** Fixed polygon blocks of maximumVertices+2*maximumEvents slots. Inactive slots are identified by links. */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUVectorStraightSkeletonOperation': {
    signature:
      'new GPUVectorStraightSkeletonOperation(props: VectorStraightSkeletonOperationProps)',
    supportedInputs: ['props: VectorStraightSkeletonOperationProps'],
    defaults: [
      'props.maximumVertices ?? 128',
      'props.maximumEvents ?? 128',
      "props.id ?? 'vector-straight-skeleton'",
      '/** Original vertices per polygon, excluding repeated closing vertices; default 128. */',
      '/** Edge collapse / reflex split events per polygon; default 128. */',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '/** float32x2, polygonCount * (maximumVertices + 2 * maximumEvents) rows. */',
      '/** Fixed polygon blocks of maximumVertices+2*maximumEvents slots. Inactive slots are identified by links. */',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPopulationSupportAlignmentOperation': {
    signature:
      'new GPUPopulationSupportAlignmentOperation(props: PopulationSupportAlignmentOperationProps)',
    supportedInputs: ['props: PopulationSupportAlignmentOperationProps'],
    defaults: ["props.id ?? 'population-support-alignment'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/gpu-population-support-alignment-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUPopulationSupportAlignmentOperation': {
    signature:
      'new CPUPopulationSupportAlignmentOperation(props: PopulationSupportAlignmentOperationProps)',
    supportedInputs: ['props: PopulationSupportAlignmentOperationProps'],
    defaults: ["props.id ?? 'population-support-alignment'"],
    limits: [
      "validateCapacity(id, 'targetCount', props.targetCount, 1)",
      "validateRequirement( id, targetIds.length === props.targetCount, 'targetIds must have targetCount rows' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePopulationSupportAlignmentOnCPU': {
    signature:
      'export function computePopulationSupportAlignmentOnCPU( input: CPUPopulationSupportAlignmentInput ): CPUPopulationSupportAlignmentResult',
    supportedInputs: ['input: CPUPopulationSupportAlignmentInput'],
    defaults: 'unknown',
    limits: [
      "validateCapacity(id, 'sourceIds.length', input.sourceIds.length, 1)",
      "validateCapacity(id, 'targetIds.length', input.targetIds.length, 1)",
      "validateRequirement( id, input.populations.length === 2 * input.sourceIds.length, 'populations must have 2 values per sourceIds row' )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/landscape-ecology/cpu-population-support-alignment-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeContinuousSpatialSampleOnCPU': {
    signature:
      'export function computeContinuousSpatialSampleOnCPU( input: CPUContinuousSpatialSampleInput, options: CPUContinuousSpatialSampleOptions ): CPUContinuousSpatialSampleResult',
    supportedInputs: [
      'input: CPUContinuousSpatialSampleInput',
      'options: CPUContinuousSpatialSampleOptions'
    ],
    defaults: [
      'options.waves ?? 1',
      'options.replicate ?? 0',
      "options.id ?? 'continuous-spatial-sample'",
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      '/** Primary wave followed by independently randomized complete replacement waves; default 1. */',
      'Maximum examined lattice candidates per wave, default 65536. An area frame must cover its whole',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '- All lattice candidates are examined: maximumCandidates is ignored and flag 4 is never set.',
      'positions.length % 2 !== 0',
      'legacySites && area && legacySites.length % 2 !== 0',
      'parameters && parameters.length !== 2',
      "validateMaximumCount( options.id ?? 'continuous-spatial-sample', 'positions arc-length lattice plus legacySites', Math.ceil(measure / spacing) + legacySiteCount, NONE - 1 )",
      'Maximum examined lattice candidates per wave, default 65536. An area frame must cover its whole'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- All lattice candidates are examined: maximumCandidates is ignored and flag 4 is never set.',
      '/** Complete-wave design properties. maximumCandidates is a GPU-only budget and is ignored. */'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUContinuousSpatialSampleOperation': {
    signature:
      'new CPUContinuousSpatialSampleOperation(props: ContinuousSpatialSampleOperationProps)',
    supportedInputs: ['props: ContinuousSpatialSampleOperationProps'],
    defaults: [
      "props.id ?? 'continuous-spatial-sample'",
      'props.legacySiteCount ?? 0',
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      '/** Primary wave followed by independently randomized complete replacement waves; default 1. */',
      'Maximum examined lattice candidates per wave, default 65536. An area frame must cover its whole',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, props.frame.legacySites !== undefined || legacySiteCount === 0, 'legacySiteCount must be zero without frame.legacySites' )",
      'positions.length !== props.frameRowCount * 2',
      "(legacySites?.length ?? 0) !== legacySiteCount * (props.frame.type === 'line' ? 1 : 2)",
      'Maximum examined lattice candidates per wave, default 65536. An area frame must cover its whole'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialBalancedSampleOnCPU': {
    signature:
      'export function computeSpatialBalancedSampleOnCPU( input: CPUSpatialBalancedSampleInput, options: CPUSpatialBalancedSampleOptions ): CPUSpatialBalancedSampleResult',
    supportedInputs: [
      'input: CPUSpatialBalancedSampleInput',
      'options: CPUSpatialBalancedSampleOptions'
    ],
    defaults: [
      'options.replicate ?? 0',
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      '/** Independent survey replicate, default 0; reproducible regardless of dispatch partitioning. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'limits of the design. Non-finite positions produce invalid-frame status 1, as on GPU.',
      'positions.length !== rowCount * 2',
      'strata && strata.length !== rowCount',
      'inclusionTickets && inclusionTickets.length !== rowCount',
      'parameters && parameters.length !== 2',
      'at most 2^24, and each mass <= sum/sampleCount. Gives exact unequal inclusion mass/step. */'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialBalancedSampleOperation': {
    signature: 'new CPUSpatialBalancedSampleOperation(props: SpatialBalancedSampleOperationProps)',
    supportedInputs: ['props: SpatialBalancedSampleOperationProps'],
    defaults: [
      "props.id ?? 'spatial-balanced-sample'",
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      '/** Independent survey replicate, default 0; reproducible regardless of dispatch partitioning. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'positions.length !== props.rowCount * 2',
      'at most 2^24, and each mass <= sum/sampleCount. Gives exact unequal inclusion mass/step. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUContinuousSpatialSampleOperation': {
    signature:
      'new GPUContinuousSpatialSampleOperation(props: ContinuousSpatialSampleOperationProps)',
    supportedInputs: ['props: ContinuousSpatialSampleOperationProps'],
    defaults: [
      "props.id ?? 'continuous-spatial-sample'",
      'props.legacySiteCount ?? 0',
      'props.waves ?? 1',
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      '/** Primary wave followed by independently randomized complete replacement waves; default 1. */',
      'Maximum examined lattice candidates per wave, default 65536. An area frame must cover its whole',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateRequirement( id, (legacySites?.length ?? 0) === legacySiteCount, 'frame.legacySites length must equal legacySiteCount; supply matching declared counts and views' )",
      'Maximum examined lattice candidates per wave, default 65536. An area frame must cover its whole'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialBalancedSampleOperation': {
    signature: 'new GPUSpatialBalancedSampleOperation(props: SpatialBalancedSampleOperationProps)',
    supportedInputs: ['props: SpatialBalancedSampleOperationProps'],
    defaults: [
      "props.id ?? 'spatial-balanced-sample'",
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      '/** Independent survey replicate, default 0; reproducible regardless of dispatch partitioning. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'at most 2^24, and each mass <= sum/sampleCount. Gives exact unequal inclusion mass/step. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computePointPatternEnvelopeOnCPU': {
    signature:
      'export function computePointPatternEnvelopeOnCPU( input: CPUPointPatternEnvelopeInput, options: CPUPointPatternEnvelopeOptions ): CPUPointPatternEnvelopeResult',
    supportedInputs: [
      'input: CPUPointPatternEnvelopeInput',
      'options: CPUPointPatternEnvelopeOptions'
    ],
    defaults: [
      'options.significanceLevel ?? 0.05',
      'options.rank ?? 1',
      'statistic defaults to maximum-deviation.'
    ],
    limits: [
      'validateRequirement(id, count > 0, `observed.length must be at least 1; received ${count}`)',
      "validateRequirement( id, options.statistic === 'extreme-rank' || input.reference?.length === count, `reference.length must equal observed.length ${count}; received ${input.reference?.length}` )",
      "validateRequirement( id, options.statistic !== 'integrated-squared-deviation' || input.integrationWeights?.length === count, `integrationWeights.length must equal observed.length ${count}; received ${input.integrationWeights?.length}` )"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'extreme-rank needs observed/simulations only. maximum-deviation also requires reference and ignores integrationWeights (including zero or absent weights). integrated-squared-deviation also requires positive finite integrationWeights. An omitted simulationCount is inferred only from an exact quotient simulations.length / observed.length of at least two; an explicit count must match. Shape/ownership/binding checks apply only to used vectors; nullModel and fittedParameters remain required.',
      'CPU maximum-deviation and integrated-squared-deviation use exact dyadic arithmetic for exceedance decisions on the stored float32 inputs. The displayed statistic is rounded to float32. With observed [1, 2^-13], nineteen simulations [1, 0], zero reference and unit weights, the exact integrated statistic is 1+2^-26, with zero exceedances and p=0.05. The displayed statistic rounds to 1.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUPointPatternEnvelopeOperation': {
    signature: 'new CPUPointPatternEnvelopeOperation(props: PointPatternEnvelopeOperationProps)',
    supportedInputs: ['props: PointPatternEnvelopeOperationProps'],
    defaults: [
      'Omitted simulationCount is inferred as simulations.length / curveRowCount at execution; the quotient must be an integer >= 2.',
      'statistic defaults to maximum-deviation.',
      'rank defaults to 1.'
    ],
    limits: [
      "validateRequirement( id, props.output.pValueInterval === undefined || props.statistic === 'extreme-rank', 'output.pValueInterval needs statistic extreme-rank' )",
      'observed.length !== props.curveRowCount',
      'Before arrays are available, construction validates an omitted-count placeholder max(2, 2 * rank); that placeholder is not the execution simulationCount default.'
    ],
    optionHandling: [
      'extreme-rank needs observed/simulations only. maximum-deviation also requires reference and ignores integrationWeights (including zero or absent weights). integrated-squared-deviation also requires positive finite integrationWeights. An omitted simulationCount is inferred only from an exact quotient simulations.length / observed.length of at least two; an explicit count must match. Shape/ownership/binding checks apply only to used vectors; nullModel and fittedParameters remain required.',
      'CPU maximum-deviation and integrated-squared-deviation use exact dyadic arithmetic for exceedance decisions on the stored float32 inputs. The displayed statistic is rounded to float32. With observed [1, 2^-13], nineteen simulations [1, 0], zero reference and unit weights, the exact integrated statistic is 1+2^-26, with zero exceedances and p=0.05. The displayed statistic rounds to 1.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUPointPatternEnvelopeOperation': {
    signature: 'new GPUPointPatternEnvelopeOperation(props: PointPatternEnvelopeOperationProps)',
    supportedInputs: ['props: PointPatternEnvelopeOperationProps'],
    defaults: ["props.id ?? 'point-pattern-envelope'", 'statistic defaults to maximum-deviation.'],
    limits: [
      "validateRequirement( id, props.statistic === 'extreme-rank' || props.reference !== undefined, 'reference is required for maximum-deviation and integrated-squared-deviation' )",
      "validateRequirement( id, props.statistic !== 'integrated-squared-deviation' || props.integrationWeights !== undefined, 'integrationWeights is required for integrated-squared-deviation' )"
    ],
    optionHandling: [
      'extreme-rank needs observed/simulations only. maximum-deviation also requires reference and ignores integrationWeights (including zero or absent weights). integrated-squared-deviation also requires positive finite integrationWeights. An omitted simulationCount is inferred only from an exact quotient simulations.length / observed.length of at least two; an explicit count must match. Shape/ownership/binding checks apply only to used vectors; nullModel and fittedParameters remain required.',
      'GPU maximum-deviation and integrated-squared-deviation use exact dyadic arithmetic for exceedance decisions on the stored float32 inputs when the float32 intermediates remain finite. The displayed statistic uses float32 arithmetic. With observed [-1], reference [2^-60] and seven simulations [1], the displayed deviations round to 1, with zero exceedances and p=0.125.'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectoryKalmanOnCPU': {
    signature:
      'export function computeTrajectoryKalmanOnCPU( input: CPUTrajectoryKalmanInput, options: CPUTrajectoryKalmanOptions = {} ): CPUTrajectoryKalmanResult',
    supportedInputs: [
      'input: CPUTrajectoryKalmanInput',
      'options: CPUTrajectoryKalmanOptions = {}'
    ],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'maximumGap=0 means unlimited even for finite time differences above the GPU float32 maximum.',
      'input.trackOffsets.length === 0',
      'input.parameters.length !== (full ? 10 : 4)',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectoryKalmanOperation': {
    signature: 'new CPUTrajectoryKalmanOperation(props: TrajectoryKalmanOperationProps)',
    supportedInputs: ['props: TrajectoryKalmanOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-kalman'",
      'props.timestamps ?? props.times',
      'props.trackOffsets ?? props.offsets',
      'first, for example with `prepareGPUTrajectoryObservations`. Defaults to native planar.',
      'and, if also given, must agree. Otherwise only valid with word times. Compile-time, default 1.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement( id, props.timestampFormat === undefined || props.timestampFormat === 'float32' || props.timestampFormat === 'uint32x2', 'timestampFormat must be float32 or uint32x2' )",
      "validateRequirement( id, Boolean(props.output.smoothedState) === Boolean(props.output.smoothedCovariance), 'output.smoothedState and output.smoothedCovariance must be given together' )",
      "validateRequirement( id, Boolean(props.timestamps || props.trackOffsets) !== Boolean(props.times || props.offsets) && Boolean(props.timestamps ?? props.times) && Boolean(props.trackOffsets ?? props.offsets), 'takes either timestamps and trackOffsets, or times and offsets' )",
      'positions.length !== props.rowCount * 2',
      "maximum gap. `maximumGap` is in the timestamps' unit and 0 means no limit.",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryKalmanOperation': {
    signature: 'new GPUTrajectoryKalmanOperation(props: TrajectoryKalmanOperationProps)',
    supportedInputs: ['props: TrajectoryKalmanOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-kalman'",
      'props.timestamps ?? props.times',
      'props.trackOffsets ?? props.offsets',
      'first, for example with `prepareGPUTrajectoryObservations`. Defaults to native planar.',
      'and, if also given, must agree. Otherwise only valid with word times. Compile-time, default 1.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "maximum gap. `maximumGap` is in the timestamps' unit and 0 means no limit.",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeTrajectoryRestrictionOnCPU': {
    signature:
      'export function computeTrajectoryRestrictionOnCPU( input: CPUTrajectoryRestrictionInput, options: CPUTrajectoryRestrictionOptions = {} ): CPUTrajectoryRestrictionResult',
    supportedInputs: [
      'input: CPUTrajectoryRestrictionInput',
      'options: CPUTrajectoryRestrictionOptions = {}'
    ],
    defaults: [
      'options.polygonRingVertexCounts ?? []',
      'options.pairwiseTracks ?? []',
      'options.intervalsPerSegment ?? 1',
      'frame. Defaults to native planar.',
      "Whether the boundary of the box or polygon belongs to it (default `'closed'`). For a polygon a",
      '/** Fixed output slots per source observation, default one. Overflow is explicit. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'validateRestrictionInputNames(id, input)',
      'pairs && options.pairwiseTracks',
      'trackOffsets.length < 1',
      '!pairwise',
      'ranges[range][0] < ranges[range - 1][1]',
      'wordTimes !== (input.timeOrigin !== undefined)',
      'input.timeOrigin && input.timeOrigin.length < 2',
      'parameters.length !== 7 + 2 * vertexCount',
      'pairs && (pairs.first.length !== pairs.second.length || (pairs.count && pairs.count.length < 1))',
      'pair.length !== 2 || !pair.every(track => Number.isInteger(track) && track >= 0 && track < trackCount)',
      "validateCapacity(id, 'output interval rows', intervalCount, 0)",
      "inclusion is explicit. `maximumGap` is in the timestamps' unit and 0 means no limit.",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUTrajectoryRestrictionOperation': {
    signature: 'new CPUTrajectoryRestrictionOperation(props: TrajectoryRestrictionOperationProps)',
    supportedInputs: ['props: TrajectoryRestrictionOperationProps'],
    defaults: [
      "props.id ?? 'trajectory-restriction'",
      'props.pairwiseTracks ?? []',
      'frame. Defaults to native planar.',
      "Whether the boundary of the box or polygon belongs to it (default `'closed'`). For a polygon a",
      '/** Fixed output slots per source observation, default one. Overflow is explicit. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 0)",
      "validateCapacity(id, 'trackCount', props.trackCount, 0)",
      "validateRequirement( id, !(props.pairs && props.pairwiseTracks), 'takes pairs or pairwiseTracks' )",
      "validateCapacity(id, 'pairCapacity', props.pairCapacity!, 0)",
      "validateRequirement(id, props.pairCapacity === undefined, 'pairCapacity requires pairs')",
      "validateRequirement( id, props.timestampFormat === undefined || ['float32', 'uint32x2'].includes(props.timestampFormat), 'timestampFormat must be float32 or uint32x2' )",
      "validateRequirement( id, (props.timestampFormat === 'uint32x2') === (props.timeOrigin !== undefined), 'timeOrigin is required only with uint32x2 timestampFormat' )",
      "validateRequirement( id, pair.length === 2 && pair.every(track => Number.isInteger(track) && track >= 0 && track < props.trackCount), 'pairwiseTracks must contain pairs of track indices below trackCount' )",
      'timestamps and trackOffsets are required',
      "validateRequirement( id, props.compactCapacity !== undefined ? compactPorts.every(value => value !== undefined) : compactPorts.every(value => value === undefined), 'compact outputs must all be present exactly when compactCapacity is given' )",
      'positions.length !== props.rowCount * 2',
      'offsets.length !== props.trackCount + 1',
      'pairs && pairs.first.length !== props.pairCapacity',
      "inclusion is explicit. `maximumGap` is in the timestamps' unit and 0 means no limit.",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-restriction/cpu-trajectory-restriction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUTrajectoryRestrictionOperation': {
    signature: 'new GPUTrajectoryRestrictionOperation(props: TrajectoryRestrictionOperationProps)',
    supportedInputs: ['props: TrajectoryRestrictionOperationProps'],
    defaults: [
      'props.intervalsPerSegment ?? 1',
      "props.id ?? 'trajectory-restriction'",
      'frame. Defaults to native planar.',
      "Whether the boundary of the box or polygon belongs to it (default `'closed'`). For a polygon a",
      '/** Fixed output slots per source observation, default one. Overflow is explicit. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "inclusion is explicit. `maximumGap` is in the timestamps' unit and 0 means no limit.",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialCrossValidationOnCPU': {
    signature:
      'export function computeSpatialCrossValidationOnCPU( input: CPUSpatialCrossValidationInput, options: CPUSpatialCrossValidationOptions ): CPUSpatialCrossValidationResult',
    supportedInputs: [
      'input: CPUSpatialCrossValidationInput',
      'options: CPUSpatialCrossValidationOptions'
    ],
    defaults: [
      'default, including generated kNNDM restarts; the GPU has a bounded default of 100 steps.',
      'options.seed ?? 0',
      'options.bufferDistance ?? 0',
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      'same folds; a different seed gives a different order. Default 0.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** Default8 deterministic farthest-first/Lloyd restarts when candidates are omitted. */'
    ],
    limits: [
      'output-length limits and a 2^24 candidate-count limit remain so float32 diagnostics preserve',
      'positions.length !== rowCount * 2',
      'mask && mask.length !== rowCount',
      "design?.mode === 'environmental' && (!features || features.length !== rowCount * design.featureCount)",
      "(design?.mode === 'nndm' || design?.mode === 'knndm') && (!predictionPositions || predictionPositions.length < 2 || predictionPositions.length % 2 !== 0)",
      "design?.mode === 'knndm' && candidates && candidates.length !== rowCount * (design.candidateCount ?? 8)"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialCrossValidationOperation': {
    signature:
      'new CPUSpatialCrossValidationOperation(props: SpatialCrossValidationOperationProps)',
    supportedInputs: ['props: SpatialCrossValidationOperationProps'],
    defaults: [
      "props.id ?? 'spatial-cross-validation'",
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      'same folds; a different seed gives a different order. Default 0.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** Default8 deterministic farthest-first/Lloyd restarts when candidates are omitted. */'
    ],
    limits: [
      "validateCapacity(id, 'design.predictionRowCount', props.design.predictionRowCount)",
      'positions.length !== props.rowCount * 2',
      "predictionPositions && (design?.mode === 'nndm' || design?.mode === 'knndm') && predictionPositions.length !== design.predictionRowCount * 2"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialCrossValidationOperation': {
    signature:
      'new GPUSpatialCrossValidationOperation(props: SpatialCrossValidationOperationProps)',
    supportedInputs: ['props: SpatialCrossValidationOperationProps'],
    defaults: [
      "props.id ?? 'spatial-cross-validation'",
      '/** Planar only; longitude-latitude is rejected. Default planar. */',
      'same folds; a different seed gives a different order. Default 0.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** Default8 deterministic farthest-first/Lloyd restarts when candidates are omitted. */'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeNestedInterpolationValidationOnCPU': {
    signature:
      'export function computeNestedInterpolationValidationOnCPU( input: CPUNestedInterpolationValidationInput, options: CPUNestedInterpolationValidationOptions ): CPUNestedInterpolationValidationResult',
    supportedInputs: [
      'input: CPUNestedInterpolationValidationInput',
      'options: CPUNestedInterpolationValidationOptions'
    ],
    defaults: [
      'options.innerFoldCount ?? 3',
      'options.neighborCount ?? 16',
      '/** Tiled row scoring by default; reference retains the original 256-row envelope. */',
      '/** Multiplier of distance before applying the radial kernel; positive, defaults 1. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'ties. uint32 output-length limits and the 2^24 exact float32 count limit remain.',
      'input.positions.length !== rowCount * 2',
      'input.innerBlocks.length !== rowCount',
      'input[name].length !== rowCount * options.foldCount'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUNestedInterpolationValidationOperation': {
    signature:
      'new CPUNestedInterpolationValidationOperation(props: NestedInterpolationValidationOperationProps)',
    supportedInputs: ['props: NestedInterpolationValidationOperationProps'],
    defaults: [
      "props.id ?? 'nested-interpolation-validation'",
      '/** Tiled row scoring by default; reference retains the original 256-row envelope. */',
      '/** Multiplier of distance before applying the radial kernel; positive, defaults 1. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: ['response.length !== props.rowCount'],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUNestedInterpolationValidationOperation': {
    signature:
      'new GPUNestedInterpolationValidationOperation(props: NestedInterpolationValidationOperationProps)',
    supportedInputs: ['props: NestedInterpolationValidationOperationProps'],
    defaults: [
      "props.id ?? 'nested-interpolation-validation'",
      '/** Tiled row scoring by default; reference retains the original 256-row envelope. */',
      '/** Multiplier of distance before applying the radial kernel; positive, defaults 1. */',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSpatialInteractionOnCPU': {
    signature:
      'export function computeSpatialInteractionOnCPU( input: CPUSpatialInteractionInput, options: CPUSpatialInteractionOptions ): CPUSpatialInteractionResult',
    supportedInputs: ['input: CPUSpatialInteractionInput', 'options: CPUSpatialInteractionOptions'],
    defaults: [
      'both probes fail preserves the interval and defaults to 32 exploratory rounds; an explicit',
      'options.maximumIterations ?? Infinity',
      'options.tolerance ?? 1e-5',
      'options.coefficientTolerance ?? 1e-5',
      'options.maximumCalibrationIterations ?? (exploring ? 32 : Infinity)',
      '/** Golden-section iterations for bounded profile calibration, 1–64; default 32. */',
      '/** Absolute coefficient bracket width required for convergence; default 1e-5. */'
    ],
    limits: [
      'maximumCalibrationIterations overrides that limit. Candidate code 1/status 3',
      'input[name] && input[name].length !== count'
    ],
    optionHandling: [
      'structural-zero covariates are ignored, as on the GPU.',
      'What differs from the GPU contributor:'
    ],
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSpatialInteractionOperation': {
    signature: 'new CPUSpatialInteractionOperation(props: SpatialInteractionOperationProps)',
    supportedInputs: ['props: SpatialInteractionOperationProps'],
    defaults: [
      "props.id ?? 'spatial-interaction'",
      '/** Golden-section iterations for bounded profile calibration, 1–64; default 32. */',
      '/** Absolute coefficient bracket width required for convergence; default 1e-5. */'
    ],
    limits: [
      "validateCapacity(id, 'pairCount', props.pairCount)",
      "validateRequirement( id, !props.output.calibration || props.calibration === 'bounded', 'output.calibration requires bounded calibration' )",
      'flows.length !== props.pairCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSpatialInteractionOperation': {
    signature: 'new GPUSpatialInteractionOperation(props: SpatialInteractionOperationProps)',
    supportedInputs: ['props: SpatialInteractionOperationProps'],
    defaults: [
      "props.id ?? 'spatial-interaction'",
      '/** Golden-section iterations for bounded profile calibration, 1–64; default 32. */',
      '/** Absolute coefficient bracket width required for convergence; default 1e-5. */'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:computeSphericalArcOverlayOnCPU': {
    signature:
      'export function computeSphericalArcOverlayOnCPU( input: CPUSphericalArcOverlayInput, options: CPUSphericalArcOverlayOptions ): CPUSphericalArcOverlayResult',
    supportedInputs: [
      'input: CPUSphericalArcOverlayInput',
      'options: CPUSphericalArcOverlayOptions'
    ],
    defaults: [
      'options.tolerance ?? 1e-6',
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'input[name].positions.length % 3',
      'input[name].ringOffsets.length < 1',
      "validateMaximumCount('spherical-arc-overlay', 'counts[2]', segments.length, 0xffffffff)",
      '/** Maximum output vertices in the entire result. GPU-only ceiling: 1024; minimum 3 in both forms. CPU has no 1024-vertex ceiling. */',
      '/** Maximum output rings in the entire result; output ringOffsets holds ringCapacity + 1 words. */',
      '/** Open assembled unit-xyz rings, with result interior on the left. Capacity 3 to {@link GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES} (1024). */'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:CPUSphericalArcOverlayOperation': {
    signature: 'new CPUSphericalArcOverlayOperation(props: SphericalArcOverlayOperationProps)',
    supportedInputs: ['props: SphericalArcOverlayOperationProps'],
    defaults: [
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '/** Maximum output vertices in the entire result. GPU-only ceiling: 1024; minimum 3 in both forms. CPU has no 1024-vertex ceiling. */',
      '/** Maximum output rings in the entire result; output ringOffsets holds ringCapacity + 1 words. */',
      '/** Open assembled unit-xyz rings, with result interior on the left. Capacity 3 to {@link GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES} (1024). */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation.ts'
  },
  '@luma-contributors/analysis/gpu-spatial-analysis:GPUSphericalArcOverlayOperation': {
    signature: 'new GPUSphericalArcOverlayOperation(props: SphericalArcOverlayOperationProps)',
    supportedInputs: ['props: SphericalArcOverlayOperationProps'],
    defaults: [
      "props.id ?? 'spherical-arc-overlay'",
      'great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      '/** Maximum output vertices in the entire result. GPU-only ceiling: 1024; minimum 3 in both forms. CPU has no 1024-vertex ceiling. */',
      '/** Maximum output rings in the entire result; output ringOffsets holds ringCapacity + 1 words. */',
      '/** Open assembled unit-xyz rings, with result interior on the left. Capacity 3 to {@link GPU_SPHERICAL_ARC_OVERLAY_MAXIMUM_OUTPUT_VERTICES} (1024). */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkTimeRoutingOnCPU': {
    signature:
      'export function computeNetworkTimeRoutingOnCPU( input: CPUNetworkTimeRoutingInput ): CPUNetworkTimeRoutingResult',
    supportedInputs: ['input: CPUNetworkTimeRoutingInput'],
    defaults: [
      'sweeps (default state count), including the final unchanged sweep that proves convergence.'
    ],
    limits: 'unknown',
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/network-optimization/cpu-network-time-routing-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkTimeRoutingOperation': {
    signature:
      'new CPUNetworkTimeRoutingOperation(originalProps: NetworkTimeRoutingOperationProps)',
    supportedInputs: ['originalProps: NetworkTimeRoutingOperationProps'],
    defaults: ["props.id ?? 'network-time-routing'"],
    limits: [
      'stateVertices.length !== props.stateCount',
      'transitions.length !== props.transitionCount * 2'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-optimization/cpu-network-time-routing-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkTimeRoutingOperation': {
    signature:
      'new GPUNetworkTimeRoutingOperation(originalProps: NetworkTimeRoutingOperationProps)',
    supportedInputs: ['originalProps: NetworkTimeRoutingOperationProps'],
    defaults: ["props.id ?? 'network-time-routing'", 'props.output.status ?? {}'],
    limits: [
      "validateRequirement( id, props.output.unrepresentable === undefined, 'output.unrepresentable requires CPUNetworkTimeRoutingOperation' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-optimization/gpu-network-time-routing-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkPointRoutingOnCPU': {
    signature:
      "export function computeNetworkPointRoutingOnCPU( input: CPUNetworkPointRoutingInput, options: {mode?: 'reference' | 'scalable'; maximumRelaxations?: number} = {} ): CPUNetworkPointRoutingResult",
    supportedInputs: [
      'input: CPUNetworkPointRoutingInput',
      "options: {mode?: 'reference' | 'scalable'; maximumRelaxations?: number} = {}"
    ],
    defaults: [
      "- The default GPU mode is scalable CSR: full-edge predecessors use reachability's smallest-parent,"
    ],
    limits: [
      "validateCapacity( 'computeNetworkPointRoutingOnCPU', 'maximumRelaxations', options.maximumRelaxations, 0, options.mode === 'reference' ? MAXIMUM_REFERENCE_RELAXATIONS : MAXIMUM_SCALABLE_RELAXATIONS )",
      '!input.costs.length',
      'input.turns.length % 2',
      'input.turnPenalties && input.turnPenalties.length * 2 !== input.turns.length',
      'input.queries.length % 2',
      'input.fractions.length !== input.queries.length'
    ],
    optionHandling: [
      "strict improvements only, as in the contributor's explicit `mode: 'reference'` route. Same-edge forward travel competes",
      'What differs from the GPU contributor:'
    ],
    source:
      'modules/analysis/src/gpu-network/network-optimization/cpu-network-point-routing-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkPointRoutingOperation': {
    signature: 'new CPUNetworkPointRoutingOperation(props: NetworkPointRoutingOperationProps)',
    supportedInputs: ['props: NetworkPointRoutingOperationProps'],
    defaults: [
      "props.id ?? 'network-point-routing'",
      '/** Scalable CSR by default; reference preserves turn-order ties and the original budgets. */',
      '/** CSR global-memory search by default; reference retains the original bounded turn-order kernel. */',
      'for both distances and zero-cost predecessor ties (default 64). Reference: 0..65536',
      'serial turn scans (default edge count). A zero budget publishes direct same-edge costs only.'
    ],
    limits: [
      "validateCapacity(id, 'edgeCount', props.edgeCount)",
      "validateCapacity(id, 'queryCount', props.queryCount)",
      "validateCapacity(id, 'queryCount * edgeCount', props.queryCount * props.edgeCount)",
      "validateCapacity( id, 'maximumRelaxations', props.maximumRelaxations, 0, props.mode === 'reference' ? MAXIMUM_REFERENCE_RELAXATIONS : MAXIMUM_SCALABLE_RELAXATIONS )",
      'edgeCosts.length !== props.edgeCount',
      'queries.length !== props.queryCount * 2'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-optimization/cpu-network-point-routing-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkPointRoutingOperation': {
    signature: 'new GPUNetworkPointRoutingOperation(props: NetworkPointRoutingOperationProps)',
    supportedInputs: ['props: NetworkPointRoutingOperationProps'],
    defaults: [
      "props.id ?? 'network-point-routing'",
      '/** Scalable CSR by default; reference preserves turn-order ties and the original budgets. */',
      '/** CSR global-memory search by default; reference retains the original bounded turn-order kernel. */',
      'for both distances and zero-cost predecessor ties (default 64). Reference: 0..65536',
      'serial turn scans (default edge count). A zero budget publishes direct same-edge costs only.'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-optimization/gpu-network-point-routing-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkFlowOnCPU': {
    signature:
      'export function computeNetworkFlowOnCPU( input: CPUNetworkFlowInput, options: CPUNetworkFlowOptions ): CPUNetworkFlowResult',
    supportedInputs: ['input: CPUNetworkFlowInput', 'options: CPUNetworkFlowOptions'],
    defaults: [
      'options.maximumAugmentations ?? Infinity',
      '(!costs ? options.maximumRelabels : options.maximumCostPhases) ?? Infinity',
      'options.maximumRelaxationRounds ?? nodeCount',
      '/** auto (default) selects parallel above the reference limits or with topology, mode or maximumCostPhases.',
      '/** Costed parallel route. Defaults to successive-shortest-path, preserving the existing',
      '/** Primal-dual route: maximum feasible shortest-path preflow batches (default 3, maximum 4096,',
      '/** Default: 1024 for reference, 8 for successive-shortest-path mode. Primal-dual and uncosted',
      'Defaults to min(nodeCount, 1024). An unfinished solve publishes the',
      '/** Uncosted parallel route: maximum global BFS relabel phases (default 3, maximum 4096,'
    ],
    limits: [
      'including the final unchanged sweep. Omitted CPU budgets are uncapped. GPU private-array caps',
      '- Total capacity may reach 0xffffffff (the uint32 output limit), rather than 0x00ffffff;',
      "validateCapacity( 'computeNetworkFlowOnCPU', name, options[name], name === 'maximumRelaxationRounds' ? 1 : 0 )",
      "validateCapacity('computeNetworkFlowOnCPU', 'nodeCount', nodeCount, 2)",
      'edges.length !== capacities.length * 2',
      'costs && costs.length !== capacities.length',
      'parameters.length !== 3',
      'topology && (topology.offsets.length !== nodeCount + 1 || topology.neighbors.length !== edgeCount)',
      '/** Primal-dual route: maximum feasible shortest-path preflow batches (default 3, maximum 4096,',
      'already feasible flow with complete=0, converged=0 and iterationLimitReached=1, unless the',
      '/** Uncosted parallel route: maximum global BFS relabel phases (default 3, maximum 4096,'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'Omit to build the residual CSR directly from the existing edge list. Weights are ignored.'
    ],
    source: 'modules/analysis/src/gpu-network/network-optimization/cpu-network-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkFlowOperation': {
    signature: 'new CPUNetworkFlowOperation(props: NetworkFlowOperationProps)',
    supportedInputs: ['props: NetworkFlowOperationProps'],
    defaults: [
      "props.id ?? 'network-flow'",
      '/** auto (default) selects parallel above the reference limits or with topology, mode or maximumCostPhases.',
      '/** Costed parallel route. Defaults to successive-shortest-path, preserving the existing',
      '/** Primal-dual route: maximum feasible shortest-path preflow batches (default 3, maximum 4096,',
      '/** Default: 1024 for reference, 8 for successive-shortest-path mode. Primal-dual and uncosted',
      'Defaults to min(nodeCount, 1024). An unfinished solve publishes the',
      '/** Uncosted parallel route: maximum global BFS relabel phases (default 3, maximum 4096,'
    ],
    limits: [
      "validateCapacity(id, 'nodeCount', props.nodeCount, 2)",
      "validateCapacity(id, 'edgeCount', props.edgeCount, 0)",
      "validateCapacity(id, 'maximumAugmentations', props.maximumAugmentations, 0)",
      "validateCapacity(id, name, props[name], name === 'maximumRelaxationRounds' ? 1 : 0)",
      'capacities.length !== props.edgeCount',
      '/** Primal-dual route: maximum feasible shortest-path preflow batches (default 3, maximum 4096,',
      'already feasible flow with complete=0, converged=0 and iterationLimitReached=1, unless the',
      '/** Uncosted parallel route: maximum global BFS relabel phases (default 3, maximum 4096,'
    ],
    optionHandling: [
      'Omit to build the residual CSR directly from the existing edge list. Weights are ignored.'
    ],
    source: 'modules/analysis/src/gpu-network/network-optimization/cpu-network-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkLouvainOnCPU': {
    signature:
      'export function computeNetworkLouvainOnCPU( input: CPUNetworkLouvainInput, options: CPUNetworkLouvainOptions ): CPUNetworkLouvainResult & {status: ReturnType<typeof decodeNetworkLouvainStatus>}',
    supportedInputs: ['input: CPUNetworkLouvainInput', 'options: CPUNetworkLouvainOptions'],
    defaults: [
      'maximumIterations (defaults 8 and 16) remain because they define the trace length and the',
      'options.maximumLevels ?? 8',
      'options.maximumIterations ?? 16',
      '/** Directed adjacency uses each arc once, including self loops. Default: undirected. */'
    ],
    limits: [
      'maximumIterations (defaults 8 and 16) remain because they define the trace length and the',
      "validateRequirement( 'computeNetworkLouvainOnCPU', Boolean(input.weights) !== Boolean(input.topology), 'exactly one of weights or topology must be supplied' )",
      'weights.length !== nodeCount ** 2',
      'parameters.length !== 2'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-network/network-optimization/cpu-network-louvain-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkLouvainOperation': {
    signature: 'new CPUNetworkLouvainOperation(props: NetworkLouvainOperationProps)',
    supportedInputs: ['props: NetworkLouvainOperationProps'],
    defaults: [
      "props.id ?? 'network-louvain'",
      '/** Directed adjacency uses each arc once, including self loops. Default: undirected. */'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-network/network-optimization/cpu-network-louvain-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkRationalAccessOnCPU': {
    signature:
      'export function computeNetworkRationalAccessOnCPU( input: CPUNetworkRationalAccessInput, options: CPUNetworkRationalAccessOptions = {} ): CPUNetworkRationalAccessResult',
    supportedInputs: [
      'input: CPUNetworkRationalAccessInput',
      'options: CPUNetworkRationalAccessOptions = {}'
    ],
    defaults: [
      '- GPU-only 64-origin, 32-facility and 4096-cycle caps are dropped. maximumCycles (default 150)',
      'and initialLimitCycles (default 20) remain algorithm controls, including nonconverged stalled',
      'options.maximumCycles ?? 150',
      'options.initialLimitCycles ?? 20',
      '/** Nonnegative demand, integral by default; total <= 2^24. */',
      '/** Fractional demands retain continuous equalizing shifts. Default integer rounding. */',
      '/** Early-cycle destination inflow cap; default 20, zero disables. */'
    ],
    limits: [
      '- GPU-only 64-origin, 32-facility and 4096-cycle caps are dropped. maximumCycles (default 150)',
      'and initialLimitCycles (default 20) remain algorithm controls, including nonconverged stalled',
      'states; the total-demand precision limit of 2^24 is retained for the float32 assignment layout.',
      "validateRequirement( 'computeNetworkRationalAccessOnCPU', input[name]!.length === 1, `${name} must contain one row` )",
      '!originCount',
      '!facilityCount',
      'costs.length !== originCount * facilityCount',
      'parameters.length !== 6',
      "validateCapacity('computeNetworkRationalAccessOnCPU', 'maximumCycles', maximumCycles, 0)",
      "validateCapacity( 'computeNetworkRationalAccessOnCPU', 'initialLimitCycles', initialLimitCycles, 0 )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/network-optimization/cpu-network-rational-access-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkRationalAccessOperation': {
    signature: 'new CPUNetworkRationalAccessOperation(props: NetworkRationalAccessOperationProps)',
    supportedInputs: ['props: NetworkRationalAccessOperationProps'],
    defaults: [
      "props.id ?? 'network-rational-access'",
      'props.maximumCycles ?? 150',
      'props.initialLimitCycles ?? 20',
      '/** Nonnegative demand, integral by default; total <= 2^24. */',
      '/** Fractional demands retain continuous equalizing shifts. Default integer rounding. */',
      '/** Early-cycle destination inflow cap; default 20, zero disables. */'
    ],
    limits: [
      "validateCapacity(id, 'originCount', props.originCount)",
      "validateCapacity(id, 'facilityCount', props.facilityCount)",
      "validateMaximumCount( id, 'originCount * facilityCount', props.originCount * props.facilityCount, 0xffffffff )",
      "validateCapacity(id, 'maximumCycles', props.maximumCycles ?? 150, 0)",
      "validateCapacity(id, 'initialLimitCycles', props.initialLimitCycles ?? 20, 0)",
      'input.demand.length !== props.originCount',
      'input.supply.length !== props.facilityCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-optimization/cpu-network-rational-access-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkFlowOperation': {
    signature: 'new GPUNetworkFlowOperation(props: NetworkFlowOperationProps)',
    supportedInputs: ['props: NetworkFlowOperationProps'],
    defaults: [
      "props.id ?? 'network-flow'",
      'props.output.status ?? {}',
      '/** auto (default) selects parallel above the reference limits or with topology, mode or maximumCostPhases.',
      '/** Costed parallel route. Defaults to successive-shortest-path, preserving the existing',
      '/** Primal-dual route: maximum feasible shortest-path preflow batches (default 3, maximum 4096,',
      '/** Default: 1024 for reference, 8 for successive-shortest-path mode. Primal-dual and uncosted',
      'Defaults to min(nodeCount, 1024). An unfinished solve publishes the',
      '/** Uncosted parallel route: maximum global BFS relabel phases (default 3, maximum 4096,'
    ],
    limits: [
      "validateRequirement( props.id ?? 'network-flow', Boolean(props.output.paths) === (props.pathCapacity !== undefined), 'pathCapacity and output.paths must be supplied together' )",
      '/** Primal-dual route: maximum feasible shortest-path preflow batches (default 3, maximum 4096,',
      'already feasible flow with complete=0, converged=0 and iterationLimitReached=1, unless the',
      '/** Uncosted parallel route: maximum global BFS relabel phases (default 3, maximum 4096,'
    ],
    optionHandling: [
      'Omit to build the residual CSR directly from the existing edge list. Weights are ignored.'
    ],
    source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkLouvainOperation': {
    signature: 'new GPUNetworkLouvainOperation(props: NetworkLouvainOperationProps)',
    supportedInputs: ['props: NetworkLouvainOperationProps'],
    defaults: [
      'props.output.status ?? {}',
      "props.id ?? 'network-louvain'",
      'props.maximumLevels ?? 8',
      'props.maximumIterations ?? 16',
      '/** Directed adjacency uses each arc once, including self loops. Default: undirected. */'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-network/network-optimization/gpu-network-louvain-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkRationalAccessOperation': {
    signature: 'new GPUNetworkRationalAccessOperation(props: NetworkRationalAccessOperationProps)',
    supportedInputs: ['props: NetworkRationalAccessOperationProps'],
    defaults: [
      "props.id ?? 'network-rational-access'",
      'props.output.status ?? {}',
      '/** Nonnegative demand, integral by default; total <= 2^24. */',
      '/** Fractional demands retain continuous equalizing shifts. Default integer rounding. */',
      '/** Early-cycle destination inflow cap; default 20, zero disables. */'
    ],
    limits: [
      "validateRequirement( props.id ?? 'network-rational-access', (props.costsComplete === undefined && props.costLimit === undefined) || props.output.status?.complete !== undefined, 'costsComplete or costLimit requires output.status.complete; inspect it before using the decision' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-optimization/gpu-network-rational-access-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeAlternativeRoutesOnCPU': {
    signature:
      'export function computeAlternativeRoutesOnCPU( input: CPUAlternativeRoutesInput, options: CPUAlternativeRoutesOptions ): CPUAlternativeRoutesResult',
    supportedInputs: ['input: CPUAlternativeRoutesInput', 'options: CPUAlternativeRoutesOptions'],
    defaults: ['options.maximumExpansions ?? Infinity', 'options.maximumFrontierSize ?? Infinity'],
    limits: [
      "validateCapacity('computeAlternativeRoutesOnCPU', name, options[name])",
      'input.targets.length !== input.sources.length',
      "validateCapacity('computeAlternativeRoutesOnCPU', 'routeCount', options.routeCount)",
      "validateCapacity( 'computeAlternativeRoutesOnCPU', 'queryCount * routeCount * nodeCount', queryCount * routeCount * nodeCount, 0 )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/alternative-routes/cpu-alternative-routes-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUAlternativeRoutesOperation': {
    signature: 'new CPUAlternativeRoutesOperation(props: AlternativeRoutesOperationProps)',
    supportedInputs: ['props: AlternativeRoutesOperationProps'],
    defaults: ["props.id ?? 'alternative-routes'"],
    limits: [
      'validateCapacity(id, name, options[name])',
      "validateCapacity(id, 'nodeCount', props.nodeCount)",
      "validateCapacity(id, 'queryCount', props.queryCount, 0)",
      "validateCapacity(id, 'routeCount', props.routeCount)",
      "validateCapacity( id, 'queryCount * routeCount * nodeCount', props.queryCount * props.routeCount * props.nodeCount, 0 )",
      'props.topology.nodePositions !== undefined',
      'nodePositions.length !== props.nodeCount * 2',
      'topology.offsets.length !== props.nodeCount + 1',
      'sources.length !== props.queryCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/alternative-routes/cpu-alternative-routes-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUAlternativeRoutesOperation': {
    signature: 'new GPUAlternativeRoutesOperation(props: AlternativeRoutesOperationProps)',
    supportedInputs: ['props: AlternativeRoutesOperationProps'],
    defaults: ["props.id ?? 'alternative-routes'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/alternative-routes/gpu-alternative-routes-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeRadiationFlowModelOnCPU': {
    signature:
      'export function computeRadiationFlowModelOnCPU( input: CPURadiationFlowModelInput ): CPURadiationFlowModelResult',
    supportedInputs: ['input: CPURadiationFlowModelInput'],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'positions.length !== zoneCount * 2',
      'outflows.length !== zoneCount',
      "validateRequirement( 'radiation-flow-model', Number.isFinite(positions[zone * 2]) && Number.isFinite(positions[zone * 2 + 1]), `positions at zone ${zone} must be finite; received [${positions[zone * 2]}, ${positions[zone * 2 + 1]}]` )",
      "validateRequirement( 'radiation-flow-model', normalization === 'none' || normalization === 'finite-size', \"normalization must be 'none' or 'finite-size'\" )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPURadiationFlowModelOperation': {
    signature: 'new CPURadiationFlowModelOperation(props: RadiationFlowModelOperationProps)',
    supportedInputs: ['props: RadiationFlowModelOperationProps'],
    defaults: [
      "props.id ?? 'radiation-flow-model'",
      "/** Prefix for generated node IDs. Defaults to `'radiation-flow-model'`. */",
      '/** Planar, or longitude and latitude great-circle (haversine). Defaults to planar. */',
      "/** Compile-time. Defaults to `'none'`. */",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'zoneCount', props.zoneCount, 2)",
      "validateRequirement( id, props.normalization === undefined || ['none', 'finite-size'].includes(props.normalization), \"normalization must be 'none' or 'finite-size'\" )",
      'populations.length !== props.zoneCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPURadiationFlowModelOperation': {
    signature: 'new GPURadiationFlowModelOperation(props: RadiationFlowModelOperationProps)',
    supportedInputs: ['props: RadiationFlowModelOperationProps'],
    defaults: [
      "props.id ?? 'radiation-flow-model'",
      "/** Prefix for generated node IDs. Defaults to `'radiation-flow-model'`. */",
      '/** Planar, or longitude and latitude great-circle (haversine). Defaults to planar. */',
      "/** Compile-time. Defaults to `'none'`. */",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/flow-aggregation/gpu-radiation-flow-model-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeMapMatchingOnCPU': {
    signature:
      'export function computeMapMatchingOnCPU( input: CPUMapMatchingInput, options: CPUMapMatchingOptions ): CPUMapMatchingResult',
    supportedInputs: ['input: CPUMapMatchingInput', 'options: CPUMapMatchingOptions'],
    defaults: [
      'may exceed 8, but the requested candidateCount and default 8 still define the model.',
      'options.candidateCount ?? 8',
      'options.traversedEdgeCapacity ?? 0',
      'options.minimumFixSpacingSigmas ?? 0',
      'options.routeNodeBudget ?? Infinity',
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'candidateOverflow, routeOverflow and budget breaks (2). Omitted, CPU searches remain uncapped.',
      "validateCapacity( 'computeMapMatchingOnCPU', 'traversedEdgeCapacity', options.traversedEdgeCapacity, 0 )",
      '!positions.length || positions.length % 2',
      '!edgeCount',
      'nodePositions.length !== nodeCount * 2',
      'weights && weights.length !== edgeCount',
      'parameters.length !== GPU_MAP_MATCHING_PARAMETER_LENGTH',
      '![sigmaValue, betaValue, searchRadius, routeFactor, routeSlack].every(Number.isFinite)',
      '!positions.every(Number.isFinite)',
      '!nodePositions.every(Number.isFinite)',
      'input.routeTable',
      'table.offsets.length !== nodeCount + 1',
      'table.costs.length !== table.targets.length',
      'table.costLimit.length !== 1',
      'table.complete && table.complete.length !== 1',
      '/** Explicit private route capacity, as on GPU; omitted keeps CPU fallback searches uncapped. */'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'edge slots are ignored. Points outside the track ranges are deterministically unmatched.'
    ],
    source: 'modules/analysis/src/gpu-network/map-matching/cpu-map-matching-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUMapMatchingOperation': {
    signature: 'new CPUMapMatchingOperation(props: MapMatchingOperationProps)',
    supportedInputs: ['props: MapMatchingOperationProps'],
    defaults: [
      "props.id ?? 'map-matching'",
      'props.traversedEdgeCapacity ?? 0',
      "/** Prefix for generated node and transient IDs. Defaults to `'map-matching'`. */",
      "`routeSlack`) in the context's length unit (meters by default, or kilometers).",
      'Compile-time candidates kept per point, 1 to 8. Defaults to 8. A candidate is a directed edge',
      'row, so the two directions of a two-way road take two slots: the default keeps the 4 nearest',
      '(33554432 on a default device). Defaults to `16 * edgeCount + 256`, capped at that maximum;',
      "Compile-time size of each route search's node table, 8 to 128. Defaults to 64. When a search",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'validateCapacity(id, name, props[name])',
      "validateRequirement( id, props.bounds.maximum[axis] > props.bounds.minimum[axis], 'bounds must have a positive extent' )",
      "validateCapacity(id, 'traversedEdgeCapacity', props.traversedEdgeCapacity ?? 0)",
      "validateCapacity(id, 'routeTable.entryCount', props.routeTable.entryCount, 0)",
      'actual !== expected',
      'tableTargets && tableTargets.length !== props.routeTable!.entryCount',
      'clamped). On longitude/latitude the maximum longitude may exceed 180 for a grid that crosses',
      'Compile-time capacity of the edge-cell table, at most {@link getGPUMapMatchingMaximumEntryCapacity}',
      '(33554432 on a default device). Defaults to `16 * edgeCount + 256`, capped at that maximum;'
    ],
    optionHandling: ['with an out-of-range target are ignored.'],
    source: 'modules/analysis/src/gpu-network/map-matching/cpu-map-matching-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUMapMatchingOperation': {
    signature: 'new GPUMapMatchingOperation(props: MapMatchingOperationProps)',
    supportedInputs: ['props: MapMatchingOperationProps'],
    defaults: [
      'props.traversedEdgeCapacity ?? 0',
      "props.id ?? 'map-matching'",
      "/** Prefix for generated node and transient IDs. Defaults to `'map-matching'`. */",
      "`routeSlack`) in the context's length unit (meters by default, or kilometers).",
      'Compile-time candidates kept per point, 1 to 8. Defaults to 8. A candidate is a directed edge',
      'row, so the two directions of a two-way road take two slots: the default keeps the 4 nearest',
      '(33554432 on a default device). Defaults to `16 * edgeCount + 256`, capped at that maximum;',
      "Compile-time size of each route search's node table, 8 to 128. Defaults to 64. When a search",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'clamped). On longitude/latitude the maximum longitude may exceed 180 for a grid that crosses',
      'Compile-time capacity of the edge-cell table, at most {@link getGPUMapMatchingMaximumEntryCapacity}',
      '(33554432 on a default device). Defaults to `16 * edgeCount + 256`, capped at that maximum;'
    ],
    optionHandling: ['with an out-of-range target are ignored.'],
    source: 'modules/analysis/src/gpu-network/map-matching/gpu-map-matching-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkCostMatrixOnCPU': {
    signature:
      'export function computeNetworkCostMatrixOnCPU( input: CPUNetworkCostMatrixInput, options: CPUNetworkCostMatrixOptions ): CPUNetworkCostMatrixResult',
    supportedInputs: ['input: CPUNetworkCostMatrixInput', 'options: CPUNetworkCostMatrixOptions'],
    defaults: [
      '(localIterations defaults to 16). Overflow waits for the next round. GPU workgroup',
      'options.startsPerRow ?? 1',
      'options.localIterations ?? 16'
    ],
    limits: [
      '- Omitted maximumIterations searches to completion. An explicit round budget permits at most',
      'maximumIterations rounds with localIterations hops and a 256-slot local queue per chunk',
      "validateCapacity(id, 'rowCount', options.rowCount)",
      "validateCapacity(id, 'startsPerRow', startsPerRow)",
      'input.startRows && input.startRows.length !== input.startNodes.length',
      'input.startCosts && input.startCosts.length !== input.startNodes.length',
      'input.costLimit && input.costLimit.length !== 1',
      'input.targetNodes && !input.targetNodes.length',
      "validateCapacity(id, 'rowCount * columnCount', options.rowCount * columnCount)"
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '/** Starts may share rows; invalid rows and targets are ignored exactly as on GPU. */'
    ],
    source:
      'modules/analysis/src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkCostMatrixOperation': {
    signature: 'new CPUNetworkCostMatrixOperation(props: NetworkCostMatrixOperationProps)',
    supportedInputs: ['props: NetworkCostMatrixOperationProps'],
    defaults: [
      "props.id ?? 'network-cost-matrix'",
      'props.startsPerRow ?? 1',
      'props.targetCount ?? 0'
    ],
    limits: [
      "validateCapacity(id, 'nodeCount', props.nodeCount)",
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateCapacity(id, 'startsPerRow', props.startsPerRow ?? 1)",
      "validateCapacity(id, 'targetCount', props.targetCount ?? 0)",
      'props.topology.nodePositions !== undefined',
      'nodePositions.length !== props.nodeCount * 2',
      'topology.offsets.length !== props.nodeCount + 1',
      'targetNodes && targetNodes.length !== props.targetCount'
    ],
    optionHandling: [
      'nodedTopology and output.status.sourceComplete are supported by the GPU operation only; use GPUNetworkCostMatrixOperation for noded live counts and build evidence. Ordinary CPU CSR nodeCount includes isolated nodes.',
      'The compatibility status type names converged, iterationLimitReached and sourceComplete; CPU accepts the two search fields and rejects sourceComplete rather than implying a source-build certificate.'
    ],
    source:
      'modules/analysis/src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkCostMatrixOperation': {
    signature: 'new GPUNetworkCostMatrixOperation(props: GPUNetworkCostMatrixOperationProps)',
    supportedInputs: ['props: GPUNetworkCostMatrixOperationProps'],
    defaults: [
      "props.id ?? 'network-cost-matrix'",
      'props.nodedTopology ?? {}',
      'props.targetCount ?? 0',
      'props.output.status ?? {}'
    ],
    limits: 'unknown',
    optionHandling: [
      'nodedTopology.nodeCount and nodedTopology.pieceCount are live graph-value inputs; topology storage remains allocated capacity. nodedTopology.piecesOverflow is required; overflow, uncertainCount and candidateOverflow retain source build evidence.',
      'output.status has exactly three optional keys: converged, iterationLimitReached, sourceComplete. With nodedTopology, sourceComplete is required and certifies the source build independently of search convergence; missing overflow or uncertainCount evidence conservatively leaves sourceComplete zero.',
      'direction accepts start-to-target or target-to-start (default start-to-target).'
    ],
    source:
      'modules/analysis/src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkPathExtractionOnCPU': {
    signature:
      'export function computeNetworkPathExtractionOnCPU( input: CPUNetworkPathExtractionInput ): CPUNetworkPathExtractionResult',
    supportedInputs: ['input: CPUNetworkPathExtractionInput'],
    defaults: ['/** Maximum nodes per path, binding on CPU and GPU. Defaults to 1024. */'],
    limits: [
      'nodeCount < 1',
      'costs.length !== nodeCount',
      'nodeIds && nodeIds.length !== nodeCount',
      'predecessorsComplete && predecessorsComplete.length !== 1',
      "typeof input.targetCount === 'object' && input.targetCount.length !== 1",
      "validateCapacity( 'network-path-extraction', 'targetCount', input.targetCount, 0, targets.length )",
      "validateCapacity('network-path-extraction', 'capacity', input.capacity, 0)",
      "validateCapacity('network-path-extraction', 'maximumPathLength', input.maximumPathLength)",
      "validateRequirement( 'network-path-extraction', predecessorEdges.length === nodeCount, `predecessorEdges length must equal predecessors length ${nodeCount}; received ${predecessorEdges.length}` )",
      "validateRequirement( 'network-path-extraction', edge === GPU_NETWORK_PATH_NO_EDGE, `predecessorEdges must use NONE for a root; received ${edge} at node ${head}` )",
      "validateRequirement( 'network-path-extraction', false, `predecessorEdges must identify a passable edge from the parent to its node; received ${edge} for parent ${tail}, node ${head}` )",
      'edges',
      'Edge output requires predecessorEdges from the search.',
      'edges.offsets.length !== nodeCount + 1',
      'edges.weights.length !== edges.neighbors.length',
      'edges.edgeIds && edges.edgeIds.length !== edges.neighbors.length',
      "validateCapacity('network-path-extraction', 'edges.capacity', edges.capacity, 0)",
      "validateMaximumCount( 'network-path-extraction', 'required node rows (split targets into smaller batches)', nodeTotal + path.length, MAXIMUM_PATH_OUTPUT_ROW_COUNT )",
      '/** Maximum nodes per path, binding on CPU and GPU. Defaults to 1024. */',
      'The safe bound is activeTargetUpperBound * min(nodeCount, maximumPathLength). Larger bounds are admitted with checked actual totals above 4294967295 rejected before publication. Numeric targetCount must lie in [0, target capacity] and rejects outside that range; the live uint32 targetCount word is clamped to target capacity. Target capacity is targets.length for the array form and targetCapacity for the named form. An omitted count uses the allocated capacity as the active target prefix; a graph-value count name selects the live prefix and uses targetCapacity as its construction-time upper bound.'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-analysis/cpu-network-path-extraction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkPathExtractionOperation': {
    signature:
      'new CPUNetworkPathExtractionOperation(props: CPUNetworkPathExtractionOperationProps)',
    supportedInputs: ['props: CPUNetworkPathExtractionOperationProps'],
    defaults: ["props.id ?? 'network-path-extraction'"],
    limits: [
      "validateCapacity(id, 'nodeCount', props.nodeCount)",
      "validateCapacity(id, 'targetCapacity', props.targetCapacity, 0)",
      "validateCapacity(id, 'targetCount', props.targetCount, 0, props.targetCapacity)",
      "validateCapacity(id, 'maximumPathLength', props.maximumPathLength)",
      'Edge output requires the predecessorEdges graph value from the search.',
      "validateCapacity(id, 'edges.edgeCount', props.edges.edgeCount, 0)",
      'predecessors.length !== props.nodeCount',
      'targets.length !== props.targetCapacity',
      'props.edges && neighbors!.length !== props.edges.edgeCount',
      "props.edges?.topology.nodePositions !== undefined && getCPUOperationFloat32Rows(id, inputs, 'nodePositions', 'float32x2').length !== props.nodeCount * 2",
      'The safe bound is activeTargetUpperBound * min(nodeCount, maximumPathLength). Larger bounds are admitted with checked actual totals above 4294967295 rejected before publication. Numeric targetCount must lie in [0, target capacity] and rejects outside that range; the live uint32 targetCount word is clamped to target capacity. Target capacity is targets.length for the array form and targetCapacity for the named form. An omitted count uses the allocated capacity as the active target prefix; a graph-value count name selects the live prefix and uses targetCapacity as its construction-time upper bound.'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-analysis/cpu-network-path-extraction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkPathExtractionOperation': {
    signature: 'new GPUNetworkPathExtractionOperation(props: NetworkPathExtractionOperationProps)',
    supportedInputs: ['props: NetworkPathExtractionOperationProps'],
    defaults: ["props.id ?? 'network-path-extraction'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-analysis/gpu-network-path-extraction-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkServiceAreasOnCPU': {
    signature:
      'export function computeNetworkServiceAreasOnCPU( input: CPUNetworkServiceAreasInput, options: CPUNetworkServiceAreasOptions = {} ): CPUNetworkServiceAreasResult',
    supportedInputs: [
      'input: CPUNetworkServiceAreasInput',
      'options: CPUNetworkServiceAreasOptions = {}'
    ],
    defaults: [
      'options.localIterations ?? 16',
      'options.maximumIterations ?? 64',
      'options.maximumLabelIterations ?? Math.min(options.maximumIterations ?? 64, 32)'
    ],
    limits: [
      "validateCapacity('computeNetworkServiceAreasOnCPU', name, options[name])",
      'input.facilityCosts && input.facilityCosts.length !== input.facilities.length',
      "typeof input.facilityCount === 'object' && input.facilityCount.length !== 1",
      "validateCapacity( 'computeNetworkServiceAreasOnCPU', 'facilityCount', input.facilityCount, 0, input.facilities.length )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/network-analysis/cpu-network-service-areas-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkServiceAreasOperation': {
    signature: 'new CPUNetworkServiceAreasOperation(props: NetworkServiceAreasOperationProps)',
    supportedInputs: ['props: NetworkServiceAreasOperationProps'],
    defaults: ["props.id ?? 'network-service-areas'"],
    limits: [
      'validateCapacity(id, name, options[name])',
      "validateCapacity(id, 'nodeCount', props.nodeCount)",
      "validateCapacity(id, 'facilityCapacity', props.facilityCapacity, 0)",
      "validateCapacity(id, 'facilityCount', props.facilityCount, 0, props.facilityCapacity)",
      'props.topology.nodePositions !== undefined',
      'nodePositions.length !== props.nodeCount * 2',
      'topology.offsets.length !== props.nodeCount + 1',
      'facilities.length !== props.facilityCapacity'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-analysis/cpu-network-service-areas-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkServiceAreasOperation': {
    signature: 'new GPUNetworkServiceAreasOperation(props: NetworkServiceAreasOperationProps)',
    supportedInputs: ['props: NetworkServiceAreasOperationProps'],
    defaults: ["props.id ?? 'network-service-areas'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-analysis/gpu-network-service-areas-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkCentralityOnCPU': {
    signature:
      'export function computeNetworkCentralityOnCPU( input: CPUNetworkCentralityInput, options: CPUNetworkCentralityOptions = {} ): CPUNetworkCentralityResult',
    supportedInputs: [
      'input: CPUNetworkCentralityInput',
      'options: CPUNetworkCentralityOptions = {}'
    ],
    defaults: [
      '- Explicit maximumIterations bounds search rounds (localIterations defaults to 16, with',
      'options.localIterations ?? 16',
      'options.maximumIterations ?? Infinity',
      'options.betweennessScale ?? 1'
    ],
    limits: [
      '- Explicit maximumIterations bounds search rounds (localIterations defaults to 16, with',
      "validateCapacity( 'computeNetworkCentralityOnCPU', 'maximumIterations', options.maximumIterations )",
      "validateCapacity('computeNetworkCentralityOnCPU', 'localIterations', localIterations)",
      "validateRequirement( 'computeNetworkCentralityOnCPU', options.computeBetweenness === false || Boolean(input.reverseTopology), 'reverseTopology is required for betweenness; supply the transpose or set computeBetweenness: false' )",
      'network.offsets.length !== nodeCount + 1 || nodeCount < 1',
      'network.weights.length !== network.neighbors.length',
      'costLimit && costLimit.length !== 1',
      "validateRequirement( 'computeNetworkCentralityOnCPU', options.sampling === undefined || options.sampling === 'exact' || options.sampling === 'sampled', \"sampling must be 'exact' or 'sampled'\" )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/network-centrality/cpu-network-centrality-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkCentralityOperation': {
    signature: 'new CPUNetworkCentralityOperation(props: NetworkCentralityOperationProps)',
    supportedInputs: ['props: NetworkCentralityOperationProps'],
    defaults: [
      "props.id ?? 'network-centrality'",
      'props.betweennessScale ?? 1',
      'props.localIterations ?? 16',
      "props.reverseEdgeCount ?? getCPUOperationUint32Rows(id, inputs, 'reverseTopology.neighbors').length",
      "/** Prefix for generated node and transient IDs. Defaults to `'network-centrality'`. */",
      'count. Defaults to `recommendLaneCount()` (about 1M expanded nodes per batch), at most the pivot',
      'delta phases of every batch, 1 to 1024. Defaults to 64. The search needs about',
      'Compile-time number of hops one workgroup chains inside a round, 1 to 64. Defaults to 16, which',
      'Defaults to 1. For estimates from sampled pivots use `nodeCount / pivotCount`.',
      "Labels deliberate pivot sampling in status.approximation. Defaults to 'exact' (the supplied"
    ],
    limits: [
      "validateCapacity(id, 'nodeCount', props.nodeCount)",
      "validateCapacity(id, 'edgeCount', props.edgeCount, 0)",
      "validateCapacity(id, 'reverseEdgeCount', props.reverseEdgeCount, 0)",
      "validateCapacity(id, 'pivotCount', props.pivotCount)",
      "validateRequirement( id, Object.entries(props.output).some( ([port, value]) => port !== 'status' && value !== undefined ), 'output must request a centrality value' )",
      "validateRequirement( id, props.output.status?.zeroWeightCycle === undefined || props.output.betweenness !== undefined || props.output.edgeBetweenness !== undefined, 'output.status.zeroWeightCycle needs a betweenness output' )",
      "validateRequirement( id, props.sampling === undefined || props.sampling === 'exact' || props.sampling === 'sampled', \"sampling must be 'exact' or 'sampled'\" )",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations)",
      "validateCapacity(id, 'localIterations', props.localIterations ?? 16)",
      'topology.offsets.length !== props.nodeCount + 1 || topology.neighbors.length !== edgeCount',
      "props[prefix]?.nodePositions !== undefined && getCPUOperationFloat32Rows(id, inputs, `${prefix}.nodePositions`, 'float32x2') .length !== props.nodeCount * 2",
      'pivotNodes.length !== props.pivotCount',
      'count. Defaults to `recommendLaneCount()` (about 1M expanded nodes per batch), at most the pivot',
      'grouping of the sums. With any closeness output, laneCount is at most 16,777,215:',
      'Scheduling caps this budget at the node count: nonnegative shortest paths need at most',
      '`nodeCount - 1` edges, and their strictly increasing tight-edge graph has at most `nodeCount`',
      'accumulation phases finished within `maximumIterations` and ordered every reached node; 0'
    ],
    optionHandling: [
      'An undirected network lists both directions of every edge. Targets `>= nodeCount` are ignored',
      'sample for estimates (set `betweennessScale`). Out-of-range nodes are ignored and their'
    ],
    source:
      'modules/analysis/src/gpu-network/network-centrality/cpu-network-centrality-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkCentralityOperation': {
    signature: 'new GPUNetworkCentralityOperation(props: NetworkCentralityOperationProps)',
    supportedInputs: ['props: NetworkCentralityOperationProps'],
    defaults: [
      "props.id ?? 'network-centrality'",
      'props.output.status ?? {}',
      "/** Prefix for generated node and transient IDs. Defaults to `'network-centrality'`. */",
      'count. Defaults to `recommendLaneCount()` (about 1M expanded nodes per batch), at most the pivot',
      'delta phases of every batch, 1 to 1024. Defaults to 64. The search needs about',
      'Compile-time number of hops one workgroup chains inside a round, 1 to 64. Defaults to 16, which',
      'Defaults to 1. For estimates from sampled pivots use `nodeCount / pivotCount`.',
      "Labels deliberate pivot sampling in status.approximation. Defaults to 'exact' (the supplied"
    ],
    limits: [
      "validateRequirement( props.id ?? 'network-centrality', !(props.output.betweenness || props.output.edgeBetweenness) || Boolean(props.reverseTopology), 'reverseTopology is required for betweenness' )",
      "validateCapacity( props.id ?? 'network-centrality', 'reverseEdgeCount', props.reverseEdgeCount, 0 )",
      "validateRequirement( props.id ?? 'network-centrality', actual === props.reverseEdgeCount, `reverseEdgeCount must equal reverseTopology.neighbors.length ${actual}; received ${props.reverseEdgeCount}` )",
      'count. Defaults to `recommendLaneCount()` (about 1M expanded nodes per batch), at most the pivot',
      'grouping of the sums. With any closeness output, laneCount is at most 16,777,215:',
      'Scheduling caps this budget at the node count: nonnegative shortest paths need at most',
      '`nodeCount - 1` edges, and their strictly increasing tight-edge graph has at most `nodeCount`',
      'accumulation phases finished within `maximumIterations` and ordered every reached node; 0'
    ],
    optionHandling: [
      'An undirected network lists both directions of every edge. Targets `>= nodeCount` are ignored',
      'sample for estimates (set `betweennessScale`). Out-of-range nodes are ignored and their'
    ],
    source:
      'modules/analysis/src/gpu-network/network-centrality/gpu-network-centrality-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkCostTableOnCPU': {
    signature:
      'export function computeNetworkCostTableOnCPU( input: CPUNetworkCostTableInput, options: CPUNetworkCostTableOptions ): CPUNetworkCostTableResult',
    supportedInputs: ['input: CPUNetworkCostTableInput', 'options: CPUNetworkCostTableOptions'],
    defaults: [
      'localIterations defaults to 16. CPU workgroups visit',
      'options.maximumIterations ?? Infinity',
      'options.localIterations ?? DEFAULT_LOCAL_ITERATIONS'
    ],
    limits: [
      "validateCapacity('computeNetworkCostTableOnCPU', 'capacity', options.capacity)",
      "validateCapacity('computeNetworkCostTableOnCPU', 'maximumIterations', maximumIterations, 1)",
      "validateCapacity( 'computeNetworkCostTableOnCPU', 'localIterations', localIterations, 1, GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS )",
      'input.costLimit.length !== 1'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/network-cost-table/cpu-network-cost-table-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkCostTableOperation': {
    signature: 'new CPUNetworkCostTableOperation(props: NetworkCostTableOperationProps)',
    supportedInputs: ['props: NetworkCostTableOperationProps'],
    defaults: ["props.id ?? 'network-cost-table'"],
    limits: [
      "validateCapacity(id, 'nodeCount', props.nodeCount, 1, NETWORK_COST_TABLE_MAXIMUM_NODE_COUNT)",
      "validateCapacity(id, 'capacity', props.capacity)",
      'props.topology.nodePositions !== undefined',
      'nodePositions.length !== props.nodeCount * 2',
      'topology.offsets.length !== props.nodeCount + 1'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-cost-table/cpu-network-cost-table-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkCostTableOperation': {
    signature: 'new GPUNetworkCostTableOperation(props: NetworkCostTableOperationProps)',
    supportedInputs: ['props: NetworkCostTableOperationProps'],
    defaults: ["props.id ?? 'network-cost-table'", 'props.output.status ?? {}'],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-cost-table/gpu-network-cost-table-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkReachabilityOnCPU': {
    signature:
      'export function computeNetworkReachabilityOnCPU( input: CPUNetworkReachabilityInput, options: CPUNetworkReachabilityOptions = {} ): CPUNetworkReachabilityResult',
    supportedInputs: [
      'input: CPUNetworkReachabilityInput',
      'options: CPUNetworkReachabilityOptions = {}'
    ],
    defaults: [
      'localIterations hops per round (default 32). GPU workgroup scheduling can finish fewer hops.',
      'options.laneCount ?? 1',
      'options.localIterations ?? 32',
      'options.maximumIterations ?? 64'
    ],
    limits: [
      '- With explicit round budgets or activeIterations, a synchronous CPU frontier consumes at most',
      '- iterationCount counts bounded frontier rounds, or 1 for an uncapped CPU search.',
      "validateCapacity('computeNetworkReachabilityOnCPU', name, value)",
      'input.costBreaks',
      '!input.costBreaks.length',
      '/** Explicit round budgets bind the deterministic CPU frontier too; omitted budgets are uncapped. */'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '- Negative/NaN weights and starts are ignored; NaN costLimit disables the cutoff, like the GPU.'
    ],
    source:
      'modules/analysis/src/gpu-network/network-reachability/cpu-network-reachability-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkReachabilityOperation': {
    signature: 'new CPUNetworkReachabilityOperation(props: NetworkReachabilityOperationProps)',
    supportedInputs: ['props: NetworkReachabilityOperationProps'],
    defaults: [
      "props.id ?? 'network-reachability'",
      'props.laneCount ?? 1',
      'props.costBreakCount ?? 0'
    ],
    limits: [
      "validateCapacity(id, 'nodeCount', props.nodeCount)",
      "validateCapacity(id, 'laneCount', props.laneCount ?? 1)",
      "validateCapacity(id, 'nodeCount * laneCount', props.nodeCount * (props.laneCount ?? 1))",
      'validateCapacity(id, name, props[name])',
      "validateRequirement( id, (props.laneCount ?? 1) === 1 || (!props.output.predecessors && !props.output.unresolvedCount), 'laneCount > 1 cannot request predecessors or unresolvedCount' )",
      "validateRequirement( id, !props.output.unresolvedCount || Boolean(props.output.predecessors), 'output.unresolvedCount requires output.predecessors' )",
      "validateRequirement( id, Boolean(props.costBreaks) === Boolean(props.output.bands || props.output.bandCounts), 'costBreaks is required exactly when bands or bandCounts is requested' )",
      "validateCapacity(id, 'costBreakCount', props.costBreakCount ?? 0)",
      "validateCapacity(id, 'startCount', props.startCount, 0)",
      'props.topology.nodePositions !== undefined',
      'nodePositions.length !== props.nodeCount * 2',
      'topology.offsets.length !== props.nodeCount + 1',
      'props.costBreakCount !== undefined && result.bandCounts.length !== props.costBreakCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-reachability/cpu-network-reachability-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkReachabilityOperation': {
    signature: 'new GPUNetworkReachabilityOperation(props: NetworkReachabilityOperationProps)',
    supportedInputs: ['props: NetworkReachabilityOperationProps'],
    defaults: [
      'props.laneCount ?? 1',
      'props.costBreakCount ?? 0',
      "props.id ?? 'network-reachability'",
      'props.output.status ?? {}'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-reachability/gpu-network-reachability-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkStructureOnCPU': {
    signature:
      "export function computeNetworkStructureOnCPU( input: | CPUNetworkStructureInput | { topology: Omit<CPUNetworkStructureInput, 'reverseEdgeSlots'>; reverseEdgeSlots?: Uint32Array; }, options: {maximumIterations?: number} = {} ): CPUNetworkStructureResult",
    supportedInputs: [
      "input: | CPUNetworkStructureInput | { topology: Omit<CPUNetworkStructureInput, 'reverseEdgeSlots'>; reverseEdgeSlots?: Uint32Array; }",
      'options: {maximumIterations?: number} = {}'
    ],
    defaults: 'unknown',
    limits: [
      'Omit the budget for an exact Tarjan traversal and status [converged=1, iterationCount=1, limit=0].',
      "validateCapacity( 'computeNetworkStructureOnCPU', 'maximumIterations', options.maximumIterations, 1, GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS )",
      'nodeCount < 1 || offsets.some( (offset, index) => offset > edgeCount || (index > 0 && offset < offsets[index - 1]) )',
      'reverseEdgeSlots && reverseEdgeSlots.length !== edgeCount'
    ],
    optionHandling: [
      'IDs are ignored. Without reverse slots only strongComponents is meaningful.',
      'What differs from the GPU contributor:'
    ],
    source: 'modules/analysis/src/gpu-network/network-structure/cpu-network-structure-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkStructureOperation': {
    signature: 'new CPUNetworkStructureOperation(props: NetworkStructureOperationProps)',
    supportedInputs: ['props: NetworkStructureOperationProps'],
    defaults: [
      "props.id ?? 'network-structure'",
      'props.strongComponentsMaximumIterations ?? GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_DEFAULT_ITERATIONS',
      '/** Default auto preserves bounded exact SCCs and uses parallel resilience. Explicit parallel opts into budgeted SCCs at every size; exact selects both reference kernels. */',
      '/** SCC state-machine budget only; exhaustion publishes 0xffffffff labels. Default 256. */'
    ],
    limits: [
      "validateCapacity(id, 'nodeCount', props.nodeCount)",
      "validateCapacity(id, 'edgeCount', props.edgeCount, 0)",
      "validateRequirement( id, props.mode === undefined || ['auto', 'parallel', 'exact'].includes(props.mode), 'mode must be auto, parallel or exact' )",
      "validateCapacity( id, 'strongComponentsMaximumIterations', props.strongComponentsMaximumIterations, 1, GPU_NETWORK_STRONGLY_CONNECTED_COMPONENTS_MAXIMUM_ITERATIONS )",
      "validateRequirement( id, !props.strongComponentsStatus || props.output.strongComponents !== undefined, 'strongComponentsStatus requires output.strongComponents' )",
      "validateRequirement( id, Object.values(props.output).some(value => value !== undefined), 'output must contain at least one result' )",
      "validateRequirement( id, !hasResilienceOutput || props.reverseEdgeSlots !== undefined, 'reverseEdgeSlots is required for resilience output' )",
      'offsets.length !== props.nodeCount + 1',
      'neighbors.length !== props.edgeCount',
      "props.topology.weights !== undefined && getCPUOperationFloat32Rows(id, inputs, 'weights').length !== props.edgeCount",
      "props.topology.nodePositions !== undefined && getCPUOperationFloat32Rows(id, inputs, 'nodePositions', 'float32x2').length !== props.nodeCount * 2"
    ],
    optionHandling: [
      'As on GPU, reverse slots are ignored when only strongComponents is requested.'
    ],
    source: 'modules/analysis/src/gpu-network/network-structure/cpu-network-structure-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkStructureOperation': {
    signature: 'new GPUNetworkStructureOperation(props: GPUNetworkStructureOperationProps)',
    supportedInputs: ['props: GPUNetworkStructureOperationProps'],
    defaults: [
      "props.id ?? 'network-structure'",
      '/** Default auto preserves bounded exact SCCs and uses parallel resilience. Explicit parallel opts into budgeted SCCs at every size; exact selects both reference kernels. */',
      '/** SCC state-machine budget only; exhaustion publishes 0xffffffff labels. Default 256. */'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-network/network-structure/gpu-network-structure-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeNetworkTravelTimePercentilesOnCPU': {
    signature:
      'export function computeNetworkTravelTimePercentilesOnCPU( input: CPUNetworkTravelTimePercentilesInput ): CPUNetworkTravelTimePercentilesResult',
    supportedInputs: ['input: CPUNetworkTravelTimePercentilesInput'],
    defaults: 'unknown',
    limits: ['input.initialState.length !== 1', 'input.window.length !== 2'],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUNetworkTravelTimePercentilesOperation': {
    signature:
      'new CPUNetworkTravelTimePercentilesOperation(originalProps: NetworkTravelTimePercentilesOperationProps)',
    supportedInputs: ['originalProps: NetworkTravelTimePercentilesOperationProps'],
    defaults: ["props.id ?? 'network-travel-time-percentiles'"],
    limits: [
      'stateVertices.length !== props.stateCount',
      'transitions.length !== props.transitionCount * 2'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUNetworkTravelTimePercentilesOperation': {
    signature:
      'new GPUNetworkTravelTimePercentilesOperation(originalProps: NetworkTravelTimePercentilesOperationProps)',
    supportedInputs: ['originalProps: NetworkTravelTimePercentilesOperationProps'],
    defaults: ["props.id ?? 'network-travel-time-percentiles'"],
    limits: [
      "validateRequirement( id, props.output.unrepresentable === undefined, 'output.unrepresentable requires CPUNetworkTravelTimePercentilesOperation' )",
      "validateCapacity(id, 'vertexCount', props.vertexCount)",
      "validateCapacity( id, 'departureCount', props.departureCount, 1, GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_DEPARTURE_COUNT )",
      "validateCapacity( id, 'percentiles rows', props.percentiles.length, 1, GPU_NETWORK_TRAVEL_TIME_PERCENTILES_MAXIMUM_PERCENTILE_COUNT )",
      "validateRequirement( id, props.percentiles.every( (percentile, index) => Number.isFinite(percentile) && percentile > 0 && percentile <= 100 && (index === 0 || percentile >= props.percentiles[index - 1]) ), 'percentiles must be ascending numbers in (0, 100]' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-network/network-time-geography/gpu-network-travel-time-percentiles-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:GPUKShortestPathsOperation': {
    signature: 'new GPUKShortestPathsOperation(props: KShortestPathsOperationProps)',
    supportedInputs: ['props: KShortestPathsOperationProps'],
    defaults: [
      "props.id ?? 'k-shortest-paths'",
      'props.maximumPathLength ?? 64',
      '/** Maximum vertices in one returned route; default 64. An overlong shortest spur stops the query. */',
      '/** Persistent Yen candidates per query; default 256. Exhaustion sets candidateOverflow. */',
      '/** Rounds per shortest-path and zero-cost tie search; default 64, with 64 local hops per round. */'
    ],
    limits: [
      '/** Maximum vertices in one returned route; default 64. An overlong shortest spur stops the query. */',
      '/** query × routeCount × maximumPathLength, padded with 0xffffffff. */'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-network/k-shortest-paths/gpu-k-shortest-paths-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:CPUKShortestPathsOperation': {
    signature: 'new CPUKShortestPathsOperation(props: KShortestPathsOperationProps)',
    supportedInputs: ['props: KShortestPathsOperationProps'],
    defaults: [
      'props.maximumPathLength ?? 64',
      "props.id ?? 'k-shortest-paths'",
      'props.candidateCapacity ?? 256',
      '/** Maximum vertices in one returned route; default 64. An overlong shortest spur stops the query. */',
      '/** Persistent Yen candidates per query; default 256. Exhaustion sets candidateOverflow. */',
      '/** Rounds per shortest-path and zero-cost tie search; default 64, with 64 local hops per round. */'
    ],
    limits: [
      "validateCapacity(id, 'queryCount', props.queryCount, 0)",
      "validateCapacity(id, 'routeCount', props.routeCount)",
      "validateCapacity(id, 'maximumPathLength', props.maximumPathLength ?? 64)",
      "validateCapacity(id, 'candidateCapacity', props.candidateCapacity ?? 256)",
      "validateCapacity(id, 'maximumIterations', props.maximumIterations)",
      '/** Maximum vertices in one returned route; default 64. An overlong shortest spur stops the query. */',
      '/** query × routeCount × maximumPathLength, padded with 0xffffffff. */'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation.ts'
  },
  '@luma-contributors/analysis/gpu-network:computeKShortestPathsOnCPU': {
    signature:
      'export function computeKShortestPathsOnCPU( input: CPUKShortestPathsInput, options: CPUKShortestPathsOptions ): CPUKShortestPathsResult',
    supportedInputs: ['input: CPUKShortestPathsInput', 'options: CPUKShortestPathsOptions'],
    defaults: ['options.maximumPathLength ?? 64', 'options.candidateCapacity ?? 256'],
    limits: [
      'offsets[row] > neighbors.length || (row > 0 && offsets[row] < offsets[row - 1])',
      "validateCapacity(id, 'sources.length', input.sources.length, 0)",
      "validateCapacity(id, 'maximumIterations', maximumIterations)",
      "validateCapacity(id, 'candidateCapacity', candidateCapacity)"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-network/k-shortest-paths/cpu-k-shortest-paths-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainTiledFlow': {
    signature: 'new GPUTerrainTiledFlow(readonly props: GPUTerrainTiledFlowProps)',
    supportedInputs: ['readonly props: GPUTerrainTiledFlowProps'],
    defaults: [
      "props.id ?? 'terrain-tiled-flow'",
      'props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024)',
      "`'auto'` (default) retains resident transport while its fraction plane fits, and otherwise",
      "Numbering of every tile's `flowDirections`: `'esri'` (default), `'whitebox'` (the",
      "WhiteboxTools default pointer) or `'taudem'`, see `GPUTerrainFlowDirectionEncoding`.",
      '/** Resident GPU DAG sweeps, default clamps width + height to [128, 1024]. Tiled D8 uses bounded logarithmic forest passes and ignores this sweep budget. */',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      "validateCapacity( this.id, 'resident grid cell count', count, 1, TERRAIN_TILED_MAXIMUM_CELL_COUNT )",
      "validateCapacity( this.id, 'D8 grid cell count', count, 1, TERRAIN_TILED_MAXIMUM_DOMAIN_CELL_COUNT )",
      "validateRequirement( this.id, props.execution === undefined || ['auto', 'resident', 'tiled'].includes(props.execution), \"execution must be 'auto', 'resident' or 'tiled'\" )",
      "validateRequirement( this.id, props.execution !== 'tiled' || props.routing !== 'branching', \"execution 'tiled' requires routing 'd8'\" )",
      "validateRequirement( this.id, props.routing === undefined || props.routing === 'd8' || props.routing === 'branching', \"routing must be 'd8' or 'branching'\" )",
      'validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount)',
      'validateRequirement( this.id, [tile.column, tile.row, tile.width, tile.height, ...tile.owned].every(Number.isSafeInteger), `${name} column, row, width, height and owned must be integers` )',
      'validateCapacity(this.id, `${name} cell count`, tile.width * tile.height)',
      'validateRequirement( this.id, tile.width > 0 && tile.height > 0 && column >= 0 && row >= 0 && width > 0 && height > 0 && column + width <= tile.width && row + height <= tile.height, `${name} owned rectangle must be non-empty and inside the tile` )',
      'validateRequirement( this.id, rectangle[0] >= 0 && rectangle[1] >= 0 && rectangle[0] + width <= props.grid.width && rectangle[1] + height <= props.grid.height, `${name} owned rectangle must be inside the grid` )',
      "validateRequirement( this.id, rectangle[0] >= other[0] + other[2] || other[0] >= rectangle[0] + width || rectangle[1] >= other[1] + other[3] || other[1] >= rectangle[1] + height, `${name} owned rectangle must not overlap another tile's` )",
      "validateCapacity( this.id, `${name} cell count with routing 'branching'`, tile.width * tile.height, 1, TERRAIN_TILED_MAXIMUM_CELL_COUNT )",
      "validateRequirement( this.id, Boolean(tile.fractions), `${name} fractions are required with routing 'branching'` )",
      "validateRequirement(this.id, !tile.fractions, `${name} fractions need routing 'branching'`)",
      "validateRequirement( this.id, ownedCount === props.grid.width * props.grid.height, 'owned rectangles must cover every grid cell exactly once' )",
      'remains unresolved (state 2). `iterationLimitReached` is 1 only when the solver exhausts',
      'maximumIterations before its no-progress stop. A stalled cycle can report both flags 0.'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainBranchingFlowOnCPU': {
    signature:
      'export function computeTerrainBranchingFlowOnCPU( input: CPUTerrainBranchingFlowInput, options: CPUTerrainBranchingFlowOptions ): CPUTerrainBranchingFlowResult',
    supportedInputs: [
      'input: CPUTerrainBranchingFlowInput',
      'options: CPUTerrainBranchingFlowOptions'
    ],
    defaults: [
      'Otherwise the distance modes measure each edge from the grid: affine units by default, or',
      "Ground model for the default edge lengths. Default `'uniform'` (affine units; geographic",
      '/** GPU sweeps; default clamps width + height to [128, 1024]. Maximum 1024. A sweep can finalize many dependency levels. */',
      '/** Bounded successor walk after each publication. Default 32. */'
    ],
    limits: [
      'fractions.length !== count * 8',
      'values.length !== count',
      'edgeLengths && edgeLengths.length !== count * 8',
      'validity && validity.length !== count',
      '/** GPU sweeps; default clamps width + height to [128, 1024]. Maximum 1024. A sweep can finalize many dependency levels. */'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'never means an exhausted budget. maximumIterations and maximumWalkLength are ignored.'
    ],
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainBranchingFlowOperation': {
    signature: 'new CPUTerrainBranchingFlowOperation(props: TerrainBranchingFlowOperationProps)',
    supportedInputs: ['props: TerrainBranchingFlowOperationProps'],
    defaults: [
      "props.id ?? 'terrain-branching-flow'",
      'props.maximumIterations ?? 128',
      'props.maximumWalkLength ?? 32',
      'Otherwise the distance modes measure each edge from the grid: affine units by default, or',
      "Ground model for the default edge lengths. Default `'uniform'` (affine units; geographic",
      '/** GPU sweeps; default clamps width + height to [128, 1024]. Maximum 1024. A sweep can finalize many dependency levels. */',
      '/** Bounded successor walk after each publication. Default 32. */'
    ],
    limits: [
      "validateCapacity(id, 'maximumIterations', props.maximumIterations ?? 128)",
      "validateCapacity(id, 'maximumWalkLength', props.maximumWalkLength ?? 32)",
      '/** GPU sweeps; default clamps width + height to [128, 1024]. Maximum 1024. A sweep can finalize many dependency levels. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainBreachOnCPU': {
    signature:
      'export function computeTerrainBreachOnCPU( input: CPUTerrainBreachInput, options: CPUTerrainBreachOptions ): CPUTerrainBreachResult',
    supportedInputs: ['input: CPUTerrainBreachInput', 'options: CPUTerrainBreachOptions'],
    defaults: 'unknown',
    limits: [
      'input.elevation.length !== count',
      'validity && validity.length !== count',
      'burnDepth && burnDepth.length !== count',
      "validateMaximumCount( 'computeTerrainBreachOnCPU', 'pitCells.length * (maximumSteps + 1)', pitCells.length * stride, TERRAIN_OPERATION_MAXIMUM_WORD_COUNT )",
      '/** Pit-major paths, pit first, exactly pitCount * (maximumSteps + 1) capacity. */'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-breach-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainBreachOperation': {
    signature: 'new CPUTerrainBreachOperation(props: TerrainBreachOperationProps)',
    supportedInputs: ['props: TerrainBreachOperationProps'],
    defaults: ["props.id ?? 'terrain-breach'"],
    limits: [
      "validateCapacity(id, 'pitCount', props.pitCount)",
      "validateMaximumCount( id, 'pitCount * (maximumSteps + 1)', props.pitCount * (props.maximumSteps + 1), TERRAIN_OPERATION_MAXIMUM_WORD_COUNT )",
      'pitCells.length !== props.pitCount',
      '/** Pit-major paths, pit first, exactly pitCount * (maximumSteps + 1) capacity. */'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-breach-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainCatchmentSummaryOnCPU': {
    signature:
      'export function computeTerrainCatchmentSummaryOnCPU( input: CPUTerrainCatchmentSummaryInput, options: CPUTerrainCatchmentSummaryOptions ): CPUTerrainCatchmentSummaryResult',
    supportedInputs: [
      'input: CPUTerrainCatchmentSummaryInput',
      'options: CPUTerrainCatchmentSummaryOptions'
    ],
    defaults: [
      'options.maximumSteps ?? count',
      'Grid. Areas and lengths are in squared and plain affine coordinate units by default',
      'Ground model for areas, stream lengths, profile distances and gradients. Default',
      'D8 codes, one per cell, ESRI by default (1 east, clockwise powers of two; 0 terminal). A',
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      'array !== undefined && array.length !== count',
      '`converged` is 1 if no profile remains in state 2. `iterationLimitReached` is 1 when a'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainCatchmentSummaryOperation': {
    signature:
      'new CPUTerrainCatchmentSummaryOperation(props: CPUTerrainCatchmentSummaryOperationProps)',
    supportedInputs: ['props: CPUTerrainCatchmentSummaryOperationProps'],
    defaults: [
      "props.id ?? 'terrain-catchment-summary'",
      'Grid. Areas and lengths are in squared and plain affine coordinate units by default',
      'Ground model for areas, stream lengths, profile distances and gradients. Default',
      'D8 codes, one per cell, ESRI by default (1 east, clockwise powers of two; 0 terminal). A',
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      '`converged` is 1 if no profile remains in state 2. `iterationLimitReached` is 1 when a'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainDepressionHierarchyOnCPU': {
    signature:
      'export function computeTerrainDepressionHierarchyOnCPU( input: CPUTerrainDepressionHierarchyInput, options: CPUTerrainDepressionHierarchyOptions ): CPUTerrainDepressionHierarchyResult',
    supportedInputs: [
      'input: CPUTerrainDepressionHierarchyInput',
      'options: CPUTerrainDepressionHierarchyOptions'
    ],
    defaults: [
      'affine units by default (geographic degrees are not ground meters), or in square meters under',
      "/** Ground model for cell areas. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */"
    ],
    limits: [
      'elevation.length !== count',
      'outlets.length !== count',
      'validity && validity.length !== count',
      "validateMaximumCount( 'computeTerrainDepressionHierarchyOnCPU', 'nodeCapacity * stages.length * 2', options.nodeCapacity * stages.length * 2, TERRAIN_OPERATION_MAXIMUM_WORD_COUNT )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-depression-hierarchy-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainDepressionHierarchyOperation': {
    signature:
      'new CPUTerrainDepressionHierarchyOperation(props: TerrainDepressionHierarchyOperationProps)',
    supportedInputs: ['props: TerrainDepressionHierarchyOperationProps'],
    defaults: [
      "props.id ?? 'terrain-depression-hierarchy'",
      'affine units by default (geographic degrees are not ground meters), or in square meters under',
      "/** Ground model for cell areas. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */"
    ],
    limits: [
      "validateCapacity(id, 'stageCount', props.stageCount)",
      "validateMaximumCount( id, 'nodeCapacity * stageCount * 2', props.nodeCapacity * props.stageCount * 2, TERRAIN_OPERATION_MAXIMUM_WORD_COUNT )",
      'stages.length !== props.stageCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/cpu-terrain-depression-hierarchy-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainBranchingFlowOperation': {
    signature: 'new GPUTerrainBranchingFlowOperation(props: TerrainBranchingFlowOperationProps)',
    supportedInputs: ['props: TerrainBranchingFlowOperationProps'],
    defaults: [
      "props.id ?? 'terrain-branching-flow'",
      'Otherwise the distance modes measure each edge from the grid: affine units by default, or',
      "Ground model for the default edge lengths. Default `'uniform'` (affine units; geographic",
      '/** GPU sweeps; default clamps width + height to [128, 1024]. Maximum 1024. A sweep can finalize many dependency levels. */',
      '/** Bounded successor walk after each publication. Default 32. */'
    ],
    limits: [
      '/** GPU sweeps; default clamps width + height to [128, 1024]. Maximum 1024. A sweep can finalize many dependency levels. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainBreachOperation': {
    signature: 'new GPUTerrainBreachOperation(props: TerrainBreachOperationProps)',
    supportedInputs: ['props: TerrainBreachOperationProps'],
    defaults: ["props.id ?? 'terrain-breach'"],
    limits: ['/** Pit-major paths, pit first, exactly pitCount * (maximumSteps + 1) capacity. */'],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-breach-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainCatchmentSummaryOperation': {
    signature:
      'new GPUTerrainCatchmentSummaryOperation(props: TerrainCatchmentSummaryOperationProps)',
    supportedInputs: ['props: TerrainCatchmentSummaryOperationProps'],
    defaults: [
      "props.id ?? 'terrain-catchment-summary'",
      'Grid. Areas and lengths are in squared and plain affine coordinate units by default',
      'Ground model for areas, stream lengths, profile distances and gradients. Default',
      'D8 codes, one per cell, ESRI by default (1 east, clockwise powers of two; 0 terminal). A',
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      '`converged` is 1 if no profile remains in state 2. `iterationLimitReached` is 1 when a'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainDepressionHierarchyOperation': {
    signature:
      'new GPUTerrainDepressionHierarchyOperation(props: TerrainDepressionHierarchyOperationProps)',
    supportedInputs: ['props: TerrainDepressionHierarchyOperationProps'],
    defaults: [
      "props.id ?? 'terrain-depression-hierarchy'",
      'affine units by default (geographic degrees are not ground meters), or in square meters under',
      "/** Ground model for cell areas. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */"
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainFlowDistanceOnCPU': {
    signature:
      'export function computeTerrainFlowDistanceOnCPU( flowDirections: Uint32Array, options: CPUTerrainFlowDistanceOptions, streams?: Uint32Array, transitionCosts?: Float32Array ): CPUTerrainFlowDistanceResult',
    supportedInputs: [
      'flowDirections: Uint32Array',
      'options: CPUTerrainFlowDistanceOptions',
      'streams?: Uint32Array',
      'transitionCosts?: Float32Array'
    ],
    defaults: [
      "options.cellSizeMode ?? 'uniform'",
      'options.maximumSteps ?? count',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      'flowDirections.length !== count',
      'streams && streams.length !== count',
      'transitionCosts && transitionCosts.length !== count'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-flow-distance-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainFlowDistanceOperation': {
    signature: 'new CPUTerrainFlowDistanceOperation(props: TerrainFlowDistanceOperationProps)',
    supportedInputs: ['props: TerrainFlowDistanceOperationProps'],
    defaults: [
      'D8 codes, one per cell, ESRI by default (1 east, clockwise powers of two; 0 terminal;',
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      'edge: affine units by default (geographic degrees are not ground meters), or ground meters',
      "/** Ground model for the default edge length. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */",
      'Bounds each path, counted in cells including the terminal one. Default cell count; cycles',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-flow-distance-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainStreamOrderOnCPU': {
    signature:
      'export function computeTerrainStreamOrderOnCPU( flowDirections: Uint32Array, streams: Uint32Array, options: {grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding} ): CPUTerrainStreamOrderResult',
    supportedInputs: [
      'flowDirections: Uint32Array',
      'streams: Uint32Array',
      'options: {grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding}'
    ],
    defaults: [
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      '- Completes the donor queue without maximumIterations, the 1024-round pull cap, or tour scratch.',
      '- iterationCount is 0 (no GPU rounds); iterationLimitReached is always 0. converged is 0 only',
      'flowDirections.length !== count',
      'streams.length !== count'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainStreamOrderOperation': {
    signature: 'new CPUTerrainStreamOrderOperation(props: TerrainStreamOrderOperationProps)',
    supportedInputs: ['props: TerrainStreamOrderOperationProps'],
    defaults: [
      "/** Prefix for node and transient IDs. Defaults to `'terrain-stream-order'`. Compile-time. */",
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      'Maximum gated rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the pull fallback',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      "validateCapacity( id, 'maximumIterations', props.maximumIterations, 1, Number.MAX_SAFE_INTEGER )",
      'round-limit word) is unknown stream status. In particular, imported 0xfffffffe is ordinary',
      'Maximum gated rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the pull fallback',
      'Cells that remain unresolved when the iteration limit is hit read `0xffffffff`, and so do'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainWatershedsOnCPU': {
    signature:
      'export function computeTerrainWatershedsOnCPU( flowDirections: Uint32Array, options: { grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding; maximumIterations?: number; }, pourPoints?: Uint32Array ): CPUTerrainWatershedsResult',
    supportedInputs: [
      'flowDirections: Uint32Array',
      'options: { grid: GPURasterMetadata; directionEncoding?: GPUTerrainFlowDirectionEncoding; maximumIterations?: number; }',
      'pourPoints?: Uint32Array'
    ],
    defaults: [
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      '- Without maximumIterations, walks complete and iterationCount is 0 (no pointer rounds).',
      'flowDirections.length !== count',
      "validateCapacity( 'computeTerrainWatershedsOnCPU', 'maximumIterations', options.maximumIterations, 1, 1024 )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-watersheds-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainWatershedsOperation': {
    signature: 'new CPUTerrainWatershedsOperation(props: TerrainWatershedsOperationProps)',
    supportedInputs: ['props: TerrainWatershedsOperationProps'],
    defaults: [
      "/** Prefix for node and transient IDs. Defaults to `'terrain-watersheds'`. Compile-time. */",
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      'Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      "validateCapacity(id, 'maximumIterations', props.maximumIterations, 1, 1024)",
      'Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells',
      'need at most 32 doubling rounds. Compile-time.'
    ],
    optionHandling: [
      'indices only, so it has no cell-size model (`uniform`, `web-mercator` and `geographic` spacing',
      'indices of invalid cells are ignored. When several points share a cell the lowest point index'
    ],
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-watersheds-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainFlowDistanceOperation': {
    signature: 'new GPUTerrainFlowDistanceOperation(props: TerrainFlowDistanceOperationProps)',
    supportedInputs: ['props: TerrainFlowDistanceOperationProps'],
    defaults: [
      "props.id ?? 'terrain-flow-distance'",
      'D8 codes, one per cell, ESRI by default (1 east, clockwise powers of two; 0 terminal;',
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      'edge: affine units by default (geographic degrees are not ground meters), or ground meters',
      "/** Ground model for the default edge length. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */",
      'Bounds each path, counted in cells including the terminal one. Default cell count; cycles',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-flow-distance-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainStreamOrderOperation': {
    signature: 'new GPUTerrainStreamOrderOperation(props: TerrainStreamOrderOperationProps)',
    supportedInputs: ['props: TerrainStreamOrderOperationProps'],
    defaults: [
      "props.id ?? 'terrain-stream-order'",
      'props.output.status ?? {}',
      "/** Prefix for node and transient IDs. Defaults to `'terrain-stream-order'`. Compile-time. */",
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      'Maximum gated rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the pull fallback',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      'round-limit word) is unknown stream status. In particular, imported 0xfffffffe is ordinary',
      'Maximum gated rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the pull fallback',
      'Cells that remain unresolved when the iteration limit is hit read `0xffffffff`, and so do'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-stream-order-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainWatershedsOperation': {
    signature: 'new GPUTerrainWatershedsOperation(props: TerrainWatershedsOperationProps)',
    supportedInputs: ['props: TerrainWatershedsOperationProps'],
    defaults: [
      "props.id ?? 'terrain-watersheds'",
      'props.output.status ?? {}',
      "/** Prefix for node and transient IDs. Defaults to `'terrain-watersheds'`. Compile-time. */",
      "Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default",
      'Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells',
      "- `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8",
      "- `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1"
    ],
    limits: [
      "validateCapacity(id, 'maximumIterations', props.maximumIterations, 1, 1024)",
      'Maximum gated pointer-jumping rounds in `[1, 1024]`. Defaults to 32: paths of up to 2^32 cells',
      'need at most 32 doubling rounds. Compile-time.'
    ],
    optionHandling: [
      'indices only, so it has no cell-size model (`uniform`, `web-mercator` and `geographic` spacing',
      'indices of invalid cells are ignored. When several points share a cell the lowest point index'
    ],
    source: 'modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainFlowOnCPU': {
    signature:
      'export function computeTerrainFlowOnCPU( input: CPUTerrainFlowInput, options: CPUTerrainFlowOptions ): CPUTerrainFlowResult',
    supportedInputs: ['input: CPUTerrainFlowInput', 'options: CPUTerrainFlowOptions'],
    defaults: [
      "route, and non-positive/nonfinite epsilon and exponent use the contributor's defaults.",
      "options.cellSizeMode ?? 'uniform'",
      "options.flowRouting ?? 'd8'",
      "/** Prefix for node and transient IDs. Defaults to `'terrain-flow'`. Compile-time. */",
      "/** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */",
      '/** Fill depressions before routing flow. Defaults to false. Compile-time. */',
      'Defaults to 512. Compile-time. After 96 plain tiled iterations the relaxation adds directional',
      'Resolved cells become `draining`; flats with no outlet stay `flat`. Defaults to false.',
      '/** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 512, or 64 with flatContinuation. Compile-time. */',
      '/** Opt-in persistent GPU flat passes, 64 rounds per pass per execution by default. Requires resolveFlats. Inputs must remain unchanged until convergence or an explicit restart. */',
      "`output.cellClasses` are always D8. Defaults to `'d8'`. Compile-time.",
      'Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the',
      "Whether each cell contributes its weight (`'cells'`, default) or its weight times its ground"
    ],
    limits: [
      'input.elevation.values.length !== count',
      'input.elevation.validity && input.elevation.validity.length !== count',
      'input.parameters.length < GPU_TERRAIN_FLOW_PARAMETER_LENGTH',
      'input.runoff && input.runoff.length !== count',
      'Maximum gated iterations in `[1, 1024]` of the relaxation fill, which runs when `fillEpsilon` > 0',
      '/** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 512, or 64 with flatContinuation. Compile-time. */',
      'Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainFlowOperation': {
    signature: 'new CPUTerrainFlowOperation(props: TerrainFlowOperationProps)',
    supportedInputs: ['props: TerrainFlowOperationProps'],
    defaults: [
      "/** Prefix for node and transient IDs. Defaults to `'terrain-flow'`. Compile-time. */",
      "/** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */",
      '/** Fill depressions before routing flow. Defaults to false. Compile-time. */',
      'Defaults to 512. Compile-time. After 96 plain tiled iterations the relaxation adds directional',
      'Resolved cells become `draining`; flats with no outlet stay `flat`. Defaults to false.',
      '/** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 512, or 64 with flatContinuation. Compile-time. */',
      '/** Opt-in persistent GPU flat passes, 64 rounds per pass per execution by default. Requires resolveFlats. Inputs must remain unchanged until convergence or an explicit restart. */',
      "`output.cellClasses` are always D8. Defaults to `'d8'`. Compile-time.",
      'Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the',
      "Whether each cell contributes its weight (`'cells'`, default) or its weight times its ground"
    ],
    limits: [
      "validateRequirement( id, ['filledElevation', 'flowDirections', 'cellClasses', 'accumulation', 'streams'].some( port => output[port as keyof typeof output] !== undefined ), 'output requires at least one per-cell output' )",
      "validateRequirement( id, !(output.filledElevation || status?.fillConverged || status?.fillIterationCount) || Boolean(props.fillDepressions), 'output.filledElevation and fill status require fillDepressions' )",
      "validateRequirement( id, !(status?.flatsConverged || status?.flatsIterationCount) || Boolean(props.resolveFlats), 'output.status flat words require resolveFlats' )",
      "validateRequirement( id, !(status?.accumulationConverged || status?.accumulationIterationCount) || Boolean(output.accumulation || output.streams), 'output.status accumulation words require output.accumulation or output.streams' )",
      'Maximum gated iterations in `[1, 1024]` of the relaxation fill, which runs when `fillEpsilon` > 0',
      '/** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 512, or 64 with flatContinuation. Compile-time. */',
      'Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/hydrology/cpu-terrain-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainFlowOperation': {
    signature: 'new GPUTerrainFlowOperation(props: TerrainFlowOperationProps)',
    supportedInputs: ['props: TerrainFlowOperationProps'],
    defaults: [
      "props.id ?? 'terrain-flow'",
      "/** Prefix for node and transient IDs. Defaults to `'terrain-flow'`. Compile-time. */",
      "/** Cell size interpretation. Defaults to `'uniform'`. Compile-time. */",
      '/** Fill depressions before routing flow. Defaults to false. Compile-time. */',
      'Defaults to 512. Compile-time. After 96 plain tiled iterations the relaxation adds directional',
      'Resolved cells become `draining`; flats with no outlet stay `flat`. Defaults to false.',
      '/** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 512, or 64 with flatContinuation. Compile-time. */',
      '/** Opt-in persistent GPU flat passes, 64 rounds per pass per execution by default. Requires resolveFlats. Inputs must remain unchanged until convergence or an explicit restart. */',
      "`output.cellClasses` are always D8. Defaults to `'d8'`. Compile-time.",
      'Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the',
      "Whether each cell contributes its weight (`'cells'`, default) or its weight times its ground"
    ],
    limits: [
      'Maximum gated iterations in `[1, 1024]` of the relaxation fill, which runs when `fillEpsilon` > 0',
      '/** Maximum gated iterations of each flat-resolution relaxation in `[1, 1024]`. Defaults to 512, or 64 with flatContinuation. Compile-time. */',
      'Maximum gated accumulation rounds in `[1, 1024]`. Defaults to 64. Compile-time. It bounds the'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/hydrology/gpu-terrain-flow-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeProfilePeaksOnCPU': {
    signature:
      'export function computeProfilePeaksOnCPU( values: Float32Array, offsets: Uint32Array, parameters: Float32Array, options: CPUProfilePeaksOptions, validity?: Uint32Array ): CPUProfilePeaksResult',
    supportedInputs: [
      'values: Float32Array',
      'offsets: Uint32Array',
      'parameters: Float32Array',
      'options: CPUProfilePeaksOptions',
      'validity?: Uint32Array'
    ],
    defaults: [
      '- Suppression honors the round budget (default 16). Within a round, decisions are visited in',
      'options.minimumSide ?? 4',
      'options.suppressionRadius ?? Math.max(2, Math.round(options.window / 4))',
      'options.maximumSuppressionIterations ?? 16'
    ],
    limits: [
      '- GPU radius limits are dropped; window and radius may exceed 4096.',
      'validity && validity.length !== values.length',
      'parameters.length < GPU_PROFILE_PEAKS_PARAMETER_LENGTH',
      'offsets.length < 1 || offsets.some( (offset, index) => offset > values.length || (index > 0 && offset < offsets[index - 1]) )'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/gpu-terrain/terrain-features/cpu-profile-peaks-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUProfilePeaksOperation': {
    signature: 'new CPUProfilePeaksOperation(props: ProfilePeaksOperationProps)',
    supportedInputs: ['props: ProfilePeaksOperationProps'],
    defaults: ['props.output.peaks?.capacity ?? props.sampleCount'],
    limits: [
      "validateCapacity(id, 'sampleCount', props.sampleCount, 1)",
      "validateRequirement( id, Object.keys(operation.outputs).length > 0, 'output must contain at least one port' )",
      "validateCapacity( id, 'maximumSuppressionIterations', props.maximumSuppressionIterations, 0, Number.MAX_SAFE_INTEGER )",
      'values.length !== props.sampleCount'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/terrain-features/cpu-profile-peaks-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeTerrainPeakProminenceOnCPU': {
    signature:
      'export function computeTerrainPeakProminenceOnCPU( elevation: CPURasterOperationBand, grid: GPURasterMetadata ): CPUTerrainPeakProminenceResult',
    supportedInputs: ['elevation: CPURasterOperationBand', 'grid: GPURasterMetadata'],
    defaults: 'unknown',
    limits: 'unknown',
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/gpu-terrain/terrain-features/cpu-terrain-peak-prominence-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUTerrainPeakProminenceOperation': {
    signature: 'new CPUTerrainPeakProminenceOperation(props: TerrainPeakProminenceOperationProps)',
    supportedInputs: ['props: TerrainPeakProminenceOperationProps'],
    defaults: 'unknown',
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-features/cpu-terrain-peak-prominence-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUProfilePeaksOperation': {
    signature: 'new GPUProfilePeaksOperation(props: ProfilePeaksOperationProps)',
    supportedInputs: ['props: ProfilePeaksOperationProps'],
    defaults: ["props.id ?? 'profile-peaks'"],
    limits: [
      'iterationLimitReached as well as count/overflow; capacity overflow 0 does not imply completion.'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/terrain-features/gpu-profile-peaks-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainPeakProminenceOperation': {
    signature: 'new GPUTerrainPeakProminenceOperation(props: TerrainPeakProminenceOperationProps)',
    supportedInputs: ['props: TerrainPeakProminenceOperationProps'],
    defaults: ["props.id ?? 'terrain-peak-prominence'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/gpu-terrain/terrain-features/gpu-terrain-peak-prominence-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:computeSolarPositionOnCPU': {
    signature:
      'export function computeSolarPositionOnCPU( positions: Float32Array, parameters: Float32Array, options: {refraction?: boolean} = {} ): CPUSolarPositionResult',
    supportedInputs: [
      'positions: Float32Array',
      'parameters: Float32Array',
      'options: {refraction?: boolean} = {}'
    ],
    defaults: 'unknown',
    limits: [
      'positions.length % 2 !== 0',
      'parameters.length < GPU_SOLAR_POSITION_PARAMETER_LENGTH'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'orbital-angle evaluation differs increasingly far from J2000 and near zenith/nadir.'
    ],
    source: 'modules/analysis/src/gpu-terrain/terrain-illumination/cpu-solar-position-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:CPUSolarPositionOperation': {
    signature: 'new CPUSolarPositionOperation(props: SolarPositionOperationProps)',
    supportedInputs: ['props: SolarPositionOperationProps'],
    defaults: 'unknown',
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 1)",
      "validateRequirement( id, Object.keys(operation.outputs).length > 0, 'output must contain at least one port' )",
      'positions.length !== props.rowCount * 2'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/terrain-illumination/cpu-solar-position-operation.ts'
  },
  '@luma-contributors/analysis/gpu-terrain:GPUSolarPositionOperation': {
    signature: 'new GPUSolarPositionOperation(props: SolarPositionOperationProps)',
    supportedInputs: ['props: SolarPositionOperationProps'],
    defaults: ["props.id ?? 'solar-position'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/gpu-terrain/terrain-illumination/gpu-solar-position-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:computeClassBreaksOnCPU': {
    signature:
      'export function computeClassBreaksOnCPU( values: Float32Array | Uint32Array | Int32Array, parameters: Float32Array, options: CPUClassBreaksOptions, mask?: Uint32Array ): CPUClassBreaksResult',
    supportedInputs: [
      'values: Float32Array | Uint32Array | Int32Array',
      'parameters: Float32Array',
      'options: CPUClassBreaksOptions',
      'mask?: Uint32Array'
    ],
    defaults: 'unknown',
    limits: [
      'NaN tails. Maximum-gap distinctness and class assignment use numeric float32 comparison keys:',
      '- Maximum-break gaps and midpoints use float64; float32 gap overflow or rounded gap ties no',
      "validateCapacity('computeClassBreaksOnCPU', 'values.length', values.length, 0)",
      'mask && mask.length !== values.length',
      'parameters.length < getGPUClassBreaksParameterLength(maximumClassCount)'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'ignored by the CPU operation. Its DP costs O(k u^2) time and O(k u) memory for u distinct values.'
    ],
    source:
      'modules/analysis/src/dataframe-analysis/column-classification/cpu-class-breaks-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:CPUClassBreaksOperation': {
    signature: 'new CPUClassBreaksOperation(inputProps: ClassBreaksOperationProps)',
    supportedInputs: ['inputProps: ClassBreaksOperationProps'],
    defaults: [
      '/** Capacity-compatible methods; chunked default selection is resolved during execute. */',
      "props.id ?? 'class-breaks'"
    ],
    limits: [
      "validateCapacity(id, 'naturalBreaksBinCount', props.naturalBreaksBinCount, 2)",
      "input === undefined || typeof input === 'number'"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/column-classification/cpu-class-breaks-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:computeColumnQuantilesOnCPU': {
    signature:
      'export function computeColumnQuantilesOnCPU( input: CPUColumnQuantilesInput, quantileCount: number ): CPUColumnQuantilesResult',
    supportedInputs: ['input: CPUColumnQuantilesInput', 'quantileCount: number'],
    defaults: 'unknown',
    limits: [
      '- At most 16,777,216 valid rows: packed float ranks and filter products round to float32,',
      'as on GPU. Above that GPU row limit, CPU rank products retain float64 precision.',
      "- The GPU's 64-quantile and float32 row-count limits are dropped; uint32 output counts remain.",
      "validateCapacity('computeColumnQuantilesOnCPU', 'quantileCount', quantileCount, 1)",
      "validateCapacity('computeColumnQuantilesOnCPU', 'values.length', values.length, 0)",
      'mask && mask.length !== values.length',
      'parameters.length < getGPUColumnQuantilesParameterLength(quantileCount)',
      'rationalProbabilities && rationalProbabilities.length < getGPUColumnQuantilesRationalProbabilityLength(quantileCount)'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/dataframe-analysis/column-classification/cpu-column-quantiles-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:CPUColumnQuantilesOperation': {
    signature: 'new CPUColumnQuantilesOperation(inputProps: ColumnQuantilesOperationProps)',
    supportedInputs: ['inputProps: ColumnQuantilesOperationProps'],
    defaults: ["props.id ?? 'column-quantiles'"],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 1)",
      "validateCapacity(id, 'quantileCount', props.quantileCount, 1)",
      'values.length !== props.rowCount'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/column-classification/cpu-column-quantiles-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUClassBreaksOperation': {
    signature: 'new GPUClassBreaksOperation(inputProps: ClassBreaksOperationProps)',
    supportedInputs: ['inputProps: ClassBreaksOperationProps'],
    defaults: ["props.id ?? 'class-breaks'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/column-classification/gpu-class-breaks-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUColumnQuantilesOperation': {
    signature: 'new GPUColumnQuantilesOperation(inputProps: ColumnQuantilesOperationProps)',
    supportedInputs: ['inputProps: ColumnQuantilesOperationProps'],
    defaults: ["props.id ?? 'column-quantiles'"],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount, 1)",
      'validateRequirement( id, values.length === props.rowCount, `values length must equal rowCount; values length ${values.length} must equal rowCount ${props.rowCount}` )'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/column-classification/gpu-column-quantiles-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:computeInequalityOnCPU': {
    signature:
      'export function computeInequalityOnCPU(input: CPUInequalityInput): CPUInequalityResult',
    supportedInputs: ['input: CPUInequalityInput'],
    defaults: 'unknown',
    limits: [
      '- The GPU scratch-binding zone limit and 1025-knot cap are dropped; uint32 layout limits remain.',
      "validateCapacity('computeInequalityOnCPU', 'zoneCount', zoneCount, 1, 0xfffffffe)",
      "validateCapacity('computeInequalityOnCPU', 'lorenzKnotCount', knotCount, 2)",
      'rows && rows.length !== values.length',
      'parameters.length < GPU_INEQUALITY_PARAMETER_LENGTH'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/dataframe-analysis/composite-indicators/cpu-inequality-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:CPUInequalityOperation': {
    signature: 'new CPUInequalityOperation(props: InequalityOperationProps)',
    supportedInputs: ['props: InequalityOperationProps'],
    defaults: ["props.id ?? 'inequality'", 'props.lorenzKnotCount ?? 11'],
    limits: [
      "validateCapacity(id, 'zoneCount', props.zoneCount, 1, 0xfffffffe)",
      "validateCapacity(id, 'lorenzKnotCount', props.lorenzKnotCount ?? 11, 2)",
      "validateRequirement( id, Object.keys(outputs).length > 0, 'output must contain at least one output' )",
      "validateCapacity( id, 'zoneCount x lorenzKnotCount', props.zoneCount * (props.lorenzKnotCount ?? 11) )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/composite-indicators/cpu-inequality-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUInequalityOperation': {
    signature: 'new GPUInequalityOperation(props: InequalityOperationProps)',
    supportedInputs: ['props: InequalityOperationProps'],
    defaults: ["props.id ?? 'inequality'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/composite-indicators/gpu-inequality-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:computeVariogramModelFitOnCPU': {
    signature:
      'export function computeVariogramModelFitOnCPU( input: VariogramModelInput, options: CPUVariogramModelFitOptions = {} ): CPUVariogramModelFitResult',
    supportedInputs: ['input: VariogramModelInput', 'options: CPUVariogramModelFitOptions = {}'],
    defaults: [
      '- CPU defaults to 60 refinement steps, GPU to 32. Explicit rangeSteps and maximumIterations',
      'options.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS',
      "options.weighting ?? 'cressie'",
      '/** Model families; output row order follows it. Defaults to spherical, exponential, Gaussian. */',
      "/** Least-squares weighting. Defaults to `'cressie'`. */",
      '/** Log-range ladder length, 3–1024. Defaults to 200. */',
      '/** Golden-section refinements per weighting pass, 0–64. Defaults to 60 on CPU. */'
    ],
    limits: 'unknown',
    optionHandling: ['What differs from the GPU fit:'],
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:CPUVariogramModelFitOperation': {
    signature: 'new CPUVariogramModelFitOperation(props: VariogramModelFitOperationProps)',
    supportedInputs: ['props: VariogramModelFitOperationProps'],
    defaults: [
      'CPU defaults remain 200 ladder points and 60 refinements when those props are omitted.',
      "props.id ?? 'variogram-model-fit'",
      'props.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS',
      "props.weighting ?? 'cressie'",
      '/** Model families; output row order follows it. Defaults to spherical, exponential, Gaussian. */',
      "/** Least-squares weighting. Defaults to `'cressie'`. */",
      '/** Log-range ladder length in both forms, 3–1024. Defaults to 200. */',
      '/** Golden-section budget per weighting pass in both forms, 0–64. Defaults to GPU 32 / CPU 60. */'
    ],
    limits: [
      'at most, and the float64 fit has no representability limit (measured in one graph on the test',
      "validateRequirement( id, models.length >= 1 && models.length <= 5, 'models must name one to five model families' )",
      "validateRequirement( id, typeof value === 'string' && value.length > 0, `inputs.${port} must be a non-empty graph value name` )",
      "validateRequirement( id, typeof value === 'string' && value.length > 0, `outputs.${port}.value must be a non-empty graph value name` )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUVariogramModelFitOperation': {
    signature: 'new GPUVariogramModelFitOperation(props: VariogramModelFitOperationProps)',
    supportedInputs: ['props: VariogramModelFitOperationProps'],
    defaults: [
      "props.id ?? 'variogram-model-fit'",
      'props.models ?? VARIOGRAM_MODEL_FIT_DEFAULT_MODELS',
      "props.weighting ?? 'cressie'",
      '/** Model families; output row order follows it. Defaults to spherical, exponential, Gaussian. */',
      "/** Least-squares weighting. Defaults to `'cressie'`. */",
      '/** Log-range ladder length in both forms, 3–1024. Defaults to 200. */',
      '/** Golden-section budget per weighting pass in both forms, 0–64. Defaults to GPU 32 / CPU 60. */'
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:computeRipleyEnvelopeOnCPU': {
    signature:
      'export function computeRipleyEnvelopeOnCPU( input: CPURipleyEnvelopeInput, options: CPURipleyEnvelopeOptions ): CPURipleyEnvelopeResult',
    supportedInputs: ['input: CPURipleyEnvelopeInput', 'options: CPURipleyEnvelopeOptions'],
    defaults: [
      'options.rank ?? 1',
      "options.statistic ?? 'l-minus-r'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      'positions.length === 0 || positions.length % 2 !== 0',
      'mask && mask.length !== rows',
      'observedK && observedK.length < options.radiusCount',
      'parameters.length < GPU_RIPLEY_PARAMETER_LENGTH'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      '@throws If positions is not float32x2 rows, mask length differs, observedK has fewer than',
      'Only the first radiusCount observedK ordinates are read; trailing capacity is ignored.'
    ],
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:CPURipleyEnvelopeOperation': {
    signature: 'new CPURipleyEnvelopeOperation(callerProps: RipleyEnvelopeOperationProps)',
    supportedInputs: ['callerProps: RipleyEnvelopeOperationProps'],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement( id, Object.values(props.output).some(value => value !== undefined), 'output needs at least one view' )",
      'positions.length !== props.rowCount * 2'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-ripley-envelope-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:computeSpaceTimeDiggleTestOnCPU': {
    signature:
      'export function computeSpaceTimeDiggleTestOnCPU( input: CPUSpaceTimeDiggleTestInput, options: CPUSpaceTimeDiggleTestOptions ): CPUSpaceTimeDiggleTestResult',
    supportedInputs: [
      'input: CPUSpaceTimeDiggleTestInput',
      'options: CPUSpaceTimeDiggleTestOptions'
    ],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'positions.length === 0 || positions.length % 2 !== 0',
      'times.length !== rows',
      'mask && mask.length !== rows',
      'parameters.length < GPU_SPACE_TIME_K_FUNCTION_PARAMETER_LENGTH',
      'randomParameters.length < GPU_SPACE_TIME_DIGGLE_RANDOM_PARAMETER_LENGTH',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-space-time-diggle-test-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:CPUSpaceTimeDiggleTestOperation': {
    signature: 'new CPUSpaceTimeDiggleTestOperation(inputProps: SpaceTimeDiggleTestOperationProps)',
    supportedInputs: ['inputProps: SpaceTimeDiggleTestOperationProps'],
    defaults: [
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      'positions.length !== props.rowCount * 2',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-space-time-diggle-test-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:GPURipleyEnvelopeOperation': {
    signature: 'new GPURipleyEnvelopeOperation(callerProps: RipleyEnvelopeOperationProps)',
    supportedInputs: ['callerProps: RipleyEnvelopeOperationProps'],
    defaults: [
      "props.id ?? 'ripley-envelope'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', rowCount)",
      "validateRequirement( id, positions.length === rowCount, 'positions.length must match rowCount' )"
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-ripley-envelope-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUSpaceTimeDiggleTestOperation': {
    signature: 'new GPUSpaceTimeDiggleTestOperation(inputProps: SpaceTimeDiggleTestOperationProps)',
    supportedInputs: ['inputProps: SpaceTimeDiggleTestOperationProps'],
    defaults: [
      "props.id ?? 'space-time-diggle-test'",
      'Absent means undeclared, never defaulted. The library checks declarations where two contexts',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'rowCount', props.rowCount)",
      "validateRequirement( id, positions.length === props.rowCount, 'positions length must match rowCount' )",
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:GPUCoregionalizationFitOperation': {
    signature: 'new GPUCoregionalizationFitOperation(props: CoregionalizationFitOperationProps)',
    supportedInputs: ['props: CoregionalizationFitOperationProps'],
    defaults: ["props.id ?? 'coregionalization-fit'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:CPUCoregionalizationFitOperation': {
    signature: 'new CPUCoregionalizationFitOperation(props: CoregionalizationFitOperationProps)',
    supportedInputs: ['props: CoregionalizationFitOperationProps'],
    defaults: ["props.id ?? 'coregionalization-fit'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts'
  },
  '@luma-contributors/analysis/dataframe-analysis:computeCoregionalizationFitOnCPU': {
    signature:
      'export function computeCoregionalizationFitOnCPU( input: CPUCoregionalizationFitInput, options: CoregionalizationFitOptions ): CPUCoregionalizationFitResult',
    supportedInputs: [
      'input: CPUCoregionalizationFitInput',
      'options: CoregionalizationFitOptions'
    ],
    defaults: ['options.maximumIterations ?? 128'],
    limits: [
      "validateRequirement( id, pairCounts.length === distances.length, 'pairCounts.length must equal distances.length' )",
      "validateRequirement( id, semivariances.length === distances.length * triangleCount, 'semivariances.length must equal distances.length * variableCount * (variableCount + 1) / 2' )",
      "validateRequirement( id, parameters.length >= 1 + 4 * structureCount, 'parameters.length must be at least 1 + 4 * structureCount' )"
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeCostDistancePathOnCPU': {
    signature:
      'export function computeCostDistancePathOnCPU( backLinks: Uint32Array, target: Uint32Array, grid: GPURasterMetadata, capacity: number ): CPUCostDistancePathResult',
    supportedInputs: [
      'backLinks: Uint32Array',
      'target: Uint32Array',
      'grid: GPURasterMetadata',
      'capacity: number'
    ],
    defaults: 'unknown',
    limits: ['backLinks.length !== cellCount', 'target.length !== 1'],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/raster-analysis/cost-distance/cpu-cost-distance-path-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPUCostDistancePathOperation': {
    signature: 'new CPUCostDistancePathOperation(props: CostDistancePathOperationProps)',
    supportedInputs: ['props: CostDistancePathOperationProps'],
    defaults: ["props.id ?? 'cost-distance-path'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/raster-analysis/cost-distance/cpu-cost-distance-path-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPUCostDistancePathOperation': {
    signature: 'new GPUCostDistancePathOperation(props: CostDistancePathOperationProps)',
    supportedInputs: ['props: CostDistancePathOperationProps'],
    defaults: ["props.id ?? 'cost-distance-path'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance-path-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterWalkingCostOnCPU': {
    signature:
      'export function computeRasterWalkingCostOnCPU( input: CPURasterWalkingCostInput, options: CPURasterWalkingCostOptions ): CPURasterWalkingCostResult',
    supportedInputs: ['input: CPURasterWalkingCostInput', 'options: CPURasterWalkingCostOptions'],
    defaults: [
      "options.cellSizeMode ?? 'uniform'",
      "How affine units become ground meters for step lengths and slopes. Default `'uniform'` (affine",
      '/** Movement model. Default Tobler. The kind is compile-time; the numbers are packed into `parameters`. */',
      '/** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */'
    ],
    limits: [
      '- Explicit maximumIterations permits at most maximumIterations + 1 synchronous rounds, just',
      'input.sources.length < length',
      'input.parameters.length < GPU_RASTER_WALKING_COST_PARAMETER_LENGTH',
      '/** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */',
      '`iterationLimitReached` (1 when the limit stopped an unconverged run) and `iterationCount`',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: [
      'grid.width * grid.height sources rows are read; surplus source rows are ignored.',
      'What differs from the GPU contributor:',
      'weight and GRASS coefficients are read from it (the numbers in `model` are ignored; `model.kind`'
    ],
    source:
      'modules/analysis/src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterWalkingCostOperation': {
    signature: 'new CPURasterWalkingCostOperation(props: RasterWalkingCostOperationProps)',
    supportedInputs: ['props: RasterWalkingCostOperationProps'],
    defaults: [
      "props.id ?? 'raster-walking-cost'",
      "How affine units become ground meters for step lengths and slopes. Default `'uniform'` (affine",
      '/** Movement model. Default Tobler. The kind is compile-time; the numbers are packed into `parameters`. */',
      '/** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */'
    ],
    limits: [
      '/** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */',
      '`iterationLimitReached` (1 when the limit stopped an unconverged run) and `iterationCount`',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: [
      'weight and GRASS coefficients are read from it (the numbers in `model` are ignored; `model.kind`'
    ],
    source:
      'modules/analysis/src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterWalkingCostOperation': {
    signature: 'new GPURasterWalkingCostOperation(props: RasterWalkingCostOperationProps)',
    supportedInputs: ['props: RasterWalkingCostOperationProps'],
    defaults: [
      "props.id ?? 'raster-walking-cost'",
      "How affine units become ground meters for step lengths and slopes. Default `'uniform'` (affine",
      '/** Movement model. Default Tobler. The kind is compile-time; the numbers are packed into `parameters`. */',
      '/** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */'
    ],
    limits: [
      '/** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */',
      '`iterationLimitReached` (1 when the limit stopped an unconverged run) and `iterationCount`',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: [
      'weight and GRASS coefficients are read from it (the numbers in `model` are ignored; `model.kind`'
    ],
    source:
      'modules/analysis/src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeCostDistanceOnCPU': {
    signature:
      'export function computeCostDistanceOnCPU( input: CPUCostDistanceInput, options: CPUCostDistanceOptions ): CPUCostDistanceResult',
    supportedInputs: ['input: CPUCostDistanceInput', 'options: CPUCostDistanceOptions'],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'cost-distance'`. */",
      "/** Cell size interpretation. Compile-time. Defaults to `'uniform'`. */",
      '/** Optional initial cost per source row, default 0. Negative or non-finite costs are ignored. Per-frame contents. */',
      'Compile-time. Defaults to 512, or to 64 with `continuation`, where it is the budget of one',
      'directional tile sweeps (right, left, down, up). Compile-time. Defaults to 96. Plain tiled',
      'Defaults to `maximumIterations`. Used only when `output.backLinks` is requested. The tie relaxation uses'
    ],
    limits: [
      'At most `maximumIterations` new rounds per execution; cumulative with continuation. Compare final'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'are ignored. Negative packed cell sizes are outside the documented GPU parameter layout.',
      '/** Buffer-backed cost distance. Stateful continuation is not supported by the mixed executor. */',
      'are per-frame. Out-of-range or impassable cells are ignored.',
      '/** Optional initial cost per source row, default 0. Negative or non-finite costs are ignored. Per-frame contents. */',
      '/** Optional one-row active source count; rows at or after it are ignored. Per-frame contents. */',
      'choosing) and submits again until it is 1 or its own budget ends. Not supported with'
    ],
    source: 'modules/analysis/src/raster-analysis/cost-distance/cpu-cost-distance-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPUCostDistanceOperation': {
    signature: 'new CPUCostDistanceOperation(props: CostDistanceOperationProps)',
    supportedInputs: ['props: CostDistanceOperationProps'],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'cost-distance'`. */",
      "/** Cell size interpretation. Compile-time. Defaults to `'uniform'`. */",
      '/** Optional initial cost per source row, default 0. Negative or non-finite costs are ignored. Per-frame contents. */',
      'Compile-time. Defaults to 512, or to 64 with `continuation`, where it is the budget of one',
      'directional tile sweeps (right, left, down, up). Compile-time. Defaults to 96. Plain tiled',
      'Defaults to `maximumIterations`. Used only when `output.backLinks` is requested. The tie relaxation uses'
    ],
    limits: [
      "validateRequirement( id, props.sources !== undefined || props.sourceMask !== undefined, 'sources or sourceMask is required' )",
      "validateRequirement( id, !(props.sourceCosts || props.sourceCount) || props.sources !== undefined, 'sourceCosts and sourceCount require sources' )",
      "validateRequirement( id, Boolean(props.costBreaks) === Boolean(props.output.bands || props.output.bandCounts), 'costBreaks is required exactly with bands or bandCounts' )",
      "validateCapacity(id, 'costBreakCount', props.costBreakCount!)",
      'props.output.bandCounts !== undefined && input.costBreaks?.length !== props.costBreakCount',
      'At most `maximumIterations` new rounds per execution; cumulative with continuation. Compare final'
    ],
    optionHandling: [
      '/** Buffer-backed cost distance. Stateful continuation is not supported by the mixed executor. */',
      'are per-frame. Out-of-range or impassable cells are ignored.',
      '/** Optional initial cost per source row, default 0. Negative or non-finite costs are ignored. Per-frame contents. */',
      '/** Optional one-row active source count; rows at or after it are ignored. Per-frame contents. */',
      'choosing) and submits again until it is 1 or its own budget ends. Not supported with'
    ],
    source: 'modules/analysis/src/raster-analysis/cost-distance/cpu-cost-distance-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPUCostDistanceOperation': {
    signature: 'new GPUCostDistanceOperation(props: CostDistanceOperationProps)',
    supportedInputs: ['props: CostDistanceOperationProps'],
    defaults: [
      "props.id ?? 'cost-distance'",
      'props.output.status ?? {}',
      "/** Prefix for generated node and transient IDs. Defaults to `'cost-distance'`. */",
      "/** Cell size interpretation. Compile-time. Defaults to `'uniform'`. */",
      '/** Optional initial cost per source row, default 0. Negative or non-finite costs are ignored. Per-frame contents. */',
      'Compile-time. Defaults to 512, or to 64 with `continuation`, where it is the budget of one',
      'directional tile sweeps (right, left, down, up). Compile-time. Defaults to 96. Plain tiled',
      'Defaults to `maximumIterations`. Used only when `output.backLinks` is requested. The tie relaxation uses'
    ],
    limits: [
      'At most `maximumIterations` new rounds per execution; cumulative with continuation. Compare final'
    ],
    optionHandling: [
      '/** Buffer-backed cost distance. Stateful continuation is not supported by the mixed executor. */',
      'are per-frame. Out-of-range or impassable cells are ignored.',
      '/** Optional initial cost per source row, default 0. Negative or non-finite costs are ignored. Per-frame contents. */',
      '/** Optional one-row active source count; rows at or after it are ignored. Per-frame contents. */',
      'choosing) and submits again until it is 1 or its own budget ends. Not supported with'
    ],
    source: 'modules/analysis/src/raster-analysis/cost-distance/gpu-cost-distance-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeStreamlinesOnCPU': {
    signature:
      'export function computeStreamlinesOnCPU( input: CPUStreamlinesInput, options: CPUStreamlinesOptions ): CPUStreamlinesResult',
    supportedInputs: ['input: CPUStreamlinesInput', 'options: CPUStreamlinesOptions'],
    defaults: [
      '- Pruning honors maximumIterations (default 32); unresolved seeds are dropped and reported by',
      'options.maximumIterations ?? 32',
      "/** Prefix for generated node and transient IDs. Defaults to `'streamlines'`. */",
      '/** GPU-gated pruning rounds. Compile-time. Defaults to 32. */'
    ],
    limits: [
      '- Pruning honors maximumIterations (default 32); unresolved seeds are dropped and reported by',
      '- The 65535 seed limit remains because priorities reserve 16 bits for seed index. GPU dimension',
      'input.velocities.length !== options.fieldWidth * options.fieldHeight * 2',
      'input.parameters.length < GPU_STREAMLINES_PARAMETER_LENGTH',
      'input.uint32Parameters.length < GPU_STREAMLINES_UINT32_PARAMETER_LENGTH',
      "validateRequirement( 'computeStreamlinesOnCPU', Number.isFinite(reciprocal) && reciprocal > 0, `${name} reciprocal must be positive and finite in float32; received ${reciprocal}` )",
      'status.iterationLimitReached == 0 to establish completeness. requiredCount == count alone'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/raster-analysis/flow-texture/cpu-streamlines-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPUStreamlinesOperation': {
    signature: 'new CPUStreamlinesOperation(props: StreamlinesOperationProps)',
    supportedInputs: ['props: StreamlinesOperationProps'],
    defaults: [
      "/** Prefix for generated node and transient IDs. Defaults to `'streamlines'`. */",
      '/** GPU-gated pruning rounds. Compile-time. Defaults to 32. */'
    ],
    limits: [
      "validateCapacity(id, 'maximumIterations', props.maximumIterations)",
      'status.iterationLimitReached == 0 to establish completeness. requiredCount == count alone'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/raster-analysis/flow-texture/cpu-streamlines-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPUStreamlinesOperation': {
    signature: 'new GPUStreamlinesOperation(props: StreamlinesOperationProps)',
    supportedInputs: ['props: StreamlinesOperationProps'],
    defaults: [
      "props.id ?? 'streamlines'",
      "/** Prefix for generated node and transient IDs. Defaults to `'streamlines'`. */",
      '/** GPU-gated pruning rounds. Compile-time. Defaults to 32. */'
    ],
    limits: [
      'status.iterationLimitReached == 0 to establish completeness. requiredCount == count alone'
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/raster-analysis/flow-texture/gpu-streamlines-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPUIsolines': {
    signature: 'new GPUIsolines(props: GPUIsolinesProps)',
    supportedInputs: ['props: GPUIsolinesProps'],
    defaults: [
      'float32; an origin near a large sample index therefore retains sub-cell coordinates. Absolute vertices (the default origin `[0, 0]`) are quantized to the float32 ulp,',
      '`grid.affine` is checked for finite values at construction but is not used for placement. A north-up grid (row 0 at the top, the default of',
      "props.id ?? 'isolines'",
      'props.tile?.ownedCells ?? [0, 0, width - 1, height - 1]',
      "/** Prefix for generated node and transient IDs. Defaults to `'isolines'`. */"
    ],
    limits: [
      'props.noDataValue !== undefined && !Number.isFinite(props.noDataValue)',
      '!output?.segments',
      'capacity < 1',
      '!output.status.count',
      '!output.status?.overflow',
      'polylines',
      '!polylines[name]',
      '!output.status.polylineOverflow'
    ],
    optionHandling: [
      'rule was the mean of the four corners, which differs when the corners are far from symmetric.)'
    ],
    source: 'modules/analysis/src/raster-analysis/isolines/gpu-isolines.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeIsolinesOnCPU': {
    signature:
      'export function computeIsolinesOnCPU( input: CPUIsolinesInput, options: CPUIsolinesOptions ): CPUIsolinesResult',
    supportedInputs: ['input: CPUIsolinesInput', 'options: CPUIsolinesOptions'],
    defaults: [
      'options.vertexCapacity ?? 0',
      "/** Prefix for generated node and transient IDs. Defaults to `'isolines'`. */"
    ],
    limits: [
      'opposing maximum-float32 samples at level zero cross at the midpoint (x = 1 on a unit',
      'input.values.length < width * height',
      'input.mask && input.mask.length < width * height',
      'input.levels.length < 1',
      'input.parameters.length < GPU_ISOLINES_PARAMETER_LENGTH'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source: 'modules/analysis/src/raster-analysis/isolines/cpu-isolines-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPUIsolinesOperation': {
    signature: 'new CPUIsolinesOperation(props: IsolinesOperationProps)',
    supportedInputs: ['props: IsolinesOperationProps'],
    defaults: ["/** Prefix for generated node and transient IDs. Defaults to `'isolines'`. */"],
    limits: [
      "validateRequirement( id, !props.output.polylines || props.output.status.polylineOverflow !== undefined, 'polylines require status.polylineOverflow' )"
    ],
    optionHandling: 'unknown',
    source: 'modules/analysis/src/raster-analysis/isolines/cpu-isolines-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPUIsolinesOperation': {
    signature: 'new GPUIsolinesOperation(props: IsolinesOperationProps)',
    supportedInputs: ['props: IsolinesOperationProps'],
    defaults: [
      "props.id ?? 'isolines'",
      "/** Prefix for generated node and transient IDs. Defaults to `'isolines'`. */"
    ],
    limits: 'unknown',
    optionHandling: 'unknown',
    source: 'modules/analysis/src/raster-analysis/isolines/gpu-isolines-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterFractionalZonalDistributionOnCPU': {
    signature:
      'export function computeRasterFractionalZonalDistributionOnCPU( input: CPURasterFractionalZonalDistributionInput, options: CPURasterFractionalZonalDistributionOptions ): CPURasterFractionalZonalDistributionResult',
    supportedInputs: [
      'input: CPURasterFractionalZonalDistributionInput',
      'options: CPURasterFractionalZonalDistributionOptions'
    ],
    defaults: [
      "options.cellSizeMode ?? 'uniform'",
      '/** Default uniform; non-uniform ground areas require an axis-aligned grid and no cellAreas. */',
      '/** Independent nonnegative per-cell reliability weights; default one. */',
      '/** Positive per-cell areas in caller-declared units (area outputs retain those units); default planar affine determinant. */',
      "What the affine's units are, for the default cell area. Default `'uniform'`: planar coordinate",
      '/** Exact distinct-value interpolation (default), or the left-continuous inverse weighted CDF. */',
      '/** Compile-time candidate lookup; bounds is the scalable default. Brute force retains its 4M pair cap. */',
      'Defaults to min(zones * cells, max(4M, output capacity * 4)). Scratch is tiled, not budget-sized.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'coverage.zoneIds length', capacity)",
      "validateCapacity(id, 'probabilities length', probabilities.length)",
      'coverage.cellIds.length !== capacity || coverage.fractions.length !== capacity',
      'coverage.zoneValidity.length !== options.zoneCount',
      'coverage[name].length < 1',
      'coverage.candidateOverflow && coverage.candidateOverflow.length < 1',
      'input[name] && input[name].length !== cellCount',
      'input.parameters && input.parameters.length < GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH - 1',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterFractionalZonalDistributionOperation': {
    signature:
      'new CPURasterFractionalZonalDistributionOperation(props: RasterFractionalZonalDistributionOperationProps)',
    supportedInputs: ['props: RasterFractionalZonalDistributionOperationProps'],
    defaults: [
      "props.id ?? 'raster-fractional-zonal-distribution'",
      '/** Default uniform; non-uniform ground areas require an axis-aligned grid and no cellAreas. */',
      '/** Independent nonnegative per-cell reliability weights; default one. */',
      '/** Positive per-cell areas in caller-declared units (area outputs retain those units); default planar affine determinant. */',
      "What the affine's units are, for the default cell area. Default `'uniform'`: planar coordinate",
      '/** Exact distinct-value interpolation (default), or the left-continuous inverse weighted CDF. */',
      '/** Compile-time candidate lookup; bounds is the scalable default. Brute force retains its 4M pair cap. */',
      'Defaults to min(zones * cells, max(4M, output capacity * 4)). Scratch is tiled, not budget-sized.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      "validateCapacity(id, 'zoneCount', props.zoneCount)",
      "validateCapacity(id, 'probabilityCount', props.probabilityCount)",
      "validateCapacity( id, 'zoneCount times probabilityCount', props.zoneCount * props.probabilityCount, 0 )",
      'probabilities.length !== props.probabilityCount',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterFractionalZonalDistributionOperation': {
    signature:
      'new GPURasterFractionalZonalDistributionOperation(props: RasterFractionalZonalDistributionOperationProps)',
    supportedInputs: ['props: RasterFractionalZonalDistributionOperationProps'],
    defaults: [
      "props.id ?? 'raster-fractional-zonal-distribution'",
      '/** Default uniform; non-uniform ground areas require an axis-aligned grid and no cellAreas. */',
      '/** Independent nonnegative per-cell reliability weights; default one. */',
      '/** Positive per-cell areas in caller-declared units (area outputs retain those units); default planar affine determinant. */',
      "What the affine's units are, for the default cell area. Default `'uniform'`: planar coordinate",
      '/** Exact distinct-value interpolation (default), or the left-continuous inverse weighted CDF. */',
      '/** Compile-time candidate lookup; bounds is the scalable default. Brute force retains its 4M pair cap. */',
      'Defaults to min(zones * cells, max(4M, output capacity * 4)). Scratch is tiled, not budget-sized.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-distribution-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterFractionalZonalStatisticsOnCPU': {
    signature:
      'export function computeRasterFractionalZonalStatisticsOnCPU( input: CPURasterFractionalZonalStatisticsInput, options: CPURasterFractionalZonalStatisticsOptions ): CPURasterFractionalZonalStatisticsResult',
    supportedInputs: [
      'input: CPURasterFractionalZonalStatisticsInput',
      'options: CPURasterFractionalZonalStatisticsOptions'
    ],
    defaults: [
      "options.cellSizeMode ?? 'uniform'",
      'Area model: uniform (default) gives coordinate units squared; geographic and web-mercator',
      '/** Compile-time candidate lookup; bounds is the scalable default. Brute force retains its 4M pair cap. */',
      'Defaults to min(zones * cells, max(4M, output capacity * 4)). Scratch is tiled, not budget-sized.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      'input.cellAreas && input.cellAreas.length !== options.grid.width * options.grid.height',
      'coverage.candidateOverflow && coverage.candidateOverflow.length < 1',
      'coverage.cellIds.length < coverage.zoneIds.length || coverage.fractions.length < coverage.zoneIds.length',
      'coverage.zoneValidity.length < options.zoneCount',
      'coverage[name].length < 1',
      'parameters && parameters.length < GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH',
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: [
      "Aggregates all zones in one tuple pass, preserving the contributor's count clamp, ignored",
      'and scalar coverage statuses beyond their first row; these tails are ignored as on the GPU.',
      'What differs from the GPU contributor:'
    ],
    source:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterFractionalZonalStatisticsOperation': {
    signature:
      'new CPURasterFractionalZonalStatisticsOperation(props: RasterFractionalZonalStatisticsOperationProps)',
    supportedInputs: ['props: RasterFractionalZonalStatisticsOperationProps'],
    defaults: [
      "props.id ?? 'fractional-zonal-statistics'",
      'Area model: uniform (default) gives coordinate units squared; geographic and web-mercator',
      '/** Compile-time candidate lookup; bounds is the scalable default. Brute force retains its 4M pair cap. */',
      'Defaults to min(zones * cells, max(4M, output capacity * 4)). Scratch is tiled, not budget-sized.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-statistics-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterFractionalZonalStatisticsOperation': {
    signature:
      'new GPURasterFractionalZonalStatisticsOperation(props: RasterFractionalZonalStatisticsOperationProps)',
    supportedInputs: ['props: RasterFractionalZonalStatisticsOperationProps'],
    defaults: [
      "props.id ?? 'fractional-zonal-statistics'",
      'Area model: uniform (default) gives coordinate units squared; geographic and web-mercator',
      '/** Compile-time candidate lookup; bounds is the scalable default. Brute force retains its 4M pair cap. */',
      'Defaults to min(zones * cells, max(4M, output capacity * 4)). Scratch is tiled, not budget-sized.',
      '/** One step of a calendar-regular axis: `interval` (a positive integer, default 1) calendar units. */'
    ],
    limits: [
      '`maximumGap` is also given in); `bigint` keeps widths beyond 2^53 ticks exact, and fractional'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-statistics-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:addRasterRegionGraphTiles': {
    signature:
      'export function addRasterRegionGraphTiles<Parameters>( graph: GPUCommandGraph<Parameters>, props: GPURasterRegionGraphTilesProps ): GPURasterRegionGraph & {readonly estimatedRecipeCommandNodeCount: number}',
    supportedInputs: [
      'graph: GPUCommandGraph<Parameters>',
      'props: GPURasterRegionGraphTilesProps'
    ],
    defaults: [
      "props.id ?? 'raster-region-tiles'",
      '/** Fixed single linkage (default) or one globally closest mean-updating merge per round. */'
    ],
    limits: [
      "validateRequirement( id, certificates.length === 0 || Boolean(props.output.status.complete), 'labelsComplete requires output.status.complete to retain source-label completeness' )",
      'validateCommandNodeBudget(id, estimatedRecipeCommandNodeCount)',
      "validateRequirement( id, view.length >= length && (!name.startsWith('output.status.') || view.length === 1), `${name} must hold ${length} entries; received ${view.length}` )",
      "validateCapacity( id, 'assembled labels and values byteLength', count * 4, 1, Math.min(graph.device.limits.maxStorageBufferBindingSize, graph.device.limits.maxBufferSize) )",
      'validateRequirement( id, assemblyStorageBindingCount <= graph.device.limits.maxStorageBuffersPerShaderStage, `tiles[${index}] assembly needs ${assemblyStorageBindingCount} storage buffers per shader stage; received limit ${graph.device.limits.maxStorageBuffersPerShaderStage}` )',
      'validateRequirement( id, Number.isInteger(column) && Number.isInteger(row) && column >= 0 && row >= 0, `tiles[${index}].origin must be nonnegative integers` )',
      'validateRequirement( id, Number.isInteger(tile.grid.width) && Number.isInteger(tile.grid.height) && tile.grid.width > 0 && tile.grid.height > 0, `tiles[${index}].grid width and height must be positive integers` )',
      'validateRequirement( id, column + tile.grid.width <= props.grid.width && row + tile.grid.height <= props.grid.height, `tiles[${index}] must lie inside grid` )',
      'validateRequirement( id, tile.labels.length === tile.grid.width * tile.grid.height && tile.values.length === tile.labels.length, `tiles[${index}] labels and values must hold exactly grid.width times grid.height entries` )',
      'validateRequirement( id, column >= previous.origin[0] + previous.grid.width || previous.origin[0] >= column + tile.grid.width || row >= previous.origin[1] + previous.grid.height || previous.origin[1] >= row + tile.grid.height, `tiles[${index}] must not overlap an earlier tile` )',
      "validateRequirement(id, area === count, 'tiles must cover grid completely')",
      'validateRequirement( id, certificate.length === 1, `labelsComplete must hold exactly one uint32 word; received ${certificate.length}` )',
      '/** Dense propagation/adaptive rounds, or Boruvka phases (at most ceil(log2(regionCount))). */'
    ],
    optionHandling: [
      'estimatedRecipeCommandNodeCount = inner plan.estimatedCommandNodeCount + tiles.length assembly commands + (certificateCount > 0 ? 1 + certificateCount : 0) certificate commands. The returned readonly recipe estimate includes assembly and certificate work; the inner contributor estimate covers only region-graph work.'
    ],
    source: 'modules/analysis/src/raster-analysis/remote-sensing/raster-region-tiles.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterRegionGraphOnCPU': {
    signature:
      'export function computeRasterRegionGraphOnCPU(labels: Uint32Array, values: Float32Array, options: CPURasterRegionGraphOptions | RasterRegionGraphOperationProps): CPURasterRegionGraphResult',
    supportedInputs: [
      'labels: Uint32Array',
      'values: Float32Array',
      'options: CPURasterRegionGraphOptions | RasterRegionGraphOperationProps'
    ],
    defaults: ["options.mergeMode ?? 'fixed'", "options.searchStructure ?? 'auto'"],
    limits: [
      '- The 256-region and 256-round GPU limits are dropped; the uint32 adjacency cell bound remains.',
      "validateRequirement( 'computeRasterRegionGraphOnCPU', options.labelsComplete === undefined || (options.labelsComplete instanceof Uint32Array && options.labelsComplete.length === 1), 'labelsComplete must be a Uint32Array holding exactly one word; pass CPU evidence instead of a graph value name' )",
      'labels.length < cellCount',
      'values.length < cellCount'
    ],
    optionHandling: [
      'What differs from the GPU contributor:',
      'CPURasterRegionGraphOptions accepts labelsComplete as exactly one Uint32Array word. The RasterRegionGraphOperationProps compatibility branch remains supported when labelsComplete is omitted; a graph-value name is not CPU certificate evidence.'
    ],
    source:
      'modules/analysis/src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterRegionGraphOperation': {
    signature: 'new CPURasterRegionGraphOperation(inputProps: RasterRegionGraphOperationProps)',
    supportedInputs: ['inputProps: RasterRegionGraphOperationProps'],
    defaults: ["props.id ?? 'raster-region-graph'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterRegionGraphOperation': {
    signature: 'new GPURasterRegionGraphOperation(inputProps: RasterRegionGraphOperationProps)',
    supportedInputs: ['inputProps: RasterRegionGraphOperationProps'],
    defaults: ["props.id ?? 'raster-region-graph'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterGeometricMedianCompositeOnCPU': {
    signature:
      'export function computeRasterGeometricMedianCompositeOnCPU( values: Float32Array, parameters: Float32Array, options: CPURasterGeometricMedianCompositeOptions, mask?: Uint32Array ): CPURasterGeometricMedianCompositeResult',
    supportedInputs: [
      'values: Float32Array',
      'parameters: Float32Array',
      'options: CPURasterGeometricMedianCompositeOptions',
      'mask?: Uint32Array'
    ],
    defaults: 'unknown',
    limits: [
      '- GPU limits of 256 slices, 16 bands and 64 iterations are dropped; uint32 shape limits remain.',
      "validateCapacity( 'computeRasterGeometricMedianCompositeOnCPU', 'iterationCount', options.iterationCount )",
      'parameters.length < GPU_RASTER_GEOMETRIC_MEDIAN_COMPOSITE_PARAMETER_LENGTH'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterGeometricMedianCompositeOperation': {
    signature:
      'new CPURasterGeometricMedianCompositeOperation(props: RasterGeometricMedianCompositeOperationProps)',
    supportedInputs: ['props: RasterGeometricMedianCompositeOperationProps'],
    defaults: ["props.id ?? 'raster-geometric-median-composite'"],
    limits: ["validateCapacity(id, 'iterationCount', props.iterationCount)"],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-time-series/cpu-raster-geometric-median-composite-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterMedoidCompositeOnCPU': {
    signature:
      'export function computeRasterMedoidCompositeOnCPU( values: Float32Array, dimensions: CPUCompositeDimensions, mask?: Uint32Array ): CPURasterMedoidCompositeResult',
    supportedInputs: [
      'values: Float32Array',
      'dimensions: CPUCompositeDimensions',
      'mask?: Uint32Array'
    ],
    defaults: 'unknown',
    limits: [
      '- GPU limits of 64 slices and 16 bands are dropped; uint32 shape/indexing limits remain.'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterMedoidCompositeOperation': {
    signature: 'new CPURasterMedoidCompositeOperation(props: RasterMedoidCompositeOperationProps)',
    supportedInputs: ['props: RasterMedoidCompositeOperationProps'],
    defaults: ["props.id ?? 'raster-medoid-composite'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-time-series/cpu-raster-medoid-composite-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterGeometricMedianCompositeOperation': {
    signature:
      'new GPURasterGeometricMedianCompositeOperation(props: RasterGeometricMedianCompositeOperationProps)',
    supportedInputs: ['props: RasterGeometricMedianCompositeOperationProps'],
    defaults: ["props.id ?? 'raster-geometric-median-composite'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterMedoidCompositeOperation': {
    signature: 'new GPURasterMedoidCompositeOperation(props: RasterMedoidCompositeOperationProps)',
    supportedInputs: ['props: RasterMedoidCompositeOperationProps'],
    defaults: ["props.id ?? 'raster-medoid-composite'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-time-series/gpu-raster-medoid-composite-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterWatershedSegmentationOnCPU': {
    signature:
      'export function computeRasterWatershedSegmentationOnCPU( input: CPURasterOperationBand, markers: Uint32Array, options: CPURasterWatershedSegmentationOptions ): CPURasterWatershedSegmentationResult',
    supportedInputs: [
      'input: CPURasterOperationBand',
      'markers: Uint32Array',
      'options: CPURasterWatershedSegmentationOptions'
    ],
    defaults: 'unknown',
    limits: [
      'unresolved (0xfffffffe), with `converged=0` and `iterationLimitReached=1`.',
      "- GPU FIFO cell and forest round limits are dropped. The forest's uint32 edge-ID cell bound",
      'markers.length < count'
    ],
    optionHandling: ['What differs from the GPU contributor:'],
    source:
      'modules/analysis/src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterWatershedSegmentationOperation': {
    signature:
      'new CPURasterWatershedSegmentationOperation(props: RasterWatershedSegmentationOperationProps)',
    supportedInputs: ['props: RasterWatershedSegmentationOperationProps'],
    defaults: ["props.id ?? 'raster-watershed-segmentation'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterWatershedSegmentationOperation': {
    signature:
      'new GPURasterWatershedSegmentationOperation(props: RasterWatershedSegmentationOperationProps)',
    supportedInputs: ['props: RasterWatershedSegmentationOperationProps'],
    defaults: ["props.id ?? 'raster-watershed-segmentation'"],
    limits: 'unknown',
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-watershed/gpu-raster-watershed-segmentation-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterReconstructionOperation': {
    signature: 'new GPURasterReconstructionOperation(props: RasterReconstructionOperationProps)',
    supportedInputs: ['props: RasterReconstructionOperationProps'],
    defaults: [
      "props.id ?? 'raster-reconstruction'",
      'neighbours), `4` the cross (edge neighbours only). Defaults to `8`, the default footprint of',
      'default was `4` before 2026-11-10.)',
      '/** Rounds before the run stops unconverged. Default 64; at most 26,209 (the command node budget). */'
    ],
    limits: [
      '/** Rounds before the run stops unconverged. Default 64; at most 26,209 (the command node budget). */',
      'beyond `maximumIterations` rounds (so the count can be maximumIterations + 1).',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-morphology/gpu-raster-reconstruction-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterReconstructionOperation': {
    signature: 'new CPURasterReconstructionOperation(props: RasterReconstructionOperationProps)',
    supportedInputs: ['props: RasterReconstructionOperationProps'],
    defaults: [
      "props.id ?? 'raster-reconstruction'",
      'neighbours), `4` the cross (edge neighbours only). Defaults to `8`, the default footprint of',
      'default was `4` before 2026-11-10.)',
      '/** Rounds before the run stops unconverged. Default 64; at most 26,209 (the command node budget). */'
    ],
    limits: [
      "validateRequirement( id, props.method === undefined || props.method === 'dilation' || props.method === 'erosion', 'method must be dilation or erosion' )",
      '/** Rounds before the run stops unconverged. Default 64; at most 26,209 (the command node budget). */',
      'beyond `maximumIterations` rounds (so the count can be maximumIterations + 1).',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterReconstructionOnCPU': {
    signature:
      'export function computeRasterReconstructionOnCPU( marker: Float32Array, mask: Float32Array, options: CPURasterReconstructionOptions ): CPURasterReconstructionResult',
    supportedInputs: [
      'marker: Float32Array',
      'mask: Float32Array',
      'options: CPURasterReconstructionOptions'
    ],
    defaults: [
      '- The GPU command-node budget is not used. maximumIterations (default 64) still bounds',
      'options.maximumIterations ?? 64',
      'options.connectivity ?? 8',
      'neighbours), `4` the cross (edge neighbours only). Defaults to `8`, the default footprint of',
      'default was `4` before 2026-11-10.)',
      '/** Rounds before the run stops unconverged. Default 64; at most 26,209 (the command node budget). */'
    ],
    limits: [
      '- The GPU command-node budget is not used. maximumIterations (default 64) still bounds',
      "validateRequirement( id, options.method === undefined || options.method === 'dilation' || options.method === 'erosion', 'method must be dilation or erosion' )",
      'marker.length < length',
      'mask.length < length',
      '/** Rounds before the run stops unconverged. Default 64; at most 26,209 (the command node budget). */',
      'beyond `maximumIterations` rounds (so the count can be maximumIterations + 1).',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: [
      'Only grid.width * grid.height marker and mask rows are read; surplus rows are ignored.',
      'What differs from the GPU contributor:'
    ],
    source:
      'modules/analysis/src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterFillHolesOperation': {
    signature: 'new GPURasterFillHolesOperation(props: RasterFillHolesOperationProps)',
    supportedInputs: ['props: RasterFillHolesOperationProps'],
    defaults: [
      "props.id ?? 'raster-fill-holes'",
      'Connectivity of the background that is flooded from the boundary. Defaults to `4`, the rule',
      '/** Reconstruction rounds before the run stops unconverged. Default 64; at most 26,207 (the command node budget). */'
    ],
    limits: [
      '/** Reconstruction rounds before the run stops unconverged. Default 64; at most 26,207 (the command node budget). */',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-morphology/gpu-raster-fill-holes-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:CPURasterFillHolesOperation': {
    signature: 'new CPURasterFillHolesOperation(props: RasterFillHolesOperationProps)',
    supportedInputs: ['props: RasterFillHolesOperationProps'],
    defaults: [
      "props.id ?? 'raster-fill-holes'",
      'Connectivity of the background that is flooded from the boundary. Defaults to `4`, the rule',
      '/** Reconstruction rounds before the run stops unconverged. Default 64; at most 26,207 (the command node budget). */'
    ],
    limits: [
      '/** Reconstruction rounds before the run stops unconverged. Default 64; at most 26,207 (the command node budget). */',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: 'unknown',
    source:
      'modules/analysis/src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation.ts'
  },
  '@luma-contributors/analysis/raster-analysis:computeRasterFillHolesOnCPU': {
    signature:
      'export function computeRasterFillHolesOnCPU( input: Uint32Array, options: CPURasterFillHolesOptions ): CPURasterFillHolesResult',
    supportedInputs: ['input: Uint32Array', 'options: CPURasterFillHolesOptions'],
    defaults: [
      'maximumIterations (default 64) bounds the flood distance just as on the GPU;',
      'options.maximumIterations ?? 64',
      'options.connectivity ?? 4',
      'Connectivity of the background that is flooded from the boundary. Defaults to `4`, the rule',
      '/** Reconstruction rounds before the run stops unconverged. Default 64; at most 26,207 (the command node budget). */'
    ],
    limits: [
      'maximumIterations (default 64) bounds the flood distance just as on the GPU;',
      'input.length < length',
      '/** Reconstruction rounds before the run stops unconverged. Default 64; at most 26,207 (the command node budget). */',
      '/** 1 when a round changed nothing before the iteration limit, else 0. */',
      '/** 1 when the iteration limit was reached without convergence, else 0. */'
    ],
    optionHandling: [
      'Only grid.width * grid.height input rows are read; any surplus rows are ignored.',
      'What differs from the GPU contributor:'
    ],
    source:
      'modules/analysis/src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation.ts'
  }
};
