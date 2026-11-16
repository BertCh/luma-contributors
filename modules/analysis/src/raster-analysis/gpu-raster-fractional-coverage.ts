// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateRasterFractionalBindingByteLength,
  validateRasterFractionalWritableViews
} from './fractional-coverage/raster-fractional-tile-kernel';
import {getViewBindingRange} from '@luma.gl/graph/gpu-core';
import {
  getIndexedRasterFractionalCoverageNodes,
  validateFractionalCoverageBindings,
  RASTER_FRACTIONAL_CLIPPING_WGSL,
  RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE
} from './fractional-coverage/raster-fractional-coverage-index';
import {
  getGraphOutputViews,
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonGeometryFields} from '../analysis-core/contracts/index';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount
} from '../analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {GPUCompaction} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateParameterTarget,
  validateRasterParameters
} from './raster-band-utils';
import {
  validateRasterExpansionViews,
  validateRasterExpansionGraph,
  requireRasterLength
} from './raster-expansion-utils';

/** Integer no-data sentinels are emitted by the public raster canonicalizer as typed literals. */
const MAXIMUM_UNSIGNED_RASTER_VALUE = 0xffffffff;
/** Signed raster sentinel limits prevent invalid i32 literals in the canonicalizer. */
const MINIMUM_SIGNED_RASTER_VALUE = -2147483648;
/** Signed raster sentinel limits prevent invalid i32 literals in the canonicalizer. */
const MAXIMUM_SIGNED_RASTER_VALUE = 2147483647;

/** Zone indices are uint32 in the zonal consumer; device-sized summary scratch is checked at build. */
const MAXIMUM_ZONE_COUNT = 0xffffffff;

/** Explicit brute-force reference limit bounds its dense zone x cell scratch. */
const MAXIMUM_CANDIDATE_PAIRS = 4194304;

/** uint32 candidate offsets reserve the last word for a saturated count. */
const MAXIMUM_INDEXED_CANDIDATES = 0xfffffffe;

/** Default candidate workload budget starts at the historical pair count for sparse zones. */
const DEFAULT_CANDIDATE_CAPACITY = 4194304;

/** Conservative allowance for triangulation, bounds reductions and saturated scans. */
const INDEX_COMMAND_NODE_ALLOWANCE = 4096;

/** Upper bound on the published compaction nodes plus clipping and publication per tile. */
const CANDIDATE_TILE_COMMAND_NODE_ALLOWANCE = 32;

/** Number of float32 values read from {@link GPURasterFractionalCoverageProps.parameters}. */
export const GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH = 8;

/**
 * Packs the grid georeference read by {@link GPURasterFractionalCoverage}.
 *
 * Layout (float32): `[originX, originY, a, b, d, e, determinant, 0]`, where
 * `grid.affine = [a, b, originX, d, e, originY]` and `determinant = a * e - b * d`. Write the result
 * into a `GPUParameterBuffer` between encodings to move the same compiled graph to another
 * georeference of the same width and height without recompiling.
 *
 * @param grid Grid whose affine is packed; it must be invertible.
 * @param target Optional destination of at least 8 values. A new array is returned when omitted.
 * @throws If the affine is singular or not finite, or `target` is too short.
 */
export function getGPURasterFractionalCoverageParameterValues(
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterFractionalCoverageParameterValues';
  validateParameterTarget(id, target, GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH);
  const [a, b, originX, d, e, originY] = grid.affine;
  const determinant = a * e - b * d;
  validateRequirement(
    id,
    grid.affine.every(Number.isFinite) && determinant !== 0,
    'grid.affine must be finite and invertible (nonzero determinant)'
  );
  target.set([originX, originY, a, b, d, e, determinant, 0]);
  return target;
}

export type GPURasterFractionalCoverageProps = {
  id?: string;
  grid: GPURasterMetadata;
  zones: GPUPolygonGeometryFields;
  /** Compile-time candidate lookup; bounds is the scalable default. Brute force retains its 4M pair cap. */
  searchStructure?: 'bounds' | 'brute-force';
  /**
   * Compile-time budget for cells in the union of zone pixel bounding rectangles, before clipping.
   * Defaults to min(zones * cells, max(4M, output capacity * 4)). Scratch is tiled, not budget-sized.
   * Exhaustion sets candidateOverflow (when supplied) and requiredCount = 0xffffffff;
   * zoneValidity is zero because the unvisited support is unknown. Increase this budget and rerun.
   */
  candidateCapacity?: number;
  /**
   * Per-frame georeference with at least {@link GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH}
   * float32 values, written with {@link getGPURasterFractionalCoverageParameterValues}. Only
   * `grid.width` and `grid.height` are compile-time.
   */
  parameters: GraphDataView<'float32'>;
  output: {
    zoneIds: GraphDataView<'uint32'>;
    cellIds: GraphDataView<'uint32'>;
    fractions: GraphDataView<'float32'>;
    status: {
      count: GraphDataView<'uint32'>;
      requiredCount: GraphDataView<'uint32'>;
      overflow: GraphDataView<'uint32'>;
      /** Indexed candidate budget exhausted; requiredCount then saturates at 0xffffffff. */
      candidateOverflow?: GraphDataView<'uint32'>;
    };
    /** Validity per source feature. Failed triangulation emits no tuples for that feature. */
    zoneValidity: GraphDataView<'uint32'>;
  };
};

/**
 * Exact planar triangle/pixel clipping followed by compact (zone,cell,coverage) tuples. Polygon
 * triangulation supplies disjoint shell-minus-hole triangles, so holes and overlapping features
 * retain their semantics. Coverage is in [0,1] for valid nonoverlapping multipart features.
 * Areas use abs(affine determinant); intensive means and extensive totals remain separate consumers.
 * The default bounds index enumerates zone bounding rectangles in disjoint fixed-size candidate
 * tiles, with stable zone/cell order and no global area normalisation. The explicit brute-force
 * route retains its 4,194,304 pair cap. No supersampling approximation is used.
 * A vertex whose pixel coordinate is within four float32 steps of an integer is placed on that
 * cell boundary, so zone edges that follow cell boundaries leave no sliver in the neighbouring
 * cells, also when the cell size is not a float32 number (0.3, a third of a degree). Elsewhere
 * float32 boundary slivers can still affect raw tuple counts (edges that cross a cell obliquely,
 * origins large against the geometry); compare area-weighted results with a coordinate-scale
 * tolerance rather than requiring exact positive-support counts.
 */
export class GPURasterFractionalCoverage implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFractionalCoverageProps;
  readonly candidateCapacity: number;
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPURasterFractionalCoverageProps) {
    this.id = props.id ?? 'raster-fractional-coverage';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [...Object.values(props.zones), props.parameters],
      getGraphOutputViews(props.output)
    );
    validateRasterFractionalWritableViews(this.id, props.output, [
      ...Object.values(props.zones),
      props.parameters
    ]);
    validateRasterParameters(
      this.id,
      props.parameters,
      GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH
    );
    const zones = props.zones.featureOffsets.length - 1;
    validateRequirement(
      this.id,
      props.searchStructure === undefined ||
        ['bounds', 'brute-force'].includes(props.searchStructure),
      "searchStructure must be 'bounds' or 'brute-force'"
    );
    validateSinglePackedView(props.zones.positions, ['float32x2'], `${this.id} zones.positions`);
    for (const name of ['featureOffsets', 'polygonOffsets', 'ringOffsets'] as const) {
      validateSinglePackedView(props.zones[name], ['uint32'], `${this.id} zones.${name}`);
      validateCapacity(this.id, `zones.${name}.length`, props.zones[name].length);
    }
    this.candidateCapacity =
      props.candidateCapacity ??
      Math.min(
        count * zones,
        Math.max(DEFAULT_CANDIDATE_CAPACITY, props.output.zoneIds.length * 4)
      );
    validateCapacity(
      this.id,
      'candidateCapacity',
      this.candidateCapacity,
      0,
      MAXIMUM_INDEXED_CANDIDATES
    );
    if (props.searchStructure === 'brute-force') {
      validateMaximumCount(
        this.id,
        'zones times grid cells',
        count * zones,
        MAXIMUM_CANDIDATE_PAIRS
      );
    }
    this.estimatedCommandNodeCount =
      INDEX_COMMAND_NODE_ALLOWANCE +
      CANDIDATE_TILE_COMMAND_NODE_ALLOWANCE *
        Math.ceil(this.candidateCapacity / RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateRequirement(
      this.id,
      props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3] !==
        0,
      'grid.affine must be invertible (nonzero determinant)'
    );
    for (const [name, view] of Object.entries({...props.output, ...props.output.status})) {
      if (name === 'status' || !view) continue;
      validateSinglePackedView(
        view as GraphDataView,
        [name === 'fractions' ? 'float32' : 'uint32'],
        `${this.id} output.${name}`
      );
    }
    const capacity = props.output.zoneIds.length;
    requireRasterLength(props.output.cellIds, capacity, this.id, 'output.cellIds');
    requireRasterLength(props.output.fractions, capacity, this.id, 'output.fractions');
    requireRasterLength(props.output.zoneValidity, zones, this.id, 'output.zoneValidity');
    requireRasterLength(props.output.status.count, 1, this.id, 'output.status.count');
    requireRasterLength(
      props.output.status.requiredCount,
      1,
      this.id,
      'output.status.requiredCount'
    );
    requireRasterLength(props.output.status.overflow, 1, this.id, 'output.status.overflow');
    requireRasterLength(
      props.output.status.candidateOverflow,
      1,
      this.id,
      'output.status.candidateOverflow'
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, zones, output} = props;
    validateFractionalCoverageBindings(graph, this);
    if (props.searchStructure !== 'brute-force') {
      return getIndexedRasterFractionalCoverageNodes(graph, this);
    }
    const cellCount = grid.width * grid.height;
    const zoneCount = zones.featureOffsets.length - 1;
    const polygonCount = zones.polygonOffsets.length - 1;
    validateRasterExpansionGraph(id, graph, [
      ...Object.values(zones),
      props.parameters,
      ...getGraphOutputViews(output)
    ]);
    const settings = props.parameters;
    const indices = createTransientView(
      graph,
      `${id}-indices`,
      'uint32',
      getPolygonTriangulationIndexCount(
        zones.positions.length,
        zones.ringOffsets.length - 1,
        polygonCount
      )
    );
    const triangleOffsets = createTransientView(
      graph,
      `${id}-triangle-offsets`,
      'uint32',
      polygonCount + 1
    );
    const valid = createTransientView(graph, `${id}-valid`, 'uint32', polygonCount);
    const coverage = createTransientView(graph, `${id}-coverage`, 'float32', cellCount * zoneCount);
    // Dense pair identifiers, selection flags, and the compacted pairs (all zone x cell long).
    const pairIds = createTransientView(graph, `${id}-pair-ids`, 'uint32', cellCount * zoneCount);
    const pairFlags = createTransientView(
      graph,
      `${id}-pair-flags`,
      'uint32',
      cellCount * zoneCount
    );
    const compactPairs = createTransientView(
      graph,
      `${id}-compact-pairs`,
      'uint32',
      cellCount * zoneCount
    );
    const requiredCount = createTransientView(graph, `${id}-required-total`, 'uint32', 1);
    return [
      ...new GPUPolygonTriangulation({
        ...zones,
        id: `${id}-triangulate`,
        output: {indices, valid, triangleOffsets}
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-intersect`,
        operation: 'GPURasterFractionalCoverage',
        invocationCount: cellCount * zoneCount,
        bindings: [
          {name: 'positions', view: zones.positions, type: 'f32', access: 'read'},
          {name: 'features', view: zones.featureOffsets, type: 'u32', access: 'read'},
          {name: 'offsets', view: triangleOffsets, type: 'u32', access: 'read'},
          {name: 'indices', view: indices, type: 'u32', access: 'read'},
          {name: 'valid', view: valid, type: 'u32', access: 'read'},
          {name: 'coverage', view: coverage, type: 'f32', access: 'read_write'},
          {name: 'settings', view: settings, type: 'f32', access: 'read'}
        ],
        // settings = [originX, originY, a, b, d, e, determinant, 0]; inverts the affine per vertex.
        declarations: RASTER_FRACTIONAL_CLIPPING_WGSL,
        body: `let zone=index/${cellCount}u;let cell=index%${cellCount}u;let column=f32(cell%${grid.width}u)${grid.pixelInterpretation === 'point' ? '-0.5' : ''};let row=f32(cell/${grid.width}u)${grid.pixelInterpretation === 'point' ? '-0.5' : ''};var area=0.0;var complete=true;
for(var polygon=features[featuresOffset+zone];polygon<features[featuresOffset+zone+1u];polygon++){
 if(valid[validOffset+polygon]==0u){complete=false;break;}
 for(var triangle=offsets[offsetsOffset+polygon];triangle<offsets[offsetsOffset+polygon+1u];triangle++){
  // Cell-local clipping avoids cancellation between large shoelace products.
  let cellOrigin=vec2<f32>(column,row);
  var vertices:array<vec2<f32>,8>;vertices[0]=point(indices[indicesOffset+triangle*3u])-cellOrigin;vertices[1]=point(indices[indicesOffset+triangle*3u+1u])-cellOrigin;vertices[2]=point(indices[indicesOffset+triangle*3u+2u])-cellOrigin;
  var size=clip(&vertices,3u,0u,0.0,true);size=clip(&vertices,size,0u,1.0,false);size=clip(&vertices,size,1u,0.0,true);size=clip(&vertices,size,1u,1.0,false);
  var doubledArea=0.0;for(var vertex=0u;vertex<size;vertex++){let first=vertices[vertex];let second=vertices[(vertex+1u)%size];doubledArea+=first.x*second.y-first.y*second.x;}area+=abs(doubledArea)*0.5;
 }
}coverage[coverageOffset+index]=select(-1.0,min(area,1.0),complete);`
      }),
      // Parallel stable compaction of the positive-coverage pairs of the dense zone x cell grid.
      // Dense pair indices ascend zone-major and cell-minor, so the order matches the former
      // serial walk exactly.
      createWGSLKernelNode(graph, {
        id: `${id}-flag`,
        operation: 'GPURasterFractionalCoverage',
        variant: 'compaction-flags',
        invocationCount: cellCount * zoneCount,
        bindings: [
          {name: 'coverage', view: coverage, type: 'f32', access: 'read'},
          {name: 'pairIds', view: pairIds, type: 'u32', access: 'read_write'},
          {name: 'flags', view: pairFlags, type: 'u32', access: 'read_write'}
        ],
        // Incomplete zones carry coverage -1 and NaN is kept, as in a "fraction <= 0 is skipped" walk.
        body: `pairIds[pairIdsOffset + index] = index;
  flags[flagsOffset + index] = u32(!(coverage[coverageOffset + index] <= 0.0));`
      }),
      ...new GPUCompaction({
        id: `${id}-compaction`,
        input: pairIds,
        flags: pairFlags,
        output: compactPairs,
        count: requiredCount
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterFractionalCoverage',
        variant: 'publish-tuples',
        invocationCount: Math.max(output.zoneIds.length, 1),
        bindings: [
          {name: 'coverage', view: coverage, type: 'f32', access: 'read'},
          {name: 'compactPairs', view: compactPairs, type: 'u32', access: 'read'},
          {name: 'required', view: requiredCount, type: 'u32', access: 'read'},
          {name: 'zoneIds', view: output.zoneIds, type: 'u32', access: 'read_write'},
          {name: 'cellIds', view: output.cellIds, type: 'u32', access: 'read_write'},
          {name: 'fractions', view: output.fractions, type: 'f32', access: 'read_write'}
        ],
        body: `if (index >= required[requiredOffset]) { return; }
  let pair = compactPairs[compactPairsOffset + index];
  zoneIds[zoneIdsOffset + index] = pair / ${cellCount}u;
  cellIds[cellIdsOffset + index] = pair % ${cellCount}u;
  fractions[fractionsOffset + index] = coverage[coverageOffset + pair];`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-zone-validity`,
        operation: 'GPURasterFractionalCoverage',
        variant: 'zone-validity',
        invocationCount: Math.max(zoneCount, 1),
        bindings: [
          {name: 'coverage', view: coverage, type: 'f32', access: 'read'},
          {name: 'zoneValidity', view: output.zoneValidity, type: 'u32', access: 'read_write'}
        ],
        // A zone is valid when its first cell was not marked incomplete (coverage -1).
        body: `if (index >= ${zoneCount}u) { return; }
  zoneValidity[zoneValidityOffset + index] = u32(coverage[coverageOffset + index * ${cellCount}u] >= 0.0);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-status`,
        operation: 'GPURasterFractionalCoverage',
        variant: 'publish-status',
        invocationCount: 1,
        bindings: [
          {name: 'total', view: requiredCount, type: 'u32', access: 'read'},
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'required', view: output.status.requiredCount, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
          ...(output.status.candidateOverflow
            ? [
                {
                  name: 'candidateOverflow',
                  view: output.status.candidateOverflow,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `let capacity = ${output.zoneIds.length}u;
  count[countOffset] = min(total[totalOffset], capacity);
  required[requiredOffset] = total[totalOffset];
  overflow[overflowOffset] = u32(total[totalOffset] > capacity);
  ${output.status.candidateOverflow ? 'candidateOverflow[candidateOverflowOffset] = 0u;' : ''}`
      })
    ];
  }
}

export type GPURasterFractionalZonalStatisticsProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GPURasterBufferBand;
  zoneCount: number;
  coverage: Pick<
    GPURasterFractionalCoverageProps['output'],
    'zoneIds' | 'cellIds' | 'fractions' | 'status' | 'zoneValidity'
  >;
  /** Extensive cell totals are multiplied by fraction; densities/intensive integrals also by cell area. */
  quantity: 'extensive' | 'density' | 'intensive';
  /**
   * The per-frame georeference view given to {@link GPURasterFractionalCoverage} (layout of
   * {@link getGPURasterFractionalCoverageParameterValues}). The cell area is read from it on every
   * encoding as `abs(determinant)`, so the areas follow a repacked georeference. When omitted the
   * cell area is `abs(det(grid.affine))` fixed at compile time: pass it whenever the coverage
   * `parameters` are rewritten for another cell size.
   */
  parameters?: GraphDataView<'float32'>;
  output: {
    totals: GraphDataView<'float32'>;
    means: GraphDataView<'float32'>;
    coveredArea: GraphDataView<'float32'>;
    validArea: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
  };
};

/** Area-aware fractional aggregation, preserving missing support and incomplete coverage. */
export class GPURasterFractionalZonalStatistics implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFractionalZonalStatisticsProps;
  constructor(props: GPURasterFractionalZonalStatisticsProps) {
    this.id = props.id ?? 'fractional-zonal-statistics';
    this.props = props;
    validateRasterExpansionViews(
      this.id,
      props.grid,
      [
        ...getRasterBandViews(props.input),
        ...getGraphOutputViews(props.coverage),
        props.parameters
      ],
      getGraphOutputViews(props.output)
    );
    validateRasterFractionalWritableViews(this.id, props.output, [
      ...getRasterBandViews(props.input),
      ...getGraphOutputViews(props.coverage),
      props.parameters
    ]);
    if (props.parameters) {
      validateRasterParameters(
        this.id,
        props.parameters,
        GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH
      );
    }
    // These scalars become f32 literals in this consumer or its public raster canonicalizer.
    const cellArea = Math.abs(
      props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
    );
    validateRequirement(
      this.id,
      props.parameters !== undefined || Number.isFinite(Math.fround(cellArea)),
      'grid.affine determinant must fit in finite float32'
    );
    for (const name of ['scale', 'offset'] as const) {
      validateRequirement(
        this.id,
        props.input[name] === undefined || Number.isFinite(Math.fround(props.input[name]!)),
        `input.${name} must fit in finite float32`
      );
    }
    if (props.input.noDataValue !== undefined) {
      const value = props.input.noDataValue;
      validateRequirement(
        this.id,
        props.input.format === 'float32'
          ? Number.isNaN(value) || Number.isFinite(Math.fround(value))
          : Number.isInteger(value) &&
              (props.input.format === 'uint32'
                ? value >= 0 && value <= MAXIMUM_UNSIGNED_RASTER_VALUE
                : value >= MINIMUM_SIGNED_RASTER_VALUE && value <= MAXIMUM_SIGNED_RASTER_VALUE),
        `input.noDataValue must fit in ${props.input.format}`
      );
    }
    validateCapacity(this.id, 'zoneCount', props.zoneCount, 0, MAXIMUM_ZONE_COUNT);
    for (const [name, view] of Object.entries(props.output)) {
      requireRasterLength(view, props.zoneCount, this.id, `output.${name}`);
    }
    requireRasterLength(
      props.coverage.zoneValidity,
      props.zoneCount,
      this.id,
      'coverage.zoneValidity'
    );
    requireRasterLength(
      props.coverage.cellIds,
      props.coverage.zoneIds.length,
      this.id,
      'coverage.cellIds'
    );
    requireRasterLength(
      props.coverage.fractions,
      props.coverage.zoneIds.length,
      this.id,
      'coverage.fractions'
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {coverage, output} = props;
    for (const view of [
      ...getRasterBandViews(props.input),
      ...getGraphOutputViews(coverage),
      props.parameters,
      ...getGraphOutputViews(output)
    ]) {
      if (view)
        validateRasterFractionalBindingByteLength(
          graph.device,
          id,
          view.buffer.id,
          getViewBindingRange(view).size,
          'split the raster or coverage into disjoint cores',
          view.buffer.byteLength
        );
    }
    validateRasterFractionalBindingByteLength(
      graph.device,
      id,
      'canonical raster bands',
      props.grid.width * props.grid.height * 4,
      'split the raster into disjoint cores'
    );
    validateRasterFractionalBindingByteLength(
      graph.device,
      id,
      'zone summary',
      Math.max(props.zoneCount, 1) * 16,
      'reduce zoneCount'
    );
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.input),
      ...getGraphOutputViews(coverage),
      props.parameters,
      ...getGraphOutputViews(output)
    ]);
    const band = getRasterBufferBandNodes(graph, id, props.input, props.grid, true);
    const summary = createTransientView(graph, `${id}-summary`, 'float32x4', props.zoneCount);
    // Slot 6 of the coverage georeference is the affine determinant; read per frame when supplied.
    const cellArea = props.parameters
      ? 'abs(georeference[georeferenceOffset+6u])'
      : getWGSLFloatLiteral(
          Math.abs(
            props.grid.affine[0] * props.grid.affine[4] -
              props.grid.affine[1] * props.grid.affine[3]
          )
        );
    return [
      ...band.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-aggregate`,
        operation: 'GPURasterFractionalZonalStatistics',
        invocationCount: props.zoneCount,
        bindings: [
          {name: 'values', view: band.band.storage.values, type: 'f32', access: 'read'},
          {name: 'validity', view: band.band.validity!, type: 'u32', access: 'read'},
          {name: 'zoneIds', view: coverage.zoneIds, type: 'u32', access: 'read'},
          {name: 'cellIds', view: coverage.cellIds, type: 'u32', access: 'read'},
          {name: 'fractions', view: coverage.fractions, type: 'f32', access: 'read'},
          {name: 'count', view: coverage.status.count, type: 'u32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'},
          ...(props.parameters
            ? [
                {
                  name: 'georeference',
                  view: props.parameters,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        body: `var total=0.0;var weighted=0.0;var covered=0.0;var observed=0.0;for(var tuple=0u;tuple<min(count[countOffset],${coverage.zoneIds.length}u);tuple++){if(zoneIds[zoneIdsOffset+tuple]!=index){continue;}let cell=cellIds[cellIdsOffset+tuple];if(cell>=${props.grid.width * props.grid.height}u){continue;}let fraction=fractions[fractionsOffset+tuple];covered+=fraction;if(validity[validityOffset+cell]!=0u){let value=values[valuesOffset+cell];weighted+=value*fraction;total+=value*fraction*${props.quantity === 'extensive' ? '1.0' : cellArea};observed+=fraction;}}summary[summaryOffset+index*4u]=total;summary[summaryOffset+index*4u+1u]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),weighted/max(observed,1.0e-30),observed>0.0);summary[summaryOffset+index*4u+2u]=covered*${cellArea};summary[summaryOffset+index*4u+3u]=observed*${cellArea};`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterFractionalZonalStatistics',
        invocationCount: props.zoneCount,
        bindings: [
          {name: 'summary', view: summary, type: 'f32', access: 'read'},
          {name: 'overflow', view: coverage.status.overflow, type: 'u32', access: 'read'},
          {name: 'zoneValidity', view: coverage.zoneValidity, type: 'u32', access: 'read'},
          {name: 'totals', view: output.totals, type: 'f32', access: 'read_write'},
          {name: 'means', view: output.means, type: 'f32', access: 'read_write'},
          {name: 'covered', view: output.coveredArea, type: 'f32', access: 'read_write'},
          {name: 'observed', view: output.validArea, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
        ],
        body: `let complete=overflow[overflowOffset]==0u&&zoneValidity[zoneValidityOffset+index]!=0u;let valid=complete&&summary[summaryOffset+index*4u+3u]>0.0;let invalid=bitcast<f32>(0x7fc00000u | (index & 0u));totals[totalsOffset+index]=select(invalid,summary[summaryOffset+index*4u],valid);means[meansOffset+index]=select(invalid,summary[summaryOffset+index*4u+1u],valid);covered[coveredOffset+index]=select(invalid,summary[summaryOffset+index*4u+2u],complete);observed[observedOffset+index]=select(invalid,summary[summaryOffset+index*4u+3u],complete);validity[validityOffset+index]=u32(valid);`
      })
    ];
  }
}
