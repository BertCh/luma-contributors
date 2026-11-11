// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGraphOutputViews,
  validateCapacity,
  validateMaximumCount,
  validateRequirement
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
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
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

/** Largest dense zone x cell candidate grid per invocation; bounds the coverage and compaction scratch. */
const MAXIMUM_CANDIDATE_PAIRS = 4194304;

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
 * The explicit dense candidate grid is capped at 4,194,304 zone/cell pairs per invocation; tile
 * larger domains explicitly without overlapping cores. No supersampling approximation is used.
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
  constructor(props: GPURasterFractionalCoverageProps) {
    this.id = props.id ?? 'raster-fractional-coverage';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [...Object.values(props.zones), props.parameters],
      getGraphOutputViews(props.output)
    );
    validateRasterParameters(
      this.id,
      props.parameters,
      GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH
    );
    const zones = props.zones.featureOffsets.length - 1;
    validateMaximumCount(this.id, 'zones times grid cells', count * zones, MAXIMUM_CANDIDATE_PAIRS);
    validateRequirement(
      this.id,
      props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3] !==
        0,
      'grid.affine must be invertible (nonzero determinant)'
    );
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
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, zones, output} = props;
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
        declarations: `fn setting(slot:u32)->f32{return settings[settingsOffset+slot];}
// A pixel coordinate within four float32 steps of an integer is that integer. A vertex meant to
// lie on a cell boundary does not come out of the inverse affine as an integer: the vertex and
// the cell size are each rounded to float32 (float32(0.9) / float32(0.3) is 2.9999998 in exact
// arithmetic), the products and the quotient round again, and GPU division need not be correctly
// rounded. Unsnapped, such a vertex leaves a sliver of coverage (of the order of 1e-7 of a cell)
// in the neighbouring cell, which gives a zone support in a cell it does not touch. The snap moves no
// vertex by more than the float32 resolution of its own pixel coordinate. It cannot absorb the
// rounding of 'position - origin', which is relative to the origin: keep the origin near the
// geometry (local coordinates).
fn snapToLattice(pixel:vec2<f32>)->vec2<f32>{let nearest=round(pixel);let tolerance=4.76837158203125e-7*max(abs(nearest),vec2<f32>(1.0));return select(pixel,nearest,abs(pixel-nearest)<=tolerance);}
// Pixel coordinates of a vertex. On an axis-aligned grid (b = d = 0) each axis is one division,
// which rounds less than the general form (two products, a difference and a quotient).
fn point(vertex:u32)->vec2<f32>{let x=positions[positionsOffset+vertex*2u]-setting(0u);let y=positions[positionsOffset+vertex*2u+1u]-setting(1u);if(setting(3u)==0.0&&setting(4u)==0.0){return snapToLattice(vec2<f32>(x/setting(2u),y/setting(5u)));}return snapToLattice(vec2<f32>((setting(5u)*x-setting(3u)*y)/setting(6u),(-setting(4u)*x+setting(2u)*y)/setting(6u)));}
fn clip(vertices:ptr<function,array<vec2<f32>,8>>,count:u32,axis:u32,bound:f32,lower:bool)->u32{
 var result:array<vec2<f32>,8>;var size=0u;if(count==0u){return 0u;}var previous=(*vertices)[count-1u];var previousInside=select(previous[axis]<=bound,previous[axis]>=bound,lower);
 for(var cursor=0u;cursor<count;cursor++){let current=(*vertices)[cursor];let inside=select(current[axis]<=bound,current[axis]>=bound,lower);if(inside!=previousInside){let fraction=(bound-previous[axis])/(current[axis]-previous[axis]);var intersection=mix(previous,current,fraction);intersection[axis]=bound;result[size]=intersection;size++;}if(inside){result[size]=current;size++;}previous=current;previousInside=inside;}
 for(var cursor=0u;cursor<size;cursor++){(*vertices)[cursor]=result[cursor];}return size;
}`,
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
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        body: `let capacity = ${output.zoneIds.length}u;
  count[countOffset] = min(total[totalOffset], capacity);
  required[requiredOffset] = total[totalOffset];
  overflow[overflowOffset] = u32(total[totalOffset] > capacity);`
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
    if (props.parameters) {
      validateRasterParameters(
        this.id,
        props.parameters,
        GPU_RASTER_FRACTIONAL_COVERAGE_PARAMETER_LENGTH
      );
    }
    validateCapacity(this.id, 'zoneCount', props.zoneCount, 0);
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
    const band = getRasterBufferBandNodes(graph, id, props.input, props.grid, true);
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.input),
      ...getGraphOutputViews(coverage),
      props.parameters,
      ...getGraphOutputViews(output)
    ]);
    const summary = createTransientView(graph, `${id}-summary`, 'float32x4', props.zoneCount);
    // Slot 6 of the coverage georeference is the affine determinant; read per frame when supplied.
    const cellArea = props.parameters
      ? 'abs(georeference[georeferenceOffset+6u])'
      : String(
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
