// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGraphOutputViews} from '../analysis-core/kernels/gpu-contributor-utils';
import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonGeometryFields} from '../analysis-core/contracts/index';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount
} from '../analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {getRasterBandViews, getRasterBufferBandNodes} from './raster-band-utils';
import {
  validateRasterExpansionViews,
  validateRasterExpansionGraph,
  requireRasterLength
} from './raster-expansion-utils';

export type GPURasterFractionalCoverageProps = {
  id?: string;
  grid: GPURasterMetadata;
  zones: GPUPolygonGeometryFields;
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
 * Float32 boundary slivers can affect raw tuple counts; compare area-weighted results with a
 * coordinate-scale tolerance rather than requiring exact positive-support counts.
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
      Object.values(props.zones),
      getGraphOutputViews(props.output)
    );
    const zones = props.zones.featureOffsets.length - 1;
    assert(count * zones <= 4194304);
    assert(
      props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3] !==
        0
    );
    requireRasterLength(props.output.cellIds, props.output.zoneIds.length);
    requireRasterLength(props.output.fractions, props.output.zoneIds.length);
    requireRasterLength(props.output.zoneValidity, zones);
    for (const view of [
      props.output.status.count,
      props.output.status.requiredCount,
      props.output.status.overflow
    ])
      requireRasterLength(view, 1);
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
      ...getGraphOutputViews(output)
    ]);
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
    const determinant = grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3];
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
          {name: 'coverage', view: coverage, type: 'f32', access: 'read_write'}
        ],
        declarations: `fn point(vertex:u32)->vec2<f32>{let x=positions[positionsOffset+vertex*2u]-${grid.affine[2]};let y=positions[positionsOffset+vertex*2u+1u]-${grid.affine[5]};return vec2<f32>((${grid.affine[4]}*x-${grid.affine[1]}*y)/${determinant},(-${grid.affine[3]}*x+${grid.affine[0]}*y)/${determinant});}
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
      createWGSLKernelNode(graph, {
        id: `${id}-compact`,
        operation: 'GPURasterFractionalCoverage',
        invocationCount: 1,
        bindings: [
          {name: 'coverage', view: coverage, type: 'f32', access: 'read'},
          {name: 'zoneIds', view: output.zoneIds, type: 'u32', access: 'read_write'},
          {name: 'cellIds', view: output.cellIds, type: 'u32', access: 'read_write'},
          {name: 'fractions', view: output.fractions, type: 'f32', access: 'read_write'},
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'required', view: output.status.requiredCount, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
          {name: 'zoneValidity', view: output.zoneValidity, type: 'u32', access: 'read_write'}
        ],
        body: `var total=0u;for(var zone=0u;zone<${zoneCount}u;zone++){zoneValidity[zoneValidityOffset+zone]=u32(coverage[coverageOffset+zone*${cellCount}u]>=0.0);for(var cell=0u;cell<${cellCount}u;cell++){let fraction=coverage[coverageOffset+zone*${cellCount}u+cell];if(fraction<=0.0){continue;}if(total<${output.zoneIds.length}u){zoneIds[zoneIdsOffset+total]=zone;cellIds[cellIdsOffset+total]=cell;fractions[fractionsOffset+total]=fraction;}total++;}}count[countOffset]=min(total,${output.zoneIds.length}u);required[requiredOffset]=total;overflow[overflowOffset]=u32(total>${output.zoneIds.length}u);`
      })
    ];
  }
}

export type GPURasterFractionalZonalStatisticsProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: import('./types').GPURasterBufferBand;
  zoneCount: number;
  coverage: Pick<
    GPURasterFractionalCoverageProps['output'],
    'zoneIds' | 'cellIds' | 'fractions' | 'status' | 'zoneValidity'
  >;
  /** Extensive cell totals are multiplied by fraction; densities/intensive integrals also by cell area. */
  quantity: 'extensive' | 'density' | 'intensive';
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
      [...getRasterBandViews(props.input), ...getGraphOutputViews(props.coverage)],
      getGraphOutputViews(props.output)
    );
    assert(Number.isInteger(props.zoneCount) && props.zoneCount >= 0);
    getGraphOutputViews(props.output).forEach(view => requireRasterLength(view, props.zoneCount));
    requireRasterLength(props.coverage.zoneValidity, props.zoneCount);
    requireRasterLength(props.coverage.cellIds, props.coverage.zoneIds.length);
    requireRasterLength(props.coverage.fractions, props.coverage.zoneIds.length);
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
      ...getGraphOutputViews(output)
    ]);
    const summary = createTransientView(graph, `${id}-summary`, 'float32x4', props.zoneCount);
    const cellArea = Math.abs(
      props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]
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
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
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
