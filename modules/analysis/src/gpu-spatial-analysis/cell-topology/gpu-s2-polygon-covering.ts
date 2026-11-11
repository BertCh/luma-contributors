// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assertUnitVector} from '../global-coordinates/assert-unit-vector';
import {
  GPU_S2_COVERING_MAXIMUM_LEVEL,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from '../global-coordinates/spherical-limits';
import {
  GPUScan,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';
import {getScanCommandNodeCount} from './s2-covering-support';

/**
 * Largest `maximumLevel`. Level 15 cells are a few hundred meters across, already near the
 * resolution of f32 unit vectors and of the largest accepted `tolerance` (1e-3 radians is 6 km,
 * the default 1e-6 is 6 m); deeper levels classify against rounding noise.
 */
export const GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL = GPU_S2_COVERING_MAXIMUM_LEVEL;

/** Fewest visited cells a traversal can have: the six cube-face cells it starts from. */
export const GPU_S2_POLYGON_COVERING_MINIMUM_VISITED_CELLS = 6;

/**
 * Largest `maximumVisitedCells`. The serial traversal keeps no list of its own, but the
 * level-synchronous traversal holds the whole frontier in two ping-pong buffers of this many
 * cells (8 bytes each), plus counts and offsets, which stays under the 128 MiB default binding
 * limit.
 */
export const GPU_S2_POLYGON_COVERING_MAXIMUM_VISITED_CELLS = 1048576;

/** Default `maximumVisitedCells`: 64 Ki cells, 512 KiB per frontier buffer. */
export const GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS = 65536;

/**
 * Depth-first stack of the serial traversal: three deferred siblings per level for the 16 levels
 * 0 to {@link GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL}, plus the four children of the cell being
 * expanded.
 */
const SERIAL_STACK_SIZE = 3 * (GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL + 1) + 4;

/** Status bit: invalid input (a malformed or non-simple ring, a non-unit vertex, an empty hemisphere). */
export const GPU_S2_POLYGON_COVERING_STATUS_INVALID = 1;
/** Status bit: more cells than `capacity`; `counts[1]` holds the required count. */
export const GPU_S2_POLYGON_COVERING_STATUS_CAPACITY = 2;
/**
 * Status bit: the traversal needed more than `maximumVisitedCells` cells. The cover is incomplete:
 * the serial traversal keeps the cells emitted so far and `counts[1]` is not a complete
 * requirement; the level-synchronous traversal writes no cells and reports `counts` of `(0, 0)`.
 */
export const GPU_S2_POLYGON_COVERING_STATUS_BUDGET = 4;

/**
 * How the quad-tree is traversed.
 * - `'serial'`: one invocation walks the tree depth first. Exhaustion of `maximumVisitedCells`
 *   keeps the cells emitted so far.
 * - `'level-synchronous'`: one invocation per frontier cell per level; each of the
 *   `maximumLevel + 1` rounds classifies the frontier, then a scan and a write build the next one.
 *   The cover and `visitedCount` equal the serial traversal's, in the same ascending cell-ID order.
 */
export type GPUS2PolygonCoveringRefinement = 'serial' | 'level-synchronous';

export type GPUS2PolygonCoveringProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  /** One prepared simple spherical polygon: CCW shell then CCW holes, unit xyz minor arcs. */
  positions: GraphDataView<'float32x3'>;
  ringOffsets: GraphDataView<'uint32'>;
  complement?: boolean;
  minimumLevel?: number;
  maximumLevel: number;
  mode?: 'exterior' | 'interior';
  capacity: number;
  /**
   * Maximum visited hierarchy cells. Exhaustion is explicit; default
   * {@link GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS}. In the level-synchronous traversal it
   * is also the frontier capacity.
   */
  maximumVisitedCells?: number;
  tolerance?: number;
  /** Optional common hemisphere pole. Omitted: each ring uses its normalized vertex mean. */
  hemisphereCenter?: readonly [number, number, number];
  /**
   * Traversal. Defaults to `'serial'`, whose outputs, including partial results on exhaustion, are
   * pinned by recorded fixtures. `'level-synchronous'` runs in parallel and scales with the
   * polygon's boundary instead of one thread.
   */
  refinement?: GPUS2PolygonCoveringRefinement;
  output: {
    cells: GraphDataView<'uint32x2'>;
    /** (written, required so far), not a complete requirement when status has bit 4. */
    counts: GraphDataView<'uint32x2'>;
    /**
     * One word, a bit set: {@link GPU_S2_POLYGON_COVERING_STATUS_INVALID} (1),
     * {@link GPU_S2_POLYGON_COVERING_STATUS_CAPACITY} (2),
     * {@link GPU_S2_POLYGON_COVERING_STATUS_BUDGET} (4); 0 is complete.
     */
    status: GraphDataView<'uint32'>;
    visitedCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Adaptive mixed-level S2 interior/exterior cover of one polygon, in a bounded GPU traversal.
 * The `'serial'` traversal runs in one invocation (an O(V^2) ring check, then a depth-first walk
 * of up to 1,048,576 cells); `refinement: 'level-synchronous'` distributes both across invocations.
 *
 * The level-synchronous rounds are the reference tier and are not gated on the live frontier
 * length: every round dispatches `maximumVisitedCells` invocations (and a scan over as many rows)
 * whatever the live list length. The published `GPUScan` has no GPU-written dispatch size, so a
 * gate would have to skip the scan nodes that dominate each round; rounds after the cover is
 * complete therefore still run, and only classify and write return early.
 */
export class GPUS2PolygonCovering implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  /** Exact number of command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUS2PolygonCoveringProps) {
    this.id = props.id ?? 's2-polygon-covering';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    if (props.hemisphereCenter)
      assertUnitVector(props.hemisphereCenter, this.id, 'hemisphereCenter');
    validateSinglePackedView(props.positions, ['float32x3'], this.id);
    validateSinglePackedView(props.ringOffsets, ['uint32'], this.id);
    validateSinglePackedView(props.output.cells, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.counts, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.status, ['uint32'], this.id);
    if (props.output.visitedCount) {
      validateSinglePackedView(props.output.visitedCount, ['uint32'], this.id);
      validateRequirement(
        this.id,
        props.output.visitedCount.length === 1,
        'output.visitedCount must hold one entry'
      );
    }
    validateCapacity(
      this.id,
      'maximumLevel',
      props.maximumLevel,
      0,
      GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL
    );
    validateCapacity(this.id, 'minimumLevel', props.minimumLevel ?? 0, 0, props.maximumLevel);
    validateCapacity(this.id, 'capacity', props.capacity);
    validateRequirement(
      this.id,
      props.output.cells.length === props.capacity &&
        props.output.counts.length === 1 &&
        props.output.status.length === 1,
      'output.cells must hold capacity entries, and output.counts and output.status one entry'
    );
    validateCapacity(
      this.id,
      'maximumVisitedCells',
      props.maximumVisitedCells ?? GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS,
      GPU_S2_POLYGON_COVERING_MINIMUM_VISITED_CELLS,
      GPU_S2_POLYGON_COVERING_MAXIMUM_VISITED_CELLS
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE
    );
    validateRequirement(
      this.id,
      props.refinement === undefined ||
        props.refinement === 'serial' ||
        props.refinement === 'level-synchronous',
      "refinement must be 'serial' or 'level-synchronous'"
    );
    validateRequirement(
      this.id,
      props.ringOffsets.length >= 2,
      'ringOffsets must hold at least 2 entries (one ring)'
    );
    // Level-synchronous: start, rings and edges nodes; per round a classify, the scan over the
    // frontier rows and a write; then the cells and publish nodes.
    this.estimatedCommandNodeCount =
      props.refinement === 'level-synchronous'
        ? 5 +
          (props.maximumLevel + 1) *
            (2 +
              getScanCommandNodeCount(
                props.maximumVisitedCells ?? GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS
              ))
        : 1;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets
    ]);
  }
  /** Shared WGSL helpers: ring containment, arc tests and cell-polygon classification. */
  private getDeclarations(): string {
    const {props} = this;
    return `${dggs.source}\n${S2_ANALYSIS_WGSL}
const COVER_TOLERANCE:f32=${props.tolerance ?? 1e-6};
fn coverPosition(row:u32)->vec3f {let offset=positionsOffset+row*3u;return vec3f(positions[offset],positions[offset+1u],positions[offset+2u]);}
fn coverOnArc(first:vec3f,second:vec3f,point:vec3f)->bool {let normal=normalize(cross(first,second));return abs(dot(normal,point))<=COVER_TOLERANCE&&dot(cross(first,point),normal)>=-COVER_TOLERANCE&&dot(cross(point,second),normal)>=-COVER_TOLERANCE;}
fn coverContains(point:vec3f)->vec2u {
  var inside=false;var boundary=false;
  for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++) {
    let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];var winding=0.0;
    for(var edge=begin;edge<end;edge++) {let first=coverPosition(edge);let second=coverPosition(select(edge+1u,begin,edge+1u==end));boundary=boundary||coverOnArc(first,second,point);let numerator=dot(point,cross(first,second));let denominator=dot(first,second)-dot(point,first)*dot(point,second);if(abs(numerator)+abs(denominator)>1e-12){winding+=s2AnalysisAtan2(numerator,denominator);}}
    if(ring==0u){inside=winding>3.14159265;}else{inside=inside&&winding<=3.14159265;}
  }
  return vec2u(select(0u,1u,${props.complement ? '!inside' : 'inside'}),select(0u,1u,boundary));
}
fn coverCellContains(normals:ptr<function,array<vec3f,4>>,point:vec3f)->bool {
  for(var edge=0u;edge<4u;edge++){if(dot((*normals)[edge],point)<-COVER_TOLERANCE){return false;}}
  return true;
}
fn coverCellOnArc(first:vec3f,second:vec3f,normal:vec3f,point:vec3f)->bool {
  return abs(dot(normal,point))<=COVER_TOLERANCE&&dot(cross(first,point),normal)>=-COVER_TOLERANCE&&dot(cross(point,second),normal)>=-COVER_TOLERANCE;
}
fn coverClassify(cell:vec2u)->vec2u {
  var intersects=false;var contained=true;var touched=false;
  var vertices:array<vec3f,4>;var normals:array<vec3f,4>;var crosses:array<vec3f,4>;
  for(var edge=0u;edge<4u;edge++){vertices[edge]=s2AnalysisVertex(cell,edge);}
  for(var edge=0u;edge<4u;edge++){crosses[edge]=cross(vertices[edge],vertices[(edge+1u)%4u]);normals[edge]=normalize(crosses[edge]);}
  // Polygon vertex containment does not depend on the current cell edge.
  for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++){
    let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];
    for(var vertex=begin;vertex<end;vertex++){if(coverCellContains(&normals,coverPosition(vertex))){touched=true;intersects=true;}}
  }
  for(var edge=0u;edge<4u;edge++) {
    let first=vertices[edge];let second=vertices[(edge+1u)%4u];let normal=normals[edge];
    let classification=coverContains(first);intersects=intersects||classification.x!=0u||classification.y!=0u;contained=contained&&classification.x!=0u&&classification.y==0u;
    for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++){
      let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];
      for(var vertex=begin;vertex<end;vertex++){
        let third=coverPosition(vertex);let fourth=coverPosition(select(vertex+1u,begin,vertex+1u==end));
        let crossing=cross(crosses[edge],cross(third,fourth));
        if(length(crossing)>COVER_TOLERANCE*COVER_TOLERANCE){
          let point=normalize(crossing);
          if((coverCellOnArc(first,second,normal,point)&&coverOnArc(third,fourth,point))||(coverCellOnArc(first,second,normal,-point)&&coverOnArc(third,fourth,-point))){touched=true;intersects=true;}
        }else if(coverCellOnArc(first,second,normal,third)||coverCellOnArc(first,second,normal,fourth)){touched=true;intersects=true;}
      }
    }
  }
  return vec2u(select(0u,1u,intersects),select(0u,1u,contained&&!touched));
}`;
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      ...Object.values(props.output)
    ]);
    return props.refinement === 'level-synchronous'
      ? this.getLevelSynchronousNodes(graph)
      : this.getSerialNodes(graph);
  }

  /** The depth-first traversal in one invocation. */
  private getSerialNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUS2PolygonCovering',
        variant: props.mode ?? 'exterior',
        invocationCount: 1,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'cells', view: props.output.cells, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          ...(props.output.visitedCount
            ? [
                {
                  name: 'visited',
                  view: props.output.visitedCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        declarations: this.getDeclarations(),
        body: `var failure=0u;var required=0u;var visitedCount=0u;
for(var ring=0u;ring<${props.ringOffsets.length - 1}u;ring++){let begin=offsets[offsetsOffset+ring];let end=offsets[offsetsOffset+ring+1u];if(begin>end||end>${props.positions.length}u||end-begin<3u){failure=1u;break;}
  var center=vec3f(0.0);for(var vertex=begin;vertex<end;vertex++){let point=coverPosition(vertex);let squared=dot(point,point);if((bitcast<u32>(squared)&0x7f800000u)==0x7f800000u||abs(squared-1.0)>1e-4){failure=1u;}center+=point;}
  ${props.hemisphereCenter ? `center = vec3f(${props.hemisphereCenter.join(',')});` : ''}
  if(length(center)<=COVER_TOLERANCE){failure=1u;break;}center=normalize(center);
  for(var vertex=begin;vertex<end&&failure==0u;vertex++){
    let next=select(vertex+1u,begin,vertex+1u==end);let first=coverPosition(vertex);let second=coverPosition(next);let normal=cross(first,second);
    if(dot(center,first)<=COVER_TOLERANCE||length(normal)<=COVER_TOLERANCE){failure=1u;break;}
    // A crossing or nonadjacent touch cannot define a simple ring or a guaranteed cover.
    for(var other=vertex+1u;other<end;other++){
      let otherNext=select(other+1u,begin,other+1u==end);if(other==next||otherNext==vertex){continue;}
      let third=coverPosition(other);let fourth=coverPosition(otherNext);let crossing=cross(normal,cross(third,fourth));
      if(length(crossing)>COVER_TOLERANCE*COVER_TOLERANCE){
        let intersection=normalize(crossing);
        if((coverOnArc(first,second,intersection)&&coverOnArc(third,fourth,intersection))||(coverOnArc(first,second,-intersection)&&coverOnArc(third,fourth,-intersection))){failure=1u;}
      }else if(coverOnArc(first,second,third)||coverOnArc(first,second,fourth)){failure=1u;}
    }
  }
}
var stack:array<vec2u,${SERIAL_STACK_SIZE}>;var size=6u;
for(var face=0u;face<6u;face++){stack[5u-face]=vec2u((2u*face+1u)<<28u,0u);}
loop {if(size==0u||failure!=0u){break;}if(visitedCount>=${props.maximumVisitedCells ?? GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS}u){failure|=4u;break;}size-=1u;let cell=stack[size];visitedCount+=1u;let classification=coverClassify(cell);if(classification.x==0u){continue;}let level=dggs_s2_get_level(cell);
  let emit=level>=${props.minimumLevel ?? 0}u&&(classification.y!=0u||(${props.mode !== 'interior'}&&level==${props.maximumLevel}u));
  if(emit){if(required<${props.capacity}u){cells[cellsOffset+required*2u]=cell.y;cells[cellsOffset+required*2u+1u]=cell.x;}required+=1u;}
  else if(level<${props.maximumLevel}u){let marker=dggs_u64_shift_left(vec2u(0u,1u),2u*(30u-level));let childMarker=dggs_u64_shift_right(marker,2u);let first=dggs_u64_add(dggs_u64_subtract(cell,marker),childMarker);let step=dggs_u64_shift_left(childMarker,1u);var child=first;for(var offset=0u;offset<4u;offset++){stack[size+3u-offset]=child;child=dggs_u64_add(child,step);}size+=4u;}
}
for(var slot=min(required,${props.capacity}u);slot<${props.capacity}u;slot++){cells[cellsOffset+slot*2u]=0u;cells[cellsOffset+slot*2u+1u]=0u;}
if(required>${props.capacity}u){failure|=2u;}counts[countsOffset]=min(required,${props.capacity}u);counts[countsOffset+1u]=required;status[statusOffset]=failure;
${props.output.visitedCount ? 'visited[visitedOffset]=visitedCount;' : ''}`
      })
    ];
  }

  /**
   * Level-synchronous traversal. The frontier is one ordered list of cells: a cell of the
   * round's level is pending, a coarser cell was emitted by an earlier round and is carried. Each
   * round classifies the list (drop 0, keep 1, split 4 children), scans the counts and writes the
   * next list, so cells stay in ascending ID order without a sort. After `maximumLevel + 1` rounds
   * every cell is final.
   */
  private getLevelSynchronousNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const operation = 'GPUS2PolygonCovering';
    const variant = `${props.mode ?? 'exterior'}-level-synchronous`;
    const ringCount = props.ringOffsets.length - 1;
    const positionCount = props.positions.length;
    const roundCount = props.maximumLevel + 1;
    const frontierCapacity =
      props.maximumVisitedCells ?? GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS;
    const budget = frontierCapacity;
    const lists = [
      createTransientView(graph, `${id}-frontier-a`, 'uint32x2', frontierCapacity),
      createTransientView(graph, `${id}-frontier-b`, 'uint32x2', frontierCapacity)
    ];
    const counts = createTransientView(graph, `${id}-round-counts`, 'uint32', frontierCapacity);
    const offsets = createTransientView(graph, `${id}-round-offsets`, 'uint32', frontierCapacity);
    // Row r holds the list length that round r reads; the last row is the length of the cover.
    const totals = createTransientView(graph, `${id}-round-totals`, 'uint32', roundCount + 1);
    // Word 0 collects status bits (1 invalid, 4 budget); word 1 counts visited cells.
    const control = createTransientView(graph, `${id}-control`, 'uint32', 2);
    const ringCenters = createTransientView(graph, `${id}-ring-centers`, 'float32x3', ringCount);
    const declarations = this.getDeclarations();
    const positionBinding = {
      name: 'positions',
      view: props.positions,
      type: 'f32',
      access: 'read'
    } as const;
    const offsetBinding = {
      name: 'offsets',
      view: props.ringOffsets,
      type: 'u32',
      access: 'read'
    } as const;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-start`,
        operation,
        variant,
        invocationCount: 1,
        bindings: [
          {name: 'list', view: lists[0], type: 'u32', access: 'read_write'},
          {name: 'totals', view: totals, type: 'u32', access: 'read_write'},
          {name: 'control', view: control, type: 'u32', access: 'read_write'}
        ],
        // The six cube-face cells, in face order, are the first list.
        body: `for (var face = 0u; face < 6u; face++) {
  list[listOffset + face * 2u] = 0u; list[listOffset + face * 2u + 1u] = (2u * face + 1u) << 28u;
}
totals[totalsOffset] = 6u; control[controlOffset] = 0u; control[controlOffset + 1u] = 0u;`
      }),
      // Per ring: shape, unit vertices and hemisphere center, one invocation per ring.
      createWGSLKernelNode(graph, {
        id: `${id}-rings`,
        operation,
        variant,
        invocationCount: ringCount,
        bindings: [
          positionBinding,
          offsetBinding,
          {name: 'centers', view: ringCenters, type: 'f32', access: 'read_write'},
          {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `let begin = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
if (begin > end || end > ${positionCount}u || end - begin < 3u) { atomicOr(&control[controlOffset], 1u); return; }
var center = vec3f(0.0); var invalid = false;
for (var vertex = begin; vertex < end; vertex++) {
  let point = coverPosition(vertex); let squared = dot(point, point);
  if ((bitcast<u32>(squared) & 0x7f800000u) == 0x7f800000u || abs(squared - 1.0) > 1e-4) { invalid = true; }
  center += point;
}
${props.hemisphereCenter ? `center = vec3f(${props.hemisphereCenter.join(',')});` : ''}
if (invalid || length(center) <= COVER_TOLERANCE) { atomicOr(&control[controlOffset], 1u); return; }
center = normalize(center);
centers[centersOffset + index * 3u] = center.x; centers[centersOffset + index * 3u + 1u] = center.y; centers[centersOffset + index * 3u + 2u] = center.z;`
      }),
      // Per edge: nonzero length, inside the hemisphere, and no crossing or touch with a later
      // nonadjacent edge of the same ring. This is the O(V^2) check, one edge per invocation.
      createWGSLKernelNode(graph, {
        id: `${id}-edges`,
        operation,
        variant,
        invocationCount: positionCount,
        bindings: [
          positionBinding,
          offsetBinding,
          {name: 'centers', view: ringCenters, type: 'f32', access: 'read'},
          {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `if (atomicLoad(&control[controlOffset]) != 0u) { return; }
var ring = 0u; var begin = 0u; var end = 0u; var found = false;
for (var candidate = 0u; candidate < ${ringCount}u; candidate++) {
  let first = offsets[offsetsOffset + candidate]; let last = offsets[offsetsOffset + candidate + 1u];
  if (index >= first && index < last) { ring = candidate; begin = first; end = last; found = true; break; }
}
if (!found) { return; }
let center = vec3f(centers[centersOffset + ring * 3u], centers[centersOffset + ring * 3u + 1u], centers[centersOffset + ring * 3u + 2u]);
let vertex = index; let next = select(vertex + 1u, begin, vertex + 1u == end);
let first = coverPosition(vertex); let second = coverPosition(next); let normal = cross(first, second);
if (dot(center, first) <= COVER_TOLERANCE || length(normal) <= COVER_TOLERANCE) { atomicOr(&control[controlOffset], 1u); return; }
for (var other = vertex + 1u; other < end; other++) {
  let otherNext = select(other + 1u, begin, other + 1u == end); if (other == next || otherNext == vertex) { continue; }
  let third = coverPosition(other); let fourth = coverPosition(otherNext); let crossing = cross(normal, cross(third, fourth));
  if (length(crossing) > COVER_TOLERANCE * COVER_TOLERANCE) {
    let intersection = normalize(crossing);
    if ((coverOnArc(first, second, intersection) && coverOnArc(third, fourth, intersection)) || (coverOnArc(first, second, -intersection) && coverOnArc(third, fourth, -intersection))) { atomicOr(&control[controlOffset], 1u); }
  } else if (coverOnArc(first, second, third) || coverOnArc(first, second, fourth)) { atomicOr(&control[controlOffset], 1u); }
}`
      })
    ];
    for (let round = 0; round < roundCount; round++) {
      const input = lists[round % 2]!;
      const output = lists[(round + 1) % 2]!;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-classify-${round}`,
          operation,
          variant,
          invocationCount: frontierCapacity,
          bindings: [
            positionBinding,
            offsetBinding,
            {name: 'list', view: input, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
            {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'},
            {name: 'totals', view: totals, type: 'u32', access: 'read'}
          ],
          declarations: `override ROUND: u32;\n${declarations}`,
          constants: {ROUND: round},
          body: `let listCount = totals[totalsOffset + ROUND];
var count = 0u;
if (index < listCount) {
  // A list longer than the frontier capacity cannot be written whole: that is exhaustion.
  if (listCount > ${frontierCapacity}u) { atomicOr(&control[controlOffset], 4u); }
  let cell = vec2u(list[listOffset + index * 2u + 1u], list[listOffset + index * 2u]);
  let level = dggs_s2_get_level(cell);
  if (level < ROUND) { count = 1u; }
  else if (atomicLoad(&control[controlOffset]) == 0u) {
    let visit = atomicAdd(&control[controlOffset + 1u], 1u);
    if (visit >= ${budget}u) { atomicOr(&control[controlOffset], 4u); }
    else {
      let classification = coverClassify(cell);
      if (classification.x != 0u) {
        let emit = level >= ${props.minimumLevel ?? 0}u && (classification.y != 0u || (${props.mode !== 'interior'} && level == ${props.maximumLevel}u));
        if (emit) { count = 1u; } else if (level < ${props.maximumLevel}u) { count = 4u; }
      }
    }
  }
}
counts[countsOffset + index] = count;`
        }),
        ...new GPUScan({
          id: `${id}-scan-${round}`,
          input: counts,
          output: offsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode(graph, {
          id: `${id}-write-${round}`,
          operation,
          variant,
          invocationCount: frontierCapacity,
          bindings: [
            {name: 'source', view: input, type: 'u32', access: 'read'},
            {name: 'destination', view: output, type: 'u32', access: 'read_write'},
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'u32', access: 'read_write'}
          ],
          declarations: `override ROUND: u32;\n${dggs.source}`,
          constants: {ROUND: round},
          body: `let count = counts[countsOffset + index]; let start = offsets[offsetsOffset + index];
// The scan total is the next round's list length; it is unclamped so overflow stays visible.
if (index == ${frontierCapacity - 1}u) { totals[totalsOffset + ROUND + 1u] = start + count; }
if (index < min(totals[totalsOffset + ROUND], ${frontierCapacity}u) && count != 0u) {
  let cell = vec2u(source[sourceOffset + index * 2u + 1u], source[sourceOffset + index * 2u]);
  if (count == 1u) {
    if (start < ${frontierCapacity}u) { destination[destinationOffset + start * 2u] = cell.y; destination[destinationOffset + start * 2u + 1u] = cell.x; }
  } else {
    let level = dggs_s2_get_level(cell);
    let marker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - level)); let childMarker = dggs_u64_shift_right(marker, 2u);
    var child = dggs_u64_add(dggs_u64_subtract(cell, marker), childMarker); let step = dggs_u64_shift_left(childMarker, 1u);
    for (var slot = 0u; slot < 4u; slot++) {
      if (start + slot < ${frontierCapacity}u) { destination[destinationOffset + (start + slot) * 2u] = child.y; destination[destinationOffset + (start + slot) * 2u + 1u] = child.x; }
      child = dggs_u64_add(child, step);
    }
  }
}`
        })
      );
    }
    const cover = lists[roundCount % 2]!;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-cells`,
        operation,
        variant,
        invocationCount: props.capacity,
        bindings: [
          {name: 'list', view: cover, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'control', view: control, type: 'u32', access: 'read'},
          {name: 'cells', view: props.output.cells, type: 'u32', access: 'read_write'}
        ],
        // Slots past the cover, and every slot of an invalid or exhausted traversal, are zero.
        body: `var low = 0u; var high = 0u;
if ((control[controlOffset] & 5u) == 0u && index < min(totals[totalsOffset + ${roundCount}u], ${frontierCapacity}u)) {
  low = list[listOffset + index * 2u]; high = list[listOffset + index * 2u + 1u];
}
cells[cellsOffset + index * 2u] = low; cells[cellsOffset + index * 2u + 1u] = high;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation,
        variant,
        invocationCount: 1,
        bindings: [
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'control', view: control, type: 'u32', access: 'read'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          ...(props.output.visitedCount
            ? [
                {
                  name: 'visited',
                  view: props.output.visitedCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `var failure = control[controlOffset]; var written = 0u; var required = 0u;
if ((failure & 5u) == 0u) {
  required = totals[totalsOffset + ${roundCount}u]; written = min(required, ${props.capacity}u);
  if (required > ${props.capacity}u) { failure |= 2u; }
}
counts[countsOffset] = written; counts[countsOffset + 1u] = required; status[statusOffset] = failure;
${props.output.visitedCount ? `visited[visitedOffset] = min(control[controlOffset + 1u], ${budget}u);` : ''}`
      })
    );
    return nodes;
  }
}
