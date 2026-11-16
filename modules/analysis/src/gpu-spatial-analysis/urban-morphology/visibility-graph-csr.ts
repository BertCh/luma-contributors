// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createSaturatedCountScanNodes} from '../../analysis-core/kernels/scan-total';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {getRecipeView} from '../recipes/expansion-recipe-utils';
import {
  getVisibilityCSRQueryNodes,
  getVisibilityCSRCountDeclarations
} from './visibility-graph-csr-query';
import {validateUrbanStorageByteLength} from './building-tessellation-validation';
import {createVisibilityGraphKernelNode} from './visibility-graph-kernel';
import {validateVisibilityGraphCSRBindings} from './visibility-graph-bindings';
import {
  addUrbanPointGrid,
  getUrbanPointGridShape,
  getUrbanPointGridDeclarations,
  getUrbanPointGridRadiusWalk
} from './visibility-graph-point-grid';
import {classifyUrbanViewpoints, type UrbanFreeSpaceRegion} from './classify-viewpoints';
import {
  addUrbanObstacleBVH,
  getUrbanRayBVHBindings,
  getUrbanRayDeclarations,
  URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT
} from './urban-ray-utils';

/** Linear CSR and grid storage replace the reference's quadratic adjacency. Device limits may lower this. */
export const VISIBILITY_GRAPH_CSR_MAXIMUM_POINT_COUNT = 1000000;
/** Bounds directed CSR storage and uint32 addressing independently of optional network metrics. */
export const VISIBILITY_GRAPH_CSR_MAXIMUM_EDGE_CAPACITY = 16777216;
/** Bounds per-point grid work, including false positives in dense grid cells. */
export const VISIBILITY_GRAPH_CSR_MAXIMUM_GRID_VISITS = 65536;
/** Ordinary city-radius queries need only a small neighbourhood; denser inputs report candidateOverflow. */
export const VISIBILITY_GRAPH_CSR_DEFAULT_GRID_VISITS = 8192;
/** Linear pivot storage supports all sources; the network recipe separately bounds command count. */
export const VISIBILITY_GRAPH_CSR_MAXIMUM_PIVOT_COUNT = VISIBILITY_GRAPH_CSR_MAXIMUM_POINT_COUNT;
/** Fallback sample size when the network recipe's automatic all-source run exceeds its command budget. */
export const VISIBILITY_GRAPH_CSR_DEFAULT_PIVOT_COUNT = 32;
/** Conservative node allowance for the published grid, BVH, saturated scan and reductions. */
export const VISIBILITY_GRAPH_CSR_COMMAND_NODE_ALLOWANCE = 1024;
/** Bounds the number of radius-query dispatches in a recipe's command budget. */
export const VISIBILITY_GRAPH_CSR_MAXIMUM_QUERY_CHUNK_COUNT = 1024;
/** Four times tolerance is emitted by the indexed segment traversal and must remain finite in float32. */
export const VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE = (2 ** 128 - 2 ** 104) / 4;
/** Runtime parameter order is maximumDistance, tolerance. Tolerance must match the obstacle index build. */
export const GPU_VISIBILITY_GRAPH_CSR_PARAMETER_LENGTH = 2;

export function getGPUVisibilityGraphCSRParameterValues(values: {
  maximumDistance: number;
  tolerance?: number;
}): Float32Array {
  return new Float32Array([values.maximumDistance, values.tolerance ?? 1e-6]);
}

export type GPUVisibilityGraphCSRStatus = {
  /** Directed CSR slots, or zero when the graph cannot be published symmetrically. */
  count: GraphDataView<'uint32'>;
  /** Directed slot requirement, saturated at uint32 maximum. Unknown on candidateOverflow (0xffffffff). */
  requiredCount: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  candidateOverflow: GraphDataView<'uint32'>;
  /** Invalid viewpoint rows, including a parameter mismatch invalidating every viewpoint. */
  invalidCount: GraphDataView<'uint32'>;
};

export type GPUVisibilityGraphCSROutput = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  /** Unit hop costs; unused capacity slots have NaN weights and 0xffffffff neighbors. */
  weights: GraphDataView<'float32'>;
  degree: GraphDataView<'uint32'>;
  /** Candidate rows examined by the count query, at most maximumGridVisits. */
  gridVisits: GraphDataView<'uint32'>;
  viewpointClassifications: GraphDataView<'uint32'>;
  status: GPUVisibilityGraphCSRStatus;
};

export type GPUVisibilityGraphCSRProps = {
  id?: string;
  mode: 'csr';
  /** Single packed planar views. Input contents and optional parameters may change each execution. */
  positions: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  spatialContext?: GPUSpatialContext;
  maximumDistance: number;
  /** Positive float32, at most VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE for indexed obstacles. */
  tolerance?: number;
  /** Optional radius/tolerance parameters. A changed tolerance invalidates the run until rebuilt. */
  parameters?: GraphDataView<'float32'>;
  /** Capacity in DIRECTED slots: every undirected pair occupies two slots. */
  edgeCapacity: number;
  maximumGridVisits?: number;
  /** Dispatch partition only, rounded up to a multiple of 64 for aligned runtime parameter views. No shader specialization. */
  queryChunkSize?: number;
  region?: UrbanFreeSpaceRegion;
  output?: Partial<Omit<GPUVisibilityGraphCSROutput, 'status'>> & {
    status?: Partial<GPUVisibilityGraphCSRStatus>;
  };
};

/** Sparse symmetric adjacency and degree only. For sampled centrality use addVisibilityGraphRecipe from @luma-contributors/analysis/gpu-network. @internal */
export function prepareVisibilityGraphCSRRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphCSRProps
) {
  const id = props.id ?? 'visibility-graph';
  const count = props.positions.length;
  validateCapacity(id, 'positions.length', count, 1, VISIBILITY_GRAPH_CSR_MAXIMUM_POINT_COUNT);
  const tolerance = props.tolerance ?? 1e-6;
  const maximumGridVisits = props.maximumGridVisits ?? VISIBILITY_GRAPH_CSR_DEFAULT_GRID_VISITS;
  const queryChunkSize = Math.ceil((props.queryChunkSize ?? count) / 64) * 64;
  validateCapacity(
    id,
    'queryChunkSize',
    props.queryChunkSize ?? count,
    1,
    VISIBILITY_GRAPH_CSR_MAXIMUM_POINT_COUNT
  );
  const queryChunkCount = Math.ceil(count / queryChunkSize);
  validateCapacity(
    id,
    'queryChunkCount',
    queryChunkCount,
    1,
    VISIBILITY_GRAPH_CSR_MAXIMUM_QUERY_CHUNK_COUNT
  );
  const queryChunkStride = Math.ceil((queryChunkSize + 4) / 64) * 64;
  const spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
  validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
  validateSinglePackedView(props.segments, ['float32x4'], `${id} segments`);
  validateCapacity(
    id,
    'segments.length',
    props.segments.length,
    1,
    URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT
  );
  validateCapacity(
    id,
    'edgeCapacity',
    props.edgeCapacity,
    1,
    VISIBILITY_GRAPH_CSR_MAXIMUM_EDGE_CAPACITY
  );
  validateCapacity(
    id,
    'maximumGridVisits',
    maximumGridVisits,
    1,
    VISIBILITY_GRAPH_CSR_MAXIMUM_GRID_VISITS
  );
  for (const [name, value] of [
    ['maximumDistance', props.maximumDistance],
    ['tolerance', tolerance]
  ] as const) {
    validateRequirement(
      id,
      Math.fround(value) > 0 && Number.isFinite(Math.fround(value)),
      `${name} must be a positive finite float32 number`
    );
  }
  validateRequirement(
    id,
    tolerance <= VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE,
    `tolerance must be at most VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE (${VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE}) so four times tolerance is finite in float32`
  );
  if (props.parameters) {
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateViewLength(
      id,
      'parameters',
      props.parameters,
      GPU_VISIBILITY_GRAPH_CSR_PARAMETER_LENGTH
    );
  }
  validateVisibilityGraphCSRBindings(
    graph,
    id,
    {
      positions: props.positions,
      segments: props.segments,
      parameters: props.parameters
    },
    props.output
  );
  let leafCapacity = 2;
  while (leafCapacity < props.segments.length) leafCapacity *= 2;
  // Includes every derived row-proportional binding before any resource or node declaration.
  for (const [name, bytes] of [
    ['point grid', getUrbanPointGridShape(count).byteLength],
    ['CSR offsets', (count + 1) * 4],
    ['CSR neighbors and weights', props.edgeCapacity * 4],
    ['count scan uint64 scratch', count * 8],
    ['query records', count * 12],
    ['query parameters and offsets', queryChunkCount * queryChunkStride * 4],
    ['prepared positions', count * 8],
    ['viewpoint classification scratch', count * 4],
    ['segment bounds', props.segments.length * 8],
    ['BVH node bounds', (2 * leafCapacity - 1) * 8],
    ['BVH leaf rows', leafCapacity * 4]
  ] as const)
    validateUrbanStorageByteLength(
      graph.device,
      id,
      name,
      bytes,
      'reduce the input length, capacity or laneCount'
    );

  const estimatedCommandNodeCount =
    VISIBILITY_GRAPH_CSR_COMMAND_NODE_ALLOWANCE + 2 * queryChunkCount;
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  for (const [name, format, length] of [
    ['offsets', 'uint32', count + 1],
    ['neighbors', 'uint32', props.edgeCapacity],
    ['weights', 'float32', props.edgeCapacity],
    ['degree', 'uint32', count],
    ['gridVisits', 'uint32', count],
    ['viewpointClassifications', 'uint32', count]
  ] as const) {
    const view = props.output?.[name];
    if (view) {
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      validateViewLength(id, `output.${name}`, view, length);
    }
  }
  for (const [name, view] of Object.entries(props.output?.status ?? {})) {
    if (view) {
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      validateViewLength(id, `output.status.${name}`, view, 1);
    }
  }
  return {
    estimatedCommandNodeCount,
    // The optional count belongs to the higher-layer composition, never to public adjacency callers.
    addNodes(additionalInvalidCount?: GraphDataView<'uint32'>) {
      const output = props.output;
      const getView = <Format extends 'uint32' | 'float32'>(
        name: string,
        format: Format,
        length: number,
        supplied?: GraphDataView<Format>
      ) => getRecipeView(graph, `${id}-csr-${name}`, format, length, supplied);
      const offsets = getView('offsets', 'uint32', count + 1, output?.offsets);
      const neighbors = getView('neighbors', 'uint32', props.edgeCapacity, output?.neighbors);
      const weights = getView('weights', 'float32', props.edgeCapacity, output?.weights);
      const degree = getView('degree', 'uint32', count, output?.degree);
      const gridVisits = getView('grid-visits', 'uint32', count, output?.gridVisits);
      const classifications = getView(
        'classifications',
        'uint32',
        count,
        output?.viewpointClassifications
      );
      const scalar = (name: keyof GPUVisibilityGraphCSRStatus) =>
        getView(name, 'uint32', 1, output?.status?.[name]);
      const status: GPUVisibilityGraphCSRStatus = {
        count: scalar('count'),
        requiredCount: scalar('requiredCount'),
        overflow: scalar('overflow'),
        candidateOverflow: scalar('candidateOverflow'),
        invalidCount: scalar('invalidCount')
      };
      const parameters =
        props.parameters ??
        getView('parameters', 'float32', GPU_VISIBILITY_GRAPH_CSR_PARAMETER_LENGTH);
      const topology = {offsets, neighbors, weights};
      if (!props.parameters) {
        graph.add(
          createVisibilityGraphKernelNode(graph, {
            id: `${id}-csr-default-parameters`,
            operation: id,
            invocationCount: 1,
            bindings: [{name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}],
            body: `parameters[parametersOffset]=${getWGSLFloatLiteral(props.maximumDistance)};parameters[parametersOffset+1u]=${getWGSLFloatLiteral(tolerance)};`
          })
        );
      }
      const obstacleBVH = addUrbanObstacleBVH(graph, {
        id: `${id}-csr-obstacles`,
        operation: id,
        segments: props.segments,
        tolerance
      });
      const classified = classifyUrbanViewpoints(graph, {
        id: `${id}-csr`,
        positions: props.positions,
        segments: props.segments,
        tolerance,
        obstacleBVH,
        region: props.region
      });
      // The shared classifier requires separate input/output handles. Publish through a copy so
      // caller classifications can occupy any validated disjoint range beside input fields.
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-publish-classifications`,
          operation: id,
          invocationCount: count,
          bindings: [
            {name: 'source', view: classified, type: 'u32', access: 'read'},
            {name: 'classifications', view: classifications, type: 'u32', access: 'read_write'}
          ],
          body: 'classifications[classificationsOffset+index]=source[sourceOffset+index];'
        })
      );
      const prepared = createTransientView(graph, `${id}-csr-prepared`, 'float32x2', count);
      const invalidFlags = getView('invalid-flags', 'uint32', count);
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-prepare`,
          operation: id,
          invocationCount: count,
          declarations: IS_FINITE_VALUE_WGSL + GET_NAN_WGSL,
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'classifications', view: classifications, type: 'u32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'prepared', view: prepared, type: 'f32', access: 'read_write'},
            {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read_write'}
          ],
          body: `let radius=parameters[parametersOffset];let inputTolerance=parameters[parametersOffset+1u];
let valid=classifications[classificationsOffset+index]==0u&&isFiniteValue(radius)&&radius>0.0&&inputTolerance==${getWGSLFloatLiteral(tolerance)};
invalidFlags[invalidFlagsOffset+index]=u32(!valid);
prepared[preparedOffset+2u*index]=select(getNaN(),positions[positionsOffset+2u*index],valid);
prepared[preparedOffset+2u*index+1u]=select(getNaN(),positions[positionsOffset+2u*index+1u],valid);`
        })
      );
      graph.add(
        new GPUReduction({
          id: `${id}-csr-invalid`,
          input: invalidFlags,
          output: status.invalidCount,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
      if (additionalInvalidCount)
        graph.add(
          createVisibilityGraphKernelNode(graph, {
            id: `${id}-csr-merge-invalid`,
            operation: id,
            invocationCount: 1,
            bindings: [
              {name: 'pivots', view: additionalInvalidCount, type: 'u32', access: 'read'},
              {name: 'invalidCount', view: status.invalidCount, type: 'u32', access: 'read_write'}
            ],
            body: 'invalidCount[invalidCountOffset]+=pivots[pivotsOffset];'
          })
        );
      const grid = addUrbanPointGrid(graph, {id: `${id}-csr-grid`, positions: prepared});
      const counts = getView('counts', 'uint32', count);
      const scanOffsets = getView('scan-offsets', 'uint32', count);
      const candidateFlags = getView('candidate-flags', 'uint32', count);
      // Counts and traversal flags share a binding, leaving eight bindings in the BVH query kernels.
      const queryRecords = getView('query-records', 'uint32', 3 * count);
      const total = getView('total', 'uint32', 1);
      const queryDeclarations =
        IS_FINITE_VALUE_WGSL +
        getUrbanPointGridDeclarations(grid) +
        getUrbanRayDeclarations(props.segments.length, tolerance, obstacleBVH) +
        `
fn readPoint(row:u32)->vec2<f32>{return vec2<f32>(positions[positionsOffset+2u*row],positions[positionsOffset+2u*row+1u]);}
// Canonical direction makes the float32 visibility predicate identical for both CSR directions.
fn isVisible(first:u32,second:u32)->bool{
 let lower=min(first,second);let upper=max(first,second);
 let origin=readPoint(lower);let displacement=readPoint(upper)-origin;let separation=length(displacement);
 if(first==second||!isFiniteValue(separation)||separation<=${getWGSLFloatLiteral(tolerance)}||separation>radiusValue){return false;}
 return castRay(origin,displacement/separation,separation).y<0.0;
}
var<private> radiusValue:f32;
`;
      const queryBindings = [
        {name: 'positions', view: prepared, type: 'f32', access: 'read'} as const,
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'} as const,
        ...getUrbanRayBVHBindings(obstacleBVH),
        {name: 'pointGrid', view: grid.data, type: 'u32', access: 'read'} as const
      ];
      // Runtime radius is copied into queryRecords' header to avoid a ninth binding in the scatter pass.
      const queryState = getView('query-state', 'uint32', queryChunkCount * queryChunkStride);
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-query-parameters`,
          operation: id,
          invocationCount: queryChunkCount,
          bindings: [
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'queryState', view: queryState, type: 'u32', access: 'read_write'}
          ],
          body: `let base=queryStateOffset+index*${queryChunkStride}u;
queryState[base]=bitcast<u32>(parameters[parametersOffset]);queryState[base+2u]=index*${queryChunkSize}u;queryState[base+3u]=min(${queryChunkSize}u,${count}u-index*${queryChunkSize}u);`
        })
      );
      graph.add(
        getVisibilityCSRQueryNodes(graph, {
          id: `${id}-csr-count`,
          count,
          chunkSize: queryChunkSize,
          chunkStride: queryChunkStride,
          queryState,
          declarations:
            queryDeclarations + getVisibilityCSRCountDeclarations(grid, maximumGridVisits),
          bindings: [
            ...queryBindings,
            {name: 'records', view: queryRecords, type: 'u32', access: 'read_write'}
          ],
          body: `radiusValue=bitcast<f32>(queryState[queryStateOffset]);let counted=countVisibleNeighbors(index);
records[recordsOffset+3u*index]=counted.x;records[recordsOffset+3u*index+1u]=counted.y;records[recordsOffset+3u*index+2u]=counted.z;`
        })
      );
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-count-fields`,
          operation: id,
          invocationCount: count,
          bindings: [
            {name: 'records', view: queryRecords, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
            {name: 'flags', view: candidateFlags, type: 'u32', access: 'read_write'},
            {name: 'visits', view: gridVisits, type: 'u32', access: 'read_write'}
          ],
          body: 'counts[countsOffset+index]=records[recordsOffset+3u*index];flags[flagsOffset+index]=records[recordsOffset+3u*index+1u];visits[visitsOffset+index]=records[recordsOffset+3u*index+2u];'
        })
      );
      graph.add(
        new GPUReduction({
          id: `${id}-csr-candidate-overflow`,
          input: candidateFlags,
          output: status.candidateOverflow,
          operation: 'max'
        }).getCommandNodes(graph)
      );
      graph.add(
        createSaturatedCountScanNodes(graph, {
          id: `${id}-csr-scan`,
          operation: id,
          counts,
          offsets: scanOffsets,
          rowCount: count,
          total
        })
      );
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-finalize-offsets`,
          operation: id,
          invocationCount: count + 1,
          bindings: [
            {name: 'scanOffsets', view: scanOffsets, type: 'u32', access: 'read'},
            {name: 'total', view: total, type: 'u32', access: 'read'},
            {
              name: 'candidateOverflow',
              view: status.candidateOverflow,
              type: 'u32',
              access: 'read'
            },
            {name: 'invalidCount', view: status.invalidCount, type: 'u32', access: 'read'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
            {name: 'queryState', view: queryState, type: 'u32', access: 'read_write'}
          ],
          body: `let usable=candidateOverflow[candidateOverflowOffset]==0u&&invalidCount[invalidCountOffset]==0u&&total[totalOffset]<=${props.edgeCapacity}u;
var row=total[totalOffset];if(index<${count}u){row=scanOffsets[scanOffsetsOffset+index];}
offsets[offsetsOffset+index]=select(0u,row,usable);
if(index<${count}u){let chunk=index/${queryChunkSize}u;let localRow=index%${queryChunkSize}u;queryState[queryStateOffset+chunk*${queryChunkStride}u+4u+localRow]=row;}
if(index<${queryChunkCount}u){queryState[queryStateOffset+index*${queryChunkStride}u+1u]=u32(usable);}`
        })
      );
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-clear-slots`,
          operation: id,
          invocationCount: props.edgeCapacity,
          bindings: [
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
            {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
          ],
          body: 'neighbors[neighborsOffset+index]=0xffffffffu;weights[weightsOffset+index]=1.0;'
        })
      );
      graph.add(
        getVisibilityCSRQueryNodes(graph, {
          id: `${id}-csr-scatter`,
          count,
          chunkSize: queryChunkSize,
          chunkStride: queryChunkStride,
          queryState,
          declarations: queryDeclarations,
          bindings: [
            ...queryBindings,
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'}
          ],
          body: `if(queryState[queryStateOffset+1u]==0u){return;}
radiusValue=bitcast<f32>(queryState[queryStateOffset]);let center=readPoint(index);var written=0u;
${getUrbanPointGridRadiusWalk(grid, 'center', 'radiusValue', 'other', `if(isVisible(index,other)){neighbors[neighborsOffset+queryState[queryStateOffset+4u+localIndex]+written]=other;written++;}`)}`
        })
      );
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-weights`,
          operation: id,
          declarations: GET_NAN_WGSL,
          invocationCount: props.edgeCapacity,
          bindings: [
            {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights, type: 'f32', access: 'read_write'}
          ],
          body: `weights[weightsOffset+index]=select(getNaN(),1.0,neighbors[neighborsOffset+index]<${count}u);`
        })
      );
      graph.add(
        createVisibilityGraphKernelNode(graph, {
          id: `${id}-csr-status`,
          operation: id,
          invocationCount: count,
          bindings: [
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'total', view: total, type: 'u32', access: 'read'},
            {
              name: 'candidateOverflow',
              view: status.candidateOverflow,
              type: 'u32',
              access: 'read'
            },
            {name: 'degree', view: degree, type: 'u32', access: 'read_write'},
            {name: 'count', view: status.count, type: 'u32', access: 'read_write'},
            {name: 'requiredCount', view: status.requiredCount, type: 'u32', access: 'read_write'},
            {name: 'overflow', view: status.overflow, type: 'u32', access: 'read_write'}
          ],
          body: `degree[degreeOffset+index]=offsets[offsetsOffset+index+1u]-offsets[offsetsOffset+index];
if(index==0u){let candidates=candidateOverflow[candidateOverflowOffset]!=0u;
count[countOffset]=offsets[offsetsOffset+${count}u];requiredCount[requiredCountOffset]=select(total[totalOffset],0xffffffffu,candidates);
overflow[overflowOffset]=u32(!candidates&&total[totalOffset]>${props.edgeCapacity}u);}`
        })
      );
      return {
        outputs: {
          offsets,
          neighbors,
          weights,
          degree,
          gridVisits,
          viewpointClassifications: classifications,
          status
        },
        topology,
        status,
        estimatedCommandNodeCount,
        provenance: {
          mode: 'csr' as const,
          spatialContext,
          parameters,
          positions: props.positions,
          segments: props.segments,
          maximumDistance: props.maximumDistance,
          tolerance
        }
      };
    }
  };
}

/** Declares spatial adjacency and degree; centrality is composed by the gpu-network entry. @internal */
export function addVisibilityGraphCSRRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphCSRProps
) {
  return prepareVisibilityGraphCSRRecipe(graph, props).addNodes();
}
