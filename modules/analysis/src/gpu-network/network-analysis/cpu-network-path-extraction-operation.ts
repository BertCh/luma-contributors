// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_NETWORK_PATH_NO_EDGE} from './gpu-network-path-extraction';
import {
  GPUNetworkPathExtractionOperation,
  snapshotNetworkPathExtractionOperationProps,
  type NetworkPathExtractionOperationProps
} from './gpu-network-path-extraction-operation';

/** Compact counts and unclamped offsets have one uint32 word per value. */
export const MAXIMUM_PATH_OUTPUT_ROW_COUNT = 0xffffffff;

/** Packed arrays of a predecessor forest and optional CSR edge resolver. */
type CPUNetworkPathExtractionRows = {
  predecessors: Uint32Array;
  costs: Float32Array;
  targets: Uint32Array;
  targetCount?: number | Uint32Array;
  predecessorsComplete?: Uint32Array;
  nodeIds?: Uint32Array;
  /** Maximum nodes per path, binding on CPU and GPU. Defaults to 1024. */
  maximumPathLength?: number;
  capacity: number;
};

/** Authoritative CSR topology for resolving predecessor slots to edge IDs. */
type CPUNetworkPathExtractionEdges = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  edgeIds?: Uint32Array;
  capacity: number;
};

/** Edge output requires the per-node CSR slots emitted by the search. */
export type CPUNetworkPathExtractionInput = CPUNetworkPathExtractionRows &
  (
    | {edges?: undefined; predecessorEdges?: Uint32Array}
    | {edges: CPUNetworkPathExtractionEdges; predecessorEdges: Uint32Array}
  );

/** Compact rows plus one-row status values, with unclamped per-target offsets. */
export type CPUNetworkPathExtractionCompactResult = {
  ids: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  candidateOverflow: Uint32Array;
  unrepresentable: Uint32Array;
  lineOffsets: Uint32Array;
};

/** Packed nodes, optional edges, and per-target costs/found flags. */
export type CPUNetworkPathExtractionResult = CPUNetworkPathExtractionCompactResult & {
  nodeRows: Uint32Array;
  pathCosts: Float32Array;
  pathFound: Uint32Array;
  edges?: CPUNetworkPathExtractionCompactResult & {edgeRows: Uint32Array};
};

/**
 * Extracts complete root-to-target paths with explicit cycle detection, packs them in target
 * order, and optionally resolves links to CSR slots or stable IDs. Invalid/inactive/unreached
 * targets and broken or cyclic chains have empty paths. A NONE root requires complete predecessors.
 * Edge extraction requires predecessorEdges from the search and resolves them through validated
 * CSR slots. Counts clamp to capacity; offsets remain unclamped.
 *
 * What differs from the GPU contributor:
 * - Cycles are detected exactly instead of waiting for the walk budget; they report candidate loss.
 *   maximumPathLength binds both forms (default 1024); CPU accepts budgets above the GPU 65536 cap.
 *   Only final output capacity sets overflow; candidateOverflow reports bounded walk loss.
 * - The safe bound is activeTargetUpperBound × min(nodeCount, maximumPathLength). Larger
 *   batches are admitted but actual totals above 4294967295 reject before publication.
 * - Unused node ID capacity is zero-filled; the GPU leaves rows beyond count unwritten. CPU edge
 *   padding is 0xffffffff for nonempty target tables and zero for empty tables; the GPU initializes
 *   edge padding only for nonempty target tables. Only rows before count are compact results.
 *
 * @throws If costs, nodeIds, targetCount, predecessorsComplete or edge arrays are inconsistent.
 */
export function computeNetworkPathExtractionOnCPU(
  input: CPUNetworkPathExtractionInput & {
    edges: NonNullable<CPUNetworkPathExtractionInput['edges']>;
    predecessorEdges: Uint32Array;
  }
): CPUNetworkPathExtractionResult & {edges: NonNullable<CPUNetworkPathExtractionResult['edges']>};
export function computeNetworkPathExtractionOnCPU(
  input: CPUNetworkPathExtractionInput & {edges?: undefined}
): Omit<CPUNetworkPathExtractionResult, 'edges'> & {edges?: undefined};
export function computeNetworkPathExtractionOnCPU(
  input: CPUNetworkPathExtractionInput
): CPUNetworkPathExtractionResult;
export function computeNetworkPathExtractionOnCPU(
  input: CPUNetworkPathExtractionInput
): CPUNetworkPathExtractionResult {
  const {predecessors, predecessorEdges, costs, targets, nodeIds, predecessorsComplete, edges} =
    input;
  validateRequirement(
    'network-path-extraction',
    !edges || Boolean(predecessorEdges),
    'edges requires predecessorEdges from the search'
  );
  const nodeCount = predecessors.length;
  if (nodeCount < 1) throw new Error('predecessors must hold at least one node');
  if (costs.length !== nodeCount) throw new Error('costs length must equal predecessors length');
  if (nodeIds && nodeIds.length !== nodeCount)
    throw new Error('nodeIds length must equal predecessors length');
  if (predecessorsComplete && predecessorsComplete.length !== 1)
    throw new Error('predecessorsComplete must hold one row');
  if (typeof input.targetCount === 'object' && input.targetCount.length !== 1)
    throw new Error('targetCount must hold one row');
  if (typeof input.targetCount === 'number')
    validateCapacity(
      'network-path-extraction',
      'targetCount',
      input.targetCount,
      0,
      targets.length
    );
  validateCapacity('network-path-extraction', 'capacity', input.capacity, 0);
  if (input.maximumPathLength !== undefined)
    validateCapacity('network-path-extraction', 'maximumPathLength', input.maximumPathLength);
  if (predecessorEdges) {
    validateRequirement(
      'network-path-extraction',
      predecessorEdges.length === nodeCount,
      `predecessorEdges length must equal predecessors length ${nodeCount}; received ${predecessorEdges.length}`
    );
  }
  const validatedProvenance = new Set<number>();
  const validatePredecessorEdge = (head: number, tail: number) => {
    if (!predecessorEdges || validatedProvenance.has(head)) return;
    const edge = predecessorEdges[head];
    if (tail === GPU_NETWORK_PATH_NO_EDGE) {
      validateRequirement(
        'network-path-extraction',
        edge === GPU_NETWORK_PATH_NO_EDGE,
        `predecessorEdges must use NONE for a root; received ${edge} at node ${head}`
      );
    } else if (
      edges &&
      (tail >= nodeCount ||
        edge < Math.min(edges.offsets[tail], edges.neighbors.length) ||
        edge >= Math.min(edges.offsets[tail + 1], edges.neighbors.length) ||
        edges.neighbors[edge] !== head ||
        !(edges.weights[edge] >= 0) ||
        !Number.isFinite(edges.weights[edge]))
    ) {
      validateRequirement(
        'network-path-extraction',
        false,
        `predecessorEdges must identify a passable edge from the parent to its node; received ${edge} for parent ${tail}, node ${head}`
      );
    }
    validatedProvenance.add(head);
  };
  if (edges) {
    if (edges.offsets.length !== nodeCount + 1)
      throw new Error('edges.offsets must hold predecessors.length + 1 rows');
    if (edges.weights.length !== edges.neighbors.length)
      throw new Error('edges.weights length must equal edges.neighbors length');
    if (edges.edgeIds && edges.edgeIds.length !== edges.neighbors.length)
      throw new Error('edges.edgeIds length must equal edges.neighbors length');
    validateCapacity('network-path-extraction', 'edges.capacity', edges.capacity, 0);
  }
  const makeCompactResult = (
    capacity: number,
    sentinel: number
  ): CPUNetworkPathExtractionCompactResult => ({
    ids: new Uint32Array(capacity).fill(sentinel),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    candidateOverflow: new Uint32Array(1),
    unrepresentable: new Uint32Array(1),
    lineOffsets: new Uint32Array(targets.length + 1)
  });
  const result: CPUNetworkPathExtractionResult = {
    ...makeCompactResult(input.capacity, 0),
    nodeRows: new Uint32Array(input.capacity),
    pathCosts: new Float32Array(targets.length).fill(Infinity),
    pathFound: new Uint32Array(targets.length),
    edges: edges
      ? {
          ...makeCompactResult(edges.capacity, targets.length === 0 ? 0 : GPU_NETWORK_PATH_NO_EDGE),
          edgeRows: new Uint32Array(edges.capacity).fill(
            targets.length === 0 ? 0 : GPU_NETWORK_PATH_NO_EDGE
          )
        }
      : undefined
  };
  const activeCount =
    typeof input.targetCount === 'object'
      ? input.targetCount[0]
      : (input.targetCount ?? targets.length);
  const safeBound =
    Math.min(activeCount, targets.length) * Math.min(nodeCount, input.maximumPathLength ?? 1024) <=
    MAXIMUM_PATH_OUTPUT_ROW_COUNT;
  let nodeTotal = 0;
  let edgeTotal = 0;
  for (let targetRow = 0; targetRow < targets.length; targetRow++) {
    result.lineOffsets[targetRow] = nodeTotal;
    if (result.edges) result.edges.lineOffsets[targetRow] = edgeTotal;
    const target = targets[targetRow];
    if (targetRow >= activeCount || target >= nodeCount || !Number.isFinite(costs[target]))
      continue;
    const reversed: number[] = [];
    const visited = new Set<number>();
    let node = target;
    let found = false;
    while (node < nodeCount && !visited.has(node)) {
      reversed.push(node);
      visited.add(node);
      const predecessor = predecessors[node];
      validatePredecessorEdge(node, predecessor);
      if (predecessor === GPU_NETWORK_PATH_NO_EDGE) {
        found = predecessorsComplete === undefined || predecessorsComplete[0] !== 0;
        break;
      }
      if (predecessor >= nodeCount) break;
      if (reversed.length >= (input.maximumPathLength ?? 1024) || visited.has(predecessor)) {
        result.candidateOverflow[0] = 1;
        if (result.edges) result.edges.candidateOverflow[0] = 1;
        break;
      }
      node = predecessor;
    }
    if (!found) continue;
    const path = reversed.reverse();
    // JS totals are exact here; reject actual unrepresentable totals before typed publication.
    if (!safeBound)
      validateMaximumCount(
        'network-path-extraction',
        'required node rows (split targets into smaller batches)',
        nodeTotal + path.length,
        MAXIMUM_PATH_OUTPUT_ROW_COUNT
      );
    result.pathFound[targetRow] = 1;
    result.pathCosts[targetRow] = costs[target];
    for (let position = 0; position < path.length; position++) {
      if (nodeTotal < input.capacity) {
        result.nodeRows[nodeTotal] = path[position];
        result.ids[nodeTotal] = nodeIds ? nodeIds[path[position]] : path[position];
      }
      nodeTotal++;
      if (edges && result.edges && position + 1 < path.length) {
        const selected = predecessorEdges![path[position + 1]];
        if (edgeTotal < edges.capacity) {
          result.edges.edgeRows[edgeTotal] = selected;
          result.edges.ids[edgeTotal] = edges.edgeIds ? edges.edgeIds[selected] : selected;
        }
        edgeTotal++;
      }
    }
  }
  const finishCompactResult = (compact: CPUNetworkPathExtractionCompactResult, total: number) => {
    compact.lineOffsets[targets.length] = total;
    compact.requiredCount[0] = total;
    compact.count[0] = Math.min(total, compact.ids.length);
    compact.overflow[0] = total > compact.ids.length ? 1 : 0;
  };
  finishCompactResult(result, nodeTotal);
  if (result.edges) finishCompactResult(result.edges, edgeTotal);
  return result;
}

/** Named edge extraction requires the predecessor slot graph value from the search. */
export type CPUNetworkPathExtractionOperationProps = NetworkPathExtractionOperationProps &
  (
    | {edges?: undefined}
    | {
        edges: NonNullable<NetworkPathExtractionOperationProps['edges']>;
        predecessorEdges: string;
      }
  );

/** CPU twin with the same path budget and separate final/candidate status as the GPU. */
export class CPUNetworkPathExtractionOperation extends GPUCPUOperation {
  constructor(originalProps: CPUNetworkPathExtractionOperationProps) {
    const props = snapshotNetworkPathExtractionOperationProps(originalProps);
    const id = props.id ?? 'network-path-extraction';
    validateCapacity(id, 'nodeCount', props.nodeCount);
    validateCapacity(id, 'targetCapacity', props.targetCapacity, 0);
    if (typeof props.targetCount === 'number')
      validateCapacity(id, 'targetCount', props.targetCount, 0, props.targetCapacity);
    if (props.maximumPathLength !== undefined)
      validateCapacity(id, 'maximumPathLength', props.maximumPathLength);
    if (props.edges) {
      validateRequirement(
        id,
        typeof props.predecessorEdges === 'string',
        'edges requires predecessorEdges from the search'
      );
      validateCapacity(id, 'edges.edgeCount', props.edges.edgeCount, 0);
    }
    const operation = new GPUNetworkPathExtractionOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const predecessors = getCPUOperationUint32Rows(id, inputs, 'predecessors');
        const targets = getCPUOperationUint32Rows(id, inputs, 'targets');
        if (predecessors.length !== props.nodeCount)
          throw new Error(`${id} predecessors length must equal nodeCount`);
        if (targets.length !== props.targetCapacity)
          throw new Error(`${id} targets length must equal targetCapacity`);
        const neighbors = props.edges
          ? getCPUOperationUint32Rows(id, inputs, 'neighbors')
          : undefined;
        if (props.edges && neighbors!.length !== props.edges.edgeCount)
          throw new Error(`${id} neighbors length must equal edges.edgeCount`);
        if (
          props.edges?.topology.nodePositions !== undefined &&
          getCPUOperationFloat32Rows(id, inputs, 'nodePositions', 'float32x2').length !==
            props.nodeCount * 2
        )
          throw new Error(`${id} nodePositions length must equal nodeCount * 2`);
        const predecessorEdges =
          props.predecessorEdges === undefined
            ? undefined
            : getCPUOperationUint32Rows(id, inputs, 'predecessorEdges');
        const rows = {
          predecessors,
          targets,
          costs: getCPUOperationFloat32Rows(id, inputs, 'costs'),
          targetCount:
            typeof props.targetCount === 'string'
              ? getCPUOperationUint32Rows(id, inputs, 'targetCount')
              : props.targetCount,
          predecessorsComplete:
            props.predecessorsComplete === undefined
              ? undefined
              : getCPUOperationUint32Rows(id, inputs, 'predecessorsComplete'),
          nodeIds:
            props.nodeIds === undefined
              ? undefined
              : getCPUOperationUint32Rows(id, inputs, 'nodeIds'),
          maximumPathLength: props.maximumPathLength,
          capacity: props.output.capacity
        };
        const result = computeNetworkPathExtractionOnCPU(
          props.edges
            ? {
                ...rows,
                predecessorEdges: predecessorEdges!,
                edges: {
                  offsets: getCPUOperationUint32Rows(id, inputs, 'offsets'),
                  neighbors: neighbors!,
                  weights: getCPUOperationFloat32Rows(id, inputs, 'weights'),
                  edgeIds:
                    props.edges.edgeIds === undefined
                      ? undefined
                      : getCPUOperationUint32Rows(id, inputs, 'edgeIds'),
                  capacity: props.edges.output.capacity
                }
              }
            : {...rows, predecessorEdges}
        );
        const outputs: Record<string, Uint32Array | Float32Array> = {};
        const addCompactResult = (
          output: NetworkPathExtractionOperationProps['output'],
          compact: CPUNetworkPathExtractionCompactResult
        ) => {
          outputs[output.ids] = compact.ids;
          outputs[output.status.count] = compact.count;
          outputs[output.status.overflow] = compact.overflow;
          outputs[output.status.requiredCount] = compact.requiredCount;
          if (output.status.unrepresentable !== undefined)
            outputs[output.status.unrepresentable] = compact.unrepresentable;
          if (output.status.candidateOverflow !== undefined)
            outputs[output.status.candidateOverflow] = compact.candidateOverflow;
          if (output.lineOffsets !== undefined) outputs[output.lineOffsets] = compact.lineOffsets;
        };
        addCompactResult(props.output, result);
        if (props.output.nodeRows !== undefined) outputs[props.output.nodeRows] = result.nodeRows;
        if (props.output.pathCosts !== undefined)
          outputs[props.output.pathCosts] = result.pathCosts;
        if (props.output.pathFound !== undefined)
          outputs[props.output.pathFound] = result.pathFound;
        if (props.edges && result.edges) {
          addCompactResult(props.edges.output, result.edges);
          if (props.edges.output.edgeRows !== undefined)
            outputs[props.edges.output.edgeRows] = result.edges.edgeRows;
        }
        return outputs;
      }
    });
  }
}
