// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUMapMatching, type GPUMapMatchingProps} from './gpu-map-matching';

/** Graph-value names replacing views; counts inferred from views become explicit. */
export type MapMatchingOperationProps = Pick<
  GPUMapMatchingProps,
  | 'id'
  | 'spatialContext'
  | 'candidateCount'
  | 'cellSize'
  | 'bounds'
  | 'entryCapacity'
  | 'routeNodeBudget'
  | 'minimumFixSpacingSigmas'
  | 'traversedEdgeCapacity'
> & {
  pointCount: number;
  trackCount: number;
  nodeCount: number;
  edgeCount: number;
  positions: string;
  trackOffsets: string;
  parameters: string;
  topology: {offsets: string; neighbors: string; nodePositions: string; weights?: string};
  routeTable?: {
    entryCount: number;
    offsets: string;
    targets: string;
    costs: string;
    costLimit: string;
    status?: {overflow?: string};
  };
  output: {
    matchedEdges: string;
    matchedFractions?: string;
    matchedOffsets?: string;
    snapDistances?: string;
    snappedPositions?: string;
    breaks?: string;
    trackLogLikelihoods?: string;
    matchedCount?: string;
    breakCount?: string;
    status?: {overflow?: string; routeOverflow?: string};
    traversedEdges?: {edges: string; counts: string; status?: {overflow?: string}};
  };
};

/** GPU map matching as a vector graph operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUMapMatchingOperation extends GPUContributorOperation {
  constructor(props: MapMatchingOperationProps) {
    const {output, routeTable} = props;
    const inputs: Record<string, string> = {
      positions: props.positions,
      trackOffsets: props.trackOffsets,
      parameters: props.parameters,
      offsets: props.topology.offsets,
      neighbors: props.topology.neighbors,
      nodePositions: props.topology.nodePositions
    };
    if (props.topology.weights !== undefined) inputs['weights'] = props.topology.weights;
    if (routeTable) {
      inputs['tableOffsets'] = routeTable.offsets;
      inputs['tableTargets'] = routeTable.targets;
      inputs['tableCosts'] = routeTable.costs;
      inputs['tableCostLimit'] = routeTable.costLimit;
      if (routeTable.status?.overflow !== undefined)
        inputs['tableOverflow'] = routeTable.status.overflow;
    }
    const outputs: Record<string, GPUContributorOperationOutput> = {
      matchedEdges: {value: output.matchedEdges, format: 'uint32', length: props.pointCount}
    };
    for (const port of ['matchedFractions', 'matchedOffsets', 'snapDistances'] as const) {
      if (output[port] !== undefined)
        outputs[port] = {value: output[port], format: 'float32', length: props.pointCount};
    }
    if (output.snappedPositions !== undefined)
      outputs['snappedPositions'] = {
        value: output.snappedPositions,
        format: 'float32x2',
        length: props.pointCount
      };
    if (output.trackLogLikelihoods !== undefined)
      outputs['trackLogLikelihoods'] = {
        value: output.trackLogLikelihoods,
        format: 'float32',
        length: props.trackCount
      };
    for (const port of ['breaks', 'matchedCount', 'breakCount'] as const) {
      if (output[port] !== undefined)
        outputs[port] = {
          value: output[port],
          format: 'uint32',
          length: port === 'breaks' ? props.pointCount : 1
        };
    }
    for (const port of ['overflow', 'routeOverflow'] as const) {
      const value = output.status?.[port];
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    }
    if (output.traversedEdges) {
      outputs['traversedEdges'] = {
        value: output.traversedEdges.edges,
        format: 'uint32',
        length: props.trackCount * (props.traversedEdgeCapacity ?? 0)
      };
      outputs['traversedCounts'] = {
        value: output.traversedEdges.counts,
        format: 'uint32',
        length: props.trackCount
      };
      if (output.traversedEdges.status?.overflow !== undefined)
        outputs['traversedOverflow'] = {
          value: output.traversedEdges.status.overflow,
          format: 'uint32',
          length: 1
        };
    }
    super({
      id: props.id ?? 'map-matching',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUMapMatching({
          id: props.id,
          spatialContext: props.spatialContext,
          positions: getView<'float32x2'>('positions'),
          trackOffsets: getView<'uint32'>('trackOffsets'),
          parameters: getView<'float32'>('parameters'),
          topology: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            nodePositions: getView<'float32x2'>('nodePositions'),
            weights:
              props.topology.weights === undefined ? undefined : getView<'float32'>('weights')
          },
          candidateCount: props.candidateCount,
          cellSize: props.cellSize,
          bounds: props.bounds,
          entryCapacity: props.entryCapacity,
          routeNodeBudget: props.routeNodeBudget,
          minimumFixSpacingSigmas: props.minimumFixSpacingSigmas,
          traversedEdgeCapacity: props.traversedEdgeCapacity,
          routeTable: routeTable
            ? {
                offsets: getView<'uint32'>('tableOffsets'),
                targets: getView<'uint32'>('tableTargets'),
                costs: getView<'float32'>('tableCosts'),
                costLimit: getView<'float32'>('tableCostLimit'),
                status: {
                  overflow:
                    routeTable.status?.overflow === undefined
                      ? undefined
                      : getView<'uint32'>('tableOverflow')
                }
              }
            : undefined,
          output: {
            matchedEdges: getView<'uint32'>('matchedEdges'),
            matchedFractions:
              output.matchedFractions === undefined
                ? undefined
                : getView<'float32'>('matchedFractions'),
            matchedOffsets:
              output.matchedOffsets === undefined
                ? undefined
                : getView<'float32'>('matchedOffsets'),
            snapDistances:
              output.snapDistances === undefined ? undefined : getView<'float32'>('snapDistances'),
            snappedPositions:
              output.snappedPositions === undefined
                ? undefined
                : getView<'float32x2'>('snappedPositions'),
            breaks: output.breaks === undefined ? undefined : getView<'uint32'>('breaks'),
            trackLogLikelihoods:
              output.trackLogLikelihoods === undefined
                ? undefined
                : getView<'float32'>('trackLogLikelihoods'),
            matchedCount:
              output.matchedCount === undefined ? undefined : getView<'uint32'>('matchedCount'),
            breakCount:
              output.breakCount === undefined ? undefined : getView<'uint32'>('breakCount'),
            status: {
              overflow:
                output.status?.overflow === undefined ? undefined : getView<'uint32'>('overflow'),
              routeOverflow:
                output.status?.routeOverflow === undefined
                  ? undefined
                  : getView<'uint32'>('routeOverflow')
            },
            traversedEdges: output.traversedEdges
              ? {
                  edges: getView<'uint32'>('traversedEdges'),
                  counts: getView<'uint32'>('traversedCounts'),
                  status: {
                    overflow:
                      output.traversedEdges.status?.overflow === undefined
                        ? undefined
                        : getView<'uint32'>('traversedOverflow')
                  }
                }
              : undefined
          }
        })
    });
  }
}
