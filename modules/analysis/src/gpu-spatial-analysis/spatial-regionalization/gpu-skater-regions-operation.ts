// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type RegionalizationWeightsStatus,
  validateRegionalizationWeightsStatus
} from './adjacency-loss-suppression';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {SpatialWeightsAlgebraGraphInput} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import {GPUSkaterRegions} from './gpu-skater-regions';
import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';

/** Packed graph values; rowCount and edgeCount make the view-inferred shapes explicit. */
export type SkaterRegionsOperationProps = {
  id?: string;
  rowCount: number;
  edgeCount: number;
  weights: SpatialWeightsAlgebraGraphInput & {
    status?: RegionalizationWeightsStatus<string>;
  };
  treeEdgeFlags: string;
  componentLabels: string;
  values: string;
  columnCount?: number;
  treeStatus?: string;
  maximumRegionCount: number;
  /** uint32 words packed by getGPUSkaterRegionsParameterValues. */
  parameters: string;
  output: {labels: string; regionCount?: string; cutEdges?: string; cutGains?: string};
};

/** @internal Preserve declared fields through deferred GPU lowering and CPU execution. */
export function snapshotSkaterOperationProps(
  props: SkaterRegionsOperationProps
): SkaterRegionsOperationProps {
  const captured = snapshotWeightsFields(props, [
    'id',
    'rowCount',
    'edgeCount',
    'weights',
    'treeEdgeFlags',
    'componentLabels',
    'values',
    'columnCount',
    'treeStatus',
    'maximumRegionCount',
    'parameters',
    'output'
  ]);
  const weights = snapshotWeightsPort(captured.weights);
  validateRegionalizationWeightsStatus(captured.id ?? 'skater-regions', weights.status);
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    const value = weights.status?.[name];
    validateRequirement(
      captured.id ?? 'skater-regions',
      value === undefined || (typeof value === 'string' && value.length > 0),
      `weights.status.${name} must be a non-empty graph value name when supplied`
    );
  }
  return Object.freeze({
    ...captured,
    weights,
    output: snapshotWeightsFields(captured.output, [
      'labels',
      'regionCount',
      'cutEdges',
      'cutGains'
    ])
  });
}

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSkaterRegionsOperation extends GPUContributorOperation {
  constructor(props: SkaterRegionsOperationProps) {
    props = snapshotSkaterOperationProps(props);
    super({
      id: props.id ?? 'skater-regions',
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.status?.overflow === undefined
          ? {}
          : {weightsOverflow: props.weights.status.overflow}),
        ...(props.weights.status?.candidateOverflow === undefined
          ? {}
          : {weightsCandidateOverflow: props.weights.status.candidateOverflow}),
        ...(props.weights.distances === undefined ? {} : {distances: props.weights.distances}),
        treeEdgeFlags: props.treeEdgeFlags,
        componentLabels: props.componentLabels,
        values: props.values,
        parameters: props.parameters,
        ...(props.treeStatus === undefined ? {} : {treeStatus: props.treeStatus})
      },
      outputs: {
        labels: {value: props.output.labels, format: 'uint32', length: props.rowCount},
        ...(props.output.regionCount === undefined
          ? {}
          : {regionCount: {value: props.output.regionCount, format: 'uint32' as const, length: 1}}),
        ...(props.output.cutEdges === undefined
          ? {}
          : {
              cutEdges: {
                value: props.output.cutEdges,
                format: 'uint32' as const,
                length: props.maximumRegionCount - 1
              }
            }),
        ...(props.output.cutGains === undefined
          ? {}
          : {
              cutGains: {
                value: props.output.cutGains,
                format: 'float32' as const,
                length: props.maximumRegionCount - 1
              }
            })
      },
      createContributor: getView =>
        new GPUSkaterRegions({
          id: props.id,
          columnCount: props.columnCount,
          maximumRegionCount: props.maximumRegionCount,
          weights: {
            ...props.weights,
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            status: {
              overflow:
                props.weights.status?.overflow === undefined
                  ? undefined
                  : getView<'uint32'>('weightsOverflow'),
              candidateOverflow:
                props.weights.status?.candidateOverflow === undefined
                  ? undefined
                  : getView<'uint32'>('weightsCandidateOverflow')
            },
            distances:
              props.weights.distances === undefined ? undefined : getView<'float32'>('distances')
          },
          treeEdgeFlags: getView<'uint32'>('treeEdgeFlags'),
          componentLabels: getView<'uint32'>('componentLabels'),
          values: getView<'float32'>('values'),
          parameters: getView<'uint32'>('parameters'),
          treeStatus: props.treeStatus === undefined ? undefined : getView<'uint32'>('treeStatus'),
          output: {
            labels: getView<'uint32'>('labels'),
            regionCount:
              props.output.regionCount === undefined ? undefined : getView<'uint32'>('regionCount'),
            cutEdges:
              props.output.cutEdges === undefined ? undefined : getView<'uint32'>('cutEdges'),
            cutGains:
              props.output.cutGains === undefined ? undefined : getView<'float32'>('cutGains')
          }
        })
    });
  }
}
