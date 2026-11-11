// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import type {SpatialWeightsAlgebraGraphInput} from '../spatial-weights/gpu-spatial-weights-algebra-operation';
import {GPUSkaterRegions} from './gpu-skater-regions';

/** Packed graph values; rowCount and edgeCount make the view-inferred shapes explicit. */
export type SkaterRegionsOperationProps = {
  id?: string;
  rowCount: number;
  edgeCount: number;
  weights: SpatialWeightsAlgebraGraphInput;
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

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSkaterRegionsOperation extends GPUContributorOperation {
  constructor(props: SkaterRegionsOperationProps) {
    super({
      id: props.id ?? 'skater-regions',
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
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
