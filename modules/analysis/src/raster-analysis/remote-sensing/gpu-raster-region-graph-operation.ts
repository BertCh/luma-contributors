// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURasterRegionGraph} from './gpu-raster-region-graph';
import {snapshotRasterRegionGrid} from './raster-region-snapshot';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Value names replacing views; matrices have exactly `regionCount ** 2` scalar rows. */
export type RasterRegionGraphOperationProps = {
  id?: string;
  grid: GPURasterMetadata;
  labels: string;
  /** Optional one-word positive completeness certificate for the source labels. */
  labelsComplete?: string;
  values: string;
  regionCount: number;
  mergeMode?: 'fixed' | 'adaptive';
  searchStructure?: 'auto' | 'dense' | 'boruvka';
  mergeThreshold: number;
  maximumIterations: number;
  output: {
    adjacency: string;
    weights: string;
    mapping: string;
    status: {iterationLimitReached: string; complete?: string};
  };
};

/** Captures topology and names before the caller can mutate construction props. */
export function snapshotRasterRegionGraphOperationProps(
  props: RasterRegionGraphOperationProps
): RasterRegionGraphOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'grid',
    'labels',
    'labelsComplete',
    'values',
    'regionCount',
    'mergeMode',
    'searchStructure',
    'mergeThreshold',
    'maximumIterations',
    'output'
  ]);
  const output = snapshotConstructionFields(captured.output, [
    'adjacency',
    'weights',
    'mapping',
    'status'
  ]);
  return {
    ...captured,
    grid: snapshotRasterRegionGrid(captured.grid),
    output: {
      ...output,
      status: snapshotConstructionFields(output.status, ['iterationLimitReached', 'complete'])
    }
  };
}

/** Operation form of the contributor, with vector-only ports and no additional behaviour. */
export class GPURasterRegionGraphOperation extends GPUContributorOperation {
  constructor(inputProps: RasterRegionGraphOperationProps) {
    const props = snapshotRasterRegionGraphOperationProps(inputProps);
    const id = props.id ?? 'raster-region-graph';
    validateRequirement(
      id,
      props.labelsComplete === undefined || props.output.status.complete !== undefined,
      'labelsComplete requires output.status.complete to retain source-label completeness'
    );
    super({
      id,
      inputs: {
        labels: props.labels,
        values: props.values,
        ...(props.labelsComplete !== undefined ? {labelsComplete: props.labelsComplete} : {})
      },
      outputs: {
        adjacency: {
          value: props.output.adjacency,
          format: 'uint32',
          length: props.regionCount ** 2
        },
        weights: {value: props.output.weights, format: 'float32', length: props.regionCount ** 2},
        mapping: {value: props.output.mapping, format: 'uint32', length: props.regionCount},
        iterationLimitReached: {
          value: props.output.status.iterationLimitReached,
          format: 'uint32',
          length: 1
        },
        ...(props.output.status.complete !== undefined
          ? {
              complete: {
                value: props.output.status.complete,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPURasterRegionGraph({
          ...props,
          id,
          labels: getView<'uint32'>('labels'),
          labelsComplete:
            props.labelsComplete !== undefined ? getView<'uint32'>('labelsComplete') : undefined,
          values: getView<'float32'>('values'),
          output: {
            adjacency: getView<'uint32'>('adjacency'),
            weights: getView<'float32'>('weights'),
            mapping: getView<'uint32'>('mapping'),
            status: {
              iterationLimitReached: getView<'uint32'>('iterationLimitReached'),
              complete:
                props.output.status.complete !== undefined
                  ? getView<'uint32'>('complete')
                  : undefined
            }
          }
        })
    });
  }
}
