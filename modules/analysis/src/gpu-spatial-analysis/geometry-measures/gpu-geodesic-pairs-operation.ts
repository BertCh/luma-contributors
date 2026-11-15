// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUGeodesicPairs} from './gpu-geodesic-pairs';

/** Graph-value properties shared by the GPU operation and CPU twin. */
export type GeodesicPairsOperationProps = {
  id?: string;
  spatialContext: GPUSpatialContext;
  origins: string;
  targets: string;
  pairCount: number;
  maximumIterations?: number;
  output: {
    distances?: string;
    initialBearings?: string;
    finalBearings?: string;
    midpoints?: string;
    status?: {converged?: string; approximation?: string; iterationLimitReached?: string};
  };
};

/** GPU geodesic pairs as an operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUGeodesicPairsOperation extends GPUContributorOperation {
  constructor(props: GeodesicPairsOperationProps) {
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const port of ['distances', 'initialBearings', 'finalBearings', 'midpoints'] as const) {
      if (props.output[port] !== undefined)
        outputs[port] = {
          value: props.output[port]!,
          format: port === 'midpoints' ? 'float32x2' : 'float32',
          length: props.pairCount
        };
    }
    for (const port of ['converged', 'approximation', 'iterationLimitReached'] as const) {
      const value = props.output.status?.[port];
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: props.pairCount};
    }
    super({
      id: props.id ?? 'geodesic-pairs',
      inputs: {origins: props.origins, targets: props.targets},
      outputs,
      createContributor: getView =>
        new GPUGeodesicPairs({
          id: props.id,
          spatialContext: props.spatialContext,
          origins: getView<'float32x2'>('origins'),
          targets: getView<'float32x2'>('targets'),
          maximumIterations: props.maximumIterations,
          output: {
            distances:
              props.output.distances !== undefined ? getView<'float32'>('distances') : undefined,
            initialBearings:
              props.output.initialBearings !== undefined
                ? getView<'float32'>('initialBearings')
                : undefined,
            finalBearings:
              props.output.finalBearings !== undefined
                ? getView<'float32'>('finalBearings')
                : undefined,
            midpoints:
              props.output.midpoints !== undefined ? getView<'float32x2'>('midpoints') : undefined,
            status: {
              converged:
                props.output.status?.converged !== undefined
                  ? getView<'uint32'>('converged')
                  : undefined,
              approximation:
                props.output.status?.approximation !== undefined
                  ? getView<'uint32'>('approximation')
                  : undefined,
              iterationLimitReached:
                props.output.status?.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined
            }
          }
        })
    });
  }
}
