// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {GPUSpatialContext} from '../../analysis-core/contracts/index';
import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSinkhornDivergence} from './gpu-sinkhorn-divergence';

/** Shared graph-value props. Grid metadata is static; masses and packed parameters are vectors. */
export type SinkhornDivergenceOperationProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  grid: GPURasterMetadata;
  source: string;
  target: string;
  parameters: string;
  maximumIterations?: number;
  blurRadius?: number;
  output: {
    divergence: string;
    components?: string;
    marginalError?: string;
    status?: {
      converged?: string;
      iterationLimitReached?: string;
      iterationCount?: string;
      invalidCount?: string;
      approximation?: string;
    };
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSinkhornDivergenceOperation extends GPUContributorOperation {
  constructor(props: SinkhornDivergenceOperationProps) {
    const outputs: Record<string, GPUContributorOperationOutput> = {
      divergence: {value: props.output.divergence, format: 'float32', length: 1}
    };
    if (props.output.components !== undefined)
      outputs['components'] = {value: props.output.components, format: 'float32', length: 3};
    if (props.output.marginalError !== undefined)
      outputs['marginalError'] = {value: props.output.marginalError, format: 'float32', length: 1};
    for (const [port, value] of Object.entries(props.output.status ?? {}))
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    super({
      id: props.id ?? 'sinkhorn-divergence',
      inputs: {source: props.source, target: props.target, parameters: props.parameters},
      outputs,
      createContributor: getView =>
        new GPUSinkhornDivergence({
          ...props,
          source: getView<'float32'>('source'),
          target: getView<'float32'>('target'),
          parameters: getView<'float32'>('parameters'),
          output: {
            divergence: getView<'float32'>('divergence'),
            components:
              props.output.components !== undefined ? getView<'float32'>('components') : undefined,
            marginalError:
              props.output.marginalError !== undefined
                ? getView<'float32'>('marginalError')
                : undefined,
            status: Object.fromEntries(
              Object.entries(props.output.status ?? {})
                .filter(([, value]) => value !== undefined)
                .map(([port]) => [port, getView<'uint32'>(port)])
            )
          }
        })
    });
  }
}
