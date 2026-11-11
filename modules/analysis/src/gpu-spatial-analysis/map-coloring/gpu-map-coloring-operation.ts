// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUMapColoring, type GPUMapColoringPriority} from './gpu-map-coloring';

/** Packed, unchunked CSR graph values. Distances and weights metadata are unused and omitted. */
export type MapColoringOperationProps = {
  id?: string;
  weights: {offsets: string; neighbors: string; weights: string};
  /** Explicit row count, otherwise inferred by the contributor from offsets. */
  rowCount: number;
  /** uint32 words from `getGPUMapColoringParameterValues`. */
  parameters: string;
  maximumIterations?: number;
  priority?: GPUMapColoringPriority;
  output: {
    colors: string;
    colorCount?: string;
    conflictCount?: string;
    status?: {converged?: string; iterationCount?: string};
  };
};

/** Map coloring operation; only whole CSR input is supported by the mixed executor. */
export class GPUMapColoringOperation extends GPUContributorOperation {
  constructor(props: MapColoringOperationProps) {
    const id = props.id ?? 'map-coloring';
    const {output} = props;
    super({
      id,
      inputs: {...props.weights, parameters: props.parameters},
      outputs: {
        colors: {value: output.colors, format: 'uint32', length: props.rowCount},
        ...(output.colorCount
          ? {colorCount: {value: output.colorCount, format: 'uint32' as const, length: 1}}
          : {}),
        ...(output.conflictCount
          ? {conflictCount: {value: output.conflictCount, format: 'uint32' as const, length: 1}}
          : {}),
        ...(output.status?.converged
          ? {converged: {value: output.status.converged, format: 'uint32' as const, length: 1}}
          : {}),
        ...(output.status?.iterationCount
          ? {
              iterationCount: {
                value: output.status.iterationCount,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUMapColoring({
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          parameters: getView<'uint32'>('parameters'),
          maximumIterations: props.maximumIterations,
          priority: props.priority,
          output: {
            colors: getView<'uint32'>('colors'),
            colorCount: output.colorCount ? getView<'uint32'>('colorCount') : undefined,
            conflictCount: output.conflictCount ? getView<'uint32'>('conflictCount') : undefined,
            status: {
              converged: output.status?.converged ? getView<'uint32'>('converged') : undefined,
              iterationCount: output.status?.iterationCount
                ? getView<'uint32'>('iterationCount')
                : undefined
            }
          }
        })
    });
  }
}
