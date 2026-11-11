// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUProfilePeaks} from './gpu-profile-peaks';

/** Graph-value names shared by the GPU operation and its CPU twin. */
export type ProfilePeaksOperationProps = {
  id?: string;
  values: string;
  validity?: string;
  offsets: string;
  parameters: string;
  /** Number of sample rows, including samples outside the CSR spans. */
  sampleCount: number;
  window: number;
  minimumSide?: number;
  suppressionRadius?: number;
  wrap?: boolean;
  maximumSuppressionIterations?: number;
  output: {
    prominence?: string;
    refinedIndex?: string;
    refinedValue?: string;
    peakMask?: string;
    peaks?: {
      ids: string;
      capacity: number;
      status: {count: string; requiredCount?: string; overflow: string};
    };
    status?: {converged?: string; iterationLimitReached?: string};
  };
};

/** Packed-vector operation form of GPUProfilePeaks. All props are compile-time except input rows. */
export class GPUProfilePeaksOperation extends GPUContributorOperation {
  constructor(props: ProfilePeaksOperationProps) {
    const id = props.id ?? 'profile-peaks';
    const {output, sampleCount} = props;
    super({
      id,
      inputs: {
        values: props.values,
        offsets: props.offsets,
        parameters: props.parameters,
        ...(props.validity !== undefined ? {validity: props.validity} : {})
      },
      outputs: {
        ...Object.fromEntries(
          (['prominence', 'refinedIndex', 'refinedValue', 'peakMask'] as const)
            .filter(name => output[name] !== undefined)
            .map(name => [
              name,
              {
                value: output[name]!,
                format: name === 'peakMask' ? ('uint32' as const) : ('float32' as const),
                length: sampleCount
              }
            ])
        ),
        ...(output.peaks
          ? {
              ids: {value: output.peaks.ids, format: 'uint32', length: output.peaks.capacity},
              count: {value: output.peaks.status.count, format: 'uint32', length: 1},
              overflow: {value: output.peaks.status.overflow, format: 'uint32', length: 1},
              ...(output.peaks.status.requiredCount !== undefined
                ? {
                    requiredCount: {
                      value: output.peaks.status.requiredCount,
                      format: 'uint32' as const,
                      length: 1
                    }
                  }
                : {})
            }
          : {}),
        ...Object.fromEntries(
          (['converged', 'iterationLimitReached'] as const)
            .filter(name => output.status?.[name] !== undefined)
            .map(name => [
              name,
              {value: output.status![name]!, format: 'uint32' as const, length: 1}
            ])
        )
      },
      createContributor: getView =>
        new GPUProfilePeaks({
          id,
          values: getView<'float32'>('values'),
          validity: props.validity !== undefined ? getView<'uint32'>('validity') : undefined,
          offsets: getView<'uint32'>('offsets'),
          parameters: getView<'float32'>('parameters'),
          window: props.window,
          minimumSide: props.minimumSide,
          suppressionRadius: props.suppressionRadius,
          wrap: props.wrap,
          maximumSuppressionIterations: props.maximumSuppressionIterations,
          output: {
            prominence:
              output.prominence !== undefined ? getView<'float32'>('prominence') : undefined,
            refinedIndex:
              output.refinedIndex !== undefined ? getView<'float32'>('refinedIndex') : undefined,
            refinedValue:
              output.refinedValue !== undefined ? getView<'float32'>('refinedValue') : undefined,
            peakMask: output.peakMask !== undefined ? getView<'uint32'>('peakMask') : undefined,
            peaks: output.peaks
              ? {
                  ids: getView<'uint32'>('ids'),
                  status: {
                    count: getView<'uint32'>('count'),
                    overflow: getView<'uint32'>('overflow'),
                    requiredCount:
                      output.peaks.status.requiredCount !== undefined
                        ? getView<'uint32'>('requiredCount')
                        : undefined
                  }
                }
              : undefined,
            status: {
              converged:
                output.status?.converged !== undefined ? getView<'uint32'>('converged') : undefined,
              iterationLimitReached:
                output.status?.iterationLimitReached !== undefined
                  ? getView<'uint32'>('iterationLimitReached')
                  : undefined
            }
          }
        })
    });
  }
}
