// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUProfilePeaks} from './gpu-profile-peaks';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {TerrainAtLeastOne} from '../hydrology/terrain-output';

type ProfilePeaksOutputPorts<Floating, Unsigned, Compact> = {
  prominence?: Floating;
  refinedIndex?: Floating;
  refinedValue?: Floating;
  peakMask?: Unsigned;
  peaks?: Compact;
  status?: {converged?: Unsigned; iterationLimitReached?: Unsigned};
};

type ProfilePeaksOutputSelection<Floating, Unsigned, Compact> =
  | TerrainAtLeastOne<
      ProfilePeaksOutputPorts<Floating, Unsigned, Compact>,
      'prominence' | 'refinedIndex' | 'refinedValue' | 'peakMask' | 'peaks'
    >
  | (ProfilePeaksOutputPorts<Floating, Unsigned, Compact> & {
      status: TerrainAtLeastOne<{converged?: Unsigned; iterationLimitReached?: Unsigned}>;
    });

/** Retain a definite terminal through snapshots and view lowering, without a type assertion. */
function selectProfilePeaksOutput<Floating, Unsigned, Compact>(
  id: string,
  output: ProfilePeaksOutputPorts<Floating, Unsigned, Compact>
): ProfilePeaksOutputSelection<Floating, Unsigned, Compact> {
  if (output.prominence !== undefined) return {...output, prominence: output.prominence};
  if (output.refinedIndex !== undefined) return {...output, refinedIndex: output.refinedIndex};
  if (output.refinedValue !== undefined) return {...output, refinedValue: output.refinedValue};
  if (output.peakMask !== undefined) return {...output, peakMask: output.peakMask};
  if (output.peaks !== undefined) return {...output, peaks: output.peaks};
  if (output.status?.converged !== undefined)
    return {...output, status: {...output.status, converged: output.status.converged}};
  const iterationLimitReached = output.status?.iterationLimitReached;
  validateRequirement(
    id,
    iterationLimitReached !== undefined,
    'output must contain at least one dense, compact or populated status terminal; received an empty selection'
  );
  return {...output, status: {...output.status, iterationLimitReached}};
}

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
  output: ProfilePeaksOutputSelection<
    string,
    string,
    {
      ids: string;
      capacity: number;
      status: {count: string; requiredCount?: string; overflow: string; candidateOverflow?: string};
    }
  >;
};

/** @internal Capture operation topology once without enumerating caller getters. */
export function snapshotProfilePeaksOperationProps(
  props: ProfilePeaksOperationProps
): ProfilePeaksOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'values',
    'validity',
    'offsets',
    'parameters',
    'sampleCount',
    'window',
    'minimumSide',
    'suppressionRadius',
    'wrap',
    'maximumSuppressionIterations',
    'output'
  ]);
  const output = snapshotConstructionFields(captured.output, [
    'prominence',
    'refinedIndex',
    'refinedValue',
    'peakMask',
    'peaks',
    'status'
  ]);
  const peaks = output.peaks
    ? snapshotConstructionFields(output.peaks, ['ids', 'capacity', 'status'])
    : undefined;
  if (peaks) {
    validateRequirement(
      captured.id ?? 'profile-peaks',
      peaks.status !== null && typeof peaks.status === 'object' && !Array.isArray(peaks.status),
      `output.peaks.status must be an object containing count and overflow; received ${peaks.status === null ? 'null' : typeof peaks.status}`
    );
  }
  return Object.freeze({
    ...captured,
    output: Object.freeze(
      selectProfilePeaksOutput(captured.id ?? 'profile-peaks', {
        ...output,
        peaks: peaks
          ? Object.freeze({
              ...peaks,
              status: snapshotConstructionFields(peaks.status, [
                'count',
                'requiredCount',
                'overflow',
                'candidateOverflow'
              ])
            })
          : undefined,
        status: output.status
          ? snapshotConstructionFields(output.status, ['converged', 'iterationLimitReached'])
          : undefined
      })
    )
  });
}

/**
 * Packed-vector operation form of GPUProfilePeaks. All props are compile-time except input rows.
 * Compact readers should request peaks.status.candidateOverflow or output.status.converged /
 * iterationLimitReached as well as count/overflow; capacity overflow 0 does not imply completion.
 */
export class GPUProfilePeaksOperation extends GPUContributorOperation {
  constructor(props: ProfilePeaksOperationProps) {
    props = snapshotProfilePeaksOperationProps(props);
    const id = props.id ?? 'profile-peaks';
    const {output, sampleCount} = props;
    validateCapacity(id, 'sampleCount', sampleCount, 1);
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
              ...(output.peaks.status.candidateOverflow !== undefined
                ? {
                    candidateOverflow: {
                      value: output.peaks.status.candidateOverflow,
                      format: 'uint32' as const,
                      length: 1
                    }
                  }
                : {}),
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
      createContributor: getView => {
        const values = getView<'float32'>('values');
        validateRequirement(
          id,
          values.length === sampleCount,
          `values length must equal sampleCount (${sampleCount}); received ${values.length}`
        );
        return new GPUProfilePeaks({
          id,
          values,
          validity: props.validity !== undefined ? getView<'uint32'>('validity') : undefined,
          offsets: getView<'uint32'>('offsets'),
          parameters: getView<'float32'>('parameters'),
          window: props.window,
          minimumSide: props.minimumSide,
          suppressionRadius: props.suppressionRadius,
          wrap: props.wrap,
          maximumSuppressionIterations: props.maximumSuppressionIterations,
          output: selectProfilePeaksOutput(id, {
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
                    candidateOverflow:
                      output.peaks.status.candidateOverflow !== undefined
                        ? getView<'uint32'>('candidateOverflow')
                        : undefined,
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
          })
        });
      }
    });
  }
}
