// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUStreamlines, type GPUStreamlinesProps} from './gpu-streamlines';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Streamline graph values, including optional pre-publication candidate diagnostics. */
export type StreamlinesOperationProps = Pick<
  GPUStreamlinesProps,
  | 'id'
  | 'fieldWidth'
  | 'fieldHeight'
  | 'gridWidth'
  | 'gridHeight'
  | 'seedColumns'
  | 'seedRows'
  | 'stepsPerDirection'
  | 'maximumIterations'
> & {
  velocities: string;
  parameters: string;
  uint32Parameters: string;
  lineCapacity: number;
  pointCapacity: number;
  output: {
    lines: {
      ids: string;
      status: {count: string; requiredCount?: string; overflow: string; candidateOverflow?: string};
    };
    lineOffsets: string;
    positions: string;
    positionCount: string;
    status?: {iterationLimitReached?: string};
    candidates?: {positions: string; spans: string};
  };
};

/** @internal Capture operation topology once, including inherited budget and status fields. */
export function snapshotStreamlinesOperationProps(
  props: StreamlinesOperationProps
): StreamlinesOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'fieldWidth',
    'fieldHeight',
    'gridWidth',
    'gridHeight',
    'seedColumns',
    'seedRows',
    'stepsPerDirection',
    'maximumIterations',
    'velocities',
    'parameters',
    'uint32Parameters',
    'lineCapacity',
    'pointCapacity',
    'output'
  ]);
  const output = snapshotConstructionFields(captured.output, [
    'lines',
    'lineOffsets',
    'positions',
    'positionCount',
    'status',
    'candidates'
  ]);
  const lines = snapshotConstructionFields(output.lines, ['ids', 'status']);
  return Object.freeze({
    ...captured,
    output: Object.freeze({
      ...output,
      lines: Object.freeze({
        ...lines,
        status: snapshotConstructionFields(lines.status, [
          'count',
          'requiredCount',
          'overflow',
          'candidateOverflow'
        ])
      }),
      status: output.status
        ? snapshotConstructionFields(output.status, ['iterationLimitReached'])
        : undefined,
      candidates: output.candidates
        ? snapshotConstructionFields(output.candidates, ['positions', 'spans'])
        : undefined
    })
  });
}

/** Vector-backed streamlines; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUStreamlinesOperation extends GPUContributorOperation {
  constructor(props: StreamlinesOperationProps) {
    props = snapshotStreamlinesOperationProps(props);
    const {output} = props;
    super({
      id: props.id ?? 'streamlines',
      inputs: {
        velocities: props.velocities,
        parameters: props.parameters,
        uint32Parameters: props.uint32Parameters
      },
      outputs: {
        ids: {value: output.lines.ids, format: 'uint32', length: props.lineCapacity},
        lineOffsets: {value: output.lineOffsets, format: 'uint32', length: props.lineCapacity + 1},
        positions: {value: output.positions, format: 'float32x2', length: props.pointCapacity},
        positionCount: {value: output.positionCount, format: 'uint32', length: 1},
        ...(output.candidates
          ? {
              candidatePositions: {
                value: output.candidates.positions,
                format: 'float32x2' as const,
                length: props.seedColumns * props.seedRows * (2 * props.stepsPerDirection + 1)
              },
              candidateSpans: {
                value: output.candidates.spans,
                format: 'uint32' as const,
                length: props.seedColumns * props.seedRows * 4
              }
            }
          : {}),
        ...Object.fromEntries(
          Object.entries({...output.lines.status, ...output.status})
            .filter(([, value]) => value !== undefined)
            .map(([port, value]) => [port, {value, format: 'uint32' as const, length: 1}])
        )
      },
      createContributor: getView =>
        new GPUStreamlines({
          ...props,
          velocities: getView<'float32x2'>('velocities'),
          parameters: getView<'float32'>('parameters'),
          uint32Parameters: getView<'uint32'>('uint32Parameters'),
          output: {
            candidates: output.candidates
              ? {
                  positions: getView<'float32x2'>('candidatePositions'),
                  spans: getView<'uint32'>('candidateSpans')
                }
              : undefined,
            lines: {
              ids: getView<'uint32'>('ids'),
              status: {
                count: getView<'uint32'>('count'),
                overflow: getView<'uint32'>('overflow'),
                candidateOverflow:
                  output.lines.status.candidateOverflow !== undefined
                    ? getView<'uint32'>('candidateOverflow')
                    : undefined,
                requiredCount:
                  output.lines.status.requiredCount !== undefined
                    ? getView<'uint32'>('requiredCount')
                    : undefined
              }
            },
            lineOffsets: getView<'uint32'>('lineOffsets'),
            positions: getView<'float32x2'>('positions'),
            positionCount: getView<'uint32'>('positionCount'),
            status: {
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
