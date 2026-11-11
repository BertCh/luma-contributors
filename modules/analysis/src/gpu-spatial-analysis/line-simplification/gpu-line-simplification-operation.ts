// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUContributorOperationOutput} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPULineSimplification, type GPULineSimplificationProps} from './gpu-line-simplification';

/** Graph value names replace packed views; lengths are explicit. */
export type LineSimplificationOperationProps = Pick<
  GPULineSimplificationProps,
  | 'id'
  | 'spatialContext'
  | 'method'
  | 'metric'
  | 'neighborhoodRadius'
  | 'maximumIterations'
  | 'finishSpanLimit'
  | 'computeImportance'
> & {
  positions: string;
  lineOffsets: string;
  timestamps?: string;
  parameters?: string;
  importance: string;
  rowCount: number;
  lineCount: number;
  capacity?: number;
  output?: {
    ids?: string;
    outputMask?: string;
    lineCounts?: string;
    lineStarts?: string;
    status?: {
      count?: string;
      overflow?: string;
      requiredCount?: string;
      converged?: string;
      iterationCount?: string;
    };
  };
};

/** GPU line simplification operation. Importance is either an input or an output, never both. */
export class GPULineSimplificationOperation extends GPUContributorOperation {
  constructor(props: LineSimplificationOperationProps) {
    const inputs: Record<string, string> = {
      positions: props.positions,
      lineOffsets: props.lineOffsets
    };
    if (props.timestamps !== undefined) inputs['timestamps'] = props.timestamps;
    if (props.parameters !== undefined) inputs['parameters'] = props.parameters;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    if (props.computeImportance === false) inputs['importance'] = props.importance;
    else
      outputs['importance'] = {value: props.importance, format: 'float32', length: props.rowCount};
    for (const port of ['ids', 'outputMask', 'lineCounts', 'lineStarts'] as const) {
      const value = props.output?.[port];
      if (value !== undefined)
        outputs[port] = {
          value,
          format: 'uint32',
          length:
            port === 'ids'
              ? (props.capacity ?? props.rowCount)
              : port === 'outputMask'
                ? props.rowCount
                : props.lineCount
        };
    }
    for (const [port, value] of Object.entries(props.output?.status ?? {}))
      if (value !== undefined) outputs[port] = {value, format: 'uint32', length: 1};
    super({
      id: props.id ?? 'line-simplification',
      inputs,
      outputs,
      createContributor: getView =>
        new GPULineSimplification({
          id: props.id,
          spatialContext: props.spatialContext,
          method: props.method,
          metric: props.metric,
          neighborhoodRadius: props.neighborhoodRadius,
          maximumIterations: props.maximumIterations,
          finishSpanLimit: props.finishSpanLimit,
          computeImportance: props.computeImportance,
          positions: getView<'float32x2'>('positions'),
          lineOffsets: getView<'uint32'>('lineOffsets'),
          importance: getView<'float32'>('importance'),
          timestamps: props.timestamps !== undefined ? getView<'float32'>('timestamps') : undefined,
          parameters: props.parameters !== undefined ? getView<'float32'>('parameters') : undefined,
          output: {
            ...(props.output?.ids !== undefined
              ? {
                  ids: getView<'uint32'>('ids'),
                  outputMask:
                    props.output.outputMask !== undefined
                      ? getView<'uint32'>('outputMask')
                      : undefined,
                  lineCounts:
                    props.output.lineCounts !== undefined
                      ? getView<'uint32'>('lineCounts')
                      : undefined,
                  lineStarts:
                    props.output.lineStarts !== undefined
                      ? getView<'uint32'>('lineStarts')
                      : undefined
                }
              : {}),
            status: {
              ...(props.output?.ids !== undefined
                ? {
                    count: getView<'uint32'>('count'),
                    overflow: getView<'uint32'>('overflow'),
                    requiredCount:
                      props.output.status?.requiredCount !== undefined
                        ? getView<'uint32'>('requiredCount')
                        : undefined
                  }
                : {}),
              converged:
                props.output?.status?.converged !== undefined
                  ? getView<'uint32'>('converged')
                  : undefined,
              iterationCount:
                props.output?.status?.iterationCount !== undefined
                  ? getView<'uint32'>('iterationCount')
                  : undefined
            }
          } as GPULineSimplificationProps['output']
        }),
      emptyOutputPorts: ['ids']
    });
  }
}
