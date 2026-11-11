// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUTrajectoryRows,
  type CPUTrajectoryRows
} from '../../analysis-core/operations/cpu-trajectory-rows';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {GPU_MOVEMENT_SENTINEL} from '../trajectory-staypoints/movement-common';
import {
  GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH,
  type GPUTrajectorySplitMode
} from './gpu-trajectory-split';
import {
  GPUTrajectorySplitOperation,
  type TrajectorySplitOperationProps
} from './gpu-trajectory-split-operation';

/** Packed trajectory and the mode-specific input columns. */
export type CPUTrajectorySplitInput = CPUTrajectoryRows & {
  parameters?: Float32Array;
  staypointIds?: Uint32Array;
  values?: Uint32Array;
};

/** Compile-time split method and output capacities. */
export type CPUTrajectorySplitOptions = {
  mode: GPUTrajectorySplitMode;
  segmentCapacity: number;
  rowCapacity: number;
};

/** Output ports, with copied geometry under distinct names from the inputs. */
export type CPUTrajectorySplitResult = {
  segmentOffsets: Uint32Array;
  segmentTracks: Uint32Array;
  rowIds: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  outputPositions: Float32Array;
  outputTimestamps: Float32Array | Uint32Array;
};

function validateOptions(id: string, options: CPUTrajectorySplitOptions): void {
  validateCapacity(id, 'segmentCapacity', options.segmentCapacity);
  validateCapacity(id, 'rowCapacity', options.rowCapacity);
  validateRequirement(
    id,
    ['gap', 'stop', 'value-change'].includes(options.mode),
    "mode must be 'gap', 'stop' or 'value-change'"
  );
}

/**
 * Splits each track into maximal runs, drops runs shorter than two rows and repeats the first
 * row of the next run in value-change mode. Segment and row capacities are independent, exactly
 * as in the contributor (including its terminal offset when segment capacity overflows).
 *
 * What differs from the GPU contributor:
 * - Time differences use float64; Int64 words are subtracted exactly before conversion. A gap
 *   decision at a float32 rounding boundary can differ.
 * - Decreasing trackOffsets after the GPU's row-count clamp throw an Error.
 * - Unwritten rowIds and copied geometry capacity rows are zero. The GPU leaves them untouched.
 * - Nonfinite timestamps follow JavaScript comparison rules; GPU behavior is unspecified there.
 *
 * @throws Error naming an inconsistent positions, timestamps, trackOffsets, parameters,
 * staypointIds or values argument.
 */
export function computeTrajectorySplitOnCPU(
  input: CPUTrajectorySplitInput,
  options: CPUTrajectorySplitOptions
): CPUTrajectorySplitResult {
  validateOptions('trajectory-split', options);
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input);
  const {mode, segmentCapacity, rowCapacity} = options;
  if (
    mode === 'gap' &&
    (!input.parameters || input.parameters.length < GPU_TRAJECTORY_SPLIT_PARAMETER_LENGTH)
  )
    throw new Error('parameters must hold the trajectory split parameter layout');
  if (input.staypointIds && input.staypointIds.length !== rowCount)
    throw new Error('staypointIds length must match positions');
  if (input.values && input.values.length !== rowCount)
    throw new Error('values length must match positions');
  if (mode === 'stop' && !input.staypointIds)
    throw new Error('staypointIds is required for stop mode');
  if (mode === 'value-change' && !input.values)
    throw new Error('values is required for value-change mode');
  const [maximumGap = 0] = input.parameters ?? [];
  const isExcludedRow = (row: number) =>
    mode === 'stop' && input.staypointIds![row] !== GPU_MOVEMENT_SENTINEL;
  const breaksBefore = (row: number) =>
    mode === 'gap'
      ? maximumGap > 0 && getTimeDifference(row, row - 1) > maximumGap
      : mode === 'stop'
        ? isExcludedRow(row)
        : input.values![row] !== input.values![row - 1];
  const result: CPUTrajectorySplitResult = {
    segmentOffsets: new Uint32Array(segmentCapacity + 1).fill(GPU_MOVEMENT_SENTINEL),
    segmentTracks: new Uint32Array(segmentCapacity).fill(GPU_MOVEMENT_SENTINEL),
    rowIds: new Uint32Array(rowCapacity),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    outputPositions: new Float32Array(rowCapacity * 2),
    outputTimestamps:
      input.timestamps instanceof Uint32Array
        ? new Uint32Array(rowCapacity * 2)
        : new Float32Array(rowCapacity)
  };
  let segmentTotal = 0;
  let rowTotal = 0;
  for (let track = 0; track + 1 < trackOffsets.length; track++) {
    const end = trackOffsets[track + 1];
    let start = trackOffsets[track];
    while (start < end) {
      if (isExcludedRow(start)) {
        start++;
        continue;
      }
      let finish = start + 1;
      while (finish < end && !breaksBefore(finish)) finish++;
      const emittedRows = finish - start + Number(mode === 'value-change' && finish < end);
      if (emittedRows >= 2) {
        if (segmentTotal < segmentCapacity) {
          result.segmentOffsets[segmentTotal] = rowTotal;
          result.segmentTracks[segmentTotal] = track;
        }
        for (let offset = 0; offset < emittedRows && rowTotal + offset < rowCapacity; offset++) {
          const source = start + offset;
          const target = rowTotal + offset;
          result.rowIds[target] = source;
          result.outputPositions.set(
            input.positions.subarray(source * 2, source * 2 + 2),
            target * 2
          );
          if (input.timestamps instanceof Uint32Array)
            result.outputTimestamps.set(
              input.timestamps.subarray(source * 2, source * 2 + 2),
              target * 2
            );
          else result.outputTimestamps[target] = input.timestamps[source];
        }
        segmentTotal++;
        rowTotal += emittedRows;
      }
      start = finish;
    }
  }
  const count = Math.min(segmentTotal, segmentCapacity);
  result.segmentOffsets[count] = Math.min(rowTotal, rowCapacity);
  result.count[0] = count;
  result.requiredCount[0] = segmentTotal;
  result.overflow[0] = Number(segmentTotal > segmentCapacity || rowTotal > rowCapacity);
  return result;
}

/** CPU split operation with the contributor's independent segment and row capacity rules. */
export class CPUTrajectorySplitOperation extends GPUCPUOperation {
  constructor(props: TrajectorySplitOperationProps) {
    const operation = new GPUTrajectorySplitOperation(props);
    const {id} = operation;
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        ['float32', 'uint32x2'].includes(props.timestampFormat),
      'timestampFormat must be float32 or uint32x2'
    );
    validateRequirement(
      id,
      props.mode !== 'gap' || props.parameters !== undefined,
      "mode 'gap' requires parameters"
    );
    validateRequirement(
      id,
      props.mode !== 'stop' || props.staypointIds !== undefined,
      "mode 'stop' requires staypointIds"
    );
    validateRequirement(
      id,
      props.mode !== 'value-change' || props.values !== undefined,
      "mode 'value-change' requires values"
    );
    if (props.temporalContext)
      resolveGPUTemporalContext(id, props.temporalContext, {
        encodings: [props.timestampFormat === 'uint32x2' ? 'int64-words' : 'float32-relative'],
        axisKinds: ['irregular']
      });
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must match rowCount`);
        const result = computeTrajectorySplitOnCPU(
          {
            positions,
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
            trackOffsets: getCPUOperationUint32Rows(id, inputs, 'trackOffsets'),
            parameters:
              props.parameters !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'parameters')
                : undefined,
            staypointIds:
              props.staypointIds !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'staypointIds')
                : undefined,
            values:
              props.values !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'values')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectorySplitResult]
          ])
        );
      }
    });
  }
}
