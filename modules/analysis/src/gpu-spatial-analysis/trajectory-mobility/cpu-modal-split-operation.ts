// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
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
import {getCPUTrajectoryDistance} from '../../analysis-core/operations/cpu-trajectory-distance';
import {GPU_MOVEMENT_SENTINEL} from '../trajectory-staypoints/movement-common';
import type {GPUModalSplitMetric} from './gpu-modal-split';
import {
  GPUModalSplitOperation,
  snapshotModalSplitOperationProps,
  type ModalSplitOperationProps
} from './gpu-modal-split-operation';

/** Trajectory rows and a sorted, capacity-sized tripleg table. */
export type CPUModalSplitInput = CPUTrajectoryRows & {
  triplegs: {
    tracks: Uint32Array;
    startRows: Uint32Array;
    endRows: Uint32Array;
    count: Uint32Array;
    overflow?: Uint32Array;
    candidateOverflow?: Uint32Array;
  };
  modes?: Uint32Array;
};
/** Plain numeric settings are compile-time, as in GPUModalSplit. */
export type CPUModalSplitOptions = {
  modeCount: number;
  metric: GPUModalSplitMetric;
  modeSpeedBreaks?: readonly number[];
  spatialContext?: GPUSpatialContext;
};
/** Track-major metric and predicted modes; incomplete input yields NaN and missing modes. */
export type CPUModalSplitResult = {modalSplit: Float32Array; predictedModes: Uint32Array};

function validateOptions(id: string, options: CPUModalSplitOptions): void {
  validateCapacity(id, 'modeCount', options.modeCount);
  validateRequirement(
    id,
    ['count', 'distance', 'duration'].includes(options.metric),
    "metric must be 'count', 'distance' or 'duration'"
  );
  if (options.modeSpeedBreaks) {
    validateRequirement(
      id,
      options.modeSpeedBreaks.length === options.modeCount,
      'modeSpeedBreaks must hold modeCount values'
    );
    validateRequirement(
      id,
      options.modeSpeedBreaks.every(
        (value, index, values) => index === 0 || value > values[index - 1]
      ),
      'modeSpeedBreaks must be strictly ascending'
    );
  }
}

/**
 * Classifies average tripleg speeds and accumulates count, path length or endpoint duration by
 * track and mode. Missing modes are excluded; other out-of-range keys enter the last mode. Speed
 * class bounds are exclusive and the final class is unbounded. As in the GPU literal builder,
 * nonfinite speed breaks become the largest finite float32.
 *
 * What differs from the GPU contributor:
 * - Lengths, speeds and sums use float64, with one rounding at the metric output. Near speed
 *   thresholds and for paths whose float32 length overflows, classifications can differ.
 * - Int64 timestamps are subtracted exactly before conversion to float64.
 * - Decreasing clamped trackOffsets and inconsistent, unordered or out-of-track tripleg arrays
 *   throw an Error. Empty triplegs are accepted for supplied-mode count/distance aggregation;
 *   prediction and duration aggregation still reject them because endRows minus one does not
 *   identify a member of an empty tripleg.
 * - Prediction omits the optional input modes port, which the contributor otherwise writes in
 *   place. predictedModes without prediction is omitted from the operation outputs.
 * - Nonfinite coordinates/times use JavaScript arithmetic; nonfinite predicted speeds are missing.
 *   The GPU does not specify nonfinite trajectory input behavior.
 *
 * @throws Error naming the inconsistent input array or missing modes.
 */
export function computeModalSplitOnCPU(
  input: CPUModalSplitInput,
  options: CPUModalSplitOptions
): CPUModalSplitResult {
  validateOptions('modal-split', options);
  const {rowCount, trackOffsets, getTimeDifference} = getCPUTrajectoryRows(input);
  const trackCount = trackOffsets.length - 1;
  const {triplegs} = input;
  const capacity = triplegs.tracks.length;
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    const flag = triplegs[name];
    if (flag && flag.length < 1)
      throw new Error(`modal-split triplegs.${name} must contain at least one row`);
  }
  if (triplegs.overflow?.[0] || triplegs.candidateOverflow?.[0]) {
    return {
      modalSplit: new Float32Array(trackCount * options.modeCount).fill(NaN),
      predictedModes: new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL)
    };
  }
  if (triplegs.startRows.length !== capacity)
    throw new Error('triplegs.startRows length must match triplegs.tracks');
  if (triplegs.endRows.length !== capacity)
    throw new Error('triplegs.endRows length must match triplegs.tracks');
  if (triplegs.count.length < 1) throw new Error('triplegs.count must contain at least one row');
  if (input.modes && input.modes.length !== capacity)
    throw new Error('modes length must match triplegs.tracks');
  if (!options.modeSpeedBreaks && !input.modes)
    throw new Error('modes is required without modeSpeedBreaks');
  const total = Math.min(triplegs.count[0], capacity);
  const needsDuration = Boolean(options.modeSpeedBreaks) || options.metric === 'duration';
  for (let tripleg = 0; tripleg < total; tripleg++) {
    const track = triplegs.tracks[tripleg];
    if (track >= trackCount || (tripleg > 0 && track < triplegs.tracks[tripleg - 1]))
      throw new Error('triplegs.tracks must contain sorted valid track indices');
    const start = triplegs.startRows[tripleg];
    const end = triplegs.endRows[tripleg];
    if (start < trackOffsets[track] || start > rowCount || start > end)
      throw new Error('triplegs.startRows must start an ordered range in its track');
    if (end > trackOffsets[track + 1]) throw new Error('triplegs.endRows must be within its track');
    if (needsDuration && start === end)
      throw new Error('triplegs.startRows must start a nonempty range for prediction or duration');
  }
  const context = resolveGPUSpatialContext('modal-split', options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  const radius = getGPUGreatCircleRadius(context);
  const totals = new Float64Array(trackCount * options.modeCount);
  const predictedModes = new Uint32Array(capacity);
  for (let tripleg = 0; tripleg < total; tripleg++) {
    const start = triplegs.startRows[tripleg];
    const end = triplegs.endRows[tripleg];
    const duration = needsDuration ? getTimeDifference(end - 1, start) : 0;
    let length = 0;
    if (options.modeSpeedBreaks || options.metric === 'distance')
      for (let row = start + 1; row < end; row++)
        length += getCPUTrajectoryDistance(input.positions, row - 1, row, radius);
    let mode = input.modes?.[tripleg] ?? GPU_MOVEMENT_SENTINEL;
    if (options.modeSpeedBreaks) {
      mode = GPU_MOVEMENT_SENTINEL;
      const speed = length / duration;
      if (duration > 0 && Number.isFinite(speed)) {
        mode = options.modeCount - 1;
        for (let candidate = 0; candidate < options.modeCount; candidate++) {
          const rawBreak = options.modeSpeedBreaks[candidate];
          const speedBreak = Number.isFinite(rawBreak) ? rawBreak : 3.4028234663852886e38;
          if (speed < speedBreak) {
            mode = candidate;
            break;
          }
        }
      }
      predictedModes[tripleg] = mode;
    }
    if (mode === GPU_MOVEMENT_SENTINEL) continue;
    mode = Math.min(mode, options.modeCount - 1);
    totals[triplegs.tracks[tripleg] * options.modeCount + mode] +=
      options.metric === 'count' ? 1 : options.metric === 'distance' ? length : duration;
  }
  return {modalSplit: Float32Array.from(totals), predictedModes};
}

/** CPU twin. Prediction uses an output-owned mode table instead of a writable input. */
export class CPUModalSplitOperation extends GPUCPUOperation {
  constructor(props: ModalSplitOperationProps) {
    props = snapshotModalSplitOperationProps(props);
    const id = props.id ?? 'modal-split';
    validateOptions(id, props);
    validateCapacity(id, 'rowCount', props.rowCount, 0);
    validateCapacity(id, 'trackCount', props.trackCount);
    validateCapacity(id, 'triplegCapacity', props.triplegCapacity, 0);
    validateCapacity(id, 'trackCount * modeCount', props.trackCount * props.modeCount);
    validateRequirement(
      id,
      Boolean(props.modeSpeedBreaks) || props.modes !== undefined,
      'modes or modeSpeedBreaks is required'
    );
    validateRequirement(
      id,
      props.timestampFormat === undefined ||
        ['float32', 'uint32x2'].includes(props.timestampFormat),
      'timestampFormat must be float32 or uint32x2'
    );
    resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    if (props.temporalContext)
      resolveGPUTemporalContext(id, props.temporalContext, {
        encodings: [props.timestampFormat === 'uint32x2' ? 'int64-words' : 'float32-relative'],
        axisKinds: ['irregular']
      });
    const operation = new GPUModalSplitOperation(props);
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
        const trackOffsets = getCPUOperationUint32Rows(id, inputs, 'trackOffsets');
        const tracks = getCPUOperationUint32Rows(id, inputs, 'triplegTracks');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must match rowCount`);
        if (trackOffsets.length !== props.trackCount + 1)
          throw new Error(`${id} trackOffsets length must match trackCount + 1`);
        if (tracks.length !== props.triplegCapacity)
          throw new Error(`${id} triplegs.tracks length must match triplegCapacity`);
        const result = computeModalSplitOnCPU(
          {
            positions,
            timestamps:
              props.timestampFormat === 'uint32x2'
                ? getCPUOperationUint32Rows(id, inputs, 'timestamps', 'uint32x2')
                : getCPUOperationFloat32Rows(id, inputs, 'timestamps'),
            trackOffsets,
            triplegs: {
              tracks,
              startRows: getCPUOperationUint32Rows(id, inputs, 'triplegStartRows'),
              endRows: getCPUOperationUint32Rows(id, inputs, 'triplegEndRows'),
              count: getCPUOperationUint32Rows(id, inputs, 'triplegCount'),
              overflow:
                props.triplegs.status.overflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'triplegOverflow')
                  : undefined,
              candidateOverflow:
                props.triplegs.status.candidateOverflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'triplegCandidateOverflow')
                  : undefined
            },
            modes:
              !props.modeSpeedBreaks && props.modes !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'modes')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUModalSplitResult]
          ])
        );
      }
    });
  }
}
