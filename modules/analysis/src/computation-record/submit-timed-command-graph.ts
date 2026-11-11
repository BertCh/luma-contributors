// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device, QuerySet} from '@luma.gl/core';
import type {
  CompiledGPUCommandGraph,
  GPUCommandGraphEncodeOptions,
  GPUCommandGraphEncoding,
  GPUCommandGraphTimingReport
} from '@luma.gl/graph/gpu-core';

/** One encoded and submitted run of a compiled graph, with its optional GPU timings. */
export type GPUCommandGraphTimedSubmission = {
  /** Published encoding statistics of the run. */
  encoding: GPUCommandGraphEncoding;
  /**
   * Whether per-node GPU timestamps were written. Timestamps give every compute node its own pass,
   * so a timestamped run is not coalesced (an observer effect; record it in `metrics[].encoding`).
   */
  timestamped: boolean;
  /**
   * Reads the per-node timings once after submission and frees the query set. Resolves to
   * `undefined` when no timestamps were written, so a record states no timing instead of zero.
   */
  readTimings(): Promise<GPUCommandGraphTimingReport | undefined>;
  /** Frees the query set without reading it; `readTimings` also frees it. */
  destroy(): void;
};

/**
 * Encodes and submits a compiled graph once for a computation record. When `timestamps` is not
 * false and the device has `timestamp-query`, the command encoder profiles every compute and render
 * pass through the public `timeProfilingQuerySet` option, and `readTimings` returns the published
 * `GPUCommandGraphTimingReport`, ready for `createGPUComputationMetrics`. Otherwise it encodes
 * exactly as a plain `compiled.encode` and `readTimings` resolves to `undefined`.
 */
export function submitGPUCommandGraphWithTimings<Parameters>(
  device: Device,
  compiled: CompiledGPUCommandGraph<Parameters>,
  encodeOptions: GPUCommandGraphEncodeOptions<Parameters>,
  options: {timestamps?: boolean} = {}
): GPUCommandGraphTimedSubmission {
  let querySet: QuerySet | undefined =
    options.timestamps !== false && device.features.has('timestamp-query')
      ? device.createQuerySet({
          type: 'timestamp',
          // Each timed pass writes a begin and an end timestamp.
          count: Math.max(2, 2 * compiled.stats.nodeOrder.length)
        })
      : undefined;
  const destroy = () => {
    querySet?.destroy();
    querySet = undefined;
  };
  const commandEncoder = device.createCommandEncoder(
    querySet ? {timeProfilingQuerySet: querySet} : {}
  );
  let encoding: GPUCommandGraphEncoding;
  try {
    encoding = compiled.encode(commandEncoder, encodeOptions);
    device.submit(commandEncoder.finish());
  } catch (error) {
    destroy();
    throw error;
  } finally {
    commandEncoder.destroy();
  }
  const timestamped = querySet !== undefined && encoding.canReadGPUTimings;
  return {
    encoding,
    timestamped,
    readTimings: async () => {
      try {
        return timestamped ? await encoding.readTimings() : undefined;
      } finally {
        destroy();
      }
    },
    destroy
  };
}
