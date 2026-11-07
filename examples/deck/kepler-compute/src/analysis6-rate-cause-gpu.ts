// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  KEPLER_RATE_CAUSES,
  addKeplerRateCauseCounts,
  type KeplerRateCause
} from '@luma-contributors/kepler';
import {packAnalysisVector} from './analysis-point-weights-cache';
import {describeRateCauseCounts} from './analysis2-policy-binding';

export type RateCauseColumns = {
  events: GPUVector<'float32'>;
  populations: GPUVector<'float32'>;
  eventValidity?: GPUVector<'uint32'>;
  populationValidity?: GPUVector<'uint32'>;
  /** Current selection (nonzero = selected); only selected rows are counted when present. */
  selection?: GPUVector<'uint32'>;
};

/**
 * Counts rate exclusion causes over every row of the GPU-resident event and population columns:
 * one atomic-counter pass and a nine-word readback, so no CPU copy of the source (and no row
 * sample) is involved. Works for GPU-resident results that have no CPU rows.
 */
export async function readGpuRateCauseCounts(
  device: Device,
  columns: RateCauseColumns
): Promise<Record<KeplerRateCause, number>> {
  const graph = new GPUCommandGraph(device, {id: 'kepler-rate-cause-counts'});
  const countsBuffer = device.createBuffer({
    id: 'kepler-rate-cause-counts',
    byteLength: KEPLER_RATE_CAUSES.length * 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const counts = graph.createDataView(
      graph.importBuffer(
        {id: 'counts', byteLength: countsBuffer.byteLength, usage: countsBuffer.usage},
        countsBuffer
      ),
      {format: 'uint32', length: KEPLER_RATE_CAUSES.length}
    );
    const packFloat32 = (name: string, vector: GPUVector<'float32'>) =>
      packAnalysisVector(graph, name, graph.importGPUVector(name, vector));
    const packUint32 = (name: string, vector: GPUVector<'uint32'> | undefined) =>
      vector ? packAnalysisVector(graph, name, graph.importGPUVector(name, vector)) : undefined;
    addKeplerRateCauseCounts(graph, {
      events: packFloat32('rate-events', columns.events),
      populations: packFloat32('rate-populations', columns.populations),
      eventValidity: packUint32('rate-event-validity', columns.eventValidity),
      populationValidity: packUint32('rate-population-validity', columns.populationValidity),
      selection: packUint32('rate-selection', columns.selection),
      counts
    });
    const program = graph.compile() as CompiledGPUCommandGraph<undefined>;
    compiled = program as CompiledGPUCommandGraph;
    const commandEncoder = device.createCommandEncoder({id: 'kepler-rate-cause-counts-encoder'});
    program.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const bytes = await countsBuffer.readAsync();
    const words = new Uint32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + KEPLER_RATE_CAUSES.length * 4)
    );
    return Object.fromEntries(
      KEPLER_RATE_CAUSES.map((cause, index) => [cause, words[index]])
    ) as Record<KeplerRateCause, number>;
  } finally {
    compiled?.destroy();
    countsBuffer.destroy();
  }
}

export function describeGpuRateCauseCounts(counts: Record<KeplerRateCause, number>): string {
  return describeRateCauseCounts(counts);
}
