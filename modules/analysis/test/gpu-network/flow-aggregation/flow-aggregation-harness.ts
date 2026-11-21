// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared by the flow aggregation correctness spec and its timing benchmark
// (`gpu-flow-aggregation-bench.spec.ts`).

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUFlowAggregation,
  type GPUFlowAggregationProps
} from '../../../src/gpu-network/flow-aggregation/index';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSeededRandom} from './flow-aggregation-oracle';

const GARBAGE = 0xdeadbeef;

export type FlowResult = {
  ids: number[];
  count: number;
  requiredCount: number;
  overflow: number;
  candidateOverflow: number;
  pairOverflow: number;
  drawInstanceCount: number;
  originZones: number[];
  destinationZones: number[];
  counts: number[];
  weights: number[];
  zoneOutCounts: number[];
  zoneInCounts: number[];
  zoneOutWeights: number[];
  zoneInWeights: number[];
};

export type Harness = {
  compiled: CompiledGPUCommandGraph<void>;
  submit(): void;
  read(): Promise<FlowResult>;
  destroy(): void;
};

export type HarnessOptions = {
  id: string;
  zoneCount: number;
  topCount: number;
  pairCapacity: number;
  hasWeights: boolean;
  maximumProbeCount?: number;
  createProps: (
    graph: GPUCommandGraph<void>
  ) => Pick<GPUFlowAggregationProps, 'zones'> & Partial<GPUFlowAggregationProps>;
};

/** Builds one flow-aggregation graph with every output imported, scrambled before each submit. */
export function createHarness(device: Device, options: HarnessOptions): Harness {
  const {topCount, zoneCount, hasWeights} = options;
  const buffers: Buffer[] = [];
  const outputs: {buffer: Buffer; length: number}[] = [];
  const graph = new GPUCommandGraph<void>(device, {id: options.id});
  const createOutput = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs.push({buffer, length: Math.max(length, 1)});
    return {
      buffer,
      view: importGraphBuffer(graph, `${options.id}-${name}`, buffer, format, length)
    };
  };
  const ids = createOutput('ids', 'uint32', topCount);
  const count = createOutput('count', 'uint32', 1);
  const overflow = createOutput('overflow', 'uint32', 1);
  const candidateOverflow = createOutput('candidate-overflow', 'uint32', 1);
  const requiredCount = createOutput('total', 'uint32', 1);
  const pairOverflow = createOutput('pair-overflow', 'uint32', 1);
  const drawInstanceCount = createOutput('draw', 'uint32', 1);
  const originZones = createOutput('origin-zones', 'uint32', topCount);
  const destinationZones = createOutput('destination-zones', 'uint32', topCount);
  const counts = createOutput('counts', 'uint32', topCount);
  const weights = createOutput('weights', 'float32', topCount);
  const zoneOutCounts = createOutput('zone-out-counts', 'uint32', zoneCount);
  const zoneInCounts = createOutput('zone-in-counts', 'uint32', zoneCount);
  const zoneOutWeights = createOutput('zone-out-weights', 'float32', zoneCount);
  const zoneInWeights = createOutput('zone-in-weights', 'float32', zoneCount);

  const contributor = new GPUFlowAggregation({
    id: options.id,
    pairCapacity: options.pairCapacity,
    maximumProbeCount: options.maximumProbeCount,
    ...options.createProps(graph),
    output: {
      ids: ids.view,
      status: {
        count: count.view,
        overflow: overflow.view,
        candidateOverflow: candidateOverflow.view,
        requiredCount: requiredCount.view,
        pairOverflow: pairOverflow.view
      },
      drawInstanceCount: drawInstanceCount.view,
      flowOriginZoneIds: originZones.view,
      flowDestinationZoneIds: destinationZones.view,
      flowCounts: counts.view,
      flowWeights: weights.view,
      zoneOutCounts: zoneOutCounts.view,
      zoneInCounts: zoneInCounts.view,
      ...(hasWeights
        ? {
            zoneOutWeights: zoneOutWeights.view,
            zoneInWeights: zoneInWeights.view
          }
        : {})
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    compiled,
    submit: () => {
      for (const output of outputs) {
        output.buffer.write(new Uint32Array(output.length).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
    },
    read: async () => ({
      ids: await readUint32(ids.buffer, topCount),
      count: (await readUint32(count.buffer, 1))[0],
      requiredCount: (await readUint32(requiredCount.buffer, 1))[0],
      overflow: (await readUint32(overflow.buffer, 1))[0],
      candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
      pairOverflow: (await readUint32(pairOverflow.buffer, 1))[0],
      drawInstanceCount: (await readUint32(drawInstanceCount.buffer, 1))[0],
      originZones: await readUint32(originZones.buffer, topCount),
      destinationZones: await readUint32(destinationZones.buffer, topCount),
      counts: await readUint32(counts.buffer, topCount),
      weights: await readFloat32(weights.buffer, topCount),
      zoneOutCounts: await readUint32(zoneOutCounts.buffer, zoneCount),
      zoneInCounts: await readUint32(zoneInCounts.buffer, zoneCount),
      zoneOutWeights: await readFloat32(zoneOutWeights.buffer, zoneCount),
      zoneInWeights: await readFloat32(zoneInWeights.buffer, zoneCount)
    }),
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Seeded random OD zone IDs. */
export function createZoneIds(seed: number, rowCount: number, zoneCount: number, outOfRange = 0) {
  const next = createSeededRandom(seed);
  const origins: number[] = [];
  const destinations: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    origins.push(Math.floor(next() * (zoneCount + outOfRange)));
    destinations.push(Math.floor(next() * (zoneCount + outOfRange)));
  }
  return {origins, destinations};
}

/** Weights that expose summation order: 1e8, 1, fractions, and non-finite values. */
export function createOrderSensitiveWeights(seed: number, rowCount: number): Float32Array {
  const next = createSeededRandom(seed);
  return Float32Array.from({length: rowCount}, (_, row) => {
    if (row % 53 === 7) return Number.NaN;
    const draw = next();
    if (draw < 0.3) return 1e8;
    if (draw < 0.6) return 1;
    return Math.fround(next() * 3.3);
  });
}
