// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {Buffer} from '@luma.gl/core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUNetworkFlow,
  GPUNetworkLouvain,
  GPUNetworkTimeRouting,
  GPUNetworkPointRouting,
  GPUNetworkFacilityOptimization,
  GPUNetworkRationalAccess
} from '../../../src/gpu-network/network-optimization/index';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/index';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/index';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Operation =
  | 'flow'
  | 'louvain'
  | 'time-routing'
  | 'point-routing'
  | 'facility'
  | 'raam'
  | 'kalman'
  | 'restriction'
  | 'similarity';
async function measure(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  operation: Operation,
  size: number,
  selective = false
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let uploadedBytes = 0;
  let outputBytes = 0;
  const uploadStart = performance.now();
  const input = <Format extends GPUVectorFormat>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    uploadedBytes += values.byteLength;
    return importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      format,
      values.byteLength / getGPUVectorFormatInfo(format).byteLength
    );
  };
  const output = <Format extends GPUVectorFormat>(format: Format, count: number) => {
    const byteLength = count * getGPUVectorFormatInfo(format).byteLength;
    const buffer = createOutputBuffer(device, byteLength / 4);
    buffers.push(buffer);
    outputBytes += byteLength;
    return {
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, count),
      buffer
    };
  };
  const summary = output('uint32', operation === 'flow' ? 8 : operation === 'similarity' ? 3 : 4);
  if (operation === 'flow') {
    const edges = Array.from({length: size - 1}, (_, edge) => [edge, edge + 1]).flat();
    graph.add(
      new GPUNetworkFlow({
        nodeCount: size,
        edges: input('uint32x2', Uint32Array.from(edges)),
        capacities: input('uint32', new Uint32Array(size - 1).fill(10)),
        costs: input('uint32', new Uint32Array(size - 1).fill(1)),
        parameters: input('uint32', Uint32Array.of(0, size - 1, 0xffffffff)),
        output: {
          flow: output('uint32', size - 1).view,
          cut: output('uint32', size).view,
          summary: summary.view
        }
      })
    );
  } else if (operation === 'louvain') {
    const weights = Float32Array.from({length: size ** 2}, (_, index) =>
      Math.floor(Math.floor(index / size) / 4) === Math.floor((index % size) / 4) &&
      Math.floor(index / size) !== index % size
        ? 1
        : 0
    );
    graph.add(
      new GPUNetworkLouvain({
        nodeCount: size,
        weights: input('float32', weights),
        parameters: input('float32', Float32Array.of(1, 1e-6)),
        output: {
          communities: output('uint32', size).view,
          objectiveTrace: output('float32', 129).view,
          summary: summary.view
        }
      })
    );
  } else if (operation === 'time-routing') {
    graph.add(
      new GPUNetworkTimeRouting({
        stateVertices: input(
          'uint32',
          Uint32Array.from({length: size}, (_, row) => row)
        ),
        transitions: input(
          'uint32x2',
          Uint32Array.from(Array.from({length: size - 1}, (_, row) => [row, row + 1]).flat())
        ),
        costs: input(
          'float32x2',
          Float32Array.from(Array.from({length: size - 1}, () => [1, 0.001]).flat())
        ),
        endpoints: input('uint32', Uint32Array.of(0, size - 1)),
        departure: input('float32', Float32Array.of(0)),
        output: {
          arrivals: output('float32', size).view,
          predecessorTransitions: output('uint32', size).view,
          summary: summary.view
        }
      })
    );
  } else if (operation === 'point-routing') {
    graph.add(
      new GPUNetworkPointRouting({
        costs: input('float32', new Float32Array(size).fill(1)),
        turns: input(
          'uint32x2',
          Uint32Array.from(Array.from({length: size - 1}, (_, row) => [row, row + 1]).flat())
        ),
        queries: input(
          'uint32x2',
          Uint32Array.from(Array.from({length: 4}, () => [0, size - 1]).flat())
        ),
        fractions: input(
          'float32x2',
          Float32Array.from(Array.from({length: 4}, () => [0.25, 0.75]).flat())
        ),
        output: {
          costs: output('float32', 4).view,
          predecessorEdges: output('uint32', 4 * size).view,
          status: summary.view
        }
      })
    );
  } else if (operation === 'facility') {
    graph.add(
      new GPUNetworkFacilityOptimization({
        operation: 'p-dispersion',
        facilityCount: size,
        selectCount: Math.min(size, 3),
        costs: input(
          'float32',
          Float32Array.from({length: size * size}, (_, row) =>
            Math.abs(Math.floor(row / size) - (row % size))
          )
        ),
        demand: input('float32', Float32Array.of(1)),
        capacities: input('float32', new Float32Array(size).fill(1)),
        threshold: input('float32', Float32Array.of(0)),
        output: {
          selectedFacilities: output('uint32', size).view,
          assignments: output('uint32', 1).view,
          objective: output('float32', 1).view,
          status: summary.view
        }
      })
    );
  } else if (operation === 'raam') {
    const facilities = Math.min(32, size);
    graph.add(
      new GPUNetworkRationalAccess({
        demandMode: 'fractional',
        costs: input(
          'float32',
          Float32Array.from({length: size * facilities}, (_, row) =>
            Math.abs((Math.floor(row / facilities) % facilities) - (row % facilities))
          )
        ),
        demand: input('float32', new Float32Array(size).fill(10)),
        supply: input('float32', new Float32Array(facilities).fill((size * 10) / facilities)),
        parameters: input('float32', Float32Array.of(10, 1, 0.2, 0.005, 50, 1e-5)),
        output: {
          assignments: output('float32', size * facilities).view,
          scores: output('float32', size).view,
          unservedDemand: output('float32', size).view,
          status: summary.view
        }
      })
    );
  } else if (operation === 'similarity') {
    const coordinates = Float32Array.from(
      Array.from({length: size}, (_, track) =>
        Array.from({length: 16}, (_, vertex) => [
          vertex,
          selective && track % 20 !== 0 ? 100 : 0.1
        ]).flat()
      ).flat()
    );
    graph.add(
      new GPUTrackSimilaritySearch({
        metric: 'frechet',
        maximumFrechetVertexCount: 32,
        survivorCapacity: selective ? Math.ceil(size / 20) : size,
        linesA: {
          kind: 'lines',
          positions: input(
            'float32x2',
            Float32Array.from(Array.from({length: 16}, (_, vertex) => [vertex, 0]).flat())
          ),
          lineOffsets: input('uint32', Uint32Array.of(0, 16))
        },
        linesB: {
          kind: 'lines',
          positions: input('float32x2', coordinates),
          lineOffsets: input(
            'uint32',
            Uint32Array.from({length: size + 1}, (_, row) => row * 16)
          )
        },
        pairA: input('uint32', new Uint32Array(size)),
        pairB: input(
          'uint32',
          Uint32Array.from({length: size}, (_, row) => row)
        ),
        threshold: input('float32', Float32Array.of(1)),
        output: {
          distance: output('float32', size).view,
          lowerBound: output('float32', size).view,
          matched: output('uint32', size).view,
          status: output('uint32', size).view,
          survivorSummary: summary.view
        }
      })
    );
  } else {
    const positions = input(
      'float32x2',
      Float32Array.from(Array.from({length: size}, (_, row) => [row % 16, Math.sin(row)]).flat())
    );
    const times = input(
      'float32',
      Float32Array.from({length: size}, (_, row) => row % 16)
    );
    const offsets = input(
      'uint32',
      Uint32Array.from({length: size / 16 + 1}, (_, row) => row * 16)
    );
    if (operation === 'kalman')
      graph.add(
        new GPUTrajectoryKalman({
          positions,
          times,
          offsets,
          parameters: input('float32', Float32Array.of(0.2, 0.5, 1, 5)),
          output: {
            filteredState: output('float32x4', size).view,
            filteredCovariance: output('float32x4', size).view,
            status: output('uint32', size).view,
            smoothedState: output('float32x4', size).view,
            smoothedCovariance: output('float32x4', size).view
          }
        })
      );
    else
      graph.add(
        new GPUTrajectoryRestriction({
          positions,
          times,
          offsets,
          parameters: input('float32', Float32Array.of(0, 15, 2, -2, 10, 2, 5)),
          timeBoundary: 'left-closed',
          output: {
            intervals: output('float32x2', size).view,
            positions: output('float32x4', size).view,
            status: output('uint32', size).view,
            relationships: output('uint32x4', size / 16).view
          }
        })
      );
  }
  const uploadAndPreparationMilliseconds = performance.now() - uploadStart;
  const compileStart = performance.now();
  const compiled = graph.compile();
  const compilationMilliseconds = performance.now() - compileStart;
  try {
    const executions: number[] = [];
    for (let run = 0; run < 4; run++) {
      await device.handle.queue.onSubmittedWorkDone();
      const started = performance.now();
      submitGraph(device, compiled, undefined);
      await device.handle.queue.onSubmittedWorkDone();
      executions.push(performance.now() - started);
    }
    const readStart = performance.now();
    const status = await readUint32(summary.buffer, summary.view.length);
    const readbackMilliseconds = performance.now() - readStart;
    if (operation === 'similarity') expect(status[2]).toBe(0);
    if (['flow', 'louvain', 'time-routing', 'facility', 'raam'].includes(operation))
      expect(status[0]).toBe(1);
    if (['flow', 'louvain', 'time-routing'].includes(operation)) expect(status[1]).toBe(1);
    if (operation === 'facility') expect(status[2]).toBe(1);
    return {
      operation,
      size,
      selective,
      uploadedBytes,
      outputBytes,
      uploadAndPreparationMilliseconds,
      compilationMilliseconds,
      firstExecutionMilliseconds: executions[0],
      medianWarmMilliseconds: executions.slice(1).sort((first, second) => first - second)[1],
      readbackMilliseconds,
      status
    };
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
}

it('network and movement supported-envelope scaling with GPU-resident repeat execution', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  const workloads: [Operation, number[]][] = [
    ['flow', [8, 32, 128]],
    ['louvain', [4, 16, 32]],
    ['time-routing', [8, 64, 256]],
    ['point-routing', [8, 64, 128]],
    ['facility', [4, 8, 16]],
    ['raam', [4, 16, 64]],
    ['kalman', [32, 512, 4096]],
    ['restriction', [32, 512, 4096]],
    ['similarity', [64, 1024, 4096]]
  ];
  for (const [operation, sizes] of workloads)
    for (const size of sizes) {
      results.push(await measure(device, operation, size));
      if (operation === 'similarity') results.push(await measure(device, operation, size, true));
    }
  console.log(`NETWORK_MOVEMENT_BENCHMARK ${JSON.stringify({device: device.info, results})}`);
}, 120000);
