// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTerrainBreach} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainDepressionHierarchy} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Opt in with VITE_TERRAIN_PROCESSING_BENCHMARK=true yarn test-browser-benchmarks
// --maxWorkers=1 modules/graph/test/gpu-terrain/terrain-processing/terrain-processing-benchmark.spec.ts
const enabled =
  (import.meta as unknown as {env?: Record<string, string>}).env
    ?.VITE_TERRAIN_PROCESSING_BENCHMARK === 'true';
const modes = ['breach', 'transport', 'hierarchy', 'catchment', 'dem', 'tiled'] as const;

for (const mode of modes) {
  it.skipIf(!enabled)(`terrain phase measurements: ${mode}`, {timeout: 120000}, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const queue = (device as unknown as {handle: GPUDevice}).handle.queue;
    for (const width of [8, 128]) {
      const count = width * width;
      const grid = {
        width,
        height: width,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      };
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const uploads: {buffer: Buffer; data: Float32Array | Uint32Array}[] = [];
      const downloads: {buffer: Buffer; format: 'float32' | 'uint32'; length: number}[] = [];
      const allocationStart = performance.now();
      const input = <Format extends 'float32' | 'uint32'>(
        format: Format,
        data: Float32Array | Uint32Array
      ) => {
        const buffer = device.createBuffer({
          data,
          usage: Buffer.STORAGE | Buffer.COPY_DST
        });
        buffers.push(buffer);
        uploads.push({buffer, data});
        return importGraphBuffer(graph, `input-${buffers.length}`, buffer, format, data.length);
      };
      const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
        const buffer = device.createBuffer({
          byteLength: Math.max(1, length) * 4,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        buffers.push(buffer);
        downloads.push({buffer, format, length});
        return importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, length);
      };
      const elevation = Float32Array.from({length: count}, (_, cell) => width - (cell % width));
      const directions = Uint32Array.from({length: count}, (_, cell) =>
        Number(cell % width !== width - 1)
      );
      const fractions = new Float32Array(count * 8);
      for (let cell = 0; cell < count; cell++) fractions[cell * 8] = directions[cell];
      let contributor: GPUCommandNodeProducer;
      let verify: (values: number[][]) => void;
      let cpuReferenceMilliseconds: number | null = null;
      if (mode === 'breach') {
        const pitCount = Math.min(width, 64);
        for (let row = 0; row < pitCount; row++) {
          elevation[row * width] = 2;
          elevation[row * width + 1] = 5;
          elevation[row * width + 2] = 0;
        }
        contributor = new GPUTerrainBreach({
          grid,
          elevation: input('float32', elevation),
          pitCells: input(
            'uint32',
            Uint32Array.from({length: pitCount}, (_, row) => row * width)
          ),
          maximumSteps: 2,
          maximumCutDepth: 4,
          elevationDecrement: 0.5,
          output: {
            elevation: output('float32', count),
            cutDepth: output('float32', count),
            modificationMask: output('uint32', count),
            pathCells: output('uint32', pitCount * 3),
            pathLengths: output('uint32', pitCount),
            pitStatus: output('uint32', pitCount),
            unresolvedCount: output('uint32', 1)
          }
        });
        verify = values => {
          expect(values[6]).toEqual([0]);
          expect(values[5]).toEqual(Array(pitCount).fill(1));
        };
      } else if (mode === 'transport') {
        contributor = new GPUTerrainBranchingFlow({
          grid,
          fractions: input('float32', fractions),
          values: input('float32', new Float32Array(count).fill(1)),
          mode: 'transport',
          maximumIterations: 16,
          output: {
            values: output('float32', count),
            status: output('uint32', count),
            unresolvedCount: output('uint32', 1)
          }
        });
        // Matching directed-chain CPU recurrence, including output allocation. This is a
        // best-case transport baseline, not an optimized general branching solver.
        const cpuStart = performance.now();
        const expected = new Float32Array(count).fill(1);
        for (let cell = 0; cell < count; cell++) {
          if (directions[cell]) expected[cell + 1] += expected[cell];
        }
        cpuReferenceMilliseconds = performance.now() - cpuStart;
        verify = values => {
          expect(values[0]).toEqual(Array.from(expected));
          expect(values[2]).toEqual([0]);
        };
      } else if (mode === 'hierarchy') {
        for (let cell = 0; cell < count; cell++) {
          elevation[cell] = ((cell % width) * 17 + Math.floor(cell / width) * 37) % 101;
        }
        contributor = new GPUTerrainDepressionHierarchy({
          grid,
          elevation: input('float32', elevation),
          outlets: input('uint32', new Uint32Array(count)),
          stages: input('float32', new Float32Array([101])),
          nodeCapacity: count,
          output: {
            nodes: output('uint32', count * 4),
            levels: output('float32', count * 2),
            cellNodes: output('uint32', count),
            curves: output('float32', count * 2),
            status: output('uint32', 3)
          }
        });
        verify = values => {
          expect(values[4][2]).toBe(0);
          const root = values[4][0] - 1;
          expect(values[0][root * 4]).toBe(0xffffffff);
          expect(values[3][root * 2]).toBe(count);
          expect(values[3][root * 2 + 1]).toBe(
            elevation.reduce((sum, value) => sum + 101 - value, 0)
          );
        };
      } else if (mode === 'catchment') {
        contributor = new GPUTerrainCatchmentSummary({
          grid,
          elevation: input('float32', elevation),
          basinIds: input(
            'uint32',
            Uint32Array.from({length: count}, (_, cell) => Math.floor(cell / width))
          ),
          basinCount: width,
          streams: input('uint32', new Uint32Array(count).fill(1)),
          flowDirections: input('uint32', directions),
          histogramBins: 8,
          output: {
            statistics: output('float32', width * 6),
            hypsometry: output('float32', width * 8),
            profiles: output('float32', count * 3),
            profileStatus: output('uint32', count)
          }
        });
        verify = values => {
          expect(values[0].slice(0, 5)).toEqual([width, (width + 1) / 2, 1, width, width - 1]);
          expect(values[3]).toEqual(Array(count).fill(1));
        };
      } else if (mode === 'dem') {
        contributor = new GPUTerrainDEMCoregistration({
          grid,
          reference: input('float32', elevation),
          moving: input(
            'float32',
            Float32Array.from(elevation, value => value + 2)
          ),
          stableMask: input('uint32', new Uint32Array(count).fill(1)),
          differenceVariance: input('float32', new Float32Array(count).fill(1)),
          output: {
            fit: output('float32', 4),
            differences: output('float32', count * 2),
            validity: output('uint32', count),
            volume: output('float32', 3)
          }
        });
        verify = values => {
          expect(values[0]).toEqual([2, 0, count, 0]);
          expect(values[3]).toEqual([0, width, count]);
        };
      } else {
        const flowDirections = input('uint32', directions);
        const runoff = input('float32', new Float32Array(count).fill(1));
        const streams = input('uint32', new Uint32Array(count).fill(1));
        contributor = new GPUTerrainTiledFlow({
          grid,
          maximumIterations: 16,
          tiles: [0, 1].map(side => ({
            column: 0,
            row: 0,
            width,
            height: width,
            owned: [(side * width) / 2, 0, width / 2, width] as const,
            flowDirections,
            runoff,
            streams,
            output: {
              accumulation: output('float32', count),
              status: output('uint32', count),
              reachIds: output('uint32', count)
            }
          }))
        });
        verify = values => {
          for (let side = 0; side < 2; side++) {
            for (let cell = 0; cell < count; cell++) {
              const owned = Math.floor((cell % width) / (width / 2)) === side;
              expect(values[side * 3 + 1][cell]).toBe(Number(owned));
              if (owned) expect(values[side * 3][cell]).toBe((cell % width) + 1);
            }
          }
        };
      }
      await queue.onSubmittedWorkDone();
      const allocationUploadMilliseconds = performance.now() - allocationStart;
      const declarationStart = performance.now();
      graph.add(contributor);
      const declarationMilliseconds = performance.now() - declarationStart;
      const compileStart = performance.now();
      const compiled = graph.compile();
      const compileMilliseconds = performance.now() - compileStart;
      const querySet = device.features.has('timestamp-query')
        ? device.createQuerySet({type: 'timestamp', count: compiled.stats.nodeOrder.length * 2})
        : undefined;
      try {
        const execute = async () => {
          const start = performance.now();
          const encoder = device.createCommandEncoder({timeProfilingQuerySet: querySet});
          const encoding = compiled.encode(encoder, {parameters: undefined});
          device.submit(encoder.finish());
          await queue.onSubmittedWorkDone();
          const submitFenceMilliseconds = performance.now() - start;
          const timing = await encoding.readTimings();
          return {submitFenceMilliseconds, gpuMilliseconds: timing.gpuTimeMilliseconds ?? null};
        };
        const first = await execute();
        const repeated = [];
        for (let repeat = 0; repeat < 3; repeat++) {
          const updateStart = performance.now();
          for (const {buffer, data} of uploads) buffer.write(data);
          await queue.onSubmittedWorkDone();
          const updateUploadMilliseconds = performance.now() - updateStart;
          repeated.push({updateUploadMilliseconds, ...(await execute())});
        }
        const readStart = performance.now();
        const values = await Promise.all(
          downloads.map(async ({buffer, format, length}) => {
            const bytes = await buffer.readAsync();
            return Array.from(
              format === 'float32'
                ? new Float32Array(bytes.buffer, bytes.byteOffset, length)
                : new Uint32Array(bytes.buffer, bytes.byteOffset, length)
            );
          })
        );
        const readbackMilliseconds = performance.now() - readStart;
        verify(values);
        console.log(
          'TERRAIN_PHASE_BENCHMARK',
          JSON.stringify({
            mode,
            width,
            count,
            device: device.info,
            maximumBindingBytes: device.limits.maxStorageBufferBindingSize,
            allocationUploadMilliseconds,
            declarationMilliseconds,
            compileMilliseconds,
            first,
            repeated,
            readbackMilliseconds,
            cpuReferenceMilliseconds,
            inputBytes: uploads.reduce((sum, entry) => sum + entry.data.byteLength, 0),
            outputBytes: downloads.reduce((sum, entry) => sum + entry.buffer.byteLength, 0),
            peakResidentScratchBytes: compiled.stats.physicalTransientResourceBytes,
            nodeCount: compiled.stats.nodeOrder.length
          })
        );
      } finally {
        querySet?.destroy();
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    }
  });
}
