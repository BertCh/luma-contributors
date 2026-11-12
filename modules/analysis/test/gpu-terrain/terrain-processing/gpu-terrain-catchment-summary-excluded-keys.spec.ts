// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('catchment radix keys include the exclusion sentinel and reset every output on re-encode', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'catchment-excluded-keys'});
  // GPUSort uses radix above 256 cells. Basin 16 needs one more key bit than basin 15.
  const cellCount = 257;
  const basinCount = 16;
  const histogramBins = 4;
  const cellArea = 1;
  const elevations = new Float32Array(cellCount).fill(1);
  const streams = new Uint32Array(cellCount);
  const directions = new Uint32Array(cellCount);
  const mixedBasinIds = new Uint32Array(cellCount);
  mixedBasinIds[128] = basinCount;
  mixedBasinIds[256] = basinCount - 1;
  const buffers: Buffer[] = [];
  const importInput = <Format extends 'float32' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, format, values.length)};
  };
  const importOutput = <Format extends 'float32' | 'uint32'>(
    id: string,
    length: number,
    format: Format
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, format, length)};
  };
  const basinIds = importInput('basin-ids', mixedBasinIds, 'uint32');
  const statistics = importOutput('statistics', basinCount * 6, 'float32');
  const hypsometry = importOutput('hypsometry', basinCount * histogramBins, 'float32');
  const profiles = importOutput('profiles', cellCount * 3, 'float32');
  const profileStatus = importOutput('profile-status', cellCount, 'uint32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: {width: cellCount, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      basinCount,
      histogramBins,
      elevation: importInput('elevation', elevations, 'float32').view,
      basinIds: basinIds.view,
      streams: importInput('streams', streams, 'uint32').view,
      flowDirections: importInput('directions', directions, 'uint32').view,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: profiles.view,
        profileStatus: profileStatus.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    // Imported buffers retain their binding in the compiled graph and can be rewritten.
    for (const [name, currentBasinIds] of [
      ['mixed', mixedBasinIds],
      ['all excluded', new Uint32Array(cellCount).fill(basinCount)],
      ['all included', new Uint32Array(cellCount)]
    ] as const) {
      basinIds.buffer.write(currentBasinIds);
      submitGraph(device, compiled, undefined);
      const expectedStatistics: number[] = [];
      const expectedHypsometry: number[] = [];
      const populations: number[] = [];
      for (let basin = 0; basin < basinCount; basin++) {
        const samples = Array.from(elevations).filter(
          (value, cell) => currentBasinIds[cell] === basin && Number.isFinite(value)
        );
        populations.push(samples.length);
        const area = samples.length * cellArea;
        const minimum = samples.length ? Math.min(...samples) : NaN;
        const maximum = samples.length ? Math.max(...samples) : NaN;
        const mean = samples.length
          ? samples.reduce((sum, value) => sum + value, 0) / samples.length
          : NaN;
        // Zero stream words produce no stream edges or drainage length.
        const streamLength = 0;
        expectedStatistics.push(
          area,
          mean,
          minimum,
          maximum,
          streamLength,
          area ? streamLength / area : NaN
        );
        for (let bin = 0; bin < histogramBins; bin++) {
          const threshold = minimum + ((maximum - minimum) * (bin + 1)) / histogramBins;
          expectedHypsometry.push(
            samples.filter(value => value <= threshold || bin + 1 === histogramBins).length *
              cellArea
          );
        }
      }
      if (name === 'mixed') {
        expect(populations).toEqual(
          Array.from({length: basinCount}, (_, basin) => (basin === 0 ? 255 : basin === 15 ? 1 : 0))
        );
      }
      expect(await readFloat32(statistics.buffer), `${name}: all six statistics per basin`).toEqual(
        expectedStatistics
      );
      expect(await readFloat32(hypsometry.buffer), `${name}: every cumulative bin`).toEqual(
        expectedHypsometry
      );
      // No stream cell has a profile; the center elevation is still published for excluded cells.
      const expectedProfiles = Array.from(elevations).flatMap(elevation => [NaN, elevation, NaN]);
      const expectedProfileStatus = Array.from(streams, () => 0);
      expect(await readFloat32(profiles.buffer), `${name}: profiles`).toEqual(expectedProfiles);
      expect(await readUint32(profileStatus.buffer), `${name}: profile status`).toEqual(
        expectedProfileStatus
      );
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
