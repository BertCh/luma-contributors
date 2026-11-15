// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let identifier = 0;
  const input = <Format extends 'uint32' | 'float32'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `input-${identifier++}`, buffer, format, values.length)
    };
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length),
      read: async () =>
        Array.from(
          await (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
        )
    };
  };
  const status = () => ({
    converged: output(1, 'uint32'),
    iterationLimitReached: output(1, 'uint32'),
    invalidCount: output(1, 'uint32')
  });
  return {device, graph, input, output, status, buffers};
}

it('joins cross-tile signed partials exactly and resets empty basins and cumulative bins on re-encode', async () => {
  const {device, graph, input, output, status, buffers} = await setup();
  const cellCount = 1025;
  const magnitude = 2 ** 60;
  const elevation = new Float32Array(cellCount);
  elevation[0] = magnitude;
  elevation[800] = -magnitude;
  elevation[512] = 1;
  elevation[1023] = 7;
  elevation[1024] = 9;
  const basinIds = new Uint32Array(cellCount);
  basinIds[1023] = 2;
  basinIds[1024] = 4;
  const validity = new Uint32Array(cellCount).fill(1);
  validity[257] = 0;
  const basins = input(basinIds, 'uint32');
  const statistics = output(24, 'float32'),
    hypsometry = output(16, 'float32'),
    profiles = output(cellCount * 3, 'float32'),
    profileStatus = output(cellCount, 'uint32');
  const diagnostics = status();
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: {
        width: cellCount,
        height: 1,
        affine: [0.7, 0, 0, 0, -1, 0],
        pixelInterpretation: 'area'
      },
      elevation: input(elevation, 'float32').view,
      validity: input(validity, 'uint32').view,
      basinIds: basins.view,
      basinCount: 4,
      histogramBins: 4,
      streams: input(new Uint32Array(cellCount), 'uint32').view,
      flowDirections: input(new Uint32Array(cellCount), 'uint32').view,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: profiles.view,
        profileStatus: profileStatus.view,
        status: {
          converged: diagnostics.converged.view,
          iterationLimitReached: diagnostics.iterationLimitReached.view,
          invalidCount: diagnostics.invalidCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const excluded of [false, true, false]) {
      basins.buffer.write(excluded ? new Uint32Array(cellCount).fill(0xffffffff) : basinIds);
      submitGraph(device, compiled, undefined);
      const area = Math.fround(0.7);
      // Hand-derived exact integer sum: 2^60 + 1 - 2^60 = 1 across three different tiles.
      const expected = excluded
        ? Array.from({length: 4}, () => [0, NaN, NaN, NaN, 0, NaN]).flat()
        : [
            Math.fround(1022 * area),
            Math.fround(1 / 1022),
            -magnitude,
            magnitude,
            0,
            0,
            0,
            NaN,
            NaN,
            NaN,
            0,
            NaN,
            area,
            7,
            7,
            7,
            0,
            0,
            0,
            NaN,
            NaN,
            NaN,
            0,
            NaN
          ];
      expect(await statistics.read()).toEqual(expected);
      const expectedBins = excluded
        ? new Array(16).fill(0)
        : [1, 1020, 1021, 1022, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0].map(count =>
            Math.fround(count * area)
          );
      expect(await hypsometry.read()).toEqual(expectedBins);
      expect(await profileStatus.read()).toEqual(new Array(cellCount).fill(0));
      expect(await diagnostics.converged.read()).toEqual([1]);
      expect(await diagnostics.iterationLimitReached.read()).toEqual([0]);
      expect(await diagnostics.invalidCount.read()).toEqual([cellCount]);
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('publishes named catchment and owned-only tiled diagnostics for resolved, cyclic and invalid re-encodes', async () => {
  const {device, graph, input, output, status, buffers} = await setup();
  const directions = input(Uint32Array.of(1, 1, 1, 0), 'uint32');
  const streams = input(new Uint32Array(4).fill(1), 'uint32');
  const elevation = input(Float32Array.of(4, 3, 2, 1), 'float32');
  const runoff = input(Float32Array.of(100, 1, 1, 1, 1, 100), 'float32');
  const tileDirections = input(Uint32Array.of(0xffffffff, 1, 1, 1, 0, 0xffffffff), 'uint32');
  const catchmentStatus = status(),
    tileStatus = status();
  const grid = {
    width: 4,
    height: 1,
    affine: [1, 0, 0, 0, -1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const named = (diagnostics: ReturnType<typeof status>) => ({
    converged: diagnostics.converged.view,
    iterationLimitReached: diagnostics.iterationLimitReached.view,
    invalidCount: diagnostics.invalidCount.view
  });
  const profileStatus = output(4, 'uint32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid,
      maximumSteps: 2,
      elevation: elevation.view,
      basinIds: input(new Uint32Array(4), 'uint32').view,
      basinCount: 1,
      histogramBins: 2,
      streams: streams.view,
      flowDirections: directions.view,
      output: {
        statistics: output(6, 'float32').view,
        hypsometry: output(2, 'float32').view,
        profiles: output(12, 'float32').view,
        profileStatus: profileStatus.view,
        status: named(catchmentStatus)
      }
    })
  );
  const tileStates = output(6, 'uint32'),
    accumulation = output(6, 'float32');
  graph.add(
    new GPUTerrainTiledFlow({
      grid,
      maximumIterations: 2,
      tiles: [
        {
          column: -1,
          row: 0,
          width: 6,
          height: 1,
          owned: [1, 0, 4, 1],
          flowDirections: tileDirections.view,
          runoff: runoff.view,
          streams: input(Uint32Array.of(0xffffffff, 1, 1, 1, 1, 0xffffffff), 'uint32').view,
          output: {
            accumulation: accumulation.view,
            status: tileStates.view,
            reachIds: output(6, 'uint32').view
          }
        }
      ],
      output: {status: named(tileStatus)}
    })
  );
  const compiled = graph.compile();
  try {
    for (const scene of ['bounded', 'cycle', 'invalid', 'resolved'] as const) {
      const chain = Uint32Array.of(1, 1, 1, 0);
      if (scene === 'cycle') chain[1] = 16;
      if (scene === 'resolved') chain.fill(0);
      directions.buffer.write(chain);
      tileDirections.buffer.write(Uint32Array.of(0xffffffff, ...chain, 0xffffffff));
      streams.buffer.write(new Uint32Array(4).fill(scene === 'invalid' ? 0 : 1));
      runoff.buffer.write(
        scene === 'invalid' ? new Float32Array(6).fill(NaN) : Float32Array.of(100, 1, 1, 1, 1, 100)
      );
      submitGraph(device, compiled, undefined);
      const expectedProfiles =
        scene === 'bounded'
          ? [2, 2, 1, 1]
          : scene === 'cycle'
            ? [2, 2, 1, 1]
            : new Array(4).fill(scene === 'invalid' ? 0 : 1);
      expect(await profileStatus.read(), scene).toEqual(expectedProfiles);
      for (const [diagnostics, unresolved, limited] of [
        [
          catchmentStatus,
          scene === 'bounded' || scene === 'cycle',
          scene === 'bounded' || scene === 'cycle'
        ],
        [tileStatus, scene === 'cycle', false]
      ] as const) {
        expect(await diagnostics.converged.read(), scene).toEqual([unresolved ? 0 : 1]);
        expect(await diagnostics.iterationLimitReached.read(), scene).toEqual([limited ? 1 : 0]);
        expect(await diagnostics.invalidCount.read(), scene).toEqual([scene === 'invalid' ? 4 : 0]);
      }
      const states = await tileStates.read();
      expect(states[0]).toBe(0);
      expect(states[5]).toBe(0);
      if (scene === 'bounded') expect(await accumulation.read()).toEqual([NaN, 1, 2, 3, 4, NaN]);
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('distinguishes a stalled tiled cycle from exhaustion of a progressing solver budget', async () => {
  const {device, graph, input, output, status, buffers} = await setup();
  const cases = [
    {name: 'two-cell cycle', directions: Uint32Array.of(1, 16), maximumIterations: 128, limited: 0},
    // Cell 2 is independent and finalizes in the first sweep; the cycle cannot finalize.
    {
      name: 'cycle with progress at the limit',
      directions: Uint32Array.of(1, 16, 0),
      maximumIterations: 1,
      limited: 1
    },
    {
      name: 'cycle with progress then an early stall',
      directions: Uint32Array.of(1, 16, 0),
      maximumIterations: 128,
      limited: 0
    }
  ];
  const results = cases.map(scene => {
    const cellCount = scene.directions.length;
    const directions = input(scene.directions, 'uint32');
    const diagnostics = status();
    const cellStates = output(cellCount, 'uint32');
    graph.add(
      new GPUTerrainTiledFlow({
        id: scene.name,
        grid: {
          width: cellCount,
          height: 1,
          affine: [1, 0, 0, 0, -1, 0],
          pixelInterpretation: 'area'
        },
        maximumIterations: scene.maximumIterations,
        tiles: [
          {
            column: 0,
            row: 0,
            width: cellCount,
            height: 1,
            owned: [0, 0, cellCount, 1],
            flowDirections: directions.view,
            runoff: input(new Float32Array(cellCount).fill(1), 'float32').view,
            streams: input(new Uint32Array(cellCount).fill(1), 'uint32').view,
            output: {
              accumulation: output(cellCount, 'float32').view,
              status: cellStates.view,
              reachIds: output(cellCount, 'uint32').view
            }
          }
        ],
        output: {
          status: {
            converged: diagnostics.converged.view,
            iterationLimitReached: diagnostics.iterationLimitReached.view,
            invalidCount: diagnostics.invalidCount.view
          }
        }
      })
    );
    return {scene, directions, diagnostics, cellStates};
  });
  const compiled = graph.compile();
  try {
    for (const resolved of [false, true, false]) {
      for (const result of results)
        result.directions.buffer.write(
          resolved ? new Uint32Array(result.scene.directions.length) : result.scene.directions
        );
      submitGraph(device, compiled, undefined);
      for (const {scene, diagnostics, cellStates} of results) {
        expect(await diagnostics.converged.read(), scene.name).toEqual([resolved ? 1 : 0]);
        expect(await diagnostics.iterationLimitReached.read(), scene.name).toEqual([
          resolved ? 0 : scene.limited
        ]);
        expect(await diagnostics.invalidCount.read(), scene.name).toEqual([0]);
        expect(await cellStates.read(), scene.name).toEqual(
          resolved
            ? new Array(scene.directions.length).fill(1)
            : [2, 2, ...new Array(scene.directions.length - 2).fill(1)]
        );
      }
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('does not label a failed catchment dependency as exhaustion of its step budget', async () => {
  const {device, graph, input, output, status, buffers} = await setup();
  const directions = input(Uint32Array.of(1, 0xffffffff), 'uint32');
  const diagnostics = status();
  const states = output(2, 'uint32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: {width: 2, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
      maximumSteps: 128,
      basinCount: 1,
      histogramBins: 2,
      elevation: input(Float32Array.of(2, 1), 'float32').view,
      basinIds: input(new Uint32Array(2), 'uint32').view,
      streams: input(Uint32Array.of(1, 1), 'uint32').view,
      flowDirections: directions.view,
      output: {
        statistics: output(6, 'float32').view,
        hypsometry: output(2, 'float32').view,
        profiles: output(6, 'float32').view,
        profileStatus: states.view,
        status: {
          converged: diagnostics.converged.view,
          iterationLimitReached: diagnostics.iterationLimitReached.view
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const scene of ['failed dependency', 'cycle', 'resolved', 'failed dependency'] as const) {
      directions.buffer.write(
        scene === 'cycle'
          ? Uint32Array.of(1, 16)
          : scene === 'resolved'
            ? Uint32Array.of(1, 0)
            : Uint32Array.of(1, 0xffffffff)
      );
      submitGraph(device, compiled, undefined);
      expect(await states.read(), scene).toEqual(scene === 'resolved' ? [1, 1] : [2, 2]);
      expect(await diagnostics.converged.read(), scene).toEqual([scene === 'resolved' ? 1 : 0]);
      expect(await diagnostics.iterationLimitReached.read(), scene).toEqual([
        scene === 'cycle' ? 1 : 0
      ]);
    }
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});
