// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUTerrainHeightAboveDrainage,
  GPU_TERRAIN_DRAINAGE_MULTIPLE,
  GPU_TERRAIN_DRAINAGE_NONE,
  GPU_TERRAIN_DRAINAGE_UNRESOLVED
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import reference from '../../oracles/terrain-dinf-hand.json';
import {createTerrainReviewFixture} from '../terrain-processing/terrain-review-fixture';

type Input = {
  width: number;
  height: number;
  elevation: number[];
  fractions: number[];
  streams: number[];
  spacing?: number[];
  flowDirections?: number[];
  validity?: number[];
  scale?: number;
  offset?: number;
  noDataValue?: number;
  maximumIterations?: number;
  drainageOnly?: boolean;
  d8?: boolean;
};

async function runHAND(device: Device, input: Input, generateFractions = false) {
  const count = input.width * input.height;
  const grid = makeGPURasterMetadata({
    width: input.width,
    height: input.height,
    cellSize: [input.spacing?.[0] ?? 1, input.spacing?.[1] ?? 1]
  });
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createInputBuffer>[] = [];
  let identifier = 0;
  const source = <Format extends 'float32' | 'uint32'>(
    values: number[] | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${identifier++}`, buffer, format, values.length);
  };
  const elevation = source(input.elevation, 'float32');
  let fractions = source(input.fractions, 'float32');
  if (generateFractions) {
    fractions = createTransientView(graph, 'computed-fractions', 'float32', count * 8);
    graph.add(
      new GPUTerrainFlowFractions({
        grid,
        elevation,
        flowDirections: source(input.flowDirections!, 'uint32'),
        parameters: source(getGPUTerrainFlowParameterValues({}, grid), 'float32'),
        routing: 'd-infinity',
        output: {
          fractions,
          edgeLengths: createTransientView(graph, 'lengths', 'float32', count * 8)
        }
      })
    );
  }
  const outputBuffers = [count, count, 1, 1, 1].map(length => createOutputBuffer(device, length));
  const height = importGraphBuffer(graph, 'height', outputBuffers[0], 'float32', count);
  const drainage = importGraphBuffer(graph, 'drainage', outputBuffers[1], 'uint32', count);
  const status = ['converged', 'iterationLimitReached', 'iterationCount'].map((name, index) =>
    importGraphBuffer(graph, name, outputBuffers[index + 2], 'uint32', 1)
  );
  graph.add(
    new GPUTerrainHeightAboveDrainage({
      grid,
      elevation: makeGPURasterBand(elevation, grid, {
        validity: input.validity ? source(input.validity, 'uint32') : undefined,
        scale: input.scale,
        offset: input.offset,
        noDataValue: input.noDataValue
      }),
      flowFractions: input.d8 ? undefined : fractions,
      flowDirections: input.d8
        ? source(input.flowDirections ?? new Array(count).fill(0), 'uint32')
        : undefined,
      streams: source(input.streams, 'uint32'),
      maximumIterations: input.maximumIterations ?? 16,
      output: {
        heightAboveDrainage: input.drainageOnly ? undefined : height,
        drainageCells: drainage,
        status: {converged: status[0], iterationLimitReached: status[1], iterationCount: status[2]}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      hand: Array.from(await readFloat32(outputBuffers[0], count)),
      drainage: Array.from(await readUint32(outputBuffers[1], count)),
      status: await Promise.all(
        outputBuffers.slice(2).map(async buffer => (await readUint32(buffer, 1))[0])
      )
    };
  } finally {
    compiled.destroy();
    for (const buffer of [...buffers, ...outputBuffers]) buffer.destroy();
  }
}

it('matches the NumPy absorbing-flow oracle on small DEMs, including Tarboton routing composition', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of reference) {
    for (const generateFractions of [false, true]) {
      const result = await runHAND(device, fixture, generateFractions);
      fixture.hand.forEach((expected, index) =>
        expect(result.hand[index], `${fixture.name}:${index}`).toBeCloseTo(expected, 4)
      );
      expect(result.drainage).toEqual(fixture.drainage);
      expect(result.status.slice(0, 2)).toEqual([1, 0]);
      expect(result.status[2]).toBeLessThan(16);
    }
  }
});

it('averages drops across distinct first streams and keeps a unique label after a split merges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fractions = new Array(32).fill(0);
  // Adjacent E and SE branches, not a D8 steepest-receiver selection.
  fractions[0] = 0.25;
  fractions[1] = 0.75;
  let result = await runHAND(device, {
    width: 2,
    height: 2,
    elevation: [20, 10, 8, 2],
    fractions,
    streams: [0, 1, 0, 1]
  });
  expect(result.hand).toEqual([16, 0, NaN, 0]); // .25*(20-10) + .75*(20-2)
  expect(result.drainage).toEqual([GPU_TERRAIN_DRAINAGE_MULTIPLE, 1, GPU_TERRAIN_DRAINAGE_NONE, 3]);
  fractions[10] = 1; // cell 1 flows south to cell 3, the sole first stream.
  result = await runHAND(device, {
    width: 2,
    height: 2,
    elevation: [20, 10, 8, 2],
    fractions,
    streams: [0, 0, 0, 1]
  });
  expect(result.hand).toEqual([18, 8, NaN, 0]);
  expect(result.drainage).toEqual([3, 3, GPU_TERRAIN_DRAINAGE_NONE, 3]);
});

it('publishes completed cells at the budget and reports cycles without false convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fractions = new Array(32).fill(0);
  fractions[0] = fractions[8] = fractions[16] = 1;
  const chain = {width: 4, height: 1, elevation: [9, 6, 3, 0], fractions, streams: [0, 0, 0, 1]};
  const limited = await runHAND(device, {...chain, maximumIterations: 1});
  expect(limited.hand).toEqual([NaN, NaN, 3, 0]);
  expect(limited.drainage).toEqual([
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    3,
    3
  ]);
  expect(limited.status).toEqual([0, 1, 1]);
  const resolved = await runHAND(device, {...chain, maximumIterations: 8});
  expect(resolved.hand).toEqual([9, 6, 3, 0]);
  expect(resolved.status).toEqual([1, 0, 4]);
  fractions.fill(0);
  fractions[0] = fractions[12] = 1;
  const cycle = await runHAND(device, {...chain, fractions, maximumIterations: 3});
  expect(cycle.hand).toEqual([NaN, NaN, NaN, 0]);
  expect(cycle.drainage).toEqual([
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    GPU_TERRAIN_DRAINAGE_NONE,
    3
  ]);
  expect(cycle.status).toEqual([0, 1, 3]);
});

it('honors band validity, nodata, calibration, unresolved rows and stream precedence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fractions = new Array(32).fill(0);
  fractions[0] = 1;
  fractions[8] = 1;
  fractions[24] = NaN; // a known stream stops before reading an unknown fraction row.
  const chain = {width: 4, height: 1, elevation: [10, 7, 4, 1], fractions, streams: [0, 0, 1, 1]};
  const calibrated = await runHAND(device, {...chain, scale: 2, offset: 99});
  expect(calibrated.hand).toEqual([12, 6, 0, 0]);
  for (const metadata of [{validity: [1, 1, 0, 1]}, {noDataValue: 4}]) {
    const result = await runHAND(device, {...chain, ...metadata});
    expect(result.hand).toEqual([NaN, NaN, NaN, 0]);
    expect(result.drainage).toEqual([
      GPU_TERRAIN_DRAINAGE_NONE,
      GPU_TERRAIN_DRAINAGE_NONE,
      GPU_TERRAIN_DRAINAGE_NONE,
      3
    ]);
  }
  fractions[8] = NaN;
  const unknown = await runHAND(device, chain);
  expect(unknown.hand).toEqual([NaN, NaN, 0, 0]);
  expect(unknown.drainage).toEqual([
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    2,
    3
  ]);
  expect(unknown.status.slice(0, 2)).toEqual([0, 0]);
  fractions[8] = 1;
  const streamUnknown = await runHAND(device, {
    ...chain,
    streams: [0, GPU_TERRAIN_FLOW_STREAM_UNRESOLVED, 1, 1]
  });
  expect(streamUnknown.drainage).toEqual(unknown.drainage);
  expect(streamUnknown.status.slice(0, 2)).toEqual([0, 0]);
  const labels = await runHAND(device, {...chain, drainageOnly: true});
  expect(labels.drainage).toEqual([2, 2, 2, 3]);
});

it('rejects contaminated positive branches and ignores zero-weight neighbors on a padded workgroup', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fractions = new Array(48).fill(0);
  fractions[0] = 0.5;
  fractions[1] = 0.5; // 0 -> stream 1 and terminal 4: any missing branch contaminates 0.
  fractions[16] = -1; // malformed row
  fractions[24] = 1; // 3 -> terminal 4
  fractions[40] = 1; // cell 5 east leaves the grid
  const result = await runHAND(device, {
    width: 3,
    height: 2,
    elevation: [10, 5, 4, 8, 2, 1],
    fractions,
    streams: [0, 1, 0, 0, 0, 0]
  });
  expect(result.hand).toEqual([NaN, 0, NaN, NaN, NaN, NaN]);
  expect(result.drainage).toEqual([
    GPU_TERRAIN_DRAINAGE_NONE,
    1,
    GPU_TERRAIN_DRAINAGE_UNRESOLVED,
    GPU_TERRAIN_DRAINAGE_NONE,
    GPU_TERRAIN_DRAINAGE_NONE,
    GPU_TERRAIN_DRAINAGE_UNRESOLVED
  ]);
  expect(result.status.slice(0, 2)).toEqual([0, 0]);
});

it('W5-01 both HAND routes distinguish unknown membership, streams and known outlets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const d8 of [true, false]) {
    for (const [stream, drainage, height, converged] of [
      [0xffffffff, 0xfffffffe, NaN, 0],
      [1, 0, 0, 1],
      [0, 0xffffffff, NaN, 1]
    ]) {
      const result = await runHAND(device, {
        width: 1,
        height: 1,
        elevation: [5],
        fractions: new Array(8).fill(0),
        streams: [stream],
        d8
      });
      expect(result.hand).toEqual([height]);
      expect(result.drainage).toEqual([drainage]);
      expect(result.status.slice(0, 2)).toEqual([converged, 0]);
      expect(result.status[2]).toBeGreaterThanOrEqual(1);
      expect(result.status[2]).toBeLessThanOrEqual(16);
    }
    const result = await runHAND(device, {
      width: 2,
      height: 1,
      elevation: [5, 3],
      fractions: [1, ...new Array(15).fill(0)],
      streams: [0, 0xffffffff],
      flowDirections: [1, 0],
      d8
    });
    expect(result.hand).toEqual([NaN, NaN]);
    expect(result.drainage).toEqual([0xfffffffe, 0xfffffffe]);
    expect(result.status.slice(0, 2)).toEqual([0, 0]);
  }
});

it('B1/B4 inherited HAND diagnostics overwrite poison on unknown-to-known replay with drainage-only and paired outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const drainageOnly of [false, true]) {
    const fixture = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 3, height: 1});
    const fractions = new Array(24).fill(0);
    fractions[0] = fractions[8] = 1;
    const streams = fixture.input('uint32', [0, 0, 0xffffffff]);
    const height = fixture.output('float32', 3),
      drainage = fixture.output('uint32', 3);
    const converged = fixture.output('uint32', 1),
      limit = fixture.output('uint32', 1),
      rounds = fixture.output('uint32', 1);
    for (const destination of [converged, limit, rounds])
      destination.buffer.write(Uint32Array.of(77));
    fixture.graph.add(
      new GPUTerrainHeightAboveDrainage({
        grid,
        elevation: makeGPURasterBand(fixture.input('float32', [15, 11, 3]).view, grid),
        flowFractions: fixture.input('float32', fractions).view,
        streams: streams.view,
        maximumIterations: 16,
        output: {
          heightAboveDrainage: drainageOnly ? undefined : height.view,
          drainageCells: drainage.view,
          status: Object.create({
            converged: converged.view,
            iterationLimitReached: limit.view,
            iterationCount: rounds.view
          })
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (const unknown of [true, false]) {
        streams.buffer.write(Uint32Array.of(0, 0, unknown ? 0xffffffff : 1));
        fixture.submit(compiled);
        expect(await drainage.read()).toEqual(
          unknown ? [0xfffffffe, 0xfffffffe, 0xfffffffe] : [2, 2, 2]
        );
        if (!drainageOnly)
          expect(await height.read()).toEqual(unknown ? [NaN, NaN, NaN] : [12, 8, 0]);
        expect(await converged.read()).toEqual([unknown ? 0 : 1]);
        expect(await limit.read()).toEqual([0]);
        expect(await rounds.read()).toEqual([3]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});
