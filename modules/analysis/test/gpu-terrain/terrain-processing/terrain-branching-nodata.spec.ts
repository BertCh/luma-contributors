// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review 2026-11-10, terrain-hydrology-01: a no-data cell donates nothing, so it
// must not block branching transport in the cells around it or downstream of them.

const NO_DATA = Number.NaN;

const grid = (width: number, height = 1) => ({
  width,
  height,
  affine: [1, 0, 0, 0, 1, 0] as const,
  pixelInterpretation: 'area' as const
});

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  const input = <Format extends 'float32' | 'uint32'>(values: number[], format: Format) => {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${identifier++}`, buffer, format, values.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length),
      async read() {
        return Array.from(
          await (format === 'float32' ? readFloat32(buffer, length) : readUint32(buffer, length))
        );
      }
    };
  };
  return {
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

/**
 * Runs `GPUTerrainFlow` (D-infinity), materializes its fractions, and transports unit runoff through
 * `GPUTerrainTiledFlow` with `routing: 'branching'` and through `GPUTerrainBranchingFlow` directly.
 */
async function runBranchingPipeline(width: number, height: number, elevation: number[]) {
  const {graph, input, output, run} = await setup();
  const count = width * height;
  const metadata = grid(width, height);
  const parameters = input(
    Array.from(getGPUTerrainFlowParameterValues({streamThreshold: 1}, metadata)),
    'float32'
  );
  const filled = output(count, 'float32'),
    directions = output(count, 'uint32'),
    flowAccumulation = output(count, 'float32'),
    streams = output(count, 'uint32'),
    fractions = output(count * 8, 'float32'),
    tiledAccumulation = output(count, 'float32'),
    tiledStatus = output(count, 'uint32'),
    directValues = output(count, 'float32'),
    directStatus = output(count, 'uint32'),
    directUnresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainFlow({
      grid: metadata,
      elevation: makeGPURasterBand(input(elevation, 'float32'), metadata),
      parameters,
      fillDepressions: true,
      flowRouting: 'd-infinity',
      output: {
        filledElevation: filled.view,
        flowDirections: directions.view,
        accumulation: flowAccumulation.view,
        streams: streams.view
      }
    })
  );
  graph.add(
    new GPUTerrainFlowFractions({
      grid: metadata,
      elevation: filled.view,
      flowDirections: directions.view,
      parameters,
      routing: 'd-infinity',
      output: {fractions: fractions.view, edgeLengths: output(count * 8, 'float32').view}
    })
  );
  graph.add(
    new GPUTerrainTiledFlow({
      grid: metadata,
      routing: 'branching',
      tiles: [
        {
          column: 0,
          row: 0,
          width,
          height,
          owned: [0, 0, width, height],
          flowDirections: directions.view,
          fractions: fractions.view,
          runoff: input(Array(count).fill(1), 'float32'),
          streams: streams.view,
          output: {
            accumulation: tiledAccumulation.view,
            status: tiledStatus.view,
            reachIds: output(count, 'uint32').view
          }
        }
      ]
    })
  );
  graph.add(
    new GPUTerrainBranchingFlow({
      grid: metadata,
      fractions: fractions.view,
      values: input(Array(count).fill(1), 'float32'),
      validity: input(
        elevation.map(value => (Number.isFinite(value) ? 1 : 0)),
        'uint32'
      ),
      mode: 'transport',
      output: {
        values: directValues.view,
        status: directStatus.view,
        unresolvedCount: directUnresolved.view
      }
    })
  );
  const destroy = run();
  try {
    return {
      fractions: await fractions.read(),
      flowAccumulation: await flowAccumulation.read(),
      tiledAccumulation: await tiledAccumulation.read(),
      tiledStatus: await tiledStatus.read(),
      directValues: await directValues.read(),
      directStatus: await directStatus.read(),
      directUnresolved: await directUnresolved.read()
    };
  } finally {
    destroy();
  }
}

it('a no-data cell beside a valid one does not block branching transport (1 x 3)', async () => {
  const result = await runBranchingPipeline(3, 1, [NO_DATA, 2, 1]);
  // Hand computed: cell 1 has no donor (its west neighbour has no data), cell 2 receives cell 1.
  const expected = [NO_DATA, 1, 2];
  expect(result.flowAccumulation).toEqual(expected);
  expect(result.tiledAccumulation).toEqual(expected);
  expect(result.tiledStatus).toEqual([0, 1, 1]);
  expect(result.directValues).toEqual(expected);
  expect(result.directStatus).toEqual([0, 1, 1]);
  // The no-data cell donates nothing: a zero row, not an unknown one.
  expect(result.fractions.slice(0, 8)).toEqual(Array(8).fill(0));
}, 120000);

it('an interior no-data hole diverts the flow around it and blocks nothing', async () => {
  // A plane falling one unit per column to the east, with one no-data cell in the middle.
  const width = 5,
    height = 5;
  const elevation = Array.from({length: width * height}, (_, cell) => 10 - (cell % width));
  elevation[2 * width + 2] = NO_DATA;
  const result = await runBranchingPipeline(width, height, elevation);
  // Hand computed with unit weights. Every cell sends all its flow east (the east facet has slope
  // 1 and a zero cross slope), except the cell west of the hole: its east neighbour has no data,
  // its first usable facet is the south-east diagonal, so it sends everything to row 3, column 2.
  // The cell east of the hole has no donor.
  const expected = [
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, NO_DATA, 1, 2],
    [1, 2, 5, 6, 7],
    [1, 2, 3, 4, 5]
  ].flat();
  expect(result.flowAccumulation).toEqual(expected);
  expect(result.tiledAccumulation).toEqual(expected);
  expect(result.directValues).toEqual(expected);
  const expectedStatus = expected.map(value => (Number.isNaN(value) ? 0 : 1));
  expect(result.tiledStatus).toEqual(expectedStatus);
  expect(result.directStatus).toEqual(expectedStatus);
  expect(result.directUnresolved).toEqual([1]);
}, 120000);
