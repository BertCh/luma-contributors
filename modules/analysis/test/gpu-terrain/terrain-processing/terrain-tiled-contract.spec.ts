// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {computeTerrainBranchingFlowOnCPU} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const routing of ['d8', 'branching'] as const) {
  it(`publishes unequal ${routing} tiles with negative halo origins and unaligned shared output ranges`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: ReturnType<typeof createOutputBuffer>[] = [];
    const width = 9;
    const height = 3;
    const tileWidths = [2, 3, 4];
    const tileLengths = tileWidths.map(tileWidth => (tileWidth + 2) * (height + 2));
    const totalLength = tileLengths.reduce((total, length) => total + length, 0) + 2;
    let identifier = 0;
    const input = <Format extends 'float32' | 'uint32'>(
      format: Format,
      values: number[],
      prefix = 1
    ) => {
      const data =
        format === 'float32'
          ? Float32Array.from([...Array(prefix).fill(12345), ...values])
          : Uint32Array.from([...Array(prefix).fill(12345), ...values]);
      // This fixture also uses the helper for caller output views that are read back.
      const buffer = createOutputBuffer(device, data.length);
      buffer.write(data);
      buffers.push(buffer);
      const backing = importGraphBuffer(
        graph,
        `input-${identifier++}`,
        buffer,
        format,
        data.length
      );
      return {
        buffer,
        view: graph.createDataView(backing.buffer, {
          format,
          byteOffset: prefix * 4,
          length: values.length
        })
      };
    };
    const accumulation = input('float32', Array(totalLength).fill(12345), 0);
    const status = input('uint32', Array(totalLength).fill(12345), 0);
    const reachIds = input('uint32', Array(totalLength).fill(12345), 0);
    const invalidCount = input('uint32', [12345], 0);
    const converged = input('uint32', [12345], 0);
    const iterationLimitReached = input('uint32', [12345], 0);
    let globalColumn = 0;
    let outputBase = 1;
    const captures = tileWidths.map((ownedWidth, tileIndex) => {
      const tileWidth = ownedWidth + 2;
      const tileHeight = height + 2;
      const count = tileWidth * tileHeight;
      const directions = Array(count).fill(0xffffffff);
      const fractions = Array(count * 8).fill(NaN);
      const runoff = Array(count).fill(1e6);
      const streams = Array(count).fill(0xffffffff);
      for (let row = 1; row <= height; row++) {
        for (let column = 1; column <= ownedWidth; column++) {
          const local = row * tileWidth + column;
          const isOutlet = globalColumn + column === width;
          directions[local] = isOutlet ? 0 : 1;
          fractions.fill(0, local * 8, local * 8 + 8);
          fractions[local * 8] = isOutlet ? 0 : 1;
          runoff[local] = 1;
          streams[local] = 2;
        }
      }
      const runoffInput = input('float32', runoff, tileIndex + 1);
      const capture = {
        outputBase,
        globalColumn,
        runoffInput,
        runoff,
        tile: {
          column: globalColumn - 1,
          row: -1,
          width: tileWidth,
          height: tileHeight,
          owned: [1, 1, ownedWidth, height] as const,
          flowDirections: input('uint32', directions, tileIndex + 1).view,
          ...(routing === 'branching'
            ? {fractions: input('float32', fractions, tileIndex + 1).view}
            : {}),
          runoff: runoffInput.view,
          streams: input('uint32', streams, tileIndex + 1).view,
          output: {
            accumulation: graph.createDataView(accumulation.view.buffer, {
              format: 'float32',
              length: count,
              byteOffset: outputBase * 4
            }),
            status: graph.createDataView(status.view.buffer, {
              format: 'uint32',
              length: count,
              byteOffset: outputBase * 4
            }),
            reachIds: graph.createDataView(reachIds.view.buffer, {
              format: 'uint32',
              length: count,
              byteOffset: outputBase * 4
            })
          }
        }
      };
      globalColumn += ownedWidth;
      outputBase += count;
      return capture;
    });
    let compiled: ReturnType<typeof graph.compile> | undefined;
    try {
      graph.add(
        new GPUTerrainTiledFlow({
          grid: {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
          routing,
          tiles: captures.map(capture => capture.tile).reverse(),
          maximumIterations: 16,
          output: {
            status: {
              invalidCount: invalidCount.view,
              converged: converged.view,
              iterationLimitReached: iterationLimitReached.view
            }
          }
        })
      );
      compiled = graph.compile();
      for (const injection of [1, 2]) {
        const globalFractions = new Float32Array(width * height * 8);
        for (let cell = 0; cell < width * height; cell++) {
          globalFractions[cell * 8] = cell % width < width - 1 ? 1 : 0;
        }
        const reference = computeTerrainBranchingFlowOnCPU(
          {fractions: globalFractions, values: new Float32Array(width * height).fill(injection)},
          {
            grid: {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
            mode: 'transport'
          }
        );
        for (const capture of captures) {
          const data = Float32Array.from([
            ...Array(captures.indexOf(capture) + 1).fill(12345),
            ...capture.runoff.map(value => (value === 1 ? injection : value))
          ]);
          capture.runoffInput.buffer.write(data);
        }
        submitGraph(device, compiled, undefined);
        const values = await readFloat32(accumulation.buffer, totalLength);
        const states = await readUint32(status.buffer, totalLength);
        const reaches = await readUint32(reachIds.buffer, totalLength);
        expect([
          values[0],
          values[totalLength - 1],
          states[0],
          states[totalLength - 1],
          reaches[0],
          reaches[totalLength - 1]
        ]).toEqual(Array(6).fill(12345));
        for (const capture of captures) {
          const {tile} = capture;
          for (let local = 0; local < tile.width * tile.height; local++) {
            const column = local % tile.width;
            const row = Math.floor(local / tile.width);
            const destination = capture.outputBase + local;
            if (column === 0 || column === tile.width - 1 || row === 0 || row === tile.height - 1) {
              expect(values[destination]).toBeNaN();
              expect(states[destination]).toBe(0);
              expect(reaches[destination]).toBe(0xffffffff);
            } else {
              // Independent closed form: each row is an eastward chain of unit injections.
              expect(values[destination]).toBe((capture.globalColumn + column) * injection);
              const cell = (row - 1) * width + capture.globalColumn + column - 1;
              expect(values[destination]).toBe(reference.values[cell]);
              expect(states[destination]).toBe(1);
              expect(states[destination]).toBe(reference.status[cell]);
              expect(reaches[destination]).toBe((row - 1) * width);
            }
          }
        }
        expect(Array.from(await readUint32(invalidCount.buffer, 1))).toEqual([0]);
        expect(Array.from(await readUint32(converged.buffer, 1))).toEqual([1]);
        expect(Array.from(await readUint32(iterationLimitReached.buffer, 1))).toEqual([0]);
      }
    } finally {
      compiled?.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  });
}

it('burns without compiling empty pit-path kernels at the largest accepted step count', async () => {
  const {GPUTerrainBreach} = await import(
    '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach'
  );
  const {createTransientView} = await import('@luma.gl/graph/gpu-core');
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const sourceBuffer = createInputBuffer(device, Float32Array.of(5, NaN, -3));
  const burnBuffer = createInputBuffer(device, Float32Array.of(1, 0, 2));
  const elevationBuffer = createOutputBuffer(device, 3);
  const depthBuffer = createOutputBuffer(device, 3);
  const maskBuffer = createOutputBuffer(device, 3);
  const unresolvedBuffer = createOutputBuffer(device, 1);
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    graph.add(
      new GPUTerrainBreach({
        grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        elevation: importGraphBuffer(graph, 'source', sourceBuffer, 'float32', 3),
        burnDepth: importGraphBuffer(graph, 'burn', burnBuffer, 'float32', 3),
        pitCells: createTransientView(graph, 'pits', 'uint32', 0),
        maximumSteps: 0xffffffff,
        maximumCutDepth: 1,
        elevationDecrement: 0.25,
        output: {
          elevation: importGraphBuffer(graph, 'edited', elevationBuffer, 'float32', 3),
          cutDepth: importGraphBuffer(graph, 'depth', depthBuffer, 'float32', 3),
          modificationMask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', 3),
          pathCells: createTransientView(graph, 'paths', 'uint32', 0),
          pathLengths: createTransientView(graph, 'lengths', 'uint32', 0),
          pitStatus: createTransientView(graph, 'status', 'uint32', 0),
          unresolvedCount: importGraphBuffer(graph, 'unresolved', unresolvedBuffer, 'uint32', 1)
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    expect(Array.from(await readFloat32(elevationBuffer, 3))).toEqual([4, NaN, -5]);
    expect(Array.from(await readFloat32(depthBuffer, 3))).toEqual([1, NaN, 2]);
    expect(Array.from(await readUint32(maskBuffer, 3))).toEqual([2, 0, 2]);
    expect(Array.from(await readUint32(unresolvedBuffer, 1))).toEqual([0]);
  } finally {
    compiled?.destroy();
    for (const buffer of [
      sourceBuffer,
      burnBuffer,
      elevationBuffer,
      depthBuffer,
      maskBuffer,
      unresolvedBuffer
    ])
      buffer.destroy();
  }
});
