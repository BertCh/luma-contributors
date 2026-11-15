// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import reference from '../../oracles/terrain-tiled-branching.json';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// The oracle independently derives routing on a plane and solves the transport linear system
// with NumPy. These are actual D-infinity/Freeman/Quinn fractions, rather than a synthetic diamond.
for (const routing of ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
  for (const reverseTileOrder of [false, true]) {
    it(`${routing} matches NumPy and the whole domain across poisoned halos (reverse=${reverseTileOrder})`, async () => {
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
              await (format === 'float32'
                ? readFloat32(buffer, length)
                : readUint32(buffer, length))
            );
          }
        };
      };
      let compiled: ReturnType<typeof graph.compile> | undefined;
      try {
        const expected = reference.cases.find(scene => scene.routing === routing)!;
        const {width, height} = reference;
        const count = width * height;
        const grid = {
          width,
          height,
          affine: [1, 0, 0, 0, 1, 0] as const,
          pixelInterpretation: 'area' as const
        };
        const whole = output(count, 'float32');
        const wholeStatus = output(count, 'uint32');
        const unresolved = output(1, 'uint32');
        graph.add(
          new GPUTerrainBranchingFlow({
            id: 'whole',
            grid,
            mode: 'transport',
            fractions: input(expected.fractions, 'float32'),
            values: input(reference.runoff, 'float32'),
            output: {values: whole.view, status: wholeStatus.view, unresolvedCount: unresolved.view}
          })
        );
        // Also check that the independently prepared fixture represents the production routing.
        const materialized = output(count * 8, 'float32');
        graph.add(
          new GPUTerrainFlowFractions({
            id: 'routing',
            grid,
            routing,
            elevation: input(reference.elevation, 'float32'),
            flowDirections: input(
              reference.elevation.map((_, cell) =>
                cell % width < width - 1 ? 1 : cell < count - width ? 4 : 0
              ),
              'uint32'
            ),
            parameters: input([1, 1, 0, 1, 0, 0, reference.flowExponent, 0], 'float32'),
            output: {fractions: materialized.view, edgeLengths: output(count * 8, 'float32').view}
          })
        );
        // Unequal tile widths, negative halo origins and a four-tile diagonal corner.
        const rectangles = [
          [0, 0, 2, 2],
          [2, 0, 3, 2],
          [0, 2, 2, 2],
          [2, 2, 3, 2]
        ];
        const captures = rectangles.map(([column, row, ownedWidth, ownedHeight]) => {
          const tileWidth = ownedWidth + 2;
          const tileHeight = ownedHeight + 2;
          const tileCount = tileWidth * tileHeight;
          const fractions = Array<number>(tileCount * 8).fill(NaN);
          const runoff = Array<number>(tileCount).fill(1000000);
          const directions = Array<number>(tileCount).fill(0xffffffff);
          const streams = Array<number>(tileCount).fill(0xffffffff);
          for (let localRow = 1; localRow <= ownedHeight; localRow++) {
            for (let localColumn = 1; localColumn <= ownedWidth; localColumn++) {
              const local = localRow * tileWidth + localColumn;
              const cell = (row + localRow - 1) * width + column + localColumn - 1;
              fractions.splice(local * 8, 8, ...expected.fractions.slice(cell * 8, cell * 8 + 8));
              runoff[local] = reference.runoff[cell];
              directions[local] = 0;
              streams[local] = 2;
            }
          }
          const accumulation = output(tileCount, 'float32');
          const status = output(tileCount, 'uint32');
          const reachIds = output(tileCount, 'uint32');
          return {
            accumulation,
            status,
            reachIds,
            tile: {
              column: column - 1,
              row: row - 1,
              width: tileWidth,
              height: tileHeight,
              owned: [1, 1, ownedWidth, ownedHeight] as const,
              fractions: input(fractions, 'float32'),
              flowDirections: input(directions, 'uint32'),
              runoff: input(runoff, 'float32'),
              streams: input(streams, 'uint32'),
              output: {
                accumulation: accumulation.view,
                status: status.view,
                reachIds: reachIds.view
              }
            }
          };
        });
        const tiles = captures.map(capture => capture.tile);
        graph.add(
          new GPUTerrainTiledFlow({
            id: 'tiled',
            grid,
            routing: 'branching',
            tiles: reverseTileOrder ? tiles.reverse() : tiles
          })
        );
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        const wholeValues = await whole.read();
        expect(await wholeStatus.read()).toEqual(Array(count).fill(1));
        expect(await unresolved.read()).toEqual([0]);
        const generatedFractions = await materialized.read();
        expected.fractions.forEach((fraction, index) => {
          expect(generatedFractions[index]).toBeCloseTo(fraction, 5);
        });
        wholeValues.forEach((value, cell) => {
          // Float32 transport versus the float64 NumPy solve; zero injections are included.
          expect(Math.abs(value - expected.accumulation[cell])).toBeLessThanOrEqual(
            2e-5 * Math.max(1, expected.accumulation[cell])
          );
        });
        expect(wholeValues[count - 1]).toBeCloseTo(
          reference.runoff.reduce((total, value) => total + value, 0),
          4
        );
        for (const capture of captures) {
          const values = await capture.accumulation.read();
          const statuses = await capture.status.read();
          const reaches = await capture.reachIds.read();
          const {tile} = capture;
          for (let local = 0; local < values.length; local++) {
            const column = local % tile.width;
            const row = Math.floor(local / tile.width);
            if (column === 0 || row === 0 || column === tile.width - 1 || row === tile.height - 1) {
              expect(values[local]).toBeNaN();
              expect(statuses[local]).toBe(0);
              expect(reaches[local]).toBe(0xffffffff);
            } else {
              const cell = (tile.row + row) * width + tile.column + column;
              // Same packed fractions and fixed neighbor order: seam equality is exact.
              expect(values[local]).toBe(wholeValues[cell]);
              expect(statuses[local]).toBe(1);
              expect(reaches[local]).toBe(expected.reachIds[cell]);
            }
          }
        }
      } finally {
        compiled?.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    });
  }
}
