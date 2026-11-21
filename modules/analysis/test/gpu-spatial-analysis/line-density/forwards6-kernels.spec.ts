// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {GPULineLengthPerPolygon} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-length-per-polygon';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createLineOracleHarness} from '../line-clip/line-output-oracle-harness';

it('B5 physical dyadic boundary ownership survives row and column permutations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const axis of ['x', 'y'] as const)
    for (const reverse of [false, true]) {
      const height = Math.fround(0.11);
      const bits = new Uint32Array(Float32Array.of(2 * height).buffer)[0];
      const neighbors = new Float32Array(Uint32Array.of(bits - 1, bits, bits + 1).buffer);
      for (let position = 0; position < 3; position++) {
        const harness = createLineOracleHarness(device);
        const {graph, view, read} = harness;
        const lengths = view('float32', 4);
        const overflow = view('uint32', 1);
        const coordinate = neighbors[position];
        graph.add(
          new GPULineDensity({
            positions: view(
              'float32x2',
              2,
              axis === 'y'
                ? [0.25, coordinate, 0.75, coordinate]
                : [coordinate, 0.25, coordinate, 0.75]
            ),
            lineOffsets: view('uint32', 2, [0, 2]),
            parameters: view(
              'float32',
              4,
              axis === 'y' ? [0, -height, 1, height] : [-height, 0, height, 1]
            ),
            columns: axis === 'x' ? 4 : 1,
            rows: axis === 'y' ? 4 : 1,
            recordCapacity: 4,
            outputRowOrder: reverse ? 'north-first' : 'south-first',
            outputColumnOrder: reverse ? 'east-first' : 'west-first',
            output: {lengths, status: {overflow}}
          })
        );
        const destroy = await harness.run();
        try {
          // (2h - (-h))/h = 3 exactly; the immediately preceding float belongs to cell 2.
          const physicalCell = position === 0 ? 2 : 3;
          const expected = [0, 0, 0, 0];
          expected[reverse ? 3 - physicalCell : physicalCell] = 0.5;
          expect(await read(lengths)).toEqual(expected);
          expect(await read(overflow)).toEqual([0]);
        } finally {
          destroy();
          harness.destroy();
        }
      }
    }
});

it('T1-F1 unified orientation retains tangent endpoints and genuine narrow intersections', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixtures = [
    {line: [-5, 1.2, 15, 1.2], polygon: [0.2, 1.2, -0.8, 2.2, 2.3, 3.3], expected: 0},
    {line: [-1, 0.5, 1, 0.5], polygon: [0, 0, 2 ** -52, 0, 2 ** -52, 1, 0, 1], expected: 2 ** -52}
  ];
  for (const fixture of fixtures)
    for (const reverse of [false, true]) {
      const harness = createLineOracleHarness(device);
      const {graph, view, read} = harness;
      const lengths = view('float32', 1);
      const overflow = view('uint32', 1);
      const candidateOverflow = view('uint32', 1);
      graph.add(
        new GPULineLengthPerPolygon({
          positions: view(
            'float32x2',
            2,
            reverse ? [...fixture.line.slice(2), ...fixture.line.slice(0, 2)] : fixture.line
          ),
          lineOffsets: view('uint32', 2, [0, 2]),
          polygons: {
            kind: 'polygons',
            positions: view('float32x2', fixture.polygon.length / 2, fixture.polygon),
            featureOffsets: view('uint32', 2, [0, 1]),
            polygonOffsets: view('uint32', 2, [0, 1]),
            ringOffsets: view('uint32', 2, [0, fixture.polygon.length / 2])
          },
          pairCapacity: 4,
          output: {clippedLengths: lengths, status: {overflow, candidateOverflow}}
        })
      );
      const destroy = await harness.run();
      try {
        expect(await read(lengths)).toEqual([fixture.expected]);
        expect(await read(overflow)).toEqual([0]);
        expect(await read(candidateOverflow)).toEqual([0]);
      } finally {
        destroy();
        harness.destroy();
      }
    }
});
