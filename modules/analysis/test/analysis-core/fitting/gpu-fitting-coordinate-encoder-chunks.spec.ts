// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUFittingCoordinateEncoder} from '../../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createFittingTestGraph} from './fitting-test-graph';

it('GPUFittingCoordinateEncoder matches packed and independent Fourier/harmonic values for three uneven chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const encoding of ['fourier', 'spherical-harmonics'] as const) {
    const coordinateRows =
      encoding === 'fourier'
        ? [
            [0, 0],
            [0.25, 0],
            [0, 0.25],
            [-0.25, 0.25],
            [0.5, 0],
            [0, 0],
            [NaN, 0],
            [0.25, -0.25],
            [0, 0.5]
          ]
        : [
            [0, 0],
            [90, 0],
            [0, 90],
            [-90, 0],
            [180, 0],
            [0, 0],
            [NaN, 0],
            [0, -90],
            [45, 45]
          ];
    const maskValues = [1, 1, 1, 1, 1, 0, 1, 1, 1];
    const expectedValidity = [1, 1, 1, 1, 1, 0, 0, 1, 1];
    const expected = coordinateRows.flatMap(([first, second], row) => {
      if (!expectedValidity[row]) return [0, 0, 0, 0];
      if (encoding === 'fourier')
        return [
          Math.cos(2 * Math.PI * first),
          Math.sin(2 * Math.PI * first),
          Math.cos(2 * Math.PI * second),
          Math.sin(2 * Math.PI * second)
        ];
      const longitude = (first * Math.PI) / 180;
      const latitude = (second * Math.PI) / 180;
      const scale = Math.sqrt(3 / (4 * Math.PI));
      return [
        1 / Math.sqrt(4 * Math.PI),
        -scale * Math.cos(latitude) * Math.sin(longitude),
        scale * Math.sin(latitude),
        -scale * Math.cos(latitude) * Math.cos(longitude)
      ];
    });
    const results: number[][] = [];
    for (const lengths of [[9], [1, 3, 5]]) {
      const test = createFittingTestGraph(device);
      const coordinates = test.floats('coordinates', coordinateRows.flat(), 1);
      const features = test.floats('features', Array<number>(36).fill(-99), 3);
      const validity = test.uints('validity', Array<number>(9).fill(99), 2);
      const mask = test.uints('mask', maskValues, 1);
      const split = <Format extends 'float32' | 'uint32'>(
        view: GraphDataView<Format>,
        width: number
      ) => {
        if (lengths.length === 1) return view;
        let rowBase = 0;
        return createVectorView(
          view.buffer.id,
          view.format,
          lengths.map(rows => {
            const chunk = test.graph.createDataView(view.buffer, {
              format: view.format,
              byteOffset: view.byteOffset + rowBase * width * 4,
              length: rows * width
            });
            rowBase += rows;
            return chunk;
          })
        );
      };
      const common = {
        rowCount: 9,
        coordinates: split(coordinates, 2),
        mask: split(mask, 1),
        output: {features: split(features, 4), validity: split(validity, 1)}
      };
      test.graph.add(
        new GPUFittingCoordinateEncoder(
          encoding === 'fourier'
            ? {
                ...common,
                encoding,
                dimensions: 2,
                frequencyCount: 2,
                frequencies: test.floats('frequencies', [1, 0, 0, 1], 1)
              }
            : {
                ...common,
                encoding,
                maximumDegree: 1,
                spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
              }
        )
      );
      try {
        test.run();
        const actual = await test.readFloats(features);
        expect(await test.readUints(validity)).toEqual(expectedValidity);
        actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 5));
        results.push(actual);
        const storage = await test.readFloatStorage(features);
        expect(storage.slice(0, 3)).toEqual([12345, 12345, 12345]);
        expect(storage.slice(-3)).toEqual([12345, 12345, 12345]);
      } finally {
        test.destroy();
      }
    }
    expect(results[1]).toEqual(results[0]);
  }
});
