// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Texture} from '@luma.gl/core';
import type {GPURasterTextureBand} from '@luma.gl/graph/gpu-raster';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterFractionalZonalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalZonalDistribution} from '../../src/raster-analysis/gpu-raster-fractional-zonal-distribution';
import {GPURasterFractionalCategoricalStatistics} from '../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {
  GPURasterEVI,
  GPURasterSAVI,
  GPURasterNBR,
  GPURasterNDWI
} from '../../src/raster-analysis/spectral-indices';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';

for (const operation of [
  'scalar',
  'distribution',
  'categorical',
  'evi',
  'savi',
  'nbr',
  'ndwi'
] as const) {
  it(`RT-09 ${operation} texture calibration, native sentinel, mask and resubmission`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createParityGraph(device);
    let serial = 0;
    const floats = (length: number) => rig.output(`float-${serial++}`, length, 'float32');
    const integers = (length: number) => rig.output(`integer-${serial++}`, length, 'uint32');
    const grid = {
      width: 4,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const categorical = operation === 'categorical';
    const format = categorical ? ('rgba32uint' as const) : ('rgba32float' as const);
    const texture = device.createTexture({
      format,
      width: 4,
      height: 1,
      usage: Texture.SAMPLE | Texture.COPY_DST
    });
    const source = rig.graph.importTexture(
      {id: 'source', format, width: 4, height: 1, usage: texture.props.usage},
      texture
    );
    const mask = rig.input('mask', [1, 1, 0, 1], 'uint32');
    const input = {
      id: 'source',
      format: categorical ? ('uint32' as const) : ('float32' as const),
      storage: {
        kind: 'texture' as const,
        channel: 2 as const,
        view: rig.graph.createTextureView(source)
      },
      validity: mask.view,
      noDataValue: categorical ? 16777217 : -9,
      scale: categorical ? 1 : 2,
      offset: categorical ? 0 : 10
    } as GPURasterTextureBand;
    const coverage = {
      zoneIds: rig.input('zones', [0, 0, 0, 0], 'uint32').view,
      cellIds: rig.input('cells', [0, 1, 2, 3], 'uint32').view,
      fractions: rig.input('fractions', [1, 1, 1, 1], 'float32').view,
      zoneValidity: rig.input('zone-validity', [1], 'uint32').view,
      status: {
        count: rig.input('count', [4], 'uint32').view,
        requiredCount: rig.input('required', [4], 'uint32').view,
        overflow: rig.input('overflow', [0], 'uint32').view
      }
    };
    const result = floats(
      operation === 'categorical' || ['evi', 'savi', 'nbr', 'ndwi'].includes(operation) ? 4 : 1
    );
    const validity = integers(result.view.length);
    const total = floats(1);
    const area = floats(1);
    const constant = (id: string, value: number) => ({
      id,
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: rig.input(id, Array(4).fill(value), 'float32').view
      }
    });
    const contributor =
      operation === 'scalar'
        ? new GPURasterFractionalZonalStatistics({
            grid,
            input,
            coverage,
            zoneCount: 1,
            quantity: 'intensive',
            output: {
              means: result.view,
              totals: total.view,
              coveredArea: floats(1).view,
              validArea: area.view,
              validity: validity.view
            }
          })
        : operation === 'distribution'
          ? new GPURasterFractionalZonalDistribution({
              grid,
              input,
              coverage,
              zoneCount: 1,
              probabilities: rig.input('probabilities', [0.5], 'float32').view,
              output: {
                means: result.view,
                populationVariances: floats(1).view,
                sampleVariances: floats(1).view,
                effectiveSampleSizes: floats(1).view,
                weightSums: total.view,
                squaredWeightSums: floats(1).view,
                coveredArea: floats(1).view,
                validArea: area.view,
                quantiles: floats(1).view,
                quantileValidity: integers(1).view,
                validity: validity.view,
                status: integers(1).view
              }
            })
          : operation === 'categorical'
            ? new GPURasterFractionalCategoricalStatistics({
                grid,
                input: input as GPURasterTextureBand<'uint32'>,
                coverage,
                zoneCount: 1,
                categoryCapacity: 4,
                output: {
                  coverageSums: result.view,
                  classAreas: floats(4).view,
                  proportions: floats(4).view,
                  coveredArea: floats(1).view,
                  validArea: area.view,
                  missingArea: floats(1).view,
                  majority: integers(1).view,
                  minority: integers(1).view,
                  validity: validity.view
                }
              })
            : operation === 'evi'
              ? new GPURasterEVI({
                  ...grid,
                  nearInfrared: input,
                  red: constant('red', 4),
                  blue: constant('blue', 1),
                  output: result.view,
                  outputValidity: validity.view
                })
              : operation === 'savi'
                ? new GPURasterSAVI({
                    ...grid,
                    nearInfrared: input,
                    red: constant('red', 4),
                    output: result.view,
                    outputValidity: validity.view
                  })
                : operation === 'nbr'
                  ? new GPURasterNBR({
                      ...grid,
                      nearInfrared: input,
                      shortwaveInfrared2: constant('swir', 4),
                      output: result.view,
                      outputValidity: validity.view
                    })
                  : new GPURasterNDWI({
                      ...grid,
                      method: 'gao',
                      nearInfrared: input,
                      shortwaveInfrared: constant('swir', 4),
                      output: result.view,
                      outputValidity: validity.view
                    });
    rig.graph.add(contributor);
    const compiled = rig.graph.compile();
    try {
      for (const frame of [0, 1]) {
        const raw = categorical
          ? frame === 0
            ? [16777217, 1, 2, 3]
            : [0, 1, 16777217, 2]
          : frame === 0
            ? [1, -9, 3, 4]
            : [2, 3, 4, 5];
        const texels = raw.flatMap(value => [999, 888, value, 777]);
        texture.writeData(categorical ? Uint32Array.from(texels) : Float32Array.from(texels));
        mask.buffer.write(Uint32Array.from(frame === 0 ? [1, 1, 0, 1] : [1, 0, 1, 1]));
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(result.buffer, result.view.length);
        if (operation === 'scalar' || operation === 'distribution') {
          // Calibrated supports are [12,18] and [14,18,20], each of area 1.
          expect(Number.isFinite(actual[0])).toBe(true);
          expect(actual[0]).toBeCloseTo(frame === 0 ? 15 : 52 / 3, 5);
          expect(await readFloat32(area.buffer, 1)).toEqual([frame === 0 ? 2 : 3]);
          expect(await readFloat32(total.buffer, 1)).toEqual([
            operation === 'scalar' ? (frame === 0 ? 30 : 52) : frame === 0 ? 2 : 3
          ]);
          expect((await readUint32(validity.buffer, 1))[0]).toBe(1);
        } else if (categorical) {
          expect(actual).toEqual(frame === 0 ? [0, 1, 0, 1] : [1, 0, 1, 0]);
          expect((await readUint32(validity.buffer, 1))[0]).toBe(1);
          expect(await readFloat32(area.buffer, 1)).toEqual([2]);
        } else {
          for (const [index, value] of (frame === 0
            ? [12, NaN, NaN, 18]
            : [14, NaN, 18, 20]
          ).entries()) {
            if (Number.isNaN(value)) {
              expect(actual[index]).toBeNaN();
              continue;
            }
            const expected =
              operation === 'evi'
                ? (2.5 * (value - 4)) / (value + 17.5)
                : operation === 'savi'
                  ? (1.5 * (value - 4)) / (value + 4.5)
                  : (value - 4) / (value + 4);
            expect(Number.isFinite(actual[index])).toBe(true);
            expect(actual[index]).toBeCloseTo(expected, 6);
          }
          expect(await readUint32(validity.buffer, 4)).toEqual(
            frame === 0 ? [1, 0, 0, 1] : [1, 0, 1, 1]
          );
        }
      }
      if (categorical) {
        // 2^24 differs from raw nodata 2^24+1, despite rounding to the same float32.
        // It is an out-of-range ID and must invalidate this zone, not become missing.
        texture.writeData(
          Uint32Array.from([0, 0, 16777216, 0, 0, 0, 1, 0, 0, 0, 16777217, 0, 0, 0, 2, 0])
        );
        submitGraph(device, compiled, undefined);
        expect((await readUint32(validity.buffer, 1))[0]).toBe(0);
      }
    } finally {
      compiled.destroy();
      texture.destroy();
      rig.destroy();
    }
  });
}
