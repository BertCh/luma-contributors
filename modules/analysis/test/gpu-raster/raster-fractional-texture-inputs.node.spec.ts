// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import type {GPURasterTextureBand} from '@luma.gl/graph/gpu-raster';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
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
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';

for (const operation of [
  'scalar',
  'distribution',
  'categorical',
  'evi',
  'savi',
  'nbr',
  'ndwi'
] as const) {
  it(`RT-09 ${operation} accepts a selected texture channel through public conversion`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floats = (length: number) =>
      createTransientView(graph, `floats-${serial++}`, 'float32', length);
    const integers = (length: number) =>
      createTransientView(graph, `integers-${serial++}`, 'uint32', length);
    const categorical = operation === 'categorical';
    const texture = graph.createTransientTexture({
      id: 'texture',
      width: 3,
      height: 1,
      format: categorical ? 'rgba32uint' : 'rgba32float',
      usage: Texture.SAMPLE
    });
    const input = {
      id: 'input',
      format: categorical ? ('uint32' as const) : ('float32' as const),
      storage: {
        kind: 'texture' as const,
        view: graph.createTextureView(texture),
        channel: 2 as const
      },
      validity: integers(3),
      noDataValue: categorical ? 16777217 : -9,
      scale: categorical ? 1 : 2,
      offset: categorical ? 0 : 10
    } as GPURasterTextureBand;
    const grid = {
      width: 3,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const coverage = {
      zoneIds: integers(3),
      cellIds: integers(3),
      fractions: floats(3),
      zoneValidity: integers(1),
      status: {
        count: integers(1),
        requiredCount: integers(1),
        overflow: integers(1),
        candidateOverflow: integers(1)
      }
    };
    const output = {output: floats(3), outputValidity: integers(3)};
    // Cast only the category discriminant; each branch has the exact public descriptor format.
    const contributor =
      operation === 'scalar'
        ? new GPURasterFractionalZonalStatistics({
            grid,
            input,
            coverage,
            zoneCount: 1,
            quantity: 'intensive',
            output: {
              totals: floats(1),
              means: floats(1),
              coveredArea: floats(1),
              validArea: floats(1),
              validity: integers(1)
            }
          })
        : operation === 'distribution'
          ? new GPURasterFractionalZonalDistribution({
              grid,
              input,
              coverage,
              zoneCount: 1,
              probabilities: floats(1),
              output: {
                means: floats(1),
                populationVariances: floats(1),
                sampleVariances: floats(1),
                effectiveSampleSizes: floats(1),
                weightSums: floats(1),
                squaredWeightSums: floats(1),
                coveredArea: floats(1),
                validArea: floats(1),
                quantiles: floats(1),
                quantileValidity: integers(1),
                validity: integers(1),
                status: integers(1)
              }
            })
          : operation === 'categorical'
            ? new GPURasterFractionalCategoricalStatistics({
                grid,
                input: input as GPURasterTextureBand<'uint32'>,
                coverage,
                zoneCount: 1,
                categoryCapacity: 3,
                output: {
                  coverageSums: floats(3),
                  classAreas: floats(3),
                  proportions: floats(3),
                  coveredArea: floats(1),
                  validArea: floats(1),
                  missingArea: floats(1),
                  majority: integers(1),
                  minority: integers(1),
                  validity: integers(1)
                }
              })
            : operation === 'evi'
              ? new GPURasterEVI({...grid, ...output, nearInfrared: input, red: input, blue: input})
              : operation === 'savi'
                ? new GPURasterSAVI({...grid, ...output, nearInfrared: input, red: input})
                : operation === 'nbr'
                  ? new GPURasterNBR({
                      ...grid,
                      ...output,
                      nearInfrared: input,
                      shortwaveInfrared2: input
                    })
                  : new GPURasterNDWI({
                      ...grid,
                      ...output,
                      method: 'gao',
                      nearInfrared: input,
                      shortwaveInfrared: input
                    });
    try {
      const sources = [...getParsedKernelSources(contributor.getCommandNodes(graph)).values()];
      const gather = sources.find(source => source.includes('textureLoad('));
      expect(gather).toContain('textureLoad(sourceTexture, vec2<i32>(globalId.xy), 0).z');
      if (categorical) {
        expect(gather).toContain('16777217u');
        expect(gather).toContain('outputValues[OUTPUT_OFFSET + pixelIndex] = rawSample');
      } else expect(gather).toContain('f32(rawSample) * 2.0 + 10.0');
    } finally {
      device.destroy();
    }
  });
}
