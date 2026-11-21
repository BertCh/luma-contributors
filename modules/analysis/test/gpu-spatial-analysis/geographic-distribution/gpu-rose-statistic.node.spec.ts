// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURoseStatistic} from '../../../src/gpu-spatial-analysis/geographic-distribution/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getWorkgroupMemoryBytes,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES
} from '../../utils/gpu-contributor-test-utils';

it('snapshots Rose circular conventions before caller mutation can change output shapes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const circular = {
      mode: 'directional' as const,
      angleUnit: 'radians' as const,
      origin: 'positive-x-counter-clockwise' as const,
      binCount: 4
    };
    const contributor = new GPURoseStatistic({
      starts: createTransientView(graph, 'starts', 'float32x2', 8),
      ends: createTransientView(graph, 'ends', 'float32x2', 8),
      circular,
      output: {
        summary: createTransientView(graph, 'summary', 'float32', 5),
        bins: createTransientView(graph, 'bins', 'float32', 4)
      }
    });
    circular.binCount = 8;
    expect(contributor.props.circular.binCount).toBe(4);
  } finally {
    device.destroy();
  }
});

for (const mode of ['directional', 'axial'] as const) {
  it(`Rose ${mode} computes standard count-based Rayleigh z, independently of descriptive weights`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const nodes = new GPURoseStatistic({
      starts: createTransientView(graph, 'starts', 'float32x2', 4),
      ends: createTransientView(graph, 'ends', 'float32x2', 4),
      weights: createTransientView(graph, 'weights', 'float32', 4),
      circular: {mode, angleUnit: 'radians', origin: 'positive-x-counter-clockwise', binCount: 4},
      output: {
        summary: createTransientView(graph, 'summary', 'float32', 5),
        bins: createTransientView(graph, 'bins', 'float32', 4)
      }
    }).getCommandNodes(graph);
    const [{source, constants}] = getGPUCommandNodeKernelSources(nodes);
    const resultantExpression = source
      .match(/let resultant = ([\s\S]*?);/)![1]
      .replace('vec2<f32>', 'vector');
    const evaluateResultant = new Function(
      'meanCosine',
      'meanSine',
      'length',
      'vector',
      'clamp',
      `return ${resultantExpression};`
    );
    const rounded = Math.fround;
    const norm = rounded(Math.sqrt(24 * 24 + 29 * 29));
    const length = ([first, second]: number[]) =>
      rounded(Math.sqrt(rounded(rounded(first * first) + rounded(second * second))));
    const resultant = evaluateResultant(
      rounded(24 / norm),
      rounded(29 / norm),
      length,
      (...values: number[]) => values,
      (value: number, lower: number, upper: number) => Math.max(lower, Math.min(value, upper))
    );
    expect(resultant).toBeLessThanOrEqual(1);
    expect(resultant).toBeGreaterThanOrEqual(0);
    device.destroy();
    expect(getWorkgroupMemoryBytes(source, constants)).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
    expect(source).toContain('directionCosine += cosineAngle;');
    expect(source).toContain('directionSine += sineAngle;');
    expect(source).toContain(
      'sharedDirectionCosine[lane] += sharedDirectionCosine[lane + stride];'
    );
    expect(source).toContain('sharedDirectionSine[lane] += sharedDirectionSine[lane + stride];');
    expect(source).toContain('let validCount = f32(sharedCount[0]);');
    expect(source).toContain(
      mode === 'axial'
        ? 'let cosineAngle = (unitDirection.x-unitDirection.y)*(unitDirection.x+unitDirection.y);'
        : 'let cosineAngle = unitDirection.x;'
    );
    // Evaluate the generated final expression in binary64, independently of the weighted
    // summary. This is a CPU test oracle only, not a production execution path.
    const expression = source.match(/let rayleighZ = ([\s\S]*?);/)![1];
    const evaluate = new Function(
      'sharedDirectionCosine',
      'sharedDirectionSine',
      'validCount',
      `return ${expression};`
    ) as (cosine: number[], sine: number[], count: number) => number;
    const angles = [0, 0, Math.PI / 2, Math.PI];
    const multiplier = mode === 'axial' ? 2 : 1;
    const cosine = angles.reduce((sum, angle) => sum + Math.cos(multiplier * angle), 0);
    const sine = angles.reduce((sum, angle) => sum + Math.sin(multiplier * angle), 0);
    const expected = mode === 'axial' ? 1 : 0.5;
    expect(evaluate([cosine], [sine], angles.length)).toBeCloseTo(expected, 14);
    // Contrast the old unit-dependent value on unequal descriptive weights.
    for (const unitScale of [1, 1000]) {
      const weights = [1, 2, 4, 8].map(weight => weight * unitScale);
      const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
      const weightedCosine = angles.reduce(
        (sum, angle, index) => sum + weights[index] * Math.cos(multiplier * angle),
        0
      );
      const weightedSine = angles.reduce(
        (sum, angle, index) => sum + weights[index] * Math.sin(multiplier * angle),
        0
      );
      expect((weightedCosine ** 2 + weightedSine ** 2) / totalWeight).not.toBeCloseTo(expected, 5);
      expect(evaluate([cosine], [sine], angles.length)).toBeCloseTo(expected, 14);
    }
    expect(evaluate([4], [0], 4)).toBe(4);
    expect(evaluate([0], [0], 4)).toBe(0);
  });
}
