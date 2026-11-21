// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPUEmergingHotSpots} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/gpu-emerging-hot-spots';
import {GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spot-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getInterpretedEmergingCompensationWGSL} from './audit-emerging-hot-spots-wgsl';
import {
  ACCUMULATION_FIXTURE_CASES,
  createAccumulationFixture,
  createDenseCoefficientFixture,
  createMaskedCoefficientFixture,
  createTemporalScaleFixture,
  expectFocalCoefficientReadbacks,
  TEMPORAL_SCALE_FIXTURE_SCORE,
  type CoefficientFixture
} from './emerging-coefficient-review-fixture';

function probeCoefficientRow(fixture: CoefficientFixture): number {
  const sliceCount = fixture.sliceCount ?? 1;
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      values: createTransientView(graph, 'order-values', 'float32', 8 * sliceCount),
      parameters: createTransientView(graph, 'order-parameters', 'float32', 8),
      sliceCount,
      output: {
        giZScores: createTransientView(graph, 'order-scores', 'float32', 8 * sliceCount),
        trendZ: createTransientView(graph, 'order-trend-z', 'float32', 8),
        trendP: createTransientView(graph, 'order-trend-p', 'float32', 8),
        trendS: createTransientView(graph, 'order-trend-s', 'sint32', 8),
        category: createTransientView(graph, 'order-category', 'uint32', 8),
        hotSliceCount: createTransientView(graph, 'order-hot', 'uint32', 8),
        coldSliceCount: createTransientView(graph, 'order-cold', 'uint32', 8)
      },
      weights: {
        offsets: createTransientView(graph, 'order-offsets', 'uint32', 9),
        neighbors: createTransientView(graph, 'order-neighbors', 'uint32', 56),
        weights: createTransientView(graph, 'order-weights', 'float32', 56)
      }
    };
    const source = getGPUCommandNodeKernelSources(
      new GPUEmergingHotSpots(props).getCommandNodes(graph)
    ).find(kernel => kernel.id.includes('gi-star'))!.source;
    const start = source.indexOf('var weighted = ');
    const end = source.indexOf('\n  }\n  // Every invocation', start);
    const instant = source.slice(start, end);
    const moment = source.match(/fn addEmergingWeightMoment[\s\S]*?\n}/)![0];
    const scale = source.match(/fn getEmergingCoefficientScale[\s\S]*?\n}/)![0];
    const spreadStart = source.indexOf('let outsideCount = count - min(count, supportCount);');
    const spread = source.slice(spreadStart, source.indexOf('if (count >=', spreadStart));
    const sumStart = source.indexOf('let residualTotal = residualMean[', spreadStart);
    const centeredSum = source.slice(sumStart, source.indexOf('zScore = ', sumStart));
    const windowStart = source.indexOf('var weightedSum = ');
    const window = source.slice(windowStart, spreadStart);
    const neighbors = Uint32Array.from(fixture.neighbors);
    expect(
      Array.from(neighbors).every((row, index) => index === 0 || row > neighbors[index - 1]),
      'S2 ascending public CSR rows'
    ).toBe(true);
    const weights = Float32Array.from(fixture.weights);
    const program = `${getInterpretedEmergingCompensationWGSL(source)}\n${moment}\n${scale}
const SELF_WEIGHT: f32 = ${fixture.selfWeight};
const CELL_COUNT: u32 = ${fixture.values.length / sliceCount}u; const SLICE_COUNT: u32 = ${sliceCount}u;
@group(0) @binding(0) var<storage, read> centeredValues: array<f32>;
@group(0) @binding(1) var<storage, read> neighbors: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
const offsets = array<u32, 2>(0u,${neighbors.length}u);
const offsetsOffset = 0u; const neighborsOffset = 0u; const weightsOffset = 0u;
const residualMeanOffset = 0u; const residualMean = array<f32, 1>(0.0);
var<private> instantWeighted: array<vec2<f32>, SLICE_COUNT>;
var<private> instantWeight: array<f32, SLICE_COUNT>;
var<private> instantSquare: array<f32, SLICE_COUNT>;
var<private> instantCount: array<u32, SLICE_COUNT>;
var<private> instantUnweighted: array<vec2<f32>, SLICE_COUNT>;
var<private> instantOrigin: array<f32, SLICE_COUNT>;
var<private> instantScale: array<f32, SLICE_COUNT>;
fn readDeviation(bin: u32) -> f32 {return centeredValues[bin];}
fn isBinValid(bin: u32, cell: u32) -> bool {return (bitcast<u32>(readDeviation(bin)) & 0x7f800000u) != 0x7f800000u;}
fn multiplyBySpatialCount(value: f32, count: u32) -> f32 {return value * f32(count);}
fn divideBySpatialCount(value: f32, count: u32) -> f32 {return value / f32(count);}
fn computeInstant(slice: u32) {
  let cell = 0u; let local = slice; let chunkRowStart = 0u;
  var coefficientOrigin = 0.0;
  ${instant}
}
@compute @workgroup_size(1) fn main() {
  for (var current = 0u; current < SLICE_COUNT; current++) {computeInstant(current);}
  let cellInGroup = 0u; let firstSlice = 0u; let slice = SLICE_COUNT - 1u; let local = slice;
  let coefficientOrigin = instantOrigin[local];
  let count = ${fixture.count}u;
  ${window}
  ${spread}
  ${centeredSum}
  output[0] = getCompensated(centeredSum) / (${fixture.deviation} * sqrt(spread));
}`;
    const centered = Float32Array.from(fixture.values, (value, row) =>
      fixture.mask[row] ? value - fixture.mean : NaN
    );
    const result = new Float32Array(1);
    new WgslExec(new WgslParser().parse(program)).dispatchWorkgroups('main', 1, {
      0: {0: centered, 1: neighbors, 2: weights, 3: result}
    });
    expect(Number.isFinite(result[0])).toBe(true);
    return result[0];
  } finally {
    device.destroy();
  }
}

it('W5-02/S2 retains coefficient deviations under valid ascending-CSR relabelings', () => {
  for (const sign of [1, -1])
    for (const variant of ['self', 'first', 'relabeled'] as const) {
      // Population mean=.8, deviation=.4; one exceptional zero gives z=-.8/.4 * sign.
      expect(
        probeCoefficientRow(createDenseCoefficientFixture(variant, sign)),
        `${sign} ${variant}`
      ).toBeCloseTo(-2 * sign, 4);
    }
});

for (const fixture of ACCUMULATION_FIXTURE_CASES) {
  it(fixture.name, () => {
    const score = probeCoefficientRow(createAccumulationFixture(fixture));
    const criticalZ = GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES[fixture.confidenceLevel];
    expect.soft(score).toBeCloseTo(fixture.scores[fixture.sliceCount - 1], 4);
    expect(score >= criticalZ).toBe(fixture.hotSliceCount === 1);
    expect(score <= -criticalZ).toBe(false);
  });
}

it('B3 masked coefficients cannot choose the origin or reverse a cold bin', () => {
  for (const relabeled of [false, true]) {
    // 16*(-7)+112*1=0, variance=(16*49+112)/128=7. One excess coefficient at -7
    // gives -7/sqrt(7)=-sqrt(7), regardless of the 129 masked coefficient values.
    expect
      .soft(probeCoefficientRow(createMaskedCoefficientFixture(relabeled)), String(relabeled))
      .toBeCloseTo(-2.6457513110645907, 4);
  }
});

it('S3 focal readback checks reject truncated output arrays', () => {
  const full = {
    scores: new Array<number>(8000).fill(-2),
    probabilities: new Array<number>(8000).fill(0.0455002638963584),
    coldCounts: new Array<number>(8000).fill(1)
  };
  for (const port of ['scores', 'probabilities', 'coldCounts'] as const) {
    expect(
      () => expectFocalCoefficientReadbacks({...full, [port]: [full[port][0]]}, 8000),
      port
    ).toThrow();
  }
  expectFocalCoefficientReadbacks(full, 8000);
});

it('B3 temporal merging rebases different usable coefficient origins', () => {
  // Finite values are 1..6, mean=3.5, variance=35/12. The .5 coefficients on values
  // 3 and 4 have opposite centered values, hence full-window weighted sum and z are zero.
  const result = probeCoefficientRow({
    values: Float32Array.of(1, 2, 3, NaN, 4, NaN, NaN, 5, NaN, 6),
    mask: Uint32Array.of(1, 1, 1, 0, 1, 0, 0, 1, 0, 1),
    neighbors: [1, 2, 3, 4],
    weights: [0.5, 0.5, 1, 1],
    selfWeight: 1,
    count: 6,
    mean: 3.5,
    deviation: 1.707825127659933,
    sliceCount: 2
  });
  expect(result).toBeCloseTo(0, 6);
});

it('merges temporal instants in one coefficient scale when a large-weight bin is missing', () => {
  for (const scale of [1, 2 ** -66, 2 ** 126]) {
    const result = probeCoefficientRow(createTemporalScaleFixture(scale));
    expect(result).toBeCloseTo(TEMPORAL_SCALE_FIXTURE_SCORE, 6);
  }
});
