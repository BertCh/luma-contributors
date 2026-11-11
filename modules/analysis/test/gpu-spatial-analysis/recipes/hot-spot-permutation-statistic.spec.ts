// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {addHotSpotAnalysisRecipe} from '../../../src/gpu-spatial-analysis/recipes/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from './recipe-harness';

// The hot spot recipe maps a Gi or Gi* z-score and confirms it by permutation. Both stages must
// describe the same statistic: the same neighbourhood sum, with the focal row in it (Gi*) or out
// of it (Gi). Expected values are the Ord and Getis (1995) closed forms in float64 on a small
// lattice whose neighbours are enumerated here.

const WIDTH = 5;
const HEIGHT = 5;
const ROW_COUNT = WIDTH * HEIGHT;
/** Non-negative values with no row at the mean. */
const VALUES = Array.from({length: ROW_COUNT}, (_, cell) => ((cell * 7) % 11) + 1 + (cell % 3) * 4);
const TOTAL = VALUES.reduce((sum, value) => sum + value, 0);
const TOTAL_SQUARES = VALUES.reduce((sum, value) => sum + value * value, 0);

/** Queen neighbours of a lattice cell (row-major `y * WIDTH + x`), the cell itself excluded. */
function getQueenNeighbors(cell: number): number[] {
  const column = cell % WIDTH;
  const row = Math.floor(cell / WIDTH);
  const neighbors: number[] = [];
  for (let rowOffset = -1; rowOffset <= 1; rowOffset++) {
    for (let columnOffset = -1; columnOffset <= 1; columnOffset++) {
      const otherColumn = column + columnOffset;
      const otherRow = row + rowOffset;
      if (
        (rowOffset !== 0 || columnOffset !== 0) &&
        otherColumn >= 0 &&
        otherColumn < WIDTH &&
        otherRow >= 0 &&
        otherRow < HEIGHT
      ) {
        neighbors.push(otherRow * WIDTH + otherColumn);
      }
    }
  }
  return neighbors;
}

function getNeighborSum(cell: number): number {
  return getQueenNeighbors(cell).reduce((sum, other) => sum + VALUES[other], 0);
}

/** One statistic: its numerator, the observed value esda reports, and the z-score of a numerator. */
type Statistic = {
  getNumerator: (cell: number) => number;
  getObserved: (numerator: number, cell: number) => number;
  getZScore: (numerator: number, cell: number) => number;
};

/** Gi, focal row excluded: `G_i = sum_j x_j / (sum x - x_i)`, moments over the other rows. */
const GI: Statistic = {
  getNumerator: getNeighborSum,
  getObserved: (numerator, cell) => numerator / (TOTAL - VALUES[cell]),
  getZScore: (numerator, cell) => {
    const neighborCount = getQueenNeighbors(cell).length;
    const mean = (TOTAL - VALUES[cell]) / (ROW_COUNT - 1);
    const deviation = Math.sqrt((TOTAL_SQUARES - VALUES[cell] ** 2) / (ROW_COUNT - 1) - mean ** 2);
    return (
      (numerator - neighborCount * mean) /
      (deviation *
        Math.sqrt(((ROW_COUNT - 1) * neighborCount - neighborCount ** 2) / (ROW_COUNT - 2)))
    );
  }
};

/** Gi*, focal row included with weight 1: `G*_i = (x_i + sum_j x_j) / sum x`, global moments. */
const GI_STAR: Statistic = {
  getNumerator: cell => VALUES[cell] + getNeighborSum(cell),
  getObserved: numerator => numerator / TOTAL,
  getZScore: (numerator, cell) => {
    const weightSum = getQueenNeighbors(cell).length + 1;
    const mean = TOTAL / ROW_COUNT;
    const deviation = Math.sqrt(TOTAL_SQUARES / ROW_COUNT - mean ** 2);
    return (
      (numerator - mean * weightSum) /
      (deviation * Math.sqrt((ROW_COUNT * weightSum - weightSum ** 2) / (ROW_COUNT - 1)))
    );
  }
};

const CELLS = Array.from({length: ROW_COUNT}, (_, cell) => cell);

async function runRecipe(selfWeight: number | undefined) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `hot-spot-statistic-${selfWeight ?? 'default'}`);
  try {
    const zScores = fixture.output('z-scores', 'float32', ROW_COUNT);
    const observed = fixture.output('observed', 'float32', ROW_COUNT);
    addHotSpotAnalysisRecipe(fixture.graph, {
      parameters: fixture.parameterBuffer(
        'autocorrelation-parameters',
        'float32',
        GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
      ),
      selfWeight,
      minimumNeighbors: 1,
      permutation: {
        parameters: fixture.parameterBuffer(
          'permutation-parameters',
          'uint32',
          GPU_PERMUTATION_PARAMETER_LENGTH
        ),
        maximumPermutations: 19
      },
      source: {
        kind: 'lattice',
        values: fixture.input('values', Float32Array.from(VALUES), 'float32', ROW_COUNT),
        width: WIDTH,
        height: HEIGHT,
        neighborCapacity: ROW_COUNT * 8
      },
      outputs: {zScores: zScores.view, permutation: {observed: observed.view}}
    });
    fixture.run();
    return {
      zScores: await fixture.readFloat32(zScores, ROW_COUNT),
      observed: await fixture.readFloat32(observed, ROW_COUNT)
    };
  } finally {
    fixture.destroy();
  }
}

function expectClose(actual: number[], expected: number[], relative: number, label: string) {
  expected.forEach((value, cell) => {
    expect(Math.abs(actual[cell] - value), `${label} ${cell}`).toBeLessThanOrEqual(
      relative * Math.max(1, Math.abs(value))
    );
  });
}

it.each([
  {name: 'selfWeight 0 (Gi)', selfWeight: 0, statistic: GI},
  {name: 'the default selfWeight (Gi*)', selfWeight: undefined, statistic: GI_STAR}
])('the map and the permutation test compute the same statistic with $name', async ({
  selfWeight,
  statistic
}) => {
  const actual = await runRecipe(selfWeight);
  // Each stage against its closed form.
  expectClose(
    actual.observed,
    CELLS.map(cell => statistic.getObserved(statistic.getNumerator(cell), cell)),
    1e-5,
    'observed'
  );
  expectClose(
    actual.zScores,
    CELLS.map(cell => statistic.getZScore(statistic.getNumerator(cell), cell)),
    1e-4,
    'z-score'
  );
  // The two stages against each other: the numerator read back from the permutation test's
  // observed statistic, standardised, is the mapped z-score.
  const zScoresFromObserved = CELLS.map(cell => {
    const unitObserved = statistic.getObserved(1, cell);
    return statistic.getZScore(actual.observed[cell] / unitObserved, cell);
  });
  expectClose(actual.zScores, zScoresFromObserved, 1e-3, 'z-score from the observed statistic');
});

it('Gi and Gi* differ on this lattice, so a mismatched permutation statistic would fail', () => {
  const differences = CELLS.map(cell =>
    Math.abs(
      GI.getObserved(GI.getNumerator(cell), cell) -
        GI_STAR.getObserved(GI_STAR.getNumerator(cell), cell)
    )
  );
  // The cross-stage comparison above allows 1e-3 of the z-score; the two statistics are much
  // further apart than that in every row.
  expect(Math.min(...differences)).toBeGreaterThan(1e-3);
});
