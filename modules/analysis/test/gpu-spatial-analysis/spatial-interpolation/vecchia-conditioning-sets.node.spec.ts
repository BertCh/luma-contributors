// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  buildVecchiaConditioningSets,
  buildVecchiaPredictionNeighborhoods,
  uploadVecchiaNeighborhoods,
  getVecchiaMaxMinOrdering
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-conditioning-sets';
import {
  getGPUVecchiaLogLikelihoodParameterValues,
  getVecchiaNuggetFloor,
  GPU_VECCHIA_PARAMETER_LENGTH,
  VECCHIA_NUGGET_FLOOR_FRACTION
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';
import {createExperimentRandom} from './vecchia-likelihood-reference';

describe('Vecchia ingress regressions', () => {
  it('rejects malformed preflight before linking rows (ING-14)', () => {
    const positions = [0, 0, 2, 0, 0.1, 0];
    expect(() =>
      buildVecchiaConditioningSets({positions, neighborCount: 1, times: [0, 1000, 1000]})
    ).toThrow(/timeScale/);
    expect(() => buildVecchiaConditioningSets({positions, neighborCount: 33})).toThrow(
      /neighborCount/
    );
    expect(() => buildVecchiaConditioningSets({positions: [0, 0, 1], neighborCount: 1})).toThrow(
      /positions/
    );
    expect(() =>
      buildVecchiaConditioningSets({positions, neighborCount: 1, validity: [1]})
    ).toThrow(/validity/);
    expect(() =>
      buildVecchiaConditioningSets({positions, neighborCount: 1, order: [1, 4, 0]})
    ).toThrow(/order/);
    expect(() =>
      buildVecchiaConditioningSets({positions, neighborCount: 1, order: [0, 0, 1]})
    ).toThrow(/order/);
    expect(() =>
      buildVecchiaConditioningSets({positions, neighborCount: 1, times: [0], timeScale: 1})
    ).toThrow(/times/);
    expect(() =>
      buildVecchiaConditioningSets({positions, neighborCount: 1, times: [0, 1, 2], timeScale: NaN})
    ).toThrow(/timeScale/);
    expect(() =>
      buildVecchiaConditioningSets({
        positions: {length: 2 ** 32},
        rowCount: 2 ** 31 - 2,
        neighborCount: 32
      })
    ).toThrow(/rowCount \* neighborCount/);
  });

  it('preserves requested width and maps uint32 consumer ports (ING-18)', () => {
    for (const result of [
      buildVecchiaConditioningSets({positions: [2, 3], neighborCount: 7}),
      buildVecchiaPredictionNeighborhoods({
        positions: [2, 3],
        queries: [0, 0, 1, 1],
        neighborCount: 7
      })
    ]) {
      expect(result.counts).toEqual({
        rowCount: result.offsets.length - 1,
        neighborCount: 7,
        entryCount: result.neighbors.length
      });
      expect(result.formats).toEqual({offsets: 'uint32', neighbors: 'uint32'});
      const uploaded = uploadVecchiaNeighborhoods(result, (values, format) => ({values, format}));
      expect(uploaded.neighborCount).toBe(7);
      expect(uploaded.neighborhoodOffsets).toEqual({values: result.offsets, format: 'uint32'});
      expect(uploaded.neighborhoods).toEqual({values: result.neighbors, format: 'uint32'});
    }
  });

  it('matches an independent scan on every exterior edge and corner (ING-01)', () => {
    const positions = [0, 0, 2, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1, 2, 2, 1];
    const queries = [-10, 1, 10, 1, 1, -10, 1, 10, -10, -10, -10, 10, 10, -10, 10, 10];
    const result = buildVecchiaPredictionNeighborhoods({positions, queries, neighborCount: 4});
    const expected = Array.from({length: queries.length / 2}, (_, query) =>
      Array.from({length: positions.length / 2}, (_, row) => row)
        .sort((first, second) => {
          const distance = (row: number) =>
            (positions[row * 2] - queries[query * 2]) ** 2 +
            (positions[row * 2 + 1] - queries[query * 2 + 1]) ** 2;
          return distance(first) - distance(second) || first - second;
        })
        .slice(0, 4)
    );
    expect(getSets(result.offsets, result.neighbors)).toEqual(expected);
  });

  it('ignores masked NaNs while retaining source row IDs (ING-02)', () => {
    const props = {positions: [0, 0, NaN, 0, 1, 1], validity: [1, 0, 1], neighborCount: 1};
    const result = buildVecchiaConditioningSets(props);
    expect(Array.from(result.offsets)).toEqual([0, 0, 0, 1]);
    expect(Array.from(result.neighbors)).toEqual([0]);
    expect(
      Array.from(buildVecchiaPredictionNeighborhoods({...props, queries: [1, 1]}).neighbors)
    ).toEqual([2]);
  });

  it('handles empty, singleton, all-masked and coincident inputs (ING-13)', () => {
    for (const positions of [[], [2, 3], [2, 3, 2, 3]]) {
      const result = buildVecchiaConditioningSets({positions, neighborCount: 1});
      expect(Array.from(result.offsets)).toEqual(
        positions.length === 4 ? [0, 0, 1] : positions.length === 2 ? [0, 0] : [0]
      );
      expect(Array.from(getVecchiaMaxMinOrdering(positions))).toEqual(
        positions.length === 4 ? [0, 1] : positions.length === 2 ? [0] : []
      );
      expect(
        Array.from(
          buildVecchiaPredictionNeighborhoods({positions, queries: [2, 3], neighborCount: 1})
            .neighbors
        )
      ).toEqual(positions.length ? [0] : []);
    }
    expect(
      Array.from(
        buildVecchiaConditioningSets({positions: [NaN, NaN], validity: [0], neighborCount: 1})
          .offsets
      )
    ).toEqual([0, 0]);
  });
});

function createPositions(rowCount: number, seed: number): Float64Array {
  const random = createExperimentRandom(seed);
  return Float64Array.from({length: rowCount * 2}, () => (random() - 0.5) * 1000);
}

/** Written independently of the code under test: keeps the `neighborCount` nearest by scan. */
function findBruteForceNeighbors(
  positions: ArrayLike<number>,
  rowCount: number,
  neighborCount: number,
  order: readonly number[]
): number[][] {
  return order.map((row, position) => {
    const candidates = order.slice(0, position).map((sample, samplePosition) => {
      const offsetX = positions[row * 2] - positions[sample * 2];
      const offsetY = positions[row * 2 + 1] - positions[sample * 2 + 1];
      return {sample, samplePosition, distance: offsetX * offsetX + offsetY * offsetY};
    });
    candidates.sort(
      (first, second) =>
        first.distance - second.distance || first.samplePosition - second.samplePosition
    );
    return candidates.slice(0, neighborCount).map(candidate => candidate.sample);
  });
}

function getSets(offsets: Uint32Array, neighbors: Uint32Array): number[][] {
  return Array.from({length: offsets.length - 1}, (_, row) =>
    Array.from(neighbors.subarray(offsets[row], offsets[row + 1]))
  );
}

describe('buildVecchiaConditioningSets', () => {
  it('matches a brute-force scan over the row order', () => {
    const rowCount = 2500;
    const neighborCount = 30;
    const positions = createPositions(rowCount, 7);
    const {offsets, neighbors} = buildVecchiaConditioningSets({positions, neighborCount});
    expect(getSets(offsets, neighbors)).toEqual(
      findBruteForceNeighbors(
        positions,
        rowCount,
        neighborCount,
        Array.from({length: rowCount}, (_, row) => row)
      )
    );
  });

  it('respects a caller order and resolves equal distances to the earlier order position', () => {
    // A 4 x 4 lattice has many equidistant neighbours.
    const positions: number[] = [];
    for (let row = 0; row < 16; row++) {
      positions.push(row % 4, Math.floor(row / 4));
    }
    const order = [...Array(16).keys()].reverse();
    const {offsets, neighbors} = buildVecchiaConditioningSets({
      positions,
      neighborCount: 5,
      order
    });
    expect(getSets(offsets, neighbors)).toEqual(
      findBruteForceNeighbors(positions, 16, 5, order).reduce<number[][]>(
        (byRow, set, position) => {
          byRow[order[position]] = set;
          return byRow;
        },
        Array.from({length: 16}, () => [])
      )
    );
  });

  it('leaves invalid rows out of every conditioning set', () => {
    const positions = createPositions(300, 11);
    const validity = Uint8Array.from({length: 300}, (_, row) => (row % 7 === 3 ? 0 : 1));
    const {offsets, neighbors} = buildVecchiaConditioningSets({
      positions,
      neighborCount: 10,
      validity
    });
    const sets = getSets(offsets, neighbors);
    sets.forEach((set, row) => {
      if (!validity[row]) {
        expect(set).toEqual([]);
      }
      expect(set.every(sample => validity[sample])).toBe(true);
    });
  });
});

describe('getVecchiaMaxMinOrdering', () => {
  /** O(n^2) exact max-min with the same rules, written independently. */
  function bruteForceMaxMin(positions: ArrayLike<number>, rowCount: number): number[] {
    let centreX = 0;
    let centreY = 0;
    for (let row = 0; row < rowCount; row++) {
      centreX += positions[row * 2] / rowCount;
      centreY += positions[row * 2 + 1] / rowCount;
    }
    let first = 0;
    let bestDistance = Infinity;
    for (let row = 0; row < rowCount; row++) {
      const distance =
        (positions[row * 2] - centreX) ** 2 + (positions[row * 2 + 1] - centreY) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        first = row;
      }
    }
    const order = [first];
    const nearest = new Float64Array(rowCount).fill(Infinity);
    for (let row = 0; row < rowCount; row++) {
      nearest[row] =
        (positions[row * 2] - positions[first * 2]) ** 2 +
        (positions[row * 2 + 1] - positions[first * 2 + 1]) ** 2;
    }
    const chosen = new Set(order);
    while (order.length < rowCount) {
      let bestRow = -1;
      let best = -1;
      for (let row = 0; row < rowCount; row++) {
        if (!chosen.has(row) && nearest[row] > best) {
          best = nearest[row];
          bestRow = row;
        }
      }
      chosen.add(bestRow);
      order.push(bestRow);
      for (let row = 0; row < rowCount; row++) {
        const distance =
          (positions[row * 2] - positions[bestRow * 2]) ** 2 +
          (positions[row * 2 + 1] - positions[bestRow * 2 + 1]) ** 2;
        nearest[row] = Math.min(nearest[row], distance);
      }
    }
    return order;
  }

  it('matches the exact greedy order on scattered points', () => {
    const positions = createPositions(1200, 23);
    expect(Array.from(getVecchiaMaxMinOrdering(positions))).toEqual(
      bruteForceMaxMin(positions, 1200)
    );
  });

  it('matches the exact greedy order on a lattice with many ties', () => {
    const positions: number[] = [];
    for (let row = 0; row < 121; row++) {
      positions.push(row % 11, Math.floor(row / 11));
    }
    expect(Array.from(getVecchiaMaxMinOrdering(positions))).toEqual(
      bruteForceMaxMin(positions, 121)
    );
  });
});

describe('getGPUVecchiaLogLikelihoodParameterValues', () => {
  it('packs the local kriging layout', () => {
    const values = getGPUVecchiaLogLikelihoodParameterValues({
      model: 'matern-3/2',
      range: 12.5,
      sill: 2,
      nugget: 0.002
    });
    expect(values).toHaveLength(GPU_VECCHIA_PARAMETER_LENGTH);
    expect(values[0]).toBeCloseTo(2.002, 6); // covariance scale
    expect(values[4]).toBe(3); // Matern 3/2 model code
    expect(values[5]).toBeCloseTo(0.002, 9);
    expect(values[6]).toBe(2);
    expect(values[7]).toBe(12.5);
  });

  it('rejects a Matern 3/2 nugget below the floor but accepts it exactly at the floor', () => {
    expect(() =>
      getGPUVecchiaLogLikelihoodParameterValues({
        model: 'matern-3/2',
        range: 1,
        sill: 7,
        nugget: 7 * 0.999e-3
      })
    ).toThrow(/at least 0.001 of the sill/);
    expect(VECCHIA_NUGGET_FLOOR_FRACTION).toBe(1e-3);
    expect(getVecchiaNuggetFloor({model: 'matern-3/2', sill: 7})).toBeCloseTo(7e-3, 12);
    expect(() =>
      getGPUVecchiaLogLikelihoodParameterValues({
        model: 'matern-3/2',
        range: 1,
        sill: 7,
        nugget: getVecchiaNuggetFloor({model: 'matern-3/2', sill: 7})
      })
    ).not.toThrow();
  });

  it('accepts an exponential nugget of zero and rejects kernels not offered', () => {
    expect(getVecchiaNuggetFloor({model: 'exponential', sill: 3})).toBe(0);
    expect(() =>
      getGPUVecchiaLogLikelihoodParameterValues({
        model: 'exponential',
        range: 1,
        sill: 1,
        nugget: 0
      })
    ).not.toThrow();
    expect(() =>
      getGPUVecchiaLogLikelihoodParameterValues({
        model: 'gaussian' as never,
        range: 1,
        sill: 1,
        nugget: 0.01
      })
    ).toThrow(/exponential or matern-3\/2/);
  });
});
