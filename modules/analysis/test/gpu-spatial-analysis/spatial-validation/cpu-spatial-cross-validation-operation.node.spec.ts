// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialCrossValidationOnCPU,
  CPUSpatialCrossValidationOperation,
  CPU_SPATIAL_CROSS_VALIDATION_MAXIMUM_CANDIDATE_COUNT
} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {
  GPUSpatialCrossValidationOperation,
  type SpatialCrossValidationOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation-operation';
import {
  constructSamplingContributor,
  createSamplingCPUInputs
} from '../point-process/validation-sampling-operation-test-utils';

it('the CPU callback unpacks candidate and prediction ports and publishes design diagnostics', async () => {
  const fixture = fixtures.knndm;
  const operation = new CPUSpatialCrossValidationOperation({
    ...PROPS,
    mask: 'mask',
    design: {
      mode: 'knndm',
      candidateCount: fixture.candidateCount,
      predictionPositions: 'predictionPositions',
      predictionRowCount: fixture.predictionPositions.length / 2,
      candidates: 'candidates',
      diagnostics: 'diagnostics'
    }
  });
  const result = await operation.execute({
    inputs: createSamplingCPUInputs({
      positions: {format: 'float32x2', values: new Float32Array(fixture.positions)},
      blocks: {format: 'uint32', values: new Uint32Array(8)},
      mask: {format: 'uint32', values: new Uint32Array(8).fill(1)},
      predictionPositions: {
        format: 'float32x2',
        values: new Float32Array(fixture.predictionPositions)
      },
      candidates: {format: 'uint32', values: new Uint32Array(fixture.candidates)}
    })
  });
  expect(result).toEqual({
    folds: new Uint32Array(fixture.expectedFolds),
    training: new Uint32Array(fixture.expectedTraining),
    testing: new Uint32Array(fixture.expectedTesting),
    counts: new Uint32Array(fixture.expectedCounts),
    status: new Uint32Array(fixture.expectedStatus),
    diagnostics: new Float32Array([
      0,
      fixture.candidateCount,
      fixture.objectives[fixture.selected]!,
      fixture.selected
    ])
  });
});
import fixtures from './spatial-cross-validation-operation-fixtures.json';

const PROPS: SpatialCrossValidationOperationProps = {
  id: 'cross-validation-test',
  rowCount: 8,
  positions: 'positions',
  blocks: 'blocks',
  foldCount: 2,
  seed: 1,
  bufferDistance: 0.6,
  output: {
    folds: 'folds',
    training: 'training',
    testing: 'testing',
    counts: 'counts',
    status: 'status'
  }
};

function expectFixture(
  result: ReturnType<typeof computeSpatialCrossValidationOnCPU>,
  fixture: typeof fixtures.knndm | typeof fixtures.environmental | typeof fixtures.generated
): void {
  expect(Array.from(result.folds)).toEqual(fixture.expectedFolds);
  expect(Array.from(result.training)).toEqual(fixture.expectedTraining);
  expect(Array.from(result.testing)).toEqual(fixture.expectedTesting);
  expect(Array.from(result.counts)).toEqual(fixture.expectedCounts);
  expect(Array.from(result.status)).toEqual(fixture.expectedStatus);
}

it('matches SciPy kmeans2 including sample standardization and a constant feature', () => {
  const fixture = fixtures.environmental;
  const input = {
    positions: new Float32Array(fixture.positions),
    blocks: new Uint32Array(6),
    features: new Float32Array(fixture.features)
  };
  const result = computeSpatialCrossValidationOnCPU(input, {
    ...PROPS,
    design: {mode: 'environmental', featureCount: 3}
  });
  expectFixture(result, fixture);
  expect(Array.from(result.diagnostics!.subarray(0, 2))).toEqual([0, 2]);
  expect(result.diagnostics![2]).toBeCloseTo(fixture.objective, 6);
  expect(Number.isNaN(result.diagnostics![3])).toBe(true);
  // A GPU budget of one step would stop; the CPU computes the converged partition.
  expect(
    computeSpatialCrossValidationOnCPU(input, {
      ...PROPS,
      design: {mode: 'environmental', featureCount: 3, maximumIterations: 1}
    })
  ).toEqual(result);
});

it('selects the minimum SciPy Wasserstein candidate and rejects an unbalanced candidate', () => {
  const fixture = fixtures.knndm;
  const result = computeSpatialCrossValidationOnCPU(
    {
      positions: new Float32Array(fixture.positions),
      blocks: new Uint32Array(8),
      predictionPositions: new Float32Array(fixture.predictionPositions),
      candidates: new Uint32Array(fixture.candidates)
    },
    {...PROPS, design: {mode: 'knndm', candidateCount: fixture.candidateCount}}
  );
  expectFixture(result, fixture);
  expect(Array.from(result.diagnostics!.subarray(0, 2))).toEqual([0, fixture.candidateCount]);
  expect(result.diagnostics![3]).toBe(fixture.selected);
  expect(result.diagnostics![2]).toBeCloseTo(fixture.objectives[fixture.selected]!, 6);
  expect(fixture.selected).not.toBe(0);
});

it('matches a hand-walked NNDM removal sequence and its inclusive surviving-neighbor radii', () => {
  const result = computeSpatialCrossValidationOnCPU(
    {
      positions: new Float32Array([0, 0, 1, 0, 3, 0, 7, 0]),
      blocks: new Uint32Array(4),
      predictionPositions: new Float32Array([20, 0, 25, 0])
    },
    {...PROPS, foldCount: 4, design: {mode: 'nndm', maximumDistance: 4}}
  );
  // Removals, by (test row, removed neighbor): (0,1),(1,0),(1,2),(2,1),
  // (0,2),(2,0),(2,3),(3,2). The next nearest radius is 6, above maximumDistance.
  expect(Array.from(result.folds)).toEqual([0, 1, 2, 3]);
  expect(result.diagnostics![1]).toBe(8);
  expect(Array.from(result.diagnostics!.subarray(4))).toEqual([7, 6, Math.fround(3.402823e38), 6]);
  expect(Array.from(result.training)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);
  expect(Array.from(result.testing)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  expect(Array.from(result.counts)).toEqual([0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 1, 2]);
  expect(Array.from(result.status)).toEqual([1, 1, 1, 0]);
});

it('keeps the last selectable candidate index exact and rejects counts beyond the float32 layout limit', () => {
  const candidateCount = CPU_SPATIAL_CROSS_VALIDATION_MAXIMUM_CANDIDATE_COUNT;
  const input = {
    positions: new Float32Array([0, 0, 1, 0]),
    blocks: new Uint32Array(2),
    predictionPositions: new Float32Array([0, 0]),
    candidates: new Uint32Array(candidateCount * 2).fill(0xffffffff)
  };
  input.candidates.set([0, 1], (candidateCount - 1) * 2);
  const result = computeSpatialCrossValidationOnCPU(input, {
    ...PROPS,
    bufferDistance: 0,
    design: {mode: 'knndm', candidateCount}
  });
  // Only the final partition is valid. Both training nearest distances are 1; prediction
  // distance is 0, so the area between their CDFs is exactly 1.
  expect(Array.from(result.folds)).toEqual([0, 1]);
  expect(Array.from(result.training)).toEqual([0, 1, 1, 0]);
  expect(Array.from(result.testing)).toEqual([1, 0, 0, 1]);
  expect(Array.from(result.counts)).toEqual([1, 1, 0, 1, 1, 0]);
  expect(Array.from(result.status)).toEqual([0, 0]);
  expect(Array.from(result.diagnostics!)).toEqual([0, candidateCount, 1, candidateCount - 1]);
  expect(result.diagnostics![3]).toBeLessThan(candidateCount);
  for (const rejectedCount of [candidateCount + 1, 16777220]) {
    expect(() =>
      computeSpatialCrossValidationOnCPU(input, {
        ...PROPS,
        design: {mode: 'knndm', candidateCount: rejectedCount}
      })
    ).toThrow(/cross-validation-test.*candidateCount/);
    expect(
      () =>
        new CPUSpatialCrossValidationOperation({
          ...PROPS,
          rowCount: 2,
          design: {
            mode: 'knndm',
            candidateCount: rejectedCount,
            candidates: 'candidates',
            predictionPositions: 'predictionPositions',
            predictionRowCount: 1
          }
        })
    ).toThrow(/cross-validation-test.*candidateCount/);
  }
  expect(() =>
    constructSamplingContributor(
      new GPUSpatialCrossValidationOperation({
        ...PROPS,
        rowCount: 2,
        design: {
          mode: 'knndm',
          candidateCount: 16777220,
          candidates: 'candidates',
          predictionPositions: 'predictionPositions',
          predictionRowCount: 1
        }
      }),
      {
        positions: {format: 'float32x2', length: 2},
        blocks: {format: 'uint32', length: 2},
        predictionPositions: {format: 'float32x2', length: 1},
        candidates: {format: 'uint32', length: 33554440}
      }
    )
  ).toThrow(/cross-validation-test.*candidateCount/);
}, 30000);

it('returns fresh zero NNDM radii after a successful frame is completely masked', () => {
  const input = {
    positions: new Float32Array([0, 0, 1, 0, 3, 0, 7, 0]),
    blocks: new Uint32Array(4),
    predictionPositions: new Float32Array([20, 0, 25, 0]),
    mask: new Uint32Array(4).fill(1)
  };
  const options = {...PROPS, foldCount: 4, design: {mode: 'nndm' as const, maximumDistance: 4}};
  const first = computeSpatialCrossValidationOnCPU(input, options);
  expect(Array.from(first.diagnostics!.subarray(4))).toEqual([7, 6, Math.fround(3.402823e38), 6]);
  input.mask.fill(0);
  const failed = computeSpatialCrossValidationOnCPU(input, options);
  expect(Array.from(failed.diagnostics!)).toEqual([1, 0, NaN, NaN, 0, 0, 0, 0]);
  expect(Array.from(failed.folds)).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
  expect(Array.from(failed.training)).toEqual(new Array(16).fill(0));
  expect(Array.from(failed.testing)).toEqual(new Array(16).fill(0));
  expect(Array.from(failed.counts)).toEqual(new Array(12).fill(0));
  expect(Array.from(failed.status)).toEqual([3, 3, 3, 3]);
  // The pure function owns its result; failure never mutates the preceding result.
  expect(Array.from(first.diagnostics!.subarray(4))).toEqual([7, 6, Math.fround(3.402823e38), 6]);
});

it('matches numeric group ranks, excludes duplicate locations even at zero buffer, and handles empty folds', () => {
  const result = computeSpatialCrossValidationOnCPU(
    {
      positions: new Float32Array([0, 0, 0, 0, 5, 0, 6, 0, 7, 0]),
      blocks: new Uint32Array([30, 10, 30, 50, 70]),
      mask: new Uint32Array([1, 1, 1, 0, 1])
    },
    {...PROPS, foldCount: 4, bufferDistance: 0, design: {mode: 'groups'}}
  );
  expect(Array.from(result.folds)).toEqual([1, 0, 1, 0xffffffff, 2]);
  expect(Array.from(result.training)).toEqual([
    0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1
  ]);
  expect(Array.from(result.counts)).toEqual([2, 1, 1, 1, 2, 1, 3, 1, 0, 4, 0, 0]);
  expect(Array.from(result.status)).toEqual([0, 0, 0, 2]);
  const empty = computeSpatialCrossValidationOnCPU(
    {positions: new Float32Array(4), blocks: new Uint32Array([1, 2]), mask: new Uint32Array(2)},
    PROPS
  );
  expect(Array.from(empty.folds)).toEqual([0xffffffff, 0xffffffff]);
  expect(Array.from(empty.status)).toEqual([3, 3]);
});

it('keeps seeded block folds invariant under row permutation and partial block masking', () => {
  const positions = new Float32Array([0, 0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0]);
  const blocks = new Uint32Array([100, 7, 100, 999, 7, 42]);
  const input = {positions, blocks};
  const result = computeSpatialCrossValidationOnCPU(input, {...PROPS, foldCount: 3});
  expect(result.folds[0]).toBe(result.folds[2]);
  expect(result.folds[1]).toBe(result.folds[4]);
  expect(new Set(result.folds).size).toBe(3);
  const permutation = [5, 4, 3, 2, 1, 0];
  const permuted = computeSpatialCrossValidationOnCPU(
    {
      positions: Float32Array.from(
        permutation.flatMap(row => [positions[row * 2], positions[row * 2 + 1]])
      ),
      blocks: Uint32Array.from(permutation, row => blocks[row])
    },
    {...PROPS, foldCount: 3}
  );
  expect(Array.from(permuted.folds)).toEqual(permutation.map(row => result.folds[row]));
  const masked = computeSpatialCrossValidationOnCPU(
    {...input, mask: new Uint32Array([0, 1, 1, 1, 1, 1])},
    {...PROPS, foldCount: 3}
  );
  expect(masked.folds.subarray(1)).toEqual(result.folds.subarray(1));
  expect(
    new Set(
      Array.from({length: 8}, (_, seed) =>
        Array.from(
          computeSpatialCrossValidationOnCPU(input, {...PROPS, foldCount: 3, seed}).folds
        ).join(',')
      )
    ).size
  ).toBeGreaterThan(1);
});

it('generates kNNDM candidates and preserves missing diagnostics on failed designs', () => {
  const input = {
    positions: new Float32Array([0, 0, 1, 1, 2, 0, 8, 0, 9, 1, 10, 0]),
    blocks: new Uint32Array(6),
    predictionPositions: new Float32Array([0, 2, 4, 2, 8, 2])
  };
  const generated = computeSpatialCrossValidationOnCPU(input, {
    ...PROPS,
    bufferDistance: 0,
    design: {mode: 'knndm', candidateCount: 3}
  });
  // Every restart splits these separated triples. Label reversal leaves distances unchanged;
  // all candidates tie, so candidate 0 wins and matches the known geometric partition.
  expect(Array.from(generated.folds)).toEqual([0, 0, 0, 1, 1, 1]);
  expect(generated.diagnostics![3]).toBe(0);
  const failed = computeSpatialCrossValidationOnCPU(
    {...input, mask: new Uint32Array(6)},
    {...PROPS, design: {mode: 'nndm', maximumDistance: 1}, foldCount: 6}
  );
  expect(failed.diagnostics![0]).toBe(1);
  expect(failed.folds.every(fold => fold === 0xffffffff)).toBe(true);
  expect(failed.status.every(status => status === 3)).toBe(true);
  const deficient = computeSpatialCrossValidationOnCPU(
    {...input, features: new Float32Array(6)},
    {...PROPS, design: {mode: 'environmental', featureCount: 1}}
  );
  expect(deficient.diagnostics![0]).toBe(2);
  expect(deficient.folds.every(fold => fold === 0xffffffff)).toBe(true);
});

it('selects a later generated restart over candidate zero, with hand-derived distances and SciPy confirmation', () => {
  const fixture = fixtures.generated;
  const input = {
    positions: new Float32Array([7, 10, 7, 9, 8, 2, 0, 6, 9, 5, 1, 11]),
    blocks: new Uint32Array(6),
    predictionPositions: new Float32Array([8, 3, 4, 5])
  };
  const predictionMean = (1 + Math.sqrt(17)) / 2;
  // Both training distance distributions lie wholly above the prediction distances [1, sqrt(17)].
  // Their Wasserstein distances are therefore differences of means, independently of a CDF scan.
  const firstObjective =
    (2 * Math.sqrt(37) + Math.sqrt(40) + Math.sqrt(80) + Math.sqrt(58) + Math.sqrt(82)) / 6 -
    predictionMean;
  const winningObjective =
    (Math.sqrt(29) + 2 * Math.sqrt(20) + Math.sqrt(50) + Math.sqrt(80) + 10) / 6 - predictionMean;
  expect(fixture.partitions[0]).toEqual([0, 0, 0, 1, 0, 1]);
  expect(fixture.partitions[1]).toEqual([1, 1, 0, 1, 0, 1]);
  expect(fixture.objectives[0]).toBeCloseTo(firstObjective, 12);
  expect(fixture.objectives[1]).toBeCloseTo(winningObjective, 12);
  for (const candidateCount of [1, 3]) {
    const result = computeSpatialCrossValidationOnCPU(input, {
      ...PROPS,
      seed: 1,
      design: {mode: 'knndm', candidateCount}
    });
    const selected = candidateCount === 1 ? 0 : 1;
    expect(Array.from(result.folds)).toEqual(fixture.partitions[selected]);
    expect(Array.from(result.diagnostics!)).toEqual([
      0,
      candidateCount,
      Math.fround(selected === 0 ? firstObjective : winningObjective),
      selected
    ]);
    if (candidateCount === 3) {
      expectFixture(result, fixture);
      expect(Array.from(result.counts)).toEqual([4, 2, 0, 2, 4, 0]);
    }
  }
});

it('validates malformed arrays and both operation constructions without applying GPU work caps to CPU', () => {
  const input = {positions: new Float32Array(16), blocks: new Uint32Array(8)};
  expect(() =>
    computeSpatialCrossValidationOnCPU({...input, positions: new Float32Array(15)}, PROPS)
  ).toThrow(/positions/);
  expect(() =>
    computeSpatialCrossValidationOnCPU({...input, mask: new Uint32Array(7)}, PROPS)
  ).toThrow(/mask/);
  expect(() =>
    computeSpatialCrossValidationOnCPU(input, {
      ...PROPS,
      design: {mode: 'environmental', featureCount: 2}
    })
  ).toThrow(/features/);
  expect(() =>
    computeSpatialCrossValidationOnCPU(input, {...PROPS, design: {mode: 'knndm'}})
  ).toThrow(/predictionPositions/);
  expect(() =>
    computeSpatialCrossValidationOnCPU(
      {...input, predictionPositions: new Float32Array(2), candidates: new Uint32Array(1)},
      {...PROPS, design: {mode: 'knndm'}}
    )
  ).toThrow(/candidates/);
  const shapes = {
    positions: {format: 'float32x2' as const, length: 8},
    blocks: {format: 'uint32' as const, length: 8}
  };
  expect(() =>
    constructSamplingContributor(new GPUSpatialCrossValidationOperation(PROPS), shapes)
  ).not.toThrow();
  for (const change of [{foldCount: 1}, {bufferDistance: -1}, {seed: -1}]) {
    expect(() => new CPUSpatialCrossValidationOperation({...PROPS, ...change})).toThrow(
      /cross-validation-test/
    );
    expect(() =>
      constructSamplingContributor(
        new GPUSpatialCrossValidationOperation({...PROPS, ...change}),
        shapes
      )
    ).toThrow(/cross-validation-test/);
  }
  for (const Operation of [GPUSpatialCrossValidationOperation, CPUSpatialCrossValidationOperation])
    expect(() => new Operation({...PROPS, output: {...PROPS.output, folds: 'positions'}})).toThrow(
      /cross-validation-test/
    );
  expect(
    () => new CPUSpatialCrossValidationOperation({...PROPS, rowCount: 65537, foldCount: 65})
  ).not.toThrow();
  expect(
    () =>
      new CPUSpatialCrossValidationOperation({
        ...PROPS,
        rowCount: 257,
        foldCount: 17,
        design: {
          mode: 'environmental',
          featureCount: 9,
          features: 'features',
          maximumIterations: 1025
        }
      })
  ).not.toThrow();
});
