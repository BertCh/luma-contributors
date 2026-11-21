// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  expectGeographicGroupBounds,
  expectSharedSourceFeatures,
  expectTruncatedPairPrefix,
  expectGroupColumnShapes,
  expectNearestDistance,
  expectProjectedUsability,
  prefillPackedOutputs,
  PACKED_PADDING_SENTINEL,
  withTestResourceCleanup
} from './geometry-output-review-test-utils';

it('B1 rejects copied-first-feature and unaligned geographic bounds when the singleton extends them', () => {
  const complete = {
    bounds: [179, 0, 181, 1, -178.5, 0.5, -178.5, 0.5],
    groupBounds: [179, 0, 181.5, 1],
    counts: [2],
    vertices: [3]
  };
  expectGeographicGroupBounds(complete, -178.5);
  for (const groupBounds of [
    [179, 0, 181, 1],
    [-178.5, 0, 181, 1]
  ])
    expect(() => expectGeographicGroupBounds({...complete, groupBounds}, -178.5)).toThrow();
});

it('B2 rejects a merged public feature range even with two correct polygons and source IDs', () => {
  const complete = {
    featureOffsets: [0, 1, 2],
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ]
      ],
      [
        [
          [2, 1],
          [5, 1],
          [5, 4],
          [2, 4]
        ]
      ]
    ],
    count: 2,
    sourceIds: [77, 77],
    overflow: 0,
    candidateOverflow: 0
  };
  expectSharedSourceFeatures(complete);
  expect(() => expectSharedSourceFeatures({...complete, featureOffsets: [0, 2, 2]})).toThrow();
});

it('B3 poisons all four complete packed output buffers before submission', () => {
  const contents = [
    new Uint32Array(27),
    new Uint32Array(9),
    new Uint32Array(54),
    new Uint32Array(27)
  ];
  prefillPackedOutputs(
    contents.map(values => ({
      buffer: {write: (written: Uint32Array) => values.set(written)},
      wordCount: values.length
    }))
  );
  contents.forEach(values =>
    expect(Array.from(values)).toEqual(new Array(values.length).fill(PACKED_PADDING_SENTINEL))
  );
});

it('B4 rejects corrupt right sentinels and nonzero left rows in a truncated prefix', () => {
  const complete = {leftRows: new Array(8).fill(0), rightRows: [0, 1, 2, 3, 4, 5, 6, 7]};
  expectTruncatedPairPrefix(complete);
  expect(() =>
    expectTruncatedPairPrefix({...complete, rightRows: [0, 1, 2, 3, 4, 5, 6, 0xffffffff]})
  ).toThrow();
  expect(() =>
    expectTruncatedPairPrefix({...complete, leftRows: [1, 0, 0, 0, 0, 0, 0, 0]})
  ).toThrow();
});

const groupColumns = {
  counts: [3, 0],
  bounds: [0, 0, 4, 8, NaN, NaN, NaN, NaN],
  meanCenters: [4 / 3, 8 / 3, NaN, NaN],
  weightedCenters: [1, 4, NaN, NaN],
  weightSums: [4, 0],
  medoids: [0, 0xffffffff],
  ellipses: [0, 1, 2, NaN, NaN, NaN],
  standardDistances: [Math.sqrt(19), NaN],
  overflow: [0]
};

it.each([
  'bounds',
  'meanCenters',
  'weightedCenters',
  'ellipses',
  'standardDistances'
])('S1 rejects a missing or extra %s suffix', name => {
  expectGroupColumnShapes(groupColumns);
  const values = groupColumns[name as keyof typeof groupColumns];
  for (const corrupted of [values.slice(0, -1), [...values, NaN]])
    expect(() => expectGroupColumnShapes({...groupColumns, [name]: corrupted})).toThrow();
});

it('S1 rejects extra nearest distance rows', () => {
  expectNearestDistance(Float32Array.of(1));
  expect(() => expectNearestDistance(Float32Array.of(1, 1))).toThrow();
});

it.each(['accessor', 'compile'])('S2 destroys owned buffers when %s fails', async stage => {
  const destroyed: string[] = [];
  const resources: {destroy(): void}[] = [];
  await expect(
    withTestResourceCleanup(resources, async () => {
      resources.push(
        {
          destroy: () => {
            destroyed.push('input');
          }
        },
        {
          destroy: () => {
            destroyed.push('readback');
          }
        }
      );
      throw new Error(`${stage} failure`);
    })
  ).rejects.toThrow(`${stage} failure`);
  expect(destroyed).toEqual(['readback', 'input']);
});

it('S2 destroys compiled resources before their buffers on success', async () => {
  const destroyed: string[] = [];
  const resources = [
    {
      destroy: () => {
        destroyed.push('buffer');
      }
    }
  ];
  const result = await withTestResourceCleanup(resources, async () => {
    resources.push({
      destroy: () => {
        destroyed.push('compiled');
      }
    });
    return 17;
  });
  expect(result).toBe(17);
  expect(destroyed).toEqual(['compiled', 'buffer']);
});

it('S3 accepts nonzero usability and rejects an unusable square or usable bowtie', () => {
  expectProjectedUsability([2, 0]);
  expect(() => expectProjectedUsability([0, 0])).toThrow();
  expect(() => expectProjectedUsability([2, 1])).toThrow();
});
