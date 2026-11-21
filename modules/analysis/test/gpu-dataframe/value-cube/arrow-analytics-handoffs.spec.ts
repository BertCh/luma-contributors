// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  prepareArrowKeyPairForGPUAnalytics,
  prepareArrowMeasureColumnsForGPUAnalytics
} from '../../../src/analysis-core/graph-support/arrow-analytics-normalization';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/index';
import {GPUGroupStatistics} from '../../../src/dataframe-analysis/group-statistics/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createVectorView} from '../../../src/testing/gpu-contributor-test-utils';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';
import {createNullableArrowMeasuresFixture} from './arrow-ingress-fixture';

it('UDF-01 Arrow paired keys gather the same labels and distinguish adjacent wide integers', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const integerKeys of [false, true]) {
    const left = integerKeys
      ? new arrow.Table({
          label: arrow.vectorFromArray([9007199254740992n, 9007199254740993n], new arrow.Uint64())
        })
      : new arrow.Table({label: arrow.vectorFromArray(['A', 'B'], new arrow.Utf8())});
    const right = integerKeys
      ? new arrow.Table({
          name: arrow.vectorFromArray([9007199254740993n, 9007199254740992n], new arrow.Uint64())
        })
      : new arrow.Table({name: arrow.vectorFromArray(['B', 'A'], new arrow.Utf8())});
    const pair = prepareArrowKeyPairForGPUAnalytics(left, 'label', right, 'name');
    const fixture = new RecipeTestFixture(device, `paired-arrow-${integerKeys}`);
    try {
      const format = pair.keyBits === 32 ? 'uint32' : 'uint32x2';
      const gathered = fixture.output('gathered', 'float32', 2);
      const counts = fixture.output('counts', 'uint32', 2);
      fixture.graph.add(
        new GPUKeyLookup({
          leftKeys: fixture.input('left', pair.left.chunks[0], format, 2),
          rightKeys: fixture.input('right', pair.right.chunks[0], format, 2),
          leftMask: fixture.input('left-validity', pair.left.validity[0], 'uint32', 2),
          rightMask: fixture.input('right-validity', pair.right.validity[0], 'uint32', 2),
          gather: [
            {
              column: fixture.input('values', Float32Array.of(20, 10), 'float32', 2),
              output: gathered.view
            }
          ],
          output: {matchCounts: counts.view}
        })
      );
      fixture.run();
      // By label/ID equality: left row 0 -> right row 1 -> 10; left row 1 -> right row 0 -> 20.
      expect(await fixture.readFloat32(gathered, 2)).toEqual([10, 20]);
      expect(await fixture.readUint32(counts, 2)).toEqual([1, 1]);
    } finally {
      fixture.destroy();
    }
  }
});

it('UDF-02 nullable Arrow measures retain COUNT(*) and independent finite group and lookup counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const source = createNullableArrowMeasuresFixture();
  const measures = prepareArrowMeasureColumnsForGPUAnalytics(source, ['x', 'y']);
  const fixture = new RecipeTestFixture(device, 'nullable-arrow-measures');
  try {
    const outputKeys = fixture.output('group-keys', 'uint32', 1);
    const groupCounts = fixture.output('group-counts', 'uint32', 1);
    const groupCount = fixture.output('group-count', 'uint32', 1);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const keys = createVectorView('keys', 'uint32', [
      fixture.input('keys-first', Uint32Array.of(7, 7), 'uint32', 2),
      fixture.input('keys-last', Uint32Array.of(7), 'uint32', 1)
    ]);
    const outputs = ['x', 'y'].map(name => ({
      name,
      count: fixture.output(`${name}-count`, 'uint32', 1),
      mean: fixture.output(`${name}-mean`, 'float32', 1),
      lookupCount: fixture.output(`${name}-lookup-count`, 'float32', 1),
      lookupMean: fixture.output(`${name}-lookup-mean`, 'float32', 1),
      gathered: fixture.output(`${name}-gathered`, 'float32', 1)
    }));
    fixture.graph.add(
      new GPUGroupStatistics({
        keys,
        columns: outputs.map(output => ({
          values: createVectorView(
            `${output.name}-chunks`,
            'float32',
            measures[output.name].chunks.map((chunk, index) =>
              fixture.input(`${output.name}-${index}`, chunk, 'float32', chunk.length)
            )
          ),
          statistics: ['count', 'mean'] as const,
          output: {counts: output.count.view, means: output.mean.view}
        })),
        output: {
          keys: outputKeys.view,
          counts: groupCounts.view,
          status: {count: groupCount.view, overflow: overflow.view}
        }
      })
    );
    const rightValues = outputs.map(output =>
      fixture.input(
        `${output.name}-lookup-values`,
        Float32Array.from(measures[output.name].chunks.flatMap(chunk => Array.from(chunk))),
        'float32',
        3
      )
    );
    const matchedRows = fixture.output('matched-rows', 'uint32', 1);
    const lookupPopulation = fixture.output('lookup-population', 'float32', 1);
    fixture.graph.add(
      new GPUKeyLookup({
        leftKeys: fixture.input('lookup-key', Uint32Array.of(7), 'uint32', 1),
        rightKeys: fixture.input('right-keys', Uint32Array.of(7, 7, 7), 'uint32', 3),
        gather: outputs.map((output, index) => ({
          column: rightValues[index],
          output: output.gathered.view
        })),
        aggregates: [
          {operation: 'count', output: lookupPopulation.view},
          ...outputs.flatMap((output, index) => [
            {
              operation: 'count' as const,
              column: rightValues[index],
              output: output.lookupCount.view
            },
            {operation: 'mean' as const, column: rightValues[index], output: output.lookupMean.view}
          ])
        ],
        output: {matchCounts: matchedRows.view}
      })
    );
    fixture.run();
    // Hand worked: 3 key-eligible rows; x=(0+2)/2=1, y=(12+4)/2=8.
    expect(await fixture.readUint32(groupCounts, 1)).toEqual([3]);
    expect(await fixture.readUint32(groupCount, 1)).toEqual([1]);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readUint32(matchedRows, 1)).toEqual([3]);
    expect(await fixture.readFloat32(lookupPopulation, 1)).toEqual([3]);
    for (const [index, expected] of [1, 8].entries()) {
      const output = outputs[index];
      expect(await fixture.readUint32(output.count, 1)).toEqual([2]);
      expect(await fixture.readFloat32(output.mean, 1)).toEqual([expected]);
      expect(await fixture.readFloat32(output.lookupCount, 1)).toEqual([2]);
      expect(await fixture.readFloat32(output.lookupMean, 1)).toEqual([expected]);
    }
    expect((await fixture.readFloat32(outputs[0].gathered, 1))[0]).toBeNaN();
    expect(await fixture.readFloat32(outputs[1].gathered, 1)).toEqual([12]);
  } finally {
    fixture.destroy();
  }
});

it('B1 decoded null dictionary keys have zero GPUKeyLookup matches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const type = new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32(), 73, false);
  const left = new arrow.Table({
    p: new arrow.Vector([
      arrow.makeData({
        type,
        data: Int32Array.of(1, 2, 0),
        dictionary: arrow.vectorFromArray(['p', null, 'q'], new arrow.Utf8())
      })
    ])
  });
  const right = new arrow.Table({q: arrow.vectorFromArray(['q', 'p'], new arrow.Utf8())});
  const pair = prepareArrowKeyPairForGPUAnalytics(left, 'p', right, 'q');
  const fixture = new RecipeTestFixture(device, 'dictionary-null-key');
  try {
    const matchCounts = fixture.output('match-counts', 'uint32', 3);
    fixture.graph.add(
      new GPUKeyLookup({
        leftKeys: fixture.input('left-keys', pair.left.chunks[0], 'uint32', 3),
        rightKeys: fixture.input('right-keys', pair.right.chunks[0], 'uint32', 2),
        leftMask: fixture.input('left-validity', pair.left.validity[0], 'uint32', 3),
        rightMask: fixture.input('right-validity', pair.right.validity[0], 'uint32', 2),
        output: {matchCounts: matchCounts.view}
      })
    );
    fixture.run();
    // Labels [null,q,p] against [q,p] have zero, one, one matches by hand.
    expect(await fixture.readUint32(matchCounts, 3)).toEqual([0, 1, 1]);
  } finally {
    fixture.destroy();
  }
});
