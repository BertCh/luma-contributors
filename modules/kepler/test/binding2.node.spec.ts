// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  describeUnpreparedFilterField,
  diffNativeFilters,
  planDictionaryCoverage
} from '../../../examples/deck/kepler-compute/src/binding2-filter-lifecycle';
import {DatasetOperationQueue} from '../../../examples/deck/kepler-compute/src/binding2-dataset-operations';
import {
  describePolygonFilterProblem,
  listPolygonFilterProblems
} from '../../../examples/deck/kepler-compute/src/binding2-polygon-reasons';
import {prepareNativePolygonFilter} from '../../../examples/deck/kepler-compute/src/resident-table-binding';
import {planKeplerComputeFilters} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

const fields = [{name: 'a'}, {name: 'b'}];
const range = (id: string, name: string, value: number[], extra = {}) => ({
  id,
  dataId: ['d'],
  name: [name],
  type: 'range',
  value,
  ...extra
});
const polygon = (coordinates: number[][]) => ({
  id: 'p',
  dataId: ['d'],
  type: 'polygon',
  layerId: ['L'],
  value: {type: 'Feature', properties: {}, geometry: {type: 'Polygon', coordinates: [coordinates]}}
});
const bowtie = polygon([
  [0, 0],
  [2, 2],
  [2, 0],
  [0, 2],
  [0, 0]
]);
const square = polygon([
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
  [0, 0]
]);

describe('BINDING2 polygon reasons', () => {
  it('reports specific invalid-shape reasons and accepts simple squares', () => {
    expect(describePolygonFilterProblem(bowtie)).toMatch(/crosses or touches itself/);
    expect(describePolygonFilterProblem(square)).toBeUndefined();
    expect(
      listPolygonFilterProblems([bowtie, {...square, id: 'q'}, {...bowtie, enabled: false}])
    ).toHaveLength(1);
  });
  it('prepareNativePolygonFilter throws the reason and names unregistered layers', () => {
    const layers = [{id: 'L', type: 'point', config: {dataId: 'd'}}];
    expect(() => prepareNativePolygonFilter(bowtie as never, 'd', layers, ['L'])).toThrow(
      /crosses/
    );
    expect(() => prepareNativePolygonFilter(square as never, 'd', layers, [])).toThrow(/: L$/);
    expect(prepareNativePolygonFilter(square as never, 'd', layers, ['L'])?.layerIds).toEqual([
      'L'
    ]);
  });
});

describe('BINDING2 filter lifecycle', () => {
  it('classifies deletion, reorder, reassignment, inversion and duplicate fields', () => {
    const one = range('r1', 'a', [0, 1]);
    const two = range('r2', 'a', [0, 2]);
    const three = range('r3', 'b', [0, 3]);
    const diff = diffNativeFilters(
      [one, two, three] as never,
      [
        {...three, name: ['a']},
        {...one, inverted: true, value: [1, 2]}
      ] as never,
      'd',
      fields
    );
    expect(diff.removed).toEqual(['r2']);
    expect(diff.reordered).toBe(true);
    expect(diff.reassigned).toEqual(['r3']);
    expect(diff.inversionChanged).toEqual(['r1']);
    expect(diff.valueChanged).toEqual(['r1']);
    expect(diff.duplicateFields).toEqual({a: ['r3', 'r1']});
  });
  it('plans multiple ranges on one field and independent inverted predicates', () => {
    const bindings = {a: {kind: 'range', dimension: 'a', domain: [0, 10]}} as never;
    const filters = [range('r1', 'a', [0, 5]), range('r2', 'a', [3, 9], {inverted: true})] as never;
    const independent = planKeplerComputeFilters('d', fields, bindings, filters, {
      independentPredicates: true
    });
    expect(independent.display).toHaveLength(2);
    expect(() => planKeplerComputeFilters('d', fields, bindings, filters)).toThrow(/Inverted/);
    const reversed = planKeplerComputeFilters('d', fields, bindings, [filters[1], filters[0]], {
      independentPredicates: true
    });
    expect(reversed.filterIds).toEqual(['r2', 'r1']);
    const disjoint = planKeplerComputeFilters('d', fields, bindings, [
      range('x', 'a', [0, 1]),
      range('y', 'a', [2, 3])
    ] as never);
    expect(disjoint.display).toEqual([{kind: 'empty', dimension: 'a'}]);
  });
  it('maps one categorical filter through each dataset dictionary', () => {
    const filter = {
      id: 'c',
      dataId: ['d1', 'd2'],
      name: ['k', 'k'],
      type: 'multiSelect',
      value: ['x', 'z']
    };
    const dictionary = (categories: string[]) =>
      ({k: {kind: 'category', dimension: 'k', domain: categories, categories}}) as never;
    const coverage = planDictionaryCoverage(filter as never, [
      {datasetId: 'd1', fields: [], bindings: dictionary(['z', 'y', 'x'])},
      {datasetId: 'd2', fields: [], bindings: dictionary(['x', 'y'])}
    ]);
    expect(coverage[0].selectedIndices).toEqual([0, 2]);
    expect(coverage[0].missingLabels).toEqual([]);
    expect(coverage[1].selectedIndices).toEqual([0]);
    expect(coverage[1].missingLabels).toEqual(['z']);
  });
  it('names the unprepared field', () => {
    const bindings = {} as never;
    expect(() =>
      planKeplerComputeFilters('d', fields, bindings, [range('r1', 'a', [0, 1])] as never)
    ).toThrow(describeUnpreparedFilterField(range('r1', 'a', [0, 1]) as never, 'd', 'a'));
  });
});

describe('BINDING2 dataset operation queue', () => {
  it('serializes operations, skips superseded replacements and survives failures', async () => {
    const queue = new DatasetOperationQueue();
    const log: string[] = [];
    const gate = new Promise<void>(resolve => setTimeout(resolve, 5));
    const first = queue.run('d', 'replace', async () => {
      log.push('replace1:start');
      await gate;
      log.push('replace1:end');
    });
    const second = queue.run('d', 'replace', async () => void log.push('replace2'));
    const removal = queue.run('d', 'remove', async () => {
      log.push('remove');
      throw new Error('boom');
    });
    const third = queue.run('d', 'replace', async () => void log.push('replace3'));
    const other = queue.run('e', 'remove', async () => void log.push('other'));
    await first;
    expect(await second).toBeUndefined();
    await expect(removal).rejects.toThrow('boom');
    await third;
    await other;
    await queue.whenIdle();
    expect(log).toEqual(['replace1:start', 'other', 'replace1:end', 'remove', 'replace3']);
    expect(queue.busyDatasetIds).toEqual([]);
  });
});
