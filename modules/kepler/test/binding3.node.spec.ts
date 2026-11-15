// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  listStyleFieldRequirements,
  requestStyleFieldPreparation
} from '../../../examples/deck/kepler-compute/src/binding4-style-field-preparation';
import {KeplerDeviceOutOfMemoryError} from '@luma-contributors/kepler';
import {KeplerAllocationError} from '../src/runtime/dataset-registry';
import {describe, expect, it} from 'vitest';
import {
  CategoryCoverageRegistry,
  describeScalarRebuildReason,
  listCategoryCoverageWarnings,
  makeFilterRevisionReport
} from '../../../examples/deck/kepler-compute/src/binding3-filter-runtime';
import {FieldPreparationCoordinator} from '../../../examples/deck/kepler-compute/src/binding3-field-preparation';
import {
  getCoordinateRoleSetKey,
  groupLayersByCoordinateRoleSet
} from '../../../examples/deck/kepler-compute/src/binding3-role-sets';
import {
  describeRestoreAllocationFailure,
  findSourcesExceedingBudget,
  isAllocationFailure,
  RestoreAllocationError
} from '../../../examples/deck/kepler-compute/src/binding3-restore-recovery';
import {RestoreTransactionError} from '../../../examples/deck/kepler-compute/src/restore-transaction';
import {planKeplerComputeFilters} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

const fields = [{name: 'kind'}, {name: 'value'}];
const dictionary = (labels: string[]) => ({
  kind: {dimension: 'kind', kind: 'category', domain: labels, categories: labels}
});
const select = (id: string, value: string[], extra = {}) => ({
  id,
  dataId: ['a', 'b'],
  name: ['kind', 'kind'],
  type: 'multiSelect',
  value,
  ...extra
});
const range = (id: string, value: number[], extra = {}) => ({
  id,
  dataId: ['a'],
  name: ['value'],
  type: 'range',
  value,
  ...extra
});

describe('BINDING3 filter revision reports', () => {
  it('explains which edit rebuilt the scalar selection', () => {
    const before = [range('r1', [0, 1]), range('r2', [0, 5])];
    const afterValue = [range('r2', [1, 4]), range('r1', [0, 1], {inverted: true})];
    const inPlace = makeFilterRevisionReport(
      before as never,
      afterValue as never,
      'a',
      fields,
      false
    );
    expect(inPlace.diff.reordered).toBe(true);
    expect(inPlace.diff.inversionChanged).toEqual(['r1']);
    expect(inPlace.rebuildReason).toBe('inversion toggled in place');
    const removed = makeFilterRevisionReport(
      before as never,
      [before[0]] as never,
      'a',
      fields,
      true
    );
    expect(removed.rebuildReason).toBe('removed r2');
    const added = makeFilterRevisionReport([] as never, before as never, 'a', fields, true);
    expect(added.diff.added).toEqual(['r1', 'r2']);
    expect(added.diff.duplicateFields['value']).toEqual(['r1', 'r2']);
    const reassigned = makeFilterRevisionReport(
      [select('s', ['x'])] as never,
      [select('s', ['x'], {name: ['value', 'value']})] as never,
      'a',
      fields,
      true
    );
    expect(describeScalarRebuildReason(reassigned.diff, true)).toBe('reassigned s');
  });
});

describe('BINDING3 categorical filters across datasets', () => {
  it('plans each dataset through its own dictionary and reports absent labels', () => {
    const filter = select('s', ['x', 'z']);
    const planA = planKeplerComputeFilters(
      'a',
      fields,
      dictionary(['z', 'y', 'x']) as never,
      [filter] as never
    );
    const planB = planKeplerComputeFilters(
      'b',
      fields,
      dictionary(['x', 'y']) as never,
      [filter] as never
    );
    expect(planA.display[0]).toMatchObject({kind: 'category', value: [0, 2]});
    expect(planB.display[0]).toMatchObject({kind: 'category', value: [0]});
    expect(
      listCategoryCoverageWarnings([filter] as never, 'a', fields, dictionary(['z', 'x']) as never)
    ).toEqual([]);
    const warnings = listCategoryCoverageWarnings(
      [filter] as never,
      'b',
      fields,
      dictionary(['x', 'y']) as never
    );
    expect(warnings).toHaveLength(1);
    expect(warnings[0].missingLabels).toEqual(['z']);
    expect(warnings[0].message).toMatch(/only its other 1 selected categories match/);
    expect(
      listCategoryCoverageWarnings(
        [select('d', ['q'], {enabled: false})] as never,
        'b',
        fields,
        dictionary(['x']) as never
      )
    ).toEqual([]);
  });

  it('registry keeps entries per owner and notifies once per change', () => {
    const registry = new CategoryCoverageRegistry();
    let notifications = 0;
    registry.subscribe(() => notifications++);
    const first = {};
    const second = {};
    const warning = (datasetId: string) => ({
      datasetId,
      filterId: 's',
      field: 'kind',
      missingLabels: ['z'],
      message: `missing in ${datasetId}`
    });
    registry.set('b', [warning('b')], first);
    registry.set('b', [warning('b')], second);
    expect(notifications).toBe(1);
    registry.clear('b', first);
    expect(registry.list()).toHaveLength(1);
    registry.clear('b', second);
    expect(registry.list()).toEqual([]);
    expect(notifications).toBe(2);
  });
});

describe('BINDING3 on-demand field preparation', () => {
  const makeHost = (sourceFields: string[], excluded: {name: string; reason: string}[] = []) => {
    const republished: string[] = [];
    const coordinator = new FieldPreparationCoordinator();
    coordinator.setHost({
      getSourceFieldNames: () => sourceFields,
      getExcludedFields: () => excluded,
      requestRepublish: datasetId => void republished.push(datasetId)
    });
    return {coordinator, republished};
  };
  it('requests one republish per dataset and never repeats a field', () => {
    const {coordinator, republished} = makeHost(['a', 'b']);
    expect(coordinator.request('d', 'a')).toEqual({status: 'requested'});
    expect(coordinator.request('d', 'b')).toEqual({status: 'pending'});
    coordinator.markSettled('d');
    expect(coordinator.request('d', 'a').status).toBe('already-attempted');
    expect(coordinator.request('d', 'b')).toEqual({status: 'requested'});
    expect(republished).toEqual(['d', 'd']);
    coordinator.forget('d');
    expect(coordinator.request('d', 'a')).toEqual({status: 'requested'});
  });
  it('explains excluded, missing and hostless requests without republishing', () => {
    const {coordinator, republished} = makeHost(
      ['a', 'big'],
      [{name: 'big', reason: 'overflows Float32'}]
    );
    expect(coordinator.request('d', 'big')).toMatchObject({status: 'already-attempted'});
    expect(coordinator.request('d', 'nope')).toMatchObject({status: 'unavailable'});
    expect(new FieldPreparationCoordinator().request('d', 'a').status).toBe('unavailable');
    expect(republished).toEqual([]);
  });
});

describe('BINDING3 coordinate role sets', () => {
  const layer = (id: string, lng: string, lat: string, dataId = 'd', altitude?: string) => ({
    id,
    type: 'point',
    config: {
      dataId,
      columns: {
        lng: {value: lng},
        lat: {value: lat},
        ...(altitude ? {altitude: {value: altitude}} : {})
      }
    }
  });
  it('groups layers by role set and marks the primary group', () => {
    const groups = groupLayersByCoordinateRoleSet(
      [
        layer('p1', 'x', 'y'),
        layer('p2', 'x', 'y'),
        layer('q', 'x2', 'y2'),
        layer('q3d', 'x2', 'y2', 'd', 'z'),
        layer('other', 'x', 'y', 'e')
      ],
      'd',
      {longitude: 'x', latitude: 'y'}
    );
    expect(groups.map(group => [group.layerIds, group.primary])).toEqual([
      [['p1', 'p2'], true],
      [['q'], false],
      [['q3d'], false]
    ]);
    expect(getCoordinateRoleSetKey(groups[1].roles)).not.toBe(
      getCoordinateRoleSetKey(groups[2].roles)
    );
  });
});

describe('BINDING3 restore allocation recovery', () => {
  it('recognizes allocation failures only', () => {
    expect(isAllocationFailure(new KeplerAllocationError('memory-budget'))).toBe(true);
    expect(
      isAllocationFailure(new Error('x', {cause: new KeplerAllocationError('reservation')}))
    ).toBe(true);
    expect(isAllocationFailure(new KeplerDeviceOutOfMemoryError('buffer out of memory'))).toBe(
      true
    );
    expect(isAllocationFailure(new Error('Memory budget exceeded for tier gpu'))).toBe(false);
    expect(isAllocationFailure(new Error('Injected restore failure'))).toBe(false);
  });
  it('reports every source that cannot fit with first-fit admission', () => {
    expect(
      findSourcesExceedingBudget(
        [
          {sourceId: 'a', bytes: 60},
          {sourceId: 'b', bytes: 60},
          {sourceId: 'c', bytes: 30},
          {sourceId: 'd', bytes: 20}
        ],
        100
      )
    ).toEqual(['b', 'd']);
  });
  it('wraps rolled-back allocation failures and passes others through', () => {
    const base = new RestoreTransactionError('Restoring b failed', 'b', ['a'], [], undefined, {
      cause: new KeplerAllocationError('memory-budget')
    });
    const wrapped = describeRestoreAllocationFailure(
      base,
      [
        {sourceId: 'a', bytes: 60},
        {sourceId: 'b', bytes: 60},
        {sourceId: 'c', bytes: 90}
      ],
      100
    );
    expect(wrapped).toBeInstanceOf(RestoreAllocationError);
    expect((wrapped as RestoreAllocationError).failedSourceIds).toEqual(['b', 'c']);
    expect((wrapped as RestoreAllocationError).rolledBackSourceIds).toEqual(['a']);
    const validation = new RestoreTransactionError('x', 'b', [], [], undefined, {
      cause: new Error('Bundle source is missing')
    });
    expect(describeRestoreAllocationFailure(validation, [], 100)).toBe(validation);
  });
});

describe('BINDING4 style field preparation', () => {
  it('lists style fields and requests only the missing ones through the coordinator', () => {
    const requested: string[] = [];
    const coordinator = {
      request: (datasetId: string, field: string) => {
        requested.push(`${datasetId}:${field}`);
        return field === 'late'
          ? ({status: 'requested'} as const)
          : ({status: 'unavailable', reason: 'Not a column.'} as const);
      }
    };
    const config = {
      colorField: {name: 'resident'},
      sizeField: {name: 'late'},
      heightField: {name: 'ghost'},
      strokeColorField: {name: 'late'}
    };
    expect(listStyleFieldRequirements(config).map(entry => entry.channel)).toEqual([
      'colorField',
      'strokeColorField',
      'sizeField',
      'heightField'
    ]);
    const messages = requestStyleFieldPreparation('d', config, {resident: {}}, coordinator);
    expect(requested).toEqual(['d:late', 'd:ghost']);
    expect(messages).toHaveLength(2);
    expect(messages[0]).toContain('Preparing it now.');
    expect(messages[1]).toContain('Not a column.');
  });
});
