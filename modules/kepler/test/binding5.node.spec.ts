// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {isKeplerAllocationError} from '../src/runtime/dataset-registry';
import {preparePointLabelSource} from '../../../examples/deck/kepler-compute/src/label-source';
import {KeplerDeviceOutOfMemoryError, runWithOutOfMemoryScope} from '@luma-contributors/kepler';
import {
  listStyleFieldRequirements,
  requestStyleFieldPreparation
} from '../../../examples/deck/kepler-compute/src/binding4-style-field-preparation';

function makeScopedDevice(outOfMemory: {message: string} | null) {
  const calls: string[] = [];
  return {
    calls,
    handle: {
      pushErrorScope: (filter: string) => calls.push(`push:${filter}`),
      popErrorScope: async () => {
        calls.push('pop');
        return outOfMemory;
      }
    }
  };
}

describe('budget throws are typed allocation errors', () => {
  it('label source byte budget is recognised, others are not', () => {
    let caught: unknown;
    try {
      preparePointLabelSource({
        fieldName: 'name',
        chunks: [{id: 'a', values: ['x', 'y']}],
        maxCharactersPerLabel: 8,
        maxGlyphs: 8,
        maxSourceBytes: 4
      });
    } catch (error) {
      caught = error;
    }
    expect(isKeplerAllocationError(caught)).toBe(true);
    expect((caught as Error).message).toBe('Label source exceeds its admitted byte budget');
    expect(
      isKeplerAllocationError(new Error('Label source exceeds its admitted byte budget'))
    ).toBe(false);
  });
});

describe('runWithOutOfMemoryScope', () => {
  it('returns the step result and pops the scope when nothing failed', async () => {
    const device = makeScopedDevice(null);
    await expect(runWithOutOfMemoryScope(device, async () => 7)).resolves.toBe(7);
    expect(device.calls).toEqual(['push:out-of-memory', 'pop']);
  });

  it('turns a reported device out-of-memory into a typed allocation error', async () => {
    const device = makeScopedDevice({message: 'buffer too large'});
    const error = await runWithOutOfMemoryScope(device, async () => 1).catch(
      (caught: unknown) => caught
    );
    expect(isKeplerAllocationError(error)).toBe(true);
    expect((error as Error).cause).toBeInstanceOf(KeplerDeviceOutOfMemoryError);
    expect((error as Error).cause).toHaveProperty(
      'message',
      'Device out of memory: buffer too large'
    );
  });

  it('rethrows the step failure and still pops the scope', async () => {
    const device = makeScopedDevice({message: 'ignored'});
    const failure = new Error('step failed');
    await expect(
      runWithOutOfMemoryScope(device, async () => {
        throw failure;
      })
    ).rejects.toBe(failure);
    expect(device.calls).toEqual(['push:out-of-memory', 'pop']);
  });

  it('runs the step unwrapped on devices without error scopes', async () => {
    await expect(runWithOutOfMemoryScope({}, () => 3)).resolves.toBe(3);
    await expect(runWithOutOfMemoryScope({handle: {}}, () => 4)).resolves.toBe(4);
  });
});

describe('style field preparation', () => {
  const requested: string[] = [];
  const coordinator = {
    request(datasetId: string, name: string) {
      requested.push(`${datasetId}:${name}`);
      if (name === 'pending') return {status: 'pending'} as const;
      return name === 'ghost'
        ? ({status: 'unavailable', reason: 'Not a column.'} as const)
        : ({status: 'requested'} as const);
    }
  };

  it('lists every style channel once in a stable order and ignores empty fields', () => {
    expect(
      listStyleFieldRequirements({
        elevationOffsetField: {name: 'e'},
        radiusField: {name: 'r'},
        colorField: {name: 'c'},
        sizeField: null,
        coverageField: {name: ''},
        heightField: undefined
      })
    ).toEqual([
      {channel: 'colorField', name: 'c'},
      {channel: 'radiusField', name: 'r'},
      {channel: 'elevationOffsetField', name: 'e'}
    ]);
    expect(listStyleFieldRequirements({})).toEqual([]);
  });

  it('requests each missing column once and skips resident columns', () => {
    requested.length = 0;
    const messages = requestStyleFieldPreparation(
      'dataset',
      {
        colorField: {name: 'resident'},
        strokeColorField: {name: 'shared'},
        sizeField: {name: 'shared'},
        radiusField: {name: 'pending'},
        heightField: {name: 'ghost'}
      },
      {resident: {}},
      coordinator
    );
    expect(requested).toEqual(['dataset:shared', 'dataset:pending', 'dataset:ghost']);
    expect(messages).toEqual([
      'strokeColor field "shared" is not on the GPU. Preparing it now.',
      'radius field "pending" is not on the GPU. Preparing it now.',
      'height field "ghost" is not on the GPU. Not a column.'
    ]);
  });

  it('returns no messages when every style field is resident', () => {
    requested.length = 0;
    expect(
      requestStyleFieldPreparation('dataset', {colorField: {name: 'a'}}, {a: {}}, coordinator)
    ).toEqual([]);
    expect(requested).toEqual([]);
  });
});
