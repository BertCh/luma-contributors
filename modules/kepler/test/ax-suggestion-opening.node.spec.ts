// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {runSuggestionOpening} from '../../../examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs';

type InputMapping = {
  mode: string;
  datasetId: string;
  geometry: string;
  longitude: string | null;
  latitude: string | null;
};

function createOpeningProbe(actualMappings?: Record<string, InputMapping>) {
  // Recorded independently before the click, with distinct coordinate names and named ports.
  const expected = {
    bindingId: 'cell-compare',
    inputMappings: {
      before: {
        mode: 'native',
        datasetId: 'baseline',
        geometry: 'numeric-point-pair',
        longitude: 'before_east',
        latitude: 'before_north'
      },
      after: {
        mode: 'native',
        datasetId: 'followup',
        geometry: 'numeric-point-pair',
        longitude: 'after_east',
        latitude: 'after_north'
      }
    },
    fieldBindings: {valueColumn: 'reading'}
  };
  const mappings = actualMappings ?? expected.inputMappings;
  const suggestion = {
    count: async () => 1,
    getAttribute: async (name: string) =>
      ({
        'data-binding-id': expected.bindingId,
        'data-input-mappings': JSON.stringify(expected.inputMappings),
        'data-field-bindings': JSON.stringify(expected.fieldBindings)
      })[name]
  };
  const fieldsets = Object.entries(mappings).map(([inputName, mapping]) => ({
    getAttribute: () => inputName,
    querySelector(selector: string) {
      const key = selector.includes('X / longitude')
        ? 'longitude'
        : selector.includes('Y / latitude')
          ? 'latitude'
          : selector.includes('geometry field')
            ? 'geometry'
            : selector.includes(' dataset')
              ? 'datasetId'
              : 'mode';
      const value = mapping[key];
      return value === null ? null : {value};
    }
  }));
  const catalog = {
    waitFor: vi.fn(),
    getByTestId(identifier: string) {
      if (identifier !== 'catalog-parameter-valueColumn')
        throw new Error(`Unexpected field ${identifier}`);
      return {
        evaluate: async (
          read: (element: {tagName: string; multiple: boolean; value: string}) => unknown
        ) => read({tagName: 'SELECT', multiple: false, value: 'reading'})
      };
    },
    locator(selector: string) {
      return {
        inputValue: async () =>
          selector === '#catalog-operation' ? expected.bindingId : 'reading',
        evaluateAll: async (evaluate: (elements: typeof fieldsets) => unknown) =>
          evaluate(fieldsets)
      };
    }
  };
  const recorder = {
    page: {getByTestId: () => catalog},
    click: vi.fn(),
    observe: async (_label: string, observe: () => unknown) => observe(),
    waitUntil: vi.fn(),
    observations: {} as Record<string, unknown>
  };
  const surface = {locator: () => ({first: () => suggestion})};
  return {expected, recorder, surface};
}

test('finding 5: opening evidence compares each named input and exact coordinates after one click', async () => {
  const {expected, recorder, surface} = createOpeningProbe();
  await runSuggestionOpening(recorder, surface);
  expect(recorder.click).toHaveBeenCalledOnce();
  expect(recorder.observations['openingEvidence']).toEqual({
    expected,
    mappings: expected.inputMappings
  });
});

test('finding 5: swapped before/after datasets fail even with the same dataset multiset', async () => {
  const original = createOpeningProbe().expected.inputMappings;
  const {recorder, surface} = createOpeningProbe({
    before: {...original.before, datasetId: original.after.datasetId},
    after: {...original.after, datasetId: original.before.datasetId}
  });
  await expect(runSuggestionOpening(recorder, surface)).rejects.toThrow(
    'input before mapping mismatch'
  );
});

test.each([
  'longitude',
  'latitude',
  'geometry'
])('finding 5: a nonempty but incorrect %s fails the opening gate', async field => {
  const original = createOpeningProbe().expected.inputMappings;
  const {recorder, surface} = createOpeningProbe({
    ...original,
    before: {...original.before, [field]: 'wrong-but-nonempty'}
  });
  await expect(runSuggestionOpening(recorder, surface)).rejects.toThrow(
    'input before mapping mismatch'
  );
});

test('finding 5: renamed or missing input ports fail rather than matching by dataset ID', async () => {
  const original = createOpeningProbe().expected.inputMappings;
  const {recorder, surface} = createOpeningProbe({first: original.before, second: original.after});
  await expect(runSuggestionOpening(recorder, surface)).rejects.toThrow(
    'input before mapping mismatch'
  );
});

test('finding 5: incomplete expected evidence cannot qualify an opening', async () => {
  const {expected, recorder, surface} = createOpeningProbe();
  Reflect.deleteProperty(expected.inputMappings.before, 'longitude');
  await expect(runSuggestionOpening(recorder, surface)).rejects.toThrow(
    'missing complete expected named input mappings'
  );
  expect(recorder.click).not.toHaveBeenCalled();
});
