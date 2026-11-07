// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  assertSandboxSourcesCovered,
  getNativeDatasetReferences
} from '../../../../examples/deck/kepler-compute/src/sandbox-source-validation';

test('native workspace source coverage includes layers, multi-source filters, charts and tooltip datasets', () => {
  const configuration = {
    config: {
      visState: {
        layers: [{id: 'layer-id', config: {dataId: 'observations'}}],
        filters: [{dataId: ['observations', 'comparison']}],
        charts: [{dataId: 'chart-source'}],
        interactionConfig: {tooltip: {fieldsToShow: {'tooltip-source': [{name: 'value'}]}}}
      }
    }
  };
  expect(getNativeDatasetReferences(configuration)).toEqual([
    'observations',
    'comparison',
    'chart-source',
    'tooltip-source'
  ]);
  expect(() => assertSandboxSourcesCovered(configuration, ['observations'])).toThrow(
    /comparison, chart-source, tooltip-source/
  );
  expect(() =>
    assertSandboxSourcesCovered(configuration, [
      'observations',
      'comparison',
      'chart-source',
      'tooltip-source'
    ])
  ).not.toThrow();
});

test('unassigned layers do not require sources and names unrelated to sources are preserved', () => {
  expect(
    getNativeDatasetReferences({
      layers: [{id: 'unassigned', config: {dataId: null}}, {config: {dataId: ''}}],
      label: 'other-dataset',
      columns: ['value']
    })
  ).toEqual([]);
  expect(() => getNativeDatasetReferences({config: {dataId: 7}})).toThrow(
    /Invalid native dataset reference/
  );
});

test('disabled foreign layers and empty tooltip field lists still require replayable source manifests', () => {
  expect(() =>
    assertSandboxSourcesCovered({layers: [{config: {isVisible: false, dataId: 'external'}}]}, [
      'observations'
    ])
  ).toThrow(/external/);
  expect(() =>
    assertSandboxSourcesCovered({tooltip: {fieldsToShow: {external: []}}}, ['observations'])
  ).toThrow(/external/);
});
