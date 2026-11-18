// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {fromRecordedLensFile, toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {LensController} from '../../../src/computation-views/hosts/lens-controller';
import {createDefaultViewRegistry} from '../../../src/computation-views/registry/default-registry';
import {makeV7Record} from '../fixtures/v7-lens-record';

test('Explore rejects a declared read absent from a supplied resource inventory', () => {
  const record = makeV7Record();
  record.graphs[0].resources = record.graphs[0].resources!.filter(
    resource => resource.id !== 'input'
  );
  const recorded = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.double',
    recordedAt: '2026-10-14T00:00:00Z',
    adapterLabel: 'Fixture GPU'
  });
  expect(() => fromRecordedLensFile(recorded)).toThrow('graphs[0].nodes[0].reads[0]');
  expect(() => fromRecordedLensFile(recorded)).toThrow('Unknown reference');
});

test('Explore decodes the gpu-core.double browser fixture and builds its table frame', () => {
  const recorded = toRecordedLensFile(makeV7Record(), {
    scenarioId: 'gpu-core.double',
    recordedAt: '2026-10-14T00:00:00Z',
    adapterLabel: 'Fixture GPU'
  });
  expect(() => {
    const {record} = fromRecordedLensFile(recorded);
    const controller = new LensController({
      record,
      registry: createDefaultViewRegistry(),
      lensId: 'browser-fixture',
      rung: 'table'
    });
    try {
      expect(controller.getFrame().tree.attributes['class']).toContain('ladder-lens-frame');
    } finally {
      controller.destroy();
    }
  }).not.toThrow();
});
