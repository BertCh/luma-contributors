// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// LENS-101: the headline of a saved record must not depend on which scenario modules a reader
// happened to import. These tests build twins in a fresh module graph with no scenario registered.

import {readFileSync} from 'node:fs';
import {beforeEach, describe, expect, test, vi} from 'vitest';

const KRIGING_ID = 'gpu-spatial-analysis.kriging';
const RAINFALL_CAPTURE_ID = 'gpu-spatial-analysis-kriging-snapshots-rainfall-0';

function readKrigingRecord(): unknown {
  return JSON.parse(
    readFileSync(
      new URL(
        '../../../../../docs/lens-records/gpu-spatial-analysis.kriging.json',
        import.meta.url
      ),
      'utf8'
    )
  );
}

async function importFreshReader() {
  vi.resetModules();
  const registry = await import('../../../src/computation-lenses/registry-implementation');
  const twin = await import('../../../src/computation-views/twin/static-twin');
  return {...registry, ...twin};
}

describe('LENS-101: standalone record presentation', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  test('the committed kriging record headlines rainfall without scenario registration', async () => {
    const {buildLensTwin, getLensScenario, listLensScenarios} = await importFreshReader();
    expect(getLensScenario(KRIGING_ID)).toBeUndefined();
    expect(listLensScenarios()).toEqual([]);
    const recorded = readKrigingRecord() as Parameters<typeof buildLensTwin>[0];
    for (const size of ['full', 'thumbnail'] as const) {
      const twin = buildLensTwin(recorded, {scenario: KRIGING_ID, rung: 'result'}, {size});
      expect(twin.disabled).toBeUndefined();
      // 4,662 = 63 x 74 ten-kilometre cells (the scenario's `cell` key space).
      expect(twin.keyNumber).toEqual({label: 'cells', value: '4,662'});
      expect(twin.description).toContain('→ rainfall');
      expect(twin.description).not.toContain('uncertainCount');
    }
  });

  test('a recording written with presentation carries its primary output', async () => {
    const {buildLensTwin, fromRecordedLensFile, toRecordedLensFile} = await importFreshReader();
    const decoded = fromRecordedLensFile(readKrigingRecord());
    const variance = decoded.record.captures!.snapshots.find(
      capture => capture.name === 'variance'
    )!;
    const rewritten = toRecordedLensFile(decoded.record, {
      scenarioId: KRIGING_ID,
      recordedAt: decoded.recordedAt,
      adapterLabel: decoded.adapterLabel,
      presentation: {primaryOutputCaptureId: variance.id}
    });
    expect(rewritten.presentation).toEqual({primaryOutputCaptureId: variance.id});
    const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(rewritten)));
    expect(restored.presentation).toEqual({primaryOutputCaptureId: variance.id});
    // The record's own choice wins over the code-side default for old recordings.
    const twin = buildLensTwin(restored, {scenario: KRIGING_ID, rung: 'result'});
    expect(twin.description).toContain('→ variance');
    expect(twin.description).not.toContain('→ rainfall');
  });

  test('invalid recorded presentation is rejected at the reader boundary', async () => {
    const {fromRecordedLensFile} = await importFreshReader();
    const raw = readKrigingRecord() as Record<string, unknown>;
    for (const presentation of [
      'rainfall',
      {primaryOutputCaptureId: 7},
      {primaryOutputCaptureId: ''}
    ]) {
      expect(() => fromRecordedLensFile({...raw, presentation})).toThrow(
        'Invalid lens record presentation'
      );
    }
  });

  test('registered scenarios take the shared presentation table', async () => {
    vi.resetModules();
    const {getLensScenarioPresentation} = await import(
      '../../../src/computation-lenses/scenario-presentations'
    );
    await import('../../../src/computation-lenses/scenarios/gpu-spatial-analysis/kriging.lens');
    const {getLensScenario} = await import(
      '../../../src/computation-lenses/registry-implementation'
    );
    expect(getLensScenarioPresentation(KRIGING_ID)).toEqual({
      primaryOutputCaptureId: RAINFALL_CAPTURE_ID
    });
    expect(getLensScenario(KRIGING_ID)?.presentation).toEqual({
      primaryOutputCaptureId: RAINFALL_CAPTURE_ID
    });
  });
});
