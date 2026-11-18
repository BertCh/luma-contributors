// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import {validateGPUComputationSidecars} from '@luma-contributors/analysis/computation-record';
import {findViewNodes} from '../../../src/computation-views/view-node';

test('G4 track paths pick track rows and legacy vertex domains never become feature picks', async () => {
  const views = await import('../../../src/computation-views/internal');
  const {fromRecordedLensFile, toRecordedLensFile} = await import(
    '../../../src/computation-lenses/registry-implementation'
  );
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../../docs/lens-records/gpu-spatial-analysis.zone-events.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const tracks = record.captures!.snapshots.find(capture => capture.name === 'tracks')!;
  const offsets = record.captures!.snapshots.find(capture => capture.name === 'trackOffsets')!;
  expect(Array.from(offsets.values).slice(1, 3)).toEqual([65, 133]);
  expect(tracks.keySpace).toBe('fix');
  for (const name of ['tracks', 'countedTracks'])
    expect(
      record.captures!.snapshots.find(capture => capture.name === name)!.geometry
    ).toHaveProperty('featureKeySpace', 'track');
  // Model a legacy recording explicitly: keep vertex keys, but remove feature-domain metadata.
  const legacyRecord = {
    ...record,
    captures: {
      ...record.captures!,
      snapshots: record.captures!.snapshots.map(capture => {
        if (capture.name !== 'tracks' && capture.name !== 'countedTracks') return capture;
        const geometry = {...capture.geometry!};
        if ('featureKeySpace' in geometry) delete geometry.featureKeySpace;
        return {...capture, geometry};
      })
    }
  };
  const context = views.createViewContext(legacyRecord);
  const legacy = views.renderResultView(views.buildResultViewModel(context), context);
  expect(
    findViewNodes(
      legacy,
      node =>
        node.tag === 'path' &&
        node.attributes['data-capture-id'] === tracks.id &&
        node.attributes['data-action'] === 'pick'
    )
  ).toHaveLength(0);
  // The untouched recaptured record is the positive fixture.
  const updated = record;
  expect(
    validateGPUComputationSidecars(updated, updated).issues.filter(issue => issue.level === 'error')
  ).toEqual([]);
  const invalid = {
    ...updated,
    captures: {
      ...updated.captures!,
      snapshots: updated.captures!.snapshots.map(capture =>
        capture.name === 'tracks'
          ? {...capture, geometry: {...capture.geometry!, featureKeySpace: 'missing'}}
          : capture
      )
    }
  };
  expect(validateGPUComputationSidecars(invalid, invalid).issues).toContainEqual(
    expect.objectContaining({
      path: 'captures.snapshots[2].geometry.featureKeySpace',
      message: 'Unknown key space',
      level: 'error'
    })
  );
  const roundTrip = fromRecordedLensFile(
    toRecordedLensFile(updated, {
      scenarioId: 'gpu-spatial-analysis.zone-events',
      recordedAt: '2026-11-18T00:00:00Z',
      adapterLabel: 'Recorded fixture'
    })
  ).record;
  expect(
    roundTrip.captures!.snapshots.find(capture => capture.name === 'tracks')!.geometry
  ).toHaveProperty('featureKeySpace', 'track');
  const controller = new views.LensController({
    record: roundTrip,
    registry: views.createDefaultViewRegistry(),
    lensId: 'G4',
    rung: 'result'
  });
  try {
    const path = findViewNodes(
      controller.getFrame().tree,
      node =>
        node.tag === 'path' &&
        node.attributes['data-capture-id'] === tracks.id &&
        node.attributes['data-feature-row'] === 1
    )[0];
    expect(path.attributes).toMatchObject({
      'data-action': 'pick',
      'data-key-space': 'track',
      'data-key': 1
    });
    controller.dispatch({
      type: 'pick',
      axis: 'entity',
      keys: [Number(path.attributes['data-key'])],
      keySpace: String(path.attributes['data-key-space'])
    });
    expect(controller.getState().clauses).toMatchObject([
      {keySpace: 'track', predicate: {keys: [1]}}
    ]);
    const highlighted = findViewNodes(
      controller.getFrame().tree,
      node => node.tag === 'path' && node.attributes['data-state'] === 'highlight'
    );
    expect(highlighted.length).toBeGreaterThan(0);
    expect(highlighted.every(node => node.attributes['data-feature-row'] === 1)).toBe(true);
  } finally {
    controller.destroy();
  }
});

test('R2: showcase exploration loads the kriging presentation descriptor', async () => {
  vi.resetModules();
  const {loadLiveLensModules} = await import(
    '../../../../../examples/deck/spatial-analysis-showcase/src/docs/load-live-lens-modules'
  );
  const {views, lenses} = await loadLiveLensModules();
  const presentation = lenses.getLensScenario('gpu-spatial-analysis.kriging')?.presentation;
  expect(presentation).toEqual({
    primaryOutputCaptureId: 'gpu-spatial-analysis-kriging-snapshots-rainfall-0'
  });
  const recorded = lenses.fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../../docs/lens-records/gpu-spatial-analysis.kriging.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  );
  const record = recorded.record;
  const twin = views.buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'result'});
  expect(twin.description).toContain('→ rainfall');
  expect(twin.keyNumber).toEqual({label: 'cells', value: '4,662'});
  const controller = new views.LensController({
    record,
    registry: views.createDefaultViewRegistry(),
    lensId: 'review',
    rung: 'result',
    presentation
  });
  try {
    expect(controller.getFrame().description.alt).toContain('→ rainfall');
    expect(controller.getFrame().description.alt).toContain('cells: 4,662');
    const marks = (domain: string) =>
      findViewNodes(
        controller.getFrame().tree,
        node =>
          node.attributes['data-action'] === 'pick' && node.attributes['data-key-space'] === domain
      );
    expect(marks('station').length).toBeGreaterThan(0);
    controller.dispatch({type: 'pick', axis: 'entity', keys: [0], keySpace: 'station'});
    expect(controller.getState().clauses).toMatchObject([
      {keySpace: 'station', predicate: {keys: [0]}}
    ]);
    expect(
      marks('station').some(
        mark => mark.attributes['data-key'] === 0 && mark.attributes['data-state'] === 'highlight'
      )
    ).toBe(true);
    expect(marks('cell').some(mark => mark.attributes['data-state'] === 'highlight')).toBe(false);
    controller.dispatch({type: 'pick', axis: 'entity', keys: [0], keySpace: 'cell'});
    expect(controller.getState().clauses).toMatchObject([
      {keySpace: 'cell', predicate: {keys: [0]}}
    ]);
    expect(
      marks('cell').some(
        mark => mark.attributes['data-key'] === 0 && mark.attributes['data-state'] === 'highlight'
      )
    ).toBe(true);
    expect(marks('station').some(mark => mark.attributes['data-state'] === 'highlight')).toBe(
      false
    );
  } finally {
    controller.destroy();
  }
});
