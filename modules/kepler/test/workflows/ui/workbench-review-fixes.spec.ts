// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {act, createElement} from 'react';
import {createRoot} from 'react-dom/client';
import {test} from 'vitest';
import {
  createKeplerCuratedAnalysisMapping,
  getKeplerCuratedAnalysis,
  getWorkflowOperationRegistry,
  profileKeplerFeatureCollection,
  replayKeplerCuratedAnalysisMapping,
  type CatalogParameterValue,
  type KeplerDatasetProfile
} from '@luma-contributors/kepler';
import {WorkbenchAnalysisMapping} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-mapping';
import {
  getSidePanelRecommendations,
  WorkbenchSidePanel
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-side-panel';
import {createWorkbenchStore} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {makeFixtureHost} from '../workflow-fixtures';

function makeProfile(rowSemantics: 'events' | 'surface-samples'): KeplerDatasetProfile {
  return profileKeplerFeatureCollection(
    {
      type: 'FeatureCollection',
      features: Array.from({length: 12}, (_, index) => ({
        type: 'Feature',
        geometry: {type: 'Point', coordinates: [index % 4, Math.floor(index / 4)]},
        properties: {rainfall: index + 1}
      }))
    },
    {
      datasetId: 'observations',
      label: 'Rain gauges',
      coordinateSystem: 'planar',
      declarations: {rowSemantics}
    }
  );
}

// External Chromium driver: the implementor sandbox cannot start a browser.
test('mapping disables Run for invalid numeric drafts until every displayed setting is valid', async () => {
  const analysis = getKeplerCuratedAnalysis('density-surface')!;
  const profiles = [makeProfile('events')];
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles);
  let parameters: Record<string, CatalogParameterValue> = {};
  let runs = 0;
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: true});
  function render() {
    root.render(
      createElement(WorkbenchAnalysisMapping, {
        analysis,
        mapping,
        profiles,
        onChange() {},
        onRun() {
          runs++;
        },
        onCancel() {},
        onParametersChange(next) {
          parameters = next;
          mapping = replayKeplerCuratedAnalysisMapping(analysis, mapping.choices, profiles, {
            parameters
          });
          render();
        }
      })
    );
  }
  const run = () => container.querySelector<HTMLButtonElement>('[data-testid="ax-mapping-run"]')!;
  async function typeSetting(name: string, value: string) {
    const input = container.querySelector<HTMLInputElement>(
      `[data-testid="ax-mapping-parameter-${name}"]`
    );
    if (!input) throw new Error(`Missing ${name}: ${container.innerHTML}`);
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(input, value);
      input.dispatchEvent(new Event('input', {bubbles: true}));
    });
  }
  function requireDisabled(disabled: boolean, stage: string) {
    if (run()?.disabled !== disabled)
      throw new Error(
        `${stage}: expected Run disabled=${disabled}; observed ${container.innerHTML}`
      );
  }
  try {
    await act(async () => render());
    requireDisabled(false, 'Initial valid mapping');
    await typeSetting('cellSize', 'abc');
    requireDisabled(true, 'Invalid cell size');
    if (parameters['cellSize'] !== undefined)
      throw new Error('Invalid draft changed the committed cell size');
    await act(async () => run().click());
    if (runs !== 0) throw new Error('Disabled Run executed an invalid draft');
    await typeSetting('bandwidth', '-1');
    await typeSetting('cellSize', '1');
    requireDisabled(true, 'Bandwidth remains invalid after cell size correction');
    await typeSetting('bandwidth', '2');
    requireDisabled(false, 'All numeric drafts corrected');
    await act(async () => run().click());
    if (Number(runs) !== 1) throw new Error(`Corrected mapping ran ${runs} times; expected once`);
  } finally {
    await act(async () => root.unmount());
    container.remove();
    Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: false});
  }
});

test('side-panel suggestions use declared surface samples and expose AX readiness/remedies', async () => {
  const profile = makeProfile('surface-samples');
  const dataset = {
    id: profile.datasetId,
    label: profile.label,
    rowCount: profile.rowCount,
    resident: false,
    fields: [],
    geometryFields: ['geometry']
  };
  const host = makeFixtureHost([dataset]);
  host.getAnalysisProfile = () => profile;
  const store = createWorkbenchStore({host, registry: getWorkflowOperationRegistry()});
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: true});
  try {
    await act(async () => root.render(createElement(WorkbenchSidePanel, {store})));
    const details = container.querySelector<HTMLDetailsElement>(
      'details[aria-label="Suggested analyses for Rain gauges"]'
    );
    if (!details) throw new Error(`Missing dataset suggestions: ${container.innerHTML}`);
    await act(async () => {
      details.open = true;
      details.dispatchEvent(new Event('toggle'));
    });
    const expected = getSidePanelRecommendations(store, dataset.id);
    if (!expected.length) throw new Error('Surface sample fixture produced no AX suggestions');
    for (const suggestion of expected) {
      const button = [...details.querySelectorAll<HTMLButtonElement>('button')].find(
        candidate => candidate.textContent === suggestion.title
      );
      if (
        !button ||
        button.dataset['status'] !== suggestion.status ||
        button.disabled !== (suggestion.status === 'blocked')
      )
        throw new Error(`Suggestion readiness differs for ${suggestion.id}: ${details.innerHTML}`);
      for (const missing of suggestion.applicability.missing) {
        if (
          !details.textContent?.includes(missing.what) ||
          (missing.remedy && !details.textContent?.includes(missing.remedy))
        )
          throw new Error(`Missing AX remedy for ${suggestion.id}: ${details.innerHTML}`);
      }
      if (
        suggestion.status !== 'blocked' &&
        ['kernel-density-grid', 'recipe-hot-spot-lattice'].includes(suggestion.bindingId)
      )
        throw new Error(`Event analysis suggested for declared rain gauges: ${suggestion.id}`);
    }
  } finally {
    await act(async () => root.unmount());
    await store.clear();
    container.remove();
    Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: false});
  }
});
