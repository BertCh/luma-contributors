// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {act, createElement} from 'react';
import {createRoot} from 'react-dom/client';
import {expect, test} from 'vitest';
import {AnalysisResultCard} from '../../../../examples/deck/kepler-compute/src/analysis-result-card';
import {createReviewCard} from '../ax-result-card-fixtures';
import {
  encodeKeplerResultPresentation,
  type KeplerResultInferenceUpdate
} from '../../src/presentation/index';

// External Chromium handoff. This tests the exact map publication encoding, not a rendered Kepler
// map. The integration driver must separately qualify actual layer replacement in the host.
test.each([
  'cluster-and-outlier-map',
  'hot-spot-map'
] as const)('inference card and publication encoding stay consistent without rerunning %s', async identifier => {
  const {model} = await createReviewCard(identifier);
  const originalJson = model.json;
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  let accepted: KeplerResultInferenceUpdate | undefined;
  let rejectDisplay = false;
  let rerunCount = 0;
  Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: true});
  try {
    const createCard = (currentModel = model) =>
      createElement(AnalysisResultCard, {
        model: currentModel,
        onNext: () => {},
        onInferenceChange: update => {
          if (rejectDisplay) throw new Error('Map replacement refused; old display retained.');
          // This is the encoder used by native map publication, with the exact accepted plan.
          for (const presentation of update.presentations) {
            const table = update.displayResult.tables.find(
              candidate => candidate.id === presentation.tableId
            )!;
            const encoding = encodeKeplerResultPresentation(table, presentation);
            if (!encoding)
              throw new Error(
                `Missing publication encoding: ${JSON.stringify({identifier, table: table.id, presentation})}`
              );
            for (const entry of encoding.legend) {
              const count = encoding.values.filter(value => value === entry.label).length;
              if (count !== entry.count)
                throw new Error(
                  `Map/legend mismatch: ${JSON.stringify({identifier, entry, count, settings: update.settings})}`
                );
            }
          }
          accepted = update;
        },
        onInferenceRerun: () => {
          rerunCount++;
        }
      });
    await act(async () => root.render(createCard()));
    const select = container.querySelector<HTMLSelectElement>(
      '[aria-label="Significance cut-off"]'
    )!;
    await act(async () => {
      select.value = '0.01';
      select.dispatchEvent(new Event('change', {bubbles: true}));
    });
    const button = [...container.querySelectorAll('button')].find(
      candidate => candidate.textContent === 'Apply to display'
    );
    if (!button || button.disabled)
      throw new Error(`Inference unavailable: ${identifier}: ${container.innerHTML}`);
    await act(async () => button.click());
    if (!accepted)
      throw new Error(`Display was not accepted: ${identifier}: ${container.innerHTML}`);
    expect(container.querySelector('[data-testid="ax-result-sentence"]')?.textContent).toBe(
      accepted.report.sentence
    );
    expect(container.textContent).toContain('Display cut-off: 0.01');
    expect(rerunCount).toBe(0);
    expect(model.json).toBe(originalJson);
    const acceptedSentence = accepted.report.sentence;
    const acceptedLegends = [...container.querySelectorAll('[aria-label$=" legend"]')].map(
      legend => legend.textContent
    );
    await act(async () => root.render(createCard({...model, result: {...model.result}})));
    expect(container.querySelector('[data-testid="ax-result-sentence"]')?.textContent).toBe(
      acceptedSentence
    );
    expect(
      [...container.querySelectorAll('[aria-label$=" legend"]')].map(legend => legend.textContent)
    ).toEqual(acceptedLegends);
    expect(
      container.querySelector<HTMLSelectElement>('[aria-label="Significance cut-off"]')?.value
    ).toBe('0.01');
    expect(rerunCount).toBe(0);
    rejectDisplay = true;
    await act(async () => {
      select.value = '0.1';
      select.dispatchEvent(new Event('change', {bubbles: true}));
    });
    await act(async () => button.click());
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      'Map replacement refused'
    );
    expect(container.querySelector('[data-testid="ax-result-sentence"]')?.textContent).toBe(
      acceptedSentence
    );
    expect(rerunCount).toBe(0);
  } finally {
    await act(async () => root.unmount());
    container.remove();
    Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: false});
  }
}, 30000);

test.each([
  ['cluster-and-outlier-map', 'sample'],
  ['cluster-and-outlier-map', 'viewport'],
  ['contiguity-correlogram', 'sample'],
  ['contiguity-correlogram', 'viewport']
] as const)(
  '%s %s inference explicitly requires a full-population rerun',
  async (identifier, scope) => {
    const {model} = await createReviewCard(identifier);
    const container = document.createElement('div');
    document.body.append(container);
    const root = createRoot(container);
    let rerunCount = 0;
    let displayCount = 0;
    let keepCount = 0;
    Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: true});
    try {
      await act(async () =>
        root.render(
          createElement(AnalysisResultCard, {
            model,
            onNext: () => {},
            inferenceScope: scope,
            onAddToStack: () => {
              keepCount++;
            },
            onSaveWorkflow: () => {
              keepCount++;
            },
            onInferenceChange: () => {
              displayCount++;
            },
            onInferenceRerun: () => {
              rerunCount++;
            }
          })
        )
      );
      expect(container.textContent).toContain('Rerun required: Recompute on the full population');
      for (const label of ['Add to stack', 'Save as workflow']) {
        const keep = [...container.querySelectorAll('button')].find(
          candidate => candidate.textContent === label
        );
        if (!keep || !keep.disabled)
          throw new Error(
            `Partial inference can be kept: ${JSON.stringify({identifier, scope, label, markup: container.innerHTML})}`
          );
        await act(async () => keep.click());
      }
      expect(keepCount).toBe(0);
      const button = [...container.querySelectorAll('button')].find(
        candidate => candidate.textContent === 'Rerun analysis'
      );
      if (!button) throw new Error(`Missing explicit rerun: ${container.innerHTML}`);
      await act(async () => button.click());
      expect(rerunCount).toBe(1);
      expect(displayCount).toBe(0);
      expect(container.querySelector('[data-testid="ax-result-sentence"]')?.textContent).toContain(
        `Significance is hidden for this ${scope} preview.`
      );
      expect(container.querySelector('[aria-label$=" legend"]')).toBeNull();
    } finally {
      await act(async () => root.unmount());
      container.remove();
      Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: false});
    }
  },
  30000
);
