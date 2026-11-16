// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {act, createElement} from 'react';
import {createRoot} from 'react-dom/client';
import {expect, test} from 'vitest';
import {AnalysisResultCard} from '../../../examples/deck/kepler-compute/src/analysis-result-card';
import {CARD_IDS, CARD_SENTENCES, createReviewCard} from './ax-result-card-fixtures';

// The external driver runs this file: Chromium cannot start in the implementor sandbox.
// CPU reference runs isolate presentation and interactions from GPU numerical qualification.
test.each(CARD_IDS)('AX-41 browser card: %s', async id => {
  const {model} = await createReviewCard(id);
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  const priorClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
  const events: string[] = [];
  let clipboardText = '';
  Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: true});
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: async (text: string) => {
        clipboardText = text;
      }
    }
  });
  try {
    await act(async () =>
      root.render(
        createElement(AnalysisResultCard, {
          model,
          onAddToStack: () => {
            events.push('stack');
          },
          onSaveWorkflow: () => {
            events.push('save');
          },
          onNext: action => {
            events.push(action.bindingId);
          }
        })
      )
    );
    const card = container.querySelector('[data-testid="ax-result-card"]');
    if (!card)
      throw new Error(`${id}: result card was not rendered. Observed: ${container.innerHTML}`);
    expect(card.querySelector('[data-testid="ax-result-sentence"]')?.textContent).toBe(
      CARD_SENTENCES[id]
    );
    const clickButton = async (label: string) => {
      const button = [...card.querySelectorAll('button')].find(
        candidate => candidate.textContent === label
      );
      if (!button || button.disabled)
        throw new Error(`${id}: ${label} is unavailable. Observed: ${card.innerHTML}`);
      await act(async () => button.click());
    };
    await clickButton('Add to stack');
    await clickButton('Save as workflow');
    await clickButton('Copy as JSON');
    expect(events).toEqual(['stack', 'save']);
    expect(JSON.parse(clipboardText).result.operationId).toBe(model.result.operationId);
    expect(card.querySelector('[role="status"]')?.textContent).toContain('Copied');
    const nextActions = [...card.querySelectorAll<HTMLElement>('[data-testid="ax-result-next"]')];
    for (const action of nextActions) {
      const button = action.querySelector('button')!;
      expect(button.disabled).toBe(action.dataset['status'] === 'blocked');
      if (!button.disabled) await act(async () => button.click());
    }
    expect(events.slice(2)).toEqual(
      model.next
        .filter(action => action.applicability.status !== 'blocked')
        .map(action => action.bindingId)
    );
  } finally {
    await act(async () => root.unmount());
    container.remove();
    if (priorClipboard) Object.defineProperty(navigator, 'clipboard', priorClipboard);
    else Reflect.deleteProperty(navigator, 'clipboard');
    Object.assign(globalThis, {IS_REACT_ACT_ENVIRONMENT: false});
  }
}, 30000);
