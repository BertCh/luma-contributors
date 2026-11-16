// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, test, vi} from 'vitest';
import {Children, isValidElement, useState, type ReactElement, type ReactNode} from 'react';
import {AnalysisResultCard} from '../../../../examples/deck/kepler-compute/src/analysis-result-card';
import {AnalysisResultInferenceControls} from '../../../../examples/deck/kepler-compute/src/analysis-result-inference-controls';
import {
  createAnalysisResultCardModel,
  type AnalysisResultCardModel
} from '../../../../examples/deck/kepler-compute/src/analysis-result-card-model';
import {createReviewCard} from '../ax-result-card-fixtures';
import {
  applyKeplerResultInference,
  type KeplerResultInferenceUpdate
} from '../../src/presentation/index';
import {getCatalogBinding} from '../../src/analysis/catalog-bindings';
import {getKeplerOperationDescriptor} from '../../src/descriptors/index';

vi.hoisted(() => vi.resetModules());
vi.mock('react', async importOriginal => {
  const original = await importOriginal<typeof import('react')>();
  return {...original, useState: vi.fn(original.useState)};
});
afterAll(() => {
  vi.doUnmock('react');
  vi.resetModules();
});

type ElementProps = Record<string, unknown> & {children?: ReactNode};

function findElement(
  node: ReactNode,
  matches: (element: ReactElement<ElementProps>) => boolean
): ReactElement<ElementProps> | undefined {
  for (const child of Children.toArray(node)) {
    if (!isValidElement<ElementProps>(child)) continue;
    if (matches(child)) return child;
    const found = findElement(child.props.children, matches);
    if (found) return found;
  }
  return undefined;
}

test('an accepted display survives reconstruction of the same run and resets for a different run', async () => {
  const {model} = await createReviewCard('cluster-and-outlier-map');
  const update = applyKeplerResultInference(
    model.result,
    getCatalogBinding(model.result.operationId)!,
    getKeplerOperationDescriptor(model.result.operationId)!,
    {significance: 0.01, multipleTesting: 'benjamini-hochberg'},
    model.context
  );
  if (update.status !== 'display') throw new Error(update.reason);
  const stateValues: unknown[] = [];
  let nextStateIndex = 0;
  // Exercise the component's actual event handler and hook state without a browser or DOM.
  vi.mocked(useState).mockImplementation(((initialValue: unknown) => {
    const stateIndex = nextStateIndex++;
    if (!(stateIndex in stateValues))
      stateValues[stateIndex] = typeof initialValue === 'function' ? initialValue() : initialValue;
    return [
      stateValues[stateIndex],
      (nextValue: unknown) => {
        stateValues[stateIndex] =
          typeof nextValue === 'function' ? nextValue(stateValues[stateIndex]) : nextValue;
      }
    ];
  }) as typeof useState);
  const acceptDisplay = vi
    .fn<(update: KeplerResultInferenceUpdate) => Promise<void>>()
    .mockResolvedValue(undefined);
  const render = (currentModel: AnalysisResultCardModel) => {
    nextStateIndex = 0;
    return AnalysisResultCard({
      model: currentModel,
      onNext: () => {},
      onInferenceChange: acceptDisplay
    });
  };
  const initial = render(model);
  const controls = findElement(
    initial,
    element => element.type === AnalysisResultInferenceControls
  );
  if (!controls) throw new Error('Missing inference controls in the result card.');
  await (controls.props['onApply'] as (nextUpdate: KeplerResultInferenceUpdate) => Promise<void>)(
    update
  );
  const accepted = render(model);
  const sentence = (tree: ReactNode) =>
    findElement(tree, element => element.props['data-testid'] === 'ax-result-sentence')?.props
      .children;
  const legend = (tree: ReactNode) =>
    JSON.stringify(
      findElement(tree, element => String(element.props['aria-label']).endsWith(' legend'))
    );
  expect(sentence(accepted)).toBe(update.report.sentence);
  const reconstructed = render({...model, result: {...model.result}});
  expect(sentence(reconstructed)).toBe(update.report.sentence);
  expect(legend(reconstructed)).toBe(legend(accepted));
  expect(
    findElement(reconstructed, element => element.type === AnalysisResultInferenceControls)?.key
  ).toBe(controls.key);
  expect(acceptDisplay).toHaveBeenCalledExactlyOnceWith(update);
  const differentModel = createAnalysisResultCardModel(
    {...model.result, parameters: {...model.result.parameters, significance: 0.1}},
    model.context
  );
  expect(differentModel.json).not.toBe(model.json);
  const different = render(differentModel);
  expect(sentence(different)).toBe(differentModel.report.sentence);
  expect(
    findElement(different, element => element.type === AnalysisResultInferenceControls)?.key
  ).not.toBe(controls.key);
});
