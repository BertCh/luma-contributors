// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, test, vi} from 'vitest';
import * as ReactHooks from 'react';
import type {ReactElement} from 'react';
import {
  createKeplerCuratedAnalysisMapping,
  getKeplerCuratedAnalysis,
  profileKeplerFeatureCollection
} from '@luma-contributors/kepler';
import {WorkbenchAnalysisMapping} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-mapping';

vi.hoisted(() => vi.resetModules());
vi.mock('react', async importOriginal => {
  const original = await importOriginal<typeof import('react')>();
  return {
    ...original,
    useState: vi.fn(original.useState),
    useRef: vi.fn(original.useRef),
    useId: vi.fn(original.useId),
    useEffect: vi.fn(original.useEffect)
  };
});
afterAll(() => {
  vi.doUnmock('react');
  vi.resetModules();
});

type Element = ReactElement<Record<string, unknown>>;
function findElement(tree: unknown, predicate: (element: Element) => boolean): Element {
  if (Array.isArray(tree)) {
    for (const child of tree) {
      try {
        return findElement(child, predicate);
      } catch {
        /* Search the next child. */
      }
    }
  } else if (tree && typeof tree === 'object' && 'props' in tree) {
    const element = tree as Element;
    if (predicate(element)) return element;
    return findElement(element.props['children'], predicate);
  }
  throw new Error('Mapping control was not found');
}

test('parameter draft problems aggregate independently, survive mapping updates and clear on correction/unmount', () => {
  const analysis = getKeplerCuratedAnalysis('density-surface')!;
  const profile = profileKeplerFeatureCollection(
    {
      type: 'FeatureCollection',
      features: Array.from({length: 12}, (_, index) => ({
        type: 'Feature',
        geometry: {type: 'Point', coordinates: [index % 4, Math.floor(index / 4)]},
        properties: {}
      }))
    },
    {
      datasetId: 'events',
      label: 'Events',
      coordinateSystem: 'planar',
      declarations: {rowSemantics: 'events'}
    }
  );
  let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
  expect(mapping.ready).toBe(true);
  const state: unknown[] = [];
  const references: {current: unknown}[] = [];
  let tree: Element;
  function render() {
    let index = 0;
    let referenceIndex = 0;
    vi.mocked(ReactHooks.useId).mockReturnValue('mapping');
    vi.mocked(ReactHooks.useEffect).mockImplementation(() => {});
    vi.mocked(ReactHooks.useRef).mockImplementation(((initial: unknown) => {
      const current = referenceIndex++;
      references[current] ??= {current: initial};
      return references[current];
    }) as typeof ReactHooks.useRef);
    vi.mocked(ReactHooks.useState).mockImplementation(((initial: unknown) => {
      const current = index++;
      if (!(current in state)) state[current] = initial;
      return [
        state[current],
        (next: unknown) => {
          state[current] = typeof next === 'function' ? next(state[current]) : next;
        }
      ];
    }) as typeof ReactHooks.useState);
    try {
      tree = WorkbenchAnalysisMapping({
        analysis,
        mapping,
        profiles: [profile],
        onChange() {},
        onParametersChange() {},
        onRun() {},
        onCancel() {}
      }) as Element;
    } finally {
      vi.mocked(ReactHooks.useId).mockReset();
      vi.mocked(ReactHooks.useEffect).mockReset();
      vi.mocked(ReactHooks.useRef).mockReset();
      vi.mocked(ReactHooks.useState).mockReset();
    }
  }
  const runDisabled = () =>
    findElement(tree, element => element.props['data-testid'] === 'ax-mapping-run').props[
      'disabled'
    ];
  function report(name: string, problem: string | null) {
    const control = findElement(
      tree,
      element => element.props['testId'] === `ax-mapping-parameter-${name}`
    );
    const change = control.props['onValidityChange'] as (problem: string | null) => void;
    expect(change).toBeTypeOf('function');
    change(problem);
    render();
  }
  render();
  expect(runDisabled()).toBe(false);
  report('cellSize', 'Cell size must be a number; got abc.');
  expect(runDisabled()).toBe(true);
  mapping = {...mapping};
  render();
  expect(runDisabled()).toBe(true);
  report('bandwidth', 'Bandwidth must be positive.');
  report('cellSize', null);
  expect(runDisabled()).toBe(true);
  report('bandwidth', null);
  expect(runDisabled()).toBe(false);
});
