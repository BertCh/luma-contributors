// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import type {Device} from '@luma.gl/core';
import type {
  S2CaptureSpec,
  S2LensRunProps,
  S2Uploads
} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import {expect, test, vi} from 'vitest';
import {getLensScenario} from '../../../src/computation-lenses/registry-implementation';
import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import {getCaptureRowCount} from '../../../src/computation-views/views/record-queries';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {findViewNodes, getViewNodeText} from '../../../src/computation-views/view-node';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-regression-recipe.lens';

const captured = vi.hoisted(() => ({
  specifications: undefined as Record<string, S2CaptureSpec> | undefined
}));
vi.mock('../../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run', () => ({
  createS2ChicagoFrame: () => ({}),
  uploadS2JoinPolygons: () => ({keySpaces: [], captures: {tracts: {}}, geometry: {}, display: {}}),
  runS2Lens: async (_device: unknown, props: Pick<S2LensRunProps, 'build'>) => {
    captured.specifications = await props.build(
      {} as Parameters<S2LensRunProps['build']>[0],
      {
        view: (_name: string, values: unknown) => ({values}),
        own: (value: unknown) => value
      } as unknown as S2Uploads
    );
    return {};
  }
}));
vi.mock(
  '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-regression-recipe.example',
  () => ({
    runExample: async () => ({
      parameters: {},
      residuals: {},
      fitted: {},
      moranZScores: {},
      coefficients: {},
      tests: {},
      decision: {}
    })
  })
);

test('R1: the scenario declares two predictor components per tract', async () => {
  const scenario = await getLensScenario('gpu-spatial-analysis.spatial-regression-recipe')!.load();
  await scenario.run({} as Device);
  expect(captured.specifications!['predictors'].components).toBe(2);
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../../docs/lens-records/gpu-spatial-analysis.spatial-regression-recipe.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const predictors = record.captures!.snapshots.find(capture => capture.name === 'predictors')!;
  const corrected = {...predictors, components: captured.specifications!['predictors'].components};
  expect(getCaptureRowCount(corrected)).toBe(199);
  expect(
    Array.from(corrected.values)
      .slice(0, 4)
      .map(value => Number(value.toPrecision(3)))
  ).toEqual([7.41, 8.85, 6.31, 9.28]);
  const correctedRecord = {
    ...record,
    captures: {
      ...record.captures!,
      snapshots: record.captures!.snapshots.map(capture =>
        capture === predictors ? corrected : capture
      )
    }
  };
  const context = createViewContext(correctedRecord);
  const tree = renderResultView(buildResultViewModel(context), context);
  const table = findViewNodes(
    tree,
    node => node.tag === 'table' && node.attributes['data-key-space'] === 'tracts'
  )[0];
  const headers = findViewNodes(table, node => node.tag === 'th').map(getViewNodeText);
  const predictorColumn = headers.indexOf('predictors');
  const rows = findViewNodes(table, node => node.tag === 'tbody')[0];
  expect(getViewNodeText(findViewNodes(table, node => node.tag === 'caption')[0])).toContain(
    '199 rows'
  );
  expect(
    findViewNodes(rows, node => node.tag === 'tr')
      .slice(0, 2)
      .map(row => getViewNodeText(findViewNodes(row, node => node.tag === 'td')[predictorColumn]))
  ).toEqual(['7.41, 8.85', '6.31, 9.28']);
});

test('R1: stale captures exceeding the declared domain withhold values instead of assigning predictors to other tracts', () => {
  const record = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../../docs/lens-records/gpu-spatial-analysis.spatial-regression-recipe.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const predictors = record.captures!.snapshots.find(capture => capture.name === 'predictors')!;
  record.captures = {
    ...record.captures!,
    snapshots: record.captures!.snapshots.map(capture =>
      capture === predictors ? {...capture, components: 1} : capture
    )
  };
  const context = createViewContext(record);
  const tree = renderResultView(buildResultViewModel(context), context);
  const table = findViewNodes(
    tree,
    node => node.tag === 'table' && node.attributes['data-key-space'] === 'tracts'
  )[0];
  expect(getViewNodeText(findViewNodes(table, node => node.tag === 'caption')[0])).toContain(
    '199 rows'
  );
  expect(getViewNodeText(tree)).toContain('capture row range exceeds key space');
  const predictorColumn = findViewNodes(table, node => node.tag === 'th')
    .map(getViewNodeText)
    .indexOf('predictors');
  const body = findViewNodes(table, node => node.tag === 'tbody')[0];
  expect(
    findViewNodes(body, node => node.tag === 'tr')
      .slice(0, 6)
      .map(row => getViewNodeText(findViewNodes(row, node => node.tag === 'td')[predictorColumn]))
  ).toEqual(Array(6).fill('—'));
});
