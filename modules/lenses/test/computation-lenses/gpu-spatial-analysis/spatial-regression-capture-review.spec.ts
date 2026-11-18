// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-regression-recipe.lens';
import {getLensScenario} from '../../../src/computation-lenses/registry-implementation';
import {getCaptureRowCount} from '../../../src/computation-views/views/record-queries';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {findViewNodes, getViewNodeText} from '../../../src/computation-views/view-node';

test('R1: captured regression predictors retain their two-component tract layout', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenario = await getLensScenario('gpu-spatial-analysis.spatial-regression-recipe')!.load();
  const record = await scenario.run(device);
  const predictors = record.captures!.snapshots.find(capture => capture.name === 'predictors')!;
  expect(predictors.components).toBe(2);
  expect(getCaptureRowCount(predictors)).toBe(199);
  const context = createViewContext(record);
  const tree = renderResultView(buildResultViewModel(context), context);
  const table = findViewNodes(
    tree,
    node => node.tag === 'table' && node.attributes['data-key-space'] === 'tracts'
  )[0];
  const headers = findViewNodes(table, node => node.tag === 'th').map(getViewNodeText);
  const predictorColumn = headers.indexOf('predictors');
  expect(getViewNodeText(findViewNodes(table, node => node.tag === 'caption')[0])).toContain(
    '199 rows'
  );
  const rows = findViewNodes(table, node => node.tag === 'tbody')[0];
  expect(
    findViewNodes(rows, node => node.tag === 'tr')
      .slice(0, 2)
      .map(row => getViewNodeText(findViewNodes(row, node => node.tag === 'td')[predictorColumn]))
  ).toEqual(['7.41, 8.85', '6.31, 9.28']);
});
