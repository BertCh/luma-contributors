// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {
  buildGPUCoreArrayResultModel,
  renderGPUCoreArrayResult
} from '../../../src/computation-views/packs/gpu-core/array-result-view';
import {findViewNodes, getViewNodeText} from '../../../src/computation-views/view-node';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';

test('both renderers separate role and key space, isolate unkeyed columns and preserve domain order', () => {
  const record = makeY2Record([
    makeY2Capture('stationInput', [11, 12], {role: 'input', keySpace: 'station'}),
    makeY2Capture('cellFirst', [21, 22], {role: 'output', keySpace: 'cell'}),
    makeY2Capture('run', [31], {role: 'output', keySpace: 'run'}),
    makeY2Capture('stationOutput', [41, 42], {role: 'output', keySpace: 'station'}),
    makeY2Capture('cellSecond', [51, 52], {role: 'output', keySpace: 'cell'}),
    makeY2Capture('unkeyedFirst', [61], {role: 'output'}),
    makeY2Capture('unkeyedSecond', [71], {role: 'output'})
  ]);
  for (const capture of record.captures!.snapshots) capture.meaning = `column:${capture.name}`;
  record.axes.keySpaces = [
    {id: 'station', label: 'station', size: 2},
    {id: 'cell', label: 'cell', size: 2},
    {id: 'run', label: 'run', size: 1}
  ];
  const context = createViewContext(record);
  const trees = [
    renderResultView(buildResultViewModel(context), context),
    renderGPUCoreArrayResult(buildGPUCoreArrayResultModel(context), context)
  ];
  for (const tree of trees) {
    const tables = findViewNodes(
      tree,
      node => node.tag === 'table' || node.attributes['role'] === 'table'
    );
    expect(tables.map(table => table.attributes['data-key-space'])).toEqual([
      'station',
      'cell',
      'run',
      'station',
      undefined,
      undefined
    ]);
    expect(
      tables.map(table =>
        findViewNodes(
          table,
          node => node.tag === 'th' || node.attributes['role'] === 'columnheader'
        ).map(getViewNodeText)
      )
    ).toEqual([
      ['stationInput'],
      ['cellFirst', 'cellSecond'],
      ['run'],
      ['stationOutput'],
      ['unkeyedFirst'],
      ['unkeyedSecond']
    ]);
    expect(getViewNodeText(tables[1])).toContain('21');
    expect(getViewNodeText(tables[2])).toContain('31');
    expect(getViewNodeText(tables[3])).toContain('41');
  }
});
