// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import recorded from '../../../../docs/lens-records/gpu-spatial-analysis.local-moran.json';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {mountLadderLens} from '../../src/computation-views/hosts/mount-lens';
import {ViewRegistry} from '../../src/computation-views/registry/view-registry';
import {RESULT_VIEW} from '../../src/computation-views/views/result-view';
import {createLocalMoranPickingRecord} from './polygon-picking.x1.test-utils';

test('X9 clicking a live-produced Local Moran polygon selects its tract', () => {
  const element = document.createElement('div');
  document.body.append(element);
  const registry = new ViewRegistry();
  registry.register(RESULT_VIEW);
  const mounted = mountLadderLens(element, {
    record: createLocalMoranPickingRecord(fromRecordedLensFile(recorded).record),
    registry,
    bind: 'x1-moran-picking',
    rung: 'result',
    urlState: false
  });
  try {
    const paths = element.querySelectorAll<SVGPathElement>(
      'path[data-action="pick"][data-key-space="tract"]'
    );
    expect(paths).toHaveLength(132);
    const path = paths[1];
    path.dispatchEvent(new MouseEvent('click', {bubbles: true}));
    expect(mounted.controller.getState().clauses).toMatchObject([
      {keySpace: 'tract', predicate: {keys: [Number(path.dataset['key'])]}}
    ]);
  } finally {
    mounted.dispose();
    element.remove();
  }
});
