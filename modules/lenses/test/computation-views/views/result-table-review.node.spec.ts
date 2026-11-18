// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  buildResultViewModel,
  renderResultView
} from '../../../src/computation-views/views/result-view';
import {
  findViewNodes,
  getViewNodeText,
  viewNodeToHtml
} from '../../../src/computation-views/view-node';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';
import {createDefaultViewRegistry} from '../../../src/computation-views/registry/default-registry';
import '../../../src/computation-views/packs/gpu-core/index';

function createArrayRegistry() {
  return createDefaultViewRegistry();
}

function render(snapshots: Parameters<typeof makeY2Record>[0], options = {}) {
  const context = createViewContext(makeY2Record(snapshots), {options});
  const model = buildResultViewModel(context);
  return {model, tree: renderResultView(model, context)};
}

describe('result table review regressions', () => {
  test('R3: preferred array tables separate same-role columns from different domains', () => {
    const context = createViewContext(
      makeY2Record([
        makeY2Capture('variance', [13500, 13000], {
          role: 'output',
          keySpace: 'cell',
          meaning: 'column:variance'
        }),
        makeY2Capture('invalidCount', [18], {
          role: 'output',
          keySpace: 'run',
          meaning: 'column:invalidCount'
        })
      ])
    );
    const resolution = createArrayRegistry().resolveRung(context, 'result');
    expect(resolution.status).toBe('enabled');
    if (resolution.status !== 'enabled') throw new Error('Expected array result');
    expect(resolution.view.id).toBe('gpu-core-array-result');
    const tree = resolution.view.render(resolution.view.build(context), context);
    const tables = findViewNodes(tree, node => node.attributes['role'] === 'table');
    expect(tables).toHaveLength(2);
    expect(tables.map(table => table.attributes['data-key-space'])).toEqual(['cell', 'run']);
    for (const capture of context.record.captures!.snapshots) capture.keySpace = undefined;
    const unkeyed = resolution.view.render(resolution.view.build(context), context);
    expect(findViewNodes(unkeyed, node => node.attributes['role'] === 'table')).toHaveLength(2);
  });

  test('R4: the preferred array view defers explicit primary IDs to the generic result view', () => {
    const record = makeY2Record([
      makeY2Capture('rainfall', [5, 6], {role: 'output', keySpace: 'cell'}),
      makeY2Capture('invalidCount', [0], {role: 'output', keySpace: 'run', meaning: 'count'})
    ]);
    const registry = createArrayRegistry();
    const context = createViewContext(record, {
      options: {primaryOutputCaptureId: 'invalidCount-id'}
    });
    const resolution = registry.resolveRung(context, 'result');
    expect(resolution.status).toBe('enabled');
    if (resolution.status !== 'enabled') throw new Error('Expected enabled result');
    expect(resolution.view.id).toBe('result-view');
    expect(resolution.view.describe(resolution.view.build(context), context).keyNumber).toEqual({
      label: 'rows',
      value: 1
    });
    expect(
      registry.resolveRung(
        createViewContext(record, {options: {primaryOutputCaptureId: 'missing'}}),
        'result'
      ).status
    ).toBe('disabled');
  });

  test('R5: first:1 columns align with entity keys rather than array positions', () => {
    const {tree} = render([
      makeY2Capture('left', [5, 6, 7], {role: 'input', keySpace: 'rows'}),
      makeY2Capture('right', [20, 30], {role: 'output', keySpace: 'rows', meaning: 'first:1'})
    ]);
    const rows = findViewNodes(tree, node => node.tag === 'tr').slice(1);
    expect(
      rows.map(row => findViewNodes(row, node => node.tag === 'td').map(getViewNodeText))
    ).toEqual([
      ['5', '—'],
      ['6', '20'],
      ['7', '30']
    ]);
  });

  test('R6: unkeyed vector strips show two complete entities without missing cells', () => {
    const {model, tree} = render([
      makeY2Capture('vector', [1, 2, 3, 4], {role: 'output', components: 2})
    ]);
    expect(model.grammar).toBe('strips');
    expect(
      findViewNodes(tree, node => node.tag === 'rect' && node.attributes['data-role'] === 'output')
    ).toHaveLength(2);
    expect(
      findViewNodes(tree, node => node.attributes['class'] === 'cv-cell-text').map(getViewNodeText)
    ).toEqual(['1, 2', '3, 4']);
    expect(
      createArrayRegistry().resolveRung(
        createViewContext(
          makeY2Record([makeY2Capture('vector', [1, 2, 3, 4], {role: 'output', components: 2})])
        ),
        'result'
      )
    ).toMatchObject({status: 'enabled', view: {id: 'result-view'}});
  });

  test('R6: vector-strip thumbnails bound the visible panel count', () => {
    const {tree} = render(
      Array.from({length: 10}, (_, index) =>
        makeY2Capture(
          `vector-${index}`,
          Array.from({length: 16}, (_, value) => value),
          {role: index === 0 ? 'input' : 'output', components: 2}
        )
      ),
      {size: 'thumbnail'}
    );
    expect(
      findViewNodes(
        tree,
        node => node.tag === 'rect' && node.attributes['data-provenance'] === 'readback'
      ).length
    ).toBeLessThanOrEqual(32);
  });

  test('R7: table thumbnails never mark uncaptured output rows as readbacks', () => {
    const {tree} = render(
      [
        makeY2Capture('input', [1, 2, 3], {role: 'input', keySpace: 'rows'}),
        makeY2Capture('output', [], {role: 'output', keySpace: 'rows'})
      ],
      {size: 'thumbnail'}
    );
    expect(
      findViewNodes(tree, node => node.tag === 'rect' && node.attributes['data-role'] === 'input')
    ).toHaveLength(3);
    expect(
      findViewNodes(
        tree,
        node =>
          node.tag === 'rect' &&
          node.attributes['data-role'] === 'output' &&
          node.attributes['data-provenance'] === 'readback'
      )
    ).toHaveLength(0);
  });

  test('R8: a lone keyed output preserves two changing frames and the final visible value', () => {
    const {model, tree} = render(
      [0, 1].map(iteration =>
        makeY2Capture('output', [iteration + 1], {
          id: `output-${iteration}`,
          iteration,
          role: 'output',
          keySpace: 'rows'
        })
      )
    );
    expect(model.frames).toHaveLength(2);
    const frames = findViewNodes(tree, node => node.attributes['class'] === 'cv-frame');
    expect(frames.map(frame => frame.attributes['hidden'])).toEqual([true, false]);
    expect(getViewNodeText(frames[1])).toBe('output2');
    expect(viewNodeToHtml(tree)).toContain('data-cv-play');
    const record = makeY2Record(
      [0, 1].map(iteration =>
        makeY2Capture('output', [iteration + 1], {
          id: `output-${iteration}`,
          iteration,
          role: 'output',
          keySpace: 'rows'
        })
      )
    );
    expect(createArrayRegistry().resolveRung(createViewContext(record), 'result')).toMatchObject({
      status: 'enabled',
      view: {id: 'result-view'}
    });
  });
});
