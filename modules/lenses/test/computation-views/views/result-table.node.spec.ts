// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/kriging.lens';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/geometry-validity.lens';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/segment-intersection.lens';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-predicate-join.lens';
import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import {getLensScenario} from '../../../src/computation-lenses/registry-implementation';
import {buildLensTwin} from '../../../src/computation-views/twin/static-twin';
import {LensController} from '../../../src/computation-views/hosts/lens-controller';
import {ViewRegistry} from '../../../src/computation-views/registry/view-registry';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {findViewNodes, getViewNodeText} from '../../../src/computation-views/view-node';
import {
  buildResultViewModel,
  renderResultView,
  RESULT_VIEW
} from '../../../src/computation-views/views/result-view';
import {makeY2Capture, makeY2Points, makeY2Record} from '../fixtures/y2-card-records';

function render(record: ReturnType<typeof makeY2Record>, options = {}) {
  const context = createViewContext(record, {options});
  const model = buildResultViewModel(context);
  const tree = renderResultView(model, context);
  return {context, model, tree};
}

describe('result table row domains', () => {
  test.each([
    {
      scenario: 'geometry-validity',
      capture: 'resultParks',
      geometry: 'polygons',
      mark: 'path',
      values: undefined,
      diagnostics: ['intersectionCount', 'overflow']
    },
    {
      scenario: 'segment-intersection',
      capture: 'intersections',
      geometry: 'points',
      mark: 'circle',
      values: 'kinds',
      diagnostics: ['uncertainCount', 'overflow']
    },
    {
      scenario: 'spatial-predicate-join',
      capture: 'joinedTracts',
      geometry: 'polygons',
      mark: 'path',
      values: 'tractMatches',
      diagnostics: ['pairCount', 'pairOverflow']
    }
  ])('$scenario selects its promised geometry in full and thumbnail results and retains diagnostics', ({
    scenario,
    capture,
    geometry,
    mark,
    values,
    diagnostics
  }) => {
    const recorded = fromRecordedLensFile(
      JSON.parse(
        readFileSync(
          new URL(
            `../../../../../docs/lens-records/gpu-spatial-analysis.${scenario}.json`,
            import.meta.url
          ),
          'utf8'
        )
      )
    );
    const presentation = getLensScenario(recorded.scenarioId)!.presentation;
    for (const size of ['full', 'thumbnail'] as const) {
      const {model, tree} = render(recorded.record, {...presentation, size});
      expect(model.output.capture.name).toBe(capture);
      expect(model.output.geometry?.kind).toBe(geometry);
      if (values) expect(model.output.values?.name).toBe(values);
      expect(findViewNodes(tree, node => node.tag === 'svg').length).toBeGreaterThan(0);
      const outputGroups = findViewNodes(
        tree,
        node => node.tag === 'g' && node.attributes['data-role'] === 'output'
      );
      expect(
        outputGroups.flatMap(group => findViewNodes(group, node => node.tag === mark)).length
      ).toBeGreaterThan(0);
      const twin = buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'result'}, {size});
      expect(twin.disabled).toBeUndefined();
      // Alt text is length-limited; the full subject retains the output after a long input list.
      expect(twin.llms).toContain(`→ ${capture}`);
      expect(twin.keyNumber.label).toBe(geometry);
      if (scenario === 'segment-intersection')
        expect(twin.keyNumber).toEqual({label: 'points', value: '1,802'});
      expect(twin.figure.html).toContain('<svg');
      if (size === 'full') {
        expect(twin.figure!.html).toContain('<svg');
        for (const diagnostic of diagnostics) {
          const tables = findViewNodes(tree, node => node.tag === 'table');
          expect(tables.some(table => getViewNodeText(table).includes(diagnostic))).toBe(true);
        }
      }
    }
  });
  test('committed kriging twins retain rainfall and separate row domains', () => {
    const twinsDirectory = new URL('../../../../../docs/lens-records/twins/', import.meta.url);
    for (const suffix of ['~result.json', '~result~thumbnail.json']) {
      const twin = JSON.parse(
        readFileSync(new URL(`gpu-spatial-analysis.kriging${suffix}`, twinsDirectory), 'utf8')
      );
      expect(twin.keyNumber).toEqual({label: 'cells', value: '4,662'});
      expect(twin.description).toContain('→ rainfall');
      if (suffix === '~result.json') {
        const tables = [
          ...String(twin.figure.html).matchAll(/<table\b([^>]*)>([\s\S]*?)<\/table>/g)
        ];
        expect(tables.map(table => table[1].match(/data-key-space="([^"]+)"/)?.[1])).toEqual([
          'station',
          'cell',
          'run'
        ]);
        expect(
          tables.map(table =>
            [...table[2].matchAll(/<th\b[^>]*>([^<]*)<\/th>/g)].map(header => header[1])
          )
        ).toEqual([['precipitation'], ['variance'], ['invalidCount', 'uncertainCount']]);
        const diagnostics = tables[2][2].match(/<tbody>([\s\S]*?)<\/tbody>/)![1];
        expect([...diagnostics.matchAll(/<tr\b/g)]).toHaveLength(1);
      }
    }
  });
  test('keeps scalar diagnostics below two spatial panels without allocating a blank panel', () => {
    const record = makeY2Points([1, 2, 3]);
    record.captures = {
      ...record.captures!,
      snapshots: [
        ...record.captures!.snapshots,
        makeY2Capture('invalidCount', [0], {role: 'output', keySpace: 'run'})
      ]
    };
    const {model, tree} = render(record, {primaryOutputCaptureId: 'output-id'});
    expect(model.layout).toBe('side-by-side');
    const panels = findViewNodes(tree, node => node.attributes['class'] === 'lens-result-panel');
    expect(panels).toHaveLength(2);
    expect(model.width).toBe(312);
    const tables = findViewNodes(tree, node => node.tag === 'table');
    expect(tables).toHaveLength(1);
    expect(tables[0].attributes['data-key-space']).toBe('run');
  });
  test('rejects a primary ID that identifies an input or is absent', () => {
    const record = makeY2Record([
      makeY2Capture('input', [1], {role: 'input'}),
      makeY2Capture('output', [2], {role: 'output'})
    ]);
    for (const primaryOutputCaptureId of ['input-id', 'missing-id', 'output']) {
      const context = createViewContext(record, {options: {primaryOutputCaptureId}});
      expect(RESULT_VIEW.satisfies(context)).toEqual({
        ok: false,
        reason: 'primaryOutputCaptureId does not identify a final output capture'
      });
      expect(() => buildResultViewModel(context)).toThrow('primaryOutputCaptureId');
    }
  });
  test('separates station, cell and run columns instead of joining their first rows', () => {
    const record = makeY2Record([
      makeY2Capture('precipitation', [56.6, 78], {role: 'input', keySpace: 'station'}),
      makeY2Capture('variance', [13500, 13000, 12300], {role: 'output', keySpace: 'cell'}),
      makeY2Capture('invalidCount', [18], {role: 'output', keySpace: 'run'}),
      makeY2Capture('uncertainCount', [0], {role: 'output', keySpace: 'run'})
    ]);
    const tables = findViewNodes(render(record).tree, node => node.tag === 'table');
    expect(
      tables.map(table => findViewNodes(table, node => node.tag === 'th').map(getViewNodeText))
    ).toEqual([['precipitation'], ['variance'], ['invalidCount', 'uncertainCount']]);
    expect(findViewNodes(tables[2], node => node.tag === 'tbody')[0].children).toHaveLength(1);
    const thumbnail = render(record, {size: 'thumbnail'});
    const cells = findViewNodes(thumbnail.tree, node => node.tag === 'rect');
    // Only two whole-run diagnostics, not a three-row station/cell/run grid.
    expect(cells).toHaveLength(2);
  });

  test('counts vector entities and renders every component in one row', () => {
    const record = makeY2Record([
      makeY2Capture('input', [1, 2, 3, 4], {role: 'input', keySpace: 'point', components: 2}),
      makeY2Capture('output', [5, 6, 7, 8], {role: 'output', keySpace: 'point', components: 2})
    ]);
    const {context, model, tree} = render(record);
    expect(RESULT_VIEW.describe(model, context).keyNumber).toEqual({label: 'rows', value: 2});
    expect(findViewNodes(tree, node => node.tag === 'table')).toHaveLength(2);
    const outputTable = findViewNodes(tree, node => node.tag === 'table').find(table =>
      findViewNodes(table, node => node.tag === 'th').some(
        header => getViewNodeText(header) === 'output'
      )
    )!;
    const rows = findViewNodes(outputTable, node => node.tag === 'tbody')[0];
    expect(rows.children).toHaveLength(2);
    expect(getViewNodeText(rows)).toContain('5, 6');
    expect(getViewNodeText(rows)).toContain('7, 8');
  });

  test('column meanings without key spaces do not establish a row association', () => {
    const record = makeY2Record([
      makeY2Capture('left', [1, 2], {role: 'input', meaning: 'column value'}),
      makeY2Capture('right', [3, 4], {role: 'output', meaning: 'column value'})
    ]);
    const tables = findViewNodes(render(record).tree, node => node.tag === 'table');
    expect(tables).toHaveLength(2);
    expect(
      tables.every(table => findViewNodes(table, node => node.tag === 'th').length === 1)
    ).toBe(true);
  });

  test('selects the committed kriging rainfall capture ahead of trailing diagnostics', () => {
    const recorded = fromRecordedLensFile(
      JSON.parse(
        readFileSync(
          new URL(
            '../../../../../docs/lens-records/gpu-spatial-analysis.kriging.json',
            import.meta.url
          ),
          'utf8'
        )
      )
    );
    const rainfall = recorded.record.captures!.snapshots.find(
      capture => capture.name === 'rainfall'
    )!;
    const {context, model, tree} = render(recorded.record, {primaryOutputCaptureId: rainfall.id});
    expect(model.output.capture.name).toBe('rainfall');
    expect(RESULT_VIEW.describe(model, context).keyNumber).toEqual({label: 'cells', value: 4662});
    const tables = findViewNodes(tree, node => node.tag === 'table');
    expect(
      tables.map(table => findViewNodes(table, node => node.tag === 'th').map(getViewNodeText))
    ).toEqual([['precipitation'], ['variance'], ['invalidCount', 'uncertainCount']]);
    const presentation = getLensScenario(recorded.scenarioId)!.presentation!;
    expect(presentation.primaryOutputCaptureId).toBe(rainfall.id);
    for (const size of ['full', 'thumbnail'] as const) {
      const twin = buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'result'}, {size});
      expect(twin.disabled).toBeUndefined();
      expect(twin.keyNumber).toEqual({label: 'cells', value: '4,662'});
      expect(twin.description).toContain('→ rainfall');
    }
    const registry = new ViewRegistry();
    registry.register(RESULT_VIEW);
    const controller = new LensController({
      record: recorded.record,
      registry,
      lensId: 'kriging-review',
      rung: 'result',
      presentation
    });
    try {
      expect(controller.getFrame().description.alt).toContain('→ rainfall');
      expect(controller.getFrame().description.alt).toContain('cells: 4,662');
      controller.dispatch({
        type: 'apply-url-state',
        state: {...controller.getState(), params: {}}
      });
      expect(controller.getFrame().description.alt).toContain('→ rainfall');
    } finally {
      controller.destroy();
    }
  });

  test('every archived result table rerenders with one declared row domain per table', () => {
    const recordsDirectory = new URL('../../../../../docs/lens-records/', import.meta.url);
    const twinsDirectory = new URL('twins/', recordsDirectory);
    let tableCount = 0;
    for (const file of readdirSync(twinsDirectory).filter(name => name.endsWith('~result.json'))) {
      const twin = JSON.parse(readFileSync(new URL(file, twinsDirectory), 'utf8'));
      if (twin.viewId !== 'result-view' || !twin.figure.html.includes('cv-table')) continue;
      const recordFile = new URL(`${twin.scenarioId}.json`, recordsDirectory);
      if (!existsSync(recordFile)) continue;
      const record = fromRecordedLensFile(JSON.parse(readFileSync(recordFile, 'utf8'))).record;
      const {tree} = render(record, getLensScenario(twin.scenarioId)?.presentation);
      for (const table of findViewNodes(tree, node => node.tag === 'table')) {
        const names = findViewNodes(table, node => node.tag === 'th').map(getViewNodeText);
        const captures = names.map(
          name => record.captures!.snapshots.find(capture => capture.name === name)!
        );
        expect(new Set(captures.map(capture => capture.keySpace)).size, file).toBe(1);
        if (!captures[0].keySpace) expect(names, file).toHaveLength(1);
        expect(
          findViewNodes(table, node => node.tag === 'caption'),
          file
        ).toHaveLength(1);
        tableCount++;
      }
    }
    expect(tableCount).toBeGreaterThan(0);
  });
});
