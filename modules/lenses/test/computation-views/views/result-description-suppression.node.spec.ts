// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {buildLensTwin} from '../../../src/computation-views/twin/static-twin';
import {generateLensDescription} from '../../../src/computation-views/twin/description';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  buildResultViewModel,
  renderResultView,
  RESULT_VIEW
} from '../../../src/computation-views/views/result-view';
import {findViewNodes} from '../../../src/computation-views/view-node';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';

function makePolygonRecord(features: number[][]) {
  let vertexCount = 0;
  const offsets = [0];
  for (const feature of features) {
    vertexCount += feature.length / 2;
    offsets.push(vertexCount);
  }
  const record = makeY2Record([
    makeY2Capture('geometry', features.flat(), {
      role: 'output',
      components: 2,
      geometry: {
        kind: 'polygons',
        coordinateModel: 'projected',
        offsetsCapture: 'offsets-id',
        featureKeySpace: 'row'
      }
    }),
    makeY2Capture('offsets', offsets)
  ]);
  record.axes.axes[0].keySpace = 'row';
  record.axes.keySpaces = [{id: 'row', label: 'feature row', size: features.length}];
  return record;
}

const RECORDED_AT = '2026-10-14T00:00:00.000Z';

function buildTwins(record: ReturnType<typeof makePolygonRecord>) {
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.cards',
    recordedAt: RECORDED_AT,
    adapterLabel: 'fixture'
  });
  return (['full', 'thumbnail'] as const).map(size =>
    buildLensTwin(file, {scenario: 'gpu-core.cards', rung: 'result'}, {size})
  );
}

// LENS-103: a collinear triangle has zero area, so the figure draws nothing and shows a fallback.
test('concise forms of an all-degenerate polygon state that nothing is drawn', () => {
  const record = makePolygonRecord([[0, 0, 1, 0, 2, 0, 0, 0]]);
  const context = createViewContext(record, {});
  const model = buildResultViewModel(context);
  const facts = RESULT_VIEW.describe(model, context);
  const statement = 'geometry not drawn: no drawable captured features (0 of 1 polygons shown)';
  expect(facts.display).toBe(statement);
  const description = generateLensDescription(facts);
  expect(description.alt.startsWith(`${statement}.`)).toBe(true);
  expect(description.live).toBe(
    `polygons 1; ${statement}; Planar coordinates (unit unknown); uniform scale.`
  );
  expect(description.llms).toContain(`- ${statement}`);
  const [full, thumbnail] = buildTwins(record);
  expect(full.description).toContain(statement);
  expect(full.caption).toContain(statement);
  expect(thumbnail.description).toContain(statement);
  // Thumbnail captions keep the first sentence, which is now the suppression statement.
  expect(thumbnail.caption).toBe(`${statement}.`);
});

test('drawable polygons keep the unchanged concise description', () => {
  const record = makePolygonRecord([[0, 0, 1, 0, 0, 1, 0, 0]]);
  const context = createViewContext(record, {});
  const facts = RESULT_VIEW.describe(buildResultViewModel(context), context);
  expect(facts.display).toBeUndefined();
  const [full, thumbnail] = buildTwins(record);
  expect(full.description).not.toContain('not drawn');
  expect(thumbnail.caption).toBe('Result of gpu-core.cards: no recorded inputs → geometry.');
});

test('empty captured geometry is stated in the concise forms', () => {
  const record = makePolygonRecord([]);
  const context = createViewContext(record, {});
  const facts = RESULT_VIEW.describe(buildResultViewModel(context), context);
  expect(facts.display).toBe('geometry not drawn: empty captured geometry (0 of 0 polygons shown)');
  expect(generateLensDescription(facts).live).toContain(
    'geometry not drawn: empty captured geometry'
  );
});

test('suppressed input and drawable output appear together in every concise form', () => {
  const record = makePolygonRecord([[0, 0, 1, 0, 0, 1]]);
  Object.freeze(record.captures!.snapshots);
  record.captures!.snapshots = [
    ...record.captures!.snapshots,
    makeY2Capture('roads', [0, 0, 0, 0], {
      role: 'input',
      components: 2,
      geometry: {kind: 'lines', coordinateModel: 'projected', offsetsCapture: 'road-offsets-id'}
    }),
    makeY2Capture('road-offsets', [0, 2])
  ];
  const context = createViewContext(record);
  const description = generateLensDescription(
    RESULT_VIEW.describe(buildResultViewModel(context), context)
  );
  const [full, thumbnail] = buildTwins(record);
  for (const text of [description.alt, description.live, full.description, thumbnail.caption]) {
    expect(text).toContain('roads not drawn');
    expect(text).toContain('geometry drawn');
  }
});

test.each([
  false,
  true
])('point output drawability is accurate beside a suppressed input; drawable=%s', drawable => {
  const record = makeY2Record([
    makeY2Capture('trees', drawable ? [0, 0] : [], {
      role: 'output',
      components: 2,
      geometry: {kind: 'points', coordinateModel: 'projected'}
    }),
    makeY2Capture('trail', [0, 0, 0, 0], {
      role: 'input',
      components: 2,
      geometry: {kind: 'lines', coordinateModel: 'projected', offsetsCapture: 'trail-offsets-id'}
    }),
    makeY2Capture('trail-offsets', [0, 2])
  ]);
  const context = createViewContext(record);
  const model = buildResultViewModel(context);
  expect(
    findViewNodes(
      renderResultView(model, context),
      node => node.tag === 'circle' && node.attributes['data-role'] === 'output'
    )
  ).toHaveLength(drawable ? 1 : 0);
  const description = generateLensDescription(RESULT_VIEW.describe(model, context));
  const [full, thumbnail] = buildTwins(record);
  for (const text of [description.alt, description.live, full.description, thumbnail.caption]) {
    expect(text).toContain(drawable ? 'trees drawn' : 'trees not drawn');
    expect(text).toContain('trail not drawn');
  }
});

test.each([2, 600])('LENS-103 text follows emitted point samples for %s source points', count => {
  const coordinates = new Array<number>(count * 2).fill(NaN);
  coordinates[2] = 1;
  coordinates[3] = 1;
  const record = makeY2Record([
    makeY2Capture('input', [], {
      role: 'input',
      components: 2,
      geometry: {kind: 'points', coordinateModel: 'projected'}
    }),
    makeY2Capture('population', coordinates, {
      role: 'output',
      components: 2,
      geometry: {kind: 'points', coordinateModel: 'projected'}
    })
  ]);
  // These external captures have no resource/node references, but still belong to one graph.
  record.graphs = [{id: 'sample', form: 'compiled', domain: 'gpu-core', nodes: [], edges: []}];
  for (const size of ['full', 'thumbnail'] as const) {
    const context = createViewContext(record, {options: {size}});
    const model = buildResultViewModel(context);
    const tree = renderResultView(model, context);
    const circles = findViewNodes(
      tree,
      node => node.tag === 'circle' && node.attributes['data-role'] === 'output'
    );
    // With two panels, budgets are 200 / 32: strides for 600 rows are 3 / 19.
    // Row 1 is absent from both samples. With two rows, stride 1 draws row 1.
    expect(circles).toHaveLength(count === 2 ? 1 : 0);
    const description = generateLensDescription(RESULT_VIEW.describe(model, context));
    const twin = buildTwins(record)[size === 'full' ? 0 : 1];
    for (const text of [description.alt, description.live, twin.caption, twin.description]) {
      expect(text).toContain(count === 2 ? 'population drawn' : 'population not drawn');
    }
    if (count === 600) expect(description.live).toContain('0 of 600 points shown');
  }
});
