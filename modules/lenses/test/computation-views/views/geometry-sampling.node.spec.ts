// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {fromRecordedLensFile, toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {buildLensTwin} from '../../../src/computation-views/twin/static-twin';
import {LensController} from '../../../src/computation-views/hosts/lens-controller';
import {LadderCoordinator} from '../../../src/computation-views/coordinator/ladder-coordinator';
import {ViewRegistry} from '../../../src/computation-views/registry/view-registry';
import {getAllCaptures} from '../../../src/computation-views/views/record-queries';
import {getFinalStateCapture} from '../../../src/computation-views/views/value-strip';
import {
  buildResultViewModel,
  renderResultView,
  RESULT_VIEW
} from '../../../src/computation-views/views/result-view';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {findViewNodes, getViewNodeText} from '../../../src/computation-views/view-node';
import {makeY2Capture, makeY2Record} from '../fixtures/y2-card-records';
import {
  listDocsMarkdownFiles,
  scanLensDirectives,
  twinKey
} from '../../computation-docs/lens-directives';

function makeGeometryRecord(kind: 'lines' | 'polygons', features: number[][]) {
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
        kind,
        coordinateModel: 'projected',
        offsetsCapture: 'offsets-id',
        featureKeySpace: 'row'
      }
    }),
    makeY2Capture('offsets', offsets)
  ]);
  // Feature rows have their own domain; coordinate rows are vertices, not selectable features.
  record.axes.axes[0].keySpace = 'row';
  record.axes.keySpaces = [{id: 'row', label: 'feature row', size: features.length}];
  return record;
}

function renderGeometry(
  record: ReturnType<typeof makeGeometryRecord>,
  thumbnail: boolean,
  entity?: number
) {
  const context = createViewContext(record, {
    bindings: entity === undefined ? {} : {entity},
    options: thumbnail ? {size: 'thumbnail'} : {}
  });
  const model = buildResultViewModel(context);
  const tree = renderResultView(model, context);
  const paths = findViewNodes(
    tree,
    node => node.tag === 'path' && node.attributes['data-role'] === 'output'
  );
  return {context, model, tree, paths};
}

function getPathVertices(path: {attributes: Record<string, unknown>}): [number, number][] {
  return [...String(path.attributes['d']).matchAll(/[ML]([^ ,]+),([^ ]+)/g)].map(match => [
    Number(match[1]),
    Number(match[2])
  ]);
}

test.each([
  'lines',
  'polygons'
] as const)('%s use the offsets feature origin independently of the vertex origin', kind => {
  for (const first of [0, 2]) {
    const record = makeGeometryRecord(kind, [[0, 0, 1, 0, 0, 1]]);
    record.captures!.snapshots[1].meaning = `first:${first}`;
    const geometry = record.captures!.snapshots[0].geometry!;
    if (geometry.kind !== 'lines' && geometry.kind !== 'polygons') throw new Error('fixture');
    geometry.valuesCapture = 'feature-values-id';
    Object.freeze(record.captures!.snapshots);
    record.captures!.snapshots = [
      ...record.captures!.snapshots,
      makeY2Capture('feature-values', [10, 20, 30], {keySpace: 'row'})
    ];
    record.axes.keySpaces[0].size = 3;
    const {paths, tree} = renderGeometry(record, false, first);
    expect(paths).toHaveLength(1);
    expect(paths[0].attributes['data-key']).toBe(first);
    expect(paths[0].attributes['data-feature-row']).toBe(first);
    expect(paths[0].attributes['data-action']).toBe('pick');
    expect(getViewNodeText(tree)).toContain(String(first === 2 ? 30 : 10));
    if (first === 2) {
      // Same value domain, but put the independently expected row-2 value (30) at row zero.
      // Comparing the mark itself prevents a diagnostic table elsewhere from satisfying this.
      const control = structuredClone(record);
      const colorAttribute = kind === 'lines' ? 'stroke' : 'fill';
      control.captures!.snapshots[1].meaning = 'first:0';
      control.captures!.snapshots[2].values = [30, 20, 10];
      expect(paths[0].attributes[colorAttribute]).toBe(
        renderGeometry(control, false).paths[0].attributes[colorAttribute]
      );
      control.captures!.snapshots[2].values = [10, 20, 30];
      expect(paths[0].attributes[colorAttribute]).not.toBe(
        renderGeometry(control, false).paths[0].attributes[colorAttribute]
      );
    }
    // Offsets refer to vertex keys, independent of the feature origin on that same capture.
    record.captures!.snapshots[0].meaning = 'first:5';
    record.captures!.snapshots[1].values = [5, 8];
    const shifted = renderGeometry(record, false, first);
    expect(shifted.paths[0].attributes['d']).toBe(paths[0].attributes['d']);
    expect(shifted.paths[0].attributes['data-key']).toBe(first);
    const coordinator = new LadderCoordinator();
    const registry = new ViewRegistry();
    registry.register(RESULT_VIEW);
    const controller = new LensController({
      record,
      coordinator,
      registry,
      lensId: 'partial',
      rung: 'result'
    });
    try {
      controller.dispatch({
        type: 'pick',
        axis: 'entity',
        keySpace: String(shifted.paths[0].attributes['data-key-space']),
        keys: [Number(shifted.paths[0].attributes['data-key'])]
      });
      expect(coordinator.getClauses('selection')[0].predicate).toEqual({
        kind: 'keys',
        keys: [first]
      });
      coordinator.clearClauses('selection', 'partial');
      expect(
        findViewNodes(
          controller.getFrame().tree,
          node => node.tag === 'path' && node.attributes['data-state'] === 'highlight'
        )
      ).toHaveLength(0);
      coordinator.pick('selection', {viewId: 'peer', keySpace: 'row'}, [first]);
      expect(
        findViewNodes(
          controller.getFrame().tree,
          node =>
            node.tag === 'path' &&
            node.attributes['data-key'] === first &&
            node.attributes['data-state'] === 'highlight'
        )
      ).toHaveLength(1);
    } finally {
      controller.destroy();
    }
  }
});

test.each([
  'lines',
  'polygons'
] as const)('AMG-65 %s playback retains changing feature keys', kind => {
  const record = makeGeometryRecord(kind, [[0, 0, 1, 0, 0, 1]]);
  const [coordinates, offsets] = record.captures!.snapshots;
  if (coordinates.geometry?.kind !== kind) throw new Error('fixture');
  coordinates.geometry.offsetsCapture = 'offsets';
  record.captures!.snapshots = [
    {...coordinates, id: 'coordinates-0', iteration: 0},
    {...coordinates, id: 'coordinates-1', iteration: 1},
    {...offsets, id: 'offsets-0', iteration: 0, meaning: 'first:0'},
    {...offsets, id: 'offsets-id', iteration: 1, meaning: 'first:2'}
  ];
  record.axes.keySpaces[0].size = 3;
  const {model, tree} = renderGeometry(record, false);
  expect(model.frames).toHaveLength(2);
  const frames = findViewNodes(tree, node => node.attributes['class'] === 'cv-frame');
  expect(
    frames.map(frame =>
      findViewNodes(
        frame,
        node => node.tag === 'path' && node.attributes['data-role'] === 'output'
      ).map(path => path.attributes['data-key'])
    )
  ).toEqual([[0], [2]]);
});

function getPolygonArea(vertices: [number, number][]): number {
  const [originX, originY] = vertices[0];
  return Math.abs(
    vertices.reduce((area, point, index) => {
      const next = vertices[(index + 1) % vertices.length];
      return (
        area +
        (point[0] - originX) * (next[1] - originY) -
        (next[0] - originX) * (point[1] - originY)
      );
    }, 0) / 2
  );
}

test.each([
  false,
  true
])('R1 spare vertex capacity leaves projected output unchanged; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord('polygons', [[0, 0, 1, 0, 0, 1]]);
  const before = renderGeometry(record, thumbnail);
  record.captures!.snapshots[0].values = [0, 0, 1, 0, 0, 1, 3.402823e38, 3.402823e38];
  const after = renderGeometry(record, thumbnail);
  expect(after.paths.map(path => path.attributes['d'])).toEqual(
    before.paths.map(path => path.attributes['d'])
  );
  expect(getPolygonArea(getPathVertices(after.paths[0]))).toBeGreaterThan(0);
});

test('R1 projection contains both positions of an animated triangle', () => {
  const record = makeGeometryRecord('polygons', [[0, 0, 1, 0, 0, 1]]);
  const capture = record.captures!.snapshots[0];
  record.captures!.snapshots = [
    {...capture, id: 'earlier', iteration: 0, values: [100, 100, 101, 100, 100, 101]},
    {...capture, iteration: 1},
    record.captures!.snapshots[1]
  ];
  const {tree, model} = renderGeometry(record, false);
  const frames = findViewNodes(tree, node => node.attributes['class'] === 'cv-frame');
  expect(frames).toHaveLength(2);
  for (const frame of frames) {
    const paths = findViewNodes(
      frame,
      node => node.tag === 'path' && node.attributes['data-role'] === 'output'
    );
    expect(paths).toHaveLength(1);
    for (const path of paths) {
      for (const [horizontal, vertical] of getPathVertices(path)) {
        expect(Number.isFinite(horizontal) && Number.isFinite(vertical)).toBe(true);
        expect(horizontal).toBeGreaterThanOrEqual(0);
        expect(horizontal).toBeLessThanOrEqual(model.width);
        expect(vertical).toBeGreaterThanOrEqual(0);
        expect(vertical).toBeLessThanOrEqual(model.height);
      }
    }
  }
});

test.each([false, true])('R2 sampling retains a non-collinear corner; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [
      feature * 3,
      0,
      feature * 3,
      1,
      feature * 3 + 2,
      0,
      feature * 3 + 1.75,
      0,
      feature * 3 + 1.5,
      0,
      feature * 3 + 1.25,
      0,
      feature * 3 + 1,
      0
    ])
  );
  const {paths} = renderGeometry(record, thumbnail);
  expect(paths).toHaveLength(thumbnail ? 21 : 133);
  for (const path of paths) {
    const area = getPolygonArea(getPathVertices(path));
    expect(Number.isFinite(area)).toBe(true);
    expect(area).toBeGreaterThan(0);
  }
});

test('R3 a linked pick reserves its unpinned output feature', () => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1])
  );
  record.axes = {
    ...record.axes,
    axes: [{id: 'entity', label: 'row', kind: 'data', keySpace: 'row'}],
    keySpaces: [{id: 'row', label: 'row', size: 201}]
  };
  const coordinator = new LadderCoordinator();
  const registry = new ViewRegistry();
  registry.register(RESULT_VIEW);
  const controller = new LensController({
    record,
    coordinator,
    registry,
    lensId: 'linked-output',
    rung: 'result'
  });
  try {
    coordinator.pick('selection', {viewId: 'peer', keySpace: 'row'}, [1]);
    const frame = controller.getFrame();
    expect(frame.state.bindings.entity).toBeUndefined();
    expect(
      findViewNodes(
        frame.tree,
        node =>
          node.tag === 'path' &&
          node.attributes['data-key'] === 1 &&
          node.attributes['data-state'] === 'highlight'
      )
    ).toHaveLength(1);
  } finally {
    controller.destroy();
  }
});

test('unkeyed polygon features do not inherit their coordinate capture selection domain', () => {
  const record = makeGeometryRecord('polygons', [
    [0, 0, 1, 0, 0, 1],
    [2, 0, 3, 0, 2, 1],
    [4, 0, 5, 0, 4, 1]
  ]);
  const capture = record.captures!.snapshots[0];
  const geometry = capture.geometry!;
  if (geometry.kind !== 'polygons') throw new Error('Expected polygon fixture');
  delete geometry.featureKeySpace;
  capture.keySpace = 'vertex';
  record.axes.axes[0].keySpace = 'vertex';
  record.axes.keySpaces = [
    ...record.axes.keySpaces,
    {id: 'vertex', label: 'coordinate row', size: 9}
  ];
  const coordinator = new LadderCoordinator();
  const registry = new ViewRegistry();
  registry.register(RESULT_VIEW);
  const controller = new LensController({
    record,
    coordinator,
    registry,
    lensId: 'unkeyed-output',
    rung: 'result'
  });
  try {
    coordinator.pick('selection', {viewId: 'peer', keySpace: 'vertex'}, [1]);
    const frame = controller.getFrame();
    const paths = findViewNodes(frame.tree, node => node.tag === 'path');
    expect(paths.map(path => path.attributes['data-feature-row'])).toEqual([0, 1, 2]);
    expect(paths.every(path => path.attributes['data-action'] === undefined)).toBe(true);
    expect(paths.every(path => path.attributes['data-key'] === undefined)).toBe(true);
    expect(
      findViewNodes(frame.tree, node => node.attributes['data-state'] === 'highlight')
    ).toHaveLength(0);
  } finally {
    controller.destroy();
  }
});

test('R7 side-by-side SVGs describe their own capture and count', () => {
  const record = makeGeometryRecord('polygons', [[0, 0, 1, 0, 0, 1]]);
  record.captures!.snapshots = [
    ...record.captures!.snapshots,
    makeY2Capture('input-shapes', [10, 0, 11, 0, 10, 1, 12, 0, 13, 0, 12, 1], {
      role: 'input',
      components: 2,
      geometry: {kind: 'polygons', offsetsCapture: 'input-offsets-id', coordinateModel: 'projected'}
    }),
    makeY2Capture('input-offsets', [0, 3, 6])
  ];
  const context = createViewContext(record, {options: {layout: 'side-by-side'}});
  const tree = renderResultView(buildResultViewModel(context), context);
  const figures = findViewNodes(tree, node => node.attributes['class'] === 'lens-result-panel');
  expect(figures).toHaveLength(2);
  const descriptions = figures.map(figure =>
    findViewNodes(figure, node => node.tag === 'desc')
      .map(getViewNodeText)
      .join(' ')
  );
  expect(descriptions[0]).toContain('input-shapes: 2 of 2 polygons shown');
  expect(descriptions[1]).toContain('geometry: 1 of 1 polygons shown');
});

test('R7 animation key-frame SVGs describe their displayed frame', () => {
  const record = makeGeometryRecord('polygons', [
    [0, 0, 1, 0, 0, 1],
    [2, 0, 3, 0, 2, 1]
  ]);
  const capture = record.captures!.snapshots[0];
  record.captures!.snapshots = [
    {
      ...capture,
      id: 'earlier',
      iteration: 0,
      values: [10, 0, 11, 0, 10, 1],
      geometry: {
        kind: 'polygons',
        coordinateModel: 'projected',
        offsetsCapture: 'earlier-offsets-id'
      }
    },
    {...capture, iteration: 1},
    record.captures!.snapshots[1],
    makeY2Capture('earlier-offsets', [0, 3])
  ];
  const {tree} = renderGeometry(record, false);
  const keyFrames = findViewNodes(tree, node => node.attributes['class'] === 'cv-key-frames')[0];
  const descriptions = findViewNodes(keyFrames, node => node.tag === 'svg').map(svg =>
    findViewNodes(svg, node => node.tag === 'desc')
      .map(getViewNodeText)
      .join(' ')
  );
  expect(descriptions).toHaveLength(2);
  expect(descriptions[0]).toContain('1 of 1 polygons shown');
  expect(descriptions[1]).toContain('2 of 2 polygons shown');
});

test.each([false, true])('samples drawable triangles before vertices; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1])
  );
  const {context, model, tree, paths} = renderGeometry(record, thumbnail);
  expect(paths.length).toBe(thumbnail ? 21 : 133);
  expect(paths.every(path => (String(path.attributes['d']).match(/[ML]/g)?.length ?? 0) >= 3)).toBe(
    true
  );
  expect(
    paths.reduce(
      (count, path) => count + (String(path.attributes['d']).match(/[ML]/g)?.length ?? 0),
      0
    )
  ).toBeLessThanOrEqual(thumbnail ? 64 : 400);
  const shown = `${paths.length} of 201 polygons shown`;
  expect(RESULT_VIEW.describe(model, context).facts?.join(' ')).toContain(shown);
  expect(getViewNodeText(tree)).toContain(shown);
});

test.each([
  false,
  true
])('reserves pinned and highlighted polygon rows; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1])
  );
  record.captures!.snapshots[0].highlight = {key: 4, label: 'reserved highlight'};
  const {paths, tree} = renderGeometry(record, thumbnail, 1);
  expect(paths.map(path => Number(path.attributes['data-key']))).toContain(1);
  expect(paths.map(path => Number(path.attributes['data-key']))).toContain(4);
  expect(
    findViewNodes(tree, node => node.tag === 'path' && node.attributes['data-role'] === 'highlight')
  ).toHaveLength(1);
});

test.each([
  false,
  true
])('preserves both endpoints when simplifying a long line; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord('lines', [
    Array.from({length: 501}, (_, vertex) => [vertex, vertex % 2]).flat()
  ]);
  const {paths, model} = renderGeometry(record, thumbnail);
  expect(paths).toHaveLength(1);
  const project = (horizontal: number, vertical: number) =>
    model.projection
      .project(horizontal, vertical)
      .map(value => Math.round(value * 100) / 100)
      .join(',');
  const path = String(paths[0].attributes['d']);
  expect(path.startsWith(`M${project(0, 0)} `)).toBe(true);
  expect(path.endsWith(`L${project(500, 0)}`)).toBe(true);
  expect(path).not.toContain('Z');
});

test.each([
  false,
  true
])('closed triangles retain three distinct vertices; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1, feature, 0])
  );
  const {paths} = renderGeometry(record, thumbnail);
  expect(paths.length).toBeGreaterThan(0);
  for (const path of paths) {
    const vertices = String(path.attributes['d']).match(/[ML]([^ ]+)/g) ?? [];
    expect(vertices).toHaveLength(3);
    expect(new Set(vertices.map(vertex => vertex.slice(1))).size).toBe(3);
  }
});

test.each([
  false,
  true
])('reports an explicit fallback for undrawable output; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord('polygons', [
    [0, 0, 1, 0],
    [0, 0, NaN, 1, 2, 0]
  ]);
  const {tree, paths} = renderGeometry(record, thumbnail);
  expect(paths).toHaveLength(0);
  expect(getViewNodeText(tree)).toContain('no drawable captured features');
});

test.each([false, true])('reports empty captured geometry; thumbnail=%s', thumbnail => {
  const {tree, paths} = renderGeometry(makeGeometryRecord('lines', []), thumbnail);
  expect(paths).toHaveLength(0);
  expect(getViewNodeText(tree)).toContain('empty captured geometry');
});

test.each([
  false,
  true
])('does not hide invalid vertices during simplification; thumbnail=%s', thumbnail => {
  const vertices = Array.from({length: 501}, (_, vertex) => [vertex, vertex % 2]).flat();
  // Vertex 2 would be skipped by even sampling at either figure size.
  vertices[4] = NaN;
  const {tree, paths} = renderGeometry(makeGeometryRecord('lines', [vertices]), thumbnail);
  expect(paths).toHaveLength(0);
  expect(getViewNodeText(tree)).toContain('no drawable captured features');
  expect(getViewNodeText(tree)).toContain('1 invalid or uncaptured features omitted');
});

test.each([
  false,
  true
])('closed lines retain an intermediate vertex and both endpoints; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord(
    'lines',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 1, feature, 0])
  );
  const {paths} = renderGeometry(record, thumbnail);
  expect(paths).toHaveLength(thumbnail ? 21 : 133);
  for (const path of paths) {
    const vertices = (String(path.attributes['d']).match(/[ML]([^ ]+)/g) ?? []).map(vertex =>
      vertex.slice(1)
    );
    expect(vertices).toHaveLength(3);
    expect(vertices[0]).toBe(vertices[2]);
    expect(vertices[1]).not.toBe(vertices[0]);
    expect(String(path.attributes['d'])).not.toContain('Z');
  }
});

test('retains both reservations when a thumbnail panel has only its minimum allocation', () => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1])
  );
  record.captures!.snapshots[0].highlight = {key: 4, label: 'reserved highlight'};
  record.captures!.snapshots = [
    ...record.captures!.snapshots,
    ...Array.from({length: 30}, (_, layer) =>
      makeY2Capture(`context-${layer}`, [0, 0], {
        role: 'context',
        components: 2,
        geometry: {kind: 'points', coordinateModel: 'projected'}
      })
    )
  ];
  const {paths} = renderGeometry(record, true, 1);
  expect(paths.map(path => Number(path.attributes['data-key']))).toEqual([1, 4]);
});

test('invalid offsets do not hide the next valid line feature', () => {
  const record = makeGeometryRecord('lines', [
    [0, 0, 1, 1],
    [2, 2, 3, 3]
  ]);
  record.captures!.snapshots[1].values = [-1, 2, 4];
  const {paths, tree} = renderGeometry(record, true);
  expect(paths).toHaveLength(1);
  expect(paths[0].attributes['data-feature-row']).toBe(1);
  expect(getViewNodeText(tree)).toContain('1 of 2 lines shown');
  expect(getViewNodeText(tree)).toContain('1 invalid or uncaptured features omitted');
});

test.each([false, true])('static twins show sampled output polygons; thumbnail=%s', thumbnail => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1])
  );
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.cards',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture GPU'
  });
  const twin = buildLensTwin(
    file,
    {scenario: 'gpu-core.cards', rung: 'result'},
    {
      size: thumbnail ? 'thumbnail' : 'full'
    }
  );
  expect(twin.figure.html).toContain('data-role="output"');
  expect(twin.figure.html).toContain('data-feature-row=');
  expect(twin.figure.html).toContain(`${thumbnail ? 21 : 133} of 201 polygons shown`);
});

test('each archived line or polygon result has drawable marks or an explicit fallback in both sizes', () => {
  const directory = new URL('../../../../../docs/lens-records/', import.meta.url);
  let checkedOutputs = 0;
  for (const file of readdirSync(directory).filter(name => name.endsWith('.json'))) {
    const record = fromRecordedLensFile(
      JSON.parse(readFileSync(new URL(file, directory), 'utf8'))
    ).record;
    const names = new Set(getAllCaptures(record).map(capture => capture.name));
    for (const name of names) {
      const capture = getFinalStateCapture(record, name)!;
      if (
        capture.role !== 'output' ||
        !['lines', 'polygons'].includes(capture.geometry?.kind ?? '')
      )
        continue;
      for (const thumbnail of [false, true]) {
        const context = createViewContext(record, {
          options: {
            primaryOutputCaptureId: capture.id,
            ...(thumbnail ? {size: 'thumbnail'} : {})
          }
        });
        const tree = renderResultView(buildResultViewModel(context), context);
        const marks = findViewNodes(
          tree,
          node =>
            node.tag === 'path' &&
            node.attributes['data-feature-row'] !== undefined &&
            node.attributes['data-capture-id'] === capture.id
        );
        const fallbacks = findViewNodes(
          tree,
          node =>
            node.attributes['data-geometry-fallback'] === 'output' &&
            node.attributes['data-capture-id'] === capture.id
        );
        expect(
          marks.length + fallbacks.length,
          `${file}, ${name}, thumbnail=${thumbnail}`
        ).toBeGreaterThan(0);
        for (const svg of findViewNodes(tree, node => node.tag === 'svg')) {
          for (const path of findViewNodes(
            svg,
            node => node.tag === 'path' && node.attributes['data-capture-id'] === capture.id
          )) {
            const vertices = getPathVertices(path);
            for (const [horizontal, vertical] of vertices) {
              expect(Number.isFinite(horizontal) && Number.isFinite(vertical), file).toBe(true);
              expect(horizontal, file).toBeGreaterThanOrEqual(0);
              expect(horizontal, file).toBeLessThanOrEqual(Number(svg.attributes['width']));
              expect(vertical, file).toBeGreaterThanOrEqual(0);
              expect(vertical, file).toBeLessThanOrEqual(Number(svg.attributes['height']));
            }
            if (capture.geometry!.kind === 'polygons') {
              const area = getPolygonArea(vertices);
              expect(Number.isFinite(area), `${file}, ${name}`).toBe(true);
              expect(area, `${file}, ${name}, thumbnail=${thumbnail}`).toBeGreaterThan(0);
            } else
              expect(new Set(vertices.map(point => point.join(','))).size, file).toBeGreaterThan(1);
          }
        }
      }
      checkedOutputs++;
    }
  }
  expect(checkedOutputs).toBeGreaterThanOrEqual(88);
});

function getDeclaredResultTwinPaths(): Set<string> {
  const declared = new Set<string>();
  for (const file of listDocsMarkdownFiles()) {
    for (const {attributes} of scanLensDirectives(file)) {
      if (attributes['rung'] !== 'result') continue;
      const key = twinKey(attributes);
      declared.add(`docs/lens-records/twins/${key}.json`);
      if (
        attributes['datum'] === undefined &&
        attributes['iteration'] === undefined &&
        attributes['view'] === undefined
      )
        declared.add(`docs/lens-records/twins/${key}~thumbnail.json`);
    }
  }
  return declared;
}

function getOutOfScopeResultTwinChanges(
  before: ReadonlyMap<string, string>,
  after: ReadonlyMap<string, string>,
  permitted: ReadonlySet<string>
): string[] {
  return [...new Set([...before.keys(), ...after.keys()])]
    .filter(path => {
      if (before.get(path) === after.get(path)) return false;
      return !permitted.has(path);
    })
    .sort();
}

test('R8 result-only refresh rejects changed kernel content outside authored directives', () => {
  const root = new URL('../../../../../', import.meta.url);
  const permitted = getDeclaredResultTwinPaths();
  const scenario = 'gpu-spatial-analysis.local-frame-projection';
  const resultPath = `docs/lens-records/twins/${scenario}~result.json`;
  const kernelPath = `docs/lens-records/twins/${scenario}~kernel.json`;
  expect(permitted.has(resultPath)).toBe(true);
  expect(permitted.has(kernelPath)).toBe(false);
  const before = new Map(
    [resultPath, kernelPath].map(path => [path, readFileSync(new URL(path, root), 'utf8')])
  );
  const recorded = JSON.parse(
    readFileSync(new URL(`docs/lens-records/${scenario}.json`, root), 'utf8')
  );
  const refreshed = buildLensTwin(recorded, {scenario, rung: 'result'});
  const after = new Map(before);
  // A result-only refresh can replace its result, and must preserve all unrelated bytes.
  after.set(resultPath, JSON.stringify(refreshed));
  expect(after.get(resultPath)).not.toBe(before.get(resultPath));
  expect(getOutOfScopeResultTwinChanges(before, after, permitted)).toEqual([]);
  const kernel = JSON.parse(before.get(kernelPath)!);
  after.set(kernelPath, JSON.stringify({...kernel, html: `${kernel.html}<p>out of scope</p>`}));
  expect(getOutOfScopeResultTwinChanges(before, after, permitted)).toEqual([kernelPath]);
  after.delete(kernelPath);
  expect(getOutOfScopeResultTwinChanges(before, after, permitted)).toEqual([kernelPath]);
});

test('R8 archived result-view twins correspond to declared result directives', () => {
  const root = new URL('../../../../../', import.meta.url);
  const declared = getDeclaredResultTwinPaths();
  // The committed directives and twins are the inventory; no local refresh log is needed.
  for (const path of declared) {
    expect(JSON.parse(readFileSync(new URL(path, root), 'utf8')).rung, path).toBe('result');
  }
  const directory = new URL('docs/lens-records/twins/', root);
  const resultPaths: string[] = [];
  for (const file of readdirSync(directory).filter(name => name.endsWith('.json'))) {
    const path = `docs/lens-records/twins/${file}`;
    const twin = JSON.parse(readFileSync(new URL(path, root), 'utf8'));
    if (twin.viewId !== 'result-view' || twin.rung !== 'result') continue;
    resultPaths.push(path);
    expect(declared.has(path), path).toBe(true);
    expect(twin.rung, path).toBe('result');
    expect(twin.viewId, path).toBe('result-view');
  }
  expect(resultPaths.length).toBeGreaterThan(0);
  const unrelated =
    'docs/lens-records/twins/gpu-spatial-analysis.local-frame-projection~kernel.json';
  expect(JSON.parse(readFileSync(new URL(unrelated, root), 'utf8')).rung).toBe('kernel');
  expect(resultPaths).not.toContain(unrelated);
});

test('live controllers rebuild feature reservations and discard samples on record replacement', () => {
  const record = makeGeometryRecord(
    'polygons',
    Array.from({length: 201}, (_, feature) => [feature, 0, feature + 1, 0, feature, 1])
  );
  const registry = new ViewRegistry();
  registry.register(RESULT_VIEW);
  const controller = new LensController({
    record,
    registry,
    lensId: 'geometry-sampling',
    rung: 'result'
  });
  try {
    const frame = controller.dispatch({type: 'bind', axis: 'entity', value: 1});
    expect(
      findViewNodes(frame.tree, node => node.tag === 'path' && node.attributes['data-key'] === 1)
        .length
    ).toBeGreaterThan(0);
    const next = controller.dispatch({
      type: 'set-record',
      record: makeGeometryRecord('lines', [[0, 0, 1, 1]])
    });
    const paths = findViewNodes(
      next.tree,
      node => node.tag === 'path' && node.attributes['data-role'] === 'output'
    );
    expect(paths).toHaveLength(1);
    expect(paths[0].attributes['data-feature-row']).toBe(0);
    expect(String(paths[0].attributes['d'])).not.toContain('Z');
  } finally {
    controller.destroy();
  }
});

test('animation frames share the geometry budget and retain complete polygons', () => {
  const features = Array.from({length: 201}, (_, feature) => [
    feature,
    0,
    feature + 1,
    0,
    feature,
    1
  ]);
  const record = makeGeometryRecord('polygons', features);
  const capture = record.captures!.snapshots[0];
  record.captures!.snapshots = [
    {...capture, id: 'earlier', iteration: 0, values: features.flat().map(value => value + 1)},
    {...capture, id: 'final', iteration: 1},
    record.captures!.snapshots[1]
  ];
  const {model, tree} = renderGeometry(record, false);
  expect(model.frames).toHaveLength(2);
  const frames = findViewNodes(tree, node => node.attributes['class'] === 'cv-frame');
  expect(frames).toHaveLength(2);
  let vertices = 0;
  for (const frame of frames) {
    const paths = findViewNodes(
      frame,
      node => node.tag === 'path' && node.attributes['data-role'] === 'output'
    );
    expect(paths).toHaveLength(66);
    for (const path of paths) {
      const count = String(path.attributes['d']).match(/[ML]/g)?.length ?? 0;
      expect(count).toBeGreaterThanOrEqual(3);
      vertices += count;
    }
  }
  expect(vertices).toBeLessThanOrEqual(400);
});

test('saved Local Moran twins contain polygon marks in both sizes', () => {
  for (const suffix of ['', '~thumbnail']) {
    const file = new URL(
      `../../../../../docs/lens-records/twins/gpu-spatial-analysis.local-moran~result${suffix}.json`,
      import.meta.url
    );
    const twin = JSON.parse(readFileSync(file, 'utf8'));
    expect(twin.keyNumber).toEqual({label: 'polygons', value: '791'});
    expect(twin.figure.html).toMatch(/<path[^>]+data-feature-row=/);
    expect(twin.figure.html).toContain('of 791 polygons shown');
  }
});

test.each([
  'lines',
  'polygons'
] as const)('B2 %s playback preserves missing-offset fallback', kind => {
  const record = makeGeometryRecord(kind, [[0, 0, 1, 0, 0, 1]]);
  const [coordinates] = record.captures!.snapshots;
  record.captures!.snapshots = [
    {...coordinates, id: 'first', iteration: 0},
    {...coordinates, id: 'second', iteration: 1}
  ];
  const {model, tree} = renderGeometry(record, false);
  expect(model.output.unknown).toContain('offsets');
  expect(getViewNodeText(tree)).toContain('unknown');
});

test.each(['lines', 'polygons'] as const)('Earlier AMG-65 %s ignores unused vertex tails', kind => {
  const record = makeGeometryRecord(kind, [[0, 0, 1, 0, 0, 1]]);
  const [coordinates, offsets] = record.captures!.snapshots;
  record.captures!.snapshots = [
    {...coordinates, id: 'first', iteration: 0, values: [0, 0, 1, 0, 0, 1, 5, 5]},
    {...coordinates, id: 'second', iteration: 1, values: [0, 0, 1, 0, 0, 1, 50, 50]},
    offsets
  ];
  expect(renderGeometry(record, false).model.frames).toEqual([]);
});
