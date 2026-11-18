// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {Buffer} from '@luma.gl/core';
import {afterEach, expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {LensCommandGraph} from '../../../src/computation-lenses/lens-view-snapshot';
import * as runner from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import {fromRecordedLensFile, getLensScenario} from '@luma-contributors/lenses/computation-lenses';
import {buildLensTwin} from '../../../src/computation-views/twin/static-twin';
import {LensController} from '../../../src/computation-views/hosts/lens-controller';
import {ViewRegistry} from '../../../src/computation-views/registry/view-registry';
import {RESULT_VIEW} from '../../../src/computation-views/views/result-view';
import {findViewNodes} from '../../../src/computation-views/view-node';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {buildDatumReceiptModel} from '../../../src/computation-views/views/datum-receipt';
import {createLocalMoranPickingRecord} from '../../computation-lenses/polygon-picking.x1.test-utils';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/segment-intersection.lens';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/geometry-validity.lens';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-predicate-join.lens';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/dot-density.lens';
import '../../../src/computation-lenses/scenarios/index';

afterEach(() => vi.restoreAllMocks());

async function buildMetadata(scenario: string) {
  const scenarioId = `gpu-spatial-analysis.${scenario}`;
  const recorded = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(`../../../../../docs/lens-records/${scenarioId}.json`, import.meta.url),
        'utf8'
      )
    )
  );
  const device = createNullWebGPUDevice();
  const graph = new LensCommandGraph<undefined>(device);
  const owned: {destroy(): void}[] = [];
  let specifications: Record<string, runner.S2CaptureSpec> = {};
  let keySpaces = recorded.record.axes.keySpaces;
  vi.spyOn(runner, 'runS2Lens').mockImplementation(async (_device, props) => {
    specifications = await props.build(graph, {
      own: resource => {
        owned.push(resource);
        return resource;
      },
      view(id, data, format, length) {
        const buffer = device.createBuffer({
          id,
          data,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        });
        owned.push(buffer);
        const handle = graph.importBuffer(
          {id, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        );
        const components = Number(format.match(/x([234])$/)?.[1] ?? 1);
        return graph.createDataView(handle, {format, length: length ?? data.length / components});
      }
    });
    keySpaces = [...(props.keySpaces ?? [])];
    return recorded.record;
  });
  try {
    await (await getLensScenario(scenarioId)!.load()).run(device);
  } finally {
    for (const resource of owned) resource.destroy();
    device.destroy();
  }
  // Reuse archived numeric evidence with the real scenario's newly emitted presentation metadata.
  const record = {
    ...recorded.record,
    axes: {...recorded.record.axes, keySpaces},
    captures: {
      ...recorded.record.captures!,
      snapshots: recorded.record.captures!.snapshots.map(capture => ({
        ...capture,
        keySpace: specifications[capture.name]?.keySpace ?? capture.keySpace,
        geometry: specifications[capture.name]?.geometry ?? capture.geometry
      }))
    }
  };
  return {recorded: {...recorded, record}, specifications};
}

test('H1 the scenario binds its published count and both rendered twins headline 1,802 points', async () => {
  const {recorded, specifications} = await buildMetadata('segment-intersection');
  expect(specifications['intersections'].geometry).toHaveProperty('countCapture', 'count');
  for (const size of ['full', 'thumbnail'] as const) {
    const twin = buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'result'}, {size});
    expect(twin.keyNumber).toEqual({label: 'points', value: '1,802'});
    expect(twin.llms).toContain('points: 1,802');
  }
});

test.each([
  ['geometry-validity', 'resultParks', 'parks', 'issueCode'],
  ['spatial-predicate-join', 'joinedTracts', 'tracts', 'tractMatches']
] as const)('X4 %s feature 1 receipts exclude vertex-domain rows after picking', async (scenario, output, domain, scalar) => {
  const {recorded, specifications} = await buildMetadata(scenario);
  expect(specifications[domain].keySpace).toBe(`${domain}-display-vertex`);
  expect(specifications[output].keySpace).toBe(`${domain}-display-vertex`);
  expect(specifications[scalar].keySpace).toBe(domain);
  const registry = new ViewRegistry();
  registry.register(RESULT_VIEW);
  const controller = new LensController({
    record: recorded.record,
    registry,
    rung: 'result',
    lensId: scenario,
    bindings: {entity: 1},
    params: {keySpace: domain},
    presentation: getLensScenario(recorded.scenarioId)!.presentation
  });
  try {
    const path = findViewNodes(
      controller.getFrame().tree,
      node =>
        node.tag === 'path' &&
        node.attributes['data-key-space'] === domain &&
        Number(node.attributes['data-key']) === 1
    )[0];
    expect(path).toBeDefined();
    controller.dispatch({
      type: 'pick',
      axis: 'entity',
      keySpace: domain,
      keys: [Number(path.attributes['data-key'])]
    });
    controller.dispatch({type: 'bind', axis: 'entity', value: 1});
    const receipt = buildDatumReceiptModel(
      createViewContext(recorded.record, {
        bindings: controller.getState().bindings,
        options: {keySpace: domain}
      })
    );
    expect(receipt.rows.map(row => row.name)).not.toContain(domain);
    expect(receipt.rows.map(row => row.name)).not.toContain(output);
    const scalarCapture = recorded.record.captures!.snapshots.find(
      capture => capture.name === scalar
    )!;
    expect(receipt.rows.find(row => row.name === scalar)?.value).toBe(
      String(scalarCapture.values[1])
    );
    expect(controller.getState().clauses).toMatchObject([
      {keySpace: domain, predicate: {keys: [1]}}
    ]);
  } finally {
    controller.destroy();
  }
});

test('X5 dot-density full and thumbnail twins headline the captured count', async () => {
  const {recorded, specifications} = await buildMetadata('dot-density');
  expect(specifications['points'].geometry).toHaveProperty('countCapture', 'count');
  const scenario = getLensScenario(recorded.scenarioId)!;
  expect(scenario.presentation?.primaryOutputCaptureId).toBe(
    recorded.record.captures!.snapshots.find(capture => capture.name === 'points')!.id
  );
  for (const size of ['full', 'thumbnail'] as const) {
    const twin = buildLensTwin(recorded, {scenario: recorded.scenarioId, rung: 'result'}, {size});
    expect(twin.keyNumber).toEqual({label: 'points', value: '1,439'});
    expect(twin.llms).toContain('points: 1,439');
    expect(twin.llms).toContain('1439 source vertices/points');
    expect(twin.llms).not.toContain('1540 source');
  }
});

test('X9 live Local Moran geometry aliases pick tracts while retaining vertex-domain receipts', () => {
  const archived = fromRecordedLensFile(
    JSON.parse(
      readFileSync(
        new URL(
          '../../../../../docs/lens-records/gpu-spatial-analysis.local-moran.json',
          import.meta.url
        ),
        'utf8'
      )
    )
  ).record;
  const record = createLocalMoranPickingRecord(archived);
  const registry = new ViewRegistry();
  registry.register(RESULT_VIEW);
  const controller = new LensController({
    record,
    registry,
    rung: 'result',
    lensId: 'moran-picking'
  });
  try {
    const paths = findViewNodes(
      controller.getFrame().tree,
      node =>
        node.tag === 'path' &&
        node.attributes['data-action'] === 'pick' &&
        node.attributes['data-key-space'] === 'tract'
    );
    expect(paths).toHaveLength(132);
    const path = paths[1];
    controller.dispatch({
      type: 'pick',
      axis: 'entity',
      keySpace: String(path.attributes['data-key-space']),
      keys: [Number(path.attributes['data-key'])]
    });
    expect(controller.getState().clauses).toMatchObject([
      {keySpace: 'tract', predicate: {keys: [Number(path.attributes['data-key'])]}}
    ]);
    expect(
      record.captures!.snapshots.find(capture => capture.name === 'localI.geometry')?.keySpace
    ).toBe('vertex');
  } finally {
    controller.destroy();
  }
});

test.each([
  'areal-interpolation',
  'minimum-clearance',
  'point-in-polygon-join',
  'pycnophylactic',
  'fleet-dwell-recipe',
  'catchment-accessibility',
  'rate-cluster-map-recipe',
  'offset-expansion',
  'segregation',
  'pair-gather',
  'coverage-validity',
  'spatial-join-candidates',
  'change-of-support-recipe',
  'spatial-predicate-join',
  'geometry-predicates',
  'geometry-validity',
  'huff-trade-areas',
  'spatial-join-prepared',
  'spatial-regression-recipe',
  'points-in-polygons-choropleth-recipe'
])('X4 %s keeps every display-coordinate consumer out of the feature receipt domain', async scenario => {
  const {specifications} = await buildMetadata(scenario);
  for (const [name, spec] of Object.entries(specifications)) {
    const geometry = spec.geometry;
    if (geometry?.kind !== 'polygons' || !geometry.offsetsCapture?.endsWith('DisplayOffsets'))
      continue;
    const domain = geometry.offsetsCapture.slice(0, -'DisplayOffsets'.length);
    expect(spec.keySpace, name).toBe(`${domain}-display-vertex`);
    expect(geometry.featureKeySpace, name).toBe(domain);
  }
});

test.each([
  ['geometry-validity', 'resultParks', 'parks', 13],
  ['spatial-predicate-join', 'joinedTracts', 'tracts', 7]
] as const)('H2 %s declares input/output feature domains and actual paths select their rows', async (scenario, output, domain, count) => {
  const {recorded, specifications} = await buildMetadata(scenario);
  expect(specifications[domain].geometry).toHaveProperty('featureKeySpace', domain);
  expect(specifications[output].geometry).toHaveProperty('featureKeySpace', domain);
  const registry = new ViewRegistry();
  registry.register(RESULT_VIEW);
  const controller = new LensController({
    record: recorded.record,
    registry,
    rung: 'result',
    lensId: scenario,
    presentation: getLensScenario(recorded.scenarioId)!.presentation
  });
  try {
    const id = recorded.record.captures!.snapshots.find(capture => capture.name === output)!.id;
    const paths = findViewNodes(
      controller.getFrame().tree,
      node =>
        node.tag === 'path' &&
        node.attributes['data-capture-id'] === id &&
        node.attributes['data-action'] === 'pick'
    );
    expect(paths).toHaveLength(count);
    const path = paths.at(-1)!;
    expect(path.attributes['data-key-space']).toBe(domain);
    const key = Number(path.attributes['data-key']);
    controller.dispatch({
      type: 'pick',
      axis: 'entity',
      keySpace: String(path.attributes['data-key-space']),
      keys: [key]
    });
    expect(controller.getState().clauses).toMatchObject([
      {keySpace: domain, predicate: {keys: [key]}}
    ]);
  } finally {
    controller.destroy();
  }
});
