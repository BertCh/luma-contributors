// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, expect, expectTypeOf, test, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import type {Device} from '@luma.gl/core';
import {captureLensScenarioRecord} from '../../../src/computation-lenses/run-lens-scenario';
import {
  getLensScenario,
  isLensScenarioId,
  listLensScenarios
} from '../../../src/computation-lenses/registry-implementation';
import {
  registerLensScenario,
  toRecordedLensFile,
  fromRecordedLensFile
} from '../../../src/computation-lenses/registry';
import type {LensScenarioDescriptor} from '../../../src/computation-lenses/types';
import {makeScanRecord} from '../fixtures/scan-record';

const descriptor: LensScenarioDescriptor = {
  id: 'registry-test.scan',
  entry: 'registry-test',
  element: 'GPUScan',
  title: 'test',
  audience: 'engineer',
  rungs: ['table'],
  defaultRung: 'table',
  load: async () => ({run: async () => makeScanRecord()})
};

const disposeRegistrations: (() => void)[] = [];
afterAll(() => {
  for (const dispose of disposeRegistrations) dispose();
});

test('B7: an explicit empty presentation preserves the last-output choice through serialization', async () => {
  const {buildLensTwin} = await import('../../../src/computation-views/twin/static-twin');
  const file = JSON.parse(
    readFileSync(
      new URL(
        '../../../../../docs/lens-records/gpu-spatial-analysis.kriging.json',
        import.meta.url
      ),
      'utf8'
    )
  );
  const decoded = fromRecordedLensFile({...file, presentation: {}});
  const directive = {scenario: decoded.scenarioId, rung: 'result'};
  const before = buildLensTwin(decoded, directive);
  // The explicit empty choice selects the last capture (uncertainCount: one row), rather than
  // the absent-presentation kriging fallback (rainfall: 4,662 cells).
  expect(before.keyNumber).toEqual({label: 'rows', value: '1'});
  const rewritten = toRecordedLensFile(decoded.record, {
    scenarioId: decoded.scenarioId,
    recordedAt: decoded.recordedAt,
    adapterLabel: decoded.adapterLabel,
    presentation: decoded.presentation
  });
  expect(buildLensTwin(rewritten, directive).keyNumber).toEqual(before.keyNumber);
  expect(fromRecordedLensFile(JSON.parse(JSON.stringify(rewritten))).presentation).toEqual({});
});

test('R6 rejects compact graphs with a missing dictionary and accepts genuine legacy graphs', () => {
  const record = makeScanRecord();
  delete record.captures;
  delete record.metrics;
  const encoded = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.scan',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture GPU'
  });
  const {encoding: ignoredEncoding, ...missingDictionary} = encoded;
  expect(ignoredEncoding).toBeDefined();
  expect(() => fromRecordedLensFile(missingDictionary)).toThrow('graph');
  expect(fromRecordedLensFile({...missingDictionary, record}).record.graphs[0].id).toBe(
    record.graphs[0].id
  );
});

test('validates scenario ids, coverage contract, duplicates and sorted lookup', () => {
  expect(isLensScenarioId('gpu-core.scan.small-data')).toBe(true);
  for (const id of ['scan', 'GPU.scan', 'gpu-.scan', 'gpu.scan.a.b', '.scan'])
    expect(isLensScenarioId(id)).toBe(false);
  expect(() => registerLensScenario({...descriptor, entry: 'wrong'})).toThrow('entry');
  expect(() => registerLensScenario({...descriptor, defaultRung: 'kernel'})).toThrow('defaultRung');
  expect(() => registerLensScenario({...descriptor, element: 'local-moran'})).toThrow('element');
  expect(() => registerLensScenario({...descriptor, id: 'bad'})).toThrow('id');
  const previous = listLensScenarios();
  const dispose = registerLensScenario(descriptor);
  disposeRegistrations.push(dispose);
  expect(getLensScenario(descriptor.id)).toBe(descriptor);
  expect(() => registerLensScenario(descriptor)).toThrow('Duplicate');
  expect(listLensScenarios().map(scenario => scenario.id)).toEqual(
    listLensScenarios()
      .map(scenario => scenario.id)
      .sort()
  );
  dispose();
  expect(getLensScenario(descriptor.id)).toBeUndefined();
  const replacement = {...descriptor, title: 'replacement'};
  const disposeReplacement = registerLensScenario(replacement);
  disposeRegistrations.push(disposeReplacement);
  dispose();
  expect(getLensScenario(descriptor.id)).toBe(replacement);
  disposeReplacement();
  expect(listLensScenarios()).toEqual(previous);
});

test.each([
  ['registration → static-twin → example-mock', ['registration', 'static-twin', 'example-mock']],
  ['example-mock → static-twin → registration', ['example-mock', 'static-twin', 'registration']]
] as const)('W2-03: fixture order %s leaves real behavior for the next caller', async (_name, order) => {
  const identifiers = order.map(name => `registry-order.${name}`);
  const trace: string[] = [];
  const fixtureDisposers: (() => void)[] = [];
  vi.resetModules();
  try {
    for (const name of [...order, 'next-caller']) {
      const registry = await import('../../../src/computation-lenses/registry-implementation');
      await import('../../../src/computation-lenses/scenarios/index');
      const publicRegistry = await import('../../../src/computation-lenses/index');
      expect(
        publicRegistry.getLensScenario('gpu-spatial-analysis.building-tessellation-recipe')?.element
      ).toBe('addBuildingTessellationRecipe');
      expect(
        publicRegistry.getLensScenario('gpu-spatial-analysis.building-tessellation-recipe')
      ).toBe(registry.getLensScenario('gpu-spatial-analysis.building-tessellation-recipe'));
      for (const id of identifiers)
        expect(registry.getLensScenario(id), `${trace.at(-1)} → ${name}: ${id}`).toBeUndefined();
      // Each next caller imports and executes real modules after the previous fixture's cleanup.
      const spatial = await import('@luma-contributors/analysis/gpu-spatial-analysis');
      expect(vi.isMockFunction(spatial.addBuildingTessellationReferenceRecipe)).toBe(false);
      expect(
        Array.from(
          spatial.getRasterResistanceConnectivityParameterValues({
            width: 1,
            height: 1,
            affine: [0.125, 0, 0, 0, 1, 0],
            pixelInterpretation: 'area'
          })
        )
      ).toContain(0.125);
      const {createNullWebGPUDevice} = await import(
        '../../../../analysis/src/testing/gpu-contributor-test-utils'
      );
      const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
      const example = await import(
        '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/building-tessellation-recipe.example'
      );
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const result = await example.runExample(graph, {
          segments: createTransientView(graph, 'order-segments', 'float32x4', 4),
          segmentBuildingIds: createTransientView(graph, 'order-owners', 'uint32', 4),
          enclosure: createTransientView(graph, 'order-enclosure', 'float32x2', 4),
          buildingCount: 1,
          sampleSpacing: 2,
          sampleCapacity: 4,
          vertexTolerance: 1 / 1024,
          capacity: {intersections: 16, nodedSegments: 16, vertices: 16, rings: 4}
        });
        expect(result.outputs.map(output => output.geometry.kind)).toEqual(['polygons']);
      } finally {
        device.destroy();
      }
      trace.push(name);
      if (name === 'next-caller') continue;
      const fixture: LensScenarioDescriptor = {
        ...descriptor,
        id: `registry-order.${name}`,
        entry: 'registry-order',
        load: async () => ({run: async () => makeScanRecord()})
      };
      const previous = registry.listLensScenarios();
      const dispose = registry.registerLensScenario(fixture);
      fixtureDisposers.push(dispose);
      expect(registry.getLensScenario(fixture.id)).toBe(fixture);
      expect(() => registry.registerLensScenario(fixture)).toThrow('Duplicate');
      expect(await (await registry.getLensScenario(fixture.id)!.load()).run({} as Device)).toEqual(
        makeScanRecord()
      );
      if (name === 'static-twin') {
        const twin = await import('../../../src/computation-views/twin/static-twin');
        const {ViewRegistry} = await import(
          '../../../src/computation-views/registry/view-registry'
        );
        const views = new ViewRegistry();
        const guardsFor = vi.fn((_target: {element?: string}) => []);
        views.registerPack({id: 'gpu-core', guardsFor});
        const recorded = registry.toRecordedLensFile(makeScanRecord(), {
          scenarioId: fixture.id,
          recordedAt: '2026-10-14T00:00:00.000Z',
          adapterLabel: 'order fixture'
        });
        const expectRegisteredTwin = (scenarioId: string) => {
          guardsFor.mockClear();
          expect(
            twin.buildLensTwin(
              {...recorded, scenarioId},
              {scenario: scenarioId, rung: 'table'},
              {registry: views}
            ).viewId
          ).toBe('table');
          expect(guardsFor).toHaveBeenCalledWith(
            expect.objectContaining({element: fixture.element})
          );
        };
        expectRegisteredTwin(fixture.id);
        // A table fallback also renders for an unknown ID. It cannot prove registry visibility.
        expect(() => expectRegisteredTwin('registry-order.unregistered')).toThrow();
      }
      dispose();
      const replacement = {...fixture, title: `replacement after ${name}`};
      const disposeReplacement = registry.registerLensScenario(replacement);
      fixtureDisposers.push(disposeReplacement);
      dispose();
      expect(registry.getLensScenario(fixture.id)).toBe(replacement);
      disposeReplacement();
      expect(registry.listLensScenarios()).toEqual(previous);
      if (name === 'example-mock') {
        vi.resetModules();
        vi.doMock('@luma-contributors/analysis/gpu-spatial-analysis', () => ({
          addBuildingTessellationReferenceRecipe: vi.fn(() => ({outputs: ['order mock']}))
        }));
        try {
          const example = await import(
            '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/building-tessellation-recipe.example'
          );
          expect(await example.runExample({} as never, {} as never)).toEqual({
            outputs: ['order mock']
          });
        } finally {
          vi.doUnmock('@luma-contributors/analysis/gpu-spatial-analysis');
          vi.resetModules();
        }
      }
    }
    expect(trace).toEqual([...order, 'next-caller']);
  } finally {
    for (const dispose of fixtureDisposers.reverse()) dispose();
    vi.doUnmock('@luma-contributors/analysis/gpu-spatial-analysis');
    vi.resetModules();
  }
});

test('JSON round trip preserves typed capture arrays and non-finite tokens', () => {
  const record = makeScanRecord();
  const capture = record.captures!.snapshots[0];
  record.captures = {
    ...record.captures!,
    snapshots: [
      {
        ...capture,
        id: 'float-values',
        elementType: 'f32',
        values: new Float32Array([NaN, Infinity, -Infinity, 1.25])
      },
      {
        ...capture,
        id: 'unsigned-values',
        elementType: 'u32',
        values: new Uint32Array([0, 4294967295])
      },
      {
        ...capture,
        id: 'signed-values',
        elementType: 'i32',
        values: new Int32Array([-2147483648, 2147483647])
      }
    ]
  };
  const serialized = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.scan',
    recordedAt: record.createdAt,
    adapterLabel: 'Apple M3'
  });
  const json = JSON.stringify(serialized);
  // Typed captures travel as base64 of their little-endian bytes (NaN and infinities included).
  expect(json).toContain('"values":"b64:AADAfwAAgH8AAID/AACgPw=="');
  expect(json).not.toContain('null');
  const restored = fromRecordedLensFile(JSON.parse(json));
  expect(restored.record.captures!.snapshots.map(snapshot => snapshot.values)).toEqual(
    record.captures.snapshots.map(snapshot => snapshot.values)
  );
  expect(restored.record.captures!.snapshots[0].values).toBeInstanceOf(Float32Array);
  expect(restored.record.captures!.snapshots[1].values).toBeInstanceOf(Uint32Array);
  expect(restored.record.captures!.snapshots[2].values).toBeInstanceOf(Int32Array);
  expect(capture.values).toBeInstanceOf(Uint32Array);
  expect(() => fromRecordedLensFile({schema: 'wrong'})).toThrow('schema');
  expect(() => fromRecordedLensFile({...serialized, recordedAt: 'bad'})).toThrow('recordedAt');
  const invalid = JSON.parse(json);
  invalid.record.captures.snapshots[0].values = [null];
  expect(() => fromRecordedLensFile(invalid)).toThrow('values');
  const duplicated = JSON.parse(json);
  duplicated.record.captures.snapshots[1].id = duplicated.record.captures.snapshots[0].id;
  expect(() => fromRecordedLensFile(duplicated)).toThrow('Duplicate capture id');
});

test('series values and unrelated attributes survive deep conversion', () => {
  const record = makeScanRecord();
  record.graphs = record.graphs.map(graph => ({...graph, attrs: {values: 'a named attribute'}}));
  record.metrics = record.metrics!.map(metrics => ({
    ...metrics,
    series: [{name: 'residual', axis: 'iteration', values: [NaN, Infinity, 0]}]
  }));
  const serialized = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.scan',
    recordedAt: record.createdAt,
    adapterLabel: 'Apple M3'
  });
  const restored = fromRecordedLensFile(JSON.parse(JSON.stringify(serialized)));
  expect(restored.record.graphs[0].attrs?.['values']).toBe('a named attribute');
  expect(restored.record.metrics![0].series![0].values).toEqual([NaN, Infinity, 0]);
});

test('writer rejects metadata that the reader would reject', () => {
  const record = makeScanRecord();
  const options = {
    scenarioId: 'gpu-core.scan',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture GPU'
  };
  expect(() => toRecordedLensFile(record, {...options, recordedAt: 'bad'})).toThrow('recordedAt');
  expect(() => toRecordedLensFile(record, {...options, adapterLabel: '  '})).toThrow(
    'adapterLabel'
  );
  expect(() => toRecordedLensFile(record, {...options, variant: 42 as unknown as string})).toThrow(
    'variant'
  );
});

test('encoded payloads require decoding before graph access', () => {
  const record = makeScanRecord();
  const serialized = toRecordedLensFile(record, {
    scenarioId: 'gpu-core.scan',
    recordedAt: record.createdAt,
    adapterLabel: 'Fixture GPU'
  });
  expectTypeOf(serialized.record).toBeUnknown();
  const payload = JSON.parse(JSON.stringify(serialized));
  expect(Array.isArray(payload.record.graphs[0])).toBe(true);
  const decoded = fromRecordedLensFile(serialized);
  expect(decoded.record.graphs[0].id).toBe(record.graphs[0].id);
  expect(decoded.record.captures!.snapshots[0].values).toBeInstanceOf(Uint32Array);
});

test('capture returns an encoded file that decodes into an inspectable recording', async () => {
  const record = makeScanRecord();
  const device = {info: {renderer: 'Fixture GPU', gpuBackend: 'webgpu'}} as unknown as Device;
  const captured = await captureLensScenarioRecord(descriptor, device, {
    recordedAt: record.createdAt
  });
  expectTypeOf(captured.record).toBeUnknown();
  const decoded = fromRecordedLensFile(captured);
  expect(decoded.record.graphs[0].id).toBe(record.graphs[0].id);
  expect(decoded.adapterLabel).toBe('Fixture GPU (webgpu)');
  const legacy = {...decoded, record};
  expect(fromRecordedLensFile(legacy).record.graphs).toEqual(record.graphs);
});
