// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import type {Device} from '@luma.gl/core';
import type {
  CatalogOperationBinding,
  CatalogRunOutput
} from '../../src/analysis/catalog-binding-types';
import {
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {KeplerAllocationError} from '../../src/runtime/dataset-registry';
import {applyKeplerDatasetProfileDeclarations} from '../../src/profile/dataset-profile';
import {profileKeplerWorkbenchData} from '../../src/descriptors/workbench-adapters';
import {makeWorkflowDataFromFeatures} from '../../src/workflows/workflow-data';
import {
  createWorkflowOperationRegistry,
  getWorkflowOperationRegistry,
  getWorkflowPortCompatibility,
  makeWorkflowOperationFromCatalogBinding,
  searchWorkflowOperations
} from '../../src/workflows/workflow-operations';
import {
  WorkflowError,
  type WorkflowData,
  type WorkflowOperation,
  type WorkflowPortCompatibility,
  type WorkflowPortKind,
  type WorkflowRunContext
} from '../../src/workflows/workflow-types';
import {
  FIXTURE_KINDS,
  makeFixtureHost,
  makeFixturePoints,
  makeFixtureTable
} from './workflow-fixtures';

function makeContext(signal = new AbortController().signal): WorkflowRunContext {
  return {
    device: null,
    engine: 'cpu',
    host: makeFixtureHost(),
    signal,
    workflowId: 'operations-test',
    stepId: 'step'
  };
}

function getOperation(id: string): WorkflowOperation {
  const operation = getWorkflowOperationRegistry().get(id);
  if (!operation) throw new Error(`Missing operation ${id}`);
  return operation;
}

function makeBinding(overrides: Partial<CatalogOperationBinding> = {}): CatalogOperationBinding {
  return {
    id: 'test-catalog',
    title: 'Test catalog',
    domain: 'analysis',
    summary: 'Catalog adapter fixture',
    capabilityIds: ['fixture-capability'],
    contributorNames: ['SharedContributor', 'CpuContributor'],
    gpuContributorNames: ['SharedContributor', 'GpuContributor'],
    engine: 'cpu-reference',
    inputs: [
      {name: 'points', label: 'Points', accepts: ['point'], maximumFeatures: 2, maximumVertices: 2}
    ],
    parameters: [],
    outputs: [
      {
        tableId: 'result',
        label: 'Result',
        geometry: 'none',
        identityColumns: ['missing', 'value', 'point_id'],
        maximumRows: 2
      }
    ],
    cleanup: 'Plain arrays',
    declaredBehavior: ['Fixture'],
    limitation: 'Fixture only',
    run: () => ({
      tables: [
        makeFixtureTable('result', [
          {id: 'a', value: 1},
          {id: 'b', value: 1}
        ])
      ],
      diagnostics: [{code: 'fixture', severity: 'info', message: 'Fixture diagnostic', count: 2}],
      summary: {rows: 2}
    }),
    ...overrides
  };
}

async function runOperation(
  id: string,
  inputs: Record<string, WorkflowData>,
  parameters: Record<string, unknown> = {}
) {
  const operation = getOperation(id);
  // These numerical fixtures declare the coordinate convention assumed by their oracles.
  const declaredInputs = Object.fromEntries(
    Object.entries(inputs).map(([name, data]) => [
      name,
      {
        ...data,
        analysisProfile: applyKeplerDatasetProfileDeclarations(profileKeplerWorkbenchData(data), {
          coordinateModel: parameters['coordinateSystem'] === 'planar' ? 'planar' : 'geographic'
        })
      }
    ])
  );
  return operation.run(makeContext(), declaredInputs, operation.normalizeParameters(parameters));
}

function createDeferred<Value>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>(resolvePromise => {
    resolve = resolvePromise;
  });
  return {promise, resolve};
}

describe('catalog operation adapter and registry', () => {
  test('adapts every binding with stable identity and exact metadata, ports and defaults', () => {
    const requiredDefaultValues: string[] = [];
    for (const binding of KEPLER_CATALOG_BINDINGS) {
      const operation = makeWorkflowOperationFromCatalogBinding(binding);
      expect(makeWorkflowOperationFromCatalogBinding(binding)).toBe(operation);
      expect(operation).toMatchObject({
        id: binding.id,
        title: binding.title,
        summary: binding.summary,
        category: binding.domain,
        origin: 'catalog',
        version: 'catalog-ax-3',
        engine: binding.engine,
        capabilityIds: binding.capabilityIds,
        declaredBehavior: binding.declaredBehavior,
        limitation: binding.limitation,
        contributorNames: [
          ...new Set([...binding.contributorNames, ...binding.gpuContributorNames])
        ]
      });
      expect(operation.inputs).toEqual(
        binding.inputs.map(input => ({
          name: input.name,
          label: input.label,
          accepts:
            input.dataKind === 'table' || input.dataKind === 'raster'
              ? [input.dataKind]
              : input.accepts,
          ...(input.dataKind === 'table' ? {geometryFree: true} : {}),
          maximumRows: input.dataKind === 'raster' ? 0 : input.maximumFeatures,
          maximumVertices: input.maximumVertices,
          ...(input.dataKind === 'raster' ? {maximumCells: input.maximumCells} : {})
        }))
      );
      expect(operation.outputs).toEqual(
        binding.outputs.map(output => ({
          name: output.tableId,
          label: output.label,
          kind:
            output.dataKind === 'raster'
              ? 'raster'
              : output.geometry === 'none'
                ? 'table'
                : output.geometry === 'mixed'
                  ? 'features'
                  : output.geometry,
          identityColumns: output.identityColumns,
          maximumRows: output.maximumRows
        }))
      );
      let defaults;
      try {
        defaults = normalizeCatalogParameters(binding, {});
      } catch {
        requiredDefaultValues.push(binding.id);
        expect(() => operation.normalizeParameters({})).toThrow(WorkflowError);
        continue;
      }
      // Column lists are written back as `a, b`; the catalog trims each name, so compare the names.
      const readColumnLists = (values: Record<string, unknown>) =>
        Object.fromEntries(
          Object.entries(values).map(([name, value]) =>
            operation.parameters.find(parameter => parameter.name === name)?.kind === 'column-list'
              ? [
                  name,
                  String(value)
                    .split(',')
                    .map(entry => entry.trim())
                ]
              : [name, value]
          )
        );
      expect(readColumnLists(operation.normalizeParameters({})), binding.id).toEqual(
        readColumnLists(defaults)
      );
    }
    // Keep exceptions explicit: a new invalid catalog default must fail this spec.
    expect(requiredDefaultValues).toEqual([]);
  });

  test('full registry preserves source, table and catalog order with unique names', () => {
    const registry = getWorkflowOperationRegistry();
    expect(getWorkflowOperationRegistry()).toBe(registry);
    expect(
      registry
        .list()
        .slice(0, 3)
        .map(operation => operation.id)
    ).toEqual(['source-dataset', 'source-geojson', 'source-drawing']);
    const tableOperations = registry.list().slice(3, -KEPLER_CATALOG_BINDINGS.length);
    expect(tableOperations.every(operation => operation.category === 'table')).toBe(true);
    // The SQL step is listed after the table glue operations.
    expect(tableOperations.at(-1)?.id).toBe('table-sql');
    expect(registry.list().slice(-KEPLER_CATALOG_BINDINGS.length)).toEqual(
      KEPLER_CATALOG_BINDINGS.map(makeWorkflowOperationFromCatalogBinding)
    );
    expect(new Set(registry.list().map(operation => operation.id)).size).toBe(
      registry.list().length
    );
    for (const operation of registry.list()) {
      expect(registry.get(operation.id)).toBe(operation);
      for (const specs of [operation.inputs, operation.outputs, operation.parameters]) {
        expect(new Set(specs.map(spec => spec.name)).size, operation.id).toBe(specs.length);
      }
    }
    expect(registry.get('not-registered')).toBeUndefined();
  });

  test('registry snapshots its ordered list and refuses duplicate ids', () => {
    const first = makeWorkflowOperationFromCatalogBinding(makeBinding());
    const second = {...first, id: 'second'};
    const operations = [first, second];
    const registry = createWorkflowOperationRegistry(operations);
    operations.reverse();
    expect(registry.list()).toEqual([first, second]);
    expect(registry.get('second')).toBe(second);
    expect(() => createWorkflowOperationRegistry([first, {...first}])).toThrowError(
      expect.objectContaining({
        code: 'duplicate-id',
        message: 'Test catalog: Duplicate operation id "test-catalog".'
      })
    );
  });

  test('uses descriptor roles, leaves undescribed strings alone and preserves catalog limits', () => {
    expect(
      getOperation('kernel-density-grid').parameters.find(
        parameter => parameter.name === 'weightProperty'
      )
    ).toMatchObject({kind: 'column', columnsFrom: 'points'});
    const binding = makeBinding({
      parameters: [
        {name: 'valueProperty', label: 'Value', kind: 'string', defaultValue: ''},
        {name: 'valueProperties', label: 'Values', kind: 'string', defaultValue: ''},
        {
          name: 'wideProperties',
          label: 'Wide values',
          kind: 'string',
          maximumLength: 512,
          defaultValue: ''
        },
        {name: 'caption', label: 'Caption', kind: 'string', defaultValue: ''}
      ]
    });
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    expect(
      operation.parameters.map(parameter => [
        parameter.kind,
        parameter.maximumLength,
        parameter.columnsFrom
      ])
    ).toEqual([
      ['string', undefined, undefined],
      ['string', undefined, undefined],
      ['string', 512, undefined],
      ['string', undefined, undefined]
    ]);
    expect(operation.normalizeParameters({valueProperties: 'a,b'})['valueProperties']).toBe('a,b');
    expect(
      operation.normalizeParameters({valueProperties: 'x'.repeat(64)})['valueProperties']
    ).toHaveLength(64);
    expect(
      operation.normalizeParameters({wideProperties: 'x'.repeat(256)})['wideProperties']
    ).toHaveLength(256);
    expect(() => operation.normalizeParameters({valueProperties: 'x'.repeat(65)})).toThrowError(
      expect.objectContaining({
        code: 'parameter-out-of-range',
        message: 'Test catalog: Values must be at most 64 characters; got 65.'
      })
    );
    const multipleInputs = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      inputs: [...binding.inputs, {...binding.inputs[0], name: 'other'}]
    });
    expect(multipleInputs.parameters.every(parameter => parameter.columnsFrom === undefined)).toBe(
      true
    );
  });

  test('parameter refusals name the operation, label and numeric limit', () => {
    const operation = getOperation('random-points-in-polygon');
    expect(operation.normalizeParameters({pointsPerPolygon: '3'})['pointsPerPolygon']).toBe(3);
    expect(() => operation.normalizeParameters({pointsPerPolygon: 1001})).toThrowError(
      expect.objectContaining({
        code: 'parameter-out-of-range',
        message:
          'Random points in polygons: Points per polygon must be between 1 and 1000; got 1001.'
      })
    );
    expect(() => operation.normalizeParameters({unknown: true})).toThrowError(
      expect.objectContaining({code: 'unknown-parameter'})
    );
  });
});

describe('catalog runs', () => {
  test.each([
    'cpu',
    'auto'
  ] as const)('omits fallback reasons for CPU-only bindings with engine %s', async engine => {
    const operation = makeWorkflowOperationFromCatalogBinding(makeBinding());
    const result = await operation.run(
      {...makeContext(), engine},
      {points: makeFixturePoints(1)},
      {}
    );
    expect(result.engine).toBe('cpu-reference');
    expect(result.fallbackReason).toBeUndefined();
    expect(Object.hasOwn(result, 'fallbackReason')).toBe(false);
  });

  test.each([
    {engine: 'auto' as const, device: null, reason: 'No GPU device supplied'},
    {engine: 'cpu' as const, device: {} as Device, reason: 'CPU engine requested'}
  ])('preserves GPU-capable CPU fallback reasons: $reason', async ({engine, device, reason}) => {
    const binding = makeBinding();
    let gpuRunCount = 0;
    const operation = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      engine: 'gpu-or-cpu',
      runGpu: async () => {
        gpuRunCount++;
        return binding.run!({}, {});
      }
    });
    const result = await operation.run(
      {...makeContext(), engine, device},
      {points: makeFixturePoints(1)},
      {}
    );
    expect(result.engine).toBe('cpu-reference');
    expect(result.fallbackReason).toBe(reason);
    expect(gpuRunCount).toBe(0);
  });

  test('omits fallback reasons when a GPU-capable binding runs on the GPU', async () => {
    const binding = makeBinding();
    const operation = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      engine: 'gpu-or-cpu',
      runGpu: async () => binding.run!({}, {})
    });
    const result = await operation.run(
      {...makeContext(), engine: 'auto', device: {} as Device},
      {points: makeFixturePoints(1)},
      {}
    );
    expect(result.engine).toBe('gpu');
    expect(result.fallbackReason).toBeUndefined();
    expect(Object.hasOwn(result, 'fallbackReason')).toBe(false);
  });

  test.each([
    '__proto__',
    'constructor',
    'toString'
  ])('requires an own input for prototype-colliding port %s', async name => {
    const binding = makeBinding();
    const operation = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      inputs: [{...binding.inputs[0], name}]
    });
    await expect(operation.run(makeContext(), {}, {})).rejects.toMatchObject({
      name: 'WorkflowError',
      code: 'input-required',
      message: 'Test catalog: Input required; "Points" must contain at least one feature.'
    });
    const data = makeFixturePoints(1);
    const inheritedInputs = Object.create({[name]: data}) as Record<string, WorkflowData>;
    await expect(operation.run(makeContext(), inheritedInputs, {})).rejects.toMatchObject({
      name: 'WorkflowError',
      code: 'input-required'
    });
    const ownInputs = Object.fromEntries([[name, data]]);
    expect(
      (await operation.run(makeContext(), ownInputs, {})).outputs['result'].table.rowCount
    ).toBe(2);
  });

  test('runs GeoJSON polygon → seeded random points → Voronoi on the CPU', async () => {
    const source = await runOperation(
      'source-geojson',
      {},
      {
        text: JSON.stringify({
          type: 'Polygon',
          coordinates: [
            [
              [0, 0],
              [10, 0],
              [10, 10],
              [0, 10],
              [0, 0]
            ]
          ]
        })
      }
    );
    expect(source.outputs['data'].table.rowCount).toBe(1);
    const points = await runOperation(
      'random-points-in-polygon',
      {polygons: source.outputs['data']},
      {pointsPerPolygon: 8, seed: 23}
    );
    expect(points.outputs['random-points'].table.rowCount).toBe(8);
    expect(points.outputs['random-points'].identityColumn).toBe('point_id');
    const pointTable = points.outputs['random-points'].table;
    const sites = pointTable.columns[pointTable.geometryColumn!].map(value => {
      const geometry = JSON.parse(String(value)) as {type: string; coordinates: [number, number]};
      expect(geometry.type).toBe('Point');
      expect(geometry.coordinates).toHaveLength(2);
      for (const coordinate of geometry.coordinates) {
        expect(Number.isFinite(coordinate)).toBe(true);
        expect(coordinate).toBeGreaterThan(0);
        expect(coordinate).toBeLessThan(10);
      }
      return geometry.coordinates;
    });
    expect(new Set(sites.map(site => JSON.stringify(site))).size).toBe(8);
    const cells = await runOperation(
      'voronoi-diagram',
      {points: points.outputs['random-points']},
      {coordinateSystem: 'planar'}
    );
    expect(cells.outputs['voronoi-cells'].table.rowCount).toBe(8);
    expect(cells.outputs['voronoi-cells'].kind).toBe('polygon');
    expect(cells.outputs['voronoi-cells'].identityColumn).toBe('cell_id');
    const cellTable = cells.outputs['voronoi-cells'].table;
    const minimumX = Math.min(...sites.map(site => site[0]));
    const maximumX = Math.max(...sites.map(site => site[0]));
    const minimumY = Math.min(...sites.map(site => site[1]));
    const maximumY = Math.max(...sites.map(site => site[1]));
    const padding = Math.max(maximumX - minimumX, maximumY - minimumY) * 0.1;
    let totalArea = 0;
    for (let index = 0; index < cellTable.rowCount; index++) {
      const sourceIndex = pointTable.columns['point_id'].findIndex(
        identity => identity === cellTable.columns['source_id'][index]
      );
      expect(sourceIndex).toBeGreaterThanOrEqual(0);
      const site = sites[sourceIndex];
      const geometry = JSON.parse(String(cellTable.columns[cellTable.geometryColumn!][index])) as {
        type: string;
        coordinates: Array<Array<[number, number]>>;
      };
      expect(geometry.type).toBe('Polygon');
      expect(geometry.coordinates).toHaveLength(1);
      const ring = geometry.coordinates[0];
      expect(ring.length).toBeGreaterThanOrEqual(4);
      expect(ring[ring.length - 1]).toEqual(ring[0]);
      let twiceArea = 0;
      for (let vertexIndex = 0; vertexIndex < ring.length - 1; vertexIndex++) {
        const [x, y] = ring[vertexIndex];
        expect(Number.isFinite(x) && Number.isFinite(y)).toBe(true);
        expect(x).toBeGreaterThanOrEqual(minimumX - padding - 1e-9);
        expect(x).toBeLessThanOrEqual(maximumX + padding + 1e-9);
        expect(y).toBeGreaterThanOrEqual(minimumY - padding - 1e-9);
        expect(y).toBeLessThanOrEqual(maximumY + padding + 1e-9);
        // Every vertex must belong to its site's nearest-point region.
        const distanceToSite = (x - site[0]) ** 2 + (y - site[1]) ** 2;
        for (const other of sites) {
          expect(distanceToSite).toBeLessThanOrEqual(
            (x - other[0]) ** 2 + (y - other[1]) ** 2 + 1e-8
          );
        }
        const next = ring[vertexIndex + 1];
        twiceArea += x * next[1] - next[0] * y;
      }
      const area = Math.abs(twiceArea) / 2;
      expect(area).toBeGreaterThan(0);
      expect(cellTable.columns['area'][index]).toBeCloseTo(area, 10);
      totalArea += area;
    }
    expect(totalArea).toBeCloseTo(
      (maximumX - minimumX + 2 * padding) * (maximumY - minimumY + 2 * padding),
      10
    );
    expect(points.engine).toBe('cpu-reference');
    expect(cells.engine).toBe('cpu-reference');
    expect(cells.fallbackReason).toBeUndefined();
    expect(cells.dispatchedContributors).toEqual([]);
    expect(
      await runOperation(
        'random-points-in-polygon',
        {polygons: source.outputs['data']},
        {pointsPerPolygon: 8, seed: 23}
      )
    ).toEqual(points);
  });

  async function makeDensityCells(cellSize: number) {
    const points = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}},
          {geometry: {type: 'Point', coordinates: [10, 10]}, properties: {}}
        ]
      },
      {id: 'points', label: 'Points'}
    );
    return runOperation(
      'kernel-density-grid',
      {points},
      {cellSize, bandwidth: 2, includeEmptyCells: true, coordinateSystem: 'planar'}
    );
  }

  test('joins keyed Gi* results back to their density cells', async () => {
    const density = await makeDensityCells(5);
    const cells = density.outputs['density-cells'];
    expect(cells.table.rowCount).toBe(9);
    // Gaussian h=2 at the nine cell centers; contributions beyond radius 10 are zero.
    const nearDensity = Math.exp(-12.5 / 8) / (8 * Math.PI);
    const farDensity = Math.exp(-62.5 / 8) / (8 * Math.PI);
    const expectedDensities = [
      nearDensity,
      2 * farDensity,
      farDensity,
      2 * farDensity,
      nearDensity,
      nearDensity,
      farDensity,
      nearDensity,
      nearDensity
    ];
    expectedDensities.forEach((value, index) => {
      expect(cells.table.columns['density'][index]).toBeCloseTo(value, 14);
      const column = index % 3;
      const row = Math.floor(index / 3);
      const geometry: unknown = JSON.parse(
        String(cells.table.columns[cells.table.geometryColumn!][index])
      );
      expect(geometry).toEqual({
        type: 'Polygon',
        coordinates: [
          [
            [column * 5, row * 5],
            [(column + 1) * 5, row * 5],
            [(column + 1) * 5, (row + 1) * 5],
            [column * 5, (row + 1) * 5],
            [column * 5, row * 5]
          ]
        ]
      });
    });
    const hotspots = await runOperation(
      'hotspot-gi-star',
      {features: cells},
      {valueProperty: 'density', coordinateSystem: 'planar', neighborCount: 2}
    );
    const attributes = hotspots.outputs['hotspots'];
    expect(attributes.table.rowCount).toBe(9);
    // Fixed grid neighbors, including self in each Gi* sum. Distance ties are broken by the
    // catalog's canonical row order (cell centre x, then y), so the result does not depend on row order.
    const neighborhoods = [
      [0, 3, 1],
      [1, 0, 4],
      [2, 1, 5],
      [3, 0, 6],
      [4, 3, 1],
      [5, 4, 2],
      [6, 3, 7],
      [7, 6, 4],
      [8, 7, 5]
    ];
    const meanDensity = expectedDensities.reduce((sum, value) => sum + value, 0) / 9;
    const centeredDensities = expectedDensities.map(value => value - meanDensity);
    const sampleVariance = centeredDensities.reduce((sum, value) => sum + value ** 2, 0) / 8;
    const standardError = Math.sqrt(sampleVariance * 3 * (1 - 3 / 9));
    neighborhoods.forEach((neighbors, index) => {
      const expectedScore =
        neighbors.reduce((sum, neighbor) => sum + centeredDensities[neighbor], 0) / standardError;
      expect(attributes.table.columns['z_score'][index]).toBeCloseTo(expectedScore, 12);
      expect(attributes.table.columns['neighbor_count'][index]).toBe(2);
    });
    expect(attributes.table.columns['classification']).toEqual([
      'not-significant',
      'not-significant',
      'not-significant',
      'not-significant',
      'not-significant',
      'not-significant',
      'not-significant',
      'not-significant',
      'hot-90'
    ]);
    // Two-sided normal tails computed independently with Python math.erfc(abs(z) / sqrt(2)).
    const expectedProbabilities = [
      0.3724846266350402, 0.6537837522474057, 0.37109377297031837, 0.37109377297031837,
      0.3724846266350402, 0.6556590289832656, 0.37109377297031837, 0.6556590289832656,
      0.07363851315272524
    ];
    expectedProbabilities.forEach((probability, index) => {
      expect(attributes.table.columns['p_value'][index]).toBeCloseTo(probability, 6);
    });
    expect(attributes.identityColumn).toBe('source_index');
    expect(attributes.table.columns['source_id']).toEqual(
      cells.table.columns['cell_id'].map(String)
    );
    expect(getOperation('hotspot-gi-star').outputs[0].kind).toBe('table');
    const joined = await runOperation(
      'table-join-attributes',
      {target: cells, attributes},
      {columns: 'z_score, classification', prefix: 'hotspot_'}
    );
    expect(joined.outputs['joined'].table.rowCount).toBe(9);
    expect(joined.outputs['joined'].kind).toBe('polygon');
    expect(joined.outputs['joined'].table.columns['hotspot_z_score']).toEqual(
      attributes.table.columns['z_score']
    );
    expect(joined.outputs['joined'].table.columns['hotspot_classification']).toEqual(
      attributes.table.columns['classification']
    );
  });

  test('refuses density grids exceeding the Gi* feature capacity before analysis', async () => {
    const density = await makeDensityCells(0.1);
    expect(density.outputs['density-cells'].table.rowCount).toBe(10201);
    await expect(
      runOperation(
        'hotspot-gi-star',
        {features: density.outputs['density-cells']},
        {valueProperty: 'density'}
      )
    ).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message:
        'Hot spot analysis (Gi*): Input feature capacity exceeded. Limits: Features 3,000 features, 200,000 vertices.'
    });
  });

  test('rejects tables, wrong geometry, missing and empty inputs with exact codes', async () => {
    const operation = getOperation('voronoi-diagram');
    const parameters = operation.normalizeParameters({});
    await expect(
      operation.run(
        makeContext(),
        {points: {kind: 'table', table: makeFixtureTable('plain', [{id: 'one', value: 1}])}},
        parameters
      )
    ).rejects.toMatchObject({
      code: 'geometry-required',
      message:
        'Voronoi diagram: "Points" needs geometry; the connected output is a table without a geometry column. Join it to features with Join attributes first.'
    });
    const polygons = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 0]
                ]
              ]
            },
            properties: {}
          }
        ]
      },
      {id: 'polygons', label: 'Polygons'}
    );
    await expect(
      operation.run(makeContext(), {points: polygons}, parameters)
    ).rejects.toMatchObject({
      code: 'geometry-kind-not-accepted',
      message: 'Voronoi diagram: Input geometry kind not accepted; "Points" accepts point.'
    });
    await expect(operation.run(makeContext(), {}, parameters)).rejects.toMatchObject({
      code: 'input-required',
      message: 'Voronoi diagram: Input required; "Points" must contain at least one feature.'
    });
    const empty = {
      kind: 'point' as const,
      table: {
        ...makeFixturePoints(0).table,
        geometryColumn: 'geometry',
        columns: {geometry: []},
        columnTypes: {geometry: 'string' as const}
      }
    };
    await expect(operation.run(makeContext(), {points: empty}, parameters)).rejects.toMatchObject({
      code: 'input-required'
    });
  });

  test('selects the first existing distinct identity column and passes through provenance', async () => {
    const binding = makeBinding();
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    const result = await operation.run(makeContext(), {points: makeFixturePoints(2)}, {});
    expect(result.outputs['result'].identityColumn).toBe('point_id');
    expect(result.outputs['result'].kind).toBe('table');
    expect(result.diagnostics).toEqual(binding.run!({}, {}).diagnostics);
    expect(result.summary).toEqual({rows: 2});
    const noIdentity = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      outputs: [{...binding.outputs[0], identityColumns: ['missing', 'value']}]
    });
    expect(
      (await noIdentity.run(makeContext(), {points: makeFixturePoints(2)}, {})).outputs['result']
        .identityColumn
    ).toBeUndefined();
    const gpu = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      engine: 'gpu',
      run: undefined,
      runGpu: async () => binding.run!({}, {})
    });
    const gpuResult = await gpu.run(
      {...makeContext(), engine: 'auto', device: {} as Device},
      {points: makeFixturePoints(2)},
      {}
    );
    expect(gpuResult.engine).toBe('gpu');
    expect(gpuResult.dispatchedContributors).toEqual(binding.gpuContributorNames);
    expect(gpuResult.fallbackReason).toBeUndefined();
  });

  test('maps capacity and GPU refusals while preserving unrelated errors', async () => {
    const binding = makeBinding();
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    await expect(
      operation.run(makeContext(), {points: makeFixturePoints(3)}, {})
    ).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message:
        'Test catalog: Input feature capacity exceeded. Limits: Points 2 features, 2 vertices.'
    });
    const vertexLimit = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      inputs: [{...binding.inputs[0], maximumVertices: 1}]
    });
    await expect(
      vertexLimit.run(makeContext(), {points: makeFixturePoints(2)}, {})
    ).rejects.toMatchObject({
      code: 'capacity-exceeded',
      message:
        'Test catalog: Input vertex capacity exceeded. Limits: Points 2 features, 1 vertices.'
    });
    const gpu = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      engine: 'gpu',
      run: undefined
    });
    await expect(gpu.run(makeContext(), {points: makeFixturePoints(1)}, {})).rejects.toMatchObject({
      code: 'gpu-required',
      message: 'Test catalog: GPU device required.'
    });
    for (const failure of [
      new Error('Unrelated failure'),
      new KeplerAllocationError('memory-budget')
    ]) {
      const failing = makeWorkflowOperationFromCatalogBinding({
        ...binding,
        run: () => {
          throw failure;
        }
      });
      await expect(failing.run(makeContext(), {points: makeFixturePoints(1)}, {})).rejects.toBe(
        failure
      );
    }
  });

  test('runs at the exact feature, vertex and output limits and refuses output overflow', async () => {
    const binding = makeBinding();
    let executedFeatureCount = 0;
    const operation = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      run: inputs => {
        executedFeatureCount = inputs['points'].features.length;
        return binding.run!(inputs, {});
      }
    });
    const inputs = {points: makeFixturePoints(2)};
    const result = await operation.run(makeContext(), inputs, {});
    expect(executedFeatureCount).toBe(2);
    expect(result.outputs['result'].table).toEqual(
      makeFixtureTable('result', [
        {id: 'a', value: 1},
        {id: 'b', value: 1}
      ])
    );
    const outputOverflow = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      outputs: [{...binding.outputs[0], maximumRows: 1}]
    });
    await expect(outputOverflow.run(makeContext(), inputs, {})).rejects.toMatchObject({
      name: 'WorkflowError',
      code: 'capacity-exceeded',
      message: 'Test catalog: Output capacity exceeded. Limits: Points 2 features, 2 vertices.'
    });
  });

  test.each([
    'cancelled-first',
    'successful-first',
    'same-turn'
  ])('isolates cancellation and results of concurrent pending GPU runs (%s)', async completionOrder => {
    const firstOutput = createDeferred<CatalogRunOutput>();
    const secondOutput = createDeferred<CatalogRunOutput>();
    const startedFeatureCounts: number[] = [];
    const binding = makeBinding();
    const operation = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      engine: 'gpu',
      run: undefined,
      runGpu: async (_context, inputs) => {
        const featureCount = inputs['points'].features.length;
        startedFeatureCounts.push(featureCount);
        return featureCount === 1 ? firstOutput.promise : secondOutput.promise;
      }
    });
    const controller = new AbortController();
    const firstContext = {
      ...makeContext(controller.signal),
      engine: 'auto' as const,
      device: {} as Device
    };
    const secondContext = {...makeContext(), engine: 'auto' as const, device: {} as Device};
    const cancelledRun = operation.run(firstContext, {points: makeFixturePoints(1)}, {});
    const cancellationAssertion = expect(cancelledRun).rejects.toMatchObject({
      name: 'WorkflowError',
      code: 'cancelled',
      message: 'Test catalog: Execution was cancelled.'
    });
    const successfulRun = operation.run(secondContext, {points: makeFixturePoints(2)}, {});
    expect(startedFeatureCounts).toEqual([1, 2]);
    const firstResult = {...binding.run!({}, {}), summary: {marker: 'cancelled result'}};
    const secondResult = {
      ...binding.run!({}, {}),
      tables: [makeFixtureTable('result', [{id: 'second', value: 42}])],
      summary: {marker: 'successful result'},
      dispatchedContributors: ['SecondGpuContributor']
    };
    if (completionOrder === 'cancelled-first') {
      controller.abort();
      firstOutput.resolve(firstResult);
      await cancellationAssertion;
      secondOutput.resolve(secondResult);
    } else if (completionOrder === 'successful-first') {
      secondOutput.resolve(secondResult);
      await successfulRun;
      controller.abort();
      firstOutput.resolve(firstResult);
    } else {
      controller.abort();
      firstOutput.resolve(firstResult);
      secondOutput.resolve(secondResult);
    }
    const result = await successfulRun;
    await cancellationAssertion;
    expect(result.outputs['result'].table).toEqual(secondResult.tables[0]);
    expect(result.outputs['result'].identityColumn).toBe('value');
    expect(result.summary).toEqual({marker: 'successful result'});
    expect(result.engine).toBe('gpu');
    expect(result.dispatchedContributors).toEqual(['SecondGpuContributor']);
    expect(result.fallbackReason).toBeUndefined();
  });

  test('checks cancellation both before execution and after the catalog returns', async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      getOperation('voronoi-diagram').run(makeContext(controller.signal), {}, {})
    ).rejects.toMatchObject({
      code: 'cancelled',
      message: 'Voronoi diagram: Execution was cancelled.'
    });
    const duringRun = new AbortController();
    const binding = makeBinding();
    const operation = makeWorkflowOperationFromCatalogBinding({
      ...binding,
      run: () => {
        duringRun.abort();
        return binding.run!({}, {});
      }
    });
    await expect(
      operation.run(makeContext(duringRun.signal), {points: makeFixturePoints(1)}, {})
    ).rejects.toMatchObject({code: 'cancelled', message: 'Test catalog: Execution was cancelled.'});
  });
});

describe('palette search and compatibility', () => {
  const base = makeWorkflowOperationFromCatalogBinding(makeBinding());
  function makeSearchOperation(
    id: string,
    overrides: Partial<WorkflowOperation>
  ): WorkflowOperation {
    return {
      ...base,
      id,
      title: 'Unrelated',
      summary: '',
      contributorNames: [],
      capabilityIds: [],
      category: 'geometry',
      ...overrides
    };
  }

  test('scores each field and sorts by score, then title', () => {
    const operations = [
      makeSearchOperation('summary', {title: 'Zulu', summary: 'Buffer data'}),
      makeSearchOperation('category', {category: 'analysis'}),
      makeSearchOperation('capability', {capabilityIds: ['buffer-family']}),
      makeSearchOperation('contributor', {contributorNames: ['GPUBuffer']}),
      makeSearchOperation('buffer-id', {}),
      makeSearchOperation('substring', {title: 'Unbuffered'}),
      makeSearchOperation('word', {title: 'Point buffer'}),
      makeSearchOperation('prefix', {title: 'Buffer points'}),
      makeSearchOperation('summary-alpha', {title: 'Alpha', summary: 'buffer'})
    ];
    const registry = createWorkflowOperationRegistry(operations);
    expect(
      searchWorkflowOperations(registry, 'BuFfEr').map(match => [match.operation.id, match.score])
    ).toEqual([
      ['prefix', 100],
      ['word', 80],
      ['substring', 60],
      ['buffer-id', 50],
      ['contributor', 40],
      ['capability', 30],
      ['summary-alpha', 10],
      ['summary', 10]
    ]);
    expect(
      searchWorkflowOperations(registry, 'analysis').map(match => [match.operation.id, match.score])
    ).toEqual([['category', 25]]);
  });

  test('requires every word, adds scores, and applies exact category filters', () => {
    const registry = createWorkflowOperationRegistry([
      makeSearchOperation('one', {
        title: 'Buffer points',
        summary: 'distance meters',
        category: 'geometry'
      }),
      makeSearchOperation('two', {title: 'Buffer lines', category: 'network'})
    ]);
    expect(
      searchWorkflowOperations(registry, ' buffer  METERS ').map(match => [
        match.operation.id,
        match.score
      ])
    ).toEqual([['one', 110]]);
    expect(searchWorkflowOperations(registry, 'buffer missing')).toEqual([]);
    expect(
      searchWorkflowOperations(registry, 'buffer', {category: 'network'}).map(
        match => match.operation.id
      )
    ).toEqual(['two']);
  });

  test('compatibility filters on the first usable input, ranking yes before maybe at equal score', () => {
    const registry = createWorkflowOperationRegistry([
      makeSearchOperation('maybe', {
        title: 'Alpha',
        inputs: [{...base.inputs[0], accepts: ['point']}]
      }),
      makeSearchOperation('yes', {
        title: 'Zulu',
        inputs: [{...base.inputs[0], accepts: ['table']}]
      }),
      makeSearchOperation('source', {inputs: []}),
      makeSearchOperation('second', {
        title: 'Omega',
        inputs: [
          {...base.inputs[0], accepts: ['line']},
          {...base.inputs[0], name: 'second', accepts: ['point']}
        ]
      })
    ]);
    expect(
      searchWorkflowOperations(registry, '', {compatibleWith: 'table'}).map(match => [
        match.operation.id,
        match.inputName,
        match.compatibility
      ])
    ).toEqual([
      ['yes', 'points', 'yes'],
      ['maybe', 'points', 'maybe'],
      ['second', 'points', 'maybe']
    ]);
    expect(
      searchWorkflowOperations(registry, '', {compatibleWith: 'point'}).map(match => [
        match.operation.id,
        match.inputName,
        match.compatibility
      ])
    ).toEqual([
      ['maybe', 'points', 'yes'],
      ['yes', 'points', 'yes'],
      ['second', 'second', 'yes']
    ]);
    expect(
      searchWorkflowOperations(registry, '', {compatibleWith: 'polygon'}).map(
        match => match.operation.id
      )
    ).toEqual(['yes']);
  });

  test('empty query preserves order and limit defaults to 200', () => {
    const operations = Array.from({length: 205}, (_, index) =>
      makeSearchOperation(`operation-${index}`, {title: `Title ${205 - index}`})
    );
    const registry = createWorkflowOperationRegistry(operations);
    expect(searchWorkflowOperations(registry, ' \t ')).toEqual(
      operations.slice(0, 200).map(operation => ({operation, score: 0}))
    );
    expect(
      searchWorkflowOperations(registry, '', {limit: 2}).map(match => match.operation)
    ).toEqual(operations.slice(0, 2));
    expect(searchWorkflowOperations(registry, '', {limit: 0})).toEqual([]);
    expect(searchWorkflowOperations(registry, '', {limit: 300})).toHaveLength(205);
  });

  test('compatibility matrix covers all five output kinds', () => {
    const cases: Array<{accepts: WorkflowPortKind[]; expected: WorkflowPortCompatibility[]}> = [
      {accepts: [], expected: ['no', 'no', 'no', 'no', 'no']},
      {accepts: ['table'], expected: ['yes', 'yes', 'yes', 'yes', 'yes']},
      {accepts: ['features'], expected: ['yes', 'yes', 'yes', 'yes', 'maybe']},
      {accepts: ['point'], expected: ['yes', 'no', 'no', 'maybe', 'maybe']},
      {accepts: ['line', 'polygon'], expected: ['no', 'yes', 'yes', 'maybe', 'maybe']},
      {accepts: ['point', 'line', 'polygon'], expected: ['yes', 'yes', 'yes', 'maybe', 'maybe']},
      {accepts: ['table', 'point'], expected: ['yes', 'yes', 'yes', 'yes', 'yes']}
    ];
    for (const {accepts, expected} of cases) {
      expect(
        FIXTURE_KINDS.map(kind => getWorkflowPortCompatibility(kind, {accepts})),
        accepts.join(',')
      ).toEqual(expected);
    }
  });
});

test('raster ports accept only raster payloads and cannot feed table or geometry operations', () => {
  for (const kind of ['point', 'line', 'polygon', 'features', 'table'] as const) {
    expect(getWorkflowPortCompatibility('raster', {accepts: [kind]})).toBe('no');
    expect(getWorkflowPortCompatibility(kind, {accepts: ['raster']})).toBe(
      kind === 'features' ? 'maybe' : 'no'
    );
  }
  expect(getWorkflowPortCompatibility('raster', {accepts: ['raster']})).toBe('yes');
  expect(getWorkflowPortCompatibility('table', {accepts: ['table', 'raster']})).toBe('yes');
});

test('geometry-free catalog ports reject known geometry without changing generic table ports', () => {
  const scalarInput = {accepts: ['table'] as const, geometryFree: true};
  for (const kind of ['point', 'line', 'polygon', 'raster'] as const) {
    expect(getWorkflowPortCompatibility(kind, scalarInput)).toBe('no');
  }
  expect(getWorkflowPortCompatibility('table', scalarInput)).toBe('yes');
  // Source ports declare features until the host has read their actual representation.
  expect(getWorkflowPortCompatibility('features', scalarInput)).toBe('maybe');
  expect(getWorkflowPortCompatibility('point', {accepts: ['table']})).toBe('yes');
});
