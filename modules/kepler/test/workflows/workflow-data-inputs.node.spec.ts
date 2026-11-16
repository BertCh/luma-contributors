// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {describe, expect, test, vi} from 'vitest';
import type {CatalogOperationBinding} from '../../src/analysis/catalog-binding-types';
import {
  matchCatalogInvocationInputs,
  runCatalogOperation,
  runCatalogOperationAsync,
  serializeCatalogInvocation
} from '../../src/analysis/catalog-bindings';
import {makeCatalogTableInput} from '../../src/analysis/catalog-data-input';
import type {CatalogOutputTable} from '../../src/analysis/catalog-geojson';
import {
  areCatalogRasterGridsAligned,
  makeCatalogRasterInput,
  validateCatalogRasterInput
} from '../../src/analysis/catalog-raster-input';
import {
  getWorkflowCatalogInput,
  getWorkflowFeatures,
  hashWorkflowData,
  makeWorkflowData,
  makeWorkflowRasterData
} from '../../src/workflows/workflow-data';
import {
  createWorkflowOperationRegistry,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import type {WorkflowDocument, WorkflowRunContext} from '../../src/workflows/workflow-types';
import {deriveKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {makeFixtureHost, makeFixtureRegistry} from './workflow-fixtures';
import {
  evaluateKeplerOperationApplicability,
  isKeplerDatasetCompatibleWithInput
} from '../../src/descriptors/applicability';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {profileKeplerColumnTable} from '../../src/profile/dataset-profile';
import {getWorkflowSourceOperations} from '../../src/workflows/workflow-source-operations';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../../examples/deck/kepler-compute/src/raster-tile-ingest';
import {
  makeWorkbenchRasterWorkflowData,
  readWorkbenchRasterInput
} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-raster-input';

function makeFlowTable(): CatalogOutputTable {
  return {
    id: 'flows',
    label: 'OD flows',
    rowCount: 2,
    columns: {
      row_id: ['first', 'second'],
      origin: ['8928308280fffff', '8928308280bffff'],
      destination: ['8928308280bffff', '8928308280fffff'],
      weight: [12, 7]
    },
    columnTypes: {row_id: 'string', origin: 'string', destination: 'string', weight: 'float64'}
  };
}

function makeRaster() {
  return makeCatalogRasterInput(
    {
      id: 'elevation',
      grid: makeGPURasterMetadata({width: 2, height: 2, cellSize: [10, 20], origin: [100, 240]}),
      band: {
        index: 1,
        values: new Float32Array([4, 3, -999, 1]),
        noDataValue: -999,
        validity: new Uint8Array([1, 1, 0, 1])
      },
      crs: 'planar',
      cellSizeMode: 'uniform'
    },
    4
  );
}

function makeBinding(kind: 'table' | 'raster'): CatalogOperationBinding {
  return {
    id: `typed-${kind}`,
    title: `Typed ${kind}`,
    domain: 'analysis',
    summary: 'Typed boundary fixture',
    capabilityIds: [],
    contributorNames: [],
    gpuContributorNames: [],
    engine: 'cpu-reference',
    inputs: [
      {
        name: 'data',
        label: 'Data',
        dataKind: kind,
        accepts: [],
        maximumFeatures: kind === 'table' ? 2 : 0,
        maximumVertices: 0,
        ...(kind === 'raster' ? {maximumCells: 4} : {})
      }
    ],
    parameters: [],
    outputs: [
      {
        tableId: 'result',
        label: 'Result',
        geometry: 'none',
        identityColumns: ['row_id'],
        maximumRows: 2
      }
    ],
    cleanup: 'Caller owns decoded arrays',
    declaredBehavior: ['Shape checked before dispatch'],
    limitation: 'Dispatch fixture only',
    runData: () => ({
      tables: [{...makeFlowTable(), id: 'result'}],
      diagnostics: [],
      summary: {rows: 2}
    })
  };
}

function makeContext(): WorkflowRunContext {
  return {
    device: null,
    engine: 'cpu',
    host: makeFixtureHost(),
    signal: new AbortController().signal,
    workflowId: 'typed',
    stepId: 'analysis'
  };
}

describe('geometry-free table inputs', () => {
  test('keeps exact OD keys, row identities, nulls and array ownership without GeoJSON', async () => {
    const table = makeFlowTable();
    const input = makeCatalogTableInput(table, {maximumRows: 2, identityColumn: 'row_id'});
    expect(input.table).toBe(table);
    const data = makeWorkflowData(table, {identityColumn: 'row_id'});
    expect(getWorkflowCatalogInput(data, 2)).toEqual(input);
    expect(() => getWorkflowFeatures(data)).toThrow(/geometry column/);
    const binding = makeBinding('table');
    const runData = vi.fn(binding.runData!);
    binding.runData = runData;
    expect(runCatalogOperation(binding, {}, {data: input}).engine).toBe('cpu-reference');
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    expect(operation.inputs[0]).toMatchObject({accepts: ['table'], maximumRows: 2});
    const result = await operation.run(makeContext(), {data}, {});
    expect(runData.mock.calls[1][0]['data']).toEqual(input);
    expect(result.outputs['result'].identityColumn).toBe('row_id');
    expect(result.outputs['result'].table.columns['origin']).toEqual(table.columns['origin']);
    expect(deriveKeplerOperationDescriptor(binding).inputs[0].kind).toBe('attributes');
    const nullable = {...table, columns: {...table.columns, weight: [null, 7]}};
    expect(makeCatalogTableInput(nullable, {maximumRows: 2}).table.columns['weight']).toEqual([
      null,
      7
    ]);
  });

  test('rejects overflow, mismatched columns, unknown identity and hidden geometry before execution', () => {
    const table = makeFlowTable();
    expect(() => makeCatalogTableInput(table, {maximumRows: 1})).toThrow(
      /flows table.rowCount.*\[0, 1\]/
    );
    expect(() =>
      makeCatalogTableInput({...table, columns: {...table.columns, weight: [1]}}, {maximumRows: 2})
    ).toThrow(/weight length/);
    expect(() =>
      makeCatalogTableInput({...table, columnTypes: {row_id: 'string'}}, {maximumRows: 2})
    ).toThrow(/same columns/);
    expect(() => makeCatalogTableInput(table, {maximumRows: 2, identityColumn: 'missing'})).toThrow(
      /identityColumn/
    );
    expect(() =>
      makeCatalogTableInput({...table, geometryColumn: 'origin'}, {maximumRows: 2})
    ).toThrow(/geometryColumn must be absent/);
    const binding = makeBinding('table');
    binding.runData = vi.fn(binding.runData!);
    const input = makeCatalogTableInput(table, {maximumRows: 2});
    expect(() =>
      runCatalogOperation(
        {...binding, inputs: [{...binding.inputs[0], maximumFeatures: 1}]},
        {},
        {data: input}
      )
    ).toThrow(/typed-table: data table.rowCount/);
    expect(binding.runData).not.toHaveBeenCalled();
    expect(() =>
      runCatalogOperation(binding, {}, {data: {type: 'FeatureCollection', features: []}})
    ).toThrow(/geometry-free table/);
  });
});

describe('bounded raster inputs', () => {
  test('preserves grid, nodata, validity and decoded samples through catalog and workflow dispatch', async () => {
    const raster = makeRaster();
    const data = makeWorkflowRasterData(raster, 4);
    expect(data.kind).toBe('raster');
    expect(data.table.rowCount).toBe(0);
    expect(Object.keys(data.table.columns)).toEqual([]);
    expect(getWorkflowCatalogInput(data, 0)).toBe(raster);
    expect(() => getWorkflowFeatures(data)).toThrow(/raster cells/);
    const binding = makeBinding('raster');
    binding.runData = vi.fn(binding.runData!);
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    expect(operation.inputs[0]).toMatchObject({
      accepts: ['raster'],
      maximumRows: 0,
      maximumCells: 4
    });
    await operation.run(makeContext(), {data}, {});
    expect(vi.mocked(binding.runData).mock.calls[0][0]['data']).toBe(raster);
    expect(raster.grid.affine).toEqual([10, 0, 100, 0, -20, 240]);
    expect(raster.band.validity).toEqual(new Uint8Array([1, 1, 0, 1]));
    expect(deriveKeplerOperationDescriptor(binding).inputs[0].kind).toBe('surface');
    expect(deriveKeplerOperationDescriptor(binding).requirements).toEqual([]);
  });

  test('refuses unbounded, oversized, malformed and CRS-inconsistent grids without scanning samples', () => {
    const raster = makeRaster();
    expect(() => validateCatalogRasterInput('raster', raster, 3)).toThrow(
      /grid cell count must be at most 3/
    );
    expect(() => validateCatalogRasterInput('raster', raster, 0)).toThrow(/maximumCells/);
    expect(() =>
      validateCatalogRasterInput('raster', {...raster, grid: {...raster.grid, width: 1.5}}, 4)
    ).toThrow(/grid.width/);
    expect(() =>
      validateCatalogRasterInput(
        'raster',
        {...raster, band: {...raster.band, validity: new Uint8Array(1)}},
        4
      )
    ).toThrow(/band.validity length/);
    expect(() =>
      validateCatalogRasterInput(
        'raster',
        {...raster, band: {...raster.band, values: new Float32Array(1)}},
        4
      )
    ).toThrow(/band.values length/);
    expect(() => validateCatalogRasterInput('raster', {...raster, crs: 'EPSG:4326'}, 4)).toThrow(
      /crs and cellSizeMode/
    );
    expect(() =>
      validateCatalogRasterInput(
        'raster',
        {...raster, grid: {...raster.grid, affine: [1, 0, 0, 0, 0, 0]}},
        4
      )
    ).toThrow(/invertible/);
    expect(() =>
      validateCatalogRasterInput(
        'raster',
        {...raster, grid: {...raster.grid, affine: [1, 0, Infinity, 0, -1, 0]}},
        4
      )
    ).toThrow(/six finite/);
    const binding = makeBinding('raster');
    expect(() =>
      runCatalogOperation(
        {...binding, inputs: [{...binding.inputs[0], maximumCells: undefined}]},
        {},
        {data: raster}
      )
    ).toThrow(/maximumCells/);
  });

  test('content keys include samples, validity, grid, CRS and exact table identities', () => {
    const raster = makeRaster();
    const hash = hashWorkflowData(makeWorkflowRasterData(raster, 4));
    for (const changed of [
      {...raster, band: {...raster.band, values: new Float32Array([8, 3, -999, 1])}},
      {...raster, band: {...raster.band, validity: new Uint8Array([1, 0, 0, 1])}},
      {...raster, grid: {...raster.grid, affine: [10, 0, 101, 0, -20, 240] as const}},
      {...raster, crs: 'EPSG:3857' as const, cellSizeMode: 'web-mercator' as const}
    ])
      expect(hashWorkflowData(makeWorkflowRasterData(changed, 4))).not.toBe(hash);
    expect(areCatalogRasterGridsAligned(raster, structuredClone(raster))).toBe(true);
    expect(
      areCatalogRasterGridsAligned(raster, {
        ...raster,
        grid: {...raster.grid, pixelInterpretation: 'point'}
      })
    ).toBe(false);
    expect(
      areCatalogRasterGridsAligned(raster, {
        ...raster,
        grid: {...raster.grid, affine: [10, 0, 101, 0, -20, 240]}
      })
    ).toBe(false);
    const table = makeFlowTable();
    expect(hashWorkflowData(makeWorkflowData(table, {identityColumn: 'row_id'}))).not.toBe(
      hashWorkflowData(makeWorkflowData(table))
    );
    const binding = makeBinding('raster');
    const invocation = JSON.parse(serializeCatalogInvocation(binding, {}, {data: raster}));
    expect(invocation.inputs['data']).toMatchObject({
      dataKind: 'raster',
      cellCount: 4,
      featureCount: 0,
      vertexCount: 0
    });
    expect(matchCatalogInvocationInputs(invocation, {data: raster})).toBe(true);
    expect(
      matchCatalogInvocationInputs(invocation, {
        data: {...raster, band: {...raster.band, validity: new Uint8Array(4)}}
      })
    ).toBe(false);
  });

  test('typed GPU callback selection and fallback preserve input objects (mock dispatch only)', async () => {
    const raster = makeRaster();
    const binding = makeBinding('raster');
    const output = binding.runData!({data: raster}, {});
    const runDataGpu = vi.fn(async () => ({
      ...output,
      dispatchedContributors: ['FixtureContributor']
    }));
    const gpuBinding = {...binding, engine: 'gpu-or-cpu' as const, runDataGpu};
    const result = await runCatalogOperationAsync(
      gpuBinding,
      {},
      {data: raster},
      {device: {} as Device}
    );
    expect(result.engine).toBe('gpu');
    expect(result.dispatchedContributors).toEqual(['FixtureContributor']);
    expect(runDataGpu.mock.calls[0]).toEqual([{device: {}}, {data: raster}, {}]);
    const failure = new Error('fixture GPU failure');
    const failing = {
      ...gpuBinding,
      runDataGpu: async () => {
        throw failure;
      }
    };
    const fallback = await runCatalogOperationAsync(
      failing,
      {},
      {data: raster},
      {device: {} as Device}
    );
    expect(fallback.engine).toBe('cpu-reference');
    expect(fallback.fallbackReason).toBe('fixture GPU failure');
    expect(fallback.diagnostics[0].code).toBe('gpu-fallback');
    await expect(
      runCatalogOperationAsync(
        {...failing, runData: undefined},
        {},
        {data: raster},
        {device: {} as Device}
      )
    ).rejects.toBe(failure);
  });
});

describe('host numeric raster adapter', () => {
  test.each([
    'EPSG:4326',
    'EPSG:3857'
  ] as const)('reuses ingested bands and north-up extent for %s', crs => {
    const tile = createNumericRasterTile({
      id: 'tile',
      width: 2,
      height: 2,
      extent: [100, 20, 120, 60],
      bands: [{values: new Int16Array([4, -999, 2, 1]), noDataValue: -999}]
    });
    const source = createNumericRasterSource({
      id: 'source',
      kind: 'single-raster',
      crs,
      tiles: [tile]
    });
    const raster = readWorkbenchRasterInput(source, {bandIndex: 1, maximumCells: 4});
    expect(raster.band).toBe(tile.bands[0]);
    expect(raster.band.values).toBe(tile.bands[0].values);
    expect(raster.band.validity).toBe(tile.bands[0].validity);
    expect(raster.band.validity).toEqual(new Uint8Array([1, 0, 1, 1]));
    expect(raster.grid.affine).toEqual([10, 0, 100, 0, -20, 60]);
    expect(raster.cellSizeMode).toBe(crs === 'EPSG:4326' ? 'geographic' : 'web-mercator');
    expect(makeWorkbenchRasterWorkflowData(source, {bandIndex: 1, maximumCells: 4}).kind).toBe(
      'raster'
    );
    expect(() => readWorkbenchRasterInput(source, {bandIndex: 2, maximumCells: 4})).toThrow(
      /bandIndex/
    );
    expect(() => readWorkbenchRasterInput(source, {bandIndex: 1, maximumCells: 3})).toThrow(
      /at most 3/
    );
    const multiTile = {...source, tiles: [tile, {...tile, id: 'other'}]};
    expect(() => readWorkbenchRasterInput(multiTile, {bandIndex: 1, maximumCells: 4})).toThrow(
      /tileId/
    );
    expect(
      readWorkbenchRasterInput(multiTile, {tileId: 'other', bandIndex: 1, maximumCells: 4}).band
    ).toBe(tile.bands[0]);
    expect(() =>
      readWorkbenchRasterInput(multiTile, {tileId: 'absent', bandIndex: 1, maximumCells: 4})
    ).toThrow(/tileId/);
  });
});

test('the real workflow executor hands a bounded raster from a host source to a typed binding', async () => {
  const raster = makeRaster();
  const source = getWorkflowSourceOperations().find(
    operation => operation.id === 'source-dataset'
  )!;
  const binding = makeBinding('raster');
  binding.runData = vi.fn(binding.runData!);
  const operation = makeWorkflowOperationFromCatalogBinding(binding);
  const registry = createWorkflowOperationRegistry([source, operation]);
  const document: WorkflowDocument = {
    schemaVersion: 1,
    id: 'raster-test',
    title: 'Raster test',
    inputs: [],
    settings: {engine: 'cpu', autoRun: false},
    steps: [
      {
        id: 'source',
        operationId: source.id,
        parameters: {
          dataset: {from: 'static', value: 'elevation'},
          geometryMode: {from: 'static', value: 'none'}
        },
        inputs: {},
        position: {x: 0, y: 0}
      },
      {
        id: 'analysis',
        position: {x: 200, y: 0},
        operationId: operation.id,
        parameters: {},
        inputs: {data: {stepId: 'source', output: 'data'}}
      }
    ]
  };
  const host = makeFixtureHost(
    [
      {
        id: 'elevation',
        label: 'Elevation',
        rowCount: 0,
        resident: false,
        fields: [],
        geometryFields: []
      }
    ],
    {elevation: makeWorkflowRasterData(raster, 4)}
  );
  const runner = new WorkflowRunner({registry, host});
  try {
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status, JSON.stringify(record)).toBe('done');
    expect(record.steps, JSON.stringify(record)).toHaveLength(2);
    expect(binding.runData).toHaveBeenCalledOnce();
    const handedOff = vi.mocked(binding.runData).mock.calls[0][0]['data'];
    expect(handedOff).toEqual(raster);
    expect(handedOff.type).toBe('raster');
    expect((await runner.run()).steps).toEqual([]);
  } finally {
    runner.destroy();
  }
});

test('mixed raster, flow table and feature ports reach one typed callback without geometry coercion', () => {
  const raster = makeRaster();
  const table = makeCatalogTableInput(makeFlowTable(), {maximumRows: 2, identityColumn: 'row_id'});
  const points = {
    type: 'FeatureCollection' as const,
    features: [{geometry: {type: 'Point' as const, coordinates: [0, 0]}, properties: {zone: 1}}]
  };
  const binding = makeBinding('raster');
  const mixed: CatalogOperationBinding = {
    ...binding,
    inputs: [
      binding.inputs[0],
      {
        name: 'flows',
        label: 'Flows',
        dataKind: 'table',
        accepts: [],
        maximumFeatures: 2,
        maximumVertices: 0
      },
      {name: 'zones', label: 'Zones', accepts: ['point'], maximumFeatures: 1, maximumVertices: 1}
    ],
    parameters: [
      {
        name: 'coordinateSystem',
        label: 'Coordinates',
        kind: 'enum',
        options: ['EPSG:4326'],
        defaultValue: 'EPSG:4326'
      }
    ],
    runData: vi.fn(binding.runData!)
  };
  const inputs = {data: raster, flows: table, zones: points};
  runCatalogOperation(mixed, {}, inputs);
  expect(vi.mocked(mixed.runData!).mock.calls[0][0]).toBe(inputs);
  expect(() =>
    runCatalogOperation({...mixed, run: () => binding.runData!(inputs, {})}, {}, inputs)
  ).toThrow(/typed engines exclusively/);
  const invocation = JSON.parse(serializeCatalogInvocation(mixed, {}, inputs));
  expect(invocation.inputs['flows']).toMatchObject({
    dataKind: 'table',
    rowCount: 2,
    featureCount: 0
  });
  expect(matchCatalogInvocationInputs(invocation, inputs)).toBe(true);
  expect(
    matchCatalogInvocationInputs(invocation, {
      ...inputs,
      flows: {...table, identityColumn: 'origin'}
    })
  ).toBe(false);
});

test('catalog content hashes distinguish non-finite samples even with an unchanged validity mask', () => {
  const binding = makeBinding('raster');
  const raster = makeRaster();
  const input = {...raster, band: {...raster.band, values: new Float32Array([NaN, 3, -999, 1])}};
  const invocation = JSON.parse(serializeCatalogInvocation(binding, {}, {data: input}));
  expect(matchCatalogInvocationInputs(invocation, {data: input})).toBe(true);
  expect(
    matchCatalogInvocationInputs(invocation, {
      data: {...input, band: {...input.band, values: new Float32Array([Infinity, 3, -999, 1])}}
    })
  ).toBe(false);
});

test.each([
  'at-limit',
  'oversize',
  'missing-payload',
  'missing-limit',
  'bad-validity'
] as const)('runner validates raster payloads before a non-catalog consumer runs (%s)', async scenario => {
  const width = scenario === 'oversize' ? 3 : 2;
  const raster = makeCatalogRasterInput(
    {
      ...makeRaster(),
      grid: makeGPURasterMetadata({width, height: 2}),
      band: {
        index: 1,
        noDataValue: null,
        values: new Float32Array(width * 2),
        validity: new Uint8Array(width * 2).fill(1)
      }
    },
    6
  );
  const source = getWorkflowSourceOperations().find(
    operation => operation.id === 'source-dataset'
  )!;
  const original = makeFixtureRegistry().get('fixture-count')!;
  const run = vi.fn(original.run);
  const consumer = {
    ...original,
    inputs: [
      {
        name: 'rows',
        label: 'Elevation',
        accepts: ['raster'] as const,
        maximumRows: 0,
        maximumCells: scenario === 'missing-limit' ? undefined : 4
      }
    ],
    run
  };
  expect(consumer.origin).toBe('builtin');
  const data = makeWorkflowRasterData(raster, 6);
  if (scenario === 'missing-payload') delete data.raster;
  if (scenario === 'bad-validity')
    data.raster = {...raster, band: {...raster.band, validity: new Uint8Array(1)}};
  const host = makeFixtureHost(
    [
      {
        id: 'elevation',
        label: 'Elevation',
        rowCount: 0,
        resident: false,
        fields: [],
        geometryFields: []
      }
    ],
    {elevation: data}
  );
  const runner = new WorkflowRunner({
    host,
    registry: createWorkflowOperationRegistry([source, consumer])
  });
  const document: WorkflowDocument = {
    schemaVersion: 1,
    id: 'raster-capacity',
    title: 'Raster capacity',
    inputs: [],
    settings: {engine: 'cpu', autoRun: false},
    steps: [
      {
        id: 'source',
        operationId: source.id,
        parameters: {
          dataset: {from: 'static', value: 'elevation'},
          geometryMode: {from: 'static', value: 'none'}
        },
        inputs: {},
        position: {x: 0, y: 0}
      },
      {
        id: 'consumer',
        operationId: consumer.id,
        parameters: {},
        inputs: {rows: {stepId: 'source', output: 'data'}},
        position: {x: 200, y: 0}
      }
    ]
  };
  try {
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.steps, JSON.stringify(record)).toHaveLength(2);
    if (scenario === 'at-limit') {
      expect(record.status, JSON.stringify(record)).toBe('done');
      expect(run).toHaveBeenCalledOnce();
    } else {
      expect(record.status, JSON.stringify(record)).toBe('error');
      expect(record.steps[1].error).toMatchObject({
        code:
          scenario === 'oversize'
            ? 'capacity-exceeded'
            : scenario === 'missing-payload'
              ? 'input-required'
              : 'operation-failed',
        stepId: 'consumer',
        port: 'rows'
      });
      const reason =
        scenario === 'oversize'
          ? /grid cell count must be at most 4/
          : scenario === 'missing-payload'
            ? /requires a raster band\/grid payload/
            : scenario === 'missing-limit'
              ? /maximumCells must be an integer/
              : /band.validity length/;
      expect(record.steps[1].error?.message).toMatch(reason);
      expect(run).not.toHaveBeenCalled();
    }
  } finally {
    await runner.destroy();
  }
});

test.each([
  'table',
  'raster'
] as const)('AX applicability accepts valid %s inputs and enforces the matching capacity', kind => {
  const binding = makeBinding(kind);
  const descriptor = deriveKeplerOperationDescriptor(binding);
  const input =
    kind === 'table' ? makeCatalogTableInput(makeFlowTable(), {maximumRows: 2}) : makeRaster();
  const profile: KeplerDatasetProfile =
    input.type === 'table'
      ? profileKeplerColumnTable(input.table, {datasetId: 'typed-data', label: 'Typed data'})
      : {
          datasetId: 'typed-data',
          label: 'Typed data',
          kind: 'surface',
          rowCount: input.grid.width * input.grid.height,
          columns: [],
          geometryColumns: []
        };
  expect(runCatalogOperation(binding, {}, {data: input}).engine).toBe('cpu-reference');
  expect(
    isKeplerDatasetCompatibleWithInput(profile, binding.inputs[0], descriptor.inputs[0].kind)
  ).toBe(true);
  const applicability = evaluateKeplerOperationApplicability(binding, descriptor, [profile]);
  expect(applicability.status, JSON.stringify(applicability)).toBe('ready');
  expect(applicability.inputAssignments).toEqual({data: 'typed-data'});
  expect(
    applicability.requirements.some(requirement => requirement.id === 'maximum-vertices')
  ).toBe(false);

  const limited = {
    ...binding,
    inputs: [
      {...binding.inputs[0], ...(kind === 'table' ? {maximumFeatures: 1} : {maximumCells: 3})}
    ]
  };
  const overflow = evaluateKeplerOperationApplicability(
    limited,
    deriveKeplerOperationDescriptor(limited),
    [profile]
  );
  expect(overflow.status, JSON.stringify(overflow)).toBe('blocked');
  expect(overflow.requirements).toEqual(
    expect.arrayContaining([
      expect.objectContaining({input: 'data', id: 'maximum-rows', state: 'failed'})
    ])
  );

  for (const incompatible of [
    {...profile, kind: kind === 'table' ? ('surface' as const) : ('attributes' as const)},
    {...profile, catalogGeometry: 'point' as const, geometryColumns: ['geometry']},
    {...profile, pointColumns: {longitude: 'longitude', latitude: 'latitude'}}
  ]) {
    expect(
      isKeplerDatasetCompatibleWithInput(incompatible, binding.inputs[0], descriptor.inputs[0].kind)
    ).toBe(false);
    const rejected = evaluateKeplerOperationApplicability(binding, descriptor, [incompatible]);
    expect(rejected.status, JSON.stringify(rejected)).toBe('blocked');
    expect(rejected.missing[0].what).toContain(kind === 'table' ? 'table' : 'raster');
    expect(rejected.missing[0].remedy).not.toMatch(/coordinate columns|geometry column/);
  }
});
