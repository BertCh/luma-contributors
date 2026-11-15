// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NullDevice} from '@luma.gl/core/test-utils';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {KeplerComputeSession} from '../../src/session/compute-session';
import type {KeplerPolygonAnalysisOperation} from '../../src/recipes/analysis-recipe';
import {preparePublishedGeometry} from '../../../../examples/deck/kepler-compute/src/source-publication';
import {
  getSandboxSource,
  getSandboxBundleSources,
  replaySandboxBundle,
  validateSandboxBundle,
  readSandboxFixture,
  readPublishedSandboxSource,
  replaySandbox,
  saveSandbox
} from '../../../../examples/deck/kepler-compute/src/sandbox-state';
import {createSandboxSample} from '../../../../examples/deck/kepler-compute/src/sandbox-samples';
import type {ComputeRuntime} from '../../../../examples/deck/kepler-compute/src/compute-runtime';
import type {PublishedSourceRuntime} from '../../../../examples/deck/kepler-compute/src/source-publication';
import type {ResidentGroupByProvenance} from '../../../../examples/deck/kepler-compute/src/resident-group-by-plan';

function fixture(rowCount = 3, memoryBudgetBytes = 1024 * 1024) {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes});
  const table = new arrow.Table({
    value: arrow.makeVector(Float32Array.from({length: rowCount}, (_, index) => index + 1))
  });
  const dataset = session.importArrowDataset({id: 'resident-observations', table});
  const source = {datasetId: dataset.id, revision: dataset.revision};
  const inspector = {dataset, currentRevision: 5, isCurrent: true};
  const analysis = {
    settings: {mode: 'z-score'},
    serializeRecipe: () =>
      JSON.stringify({
        schemaVersion: 1,
        algorithmVersion: 1,
        id: 'value-analysis',
        source,
        population: 'selected',
        selectionRevision: inspector.currentRevision + 1,
        backendPolicy: 'require-gpu',
        operation: {kind: 'standardize', column: 'value', method: 'z-score', variance: 'sample'}
      }),
    restoreRecipe: vi.fn(),
    setMode: vi.fn()
  };
  const publications = new Map<string, PublishedSourceRuntime>();
  const runtime = {
    publications,
    session,
    explorer: inspector,
    analysis,
    requestUpdate: vi.fn()
  } as unknown as ComputeRuntime;
  const getNativeConfig = () => ({
    version: 'v1',
    config: {
      visState: {
        filters: [
          {
            id: 'range',
            dataId: ['resident-observations'],
            name: ['value'],
            type: 'range',
            value: [1, 2]
          }
        ]
      }
    }
  });
  const applyNativeConfig = vi.fn(() => {
    inspector.currentRevision++;
  });
  return {
    device,
    session,
    table,
    inspector,
    publications,
    analysis,
    runtime,
    getNativeConfig,
    applyNativeConfig,
    cleanup: async () => {
      await session.destroy(Promise.resolve());
      device.destroy();
    }
  };
}

test('host state save/replay applies native filters and rebinds selected analysis to the replay revision', async () => {
  const context = fixture();
  const serialized = saveSandbox(context);
  context.inspector.currentRevision = 10;
  const runtime = await replaySandbox(serialized, context);
  expect(runtime).toBe(context.runtime);
  expect(context.applyNativeConfig).toHaveBeenCalledWith(context.getNativeConfig());
  const restored = JSON.parse(context.analysis.restoreRecipe.mock.calls[0][0]);
  expect(restored.selectionRevision).toBe(12);
  expect(restored.source.revision).toBe(context.inspector.dataset.revision);
  expect(runtime.requestUpdate).toHaveBeenCalledOnce();
  expect(readSandboxFixture(getSandboxSource(runtime))).toEqual({seed: 42, rowCount: 3});
  await context.cleanup();
});

test('explicit host fixture reimport advances source revision and restores selection before analysis', async () => {
  const context = fixture();
  const serialized = saveSandbox(context);
  const reimportSource = vi.fn(async source => {
    expect(readSandboxFixture(source)).toEqual({seed: 42, rowCount: 3});
    context.inspector.dataset = context.session.importArrowDataset({
      id: 'resident-observations',
      table: context.table
    });
    context.inspector.currentRevision = 0;
    return context.runtime;
  });
  await replaySandbox(serialized, {...context, reimportSource}, true);
  expect(reimportSource).toHaveBeenCalledOnce();
  const restored = JSON.parse(context.analysis.restoreRecipe.mock.calls[0][0]);
  expect(restored.source.revision).toBe(2);
  expect(restored.selectionRevision).toBe(2);
  expect(context.applyNativeConfig).toHaveBeenCalledOnce();
  await context.cleanup();
});

test('cartogram replay retains explicit planar roles and refuses excessive source rows before native mutation', async () => {
  for (const rowCount of [3, 65537]) {
    const context = fixture(rowCount, 64 * 1024 * 1024);
    try {
      const state = JSON.parse(saveSandbox(context));
      state.analyses[0].operation = {
        kind: 'dorling-cartogram',
        coordinateSpace: 'planar',
        metric: 'native',
        units: 'meters',
        xColumn: 'value',
        yColumn: 'value',
        valueColumn: 'value',
        crs: 'EPSG:3857',
        bounds: [0, 0, rowCount + 1, rowCount + 1],
        areaScale: 1,
        maximumIterations: 0,
        overlapTolerance: 0.01,
        attraction: 0
      };
      if (rowCount > 65536) {
        await expect(replaySandbox(JSON.stringify(state), context)).rejects.toThrow(/65536/);
        expect(context.applyNativeConfig).not.toHaveBeenCalled();
        expect(context.analysis.restoreRecipe).not.toHaveBeenCalled();
      } else {
        await replaySandbox(JSON.stringify(state), context);
        expect(context.applyNativeConfig).toHaveBeenCalledOnce();
        expect(JSON.parse(context.analysis.restoreRecipe.mock.lastCall![0]).operation).toEqual({
          ...state.analyses[0].operation,
          overlapTolerance: Math.fround(0.01)
        });
      }
    } finally {
      await context.cleanup();
    }
  }
});

test('host replay refuses incompatible schema and contradictory selection before changing native state', async () => {
  const context = fixture();
  const state = JSON.parse(saveSandbox(context));
  state.sources[0].schema[0].type = 'Float64';
  await expect(replaySandbox(JSON.stringify(state), context)).rejects.toThrow(/does not match/);
  expect(context.applyNativeConfig).not.toHaveBeenCalled();
  const contradictory = JSON.parse(saveSandbox(context));
  contradictory.selections[0].value.filters = [];
  await expect(replaySandbox(JSON.stringify(contradictory), context)).rejects.toThrow(
    /do not match/
  );
  expect(context.applyNativeConfig).not.toHaveBeenCalled();
  await context.cleanup();
});

test('native config serialization omits optional undefined fields but rejects GPU-like objects', async () => {
  const context = fixture();
  expect(() =>
    saveSandbox({
      ...context,
      getNativeConfig: () => ({...context.getNativeConfig(), optional: undefined})
    })
  ).not.toThrow();
  expect(() =>
    saveSandbox({
      ...context,
      getNativeConfig: () => ({...context.getNativeConfig(), buffer: new Uint32Array([1])})
    })
  ).toThrow(/plain/);
  await context.cleanup();
});

test('host save rejects native datasets missing from its source manifest', async () => {
  const context = fixture();
  try {
    expect(() =>
      saveSandbox({
        ...context,
        getNativeConfig: () => ({
          ...context.getNativeConfig(),
          externalLayer: {config: {dataId: 'external-data'}}
        })
      })
    ).toThrow(/source manifest does not include external-data/);
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

test('host replay rejects unresolved native sources before reimporting or applying configuration', async () => {
  const context = fixture();
  try {
    const saved = JSON.parse(saveSandbox(context));
    saved.nativeConfig.config.visState.layers = [{config: {dataId: 'external-data'}}];
    const reimportSource = vi.fn(async () => context.runtime);
    await expect(
      replaySandbox(JSON.stringify(saved), {...context, reimportSource}, true)
    ).rejects.toThrow(/source manifest does not include external-data/);
    expect(reimportSource).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(context.analysis.restoreRecipe).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

test('host fixture reimport rejects a foreign logical source before replacing the current dataset', async () => {
  const context = fixture();
  try {
    const saved = saveSandbox(context).replaceAll('resident-observations', 'other-dataset');
    const reimportSource = vi.fn(async () => context.runtime);
    await expect(replaySandbox(saved, {...context, reimportSource}, true)).rejects.toThrow(
      /another dataset/
    );
    expect(reimportSource).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

test('fixture schema and normalization mismatches reject before destructive reimport', async () => {
  const context = fixture();
  try {
    const reimportSource = vi.fn(async () => context.runtime);
    for (const change of ['schema', 'normalization']) {
      const saved = JSON.parse(saveSandbox(context));
      if (change === 'schema') saved.sources[0].schema[0].type = 'Float64';
      else saved.sources[0].normalization = {float64: {mode: 'exact'}};
      await expect(
        replaySandbox(JSON.stringify(saved), {...context, reimportSource}, true)
      ).rejects.toThrow(/does not match/);
    }
    expect(reimportSource).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(context.analysis.restoreRecipe).not.toHaveBeenCalled();
    expect(context.inspector.dataset.revision).toBe(1);
  } finally {
    await context.cleanup();
  }
});

function addPublication(context: ReturnType<typeof fixture>) {
  const table = new arrow.Table({
    longitude: arrow.makeVector(new Float32Array([-73, -74])),
    latitude: arrow.makeVector(new Float32Array([40, 41])),
    amount: arrow.makeVector(new Float32Array([10, 20]))
  });
  const dataset = context.session.importArrowDataset({id: 'uploaded-points', table});
  const explorer = {dataset, currentRevision: 7, isCurrent: true};
  const publication = {
    dataset,
    label: 'Uploaded points',
    coordinates: {longitude: 'longitude', latitude: 'latitude'},
    consumer: {explorer}
  };
  context.publications.set(dataset.id, publication as unknown as PublishedSourceRuntime);
  return {table, explorer, publication};
}

function createBundle(context: ReturnType<typeof fixture>) {
  return {
    state: JSON.parse(saveSandbox(context)),
    sources: getSandboxBundleSources(context.runtime)
  };
}

test('completed Group By bundle preserves historical lineage without a live parent', async () => {
  const context = fixture();
  try {
    const table = new arrow.Table({
      category: arrow.vectorFromArray(['b', null, 'a'], new arrow.Utf8()),
      amount_count: arrow.makeVector(new Uint32Array([2, 1, 2]))
    });
    const dataset = context.session.importArrowDataset({id: 'grouped-result', table});
    const groupBy: ResidentGroupByProvenance = {
      schemaVersion: 1,
      kind: 'group-by',
      source: {datasetId: 'removed-parent', revision: 17},
      population: 'all',
      operationId: 'group-operation',
      resultId: dataset.id,
      resultLabel: 'Grouped snapshot',
      keyColumn: 'category',
      aggregations: [{column: 'amount', technique: 'count', name: 'amount_count'}],
      fields: [
        {name: 'category', type: 'string', analyzerType: 'string'},
        {name: 'amount_count', type: 'integer', analyzerType: 'integer'}
      ],
      numericalPolicy: 'float32',
      rowOrder: 'first-source-occurrence'
    };
    context.publications.set(dataset.id, {
      dataset,
      label: groupBy.resultLabel,
      groupBy,
      consumer: {explorer: {dataset, currentRevision: 0, isCurrent: true}}
    } as unknown as PublishedSourceRuntime);
    const bundle = createBundle(context);
    const descriptor = bundle.state.sources[1];
    expect(readPublishedSandboxSource(descriptor)).toEqual({label: groupBy.resultLabel, groupBy});
    expect(readPublishedSandboxSource(descriptor).groupBy).not.toBe(groupBy);
    expect(bundle.state.sources.some(source => source.dataset.datasetId === 'removed-parent')).toBe(
      false
    );
    expect([...bundle.sources[1].table.getChild('category')!]).toEqual(['b', null, 'a']);
    validateSandboxBundle(bundle, context.runtime);
    const restoreBundle = vi.fn(async restored => {
      expect(readPublishedSandboxSource(restored.state.sources[1]).groupBy?.source).toEqual({
        datasetId: 'removed-parent',
        revision: 17
      });
      return context.runtime;
    });
    await replaySandboxBundle(bundle, {...context, restoreBundle});
    expect(restoreBundle).toHaveBeenCalledOnce();
    const altered = structuredClone(descriptor);
    altered.reference.parameters.groupBy.resultId = 'unrelated-result';
    expect(() => readPublishedSandboxSource(altered)).toThrow(/matching table-only result schema/);
    altered.reference.parameters.groupBy.resultId = dataset.id;
    altered.schema.reverse();
    expect(() => readPublishedSandboxSource(altered)).toThrow(/matching table-only result schema/);
    altered.schema.reverse();
    altered.reference.parameters.coordinates = {
      longitude: 'amount_count',
      latitude: 'amount_count'
    };
    expect(() => readPublishedSandboxSource(altered)).toThrow(/matching table-only result schema/);
  } finally {
    await context.cleanup();
  }
});

test('workspace state saves and replays all published sources without discarding original roles', async () => {
  const context = fixture();
  try {
    const publication = addPublication(context);
    const serialized = saveSandbox(context);
    const saved = JSON.parse(serialized);
    expect(saved.sources).toHaveLength(2);
    expect(saved.sources[1].reference.parameters).toEqual({
      label: 'Uploaded points',
      coordinates: {longitude: 'longitude', latitude: 'latitude'},
      geometryColumn: null
    });
    expect(
      saved.selections.map((selection: {source: {datasetId: string}}) => selection.source.datasetId)
    ).toEqual(['resident-observations', 'uploaded-points']);
    await replaySandbox(serialized, context);
    expect(context.applyNativeConfig).toHaveBeenCalledOnce();
    publication.explorer.dataset = {...publication.explorer.dataset, revision: 2};
    publication.publication.dataset = publication.explorer.dataset;
    await expect(replaySandbox(serialized, context)).rejects.toThrow(/Dataset changed/);
    const reimportSource = vi.fn(async () => context.runtime);
    await expect(replaySandbox(serialized, {...context, reimportSource}, true)).rejects.toThrow(
      /portable bundle/
    );
    expect(reimportSource).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

test('bundle restoration rebinds fixture and publication revisions before replaying native config and analysis', async () => {
  const context = fixture();
  try {
    const publication = addPublication(context);
    const bundle = createBundle(context);
    const restoreBundle = vi.fn(async () => {
      context.inspector.dataset = context.session.importArrowDataset({
        id: 'resident-observations',
        table: context.table
      });
      const dataset = context.session.importArrowDataset({
        id: 'uploaded-points',
        table: publication.table
      });
      publication.explorer.dataset = dataset;
      publication.publication.dataset = dataset;
      return context.runtime;
    });
    await replaySandboxBundle(bundle, {...context, restoreBundle});
    expect(restoreBundle).toHaveBeenCalledOnce();
    expect(context.applyNativeConfig).toHaveBeenCalledOnce();
    expect(JSON.parse(context.analysis.restoreRecipe.mock.calls[0][0]).source.revision).toBe(2);
    expect(publication.explorer.dataset.revision).toBe(2);
  } finally {
    await context.cleanup();
  }
});

test('bundle preflight rejects every malformed source and unsupported analysis before host replacement', async () => {
  const context = fixture();
  try {
    addPublication(context);
    const restoreBundle = vi.fn(async () => context.runtime);
    const mutations: ((bundle: ReturnType<typeof createBundle>) => void)[] = [
      bundle => {
        bundle.sources.pop();
      },
      bundle => {
        bundle.sources[1].revision++;
      },
      bundle => {
        bundle.state.sources[1].schema[0].type = 'Float64';
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.coordinates.longitude = 'missing';
      },
      bundle => {
        bundle.state.sources[1].normalization.columns = ['amount'];
      },
      bundle => {
        bundle.state.sources[0].reference.parameters.rowCount++;
      },
      bundle => {
        bundle.state.analyses[0].operation.column = 'missing';
      }
    ];
    for (const mutate of mutations) {
      const bundle = createBundle(context);
      mutate(bundle);
      await expect(replaySandboxBundle(bundle, {...context, restoreBundle})).rejects.toThrow();
    }
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(context.analysis.restoreRecipe).not.toHaveBeenCalled();
    expect(context.inspector.dataset.revision).toBe(1);
  } finally {
    await context.cleanup();
  }
});

test('bundle preflight prepares coordinates on the CPU and hides invalid source rows', async () => {
  const context = fixture();
  try {
    const publication = addPublication(context);
    const bundle = createBundle(context);
    expect(() => validateSandboxBundle(bundle, context.runtime)).not.toThrow();
    publication.table.getChild('latitude')!.set(0, 100);
    // Invalid lng/lat rows are hidden with NaN positions, matching native Kepler, instead of failing preflight.
    expect(() => validateSandboxBundle(bundle, context.runtime)).not.toThrow();
  } finally {
    await context.cleanup();
  }
});

function addPublicationAnalysis(publication: {
  explorer: ReturnType<typeof addPublication>['explorer'];
  publication: object;
}) {
  const analysis = {
    settings: {mode: 'quantile', column: 'amount', population: 'selected', classCount: 3},
    serializeRecipe: () =>
      JSON.stringify({
        schemaVersion: 1,
        algorithmVersion: 1,
        id: 'uploaded-analysis',
        source: {
          datasetId: publication.explorer.dataset.id,
          revision: publication.explorer.dataset.revision
        },
        population: 'selected',
        selectionRevision: publication.explorer.currentRevision + 1,
        backendPolicy: 'require-gpu',
        operation: {
          kind: 'classify',
          column: 'amount',
          method: 'quantile',
          classCount: 3,
          naturalBreaksMode: 'approximate'
        }
      }),
    restoreRecipe: vi.fn(),
    setMode: vi.fn()
  };
  const getAnalysis = vi.fn(() => analysis);
  Object.assign(publication.publication, {analysis, getAnalysis});
  return {analysis, getAnalysis};
}

test('workspace saves one active analysis per source and restores each by dataset and column', async () => {
  const context = fixture();
  try {
    const publication = addPublication(context);
    const {analysis, getAnalysis} = addPublicationAnalysis(publication);
    const saved = JSON.parse(saveSandbox(context));
    expect(saved.analyses).toHaveLength(2);
    expect(saved.analyses[1].operation.column).toBe('amount');
    publication.explorer.currentRevision = 20;
    await replaySandbox(JSON.stringify(saved), context);
    expect(getAnalysis).toHaveBeenCalledOnce();
    const replayed = JSON.parse(analysis.restoreRecipe.mock.calls[0][0]);
    expect(replayed.source.datasetId).toBe('uploaded-points');
    expect(replayed.operation.column).toBe('amount');
    expect(replayed.selectionRevision).toBe(21);
    expect(context.analysis.restoreRecipe).toHaveBeenCalledOnce();
    saved.analyses.pop();
    await replaySandbox(JSON.stringify(saved), context);
    expect(analysis.setMode).toHaveBeenCalledWith('off');
    expect(getAnalysis).toHaveBeenCalledOnce();
  } finally {
    await context.cleanup();
  }
});

test('analysis bundle preflight rejects duplicate source analyses and non-Float32 normalized fields', async () => {
  const context = fixture();
  try {
    addPublicationAnalysis(addPublication(context));
    const restoreBundle = vi.fn(async () => context.runtime);
    const duplicate = createBundle(context);
    duplicate.state.analyses.push({...duplicate.state.analyses[1], id: 'second-uploaded-analysis'});
    await expect(replaySandboxBundle(duplicate, {...context, restoreBundle})).rejects.toThrow(
      /at most one analysis/
    );
    const nonFloat = createBundle(context);
    const table = new arrow.Table({
      longitude: arrow.makeVector(new Float32Array([-73, -74])),
      latitude: arrow.makeVector(new Float32Array([40, 41])),
      amount: arrow.makeVector(new Uint32Array([10, 20]))
    });
    nonFloat.sources[1].table = table;
    nonFloat.state.sources[1].schema = table.schema.fields.map(field => ({
      name: field.name,
      type: field.type.toString(),
      nullable: field.nullable
    }));
    await expect(replaySandboxBundle(nonFloat, {...context, restoreBundle})).rejects.toThrow(
      /Float32/
    );
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(context.analysis.restoreRecipe).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

test('exact natural-breaks capacity rejects before current replay or bundle replacement', async () => {
  const context = fixture(1025);
  try {
    const bundle = createBundle(context);
    bundle.state.analyses[0].operation = {
      kind: 'classify',
      column: 'value',
      method: 'natural-breaks',
      classCount: 3,
      naturalBreaksMode: 'exact'
    };
    const restoreBundle = vi.fn(async () => context.runtime);
    const reimportSource = vi.fn(async () => context.runtime);
    await expect(
      replaySandbox(JSON.stringify(bundle.state), {...context, reimportSource}, true)
    ).rejects.toThrow(/1,?024/);
    expect(reimportSource).not.toHaveBeenCalled();
    await expect(replaySandbox(JSON.stringify(bundle.state), context)).rejects.toThrow(/1,?024/);
    await expect(replaySandboxBundle(bundle, {...context, restoreBundle})).rejects.toThrow(
      /1,?024/
    );
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(restoreBundle).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

test('excess-risk preflight validates the denominator before replay or workspace replacement', async () => {
  const context = fixture();
  try {
    addPublicationAnalysis(addPublication(context));
    const restoreBundle = vi.fn(async () => context.runtime);
    const missing = createBundle(context);
    missing.state.analyses[1].operation = {
      kind: 'excess-risk',
      events: 'amount',
      populations: 'missing'
    };
    await expect(replaySandbox(JSON.stringify(missing.state), context)).rejects.toThrow(
      /field is missing/
    );
    await expect(replaySandboxBundle(missing, {...context, restoreBundle})).rejects.toThrow(
      /field is missing/
    );
    const wrongType = createBundle(context);
    wrongType.state.analyses[1].operation = {
      kind: 'excess-risk',
      events: 'longitude',
      populations: 'amount'
    };
    const table = new arrow.Table({
      longitude: arrow.makeVector(new Float32Array([-73, -74])),
      latitude: arrow.makeVector(new Float32Array([40, 41])),
      amount: arrow.makeVector(new Uint32Array([10, 20]))
    });
    wrongType.sources[1].table = table;
    wrongType.state.sources[1].schema = table.schema.fields.map(field => ({
      name: field.name,
      type: field.type.toString(),
      nullable: field.nullable
    }));
    await expect(replaySandboxBundle(wrongType, {...context, restoreBundle})).rejects.toThrow(
      /Float32/
    );
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    const valid = createBundle(context);
    valid.state.analyses[1].operation = {
      kind: 'excess-risk',
      events: 'amount',
      populations: 'longitude'
    };
    expect(() => validateSandboxBundle(valid, context.runtime)).not.toThrow();
    await replaySandbox(JSON.stringify(valid.state), context);
    expect(context.applyNativeConfig).toHaveBeenCalledOnce();
  } finally {
    await context.cleanup();
  }
});

function addPolygonPublication(context: ReturnType<typeof fixture>) {
  const table = new arrow.Table({
    geometry: arrow.vectorFromArray(
      [0, 1, 2].map(longitude =>
        JSON.stringify({
          type: 'Polygon',
          coordinates: [
            [
              [longitude, 0],
              [longitude + 1, 0],
              [longitude + 1, 1],
              [longitude, 1],
              [longitude, 0]
            ]
          ]
        })
      ),
      new arrow.Utf8()
    ),
    amount: arrow.makeVector(new Float32Array([10, 20, 30])),
    population: arrow.makeVector(new Float32Array([100, 200, 300])),
    unprepared: arrow.makeVector(new Uint32Array([10, 20, 30]))
  });
  const dataset = context.session.importArrowDataset({
    id: 'uploaded-polygons',
    table,
    normalization: {columns: ['amount', 'population', 'unprepared']}
  });
  const explorer = {dataset, currentRevision: 7, isCurrent: true};
  const publication = {
    dataset,
    label: 'Adjacent polygons',
    geometryColumn: 'geometry',
    polygonSource: {column: 'geometry', source: preparePublishedGeometry(table, 'geometry')},
    consumer: {explorer}
  };
  context.publications.set(dataset.id, publication as unknown as PublishedSourceRuntime);
  const analysis = addPublicationAnalysis({explorer, publication});
  return {table, explorer, publication, ...analysis};
}

const polygonWeights = {
  kind: 'contiguity',
  geometryColumn: 'geometry',
  criterion: 'queen',
  equality: 'exact-source-xy',
  neighborCapacity: 16,
  pairCapacity: 16
} as const;

const polygonOperations: readonly KeplerPolygonAnalysisOperation[] = [
  {
    kind: 'spatial-rate-polygons',
    events: 'amount',
    populations: 'population',
    weights: polygonWeights,
    scale: 1000,
    islandPolicy: 'undefined'
  },
  {
    kind: 'hot-spots-polygons',
    column: 'amount',
    weights: polygonWeights,
    minimumNeighbors: 1,
    selfWeight: 1,
    multipleTesting: 'none',
    islands: 'exclude',
    inference: 'analytic-two-sided'
  },
  {
    kind: 'local-moran-polygons',
    column: 'amount',
    weights: polygonWeights,
    minimumNeighbors: 1,
    multipleTesting: 'none',
    islands: 'exclude',
    significanceLevel: 0.05,
    inference: 'analytic-conditional-normal-two-sided'
  },
  {
    kind: 'quantile-lisa-polygons',
    column: 'amount',
    weights: polygonWeights,
    quantileCount: 2,
    selectedQuantile: 1,
    seed: 7,
    permutationCount: 99,
    significanceLevel: 0.05,
    inference: 'legacy-directed-pseudo',
    multipleTesting: 'none',
    islands: 'exclude',
    quantileMethod: 'type-5'
  }
];

test('polygon replay accepts retained geometry roles independently of Float32 analysis measures', async () => {
  const context = fixture(3, 64 * 1024 * 1024);
  try {
    const publication = addPolygonPublication(context);
    expect(publication.table.getChild('geometry')?.type.toString()).toBe('Utf8');
    expect(publication.explorer.dataset.value.columns['geometry']).toBeUndefined();
    for (const operation of polygonOperations) {
      const bundle = createBundle(context);
      bundle.state.analyses[1].operation = operation;
      expect(() => validateSandboxBundle(bundle, context.runtime)).not.toThrow();
      await replaySandbox(JSON.stringify(bundle.state), context);
      const restored = JSON.parse(publication.analysis.restoreRecipe.mock.lastCall![0]);
      expect(restored.operation).toEqual(operation);
    }
    expect(context.applyNativeConfig).toHaveBeenCalledTimes(polygonOperations.length);
    expect(publication.getAnalysis).toHaveBeenCalledTimes(polygonOperations.length);
  } finally {
    await context.cleanup();
  }
});

test('polygon replay requires Float32 measures even when a measure also names the geometry role', async () => {
  const context = fixture(3, 64 * 1024 * 1024);
  try {
    const publication = addPolygonPublication(context);
    const restoreBundle = vi.fn(async () => context.runtime);
    for (const operation of polygonOperations) {
      for (const column of ['unprepared', 'geometry']) {
        const numericFields =
          operation.kind === 'spatial-rate-polygons' ? ['events', 'populations'] : ['column'];
        for (const numericField of numericFields) {
          const bundle = createBundle(context);
          bundle.state.analyses[1].operation = {...operation, [numericField]: column};
          await expect(replaySandbox(JSON.stringify(bundle.state), context)).rejects.toThrow(
            /Float32/
          );
          await expect(replaySandboxBundle(bundle, {...context, restoreBundle})).rejects.toThrow(
            /Float32/
          );
        }
      }
    }
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(context.analysis.restoreRecipe).not.toHaveBeenCalled();
    expect(publication.analysis.restoreRecipe).not.toHaveBeenCalled();
    expect(publication.getAnalysis).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

test('polygon replay rejects an unavailable retained topology before applying native configuration', async () => {
  const context = fixture(3, 64 * 1024 * 1024);
  try {
    const publication = addPolygonPublication(context);
    Object.assign(publication.publication, {
      polygonSource: undefined,
      polygonAnalysisUnavailableReason: 'Polygon topology exceeds the retained-source limit'
    });
    for (const operation of polygonOperations) {
      const state = JSON.parse(saveSandbox(context));
      state.analyses[1].operation = operation;
      await expect(replaySandbox(JSON.stringify(state), context)).rejects.toThrow(
        /Polygon topology exceeds the retained-source limit/
      );
    }
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(context.analysis.restoreRecipe).not.toHaveBeenCalled();
    expect(publication.analysis.restoreRecipe).not.toHaveBeenCalled();
    expect(publication.getAnalysis).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});

function addTripPublication(context: ReturnType<typeof fixture>) {
  const sample = createSandboxSample('trips');
  const normalization = {
    columns: sample.table.schema.fields
      .map(field => field.name)
      .filter(name => name !== sample.geometryColumn)
  };
  const dataset = context.session.importArrowDataset({
    id: 'uploaded-trips',
    table: sample.table,
    normalization
  });
  const explorer = {dataset, currentRevision: 3, isCurrent: true};
  const publication = {
    dataset,
    label: 'Timed routes',
    geometryColumn: sample.geometryColumn,
    tripOptions: sample.tripOptions,
    consumer: {explorer}
  };
  context.publications.set(dataset.id, publication as unknown as PublishedSourceRuntime);
  return {sample, normalization, explorer, publication};
}

test('Trips source manifest and bundle replay preserve explicit units, original rows and batches', async () => {
  const context = fixture();
  try {
    const trip = addTripPublication(context);
    trip.publication.tripOptions = {...trip.sample.tripOptions!, originMilliseconds: 1767225600000};
    const bundle = createBundle(context);
    const descriptor = bundle.state.sources[1];
    expect(readPublishedSandboxSource(descriptor)).toEqual({
      label: 'Timed routes',
      geometryColumn: 'geometry',
      tripOptions: {
        timestampEncoding: 'unix-second',
        maximumTimeErrorMilliseconds: 0.1,
        originMilliseconds: 1767225600000
      }
    });
    expect(descriptor.normalization.columns).not.toContain('geometry');
    expect(bundle.sources[1].table.getChild('geometry')).not.toBeNull();
    expect(bundle.sources[1].table.batches.map(batch => batch.numRows)).toEqual([4, 4]);
    validateSandboxBundle(bundle, context.runtime);
    const restoreBundle = vi.fn(async restored => {
      const source = readPublishedSandboxSource(restored.state.sources[1]);
      expect(source.tripOptions).toEqual(trip.publication.tripOptions);
      return context.runtime;
    });
    await replaySandboxBundle(bundle, {...context, restoreBundle});
    expect(restoreBundle).toHaveBeenCalledOnce();
    expect([...trip.publication.dataset.value.source.getChild('source_row_id')!]).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7
    ]);
  } finally {
    await context.cleanup();
  }
});

test('Trips bundle preflight rejects invalid options or unrepresentable timed paths before workspace replacement', async () => {
  const context = fixture();
  try {
    const trip = addTripPublication(context);
    const restoreBundle = vi.fn(async () => context.runtime);
    const mutations: ((bundle: ReturnType<typeof createBundle>) => void)[] = [
      bundle => {
        bundle.state.sources[1].reference.parameters.tripOptions.timestampEncoding = 'auto';
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.tripOptions.crs = 'EPSG:4326';
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.tripOptions.originMilliseconds =
          Number.MAX_SAFE_INTEGER + 1;
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.tripOptions.maximumTimeErrorMilliseconds = -1;
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.geometryColumn = null;
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.coordinates = {
          longitude: 'passengers',
          latitude: 'duration_seconds'
        };
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.tripOptions.timestampEncoding = 'iso8601';
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.tripOptions.originMilliseconds = 0;
      },
      bundle => {
        bundle.state.sources[1].normalization.columns.push('geometry');
      }
    ];
    for (const mutate of mutations) {
      const bundle = createBundle(context);
      mutate(bundle);
      await expect(replaySandboxBundle(bundle, {...context, restoreBundle})).rejects.toThrow();
    }
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    expect(context.publications.get('uploaded-trips')?.dataset).toBe(trip.publication.dataset);
    expect(trip.publication.dataset.revision).toBe(1);
  } finally {
    await context.cleanup();
  }
});

function addH3Publication(context: ReturnType<typeof fixture>, representation: 'uint64' | 'utf8') {
  const hexadecimal = ['8928308280fffff', '8928308280bffff', '89283082807ffff'];
  const cells =
    representation === 'uint64'
      ? arrow.vectorFromArray(
          [null, ...hexadecimal.map(value => BigInt(`0x${value}`))],
          new arrow.Uint64()
        )
      : arrow.vectorFromArray([null, ...hexadecimal], new arrow.Utf8());
  const original = new arrow.Table({
    h3: cells,
    measure: arrow.makeVector(new Float32Array([1, 2, 3, 4]))
  });
  const table = original.slice(0, 2).concat(original.slice(2));
  const normalization = {columns: ['measure']};
  const dataset = context.session.importArrowDataset({id: 'uploaded-h3', table, normalization});
  const explorer = {dataset, currentRevision: 3, isCurrent: true};
  const publication = {dataset, label: 'H3 cells', h3Column: 'h3', consumer: {explorer}};
  context.publications.set(dataset.id, publication as unknown as PublishedSourceRuntime);
  return {table, publication};
}

for (const representation of ['uint64', 'utf8'] as const) {
  test(`H3 ${representation} source roles survive portable replay without normalizing exact IDs`, async () => {
    const context = fixture();
    try {
      const h3 = addH3Publication(context, representation);
      const bundle = createBundle(context);
      expect(readPublishedSandboxSource(bundle.state.sources[1])).toEqual({
        label: 'H3 cells',
        h3Column: 'h3'
      });
      expect(bundle.state.sources[1].normalization.columns).toEqual(['measure']);
      expect(bundle.sources[1].table.batches.map(batch => batch.numRows)).toEqual([2, 2]);
      const originalValues = [...h3.table.getChild('h3')!];
      const transported = arrow.tableFromIPC(arrow.tableToIPC(bundle.sources[1].table));
      expect([...transported.getChild('h3')!]).toEqual(originalValues);
      bundle.sources[1].table = transported;
      validateSandboxBundle(bundle, context.runtime);
      const restoreBundle = vi.fn(async restored => {
        expect(readPublishedSandboxSource(restored.state.sources[1]).h3Column).toBe('h3');
        return context.runtime;
      });
      await replaySandboxBundle(bundle, {...context, restoreBundle});
      expect(restoreBundle).toHaveBeenCalledOnce();
      expect([...h3.publication.dataset.value.source.getChild('h3')!]).toEqual(originalValues);
    } finally {
      await context.cleanup();
    }
  });
}

test('H3 bundle preflight rejects conflicting roles and ambiguous or unusable IDs before replacement', async () => {
  const context = fixture();
  try {
    const h3 = addH3Publication(context, 'utf8');
    const restoreBundle = vi.fn(async () => context.runtime);
    const mutations: ((bundle: ReturnType<typeof createBundle>) => void)[] = [
      bundle => {
        bundle.state.sources[1].reference.parameters.h3Column = 2;
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.h3Column = 'missing';
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.geometryColumn = 'h3';
      },
      bundle => {
        bundle.state.sources[1].reference.parameters.coordinates = {
          longitude: 'measure',
          latitude: 'measure'
        };
      },
      bundle => {
        bundle.state.sources[1].normalization.columns.push('h3');
      },
      bundle => {
        bundle.sources[1].table = new arrow.Table({
          h3: arrow.vectorFromArray(['invalid', 'invalid', null, null], new arrow.Utf8()),
          measure: arrow.makeVector(new Float32Array([1, 2, 3, 4]))
        });
      }
    ];
    for (const mutate of mutations) {
      const bundle = createBundle(context);
      mutate(bundle);
      await expect(replaySandboxBundle(bundle, {...context, restoreBundle})).rejects.toThrow();
    }
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
    const numeric = createBundle(context);
    numeric.sources[1].table = new arrow.Table({
      h3: arrow.makeVector(new Float64Array([Number(BigInt('0x8928308280fffff')), 0, 0, 0])),
      measure: arrow.makeVector(new Float32Array([1, 2, 3, 4]))
    });
    numeric.state.sources[1].schema = numeric.sources[1].table.schema.fields.map(field => ({
      name: field.name,
      type: field.type.toString(),
      nullable: field.nullable
    }));
    await expect(replaySandboxBundle(numeric, {...context, restoreBundle})).rejects.toThrow(
      /Number|number|precision/
    );
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.publications.get('uploaded-h3')?.dataset).toBe(h3.publication.dataset);
  } finally {
    await context.cleanup();
  }
});

test('H3 bundle preflight permits explicit invalid geometry omissions while retaining all source rows', async () => {
  const context = fixture();
  try {
    addH3Publication(context, 'utf8');
    const bundle = createBundle(context);
    bundle.sources[1].table = new arrow.Table({
      h3: arrow.vectorFromArray(
        [null, '8928308280fffff', 'not-a-cell', '8928308280bffff'],
        new arrow.Utf8()
      ),
      measure: arrow.makeVector(new Float32Array([1, 2, 3, 4]))
    });
    expect(() => validateSandboxBundle(bundle, context.runtime)).not.toThrow();
    expect(bundle.sources[1].table.numRows).toBe(4);
    expect(bundle.sources[1].table.getChild('h3')!.get(2)).toBe('not-a-cell');
  } finally {
    await context.cleanup();
  }
});

test('optional point altitude survives portable source descriptors and replay while legacy XY stays compatible', async () => {
  const context = fixture();
  try {
    const publication = addPublication(context);
    const legacy = createBundle(context);
    expect(readPublishedSandboxSource(legacy.state.sources[1]).coordinates).toEqual({
      longitude: 'longitude',
      latitude: 'latitude'
    });
    Object.assign(publication.publication.coordinates, {altitude: 'amount'});
    const bundle = createBundle(context);
    expect(readPublishedSandboxSource(bundle.state.sources[1]).coordinates).toEqual({
      longitude: 'longitude',
      latitude: 'latitude',
      altitude: 'amount'
    });
    expect(() => validateSandboxBundle(bundle, context.runtime)).not.toThrow();
    const restoreBundle = vi.fn(async restored => {
      expect(readPublishedSandboxSource(restored.state.sources[1]).coordinates?.altitude).toBe(
        'amount'
      );
      expect(restored.sources[1].table.getChild('amount')?.toArray()).toEqual(
        new Float32Array([10, 20])
      );
      return context.runtime;
    });
    await replaySandboxBundle(bundle, {...context, restoreBundle});
    expect(restoreBundle).toHaveBeenCalledOnce();
    expect(context.applyNativeConfig).toHaveBeenCalledOnce();
  } finally {
    await context.cleanup();
  }
});

test('invalid saved altitude roles reject before portable workspace mutation', async () => {
  const context = fixture();
  try {
    addPublication(context);
    const restoreBundle = vi.fn(async () => context.runtime);
    for (const altitude of ['missing', null, 10]) {
      const bundle = createBundle(context);
      bundle.state.sources[1].reference.parameters.coordinates.altitude = altitude;
      await expect(replaySandboxBundle(bundle, {...context, restoreBundle})).rejects.toThrow(
        /coordinate roles/
      );
    }
    expect(restoreBundle).not.toHaveBeenCalled();
    expect(context.applyNativeConfig).not.toHaveBeenCalled();
  } finally {
    await context.cleanup();
  }
});
