// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {describe, expect, test, vi} from 'vitest';
import {
  buildKeplerCuratedAnalysisRequest,
  createCuratedWorkflowAnalysisTemplate,
  createKeplerCuratedAnalysisMapping,
  getCatalogBinding,
  getKeplerCuratedAnalysis,
  getKeplerWorkbenchCuratedPresets,
  getWorkflowOperationRegistry,
  getWorkflowFeatures,
  makeWorkflowDataFromFeatures,
  profileKeplerFeatureCollection,
  replayKeplerCuratedAnalysisMapping,
  evaluateKeplerWorkbenchInputs,
  countCatalogCatchmentSeedCollisions,
  type CatalogFeatureCollection,
  type KeplerDatasetProfile,
  type WorkflowRunRecord
} from '@luma-contributors/kepler';
import {makeFixtureHost} from '../workflow-fixtures';
import {createWorkbenchStore} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {WorkbenchAnalysisGallery} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-gallery';
import {WorkbenchAnalysisMapping} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-mapping';
import {
  createWorkbenchAnalysisDocument,
  createWorkbenchSandboxAnalysis,
  getWorkbenchAnalysisMappingIssues,
  getWorkbenchMappingParameterRows,
  launchWorkbenchAnalysis,
  launchWorkbenchSandboxAnalysis,
  updateWorkbenchAnalysisMapping
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-launch';
import {
  getWorkbenchAnalysisSampleData,
  getWorkbenchCuratedSampleId,
  getWorkbenchCuratedSampleParameters,
  getWorkbenchSandboxSampleId
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-analysis-samples';
import {
  SANDBOX_SAMPLES,
  createSandboxSample
} from '../../../../../examples/deck/kepler-compute/src/sandbox-samples';

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

function getAnalysis(analysisId: string) {
  const analysis = getKeplerCuratedAnalysis(analysisId);
  if (!analysis) throw new Error(`Fixture analysis missing: ${analysisId}`);
  return analysis;
}

function makeStore(gpuAvailable = false) {
  const host = makeFixtureHost();
  if (gpuAvailable) host.getDevice = () => ({}) as NonNullable<ReturnType<typeof host.getDevice>>;
  return createWorkbenchStore({host, registry: getWorkflowOperationRegistry()});
}

function replaceExecution(store: ReturnType<typeof makeStore>) {
  return vi.spyOn(store, 'run').mockImplementation(
    async () =>
      ({
        schemaVersion: 1,
        runId: 'fixture-run',
        workflowId: store.getState().activeWorkflowId!,
        startedAt: 0,
        durationMilliseconds: 0,
        status: 'done',
        steps: []
      }) satisfies WorkflowRunRecord
  );
}

const COLLECTION: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: Array.from({length: 12}, (_, index) => ({
    type: 'Feature',
    geometry: {
      type: 'Point',
      coordinates: [100 + (index % 4) * 2, 200 + Math.floor(index / 4) * 3]
    },
    properties: {renamed_amount: index + 1, caption: `row-${index}`}
  }))
};

function makeProfile(datasetId = 'observations'): KeplerDatasetProfile {
  return {
    ...profileKeplerFeatureCollection(COLLECTION, {
      datasetId,
      label: datasetId,
      coordinateSystem: 'planar',
      declarations: {rowSemantics: 'events'}
    }),
    geometryColumns: ['geometry']
  };
}

function makeDataStore(profile = makeProfile()) {
  const data = makeWorkflowDataFromFeatures(COLLECTION, {
    id: profile.datasetId,
    label: profile.label
  });
  data.analysisProfile = profile;
  const host = makeFixtureHost(
    [
      {
        id: profile.datasetId,
        label: profile.label,
        rowCount: profile.rowCount,
        resident: false,
        fields: Object.entries(data.table.columnTypes).map(([name, type]) => ({name, type})),
        geometryFields: ['geometry']
      }
    ],
    {[profile.datasetId]: data}
  );
  host.getAnalysisProfile = datasetId => (datasetId === profile.datasetId ? profile : undefined);
  return createWorkbenchStore({host, registry: getWorkflowOperationRegistry()});
}

function renderMapping(
  analysisId: string,
  mapping: ReturnType<typeof createKeplerCuratedAnalysisMapping>,
  profiles: KeplerDatasetProfile[],
  executionIssues: string[] = []
) {
  return renderToStaticMarkup(
    createElement(WorkbenchAnalysisMapping, {
      analysis: getAnalysis(analysisId),
      mapping,
      profiles,
      executionIssues,
      onChange: () => {},
      onParametersChange: () => {},
      onRun: () => {},
      onCancel: () => {}
    })
  );
}

const enabledSamples = getKeplerWorkbenchCuratedPresets().filter(
  preset => preset.sampleGeoJSONByInput
);

describe('question gallery and mapping', () => {
  test('uses the shared questions and retains every current curated template', () => {
    const markup = renderToStaticMarkup(
      createElement(WorkbenchAnalysisGallery, {store: makeStore()})
    );
    expect(markup).toContain('data-testid="ax-workflow-gallery"');
    expect(markup).toContain('Where are things concentrated?');
    for (const preset of getKeplerWorkbenchCuratedPresets())
      expect(markup).toContain(`data-testid="ax-workflow-${preset.curatedAnalysisId}"`);
    expect(markup).toContain('MIT');
    expect(markup).toContain('vis.gl contributors');
  });

  test('renders the shared counter and disables wrong-type columns with an explanation', () => {
    const profiles = [makeProfile()];
    const mapping = createKeplerCuratedAnalysisMapping(getAnalysis('idw-surface'), profiles);
    const markup = renderMapping('idw-surface', mapping, profiles);
    expect(markup).toContain(`${mapping.progress.mapped} of ${mapping.progress.required} mapped`);
    expect(markup).toMatch(/<option[^>]*value="caption"[^>]*disabled=""[^>]*title="[^"]+"/);
    expect(markup).toMatch(/<option[^>]*value="renamed_amount"(?![^>]*disabled)/);
  });

  test('dataset switches invalidate CRS/time confirmations and retained overrides stay editable', async () => {
    const analysis = getAnalysis('density-surface');
    const planar = makeProfile('planar');
    const guessed: KeplerDatasetProfile = {
      ...planar,
      datasetId: 'guessed',
      coordinates: {model: 'geographic', source: 'guess'}
    };
    const initial = createKeplerCuratedAnalysisMapping(analysis, [guessed], {
      parameters: {coordinateSystem: 'EPSG:4326', timestampEncoding: 'unix-millisecond'}
    });
    const changed = updateWorkbenchAnalysisMapping(
      initial,
      {slot: 'points', datasetId: 'planar'},
      analysis,
      [guessed, planar]
    );
    expect(changed.explicitParameterNames).not.toContain('coordinateSystem');
    expect(changed.explicitParameterNames).not.toContain('timestampEncoding');
    expect(
      buildKeplerCuratedAnalysisRequest(analysis, changed).parameters['coordinateSystem']
    ).not.toBe('EPSG:4326');
    const retained = replayKeplerCuratedAnalysisMapping(analysis, changed.choices, [planar], {
      parameters: {coordinateSystem: 'EPSG:4326'}
    });
    expect(renderMapping(analysis.id, retained, [planar])).toContain(
      'data-testid="ax-mapping-coordinate-system"'
    );
    const store = makeDataStore(planar);
    const record = await launchWorkbenchAnalysis(
      createCuratedWorkflowAnalysisTemplate(analysis),
      store,
      changed
    );
    expect(record.status).toBe('done');
    expect(store.getState().runner?.steps['analysis'].result?.parameters['coordinateSystem']).toBe(
      'planar'
    );
  });

  test('complete numeric track mapping exposes timestamp units and blocks Run until supplied', () => {
    const analysis = getAnalysis('track-quality');
    const source = getWorkbenchAnalysisSampleData(
      getWorkbenchCuratedSampleId(analysis.id, 'tracks')
    )!;
    const profile = {...source.analysisProfile!, datasetId: 'user-tracks'};
    const store = makeStore(true);
    const fields = Object.entries(source.table.columnTypes).map(([name, type]) => ({name, type}));
    store.host.listDatasets = () => [
      {
        id: profile.datasetId,
        label: profile.label,
        rowCount: profile.rowCount,
        fields,
        resident: false,
        geometryFields: ['geometry']
      }
    ];
    store.host.getAnalysisProfile = () => profile;
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    expect(mapping.progress).toEqual({mapped: 3, required: 3});
    expect(mapping.ready).toBe(true);
    expect(
      getWorkbenchMappingParameterRows(analysis, mapping, [profile]).map(row => row.spec.name)
    ).toContain('timestampEncoding');
    const template = createCuratedWorkflowAnalysisTemplate(analysis);
    const issues = getWorkbenchAnalysisMappingIssues(template, store, mapping);
    expect(issues.join(' ')).toContain('timestampEncoding');
    const markup = renderMapping(analysis.id, mapping, [profile], issues);
    expect(markup).toContain('ax-mapping-parameter-timestampEncoding');
    expect(markup).toMatch(/data-testid="ax-mapping-run"[^>]*disabled=""/);
    const confirmed = replayKeplerCuratedAnalysisMapping(analysis, mapping.choices, [profile], {
      parameters: {timestampEncoding: 'unix-millisecond'}
    });
    expect(getWorkbenchAnalysisMappingIssues(template, store, confirmed)).toEqual([]);
  });

  test.each([
    'drainage-network',
    'surface-preparation',
    'zonal-exposure',
    'regional-raster-change',
    'landscape-metrics',
    'dasymetric-redistribution',
    'cell-flow-topology'
  ])('launches typed sample %s through the native source adapter', async analysisId => {
    const analysis = getAnalysis(analysisId);
    const store = makeStore(true);
    const template = createCuratedWorkflowAnalysisTemplate(analysis);
    const document = createWorkbenchAnalysisDocument(template, store);
    for (const slot of analysis.slots) {
      const data = getWorkbenchAnalysisSampleData(
        getWorkbenchCuratedSampleId(analysisId, slot.name)
      )!;
      const source = document.steps.find(step => step.id === `source-${slot.name}`)!;
      const operation = store.registry.get('source-dataset')!;
      const parameters = Object.fromEntries(
        Object.entries(source.parameters).map(([name, value]) => [
          name,
          value.from === 'static' ? value.value : null
        ])
      );
      const result = await operation.run(
        {host: store.host, signal: new AbortController().signal} as Parameters<
          typeof operation.run
        >[0],
        {},
        operation.normalizeParameters(parameters)
      );
      expect(result.outputs['data']).toBe(data);
      if (slot.kind === 'surface') {
        expect(data.kind).toBe('raster');
        expect(data.raster?.band.values.length).toBeGreaterThan(0);
      } else if (slot.kind === 'attributes') {
        expect(data.kind).toBe('table');
        expect(data.table.geometryColumn).toBeUndefined();
      }
    }
    const run = replaceExecution(store);
    await launchWorkbenchAnalysis(template, store);
    expect(run).toHaveBeenCalledOnce();
  });

  test('uses source-dataset for the full emerging sample, beyond the text source limit', async () => {
    const analysis = getAnalysis('emerging-hot-spots');
    const store = makeStore(true);
    const document = createWorkbenchAnalysisDocument(
      createCuratedWorkflowAnalysisTemplate(analysis),
      store
    );
    const source = document.steps.find(step => step.operationId === 'source-dataset')!;
    expect(document.steps.some(step => step.operationId === 'source-geojson')).toBe(false);
    const binding = source.parameters['dataset'];
    if (binding?.from !== 'static') throw new Error('Sample source dataset missing');
    const data = await store.host.readDataset(
      {
        datasetId: String(binding.value),
        geometry: {mode: 'geojson-field', field: 'geometry'},
        population: 'all',
        maximumRows: 100000
      },
      new AbortController().signal
    );
    expect(data.table.rowCount).toBe(5472);
    expect(JSON.stringify(getWorkflowFeatures(data)).length).toBeGreaterThan(262144);
    expect(
      data.analysisProfile?.columns.find(column => column.name === 'observed')?.nullCount
    ).toBe(0);
  });

  test.each(
    enabledSamples.map(preset => [preset.curatedAnalysisId])
  )('launches enabled sample %s through real preflight before the execution boundary', async analysisId => {
    const store = makeStore(true);
    const execute = replaceExecution(store);
    const record = await launchWorkbenchAnalysis(
      createCuratedWorkflowAnalysisTemplate(getAnalysis(analysisId)),
      store
    );
    expect(record.status).toBe('done');
    expect(execute).toHaveBeenCalledOnce();
    const sources = store
      .getState()
      .document!.steps.filter(step => step.operationId === 'source-dataset');
    for (const source of sources) {
      const dataset = source.parameters['dataset'];
      if (dataset?.from !== 'static') throw new Error('Missing dataset');
      expect(getWorkbenchAnalysisSampleData(String(dataset.value))!.table.rowCount).toBeGreaterThan(
        0
      );
    }
  });

  test.each(
    enabledSamples
      .filter(preset => getCatalogBinding(preset.bindingId)!.engine !== 'gpu')
      .map(preset => [preset.curatedAnalysisId])
  )('executes the CPU path of sample %s to a result', async analysisId => {
    const store = makeStore();
    const record = await launchWorkbenchAnalysis(
      createCuratedWorkflowAnalysisTemplate(getAnalysis(analysisId)),
      store
    );
    expect(record.status).toBe('done');
    expect(
      Object.keys(store.getState().runner!.steps['analysis'].result!.outputs).length
    ).toBeGreaterThan(0);
  });

  test.each(
    SANDBOX_SAMPLES.map(sample => [sample.id])
  )('pairs original sample %s with its full population and one launch', async sampleId => {
    const store = makeStore(true);
    const execute = replaceExecution(store);
    const template = createWorkbenchSandboxAnalysis(sampleId);
    expect(template.title.endsWith('?')).toBe(true);
    expect(template.sample?.id).toBe(sampleId);
    expect((await launchWorkbenchSandboxAnalysis(store, sampleId)).status).toBe('done');
    expect(execute).toHaveBeenCalledOnce();
    expect(
      getWorkbenchAnalysisSampleData(getWorkbenchSandboxSampleId(sampleId))!.table.rowCount
    ).toBe(createSandboxSample(sampleId).table.numRows);
  });

  test.each(
    SANDBOX_SAMPLES.filter(sample => sample.id !== 'trip-observations').map(sample => [sample.id])
  )('executes original sample %s through the CPU runner', async sampleId => {
    const store = makeStore();
    expect((await launchWorkbenchSandboxAnalysis(store, sampleId)).status).toBe('done');
  });

  test('preserves exact H3 identities and invalid source rows', () => {
    const data = getWorkbenchAnalysisSampleData(getWorkbenchSandboxSampleId('h3-cells'))!;
    const original = createSandboxSample('h3-cells');
    const features = getWorkflowFeatures(data).features;
    expect(features.filter(feature => feature.geometry === null)).toHaveLength(2);
    for (let index = 0; index < features.length; index++) {
      const value: unknown = original.table.getChild(original.h3Column!)!.get(index);
      expect(features[index].properties?.['h3_cell']).toBe(
        typeof value === 'bigint' ? String(value) : value
      );
    }
  });

  test('exact catchment facts stop certifying readiness when either revision or grid changes', () => {
    const analysis = getAnalysis('straight-line-catchments');
    const store = makeStore(true);
    createWorkbenchAnalysisDocument(createCuratedWorkflowAnalysisTemplate(analysis), store);
    const facilities = getWorkbenchAnalysisSampleData(
      getWorkbenchCuratedSampleId(analysis.id, 'facilities')
    )!.analysisProfile!;
    const observations = getWorkbenchAnalysisSampleData(
      getWorkbenchCuratedSampleId(analysis.id, 'observations')
    )!.analysisProfile!;
    expect(facilities.catchmentSeeds?.collisionCount).toBe(0);
    const options = {
      executionEngine: 'gpu' as const,
      overrides: {
        ...getWorkbenchCuratedSampleParameters(analysis.id),
        cellSize: facilities.catchmentSeeds!.cellSize
      }
    };
    const binding = getCatalogBinding('recipe-straight-line-catchments')!;
    const ready = evaluateKeplerWorkbenchInputs(binding, {facilities, observations}, options);
    expect(ready.status).toBe('ready');
    for (const changed of [
      {facilities: {...facilities, revision: 99}, observations},
      {facilities, observations: {...observations, revision: 99}},
      {
        facilities: {...facilities, catchmentSeeds: {...facilities.catchmentSeeds!, cellSize: 1}},
        observations
      }
    ])
      expect(evaluateKeplerWorkbenchInputs(binding, changed, options).status).not.toBe('ready');
  });

  test('counts actual seed-cell collisions and validates grid settings', () => {
    const points = (coordinates: number[][]): CatalogFeatureCollection => ({
      type: 'FeatureCollection',
      features: coordinates.map(coordinates => ({
        geometry: {type: 'Point', coordinates},
        properties: {}
      }))
    });
    const inputs = {
      facilities: points([
        [0, 0],
        [0.1, 0.1],
        [3, 0]
      ]),
      observations: points([[4, 4]])
    };
    expect(
      countCatalogCatchmentSeedCollisions(inputs, {cellSize: 1, coordinateSystem: 'planar'})
    ).toBe(1);
    expect(() =>
      countCatalogCatchmentSeedCollisions(inputs, {cellSize: 0, coordinateSystem: 'planar'})
    ).toThrow(/cellSize/);
  });

  test('replays user mappings before adding a workflow when a field disappears', async () => {
    const profile = makeProfile();
    const analysis = getAnalysis('idw-surface');
    let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    mapping = updateWorkbenchAnalysisMapping(
      mapping,
      {slot: 'points', role: 'value', columns: ['renamed_amount']},
      analysis,
      [profile]
    );
    const store = makeDataStore({
      ...profile,
      columns: profile.columns.filter(column => column.name !== 'renamed_amount')
    });
    const add = vi.spyOn(store, 'addWorkflow');
    await expect(
      launchWorkbenchAnalysis(createCuratedWorkflowAnalysisTemplate(analysis), store, mapping)
    ).rejects.toThrow(/mapping must be ready/);
    expect(add).not.toHaveBeenCalled();
  });

  test('does not run another workflow when a subscriber changes activation during launch', async () => {
    const store = makeStore();
    const otherId = store.createWorkflow({title: 'Other workflow'});
    const execute = replaceExecution(store);
    const unsubscribe = store.subscribe(() => {
      if (store.getState().document?.id === 'density-surface') store.setActiveWorkflow(otherId);
    });
    await expect(
      launchWorkbenchAnalysis(
        createCuratedWorkflowAnalysisTemplate(getAnalysis('density-surface')),
        store
      )
    ).rejects.toThrow(/active workflow changed/);
    unsubscribe();
    expect(execute).not.toHaveBeenCalled();
  });

  test('rejects oversized mapped populations before adding a workflow', async () => {
    const analysis = getAnalysis('density-surface');
    const profile = {...makeProfile(), rowCount: 1000001};
    const store = makeDataStore(profile);
    const mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    const add = vi.spyOn(store, 'addWorkflow');
    await expect(
      launchWorkbenchAnalysis(createCuratedWorkflowAnalysisTemplate(analysis), store, mapping)
    ).rejects.toThrow(/100,?000|capacity|rows/i);
    expect(add).not.toHaveBeenCalled();
  });

  test('keeps ambiguous required fields disabled until the user confirms a choice', () => {
    const profile = makeProfile();
    const analysis = getAnalysis('idw-surface');
    const profiles = [
      {
        ...profile,
        columns: [
          ...profile.columns,
          {
            ...profile.columns.find(column => column.name === 'renamed_amount')!,
            name: 'other_amount'
          }
        ]
      }
    ];
    const mapping = createKeplerCuratedAnalysisMapping(analysis, profiles);
    expect(mapping.ready).toBe(false);
    expect(renderMapping(analysis.id, mapping, profiles)).toMatch(
      /data-testid="ax-mapping-run"[^>]*disabled=""/
    );
  });

  test('workbench network aliases preserve valid facts and never repair a mismatched fact', () => {
    const analysis = getAnalysis('drive-time-catchments');
    const store = makeStore(true);
    const document = createWorkbenchAnalysisDocument(
      createCuratedWorkflowAnalysisTemplate(analysis),
      store
    );
    const profiles = Object.fromEntries(
      analysis.slots.map(slot => [
        analysis.execution.status === 'available'
          ? analysis.execution.inputs[slot.name]
          : slot.name,
        getWorkbenchAnalysisSampleData(getWorkbenchCuratedSampleId(analysis.id, slot.name))!
          .analysisProfile!
      ])
    );
    const network = profiles['lines'];
    expect(network.network?.edgeCount).toBeGreaterThan(0);
    const originalIdentity = network.network!.datasetId;
    const options = {
      executionEngine: 'gpu' as const,
      overrides: getWorkbenchCuratedSampleParameters(analysis.id)
    };
    const binding = getCatalogBinding(
      document.steps.find(step => step.id === 'analysis')!.operationId
    )!;
    expect(evaluateKeplerWorkbenchInputs(binding, profiles, options).status).toBe('ready');
    expect(network.network!.datasetId).toBe(originalIdentity);
    expect(
      evaluateKeplerWorkbenchInputs(
        binding,
        {
          ...profiles,
          lines: {...network, network: {...network.network!, datasetId: 'another-source'}}
        },
        options
      ).status
    ).not.toBe('ready');
  });
});
