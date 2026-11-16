// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {beforeAll, describe, expect, test} from 'vitest';
import {
  applyKeplerDatasetProfileDeclarations,
  getCatalogBinding,
  getKeplerOperationDescriptor,
  getWorkflowOperationRegistry,
  importWorkflowJSON,
  exportWorkflowJSON,
  makeWorkflowDataFromFeatures,
  makeWorkflowData,
  makeWorkflowOperationFromCatalogBinding,
  profileKeplerWorkbenchData,
  profileKeplerFeatureCollection,
  profileKeplerCatalogOutput,
  parseKeplerDatasetProfileDeclarations,
  serializeKeplerDatasetProfileDeclarations,
  getWorkflowAnalysisProfiles,
  runCatalogOperationAsync,
  removeWorkflowSteps,
  resolveKeplerOperationParameters,
  WorkflowRunner,
  type CatalogFeatureCollection,
  type KeplerDatasetProfile
} from '../src/index';
import {AnalysisResultCard} from '../../../examples/deck/kepler-compute/src/analysis-result-card';
import {
  createAnalysisResultCardModel,
  createAnalysisResultNextDraft,
  addAnalysisResultNextStep,
  getAnalysisResultNextActions,
  makeAnalysisResultWorkflow,
  hasAnalysisResultStackEntry,
  keepAnalysisResultWorkflow,
  type AnalysisResultCardModel
} from '../../../examples/deck/kepler-compute/src/analysis-result-card-model';
import {makeFixtureHost} from './workflows/workflow-fixtures';
import {createWorkbenchStore} from '../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {
  createAnalysisCatalogFormModel,
  resolveAnalysisCatalogInputProfiles
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';

import {CARD_IDS, CARD_SENTENCES, createReviewCard, type CardId} from './ax-result-card-fixtures';

const fixtures = new Map<
  CardId,
  {model: AnalysisResultCardModel; inputs: Record<string, CatalogFeatureCollection>}
>();

beforeAll(async () => {
  for (const id of CARD_IDS) fixtures.set(id, await createReviewCard(id));
}, 120000);

function renderCard(model: AnalysisResultCardModel, inStack = false) {
  return renderToStaticMarkup(
    createElement(AnalysisResultCard, {
      model,
      inStack,
      onNext: () => {},
      onAddToStack: () => {},
      onSaveWorkflow: () => {}
    })
  );
}

describe('AX-41 twelve documented result cards', () => {
  test.each(CARD_IDS)('%s renders its actual run sentence, choices, cautions and engine', id => {
    const {model} = fixtures.get(id)!;
    expect(model.report.sentence).toBe(CARD_SENTENCES[id]);
    expect({
      sentence: model.report.sentence,
      otherSentences: model.report.outputs
        .filter(output => output.sentence !== model.report.sentence)
        .map(output => output.sentence),
      chosen: model.report.chosen,
      cautions: model.report.cautions.map(caution => caution.id),
      engine: model.report.engine,
      fallback: model.report.fallbackReason
    }).toMatchSnapshot();
    const markup = renderCard(model);
    expect(markup).toContain('data-testid="ax-result-card"');
    expect(markup).toContain(
      renderToStaticMarkup(
        createElement('p', {'data-testid': 'ax-result-sentence'}, model.report.sentence)
      )
    );
    expect(markup).toContain('What was chosen');
    expect(markup).toContain('All executed settings');
    expect(markup).toContain('Add to stack');
    expect(markup).toContain('Save as workflow');
    expect(markup).toContain('Copy as JSON');
    for (const caution of model.report.cautions) expect(markup).toContain(caution.title);
    expect(JSON.parse(model.json)).toEqual(
      JSON.parse(
        JSON.stringify({
          schemaVersion: 1,
          result: model.result,
          context: model.context,
          report: model.report
        })
      )
    );
  });

  test('the report retains run defaults after a later form edit and explains explicit overrides', () => {
    const {model} = fixtures.get('density-surface')!;
    const priorJSON = model.json;
    const binding = getCatalogBinding(model.result.operationId)!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const later = resolveKeplerOperationParameters(
      binding,
      descriptor,
      model.context.profiles as Record<string, KeplerDatasetProfile>,
      {overrides: {bandwidth: 987}}
    );
    const edited = createAnalysisResultCardModel(
      {...model.result, parameters: later.parameters},
      {...model.context, resolutions: later.resolutions}
    );
    expect(edited.report.chosen.join(' ')).toContain('set by you');
    expect(edited.report.chosen.join(' ')).toContain('987');
    expect(model.json).toBe(priorJSON);
    expect(model.report.chosen.join(' ')).not.toContain('987');
    expect(model.report.chosen.join(' ')).toMatch(/derived|bandwidth|distance/i);
  });

  test('a missing default statistic and no GPU are explained on the card', () => {
    const {model} = fixtures.get('density-surface')!;
    const withFallback = createAnalysisResultCardModel(
      {...model.result, fallbackReason: 'No GPU device supplied'},
      {
        ...model.context,
        resolutions: [{parameter: 'bandwidth', source: 'fallback', reason: 'No spacing statistic'}]
      }
    );
    expect(withFallback.report.chosen.join(' ')).toContain('fixed fallback was used');
    expect(withFallback.report.chosen.join(' ')).toContain('No spacing statistic');
    expect(renderCard(withFallback)).toContain(
      'No GPU was available, so the operation ran on the CPU.'
    );
  });

  test('a table-only coefficient report has a sentence and stays inspectable without a map', () => {
    const {model} = fixtures.get('regression-with-diagnostics')!;
    const table = model.result.tables.find(output => !output.geometryColumn)!;
    const report = createAnalysisResultCardModel({...model.result, tables: [table]}, model.context);
    expect(report.result.tables).toHaveLength(1);
    expect(report.result.tables[0].geometryColumn).toBeUndefined();
    expect(report.report.sentence.length).toBeGreaterThan(10);
    expect(renderCard(report)).toContain('ax-result-card');
  });

  test('Next checks actual output assignments, shows remedies and blocks an incompatible input', () => {
    const {model} = fixtures.get('density-surface')!;
    const profile = profileKeplerWorkbenchData({
      kind: 'table',
      table: {
        id: 'coefficients',
        label: 'Coefficients',
        rowCount: 1,
        columns: {coefficient: [2]},
        columnTypes: {coefficient: 'float64'}
      }
    });
    const report = {
      ...model.report,
      next: [{question: 'where-concentrated' as const, label: 'Where are they concentrated?'}]
    };
    const actions = getAnalysisResultNextActions(report, [profile], 'cpu', [
      getCatalogBinding('kernel-density-grid')!
    ]);
    expect(actions).toHaveLength(1);
    expect(actions[0].applicability.status).toBe('blocked');
    expect(actions[0].applicability.missing[0].remedy).toBeTruthy();
    const markup = renderCard({...model, next: actions});
    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>Where are they concentrated/);
    const points = profileKeplerWorkbenchData(
      makeWorkflowDataFromFeatures(fixtures.get('density-surface')!.inputs['points'], {
        id: 'points',
        label: 'Points'
      })
    );
    const compatible = getAnalysisResultNextActions(report, [points], 'cpu', [
      getCatalogBinding('kernel-density-grid')!
    ]);
    expect(compatible[0].applicability.status).not.toBe('blocked');
    expect(Object.values(compatible[0].applicability.inputAssignments)).toContain('points');
    expect(renderCard({...model, next: compatible})).toContain('data-testid="ax-result-next"');
  });

  test('keeping a run uses history promotion, preserves every executed value and round-trips', () => {
    const {model, inputs} = fixtures.get('density-surface')!;
    const document = makeAnalysisResultWorkflow(model, inputs, getWorkflowOperationRegistry());
    const restored = importWorkflowJSON(exportWorkflowJSON(document));
    const step = restored.steps.find(
      candidate => candidate.operationId === model.result.operationId
    )!;
    expect(
      Object.fromEntries(
        Object.entries(step.parameters).map(([name, value]) => [
          name,
          value.from === 'static' ? value.value : undefined
        ])
      )
    ).toEqual(model.result.parameters);
    const source = restored.steps.find(candidate => candidate.operationId === 'source-geojson')!;
    expect(source.parameters['coordinateSystem']).toEqual({from: 'static', value: 'EPSG:4326'});
    expect(source.parameters['text']).toEqual({
      from: 'static',
      value: JSON.stringify(inputs['points'])
    });
    const declarations = source.parameters['profileDeclarations'];
    if (declarations.from !== 'static') throw new Error('Saved source declarations must be static');
    const savedDeclarations = parseKeplerDatasetProfileDeclarations(declarations.value as string);
    expect(savedDeclarations.rowSemantics).toBeUndefined();
    expect(savedDeclarations.measurement).toEqual({});
    expect(renderCard(model, true)).toContain('In stack');
  });

  test('Next opens the exact result rows in the catalog and wires that output in the workbench', () => {
    const {model, inputs} = fixtures.get('clusters-and-footprints')!;
    const table = model.result.tables.find(
      output => output.geometryColumn && output.rowCount === inputs['points'].features.length
    )!;
    const profile = applyKeplerDatasetProfileDeclarations(
      profileKeplerWorkbenchData(makeWorkflowData(table)),
      {coordinateModel: 'geographic'}
    );
    const actions = getAnalysisResultNextActions(
      {
        ...model.report,
        next: [{question: 'where-concentrated', label: 'Where are they concentrated?'}]
      },
      [profile],
      'cpu',
      [getCatalogBinding('kernel-density-grid')!]
    );
    const action = actions[0];
    expect(action.applicability.status).not.toBe('blocked');
    const draft = createAnalysisResultNextDraft(model, action);
    expect(draft.operationId).toBe('kernel-density-grid');
    expect(JSON.parse(draft.drafts['kernel-density-grid:points'].text).features).toHaveLength(
      table.rowCount
    );
    expect(draft.overrides['coordinateSystem']).toBe('EPSG:4326');
    const registry = getWorkflowOperationRegistry();
    const document = makeAnalysisResultWorkflow(model, inputs, registry);
    const source = document.steps.find(step => step.operationId === model.result.operationId)!;
    const added = addAnalysisResultNextStep(document, registry, action, {
      stepId: source.id,
      output: table.id
    });
    expect(added.document.steps.at(-1)?.inputs['points']).toEqual({
      stepId: source.id,
      output: table.id
    });
    expect(() =>
      createAnalysisResultNextDraft(model, {
        ...action,
        applicability: {...action.applicability, status: 'blocked'}
      })
    ).toThrow('blocked');
  });

  test('a saved run re-executes through the existing workflow runner', async () => {
    const {model, inputs} = fixtures.get('clusters-and-footprints')!;
    const registry = getWorkflowOperationRegistry();
    const document = importWorkflowJSON(
      exportWorkflowJSON(makeAnalysisResultWorkflow(model, inputs, registry))
    );
    const runner = new WorkflowRunner({host: makeFixtureHost(), registry});
    try {
      runner.setDocument(document);
      const run = await runner.run();
      expect(run.steps.map(step => ({status: step.status, error: step.error}))).toEqual(
        document.steps.map(() => ({status: 'done', error: undefined}))
      );
      const step = document.steps.find(
        candidate => candidate.operationId === model.result.operationId
      )!;
      expect(
        Object.values(runner.getState().steps[step.id].result!.outputs).map(output => output.table)
      ).toEqual(
        model.result.tables.map(table => {
          // Workflow feature sources assign row-index identities when the original sample had
          // no ids. Numerical columns, geometry and source_index must still match exactly.
          return table.columns['source_id']
            ? {
                ...table,
                columns: {
                  ...table.columns,
                  source_id: Array.from({length: table.rowCount}, (_, index) => String(index))
                }
              }
            : table;
        })
      );
    } finally {
      await runner.destroy();
    }
  });

  test('workbench output retains the exact run report context independently of parameter edits', async () => {
    const {model, inputs} = fixtures.get('clusters-and-footprints')!;
    const operation = makeWorkflowOperationFromCatalogBinding(
      getCatalogBinding(model.result.operationId)!
    );
    const data = makeWorkflowDataFromFeatures(inputs['points'], {id: 'points', label: 'Points'});
    data.analysisProfile = model.context.profiles!['points'] as KeplerDatasetProfile;
    const output = await operation.run(
      {
        device: null,
        engine: 'cpu',
        host: makeFixtureHost(),
        signal: new AbortController().signal,
        workflowId: 'review',
        stepId: 'clusters',
        parameterOverrides: model.result.parameters
      },
      {points: data},
      model.result.parameters
    );
    const retained = Object.values(output.outputs)[0];
    expect(retained.analysisRun?.result.operationId).toBe(model.result.operationId);
    expect(retained.analysisRun?.context.profiles?.['points']).toEqual(data.analysisProfile);
    expect(retained.analysisRun?.context.fieldBindings).toBeDefined();
    expect(retained.analysisRun?.context.resolutions?.length).toBeGreaterThan(0);
    const card = createAnalysisResultCardModel(
      retained.analysisRun!.result,
      retained.analysisRun!.context,
      {report: retained.analysisReport}
    );
    expect(card.report).toBe(retained.analysisReport);
    expect(renderCard(card, true)).toContain('In stack');
  });
});

/** Values resemble longitude/latitude, but their declared units are deliberately planar. */
function makeDeclaredPlanarSamples(): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: Array.from({length: 12}, (_, index) => ({
      type: 'Feature',
      id: `sample-${index}`,
      geometry: {
        type: 'Point',
        coordinates: [-87.65 + (index % 4) * 0.02, 41.85 + Math.floor(index / 4) * 0.02]
      },
      properties: {price: 10 + index * index}
    }))
  };
}

describe('AX-41 independent review regressions', () => {
  test('F1: aggregation Next retains declared planar units through applicability and catalog drafts', async () => {
    const points = makeDeclaredPlanarSamples();
    const profile = applyKeplerDatasetProfileDeclarations(
      profileKeplerFeatureCollection(points, {datasetId: 'planar-points', label: 'Planar points'}),
      {coordinateModel: 'planar', rowSemantics: 'events'}
    );
    const result = await runCatalogOperationAsync(
      getCatalogBinding('cell-aggregation')!,
      {cellSize: 0.01, valueProperty: 'price'},
      {points},
      {engine: 'cpu'}
    );
    expect(result.parameters['coordinateSystem']).toBeUndefined();
    const model = createAnalysisResultCardModel(
      result,
      {profiles: {points: profile}},
      {featureInputsOnly: true}
    );
    expect(model.outputProfiles[0].coordinates).toMatchObject({
      model: 'planar',
      source: 'declared'
    });
    const enabled = model.next.filter(action => action.applicability.status !== 'blocked');
    expect(enabled.length).toBeGreaterThan(0);
    for (const action of enabled) {
      const binding = getCatalogBinding(action.bindingId)!;
      const next = createAnalysisResultNextDraft(model, action);
      const input = Object.entries(action.applicability.inputAssignments).find(
        ([, datasetId]) => datasetId === action.tableId
      )![0];
      const draft = next.drafts[`${binding.id}:${input}`];
      expect(parseKeplerDatasetProfileDeclarations(draft.profileDeclarations!)).toEqual({
        coordinateModel: 'planar'
      });
      const checked = resolveAnalysisCatalogInputProfiles(binding, next.drafts, () => undefined);
      expect(checked.profiles[input]?.coordinates).toMatchObject({
        model: 'planar',
        source: 'declared'
      });
      const form = createAnalysisCatalogFormModel(binding, checked.profiles, {
        overrides: next.overrides,
        executionEngine: 'cpu'
      });
      if (binding.parameters.some(parameter => parameter.name === 'coordinateSystem')) {
        expect(action.applicability.parameters['coordinateSystem']).toBe('planar');
        expect(next.overrides['coordinateSystem']).toBe('planar');
        expect(form.resolved.parameters['coordinateSystem']).toBe('planar');
      }
    }
  });

  test('F1: output CRS and projection direction take precedence; conflicting or transformed lineage stays unconfirmed', () => {
    const points = makeDeclaredPlanarSamples();
    const data = makeWorkflowDataFromFeatures(points, {id: 'cells', label: 'Cells'});
    const profile = applyKeplerDatasetProfileDeclarations(profileKeplerWorkbenchData(data), {
      coordinateModel: 'planar'
    });
    const descriptor = getKeplerOperationDescriptor('cell-aggregation')!;
    const geographic = applyKeplerDatasetProfileDeclarations(profile, {
      coordinateModel: 'geographic'
    });
    const conflict = profileKeplerCatalogOutput(
      data,
      {operationId: 'cell-aggregation', parameters: {}},
      {first: profile, second: geographic},
      descriptor
    );
    expect(conflict.coordinates?.source).toBe('guess');
    const explicit = profileKeplerCatalogOutput(
      {...data, table: {...data.table, coordinateSystem: 'planar'}},
      {operationId: 'cell-aggregation', parameters: {coordinateSystem: 'EPSG:4326'}},
      {points: geographic},
      descriptor
    );
    expect(explicit.coordinates).toMatchObject({model: 'planar', source: 'declared'});
    for (const [direction, model] of [
      ['forward', 'planar'],
      ['inverse', 'geographic']
    ] as const) {
      const projected = profileKeplerCatalogOutput(
        data,
        {operationId: 'geometry-local-frame-projection', parameters: {direction}},
        {features: profile},
        getKeplerOperationDescriptor('geometry-local-frame-projection')!
      );
      expect(projected.coordinates).toMatchObject({model, source: 'declared'});
    }
    const transformed = profileKeplerCatalogOutput(
      data,
      {operationId: 'affine-transform', parameters: {}},
      {features: profile},
      getKeplerOperationDescriptor('affine-transform')!
    );
    expect(transformed.coordinates?.source).toBe('guess');
    const inherited = profileKeplerCatalogOutput(
      data,
      {operationId: 'cell-aggregation', parameters: {}},
      {points: profile, other: geographic},
      {...descriptor, outputs: descriptor.outputs.map(output => ({...output, kindFrom: 'points'}))}
    );
    expect(inherited.coordinates).toMatchObject({model: 'planar', source: 'declared'});
  });

  test('F2: saved IDW sources restore declared surface samples and measurements that contradict guesses', async () => {
    const points = makeDeclaredPlanarSamples();
    const inferred = profileKeplerFeatureCollection(points, {datasetId: 'prices', label: 'Prices'});
    expect(inferred.rowSemantics?.value).toBe('objects');
    expect(inferred.rowSemantics?.source).toBe('guess');
    const declared = applyKeplerDatasetProfileDeclarations(inferred, {
      coordinateModel: 'planar',
      rowSemantics: 'surface-samples',
      measurement: {price: 'amount'}
    });
    const binding = getCatalogBinding('idw-grid')!;
    const form = createAnalysisCatalogFormModel(
      binding,
      {points: declared},
      {
        executionEngine: 'cpu',
        overrides: {valueProperty: 'price', cellSize: 0.01, searchRadius: 0.1}
      }
    );
    expect(form.status).toBe('ready');
    const result = await runCatalogOperationAsync(
      binding,
      form.resolved.parameters,
      {points},
      {engine: 'cpu'}
    );
    const model = createAnalysisResultCardModel(result, {
      profiles: {points: declared},
      fieldBindings: form.applicability.fieldBindings,
      resolutions: form.resolved.resolutions
    });
    const registry = getWorkflowOperationRegistry();
    const document = importWorkflowJSON(
      exportWorkflowJSON(makeAnalysisResultWorkflow(model, {points}, registry))
    );
    const source = document.steps.find(step => step.operationId === 'source-geojson')!;
    const operation = document.steps.find(step => step.operationId === 'idw-grid')!;
    const declarationBinding = source.parameters['profileDeclarations'];
    expect(declarationBinding.from).toBe('static');
    if (declarationBinding.from !== 'static')
      throw new Error('F2: source declarations must be static');
    expect(parseKeplerDatasetProfileDeclarations(declarationBinding.value as string)).toEqual({
      coordinateModel: 'planar',
      rowSemantics: 'surface-samples',
      measurement: {price: 'amount'}
    });
    expect(
      getWorkflowAnalysisProfiles(document, operation, registry)['points']?.rowSemantics
    ).toMatchObject({value: 'surface-samples', source: 'declared'});
    // A declaration-only edit must invalidate the planner's embedded-source profile cache.
    const withoutMeaning = {
      ...document,
      steps: document.steps.map(step =>
        step.id === source.id
          ? {
              ...step,
              parameters: {
                ...step.parameters,
                profileDeclarations: {
                  from: 'static' as const,
                  value: serializeKeplerDatasetProfileDeclarations({
                    coordinateModel: 'planar',
                    measurement: {price: 'amount'}
                  })
                }
              }
            }
          : step
      )
    };
    expect(
      getWorkflowAnalysisProfiles(withoutMeaning, operation, registry)['points']?.rowSemantics
    ).toMatchObject({value: 'objects', source: 'guess'});
    const runner = new WorkflowRunner({registry, host: makeFixtureHost()});
    try {
      runner.setDocument(document);
      const run = await runner.run();
      expect(run.steps.map(step => ({status: step.status, error: step.error}))).toEqual(
        document.steps.map(() => ({status: 'done', error: undefined}))
      );
      const restored = runner.getState().steps[source.id].result!.outputs['data'].analysisProfile!;
      expect(restored.rowSemantics).toMatchObject({value: 'surface-samples', source: 'declared'});
      expect(restored.coordinates).toMatchObject({model: 'planar', source: 'declared'});
      expect(restored.columns.find(column => column.name === 'price')).toMatchObject({
        measurement: 'amount',
        measurementSource: 'declared'
      });
      expect(
        Object.values(runner.getState().steps[operation.id].result!.outputs).map(
          output => output.table
        )
      ).toEqual(result.tables);
    } finally {
      await runner.destroy();
    }
  });

  test('F2: malformed source declarations fail normalization before execution', () => {
    const source = getWorkflowOperationRegistry().get('source-geojson')!;
    expect(() =>
      source.normalizeParameters({
        text: JSON.stringify(makeDeclaredPlanarSamples()),
        profileDeclarations: '{"version":1,"rowSemantics":"not-a-semantics"}'
      })
    ).toThrow(/GeoJSON: profileDeclarations/);
  });

  test('F2: identical input rows with different declarations retain separate promoted sources', async () => {
    const points = makeDeclaredPlanarSamples();
    const inferred = profileKeplerFeatureCollection(points, {datasetId: 'points', label: 'Points'});
    const first = applyKeplerDatasetProfileDeclarations(inferred, {
      coordinateModel: 'planar',
      rowSemantics: 'events',
      measurement: {price: 'amount'}
    });
    const second = applyKeplerDatasetProfileDeclarations(inferred, {
      coordinateModel: 'planar',
      rowSemantics: 'objects',
      measurement: {price: 'rate'}
    });
    const inputs = {points, targets: points};
    const result = await runCatalogOperationAsync(
      getCatalogBinding('nearest-feature-join')!,
      {coordinateSystem: 'planar'},
      inputs,
      {engine: 'cpu'}
    );
    const model = createAnalysisResultCardModel(result, {
      profiles: {points: first, targets: second}
    });
    const document = makeAnalysisResultWorkflow(model, inputs, getWorkflowOperationRegistry());
    const operation = document.steps.find(step => step.operationId === result.operationId)!;
    expect(operation.inputs['points'].stepId).not.toBe(operation.inputs['targets'].stepId);
    for (const [name, rowSemantics, measurement] of [
      ['points', 'events', 'amount'],
      ['targets', 'objects', 'rate']
    ] as const) {
      const source = document.steps.find(step => step.id === operation.inputs[name].stepId)!;
      const binding = source.parameters['profileDeclarations'];
      if (binding.from !== 'static')
        throw new Error('F2: promoted source declarations must be static');
      expect(parseKeplerDatasetProfileDeclarations(binding.value as string)).toMatchObject({
        rowSemantics,
        measurement: {price: measurement}
      });
    }
  });

  test('F3: stack membership follows remapped insertion IDs, undo, redo, removal and deletion', async () => {
    const {model, inputs} = fixtures.get('clusters-and-footprints')!;
    const registry = getWorkflowOperationRegistry();
    const store = createWorkbenchStore({host: makeFixtureHost(), registry});
    const document = makeAnalysisResultWorkflow(model, inputs, registry);
    try {
      store.addWorkflow(document);
      const previousIds = new Set(store.getState().document!.steps.map(step => step.id));
      const entry = keepAnalysisResultWorkflow(document, store, false);
      expect(entry.stepIds.length).toBe(document.steps.length);
      expect(entry.stepIds.every(stepId => !previousIds.has(stepId))).toBe(true);
      const membership = () => hasAnalysisResultStackEntry(entry, store.getState().workflows);
      const updates: boolean[] = [];
      const unsubscribe = store.subscribe(() => updates.push(membership()));
      expect(membership()).toBe(true);
      store.undo();
      expect(membership()).toBe(false);
      expect(renderCard(model, membership())).toContain('Add to stack');
      expect(renderCard(model, membership())).not.toContain('In stack');
      store.redo();
      expect(membership()).toBe(true);
      store.edit('remove-inserted-source', current =>
        removeWorkflowSteps(current, [entry.stepIds[0]])
      );
      expect(membership()).toBe(false);
      store.undo();
      expect(membership()).toBe(true);
      store.createWorkflow({title: 'Another stack'});
      expect(membership()).toBe(true);
      store.deleteWorkflow(entry.workflowId);
      expect(membership()).toBe(false);
      const reinserted = keepAnalysisResultWorkflow(document, store, false);
      expect(hasAnalysisResultStackEntry(reinserted, store.getState().workflows)).toBe(true);
      expect(updates).toContain(false);
      expect(updates).toContain(true);
      unsubscribe();
    } finally {
      await store.clear();
    }
  });
});
