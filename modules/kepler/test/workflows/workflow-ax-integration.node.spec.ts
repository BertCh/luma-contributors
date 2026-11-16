// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {getCatalogBinding} from '../../src/analysis/catalog-bindings';
import {
  evaluateKeplerWorkbenchInputs,
  getKeplerWorkbenchParameterDefaults,
  profileKeplerWorkbenchData
} from '../../src/descriptors/workbench-adapters';
import {
  profileKeplerFeatureCollection,
  profileKeplerColumnTable,
  applyKeplerDatasetProfileDeclarations
} from '../../src/profile/dataset-profile';
import {
  encodeKeplerResultPresentation,
  getKeplerBindingResultPresentations
} from '../../src/presentation/result-presentation';
import {
  createWorkflowDocument,
  parseWorkflowDocument,
  serializeWorkflowDocument
} from '../../src/workflows/workflow-document';
import {addWorkflowStep, promoteWorkflowParameter} from '../../src/workflows/workflow-edit';
import {
  getWorkflowOperationRegistry,
  createWorkflowOperationRegistry,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {makeWorkflowData} from '../../src/workflows/workflow-data';
import type {WorkflowData, WorkflowRunContext} from '../../src/workflows/workflow-types';
import {planWorkflow} from '../../src/workflows/workflow-plan';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {
  getWorkbenchParameterRows,
  getWorkbenchAnalysisOptions,
  getWorkbenchColumnSuggestions,
  applyWorkbenchCuratedPreset
} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-model';
import {makeFixtureHost} from './workflow-fixtures';

const registry = getWorkflowOperationRegistry();
const collection = {
  type: 'FeatureCollection' as const,
  features: Array.from({length: 12}, (_, index) => ({
    type: 'Feature' as const,
    id: index,
    properties: {measurement: index + 1},
    geometry: {
      type: 'Point',
      coordinates: [-87.6 + (index % 4) * 0.001, 41.8 + Math.floor(index / 4) * 0.001]
    }
  }))
};
const profile = {
  ...profileKeplerFeatureCollection(collection, {
    datasetId: 'observations',
    label: 'Observations',
    coordinateSystem: 'EPSG:4326',
    declarations: {rowSemantics: 'events'}
  }),
  geometryColumns: ['geometry']
};

function makeDensityDocument() {
  let document = createWorkflowDocument({id: 'analysis-test', title: 'AX density'});
  const source = addWorkflowStep(document, registry, {
    operationId: 'source-dataset',
    parameters: {
      dataset: profile.datasetId,
      geometryMode: 'geojson-field',
      geometryField: 'geometry'
    }
  });
  const density = addWorkflowStep(source.document, registry, {
    operationId: 'kernel-density-grid',
    connectFrom: {stepId: source.stepId, output: 'data'},
    connectTo: 'points'
  });
  document = density.document;
  return {document, sourceId: source.stepId, densityId: density.stepId};
}

function makeHost() {
  const host = makeFixtureHost();
  host.listDatasets = () => [
    {
      id: profile.datasetId,
      label: profile.label,
      rowCount: profile.rowCount,
      resident: true,
      geometryFields: ['geometry'],
      fields: []
    }
  ];
  host.getAnalysisProfile = () => profile;
  host.readDataset = async () => ({
    kind: 'point',
    analysisProfile: profile,
    table: {
      id: profile.datasetId,
      label: profile.label,
      rowCount: collection.features.length,
      columns: {
        geometry: collection.features.map(feature => JSON.stringify(feature.geometry)),
        measurement: collection.features.map(feature => feature.properties.measurement)
      },
      columnTypes: {geometry: 'string', measurement: 'int32'},
      geometryColumn: 'geometry'
    }
  });
  return host;
}

describe('AX integrated with actual workbench contracts', () => {
  test('planner, generated form and runner use the same data defaults and retain provenance', async () => {
    const {document, densityId} = makeDensityDocument();
    const host = makeHost();
    const plan = planWorkflow(document, {registry, host});
    const planned = plan.steps[densityId];
    expect(planned.issues.filter(issue => issue.severity === 'error')).toEqual([]);
    expect(planned.analysis?.status).toBe('ready');
    expect(
      planned.analysis?.resolutions.find(entry => entry.parameter === 'cellSize')?.source
    ).toBe('data');
    const step = document.steps.find(candidate => candidate.id === densityId)!;
    const operation = registry.get(step.operationId)!;
    const rows = getWorkbenchParameterRows(
      document,
      step,
      operation,
      getWorkbenchAnalysisOptions(document, step, operation, registry, host)
    );
    expect(rows.find(row => row.spec.name === 'cellSize')?.value).toBe(
      planned.parameters?.['cellSize']
    );
    const runner = new WorkflowRunner({registry, host});
    runner.setDocument(document);
    await runner.run();
    const state = runner.getState().steps[densityId];
    expect(state.status, JSON.stringify(state.issues)).toBe('done');
    expect(state.result?.parameters).toEqual(planned.parameters);
    const output = state.result?.outputs['density-cells'];
    expect(output?.analysisReport?.chosen.join(' ')).toMatch(/extent|cell|Cell/);
    expect(output?.analysisPresentation?.presentation?.breaks).toBeDefined();
    expect(
      encodeKeplerResultPresentation(output!.table, output!.analysisPresentation!)?.legend
    ).toEqual(output!.analysisPresentation!.legend);
    runner.destroy();
  });

  test('unknown mandatory evidence prevents planning until supplied; explicit edits survive richer profiles', () => {
    const {document, densityId} = makeDensityDocument();
    const host = makeHost();
    host.getAnalysisProfile = () => ({...profile, coordinates: undefined, spacing: undefined});
    const blocked = planWorkflow(document, {registry, host}).steps[densityId];
    expect(blocked.status).toBe('not-configured');
    expect(blocked.issues.some(issue => issue.code === 'analysis-requirement')).toBe(true);
    document.steps.find(step => step.id === densityId)!.parameters['bandwidth'] = {
      from: 'static',
      value: 750
    };
    host.getAnalysisProfile = () => profile;
    const confirmed = planWorkflow(document, {registry, host}).steps[densityId];
    expect(confirmed.parameters?.['bandwidth']).toBe(750);
    expect(
      confirmed.analysis?.resolutions.find(entry => entry.parameter === 'bandwidth')?.source
    ).toBe('override');
  });

  test('assignments are port-specific, including the same dataset on two ports', () => {
    const binding = getCatalogBinding('recipe-points-in-polygons-choropleth')!;
    expect(binding).toBeDefined();
    const names = binding.inputs.map(input => input.name);
    const result = evaluateKeplerWorkbenchInputs(binding, {
      [names[0]]: profile,
      [names[1]]: profile
    });
    expect(result.status).toBe('blocked');
    expect(result.missing.map(entry => entry.what).join(' ')).toMatch(/polygon|area/i);
  });

  test('defaults read the named input, and typed roles restrict pre-run column choices', () => {
    const facilities = {...profile, extent: [-87.6, 41.8, -87.599, 41.801] as const};
    const observations = {...profile, extent: [-87.6, 41.8, -87.59, 41.81] as const};
    const getDefaults = (facilityProfile: typeof profile, observationProfile: typeof profile) =>
      getKeplerWorkbenchParameterDefaults('recipe-straight-line-catchments', undefined, {
        profilesByInput: {facilities: facilityProfile, observations: observationProfile}
      });
    const wide = getDefaults(facilities, observations);
    expect(wide.values['cellSize']).toBe(
      getDefaults(observations, observations).values['cellSize']
    );
    expect(wide.values['cellSize']).toBeGreaterThan(
      getDefaults(observations, facilities).values['cellSize'] as number
    );
    const {document, densityId} = makeDensityDocument();
    const step = document.steps.find(candidate => candidate.id === densityId)!;
    const operation = registry.get(step.operationId)!;
    const spec = operation.parameters.find(parameter => parameter.name === 'weightProperty')!;
    const choices = getWorkbenchColumnSuggestions(
      document,
      step,
      operation,
      spec,
      registry,
      null,
      [],
      {
        points: {
          ...profile,
          columns: [
            ...profile.columns,
            {name: 'description', type: 'category', rowCount: 12, nullCount: 0}
          ]
        }
      }
    );
    expect(choices.columns).toContain('measurement');
    expect(choices.columns).not.toContain('description');
  });

  test('actual runtime requirements cannot be bypassed by unavailable planning profiles', async () => {
    const {document, densityId} = makeDensityDocument();
    const host = makeHost();
    host.getAnalysisProfile = undefined;
    const readDataset = host.readDataset;
    host.readDataset = async (request, signal) => {
      const data = await readDataset(request, signal);
      return {...data, analysisProfile: {...profile, coordinates: undefined, spacing: undefined}};
    };
    const runner = new WorkflowRunner({registry, host});
    runner.setDocument(document);
    await runner.run();
    const state = runner.getState().steps[densityId];
    expect(state.status).toBe('error');
    expect(state.error).toMatchObject({code: 'analysis-requirement'});
    expect(host.published).toEqual([]);
    await runner.destroy();
  });

  test('a source revision during extraction cancels the obsolete run before publication', async () => {
    const {document, densityId} = makeDensityDocument();
    document.steps.find(step => step.id === densityId)!.display = ['density-cells'];
    const host = makeHost();
    let release!: () => void;
    let entered!: () => void;
    const blocked = new Promise<void>(resolve => {
      release = resolve;
    });
    const reading = new Promise<void>(resolve => {
      entered = resolve;
    });
    const readDataset = host.readDataset;
    host.readDataset = async (request, signal) => {
      entered();
      await blocked;
      return readDataset(request, signal);
    };
    const runner = new WorkflowRunner({registry, host});
    runner.setDocument(document);
    const running = runner.run();
    await reading;
    host.revision++;
    runner.refreshExternal();
    release();
    expect((await running).status).toBe('cancelled');
    expect(host.published).toEqual([]);
    await runner.destroy();
  });

  test('profile changes invalidate source keys through intervening table operations', async () => {
    const initial = makeDensityDocument();
    const filtered = addWorkflowStep(initial.document, registry, {
      operationId: 'table-filter-rows',
      parameters: {expression: 'measurement >= 0'},
      connectFrom: {stepId: initial.sourceId, output: 'data'},
      connectTo: 'rows'
    });
    filtered.document.steps.find(step => step.id === initial.densityId)!.inputs['points'] = {
      stepId: filtered.stepId,
      output: 'filtered'
    };
    const host = makeHost();
    const runner = new WorkflowRunner({registry, host});
    runner.setDocument(filtered.document);
    const record = await runner.run();
    expect(record.status, JSON.stringify(record)).toBe('done');
    const before = runner.getState().steps[initial.sourceId].key;
    host.getAnalysisProfile = () => ({
      ...profile,
      coordinates: {model: 'planar', source: 'declared'}
    });
    runner.refreshExternal();
    expect(runner.getState().steps[initial.sourceId].key).not.toBe(before);
    expect(runner.getState().steps[initial.densityId].status).toBe('stale');
    await runner.destroy();
  });

  test('an explicit catalog constant invalidates a derived result after an intermediate step', async () => {
    const source = addWorkflowStep(
      createWorkflowDocument({title: 'Explicit density size'}),
      registry,
      {
        operationId: 'source-geojson',
        parameters: {text: JSON.stringify(collection), coordinateSystem: 'EPSG:4326'}
      }
    );
    const filtered = addWorkflowStep(source.document, registry, {
      operationId: 'table-filter-rows',
      parameters: {expression: 'measurement >= 0'},
      connectFrom: {stepId: source.stepId, output: 'data'},
      connectTo: 'rows'
    });
    const density = addWorkflowStep(filtered.document, registry, {
      operationId: 'kernel-density-grid',
      connectFrom: {stepId: filtered.stepId, output: 'filtered'},
      connectTo: 'points'
    });
    const runner = new WorkflowRunner({registry, host: makeFixtureHost()});
    runner.setDocument(density.document);
    await runner.run();
    const before = runner.getState().steps[density.stepId];
    expect(before.status, JSON.stringify(before.issues)).toBe('done');
    expect(before.result?.parameters['cellSize']).not.toBe(100);
    const changed = parseWorkflowDocument(serializeWorkflowDocument(density.document));
    changed.steps.find(step => step.id === density.stepId)!.parameters['cellSize'] = {
      from: 'static',
      value: 100
    };
    runner.setDocument(changed);
    expect(runner.getState().steps[density.stepId].key).not.toBe(before.key);
    expect(runner.getState().steps[density.stepId].status).toBe('stale');
    await runner.run();
    expect(runner.getState().steps[density.stepId].result?.parameters['cellSize']).toBe(100);
    // Promoted controls must also distinguish their resolved explicit values from automatic ones.
    changed.inputs.push({id: 'size', label: 'Size', kind: 'number', value: 100});
    changed.steps.find(step => step.id === density.stepId)!.parameters['cellSize'] = {
      from: 'input',
      inputId: 'size'
    };
    expect(planWorkflow(changed, {registry}).steps[density.stepId].key).toBe(
      runner.getState().steps[density.stepId].key
    );
    changed.inputs[0].value = 120;
    expect(planWorkflow(changed, {registry}).steps[density.stepId].key).not.toBe(
      runner.getState().steps[density.stepId].key
    );
    await runner.destroy();
  });

  test('AX kriging parameters without static defaults remain editable and executable', async () => {
    const sampleProfile = {
      ...profileKeplerFeatureCollection(collection, {
        datasetId: profile.datasetId,
        label: profile.label,
        coordinateSystem: 'EPSG:4326',
        declarations: {rowSemantics: 'surface-samples', measurement: {measurement: 'amount'}}
      }),
      geometryColumns: ['geometry']
    };
    const host = makeHost();
    host.getAnalysisProfile = () => sampleProfile;
    const readDataset = host.readDataset;
    host.readDataset = async (...arguments_) => ({
      ...(await readDataset(...arguments_)),
      analysisProfile: sampleProfile
    });
    const initial = makeDensityDocument();
    initial.document.steps.find(step => step.id === initial.densityId)!.operationId =
      'kriging-grid';
    const plan = planWorkflow(initial.document, {registry, host}).steps[initial.densityId];
    expect(plan.analysis?.status, JSON.stringify(plan.analysis?.missing)).toBe('ready');
    expect(plan.status, JSON.stringify(plan.issues)).toBe('ready');
    expect(plan.parameters?.['sill']).toBe(13);
    const step = initial.document.steps.find(step => step.id === initial.densityId)!;
    const operation = registry.get('kriging-grid')!;
    const rows = getWorkbenchParameterRows(
      initial.document,
      step,
      operation,
      getWorkbenchAnalysisOptions(initial.document, step, operation, registry, host)
    );
    for (const parameter of ['sill', 'range', 'nugget']) {
      expect(operation.parameters.find(spec => spec.name === parameter)).toBeDefined();
      expect(operation.normalizeParameters({})).not.toHaveProperty(parameter);
    }
    expect(rows.find(row => row.spec.name === 'sill')?.value).toBe(13);
    expect(rows.find(row => row.spec.name === 'range')?.value).toBe(plan.parameters?.['range']);
    expect(operation.normalizeParameters({sill: '17', range: '120', nugget: '0.1'})).toMatchObject({
      sill: 17,
      range: 120,
      nugget: 0.1
    });
    expect(() => operation.normalizeParameters({sill: -1})).toThrow(/at least|between/);
    expect(() => promoteWorkflowParameter(initial.document, registry, step.id, 'sill')).toThrow(
      /Set .* before promoting/
    );
    const promoted = promoteWorkflowParameter(initial.document, registry, step.id, 'sill', {
      value: rows.find(row => row.spec.name === 'sill')!.value
    });
    expect(promoted.document.inputs.find(input => input.id === promoted.inputId)?.value).toBe(13);
    expect(
      planWorkflow(promoted.document, {registry, host}).steps[step.id].parameters?.['sill']
    ).toBe(13);

    const runner = new WorkflowRunner({registry, host});
    runner.setDocument(initial.document);
    await runner.run();
    const result = runner.getState().steps[initial.densityId];
    expect(result.status, JSON.stringify(result.issues)).toBe('done');
    expect(result.result?.parameters['sill']).toBe(13);
    expect(result.result?.outputs['kriging-cells'].table.rowCount).toBeGreaterThan(0);
    await runner.destroy();
  });

  test('column-dependent form defaults and provenance share applicability field assignments', () => {
    const sampleProfile = {
      ...profileKeplerFeatureCollection(collection, {
        datasetId: profile.datasetId,
        label: profile.label,
        coordinateSystem: 'EPSG:4326',
        declarations: {rowSemantics: 'surface-samples', measurement: {measurement: 'amount'}}
      }),
      geometryColumns: ['geometry']
    };
    const applicability = evaluateKeplerWorkbenchInputs(
      'validation-vecchia-likelihood',
      {points: sampleProfile},
      {executionEngine: 'gpu'}
    );
    expect(applicability.status, JSON.stringify(applicability.missing)).toBe('ready');
    expect(applicability.parameters['sill']).toBe(13);
    const defaults = getKeplerWorkbenchParameterDefaults(
      'validation-vecchia-likelihood',
      undefined,
      {profilesByInput: {points: sampleProfile}, executionEngine: 'gpu'}
    );
    expect(defaults.values).toEqual(applicability.parameters);
    expect(defaults.resolutions).toEqual(applicability.resolutions);
    const initial = makeDensityDocument();
    const step = initial.document.steps.find(candidate => candidate.id === initial.densityId)!;
    step.operationId = 'validation-vecchia-likelihood';
    const rows = getWorkbenchParameterRows(
      initial.document,
      step,
      registry.get(step.operationId)!,
      {profilesByInput: {points: sampleProfile}, executionEngine: 'gpu'}
    );
    expect(rows.find(row => row.spec.name === 'sill')?.value).toBe(13);
    expect(rows.find(row => row.spec.name === 'valueProperty')?.value).toBe('measurement');
    expect(rows.find(row => row.spec.name === 'sill')?.spec.description).toMatch(/data/);
  });

  test('runner boolean significance classes encode the same counts as the AX legend', async () => {
    const source = addWorkflowStep(
      createWorkflowDocument({title: 'Significance flags'}),
      registry,
      {
        operationId: 'source-geojson',
        parameters: {
          text: JSON.stringify({
            ...collection,
            features: collection.features.slice(0, 3).map((feature, index) => ({
              ...feature,
              properties: {probability: [0.001, 0.9, 0.002][index]}
            }))
          }),
          coordinateSystem: 'EPSG:4326'
        }
      }
    );
    const adjusted = addWorkflowStep(source.document, registry, {
      operationId: 'multiple-testing-adjustment',
      parameters: {pValueProperty: 'probability', method: 'bonferroni', alpha: 0.05},
      connectFrom: {stepId: source.stepId, output: 'data'},
      connectTo: 'features'
    });
    const runner = new WorkflowRunner({registry, host: makeFixtureHost()});
    runner.setDocument(adjusted.document);
    await runner.run();
    const state = runner.getState().steps[adjusted.stepId];
    expect(state.status, JSON.stringify(state.issues)).toBe('done');
    const data = state.result!.outputs['adjusted'];
    const encoded = encodeKeplerResultPresentation(data.table, data.analysisPresentation!)!;
    expect(encoded.values).toEqual(['Significant', 'Not significant', 'Significant']);
    for (let index = 0; index < encoded.legend.length; index++) {
      expect(encoded.values.filter(value => value === encoded.categories[index]).length).toBe(
        encoded.legend[index].count
      );
    }
    await runner.destroy();
  });

  test('a real other category stays distinct from the synthetic overflow category', () => {
    const values = [
      'other',
      'other',
      'other',
      ...Array.from({length: 17}, (_, index) => `kind-${index}`)
    ];
    const table = {
      id: 'categories',
      label: 'Categories',
      rowCount: values.length,
      columns: {kind: values},
      columnTypes: {kind: 'string' as const}
    };
    const presentation = getKeplerBindingResultPresentations(
      {
        operationId: 'categories',
        engine: 'cpu-reference',
        parameters: {},
        tables: [table],
        diagnostics: [],
        summary: {},
        dispatchedContributors: []
      },
      {
        ...registry.get('kernel-density-grid')!.analysisDescriptor!,
        outputs: [
          {
            tableId: table.id,
            kind: 'attributes',
            presentation: {column: 'kind', scale: 'categorical'},
            interpretation: {summary: 'value-range', cautions: []},
            next: []
          }
        ]
      }
    )[0];
    expect(presentation.presentation?.other?.count).toBe(11);
    const encoded = encodeKeplerResultPresentation(table, presentation)!;
    expect(encoded.values.slice(0, 3)).toEqual(['other', 'other', 'other']);
    for (let index = 0; index < encoded.legend.length; index++) {
      expect(encoded.values.filter(value => value === encoded.categories[index]).length).toBe(
        encoded.legend[index].count
      );
    }
  });

  test('a filter keeps confirmed measurements and sample semantics for downstream kriging', async () => {
    const sampleProfile = profileKeplerFeatureCollection(collection, {
      datasetId: profile.datasetId,
      label: profile.label,
      coordinateSystem: 'EPSG:4326',
      declarations: {rowSemantics: 'surface-samples', measurement: {measurement: 'amount'}}
    });
    const host = makeHost();
    host.getAnalysisProfile = () => ({...sampleProfile, geometryColumns: ['geometry']});
    const readDataset = host.readDataset;
    host.readDataset = async (...arguments_) => ({
      ...(await readDataset(...arguments_)),
      analysisProfile: sampleProfile
    });
    const initial = makeDensityDocument();
    const filtered = addWorkflowStep(initial.document, registry, {
      operationId: 'table-filter-rows',
      parameters: {expression: 'measurement >= 0'},
      connectFrom: {stepId: initial.sourceId, output: 'data'},
      connectTo: 'rows'
    });
    const kriging = filtered.document.steps.find(step => step.id === initial.densityId)!;
    kriging.operationId = 'kriging-grid';
    kriging.inputs['points'] = {stepId: filtered.stepId, output: 'filtered'};
    const runner = new WorkflowRunner({registry, host});
    runner.setDocument(filtered.document);
    await runner.run();
    const output = runner.getState().steps[filtered.stepId].result!.outputs['filtered'];
    expect(output.analysisProfile?.rowSemantics).toMatchObject({
      value: 'surface-samples',
      source: 'declared'
    });
    expect(
      output.analysisProfile?.columns.find(column => column.name === 'measurement')
    ).toMatchObject({
      measurement: 'amount',
      measurementSource: 'declared'
    });
    const state = runner.getState().steps[kriging.id];
    expect(state.status, JSON.stringify(state.issues)).toBe('done');
    expect(state.result?.parameters['sill']).toBe(13);
    await runner.destroy();
  });

  test('table edits recompute statistics and track declarations through renames and calculations', async () => {
    const host = makeHost();
    const input = await host.readDataset(
      {
        datasetId: profile.datasetId,
        geometry: {mode: 'geojson-field', field: 'geometry'},
        population: 'all',
        maximumRows: 12
      },
      new AbortController().signal
    );
    input.analysisProfile = applyKeplerDatasetProfileDeclarations(profile, {
      rowSemantics: 'surface-samples',
      measurement: {measurement: 'amount'}
    });
    const context: WorkflowRunContext = {
      device: null,
      engine: 'cpu',
      host,
      signal: new AbortController().signal,
      workflowId: 'declarations',
      stepId: 'table'
    };
    const run = async (
      id: string,
      data: WorkflowData,
      parameters: Record<string, string | number>
    ) => {
      const operation = registry.get(id)!;
      const result = await operation.run(
        context,
        {rows: data},
        operation.normalizeParameters(parameters)
      );
      return result.outputs[operation.outputs[0].name];
    };
    const filtered = await run('table-filter-rows', input, {expression: 'measurement >= 7'});
    expect(filtered.analysisProfile?.rowCount).toBe(6);
    const measured = filtered.analysisProfile?.columns.find(
      column => column.name === 'measurement'
    );
    expect(measured).toMatchObject({measurement: 'amount', measurementSource: 'declared'});
    expect(measured?.mean).toBe(9.5);
    const renamed = await run('table-select-columns', filtered, {
      columns: 'measurement',
      mode: 'keep',
      rename: 'measurement=observed'
    });
    expect(
      renamed.analysisProfile?.columns.find(column => column.name === 'observed')
    ).toMatchObject({
      measurement: 'amount',
      measurementSource: 'declared'
    });
    expect(renamed.analysisProfile?.columns.some(column => column.name === 'measurement')).toBe(
      false
    );
    const calculated = await run('table-calculate-column', renamed, {
      columnName: 'extra',
      expression: 'observed * 2'
    });
    expect(
      calculated.analysisProfile?.columns.find(column => column.name === 'observed')
        ?.measurementSource
    ).toBe('declared');
    expect(
      calculated.analysisProfile?.columns.find(column => column.name === 'extra')?.measurementSource
    ).not.toBe('declared');
    const replaced = await run('table-calculate-column', renamed, {
      columnName: 'observed',
      expression: 'observed * 2'
    });
    expect(
      replaced.analysisProfile?.columns.find(column => column.name === 'observed')
        ?.measurementSource
    ).not.toBe('declared');
    expect(replaced.analysisProfile?.rowSemantics).toMatchObject({
      value: 'surface-samples',
      source: 'declared'
    });
    const grouped = await run('table-summarize', renamed, {
      groupBy: 'observed',
      valueColumn: 'observed'
    });
    expect(grouped.analysisProfile?.rowSemantics?.source).not.toBe('declared');
    expect(
      grouped.analysisProfile?.columns.find(column => column.name === 'observed')?.measurementSource
    ).toBe('declared');
    expect(
      grouped.analysisProfile?.columns.find(column => column.name === 'value_sum')
        ?.measurementSource
    ).not.toBe('declared');
  });

  test('joins preserve target row meaning and concatenation requires agreeing declarations', async () => {
    const host = makeHost();
    const first = await host.readDataset(
      {
        datasetId: profile.datasetId,
        geometry: {mode: 'geojson-field', field: 'geometry'},
        population: 'all',
        maximumRows: 12
      },
      new AbortController().signal
    );
    first.analysisProfile = applyKeplerDatasetProfileDeclarations(profile, {
      rowSemantics: 'surface-samples',
      measurement: {measurement: 'amount'}
    });
    const context: WorkflowRunContext = {
      device: null,
      engine: 'cpu',
      host,
      signal: new AbortController().signal,
      workflowId: 'declarations',
      stepId: 'table'
    };
    const attributes = makeWorkflowData({
      id: 'attributes',
      label: 'Rates',
      rowCount: 12,
      columns: {key: Array.from({length: 12}, (_, index) => index + 1), rate: Array(12).fill(0.5)},
      columnTypes: {key: 'int32', rate: 'float64'}
    });
    attributes.analysisProfile = applyKeplerDatasetProfileDeclarations(
      profileKeplerWorkbenchData(attributes),
      {
        rowSemantics: 'objects',
        measurement: {rate: 'rate'}
      }
    );
    const join = registry.get('table-join-attributes')!;
    const joined = (
      await join.run(
        context,
        {target: first, attributes},
        join.normalizeParameters({
          targetKey: 'measurement',
          attributesKey: 'key',
          columns: 'rate',
          prefix: 'joined_'
        })
      )
    ).outputs['joined'].analysisProfile!;
    expect(joined.rowSemantics).toMatchObject({value: 'surface-samples', source: 'declared'});
    expect(joined.columns.find(column => column.name === 'measurement')).toMatchObject({
      measurement: 'amount',
      measurementSource: 'declared'
    });
    expect(joined.columns.find(column => column.name === 'joined_rate')).toMatchObject({
      measurement: 'rate',
      measurementSource: 'declared'
    });
    const concatenate = registry.get('table-concatenate')!;
    const agreed = (await concatenate.run(context, {first, second: first}, {})).outputs['combined']
      .analysisProfile!;
    expect(agreed.rowCount).toBe(24);
    expect(agreed.rowSemantics).toMatchObject({value: 'surface-samples', source: 'declared'});
    expect(agreed.columns.find(column => column.name === 'measurement')?.measurementSource).toBe(
      'declared'
    );
    const second = {
      ...first,
      analysisProfile: applyKeplerDatasetProfileDeclarations(first.analysisProfile, {
        rowSemantics: 'objects',
        measurement: {measurement: 'rate'}
      })
    };
    const conflicting = (await concatenate.run(context, {first, second}, {})).outputs['combined']
      .analysisProfile!;
    expect(conflicting.rowSemantics?.source).not.toBe('declared');
    expect(
      conflicting.columns.find(column => column.name === 'measurement')?.measurementSource
    ).not.toBe('declared');
    expect(conflicting.coordinates).toMatchObject({model: 'geographic', source: 'declared'});
  });

  test('explicit geometry-free workflow tables suppress inferred points at typed dispatch', async () => {
    const table = {
      id: 'flows',
      label: 'Flows',
      rowCount: 2,
      columns: {
        longitude: [-87.6, -87.5],
        latitude: [41.8, 41.9],
        origin: ['a', 'b'],
        destination: ['b', 'a'],
        weight: [12, 7]
      },
      columnTypes: {
        longitude: 'float64' as const,
        latitude: 'float64' as const,
        origin: 'string' as const,
        destination: 'string' as const,
        weight: 'float64' as const
      }
    };
    const data = makeWorkflowData(table);
    expect(data.kind).toBe('table');
    const inferred = profileKeplerColumnTable(
      {
        rowCount: table.rowCount,
        columns: Object.entries(table.columns).map(([name, values]) => ({name, values}))
      },
      {datasetId: table.id, label: table.label}
    );
    expect(inferred.kind).toBe('points');
    const typedProfile = profileKeplerWorkbenchData(data);
    expect(typedProfile.kind).toBe('attributes');
    for (const key of [
      'catalogGeometry',
      'pointColumns',
      'extent',
      'spacing',
      'coordinates',
      'vertexCount'
    ])
      expect(typedProfile).not.toHaveProperty(key);
    const base = getCatalogBinding('kernel-density-grid')!;
    const binding = {
      ...base,
      id: 'private-typed-flow',
      run: undefined,
      runGpu: undefined,
      parameters: [],
      inputs: [
        {
          name: 'flows',
          label: 'Flows',
          dataKind: 'table' as const,
          accepts: [],
          maximumFeatures: 2,
          maximumVertices: 0
        }
      ],
      outputs: [
        {
          tableId: 'flows',
          label: 'Flows',
          geometry: 'none' as const,
          identityColumns: [],
          maximumRows: 2
        }
      ],
      runData: () => ({tables: [table], diagnostics: [], summary: {rows: 2}})
    };
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    const result = await operation.run(
      {
        device: null,
        engine: 'cpu',
        host: makeFixtureHost(),
        signal: new AbortController().signal,
        workflowId: 'flows',
        stepId: 'flow'
      },
      {flows: data},
      {}
    );
    expect(result.outputs['flows'].table.columns).toEqual(table.columns);
  });

  test('private adapted bindings use their actual definition in planning, form and execution', async () => {
    const base = getCatalogBinding('kernel-density-grid')!;
    const binding = {
      ...base,
      id: 'private-density',
      title: 'Private density',
      parameters: base.parameters.map(parameter =>
        parameter.name === 'cellSize' ? {...parameter, defaultValue: 91} : parameter
      )
    };
    expect(getCatalogBinding(binding.id)).toBeUndefined();
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    const privateRegistry = createWorkflowOperationRegistry([...registry.list(), operation]);
    const source = addWorkflowStep(
      createWorkflowDocument({title: 'Private operation'}),
      privateRegistry,
      {
        operationId: 'source-geojson',
        parameters: {text: JSON.stringify(collection), coordinateSystem: 'EPSG:4326'}
      }
    );
    const added = addWorkflowStep(source.document, privateRegistry, {
      operationId: operation.id,
      connectFrom: {stepId: source.stepId, output: 'data'},
      connectTo: 'points'
    });
    const host = makeFixtureHost();
    const planned = planWorkflow(added.document, {registry: privateRegistry, host}).steps[
      added.stepId
    ];
    expect(planned.status, JSON.stringify(planned.issues)).toBe('ready');
    expect(planned.parameters?.['cellSize']).toBe(91);
    const step = added.document.steps.find(candidate => candidate.id === added.stepId)!;
    const rows = getWorkbenchParameterRows(
      added.document,
      step,
      operation,
      getWorkbenchAnalysisOptions(added.document, step, operation, privateRegistry, host)
    );
    expect(rows.find(row => row.spec.name === 'cellSize')?.value).toBe(91);
    const runner = new WorkflowRunner({registry: privateRegistry, host});
    runner.setDocument(added.document);
    await runner.run();
    const state = runner.getState().steps[added.stepId];
    expect(state.status, JSON.stringify(state.issues)).toBe('done');
    expect(state.result?.parameters['cellSize']).toBe(91);
    await runner.destroy();
  });

  test('selecting a matching curated preset invalidates cached provenance', async () => {
    const initial = makeDensityDocument();
    const step = initial.document.steps.find(candidate => candidate.id === initial.densityId)!;
    step.parameters['kernel'] = {from: 'static', value: 'gaussian'};
    step.parameters['includeEmptyCells'] = {from: 'static', value: false};
    const operation = registry.get(step.operationId)!;
    const preset = operation.analysisPresets!.find(
      candidate => candidate.curatedAnalysisId === 'density-surface'
    )!;
    expect(preset).toBeDefined();
    const runner = new WorkflowRunner({registry, host: makeHost()});
    runner.setDocument(initial.document);
    await runner.run();
    const before = runner.getState().steps[step.id];
    expect(before.status, JSON.stringify(before.issues)).toBe('done');
    expect(before.result?.curatedAnalysisId).toBeUndefined();
    const applied = applyWorkbenchCuratedPreset(
      initial.document,
      step,
      operation,
      preset.curatedAnalysisId
    );
    runner.setDocument(applied);
    const changed = runner.getState().steps[step.id];
    expect(planWorkflow(applied, {registry, host: makeHost()}).steps[step.id].parameters).toEqual(
      planWorkflow(initial.document, {registry, host: makeHost()}).steps[step.id].parameters
    );
    expect(changed.key).not.toBe(before.key);
    expect(changed.status).toBe('stale');
    await runner.run();
    const result = runner.getState().steps[step.id].result!;
    expect(result.curatedAnalysisId).toBe(preset.curatedAnalysisId);
    expect(result.outputs['density-cells'].analysisReport?.notes).toContain(
      `Curated analysis: ${preset.title}`
    );
    await runner.destroy();
  });

  test('number lists survive catalog → adapter → operation → form', () => {
    const operation = registry.get('network-service-area')!;
    const defaults = getKeplerWorkbenchParameterDefaults(operation.id);
    expect(defaults.values['budgets']).toEqual([500, 1000]);
    expect(operation.parameters.find(parameter => parameter.name === 'budgets')?.kind).toBe(
      'number-list'
    );
    expect(defaults.resolutions.find(entry => entry.parameter === 'budgets')).toBeDefined();
  });

  test('roles, curated identity and presets are actual workflow metadata and serialize', () => {
    const operation = registry.get('local-moran-statistics')!;
    expect(
      operation.parameters.find(parameter => parameter.name === 'valueProperty')?.fieldRole?.role
    ).toBe('value');
    expect(
      operation.parameters.find(parameter => parameter.name === 'valueProperty')?.columnsFrom
    ).toBeDefined();
    const preset = operation.analysisPresets![0];
    const added = addWorkflowStep(createWorkflowDocument({title: 'Curated settings'}), registry, {
      operationId: operation.id,
      parameters: {valueProperty: 'measurement'}
    });
    const step = added.document.steps.find(candidate => candidate.id === added.stepId)!;
    const applied = applyWorkbenchCuratedPreset(
      added.document,
      step,
      operation,
      preset.curatedAnalysisId
    );
    const saved = parseWorkflowDocument(serializeWorkflowDocument(applied)).steps.find(
      candidate => candidate.id === added.stepId
    )!;
    expect(saved.curatedAnalysisId).toBe(preset.curatedAnalysisId);
    expect(saved.parameters['valueProperty']).toEqual({from: 'static', value: 'measurement'});
    for (const [name, value] of Object.entries(preset.executionParameters)) {
      if (!operation.parameters.find(parameter => parameter.name === name)?.fieldRole)
        expect(saved.parameters[name]).toEqual({from: 'static', value});
    }
  });

  test('numeric ties, missing values and Float32 boundary values match the exact legend', () => {
    const operation = registry.get('kernel-density-grid')!;
    const table = {
      id: 'density-cells',
      label: 'Density',
      rowCount: 8,
      columns: {density: [0, 1, 1, 1.000000001, 2, 3, null, 9]},
      columnTypes: {density: 'float64' as const}
    };
    const presentation = getKeplerBindingResultPresentations(
      {
        operationId: operation.id,
        engine: 'cpu-reference',
        parameters: {},
        tables: [table],
        diagnostics: [],
        summary: {},
        dispatchedContributors: []
      },
      operation.analysisDescriptor!
    )[0];
    const encoded = encodeKeplerResultPresentation(table, presentation)!;
    expect(encoded.values[1]).toBe(encoded.values[2]);
    for (let index = 0; index < encoded.legend.length; index++) {
      expect(encoded.values.filter(value => value === encoded.categories[index]).length).toBe(
        encoded.legend[index].count
      );
    }
  });
});

test('GeoJSON planning uses canonical property names and declarations, matching execution', async () => {
  const features = {
    ...collection,
    features: collection.features.map(feature => ({
      ...feature,
      properties: {feature_id: feature.properties.measurement, feature_id_property: 20, geometry: 3}
    }))
  };
  const source = addWorkflowStep(
    createWorkflowDocument({title: 'Reserved property names'}),
    registry,
    {
      id: 'measurements',
      operationId: 'source-geojson',
      parameters: {
        text: JSON.stringify(features),
        coordinateSystem: 'planar',
        profileDeclarations: JSON.stringify({
          version: 1,
          rowSemantics: 'surface-samples',
          measurement: {feature_id: 'amount'}
        })
      }
    }
  );
  const interpolation = addWorkflowStep(source.document, registry, {
    id: 'surface',
    operationId: 'idw-grid',
    parameters: {valueProperty: 'feature_id_property_property', cellSize: 1},
    connectFrom: {stepId: source.stepId, output: 'data'},
    connectTo: 'points'
  });
  const host = makeFixtureHost();
  const plan = planWorkflow(interpolation.document, {registry, host});
  expect(
    plan.steps['surface'].analysis?.status,
    JSON.stringify(
      Object.fromEntries(Object.entries(plan.steps).map(([id, step]) => [id, step.issues]))
    )
  ).toBe('ready');
  expect(plan.steps['surface'].parameters?.['valueProperty']).toBe('feature_id_property_property');
  const wrong = structuredClone(interpolation.document);
  wrong.steps.find(step => step.id === 'surface')!.parameters['valueProperty'] = {
    from: 'static',
    value: 'feature_id'
  };
  expect(planWorkflow(wrong, {registry, host}).steps['surface'].analysis?.status).toBe('blocked');
  const runner = new WorkflowRunner({registry, host});
  runner.setDocument(interpolation.document);
  try {
    await runner.run();
    const state = runner.getState().steps;
    expect(state['surface'].status, JSON.stringify(state['surface'].issues)).toBe('done');
    expect(state['surface'].result?.parameters).toEqual(plan.steps['surface'].parameters);
    const sourceProfile = state['measurements'].result?.outputs['data'].analysisProfile;
    expect(sourceProfile?.datasetId).toBe('measurements');
    expect(
      sourceProfile?.columns.find(column => column.name === 'feature_id_property_property')
        ?.measurement
    ).toBe('amount');
    expect(
      sourceProfile?.columns.find(column => column.name === 'feature_id_property_property')
        ?.measurementSource
    ).toBe('declared');
    expect(sourceProfile?.columns.find(column => column.name === 'feature_id')?.type).toBe(
      'category'
    );
    expect(
      sourceProfile?.columns.find(column => column.name === 'geometry_property')
    ).toBeDefined();
  } finally {
    await runner.destroy();
  }
});

test('independent filtered outputs retain distinct dataset identities despite identical table ids', async () => {
  let document = createWorkflowDocument({title: 'Compare filtered populations'});
  for (const [index, period] of ['before', 'after'].entries()) {
    const source = addWorkflowStep(document, registry, {
      id: period,
      operationId: 'source-geojson',
      parameters: {
        text: JSON.stringify({
          ...collection,
          features: collection.features.map(feature => ({
            ...feature,
            properties: {measurement: feature.properties.measurement + index}
          }))
        }),
        coordinateSystem: 'planar',
        profileDeclarations: JSON.stringify({version: 1, rowSemantics: 'events'})
      }
    });
    document = addWorkflowStep(source.document, registry, {
      id: `${period}-filtered`,
      operationId: 'table-filter-rows',
      parameters: {expression: 'true'},
      connectFrom: {stepId: period, output: 'data'},
      connectTo: 'rows'
    }).document;
  }
  const comparison = addWorkflowStep(document, registry, {
    id: 'comparison',
    operationId: 'cell-compare',
    parameters: {cellSize: 1}
  });
  document = comparison.document;
  document.steps.find(step => step.id === 'comparison')!.inputs = {
    before: {stepId: 'before-filtered', output: 'filtered'},
    after: {stepId: 'after-filtered', output: 'filtered'}
  };
  const runner = new WorkflowRunner({registry, host: makeFixtureHost()});
  try {
    runner.setDocument(document);
    await runner.run();
    const steps = runner.getState().steps;
    expect(steps['comparison'].status, JSON.stringify(steps['comparison'].issues)).toBe('done');
    const before = steps['before-filtered'].result!.outputs['filtered'];
    const after = steps['after-filtered'].result!.outputs['filtered'];
    expect(before.table.id).toBe(after.table.id);
    expect(before.analysisProfile?.sourceDatasetId).not.toBe(
      after.analysisProfile?.sourceDatasetId
    );
    expect(
      evaluateKeplerWorkbenchInputs(
        'cell-compare',
        {
          before: before.analysisProfile,
          after: before.analysisProfile
        },
        {overrides: {cellSize: 1}}
      ).status
    ).toBe('blocked');
  } finally {
    await runner.destroy();
  }
});
