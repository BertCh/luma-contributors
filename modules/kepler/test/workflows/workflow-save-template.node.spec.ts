// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {describe, expect, test} from 'vitest';
import {
  addWorkflowStep,
  connectWorkflowSteps,
  createWorkflowDocument,
  exportWorkflowAnalysisTemplateJSON,
  importWorkflowAnalysisTemplateJSON,
  parseWorkflowAnalysisTemplate,
  getWorkflowOperationRegistry,
  getWorkflowSaveTemplateSlots,
  getKeplerOperationDescriptor,
  mapWorkflowAnalysisTemplate,
  saveWorkflowAnalysisTemplate,
  profileKeplerWorkbenchData,
  parseWorkflowCollection,
  serializeWorkflowCollection,
  promoteWorkflowParameter,
  validateWorkflowAnalysisTemplate,
  WorkflowRunner,
  type WorkflowDocument,
  type WorkflowData,
  type KeplerDatasetProfile,
  type WorkflowDatasetRequest,
  type WorkflowIssue
} from '@luma-contributors/kepler';
import {makeFixtureHost} from './workflow-fixtures';
import {createWorkbenchStore} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {
  WorkbenchSaveTemplate,
  WorkbenchSavedTemplate
} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-save-template';
import {WorkbenchDialogs} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-dialogs';
import {
  getWorkbenchAssistantTools,
  runWorkbenchAssistantTool
} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-assistant';

const registry = getWorkflowOperationRegistry();
const hostRequire = createRequire(
  new URL('../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');
function makeDocument(): WorkflowDocument {
  let document = createWorkflowDocument({id: 'ranking', title: 'Which observations rank highest?'});
  const steps: Parameters<typeof addWorkflowStep>[2][] = [
    {
      id: 'source',
      operationId: 'source-dataset',
      parameters: {dataset: 'original-data', geometryMode: 'none'}
    },
    {
      id: 'sort',
      operationId: 'table-sort-rows',
      parameters: {column: 'old_value', order: 'descending', limit: 2}
    },
    {id: 'drop', operationId: 'table-drop-geometry'}
  ];
  for (const step of steps) document = addWorkflowStep(document, registry, step).document;
  document = connectWorkflowSteps(document, registry, {
    from: {stepId: 'source', output: 'data'},
    to: {stepId: 'sort', input: 'rows'}
  });
  return connectWorkflowSteps(document, registry, {
    from: {stepId: 'sort', output: 'sorted'},
    to: {stepId: 'drop', input: 'rows'}
  });
}
function makeData(datasetId: string, column: string, values: number[]): WorkflowData {
  return {
    kind: 'table',
    table: {
      id: datasetId,
      label: datasetId,
      rowCount: values.length,
      columns: {name: values.map((_, index) => `row-${index}`), [column]: values},
      columnTypes: {name: 'string', [column]: 'float64'}
    }
  };
}
function makeProfile(datasetId: string, data: WorkflowData): KeplerDatasetProfile {
  return {...profileKeplerWorkbenchData(data), datasetId, label: datasetId};
}
function makeHost(datasets: Record<string, WorkflowData>) {
  const host = makeFixtureHost(
    Object.entries(datasets).map(([id, data]) => ({
      id,
      label: id,
      resident: false,
      rowCount: data.table.rowCount,
      geometryFields: [],
      fields: Object.entries(data.table.columnTypes).map(([name, type]) => ({
        name,
        type: type === 'string' ? 'string' : 'real'
      }))
    })),
    datasets
  );
  host.getAnalysisProfile = datasetId => makeProfile(datasetId, datasets[datasetId]);
  return host;
}
function save(document = makeDocument()) {
  return saveWorkflowAnalysisTemplate(document, {
    id: 'portable-ranking',
    title: document.title,
    question: 'prepare-data',
    method: 'Rank observations and retain the first rows.',
    slots: getWorkflowSaveTemplateSlots(document),
    exposed: [{stepId: 'sort', parameter: 'limit'}]
  });
}
function roundTrip(document = makeDocument()) {
  return importWorkflowAnalysisTemplateJSON(exportWorkflowAnalysisTemplateJSON(save(document)));
}
function map(
  template: ReturnType<typeof save>,
  datasetId: string,
  data: WorkflowData,
  column: string
) {
  return mapWorkflowAnalysisTemplate(template, {
    datasets: {source: makeProfile(datasetId, data)},
    columns: Object.fromEntries(
      template.slots[0].fields.map(field => [
        `source.${field.role}`,
        field.sources.some(source => source.parameter === 'groupBy') ? [] : [column]
      ])
    )
  });
}

describe('AX-51 saved templates on the integration tree', () => {
  test('exports chosen roles and promoted settings through AX-30 without embedding dataset identities', () => {
    const document = makeDocument();
    const before = structuredClone(document);
    const template = save(document);
    const text = exportWorkflowAnalysisTemplateJSON(template);
    expect(parseWorkflowAnalysisTemplate(text)).toEqual(template);
    expect(validateWorkflowAnalysisTemplate(template).issues).toEqual([]);
    expect(template.exposed).toEqual([
      {
        inputId: 'limit',
        target: {stepId: 'sort', parameter: 'limit'},
        defaultPolicy: {from: 'constant'}
      }
    ]);
    expect(template.document.inputs[0].value).toBe(2);
    expect(template.slots[0].fields[0].sources).toEqual([{stepId: 'sort', parameter: 'column'}]);
    expect(text).not.toContain('original-data');
    expect(text).not.toContain('old_value');
    expect(template.sample).toBeUndefined();
    expect(document).toEqual(before);
  });
  test('actually reruns the saved multi-step stack with renamed columns on a second dataset', async () => {
    const first = makeData('first', 'score', [3, 9, 5]);
    const second = makeData('second', 'renamed_score', [20, 10, 40]);
    const runner = new WorkflowRunner({host: makeHost({first, second}), registry});
    const template = roundTrip();
    try {
      for (const [datasetId, data, column, expected] of [
        ['first', first, 'score', [9, 5]],
        ['second', second, 'renamed_score', [40, 20]]
      ] as const) {
        runner.setDocument(map(template, datasetId, data, column));
        const record = await runner.run();
        expect(record.status, JSON.stringify(record)).toBe('done');
        expect(
          runner.getState().steps['drop'].result!.outputs['table'].table.columns[column]
        ).toEqual(expected);
      }
    } finally {
      await runner.destroy();
    }
  });
  test('preserves a 100000-row source budget while executing 20001 rows', async () => {
    const values = Array.from({length: 20001}, (_, index) => index);
    const original = makeData('original-data', 'old_value', values);
    const next = makeData('next', 'new_value', values);
    const document = makeDocument();
    document.steps[0].parameters['maximumRows'] = {from: 'static', value: 100000};
    const runner = new WorkflowRunner({
      host: makeHost({'original-data': original, next}),
      registry
    });
    try {
      runner.setDocument(document);
      expect((await runner.run()).status).toBe('done');
      const template = roundTrip(document);
      expect(template.document.steps[0].parameters['maximumRows']).toEqual({
        from: 'static',
        value: 100000
      });
      const mapped = map(template, 'next', next, 'new_value');
      expect(mapped.steps[0].parameters['maximumRows']).toEqual({from: 'static', value: 100000});
      runner.setDocument(mapped);
      const record = await runner.run();
      expect(record.status, JSON.stringify(record)).toBe('done');
      expect(runner.getState().steps['source'].result!.outputs['data'].table.rowCount).toBe(20001);
      expect(
        runner.getState().steps['drop'].result!.outputs['table'].table.columns['new_value']
      ).toEqual([20000, 19999]);
    } finally {
      await runner.destroy();
    }
  });
  test('preserves selected population in the host request and computed result', async () => {
    const original = makeData('original-data', 'old_value', [3, 9, 5]);
    const next = makeData('next', 'new_value', [20, 10, 40]);
    const datasets: Record<string, WorkflowData> = {'original-data': original, next};
    const host = makeHost(datasets);
    const requests: WorkflowDatasetRequest[] = [];
    host.readDataset = async request => {
      requests.push(request);
      const data = datasets[request.datasetId];
      return request.population === 'selected'
        ? {
            ...data,
            table: {
              ...data.table,
              rowCount: 2,
              columns: Object.fromEntries(
                Object.entries(data.table.columns).map(([name, values]) => [
                  name,
                  [values[0], values[2]]
                ])
              )
            }
          }
        : data;
    };
    const document = makeDocument();
    document.steps[0].parameters['population'] = {from: 'static', value: 'selected'};
    const runner = new WorkflowRunner({host, registry});
    try {
      runner.setDocument(document);
      expect((await runner.run()).status).toBe('done');
      expect(
        runner.getState().steps['drop'].result!.outputs['table'].table.columns['old_value']
      ).toEqual([5, 3]);
      const template = roundTrip(document);
      expect(template.document.steps[0].parameters['population']).toEqual({
        from: 'static',
        value: 'selected'
      });
      runner.setDocument(map(template, 'next', next, 'new_value'));
      expect((await runner.run()).status).toBe('done');
      expect(requests.map(request => request.population)).toEqual(['selected', 'selected']);
      expect(
        runner.getState().steps['drop'].result!.outputs['table'].table.columns['new_value']
      ).toEqual([40, 20]);
    } finally {
      await runner.destroy();
    }
  });
  test.each([
    'old_value > 0 && contains("old_value", "value")',
    '`old_value` > 0'
  ])('blocks unsafe filter remapping before Run: %s', async expression => {
    let document = addWorkflowStep(makeDocument(), registry, {
      id: 'filter',
      operationId: 'table-filter-rows',
      parameters: {expression}
    }).document;
    document = connectWorkflowSteps(document, registry, {
      from: {stepId: 'source', output: 'data'},
      to: {stepId: 'filter', input: 'rows'}
    });
    document = connectWorkflowSteps(document, registry, {
      from: {stepId: 'filter', output: 'filtered'},
      to: {stepId: 'sort', input: 'rows'}
    });
    const template = roundTrip(document);
    const renamed = makeData('next', 'new_value', [-1, 9, 5]);
    expect(() => map(template, 'next', renamed, 'new_value')).toThrow(
      /filter.expression.*old_value.*expression-column remapping is unsupported/
    );
    const compatible = makeData('compatible', 'old_value', [-1, 9, 5]);
    const runner = new WorkflowRunner({host: makeHost({compatible}), registry});
    try {
      runner.setDocument(map(template, 'compatible', compatible, 'old_value'));
      expect((await runner.run()).status).toBe('done');
      expect(
        runner.getState().steps['drop'].result!.outputs['table'].table.columns['old_value']
      ).toEqual([9, 5]);
    } finally {
      await runner.destroy();
    }
  });
  test('allows literal-only expressions and blocks unknown expression provenance', () => {
    let document = addWorkflowStep(makeDocument(), registry, {
      id: 'filter',
      operationId: 'table-filter-rows',
      parameters: {expression: 'contains("old_value", "value")'}
    }).document;
    document = connectWorkflowSteps(document, registry, {
      from: {stepId: 'source', output: 'data'},
      to: {stepId: 'filter', input: 'rows'}
    });
    const next = makeData('next', 'new_value', [1, 9, 5]);
    expect(() => map(roundTrip(document), 'next', next, 'new_value')).not.toThrow();
    document = addWorkflowStep(document, registry, {
      id: 'calculate',
      operationId: 'table-calculate-column',
      parameters: {columnName: 'derived', expression: 'old_value * 2'}
    }).document;
    document = connectWorkflowSteps(document, registry, {
      from: {stepId: 'source', output: 'data'},
      to: {stepId: 'calculate', input: 'rows'}
    });
    document = connectWorkflowSteps(document, registry, {
      from: {stepId: 'calculate', output: 'calculated'},
      to: {stepId: 'filter', input: 'rows'}
    });
    document.steps.find(step => step.id === 'filter')!.parameters['expression'] = {
      from: 'static',
      value: 'derived > 0'
    };
    const compatible = makeData('next', 'old_value', [1, 9, 5]);
    expect(() => map(roundTrip(document), 'next', compatible, 'old_value')).toThrow(
      /column-changing steps.*expression-column remapping is unsupported/
    );
  });
  test.each([
    false,
    true
  ])('preserves independent count-only and required regionalization roles (reverse=%s)', reverse => {
    let document = createWorkflowDocument({id: 'counts-and-regions', title: 'Counts and regions'});
    const steps: Parameters<typeof addWorkflowStep>[2][] = [
      {
        id: 'source',
        operationId: 'source-dataset',
        parameters: {dataset: 'original-data', geometryField: 'geometry'}
      },
      {id: 'counts', operationId: 'cell-aggregation', parameters: {valueProperty: '', cellSize: 1}},
      {
        id: 'regions',
        operationId: 'redcap-single-linkage',
        parameters: {valueProperty: 'old_value', radius: 10, coordinateSystem: 'planar'}
      }
    ];
    for (const step of reverse ? [steps[0], steps[2], steps[1]] : steps)
      document = addWorkflowStep(document, registry, step).document;
    for (const stepId of ['counts', 'regions'])
      document = connectWorkflowSteps(document, registry, {
        from: {stepId: 'source', output: 'data'},
        to: {stepId, input: 'points'}
      });
    const slots = getWorkflowSaveTemplateSlots(document);
    const countRole = slots[0].fields.find(field =>
      field.sources.some(source => source.stepId === 'counts')
    )!;
    const regionRole = slots[0].fields.find(field =>
      field.sources.some(source => source.stepId === 'regions')
    )!;
    expect(countRole.required).toBe(false);
    expect(regionRole.required).toBe(true);
    for (const [role, operationId] of [
      [countRole, 'cell-aggregation'],
      [regionRole, 'redcap-single-linkage']
    ] as const) {
      const descriptor = getKeplerOperationDescriptor(operationId)!.inputs[0].fields[0];
      expect(role.accepts).toEqual(descriptor.accepts);
      expect(role.measurement).toEqual(descriptor.measurement);
    }
    const template = saveWorkflowAnalysisTemplate(document, {
      id: document.id,
      title: document.title,
      method: 'Count and regionalize independently.',
      question: 'prepare-data',
      slots,
      exposed: []
    });
    const profile: KeplerDatasetProfile = {
      datasetId: 'next',
      label: 'Next',
      kind: 'points',
      rowCount: 4,
      geometryColumns: ['geometry'],
      columns: [{name: 'new_value', type: 'number', rowCount: 4, nullCount: 0}]
    };
    const mapped = mapWorkflowAnalysisTemplate(template, {
      datasets: {source: profile},
      columns: {[`source.${regionRole.role}`]: ['new_value']}
    });
    expect(mapped.steps.find(step => step.id === 'counts')!.parameters['valueProperty']).toEqual({
      from: 'static',
      value: ''
    });
    expect(mapped.steps.find(step => step.id === 'regions')!.parameters['valueProperty']).toEqual({
      from: 'static',
      value: 'new_value'
    });
  });
  test('saves only selected slots and parameters, preserving fixed sources and values', () => {
    let document = promoteWorkflowParameter(makeDocument(), registry, 'sort', 'order').document;
    document = addWorkflowStep(document, registry, {
      id: 'reference',
      operationId: 'source-dataset',
      parameters: {dataset: 'reference-data', geometryMode: 'none'}
    }).document;
    const template = saveWorkflowAnalysisTemplate(document, {
      id: 'chosen',
      title: document.title,
      method: 'Keep reference fixed.',
      question: 'prepare-data',
      slots: [{...getWorkflowSaveTemplateSlots(document)[0], fields: []}],
      exposed: [{stepId: 'sort', parameter: 'limit'}]
    });
    expect(template.fixedSources).toEqual(['reference']);
    expect(template.document.inputs.map(input => input.id)).toEqual(['limit']);
    const sort = template.document.steps.find(step => step.id === 'sort')!;
    expect(sort.parameters['order']).toEqual({from: 'static', value: 'descending'});
    expect(sort.parameters['column']).toEqual({from: 'static', value: 'old_value'});
    expect(validateWorkflowAnalysisTemplate(template).issues).toEqual([]);
    expect(
      validateWorkflowAnalysisTemplate({...template, fixedSources: ['sort']}).issues.map(
        issue => issue.code
      )
    ).toContain('invalid-fixed-source');
  });
  test('maps downstream columns through sort while preserving optional group-by fields', async () => {
    let document = addWorkflowStep(makeDocument(), registry, {
      id: 'summary',
      operationId: 'table-summarize',
      parameters: {groupBy: '', valueColumn: 'old_value'}
    }).document;
    document = connectWorkflowSteps(document, registry, {
      from: {stepId: 'sort', output: 'sorted'},
      to: {stepId: 'summary', input: 'rows'}
    });
    const data = makeData('next', 'new_value', [4, 8, 1]);
    const runner = new WorkflowRunner({host: makeHost({next: data}), registry});
    try {
      runner.setDocument(map(roundTrip(document), 'next', data, 'new_value'));
      expect((await runner.run()).status).toBe('done');
      expect(
        runner.getState().steps['summary'].result!.outputs['summary'].table.columns['value_sum']
      ).toEqual([12]);
    } finally {
      await runner.destroy();
    }
  });
  test('rejects lost, wrong-kind, wrong-type, overlong and invalid exposed input choices', () => {
    const template = save();
    const data = makeData('next', 'new_value', [1, 3, 2]);
    const profile = makeProfile('next', data);
    const key = `source.${template.slots[0].fields[0].role}`;
    expect(() => mapWorkflowAnalysisTemplate(template, {datasets: {}, columns: {}})).toThrow(
      /needs a dataset/
    );
    expect(() =>
      mapWorkflowAnalysisTemplate(template, {
        datasets: {source: {...profile, kind: 'points'}},
        columns: {}
      })
    ).toThrow(/needs attributes/);
    expect(() =>
      mapWorkflowAnalysisTemplate(template, {datasets: {source: profile}, columns: {}})
    ).toThrow(/distinct columns/);
    expect(() =>
      mapWorkflowAnalysisTemplate(template, {
        datasets: {source: profile},
        columns: {[key]: ['missing']}
      })
    ).toThrow(/compatible columns/);
    const numeric = {
      ...template,
      slots: [
        {
          ...template.slots[0],
          fields: [{...template.slots[0].fields[0], accepts: ['number' as const]}]
        }
      ]
    };
    expect(() =>
      mapWorkflowAnalysisTemplate(numeric, {
        datasets: {source: profile},
        columns: {[key]: ['name']}
      })
    ).toThrow(/compatible columns/);
    const longName = 'x'.repeat(65);
    const longData = makeData('long', longName, [1, 3, 2]);
    expect(() => map(template, 'long', longData, longName)).toThrow(/64/);
    expect(() =>
      mapWorkflowAnalysisTemplate(template, {
        datasets: {source: profile},
        columns: {[key]: ['new_value']},
        parameters: {limit: -1}
      })
    ).toThrow(/at least/);
    expect(() =>
      mapWorkflowAnalysisTemplate(template, {
        datasets: {source: profile},
        columns: {[key]: ['new_value']},
        parameters: {absent: 1}
      })
    ).toThrow(/exposed input/);
  });
  test('persists AX metadata with legacy templates and clears the last deleted card', async () => {
    const data = makeData('next', 'new_value', [1, 3, 2]);
    const host = makeHost({next: data});
    const store = createWorkbenchStore({host, registry});
    const restored = createWorkbenchStore({host, registry});
    try {
      store.addWorkflow(makeDocument());
      store.saveTemplate();
      store.saveAnalysisTemplate!(save());
      expect(store.saveAnalysisTemplate!(save())).toBe('portable-ranking-2');
      restored.restore(parseWorkflowCollection(JSON.parse(JSON.stringify(store.serialize()))));
      expect(restored.getState().templates).toHaveLength(1);
      expect(restored.getState().analysisTemplates).toEqual(store.getState().analysisTemplates);
      restored.addWorkflow(
        map(restored.getState().analysisTemplates![0], 'next', data, 'new_value')
      );
      expect((await restored.run())?.status).toBe('done');
      restored.deleteAnalysisTemplate!('portable-ranking');
      restored.deleteAnalysisTemplate!('portable-ranking-2');
      expect(restored.getState().analysisTemplates ?? []).toEqual([]);
      const collection = {schemaVersion: 1 as const, workflows: [], analysisTemplates: [save()]};
      expect(parseWorkflowCollection(serializeWorkflowCollection(collection))).toEqual(collection);
      expect(() =>
        parseWorkflowCollection({...collection, analysisTemplates: [save(), save()]})
      ).toThrow(/duplicate/);
    } finally {
      await store.clear();
      await restored.clear();
    }
  });
  test('saving templates retains landed assistant review identities and nine advertised tools', async () => {
    const data = makeData('original-data', 'old_value', [3, 9, 5]);
    const store = createWorkbenchStore({host: makeHost({'original-data': data}), registry});
    try {
      const workflowId = store.addWorkflow(makeDocument());
      const document = store.getState().document!;
      const review: WorkflowIssue[] = [
        {severity: 'warning', code: 'capacity-may-exceed', message: 'Reviewed advisory'}
      ];
      store.recordAssistantReview(document, review);
      store.saveAnalysisTemplate!(save(document));
      expect(store.getState().document).toBe(document);
      expect(store.getReviewedAssistantIssues(document)).toEqual(review);
      expect(getWorkbenchAssistantTools().map(tool => tool.name)).toEqual([
        'listAnalysisOperations',
        'describeAnalysisOperation',
        'defineWorkflow',
        'listWorkflows',
        'describeWorkflow',
        'setWorkflowInputs',
        'runWorkflow',
        'listAnalysisTemplates',
        'describeAnalysisTemplate'
      ]);
      const preview = runWorkbenchAssistantTool(store, {
        name: 'runWorkflow',
        arguments: {workflowId}
      });
      expect(preview).toMatchObject({
        status: 'needs-approval',
        cachedResultsKnown: true,
        steps: 3,
        issues: expect.any(Array),
        message: expect.stringContaining('explicit user approval')
      });
      expect(store.getState().runLog).toEqual([]);
    } finally {
      await store.clear();
    }
  });
  test('renders save choices, remapping and export without a false empty-state message', () => {
    const store = createWorkbenchStore({host: makeFixtureHost(), registry});
    const form = renderToStaticMarkup(
      createElement(WorkbenchSaveTemplate, {document: makeDocument(), store, onClose() {}})
    );
    expect(form).toContain('as a dataset slot');
    expect(form).toContain('sort.column');
    expect(form).toContain('Exposed settings');
    const card = renderToStaticMarkup(
      createElement(WorkbenchSavedTemplate, {template: save(), store})
    );
    expect(card).toContain('Map my data');
    expect(card).toContain('Export template JSON');
    store.saveAnalysisTemplate!(save());
    store.setDialog('templates');
    const gallery = renderToStaticMarkup(createElement(WorkbenchDialogs, {store}));
    expect(gallery).toContain('ax-saved-template-portable-ranking');
    expect(gallery).not.toContain('No saved templates.');
  });
});
