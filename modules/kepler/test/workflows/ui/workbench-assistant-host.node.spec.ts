// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, test, vi} from 'vitest';
import {
  parseWorkflowDraft,
  validateWorkflowForAssistant
} from '../../../src/workflows/workflow-assistant';
import type {CatalogParameterValue} from '../../../src/analysis/catalog-binding-types';
import {getWorkflowOperationRegistry} from '../../../src/workflows/workflow-operations';
import {profileKeplerFeatureCollection} from '../../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../../src/profile/dataset-profile-types';
import {runWorkbenchAssistantTool} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-assistant';
import {createWorkbenchStore} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {
  readWorkbenchAssistantCall,
  withWorkbenchAssistant
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-assistant-host';
import type {AssistantHost} from '../../../../../examples/deck/kepler-compute/src/assistant-host';
import type {
  WorkbenchDraft,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from '../workflow-fixtures';

function makeStore(): {store: WorkbenchStore; drafts: WorkbenchDraft[]} {
  const drafts: WorkbenchDraft[] = [];
  const store = {
    registry: makeFixtureRegistry(),
    getState: () => ({workflows: []}),
    proposeDraft: (draft: WorkbenchDraft) => drafts.push(draft)
  } as unknown as WorkbenchStore;
  return {store, drafts};
}

function makeHost(calls: unknown[]): AssistantHost {
  return {
    getContext: () => ({}) as ReturnType<AssistantHost['getContext']>,
    runOperation: async (_approved, operation) => {
      calls.push(operation);
      return {accepted: false, code: 'unknown-operation', reason: 'wrapped host'};
    }
  };
}

describe('workbench assistant host', () => {
  test('reads envelope and flat tool calls and ignores other operations', () => {
    expect(
      readWorkbenchAssistantCall({toolName: 'describeAnalysisOperation', args: {operationId: 'x'}})
    ).toEqual({name: 'describeAnalysisOperation', arguments: {operationId: 'x'}});
    expect(readWorkbenchAssistantCall({name: 'listAnalysisOperations'})).toEqual({
      name: 'listAnalysisOperations',
      arguments: {}
    });
    expect(readWorkbenchAssistantCall({type: 'listAnalysisOperations', query: 'scale'})).toEqual({
      name: 'listAnalysisOperations',
      arguments: {query: 'scale'}
    });
    expect(readWorkbenchAssistantCall({type: 'setFilter', field: 'value'})).toBeNull();
    expect(readWorkbenchAssistantCall('defineWorkflow')).toBeNull();
  });

  test('a draft is proposed without approval and nothing reaches the wrapped host', async () => {
    const {store, drafts} = makeStore();
    const calls: unknown[] = [];
    const host = withWorkbenchAssistant(makeHost(calls), () => store);
    const result = await host.runOperation(false, {
      toolName: 'defineWorkflow',
      args: {
        title: 'Scaled points',
        steps: [
          {id: 'source', operationId: 'fixture-source'},
          {id: 'scale', operationId: 'fixture-scale', inputs: {points: 'source'}}
        ]
      }
    });
    expect(result.accepted).toBe(true);
    expect(drafts).toHaveLength(1);
    expect(drafts[0].source).toBe('assistant');
    expect(drafts[0].document.steps.map(step => step.operationId)).toEqual([
      'fixture-source',
      'fixture-scale'
    ]);
    expect(calls).toEqual([]);
  });

  test('running a saved workflow is refused without approval', async () => {
    const calls: unknown[] = [];
    const runs: unknown[] = [];
    const document = makeFixtureDocument();
    let activeWorkflowId: string | null = null;
    const store = {
      registry: makeFixtureRegistry(),
      getState: () => ({
        workflows: [document],
        activeWorkflowId,
        document: activeWorkflowId ? document : null,
        runner: null
      }),
      getRunner: () => null,
      recordAssistantReview: () => undefined,
      getReviewedAssistantIssues: () => [],
      setNotice: () => undefined,
      setActiveWorkflow: (workflowId: string | null) => {
        activeWorkflowId = workflowId;
      },
      run: async (options: unknown) => {
        runs.push(options);
        return null;
      }
    } as unknown as WorkbenchStore;
    const host = withWorkbenchAssistant(makeHost(calls), () => store);
    const call = {toolName: 'runWorkflow', args: {workflowId: document.id}};
    expect(await host.runOperation(false, call)).toMatchObject({
      accepted: false,
      code: 'not-approved'
    });
    expect(runs).toEqual([]);
    expect(await host.runOperation(true, call)).toMatchObject({accepted: true});
    expect(runs).toHaveLength(1);
    expect(calls).toEqual([]);
  });

  test('a malformed draft is rejected and other operations pass through', async () => {
    const {store, drafts} = makeStore();
    const calls: unknown[] = [];
    const host = withWorkbenchAssistant(makeHost(calls), () => store);
    const rejected = await host.runOperation(true, {toolName: 'defineWorkflow', args: {steps: 3}});
    expect(rejected).toMatchObject({accepted: false, code: 'malformed'});
    expect(drafts).toEqual([]);
    const passed = await host.runOperation(true, {type: 'setFilter'});
    expect(passed).toMatchObject({accepted: false, reason: 'wrapped host'});
    expect(calls).toEqual([{type: 'setFilter'}]);
  });
});

function makeAnalysisStore(variant = 'events') {
  const collection = {
    type: 'FeatureCollection' as const,
    features: Array.from({length: 12}, (_, index) => {
      const longitude = -87.6 + (index % 4) * 0.001;
      const latitude = 41.8 + Math.floor(index / 4) * 0.001;
      return {
        type: 'Feature' as const,
        id: index,
        properties: {measurement: index + 1, cases: index + 1, description: `category-${index}`},
        geometry:
          variant === 'areas'
            ? {
                type: 'Polygon',
                coordinates: [
                  [
                    [longitude, latitude],
                    [longitude + 0.001, latitude],
                    [longitude + 0.001, latitude + 0.001],
                    [longitude, latitude + 0.001],
                    [longitude, latitude]
                  ]
                ]
              }
            : {type: 'Point', coordinates: [longitude, latitude]}
      };
    })
  };
  let profile: KeplerDatasetProfile | undefined = {
    ...profileKeplerFeatureCollection(collection, {
      datasetId: 'observations',
      label: 'Observations',
      coordinateSystem: 'EPSG:4326',
      declarations: {
        rowSemantics:
          variant === 'surface' ? 'surface-samples' : variant === 'areas' ? 'objects' : 'events',
        measurement: {measurement: 'amount', cases: 'count'}
      }
    }),
    geometryColumns: ['geometry']
  };
  const initial = profile;
  if (variant === 'unknown-units') profile = {...profile, coordinates: undefined};
  if (variant === 'missing') profile = undefined;
  if (variant === 'over-capacity') profile = {...initial, rowCount: 100000000};
  const host = makeFixtureHost();
  host.listDatasets = () => [
    {
      id: initial.datasetId,
      label: initial.label,
      rowCount: initial.rowCount,
      resident: true,
      geometryFields: ['geometry'],
      fields: []
    }
  ];
  host.getAnalysisProfile = () => profile;
  const reads = vi.spyOn(host, 'readDataset');
  const store = createWorkbenchStore({host, registry: getWorkflowOperationRegistry()});
  const runs = vi.spyOn(store, 'run').mockResolvedValue(null);
  return {
    store,
    reads,
    runs,
    initial,
    setProfile: (next: KeplerDatasetProfile | undefined) => {
      profile = next;
    }
  };
}

const unanswerable = JSON.parse(
  readFileSync(new URL('../fixtures/assistant-unanswerable.json', import.meta.url), 'utf8')
) as {
  id: string;
  request: string;
  profile: string;
  call: {name: string; arguments: unknown};
  requirement: string;
}[];

describe('AX assistant host safety', () => {
  test.each(
    unanswerable
  )('refuses $id with requirement text, without proposing or running', async fixture => {
    const {store, reads, runs} = makeAnalysisStore(fixture.profile);
    const calls: unknown[] = [];
    const host = withWorkbenchAssistant(makeHost(calls), () => store);
    const result = await host.runOperation(true, {
      toolName: fixture.call.name,
      arguments: fixture.call.arguments
    });
    expect(result, fixture.request).toMatchObject({
      accepted: false,
      reason: expect.stringMatching(new RegExp(fixture.requirement))
    });
    expect(store.getState().draft).toBeNull();
    expect(store.getState().workflows).toEqual([]);
    expect(reads).not.toHaveBeenCalled();
    expect(runs).not.toHaveBeenCalled();
    expect(calls).toEqual([]);
    await store.clear();
  });

  test('proposes a mapped template whole, rechecks current profiles on acceptance and again on run', async () => {
    const {store, reads, runs, initial, setProfile} = makeAnalysisStore();
    const host = withWorkbenchAssistant(makeHost([]), () => store);
    const proposal = await host.runOperation(false, {
      toolName: 'defineWorkflow',
      arguments: {
        templateId: 'density-surface',
        datasets: {points: 'observations'},
        columns: {},
        parameters: {bandwidth: 750, cellSize: 500}
      }
    });
    expect(proposal, JSON.stringify(proposal)).toMatchObject({accepted: true});
    const reviewed = store.getState().draft!;
    expect(
      reviewed.document.steps.find(step => step.id === 'analysis')?.parameters['bandwidth']
    ).toEqual({from: 'static', value: 750});
    expect(store.getState().workflows).toEqual([]);
    expect(reads).not.toHaveBeenCalled();
    expect(runs).not.toHaveBeenCalled();
    setProfile(undefined);
    expect(store.acceptDraft()).toBeNull();
    expect(store.getState().notice?.message).toMatch(/current profile/);
    expect(store.getState().draft?.document).toEqual(reviewed.document);
    expect(store.getState().workflows).toEqual([]);
    setProfile(initial);
    const workflowId = store.acceptDraft()!;
    expect(workflowId).toBeTruthy();
    expect(runs).not.toHaveBeenCalled();
    // Explicit coordinate settings are frozen from the reviewed profile. A changed row meaning
    // is still a fresh contradiction and cannot be remedied by silently choosing another method.
    setProfile({...initial, rowSemantics: {value: 'surface-samples', source: 'declared'}});
    const run = {toolName: 'runWorkflow', arguments: {workflowId}};
    const rejected = await host.runOperation(true, run);
    expect(rejected, JSON.stringify(rejected)).toMatchObject({
      accepted: false,
      reason: expect.stringMatching(/events|occurrence|row|samples/i)
    });
    expect(runs).not.toHaveBeenCalled();
    setProfile(initial);
    expect(await host.runOperation(false, run)).toMatchObject({
      accepted: false,
      code: 'not-approved'
    });
    expect(await host.runOperation(true, run)).toMatchObject({accepted: true});
    expect(runs).toHaveBeenCalledExactlyOnceWith({target: undefined});
    expect(store.getState().document?.steps).toEqual(reviewed.document.steps);
    await store.clear();
  });

  test('plain operation plans receive AX checks and preserve data-derived defaults after review', async () => {
    const {store, initial, setProfile, runs} = makeAnalysisStore();
    const result = runWorkbenchAssistantTool(store, {
      name: 'defineWorkflow',
      arguments: {
        title: 'Density',
        steps: [
          {
            id: 'source',
            operationId: 'source-dataset',
            parameters: {
              dataset: 'observations',
              geometryMode: 'geojson-field',
              geometryField: 'geometry'
            }
          },
          {id: 'analysis', operationId: 'kernel-density-grid', inputs: {points: 'source.data'}}
        ]
      }
    });
    expect(result, JSON.stringify(result)).toMatchObject({status: 'awaiting-approval'});
    const settings = store.getState().draft!.document.steps[1].parameters;
    expect(settings['cellSize']?.from).toBe('static');
    setProfile({...initial, extent: [-87.6, 41.8, -87.59, 41.81]});
    expect(store.acceptDraft()).toBeTruthy();
    expect(store.getState().document!.steps[1].parameters).toEqual(settings);
    expect(runs).not.toHaveBeenCalled();
    await store.clear();
  });

  test('template discovery and description are routed without approval', async () => {
    const {store, reads, runs} = makeAnalysisStore();
    const host = withWorkbenchAssistant(makeHost([]), () => store);
    expect(
      await host.runOperation(false, {type: 'listAnalysisTemplates', query: 'density'})
    ).toMatchObject({accepted: true});
    expect(
      await host.runOperation(false, {
        type: 'describeAnalysisTemplate',
        templateId: 'density-surface'
      })
    ).toMatchObject({accepted: true});
    expect(reads).not.toHaveBeenCalled();
    expect(runs).not.toHaveBeenCalled();
    await store.clear();
  });

  test('a changed document during activation cannot run under the earlier approval', async () => {
    const {store, runs} = makeAnalysisStore();
    const document = makeFixtureDocument();
    // Use a separate fixture store to exercise the activation race without AX prerequisites.
    const fixtureStore = createWorkbenchStore({
      registry: makeFixtureRegistry(),
      host: makeFixtureHost()
    });
    const workflowId = fixtureStore.addWorkflow(document);
    const fixtureRuns = vi.spyOn(fixtureStore, 'run').mockResolvedValue(null);
    const activate = fixtureStore.setActiveWorkflow.bind(fixtureStore);
    vi.spyOn(fixtureStore, 'setActiveWorkflow').mockImplementation(identifier => {
      activate(identifier);
      fixtureStore.edit('changed-during-activation', current => ({
        ...current,
        title: 'Changed plan'
      }));
    });
    expect(
      runWorkbenchAssistantTool(
        fixtureStore,
        {name: 'runWorkflow', arguments: {workflowId}},
        {approved: true}
      )
    ).toMatchObject({
      status: 'error',
      message: expect.stringContaining('approved workflow changed')
    });
    expect(fixtureRuns).not.toHaveBeenCalled();
    expect(runs).not.toHaveBeenCalled();
    await fixtureStore.clear();
    await store.clear();
  });

  test('explicit measurement mismatches keep the shared warning for whole-plan review', async () => {
    const {store, initial, setProfile} = makeAnalysisStore();
    setProfile({
      ...initial,
      columns: initial.columns.map(column =>
        column.name === 'measurement'
          ? {...column, measurement: 'rate', measurementSource: 'declared'}
          : column
      )
    });
    const result = runWorkbenchAssistantTool(store, {
      name: 'defineWorkflow',
      arguments: {
        title: 'Weighted density',
        steps: [
          {
            id: 'source',
            operationId: 'source-dataset',
            parameters: {
              dataset: 'observations',
              geometryMode: 'geojson-field',
              geometryField: 'geometry'
            }
          },
          {
            id: 'analysis',
            operationId: 'kernel-density-grid',
            inputs: {points: 'source.data'},
            parameters: {weightProperty: 'measurement'}
          }
        ]
      }
    });
    expect(result, JSON.stringify(result)).toMatchObject({status: 'awaiting-approval'});
    expect(store.getState().draft?.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          severity: 'warning',
          message: expect.stringMatching(/rate|count|amount/)
        })
      ])
    );
    await store.clear();
  });
});

function makeDensityProposal(parameters: Record<string, CatalogParameterValue> = {}) {
  return {
    name: 'defineWorkflow',
    arguments: {
      title: 'Density review regression',
      steps: [
        {
          id: 'source',
          operationId: 'source-dataset',
          parameters: {
            dataset: 'observations',
            geometryMode: 'geojson-field',
            geometryField: 'geometry'
          }
        },
        {
          id: 'analysis',
          operationId: 'kernel-density-grid',
          inputs: {points: 'source.data'},
          parameters
        }
      ]
    }
  };
}

describe('AX assistant independent review regressions', () => {
  test('preserves explicitly requested AX values equal to catalog defaults and derives only omissions', async () => {
    const {store} = makeAnalysisStore();
    expect(
      runWorkbenchAssistantTool(store, makeDensityProposal({cellSize: 100, bandwidth: 500}))
    ).toMatchObject({status: 'awaiting-approval'});
    expect(store.getState().draft!.document.steps[1].parameters).toMatchObject({
      cellSize: {from: 'static', value: 100},
      bandwidth: {from: 'static', value: 500}
    });
    expect(runWorkbenchAssistantTool(store, makeDensityProposal({bandwidth: 500}))).toMatchObject({
      status: 'awaiting-approval'
    });
    const parameters = store.getState().draft!.document.steps[1].parameters;
    expect(parameters['bandwidth']).toEqual({from: 'static', value: 500});
    expect(parameters['cellSize']).not.toEqual({from: 'static', value: 100});
    await store.clear();
  });

  test('retains an explicit coordinate confirmation equal to the catalog default', async () => {
    const {store} = makeAnalysisStore('unknown-units');
    const result = runWorkbenchAssistantTool(
      store,
      makeDensityProposal({coordinateSystem: 'EPSG:4326', cellSize: 100, bandwidth: 500})
    );
    expect(result, JSON.stringify(result)).toMatchObject({status: 'awaiting-approval'});
    expect(store.getState().draft!.document.steps[1].parameters['coordinateSystem']).toEqual({
      from: 'static',
      value: 'EPSG:4326'
    });
    await store.clear();
  });

  test('retains newly introduced warnings in the draft and requires another whole-plan review', async () => {
    const {store, initial, setProfile, runs} = makeAnalysisStore();
    runWorkbenchAssistantTool(store, makeDensityProposal({cellSize: 100, bandwidth: 500}));
    const reviewedDocument = store.getState().draft!.document;
    setProfile({...initial, coordinates: {model: 'planar', source: 'declared'}});
    expect(store.acceptDraft()).toBeNull();
    expect(store.getState().workflows).toEqual([]);
    expect(store.getState().draft?.document).toEqual(reviewedDocument);
    expect(store.getState().draft?.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          severity: 'warning',
          message: expect.stringMatching(/planar.*longitude|longitude.*planar/)
        })
      ])
    );
    expect(store.getState().notice?.message).toMatch(/review/i);
    expect(store.acceptDraft()).toBeTruthy();
    expect(store.getState().document?.steps).toEqual(reviewedDocument.steps);
    expect(runs).not.toHaveBeenCalled();
    const workflowId = store.getState().activeWorkflowId!;
    expect(
      runWorkbenchAssistantTool(
        store,
        {name: 'runWorkflow', arguments: {workflowId}},
        {approved: true}
      )
    ).toMatchObject({
      status: 'started',
      issues: expect.arrayContaining([
        expect.objectContaining({severity: 'warning', message: expect.stringContaining('planar')})
      ])
    });
    expect(runs).toHaveBeenCalledOnce();
    await store.clear();
  });

  test('a newly contradictory approved run displays warnings and waits for renewed review', async () => {
    const {store, initial, setProfile, runs} = makeAnalysisStore();
    runWorkbenchAssistantTool(store, makeDensityProposal({cellSize: 100, bandwidth: 500}));
    const workflowId = store.acceptDraft()!;
    setProfile({...initial, coordinates: {model: 'planar', source: 'declared'}});
    const host = withWorkbenchAssistant(makeHost([]), () => store);
    const call = {toolName: 'runWorkflow', arguments: {workflowId}};
    expect(await host.runOperation(true, call)).toMatchObject({
      accepted: false,
      code: 'not-approved',
      reason: expect.stringMatching(/planar.*longitude|longitude.*planar/)
    });
    expect(runs).not.toHaveBeenCalled();
    expect(store.getState().notice?.message).toMatch(/review/i);
    // Retrying the previously approved call does not acknowledge the new warnings.
    expect(await host.runOperation(true, call)).toMatchObject({
      accepted: false,
      code: 'not-approved'
    });
    expect(runs).not.toHaveBeenCalled();
    const review = runWorkbenchAssistantTool(store, {name: 'runWorkflow', arguments: {workflowId}});
    expect(review).toMatchObject({
      status: 'needs-approval',
      issues: expect.arrayContaining([
        expect.objectContaining({severity: 'warning', message: expect.stringContaining('planar')})
      ])
    });
    expect(await host.runOperation(true, call)).toMatchObject({accepted: true});
    expect(runs).toHaveBeenCalledExactlyOnceWith({target: undefined});
    await store.clear();
  });

  test('targeted runs check only the target applicability and its ancestors but retain structural checks', async () => {
    const {store, runs} = makeAnalysisStore();
    const document = parseWorkflowDraft(
      {
        title: 'Materialize before interpolation',
        steps: [
          {
            id: 'source',
            operationId: 'source-geojson',
            parameters: {
              coordinateSystem: 'EPSG:4326',
              text: JSON.stringify({
                type: 'FeatureCollection',
                features: Array.from({length: 12}, (_, index) => ({
                  type: 'Feature',
                  properties: {measurement: index + 1},
                  geometry: {
                    type: 'Point',
                    coordinates: [-87.6 + (index % 4) * 0.001, 41.8 + Math.floor(index / 4) * 0.001]
                  }
                }))
              })
            }
          },
          {
            id: 'filter',
            operationId: 'table-filter-rows',
            inputs: {rows: 'source.data'},
            parameters: {expression: 'measurement >= 1'}
          },
          {
            id: 'interpolation',
            operationId: 'idw-grid',
            inputs: {points: 'filter.filtered'},
            parameters: {valueProperty: 'measurement'}
          }
        ]
      },
      {registry: store.registry}
    ).document;
    const workflowId = store.addWorkflow(document);
    const call = (target?: string) => {
      const request = {name: 'runWorkflow', arguments: {workflowId, ...(target ? {target} : {})}};
      runWorkbenchAssistantTool(store, request);
      return runWorkbenchAssistantTool(store, request, {approved: true});
    };
    expect(call('source')).toMatchObject({status: 'started'});
    expect(call('filter')).toMatchObject({status: 'started'});
    expect(runs.mock.calls).toEqual([[{target: 'source'}], [{target: 'filter'}]]);
    expect(call('interpolation')).toMatchObject({
      status: 'error',
      message: expect.stringContaining('current profile')
    });
    expect(call()).toMatchObject({
      status: 'error',
      message: expect.stringContaining('current profile')
    });
    expect(runs).toHaveBeenCalledTimes(2);
    store.edit('invalid-downstream-structure', current => ({
      ...current,
      steps: current.steps.map(step =>
        step.id === 'interpolation' ? {...step, operationId: 'missing-operation'} : step
      )
    }));
    expect(call('source')).toMatchObject({
      status: 'error',
      message: expect.stringMatching(/not registered|not available|[Uu]nknown/)
    });
    expect(runs).toHaveBeenCalledTimes(2);
    await store.clear();
  });

  test('targeted applicability still rejects an invalid AX ancestor', async () => {
    const {store, runs} = makeAnalysisStore();
    const document = parseWorkflowDraft(
      makeDensityProposal({weightProperty: 'description'}).arguments,
      {registry: store.registry}
    ).document;
    document.steps.push({
      id: 'filter',
      operationId: 'table-filter-rows',
      inputs: {rows: {stepId: 'analysis', output: 'density-cells'}},
      parameters: {expression: {from: 'static', value: 'density >= 0'}},
      position: {x: 500, y: 0}
    });
    const workflowId = store.addWorkflow(document);
    expect(
      validateWorkflowForAssistant(document, {
        registry: store.registry,
        host: store.host,
        target: 'source'
      }).filter(issue => issue.severity === 'error')
    ).toEqual([]);
    runWorkbenchAssistantTool(store, {
      name: 'runWorkflow',
      arguments: {workflowId, target: 'source'}
    });
    expect(
      runWorkbenchAssistantTool(
        store,
        {name: 'runWorkflow', arguments: {workflowId, target: 'source'}},
        {approved: true}
      )
    ).toMatchObject({status: 'started'});
    expect(
      runWorkbenchAssistantTool(
        store,
        {name: 'runWorkflow', arguments: {workflowId, target: 'filter'}},
        {approved: true}
      )
    ).toMatchObject({status: 'error', message: expect.stringMatching(/number|[Ww]eight|numeric/)});
    expect(runs).toHaveBeenCalledExactlyOnceWith({target: 'source'});
    await store.clear();
  });
});

describe('review-fix-host exact plan approval', () => {
  test.each([
    'parameters',
    'target'
  ] as const)('requires renewed review after %s changes without new warnings', async change => {
    const store = createWorkbenchStore({registry: makeFixtureRegistry(), host: makeFixtureHost()});
    const workflowId = store.addWorkflow(makeFixtureDocument());
    const execute = vi.spyOn(store, 'run').mockResolvedValue(null);
    const call = {name: 'runWorkflow', arguments: {workflowId}};
    expect(runWorkbenchAssistantTool(store, call)).toMatchObject({status: 'needs-approval'});
    if (change === 'parameters')
      store.edit('factor', document => ({
        ...document,
        steps: document.steps.map(step =>
          step.operationId === 'fixture-scale'
            ? {
                ...step,
                parameters: {...step.parameters, factor: {from: 'static', value: 99}}
              }
            : step
        )
      }));
    const changedCall =
      change === 'target' ? {...call, arguments: {workflowId, target: 'source'}} : call;
    expect(runWorkbenchAssistantTool(store, changedCall, {approved: true})).toMatchObject({
      status: 'needs-approval'
    });
    expect(execute).not.toHaveBeenCalled();
    expect(runWorkbenchAssistantTool(store, changedCall)).toMatchObject({status: 'needs-approval'});
    expect(runWorkbenchAssistantTool(store, changedCall, {approved: true})).toMatchObject({
      status: 'started'
    });
    expect(execute).toHaveBeenCalledOnce();
    await store.clear();
  });

  test('missing review cannot be replaced by an approved flag', async () => {
    const store = createWorkbenchStore({registry: makeFixtureRegistry(), host: makeFixtureHost()});
    const workflowId = store.addWorkflow(makeFixtureDocument());
    const execute = vi.spyOn(store, 'run').mockResolvedValue(null);
    expect(
      runWorkbenchAssistantTool(
        store,
        {name: 'runWorkflow', arguments: {workflowId}},
        {approved: true}
      )
    ).toMatchObject({status: 'needs-approval'});
    expect(execute).not.toHaveBeenCalled();
    await store.clear();
  });
});
