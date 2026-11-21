// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {describe, expect, test} from 'vitest';
import type {ComponentProps, ReactElement, ReactNode} from 'react';
import {setWorkflowParameter} from '../../../src/workflows/workflow-edit';
import {WorkflowHistory} from '../../../src/workflows/workflow-history';
import {WorkflowError} from '../../../src/workflows/workflow-types';
import {getWorkflowSqlOperations} from '../../../src/workflows/workflow-sql-operation';
import {getWorkflowSourceOperations} from '../../../src/workflows/workflow-source-operations';
import type {
  WorkflowDatasetProfile,
  WorkflowDrawing,
  WorkflowDocument,
  WorkflowOperation,
  WorkflowOperationRegistry,
  WorkflowParameterSpec,
  WorkflowRunnerState,
  WorkflowIssue,
  WorkflowStep,
  WorkflowStepResult
} from '../../../src/workflows/workflow-types';
import {getWorkflowParameterProblem} from '../../../src/workflows/workflow-parameters';
import {
  makeFixtureDocument,
  makeFixtureHost,
  makeFixturePoints,
  makeFixtureRegistry
} from '../workflow-fixtures';
import {WORKBENCH_TEST_IDS} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import {WorkbenchInspector} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-inspector';
import {
  WorkbenchParameterForm,
  WorkbenchParameterValue
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-form';
import {
  coerceWorkbenchParameterText,
  formatWorkbenchCapacity,
  formatWorkbenchParameterHint,
  getWorkbenchInspectorIssues,
  hasWorkbenchRunOptionOverrides,
  getWorkbenchDisplayColumns,
  getWorkbenchIssueSuggestions,
  getWorkbenchColumnSuggestions,
  getWorkbenchConnectionChoices,
  getWorkbenchDatasetParameters,
  getWorkbenchSourceParameters,
  getWorkbenchInspectorTabIndex,
  getWorkbenchParameterRows,
  getWorkbenchParameterControlSpec,
  insertWorkbenchExpressionColumn
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-model';
import {createWorkbenchStore} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import type {
  WorkbenchState,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement, isValidElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

// A synchronous dispatcher keeps hook state and flushes effects between handler probes.
// It exercises the component's real callbacks without claiming to emulate browser scheduling.
function makeHookHarness<Properties>(
  component: (properties: Properties) => React.JSX.Element | null
) {
  const internals = (
    hostRequire('react') as {
      __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
    }
  ).__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const slots: unknown[] = [];
  const cleanups = new Map<number, () => void>();
  let cursor = 0;
  let dirty = false;
  let effects: Array<() => void> = [];
  function useSlot<Value>(create: () => Value): {value: Value; index: number} {
    const index = cursor++;
    if (slots.length <= index) slots.push(create());
    return {value: slots[index] as Value, index};
  }
  function equalDependencies(
    left: readonly unknown[] | undefined,
    right: readonly unknown[] | undefined
  ): boolean {
    return Boolean(
      left &&
        right &&
        left.length === right.length &&
        left.every((value, index) => Object.is(value, right[index]))
    );
  }
  const dispatcher = {
    useState<Value>(initial: Value | (() => Value)) {
      const slot = useSlot(() => ({
        current: typeof initial === 'function' ? (initial as () => Value)() : initial
      })).value;
      return [
        slot.current,
        (next: Value | ((previous: Value) => Value)) => {
          const value =
            typeof next === 'function' ? (next as (previous: Value) => Value)(slot.current) : next;
          if (!Object.is(value, slot.current)) {
            slot.current = value;
            dirty = true;
          }
        }
      ];
    },
    useRef<Value>(initial: Value) {
      return useSlot(() => ({current: initial})).value;
    },
    useId() {
      return useSlot(() => `probe-${cursor}`).value;
    },
    useMemo<Value>(create: () => Value, dependencies: readonly unknown[] | undefined) {
      const slot = useSlot(() => ({
        initialized: false,
        current: undefined as Value | undefined,
        dependencies: undefined as readonly unknown[] | undefined
      })).value;
      if (!slot.initialized || !equalDependencies(slot.dependencies, dependencies)) {
        slot.current = create();
        slot.dependencies = dependencies;
        slot.initialized = true;
      }
      return slot.current;
    },
    useEffect(effect: () => void | (() => void), dependencies: readonly unknown[] | undefined) {
      const {value: slot, index} = useSlot(() => ({
        initialized: false,
        dependencies: undefined as readonly unknown[] | undefined
      }));
      if (!slot.initialized || !equalDependencies(slot.dependencies, dependencies)) {
        slot.initialized = true;
        slot.dependencies = dependencies;
        effects.push(() => {
          cleanups.get(index)?.();
          const cleanup = effect();
          if (cleanup) cleanups.set(index, cleanup);
          else cleanups.delete(index);
        });
      }
    },
    useSyncExternalStore<Value>(_subscribe: unknown, snapshot: () => Value) {
      useSlot(() => null);
      return snapshot();
    }
  };
  return {
    render(properties: Properties): React.JSX.Element | null {
      for (let iteration = 0; iteration < 20; iteration++) {
        cursor = 0;
        dirty = false;
        effects = [];
        const previous = internals.H;
        let tree: React.JSX.Element | null;
        try {
          internals.H = dispatcher;
          tree = component(properties);
        } finally {
          internals.H = previous;
        }
        for (const effect of effects) effect();
        if (!dirty) return tree;
      }
      throw new Error('Hook probe did not settle within 20 renders.');
    },
    unmount() {
      for (const cleanup of cleanups.values()) cleanup();
    }
  };
}

type ProbeElement = ReactElement<Record<string, unknown>>;

function findProbeElement(
  tree: ReactNode,
  predicate: (element: ProbeElement) => boolean
): ProbeElement {
  function find(node: ReactNode): ProbeElement | undefined {
    if (Array.isArray(node)) {
      for (const child of node) {
        const found = find(child);
        if (found) return found;
      }
    } else if (isValidElement<Record<string, unknown>>(node)) {
      if (predicate(node)) return node;
      return find(node.props['children'] as ReactNode);
    }
    return undefined;
  }
  const element = find(tree);
  if (!element) throw new Error('Probe element was not found.');
  return element;
}

function findProbeControl(tree: ReactNode, testId: string): ProbeElement {
  return findProbeElement(tree, element => element.props['data-testid'] === testId);
}

function invokeProbeHandler(
  element: ProbeElement,
  name: string,
  ...handlerArguments: unknown[]
): unknown {
  const handler = element.props[name];
  if (typeof handler !== 'function') throw new Error(`Probe handler ${name} was not found.`);
  return (handler as (...values: unknown[]) => unknown)(...handlerArguments);
}

function makeDeferredText() {
  let complete: ((value: string) => void) | undefined;
  let fail: ((reason: Error) => void) | undefined;
  const promise = new Promise<string>((resolve, reject) => {
    complete = resolve;
    fail = reject;
  });
  return {
    promise,
    resolve(value: string) {
      complete!(value);
    },
    reject(error: Error) {
      fail!(error);
    }
  };
}

function makeFileEvent(text: Promise<string>, size = 5) {
  return {target: {value: 'file.geojson', files: [{size, text: () => text}]}};
}

function makeParameterProbe(store: WorkbenchStore, name: string, stepId = 'scale') {
  const form = makeHookHarness(WorkbenchParameterForm);
  const control = makeHookHarness(WorkbenchParameterValue);
  return {
    renderForm() {
      return form.render({store, stepId});
    },
    render() {
      const tree = form.render({store, stepId});
      const element = findProbeElement(
        tree,
        candidate =>
          candidate.type === WorkbenchParameterValue &&
          candidate.props['testId'] === WORKBENCH_TEST_IDS.parameter(name)
      );
      const properties = element.props as unknown as ComponentProps<typeof WorkbenchParameterValue>;
      return control.render(properties);
    }
  };
}

function makeStepInspectorProbe(store: WorkbenchStore) {
  const inspector = makeHookHarness(WorkbenchInspector);
  const parameters = makeHookHarness((element: ProbeElement) => {
    if (typeof element.type !== 'function')
      throw new Error('Step parameters component was not found.');
    return (element.type as (properties: Record<string, unknown>) => React.JSX.Element)(
      element.props
    );
  });
  return {
    render() {
      const tree = inspector.render({store});
      const element = findProbeElement(tree, candidate =>
        Boolean(
          candidate.props['operation'] && candidate.props['step'] && candidate.props['document']
        )
      );
      return parameters.render(element);
    }
  };
}

const DATASETS: readonly WorkflowDatasetProfile[] = [
  {
    id: 'cities',
    label: 'Cities',
    rowCount: 1204,
    resident: true,
    fields: [
      {name: 'population', type: 'real'},
      {name: 'shape', type: 'geojson'}
    ],
    geometryFields: ['shape']
  }
];

function makeResult(): WorkflowStepResult {
  return {
    key: 'old',
    outputs: {data: makeFixturePoints(2)},
    diagnostics: [],
    summary: {},
    engine: 'host',
    dispatchedContributors: [],
    parameters: {},
    durationMilliseconds: 4,
    cacheHit: false,
    completedAt: 10
  };
}

function makeRunner(): WorkflowRunnerState {
  return {
    revision: 1,
    running: false,
    order: ['source', 'scale', 'count'],
    issues: [],
    steps: {
      source: {key: 'new', status: 'stale', issues: [], result: makeResult()},
      scale: {key: 'scale', status: 'ready', issues: []},
      count: {key: 'count', status: 'ready', issues: []}
    }
  };
}

function makeStore(patch: Partial<WorkbenchState> = {}, registry = makeFixtureRegistry()) {
  const document = makeFixtureDocument();
  let state: WorkbenchState = {
    revision: 1,
    workflows: [document],
    activeWorkflowId: document.id,
    document,
    runner: makeRunner(),
    selectedStepIds: ['scale'],
    canUndo: false,
    canRedo: false,
    dock: {open: true, height: 380, inspectorTab: 'parameters', logOpen: false},
    palette: {open: false},
    dialog: null,
    draft: null,
    notice: null,
    runLog: [],
    templates: [],
    clipboardStepCount: 0,
    ...patch
  };
  const history = new WorkflowHistory(state.document ?? document, {now: () => 100});
  const refusals: unknown[] = [];
  const calls: unknown[][] = [];
  const record = (name: string, ...values: unknown[]) => {
    calls.push([name, ...values]);
  };
  const store: WorkbenchStore = {
    registry,
    host: makeFixtureHost(DATASETS),
    getState: () => state,
    subscribe: () => () => {},
    createWorkflow: options => {
      record('createWorkflow', options);
      return '';
    },
    addWorkflow: value => {
      record('addWorkflow', value);
      return '';
    },
    duplicateWorkflow: value => {
      record('duplicateWorkflow', value);
      return '';
    },
    deleteWorkflow: value => record('deleteWorkflow', value),
    setActiveWorkflow: value => record('setActiveWorkflow', value),
    edit: (tag, change) => {
      record('edit', tag, change);
      if (!state.document) return false;
      try {
        history.apply(change(state.document), tag);
        state = {...state, document: history.document, canUndo: history.canUndo, notice: null};
        return true;
      } catch (error) {
        refusals.push(error);
        state = {
          ...state,
          notice: {kind: 'error', message: error instanceof Error ? error.message : String(error)}
        };
        return false;
      }
    },
    undo: () => {
      record('undo');
      history.undo();
      state = {...state, document: history.document, canUndo: history.canUndo};
    },
    redo: () => {
      record('redo');
      history.redo();
      state = {...state, document: history.document, canUndo: history.canUndo};
    },
    select: value => {
      record('select', value);
      state = {...state, selectedStepIds: value};
    },
    run: async options => {
      record('run', options);
      return null;
    },
    cancel: () => record('cancel'),
    notifyExternalChange: () => record('notifyExternalChange'),
    getRunner: () => null,
    setDock: value => {
      record('setDock', value);
      state = {...state, dock: {...state.dock, ...value}};
    },
    openPalette: value => record('openPalette', value),
    closePalette: () => record('closePalette'),
    setDialog: value => record('setDialog', value),
    setNotice: value => record('setNotice', value),
    proposeDraft: value => record('proposeDraft', value),
    acceptDraft: () => {
      record('acceptDraft');
      return null;
    },
    rejectDraft: () => record('rejectDraft'),
    saveTemplate: () => '',
    deleteTemplate: () => {},
    useTemplate: () => '',
    insertWorkflow: () => false,
    copySelection: () => false,
    cutSelection: () => false,
    paste: () => false,
    serialize: () => undefined,
    restore: value => record('restore', value),
    getReviewedAssistantIssues: () => [],
    recordAssistantReview: () => {},
    clear: async () => {
      record('clear');
    }
  };
  return {store, calls, history, refusals};
}

function replaceOperation(operation: WorkflowOperation): WorkflowOperationRegistry {
  const registry = makeFixtureRegistry();
  return {
    get: identifier => (identifier === operation.id ? operation : registry.get(identifier)),
    list: () =>
      registry.list().map(candidate => (candidate.id === operation.id ? operation : candidate))
  };
}

function makeKindsOperation(): WorkflowOperation {
  const parameter = (
    name: string,
    kind: WorkflowParameterSpec['kind'],
    defaultValue: WorkflowParameterSpec['defaultValue'],
    extra: Partial<WorkflowParameterSpec> = {}
  ): WorkflowParameterSpec => ({name, label: `Field ${name}`, kind, defaultValue, ...extra});
  return {
    ...makeFixtureRegistry().get('fixture-scale')!,
    parameters: [
      parameter('distance', 'number', 500, {unit: 'crs', minimum: 0, maximum: 100000}),
      parameter('count', 'integer', 4, {minimum: 1, maximum: 50}),
      parameter('enabled', 'boolean', true),
      parameter('mode', 'enum', 'first', {options: ['first', 'second', 'third']}),
      parameter('selectMode', 'enum', 'a', {options: ['a', 'b', 'c', 'd']}),
      parameter('name', 'string', 'name'),
      parameter('levels', 'number-list', [1, 2]),
      parameter('geojson', 'text', '{}', {maximumLength: 100}),
      parameter('datasetId', 'dataset', 'cities'),
      parameter('datasetLabel', 'string', 'Cities'),
      parameter('field', 'column', 'value'),
      parameter('fields', 'column-list', 'point_id, value'),
      parameter('expression', 'expression', 'value * 2'),
      parameter('hidden', 'string', 'secret', {
        visibleWhen: {parameter: 'enabled', equals: [false]}
      })
    ]
  };
}

function makeFileStore() {
  const original = makeKindsOperation();
  const operation = {
    ...original,
    id: 'source-geojson',
    parameters: original.parameters.map(spec =>
      spec.name === 'geojson' ? {...spec, name: 'text'} : spec
    )
  };
  const document = makeFixtureDocument();
  document.steps[1] = {...document.steps[1], operationId: operation.id};
  return makeStore({document}, replaceOperation(operation));
}

describe('workbench parameter model', () => {
  test('resolves defaults, static values and workflow inputs before visibility', () => {
    const document = makeFixtureDocument();
    const operation = makeKindsOperation();
    const step = document.steps[1];
    document.inputs = [{id: 'distance', label: 'Radius', kind: 'number', value: 20}];
    step.parameters = {
      distance: {from: 'input', inputId: 'distance'},
      levels: {from: 'static', value: [1, 2]},
      name: {from: 'static', value: 'custom'}
    };
    const rows = getWorkbenchParameterRows(document, step, operation);
    expect(rows.map(row => row.spec.name)).toEqual([
      'distance',
      'count',
      'enabled',
      'mode',
      'selectMode',
      'name',
      'levels',
      'geojson',
      'datasetId',
      'datasetLabel',
      'field',
      'fields',
      'expression'
    ]);
    expect(rows[0]).toEqual({
      spec: operation.parameters[0],
      value: 20,
      bindingKind: 'input',
      inputId: 'distance',
      isDefault: false
    });
    expect(rows[1]).toEqual({
      spec: operation.parameters[1],
      value: 4,
      bindingKind: 'default',
      isDefault: true
    });
    expect(rows[6]).toEqual({
      spec: operation.parameters[6],
      value: [1, 2],
      bindingKind: 'static',
      isDefault: true
    });
    step.parameters['enabled'] = {from: 'static', value: false};
    expect(getWorkbenchParameterRows(document, step, operation).at(-1)?.spec.name).toBe('hidden');
  });

  test('coerces valid text without losing string whitespace', () => {
    const operation = makeKindsOperation();
    expect(coerceWorkbenchParameterText(operation.parameters[0], '1e3')).toBe(1000);
    expect(coerceWorkbenchParameterText(operation.parameters[1], '4')).toBe(4);
    expect(coerceWorkbenchParameterText(operation.parameters[6], ' 1, -2, 3.5 ')).toEqual([
      1, -2, 3.5
    ]);
    expect(coerceWorkbenchParameterText(operation.parameters[5], ' a b ')).toBe(' a b ');
    expect(getWorkflowParameterProblem(operation.parameters[0], '-')).toBe(
      'must be a number; got "-"'
    );
    expect(getWorkflowParameterProblem(operation.parameters[0], '1e')).toBe(
      'must be a number; got "1e"'
    );
    expect(getWorkflowParameterProblem(operation.parameters[1], '1.5')).toBe(
      'must be a whole number; got 1.5'
    );
    expect(getWorkflowParameterProblem(operation.parameters[0], '-1')).toBe(
      'must be between 0 and 100000; got -1'
    );
  });

  test('only offers valid connections and refuses self, cycles and incompatible table outputs', () => {
    const document = makeFixtureDocument();
    const registry = makeFixtureRegistry();
    const input = registry.get('fixture-scale')!.inputs[0];
    expect(getWorkbenchConnectionChoices(document, registry, null, 'scale', input)).toEqual([
      {
        value: '["source","data"]',
        label: 'Fixture source · Data (Features)',
        connection: {stepId: 'source', output: 'data'}
      }
    ]);
    const source = document.steps[0];
    source.operationId = 'fixture-scale';
    source.inputs = {points: {stepId: 'scale', output: 'scaled'}};
    expect(getWorkbenchConnectionChoices(document, registry, null, 'scale', input)).toEqual([]);
  });

  test('checks numeric, list and text limits including boundary values', () => {
    const operation = makeKindsOperation();
    expect(getWorkflowParameterProblem(operation.parameters[0], '100000')).toBeNull();
    expect(getWorkflowParameterProblem(operation.parameters[0], '100001')).toBe(
      'must be between 0 and 100000; got 100001'
    );
    expect(getWorkflowParameterProblem(operation.parameters[6], '1,2,3,4,5,6,7,8')).toBeNull();
    expect(getWorkflowParameterProblem(operation.parameters[6], '1,2,3,4,5,6,7,8,9')).toBe(
      'must have 1 to 8 numbers; got 9'
    );
    expect(getWorkflowParameterProblem(operation.parameters[6], '1,')).toBe(
      'entry 2 must be a number; got ""'
    );
    expect(getWorkflowParameterProblem(operation.parameters[7], 'x'.repeat(100))).toBeNull();
    expect(getWorkflowParameterProblem(operation.parameters[7], 'x'.repeat(101))).toBe(
      'must be at most 100 characters; got 101'
    );
  });

  test('formats declared row and vertex limits', () => {
    expect(
      formatWorkbenchCapacity({
        name: 'points',
        label: 'Points',
        accepts: ['point'],
        maximumRows: 1204,
        maximumVertices: 100000
      })
    ).toBe('Up to 1,204 rows · 100,000 vertices');
    expect(
      formatWorkbenchCapacity({
        name: 'rows',
        label: 'Rows',
        kind: 'table',
        identityColumns: [],
        maximumRows: 0
      })
    ).toBe('Up to 0 rows');
  });

  test('suggests non-geometry columns from the latest stale producer result', () => {
    const document = makeFixtureDocument();
    const registry = makeFixtureRegistry();
    const operation = registry.get('fixture-scale')!;
    const spec: WorkflowParameterSpec = {
      name: 'field',
      label: 'Field',
      kind: 'column',
      columnsFrom: 'points',
      defaultValue: ''
    };
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        spec,
        registry,
        makeRunner()
      )
    ).toEqual({columns: ['point_id', 'value'], hints: []});
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        {...spec, columnsFrom: 'missing'},
        registry,
        makeRunner()
      )
    ).toEqual({columns: [], hints: []});
  });

  test('suggests identities with a run hint until the producer has a result', () => {
    const document = makeFixtureDocument();
    document.steps[0].label = 'Cities source';
    const registry = makeFixtureRegistry();
    const operation = registry.get('fixture-scale')!;
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        {name: 'field', label: 'Field', kind: 'column', defaultValue: ''},
        registry,
        null
      )
    ).toEqual({columns: ['point_id'], hints: ['Run Cities source to list its columns']});
    document.steps[1].inputs = {};
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        {name: 'field', label: 'Field', kind: 'column', defaultValue: ''},
        registry,
        null
      )
    ).toEqual({columns: [], hints: []});
  });

  test('suggests profile fields from a dataset parameter', () => {
    const document = makeFixtureDocument();
    const operation = makeKindsOperation();
    const spec = {...operation.parameters[10], columnsFrom: 'datasetId'};
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        spec,
        makeFixtureRegistry(),
        null,
        DATASETS
      )
    ).toEqual({columns: ['population', 'shape'], hints: []});
    document.steps[1].parameters['datasetId'] = {from: 'static', value: 'missing'};
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        spec,
        makeFixtureRegistry(),
        null,
        DATASETS
      )
    ).toEqual({columns: [], hints: []});
  });

  test('deduplicates columns and hints across connected input ports', () => {
    const document = makeFixtureDocument();
    const registry = makeFixtureRegistry();
    const original = registry.get('fixture-scale')!;
    const operation = {
      ...original,
      inputs: [...original.inputs, {...original.inputs[0], name: 'other'}]
    };
    document.steps[1].inputs['other'] = {stepId: 'source', output: 'data'};
    const spec: WorkflowParameterSpec = {
      name: 'field',
      label: 'Field',
      kind: 'column',
      defaultValue: ''
    };
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        spec,
        registry,
        makeRunner()
      )
    ).toEqual({columns: ['point_id', 'value'], hints: []});
    expect(
      getWorkbenchColumnSuggestions(document, document.steps[1], operation, spec, registry, null)
    ).toEqual({columns: ['point_id'], hints: ['Run Fixture source to list its columns']});
  });

  test('resolves a hidden dataset parameter for suggestions without depending on form visibility', () => {
    const document = makeFixtureDocument();
    const original = makeKindsOperation();
    const operation = {
      ...original,
      parameters: original.parameters.map(spec =>
        spec.name === 'datasetId'
          ? {...spec, visibleWhen: {parameter: 'enabled', equals: [false]}}
          : spec
      )
    };
    expect(
      getWorkbenchParameterRows(document, document.steps[1], operation).some(
        row => row.spec.name === 'datasetId'
      )
    ).toBe(false);
    expect(
      getWorkbenchColumnSuggestions(
        document,
        document.steps[1],
        operation,
        {...original.parameters[10], columnsFrom: 'datasetId'},
        makeFixtureRegistry(),
        null,
        DATASETS
      )
    ).toEqual({columns: ['population', 'shape'], hints: []});
  });

  test('infers dataset geometry and label while preserving custom and promoted settings', () => {
    const operation = {
      ...makeKindsOperation(),
      parameters: [
        ...makeKindsOperation().parameters,
        ...['geometryMode', 'geometryField', 'longitudeField', 'latitudeField'].map(name => ({
          name,
          label: name,
          kind: 'string' as const,
          defaultValue: ''
        }))
      ]
    };
    const step = makeFixtureDocument().steps[1];
    expect(getWorkbenchDatasetParameters(step, operation, 'datasetId', DATASETS[0])).toEqual({
      datasetId: 'cities',
      datasetLabel: 'Cities',
      geometryMode: 'geojson-field',
      geometryField: 'shape'
    });
    step.parameters['geometryMode'] = {from: 'input', inputId: 'mode'};
    step.parameters['geometryField'] = {from: 'static', value: 'custom'};
    expect(getWorkbenchDatasetParameters(step, operation, 'datasetId', DATASETS[0])).toEqual({
      datasetId: 'cities',
      datasetLabel: 'Cities'
    });
    step.parameters = {};
    const pointDataset = {
      ...DATASETS[0],
      geometryFields: [],
      pointColumns: {longitude: 'lng', latitude: 'lat'}
    };
    expect(getWorkbenchDatasetParameters(step, operation, 'datasetId', pointDataset)).toEqual({
      datasetId: 'cities',
      datasetLabel: 'Cities',
      geometryMode: 'point-columns',
      longitudeField: 'lng',
      latitudeField: 'lat'
    });
    expect(
      getWorkbenchDatasetParameters(step, operation, 'datasetId', {
        ...DATASETS[0],
        geometryFields: []
      })
    ).toEqual({datasetId: 'cities', datasetLabel: 'Cities', geometryMode: 'none'});
  });

  test('inserts expression columns at the selection with safe quoting', () => {
    expect(insertWorkbenchExpressionColumn('value + 2', 'population', 0, 5)).toEqual({
      text: 'population + 2',
      cursor: 10
    });
    expect(insertWorkbenchExpressionColumn('2 * ', 'city name', 4)).toEqual({
      text: '2 * `city name`',
      cursor: 15
    });
    expect(insertWorkbenchExpressionColumn('', 'true', 0)).toEqual({text: '`true`', cursor: 6});
    expect(insertWorkbenchExpressionColumn('', 'a`b\\c', 0)).toEqual({
      text: '`a\\`b\\\\c`',
      cursor: 9
    });
  });

  test('maps tab navigation including wraparound and ignores typing', () => {
    expect(getWorkbenchInspectorTabIndex('ArrowRight', 2)).toBe(0);
    expect(getWorkbenchInspectorTabIndex('ArrowLeft', 0)).toBe(2);
    expect(getWorkbenchInspectorTabIndex('ArrowDown', 0)).toBe(1);
    expect(getWorkbenchInspectorTabIndex('ArrowUp', 1)).toBe(0);
    expect(getWorkbenchInspectorTabIndex('Home', 2)).toBe(0);
    expect(getWorkbenchInspectorTabIndex('End', 0)).toBe(2);
    expect(getWorkbenchInspectorTabIndex('Enter', 0)).toBeNull();
  });
});

describe('workbench inspector markup', () => {
  test('shows a selected step, connection choices, errors and engine controls', () => {
    const document = makeFixtureDocument();
    document.steps[1].parameters['factor'] = {from: 'static', value: 150};
    const registry = makeFixtureRegistry();
    const operation = {
      ...registry.get('fixture-scale')!,
      parameters: registry
        .get('fixture-scale')!
        .parameters.map(spec => (spec.name === 'factor' ? {...spec, unit: 'crs'} : spec))
    };
    const runner = makeRunner();
    runner.steps['scale'].status = 'not-configured';
    runner.steps['scale'].issues = [
      {
        severity: 'error',
        code: 'parameter-out-of-range',
        stepId: 'scale',
        message: 'Fixture scale: Factor must be between 0 and 100; got 150.'
      }
    ];
    const {store, calls} = makeStore({document, runner}, replaceOperation(operation));
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain('data-testid="workflow-inspector"');
    expect(markup).toContain('aria-label="Step inspector"');
    expect(markup).toContain('data-testid="workflow-step-label"');
    expect(markup).toContain('placeholder="Fixture scale"');
    expect(markup).toContain('data-status="not-configured"');
    expect(markup).toContain('Not configured');
    expect(markup).toContain('error: Factor must be between 0 and 100; got 150.');
    expect(markup).toContain('data-testid="workflow-input-points"');
    expect(markup).toContain('Not connected');
    expect(markup).toContain('Fixture source · Data (Features)');
    expect(markup).not.toContain('Fixture count · Count (Table)');
    expect(markup).toContain('Up to 10 rows · 10 vertices');
    expect(markup).toContain('CRS units');
    expect(markup).toContain('0 to 100');
    expect(markup).toContain('0 to 100 · default 2');
    expect(markup).toContain('data-testid="workflow-parameter-error-factor"');
    expect(markup).toMatch(/aria-invalid="true" aria-describedby="[^"]+-hint [^"]+-error"/);
    expect(markup).toContain(
      'role="alert">Fixture scale: Factor must be between 0 and 100; got 150.'
    );
    expect(markup).toContain('Workflow default');
    expect(markup).toContain('GPU when available');
    expect(markup).toContain('Continue on CPU and warn');
    expect(markup).toContain('aria-label="Show Scaled on the map"');
    expect(markup).toContain('aria-label="Disabled"');
    expect(calls).toEqual([]);
  });

  test('shows promoted parameter values and pressed promote controls', () => {
    const document = makeFixtureDocument();
    document.inputs = [{id: 'radius', label: 'Radius', kind: 'number', value: 8}];
    document.steps[1].parameters['factor'] = {from: 'input', inputId: 'radius'};
    const {store} = makeStore({document});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain('Workflow input: Radius');
    expect(markup).toContain('value="8"');
    expect(markup).toContain('data-testid="workflow-parameter-promote-factor"');
    expect(markup).toContain('aria-pressed="true"');
  });

  test('shows description and workflow input editor when no steps are selected', () => {
    const document = makeFixtureDocument();
    document.description = 'My workflow';
    document.inputs = [
      {
        id: 'radius',
        label: 'Radius',
        kind: 'number',
        value: 8,
        minimum: 0,
        maximum: 20,
        unit: 'meters'
      }
    ];
    document.steps[1].parameters['factor'] = {from: 'input', inputId: 'radius'};
    const runner = makeRunner();
    runner.issues = [
      {severity: 'error', code: 'cycle', message: 'Fixture workflow: Steps form a cycle.'}
    ];
    const {store} = makeStore({document, selectedStepIds: [], runner});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain('aria-label="Description"');
    expect(markup).toContain('My workflow');
    expect(markup).toContain('aria-label="Label for Radius"');
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.workflowInput('radius')}"`);
    expect(markup).toContain('value="8"');
    expect(markup).toContain('0 to 20');
    expect(markup).toContain('meters');
    expect(markup).toContain('Used by Fixture scale');
    expect(markup).toContain('aria-label="Remove workflow input Radius"');
    expect(markup).toContain('error: Fixture workflow: Steps form a cycle.');
  });

  test('shows bulk actions for multiple steps', () => {
    const {store} = makeStore({selectedStepIds: ['source', 'scale']});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain('2 steps selected');
    for (const label of ['Disable', 'Enable', 'Duplicate', 'Delete'])
      expect(markup).toContain(`>${label}</button>`);
    expect(markup).not.toContain('role="tablist"');
  });

  test('unknown operations only offer label, note and delete', () => {
    const document = makeFixtureDocument();
    document.steps[1].operationId = 'unknown-operation';
    const {store} = makeStore({document});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain('Operation <code>unknown-operation</code> is not in this build.');
    expect(markup).toContain('aria-label="Step label"');
    expect(markup).toContain('aria-label="Note"');
    expect(markup).toContain('>Delete</button>');
    expect(markup).not.toContain('role="tablist"');
    expect(markup).not.toContain('workflow-parameter-');
  });

  test('renders no workflow, an empty workflow, a missing selection and a null runner', () => {
    const empty: WorkflowDocument = {...makeFixtureDocument(), steps: []};
    expect(
      renderToStaticMarkup(
        createElement(WorkbenchInspector, {
          store: makeStore({document: null, workflows: [], runner: null}).store
        })
      )
    ).toContain('Create or select a workflow to inspect it.');
    expect(
      renderToStaticMarkup(
        createElement(WorkbenchInspector, {store: makeStore({document: empty}).store})
      )
    ).toContain('Workflow settings');
    expect(
      renderToStaticMarkup(
        createElement(WorkbenchInspector, {store: makeStore({selectedStepIds: ['missing']}).store})
      )
    ).toContain('Workflow settings');
    expect(
      renderToStaticMarkup(
        createElement(WorkbenchInspector, {store: makeStore({runner: null}).store})
      )
    ).toContain('data-status="not-configured"');
  });

  test('renders running and stale statuses as text', () => {
    for (const status of ['running', 'stale'] as const) {
      const runner = makeRunner();
      runner.running = status === 'running';
      runner.steps['scale'].status = status;
      const markup = renderToStaticMarkup(
        createElement(WorkbenchInspector, {store: makeStore({runner}).store})
      );
      expect(markup).toContain(`data-status="${status}"`);
      expect(markup).toContain(status === 'running' ? 'Running' : 'Stale');
      expect(markup).toContain('aria-live="polite"');
    }
  });

  test('contract names behaviour, limits, engine, origin and id', () => {
    const {store} = makeStore({
      dock: {open: true, height: 380, inspectorTab: 'contract', logOpen: false}
    });
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain('Fixture operation.');
    expect(markup).toContain('Fixture only.');
    expect(markup).toContain('GPU when available, with a CPU reference');
    expect(markup).toContain('Workflow operation');
    expect(markup).toContain('fixture-scale</code>');
    expect(markup).toContain('Up to 10 rows · 10 vertices');
    expect(markup).toMatch(
      /role="tab"[^>]*data-testid="workflow-inspector-tab-contract"[^>]*aria-selected="true"/
    );
    expect(markup).toContain('role="tabpanel"');
    expect(markup).not.toContain('data-testid="workflow-step-label"');
  });

  test('result tab renders the other stream result body and its table', () => {
    const {store} = makeStore({
      dock: {open: true, height: 380, inspectorTab: 'result', logOpen: false},
      selectedStepIds: ['source']
    });
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toMatch(/data-testid="workflow-inspector-tab-result"[^>]*aria-selected="true"/);
    expect(markup).toContain('data-testid="workflow-result"');
    expect(markup).toContain('data-testid="workflow-result-table-data"');
    expect(markup).toContain('point_id');
    expect(markup).toContain('p0');
    expect(markup).toContain('This result is from an earlier configuration.');
    expect(markup).not.toContain('data-testid="workflow-step-label"');
  });

  test('renders every parameter kind, help and column suggestions without generic file loading', () => {
    const operation = makeKindsOperation();
    const {store} = makeStore({}, replaceOperation(operation));
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {store, stepId: 'scale'})
    );
    for (const spec of operation.parameters.filter(spec => spec.name !== 'hidden')) {
      expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.parameter(spec.name)}"`);
    }
    expect(markup).not.toContain('data-testid="workflow-parameter-hidden"');
    expect(markup).toContain('inputMode="decimal"');
    expect(markup).toContain('role="radiogroup"');
    expect(markup).toContain('aria-label="Field enabled"');
    expect(markup).toContain('type="checkbox"');
    expect(markup).toContain('rows="6"');
    expect(markup).toContain('rows="3"');
    expect(markup).toContain('2 / 100 characters');
    expect(markup).not.toContain('type="file"');
    expect(markup).not.toContain('Load file');
    expect(markup).toContain('Cities · 1,204 rows · resident');
    expect(markup).toContain('<datalist');
    expect(markup).toContain('<option value="value"');
    expect(markup).toContain('aria-label="Insert column value into Field fields"');
    expect(markup).toContain('aria-label="Insert column value into Field expression"');
    expect(markup).toContain('Functions</summary>');
    expect(markup).toContain('coalesce');
    expect(markup).toContain('CRS units');
    expect(markup).toContain('0 to 100,000');
    expect(markup).not.toContain('workflow-parameter-promote-geojson');
    expect(markup).not.toContain('workflow-parameter-promote-expression');
  });

  test('keeps a missing dataset visible by recorded label and supplies run hints', () => {
    const document = makeFixtureDocument();
    document.steps[1].parameters = {
      datasetId: {from: 'static', value: 'missing'},
      datasetLabel: {from: 'static', value: 'Old cities'}
    };
    const {store} = makeStore({document, runner: null}, replaceOperation(makeKindsOperation()));
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {store, stepId: 'scale'})
    );
    expect(markup).toContain('Missing dataset &quot;Old cities&quot;');
    expect(markup).toContain('value="missing" selected=""');
    expect(markup).toContain('Run Fixture source to list its columns');
  });

  test('parameter form gracefully handles absent documents, steps and operations', () => {
    expect(
      renderToStaticMarkup(
        createElement(WorkbenchParameterForm, {
          store: makeStore({document: null}).store,
          stepId: 'scale'
        })
      )
    ).toBe('');
    expect(
      renderToStaticMarkup(
        createElement(WorkbenchParameterForm, {store: makeStore().store, stepId: 'missing'})
      )
    ).toBe('');
    const document = makeFixtureDocument();
    document.steps[1].operationId = 'missing';
    expect(
      renderToStaticMarkup(
        createElement(WorkbenchParameterForm, {store: makeStore({document}).store, stepId: 'scale'})
      )
    ).toBe('');
  });
});

describe('inspector review regressions', () => {
  const textSpec: WorkflowParameterSpec = {
    name: 'text',
    label: 'GeoJSON',
    kind: 'text',
    defaultValue: '',
    maximumLength: 100
  };

  test.each([
    'value',
    'binding'
  ] as const)('F1: external %s changes invalidate a pending file read', async changed => {
    let value = 'initial';
    let bindingIdentity = 'static';
    const commits: unknown[] = [];
    const harness = makeHookHarness(WorkbenchParameterValue);
    const render = () =>
      harness.render({
        spec: textSpec,
        operationId: 'source-geojson',
        value,
        bindingIdentity,
        testId: 'probe',
        onCommit: next => {
          commits.push(next);
          value = String(next);
        }
      });
    let tree = render();
    const deferred = makeDeferredText();
    const loading = invokeProbeHandler(
      findProbeElement(tree, element => element.props['type'] === 'file'),
      'onChange',
      makeFileEvent(deferred.promise)
    );
    if (changed === 'value') value = 'undo value';
    else bindingIdentity = 'input:shared';
    tree = render();
    expect(findProbeControl(tree, 'probe').props['value']).toBe(value);
    deferred.resolve('loaded');
    await loading;
    tree = render();
    expect(commits).toEqual([]);
    expect(value).toBe(changed === 'value' ? 'undo value' : 'initial');
    expect(findProbeControl(tree, 'probe').props['value']).toBe(value);
    harness.unmount();
  });

  test('F2: a repaired text draft commits on blur after undo preserved an invalid draft', () => {
    let value = 'initial';
    const commits: unknown[] = [];
    const harness = makeHookHarness(WorkbenchParameterValue);
    const render = () =>
      harness.render({
        spec: textSpec,
        operationId: 'source-geojson',
        value,
        testId: 'probe',
        onCommit: next => {
          commits.push(next);
          value = String(next);
        }
      });
    let tree = render();
    invokeProbeHandler(findProbeControl(tree, 'probe'), 'onFocus');
    invokeProbeHandler(findProbeControl(tree, 'probe'), 'onChange', {
      target: {value: 'x'.repeat(101)}
    });
    tree = render();
    expect(findProbeControl(tree, 'probe').props['aria-invalid']).toBe(true);
    value = 'undo value';
    tree = render();
    expect(findProbeControl(tree, 'probe').props['value']).toBe('x'.repeat(101));
    invokeProbeHandler(findProbeControl(tree, 'probe'), 'onChange', {target: {value: 'fixed'}});
    tree = render();
    expect(findProbeControl(tree, 'probe').props['aria-invalid']).toBe(false);
    expect(commits).toEqual([]);
    invokeProbeHandler(findProbeControl(tree, 'probe'), 'onBlur');
    tree = render();
    expect(commits).toEqual(['fixed']);
    expect(value).toBe('fixed');
    expect(findProbeControl(tree, 'probe').props['value']).toBe('fixed');
    harness.unmount();
  });

  test('F3: promoted controls refuse tighter workflow input limits inline with a real store', async () => {
    const document = makeFixtureDocument();
    document.inputs = [
      {
        id: 'shared-factor',
        label: 'Shared factor',
        kind: 'number',
        minimum: 0,
        maximum: 10,
        value: 3
      }
    ];
    document.steps[1].parameters['factor'] = {from: 'input', inputId: 'shared-factor'};
    const store = createWorkbenchStore({host: makeFixtureHost(), registry: makeFixtureRegistry()});
    try {
      store.addWorkflow(document);
      const probe = makeParameterProbe(store, 'factor');
      let tree = probe.render();
      invokeProbeHandler(
        findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('factor')),
        'onChange',
        {target: {value: '20'}}
      );
      tree = probe.render();
      expect(store.getState().document?.inputs[0].value).toBe(3);
      expect(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('factor')).props['value']).toBe(
        '20'
      );
      expect(
        findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('factor')).props['aria-invalid']
      ).toBe(true);
      expect(renderToStaticMarkup(tree!)).toContain('Factor must be between 0 and 10; got 20.');
      expect(store.getState().notice).toBeNull();
      invokeProbeHandler(
        findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('factor')),
        'onChange',
        {target: {value: '10'}}
      );
      tree = probe.render();
      expect(store.getState().document?.inputs[0].value).toBe(10);
      expect(
        findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('factor')).props['aria-invalid']
      ).toBe(false);
    } finally {
      await store.clear();
    }
  });

  test.each([
    'Disable',
    'Enable',
    'Duplicate',
    'Delete'
  ])('F4: %s on different selections has separate undo entries', action => {
    const document = makeFixtureDocument();
    document.steps.push({...document.steps[0], id: 'source-two'});
    if (action === 'Enable')
      document.steps = document.steps.map(step => ({...step, disabled: true}));
    const {store, calls} = makeStore({document, selectedStepIds: ['source', 'scale']});
    const harness = makeHookHarness(WorkbenchInspector);
    const apply = () => {
      const tree = harness.render({store});
      const button = findProbeElement(tree, element => element.props['children'] === action);
      invokeProbeHandler(button, 'onClick');
    };
    apply();
    const firstDocument = store.getState().document;
    expect(firstDocument).not.toEqual(document);
    store.select(['source-two', 'count']);
    apply();
    expect(store.getState().document).not.toEqual(firstDocument);
    store.undo();
    expect(store.getState().document).toEqual(firstDocument);
    expect(calls.filter(call => call[0] === 'edit').map(call => call[1])).toEqual([
      `${action.toLowerCase()}-selection:scale,source`,
      `${action.toLowerCase()}-selection:count,source-two`
    ]);
    if (action === 'Duplicate') expect(calls.filter(call => call[0] === 'select')).toHaveLength(3);
  });

  test('F5: the real dataset source retains its hidden recorded dataset label', () => {
    const source = getWorkflowSourceOperations().find(
      operation => operation.id === 'source-dataset'
    )!;
    const document = makeFixtureDocument();
    document.steps[0] = {
      ...document.steps[0],
      operationId: source.id,
      parameters: {
        dataset: {from: 'static', value: 'old-id'},
        datasetLabel: {from: 'static', value: 'Saved cities'}
      }
    };
    expect(
      getWorkbenchParameterRows(document, document.steps[0], source).some(
        row => row.spec.name === 'datasetLabel'
      )
    ).toBe(false);
    const {store} = makeStore({document}, replaceOperation(source));
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {store, stepId: 'source'})
    );
    expect(markup).toContain('Missing dataset &quot;Saved cities&quot;');
    expect(markup).not.toContain('Missing dataset &quot;old-id&quot;');
    expect(markup).not.toContain('data-testid="workflow-parameter-datasetLabel"');
  });
});

describe('inspector edit and asynchronous control coverage', () => {
  test('the control spec intersects shared input ranges, integer requirements and enum choices', () => {
    const numberSpec: WorkflowParameterSpec = {
      name: 'number',
      label: 'Number',
      kind: 'number',
      minimum: 0,
      maximum: 100,
      defaultValue: 3
    };
    const input = {
      id: 'shared',
      label: 'Shared',
      kind: 'integer' as const,
      minimum: 2,
      maximum: 10,
      value: 3
    };
    const controlSpec = getWorkbenchParameterControlSpec(numberSpec, input);
    expect(controlSpec).toEqual({...numberSpec, kind: 'integer', minimum: 2, maximum: 10});
    expect(getWorkflowParameterProblem(controlSpec, '2.5')).toBe('must be a whole number; got 2.5');
    expect(
      getWorkbenchParameterControlSpec(numberSpec, {...input, minimum: -100, maximum: 1000})
    ).toEqual({...numberSpec, kind: 'integer'});
    expect(getWorkbenchParameterControlSpec(numberSpec)).toBe(numberSpec);
    const enumSpec: WorkflowParameterSpec = {
      name: 'method',
      label: 'Method',
      kind: 'enum',
      options: ['a', 'b', 'c'],
      defaultValue: 'a'
    };
    expect(
      getWorkbenchParameterControlSpec(enumSpec, {
        id: 'method',
        label: 'Method',
        kind: 'enum',
        options: ['b', 'c', 'd'],
        value: 'b'
      })
    ).toEqual({...enumSpec, options: ['b', 'c']});
  });

  test('actual parameter callbacks apply edits and expose typed refusals without changing history', () => {
    const document = makeFixtureDocument();
    document.inputs = [
      {id: 'shared', label: 'Shared factor', kind: 'number', maximum: 10, value: 3}
    ];
    document.steps[1].parameters['factor'] = {from: 'input', inputId: 'shared'};
    const {store, refusals, history} = makeStore({document});
    const probe = makeParameterProbe(store, 'factor');
    let form = probe.renderForm();
    let parameter = findProbeElement(
      form,
      element =>
        element.type === WorkbenchParameterValue &&
        element.props['testId'] === WORKBENCH_TEST_IDS.parameter('factor')
    );
    invokeProbeHandler(parameter, 'onCommit', 8);
    expect(store.getState().document?.inputs[0].value).toBe(8);
    const validDocument = store.getState().document;
    form = probe.renderForm();
    parameter = findProbeElement(
      form,
      element =>
        element.type === WorkbenchParameterValue &&
        element.props['testId'] === WORKBENCH_TEST_IDS.parameter('factor')
    );
    invokeProbeHandler(parameter, 'onCommit', 20);
    expect(store.getState().document).toBe(validDocument);
    expect(refusals).toHaveLength(1);
    expect(refusals[0]).toBeInstanceOf(WorkflowError);
    expect((refusals[0] as WorkflowError).code).toBe('parameter-out-of-range');
    expect(store.getState().notice).toEqual({
      kind: 'error',
      message: 'Shared factor: Value must be at most 10; got 20.'
    });
    expect(history.canUndo).toBe(true);
    store.undo();
    expect(store.getState().document).toBe(document);
  });

  test('numeric edits keep partial input, replace it on ordinary undo, and retain invalid focused drafts', () => {
    const {store} = makeStore();
    const probe = makeParameterProbe(store, 'factor');
    let tree = probe.render();
    const control = () => findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('factor'));
    invokeProbeHandler(control(), 'onFocus');
    invokeProbeHandler(control(), 'onChange', {target: {value: '4.'}});
    tree = probe.render();
    expect(control().props['value']).toBe('4.');
    expect(store.getState().document?.steps[1].parameters['factor']).toEqual({
      from: 'static',
      value: 4
    });
    store.undo();
    tree = probe.render();
    expect(control().props['value']).toBe('3');
    invokeProbeHandler(control(), 'onChange', {target: {value: '5'}});
    tree = probe.render();
    invokeProbeHandler(control(), 'onChange', {target: {value: '-'}});
    tree = probe.render();
    expect(control().props['aria-invalid']).toBe(true);
    expect(store.getState().document?.steps[1].parameters['factor']).toEqual({
      from: 'static',
      value: 5
    });
    store.undo();
    tree = probe.render();
    expect(control().props['value']).toBe('-');
    invokeProbeHandler(control(), 'onBlur');
    tree = probe.render();
    expect(control().props['value']).toBe('3');
    expect(control().props['aria-invalid']).toBe(false);
  });

  test.each([
    'Enter',
    'blur'
  ])('text commits through %s and undo replaces the committed draft', action => {
    const operation = makeKindsOperation();
    const {store} = makeStore({}, replaceOperation(operation));
    const probe = makeParameterProbe(store, 'geojson');
    let tree = probe.render();
    const control = () => findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('geojson'));
    invokeProbeHandler(control(), 'onFocus');
    invokeProbeHandler(control(), 'onChange', {target: {value: '{"ok":true}'}});
    tree = probe.render();
    expect(store.getState().document?.steps[1].parameters['geojson']).toBeUndefined();
    const preventions: string[] = [];
    if (action === 'Enter')
      invokeProbeHandler(control(), 'onKeyDown', {
        key: 'Enter',
        ctrlKey: true,
        metaKey: false,
        preventDefault: () => preventions.push('Enter')
      });
    else invokeProbeHandler(control(), 'onBlur');
    tree = probe.render();
    expect(store.getState().document?.steps[1].parameters['geojson']).toEqual({
      from: 'static',
      value: '{"ok":true}'
    });
    expect(preventions).toEqual(action === 'Enter' ? ['Enter'] : []);
    store.undo();
    tree = probe.render();
    expect(control().props['value']).toBe('{}');
  });

  test('selecting a real source dataset fills hidden label and geometry in one undoable edit', () => {
    const source = getWorkflowSourceOperations().find(
      operation => operation.id === 'source-dataset'
    )!;
    const document = makeFixtureDocument();
    document.steps[0] = {...document.steps[0], operationId: source.id};
    const {store, calls} = makeStore({document}, replaceOperation(source));
    const probe = makeParameterProbe(store, 'dataset', 'source');
    const tree = probe.render();
    invokeProbeHandler(
      findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('dataset')),
      'onChange',
      {target: {value: 'cities'}}
    );
    expect(store.getState().document?.steps[0].parameters).toEqual({
      dataset: {from: 'static', value: 'cities'},
      datasetLabel: {from: 'static', value: 'Cities'},
      geometryMode: {from: 'static', value: 'geojson-field'},
      geometryField: {from: 'static', value: 'shape'}
    });
    expect(calls.filter(call => call[0] === 'edit').map(call => call[1])).toEqual([
      'parameter:source:dataset'
    ]);
    store.undo();
    expect(store.getState().document).toBe(document);
  });

  test('promote, demote and reset execute their document edits', () => {
    const {store} = makeStore();
    const probe = makeParameterProbe(store, 'factor');
    const promote = () =>
      findProbeControl(probe.renderForm(), WORKBENCH_TEST_IDS.parameterPromote('factor'));
    invokeProbeHandler(promote(), 'onClick');
    const input = store.getState().document?.inputs[0];
    expect(input).toMatchObject({
      label: 'Factor',
      value: 3,
      kind: 'number',
      minimum: 0,
      maximum: 100
    });
    expect(store.getState().document?.steps[1].parameters['factor']).toEqual({
      from: 'input',
      inputId: input?.id
    });
    invokeProbeHandler(promote(), 'onClick');
    expect(store.getState().document?.steps[1].parameters['factor']).toEqual({
      from: 'static',
      value: 3
    });
    const reset = findProbeElement(
      probe.render(),
      element => element.props['aria-label'] === 'Reset Factor'
    );
    invokeProbeHandler(reset, 'onClick');
    expect(store.getState().document?.steps[1].parameters['factor']).toBeUndefined();
    expect(
      findProbeControl(probe.render(), WORKBENCH_TEST_IDS.parameter('factor')).props['value']
    ).toBe('2');
  });

  test('asynchronous file reads accept the character limit, refuse one over and retain the last value', async () => {
    const {store} = makeFileStore();
    const probe = makeParameterProbe(store, 'text');
    let tree = probe.render();
    const file = () => findProbeElement(tree, element => element.props['type'] === 'file');
    await invokeProbeHandler(
      file(),
      'onChange',
      makeFileEvent(Promise.resolve('é'.repeat(100)), 200)
    );
    tree = probe.render();
    expect(store.getState().document?.steps[1].parameters['text']).toEqual({
      from: 'static',
      value: 'é'.repeat(100)
    });
    await invokeProbeHandler(
      file(),
      'onChange',
      makeFileEvent(Promise.resolve('x'.repeat(101)), 101)
    );
    tree = probe.render();
    expect(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('text')).props['value']).toBe(
      'é'.repeat(100)
    );
    expect(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('text')).props['aria-invalid']).toBe(
      true
    );
    expect(renderToStaticMarkup(tree!)).toContain(
      'Fixture scale: Field geojson must be at most 100 characters; got 101.'
    );
    expect(store.getState().document?.steps[1].parameters['text']).toEqual({
      from: 'static',
      value: 'é'.repeat(100)
    });
  });

  test('later file selection wins when asynchronous reads complete in reverse order', async () => {
    const {store} = makeFileStore();
    const probe = makeParameterProbe(store, 'text');
    let tree = probe.render();
    const earlier = makeDeferredText();
    const later = makeDeferredText();
    const firstRead = invokeProbeHandler(
      findProbeElement(tree, element => element.props['type'] === 'file'),
      'onChange',
      makeFileEvent(earlier.promise)
    );
    tree = probe.render();
    const secondRead = invokeProbeHandler(
      findProbeElement(tree, element => element.props['type'] === 'file'),
      'onChange',
      makeFileEvent(later.promise)
    );
    later.resolve('later');
    await secondRead;
    tree = probe.render();
    earlier.resolve('earlier');
    await firstRead;
    tree = probe.render();
    expect(store.getState().document?.steps[1].parameters['text']).toEqual({
      from: 'static',
      value: 'later'
    });
    expect(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('text')).props['value']).toBe(
      'later'
    );
  });

  test('an asynchronous read failure is visible and unmount prevents a late commit', async () => {
    let value = 'initial';
    const commits: unknown[] = [];
    const harness = makeHookHarness(WorkbenchParameterValue);
    const render = () =>
      harness.render({
        spec: {name: 'text', label: 'GeoJSON', kind: 'text', defaultValue: ''},
        operationId: 'source-geojson',
        value,
        testId: 'probe',
        onCommit: next => {
          commits.push(next);
          value = String(next);
        }
      });
    let tree = render();
    const failed = makeDeferredText();
    const read = invokeProbeHandler(
      findProbeElement(tree, element => element.props['type'] === 'file'),
      'onChange',
      makeFileEvent(failed.promise)
    );
    failed.reject(new Error('disk unavailable'));
    await read;
    tree = render();
    expect(renderToStaticMarkup(tree!)).toContain(
      'GeoJSON file could not be read: disk unavailable.'
    );
    expect(commits).toEqual([]);
    const late = makeDeferredText();
    const loading = invokeProbeHandler(
      findProbeElement(tree, element => element.props['type'] === 'file'),
      'onChange',
      makeFileEvent(late.promise)
    );
    harness.unmount();
    late.resolve('late');
    await loading;
    expect(commits).toEqual([]);
    expect(value).toBe('initial');
  });

  test('pending file completion cannot overwrite a real store undo', async () => {
    const {store} = makeFileStore();
    store.edit('previous-text', document =>
      setWorkflowParameter(document, 'scale', 'text', {from: 'static', value: 'before upload'})
    );
    const probe = makeParameterProbe(store, 'text');
    let tree = probe.render();
    const deferred = makeDeferredText();
    const loading = invokeProbeHandler(
      findProbeElement(tree, element => element.props['type'] === 'file'),
      'onChange',
      makeFileEvent(deferred.promise)
    );
    store.undo();
    tree = probe.render();
    expect(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('text')).props['value']).toBe('{}');
    deferred.resolve('loaded');
    await loading;
    tree = probe.render();
    expect(store.getState().document?.steps[1].parameters['text']).toBeUndefined();
    expect(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('text')).props['value']).toBe('{}');
  });
});

function getInspectorDetailsMarkup(markup: string, summary: string): string {
  const details = [...markup.matchAll(/<details\b[^>]*>[\s\S]*?<\/details>/g)].find(match =>
    match[0].includes(`>${summary}</summary>`)
  );
  if (!details) throw new Error(`Disclosure "${summary}" was not found.`);
  return details[0];
}

describe('compact inspector layout', () => {
  test('I1 and I4: split issue severities, remove only a matching step prefix and localize capacity counts', () => {
    const issues: WorkflowIssue[] = [
      {
        severity: 'info',
        code: 'capacity-may-exceed',
        message: 'Custom scale: Points accepts at most 1000 rows; Data allows up to 100000 rows.'
      },
      {
        severity: 'error',
        code: 'parameter-out-of-range',
        message: 'Custom scale: Factor must be at least 1.'
      },
      {severity: 'warning', code: 'unverified-port', message: 'Another step: Verify the geometry.'},
      {
        severity: 'info',
        code: 'capacity-may-exceed',
        message: 'Custom scale: source-id-100000 has 1,000 rows and at most 100000 vertices.'
      }
    ];
    const {problems, notes} = getWorkbenchInspectorIssues(issues, 'Custom scale');
    expect(problems).toEqual([{...issues[1], message: 'Factor must be at least 1.'}, issues[2]]);
    expect(notes).toEqual([
      {...issues[0], message: 'Points accepts at most 1,000 rows; Data allows up to 100,000 rows.'},
      {...issues[3], message: 'source-id-100000 has 1,000 rows and at most 100,000 vertices.'}
    ]);
    expect(issues[0].message).toBe(
      'Custom scale: Points accepts at most 1000 rows; Data allows up to 100000 rows.'
    );
    expect(
      getWorkbenchInspectorIssues(
        [{severity: 'error', code: 'input-required', message: 'Custom scale needs an input.'}],
        'Custom scale'
      ).problems[0].message
    ).toBe('needs an input.');
    expect(
      getWorkbenchInspectorIssues(
        [
          {
            severity: 'error',
            code: 'input-required',
            message: 'Custom scaled points need an input.'
          }
        ],
        'Custom scale'
      ).problems[0].message
    ).toBe('Custom scaled points need an input.');
  });

  test('I1 and I2: problems precede parameters, with inputs, collapsed run options and collapsed notes after them', () => {
    const document = makeFixtureDocument();
    document.steps[1] = {...document.steps[1], label: 'Custom scale', display: []};
    const runner = makeRunner();
    runner.steps['scale'].issues = [
      {
        severity: 'info',
        code: 'capacity-may-exceed',
        message: 'Custom scale: Points accepts at most 1000 rows; Data allows up to 100000 rows.'
      },
      {
        severity: 'error',
        code: 'parameter-out-of-range',
        message: 'Custom scale: Factor is out of range.'
      },
      {
        severity: 'warning',
        code: 'unverified-port',
        message: 'Custom scale: Verify Points before running.'
      },
      {
        severity: 'info',
        code: 'capacity-may-exceed',
        message: 'Custom scale: Output allows 100000 vertices.'
      }
    ];
    const {store} = makeStore({document, runner});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    const list = markup.match(/<ul data-testid="workflow-step-issues">([\s\S]*?)<\/ul>/)?.[1];
    expect(list).toBe(
      '<li>error: Factor is out of range.</li><li>warning: Verify Points before running.</li>'
    );
    const order = [
      'data-testid="workflow-step-label"',
      'data-testid="workflow-step-issues"',
      '<h3>Parameters</h3>',
      '<h3>Inputs</h3>',
      '>Run options</summary>',
      '>Notes (2)</summary>'
    ].map(fragment => markup.indexOf(fragment));
    expect(order.every(position => position >= 0)).toBe(true);
    expect(order).toEqual([...order].sort((first, second) => first - second));
    const runOptions = getInspectorDetailsMarkup(markup, 'Run options');
    expect(runOptions).not.toMatch(/^<details\b[^>]*\bopen=/);
    const notes = getInspectorDetailsMarkup(markup, 'Notes (2)');
    expect(notes).not.toMatch(/^<details\b[^>]*\bopen=/);
    expect(notes).toContain(
      '<li>info: Points accepts at most 1,000 rows; Data allows up to 100,000 rows.</li>'
    );
    expect(notes).toContain('<li>info: Output allows 100,000 vertices.</li>');
    expect(notes).not.toContain('Custom scale:');
    expect(runOptions).toContain('aria-label="Note"');
  });

  test.each([
    false,
    true
  ])('I1: no empty problem list is rendered with info-only issues=%s', withNotes => {
    const runner = makeRunner();
    runner.steps['scale'].issues = withNotes
      ? [
          {
            severity: 'info',
            code: 'capacity-may-exceed',
            message: 'Fixture scale: Points allow 1000 rows.'
          }
        ]
      : [];
    const {store} = makeStore({runner});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).not.toContain('data-testid="workflow-step-issues"');
    if (withNotes) expect(getInspectorDetailsMarkup(markup, 'Notes (1)')).toContain('1,000 rows');
    else expect(markup).not.toContain('>Notes (');
  });

  const runOptions: Array<{label: string; patch: Partial<WorkflowStep>; open: boolean}> = [
    {label: 'all defaults', patch: {}, open: false},
    {
      label: 'explicit defaults',
      patch: {onFallback: 'warn', disabled: false, display: [], note: ''},
      open: false
    },
    {label: 'CPU override', patch: {engine: 'cpu'}, open: true},
    {label: 'automatic engine override', patch: {engine: 'auto'}, open: true},
    {label: 'stop on fallback', patch: {onFallback: 'stop'}, open: true},
    {label: 'displayed output', patch: {display: ['scaled']}, open: true},
    {label: 'disabled', patch: {disabled: true}, open: true},
    {label: 'note', patch: {note: 'Check distances.'}, open: true}
  ];
  test.each(runOptions)('I2: run options disclosure reflects $label', ({patch, open}) => {
    const document = makeFixtureDocument();
    const step = {...document.steps[1], display: undefined, ...patch};
    document.steps[1] = step;
    expect(hasWorkbenchRunOptionOverrides(step)).toBe(open);
    const {store} = makeStore({document});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    const details = getInspectorDetailsMarkup(markup, 'Run options');
    expect(/^<details\b[^>]*\bopen=/.test(details)).toBe(open);
  });

  test('I3 and I4: hints combine localized ranges and differing defaults on one line', () => {
    const spec: WorkflowParameterSpec = {
      name: 'count',
      label: 'Count',
      kind: 'integer',
      minimum: 1,
      maximum: 1000,
      defaultValue: 10
    };
    expect(formatWorkbenchParameterHint(spec, 10)).toBe('1 to 1,000 · default 10');
    expect(formatWorkbenchParameterHint(spec)).toBe('1 to 1,000');
    expect(formatWorkbenchParameterHint({...spec, maximum: undefined}, 100000)).toBe(
      'At least 1 · default 100,000'
    );
    expect(formatWorkbenchParameterHint({...spec, minimum: undefined}, [1000, 100000])).toBe(
      'At most 1,000 · default 1,000, 100,000'
    );
    expect(
      formatWorkbenchParameterHint({...spec, minimum: undefined, maximum: undefined}, false)
    ).toBe('default false');
    expect(formatWorkbenchParameterHint({...spec, minimum: undefined, maximum: undefined})).toBe(
      ''
    );
  });

  test.each([
    20, 10
  ])('I3: range and Reset share one hint row; default is shown only when different (value %s)', value => {
    const document = makeFixtureDocument();
    const original = makeFixtureRegistry().get('fixture-scale')!;
    const operation = {
      ...original,
      parameters: original.parameters.map(spec =>
        spec.name === 'factor' ? {...spec, minimum: 1, maximum: 1000, defaultValue: 10} : spec
      )
    };
    document.steps[1].parameters['factor'] = {from: 'static', value};
    const {store} = makeStore({document}, replaceOperation(operation));
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {store, stepId: 'scale'})
    );
    const hintRows = [...markup.matchAll(/<div id="[^"]+-hint"[^>]*>([\s\S]*?)<\/div>/g)];
    const factorHint = hintRows.find(match => match[1].includes('aria-label="Reset Factor"'))?.[1];
    expect(factorHint).toBeDefined();
    expect(factorHint).toContain(
      value === 20 ? '>1 to 1,000 · default 10</span>' : '>1 to 1,000</span>'
    );
    expect(factorHint).toContain('>Reset</button>');
    expect(factorHint?.indexOf('</span>')).toBeLessThan(
      factorHint?.indexOf('aria-label="Reset Factor"') ?? -1
    );
    if (value === 10) expect(factorHint).not.toContain('default 10');
    expect(markup).not.toContain('Default 10');
  });

  test('I3: fields without a hint or reset do not render an empty hint row', () => {
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterValue, {
        spec: {name: 'name', label: 'Name', kind: 'string', defaultValue: ''},
        value: '',
        testId: 'probe',
        onCommit: () => {}
      })
    );
    expect(markup).not.toMatch(/id="[^"]+-hint"/);
    expect(markup).not.toContain('aria-describedby=');
  });
});

describe('second-wave inspector controls', () => {
  test.each([
    'error',
    'warning'
  ] as const)('suggested repairs render under %s issues and apply as undoable edits', severity => {
    const document = makeFixtureDocument();
    document.steps[1].parameters['factor'] = {from: 'static', value: 150};
    const runner = makeRunner();
    runner.steps['scale'].issues = [
      {
        severity,
        code: 'parameter-out-of-range',
        stepId: 'scale',
        parameter: 'factor',
        message: 'Fixture scale: Factor must be between 0 and 100; got 150.'
      }
    ];
    const {store, calls} = makeStore({document, runner});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain(
      `data-testid="${WORKBENCH_TEST_IDS.suggestion('set-bound:scale:factor')}"`
    );
    expect(markup).toContain('title="Set the parameter to the nearest allowed bound."');
    expect(markup).toContain('Set &quot;Factor&quot; to 100');
    const probe = makeStepInspectorProbe(store);
    const tree = probe.render();
    const button = findProbeControl(tree, WORKBENCH_TEST_IDS.suggestion('set-bound:scale:factor'));
    invokeProbeHandler(button, 'onClick');
    expect(store.getState().document?.steps[1].parameters['factor']).toEqual({
      from: 'static',
      value: 100
    });
    expect(calls.filter(call => call[0] === 'edit').map(call => call[1])).toEqual([
      'suggestion:set-bound:scale:factor'
    ]);
    store.undo();
    expect(store.getState().document?.steps[1].parameters['factor']).toEqual({
      from: 'static',
      value: 150
    });
  });

  test('repair provider failures offer no buttons and informational issues are excluded', () => {
    const document = makeFixtureDocument();
    const issue: WorkflowIssue = {
      severity: 'error',
      code: 'input-required',
      stepId: 'scale',
      port: 'points',
      message: 'Fixture scale: Points needs an input.'
    };
    let lookups = 0;
    const registry: WorkflowOperationRegistry = {
      list: () => [],
      get: () => {
        lookups++;
        throw new Error('Registry unavailable');
      }
    };
    expect(
      getWorkbenchIssueSuggestions(document, registry, null, [
        issue,
        {...issue, severity: 'warning'}
      ])
    ).toEqual([[], []]);
    expect(lookups).toBe(2);
    expect(
      getWorkbenchIssueSuggestions(document, registry, null, [{...issue, severity: 'info'}])
    ).toEqual([]);
    expect(lookups).toBe(2);
  });

  test('display columns use declared identities before a run and the latest non-geometry schema afterward', () => {
    const output = makeFixtureRegistry().get('fixture-scale')!.outputs[0];
    expect(getWorkbenchDisplayColumns('scale', output, null)).toEqual(['point_id']);
    const runner = makeRunner();
    runner.steps['scale'] = {
      ...runner.steps['scale'],
      status: 'stale',
      result: {...makeResult(), outputs: {scaled: makeFixturePoints(2)}}
    };
    expect(getWorkbenchDisplayColumns('scale', output, runner)).toEqual(['point_id', 'value']);
    const document = makeFixtureDocument();
    document.steps[1].displayStyle = {
      scaled: {colorColumn: 'value', colorScale: 'quantile', palette: 'diverging'}
    };
    const {store} = makeStore({document, runner});
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.displayColorColumn('scaled')}"`);
    expect(markup).toContain('<option value="value" selected="">value</option>');
    expect(markup).not.toContain('<option value="geometry"');
    expect(markup).toContain('<option value="quantile" selected="">Quantile</option>');
    expect(markup).toContain('<option value="quantize">Equal interval</option>');
    expect(markup).toContain('<option value="ordinal">Categories</option>');
    expect(markup).toContain('<option value="diverging" selected="">Diverging</option>');
    document.steps[1] = {...document.steps[1], display: []};
    expect(hasWorkbenchRunOptionOverrides(document.steps[1])).toBe(true);
    const hidden = renderToStaticMarkup(
      createElement(WorkbenchInspector, {store: makeStore({document, runner}).store})
    );
    expect(getInspectorDetailsMarkup(hidden, 'Run options')).toMatch(/^<details\b[^>]*\bopen=/);
    expect(hidden).not.toContain('workflow-display-color-column-');
  });

  test('display style controls merge changes under one tag, undo together and clear to host defaults', () => {
    const {store, calls} = makeStore();
    const probe = makeStepInspectorProbe(store);
    let tree = probe.render();
    expect(
      findProbeControl(tree, WORKBENCH_TEST_IDS.displayColorColumn('scaled')).props['value']
    ).toBe('');
    invokeProbeHandler(
      findProbeControl(tree, WORKBENCH_TEST_IDS.displayColorColumn('scaled')),
      'onChange',
      {target: {value: 'point_id'}}
    );
    tree = probe.render();
    invokeProbeHandler(
      findProbeControl(tree, WORKBENCH_TEST_IDS.displayColorScale('scaled')),
      'onChange',
      {target: {value: 'quantize'}}
    );
    tree = probe.render();
    invokeProbeHandler(
      findProbeControl(tree, WORKBENCH_TEST_IDS.displayPalette('scaled')),
      'onChange',
      {target: {value: 'categorical'}}
    );
    expect(store.getState().document?.steps[1].displayStyle).toEqual({
      scaled: {colorColumn: 'point_id', colorScale: 'quantize', palette: 'categorical'}
    });
    expect(calls.filter(call => call[0] === 'edit').map(call => call[1])).toEqual(
      Array(3).fill('display-style:scale:scaled')
    );
    store.undo();
    expect(store.getState().document?.steps[1].displayStyle).toBeUndefined();
    store.redo();
    tree = probe.render();
    invokeProbeHandler(
      findProbeControl(tree, WORKBENCH_TEST_IDS.displayColorScale('scaled')),
      'onChange',
      {target: {value: ''}}
    );
    tree = probe.render();
    expect(store.getState().document?.steps[1].displayStyle).toEqual({
      scaled: {colorColumn: 'point_id', palette: 'categorical'}
    });
    invokeProbeHandler(
      findProbeControl(tree, WORKBENCH_TEST_IDS.displayColorColumn('scaled')),
      'onChange',
      {target: {value: ''}}
    );
    tree = probe.render();
    expect(store.getState().document?.steps[1].displayStyle).toBeUndefined();
    expect(renderToStaticMarkup(tree!)).not.toContain('workflow-display-color-scale-scaled');
    expect(renderToStaticMarkup(tree!)).not.toContain('workflow-display-palette-scaled');
  });

  test('SQL uses an eight-row text editor with table help and commits on blur; only GeoJSON offers file loading', () => {
    const operation = getWorkflowSqlOperations()[0];
    const document = makeFixtureDocument();
    document.steps[1] = {...document.steps[1], operationId: operation.id, parameters: {}};
    const {store} = makeStore({document}, replaceOperation(operation));
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {store, stepId: 'scale'})
    );
    expect(markup).toContain('rows="8"');
    expect(markup).toContain('font-family:monospace');
    expect(markup).toContain('Tables: input1, input2, input3');
    expect(markup).not.toContain('Load file');
    expect(markup).not.toContain('type="file"');
    const probe = makeParameterProbe(store, 'query');
    let tree = probe.render();
    invokeProbeHandler(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('query')), 'onChange', {
      target: {value: 'SELECT value FROM input1'}
    });
    tree = probe.render();
    expect(store.getState().document?.steps[1].parameters['query']).toBeUndefined();
    invokeProbeHandler(findProbeControl(tree, WORKBENCH_TEST_IDS.parameter('query')), 'onBlur');
    expect(store.getState().document?.steps[1].parameters['query']).toEqual({
      from: 'static',
      value: 'SELECT value FROM input1'
    });
    const source = getWorkflowSourceOperations().find(
      candidate => candidate.id === 'source-geojson'
    )!;
    document.steps[0] = {...document.steps[0], operationId: source.id};
    const geojson = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {
        store: makeStore({document}, replaceOperation(source)).store,
        stepId: 'source'
      })
    );
    expect(geojson).toContain('rows="6"');
    expect(geojson).toContain('aria-label="Load file for GeoJSON"');
    expect(geojson).toContain('accept=".json,.geojson"');
  });

  test.each([
    {drawings: null, expected: null},
    {
      drawings: [] as WorkflowDrawing[],
      expected: 'No shapes drawn yet. Use the draw tools on the map.'
    },
    {
      drawings: [
        {id: 'polygon', label: 'Polygon 1', kind: 'polygon'},
        {id: 'rectangle', label: 'Rectangle 1', kind: 'polygon'}
      ] as WorkflowDrawing[],
      expected: '2 shapes: Polygon 1, Rectangle 1'
    }
  ])('drawing source lists available shapes with host support: $expected', ({
    drawings,
    expected
  }) => {
    const operation = getWorkflowSourceOperations().find(
      candidate => candidate.id === 'source-drawing'
    )!;
    const document = makeFixtureDocument();
    document.steps[0] = {...document.steps[0], operationId: operation.id};
    const {store} = makeStore({document}, replaceOperation(operation));
    if (drawings) store.host.listDrawings = () => drawings;
    const markup = renderToStaticMarkup(
      createElement(WorkbenchParameterForm, {store, stepId: 'source'})
    );
    if (expected) {
      expect(markup).toContain(expected);
      expect(markup.indexOf('data-testid="workflow-parameter-shapes"')).toBeLessThan(
        markup.indexOf(expected)
      );
    } else expect(markup).not.toContain('Use the draw tools on the map.');
  });
});

describe('source type selector', () => {
  function makeSourceStore() {
    const sources = getWorkflowSourceOperations();
    const fixtures = makeFixtureRegistry();
    const operations = [
      ...sources,
      ...fixtures.list().filter(operation => operation.category !== 'source')
    ];
    const registry: WorkflowOperationRegistry = {
      list: () => operations,
      get: identifier => operations.find(operation => operation.id === identifier)
    };
    const document = makeFixtureDocument();
    document.steps[0] = {
      ...document.steps[0],
      operationId: 'source-geojson',
      parameters: {text: {from: 'static', value: '{"type":"FeatureCollection","features":[]}'}}
    };
    return makeStore({document, selectedStepIds: ['source']}, registry);
  }

  test('source steps offer titled source operations above the step label; other operations do not', () => {
    const {store} = makeSourceStore();
    const markup = renderToStaticMarkup(createElement(WorkbenchInspector, {store}));
    expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.sourceType}"`);
    expect(markup).toContain('aria-label="Source"');
    const select = markup.match(
      /<select[^>]*data-testid="workflow-source-type"[^>]*>([\s\S]*?)<\/select>/
    )?.[1];
    expect(select).toBe(
      '<option value="source-dataset">Dataset</option><option value="source-geojson" selected="">GeoJSON</option><option value="source-drawing">Drawn shapes</option><option value="source-table">Table snapshot</option>'
    );
    expect(markup.indexOf('workflow-source-type')).toBeLessThan(
      markup.indexOf('workflow-step-label')
    );
    store.select(['scale']);
    expect(renderToStaticMarkup(createElement(WorkbenchInspector, {store}))).not.toContain(
      'workflow-source-type'
    );
  });

  test('a sole dataset presets its label and geometry without retaining the old source configuration', () => {
    const operation = getWorkflowSourceOperations().find(
      candidate => candidate.id === 'source-dataset'
    )!;
    const step = makeFixtureDocument().steps[0];
    step.parameters = {
      geometryMode: {from: 'static', value: 'none'},
      geometryField: {from: 'static', value: 'old-field'}
    };
    expect(getWorkbenchSourceParameters(step, operation, DATASETS)).toEqual({
      dataset: 'cities',
      datasetLabel: 'Cities',
      geometryMode: 'geojson-field',
      geometryField: 'shape'
    });
    expect(
      getWorkbenchSourceParameters(step, operation, [
        {
          ...DATASETS[0],
          geometryFields: [],
          pointColumns: {longitude: 'longitude', latitude: 'latitude'}
        }
      ])
    ).toEqual({
      dataset: 'cities',
      datasetLabel: 'Cities',
      geometryMode: 'point-columns',
      longitudeField: 'longitude',
      latitudeField: 'latitude'
    });
    expect(
      getWorkbenchSourceParameters(step, operation, [{...DATASETS[0], geometryFields: []}])
    ).toEqual({dataset: 'cities', datasetLabel: 'Cities', geometryMode: 'none'});
    expect(step.parameters).toEqual({
      geometryMode: {from: 'static', value: 'none'},
      geometryField: {from: 'static', value: 'old-field'}
    });
  });

  test('zero or multiple datasets and other source types leave parameters at their defaults', () => {
    const operations = getWorkflowSourceOperations();
    const dataset = operations.find(operation => operation.id === 'source-dataset')!;
    const step = makeFixtureDocument().steps[0];
    expect(getWorkbenchSourceParameters(step, dataset, [])).toEqual({});
    expect(
      getWorkbenchSourceParameters(step, dataset, [...DATASETS, {...DATASETS[0], id: 'other'}])
    ).toEqual({});
    expect(
      getWorkbenchSourceParameters(
        step,
        operations.find(operation => operation.id === 'source-geojson')!,
        DATASETS
      )
    ).toEqual({});
  });

  test('selecting the current source uses the step tag without resetting the document or adding undo history', () => {
    const {store, calls} = makeSourceStore();
    const original = store.getState().document;
    const probe = makeStepInspectorProbe(store);
    const tree = probe.render();
    invokeProbeHandler(findProbeControl(tree, WORKBENCH_TEST_IDS.sourceType), 'onChange', {
      target: {value: 'source-geojson'}
    });
    expect(calls.filter(call => call[0] === 'edit').map(call => call[1])).toEqual([
      'source-type:source'
    ]);
    expect(store.getState().document).toBe(original);
    expect(store.getState().canUndo).toBe(false);
  });

  test.each([
    'source-dataset',
    'source-drawing'
  ])('switching to %s preserves downstream connections and supports undo', operationId => {
    const {store, calls, refusals} = makeSourceStore();
    const original = store.getState().document!;
    const probe = makeStepInspectorProbe(store);
    invokeProbeHandler(
      findProbeControl(probe.render(), WORKBENCH_TEST_IDS.sourceType),
      'onChange',
      {target: {value: operationId}}
    );
    const changed = store.getState().document!;
    expect(refusals).toEqual([]);
    expect(changed.steps[0].id).toBe('source');
    expect(changed.steps[0].operationId).toBe(operationId);
    expect(changed.steps[0].position).toEqual(original.steps[0].position);
    expect(changed.steps[1]).toBe(original.steps[1]);
    expect(changed.steps[1].inputs['points']).toEqual({stepId: 'source', output: 'data'});
    expect(changed.steps[0].parameters).toEqual(
      operationId === 'source-dataset'
        ? {
            dataset: {from: 'static', value: 'cities'},
            datasetLabel: {from: 'static', value: 'Cities'},
            geometryMode: {from: 'static', value: 'geojson-field'},
            geometryField: {from: 'static', value: 'shape'}
          }
        : {}
    );
    expect(calls.filter(call => call[0] === 'edit').map(call => call[1])).toEqual([
      'source-type:source'
    ]);
    expect(findProbeControl(probe.render(), WORKBENCH_TEST_IDS.sourceType).props['value']).toBe(
      operationId
    );
    store.undo();
    expect(store.getState().document).toBe(original);
    expect(findProbeControl(probe.render(), WORKBENCH_TEST_IDS.sourceType).props['value']).toBe(
      'source-geojson'
    );
  });
});
