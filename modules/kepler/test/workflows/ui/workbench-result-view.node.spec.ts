// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {describe, expect, test, vi} from 'vitest';
import type {ReactElement, ReactNode} from 'react';
import {
  getWorkflowColumnNames,
  makeWorkflowDataFromFeatures,
  makeWorkflowGeoJSONText
} from '../../../src/workflows/workflow-data';
import type {
  WorkflowData,
  WorkflowDocument,
  WorkflowStepResult,
  WorkflowStepState
} from '../../../src/workflows/workflow-types';
import {
  makeFixtureDocument,
  makeFixtureHost,
  makeFixturePoints,
  makeFixtureRegistry
} from '../workflow-fixtures';
import {WORKBENCH_TEST_IDS} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-contract';
import {WorkbenchResultView} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-result-view';
import {
  describeWorkbenchGeometry,
  formatWorkbenchCell,
  formatWorkbenchDuration,
  formatWorkbenchSummaryKey,
  getWorkbenchPreviewPage,
  getWorkbenchColumnDisplayStyle,
  makeWorkbenchDownloadName,
  makeWorkbenchHistogram
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-result-model';
import type {
  WorkbenchState,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';

const hostRequire = createRequire(
  new URL('../../../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement, isValidElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');
const {ServerStyleSheet} = hostRequire('styled-components') as typeof import('styled-components');

function makeResult(): WorkflowStepResult {
  return {
    key: 'current',
    outputs: {scaled: makeFixturePoints(2)},
    diagnostics: [
      {severity: 'warning', code: 'rounding', count: 2, message: 'Rounded two coordinates.'},
      {severity: 'error', code: 'missing', count: 1, message: 'A value is missing.'},
      {severity: 'info', code: 'retained', count: 2, message: 'Kept two rows.'}
    ],
    summary: {inputRows: 1204, maximumDistance: 1.23456, note: 'Two points'},
    engine: 'gpu',
    dispatchedContributors: ['GPUScale', 'GPUCopy'],
    parameters: {factor: 3, fail: false},
    durationMilliseconds: 1800,
    cacheHit: false,
    completedAt: Date.UTC(2026, 10, 14, 12, 34, 56)
  };
}

function makeStore(
  stepState: WorkflowStepState | null = {
    status: 'done',
    key: 'current',
    issues: [],
    result: makeResult()
  },
  patch: Partial<WorkbenchState> = {}
) {
  const document = makeFixtureDocument();
  const state: WorkbenchState = {
    revision: 1,
    workflows: [document],
    activeWorkflowId: document.id,
    document,
    runner: stepState
      ? {
          revision: 1,
          running: stepState.status === 'running',
          order: ['scale'],
          issues: [],
          steps: {scale: stepState}
        }
      : null,
    selectedStepIds: ['scale'],
    canUndo: false,
    canRedo: false,
    dock: {open: true, height: 380, inspectorTab: 'result', logOpen: false},
    palette: {open: false},
    dialog: null,
    draft: null,
    notice: null,
    runLog: [],
    templates: [],
    clipboardStepCount: 0,
    ...patch
  };
  const calls: unknown[][] = [];
  const record = (name: string, ...values: unknown[]) => {
    calls.push([name, ...values]);
  };
  const store: WorkbenchStore = {
    registry: makeFixtureRegistry(),
    host: makeFixtureHost(),
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
      return true;
    },
    undo: () => record('undo'),
    redo: () => record('redo'),
    select: value => record('select', value),
    run: async options => {
      record('run', options);
      return null;
    },
    cancel: () => record('cancel'),
    notifyExternalChange: () => record('notifyExternalChange'),
    getRunner: () => null,
    setDock: value => record('setDock', value),
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
    clear: async () => {
      record('clear');
    }
  };
  return {store, calls};
}
function render(stepState?: WorkflowStepState | null, patch?: Partial<WorkbenchState>): string {
  return renderToStaticMarkup(
    createElement(WorkbenchResultView, {store: makeStore(stepState, patch).store})
  );
}
function makeTable(values: readonly (number | null)[]): WorkflowData {
  return {
    kind: 'table',
    table: {
      id: 'values',
      label: 'Values',
      rowCount: values.length,
      columns: {value: values},
      columnTypes: {value: 'float64'}
    }
  };
}

describe('workbench result model', () => {
  test('formats nulls, booleans, integers, decimals and non-finite values exactly', () => {
    expect(formatWorkbenchCell(null, 'float64')).toBe('null');
    expect(formatWorkbenchCell(undefined, 'string')).toBe('null');
    expect(formatWorkbenchCell(true, 'boolean')).toBe('true');
    expect(formatWorkbenchCell(false, 'boolean')).toBe('false');
    expect(formatWorkbenchCell(1204, 'int32')).toBe('1,204');
    expect(formatWorkbenchCell(1.234567, 'float64')).toBe('1.235');
    expect(formatWorkbenchCell(0.000012345, 'float64')).toBe('0.00001235');
    expect(formatWorkbenchCell(NaN, 'float64')).toBe('NaN');
    expect(formatWorkbenchCell(Infinity, 'float64')).toBe('∞');
    expect(formatWorkbenchCell(-Infinity, 'float64')).toBe('−∞');
    expect(formatWorkbenchCell(-0, 'float64')).toBe('-0');
  });
  test('truncates long strings to 120 characters and preserves short strings', () => {
    expect(formatWorkbenchCell('x'.repeat(120), 'string')).toBe('x'.repeat(120));
    expect(formatWorkbenchCell('x'.repeat(10000), 'string')).toBe(`${'x'.repeat(119)}…`);
    expect(formatWorkbenchCell('<script>', 'string')).toBe('<script>');
  });
  test('describes supported geometry with exact counts', () => {
    const geometry = (type: string, coordinates: unknown) => JSON.stringify({type, coordinates});
    expect(describeWorkbenchGeometry(geometry('Point', [1, 2]))).toBe('Point');
    expect(formatWorkbenchCell(geometry('Point', [1, 2, 3]), 'geometry')).toBe('Point');
    expect(
      describeWorkbenchGeometry(
        geometry('Polygon', [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0]
          ]
        ])
      )
    ).toBe('Polygon · 5 vertices');
    expect(
      describeWorkbenchGeometry(
        geometry('MultiLineString', [
          [
            [0, 0],
            [1, 1]
          ],
          [
            [2, 2],
            [3, 3]
          ],
          [
            [4, 4],
            [5, 5]
          ]
        ])
      )
    ).toBe('MultiLineString · 3 parts');
    expect(
      describeWorkbenchGeometry(
        geometry('MultiPolygon', [
          [
            [
              [0, 0],
              [1, 0],
              [1, 1],
              [0, 0]
            ]
          ]
        ])
      )
    ).toBe('MultiPolygon · 1 part');
    expect(
      describeWorkbenchGeometry(
        geometry('LineString', [
          [0, 0],
          [1, 1]
        ])
      )
    ).toBe('LineString · 2 vertices');
    expect(describeWorkbenchGeometry(geometry('MultiPoint', [[0, 0]]))).toBe(
      'MultiPoint · 1 vertex'
    );
  });
  test('rejects malformed geometry text and invalid coordinate shapes', () => {
    for (const text of [
      '',
      '{',
      'null',
      '[]',
      '{}',
      '"Point"',
      '{"type":"Point","coordinates":["1",2]}',
      '{"type":"Point","coordinates":[1]}',
      '{"type":"Polygon","coordinates":[1,2]}',
      '{"type":"Polygon","coordinates":[]}',
      '{"type":"Point","coordinates":[1e999,2]}',
      '{"type":"Unknown","coordinates":[1,2]}'
    ]) {
      expect(describeWorkbenchGeometry(text)).toBe('Invalid geometry');
    }
  });
  test('pages with identity first, retained table order next, geometry last and clamped indexes', () => {
    const data = makeFixturePoints(53);
    data.table.columns = {
      geometry: data.table.columns['geometry'],
      value: data.table.columns['value'],
      point_id: data.table.columns['point_id']
    };
    const page = getWorkbenchPreviewPage(data, 0, 50);
    expect(page.columns).toEqual(['point_id', 'value', 'geometry']);
    expect(page.rows.length).toBe(50);
    expect(page.rows[0]).toEqual(['p0', 0, '{"type":"Point","coordinates":[-87.6,41.8]}']);
    expect([page.startRow, page.endRow, page.pageIndex, page.pageCount, page.rowCount]).toEqual([
      1, 50, 0, 2, 53
    ]);
    const finalPage = getWorkbenchPreviewPage(data, 100, 50);
    expect([
      finalPage.startRow,
      finalPage.endRow,
      finalPage.pageIndex,
      finalPage.rows.length
    ]).toEqual([51, 53, 1, 3]);
    expect(getWorkbenchPreviewPage(data, -4, 2).pageIndex).toBe(0);
    expect(getWorkbenchPreviewPage(data, 1.9, 2).rows.map(row => row[0])).toEqual(['p2', 'p3']);
    expect(getWorkbenchPreviewPage(data, NaN, 10000).rows.length).toBe(50);
    expect(getWorkbenchPreviewPage(data, 0, 0).rows.length).toBe(1);
  });
  test('handles empty and all-null tables without manufacturing rows', () => {
    expect(getWorkbenchPreviewPage(makeTable([]), 4, 50)).toEqual({
      columns: ['value'],
      rows: [],
      pageIndex: 0,
      pageCount: 0,
      rowCount: 0,
      startRow: 0,
      endRow: 0
    });
    expect(getWorkbenchPreviewPage(makeTable([null, null]), 0, 50).rows).toEqual([[null], [null]]);
  });
  test('builds equal-width histogram bins and includes the maximum in the last bin', () => {
    expect(makeWorkbenchHistogram([0, 1, 2, 3, 4], 4)).toEqual({
      edges: [0, 1, 2, 3, 4],
      counts: [1, 1, 1, 2]
    });
    expect(makeWorkbenchHistogram([-4, -2, 0, 2, 4], 2)).toEqual({
      edges: [-4, 0, 4],
      counts: [2, 3]
    });
    expect(makeWorkbenchHistogram([0, 12], 12)).toEqual({
      edges: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      counts: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    });
  });
  test('histograms ignore missing and non-finite values and collapse constants', () => {
    expect(makeWorkbenchHistogram([], 12)).toEqual({edges: [], counts: []});
    expect(makeWorkbenchHistogram([null, null, NaN, Infinity, -Infinity], 12)).toEqual({
      edges: [],
      counts: []
    });
    expect(makeWorkbenchHistogram([7], 12)).toEqual({edges: [7, 7], counts: [1]});
    expect(makeWorkbenchHistogram([7, 7, null, NaN], 12)).toEqual({edges: [7, 7], counts: [2]});
    expect(makeWorkbenchHistogram([0, 2], 0)).toEqual({edges: [0, 2], counts: [2]});
    expect(makeWorkbenchHistogram([-Number.MAX_VALUE, 0, Number.MAX_VALUE], 2)).toEqual({
      edges: [-Number.MAX_VALUE, 0, Number.MAX_VALUE],
      counts: [1, 2]
    });
  });
  test('formats durations, summary labels and safe download names', () => {
    expect([0, 42, 999, 1000, 1800, -1, NaN].map(formatWorkbenchDuration)).toEqual([
      '0 ms',
      '42 ms',
      '999 ms',
      '1 s',
      '1.8 s',
      '0 ms',
      '0 ms'
    ]);
    expect(
      ['inputRows', 'GPUTime', 'maximum_distance', 'row-count', ''].map(formatWorkbenchSummaryKey)
    ).toEqual(['Input rows', 'Gpu time', 'Maximum distance', 'Row count', '']);
    expect(makeWorkbenchDownloadName('fixture-workflow', 'scale', 'scaled', 'csv')).toBe(
      'fixture-workflow-scale-scaled.csv'
    );
    expect(makeWorkbenchDownloadName('../my map', 'step/2', 'a"<b>', '.geojson')).toBe(
      'my-map-step-2-a-b.geojson'
    );
    expect(makeWorkbenchDownloadName('', '', '', '')).toBe('output-output-output.output');
  });
});

describe('workbench result view', () => {
  test('renders done output with provenance, summary, preview, downloads and map toggle', () => {
    const markup = render();
    for (const identifier of [
      WORKBENCH_TEST_IDS.result,
      WORKBENCH_TEST_IDS.resultEngine,
      WORKBENCH_TEST_IDS.resultOutput('scaled'),
      WORKBENCH_TEST_IDS.resultTable('scaled'),
      WORKBENCH_TEST_IDS.resultDisplay('scaled'),
      WORKBENCH_TEST_IDS.resultDownload('scaled', 'csv'),
      WORKBENCH_TEST_IDS.resultDownload('scaled', 'geojson')
    ])
      expect(markup).toContain(`data-testid="${identifier}"`);
    expect(markup).toContain('data-status="done"');
    expect(markup).toContain('>Done</span>');
    expect(markup).toContain('GPU · GPUScale, GPUCopy');
    expect(markup).toContain('1.8 s');
    expect(markup).toContain('dateTime="2026-11-14T12:34:56.000Z"');
    expect(markup).toMatch(/<time[^>]+>\d\d:\d\d:\d\d<\/time>/);
    expect(markup).toContain('<dt>Input rows</dt><dd>1,204</dd>');
    expect(markup).toContain('<dt>Maximum distance</dt><dd>1.235</dd>');
    expect(markup).toContain('2 rows · 3 columns');
    expect(markup).toContain('aria-label="Scaled preview"');
    expect(markup).toContain('aria-label="Summarize point_id"');
    expect(markup).toContain('>p0</td>');
    expect(markup).toContain('class="number">1.5</td>');
    expect(markup).toContain('>Point</td>');
    expect(markup).toContain('>2 rows</span>');
    expect(markup).toMatch(/aria-label="Show Scaled on the map"[^>]*aria-pressed="true"/);
    expect(markup).toContain('aria-label="Download Scaled CSV"');
    expect(markup).toContain('aria-label="Download Scaled GeoJSON"');
    expect(markup).toMatch(/<details[^>]*><summary>Parameters used \(2\)<\/summary>/);
    expect(markup).toContain('<dt>Factor</dt><dd>3</dd>');
    expect(markup).not.toContain('role="dialog"');
    expect(markup.match(/<strong>(Error|Warning|Info)<\/strong>/g)).toEqual([
      '<strong>Error</strong>',
      '<strong>Warning</strong>',
      '<strong>Info</strong>'
    ]);
    expect(markup).toContain('missing · 1 · A value is missing.');
  });
  test('derives staleness from the result key and offers run to here', () => {
    const markup = render({status: 'stale', key: 'new', issues: [], result: makeResult()});
    expect(markup).toContain('data-status="stale"');
    expect(markup).toContain(
      'This result is from an earlier configuration. Run the step to update it.'
    );
    expect(markup).toContain('>Run to here</button>');
    expect(markup).toContain('workflow-result-table-scaled');
  });
  test('places the last run error above an older result', () => {
    const message = 'Fixture scale: Factor failed; change Factor and run again.';
    const markup = render({
      status: 'error',
      key: 'new',
      issues: [],
      error: {severity: 'error', code: 'operation-failed', message},
      result: makeResult()
    });
    expect(markup).toContain('data-status="error"');
    expect(markup).toContain(message);
    expect(markup.indexOf(message)).toBeGreaterThan(
      markup.indexOf(WORKBENCH_TEST_IDS.resultEngine)
    );
    expect(markup.indexOf(message)).toBeLessThan(
      markup.indexOf(WORKBENCH_TEST_IDS.resultTable('scaled'))
    );
    expect(markup).toContain('>p0</td>');
  });
  test('shows ready, running, disabled and configuration error states without a result', () => {
    expect(render({status: 'ready', key: 'current', issues: []})).toContain(
      'Run this step to see its output'
    );
    const running = render({status: 'running', key: 'current', issues: []});
    expect(running).toContain('data-status="running"');
    expect(running).toMatch(/data-status="running" aria-live="polite"[^>]*>Running…<\/span>/);
    expect(render({status: 'disabled', key: null, issues: []})).toContain('>Disabled</span>');
    const invalid = render({
      status: 'not-configured',
      key: null,
      issues: [
        {
          severity: 'error',
          code: 'input-required',
          message: 'Fixture scale needs an input for "Points".'
        }
      ]
    });
    expect(invalid).toContain('Fixture scale needs an input for &quot;Points&quot;.');
    expect(invalid).not.toContain('workflow-result-table-scaled');
  });
  test('omits GeoJSON for a table and keeps a disabled map toggle off', () => {
    const document = makeFixtureDocument();
    document.steps[1].display = [];
    const result = {
      ...makeResult(),
      outputs: {scaled: makeTable([null, NaN, Infinity])},
      engine: 'cpu-reference' as const,
      fallbackReason: 'No GPU device supplied',
      cacheHit: true
    };
    const markup = render({status: 'done', key: 'current', issues: [], result}, {document});
    expect(markup).toContain('CPU reference');
    expect(markup).toContain('from cache');
    expect(markup).not.toContain('1.8 s');
    expect(markup).toContain('GPU not used: No GPU device supplied');
    expect(markup).toContain('aria-pressed="false"');
    expect(markup).toContain('workflow-result-download-scaled-csv');
    expect(markup).not.toContain('workflow-result-download-scaled-geojson');
    expect(markup).toMatch(/>null<\/span>/);
    expect(markup).toContain('>NaN</td>');
    expect(markup).toContain('>∞</td>');
  });
  test('shows published dataset labels and publication failures', () => {
    const markup = render({
      status: 'done',
      key: 'current',
      issues: [
        {
          severity: 'error',
          code: 'publish-failed',
          port: 'scaled',
          message: 'Fixture scale: Scaled could not be shown on the map.'
        }
      ],
      result: makeResult(),
      published: {scaled: {datasetId: 'map-scaled', label: 'Scaled points', key: 'current'}}
    });
    expect(markup).toContain('On the map as &quot;Scaled points&quot;');
    expect(markup).toContain('Fixture scale: Scaled could not be shown on the map.');
  });
  test('renders empty workflows, absent selection, multiple selection and a null runner', () => {
    for (const patch of [
      {document: null, workflows: []},
      {document: {...makeFixtureDocument(), steps: []}},
      {selectedStepIds: []},
      {selectedStepIds: ['missing']},
      {selectedStepIds: ['source', 'scale']}
    ]) {
      expect(render(null, patch)).toContain('Select one step to see its output.');
    }
    expect(render(null)).toContain('Run this step to see its output');
    const document = makeFixtureDocument();
    document.steps[1].operationId = 'removed-operation';
    expect(render(null, {document})).toContain('removed-operation: This operation is unavailable.');
    expect(render(null, {document})).toContain('data-status="not-configured"');
    expect(render(undefined, {document})).toContain('workflow-result-table-scaled');
  });
  test('renders at most 50 rows, retains full string titles and geometry last', () => {
    const data = makeFixturePoints(1204);
    const longText = 'a'.repeat(150);
    data.table.columns = {
      geometry: data.table.columns['geometry'],
      point_id: [longText, ...data.table.columns['point_id'].slice(1)],
      value: data.table.columns['value']
    };
    const markup = render({
      status: 'done',
      key: 'current',
      issues: [],
      result: {...makeResult(), outputs: {scaled: data}}
    });
    expect(markup).toContain('Rows 1 to 50 of 1,204');
    expect(markup).toContain(`title="${longText}"`);
    expect(markup).toContain(`${'a'.repeat(119)}…</td>`);
    expect(markup.match(/<tbody>(.*?)<\/tbody>/)?.[1].match(/<tr>/g)).toHaveLength(50);
    expect(markup).not.toContain('>p50</td>');
    expect(markup.indexOf('Summarize value')).toBeLessThan(markup.indexOf('Summarize geometry'));
  });
  test('stacks two outputs with only the first preview expanded and labels Host runs', () => {
    const markup = render({
      status: 'done',
      key: 'current',
      issues: [],
      result: {
        ...makeResult(),
        engine: 'host',
        outputs: {scaled: makeFixturePoints(1), counts: makeTable([1])}
      }
    });
    expect(markup).toContain('Host');
    expect(markup).toContain('workflow-result-output-scaled');
    expect(markup).toContain('workflow-result-output-counts');
    expect(markup).toContain('workflow-result-table-scaled');
    expect(markup).not.toContain('workflow-result-table-counts');
    expect(markup).toMatch(
      /<details open=""[^>]*><summary><span aria-hidden="true">▾<\/span><span[^>]*><strong>Scaled/
    );
    expect(markup).toMatch(
      /<details[^>]*><summary><span aria-hidden="true">▸<\/span><span[^>]*><strong>Values/
    );
  });
});

// Each probe owns a hook dispatcher and restores React immediately after a shallow render.
// These invoke real callbacks; DOM layout, native events and React scheduling need browser QA.
function makeHookProbe(renderComponent: () => React.JSX.Element | null) {
  const internals = (
    hostRequire('react') as {
      __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
    }
  ).__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const slots = new Map<number, unknown>();
  const dependencies = new Map<number, readonly unknown[] | undefined>();
  let cursor = 0;
  const dispatcher = {
    useState<Value>(initial: Value) {
      const position = cursor++;
      if (!slots.has(position)) slots.set(position, initial);
      return [slots.get(position), (value: Value) => slots.set(position, value)];
    },
    useRef<Value>(initial: Value) {
      const position = cursor++;
      if (!slots.has(position)) slots.set(position, {current: initial});
      return slots.get(position);
    },
    useMemo<Value>(calculate: () => Value, next: readonly unknown[]) {
      const position = cursor++;
      const previous = dependencies.get(position);
      if (
        !previous ||
        previous.length !== next.length ||
        next.some((value, index) => !Object.is(value, previous[index]))
      ) {
        slots.set(position, calculate());
        dependencies.set(position, next);
      }
      return slots.get(position);
    },
    useEffect() {},
    useSyncExternalStore(_subscribe: unknown, getSnapshot: () => unknown) {
      cursor++;
      return getSnapshot();
    }
  };
  return {
    read() {
      const previous = internals.H;
      cursor = 0;
      try {
        internals.H = dispatcher;
        return renderComponent();
      } finally {
        internals.H = previous;
      }
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
  const found = find(tree);
  if (!found) throw new Error('Result probe: expected element was absent.');
  return found;
}
function findComponent(tree: ReactNode, name: string): ProbeElement {
  return findProbeElement(
    tree,
    element => typeof element.type === 'function' && element.type.name === name
  );
}
function probeComponent(element: ProbeElement) {
  const component = element.type as (
    properties: Record<string, unknown>
  ) => React.JSX.Element | null;
  return makeHookProbe(() => component(element.props));
}
function invokeProbe(element: ProbeElement, handler: string, event?: unknown): void {
  const callback = element.props[handler];
  if (typeof callback !== 'function') throw new Error(`Result probe: ${handler} was absent.`);
  callback(event);
}
function findButton(tree: ReactNode, name: string): ProbeElement {
  return findProbeElement(tree, element => element.props['aria-label'] === name);
}
function makeOutputProbe(result = makeResult(), patch: Partial<WorkbenchState> = {}) {
  const fixture = makeStore({status: 'done', key: 'current', issues: [], result}, patch);
  const root = makeHookProbe(() => WorkbenchResultView({store: fixture.store}));
  const output = probeComponent(findComponent(root.read(), 'ResultOutput'));
  return {...fixture, root, output};
}
function renderData(data: WorkflowData, summary: WorkflowStepResult['summary'] = {}): string {
  return render({
    status: 'done',
    key: 'current',
    issues: [],
    result: {...makeResult(), outputs: {scaled: data}, summary}
  });
}

// Regression cases first ran against the reviewed implementation to confirm each finding.
describe('result review regressions', () => {
  test('F1: rejects non-string types without invoking coercion', () => {
    const text = '{"type":{"toString":null},"coordinates":[1,2]}';
    expect(describeWorkbenchGeometry(text)).toBe('Invalid geometry');
    const data = makeFixturePoints(1);
    data.table.columns = {...data.table.columns, geometry: [text]};
    expect(renderData(data)).toContain('>Invalid geometry</td>');
  });
  test('F1: rejects inherited type names before examining deeply nested coordinates', () => {
    for (const type of ['constructor', '__proto__', 'toString', 'hasOwnProperty']) {
      const text = `{"type":"${type}","coordinates":${'['.repeat(10000)}0${']'.repeat(10000)}}`;
      expect(describeWorkbenchGeometry(text)).toBe('Invalid geometry');
    }
  });
  test('F2: assigns exact nonzero-minimum boundaries to the left-closed bin', () => {
    expect(makeWorkbenchHistogram([1, 7, 13], 12)).toEqual({
      edges: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
      counts: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    });
  });
  test('F3: preserves an empty geometry column name and offers GeoJSON', () => {
    const data: WorkflowData = {
      kind: 'point',
      table: {
        id: 'empty-name',
        label: 'Empty name',
        rowCount: 1,
        geometryColumn: '',
        columns: {'': ['{"type":"Point","coordinates":[1,2]}'], value: [3]},
        columnTypes: {'': 'string', value: 'int32'}
      }
    };
    expect(JSON.parse(makeWorkflowGeoJSONText(data)).features[0].geometry).toEqual({
      type: 'Point',
      coordinates: [1, 2]
    });
    expect(getWorkbenchPreviewPage(data, 0).columns).toEqual(['value', '']);
    expect(getWorkbenchPreviewPage(data, 0).rows).toEqual([
      [3, '{"type":"Point","coordinates":[1,2]}']
    ]);
    expect(renderData(data)).toContain(
      `data-testid="${WORKBENCH_TEST_IDS.resultDownload('scaled', 'geojson')}"`
    );
    expect(renderData(data)).toContain('>Point</td>');
  });
  test('F4: summarizes valid GeometryCollections, including nested and empty collections', () => {
    const collection = {
      type: 'GeometryCollection',
      geometries: [{type: 'Point', coordinates: [1, 2]}]
    };
    expect(describeWorkbenchGeometry(JSON.stringify(collection))).toBe(
      'GeometryCollection · 1 part'
    );
    expect(
      describeWorkbenchGeometry(
        JSON.stringify({type: 'GeometryCollection', geometries: [collection, collection]})
      )
    ).toBe('GeometryCollection · 2 parts');
    expect(describeWorkbenchGeometry('{"type":"GeometryCollection","geometries":[]}')).toBe(
      'GeometryCollection · 0 parts'
    );
    const data = makeWorkflowDataFromFeatures(
      {type: 'FeatureCollection', features: [{geometry: collection}]},
      {id: 'collection', label: 'Collection'}
    );
    expect(JSON.parse(makeWorkflowGeoJSONText(data)).features[0].geometry).toEqual(collection);
    expect(renderData(data)).toContain('>GeometryCollection · 1 part</td>');
  });
  test('F5: rejects short lines, short rings and unclosed rings in every coordinate nesting', () => {
    const invalidGeometries = [
      {type: 'LineString', coordinates: [[1, 2]]},
      {type: 'MultiLineString', coordinates: [[[1, 2]]]},
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [0, 0]
          ]
        ]
      },
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
          ]
        ]
      },
      {
        type: 'MultiPolygon',
        coordinates: [
          [
            [
              [0, 0],
              [1, 0],
              [1, 1],
              [0, 1]
            ]
          ]
        ]
      },
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 2]
          ]
        ]
      }
    ];
    for (const geometry of invalidGeometries)
      expect(describeWorkbenchGeometry(JSON.stringify(geometry))).toBe('Invalid geometry');
  });
  test('F6: preserves the complete run-summary string', () => {
    const note = `${'x'.repeat(120)} Important final detail.`;
    expect(renderData(makeFixturePoints(1), {note})).toContain(`<dt>Note</dt><dd>${note}</dd>`);
  });
  test('F7: honors retained column order for integer-like names', () => {
    const data = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {geometry: {type: 'Point', coordinates: [1, 2]}, properties: {first: 1}},
          {geometry: {type: 'Point', coordinates: [3, 4]}, properties: {'1': 2}}
        ]
      },
      {id: 'ordered', label: 'Ordered'}
    );
    expect(getWorkflowColumnNames(data)).toEqual(['first', '1']);
    expect(getWorkbenchPreviewPage(data, 0).columns).toEqual(['first', '1', 'geometry']);
    const headers = renderData(data).match(/aria-label="Summarize [^"]*"/g);
    expect(headers).toEqual([
      'aria-label="Summarize first"',
      'aria-label="Summarize 1"',
      'aria-label="Summarize geometry"'
    ]);
  });
  test('F8: reuses column summaries after collapsing and remounting the same output', () => {
    const data = makeTable([1, 7, 13]);
    const values = data.table.columns['value'];
    let scans = 0;
    Object.defineProperty(values, Symbol.iterator, {
      value: function* () {
        scans++;
        yield 1;
        yield 7;
        yield 13;
      }
    });
    const {output} = makeOutputProbe({
      ...makeResult(),
      outputs: {scaled: data, other: makeTable([2])}
    });
    let tree = output.read();
    let preview = probeComponent(findComponent(tree, 'OutputPreview'));
    expect(scans).toBe(0);
    invokeProbe(findButton(preview.read(), 'Summarize value'), 'onClick', {
      currentTarget: {focus() {}}
    });
    const openedTree = preview.read();
    const summary = findComponent(openedTree, 'ColumnSummary');
    expect(summary.props['summary']).toMatchObject({
      minimum: 1,
      maximum: 13,
      mean: 7,
      nullCount: 0
    });
    expect(scans).toBe(3);
    const disclosure = findProbeElement(
      tree,
      element => typeof element.props['onToggle'] === 'function'
    );
    invokeProbe(disclosure, 'onToggle', {currentTarget: {open: false}});
    tree = output.read();
    expect(() => findComponent(tree, 'OutputPreview')).toThrow('expected element was absent');
    invokeProbe(
      findProbeElement(tree, element => typeof element.props['onToggle'] === 'function'),
      'onToggle',
      {currentTarget: {open: true}}
    );
    tree = output.read();
    expect(findComponent(tree, 'OutputPreview').props['data']).toBe(data);
    preview = probeComponent(findComponent(tree, 'OutputPreview'));
    invokeProbe(findButton(preview.read(), 'Summarize value'), 'onClick', {
      currentTarget: {focus() {}}
    });
    expect(findComponent(preview.read(), 'ColumnSummary').props['summary']).toBe(
      summary.props['summary']
    );
    expect(scans).toBe(3);
  });
});

test('bounds collection traversal and rejects invalid collection members', () => {
  expect(
    describeWorkbenchGeometry(
      '{"type":"GeometryCollection","geometries":[{"type":"LineString","coordinates":[[1,2]]}]}'
    )
  ).toBe('Invalid geometry');
  expect(describeWorkbenchGeometry('{"type":"GeometryCollection","geometries":null}')).toBe(
    'Invalid geometry'
  );
  const prefix = '{"type":"GeometryCollection","geometries":[';
  const point = '{"type":"Point","coordinates":[1,2]}';
  expect(describeWorkbenchGeometry(`${prefix.repeat(32)}${point}${']}'.repeat(32)}`)).toBe(
    'GeometryCollection · 1 part'
  );
  expect(describeWorkbenchGeometry(`${prefix.repeat(33)}${point}${']}'.repeat(33)}`)).toBe(
    'Invalid geometry'
  );
  expect(
    describeWorkbenchGeometry(`${prefix}${Array.from({length: 10000}, () => point).join(',')}]}`)
  ).toBe('Invalid geometry');
});

test('executes next and previous controls with exact visible rows and disabled limits', () => {
  const {output} = makeOutputProbe({...makeResult(), outputs: {scaled: makeFixturePoints(53)}});
  const preview = probeComponent(findComponent(output.read(), 'OutputPreview'));
  expect(findButton(preview.read(), 'Previous Scaled page').props['disabled']).toBe(true);
  expect(findButton(preview.read(), 'Next Scaled page').props['disabled']).toBe(false);
  invokeProbe(findButton(preview.read(), 'Next Scaled page'), 'onClick');
  const finalPage = renderToStaticMarkup(preview.read());
  expect(finalPage).toContain('Rows 51 to 53 of 53');
  expect(finalPage).toContain('>p50</td>');
  expect(finalPage.match(/<tr>/g)).toHaveLength(4);
  expect(findButton(preview.read(), 'Next Scaled page').props['disabled']).toBe(true);
  expect(findButton(preview.read(), 'Previous Scaled page').props['disabled']).toBe(false);
  invokeProbe(findButton(preview.read(), 'Previous Scaled page'), 'onClick');
  const firstPage = renderToStaticMarkup(preview.read());
  expect(firstPage).toContain('Rows 1 to 50 of 53');
  expect(firstPage).toContain('>p0</td>');
  expect(firstPage).not.toContain('>p50</td>');
});

test('opens numeric and string summaries, closes on Escape and returns focus to the opener', () => {
  const data = makeTable([null, 1, 7, 13]);
  data.table.columns = {...data.table.columns, label: ['first', 'first', 'second', null]};
  data.table.columnTypes = {...data.table.columnTypes, label: 'string'};
  const {output} = makeOutputProbe({...makeResult(), outputs: {scaled: data}});
  const preview = probeComponent(findComponent(output.read(), 'OutputPreview'));
  const focus = vi.fn();
  invokeProbe(findButton(preview.read(), 'Summarize value'), 'onClick', {currentTarget: {focus}});
  let summary = findComponent(preview.read(), 'ColumnSummary');
  expect(summary.props['summary']).toEqual({
    name: 'value',
    type: 'float64',
    rowCount: 4,
    nullCount: 1,
    minimum: 1,
    maximum: 13,
    mean: 7,
    nonFiniteCount: 0
  });
  expect(summary.props['histogram']).toEqual({
    edges: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    counts: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  });
  const numericMarkup = renderToStaticMarkup(summary);
  expect(numericMarkup).toContain('role="dialog" aria-label="value summary"');
  expect(numericMarkup).toContain('Distribution: most values between 1 and 2');
  expect(numericMarkup.match(/<rect /g)).toHaveLength(12);
  const dialog = probeComponent(summary);
  const stopPropagation = vi.fn();
  invokeProbe(
    findProbeElement(dialog.read(), element => element.props['role'] === 'dialog'),
    'onKeyDown',
    {key: 'Escape', stopPropagation}
  );
  expect(stopPropagation).toHaveBeenCalledTimes(1);
  expect(focus).toHaveBeenCalledTimes(1);
  expect(() => findComponent(preview.read(), 'ColumnSummary')).toThrow(
    'expected element was absent'
  );
  invokeProbe(findButton(preview.read(), 'Summarize label'), 'onClick', {currentTarget: {focus}});
  summary = findComponent(preview.read(), 'ColumnSummary');
  expect(summary.props['summary']).toEqual({
    name: 'label',
    type: 'string',
    rowCount: 4,
    nullCount: 1,
    distinctCount: 2,
    distinctLimitReached: false,
    topValues: [
      {value: 'first', count: 2},
      {value: 'second', count: 1}
    ]
  });
  const stringMarkup = renderToStaticMarkup(summary);
  expect(stringMarkup).toContain('first: 2</li>');
  expect(stringMarkup).toContain('second: 1</li>');
  expect(stringMarkup).not.toContain('<svg');
  invokeProbe(findButton(probeComponent(summary).read(), 'Close label summary'), 'onClick');
  expect(focus).toHaveBeenCalledTimes(2);
});

test('keeps cached columns distinct and recomputes summaries for a different data object', () => {
  const first = makeTable([1, 3]);
  const second = makeTable([10, 20]);
  function openSummary(data: WorkflowData) {
    const {output} = makeOutputProbe({...makeResult(), outputs: {scaled: data}});
    const preview = probeComponent(findComponent(output.read(), 'OutputPreview'));
    invokeProbe(findButton(preview.read(), 'Summarize value'), 'onClick', {
      currentTarget: {focus() {}}
    });
    return findComponent(preview.read(), 'ColumnSummary').props['summary'];
  }
  expect(openSummary(first)).toMatchObject({minimum: 1, maximum: 3, mean: 2});
  expect(openSummary(second)).toMatchObject({minimum: 10, maximum: 20, mean: 15});
  expect(openSummary(first)).not.toBe(openSummary(second));
});

test('executes map display edits using the current document and leaves the input unchanged', () => {
  const {store, output, calls} = makeOutputProbe();
  const initial = store.getState().document!;
  invokeProbe(findButton(output.read(), 'Show Scaled on the map'), 'onClick');
  const edit = calls[0][2] as (document: typeof initial) => typeof initial;
  expect(calls[0].slice(0, 2)).toEqual(['edit', 'display:scale:scaled']);
  expect(edit(initial).steps[1].display).toBeUndefined();
  expect(initial.steps[1].display).toEqual(['scaled']);
  const newer = {
    ...initial,
    steps: initial.steps.map(step => (step.id === 'scale' ? {...step, display: ['other']} : step))
  };
  expect(edit(newer).steps[1].display).toEqual(['other', 'scaled']);
  expect(newer.steps[1].display).toEqual(['other']);
});

test('executes run to here for the stale step and disables it while running', () => {
  const fixture = makeStore({status: 'stale', key: 'new', issues: [], result: makeResult()});
  const root = makeHookProbe(() => WorkbenchResultView({store: fixture.store}));
  const button = findProbeElement(
    root.read(),
    element => element.props['children'] === 'Run to here'
  );
  invokeProbe(button, 'onClick');
  expect(fixture.calls).toEqual([['run', {target: 'scale'}]]);
  fixture.store.getState().runner!.running = true;
  expect(
    findProbeElement(root.read(), element => element.props['children'] === 'Run to here').props[
      'disabled'
    ]
  ).toBe(true);
});

test('downloads exact CSV and GeoJSON bytes through temporary anchors and revokes their URLs', async () => {
  const data: WorkflowData = {
    kind: 'point',
    table: {
      id: 'download',
      label: 'Download',
      rowCount: 1,
      geometryColumn: '',
      columns: {name: ['a,b'], '': ['{"type":"Point","coordinates":[1,2]}']},
      columnTypes: {name: 'string', '': 'string'}
    }
  };
  const {output, calls} = makeOutputProbe({...makeResult(), outputs: {scaled: data}});
  const anchors: Array<{
    href: string;
    download: string;
    click: ReturnType<typeof vi.fn>;
    remove: ReturnType<typeof vi.fn>;
  }> = [];
  const blobs: Blob[] = [];
  const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockImplementation(blob => {
    blobs.push(blob as Blob);
    return `blob:result-${blobs.length}`;
  });
  const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  const append = vi.fn();
  vi.useFakeTimers();
  vi.stubGlobal('document', {
    body: {append},
    createElement: vi.fn((tag: string) => {
      expect(tag).toBe('a');
      const anchor = {href: '', download: '', click: vi.fn(), remove: vi.fn()};
      anchors.push(anchor);
      return anchor;
    })
  });
  try {
    invokeProbe(findButton(output.read(), 'Download Scaled CSV'), 'onClick');
    invokeProbe(findButton(output.read(), 'Download Scaled GeoJSON'), 'onClick');
    expect(blobs).toHaveLength(2);
    expect(blobs.map(blob => blob.type)).toEqual([
      'text/csv;charset=utf-8',
      'application/geo+json'
    ]);
    expect(await blobs[0].text()).toBe('name,\n"a,b","{""type"":""Point"",""coordinates"":[1,2]}"');
    expect(await blobs[1].text()).toBe(
      '{"type":"FeatureCollection","features":[{"type":"Feature","id":0,"properties":{"name":"a,b"},"geometry":{"type":"Point","coordinates":[1,2]}}]}'
    );
    expect(anchors.map(anchor => ({href: anchor.href, download: anchor.download}))).toEqual([
      {href: 'blob:result-1', download: 'fixture-workflow-scale-scaled.csv'},
      {href: 'blob:result-2', download: 'fixture-workflow-scale-scaled.geojson'}
    ]);
    expect(append.mock.calls).toEqual([[anchors[0]], [anchors[1]]]);
    for (const anchor of anchors) {
      expect(anchor.click).toHaveBeenCalledTimes(1);
      expect(anchor.remove).toHaveBeenCalledTimes(1);
    }
    expect(calls).toEqual([]);
    expect(revokeObjectURL).not.toHaveBeenCalled();
    vi.runAllTimers();
    expect(revokeObjectURL.mock.calls).toEqual([['blob:result-1'], ['blob:result-2']]);
    expect(createObjectURL).toHaveBeenCalledTimes(2);
  } finally {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  }
});

test('reports a download failure and cleans up the anchor and URL', () => {
  const {output, calls} = makeOutputProbe();
  const click = vi.fn(() => {
    throw new Error('Download blocked');
  });
  const remove = vi.fn();
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:blocked');
  const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  vi.useFakeTimers();
  vi.stubGlobal('document', {body: {append: vi.fn()}, createElement: () => ({click, remove})});
  try {
    invokeProbe(findButton(output.read(), 'Download Scaled CSV'), 'onClick');
    expect(calls).toEqual([
      [
        'setNotice',
        {kind: 'error', message: 'Fixture scale: Scaled download failed: Download blocked'}
      ]
    ]);
    expect(remove).toHaveBeenCalledTimes(1);
    vi.runAllTimers();
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:blocked');
  } finally {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  }
});

describe('result tab layout revisions', () => {
  test('R1: keeps run metadata compact and outputs before counted, collapsed details', () => {
    const result = {...makeResult(), diagnostics: []};
    const markup = render({status: 'done', key: 'current', issues: [], result});
    const runLine = markup.match(
      /<div data-testid="workflow-result-engine"[^>]*>(.*?)<\/div>/
    )?.[1];
    expect(runLine).toBeDefined();
    expect(runLine).toContain('>Done</span>');
    expect(runLine).toContain('GPU · GPUScale, GPUCopy');
    expect(runLine).toContain('1.8 s');
    expect(runLine).toContain('<time');
    const tablePosition = markup.indexOf(
      `data-testid="${WORKBENCH_TEST_IDS.resultTable('scaled')}"`
    );
    expect(tablePosition).toBeGreaterThan(0);
    const details = [...markup.matchAll(/<details([^>]*)><summary>([^<]*)<\/summary>/g)];
    expect(details.map(match => match[2])).toEqual([
      'Summary (3)',
      'Diagnostics (0)',
      'Parameters used (2)'
    ]);
    for (const match of details) {
      expect(match[1]).not.toMatch(/\bopen(?:=|\s|$)/);
      expect(match.index).toBeGreaterThan(tablePosition);
    }
    for (const name of [
      'Show Scaled on the map',
      'Download Scaled CSV',
      'Download Scaled GeoJSON'
    ]) {
      const button = markup.match(
        new RegExp(`<button[^>]*aria-label="${name}"[^>]*>(.*?)<\\/button>`)
      );
      expect(button).not.toBeNull();
      expect(button![0]).toContain(`title="${name}"`);
      expect(button![1]).toContain('aria-hidden="true"');
      expect(button![1]).toContain('<svg');
      expect(button![1]).not.toContain(name);
    }
    expect(markup.indexOf('workflow-result-output-scaled')).toBeLessThan(tablePosition);
    expect(markup.indexOf('workflow-result-display-scaled')).toBeLessThan(tablePosition);
  });

  test('R1: opens diagnostics by default only for warnings and errors', () => {
    for (const severity of ['info', 'warning', 'error'] as const) {
      const result = {
        ...makeResult(),
        diagnostics: [{severity, code: 'example', count: 1, message: 'Example diagnostic.'}]
      };
      const markup = render({status: 'done', key: 'current', issues: [], result});
      const diagnosticDetails = markup.match(
        /<details([^>]*)><summary>Diagnostics \(1\)<\/summary>/
      );
      expect(diagnosticDetails).not.toBeNull();
      expect(/\bopen=""/.test(diagnosticDetails![1])).toBe(severity !== 'info');
      expect(markup).toContain('<summary>Summary (3)</summary>');
    }
  });

  test('R1: places the optional fallback on its own line after provenance and before outputs', () => {
    const result = {
      ...makeResult(),
      engine: 'cpu-reference' as const,
      fallbackReason: 'No GPU device supplied'
    };
    const markup = render({status: 'done', key: 'current', issues: [], result});
    expect(markup).toMatch(/<\/time><\/div><div>GPU not used: No GPU device supplied<\/div>/);
    expect(markup.indexOf('GPU not used:')).toBeLessThan(
      markup.indexOf('workflow-result-output-scaled')
    );
    expect(render()).not.toContain('GPU not used:');
  });

  test('R2: emits a flex height path, an uncapped preview and a separate nonshrinking pager', () => {
    const sheet = new ServerStyleSheet();
    try {
      const markup = renderToStaticMarkup(
        sheet.collectStyles(createElement(WorkbenchResultView, {store: makeStore().store}))
      );
      const styles = sheet.getStyleTags();
      expect(styles).not.toContain('max-height:220px');
      expect(styles).toMatch(/\{flex:1 1 auto;min-height:96px;overflow:auto;\}/);
      expect(styles).toContain('flex:none;flex-wrap:nowrap;justify-content:space-between;');
      expect(styles).toMatch(
        /\[data-testid='workflow-inspector'\]:has\([^)]*\)\{display:flex;flex-direction:column;height:100%;min-height:0;\}/
      );
      expect(styles).toMatch(
        /\[role='tabpanel'\]:has\(>[^)]*\)\{display:flex;flex-direction:column;flex:1;min-height:0;\}/
      );
      // A scroll container ends immediately after the table; the pager follows it.
      expect(markup).toMatch(
        /<\/table><\/div><div[^>]*><span aria-live="polite">2 rows<\/span><\/div>/
      );
    } finally {
      sheet.seal();
    }
  });

  test('R3: puts identity first without losing retained order or duplicating geometry', () => {
    const data = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [
          {id: 'first-id', geometry: {type: 'Point', coordinates: [1, 2]}, properties: {first: 1}},
          {id: 'second-id', geometry: {type: 'Point', coordinates: [3, 4]}, properties: {'1': 2}}
        ]
      },
      {id: 'identity-order', label: 'Identity order'}
    );
    expect(data.identityColumn).toBe('feature_id');
    expect(getWorkbenchPreviewPage(data, 0).columns).toEqual([
      'feature_id',
      'first',
      '1',
      'geometry'
    ]);
    expect(getWorkbenchPreviewPage(data, 0).rows[0]).toEqual([
      'first-id',
      1,
      null,
      '{"type":"Point","coordinates":[1,2]}'
    ]);
    expect(renderData(data).match(/aria-label="Summarize [^"]*"/g)).toEqual([
      'aria-label="Summarize feature_id"',
      'aria-label="Summarize first"',
      'aria-label="Summarize 1"',
      'aria-label="Summarize geometry"'
    ]);
    expect(getWorkbenchPreviewPage({...data, identityColumn: 'missing'}, 0).columns).toEqual([
      'first',
      '1',
      'feature_id',
      'geometry'
    ]);
    expect(getWorkbenchPreviewPage({...data, identityColumn: 'geometry'}, 0).columns).toEqual([
      'first',
      '1',
      'feature_id',
      'geometry'
    ]);
    const table: WorkflowData = {
      kind: 'table',
      identityColumn: '',
      table: {
        id: 'empty-identity',
        label: 'Empty identity',
        rowCount: 1,
        columns: {value: [3], '': ['id']},
        columnTypes: {value: 'int32', '': 'string'}
      }
    };
    expect(getWorkbenchPreviewPage(table, 0).columns).toEqual(['', 'value']);
  });
});

describe('result pager layout regression', () => {
  test('R4: keeps the 40-row pager before the disclosures without reserving a smaller output box', () => {
    const result = {
      ...makeResult(),
      outputs: {scaled: makeFixturePoints(40)},
      summary: {inputRows: 40, outputRows: 40},
      diagnostics: []
    };
    const sheet = new ServerStyleSheet();
    try {
      const markup = renderToStaticMarkup(
        sheet.collectStyles(
          createElement(WorkbenchResultView, {
            store: makeStore({status: 'done', key: 'current', issues: [], result}).store
          })
        )
      );
      const styles = sheet.getStyleTags();
      expect(styles).toContain(
        'display:flex;flex-direction:column;flex:1 1 auto;min-height:auto;padding:4px 8px;'
      );
      expect(styles).toMatch(/\{flex:1 1 auto;min-height:96px;overflow:auto;\}/);
      expect(styles).toContain('flex:none;flex-wrap:nowrap;justify-content:space-between;');
      expect(styles).toContain('flex:none;padding:0 8px;');
      expect(styles).toContain('flex:none;margin:0;');
      expect(styles).not.toContain('180px');
      expect(styles).not.toMatch(/position:absolute|margin[^;]*:-/);
      expect(markup).toMatch(
        /<\/table><\/div><div[^>]*><span aria-live="polite">40 rows<\/span><\/div><\/section><div[^>]*><details[^>]*><summary>Summary \(2\)<\/summary>/
      );
      expect(markup).not.toContain('Previous');
      expect(markup).not.toContain('Next');
    } finally {
      sheet.seal();
    }
  });

  test('R4: omits both navigation controls whenever all rows fit on one page', () => {
    for (const count of [0, 1, 40, 50]) {
      const {output} = makeOutputProbe({
        ...makeResult(),
        outputs: {scaled: makeFixturePoints(count)}
      });
      const preview = probeComponent(findComponent(output.read(), 'OutputPreview'));
      const tree = preview.read();
      const markup = renderToStaticMarkup(tree);
      expect(markup).toContain(`<span aria-live="polite">${count} rows</span>`);
      expect(() => findButton(tree, 'Previous Scaled page')).toThrow('expected element was absent');
      expect(() => findButton(tree, 'Next Scaled page')).toThrow('expected element was absent');
      expect(markup).not.toContain('Rows 1 to');
    }
  });

  test('R4: retains navigation and the range label for 51 rows', () => {
    const {output} = makeOutputProbe({...makeResult(), outputs: {scaled: makeFixturePoints(51)}});
    const preview = probeComponent(findComponent(output.read(), 'OutputPreview'));
    expect(renderToStaticMarkup(preview.read())).toContain('Rows 1 to 50 of 51');
    expect(findButton(preview.read(), 'Previous Scaled page').props['disabled']).toBe(true);
    invokeProbe(findButton(preview.read(), 'Next Scaled page'), 'onClick');
    expect(renderToStaticMarkup(preview.read())).toContain('Rows 51 to 51 of 51');
    expect(findButton(preview.read(), 'Next Scaled page').props['disabled']).toBe(true);
  });
});

describe('result fixes and column colouring', () => {
  function openColumnPanel(output: ReturnType<typeof makeOutputProbe>['output'], column: string) {
    const preview = probeComponent(findComponent(output.read(), 'OutputPreview'));
    invokeProbe(findButton(preview.read(), `Summarize ${column}`), 'onClick', {
      currentTarget: {focus() {}}
    });
    return probeComponent(findComponent(preview.read(), 'ColumnSummary'));
  }
  function findColourButton(tree: ReactNode): ProbeElement {
    return findProbeElement(
      tree,
      element => element.props['data-testid'] === WORKBENCH_TEST_IDS.displayColorColumn('scaled')
    );
  }

  test('chooses diverging colours only across zero and ordinal scales for categories', () => {
    const numeric = {name: 'value', type: 'float64' as const, rowCount: 3, nullCount: 0};
    expect(getWorkbenchColumnDisplayStyle({...numeric, minimum: -2, maximum: 3})).toEqual({
      colorColumn: 'value',
      palette: 'diverging'
    });
    expect(getWorkbenchColumnDisplayStyle({...numeric, minimum: 0, maximum: 3})).toEqual({
      colorColumn: 'value'
    });
    expect(getWorkbenchColumnDisplayStyle({...numeric, minimum: -3, maximum: 0})).toEqual({
      colorColumn: 'value'
    });
    expect(getWorkbenchColumnDisplayStyle({...numeric, nullCount: 3})).toEqual({
      colorColumn: 'value'
    });
    expect(getWorkbenchColumnDisplayStyle({...numeric, name: 'name', type: 'string'})).toEqual({
      colorColumn: 'name',
      colorScale: 'ordinal'
    });
    expect(getWorkbenchColumnDisplayStyle({...numeric, name: 'selected', type: 'boolean'})).toEqual(
      {colorColumn: 'selected', colorScale: 'ordinal'}
    );
  });

  test('shows run-error repairs under the message and applies them with the inspector edit tag', () => {
    const document = makeFixtureDocument();
    document.steps[1].onFallback = 'stop';
    const message = 'Fixture scale: GPU cannot run; CPU fallback is disabled.';
    const failed: WorkflowStepState = {
      status: 'error',
      key: 'current',
      issues: [],
      result: makeResult(),
      error: {code: 'gpu-fallback', severity: 'error', message}
    };
    const {store, calls} = makeStore(failed, {document});
    const markup = renderToStaticMarkup(createElement(WorkbenchResultView, {store}));
    const identifier = WORKBENCH_TEST_IDS.suggestion('continue-cpu:scale');
    expect(markup).toContain(`data-testid="${identifier}"`);
    expect(markup.indexOf(message)).toBeLessThan(markup.indexOf(identifier));
    expect(markup.indexOf(identifier)).toBeLessThan(
      markup.indexOf(WORKBENCH_TEST_IDS.resultOutput('scaled'))
    );
    const root = makeHookProbe(() => WorkbenchResultView({store}));
    const button = findProbeElement(
      root.read(),
      element => element.props['data-testid'] === identifier
    );
    expect(button.props['title']).toBe(
      'Allow CPU fallback with a warning when the GPU cannot run.'
    );
    invokeProbe(button, 'onClick');
    expect(calls).toHaveLength(1);
    expect(calls[0].slice(0, 2)).toEqual(['edit', 'suggestion:continue-cpu:scale']);
    const apply = calls[0][2] as (current: WorkflowDocument) => WorkflowDocument;
    expect(apply(document).steps[1].onFallback).toBe('warn');
    expect(document.steps[1].onFallback).toBe('stop');
    expect(
      render({...failed, error: {code: 'operation-failed', severity: 'error', message}})
    ).not.toContain('workflow-suggestion-');
  });

  test('enables display and colours a signed column in one edit, retaining other output styles', () => {
    const document = makeFixtureDocument();
    document.steps[1].display = [];
    document.steps[1].displayStyle = {scaled: {colorColumn: 'point_id', colorScale: 'ordinal'}};
    const data = makeFixturePoints(3);
    data.table.columns = {...data.table.columns, value: [-2, 0, 3]};
    const result = {...makeResult(), outputs: {scaled: data}};
    const fixture = makeOutputProbe(result, {document});
    const panel = openColumnPanel(fixture.output, 'value');
    const button = findColourButton(panel.read());
    expect(button.props['children']).toBe('Colour the map by this column');
    expect(button.props['aria-pressed']).toBe(false);
    invokeProbe(button, 'onClick');
    expect(fixture.calls).toHaveLength(1);
    expect(fixture.calls[0].slice(0, 2)).toEqual(['edit', 'display-style:scale:scaled']);
    const apply = fixture.calls[0][2] as (current: WorkflowDocument) => WorkflowDocument;
    const newer = {
      ...document,
      steps: document.steps.map(step =>
        step.id === 'scale'
          ? {
              ...step,
              display: ['other'],
              displayStyle: {...step.displayStyle, other: {colorColumn: 'rows'}}
            }
          : step
      )
    };
    const changed = apply(newer);
    expect(changed.steps[1].display).toEqual(['other', 'scaled']);
    expect(changed.steps[1].displayStyle).toEqual({
      scaled: {colorColumn: 'value', palette: 'diverging'},
      other: {colorColumn: 'rows'}
    });
    expect(newer.steps[1].display).toEqual(['other']);
    expect(document.steps[1].displayStyle).toEqual({
      scaled: {colorColumn: 'point_id', colorScale: 'ordinal'}
    });
    const markup = render(
      {status: 'done', key: 'current', issues: [], result},
      {document: changed}
    );
    expect(markup).toContain('columns · coloured by value');
    expect(markup.indexOf('coloured by value')).toBeLessThan(
      markup.indexOf(WORKBENCH_TEST_IDS.resultTable('scaled'))
    );
  });

  test('removes colouring without retracting the displayed output or clearing another style', () => {
    const document = makeFixtureDocument();
    document.steps[1].displayStyle = {
      scaled: {colorColumn: 'value', palette: 'diverging'},
      other: {colorColumn: 'rows'}
    };
    const fixture = makeOutputProbe(makeResult(), {document});
    const button = findColourButton(openColumnPanel(fixture.output, 'value').read());
    expect(button.props['children']).toBe('Remove colouring');
    expect(button.props['aria-pressed']).toBe(true);
    invokeProbe(button, 'onClick');
    const apply = fixture.calls[0][2] as (current: WorkflowDocument) => WorkflowDocument;
    const changed = apply(document);
    expect(changed.steps[1].display).toEqual(['scaled']);
    expect(changed.steps[1].displayStyle).toEqual({other: {colorColumn: 'rows'}});
    expect(document.steps[1].displayStyle?.['scaled']).toEqual({
      colorColumn: 'value',
      palette: 'diverging'
    });
    expect(render(undefined, {document: changed})).not.toContain('coloured by value');
  });

  test('F3: a hidden output with a retained style offers colouring and enables display', () => {
    const document = makeFixtureDocument();
    document.steps[1].display = [];
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value'}};
    const fixture = makeOutputProbe(makeResult(), {document});
    const button = findColourButton(openColumnPanel(fixture.output, 'value').read());
    expect(button.props['children']).toBe('Colour the map by this column');
    expect(button.props['aria-pressed']).toBe(false);
    invokeProbe(button, 'onClick');
    expect(fixture.calls).toHaveLength(1);
    const apply = fixture.calls[0][2] as (current: WorkflowDocument) => WorkflowDocument;
    const changed = apply(document);
    expect(changed.steps[1].display).toEqual(['scaled']);
    expect(changed.steps[1].displayStyle).toEqual({scaled: {colorColumn: 'value'}});
    expect(document.steps[1].display).toEqual([]);
  });

  test('F3: the edit checks current display state before clearing an existing colour', () => {
    const document = makeFixtureDocument();
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value'}};
    const fixture = makeOutputProbe(makeResult(), {document});
    const button = findColourButton(openColumnPanel(fixture.output, 'value').read());
    expect(button.props['children']).toBe('Remove colouring');
    invokeProbe(button, 'onClick');
    const apply = fixture.calls[0][2] as (current: WorkflowDocument) => WorkflowDocument;
    const hidden = {
      ...document,
      steps: document.steps.map(step => (step.id === 'scale' ? {...step, display: []} : step))
    };
    const changed = apply(hidden);
    expect(changed.steps[1].display).toEqual(['scaled']);
    expect(changed.steps[1].displayStyle).toEqual({scaled: {colorColumn: 'value'}});
    expect(hidden.steps[1].display).toEqual([]);
  });

  test('only offers colouring for non-geometry columns of displayed or publishable outputs', () => {
    const fixture = makeOutputProbe();
    expect(() => findColourButton(openColumnPanel(fixture.output, 'geometry').read())).toThrow(
      'expected element was absent'
    );
    const document = makeFixtureDocument();
    document.steps[1].display = [];
    const unavailable = makeOutputProbe(makeResult(), {document});
    delete unavailable.store.host.publishOutput;
    expect(() => findColourButton(openColumnPanel(unavailable.output, 'value').read())).toThrow(
      'expected element was absent'
    );
    delete fixture.store.host.publishOutput;
    expect(
      findColourButton(openColumnPanel(fixture.output, 'value').read()).props['children']
    ).toBe('Colour the map by this column');
  });
});

describe('focus a published result on the map', () => {
  test('shows an accessible icon beside display and focuses the published dataset', () => {
    const {store, calls} = makeStore({
      status: 'done',
      key: 'current',
      issues: [],
      result: makeResult(),
      published: {
        scaled: {datasetId: 'host-dataset-scaled-17', label: 'Scaled points', key: 'current'}
      }
    });
    const focusOutput = vi.fn();
    store.host.focusOutput = focusOutput;
    const markup = renderToStaticMarkup(createElement(WorkbenchResultView, {store}));
    expect(markup.match(/data-testid="workflow-result-(?:display|focus|download)[^"]*"/g)).toEqual([
      `data-testid="${WORKBENCH_TEST_IDS.resultDisplay('scaled')}"`,
      `data-testid="${WORKBENCH_TEST_IDS.resultFocus('scaled')}"`,
      `data-testid="${WORKBENCH_TEST_IDS.resultDownload('scaled', 'csv')}"`,
      `data-testid="${WORKBENCH_TEST_IDS.resultDownload('scaled', 'geojson')}"`
    ]);
    const buttonMarkup = markup.match(
      /<button[^>]*aria-label="Zoom the map to this output"[^>]*>(.*?)<\/button>/
    );
    expect(buttonMarkup).not.toBeNull();
    expect(buttonMarkup![0]).toContain('title="Zoom the map to this output"');
    expect(buttonMarkup![1]).toContain('<svg aria-hidden="true"');
    expect(focusOutput).not.toHaveBeenCalled();
    const root = makeHookProbe(() => WorkbenchResultView({store}));
    const output = probeComponent(findComponent(root.read(), 'ResultOutput'));
    invokeProbe(findButton(output.read(), 'Zoom the map to this output'), 'onClick');
    expect(focusOutput.mock.calls).toEqual([['host-dataset-scaled-17']]);
    expect(calls).toEqual([]);
  });

  test('omits focus when the output is unpublished or the host lacks focus support', () => {
    const identifier = `data-testid="${WORKBENCH_TEST_IDS.resultFocus('scaled')}"`;
    const unpublished = makeStore();
    unpublished.store.host.focusOutput = vi.fn();
    expect(
      renderToStaticMarkup(createElement(WorkbenchResultView, {store: unpublished.store}))
    ).not.toContain(identifier);
    const published = makeStore({
      status: 'done',
      key: 'current',
      issues: [],
      result: makeResult(),
      published: {scaled: {datasetId: 'published-scaled', label: 'Scaled points', key: 'current'}}
    });
    expect(
      renderToStaticMarkup(createElement(WorkbenchResultView, {store: published.store}))
    ).not.toContain(identifier);
  });
});

test('F1: a failing repair provider leaves the original run error and older result readable', () => {
  const message = 'Fixture scale: Factor failed; change Factor and run again.';
  const {store} = makeStore({
    status: 'error',
    key: 'new',
    issues: [],
    result: makeResult(),
    error: {severity: 'error', code: 'operation-failed', message}
  });
  const originalGet = store.registry.get;
  const operation = originalGet('fixture-scale')!;
  let failures = 0;
  // Fail inside the real provider without mocking shared modules or the React runtime.
  const unavailable = {
    ...operation,
    get inputs(): typeof operation.inputs {
      failures++;
      throw new Error('repair provider failed');
    }
  };
  store.registry.get = identifier =>
    identifier === operation.id ? unavailable : originalGet(identifier);
  const markup = renderToStaticMarkup(createElement(WorkbenchResultView, {store}));
  expect(failures).toBe(1);
  expect(markup).toContain(message);
  expect(markup).toContain('data-status="error"');
  expect(markup).toContain(`data-testid="${WORKBENCH_TEST_IDS.resultTable('scaled')}"`);
  expect(markup).not.toContain('workflow-suggestion-');
  expect(markup).not.toContain('repair provider failed');
});
