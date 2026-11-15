// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {makeWorkflowDot, makeWorkflowRunReport} from '../../src/workflows/workflow-report';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import type {
  WorkflowOperationRegistry,
  WorkflowRunnerState
} from '../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from './workflow-fixtures';

const STARTED_AT = Date.UTC(2026, 10, 14, 12);

async function runFixture(document = makeFixtureDocument(), registry = makeFixtureRegistry()) {
  const runner = new WorkflowRunner({host: makeFixtureHost(), registry, now: () => STARTED_AT});
  runner.setDocument(document);
  try {
    const record = await runner.run();
    return {document, registry, record, state: runner.getState()};
  } finally {
    runner.destroy();
  }
}

const FIXTURE_DOT = `digraph "fixture-workflow" {
  rankdir=LR;
  "source" [label="Fixture source\\nFixture source"];
  "scale" [label="Fixture scale\\nFixture scale"];
  "count" [label="Fixture count\\nFixture count"];
  "source" -> "scale" [label="Data → Points"];
  "scale" -> "count" [label="Scaled → Rows"];
}
`;

const FIXTURE_REPORT = `# Fixture workflow

Run fixture-workflow-1 · done · 3 steps · 0 ms · started 2026-11-14T12:00:00.000Z

## Steps

| # | Step | Operation | Status | Engine | Time | Output rows |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Fixture source | Fixture source | done | Host | 0 ms | data: 4 |
| 2 | Fixture scale | Fixture scale | done | CPU | 0 ms | scaled: 4 |
| 3 | Fixture count | Fixture count | done | CPU | 0 ms | count: 1 |

## Details

### 1. Fixture source

- Operation: fixture-source
- Point count: 4

### 2. Fixture scale

- Operation: fixture-scale
- Points ← Fixture source · Data (4 rows)
- Factor: 3
- Fail: false
- Fallback: No GPU device supplied

### 3. Fixture count

- Operation: fixture-count
- Rows ← Fixture scale · Scaled (4 rows)

## Notes

None.
`;

describe('workflow reports', () => {
  test('pins the report and DOT of a real fixture run', async () => {
    const {document, registry, state, record} = await runFixture();
    expect(record.status).toBe('done');
    expect(record.steps.map(step => step.status)).toEqual(['done', 'done', 'done']);
    expect(makeWorkflowRunReport(document, registry, state, record)).toBe(FIXTURE_REPORT);
    expect(makeWorkflowDot(document, registry)).toBe(FIXTURE_DOT);
    expect(makeWorkflowDot(document, registry, null)).toBe(FIXTURE_DOT);
    expect(makeWorkflowDot(document, registry, state)).toBe(`digraph "fixture-workflow" {
  rankdir=LR;
  "source" [label="Fixture source\\nFixture source\\ndone · 4 rows"];
  "scale" [label="Fixture scale\\nFixture scale\\ndone · 4 rows"];
  "count" [label="Fixture count\\nFixture count\\ndone · 1 rows"];
  "source" -> "scale" [label="Data → Points"];
  "scale" -> "count" [label="Scaled → Rows"];
}
`);
  });

  test('reports an exact failure and the dependent step skipped by a real run', async () => {
    const document = makeFixtureDocument();
    document.steps[1].parameters['fail'] = {from: 'static', value: true};
    const {registry, state, record} = await runFixture(document);
    expect(record.status).toBe('error');
    expect(record.steps.map(step => step.status)).toEqual(['done', 'error', 'skipped']);
    expect(record.steps[1].error?.code).toBe('operation-failed');
    const report = makeWorkflowRunReport(document, registry, state, record);
    expect(report).toContain('| 2 | Fixture scale | Fixture scale | error | — | 0 ms |  |');
    expect(report).toContain('- Error: operation-failed · Fixture scale: asked to fail');
    expect(report).toContain('## Notes\n\n- Fixture count: skipped\n');
    expect(report).not.toContain('- Factor:');
    expect(report).not.toContain('- Fallback:');
  });

  test('uses run parameters and provenance, with inputs, durations, caching and diagnostics', async () => {
    const fixture = await runFixture();
    const document = structuredClone(fixture.document);
    const state = structuredClone(fixture.state);
    const record = structuredClone(fixture.record);
    document.inputs = [
      {id: 'distance', label: 'Distance', kind: 'number', value: 12, unit: 'meters'},
      {id: 'breaks', label: 'Breaks', kind: 'number-list', value: [1, 2, 3]}
    ];
    document.steps[1].parameters['factor'] = {from: 'static', value: 99};
    document.steps[1].inputs = {};
    record.durationMilliseconds = 1800;
    record.steps[0].durationMilliseconds = 42;
    record.steps[1].durationMilliseconds = 1800;
    record.steps[2].cacheHit = true;
    record.steps[1].engine = 'gpu';
    record.steps[1].dispatchedContributors = ['ScaleContributor'];
    state.steps['scale'].result!.parameters['longText'] = 'x'.repeat(121);
    state.steps['scale'].result!.parameters['shortText'] = 'y'.repeat(120);
    state.steps['scale'].result!.parameters['levels'] = [2, 4];
    state.steps['scale'].result!.diagnostics = [
      {severity: 'warning', code: 'dropped-rows', count: 2, message: 'Two rows were omitted.'},
      {severity: 'info', code: 'processed', count: 4, message: 'Processed rows.'}
    ];
    state.issues = [{severity: 'warning', code: 'publish-failed', message: 'Map unavailable.'}];
    state.steps['scale'].issues.push({
      severity: 'warning',
      code: 'gpu-fallback',
      message: 'Scale: fallback.'
    });
    const report = makeWorkflowRunReport(document, fixture.registry, state, record);
    expect(report).toContain('3 steps · 1.8 s · started 2026-11-14T12:00:00.000Z');
    expect(report).toContain('## Inputs\n\n- Distance: 12 meters\n- Breaks: 1, 2, 3');
    expect(report).toContain('| done | Host | 42 ms | data: 4 |');
    expect(report).toContain('| done | GPU | 1.8 s | scaled: 4 |');
    expect(report).toContain('| done | CPU | cached | count: 1 |');
    expect(report).toContain('- Points ← Fixture source · Data (4 rows)');
    expect(report).toContain('- Factor: 3');
    expect(report).not.toContain('- Factor: 99');
    expect(report).toContain('- longText: 121 characters');
    expect(report).toContain(`- shortText: ${'y'.repeat(120)}`);
    expect(report).toContain('- levels: 2, 4');
    expect(report).toContain('- GPU contributors: ScaleContributor');
    expect(report).toContain('- Diagnostic: warning · dropped-rows · 2 · Two rows were omitted.');
    expect(report).toContain('- Diagnostic: info · processed · 4 · Processed rows.');
    expect(report).toContain(
      '## Notes\n\n- Warning: publish-failed · Map unavailable.\n- Warning: gpu-fallback · Scale: fallback.\n- Warning: dropped-rows · Two rows were omitted.'
    );
  });

  test('escapes hostile Markdown labels, table cells, inputs and diagnostic text', async () => {
    const fixture = await runFixture();
    const document = structuredClone(fixture.document);
    const state = structuredClone(fixture.state);
    const record = structuredClone(fixture.record);
    document.title = 'Title | "quoted"\n<script>';
    document.steps[0].label = 'Source | "quoted"\r\nnew \\ path';
    document.steps[1].label = 'Scale | "quoted"\nnew';
    document.inputs = [
      {id: 'name', label: 'Name|\nline', kind: 'string', value: '<img> |\nvalue', unit: 'm|s'}
    ];
    const operations = fixture.registry.list().map(operation => ({
      ...operation,
      title: `${operation.title}|\n"title"`,
      inputs: operation.inputs.map(port => ({...port, label: `${port.label}|\n"in"`})),
      outputs: operation.outputs.map(port => ({...port, label: `${port.label}|\n"out"`}))
    }));
    const registry: WorkflowOperationRegistry = {
      list: () => operations,
      get: operationId => operations.find(operation => operation.id === operationId)
    };
    record.steps[1].outputRows = {'scaled|\n"rows"': 4};
    state.steps['scale'].result!.diagnostics = [
      {severity: 'error', code: 'bad|code', count: 1, message: '<script>\n| "message"'}
    ];
    const report = makeWorkflowRunReport(document, registry, state, record);
    expect(report).toContain('# Title \\| "quoted"\\n&lt;script&gt;\n');
    expect(report).toContain(
      '| 1 | Source \\| "quoted"\\nnew \\\\ path | Fixture source\\|\\n"title" |'
    );
    expect(report).toContain('| scaled\\|\\n"rows": 4 |');
    expect(report).toContain('- Name\\|\\nline: &lt;img&gt; \\|\\nvalue m\\|s');
    expect(report).toContain(
      '- Points\\|\\n"in" ← Source \\| "quoted"\\nnew \\\\ path · Data\\|\\n"out" (4 rows)'
    );
    expect(report).toContain(
      '- Diagnostic: error · bad\\|code · 1 · &lt;script&gt;\\n\\| "message"'
    );
    expect(report).not.toContain('<script>');
    expect(report.split('\n').filter(line => line.startsWith('|'))).toHaveLength(5);
  });

  test('does not present a stale result as evidence for an older run', async () => {
    const {document, registry, record, state: originalState} = await runFixture();
    const state = structuredClone(originalState);
    state.steps['scale'].result!.key = 'a-different-run';
    state.steps['scale'].result!.parameters['factor'] = 98;
    const report = makeWorkflowRunReport(document, registry, state, record);
    expect(report).not.toContain('- Factor:');
    expect(report).toContain('- Fallback: No GPU device supplied');
    expect(report).toContain(
      '| 2 | Fixture scale | Fixture scale | done | CPU | 0 ms | scaled: 4 |'
    );
  });

  test('handles cancellations, missing operations and an empty run', async () => {
    const {document, state: originalState, record: originalRecord} = await runFixture();
    const registry: WorkflowOperationRegistry = {list: () => [], get: () => undefined};
    const record = structuredClone(originalRecord);
    record.status = 'cancelled';
    record.steps[1].status = 'cancelled';
    record.steps[2].status = 'skipped';
    const state: WorkflowRunnerState = {...originalState, steps: {}};
    const report = makeWorkflowRunReport(document, registry, state, record);
    expect(report).toContain('- points ← source · data (4 rows)');
    expect(report).toContain('## Notes\n\n- scale: cancelled\n- count: skipped\n');
    expect(
      makeWorkflowRunReport({...document, steps: []}, registry, state, {...record, steps: []})
    ).toBe(`# Fixture workflow

Run fixture-workflow-1 · cancelled · 0 steps · 0 ms · started 2026-11-14T12:00:00.000Z

## Steps

| # | Step | Operation | Status | Engine | Time | Output rows |
| --- | --- | --- | --- | --- | --- | --- |

## Details

## Notes

None.
`);
  });
});

describe('workflow DOT', () => {
  test('quotes hostile ids and labels, including ports, and marks disabled nodes', () => {
    const registry = makeFixtureRegistry();
    const operations = registry.list().map(operation => ({
      ...operation,
      title: `${operation.title} "\\\nline`,
      inputs: operation.inputs.map(port => ({...port, label: 'In "\\\nport'})),
      outputs: operation.outputs.map(port => ({...port, label: 'Out "\\\nport'}))
    }));
    const hostileRegistry: WorkflowOperationRegistry = {
      list: () => operations,
      get: operationId => operations.find(operation => operation.id === operationId)
    };
    const document = makeFixtureDocument();
    document.id = 'graph"\\\nend';
    document.steps = document.steps.slice(0, 2);
    document.steps[0].id = 'source"\\\nnode';
    document.steps[0].label = 'Label | "\\\nline';
    document.steps[1].inputs['points'].stepId = document.steps[0].id;
    document.steps[1].disabled = true;
    expect(makeWorkflowDot(document, hostileRegistry)).toBe(String.raw`digraph "graph\"\\\nend" {
  rankdir=LR;
  "source\"\\\nnode" [label="Label | \"\\\nline\nFixture source \"\\\nline"];
  "scale" [label="Fixture scale \"\\\nline\nFixture scale \"\\\nline", style=dashed];
  "source\"\\\nnode" -> "scale" [label="Out \"\\\nport → In \"\\\nport"];
}
`);
  });

  test('orders topologically with id ties regardless of insertion order and canvas position', () => {
    const registry = makeFixtureRegistry();
    const document = makeFixtureDocument();
    const branch = {...document.steps[0], id: 'another-source', position: {x: 999, y: 999}};
    document.steps = [document.steps[2], branch, document.steps[1], document.steps[0]];
    const expected = FIXTURE_DOT.replace(
      '  "source" [',
      '  "another-source" [label="Fixture source\\nFixture source"];\n  "source" ['
    );
    expect(makeWorkflowDot(document, registry)).toBe(expected);
    document.steps.reverse();
    document.steps.forEach(step => {
      step.position = {x: -100, y: 5};
    });
    expect(makeWorkflowDot(document, registry)).toBe(expected);
  });

  test('handles unknown operations, missing state and zero-row outputs', async () => {
    const {document, registry, state: originalState} = await runFixture();
    const state = structuredClone(originalState);
    document.steps[2].operationId = 'removed-operation';
    document.steps[2].disabled = true;
    delete state.steps['count'];
    state.steps['scale'].status = 'stale';
    state.steps['scale'].result!.outputs['scaled'].table.rowCount = 0;
    const dot = makeWorkflowDot(document, registry, state);
    expect(dot).toContain('"scale" [label="Fixture scale\\nFixture scale\\nstale · 0 rows"];');
    expect(dot).toContain(
      '"count" [label="removed-operation\\nremoved-operation\\nnot-configured", style=dashed];'
    );
    expect(dot).toContain('"scale" -> "count" [label="Scaled → rows"];');
  });
});
