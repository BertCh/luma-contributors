// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolve} from 'node:path';
import {runInThisContext} from 'node:vm';
import ts from 'typescript';
import {describe, expect, test} from 'vitest';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {getCatalogBinding, serializeCatalogInvocation} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {
  parseWorkflowDocument,
  serializeWorkflowDocument
} from '../../src/workflows/workflow-document';
import {
  describeWorkflow,
  exportWorkflowJSON,
  exportWorkflowTypeScript,
  importWorkflowJSON,
  makeWorkflowFromCatalogHistory,
  WORKFLOW_JSON_LIMITS,
  type WorkflowHistoryInputReference,
  type WorkflowHistoryRecord
} from '../../src/workflows/workflow-export';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {planWorkflow} from '../../src/workflows/workflow-plan';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowDocument,
  type WorkflowHost,
  type WorkflowOperationRegistry
} from '../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from './workflow-fixtures';

function expectWorkflowError(action: () => unknown, code: WorkflowError['code']): WorkflowError {
  try {
    action();
  } catch (error) {
    expect(error).toBeInstanceOf(WorkflowError);
    expect((error as WorkflowError).code).toBe(code);
    return error as WorkflowError;
  }
  throw new Error(`Expected WorkflowError with code ${code}`);
}

type GeneratedFactory = (
  host: WorkflowHost,
  inputs?: Record<string, CatalogParameterValue>
) => WorkflowRunner;

function loadExport(text: string): GeneratedFactory {
  const result = ts.transpileModule(text, {
    reportDiagnostics: true,
    compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022}
  });
  expect(result.diagnostics).toEqual([]);
  const exported: Record<string, GeneratedFactory> = {};
  const requireModule = (name: string) => {
    expect(name).toBe('@luma-contributors/kepler');
    return {WorkflowRunner, getWorkflowOperationRegistry, parseWorkflowDocument};
  };
  // Keep the VM in the current realm: the strict parser refuses foreign object prototypes.
  const execute = runInThisContext(`(function(require, exports) {${result.outputText}\n})`) as (
    requireModule: (name: string) => unknown,
    exported: object
  ) => void;
  execute(requireModule, exported);
  expect(Object.keys(exported)).toHaveLength(1);
  return Object.values(exported)[0];
}

function checkGeneratedModules(documents: readonly WorkflowDocument[]): string[] {
  const configurationPath = resolve('tsconfig.json');
  const configuration = ts.readConfigFile(configurationPath, ts.sys.readFile);
  if (configuration.error)
    throw new Error(ts.flattenDiagnosticMessageText(configuration.error.messageText, '\n'));
  const parsed = ts.parseJsonConfigFileContent(configuration.config, ts.sys, resolve('.'));
  const options = {...parsed.options, noEmit: true, incremental: false};
  const sources = new Map(
    documents.map((document, index) => [
      resolve(`.typecheck/kewf/generated-export-${index}.ts`),
      exportWorkflowTypeScript(document, makeFixtureRegistry())
    ])
  );
  const host = ts.createCompilerHost(options);
  const readSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreateNewSourceFile) => {
    const text = sources.get(fileName);
    return text === undefined
      ? readSourceFile(fileName, languageVersion, onError, shouldCreateNewSourceFile)
      : ts.createSourceFile(fileName, text, languageVersion, true);
  };
  const program = ts.createProgram([...sources.keys()], options, host);
  // Resolve the real package API; other streams' implementation diagnostics belong to their gates.
  return ts
    .getPreEmitDiagnostics(program)
    .filter(diagnostic => !diagnostic.file || sources.has(diagnostic.file.fileName))
    .map(
      diagnostic =>
        `TS${diagnostic.code}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
    );
}

function makeDocumentAtJSONSize(characters: number, special?: string): WorkflowDocument {
  const document = makeFixtureDocument();
  const emptyCollection = '{"type":"FeatureCollection","features":[]}';
  document.steps = Array.from({length: 4}, (_, index) => ({
    id: `source-${index}`,
    operationId: 'source-geojson',
    inputs: {},
    parameters: {text: {from: 'static', value: emptyCollection}},
    position: {x: 0, y: 0}
  }));
  if (special !== undefined) {
    document.steps[0].parameters['special'] = {from: 'static', value: special};
  }
  let remaining = characters - JSON.stringify(serializeWorkflowDocument(document)).length;
  for (const step of document.steps) {
    const padding = Math.min(
      remaining,
      WORKFLOW_LIMITS.maximumTextCharacters - emptyCollection.length
    );
    step.parameters['text'] = {from: 'static', value: emptyCollection + ' '.repeat(padding)};
    remaining -= padding;
  }
  expect(remaining).toBe(0);
  return parseWorkflowDocument(document);
}

function makeInputDocument(): WorkflowDocument {
  const document = makeFixtureDocument();
  document.inputs = [
    {
      id: 'distance',
      label: 'Distance',
      kind: 'number',
      value: 3,
      unit: 'm',
      minimum: 0,
      maximum: 100
    },
    {id: 'whole-count', label: 'Count', kind: 'integer', value: 4},
    {id: 'enabled', label: 'Enabled', kind: 'boolean', value: true},
    {id: 'mode', label: 'Mode', kind: 'enum', value: 'fast', options: ['fast', 'careful']},
    {id: 'name', label: 'Name', kind: 'string', value: 'analysis'},
    {id: 'bands', label: 'Bands', kind: 'number-list', value: [1, 2]}
  ];
  document.steps[1].parameters['factor'] = {from: 'input', inputId: 'distance'};
  return document;
}

const POINT_FEATURE = {
  type: 'Feature',
  geometry: {type: 'Point', coordinates: [0, 0]},
  properties: {value: 1}
};
const POINTS: CatalogFeatureCollection = {
  type: 'FeatureCollection',
  features: [POINT_FEATURE]
};
const POINT_TEXT = JSON.stringify(POINTS);

function makeHistoryRecord(
  id: string,
  operationId: string,
  reference: WorkflowHistoryInputReference,
  parameters: Record<string, CatalogParameterValue> = {}
): WorkflowHistoryRecord {
  const binding = getCatalogBinding(operationId);
  if (!binding) throw new Error(`Missing catalog binding ${operationId}`);
  return {
    id,
    operationId,
    invocation: serializeCatalogInvocation(
      binding,
      parameters,
      Object.fromEntries(binding.inputs.map(input => [input.name, POINTS]))
    ),
    inputs: Object.fromEntries(binding.inputs.map(input => [input.name, reference])),
    engine: 'auto'
  };
}

function findStep(document: WorkflowDocument, stepId: string) {
  const step = document.steps.find(candidate => candidate.id === stepId);
  if (!step) throw new Error(`Missing step ${stepId}`);
  return step;
}

describe('workflow JSON export', () => {
  test('round trips a document at the compact character limit despite pretty-print overhead', () => {
    const document = makeDocumentAtJSONSize(WORKFLOW_LIMITS.maximumDocumentCharacters);
    expect(JSON.stringify(document).length).toBe(WORKFLOW_LIMITS.maximumDocumentCharacters);
    const text = exportWorkflowJSON(document);
    expect(text).toBe(`${JSON.stringify(serializeWorkflowDocument(document), null, 2)}\n`);
    expect(text.length).toBeGreaterThan(WORKFLOW_LIMITS.maximumDocumentCharacters);
    expect(importWorkflowJSON(text)).toEqual(document);
    expect(exportWorkflowJSON(importWorkflowJSON(text))).toBe(text);
  });

  test('budgets formatting separately while retaining whitespace and escapes inside values', () => {
    const special = '\"\\\n\t😀';
    const document = makeDocumentAtJSONSize(WORKFLOW_LIMITS.maximumDocumentCharacters, special);
    expect(JSON.stringify(document).length).toBe(WORKFLOW_LIMITS.maximumDocumentCharacters);
    const text = exportWorkflowJSON(document);
    expect(text.length).toBeGreaterThan(WORKFLOW_LIMITS.maximumDocumentCharacters);
    expect(importWorkflowJSON(text)).toEqual(document);
    const oversized = text.replace(JSON.stringify(special), JSON.stringify(`${special}😀`));
    const error = expectWorkflowError(() => importWorkflowJSON(oversized), 'limit-exceeded');
    expect(error.message).toBe(
      `Workflow JSON: maximumDocumentCharacters is ${WORKFLOW_LIMITS.maximumDocumentCharacters}; got ${WORKFLOW_LIMITS.maximumDocumentCharacters + 2}.`
    );
  });

  test('refuses formatting beyond its named limit before attempting to parse', () => {
    const count = WORKFLOW_JSON_LIMITS.maximumFormattedCharacters + 1;
    const error = expectWorkflowError(
      () => importWorkflowJSON(' '.repeat(count)),
      'limit-exceeded'
    );
    expect(error.message).toBe(
      `Workflow JSON: maximumFormattedCharacters is ${WORKFLOW_JSON_LIMITS.maximumFormattedCharacters}; got ${count}.`
    );
    const document = makeFixtureDocument();
    expect(importWorkflowJSON(exportWorkflowJSON(document).padEnd(count - 1))).toEqual(document);
  });

  test('does not join invalid JSON tokens when counting compact characters', () => {
    const text = '[1 2]'.padEnd(WORKFLOW_LIMITS.maximumDocumentCharacters + 1);
    const error = expectWorkflowError(() => importWorkflowJSON(text), 'malformed');
    expect(error.message).toMatch(/^Workflow JSON is not valid JSON: /);
  });

  test('round trips canonical JSON and preserves workflow inputs', () => {
    const document = makeInputDocument();
    const original = structuredClone(document);
    const text = exportWorkflowJSON(document);
    expect(text).toBe(`${JSON.stringify(serializeWorkflowDocument(document), null, 2)}\n`);
    expect(importWorkflowJSON(text)).toEqual(document);
    expect(document).toEqual(original);
    expect(exportWorkflowJSON(importWorkflowJSON(text))).toBe(text);
  });

  test('reports JSON syntax errors with their parse reason', () => {
    let reason = '';
    try {
      JSON.parse('{');
    } catch (error) {
      reason = (error as Error).message;
    }
    const error = expectWorkflowError(() => importWorkflowJSON('{'), 'malformed');
    expect(error.message).toBe(`Workflow JSON is not valid JSON: ${reason}.`);
    expect(error.cause).toBeInstanceOf(SyntaxError);
  });

  test('refuses oversized text before parsing and accepts the exact character limit', () => {
    const error = expectWorkflowError(
      () => importWorkflowJSON('['.repeat(WORKFLOW_LIMITS.maximumDocumentCharacters + 1)),
      'limit-exceeded'
    );
    expect(error.message).toBe(
      `Workflow JSON: maximumDocumentCharacters is ${WORKFLOW_LIMITS.maximumDocumentCharacters}; got ${WORKFLOW_LIMITS.maximumDocumentCharacters + 1}.`
    );
    const document = makeFixtureDocument();
    expect(
      importWorkflowJSON(
        exportWorkflowJSON(document).padEnd(WORKFLOW_LIMITS.maximumDocumentCharacters)
      )
    ).toEqual(document);
  });

  test('delegates structural refusals to the strict parser', () => {
    expectWorkflowError(() => importWorkflowJSON('null'), 'malformed');
    const document = makeFixtureDocument();
    expectWorkflowError(
      () => importWorkflowJSON(JSON.stringify({...document, unexpected: true})),
      'malformed'
    );
    expectWorkflowError(
      () =>
        importWorkflowJSON(
          JSON.stringify({...document, steps: [document.steps[0], document.steps[0]]})
        ),
      'duplicate-id'
    );
    expectWorkflowError(
      () => importWorkflowJSON(JSON.stringify({...document, schemaVersion: 2})),
      'malformed'
    );
    expectWorkflowError(
      () =>
        importWorkflowJSON(JSON.stringify({...document, steps: Array(65).fill(document.steps[0])})),
      'limit-exceeded'
    );
  });
});

describe('workflow outline', () => {
  test('pins the fixture outline in topological order and omits explicit defaults', () => {
    const document = makeFixtureDocument();
    document.steps[0].parameters['count'] = {from: 'static', value: 4};
    document.steps[1].parameters['fail'] = {from: 'static', value: false};
    document.steps.reverse();
    expect(describeWorkflow(document, makeFixtureRegistry())).toBe(
      'Fixture workflow: 3 steps.\n' +
        '1. Fixture source — Fixture source\n' +
        '2. Fixture scale — Fixture scale\n' +
        '  reads Points from Fixture source · Data\n' +
        '  Factor: 3\n' +
        '  shown on the map\n' +
        '3. Fixture count — Fixture count\n' +
        '  reads Rows from Fixture scale · Scaled\n'
    );
  });

  test('pins workflow inputs, unknown operations, long text and disabled steps', () => {
    const document = makeFixtureDocument();
    document.inputs = [{id: 'distance', label: 'Distance', kind: 'number', value: 5, unit: 'm'}];
    document.steps = [
      {
        ...document.steps[0],
        label: 'Missing step',
        operationId: 'removed-operation',
        disabled: true,
        parameters: {
          factor: {from: 'input', inputId: 'distance'},
          text: {from: 'static', value: 'x'.repeat(121)}
        }
      }
    ];
    expect(describeWorkflow(document, makeFixtureRegistry())).toBe(
      'Fixture workflow: 1 steps.\n' +
        '  Distance = 5 m\n' +
        '1. Missing step — unknown operation "removed-operation" [disabled]\n' +
        '  factor: workflow input "Distance"\n' +
        '  text: 121 characters\n'
    );
  });

  test('formats array, boolean and string values and compares array defaults by value', () => {
    const document = makeFixtureDocument();
    document.steps = [document.steps[0]];
    document.steps[0].parameters = {
      array: {from: 'static', value: [1, 2]},
      changed: {from: 'static', value: [2, 3]},
      enabled: {from: 'static', value: true},
      name: {from: 'static', value: 'a "quoted" name'}
    };
    const operation = makeFixtureRegistry().list()[0];
    const customized = {
      ...operation,
      parameters: [
        {name: 'array', label: 'Array', kind: 'number-list' as const, defaultValue: [1, 2]}
      ]
    };
    const registry: WorkflowOperationRegistry = {get: () => customized, list: () => [customized]};
    expect(describeWorkflow(document, registry)).toBe(
      'Fixture workflow: 1 steps.\n' +
        '1. Fixture source — Fixture source\n' +
        '  changed: [2,3]\n' +
        '  enabled: true\n' +
        '  name: "a \\"quoted\\" name"\n'
    );
  });
});

describe('workflow TypeScript export', () => {
  test('fully typechecks generated modules including a sole constructor input', () => {
    const constructorDocument = makeFixtureDocument();
    constructorDocument.inputs = [
      {id: 'constructor', label: 'Constructor', kind: 'number', value: 1}
    ];
    const reservedDocument = makeFixtureDocument();
    reservedDocument.title = '123';
    reservedDocument.inputs = [
      'constructor',
      'constructor2',
      'to-string',
      'value-of',
      'has-own-property',
      'is-prototype-of',
      'property-is-enumerable',
      'to-locale-string',
      'a-b',
      'a--b',
      'a-b2',
      '123'
    ].map(id => ({id, label: id, kind: 'number', value: 1}));
    const hostile = '\"quotes\" `backticks` ${injection} */\n</script>';
    const hostileDocument = makeInputDocument();
    hostileDocument.title = hostile;
    hostileDocument.inputs[3] = {
      id: 'mode',
      label: hostile,
      kind: 'enum',
      value: hostile,
      options: [hostile]
    };
    expect(
      checkGeneratedModules([
        makeFixtureDocument(),
        makeInputDocument(),
        constructorDocument,
        reservedDocument,
        hostileDocument
      ])
    ).toEqual([]);
  }, 60000);

  test('transpiles, rebuilds a runner and accepts overridden workflow inputs', async () => {
    const document = makeInputDocument();
    const text = exportWorkflowTypeScript(document, makeFixtureRegistry());
    expect(text).toBe(exportWorkflowTypeScript(document, makeFixtureRegistry()));
    expect(text.endsWith('\n')).toBe(true);
    expect(text).toContain('export function createFixtureWorkflowWorkflow(');
    expect(text).toContain('"wholeCount"?: number;');
    expect(text).toContain('"mode"?: "fast" | "careful";');
    expect(text).toContain('"bands"?: number[];');
    expect(text).toContain('Distance Unit: m. Minimum: 0. Maximum: 100. Default: 3.');
    const createWorkflow = loadExport(text);
    const runner = createWorkflow(makeFixtureHost());
    expect(runner).toBeInstanceOf(WorkflowRunner);
    expect(runner.document).toEqual(document);
    const overridden = createWorkflow(makeFixtureHost(), {
      distance: 0,
      wholeCount: 8,
      enabled: false,
      mode: 'careful',
      name: 'override',
      bands: [4, 8]
    });
    expect(overridden.document?.inputs.map(input => input.value)).toEqual([
      0,
      8,
      false,
      'careful',
      'override',
      [4, 8]
    ]);
    expect(overridden.document?.steps).toEqual(document.steps);
    await runner.destroy();
    await overridden.destroy();
  });

  test('emits topological step order with comments', async () => {
    const document = makeFixtureDocument();
    document.steps.reverse();
    const text = exportWorkflowTypeScript(document, makeFixtureRegistry());
    expect(text.indexOf('// Fixture source')).toBeLessThan(text.indexOf('// Fixture scale'));
    expect(text.indexOf('// Fixture scale')).toBeLessThan(text.indexOf('// Fixture count'));
    const runner = loadExport(text)(makeFixtureHost());
    expect(runner.document?.steps.map(step => step.id)).toEqual(['source', 'scale', 'count']);
    expect(runner.document).toEqual(makeFixtureDocument());
    await runner.destroy();
  });

  test('hostile strings remain literal data in source, comments and enum types', async () => {
    const hostile = '"quotes" `backticks` ${injection} */\n</script>';
    const document = makeFixtureDocument();
    document.title = hostile;
    document.inputs = [
      {id: 'mode', label: hostile, unit: hostile, kind: 'enum', value: hostile, options: [hostile]}
    ];
    document.steps[0].label = hostile;
    document.steps[0].note = hostile;
    document.steps[0].parameters['text'] = {from: 'static', value: hostile};
    const operation = makeFixtureRegistry().list()[0];
    const registry: WorkflowOperationRegistry = {
      list: () => [{...operation, title: hostile}],
      get: () => ({...operation, title: hostile})
    };
    const text = exportWorkflowTypeScript(document, registry);
    const runner = loadExport(text)(makeFixtureHost());
    expect(runner.document).toEqual(document);
    expect(text).toContain(JSON.stringify(hostile));
    expect(text).toContain('* /');
    await runner.destroy();
  });

  test('uses valid, distinct names for punctuation, leading digits and colliding input ids', async () => {
    const document = makeFixtureDocument();
    document.title = '!!!';
    document.inputs = ['a-b', 'a--b', 'a-b2', '123', 'constructor'].map(id => ({
      id,
      label: id,
      kind: 'number',
      value: 1
    }));
    const text = exportWorkflowTypeScript(document, makeFixtureRegistry());
    expect(text).toContain('createUntitledWorkflow');
    expect(text).toContain('"input123"?: number');
    const createWorkflow = loadExport(text);
    const runner = createWorkflow(makeFixtureHost());
    expect(runner.document).toEqual(document);
    const overridden = createWorkflow(makeFixtureHost(), {
      aB: 2,
      aB2: 3,
      aB22: 4,
      input123: 5,
      constructor2: 6
    });
    expect(overridden.document?.inputs.map(input => input.value)).toEqual([2, 3, 4, 5, 6]);
    await runner.destroy();
    await overridden.destroy();
  });

  test('reserves inherited property names and resolves resulting collisions deterministically', async () => {
    const document = makeFixtureDocument();
    document.inputs = ['constructor', 'constructor2', 'to-string', 'to-string2', 'value-of'].map(
      id => ({
        id,
        label: id,
        kind: 'number',
        value: 1
      })
    );
    const text = exportWorkflowTypeScript(document, makeFixtureRegistry());
    expect(text).toBe(exportWorkflowTypeScript(document, makeFixtureRegistry()));
    expect(text).toContain('"constructor2"?: number;');
    expect(text).toContain('"constructor22"?: number;');
    expect(text).toContain('"toString2"?: number;');
    expect(text).toContain('"toString22"?: number;');
    expect(text).toContain('"valueOf2"?: number;');
    const createWorkflow = loadExport(text);
    const runner = createWorkflow(makeFixtureHost());
    expect(runner.document).toEqual(document);
    const overridden = createWorkflow(makeFixtureHost(), {
      constructor2: 2,
      constructor22: 3,
      toString2: 4,
      toString22: 5,
      valueOf2: 6
    });
    expect(overridden.document?.inputs.map(input => input.value)).toEqual([2, 3, 4, 5, 6]);
    await runner.destroy();
    await overridden.destroy();
  });
});

describe('catalog history promotion', () => {
  test('shares text sources across operations and records only non-default parameters', () => {
    const records = [
      {
        ...makeHistoryRecord(
          'buffer',
          'geometry-point-buffer',
          {mode: 'text', text: POINT_TEXT},
          {radius: 25}
        ),
        engine: 'cpu' as const
      },
      makeHistoryRecord('measures', 'geometry-measures', {mode: 'text', text: POINT_TEXT})
    ];
    const original = structuredClone(records);
    const document = makeWorkflowFromCatalogHistory(records, {
      registry: getWorkflowOperationRegistry(),
      takenIds: ['promoted-from-history', 'promoted-from-history-2']
    });
    expect(document.id).toBe('promoted-from-history-3');
    expect(document.title).toBe('Promoted from history');
    expect(document.steps).toHaveLength(3);
    const source = document.steps[0];
    expect(source).toEqual({
      id: 'geojson-1',
      operationId: 'source-geojson',
      label: 'GeoJSON 1',
      inputs: {},
      parameters: {text: {from: 'static', value: POINT_TEXT}},
      position: {x: 0, y: 0}
    });
    expect(findStep(document, 'buffer').parameters).toEqual(
      Object.fromEntries(
        Object.entries(JSON.parse(records[0].invocation).parameters).map(([name, value]) => [
          name,
          {from: 'static', value}
        ])
      )
    );
    expect(findStep(document, 'buffer').engine).toBe('cpu');
    expect(findStep(document, 'measures').parameters).toEqual(
      Object.fromEntries(
        Object.entries(JSON.parse(records[1].invocation).parameters).map(([name, value]) => [
          name,
          {from: 'static', value}
        ])
      )
    );
    expect(findStep(document, 'measures').engine).toBeUndefined();
    expect(findStep(document, 'buffer').inputs).toEqual({
      points: {stepId: source.id, output: 'data'}
    });
    expect(findStep(document, 'measures').inputs).toEqual({
      features: {stepId: source.id, output: 'data'}
    });
    expect(findStep(document, 'buffer').position).toEqual({x: 280, y: 0});
    expect(findStep(document, 'measures').position).toEqual({x: 280, y: 150});
    expect(records).toEqual(original);
    expect(
      makeWorkflowFromCatalogHistory(records, {registry: getWorkflowOperationRegistry()})
    ).toEqual({
      ...document,
      id: 'promoted-from-history'
    });
  });

  test('resolves sample text, shares identical text, and leaves unavailable samples unconnected', () => {
    const records = [
      makeHistoryRecord('sample', 'geometry-point-buffer', {mode: 'sample'}),
      makeHistoryRecord('same-text', 'geometry-measures', {mode: 'text', text: POINT_TEXT}),
      makeHistoryRecord('missing', 'geometry-measures', {mode: 'sample'})
    ];
    const requests: string[] = [];
    const document = makeWorkflowFromCatalogHistory(records, {
      registry: getWorkflowOperationRegistry(),
      getSampleText: (operationId, inputName) => {
        requests.push(`${operationId}:${inputName}`);
        return operationId === 'geometry-point-buffer' ? POINT_TEXT : undefined;
      }
    });
    expect(requests).toEqual(['geometry-point-buffer:points', 'geometry-measures:features']);
    expect(document.steps.filter(step => step.operationId === 'source-geojson')).toHaveLength(1);
    expect(document.steps[0].label).toBe('Sample: Points');
    expect(findStep(document, 'sample').inputs['points']).toEqual(
      findStep(document, 'same-text').inputs['features']
    );
    expect(findStep(document, 'missing').inputs).toEqual({});
    const unavailable = makeWorkflowFromCatalogHistory([records[0]], {
      registry: getWorkflowOperationRegistry()
    });
    expect(unavailable.steps).toHaveLength(1);
    expect(unavailable.steps[0].inputs).toEqual({});
  });

  test('shares native datasets by id and geometry field, keeping different fields separate', () => {
    const document = makeWorkflowFromCatalogHistory(
      [
        makeHistoryRecord('first', 'geometry-point-buffer', {
          mode: 'native',
          datasetId: 'roads',
          field: 'geometry'
        }),
        makeHistoryRecord('second', 'geometry-measures', {
          mode: 'native',
          datasetId: 'roads',
          field: 'geometry'
        }),
        makeHistoryRecord('third', 'geometry-measures', {
          mode: 'native',
          datasetId: 'roads',
          field: 'other'
        })
      ],
      {registry: getWorkflowOperationRegistry(), title: 'Dataset runs'}
    );
    expect(document.title).toBe('Dataset runs');
    expect(document.id).toBe('dataset-runs');
    const sources = document.steps.filter(step => step.operationId === 'source-dataset');
    expect(sources).toHaveLength(2);
    expect(sources.map(source => source.label)).toEqual(['Dataset roads', 'Dataset roads']);
    expect(sources.map(source => source.parameters)).toEqual([
      {
        dataset: {from: 'static', value: 'roads'},
        geometryMode: {from: 'static', value: 'geojson-field'},
        geometryField: {from: 'static', value: 'geometry'}
      },
      {
        dataset: {from: 'static', value: 'roads'},
        geometryMode: {from: 'static', value: 'geojson-field'},
        geometryField: {from: 'static', value: 'other'}
      }
    ]);
    expect(findStep(document, 'first').inputs['points']).toEqual(
      findStep(document, 'second').inputs['features']
    );
    expect(findStep(document, 'third').inputs['features'].stepId).not.toBe(sources[0].id);
    expect(parseWorkflowDocument(serializeWorkflowDocument(document))).toEqual(document);
  });

  test('skips invalid or unknown runs and takes the operation id from the invocation', () => {
    const record = makeHistoryRecord('valid', 'geometry-measures', {mode: 'sample'});
    const broken = {...record, id: 'broken', invocation: '{'};
    const unknown = {
      ...record,
      id: 'unknown',
      invocation: JSON.stringify({schemaVersion: 1, operationId: 'gone'})
    };
    const invalidParameters = makeHistoryRecord('invalid-parameters', 'geometry-point-buffer', {
      mode: 'sample'
    });
    invalidParameters.invocation = invalidParameters.invocation.replace(
      '"radius":100',
      '"radius":-1'
    );
    const mismatched = {...record, operationId: 'not-the-operation'};
    const document = makeWorkflowFromCatalogHistory(
      [broken, unknown, invalidParameters, mismatched],
      {registry: getWorkflowOperationRegistry()}
    );
    expect(document.steps.map(step => step.operationId)).toEqual(['geometry-measures']);
    const registry: WorkflowOperationRegistry = {list: () => [], get: () => undefined};
    expectWorkflowError(() => makeWorkflowFromCatalogHistory([record], {registry}), 'malformed');
    expectWorkflowError(
      () =>
        makeWorkflowFromCatalogHistory([broken, unknown], {
          registry: getWorkflowOperationRegistry()
        }),
      'malformed'
    );
    expectWorkflowError(() => makeWorkflowFromCatalogHistory([], {registry}), 'malformed');
  });

  test('fills the exact step limit with a shared source and stops before an overflowing record', () => {
    const records = Array.from({length: WORKFLOW_LIMITS.maximumSteps}, (_, index) =>
      makeHistoryRecord(`run-${index}`, 'geometry-point-buffer', {mode: 'text', text: POINT_TEXT})
    );
    const document = makeWorkflowFromCatalogHistory(records, {
      registry: getWorkflowOperationRegistry()
    });
    expect(document.steps).toHaveLength(WORKFLOW_LIMITS.maximumSteps);
    expect(document.steps.filter(step => step.operationId === 'source-geojson')).toHaveLength(1);
    expect(document.steps.at(-1)?.id).toBe('run-62');
  });

  test('does not add orphan sources or process later records after reaching the limit', () => {
    const records = Array.from({length: 63}, (_, index) =>
      makeHistoryRecord(`run-${index}`, 'geometry-measures', {mode: 'sample'})
    );
    records.push(
      makeHistoryRecord('overflow', 'geometry-point-buffer', {mode: 'text', text: POINT_TEXT})
    );
    records.push(makeHistoryRecord('later', 'geometry-measures', {mode: 'sample'}));
    const document = makeWorkflowFromCatalogHistory(records, {
      registry: getWorkflowOperationRegistry()
    });
    expect(document.steps).toHaveLength(63);
    expect(document.steps.map(step => step.id)).toEqual(
      Array.from({length: 63}, (_, index) => `run-${index}`)
    );
    expect(document.steps.some(step => step.operationId === 'source-geojson')).toBe(false);
  });
});

test('history promotion preserves constants as explicit density settings through planning and execution', async () => {
  const registry = getWorkflowOperationRegistry();
  const record = makeHistoryRecord(
    'density',
    'kernel-density-grid',
    {mode: 'text', text: POINT_TEXT},
    {
      cellSize: 100,
      bandwidth: 500,
      coordinateSystem: 'planar'
    }
  );
  const document = makeWorkflowFromCatalogHistory([record], {registry});
  const parameters = JSON.parse(record.invocation).parameters;
  expect(findStep(document, 'density').parameters).toEqual(
    Object.fromEntries(
      Object.entries(parameters).map(([name, value]) => [name, {from: 'static', value}])
    )
  );
  const host = makeFixtureHost();
  const planned = planWorkflow(document, {registry, host}).steps['density'];
  expect(planned.parameters).toMatchObject({cellSize: 100, bandwidth: 500});
  const runner = new WorkflowRunner({registry, host});
  runner.setDocument(document);
  try {
    await runner.run();
    const state = runner.getState().steps['density'];
    expect(state.status, JSON.stringify(state.issues)).toBe('done');
    expect(state.result?.parameters).toMatchObject({cellSize: 100, bandwidth: 500});
  } finally {
    await runner.destroy();
  }
});
