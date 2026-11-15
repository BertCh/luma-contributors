// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="node" />

import {mkdtemp, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {afterEach, describe, expect, test, vi} from 'vitest';
import {
  parseWorkflowCollection,
  serializeWorkflowCollection
} from '../../../src/workflows/workflow-document';
import {WorkflowRunner} from '../../../src/workflows/workflow-runner';
import {disconnectWorkflowInput} from '../../../src/workflows/workflow-edit';
import type {WorkflowCollection} from '../../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from '../workflow-fixtures';

type Point = {x: number; y: number};
type Bounds = {left: number; right: number; top: number; bottom: number};
type Frame = {canvas: Bounds; node: Bounds; transform: string};
type ColorSample = {foreground: number[]; background: number[]; opacities: number[]};
type Gesture = {
  destination: Point;
  destinationHit: {matched: boolean; expected: string};
};
type AuditProblem = {surface: string; kind: string; message?: string; controls?: unknown[]};
type AccessibilityHelpers = {
  checkNames(
    page: unknown,
    root: unknown,
    context: string,
    problems?: AuditProblem[]
  ): Promise<number>;
  checkContrast(
    root: unknown,
    context: string,
    selector: string,
    problems?: AuditProblem[]
  ): Promise<ColorSample[]>;
  calculateContrast(foreground: number[], background: number[]): number;
  readContrastSamples(elements: unknown[]): ColorSample[];
  auditResultInspector(
    page: unknown,
    directory: string,
    problems?: AuditProblem[]
  ): Promise<Record<string, number>>;
  readControlNames(root: unknown): Array<{name: string}>;
  finishAccessibilityAudit(problems: AuditProblem[]): void;
  collectSurfaceProblems(
    problems: AuditProblem[],
    surface: string,
    check: () => unknown
  ): Promise<unknown>;
  auditTabbedSurface(
    page: unknown,
    root: unknown,
    context: string,
    problems: AuditProblem[]
  ): Promise<Record<string, number>>;
  readWorkbenchDuplicateIds(options: {
    surfaceIds: string[];
    dialogTitles: string[];
  }): Array<[string, number]>;
};
type WorkflowHelpers = {
  connectPorts(
    page: unknown,
    from: string,
    output: string,
    to: string,
    input: string
  ): Promise<Gesture>;
  checkFrameTransition(before: Frame, after: Frame): void;
  getWorkbenchState(
    page: unknown
  ): Promise<{steps: Record<string, {result: {completedAt: number}}>}>;
};

// Variable URLs keep TypeScript from requiring declarations for the executable .mjs scripts.
const scriptDirectory = process.env['KE_WF_ACCEPTANCE_SCRIPT_DIRECTORY']
  ? new URL(process.env['KE_WF_ACCEPTANCE_SCRIPT_DIRECTORY'])
  : new URL('../../../../../examples/deck/kepler-compute/scripts/', import.meta.url);
const accessibilityPath = new URL('workflow-accessibility.mjs', scriptDirectory).href;
const utilitiesPath = new URL('workflow-utils.mjs', scriptDirectory).href;
const accessibility = (await import(accessibilityPath)) as AccessibilityHelpers;
const utilities = (await import(utilitiesPath)) as WorkflowHelpers;
const editPath = new URL('workflow-edit.mjs', scriptDirectory).href;
const edit = (await import(editPath)) as {
  checkMinimumPointsProblem(message: string): void;
  checkRunToHere(before: unknown, after: unknown, targetId: string): void;
  checkInvalidDraft(before: unknown, after: unknown, stepId: string, parameter: string): void;
};
const persistencePath = new URL('workflow-persistence.mjs', scriptDirectory).href;
const persistence = (await import(persistencePath)) as {
  restoreWorkflowCollection(
    value: unknown,
    parseCollection: (value: unknown) => WorkflowCollection
  ): Promise<{restored: boolean; code?: string; message?: string}>;
};

type ControlFixture = {
  tagName: string;
  id: string;
  textContent: string;
  ariaLabel: string;
  type: string;
  value: string;
  labels: {textContent: string}[];
  ownerDocument: {getElementById(identifier: string): {textContent: string} | null};
  getAttribute(name: string): string | null;
  closest(selector: string): {textContent: string} | null;
};

function makeControl(
  tagName: string,
  options: {
    text?: string;
    ariaLabel?: string;
    labelledBy?: string;
    labels?: string[];
    enclosingLabel?: string;
    references?: Record<string, string>;
    type?: string;
    value?: string;
    role?: string;
    title?: string;
  } = {}
): ControlFixture {
  return {
    tagName,
    id: '',
    textContent: options.text ?? '',
    ariaLabel: options.ariaLabel ?? '',
    type: options.type ?? 'text',
    value: options.value ?? '',
    labels: (options.labels ?? []).map(textContent => ({textContent})),
    ownerDocument: {
      getElementById(identifier) {
        const textContent = options.references?.[identifier];
        return textContent === undefined ? null : {textContent};
      }
    },
    getAttribute(name) {
      if (name === 'aria-labelledby') return options.labelledBy ?? null;
      if (name === 'role') return options.role ?? null;
      if (name === 'title') return options.title ?? null;
      return null;
    },
    closest: selector =>
      selector === 'label' && options.enclosingLabel ? {textContent: options.enclosingLabel} : null
  };
}

function makeControlRoot(controls: ControlFixture[]) {
  return {
    evaluate: (callback: (root: unknown) => unknown) => callback({querySelectorAll: () => controls})
  };
}

function installNameDocument() {
  vi.stubGlobal('getComputedStyle', () => ({display: 'block', visibility: 'visible'}));
  vi.stubGlobal('document', {querySelectorAll: () => []});
  return {
    evaluate: (callback: (argument: unknown) => unknown, argument: unknown) => callback(argument)
  };
}

afterEach(() => vi.unstubAllGlobals());

describe('workflow acceptance targeted run records', () => {
  test('matches the actual runner when a middle step changes and its source is current', async () => {
    const document = makeFixtureDocument();
    const calls: string[] = [];
    let now = 0;
    const runner = new WorkflowRunner({
      host: makeFixtureHost(),
      registry: makeFixtureRegistry(calls),
      now: () => ++now
    });
    try {
      runner.setDocument(document);
      expect((await runner.run()).status).toBe('done');
      const completedAt = runner.getState().steps['source'].result?.completedAt;
      const changed = {
        ...document,
        steps: document.steps.map(step =>
          step.id === 'scale'
            ? {...step, parameters: {factor: {from: 'static' as const, value: 4}}}
            : step
        )
      };
      runner.setDocument(changed);
      const before = {document: changed, steps: runner.getState().steps};
      const record = await runner.run({target: 'scale'});
      const after = {steps: runner.getState().steps, runLog: [record]};
      edit.checkRunToHere(before, after, 'scale');
      expect(record.steps.map(step => [step.stepId, step.cacheHit])).toEqual([['scale', false]]);
      expect(after.steps['source'].status).toBe('done');
      expect(after.steps['source'].result?.completedAt).toBe(completedAt);
      expect(after.steps['count'].status).toBe('stale');
      expect(calls).toEqual([
        'fixture-source:source',
        'fixture-scale:scale',
        'fixture-count:count',
        'fixture-scale:scale'
      ]);
    } finally {
      await runner.destroy();
    }
  });

  function makeSnapshots(sourceStatus = 'done') {
    const before = {
      document: {
        steps: [
          {id: 'samples', inputs: {}},
          {id: 'clusters', inputs: {points: {stepId: 'samples', output: 'data'}}},
          {id: 'members', inputs: {rows: {stepId: 'clusters', output: 'points'}}}
        ]
      },
      steps: {
        samples: {status: sourceStatus, result: {completedAt: 11}},
        clusters: {status: 'stale', result: {completedAt: 12}},
        members: {status: 'stale', result: {completedAt: 13}}
      },
      runLog: []
    };
    const after = {
      ...structuredClone(before),
      runLog: [
        {
          status: 'done',
          target: 'clusters',
          steps:
            sourceStatus === 'done'
              ? [{stepId: 'clusters', cacheHit: false}]
              : [
                  {stepId: 'samples', cacheHit: false},
                  {stepId: 'clusters', cacheHit: false}
                ]
        }
      ]
    };
    after.steps.clusters.status = 'done';
    after.steps.clusters.result.completedAt = 22;
    after.steps.samples.status = 'done';
    return {before, after};
  }

  test('accepts a record containing only the stale target and preserves a current source', () => {
    const {before, after} = makeSnapshots();
    expect(() => edit.checkRunToHere(before, after, 'clusters')).not.toThrow();
  });

  test('includes a stale upstream step instead of treating it as an already current source', () => {
    const {before, after} = makeSnapshots('stale');
    after.steps.samples.result.completedAt = 21;
    expect(() => edit.checkRunToHere(before, after, 'clusters')).not.toThrow();
  });

  test('rejects a cache-hit entry for an upstream step that was already done', () => {
    const {before, after} = makeSnapshots();
    after.runLog[0].steps.unshift({stepId: 'samples', cacheHit: true});
    expect(() => edit.checkRunToHere(before, after, 'clusters')).toThrow('steps that needed a run');
  });

  test('rejects a record that omits a stale upstream step', () => {
    const {before, after} = makeSnapshots('stale');
    after.runLog[0].steps.shift();
    expect(() => edit.checkRunToHere(before, after, 'clusters')).toThrow('steps that needed a run');
  });

  test('rejects recomputing an already current source', () => {
    const {before, after} = makeSnapshots();
    after.steps.samples.result.completedAt = 21;
    expect(() => edit.checkRunToHere(before, after, 'clusters')).toThrow(
      'preserve the upstream result completedAt'
    );
  });

  test('rejects a source that did not stay done', () => {
    const {before, after} = makeSnapshots();
    after.steps.samples.status = 'stale';
    expect(() => edit.checkRunToHere(before, after, 'clusters')).toThrow(
      'current upstream must stay done'
    );
  });

  test('projects completedAt from the live runner result', async () => {
    const runner = {
      steps: {
        samples: {
          status: 'done',
          result: {
            key: 'source-key',
            engine: 'host',
            summary: {},
            outputs: {},
            durationMilliseconds: 1,
            completedAt: 11
          }
        }
      }
    };
    vi.stubGlobal('window', {__keplerWorkflowTest: {store: {getState: () => ({runner})}}});
    const state = await utilities.getWorkbenchState({
      evaluate: (callback: () => unknown) => callback()
    });
    expect(state.steps['samples'].result.completedAt).toBe(11);
  });
});

describe('workflow acceptance invalid drafts and required inputs', () => {
  function makeSnapshot() {
    return {
      document: makeFixtureDocument(),
      steps: {scale: {status: 'stale'}}
    };
  }

  test('an invalid draft preserves the binding and existing status', () => {
    const before = makeSnapshot();
    expect(() =>
      edit.checkInvalidDraft(before, structuredClone(before), 'scale', 'factor')
    ).not.toThrow();
  });

  test('rejects an invalid draft committed to the document', () => {
    const before = makeSnapshot();
    const after = structuredClone(before);
    after.document.steps[1].parameters['factor'] = {from: 'static', value: -1};
    expect(() => edit.checkInvalidDraft(before, after, 'scale', 'factor')).toThrow(
      'preserve the last valid binding'
    );
  });

  test('rejects an invalid draft that changes the step status', () => {
    const before = makeSnapshot();
    const after = structuredClone(before);
    after.steps.scale.status = 'not-configured';
    expect(() => edit.checkInvalidDraft(before, after, 'scale', 'factor')).toThrow(
      'preserve the step status'
    );
  });

  test('disconnecting a required input creates not-configured and blocked until undo', async () => {
    const document = makeFixtureDocument();
    let now = 0;
    const runner = new WorkflowRunner({
      host: makeFixtureHost(),
      registry: makeFixtureRegistry(),
      now: () => ++now
    });
    try {
      runner.setDocument(document);
      expect((await runner.run()).status).toBe('done');
      const completedAt = runner.getState().steps['scale'].result?.completedAt;
      const disconnected = disconnectWorkflowInput(document, 'scale', 'points');
      expect(disconnected.steps[1].inputs).toEqual({});
      runner.setDocument(disconnected);
      const state = runner.getState();
      expect(state.steps['source'].status).toBe('done');
      expect(state.steps['scale'].status).toBe('not-configured');
      expect(state.steps['scale'].issues).toEqual(
        expect.arrayContaining([expect.objectContaining({code: 'input-required', port: 'points'})])
      );
      expect(state.steps['count'].status).toBe('blocked');
      runner.setDocument(document);
      const restored = runner.getState();
      expect(Object.values(restored.steps).map(step => step.status)).toEqual([
        'done',
        'done',
        'done'
      ]);
      expect(restored.steps['scale'].result?.completedAt).toBe(completedAt);
    } finally {
      await runner.destroy();
    }
  });
});

describe('workflow acceptance collection persistence', () => {
  function makeStore() {
    const document = makeFixtureDocument();
    const saved = serializeWorkflowCollection({
      schemaVersion: 1,
      workflows: [document],
      activeWorkflowId: document.id
    });
    let current = structuredClone(saved);
    const restore = vi.fn((collection: WorkflowCollection) => {
      current = serializeWorkflowCollection(collection);
    });
    const store = {restore, serialize: () => current};
    vi.stubGlobal('window', {__keplerWorkflowTest: {store}});
    return {store, saved};
  }

  test('validates and restores a serialized collection with the public parser', async () => {
    const {store, saved} = makeStore();
    const parser = vi.fn(parseWorkflowCollection);
    expect(await persistence.restoreWorkflowCollection(saved, parser)).toEqual({restored: true});
    expect(parser).toHaveBeenCalledExactlyOnceWith(saved);
    expect(store.restore).toHaveBeenCalledTimes(1);
    expect(store.restore.mock.calls[0][0]).not.toBe(saved);
    expect(store.serialize()).toEqual(saved);
  });

  test('malformed workflows throw in preflight and never reach store.restore', async () => {
    const {store, saved} = makeStore();
    const malformed = {schemaVersion: 1, workflows: 'not-an-array'};
    expect(() => parseWorkflowCollection(malformed)).toThrow(
      'collection.workflows: must be an array.'
    );
    expect(await persistence.restoreWorkflowCollection(malformed, parseWorkflowCollection)).toEqual(
      {restored: false, code: 'malformed', message: 'collection.workflows: must be an array.'}
    );
    expect(store.restore).not.toHaveBeenCalled();
    expect(store.serialize()).toEqual(saved);
  });
});

type IdElement = {
  id: string;
  querySelectorAll(selector: string): IdElement[];
  getAttribute(name: string): string | null;
};

function makeIdElement(
  id: string,
  descendants: IdElement[] = [],
  attributes: Record<string, string> = {}
): IdElement {
  return {id, querySelectorAll: () => descendants, getAttribute: name => attributes[name] ?? null};
}

function installIdDocument(roots: IdElement[], host: IdElement[] = [], dialogs: IdElement[] = []) {
  vi.stubGlobal('document', {
    querySelectorAll(selector: string) {
      if (selector === '[role="dialog"][aria-modal="true"]') return dialogs;
      if (selector === '[id]')
        return [...host, ...roots.flatMap(root => [root, ...root.querySelectorAll('[id]')])];
      return roots;
    },
    getElementById: (identifier: string) =>
      identifier === 'template-heading' ? {textContent: 'Workflow templates'} : null
  });
}

describe('workflow acceptance owned duplicate IDs', () => {
  const options = {
    surfaceIds: ['workflow-dock', 'workflow-palette', 'workflow-side-panel'],
    dialogTitles: ['Workflow templates', 'Import and export', 'Keyboard shortcuts']
  };

  test('does not fail the dock for duplicate storage-action IDs in unrelated host panels', async () => {
    const page = installNameDocument();
    installIdDocument(
      [makeIdElement('owned-root', [makeIdElement('owned-control')])],
      [makeIdElement('storage-action'), makeIdElement('storage-action')]
    );
    expect(
      await accessibility.checkNames(
        page,
        makeControlRoot([makeControl('BUTTON', {text: 'Run'})]),
        'Dock'
      )
    ).toBe(1);
  });

  test('counts an element in overlapping dock and palette roots only once', () => {
    const paletteControl = makeIdElement('palette-control');
    const palette = makeIdElement('palette', [paletteControl]);
    installIdDocument([makeIdElement('dock', [palette, paletteControl]), palette]);
    expect(accessibility.readWorkbenchDuplicateIds(options)).toEqual([]);
  });

  test('still rejects duplicates within an owned surface', async () => {
    const page = installNameDocument();
    installIdDocument([
      makeIdElement('dock', [makeIdElement('duplicate'), makeIdElement('duplicate')])
    ]);
    await expect(
      accessibility.checkNames(
        page,
        makeControlRoot([makeControl('BUTTON', {text: 'Run'})]),
        'Dock'
      )
    ).rejects.toThrow('duplicate workbench ids: [["duplicate",2]]');
  });

  test('compares owned modal IDs across surfaces while excluding host dialogs', () => {
    const modal = makeIdElement('modal', [makeIdElement('shared')], {
      'aria-labelledby': 'template-heading'
    });
    const hostDialog = makeIdElement(
      'host-dialog',
      [makeIdElement('storage-action'), makeIdElement('storage-action')],
      {'aria-label': 'Save map'}
    );
    installIdDocument([makeIdElement('dock', [makeIdElement('shared')])], [], [modal, hostDialog]);
    expect(accessibility.readWorkbenchDuplicateIds(options)).toEqual([['shared', 2]]);
  });
});

describe('workflow acceptance accessible names (F1)', () => {
  test.each([
    ['SELECT', 'All rows'],
    ['TEXTAREA', 'Existing workflow JSON'],
    ['INPUT', 'Unrelated content']
  ])('rejects unlabeled %s even when it contains text or a value', async (tagName, text) => {
    const page = installNameDocument();
    await expect(
      accessibility.checkNames(
        page,
        makeControlRoot([makeControl(tagName, {text, value: text})]),
        'Unnamed form control'
      )
    ).rejects.toThrow('every control must have a nonempty accessible name');
  });

  test.each([
    {ariaLabel: 'Population'},
    {labels: ['Population']},
    {enclosingLabel: 'Population'},
    {labelledBy: 'population-label', references: {'population-label': 'Population'}}
  ])('accepts a select with an explicit name %j', async options => {
    const page = installNameDocument();
    expect(
      await accessibility.checkNames(
        page,
        makeControlRoot([makeControl('SELECT', {...options, text: 'All rows'})]),
        'Named select'
      )
    ).toBe(1);
  });

  test('resolves an external label-for association through native labels', () => {
    installNameDocument();
    const control = makeControl('SELECT', {text: 'All rows'});
    control.id = 'rows';
    const label = {htmlFor: 'rows', textContent: 'Rows to analyze'};
    Object.defineProperty(control, 'labels', {
      get: () => (label.htmlFor === control.id ? [label] : [])
    });
    const readNames = () => accessibility.readControlNames({querySelectorAll: () => [control]});
    expect(readNames()[0].name).toBe('Rows to analyze');
    control.id = 'unassociated';
    expect(readNames()[0].name).toBe('');
  });

  test('uses a wrapping label for a textarea without using its draft value', () => {
    installNameDocument();
    const control = makeControl('TEXTAREA', {enclosingLabel: 'Workflow JSON', value: 'draft'});
    expect(accessibility.readControlNames({querySelectorAll: () => [control]})[0].name).toBe(
      'Workflow JSON'
    );
  });

  test('concatenates multiple aria-labelledby references in attribute order', () => {
    installNameDocument();
    const control = makeControl('SELECT', {
      labelledBy: 'second missing first',
      ariaLabel: 'Fallback',
      references: {first: '  selected rows  ', second: 'Analyze\n only'}
    });
    expect(accessibility.readControlNames({querySelectorAll: () => [control]})[0].name).toBe(
      'Analyze only selected rows'
    );
  });

  test('uses title only as a button name fallback', () => {
    installNameDocument();
    const controls = [
      makeControl('BUTTON', {title: 'Close dialog'}),
      makeControl('DIV', {role: 'button', title: 'Open palette'}),
      makeControl('INPUT', {type: 'button', title: 'Run'}),
      makeControl('BUTTON', {text: 'Cancel', title: 'Lower priority'}),
      makeControl('BUTTON', {ariaLabel: 'Close', text: 'Cancel', title: 'Lower priority'}),
      makeControl('TEXTAREA', {title: 'Must still have a label', value: 'draft'})
    ];
    expect(
      accessibility
        .readControlNames({querySelectorAll: () => controls})
        .map(control => control.name)
    ).toEqual(['Close dialog', 'Open palette', 'Run', 'Cancel', 'Close', '']);
  });

  test('rejects whitespace labels and missing labelledby targets', async () => {
    const page = installNameDocument();
    await expect(
      accessibility.checkNames(
        page,
        makeControlRoot([
          makeControl('SELECT', {
            ariaLabel: '  ',
            labelledBy: 'missing',
            labels: ['  '],
            text: 'All rows'
          })
        ]),
        'Missing label'
      )
    ).rejects.toThrow('nonempty accessible name');
  });

  test('accepts button text and native input button values', async () => {
    const page = installNameDocument();
    expect(
      await accessibility.checkNames(
        page,
        makeControlRoot([
          makeControl('BUTTON', {text: 'Run'}),
          makeControl('INPUT', {type: 'button', value: 'Run'}),
          makeControl('DIV', {role: 'tab', text: 'Result'})
        ]),
        'Button names'
      )
    ).toBe(3);
  });
});

describe('workflow acceptance aggregated accessibility findings', () => {
  test('collects every nameless control across surfaces before failing', async () => {
    const page = installNameDocument();
    const problems: AuditProblem[] = [];
    const first = makeControl('TEXTAREA', {value: 'draft'});
    first.id = 'workflow-import-text';
    const second = makeControl('SELECT', {text: 'All rows'});
    second.id = 'rows';
    await expect(
      accessibility.checkNames(page, makeControlRoot([first, second]), 'Import', problems)
    ).resolves.toBe(2);
    await expect(
      accessibility.checkNames(page, makeControlRoot([makeControl('INPUT')]), 'Result', problems)
    ).resolves.toBe(1);
    expect(
      problems.map(problem => [problem.surface, problem.kind, problem.controls?.length])
    ).toEqual([
      ['Import', 'nameless-controls', 2],
      ['Result', 'nameless-controls', 1]
    ]);
    expect(() => accessibility.finishAccessibilityAudit(problems)).toThrow('workflow-import-text');
    expect(() => accessibility.finishAccessibilityAudit(problems)).toThrow('Result');
  });

  test('records a failed surface without preventing the next audit', async () => {
    const problems: AuditProblem[] = [];
    await accessibility.collectSurfaceProblems(problems, 'Palette', async () => {
      throw new Error('Palette did not open');
    });
    let visited = false;
    await accessibility.collectSurfaceProblems(problems, 'Templates', async () => {
      visited = true;
    });
    expect(visited).toBe(true);
    expect(problems).toEqual([
      {surface: 'Palette', kind: 'surface-error', message: 'Palette did not open'}
    ]);
  });

  test('visits all dialog tabs after an unnamed Import control and a broken tab', async () => {
    const page = installNameDocument();
    const labels = [
      'Export JSON',
      'Export TypeScript',
      'Import',
      'From catalog history',
      'Run report',
      'Graph (DOT)'
    ];
    let active = '';
    const visited: string[] = [];
    const problems: AuditProblem[] = [];
    const root = {
      evaluate(callback: (root: unknown) => unknown) {
        return callback({
          querySelectorAll: () =>
            active === 'Import'
              ? [makeControl('TEXTAREA', {value: 'draft'})]
              : [makeControl('BUTTON', {text: 'Close'})]
        });
      },
      getByRole: () => ({
        count: async () => labels.length,
        nth: (index: number) => ({
          textContent: async () => labels[index],
          click: async () => {
            active = labels[index];
            visited.push(active);
            if (active === 'From catalog history') throw new Error('Cannot activate history');
          }
        })
      })
    };
    const audited = await accessibility.auditTabbedSurface(
      page,
      root,
      'Import and export',
      problems
    );
    expect(visited).toEqual(labels);
    expect(Object.keys(audited)).toEqual(
      labels
        .filter(label => label !== 'From catalog history')
        .map(label => `Import and export/${label}`)
    );
    expect(problems.map(problem => [problem.surface, problem.kind])).toEqual([
      ['Import and export/Import', 'nameless-controls'],
      ['Import and export/tab 4', 'surface-error']
    ]);
    expect(() => accessibility.finishAccessibilityAudit(problems)).toThrow(
      'Cannot activate history'
    );
  });
});

type PaintedElement = {
  textContent: string;
  style: Record<string, string>;
  parentElement: PaintedElement | null;
  getClientRects(): number[];
  closest(selector: string): null;
};

function makePaintedElement(
  style: Record<string, string>,
  parentElement: PaintedElement | null = null
): PaintedElement {
  return {
    textContent: 'Disabled',
    style,
    parentElement,
    getClientRects: () => [1],
    closest: () => null
  };
}

function installPaintStyles() {
  vi.stubGlobal('getComputedStyle', (element: PaintedElement) => ({
    color: 'rgb(138, 148, 166)',
    backgroundColor: 'rgba(0, 0, 0, 0)',
    backgroundImage: 'none',
    opacity: '1',
    ...element.style
  }));
}

describe('workflow acceptance contrast (F2)', () => {
  test('rejects disabled status text whose ancestor opacity drops contrast below 4.5', async () => {
    installPaintStyles();
    const canvas = makePaintedElement({backgroundColor: 'rgb(27, 32, 40)'});
    const node = makePaintedElement({backgroundColor: 'rgb(27, 32, 40)', opacity: '0.7'}, canvas);
    const badge = makePaintedElement({}, node);
    const root = {
      locator: () => ({
        evaluateAll: (callback: (elements: unknown[]) => unknown) => callback([badge])
      })
    };
    await expect(accessibility.checkContrast(root, 'Disabled badge', 'span')).rejects.toThrow(
      'must have contrast >= 4.5:1; saw 3.334:1'
    );
    const [sample] = accessibility.readContrastSamples([badge]);
    expect(sample.foreground).toEqual([104.69999999999999, 113.19999999999999, 128.2]);
    expect(sample.background).toEqual([27, 32, 40]);
    expect(accessibility.calculateContrast(sample.foreground, sample.background)).toBeCloseTo(
      3.3343279585350953,
      12
    );
  });

  test('collects all low-contrast samples and proceeds to other surfaces', async () => {
    installPaintStyles();
    const canvas = makePaintedElement({backgroundColor: 'rgb(27, 32, 40)'});
    const node = makePaintedElement({backgroundColor: 'rgb(27, 32, 40)', opacity: '0.7'}, canvas);
    const samples = [makePaintedElement({}, node), makePaintedElement({}, node)];
    const root = {
      locator: () => ({
        evaluateAll: (callback: (elements: unknown[]) => unknown) => callback(samples)
      })
    };
    const problems: AuditProblem[] = [];
    expect(
      await accessibility.checkContrast(root, 'Disabled nodes', 'span', problems)
    ).toHaveLength(2);
    expect(await accessibility.checkContrast(root, 'Toolbar', 'span', problems)).toHaveLength(2);
    expect(problems.map(problem => [problem.surface, problem.kind])).toEqual([
      ['Disabled nodes', 'contrast'],
      ['Disabled nodes', 'contrast'],
      ['Toolbar', 'contrast'],
      ['Toolbar', 'contrast']
    ]);
    expect(() => accessibility.finishAccessibilityAudit(problems)).toThrow('4 problems');
  });

  test('preserves the opaque contrast and accepts an accessible pair', async () => {
    installPaintStyles();
    const badge = makePaintedElement({backgroundColor: 'rgb(27, 32, 40)'});
    const root = {
      locator: () => ({
        evaluateAll: (callback: (elements: unknown[]) => unknown) => callback([badge])
      })
    };
    const [sample] = await accessibility.checkContrast(root, 'Opaque badge', 'span');
    expect(accessibility.calculateContrast(sample.foreground, sample.background)).toBeCloseTo(
      5.348962637770938,
      12
    );
  });

  test('composites nested group opacity and a different translucent background', () => {
    installPaintStyles();
    const canvas = makePaintedElement({backgroundColor: 'rgb(0, 0, 0)'});
    const parent = makePaintedElement(
      {backgroundColor: 'rgb(100, 100, 100)', opacity: '0.5'},
      canvas
    );
    const child = makePaintedElement(
      {
        color: 'rgba(255, 255, 255, 0.5)',
        backgroundColor: 'rgba(200, 200, 200, 0.5)',
        opacity: '0.5'
      },
      parent
    );
    const [sample] = accessibility.readContrastSamples([child]);
    expect(sample.background).toEqual([62.5, 62.5, 62.5]);
    expect(sample.foreground).toEqual([75.625, 75.625, 75.625]);
    expect(sample.opacities).toEqual([0.5, 0.5, 1]);
  });
});

function makeDragPage(options: {coveredInput?: boolean; shiftDuringDrag?: boolean} = {}) {
  let destinationY = 400;
  let pressed = false;
  let releases = 0;
  let drop: Point | undefined;
  const events: string[] = [];
  type HitElement = {
    getAttribute(name: string): string;
    contains(element: unknown): boolean;
    ownerDocument: {elementFromPoint(x: number, y: number): unknown};
  };
  const hitDocument = {
    elementFromPoint(x: number, y: number): unknown {
      if (x === 105 && y === 105) return sourceElement;
      if (!options.coveredInput && x === 305 && y === destinationY + 5) return destinationElement;
      return {tagName: 'DIV', getAttribute: () => 'unexpected-overlay'};
    }
  };
  const sourceElement: HitElement = {
    getAttribute: () => 'workflow-port-out-source-data',
    contains: () => false,
    ownerDocument: hitDocument
  };
  const destinationElement: HitElement = {
    getAttribute: () => 'workflow-port-in-target-points',
    contains: () => false,
    ownerDocument: hitDocument
  };
  const source = {
    scrollIntoViewIfNeeded: async () => {
      events.push('scroll-source');
      destinationY = 200;
    },
    boundingBox: async () => {
      events.push('measure-source');
      return {x: 100, y: 100, width: 10, height: 10};
    },
    evaluate: (callback: (element: HitElement, point: Point) => unknown, point: Point) =>
      callback(sourceElement, point)
  };
  const destination = {
    scrollIntoViewIfNeeded: async () => {
      events.push('scroll-target');
    },
    boundingBox: async () => {
      events.push('measure-target');
      return {x: 300, y: destinationY, width: 10, height: 10};
    },
    evaluate: (callback: (element: HitElement, point: Point) => unknown, point: Point) =>
      callback(destinationElement, point)
  };
  const container = {
    getByTestId(identifier: string) {
      if (identifier === 'workflow-port-out-source-data') return source;
      if (identifier === 'workflow-port-in-target-points') return destination;
      return {focus: async () => {}};
    }
  };
  const page = {
    getByTestId: () => container,
    evaluate: async () => {},
    keyboard: {press: async () => {}},
    mouse: {
      move: async (x: number, y: number) => {
        drop = {x, y};
        if (pressed && options.shiftDuringDrag) destinationY = 100;
      },
      down: async () => {
        pressed = true;
        events.push('mouse-down');
      },
      up: async () => {
        pressed = false;
        releases++;
      }
    }
  };
  return {page, events, readDrop: () => drop, readReleases: () => releases};
}

describe('workflow acceptance port dragging (F3)', () => {
  test('measures after both scrolls and verifies the actual input at release', async () => {
    const fixture = makeDragPage();
    const gesture = await utilities.connectPorts(
      fixture.page,
      'source',
      'data',
      'target',
      'points'
    );
    expect(fixture.events.slice(0, 3)).toEqual([
      'scroll-source',
      'scroll-target',
      'measure-target'
    ]);
    expect(fixture.readDrop()).toEqual({x: 305, y: 205});
    expect(gesture.destinationHit).toMatchObject({
      matched: true,
      expected: 'workflow-port-in-target-points'
    });
    expect(fixture.readReleases()).toBe(1);
  });

  test('rejects an obstructed drop instead of accepting an unchanged document as a refusal', async () => {
    const fixture = makeDragPage({coveredInput: true});
    await expect(
      utilities.connectPorts(fixture.page, 'source', 'data', 'target', 'points')
    ).rejects.toThrow('Input before drag must be under (305, 205)');
    expect(fixture.events).not.toContain('mouse-down');
    expect(fixture.readReleases()).toBe(0);
  });

  test('rejects a target that moves during dragging and always releases the mouse', async () => {
    const fixture = makeDragPage({shiftDuringDrag: true});
    await expect(
      utilities.connectPorts(fixture.page, 'source', 'data', 'target', 'points')
    ).rejects.toThrow('Input at drop must be under (305, 205)');
    expect(fixture.readReleases()).toBe(1);
  });
});

describe('workflow acceptance framing', () => {
  const canvas = {left: 0, right: 600, top: 0, bottom: 300};
  const before = {
    canvas,
    node: {left: 5000, right: 5240, top: 5000, bottom: 5150},
    transform: 'translate(0px, 0px) scale(1)'
  };
  const after = {
    canvas,
    node: {left: 100, right: 340, top: 50, bottom: 200},
    transform: 'translate(-4900px, -4950px) scale(1)'
  };

  test('accepts F only when an offscreen node is framed by a changed viewport', () => {
    expect(() => utilities.checkFrameTransition(before, after)).not.toThrow();
  });
  test('rejects a no-op F', () => {
    expect(() => utilities.checkFrameTransition(before, before)).toThrow('bring the complete node');
  });
  test('rejects an already visible fixture', () => {
    expect(() => utilities.checkFrameTransition(after, after)).toThrow(
      'fixture must start outside'
    );
  });
  test('rejects moved node geometry without a changed viewport', () => {
    expect(() =>
      utilities.checkFrameTransition(before, {...after, transform: before.transform})
    ).toThrow('must change the viewport transform');
  });
});

describe('workflow acceptance exact inline limit', () => {
  const message =
    'Find clusters: Minimum points (including the point) must be between 1 and 1000; got 0.';
  test('accepts the exact parameter range and rejected value', () => {
    expect(() => edit.checkMinimumPointsProblem(message)).not.toThrow();
  });
  test.each([
    'Minimum points must be at least 10.',
    message.replace('1000', '100'),
    message.replace('got 0', 'got 10'),
    message.replace('Find clusters: ', '')
  ])('rejects a wrong or incomplete limit message: %s', incorrect => {
    expect(() => edit.checkMinimumPointsProblem(incorrect)).toThrow(
      'exact range and rejected value'
    );
  });
});

describe('workflow acceptance Result audits', () => {
  test.each([
    'named controls',
    'unnamed Result control',
    'unnamed summary control',
    'collected unnamed controls'
  ])('audits every output and column dialog: %s', async mode => {
    installNameDocument();
    const directory = await mkdtemp(join(tmpdir(), 'kewf-acceptance-'));
    const opened: string[] = [];
    const closed: string[] = [];
    const screenshots: string[] = [];
    const outputIds = ['workflow-result-output-points', 'workflow-result-output-clusters'];
    const columns = ['source_id', 'value'];
    let activeOutput = '';
    let activeColumn = '';
    const namedRoot = makeControlRoot([makeControl('BUTTON', {text: 'Close'})]);
    const unnamedRoot = makeControlRoot([makeControl('SELECT', {text: 'All rows'})]);
    const tab = {click: async () => {}};
    const inspector = {
      ...(['unnamed Result control', 'collected unnamed controls'].includes(mode)
        ? unnamedRoot
        : namedRoot),
      getByTestId(identifier: string): unknown {
        if (identifier.startsWith('workflow-inspector-tab-')) return tab;
        if (identifier === 'workflow-result') return {waitFor: async () => {}};
        let expanded = false;
        const table = {
          waitFor: async () => {
            expect(expanded).toBe(true);
          },
          getByRole(_role: string, options: {name: string | RegExp}) {
            if (typeof options.name === 'string')
              return {
                click: async () => {
                  activeOutput = identifier;
                  activeColumn = String(options.name).slice('Summarize '.length);
                  opened.push(`${identifier}/${activeColumn}`);
                }
              };
            return {
              evaluateAll: (callback: (elements: unknown[]) => unknown) =>
                callback(columns.map(column => ({getAttribute: () => `Summarize ${column}`})))
            };
          }
        };
        return {
          ...namedRoot,
          locator(selector: string) {
            if (selector === 'table') return table;
            return {
              first: () => ({
                count: async () => 1,
                getAttribute: async () => (expanded ? '' : null),
                locator: () => ({
                  first: () => ({
                    click: async () => {
                      expanded = true;
                    }
                  })
                })
              })
            };
          }
        };
      },
      locator: () => ({
        evaluateAll: (callback: (elements: unknown[]) => unknown) =>
          callback(outputIds.map(identifier => ({getAttribute: () => identifier})))
      }),
      getByRole(_role: string, options: {name: string}) {
        expect(options.name).toBe(`${activeColumn} summary`);
        return {
          ...(['unnamed summary control', 'collected unnamed controls'].includes(mode)
            ? unnamedRoot
            : namedRoot),
          waitFor: async () => {},
          getByRole(_buttonRole: string, closeOptions: {name: string}) {
            expect(closeOptions.name).toBe(`Close ${activeColumn} summary`);
            return {
              click: async () => {
                closed.push(`${activeOutput}/${activeColumn}`);
              }
            };
          }
        };
      }
    };
    const page = {
      evaluate: (callback: (argument: unknown) => unknown, argument: unknown) => callback(argument),
      getByTestId: () => ({getByTestId: () => inspector}),
      screenshot: async (options: {path: string}) => {
        screenshots.push(options.path);
      }
    };
    try {
      if (mode !== 'named controls' && mode !== 'collected unnamed controls') {
        await expect(accessibility.auditResultInspector(page, directory)).rejects.toThrow(
          'every control must have a nonempty accessible name'
        );
        expect(opened).toHaveLength(mode === 'unnamed Result control' ? 0 : 1);
        expect(closed).toEqual([]);
        return;
      }
      const problems = mode === 'collected unnamed controls' ? ([] as AuditProblem[]) : undefined;
      const audited = await accessibility.auditResultInspector(page, directory, problems);
      const expected = outputIds.flatMap(identifier =>
        columns.map(column => `${identifier}/${column}`)
      );
      expect(opened).toEqual(expected);
      expect(closed).toEqual(expected);
      expect(Object.keys(audited).sort()).toEqual(['result', ...outputIds, ...expected].sort());
      expect(screenshots).toHaveLength(5);
      if (problems) {
        expect(problems.map(problem => problem.surface)).toEqual([
          'Result inspector',
          ...expected.map(surface => `${surface} summary`)
        ]);
        expect(() => accessibility.finishAccessibilityAudit(problems)).toThrow('5 problems');
      }
    } finally {
      await rm(directory, {recursive: true, force: true});
    }
  });
});
