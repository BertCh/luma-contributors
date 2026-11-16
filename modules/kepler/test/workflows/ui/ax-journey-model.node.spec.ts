// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {deflateSync} from 'node:zlib';
import {describe, expect, test, vi} from 'vitest';
import {
  JourneyRecorder,
  openWorkflowJourneySurface,
  parseOptions,
  readSourceIdentity,
  runSuggestionToResult,
  waitForOptionalSurface
} from '../../../../../examples/deck/kepler-compute/scripts/ax-journey-baseline.mjs';
import {
  AX_SURFACE_IDENTIFIERS,
  accountJourney,
  compareJourneys,
  evaluateJourneyTargets,
  GALLERY_SAMPLE_IDENTIFIERS,
  validateJourneyTargets
} from '../../../../../examples/deck/kepler-compute/scripts/ax-journey-model.mjs';
import {SANDBOX_SAMPLES} from '../../../../../examples/deck/kepler-compute/src/sandbox-samples';

type JourneyTarget = {
  id: string;
  required: boolean;
  outcome: string;
  maximum?: Record<string, number>;
};
const targets: {schemaVersion: number; journeys: JourneyTarget[]} = validateJourneyTargets(
  JSON.parse(
    readFileSync(
      new URL(
        '../../../../../examples/deck/kepler-compute/scripts/ax-journey-targets.json',
        import.meta.url
      ),
      'utf8'
    )
  )
);
const createLegacyRecords = () =>
  GALLERY_SAMPLE_IDENTIFIERS.map(sampleIdentifier => ({
    id: `first-result:${sampleIdentifier}`,
    outcome: 'result-visible'
  }));
const createSuggestionEvidence = (sampleIdentifier: string) => ({
  sampleIdentifier,
  suggestionIdentifier: 'fixture-suggestion',
  previousRunIdentifier: 'previous-run'
});
const createResultEvidence = (sampleIdentifier: string) => ({
  status: 'done',
  busy: false,
  stale: false,
  hasError: false,
  sampleIdentifier,
  suggestionIdentifier: 'fixture-suggestion',
  runIdentifier: 'completed-run',
  text: 'Completed analysis for the selected sample'
});
const createSuccessfulRecords = () => [
  ...createLegacyRecords(),
  ...targets.journeys.map(target => ({
    id: target.id,
    outcome: target.outcome,
    analysisActionCount: target.id.startsWith('suggestion-result:') ? 1 : 4,
    observations: target.id.startsWith('suggestion-result:')
      ? {
          suggestionEvidence: createSuggestionEvidence(
            target.id.slice('suggestion-result:'.length)
          ),
          resultEvidence: createResultEvidence(target.id.slice('suggestion-result:'.length))
        }
      : {}
  }))
];

describe('AX-70 pure journey accounting', () => {
  test('only selector timeouts make an optional surface absent', async () => {
    expect(await waitForOptionalSurface({waitFor: async () => undefined}, 10)).toBe(true);
    const timeout = new Error('Selector not found');
    timeout.name = 'TimeoutError';
    expect(
      await waitForOptionalSurface(
        {
          waitFor: async () => {
            throw timeout;
          }
        },
        10
      )
    ).toBe(false);
    await expect(
      waitForOptionalSurface(
        {
          waitFor: async () => {
            throw new Error('Page closed');
          }
        },
        10
      )
    ).rejects.toThrow('Page closed');
  });

  test('counts setup separately and includes waits in analysis timing', () => {
    expect(
      accountJourney({
        steps: [
          {action: 'click', phase: 'setup'},
          {action: 'select-file', phase: 'setup'},
          {action: 'fill', phase: 'analysis'},
          {action: 'press', phase: 'analysis'}
        ],
        startedAt: 100.2,
        setupFinishedAt: 300.2,
        finishedAt: 700.6
      })
    ).toEqual({
      actionCount: 4,
      setupActionCount: 2,
      analysisActionCount: 2,
      milliseconds: 600,
      setupMilliseconds: 200,
      analysisMilliseconds: 400
    });
    expect(
      accountJourney({steps: [], startedAt: null, setupFinishedAt: null, finishedAt: 900})
    ).toEqual({
      actionCount: 0,
      setupActionCount: 0,
      analysisActionCount: 0,
      milliseconds: null,
      setupMilliseconds: null,
      analysisMilliseconds: null
    });
  });

  test('retains the attempted action and cause after a missing required selector', async () => {
    const recorder = new JourneyRecorder('fixture', 'Fixture', {on: () => undefined});
    await expect(
      recorder.act('click', '[data-testid="missing"]', () => {
        throw new Error('Selector unavailable');
      })
    ).rejects.toThrow(
      'Journey fixture: click [data-testid="missing"] failed: Selector unavailable'
    );
    recorder.markFinished();
    expect(recorder.toRecord({outcome: 'failed'})).toMatchObject({
      outcome: 'failed',
      actionCount: 1,
      setupActionCount: 1,
      steps: [{action: 'click', target: '[data-testid="missing"]', phase: 'setup'}]
    });
  });

  test('the authoritative gallery and targets cover all eight samples and all five surfaces', () => {
    expect([...GALLERY_SAMPLE_IDENTIFIERS].sort()).toEqual(
      SANDBOX_SAMPLES.map(sample => sample.id).sort()
    );
    expect(GALLERY_SAMPLE_IDENTIFIERS).toHaveLength(8);
    expect(AX_SURFACE_IDENTIFIERS).toHaveLength(5);
    for (const sampleIdentifier of GALLERY_SAMPLE_IDENTIFIERS) {
      expect(
        targets.journeys.find(target => target.id === `suggestion-result:${sampleIdentifier}`)
      ).toMatchObject({required: false, maximum: {analysisActionCount: 1}});
    }
    expect(evaluateJourneyTargets(createSuccessfulRecords(), targets)).toMatchObject({
      status: 'passed',
      qualified: true
    });
  });

  test('missing, duplicate and failed legacy sample journeys fail complete coverage', () => {
    const records = createSuccessfulRecords();
    for (const incomplete of [
      records.slice(1),
      [...records, records[0]],
      records.map((record, index) => (index === 0 ? {...record, outcome: 'failed'} : record))
    ]) {
      expect(evaluateJourneyTargets(incomplete, targets)).toMatchObject({
        status: 'failed',
        qualified: false
      });
    }
  });

  test('absent optional surfaces are unqualified, but optional execution failures propagate', () => {
    const records = createSuccessfulRecords().map(record =>
      record.id.startsWith('first-result:')
        ? record
        : {
            ...record,
            outcome: 'surface-not-present',
            reason: 'surface not present'
          }
    );
    const result = evaluateJourneyTargets(records, targets);
    expect(result).toMatchObject({status: 'incomplete', qualified: false});
    expect(result.gates.filter(gate => gate.status === 'not-run')).toHaveLength(
      targets.journeys.length
    );
    expect(
      evaluateJourneyTargets(
        [...records, {id: 'untargeted', outcome: 'failed', error: 'Timeout'}],
        targets
      )
    ).toMatchObject({status: 'failed', qualified: false});
    expect(evaluateJourneyTargets(createLegacyRecords(), targets)).toMatchObject({
      status: 'incomplete'
    });
  });

  test('checks limits inclusively and rejects missing or nonfinite measurements', () => {
    for (const analysisActionCount of [2, null, undefined, Number.NaN, Infinity]) {
      const records = createSuccessfulRecords().map(record =>
        record.id === 'suggestion-result:points-time' ? {...record, analysisActionCount} : record
      );
      const result = evaluateJourneyTargets(records, targets);
      expect(result.status).toBe('failed');
      expect(
        result.gates.find(gate => gate.id === 'suggestion-result:points-time')?.reason
      ).toContain('analysisActionCount: expected at most 1');
    }
  });

  test('compares matching successful records without mutating the baseline', () => {
    const baseline = [
      {
        id: 'sample',
        outcome: 'result-visible',
        actionCount: 9,
        analysisActionCount: 2,
        milliseconds: 1000
      }
    ];
    const serialized = JSON.stringify(baseline);
    expect(
      compareJourneys(
        [{...baseline[0], actionCount: 8, analysisActionCount: 1, milliseconds: 1200}],
        baseline
      )
    ).toMatchObject([{actionCountDelta: -1, analysisActionCountDelta: -1, millisecondsDelta: 200}]);
    for (const record of [
      {id: 'missing', outcome: 'result-visible'},
      {id: 'sample', outcome: 'failed'},
      {id: 'sample', outcome: 'surface-not-present'}
    ]) {
      expect(compareJourneys([record], baseline)).toMatchObject([
        {actionCountDelta: null, analysisActionCountDelta: null, millisecondsDelta: null}
      ]);
    }
    expect(JSON.stringify(baseline)).toBe(serialized);
  });

  test('refuses invalid or mistyped target assertions', () => {
    for (const invalid of [
      {},
      {schemaVersion: 2, journeys: []},
      {schemaVersion: 1, journeys: [targets.journeys[0], targets.journeys[0]]},
      {schemaVersion: 1, journeys: [{...targets.journeys[0], maximum: {typo: 1}}]},
      {schemaVersion: 1, journeys: [{...targets.journeys[0], maximum: {analysisActionCount: -1}}]}
    ]) {
      expect(() => validateJourneyTargets(invalid)).toThrow('AX-70 targets:');
    }
  });
});

describe('AX-70 reviewer regressions', () => {
  test('finding 1: suggestion journeys require a fresh successful result for the chosen input', async () => {
    const expected = createSuggestionEvidence('points-time');
    const completed = createResultEvidence('points-time');
    const rejectedResults = [
      {...completed, status: 'error', text: 'Error: GPU computation failed'},
      {...completed, text: 'Error: GPU computation failed'},
      {...completed, hasError: true},
      {...completed, status: 'loading', busy: true},
      {...completed, stale: true},
      {...completed, runIdentifier: 'previous-run'},
      {...completed, sampleIdentifier: 'districts'},
      {...completed, suggestionIdentifier: 'another-suggestion'},
      {...completed, runIdentifier: ''},
      {...completed, status: null}
    ];
    for (const evidence of [...rejectedResults, completed]) {
      const attributes: Record<string, string | null> = {
        'data-status': evidence.status,
        'aria-busy': String(evidence.busy),
        'data-stale': String(evidence.stale),
        'data-sample-id': evidence.sampleIdentifier,
        'data-suggestion-id': evidence.suggestionIdentifier,
        'data-run-id': evidence.runIdentifier
      };
      const card = {
        getAttribute: (name: string) => attributes[name] ?? null,
        getClientRects: () => [{}],
        querySelector: () => (evidence.hasError ? {} : null),
        textContent: evidence.text
      };
      vi.stubGlobal('document', {querySelector: () => card});
      try {
        const suggestion = {
          first() {
            return this;
          },
          getAttribute: async (name: string) =>
            name === 'data-sample-id' ? expected.sampleIdentifier : expected.suggestionIdentifier,
          click: async () => undefined,
          toString: () => 'selected-suggestion'
        };
        const page = {
          on: () => undefined,
          getByTestId: () => ({
            first() {
              return this;
            },
            count: async () => 1,
            getAttribute: async () => expected.previousRunIdentifier
          }),
          waitForFunction: async (predicate: (argument: unknown) => unknown, argument: unknown) => {
            const result = predicate(argument);
            if (!result) throw new Error('Result never completed a fresh run');
            return {jsonValue: async () => result, dispose: async () => undefined};
          }
        };
        const recorder = new JourneyRecorder('suggestion-result:points-time', 'Suggestion', page);
        const result = runSuggestionToResult(recorder, {locator: () => suggestion}, 'points-time');
        if (evidence === completed) {
          await expect(result).resolves.toEqual(completed);
        } else {
          await expect(result).rejects.toThrow();
          const records = createSuccessfulRecords().map(record =>
            record.id === 'suggestion-result:points-time'
              ? {...record, observations: {suggestionEvidence: expected, resultEvidence: evidence}}
              : record
          );
          expect(evaluateJourneyTargets(records, targets).qualified).toBe(false);
        }
      } finally {
        vi.unstubAllGlobals();
      }
    }
    const forged = createSuccessfulRecords().map(record =>
      record.id === 'suggestion-result:points-time' ? {...record, observations: {}} : record
    );
    expect(evaluateJourneyTargets(forged, targets).qualified).toBe(false);
  });

  test.each([
    'ax-command-palette',
    'ax-workflow-gallery'
  ])('finding 2: opens %s through the initially collapsed, empty host', async surfaceIdentifier => {
    const state = {
      sidePanelOpen: false,
      dockOpen: false,
      workflowPresent: false,
      surfaceOpen: false
    };
    const tabSelector = '.side-panel__tab[data-for="workflows-nav"]';
    const createLocator = (name: string) => ({
      first() {
        return this;
      },
      getByTestId: (identifier: string) => createLocator(`${name}/${identifier}`),
      toString: () => name,
      isVisible: async () => {
        if (name === tabSelector) return true;
        if (name === 'workflow-side-panel') return state.sidePanelOpen;
        if (name === 'workflow-dock/workflow-dock-toggle') return !state.dockOpen;
        if (name.startsWith('workflow-dock/')) return state.dockOpen;
        return state.surfaceOpen;
      },
      isEnabled: async () => state.workflowPresent,
      inputValue: async () => (state.workflowPresent ? 'new-workflow' : ''),
      waitFor: async () => {
        if (!(await createLocator(name).isVisible())) throw new Error(`${name} is hidden`);
      },
      click: async () => {
        if (name === tabSelector) state.sidePanelOpen = true;
        else if (name.endsWith('/workflow-dock-toggle')) state.dockOpen = true;
        else if (name.endsWith('/workflow-new')) state.workflowPresent = true;
        else if (name.endsWith('/workflow-add-step') || name.endsWith('/workflow-templates'))
          state.surfaceOpen = true;
      }
    });
    const page = {on: () => undefined, getByTestId: createLocator, locator: createLocator};
    const recorder = new JourneyRecorder(`surface:${surfaceIdentifier}`, 'Workflow surface', page);
    await expect(openWorkflowJourneySurface(recorder, surfaceIdentifier)).resolves.toBe(true);
    expect(await page.getByTestId(surfaceIdentifier).isVisible()).toBe(true);
    const toolbar = 'workflow-dock/workflow-toolbar';
    expect(recorder.steps.map(step => step.target)).toEqual([
      tabSelector,
      'workflow-dock/workflow-dock-toggle',
      ...(surfaceIdentifier === 'ax-command-palette' ? [`${toolbar}/workflow-new`] : []),
      `${toolbar}/${surfaceIdentifier === 'ax-command-palette' ? 'workflow-add-step' : 'workflow-templates'}`
    ]);
    expect(recorder.steps.every(step => step.phase === 'setup')).toBe(true);
  });

  test.each([
    {maximum: 1},
    {maximum: 'limits'},
    {maximum: null},
    {maximum: []},
    {maximum: new Date()},
    {maximium: {analysisActionCount: 1}},
    {maximum: {analysisActionCounts: 1}}
  ])('finding 3: rejects malformed or misspelled assertions %j', invalidFields => {
    const {maximum, ...baseTarget} = targets.journeys[0];
    const invalidPolicy = {schemaVersion: 1, journeys: [{...baseTarget, ...invalidFields}]};
    expect(() => validateJourneyTargets(invalidPolicy)).toThrow('AX-70 targets:');
    const nineActionRecords = createSuccessfulRecords().map(record => ({
      ...record,
      analysisActionCount: 9
    }));
    expect(() => evaluateJourneyTargets(nineActionRecords, invalidPolicy)).toThrow(
      'AX-70 targets:'
    );
  });
});

describe('AX-70 release evidence protection', () => {
  test('fingerprints clean and dirty trees including deletions and untracked source', () => {
    const repositoryDirectory = mkdtempSync(path.join(tmpdir(), 'ax-journey-identity-'));
    const objectDirectory = path.join(repositoryDirectory, '.git/objects');
    const writeObject = (kind: string, bytes: Buffer) => {
      const content = Buffer.concat([Buffer.from(`${kind} ${bytes.length}\0`), bytes]);
      const identifier = createHash('sha1').update(content).digest('hex');
      mkdirSync(path.join(objectDirectory, identifier.slice(0, 2)), {recursive: true});
      writeFileSync(
        path.join(objectDirectory, identifier.slice(0, 2), identifier.slice(2)),
        deflateSync(content)
      );
      return identifier;
    };
    try {
      const blob = writeObject('blob', Buffer.from('Original'));
      const tree = writeObject(
        'tree',
        Buffer.concat([Buffer.from('100644 source with spaces.ts\0'), Buffer.from(blob, 'hex')])
      );
      const commit = writeObject('commit', Buffer.from(`tree ${tree}\n\nFixture\n`));
      writeFileSync(path.join(repositoryDirectory, '.git/HEAD'), `${commit}\n`);
      writeFileSync(path.join(repositoryDirectory, 'source with spaces.ts'), 'Original');
      const clean = readSourceIdentity('/tmp/ax-journeys-unused-output', repositoryDirectory);
      expect(clean.dirty).toBe(false);
      writeFileSync(path.join(repositoryDirectory, 'source with spaces.ts'), 'Changed');
      const changed = readSourceIdentity('/tmp/ax-journeys-unused-output', repositoryDirectory);
      expect(changed.dirty).toBe(true);
      expect(changed.changedPaths).toEqual(['source with spaces.ts']);
      expect(changed.treeHash).not.toBe(clean.treeHash);
      rmSync(path.join(repositoryDirectory, 'source with spaces.ts'));
      writeFileSync(path.join(repositoryDirectory, 'new.ts'), 'New');
      const deleted = readSourceIdentity('/tmp/ax-journeys-unused-output', repositoryDirectory);
      expect(deleted.changedPaths).toEqual(['source with spaces.ts']);
      expect(deleted.untrackedPaths).toEqual(['new.ts']);
    } finally {
      rmSync(repositoryDirectory, {recursive: true, force: true});
    }
  });

  test('defaults to a timestamped release folder and refuses historical or existing evidence', () => {
    const options = parseOptions(['--release', 'v-next']);
    expect(options.out).toContain('/releases/v-next/');
    expect(options.out).not.toContain('/baseline');
    for (const argumentsList of [
      ['--out', 'dev-docs/research/analysis-workflow-ux-2026-11-14/baseline'],
      ['--out', 'dev-docs/research/analysis-workflow-ux-2026-11-14/baseline/nested'],
      ['--out', 'examples/deck/kepler-compute/scripts'],
      ['--release', '../baseline'],
      ['--rows', 'NaN'],
      ['--timeout-ms', '0'],
      ['--only']
    ])
      expect(() => parseOptions(argumentsList)).toThrow();
  });

  test('records source and dirty identity without invoking git', () => {
    const identity = readSourceIdentity('/tmp/ax-journeys-unused-output');
    expect(identity.treeHash).toMatch(/^[a-f0-9]{64}$/);
    expect(identity.indexHash).toMatch(/^[a-f0-9]{64}$/);
    expect(identity.files.some(file => file.path.endsWith('ax-journey-model.mjs'))).toBe(true);
    expect(identity.headTree).toMatch(/^[a-f0-9]{40}$/);
    expect(typeof identity.dirty).toBe('boolean');
  });
});
