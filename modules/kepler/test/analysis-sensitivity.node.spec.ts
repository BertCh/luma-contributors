// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {createElement, isValidElement, type ReactElement, type ReactNode} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {afterEach, expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  AnalysisSensitivitySession,
  replayAnalysisSensitivity,
  parseAnalysisSensitivityProvenance,
  getAnalysisSensitivityUnavailableReason,
  getCatalogBinding,
  runCatalogOperation,
  getWorkflowOperationRegistry,
  WorkflowRunner,
  exportWorkflowJSON,
  importWorkflowJSON,
  parseWorkflowCollection,
  serializeWorkflowCollection,
  createCuratedWorkflowAnalysisTemplate,
  KEPLER_CURATED_ANALYSES,
  KeplerDatasetRegistry,
  makeWorkflowRunReport,
  type WorkflowDocument,
  type WorkflowRunRecord,
  type WorkflowRunnerState
} from '../src/index';
import {
  createAnalysisSensitivityMaps,
  getAnalysisSensitivityRadiusValues,
  getAnalysisSensitivityRuntimeKey,
  AnalysisSensitivityCleanupOwner
} from '../../../examples/deck/kepler-compute/src/analysis-sensitivity-model';
import {
  AnalysisSensitivityStrip,
  AnalysisSensitivityCleanupStatus,
  type AnalysisSensitivityControl
} from '../../../examples/deck/kepler-compute/src/analysis-sensitivity-strip';
import {
  recordCatalogInvocation,
  clearCatalogInvocations,
  listCatalogInvocations,
  serializeCatalogInvocationRecords,
  parseCatalogInvocationRecords
} from '../../../examples/deck/kepler-compute/src/catalog-invocation-store';
import {
  collectSandboxExtensions,
  applySandboxExtensions,
  validateSandboxExtensions
} from '../../../examples/deck/kepler-compute/src/sandbox-host-extensions';
import {AnalysisCatalogPanel} from '../../../examples/deck/kepler-compute/src/analysis-catalog-panel';
import {AnalysisSheetParameter} from '../../../examples/deck/kepler-compute/src/analysis-sheet';
import {AnalysisResultCard} from '../../../examples/deck/kepler-compute/src/analysis-result-card';
import {
  makeAnalysisResultWorkflow,
  createAnalysisResultCardModel
} from '../../../examples/deck/kepler-compute/src/analysis-result-card-model';
import {makeFixtureHost} from './workflows/workflow-fixtures';

const INPUTS = {
  points: {
    type: 'FeatureCollection' as const,
    features: [
      {id: 'first', geometry: {type: 'Point', coordinates: [0, 0]}},
      {id: 'second', geometry: {type: 'Point', coordinates: [10, 10]}}
    ]
  }
};
const PARAMETERS = {coordinateSystem: 'planar', radius: 2, segmentsPerQuarter: 1};
const VALUES = [1, 2, 3, 4, 5, 6];
function makeFixture(
  budget = 110,
  scope: 'full-population' | 'sample' | 'viewport' = 'full-population'
) {
  const original = getCatalogBinding('geometry-point-buffer')!;
  const execute = vi.fn(async parameters => {
    const output = runCatalogOperation(original, parameters, INPUTS);
    // An injected inference column proves suppression actively removes significance evidence.
    output.tables[0].columns['significance_class'] = ['hot', 'cold'];
    output.tables[0].columnTypes['significance_class'] = 'string';
    return output;
  });
  const initialize = vi.fn(async () => undefined);
  const destroy = vi.fn(async () => undefined);
  const create = vi.fn(async () => ({execute, initialize, destroy}));
  const binding = {
    ...original,
    planPreview: vi.fn(() => ({key: 'one-graph', gpuByteLength: 30, cpuByteLength: 10, create}))
  };
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: budget});
  let current = true;
  const options = {
    binding,
    registry,
    device: new NullDevice({}),
    inputs: INPUTS,
    parameters: PARAMETERS,
    parameter: 'radius',
    values: VALUES,
    revisionKey: 'points@1:all',
    population: {
      scope,
      rowCount: 2,
      sourceRowCount: scope === 'full-population' ? 2 : 20,
      selectionKey: 'all-or-sample-42'
    },
    isCurrent: () => current
  };
  return {
    options,
    session: new AnalysisSensitivitySession(options),
    registry,
    create,
    execute,
    initialize,
    destroy,
    invalidate: () => {
      current = false;
    }
  };
}
afterEach(() => {
  vi.restoreAllMocks();
  clearCatalogInvocations();
});

test('six settings share one executor and admission; retained maps follow an independent square oracle', async () => {
  const fixture = makeFixture();
  const frames = (await fixture.session.run())!;
  expect(frames.map(frame => frame.value)).toEqual(VALUES);
  expect(fixture.create).toHaveBeenCalledOnce();
  expect(fixture.initialize).toHaveBeenCalledOnce();
  expect(fixture.execute).toHaveBeenCalledTimes(6);
  expect(fixture.registry.memoryUsage.reservedBytes).toEqual({gpu: 30, cpu: 80});
  const maps = createAnalysisSensitivityMaps(frames);
  expect(maps.maps).toHaveLength(6);
  expect(new Set(maps.maps.map(map => map.paths.join(' '))).size).toBe(6);
  expect(maps.viewBox.split(' ').map(Number)).toEqual([-7.1, -17.1, 24.2, 24.2]);
  frames.forEach(frame =>
    expect(frame.result.tables[0].columns['area'][0]).toBeCloseTo(2 * frame.value ** 2, 10)
  );
  expect(await fixture.session.run()).toBe(frames);
  await fixture.session.destroy();
  expect(fixture.destroy).toHaveBeenCalledOnce();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
});

test('insufficient memory fails before allocation or compilation', async () => {
  const fixture = makeFixture(109);
  await expect(fixture.session.run()).rejects.toThrow();
  expect(fixture.create).not.toHaveBeenCalled();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
  await fixture.session.destroy();
});

test.each([
  'cancel',
  'revision',
  'device'
])('%s during execution rejects partial strips and fences disposal', async mode => {
  const fixture = makeFixture();
  let finish!: () => void;
  let start!: () => void;
  const pending = new Promise<void>(resolve => {
    finish = resolve;
  });
  const started = new Promise<void>(resolve => {
    start = resolve;
  });
  fixture.execute.mockImplementationOnce(async parameters => {
    start();
    await pending;
    return runCatalogOperation(fixture.options.binding, parameters, INPUTS);
  });
  const execution = fixture.session.run();
  await started;
  if (mode === 'cancel') fixture.session.cancel();
  else if (mode === 'revision') fixture.invalidate();
  else fixture.registry.handleDeviceLoss();
  const destruction = fixture.session.destroy();
  expect(fixture.destroy).not.toHaveBeenCalled();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(mode === 'device' ? 0 : 110);
  finish();
  expect(await execution).toBeUndefined();
  await destruction;
  expect(fixture.execute).toHaveBeenCalledOnce();
  expect(() => fixture.session.choose(1)).toThrow('chosenValue');
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
});

test.each([
  'sample',
  'viewport'
] as const)('%s suppresses significance attributes and retains actual population evidence', async scope => {
  const fixture = makeFixture(110, scope);
  const frames = (await fixture.session.run())!;
  for (const frame of frames) {
    expect(Object.keys(frame.result.tables[0].columns)).toEqual(['geometry']);
    expect(frame.result.diagnostics).toEqual([]);
    expect(JSON.stringify(frame.result)).not.toContain('significance_class');
  }
  expect(fixture.session.choose(3).population).toEqual(fixture.options.population);
  await fixture.session.destroy();
});

test('sandbox serialization and replay preserve the complete provenance before any new choice', async () => {
  const fixture = makeFixture();
  await fixture.session.run();
  const provenance = fixture.session.choose(4);
  recordCatalogInvocation({
    operationId: fixture.options.binding.id,
    invocation: provenance.invocation,
    inputs: {points: {mode: 'sample'}},
    engine: 'auto',
    sensitivity: provenance
  });
  const serialized = serializeCatalogInvocationRecords();
  expect(
    parseCatalogInvocationRecords(JSON.parse(JSON.stringify(serialized)))[0].sensitivity
  ).toEqual(provenance);
  const extensions = {catalogInvocations: serialized!};
  expect(validateSandboxExtensions(extensions).catalog[0].sensitivity?.chosenValue).toBe(4);
  clearCatalogInvocations();
  applySandboxExtensions(extensions, () => {});
  expect(listCatalogInvocations()[0].sensitivity).toEqual(provenance);
  expect(collectSandboxExtensions([])?.['catalogInvocations']).toEqual(serialized);
  await fixture.session.destroy();
  const replay = replayAnalysisSensitivity({...fixture.options, provenance});
  expect(replay.provenance).toEqual(provenance);
  expect((await replay.run())?.map(frame => frame.value)).toEqual(VALUES);
  expect(replay.provenance).toEqual(provenance);
  await replay.destroy();
  expect(() =>
    replayAnalysisSensitivity({...fixture.options, revisionKey: 'points@2', provenance})
  ).toThrow('replay inputs');
  expect(() =>
    replayAnalysisSensitivity({
      ...fixture.options,
      inputs: {points: {...INPUTS.points, features: []}},
      provenance
    })
  ).toThrow('replay inputs');
  expect(() => parseAnalysisSensitivityProvenance({...provenance, chosenValue: 20})).toThrow(
    'provenance'
  );
  expect(() =>
    parseAnalysisSensitivityProvenance({...provenance, values: [1, 1, 2, 3, 4, 5]})
  ).toThrow('provenance');
});

test('unsupported parameters and changing compile keys are rejected before allocation', () => {
  const fixture = makeFixture();
  expect(
    getAnalysisSensitivityUnavailableReason(fixture.options.binding, 'segmentsPerQuarter')
  ).toMatch(/no qualified/);
  expect(
    () => new AnalysisSensitivitySession({...fixture.options, parameter: 'segmentsPerQuarter'})
  ).toThrow('qualified');
  expect(
    () =>
      new AnalysisSensitivitySession({
        ...fixture.options,
        binding: {
          ...fixture.options.binding,
          planPreview: (_inputs, parameters) => ({
            key: String(parameters['radius']),
            gpuByteLength: 30,
            cpuByteLength: 10,
            create: fixture.create
          })
        }
      })
  ).toThrow('one compile-time plan');
  expect(fixture.create).not.toHaveBeenCalled();
});

test('the view states the preview policy and sweep values respect the binding limits', () => {
  const fixture = makeFixture();
  const markup = renderToStaticMarkup(
    createElement(AnalysisSensitivityStrip, {
      control: {...fixture.options, readInputs: async () => INPUTS, onChoose: () => {}}
    })
  );
  expect(markup).toContain('Explore Radius');
  expect(markup).toContain('Exploratory parameter search');
  expect(markup).toContain('Full selected population');
  for (const radius of [1e-9, 2, 1e7]) {
    const values = getAnalysisSensitivityRadiusValues(radius);
    expect(new Set(values).size).toBe(6);
    expect(Math.min(...values)).toBeGreaterThanOrEqual(1e-9);
    expect(Math.max(...values)).toBeLessThanOrEqual(1e7);
  }
});

test('workflow reports retain the chosen setting, population and error-control caution', async () => {
  const fixture = makeFixture();
  await fixture.session.run();
  const sensitivity = fixture.session.choose(4);
  const record: WorkflowRunRecord = {
    schemaVersion: 1,
    runId: 'search',
    workflowId: 'fixture',
    startedAt: 0,
    durationMilliseconds: 1,
    status: 'done',
    steps: [
      {
        stepId: 'buffer',
        operationId: fixture.options.binding.id,
        key: 'key',
        status: 'done',
        cacheHit: false,
        durationMilliseconds: 1,
        inputs: [],
        sensitivity
      }
    ]
  };
  const document = {
    id: 'fixture',
    title: 'Search',
    steps: [],
    inputs: []
  } as unknown as WorkflowDocument;
  const state = {steps: {}, issues: []} as unknown as WorkflowRunnerState;
  const report = makeWorkflowRunReport(
    document,
    {get: () => undefined, list: () => []},
    state,
    record
  );
  expect(report).toContain('radius = 1, 2, 3, 4, 5, 6; chosen 4');
  expect(report).toContain(
    'full-population; 2/2 rows; selection all-or-sample-42; revision points@1:all'
  );
  expect(report).toContain('does not control error across this parameter search');
  await fixture.session.destroy();
});

// Real handlers/effect lifetimes with deterministic state; no browser timing claim.
function createComponentProbe<Properties>(
  component: (properties: Properties) => React.JSX.Element | null
) {
  const hostRequire = createRequire(
    new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
  );
  const react = hostRequire('react') as {
    __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
  };
  const states: unknown[] = [];
  const references: {current: unknown}[] = [];
  const dependencies = new Map<number, readonly unknown[]>();
  const cleanups = new Map<number, () => void>();
  const flatten = (node: ReactNode): ReactElement<Record<string, unknown>>[] =>
    Array.isArray(node)
      ? node.flatMap(flatten)
      : isValidElement<Record<string, unknown>>(node)
        ? [node, ...flatten(node.props['children'] as ReactNode)]
        : [];
  return {
    render(properties: Properties) {
      let stateIndex = 0;
      let referenceIndex = 0;
      let effectIndex = 0;
      const effects: (() => void)[] = [];
      const internals = react.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
      const previous = internals.H;
      internals.H = {
        useState(initial: unknown) {
          const index = stateIndex++;
          if (!(index in states))
            states[index] = typeof initial === 'function' ? initial() : initial;
          return [
            states[index],
            (next: unknown) => {
              states[index] = typeof next === 'function' ? next(states[index]) : next;
            }
          ];
        },
        useRef(initial: unknown) {
          const index = referenceIndex++;
          references[index] ??= {current: initial};
          return references[index];
        },
        useMemo: (factory: () => unknown) => factory(),
        useSyncExternalStore: (_subscribe: unknown, snapshot: () => unknown) => snapshot(),
        useEffect(effect: () => void | (() => void), next: readonly unknown[]) {
          const index = effectIndex++;
          const previousDependencies = dependencies.get(index);
          if (
            previousDependencies?.length === next.length &&
            next.every((value, position) => Object.is(value, previousDependencies[position]))
          )
            return;
          dependencies.set(index, next);
          effects.push(() => {
            cleanups.get(index)?.();
            const cleanup = effect();
            if (cleanup) cleanups.set(index, cleanup);
            else cleanups.delete(index);
          });
        }
      };
      try {
        const tree = component(properties);
        for (const effect of effects) effect();
        return flatten(tree);
      } finally {
        internals.H = previous;
      }
    },
    unmount() {
      for (const cleanup of cleanups.values()) cleanup();
    }
  };
}

test('choosing the current radius preserves evidence through the sheet, history reload and workflow execution', async () => {
  vi.useFakeTimers();
  const binding = getCatalogBinding('geometry-point-buffer')!;
  const execute = vi.fn();
  vi.spyOn(binding, 'planPreview').mockImplementation(inputs => {
    execute.mockImplementation(async parameters =>
      runCatalogOperation(binding, parameters, inputs)
    );
    return {
      key: 'retained',
      gpuByteLength: 1,
      cpuByteLength: 1,
      create: async () => ({execute, destroy: async () => undefined})
    };
  });
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
  const properties = {datasets: [], device: new NullDevice({}), registry, onPublish: vi.fn()};
  const probe = createComponentProbe(AnalysisCatalogPanel);
  let sweep: AnalysisSensitivitySession | undefined;
  try {
    let nodes = probe.render(properties);
    (
      nodes.find(node => node.props['id'] === 'catalog-operation')!.props['onChange'] as (
        event: unknown
      ) => void
    )({target: {value: binding.id}});
    nodes = probe.render(properties);
    const radius = nodes.find(
      node =>
        node.type === AnalysisSheetParameter &&
        (node.props['spec'] as {name: string}).name === 'radius'
    )!;
    const control = radius.props['sensitivity'] as AnalysisSensitivityControl;
    expect(control).toBeTruthy();
    const currentRadius = Number(control.parameters['radius']);
    const inputs = await control.readInputs();
    const rowCount = inputs['points'].features.length;
    sweep = new AnalysisSensitivitySession({
      ...control,
      inputs,
      parameter: 'radius',
      values: [1, 1.1, 1.2, 1.3, 1.4, 1.5].map(factor => currentRadius * factor),
      population: {
        scope: 'full-population',
        rowCount,
        sourceRowCount: rowCount,
        selectionKey: control.revisionKey
      }
    });
    await sweep.run();
    const provenance = sweep.choose(currentRadius);
    control.onChoose(provenance);
    nodes = probe.render(properties);
    await vi.runAllTimersAsync();
    nodes = probe.render(properties);
    const card = nodes.find(node => node.type === AnalysisResultCard)!.props as Parameters<
      typeof AnalysisResultCard
    >[0];
    expect(card.model.result.sensitivity).toEqual(provenance);
    expect(card.model.context.parameterSearch).toBe(true);
    expect(JSON.parse(card.model.json).result.sensitivity.chosenValue).toBe(currentRadius);
    expect(listCatalogInvocations().at(-1)?.sensitivity).toEqual(provenance);
    expect(execute).toHaveBeenCalledTimes(7);
    await (
      nodes.find(node => node.type === 'button' && node.props['children'] === 'Load into form')!
        .props['onClick'] as () => Promise<void>
    )();
    nodes = probe.render(properties);
    (
      nodes.find(node => node.props['data-testid'] === 'catalog-run')!.props[
        'onClick'
      ] as () => void
    )();
    await vi.runAllTimersAsync();
    nodes = probe.render(properties);
    const restoredCard = nodes.find(node => node.type === AnalysisResultCard)!.props as Parameters<
      typeof AnalysisResultCard
    >[0];
    expect(restoredCard.model.result.sensitivity).toEqual(provenance);
    expect(restoredCard.model.context.parameterSearch).toBe(true);
    const operations = getWorkflowOperationRegistry();
    const document = importWorkflowJSON(
      exportWorkflowJSON(makeAnalysisResultWorkflow(card.model, inputs, operations))
    );
    expect(document.steps.find(step => step.operationId === binding.id)?.sensitivity).toEqual(
      provenance
    );
    document.settings.engine = 'cpu';
    const runner = new WorkflowRunner({registry: operations, host: makeFixtureHost()});
    try {
      runner.setDocument(document);
      const record = await runner.run();
      expect(record.status).toBe('done');
      expect(record.steps.find(step => step.operationId === binding.id)?.sensitivity).toEqual(
        provenance
      );
      expect(makeWorkflowRunReport(document, operations, runner.getState(), record)).toContain(
        'Exploratory parameter search'
      );
    } finally {
      runner.destroy();
    }
  } finally {
    await sweep?.destroy();
    probe.unmount();
    await vi.runAllTimersAsync();
    vi.useRealTimers();
  }
  expect(registry.memoryUsage.totalBytes).toBe(0);
});

test('pre-aborted sweeps allocate nothing; failed cleanup retains admission until retry', async () => {
  const aborted = makeFixture();
  const controller = new AbortController();
  controller.abort();
  expect(await aborted.session.run(controller.signal)).toBeUndefined();
  expect(aborted.create).not.toHaveBeenCalled();
  await aborted.session.destroy();
  const fixture = makeFixture();
  await fixture.session.run();
  fixture.destroy.mockRejectedValueOnce(new Error('cleanup failed'));
  await expect(fixture.session.destroy()).rejects.toThrow('cleanup failed');
  expect(fixture.registry.memoryUsage.totalBytes).toBe(110);
  await fixture.session.destroy();
  expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
  expect(fixture.destroy).toHaveBeenCalledTimes(2);
});

test('completed sweeps cannot be chosen after revision or device replacement', async () => {
  const revision = makeFixture();
  await revision.session.run();
  revision.invalidate();
  expect(() => revision.session.choose(2)).toThrow('current sweep');
  await revision.session.destroy();
  const device = makeFixture();
  await device.session.run();
  device.registry.handleDeviceLoss();
  device.registry.recoverDevice();
  expect(() => device.session.choose(2)).toThrow('current sweep');
  await device.session.destroy();
});

test('north-first circle vertices render clockwise from the top', () => {
  const result = runCatalogOperation(
    getCatalogBinding('geometry-point-buffer')!,
    PARAMETERS,
    INPUTS
  );
  result.tables[0].columns['geometry'] = [
    JSON.stringify({
      type: 'Polygon',
      coordinates: [
        [
          [0, 2],
          [2, 0],
          [0, -2],
          [-2, 0],
          [0, 2]
        ]
      ]
    })
  ];
  expect(createAnalysisSensitivityMaps([{value: 2, result}]).maps[0].paths[0]).toBe(
    'M0,-2 L2,0 L0,2 L-2,0 L0,-2 Z'
  );
});
function readSensitivityRun(
  probe: ReturnType<typeof createComponentProbe<{control: AnalysisSensitivityControl}>>,
  control: AnalysisSensitivityControl
) {
  const child = probe.render({control})[0];
  return {
    key: child.key,
    component: child.type as (properties: {
      control: AnalysisSensitivityControl;
    }) => React.JSX.Element,
    properties: child.props as {control: AnalysisSensitivityControl}
  };
}
async function completeSensitivityMaps(
  probe: ReturnType<typeof createComponentProbe<{control: AnalysisSensitivityControl}>>,
  properties: {control: AnalysisSensitivityControl}
) {
  const nodes = probe.render(properties);
  (
    nodes.find(node => node.props['data-testid'] === 'ax-sensitivity-open')!.props[
      'onClick'
    ] as () => void
  )();
  for (let index = 0; index < 100; index++) await Promise.resolve();
  const completed = probe.render(properties);
  expect(completed.filter(node => node.props['data-testid'] === 'ax-sensitivity-map')).toHaveLength(
    6
  );
  return completed;
}

test.each([
  'device',
  'registry',
  'device recovery'
])('%s replacement disposes the predecessor and remounts a usable strip', async replacement => {
  const fixture = makeFixture();
  const cleanupOwner = new AnalysisSensitivityCleanupOwner();
  const control: AnalysisSensitivityControl = {
    ...fixture.options,
    cleanupOwner,
    readInputs: async () => INPUTS,
    onChoose: vi.fn()
  };
  const wrapper = createComponentProbe(AnalysisSensitivityStrip);
  const first = readSensitivityRun(wrapper, control);
  const oldProbe = createComponentProbe(first.component);
  await completeSensitivityMaps(oldProbe, first.properties);
  expect(readSensitivityRun(wrapper, control).key).toBe(first.key);
  let nextControl = control;
  if (replacement === 'device') nextControl = {...control, device: new NullDevice({})};
  else if (replacement === 'registry')
    nextControl = {...control, registry: new KeplerDatasetRegistry({memoryBudgetBytes: 110})};
  else {
    control.registry.handleDeviceLoss();
    control.registry.recoverDevice();
  }
  const next = readSensitivityRun(wrapper, nextControl);
  expect(next.key).not.toBe(first.key);
  expect(getAnalysisSensitivityRuntimeKey(nextControl)).not.toBe(first.key);
  oldProbe.unmount();
  await cleanupOwner.retry();
  expect(fixture.destroy).toHaveBeenCalledOnce();
  const nextProbe = createComponentProbe(next.component);
  try {
    const initial = nextProbe.render(next.properties);
    expect(initial.some(node => node.props['data-testid'] === 'ax-sensitivity-map')).toBe(false);
    expect(
      initial.find(node => node.props['data-testid'] === 'ax-sensitivity-open')?.props['disabled']
    ).toBe(false);
    const completed = await completeSensitivityMaps(nextProbe, next.properties);
    (
      completed.find(node => node.type === 'button' && Array.isArray(node.props['children']))!
        .props['onClick'] as () => void
    )();
    expect(control.onChoose).toHaveBeenCalledOnce();
    expect(fixture.create).toHaveBeenCalledTimes(2);
  } finally {
    nextProbe.unmount();
    wrapper.unmount();
    await cleanupOwner.retry();
    await fixture.session.destroy();
  }
  expect(cleanupOwner.pendingCount).toBe(0);
  expect(control.registry.memoryUsage.totalBytes).toBe(0);
  expect(nextControl.registry.memoryUsage.totalBytes).toBe(0);
});

test('failed unmount cleanup transfers ownership, reports errors and retries automatically and manually', async () => {
  vi.useFakeTimers();
  const fixture = makeFixture();
  const cleanupOwner = new AnalysisSensitivityCleanupOwner();
  const notification = vi.fn();
  const unsubscribe = cleanupOwner.subscribe(notification);
  const control: AnalysisSensitivityControl = {
    ...fixture.options,
    cleanupOwner,
    readInputs: async () => INPUTS,
    onChoose: () => {}
  };
  const wrapper = createComponentProbe(AnalysisSensitivityStrip);
  const run = readSensitivityRun(wrapper, control);
  const probe = createComponentProbe(run.component);
  try {
    await completeSensitivityMaps(probe, run.properties);
    fixture.destroy
      .mockRejectedValueOnce(new Error('first disposal failed'))
      .mockRejectedValueOnce(new Error('retry disposal failed'));
    probe.unmount();
    wrapper.unmount();
    expect(cleanupOwner.pendingCount).toBe(1);
    await cleanupOwner.retry();
    expect(fixture.registry.memoryUsage.totalBytes).toBe(110);
    expect(cleanupOwner.failures).toHaveLength(1);
    expect(notification).toHaveBeenCalled();
    const markup = renderToStaticMarkup(
      createElement(AnalysisSensitivityCleanupStatus, {owner: cleanupOwner})
    );
    expect(markup).toContain('Sensitivity cleanup failed');
    expect(markup).toContain('first disposal failed');
    expect(markup).toContain('Retry sensitivity cleanup');
    await vi.advanceTimersByTimeAsync(250);
    expect(fixture.destroy).toHaveBeenCalledTimes(2);
    expect(cleanupOwner.pendingCount).toBe(1);
    expect(fixture.registry.memoryUsage.totalBytes).toBe(110);
    const status = createComponentProbe(AnalysisSensitivityCleanupStatus);
    const nodes = status.render({owner: cleanupOwner});
    (nodes.find(node => node.type === 'button')!.props['onClick'] as () => void)();
    await cleanupOwner.retry();
    status.unmount();
    expect(cleanupOwner.pendingCount).toBe(0);
    expect(cleanupOwner.failures).toEqual([]);
    expect(fixture.destroy).toHaveBeenCalledTimes(3);
    expect(fixture.registry.memoryUsage.totalBytes).toBe(0);
  } finally {
    unsubscribe();
    await cleanupOwner.retry();
    await fixture.session.destroy();
    vi.useRealTimers();
  }
});

test('cached and replanned results use only current step search evidence, including its absence', async () => {
  const fixture = makeFixture();
  const sensitivity = parseAnalysisSensitivityProvenance({
    ...fixture.session.provenance,
    chosenValue: 4
  });
  const anotherSearch = parseAnalysisSensitivityProvenance({
    ...sensitivity,
    values: [2, 3, 4, 5, 6, 7],
    population: {...sensitivity.population, selectionKey: 'another-search'}
  });
  const binding = getCatalogBinding('geometry-point-buffer')!;
  const operations = getWorkflowOperationRegistry();
  const result = runCatalogOperation(binding, {...PARAMETERS, radius: 4}, INPUTS);
  const document = makeAnalysisResultWorkflow(
    createAnalysisResultCardModel(result),
    INPUTS,
    operations
  );
  document.settings.engine = 'cpu';
  const original = document.steps.find(step => step.operationId === binding.id)!;
  const runner = new WorkflowRunner({registry: operations, host: makeFixtureHost()});
  try {
    runner.setDocument(document);
    await runner.run();
    const firstOutput = runner.getState().steps[original.id].result!.outputs;
    document.steps.push(
      {...structuredClone(original), id: 'searched-copy', sensitivity},
      {...structuredClone(original), id: 'plain-copy'},
      {...structuredClone(original), id: 'other-search-copy', sensitivity: anotherSearch}
    );
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe('done');
    for (const stepId of ['searched-copy', 'plain-copy', 'other-search-copy'])
      expect(record.steps.find(step => step.stepId === stepId)?.cacheHit).toBe(true);
    expect(runner.getState().steps['searched-copy'].result?.sensitivity).toEqual(sensitivity);
    expect(runner.getState().steps['plain-copy'].result?.sensitivity).toBeUndefined();
    expect(runner.getState().steps['other-search-copy'].result?.sensitivity).toEqual(anotherSearch);
    original.sensitivity = anotherSearch;
    delete document.steps.find(step => step.id === 'searched-copy')!.sensitivity;
    document.steps.find(step => step.id === 'plain-copy')!.sensitivity = sensitivity;
    delete document.steps.find(step => step.id === 'other-search-copy')!.sensitivity;
    runner.setDocument(document);
    expect(runner.getState().steps[original.id].result?.sensitivity).toEqual(anotherSearch);
    expect(runner.getState().steps[original.id].result?.outputs).toBe(firstOutput);
    expect(runner.getState().steps['searched-copy'].result?.sensitivity).toBeUndefined();
    expect(runner.getState().steps['plain-copy'].result?.sensitivity).toEqual(sensitivity);
    expect(runner.getState().steps['other-search-copy'].result?.sensitivity).toBeUndefined();
    original.parameters['radius'] = {from: 'static', value: 8};
    runner.setDocument(document);
    await runner.run();
    original.parameters['radius'] = {from: 'static', value: 4};
    original.sensitivity = sensitivity;
    runner.setDocument(document);
    expect(runner.getState().steps[original.id].result?.sensitivity).toEqual(sensitivity);
    expect(runner.getState().steps[original.id].result?.cacheHit).toBe(true);
  } finally {
    runner.destroy();
    await fixture.session.destroy();
  }
});

test('sensitivity serializes beside current curated identities and both template formats', async () => {
  const fixture = makeFixture();
  const sensitivity = parseAnalysisSensitivityProvenance({
    ...fixture.session.provenance,
    chosenValue: 4
  });
  const operations = getWorkflowOperationRegistry();
  const result = {
    ...runCatalogOperation(
      getCatalogBinding('geometry-point-buffer')!,
      {...PARAMETERS, radius: 4},
      INPUTS
    ),
    sensitivity
  };
  const document = makeAnalysisResultWorkflow(
    createAnalysisResultCardModel(result),
    INPUTS,
    operations
  );
  const step = document.steps.find(candidate => candidate.operationId === result.operationId)!;
  step.curatedAnalysisId = 'fixture-curated-identity';
  const template = createCuratedWorkflowAnalysisTemplate(
    KEPLER_CURATED_ANALYSES.find(analysis => analysis.id === 'density-surface')!
  );
  const collection = {
    schemaVersion: 1 as const,
    workflows: [document],
    activeWorkflowId: document.id,
    templates: [document],
    analysisTemplates: [template]
  };
  const restored = parseWorkflowCollection(
    JSON.parse(JSON.stringify(serializeWorkflowCollection(collection)))
  );
  const restoredStep = restored.workflows[0].steps.find(candidate => candidate.id === step.id)!;
  expect(restoredStep.curatedAnalysisId).toBe(step.curatedAnalysisId);
  expect(restoredStep.sensitivity).toEqual(sensitivity);
  expect(
    restored.templates?.[0].steps.find(candidate => candidate.id === step.id)?.sensitivity
  ).toEqual(sensitivity);
  expect(restored.analysisTemplates).toEqual([template]);
  const malformed = structuredClone(document);
  malformed.steps.find(candidate => candidate.id === step.id)!.sensitivity = {
    ...sensitivity,
    operationId: 'not-the-buffer'
  };
  expect(() => importWorkflowJSON(JSON.stringify(malformed))).toThrow('sensitivity');
  await fixture.session.destroy();
});
