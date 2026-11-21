// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import type {Locator} from 'playwright';
import type {ReactElement, ReactNode} from 'react';
import type KeplerTable from '@kepler.gl/table';
import {afterEach, describe, expect, test, vi} from 'vitest';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  getCatalogBinding,
  runCatalogOperation,
  KEPLER_CURATED_ANALYSES,
  KeplerDatasetRegistry,
  type KeplerDatasetProfile
} from '../src/index';
import {
  createAnalysisSheetModel,
  describeAnalysisSheetMethod,
  getAnalysisSheetQualificationCases,
  getAnalysisSheetPreviewDelay,
  ANALYSIS_SHEET_PREVIEW_POLICY
} from '../../../examples/deck/kepler-compute/src/analysis-sheet-model';
import {
  createAnalysisCatalogFormModel,
  resolveAnalysisCatalogInputProfiles
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-form-model';
import {AnalysisCatalogPanel} from '../../../examples/deck/kepler-compute/src/analysis-catalog-panel';
import {AnalysisSheetParameter} from '../../../examples/deck/kepler-compute/src/analysis-sheet';
import {AnalysisResultCard} from '../../../examples/deck/kepler-compute/src/analysis-result-card';
import {
  setAnalysisSheetParameter,
  readAnalysisSheetField
} from '../../../examples/deck/kepler-compute/scripts/analysis-sheet-controls.mjs';
import {WorkbenchParameterValue} from '../../../examples/deck/kepler-compute/src/workflow/workbench-parameter-form';

const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const react = hostRequire('react') as typeof import('react') & {
  __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
};
type Element = ReactElement<Record<string, unknown>>;
function elements(node: ReactNode): Element[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  return react.isValidElement<Record<string, unknown>>(node)
    ? [node, ...elements(node.props['children'] as ReactNode)]
    : [];
}

// Real callbacks and effect lifetimes, with deterministic state; no browser timing claim.
function createProbe<Properties>(component: (properties: Properties) => ReactNode) {
  const states: unknown[] = [];
  const references: {current: unknown}[] = [];
  const dependencies = new Map<number, readonly unknown[]>();
  const cleanups = new Map<number, () => void>();
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
        useId: () => 'probe-parameter',
        useContext: (context: React.Context<unknown>): unknown =>
          Reflect.get(context, '_currentValue'),
        useMemo: (factory: () => unknown) => factory(),
        useSyncExternalStore: (_subscribe: unknown, snapshot: () => unknown) => snapshot(),
        useEffect(effect: () => void | (() => void), next: readonly unknown[]) {
          const index = effectIndex++;
          const previousDependencies = dependencies.get(index);
          if (
            previousDependencies &&
            previousDependencies.length === next.length &&
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
        return elements(tree);
      } finally {
        internals.H = previous;
      }
    },
    unmount() {
      for (const cleanup of cleanups.values()) cleanup();
    }
  };
}
function renderAnalysisSheetParameter(properties: Parameters<typeof AnalysisSheetParameter>[0]) {
  const probe = createProbe(AnalysisSheetParameter);
  try {
    return probe.render(properties);
  } finally {
    probe.unmount();
  }
}
function change(nodes: Element[], name: string, value: string) {
  const control = nodes.find(
    element => element.props['id'] === name || element.props['aria-label'] === name
  )!;
  (control.props['onChange'] as (event: {target: {value: string}}) => void)({target: {value}});
}
function find(nodes: Element[], testId: string) {
  return nodes.find(element => element.props['data-testid'] === testId)!;
}
function click(nodes: Element[], testId: string): Promise<void> {
  return (find(nodes, testId).props['onClick'] as () => Promise<void>)();
}
function model(bindingId: string, overrides: Record<string, unknown> = {}) {
  const binding = getCatalogBinding(bindingId)!;
  const profiles = resolveAnalysisCatalogInputProfiles(binding, {}, () => undefined).profiles;
  const form = createAnalysisCatalogFormModel(binding, profiles, {overrides});
  return {binding, profiles, form, sheet: createAnalysisSheetModel(binding, form, profiles)};
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('generated analysis sheet', () => {
  test('trajectory catalog offers resident scalar observations with manual coordinate mapping', async () => {
    const values: Record<string, unknown[]> = {
      track_id: ['a', 'a', 'a'],
      timestamp: [1767225602000, 1767225600000, 1767225601000],
      east: [-73.98, -74, -73.99],
      north: [40.74, 40.74, 40.74]
    };
    const resident = {
      id: 'trajectory-native-point-observations',
      label: 'Trajectory observations',
      rowCount: 3,
      provider: {
        fields: Object.keys(values).map(name => ({name, sortable: true})),
        maximumPageSize: 16,
        getRevision: () => '1',
        readPage: vi.fn(),
        readColumnValues: vi.fn(async ({column}: {column: string}) => ({
          values: values[column],
          fieldType: column === 'track_id' ? 'string' : 'real'
        }))
      },
      isCurrent: () => true
    };
    const probe = createProbe(AnalysisCatalogPanel);
    const properties = {datasets: [], residentDatasets: [resident], onPublish: vi.fn()};
    try {
      let nodes = probe.render(properties);
      change(nodes, 'catalog-operation', 'trajectory-thinning');
      nodes = probe.render(properties);
      change(nodes, 'Trajectory observations source', 'native');
      nodes = probe.render(properties);
      const selector = nodes.find(
        node => node.props['aria-label'] === 'Trajectory observations dataset'
      )!;
      expect(
        elements(selector).some(
          node => node.type === 'option' && node.props['value'] === resident.id
        )
      ).toBe(true);
      change(nodes, 'Trajectory observations dataset', resident.id);
      nodes = probe.render(properties);
      change(nodes, 'Trajectory observations geometry field', '@point:["",""]');
      nodes = probe.render(properties);
      change(nodes, 'Trajectory observations X / longitude column', 'east');
      nodes = probe.render(properties);
      change(nodes, 'Trajectory observations Y / latitude column', 'north');
      nodes = probe.render(properties);
      for (const [name, value] of Object.entries({
        coordinateSystem: 'EPSG:4326',
        timestampEncoding: 'unix-millisecond',
        tolerance: 0
      })) {
        const parameter = nodes.find(
          node => (node.props['spec'] as {name?: string} | undefined)?.name === name
        )!;
        (parameter.props['onChange'] as (value: string | number) => void)(value);
        nodes = probe.render(properties);
      }
      await click(nodes, 'catalog-check-inputs');
      nodes = probe.render(properties);
      expect(
        find(nodes, 'catalog-run').props['disabled'],
        JSON.stringify(
          nodes
            .filter(node => node.type === 'li' || node.props['data-testid'] === 'catalog-error')
            .map(node => node.props['children'])
        )
      ).toBe(false);
      await click(nodes, 'catalog-run');
      await vi.waitFor(() => {
        nodes = probe.render(properties);
        expect(find(nodes, 'catalog-error')?.props['children']).toBeUndefined();
        expect(find(nodes, 'catalog-result')).toBeDefined();
      });
      const summary = String(find(nodes, 'catalog-result-summary').props['children']);
      expect(summary).toContain('trackCount: 1');
      expect(summary).toContain('fixCount: 3');
      expect(resident.provider.readColumnValues).toHaveBeenCalled();
    } finally {
      probe.unmount();
    }
  });

  test('typed roles use their own input, accept integers as numbers, and retain explicit empty mappings', () => {
    const {binding, profiles} = model('idw-grid');
    const profile = profiles['points']!;
    const typed: KeplerDatasetProfile = {
      ...profile,
      columns: [
        {name: 'observed', type: 'number', rowCount: 10, nullCount: 0},
        {name: 'counts', type: 'integer', rowCount: 10, nullCount: 0},
        {name: 'category', type: 'category', rowCount: 10, nullCount: 0}
      ]
    };
    const form = createAnalysisCatalogFormModel(
      binding,
      {points: typed},
      {overrides: {valueProperty: ''}}
    );
    const sheet = createAnalysisSheetModel(binding, form, {points: typed});
    const role = sheet.rows.find(row => row.specification.name === 'valueProperty')!;
    expect(role.input).toBe('points');
    expect(role.columns).toEqual(['observed', 'counts']);
    expect(role.value).toBe('');
    expect(role.required).toBe(true);
    expect(form.runDisabled).toBe(true);
    expect(sheet.basic.some(row => row === role)).toBe(false);
    expect(sheet.advanced.some(row => row === role)).toBe(false);
  });

  test('conditional settings, prominence and source explanations come from AX resolution', () => {
    const nearest = model('local-moran-statistics', {neighborhood: 'knn'});
    const band = model('local-moran-statistics', {neighborhood: 'distance-band'});
    expect(nearest.sheet.rows.some(row => row.specification.name === 'distanceBand')).toBe(false);
    expect(band.sheet.rows.some(row => row.specification.name === 'distanceBand')).toBe(true);
    expect(band.sheet.rows.some(row => row.specification.name === 'neighborCount')).toBe(false);
    expect(nearest.sheet.advanced.some(row => row.specification.name === 'permutations')).toBe(
      true
    );
    const permutation = nearest.sheet.rows.find(row => row.specification.name === 'permutations')!;
    expect(permutation.source).toContain('999');
    expect(permutation.apply).toBe('rebuild');
    expect(
      nearest.sheet.rows.find(row => row.specification.name === 'multipleTesting')?.apply
    ).toBe('live');
    const edited = model('geometry-point-buffer', {radius: 17});
    expect(edited.sheet.rows.find(row => row.specification.name === 'radius')?.source).toContain(
      'override'
    );
  });

  test('conditionally required field roles change with the controlling parameter', () => {
    const binding = getCatalogBinding('local-lisa-variants')!;
    expect(binding).toBeDefined();
    const profiles = resolveAnalysisCatalogInputProfiles(binding, {}, () => undefined).profiles;
    const build = (variant: string) =>
      createAnalysisSheetModel(
        binding,
        createAnalysisCatalogFormModel(binding, profiles, {overrides: {variant}}),
        profiles
      );
    const single = build('local-getis-ord-g-star');
    const bivariate = build('local-bivariate-moran');
    const second = bivariate.rows.find(
      row =>
        row.specification.fieldRole?.required &&
        typeof row.specification.fieldRole.required === 'object'
    );
    expect(second?.required).toBe(true);
    expect(
      single.rows.find(row => row.specification.name === second?.specification.name)?.required
    ).toBe(false);
  });

  test('duplicate methods state their differences and preview policy applies to full selected inputs', () => {
    expect(describeAnalysisSheetMethod('local-moran-statistics')).toMatch(
      /row-standardized.*999.*binary.*analytic/
    );
    expect(describeAnalysisSheetMethod('kriging-grid')).toMatch(
      /without curve fitting or cross-validation.*offers variogram fitting/
    );
    expect(ANALYSIS_SHEET_PREVIEW_POLICY.scope).toBe('full-population');
    expect(ANALYSIS_SHEET_PREVIEW_POLICY.inference).toMatch(
      /suppress significance.*full-population/
    );
    const cases = getAnalysisSheetQualificationCases();
    expect(cases.map(item => item.templateId)).toEqual(
      KEPLER_CURATED_ANALYSES.filter(item => item.wave === 1).map(item => item.id)
    );
    // The twelve wave-one analyses of the design table plus the drive-time bands recipe.
    expect(cases).toHaveLength(13);
    const density = cases.find(item => item.templateId === 'density-surface')!;
    expect(density.unavailable).toBeUndefined();
    expect(density.live.some(parameter => parameter.name === 'bandwidth')).toBe(true);
    for (const item of cases) {
      const binding = item.bindingId ? getCatalogBinding(item.bindingId) : undefined;
      if (!binding?.planPreview) expect(item.unavailable).toBeTruthy();
    }
  });

  test('shared controls commit rebuild text on blur or Enter and live text immediately', () => {
    expect(getAnalysisSheetPreviewDelay('live')).toBeLessThanOrEqual(16);
    const onCommit = vi.fn();
    const probe = createProbe(WorkbenchParameterValue);
    const properties = {
      spec: {name: 'radius', label: 'Radius', kind: 'number' as const, minimum: 0},
      value: 1,
      testId: 'radius',
      onCommit,
      commitOnRelease: true
    };
    let nodes = probe.render(properties);
    change(nodes, 'Radius', '23');
    expect(onCommit).not.toHaveBeenCalled();
    nodes = probe.render(properties);
    (find(nodes, 'radius').props['onBlur'] as () => void)();
    expect(onCommit).toHaveBeenLastCalledWith(23);
    change(nodes, 'Radius', '-1');
    nodes = probe.render(properties);
    (find(nodes, 'radius').props['onBlur'] as () => void)();
    expect(onCommit).toHaveBeenCalledTimes(1);
    change(nodes, 'Radius', '25');
    nodes = probe.render(properties);
    (find(nodes, 'radius').props['onKeyDown'] as (event: {key: string}) => void)({key: 'Enter'});
    expect(onCommit).toHaveBeenLastCalledWith(25);
    nodes = probe.render({...properties, commitOnRelease: false});
    change(nodes, 'Radius', '24');
    expect(onCommit).toHaveBeenLastCalledWith(24);
    probe.unmount();
  });

  test.each([
    'edit',
    'invalid text',
    'cancel',
    'runtime replacement'
  ])('panel rejects a late retained result after %s and exposes checks and grouping', async mutation => {
    const binding = getCatalogBinding('geometry-point-buffer')!;
    let complete!: () => void;
    let started!: () => void;
    const executing = new Promise<void>(resolve => {
      started = resolve;
    });
    const pending = new Promise<void>(resolve => {
      complete = resolve;
    });
    const destroy = vi.fn(async () => undefined);
    vi.spyOn(binding, 'planPreview').mockImplementation(() => ({
      key: 'fixture',
      gpuByteLength: 1,
      cpuByteLength: 1,
      create: async () => ({
        execute: async () => {
          started();
          await pending;
          return {tables: [], diagnostics: [], summary: {value: 1}};
        },
        destroy
      })
    }));
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    const probe = createProbe(AnalysisCatalogPanel);
    let currentRegistry: KeplerDatasetRegistry | undefined = registry;
    const properties = {
      datasets: [],
      device: new NullDevice({}),
      registry,
      getCurrentRegistry: () => currentRegistry,
      onPublish: vi.fn()
    };
    let nodes = probe.render(properties);
    change(nodes, 'catalog-operation', binding.id);
    nodes = probe.render(properties);
    expect(nodes.some(element => element.props['aria-label'] === 'Key settings')).toBe(true);
    expect(find(nodes, 'ax-all-settings')).toBeDefined();
    expect(find(nodes, 'catalog-preflight')).toBeDefined();
    const run = click(nodes, 'catalog-run');
    await executing;
    nodes = probe.render(properties);
    const radius = nodes.find(
      element => (element.props['spec'] as {name?: string} | undefined)?.name === 'radius'
    )!;
    if (mutation === 'edit') (radius.props['onChange'] as (value: number) => void)(41);
    else if (mutation === 'invalid text')
      (radius.props['onValidityChange'] as (problem: string) => void)('must be positive');
    else if (mutation === 'cancel') await click(nodes, 'ax-preview-cancel');
    else currentRegistry = undefined;
    probe.render(properties);
    complete();
    await run;
    await new Promise(resolve => setTimeout(resolve, 0));
    nodes = probe.render(properties);
    expect(nodes.some(element => element.props['data-testid'] === 'catalog-result')).toBe(false);
    expect(find(nodes, 'catalog-run').props['disabled']).toBe(mutation === 'invalid text');
    probe.unmount();
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(destroy).toHaveBeenCalledOnce();
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });
  test('a failed automatic replacement preserves the previous display and disables keeping it', async () => {
    vi.useFakeTimers();
    const binding = getCatalogBinding('geometry-point-buffer')!;
    const execute = vi
      .fn()
      .mockResolvedValueOnce({tables: [], diagnostics: [], summary: {value: 1}})
      .mockRejectedValueOnce(new Error('Replacement failed'));
    const destroy = vi.fn(async () => undefined);
    vi.spyOn(binding, 'planPreview').mockImplementation(() => ({
      key: 'retained',
      gpuByteLength: 1,
      cpuByteLength: 1,
      create: async () => ({execute, destroy})
    }));
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    const probe = createProbe(AnalysisCatalogPanel);
    const properties = {datasets: [], device: new NullDevice({}), registry, onPublish: vi.fn()};
    let nodes = probe.render(properties);
    change(nodes, 'catalog-operation', binding.id);
    nodes = probe.render(properties);
    await click(nodes, 'catalog-run');
    await vi.runAllTimersAsync();
    nodes = probe.render(properties);
    expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
    const radius = nodes.find(
      element => (element.props['spec'] as {name?: string} | undefined)?.name === 'radius'
    )!;
    (radius.props['onChange'] as (value: number) => void)(42);
    probe.render(properties);
    await vi.runAllTimersAsync();
    nodes = probe.render(properties);
    expect(find(nodes, 'catalog-result').props['data-current']).toBe(false);
    expect(find(nodes, 'catalog-error').props['children']).toBe('Replacement failed');
    expect(execute).toHaveBeenCalledTimes(2);
    expect(registry.memoryUsage.totalBytes).toBe(2);
    await click(nodes, 'ax-clear-result');
    nodes = probe.render(properties);
    await vi.runAllTimersAsync();
    expect(nodes.some(element => element.props['data-testid'] === 'catalog-result')).toBe(false);
    expect(registry.memoryUsage.totalBytes).toBe(0);
    probe.unmount();
    await vi.runAllTimersAsync();
    expect(destroy).toHaveBeenCalledOnce();
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });
  test('replacing a native container with unchanged metadata rebuilds retained inputs after an accepted run', async () => {
    vi.useFakeTimers();
    const binding = getCatalogBinding('geometry-point-buffer')!;
    const createExecutor = vi.fn(async (horizontal: number) => ({
      execute: async () => ({tables: [], diagnostics: [], summary: {horizontal}}),
      destroy: vi.fn(async () => undefined)
    }));
    vi.spyOn(binding, 'planPreview').mockImplementation(inputs => ({
      key: 'same-topology',
      gpuByteLength: 1,
      cpuByteLength: 1,
      create: () =>
        createExecutor(
          (inputs['points'] as {features: {geometry: {coordinates: number[]}}[]}).features[0]
            .geometry.coordinates[0]
        )
    }));
    const makeDataset = (horizontal: number) =>
      ({
        id: 'replaced-native',
        label: 'Replaced native',
        dataRevision: 1,
        fields: [{name: 'geometry'}],
        dataContainer: {
          numRows: () => 1,
          valueAt: () => JSON.stringify({type: 'Point', coordinates: [horizontal, 0]})
        }
      }) as unknown as KeplerTable;
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    const probe = createProbe(AnalysisCatalogPanel);
    let properties = {
      datasets: [makeDataset(0)],
      device: new NullDevice({}),
      registry,
      onPublish: vi.fn()
    };
    try {
      let nodes = probe.render(properties);
      change(nodes, 'catalog-operation', binding.id);
      nodes = probe.render(properties);
      change(nodes, 'Points source', 'native');
      nodes = probe.render(properties);
      change(nodes, 'Points dataset', 'replaced-native');
      nodes = probe.render(properties);
      change(nodes, 'Points geometry field', 'geometry');
      nodes = probe.render(properties);
      const coordinates = nodes.find(
        element =>
          (element.props['spec'] as {name?: string} | undefined)?.name === 'coordinateSystem'
      )!;
      (coordinates.props['onChange'] as (value: string) => void)('planar');
      nodes = probe.render(properties);
      const radius = nodes.find(
        element => (element.props['spec'] as {name?: string} | undefined)?.name === 'radius'
      )!;
      (radius.props['onChange'] as (value: number) => void)(37);
      nodes = probe.render(properties);
      await click(nodes, 'catalog-check-inputs');
      nodes = probe.render(properties);
      expect(
        find(nodes, 'catalog-run').props['disabled'],
        JSON.stringify(
          nodes
            .filter(
              element => element.type === 'li' || element.props['data-testid'] === 'catalog-error'
            )
            .map(element => element.props['children'])
        )
      ).toBe(false);
      await click(nodes, 'catalog-run');
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      expect(find(nodes, 'catalog-result-summary').props['children']).toContain('horizontal: 0');
      const acceptedParameters = find(nodes, 'catalog-result').props['data-parameters'];
      properties = {...properties, datasets: [makeDataset(10)]};
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(false);
      expect(find(nodes, 'catalog-run').props['disabled']).toBe(true);
      await click(nodes, 'catalog-check-inputs');
      nodes = probe.render(properties);
      await click(nodes, 'catalog-run');
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-parameters']).toBe(acceptedParameters);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      expect(find(nodes, 'catalog-result-summary').props['children']).toContain('horizontal: 10');
      expect(createExecutor).toHaveBeenCalledTimes(2);
      expect((await createExecutor.mock.results[0].value).destroy).toHaveBeenCalledOnce();
    } finally {
      probe.unmount();
      await vi.runAllTimersAsync();
    }
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });

  test('focusing and blurring an unchanged rebuild control preserves the accepted result and staging', async () => {
    vi.useFakeTimers();
    const binding = getCatalogBinding('geometry-point-buffer')!;
    const execute = vi.fn();
    vi.spyOn(binding, 'planPreview').mockImplementation((inputs, parameters) => {
      execute.mockImplementation(async () => runCatalogOperation(binding, parameters, inputs));
      return {
        key: 'same-topology',
        gpuByteLength: 1,
        cpuByteLength: 1,
        create: async () => ({execute, destroy: async () => undefined})
      };
    });
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    const probe = createProbe(AnalysisCatalogPanel);
    const controlProbe = createProbe(WorkbenchParameterValue);
    const properties = {datasets: [], device: new NullDevice({}), registry, onPublish: vi.fn()};
    try {
      let nodes = probe.render(properties);
      change(nodes, 'catalog-operation', binding.id);
      nodes = probe.render(properties);
      await click(nodes, 'catalog-run');
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      const parameter = nodes.find(
        element =>
          (element.props['spec'] as {name?: string} | undefined)?.name === 'segmentsPerQuarter'
      )!;
      const parameterNodes = renderAnalysisSheetParameter(
        parameter.props as Parameters<typeof AnalysisSheetParameter>[0]
      );
      const sharedControl = parameterNodes.find(
        element => element.type === WorkbenchParameterValue
      )!;
      let controlNodes = controlProbe.render(
        sharedControl.props as Parameters<typeof WorkbenchParameterValue>[0]
      );
      const control = find(controlNodes, 'catalog-parameter-segmentsPerQuarter');
      (control.props['onFocus'] as () => void)();
      (control.props['onBlur'] as () => void)();
      nodes = probe.render(properties);
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      expect(find(nodes, 'catalog-publish').props['disabled']).toBe(false);
      expect(execute).toHaveBeenCalledOnce();
      // A numerically equivalent edit also keeps the data-derived/default provenance unchanged.
      change(controlNodes, 'Segments per quarter', '08');
      controlNodes = controlProbe.render(
        sharedControl.props as Parameters<typeof WorkbenchParameterValue>[0]
      );
      (find(controlNodes, 'catalog-parameter-segmentsPerQuarter').props['onBlur'] as () => void)();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      expect(execute).toHaveBeenCalledOnce();
      await click(nodes, 'catalog-publish');
      expect(properties.onPublish).toHaveBeenCalledOnce();
    } finally {
      controlProbe.unmount();
      probe.unmount();
      await vi.runAllTimersAsync();
    }
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });
  test('qualification drivers select the real coordinate radio control through the shared helper', async () => {
    const {binding, sheet} = model('geometry-point-buffer');
    const row = sheet.rows.find(candidate => candidate.specification.name === 'coordinateSystem')!;
    const onChange = vi.fn();
    const parameterNodes = renderAnalysisSheetParameter({
      spec: binding.parameters.find(candidate => candidate.name === 'coordinateSystem')!,
      row,
      value: row.value,
      source: row.source,
      onChange
    });
    const sharedControl = parameterNodes.find(element => element.type === WorkbenchParameterValue)!;
    const probe = createProbe(WorkbenchParameterValue);
    try {
      const nodes = probe.render(
        sharedControl.props as Parameters<typeof WorkbenchParameterValue>[0]
      );
      const group = find(nodes, 'catalog-parameter-coordinateSystem');
      const radios = elements(group.props['children'] as ReactNode).filter(
        element => element.type === 'input'
      );
      const selectOption = vi.fn(async () => {
        throw new Error('selectOption cannot set a radio group');
      });
      const control = {
        locator: () => ({all: async () => []}),
        evaluate: async (
          read: (element: {tagName: string; getAttribute: (name: string) => unknown}) => unknown
        ) =>
          read({
            tagName: String(group.type).toUpperCase(),
            getAttribute: name => group.props[name]
          }),
        getByRole: () => ({
          all: async () =>
            radios.map(option => ({
              inputValue: async () => String(option.props['value']),
              check: async () => (option.props['onChange'] as () => void)()
            }))
        }),
        selectOption
      } as unknown as Locator;
      const getByTestId = vi.fn(() => control);
      await setAnalysisSheetParameter(
        {getByTestId} as unknown as Locator,
        'coordinateSystem',
        'planar'
      );
      expect(getByTestId).toHaveBeenCalledWith('catalog-parameter-coordinateSystem');
      expect(selectOption).not.toHaveBeenCalled();
      expect(onChange).toHaveBeenCalledExactlyOnceWith('planar');
      // These drivers formerly called selectOption on the newly generated radio group.
      for (const driver of [
        'ax-result-publication',
        'analysis-catalog',
        'analysis-catalog-pointcloud',
        'save-extensions'
      ]) {
        const source = readFileSync(
          new URL(`../../../examples/deck/kepler-compute/scripts/${driver}.mjs`, import.meta.url),
          'utf8'
        );
        expect(source).toContain("from './analysis-sheet-controls.mjs'");
        expect(source).toContain("setAnalysisSheetParameter(panel, 'coordinateSystem'");
        expect(source).not.toContain('getByLabel(/^Coordinate system/)');
      }
    } finally {
      probe.unmount();
    }
  });

  test('qualification helper opens folded settings and commits rebuild text before returning', async () => {
    const probe = createProbe(WorkbenchParameterValue);
    const onCommit = vi.fn();
    const properties = {
      spec: {
        name: 'segmentsPerQuarter',
        label: 'Segments per quarter',
        kind: 'integer' as const,
        minimum: 1,
        maximum: 64
      },
      value: 8,
      onCommit,
      testId: 'segments',
      commitOnRelease: true
    };
    let nodes = probe.render(properties);
    let open = false;
    const openDisclosure = vi.fn(async () => {
      open = true;
    });
    const disclosure = {
      evaluate: async (read: (element: {open: boolean}) => unknown) => read({open}),
      locator: () => ({click: openDisclosure})
    };
    const press = vi.fn(async (key: string) => {
      expect(key).toBe('Tab');
      expect(onCommit).not.toHaveBeenCalled();
      (find(nodes, 'segments').props['onBlur'] as () => void)();
    });
    const control = {
      locator: () => ({all: async () => [disclosure]}),
      evaluate: async (
        read: (element: {
          tagName: string;
          type: unknown;
          getAttribute: (name: string) => unknown;
        }) => unknown
      ) =>
        read({
          tagName: 'INPUT',
          type: find(nodes, 'segments').props['type'],
          getAttribute: name => find(nodes, 'segments').props[name]
        }),
      fill: async (value: string) => {
        expect(open).toBe(true);
        change(nodes, 'Segments per quarter', value);
        nodes = probe.render(properties);
      },
      press
    } as unknown as Locator;
    try {
      await setAnalysisSheetParameter(
        {getByTestId: () => control} as unknown as Locator,
        'segmentsPerQuarter',
        16
      );
      expect(openDisclosure).toHaveBeenCalledOnce();
      expect(press).toHaveBeenCalledOnce();
      expect(onCommit).toHaveBeenCalledExactlyOnceWith(16);
    } finally {
      probe.unmount();
    }
  });

  test('qualification field helpers preserve every selected column and skip only explicitly optional controls', async () => {
    const selectOption = vi.fn();
    const control = {
      locator: () => ({all: async () => []}),
      evaluate: async (
        read: (element: {
          tagName: string;
          multiple: boolean;
          selectedOptions: {value: string}[];
          getAttribute: () => null;
        }) => unknown
      ) =>
        read({
          tagName: 'SELECT',
          multiple: true,
          selectedOptions: [{value: 'first'}, {value: 'second'}],
          getAttribute: () => null
        }),
      selectOption
    } as unknown as Locator;
    const panel = {getByTestId: () => control} as unknown as Locator;
    await setAnalysisSheetParameter(panel, 'predictors', ['first', 'second']);
    expect(selectOption).toHaveBeenCalledExactlyOnceWith(['first', 'second']);
    expect(await readAnalysisSheetField(panel, 'predictors')).toBe('first,second');
    const missing = {getByTestId: () => ({count: async () => 0})} as unknown as Locator;
    expect(await setAnalysisSheetParameter(missing, 'unavailable', 1, {optional: true})).toBe(
      false
    );
  });
  test('invalid radius text blocks execution and staging, including handlers captured before the edit', async () => {
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
    const probe = createProbe(AnalysisCatalogPanel);
    const controlProbe = createProbe(WorkbenchParameterValue);
    const properties = {datasets: [], device: new NullDevice({}), registry, onPublish: vi.fn()};
    try {
      let nodes = probe.render(properties);
      change(nodes, 'catalog-operation', binding.id);
      nodes = probe.render(properties);
      await click(nodes, 'catalog-run');
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      const previousRun = find(nodes, 'catalog-run').props['onClick'] as () => void;
      const previousPublish = find(nodes, 'catalog-publish').props['onClick'] as () => void;
      const acceptedRadius = JSON.parse(
        String(find(nodes, 'catalog-result').props['data-parameters'])
      ).radius as number;
      const parameter = nodes.find(
        element => (element.props['spec'] as {name?: string} | undefined)?.name === 'radius'
      )!;
      const sharedControl = renderAnalysisSheetParameter(
        parameter.props as Parameters<typeof AnalysisSheetParameter>[0]
      ).find(element => element.type === WorkbenchParameterValue)!;
      const controlProperties = sharedControl.props as Parameters<
        typeof WorkbenchParameterValue
      >[0];
      let controlNodes = controlProbe.render(controlProperties);
      (find(controlNodes, 'catalog-parameter-radius').props['onFocus'] as () => void)();
      change(
        controlNodes,
        String(find(controlNodes, 'catalog-parameter-radius').props['aria-label']),
        '-1'
      );
      controlNodes = controlProbe.render(controlProperties);
      expect(find(controlNodes, 'catalog-parameter-radius').props['aria-invalid']).toBe(true);
      // A validity report must guard actions synchronously, before the next host render.
      previousRun();
      previousPublish();
      await vi.runAllTimersAsync();
      expect(execute).toHaveBeenCalledOnce();
      expect(properties.onPublish).not.toHaveBeenCalled();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-run').props['disabled']).toBe(true);
      expect(find(nodes, 'catalog-preflight-status').props['children']).toBe('blocked');
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(false);
      expect(find(nodes, 'catalog-publish').props['disabled']).toBe(true);
      await click(nodes, 'catalog-run');
      await click(nodes, 'catalog-publish');
      await vi.runAllTimersAsync();
      expect(execute).toHaveBeenCalledOnce();
      expect(properties.onPublish).not.toHaveBeenCalled();
      change(
        controlNodes,
        String(find(controlNodes, 'catalog-parameter-radius').props['aria-label']),
        String(acceptedRadius)
      );
      controlNodes = controlProbe.render(controlProperties);
      (find(controlNodes, 'catalog-parameter-radius').props['onBlur'] as () => void)();
      nodes = probe.render(properties);
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-run').props['disabled']).toBe(false);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      expect(find(nodes, 'catalog-publish').props['disabled']).toBe(false);
      expect(execute).toHaveBeenCalledOnce();
      await click(nodes, 'catalog-publish');
      expect(properties.onPublish).toHaveBeenCalledOnce();
      // A real committed edit still invalidates the accepted snapshot before React renders it.
      const publishAccepted = find(nodes, 'catalog-publish').props['onClick'] as () => void;
      change(
        controlNodes,
        String(find(controlNodes, 'catalog-parameter-radius').props['aria-label']),
        String(acceptedRadius + 1)
      );
      controlNodes = controlProbe.render(controlProperties);
      (find(controlNodes, 'catalog-parameter-radius').props['onBlur'] as () => void)();
      publishAccepted();
      await vi.runAllTimersAsync();
      expect(properties.onPublish).toHaveBeenCalledOnce();
      nodes = probe.render(properties);
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(execute).toHaveBeenCalledTimes(2);
      expect(
        JSON.parse(String(find(nodes, 'catalog-result').props['data-parameters'])).radius
      ).toBe(acceptedRadius + 1);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
    } finally {
      controlProbe.unmount();
      probe.unmount();
      await vi.runAllTimersAsync();
    }
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });

  test.each([
    'completed',
    'pending replacement'
  ])('cancel preserves the accepted result after %s while input and runtime checks remain enforced', async phase => {
    vi.useFakeTimers();
    const binding = getCatalogBinding('geometry-point-buffer')!;
    let complete!: () => void;
    let started!: () => void;
    const pending = new Promise<void>(resolve => {
      complete = resolve;
    });
    const executing = new Promise<void>(resolve => {
      started = resolve;
    });
    const execute = vi.fn();
    vi.spyOn(binding, 'planPreview').mockImplementation(inputs => {
      execute.mockImplementation(async parameters => {
        const invocation = execute.mock.calls.length;
        if (invocation === 2) {
          started();
          await pending;
        }
        const output = runCatalogOperation(binding, parameters, inputs);
        return {...output, summary: {...output.summary, invocation}};
      });
      return {
        key: 'retained',
        gpuByteLength: 1,
        cpuByteLength: 1,
        create: async () => ({execute, destroy: async () => undefined})
      };
    });
    const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 100});
    let currentRegistry: KeplerDatasetRegistry | undefined = registry;
    const probe = createProbe(AnalysisCatalogPanel);
    const properties = {
      datasets: [],
      device: new NullDevice({}),
      registry,
      getCurrentRegistry: () => currentRegistry,
      onPublish: vi.fn()
    };
    try {
      let nodes = probe.render(properties);
      change(nodes, 'catalog-operation', binding.id);
      nodes = probe.render(properties);
      await click(nodes, 'catalog-run');
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      if (phase === 'pending replacement') {
        await click(nodes, 'catalog-run');
        await executing;
        nodes = probe.render(properties);
      }
      await click(nodes, 'ax-preview-cancel');
      complete();
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(true);
      expect(find(nodes, 'catalog-result-summary').props['children']).toContain('invocation: 1');
      expect(find(nodes, 'catalog-publish').props['disabled']).toBe(false);
      expect(execute).toHaveBeenCalledTimes(phase === 'completed' ? 1 : 2);
      await click(nodes, 'catalog-publish');
      expect(properties.onPublish).toHaveBeenCalledOnce();
      // Cancellation must not weaken the immediate runtime guard on completed publications.
      currentRegistry = undefined;
      await click(nodes, 'catalog-publish');
      expect(properties.onPublish).toHaveBeenCalledOnce();
      nodes = probe.render(properties);
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(false);
      expect(find(nodes, 'catalog-publish').props['disabled']).toBe(true);
    } finally {
      complete();
      probe.unmount();
      await vi.runAllTimersAsync();
    }
    expect(registry.memoryUsage.totalBytes).toBe(0);
  });
  test('integrated result cards retain accepted report evidence and guard every keep action while the sheet is invalid', async () => {
    vi.useFakeTimers();
    const probe = createProbe(AnalysisCatalogPanel);
    const properties = {datasets: [], onPublish: vi.fn()};
    const readCard = (nodes: Element[]) =>
      nodes.find(element => element.type === AnalysisResultCard)!.props as Parameters<
        typeof AnalysisResultCard
      >[0];
    try {
      let nodes = probe.render(properties);
      change(nodes, 'catalog-operation', 'geometry-point-buffer');
      nodes = probe.render(properties);
      await click(nodes, 'catalog-run');
      await vi.runAllTimersAsync();
      nodes = probe.render(properties);
      const acceptedCard = readCard(nodes);
      expect(acceptedCard.actionsDisabledReason).toBeUndefined();
      expect(acceptedCard.model.result.parameters).toEqual(
        JSON.parse(String(find(nodes, 'catalog-result').props['data-parameters']))
      );
      expect(acceptedCard.model.context.resolutions?.length).toBeGreaterThan(0);
      expect(acceptedCard.model.report.chosen).toContainEqual(
        expect.stringContaining('Population: full selected inputs;')
      );
      expect(JSON.parse(acceptedCard.model.json).report).toEqual(acceptedCard.model.report);
      await click(nodes, 'catalog-publish');
      expect(properties.onPublish).toHaveBeenCalledOnce();
      expect(properties.onPublish.mock.calls[0][4].report).toBe(acceptedCard.model.report);
      const parameter = nodes.find(
        element => (element.props['spec'] as {name?: string} | undefined)?.name === 'radius'
      )!;
      (parameter.props['onValidityChange'] as (problem: string | null) => void)('must be positive');
      // Handlers captured from an accepted card must not keep an invalid draft before rerender.
      await acceptedCard.onAddToStack!();
      await acceptedCard.onSaveWorkflow!();
      const next = acceptedCard.model.next.find(
        action => action.applicability.status !== 'blocked'
      );
      if (next) await acceptedCard.onNext(next);
      nodes = probe.render(properties);
      expect(readCard(nodes).model).toBe(acceptedCard.model);
      expect(readCard(nodes).actionsDisabledReason).toContain('current full population');
      expect(find(nodes, 'catalog-result').props['data-current']).toBe(false);
      expect(
        nodes.find(element => element.props['id'] === 'catalog-operation')?.props['value']
      ).toBe('geometry-point-buffer');
      (parameter.props['onValidityChange'] as (problem: string | null) => void)(null);
      nodes = probe.render(properties);
      await click(nodes, 'ax-preview-cancel');
      nodes = probe.render(properties);
      expect(readCard(nodes).model).toBe(acceptedCard.model);
      expect(readCard(nodes).actionsDisabledReason).toBeUndefined();
      await click(nodes, 'catalog-publish');
      expect(properties.onPublish).toHaveBeenCalledTimes(2);
      expect(properties.onPublish.mock.calls[1][4].report).toBe(acceptedCard.model.report);
    } finally {
      probe.unmount();
      await vi.runAllTimersAsync();
    }
  });

  test('control validity reports invalid drafts immediately and clears its error on unmount', () => {
    const probe = createProbe(WorkbenchParameterValue);
    const onValidityChange = vi.fn();
    const properties = {
      spec: {name: 'radius', label: 'Radius', kind: 'number' as const, minimum: 1},
      value: 8,
      onCommit: vi.fn(),
      testId: 'radius',
      commitOnRelease: true,
      onValidityChange
    };
    const nodes = probe.render(properties);
    expect(onValidityChange).toHaveBeenLastCalledWith(null);
    change(nodes, 'Radius', '-1');
    expect(onValidityChange).toHaveBeenLastCalledWith(expect.stringContaining('at least 1'));
    expect(properties.onCommit).not.toHaveBeenCalled();
    probe.unmount();
    expect(onValidityChange).toHaveBeenLastCalledWith(null);
  });
});
