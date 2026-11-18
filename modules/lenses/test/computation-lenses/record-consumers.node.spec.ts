// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {ResourceBars} from '../../../../website/src/components/meta/resource-bars';
import {extractLensViewModel} from '../../../../website/src/lens/extract-lens-view-model';
import * as formatting from '../../../../website/src/lens/format';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {
  createGPUComputationRecord,
  createGPUComputationCaptures,
  createGPUComputationMetrics,
  validateGPUComputationSidecars,
  type GPUCommandGraphTimedSubmission
} from '@luma-contributors/analysis/computation-record';
import {createDataFrameLensRecord} from '../../src/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support';
import {readLensRunMetrics} from '../../src/computation-lenses/lens-view-snapshot';
import {toRecordedLensFile, fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {createViewContext} from '../../src/computation-views/registry/view-registry';
import * as snapshotHelpers from '../../src/computation-lenses/lens-view-snapshot';
import {runS2Lens} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s2-lens-run';
import {vi} from 'vitest';
import {readFileSync} from 'node:fs';
import {runS3Lens} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-run';
import {
  getGPUComputationRunId,
  getGPUComputationAdapter
} from '@luma-contributors/analysis/computation-record';
import {finishSpatialWeightsRecord} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {getSpatialGuardMetadata} from '../../src/gpu-spatial-analysis/computation-pack/guards';

test('final1 lenses-consumers:F10 decoded legacy envelopes validate identity scalars', () => {
  const envelope = JSON.parse(
    readFileSync(
      new URL('../../../../docs/lens-records/gpu-spatial-analysis.kriging.json', import.meta.url),
      'utf8'
    )
  );
  const record = fromRecordedLensFile(envelope).record;
  const malformed = toRecordedLensFile(
    {...record, parameters: {radius: {value: 25}} as never},
    {
      scenarioId: 'gpu-spatial-analysis.kriging',
      recordedAt: '2026-11-17T00:00:00.000Z',
      adapterLabel: 'fixture'
    }
  );
  expect(() => fromRecordedLensFile(malformed)).toThrow('parameters.radius');
});

test('final1 lenses-consumers:F14 missing inventory is unavailable while an empty inventory is exact', () => {
  const model = (resources?: []) =>
    extractLensViewModel('fixture.inventory', {
      runId: 'run',
      graphs: [
        {
          id: 'compiled',
          form: 'compiled',
          nodes: [
            {id: 'compute', label: 'Compute', kind: 'compute', reads: ['source'], writes: ['out']}
          ],
          edges: [],
          ...(resources && {resources})
        }
      ]
    });
  expect(model().resources.totalBytesKind).toBe('unavailable');
  expect(model().resources.allocatedBytesKind).toBe('unavailable');
  expect(model().resources.inventoryKnown).toBe(false);
  const html = renderToStaticMarkup(createElement(ResourceBars, {model: model()}));
  expect(html).toContain('inventory unavailable');
  expect(html).not.toContain('0 buffers');
  expect(model([]).resources).toMatchObject({
    count: 0,
    inventoryKnown: true,
    totalBytes: 0,
    totalBytesKind: 'logical',
    allocatedBytesKind: 'logical'
  });
});

test('final1 lenses-consumers:F11 final S3 choices reproduce identity and observations do not alter choices', async () => {
  const device = createNullWebGPUDevice();
  const submit = vi
    .spyOn(snapshotHelpers, 'submitLensGraph')
    .mockImplementation(() => ({}) as never);
  const read = vi.spyOn(snapshotHelpers, 'readLensViewSnapshot').mockResolvedValue([]);
  const inputs = {
    scenarioId: 'fixture.s3',
    element: 'GPURoseStatistic',
    datasets: ['fixture'],
    choices: {mode: 'axial', binCount: 8},
    seed: 7
  };
  try {
    const record = await runS3Lens(device, {
      ...inputs,
      build: () => ({}),
      metadata: {coordinateModel: 'projected'}
    });
    expect(record.choices.some(choice => choice.name === 'coordinateModel')).toBe(true);
    expect(record.runId).toBe(
      getGPUComputationRunId({
        ...record,
        adapter: getGPUComputationAdapter(device),
        datasetVersion: 'fixture',
        seed: 7
      })
    );
    expect(validateGPUComputationSidecars(record, {captures: record.captures}).valid).toBe(true);
    const observed = await runS3Lens(device, {
      ...inputs,
      build: () => ({}),
      metadata: {coordinateModel: 'projected'},
      observations: () => ({islandCount: 3})
    });
    expect(observed.choices.some(choice => choice.name === 'islandCount')).toBe(false);
    expect(observed.runId).toBe(record.runId);
    expect(getSpatialGuardMetadata(observed)['islandCount']).toBe(3);
    const counts = {
      id: 'counts',
      name: 'counts',
      graph: observed.graphs[0].id,
      elementType: 'u32' as const,
      provenance: 'gpu-readback' as const,
      values: new Uint32Array([0, 0, 2])
    };
    const finished = finishSpatialWeightsRecord(
      {
        ...observed,
        captures: createGPUComputationCaptures({
          runId: observed.runId,
          executionId: observed.executionId,
          snapshots: [counts]
        })
      },
      {}
    );
    expect(finished.choices).toEqual(observed.choices);
    expect(getSpatialGuardMetadata(finished)['islandCount']).toBe(2);
    expect(finished.runId).toBe(
      getGPUComputationRunId({...finished, datasetVersion: 'fixture', seed: 7})
    );
    expect(validateGPUComputationSidecars(finished, {captures: finished.captures}).valid).toBe(
      true
    );
  } finally {
    read.mockRestore();
    submit.mockRestore();
    device.destroy();
  }
});

test('R1 kernel coverage recordings require an ISO timestamp and roundtrip with the fixed fixture date', () => {
  const record = createGPUComputationRecord({
    id: 'fixture',
    domains: [],
    adapter: {backend: 'null'},
    graphs: []
  });
  expect(() =>
    toRecordedLensFile(record, {
      scenarioId: 'fixture.record',
      recordedAt: 'measure',
      adapterLabel: 'fixture'
    })
  ).toThrow('recordedAt');
  const envelope = toRecordedLensFile(record, {
    scenarioId: 'fixture.record',
    recordedAt: '2026-11-17T00:00:00.000Z',
    adapterLabel: 'fixture'
  });
  expect(fromRecordedLensFile(envelope).record.executionId).toBe(record.executionId);
});

test('R5 shared allocation bounds are order independent and unknown ownership prevents exact totals', () => {
  const summarize = (
    resources: Parameters<typeof extractLensViewModel>[1]['graphs'][number]['resources']
  ) =>
    extractLensViewModel('fixture.memory', {
      runId: 'run',
      graphs: [{id: 'compiled', form: 'compiled', edges: [], nodes: [], resources}]
    }).resources;
  const resources = [
    {
      id: 'large',
      bytes: 64,
      bytesKind: 'logical' as const,
      ownership: 'transient' as const,
      allocation: 'shared'
    },
    {
      id: 'small',
      bytes: 16,
      bytesKind: 'logical' as const,
      ownership: 'transient' as const,
      allocation: 'shared'
    }
  ];
  for (const order of [resources, [...resources].reverse()])
    expect(summarize(order)).toMatchObject({allocatedBytes: 64, allocatedBytesKind: 'lower-bound'});
  expect(
    summarize([{id: 'unknown', bytes: 8, bytesKind: 'lower-bound', ownership: 'unknown'}])
      .allocatedBytesKind
  ).toBe('unspecified');
  expect(summarize([{...resources[0], allocationBytes: 128}, resources[1]])).toMatchObject({
    allocatedBytes: 128,
    allocatedBytesKind: 'logical'
  });
  expect(
    summarize([{id: 'transient', bytes: 8, bytesKind: 'logical', ownership: 'transient'}])
      .allocatedBytesKind
  ).toBe('unspecified');
});

test('R6 rose sampling remains an identity parameter even when omitted by display templates', async () => {
  const device = createNullWebGPUDevice();
  const submission = vi
    .spyOn(snapshotHelpers, 'submitLensGraph')
    .mockImplementation(() => ({}) as never);
  try {
    const run = (rayleighSampling: string) =>
      runS2Lens(device, {
        scenarioId: 'gpu-spatial-analysis.rose-statistic',
        element: 'GPURoseStatistic',
        datasets: ['fixture'],
        choices: {mode: 'axial', binCount: 36, rayleighSampling},
        build: () => ({})
      });
    const first = await run('one-per-segment');
    const second = await run('length-weighted');
    expect(first.parameters?.['rayleighSampling']).toBe('one-per-segment');
    expect(second.parameters?.['rayleighSampling']).toBe('length-weighted');
    expect(first.runId).not.toBe(second.runId);
  } finally {
    submission.mockRestore();
    device.destroy();
  }
});

test('R9 website sidecars always require a matching runId, including legacy executions', () => {
  for (const executionId of [undefined, 'current']) {
    const record = {
      runId: 'run',
      executionId,
      graphs: [{id: 'compiled', form: 'compiled', edges: [], nodes: []}]
    };
    expect(() =>
      extractLensViewModel('fixture.join', {
        ...record,
        captures: {executionId, snapshots: []}
      } as never)
    ).toThrow('runId');
    expect(() =>
      extractLensViewModel('fixture.join', {
        ...record,
        metrics: [{executionId, graph: 'compiled', timebase: 'ns', nodes: {}}]
      } as never)
    ).toThrow('runId');
  }
});

test('R11 a displayed lower bound never rounds above its observed byte count', () => {
  expect(formatting.formatResourceBytes(1040, 'lower-bound')).toBe('≥1040 B');
});

test('resource views preserve unknown ownership and byte precision without merging equal sizes', () => {
  const model = extractLensViewModel('fixture.resources', {
    runId: 'run',
    graphs: [
      {
        id: 'compiled',
        form: 'compiled',
        edges: [],
        nodes: [
          {
            id: 'read',
            label: 'Read',
            kind: 'compute',
            reads: ['bounded', 'known', 'legacy'],
            writes: []
          }
        ],
        resources: [
          {id: 'bounded', ownership: 'unknown', bytes: 8, bytesKind: 'lower-bound'},
          {id: 'known', ownership: 'imported', bytes: 8, bytesKind: 'logical'},
          {id: 'legacy', ownership: 'imported', bytes: 8}
        ]
      }
    ]
  });
  expect(model.resources).toMatchObject({
    totalBytes: 24,
    totalBytesKind: 'unspecified',
    importedBytes: 8,
    unknownBytes: 16
  });
  expect(model.resources.largest).toHaveLength(3);
  expect(model.lifetimes?.find(value => value.id === 'bounded')).toMatchObject({
    ownership: 'unknown',
    bytesKind: 'lower-bound'
  });
  expect(model.lifetimes?.find(value => value.id === 'legacy')?.ownership).toBe('unknown');
  expect(formatting.formatResourceBytes(8, 'lower-bound')).toBe('≥8 B');
  expect(formatting.formatResourceBytes(8, 'estimated')).toBe('~8 B');
  expect(formatting.formatResourceBytes(8)).toBe('8 B (precision unknown)');
  expect(formatting.formatResourceBytes(8, 'logical')).toBe('8 B');
  const html = renderToStaticMarkup(createElement(ResourceBars, {model}));
  expect(html).toContain('≥8 B');
  expect(html).toContain('precision unknown');
});

test('dataframe captures and timestamp-free lens metrics keep execution identity across repeated runs', async () => {
  const device = createNullWebGPUDevice();
  try {
    const props = {scenarioId: 'fixture.repeat', device, graphs: [], keySpaces: [], captures: []};
    const first = createDataFrameLensRecord(props);
    const second = createDataFrameLensRecord(props);
    expect(validateGPUComputationSidecars(first, first).issues).toEqual([]);
    expect(first.runId).toBe(second.runId);
    expect(validateGPUComputationSidecars(second, {captures: first.captures}).valid).toBe(false);
    const submission = {
      readTimings: async () => undefined,
      encoding: {stats: {coalescedComputeNodeCount: 0}}
    } as unknown as GPUCommandGraphTimedSubmission;
    const metrics = await readLensRunMetrics(submission, {
      runId: first.runId,
      executionId: first.executionId,
      graphId: 'compiled',
      taps: 1
    });
    expect(metrics.executionId).toBe(first.executionId);
  } finally {
    device.destroy();
  }
});

test('website extraction refuses captures and metrics from another execution of the same structure', () => {
  const record = createGPUComputationRecord({
    id: 'fixture',
    domains: [],
    adapter: {backend: 'null'},
    graphs: [{id: 'compiled', form: 'compiled', domain: 'test', edges: [], nodes: []}]
  });
  const captures = createGPUComputationCaptures({runId: record.runId, executionId: 'stale'});
  expect(() => extractLensViewModel('fixture.repeat', {...record, captures})).toThrow(
    'executionId'
  );
  const metrics = createGPUComputationMetrics({
    runId: record.runId,
    executionId: 'stale',
    graph: 'compiled',
    timing: {nodes: []},
    encoding: {coalesced: false, timestamped: false, taps: 0, lensTwin: true}
  });
  expect(() => extractLensViewModel('fixture.repeat', {...record, metrics: [metrics]})).toThrow(
    'executionId'
  );
});

test('lens serialization and pure view contexts refuse stale executions and preserve matching identity', () => {
  const record = createGPUComputationRecord({
    id: 'fixture',
    domains: [],
    adapter: {backend: 'null'},
    graphs: []
  });
  const captures = createGPUComputationCaptures({
    runId: record.runId,
    executionId: record.executionId
  });
  const result = {...record, captures};
  const options = {
    scenarioId: 'fixture.repeat',
    recordedAt: '2026-11-17T00:00:00Z',
    adapterLabel: 'fixture'
  };
  const envelope = toRecordedLensFile(result, options);
  expect(fromRecordedLensFile(envelope).record.executionId).toBe(record.executionId);
  const stale = {...result, captures: {...captures, executionId: 'stale'}};
  expect(() => createViewContext(stale)).toThrow('executionId');
  expect(() => toRecordedLensFile(stale, options)).toThrow('executionId');
  const changed = structuredClone(fromRecordedLensFile(envelope));
  changed.record.captures!.executionId = 'stale';
  expect(() => fromRecordedLensFile(changed)).toThrow('executionId');
});
