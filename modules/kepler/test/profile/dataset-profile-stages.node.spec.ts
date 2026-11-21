// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {describe, expect, it, vi} from 'vitest';
import {computeGPUAnalysisSpacingStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {GPU_COLUMN_PROFILE_STATISTIC_COUNT} from '@luma-contributors/analysis/dataframe-analysis';
import {createNullWebGPUDevice} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {createKeplerDatasetProfileCache} from '../../src/profile/dataset-profile-cache';
import {createKeplerDatasetProfileStages} from '../../src/profile/dataset-profile-stages';
import {createKeplerDatasetProfileGpuRunner} from '../../src/profile/dataset-profile-gpu';
import {
  applyKeplerDatasetProfileDeclarations,
  getKeplerDatasetProfileUndeclared,
  profileKeplerColumnTable,
  profileKeplerFeatureCollection,
  toKeplerRecommendationSubject,
  withKeplerDatasetProfileStatistics
} from '../../src/profile/dataset-profile';
import type {
  KeplerDatasetProfile,
  KeplerDatasetProfileProgress
} from '../../src/profile/dataset-profile-types';

function metadata(): KeplerDatasetProfile {
  return {
    datasetId: 'resident',
    label: 'Resident',
    revision: 1,
    kind: 'points',
    rowCount: 2,
    vertexCount: 2,
    catalogGeometry: 'point',
    geometryColumns: [],
    geometryPackedFeatureCounts: {point: 2, line: 0, polygon: 0},
    coordinates: {model: 'planar', source: 'declared', reason: 'Source CRS'},
    columns: [{name: 'value', type: 'number', rowCount: 2, nullCount: 0, sampleSize: 0}]
  };
}
function deferred<Value>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}
function spacing(values = [0, 0, 10, 0], model: 'planar' | 'geographic' = 'planar') {
  return {
    ...computeGPUAnalysisSpacingStatistics(Float64Array.from(values), {coordinateModel: model})!,
    locationCoverage: 'all-locations' as const
  };
}
function harness() {
  const cache = createKeplerDatasetProfileCache({maximumEntries: 4});
  const progress: KeplerDatasetProfileProgress[] = [];
  const stages = createKeplerDatasetProfileStages({
    cache,
    onProgress: value => progress.push(value)
  });
  return {cache, progress, stages};
}

describe('resident staged profiles', () => {
  it('publishes metadata in the caller frame, then columns, spacing and queen contiguity', async () => {
    const {cache, stages, progress} = harness();
    const source = metadata();
    const order: string[] = [];
    const job = stages.start({
      metadata: source,
      getRevision: () => 1,
      runner: {
        columns: async ({profile}) => {
          order.push('columns');
          return profile.columns.map(column => ({...column, minimum: 5}));
        },
        spacing: async ({profile}) => {
          expect(profile.columns[0].minimum).toBe(5);
          order.push('spacing');
          return spacing();
        },
        contiguity: async ({profile}) => {
          expect(profile.spacing?.pointCount).toBe(2);
          order.push('contiguity');
          return {minimum: 0, median: 1, maximum: 2, islandCount: 1};
        }
      }
    });
    expect(cache.get('resident', 1)).toBe(source);
    expect(progress.map(event => event.stage)).toEqual(['metadata']);
    expect(order).toEqual([]);
    const result = await job.finished;
    expect(order).toEqual(['columns', 'spacing', 'contiguity']);
    expect(result?.contiguity?.islandCount).toBe(1);
    expect(result?.geometryPackedFeatureCounts).toEqual(source.geometryPackedFeatureCounts);
    expect(progress.at(-1)?.stage).toBe('complete');
    expect(Object.keys(progress.at(-1)!.timings)).toEqual(order);
  });

  it.each([
    'revision',
    'invalidate',
    'replace',
    'reinsert'
  ] as const)('rejects stale %s results', async change => {
    const {cache, stages, progress} = harness();
    const pending = deferred<KeplerDatasetProfile['columns']>();
    let revision = 1;
    const source = metadata();
    const later = vi.fn();
    const job = stages.start({
      metadata: source,
      getRevision: () => revision,
      runner: {columns: () => pending.promise, spacing: later}
    });
    await Promise.resolve();
    if (change === 'revision') revision++;
    if (change === 'invalidate' || change === 'reinsert') cache.invalidate(source.datasetId);
    if (change === 'replace' || change === 'reinsert') cache.set(source);
    pending.resolve(source.columns);
    expect(await job.finished).toBeUndefined();
    expect(later).not.toHaveBeenCalled();
    expect(progress.at(-1)?.stage).toBe('stale');
  });

  it.each([
    'job',
    'signal',
    'service'
  ] as const)('drains a pending provider on %s cancellation', async mode => {
    const {stages, progress} = harness();
    const pending = deferred<KeplerDatasetProfile['columns']>();
    const controller = new AbortController();
    const later = vi.fn();
    const job = stages.start({
      metadata: metadata(),
      getRevision: () => 1,
      signal: controller.signal,
      runner: {columns: () => pending.promise, spacing: later}
    });
    let settled = false;
    void job.finished.then(() => {
      settled = true;
    });
    await Promise.resolve();
    if (mode === 'job') job.cancel();
    if (mode === 'signal') controller.abort();
    if (mode === 'service') stages.cancel('resident');
    await Promise.resolve();
    expect(settled).toBe(false);
    pending.resolve(metadata().columns);
    expect(await job.finished).toBeUndefined();
    expect(later).not.toHaveBeenCalled();
    expect(progress.at(-1)?.stage).toBe('cancelled');
  });

  it('does not let a superseded job overwrite a replacement or its progress', async () => {
    const {cache, stages, progress} = harness();
    const pending = deferred<KeplerDatasetProfile['columns']>();
    const old = stages.start({
      metadata: metadata(),
      getRevision: () => 1,
      runner: {columns: () => pending.promise}
    });
    await Promise.resolve();
    const newer = stages.start({
      metadata: {...metadata(), label: 'Replacement'},
      getRevision: () => 1,
      runner: {}
    });
    await newer.finished;
    const events = progress.length;
    pending.resolve(metadata().columns);
    expect(await old.finished).toBeUndefined();
    expect(cache.get('resident', 1)?.label).toBe('Replacement');
    expect(progress.length).toBe(events);
  });

  it('retains options.declarations across every stage without cache-stored declarations and keeps a removable baseline', async () => {
    const {cache, stages, progress} = harness();
    const source = applyKeplerDatasetProfileDeclarations(metadata(), {
      measurement: {value: 'rate'},
      rowSemantics: 'objects'
    });
    const result = await stages.start({
      metadata: source,
      getRevision: () => 1,
      runner: {
        columns: async ({profile}) => profile.columns.map(column => ({...column, minimum: 5})),
        spacing: async () => spacing(),
        contiguity: async () => ({minimum: 0, median: 0, maximum: 0, islandCount: 2})
      }
    }).finished;
    for (const event of progress) {
      expect(event.profile?.columns[0].measurement).toBe('rate');
      expect(event.profile?.rowSemantics?.value).toBe('objects');
    }
    expect(getKeplerDatasetProfileUndeclared(result!).columns[0].measurement).toBeUndefined();
    cache.setDeclarations('resident', {measurement: {value: 'amount'}});
    expect(cache.get('resident', 1)?.columns[0].measurement).toBe('amount');
    cache.setDeclarations('resident', undefined);
    expect(cache.get('resident', 1)?.columns[0].measurement).toBeUndefined();
    expect(cache.get('resident', 1)?.columns[0].minimum).toBe(5);
  });

  it('does not resurrect a declaration copied by a provider before removal', async () => {
    const {cache, stages} = harness();
    const pending = deferred<void>();
    cache.setDeclarations('resident', {measurement: {value: 'rate'}, rowSemantics: 'objects'});
    const job = stages.start({
      metadata: metadata(),
      getRevision: () => 1,
      runner: {
        columns: async ({profile}) => {
          await pending.promise;
          return profile.columns.map(column => ({...column, minimum: 2}));
        }
      }
    });
    await Promise.resolve();
    cache.setDeclarations('resident', undefined);
    pending.resolve();
    expect((await job.finished)?.columns[0].measurement).toBeUndefined();
  });

  it.each([
    'cells',
    'surface',
    'trajectories'
  ] as const)('preserves native %s metadata', async kind => {
    const {stages} = harness();
    const source = {
      ...metadata(),
      kind,
      scalarGrid: {width: 2, height: 1, valueColumn: 'value'},
      trajectory: {parameters: {timeUnit: 'seconds'}, fixCount: 2, trackCount: 1}
    };
    const result = await stages.start({
      metadata: source,
      getRevision: () => 1,
      runner: {columns: async () => source.columns}
    }).finished;
    expect(result).toMatchObject({
      kind,
      scalarGrid: source.scalarGrid,
      trajectory: source.trajectory
    });
  });

  it('retains completed facts and reports a provider failure', async () => {
    const {cache, stages, progress} = harness();
    const failure = new Error('readback failed');
    const job = stages.start({
      metadata: metadata(),
      getRevision: () => 1,
      runner: {
        columns: async () => metadata().columns.map(column => ({...column, minimum: 3})),
        spacing: async () => {
          throw failure;
        }
      }
    });
    await expect(job.finished).rejects.toBe(failure);
    expect(cache.get('resident', 1)?.columns[0].minimum).toBe(3);
    expect(progress.at(-1)).toMatchObject({stage: 'failed', error: failure});
  });

  it.each([
    'noded',
    'not-noded'
  ] as const)('preserves source generation and lazy network cache during %s lookups', async crossings => {
    const {cache, stages} = harness();
    const pending = deferred<KeplerDatasetProfile['columns']>();
    const source = {
      ...metadata(),
      kind: 'lines' as const,
      catalogGeometry: 'line' as const,
      rowCount: 1
    };
    const job = stages.start({
      metadata: source,
      getRevision: () => 1,
      runner: {columns: () => pending.promise}
    });
    await Promise.resolve();
    const generation = cache.getGeneration(source.datasetId);
    const parameters = {
      coordinateSystem: 'planar',
      snapTolerance: 0,
      splitAtSharedVertices: true,
      costMode: 'length',
      forwardCostProperty: '',
      reverseCostProperty: '',
      onewayProperty: '',
      crossings
    };
    const load = vi.fn(() => ({
      type: 'FeatureCollection' as const,
      features: [
        {
          geometry: {
            type: 'LineString' as const,
            coordinates: [
              [0, 0],
              [1, 0]
            ]
          },
          properties: {}
        }
      ]
    }));
    const network = cache.profileNetwork(source.datasetId, 1, parameters, load, 'gpu');
    expect(cache.getGeneration(source.datasetId)).toBe(generation);
    if (crossings === 'noded') {
      expect(network).toBe(source);
      expect(load).not.toHaveBeenCalled();
    } else expect(network?.network?.nodeCount).toBe(2);
    pending.resolve(source.columns.map(column => ({...column, minimum: 5})));
    const result = await job.finished;
    expect(result?.columns[0].minimum).toBe(5);
    expect(result?.network).toBe(network?.network);
    expect(cache.getGeneration(source.datasetId)).toBe(generation);
    cache.profileNetwork(source.datasetId, 1, parameters, load, 'gpu');
    expect(load).toHaveBeenCalledTimes(crossings === 'noded' ? 0 : 1);
  });
});

describe('resident geometry evidence', () => {
  it('repairs point extents per model and replays geographic and planar corrections', () => {
    const original = {
      ...metadata(),
      coordinates: {model: 'geographic' as const, source: 'guess' as const, reason: 'Extent'},
      extent: [179, 0, -179, 0] as const
    };
    const declared = applyKeplerDatasetProfileDeclarations(original, {coordinateModel: 'planar'});
    const refined = withKeplerDatasetProfileStatistics(declared, {
      spacing: spacing([-179, 0, 179, 0])
    });
    expect(refined.extent).toEqual([-179, 0, 179, 0]);
    expect((refined as {notes?: string[]}).notes?.join(' ') ?? '').not.toMatch(/not recomputed/);
    expect(
      applyKeplerDatasetProfileDeclarations(refined, {coordinateModel: 'geographic'}).extent
    ).toEqual([179, 0, -179, 0]);
    expect(
      applyKeplerDatasetProfileDeclarations(refined, {coordinateModel: 'planar'}).extent
    ).toEqual([-179, 0, 179, 0]);
  });

  it.each([
    'feature-anchors',
    'all-locations',
    undefined
  ] as const)('preserves MultiPoint bounds and spacing when one anchor claims %s', locationCoverage => {
    const source = profileKeplerFeatureCollection(
      {
        type: 'FeatureCollection',
        features: [
          {
            properties: {},
            geometry: {
              type: 'MultiPoint',
              coordinates: [
                [0, 0],
                [10, 0]
              ]
            }
          }
        ]
      },
      {datasetId: 'multi', label: 'MultiPoint', coordinateSystem: 'planar'}
    );
    expect(source.rowCount).toBe(1);
    expect(source.vertexCount).toBe(2);
    const result = withKeplerDatasetProfileStatistics(source, {
      spacing: {...spacing([0, 0]), locationCoverage}
    });
    expect(result.extent).toEqual([0, 0, 10, 0]);
    expect(result.spacing).toBe(source.spacing);
    const residentOnly = {...source, spacing: undefined};
    expect(
      withKeplerDatasetProfileStatistics(residentOnly, {
        spacing: {...spacing([0, 0]), locationCoverage}
      }).spacing
    ).toBeUndefined();
  });

  it.each([
    'polygon',
    'point'
  ] as const)('keeps uncertainty for %s extents without full locations', catalogGeometry => {
    const source = {
      ...metadata(),
      catalogGeometry,
      extent: [179, 0, -179, 0] as const,
      coordinates: {model: 'geographic' as const, source: 'guess' as const, reason: 'Extent'}
    };
    const declared = applyKeplerDatasetProfileDeclarations(source, {coordinateModel: 'planar'});
    const result = withKeplerDatasetProfileStatistics(declared, {spacing: spacing([-179, 0])});
    expect(result.extent).toEqual([179, 0, -179, 0]);
    expect((result as {notes?: string[]}).notes?.join(' ')).toMatch(/antimeridian.*not recomputed/);
  });
});

it('keeps nonstatistical storage metadata when decoding GPU columns', async () => {
  const device = createNullWebGPUDevice();
  const source = profileKeplerColumnTable(
    {
      rowCount: 2,
      columns: [
        {name: 'large', type: 'integer', values: [4294967296, 4294967808]},
        {name: 'time', type: 'time', values: [1700000000000, 1700000001000]}
      ]
    },
    {datasetId: 'storage', label: 'Storage'}
  );
  const buffers = source.columns.map(() =>
    device.createBuffer({data: new Float32Array([1, 2]), usage: Buffer.STORAGE})
  );
  const values = new Float32Array(source.columns.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT).fill(
    NaN
  );
  for (let index = 0; index < 2; index++) {
    values[index * 11 + 2] = index ? NaN : 4294967296;
    values[index * 11 + 3] = index ? NaN : 4294967808;
    values[index * 11 + 9] = 2;
    values[index * 11 + 10] = 0;
  }
  const run = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  const floatRead = vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockResolvedValue(values);
  const integerRead = vi
    .spyOn(CatalogGpuSession.prototype, 'readUint32')
    .mockImplementation(async output =>
      output.length === 4 ? new Uint32Array([2, 0, 2, 0]) : new Uint32Array(output.length)
    );
  try {
    const runner = createKeplerDatasetProfileGpuRunner({
      device,
      rowCount: 2,
      columns: buffers.map((buffer, index) => ({
        buffer,
        length: 2,
        name: source.columns[index].name
      }))
    });
    const columns = await runner.columns!({profile: source, signal: new AbortController().signal});
    const refined = withKeplerDatasetProfileStatistics(source, {columns});
    expect(toKeplerRecommendationSubject(refined).columns).toEqual(
      toKeplerRecommendationSubject(source).columns
    );
    expect(toKeplerRecommendationSubject(refined).columns.map(column => column.type)).toEqual([
      'float64',
      'float64'
    ]);
    expect(refined.columns[1].minimum).toBeUndefined();
    expect(refined.columns[1].mean).toBeUndefined();
    expect(buffers.every(buffer => !buffer.destroyed)).toBe(true);
  } finally {
    run.mockRestore();
    floatRead.mockRestore();
    integerRead.mockRestore();
    buffers.forEach(buffer => buffer.destroy());
    device.destroy();
  }
});

it('accepts all MultiPoint locations independently of table row count', async () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({
    data: new Float32Array([0, 0, 10, 0]),
    usage: Buffer.STORAGE
  });
  const words = new Uint32Array(544);
  words[0] = 2;
  words[3] = 6;
  words[5] = 3;
  words[7] = 1;
  words[25] = 2;
  const values = new Float32Array(words.buffer);
  values.set([0, 0, 10, 0], 8);
  values.set([5, 0, 5, 5], 12);
  values.fill(10, 16, 23);
  values[24] = 10;
  values[28] = 5;
  const run = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  const read = vi.spyOn(CatalogGpuSession.prototype, 'readUint32').mockResolvedValue(words);
  try {
    const source = {...metadata(), rowCount: 1, extent: [0, 0, 10, 0] as const};
    const runner = createKeplerDatasetProfileGpuRunner({
      device,
      rowCount: 1,
      columns: [],
      positions: {buffer, length: 2},
      positionsCoverage: 'all-locations'
    });
    const result = withKeplerDatasetProfileStatistics(source, {
      spacing: await runner.spacing!({profile: source, signal: new AbortController().signal})
    });
    expect(result.rowCount).toBe(1);
    expect(result.spacing?.pointCount).toBe(2);
    expect(result.spacing?.nearestNeighbor?.maximum).toBe(10);
    expect(result.extent).toEqual([0, 0, 10, 0]);
    expect(read).toHaveBeenCalledTimes(1);
    expect(buffer.destroyed).toBe(false);
  } finally {
    run.mockRestore();
    read.mockRestore();
    buffer.destroy();
    device.destroy();
  }
});

it('retains declarations supplied directly to the metadata profiler through refinements and removal', async () => {
  const source = profileKeplerColumnTable(
    {rowCount: 2, columns: [{name: 'value', type: 'number', values: [0.1, 0.8]}]},
    {
      datasetId: 'resident',
      label: 'Metadata declarations',
      revision: 1,
      declarations: {measurement: {value: 'rate'}, rowSemantics: 'objects'}
    }
  );
  const baseline = getKeplerDatasetProfileUndeclared(source);
  const {cache, stages, progress} = harness();
  await stages.start({
    metadata: source,
    getRevision: () => 1,
    runner: {
      columns: async ({profile}) => profile.columns.map(column => ({...column, mean: 0.45})),
      contiguity: async () => ({minimum: 0, median: 0, maximum: 0, islandCount: 2})
    }
  }).finished;
  expect(
    progress.every(
      event =>
        event.profile?.columns[0].measurement === 'rate' &&
        event.profile.rowSemantics?.value === 'objects'
    )
  ).toBe(true);
  cache.setDeclarations('resident', {measurement: {value: 'amount'}});
  const removed = cache.setDeclarations('resident', undefined);
  expect(removed?.columns[0].measurement).toBe(baseline.columns[0].measurement);
  expect(removed?.columns[0].measurementSource).not.toBe('declared');
  expect(removed?.columns[0].mean).toBe(0.45);
});

it.each([
  'preaborted',
  'stale'
] as const)('does not publish %s initial metadata or invoke providers', async condition => {
  const {cache, stages, progress} = harness();
  const controller = new AbortController();
  if (condition === 'preaborted') controller.abort();
  const columns = vi.fn();
  const job = stages.start({
    metadata: metadata(),
    getRevision: () => (condition === 'stale' ? 2 : 1),
    signal: controller.signal,
    runner: {columns}
  });
  expect(await job.finished).toBeUndefined();
  expect(cache.size).toBe(0);
  expect(columns).not.toHaveBeenCalled();
  expect(progress.at(-1)?.stage).toBe(condition === 'stale' ? 'stale' : 'cancelled');
});

it('drains every column mapping before disposing a failed GPU stage', async () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({data: new Float32Array([1, 2]), usage: Buffer.STORAGE});
  const pending = deferred<Uint32Array>();
  const mappingStarted = deferred<void>();
  const failure = new Error('statistics mapping failed');
  const run = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
  const dispose = vi.spyOn(CatalogGpuSession.prototype, 'dispose');
  const floatRead = vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockRejectedValue(failure);
  const integerRead = vi
    .spyOn(CatalogGpuSession.prototype, 'readUint32')
    .mockImplementation(output => {
      if (output.length === 2) {
        mappingStarted.resolve();
        return pending.promise;
      }
      return Promise.resolve(new Uint32Array(output.length));
    });
  let completion: Promise<unknown> | undefined;
  try {
    const runner = createKeplerDatasetProfileGpuRunner({
      device,
      rowCount: 2,
      columns: [{name: 'value', buffer, length: 2}]
    });
    let settled = false;
    completion = runner.columns!({profile: metadata(), signal: new AbortController().signal}).then(
      () => {
        settled = true;
      },
      error => {
        settled = true;
        return error;
      }
    );
    await mappingStarted.promise;
    await new Promise(resolve => setTimeout(resolve, 0));
    expect(settled).toBe(false);
    expect(dispose).not.toHaveBeenCalled();
    expect(integerRead).toHaveBeenCalledTimes(2);
    pending.resolve(new Uint32Array([2, 0]));
    expect(await completion).toBe(failure);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(buffer.destroyed).toBe(false);
  } finally {
    pending.resolve(new Uint32Array([2, 0]));
    await completion;
    run.mockRestore();
    dispose.mockRestore();
    floatRead.mockRestore();
    integerRead.mockRestore();
    buffer.destroy();
    device.destroy();
  }
});
