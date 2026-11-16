// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareAnalysisResultPublication} from '../../../../examples/deck/kepler-compute/src/analysis-result-publication';
import {describe, expect, it, vi} from 'vitest';
import type {Buffer, Device} from '@luma.gl/core';
import {createNullWebGPUDevice} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  CURATED_LANDSCAPE_METRICS_BINDING,
  compactLandscapePatches,
  prepareLandscapeMetrics,
  makeLandscapeCellsTable,
  LANDSCAPE_METRICS_LIMITS,
  LANDSCAPE_METRICS_SHADER_LIMITS
} from '../../src/curated/bindings/index';
import {
  createLandscapeMetricsSample,
  LANDSCAPE_METRICS_SAMPLE_LICENSE
} from '../../src/curated/samples/index';
import {
  createKeplerCuratedAnalysisSample,
  getKeplerCuratedAnalysisSampleMetadata,
  KEPLER_CURATED_ANALYSIS_SAMPLE_IDS
} from '../../src/curated/curated-analysis-samples';
import {CURATED_LANDSCAPE_METRICS_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-landscape-metrics';
import {
  getCatalogBinding,
  runCatalogOperationAsync,
  KEPLER_CATALOG_BINDINGS
} from '../../src/analysis/catalog-bindings';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {
  getKeplerCuratedAnalysis,
  KEPLER_CURATED_ANALYSES
} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {makeWorkflowData, makeWorkflowRasterData} from '../../src/workflows/workflow-data';
import {makeWorkflowPublication} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-host-data';
import {profileKeplerWorkbenchData} from '../../src/descriptors/workbench-adapters';
import {createCuratedWorkflowAnalysisTemplate} from '../../src/workflows/workflow-analysis-template';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import type {CatalogRasterInput} from '../../src/analysis/catalog-raster-input';

import {
  evaluateKeplerOperationApplicability,
  suggestKeplerAnalyses
} from '../../src/descriptors/applicability';
import {profileGallerySample} from '../descriptors/gallery-sample-profiles';

const binding = CURATED_LANDSCAPE_METRICS_BINDING;
function makeGrid(values: number[], width = values.length): CatalogRasterInput {
  return {
    type: 'raster',
    id: 'capacity-grid',
    crs: 'planar',
    cellSizeMode: 'uniform',
    grid: makeGPURasterMetadata({width, height: values.length / width}),
    band: {
      index: 1,
      values: Float64Array.from(values),
      noDataValue: -999,
      validity: new Uint8Array(values.length).fill(1)
    }
  };
}

describe('curated landscape preparation and input contract', () => {
  it('keeps zero, negative and large exact class codes, excluding masks/nodata only', () => {
    const input = makeGrid([0, -7, 2 ** 40, -999, 999, 0]);
    input.band.validity[4] = 0;
    const prepared = prepareLandscapeMetrics(input);
    expect(prepared.classCodes).toEqual([-7, 0, 2 ** 40]);
    expect([...prepared.classes]).toEqual([1, 0, 2, 0xffffffff, 0xffffffff, 1]);
    expect(prepared.excludedCellCount).toBe(2);
    expect(
      prepareLandscapeMetrics(input, {excludeClass: true, excludedClass: 0}).classCodes
    ).toEqual([-7, 2 ** 40]);
    expect(input.band.values[0]).toBe(0);
  });
  it('rejects continuous, nonfinite and empty observed classes with named limits', () => {
    for (const value of [0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])
      expect(() => prepareLandscapeMetrics(makeGrid([value]))).toThrow(
        /curated-landscape-metrics.*valid class values.*safe integers/
      );
    expect(() => prepareLandscapeMetrics(makeGrid([-999]))).toThrow(
      /curated-landscape-metrics.*classCount/
    );
    expect(() => prepareLandscapeMetrics(makeGrid([1]), {excludedClass: 'invalid'})).toThrow(
      /excludedClass.*integer/
    );
    expect(() =>
      prepareLandscapeMetrics(makeGrid([1]), {excludeClass: true, excludedClass: 1})
    ).toThrow(/classCount/);
  });
  it('requires complete planar rectangular area pixels and valid band lengths', () => {
    const input = createLandscapeMetricsSample('second').landcover;
    expect(prepareLandscapeMetrics(input).cellSize).toEqual([2, 3]);
    for (const crs of ['EPSG:4326', 'EPSG:3857'] as const)
      expect(() =>
        prepareLandscapeMetrics({
          ...input,
          crs,
          cellSizeMode: crs === 'EPSG:4326' ? 'geographic' : 'web-mercator'
        })
      ).toThrow(/landcover.crs.*planar/);
    expect(() =>
      prepareLandscapeMetrics({...input, grid: {...input.grid, affine: [2, 1, 100, 0, 3, 200]}})
    ).toThrow(/axis-aligned/);
    expect(() =>
      prepareLandscapeMetrics({...input, grid: {...input.grid, pixelInterpretation: 'point'}})
    ).toThrow(/area pixels/);
    expect(() =>
      prepareLandscapeMetrics({...input, band: {...input.band, validity: new Uint8Array(1)}})
    ).toThrow(/band.validity length/);
    expect(() => prepareLandscapeMetrics({type: 'FeatureCollection', features: []})).toThrow(
      /landcover.*native raster/
    );
  });
  it('withholds geographic publication of unreferenced planar grids regardless of coordinate range', () => {
    const source = createLandscapeMetricsSample('second').landcover;
    for (const origin of [
      [100, 200],
      [500000, 4500000],
      [12, 34]
    ]) {
      const prepared = prepareLandscapeMetrics({
        ...source,
        grid: {...source.grid, affine: [2, 0, origin[0], 0, 3, origin[1]]}
      });
      const compacted = compactLandscapePatches(prepared, [
        new Uint32Array([1, 1, 0, 4, 4, 0, 0, 1, 1, 0, 4, 4, 0, 0, 1, 1, 0, 4, 4, 0, 0]),
        new Uint32Array([0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 6, 6, 0, 0, 0, 0, 0, 6, 6])
      ]);
      const table = makeLandscapeCellsTable(prepared, compacted, new Uint32Array(21).fill(1));
      expect(table.geometryColumn).toBeUndefined();
      expect(prepareAnalysisResultPublication(table).analysisResult.destination).toBe('report');
      expect(Object.keys(table.columns)).not.toContain('geometry');
      expect(table.columns['cell_index']).toEqual([
        0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20
      ]);
      expect(table.columns['cell_column']).toEqual([
        0, 1, 3, 4, 5, 6, 0, 1, 3, 4, 5, 6, 0, 1, 3, 4, 5, 6
      ]);
      expect(table.columns['cell_row']).toEqual([
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2
      ]);
      const data = makeWorkflowData(table);
      expect(data.kind).toBe('table');
      const publication = makeWorkflowPublication(data, 'Planar landscape');
      expect(publication.geometryColumn).toBeUndefined();
      expect(publication.coordinates).toBeUndefined();
      expect(publication.table.schema.fields.map(field => field.name)).not.toContain('geometry');
    }
    expect(binding.outputs.every(output => output.geometry === 'none')).toBe(true);
    expect(
      CURATED_LANDSCAPE_METRICS_DESCRIPTOR.outputs.every(
        output => output.kind === 'attributes' && output.presentation === undefined
      )
    ).toBe(true);
  });
  for (const fixture of [
    {name: 'cell area overflow', width: 1, height: 1, cellSize: [1e20, 1e20], prop: 'cellArea'},
    {name: 'cell area underflow', width: 1, height: 1, cellSize: [1e-30, 1e-30], prop: 'cellArea'},
    {name: 'domain width overflow', width: 2, height: 1, cellSize: [2e38, 1], prop: 'domainWidth'},
    {name: 'domain height overflow', width: 1, height: 2, cellSize: [1, 2e38], prop: 'domainHeight'}
  ]) {
    it(`rejects derived shader ${fixture.name} before allocating GPU resources`, async () => {
      const input = makeGrid(new Array(fixture.width * fixture.height).fill(1), fixture.width);
      const landcover = {
        ...input,
        grid: makeGPURasterMetadata({
          width: fixture.width,
          height: fixture.height,
          cellSize: fixture.cellSize as [number, number]
        })
      };
      expect(
        fixture.cellSize.every(
          value => Number.isFinite(Math.fround(value)) && Math.fround(value) > 0
        )
      ).toBe(true);
      const createBuffer = vi.fn();
      const device = {createBuffer} as unknown as Device;
      await expect(binding.runDataGpu!({device}, {landcover}, {})).rejects.toThrow(
        new RegExp(
          `curated-landscape-metrics landcover\\.grid\\.${fixture.prop} must be a finite number in`
        )
      );
      expect(createBuffer).not.toHaveBeenCalled();
    });
  }
  it('accepts representable shader constants and rejects positive values that round or flush to zero', () => {
    const input = makeGrid([1]);
    expect(() =>
      prepareLandscapeMetrics({
        ...input,
        grid: makeGPURasterMetadata({width: 1, height: 1, cellSize: [1e-50, 1]})
      })
    ).toThrow(/curated-landscape-metrics.*cellSize/);
    const supported = {
      ...input,
      grid: makeGPURasterMetadata({width: 1, height: 1, cellSize: [1e-10, 1e-10]})
    };
    expect(prepareLandscapeMetrics(supported, {edgeDepth: 0, searchRadius: 0}).cellSize).toEqual([
      1e-10, 1e-10
    ]);
    for (const parameter of ['edgeDepth', 'searchRadius'])
      expect(() => prepareLandscapeMetrics(input, {[parameter]: 1e-50})).toThrow(
        new RegExp(`curated-landscape-metrics ${parameter} must be a finite number in`)
      );
    expect(LANDSCAPE_METRICS_SHADER_LIMITS.minimumPositive).toBe(Math.fround(2 ** -126));
    expect(LANDSCAPE_METRICS_SHADER_LIMITS.maximum).toBe(Math.fround((2 - 2 ** -23) * 2 ** 127));
  });
  it('rejects cell/class/patch and scratch overflow before metrics allocation', () => {
    expect(() => prepareLandscapeMetrics(makeGrid(new Array(4097).fill(1)))).toThrow(
      /grid cell count/
    );
    expect(() =>
      prepareLandscapeMetrics(makeGrid(Array.from({length: 33}, (_, index) => index)))
    ).toThrow(/classCount/);
    const prepared = prepareLandscapeMetrics(makeGrid(new Array(257).fill(1)));
    expect(() =>
      compactLandscapePatches(prepared, [Uint32Array.from({length: 257}, (_, index) => index + 1)])
    ).toThrow(/patchCount/);
    expect(LANDSCAPE_METRICS_LIMITS.patches * LANDSCAPE_METRICS_LIMITS.cells).toBe(
      LANDSCAPE_METRICS_LIMITS.patchCells
    );
    expect(() => compactLandscapePatches(prepared, [new Uint32Array(257)])).toThrow(
      /incomplete segmentation/
    );
    expect(() => compactLandscapePatches(prepared, [])).toThrow(/one complete label grid/);
  });
  it('compacts minimum-cell GPU roots across classes into stable source row identities', () => {
    const prepared = prepareLandscapeMetrics(createLandscapeMetricsSample().landcover);
    const compacted = compactLandscapePatches(prepared, [
      new Uint32Array([1, 1, 0, 0, 1, 1, 0, 0]),
      new Uint32Array([0, 0, 3, 3, 0, 0, 3, 3])
    ]);
    expect([...compacted.patches]).toEqual([0, 0, 1, 1, 0, 0, 1, 1]);
    expect([...compacted.patchClasses]).toEqual([0, 1]);
    expect(compacted.rootCells).toEqual([0, 2]);
  });
  it('validates settings independently of callers and never claims a CPU fallback', async () => {
    const input = createLandscapeMetricsSample().landcover;
    const invalidSettings: Record<string, CatalogParameterValue>[] = [
      {edgeDepth: -1},
      {searchRadius: Infinity},
      {connectivity: '6'},
      {distanceModel: 'centroid'}
    ];
    for (const settings of invalidSettings)
      expect(() => prepareLandscapeMetrics(input, settings)).toThrow(/curated-landscape-metrics/);
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    expect(binding.runData).toBeUndefined();
    await expect(runCatalogOperationAsync(binding, {}, {landcover: input})).rejects.toThrow(
      /GPU|device/
    );
  });
});

describe('curated landscape mapping and reusable template', () => {
  it('preserves point-time suggestion rankings when the native landscape registration is added', () => {
    const profiles = [profileGallerySample('points-time')];
    const baseline = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS.filter(candidate => candidate.id !== binding.id),
      profiles,
      {
        maximumCount: 6,
        curated: KEPLER_CURATED_ANALYSES.filter(analysis => analysis.id !== 'landscape-metrics')
      }
    );
    const integrated = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, profiles, {maximumCount: 6});
    const rank = (result: typeof baseline) =>
      result.suggestions.map(suggestion => ({
        id: suggestion.curatedAnalysisId ?? suggestion.bindingId,
        status: suggestion.status,
        score: suggestion.score,
        fields: suggestion.applicability.fieldBindings
      }));
    expect(rank(integrated)).toEqual(rank(baseline));
    expect(rank(integrated).map(suggestion => suggestion.id)).not.toContain('landscape-metrics');
  });

  it('registers the stream additively, with typed sample metadata and report-only output policies', () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(getCatalogBinding('curated-dasymetric-redistribution')).toBeDefined();
    expect(getKeplerOperationDescriptor(binding.id)).toBe(CURATED_LANDSCAPE_METRICS_DESCRIPTOR);
    expect(
      validateKeplerOperationDescriptor(CURATED_LANDSCAPE_METRICS_DESCRIPTOR, binding)
    ).toEqual([]);
    expect(binding.outputs.map(output => output.tableId)).toEqual([
      'cells',
      'patches',
      'classes',
      'adjacency',
      'distances',
      'landscape'
    ]);
    expect(binding.outputs.every(output => output.identityColumns.length > 0)).toBe(true);
    expect(
      CURATED_LANDSCAPE_METRICS_DESCRIPTOR.parameters.every(
        parameter => parameter.apply === 'rebuild'
      )
    ).toBe(true);
    expect(KEPLER_CURATED_ANALYSIS_SAMPLE_IDS).toContain('dasymetric-fractional-support');
    expect(createKeplerCuratedAnalysisSample('landscape-metrics')).toEqual(
      createLandscapeMetricsSample()
    );
    expect(getKeplerCuratedAnalysisSampleMetadata('landscape-metrics')).toEqual(
      LANDSCAPE_METRICS_SAMPLE_LICENSE
    );
    const analysis = getKeplerCuratedAnalysis('landscape-metrics')!;
    const template = createCuratedWorkflowAnalysisTemplate(analysis);
    expect(template.slots[0].kind).toBe('surface');
    expect(template.exposed).toHaveLength(4);
    expect(template.sample).toEqual(LANDSCAPE_METRICS_SAMPLE_LICENSE);
    expect(template.reads.cautions.join(' ')).toMatch(/4096 cells/);
  });
  for (const variant of ['sample', 'second'] as const) {
    it(`maps the native ${variant} grid without fake feature columns`, () => {
      const analysis = getKeplerCuratedAnalysis('landscape-metrics')!;
      const sample = createLandscapeMetricsSample(variant);
      const profile = profileKeplerWorkbenchData(makeWorkflowRasterData(sample.landcover, 4096));
      let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'landcover', datasetId: profile.datasetId},
        analysis,
        [profile]
      );
      expect(mapping.complete).toBe(true);
      expect(mapping.slots[0].roles).toEqual([]);
      const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
      expect(request.bindingId).toBe(binding.id);
      expect(request.inputDatasetIds).toEqual({landcover: profile.datasetId});
      expect(prepareLandscapeMetrics(sample.landcover, request.parameters).classCodes).toEqual(
        variant === 'sample' ? [10, 90] : [-7, 90]
      );
    });
  }
});

describe('landscape profile and readback regressions', () => {
  it('blocks geographic rasters and requires confirmation of guessed or missing coordinates', () => {
    const planar = profileKeplerWorkbenchData(
      makeWorkflowRasterData(createLandscapeMetricsSample().landcover, 4096)
    );
    const assess = (coordinates: typeof planar.coordinates) =>
      evaluateKeplerOperationApplicability(binding, CURATED_LANDSCAPE_METRICS_DESCRIPTOR, [
        {...planar, coordinates}
      ]);
    expect(assess({model: 'planar', source: 'declared', reason: 'CRS'}).status).toBe('ready');
    expect(assess({model: 'geographic', source: 'declared', reason: 'EPSG:4326'}).status).toBe(
      'blocked'
    );
    const guessed = assess({
      model: 'planar',
      source: 'guess',
      confidence: 'high',
      reason: 'Extent'
    });
    expect(guessed.status).toBe('needs-confirmation');
    expect(guessed.warnings.join(' ')).toMatch(/guess/);
    expect(guessed.requirements).toContainEqual(
      expect.objectContaining({id: 'declared-coordinates', state: 'unknown', evidence: 'guess'})
    );
    expect(
      assess({model: 'geographic', source: 'declared', reason: 'EPSG:4326'}).requirements
    ).toContainEqual(expect.objectContaining({id: 'planar-coordinates', state: 'failed'}));
    expect(assess(undefined).status).toBe('needs-confirmation');
    const geographic = {
      ...makeGrid([1, 1]),
      crs: 'EPSG:4326' as const,
      cellSizeMode: 'geographic' as const
    };
    const geographicProfile = profileKeplerWorkbenchData(makeWorkflowRasterData(geographic, 4096));
    expect(
      evaluateKeplerOperationApplicability(binding, CURATED_LANDSCAPE_METRICS_DESCRIPTOR, [
        geographicProfile
      ]).status
    ).toBe('blocked');
    expect(() => prepareLandscapeMetrics(geographic)).toThrow(/planar|linear/);
  });

  it.each([
    'convergence',
    'labels',
    'metrics'
  ] as const)('drains every started %s readback before rejecting and disposing', async target => {
    const device = createNullWebGPUDevice();
    const failure = new Error(`${target} failed`);
    let releaseSibling!: () => void;
    const sibling = new Promise<void>(resolve => {
      releaseSibling = resolve;
    });
    let markStarted!: () => void;
    const started = new Promise<void>(resolve => {
      markStarted = resolve;
    });
    const pendingBuffers: Buffer[] = [];
    let targetCount = 0;
    const readTarget = <Value>(value: Value, buffer: Buffer): Promise<Value> => {
      targetCount++;
      if (targetCount <= 2) pendingBuffers.push(buffer);
      if (targetCount === 1) return Promise.reject(failure);
      if (targetCount === 2) {
        markStarted();
        return sibling.then(() => value);
      }
      return Promise.resolve(value);
    };
    const run = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
    const dispose = vi.spyOn(CatalogGpuSession.prototype, 'dispose');
    const scalar = vi
      .spyOn(CatalogGpuSession.prototype, 'readScalar')
      .mockImplementation(output =>
        target === 'convergence' ? readTarget(1, output.buffer) : Promise.resolve(1)
      );
    const integers = vi
      .spyOn(CatalogGpuSession.prototype, 'readUint32')
      .mockImplementation(output => {
        const labels = output.buffer.id.includes('-labels-labels-');
        const values = labels
          ? new Uint32Array(
              output.buffer.id.endsWith('labels-0')
                ? [1, 1, 0, 0, 1, 1, 0, 0]
                : [0, 0, 3, 3, 0, 0, 3, 3]
            )
          : new Uint32Array(output.length);
        return (target === 'labels' && labels) || (target === 'metrics' && !labels)
          ? readTarget(values, output.buffer)
          : Promise.resolve(values);
      });
    const floats = vi
      .spyOn(CatalogGpuSession.prototype, 'readFloat32')
      .mockImplementation(output =>
        target === 'metrics'
          ? readTarget(new Float32Array(output.length * output.components), output.buffer)
          : Promise.resolve(new Float32Array(output.length * output.components))
      );
    let settled = false;
    const completion = binding.runDataGpu!(
      {device},
      {landcover: makeGrid([10, 10, 90, 90, 10, 10, 90, 90], 4)},
      {}
    ).then(
      () => {
        settled = true;
        return undefined;
      },
      error => {
        settled = true;
        return error;
      }
    );
    try {
      await started;
      await new Promise(resolve => setTimeout(resolve, 0));
      const settledEarly = settled;
      const destroyedEarly = pendingBuffers.some(buffer => buffer.destroyed);
      const disposalsEarly = dispose.mock.calls.length;
      releaseSibling();
      expect(await completion).toBe(failure);
      expect(settledEarly).toBe(false);
      expect(destroyedEarly).toBe(false);
      expect(disposalsEarly).toBe(target === 'metrics' ? 1 : 0);
      expect(targetCount).toBe(target === 'metrics' ? 10 : 2);
      expect(pendingBuffers.every(buffer => buffer.destroyed)).toBe(true);
    } finally {
      releaseSibling();
      await completion;
      run.mockRestore();
      dispose.mockRestore();
      scalar.mockRestore();
      integers.mockRestore();
      floats.mockRestore();
      device.destroy();
    }
  });
});
