// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterConnectedComponents} from '@luma.gl/graph/gpu-raster';
import {expect, it, vi} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogOutputTable} from '../../src/analysis/catalog-geojson';
import {CURATED_LANDSCAPE_METRICS_BINDING} from '../../src/curated/bindings/index';
import {createLandscapeMetricsSample} from '../../src/curated/samples/index';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {makeWorkflowData, makeWorkflowRasterData} from '../../src/workflows/workflow-data';
import {profileKeplerWorkbenchData} from '../../src/descriptors/workbench-adapters';
import {CURATED_LANDSCAPE_METRICS_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-landscape-metrics';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import reference from '../../../analysis/test/oracles/fixtures/urban-ecology.json';

/** The external headless driver does not print console output: include full values in errors. */
function checkColumn(
  table: CatalogOutputTable,
  column: string,
  expected: readonly (number | null)[]
) {
  const actual = table.columns[column];
  if (
    actual.length !== expected.length ||
    actual.some((value, index) =>
      expected[index] === null
        ? value !== null
        : typeof value !== 'number' ||
          !Number.isFinite(value) ||
          Math.abs(value - expected[index]!) > 3e-4 * (1 + Math.abs(expected[index]!))
    )
  )
    throw new Error(
      `Landscape oracle mismatch ${table.id}.${column}: expected ${JSON.stringify(expected)}, received ${JSON.stringify(actual)}`
    );
}

for (const variant of ['sample', 'second'] as const) {
  it(`dispatches both landscape recipes on the mapped ${variant} raster and retains report-only cell identity`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const inputs = createLandscapeMetricsSample(variant);
    const analysis = getKeplerCuratedAnalysis('landscape-metrics')!;
    const profile = profileKeplerWorkbenchData(makeWorkflowRasterData(inputs.landcover, 4096));
    let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'landcover', datasetId: profile.datasetId},
      analysis,
      [profile]
    );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
    const parameters = {...request.parameters, edgeDepth: 0.4, searchRadius: 2};
    expect(request.inputDatasetIds['landcover']).toBe(inputs.landcover.id);
    const result = await runCatalogOperationAsync(
      CURATED_LANDSCAPE_METRICS_BINDING,
      parameters,
      inputs,
      {device}
    );
    expect(result.engine).toBe('gpu');
    expect(result.dispatchedContributors).toEqual([
      'GPURasterConnectedComponents',
      'addLandscapeMetricsRecipe',
      'addPatchProximityRecipe'
    ]);
    expect(result.summary['labelsConverged']).toBe(1);
    expect(result.tables.map(table => table.id)).toEqual([
      'cells',
      'patches',
      'classes',
      'adjacency',
      'distances',
      'landscape'
    ]);
    const table = (id: string) => result.tables.find(candidate => candidate.id === id)!;
    const patches = table('patches');
    const classes = table('classes');
    const landscape = table('landscape');
    const cells = table('cells');
    expect(result.tables.every(output => output.geometryColumn === undefined)).toBe(true);
    expect(Object.keys(cells.columns)).not.toContain('geometry');
    expect(makeWorkflowData(cells).kind).toBe('table');
    expect(CURATED_LANDSCAPE_METRICS_DESCRIPTOR.outputs[0].kind).toBe('attributes');
    expect(result.summary['mapPublication']).toBe('withheld-unreferenced-planar');
    expect(result.summary['gridAffine']).toBe(JSON.stringify(inputs.landcover.grid.affine));
    expect(
      result.diagnostics.some(diagnostic => diagnostic.code === 'unreferenced-planar-grid')
    ).toBe(true);
    expect(new Set(cells.columns['source_id'])).toEqual(new Set([inputs.landcover.id]));
    expect(new Set(cells.columns['band_index'])).toEqual(new Set([inputs.landcover.band.index]));
    for (let cell = 0; cell < cells.rowCount; cell++) {
      const patch = Number(cells.columns['patch_id'][cell]);
      expect(cells.columns['class_code'][cell]).toBe(patches.columns['class_code'][patch]);
      expect(cells.columns['patch_root_cell'][cell]).toBe(
        patches.columns['patch_root_cell'][patch]
      );
      const sourceCell = Number(cells.columns['cell_index'][cell]);
      expect(cells.columns['class_code'][cell]).toBe(inputs.landcover.band.values[sourceCell]);
      expect(cells.columns['cell_column'][cell]).toBe(sourceCell % inputs.landcover.grid.width);
      expect(cells.columns['cell_row'][cell]).toBe(
        Math.floor(sourceCell / inputs.landcover.grid.width)
      );
    }
    if (variant === 'sample') {
      // Same labels, dimensions and cell size as the existing urban-ecology recipe fixture.
      ['area', 'perimeter', 'mesh', 'cohesion'].forEach((field, index) =>
        checkColumn(classes, field, [
          reference.landscape.classMetrics[index],
          reference.landscape.classMetrics[4 + index]
        ])
      );
      checkColumn(classes, 'aggregation', reference.landscape.aggregation);
      checkColumn(table('adjacency'), 'side_count', reference.landscape.classAdjacency);
      ['contagion', 'mesh', 'area', 'richness'].forEach((field, index) =>
        checkColumn(landscape, field, [reference.landscape.landscape[index]])
      );
      checkColumn(patches, 'class_code', [10, 90]);
      checkColumn(patches, 'area', [4, 4]);
      checkColumn(patches, 'perimeter', [8, 8]);
      checkColumn(patches, 'core_area', [4, 4]);
      checkColumn(patches, 'nearest_same_class', [null, null]);
      checkColumn(patches, 'proximity', [0, 0]);
      checkColumn(table('distances'), 'distance', [0, 0]);
      checkColumn(cells, 'patch_id', [0, 0, 1, 1, 0, 0, 1, 1]);
      expect(result.diagnostics.some(diagnostic => diagnostic.code === 'undefined-metrics')).toBe(
        true
      );
    } else {
      // Three 4x9 rectangles have area 36 and perimeter 26. The same-class pair has a
      // 2-unit boundary gap, so proximity = neighbouring area / gap squared = 36/4 = 9.
      checkColumn(patches, 'patch_root_cell', [0, 3, 5]);
      checkColumn(patches, 'class_code', [-7, -7, 90]);
      checkColumn(patches, 'area', [36, 36, 36]);
      checkColumn(patches, 'perimeter', [26, 26, 26]);
      checkColumn(patches, 'core_area', [36, 36, 36]);
      checkColumn(patches, 'nearest_same_class', [2, 2, null]);
      checkColumn(patches, 'proximity', [9, 9, 0]);
      checkColumn(classes, 'area', [72, 36]);
      checkColumn(classes, 'perimeter', [52, 26]);
      checkColumn(classes, 'mesh', [24, 12]);
      checkColumn(landscape, 'area', [108]);
      checkColumn(landscape, 'mesh', [36]);
      checkColumn(table('distances'), 'distance', [2, 6, 2, 0, 6, 0]);
      checkColumn(table('adjacency'), 'side_count', [28, 3, 3, 14]);
      expect(result.summary['excludedCellCount']).toBe(3);
      expect(cells.columns['cell_index']).not.toContain(2);
      const restricted = await runCatalogOperationAsync(
        CURATED_LANDSCAPE_METRICS_BINDING,
        {...parameters, searchRadius: 1.9},
        inputs,
        {device}
      );
      const restrictedPatches = restricted.tables.find(candidate => candidate.id === 'patches')!;
      checkColumn(restrictedPatches, 'proximity', [0, 0, 0]);
      checkColumn(restrictedPatches, 'nearest_same_class', [2, 2, null]);
    }
  }, 120000);
}

it('handles diagonal connectivity, excluded class zero and zero boundary separation explicitly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const source = createLandscapeMetricsSample().landcover;
  const landcover = {
    ...source,
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    band: {
      ...source.band,
      values: new Int32Array([1, 0, 0, 1]),
      validity: new Uint8Array(4).fill(1)
    }
  };
  const run = (connectivity: string, distanceModel: string) =>
    runCatalogOperationAsync(
      CURATED_LANDSCAPE_METRICS_BINDING,
      {connectivity, excludeClass: true, excludedClass: 0, distanceModel},
      {landcover},
      {device}
    );
  const four = await run('4', 'cell-boundary');
  const fourPatches = four.tables.find(table => table.id === 'patches')!;
  checkColumn(fourPatches, 'area', [1, 1]);
  checkColumn(fourPatches, 'nearest_same_class', [0, 0]);
  checkColumn(fourPatches, 'proximity', [null, null]);
  expect(four.diagnostics.some(diagnostic => diagnostic.code === 'undefined-metrics')).toBe(true);
  const centered = await run('4', 'cell-center');
  const centeredPatches = centered.tables.find(table => table.id === 'patches')!;
  checkColumn(centeredPatches, 'nearest_same_class', [Math.SQRT2, Math.SQRT2]);
  checkColumn(centeredPatches, 'proximity', [0.5, 0.5]);
  const eight = await run('8', 'cell-boundary');
  const eightPatches = eight.tables.find(table => table.id === 'patches')!;
  checkColumn(eightPatches, 'area', [2]);
  checkColumn(eightPatches, 'perimeter', [8]);
  checkColumn(eightPatches, 'nearest_same_class', [null]);
}, 120000);

it('refuses to publish when any GPU class label has not converged', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const dispatch = vi
    .spyOn(GPURasterConnectedComponents.prototype, 'addToGraph')
    .mockImplementation(() => {});
  try {
    await expect(
      runCatalogOperationAsync(
        CURATED_LANDSCAPE_METRICS_BINDING,
        {},
        createLandscapeMetricsSample(),
        {device}
      )
    ).rejects.toThrow(/curated-landscape-metrics.*converged must equal 1/);
  } finally {
    dispatch.mockRestore();
  }
}, 120000);
