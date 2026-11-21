// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogRasterInput} from '../../src/analysis/catalog-raster-input';
import {CURATED_DRAINAGE_NETWORK_BINDING} from '../../src/curated/bindings/curated-drainage-network-binding';
import {
  createDrainageNetworkSample,
  createDrainageNetworkSecondDataset,
  CURATED_DRAINAGE_NETWORK_ANALYSIS,
  DRAINAGE_NETWORK_EXPECTATIONS,
  DRAINAGE_NETWORK_SECOND_EXPECTATIONS
} from '../../src/curated/samples/drainage-network';
import {
  createKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {CURATED_DRAINAGE_NETWORK_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-drainage-network';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {makeWorkflowOperationFromCatalogBinding} from '../../src/workflows/workflow-operations';
import {makeWorkflowRasterData} from '../../src/workflows/workflow-data';
import {makeFixtureHost} from '../workflows/workflow-fixtures';
import {
  compareDrainageColumn as compareColumn,
  createDrainageNoDataCollisionRaster
} from './curated-drainage-network-test-support';

function mapRaster(raster: CatalogRasterInput) {
  const profile: KeplerDatasetProfile = {
    datasetId: raster.id,
    label: raster.band.name!,
    kind: 'surface',
    rowCount: raster.grid.width * raster.grid.height,
    geometryColumns: [],
    columns: [],
    coordinates: {model: 'planar', source: 'declared'}
  };
  const mapping = createKeplerCuratedAnalysisMapping(CURATED_DRAINAGE_NETWORK_ANALYSIS, [profile]);
  expect(mapping.ready).toBe(true);
  return buildKeplerCuratedAnalysisRequest(CURATED_DRAINAGE_NETWORK_ANALYSIS, mapping, {
    parameterOverrides: {streamThreshold: 1}
  });
}

it('dispatches the real drainage recipe on its existing 3x3 fixture and independently mapped two-outlet survey', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = CURATED_DRAINAGE_NETWORK_BINDING;
  for (const [raster, expected] of [
    [createDrainageNetworkSample().elevation, DRAINAGE_NETWORK_EXPECTATIONS],
    [createDrainageNetworkSecondDataset().survey, DRAINAGE_NETWORK_SECOND_EXPECTATIONS]
  ] as const) {
    const request = mapRaster(raster);
    const mappedDataset = {[raster.id]: raster};
    const result = await runCatalogOperationAsync(
      binding,
      request.parameters,
      {
        elevation: mappedDataset[request.inputDatasetIds['elevation']]
      },
      {device}
    );
    expect(result.engine, JSON.stringify(result)).toBe('gpu');
    expect(result.dispatchedContributors).toEqual(binding.gpuContributorNames);
    compareColumn(result, 'flow_direction', expected.flowDirections);
    compareColumn(result, 'accumulation', expected.accumulation);
    compareColumn(result, 'catchment_id', expected.catchments);
    compareColumn(result, 'stream_order', expected.streamOrder);
    // Float32 sums of at most four short segments, compared to hand-computed lengths.
    compareColumn(result, 'downstream_distance', expected.distance, 1e-5);
    compareColumn(result, 'terminal_cell', expected.catchments);
    const streams = result.tables.find(table => table.id === 'streams')!;
    expect(streams.rowCount).toBe(expected.reachCount);
    const center = (cell: number) => [
      raster.grid.affine[2] + raster.grid.affine[0] * ((cell % raster.grid.width) + 0.5),
      raster.grid.affine[5] + raster.grid.affine[4] * (Math.floor(cell / raster.grid.width) + 0.5)
    ];
    for (let row = 0; row < streams.rowCount; row++) {
      const geometry = JSON.parse(String(streams.columns['geometry'][row]));
      const coordinates = geometry.coordinates as number[][];
      expect(geometry.type).toBe('LineString');
      expect(coordinates.length).toBeGreaterThanOrEqual(2);
      expect(coordinates[0]).toEqual(center(Number(streams.columns['source_cell'][row])));
      expect(coordinates.at(-1)).toEqual(center(Number(streams.columns['target_cell'][row])));
      let length = 0;
      for (let vertex = 1; vertex < coordinates.length; vertex++) {
        length += Math.hypot(
          coordinates[vertex][0] - coordinates[vertex - 1][0],
          coordinates[vertex][1] - coordinates[vertex - 1][1]
        );
      }
      expect(Math.abs(length - Number(streams.columns['length'][row]))).toBeLessThan(1e-5);
    }
    const summaries = result.tables.find(table => table.id === 'catchments')!;
    const catchmentIds = [...new Set(expected.catchments)];
    expect(summaries.rowCount).toBe(catchmentIds.length);
    for (const catchmentId of catchmentIds) {
      const row = summaries.columns['catchment_id'].indexOf(catchmentId);
      const members = expected.catchments
        .map((value, cell) => (value === catchmentId ? cell : -1))
        .filter(cell => cell >= 0);
      expect(summaries.columns['cell_count'][row]).toBe(members.length);
      const meanDistance =
        members.reduce((sum, cell) => sum + expected.distance[cell], 0) / members.length;
      expect(
        Math.abs(Number(summaries.columns['mean_downstream_distance'][row]) - meanDistance)
      ).toBeLessThan(1e-5);
    }
    for (const table of result.tables) {
      expect(table.columns['source_id']).toEqual(Array(table.rowCount).fill(raster.id));
      expect(table.columns['band_index']).toEqual(Array(table.rowCount).fill(raster.band.index));
      const descriptor = CURATED_DRAINAGE_NETWORK_DESCRIPTOR.outputs.find(
        output => output.tableId === table.id
      )!;
      if (table.geometryColumn) {
        expect(
          table.coordinateSystem,
          JSON.stringify({table: table.id, summary: result.summary})
        ).toBe('planar');
        expect(planKeplerResultPresentation(table, descriptor)).toBeDefined();
      }
    }
    // Matches expansion-recipes.spec.ts: distance of center cell 4 is 2, terminal cell is 8.
    if (raster.grid.width === 3) {
      expect(result.tables[1].columns['downstream_distance'][4]).toBe(2);
      expect(result.tables[1].columns['terminal_cell'][4]).toBe(8);
    }
    expect(result.summary).toMatchObject({
      converged: 1,
      overflow: 0,
      unresolvedCount: 0
    });
    expect(result.summary).not.toHaveProperty('uncertainCount');
    expect(result.summary['recipeStatus']).toContain('"unresolvedCount":0');
    expect(result.summary['recipeStatus']).not.toContain('"uncertainCount"');
  }
});

it.each([
  'downstream-distance',
  'stream-network'
])('withholds unresolved-only %s device readbacks and publishes after recovery', async stage => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let unresolvedCount = 1;
  const originalRead = CatalogGpuSession.prototype.readUint32;
  const read = vi
    .spyOn(CatalogGpuSession.prototype, 'readUint32')
    .mockImplementation(async function (this: CatalogGpuSession, output) {
      const values = await originalRead.call(this, output);
      if (output.buffer.id.endsWith(`${stage}-unresolvedCount`)) values[0] = unresolvedCount;
      return values;
    });
  const run = () =>
    CURATED_DRAINAGE_NETWORK_BINDING.runDataGpu!({device}, createDrainageNetworkSample(), {});
  try {
    await expect(run()).rejects.toThrow(new RegExp(`${stage}.*unresolvedCount`));
    unresolvedCount = 0;
    const recovered = await run();
    expect(recovered.summary['unresolvedCount']).toBe(0);
    // Hand-worked 3x3 D8 fixture: three vertical columns merge along the bottom row.
    compareColumn(recovered, 'accumulation', [1, 1, 1, 2, 2, 2, 3, 6, 9]);
  } finally {
    read.mockRestore();
  }
});

it('preserves the native raster in a single workflow step and publishes geometry-bearing outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {survey} = createDrainageNetworkSecondDataset();
  const operation = makeWorkflowOperationFromCatalogBinding(CURATED_DRAINAGE_NETWORK_BINDING);
  const result = await operation.run(
    {
      device,
      engine: 'auto',
      signal: new AbortController().signal,
      host: makeFixtureHost(),
      workflowId: 'drainage',
      stepId: 'analysis'
    },
    {elevation: makeWorkflowRasterData(survey, 8)},
    {streamThreshold: 1}
  );
  expect(result.engine).toBe('gpu');
  expect(result.outputs['streams'].kind).toBe('line');
  expect(result.outputs['streams'].table.geometryColumn).toBe('geometry');
  expect(result.outputs['catchment-cells'].identityColumn).toBe('cell_index');
  expect(result.outputs['catchments'].identityColumn).toBe('catchment_id');
});

it('honors independent validity, nodata and nonfinite samples on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {elevation} = createDrainageNetworkSample();
  elevation.band.validity[0] = 0;
  elevation.band.values[2] = NaN;
  const result = await runCatalogOperationAsync(
    CURATED_DRAINAGE_NETWORK_BINDING,
    {streamThreshold: 1},
    {elevation: {...elevation, band: {...elevation.band, noDataValue: 8}}},
    {device}
  );
  expect(result.engine).toBe('gpu');
  expect(result.summary['invalidCellCount']).toBe(3);
  expect(result.tables[1].columns['status'].slice(0, 3)).toEqual(['no-data', 'no-data', 'no-data']);
  expect(result.tables[1].columns['downstream_distance'].slice(0, 3)).toEqual([null, null, null]);
  expect(
    result.tables[2].columns['cell_count'].reduce<number>((sum, value) => sum + Number(value), 0)
  ).toBe(6);
});

it('preserves a valid cell whose float32 value collides with the raw Float64 or integer nodata sentinel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {values, sentinel} of [
    {values: new Float64Array([16777216, 16777217]), sentinel: 16777217},
    {values: new Int32Array([-16777216, -16777217]), sentinel: -16777217},
    {values: new Uint32Array([4294967294, 4294967295]), sentinel: 4294967295}
  ]) {
    const elevation = createDrainageNoDataCollisionRaster(values, sentinel);
    const result = await runCatalogOperationAsync(
      CURATED_DRAINAGE_NETWORK_BINDING,
      {streamThreshold: 1},
      {elevation},
      {device}
    );
    expect(result.engine).toBe('gpu');
    const cells = result.tables.find(table => table.id === 'catchment-cells')!;
    expect(cells.columns['status']).toEqual(['analysed', 'no-data']);
    expect(cells.columns['accumulation']).toEqual([1, null]);
    expect(cells.columns['catchment_id']).toEqual([0, null]);
    expect(cells.columns['terminal_cell']).toEqual([0, null]);
    expect(result.tables.find(table => table.id === 'catchments')!.columns['cell_count']).toEqual([
      1
    ]);
  }
});

it('rejects GPU path truncation and stream geometry overflow before returning publishable tables', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const parameters of [{maximumSteps: 1}, {reachCapacity: 1}, {vertexCapacity: 1}]) {
    await expect(
      runCatalogOperationAsync(
        CURATED_DRAINAGE_NETWORK_BINDING,
        {streamThreshold: 1, ...parameters},
        createDrainageNetworkSample(),
        {device}
      )
    ).rejects.toThrow(/curated-drainage-network.*recipe status.*(overflow|unresolvedCount)/);
  }
});
