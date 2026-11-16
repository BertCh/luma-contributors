// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogDataInput} from '../../src/analysis/catalog-data-input';
import {CURATED_REGIONAL_RASTER_CHANGE_BINDING} from '../../src/curated/bindings/curated-regional-raster-change-binding';
import {
  createRegionalRasterChangeSample,
  REGIONAL_RASTER_CHANGE_ANALYSIS_UPDATE
} from '../../src/curated/samples/regional-raster-change';
import {CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-regional-raster-change';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';

import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';

const REGIONAL_RASTER_CHANGE_ANALYSIS = {
  ...getKeplerCuratedAnalysis('regional-raster-change')!,
  ...REGIONAL_RASTER_CHANGE_ANALYSIS_UPDATE
};

for (const secondDataset of [false, true]) {
  it(`dispatches regional change on mapped ${secondDataset ? 'independent north-up data' : 'normalized recipe fixture'} and publishes coverage`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createRegionalRasterChangeSample(secondDataset);
    // Slot mapping addresses genuine decoded bands by dataset identity. No fake points or
    // column-name roles are introduced for an already selected numeric raster band.
    const datasets: Record<string, CatalogDataInput> = {
      [input.before.id]: input.before,
      [input.after.id]: input.after,
      districts: input.zones
    };
    const profiles: KeplerDatasetProfile[] = [
      ...[input.before, input.after].map(raster => ({
        datasetId: raster.id,
        label: raster.band.name!,
        kind: 'surface' as const,
        rowCount: raster.grid.width * raster.grid.height,
        geometryColumns: [],
        columns: []
      })),
      profileKeplerFeatureCollection(input.zones, {
        datasetId: 'districts',
        label: 'Districts',
        coordinateSystem: 'planar'
      })
    ];
    let mapping = createKeplerCuratedAnalysisMapping(REGIONAL_RASTER_CHANGE_ANALYSIS, profiles);
    for (const [slot, datasetId] of [
      ['before', input.before.id],
      ['after', input.after.id],
      ['zones', 'districts']
    ] as const)
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot, datasetId},
        REGIONAL_RASTER_CHANGE_ANALYSIS,
        profiles
      );
    const request = buildKeplerCuratedAnalysisRequest(REGIONAL_RASTER_CHANGE_ANALYSIS, mapping, {
      parameterOverrides: {threshold: 1, confirmComparable: true}
    });
    const result = await runCatalogOperationAsync(
      CURATED_REGIONAL_RASTER_CHANGE_BINDING,
      request.parameters,
      Object.fromEntries(
        Object.entries(request.inputDatasetIds).map(([slot, datasetId]) => [
          slot,
          datasets[datasetId]
        ])
      ),
      {device}
    );
    const mismatches: string[] = [];
    const check = (name: string, actual: unknown, expected: unknown) => {
      if (
        typeof actual === 'number' && typeof expected === 'number'
          ? !Number.isFinite(actual) ||
            Math.abs(actual - expected) > 1e-5 * Math.max(1, Math.abs(expected))
          : JSON.stringify(actual) !== JSON.stringify(expected)
      )
        mismatches.push(`${name}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
    };
    check('engine', result.engine, 'gpu');
    check('dispatch', result.dispatchedContributors, [
      'GPUPolygonRasterization',
      'GPUChangeDetection',
      'GPURasterZonalStatistics'
    ]);
    const regions = result.tables.find(table => table.id === 'regions');
    const differences = result.tables.find(table => table.id === 'differences');
    if (!regions || !differences)
      throw new Error(`Missing publication tables: ${JSON.stringify(result)}`);
    check('region rows', regions.rowCount, secondDataset ? 3 : 2);
    check('difference rows', differences.rowCount, secondDataset ? 6 : 4);
    check('geometry column', regions.geometryColumn, 'geometry');
    check('geometry CRS', regions.coordinateSystem, 'planar');
    check(
      'source IDs',
      regions.columns['source_id'],
      input.zones.features.map(feature => feature.id)
    );
    check('zone identity', regions.columns['zone_index'], secondDataset ? [0, 1, 2] : [0, 1]);
    check(
      'cell identity',
      differences.columns['cell_index'],
      secondDataset ? [0, 1, 2, 3, 4, 5] : [0, 1, 2, 3]
    );
    check(
      'cell ownership',
      differences.columns['zone_index'],
      secondDataset ? [0, 0, 1, 0, 0, 1] : [0, 0, 1, 1]
    );
    check(
      'common mask',
      differences.columns['common_observation'],
      secondDataset ? [true, true, false, true, false, true] : [true, false, true, true]
    );
    check(
      'differences',
      differences.columns['difference'],
      secondDataset ? [2, -2, null, 1, null, 0] : [2, null, -1, 0]
    );
    const expectedColumns = secondDataset
      ? {
          valid_count: [3, 1, 0],
          mean_change: [1 / 3, 0, null],
          total_area: [24, 12, 0],
          valid_area: [18, 6, 0],
          changed_area: [18, 0, 0],
          coverage: [0.75, 0.5, null],
          changed_share: [1, 0, null]
        }
      : {
          // expansion-recipes.spec.ts: same differences, means [2,-0.5], coverage [0.5,1].
          // The fixture injects areas [2,2,3,3]; this realizable uniform planar grid uses area 2.
          // Thus its changedArea [2,3] becomes [2,2], with unchanged coverage and means.
          valid_count: [1, 2],
          mean_change: [2, -0.5],
          total_area: [4, 4],
          valid_area: [2, 4],
          changed_area: [2, 2],
          coverage: [0.5, 1],
          changed_share: [1, 0.5]
        };
    for (const [column, expected] of Object.entries(expectedColumns)) {
      expected.forEach((value, row) =>
        check(`${column}[${row}]`, regions.columns[column][row], value)
      );
      check(
        `schema ${column}`,
        regions.columnTypes[column],
        column === 'valid_count' ? 'int32' : 'float64'
      );
    }
    for (let zone = 0; zone < regions.rowCount; zone++)
      check(
        `geometry ${zone}`,
        JSON.parse(String(regions.columns['geometry'][zone])),
        input.zones.features[zone].geometry
      );
    const presentation = planKeplerResultPresentation(
      regions,
      CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR.outputs[0]
    );
    check('published column', presentation?.column, 'mean_change');
    // The map colours zone means, not cell differences. The second dataset has means
    // [1/3, 0, null], so the shared planner adapts its signed measure to sequential.
    // The first fixture's means [2, -0.5] retain the diverging presentation gate.
    check('published scale', presentation?.scale, secondDataset ? 'sequential' : 'diverging');
    check('status overflow', result.summary['rasterOverflow'], 0);
    check('common count', result.summary['commonObservationCount'], secondDataset ? 4 : 3);
    check(
      'missing diagnostic',
      result.diagnostics.find(diagnostic => diagnostic.code === 'missing-common-observations')
        ?.count,
      secondDataset ? 2 : 1
    );
    if (mismatches.length)
      throw new Error(
        `Regional raster change (${secondDataset ? 'second dataset' : 'recipe fixture'}):\n${mismatches.join('\n')}\nSummary: ${JSON.stringify(result.summary)}`
      );
  }, 120000);
}

it('keeps holes/outside cells unassigned, honours point pixels and resolves overlaps by source row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createRegionalRasterChangeSample();
  const outer = [
    [-0.5, -1],
    [1.5, -1],
    [1.5, 3],
    [-0.5, 3],
    [-0.5, -1]
  ];
  // Point pixels have centres at (0,0), (1,0), (0,2), (1,2). Exclude the first centre.
  const hole = [
    [-0.25, -0.25],
    [0.25, -0.25],
    [0.25, 0.25],
    [-0.25, 0.25],
    [-0.25, -0.25]
  ];
  const zones = {
    type: 'FeatureCollection' as const,
    features: [
      {id: 'with-hole', geometry: {type: 'Polygon', coordinates: [outer, hole]}},
      {id: 'overlap', geometry: {type: 'Polygon', coordinates: [outer]}}
    ]
  };
  const grid = {...sample.before.grid, pixelInterpretation: 'point' as const};
  const result = await runCatalogOperationAsync(
    CURATED_REGIONAL_RASTER_CHANGE_BINDING,
    {threshold: 1, confirmComparable: true},
    {...sample, before: {...sample.before, grid}, after: {...sample.after, grid}, zones},
    {device}
  );
  const cells = result.tables.find(table => table.id === 'differences')!;
  if (JSON.stringify(cells.columns['zone_index']) !== '[1,0,0,0]')
    throw new Error(
      `Point-pixel hole/overlap ownership mismatch: ${JSON.stringify(cells.columns)}`
    );
  const outside = await runCatalogOperationAsync(
    CURATED_REGIONAL_RASTER_CHANGE_BINDING,
    {threshold: 1, confirmComparable: true},
    {
      ...sample,
      before: {...sample.before, grid},
      after: {...sample.after, grid},
      zones: {...zones, features: [zones.features[0]]}
    },
    {device}
  );
  if (
    outside.summary['outsideCellCount'] !== 1 ||
    outside.tables[1].columns['zone_index'][0] !== null
  )
    throw new Error(`Outside cell was assigned: ${JSON.stringify(outside)}`);
});

it('distinguishes wholly unobserved support from no assigned grid support, and includes zero change at threshold zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createRegionalRasterChangeSample(true);
  const missing = await runCatalogOperationAsync(
    CURATED_REGIONAL_RASTER_CHANGE_BINDING,
    {threshold: 0, confirmComparable: true},
    {
      ...sample,
      before: {...sample.before, band: {...sample.before.band, validity: new Uint8Array(6)}}
    },
    {device}
  );
  const regions = missing.tables[0];
  const expected = {
    valid_count: [0, 0, 0],
    mean_change: [null, null, null],
    coverage: [0, 0, null],
    valid_area: [0, 0, 0],
    changed_share: [null, null, null]
  };
  for (const [column, values] of Object.entries(expected))
    if (JSON.stringify(regions.columns[column]) !== JSON.stringify(values))
      throw new Error(
        `Unobserved ${column}: ${JSON.stringify(regions.columns[column])} != ${JSON.stringify(values)}; ${JSON.stringify(missing.summary)}`
      );
  const zero = await runCatalogOperationAsync(
    CURATED_REGIONAL_RASTER_CHANGE_BINDING,
    {threshold: 0, confirmComparable: true},
    sample,
    {device}
  );
  if (JSON.stringify(zero.tables[0].columns['changed_area']) !== '[18,6,0]')
    throw new Error(
      `Threshold zero must include observed zero changes: ${JSON.stringify(zero.tables[0].columns)}`
    );
});
