// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {
  CatalogCoordinateSystem,
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {
  CURATED_NETWORK_DISRUPTION_BINDING,
  MAXIMUM_NETWORK_DISRUPTION_ARC_COST,
  type CuratedNetworkDisruptionReadback
} from '../../src/curated/bindings/curated-network-disruption-binding';
import {
  CURATED_NETWORK_DISRUPTION_ANALYSIS,
  createNetworkDisruptionSample,
  createNetworkDisruptionSecondDataset,
  NETWORK_DISRUPTION_SAMPLE,
  NETWORK_DISRUPTION_EXPECTATIONS,
  NETWORK_DISRUPTION_SECOND_SAMPLE,
  NETWORK_DISRUPTION_SECOND_EXPECTATIONS
} from '../../src/curated/samples/network-disruption';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {CURATED_NETWORK_DISRUPTION_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-network-disruption';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';

/** Fail with the exact row and actual numeric value; NaN never passes an absolute-error check. */
function compareColumn(
  table: CatalogOutputTable,
  column: string,
  expected: ArrayLike<number>,
  identifier: string
) {
  const actual = table.columns[column];
  const mismatches: string[] = [];
  if (actual.length !== expected.length)
    mismatches.push(`length ${actual.length} != ${expected.length}`);
  Array.from(expected).forEach((value, row) => {
    const observed = actual[row];
    if (!Number.isFinite(value)) {
      if (observed !== null)
        mismatches.push(`row ${row}: ${String(observed)} != null (${String(value)})`);
    } else if (
      typeof observed !== 'number' ||
      !Number.isFinite(observed) ||
      Math.abs(observed - value) > 1e-5
    ) {
      mismatches.push(`row ${row}: ${String(observed)} != ${value} (absolute tolerance 1e-5)`);
    }
  });
  if (mismatches.length)
    throw new Error(
      `network-disruption ${identifier} ${table.id}.${column}:\n${mismatches.join('\n')}`
    );
}

function mapInputs(
  inputs: Record<string, CatalogFeatureCollection>,
  columns: Readonly<Record<string, string>>,
  identifier: string,
  maximumCost: number,
  coordinateSystem: CatalogCoordinateSystem
) {
  const analysis = CURATED_NETWORK_DISRUPTION_ANALYSIS;
  const profiles = Object.entries(inputs).map(([slot, collection]) =>
    profileKeplerFeatureCollection(collection, {
      datasetId: `${identifier}-${slot}`,
      label: `${identifier} ${slot}`,
      coordinateSystem,
      declarations: {
        coordinateModel: coordinateSystem === 'planar' ? 'planar' : 'geographic',
        rowSemantics: 'objects'
      }
    })
  );
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {
    parameters: {maximumCost, maximumSnapDistance: 0}
  });
  for (const slot of analysis.slots)
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: slot.name, datasetId: `${identifier}-${slot.name}`},
      analysis,
      profiles
    );
  for (const [key, column] of Object.entries(columns)) {
    const [slot, role] = key.split('.');
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot, role, columns: [column]},
      analysis,
      profiles
    );
  }
  expect(mapping.ready, JSON.stringify(mapping.problems)).toBe(true);
  const request = buildKeplerCuratedAnalysisRequest(analysis, mapping);
  // Mapping requests leave coordinate defaults unresolved. Resolve from the declared profiles
  // before catalog normalization can fall back to its geographic constant.
  const profilesByInput = Object.fromEntries(
    Object.entries(request.inputDatasetIds).map(([input, datasetId]) => [
      input,
      profiles.find(profile => profile.datasetId === datasetId)!
    ])
  );
  const resolved = resolveKeplerOperationParameters(
    CURATED_NETWORK_DISRUPTION_BINDING,
    CURATED_NETWORK_DISRUPTION_DESCRIPTOR,
    profilesByInput,
    {overrides: request.parameters, executionEngine: 'gpu'}
  );

  const datasets = Object.fromEntries(
    Object.entries(inputs).map(([slot, collection]) => [`${identifier}-${slot}`, collection])
  );
  return {
    ...request,
    parameters: resolved.parameters,
    inputs: Object.fromEntries(
      Object.entries(request.inputDatasetIds).map(([slot, datasetId]) => [
        slot,
        datasets[datasetId]
      ])
    )
  };
}

it('dispatches actual disruption routing on the mapped recipe fixture and independent renamed parallel-edge dataset', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = CURATED_NETWORK_DISRUPTION_BINDING;
  for (const [inputs, specification, expected] of [
    [createNetworkDisruptionSample(), NETWORK_DISRUPTION_SAMPLE, NETWORK_DISRUPTION_EXPECTATIONS],
    [
      createNetworkDisruptionSecondDataset(),
      NETWORK_DISRUPTION_SECOND_SAMPLE,
      NETWORK_DISRUPTION_SECOND_EXPECTATIONS
    ]
  ] as const) {
    const request = mapInputs(
      inputs,
      specification.columns,
      specification.id,
      specification.parameters.maximumCost,
      specification.parameters.coordinateSystem
    );
    expect(request.parameters['coordinateSystem'], specification.id).toBe(
      specification.parameters.coordinateSystem
    );
    const result = await runCatalogOperationAsync(binding, request.parameters, request.inputs, {
      device
    });
    expect(result.engine, JSON.stringify(result)).toBe('gpu');
    expect(result.dispatchedContributors).toEqual(binding.gpuContributorNames);
    const pairs = result.tables.find(table => table.id === 'pairs')!;
    const demand = result.tables.find(table => table.id === 'demand-access')!;
    const columns: [keyof CuratedNetworkDisruptionReadback, CatalogOutputTable, string][] = [
      ['baselineCosts', pairs, 'baseline_cost'],
      ['scenarioCosts', pairs, 'scenario_cost'],
      ['costDifferences', pairs, 'cost_difference'],
      ['baselineAccessibility', demand, 'baseline_accessibility'],
      ['scenarioAccessibility', demand, 'scenario_accessibility'],
      ['accessibilityDifferences', demand, 'accessibility_difference']
    ];
    for (const [name, table, column] of columns)
      compareColumn(table, column, expected[name] as ArrayLike<number>, specification.id);
    expect(pairs.columns['newly_unreachable']).toEqual(
      Array.from(expected.newlyUnreachable, Boolean)
    );
    expect(demand.columns['demand_source_id']).toEqual(
      inputs['demand'].features.map(feature => feature.id)
    );
    expect(
      pairs.columns['facility_source_id'].slice(0, inputs['facilities'].features.length)
    ).toEqual(inputs['facilities'].features.map(feature => feature.id));
    expect(demand.columns['geometry']).toEqual(
      inputs['demand'].features.map(feature => JSON.stringify(feature.geometry))
    );
    expect(result.summary).toMatchObject({
      converged: 1,
      costUnit: 'minutes',
      complete: 1,
      baselineRoutingConverged: 1,
      scenarioRoutingConverged: 1,
      baselineAccessibilityComplete: 1,
      scenarioAccessibilityComplete: 1
    });
    for (const table of result.tables) {
      expect(Object.values(table.columns).every(column => column.length === table.rowCount)).toBe(
        true
      );
      if (table.geometryColumn) {
        const descriptor = CURATED_NETWORK_DISRUPTION_DESCRIPTOR.outputs.find(
          output => output.tableId === table.id
        )!;
        const presentation = planKeplerResultPresentation(table, descriptor);
        expect(presentation, `${specification.id}/${table.id}`).toBeDefined();
      }
    }
    const arcs = result.tables.find(table => table.id === 'scenario-arcs')!;
    expect(arcs.columns['csr_slot']).toEqual(inputs['network'].features.map((_, index) => index));
    expect(arcs.columns['source_id']).toEqual(
      inputs['network'].features.map(feature => feature.id)
    );
  }
});

it.each([
  'baseline-accessibility',
  'scenario-accessibility'
])('refuses device publication when %s scoring evidence is incomplete', async failedStage => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const originalRead = CatalogGpuSession.prototype.readScalar;
  const submitted = vi.spyOn(device, 'submit');
  let matchingReadCompleted = false;
  const read = vi
    .spyOn(CatalogGpuSession.prototype, 'readScalar')
    .mockImplementation(async function (this: CatalogGpuSession, output) {
      const value = await originalRead.call(this, output);
      if (output.buffer.id.endsWith(`${failedStage}-complete`)) {
        matchingReadCompleted = true;
        return 0;
      }
      return value;
    });
  try {
    await expect(
      CURATED_NETWORK_DISRUPTION_BINDING.runGpu!(
        {device},
        createNetworkDisruptionSample(),
        NETWORK_DISRUPTION_SAMPLE.parameters
      )
    ).rejects.toThrow(
      `${failedStage} must be complete before publishing accessibility comparisons`
    );
    expect(submitted).toHaveBeenCalledOnce();
    expect(submitted.mock.results[0]?.type).toBe('return');
    expect(matchingReadCompleted).toBe(true);
  } finally {
    read.mockRestore();
    submitted.mockRestore();
  }
});

it('recomputes baseline/scenario on closure edits, keeps parallel CSR slots, and distinguishes complete cut-off from zero supply', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = createNetworkDisruptionSecondDataset();
  const specification = NETWORK_DISRUPTION_SECOND_SAMPLE;
  const first = await runCatalogOperationAsync(
    CURATED_NETWORK_DISRUPTION_BINDING,
    specification.parameters,
    inputs,
    {device}
  );
  inputs['network'].features[0].properties!['blocked'] = 0;
  const reopened = await runCatalogOperationAsync(
    CURATED_NETWORK_DISRUPTION_BINDING,
    specification.parameters,
    inputs,
    {device}
  );
  compareColumn(reopened.tables[1], 'scenario_cost', [3, 0, 1], 'reopened');
  compareColumn(reopened.tables[0], 'accessibility_difference', [0], 'reopened');
  expect(first.tables[2].columns['csr_slot']).toEqual(reopened.tables[2].columns['csr_slot']);
  expect(first.tables[2].columns['edge_id']).toEqual(reopened.tables[2].columns['edge_id']);
  inputs['facilities'].features = [inputs['facilities'].features[0]];
  inputs['facilities'].features[0].properties!['capacity'] = 0;
  inputs['network'].features.forEach(feature => {
    feature.properties!['blocked'] = 1;
  });
  const closed = await runCatalogOperationAsync(
    CURATED_NETWORK_DISRUPTION_BINDING,
    specification.parameters,
    inputs,
    {device}
  );
  compareColumn(closed.tables[1], 'baseline_cost', [3], 'all-closed');
  compareColumn(closed.tables[1], 'scenario_cost', [Infinity], 'all-closed');
  compareColumn(closed.tables[0], 'accessibility_difference', [0], 'zero-supply');
  expect(closed.tables[0].columns['cut_off']).toEqual([true]);
  expect(closed.summary).toMatchObject({newlyUnreachableCount: 1, cutOffCount: 1, converged: 1});
});

it('rejects overflow-prone detours and preserves reachability at the supported arc-cost ceiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = createNetworkDisruptionSecondDataset();
  inputs['network'].features[0].geometry = {
    type: 'LineString',
    coordinates: [
      [100, 200],
      [120, 200]
    ]
  };
  inputs['facilities'].features = [inputs['facilities'].features[0]];
  for (const feature of inputs['network'].features.slice(1))
    feature.properties!['travelTime'] = 2e38;
  const parameters = NETWORK_DISRUPTION_SECOND_SAMPLE.parameters;
  await expect(
    runCatalogOperationAsync(CURATED_NETWORK_DISRUPTION_BINDING, parameters, inputs, {device})
  ).rejects.toThrow(/maximumArcCost.*rescale/);
  for (const feature of inputs['network'].features.slice(1))
    feature.properties!['travelTime'] = MAXIMUM_NETWORK_DISRUPTION_ARC_COST;
  const result = await runCatalogOperationAsync(
    CURATED_NETWORK_DISRUPTION_BINDING,
    parameters,
    inputs,
    {device}
  );
  expect(result.engine, JSON.stringify(result)).toBe('gpu');
  compareColumn(result.tables[1], 'baseline_cost', [1], 'cost-ceiling');
  compareColumn(
    result.tables[1],
    'scenario_cost',
    [2 * MAXIMUM_NETWORK_DISRUPTION_ARC_COST],
    'cost-ceiling'
  );
  expect(result.tables[1].columns['newly_unreachable']).toEqual([false]);
  expect(result.tables[0].columns['cut_off']).toEqual([false]);
  expect(result.summary).toMatchObject({newlyUnreachableCount: 0, cutOffCount: 0, converged: 1});
});
