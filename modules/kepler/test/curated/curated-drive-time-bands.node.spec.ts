// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, it, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {getDriveTimeOracle} from '../../../analysis/test/gpu-network/recipes/drive-time-oracle';
import {snapOracle} from '../../../analysis/test/gpu-network/network-accessibility/network-accessibility-oracle';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {
  CURATED_DRIVE_TIME_BANDS_BINDING,
  DRIVE_TIME_BANDS_LIMITS,
  prepareCuratedDriveTimeBands,
  publishCuratedDriveTimeBands,
  type PreparedCuratedDriveTimeBands,
  type CuratedDriveTimeBandsReadback
} from '../../src/curated/bindings/curated-drive-time-bands-binding';
import {
  CURATED_DRIVE_TIME_BANDS_ANALYSIS,
  DRIVE_TIME_BANDS_SAMPLE,
  DRIVE_TIME_BANDS_EXPECTATIONS,
  DRIVE_TIME_BANDS_SECOND_SAMPLE,
  DRIVE_TIME_BANDS_SECOND_EXPECTATIONS,
  createDriveTimeBandsSample,
  createDriveTimeBandsSecondDataset,
  createDriveTimeBandsTemplate
} from '../../src/curated/samples/drive-time-bands';
import {KEPLER_CURATED_ANALYSES} from '../../src/curated/curated-analyses';
import {CURATED_DRIVE_TIME_BANDS_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-drive-time-bands';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import * as operationDescriptors from '../../src/descriptors/operation-descriptors';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest,
  validateKeplerCuratedAnalysis
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {
  createWorkflowOperationRegistry,
  getWorkflowOperationRegistry,
  makeWorkflowOperationFromCatalogBinding
} from '../../src/workflows/workflow-operations';
import {
  parseWorkflowAnalysisTemplate,
  serializeWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template-parser';

const binding = CURATED_DRIVE_TIME_BANDS_BINDING;
const descriptor = CURATED_DRIVE_TIME_BANDS_DESCRIPTOR;
const cases = [
  {
    inputs: createDriveTimeBandsSample,
    sample: DRIVE_TIME_BANDS_SAMPLE,
    expected: DRIVE_TIME_BANDS_EXPECTATIONS
  },
  {
    inputs: createDriveTimeBandsSecondDataset,
    sample: DRIVE_TIME_BANDS_SECOND_SAMPLE,
    expected: DRIVE_TIME_BANDS_SECOND_EXPECTATIONS
  }
] as const;

/** Same independent inserted-node Dijkstra oracle as drive-time-catchment-recipe.spec.ts.
 * Only CSR source rows are derived here; neither routing nor snapping is reimplemented.
 */
function getReference(prepared: PreparedCuratedDriveTimeBands) {
  const sources = new Uint32Array(prepared.topology.neighbors.length);
  for (let node = 0; node < prepared.graph.nodes.length; node++)
    sources.fill(node, prepared.topology.offsets[node], prepared.topology.offsets[node + 1]);
  const topology = {...prepared.topology, sources};
  const options = {edgeCosts: topology.weights, maximumSnapDistance: prepared.maximumSnapDistance};
  const facilities = snapOracle(
    prepared.facilityPositions,
    prepared.nodePositions,
    sources,
    topology.neighbors,
    options
  );
  const demand = snapOracle(
    prepared.demandPositions,
    prepared.nodePositions,
    sources,
    topology.neighbors,
    options
  );
  return {
    facilities,
    demand,
    ...getDriveTimeOracle(topology, prepared.graph.nodes.length, facilities, demand)
  };
}

function createReadback(
  prepared: PreparedCuratedDriveTimeBands,
  expected: (typeof cases)[number]['expected']
): CuratedDriveTimeBandsReadback {
  const reference = getReference(prepared);
  return {
    times: reference.demandTimes,
    demandBands: expected.demandBands,
    demandEdges: reference.demand.map(snap => snap.edge),
    facilityEdges: reference.facilities.map(snap => snap.edge),
    keys: expected.keys,
    counts: expected.counts,
    sums: expected.sums,
    means: expected.means,
    converged: 1,
    bandCount: expected.keys.length,
    requiredBandCount: expected.keys.length,
    bandOverflow: 0,
    triangleCount: 1,
    triangleOverflow: 0,
    triangles: [0, 0, 1, 0, 0, 1],
    triangleBands: [0]
  };
}

function mapInputs(
  inputs: Record<string, CatalogFeatureCollection>,
  columns: Readonly<Record<string, string>>,
  parameters: Readonly<Record<string, CatalogParameterValue>>
) {
  const analysis = CURATED_DRIVE_TIME_BANDS_ANALYSIS;
  const profiles = Object.entries(inputs).map(([slot, collection]) =>
    profileKeplerFeatureCollection(collection, {
      datasetId: slot,
      label: slot,
      coordinateSystem: 'planar',
      declarations: {
        coordinateModel: 'planar',
        rowSemantics: 'objects',
        measurement: Object.fromEntries(
          Object.entries(columns)
            .filter(([key]) => key.startsWith(`${slot}.`) && !key.endsWith('one-way'))
            .map(([, column]) => [column, 'amount'])
        )
      }
    })
  );
  let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles, {parameters});
  for (const slot of analysis.slots)
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: slot.name, datasetId: slot.name},
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
  return buildKeplerCuratedAnalysisRequest(analysis, mapping);
}

describe('curated drive-time bands', () => {
  it('keeps reachable demand under its truthful existing description and separate identity', () => {
    const accessibility = KEPLER_CURATED_ANALYSES.find(
      analysis => analysis.id === 'drive-time-catchments'
    )!;
    expect(accessibility.name).toBe('Demand within reach of each facility');
    expect(accessibility.execution).toMatchObject({bindingId: 'network-accessibility'});
    expect(accessibility.method).toContain('not bands of travel time or catchment polygons');
    expect(CURATED_DRIVE_TIME_BANDS_ANALYSIS.id).toBe('drive-time-bands');
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
  });

  it('matches both hand-computed fixtures with the existing recipe oracle', () => {
    for (const fixture of cases) {
      const prepared = prepareCuratedDriveTimeBands(fixture.inputs(), fixture.sample.parameters);
      const reference = getReference(prepared);
      reference.demandTimes.forEach((time, row) => {
        if (time === Infinity) expect(fixture.expected.times[row]).toBe(Infinity);
        else
          expect(time, `${fixture.sample.id} row ${row}`).toBeCloseTo(
            fixture.expected.times[row],
            7
          );
      });
      expect(Array.from(prepared.costBreaks)).toEqual(fixture.sample.parameters.costBreaks);
    }
  });

  it('maps both datasets with renamed columns and preserves source identity and time units', () => {
    for (const fixture of cases) {
      const inputs = fixture.inputs();
      const request = mapInputs(inputs, fixture.sample.columns, fixture.sample.parameters);
      expect(request.bindingId).toBe(binding.id);
      expect(request.inputDatasetIds).toEqual({
        network: 'network',
        facilities: 'facilities',
        demand: 'demand'
      });
      expect(request.parameters).toMatchObject(fixture.sample.parameters);
      const prepared = prepareCuratedDriveTimeBands(inputs, request.parameters);
      const result = publishCuratedDriveTimeBands(
        prepared,
        createReadback(prepared, fixture.expected)
      );
      const [demand, bands, isobands] = result.tables;
      expect(demand.columns['source_id']).toEqual(
        inputs['demand'].features.map(feature => feature.id)
      );
      expect(demand.columns['source_index']).toEqual(
        inputs['demand'].features.map((_, index) => index)
      );
      expect(demand.columns['cost_unit']).toEqual(
        inputs['demand'].features.map(() => fixture.sample.parameters.costUnit)
      );
      expect(bands.columns['demand_count']).toEqual(fixture.expected.counts);
      expect(bands.columns['demand_sum']).toEqual(fixture.expected.sums);
      expect(bands.columns['demand_mean']).toEqual(fixture.expected.means);
      for (const table of result.tables) {
        const specification = binding.outputs.find(output => output.tableId === table.id)!;
        expect(specification.identityColumns.every(column => column in table.columns)).toBe(true);
        expect(Object.values(table.columns).every(column => column.length === table.rowCount)).toBe(
          true
        );
        expect(Object.keys(table.columns)).toEqual(Object.keys(table.columnTypes));
        expect(table.rowCount).toBeLessThanOrEqual(specification.maximumRows);
        if (table.geometryColumn)
          expect(
            planKeplerResultPresentation(
              table,
              descriptor.outputs.find(output => output.tableId === table.id)!
            )
          ).toBeDefined();
      }
      const polygon = JSON.parse(String(isobands.columns['geometry'][0]));
      expect(polygon.type).toBe('Polygon');
      expect(polygon.coordinates[0][0]).toEqual(polygon.coordinates[0][3]);
    }
  });

  it('retains bends, asymmetric costs and one-way travel without a chord shortcut', () => {
    const prepared = prepareCuratedDriveTimeBands(
      createDriveTimeBandsSecondDataset(),
      DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters
    );
    expect(Array.from(prepared.topology.weights).sort((first, second) => first - second)).toEqual([
      4, 4, 4, 8, 8
    ]);
    expect(prepared.graph.nodes).toHaveLength(4);
    expect(getReference(prepared).demandTimes).toEqual([Infinity, 2, 4, 6, 10]);
    const result = publishCuratedDriveTimeBands(
      prepared,
      createReadback(prepared, DRIVE_TIME_BANDS_SECOND_EXPECTATIONS)
    );
    expect(result.tables[0].columns['status']).toEqual([
      'unreachable',
      'within-band',
      'within-band',
      'within-band',
      'beyond-bands'
    ]);
  });

  it('joins shared two-dimensional vertices when source roads also carry elevation ordinates', () => {
    const inputs = createDriveTimeBandsSecondDataset();
    inputs['network'].features[0].geometry!.coordinates = [
      [100, 200, 5],
      [110, 200, 10]
    ];
    inputs['network'].features[1].geometry!.coordinates = [
      [110, 200, 20],
      [110, 210, 30],
      [120, 210, 40]
    ];
    const prepared = prepareCuratedDriveTimeBands(
      inputs,
      DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters
    );
    expect(getReference(prepared).demandTimes).toEqual([Infinity, 2, 4, 6, 10]);
  });

  it('distinguishes inclusive boundary, beyond-band, unreachable and unsnapped results', () => {
    const prepared = prepareCuratedDriveTimeBands(
      createDriveTimeBandsSample(),
      DRIVE_TIME_BANDS_SAMPLE.parameters
    );
    const result = publishCuratedDriveTimeBands(
      prepared,
      createReadback(prepared, DRIVE_TIME_BANDS_EXPECTATIONS)
    );
    expect(result.tables[0].columns['band']).toEqual([0, 0, 1, 2, 2, null, null, null]);
    expect(result.tables[0].columns['travel_time']).toEqual([0, 1, 4, 6, 8, 10, null, null]);
    expect(result.tables[0].columns['status']).toEqual([
      'within-band',
      'within-band',
      'within-band',
      'within-band',
      'within-band',
      'beyond-bands',
      'unreachable',
      'unsnapped'
    ]);
  });

  it('reports empty bands and optional demand point counts without dropping zero-valued demand', () => {
    const inputs = createDriveTimeBandsSecondDataset();
    inputs['demand'].features[2].properties!['customers'] = 0;
    const prepared = prepareCuratedDriveTimeBands(inputs, {
      ...DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters,
      weightProperty: '',
      costBreaks: [0, 2, 4, 6]
    });
    expect(Array.from(prepared.demandValues)).toEqual([1, 1, 1, 1, 1]);
    const reference = getReference(prepared);
    const output = publishCuratedDriveTimeBands(prepared, {
      ...createReadback(prepared, DRIVE_TIME_BANDS_SECOND_EXPECTATIONS),
      times: reference.demandTimes,
      demandBands: [0xffffffff, 1, 2, 3, 0xffffffff],
      keys: [1, 2, 3],
      counts: [1, 1, 1],
      sums: [1, 1, 1],
      means: [1, 1, 1]
    });
    expect(output.tables[1].columns['demand_count']).toEqual([0, 1, 1, 1]);
    expect(output.tables[1].columns['demand_mean']).toEqual([null, 1, 1, 1]);
    const weighted = prepareCuratedDriveTimeBands(
      inputs,
      DRIVE_TIME_BANDS_SECOND_SAMPLE.parameters
    );
    expect(Array.from(weighted.demandValues)).toEqual([100, 2, 0, 4, 5]);
  });

  it('rejects missing inputs, geometry, capacities, wrong units and invalid costs before allocation', () => {
    const inputs = createDriveTimeBandsSample();
    for (const name of ['network', 'facilities', 'demand']) {
      const missing = {...inputs};
      delete missing[name];
      expect(() =>
        prepareCuratedDriveTimeBands(missing, DRIVE_TIME_BANDS_SAMPLE.parameters)
      ).toThrow(new RegExp(`curated-drive-time-bands.*${name}`));
    }
    const invalidParameters: Record<string, CatalogParameterValue>[] = [
      {forwardCostProperty: ''},
      {costUnit: 'meters'},
      {costBreaks: []},
      {costBreaks: [4, 1]},
      {costBreaks: [1, 1 + 1e-10]},
      {costBreaks: [Infinity]},
      {costBreaks: [-1]},
      {costBreaks: Array(9).fill(1)},
      {maximumSnapDistance: -1},
      {gridSize: 8.5},
      {bufferRadius: 0}
    ];
    for (const overrides of invalidParameters)
      expect(() =>
        prepareCuratedDriveTimeBands(inputs, {...DRIVE_TIME_BANDS_SAMPLE.parameters, ...overrides})
      ).toThrow(/curated-drive-time-bands/);
    inputs['network'].features[0].properties!['minutes'] = -1;
    expect(() => prepareCuratedDriveTimeBands(inputs, DRIVE_TIME_BANDS_SAMPLE.parameters)).toThrow(
      /forwardCostProperty/
    );
    inputs['network'].features[0].properties!['minutes'] = 1e30;
    expect(() => prepareCuratedDriveTimeBands(inputs, DRIVE_TIME_BANDS_SAMPLE.parameters)).toThrow(
      /travel time arc costs/
    );
    const excessive = createDriveTimeBandsSample();
    excessive['demand'].features = Array(DRIVE_TIME_BANDS_LIMITS.demand + 1).fill(
      excessive['demand'].features[0]
    );
    expect(() =>
      prepareCuratedDriveTimeBands(excessive, DRIVE_TIME_BANDS_SAMPLE.parameters)
    ).toThrow(/demand feature count/);
    const invalidGeometry = createDriveTimeBandsSample();
    invalidGeometry['facilities'] = invalidGeometry['network'];
    expect(() =>
      prepareCuratedDriveTimeBands(invalidGeometry, DRIVE_TIME_BANDS_SAMPLE.parameters)
    ).toThrow(/facilities (geometry|vertex count)/);
    const invalidValue = createDriveTimeBandsSample();
    invalidValue['demand'].features[0].properties!['residents'] = null;
    expect(() =>
      prepareCuratedDriveTimeBands(invalidValue, DRIVE_TIME_BANDS_SAMPLE.parameters)
    ).toThrow(/weightProperty/);
  });

  it('offers wrong-type travel-time fields and wrong input geometry as unmappable before Run', () => {
    const inputs = createDriveTimeBandsSample();
    inputs['network'].features.forEach(feature => {
      feature.properties!['minutes'] = 'ten';
    });
    const profiles = Object.entries(inputs).map(([name, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId: name,
        label: name,
        coordinateSystem: 'planar'
      })
    );
    const mapping = createKeplerCuratedAnalysisMapping(CURATED_DRIVE_TIME_BANDS_ANALYSIS, profiles);
    expect(mapping.ready).toBe(false);
    const network = mapping.slots.find(slot => slot.slot === 'network')!;
    expect(network.candidates.find(candidate => candidate.datasetId === 'network')?.fit).toBe(
      'missing-columns'
    );
    expect(network.candidates.find(candidate => candidate.datasetId === 'demand')?.fit).toBe(
      'wrong-kind'
    );
    expect(
      network.roles
        .find(role => role.role === 'forward-cost')
        ?.candidates.some(candidate => candidate.name === 'minutes')
    ).toBe(false);
  });

  it('bounds source vertices, derived segments and derived nodes independently', () => {
    const inputs = createDriveTimeBandsSample();
    const network: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {
          geometry: {
            type: 'LineString',
            coordinates: Array.from({length: DRIVE_TIME_BANDS_LIMITS.vertices + 1}, (_, index) => [
              index,
              0
            ])
          },
          properties: {minutes: 100, direction: 'both'}
        }
      ]
    };
    expect(() =>
      prepareCuratedDriveTimeBands({...inputs, network}, DRIVE_TIME_BANDS_SAMPLE.parameters)
    ).toThrow(/network vertex count/);
    network.features[0].geometry!.coordinates = Array.from(
      {length: DRIVE_TIME_BANDS_LIMITS.nodes + 1},
      (_, index) => [index, 0]
    );
    expect(() =>
      prepareCuratedDriveTimeBands({...inputs, network}, DRIVE_TIME_BANDS_SAMPLE.parameters)
    ).toThrow(/nodeCount/);
    network.features = Array.from({length: 2}, () => ({
      geometry: {
        type: 'LineString',
        coordinates: Array.from({length: 2501}, (_, index) => [index, 0])
      },
      properties: {minutes: 100, direction: 'both'}
    }));
    expect(() =>
      prepareCuratedDriveTimeBands({...inputs, network}, DRIVE_TIME_BANDS_SAMPLE.parameters)
    ).toThrow(/edgeCount|segmentCount/);
  });

  it('reports missing street costs with the existing network diagnostics', () => {
    const inputs = createDriveTimeBandsSample();
    inputs['network'].features[0].properties!['minutes'] = null;
    const prepared = prepareCuratedDriveTimeBands(inputs, DRIVE_TIME_BANDS_SAMPLE.parameters);
    expect(
      prepared.graph.diagnostics.some(diagnostic => diagnostic.code === 'invalid-cost-arc-removed')
    ).toBe(true);
  });

  it('uses a centered local frame and returns geographic map geometry in longitude/latitude', () => {
    const inputs = createDriveTimeBandsSample();
    for (const collection of Object.values(inputs))
      for (const feature of collection.features) {
        const geometry = feature.geometry!;
        const transform = (position: number[]) => [
          -87 + position[0] / 100000,
          42 + position[1] / 100000
        ];
        geometry.coordinates =
          geometry.type === 'Point'
            ? transform(geometry.coordinates as number[])
            : (geometry.coordinates as number[][]).map(transform);
      }
    const prepared = prepareCuratedDriveTimeBands(inputs, {
      ...DRIVE_TIME_BANDS_SAMPLE.parameters,
      coordinateSystem: 'EPSG:4326',
      maximumSnapDistance: 100
    });
    expect(Math.max(...prepared.nodePositions)).toBeLessThan(100);
    const inverse = prepared.frame.inverse(prepared.nodePositions[0], prepared.nodePositions[1]);
    expect(inverse[0]).toBeCloseTo(-87, 7);
    expect(inverse[1]).toBeCloseTo(42, 7);
    expect(prepared.graph.coordinateSystem).toBe('EPSG:4326');
  });

  it('refuses publication on nonconvergence, overflow, malformed readback or inconsistent bands', () => {
    const prepared = prepareCuratedDriveTimeBands(
      createDriveTimeBandsSample(),
      DRIVE_TIME_BANDS_SAMPLE.parameters
    );
    const readback = createReadback(prepared, DRIVE_TIME_BANDS_EXPECTATIONS);
    for (const overrides of [
      {converged: 0},
      {bandOverflow: 1},
      {triangleOverflow: 1},
      {requiredBandCount: 4},
      {times: []},
      {times: [NaN, ...Array.from(readback.times).slice(1)]},
      {demandBands: Array(8).fill(0)},
      {triangleBands: [10]},
      {keys: [0, 0, 1]},
      {sums: [NaN, 3, 9]},
      {counts: [0, 1, 2]}
    ])
      expect(() => publishCuratedDriveTimeBands(prepared, {...readback, ...overrides})).toThrow(
        /curated-drive-time-bands/
      );
  });

  it('constructs the actual recipe graph and cleans up on submission failure', async () => {
    const device = createNullWebGPUDevice();
    const disposed = vi.spyOn(CatalogGpuSession.prototype, 'dispose');
    const submitted = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(function (
      this: CatalogGpuSession
    ) {
      const sources = getGPUCommandNodeKernelSources(
        (this.graph as unknown as {nodes: unknown[]}).nodes
      );
      for (const name of ['facility-starts', 'demand-times', 'demand-bands', 'isobands'])
        expect(
          sources.some(source => source.id.includes(name)),
          name
        ).toBe(true);
      throw new Error('drive-time recipe reached submission');
    });
    try {
      await expect(
        binding.runGpu!({device}, createDriveTimeBandsSample(), DRIVE_TIME_BANDS_SAMPLE.parameters)
      ).rejects.toThrow('drive-time recipe reached submission');
      expect(submitted).toHaveBeenCalledOnce();
      expect(disposed).toHaveBeenCalledOnce();
    } finally {
      submitted.mockRestore();
      disposed.mockRestore();
      device.destroy();
    }
  });

  it('has complete descriptors, matching record and a parser-valid reusable template', () => {
    expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
    const originalLookup = operationDescriptors.getKeplerOperationDescriptor;
    const lookup = vi
      .spyOn(operationDescriptors, 'getKeplerOperationDescriptor')
      .mockImplementation(identifier =>
        identifier === binding.id ? descriptor : originalLookup(identifier)
      );
    try {
      expect(validateKeplerCuratedAnalysis(CURATED_DRIVE_TIME_BANDS_ANALYSIS)).toEqual([]);
      const template = createDriveTimeBandsTemplate();
      const operation = makeWorkflowOperationFromCatalogBinding(binding);
      const registry = createWorkflowOperationRegistry([
        ...getWorkflowOperationRegistry()
          .list()
          .filter(candidate => candidate.id !== binding.id),
        operation
      ]);
      const validation = validateWorkflowAnalysisTemplate(template, {registry});
      expect(validation.issues, JSON.stringify(validation.issues)).toEqual([]);
      expect(validation.executable).toBe(true);
      expect(
        parseWorkflowAnalysisTemplate(JSON.stringify(serializeWorkflowAnalysisTemplate(template)))
      ).toEqual(template);
      expect(
        validateWorkflowAnalysisTemplate({...template, slots: template.slots.slice(1)}, {registry})
          .executable
      ).toBe(false);
      expect(template.exposed).toHaveLength(4);
      expect(template.exposed.some(setting => setting.target.parameter === 'costBreaks')).toBe(
        true
      );
      expect(template.sample?.license).toBe('MIT');
      expect(
        readFileSync(new URL('../../../../THIRD-PARTY-DATA.md', import.meta.url), 'utf8')
      ).toContain('## Curated drive-time bands synthetic samples');
    } finally {
      lookup.mockRestore();
    }
  });
});

it('drive-time raw geometry validation rejects malformed vertices before shortcut preparation', () => {
  for (const malformed of [null, ['invalid', 5], [4], [4, NaN], [4, Infinity], 'vertex']) {
    for (const type of ['LineString', 'MultiLineString']) {
      const inputs = createDriveTimeBandsSample();
      const coordinates = [[0, 0], malformed, [8, 0]];
      inputs['network'].features[0].geometry = {
        type,
        coordinates: type === 'MultiLineString' ? [coordinates] : coordinates
      };
      expect(
        () => prepareCuratedDriveTimeBands(inputs, DRIVE_TIME_BANDS_SAMPLE.parameters),
        `${type}: ${JSON.stringify(malformed)}`
      ).toThrow(/curated-drive-time-bands network coordinates/);
    }
  }
});

it('drive-time radius fits the splatter window at grid size 64 and extreme aspect ratios', () => {
  for (const gridSize of [8, 32, 64])
    for (const radius of [0.001, 5, 1000000]) {
      const prepared = prepareCuratedDriveTimeBands(createDriveTimeBandsSample(), {
        ...DRIVE_TIME_BANDS_SAMPLE.parameters,
        gridSize,
        bufferRadius: radius
      });
      const [minimumHorizontal, minimumVertical, maximumHorizontal, maximumVertical] =
        prepared.extent;
      const cellWidth = (maximumHorizontal - minimumHorizontal) / gridSize;
      const cellHeight = (maximumVertical - minimumVertical) / gridSize;
      const effectiveRadius = Math.max(radius, 0.5 * Math.hypot(cellWidth, cellHeight));
      expect(
        Math.ceil(effectiveRadius / cellWidth),
        `horizontal reach ${gridSize}/${radius}`
      ).toBeLessThanOrEqual(DRIVE_TIME_BANDS_LIMITS.bufferPixels);
      expect(
        Math.ceil(effectiveRadius / cellHeight),
        `vertical reach ${gridSize}/${radius}`
      ).toBeLessThanOrEqual(DRIVE_TIME_BANDS_LIMITS.bufferPixels);
    }
});

it('drive-time template remapping resolves geographic distance units from the new profiles', () => {
  const template = createDriveTimeBandsTemplate();
  const step = template.document.steps.find(candidate => candidate.id === 'analysis')!;
  expect(step.parameters['coordinateSystem']).toBeUndefined();
  const inputs = createDriveTimeBandsSample();
  for (const collection of Object.values(inputs))
    for (const feature of collection.features) {
      const geometry = feature.geometry!;
      const transform = (position: number[]) => [
        -87 + position[0] / 100000,
        42 + position[1] / 100000
      ];
      geometry.coordinates =
        geometry.type === 'Point'
          ? transform(geometry.coordinates as number[])
          : (geometry.coordinates as number[][]).map(transform);
    }
  const profiles = Object.entries(inputs).map(([name, collection]) =>
    profileKeplerFeatureCollection(collection, {
      datasetId: name,
      label: name,
      coordinateSystem: 'EPSG:4326',
      declarations: {coordinateModel: 'geographic', rowSemantics: 'objects'}
    })
  );
  const overrides = Object.fromEntries(
    Object.entries(step.parameters).map(([name, parameter]) => [
      name,
      parameter.from === 'static'
        ? parameter.value
        : template.document.inputs.find(input => input.id === parameter.inputId)!.value
    ])
  );
  let mapping = createKeplerCuratedAnalysisMapping(CURATED_DRIVE_TIME_BANDS_ANALYSIS, profiles, {
    parameters: overrides
  });
  for (const slot of template.slots)
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: slot.name, datasetId: slot.name},
      CURATED_DRIVE_TIME_BANDS_ANALYSIS,
      profiles
    );
  for (const [key, column] of Object.entries(DRIVE_TIME_BANDS_SAMPLE.columns)) {
    const [slot, role] = key.split('.');
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot, role, columns: [column]},
      CURATED_DRIVE_TIME_BANDS_ANALYSIS,
      profiles
    );
  }
  expect(mapping.ready, JSON.stringify(mapping.problems)).toBe(true);
  const request = buildKeplerCuratedAnalysisRequest(CURATED_DRIVE_TIME_BANDS_ANALYSIS, mapping);
  const profilesByInput = Object.fromEntries(profiles.map(profile => [profile.datasetId, profile]));
  const resolved = resolveKeplerOperationParameters(binding, descriptor, profilesByInput, {
    overrides: request.parameters
  });
  expect(resolved.parameters['coordinateSystem']).toBe('EPSG:4326');
  const prepared = prepareCuratedDriveTimeBands(inputs, resolved.parameters);
  expect(prepared.graph.coordinateSystem).toBe('EPSG:4326');
  // The 88-coordinate street span is about 73 ground metres here; treating degrees as planar
  // would instead leave an extent below 0.001 coordinate units before the map buffer.
  expect(Math.max(...prepared.nodePositions) - Math.min(...prepared.nodePositions)).toBeGreaterThan(
    50
  );
});
