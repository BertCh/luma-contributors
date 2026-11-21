// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareAnalysisResultPublication} from '../../../../examples/deck/kepler-compute/src/analysis-result-publication';
import {describe, expect, it, vi} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {
  computeTerrainFlowOnCPU,
  computeTerrainFlowDistanceOnCPU,
  computeTerrainStreamOrderOnCPU,
  computeTerrainWatershedsOnCPU,
  getGPUTerrainFlowParameterValues
} from '@luma-contributors/analysis/gpu-terrain';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import {
  CURATED_DRAINAGE_NETWORK_BINDING,
  MAXIMUM_DRAINAGE_CELLS,
  prepareCuratedDrainageNetwork,
  publishCuratedDrainageNetwork,
  type CuratedDrainageReadback
} from '../../src/curated/bindings/curated-drainage-network-binding';
import {
  createDrainageNetworkSample,
  createDrainageNetworkSecondDataset,
  CURATED_DRAINAGE_NETWORK_ANALYSIS,
  DRAINAGE_NETWORK_EXPECTATIONS,
  DRAINAGE_NETWORK_SECOND_EXPECTATIONS,
  DRAINAGE_NETWORK_SAMPLE
} from '../../src/curated/samples/drainage-network';
import {CURATED_DRAINAGE_NETWORK_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-drainage-network';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {makeWorkflowOperationFromCatalogBinding} from '../../src/workflows/workflow-operations';
import {
  compareDrainageColumn,
  createDrainageNoDataCollisionRaster
} from './curated-drainage-network-test-support';

const binding = CURATED_DRAINAGE_NETWORK_BINDING;

/** Existing recipe fixture, with all D8 outputs worked out by hand. */
function makeReadback(): CuratedDrainageReadback {
  return {
    ...DRAINAGE_NETWORK_EXPECTATIONS,
    streams: Array(9).fill(1),
    validity: Array(9).fill(1),
    terminalCells: Array(9).fill(8),
    sourceCells: [0, 1, 2, 7],
    targetCells: [7, 7, 8, 8],
    vertexOffsets: [0, 4, 7, 10, 12],
    positions: [
      0.5, 0.5, 0.5, 1.5, 0.5, 2.5, 1.5, 2.5, 1.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 0.5, 2.5, 1.5, 2.5,
      2.5, 1.5, 2.5, 2.5, 2.5
    ],
    lengths: [3, 2, 2, 1],
    vertexCount: 12,
    catchmentIds: [8],
    cellCounts: [9],
    meanDownstreamDistance: [2],
    summaryCount: 1,
    stages: [
      {stage: 'depression-fill', converged: 1},
      {stage: 'flat-routing', converged: 1},
      {stage: 'accumulation', converged: 1},
      {stage: 'stream-order', converged: 1},
      {stage: 'catchments', converged: 1},
      {stage: 'downstream-distance', unresolvedCount: 0},
      {stage: 'stream-network', overflow: 0, unresolvedCount: 0},
      {stage: 'catchment-summary', overflow: 0}
    ]
  };
}

describe('curated drainage preparation', () => {
  it('classifies Float64 nodata before rounding and retains the other cell as its own catchment', () => {
    const elevation = createDrainageNoDataCollisionRaster(
      new Float64Array([16777216, 16777217]),
      16777217
    );
    const prepared = prepareCuratedDrainageNetwork({elevation}, {});
    expect(Array.from(prepared.elevationBand.validity)).toEqual([1, 0]);
    expect(Array.from(prepared.elevationBand.values)).toEqual([16777216, 16777216]);
    const flow = computeTerrainFlowOnCPU(
      {
        elevation: {id: elevation.id, ...prepared.elevationBand},
        parameters: getGPUTerrainFlowParameterValues({streamThreshold: 1}, prepared.grid)
      },
      {grid: prepared.grid, fillDepressions: true, resolveFlats: true}
    );
    expect(flow.accumulation[0]).toBe(1);
    expect(flow.flowDirections[0]).toBe(0);
    expect(flow.flowDirections[1]).toBe(0xffffffff);
    expect(
      Array.from(computeTerrainWatershedsOnCPU(flow.flowDirections, {grid: prepared.grid}).labels)
    ).toEqual([0, 0xffffffff]);
    expect(Array.from(elevation.band.validity)).toEqual([1, 1]);
    expect(Array.from(elevation.band.values)).toEqual([16777216, 16777217]);
  });

  it('retains signed and unsigned integer source values and exactly widens narrower integer bands', () => {
    const cases = [
      {values: new Int32Array([-16777216, -16777217]), sentinel: -16777217, format: 'sint32'},
      {values: new Uint32Array([4294967294, 4294967295]), sentinel: 4294967295, format: 'uint32'},
      {values: new Int16Array([-123, -124]), sentinel: -124, format: 'sint32'},
      {values: new Uint16Array([123, 124]), sentinel: 124, format: 'uint32'},
      {values: new Int8Array([-12, -13]), sentinel: -13, format: 'sint32'},
      {values: new Uint8Array([12, 13]), sentinel: 13, format: 'uint32'}
    ];
    for (const {values, sentinel, format} of cases) {
      const elevation = createDrainageNoDataCollisionRaster(values, sentinel);
      const prepared = prepareCuratedDrainageNetwork({elevation}, {});
      expect(prepared.elevationBand.format).toBe(format);
      expect(Array.from(prepared.elevationBand.values)).toEqual(Array.from(values));
      expect(Array.from(prepared.elevationBand.validity)).toEqual([1, 0]);
      if (values instanceof Int32Array || values instanceof Uint32Array)
        expect(prepared.elevationBand.values).toBe(values);
    }
    const elevation = createDrainageNoDataCollisionRaster(
      new Float32Array([16777216, 0]),
      16777217
    );
    expect(
      Array.from(prepareCuratedDrainageNetwork({elevation}, {}).elevationBand.validity)
    ).toEqual([1, 1]);
  });

  it('checks both hand-computed fixture expectations against the public terrain CPU twins', () => {
    for (const [raster, expected] of [
      [createDrainageNetworkSample().elevation, DRAINAGE_NETWORK_EXPECTATIONS],
      [createDrainageNetworkSecondDataset().survey, DRAINAGE_NETWORK_SECOND_EXPECTATIONS]
    ] as const) {
      const prepared = prepareCuratedDrainageNetwork({elevation: raster}, {streamThreshold: 1});
      const {grid} = prepared;
      const flow = computeTerrainFlowOnCPU(
        {
          elevation: {
            id: raster.id,
            format: 'float32',
            values: Float32Array.from(raster.band.values)
          },
          parameters: getGPUTerrainFlowParameterValues({streamThreshold: 1}, grid)
        },
        {grid, fillDepressions: true, resolveFlats: true}
      );
      expect(Array.from(flow.flowDirections)).toEqual(expected.flowDirections);
      expect(Array.from(flow.accumulation)).toEqual(expected.accumulation);
      expect(
        Array.from(
          computeTerrainStreamOrderOnCPU(flow.flowDirections, flow.streams, {grid}).streamOrder
        )
      ).toEqual(expected.streamOrder);
      expect(Array.from(computeTerrainWatershedsOnCPU(flow.flowDirections, {grid}).labels)).toEqual(
        expected.catchments
      );
      const distance = computeTerrainFlowDistanceOnCPU(flow.flowDirections, {grid});
      expect(Array.from(distance.terminalCells)).toEqual(expected.catchments);
      expected.distance.forEach((value, cell) =>
        expect(Math.abs(distance.distance[cell] - value)).toBeLessThan(1e-5)
      );
    }
  });

  it('uses real bounded band/grid data and derives path and allocation bounds from cell count', () => {
    const inputs = createDrainageNetworkSample();
    const prepared = prepareCuratedDrainageNetwork(inputs, normalizeCatalogParameters(binding, {}));
    expect(prepared.raster).toBe(inputs.elevation);
    expect(prepared.cellCount).toBe(9);
    expect(prepared.maximumSteps).toBe(9);
    expect(prepared.reachCapacity).toBe(9);
    expect(prepared.vertexCapacity).toBe(18);
    expect(prepared.fillDepressions).toBe(true);
    expect(prepared.resolveFlats).toBe(true);
    expect(DRAINAGE_NETWORK_SAMPLE).toMatchObject({
      license: 'MIT',
      attribution: 'vis.gl contributors'
    });
  });

  it('keeps input ownership, identity, band validity and float64 translation on the second grid', () => {
    const {survey} = createDrainageNetworkSecondDataset();
    const prepared = prepareCuratedDrainageNetwork({elevation: survey}, {});
    expect(prepared.raster.band.values).toBe(survey.band.values);
    expect(prepared.raster.band.validity).toBe(survey.band.validity);
    expect(prepared.grid.affine).toEqual([2, 0, 0, 0, -3, 0]);
    expect(survey.grid.affine).toEqual([2, 0, 1e9, 0, -3, 4.5e9]);
    expect(prepared.raster.band.index).toBe(2);
  });

  it('rejects wrong kinds, geographic/Mercator units, rotation, excessive cells and malformed bands', () => {
    const {elevation} = createDrainageNetworkSample();
    expect(() => prepareCuratedDrainageNetwork({}, {})).toThrow(
      /curated-drainage-network.*elevation/
    );
    expect(() =>
      prepareCuratedDrainageNetwork({elevation: {type: 'FeatureCollection', features: []}}, {})
    ).toThrow(/elevation must be a raster/);
    for (const [crs, cellSizeMode] of [
      ['EPSG:4326', 'geographic'],
      ['EPSG:3857', 'web-mercator']
    ] as const) {
      expect(() =>
        prepareCuratedDrainageNetwork({elevation: {...elevation, crs, cellSizeMode}}, {})
      ).toThrow(/crs must be planar/);
    }
    expect(() =>
      prepareCuratedDrainageNetwork(
        {elevation: {...elevation, grid: {...elevation.grid, affine: [1, 0.1, 0, 0, 1, 0]}}},
        {}
      )
    ).toThrow(/axis aligned/);
    expect(() =>
      prepareCuratedDrainageNetwork(
        {
          elevation: {
            ...elevation,
            grid: {...elevation.grid, width: MAXIMUM_DRAINAGE_CELLS, height: 2}
          }
        },
        {}
      )
    ).toThrow(/grid cell count/);
    expect(() =>
      prepareCuratedDrainageNetwork(
        {elevation: {...elevation, band: {...elevation.band, validity: new Uint8Array(8)}}},
        {}
      )
    ).toThrow(/validity length/);
  });

  it('checks named parameter limits before allocating a GPU graph', () => {
    const invalidParameters: Record<string, CatalogParameterValue>[] = [
      {streamThreshold: NaN},
      {streamThreshold: 0},
      {maximumSteps: -1},
      {maximumFillIterations: 1025},
      {maximumFlatIterations: 1.5},
      {reachCapacity: -1},
      {vertexCapacity: 2 * MAXIMUM_DRAINAGE_CELLS + 1},
      {fillDepressions: 'yes'}
    ];
    for (const values of invalidParameters) {
      expect(() => prepareCuratedDrainageNetwork(createDrainageNetworkSample(), values)).toThrow(
        /curated-drainage-network/
      );
    }
  });
});

describe('curated drainage mapping and publication', () => {
  it('uploads source-domain validity and native integer formats into the real recipe', async () => {
    const device = createNullWebGPUDevice();
    const uploads = vi.spyOn(CatalogGpuSession.prototype, 'input');
    const submitted = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {
      throw new Error('upload probe reached submission');
    });
    try {
      for (const {values, sentinel, format} of [
        {values: new Float64Array([16777216, 16777217]), sentinel: 16777217, format: 'float32'},
        {values: new Int32Array([-16777216, -16777217]), sentinel: -16777217, format: 'sint32'},
        {values: new Uint32Array([4294967294, 4294967295]), sentinel: 4294967295, format: 'uint32'}
      ]) {
        uploads.mockClear();
        const elevation = createDrainageNoDataCollisionRaster(values, sentinel);
        await expect(binding.runDataGpu!({device}, {elevation}, {})).rejects.toThrow(
          'upload probe reached submission'
        );
        const validityUpload = uploads.mock.calls.find(([name]) => name === 'validity')!;
        const elevationUpload = uploads.mock.calls.find(([name]) => name === 'elevation')!;
        expect(Array.from(validityUpload[1])).toEqual([1, 0]);
        expect(elevationUpload[2]).toBe(format);
        if (format !== 'float32')
          expect(Array.from(elevationUpload[1])).toEqual(Array.from(values));
      }
    } finally {
      uploads.mockRestore();
      submitted.mockRestore();
      device.destroy();
    }
  });

  it('rejects NaN and infinities in the exact comparator used by GPU parity tests', () => {
    const prepared = prepareCuratedDrainageNetwork(createDrainageNetworkSample(), {});
    const output = publishCuratedDrainageNetwork(prepared, makeReadback());
    for (const table of output.tables.filter(table => table.geometryColumn)) {
      const publication = prepareAnalysisResultPublication(table);
      expect(table.coordinateSystem).toBe('planar');
      expect(publication.analysisResult.destination).toBe('report');
      expect(publication.analysisResult.mapMessage).toMatch(/no declared projection/);
    }

    const expected = DRAINAGE_NETWORK_EXPECTATIONS.accumulation;
    expect(() => compareDrainageColumn(output, 'accumulation', expected)).not.toThrow();
    for (const value of [NaN, Infinity, -Infinity]) {
      const tables = output.tables.map(table =>
        table.id === 'catchment-cells'
          ? {...table, columns: {...table.columns, accumulation: Array(9).fill(value)}}
          : table
      );
      expect(() => compareDrainageColumn({...output, tables}, 'accumulation', expected)).toThrow(
        /accumulation mismatch/
      );
      expect(() => compareDrainageColumn({...output, tables}, 'accumulation', expected)).toThrow(
        String(value)
      );
    }
  });

  it('constructs the real recipe and publication nodes, then releases its session when submission fails', async () => {
    const device = createNullWebGPUDevice();
    const destroyed = vi.spyOn(CatalogGpuSession.prototype, 'dispose');
    const submitted = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(function (
      this: CatalogGpuSession
    ) {
      const nodes = (this.graph as unknown as {nodes: unknown[]}).nodes;
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(kernels.some(kernel => kernel.id.startsWith('drainage-network-recipe-flow'))).toBe(
        true
      );
      expect(kernels.some(kernel => kernel.id === 'publish-positions')).toBe(true);
      expect(kernels.some(kernel => kernel.id === 'publish-stream-network-overflow')).toBe(true);
      for (const stage of ['downstream-distance', 'stream-network']) {
        expect(kernels.some(kernel => kernel.id === `publish-${stage}-unresolvedCount`)).toBe(true);
        expect(kernels.some(kernel => kernel.id === `publish-${stage}-uncertainCount`)).toBe(false);
      }
      // This is a construction/cleanup probe, not GPU numerical qualification.
      throw new Error('drainage construction reached submission');
    });
    try {
      await expect(
        binding.runDataGpu!({device}, createDrainageNetworkSample(), {})
      ).rejects.toThrow('drainage construction reached submission');
      expect(submitted).toHaveBeenCalledOnce();
      expect(destroyed).toHaveBeenCalledOnce();
    } finally {
      submitted.mockRestore();
      destroyed.mockRestore();
      device.destroy();
    }
  });

  it('maps both independently identified surface datasets through the existing mapping model', () => {
    for (const raster of [
      createDrainageNetworkSample().elevation,
      createDrainageNetworkSecondDataset().survey
    ]) {
      const profile: KeplerDatasetProfile = {
        datasetId: raster.id,
        label: raster.band.name!,
        kind: 'surface',
        rowCount: raster.grid.width * raster.grid.height,
        geometryColumns: [],
        columns: [],
        coordinates: {model: 'planar', source: 'declared'}
      };
      const mapping = createKeplerCuratedAnalysisMapping(CURATED_DRAINAGE_NETWORK_ANALYSIS, [
        profile
      ]);
      expect(mapping.ready).toBe(true);
      expect(
        buildKeplerCuratedAnalysisRequest(CURATED_DRAINAGE_NETWORK_ANALYSIS, mapping)
      ).toMatchObject({
        bindingId: binding.id,
        inputDatasetIds: {elevation: raster.id}
      });
    }
  });

  it('declares typed workflow ports and complete descriptor policies without pretending to have a CPU path', async () => {
    expect(validateKeplerOperationDescriptor(CURATED_DRAINAGE_NETWORK_DESCRIPTOR, binding)).toEqual(
      []
    );
    expect(binding.engine).toBe('gpu');
    expect(binding.runData).toBeUndefined();
    expect(binding.run).toBeUndefined();
    const operation = makeWorkflowOperationFromCatalogBinding(binding);
    expect(operation.inputs[0]).toMatchObject({name: 'elevation', accepts: ['raster']});
    expect(operation.outputs.map(output => output.kind)).toEqual(['line', 'polygon', 'table']);
    await expect(
      runCatalogOperationAsync(binding, {}, createDrainageNetworkSample())
    ).rejects.toThrow(/GPU device required/);
  });

  it('publishes fixture identities, connected lines, cell footprints and catchment summaries with presentation plans', () => {
    const prepared = prepareCuratedDrainageNetwork(createDrainageNetworkSample(), {
      streamThreshold: 1
    });
    const output = publishCuratedDrainageNetwork(prepared, makeReadback());
    const [streams, cells, summary] = output.tables;
    expect(streams.rowCount).toBe(4);
    expect(streams.columns['source_cell']).toEqual([0, 1, 2, 7]);
    expect(streams.columns['target_cell']).toEqual([7, 7, 8, 8]);
    expect(JSON.parse(String(streams.columns['geometry'][0]))).toEqual({
      type: 'LineString',
      coordinates: [
        [0.5, 0.5],
        [0.5, 1.5],
        [0.5, 2.5],
        [1.5, 2.5]
      ]
    });
    expect(cells.columns['cell_index']).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(cells.columns['source_id']).toEqual(Array(9).fill(DRAINAGE_NETWORK_SAMPLE.id));
    expect(cells.columns['catchment_id']).toEqual(DRAINAGE_NETWORK_EXPECTATIONS.catchments);
    expect(cells.columns['downstream_distance'][4]).toBe(2);
    expect(summary.columns['cell_count']).toEqual([9]);
    expect(summary.columns['mean_downstream_distance']).toEqual([2]);
    expect(output.summary).toMatchObject({
      converged: 1,
      overflow: 0,
      unresolvedCount: 0,
      accumulationUnit: 'cells',
      distanceUnit: 'coordinate-unit'
    });
    for (const table of output.tables) {
      const specification = binding.outputs.find(candidate => candidate.tableId === table.id)!;
      expect(Object.keys(table.columnTypes)).toEqual(Object.keys(table.columns));
      expect(Object.values(table.columns).every(column => column.length === table.rowCount)).toBe(
        true
      );
      expect(specification.identityColumns.every(column => column in table.columns)).toBe(true);
      const descriptor = CURATED_DRAINAGE_NETWORK_DESCRIPTOR.outputs.find(
        candidate => candidate.tableId === table.id
      )!;
      if (specification.geometry !== 'none')
        expect(planKeplerResultPresentation(table, descriptor)).toBeDefined();
    }
  });

  it('preserves no-data rows as explicit nulls and distinguishes point-grid footprints', () => {
    const {elevation} = createDrainageNetworkSample();
    const prepared = prepareCuratedDrainageNetwork(
      {elevation: {...elevation, grid: {...elevation.grid, pixelInterpretation: 'point'}}},
      {}
    );
    const fixture = makeReadback();
    const output = publishCuratedDrainageNetwork(prepared, {
      ...fixture,
      flowDirections: [0xffffffff, ...Array.from(fixture.flowDirections).slice(1)],
      catchments: [0xffffffff, ...Array.from(fixture.catchments).slice(1)],
      validity: [0, ...Array(8).fill(1)],
      terminalCells: [0xffffffff, ...Array(8).fill(8)]
    });
    const cells = output.tables[1];
    expect(cells.columns['status'][0]).toBe('no-data');
    for (const column of [
      'catchment_id',
      'flow_direction',
      'accumulation',
      'downstream_distance',
      'terminal_cell'
    ])
      expect(cells.columns[column][0]).toBeNull();
    expect(JSON.parse(String(cells.columns['geometry'][0])).coordinates[0][0]).toEqual([
      -0.5, -0.5
    ]);
    expect(output.diagnostics.find(diagnostic => diagnostic.code === 'no-data')?.count).toBe(1);
  });

  it('rejects every nonconverged, overflowed or unresolved stage with readable status before publication', () => {
    const prepared = prepareCuratedDrainageNetwork(createDrainageNetworkSample(), {});
    for (const status of [
      {stage: 'depression-fill', converged: 0},
      {stage: 'flat-routing', converged: 0},
      {stage: 'accumulation', converged: 0},
      {stage: 'stream-order', converged: 0},
      {stage: 'catchments', converged: 0},
      {stage: 'stream-network', overflow: 1},
      {stage: 'catchment-summary', overflow: 1},
      {stage: 'downstream-distance', unresolvedCount: 3},
      {stage: 'stream-network', unresolvedCount: 2}
    ]) {
      expect(() =>
        publishCuratedDrainageNetwork(prepared, {...makeReadback(), stages: [status]})
      ).toThrow(new RegExp(status.stage));
    }
  });

  it.each([
    'downstream-distance',
    'stream-network'
  ])('gates canonical unresolved-only and overflow-only %s stages and recovers without adding aliases', stage => {
    const prepared = prepareCuratedDrainageNetwork(createDrainageNetworkSample(), {});
    const status = {stage, unresolvedCount: 0, overflow: 0};
    const readback = {...makeReadback(), stages: [status]};
    const publish = () => publishCuratedDrainageNetwork(prepared, readback);
    expect(publish().summary['unresolvedCount']).toBe(0);
    status.unresolvedCount = 3;
    expect(publish).toThrow(/unresolvedCount.*3/);
    status.unresolvedCount = 0;
    status.overflow = 1;
    expect(publish).toThrow(/overflow.*1/);
    status.overflow = 0;
    const recovered = publish();
    expect(recovered.summary['unresolvedCount']).toBe(0);
    expect(recovered.summary['recipeStatus']).toContain('"unresolvedCount":0');
  });

  it('preserves omitted unresolved evidence and publishes only supplied zeros (B6)', () => {
    const prepared = prepareCuratedDrainageNetwork(createDrainageNetworkSample(), {});
    for (const stage of ['downstream-distance', 'stream-network']) {
      const publish = (evidence: {unresolvedCount?: number} | {uncertainCount: number}) =>
        publishCuratedDrainageNetwork(prepared, {
          ...makeReadback(),
          stages: [{stage, overflow: 0, ...evidence}]
        }).summary;
      expect(publish({})['recipeStatus']).not.toContain('unresolvedCount');
      expect(publish({})).not.toHaveProperty('unresolvedCount');
      expect(() => publish({uncertainCount: 0})).toThrow(
        /uncertainCount was removed; use unresolvedCount/
      );
      expect(publish({unresolvedCount: 0})['recipeStatus']).toContain('"unresolvedCount":0');
    }
  });
});

it('retains planar CRS for both geometry outputs even when coordinates resemble longitude and latitude', () => {
  const {elevation} = createDrainageNetworkSample();
  for (const origin of [0, 1e9]) {
    const raster = {
      ...elevation,
      grid: {...elevation.grid, affine: [1, 0, origin, 0, 1, origin * 4.5] as const}
    };
    const output = publishCuratedDrainageNetwork(
      prepareCuratedDrainageNetwork({elevation: raster}, {}),
      makeReadback()
    );
    expect(output.summary['crs']).toBe('planar');
    for (const table of output.tables.filter(table => table.geometryColumn)) {
      expect(table.coordinateSystem).toBe('planar');
      expect(table.columns['geometry'].length).toBe(table.rowCount);
    }
    const polygon = JSON.parse(output.tables[1].columns['geometry'][0] as string);
    expect(polygon.coordinates[0][0]).toEqual([origin, origin * 4.5]);
  }
});
