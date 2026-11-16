// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  addFleetDwellRecipe,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {
  CURATED_FLEET_DWELL_BINDING,
  prepareFleetDwell,
  makeFleetDwellOutput,
  getFleetDwellContributorNames,
  type FleetDwellReadback
} from '../../src/curated/bindings/index';
import {createFleetDwellSample, FLEET_DWELL_SAMPLE_LICENSE} from '../../src/curated/samples/index';
import {getCatalogBinding} from '../../src/analysis/catalog-bindings';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {CURATED_FLEET_DWELL_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-fleet-dwell';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {
  validateKeplerCuratedAnalysis,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {createCuratedWorkflowAnalysisTemplate} from '../../src/workflows/workflow-analysis-template';
import {
  parseWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template-parser';
import {planKeplerResultPresentation} from '../../src/presentation/result-presentation';
import {
  computeTrajectoryOracle,
  createDoubleOracleTime
} from '../../../analysis/test/gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-oracle';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {
  isTrajectoryCatalogOperation,
  TRAJECTORY_POINT_FIELD,
  makeCatalogTrajectoryNativeSource
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-trajectory';
import {
  makeCatalogPointColumns,
  getCatalogSampleInputs
} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {readCatalogNativeInput} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-native-input';
import * as arrow from 'apache-arrow';

const binding = CURATED_FLEET_DWELL_BINDING;
const analysis = getKeplerCuratedAnalysis('fleet-dwell')!;

describe('curated fleet dwell', () => {
  it('reports named stages from the actual recipe while excluding its stop-mask glue adapter', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device, {id: 'fleet-dwell-dispatch-metadata'});
    let serial = 0;
    const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `fleet-${serial++}`, format, length);
    try {
      const recipe = addFleetDwellRecipe(graph, {
        trajectories: {
          kind: 'trajectories',
          positions: view('float32x2', 3),
          timestamps: view('uint32x2', 3),
          trackOffsets: view('uint32', 2)
        },
        parameters: view('float32', GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH),
        stopCapacity: 3,
        zones: {
          kind: 'polygons',
          positions: view('float32x2', 4),
          featureOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          ringOffsets: view('uint32', 2),
          candidateCapacity: 3
        }
      });
      expect(recipe.contributors.map(contributor => contributor.constructor.name)).toContain(
        'Object'
      );
      expect(getFleetDwellContributorNames(recipe.contributors)).toEqual(
        binding.gpuContributorNames
      );
      expect(getFleetDwellContributorNames(recipe.contributors)).toHaveLength(3);
    } finally {
      device.destroy();
    }
  });

  it('routes fleet track inputs by descriptor metadata and preserves native Date timestamps and retained sources', async () => {
    expect(isTrajectoryCatalogOperation(binding.id)).toBe(true);
    expect(isTrajectoryCatalogOperation(binding.id, 'tracks')).toBe(true);
    expect(isTrajectoryCatalogOperation(binding.id, 'zones')).toBe(false);
    expect(isTrajectoryCatalogOperation('trajectory-does-not-exist')).toBe(false);
    const parameters = {
      coordinateSystem: 'planar',
      timestampEncoding: 'iso8601',
      longitudeProperty: 'east',
      latitudeProperty: 'north'
    };
    const rows = [0, 10000, 20000].map(timestamp => ['van', new Date(timestamp), 0, 0]);
    const source = {
      id: 'native',
      label: 'Stationary native points',
      fields: ['track_id', 'timestamp', 'east', 'north'].map(name => ({name})),
      dataContainer: {
        numRows: () => rows.length,
        valueAt: (row: number, column: number) => rows[row][column]
      }
    };
    for (const field of [makeCatalogPointColumns('east', 'north'), TRAJECTORY_POINT_FIELD]) {
      const tracks = await readCatalogNativeInput({
        source,
        field,
        isTrajectory: isTrajectoryCatalogOperation(binding.id, 'tracks'),
        parameters
      });
      const prepared = prepareFleetDwell({...createFleetDwellSample(), tracks}, parameters);
      expect(prepared.trajectory.fixes.map(fix => fix.timestamp)).toEqual([0, 10000, 20000]);
      expect(prepared.trajectory.trackIds).toEqual(['van']);
    }
    const retained = makeCatalogTrajectoryNativeSource({
      id: 'retained',
      label: 'Retained trajectory',
      table: arrow.tableFromArrays({
        track_id: ['van', 'van', 'van'],
        timestamp: [0, 10, 20],
        east: [0, 0, 0],
        north: [0, 0, 0]
      })
    });
    const tracks = await readCatalogNativeInput({
      source: retained,
      field: TRAJECTORY_POINT_FIELD,
      isTrajectory: isTrajectoryCatalogOperation(binding.id, 'tracks'),
      parameters: {...parameters, timestampEncoding: 'unix-second'}
    });
    expect(
      prepareFleetDwell(
        {...createFleetDwellSample(), tracks},
        {...parameters, timestampEncoding: 'unix-second'}
      ).trajectory.fixes.map(fix => fix.timestamp)
    ).toEqual([0, 10000, 20000]);
    const samples = getCatalogSampleInputs(binding.id);
    expect(Object.keys(samples).sort()).toEqual(['tracks', 'zones']);
    expect(
      prepareFleetDwell(
        Object.fromEntries(
          Object.entries(samples).map(([input, text]) => [input, JSON.parse(text)])
        ),
        {}
      ).trajectory.fixes
    ).toHaveLength(384);
  });

  it('carries validated native field mappings and seconds encoding through applicability with explicit choices taking precedence', () => {
    const tracks = {
      type: 'FeatureCollection' as const,
      features: [0, 10, 20].map(recorded_at => ({
        geometry: {type: 'Point', coordinates: [0, 0]},
        properties: {vehicle_code: 'van', recorded_at, east: 0, north: 0}
      }))
    };
    const parameters = {
      trackProperty: 'vehicle_code',
      timeProperty: 'recorded_at',
      timestampEncoding: 'unix-second',
      longitudeProperty: 'east',
      latitudeProperty: 'north'
    };
    const prepared = prepareFleetDwell(
      {...createFleetDwellSample(), tracks},
      {...parameters, coordinateSystem: 'planar'}
    );
    const profile = {
      ...profileKeplerFeatureCollection(tracks, {
        datasetId: 'native',
        label: 'Native track fixes',
        coordinateSystem: 'planar'
      }),
      trajectory: {
        parameters,
        fixCount: prepared.trajectory.fixes.length,
        trackCount: prepared.trajectory.trackIds.length
      }
    };
    const zones = profileKeplerFeatureCollection(createFleetDwellSample()['zones'], {
      datasetId: 'zones',
      label: 'Zones',
      coordinateSystem: 'planar'
    });
    const options = {inputAssignments: {tracks: 'native', zones: 'zones'}};
    const result = evaluateKeplerOperationApplicability(
      binding,
      CURATED_FLEET_DWELL_DESCRIPTOR,
      [profile, zones],
      options
    );
    expect(result.status, JSON.stringify(result.missing)).toBe('ready');
    for (const [name, value] of Object.entries(parameters))
      expect(result.parameters[name], name).toBe(value);
    const explicit = evaluateKeplerOperationApplicability(
      binding,
      CURATED_FLEET_DWELL_DESCRIPTOR,
      [profile, zones],
      {...options, overrides: {timestampEncoding: 'unix-millisecond'}}
    );
    expect(explicit.parameters['timestampEncoding']).toBe('unix-millisecond');
    const differentlyNamed = {...binding, id: 'specialist-dwell'};
    const renamedDescriptor = {...CURATED_FLEET_DWELL_DESCRIPTOR, bindingId: differentlyNamed.id};
    expect(
      evaluateKeplerOperationApplicability(
        differentlyNamed,
        renamedDescriptor,
        [profile, zones],
        options
      ).parameters['timeProperty']
    ).toBe('recorded_at');
  });

  it('registers a recipe-backed GPU operation, descriptor, sample and portable template', () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    expect(validateKeplerOperationDescriptor(CURATED_FLEET_DWELL_DESCRIPTOR, binding)).toEqual([]);
    expect(validateKeplerCuratedAnalysis(analysis)).toEqual([]);
    const template = createCuratedWorkflowAnalysisTemplate(analysis);
    expect(parseWorkflowAnalysisTemplate(JSON.stringify(template))).toEqual(template);
    expect(template.exposed).toHaveLength(3);
    expect(template.slots.map(slot => slot.name)).toEqual(['tracks', 'zones']);
    expect(validateWorkflowAnalysisTemplate(template).issues).toEqual([]);
    const encoding = template.exposed.find(
      parameter => parameter.target.parameter === 'timestampEncoding'
    )!;
    expect(encoding.defaultPolicy).toEqual({from: 'time-encoding', input: 'tracks', role: 'time'});
    const malformed = JSON.parse(JSON.stringify(template));
    delete malformed.exposed[2].defaultPolicy.role;
    expect(() => parseWorkflowAnalysisTemplate(malformed)).toThrow(/defaultPolicy.role/);
    malformed.exposed[2].defaultPolicy.role = 'absent';
    expect(
      validateWorkflowAnalysisTemplate(malformed).issues.some(
        issue => issue.code === 'unknown-role'
      )
    ).toBe(true);
    expect(FLEET_DWELL_SAMPLE_LICENSE.license).toBe('MIT');
    expect(template.sample).toEqual(FLEET_DWELL_SAMPLE_LICENSE);
    expect(template.sample?.attribution).toBe('vis.gl contributors');
    expect(createFleetDwellSample()).toEqual(createFleetDwellSample());
    expect(createFleetDwellSample()).not.toEqual(createFleetDwellSample(101));
  });

  it('prepares the sample through shared signed Int64 time and local-frame trajectory handling', () => {
    const prepared = prepareFleetDwell(createFleetDwellSample(), {});
    expect(prepared.trajectory.fixes).toHaveLength(384);
    expect(prepared.trajectory.trackIds).toHaveLength(32);
    expect(prepared.zones.featureCount).toBe(3);
    const {trajectory} = prepared;
    const oracle = computeTrajectoryOracle(
      {
        positions: trajectory.positions,
        timestamps: new Float32Array(384),
        trackOffsets: [...trajectory.trackOffsets]
      },
      0.001,
      20000,
      createDoubleOracleTime(trajectory.fixes.map(fix => fix.timestamp))
    );
    expect(oracle.stops).toHaveLength(128);
    expect(oracle.trackStopCounts).toEqual(Array(32).fill(4));
    expect(oracle.stops.slice(0, 4).map(stop => stop.duration)).toEqual([
      20000, 20000, 20000, 20000
    ]);
    expect(prepared.metricParameters[0]).toBeCloseTo(0.001, 8);
    expect([...trajectory.timestamps.slice(0, 2)]).toEqual([3260150784, 396]);
    const position = trajectory.frame.inverse(trajectory.positions[0], trajectory.positions[1]);
    expect(position[0]).toBeCloseTo(-87.65 - 0.9 / 83000, 7);
  });

  it('maps independently named point fields and retains native row identity after sorting', () => {
    const sample = createFleetDwellSample(101);
    sample['tracks'].features.reverse();
    for (const feature of sample['tracks'].features)
      feature.properties = {
        vehicle_code: feature.properties!['track_id'],
        recorded_at: new Date(Number(feature.properties!['timestamp'])).toISOString()
      };
    const profiles = Object.entries(sample).map(([datasetId, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId,
        label: datasetId,
        coordinateSystem: 'EPSG:4326'
      })
    );
    let mapping = createKeplerCuratedAnalysisMapping(analysis, profiles);
    for (const [role, column] of [
      ['track', 'vehicle_code'],
      ['time', 'recorded_at']
    ])
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'tracks', role, columns: [column]},
        analysis,
        profiles
      );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
      parameterOverrides: {timestampEncoding: 'iso8601'}
    });
    expect(request.inputDatasetIds).toEqual({tracks: 'tracks', zones: 'zones'});
    expect(request.parameters['trackProperty']).toBe('vehicle_code');
    expect(request.parameters['timeProperty']).toBe('recorded_at');
    const prepared = prepareFleetDwell(sample, request.parameters);
    expect(prepared.trajectory.fixes[0].sourceIndex).toBe(11);
    expect(prepared.trajectory.fixes[0].timestamp).toBe(Date.UTC(2024, 0, 1));
  });

  it('supports timed lines, embedded IDs/times, pre-epoch seconds, holes and multipart zones', () => {
    const sample = createFleetDwellSample();
    sample['tracks'] = {
      type: 'FeatureCollection',
      features: [
        {
          id: 'line-one',
          geometry: {
            type: 'LineString',
            coordinates: [
              [0, 0, 0, -3],
              [0, 0, 0, -2],
              [0, 0, 0, -1]
            ]
          },
          properties: {}
        }
      ]
    };
    sample['zones'] = {
      type: 'FeatureCollection',
      features: [
        {
          id: 'multi',
          geometry: {
            type: 'MultiPolygon',
            coordinates: [
              [
                [
                  [-5, -5],
                  [5, -5],
                  [5, 5],
                  [-5, 5],
                  [-5, -5]
                ],
                [
                  [-1, -1],
                  [-1, 1],
                  [1, 1],
                  [1, -1],
                  [-1, -1]
                ]
              ],
              [
                [
                  [10, 10],
                  [20, 10],
                  [20, 20],
                  [10, 20],
                  [10, 10]
                ]
              ]
            ]
          }
        }
      ]
    };
    const prepared = prepareFleetDwell(sample, {
      coordinateSystem: 'planar',
      timestampEncoding: 'unix-second'
    });
    expect(prepared.trajectory.trackIds).toEqual(['line-one']);
    expect(
      prepared.trajectory.fixes.map(fix => [fix.sourceIndex, fix.vertexIndex, fix.timestamp])
    ).toEqual([
      [0, 0, -3000],
      [0, 1, -2000],
      [0, 2, -1000]
    ]);
    expect([...prepared.trajectory.timestamps.slice(0, 2)]).toEqual([4294964296, 4294967295]);
    expect([...prepared.zones.featureOffsets]).toEqual([0, 2]);
    expect([...prepared.zones.polygonOffsets]).toEqual([0, 2, 3]);
    expect([...prepared.zones.ringOffsets]).toEqual([0, 4, 8, 12]);
  });

  it('rejects invalid mappings, duplicate times and named capacity limits before allocation', () => {
    const sample = createFleetDwellSample();
    const invalidOverrides: Record<string, CatalogParameterValue>[] = [
      {stopCapacity: 0},
      {candidateCapacity: 0},
      {stopMinimumDuration: -1},
      {stopSpeedThreshold: NaN},
      {timestampEncoding: 'guess'}
    ];
    for (const overrides of invalidOverrides)
      expect(() => prepareFleetDwell(sample, overrides)).toThrow(/curated-fleet-dwell/);
    expect(() => prepareFleetDwell(sample, {timeProperty: 'absent'})).toThrow(/timestamp/);
    const clean = createFleetDwellSample();
    sample['tracks'].features.push(sample['tracks'].features[0]);
    expect(() => prepareFleetDwell(sample, {})).toThrow(/distinct.*increasing/);
    expect(() =>
      prepareFleetDwell(
        {
          ...clean,
          zones: {type: 'FeatureCollection', features: Array(129).fill(sample['zones'].features[0])}
        },
        {}
      )
    ).toThrow(/curated-fleet-dwell.*zones.length/);
    expect(() =>
      prepareFleetDwell(
        {
          ...clean,
          tracks: {
            type: 'FeatureCollection',
            features: Array.from({length: 129}, (_, index) => ({
              ...clean['tracks'].features[0],
              properties: {track_id: `track-${index}`, timestamp: 0}
            }))
          }
        },
        {}
      )
    ).toThrow(/trackCount/);
    expect(() =>
      prepareFleetDwell(
        {
          ...clean,
          zones: {
            type: 'FeatureCollection',
            features: [{geometry: {type: 'Polygon', coordinates: [Array(16385).fill([0, 0])]}}]
          }
        },
        {}
      )
    ).toThrow(/zones.vertexCount/);
    expect(() =>
      prepareFleetDwell(
        {
          ...clean,
          tracks: {...clean['tracks'], features: Array(2049).fill(clean['tracks'].features[0])}
        },
        {}
      )
    ).toThrow(/fixCount/);
    expect(() =>
      prepareFleetDwell({...clean, zones: {type: 'FeatureCollection', features: []}}, {})
    ).toThrow(/zones.length/);
    for (const geometry of [
      null,
      {type: 'Polygon'},
      {
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [1, 1],
            ['bad', 0],
            [0, 0]
          ]
        ]
      }
    ])
      expect(() =>
        prepareFleetDwell(
          {...clean, zones: {type: 'FeatureCollection', features: [{geometry}]}},
          {}
        )
      ).toThrow(/curated-fleet-dwell.*zones must contain finite/);
  });

  it('decodes publication units, exact source identities, outside stops, empty zones and presentation', () => {
    const sample = createFleetDwellSample();
    const prepared = prepareFleetDwell(sample, {});
    const result: FleetDwellReadback = {
      count: 2,
      requiredCount: 2,
      stopOverflow: 0,
      joinOverflow: 0,
      joinUncertainCount: 0,
      tableOverflow: 0,
      tableCount: 3,
      tracks: Uint32Array.of(0, 0),
      startRows: Uint32Array.of(0, 6),
      endRows: Uint32Array.of(2, 8),
      centroids: Float32Array.from([
        ...prepared.trajectory.positions.slice(0, 2),
        ...prepared.trajectory.positions.slice(12, 14)
      ]),
      durations: Float32Array.of(20000, 20000),
      stopZones: Uint32Array.of(0, 0xffffffff),
      counts: Uint32Array.of(1, 0, 0),
      totals: Float32Array.of(20000, 0, 0),
      means: Float32Array.of(20000, NaN, NaN),
      maximums: Float32Array.of(20000, NaN, NaN)
    };
    const output = makeFleetDwellOutput(prepared, result);
    expect(output.tables[0].columns['duration_seconds']).toEqual([20, 20]);
    expect(output.tables[0].columns['source_index']).toEqual([0, 6]);
    expect(output.tables[0].columns['zone_id']).toEqual(['depot-west', null]);
    expect(output.tables[1].columns['total_dwell_seconds']).toEqual([20, 0, 0]);
    expect(output.tables[1].columns['mean_dwell_seconds']).toEqual([20, null, null]);
    expect(output.tables[1].columns['geometry']).toEqual(
      sample['zones'].features.map(feature => JSON.stringify(feature.geometry))
    );
    expect(output.summary['unassignedStopCount']).toBe(1);
    for (const [index, table] of output.tables.entries()) {
      const presentation = planKeplerResultPresentation(
        table,
        CURATED_FLEET_DWELL_DESCRIPTOR.outputs[index]
      );
      expect(presentation?.column).toBe(index ? 'total_dwell_seconds' : 'duration_seconds');
    }
    for (const key of [
      'stopOverflow',
      'joinOverflow',
      'tableOverflow',
      'joinUncertainCount'
    ] as const)
      expect(() => makeFleetDwellOutput(prepared, {...result, [key]: 1}), key).toThrow(
        /no (partial|definitive)/
      );
    expect(() => makeFleetDwellOutput(prepared, {...result, requiredCount: 2049})).toThrow(
      /capacity overflow/
    );
    expect(() =>
      makeFleetDwellOutput(prepared, {
        ...result,
        count: 0,
        requiredCount: 0,
        tracks: new Uint32Array()
      })
    ).not.toThrow();
  });
});
