// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateGPUAnalysisResultEncoding,
  type GPUAnalysisCoordinateModel
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {describe, expect, it} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  runCatalogOperation
} from '../../src/analysis/catalog-bindings';
import {
  KEPLER_GRID_CELL_CAPS,
  resolveKeplerOperationParameters
} from '../../src/descriptors/default-parameters';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import type {
  CatalogOutputTable,
  CatalogFeatureCollection
} from '../../src/analysis/catalog-geojson';
import {CLUSTERING_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-clustering';
import {INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-interpolation-density';
import {STATISTICS_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-statistics';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import type {KeplerOperationDescriptor} from '../../src/descriptors/operation-descriptor-types';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';

const STATISTICS_IDS = [
  'spatial-weights',
  'global-spatial-statistics',
  'local-moran-statistics',
  'hotspot-gi-star',
  'local-geary-statistics',
  'empirical-bayes-rates',
  'multiple-testing-adjustment',
  'knox-test',
  'similar-locations',
  'location-allocation',
  'weights-algebra',
  'weights-transpose',
  'weights-transform',
  'regression-ols',
  'global-join-counts-getis-ord',
  'local-lisa-variants',
  'recipe-hot-spot-lattice',
  'regression-gwr',
  'regression-probit',
  'regression-sur'
];
const CLUSTERING_IDS = [
  'density-clustering',
  'optics-ordering',
  'local-outlier-factor',
  'hierarchical-density-clustering',
  'skater-regions'
];
const INTERPOLATION_DENSITY_IDS = [
  'idw-grid',
  'kriging-grid',
  'areal-interpolation',
  'kernel-density-grid',
  'point-density-grid',
  'line-density-grid'
];
const ALL_IDS = [...STATISTICS_IDS, ...CLUSTERING_IDS, ...INTERPOLATION_DENSITY_IDS];

const ALL_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  ...STATISTICS_OPERATION_DESCRIPTORS,
  ...CLUSTERING_OPERATION_DESCRIPTORS,
  ...INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS
];

/** GPU-only bindings have no CPU engine; their descriptors are verified by reading. */
const GPU_ONLY_IDS = {
  'global-join-counts-getis-ord': 'engine gpu: no CPU run (needs a device); verified by reading',
  'local-lisa-variants': 'engine gpu: no CPU run (needs a device); verified by reading'
} as const;

/** Units of a parameter whose value is a distance in the coordinate system of the input. */
const LENGTH_UNITS = ['crs', 'm', 'meter', 'metre', 'coordinate-units'];

// ---------------------------------------------------------------------------------------------
// Synthetic inputs. A deterministic generator keeps every run reproducible.

function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

type Properties = Record<string, string | number>;

/**
 * One set of inputs in one coordinate system. Positions are built as offsets in degrees from an
 * origin and then placed: `geographic` adds them to a longitude and latitude (central London);
 * `planar` converts them to metres on a local tangent plane, so the same shapes exist in both.
 */
type Fixture = {
  name: 'geographic' | 'planar';
  model: GPUAnalysisCoordinateModel;
  coordinateSystem: 'EPSG:4326' | 'planar';
  lattice: CatalogFeatureCollection;
  points: CatalogFeatureCollection;
  candidates: CatalogFeatureCollection;
  lines: CatalogFeatureCollection;
  targets: CatalogFeatureCollection;
};

const ORIGIN_LONGITUDE = -0.13;
const ORIGIN_LATITUDE = 51.5;
const METRES_PER_DEGREE_LATITUDE = 110574;
const METRES_PER_DEGREE_LONGITUDE = 111320 * Math.cos((ORIGIN_LATITUDE * Math.PI) / 180);

function makeFixture(name: Fixture['name']): Fixture {
  const place = (offsetX: number, offsetY: number): [number, number] =>
    name === 'geographic'
      ? [ORIGIN_LONGITUDE + offsetX, ORIGIN_LATITUDE + offsetY]
      : [offsetX * METRES_PER_DEGREE_LONGITUDE, offsetY * METRES_PER_DEGREE_LATITUDE];
  const makePoint = (offsetX: number, offsetY: number, properties: Properties) => ({
    type: 'Feature' as const,
    geometry: {type: 'Point' as const, coordinates: place(offsetX, offsetY)},
    properties
  });
  const makeSquare = (offsetX: number, offsetY: number, size: number, properties: Properties) => ({
    type: 'Feature' as const,
    geometry: {
      type: 'Polygon' as const,
      coordinates: [
        [
          place(offsetX, offsetY),
          place(offsetX + size, offsetY),
          place(offsetX + size, offsetY + size),
          place(offsetX, offsetY + size),
          place(offsetX, offsetY)
        ]
      ]
    },
    properties
  });
  const makeCollection = (features: unknown[]) =>
    ({type: 'FeatureCollection', features}) as CatalogFeatureCollection;

  // A 6 by 6 lattice of 0.01 degree squares with a high-value block in one corner: a rate
  // (value), counts and populations, a p-value, and two attributes.
  const latticeRandom = makeRandom(7);
  const latticeFeatures: unknown[] = [];
  for (let row = 0; row < 6; row++)
    for (let column = 0; column < 6; column++) {
      const hot = row >= 3 && column >= 3;
      latticeFeatures.push(
        makeSquare(column * 0.01, row * 0.01, 0.01, {
          name: `cell-${row}-${column}`,
          value: 10 + (hot ? 8 : 0) + latticeRandom() * 2,
          cases: 1 + Math.floor(latticeRandom() * 30),
          population: 300 + Math.floor(latticeRandom() * 4000),
          p: hot ? latticeRandom() * 0.01 : latticeRandom(),
          a: 5 + (hot ? 4 : 0) + latticeRandom(),
          b: latticeRandom() * 5
        })
      );
    }

  // Fifty-one points in about 1.4 by 1.3 km: three tight clusters of 15 and six scattered
  // points, each with a measured value, a numeric time, predictors, two responses, a binary
  // outcome and a weight. The extent is small so the binding defaults (10 m lattice, 100 m grid)
  // stay within their cell caps.
  const random = makeRandom(11);
  const pointFeatures: unknown[] = [];
  const addPoint = (offsetX: number, offsetY: number, cluster: number) => {
    const x1 = random() * 10;
    const x2 = random() * 10;
    const x3 = random() * 10;
    pointFeatures.push(
      makePoint(offsetX, offsetY, {
        value: 20 + cluster * 5 + random() * 3,
        time: cluster * 30 + random() * 10,
        x1,
        x2,
        x3,
        y: 2 + 3 * x1 - x2 + random() * 2,
        y2: 1 + 0.5 * x1 + 2 * x3 + random() * 2,
        flag: x1 - 5 + (random() - 0.5) * 8 > 0 ? 1 : 0,
        weight: 1 + Math.floor(random() * 5)
      })
    );
  };
  [
    [0.003, 0.002],
    [0.016, 0.003],
    [0.009, 0.01]
  ].forEach(([centerX, centerY], cluster) => {
    for (let index = 0; index < 15; index++)
      addPoint(centerX + (random() - 0.5) * 0.003, centerY + (random() - 0.5) * 0.002, cluster);
  });
  for (let index = 0; index < 6; index++) addPoint(random() * 0.02, random() * 0.012, 3);

  const candidateRandom = makeRandom(13);
  const candidateFeatures: unknown[] = [];
  for (let index = 0; index < 8; index++)
    candidateFeatures.push(makePoint(candidateRandom() * 0.02, candidateRandom() * 0.012, {}));

  const makeLine = (offsets: number[][]) => ({
    type: 'Feature' as const,
    geometry: {
      type: 'LineString' as const,
      coordinates: offsets.map(([offsetX, offsetY]) => place(offsetX, offsetY))
    },
    properties: {}
  });
  const lines = makeCollection([
    makeLine([
      [0, 0],
      [0.01, 0.004],
      [0.02, 0.008]
    ]),
    makeLine([
      [0, 0.012],
      [0.012, 0.006],
      [0.02, 0]
    ]),
    makeLine([
      [0.005, 0],
      [0.005, 0.012]
    ])
  ]);

  const targetFeatures: unknown[] = [];
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++)
      targetFeatures.push(
        makeSquare(column * 0.02, row * 0.02, 0.02, {name: `zone-${row}-${column}`})
      );

  return {
    name,
    model: name === 'geographic' ? 'geographic' : 'planar',
    coordinateSystem: name === 'geographic' ? 'EPSG:4326' : 'planar',
    lattice: makeCollection(latticeFeatures),
    points: makeCollection(pointFeatures),
    candidates: makeCollection(candidateFeatures),
    lines,
    targets: makeCollection(targetFeatures)
  };
}

const GEOGRAPHIC = makeFixture('geographic');
const PLANAR = makeFixture('planar');
const LATTICE = GEOGRAPHIC.lattice;
const POINTS = GEOGRAPHIC.points;

function makeSquare(x: number, y: number, size: number, properties: Properties) {
  return {
    type: 'Feature' as const,
    geometry: {
      type: 'Polygon' as const,
      coordinates: [
        [
          [x, y],
          [x + size, y],
          [x + size, y + size],
          [x, y + size],
          [x, y]
        ]
      ]
    },
    properties
  };
}

function makeCollection(features: unknown[]): CatalogFeatureCollection {
  return {type: 'FeatureCollection', features} as CatalogFeatureCollection;
}

// ---------------------------------------------------------------------------------------------
// One run case per binding: the inputs, the column each field parameter receives, and parameter
// overrides needed to make a conditional field role take effect.

type RunCase = {
  inputs: Record<string, CatalogFeatureCollection>;
  fields: Record<string, string>;
  extra?: Record<string, string | number>;
};

function makeRunCases(fixture: Fixture): Record<string, RunCase> {
  const {
    lattice: LATTICE,
    points: POINTS,
    candidates: CANDIDATES,
    lines: LINES,
    targets: TARGETS
  } = fixture;
  return {
    'spatial-weights': {inputs: {features: LATTICE}, fields: {}},
    'global-spatial-statistics': {inputs: {features: LATTICE}, fields: {valueProperty: 'value'}},
    'local-moran-statistics': {inputs: {features: LATTICE}, fields: {valueProperty: 'value'}},
    'hotspot-gi-star': {inputs: {features: LATTICE}, fields: {valueProperty: 'value'}},
    'local-geary-statistics': {inputs: {features: LATTICE}, fields: {valueProperty: 'value'}},
    'empirical-bayes-rates': {
      inputs: {features: LATTICE},
      fields: {numeratorProperty: 'cases', denominatorProperty: 'population'}
    },
    'multiple-testing-adjustment': {inputs: {features: LATTICE}, fields: {pValueProperty: 'p'}},
    'knox-test': {inputs: {points: POINTS}, fields: {timeProperty: 'time'}},
    'similar-locations': {
      inputs: {features: LATTICE},
      fields: {properties: 'a,b'},
      extra: {referenceIndex: 0}
    },
    'location-allocation': {
      inputs: {demand: POINTS, candidates: CANDIDATES},
      fields: {weightProperty: 'weight'}
    },
    'weights-algebra': {inputs: {points: POINTS}, fields: {}},
    'weights-transpose': {inputs: {points: POINTS}, fields: {}},
    'weights-transform': {inputs: {points: POINTS}, fields: {}},
    'regression-ols': {
      inputs: {points: POINTS},
      fields: {responseProperty: 'y', predictorProperties: 'x1,x2'}
    },
    'recipe-hot-spot-lattice': {inputs: {points: POINTS}, fields: {valueProperty: 'weight'}},
    'regression-gwr': {
      inputs: {points: POINTS},
      fields: {responseProperty: 'y', predictorProperties: 'x1,x2'}
    },
    'regression-probit': {
      inputs: {points: POINTS},
      fields: {responseProperty: 'flag', predictorProperties: 'x1,x2'}
    },
    'regression-sur': {
      inputs: {points: POINTS},
      fields: {responseProperties: 'y,y2', predictorProperties: 'x1,x2;x1,x3'}
    },
    'density-clustering': {inputs: {points: POINTS}, fields: {}},
    'optics-ordering': {inputs: {points: POINTS}, fields: {}},
    'local-outlier-factor': {inputs: {points: POINTS}, fields: {}},
    'hierarchical-density-clustering': {inputs: {points: POINTS}, fields: {}},
    'skater-regions': {inputs: {features: LATTICE}, fields: {properties: 'a,b'}},
    'idw-grid': {inputs: {points: POINTS}, fields: {valueProperty: 'value'}},
    'kriging-grid': {inputs: {points: POINTS}, fields: {valueProperty: 'value'}},
    'areal-interpolation': {
      inputs: {sources: LATTICE, targets: TARGETS},
      fields: {valueProperty: 'cases'},
      extra: {variableType: 'extensive'}
    },
    'kernel-density-grid': {inputs: {points: POINTS}, fields: {weightProperty: 'weight'}},
    // The weight only takes effect for the sum and mean statistics.
    'point-density-grid': {
      inputs: {points: POINTS},
      fields: {weightProperty: 'weight'},
      extra: {statistic: 'sum'}
    },
    'line-density-grid': {inputs: {lines: LINES}, fields: {}}
  };
}

const RUN_CASES = makeRunCases(GEOGRAPHIC);

const MISSING_COLUMN = 'no_such_column';

/** What the run produced, or the message it threw, for comparing two runs. */
function runOutcome(id: string, parameters: Record<string, string | number>, runCase: RunCase) {
  const binding = getCatalogBinding(id)!;
  try {
    const result = runCatalogOperation(binding, parameters, runCase.inputs);
    return {tables: result.tables, error: undefined as string | undefined};
  } catch (failure) {
    return {tables: [] as CatalogOutputTable[], error: String((failure as Error).message)};
  }
}

function summarizeTables(tables: readonly CatalogOutputTable[]): string {
  return JSON.stringify(tables.map(table => ({id: table.id, columns: table.columns})));
}

describe('statistics, clustering, interpolation and density descriptors: coverage', () => {
  it('covers the 31 ids exactly once, in order, in three arrays', () => {
    expect(STATISTICS_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      STATISTICS_IDS
    );
    expect(CLUSTERING_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      CLUSTERING_IDS
    );
    expect(
      INTERPOLATION_DENSITY_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)
    ).toEqual(INTERPOLATION_DENSITY_IDS);
    expect(ALL_IDS).toHaveLength(31);
    expect(new Set(ALL_IDS).size).toBe(31);
    expect(Object.keys(RUN_CASES).sort()).toEqual(
      ALL_IDS.filter(id => !(id in GPU_ONLY_IDS)).sort()
    );
  });

  it('keeps the GPU-only list honest: those bindings have no CPU engine and the rest do', () => {
    for (const id of ALL_IDS) {
      const binding = getCatalogBinding(id)!;
      expect(binding, id).toBeDefined();
      if (id in GPU_ONLY_IDS) {
        expect(binding.run, `${id} gained a CPU engine: add a run case`).toBeUndefined();
        expect(binding.engine).toBe('gpu');
      } else {
        expect(binding.run, `${id} has no CPU engine: list it as skipped`).toBeDefined();
      }
    }
  });
});

describe('descriptors agree with their bindings', () => {
  for (const descriptor of ALL_DESCRIPTORS) {
    it(`${descriptor.bindingId}: no validator issues, capacity matches the binding`, () => {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
      for (const requirement of descriptor.requirements) {
        const input = binding.inputs.find(candidate => candidate.name === requirement.input)!;
        if (requirement.id === 'maximum-rows') {
          expect(requirement.limit, `${requirement.input} rows`).toBe(input.maximumFeatures);
        }
        if (requirement.id === 'maximum-vertices') {
          expect(requirement.limit, `${requirement.input} vertices`).toBe(input.maximumVertices);
        }
      }
    });

    it(`${descriptor.bindingId}: at most four basic parameters, no column left as a plain string`, () => {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      expect(
        descriptor.parameters.filter(parameter => parameter.level === 'basic').length
      ).toBeLessThanOrEqual(4);
      for (const parameter of descriptor.parameters) {
        const spec = binding.parameters.find(candidate => candidate.name === parameter.name)!;
        expect(spec.kind, `${parameter.name} is a column name`).not.toBe('string');
      }
    });
  }

  it('gives the first analyses prominence 1 and the building blocks prominence 3', () => {
    const byId = new Map(ALL_DESCRIPTORS.map(descriptor => [descriptor.bindingId, descriptor]));
    for (const id of [
      'recipe-hot-spot-lattice',
      'kernel-density-grid',
      'density-clustering',
      'local-moran-statistics',
      'hotspot-gi-star',
      'regression-ols',
      'idw-grid'
    ]) {
      expect(byId.get(id)!.prominence, id).toBe(1);
    }
    for (const id of [
      'weights-algebra',
      'weights-transpose',
      'weights-transform',
      'multiple-testing-adjustment'
    ]) {
      expect(byId.get(id)!.prominence, id).toBe(3);
    }
  });

  it('orders accepted column types by preference: integer first for counts and for the Knox time', () => {
    const roleOf = (id: string, role: string) =>
      ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === id)!
        .inputs.flatMap(input => input.fields)
        .find(field => field.role === role)!;
    // ISO text is accepted by a `time` column but the binding then fails with "Too few features"
    expect(roleOf('knox-test', 'time').accepts).toEqual(['integer', 'number']);
    expect(roleOf('empirical-bayes-rates', 'cases').accepts[0]).toBe('integer');
    expect(roleOf('empirical-bayes-rates', 'population').accepts[0]).toBe('integer');
    expect(roleOf('regression-probit', 'response').accepts[0]).toBe('integer');
    expect(
      ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === 'empirical-bayes-rates')!
        .questions
    ).toEqual(['how-much-per-area', 'prepare-data']);
  });

  it('names the primary question the brief assigns', () => {
    const primary = new Map(
      ALL_DESCRIPTORS.map(descriptor => [descriptor.bindingId, descriptor.questions[0]])
    );
    for (const id of [
      'hotspot-gi-star',
      'local-moran-statistics',
      'local-geary-statistics',
      'local-lisa-variants',
      'recipe-hot-spot-lattice',
      'global-spatial-statistics',
      'global-join-counts-getis-ord',
      'local-outlier-factor',
      'knox-test'
    ]) {
      expect(primary.get(id), id).toBe('where-significant');
    }
    for (const id of ['kernel-density-grid', 'point-density-grid', 'line-density-grid']) {
      expect(primary.get(id), id).toBe('where-concentrated');
    }
    for (const id of ['idw-grid', 'kriging-grid'])
      expect(primary.get(id), id).toBe('value-where-unmeasured');
    for (const id of ['regression-ols', 'regression-gwr', 'regression-probit', 'regression-sur']) {
      expect(primary.get(id), id).toBe('how-related');
    }
    for (const id of [
      'density-clustering',
      'optics-ordering',
      'hierarchical-density-clustering',
      'skater-regions'
    ]) {
      expect(primary.get(id), id).toBe('which-groups-or-regions');
    }
    expect(primary.get('location-allocation')).toBe('what-is-near-or-reachable');
    expect(primary.get('areal-interpolation')).toBe('prepare-data');
    expect(
      ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === 'areal-interpolation')!.questions
    ).toEqual(['prepare-data', 'how-much-per-area']);
    expect(
      ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === 'density-clustering')!.questions
    ).toContain('where-concentrated');
  });
});

describe('measurement levels', () => {
  const roleOf = (id: string, inputName: string, role: string) =>
    ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === id)!
      .inputs.find(input => input.name === inputName)!
      .fields.find(field => field.role === role)!;

  it('area statistics take rates and amounts, not raw counts', () => {
    for (const id of [
      'global-spatial-statistics',
      'local-moran-statistics',
      'hotspot-gi-star',
      'local-geary-statistics',
      'global-join-counts-getis-ord',
      'local-lisa-variants'
    ]) {
      const measurement = roleOf(id, 'features', 'value').measurement!;
      expect(measurement, id).toContain('rate');
      expect(measurement, id).not.toContain('count');
      // a total per polygon mostly measures the size of the polygon
      expect(roleOf(id, 'features', 'value').measurementByGeometry?.polygon, id).toEqual(['rate']);
    }
  });

  it('rate smoothing takes counts, interpolation takes measurements, regressions skip identifiers', () => {
    expect(roleOf('empirical-bayes-rates', 'features', 'cases').measurement).toEqual(['count']);
    expect(roleOf('empirical-bayes-rates', 'features', 'population').measurement).toEqual([
      'count'
    ]);
    for (const id of ['kriging-grid', 'idw-grid']) {
      const measurement = roleOf(id, 'points', 'value').measurement!;
      expect(measurement, id).not.toContain('count');
    }
    for (const id of ['regression-ols', 'regression-gwr', 'regression-sur']) {
      const role = ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === id)!.inputs[0]
        .fields[0];
      expect(role.measurement, id).not.toContain('identifier');
      expect(role.measurement, id).not.toContain('cyclic');
    }
  });
});

describe('data-derived defaults', () => {
  const derived = ALL_DESCRIPTORS.flatMap(descriptor =>
    descriptor.parameters
      .filter(parameter => parameter.defaultPolicy.from === 'data')
      .map(parameter => ({descriptor, parameter}))
  );

  it('derives at least one parameter in most descriptors of the three groups', () => {
    expect(derived.length).toBeGreaterThanOrEqual(15);
  });

  for (const {descriptor, parameter} of derived) {
    it(`${descriptor.bindingId}.${parameter.name}: a number whose unit is a length`, () => {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      const spec = binding.parameters.find(candidate => candidate.name === parameter.name)!;
      expect(['number', 'integer']).toContain(spec.kind);
      const policy = parameter.defaultPolicy;
      // A rule that reads a column gives a number in that column's unit, not a length.
      if (policy.from === 'data' && !policy.rule.startsWith('column-')) {
        expect(LENGTH_UNITS, `${parameter.name} unit ${spec.unit}`).toContain(spec.unit);
      }
      // A count or a probability is never derived from the data.
      expect(spec.kind === 'integer', 'an integer parameter is a count').toBe(false);
    });
  }

  it('keeps the largest nearest-neighbor distance meaning: every feature has a neighbor', async () => {
    // Planar lattice of 5 by 5 unit squares and one square far to the right. The largest
    // nearest-neighbor distance between centroids is the isolated square's distance to the
    // lattice; at that band no feature is an island, just below it one is.
    const squares: ReturnType<typeof makeSquare>[] = [];
    for (let row = 0; row < 5; row++)
      for (let column = 0; column < 5; column++)
        squares.push(makeSquare(column, row, 1, {value: row + column}));
    squares.push(makeSquare(8, 0, 1, {value: 1}));
    const features = makeCollection(squares);
    const anchors = squares.map(square => {
      const ring = square.geometry.coordinates[0];
      return [(ring[0][0] + ring[2][0]) / 2, (ring[0][1] + ring[2][1]) / 2];
    });
    const maximumNearest = Math.max(
      ...anchors.map((anchor, index) =>
        Math.min(
          ...anchors
            .filter((_, other) => other !== index)
            .map(other => Math.hypot(anchor[0] - other[0], anchor[1] - other[1]))
        )
      )
    );
    expect(maximumNearest).toBeCloseTo(4, 12);
    const binding = getCatalogBinding('spatial-weights')!;
    const run = (distanceBand: number) =>
      runCatalogOperation(
        binding,
        {neighborhood: 'distance-band', distanceBand, coordinateSystem: 'planar'},
        {features}
      );
    expect(run(maximumNearest).summary['islandCount']).toBe(0);
    expect(run(maximumNearest * 0.99).summary['islandCount']).toBeGreaterThan(0);
  });

  it('keeps grid defaults inside each binding cell cap', () => {
    const policyOf = (id: string) =>
      ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === id)!.parameters.find(
        parameter => parameter.name === 'cellSize'
      )!.defaultPolicy;
    const dataPolicy = (id: string) => {
      const policy = policyOf(id);
      expect(policy.from, id).toBe('data');
      if (policy.from !== 'data') throw new Error(id);
      return policy;
    };
    const cellsForExtent = (factor: number, width: number, height: number) => {
      const cellSize = (Math.min(width, height) / 250) * factor;
      return Math.ceil(width / cellSize) * Math.ceil(height / cellSize);
    };
    // Density sets the starting resolution; the resolver raises it to fit the actual grid cap,
    // including the extra boundary cells and elongated extents. A factor alone cannot ensure it.
    for (const [id, pointLimit, cap] of [
      ['recipe-hot-spot-lattice', 100000, 40000],
      ['point-density-grid', 20000, 250000]
    ] as const) {
      const policy = dataPolicy(id);
      expect(policy.rule, id).toBe('mean-point-spacing');
      for (const [width, height] of [
        [1000, 1000],
        [100000, 10]
      ]) {
        const baseProfile = profileKeplerFeatureCollection(
          planarPoints([
            [0, 0],
            [width, height]
          ]),
          {
            datasetId: 'points',
            label: 'Points',
            coordinateSystem: 'planar'
          }
        );
        // An exact extent/count profile exercises capacity without allocating 100,000 features.
        const profile = {
          ...baseProfile,
          rowCount: pointLimit,
          vertexCount: pointLimit,
          spacing: {...baseProfile.spacing!, pointCount: pointLimit}
        };
        const resolved = resolveKeplerOperationParameters(
          getCatalogBinding(id)!,
          descriptorOf(id),
          {points: profile},
          {maximumCellCount: cap}
        );
        const cellSize = Number(resolved.parameters['cellSize']);
        const cells = (Math.floor(width / cellSize) + 1) * (Math.floor(height / cellSize) + 1);
        expect(cells, `${id}, extent ${width} by ${height}`).toBeLessThanOrEqual(cap);
      }
    }
    // Display grids keep the extent rule, scaled to the cap of the binding.
    expect(dataPolicy('kriging-grid').rule).toBe('extent-shorter-side-over-250');
    expect(cellsForExtent(dataPolicy('kriging-grid').factor ?? 1, 2000, 1000)).toBeLessThanOrEqual(
      40000
    );
    // A rule without the factor would break the kriging cap on a square extent.
    expect(cellsForExtent(1, 1000, 1000)).toBeGreaterThan(40000);
    for (const id of ['idw-grid', 'kernel-density-grid', 'line-density-grid']) {
      expect(cellsForExtent(dataPolicy(id).factor ?? 1, 4000, 1000), id).toBeLessThanOrEqual(
        250000
      );
    }
  });
});

describe('descriptors run their bindings on synthetic data', () => {
  for (const id of ALL_IDS) {
    if (id in GPU_ONLY_IDS) {
      it.skip(`${id}: ${GPU_ONLY_IDS[id as keyof typeof GPU_ONLY_IDS]}`, () => {});
      continue;
    }
    const descriptor = ALL_DESCRIPTORS.find(candidate => candidate.bindingId === id)!;
    const runCase = RUN_CASES[id];
    const baseParameters = {...runCase.fields, ...runCase.extra};

    it(`${id}: runs with defaults plus the field parameters, and every field parameter is used`, () => {
      const fieldParameters = descriptor.inputs.flatMap(input =>
        input.fields.map(field => field.parameter)
      );
      expect(Object.keys(runCase.fields).sort(), 'field parameters in the run case').toEqual(
        [...fieldParameters].sort()
      );
      const outcome = runOutcome(id, baseParameters, runCase);
      expect(outcome.error).toBeUndefined();
      expect(outcome.tables.length).toBeGreaterThan(0);
    });

    it(`${id}: each presentation column exists and holds values its scale can hold`, () => {
      const outcome = runOutcome(id, baseParameters, runCase);
      expect(outcome.error).toBeUndefined();
      for (const output of descriptor.outputs) {
        const table = outcome.tables.find(candidate => candidate.id === output.tableId);
        expect(table, `table ${output.tableId}`).toBeDefined();
        if (!output.presentation) continue;
        const {column, scale, classScheme} = output.presentation;
        expect(column in table!.columns, `${output.tableId}.${column}`).toBe(true);
        expect(table!.rowCount, `${output.tableId} rows`).toBeGreaterThan(0);
        const values = table!.columns[column];
        if (scale === 'class-scheme') {
          // The scheme and encoding the descriptor names must account for every value the real
          // run wrote: a share of 1.
          expect(
            validateGPUAnalysisResultEncoding(
              classScheme!,
              output.presentation.classEncoding ?? 'contributor-integer',
              values
            ),
            `${output.tableId}.${column} as ${classScheme} (${output.presentation.classEncoding})`
          ).toBe(1);
        } else if (scale === 'categorical') {
          values.forEach((value, row) => expect(value, `${column}[${row}]`).not.toBeNull());
        } else if (scale === 'sequential' || scale === 'diverging') {
          values.forEach((value, row) =>
            expect(
              value === null || (typeof value === 'number' && Number.isFinite(value)),
              `${column}[${row}] = ${String(value)}`
            ).toBe(true)
          );
        }
        if (scale === 'diverging') expect(output.presentation.center).toBe(0);
      }
    });

    for (const input of descriptor.inputs) {
      for (const field of input.fields) {
        it(`${id}: setting ${field.parameter} to a missing column changes the outcome`, () => {
          const baseline = runOutcome(id, baseParameters, runCase);
          const changed = runOutcome(
            id,
            {...baseParameters, [field.parameter]: MISSING_COLUMN},
            runCase
          );
          expect(baseline.error).toBeUndefined();
          const different =
            changed.error !== undefined ||
            summarizeTables(changed.tables) !== summarizeTables(baseline.tables);
          expect(different, `${field.parameter} is ignored`).toBe(true);
        });
      }
    }
  }

  it('the encodings the class schemes rely on are what the bindings produce', () => {
    const moran = runOutcome(
      'local-moran-statistics',
      {valueProperty: 'value'},
      {
        inputs: {features: LATTICE},
        fields: {}
      }
    );
    const clusters = new Set(moran.tables[0].columns['cluster']);
    expect(
      validateGPUAnalysisResultEncoding('local-moran-clusters', 'catalog-text', [...clusters])
    ).toBe(1);
    // the contributor integer encoding does not fit the text column: the check can fail
    expect(
      validateGPUAnalysisResultEncoding('local-moran-clusters', 'contributor-integer', [
        ...clusters
      ])
    ).toBe(0);
    const giStar = runOutcome(
      'hotspot-gi-star',
      {valueProperty: 'value'},
      {
        inputs: {features: LATTICE},
        fields: {}
      }
    );
    const classes = new Set(giStar.tables[0].columns['classification']);
    expect(
      validateGPUAnalysisResultEncoding('hot-spot-confidence-bins', 'catalog-text', [...classes])
    ).toBe(1);
    // The clustered fixture has a strong high-value block: Gi* must flag at least one hot spot.
    expect([...classes].some(value => String(value).startsWith('hot-'))).toBe(true);
    const lattice = runOutcome(
      'recipe-hot-spot-lattice',
      {valueProperty: 'weight'},
      {
        inputs: {points: POINTS},
        fields: {}
      }
    );
    const bins = lattice.tables[0].columns['bin'] as number[];
    expect(Math.min(...bins)).toBeGreaterThanOrEqual(-3);
    expect(Math.max(...bins)).toBeLessThanOrEqual(3);
    expect(bins.some(value => value !== 0)).toBe(true);
    const dbscan = runOutcome('density-clustering', {}, {inputs: {points: POINTS}, fields: {}});
    const labels = dbscan.tables[0].columns['cluster'] as number[];
    expect(labels.some(label => label >= 0)).toBe(true);
    expect(Math.min(...labels)).toBeGreaterThanOrEqual(-1);
  });
});

// ---------------------------------------------------------------------------------------------
// Policies, and defaults resolved through the engine (`resolveKeplerOperationParameters`) on data
// unlike the main fixtures: the situations an independent reviewer found wrong.

type Policy = KeplerOperationDescriptor['parameters'][number]['defaultPolicy'];

const descriptorOf = (id: string) =>
  ALL_DESCRIPTORS.find(descriptor => descriptor.bindingId === id)!;
const parameterOf = (id: string, name: string) =>
  descriptorOf(id).parameters.find(parameter => parameter.name === name)!;

describe('policies stated by the descriptors', () => {
  it('presets 999 permutations when the binding defaults lower or already matches', () => {
    for (const [id, defaultPermutations] of [
      ['local-moran-statistics', 999],
      ['local-geary-statistics', 99],
      ['local-lisa-variants', 199],
      ['knox-test', 199]
    ] as const) {
      const policy = parameterOf(id, 'permutations').defaultPolicy;
      expect(policy.from, id).toBe('preset');
      if (policy.from === 'preset') {
        expect(policy.value).toBe(999);
        expect(policy.reason.length).toBeGreaterThan(20);
        const spec = getCatalogBinding(id)!.parameters.find(
          parameter => parameter.name === 'permutations'
        )!;
        expect(Number(spec.defaultValue), id).toBe(defaultPermutations);
        expect(spec.maximum, id).toBeGreaterThanOrEqual(999);
      }
    }
  });

  it('presets queen contiguity for polygons and Benjamini-Hochberg for the lattice', () => {
    for (const id of [
      'spatial-weights',
      'global-spatial-statistics',
      'local-moran-statistics',
      'hotspot-gi-star',
      'local-geary-statistics',
      'global-join-counts-getis-ord',
      'local-lisa-variants'
    ]) {
      const policy = parameterOf(id, 'neighborhood').defaultPolicy;
      expect(policy, id).toMatchObject({from: 'preset', value: 'queen', whenGeometry: 'polygon'});
    }
    expect(parameterOf('recipe-hot-spot-lattice', 'multipleTesting').defaultPolicy).toEqual({
      from: 'constant'
    });
    expect(
      getCatalogDefaultParameters(getCatalogBinding('recipe-hot-spot-lattice')!)['multipleTesting']
    ).toBe('benjamini-hochberg');
    expect(parameterOf('optics-ordering', 'minimumSamples').defaultPolicy).toMatchObject({
      from: 'preset',
      value: 4
    });
    expect(
      parameterOf('hierarchical-density-clustering', 'minimumSamples').defaultPolicy
    ).toMatchObject({from: 'preset', value: 3});
  });

  it('switches the weights parameters on and off with the neighborhood', () => {
    for (const id of [
      'spatial-weights',
      'global-spatial-statistics',
      'local-moran-statistics',
      'hotspot-gi-star',
      'local-geary-statistics',
      'global-join-counts-getis-ord',
      'local-lisa-variants',
      'skater-regions'
    ]) {
      expect(parameterOf(id, 'neighborCount').activeWhen, id).toEqual({
        parameter: 'neighborhood',
        equals: ['knn']
      });
      expect(parameterOf(id, 'distanceBand').activeWhen, id).toEqual({
        parameter: 'neighborhood',
        equals: ['distance-band']
      });
    }
    expect(parameterOf('weights-algebra', 'weightRule').activeWhen).toMatchObject({
      parameter: 'operation'
    });
  });

  it('derives the distance bands with the outlier-resistant rule and the radii with the knee', () => {
    for (const id of ['hotspot-gi-star', 'weights-algebra', 'skater-regions']) {
      const policy = parameterOf(id, 'distanceBand').defaultPolicy;
      expect(policy, id).toMatchObject({
        from: 'data',
        rule: 'outlier-resistant-nearest-neighbor-distance',
        factor: 1.01
      });
    }
    for (const [id, name] of [
      ['density-clustering', 'epsilon'],
      ['optics-ordering', 'extractionDistance'],
      ['hierarchical-density-clustering', 'cutDistance']
    ] as const) {
      expect(parameterOf(id, name).defaultPolicy, id).toMatchObject({
        from: 'data',
        rule: 'k-distance-knee',
        factor: 1.01
      });
    }
  });

  it('carries the canonical bandwidth scale of each kernel, kriging sill from the column variance', () => {
    const policy: Policy = parameterOf('kernel-density-grid', 'bandwidth').defaultPolicy;
    expect(policy.from).toBe('data');
    if (policy.from === 'data') {
      expect(policy.factorByParameter?.parameter).toBe('kernel');
      const factors = policy.factorByParameter!.factors;
      expect(factors['gaussian']).toBe(1);
      expect(factors['epanechnikov']).toBeCloseTo(192 ** (1 / 6), 12);
      expect(factors['quartic']).toBeCloseTo((2304 / 5) ** (1 / 6), 12);
      expect(factors['quartic']).toBeGreaterThan(factors['epanechnikov']);
    }
    expect(parameterOf('kriging-grid', 'sill').defaultPolicy).toMatchObject({
      from: 'data',
      rule: 'column-variance',
      role: 'value'
    });
    for (const [id, name, rule] of [
      ['knox-test', 'temporalThreshold', 'column-range'],
      ['knox-test', 'spatialThreshold', 'median-nearest-neighbor-distance'],
      ['regression-gwr', 'bandwidth', 'standard-distance']
    ] as const) {
      expect(parameterOf(id, name).defaultPolicy, `${id}.${name}`).toMatchObject({
        from: 'data',
        rule
      });
    }
    // Roles named by a column rule exist on the input the rule reads.
    for (const descriptor of ALL_DESCRIPTORS) {
      for (const parameter of descriptor.parameters) {
        const policy = parameter.defaultPolicy;
        if (policy.from === 'data' && policy.role) {
          const input = descriptor.inputs.find(candidate => candidate.name === policy.input)!;
          expect(
            input.fields.some(field => field.role === policy.role),
            `${descriptor.bindingId}.${parameter.name}`
          ).toBe(true);
        }
      }
    }
  });

  it('chooses the areal variable type from the column measurement', () => {
    expect(parameterOf('areal-interpolation', 'variableType').defaultPolicy).toEqual({
      from: 'field-measurement',
      input: 'sources',
      role: 'value',
      values: {rate: 'intensive', count: 'extensive', amount: 'extensive'}
    });
  });

  it('lists only rates for polygon input of the other area statistics too', () => {
    for (const [id, input, role] of [
      ['similar-locations', 'features', 'attributes'],
      ['skater-regions', 'features', 'attributes'],
      ['local-lisa-variants', 'features', 'secondValue']
    ] as const) {
      const field = descriptorOf(id)
        .inputs.find(candidate => candidate.name === input)!
        .fields.find(candidate => candidate.role === role)!;
      expect(field.measurementByGeometry?.polygon, id).toEqual(['rate']);
    }
  });

  it('states the regression requirements: constant response blocks, missing values warn', () => {
    for (const id of ['regression-ols', 'regression-gwr', 'regression-probit', 'regression-sur']) {
      const requirements = descriptorOf(id).requirements;
      expect(
        requirements.some(
          requirement =>
            requirement.id === 'varying-values' &&
            requirement.role === 'response' &&
            requirement.severity === 'block'
        ),
        id
      ).toBe(true);
      expect(
        requirements.some(
          requirement => requirement.id === 'no-missing-values' && requirement.severity === 'warn'
        ),
        id
      ).toBe(true);
    }
    for (const id of ['regression-ols', 'regression-gwr']) {
      const measurement = descriptorOf(id).inputs[0].fields[0].measurement!;
      expect(measurement, `${id}: a count draws the warning`).not.toContain('count');
    }
  });

  it('carries notes for the differences a user must read', () => {
    for (const id of [
      'local-moran-statistics',
      'kriging-grid',
      'hierarchical-density-clustering',
      'hotspot-gi-star',
      'regression-ols',
      'local-outlier-factor',
      'density-clustering',
      'similar-locations',
      'empirical-bayes-rates',
      'location-allocation'
    ]) {
      expect(descriptorOf(id).notes?.length, id).toBeGreaterThan(0);
    }
    expect(descriptorOf('local-moran-statistics').notes!.join(' ')).toMatch(/999/);
    expect(descriptorOf('density-clustering').notes!.join(' ')).toMatch(/6,324/);
    expect(descriptorOf('kriging-grid').action).toBe('Krige with a stated variogram');
    expect(descriptorOf('hierarchical-density-clustering').action).toBe(
      'Cluster by density (one cut)'
    );
    const kriging = descriptorOf('kriging-grid').parameters;
    expect(kriging.find(parameter => parameter.name === 'sill')!.level).toBe('basic');
    expect(kriging.find(parameter => parameter.name === 'nugget')!.level).toBe('basic');
    expect(kriging.find(parameter => parameter.name === 'cellSize')!.level).toBe('advanced');
  });
});

function gaussianDraw(random: () => number): number {
  return Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
}

function planarPoints(
  positions: [number, number][],
  properties: (index: number) => Properties = () => ({})
): CatalogFeatureCollection {
  return makeCollection(
    positions.map((position, index) => ({
      type: 'Feature',
      geometry: {type: 'Point', coordinates: position},
      properties: properties(index)
    }))
  );
}

function resolveThroughEngine(
  id: string,
  inputs: Record<string, CatalogFeatureCollection>,
  coordinateSystem: 'EPSG:4326' | 'planar',
  fields: Record<string, string> = {},
  overrides: Record<string, string | number> = {}
) {
  const profiles = Object.fromEntries(
    Object.entries(inputs).map(([name, collection]) => [
      name,
      profileKeplerFeatureCollection(collection, {datasetId: name, label: name, coordinateSystem})
    ])
  );
  return resolveKeplerOperationParameters(
    getCatalogBinding(id)!,
    descriptorOf(id),
    profiles as never,
    {
      fieldBindings: fields,
      overrides,
      maximumCellCount: KEPLER_GRID_CELL_CAPS[id]?.maximumCellCount
    }
  );
}

const explanationOf = (resolved: ReturnType<typeof resolveThroughEngine>, name: string) =>
  resolved.resolutions.find(entry => entry.parameter === name)?.resolution?.explanation ?? '';

function runResolved(
  id: string,
  resolved: ReturnType<typeof resolveThroughEngine>,
  inputs: Record<string, CatalogFeatureCollection>,
  extra: Record<string, string | number> = {}
) {
  return runCatalogOperation(getCatalogBinding(id)!, {...resolved.parameters, ...extra}, inputs);
}

describe('defaults resolved through the engine run on every fixture', () => {
  for (const fixture of [GEOGRAPHIC, PLANAR]) {
    const runCases = makeRunCases(fixture);
    for (const descriptor of ALL_DESCRIPTORS) {
      const id = descriptor.bindingId;
      if (id in GPU_ONLY_IDS || !runCases[id]) continue;
      if (descriptor.parameters.every(parameter => parameter.defaultPolicy.from === 'constant')) {
        continue;
      }
      it(`${id} (${fixture.name}): runs with the resolved parameters`, () => {
        const runCase = runCases[id];
        const resolved = resolveThroughEngine(
          id,
          runCase.inputs,
          fixture.coordinateSystem,
          runCase.fields,
          runCase.extra ?? {}
        );
        const outcome = runCatalogOperation(
          getCatalogBinding(id)!,
          resolved.parameters,
          runCase.inputs
        );
        expect(outcome.tables.length).toBeGreaterThan(0);
        expect(outcome.tables[0].rowCount).toBeGreaterThan(0);
      });
    }

    it(`a distance band at the resolved value leaves no feature without a neighbor (${fixture.name})`, () => {
      const inputs = {features: fixture.lattice};
      const resolved = resolveThroughEngine(
        'hotspot-gi-star',
        inputs,
        fixture.coordinateSystem,
        {
          valueProperty: 'value'
        },
        {neighborhood: 'distance-band'}
      );
      const band = resolved.parameters['distanceBand'] as number;
      const weights = (distanceBand: number) =>
        runCatalogOperation(
          getCatalogBinding('spatial-weights')!,
          {neighborhood: 'distance-band', distanceBand, coordinateSystem: fixture.coordinateSystem},
          inputs
        );
      expect(weights(band).summary['islandCount']).toBe(0);
      expect(weights(band * 0.9).summary['islandCount']).toBeGreaterThan(0);
    });

    it(`density clustering at the resolved knee finds the three planted clusters (${fixture.name})`, () => {
      const inputs = {points: fixture.points};
      const resolved = resolveThroughEngine('density-clustering', inputs, fixture.coordinateSystem);
      const outcome = runResolved('density-clustering', resolved, inputs);
      const labels = new Set(
        (outcome.tables[0].columns['cluster'] as number[]).filter(label => label >= 0)
      );
      expect(labels.size, explanationOf(resolved, 'epsilon')).toBe(3);
    });
  }
});

describe('situations found by an independent reviewer', () => {
  it('allows signed bivariate Moran values and applies the G and join checks only to their variants', () => {
    const binding = getCatalogBinding('local-lisa-variants')!;
    const descriptor = descriptorOf(binding.id);
    const profile = profileKeplerFeatureCollection(
      planarPoints(
        Array.from({length: 36}, (_, index) => [index % 6, Math.floor(index / 6)]),
        index => ({rate: index - 18, secondRate: 18 - index})
      ),
      {datasetId: 'signed', label: 'Signed values', coordinateSystem: 'planar'}
    );
    for (const variant of [
      'local-bivariate-moran',
      'local-getis-ord-g',
      'local-getis-ord-g-star',
      'local-join-count'
    ]) {
      const applicability = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
        fieldBindings: {valueProperty: 'rate', secondValueProperty: 'secondRate'},
        overrides: {variant}
      });
      const getisOrd = variant === 'local-getis-ord-g' || variant === 'local-getis-ord-g-star';
      expect(
        applicability.requirements.some(requirement => requirement.id === 'non-negative-values'),
        variant
      ).toBe(getisOrd);
      expect(
        applicability.requirements.some(
          requirement => requirement.id === 'minimum-distinct-values'
        ),
        variant
      ).toBe(variant === 'local-join-count');
      if (getisOrd) expect(applicability.status, variant).toBe('blocked');
      if (variant === 'local-bivariate-moran') expect(applicability.status).toBe('ready');
    }
  });

  it('states the corrected count test, default correction and cell-size sensitivity', () => {
    const descriptor = descriptorOf('recipe-hot-spot-lattice');
    const notes = descriptor.notes!.join(' ');
    expect(notes).toMatch(/conservative binomial tails/);
    expect(notes).toMatch(/partial boundary-cell exposure/);
    expect(notes).toMatch(/Benjamini-Hochberg is the default/);
    expect(notes).toMatch(/overlapping neighborhoods share counts/);
    expect(notes).toMatch(/half the size/);
    const policy = parameterOf(descriptor.bindingId, 'multipleTesting').defaultPolicy;
    expect(policy).toEqual({from: 'constant'});
  });

  it('separates kriging variance units from calibration', () => {
    expect(descriptorOf('kriging-grid').notes).toContain(
      'The variance uses the squared units specified by the variogram. Its calibration depends on how well that variogram describes the data.'
    );
  });

  it('declares overlapping areal coverage and the input geometry retained by feature outputs', () => {
    expect(descriptorOf('areal-interpolation').overlappingInputs).toEqual([['sources', 'targets']]);
    for (const [bindingId, tableId, input] of [
      ['local-moran-statistics', 'local-moran', 'features'],
      ['hotspot-gi-star', 'hotspots', 'features'],
      ['local-geary-statistics', 'local-geary', 'features'],
      ['empirical-bayes-rates', 'rates', 'features'],
      ['multiple-testing-adjustment', 'adjusted', 'features'],
      ['similar-locations', 'similarity', 'features'],
      ['local-lisa-variants', 'local-lisa', 'features'],
      ['location-allocation', 'facilities', 'candidates'],
      ['regression-ols', 'ols-residuals', 'points'],
      ['regression-gwr', 'gwr-local', 'points'],
      ['regression-probit', 'probit-fitted', 'points'],
      ['regression-sur', 'sur-residuals', 'points'],
      ['density-clustering', 'points', 'points'],
      ['optics-ordering', 'points', 'points'],
      ['local-outlier-factor', 'outliers', 'points'],
      ['hierarchical-density-clustering', 'points', 'points'],
      ['skater-regions', 'regions', 'features'],
      ['areal-interpolation', 'targets', 'targets']
    ]) {
      expect(
        descriptorOf(bindingId).outputs.find(output => output.tableId === tableId)?.kindFrom,
        `${bindingId}.${tableId}`
      ).toBe(input);
    }
    for (const bindingId of [
      'density-clustering',
      'optics-ordering',
      'hierarchical-density-clustering'
    ]) {
      expect(
        descriptorOf(bindingId).outputs.find(output => output.tableId === 'clusters')?.kindFrom
      ).toBeUndefined();
    }
    expect(descriptorOf('recipe-hot-spot-lattice').outputs[0].kindFrom).toBeUndefined();
    expect(descriptorOf('location-allocation').overlappingInputs).toBeUndefined();
  });

  it('compares lattice factors against null runs and planted 50-point clusters', () => {
    // Forty paired seeds per factor and size assess discovery frequency, not the number of
    // flagged cells. This is a finite regression envelope, not nominal 10% error control.
    const seedCount = 40;
    // Allow three runs above the largest measured 11/40, while rejecting a material worsening.
    const maximumNullDiscoveryShare = 0.35;
    const measurements: {points: number; factor: number; nullShare: number; powerShare: number}[] =
      [];
    for (const count of [300, 1000, 5000]) {
      const discoveries = [0, 0, 0];
      const detections = [0, 0, 0];
      for (let seed = 0; seed < seedCount; seed++) {
        const nullRandom = makeRandom(19000 + seed);
        const plantedRandom = makeRandom(32000 + seed);
        const nullPositions = Array.from(
          {length: count},
          () => [nullRandom() * 1000, nullRandom() * 1000] as [number, number]
        );
        const plantedPositions = Array.from(
          {length: count - 50},
          () => [plantedRandom() * 1000, plantedRandom() * 1000] as [number, number]
        );
        for (let index = 0; index < 50; index++) {
          // Same sine Box-Muller draw as review c5, including the two original missed seeds.
          const draw = () =>
            Math.sqrt(-2 * Math.log(Math.max(1e-12, plantedRandom()))) *
            Math.sin(2 * Math.PI * plantedRandom());
          plantedPositions.push([400 + draw() * 10, 400 + draw() * 10]);
        }
        for (const [factorIndex, factor] of [1.5, 2, 2.5].entries()) {
          const descriptor = descriptorOf('recipe-hot-spot-lattice');
          const candidate = {
            ...descriptor,
            parameters: descriptor.parameters.map(parameter =>
              parameter.name === 'cellSize' && parameter.defaultPolicy.from === 'data'
                ? {...parameter, defaultPolicy: {...parameter.defaultPolicy, factor}}
                : parameter
            )
          };
          for (const [planted, positions] of [
            [false, nullPositions],
            [true, plantedPositions]
          ] as const) {
            const inputs = {points: planarPoints(positions)};
            const profile = profileKeplerFeatureCollection(inputs.points, {
              datasetId: 'points',
              label: 'Points',
              coordinateSystem: 'planar'
            });
            const resolved = resolveKeplerOperationParameters(
              getCatalogBinding(descriptor.bindingId)!,
              candidate,
              {points: profile},
              {executionEngine: 'cpu'}
            );
            expect(resolved.parameters['multipleTesting']).toBe('benjamini-hochberg');
            const table = runResolved(descriptor.bindingId, resolved, inputs).tables[0];
            const bins = table.columns['bin'] as number[];
            if (!planted && bins.some(bin => bin !== 0)) discoveries[factorIndex]++;
            if (planted) {
              const minimumX = Math.min(...positions.map(position => position[0]));
              const minimumY = Math.min(...positions.map(position => position[1]));
              const cellSize = resolved.parameters['cellSize'] as number;
              const centerColumn = Math.floor((400 - minimumX) / cellSize);
              const centerRow = Math.floor((400 - minimumY) / cellSize);
              // Find a hot cell whose radius-one neighbourhood includes the planted centre;
              // a remote false positive must not count as finding the cluster.
              if (
                bins.some(
                  (bin, index) =>
                    bin > 0 &&
                    Math.abs(Number(table.columns['column'][index]) - centerColumn) <= 1 &&
                    Math.abs(Number(table.columns['row'][index]) - centerRow) <= 1
                )
              )
                detections[factorIndex]++;
            }
          }
        }
      }
      for (const [factorIndex, factor] of [1.5, 2, 2.5].entries()) {
        const measurement = {
          points: count,
          factor,
          nullShare: discoveries[factorIndex] / seedCount,
          powerShare: detections[factorIndex] / seedCount
        };
        measurements.push(measurement);
        expect(measurement.nullShare, JSON.stringify(measurement)).toBeLessThanOrEqual(
          maximumNullDiscoveryShare
        );
      }
    }
    const policy = parameterOf('recipe-hot-spot-lattice', 'cellSize').defaultPolicy;
    expect(policy.from).toBe('data');
    if (policy.from === 'data') {
      for (const measurement of measurements.filter(
        candidate => candidate.factor === policy.factor
      )) {
        // At least 95% of these planted clusters must be found at the selected default.
        // The old factor 2 misses four of the forty at 300 points and fails this bound.
        expect(measurement.powerShare, JSON.stringify(measurement)).toBeGreaterThanOrEqual(0.95);
      }
      expect(measurements.filter(candidate => candidate.factor === policy.factor)).toHaveLength(3);
    }
    process.stdout.write(
      `AX lattice factor comparison (40 seeds): ${JSON.stringify(measurements)}\n`
    );
  }, 120000);

  it('geographic lattices: the three clustering bindings do not turn everything into noise', () => {
    for (const [latitude, step] of [
      [51.5, 0.001],
      [65, 0.001],
      [51.5, 0.0009],
      [0, 0.0005]
    ] as const) {
      const positions: [number, number][] = [];
      for (let i = 0; i < 20; i++)
        for (let j = 0; j < 20; j++) positions.push([-0.1 + i * step, latitude + j * step]);
      const inputs = {points: planarPoints(positions, index => ({value: index % 7}))};
      for (const id of [
        'density-clustering',
        'optics-ordering',
        'hierarchical-density-clustering'
      ]) {
        const resolved = resolveThroughEngine(id, inputs, 'EPSG:4326');
        const outcome = runResolved(id, resolved, inputs);
        const labels = outcome.tables[0].columns['cluster'] as number[];
        const noise = labels.filter(label => label < 0).length / labels.length;
        expect(noise, `${id} at ${latitude}, step ${step}`).toBeLessThan(0.5);
        expect(Math.max(...labels), `${id} finds a cluster`).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it('one far outlier does not give every feature hundreds of neighbors', () => {
    const random = makeRandom(5);
    const positions: [number, number][] = [];
    for (const [centerX, centerY] of [
      [500, 500],
      [1500, 600],
      [900, 1600]
    ])
      for (let i = 0; i < 100; i++)
        positions.push([centerX + 50 * gaussianDraw(random), centerY + 50 * gaussianDraw(random)]);
    positions.push([40000, 40000]);
    const inputs = {features: planarPoints(positions, () => ({value: random() * 10}))};
    const resolved = resolveThroughEngine(
      'hotspot-gi-star',
      inputs,
      'planar',
      {valueProperty: 'value'},
      {neighborhood: 'distance-band'}
    );
    const outcome = runResolved('hotspot-gi-star', resolved, inputs);
    const counts = [...(outcome.tables[0].columns['neighbor_count'] as number[])].sort(
      (a, b) => a - b
    );
    // The largest nearest-neighbor distance (35,000 m here) gave every feature 299 neighbors.
    expect(counts[Math.floor(counts.length / 2)]).toBeLessThan(60);
    // The fence is an unscaled statistic; final margin/rounding can change isolation.
    expect(explanationOf(resolved, 'distanceBand')).toMatch(/beyond the unscaled spacing fence/);
    const distanceBand = Number(resolved.parameters['distanceBand']);
    const expectedIslands = positions.filter((position, index) =>
      positions.every(
        (other, otherIndex) =>
          index === otherIndex ||
          Math.hypot(position[0] - other[0], position[1] - other[1]) > distanceBand
      )
    ).length;
    expect(expectedIslands).toBeGreaterThan(0);
    expect(counts.filter(count => count === 0)).toHaveLength(expectedIslands);
  });

  it('polygons of very unequal size get queen neighbors (3 to 8)', () => {
    const geometric = (count: number, ratio: number) => {
      const out = [0];
      let width = 100;
      for (let i = 0; i < count; i++) {
        out.push(out[i] + width);
        width *= ratio;
      }
      return out;
    };
    const xs = geometric(18, 1.35);
    const random = makeRandom(4);
    const features: unknown[] = [];
    for (let j = 0; j < 18; j++)
      for (let i = 0; i < 18; i++)
        features.push({
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [xs[i], xs[j]],
                [xs[i + 1], xs[j]],
                [xs[i + 1], xs[j + 1]],
                [xs[i], xs[j + 1]],
                [xs[i], xs[j]]
              ]
            ]
          },
          properties: {rate: 10 + gaussianDraw(random)}
        });
    const inputs = {features: makeCollection(features)};
    const resolved = resolveThroughEngine('hotspot-gi-star', inputs, 'planar', {
      valueProperty: 'rate'
    });
    expect(resolved.parameters['neighborhood']).toBe('queen');
    const outcome = runResolved('hotspot-gi-star', resolved, inputs);
    const counts = outcome.tables[0].columns['neighbor_count'] as number[];
    expect(Math.min(...counts)).toBeGreaterThanOrEqual(3);
    expect(Math.max(...counts)).toBeLessThanOrEqual(8);
  });

  it('the lattice cell follows point density: about 2.25 points per cell', () => {
    const random = makeRandom(3);
    const positions = Array.from(
      {length: 600},
      () => [random() * 8000, random() * 8000] as [number, number]
    );
    const inputs = {points: planarPoints(positions)};
    const resolved = resolveThroughEngine('recipe-hot-spot-lattice', inputs, 'planar');
    const meanSpacing = Math.sqrt((8000 * 8000) / 600);
    expect(resolved.parameters['cellSize'] as number).toBeGreaterThan(1.35 * meanSpacing);
    expect(resolved.parameters['cellSize'] as number).toBeLessThan(1.65 * meanSpacing);
  });

  it('the lattice finds the three strong planted clusters', () => {
    const random = makeRandom(104729);
    const positions: [number, number][] = [];
    for (const [centerX, centerY] of [
      [2000, 2000],
      [6000, 2500],
      [4000, 6000]
    ])
      for (let i = 0; i < 100; i++)
        positions.push([
          centerX + 150 * gaussianDraw(random),
          centerY + 150 * gaussianDraw(random)
        ]);
    for (let i = 0; i < 300; i++) positions.push([random() * 8000, random() * 8000]);
    const inputs = {points: planarPoints(positions)};
    const resolved = resolveThroughEngine('recipe-hot-spot-lattice', inputs, 'planar');
    const bins = runResolved('recipe-hot-spot-lattice', resolved, inputs).tables[0].columns[
      'bin'
    ] as number[];
    expect(bins.filter(bin => bin > 0).length).toBeGreaterThanOrEqual(5);
  });

  it('kriging white noise gives a variance in squared data units', () => {
    const random = makeRandom(6);
    const values: number[] = [];
    const inputs = {
      points: planarPoints(
        Array.from({length: 120}, () => [random() * 4000, random() * 4000] as [number, number]),
        () => {
          const value = 50 + 10 * gaussianDraw(random);
          values.push(value);
          return {value};
        }
      )
    };
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance =
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
    const resolved = resolveThroughEngine('kriging-grid', inputs, 'planar', {
      valueProperty: 'value'
    });
    expect(resolved.parameters['sill'] as number).toBeGreaterThan(0.5 * variance);
    expect(resolved.parameters['sill'] as number).toBeLessThan(1.5 * variance);
    const table = runResolved('kriging-grid', resolved, inputs).tables[0];
    const kriged = [...(table.columns['variance'] as number[])].sort((a, b) => a - b);
    const median = kriged[Math.floor(kriged.length / 2)];
    // With the sill of 1 the variance was 0.002 to 0.5 for data of variance 100.
    expect(median).toBeGreaterThan(0.05 * variance);
    expect(median).toBeLessThan(2 * variance);
  });

  it('Knox in epoch milliseconds detects the planted interaction', () => {
    const random = makeRandom(2);
    const day = 86400000;
    const start = Date.parse('2026-01-01');
    const positions: [number, number][] = [];
    const times: number[] = [];
    for (const [centerX, centerY, centerTime] of [
      [-0.1, 51.5, 0],
      [-0.05, 51.52, 10],
      [-0.12, 51.48, 20]
    ])
      for (let i = 0; i < 20; i++) {
        positions.push([
          centerX + 0.002 * gaussianDraw(random),
          centerY + 0.002 * gaussianDraw(random)
        ]);
        times.push(start + (centerTime + random()) * day);
      }
    const inputs = {points: planarPoints(positions, index => ({t: times[index]}))};
    const resolved = resolveThroughEngine('knox-test', inputs, 'EPSG:4326', {timeProperty: 't'});
    // The constants 1,000 m and 1 (millisecond) found no joint pairs and gave p = 1.
    expect(resolved.parameters['temporalThreshold'] as number).toBeGreaterThan(day / 2);
    const table = runResolved('knox-test', resolved, inputs).tables[0];
    expect((table.columns['p_value'] as number[])[0]).toBeLessThanOrEqual(0.05);
  });

  it('GWR at the resolved bandwidth has no singular local fit, at any scale or shape', () => {
    for (const scale of [200, 20000, 2000000])
      for (const clustered of [false, true])
        for (const seed of [1, 2, 3]) {
          const random = makeRandom(seed);
          const positions: [number, number][] = [];
          if (clustered) {
            for (const [centerX, centerY] of [
              [0.25, 0.25],
              [0.7, 0.3],
              [0.4, 0.75]
            ])
              for (let i = 0; i < 33; i++)
                positions.push([
                  (centerX + 0.06 * gaussianDraw(random)) * scale,
                  (centerY + 0.06 * gaussianDraw(random)) * scale
                ]);
          } else {
            for (let i = 0; i < 100; i++) positions.push([random() * scale, random() * scale]);
          }
          const inputs = {
            points: planarPoints(positions, index => ({
              y: 1 + (positions[index][0] / scale) * 5 + random(),
              x1: (positions[index][0] / scale) * 5 + random(),
              x2: random() * 5
            }))
          };
          const resolved = resolveThroughEngine('regression-gwr', inputs, 'planar', {
            responseProperty: 'y',
            predictorProperties: 'x1,x2'
          });
          const outcome = runResolved('regression-gwr', resolved, inputs);
          expect(
            outcome.summary['singularCount'],
            `scale ${scale}, clustered ${clustered}, seed ${seed}, bandwidth ${resolved.parameters['bandwidth']}`
          ).toBe(0);
        }
  });

  it('two halves with a rate column interpolate as intensive', () => {
    const half = (x0: number, x1: number, rate: number) => ({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [x0, 0],
            [x1, 0],
            [x1, 10],
            [x0, 10],
            [x0, 0]
          ]
        ]
      },
      properties: {rate, count: 100}
    });
    const inputs = {
      sources: makeCollection([half(0, 10, 10), half(10, 20, 20)]),
      targets: makeCollection([half(0, 20, 0)])
    };
    const resolved = resolveThroughEngine('areal-interpolation', inputs, 'planar', {
      valueProperty: 'rate'
    });
    expect(resolved.parameters['variableType']).toBe('intensive');
    const value = (
      runResolved('areal-interpolation', resolved, inputs).tables[0].columns['value'] as number[]
    )[0];
    // The extensive form summed the two rates to 30.
    expect(value).toBeCloseTo(15, 0);
  });

  it('the kernel bandwidth follows the kernel', () => {
    const random = makeRandom(2);
    const inputs = {
      points: planarPoints(
        Array.from(
          {length: 300},
          () => [1000 * gaussianDraw(random), 1000 * gaussianDraw(random)] as [number, number]
        )
      )
    };
    const bandwidth = (kernel: string) =>
      resolveThroughEngine('kernel-density-grid', inputs, 'planar', {}, {kernel}).parameters[
        'bandwidth'
      ] as number;
    expect(bandwidth('epanechnikov') / bandwidth('gaussian')).toBeCloseTo(192 ** (1 / 6), 1);
    expect(bandwidth('quartic') / bandwidth('gaussian')).toBeCloseTo((2304 / 5) ** (1 / 6), 1);
  });
});

// ---------------------------------------------------------------------------------------------
// The knee rule declines on about half of real data sets; the fallback must still give a radius
// that follows the data's scale (the constant 200 labelled every point noise).

describe('the radius fallback of the density clustering bindings', () => {
  it('every knee policy names the median nearest-neighbor distance times 3 as its fallback', () => {
    let found = 0;
    for (const descriptor of ALL_DESCRIPTORS) {
      for (const parameter of descriptor.parameters) {
        const policy = parameter.defaultPolicy;
        if (policy.from === 'data' && policy.rule === 'k-distance-knee') {
          found++;
          expect(policy.fallback?.rule, `${descriptor.bindingId}.${parameter.name}`).toBe(
            'median-nearest-neighbor-distance'
          );
          expect(policy.fallback?.factor).toBeGreaterThanOrEqual(3);
        }
      }
    }
    expect(found).toBe(4);
  });

  const makeSets = (): Record<string, [number, number][]> => {
    const sets: Record<string, [number, number][]> = {};
    const uniform = (count: number, seed: number) => {
      const random = makeRandom(seed);
      return Array.from(
        {length: count},
        () => [random() * 1000, random() * 1000] as [number, number]
      );
    };
    sets['uniform 30'] = uniform(30, 2);
    sets['uniform 500'] = uniform(500, 1);
    sets['uniform 3000'] = uniform(3000, 3);
    sets['lattice'] = Array.from({length: 400}, (_, index) => [
      (index % 20) * 50,
      Math.floor(index / 20) * 50
    ]);
    const random = makeRandom(4);
    const planted: [number, number][] = [];
    for (const [x, y] of [
      [200, 200],
      [700, 300],
      [400, 800]
    ])
      for (let i = 0; i < 100; i++)
        planted.push([x + 15 * gaussianDraw(random), y + 15 * gaussianDraw(random)]);
    for (let i = 0; i < 200; i++) planted.push([random() * 1000, random() * 1000]);
    sets['three clusters in noise'] = planted;
    const blobRandom = makeRandom(5);
    sets['gaussian blob'] = Array.from(
      {length: 500},
      () =>
        [500 + 100 * gaussianDraw(blobRandom), 500 + 100 * gaussianDraw(blobRandom)] as [
          number,
          number
        ]
    );
    const lineRandom = makeRandom(6);
    const lines: [number, number][] = [];
    for (let i = 0; i < 200; i++) {
      const t = lineRandom();
      lines.push([t * 1000 + 3 * gaussianDraw(lineRandom), 200 + 3 * gaussianDraw(lineRandom)]);
      lines.push([
        t * 1000 + 3 * gaussianDraw(lineRandom),
        600 + 0.3 * t * 1000 + 3 * gaussianDraw(lineRandom)
      ]);
      lines.push([800 + 3 * gaussianDraw(lineRandom), lineRandom() * 1000]);
    }
    sets['points along lines'] = lines;
    return sets;
  };

  const run = (positions: [number, number][], coordinateSystem: 'planar' | 'EPSG:4326') => {
    const inputs = {points: planarPoints(positions)};
    const resolved = resolveThroughEngine('density-clustering', inputs, coordinateSystem);
    const labels = runResolved('density-clustering', resolved, inputs).tables[0].columns[
      'cluster'
    ] as number[];
    const sizes = new Map<number, number>();
    for (const label of labels) if (label >= 0) sizes.set(label, (sizes.get(label) ?? 0) + 1);
    return {
      resolved,
      clusterCount: sizes.size,
      noise: labels.filter(label => label < 0).length / labels.length,
      largest: Math.max(0, ...sizes.values()) / labels.length
    };
  };

  it('the engine resolves presets and tries the data fallback', () => {
    expect(
      resolveThroughEngine('local-moran-statistics', {features: LATTICE}, 'EPSG:4326', {
        valueProperty: 'value'
      }).parameters['permutations']
    ).toBe(999);
    const resolved = run(makeSets()['uniform 500'], 'planar').resolved;
    expect(resolved.parameters['epsilon']).not.toBe(200);
    expect(
      resolved.resolutions.find(resolution => resolution.parameter === 'epsilon')?.source
    ).toBe('data');
    expect(
      resolved.resolutions.find(resolution => resolution.parameter === 'epsilon')?.resolution?.rule
    ).toBe('median-nearest-neighbor-distance');
  });

  // Waits for the spacing/default-rule owner to derive the median from distinct locations
  // when repeats make the ordinary median zero. Keep this regression unconditional.
  it('PENDING distinct-location median rule: duplicate-heavy locations keep their radius and partition proportional at three scales', () => {
    for (const bindingId of [
      'density-clustering',
      'optics-ordering',
      'hierarchical-density-clustering'
    ]) {
      const radiusParameter =
        bindingId === 'density-clustering'
          ? 'epsilon'
          : bindingId === 'optics-ordering'
            ? 'extractionDistance'
            : 'cutDistance';
      const radii: number[] = [];
      const partitions: number[][] = [];
      for (const scale of [0.001, 1, 1000]) {
        const positions = Array.from(
          {length: 400},
          (_, index) =>
            [
              (Math.floor(index / 10) % 8) * 100 * scale,
              Math.floor(Math.floor(index / 10) / 8) * 100 * scale
            ] as [number, number]
        );
        const inputs = {points: planarPoints(positions)};
        const resolved = resolveThroughEngine(bindingId, inputs, 'planar');
        radii.push(Number(resolved.parameters[radiusParameter]) / scale);
        expect
          .soft(
            resolved.resolutions.find(resolution => resolution.parameter === radiusParameter)
              ?.source,
            `${bindingId} at scale ${scale}: waits for distinct-location median`
          )
          .toBe('data');
        const labels = runResolved(bindingId, resolved, inputs).tables[0].columns[
          'cluster'
        ] as number[];
        const sizes = new Map<number, number>();
        for (const label of labels) sizes.set(label, (sizes.get(label) ?? 0) + 1);
        partitions.push([...sizes.values()].sort((first, second) => first - second));
      }
      expect.soft(radii[0], bindingId).toBeCloseTo(radii[1], 6);
      expect.soft(radii[2], bindingId).toBeCloseTo(radii[1], 6);
      expect.soft(partitions[0], bindingId).toEqual(partitions[1]);
      expect.soft(partitions[2], bindingId).toEqual(partitions[1]);
    }
  });

  it('the resolved radius is neither all noise nor one cluster holding everything where structure exists', () => {
    const sets = makeSets();
    for (const name of Object.keys(sets)) {
      const result = run(sets[name], 'planar');
      expect(result.noise, `${name}: noise share`).toBeLessThan(0.5);
    }
    const planted = run(sets['three clusters in noise'], 'planar');
    expect(planted.clusterCount).toBeGreaterThanOrEqual(3);
    expect(planted.largest).toBeLessThan(0.4);
    expect(planted.noise).toBeGreaterThan(0.25);
    const lines = run(sets['points along lines'], 'planar');
    expect(lines.clusterCount).toBeGreaterThanOrEqual(3);
    expect(lines.largest).toBeLessThan(0.5);
    // Structureless data is unremarkable: no scatter of dozens of invented clusters.
    expect(run(sets['uniform 500'], 'planar').clusterCount).toBeLessThanOrEqual(10);
    expect(run(sets['uniform 3000'], 'planar').clusterCount).toBeLessThanOrEqual(20);
  });

  it('weather stations and points across the antimeridian are not all noise', () => {
    const random = makeRandom(9);
    const stations: [number, number][] = Array.from(
      {length: 60},
      () => [-5 + random() * 15, 42 + random() * 9] as [number, number]
    );
    const crossing: [number, number][] = Array.from({length: 500}, () => {
      const longitude = 175 + random() * 10;
      return [longitude > 180 ? longitude - 360 : longitude, -17 + random() * 2] as [
        number,
        number
      ];
    });
    for (const [name, positions] of [
      ['60 weather stations', stations],
      ['500 points across the antimeridian', crossing]
    ] as const) {
      const result = run(positions, 'EPSG:4326');
      expect(result.noise, name).toBeLessThan(0.5);
    }
  });
});
