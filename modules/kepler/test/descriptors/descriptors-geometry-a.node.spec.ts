// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_ANALYSIS_DEFAULT_RULES,
  GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  getGPUGeometryOrientationParameterValues,
  getGPULineSimplificationParameterValues,
  resolveGPUAnalysisDefault,
  validateGPUAnalysisResultEncoding
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {describe, expect, it} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {GEOMETRY_A_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-geometry-a';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';

const EXPECTED_IDS = [
  'geometry-validity',
  'geometry-measures',
  'geometry-point-buffer',
  'geometry-line-segmentize',
  'geometry-convex-clip',
  'line-simplification',
  'line-smooth',
  'line-precision-reduction',
  'line-merge',
  'line-chunk',
  'line-locate',
  'linear-referencing',
  'group-convex-hull',
  'nearest-feature-join',
  'point-in-polygon-join',
  'buffer-selection',
  'geodesic-destination',
  'geodesic-pairs',
  'geometry-orientation',
  'affine-transform',
  'vertex-snap',
  'grid-generator',
  'random-points-in-polygon',
  'random-points-on-line',
  'k-means',
  'geometry-line-buffer',
  'geometry-polygon-buffer',
  'geometry-overlay-intersection',
  'geometry-overlay-union',
  'geometry-overlay-difference',
  'geometry-topology-repair',
  'geometry-shape-descriptors'
];

// Bindings with engine 'gpu' have no CPU engine and cannot run in node. Their output columns are
// read from `catalog-gpu-geometry.ts` (the `makeCatalogTable` calls of the buffer, overlay and
// repair runners) and listed here so a presentation column that does not exist still fails.
const GPU_ONLY_COLUMNS: Record<string, Record<string, readonly string[]>> = {
  'geometry-line-buffer': {
    buffers: [
      'buffer_id',
      'component_index',
      'source_index',
      'source_id',
      'distance',
      'area',
      'ring_count',
      'hole_count',
      'geometry'
    ]
  },
  'geometry-polygon-buffer': {
    buffers: [
      'buffer_id',
      'component_index',
      'source_index',
      'source_id',
      'distance',
      'area',
      'ring_count',
      'hole_count',
      'geometry'
    ]
  },
  'geometry-overlay-intersection': {
    overlay: ['overlay_id', 'component_index', 'area', 'ring_count', 'hole_count', 'geometry']
  },
  'geometry-overlay-union': {
    overlay: ['overlay_id', 'component_index', 'area', 'ring_count', 'hole_count', 'geometry']
  },
  'geometry-overlay-difference': {
    overlay: ['overlay_id', 'component_index', 'area', 'ring_count', 'hole_count', 'geometry']
  },
  'geometry-topology-repair': {
    repaired: [
      'source_index',
      'source_id',
      'status',
      'polygon_count',
      'input_area',
      'repaired_area',
      'geometry'
    ]
  }
};

type Properties = Record<string, unknown>;

function makePoint(x: number, y: number, properties: Properties = {}) {
  return {
    type: 'Feature' as const,
    geometry: {type: 'Point', coordinates: [x, y]},
    properties
  };
}

function makeLine(coordinates: number[][]) {
  return {
    type: 'Feature' as const,
    geometry: {type: 'LineString', coordinates},
    properties: {}
  };
}

function makeSquareRing(x: number, y: number, size: number): number[][] {
  return [
    [x, y],
    [x + size, y],
    [x + size, y + size],
    [x, y + size],
    [x, y]
  ];
}

function makePolygon(rings: number[][][]) {
  return {
    type: 'Feature' as const,
    geometry: {type: 'Polygon', coordinates: rings},
    properties: {}
  };
}

function makeCollection(features: unknown[]): CatalogFeatureCollection {
  return {type: 'FeatureCollection', features} as CatalogFeatureCollection;
}

// Two groups (`kind` = a or b) of three points each, near 10E 50N.
const POINTS = makeCollection([
  makePoint(10.0, 50.0, {kind: 'a'}),
  makePoint(10.01, 50.01, {kind: 'a'}),
  makePoint(10.02, 50.0, {kind: 'b'}),
  makePoint(10.03, 50.02, {kind: 'b'}),
  makePoint(10.01, 50.03, {kind: 'a'}),
  makePoint(10.04, 50.01, {kind: 'b'})
]);
// The second line starts where the first ends, so line-merge has something to join.
const LINES = makeCollection([
  makeLine([
    [10, 50],
    [10.01, 50.002],
    [10.02, 50],
    [10.03, 50.001]
  ]),
  makeLine([
    [10.03, 50.001],
    [10.05, 50.0]
  ])
]);
// A square and a square with a square hole.
const POLYGONS = makeCollection([
  makePolygon([makeSquareRing(10, 50, 0.02)]),
  makePolygon([
    makeSquareRing(10.03, 50, 0.02),
    [
      [10.035, 50.005],
      [10.035, 50.01],
      [10.04, 50.01],
      [10.04, 50.005],
      [10.035, 50.005]
    ]
  ])
]);
const CLIP = makeCollection([makePolygon([makeSquareRing(10.005, 50.005, 0.03)])]);

function makeInputs(
  bindingId: string,
  inputNames: readonly string[]
): Record<string, CatalogFeatureCollection> {
  const byName: Record<string, CatalogFeatureCollection> = {
    polygons: POLYGONS,
    lines: LINES,
    points: POINTS,
    selectors: POINTS,
    features: bindingId === 'buffer-selection' ? POINTS : POLYGONS,
    targets: POLYGONS,
    extent: POLYGONS,
    clip: CLIP,
    left: POLYGONS,
    right: CLIP,
    origins: POINTS,
    destinations: POINTS
  };
  return Object.fromEntries(inputNames.map(name => [name, byName[name]]));
}

function getDescriptor(bindingId: string) {
  const descriptor = GEOMETRY_A_OPERATION_DESCRIPTORS.find(item => item.bindingId === bindingId);
  if (!descriptor) throw new Error(`No descriptor for ${bindingId}`);
  return descriptor;
}

describe('geometry-a descriptors', () => {
  it('cover the 32 ids exactly once, in order', () => {
    expect(GEOMETRY_A_OPERATION_DESCRIPTORS.map(item => item.bindingId)).toEqual(EXPECTED_IDS);
    expect(new Set(EXPECTED_IDS).size).toBe(32);
  });

  it('describe only bindings of the geometry domain', () => {
    for (const id of EXPECTED_IDS) expect(getCatalogBinding(id)?.domain).toBe('geometry');
  });

  it('validate against their bindings with no issues', () => {
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId);
      expect(binding, descriptor.bindingId).toBeDefined();
      expect(validateKeplerOperationDescriptor(descriptor, binding!)).toEqual([]);
    }
  });

  it('catches a descriptor that omits a parameter (the validator can fail)', () => {
    const descriptor = getDescriptor('geometry-point-buffer');
    const broken = {
      ...descriptor,
      parameters: descriptor.parameters.filter(item => item.name !== 'radius')
    };
    expect(
      validateKeplerOperationDescriptor(broken, getCatalogBinding('geometry-point-buffer')!)
    ).not.toEqual([]);
  });

  it('mark at most four parameters basic', () => {
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      const basicCount = descriptor.parameters.filter(item => item.level === 'basic').length;
      expect(basicCount, descriptor.bindingId).toBeLessThanOrEqual(4);
    }
  });

  it('state capacity limits that equal the binding input limits', () => {
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      for (const input of binding.inputs) {
        const rows = descriptor.requirements.filter(
          item => item.input === input.name && item.id === 'maximum-rows'
        );
        expect(
          rows.map(item => item.limit),
          `${binding.id}.${input.name} rows`
        ).toEqual([input.maximumFeatures]);
        const vertices = descriptor.requirements.filter(
          item => item.input === input.name && item.id === 'maximum-vertices'
        );
        // A vertex limit equal to the row limit adds nothing and is left out.
        const needsVertexLimit = input.maximumVertices !== input.maximumFeatures;
        expect(
          vertices.map(item => item.limit),
          `${binding.id}.${input.name} vertices`
        ).toEqual(needsVertexLimit ? [input.maximumVertices] : []);
      }
      for (const requirement of descriptor.requirements) {
        if (requirement.id === 'maximum-rows' || requirement.id === 'maximum-vertices') {
          expect(requirement.severity).toBe('block');
        }
      }
    }
  });

  it('use a class scheme only on k-means cluster ids, and no GPU-only output claims one', () => {
    const schemed: string[] = [];
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      for (const output of descriptor.outputs) {
        if (output.presentation?.scale === 'class-scheme') {
          schemed.push(`${descriptor.bindingId}.${output.tableId}.${output.presentation.column}`);
          expect(getCatalogBinding(descriptor.bindingId)!.run).toBeDefined();
        }
      }
    }
    expect(schemed).toEqual(['k-means.kmeans-assignments.cluster']);
  });

  it('derive defaults only for number parameters in a unit the rule shares, with a known rule', () => {
    const derived: string[] = [];
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      for (const parameter of descriptor.parameters) {
        if (parameter.defaultPolicy.from !== 'data') continue;
        derived.push(`${binding.id}.${parameter.name}`);
        const spec = binding.parameters.find(item => item.name === parameter.name)!;
        expect(spec.kind).toBe('number');
        // Rules return metres for geographic data and coordinate units for planar data, which is
        // what `crs` means. A parameter in coordinate units (degrees for geographic data) may
        // only use the source-units rule; a fixed unit such as degrees must stay a constant.
        if (parameter.defaultPolicy.rule === 'extent-shorter-side-over-250-in-source-units') {
          expect(spec.unit).toBe('coordinate units');
        } else {
          expect(spec.unit).toBe('crs');
        }
        expect(GPU_ANALYSIS_DEFAULT_RULES).toContain(parameter.defaultPolicy.rule);
        expect(parameter.defaultPolicy.factor ?? 1).toBeGreaterThan(0);
      }
    }
    expect(derived).toEqual([
      'geometry-point-buffer.radius',
      'line-simplification.tolerance',
      'line-locate.maximumOffset',
      'grid-generator.cellWidth',
      'grid-generator.cellHeight'
    ]);
  });

  it('use only geometry kinds, matching what each binding accepts and produces', () => {
    const geometryByKind = {points: 'point', lines: 'line', polygons: 'polygon'} as const;
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      for (const input of descriptor.inputs) {
        // `regions` is a label table and `cells` an H3 or Quadbin table: neither is geometry.
        expect(Object.keys(geometryByKind), `${binding.id}.${input.name}`).toContain(input.kind);
        const geometry = geometryByKind[input.kind as keyof typeof geometryByKind];
        expect(binding.inputs.find(item => item.name === input.name)!.accepts).toContain(geometry);
      }
      for (const output of descriptor.outputs) {
        expect(Object.keys(geometryByKind), `${binding.id}.${output.tableId}`).toContain(
          output.kind
        );
        const declared = binding.outputs.find(item => item.tableId === output.tableId)!.geometry;
        // Mixed output retains the input geometry through kindFrom.
        if (declared === 'mixed') {
          expect(output.kindFrom).toBe('features');
        } else if (declared !== 'none') {
          expect(geometryByKind[output.kind as keyof typeof geometryByKind]).toBe(declared);
        }
      }
    }
  });

  it('warn that the coordinate system is declared wherever the binding takes one', () => {
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      const hasCoordinateSystem = binding.parameters.some(item => item.name === 'coordinateSystem');
      const declared = descriptor.requirements.filter(item => item.id === 'declared-coordinates');
      expect(declared.length > 0, descriptor.bindingId).toBe(hasCoordinateSystem);
      for (const requirement of declared) {
        expect(requirement.severity).toBe('warn');
        expect(requirement.remedy).toBeTruthy();
      }
    }
  });

  it('warn about raw coordinates where the binding says its method is planar', () => {
    const planar = [
      'geometry-validity',
      'geometry-convex-clip',
      'group-convex-hull',
      'point-in-polygon-join',
      'geometry-orientation',
      'affine-transform',
      'grid-generator',
      'random-points-in-polygon'
    ];
    for (const id of planar) {
      const requirements = getDescriptor(id).requirements.filter(
        item => item.id === 'planar-coordinates'
      );
      expect(requirements.length, id).toBe(1);
      expect(requirements[0].severity).toBe('warn');
      expect(requirements[0].remedy).toBeTruthy();
    }
  });

  it('keep the group role the same as the other grouping bindings and say what a missing value does', () => {
    const field = getDescriptor('group-convex-hull').inputs[0].fields[0];
    expect(field.accepts).toEqual(['category', 'text', 'integer', 'boolean']);
    const requirements = getDescriptor('group-convex-hull').requirements;
    expect(
      requirements.find(item => item.id === 'no-missing-values' && item.role === field.role)
    ).toBeDefined();
    expect(requirements.find(item => item.id === 'minimum-rows')!.remedy).toContain('per group');
  });

  it('never repeat the primary question as a next question, and name no empty next list', () => {
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      for (const output of descriptor.outputs) {
        if (output.next === undefined) continue;
        expect(output.next.length).toBeGreaterThan(0);
        expect(output.next, `${descriptor.bindingId}.${output.tableId}`).not.toContain(
          descriptor.questions[0]
        );
      }
    }
  });

  it('keeps a data check out of the leading suggestions and plumbing out of the offered ones', () => {
    expect(getDescriptor('geometry-validity').prominence).toBe(2);
    const offered = GEOMETRY_A_OPERATION_DESCRIPTORS.filter(
      item => item.prominence === 2 && item.questions[0] === 'prepare-data'
    ).map(item => item.bindingId);
    expect(offered).toEqual([
      'geometry-validity',
      'geometry-measures',
      'geometry-overlay-intersection',
      'geometry-shape-descriptors'
    ]);
    expect(GEOMETRY_A_OPERATION_DESCRIPTORS.every(item => item.prominence !== 1)).toBe(true);
  });

  it('state the output caps that throw after the run, and the GPU need', () => {
    const notesOf = (id: string) => (getDescriptor(id).notes ?? []).join(' ');
    expect(notesOf('grid-generator')).toContain('20,000');
    expect(notesOf('random-points-in-polygon')).toContain('20,000');
    expect(notesOf('random-points-on-line')).toContain('20,000');
    expect(notesOf('line-chunk')).toContain('50,000');
    expect(notesOf('geometry-line-segmentize')).toContain('500,000');
    for (const id of Object.keys(GPU_ONLY_COLUMNS)) expect(notesOf(id)).toContain('GPU device');
  });

  it('activate the mitre limit only for mitre joins', () => {
    for (const id of ['geometry-line-buffer', 'geometry-polygon-buffer']) {
      const parameter = getDescriptor(id).parameters.find(item => item.name === 'mitreLimit')!;
      expect(parameter.activeWhen).toEqual({parameter: 'joinStyle', equals: ['mitre']});
    }
  });

  it('describe every string parameter of these bindings as a field role, and nothing else', () => {
    // Free-text and id-prefix strings would have to be left out of the roles; the audit found none.
    const stringParameters: string[] = [];
    const roleParameters: string[] = [];
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      for (const parameter of binding.parameters) {
        if (parameter.kind === 'string') stringParameters.push(`${binding.id}.${parameter.name}`);
      }
      for (const input of descriptor.inputs) {
        for (const field of input.fields) roleParameters.push(`${binding.id}.${field.parameter}`);
      }
    }
    expect(roleParameters).toEqual(stringParameters);
    expect(roleParameters).toEqual(['group-convex-hull.groupProperty']);
  });
});

describe('geometry-a descriptors run against the CPU engine', () => {
  for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
    const binding = getCatalogBinding(descriptor.bindingId)!;

    if (!binding.run) {
      it(`${binding.id}: skipped (engine gpu, no CPU engine); columns read from code`, async () => {
        const inputs = makeInputs(
          binding.id,
          binding.inputs.map(input => input.name)
        );
        await expect(runCatalogOperationAsync(binding, {}, inputs)).rejects.toThrow(
          'GPU device required'
        );
        const columnsByTable = GPU_ONLY_COLUMNS[binding.id];
        expect(Object.keys(columnsByTable).sort()).toEqual(
          binding.outputs.map(output => output.tableId).sort()
        );
        for (const output of descriptor.outputs) {
          const column = output.presentation?.column;
          if (column) expect(columnsByTable[output.tableId]).toContain(column);
        }
        for (const output of binding.outputs) {
          for (const identity of output.identityColumns) {
            expect(columnsByTable[output.tableId]).toContain(identity);
          }
        }
      });
      continue;
    }

    it(`${binding.id}: runs with defaults and its field parameters, presentation columns exist`, async () => {
      const inputs = makeInputs(
        binding.id,
        binding.inputs.map(input => input.name)
      );
      const fieldParameters: Record<string, string> = {};
      for (const input of descriptor.inputs) {
        for (const field of input.fields) fieldParameters[field.parameter] = 'kind';
      }
      const parameters = {...getCatalogDefaultParameters(binding), ...fieldParameters};
      const result = await runCatalogOperationAsync(binding, parameters, inputs, {engine: 'cpu'});
      expect(result.tables.map(table => table.id).sort()).toEqual(
        descriptor.outputs.map(output => output.tableId).sort()
      );
      for (const output of descriptor.outputs) {
        const table = result.tables.find(item => item.id === output.tableId)!;
        expect(table.rowCount, `${binding.id}.${output.tableId} rows`).toBeGreaterThan(0);
        if (output.presentation) {
          expect(Object.keys(table.columns), `${binding.id} presentation`).toContain(
            output.presentation.column
          );
          const values = table.columns[output.presentation.column];
          if (output.presentation.scale === 'class-scheme') {
            expect(output.presentation.classScheme, binding.id).toBeDefined();
            expect(
              validateGPUAnalysisResultEncoding(
                output.presentation.classScheme!,
                output.presentation.classEncoding ?? 'contributor-integer',
                values
              ),
              `${binding.id} class encoding`
            ).toBe(1);
          } else if (output.presentation.scale === 'categorical') {
            // A text status column is not any scheme: it scores 0 as a validity flag.
            expect(
              validateGPUAnalysisResultEncoding('validity-flag', 'contributor-integer', values),
              `${binding.id} categorical column is not a flag`
            ).toBeLessThan(1);
          }
          expect(
            values.some(value => value !== null),
            `${binding.id} column is all null`
          ).toBe(true);
        }
      }
    });

    for (const input of descriptor.inputs) {
      for (const field of input.fields) {
        it(`${binding.id}: field parameter ${field.parameter} names a feature property`, async () => {
          const inputs = makeInputs(
            binding.id,
            binding.inputs.map(item => item.name)
          );
          const defaults = getCatalogDefaultParameters(binding);
          const named = await runCatalogOperationAsync(
            binding,
            {...defaults, [field.parameter]: 'kind'},
            inputs,
            {engine: 'cpu'}
          );
          const missing = await runCatalogOperationAsync(
            binding,
            {...defaults, [field.parameter]: 'no_such_column'},
            inputs,
            {engine: 'cpu'}
          );
          expect(JSON.stringify(missing.tables)).not.toEqual(JSON.stringify(named.tables));
          // The named column splits the six points into two groups; a missing one into none.
          expect(named.tables[0].rowCount).toBe(2);
          expect(missing.tables[0].rowCount).toBe(1);
        });
      }
    }
  }
});

// Projected copies of the fixtures: 0.01 degree becomes 1,000 m, centred near (500000, 5000000).
function projectCollection(collection: CatalogFeatureCollection): CatalogFeatureCollection {
  const project = (position: number[]) => [
    500000 + (position[0] - 10) * 100000,
    5000000 + (position[1] - 50) * 100000
  ];
  const projectCoordinates = (value: unknown): unknown =>
    Array.isArray(value) && typeof value[0] === 'number'
      ? project(value as number[])
      : (value as unknown[]).map(projectCoordinates);
  return {
    type: 'FeatureCollection',
    features: collection.features.map(feature => ({
      ...feature,
      geometry: feature.geometry && {
        ...feature.geometry,
        coordinates: projectCoordinates(feature.geometry.coordinates)
      }
    }))
  };
}

// Binding id and the output column that must hold a number when the data is planar and the
// parameters are resolved from the data. With the default EPSG:4326 these columns come out null
// or absurd (area 3.7e13 for a 100 m square).
const PLANAR_METRIC_COLUMNS: readonly [string, string, string][] = [
  ['geometry-measures', 'measures', 'area'],
  ['geometry-shape-descriptors', 'shape-descriptors', 'compactness'],
  ['geometry-point-buffer', 'buffers', 'area'],
  ['nearest-feature-join', 'nearest-feature-join', 'distance'],
  ['buffer-selection', 'buffer-selection', 'distance'],
  ['geometry-line-segmentize', 'segmentized', 'length'],
  ['line-chunk', 'line-chunks', 'length'],
  ['linear-referencing', 'measure-points', 'distance_along'],
  ['group-convex-hull', 'group-hulls', 'area'],
  ['k-means', 'kmeans-assignments', 'distance'],
  ['geodesic-pairs', 'geodesic-pairs', 'distance'],
  ['line-locate', 'line-locations', 'offset']
];

describe('geometry-a descriptors on planar data', () => {
  for (const [bindingId, tableId, column] of PLANAR_METRIC_COLUMNS) {
    it(`${bindingId}: resolved parameters give a ${column} that is a number`, async () => {
      const binding = getCatalogBinding(bindingId)!;
      const descriptor = getDescriptor(bindingId);
      const inputs = Object.fromEntries(
        Object.entries(
          makeInputs(
            bindingId,
            binding.inputs.map(input => input.name)
          )
        ).map(([name, collection]) => [name, projectCollection(collection)])
      );
      const profiles = Object.fromEntries(
        Object.entries(inputs).map(([name, collection]) => [
          name,
          profileKeplerFeatureCollection(collection, {
            datasetId: name,
            label: name,
            coordinateSystem: 'planar'
          })
        ])
      );
      const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles);
      // The coordinate system follows the data, so the metric columns are not read as degrees.
      expect(resolved.parameters['coordinateSystem']).toBe('planar');
      const result = await runCatalogOperationAsync(binding, resolved.parameters, inputs, {
        engine: 'cpu'
      });
      const values = result.tables.find(table => table.id === tableId)!.columns[column];
      expect(values.length, bindingId).toBeGreaterThan(0);
      expect(
        values.every(value => typeof value === 'number' && Number.isFinite(value)),
        `${bindingId}.${column} ${JSON.stringify(values)}`
      ).toBe(true);
    });
  }

  it('shape descriptors reject projected data under the default EPSG:4326', async () => {
    const binding = getCatalogBinding('geometry-shape-descriptors')!;
    await expect(
      runCatalogOperationAsync(
        binding,
        {},
        {polygons: projectCollection(POLYGONS)},
        {engine: 'cpu'}
      )
    ).rejects.toThrow('geometry-shape-descriptors: coordinateSystem');
  });
});

function makeGridExtent(width: number, height: number, planar: boolean): CatalogFeatureCollection {
  const scale = planar ? 1 : 1 / 100000;
  const origin = planar ? [500000, 5000000] : [10, 50];
  return makeCollection([
    makePolygon([
      [
        [origin[0], origin[1]],
        [origin[0] + width * scale, origin[1]],
        [origin[0] + width * scale, origin[1] + height * scale],
        [origin[0], origin[1] + height * scale],
        [origin[0], origin[1]]
      ]
    ])
  ]);
}

describe('grid-generator default cell size', () => {
  const binding = getCatalogBinding('grid-generator')!;
  const descriptor = getDescriptor('grid-generator');
  // Square, five times longer than wide, and the reverse; in projected metres and in degrees.
  const shapes: [number, number][] = [
    [100000, 100000],
    [500000, 100000],
    [100000, 500000],
    [3000, 3000]
  ];
  for (const planar of [true, false]) {
    for (const [width, height] of shapes) {
      it(`stays under the 20,000-cell cap for ${width} x ${height} m ${planar ? 'planar' : 'geographic'}`, async () => {
        const extent = makeGridExtent(width, height, planar);
        const profile = profileKeplerFeatureCollection(extent, {
          datasetId: 'extent',
          label: 'extent',
          coordinateSystem: planar ? 'planar' : 'EPSG:4326'
        });
        const resolved = resolveKeplerOperationParameters(binding, descriptor, {extent: profile});
        for (const name of ['cellWidth', 'cellHeight']) {
          const resolution = resolved.resolutions.find(item => item.parameter === name)!;
          expect(resolution.source, name).toBe('data');
        }
        const result = await runCatalogOperationAsync(
          binding,
          resolved.parameters,
          {extent},
          {
            engine: 'cpu'
          }
        );
        expect(result.tables[0].rowCount).toBeGreaterThan(100);
        expect(result.tables[0].rowCount).toBeLessThanOrEqual(20000);
      });
    }
  }

  it('the binding default of 1 coordinate unit would throw on a 100 km planar square', async () => {
    const extent = makeGridExtent(100000, 100000, true);
    await expect(
      runCatalogOperationAsync(binding, {}, {extent}, {engine: 'cpu'})
    ).rejects.toThrow();
  });
});

describe('geometry-a descriptors on every accepted geometry', () => {
  it('geometry-measures explicitly uses a uniform style because no column exists for all three kinds', () => {
    const output = getDescriptor('geometry-measures').outputs[0];
    expect(output.presentation).toBeUndefined();
    expect(output.presentationException?.type).toBe('uniform-style');
    expect(output.presentationException?.reason).toContain('no shared measure');
  });

  it('nearest-feature-join and buffer-selection give a distance for point, line and polygon features', async () => {
    const featureKinds: [string, CatalogFeatureCollection][] = [
      ['points', POINTS],
      ['lines', LINES],
      ['polygons', POLYGONS]
    ];
    for (const [kind, collection] of featureKinds) {
      const nearest = await runCatalogOperationAsync(
        getCatalogBinding('nearest-feature-join')!,
        {},
        {points: POINTS, targets: collection},
        {engine: 'cpu'}
      );
      expect(
        nearest.tables[0].columns['distance'].every(value => typeof value === 'number'),
        `nearest ${kind}`
      ).toBe(true);
      const selection = await runCatalogOperationAsync(
        getCatalogBinding('buffer-selection')!,
        {distance: 100000},
        {selectors: POINTS, features: collection},
        {engine: 'cpu'}
      );
      expect(selection.tables[0].rowCount, `selection ${kind}`).toBeGreaterThan(0);
      expect(
        selection.tables[0].columns['distance'].every(value => typeof value === 'number'),
        `selection ${kind}`
      ).toBe(true);
    }
  });
});

describe('line-simplification default tolerance', () => {
  it('derives a tolerance from the extent of a single long detailed line', async () => {
    const binding = getCatalogBinding('line-simplification')!;
    const descriptor = getDescriptor('line-simplification');
    // One line of 2,000 vertices over 1 degree of longitude: the spacing statistics see one
    // anchor, so the extent has to come from the line's own vertices.
    const coordinates = Array.from({length: 2000}, (_, index) => [
      10 + index / 2000,
      50 + 0.01 * Math.sin(index / 40) + 0.0002 * Math.sin(index * 1.7)
    ]);
    const lines = makeCollection([makeLine(coordinates)]);
    const profile = profileKeplerFeatureCollection(lines, {
      datasetId: 'lines',
      label: 'lines',
      coordinateSystem: 'EPSG:4326'
    });
    const resolved = resolveKeplerOperationParameters(binding, descriptor, {lines: profile});
    const resolution = resolved.resolutions.find(item => item.parameter === 'tolerance')!;
    expect(resolution.source, resolution.reason).toBe('data');
    const result = await runCatalogOperationAsync(
      binding,
      resolved.parameters,
      {lines},
      {
        engine: 'cpu'
      }
    );
    const after = result.tables[0].columns['vertices_after'][0] as number;
    expect(after).toBeLessThan(2000);
    expect(after).toBeGreaterThan(2);
  });
});

describe('review regressions: geometry-a output kinds and overlap semantics', () => {
  it('inherits output kinds only from the features retained by the operation', async () => {
    const inherited: Record<string, string> = {
      'geometry-validity': 'polygons',
      'geometry-topology-repair': 'polygons',
      'geometry-measures': 'features',
      'geometry-line-segmentize': 'lines',
      'geometry-convex-clip': 'features',
      'line-simplification': 'lines',
      'line-smooth': 'lines',
      'line-precision-reduction': 'lines',
      'nearest-feature-join': 'points',
      'point-in-polygon-join': 'points',
      'buffer-selection': 'features',
      'geodesic-destination': 'points',
      'geometry-orientation': 'polygons',
      'affine-transform': 'features',
      'vertex-snap': 'lines',
      'k-means': 'points',
      'geometry-shape-descriptors': 'polygons'
    };
    for (const [identifier, input] of Object.entries(inherited)) {
      expect(getDescriptor(identifier).outputs[0].kindFrom, identifier).toBe(input);
    }
    for (const identifier of ['geometry-measures', 'affine-transform']) {
      for (const features of [POINTS, LINES, POLYGONS]) {
        const result = await runCatalogOperationAsync(
          getCatalogBinding(identifier)!,
          {},
          {features},
          {engine: 'cpu'}
        );
        const outputGeometry = JSON.parse(String(result.tables[0].columns['geometry'][0]));
        expect(outputGeometry.type, identifier).toBe(features.features[0].geometry!.type);
        expect(getDescriptor(identifier).outputs[0].kindFrom).toBe('features');
      }
    }
  });

  it('declares overlap for convex clips, polygon joins and overlay intersection only', () => {
    const pairs: Record<string, readonly [string, string][]> = {
      'geometry-convex-clip': [['features', 'clip']],
      'point-in-polygon-join': [['points', 'polygons']],
      'geometry-overlay-intersection': [['left', 'right']]
    };
    for (const descriptor of GEOMETRY_A_OPERATION_DESCRIPTORS) {
      expect(descriptor.overlappingInputs ?? [], descriptor.bindingId).toEqual(
        pairs[descriptor.bindingId] ?? []
      );
    }
  });

  it('reserves cell-cap headroom after two-significant-digit rounding for a 5:1 rectangle', () => {
    const descriptor = getDescriptor('grid-generator');
    for (const [width, height] of [
      [38125, 7625],
      [7625, 38125]
    ]) {
      const extent = makeGridExtent(width, height, true);
      const profile = profileKeplerFeatureCollection(extent, {
        datasetId: 'extent',
        label: 'extent',
        coordinateSystem: 'planar'
      });
      const dimensions = ['cellWidth', 'cellHeight'].map(name => {
        const policy = descriptor.parameters.find(
          parameter => parameter.name === name
        )!.defaultPolicy;
        if (policy.from !== 'data')
          throw new Error('grid-generator defaultPolicy.from must be data');
        return Number(
          resolveGPUAnalysisDefault(policy.rule, profile, policy.factor).value!.toPrecision(2)
        );
      });
      expect(
        Math.ceil(width / dimensions[0]) * Math.ceil(height / dimensions[1])
      ).toBeLessThanOrEqual(20000);
    }
  });

  it.each([
    [38125, 7625],
    [100000, 1000],
    [1000, 100000]
  ])('runs the rounded grid defaults for a %s by %s extent', async (width, height) => {
    const extent = makeGridExtent(width, height, true);
    const profile = profileKeplerFeatureCollection(extent, {
      datasetId: 'extent',
      label: 'extent',
      coordinateSystem: 'planar'
    });
    const binding = getCatalogBinding('grid-generator')!;
    const resolved = resolveKeplerOperationParameters(binding, getDescriptor(binding.id), {
      extent: profile
    });
    const result = await runCatalogOperationAsync(
      binding,
      resolved.parameters,
      {extent},
      {engine: 'cpu'}
    );
    expect(result.tables[0].rowCount).toBeLessThanOrEqual(20000);
  });
});

it('grid-generator checks a wrapped profile against the raw longitude span', async () => {
  const extent = makeCollection(
    Array.from({length: 1200}, (_, index) => ({
      geometry: {
        type: 'Point',
        coordinates: [
          index % 40 < 20 ? 179 + (index % 40) / 20 : -180 + ((index % 40) - 20) / 20,
          40 + Math.floor(index / 40) / 30
        ]
      },
      properties: {}
    }))
  );
  const profile = profileKeplerFeatureCollection(extent, {
    datasetId: 'extent',
    label: 'extent',
    coordinateSystem: 'EPSG:4326'
  });
  expect(profile.extent![0]).toBeGreaterThan(profile.extent![2]);
  const binding = getCatalogBinding('grid-generator')!;
  const resolved = resolveKeplerOperationParameters(binding, getDescriptor(binding.id), {
    extent: profile
  });
  const result = await runCatalogOperationAsync(
    binding,
    resolved.parameters,
    {extent},
    {engine: 'cpu'}
  );
  expect(result.tables[0].rowCount).toBeLessThanOrEqual(20000);
});

describe('geometry-a unconditional overlap regression', () => {
  const left = makeCollection([makePolygon([makeSquareRing(3000, 7000, 10)])]);
  const right = makeCollection([makePolygon([makeSquareRing(3040, 7000, 10)])]);

  function assessDisjoint(identifier: string, inputs: Record<string, CatalogFeatureCollection>) {
    return evaluateKeplerOperationApplicability(
      getCatalogBinding(identifier)!,
      getDescriptor(identifier),
      Object.entries(inputs).map(([name, features]) =>
        profileKeplerFeatureCollection(features, {
          datasetId: name,
          label: name,
          coordinateSystem: 'planar'
        })
      ),
      {
        executionEngine: identifier.startsWith('geometry-overlay-') ? 'gpu' : 'cpu',
        inputAssignments: Object.fromEntries(Object.keys(inputs).map(name => [name, name]))
      }
    );
  }

  it.each([
    'geometry-overlay-union',
    'geometry-overlay-difference'
  ])('%s permits disjoint operands at the engine boundary', identifier => {
    expect(assessDisjoint(identifier, {left, right}).status).toBe('ready');
  });

  it.each([
    'geometry-convex-clip',
    'point-in-polygon-join'
  ])('%s blocks disjoint inputs whose binding has no matches', async identifier => {
    const inputs: Record<string, CatalogFeatureCollection> =
      identifier === 'geometry-convex-clip'
        ? {features: left, clip: right}
        : {points: makeCollection([makePoint(3005, 7005)]), polygons: right};
    const assessment = assessDisjoint(identifier, inputs);
    expect(assessment.status).toBe('blocked');
    expect(assessment.missing.some(item => item.what.includes('do not overlap'))).toBe(true);
    const result = await runCatalogOperationAsync(getCatalogBinding(identifier)!, {}, inputs, {
      engine: 'cpu'
    });
    if (identifier === 'geometry-convex-clip') {
      expect(result.tables[0].rowCount).toBe(0);
    } else {
      expect(result.tables[0].columns['match_count']).toEqual([0]);
      expect(result.tables[0].columns['polygon_index']).toEqual([null]);
    }
  });

  it('intersection blocks disjoint operands; numerical execution requires a GPU', async () => {
    expect(assessDisjoint('geometry-overlay-intersection', {left, right}).status).toBe('blocked');
    // This binding has no CPU runner. The CPU convex intersection above checks the same operands.
    await expect(
      runCatalogOperationAsync(
        getCatalogBinding('geometry-overlay-intersection')!,
        {coordinateSystem: 'planar'},
        {left, right}
      )
    ).rejects.toThrow('GPU device required');
  });
});

describe('uniform parameter review regressions', () => {
  it('keeps line simplification tolerance live while selection changes in a fixed parameter buffer', async () => {
    const binding = getCatalogBinding('line-simplification')!;
    const descriptor = getDescriptor(binding.id);
    expect(descriptor.parameters.find(parameter => parameter.name === 'tolerance')?.apply).toBe(
      'live'
    );
    // Coordinate interpretation still changes preparation and therefore requires a rebuild.
    expect(
      descriptor.parameters.find(parameter => parameter.name === 'coordinateSystem')?.apply
    ).toBe('rebuild');
    const coordinates = [
      [0, 0],
      [5, 3],
      [10, 0]
    ];
    const lines = makeCollection([makeLine(coordinates)]);
    const parameters = new Float32Array(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH);
    const parameterBuffer = parameters.buffer;
    for (const [tolerance, expectedVertices] of [
      [1, 3],
      [5, 2]
    ]) {
      expect(getGPULineSimplificationParameterValues({tolerance}, parameters)).toBe(parameters);
      expect(parameters.buffer).toBe(parameterBuffer);
      expect(parameters).toHaveLength(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH);
      expect(parameters[0]).toBe(tolerance);
      const result = await runCatalogOperationAsync(
        binding,
        {tolerance, coordinateSystem: 'planar'},
        {lines},
        {engine: 'cpu'}
      );
      const table = result.tables[0];
      expect(table.rowCount).toBe(1);
      expect(table.columns['vertices_before']).toEqual([3]);
      expect(table.columns['vertices_after']).toEqual([expectedVertices]);
      const geometry = JSON.parse(table.columns['geometry'][0] as string) as {
        coordinates: number[][];
      };
      expect(geometry.coordinates).toEqual(
        expectedVertices === 3 ? coordinates : [coordinates[0], coordinates[2]]
      );
    }
  });

  it('keeps shell orientation live while winding changes without changing the polygon layout', async () => {
    const binding = getCatalogBinding('geometry-orientation')!;
    const descriptor = getDescriptor(binding.id);
    expect(
      descriptor.parameters.find(parameter => parameter.name === 'shellOrientation')?.apply
    ).toBe('live');
    const ring = makeSquareRing(0, 0, 10);
    const polygons = makeCollection([makePolygon([ring])]);
    const parameters = new Float32Array(GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH);
    const parameterBuffer = parameters.buffer;
    for (const shellOrientation of ['counterclockwise', 'clockwise'] as const) {
      const exteriorClockwise = shellOrientation === 'clockwise';
      expect(getGPUGeometryOrientationParameterValues({exteriorClockwise}, parameters)).toBe(
        parameters
      );
      expect(parameters.buffer).toBe(parameterBuffer);
      expect(parameters).toHaveLength(GPU_GEOMETRY_ORIENTATION_PARAMETER_LENGTH);
      expect(parameters[0]).toBe(exteriorClockwise ? 1 : 0);
      const result = await runCatalogOperationAsync(
        binding,
        {shellOrientation},
        {polygons},
        {engine: 'cpu'}
      );
      const table = result.tables[0];
      expect(table.rowCount).toBe(1);
      expect(table.columns['rings_reversed']).toEqual([exteriorClockwise ? 1 : 0]);
      const geometry = JSON.parse(table.columns['geometry'][0] as string) as {
        coordinates: number[][][];
      };
      expect(geometry.coordinates).toEqual([exteriorClockwise ? [...ring].reverse() : ring]);
    }
  });
});
