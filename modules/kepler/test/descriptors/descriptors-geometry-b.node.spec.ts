// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisParameterComparisons} from '@luma-contributors/analysis/gpu-spatial-analysis';

import {describe, expect, it} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {
  CatalogParameterValue,
  CatalogRunResult
} from '../../src/analysis/catalog-binding-types';
import type {CatalogFeature, CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {validateGPUAnalysisResultEncoding} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KeplerOperationDescriptor} from '../../src/descriptors/operation-descriptor-types';
import {GEOMETRY_B_OPERATION_DESCRIPTORS} from '../../src/descriptors/descriptors-geometry-b';
import {evaluateKeplerOperationApplicability} from '../../src/descriptors/applicability';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';

const EXPECTED_IDS = [
  'geometry-minimum-bounds',
  'geometry-label-point',
  'geometry-maximum-inscribed-circle',
  'geometry-minimum-clearance',
  'geographic-distribution',
  'spatial-hilbert-order',
  'spatial-join-bounds-candidates',
  'spatial-join-predicate',
  'geometry-coverage-validity',
  'voronoi-diagram',
  'delaunay-triangulation',
  'concave-hull',
  'polygon-triangulation',
  'polygon-outline',
  'geometry-shape-generator',
  'geometry-dot-density',
  'geometry-local-frame-projection',
  'line-split-at-points',
  'line-clip-by-polygon',
  'line-segment-intersections',
  'line-shared-paths',
  'line-great-circle-arcs',
  'geometry-offset-curve',
  'line-loop-removal',
  'geometry-cleanup',
  'polygon-precision-reduction',
  'geometry-polygonize',
  'spherical-point-in-polygon',
  's2-cell-hierarchy',
  's2-cell-measures',
  's2-cap-covering',
  's2-polygon-covering'
];

// Synthetic data near longitude 10, latitude 50. `at` maps a unit-less grid position to degrees
// (0.01 degree is about 700 m by 1100 m), so every default tolerance sees sensible distances.
const at = (x: number, y: number): [number, number] => [10 + x * 0.01, 50 + y * 0.01];
const collection = (features: CatalogFeature[]): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features
});
const pointFeature = (x: number, y: number, properties: Record<string, unknown> = {}) => ({
  geometry: {type: 'Point', coordinates: at(x, y)},
  properties
});
const lineFeature = (
  coordinates: [number, number][],
  properties: Record<string, unknown> = {}
) => ({
  geometry: {type: 'LineString', coordinates: coordinates.map(([x, y]) => at(x, y))},
  properties
});
const square = (x: number, y: number, size: number): [number, number][] => [
  [x, y],
  [x + size, y],
  [x + size, y + size],
  [x, y + size],
  [x, y]
];
const polygonFeature = (rings: [number, number][][], properties: Record<string, unknown> = {}) => ({
  geometry: {type: 'Polygon', coordinates: rings.map(ring => ring.map(([x, y]) => at(x, y)))},
  properties
});

const SCATTER = collection([
  pointFeature(0, 0, {group: 'a', weight: 2, size: 30}),
  pointFeature(4, 1, {group: 'a', weight: 1, size: 40}),
  pointFeature(2, 5, {group: 'a', weight: 3, size: 20}),
  pointFeature(9, 0, {group: 'b', weight: 2, size: 30}),
  pointFeature(12, 4, {group: 'b', weight: 1, size: 50}),
  pointFeature(8, 6, {group: 'b', weight: 4, size: 35})
]);
const AREAS = collection([
  polygonFeature([square(0, 0, 6)], {total: 40}),
  polygonFeature([square(8, 0, 6)], {total: 25})
]);
const OVERLAPPING_AREAS = collection([
  polygonFeature([square(0, 0, 6)], {total: 40}),
  polygonFeature([square(4, 4, 6)], {total: 25})
]);
const CROSSING_LINES = collection([
  lineFeature([
    [0, 0],
    [10, 10]
  ]),
  lineFeature([
    [0, 10],
    [10, 0]
  ])
]);
const LOOPED_LINE = collection([
  lineFeature([
    [0, 0],
    [20, 0],
    [20, 20],
    [10, 20],
    [10, -10]
  ])
]);
const PAIRED_LINES = collection([
  lineFeature([
    [0, 0],
    [5, 0],
    [10, 0]
  ])
]);
// Two squares that share the middle edge, as lines that meet at shared vertices.
const FACE_LINES = collection([
  lineFeature([
    [0, 0],
    [10, 0],
    [20, 0],
    [20, 10],
    [10, 10],
    [0, 10],
    [0, 0]
  ]),
  lineFeature([
    [10, 0],
    [10, 10]
  ])
]);
const SPIKY_POLYGON = collection([
  polygonFeature(
    [
      [
        [0, 0],
        [10, 0],
        [10, 0],
        [10, 10],
        [0, 10],
        [0, 0]
      ]
    ],
    {}
  )
]);

type Fixture = {
  inputs: Record<string, CatalogFeatureCollection>;
  /** Field role (`<input>.<role>`) to the column of the synthetic data that fills it. */
  fields?: Record<string, string>;
};

// One fixture per binding. `fields` names the real column for each field role of the descriptor,
// keyed by the binding parameter that receives it.
const FIXTURES: Record<string, Fixture> = {
  'geometry-minimum-bounds': {inputs: {features: AREAS}},
  'geometry-label-point': {inputs: {polygons: AREAS}},
  'geometry-maximum-inscribed-circle': {inputs: {polygons: AREAS}},
  'geometry-minimum-clearance': {inputs: {features: AREAS}},
  'geographic-distribution': {
    inputs: {features: SCATTER},
    fields: {groupProperty: 'group', weightProperty: 'weight'}
  },
  'spatial-hilbert-order': {inputs: {features: SCATTER}},
  'spatial-join-bounds-candidates': {inputs: {left: SCATTER, right: AREAS}},
  'spatial-join-predicate': {inputs: {left: SCATTER, right: AREAS}},
  'geometry-coverage-validity': {inputs: {polygons: OVERLAPPING_AREAS}},
  'voronoi-diagram': {inputs: {points: SCATTER}},
  'delaunay-triangulation': {inputs: {points: SCATTER}},
  'concave-hull': {inputs: {features: SCATTER}, fields: {groupProperty: 'group'}},
  'polygon-triangulation': {inputs: {polygons: AREAS}},
  'polygon-outline': {inputs: {polygons: AREAS}},
  'geometry-shape-generator': {inputs: {points: SCATTER}, fields: {radiusProperty: 'size'}},
  'geometry-dot-density': {inputs: {polygons: AREAS}, fields: {valueProperty: 'total'}},
  'geometry-local-frame-projection': {inputs: {features: AREAS}},
  'line-split-at-points': {
    inputs: {lines: PAIRED_LINES, splitters: collection([pointFeature(5, 0.001)])}
  },
  'line-clip-by-polygon': {
    inputs: {
      lines: collection([
        lineFeature([
          [-5, 3],
          [20, 3]
        ])
      ]),
      polygons: AREAS
    }
  },
  'line-segment-intersections': {inputs: {lines: CROSSING_LINES}},
  'line-shared-paths': {inputs: {left: PAIRED_LINES, right: PAIRED_LINES}},
  'line-great-circle-arcs': {
    inputs: {
      origins: collection([pointFeature(0, 0), pointFeature(5, 5)]),
      destinations: collection([pointFeature(100, 40), pointFeature(120, 80)])
    }
  },
  'geometry-offset-curve': {inputs: {lines: LOOPED_LINE}},
  'line-loop-removal': {inputs: {lines: LOOPED_LINE}},
  'geometry-cleanup': {inputs: {features: SPIKY_POLYGON}},
  'polygon-precision-reduction': {inputs: {polygons: AREAS}},
  'geometry-polygonize': {inputs: {lines: FACE_LINES}},
  'spherical-point-in-polygon': {inputs: {points: SCATTER, polygons: AREAS}},
  's2-cell-hierarchy': {inputs: {points: SCATTER}},
  's2-cell-measures': {inputs: {points: SCATTER}},
  's2-cap-covering': {inputs: {points: collection([pointFeature(0, 0)])}},
  's2-polygon-covering': {
    inputs: {
      polygons: collection([
        polygonFeature([
          [
            [0, 0],
            [300, 0],
            [300, 300],
            [0, 300],
            [0, 0]
          ]
        ])
      ])
    }
  }
};

// Bindings whose CPU engine does not exist. All 32 have one, so nothing is skipped; the list stays
// so that a binding that loses its CPU engine is reported here instead of silently untested.
const GPU_ONLY_SKIPPED: Record<string, string> = {};

const NONEXISTENT_COLUMN = 'no_such_column_anywhere';

function snapshotResult(result: CatalogRunResult): string {
  return JSON.stringify({tables: result.tables, diagnostics: result.diagnostics});
}

async function runWithFields(
  bindingId: string,
  fieldValues: Record<string, string>
): Promise<CatalogRunResult> {
  const binding = getCatalogBinding(bindingId)!;
  const parameters: Record<string, CatalogParameterValue> = {
    ...getCatalogDefaultParameters(binding),
    ...fieldValues
  };
  return runCatalogOperationAsync(binding, parameters, FIXTURES[bindingId].inputs, {
    engine: 'cpu'
  });
}

describe('GEOMETRY_B_OPERATION_DESCRIPTORS', () => {
  it('covers the 32 bindings once, in order', () => {
    expect(GEOMETRY_B_OPERATION_DESCRIPTORS.map(descriptor => descriptor.bindingId)).toEqual(
      EXPECTED_IDS
    );
    expect(new Set(EXPECTED_IDS).size).toBe(32);
    expect(Object.keys(FIXTURES).sort()).toEqual([...EXPECTED_IDS].sort());
  });

  it('has a registered binding for every descriptor', () => {
    for (const id of EXPECTED_IDS) expect(getCatalogBinding(id), id).toBeDefined();
  });

  it('flags a descriptor that disagrees with its binding', () => {
    const descriptor = GEOMETRY_B_OPERATION_DESCRIPTORS[0];
    const binding = getCatalogBinding('geometry-label-point')!;
    expect(validateKeplerOperationDescriptor(descriptor, binding).length).toBeGreaterThan(0);
    const broken = {...descriptor, parameters: descriptor.parameters.slice(1)};
    expect(
      validateKeplerOperationDescriptor(broken, getCatalogBinding(descriptor.bindingId)!)
    ).not.toEqual([]);
  });

  for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
    describe(descriptor.bindingId, () => {
      const binding = getCatalogBinding(descriptor.bindingId)!;

      it('passes the structural validator with zero issues', () => {
        expect(validateKeplerOperationDescriptor(descriptor, binding)).toEqual([]);
      });

      it('keeps to the descriptor rules', () => {
        expect(descriptor.action.length).toBeLessThanOrEqual(40);
        expect(
          descriptor.parameters.filter(parameter => parameter.level === 'basic').length
        ).toBeLessThanOrEqual(4);
        for (const parameter of descriptor.parameters) {
          if (parameter.defaultPolicy.from !== 'data') continue;
          const spec = binding.parameters.find(candidate => candidate.name === parameter.name)!;
          if (parameter.defaultPolicy.rule === 'column-sum') {
            // Dot values have the selected column's unit, independent of coordinate units.
            expect(spec.name).toBe('dotValue');
            expect(spec.unit).toBeUndefined();
            expect(parameter.defaultPolicy).toEqual({
              from: 'data',
              input: 'polygons',
              rule: 'column-sum',
              role: 'value',
              factor: 1 / 50000
            });
          } else {
            // Distance defaults must use the coordinate unit that their rules return.
            expect(spec.unit).toBe('crs');
          }
        }
      });

      it('repeats the binding input capacities exactly', () => {
        for (const input of binding.inputs) {
          const limits = descriptor.requirements.filter(
            requirement => requirement.input === input.name
          );
          const rows = limits.find(requirement => requirement.id === 'maximum-rows');
          const vertices = limits.find(requirement => requirement.id === 'maximum-vertices');
          expect(rows?.limit, `${input.name} rows`).toBe(input.maximumFeatures);
          expect(vertices?.limit, `${input.name} vertices`).toBe(input.maximumVertices);
          expect(rows?.severity).toBe('block');
          expect(vertices?.severity).toBe('block');
        }
      });

      const skipReason = GPU_ONLY_SKIPPED[descriptor.bindingId];
      if (skipReason) {
        it.skip(`runs on the CPU engine (${skipReason})`, () => {});
        return;
      }

      const fixture = FIXTURES[descriptor.bindingId];
      const fieldParameters = descriptor.inputs.flatMap(input =>
        input.fields.map(field => field.parameter)
      );

      it('has a fixture column for every field parameter', () => {
        expect(Object.keys(fixture.fields ?? {}).sort()).toEqual([...fieldParameters].sort());
      });

      it('runs with defaults plus the field columns and fills every output', async () => {
        const result = await runWithFields(descriptor.bindingId, fixture.fields ?? {});
        for (const output of descriptor.outputs) {
          const table = result.tables.find(candidate => candidate.id === output.tableId);
          expect(table, `${output.tableId} produced`).toBeDefined();
          expect(table!.rowCount, `${output.tableId} has rows`).toBeGreaterThan(0);
          if (output.presentation) {
            const column = table!.columns[output.presentation.column];
            expect(column, `${output.tableId}.${output.presentation.column}`).toBeDefined();
            // A presentation column that is empty or all null would colour nothing.
            expect(column.some(value => value !== null)).toBe(true);
          }
        }
      });

      it('writes the encoding a class-scheme presentation declares', async () => {
        const result = await runWithFields(descriptor.bindingId, fixture.fields ?? {});
        for (const output of descriptor.outputs) {
          const presentation = output.presentation;
          if (presentation?.scale !== 'class-scheme') continue;
          const table = result.tables.find(candidate => candidate.id === output.tableId)!;
          expect(
            validateGPUAnalysisResultEncoding(
              presentation.classScheme!,
              presentation.classEncoding ?? 'contributor-integer',
              table.columns[presentation.column]
            )
          ).toBe(1);
        }
      });

      for (const parameter of fieldParameters) {
        it(`treats ${parameter} as a column: a missing column changes the outcome`, async () => {
          const baseline = await runWithFields(descriptor.bindingId, fixture.fields ?? {});
          let changed = false;
          try {
            const altered = await runWithFields(descriptor.bindingId, {
              ...fixture.fields,
              [parameter]: NONEXISTENT_COLUMN
            });
            changed = snapshotResult(altered) !== snapshotResult(baseline);
          } catch {
            changed = true;
          }
          expect(changed).toBe(true);
        });
      }
    });
  }

  it('describes the five column parameters of the half and no other string parameter', () => {
    const fieldParameters = GEOMETRY_B_OPERATION_DESCRIPTORS.flatMap(descriptor =>
      descriptor.inputs.flatMap(input => input.fields.map(field => field.parameter))
    ).sort();
    expect(fieldParameters).toEqual(
      ['groupProperty', 'groupProperty', 'radiusProperty', 'valueProperty', 'weightProperty'].sort()
    );
    for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      const stringParameters = binding.parameters
        .filter(parameter => parameter.kind === 'string')
        .map(parameter => parameter.name)
        .sort();
      const described = descriptor.inputs
        .flatMap(input => input.fields.map(field => field.parameter))
        .sort();
      expect(described).toEqual(stringParameters);
    }
  });

  it('scores the coverage status text 0 under the validity flag, so it stays categorical', async () => {
    const result = await runWithFields('geometry-coverage-validity', {});
    const status = result.tables.find(table => table.id === 'coverage-features')!.columns['status'];
    expect(status).toContain('overlaps');
    for (const value of status) expect(['valid', 'overlaps']).toContain(value);
    for (const encoding of ['contributor-integer', 'catalog-text'] as const) {
      let score: number | undefined;
      try {
        score = validateGPUAnalysisResultEncoding('validity-flag', encoding, status);
      } catch {
        score = 0; // The scheme has no such encoding.
      }
      expect(score).toBe(0);
    }
    const output = GEOMETRY_B_OPERATION_DESCRIPTORS.find(
      descriptor => descriptor.bindingId === 'geometry-coverage-validity'
    )!.outputs.find(candidate => candidate.tableId === 'coverage-features')!;
    expect(output.presentation?.scale).toBe('categorical');
  });

  describe('review fixes', () => {
    const byId = (id: string) =>
      GEOMETRY_B_OPERATION_DESCRIPTORS.find(descriptor => descriptor.bindingId === id)!;

    it('declares the coordinate system wherever a binding has the parameter', () => {
      for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
        const binding = getCatalogBinding(descriptor.bindingId)!;
        if (!binding.parameters.some(parameter => parameter.name === 'coordinateSystem')) continue;
        const declared = descriptor.requirements.find(
          requirement => requirement.id === 'declared-coordinates'
        );
        expect(declared, descriptor.bindingId).toBeDefined();
        expect(declared?.severity).toBe('warn');
        expect(declared?.remedy?.length).toBeGreaterThan(0);
      }
    });

    it('warns about raw coordinates for the overlap check', () => {
      expect(
        byId('geometry-coverage-validity').requirements.some(
          requirement => requirement.id === 'planar-coordinates' && requirement.severity === 'warn'
        )
      ).toBe(true);
    });

    it('does not derive an offset distance from the extent', () => {
      const distance = byId('geometry-offset-curve').parameters.find(
        parameter => parameter.name === 'distance'
      );
      expect(distance?.defaultPolicy.from).toBe('constant');
    });

    it('tells the user how to keep dot density under its dot cap', () => {
      const descriptor = byId('geometry-dot-density');
      const cap = getCatalogBinding('geometry-dot-density')!.parameters.find(
        parameter => parameter.name === 'maximumDots'
      )!.defaultValue;
      expect(cap).toBe(100000);
      expect((descriptor.notes ?? []).join(' ')).toContain('100,000');
      const remedies = descriptor.requirements
        .filter(requirement => requirement.role === 'value')
        .map(requirement => requirement.remedy)
        .join(' ');
      expect(remedies).toContain('100,000');
      expect(remedies).toContain('value per dot');
    });

    for (const total of [1000000, 50]) {
      it(`derives a value per dot for a column totaling ${total} and runs within half the cap`, async () => {
        // These fixtures require column-sum in both the analysis rule and resolver rounding table.
        // Unequal values exercise the sum of per-polygon floors, rather than one aggregate floor.
        const polygons = collection(
          [0.33333, 0.66667].map((fraction, index) =>
            polygonFeature([square(index * 8, 0, 6)], {
              total: total * fraction,
              unrelated: 1000000000
            })
          )
        );
        const binding = getCatalogBinding('geometry-dot-density')!;
        const profile = profileKeplerFeatureCollection(polygons, {
          datasetId: 'area-totals',
          label: 'Area totals',
          coordinateSystem: 'EPSG:4326'
        });
        const resolved = resolveKeplerOperationParameters(
          binding,
          byId(binding.id),
          {polygons: profile},
          {fieldBindings: {valueProperty: 'total'}}
        );
        const result = await runCatalogOperationAsync(
          binding,
          resolved.parameters,
          {polygons},
          {engine: 'cpu'}
        );
        expect(resolved.resolutions.find(entry => entry.parameter === 'dotValue')?.source).toBe(
          'data'
        );
        expect(resolved.parameters['dotValue']).toBeCloseTo(total / 50000, 9);
        const count = result.tables.find(table => table.id === 'dots')!.rowCount;
        // At most one dot per polygon is lost to flooring; the target is 50% of the default cap.
        expect(count).toBeGreaterThanOrEqual(49998);
        expect(count).toBeLessThanOrEqual(50000);
        expect(count).toBeLessThan(Number(resolved.parameters['maximumDots']));
      });
    }

    it('runs the reviewer parcel fixture after deriving its value per dot', async () => {
      // This also requires column-sum support in the resolver's rounding table.
      const polygons = collection(
        Array.from({length: 16}, (_, index) => {
          const west = 1000 + (index % 4) * 100;
          const south = 1000 + Math.floor(index / 4) * 100;
          return {
            geometry: {type: 'Polygon', coordinates: [square(west, south, 100)]},
            properties: {assessed_value: 250000 + index * 1000}
          };
        })
      );
      const binding = getCatalogBinding('geometry-dot-density')!;
      const profile = profileKeplerFeatureCollection(polygons, {
        datasetId: 'parcels',
        label: 'Parcels',
        coordinateSystem: 'planar'
      });
      const resolved = resolveKeplerOperationParameters(
        binding,
        byId(binding.id),
        {polygons: profile},
        {fieldBindings: {valueProperty: 'assessed_value'}}
      );
      const result = await runCatalogOperationAsync(
        binding,
        resolved.parameters,
        {polygons},
        {engine: 'cpu'}
      );
      // Two significant digits in the resolver can move the half-cap target slightly.
      const count = result.tables.find(table => table.id === 'dots')!.rowCount;
      expect(count).toBeGreaterThanOrEqual(49000);
      expect(count).toBeLessThanOrEqual(51000);
    });

    it('keeps the binding value per dot as fallback when no value column is bound', () => {
      const binding = getCatalogBinding('geometry-dot-density')!;
      const resolved = resolveKeplerOperationParameters(binding, byId(binding.id), {});
      expect(resolved.parameters['dotValue']).toBe(
        getCatalogDefaultParameters(binding)['dotValue']
      );
      expect(resolved.resolutions.find(entry => entry.parameter === 'dotValue')?.source).toBe(
        'fallback'
      );
    });

    it('explains S2 levels in notes and keeps the level parameters basic', () => {
      for (const id of ['s2-cell-hierarchy', 's2-cell-measures', 's2-cap-covering']) {
        const descriptor = byId(id);
        expect((descriptor.notes ?? []).join(' ')).toContain('km');
        expect(descriptor.parameters.find(parameter => parameter.name === 'level')?.level).toBe(
          'basic'
        );
      }
      const covering = byId('s2-polygon-covering');
      expect((covering.notes ?? []).join(' ')).toContain('km');
      expect(covering.parameters.find(parameter => parameter.name === 'maximumLevel')?.level).toBe(
        'basic'
      );
    });

    it('accepts any column type for a group and flags missing group values', () => {
      for (const id of ['geographic-distribution', 'concave-hull']) {
        const descriptor = byId(id);
        const group = descriptor.inputs[0].fields.find(field => field.role === 'group')!;
        expect([...group.accepts].sort()).toEqual(['boolean', 'category', 'integer', 'text']);
        expect(
          descriptor.requirements.some(
            requirement =>
              requirement.id === 'no-missing-values' &&
              requirement.role === 'group' &&
              requirement.severity === 'warn' &&
              requirement.remedy
          )
        ).toBe(true);
      }
    });

    it('says that single points give no minimum bounds', () => {
      expect((byId('geometry-minimum-bounds').notes ?? []).join(' ')).toContain('single');
    });

    it('gives tables without geometry no presentation', () => {
      for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
        const binding = getCatalogBinding(descriptor.bindingId)!;
        for (const output of descriptor.outputs) {
          const spec = binding.outputs.find(candidate => candidate.tableId === output.tableId)!;
          if (spec.geometry === 'none') {
            expect(output.presentation, output.tableId).toBeUndefined();
            expect(output.kind).toBe('pairs');
          }
        }
        // Dots and covers have no column worth colouring by.
      }
    });

    it('never repeats the primary question or a question twice in next', () => {
      for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
        for (const output of descriptor.outputs) {
          const next = output.next ?? [];
          expect(next).not.toContain(descriptor.questions[0]);
          expect(new Set(next).size).toBe(next.length);
        }
      }
    });

    it('uses plain action phrases', () => {
      for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
        expect(descriptor.action, descriptor.bindingId).not.toMatch(/S2|neighbour|Project to/);
      }
      expect(byId('delaunay-triangulation').action).toContain('triangles');
    });

    it('keeps only a few prepare-data operations at prominence 2', () => {
      const offered = GEOMETRY_B_OPERATION_DESCRIPTORS.filter(
        descriptor => descriptor.questions[0] === 'prepare-data' && descriptor.prominence === 2
      ).map(descriptor => descriptor.bindingId);
      expect(offered.sort()).toEqual([
        'geometry-cleanup',
        'geometry-coverage-validity',
        'geometry-label-point'
      ]);
    });

    it('marks parameters that only matter under another parameter with activeWhen', () => {
      const active = (id: string, name: string) =>
        byId(id).parameters.find(parameter => parameter.name === name)?.activeWhen;
      expect(active('geometry-local-frame-projection', 'useDataCenter')).toEqual({
        parameter: 'direction',
        equals: ['forward']
      });
      expect(active('concave-hull', 'ratio')).toEqual({
        parameter: 'maximumEdgeLength',
        equals: [0]
      });
      expect(active('s2-cell-hierarchy', 'parentLevel')).toEqual({
        parameter: 'operation',
        equals: ['parent']
      });
      expect(active('geometry-shape-generator', 'circleSegments')).toEqual({
        parameter: 'shape',
        equals: ['circle']
      });
      for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
        const names = getCatalogBinding(descriptor.bindingId)!.parameters.map(
          parameter => parameter.name
        );
        for (const parameter of descriptor.parameters) {
          if (parameter.activeWhen)
            for (const comparison of getGPUAnalysisParameterComparisons(parameter.activeWhen))
              expect(names).toContain(
                'parameter' in comparison ? comparison.parameter : comparison.whenParameter
              );
        }
      }
    });

    it('states the per-feature and pair caps no requirement can express', () => {
      const notesOf = (id: string) => (byId(id).notes ?? []).join(' ');
      expect(notesOf('geometry-minimum-clearance')).toContain('5,000');
      expect(notesOf('polygon-triangulation')).toContain('1,500');
      expect(notesOf('spatial-join-bounds-candidates')).toContain('200,000');
      expect(notesOf('spatial-join-predicate')).toContain('200,000');
      expect(notesOf('spherical-point-in-polygon')).toContain('500,000');
    });

    // Presentation columns must hold values for every geometry kind an input accepts.
    const KIND_FEATURES: Record<string, (index: number) => CatalogFeature> = {
      point: index =>
        pointFeature(index * 3, (index * 7) % 5, {group: 'g' + (index % 2), weight: 1, size: 20}),
      line: index =>
        lineFeature(
          [
            [index * 3, 0],
            [index * 3 + 2, 2],
            [index * 3 + 4, 0]
          ],
          {group: 'g' + (index % 2), weight: 1, size: 20}
        ),
      polygon: index =>
        polygonFeature([square(index * 3, (index % 3) * 4, 2)], {
          group: 'g' + (index % 2),
          weight: 1,
          size: 20,
          total: 5
        })
    };
    for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      for (const input of binding.inputs) {
        if (input.accepts.length < 2) continue;
        for (const kind of input.accepts) {
          it(`${descriptor.bindingId}: presentation columns hold values for ${kind} ${input.name}`, async () => {
            const features = collection(
              Array.from({length: 8}, (_, index) => KIND_FEATURES[kind](index))
            );
            const fixture = FIXTURES[descriptor.bindingId];
            const parameters = {
              ...getCatalogDefaultParameters(binding),
              ...fixture.fields
            };
            const result = await runCatalogOperationAsync(
              binding,
              parameters,
              {...fixture.inputs, [input.name]: features},
              {engine: 'cpu'}
            );
            for (const output of descriptor.outputs) {
              if (!output.presentation) continue;
              const table = result.tables.find(candidate => candidate.id === output.tableId)!;
              expect(table.columns[output.presentation.column]).toBeDefined();
              if (table.rowCount > 0) {
                expect(
                  table.columns[output.presentation.column].some(value => value !== null),
                  `${output.tableId}.${output.presentation.column}`
                ).toBe(true);
              }
            }
          });
        }
      }
    }
  });
});

describe('review regressions: geometry-b metadata and descriptor validation', () => {
  function getDescriptor(identifier: string) {
    return GEOMETRY_B_OPERATION_DESCRIPTORS.find(
      descriptor => descriptor.bindingId === identifier
    )!;
  }

  it('inherits retained feature kinds but keeps representative points fixed', async () => {
    for (const [identifier, input] of Object.entries({
      'geometry-coverage-validity': 'polygons',
      'geometry-local-frame-projection': 'features',
      'line-loop-removal': 'lines',
      'geometry-cleanup': 'features',
      'polygon-precision-reduction': 'polygons',
      'spherical-point-in-polygon': 'points'
    })) {
      expect(
        getDescriptor(identifier).outputs.find(output => output.kindFrom === input),
        identifier
      ).toBeDefined();
    }
    expect(getDescriptor('spatial-hilbert-order').outputs[0].kindFrom).toBeUndefined();
    const result = await runCatalogOperationAsync(
      getCatalogBinding('geometry-local-frame-projection')!,
      {direction: 'inverse', originLongitude: 8, originLatitude: 46},
      {features: collection([{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}])},
      {engine: 'cpu'}
    );
    expect(JSON.parse(String(result.tables[0].columns['geometry'][0])).coordinates).toEqual([
      8, 46
    ]);
    expect(getDescriptor('geometry-local-frame-projection').notes?.join(' ')).toContain(
      'Forward outputs planar metres; inverse outputs longitude and latitude. Declare the output coordinate system accordingly.'
    );
  });

  it('requires raw extent overlap only for the planar predicate join', () => {
    const pairs: Record<string, readonly [string, string][]> = {
      'spatial-join-predicate': [['left', 'right']]
    };
    for (const descriptor of GEOMETRY_B_OPERATION_DESCRIPTORS) {
      expect(descriptor.overlappingInputs ?? [], descriptor.bindingId).toEqual(
        pairs[descriptor.bindingId] ?? []
      );
    }
  });

  it('validates both parameter and requirement activeWhen controllers and possible values', () => {
    const binding = getCatalogBinding('geometry-shape-generator')!;
    const original = getDescriptor(binding.id);
    const conditions: NonNullable<KeplerOperationDescriptor['parameters'][number]['activeWhen']>[] =
      [
        {parameter: 'missing-control', equals: ['circle']},
        {parameter: 'shape', equals: ['not-a-shape']},
        {parameter: 'shape', equals: [true]},
        {parameter: 'shape', equals: []},
        {parameter: 'circleSegments', equals: [0]},
        {parameter: 'circleSegments', equals: [3.5]},
        {parameter: 'circleSegments', equals: [Infinity]},
        {parameter: 'circleSegments', equals: ['64']}
      ];
    for (const condition of conditions) {
      const parameterDescriptor = {
        ...original,
        parameters: original.parameters.map(parameter =>
          parameter.name === 'circleSegments' ? {...parameter, activeWhen: condition} : parameter
        )
      };
      const requirementDescriptor = {
        ...original,
        requirements: original.requirements.map(requirement => ({
          ...requirement,
          activeWhen: condition
        }))
      };
      for (const descriptor of [parameterDescriptor, requirementDescriptor]) {
        expect(
          validateKeplerOperationDescriptor(descriptor, binding).some(issue =>
            issue.includes('activeWhen')
          ),
          JSON.stringify(condition)
        ).toBe(true);
      }
    }
    const projection = getDescriptor('geometry-local-frame-projection');
    const projectionBinding = getCatalogBinding(projection.bindingId)!;
    for (const value of [true, false, 'true', 1]) {
      const descriptor: KeplerOperationDescriptor = {
        ...projection,
        parameters: projection.parameters.map(parameter =>
          parameter.name === 'originLongitude'
            ? {...parameter, activeWhen: {parameter: 'useDataCenter', equals: [value]}}
            : parameter
        )
      };
      expect(validateKeplerOperationDescriptor(descriptor, projectionBinding).length > 0).toBe(
        typeof value !== 'boolean'
      );
    }
    expect(
      validateKeplerOperationDescriptor(
        {
          ...original,
          requirements: original.requirements.map(requirement => ({
            ...requirement,
            activeWhen: {parameter: 'shape', equals: ['circle']}
          }))
        },
        binding
      )
    ).toEqual([]);
    expect(validateKeplerOperationDescriptor(original, binding)).toEqual([]);
  });

  it('rejects inherited kinds and overlap pairs that name nonexistent inputs', () => {
    const original = getDescriptor('geometry-cleanup');
    const binding = getCatalogBinding(original.bindingId)!;
    expect(
      validateKeplerOperationDescriptor(
        {
          ...original,
          outputs: original.outputs.map(output => ({...output, kindFrom: 'missing-input'}))
        },
        binding
      ).some(issue => issue.includes('kindFrom') && issue.includes('missing-input'))
    ).toBe(true);
    for (const pair of [
      ['missing-input', 'features'],
      ['features', 'missing-input']
    ] as const) {
      expect(
        validateKeplerOperationDescriptor({...original, overlappingInputs: [pair]}, binding).some(
          issue => issue.includes('overlappingInputs') && issue.includes('missing-input')
        )
      ).toBe(true);
    }
    expect(validateKeplerOperationDescriptor(original, binding)).toEqual([]);
  });
});

describe('geometry-b unconditional overlap and dot capacity regressions', () => {
  function assessInputs(
    identifier: string,
    inputs: Record<string, CatalogFeatureCollection>,
    overrides: Record<string, CatalogParameterValue> = {},
    fieldBindings: Record<string, string> = {},
    coordinateSystem: 'planar' | 'EPSG:4326' = 'planar'
  ) {
    const binding = getCatalogBinding(identifier)!;
    const descriptor = GEOMETRY_B_OPERATION_DESCRIPTORS.find(
      descriptor => descriptor.bindingId === identifier
    )!;
    const profiles = Object.entries(inputs).map(([name, features]) =>
      profileKeplerFeatureCollection(features, {
        datasetId: name,
        label: name,
        coordinateSystem
      })
    );
    return evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      overrides,
      fieldBindings,
      executionEngine: 'cpu',
      inputAssignments: Object.fromEntries(Object.keys(inputs).map(name => [name, name]))
    });
  }

  it('bounds candidates execute across a gap covered by distance', async () => {
    const inputs = {
      left: collection([{geometry: {type: 'Point', coordinates: [3000, 7000]}, properties: {}}]),
      right: collection([
        {
          geometry: {type: 'Polygon', coordinates: [square(3005, 7000, 1)]},
          properties: {}
        }
      ])
    };
    const assessment = assessInputs('spatial-join-bounds-candidates', inputs, {distance: 6});
    expect(assessment.status).toBe('ready');
    const result = await runCatalogOperationAsync(
      getCatalogBinding('spatial-join-bounds-candidates')!,
      assessment.parameters,
      inputs
    );
    expect(result.tables[0].rowCount).toBe(1);
  });

  it('shared paths execute across a gap covered by tolerance', async () => {
    const makeLines = (height: number) =>
      collection([
        {
          geometry: {
            type: 'LineString',
            coordinates: [
              [3000, height],
              [3120, height]
            ]
          },
          properties: {}
        }
      ]);
    const inputs = {left: makeLines(7000), right: makeLines(7008)};
    const assessment = assessInputs('line-shared-paths', inputs, {tolerance: 9});
    expect(assessment.status).toBe('ready');
    const result = await runCatalogOperationAsync(
      getCatalogBinding('line-shared-paths')!,
      assessment.parameters,
      inputs
    );
    expect(result.tables[0].columns['length']).toEqual([120]);
  });

  it.each([
    'intersects',
    'within',
    'contains'
  ])('predicate %s blocks disjoint inputs and produces no pairs', async predicate => {
    const inputs = {left: collection([pointFeature(40, 40)]), right: AREAS};
    const assessment = assessInputs('spatial-join-predicate', inputs, {predicate}, {}, 'EPSG:4326');
    expect(assessment.status).toBe('blocked');
    expect(assessment.missing.some(item => item.what.includes('do not overlap'))).toBe(true);
    const result = await runCatalogOperationAsync(
      getCatalogBinding('spatial-join-predicate')!,
      assessment.parameters,
      inputs
    );
    expect(result.tables[0].rowCount).toBe(0);
  });

  it.each(['outside', 'both'])('line clipping keeps disjoint lines when keep is %s', async keep => {
    const inputs = {
      lines: collection([
        lineFeature([
          [40, 40],
          [41, 40]
        ])
      ]),
      polygons: AREAS
    };
    const assessment = assessInputs('line-clip-by-polygon', inputs, {keep}, {}, 'EPSG:4326');
    expect(assessment.status).toBe('ready');
    const result = await runCatalogOperationAsync(
      getCatalogBinding('line-clip-by-polygon')!,
      assessment.parameters,
      inputs
    );
    expect(result.tables[0].columns['side']).toEqual(['outside']);
  });

  it('spherical containment can match beyond the vertex latitude extent', async () => {
    const inputs = {
      points: collection([{geometry: {type: 'Point', coordinates: [0, 65]}, properties: {}}]),
      polygons: collection([
        {
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [-45, 60],
                [0, 0],
                [45, 60],
                [-45, 60]
              ]
            ]
          },
          properties: {}
        }
      ])
    };
    const assessment = assessInputs('spherical-point-in-polygon', inputs, {}, {}, 'EPSG:4326');
    expect(assessment.status).toBe('ready');
    const result = await runCatalogOperationAsync(
      getCatalogBinding('spherical-point-in-polygon')!,
      assessment.parameters,
      inputs
    );
    expect(result.tables[0].rowCount).toBe(1);
  });

  it('blocks the verifier signed-total fixture before its oversized dot request can execute', async () => {
    const polygons = collection([
      polygonFeature([square(0, 0, 1)], {amount: 900000}),
      polygonFeature([square(2, 0, 1)], {amount: -800000})
    ]);
    const assessment = assessInputs(
      'geometry-dot-density',
      {polygons},
      {dotValue: 2},
      {valueProperty: 'amount'},
      'EPSG:4326'
    );
    const binding = getCatalogBinding('geometry-dot-density')!;
    // Keep the verifier's unsafe value explicit so the requirement, not default resolution, blocks it.
    expect(assessment.parameters['dotValue']).toBe(2);
    expect(assessment.status).toBe('blocked');
    expect(assessment.requirements).toContainEqual(
      expect.objectContaining({
        id: 'non-negative-values',
        role: 'value',
        severity: 'block',
        state: 'failed'
      })
    );
    expect(assessment.missing.some(item => item.remedy?.includes('non-negative area totals'))).toBe(
      true
    );
    // Direct binding calls bypass AX requirements and retain the catalog's capacity failure.
    await expect(
      runCatalogOperationAsync(binding, assessment.parameters, {polygons})
    ).rejects.toThrow('Dot capacity exceeded');
  });

  it('explains proportional dotValue adjustment when maximumDots is lowered', async () => {
    const polygons = AREAS;
    const binding = getCatalogBinding('geometry-dot-density')!;
    const descriptor = GEOMETRY_B_OPERATION_DESCRIPTORS.find(
      descriptor => descriptor.bindingId === binding.id
    )!;
    const assessment = assessInputs(
      binding.id,
      {polygons},
      {},
      {valueProperty: 'total'},
      'EPSG:4326'
    );
    const defaultMaximumDots = Number(getCatalogDefaultParameters(binding)['maximumDots']);
    const maximumDots = defaultMaximumDots / 10;
    const dotValue = (Number(assessment.parameters['dotValue']) * defaultMaximumDots) / maximumDots;
    const notes = descriptor.notes?.join(' ') ?? '';
    expect(notes).toContain('assumes the default dot cap');
    expect(notes).toContain('in proportion');
    const result = await runCatalogOperationAsync(
      binding,
      {
        ...assessment.parameters,
        maximumDots,
        dotValue
      },
      {polygons}
    );
    expect(result.tables[0].rowCount).toBeGreaterThan(0);
    expect(result.tables[0].rowCount).toBeLessThanOrEqual(maximumDots);
  });
});
