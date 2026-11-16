// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getCatalogBinding,
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {parseCatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {KEPLER_CURATED_ANALYSES} from '../../src/curated/curated-analyses';
import {
  evaluateKeplerWorkbenchInputs,
  evaluateKeplerWorkbenchApplicability,
  getKeplerWorkbenchCuratedPresets,
  getKeplerWorkbenchParameterDefaults,
  type KeplerWorkbenchSubject
} from '../../src/descriptors/workbench-adapters';
import {buildSampleProfile} from './engine-test-profiles';

const POINT_SUBJECT: KeplerWorkbenchSubject = {
  kind: 'point',
  rowCount: 500,
  columns: [
    {name: 'severity', type: 'int32'},
    {name: 'note', type: 'string'},
    {name: 'weight', type: 'float64'},
    {name: 'flagged', type: 'boolean'}
  ]
};

function isFormValue(value: unknown): boolean {
  return (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean' ||
    (Array.isArray(value) && value.every(entry => typeof entry === 'number'))
  );
}

describe('evaluateKeplerWorkbenchApplicability', () => {
  it('returns exactly the specified shape for a passing operation', () => {
    const result = evaluateKeplerWorkbenchApplicability('kernel-density-grid', POINT_SUBJECT);
    expect(Object.keys(result).sort()).toEqual([
      'outputKinds',
      'parameters',
      'reason',
      'remedy',
      'status'
    ]);
    expect(result.status).toBe('unknown');
    expect(result.remedy).toBeTruthy();
    expect(Object.values(result.parameters).every(isFormValue)).toBe(true);
    expect(result.parameters['cellSize']).toBe(100);
    expect(result.parameters['kernel']).toBe('gaussian');
  });

  it('fails a point operation for a table without geometry, with a reason and a remedy', () => {
    const result = evaluateKeplerWorkbenchApplicability('kernel-density-grid', {
      kind: 'table',
      rowCount: 10,
      columns: [{name: 'amount', type: 'float64'}]
    });
    expect(result.status).toBe('failed');
    expect(result.reason).toMatch(/dataset of points/);
    expect(result.remedy).toMatch(/coordinate columns or a geometry column/);
  });

  it('fails when the subject geometry is not one the operation reads', () => {
    const polygon = evaluateKeplerWorkbenchApplicability('kernel-density-grid', {
      ...POINT_SUBJECT,
      kind: 'polygon'
    });
    expect(polygon.status).toBe('failed');
    const line = evaluateKeplerWorkbenchApplicability('network-centrality', {
      ...POINT_SUBJECT,
      kind: 'point'
    });
    expect(line.status).toBe('failed');
  });

  it('judges a features subject as the first geometry the operation accepts', () => {
    const points = evaluateKeplerWorkbenchApplicability('kernel-density-grid', {
      ...POINT_SUBJECT,
      kind: 'features'
    });
    expect(points.status).toBe('unknown');
    const lines = evaluateKeplerWorkbenchApplicability('network-centrality', {
      ...POINT_SUBJECT,
      kind: 'features'
    });
    expect(lines.status).not.toBe('failed');
    const polygons = evaluateKeplerWorkbenchApplicability('geometry-validity', {
      ...POINT_SUBJECT,
      kind: 'features'
    });
    expect(polygons.status).not.toBe('failed');
  });

  it('fails a subject over a capacity and names the limit', () => {
    const result = evaluateKeplerWorkbenchApplicability('recipe-hot-spot-lattice', {
      ...POINT_SUBJECT,
      rowCount: 250_000
    });
    expect(result.status).toBe('failed');
    expect(result.reason).toMatch(/250,000 rows; at most 100,000/);
    expect(result.remedy).toBeTruthy();
  });

  it('does not fail on counts when the row count is not known', () => {
    const {rowCount: _rowCount, ...unknownSize} = POINT_SUBJECT;
    const result = evaluateKeplerWorkbenchApplicability('recipe-hot-spot-lattice', unknownSize);
    // Mandatory unknown evidence is unresolved, even for a subject-only call.
    expect(result.status).toBe('unknown');
    expect(result.remedy).toBeTruthy();
  });

  it('keeps mandatory unknowns unresolved for subjects and profiles', () => {
    const polygons: KeplerWorkbenchSubject = {
      kind: 'polygon',
      rowCount: 144,
      columns: [{name: 'cases', type: 'int32'}]
    };
    // Metadata alone still lacks mandatory capacity evidence. A full profile settles it;
    // the proposed count column is then a visible measurement warning, not a blocked run.
    const subjectOnly = evaluateKeplerWorkbenchApplicability('local-moran-statistics', polygons);
    expect(subjectOnly.status).toBe('unknown');
    expect(subjectOnly.remedy).toBeTruthy();
    const withProfile = evaluateKeplerWorkbenchApplicability(
      'local-moran-statistics',
      polygons,
      buildSampleProfile('districts', {keepColumns: ['district', 'cases']})
    );
    expect(withProfile.status).toBe('passed');
    expect(withProfile.parameters['valueProperty']).toBe('cases');
    // An unknown vertex count remains unresolved until the host supplies it.
    const polygonOperation = evaluateKeplerWorkbenchApplicability('geometry-validity', {
      kind: 'polygon',
      rowCount: 50,
      columns: []
    });
    expect(polygonOperation.status).toBe('unknown');
  });

  it('accepts a vertex count and treats a table with coordinate columns as points', () => {
    const table: KeplerWorkbenchSubject = {
      kind: 'table',
      rowCount: 500,
      pointColumns: {longitude: 'lng', latitude: 'lat'},
      columns: [
        {name: 'lng', type: 'float64'},
        {name: 'lat', type: 'float64'}
      ]
    };
    expect(evaluateKeplerWorkbenchApplicability('kernel-density-grid', table).status).toBe(
      'unknown'
    );
    const bigPolygons: KeplerWorkbenchSubject = {
      kind: 'polygon',
      rowCount: 100,
      vertexCount: 5_000_000,
      columns: []
    };
    const result = evaluateKeplerWorkbenchApplicability('geometry-validity', bigPolygons);
    expect(result.status).toBe('failed');
    expect(result.reason).toMatch(/vertices; at most/);
  });

  it('takes the execution engine: a GPU-only operation fails on the CPU', () => {
    const gpuOnly = KEPLER_CATALOG_BINDINGS.find(binding => binding.engine === 'gpu');
    expect(gpuOnly).toBeDefined();
    const subject: KeplerWorkbenchSubject = {kind: 'point', rowCount: 10, columns: []};
    const onGpu = evaluateKeplerWorkbenchApplicability(gpuOnly!.id, subject);
    const onCpu = evaluateKeplerWorkbenchApplicability(gpuOnly!.id, subject, undefined, {
      executionEngine: 'cpu'
    });
    expect(onCpu.status).toBe('failed');
    expect(onCpu.reason).toMatch(/needs a GPU device/);
    expect(onGpu.reason).not.toMatch(/needs a GPU device/);
  });

  it('fails an unregistered operation with a reason instead of throwing', () => {
    const result = evaluateKeplerWorkbenchApplicability('no-such-operation', POINT_SUBJECT);
    expect(result.status).toBe('failed');
    expect(result.reason).toMatch(/not a registered operation/);
    expect(result.parameters).toEqual({});
  });

  it('uses a given profile instead of building one, so derived defaults show', () => {
    const withProfile = evaluateKeplerWorkbenchApplicability(
      'kernel-density-grid',
      POINT_SUBJECT,
      buildSampleProfile('incidents')
    );
    expect(withProfile.parameters['cellSize']).not.toBe(100);
    expect(withProfile.parameters['coordinateSystem']).toBe('EPSG:4326');
  });

  it('preserves parameters that are lists of numbers, for every binding that has one', () => {
    const withLists = KEPLER_CATALOG_BINDINGS.filter(binding =>
      binding.parameters.some(parameter => parameter.kind === 'number-list')
    );
    expect(withLists.length).toBeGreaterThan(0);
    for (const binding of withLists) {
      const defaults = getKeplerWorkbenchParameterDefaults(binding.id);
      for (const parameter of binding.parameters.filter(entry => entry.kind === 'number-list')) {
        expect(defaults.values[parameter.name], `${binding.id}.${parameter.name}`).toEqual(
          parameter.defaultValue
        );
        expect([...defaults.basic, ...defaults.advanced, ...defaults.inactive]).toContain(
          parameter.name
        );
      }
      expect(Object.values(defaults.values).every(isFormValue)).toBe(true);
    }
  });
});

describe('getKeplerWorkbenchParameterDefaults', () => {
  it('returns values, basic and advanced, which together name every parameter once', () => {
    const defaults = getKeplerWorkbenchParameterDefaults('kernel-density-grid');
    expect(Object.keys(defaults).sort()).toEqual([
      'advanced',
      'basic',
      'confirmations',
      'inactive',
      'resolutions',
      'values'
    ]);
    const binding = getCatalogBinding('kernel-density-grid')!;
    expect([...defaults.basic, ...defaults.advanced, ...defaults.inactive].sort()).toEqual(
      binding.parameters.map(parameter => parameter.name).sort()
    );
    expect(Object.keys(defaults.values).sort()).toEqual(
      binding.parameters.map(parameter => parameter.name).sort()
    );
    expect(defaults.basic).toContain('cellSize');
    expect(defaults.values['cellSize']).toBe(100);
  });

  it('derives values from a profile and rejects an unknown binding', () => {
    const defaults = getKeplerWorkbenchParameterDefaults(
      'kernel-density-grid',
      buildSampleProfile('incidents')
    );
    expect(defaults.values['cellSize']).not.toBe(100);
    expect(defaults.values['bandwidth']).not.toBe(500);
    const binding = getCatalogBinding('kernel-density-grid')!;
    expect(normalizeCatalogParameters(binding, defaults.values)).toEqual(defaults.values);
    expect(() => getKeplerWorkbenchParameterDefaults('no-such-operation')).toThrow(
      /getKeplerWorkbenchParameterDefaults bindingId "no-such-operation"/
    );
  });
});

describe('getKeplerWorkbenchCuratedPresets', () => {
  const presets = getKeplerWorkbenchCuratedPresets();

  it('has one preset per available curated analysis, with the specified shape', () => {
    const available = KEPLER_CURATED_ANALYSES.filter(
      analysis => analysis.execution.status === 'available'
    );
    expect(available.length).toBeGreaterThan(5);
    expect(presets).toHaveLength(available.length);
    for (const preset of presets) {
      expect(
        Object.keys(preset).every(key =>
          [
            'curatedAnalysisId',
            'executionParameters',
            'bindingId',
            'title',
            'parameters',
            'sampleGeoJSONByInput'
          ].includes(key)
        )
      ).toBe(true);
      expect(preset.title.length).toBeGreaterThan(0);
      expect(Object.values(preset.parameters).every(isFormValue)).toBe(true);
    }
  });

  it('references existing bindings and parameters that the binding accepts', () => {
    for (const preset of presets) {
      const binding = getCatalogBinding(preset.bindingId);
      expect(binding, preset.bindingId).toBeDefined();
      for (const name of Object.keys(preset.parameters)) {
        expect(
          binding!.parameters.some(parameter => parameter.name === name),
          `${preset.bindingId}.${name}`
        ).toBe(true);
      }
      expect(
        () => normalizeCatalogParameters(binding!, preset.parameters),
        preset.title
      ).not.toThrow();
    }
  });

  it('gives each sample as GeoJSON text keyed by binding input name', () => {
    let withSamples = 0;
    for (const preset of presets) {
      if (!preset.sampleGeoJSONByInput) {
        continue;
      }
      withSamples++;
      const binding = getCatalogBinding(preset.bindingId)!;
      const inputNames = binding.inputs.map(input => input.name);
      for (const [inputName, text] of Object.entries(preset.sampleGeoJSONByInput)) {
        expect(inputNames, `${preset.title}: ${inputName}`).toContain(inputName);
        const collection = parseCatalogFeatureCollection(text);
        expect(collection.features.length).toBeGreaterThan(0);
      }
      // Every input of the binding has a sample, so the preset runs as it is.
      expect(Object.keys(preset.sampleGeoJSONByInput).sort(), preset.title).toEqual(
        [...inputNames].sort()
      );
    }
    expect(withSamples).toBeGreaterThan(5);
  });

  it('writes the sample columns into the column parameters', () => {
    const hotSpot = presets.find(preset => preset.bindingId === 'local-moran-statistics');
    expect(hotSpot?.parameters['valueProperty']).toBe('rate');
    const kernel = presets.find(preset => preset.bindingId === 'kernel-density-grid');
    // The samples leave the scale parameters to the defaults derived from the data.
    expect(kernel?.parameters['cellSize']).toBeUndefined();
    expect(kernel?.parameters['bandwidth']).toBeUndefined();
    expect(kernel?.parameters['kernel']).toBe('gaussian');
  });
});

it('exposes resolved output kinds to workbench hosts', () => {
  const result = evaluateKeplerWorkbenchApplicability('kernel-density-grid', {
    kind: 'point',
    rowCount: 100,
    columns: []
  });
  expect(result.outputKinds).toEqual({'density-cells': 'surface'});
});

it('retains dataset identity behind port aliases for same-layer comparison checks', () => {
  const sample = buildSampleProfile('before');
  const before = {
    ...sample,
    coordinates: {model: 'geographic' as const, source: 'declared' as const}
  };
  const options = {overrides: {cellSize: 100}};
  const identical = evaluateKeplerWorkbenchInputs('cell-compare', {before, after: before}, options);
  expect(identical.status).toBe('blocked');
  expect(identical.missing.some(problem => problem.what.includes('same rows'))).toBe(true);
  const independent = evaluateKeplerWorkbenchInputs(
    'cell-compare',
    {
      before,
      after: {...before, datasetId: 'independent', label: 'Independent observations'}
    },
    options
  );
  expect(independent.status, JSON.stringify(independent.missing)).toBe('ready');
  expect(before.datasetId).toBe('before');
});
