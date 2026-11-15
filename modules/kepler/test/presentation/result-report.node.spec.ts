// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  describeGPUAnalysisPermutationResolution,
  formatGPUAnalysisMeasuredValue,
  getGPUAnalysisCautionText,
  getGPUAnalysisSignificanceAttainability,
  getGPUAnalysisQuestion,
  summarizeGPUAnalysisResult
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUAnalysisDefaultResolution} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {describe, expect, it} from 'vitest';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunResult
} from '../../src/analysis/catalog-binding-types';
import {getCatalogBinding, runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {
  CatalogFeatureCollection,
  CatalogOutputTable
} from '../../src/analysis/catalog-geojson';
import {
  KEPLER_GRID_CELL_CAPS,
  resolveKeplerOperationParameters
} from '../../src/descriptors/default-parameters';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisSample,
  KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS
} from '../../src/curated/curated-analysis-samples';
import {
  getKeplerOperationDescriptor,
  KEPLER_OPERATION_DESCRIPTORS
} from '../../src/descriptors/operation-descriptors';
import {CURATED_RUNTIME_CAUTION_SUMMARY_KEYS} from '../../src/descriptors/descriptors-curated';
import type {KeplerOperationDescriptor} from '../../src/descriptors/operation-descriptor-types';
import {
  describeKeplerCatalogResult,
  formatKeplerResultReport,
  KEPLER_RESULT_FACT_SOURCES
} from '../../src/presentation/result-report';
import type {KeplerResultReport} from '../../src/presentation/result-report';
import {
  getKeplerBindingResultPresentations,
  planKeplerResultPresentation
} from '../../src/presentation/result-presentation';

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

type ColumnValue = string | number | boolean | null;

const ALL_DESCRIPTORS: readonly KeplerOperationDescriptor[] = KEPLER_OPERATION_DESCRIPTORS;

/**
 * The descriptor of a binding with the class encoding of its class-scheme outputs filled in when
 * the descriptor does not name one yet (the descriptor authors are adding the field).
 */
function getDescriptor(
  bindingId: string,
  encoding: 'contributor-integer' | 'catalog-text' | 'catalog-signed-label' = 'catalog-text'
): KeplerOperationDescriptor {
  const descriptor = getKeplerOperationDescriptor(bindingId);
  if (!descriptor) throw new Error(`getDescriptor: no descriptor for ${bindingId}`);
  return {
    ...descriptor,
    outputs: descriptor.outputs.map(output =>
      output.presentation?.classScheme
        ? {
            ...output,
            presentation: {
              ...output.presentation,
              classEncoding: output.presentation.classEncoding ?? encoding
            }
          }
        : output
    )
  };
}

/** A resolution record as `resolveGPUAnalysisDefault` returns it, with the fields a spec cares about. */
function makeResolution(fields: {
  explanation: string;
  value?: number;
  unit?: GPUAnalysisDefaultResolution['unit'];
  sampled?: boolean;
}): GPUAnalysisDefaultResolution {
  return {
    rule: 'median-nearest-neighbor-distance',
    value: fields.value,
    factor: 1,
    unit: fields.unit ?? 'coordinate-unit',
    explanation: fields.explanation,
    sampled: fields.sampled ?? false
  };
}

function getBinding(bindingId: string): CatalogOperationBinding {
  const binding = getCatalogBinding(bindingId);
  if (!binding) throw new Error(`getBinding: no binding ${bindingId}`);
  return binding;
}

async function runCuratedAnalysis(
  analysisId: string,
  overrides: Record<string, CatalogParameterValue> = {}
): Promise<CatalogRunResult> {
  const analysis = getKeplerCuratedAnalysis(analysisId);
  if (!analysis || analysis.execution.status !== 'available' || !analysis.sampleId) {
    throw new Error(`runCuratedAnalysis: ${analysisId} is not an available analysis with a sample`);
  }
  const execution = analysis.execution;
  const sample = createKeplerCuratedAnalysisSample(analysis.sampleId);
  const columns =
    KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS[
      analysis.sampleId as keyof typeof KEPLER_CURATED_ANALYSIS_SAMPLE_COLUMNS
    ];
  const inputs: Record<string, CatalogFeatureCollection> = {};
  const profiles: Record<string, ReturnType<typeof profileKeplerFeatureCollection>> = {};
  for (const [slotName, inputName] of Object.entries(execution.inputs)) {
    inputs[inputName] = sample[slotName];
    profiles[inputName] = profileKeplerFeatureCollection(sample[slotName], {
      datasetId: slotName,
      label: slotName
    });
  }
  const fieldBindings: Record<string, string> = {};
  for (const [key, parameterName] of Object.entries(execution.fieldParameters)) {
    if (columns[key]) fieldBindings[parameterName] = columns[key];
  }
  const binding = getCatalogBinding(execution.bindingId)!;
  const descriptor = getKeplerOperationDescriptor(binding.id)!;
  // The curated analyses leave the scale parameters to the defaults engine.
  const maximumCellCount = KEPLER_GRID_CELL_CAPS[binding.id]?.maximumCellCount;
  const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles, {
    fieldBindings,
    overrides: {...execution.parameters, ...overrides},
    ...(maximumCellCount ? {maximumCellCount} : {})
  });
  return runCatalogOperationAsync(binding, resolved.parameters, inputs, {engine: 'cpu'});
}

function getTable(result: CatalogRunResult, tableId: string): CatalogOutputTable {
  const table = result.tables.find(candidate => candidate.id === tableId);
  if (!table) throw new Error(`getTable: no table ${tableId}`);
  return table;
}

function makeTable(id: string, columns: Record<string, ColumnValue[]>): CatalogOutputTable {
  const names = Object.keys(columns);
  return {
    id,
    label: id,
    rowCount: names.length ? columns[names[0]].length : 0,
    columns,
    columnTypes: Object.fromEntries(names.map(name => [name, 'string'])) as Record<string, 'string'>
  };
}

function makeResult(overrides: Partial<CatalogRunResult>): CatalogRunResult {
  return {
    operationId: 'synthetic',
    engine: 'cpu-reference',
    dispatchedContributors: [],
    parameters: {},
    tables: [],
    diagnostics: [],
    summary: {},
    ...overrides
  };
}

/** Thousands separators, written without the code under test. */
function group(value: number): string {
  return value.toLocaleString('en-US');
}

function countWhere(values: readonly ColumnValue[], test: (value: ColumnValue) => boolean): number {
  return values.filter(test).length;
}

function getCautionIds(report: KeplerResultReport): string[] {
  return report.cautions.map(caution => caution.id);
}

/** A square lattice of polygons in planar coordinates with a hot corner. */
function makeLattice(size: number): CatalogFeatureCollection {
  const features = [];
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      features.push({
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [column, row],
              [column + 1, row],
              [column + 1, row + 1],
              [column, row + 1],
              [column, row]
            ]
          ]
        },
        properties: {
          value: column * row + (column + row) * 0.5 + ((column * 7 + row * 3) % 5)
        }
      });
    }
  }
  return {type: 'FeatureCollection', features};
}

const LATTICE_PARAMETERS = {
  valueProperty: 'value',
  neighborhood: 'queen',
  coordinateSystem: 'planar'
} as const;

// ---------------------------------------------------------------------------------------------
// Sentences from real results
// ---------------------------------------------------------------------------------------------

describe('result report: sentences from real catalog results', () => {
  it('hot spot map: the sentence counts the hot cells of the table and names the correction', async () => {
    const result = await runCuratedAnalysis('hot-spot-map');
    const table = getTable(result, 'recipe-hot-spots');
    const report = describeKeplerCatalogResult(
      result,
      getBinding('recipe-hot-spot-lattice'),
      getDescriptor('recipe-hot-spot-lattice', 'contributor-integer')
    );
    const bins = table.columns['bin'];
    const hotCount = countWhere(bins, bin => (bin as number) > 0);
    const coldCount = countWhere(bins, bin => (bin as number) < 0);
    // Cells that have a z-score are the cells that were tested.
    const testedCount = countWhere(table.columns['z_score'], value => value !== null);
    expect(hotCount).toBeGreaterThan(0);
    expect(report.sentence).toMatch(
      new RegExp(
        `^${hotCount} of ${group(testedCount)} cells are significant hot spots and ${coldCount} are (significant )?cold spots at 90% confidence after false discovery rate correction\\.$`
      )
    );
    expect(report.outputs).toEqual([
      {
        tableId: 'recipe-hot-spots',
        template: 'significant-hot-cold-count',
        usedTemplate: 'significant-hot-cold-count',
        sentence: report.sentence
      }
    ]);
    // Corrected with Benjamini-Hochberg, so no multiple-testing caution; the 1,024 cells are enough.
    expect(getCautionIds(report)).toEqual([]);
    expect(report.engine).toBe('Computed on the CPU with the reference engine.');
    expect(report.fallbackReason).toBe('The CPU engine was requested, so the GPU was not used.');
    expect(report.diagnostics).toBe(result.diagnostics);
  });

  it('hot spot map without correction says so and carries the uncorrected caution', async () => {
    const result = await runCuratedAnalysis('hot-spot-map', {multipleTesting: 'none'});
    const report = describeKeplerCatalogResult(
      result,
      getBinding('recipe-hot-spot-lattice'),
      getDescriptor('recipe-hot-spot-lattice', 'contributor-integer')
    );
    expect(report.sentence).toMatch(/at 90% confidence without correction for multiple testing\.$/);
    expect(getCautionIds(report)).toEqual(['multiple-testing-uncorrected']);
  });

  it('cluster and outlier map: the sentence names each class count and the report carries both cautions', async () => {
    const result = await runCuratedAnalysis('cluster-and-outlier-map', {multipleTesting: 'none'});
    const table = getTable(result, 'local-moran');
    const clusters = table.columns['cluster'];
    const count = (label: string): number => countWhere(clusters, value => value === label);
    const significantCount = count('HH') + count('LL') + count('HL') + count('LH');
    const report = describeKeplerCatalogResult(
      result,
      getBinding('local-moran-statistics'),
      getDescriptor('local-moran-statistics')
    );
    expect(significantCount).toBeGreaterThan(0);
    const parts = [
      [count('HH'), 'high-high core', 'high-high cores'],
      [count('LL'), 'low-low core', 'low-low cores'],
      [count('HL'), 'high-low outlier', 'high-low outliers'],
      [count('LH'), 'low-high outlier', 'low-high outliers']
    ]
      .filter(([amount]) => (amount as number) > 0)
      .map(([amount, singular, plural]) => `${amount} ${amount === 1 ? singular : plural}`);
    const list =
      parts.length === 1
        ? parts[0]
        : `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`;
    expect(report.sentence).toBe(
      `${significantCount} of ${table.rowCount} features are significant at a pseudo p-value of 0.05 or less without correction for multiple testing: ${list}.`
    );
    // No correction and permutations: both cautions, with their stock text.
    expect(report.cautions).toEqual([
      {id: 'pseudo-p-values', ...getGPUAnalysisCautionText('pseudo-p-values')},
      {
        id: 'multiple-testing-uncorrected',
        ...getGPUAnalysisCautionText('multiple-testing-uncorrected', {significanceLevel: 0.05})
      }
    ]);
    expect(report.next.map(step => step.question)).toEqual(expect.arrayContaining(['how-related']));
    for (const step of report.next) {
      expect(step.label).toBe(getGPUAnalysisQuestion(step.question).label);
    }
  });

  it('too few permutations for the significance level add the resolution to the pseudo p-value caution', async () => {
    const result = await runCuratedAnalysis('cluster-and-outlier-map', {permutations: 9});
    const report = describeKeplerCatalogResult(
      result,
      getBinding('local-moran-statistics'),
      getDescriptor('local-moran-statistics')
    );
    const pseudo = report.cautions.find(caution => caution.id === 'pseudo-p-values')!;
    // 9 permutations give a smallest pseudo p-value of 0.1, above the level of 0.05.
    expect(describeGPUAnalysisPermutationResolution(9, 0.05)).toMatch(/at least 19 permutations/);
    expect(pseudo.text).toBe(
      `${getGPUAnalysisCautionText('pseudo-p-values').text} ${describeGPUAnalysisPermutationResolution(9, 0.05)}`
    );
    // With 999 the stock text stands alone.
    const enough = describeKeplerCatalogResult(
      await runCuratedAnalysis('cluster-and-outlier-map'),
      getBinding('local-moran-statistics'),
      getDescriptor('local-moran-statistics')
    );
    expect(enough.cautions.find(caution => caution.id === 'pseudo-p-values')!.text).toBe(
      getGPUAnalysisCautionText('pseudo-p-values').text
    );
  });

  it('local Moran with no permutations tests nothing and the sentence says so', async () => {
    const result = await runCuratedAnalysis('cluster-and-outlier-map', {
      permutations: 0,
      multipleTesting: 'none'
    });
    const table = getTable(result, 'local-moran');
    expect(countWhere(table.columns['cluster'], value => value === 'not-tested')).toBe(
      table.rowCount
    );
    const report = describeKeplerCatalogResult(
      result,
      getBinding('local-moran-statistics'),
      getDescriptor('local-moran-statistics')
    );
    // No p-values exist, so the package says so rather than "no features to test".
    expect(report.sentence).toBe(
      summarizeGPUAnalysisResult('local-cluster-count', {
        featureCount: 0,
        highHighCount: 0,
        lowLowCount: 0,
        highLowCount: 0,
        lowHighCount: 0,
        significanceLevel: 0.05,
        pValueKind: 'pseudo',
        correction: 'none',
        permutationCount: 0,
        pValuesComputed: false,
        analyzedCount: table.rowCount
      })
    );
    expect(report.sentence).toContain(`none of the ${table.rowCount} features was tested`);
  });

  it('a small lattice raises the fewer-than-thirty caution and a large one does not', async () => {
    const binding = getBinding('local-moran-statistics');
    const small = await runCatalogOperationAsync(
      binding,
      {...LATTICE_PARAMETERS, permutations: 99},
      {features: makeLattice(4)},
      {engine: 'cpu'}
    );
    const smallReport = describeKeplerCatalogResult(
      small,
      binding,
      getDescriptor('local-moran-statistics')
    );
    expect(getCautionIds(smallReport)).toEqual([
      'fewer-than-thirty-features',
      'pseudo-p-values',
      'multiple-testing-uncorrected'
    ]);
    const large = await runCatalogOperationAsync(
      binding,
      {...LATTICE_PARAMETERS, permutations: 99},
      {features: makeLattice(6)},
      {engine: 'cpu'}
    );
    expect(
      getCautionIds(
        describeKeplerCatalogResult(large, binding, getDescriptor('local-moran-statistics'))
      )
    ).toEqual(['pseudo-p-values', 'multiple-testing-uncorrected']);
  });

  it('clustering: the clusters, the noise share and the largest cluster from the points table', async () => {
    const result = await runCuratedAnalysis('clusters-and-footprints');
    const clusters = getTable(result, 'points').columns['cluster'];
    const pointCount = clusters.length;
    const noiseCount = countWhere(clusters, value => value === -1);
    const labels = [...new Set(clusters as number[])].filter(label => label >= 0);
    const largest = Math.max(
      ...labels.map(label => countWhere(clusters, value => value === label))
    );
    const report = describeKeplerCatalogResult(
      result,
      getBinding('curated-clusters-and-footprints'),
      getDescriptor('curated-clusters-and-footprints', 'catalog-signed-label')
    );
    expect(noiseCount).toBeGreaterThan(0);
    expect(report.sentence).toBe(
      `Found ${labels.length} clusters; ${noiseCount} of ${pointCount} points (${((noiseCount / pointCount) * 100).toFixed(1)}%) are noise and the largest cluster holds ${((largest / pointCount) * 100).toFixed(1)}% of all points.`
    );
    // The summary of the binding agrees with what was read from the table.
    expect(result.summary['clusterCount']).toBe(labels.length);
    expect(result.summary['noiseCount']).toBe(noiseCount);
    expect(report.outputs.map(output => output.usedTemplate)).toEqual([
      'cluster-count-and-noise',
      'row-count'
    ]);
    expect(report.outputs[1].sentence).toBe(`The result has ${labels.length} rows.`);
    expect(getTable(result, 'footprints').rowCount).toBe(labels.length);
  });

  it('density surface: the range is written with significant digits, not as zero', async () => {
    const result = await runCuratedAnalysis('density-surface');
    const values = getTable(result, 'density-cells').columns['density'] as number[];
    const report = describeKeplerCatalogResult(
      result,
      getBinding('kernel-density-grid'),
      ALL_DESCRIPTORS.find(item => item.bindingId === 'kernel-density-grid')!
    );
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    expect(maximum).toBeLessThan(0.005);
    expect(report.sentence).toBe(
      `Density ranges from ${formatMeasured(minimum)} to ${formatMeasured(maximum)} across ${group(values.length)} values.`
    );
    expect(report.sentence).not.toMatch(/from 0 to 0/);
  });

  it('IDW surface: the sentence reports the value range of the grid', async () => {
    const result = await runCuratedAnalysis('idw-surface');
    const values = getTable(result, 'idw-cells').columns['value'] as number[];
    const report = describeKeplerCatalogResult(
      result,
      getBinding('idw-grid'),
      ALL_DESCRIPTORS.find(item => item.bindingId === 'idw-grid')!
    );
    expect(report.sentence).toBe(
      `Value ranges from ${formatMeasured(Math.min(...values))} to ${formatMeasured(Math.max(...values))} across ${group(values.length)} values.`
    );
  });

  it('count by area: the total, the area count and the largest count come from the point_count column', async () => {
    const result = await runCuratedAnalysis('count-by-area');
    const counts = getTable(result, 'recipe-choropleth').columns['point_count'] as number[];
    const total = counts.reduce((sum, value) => sum + value, 0);
    const report = describeKeplerCatalogResult(
      result,
      getBinding('recipe-points-in-polygons-choropleth'),
      ALL_DESCRIPTORS.find(item => item.bindingId === 'recipe-points-in-polygons-choropleth')!
    );
    expect(total).toBeGreaterThan(0);
    expect(report.outputs[0].usedTemplate).toBe('count-per-area');
    expect(report.sentence).toBe(
      `${group(total)} points across ${counts.length} areas: ${(total / counts.length).toFixed(2).replace(/\.?0+$/, '')} per area on average, up to ${Math.max(...counts)} in one area.`
    );
  });

  it('count by area: points in overlapping areas are reported as memberships, not as points', async () => {
    // Two unit-overlapping squares, [0, 2] and [1, 3] on x. Three points: one in the first only,
    // one in the overlap, one in the second only. Each area holds 2, so the counts add up to 4
    // for 3 points.
    const makeSquare = (west: number): CatalogFeatureCollection['features'][number] => ({
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [west, 0],
            [west + 2, 0],
            [west + 2, 2],
            [west, 2],
            [west, 0]
          ]
        ]
      },
      properties: {}
    });
    const polygons: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [makeSquare(0), makeSquare(1)]
    };
    const points: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [0.5, 1.5, 2.5].map(x => ({
        geometry: {type: 'Point', coordinates: [x, 1]},
        properties: {}
      }))
    };
    const binding = getBinding('recipe-points-in-polygons-choropleth');
    const descriptor = getDescriptor('recipe-points-in-polygons-choropleth');
    const result = await runCatalogOperationAsync(
      binding,
      {statistic: 'count', coordinateSystem: 'planar'},
      {points, polygons},
      {engine: 'cpu'}
    );
    expect(getTable(result, 'recipe-choropleth').columns['point_count']).toEqual([2, 2]);
    expect(result.summary['multipleMatchCount']).toBe(1);
    const report = describeKeplerCatalogResult(result, binding, descriptor);
    expect(report.sentence).toMatch(/^4 point memberships across 2 areas/);
    expect(report.sentence).toMatch(
      /A point in overlapping areas contributes once to each area\.$/
    );

    // Without an overlap the same producer counts points.
    const disjoint = await runCatalogOperationAsync(
      binding,
      {statistic: 'count', coordinateSystem: 'planar'},
      {points, polygons: {type: 'FeatureCollection', features: [makeSquare(0), makeSquare(2.25)]}},
      {engine: 'cpu'}
    );
    expect(disjoint.summary['multipleMatchCount']).toBe(0);
    const disjointReport = describeKeplerCatalogResult(disjoint, binding, descriptor);
    expect(disjointReport.sentence).toMatch(/^3 points across 2 areas/);
    expect(disjointReport.sentence).not.toMatch(/membership|overlapping/);
  });

  it('regression: R squared, adjusted R squared and the observation count come from the summary', async () => {
    // Oracle: /usr/bin/python3 with NumPy lstsq on these 14 rows,
    //   x1 = 1..14, x2 = [5,3,8,1,9,2,7,4,6,10,3,8,2,9], y = 3 + 2*x1 - 1.5*x2 + 8*noise,
    //   noise = [0.4,-0.7,0.2,0.9,-0.5,0.3,-0.8,0.6,0.1,-0.3,0.7,-0.2,0.5,-0.6]
    // gives R squared 0.9030990487 and adjusted R squared 0.8854806939.
    const x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
    const x2 = [5, 3, 8, 1, 9, 2, 7, 4, 6, 10, 3, 8, 2, 9];
    const noise = [0.4, -0.7, 0.2, 0.9, -0.5, 0.3, -0.8, 0.6, 0.1, -0.3, 0.7, -0.2, 0.5, -0.6];
    const points: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: x1.map((value, index) => ({
        geometry: {
          type: 'Point',
          coordinates: [-87.65 + index * 0.003, 41.85 + (index % 4) * 0.002]
        },
        properties: {
          x1: value,
          x2: x2[index],
          y: Math.round((3 + 2 * value - 1.5 * x2[index] + 8 * noise[index]) * 1e6) / 1e6
        }
      }))
    };
    const binding = getBinding('regression-ols');
    const result = await runCatalogOperationAsync(
      binding,
      {responseProperty: 'y', predictorProperties: 'x1,x2'},
      {points},
      {engine: 'cpu'}
    );
    const descriptor = getDescriptor('regression-ols');
    const report = describeKeplerCatalogResult(result, binding, descriptor);
    expect(report.sentence).toBe(
      'R² is 0.90 on 14 observations (adjusted R² 0.89), a measure of how closely the model fits the data rather than of cause.'
    );
    expect(result.summary['rowCount']).toBe(14);
    expect(report.outputs.map(output => output.usedTemplate)).toEqual([
      'regression-fit',
      'regression-fit'
    ]);

    // The sentence follows the summary: change R squared there and the sentence changes.
    const changed = describeKeplerCatalogResult(
      {...result, summary: {...result.summary, rSquared: 0.5, adjustedRSquared: 0.4}},
      binding,
      descriptor
    );
    expect(changed.sentence).toMatch(/^R² is 0\.50 on 14 observations \(adjusted R² 0\.40\)/);
    // An R squared that is not a number, or a missing key, cannot be stated.
    for (const summary of [
      {...result.summary, rSquared: Number.NaN},
      {...result.summary, rSquared: 1.5},
      {...result.summary, rowCount: undefined as unknown as number}
    ]) {
      const fallback = describeKeplerCatalogResult({...result, summary}, binding, descriptor);
      expect(fallback.outputs[0].usedTemplate).toBe('row-count');
      // The coefficient table has an intercept and two predictors.
      expect(fallback.sentence).toBe('The result has 3 rows.');
    }
  });

  it('global statistic: Moran I and its p-value come from the row of the global table', async () => {
    const areas = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];
    const binding = getBinding('global-spatial-statistics');
    const result = await runCatalogOperationAsync(
      binding,
      {
        valueProperty: 'rate',
        neighborhood: 'queen',
        coordinateSystem: 'EPSG:4326'
      },
      {features: areas},
      {engine: 'cpu'}
    );
    const table = getTable(result, 'global');
    const row = table.columns['statistic'].indexOf('morans-i');
    const value = table.columns['value'][row] as number;
    const pValue = table.columns['p_value'][row] as number;
    const report = describeKeplerCatalogResult(
      result,
      binding,
      ALL_DESCRIPTORS.find(item => item.bindingId === 'global-spatial-statistics')!
    );
    const pText = pValue > 0 && pValue < 0.001 ? 'less than 0.001' : pValue.toFixed(3);
    expect(report.sentence).toContain(`Moran's I is ${formatMeasured(value)}`);
    expect(report.sentence).toContain(`p-value ${pText}`);
    // An analytic test has no permutations, so the pseudo p-value caution does not apply.
    expect(report.sentence).not.toContain('pseudo');
    expect(getCautionIds(report)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------------------------
// Sentences that cannot be built
// ---------------------------------------------------------------------------------------------

describe('result report: falling back to the row count', () => {
  it('local-lisa-variants has only a significance flag, so the sentence is the row count', () => {
    const table = makeTable('local-lisa', {significant: [true, false, false, true]});
    const result = makeResult({
      tables: [table],
      parameters: {significance: 0.05, permutations: 99},
      summary: {featureCount: 4, permutations: 99}
    });
    const report = describeKeplerCatalogResult(
      result,
      getBinding('local-lisa-variants'),
      getDescriptor('local-lisa-variants', 'contributor-integer')
    );
    expect(report.outputs).toEqual([
      {
        tableId: 'local-lisa',
        template: 'local-cluster-count',
        usedTemplate: 'row-count',
        sentence: 'The result has 4 rows.'
      }
    ]);
  });

  it('regression-gwr, regression-probit and regression-sur publish no R squared, so their sentences are the range of what they publish', () => {
    const cases = [
      [
        'regression-gwr',
        'gwr-local',
        {residual: [-1.5, 2, 3]},
        'Residual ranges from -1.5 to 3 across 3 values.'
      ],
      [
        'regression-probit',
        'probit-fitted',
        {probability: [0.1, 0.2, 0.9]},
        'Probability ranges from 0.1 to 0.9 across 3 values.'
      ],
      [
        'regression-sur',
        'sur-residuals',
        {residual_0: [-2, 0, 4]},
        'Residual 0 ranges from -2 to 4 across 3 values.'
      ]
    ] as const;
    for (const [bindingId, tableId, columns, expected] of cases) {
      const result = makeResult({
        tables: [
          makeTable(tableId, JSON.parse(JSON.stringify(columns)) as Record<string, ColumnValue[]>)
        ],
        summary: {rowCount: 3, residualSumOfSquares: 4.5}
      });
      const report = describeKeplerCatalogResult(
        result,
        getBinding(bindingId),
        getDescriptor(bindingId)
      );
      expect(report.outputs[0].usedTemplate, bindingId).toBe('value-range');
      expect(report.sentence, bindingId).toBe(expected);
    }
    // Their coefficient tables have no R squared either, so they are the row count.
    const probit = describeKeplerCatalogResult(
      makeResult({
        tables: [makeTable('probit-coefficients', {term: ['a', 'b']})],
        summary: {rowCount: 3}
      }),
      getBinding('regression-probit'),
      getDescriptor('regression-probit')
    );
    expect(probit.outputs[0].usedTemplate).toBe('row-count');
    expect(probit.sentence).toBe('The result has 2 rows.');
  });

  it('a hot spot result whose counts are inconsistent is not stated', () => {
    const bins = [3, 3, 2, 0, 0, 0];
    const result = makeResult({
      tables: [makeTable('recipe-hot-spots', {bin: bins})],
      parameters: {multipleTesting: 'none'},
      // One tested cell cannot hold three hot cells.
      summary: {testedCellCount: 1}
    });
    const report = describeKeplerCatalogResult(
      result,
      getBinding('recipe-hot-spot-lattice'),
      getDescriptor('recipe-hot-spot-lattice', 'contributor-integer')
    );
    expect(report.outputs[0].usedTemplate).toBe('row-count');
    expect(report.sentence).toBe('The result has 6 rows.');
    // Without the tested count at all the same happens.
    const missing = describeKeplerCatalogResult(
      {...result, summary: {}},
      getBinding('recipe-hot-spot-lattice'),
      getDescriptor('recipe-hot-spot-lattice', 'contributor-integer')
    );
    expect(missing.outputs[0].usedTemplate).toBe('row-count');
    // An unknown correction value cannot be named.
    const unknown = describeKeplerCatalogResult(
      {...result, summary: {testedCellCount: 6}, parameters: {multipleTesting: 'whatever'}},
      getBinding('recipe-hot-spot-lattice'),
      getDescriptor('recipe-hot-spot-lattice', 'contributor-integer')
    );
    expect(unknown.outputs[0].usedTemplate).toBe('row-count');
    // The consistent version is stated.
    const consistent = describeKeplerCatalogResult(
      {...result, summary: {testedCellCount: 6}},
      getBinding('recipe-hot-spot-lattice'),
      getDescriptor('recipe-hot-spot-lattice', 'contributor-integer')
    );
    expect(consistent.sentence).toMatch(
      /^3 of 6 cells are significant hot spots and 0 are (significant )?cold spots at 90% confidence without correction for multiple testing\.$/
    );
  });

  it('a hot spot table with no rows says there are no cells to test', () => {
    const result = makeResult({
      tables: [makeTable('recipe-hot-spots', {bin: []})],
      parameters: {multipleTesting: 'none'},
      summary: {testedCellCount: 0}
    });
    const report = describeKeplerCatalogResult(
      result,
      getBinding('recipe-hot-spot-lattice'),
      getDescriptor('recipe-hot-spot-lattice', 'contributor-integer')
    );
    expect(report.sentence).toBe('There are no cells to test.');
  });

  it('local Moran without a significance parameter cannot name its level', async () => {
    const result = await runCuratedAnalysis('cluster-and-outlier-map');
    const {significance: _removed, ...parameters} = result.parameters;
    const report = describeKeplerCatalogResult(
      {...result, parameters},
      getBinding('local-moran-statistics'),
      getDescriptor('local-moran-statistics')
    );
    expect(report.outputs[0].usedTemplate).toBe('row-count');
  });

  it('a text column that is not the scheme gives the row count, not a made-up classification', () => {
    const result = makeResult({
      tables: [makeTable('local-moran', {cluster: ['a', 'b', 'c', 'a']})],
      parameters: {significance: 0.05, permutations: 99}
    });
    const report = describeKeplerCatalogResult(
      result,
      getBinding('local-moran-statistics'),
      getDescriptor('local-moran-statistics')
    );
    expect(report.sentence).toBe('The result has 4 rows.');
  });

  it('knox: reads the ratio and p-value of its one row, and falls back when the ratio is null', () => {
    const binding = getBinding('knox-test');
    const descriptor = ALL_DESCRIPTORS.find(item => item.bindingId === 'knox-test')!;
    const make = (ratio: number | null) =>
      makeResult({
        tables: [makeTable('knox', {ratio: [ratio], p_value: [0.012], joint_close_pairs: [18]})],
        parameters: {permutations: 99}
      });
    const report = describeKeplerCatalogResult(make(1.8), binding, descriptor);
    expect(report.sentence).toContain(
      'The ratio of observed to expected pairs that are close in space and time is 1.8'
    );
    expect(report.sentence).toContain('pseudo p-value 0.012 from 99 permutations');
    expect(getCautionIds(report)).toEqual(['pseudo-p-values']);
    expect(describeKeplerCatalogResult(make(null), binding, descriptor).sentence).toBe(
      'The result has 1 row.'
    );
  });

  it('a result with no table of the descriptor reports the rows of whatever tables it has', () => {
    const result = makeResult({tables: [makeTable('unexpected', {a: [1, 2, 3]})]});
    const report = describeKeplerCatalogResult(
      result,
      getBinding('local-moran-statistics'),
      getDescriptor('local-moran-statistics')
    );
    expect(report.outputs).toEqual([]);
    expect(report.sentence).toBe('The result has 3 rows.');
  });
});

// ---------------------------------------------------------------------------------------------
// Chosen parameters
// ---------------------------------------------------------------------------------------------

describe('result report: what was chosen', () => {
  const binding = getBinding('local-moran-statistics');
  const descriptor = getDescriptor('local-moran-statistics');
  const emptyTable = makeTable('local-moran', {cluster: []});

  function describeChosen(
    parameters: Record<string, CatalogParameterValue>,
    resolutions: NonNullable<Parameters<typeof describeKeplerCatalogResult>[3]>['resolutions']
  ): readonly string[] {
    return describeKeplerCatalogResult(
      makeResult({
        tables: [emptyTable],
        parameters: {neighborhood: 'distance-band', ...parameters}
      }),
      binding,
      descriptor,
      {resolutions, profiles: {features: {kind: 'points'}}}
    ).chosen;
  }

  it('the binding labels the parameter the way the lines assume', () => {
    expect(binding.parameters.find(item => item.name === 'distanceBand')?.label).toBe(
      'Distance band'
    );
  });

  it('data: label, value with unit, then the reason without the bracketed value', () => {
    expect(
      describeChosen({distanceBand: 1240}, [
        {
          parameter: 'distanceBand',
          source: 'data',
          resolution: makeResolution({
            value: 1240,
            unit: 'metre',
            explanation:
              'Largest nearest-neighbour distance (1,240 m), so every point has a neighbour'
          })
        }
      ])
    ).toEqual([
      'Distance band 1,240 m: largest nearest-neighbour distance, so every point has a neighbour'
    ]);
  });

  it('data computed on a sample says so, and a proper name keeps its capital', () => {
    expect(
      describeChosen({distanceBand: 400.5}, [
        {
          parameter: 'distanceBand',
          source: 'data',
          resolution: makeResolution({
            sampled: true,
            value: 400.5,
            unit: 'coordinate-unit',
            explanation: "Scott's normal reference bandwidth (400.5)"
          })
        }
      ])
    ).toEqual([
      "Distance band 401: Scott's normal reference bandwidth (computed on a sample of the rows)"
    ]);
    // An acronym is not lower-cased either.
    expect(
      describeChosen({distanceBand: 90}, [
        {
          parameter: 'distanceBand',
          source: 'data',
          resolution: makeResolution({
            unit: 'degree',
            explanation: 'IDW search radius (90 degrees)'
          })
        }
      ])
    ).toEqual(['Distance band 90°: IDW search radius']);
  });

  it('data uses the value the run used, not the one the resolution proposed', () => {
    expect(
      describeChosen({distanceBand: 2000}, [
        {
          parameter: 'distanceBand',
          source: 'data',
          resolution: makeResolution({
            value: 1240,
            unit: 'metre',
            explanation: 'Median spacing (1,240 m)'
          })
        }
      ])
    ).toEqual(['Distance band 2,000 m: median spacing']);
  });

  it('data without a resolution says only that it was derived', () => {
    expect(
      describeChosen({distanceBand: 300}, [{parameter: 'distanceBand', source: 'data'}])
    ).toEqual(['Distance band 300: derived from your data']);
  });

  it('override: set by you', () => {
    expect(
      describeChosen({distanceBand: 500}, [{parameter: 'distanceBand', source: 'override'}])
    ).toEqual(['Distance band 500: set by you']);
    expect(
      describeChosen({permutations: 999}, [{parameter: 'permutations', source: 'override'}])
    ).toEqual(['Conditional permutations (0 disables) 999: set by you']);
  });

  it('fallback: the default could not be derived', () => {
    expect(
      describeChosen({distanceBand: 1000}, [{parameter: 'distanceBand', source: 'fallback'}])
    ).toEqual([
      'Distance band 1,000: the default could not be derived from your data, so the fixed fallback was used'
    ]);
  });

  it('a parameter that has no effect in this run is not listed (activeWhen)', () => {
    const records = [
      {parameter: 'distanceBand', source: 'override'} as const,
      {parameter: 'neighborCount', source: 'override'} as const
    ];
    const lines = (neighborhood: string) =>
      describeKeplerCatalogResult(
        makeResult({
          tables: [emptyTable],
          parameters: {neighborhood, distanceBand: 500, neighborCount: 8}
        }),
        binding,
        descriptor,
        {resolutions: records}
      ).chosen;
    // The descriptor says the band matters only for `distance-band` and the count only for `knn`.
    expect(lines('knn')).toEqual(['Nearest neighbors (knn) 8: set by you']);
    expect(lines('distance-band')).toEqual(['Distance band 500: set by you']);
    expect(lines('queen')).toEqual([]);
    // A record the defaults engine marked inactive is skipped too.
    expect(
      describeKeplerCatalogResult(
        makeResult({
          tables: [emptyTable],
          parameters: {neighborhood: 'distance-band', distanceBand: 500}
        }),
        binding,
        descriptor,
        {resolutions: [{parameter: 'distanceBand', source: 'override', active: false}]}
      ).chosen
    ).toEqual([]);
  });

  it('a value fixed by the curated analysis is set by this analysis, with the descriptor reason', () => {
    const reason = descriptor.parameters.find(item => item.name === 'permutations')!.defaultPolicy;
    expect(reason.from).toBe('preset');
    const presetReason = reason.from === 'preset' ? reason.reason : '';
    // A `preset` record carries its reason.
    expect(
      describeChosen({permutations: 999}, [
        {parameter: 'permutations', source: 'preset', reason: 'Because tests say so'}
      ])
    ).toEqual([
      'Conditional permutations (0 disables) 999: set by this analysis: Because tests say so'
    ]);
    // An override of a parameter the curated analysis fixed is not the user's: the reason comes from the descriptor.
    const report = describeKeplerCatalogResult(
      makeResult({tables: [emptyTable], parameters: {permutations: 999}}),
      binding,
      descriptor,
      {
        resolutions: [{parameter: 'permutations', source: 'override'}],
        presetParameters: ['permutations']
      }
    );
    expect(report.chosen).toEqual([
      `Conditional permutations (0 disables) 999: set by this analysis: ${presetReason}`
    ]);
    // Without the list the same record is the user's.
    expect(
      describeChosen({permutations: 999}, [{parameter: 'permutations', source: 'override'}])
    ).toEqual(['Conditional permutations (0 disables) 999: set by you']);
  });

  it('the coordinate system is declared or guessed according to the profile', () => {
    const lines = (source: 'guess' | 'declared' | undefined) =>
      describeKeplerCatalogResult(
        makeResult({tables: [emptyTable], parameters: {coordinateSystem: 'EPSG:4326'}}),
        binding,
        descriptor,
        {
          resolutions: [{parameter: 'coordinateSystem', source: 'data'}],
          profiles: {
            features: source
              ? {kind: 'polygons', coordinates: {model: 'geographic', source}}
              : undefined
          }
        }
      ).chosen;
    expect(lines('declared')).toEqual(['Coordinate system EPSG:4326: declared for the dataset']);
    expect(lines('guess')).toEqual([
      'Coordinate system EPSG:4326: guessed from the coordinate values'
    ]);
    expect(lines(undefined)).toEqual([
      'Coordinate system EPSG:4326: taken from the dataset profile'
    ]);
  });

  it('a guessed coordinate system is shown as a guess, and a low-confidence guess gets a line of its own', () => {
    const lines = (confidence: 'high' | 'low', guessed = true) =>
      describeKeplerCatalogResult(
        makeResult({tables: [emptyTable], parameters: {coordinateSystem: 'EPSG:4326'}}),
        binding,
        descriptor,
        {
          resolutions: [
            {
              parameter: 'coordinateSystem',
              source: 'data',
              ...(guessed
                ? {
                    guessed: true,
                    reason: `The coordinate model is a guess (geographic${confidence === 'low' ? ', low confidence' : ''}) from the extent, not a declaration`
                  }
                : {})
            }
          ],
          profiles: {
            features: {
              kind: 'polygons',
              coordinates: {
                model: 'geographic',
                source: guessed ? 'guess' : 'declared',
                confidence,
                reason: 'the extent is a few degrees wide'
              }
            }
          }
        }
      );
    expect(lines('high').chosen).toEqual([
      'Coordinate system EPSG:4326: guessed from the coordinate values'
    ]);
    expect(lines('low').chosen).toEqual([
      'Coordinate system EPSG:4326: guessed from the coordinate values',
      'Coordinate system guess has low confidence (the extent is a few degrees wide): confirm it before trusting distances or areas'
    ]);
    // A declared system is never a guess, whatever the confidence field says.
    expect(lines('low', false).chosen).toEqual([
      'Coordinate system EPSG:4326: declared for the dataset'
    ]);
    expect(getCautionIds(lines('low'))).toContain('guessed-coordinates');
    expect(getCautionIds(lines('low', false))).not.toContain('guessed-coordinates');
  });

  it('a fallback prints the reason the engine gives', () => {
    expect(
      describeChosen({distanceBand: 1000}, [
        {parameter: 'distanceBand', source: 'fallback', reason: 'the points are all at one place'}
      ])
    ).toEqual([
      'Distance band 1,000: the default could not be derived from your data (the points are all at one place), so the fixed fallback was used'
    ]);
  });

  it('says feature, not point, unless the input is known to hold points', () => {
    const resolution = makeResolution({
      value: 670,
      unit: 'metre',
      explanation:
        'Largest nearest-neighbour distance (663 m) times 1.01 = 670 m, so every point has at least one neighbour.'
    });
    const lines = (kind: 'points' | 'polygons' | undefined) =>
      describeKeplerCatalogResult(
        makeResult({
          tables: [emptyTable],
          parameters: {neighborhood: 'distance-band', distanceBand: 670}
        }),
        binding,
        descriptor,
        {
          resolutions: [{parameter: 'distanceBand', source: 'data', resolution}],
          profiles: {features: kind ? {kind} : undefined}
        }
      ).chosen;
    expect(lines('points')).toEqual([
      'Distance band 670 m: largest nearest-neighbour distance (663 m) times 1.01, so every point has at least one neighbour'
    ]);
    expect(lines('polygons')).toEqual([
      'Distance band 670 m: largest nearest-neighbour distance (663 m) times 1.01, so every feature has at least one neighbour'
    ]);
    expect(lines(undefined)).toEqual(lines('polygons'));
  });

  it('constants get no line, and an unknown parameter keeps its name', () => {
    expect(
      describeChosen({neighborhood: 'queen', mystery: 5}, [
        {parameter: 'neighborhood', source: 'constant'},
        {parameter: 'mystery', source: 'override'}
      ])
    ).toEqual(['mystery 5: set by you']);
  });

  it('a text value is shown as it is', () => {
    expect(
      describeChosen({neighborhood: 'knn'}, [{parameter: 'neighborhood', source: 'override'}])
    ).toEqual(['Neighborhood knn: set by you']);
  });
});

// ---------------------------------------------------------------------------------------------
// Cautions from context
// ---------------------------------------------------------------------------------------------

describe('result report: cautions from context', () => {
  const binding = getBinding('recipe-hot-spot-lattice');
  const descriptor = getDescriptor('recipe-hot-spot-lattice', 'contributor-integer');
  const result = makeResult({
    tables: [makeTable('recipe-hot-spots', {bin: [0, 0, 0, 0]})],
    parameters: {multipleTesting: 'benjamini-hochberg'},
    summary: {testedCellCount: 4}
  });
  // A corrected result with 4 cells carries only the small-number caution to start with.
  const baseline = ['fewer-than-thirty-features'];

  it('baseline', () => {
    expect(getCautionIds(describeKeplerCatalogResult(result, binding, descriptor))).toEqual(
      baseline
    );
  });

  it('each context fact adds its own caution and only that one', () => {
    const cases: [string, Parameters<typeof describeKeplerCatalogResult>[3], string][] = [
      ['guessed coordinates', {guessedCoordinates: true}, 'guessed-coordinates'],
      ['parameter search', {parameterSearch: true}, 'exploratory-parameter-search'],
      [
        'sampled resolution',
        {
          resolutions: [
            {
              parameter: 'cellSize',
              source: 'data',
              resolution: makeResolution({explanation: 'Extent', sampled: true})
            }
          ]
        },
        'sampled-statistics'
      ],
      [
        'fallback default',
        {resolutions: [{parameter: 'cellSize', source: 'fallback'}]},
        'default-from-fallback'
      ]
    ];
    for (const [name, context, expected] of cases) {
      const ids = getCautionIds(describeKeplerCatalogResult(result, binding, descriptor, context));
      expect(ids, name).toContain(expected);
      expect(
        ids.filter(id => !baseline.includes(id)),
        name
      ).toEqual([expected]);
    }
  });

  it('a count or amount column bound to a role that does not list that level raises the measurement caution', () => {
    const moran = getBinding('local-moran-statistics');
    const moranDescriptor = getDescriptor('local-moran-statistics');
    // The value role of local Moran lists rate and amount, not count.
    const moranRole = moranDescriptor.inputs[0].fields.find(
      field => field.parameter === 'valueProperty'
    )!;
    expect(moranRole.measurement).toEqual(expect.arrayContaining(['rate', 'amount']));
    expect(moranRole.measurement).not.toContain('count');
    const table = makeTable('local-moran', {
      cluster: Array.from({length: 40}, () => 'not-significant')
    });
    const run = makeResult({tables: [table], parameters: {significance: 0.05, permutations: 999}});
    const ids = (level: string | undefined, column = 'cases') =>
      getCautionIds(
        describeKeplerCatalogResult(run, moran, moranDescriptor, {
          fieldBindings: {valueProperty: column},
          profiles: {
            features: {
              columns: [
                {name: 'cases', measurement: level as 'count' | undefined},
                {name: 'rate', measurement: 'rate'}
              ]
            }
          }
        })
      );
    expect(ids('count')).toContain('counts-where-rates-expected');
    // A rate column, a level the role lists, an unknown level and an unknown column raise nothing.
    expect(ids('rate', 'rate')).not.toContain('counts-where-rates-expected');
    expect(ids('amount')).not.toContain('counts-where-rates-expected');
    expect(ids(undefined)).not.toContain('counts-where-rates-expected');
    expect(ids('count', 'not-a-column')).not.toContain('counts-where-rates-expected');
    // No field bindings or profiles, nothing to compare.
    expect(getCautionIds(describeKeplerCatalogResult(run, moran, moranDescriptor))).not.toContain(
      'counts-where-rates-expected'
    );
    // A list of columns is checked column by column.
    expect(
      getCautionIds(
        describeKeplerCatalogResult(run, moran, moranDescriptor, {
          fieldBindings: {valueProperty: ['rate', 'cases']},
          profiles: {
            features: {
              columns: [
                {name: 'cases', measurement: 'count'},
                {name: 'rate', measurement: 'rate'}
              ]
            }
          }
        })
      )
    ).toContain('counts-where-rates-expected');
  });

  it('a failed row-semantics requirement is about what rows stand for and raises no measurement caution', () => {
    // The old trigger: a true rate column with a failed row-semantics check was told it "holds counts".
    const moran = getBinding('local-moran-statistics');
    const table = makeTable('local-moran', {
      cluster: Array.from({length: 40}, () => 'not-significant')
    });
    const report = describeKeplerCatalogResult(
      makeResult({tables: [table], parameters: {significance: 0.05, permutations: 999}}),
      moran,
      getDescriptor('local-moran-statistics'),
      {
        fieldBindings: {valueProperty: 'rate'},
        profiles: {features: {columns: [{name: 'rate', measurement: 'rate'}]}}
      }
    );
    expect(getCautionIds(report)).not.toContain('counts-where-rates-expected');
  });

  it('host warnings are passed through', () => {
    const report = describeKeplerCatalogResult(result, binding, descriptor, {
      warnings: ['Two rows were dropped.']
    });
    expect(report.warnings).toEqual(['Two rows were dropped.']);
    expect(describeKeplerCatalogResult(result, binding, descriptor).warnings).toEqual([]);
  });

  it('lattice cells without a z-score raise the rows-without-neighbours caution', () => {
    const edges = makeResult({
      tables: [makeTable('recipe-hot-spots', {bin: Array.from({length: 40}, () => 0)})],
      parameters: {multipleTesting: 'none'},
      summary: {testedCellCount: 36}
    });
    // Cautions come in the order of the analysis package's list.
    expect(getCautionIds(describeKeplerCatalogResult(edges, binding, descriptor))).toEqual([
      'multiple-testing-uncorrected',
      'rows-without-neighbors'
    ]);
  });

  it('island diagnostics raise the rows-without-neighbours caution for the local statistics', () => {
    const moran = getBinding('local-moran-statistics');
    const table = makeTable('local-moran', {
      cluster: Array.from({length: 40}, () => 'not-significant')
    });
    const islands = makeResult({
      tables: [table],
      parameters: {significance: 0.05, permutations: 999},
      diagnostics: [
        {
          code: 'island-features',
          severity: 'warning',
          count: 3,
          message: 'Features without neighbors'
        }
      ]
    });
    const ids = getCautionIds(
      describeKeplerCatalogResult(islands, moran, getDescriptor('local-moran-statistics'))
    );
    expect(ids).toContain('rows-without-neighbors');
    const without = getCautionIds(
      describeKeplerCatalogResult(
        {...islands, diagnostics: []},
        moran,
        getDescriptor('local-moran-statistics')
      )
    );
    expect(without).not.toContain('rows-without-neighbors');
  });

  it('the feature counts of the context stand in when the result does not say how many were analysed', () => {
    const knox = getBinding('knox-test');
    const descriptorKnox = getDescriptor('knox-test');
    const noSummary = makeResult({
      tables: [makeTable('knox', {ratio: [1.8], p_value: [0.4], joint_close_pairs: [18]})],
      parameters: {permutations: 999},
      summary: {}
    });
    // The binding's summary holds `eventCount`; here it is missing, so the host's count is used.
    expect(
      getCautionIds(
        describeKeplerCatalogResult(noSummary, knox, descriptorKnox, {featureCounts: {points: 12}})
      )
    ).toContain('fewer-than-thirty-features');
    expect(
      getCautionIds(
        describeKeplerCatalogResult(noSummary, knox, descriptorKnox, {featureCounts: {points: 400}})
      )
    ).not.toContain('fewer-than-thirty-features');
    expect(
      getCautionIds(describeKeplerCatalogResult(noSummary, knox, descriptorKnox))
    ).not.toContain('fewer-than-thirty-features');
    // With the summary the analysed count decides, whatever the host says.
    const withSummary = {...noSummary, summary: {eventCount: 16}};
    expect(
      getCautionIds(
        describeKeplerCatalogResult(withSummary, knox, descriptorKnox, {
          featureCounts: {points: 400}
        })
      )
    ).toContain('fewer-than-thirty-features');
  });

  it('a global Moran on 16 areas raises the fewer-than-thirty caution from the summary', async () => {
    const binding = getBinding('global-spatial-statistics');
    const result = await runCatalogOperationAsync(
      binding,
      {...LATTICE_PARAMETERS},
      {features: makeLattice(4)},
      {engine: 'cpu'}
    );
    expect(result.summary['featureCount']).toBe(16);
    expect(
      getCautionIds(
        describeKeplerCatalogResult(result, binding, getDescriptor('global-spatial-statistics'))
      )
    ).toContain('fewer-than-thirty-features');
  });

  it('a binding that is not a statistical test never gets the small-sample caution', async () => {
    const result = await runCuratedAnalysis('idw-surface');
    const report = describeKeplerCatalogResult(
      result,
      getBinding('idw-grid'),
      ALL_DESCRIPTORS.find(item => item.bindingId === 'idw-grid')!,
      {featureCounts: {points: 5}}
    );
    expect(getCautionIds(report)).toEqual([]);
  });
});

// ---------------------------------------------------------------------------------------------
// Engine, next steps and text
// ---------------------------------------------------------------------------------------------

describe('result report: engine, next questions and text', () => {
  const binding = getBinding('hotspot-gi-star');
  const descriptor = getDescriptor('hotspot-gi-star');
  const table = makeTable('hotspots', {
    classification: ['hot-99', 'not-significant', 'cold-90', 'not-significant']
  });

  it('names the engine and the contributors for a GPU run, and has no fallback reason', () => {
    const report = describeKeplerCatalogResult(
      makeResult({
        tables: [table],
        engine: 'gpu',
        dispatchedContributors: ['GPUHotSpotAnalysis'],
        summary: {featureCount: 4}
      }),
      binding,
      descriptor
    );
    expect(report.engine).toBe('Computed on the GPU (GPUHotSpotAnalysis).');
    expect(report.fallbackReason).toBeUndefined();
    expect(
      describeKeplerCatalogResult(makeResult({tables: [table], engine: 'gpu'}), binding, descriptor)
        .engine
    ).toBe('Computed on the GPU.');
  });

  it('puts each fallback reason in plain words', () => {
    const reasons: [string, string][] = [
      ['CPU engine requested', 'The CPU engine was requested, so the GPU was not used.'],
      [
        'This operation has no GPU path',
        'This operation has no GPU version, so it ran on the CPU.'
      ],
      ['No GPU device supplied', 'No GPU was available, so the operation ran on the CPU.'],
      [
        'Out of memory',
        'The GPU could not run this operation (Out of memory), so it ran on the CPU.'
      ]
    ];
    for (const [reason, words] of reasons) {
      expect(
        describeKeplerCatalogResult(
          makeResult({tables: [table], fallbackReason: reason}),
          binding,
          descriptor
        ).fallbackReason
      ).toBe(words);
    }
  });

  it('lists the next questions of every output once, with the labels of the analysis package', () => {
    const clustering = getDescriptor('density-clustering', 'catalog-signed-label');
    const report = describeKeplerCatalogResult(
      makeResult({
        tables: [makeTable('points', {cluster: [0, -1]}), makeTable('clusters', {size: [1]})]
      }),
      getBinding('density-clustering'),
      clustering
    );
    const expected = [...new Set(clustering.outputs.flatMap(output => output.next ?? []))];
    expect(expected.length).toBeGreaterThan(0);
    expect(report.next).toEqual(
      expected.map(question => ({question, label: getGPUAnalysisQuestion(question).label}))
    );
  });

  it('formats the report as plain text with a section for each part and none for an empty one', () => {
    const report: KeplerResultReport = {
      sentence: 'Found 3 clusters.',
      outputs: [],
      notes: ['This form uses row-standardized weights.'],
      chosen: ['Distance band 1,240 m: largest nearest-neighbour distance'],
      cautions: [{id: 'pseudo-p-values', title: 'Pseudo p-values', text: 'Permutation based.'}],
      next: [{question: 'how-related', label: 'How are these things related?'}],
      engine: 'Computed on the CPU with the reference engine.',
      fallbackReason: 'No GPU was available, so the operation ran on the CPU.',
      warnings: ['Two rows were dropped.'],
      diagnostics: []
    };
    expect(formatKeplerResultReport(report)).toBe(
      [
        'This form uses row-standardized weights.',
        '',
        'Found 3 clusters.',
        '',
        'What was chosen',
        '- Distance band 1,240 m: largest nearest-neighbour distance',
        '',
        'Read with care',
        '- Pseudo p-values: Permutation based.',
        '',
        'Warnings',
        '- Two rows were dropped.',
        '',
        'Ask next',
        '- How are these things related?',
        '',
        'Computed on the CPU with the reference engine.',
        'No GPU was available, so the operation ran on the CPU.'
      ].join('\n')
    );
    expect(
      formatKeplerResultReport({
        ...report,
        notes: [],
        chosen: [],
        cautions: [],
        warnings: [],
        next: [],
        fallbackReason: undefined
      })
    ).toBe('Found 3 clusters.\n\nComputed on the CPU with the reference engine.');
  });
});

describe('fact table', () => {
  it('names only bindings that exist and parameters those bindings have', () => {
    for (const [bindingId, source] of Object.entries(KEPLER_RESULT_FACT_SOURCES)) {
      const binding = getCatalogBinding(bindingId);
      expect(binding, bindingId).toBeDefined();
      const names = binding!.parameters.map(parameter => parameter.name);
      if (source.permutationsParameter) expect(names).toContain(source.permutationsParameter);
      if (source.significanceParameter) expect(names).toContain(source.significanceParameter);
      if (source.correction && 'parameter' in source.correction) {
        expect(names).toContain(source.correction.parameter);
        // Every value the table lists is an option of the parameter.
        const options = binding!.parameters.find(
          parameter => parameter.name === (source.correction as {parameter: string}).parameter
        )!.options!;
        for (const value of Object.keys(source.correction.values)) expect(options).toContain(value);
      }
      if (source.statistic) {
        expect(binding!.outputs.map(output => output.tableId)).toContain(source.statistic.tableId);
      }
    }
  });
});

// ---------------------------------------------------------------------------------------------
// The defaults engine, field columns and clamped values
// ---------------------------------------------------------------------------------------------

describe('result report: parameters from the real defaults engine', () => {
  async function runResolved(
    bindingId: string,
    inputName: string,
    collection: CatalogFeatureCollection,
    fieldBindings: Record<string, string>
  ) {
    const binding = getBinding(bindingId);
    const descriptor = getDescriptor(bindingId);
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'sample',
      label: 'Sample'
    });
    const resolved = resolveKeplerOperationParameters(
      binding,
      descriptor,
      {[inputName]: profile},
      {fieldBindings}
    );
    const result = await runCatalogOperationAsync(
      binding,
      resolved.parameters,
      {[inputName]: collection},
      {engine: 'cpu'}
    );
    return {
      resolved,
      report: describeKeplerCatalogResult(result, binding, descriptor, {
        resolutions: resolved.resolutions,
        profiles: {[inputName]: profile}
      })
    };
  }

  it('hot spot map on the incidents sample: the column, derived cell size, binding correction and coordinate system', async () => {
    const {resolved, report} = await runResolved(
      'recipe-hot-spot-lattice',
      'points',
      createKeplerCuratedAnalysisSample('city-incidents')['points'],
      {valueProperty: 'severity'}
    );
    const cellSize = resolved.parameters['cellSize'] as number;
    const correction = resolved.resolutions.find(record => record.parameter === 'multipleTesting')!;
    expect(correction.source).toBe('constant');
    expect(resolved.parameters['multipleTesting']).toBe('benjamini-hochberg');
    expect(report.chosen).toHaveLength(3);
    expect(report.chosen[0]).toBe('Value to sum per cell (empty counts points): severity');
    // The derived size, then the reason without its sum and with the input known to be points.
    expect(report.chosen[1]).toMatch(
      new RegExp(`^Lattice cell size ${group(cellSize)} m: [a-z].*`)
    );
    expect(report.chosen[1]).not.toContain(' = ');
    // The profile guessed the coordinate system, and the report says so, with the caution.
    expect(report.chosen[2]).toBe(
      'Coordinate system EPSG:4326: guessed from the coordinate values'
    );
    expect(getCautionIds(report)).toContain('guessed-coordinates');
    // The notes of the descriptor travel with the report.
    expect(report.notes).toEqual(getDescriptor('recipe-hot-spot-lattice').notes ?? []);
  });

  it('cluster and outlier map on the district sample: only parameters that act are listed', async () => {
    const {resolved, report} = await runResolved(
      'local-moran-statistics',
      'features',
      createKeplerCuratedAnalysisSample('city-district-rates')['areas'],
      {valueProperty: 'rate'}
    );
    expect(report.chosen[0]).toBe('Value to map: rate');
    const policy = getDescriptor('local-moran-statistics').parameters.find(
      item => item.name === 'distanceBand'
    );
    const bandIsListed = report.chosen.some(line => line.startsWith('Distance band'));
    // The distance band is derived whatever the neighbourhood, but it acts only for `distance-band`.
    if (policy?.activeWhen) {
      expect(bandIsListed).toBe(
        policy.activeWhen.equals.includes(resolved.parameters['neighborhood'] as string)
      );
    }
    expect(
      report.chosen.some(line => line.startsWith('Coordinate system EPSG:4326: guessed'))
    ).toBe(true);
    expect(getCautionIds(report)).toEqual(
      expect.arrayContaining(['pseudo-p-values', 'multiple-testing-uncorrected'])
    );
  });

  it('a column chosen for a role is named by the role, an overridden one is set by you', () => {
    const binding = getBinding('local-moran-statistics');
    const descriptor = getDescriptor('local-moran-statistics');
    const make = (source: 'field' | 'override') =>
      describeKeplerCatalogResult(
        makeResult({
          tables: [makeTable('local-moran', {cluster: []})],
          parameters: {valueProperty: 'fare'}
        }),
        binding,
        descriptor,
        {resolutions: [{parameter: 'valueProperty', source}]}
      ).chosen;
    expect(make('field')).toEqual(['Value to map: fare']);
    expect(make('override')).toEqual(['Value property fare: set by you']);
  });

  it('a derived value that was moved into the range says how', () => {
    const binding = getBinding('recipe-hot-spot-lattice');
    const descriptor = getDescriptor('recipe-hot-spot-lattice', 'contributor-integer');
    const make = (used: number, from: number, reason?: string) =>
      describeKeplerCatalogResult(
        makeResult({
          tables: [makeTable('recipe-hot-spots', {bin: []})],
          parameters: {cellSize: used}
        }),
        binding,
        descriptor,
        {
          resolutions: [
            {
              parameter: 'cellSize',
              source: 'data',
              resolution: makeResolution({explanation: 'Extent over 250 (0.4 m)', unit: 'metre'}),
              clamped: {from, to: used, ...(reason ? {reason} : {})}
            }
          ]
        }
      ).chosen;
    expect(make(1, 0.4)).toEqual([
      'Lattice cell size 1 m: derived from your data; the value derived from your data (0.4) was raised to the smallest this parameter allows'
    ]);
    // The reason the defaults engine gives is printed as it is.
    const reason =
      'A cell size of 94 would make a grid of 70,000 cells over this extent; recipe-hot-spot-lattice allows at most 40,000';
    expect(make(120, 94, reason)).toEqual([
      `Lattice cell size 120 m: derived from your data; the value derived from your data (94) was changed to 120: ${reason}`
    ]);
    expect(make(1e7, 2e9)).toEqual([
      'Lattice cell size 10,000,000 m: derived from your data; the value derived from your data (2,000,000,000) was limited to the largest this parameter allows'
    ]);
  });
});

// ---------------------------------------------------------------------------------------------
// The fused curated bindings, on their curated samples
// ---------------------------------------------------------------------------------------------

/** The part of a local cluster sentence after the colon, from the four counts. */
function describeClusterParts(counts: Record<string, number>): string {
  const parts = [
    ['HH', 'high-high core', 'high-high cores'],
    ['LL', 'low-low core', 'low-low cores'],
    ['HL', 'high-low outlier', 'high-low outliers'],
    ['LH', 'low-high outlier', 'low-high outliers']
  ]
    .filter(([label]) => counts[label] > 0)
    .map(
      ([label, singular, plural]) => `${counts[label]} ${counts[label] === 1 ? singular : plural}`
    );
  return parts.length > 1
    ? `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`
    : parts[0];
}

/** A measured value as the sentences write it: the analysis package's own formatter. */
function formatMeasured(value: number): string {
  return formatGPUAnalysisMeasuredValue(value);
}

describe('result report: the fused curated bindings on their samples', () => {
  it('rate cluster map: counts the classes of the table, names Benjamini-Hochberg and carries the pseudo p-value caution', async () => {
    const result = await runCuratedAnalysis('rate-cluster-map');
    const table = getTable(result, 'rate-clusters');
    const clusters = table.columns['cluster'];
    const counts = Object.fromEntries(
      ['HH', 'LL', 'HL', 'LH', 'not-tested'].map(label => [
        label,
        countWhere(clusters, value => value === label)
      ])
    );
    const significant = counts['HH'] + counts['LL'] + counts['HL'] + counts['LH'];
    const tested = table.rowCount - counts['not-tested'];
    expect(significant).toBeGreaterThan(0);
    expect(result.summary['highHighCount']).toBe(counts['HH']);
    expect(result.summary['testedCount']).toBe(tested);
    const binding = getBinding('curated-rate-cluster-map');
    const descriptor = getDescriptor('curated-rate-cluster-map');
    const report = describeKeplerCatalogResult(result, binding, descriptor);
    expect(report.sentence).toBe(
      `${significant} of ${tested} features are significant at a pseudo p-value of 0.05 or less after false discovery rate correction: ${describeClusterParts(counts)}.`
    );
    expect(getCautionIds(report)).toEqual(['pseudo-p-values']);

    // Without correction the sentence and the cautions change.
    const uncorrected = describeKeplerCatalogResult(
      await runCuratedAnalysis('rate-cluster-map', {multipleTesting: 'none'}),
      binding,
      descriptor
    );
    expect(uncorrected.sentence).toContain('without correction for multiple testing');
    expect(getCautionIds(uncorrected)).toEqual(['pseudo-p-values', 'multiple-testing-uncorrected']);
    // Holm is named; the Sidak option has no name in the sentences, so it is not stated.
    expect(
      describeKeplerCatalogResult(
        await runCuratedAnalysis('rate-cluster-map', {multipleTesting: 'holm'}),
        binding,
        descriptor
      ).sentence
    ).toContain('after Holm correction');
    // Sidak has no name in the analysis package's sentences; the report words it itself.
    const sidak = describeKeplerCatalogResult(
      await runCuratedAnalysis('rate-cluster-map', {multipleTesting: 'sidak'}),
      binding,
      descriptor
    );
    expect(sidak.outputs[0].usedTemplate).toBe('local-cluster-count');
    expect(sidak.sentence).toContain('Šidák correction');
    expect(sidak.sentence).not.toContain('without correction');
  });

  it('rate cluster map with a correction the permutations cannot reach: not a finding, and the numbers agree with the run', async () => {
    const result = await runCuratedAnalysis('rate-cluster-map', {
      multipleTesting: 'bonferroni',
      permutations: 999
    });
    const binding = getBinding('curated-rate-cluster-map');
    const report = describeKeplerCatalogResult(
      result,
      binding,
      getDescriptor('curated-rate-cluster-map')
    );
    const tested = result.summary['testedCount'] as number;
    // The binding computed the same attainability and wrote it to its summary.
    expect(result.summary['significanceNotAttainable']).toBe(1);
    const needed = result.summary['permutationsNeeded'] as number;
    // The report passes the facts and the package words the sentence: nothing is reworded here.
    const table = getTable(result, 'rate-clusters');
    expect(report.sentence).toBe(
      summarizeGPUAnalysisResult('local-cluster-count', {
        featureCount: tested,
        highHighCount: 0,
        lowLowCount: 0,
        highLowCount: 0,
        lowHighCount: 0,
        significanceLevel: 0.05,
        pValueKind: 'pseudo',
        correction: 'bonferroni',
        permutationCount: 999
      })
    );
    expect(
      countWhere(table.columns['cluster'], value =>
        ['HH', 'LL', 'HL', 'LH'].includes(value as string)
      )
    ).toBe(0);
    expect(report.sentence).toContain(`at least ${group(needed)} permutations are needed`);
    expect(report.sentence).toContain(`over ${group(tested)} tests`);
    expect(getCautionIds(report)).toContain('significance-not-attainable');
    const caution = report.cautions.find(item => item.id === 'significance-not-attainable')!;
    expect(caution.text).toBe(
      getGPUAnalysisCautionText('significance-not-attainable', {
        correction: 'bonferroni',
        testCount: tested,
        permutationCount: 999,
        significanceLevel: 0.05
      }).text
    );
    expect(caution.text).toContain(group(needed));
    // With enough permutations the same run is attainable and the sentence is a finding.
    const attainable = describeKeplerCatalogResult(
      await runCuratedAnalysis('rate-cluster-map', {
        multipleTesting: 'bonferroni',
        permutations: needed
      }),
      binding,
      getDescriptor('curated-rate-cluster-map')
    );
    expect(attainable.sentence).not.toContain('permutations are needed');
    expect(getCautionIds(attainable)).not.toContain('significance-not-attainable');
  });

  it('excess risk: the range of the column the descriptor colours by', async () => {
    const result = await runCuratedAnalysis('excess-risk');
    const descriptor = getDescriptor('curated-excess-risk');
    const column = descriptor.outputs[0].presentation!.column;
    const values = (getTable(result, 'excess-risk').columns[column] as (number | null)[]).filter(
      (value): value is number => value !== null && Number.isFinite(value)
    );
    const report = describeKeplerCatalogResult(
      result,
      getBinding('curated-excess-risk'),
      descriptor
    );
    const label = column.replace(/_/g, ' ').replace(/^./, letter => letter.toUpperCase());
    expect(values.length).toBeGreaterThan(0);
    expect(report.sentence).toBe(
      `${label} ranges from ${formatMeasured(Math.min(...values))} to ${formatMeasured(Math.max(...values))} across ${group(values.length)} values.`
    );
    // The summary counts the areas whose expected cases are too few; the caution follows it.
    const smallCount = result.summary['smallDenominatorAreaCount'] as number;
    expect(getCautionIds(report).includes('small-denominators')).toBe(smallCount > 0);
  });

  it('clusters and footprints: three clusters and the noise share from the points table', async () => {
    const result = await runCuratedAnalysis('clusters-and-footprints');
    expect(result.operationId).toBe('curated-clusters-and-footprints');
    const clusters = getTable(result, 'points').columns['cluster'];
    const noiseCount = countWhere(clusters, value => value === -1);
    const report = describeKeplerCatalogResult(
      result,
      getBinding('curated-clusters-and-footprints'),
      getDescriptor('curated-clusters-and-footprints')
    );
    const labels = new Set((clusters as number[]).filter(label => label >= 0));
    expect(report.sentence).toMatch(
      new RegExp(`^Found ${labels.size} clusters; ${noiseCount} of ${clusters.length} points \\(`)
    );
    expect(result.summary['noiseCount']).toBe(noiseCount);
  });

  it('regression with diagnostics: R squared and observations, then the residual Moran I as the second sentence', async () => {
    const result = await runCuratedAnalysis('regression-with-diagnostics');
    const rSquared = result.summary['rSquared'] as number;
    const adjusted = result.summary['adjustedRSquared'] as number;
    const rowCount = result.summary['rowCount'] as number;
    const binding = getBinding('curated-regression-with-diagnostics');
    const descriptor = getDescriptor('curated-regression-with-diagnostics');
    const report = describeKeplerCatalogResult(result, binding, descriptor);
    expect(rSquared).toBeLessThan(0.995);
    expect(report.sentence).toBe(
      `R² is ${rSquared.toFixed(2)} on ${rowCount} observations (adjusted R² ${adjusted.toFixed(2)}), a measure of how closely the model fits the data rather than of cause.`
    );
    const diagnostics = getTable(result, 'diagnostics');
    const row = diagnostics.columns['diagnostic'].indexOf('residual-morans-i');
    const moran = diagnostics.columns['value'][row] as number;
    const pValue = diagnostics.columns['p_value'][row] as number;
    const permutations = result.parameters['permutations'] as number;
    expect(permutations).toBeLessThanOrEqual(999);
    const second = report.outputs.find(output => output.tableId === 'diagnostics')!;
    expect(second.usedTemplate).toBe('global-statistic');
    expect(second.sentence).toContain(`The residual Moran's I is ${formatMeasured(moran)}`);
    expect(second.sentence).toContain(
      `pseudo p-value ${pValue.toFixed(3)} from ${group(permutations)} permutations`
    );
    expect(getCautionIds(report)).toEqual(['pseudo-p-values']);
  });

  it('kriging with diagnostics: the range of the predictions; the leave-one-out error fits no template', async () => {
    const result = await runCuratedAnalysis('kriging-with-diagnostics');
    const values = (getTable(result, 'kriging-cells').columns['value'] as (number | null)[]).filter(
      (value): value is number => value !== null && Number.isFinite(value)
    );
    const report = describeKeplerCatalogResult(
      result,
      getBinding('curated-kriging-with-diagnostics'),
      getDescriptor('curated-kriging-with-diagnostics')
    );
    expect(report.sentence).toBe(
      `Value ranges from ${formatMeasured(Math.min(...values))} to ${formatMeasured(Math.max(...values))} across ${group(values.length)} values.`
    );
    // The error is in the summary but the value-range template has no place for it.
    expect(typeof result.summary['rootMeanSquareError']).toBe('number');
    expect(report.outputs.map(output => output.sentence).join(' ')).not.toMatch(/error|RMSE/i);
  });

  it('contiguity correlogram: Moran I at order 1 with its permutation p-value', async () => {
    const result = await runCuratedAnalysis('contiguity-correlogram');
    const table = getTable(result, 'correlogram');
    const row = table.columns['order'].indexOf(1);
    const moran = table.columns['morans_i'][row] as number;
    const pValue = table.columns['pseudo_p_value'][row] as number;
    const permutations = result.parameters['permutations'] as number;
    expect(permutations).toBeLessThanOrEqual(999);
    expect(moran).toBe(result.summary['orderOneMoransI']);
    const report = describeKeplerCatalogResult(
      result,
      getBinding('curated-contiguity-correlogram'),
      getDescriptor('curated-contiguity-correlogram')
    );
    expect(report.sentence).toContain(`Moran's I at neighbour order 1 is ${formatMeasured(moran)}`);
    expect(report.sentence).toContain(
      `pseudo p-value ${pValue.toFixed(3)} from ${group(permutations)} permutations`
    );
    expect(getCautionIds(report)).toEqual(['pseudo-p-values']);
  });
});

// ---------------------------------------------------------------------------------------------
// Islands, attainability and cautions read from the run
// ---------------------------------------------------------------------------------------------

function makeIsland(x: number, value: number) {
  return {
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [x, 100],
          [x + 1, 100],
          [x + 1, 101],
          [x, 101],
          [x, 100]
        ]
      ]
    },
    properties: {value}
  };
}

/** A 12 by 12 lattice with a hot corner and three single squares that touch nothing. */
function makeLatticeWithIslands(): CatalogFeatureCollection {
  const lattice = makeLattice(12);
  return {
    type: 'FeatureCollection',
    features: [
      ...lattice.features.map((feature, index) => ({
        ...feature,
        properties: {
          value: (index % 12 < 4 && Math.floor(index / 12) < 4 ? 40 : 2) + (index % 5)
        }
      })),
      makeIsland(100, 60),
      makeIsland(102, -30),
      makeIsland(104, 3)
    ]
  };
}

describe('result report: rows with no neighbour', () => {
  it('gi-star: islands are left out of the hot, cold and tested counts and counted as a fact', async () => {
    const binding = getBinding('hotspot-gi-star');
    const result = await runCatalogOperationAsync(
      binding,
      {valueProperty: 'value', neighborhood: 'queen', coordinateSystem: 'planar'},
      {features: makeLatticeWithIslands()},
      {engine: 'cpu'}
    );
    const table = getTable(result, 'hotspots');
    const classes = table.columns['classification'] as string[];
    const neighbors = table.columns['neighbor_count'] as number[];
    const islandRows = neighbors.filter(count => count === 0).length;
    expect(islandRows).toBe(3);
    // The producer marks all three islands not-tested, with null inference.
    expect(classes.filter((_, row) => neighbors[row] === 0)).toEqual([
      'not-tested',
      'not-tested',
      'not-tested'
    ]);
    expect(table.columns['z_score'].filter((_, row) => neighbors[row] === 0)).toEqual([
      null,
      null,
      null
    ]);
    expect(table.columns['p_value'].filter((_, row) => neighbors[row] === 0)).toEqual([
      null,
      null,
      null
    ]);
    const hotWithNeighbors = classes.filter(
      (cls, row) => neighbors[row] > 0 && cls.startsWith('hot')
    ).length;
    const coldWithNeighbors = classes.filter(
      (cls, row) => neighbors[row] > 0 && cls.startsWith('cold')
    ).length;
    const report = describeKeplerCatalogResult(result, binding, getDescriptor('hotspot-gi-star'));
    expect(report.sentence).toMatch(
      new RegExp(
        `^${hotWithNeighbors} of ${table.rowCount - islandRows} features are significant hot spots and ${coldWithNeighbors} are (significant )?cold spots at 90% confidence without correction for multiple testing\\.$`
      )
    );
    const caution = report.cautions.find(item => item.id === 'rows-without-neighbors')!;
    expect(caution).toBeDefined();
    // The producer itself excludes islands from the tested classes.
    expect(caution.text).toBe(
      getGPUAnalysisCautionText('rows-without-neighbors', {rowsExcluded: true}).text
    );
    // Without islands there is no caution and every row counts.
    const clean = await runCatalogOperationAsync(
      binding,
      {valueProperty: 'value', neighborhood: 'queen', coordinateSystem: 'planar'},
      {features: makeLattice(12)},
      {engine: 'cpu'}
    );
    const cleanReport = describeKeplerCatalogResult(
      clean,
      binding,
      getDescriptor('hotspot-gi-star')
    );
    expect(getCautionIds(cleanReport)).not.toContain('rows-without-neighbors');
    expect(cleanReport.sentence).toContain(' of 144 features ');
  });

  it('local Moran: islands are not tested, and the caution says the producer left them out', async () => {
    const binding = getBinding('local-moran-statistics');
    const result = await runCatalogOperationAsync(
      binding,
      {
        valueProperty: 'value',
        neighborhood: 'queen',
        coordinateSystem: 'planar',
        permutations: 99,
        multipleTesting: 'none'
      },
      {features: makeLatticeWithIslands()},
      {engine: 'cpu'}
    );
    const table = getTable(result, 'local-moran');
    const notTested = countWhere(table.columns['cluster'], value => value === 'not-tested');
    expect(notTested).toBe(3);
    const report = describeKeplerCatalogResult(
      result,
      binding,
      getDescriptor('local-moran-statistics')
    );
    expect(report.sentence).toContain(` of ${table.rowCount - notTested} features `);
    const caution = report.cautions.find(item => item.id === 'rows-without-neighbors')!;
    expect(caution).toBeDefined();
    expect(caution.text).toBe(
      getGPUAnalysisCautionText('rows-without-neighbors', {rowsExcluded: true}).text
    );
  });
});

describe('result report: significance that cannot be reached', () => {
  // Oracles (/usr/bin/python3): 144 / 0.05 - 1 = 2879; 1 / (1 - 0.95 ** (1 / 144)) - 1 = 2806.9,
  // so 2807; without correction or with Benjamini-Hochberg 1 / 0.05 - 1 = 19.
  it('uses the package to decide whether a flag is possible, with the numbers of the oracle', () => {
    // Oracles (/usr/bin/python3): 144 / 0.05 - 1 = 2879; 1 / (1 - 0.95 ** (1 / 144)) - 1 = 2806.9,
    // so 2807; without correction or with Benjamini-Hochberg 1 / 0.05 - 1 = 19.
    const check = (
      correction: 'none' | 'bonferroni' | 'holm' | 'sidak' | 'false-discovery-rate',
      testCount: number,
      permutationCount: number,
      significanceLevel = 0.05
    ) =>
      getGPUAnalysisSignificanceAttainability({
        correction,
        testCount,
        permutationCount,
        significanceLevel
      });
    expect(check('bonferroni', 144, 999)).toMatchObject({
      attainable: false,
      requiredPermutationCount: 2879
    });
    expect(check('sidak', 144, 999)).toMatchObject({
      attainable: false,
      requiredPermutationCount: 2807
    });
    expect(check('none', 144, 18)).toMatchObject({attainable: false, requiredPermutationCount: 19});
    expect(check('false-discovery-rate', 144, 19).attainable).toBe(true);
    expect(check('none', 1, 99, 0.001)).toMatchObject({
      attainable: false,
      requiredPermutationCount: 999
    });
  });

  it("local Moran with 9 permutations and nothing flagged: the sentence is the package's own, and says so", async () => {
    const binding = getBinding('local-moran-statistics');
    const features = makeLattice(8);
    const run = (permutations: number, significance: number) =>
      runCatalogOperationAsync(
        binding,
        {...LATTICE_PARAMETERS, permutations, significance, multipleTesting: 'none'},
        {features},
        {engine: 'cpu'}
      );
    const result = await run(9, 0.05);
    const table = getTable(result, 'local-moran');
    // With 9 permutations the smallest pseudo p-value is 0.1, above the level: nothing can be flagged.
    expect(
      countWhere(table.columns['cluster'], value =>
        ['HH', 'LL', 'HL', 'LH'].includes(value as string)
      )
    ).toBe(0);
    const report = describeKeplerCatalogResult(
      result,
      binding,
      getDescriptor('local-moran-statistics')
    );
    // The report passes the facts and the package words them: the sentences are equal.
    const expected = summarizeGPUAnalysisResult('local-cluster-count', {
      featureCount: 64,
      highHighCount: 0,
      lowLowCount: 0,
      highLowCount: 0,
      lowHighCount: 0,
      significanceLevel: 0.05,
      pValueKind: 'pseudo',
      correction: 'none',
      permutationCount: 9
    });
    expect(report.sentence).toBe(expected);
    expect(report.sentence).toContain('at least 19 permutations are needed');
    expect(getCautionIds(report)).toContain('significance-not-attainable');
    const reason = getGPUAnalysisSignificanceAttainability({
      correction: 'none',
      testCount: 64,
      permutationCount: 9,
      significanceLevel: 0.05
    }).reason;
    // The caution text states the same reason.
    expect(report.cautions.find(item => item.id === 'significance-not-attainable')!.text).toBe(
      getGPUAnalysisCautionText('significance-not-attainable', {
        correction: 'none',
        testCount: 64,
        permutationCount: 9,
        significanceLevel: 0.05
      }).text
    );
    expect(report.cautions.find(item => item.id === 'significance-not-attainable')!.text).toContain(
      reason.charAt(0).toUpperCase() + reason.slice(1)
    );
    // 99 permutations at the level 0.001 cannot reach it either.
    const strict = describeKeplerCatalogResult(
      await run(99, 0.001),
      binding,
      getDescriptor('local-moran-statistics')
    );
    expect(strict.sentence).toContain('at least 999 permutations are needed');
    // When the permutations suffice there is no such sentence or caution.
    const enough = describeKeplerCatalogResult(
      await run(99, 0.05),
      binding,
      getDescriptor('local-moran-statistics')
    );
    expect(enough.sentence).not.toContain('permutations are needed');
    expect(getCautionIds(enough)).not.toContain('significance-not-attainable');
  });

  it("a global permutation test with too few permutations says so too (knox), with the package's reason", () => {
    const binding = getBinding('knox-test');
    const make = (permutations: number) =>
      describeKeplerCatalogResult(
        makeResult({
          tables: [makeTable('knox', {ratio: [1.1], p_value: [0.3], joint_close_pairs: [4]})],
          parameters: {permutations},
          summary: {eventCount: 60}
        }),
        binding,
        getDescriptor('knox-test')
      );
    const reason = getGPUAnalysisSignificanceAttainability({
      correction: 'none',
      testCount: 1,
      permutationCount: 9,
      significanceLevel: 0.05
    }).reason;
    expect(make(9).sentence).toContain(`This is not a finding: ${reason}.`);
    expect(getCautionIds(make(9))).toContain('significance-not-attainable');
    expect(make(99).sentence).not.toContain('This is not a finding');
    // A single test is not "uncorrected testing of many locations".
    expect(getCautionIds(make(9))).not.toContain('multiple-testing-uncorrected');
  });

  it('the level reaches the uncorrected-testing caution text through the package', async () => {
    const binding = getBinding('local-moran-statistics');
    const make = async (significance: number) =>
      describeKeplerCatalogResult(
        await runCatalogOperationAsync(
          binding,
          {...LATTICE_PARAMETERS, permutations: 999, significance},
          {features: makeLattice(8)},
          {engine: 'cpu'}
        ),
        binding,
        getDescriptor('local-moran-statistics')
      ).cautions.find(item => item.id === 'multiple-testing-uncorrected')!;
    for (const level of [0.05, 0.01]) {
      expect((await make(level)).text).toBe(
        getGPUAnalysisCautionText('multiple-testing-uncorrected', {significanceLevel: level}).text
      );
    }
    expect((await make(0.01)).text).toContain('at the 0.01 level, about 1 in 100');
  });
});

describe('result report: cautions the run itself raises', () => {
  const cases: [string, string, string, Record<string, number>, Record<string, number>][] = [
    // bindingId, caution, summary key, summary that raises it, summary that does not
    [
      'curated-kriging-with-diagnostics',
      'poor-model-fit',
      'poorModelFit',
      {poorModelFit: 1},
      {poorModelFit: 0}
    ],
    [
      'curated-excess-risk',
      'small-denominators',
      'smallDenominatorAreaCount',
      {smallDenominatorAreaCount: 3},
      {smallDenominatorAreaCount: 0}
    ],
    [
      'curated-contiguity-correlogram',
      'row-order-dependent-inference',
      'tiedAnchorCount',
      {tiedAnchorCount: 2},
      {tiedAnchorCount: 0}
    ],
    [
      'curated-regression-with-diagnostics',
      'row-order-dependent-inference',
      'tiedAnchorCount',
      {tiedAnchorCount: 1},
      {tiedAnchorCount: 0}
    ],
    [
      'curated-rate-cluster-map',
      'significance-not-attainable',
      'significanceNotAttainable',
      {significanceNotAttainable: 1},
      {significanceNotAttainable: 0}
    ]
  ];

  for (const [bindingId, caution, key, raises, quiet] of cases) {
    it(`${bindingId}: ${caution} follows summary key ${key}`, () => {
      const descriptor = getDescriptor(bindingId);
      const table = makeTable(descriptor.outputs[0].tableId, {x: [1, 2, 3]});
      const make = (summary: Record<string, number>) =>
        describeKeplerCatalogResult(
          makeResult({tables: [table], summary}),
          getBinding(bindingId),
          descriptor
        );
      const raised = make(raises).cautions.find(item => item.id === caution);
      expect(raised).toBeDefined();
      expect(raised!.title.length).toBeGreaterThan(0);
      expect(raised!.text.length).toBeGreaterThan(20);
      expect(getCautionIds(make(quiet))).not.toContain(caution);
      expect(getCautionIds(make({}))).not.toContain(caution);
    });
  }

  it('the convention is the one the curated descriptors document, and every entry names a summary key the binding writes', async () => {
    // The author of the fused bindings exports `CURATED_RUNTIME_CAUTION_SUMMARY_KEYS`: a summary
    // key and `equals 1` or `is above 0`. The report reads exactly that; here each key is checked
    // against a real run of its binding on its curated sample.
    const analyses: Record<string, string> = {
      'curated-kriging-with-diagnostics': 'kriging-with-diagnostics',
      'curated-excess-risk': 'excess-risk',
      'curated-contiguity-correlogram': 'contiguity-correlogram',
      'curated-regression-with-diagnostics': 'regression-with-diagnostics',
      'curated-rate-cluster-map': 'rate-cluster-map'
    };
    for (const [bindingId, analysisId] of Object.entries(analyses)) {
      const result = await runCuratedAnalysis(analysisId);
      for (const entry of Object.values(CURATED_RUNTIME_CAUTION_SUMMARY_KEYS[bindingId])) {
        expect(typeof result.summary[entry.key], `${bindingId}.${entry.key}`).toBe('number');
      }
    }
  });
});

describe('result report: sentences for more bindings', () => {
  it('multiple-testing-adjustment: how many p-values are at or below the level after the adjustment', async () => {
    const binding = getBinding('multiple-testing-adjustment');
    const features = {
      type: 'FeatureCollection' as const,
      features: Array.from({length: 20}, (_, index) => ({
        geometry: {type: 'Point', coordinates: [index, 0]},
        properties: {p: index < 5 ? 0.0001 * (index + 1) : 0.02 * (index + 1)}
      }))
    };
    const result = await runCatalogOperationAsync(
      binding,
      {pValueProperty: 'p', method: 'holm', alpha: 0.05},
      {features},
      {engine: 'cpu'}
    );
    const rejected = countWhere(
      getTable(result, 'adjusted').columns['rejected'],
      value => value === true
    );
    expect(rejected).toBeGreaterThan(0);
    const report = describeKeplerCatalogResult(
      result,
      binding,
      getDescriptor('multiple-testing-adjustment')
    );
    expect(report.sentence).toBe(
      `${rejected} of 20 p-values are at or below 0.05 after Holm adjustment.`
    );
    expect(result.summary['rejectedCount']).toBe(rejected);
  });

  it('network K function: no significance test, so the range of the excess over the reference', () => {
    const binding = getBinding('network-k-function');
    const descriptor = getDescriptor('network-k-function');
    const report = describeKeplerCatalogResult(
      makeResult({
        tables: [makeTable('network-k', {radius: [100, 200, 400], excess: [0.5, 1.25, 3]})]
      }),
      binding,
      descriptor
    );
    expect(report.outputs[0].usedTemplate).toBe('value-range');
    expect(report.sentence).toBe('Excess ranges from 0.5 to 3 across 3 values.');
    // No p-value is invented, so no pseudo p-value caution.
    expect(getCautionIds(report)).toEqual([]);
  });

  it('point density grid: a count per cell only when the statistic is count', () => {
    const binding = getBinding('point-density-grid');
    const descriptor = getDescriptor('point-density-grid');
    const make = (statistic: string) =>
      describeKeplerCatalogResult(
        makeResult({
          tables: [makeTable('density-cells', {value: [1, 0, 4, 2]})],
          parameters: {statistic}
        }),
        binding,
        descriptor
      );
    expect(make('count').outputs[0].usedTemplate).toBe('count-per-area');
    expect(make('count').sentence).toContain('7 points across 4 cells');
    expect(make('mean').outputs[0].usedTemplate).not.toBe('count-per-area');
  });
});

describe('result report: facts passed to the analysis package', () => {
  it("global Moran: sidedness, level and the meaning of the sign are passed, and the sentence is the package's", async () => {
    const areas = createKeplerCuratedAnalysisSample('city-district-rates')['areas'];
    const binding = getBinding('global-spatial-statistics');
    const result = await runCatalogOperationAsync(
      binding,
      {valueProperty: 'rate', neighborhood: 'queen', coordinateSystem: 'EPSG:4326'},
      {features: areas},
      {engine: 'cpu'}
    );
    const table = getTable(result, 'global');
    const row = table.columns['statistic'].indexOf('morans-i');
    const value = table.columns['value'][row] as number;
    const pValue = table.columns['p_value'][row] as number;
    expect(value).toBeGreaterThan(0);
    const report = describeKeplerCatalogResult(
      result,
      binding,
      getDescriptor('global-spatial-statistics')
    );
    expect(report.sentence).toBe(
      summarizeGPUAnalysisResult('global-statistic', {
        statisticName: "Moran's I",
        value,
        pValue,
        permutationCount: 0,
        significanceLevel: 0.05,
        sidedness: 'two-sided',
        positiveMeaning: 'similar values near each other',
        negativeMeaning: 'dissimilar values near each other'
      })
    );
    expect(report.sentence).toContain('(a positive value means similar values near each other)');
    expect(report.sentence).toContain('in either direction');
  });

  it('knox: the null is no space-time interaction and the p-value is one-sided', () => {
    const report = describeKeplerCatalogResult(
      makeResult({
        tables: [makeTable('knox', {ratio: [1.8], p_value: [0.012], joint_close_pairs: [18]})],
        parameters: {permutations: 99},
        summary: {eventCount: 60}
      }),
      getBinding('knox-test'),
      getDescriptor('knox-test')
    );
    expect(report.sentence).toBe(
      summarizeGPUAnalysisResult('global-statistic', {
        statisticName: 'The ratio of observed to expected pairs that are close in space and time',
        value: 1.8,
        pValue: 0.012,
        permutationCount: 99,
        significanceLevel: 0.05,
        sidedness: 'greater',
        nullDescription: 'no space-time interaction'
      })
    );
    expect(report.sentence).toContain('no space-time interaction');
  });

  it('the correlogram uses its own significance level for the sentence', async () => {
    const result = await runCuratedAnalysis('contiguity-correlogram', {significance: 0.01});
    const report = describeKeplerCatalogResult(
      result,
      getBinding('curated-contiguity-correlogram'),
      getDescriptor('curated-contiguity-correlogram')
    );
    expect(report.sentence).toMatch(/at the 0\.01 level\.$/);
  });

  it('small denominators: the area count and the threshold of the summary reach the caution text', () => {
    const descriptor = getDescriptor('curated-excess-risk');
    const report = describeKeplerCatalogResult(
      makeResult({
        tables: [makeTable('excess-risk', {x: [1]})],
        summary: {smallDenominatorAreaCount: 3, smallDenominatorThreshold: 5}
      }),
      getBinding('curated-excess-risk'),
      descriptor
    );
    expect(report.cautions.find(item => item.id === 'small-denominators')!.text).toBe(
      getGPUAnalysisCautionText('small-denominators', {
        smallDenominatorAreaCount: 3,
        expectedCountLimit: 5
      }).text
    );
    expect(report.cautions.find(item => item.id === 'small-denominators')!.text).toContain(
      '3 areas have an expected count below 5.'
    );
  });

  it('poor model fit: the reasons of the kriging summary reach the caution text', async () => {
    const binding = getBinding('curated-kriging-with-diagnostics');
    const descriptor = getDescriptor('curated-kriging-with-diagnostics');
    const table = makeTable('kriging-cells', {value: [1, 2, 3]});
    const report = describeKeplerCatalogResult(
      makeResult({
        tables: [table],
        summary: {
          poorModelFit: 1,
          poorModelFitReasons: 'the nugget is above 75 percent of the sill'
        }
      }),
      binding,
      descriptor
    );
    expect(report.cautions.find(item => item.id === 'poor-model-fit')!.text).toContain(
      'Reason given: the nugget is above 75 percent of the sill.'
    );
    // A real run: the caution follows the binding's own key, and carries its reasons when raised.
    const result = await runCuratedAnalysis('kriging-with-diagnostics');
    const real = describeKeplerCatalogResult(result, binding, descriptor);
    const caution = real.cautions.find(item => item.id === 'poor-model-fit');
    expect(Boolean(caution)).toBe(result.summary['poorModelFit'] === 1);
    if (caution) expect(caution.text).toContain(String(result.summary['poorModelFitReasons']));
  });

  it('row-order dependence and an unreachable significance take the package texts', () => {
    const report = describeKeplerCatalogResult(
      makeResult({
        tables: [makeTable('correlogram', {order: [1], morans_i: [0.3], pseudo_p_value: [0.2]})],
        parameters: {permutations: 999, significance: 0.05},
        summary: {tiedAnchorCount: 2, areaCount: 40}
      }),
      getBinding('curated-contiguity-correlogram'),
      getDescriptor('curated-contiguity-correlogram')
    );
    expect(report.cautions.find(item => item.id === 'row-order-dependent-inference')!.text).toBe(
      getGPUAnalysisCautionText('row-order-dependent-inference').text
    );
  });
});

describe('review regressions: reports use the facts of the run', () => {
  it('retains all statistical output sentences in the text report', async () => {
    const result = await runCuratedAnalysis('regression-with-diagnostics', {permutations: 99});
    const report = describeKeplerCatalogResult(
      result,
      getBinding(result.operationId),
      getDescriptor(result.operationId)
    );
    const diagnostic = report.outputs.find(output => output.usedTemplate === 'global-statistic');
    expect(diagnostic?.sentence).toContain('Moran');
    const text = formatKeplerResultReport(report);
    for (const output of report.outputs) expect(text).toContain(output.sentence);
    expect(text.split(report.sentence)).toHaveLength(2);
  });

  it('shares range precision with real kriging legends and adjacent float legends', async () => {
    const points: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: Array.from({length: 20}, (_, index) => ({
        geometry: {
          type: 'Point',
          coordinates: [
            1000 + (index % 10) * 12 + Math.sin(index),
            2000 + Math.floor(index / 10) * 15 + Math.cos(index)
          ]
        },
        properties: {value: 100000 + index / 500}
      }))
    };
    const binding = getBinding('curated-kriging-with-diagnostics');
    const descriptor = getDescriptor(binding.id);
    const result = await runCatalogOperationAsync(
      binding,
      {valueProperty: 'value', coordinateSystem: 'planar', cellSize: 30},
      {points},
      {engine: 'cpu'}
    );
    const report = describeKeplerCatalogResult(result, binding, descriptor);
    const entries = getKeplerBindingResultPresentations(result, descriptor)[0].legend;
    const firstText = entries[0].label.replace('above ', '').split(' to ')[0];
    const lastText = entries[entries.length - 1].label.split(' to ').at(-1)!;
    expect(firstText).not.toBe(lastText);
    expect(report.sentence).toContain(`from ${firstText} to ${lastText}`);
    const table = makeTable('network-k', {excess: [1, 1 + Number.EPSILON, 1 + 2 * Number.EPSILON]});
    const adjacent = describeKeplerCatalogResult(
      makeResult({tables: [table]}),
      getBinding('network-k-function'),
      getDescriptor('network-k-function')
    );
    expect(adjacent.sentence).toContain('from 1 to 1.0000000000000004');
  });

  it('uses the lowest counted Gi-star confidence for cautions and excludes the same islands in the legend', async () => {
    const binding = getBinding('hotspot-gi-star');
    const descriptor = getDescriptor(binding.id);
    const result = await runCatalogOperationAsync(
      binding,
      {valueProperty: 'value', neighborhood: 'queen', coordinateSystem: 'planar'},
      {features: makeLatticeWithIslands()},
      {engine: 'cpu'}
    );
    const report = describeKeplerCatalogResult(result, binding, descriptor);
    expect(report.sentence).toContain('90% confidence');
    expect(
      report.cautions.find(caution => caution.id === 'multiple-testing-uncorrected')?.text
    ).toContain('0.1 level, about 10 in 100');
    const table = getTable(result, 'hotspots');
    const plan = planKeplerResultPresentation(table, descriptor.outputs[0])!;
    const hot = plan.legend
      .filter(entry => ['strong-high', 'high', 'weak-high'].includes(entry.tone))
      .reduce((total, entry) => total + entry.count, 0);
    const eligible = (table.columns['neighbor_count'] as number[]).filter(
      value => value > 0
    ).length;
    expect(report.sentence).toContain(`${hot} of ${eligible}`);
    expect(plan.excludedRows).toHaveLength(3);
    const legend = getKeplerBindingResultPresentations(result, descriptor)[0].legend;
    expect(legend.find(entry => entry.label === 'Excluded: no neighbours')?.count).toBe(3);
    expect(legend.reduce((total, entry) => total + (entry.count ?? 0), 0)).toBe(table.rowCount);
  });

  it('does not describe tests that were not performed, or invent a correction for nine permutations', async () => {
    const binding = getBinding('local-moran-statistics');
    for (const permutations of [0, 9]) {
      const result = await runCatalogOperationAsync(
        binding,
        {valueProperty: 'value', neighborhood: 'queen', coordinateSystem: 'planar', permutations},
        {features: makeLattice(8)},
        {engine: 'cpu'}
      );
      const report = describeKeplerCatalogResult(result, binding, getDescriptor(binding.id));
      if (permutations === 0) {
        expect(getCautionIds(report)).not.toContain('multiple-testing-uncorrected');
        expect(getCautionIds(report)).not.toContain('pseudo-p-values');
        expect(formatKeplerResultReport(report)).not.toContain('Every location was tested');
      }
      const caution = report.cautions.find(
        candidate => candidate.id === 'significance-not-attainable'
      );
      expect(caution).toBeDefined();
      expect(caution?.text).not.toContain('correction over many tests');
    }
  });

  it('resolves measurement requirements by actual geometry, preserving point amounts', () => {
    const binding = getBinding('local-moran-statistics');
    const descriptor = getDescriptor(binding.id);
    for (const catalogGeometry of ['point', 'polygon'] as const) {
      const report = describeKeplerCatalogResult(makeResult({}), binding, descriptor, {
        fieldBindings: {valueProperty: 'value'},
        profiles: {
          features: {
            kind: 'attributes',
            catalogGeometry,
            columns: [{name: 'value', measurement: 'amount'}]
          }
        }
      });
      expect(getCautionIds(report).includes('counts-where-rates-expected')).toBe(
        catalogGeometry === 'polygon'
      );
    }
  });

  it('filters inactive fallback cautions and honours explicit coordinate provenance', () => {
    const binding = getBinding('hotspot-gi-star');
    const descriptor = getDescriptor(binding.id);
    const result = makeResult({
      parameters: {neighborhood: 'knn', coordinateSystem: 'planar', distanceBand: 120000}
    });
    const report = describeKeplerCatalogResult(result, binding, descriptor, {
      profiles: {features: {kind: 'polygons', coordinates: {model: 'geographic', source: 'guess'}}},
      resolutions: [
        {parameter: 'distanceBand', source: 'fallback', active: false},
        {parameter: 'coordinateSystem', source: 'override'}
      ]
    });
    expect(getCautionIds(report)).not.toContain('default-from-fallback');
    expect(getCautionIds(report)).not.toContain('guessed-coordinates');
    const coordinateRecord = {parameter: 'coordinateSystem', source: 'override' as const};
    const mismatch = describeKeplerCatalogResult(
      {...result, parameters: {...result.parameters, neighborhood: 'distance-band'}},
      binding,
      descriptor,
      {
        resolutions: [
          coordinateRecord,
          {
            parameter: 'distanceBand',
            source: 'data',
            resolution: makeResolution({
              unit: 'metre',
              explanation: 'Distance 111 km, so all features have neighbours'
            })
          }
        ]
      }
    );
    expect(mismatch.chosen.join(' ')).toContain('Distance band 120,000 coordinate units');
    expect(mismatch.chosen.join(' ')).toContain('different coordinate model');
    expect(mismatch.chosen.join(' ')).not.toContain('111 km');
  });

  it('drops occupancy claims after the cell size is clamped', () => {
    const binding = getBinding('recipe-hot-spot-lattice');
    const report = describeKeplerCatalogResult(
      makeResult({parameters: {cellSize: 13000, coordinateSystem: 'planar'}}),
      binding,
      getDescriptor(binding.id),
      {
        resolutions: [
          {
            parameter: 'cellSize',
            source: 'data',
            resolution: makeResolution({
              value: 630,
              explanation: 'A cell of this size holds about 4 points on average'
            }),
            clamped: {from: 630, to: 13000, reason: 'Grid capacity limit'}
          }
        ]
      }
    );
    expect(report.chosen[0]).toContain('13,000 coordinate units');
    expect(report.chosen[0]).toContain('Grid capacity limit');
    expect(report.chosen[0]).not.toContain('holds about 4');
  });

  it('calls polygon outputs areas and describes overlapping-zone memberships honestly', async () => {
    const polygons: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [0, 1].map(offset => ({
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [offset, 0],
              [offset + 2, 0],
              [offset + 2, 2],
              [offset, 2],
              [offset, 0]
            ]
          ]
        },
        properties: {}
      }))
    };
    const points: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        [0.5, 1],
        [1.5, 1]
      ].map(coordinates => ({geometry: {type: 'Point', coordinates}, properties: {}}))
    };
    for (const bindingId of ['raster-join-points-in-polygons', 'zonal-point-statistics']) {
      const binding = getBinding(bindingId);
      const result = await runCatalogOperationAsync(
        binding,
        {},
        {polygons, points},
        {engine: 'cpu'}
      );
      const report = describeKeplerCatalogResult(result, binding, getDescriptor(bindingId));
      if (bindingId === 'zonal-point-statistics') {
        expect(report.sentence).toContain('3 point memberships across 2 zones');
        expect(report.sentence).toContain('once to each zone');
        expect(report.sentence).not.toContain('3 points');
      } else {
        expect(report.sentence).toContain('areas');
        expect(report.sentence).not.toContain('cell');
      }
    }
  });
});

it.each([
  ['none', 'none', [16, 1, 1, 5]],
  ['benjamini-hochberg', 'false-discovery-rate', [12, 0, 0, 0]],
  ['bonferroni', 'bonferroni', [0, 0, 0, 0]]
] as const)('Local Moran reports the selected %s correction from a real run', async (multipleTesting, correction, expectedCounts) => {
  const result = await runCuratedAnalysis('cluster-and-outlier-map', {multipleTesting});
  const table = getTable(result, 'local-moran');
  const counts = ['HH', 'LL', 'HL', 'LH'].map(cluster =>
    countWhere(table.columns['cluster'], value => value === cluster)
  );
  expect(counts).toEqual(expectedCounts);
  const report = describeKeplerCatalogResult(
    result,
    getBinding('local-moran-statistics'),
    getDescriptor('local-moran-statistics')
  );
  expect(report.sentence).toBe(
    summarizeGPUAnalysisResult('local-cluster-count', {
      featureCount: 144,
      highHighCount: counts[0],
      lowLowCount: counts[1],
      highLowCount: counts[2],
      lowHighCount: counts[3],
      significanceLevel: 0.05,
      pValueKind: 'pseudo',
      correction,
      permutationCount: 999
    })
  );
});
