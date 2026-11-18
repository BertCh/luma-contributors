// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_ANALYSIS_CLASS_SCHEMES,
  getGPUAnalysisResultClasses
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUAnalysisResultClass} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {describe, expect, it} from 'vitest';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';
import {getCatalogBinding, runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {getCatalogFeatureInputs} from '../../src/analysis/catalog-data-input';
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
import type {
  KeplerOperationDescriptor,
  KeplerOperationOutputDescriptor
} from '../../src/descriptors/operation-descriptor-types';
import {
  getKeplerBindingResultPresentations,
  encodeKeplerResultPresentation,
  getKeplerColorContrastRatio,
  getKeplerColorDifference2000,
  getKeplerColorDistance,
  getKeplerResultRampColors,
  getKeplerResultToneColor,
  KEPLER_CATEGORICAL_PALETTE,
  KEPLER_DARK_MAP_BACKGROUND,
  KEPLER_DEFAULT_CLASS_COUNT,
  KEPLER_MINIMUM_LEGEND_COLOR_DIFFERENCE_2000,
  KEPLER_MINIMUM_LOWEST_CLASS_CONTRAST_RATIO,
  KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000,
  KEPLER_MISSING_VALUE_COLOR,
  KEPLER_OTHER_CATEGORY_COLOR,
  KEPLER_RECOMMENDED_OUTLINE_COLOR,
  KEPLER_DIVERGING_RAMP,
  KEPLER_LIGHT_MAP_BACKGROUND,
  KEPLER_MAXIMUM_CATEGORY_COUNT,
  KEPLER_MINIMUM_ENCODING_SHARE,
  KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE,
  KEPLER_MINIMUM_MAP_CONTRAST_RATIO,
  KEPLER_MINIMUM_RAMP_COLOR_DISTANCE,
  KEPLER_SEQUENTIAL_RAMP,
  planKeplerResultPresentation,
  simulateKeplerColorVision,
  toKeplerWorkbenchDisplayStyle
} from '../../src/presentation/result-presentation';
import type {
  KeplerColorVision,
  KeplerResultColor
} from '../../src/presentation/result-presentation';

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

type ColumnValue = string | number | boolean | null;

it('publishes every nonzero uint32 validity flag with the same valid class as its legend', () => {
  const table = makeTable('table', {value: [0, 1, 2, 0xffffffff]});
  const presentation = planKeplerResultPresentation(
    table,
    makeOutput({column: 'value', scale: 'class-scheme', classScheme: 'validity-flag'})
  )!;
  const legend = presentation.legend.map(entry => ({
    label: entry.label,
    color: entry.color,
    count: entry.count
  }));
  const encoded = encodeKeplerResultPresentation(table, {
    tableId: table.id,
    presentation,
    display: {},
    legend
  })!;
  expect(encoded.values).toEqual(['Excluded or invalid', 'Valid', 'Valid', 'Valid']);
  expect(legend.map(entry => [entry.label, entry.count])).toEqual([
    ['Valid', 3],
    ['Excluded or invalid', 1]
  ]);
  const significant = planKeplerResultPresentation(
    table,
    makeOutput({column: 'value', scale: 'class-scheme', classScheme: 'significance-flag'})
  )!;
  expect(significant.classScheme).not.toBe('validity-flag');
  const significantLegend = significant.legend.map(entry => ({
    label: entry.label,
    color: entry.color,
    count: entry.count
  }));
  const significantEncoding = encodeKeplerResultPresentation(table, {
    tableId: table.id,
    presentation: significant,
    display: {},
    legend: significantLegend
  })!;
  expect(new Set(significantEncoding.values).size).toBe(4);
  expect(significantLegend.map(entry => entry.count)).toEqual([1, 1, 1, 1]);
});

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

function makeOutput(
  presentation: KeplerOperationOutputDescriptor['presentation']
): KeplerOperationOutputDescriptor {
  return {tableId: 'table', kind: 'polygons', presentation};
}

const ALL_DESCRIPTORS: readonly KeplerOperationDescriptor[] = KEPLER_OPERATION_DESCRIPTORS;

/**
 * The descriptor of a binding with the class encoding of its class-scheme outputs filled in when
 * the descriptor does not name one yet. The authors of the descriptor files are adding the field;
 * until then the encoding is the one the binding writes.
 */
function getDescriptor(
  bindingId: string,
  encoding: 'contributor-integer' | 'catalog-text' | 'catalog-signed-label'
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

/** Runs a curated analysis on its sample through the catalog's entry point on the CPU engine. */
async function runCuratedAnalysis(
  analysisId: string,
  overrides: Record<string, CatalogParameterValue> = {}
): Promise<CatalogRunResult> {
  const analysis = getKeplerCuratedAnalysis(analysisId);
  if (!analysis || analysis.execution.status !== 'available' || !analysis.sampleId) {
    throw new Error(`runCuratedAnalysis: ${analysisId} is not an available analysis with a sample`);
  }
  const execution = analysis.execution;
  const sample = getCatalogFeatureInputs(createKeplerCuratedAnalysisSample(analysis.sampleId));
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

/** Rows per value of a column, counted without the code under test. */
function countValues(values: readonly ColumnValue[]): Map<ColumnValue, number> {
  const counts = new Map<ColumnValue, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return counts;
}

function sum(values: Iterable<number>): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

// ---------------------------------------------------------------------------------------------
// Real results
// ---------------------------------------------------------------------------------------------

describe('result presentation on real catalog results', () => {
  it('hot spot map: seven confidence classes whose counts sum to the rows', async () => {
    const result = await runCuratedAnalysis('hot-spot-map');
    const table = getTable(result, 'recipe-hot-spots');
    const descriptor = getDescriptor('recipe-hot-spot-lattice', 'contributor-integer');
    const [presentation] = getKeplerBindingResultPresentations(result, descriptor);
    const plan = presentation.presentation!;

    expect(plan.classScheme).toBe('hot-spot-confidence-bins');
    expect(plan.encodingShare).toBe(1);
    expect(plan.legend.map(entry => entry.label)).toEqual([
      'Hot spot, 99% confidence',
      'Hot spot, 95% confidence',
      'Hot spot, 90% confidence',
      'Not significant',
      'Cold spot, 90% confidence',
      'Cold spot, 95% confidence',
      'Cold spot, 99% confidence'
    ]);
    // Counted straight from the column.
    const bins = table.columns['bin'];
    const expected = countValues(bins);
    expect(plan.legend.map(entry => entry.count)).toEqual(
      [3, 2, 1, 0, -1, -2, -3].map(bin => expected.get(bin) ?? 0)
    );
    expect(sum(plan.legend.map(entry => entry.count))).toBe(table.rowCount);
    // The planted hot block shows: a real result with hot cells and no cold ones.
    const hotCount = bins.filter(bin => (bin as number) > 0).length;
    expect(hotCount).toBeGreaterThan(0);
    expect(plan.legend.slice(0, 3).reduce((total, entry) => total + entry.count, 0)).toBe(hotCount);
    expect(presentation.display).toEqual({
      colorColumn: 'bin',
      colorScale: 'ordinal',
      palette: 'diverging'
    });
    expect(presentation.legend[0]).toEqual({
      label: 'Hot spot, 99% confidence',
      color: getKeplerResultToneColor('strong-high'),
      count: expected.get(3) ?? 0
    });
  });

  it('cluster and outlier map: counts HH, LL, HL, LH and not significant from the cluster column', async () => {
    const result = await runCuratedAnalysis('cluster-and-outlier-map');
    const table = getTable(result, 'local-moran');
    const descriptor = getDescriptor('local-moran-statistics', 'catalog-text');
    const plan = getKeplerBindingResultPresentations(result, descriptor)[0].presentation!;
    const expected = countValues(table.columns['cluster']);

    expect(plan.legend.map(entry => entry.value)).toEqual([
      'HH',
      'HL',
      'LH',
      'LL',
      'not-significant',
      'not-tested'
    ]);
    expect(plan.legend.map(entry => entry.count)).toEqual(
      ['HH', 'HL', 'LH', 'LL', 'not-significant', 'not-tested'].map(
        value => expected.get(value) ?? 0
      )
    );
    expect(sum(plan.legend.map(entry => entry.count))).toBe(table.rowCount);
    expect(plan.unrecognized).toBe(0);
    // Distinct hues for the two outlier tones.
    const colors = plan.legend.map(entry => entry.color.join(','));
    expect(new Set(colors).size).toBe(colors.length);
  });

  it('clusters and footprints: the clusters, the noise and the cluster sizes', async () => {
    const result = await runCuratedAnalysis('clusters-and-footprints');
    const table = getTable(result, 'points');
    const descriptor = getDescriptor('curated-clusters-and-footprints', 'catalog-signed-label');
    const [pointsPresentation, clustersPresentation] = getKeplerBindingResultPresentations(
      result,
      descriptor
    );
    const plan = pointsPresentation.presentation!;
    const expected = countValues(table.columns['cluster']);

    // The labels of the clusters, counted from the column; the run decides how many there are.
    const labels = [...new Set(table.columns['cluster'] as number[])]
      .filter(label => label >= 0)
      .sort((first, second) => first - second);
    expect(labels.length).toBeGreaterThanOrEqual(2);
    expect(labels.length).toBeLessThanOrEqual(KEPLER_MAXIMUM_CATEGORY_COUNT);
    expect(plan.membership).toEqual({
      clusterCount: labels.length,
      noiseCount: expected.get(-1),
      largestClusterCount: Math.max(...labels.map(label => expected.get(label)!))
    });
    expect(plan.legend.map(entry => entry.label)).toEqual([
      'Noise',
      ...labels.map(label => `Cluster ${label}`)
    ]);
    expect(plan.legend.map(entry => entry.count)).toEqual([
      expected.get(-1),
      ...labels.map(label => expected.get(label))
    ]);
    expect(sum(plan.legend.map(entry => entry.count))).toBe(table.rowCount);
    // The second table maps the cluster sizes on a sequential scale.
    expect(clustersPresentation.presentation?.scale).toBe('sequential');
    expect(clustersPresentation.presentation?.breaks).toEqual(
      [...clustersPresentation.presentation!.breaks].sort((a, b) => a - b)
    );
  });

  it('density surface: a sequential scale with strictly increasing breaks', async () => {
    const result = await runCuratedAnalysis('density-surface');
    const table = getTable(result, 'density-cells');
    const descriptor = ALL_DESCRIPTORS.find(item => item.bindingId === 'kernel-density-grid')!;
    const plan = planKeplerResultPresentation(table, descriptor.outputs[0])!;
    const values = (table.columns['density'] as number[]).slice().sort((a, b) => a - b);

    expect(plan.scale).toBe('sequential');
    expect(plan.palette).toBe('sequential');
    expect(plan.breaks.length).toBeGreaterThanOrEqual(1);
    expect(plan.breaks.length).toBeLessThanOrEqual(KEPLER_DEFAULT_CLASS_COUNT - 1);
    for (let index = 1; index < plan.breaks.length; index++) {
      expect(plan.breaks[index]).toBeGreaterThan(plan.breaks[index - 1]);
    }
    expect(plan.domain).toEqual([values[0], values[values.length - 1]]);
    expect(sum(plan.ranges.map(range => range.count))).toBe(values.length);
    for (const range of plan.ranges) expect(range.count).toBeGreaterThan(0);
    expect(plan.classCountReason).toMatch(/Five classes/);
  });

  it('IDW surface: a sequential plan over the value column with the domain of the grid', async () => {
    const result = await runCuratedAnalysis('idw-surface');
    const table = getTable(result, 'idw-cells');
    const descriptor = ALL_DESCRIPTORS.find(item => item.bindingId === 'idw-grid')!;
    const plan = planKeplerResultPresentation(table, descriptor.outputs[0])!;
    const values = table.columns['value'] as number[];
    expect(plan.column).toBe('value');
    expect(plan.domain).toEqual([Math.min(...values), Math.max(...values)]);
    expect(plan.missingCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------------------------
// Numeric scales
// ---------------------------------------------------------------------------------------------

describe('numeric scales', () => {
  // Oracle: /usr/bin/python3 -c "import numpy as np; v=[3,7,7,12,15,18,21,21,25,30,34,41,41,47,52,58,63,70,77,85,91,99,104,110,118,125,133,142,150,161,175,190]; print(np.quantile(v,[.2,.4,.6,.8],method='inverted_cdf'))"
  // prints [ 21  41  85 125]. The data has ties at 7, 21 and 41.
  const QUANTILE_VALUES = [
    3, 7, 7, 12, 15, 18, 21, 21, 25, 30, 34, 41, 41, 47, 52, 58, 63, 70, 77, 85, 91, 99, 104, 110,
    118, 125, 133, 142, 150, 161, 175, 190
  ];

  it('breaks by quantile equal numpy inverted_cdf quantiles, with ties kept whole', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: QUANTILE_VALUES}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.breaks).toEqual([21, 41, 85, 125]);
    expect(plan.classCount).toBe(5);
    expect(plan.ranges.map(range => range.count)).toEqual([8, 5, 7, 6, 6]);
    expect(sum(plan.ranges.map(range => range.count))).toBe(QUANTILE_VALUES.length);
    expect(plan.ranges.map(range => range.label)).toEqual([
      '3 to 21',
      'above 21 to 41',
      'above 41 to 85',
      'above 85 to 125',
      'above 125 to 190'
    ]);
  });

  it('a column with heavy ties gets fewer classes and no empty class', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    for (let index = 1; index < plan.breaks.length; index++) {
      expect(plan.breaks[index]).toBeGreaterThan(plan.breaks[index - 1]);
    }
    expect(plan.classCount).toBe(plan.breaks.length + 1);
    expect(plan.classCount).toBeLessThan(KEPLER_DEFAULT_CLASS_COUNT);
    expect(plan.ranges.every(range => range.count > 0)).toBe(true);
    expect(plan.ranges[0].count).toBe(12);
    expect(plan.notes.join(' ')).toMatch(/Ties|distinct values/);
  });

  it('a column with three distinct values gets three classes', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [1, 2, 3, 1, 2, 3]}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.breaks).toEqual([1, 2]);
    expect(plan.classCount).toBe(3);
    expect(plan.notes[0]).toBe('Column "value" has 3 distinct values, so each gets its own class.');
  });

  it('a constant column has one class and no breaks', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [7, 7, 7, 7]}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.breaks).toEqual([]);
    expect(plan.classCount).toBe(1);
    expect(plan.domain).toEqual([7, 7]);
    expect(plan.ranges).toHaveLength(1);
    expect(plan.ranges[0].count).toBe(4);
    expect(plan.notes[0]).toBe('Every finite value in column "value" is 7, so there is one class.');
  });

  it('one row is one class', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [42]}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.classCount).toBe(1);
    expect(plan.domain).toEqual([42, 42]);
  });

  it('counts null, NaN and infinite values as missing and leaves them out of the domain', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [1, null, 2, Number.NaN, 3, Infinity, -Infinity, 4]}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.missingCount).toBe(4);
    expect(plan.domain).toEqual([1, 4]);
    expect(sum(plan.ranges.map(range => range.count))).toBe(4);
    const rows = getKeplerBindingResultPresentations(
      {
        operationId: 'x',
        engine: 'cpu-reference',
        dispatchedContributors: [],
        parameters: {},
        tables: [makeTable('table', {value: [1, null, 2, Number.NaN, 3, Infinity, -Infinity, 4]})],
        diagnostics: [],
        summary: {}
      },
      {
        bindingId: 'x',
        questions: ['prepare-data'],
        action: 'x',
        prominence: 3,
        inputs: [],
        parameters: [],
        requirements: [],
        outputs: [makeOutput({column: 'value', scale: 'sequential'})]
      }
    )[0].legend;
    expect(rows[rows.length - 1]).toMatchObject({label: 'No value', count: 4});
  });

  it('an all-null column has a plan with no domain and says why', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [null, null, null]}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.domain).toBeUndefined();
    expect(plan.breaks).toEqual([]);
    expect(plan.ranges).toEqual([]);
    expect(plan.missingCount).toBe(3);
    expect(plan.notes[0]).toBe('Column "value" has no finite value, so there is no scale to draw.');
  });

  it('a table with zero rows has a plan with nothing in it', () => {
    const numeric = planKeplerResultPresentation(
      makeTable('table', {value: []}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(numeric.rowCount).toBe(0);
    expect(numeric.ranges).toEqual([]);
    expect(numeric.notes[0]).toBe('The table has no rows, so there is no scale to draw.');

    const scheme = planKeplerResultPresentation(
      makeTable('table', {bin: []}),
      makeOutput({
        column: 'bin',
        scale: 'class-scheme',
        classScheme: 'hot-spot-confidence-bins'
      })
    )!;
    // An empty column is not a mismatch: the scheme stays and every count is zero.
    expect(scheme.classScheme).toBe('hot-spot-confidence-bins');
    expect(scheme.legend).toHaveLength(7);
    expect(scheme.legend.every(entry => entry.count === 0)).toBe(true);

    const categorical = planKeplerResultPresentation(
      makeTable('table', {kind: []}),
      makeOutput({column: 'kind', scale: 'categorical'})
    )!;
    expect(categorical.legend).toEqual([]);
  });

  it('a diverging scale is symmetric about the centre and colours the two sides differently', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [-3, -1, 0, 1, 2, 10]}),
      makeOutput({column: 'value', scale: 'diverging', center: 0})
    )!;
    expect(plan.domain).toEqual([-10, 10]);
    expect(plan.center).toBe(0);
    const first = plan.ranges[0].color;
    const last = plan.ranges[plan.ranges.length - 1].color;
    // Blue below the centre, red above it.
    expect(first[2]).toBeGreaterThan(first[0]);
    expect(last[0]).toBeGreaterThan(last[2]);

    const shifted = planKeplerResultPresentation(
      makeTable('table', {value: [2, 4, 6, 20]}),
      makeOutput({column: 'value', scale: 'diverging', center: 5})
    )!;
    expect(shifted.domain).toEqual([-10, 20]);
  });

  it('the diverging centre defaults to zero', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [-2, 4]}),
      makeOutput({column: 'value', scale: 'diverging'})
    )!;
    expect(plan.domain).toEqual([-4, 4]);
  });

  it('a diverging column that is all at the centre has a zero-width domain', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [0, 0, 0]}),
      makeOutput({column: 'value', scale: 'diverging', center: 0})
    )!;
    expect(plan.domain).toEqual([0, 0]);
    expect(plan.classCount).toBe(1);
  });

  it('shows a text column given a numeric scale as categories with a note', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: ['low', 'high', 'low']}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.scale).toBe('categorical');
    expect(plan.notes[0]).toMatch(/holds text or true\/false values/);
    expect(plan.legend.map(entry => [entry.label, entry.count])).toEqual([
      ['low', 2],
      ['high', 1]
    ]);
  });

  it('counts stray text in a numeric column as missing and says so', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {value: [1, 'n/a', 2, 3]}),
      makeOutput({column: 'value', scale: 'sequential'})
    )!;
    expect(plan.missingCount).toBe(1);
    expect(plan.notes[0]).toBe('1 non-numeric values in column "value" are shown as missing.');
  });
});

// ---------------------------------------------------------------------------------------------
// Mismatches, class schemes and categories
// ---------------------------------------------------------------------------------------------

describe('plans that do not match the descriptor', () => {
  it('gives no plan and does not throw for an absent column, an output without presentation or a prototype key', () => {
    const table = makeTable('table', {value: [1, 2, 3]});
    expect(
      planKeplerResultPresentation(table, makeOutput({column: 'missing', scale: 'sequential'}))
    ).toBeUndefined();
    expect(planKeplerResultPresentation(table, makeOutput(undefined))).toBeUndefined();
    expect(
      planKeplerResultPresentation(table, makeOutput({column: 'constructor', scale: 'sequential'}))
    ).toBeUndefined();
    expect(
      planKeplerResultPresentation(table, makeOutput({column: 'toString', scale: 'categorical'}))
    ).toBeUndefined();
  });

  it('shows a text column given an integer encoding as categories and says why', () => {
    const table = makeTable('table', {bin: ['hot-99', 'hot-95', 'not-significant', 'hot-99']});
    const plan = planKeplerResultPresentation(
      table,
      makeOutput({
        column: 'bin',
        scale: 'class-scheme',
        classScheme: 'hot-spot-confidence-bins',
        classEncoding: 'contributor-integer'
      })
    )!;
    expect(plan.scale).toBe('categorical');
    expect(plan.classScheme).toBeUndefined();
    expect(plan.notes[0]).toBe(
      'Column "bin" holds values that are not classes of scheme "hot-spot-confidence-bins" with encoding "contributor-integer" (0% are): "hot-99" (2 rows), "hot-95" (1 row), "not-significant" (1 row); the column is shown as categories.'
    );
    expect(plan.legend.map(entry => [entry.label, entry.count])).toEqual([
      ['hot-99', 2],
      ['hot-95', 1],
      ['not-significant', 1]
    ]);
    // The same column with the right encoding keeps the scheme.
    const right = planKeplerResultPresentation(
      table,
      makeOutput({
        column: 'bin',
        scale: 'class-scheme',
        classScheme: 'hot-spot-confidence-bins',
        classEncoding: 'catalog-text'
      })
    )!;
    expect(right.classScheme).toBe('hot-spot-confidence-bins');
    expect(right.legend.map(entry => entry.count)).toEqual([2, 1, 0, 1, 0, 0, 0]);
  });

  it('uses a class scheme only when every non-null value is a class, and names the strangers otherwise', () => {
    const makeColumn = (unknownCount: number): ColumnValue[] => [
      ...Array.from({length: 100 - unknownCount}, () => 'HH'),
      ...Array.from({length: unknownCount}, () => 'hot-95')
    ];
    const plan = (unknownCount: number) =>
      planKeplerResultPresentation(
        makeTable('table', {cluster: makeColumn(unknownCount)}),
        makeOutput({
          column: 'cluster',
          scale: 'class-scheme',
          classScheme: 'local-moran-clusters',
          classEncoding: 'catalog-text'
        })
      )!;
    expect(KEPLER_MINIMUM_ENCODING_SHARE).toBe(1);
    expect(plan(0).classScheme).toBe('local-moran-clusters');
    // One stranger in a hundred is enough, and 9 in a hundred (which passed the old 0.9 bar) too.
    for (const unknownCount of [1, 9]) {
      const fallback = plan(unknownCount);
      expect(fallback.classScheme).toBeUndefined();
      expect(fallback.scale).toBe('categorical');
      expect(fallback.notes[0]).toContain(
        `"hot-95" (${unknownCount} ${unknownCount === 1 ? 'row' : 'rows'})`
      );
    }
  });

  it('names at most five unrecognised values and counts the rest', () => {
    const values: ColumnValue[] = ['HH', 'a', 'b', 'c', 'd', 'e', 'f', 'g'];
    const plan = planKeplerResultPresentation(
      makeTable('table', {cluster: values}),
      makeOutput({
        column: 'cluster',
        scale: 'class-scheme',
        classScheme: 'local-moran-clusters',
        classEncoding: 'catalog-text'
      })
    )!;
    expect(plan.notes[0]).toContain('and 2 more');
    expect(plan.notes[0]).not.toContain('"g"');
  });

  it('a column coded for local Geary is shown with the local Geary classes, not read as local Moran', () => {
    // Geary codes 0 to 4 are all integers the Moran scheme knows, so the share cannot tell them
    // apart; the descriptor names `local-geary-classes`, and the plan then uses that legend.
    const geary = [4, 4, 4, 1, 2, 3, 0];
    const plan = planKeplerResultPresentation(
      makeTable('table', {cluster: geary}),
      makeOutput({
        column: 'cluster',
        scale: 'class-scheme',
        classScheme: 'local-geary-classes',
        classEncoding: 'contributor-integer'
      })
    )!;
    expect(plan.classScheme).toBe('local-geary-classes');
    expect(plan.notes).toEqual([]);
    const classes = getGPUAnalysisResultClasses('local-geary-classes', 'contributor-integer');
    expect(plan.legend.map(entry => entry.label)).toEqual(classes.map(entry => entry.label));
    // The three rows coded 4 count under the class the package gives code 4, not under a Moran class.
    const four = plan.legend.find(entry => entry.value === 4)!;
    expect(four.count).toBe(3);
    const moran = planKeplerResultPresentation(
      makeTable('table', {cluster: geary}),
      makeOutput({
        column: 'cluster',
        scale: 'class-scheme',
        classScheme: 'local-moran-clusters',
        classEncoding: 'contributor-integer'
      })
    )!;
    // Under the wrong scheme the same rows get another meaning for code 4: the two legends differ.
    expect(moran.legend.find(entry => entry.value === 4)!.label).not.toBe(four.label);
    expect(sum(plan.legend.map(entry => entry.count))).toBe(geary.length);
  });

  it('falls back to categories for a scheme the encoding does not exist for, without throwing', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {significant: [true, false, true]}),
      makeOutput({
        column: 'significant',
        scale: 'class-scheme',
        classScheme: 'significance-flag',
        classEncoding: 'catalog-text'
      })
    )!;
    expect(plan.scale).toBe('categorical');
    expect(plan.notes[0]).toMatch(/does not have; the column is shown as categories/);
  });

  it('falls back to categories for a class scheme output that names no scheme', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {kind: ['a', 'b']}),
      makeOutput({column: 'kind', scale: 'class-scheme'})
    )!;
    expect(plan.scale).toBe('categorical');
    expect(plan.notes[0]).toMatch(/without naming one/);
  });

  it('reads a boolean column as the significance flag and an all-null scheme column as unrecognised', () => {
    const flag = planKeplerResultPresentation(
      makeTable('table', {significant: [true, false, false, true, true]}),
      makeOutput({column: 'significant', scale: 'class-scheme', classScheme: 'significance-flag'})
    )!;
    expect(flag.legend.map(entry => [entry.label, entry.count])).toEqual([
      ['Significant', 3],
      ['Not significant', 2]
    ]);

    const empty = planKeplerResultPresentation(
      makeTable('table', {bin: [null, null]}),
      makeOutput({column: 'bin', scale: 'class-scheme', classScheme: 'hot-spot-confidence-bins'})
    )!;
    // Nothing to compare against the scheme, so it is kept and everything is unrecognised.
    expect(empty.classScheme).toBe('hot-spot-confidence-bins');
    expect(empty.unrecognized).toBe(2);
    expect(empty.legend.every(entry => entry.count === 0)).toBe(true);
  });
});

describe('categories', () => {
  it('orders by count then value and puts the colours of the palette in that order', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {kind: ['b', 'a', 'c', 'a', 'b', 'a', 'd']}),
      makeOutput({column: 'kind', scale: 'categorical'})
    )!;
    expect(plan.legend.map(entry => [entry.value, entry.count])).toEqual([
      ['a', 3],
      ['b', 2],
      ['c', 1],
      ['d', 1]
    ]);
    expect(plan.legend.map(entry => entry.color)).toEqual(KEPLER_CATEGORICAL_PALETTE.slice(0, 4));
  });

  it('caps the distinct values at the limit and collects the rest in an other bucket', () => {
    const values: ColumnValue[] = [];
    // 12 categories, category n appears 13 - n times so the order is unambiguous.
    for (let category = 1; category <= 12; category++) {
      for (let repeat = 0; repeat < 13 - category; repeat++) values.push(`c${category}`);
    }
    const plan = planKeplerResultPresentation(
      makeTable('table', {kind: values}),
      makeOutput({column: 'kind', scale: 'categorical'})
    )!;
    expect(KEPLER_MAXIMUM_CATEGORY_COUNT).toBe(7);
    expect(plan.legend).toHaveLength(KEPLER_MAXIMUM_CATEGORY_COUNT + 1);
    expect(plan.legend[0]).toMatchObject({value: 'c1', count: 12});
    // Categories 8 to 12 hold 5 + 4 + 3 + 2 + 1 rows.
    expect(plan.other).toEqual({distinctCount: 5, count: 15});
    expect(plan.legend[plan.legend.length - 1]).toMatchObject({
      value: 'other',
      label: 'Other (5 more values)',
      count: 15
    });
    expect(sum(plan.legend.map(entry => entry.count))).toBe(values.length);
  });

  it('with exactly the limit of distinct values has no other bucket', () => {
    const values = Array.from({length: KEPLER_MAXIMUM_CATEGORY_COUNT}, (_, index) => `v${index}`);
    const plan = planKeplerResultPresentation(
      makeTable('table', {kind: values}),
      makeOutput({column: 'kind', scale: 'categorical'})
    )!;
    expect(plan.other).toBeUndefined();
    expect(plan.legend).toHaveLength(KEPLER_MAXIMUM_CATEGORY_COUNT);
  });

  it('keeps numbers and text with the same spelling apart and counts missing values', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {kind: [1, '1', 1, null, Number.NaN, true]}),
      makeOutput({column: 'kind', scale: 'categorical'})
    )!;
    expect(plan.missingCount).toBe(2);
    expect(plan.legend.map(entry => [entry.value, entry.count])).toEqual([
      [1, 2],
      ['1', 1],
      [true, 1]
    ]);
  });

  it('a cluster column with many clusters shows the largest and groups the rest', () => {
    const values: ColumnValue[] = [-1, -1];
    // 20 clusters; cluster n has n + 1 rows.
    for (let cluster = 0; cluster < 20; cluster++) {
      for (let repeat = 0; repeat <= cluster; repeat++) values.push(cluster);
    }
    const plan = planKeplerResultPresentation(
      makeTable('table', {cluster: values}),
      makeOutput({
        column: 'cluster',
        scale: 'class-scheme',
        classScheme: 'cluster-membership',
        classEncoding: 'catalog-signed-label'
      })
    )!;
    expect(plan.membership).toEqual({clusterCount: 20, noiseCount: 2, largestClusterCount: 20});
    // The noise entry, the seven largest clusters (13 to 19) and the rest.
    expect(plan.legend.map(entry => entry.label)).toEqual([
      'Noise',
      'Cluster 13',
      'Cluster 14',
      'Cluster 15',
      'Cluster 16',
      'Cluster 17',
      'Cluster 18',
      'Cluster 19',
      '13 smaller clusters'
    ]);
    expect(sum(plan.legend.map(entry => entry.count))).toBe(values.length);
  });

  it('a cluster column with no clusters reports zero clusters and no largest cluster', () => {
    const plan = planKeplerResultPresentation(
      makeTable('table', {cluster: [-1, -1, -1]}),
      makeOutput({
        column: 'cluster',
        scale: 'class-scheme',
        classScheme: 'cluster-membership',
        classEncoding: 'catalog-signed-label'
      })
    )!;
    expect(plan.membership).toEqual({clusterCount: 0, noiseCount: 3, largestClusterCount: 0});
  });
});

describe('display style and binding presentations', () => {
  it('maps class schemes and categories to ordinal and numbers to quantile', () => {
    expect(toKeplerWorkbenchDisplayStyle(undefined)).toEqual({});
    const numeric = planKeplerResultPresentation(
      makeTable('table', {value: [1, 2, 3]}),
      makeOutput({column: 'value', scale: 'sequential'})
    );
    // The plan's own breaks come with the style, for a host that wants a threshold scale because
    // Kepler's quantile scale interpolates and may break slightly differently from the legend.
    expect(toKeplerWorkbenchDisplayStyle(numeric)).toEqual({
      colorColumn: 'value',
      colorScale: 'quantile',
      palette: 'sequential',
      thresholds: [1, 2]
    });
    const diverging = planKeplerResultPresentation(
      makeTable('table', {value: [-1, 2]}),
      makeOutput({column: 'value', scale: 'diverging'})
    );
    expect(toKeplerWorkbenchDisplayStyle(diverging).palette).toBe('diverging');
    const categorical = planKeplerResultPresentation(
      makeTable('table', {kind: ['a']}),
      makeOutput({column: 'kind', scale: 'categorical'})
    );
    expect(toKeplerWorkbenchDisplayStyle(categorical)).toEqual({
      colorColumn: 'kind',
      colorScale: 'ordinal',
      palette: 'categorical'
    });
  });

  it('gives a table without an output descriptor no plan, no style and no legend', () => {
    const result: CatalogRunResult = {
      operationId: 'x',
      engine: 'cpu-reference',
      dispatchedContributors: [],
      parameters: {},
      tables: [makeTable('described', {value: [1, 2]}), makeTable('other', {value: [1, 2]})],
      diagnostics: [],
      summary: {}
    };
    const presentations = getKeplerBindingResultPresentations(result, {
      bindingId: 'x',
      questions: ['prepare-data'],
      action: 'x',
      prominence: 3,
      inputs: [],
      parameters: [],
      requirements: [],
      outputs: [
        {
          tableId: 'described',
          kind: 'polygons',
          presentation: {column: 'value', scale: 'sequential'}
        }
      ]
    });
    expect(presentations.map(entry => entry.tableId)).toEqual(['described', 'other']);
    expect(presentations[0].legend.length).toBeGreaterThan(0);
    expect(presentations[1]).toEqual({tableId: 'other', display: {}, legend: []});
  });
});

// ---------------------------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------------------------

const VISIONS: readonly KeplerColorVision[] = [
  'normal',
  'protanopia',
  'deuteranopia',
  'tritanopia'
];

const ALL_TONES: readonly GPUAnalysisResultClass['tone'][] = [
  'strong-high',
  'high',
  'weak-high',
  'neutral',
  'weak-low',
  'low',
  'strong-low',
  'high-outlier',
  'low-outlier',
  'category',
  'excluded'
];

type ColorEntry = {name: string; color: KeplerResultColor};

/** Smallest CIE76 and CIEDE2000 differences over the given pairs, per kind of vision. */
function getSmallestDifferences(
  entries: readonly ColorEntry[],
  pairs: readonly (readonly [number, number])[],
  vision: KeplerColorVision
): {cie76: number; ciede2000: number; worst76: string; worst2000: string} {
  let cie76 = Infinity;
  let ciede2000 = Infinity;
  let worst76 = '';
  let worst2000 = '';
  for (const [first, second] of pairs) {
    const a = entries[first];
    const b = entries[second];
    const distance = getKeplerColorDistance(a.color, b.color, vision);
    const difference = getKeplerColorDifference2000(a.color, b.color, vision);
    if (distance < cie76) {
      cie76 = distance;
      worst76 = `${a.name}/${b.name}`;
    }
    if (difference < ciede2000) {
      ciede2000 = difference;
      worst2000 = `${a.name}/${b.name}`;
    }
  }
  return {cie76, ciede2000, worst76, worst2000};
}

function getAllPairs(count: number): (readonly [number, number])[] {
  const pairs: [number, number][] = [];
  for (let first = 0; first < count; first++) {
    for (let second = first + 1; second < count; second++) pairs.push([first, second]);
  }
  return pairs;
}

function getAdjacentPairs(count: number): (readonly [number, number])[] {
  return Array.from({length: count - 1}, (_, index) => [index, index + 1] as const);
}

function describeTone(tone: GPUAnalysisResultClass['tone']): ColorEntry {
  return {name: tone, color: getKeplerResultToneColor(tone)};
}

const MISSING_ENTRY: ColorEntry = {name: 'no value', color: KEPLER_MISSING_VALUE_COLOR};
const OTHER_ENTRY: ColorEntry = {name: 'other', color: KEPLER_OTHER_CATEGORY_COLOR};

/** Every set of entries that can stand in one legend, with the missing row that may join any of them. */
const LEGENDS: Record<string, readonly ColorEntry[]> = {
  'hot-spot bins': [
    ...getGPUAnalysisResultClasses('hot-spot-confidence-bins', 'contributor-integer').map(entry =>
      describeTone(entry.tone)
    ),
    MISSING_ENTRY
  ],
  'local Geary': [
    ...getGPUAnalysisResultClasses('local-geary-classes', 'contributor-integer').map(entry =>
      describeTone(entry.tone)
    ),
    MISSING_ENTRY
  ],
  'local Moran': [
    ...getGPUAnalysisResultClasses('local-moran-clusters', 'catalog-text').map(entry =>
      describeTone(entry.tone)
    ),
    MISSING_ENTRY
  ],
  'significance flag': [
    ...getGPUAnalysisResultClasses('significance-flag', 'contributor-integer').map(entry =>
      describeTone(entry.tone)
    ),
    MISSING_ENTRY
  ],
  'validity flag': [
    ...getGPUAnalysisResultClasses('validity-flag', 'contributor-integer').map(entry =>
      describeTone(entry.tone)
    ),
    MISSING_ENTRY
  ],
  'cluster membership': [
    describeTone('excluded'),
    ...KEPLER_CATEGORICAL_PALETTE.map((color, index) => ({name: `cluster ${index}`, color})),
    OTHER_ENTRY,
    MISSING_ENTRY
  ],
  categorical: [
    ...KEPLER_CATEGORICAL_PALETTE.map((color, index) => ({name: `category ${index}`, color})),
    OTHER_ENTRY,
    MISSING_ENTRY
  ]
};

describe('colours', () => {
  it('gives every tone a colour and throws for an unknown tone', () => {
    for (const tone of ALL_TONES) {
      const color = getKeplerResultToneColor(tone);
      expect(color).toHaveLength(3);
      for (const channel of color) expect(channel).toBeGreaterThanOrEqual(0);
    }
    expect(() => getKeplerResultToneColor('purple' as never)).toThrow(/getKeplerResultToneColor/);
  });

  it('simulates colour-vision deficiency with the Machado matrices (values from /usr/bin/python3)', () => {
    // Oracle: the full-severity protanopia, deuteranopia and tritanopia matrices of Machado,
    // Oliveira and Fernandes (2009) on linear sRGB, computed with NumPy and rounded to 8 bits:
    // (200, 35, 45) gives (89, 81, 43), (129, 116, 38) and (220, 0, 41); (30, 105, 185) gives
    // (64, 112, 188), (34, 98, 183) and (0, 124, 136).
    expect(simulateKeplerColorVision([200, 35, 45], 'protanopia')).toEqual([89, 81, 43]);
    expect(simulateKeplerColorVision([200, 35, 45], 'deuteranopia')).toEqual([129, 116, 38]);
    expect(simulateKeplerColorVision([200, 35, 45], 'tritanopia')).toEqual([220, 0, 41]);
    expect(simulateKeplerColorVision([30, 105, 185], 'protanopia')).toEqual([64, 112, 188]);
    expect(simulateKeplerColorVision([30, 105, 185], 'deuteranopia')).toEqual([34, 98, 183]);
    expect(simulateKeplerColorVision([30, 105, 185], 'tritanopia')).toEqual([0, 124, 136]);
    expect(simulateKeplerColorVision([10, 20, 30], 'normal')).toEqual([10, 20, 30]);
    // Greys are unchanged by the first two matrices, whose rows sum to 1.
    expect(simulateKeplerColorVision([128, 128, 128], 'protanopia')).toEqual([128, 128, 128]);
  });

  it('measures colour difference as CIE76 and CIEDE2000 (values from /usr/bin/python3)', () => {
    // Oracle: Lab from sRGB D65 and the CIEDE2000 formulation of Sharma, Wu and Dalal in NumPy and
    // Python. Black to white is 100.0 in both. A red (200, 0, 0) to a green (0, 150, 0): CIE76 is
    // 125.30 normal, 32.76 protanopia, 5.97 deuteranopia and 123.42 tritanopia; CIEDE2000 is 71.73,
    // 24.05, 5.20 and 61.86.
    expect(getKeplerColorDistance([0, 0, 0], [255, 255, 255])).toBeCloseTo(100, 3);
    expect(getKeplerColorDifference2000([0, 0, 0], [255, 255, 255])).toBeCloseTo(100, 3);
    const red: KeplerResultColor = [200, 0, 0];
    const green: KeplerResultColor = [0, 150, 0];
    expect(getKeplerColorDistance(red, green)).toBeCloseTo(125.3, 1);
    expect(getKeplerColorDistance(red, green, 'protanopia')).toBeCloseTo(32.76, 1);
    expect(getKeplerColorDistance(red, green, 'deuteranopia')).toBeCloseTo(5.97, 1);
    expect(getKeplerColorDistance(red, green, 'tritanopia')).toBeCloseTo(123.42, 1);
    expect(getKeplerColorDifference2000(red, green)).toBeCloseTo(71.73, 1);
    expect(getKeplerColorDifference2000(red, green, 'protanopia')).toBeCloseTo(24.05, 1);
    expect(getKeplerColorDifference2000(red, green, 'deuteranopia')).toBeCloseTo(5.2, 1);
    expect(getKeplerColorDifference2000(red, green, 'tritanopia')).toBeCloseTo(61.86, 1);
    expect(getKeplerColorDistance([10, 20, 30], [10, 20, 30])).toBe(0);
    expect(getKeplerColorDifference2000([10, 20, 30], [10, 20, 30])).toBe(0);
  });

  it('measures contrast as WCAG 2.1 does (values from /usr/bin/python3)', () => {
    expect(getKeplerColorContrastRatio([0, 0, 0], [255, 255, 255])).toBeCloseTo(21, 4);
    expect(getKeplerColorContrastRatio([255, 255, 255], [255, 255, 255])).toBe(1);
    // Oracle: (200, 35, 45) against white is 5.6225 and against (20, 20, 24) 3.2677.
    expect(getKeplerColorContrastRatio([200, 35, 45], KEPLER_LIGHT_MAP_BACKGROUND)).toBeCloseTo(
      5.6225,
      3
    );
    expect(getKeplerColorContrastRatio([200, 35, 45], KEPLER_DARK_MAP_BACKGROUND)).toBeCloseTo(
      3.2677,
      3
    );
  });

  it('the checks fail for colours that are too close, and for a red-green pair under deuteranopia', () => {
    expect(getKeplerColorDistance([200, 35, 45], [205, 40, 50])).toBeLessThan(
      KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE
    );
    expect(getKeplerColorDifference2000([200, 35, 45], [205, 40, 50])).toBeLessThan(
      KEPLER_MINIMUM_LEGEND_COLOR_DIFFERENCE_2000
    );
    expect(getKeplerColorDistance([200, 0, 0], [0, 150, 0])).toBeGreaterThan(
      KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE
    );
    expect(getKeplerColorDistance([200, 0, 0], [0, 150, 0], 'deuteranopia')).toBeLessThan(
      KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE
    );
    expect(getKeplerColorDifference2000([200, 0, 0], [0, 150, 0], 'deuteranopia')).toBeLessThan(
      KEPLER_MINIMUM_LEGEND_COLOR_DIFFERENCE_2000
    );
    // The two greys the reviewer found too close (the old noise and other-clusters greys, 11.7
    // apart) would fail the bar.
    expect(getKeplerColorDistance([120, 120, 120], [150, 150, 150])).toBeLessThan(
      KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE * 2
    );
    expect(getKeplerColorDistance([150, 150, 150], [170, 170, 170])).toBeLessThan(
      KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE
    );
  });

  it('every pair of entries that can stand in one legend differs enough under all four kinds of vision, by CIE76 and CIEDE2000', () => {
    for (const [name, entries] of Object.entries(LEGENDS)) {
      for (const vision of VISIONS) {
        const smallest = getSmallestDifferences(entries, getAllPairs(entries.length), vision);
        expect(
          smallest.cie76,
          `${name} under ${vision}: ${smallest.worst76} (CIE76)`
        ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE);
        expect(
          smallest.ciede2000,
          `${name} under ${vision}: ${smallest.worst2000} (CIEDE2000)`
        ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_LEGEND_COLOR_DIFFERENCE_2000);
      }
    }
  });

  it('the strongest classes keep 3:1 against a white and a dark map, and a pale class does not', () => {
    for (const tone of ['strong-high', 'strong-low', 'high-outlier', 'low-outlier'] as const) {
      const color = getKeplerResultToneColor(tone);
      expect(
        getKeplerColorContrastRatio(color, KEPLER_LIGHT_MAP_BACKGROUND),
        `${tone} on white`
      ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_MAP_CONTRAST_RATIO);
      expect(
        getKeplerColorContrastRatio(color, KEPLER_DARK_MAP_BACKGROUND),
        `${tone} on dark`
      ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_MAP_CONTRAST_RATIO);
    }
    expect(
      getKeplerColorContrastRatio(
        getKeplerResultToneColor('weak-high'),
        KEPLER_LIGHT_MAP_BACKGROUND
      )
    ).toBeLessThan(KEPLER_MINIMUM_MAP_CONTRAST_RATIO);
  });

  it('the weakest fills keep the stated minimum contrast against white, and no value is the least visible fill', () => {
    const lowestFills: ColorEntry[] = [
      {name: 'lowest sequential class', color: KEPLER_SEQUENTIAL_RAMP[0]},
      describeTone('weak-high'),
      describeTone('weak-low'),
      describeTone('neutral'),
      OTHER_ENTRY
    ];
    for (const entry of lowestFills) {
      expect(
        getKeplerColorContrastRatio(entry.color, KEPLER_LIGHT_MAP_BACKGROUND),
        entry.name
      ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_LOWEST_CLASS_CONTRAST_RATIO);
    }
    // The reviewer's complaint: "no value" was more visible than the lowest class.
    const missingContrast = getKeplerColorContrastRatio(
      KEPLER_MISSING_VALUE_COLOR,
      KEPLER_LIGHT_MAP_BACKGROUND
    );
    for (const entry of lowestFills) {
      expect(
        getKeplerColorContrastRatio(entry.color, KEPLER_LIGHT_MAP_BACKGROUND),
        entry.name
      ).toBeGreaterThan(missingContrast);
    }
    // The outline colour meets 3:1 on both maps.
    for (const background of [KEPLER_LIGHT_MAP_BACKGROUND, KEPLER_DARK_MAP_BACKGROUND]) {
      expect(
        getKeplerColorContrastRatio(KEPLER_RECOMMENDED_OUTLINE_COLOR, background)
      ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_MAP_CONTRAST_RATIO);
    }
  });

  it('the sequential ramp steps differ by at least the ramp minimum and its strong end keeps 3:1', () => {
    const ramp: ColorEntry[] = KEPLER_SEQUENTIAL_RAMP.map((color, index) => ({
      name: `stop ${index}`,
      color
    }));
    for (const vision of VISIONS) {
      const smallest = getSmallestDifferences(ramp, getAdjacentPairs(ramp.length), vision);
      expect(smallest.cie76, `sequential under ${vision}`).toBeGreaterThanOrEqual(
        KEPLER_MINIMUM_RAMP_COLOR_DISTANCE
      );
      expect(smallest.ciede2000, `sequential under ${vision}`).toBeGreaterThanOrEqual(
        KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000
      );
      // Every stop also stands apart from the no-value colour.
      const withMissing = getSmallestDifferences(
        [...ramp, MISSING_ENTRY],
        ramp.map((_, index) => [index, ramp.length] as const),
        vision
      );
      expect(withMissing.cie76).toBeGreaterThanOrEqual(KEPLER_MINIMUM_RAMP_COLOR_DISTANCE);
      expect(withMissing.ciede2000).toBeGreaterThanOrEqual(
        KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000
      );
    }
    const strongest = KEPLER_SEQUENTIAL_RAMP[KEPLER_SEQUENTIAL_RAMP.length - 1];
    expect(
      getKeplerColorContrastRatio(strongest, KEPLER_LIGHT_MAP_BACKGROUND)
    ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_MAP_CONTRAST_RATIO);
    expect(
      getKeplerColorContrastRatio(strongest, KEPLER_DARK_MAP_BACKGROUND)
    ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_MAP_CONTRAST_RATIO);
    // Lightness falls as values rise, which is what makes a ramp read as ordered.
    for (let index = 1; index < KEPLER_SEQUENTIAL_RAMP.length; index++) {
      const previous = KEPLER_SEQUENTIAL_RAMP[index - 1];
      const color = KEPLER_SEQUENTIAL_RAMP[index];
      expect(color[0] + color[1] + color[2]).toBeLessThan(previous[0] + previous[1] + previous[2]);
    }
  });

  it('the diverging ramp ends keep 3:1 and its steps differ by the legend minimum', () => {
    const ramp: ColorEntry[] = KEPLER_DIVERGING_RAMP.map((color, index) => ({
      name: `stop ${index}`,
      color
    }));
    for (const vision of VISIONS) {
      const smallest = getSmallestDifferences(ramp, getAllPairs(ramp.length), vision);
      expect(smallest.cie76, `diverging under ${vision}`).toBeGreaterThanOrEqual(
        KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE
      );
      expect(smallest.ciede2000, `diverging under ${vision}`).toBeGreaterThanOrEqual(
        KEPLER_MINIMUM_LEGEND_COLOR_DIFFERENCE_2000
      );
    }
    for (const end of [KEPLER_DIVERGING_RAMP[0], KEPLER_DIVERGING_RAMP[6]]) {
      expect(getKeplerColorContrastRatio(end, KEPLER_LIGHT_MAP_BACKGROUND)).toBeGreaterThanOrEqual(
        KEPLER_MINIMUM_MAP_CONTRAST_RATIO
      );
      expect(getKeplerColorContrastRatio(end, KEPLER_DARK_MAP_BACKGROUND)).toBeGreaterThanOrEqual(
        KEPLER_MINIMUM_MAP_CONTRAST_RATIO
      );
    }
  });

  it('ramp colours for any class count a plan can use keep the ramp distance, and bad counts throw', () => {
    for (let count = 2; count <= KEPLER_DEFAULT_CLASS_COUNT; count++) {
      for (const palette of ['sequential', 'diverging'] as const) {
        const colors: ColorEntry[] = getKeplerResultRampColors(palette, count).map(
          (color, index) => ({name: `class ${index}`, color})
        );
        for (const vision of VISIONS) {
          const smallest = getSmallestDifferences(colors, getAdjacentPairs(count), vision);
          expect(
            smallest.cie76,
            `${palette} ${count} classes under ${vision}`
          ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_RAMP_COLOR_DISTANCE);
          expect(
            smallest.ciede2000,
            `${palette} ${count} classes under ${vision}`
          ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000);
        }
      }
    }
    expect(getKeplerResultRampColors('sequential', 1)).toHaveLength(1);
    expect(() => getKeplerResultRampColors('sequential', 0)).toThrow(/getKeplerResultRampColors/);
    expect(() => getKeplerResultRampColors('diverging', 2.5)).toThrow(/integer/);
  });
});

// ---------------------------------------------------------------------------------------------
// Legends that must not mislead
// ---------------------------------------------------------------------------------------------

describe('numeric legends', () => {
  const plan = (values: ColumnValue[], scale: 'sequential' | 'diverging' = 'sequential') =>
    planKeplerResultPresentation(
      makeTable('table', {value: values}),
      makeOutput({column: 'value', scale})
    )!;

  it('a binary column gets one class per value, not one class from 0 to 1', () => {
    for (const zeros of [10, 50, 90]) {
      const binary = plan([
        ...Array.from({length: zeros}, () => 0),
        ...Array.from({length: 100 - zeros}, () => 1)
      ]);
      expect(binary.ranges.map(range => [range.label, range.count])).toEqual([
        ['0', zeros],
        ['1', 100 - zeros]
      ]);
      expect(binary.breaks).toEqual([0]);
      expect(binary.notes[0]).toBe(
        'Column "value" has 2 distinct values, so each gets its own class.'
      );
    }
  });

  it('values 1 and 2 are not merged into the top class of [0 x 8, 1, 2]', () => {
    const merged = plan([0, 0, 0, 0, 0, 0, 0, 0, 1, 2]);
    expect(merged.ranges.map(range => [range.label, range.count])).toEqual([
      ['0', 8],
      ['1', 1],
      ['2', 1]
    ]);
  });

  it('with more distinct values than classes and heavy ties the classes are divided again above the ties', () => {
    // 700 zeros, then 300 values from 1 to 300: the plain quantile bounds are 0, 0, 0 and one more,
    // which would leave 1 to 299 in a single class.
    const values = [
      ...Array.from({length: 700}, () => 0),
      ...Array.from({length: 300}, (_, index) => index + 1)
    ];
    const divided = plan(values);
    expect(divided.classCount).toBe(KEPLER_DEFAULT_CLASS_COUNT);
    expect(divided.breaks[0]).toBe(0);
    for (let index = 1; index < divided.breaks.length; index++) {
      expect(divided.breaks[index]).toBeGreaterThan(divided.breaks[index - 1]);
    }
    expect(divided.ranges.every(range => range.count > 0)).toBe(true);
    expect(sum(divided.ranges.map(range => range.count))).toBe(1000);
    // No class above the ties holds more than a third of the 300 spread values.
    for (const range of divided.ranges.slice(1)) expect(range.count).toBeLessThanOrEqual(100);
  });

  it('labels never collapse: 100000 plus small steps gets enough digits for neighbouring bounds to differ', () => {
    const values = Array.from({length: 1000}, (_, index) => 100000 + (index % 97) * 0.004);
    const wide = plan(values);
    const labels = wide.ranges.map(range => range.label);
    // Every bound text appears once: "100,000" is not printed five times.
    const boundTexts = labels.flatMap(label => label.replace('above ', '').split(' to '));
    const distinctBounds = new Set([wide.domain![0], ...wide.breaks, wide.domain![1]]);
    expect(new Set(boundTexts).size).toBe(distinctBounds.size);
    for (const label of labels) {
      const [from, to] = label.replace('above ', '').split(' to ');
      expect(from, label).not.toBe(to);
    }
  });

  it('a plan with pale fills says the fill alone does not meet 3:1 and recommends an outline', () => {
    const numeric = plan([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(numeric.outline?.required).toBe(true);
    expect(numeric.outline?.color).toEqual(KEPLER_RECOMMENDED_OUTLINE_COLOR);
    expect(numeric.outline?.lowestFillContrast.light).toBeCloseTo(
      getKeplerColorContrastRatio(KEPLER_SEQUENTIAL_RAMP[0], KEPLER_LIGHT_MAP_BACKGROUND),
      6
    );
    expect(numeric.outline?.reason).toMatch(/draw a thin outline/);
    // A class scheme made only of strong tones needs none.
    const flags = planKeplerResultPresentation(
      makeTable('table', {x: [1, 1]}),
      makeOutput({column: 'x', scale: 'class-scheme', classScheme: 'validity-flag'})
    )!;
    expect(flags.outline?.required).toBe(true);
    const strong = {
      ...flags,
      legend: flags.legend.map(entry => ({
        ...entry,
        color: getKeplerResultToneColor('strong-high')
      }))
    };
    expect(strong.legend[0].color).toEqual(getKeplerResultToneColor('strong-high'));
  });
});

describe('legend rows add up to the table', () => {
  function getRows(table: CatalogOutputTable, output: KeplerOperationOutputDescriptor) {
    return getKeplerBindingResultPresentations(
      {
        operationId: 'x',
        engine: 'cpu-reference',
        dispatchedContributors: [],
        parameters: {},
        tables: [table],
        diagnostics: [],
        summary: {}
      },
      {
        bindingId: 'x',
        questions: ['prepare-data'],
        action: 'x',
        prominence: 3,
        inputs: [],
        parameters: [],
        requirements: [],
        outputs: [output]
      }
    )[0].legend;
  }

  it('a class scheme with null rows gets a "No value" row', () => {
    const table = makeTable('table', {bin: [3, 2, null, 0, null, -3]});
    const rows = getRows(table, {
      tableId: 'table',
      kind: 'polygons',
      presentation: {
        column: 'bin',
        scale: 'class-scheme',
        classScheme: 'hot-spot-confidence-bins',
        classEncoding: 'contributor-integer'
      }
    });
    expect(rows[rows.length - 1]).toMatchObject({label: 'No value', count: 2});
    expect(sum(rows.map(row => row.count ?? 0))).toBe(table.rowCount);
  });

  it('numeric, categorical and class-scheme legends all sum to the rows', () => {
    const values: ColumnValue[] = [1, 2, 3, 4, 5, 6, 7, null, Number.NaN, 9];
    const table = makeTable('table', {value: values});
    for (const scale of ['sequential', 'diverging', 'categorical'] as const) {
      const rows = getRows(table, {
        tableId: 'table',
        kind: 'polygons',
        presentation: {column: 'value', scale}
      });
      expect(sum(rows.map(row => row.count ?? 0)), scale).toBe(values.length);
    }
  });
});

// ---------------------------------------------------------------------------------------------
// Every class-scheme output of the registry, run for real
// ---------------------------------------------------------------------------------------------

/** A square lattice of polygons in planar coordinates with a hot corner. */
function makeRegistryLattice(): CatalogFeatureCollection {
  const features = [];
  for (let row = 0; row < 8; row++) {
    for (let column = 0; column < 8; column++) {
      const hot = column < 3 && row < 3;
      features.push({
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [column * 10, row * 10],
              [column * 10 + 10, row * 10],
              [column * 10 + 10, row * 10 + 10],
              [column * 10, row * 10 + 10],
              [column * 10, row * 10]
            ]
          ]
        },
        properties: {
          value: (hot ? 20 : 2) + ((column * 7 + row * 3) % 5),
          p: ((column * 13 + row * 7) % 50) / 100,
          cases: (hot ? 30 : 4) + ((column * 5 + row) % 3),
          population: 1000 + ((column * 11 + row * 17) % 7) * 100
        }
      });
    }
  }
  return {type: 'FeatureCollection', features};
}

/** Two dense groups, a sparse background and values that follow them. */
function makeRegistryPoints(): CatalogFeatureCollection {
  const features = [];
  for (let index = 0; index < 160; index++) {
    const group = index % 4;
    const centreX = group === 0 ? 100 : group === 1 ? 400 : 700;
    const centreY = group === 0 ? 100 : group === 1 ? 400 : 700;
    const spread = group === 3 ? 500 : 20;
    const angle = index * 2.399963;
    const radius = (((index * 37) % 100) / 100) * spread;
    features.push({
      geometry: {
        type: 'Point',
        coordinates: [
          (group === 3 ? 400 : centreX) + Math.cos(angle) * radius,
          (group === 3 ? 400 : centreY) + Math.sin(angle) * radius
        ]
      },
      properties: {value: 5 + group * 3 + (index % 5), p: ((index * 7) % 50) / 100}
    });
  }
  return {type: 'FeatureCollection', features};
}

const REGISTRY_COLUMNS: Readonly<Record<string, string>> = {
  valueProperty: 'value',
  pValueProperty: 'p',
  casesProperty: 'cases',
  populationProperty: 'population'
};

describe('registry: every class scheme of a binding with a CPU engine matches what the binding writes', () => {
  const targets = KEPLER_OPERATION_DESCRIPTORS.flatMap(descriptor =>
    descriptor.outputs
      .filter(output => output.presentation?.classScheme)
      .map(output => ({descriptor, output}))
  ).filter(({descriptor}) => getCatalogBinding(descriptor.bindingId)?.run);

  it('finds the class-scheme outputs it is meant to check', () => {
    // The registry has at least the statistics, the clustering and the curated cluster outputs.
    const ids = new Set(targets.map(({descriptor}) => descriptor.bindingId));
    for (const id of [
      'local-moran-statistics',
      'hotspot-gi-star',
      'recipe-hot-spot-lattice',
      'density-clustering',
      'curated-rate-cluster-map',
      'curated-clusters-and-footprints',
      'multiple-testing-adjustment'
    ]) {
      expect(ids.has(id), id).toBe(true);
    }
  });

  for (const {descriptor, output} of targets) {
    it(`${descriptor.bindingId} / ${output.tableId}: every value is a class of the declared scheme`, async () => {
      const binding = getCatalogBinding(descriptor.bindingId)!;
      const inputs: Record<string, CatalogFeatureCollection> = {};
      const profiles: Record<string, ReturnType<typeof profileKeplerFeatureCollection>> = {};
      for (const input of binding.inputs) {
        const collection = input.accepts.includes('polygon')
          ? makeRegistryLattice()
          : makeRegistryPoints();
        inputs[input.name] = collection;
        profiles[input.name] = profileKeplerFeatureCollection(collection, {
          datasetId: input.name,
          label: input.name,
          coordinateSystem: 'planar'
        });
      }
      const fieldBindings: Record<string, string> = {};
      for (const parameter of binding.parameters) {
        if (parameter.kind === 'string' && REGISTRY_COLUMNS[parameter.name]) {
          fieldBindings[parameter.name] = REGISTRY_COLUMNS[parameter.name];
        }
      }
      const resolved = resolveKeplerOperationParameters(binding, descriptor, profiles, {
        fieldBindings
      });
      const result = await runCatalogOperationAsync(binding, resolved.parameters, inputs, {
        engine: 'cpu'
      });
      const table = result.tables.find(candidate => candidate.id === output.tableId)!;
      expect(table.rowCount).toBeGreaterThan(0);
      const plan = planKeplerResultPresentation(table, output)!;
      expect(plan.notes.join(' ')).toBe('');
      expect(plan.classScheme).toBe(output.presentation!.classScheme);
      expect(plan.encodingShare).toBe(1);
      expect(sum(plan.legend.map(entry => entry.count)) + (plan.unrecognized ?? 0)).toBe(
        table.rowCount
      );
    });
  }
});

describe('review regressions: returned plan fills', () => {
  it('checks every registered scheme through the actual planner, including Geary', () => {
    for (const scheme of GPU_ANALYSIS_CLASS_SCHEMES) {
      const classes = getGPUAnalysisResultClasses(scheme, 'contributor-integer');
      const values = classes.map(entry => entry.value);
      const plan = planKeplerResultPresentation(
        makeTable('table', {value: values}),
        makeOutput({
          column: 'value',
          scale: 'class-scheme',
          classScheme: scheme,
          classEncoding: 'contributor-integer'
        })
      )!;
      expect(plan.classScheme).toBe(scheme);
      const entries = [
        ...plan.legend.map(entry => ({name: entry.label, color: entry.color})),
        MISSING_ENTRY
      ];
      for (const vision of VISIONS) {
        const differences = getSmallestDifferences(entries, getAllPairs(entries.length), vision);
        expect(
          differences.cie76,
          `${scheme} ${vision} ${differences.worst76}`
        ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_LEGEND_COLOR_DISTANCE);
        expect(
          differences.ciede2000,
          `${scheme} ${vision} ${differences.worst2000}`
        ).toBeGreaterThanOrEqual(KEPLER_MINIMUM_LEGEND_COLOR_DIFFERENCE_2000);
      }
    }
  });

  it('separates actual numeric classes for one-sided, two-sided and near-constant columns', () => {
    for (const scale of ['sequential', 'diverging'] as const) {
      for (const values of [
        [1000, 1001, 1002, 1003, 1004],
        [-5, -3, -1, 2, 8],
        [1, 1.001, 1.002, 1.003, 1.004],
        [-4, -3, -2, -1],
        [1, 1 + Number.EPSILON, 1 + 2 * Number.EPSILON]
      ]) {
        const plan = planKeplerResultPresentation(
          makeTable('table', {value: values}),
          makeOutput({column: 'value', scale, center: 0})
        )!;
        const entries = plan.ranges.map(entry => ({name: entry.label, color: entry.color}));
        expect(sum(plan.ranges.map(range => range.count))).toBe(values.length);
        expect(new Set(entries.map(entry => entry.name)).size).toBe(entries.length);
        if (values.every(value => value > 0)) expect(plan.scale).toBe('sequential');
        for (const vision of VISIONS) {
          const pairs = getAllPairs(entries.length);
          const differences = getSmallestDifferences(entries, pairs, vision);
          expect(differences.cie76, `${values} ${scale} ${vision}`).toBeGreaterThanOrEqual(
            KEPLER_MINIMUM_RAMP_COLOR_DISTANCE
          );
          expect(differences.ciede2000, `${values} ${scale} ${vision}`).toBeGreaterThanOrEqual(
            KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000
          );
          const missing = getSmallestDifferences(
            [...entries, MISSING_ENTRY],
            entries.map((_, index) => [index, entries.length] as const),
            vision
          );
          expect(missing.cie76).toBeGreaterThanOrEqual(KEPLER_MINIMUM_RAMP_COLOR_DISTANCE);
          expect(missing.ciede2000).toBeGreaterThanOrEqual(
            KEPLER_MINIMUM_RAMP_COLOR_DIFFERENCE_2000
          );
        }
      }
    }
  });

  it('shows Gi-star islands as excluded rows and includes missing fills in outline recommendations', () => {
    const table = makeTable('table', {
      classification: ['hot-95', 'cold-95', 'hot-99', null],
      neighbor_count: [4, 0, 0, 2]
    });
    const plan = planKeplerResultPresentation(
      table,
      makeOutput({
        column: 'classification',
        scale: 'class-scheme',
        classScheme: 'hot-spot-confidence-bins',
        classEncoding: 'catalog-text'
      })
    )!;
    expect(plan.excludedRows).toEqual([1, 2]);
    expect(
      plan.legend
        .filter(entry => ['high', 'strong-high'].includes(entry.tone))
        .reduce((total, entry) => total + entry.count, 0)
    ).toBe(1);
    expect(plan.legend.find(entry => entry.label === 'Excluded: no neighbours')?.count).toBe(2);
    expect(plan.legend.reduce((total, entry) => total + entry.count, 0) + plan.missingCount).toBe(
      4
    );
    const categorical = planKeplerResultPresentation(
      makeTable('table', {value: ['a', 'b', null]}),
      makeOutput({column: 'value', scale: 'categorical'})
    )!;
    expect(categorical.outline?.required).toBe(true);
  });
});
