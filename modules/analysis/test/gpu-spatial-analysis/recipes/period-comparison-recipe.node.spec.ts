// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {GPUCellAggregation} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {GPUCellTableCompare} from '../../../src/gpu-spatial-analysis/cell-table-compare/index';
import {
  addPeriodComparisonRecipe,
  type GPUPeriodComparisonRecipeResult
} from '../../../src/gpu-spatial-analysis/recipes/period-comparison-recipe';
import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues
} from '../../../src/dataframe-analysis/column-classification/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('forwards one selected sum scale to both period aggregations and the comparison', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const parameterValues = getGPUClassBreaksParameterValues(
      {method: 'custom', customEdges: [-1, 0, 1]},
      2
    );
    const colorValues = getGPUColorScaleParameterValues({
      scale: 'threshold',
      domainCount: 3,
      paletteCount: 2
    });
    const props = {
      family: 'quadbin' as const,
      resolution: 0,
      tableCapacity: 1,
      measure: 'sum' as const,
      sumScale: 2 ** 18,
      before: {
        positions: createTransientView(graph, 'before-positions', 'float32x2', 1),
        values: createTransientView(graph, 'before-values', 'float32', 1)
      },
      after: {
        positions: createTransientView(graph, 'after-positions', 'float32x2', 1),
        values: createTransientView(graph, 'after-values', 'float32', 1)
      },
      classBreaksParameters: createTransientView(
        graph,
        'break-parameters',
        'float32',
        parameterValues.length
      ),
      maximumClassCount: 2,
      methods: ['custom'] as const,
      palette: createTransientView(graph, 'palette', 'uint32', 2),
      colorScaleParameters: createTransientView(
        graph,
        'color-parameters',
        'float32',
        colorValues.length
      ),
      maximumPaletteCount: 2
    };
    const providedSumOverflow = createTransientView(
      graph,
      'provided-before-sum-overflow',
      'uint32',
      1
    );
    const providedComparisonOverflow = createTransientView(
      graph,
      'provided-compare-sum-overflow',
      'uint32',
      1
    );
    const inputs = {
      ...props,
      scratch: {before: {status: {sumOverflow: providedSumOverflow}}},
      output: {comparison: {status: {sumOverflow: providedComparisonOverflow}}}
    };
    const recipe = addPeriodComparisonRecipe(graph, inputs);
    inputs.scratch.before.status.sumOverflow = createTransientView(
      graph,
      'mutated-before-sum-overflow',
      'uint32',
      1
    );
    expect(recipe.before.status.sumOverflow).toBe(providedSumOverflow);
    expect(recipe.comparison.status.sumOverflow).toBe(providedComparisonOverflow);
    const compare = recipe.contributors.find(
      contributor => contributor instanceof GPUCellTableCompare
    );
    expect(compare instanceof GPUCellTableCompare && compare.props.before.status.sumOverflow).toBe(
      providedSumOverflow
    );
    const sumScales = recipe.contributors.flatMap(contributor =>
      contributor instanceof GPUCellAggregation || contributor instanceof GPUCellTableCompare
        ? [contributor.sumScale]
        : []
    );
    expect(sumScales).toEqual([2 ** 18, 2 ** 18, 2 ** 18]);
    expect(recipe.before.status.sumOverflow).toBeDefined();
    expect(recipe.after.status.sumOverflow).toBeDefined();
    expect(recipe.comparison.status).toHaveProperty('sumOverflow');
    expect(recipe.status.stages.map(stage => stage.status.sumOverflow)).toEqual([
      recipe.before.status.sumOverflow,
      recipe.after.status.sumOverflow,
      recipe.comparison.status.sumOverflow
    ]);
  } finally {
    device.destroy();
  }
});

it('period stage types expose their optional sum-saturation word (S1)', () => {
  type Stage = GPUPeriodComparisonRecipeResult['status']['stages'][number];
  expectTypeOf<Stage['status']['sumOverflow']>().toEqualTypeOf<
    GraphDataView<'uint32'> | undefined
  >();
});
