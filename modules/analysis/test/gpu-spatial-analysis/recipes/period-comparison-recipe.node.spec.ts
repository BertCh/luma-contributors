// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellAggregation} from '../../../src/gpu-spatial-analysis/cell-aggregation/index';
import {GPUCellTableCompare} from '../../../src/gpu-spatial-analysis/cell-table-compare/index';
import {addPeriodComparisonRecipe} from '../../../src/gpu-spatial-analysis/recipes/period-comparison-recipe';
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
    const recipe = addPeriodComparisonRecipe(graph, props);
    const sumScales = recipe.contributors.flatMap(contributor =>
      contributor instanceof GPUCellAggregation || contributor instanceof GPUCellTableCompare
        ? [contributor.sumScale]
        : []
    );
    expect(sumScales).toEqual([2 ** 18, 2 ** 18, 2 ** 18]);
  } finally {
    device.destroy();
  }
});
