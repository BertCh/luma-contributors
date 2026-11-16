// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expectTypeOf, it} from 'vitest';
import {
  GPUScalarScale,
  type GPUScalarScaleProps
} from '../../../src/dataframe-analysis/column-classification/gpu-scalar-scale';
import {
  GPUColorScale,
  type GPUColorScaleOutput
} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import {
  GPUBivariateClassification,
  type GPUBivariateClassificationOutput
} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';
import {
  GPUColumnQuantiles,
  type GPUColumnQuantilesOutput
} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles';
import type {ClassificationRowView} from '../../../src/dataframe-analysis/column-classification/column-classification-rows';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

it('preserves legacy packed output declarations and inferred contributor props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = 4) =>
    createTransientView(graph, name, format, length);
  try {
    const values = view('values', 'float32');
    const domain = view('domain', 'float32', 9);
    const parameters = view('parameters', 'float32', 32);
    const palette = view('palette', 'uint32', 64);
    const scalarOutput: GPUScalarScaleProps['output'] = {
      values: view('scalar', 'float32'),
      validity: view('validity', 'uint32')
    };
    const colorOutput: GPUColorScaleOutput = {
      colors: view('colors', 'uint32'),
      classIndices: view('classes', 'uint32')
    };
    const bivariateOutput: GPUBivariateClassificationOutput = {
      colors: view('joint-colors', 'uint32'),
      classIds: view('joint-classes', 'uint32')
    };
    const quantileOutput: GPUColumnQuantilesOutput = {
      quantiles: view('quantiles', 'float32', 1),
      validCount: view('count', 'uint32', 1),
      outputMask: view('filtered', 'uint32')
    };
    const scalar = new GPUScalarScale({
      values,
      domain,
      range: view('range', 'float32', 2),
      parameters,
      maximumDomainCount: 8,
      output: scalarOutput
    });
    const color = new GPUColorScale({
      values,
      domain,
      palette,
      parameters,
      maximumDomainCount: 9,
      maximumPaletteCount: 8,
      output: colorOutput
    });
    const bivariate = new GPUBivariateClassification({
      valuesX: values,
      valuesY: values,
      breaksX: domain,
      breaksY: domain,
      palette,
      parameters,
      maximumClassCount: 8,
      output: bivariateOutput
    });
    const quantile = new GPUColumnQuantiles({
      values,
      parameters,
      quantileCount: 1,
      output: quantileOutput
    });
    // These assignments are the legacy consumer pattern reported by the cross-stream reviewer.
    const scaled: GraphDataView<'float32'> = scalar.props.output.values;
    const validity: GraphDataView<'uint32'> | undefined = scalar.props.output.validity;
    const colors: GraphDataView<'uint32'> | undefined = color.props.output.colors;
    const classes: GraphDataView<'uint32'> | undefined = color.props.output.classIndices;
    const jointColors: GraphDataView<'uint32'> | undefined = bivariate.props.output.colors;
    const jointClasses: GraphDataView<'uint32'> | undefined = bivariate.props.output.classIds;
    const filtered: GraphDataView<'uint32'> | undefined = quantile.props.output.outputMask;
    expectTypeOf(scaled).toEqualTypeOf<GraphDataView<'float32'>>();
    for (const packed of [validity, colors, classes, jointColors, jointClasses, filtered])
      expectTypeOf(packed).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    const chunkedValues = createVectorView('chunked-values', 'float32', [values]);
    const chunkedOutput = createVectorView('chunked-output', 'float32', [
      view('chunked-scaled', 'float32')
    ]);
    const chunked = new GPUScalarScale({
      values: chunkedValues,
      domain,
      range: scalar.props.range,
      parameters,
      maximumDomainCount: 8,
      output: {values: chunkedOutput}
    });
    expectTypeOf(chunked.props.output.values).toEqualTypeOf<GraphVectorView<'float32'>>();
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('keeps optional packed outputs when never inputs infer the row type (provided: %s)', provided => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'uint32'>(name: string, format: Format, length = 4) =>
    createTransientView(graph, name, format, length);
  try {
    // Existing callers use this cast when their runtime-selected input format is a union.
    const values = view('values', 'float32') as never;
    const optionalView = (name: string): GraphDataView<'uint32'> | undefined =>
      provided ? view(name, 'uint32') : undefined;
    const parameters = view('parameters', 'float32', 32);
    const domain = view('domain', 'float32', 9);
    const palette = view('palette', 'uint32', 64);
    const scalar = new GPUScalarScale({
      values,
      mask: optionalView('scalar-mask'),
      domain,
      range: view('range', 'float32', 2),
      parameters,
      maximumDomainCount: 8,
      output: {values: view('scaled', 'float32'), validity: optionalView('validity')}
    });
    const color = new GPUColorScale({
      values,
      mask: optionalView('color-mask'),
      domain,
      palette,
      parameters,
      maximumDomainCount: 9,
      maximumPaletteCount: 8,
      output: {
        colors: optionalView('colors'),
        classIndices: optionalView('classes'),
        classCounts: view('color-counts', 'uint32', 8)
      }
    });
    const bivariate = new GPUBivariateClassification({
      valuesX: values,
      valuesY: view('values-y', 'float32'),
      mask: optionalView('joint-mask'),
      breaksX: domain,
      breaksY: domain,
      palette,
      parameters,
      maximumClassCount: 8,
      output: {
        colors: optionalView('joint-colors'),
        classIds: optionalView('joint-classes'),
        classCounts: view('joint-counts', 'uint32', 64)
      }
    });
    const quantile = new GPUColumnQuantiles({
      values,
      mask: optionalView('quantile-mask'),
      parameters,
      quantileCount: 1,
      output: {
        quantiles: view('quantile', 'float32', 1),
        validCount: view('count', 'uint32', 1),
        outputMask: optionalView('filtered')
      }
    });
    expectTypeOf<ClassificationRowView<'uint32', never>>().toEqualTypeOf<GraphDataView<'uint32'>>();
    expectTypeOf(scalar.props.output.values).toEqualTypeOf<GraphDataView<'float32'>>();
    expectTypeOf(scalar.props.output.validity).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    expectTypeOf(color.props.output.colors).toEqualTypeOf<GraphDataView<'uint32'> | undefined>();
    expectTypeOf(color.props.output.classIndices).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    expectTypeOf(bivariate.props.output.colors).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    expectTypeOf(bivariate.props.output.classIds).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    expectTypeOf(quantile.props.output.outputMask).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
  } finally {
    device.destroy();
  }
});
