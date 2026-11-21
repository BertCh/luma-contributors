// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUColorScaleOutput} from '../../../src/dataframe-analysis/column-classification/gpu-color-scale';
import type {GPUBivariateClassificationOutput} from '../../../src/dataframe-analysis/column-classification/gpu-bivariate-classification';

/** Establish a definite terminal after selecting a packed fixture's optional siblings. */
export function selectColorScaleOutput(output: Partial<GPUColorScaleOutput>): GPUColorScaleOutput {
  if (output.colors) return {...output, colors: output.colors};
  if (output.classIndices) return {...output, classIndices: output.classIndices};
  if (output.classCounts) return {...output, classCounts: output.classCounts};
  throw new Error('Color fixture must select at least one output');
}

/** Establish a definite terminal after selecting a packed fixture's optional siblings. */
export function selectBivariateOutput(
  output: Partial<GPUBivariateClassificationOutput>
): GPUBivariateClassificationOutput {
  if (output.colors) return {...output, colors: output.colors};
  if (output.classIds) return {...output, classIds: output.classIds};
  if (output.classCounts) return {...output, classCounts: output.classCounts};
  throw new Error('Bivariate fixture must select at least one output');
}
