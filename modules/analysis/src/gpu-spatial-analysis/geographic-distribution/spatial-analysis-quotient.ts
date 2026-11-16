// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** Finite spatial ratios use exponent arithmetic; existing missing/zero policies stay with callers.
 * Include the shared isFiniteValue declaration once in the containing shader. @internal */
export const SPATIAL_ANALYSIS_QUOTIENT_WGSL = /* wgsl */ `
${SCALED_QUOTIENT_WGSL}
fn getSpatialAnalysisQuotient(value:f32,denominator:f32)->f32 {
  if(isFiniteValue(value)&&isFiniteValue(denominator)&&denominator!=0.0){
    return getScaledQuotient(value,denominator,1.0,0);
  }
  return value/denominator;
}`;
