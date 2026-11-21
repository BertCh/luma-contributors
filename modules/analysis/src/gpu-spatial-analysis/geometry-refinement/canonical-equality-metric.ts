// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateOption} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GEOMETRY_EQUALITY_DISTANCE_WGSL} from '../geometry-predicates/geometry-equality-distance-wgsl';

export function validateCanonicalEqualityMetric(id: string, metric: string | undefined): void {
  validateOption(id, 'toleranceMetric', metric === undefined ? 'per-coordinate' : metric, [
    'per-coordinate',
    'euclidean'
  ]);
}

export function getCanonicalEqualityMetricWGSL(
  tolerance: number,
  metric: string | undefined
): string {
  return `${GEOMETRY_EQUALITY_DISTANCE_WGSL}
fn withinTolerance(left:vec2f,right:vec2f)->bool {
  let tolerance=${getWGSLFloatLiteral(tolerance)};
  ${
    metric === 'euclidean'
      ? 'return geometryWithinTolerance(left,right,tolerance);'
      : 'let delta=abs(left-right);let magnitude=max(delta.x,delta.y);\n  if(magnitude>tolerance) {return false;}\n  return true;'
  }
}`;
}
