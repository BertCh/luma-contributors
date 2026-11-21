// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUTerrainCurvatureKind,
  GPUTerrainCurvatureOutput
} from '@luma-contributors/analysis/gpu-terrain';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';

/** Each branch carries a required named leaf, preserving the nonempty terminal contract. */
export function createSingleCurvatureOutput(
  kind: GPUTerrainCurvatureKind,
  output: GraphDataView<'float32'>
): GPUTerrainCurvatureOutput {
  const selections: Record<GPUTerrainCurvatureKind, GPUTerrainCurvatureOutput> = {
    profile: {curvatures: {profile: output}},
    plan: {curvatures: {plan: output}},
    tangential: {curvatures: {tangential: output}},
    mean: {curvatures: {mean: output}},
    gaussian: {curvatures: {gaussian: output}},
    minimal: {curvatures: {minimal: output}},
    maximal: {curvatures: {maximal: output}},
    unsphericity: {curvatures: {unsphericity: output}},
    difference: {curvatures: {difference: output}},
    'horizontal-excess': {curvatures: {'horizontal-excess': output}},
    'vertical-excess': {curvatures: {'vertical-excess': output}},
    accumulation: {curvatures: {accumulation: output}},
    ring: {curvatures: {ring: output}},
    rotor: {curvatures: {rotor: output}},
    laplacian: {curvatures: {laplacian: output}}
  };
  validateRequirement(
    'single-curvature-output',
    Object.hasOwn(selections, kind),
    `kind must be one of ${Object.keys(selections).join(', ')}; received ${String(kind)}`
  );
  return selections[kind];
}
