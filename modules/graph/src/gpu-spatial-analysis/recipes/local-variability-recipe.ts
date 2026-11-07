// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {
  createRecipeKernelNode,
  getRecipeView as getOrCreateView,
  validateRecipeBoundary
} from './expansion-recipe-utils';
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {GPULocalSpatialHeteroscedasticity} from '../spatial-autocorrelation/gpu-local-variability';
import {GPULocalMoran} from '../spatial-autocorrelation/gpu-local-moran';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {RecipeBuilder, type GPURecipeResult} from './recipe-utils';

export type GPULocalVariabilityRecipeOutputs = {
  losh: GraphDataView<'float32'>;
  loshStatus: GraphDataView<'uint32'>;
  localI: GraphDataView<'float32'>;
  zScores: GraphDataView<'float32'>;
  pValues: GraphDataView<'float32'>;
  adjustedPValues: GraphDataView<'float32'>;
  quadrants: GraphDataView<'uint32'>;
  /** 0 analytic inference available, 1 statistic/inference undefined. */
  moranStatus: GraphDataView<'uint32'>;
};
export type GPULocalVariabilityRecipeProps = {
  id?: string;
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** Existing spatial-autocorrelation parameter layout; shared neighborhood and input mask. */
  parameters: GraphDataView<'float32'>;
  residualPower?: 1 | 2;
  minimumNeighbors?: number;
  /** Correction applies only to the Moran test family, never jointly with descriptive LOSH. */
  multipleTesting?: GPUMultipleTestingMethod;
  outputs?: Partial<GPULocalVariabilityRecipeOutputs>;
};
export type GPULocalVariabilityRecipeResult = GPURecipeResult<
  GPULocalVariabilityRecipeOutputs,
  {weights: GPUSpatialWeights; values: GraphDataView<'float32'>}
> & {
  inference: {
    moran: 'analytic-normal';
    losh: 'descriptive-only';
    multipleTesting: GPUMultipleTestingMethod;
    correctionFamily: 'moran-rows';
  };
};

/**
 * Compare local mean association with local dispersion on the same explicit graph and complete
 * input mask. LOSH is descriptive; Moran's analytic-normal p-values and quadrants remain separate.
 * Both exclude islands. No combined significance, dispersion class or permutation p-value is
 * inferred. Upstream overflow and stage-specific invalid counts are retained, with per-row codes.
 */
export function addLocalVariabilityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPULocalVariabilityRecipeProps
): GPULocalVariabilityRecipeResult {
  const id = props.id ?? 'local-variability';
  const rows = props.values.length;
  const supplied = props.outputs;
  const floatView = (name: 'losh' | 'localI' | 'zScores' | 'pValues' | 'adjustedPValues') =>
    getOrCreateView(graph, `${id}-${name}`, 'float32', rows, supplied?.[name]);
  const integerView = (name: 'loshStatus' | 'quadrants' | 'moranStatus') =>
    getOrCreateView(graph, `${id}-${name}`, 'uint32', rows, supplied?.[name]);
  const outputs: GPULocalVariabilityRecipeOutputs = {
    losh: floatView('losh'),
    loshStatus: integerView('loshStatus'),
    localI: floatView('localI'),
    zScores: floatView('zScores'),
    pValues: floatView('pValues'),
    adjustedPValues: floatView('adjustedPValues'),
    quadrants: integerView('quadrants'),
    moranStatus: integerView('moranStatus')
  };
  validateRecipeBoundary(
    graph,
    id,
    [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.weights.status?.overflow,
      props.values,
      props.mask,
      props.parameters
    ],
    Object.values(outputs)
  );
  const builder = new RecipeBuilder(graph);
  builder.add(
    new GPULocalSpatialHeteroscedasticity({
      id: `${id}-losh`,
      weights: props.weights,
      values: props.values,
      mask: props.mask,
      residualPower: props.residualPower,
      output: {statistic: outputs.losh, status: outputs.loshStatus}
    })
  );
  builder.add(
    new GPULocalMoran({
      id: `${id}-moran`,
      weights: props.weights,
      values: props.values,
      mask: props.mask,
      parameters: props.parameters,
      islands: 'exclude',
      minimumNeighbors: props.minimumNeighbors,
      multipleTesting: props.multipleTesting,
      output: {
        localI: outputs.localI,
        zScores: outputs.zScores,
        pValues: outputs.pValues,
        adjustedPValues: outputs.adjustedPValues,
        quadrants: outputs.quadrants
      }
    })
  );
  const loshInvalidCount = getOrCreateView(graph, `${id}-losh-invalid`, 'uint32', 1);
  const moranInvalidCount = getOrCreateView(graph, `${id}-moran-invalid`, 'uint32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-status`,
      operation: 'GPULocalVariabilityRecipe',
      variant: 'status',
      invocationCount: 1,
      bindings: [
        {name: 'loshStatus', view: outputs.loshStatus, type: 'u32', access: 'read'},
        {name: 'zScores', view: outputs.zScores, type: 'f32', access: 'read'},
        {name: 'moranStatus', view: outputs.moranStatus, type: 'u32', access: 'read_write'},
        {name: 'loshInvalidCount', view: loshInvalidCount, type: 'u32', access: 'read_write'},
        {name: 'moranInvalidCount', view: moranInvalidCount, type: 'u32', access: 'read_write'}
      ],
      declarations: IS_FINITE_BITS_WGSL,
      body: `var loshInvalid = 0u; var moranInvalid = 0u;
for (var row = 0u; row < ${rows}u; row++) {
  loshInvalid += select(0u, 1u, loshStatus[loshStatusOffset + row] != 0u);
  let code = select(1u, 0u, isFiniteBits(zScores[zScoresOffset + row]));
  moranStatus[moranStatusOffset + row] = code; moranInvalid += code;
}
loshInvalidCount[loshInvalidCountOffset] = loshInvalid; moranInvalidCount[moranInvalidCountOffset] = moranInvalid;`
    })
  );
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {weights: props.weights, values: props.values},
    status: {
      stages: [
        {stage: 'weights', status: props.weights.status ?? {}},
        {stage: 'losh', status: {invalidCount: loshInvalidCount}},
        {stage: 'local-moran', status: {invalidCount: moranInvalidCount}}
      ]
    },
    inference: {
      moran: 'analytic-normal',
      losh: 'descriptive-only',
      multipleTesting: props.multipleTesting ?? 'none',
      correctionFamily: 'moran-rows'
    }
  };
}
