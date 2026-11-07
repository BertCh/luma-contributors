// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {validatePackedView} from '../../utils/gpu-contributor-utils';
import {GPUColumnQuantiles} from '../../gpu-dataframe/column-classification/gpu-column-quantiles';
import {
  GPULocalPermutationTest,
  type GPULocalPermutationTestOutput
} from '../permutation-inference/gpu-local-permutation-test';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  wrapGPUSpatialWeightsChunkBody,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {
  createRecipeKernelNode,
  getRecipeView,
  validateRecipeBoundary
} from './expansion-recipe-utils';

/** Quantile LISA labels, including explicit missing and isolated rows. */
export const GPU_QUANTILE_LISA_CLASS = {
  notSignificant: 0,
  significant: 1,
  undefined: 2,
  isolated: 3
} as const;

export type GPUQuantileLISAOutput = Omit<
  GPULocalPermutationTestOutput,
  'classes' | 'quadrants' | 'adjustedPValues' | 'permutationCounts'
> & {
  /** Selected-class indicator, zero for missing rows. */
  membership: GraphDataView<'uint32'>;
  /** Type-5 lower and upper selected-quantile boundaries. Exterior bounds are infinities. */
  boundaries: GraphDataView<'float32'>;
  /** Count of finite unmasked source rows, before island removal. */
  validCount: GraphDataView<'uint32'>;
  /** GPU_QUANTILE_LISA_CLASS code, gated by the inference significance level. */
  classes: GraphDataView<'uint32'>;
  /** One when k/q is invalid; independent of inference capacity overflow. */
  invalidQuantile: GraphDataView<'uint32'>;
};

export type GPUQuantileLISAProps = {
  id?: string;
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Resident uint32 [numberOfQuantiles, selectedQuantile], with 2 <= k < rows and 1 <= q <= k. */
  quantileParameters: GraphDataView<'uint32'>;
  /** Resident permutation parameters from getGPUPermutationParameterValues. */
  parameters: GraphDataView<'uint32'>;
  maximumPermutations: number;
  maximumNeighbors?: number;
  output: GPUQuantileLISAOutput;
};

/**
 * Selected-quantile conditional local join counts. Quantile boundaries follow pinned libgeoda
 * gda_quantilelisa: type-5 percentiles and lower-inclusive, upper-exclusive membership.
 * Missing values are omitted before quantiling (the pinned upstream missing-value defect is
 * deliberately corrected). Uses the legacy directed pseudo p-value for GeoDa parity; it is
 * anti-conservative and does not provide a two-sided error-control guarantee.
 * Inputs are packed resident views; chunked inputs must be packed explicitly by the caller.
 * No submission, readback or input ownership transfer occurs.
 */
export class GPUQuantileLISA implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUQuantileLISAProps) {
    this.id = props.id ?? 'quantile-lisa';
    validatePackedView(props.values, ['float32'], this.id);
    validatePackedView(props.quantileParameters, ['uint32'], this.id);
    if (props.mask) validatePackedView(props.mask, ['uint32'], this.id);
    for (const key of ['membership', 'classes', 'validCount', 'invalidQuantile'] as const)
      validatePackedView(props.output[key], ['uint32'], this.id);
    validatePackedView(props.output.boundaries, ['float32'], this.id);
    // Quantile settings hold exactly the two resident control words.
    assert(props.quantileParameters.format === 'uint32' && props.quantileParameters.length === 2);
    assert(props.values.length > 2);
    assert(!props.mask || props.mask.length === props.values.length);
    const rows = props.values.length;
    for (const key of ['membership', 'classes'] as const) assert(props.output[key].length === rows);
    assert(props.output.boundaries.length === 2);
    assert(props.output.validCount.length === 1 && props.output.invalidQuantile.length === 1);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = props.values.length;
    const output = props.output;
    validateRecipeBoundary(
      graph,
      id,
      [
        props.values,
        props.mask,
        props.quantileParameters,
        props.parameters,
        ...getGPUSpatialWeightsOrChunksViews(props.weights),
        props.weights.status?.overflow
      ],
      [
        output.membership,
        output.boundaries,
        output.validCount,
        output.classes,
        output.invalidQuantile,
        output.exceedances,
        output.pseudoPValues,
        output.observed,
        output.significant,
        output.status.overflow
      ]
    );
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => getRecipeView(graph, `${id}-${name}`, format, length);
    const countParameters = view('count-parameters', 'float32', 5);
    const boundaryParameters = view('boundary-parameters', 'float32', 6);
    const unusedQuantile = view('count-quantile', 'float32', 1);
    const unusedCount = view('boundary-count', 'uint32', 1);
    const rawBoundaries = view('raw-boundaries', 'float32', 2);
    const validMask = view('valid-mask', 'uint32', rows);
    const observed = output.observed ?? view('observed', 'float32', rows);
    const significant = output.significant ?? view('significant', 'uint32', rows);
    const upperExceedances = view('upper-exceedances', 'uint32', rows);
    const unusedPValues = view('upper-p-values', 'float32', rows);
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createRecipeKernelNode(graph, {
        id: `${id}-count-parameters`,
        operation: 'GPUQuantileLISA',
        variant: 'count-parameters',
        invocationCount: 1,
        bindings: [
          {name: 'outputParameters', view: countParameters, type: 'f32', access: 'read_write'}
        ],
        body: `outputParameters[outputParametersOffset] = 3.0;
  outputParameters[outputParametersOffset + 1u] = 0.0;
  outputParameters[outputParametersOffset + 2u] = 1.0;
  outputParameters[outputParametersOffset + 3u] = 0.0;
  outputParameters[outputParametersOffset + 4u] = 0.0;`
      })
    );
    nodes.push(
      ...new GPUColumnQuantiles({
        id: `${id}-count`,
        values: props.values,
        mask: props.mask,
        parameters: countParameters,
        quantileCount: 1,
        output: {quantiles: unusedQuantile, validCount: output.validCount}
      }).getCommandNodes(graph)
    );
    nodes.push(
      createRecipeKernelNode(graph, {
        id: `${id}-boundary-parameters`,
        operation: 'GPUQuantileLISA',
        variant: 'type-five-probabilities',
        declarations: GET_NAN_WGSL,
        invocationCount: 1,
        bindings: [
          {name: 'settings', view: props.quantileParameters, type: 'u32', access: 'read'},
          {name: 'count', view: output.validCount, type: 'u32', access: 'read'},
          {name: 'outputParameters', view: boundaryParameters, type: 'f32', access: 'read_write'},
          {name: 'invalid', view: output.invalidQuantile, type: 'u32', access: 'read_write'}
        ],
        body: `let quantileCount = settings[settingsOffset];
  let selectedQuantile = settings[settingsOffset + 1u];
  let valid = quantileCount >= 2u && quantileCount < ${rows}u && selectedQuantile >= 1u && selectedQuantile <= quantileCount;
  invalid[invalidOffset] = select(1u, 0u, valid);
  outputParameters[outputParametersOffset] = 3.0;
  outputParameters[outputParametersOffset + 1u] = 0.0;
  outputParameters[outputParametersOffset + 2u] = 1.0;
  outputParameters[outputParametersOffset + 3u] = 0.0;
  let countValue = f32(count[countOffset]);
  for (var boundary = 0u; boundary < 2u; boundary++) {
    // Type 5 h = n*p - 0.5, expressed as a type 7 probability for resident radix selection.
    let probability = f32(selectedQuantile - 1u + boundary) / f32(max(quantileCount, 1u));
    let rank = clamp(countValue * probability - 0.5, 0.0, max(countValue - 1.0, 0.0));
    outputParameters[outputParametersOffset + 4u + boundary] = select(getNaN(), rank / max(countValue - 1.0, 1.0), valid);
  }`
      })
    );
    nodes.push(
      ...new GPUColumnQuantiles({
        id: `${id}-boundaries`,
        values: props.values,
        mask: props.mask,
        parameters: boundaryParameters,
        quantileCount: 2,
        output: {quantiles: rawBoundaries, validCount: unusedCount}
      }).getCommandNodes(graph)
    );
    nodes.push(
      createRecipeKernelNode(graph, {
        id: `${id}-membership`,
        operation: 'GPUQuantileLISA',
        variant: 'lower-inclusive',
        declarations: IS_FINITE_BITS_WGSL,
        invocationCount: rows,
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'settings', view: props.quantileParameters, type: 'u32', access: 'read'},
          {name: 'invalid', view: output.invalidQuantile, type: 'u32', access: 'read'},
          {name: 'rawBoundaries', view: rawBoundaries, type: 'f32', access: 'read'},
          {name: 'boundaries', view: output.boundaries, type: 'f32', access: 'read_write'},
          {name: 'membership', view: output.membership, type: 'u32', access: 'read_write'},
          {name: 'validMask', view: validMask, type: 'u32', access: 'read_write'}
        ],
        body: `let value = values[valuesOffset + index];
  let valid = isFiniteBits(value) ${props.mask ? '&& mask[maskOffset + index] != 0u' : ''} && invalid[invalidOffset] == 0u;
  var infinityBits = 0x7f800000u;
  let lower = select(rawBoundaries[rawBoundariesOffset], bitcast<f32>(infinityBits | 0x80000000u), settings[settingsOffset + 1u] == 1u);
  let upper = select(rawBoundaries[rawBoundariesOffset + 1u], bitcast<f32>(infinityBits), settings[settingsOffset + 1u] == settings[settingsOffset]);
  if (index == 0u) { boundaries[boundariesOffset] = lower; boundaries[boundariesOffset + 1u] = upper; }
  validMask[validMaskOffset + index] = select(0u, 1u, valid);
  membership[membershipOffset + index] = select(0u, 1u, valid && value >= lower && value < upper);`
      })
    );
    nodes.push(
      ...new GPULocalPermutationTest({
        id: `${id}-inference`,
        weights: props.weights,
        values: output.membership,
        mask: validMask,
        parameters: props.parameters,
        statistic: 'localJoinCount',
        alternative: 'greater',
        islands: 'exclude',
        maximumPermutations: props.maximumPermutations,
        maximumNeighbors: props.maximumNeighbors,
        output: {
          exceedances: upperExceedances,
          pseudoPValues: unusedPValues,
          observed,
          islandCount: output.islandCount,
          status: output.status
        }
      }).getCommandNodes(graph)
    );
    // GeoDa uses P-g for the strict lower tail; shared 'directed' deliberately includes ties.
    nodes.push(
      createRecipeKernelNode(graph, {
        id: `${id}-geoda-tail`,
        operation: 'GPUQuantileLISA',
        variant: 'strict-complement-tail',
        invocationCount: rows,
        declarations: GET_NAN_WGSL,
        bindings: [
          {name: 'upperExceedances', view: upperExceedances, type: 'u32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'exceedances', view: output.exceedances, type: 'u32', access: 'read_write'},
          {name: 'pseudoPValues', view: output.pseudoPValues, type: 'f32', access: 'read_write'},
          {name: 'significant', view: significant, type: 'u32', access: 'read_write'}
        ],
        body: `let upperCount = upperExceedances[upperExceedancesOffset + index];
  let permutationCount = parameters[parametersOffset + 2u];
  var count = 0xffffffffu;
  var probability = getNaN();
  var isSignificant = 0u;
  if (upperCount != 0xffffffffu) {
    count = min(upperCount, permutationCount - upperCount);
    probability = f32(count + 1u) / f32(permutationCount + 1u);
    let level = bitcast<f32>(parameters[parametersOffset + 3u]);
    isSignificant = select(0u, 1u, f32(count + 1u) <= level * f32(permutationCount + 1u));
  }
  exceedances[exceedancesOffset + index] = count;
  pseudoPValues[pseudoPValuesOffset + index] = probability;
  significant[significantOffset + index] = isSignificant;`
      })
    );
    nodes.push(
      ...getGPUSpatialWeightsChunkDispatches(props.weights).map(dispatch => {
        const localIndex = dispatch.isWhole ? 'index' : '(index - WEIGHTS_ROW_START)';
        return createRecipeKernelNode(graph, {
          id: `${id}-classes${dispatch.idSuffix}`,
          operation: 'GPUQuantileLISA',
          variant: 'classes',
          declarations: `${IS_FINITE_BITS_WGSL}\n${dispatch.constantsWGSL}`,
          invocationCount: dispatch.rowCount,
          bindings: [
            {name: 'validMask', view: validMask, type: 'u32', access: 'read'},
            {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
            {name: 'significant', view: significant, type: 'u32', access: 'read'},
            {name: 'classes', view: output.classes, type: 'u32', access: 'read_write'}
          ],
          body: wrapGPUSpatialWeightsChunkBody(
            dispatch,
            `var classification = select(0u, 1u, significant[significantOffset + index] != 0u);
  if (validMask[validMaskOffset + index] == 0u) { classification = 2u; }
  else {
    var neighborCount = 0u;
    for (var slot = offsets[offsetsOffset + ${localIndex}]; slot < min(offsets[offsetsOffset + ${localIndex} + 1u], ${dispatch.weights.neighbors.length}u); slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      let weight = weights[weightsOffset + slot];
      if (neighbor < ${rows}u && neighbor != index && weight > 0.0 && isFiniteBits(weight)) {
        if (validMask[validMaskOffset + neighbor] != 0u) { neighborCount++; }
      }
    }
    if (neighborCount == 0u) { classification = 3u; }
  }
  classes[classesOffset + index] = classification;`
          )
        });
      })
    );
    return nodes;
  }
}
