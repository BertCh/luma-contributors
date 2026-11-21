// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validatePermutationInputBindings} from '../permutation-inference/permutation-inference-kernels';
import {PERMUTATION_SIGNIFICANCE_WGSL} from '../permutation-inference/permutation-significance';
import type {GPUQuantileLISAAlternative} from '../../analysis-core/contracts/inference';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotWeightsDescription} from '../spatial-weights/weights-construction-snapshot';
import {snapshotSpatialAutocorrelationProps} from './spatial-autocorrelation-construction';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';
import {ORDER_STATISTIC_WGSL} from '../../dataframe-analysis/column-classification/order-statistic-wgsl';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUColumnQuantiles} from '../../dataframe-analysis/column-classification/gpu-column-quantiles';
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
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

// GeoDa requires 2 <= quantile count < source row count.
const MINIMUM_ROW_COUNT = 3;
// Shared permutation ranks and type-5 ranks must remain exact float32 integers.
const MAXIMUM_ROW_COUNT = 2 ** 24 - 1;
// Bound graph compilation cost for a multivariate request (two quantile passes per variable).
const MAXIMUM_VARIABLE_COUNT = 64;
// Two ten-node radix selections plus three glue nodes; leave room for primitive evolution.
const VARIABLE_COMMAND_NODE_ALLOWANCE = 32;
// Upper bound for scan/reduction lowering at MAXIMUM_ROW_COUNT, plus per-chunk front-end kernels.
const INFERENCE_COMMAND_NODE_ALLOWANCE = 1024;
const CHUNK_COMMAND_NODE_ALLOWANCE = 16;

/** An additional variable with its own resident [numberOfQuantiles, selectedQuantile]. */
export type GPUQuantileLISAVariable = {
  values: GraphDataView<'float32'>;
  quantileParameters: GraphDataView<'uint32'>;
};

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
  /** One for rows actually tested; zero for missing, isolated or suppressed rows. */
  tested?: GraphDataView<'uint32'>;
  /** Intersection of every selected-class indicator, zero for missing rows. */
  membership: GraphDataView<'uint32'>;
  /** Type-5 [lower, upper] pairs in variable order. Exterior bounds are infinities. */
  boundaries: GraphDataView<'float32'>;
  /** Count of finite unmasked source rows per variable, before island removal. */
  validCount: GraphDataView<'uint32'>;
  /** GPU_QUANTILE_LISA_CLASS code, gated by the inference significance level. */
  classes: GraphDataView<'uint32'>;
  /** One when k/q is invalid; independent of inference capacity overflow. */
  invalidQuantile: GraphDataView<'uint32'>;
};

export type GPUQuantileLISAProps = {
  id?: string;
  values: GraphDataView<'float32'>;
  /** Additional variables; the primary values/quantileParameters pair is first. At most 64 total. */
  additionalVariables?: readonly GPUQuantileLISAVariable[];
  /** Common row mask applied independently to every variable before quantiling. */
  mask?: GraphDataView<'uint32'>;
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Resident uint32 [numberOfQuantiles, selectedQuantile], with 2 <= k < rows and 1 <= q <= k. */
  quantileParameters: GraphDataView<'uint32'>;
  /** Resident permutation parameters from getGPUPermutationParameterValues. */
  parameters: GraphDataView<'uint32'>;
  /** Explicit inference rule. Inclusive tails retain tied draws; legacy preserves the former implicit rule. */
  alternative: GPUQuantileLISAAlternative;
  maximumPermutations: number;
  maximumNeighbors?: number;
  output: GPUQuantileLISAOutput;
};

/**
 * Univariate or multivariate selected-quantile conditional local join counts. Quantile boundaries follow pinned libgeoda
 * gda_quantilelisa: type-5 percentiles and lower-inclusive, upper-exclusive membership.
 * Missing values are omitted before quantiling (the pinned upstream missing-value defect is
 * deliberately corrected). Requires 'greater', 'two-sided' (inclusive ties) or 'geoda-legacy'
 * (the former implicit, anti-conservative strict-complement rule). Optional output.tested distinguishes
 * untested rows without changing existing class codes.
 * Quantile each variable independently, then intersect the indicators and finite-row masks.
 * An empty intersection stays empty; no bivariate no-colocation fallback is applied.
 * Per-frame: values, common mask, quantile settings, weights, permutation parameters.
 * Compile-time: variable count, view layouts, alternative, maximumPermutations and maximumNeighbors.
 * The local inference route supports at most MAXIMUM_PERMUTATION_COUNT = 1,048,576 draws per submission.
 * Inputs are packed resident views; chunked inputs must be packed explicitly by the caller.
 * The largest row-proportional inference binding is 4 * (rows + 1) bytes for row positions;
 * the eight preparation columns use separate 4 * rows bindings. Source ranks cap rows at 16,777,215.
 * No submission, readback or input ownership transfer occurs.
 */
export class GPUQuantileLISA implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly variables: readonly GPUQuantileLISAVariable[];
  constructor(readonly props: GPUQuantileLISAProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'values',
      'additionalVariables',
      'mask',
      'weights',
      'quantileParameters',
      'parameters',
      'maximumPermutations',
      'maximumNeighbors',
      'alternative',
      'output'
    ]);
    const capturedOutput = snapshotConstructionFields(props.output, [
      'analysisCount',
      'islandCount',
      'exceedances',
      'pseudoPValues',
      'observed',
      'significant',
      'status',
      'membership',
      'boundaries',
      'validCount',
      'classes',
      'invalidQuantile',
      'tested'
    ]);
    props = Object.freeze({...props, output: capturedOutput});
    const tested = props.output.tested;
    const status = snapshotConstructionFields(props.output.status, [
      'overflow',
      'invalidCount',
      'weightsOverflow',
      'candidateOverflow',
      'neighborOverflow',
      'invalidInput'
    ]);
    const weights = snapshotWeightsDescription(props.weights);
    props = snapshotSpatialAutocorrelationProps({...props, output: {...props.output, status}});
    props = Object.freeze({
      ...props,
      weights,
      output: Object.freeze({...props.output, tested, status})
    });
    this.props = props;
    this.id = props.id ?? 'quantile-lisa';
    validateRequirement(
      this.id,
      ['greater', 'two-sided', 'geoda-legacy'].includes(props.alternative),
      `alternative must be 'greater', 'two-sided' or 'geoda-legacy'; received ${String(props.alternative)}`
    );
    if (tested) {
      validateSinglePackedView(tested, ['uint32'], `${this.id} output.tested`);
      validateViewLength(this.id, 'output.tested', tested, props.values.length);
    }
    this.variables = [props, ...(props.additionalVariables ?? [])];
    validateCapacity(this.id, 'variable count', this.variables.length, 1, MAXIMUM_VARIABLE_COUNT);
    const rows = props.values.length;
    validateCapacity(this.id, 'values length', rows, MINIMUM_ROW_COUNT, MAXIMUM_ROW_COUNT);
    this.estimatedCommandNodeCount =
      VARIABLE_COMMAND_NODE_ALLOWANCE * this.variables.length +
      INFERENCE_COMMAND_NODE_ALLOWANCE +
      CHUNK_COMMAND_NODE_ALLOWANCE * getGPUSpatialWeightsChunkDispatches(props.weights).length;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    for (const [variableIndex, variable] of this.variables.entries()) {
      validateSinglePackedView(variable.values, ['float32'], this.id);
      validateSinglePackedView(variable.quantileParameters, ['uint32'], this.id);
      validateViewLength(this.id, `variables[${variableIndex}].values`, variable.values, rows);
      validateViewLength(
        this.id,
        `variables[${variableIndex}].quantileParameters`,
        variable.quantileParameters,
        2
      );
    }
    if (props.mask) validateSinglePackedView(props.mask, ['uint32'], this.id);
    for (const key of ['membership', 'classes', 'validCount', 'invalidQuantile'] as const)
      validateSinglePackedView(props.output[key], ['uint32'], this.id);
    validateSinglePackedView(props.output.boundaries, ['float32'], this.id);
    if (props.mask) validateViewLength(this.id, 'mask', props.mask, rows);
    for (const key of ['membership', 'classes'] as const)
      validateViewLength(this.id, `output.${key}`, props.output[key], rows);
    // Boundaries hold the lower and upper bound of the selected quantile.
    validateViewLength(
      this.id,
      'output.boundaries',
      props.output.boundaries,
      2 * this.variables.length
    );
    validateViewLength(
      this.id,
      'output.validCount',
      props.output.validCount,
      this.variables.length
    );
    validateViewLength(this.id, 'output.invalidQuantile', props.output.invalidQuantile, 1);
    // Validate the composed inference contract at construction, before graph nodes exist.
    new GPULocalPermutationTest({
      ...props,
      id: this.id,
      statistic: 'localJoinCount',
      alternative: props.alternative === 'geoda-legacy' ? 'greater' : props.alternative,
      output: {
        analysisCount: props.output.analysisCount,
        exceedances: props.output.exceedances,
        pseudoPValues: props.output.pseudoPValues,
        observed: props.output.observed,
        significant: props.output.significant,
        islandCount: props.output.islandCount,
        status: props.output.status
      }
    });
    const outputs = [
      props.output.analysisCount,
      props.output.membership,
      props.output.boundaries,
      props.output.validCount,
      props.output.classes,
      props.output.invalidQuantile,
      props.output.exceedances,
      props.output.pseudoPValues,
      props.output.observed,
      props.output.significant,
      props.output.islandCount,
      ...Object.values(props.output.status),
      tested
    ];
    validateGraphOutputsDisjointFromInputs(this.id, outputs, [
      ...this.variables.flatMap(variable => [variable.values, variable.quantileParameters]),
      props.mask,
      props.parameters,
      ...getGPUSpatialWeightsOrChunksViews(props.weights, true),
      props.weights.status?.overflow
    ]);
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    validateRequirement(
      this.id,
      new Set(outputBuffers).size === outputBuffers.length,
      'outputs must not share a buffer'
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = props.values.length;
    const output = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      ...this.variables.flatMap(variable => [variable.values, variable.quantileParameters]),
      props.mask,
      props.quantileParameters,
      props.parameters,
      ...getGPUSpatialWeightsOrChunksViews(props.weights, true),
      props.weights.status?.overflow,
      output.membership,
      output.analysisCount,
      output.boundaries,
      output.validCount,
      output.classes,
      output.invalidQuantile,
      output.exceedances,
      output.pseudoPValues,
      output.observed,
      output.significant,
      ...Object.values(output.status),
      output.tested,
      output.islandCount
    ]);
    // Preflight the actual split-column preparation and all borrowed bindings before scratch.
    validatePermutationInputBindings(
      graph,
      {
        ...props,
        id,
        operation: 'GPUQuantileLISA',
        analysisCount: output.analysisCount,
        islandCount: output.islandCount
      },
      [
        ...this.variables.flatMap(variable => [variable.values, variable.quantileParameters]),
        output.membership,
        output.boundaries,
        output.validCount,
        output.classes,
        output.invalidQuantile,
        output.exceedances,
        output.pseudoPValues,
        output.observed,
        output.significant,
        output.tested,
        ...Object.values(output.status)
      ]
    );
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const validMask = view('valid-mask', 'uint32', rows);
    const observed = output.observed ?? view('observed', 'float32', rows);
    const significant = output.significant ?? view('significant', 'uint32', rows);
    const upperExceedances = view('upper-exceedances', 'uint32', rows);
    const unusedPValues = view('upper-p-values', 'float32', rows);
    const nodes: GPUCommandNode<Parameters>[] = [];
    for (const [variableIndex, variable] of this.variables.entries()) {
      const variableId = variableIndex === 0 ? id : `${id}-variable-${variableIndex}`;
      const countParameters = view(`count-parameters-${variableIndex}`, 'float32', 5);
      const boundaryParameters = view(`boundary-parameters-${variableIndex}`, 'float32', 8);
      const boundaryRanks = view(`boundary-ranks-${variableIndex}`, 'uint32', 8);
      const unusedQuantile = view(`count-quantile-${variableIndex}`, 'float32', 1);
      const unusedCount = view(`boundary-count-${variableIndex}`, 'uint32', 1);
      const variableCount = view(`valid-count-${variableIndex}`, 'uint32', 1);
      // Four exact order statistics followed by the two type-5 interpolation fractions.
      const rawBoundaries = view(`raw-boundaries-${variableIndex}`, 'float32', 6);
      const selectedBoundaries = graph.createDataView(rawBoundaries.buffer, {
        format: 'float32',
        byteOffset: rawBoundaries.byteOffset,
        length: 4
      });
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${variableId}-count-parameters`,
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
          id: `${variableId}-count`,
          values: variable.values,
          mask: props.mask,
          parameters: countParameters,
          quantileCount: 1,
          output: {quantiles: unusedQuantile, validCount: variableCount}
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${variableId}-boundary-parameters`,
          operation: 'GPUQuantileLISA',
          variant: 'type-five-probabilities',
          declarations: `${GET_NAN_WGSL}\n${UINT64_WGSL}`,
          invocationCount: 1,
          bindings: [
            {name: 'settings', view: variable.quantileParameters, type: 'u32', access: 'read'},
            {name: 'count', view: variableCount, type: 'u32', access: 'read'},
            {name: 'counts', view: output.validCount, type: 'u32', access: 'read_write'},
            {name: 'outputParameters', view: boundaryParameters, type: 'f32', access: 'read_write'},
            {name: 'ranks', view: boundaryRanks, type: 'u32', access: 'read_write'},
            {name: 'rawBoundaries', view: rawBoundaries, type: 'f32', access: 'read_write'},
            {name: 'invalid', view: output.invalidQuantile, type: 'u32', access: 'read_write'}
          ],
          body: `let quantileCount = settings[settingsOffset];
  let selectedQuantile = settings[settingsOffset + 1u];
  let valid = quantileCount >= 2u && quantileCount < ${rows}u && selectedQuantile >= 1u && selectedQuantile <= quantileCount;
  invalid[invalidOffset] = ${variableIndex === 0 ? '0u' : 'invalid[invalidOffset]'} | select(1u, 0u, valid);
  counts[countsOffset + ${variableIndex}u] = count[countOffset];
  outputParameters[outputParametersOffset] = 3.0;
  outputParameters[outputParametersOffset + 1u] = 0.0;
  outputParameters[outputParametersOffset + 2u] = 1.0;
  outputParameters[outputParametersOffset + 3u] = 0.0;
  let countValue = count[countOffset];
  // h = (2*n*k-q)/(2*q). Compute its integer part/remainder without a float probability
  // round trip, then select floor(h) and ceil(h) through exact rational type-7 ranks.
  ${[0, 1]
    .map(
      boundary => `{
    let boundary = ${boundary}u;
    var lower = 0u; var upper = 0u; var fraction = getNaN(); var denominator = 0u;
    if (valid && countValue > 0u) {
      let term = multiplyUint32Wide(countValue, 2u * (selectedQuantile - 1u + boundary));
      denominator = max(countValue - 1u, 1u); fraction = 0.0;
      if (compareUint64(term, vec2<u32>(0u, quantileCount)) > 0i) {
        let rank = divideUint64ByUint32(subtractUint64(term, vec2<u32>(0u, quantileCount)), 2u * quantileCount);
        lower = min(rank.quotient.y, countValue - 1u); upper = min(lower + u32(rank.remainder != 0u), countValue - 1u);
        if (upper != lower) { fraction = f32(rank.remainder) / f32(2u * quantileCount); }
      }
    }
    let base = ranksOffset + boundary * 4u;
    ranks[base] = lower; ranks[base + 1u] = denominator; ranks[base + 2u] = upper; ranks[base + 3u] = denominator;
    rawBoundaries[rawBoundariesOffset + 4u + boundary] = fraction;
    outputParameters[outputParametersOffset + 4u + 2u * boundary] = 0.0;
    outputParameters[outputParametersOffset + 5u + 2u * boundary] = 0.0;
  }`
    )
    .join('\n')}`
        })
      );
      nodes.push(
        ...new GPUColumnQuantiles({
          id: `${variableId}-boundaries`,
          values: variable.values,
          mask: props.mask,
          parameters: boundaryParameters,
          quantileCount: 4,
          rationalProbabilities: boundaryRanks,
          output: {quantiles: selectedBoundaries, validCount: unusedCount}
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${variableId}-membership`,
          operation: 'GPUQuantileLISA',
          variant: 'lower-inclusive',
          declarations: `${ORDER_STATISTIC_WGSL}\n${IS_FINITE_BITS_WGSL}`,
          invocationCount: rows,
          bindings: [
            {name: 'values', view: variable.values, type: 'f32', access: 'read'},
            ...(props.mask
              ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
              : []),
            {name: 'settings', view: variable.quantileParameters, type: 'u32', access: 'read'},
            {name: 'invalid', view: output.invalidQuantile, type: 'u32', access: 'read'},
            {name: 'rawBoundaries', view: rawBoundaries, type: 'f32', access: 'read'},
            {name: 'boundaries', view: output.boundaries, type: 'f32', access: 'read_write'},
            {name: 'membership', view: output.membership, type: 'u32', access: 'read_write'},
            {name: 'validMask', view: validMask, type: 'u32', access: 'read_write'}
          ],
          body: `let value = values[valuesOffset + index];
  let valid = isFiniteBits(value) ${props.mask ? '&& mask[maskOffset + index] != 0u' : ''} && invalid[invalidOffset] == 0u;
  var infinityBits = 0x7f800000u;
  let zero = settings[settingsOffset] >> 31u;
  let lowerValue = interpolateOrderStatistic(rawBoundaries[rawBoundariesOffset], rawBoundaries[rawBoundariesOffset + 1u], rawBoundaries[rawBoundariesOffset + 4u], zero);
  let upperValue = interpolateOrderStatistic(rawBoundaries[rawBoundariesOffset + 2u], rawBoundaries[rawBoundariesOffset + 3u], rawBoundaries[rawBoundariesOffset + 5u], zero);
  let lower = select(lowerValue, bitcast<f32>(infinityBits | 0x80000000u), settings[settingsOffset + 1u] == 1u);
  let upper = select(upperValue, bitcast<f32>(infinityBits), settings[settingsOffset + 1u] == settings[settingsOffset]);
  if (index == 0u) { boundaries[boundariesOffset + ${2 * variableIndex}u] = lower; boundaries[boundariesOffset + ${2 * variableIndex + 1}u] = upper; }
  validMask[validMaskOffset + index] = select(0u, 1u, valid)${variableIndex === 0 ? '' : ' & validMask[validMaskOffset + index]'};
  membership[membershipOffset + index] = select(0u, 1u, valid && value >= lower && value < upper)${variableIndex === 0 ? '' : ' & membership[membershipOffset + index]'};`
        })
      );
    }
    nodes.push(
      ...new GPULocalPermutationTest({
        id: `${id}-inference`,
        weights: props.weights,
        values: output.membership,
        mask: validMask,
        parameters: props.parameters,
        statistic: 'localJoinCount',
        alternative: props.alternative === 'geoda-legacy' ? 'greater' : props.alternative,
        islands: 'exclude',
        maximumPermutations: props.maximumPermutations,
        maximumNeighbors: props.maximumNeighbors,
        output: {
          analysisCount: output.analysisCount,
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
      createWGSLKernelNode(graph, {
        id: `${id}-geoda-tail`,
        operation: 'GPUQuantileLISA',
        variant: props.alternative === 'geoda-legacy' ? 'strict-complement-tail' : 'inclusive-tail',
        invocationCount: rows,
        declarations: `${GET_NAN_WGSL}\n${PERMUTATION_SIGNIFICANCE_WGSL}`,
        bindings: [
          {name: 'upperExceedances', view: upperExceedances, type: 'u32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'exceedances', view: output.exceedances, type: 'u32', access: 'read_write'},
          {name: 'pseudoPValues', view: output.pseudoPValues, type: 'f32', access: 'read_write'},
          {name: 'significant', view: significant, type: 'u32', access: 'read_write'},
          ...(output.tested
            ? [
                {
                  name: 'tested',
                  view: output.tested,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `let upperCount = upperExceedances[upperExceedancesOffset + index];
  let permutationCount = parameters[parametersOffset + 2u];
  var count = 0xffffffffu;
  var probability = getNaN();
  var isSignificant = 0u;
  if (upperCount != 0xffffffffu) {
    count = ${props.alternative === 'geoda-legacy' ? 'min(upperCount, permutationCount - upperCount)' : 'upperCount'};
    probability = min(1.0, f32(${props.alternative === 'two-sided' ? '2u * ' : ''}(count + 1u)) / f32(permutationCount + 1u));
    let level = bitcast<f32>(parameters[parametersOffset + 3u]);
    isSignificant = select(0u, 1u, isPermutationRatioSignificant(${props.alternative === 'two-sided' ? '2u * ' : ''}(count + 1u), permutationCount + 1u, level));
  }
  exceedances[exceedancesOffset + index] = count;
  pseudoPValues[pseudoPValuesOffset + index] = probability;
  significant[significantOffset + index] = isSignificant;
  ${output.tested ? 'tested[testedOffset + index] = u32(upperCount != 0xffffffffu);' : ''}`
      })
    );
    nodes.push(
      ...getGPUSpatialWeightsChunkDispatches(props.weights).map(dispatch => {
        const localIndex = dispatch.isWhole ? 'index' : '(index - WEIGHTS_ROW_START)';
        return createWGSLKernelNode(graph, {
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
