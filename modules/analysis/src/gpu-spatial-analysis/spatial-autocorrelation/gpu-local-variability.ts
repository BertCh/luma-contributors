// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';
import {
  createWeightsIncompleteNodes,
  validateWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';
import {createSpatialStatisticsRowNode} from '../spatial-weights/spatial-statistics-row-dispatch';
import {validateSpatialStatisticsWritableDispatches} from '../spatial-weights/spatial-statistics-writable-dispatches';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../spatial-weights/spatial-statistics-counts-wgsl';
import {
  getLocalVariabilityMomentNodes,
  LOCAL_VARIABILITY_MAXIMUM_ROWS,
  LOCAL_VARIABILITY_REFERENCE_MAXIMUM_ROWS,
  LOCAL_VARIABILITY_MAXIMUM_VARIABLE_COUNT,
  LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS
} from './local-variability-moments';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  validateCapacity,
  validateViewLength,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  isGPUChunkedSpatialWeights,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsChunkDispatch
} from '../spatial-weights/index';

import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  resolveStatisticsPopulationCount,
  type SpatialStatisticsPopulationOutput
} from '../spatial-weights/spatial-statistics-population-output';
import * as variabilityParameters from './local-variability-parameters';

/** Per-row status: valid, excluded input, undefined statistic, incomplete weights. */
export const GPU_LOCAL_VARIABILITY_STATUS = {
  OK: 0,
  EXCLUDED: 1,
  UNDEFINED: 2,
  INCOMPLETE: 3
} as const;

type LocalVariabilityProps = {
  id?: string;
  /** Whole weights, or row chunks (one dispatch per chunk, same per-row result). */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Row-major values, `rowCount * variableCount`. Nonfinite rows are excluded jointly. */
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  output: {
    statistic: GraphDataView<'float32'>;
    /** Required per-row completion evidence; see GPU_LOCAL_VARIABILITY_STATUS. */
    status: GraphDataView<'uint32'>;
  };
};

function snapshotLocalVariabilityProps<T extends LocalVariabilityProps>(value: T): T {
  const props = snapshotWeightsFields(value, [
    'id',
    'weights',
    'values',
    'mask',
    'output',
    'residualPower',
    'variableCount',
    'mode'
  ]);
  return Object.freeze({
    ...props,
    weights: snapshotWeightsPort(props.weights),
    output: snapshotWeightsFields(props.output, [
      'statistic',
      'status',
      'normalizedResiduals',
      'residualCount',
      'includedCount',
      'count'
    ])
  });
}

/** LOSH inputs. Positive weights are normalized over included neighbors. Islands are excluded. */
export type GPULocalSpatialHeteroscedasticityProps = LocalVariabilityProps & {
  /**
   * Absolute-deviation power; 2 (default) measures variance, 1 absolute deviation. A packed
   * float32 GPUParameterBuffer view from getGPULocalSpatialHeteroscedasticityParameterValues
   * allows the power to change between encodings without recompiling.
   */
  residualPower?: 1 | 2 | GraphDataView<'float32'>;
  output: LocalVariabilityProps['output'] & {
    /** Optional normalized residual powers for explicit residual-label randomization. */
    normalizedResiduals?: GraphDataView<'float32'>;
    /** Exact number of finite, non-island residuals in the normalization population. */
    residualCount?: GraphDataView<'uint32'>;
  };
};

/**
 * LOSH H_i = weightedMean_j(|y_j - weightedMean_k(y_k)|^a) / mean(residual^a).
 * The denominator includes finite residuals, excluding islands. A fixed-order maximum and sum
 * reduction form the mean from residuals scaled by their maximum, avoiding overflow of their
 * unscaled total. No invocation walks all rows. No inferential p-value is implied.
 * Compile-time: view shapes, optional views and chunk topology. Per-frame: input/mask contents
 * and a residual-power parameter view. Numeric powers are uploaded defaults.
 *
 * Masked rows are removed before local means; neighbors whose residual is undefined are removed
 * before the final weighted mean. Either upstream loss flag, including chunk-local flags,
 * invalidates every statistic and normalized residual with NaN, writes INCOMPLETE status,
 * and publishes residualCount 0. Clearing the flags restores normal computation on resubmission.
 */
export class GPULocalSpatialHeteroscedasticity implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly residualPowerValues?: Float32Array;
  constructor(readonly props: GPULocalSpatialHeteroscedasticityProps) {
    props = snapshotLocalVariabilityProps(props);
    this.props = props;
    this.id = props.id ?? 'local-spatial-heteroscedasticity';
    validateProps(this.id, props, 1);
    const outputs = [
      props.output.statistic,
      props.output.status,
      props.output.normalizedResiduals,
      props.output.residualCount
    ];
    validateSpatialStatisticsWritableDispatches(
      this.id,
      {
        statistic: props.output.statistic,
        status: props.output.status,
        normalizedResiduals: props.output.normalizedResiduals,
        residualCount: props.output.residualCount
      },
      [['statistic', 'status']]
    );
    if (props.output.residualCount) {
      validateSinglePackedView(
        props.output.residualCount,
        ['uint32'],
        `${this.id} output.residualCount`
      );
      validateCapacity(
        this.id,
        'output.residualCount.length',
        props.output.residualCount.length,
        1,
        1
      );
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [props.output.residualCount],
        getAllInputViews(props)
      );
    }
    const residualPower = props.residualPower ?? 2;
    if (typeof residualPower === 'number') {
      validateCapacity(this.id, 'residualPower', residualPower, 1, 2);
      this.residualPowerValues =
        variabilityParameters.getGPULocalSpatialHeteroscedasticityParameterValues({
          residualPower
        });
    } else {
      validateSinglePackedView(residualPower, ['float32'], `${this.id} residualPower`);
      validateCapacity(
        this.id,
        'residualPower.length',
        residualPower.length,
        variabilityParameters.GPU_LOCAL_SPATIAL_HETEROSCEDASTICITY_PARAMETER_LENGTH
      );
      validateGraphOutputsDisjointFromInputs(this.id, outputs, [residualPower]);
    }
    if (props.output.normalizedResiduals) {
      validateSinglePackedView(
        props.output.normalizedResiduals,
        ['float32'],
        `${this.id} normalizedResiduals`
      );
      validateViewLength(
        this.id,
        'normalizedResiduals',
        props.output.normalizedResiduals,
        getRowCount(props)
      );
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [props.output.normalizedResiduals],
        getAllInputViews(props)
      );
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id} = this;
    validateSpatialStatisticsBindings(graph, id, this.props);
    const loss = createWeightsIncompleteNodes(graph, id, [this.props.weights]);
    const props = {...this.props, weights: {...this.props.weights, status: {overflow: loss.view}}};
    const rows = getRowCount(props);
    validateSpatialStatisticsBindings(graph, id, props, {
      'row scratch': rows * 4,
      'moment partials':
        Math.ceil(rows / LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS) *
        ('variableCount' in props ? Number(props.variableCount) : 1) *
        12
    });
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'residual scratch',
      rows * Float32Array.BYTES_PER_ELEMENT,
      'reduce the row count'
    );
    const finiteResiduals = createTransientView(graph, `${id}-finite-residuals`, 'float32', rows);
    const residualCounts = createTransientView(graph, `${id}-residual-counts`, 'uint32', rows);
    const maximumResidual = createTransientView(graph, `${id}-maximum-residual`, 'float32', 1);
    const normalizedSum = createTransientView(graph, `${id}-normalized-sum`, 'float32', 1);
    const residualCount =
      props.output.residualCount ?? createTransientView(graph, `${id}-residual-count`, 'uint32', 1);
    const residuals = createTransientView(graph, `${id}-residuals`, 'float32', rows);
    const mean = createTransientView(graph, `${id}-mean`, 'float32', 1);
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      props.output.statistic,
      props.output.status,
      props.output.normalizedResiduals,
      typeof props.residualPower === 'object' ? props.residualPower : undefined
    ]);
    const runtime = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-parameters`,
      this.residualPowerValues ?? (props.residualPower as GraphDataView<'float32'>)
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      ...loss.nodes,
      ...runtime.nodes,
      ...dispatches.map(dispatch =>
        createSpatialStatisticsRowNode(graph, {
          rowBase: dispatch.rowStart,
          id: `${id}-residuals${dispatch.idSuffix}`,
          operation: 'GPULocalSpatialHeteroscedasticity',
          variant: 'residuals',
          invocationCount: dispatch.rowCount,
          bindings: [
            ...getBindings(props, dispatch.weights),
            {name: 'parameters', view: runtime.parameters, type: 'f32', access: 'read'},
            {name: 'residuals', view: residuals, type: 'f32', access: 'read_write'}
          ],
          declarations: getCommon(props, rows, 1, dispatch),
          body: `var result = getNaN();
if (included(index)) {
  var weighted = 0.0;
  var total = 0.0;
  ${neighborLoop(`weighted += weight * values[valuesOffset + neighbor]; total += weight;`, dispatch)}
  if (total > 0.0) {
    let deviation = abs(values[valuesOffset + index] - weighted / total);
    result = select(deviation * deviation, deviation, parameters[parametersOffset] == 1.0);
  }
}
residuals[residualsOffset + index] = result;`
        })
      ),
      createWGSLKernelNode(graph, {
        id: `${id}-finite-residuals`,
        operation: 'GPULocalSpatialHeteroscedasticity',
        variant: 'finite-residuals',
        invocationCount: rows,
        bindings: [
          {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
          {name: 'finiteResiduals', view: finiteResiduals, type: 'f32', access: 'read_write'},
          {name: 'residualCounts', view: residualCounts, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_BITS_WGSL,
        body: `let value = residuals[residualsOffset + index];
let finite = isFiniteBits(value);
finiteResiduals[finiteResidualsOffset + index] = select(0.0, value, finite);
residualCounts[residualCountsOffset + index] = u32(finite);`
      }),
      ...new GPUReduction({
        id: `${id}-maximum-residual`,
        input: finiteResiduals,
        output: maximumResidual,
        operation: 'max'
      }).getCommandNodes(graph),
      ...new GPUReduction({
        id: `${id}-residual-count`,
        input: residualCounts,
        output: residualCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-scale-residuals`,
        operation: 'GPULocalSpatialHeteroscedasticity',
        variant: 'scale-residuals',
        invocationCount: rows,
        bindings: [
          {name: 'finiteResiduals', view: finiteResiduals, type: 'f32', access: 'read_write'},
          {name: 'maximumResidual', view: maximumResidual, type: 'f32', access: 'read'}
        ],
        declarations: SCALED_QUOTIENT_WGSL,
        body: `let maximum = maximumResidual[maximumResidualOffset];
var scaled = 0.0;
if (maximum > 0.0) {
  // A direct division can flush the reciprocal of a large maximum to zero.
  scaled = getScaledQuotient(finiteResiduals[finiteResidualsOffset + index], maximum, 1.0, 0);
}
finiteResiduals[finiteResidualsOffset + index] = scaled;`
      }),
      ...new GPUReduction({
        id: `${id}-normalized-sum`,
        input: finiteResiduals,
        output: normalizedSum,
        operation: 'sum'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-mean`,
        operation: 'GPULocalSpatialHeteroscedasticity',
        variant: 'mean',
        // contract-allow: single-invocation -- count division takes exactly 64 integer steps.
        invocationCount: 1,
        bindings: [
          {name: 'maximumResidual', view: maximumResidual, type: 'f32', access: 'read'},
          {name: 'normalizedSum', view: normalizedSum, type: 'f32', access: 'read'},
          {name: 'residualCount', view: residualCount, type: 'u32', access: 'read'},
          {name: 'mean', view: mean, type: 'f32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}
${SPATIAL_STATISTICS_COUNTS_WGSL}
// 2^126 is finite; splitting larger exponents avoids an overflowing ldexp scale.
const MAXIMUM_MEAN_SCALE_EXPONENT: i32 = 126;`,
        body: `let count = residualCount[residualCountOffset];
let maximum = frexp(maximumResidual[maximumResidualOffset]);
// Keep the maximum's exponent out of float products, including compiler reassociation.
let fraction = maximum.fract * divideBySpatialCount(normalizedSum[normalizedSumOffset], count);
let firstExponent = min(maximum.exp, MAXIMUM_MEAN_SCALE_EXPONENT);
let average = ldexp(ldexp(fraction, firstExponent), maximum.exp - firstExponent);
mean[meanOffset] = select(getNaN(), average, count > 0u);`
      }),
      ...dispatches.map(dispatch =>
        createSpatialStatisticsRowNode(graph, {
          rowBase: dispatch.rowStart,
          id: `${id}-finish${dispatch.idSuffix}`,
          operation: 'GPULocalSpatialHeteroscedasticity',
          variant: 'finish',
          invocationCount: dispatch.rowCount,
          // A compact scratch stage keeps the standard eight storage-binding limit.
          bindings: [
            {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
            {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
            {name: 'mean', view: mean, type: 'f32', access: 'read'},
            {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
            ...(props.weights.status?.overflow
              ? [
                  {
                    name: 'overflow',
                    view: props.weights.status.overflow,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : [])
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${SCALED_QUOTIENT_WGSL}\n`,
          body: `var resultValue = getNaN(); var state = 1u;
if (isFiniteBits(residuals[residualsOffset + index])) {
  var total = 0.0; var weighted = 0.0;
  let rowBegin = offsets[offsetsOffset + ${getLocalRow(dispatch)}];
  let rowEnd = offsets[offsetsOffset + ${getLocalRow(dispatch)} + 1u];
  for (var slot = rowBegin; slot < min(rowEnd, CAPACITY); slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
    if (neighbor >= ${rows}u || neighbor == index || !(weight > 0.0) || !isFiniteBits(weight)) { continue; }
    let residual = residuals[residualsOffset + neighbor];
    if (isFiniteBits(residual)) { weighted += weight * residual; total += weight; }
  }
  state = 2u;
  if (total > 0.0 && mean[meanOffset] > 0.0) {
    if (isFiniteBits(weighted) && isFiniteBits(total) && isFiniteBits(mean[meanOffset])) {
      resultValue = getScaledQuotient(weighted, total, mean[meanOffset], 0);
    } else {
      resultValue = weighted / total / mean[meanOffset];
    }
    state = 0u;
  }
}
${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { state = 3u; resultValue = getNaN(); }' : ''}
if (!isFiniteBits(resultValue) && state == 0u) { state = 2u; }
result[resultOffset + index] = resultValue; status[statusOffset + index] = state;`
        })
      )
    ];
    if (props.output.normalizedResiduals) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-normalized-residuals`,
          operation: 'GPULocalSpatialHeteroscedasticity',
          variant: 'normalized-residuals',
          invocationCount: rows,
          bindings: [
            {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
            {name: 'mean', view: mean, type: 'f32', access: 'read'},
            {
              name: 'normalized',
              view: props.output.normalizedResiduals,
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${SCALED_QUOTIENT_WGSL}`,
          body: `let average = mean[meanOffset];
let residual = residuals[residualsOffset + index];
var normalizedValue = getNaN();
if (average > 0.0 && isFiniteBits(average)) {
  normalizedValue = residual;
  if (isFiniteBits(residual)) {
    normalizedValue = getScaledQuotient(residual, average, 1.0, 0);
  }
}
normalized[normalizedOffset + index] = normalizedValue;`
        })
      );
    }
    return nodes;
  }
}

/** Multivariate local Geary inputs, with population standardization and joint complete-case rows. */
export type GPUMultivariateLocalGearyProps = LocalVariabilityProps & {
  variableCount: number;
  /** Tiled moments by default; preserve the former serial reference for explicit comparison. */
  mode?: 'tiled' | 'reference';
  output: LocalVariabilityProps['output'] & SpatialStatisticsPopulationOutput;
};

/**
 * Mean across variables of row-standardized local Geary. Each variable uses its population
 * variance over joint complete-case rows, including islands; island results are undefined.
 * A constant variable invalidates the statistic. This is descriptive: no univariate permutation
 * adapter is reused because valid joint randomization must preserve row pairing across variables.
 */
export class GPUMultivariateLocalGeary implements GPUCommandNodeProducer {
  readonly id: string;
  private readonly populationCount?: GraphDataView<'uint32'>;
  constructor(readonly props: GPUMultivariateLocalGearyProps) {
    props = snapshotLocalVariabilityProps(props);
    this.props = props;
    this.id = props.id ?? 'multivariate-local-geary';
    this.populationCount = resolveStatisticsPopulationCount(this.id, props.output);
    validateProps(this.id, props, props.variableCount);
    if (props.mode !== undefined && props.mode !== 'tiled' && props.mode !== 'reference')
      throw new Error(`${this.id} mode must be tiled or reference`);
    if (props.mode === 'reference')
      validateCapacity(
        this.id,
        'weights.rowCount (LOCAL_VARIABILITY_REFERENCE_MAXIMUM_ROWS)',
        getRowCount(props),
        1,
        LOCAL_VARIABILITY_REFERENCE_MAXIMUM_ROWS
      );
    if (this.populationCount) {
      validateSinglePackedView(
        this.populationCount,
        ['uint32'],
        `${this.id} output.${props.output.includedCount ? 'includedCount' : 'count'}`
      );
      validateCapacity(
        this.id,
        `output.${props.output.includedCount ? 'includedCount' : 'count'}.length`,
        this.populationCount.length,
        1
      );
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [this.populationCount],
        getAllInputViews(props)
      );
    }
    validateSpatialStatisticsWritableDispatches(
      this.id,
      {statistic: props.output.statistic, status: props.output.status, count: this.populationCount},
      [['statistic', 'status']]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id} = this;
    validateSpatialStatisticsBindings(graph, id, this.props);
    const loss = createWeightsIncompleteNodes(graph, id, [this.props.weights]);
    const props = {...this.props, weights: {...this.props.weights, status: {overflow: loss.view}}};
    const rows = getRowCount(props);
    validateSpatialStatisticsBindings(graph, id, props, {
      'row scratch': rows * 4,
      'moment partials':
        Math.ceil(rows / LOCAL_VARIABILITY_MOMENT_BLOCK_ROWS) *
        ('variableCount' in props ? Number(props.variableCount) : 1) *
        12
    });
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const variance = createTransientView(graph, `${id}-variance`, 'float32', props.variableCount);
    const bindings = getBindings(props, dispatches[0].weights);
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      props.output.statistic,
      props.output.status
    ]);
    const common = getCommon(props, rows, props.variableCount, dispatches[0]);
    const valueBindings = bindings.filter(binding =>
      ['values', 'mask', 'overflow'].includes(binding.name)
    );
    const nodes: GPUCommandNode<Parameters>[] = [
      ...loss.nodes,
      ...(props.mode === 'reference'
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-moments`,
              operation: 'GPUMultivariateLocalGeary',
              variant: 'moments',
              invocationCount: props.variableCount,
              bindings: [
                ...valueBindings,
                {name: 'variance', view: variance, type: 'f32', access: 'read_write'},
                ...(this.populationCount
                  ? [
                      {
                        name: 'populationCount',
                        view: this.populationCount,
                        type: 'u32' as const,
                        access: 'read_write' as const
                      }
                    ]
                  : [])
              ],
              declarations: common,
              body: `var count = 0.0; var mean = 0.0; var squared = 0.0;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row)) { continue; }
  let value = values[valuesOffset + row * ${props.variableCount}u + index];
  count += 1.0; let difference = value - mean; mean += difference / count; squared += difference * (value - mean);
}
variance[varianceOffset + index] = select(getNaN(), squared / count, count > 0.0);
${this.populationCount ? 'if (index == 0u) { populationCount[populationCountOffset] = u32(count); }' : ''}`
            })
          ]
        : getLocalVariabilityMomentNodes(graph, {
            id,
            rows,
            columns: props.variableCount,
            bindings: valueBindings,
            declarations: common,
            variance,
            count: this.populationCount
          })),
      ...dispatches.map(dispatch =>
        createSpatialStatisticsRowNode(graph, {
          rowBase: dispatch.rowStart,
          id: `${id}-statistic${dispatch.idSuffix}`,
          operation: 'GPUMultivariateLocalGeary',
          variant: 'statistic',
          invocationCount: dispatch.rowCount,
          bindings: [
            ...getBindings(props, dispatch.weights).filter(binding => binding.name !== 'overflow'),
            {name: 'variance', view: variance, type: 'f32', access: 'read'},
            {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${getCommon(
            {...props, weights: {...props.weights, status: undefined}},
            rows,
            props.variableCount,
            dispatch
          )}\n${SCALED_QUOTIENT_WGSL}`,
          body: `var state = 1u; var resultValue = getNaN();
if (included(index)) {
  var sum = 0.0; var total = 0.0; var defined = true;
  for (var column = 0u; column < ${props.variableCount}u; column++) { defined = defined && variance[varianceOffset + column] > 0.0 && isFiniteBits(variance[varianceOffset + column]); }
  if (defined) {
    ${neighborLoop(
      `var distance = 0.0;
      for (var column = 0u; column < ${props.variableCount}u; column++) {
        let difference = values[valuesOffset + index * ${props.variableCount}u + column] - values[valuesOffset + neighbor * ${props.variableCount}u + column];
        // Normalize before squaring: the squared raw difference can overflow although the
        // variance and the dimensionless Geary statistic are finite. Exponent scaling also
        // prevents Metal's reciprocal lowering from flushing a large denominator to zero.
        let standardized = getScaledQuotient(difference, sqrt(variance[varianceOffset + column]), 1.0, 0);
        distance += standardized * standardized;
      }
      sum += weight * distance; total += weight;`,
      dispatch
    )}
  }
  state = 2u;
  if (defined && total > 0.0) { resultValue = sum / total / ${props.variableCount}.0; state = select(2u, 0u, isFiniteBits(resultValue)); }
}
result[resultOffset + index] = resultValue; status[statusOffset + index] = state;`
        })
      )
    ];
    if (props.weights.status?.overflow)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-upstream-status`,
          operation: 'GPUMultivariateLocalGeary',
          variant: 'upstream-status',
          invocationCount: rows,
          bindings: [
            {name: 'overflow', view: props.weights.status.overflow, type: 'u32', access: 'read'},
            {name: 'result', view: props.output.statistic, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: GET_NAN_WGSL,
          body: 'if (overflow[overflowOffset] != 0u) { result[resultOffset + index] = getNaN(); status[statusOffset + index] = 3u; }'
        })
      );
    return nodes;
  }
}

function validateProps(id: string, props: LocalVariabilityProps, variableCount: number): void {
  const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
  validateCapacity(
    id,
    'weights.rowCount (LOCAL_VARIABILITY_MAXIMUM_ROWS)',
    rows,
    1,
    LOCAL_VARIABILITY_MAXIMUM_ROWS
  );
  validateCapacity(
    id,
    'variableCount (LOCAL_VARIABILITY_MAXIMUM_VARIABLE_COUNT)',
    variableCount,
    1,
    LOCAL_VARIABILITY_MAXIMUM_VARIABLE_COUNT
  );
  validateCapacity(
    id,
    'values.length (uint32 row-major addressing)',
    props.values.length,
    1,
    0xffffffff
  );
  validateDisjointWritableViews(id, {
    statistic: props.output.statistic,
    status: props.output.status
  });
  const views = [props.values, props.output.statistic, props.output.status, props.mask];
  const lengths = [rows * variableCount, rows, rows, rows];
  for (let index = 0; index < views.length; index++) {
    const view = views[index];
    if (!view) continue;
    validateSinglePackedView(view, [index < 2 ? 'float32' : 'uint32'], id);
    if (view.length !== lengths[index]) throw new Error(`${id}: invalid row count`);
  }
  validateGraphOutputsDisjointFromInputs(
    id,
    [props.output.statistic, props.output.status],
    getAllInputViews(props)
  );
}
function getRowCount(props: LocalVariabilityProps): number {
  return isGPUChunkedSpatialWeights(props.weights)
    ? props.weights.rowCount
    : props.weights.offsets.length - 1;
}
function getAllInputViews(
  props: LocalVariabilityProps
): (GraphDataView<'uint32' | 'float32'> | undefined)[] {
  return [
    props.values,
    props.mask,
    ...validateWeightsLossSources(props.id ?? 'local-variability', [props.weights]),
    ...getGPUSpatialWeightsOrChunksViews(props.weights)
  ];
}
function getLocalRow(dispatch: GPUSpatialWeightsChunkDispatch): string {
  void dispatch;
  return 'index - WEIGHTS_ROW_START';
}
function getBindings(
  props: LocalVariabilityProps,
  weights: GPUSpatialWeights
): WGSLKernelBinding[] {
  return [
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : []),
    ...(props.weights.status?.overflow
      ? [
          {
            name: 'overflow',
            view: props.weights.status.overflow,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [])
  ];
}
function getCommon(
  props: LocalVariabilityProps,
  rows: number,
  variableCount: number,
  dispatch: GPUSpatialWeightsChunkDispatch
): string {
  void dispatch;
  return `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
const ROWS: u32 = ${rows}u;

fn included(row: u32) -> bool {
  if (row >= ROWS) { return false; }
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  ${props.weights.status?.overflow ? 'if (overflow[overflowOffset] != 0u) { return false; }' : ''}
  for (var column = 0u; column < ${variableCount}u; column++) {
    if (!isFiniteBits(values[valuesOffset + row * ${variableCount}u + column])) { return false; }
  }
  return true;
}`;
}
function neighborLoop(action: string, dispatch: GPUSpatialWeightsChunkDispatch): string {
  const localRow = getLocalRow(dispatch);
  return `for (var slot = offsets[offsetsOffset + ${localRow}]; slot < min(offsets[offsetsOffset + ${localRow} + 1u], CAPACITY); slot++) {
    let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
    if (neighbor == index || !included(neighbor) || !(weight > 0.0) || !isFiniteBits(weight)) { continue; }
    ${action}
  }`;
}
