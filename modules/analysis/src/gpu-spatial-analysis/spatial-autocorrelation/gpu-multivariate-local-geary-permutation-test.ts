// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getLocalVariabilityCenteringNodes} from './local-variability-centering';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {
  snapshotWeightsFields,
  snapshotWeightsDescription
} from '../spatial-weights/weights-construction-snapshot';
import {
  createWeightsIncompleteNodes,
  validateWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';
import {
  createTransientView,
  GPUReduction,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  getGPUMultivariateLocalGearyPermutationTestParameterValues,
  GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH,
  validateMultivariateGearyOptionNames,
  validateMultivariateGearySettings
} from './multivariate-geary-parameters';
import {IS_FINITE_BITS_WGSL, GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {PERMUTATION_RANDOM_WGSL} from '../../analysis-core/kernels/permutation-random';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';
import {
  GPUMultipleTestingAdjustment,
  resolveGPUMultipleTestingMethod
} from '../multiple-testing/gpu-multiple-testing-adjustment';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  getGPUSpatialWeightsLossAndCompletenessViews,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {
  DEFAULT_GEARY_MAXIMUM_NEIGHBORS,
  DEFAULT_GEARY_PERMUTATION_TILE_SIZE,
  DEFAULT_GEARY_ROW_TILE_SIZE,
  MAXIMUM_GEARY_MAXIMUM_NEIGHBORS,
  getTiledMultivariateLocalGearyNodes
} from './gpu-multivariate-local-geary-tiled';

/** Rows handled by the exact baseline strategy and the default strategy cutoff. */
export const EXACT_MULTIVARIATE_GEARY_MAXIMUM_ROWS = 256;
/** Largest row count of the tiled strategy. */
export const TILED_MULTIVARIATE_GEARY_MAXIMUM_ROWS = 2 ** 24 - 1;
/** Largest permutation count of the tiled strategy. */
export const TILED_MULTIVARIATE_GEARY_MAXIMUM_PERMUTATIONS = 9999;

// One family flag pass, at most three 256-way reduction levels for 2^24 rows,
// one reduction finalization, one probability multiplication and one class pass.
const MULTIVARIATE_GEARY_BONFERRONI_MAXIMUM_NODES = 7;

export type GPUMultivariateLocalGearyPermutationTestProps = {
  id?: string;
  /**
   * Whole CSR or consecutive row chunks; chunks are dispatched one at a time, never repacked.
   * A supplied false status.complete certificate, overflow or candidateOverflow suppresses
   * numerical inference on both strategies. Completeness views are captured at construction.
   */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Row-major variables; a common complete-case mask applies to all variables. */
  values: GraphDataView<'float32'>;
  variableCount: number;
  mask?: GraphDataView<'uint32'>;
  /** Fixed draw count. Defaults to 999; this is not a per-frame maximum. */
  permutationCount?: number;
  /** Numeric default when parameters is omitted. */
  seed?: number | readonly [number, number];
  /** Numeric default when parameters is omitted. */
  significanceLevel?: number;
  /**
   * Per-frame [seedLow, seedHigh, float32 bits of significance], packed with
   * getGPUMultivariateLocalGearyPermutationTestParameterValues in a uint32 GPUParameterBuffer.
   * When present, overrides seed and significanceLevel. Numeric defaults use the same layout.
   */
  parameters?: GraphDataView<'uint32'>;
  /** Bonferroni family is every included row (including islands); default no adjustment. */
  correction?: 'none' | 'bonferroni';
  /** Bonferroni population; defaults to complete for correction and tested for multipleTesting. */
  family?: 'tested' | 'complete';
  /**
   * Opt-in family-wise or false-discovery-rate correction of the pseudo p-values through
   * `GPUMultipleTestingAdjustment`: `'bonferroni' | 'holm' | 'benjamini-hochberg' |
   * 'benjamini-yekutieli'` (use `'benjamini-yekutieli'` for local statistics, whose p-values are
   * dependent). `output.adjustedPValues` then holds the adjusted p-value and `output.classes` marks a row
   * significant when the adjusted p-value is at most `significanceLevel`, in the direction of the
   * uncorrected test. The family is every row with a finite p-value (islands and rows beyond
   * `maximumNeighbors` are untested), unlike `correction: 'bonferroni'`, whose family is every
   * included row. Mutually exclusive with `correction: 'bonferroni'`. Default `'none'`: the
   * classes use the uncorrected (or `correction`) p-value, which paints about `significanceLevel` of
   * the rows significant under the null. Whether to default to a correction is a maintainer
   * decision (SA2-14).
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /** equal-tail doubles inclusive tails; esda-min-tail matches esda's undoubled minimum tail. */
  tail?: 'equal-tail' | 'esda-min-tail';
  /**
   * `exact` is the original single-dispatch baseline (<=256 rows, <=4096 permutations, optional
   * simulations output) and is the oracle. `tiled` yields the same statistic, p-values and
   * classes with row- and permutation-tiled dispatches. Default: `exact` up to 256 rows and 4096
   * permutations, else `tiled`.
   */
  strategy?: 'exact' | 'tiled';
  /** Tiled only: rows per simulation dispatch. */
  rowTileSize?: number;
  /** Tiled only: conditional permutations per simulation dispatch. */
  permutationTileSize?: number;
  /**
   * Tiled only: largest active degree whose randomization is evaluated (default 64, at most 128).
   * Rows with more active neighbors are reported untested (NaN p, class 0xffffffff).
   */
  maximumNeighbors?: number;
  output: {
    statistic: GraphDataView<'float32'>;
    /** Unadjusted pseudo p-value in every correction mode. */
    pseudoPValues?: GraphDataView<'float32'>;
    /** Corrected p-value, or the raw p-value when no correction is requested. */
    adjustedPValues?: GraphDataView<'float32'>;
    /** 0 nonsignificant, 1 significant similarity, 2 significant dissimilarity, 0xffffffff invalid. */
    classes: GraphDataView<'uint32'>;
    /** Optional row-major replay of every joint-vector conditional randomization. */
    simulations?: GraphDataView<'float32'>;
  };
};

/** Caller-owned outputs of the multivariate conditional test. */
export type GPUMultivariateLocalGearyPermutationTestOutput =
  GPUMultivariateLocalGearyPermutationTestProps['output'];

/**
 * Per-frame: values, mask, weights and parameter contents (seed and significance).
 * Compile-time: view shapes, variableCount, permutationCount, strategy, tile sizes,
 * maximumNeighbors, tail/correction choices and which optional outputs are present.
 *
 * Defaults: 999 fixed draws, seed words [0, 0], significance level 0.05 and inclusive equal tails.
 * Mean multivariate local Geary using population-standardized variables and row-standardized
 * active weights. Conditional randomization fixes the focal vector and samples distinct whole
 * neighbor vectors without replacement from all other complete rows. It never permutes columns
 * independently. Philox conditional streams match the shared local-permutation purpose.
 * Conservative equal-tail Monte Carlo p = min(1, 2 min((lower+1)/(P+1),(upper+1)/(P+1))).
 * A constant variable invalidates the joint statistic; islands/masked rows are unknown, not zero.
 * Exact baseline (`strategy: 'exact'`): <=256 rows, <=16 variables, <=4096 permutations. The
 * `tiled` strategy handles up to 2^24-1 rows and 9999 permutations with identical results.
 */
export class GPUMultivariateLocalGearyPermutationTest implements GPUCommandNodeProducer {
  readonly id: string;
  /** Construction upper bound, including tile dispatches, preparation, loss and correction. */
  readonly estimatedCommandNodeCount: number;
  private readonly parameterValues?: Uint32Array;
  constructor(readonly props: GPUMultivariateLocalGearyPermutationTestProps) {
    const captured = snapshotWeightsFields(props, [
      'id',
      'weights',
      'values',
      'variableCount',
      'mask',
      'permutationCount',
      'seed',
      'significanceLevel',
      'parameters',
      'correction',
      'family',
      'multipleTesting',
      'tail',
      'strategy',
      'rowTileSize',
      'permutationTileSize',
      'maximumNeighbors',
      'output'
    ]);
    const id = captured.id ?? 'multivariate-local-geary-permutation-test';
    validateMultivariateGearyOptionNames(id, props);
    const settings = validateMultivariateGearySettings(id, captured);
    validateRequirement(
      id,
      !('pValues' in captured.output),
      'output.pValues was removed. Use output.pseudoPValues for raw values or output.adjustedPValues for corrected values'
    );
    props = Object.freeze({
      ...captured,
      weights: snapshotWeightsDescription(captured.weights),
      permutationCount: settings.permutationCount,
      significanceLevel: settings.significanceLevel,
      seed: Object.freeze(settings.seed),
      output: snapshotWeightsFields(captured.output, [
        'statistic',
        'pseudoPValues',
        'adjustedPValues',
        'classes',
        'simulations'
      ])
    });
    this.props = props;
    this.id = id;
    validateWeightsLossSources(this.id, [props.weights], true);
    for (const complete of this.getWeightsCompletenessViews()) {
      validateSinglePackedView(complete, ['uint32'], `${this.id} weights.status.complete`);
      validateCapacity(this.id, 'weights.status.complete.length', complete.length, 1);
    }
    validateRequirement(
      this.id,
      props.strategy === undefined || props.strategy === 'exact' || props.strategy === 'tiled',
      'strategy must be exact or tiled'
    );
    if (props.parameters) {
      validateSinglePackedView(props.parameters, ['uint32'], `${this.id} parameters`);
      validateCapacity(
        this.id,
        'parameters.length',
        props.parameters.length,
        GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH
      );
    } else {
      this.parameterValues = getGPUMultivariateLocalGearyPermutationTestParameterValues(props);
    }
    const rows = validateGPUSpatialWeightsOrChunks(this.id, props.weights);
    const tiled = this.getStrategy(rows) === 'tiled';
    validateCapacity(
      this.id,
      'rows',
      rows,
      2,
      tiled ? TILED_MULTIVARIATE_GEARY_MAXIMUM_ROWS : EXACT_MULTIVARIATE_GEARY_MAXIMUM_ROWS
    );
    validateCapacity(this.id, 'variableCount', props.variableCount, 1, 16);
    validateCapacity(
      this.id,
      'permutationCount',
      props.permutationCount ?? 999,
      1,
      tiled ? TILED_MULTIVARIATE_GEARY_MAXIMUM_PERMUTATIONS : 4096
    );
    if (tiled) {
      if (props.output.simulations)
        throw new Error(
          `${this.id} output.simulations requires strategy exact because tiled inference does not retain the draw matrix; use exact within 256 rows and 4096 permutations or omit output.simulations`
        );
      validateCapacity(
        this.id,
        'maximumNeighbors',
        props.maximumNeighbors ?? DEFAULT_GEARY_MAXIMUM_NEIGHBORS,
        1,
        MAXIMUM_GEARY_MAXIMUM_NEIGHBORS
      );
      validateCapacity(this.id, 'rowTileSize', props.rowTileSize ?? DEFAULT_GEARY_ROW_TILE_SIZE);
      validateCapacity(
        this.id,
        'permutationTileSize',
        props.permutationTileSize ?? DEFAULT_GEARY_PERMUTATION_TILE_SIZE
      );
    }
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const permutationTiles = Math.ceil(
      (props.permutationCount ?? 999) /
        (props.permutationTileSize ?? DEFAULT_GEARY_PERMUTATION_TILE_SIZE)
    );
    const simulations = tiled
      ? dispatches.reduce(
          (total, dispatch) =>
            total +
            Math.ceil(dispatch.rowCount / (props.rowTileSize ?? DEFAULT_GEARY_ROW_TILE_SIZE)) *
              permutationTiles,
          0
        )
      : dispatches.length;
    const observedTiles = tiled
      ? dispatches.reduce(
          (total, dispatch) =>
            total +
            Math.ceil(dispatch.rowCount / (props.rowTileSize ?? DEFAULT_GEARY_ROW_TILE_SIZE)),
          0
        )
      : dispatches.length;
    // Preparation: parameter upload, complete rows, hierarchical compaction/centering reductions,
    // moments, observed rows and epilogue. At <=2^24 rows these require fewer than 192 nodes.
    // Shared adjustment uses a 32-bit radix sort and scans: fewer than 1024 nodes at this bound.
    const setupNodes = 192 + observedTiles;
    const lossNodes =
      2 * getGPUSpatialWeightsLossAndCompletenessViews(props.weights, this.id).loss.length +
      this.getWeightsCompletenessViews().length +
      9;
    const method = resolveGPUMultipleTestingMethod(this.id, props.multipleTesting);
    const correctionNodes =
      props.correction === 'bonferroni' || method === 'bonferroni'
        ? MULTIVARIATE_GEARY_BONFERRONI_MAXIMUM_NODES
        : method !== 'none'
          ? 1024
          : props.output.adjustedPValues
            ? 2
            : 0;
    this.estimatedCommandNodeCount = simulations + setupNodes + lossNodes + correctionNodes;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    if (method !== 'none' && props.correction === 'bonferroni') {
      throw new Error(`${this.id}: use either correction or multipleTesting`);
    }
    validateRequirement(
      this.id,
      props.tail === undefined || ['equal-tail', 'esda-min-tail'].includes(props.tail),
      `tail must be equal-tail or esda-min-tail; received ${props.tail}`
    );
    validateRequirement(
      this.id,
      ['none', 'bonferroni'].includes(props.correction ?? 'none'),
      'correction must be none or bonferroni'
    );
    validateRequirement(
      this.id,
      props.family === undefined || ['tested', 'complete'].includes(props.family),
      `family must be tested or complete; received ${props.family}`
    );
    validateRequirement(
      this.id,
      props.family === undefined ||
        props.correction === 'bonferroni' ||
        props.multipleTesting === 'bonferroni',
      'family requires correction or multipleTesting bonferroni'
    );
    for (const [view, format, length] of [
      [props.values, 'float32', rows * props.variableCount],
      [props.mask, 'uint32', rows],
      [props.output.statistic, 'float32', rows],
      [props.output.pseudoPValues, 'float32', rows],
      [props.output.adjustedPValues, 'float32', rows],
      [props.output.classes, 'uint32', rows],
      [props.output.simulations, 'float32', rows * (props.permutationCount ?? 999)]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateDisjointWritableViews(this.id, props.output);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.values,
      props.mask,
      props.parameters,
      ...this.getWeightsCompletenessViews(),
      ...getGPUSpatialWeightsOrChunksViews(props.weights, true),
      props.weights.status?.overflow
    ]);
  }
  /**
   * Reports captured words for fixed settings (the argument cannot override them). With a
   * resident parameter view, current settings or packed uint32 words are required: GPU contents
   * cannot be inferred from the numeric defaults. Words may be an offset view with padding.
   */
  getInferenceMetadata(
    settings?:
      | import('./multivariate-geary-parameters').GPUMultivariateLocalGearyPermutationTestSettings
      | Uint32Array
  ) {
    validateRequirement(
      this.id,
      Boolean(this.parameterValues || settings),
      'getInferenceMetadata requires current settings or packed words for resident parameters'
    );
    const packed =
      this.parameterValues ??
      (settings instanceof Uint32Array
        ? settings
        : getGPUMultivariateLocalGearyPermutationTestParameterValues(settings!));
    validateCapacity(
      this.id,
      'metadata packed words.length',
      packed.length,
      GPU_MULTIVARIATE_LOCAL_GEARY_PERMUTATION_TEST_PARAMETER_LENGTH
    );
    const significanceLevel = new Float32Array(packed.buffer, packed.byteOffset, packed.length)[2];
    validateMultivariateGearySettings(this.id, {seed: [packed[0], packed[1]], significanceLevel});
    return {
      seed: [packed[0], packed[1]] as readonly [number, number],
      permutationCount: this.props.permutationCount ?? 999,
      significanceLevel,
      tail: this.props.tail ?? 'equal-tail',
      includeObserved: true,
      correction: this.props.correction ?? 'none',
      family: this.props.family ?? (this.props.correction === 'bonferroni' ? 'complete' : 'tested'),
      multipleTesting: this.props.multipleTesting ?? 'none'
    };
  }
  private _getRowCount(): number {
    const {weights} = this.props;
    return 'chunks' in weights ? weights.rowCount : weights.offsets.length - 1;
  }
  /** Resolves the requested strategy against the row count. */
  getStrategy(rows = this._getRowCount()): 'exact' | 'tiled' {
    return (
      this.props.strategy ??
      (rows > EXACT_MULTIVARIATE_GEARY_MAXIMUM_ROWS || (this.props.permutationCount ?? 999) > 4096
        ? 'tiled'
        : 'exact')
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateSpatialStatisticsBindings(graph, this.id, this.props, {
      'row scratch': this.props.values.length * 4,
      'values and column scales': (this.props.values.length + this.props.variableCount + 1) * 4,
      'tiled accumulators': this._getRowCount() * 16,
      'moment partials': Math.ceil(this._getRowCount() / 4096) * this.props.variableCount * 12
    });
    const {props, id} = this;
    const rows = this._getRowCount();
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      props.parameters,
      ...this.getWeightsCompletenessViews(),
      ...getGPUSpatialWeightsOrChunksViews(props.weights, true),
      props.weights.status?.overflow,
      ...Object.values(props.output)
    ]);
    if (props.parameters) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'parameters binding',
        getViewBindingRange(props.parameters).size,
        'use a smaller packed parameter view'
      );
    }
    const runtime = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-parameters`,
      this.parameterValues ?? props.parameters!
    );
    const method = resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    if (method !== 'none' || props.correction === 'bonferroni' || props.output.adjustedPValues) {
      return [...runtime.nodes, ...this.getStableProbabilityNodes(graph, runtime.parameters)];
    }
    const raw =
      props.output.pseudoPValues ?? createTransientView(graph, `${id}-raw`, 'float32', rows);
    return [...runtime.nodes, ...this.getInferenceNodes(graph, runtime.parameters, raw)];
  }

  private getStableProbabilityNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    parameters: GraphDataView<'uint32'>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = this._getRowCount();
    const method = resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    const bonferroni = props.correction === 'bonferroni' || method === 'bonferroni';
    const family = props.family ?? (props.correction === 'bonferroni' ? 'complete' : 'tested');
    const raw =
      props.output.pseudoPValues ?? createTransientView(graph, `${id}-raw`, 'float32', rows);
    const adjusted =
      props.output.adjustedPValues ?? createTransientView(graph, `${id}-adjusted`, 'float32', rows);
    const directions = createTransientView(graph, `${id}-directions`, 'uint32', rows);
    const inner = new GPUMultivariateLocalGearyPermutationTest({
      ...props,
      id: `${id}-uncorrected`,
      correction: 'none',
      multipleTesting: 'none',
      family: undefined,
      output: {
        statistic: props.output.statistic,
        pseudoPValues: raw,
        classes: directions,
        simulations: props.output.simulations
      }
    });
    // Both strategies retain the shared weights-loss aggregate inside getInferenceNodes.
    const nodes: GPUCommandNode<Parameters>[] = [
      ...inner.getInferenceNodes(graph, parameters, raw, true)
    ];
    if (bonferroni) {
      const flags = createTransientView(graph, `${id}-family-flags`, 'uint32', rows);
      const count = createTransientView(graph, `${id}-family-count`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-family-flags`,
          operation: 'GPUMultivariateLocalGearyPermutationTest',
          variant: `family-${family}`,
          invocationCount: rows,
          bindings:
            family === 'tested'
              ? [
                  {name: 'raw', view: raw, type: 'f32', access: 'read'},
                  {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
                ]
              : [
                  {name: 'values', view: props.values, type: 'f32', access: 'read'},
                  ...(props.mask
                    ? [
                        {
                          name: 'mask',
                          view: props.mask,
                          type: 'u32' as const,
                          access: 'read' as const
                        }
                      ]
                    : []),
                  {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
                ],
          declarations: IS_FINITE_BITS_WGSL,
          body:
            family === 'tested'
              ? 'flags[flagsOffset + index] = u32(isFiniteBits(raw[rawOffset + index]));'
              : `var included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var column = 0u; column < ${props.variableCount}u; column++) {
  included = included && isFiniteBits(values[valuesOffset + index * ${props.variableCount}u + column]);
}
flags[flagsOffset + index] = u32(included);`
        })
      );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-family-count`,
          input: flags,
          output: count,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-bonferroni`,
          operation: 'GPUMultivariateLocalGearyPermutationTest',
          variant: 'bonferroni',
          invocationCount: rows,
          bindings: [
            {name: 'raw', view: raw, type: 'f32', access: 'read'},
            {name: 'count', view: count, type: 'u32', access: 'read'},
            {name: 'adjusted', view: adjusted, type: 'f32', access: 'read_write'}
          ],
          declarations: IS_FINITE_BITS_WGSL,
          body: `let probability = raw[rawOffset + index];
adjusted[adjustedOffset + index] = select(probability, min(1.0, probability * f32(count[countOffset])), isFiniteBits(probability));`
        })
      );
    } else {
      nodes.push(
        ...new GPUMultipleTestingAdjustment({
          id: `${id}-multiple-testing`,
          pValues: raw,
          method,
          output: {adjustedPValues: adjusted}
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-probabilities`,
        operation: 'GPUMultivariateLocalGearyPermutationTest',
        variant: 'publish-probabilities',
        invocationCount: rows,
        bindings: [
          {name: 'adjusted', view: adjusted, type: 'f32', access: 'read'},
          {name: 'directions', view: directions, type: 'u32', access: 'read'},
          {name: 'classes', view: props.output.classes, type: 'u32', access: 'read_write'},
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'}
        ],
        body: `let probability = adjusted[adjustedOffset + index];
let direction = directions[directionsOffset + index];
classes[classesOffset + index] = select(select(0u, direction, probability <= bitcast<f32>(parameters[parametersOffset + 2u])), 0xffffffffu, direction == 0xffffffffu);`
      })
    );
    return nodes;
  }

  private getWeightsCompletenessViews(): GraphDataView<'uint32'>[] {
    return getGPUSpatialWeightsLossAndCompletenessViews(this.props.weights, this.id).complete;
  }

  private getInferenceNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    parameters: GraphDataView<'uint32'>,
    raw: GraphDataView<'float32'>,
    classifyDirections = false
  ): readonly GPUCommandNode<Parameters>[] {
    const {id} = this;
    const loss = createWeightsIncompleteNodes(graph, id, [this.props.weights]);
    const props = {...this.props, weights: {...this.props.weights, status: {overflow: loss.view}}};
    const rows = this._getRowCount();
    const columns = props.variableCount;
    const permutations = props.permutationCount ?? 999;
    if (this.getStrategy(rows) === 'tiled') {
      validateGraphViewsBelongToGraph(id, graph, [
        props.values,
        props.mask,
        parameters,
        ...getGPUSpatialWeightsOrChunksViews(props.weights, true),
        props.weights.status?.overflow,
        raw,
        ...Object.values(props.output)
      ]);
      return [
        ...loss.nodes,
        ...getTiledMultivariateLocalGearyNodes(graph, {
          id,
          rows,
          columns,
          permutations,
          parameters,
          classifyDirections,
          tail: props.tail ?? 'equal-tail',
          rowTileSize: Math.min(props.rowTileSize ?? DEFAULT_GEARY_ROW_TILE_SIZE, rows),
          permutationTileSize: Math.min(
            props.permutationTileSize ?? DEFAULT_GEARY_PERMUTATION_TILE_SIZE,
            permutations
          ),
          maximumNeighbors: props.maximumNeighbors ?? DEFAULT_GEARY_MAXIMUM_NEIGHBORS,
          weights: props.weights,
          values: props.values,
          mask: props.mask,
          output: {...props.output, pValues: raw}
        })
      ];
    }
    const valid = createTransientView(graph, `${id}-valid`, 'uint32', rows);
    const standardized = createTransientView(
      graph,
      `${id}-standardized`,
      'float32',
      rows * columns + columns + 1
    );
    const simulations =
      props.output.simulations ??
      createTransientView(graph, `${id}-simulations`, 'float32', rows * permutations);
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' = 'f32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const node = (
      name: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: 'GPUMultivariateLocalGearyPermutationTest',
        variant: name,
        invocationCount,
        bindings,
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}\n${declarations}`,
        body
      });
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.mask,
      parameters,
      ...getGPUSpatialWeightsOrChunksViews(props.weights, true),
      props.weights.status?.overflow,
      ...Object.values(props.output)
    ]);
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const localRow = (dispatch: {isWhole: boolean}) =>
      dispatch.isWhole ? 'row' : '(row - WEIGHTS_ROW_START)';
    const centered = getLocalVariabilityCenteringNodes(graph, {
      id,
      rows,
      columns,
      values: props.values,
      bindings: [binding('valid', valid, 'read', 'u32')],
      declarations: 'fn included(row: u32) -> bool { return valid[validOffset + row] != 0u; }'
    });
    return [
      ...loss.nodes,
      node(
        'complete',
        rows,
        [
          binding('values', props.values),
          ...(props.mask ? [binding('mask', props.mask, 'read', 'u32')] : []),
          binding('valid', valid, 'read_write', 'u32'),
          ...(props.weights.status?.overflow
            ? [binding('overflow', props.weights.status.overflow, 'read', 'u32')]
            : [])
        ],
        `var included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'} ${props.weights.status?.overflow ? '&& overflow[overflowOffset] == 0u' : ''};
for (var column = 0u; column < ${columns}u; column++) { included = included && isFiniteBits(values[valuesOffset + index * ${columns}u + column]); }
valid[validOffset + index] = select(0u, 1u, included);`
      ),
      ...centered.nodes,
      node(
        'standardize',
        columns,
        [
          binding('values', centered.values),
          binding('rawValues', props.values),
          binding('valid', valid, 'read', 'u32'),
          binding('standardized', standardized, 'read_write')
        ],
        `var count = 0u; var mean = 0.0; var squares = 0.0;
for (var row = 0u; row < ${rows}u; row++) { if (valid[validOffset + row] != 0u) { count++; let value = values[valuesOffset + row * ${columns}u + index]; let difference = value - mean; mean += difference / f32(count); squares += difference * (value - mean); } }
var deviation = getNaN(); if (count > 1u && squares > 0.0 && isFiniteBits(squares)) { deviation = sqrt(squares / f32(count)); }
standardized[standardizedOffset + ${rows * columns}u + index] = deviation;
if (index == 0u) { standardized[standardizedOffset + ${rows * columns + columns}u] = 0.0; }
for (var row = 0u; row < ${rows}u; row++) { var value = getNaN(); if (deviation > 0.0 && isFiniteBits(deviation) && valid[validOffset + row] != 0u) { value = rawValues[rawValuesOffset + row * ${columns}u + index]; } standardized[standardizedOffset + row * ${columns}u + index] = value; }`
      ),
      ...dispatches.map(dispatch =>
        node(
          `draws${dispatch.idSuffix}`,
          dispatch.rowCount * (permutations + 1),
          [
            binding('standardized', standardized),
            binding('valid', valid, 'read', 'u32'),
            binding('offsets', dispatch.weights.offsets, 'read', 'u32'),
            binding('neighbors', dispatch.weights.neighbors, 'read', 'u32'),
            binding('weights', dispatch.weights.weights),
            binding('observed', props.output.statistic, 'read_write'),
            binding('simulations', simulations, 'read_write'),
            binding('parameters', parameters, 'read', 'u32')
          ],
          `let row = index / ${permutations + 1}u${dispatch.isWhole ? '' : ' + WEIGHTS_ROW_START'}; let replicate = index % ${permutations + 1}u;
var candidates: array<u32, ${rows}>; var count = 0u;
for (var other = 0u; other < ${rows}u; other++) { if (other != row && valid[validOffset + other] != 0u) { candidates[count] = other; count++; } }
var value = getNaN();
if (valid[validOffset + row] != 0u && count > 0u) {
 var total = 0.0; var weightSum = 0.0; var chosen = 0u;
 var stream = createPhiloxStream(vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]), row, replicate, ${PHILOX_PURPOSES.localPermutationTest.id}u);
 for (var slot = offsets[offsetsOffset + ${localRow(dispatch)}]; slot < offsets[offsetsOffset + ${localRow(dispatch)} + 1u]; slot++) {
  let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
  if (neighbor >= ${rows}u || neighbor == row || !(weight > 0.0)) { continue; }
  if (valid[validOffset + neighbor] == 0u) { continue; }
  var candidate = neighbor;
  if (replicate > 0u) { let selection = chosen + nextPhiloxBelow(&stream, count - chosen); candidate = candidates[selection]; candidates[selection] = candidates[chosen]; candidates[chosen] = candidate; chosen++; }
  var distance = 0.0;
  for (var column = 0u; column < ${columns}u; column++) { let difference = getRoundedDifference(standardized[standardizedOffset + row * ${columns}u + column], standardized[standardizedOffset + candidate * ${columns}u + column], bitcast<u32>(standardized[standardizedOffset + ${rows * columns + columns}u])); let deviation = standardized[standardizedOffset + ${rows * columns}u + column]; var normalized = getNaN(); if (deviation > 0.0 && isFiniteBits(deviation)) { normalized = getScaledQuotient(difference, deviation, 1.0, 0); } distance += normalized * normalized; }
  total += weight * distance / ${columns}.0; weightSum += weight;
 }
 if (weightSum > 0.0) { value = total / weightSum; }
}
if (replicate == 0u) { observed[observedOffset + row] = value; } else { simulations[simulationsOffset + row * ${permutations}u + replicate - 1u] = value; }`,
          `${PERMUTATION_RANDOM_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}\n${SCALED_QUOTIENT_WGSL}${dispatch.isWhole ? '' : `\n${dispatch.constantsWGSL}`}`
        )
      ),
      node(
        'inference',
        rows,
        [
          binding('observed', props.output.statistic),
          binding('simulations', simulations),
          binding('pValues', raw, 'read_write'),
          binding('classes', props.output.classes, 'read_write', 'u32'),
          binding('parameters', parameters, 'read', 'u32')
        ],
        `let observed = observed[observedOffset + index]; var lower = 0u; var upper = 0u; var differenceAverage = 0.0; var finite = isFiniteBits(observed);
for (var replicate = 0u; replicate < ${permutations}u; replicate++) { let value = simulations[simulationsOffset + index * ${permutations}u + replicate]; finite = finite && isFiniteBits(value); lower += select(0u, 1u, value <= observed); upper += select(0u, 1u, value >= observed); if (value != observed) { differenceAverage += (value - observed) / ${permutations}.0; } }
var probability = getNaN(); var classification = 0xffffffffu;
if (finite) { probability = ${props.tail === 'esda-min-tail' ? `f32(min(upper, ${permutations}u - upper) + 1u) / ${permutations + 1}.0` : `min(1.0, 2.0 * f32(min(lower, upper) + 1u) / ${permutations + 1}.0)`};
classification = 0u;
if (probability <= ${classifyDirections ? '0.999999' : 'bitcast<f32>(parameters[parametersOffset + 2u])'}) { classification = select(2u, 1u, differenceAverage > 0.0); }
}
pValues[pValuesOffset + index] = probability; classes[classesOffset + index] = classification;`
      )
    ];
  }
}
