// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
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

export type GPUMultivariateLocalGearyPermutationTestProps = {
  id?: string;
  /** Whole CSR or consecutive row chunks; chunks are dispatched one at a time, never repacked. */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Row-major variables; a common complete-case mask applies to all variables. */
  values: GraphDataView<'float32'>;
  variableCount: number;
  mask?: GraphDataView<'uint32'>;
  maximumPermutations?: number;
  seed?: readonly [number, number];
  significance?: number;
  /** Bonferroni family is every included row (including islands); default no adjustment. */
  correction?: 'none' | 'bonferroni';
  /**
   * Opt-in family-wise or false-discovery-rate correction of the pseudo p-values through
   * `GPUMultipleTestingAdjustment`: `'bonferroni' | 'holm' | 'benjamini-hochberg' |
   * 'benjamini-yekutieli'` (use `'benjamini-yekutieli'` for local statistics, whose p-values are
   * dependent). `output.pValues` then holds the adjusted p-value and `output.classes` marks a row
   * significant when the adjusted p-value is at most `significance`, in the direction of the
   * uncorrected test. The family is every row with a finite p-value (islands and rows beyond
   * `maximumNeighbors` are untested), unlike `correction: 'bonferroni'`, whose family is every
   * included row. Mutually exclusive with `correction: 'bonferroni'`. Default `'none'`: the
   * classes use the uncorrected (or `correction`) p-value, which paints about `significance` of
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
    pValues: GraphDataView<'float32'>;
    /** 0 nonsignificant, 1 significant similarity, 2 significant dissimilarity, 0xffffffff invalid. */
    classes: GraphDataView<'uint32'>;
    /** Optional row-major replay of every joint-vector conditional randomization. */
    simulations?: GraphDataView<'float32'>;
  };
};

/**
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
  constructor(readonly props: GPUMultivariateLocalGearyPermutationTestProps) {
    this.id = props.id ?? 'multivariate-local-geary-permutation-test';
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
      'maximumPermutations',
      props.maximumPermutations ?? 999,
      1,
      tiled ? TILED_MULTIVARIATE_GEARY_MAXIMUM_PERMUTATIONS : 4096
    );
    if (tiled) {
      if (props.output.simulations) throw new Error(`${this.id}: tiled has no simulations output`);
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
    if (
      resolveGPUMultipleTestingMethod(this.id, props.multipleTesting) !== 'none' &&
      props.correction === 'bonferroni'
    ) {
      throw new Error(`${this.id}: use either correction or multipleTesting`);
    }
    if (props.tail && !['equal-tail', 'esda-min-tail'].includes(props.tail))
      throw new Error(`${this.id}: invalid tail convention`);
    if (
      (props.seed ?? [0, 0]).length !== 2 ||
      !(props.seed ?? [0, 0]).every(
        value => Number.isInteger(value) && value >= 0 && value <= 0xffffffff
      ) ||
      !((props.significance ?? 0.05) > 0 && (props.significance ?? 0.05) < 1) ||
      !['none', 'bonferroni'].includes(props.correction ?? 'none')
    )
      throw new Error(`${this.id}: invalid inference settings`);
    for (const [view, format, length] of [
      [props.values, 'float32', rows * props.variableCount],
      [props.mask, 'uint32', rows],
      [props.output.statistic, 'float32', rows],
      [props.output.pValues, 'float32', rows],
      [props.output.classes, 'uint32', rows],
      [props.output.simulations, 'float32', rows * (props.maximumPermutations ?? 999)]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.values,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow
    ]);
  }
  private _getRowCount(): number {
    const {weights} = this.props;
    return 'chunks' in weights ? weights.rowCount : weights.offsets.length - 1;
  }
  /** Resolves the requested strategy against the row count. */
  getStrategy(rows = this._getRowCount()): 'exact' | 'tiled' {
    return (
      this.props.strategy ??
      (rows > EXACT_MULTIVARIATE_GEARY_MAXIMUM_ROWS ||
      (this.props.maximumPermutations ?? 999) > 4096
        ? 'tiled'
        : 'exact')
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = this._getRowCount();
    const columns = props.variableCount;
    const permutations = props.maximumPermutations ?? 999;
    const method = resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    if (method !== 'none') {
      // Uncorrected test with every finite row classified in its direction (significance just
      // below 1), then the family adjustment, then the classes at the requested level.
      const significance = props.significance ?? 0.05;
      const rawPValues = createTransientView(graph, `${id}-raw-p-values`, 'float32', rows);
      const directions = createTransientView(graph, `${id}-directions`, 'uint32', rows);
      const inner = new GPUMultivariateLocalGearyPermutationTest({
        ...props,
        id: `${id}-uncorrected`,
        multipleTesting: undefined,
        correction: 'none',
        significance: 0.999999,
        output: {...props.output, pValues: rawPValues, classes: directions}
      });
      return [
        ...inner.getCommandNodes(graph),
        ...new GPUMultipleTestingAdjustment({
          id: `${id}-multiple-testing`,
          pValues: rawPValues,
          method,
          output: {adjustedPValues: props.output.pValues}
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify-adjusted`,
          operation: 'GPUMultivariateLocalGearyPermutationTest',
          variant: 'classify-adjusted',
          invocationCount: rows,
          bindings: [
            {name: 'directions', view: directions, type: 'u32', access: 'read'},
            {name: 'adjusted', view: props.output.pValues, type: 'f32', access: 'read'},
            {name: 'classes', view: props.output.classes, type: 'u32', access: 'read_write'}
          ],
          body: `let direction = directions[directionsOffset + index];
  classes[classesOffset + index] = select(
    direction,
    select(0u, direction, adjusted[adjustedOffset + index] <= ${getWGSLFloatLiteral(significance)}),
    direction != 0xffffffffu
  );`
        })
      ];
    }
    if (this.getStrategy(rows) === 'tiled') {
      validateGraphViewsBelongToGraph(id, graph, [
        props.values,
        props.mask,
        ...getGPUSpatialWeightsOrChunksViews(props.weights),
        props.weights.status?.overflow,
        ...Object.values(props.output)
      ]);
      return getTiledMultivariateLocalGearyNodes(graph, {
        id,
        rows,
        columns,
        permutations,
        seed: props.seed ?? [0, 0],
        significance: props.significance ?? 0.05,
        correction: props.correction ?? 'none',
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
        output: props.output
      });
    }
    const valid = createTransientView(graph, `${id}-valid`, 'uint32', rows);
    const standardized = createTransientView(
      graph,
      `${id}-standardized`,
      'float32',
      rows * columns
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
      ...getGPUSpatialWeightsOrChunksViews(props.weights),
      props.weights.status?.overflow,
      ...Object.values(props.output)
    ]);
    const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
    const localRow = (dispatch: {isWhole: boolean}) =>
      dispatch.isWhole ? 'row' : '(row - WEIGHTS_ROW_START)';
    return [
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
      node(
        'standardize',
        columns,
        [
          binding('values', props.values),
          binding('valid', valid, 'read', 'u32'),
          binding('standardized', standardized, 'read_write')
        ],
        `var count = 0u; var mean = 0.0; var squares = 0.0;
for (var row = 0u; row < ${rows}u; row++) { if (valid[validOffset + row] != 0u) { count++; let value = values[valuesOffset + row * ${columns}u + index]; let difference = value - mean; mean += difference / f32(count); squares += difference * (value - mean); } }
for (var row = 0u; row < ${rows}u; row++) { var value = getNaN(); if (count > 1u && squares > 0.0 && isFiniteBits(squares) && isFiniteBits(mean) && valid[validOffset + row] != 0u) { value = (values[valuesOffset + row * ${columns}u + index] - mean) * sqrt(f32(count) / squares); } standardized[standardizedOffset + row * ${columns}u + index] = value; }`
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
            binding('simulations', simulations, 'read_write')
          ],
          `let row = index / ${permutations + 1}u${dispatch.isWhole ? '' : ' + WEIGHTS_ROW_START'}; let replicate = index % ${permutations + 1}u;
var candidates: array<u32, ${rows}>; var count = 0u;
for (var other = 0u; other < ${rows}u; other++) { if (other != row && valid[validOffset + other] != 0u) { candidates[count] = other; count++; } }
var value = getNaN();
if (valid[validOffset + row] != 0u && count > 0u) {
 var total = 0.0; var weightSum = 0.0; var chosen = 0u;
 var stream = createPhiloxStream(vec2<u32>(${props.seed?.[0] ?? 0}u, ${props.seed?.[1] ?? 0}u), row, replicate, ${PHILOX_PURPOSES.localPermutationTest.id}u);
 for (var slot = offsets[offsetsOffset + ${localRow(dispatch)}]; slot < offsets[offsetsOffset + ${localRow(dispatch)} + 1u]; slot++) {
  let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
  if (neighbor >= ${rows}u || neighbor == row || !(weight > 0.0)) { continue; }
  if (valid[validOffset + neighbor] == 0u) { continue; }
  var candidate = neighbor;
  if (replicate > 0u) { let selection = chosen + nextPhiloxBelow(&stream, count - chosen); candidate = candidates[selection]; candidates[selection] = candidates[chosen]; candidates[chosen] = candidate; chosen++; }
  var distance = 0.0;
  for (var column = 0u; column < ${columns}u; column++) { let difference = standardized[standardizedOffset + row * ${columns}u + column] - standardized[standardizedOffset + candidate * ${columns}u + column]; distance += difference * difference; }
  total += weight * distance / ${columns}.0; weightSum += weight;
 }
 if (weightSum > 0.0) { value = total / weightSum; }
}
if (replicate == 0u) { observed[observedOffset + row] = value; } else { simulations[simulationsOffset + row * ${permutations}u + replicate - 1u] = value; }`,
          `${PERMUTATION_RANDOM_WGSL}${dispatch.isWhole ? '' : `\n${dispatch.constantsWGSL}`}`
        )
      ),
      node(
        'inference',
        rows,
        [
          binding('observed', props.output.statistic),
          binding('simulations', simulations),
          binding('valid', valid, 'read', 'u32'),
          binding('pValues', props.output.pValues, 'read_write'),
          binding('classes', props.output.classes, 'read_write', 'u32')
        ],
        `let observed = observed[observedOffset + index]; var lower = 0u; var upper = 0u; var average = 0.0; var finite = isFiniteBits(observed);
for (var replicate = 0u; replicate < ${permutations}u; replicate++) { let value = simulations[simulationsOffset + index * ${permutations}u + replicate]; finite = finite && isFiniteBits(value); lower += select(0u, 1u, value <= observed); upper += select(0u, 1u, value >= observed); average += value / ${permutations}.0; }
var probability = getNaN(); var classification = 0xffffffffu;
if (finite) { probability = ${props.tail === 'esda-min-tail' ? `f32(min(upper, ${permutations}u - upper) + 1u) / ${permutations + 1}.0` : `min(1.0, 2.0 * f32(min(lower, upper) + 1u) / ${permutations + 1}.0)`}; var family = 1u;
${props.correction === 'bonferroni' ? `family = 0u; for (var row = 0u; row < ${rows}u; row++) { family += valid[validOffset + row]; }` : ''}
probability = min(1.0, probability * f32(family)); classification = 0u;
if (probability <= ${getWGSLFloatLiteral(props.significance ?? 0.05)}) { classification = select(2u, 1u, observed < average); }
}
pValues[pValuesOffset + index] = probability; classes[classesOffset + index] = classification;`
      )
    ];
  }
}
