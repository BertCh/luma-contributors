// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {
  getRowTileMergeScratchLength,
  REGRESSION_TILE_MERGE_GROUP_SIZE
} from './regression-row-tiles';
import {
  getAlignedSubView,
  padToAlignedElementCount,
  type ProbeBatches
} from './spatial-maximum-likelihood-nodes';
import {
  getGramLoadWGSL,
  getGramPairwiseSumWGSL,
  SPATIAL_WEIGHTS_GRAM_WGSL
} from './spatial-weights-gram-wgsl';

/** Tuple storage for one probe vector and two Gram moments per tile or probe. @internal */
export function getSpatialWeightsGramProbeLengths(
  rowCount: number,
  tileCount: number,
  probeTotal: number,
  batchProbeCount: number
) {
  const groups = Math.ceil(tileCount / REGRESSION_TILE_MERGE_GROUP_SIZE);
  return {
    'trace norm partials': 3 * tileCount,
    'Gram norm merge groups': getRowTileMergeScratchLength(tileCount, 3),
    'Gram protection': 1,
    'Gram probe vectors': 2 * batchProbeCount * rowCount,
    'Gram probe partials': 4 * batchProbeCount * tileCount,
    'Gram probe groups': groups > 1 ? 4 * batchProbeCount * groups : 0,
    'Gram probe dots':
      4 * Math.ceil(probeTotal / batchProbeCount) * padToAlignedElementCount(batchProbeCount)
  };
}

/** Wide vector, two tile passes, and one or two reductions per batch. @internal */
export function getProtectedGramCommandCount(
  tileCount: number,
  probeTotal: number,
  batchProbeCount: number
): number {
  return (
    Math.ceil(probeTotal / batchProbeCount) *
    (4 + Number(tileCount > REGRESSION_TILE_MERGE_GROUP_SIZE))
  );
}

/** Conservative exponent envelope for the scalar cubic and quartic probe products. @internal */
export const GRAM_PROTECTION_WGSL = /* wgsl */ `
fn needsProtectedGram(
  normBits: u32, inverseMinimum: f32, maximum: f32,
  edgeBound: i32, rowBound: i32, probeBound: i32
) -> bool {
  if (inverseMinimum == 0.0) { return false; }
  let normBiased = normBits >> 23u;
  if (normBiased == 0u || normBiased >= 255u || inverseMinimum == 256.0 || maximum == 255.0) {
    return true;
  }
  let minimumExponent = 129 - i32(inverseMinimum);
  let normExponent = i32(normBiased) - 127;
  let normUpper = normExponent + 1;
  let normalizedLower = minimumExponent - normUpper;
  let weightSumUpper = i32(maximum) - 126 + edgeBound;
  // x=(W/s)z has magnitude at most one per row apart from rounding. The scalar
  // cubic has three normalized factors; the quartic pre-merge has two normalized
  // and two raw factors. Include the norm powers and the tile and probe sums.
  let lower = min(3 * normExponent, min(3 * normalizedLower, 2 * (minimumExponent + normalizedLower)));
  let upper = max(3 * normUpper + rowBound, 2 * (weightSumUpper + normUpper) + probeBound);
  // Leave headroom at both endpoints for roundoff and small cancellation residuals.
  return lower < -114 || upper > 123;
}
`;

/**
 * For x = Wz, the protected cubic probe moment is x'(Wx), and the quartic is ||W'x||².
 * The raw weights and per-value exponents keep these moments independent of the power-series norm.
 * @internal
 */
export function createSpatialWeightsGramProbes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    weights: GPUSpatialWeights;
    transposed: GPUSpatialWeights;
    rowCount: number;
    tileCount: number;
    constants: string;
    probeTotal: number;
    batches: ProbeBatches<Parameters>;
    probeDeclarations: string;
    gramProtection: GraphDataView<'uint32'>;
  }
) {
  const {id, weights, transposed, rowCount, tileCount, constants, probeTotal, batches} = props;
  const {batchProbeCount, blockStride} = batches;
  const groupSize = REGRESSION_TILE_MERGE_GROUP_SIZE;
  const groupCount = Math.ceil(tileCount / groupSize);
  const lengths = getSpatialWeightsGramProbeLengths(
    rowCount,
    tileCount,
    probeTotal,
    batchProbeCount
  );
  const transient = (name: keyof typeof lengths) =>
    createTransientView(
      graph,
      `${id}-${name.toLowerCase().replaceAll(' ', '-')}`,
      'float32',
      lengths[name]
    );
  const vector = transient('Gram probe vectors');
  const partials = transient('Gram probe partials');
  const groups = groupCount > 1 ? transient('Gram probe groups') : undefined;
  const dots = transient('Gram probe dots');
  const csrBindings = (csr: GPUSpatialWeights): WGSLKernelBinding[] => [
    {name: 'offsets', view: csr.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: csr.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: csr.weights, type: 'f32', access: 'read'}
  ];
  const f32 = (name: string, view: GraphDataView<'float32'>, access: 'read' | 'read_write') =>
    ({name, view, type: 'f32', access}) as const;
  const kernel = (
    variant: string,
    batchIndex: number,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    extraDeclarations = ''
  ) =>
    createWGSLKernelNode(graph, {
      id: `${id}-${variant}-${batchIndex}`,
      operation: 'GPUSpatialWeightsTraces',
      variant,
      invocationCount,
      workgroupSize: 64,
      bindings: [
        ...bindings,
        {name: 'gramProtection', view: props.gramProtection, type: 'u32', access: 'read'}
      ],
      declarations: `${constants}\n${SPATIAL_WEIGHTS_GRAM_WGSL}\n${extraDeclarations}`,
      body: `if (gramProtection[gramProtectionOffset] == 0u) { return; }\n${body}`
    });

  return {
    dots,
    getBatchNodes(batchIndex: number): GPUCommandNode<Parameters>[] {
      const batchInformation = {
        name: 'batchInformation',
        view: batches.getBatchInformation(batchIndex),
        type: 'u32',
        access: 'read'
      } as const;
      const dotBlock = getAlignedSubView(
        graph,
        dots,
        4 * batchIndex * blockStride,
        4 * batchProbeCount
      );
      const reduce = (
        groupPass: boolean,
        source: GraphDataView<'float32'>,
        destination: GraphDataView<'float32'>
      ) => {
        const inputCount = groupPass || !groups ? tileCount : groupCount;
        const outputCount = groupPass ? groupCount : 1;
        return kernel(
          groupPass ? 'gram-probe-groups' : 'gram-probe-merge',
          batchIndex,
          2 * batchProbeCount * outputCount,
          [
            f32('items', source, 'read'),
            f32('totals', destination, 'read_write'),
            batchInformation
          ],
          `let probe = index / ${2 * outputCount}u;
  if (probe >= batchInformation[batchInformationOffset + 1u]) { return; }
  let moment = index % 2u;
  let firstItem = ${groupPass ? `((index / 2u) % ${groupCount}u) * ${groupSize}u` : '0u'};
  let lastItem = min(firstItem + ${groupSize}u, ${inputCount}u);
  ${getGramPairwiseSumWGSL(
    'lastItem - firstItem',
    getGramLoadWGSL('items', `(probe * ${inputCount}u + firstItem + gramItem) * 2u + moment`)
  )}
  totals[totalsOffset + index * 2u] = gramSum.x;
  totals[totalsOffset + index * 2u + 1u] = gramSum.y;`
        );
      };
      const momentTiles = (cubic: boolean) =>
        kernel(
          cubic ? 'gram-probe-cubic' : 'gram-probe-moments',
          batchIndex,
          batchProbeCount * tileCount,
          [
            ...csrBindings(cubic ? weights : transposed),
            f32('gramVector', vector, 'read'),
            f32('gramTiles', partials, 'read_write'),
            batchInformation
          ],
          `let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) { return; }
  let firstRow = (index % TILE_COUNT) * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let base = probe * ROWS;
  var moment = vec2<f32>(0.0);
  for (var row = firstRow; row < lastRow; row++) {
    var sum = vec2<f32>(0.0);
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) { continue; }
      let weight = getGramParts(weights[weightsOffset + slot]);
      sum = addGram(sum, multiplyGram(weight, ${getGramLoadWGSL('gramVector', 'base + neighbor')}));
    }
    moment = addGram(moment, multiplyGram(sum, ${cubic ? getGramLoadWGSL('gramVector', 'base + row') : 'sum'}));
  }
  let destination = gramTilesOffset + index * 4u + ${cubic ? '0u' : '2u'};
  gramTiles[destination] = moment.x;
  gramTiles[destination + 1u] = moment.y;`
        );
      return [
        kernel(
          'gram-probe-vector',
          batchIndex,
          batchProbeCount * tileCount,
          [...csrBindings(weights), f32('gramVector', vector, 'read_write'), batchInformation],
          `let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) { return; }
  let globalProbe = batchInformation[batchInformationOffset] + probe;
  let probeSeed = batchInformation[batchInformationOffset + 2u];
  let firstRow = (index % TILE_COUNT) * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var sum = vec2<f32>(0.0);
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) { continue; }
      let weight = getGramParts(weights[weightsOffset + slot]);
      let sign = getProbeValue(probeSeed, globalProbe, neighbor);
      sum = addGram(sum, vec2<f32>(weight.x * sign, weight.y));
    }
    let base = gramVectorOffset + (probe * ROWS + row) * 2u;
    gramVector[base] = sum.x;
    gramVector[base + 1u] = sum.y;
  }`,
          props.probeDeclarations
        ),
        momentTiles(true),
        momentTiles(false),
        ...(groups
          ? [reduce(true, partials, groups), reduce(false, groups, dotBlock)]
          : [reduce(false, partials, dotBlock)])
      ];
    }
  };
}
