// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPolygonStudyWindowViews} from '../../analysis-core/contracts/study-window';
import {getStudyWindowBinding} from '../../analysis-core/kernels/study-window-wgsl';
import {getKernelDensityStudyWindowWGSL} from './kernel-density-wgsl';
import {
  getDensityDomainConstantsWGSL,
  DENSITY_RECTANGLE_MEMBERSHIP_WGSL
} from './density-neighbor-wgsl';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createDensityPointPass} from './density-point-passes';
import {
  createDensityScalarScratch,
  createDensityExtremumNodes,
  createDensityInvalidCountNodes,
  createDensityWeightSumNodes,
  needsDensitySplitScratch
} from './density-scratch';
import {
  getGPUKernelDensityCanonicalBandwidthScale,
  type GPUKernelDensityBandwidthRule,
  type GPUKernelDensityKernel
} from './kernel-density-math';

export type GPUKernelDensityBandwidthPopulation = 'included' | 'all-finite';

const OPERATION = 'GPUKernelDensity';

/** Existing construction-time requirements of fixed KDE's optional automatic bandwidth. */
export function validateKernelDensityBandwidthRule(
  id: string,
  rule: GPUKernelDensityBandwidthRule | undefined,
  bandwidth: GraphDataView<'float32'> | undefined
): void {
  validateRequirement(
    id,
    rule === undefined || ['scott', 'silverman'].includes(rule),
    'bandwidthRule must be scott or silverman'
  );
  validateRequirement(id, !bandwidth || Boolean(rule), 'output.bandwidth needs bandwidthRule');
  if (bandwidth) {
    validateSinglePackedView(bandwidth, ['float32'], `${id} output.bandwidth`);
    validateRequirement(
      id,
      bandwidth.length === 1,
      'output.bandwidth must contain exactly one float32 row'
    );
  }
}

/** Two centered passes and deterministic fixed-point moments for Scott/Silverman bandwidths.
 * Input rows and eligibility are unchanged: finite positions and finite nonnegative weights,
 * with the index-domain and edge-window filter only for bandwidthPopulation: 'included'. A shared eligible-coordinate
 * minimum among positive-weight rows is the origin for both passes, independent of output-grid placement.
 */
export function createKernelDensityBandwidthNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  inputs: {
    id: string;
    kernel: GPUKernelDensityKernel;
    bandwidthState: GraphDataView<'float32'>;
    populationCount?: GraphDataView<'uint32'>;
    studyWindow?: GPUPolygonStudyWindowViews;
    edgeCorrection: 'none' | 'uniform' | 'diggle';
    props: {
      positions: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
      weights?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
      parameters: GraphDataView<'float32'>;
      indexBounds: readonly [number, number, number, number];
      bandwidthPopulation?: 'included' | 'all-finite';
    };
  }
): GPUCommandNode<Parameters>[] {
  const {id, props, kernel, bandwidthState, populationCount, studyWindow, edgeCorrection} = inputs;
  const includedPopulation = props.bandwidthPopulation === 'included';
  const rectangleWindow = includedPopulation && !studyWindow && edgeCorrection !== 'none';
  const populationDeclarations = includedPopulation
    ? `${getDensityDomainConstantsWGSL(props.indexBounds)}${IS_FINITE_VALUE_WGSL}${rectangleWindow ? DENSITY_RECTANGLE_MEMBERSHIP_WGSL : ''}${studyWindow ? getKernelDensityStudyWindowWGSL(studyWindow) : ''}`
    : IS_FINITE_VALUE_WGSL;
  const splitPointScratch = needsDensitySplitScratch(graph, props.positions.length);
  const terms = ['weight', 'weight-square', 'x', 'y'].map(name =>
    createDensityScalarScratch(
      graph,
      `${id}-bandwidth-term-${name}`,
      props.positions,
      'float32',
      !splitPointScratch
    )
  );
  const [termWeight, termWeightSquare, termX, termY] = terms;
  const termChunks = terms.map(term => getGraphViewChunks(term));
  const sums = ['weight', 'weight-square', 'x', 'y', 'spread-x', 'spread-y'].map(name =>
    createTransientView(graph, `${id}-bandwidth-sum-${name}`, 'float32', 1)
  );
  const [sumWeight, sumWeightSquare, sumX, sumY, sumSpreadX, sumSpreadY] = sums;
  const momentControls = createTransientView(
    graph,
    `${id}-bandwidth-moment-controls`,
    'float32',
    3
  );
  const controlView = (index: number) =>
    graph.createDataView(momentControls.buffer, {
      format: 'float32',
      byteOffset: momentControls.byteOffset + index * 4,
      length: 1
    });
  const maximumWeight = controlView(2);
  const populationRows = populationCount
    ? createDensityScalarScratch(
        graph,
        `${id}-bandwidth-population-rows`,
        props.positions,
        'uint32',
        !splitPointScratch
      )
    : undefined;
  const populationChunks = populationRows ? getGraphViewChunks(populationRows) : undefined;
  const means = createTransientView(graph, `${id}-bandwidth-means`, 'float32', 2);
  const originX = controlView(0);
  const originY = controlView(1);
  const positionChunks = getGraphViewChunks(props.positions);
  const weightChunks = props.weights ? getGraphViewChunks(props.weights) : undefined;
  const inputBindings = (chunkIndex: number, withOrigin = true): WGSLKernelBinding[] => [
    {name: 'positions', view: positionChunks[chunkIndex], type: 'f32', access: 'read'},
    ...(weightChunks
      ? [
          {
            name: 'inputWeights',
            view: weightChunks[chunkIndex],
            type: 'f32',
            access: 'read'
          } as const
        ]
      : []),
    ...(rectangleWindow
      ? [{name: 'params', view: props.parameters, type: 'f32', access: 'read'} as const]
      : []),
    ...(includedPopulation && studyWindow ? [getStudyWindowBinding(studyWindow)] : []),
    ...(withOrigin
      ? [{name: 'momentControls', view: momentControls, type: 'f32', access: 'read'} as const]
      : [])
  ];
  let rowStart = 0;
  // Rows with a non-finite position or a non-finite or negative weight contribute nothing.
  const rowEligibility = `let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
let rawWeight = ${props.weights ? 'inputWeights[inputWeightsOffset + index]' : '1.0'};
let included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
  isFiniteValue(rawWeight) && rawWeight >= 0.0
  ${includedPopulation ? '&& all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX)' : ''}
  ${includedPopulation && studyWindow ? '&& isInsideStudyWindow(position)' : rectangleWindow ? '&& isInsideDensityRectangle(position, vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]), vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]))' : ''};`;
  const rowPrelude = `let origin = vec2f(momentControls[momentControlsOffset], momentControls[momentControlsOffset + 1u]);
${rowEligibility}
let weight = select(0.0, ldexp(rawWeight, -frexp(momentControls[momentControlsOffset + 2u]).exp), included);
let relative = select(vec2f(0.0), position - origin, included);`;
  const variantSuffix = `${props.weights ? 'weighted' : 'unweighted'}-${props.bandwidthPopulation ?? 'all-finite'}`;
  // Reuse x/y term scratch for eligible coordinate minima before the moment passes.
  // One shared data-relative origin makes the rule independent of output-grid placement.
  let minimumRowStart = 0;
  const originNodes: GPUCommandNode<Parameters>[] = [
    ...positionChunks.flatMap((chunk, chunkIndex) => {
      const start = minimumRowStart;
      minimumRowStart += chunk.length;
      return chunk.length === 0
        ? []
        : [
            createDensityPointPass(graph, {
              id: `${id}-bandwidth-origin-terms${positionChunks.length === 1 ? '' : `-${chunkIndex}`}`,
              operation: OPERATION,
              variant: `bandwidth-origin-${variantSuffix}`,
              bindings: [
                ...inputBindings(chunkIndex, false),
                {
                  name: 'termWeight',
                  view: termChunks[0][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                },
                ...(populationChunks
                  ? [
                      {
                        name: 'populationRows',
                        view: populationChunks[splitPointScratch ? chunkIndex : 0],
                        type: 'u32',
                        access: 'read_write'
                      } as const
                    ]
                  : []),
                {
                  name: 'termX',
                  view: termChunks[2][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                },
                {
                  name: 'termY',
                  view: termChunks[3][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                }
              ],
              invocationCount: chunk.length,
              runtimeConstants: {ROW_START: splitPointScratch ? 0 : start},
              declarations: populationDeclarations,
              body: `${rowEligibility}
  termWeight[termWeightOffset + ROW_START + index] = select(0.0, rawWeight, included);
  ${populationRows ? 'populationRows[populationRowsOffset + ROW_START + index] = select(0u, 1u, included);' : ''}
  termX[termXOffset + ROW_START + index] = select(3.402823466e38, position.x, included && rawWeight > 0.0);
  termY[termYOffset + ROW_START + index] = select(3.402823466e38, position.y, included && rawWeight > 0.0);`
            })
          ];
    }),
    ...createDensityExtremumNodes(
      graph,
      `${id}-bandwidth-maximum-weight`,
      termWeight,
      maximumWeight,
      'max'
    ),
    ...(populationRows && populationCount
      ? createDensityInvalidCountNodes(
          graph,
          `${id}-bandwidth-population-count`,
          populationRows,
          populationCount
        )
      : []),
    ...createDensityExtremumNodes(graph, `${id}-bandwidth-origin-min-x`, termX, originX, 'min'),
    ...createDensityExtremumNodes(graph, `${id}-bandwidth-origin-min-y`, termY, originY, 'min')
  ];
  const nodes: GPUCommandNode<Parameters>[] = [
    ...originNodes,
    ...positionChunks.flatMap((chunk, chunkIndex) => {
      const start = rowStart;
      rowStart += chunk.length;
      return chunk.length === 0
        ? []
        : [
            createDensityPointPass<Parameters>(graph, {
              id: `${id}-bandwidth-terms${positionChunks.length === 1 ? '' : `-${chunkIndex}`}`,
              operation: OPERATION,
              variant: `bandwidth-terms-${variantSuffix}`,
              bindings: [
                ...inputBindings(chunkIndex),
                {
                  name: 'termWeight',
                  view: termChunks[0][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                },
                {
                  name: 'termWeightSquare',
                  view: termChunks[1][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                },
                {
                  name: 'termX',
                  view: termChunks[2][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                },
                {
                  name: 'termY',
                  view: termChunks[3][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                }
              ],
              invocationCount: chunk.length,
              runtimeConstants: {ROW_START: splitPointScratch ? 0 : start},
              declarations: populationDeclarations,
              body: `${rowPrelude}
termWeight[termWeightOffset + ROW_START + index] = weight;
termWeightSquare[termWeightSquareOffset + ROW_START + index] = weight * weight;
termX[termXOffset + ROW_START + index] = weight * relative.x;
termY[termYOffset + ROW_START + index] = weight * relative.y;`
            })
          ];
    })
  ];
  for (const [name, input, output] of [
    ['weight', termWeight, sumWeight],
    ['weight-square', termWeightSquare, sumWeightSquare],
    ['x', termX, sumX],
    ['y', termY, sumY]
  ] as const) {
    nodes.push(...createDensityWeightSumNodes(graph, `${id}-bandwidth-sum-${name}`, input, output));
  }
  rowStart = 0;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-bandwidth-means`,
      operation: OPERATION,
      variant: 'bandwidth-means',
      bindings: [
        {name: 'sumWeight', view: sumWeight, type: 'f32', access: 'read'},
        {name: 'sumX', view: sumX, type: 'f32', access: 'read'},
        {name: 'sumY', view: sumY, type: 'f32', access: 'read'},
        {name: 'means', view: means, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `let total = sumWeight[sumWeightOffset];
let valid = total > 0.0;
means[meansOffset] = select(0.0, sumX[sumXOffset] / total, valid);
means[meansOffset + 1u] = select(0.0, sumY[sumYOffset] / total, valid);`
    }),
    // Second pass: centered terms overwrite the first-pass x and y term buffers, which the
    // reductions above have already consumed (nodes execute in order).
    ...positionChunks.flatMap((chunk, chunkIndex) => {
      const start = rowStart;
      rowStart += chunk.length;
      return chunk.length === 0
        ? []
        : [
            createDensityPointPass<Parameters>(graph, {
              id: `${id}-bandwidth-spread${positionChunks.length === 1 ? '' : `-${chunkIndex}`}`,
              operation: OPERATION,
              variant: `bandwidth-spread-${variantSuffix}`,
              bindings: [
                ...inputBindings(chunkIndex),
                {name: 'means', view: means, type: 'f32', access: 'read'},
                {
                  name: 'termX',
                  view: termChunks[2][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                },
                {
                  name: 'termY',
                  view: termChunks[3][splitPointScratch ? chunkIndex : 0],
                  type: 'f32',
                  access: 'read_write'
                }
              ],
              invocationCount: chunk.length,
              runtimeConstants: {ROW_START: splitPointScratch ? 0 : start},
              declarations: populationDeclarations,
              body: `${rowPrelude}
let centered = relative - vec2f(means[meansOffset], means[meansOffset + 1u]);
termX[termXOffset + ROW_START + index] = select(0.0, weight * centered.x * centered.x, included);
termY[termYOffset + ROW_START + index] = select(0.0, weight * centered.y * centered.y, included);`
            })
          ];
    })
  );
  for (const [name, input, output] of [
    ['spread-x', termX, sumSpreadX],
    ['spread-y', termY, sumSpreadY]
  ] as const) {
    nodes.push(...createDensityWeightSumNodes(graph, `${id}-bandwidth-sum-${name}`, input, output));
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-bandwidth-rule`,
      operation: OPERATION,
      variant: `bandwidth-rule-${kernel}`,
      bindings: [
        {name: 'sumWeight', view: sumWeight, type: 'f32', access: 'read'},
        {name: 'sumWeightSquare', view: sumWeightSquare, type: 'f32', access: 'read'},
        {name: 'sumSpreadX', view: sumSpreadX, type: 'f32', access: 'read'},
        {name: 'sumSpreadY', view: sumSpreadY, type: 'f32', access: 'read'},
        {name: 'bandwidthState', view: bandwidthState, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${IS_FINITE_VALUE_WGSL}${GET_NAN_WGSL}`,
      // scipy: scotts_factor = n_eff^(-1/(d+4)); silverman_factor = (n_eff (d+2)/4)^(-1/(d+4));
      // both n_eff^(-1/6) for d = 2. Covariance is the unbiased weighted one: the weighted
      // spread divided by 1 - sum(w^2) / sum(w)^2.
      body: `let total = sumWeight[sumWeightOffset];
let totalSquare = sumWeightSquare[sumWeightSquareOffset];
let effective = total * total / totalSquare;
let correction = 1.0 - totalSquare / (total * total);
let variance = 0.5 * (sumSpreadX[sumSpreadXOffset] + sumSpreadY[sumSpreadYOffset]) / (total * correction);
let bandwidth = ${getGPUKernelDensityCanonicalBandwidthScale(kernel)} * pow(effective, -1.0 / 6.0) * sqrt(variance);
let valid = total > 0.0 && totalSquare > 0.0 && effective > 1.0 && correction > 0.0 &&
  isFiniteValue(bandwidth) && bandwidth > 0.0;
bandwidthState[bandwidthStateOffset] = select(getNaN(), bandwidth, valid);`
    })
  );
  return nodes;
}
