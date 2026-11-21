import {snapshotWeightsDescription} from '../spatial-weights/weights-construction-snapshot';
// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveRecipeDestinations} from './recipe-utils';
import {
  createWeightsLossNodes,
  getWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUEmpiricalBayesRates, GPUSpatialEmpiricalBayesRates} from '../rate-smoothing/index';
import type {GPUFloat32OrUint32Values} from '../spatial-weights/float32-values';
import {createFloat32ValuesNodes} from '../spatial-weights/float32-values';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {getGPUSpatialWeightsNeighborLoopWGSL} from '../spatial-weights/spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  wrapGPUSpatialWeightsChunkBody,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {addPreparedRecipe} from './prepared-recipe';
import {
  addRecipeCountNodes,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeResult
} from './recipe-utils';

type RateRecipeInputs = {
  /** Unique prefix when composing multiple recipes. */
  id?: string;
  /** Event counts, converted to float32 on the GPU when uint32. */
  events: GPUFloat32OrUint32Values;
  /** Positive population at risk. Invalid rows are excluded. */
  populations: GPUFloat32OrUint32Values;
  /** Nonzero includes a row as both focus and neighbor. */
  mask?: GraphDataView<'uint32'>;
};

/** Inputs for dimensionless pooled excess risk. */
export type GPUExcessRiskRecipeProps = RateRecipeInputs & {
  /** Optional caller-owned terminal output; otherwise graph-owned. */
  output?: GraphDataView<'float32'>;
};

/** Inputs for binary-neighborhood pooled spatial rates. */
export type GPUSpatialRateRecipeProps = RateRecipeInputs & {
  /** Positive-weight slots select neighbors; magnitudes are ignored and zero-weight slots excluded. */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  /** Compile-time rate multiplier, e.g. 100000. Defaults to 1. */
  scale?: number;
  /** GeoDa marks islands undefined. 'own-rate' retains the existing spatial EB convention. */
  islandPolicy?: 'undefined' | 'own-rate';
  /** Optional caller-owned terminal output; otherwise graph-owned. */
  output?: GraphDataView<'float32'>;
};

/**
 * Appends observed/expected excess risk E_i / (P_i * sum(E) / sum(P)).
 * Uses the existing empirical-Bayes reductions without changing its other outputs. Zero or
 * negative expected counts produce NaN; events must be finite and nonnegative and populations
 * finite and positive. The optional mask excludes rows from both totals and outputs.
 * No rate scale applies: the result is dimensionless. Inputs are packed graph views, not vectors
 * to concatenate; caller owns imported chunks. No submission or readback occurs.
 */
export function addExcessRiskRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUExcessRiskRecipeProps
): GPURecipeResult<{excessRisk: GraphDataView<'float32'>}> {
  return addPreparedRecipe(graph, props, declareExcessRiskRecipe);
}

function declareExcessRiskRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUExcessRiskRecipeProps
): GPURecipeResult<{excessRisk: GraphDataView<'float32'>}> {
  const id = props.id ?? 'excess-risk';
  const destinations = resolveRecipeDestinations(id, props);
  const excessRisk = getOrCreateView(
    graph,
    `${id}-output`,
    'float32',
    props.events.length,
    destinations
  );
  const builder = new RecipeBuilder(graph, true);
  builder.add(
    new GPUEmpiricalBayesRates({
      events: props.events,
      populations: props.populations,
      mask: props.mask,
      id,
      output: {excessRisk}
    })
  );
  // The contributor has no status port. It writes NaN to every excluded row, including rows the
  // mask deselects on purpose, so only rows the mask includes can be invalid here: those with an
  // undefined rate (zero or negative expected count, or unusable events or population).
  const invalidCount = createTransientView(graph, `${id}-invalid`, 'uint32', 1);
  addRecipeCountNodes(builder, {
    id: `${id}-invalid-rows`,
    operation: 'addExcessRiskRecipe',
    rowCount: props.events.length,
    declarations: IS_FINITE_BITS_WGSL,
    bindings: [
      {name: 'excessRisk', view: excessRisk, type: 'f32', access: 'read'},
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32', access: 'read'} as const]
        : [])
    ],
    counts: [
      {
        name: 'invalid',
        output: invalidCount,
        value: `u32(${props.mask ? 'mask[maskOffset + index] != 0u && ' : ''}!isFiniteBits(excessRisk[excessRiskOffset + index]))`
      }
    ]
  });
  builder.addDeferredStages();
  return {
    contributors: builder.contributors,
    outputs: {excessRisk},
    intermediates: {},
    status: {stages: [{stage: 'excess-risk', status: {invalidCount}}]}
  };
}

/**
 * Appends scale * (E_i + sum neighbors E) / (P_i + sum neighbors P), never an average of rates.
 * Included neighbors have finite nonnegative events, finite positive populations, nonzero mask
 * and positive stored weight (magnitude is ignored); self edges
 * are skipped because self is already included once. Duplicate nonself slots count repeatedly.
 * Islands (no included nonself neighbor) produce NaN by default, matching GeoDa's undefined flag.
 * The underlying spatial EB contributor retains its own-rate island behavior unchanged.
 */
export function addSpatialRateRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSpatialRateRecipeProps
): GPURecipeResult<
  {spatialRates: GraphDataView<'float32'>},
  {pooledRates: GraphDataView<'float32'>}
> {
  return addPreparedRecipe(graph, props, declareSpatialRateRecipe);
}

function declareSpatialRateRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSpatialRateRecipeProps
): GPURecipeResult<
  {spatialRates: GraphDataView<'float32'>},
  {pooledRates: GraphDataView<'float32'>}
> {
  const id = props.id ?? 'spatial-rate';
  const destinations = resolveRecipeDestinations(id, props);
  validateRequirement(
    id,
    props.islandPolicy === undefined ||
      props.islandPolicy === 'undefined' ||
      props.islandPolicy === 'own-rate',
    `islandPolicy must be undefined or own-rate; received ${props.islandPolicy}`
  );
  const scale = props.scale ?? 1;
  if (!Number.isFinite(Math.fround(scale)) || scale < 0) {
    throw new Error(`${id} scale must be finite and nonnegative`);
  }
  const rows = props.events.length;
  const spatialRates = getOrCreateView(graph, `${id}-output`, 'float32', rows, destinations);
  validateSinglePackedView(spatialRates, ['float32'], `${id} output`);
  if (spatialRates.length !== rows) throw new Error(`${id} output length must equal events length`);
  validateGraphOutputsDisjointFromInputs(
    id,
    [spatialRates],
    [
      props.events,
      props.populations,
      props.mask,
      ...getGPUSpatialWeightsOrChunksViews(props.weights)
    ]
  );
  validateGraphViewsBelongToGraph(id, graph, [spatialRates]);
  const pooledRates = createTransientView(graph, `${id}-pooled`, 'float32', rows);
  const builder = new RecipeBuilder(graph, true);
  builder.add(
    new GPUSpatialEmpiricalBayesRates({
      events: props.events,
      populations: props.populations,
      mask: props.mask,
      weights: props.weights,
      id,
      output: {spatialRates: pooledRates}
    })
  );
  // Conversion helpers preserve GPU residency, including uint32 inputs. The final pass uses no
  // positive-weight eligibility and at most eight storage bindings, including the optional mask.
  const eventValues = createFloat32ValuesNodes(graph, `${id}-island-events`, props.events);
  const populationValues = createFloat32ValuesNodes(
    graph,
    `${id}-island-populations`,
    props.populations
  );
  const nodes = [...eventValues.nodes, ...populationValues.nodes];
  const mask = props.mask;
  nodes.push(
    ...getGPUSpatialWeightsChunkDispatches(props.weights).map(dispatch => {
      return createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish${dispatch.idSuffix}`,
        operation: 'addSpatialRateRecipe',
        variant: 'scale-islands',
        bindings: [
          {name: 'rates', view: pooledRates, type: 'f32', access: 'read'},
          {name: 'events', view: eventValues.view, type: 'f32', access: 'read'},
          {name: 'populations', view: populationValues.view, type: 'f32', access: 'read'},
          {name: 'offsets', view: dispatch.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: dispatch.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: dispatch.weights.weights, type: 'f32', access: 'read'},
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'output', view: spatialRates, type: 'f32', access: 'read_write'}
        ],
        invocationCount: dispatch.rowCount,
        declarations: `const ROWS: u32 = ${rows}u;\nconst CAPACITY: u32 = ${dispatch.weights.neighbors.length}u;\n${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}\n${dispatch.constantsWGSL}`,
        body: wrapGPUSpatialWeightsChunkBody(
          dispatch,
          `var hasNeighbor = ${props.islandPolicy === 'own-rate' ? 'true' : 'false'};
  ${getGPUSpatialWeightsNeighborLoopWGSL(
    `${mask ? 'mask[maskOffset + neighbor] != 0u && ' : ''}isFiniteFloat(events[eventsOffset + neighbor]) && events[eventsOffset + neighbor] >= 0.0 && isFiniteFloat(populations[populationsOffset + neighbor]) && populations[populationsOffset + neighbor] > 0.0`,
    'hasNeighbor = true; break;',
    'index',
    dispatch.isWhole ? undefined : 'WEIGHTS_ROW_START'
  )}
  output[outputOffset + index] = select(getQuietNaN(index), rates[ratesOffset + index] * ${getWGSLFloatLiteral(scale)}, hasNeighbor);`
        )
      });
    })
  );
  builder.addGlue('island-scaling', nodes);
  const losses = getWeightsLossSources([props.weights]);
  const weightsStatus = {
    ...snapshotWeightsDescription(props.weights).status,
    overflow:
      losses.overflow.length === 1
        ? losses.overflow[0]
        : createTransientView(graph, `${id}-weights-overflow`, 'uint32', 1),
    candidateOverflow:
      losses.candidateOverflow.length === 1
        ? losses.candidateOverflow[0]
        : createTransientView(graph, `${id}-weights-candidate-overflow`, 'uint32', 1)
  };
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    if (losses[name].length === 1) continue;
    builder.addGlue(
      `weights-${name}`,
      createWeightsLossNodes(graph, `${id}-weights-${name}`, losses[name], weightsStatus[name])
    );
  }
  builder.addDeferredStages();
  return {
    contributors: builder.contributors,
    outputs: {spatialRates},
    intermediates: {pooledRates},
    status: {stages: [{stage: 'weights', status: weightsStatus}]}
  };
}
