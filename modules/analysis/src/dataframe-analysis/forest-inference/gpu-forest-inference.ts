// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_FOREST_INFERENCE_MAXIMUM_RASTER_BAND_COUNT,
  GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT,
  GPU_FOREST_INFERENCE_MAXIMUM_WORK,
  GPU_FOREST_INFERENCE_NODE_STRIDE,
  GPU_FOREST_INFERENCE_TREE_STRIDE,
  getGPUForestInferenceParameterLength,
  validateForestInferenceModelShape,
  type GPUForestInferenceModelShape
} from './forest-inference-parameters';

import {
  GET_NAN_WGSL,
  IS_NAN_VALUE_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';

const OPERATION = 'GPUForestInference';
/** All flat-view indexing is uint32, even on devices with larger storage bindings. */
const MAXIMUM_ELEMENT_COUNT = 0xffffffff;

export type GPUForestInferenceOutput = {
  /** Regression value, or classification label index into the imported model's classes. Invalid rows receive NaN. */
  predictions: GraphDataView<'float32'>;
  /** 1 for a valid prediction; 0 for mask, nonfinite input/result, or malformed model traversal. */
  validity: GraphDataView<'uint32'>;
  /** Optional row-major probabilities (classification) or regression value, rowCount * outputCount. Invalid rows receive NaN. */
  scores?: GraphDataView<'float32'>;
  /** Optional row-major original-format leaf ids, rowCount * treeCount. Every entry of an invalid row is 0xffffffff. */
  leafIndices?: GraphDataView<'uint32'>;
};

export type GPUForestInferenceProps = {
  id?: string;
  /** Compile-time imported model shape. Contents of the model views may change without compilation. */
  model: GPUForestInferenceModelShape;
  /** Sparse node words from readScikitLearnForestModel, readXGBoostForestModel or readLightGBMForestModel. */
  nodes: GraphDataView<'uint32'>;
  trees: GraphDataView<'uint32'>;
  leafValues: GraphDataView<'float32'>;
  /** getGPUForestInferenceParameterValues layout: sigmoid and base scores. */
  parameters: GraphDataView<'float32'>;
  /** Source row count, including padding rows; padding is excluded with mask. */
  rowCount: number;
  /** Packed float32 features, row-major by default. Mutually exclusive with rasterStack. */
  features?: GraphDataView<'float32'>;
  featureLayout?: 'row-major' | 'feature-major';
  /**
   * Co-registered, calibrated float32 buffer bands in feature order; one row per grid cell.
   * Band validity becomes a missing feature and follows the model's missing-value branch.
   * Texture bands and raw scale/offset/noDataValue metadata must be prepared upstream first.
   * No spatial measures are computed, so the affine and cellSizeMode do not affect inference.
   */
  rasterStack?: {bands: readonly GPURasterBand<'float32'>[]; grid: GPURasterMetadata};
  /** Nonzero keeps a row. Zero invalidates all predictions, scores and leaf ids for that row. */
  mask?: GraphDataView<'uint32'>;
  output: GPUForestInferenceOutput;
};

/**
 * Data-parallel inference of an imported sparse numerical forest. One invocation per row traverses
 * at most 1024 trees of depth at most 64, accumulating at most 32 scores in fixed tree order.
 * Random forests average probabilities/values; boosted trees sum raw scores before sigmoid/softmax.
 * Classification ties choose the first class. This applies caller-supplied weights; it does not fit.
 *
 * Compile-time: model shape/task/link, rowCount, featureLayout, raster band count and optional views.
 * Per-frame: features, mask, band validity, model buffers, parameters. Inputs are single packed views.
 * GPU model writers must maintain the importers' layout and finite weights/leaves; malformed traversal
 * is bounded and invalidates the row. NaN follows a declared missing branch; infinity invalidates.
 * Unsupported missing values in sklearn gradient boosting invalidate a row if visited.
 * Computation is float32; use a tolerance for ensemble sums and transcendental links.
 */
export class GPUForestInference implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUForestInferenceProps;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUForestInferenceProps) {
    this.id = props.id ?? OPERATION;
    this.props = props;
    const {id} = this;
    const {model, output, rowCount, rasterStack} = props;
    validateForestInferenceModelShape(id, model);
    validateCapacity(id, 'rowCount', rowCount, 0, GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT);
    validateRequirement(
      id,
      Boolean(props.features) !== Boolean(rasterStack),
      'exactly one of features or rasterStack must be supplied'
    );
    validateRequirement(
      id,
      props.featureLayout === undefined ||
        ['row-major', 'feature-major'].includes(props.featureLayout),
      'featureLayout must be row-major or feature-major'
    );
    validateMaximumCount(
      id,
      'rowCount * treeCount * (maximumDepth + outputCount + 1) work',
      rowCount * model.treeCount * (model.maximumDepth + model.outputCount + 1),
      GPU_FOREST_INFERENCE_MAXIMUM_WORK
    );
    for (const [name, count] of [
      ['features elements', rowCount * model.featureCount],
      ['scratch elements', rowCount * (model.outputCount + 2)],
      ['leafIndices elements', output.leafIndices ? rowCount * model.treeCount : 0]
    ] as const)
      validateMaximumCount(id, name, count, MAXIMUM_ELEMENT_COUNT);
    const check = (
      name: string,
      view: GraphDataView,
      format: 'uint32' | 'float32',
      length: number
    ) => {
      validateSinglePackedView(view, [format], `${id} ${name}`);
      validateViewLength(id, name, view, length);
    };
    check('nodes', props.nodes, 'uint32', model.nodeCount * GPU_FOREST_INFERENCE_NODE_STRIDE);
    check('trees', props.trees, 'uint32', model.treeCount * GPU_FOREST_INFERENCE_TREE_STRIDE);
    check('leafValues', props.leafValues, 'float32', model.nodeCount * model.outputCount);
    check(
      'parameters',
      props.parameters,
      'float32',
      getGPUForestInferenceParameterLength(model.outputCount)
    );
    if (props.features) check('features', props.features, 'float32', rowCount * model.featureCount);
    if (props.mask) check('mask', props.mask, 'uint32', rowCount);
    check('output.predictions', output.predictions, 'float32', rowCount);
    check('output.validity', output.validity, 'uint32', rowCount);
    if (output.scores)
      check('output.scores', output.scores, 'float32', rowCount * model.outputCount);
    if (output.leafIndices)
      check('output.leafIndices', output.leafIndices, 'uint32', rowCount * model.treeCount);
    if (rasterStack) {
      validateCapacity(
        id,
        'rasterStack.bands.length',
        rasterStack.bands.length,
        1,
        GPU_FOREST_INFERENCE_MAXIMUM_RASTER_BAND_COUNT
      );
      validateRequirement(
        id,
        rasterStack.bands.length === model.featureCount,
        'rasterStack.bands.length must equal featureCount'
      );
      validateCapacity(
        id,
        'grid.width',
        rasterStack.grid.width,
        1,
        GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT
      );
      validateCapacity(
        id,
        'grid.height',
        rasterStack.grid.height,
        1,
        GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT
      );
      validateRequirement(
        id,
        rasterStack.grid.width * rasterStack.grid.height === rowCount,
        'grid.width * grid.height must equal rowCount'
      );
      rasterStack.bands.forEach((band, index) => {
        validateRequirement(
          id,
          band.format === 'float32' && band.storage.kind === 'buffer',
          `rasterStack.bands[${index}] must be a float32 buffer band`
        );
        validateRequirement(
          id,
          (band.scale ?? 1) === 1 && (band.offset ?? 0) === 0 && band.noDataValue === undefined,
          `rasterStack.bands[${index}] must be calibrated (scale 1, offset 0, noDataValue undefined); supply validity`
        );
        check(`rasterStack.bands[${index}].values`, band.storage.values, 'float32', rowCount);
        if (band.validity)
          check(`rasterStack.bands[${index}].validity`, band.validity, 'uint32', rowCount);
      });
    }
    const inputs = this.getInputViews();
    const outputs = [output.predictions, output.validity, output.scores, output.leafIndices].filter(
      (view): view is GraphDataView<'float32'> | GraphDataView<'uint32'> => Boolean(view)
    );
    validateRequirement(
      id,
      new Set(outputs.map(view => view.buffer)).size === outputs.length,
      'output buffers must be distinct'
    );
    validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
    this.estimatedCommandNodeCount = 2 + (rasterStack?.bands.length ?? 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getInputViews(): GraphDataView[] {
    const props = this.props;
    return [
      props.nodes,
      props.trees,
      props.leafValues,
      props.parameters,
      props.features,
      props.mask,
      ...(props.rasterStack?.bands.flatMap(band => [
        band.storage.kind === 'buffer' ? band.storage.values : undefined,
        band.validity
      ]) ?? [])
    ].filter((view): view is GraphDataView<'float32'> | GraphDataView<'uint32'> => Boolean(view));
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {model, output, rowCount} = props;
    const allViews = [
      ...this.getInputViews(),
      output.predictions,
      output.validity,
      output.scores,
      output.leafIndices
    ].filter((view): view is GraphDataView<'float32'> | GraphDataView<'uint32'> => Boolean(view));
    validateGraphViewsBelongToGraph(id, graph, allViews);
    // Every binding, including row-proportional scratch and offset backing views, is checked before creating nodes.
    for (const view of allViews)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'view binding',
        getViewBindingRange(view).size,
        'split rows or reduce model size'
      );
    for (const [name, elements] of [
      ['scratch', rowCount * (model.outputCount + 2)],
      ['packed raster features', props.rasterStack ? rowCount * model.featureCount : 0]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        elements * 4,
        'split rows into smaller batches'
      );
    }
    const read = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read_write'
    });
    const commandNodes: GPUCommandNode<Parameters>[] = [];
    const features =
      props.features ??
      createTransientView(graph, `${id}-features`, 'float32', rowCount * model.featureCount);
    props.rasterStack?.bands.forEach((band, featureIndex) => {
      if (band.storage.kind !== 'buffer') return;
      commandNodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-pack-${featureIndex}`,
          operation: OPERATION,
          variant: 'pack-band',
          invocationCount: rowCount,
          bindings: [
            read('bandValues', band.storage.values, 'f32'),
            ...(band.validity ? [read('bandValidity', band.validity, 'u32')] : []),
            write('features', features, 'f32')
          ],
          declarations: GET_NAN_WGSL,
          body: `var value = bandValues[bandValuesOffset + index];
${band.validity ? 'if (bandValidity[bandValidityOffset + index] == 0u) { value = getNaN(); }' : ''}
features[featuresOffset + ${featureIndex * rowCount}u + index] = value;`
        })
      );
    });
    const scratch = createTransientView(
      graph,
      `${id}-scratch`,
      'float32',
      rowCount * (model.outputCount + 2)
    );
    const featureIndex =
      props.rasterStack || props.featureLayout === 'feature-major'
        ? `feature * ${rowCount}u + index`
        : `index * ${model.featureCount}u + feature`;
    commandNodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-infer`,
        operation: OPERATION,
        variant: 'infer',
        invocationCount: rowCount,
        bindings: [
          read('features', features, 'f32'),
          read('modelNodes', props.nodes, 'u32'),
          read('modelTrees', props.trees, 'u32'),
          read('leafValues', props.leafValues, 'f32'),
          read('parameters', props.parameters, 'f32'),
          ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
          write('scratch', scratch, 'f32'),
          ...(output.leafIndices ? [write('leafIndices', output.leafIndices, 'u32')] : [])
        ],
        declarations: `${GET_NAN_WGSL}
${IS_NAN_VALUE_WGSL}
${IS_FINITE_VALUE_WGSL}`,
        body: `let nullValue = getNaN();
var valid = ${props.mask ? 'rowMask[rowMaskOffset + index] != 0u' : 'true'};
var scores: array<f32, ${model.outputCount}>;
for (var channel = 0u; channel < ${model.outputCount}u; channel++) {
  scores[channel] = parameters[parametersOffset + 1u + channel];
}
for (var treeIndex = 0u; treeIndex < ${model.treeCount}u; treeIndex++) {
  ${output.leafIndices ? `leafIndices[leafIndicesOffset + index * ${model.treeCount}u + treeIndex] = 0xffffffffu;` : ''}
  if (!valid) { continue; }
  let treeOffset = modelTreesOffset + treeIndex * 4u;
  let root = modelTrees[treeOffset];
  let end = modelTrees[treeOffset + 1u];
  let weight = bitcast<f32>(modelTrees[treeOffset + 2u]);
  var node = root;
  var foundLeaf = false;
  for (var depth = 0u; depth <= ${model.maximumDepth}u; depth++) {
    if (node < root || node >= end || node >= ${model.nodeCount}u) { break; }
    let nodeOffset = modelNodesOffset + node * 6u;
    let feature = modelNodes[nodeOffset];
    if (feature == 0xffffffffu) {
      for (var channel = 0u; channel < ${model.outputCount}u; channel++) {
        scores[channel] += weight * leafValues[leafValuesOffset + node * ${model.outputCount}u + channel];
      }
      ${output.leafIndices ? `leafIndices[leafIndicesOffset + index * ${model.treeCount}u + treeIndex] = modelNodes[nodeOffset + 5u];` : ''}
      foundLeaf = true;
      break;
    }
    if (feature >= ${model.featureCount}u) { break; }
    var value = features[featuresOffset + ${featureIndex}];
    let flags = modelNodes[nodeOffset + 4u];
    if (isNanValue(value) && (flags & 8u) != 0u) { value = 0.0; }
    var goLeft = false;
    if (isNanValue(value) || ((flags & 4u) != 0u && abs(value) <= 1e-35)) {
      if ((flags & 2u) == 0u) { break; }
      goLeft = (flags & 1u) != 0u;
    } else {
      if (!isFiniteValue(value)) { break; }
      let threshold = bitcast<f32>(modelNodes[nodeOffset + 3u]);
      if (isNanValue(threshold)) { break; }
      goLeft = select(value <= threshold, value < threshold, (flags & 16u) != 0u);
    }
    node = select(modelNodes[nodeOffset + 2u], modelNodes[nodeOffset + 1u], goLeft);
  }
  valid = foundLeaf;
}
for (var channel = 0u; channel < ${model.outputCount}u; channel++) { valid = valid && isFiniteValue(scores[channel]); }
${
  model.transform === 'sigmoid'
    ? `let multiplier = parameters[parametersOffset];
valid = valid && isFiniteValue(multiplier) && multiplier > 0.0;
let margin = scores[1] * multiplier;
// Stable expit avoids overflowing exp for large negative margins.
let exponential = exp(-abs(margin));
scores[1] = select(exponential / (1.0 + exponential), 1.0 / (1.0 + exponential), margin >= 0.0);
scores[0] = 1.0 - scores[1];`
    : ''
}
${
  model.transform === 'softmax'
    ? `var maximumScore = scores[0];
for (var channel = 1u; channel < ${model.outputCount}u; channel++) { maximumScore = max(maximumScore, scores[channel]); }
var total = 0.0;
for (var channel = 0u; channel < ${model.outputCount}u; channel++) { scores[channel] = exp(scores[channel] - maximumScore); total += scores[channel]; }
for (var channel = 0u; channel < ${model.outputCount}u; channel++) { scores[channel] /= total; }`
    : ''
}
var prediction = scores[0];
${
  model.task === 'classification'
    ? `var bestClass = 0u;
for (var channel = 1u; channel < ${model.outputCount}u; channel++) { if (scores[channel] > scores[bestClass]) { bestClass = channel; } }
prediction = f32(bestClass);`
    : ''
}
let rowOffset = scratchOffset + index * ${model.outputCount + 2}u;
for (var channel = 0u; channel < ${model.outputCount}u; channel++) { scratch[rowOffset + channel] = select(nullValue, scores[channel], valid); }
scratch[rowOffset + ${model.outputCount}u] = select(nullValue, prediction, valid);
scratch[rowOffset + ${model.outputCount + 1}u] = select(0.0, 1.0, valid);
${
  output.leafIndices
    ? `if (!valid) {
  for (var treeIndex = 0u; treeIndex < ${model.treeCount}u; treeIndex++) { leafIndices[leafIndicesOffset + index * ${model.treeCount}u + treeIndex] = 0xffffffffu; }
}`
    : ''
}`
      })
    );
    commandNodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        invocationCount: rowCount,
        bindings: [
          read('scratch', scratch, 'f32'),
          write('predictions', output.predictions, 'f32'),
          write('validity', output.validity, 'u32'),
          ...(output.scores ? [write('scores', output.scores, 'f32')] : [])
        ],
        body: `let rowOffset = scratchOffset + index * ${model.outputCount + 2}u;
predictions[predictionsOffset + index] = scratch[rowOffset + ${model.outputCount}u];
validity[validityOffset + index] = u32(scratch[rowOffset + ${model.outputCount + 1}u]);
${output.scores ? `for (var channel = 0u; channel < ${model.outputCount}u; channel++) { scores[scoresOffset + index * ${model.outputCount}u + channel] = scratch[rowOffset + channel]; }` : ''}`
      })
    );
    return commandNodes;
  }
}
