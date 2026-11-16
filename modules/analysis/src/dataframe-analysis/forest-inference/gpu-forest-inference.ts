// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from '../../analysis-core/kernels/graph-chunk-groups';

import {
  createTransientView,
  getBoundedDispatchLayout,
  GraphDataView,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {type WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  planGraphChunkDispatch,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
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
/** Fixed workgroup size keeps unequal row batches on the same pipeline specialization. */
const WORKGROUP_SIZE = 256;
/** All flat-view indexing is uint32, even on devices with larger storage bindings. */
const MAXIMUM_ELEMENT_COUNT = 0xffffffff;

/** Preserves packed row types by default and the caller's chunked view kind when requested. */
type GPUForestInferenceRowView<
  Format extends GraphDataView['format'],
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

export type GPUForestInferenceOutput<Rows extends GraphDataView | GraphVectorView = GraphDataView> =
  {
    /** Regression value, or classification label index into the imported model's classes. Invalid rows receive NaN. */
    predictions: GPUForestInferenceRowView<'float32', Rows>;
    /** 1 for a valid prediction; 0 for mask, nonfinite input/result, or malformed model traversal. */
    validity: GPUForestInferenceRowView<'uint32', Rows>;
    /** Optional row-major probabilities (classification) or regression value, rowCount * outputCount. Invalid rows receive NaN. */
    scores?: GPUForestInferenceRowView<'float32', Rows>;
    /** Optional row-major original-format leaf ids, rowCount * treeCount. Every entry of an invalid row is 0xffffffff. */
    leafIndices?: GPUForestInferenceRowView<'uint32', Rows>;
  };

export type GPUForestInferenceProps<Rows extends GraphDataView | GraphVectorView = GraphDataView> =
  {
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
    /** Scalar-packed float32 features, row-major within each chunk by default. Mutually exclusive with rasterStack. */
    features?: GPUForestInferenceRowView<'float32', Rows>;
    featureLayout?: 'row-major' | 'feature-major';
    /**
     * Co-registered, calibrated float32 buffer bands in feature order; one row per grid cell.
     * Band validity becomes a missing feature and follows the model's missing-value branch.
     * Texture bands and raw scale/offset/noDataValue metadata must be prepared upstream first.
     * No spatial measures are computed, so the affine and cellSizeMode do not affect inference.
     */
    rasterStack?: {bands: readonly GPURasterBand<'float32'>[]; grid: GPURasterMetadata};
    /** Nonzero keeps a row. Zero invalidates all predictions, scores and leaf ids for that row. */
    mask?: GPUForestInferenceRowView<'uint32', Rows>;
    output: GPUForestInferenceOutput<Rows>;
  };

/**
 * Data-parallel inference of an imported sparse numerical forest. One invocation per row traverses
 * at most 1024 trees of depth at most 64, accumulating at most 32 scores in fixed tree order.
 * Random forests average probabilities/values; boosted trees sum raw scores before sigmoid/softmax.
 * Classification ties choose the first class. This applies caller-supplied weights; it does not fit.
 *
 * Compile-time: model shape/task/link, featureLayout, raster band count and optional views.
 * Per-frame: features, mask, band validity, model buffers, parameters. Features and row outputs may
 * be chunked at identical source-row boundaries; each feature chunk holds a complete row-major or
 * feature-major batch. Scores and leaves hold batch rows times outputCount/treeCount scalars.
 * Raster bands and model buffers stay packed; raster bands are borrowed at the output row boundaries.
 * Traversal work and uint32 flat indexing budgets apply per batch. Total rows may reach uint32;
 * each actual aligned binding and the reusable per-batch scratch must fit the device limit.
 * GPU model writers must maintain the importers' layout and finite weights/leaves; malformed traversal
 * is bounded and invalidates the row. NaN follows a declared missing branch; infinity invalidates.
 * Unsupported missing values in sklearn gradient boosting invalidate a row if visited.
 * Computation is float32; use a tolerance for ensemble sums and transcendental links.
 */
export class GPUForestInference<
  Props extends GPUForestInferenceProps<GraphDataView | GraphVectorView> = GPUForestInferenceProps
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: Props & GPUForestInferenceProps<Props['output']['predictions']>;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly rasterChunks: readonly (readonly {
    values: GraphDataView;
    validity?: GraphDataView;
  }[])[];

  constructor(props: Props) {
    this.id = props.id ?? OPERATION;
    this.props = props as Props & GPUForestInferenceProps<Props['output']['predictions']>;
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
    // Keep the original packed-route work diagnostic before validating model/view lengths.
    if (
      !(props.features instanceof GraphVectorView) &&
      !(output.predictions instanceof GraphVectorView)
    ) {
      validateMaximumCount(
        id,
        'rowCount * treeCount * (maximumDepth + outputCount + 1) work',
        rowCount * model.treeCount * (model.maximumDepth + model.outputCount + 1),
        GPU_FOREST_INFERENCE_MAXIMUM_WORK
      );
    }
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
    const rowFields = [
      ['predictions', output.predictions, 1, 'float32'],
      ['validity', output.validity, 1, 'uint32'],
      ['features', props.features, model.featureCount, 'float32'],
      ['mask', props.mask, 1, 'uint32'],
      ['scores', output.scores, model.outputCount, 'float32'],
      ['leafIndices', output.leafIndices, model.treeCount, 'uint32']
    ] as const;
    const rowViews = Object.fromEntries(
      rowFields.map(([name, view, elementsPerRow, format]) => {
        if (!view) return [name, undefined];
        validateViewLength(id, name, view, rowCount * elementsPerRow);
        const data = getGraphViewChunks(view).map(chunk => {
          validateSinglePackedView(chunk, [format], `${id} ${name}`);
          validateCapacity(id, `${name} chunk elements`, chunk.length, 0, MAXIMUM_ELEMENT_COUNT);
          validateRequirement(
            id,
            chunk.length % elementsPerRow === 0,
            `${name} chunks must contain whole rows of ${elementsPerRow} elements`
          );
          // Only row topology is passed to the planner; dispatches bind the full scalar view below.
          return new GraphDataView(chunk.buffer, {...chunk, length: chunk.length / elementsPerRow});
        });
        return [
          name,
          new GraphVectorView({
            id: `${id}-${name}-rows`,
            name,
            format: view.format,
            length: rowCount,
            valueLength: rowCount,
            stride: 1,
            byteStride: 4,
            rowByteLength: 4,
            data
          })
        ];
      })
    );
    this.chunkPlan = planGraphChunkDispatch(id, rowViews);
    for (const chunk of this.chunkPlan.chunks) {
      validateMaximumCount(
        id,
        `chunk ${chunk.chunkIndex} rowCount * treeCount * (maximumDepth + outputCount + 1) work`,
        chunk.length * model.treeCount * (model.maximumDepth + model.outputCount + 1),
        GPU_FOREST_INFERENCE_MAXIMUM_WORK
      );
      if (rasterStack) {
        validateMaximumCount(
          id,
          `chunk ${chunk.chunkIndex} packed raster feature elements`,
          chunk.length * model.featureCount,
          MAXIMUM_ELEMENT_COUNT
        );
      }
      validateMaximumCount(
        id,
        `chunk ${chunk.chunkIndex} scratch elements`,
        chunk.length * (model.outputCount + 2),
        MAXIMUM_ELEMENT_COUNT
      );
    }
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
    {
      const writableOutputs: Readonly<Record<string, GraphDataView | GraphVectorView | undefined>> =
        output;
      const publicationPasses: readonly (readonly (
        | GraphDataView
        | GraphVectorView
        | undefined
      )[])[] = [[output.predictions, output.validity, output.scores]];
      // Model all logical writes as separate chunks; only the explicit publication groups coexist.
      const writableWordChunks = Object.values(writableOutputs)
        .flatMap(view => (view ? getGraphViewChunks(view) : []))
        .map(
          view =>
            new GraphDataView(view.buffer, {
              format: 'uint32',
              length: (view.length * view.rowByteLength) / 4,
              byteOffset: view.byteOffset,
              byteStride: 4,
              rowByteLength: 4
            })
        );
      const writableWordCount = writableWordChunks.reduce(
        (total, chunk) => total + chunk.length,
        0
      );
      validateDisjointWritableViews(id, {
        outputs: new GraphVectorView({
          id: `${id}-writable-ranges`,
          name: 'outputs',
          format: 'uint32',
          length: writableWordCount,
          valueLength: writableWordCount,
          stride: 1,
          byteStride: 4,
          rowByteLength: 4,
          data: writableWordChunks
        })
      });
      for (const fields of publicationPasses) {
        validateDisjointWritableViews(
          id,
          Object.fromEntries(fields.map((view, index) => [String(index), view]))
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(output), this.getInputViews());
    // Raster bands remain public packed views. Borrow each batch without repacking or binding the
    // whole band; validate these exact aligned ranges before declaring any resources or nodes.
    this.rasterChunks = this.chunkPlan.chunks.map(
      chunk =>
        rasterStack?.bands.map(band => {
          const values = band.storage.kind === 'buffer' ? band.storage.values : undefined;
          validateRequirement(id, values !== undefined, 'rasterStack bands must use buffer values');
          return {
            values: new GraphDataView(values.buffer, {
              ...values,
              byteOffset: values.byteOffset + chunk.rowBase * 4,
              length: chunk.length
            }),
            validity: band.validity
              ? new GraphDataView(band.validity.buffer, {
                  ...band.validity,
                  byteOffset: band.validity.byteOffset + chunk.rowBase * 4,
                  length: chunk.length
                })
              : undefined
          };
        }) ?? []
    );
    this.estimatedCommandNodeCount =
      this.chunkPlan.nonemptyChunkCount * (2 + (rasterStack?.bands.length ?? 0));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getInputViews(): (GraphDataView | GraphVectorView)[] {
    const props = this.props;
    const views: (GraphDataView | GraphVectorView | undefined)[] = [
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
    ];
    return views.filter((view): view is GraphDataView | GraphVectorView => Boolean(view));
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, chunkPlan} = this;
    const {model, output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      nodes: props.nodes,
      trees: props.trees,
      leafValues: props.leafValues,
      parameters: props.parameters,
      features: props.features,
      mask: props.mask,
      ...output,
      ...Object.fromEntries(
        this.rasterChunks.flatMap((bands, chunkIndex) =>
          bands.flatMap((band, bandIndex) => [
            [`band ${bandIndex} chunk ${chunkIndex}`, band.values],
            [`band validity ${bandIndex} chunk ${chunkIndex}`, band.validity]
          ])
        )
      )
    });
    if (props.rasterStack) validateGraphChunkDispatchBindings(graph, id, chunkPlan);
    const maximumBatchRowCount = chunkPlan.chunks.reduce(
      (maximum, chunk) => Math.max(maximum, chunk.length),
      1
    );
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      6 + Number(Boolean(props.mask)) + Number(Boolean(output.leafIndices)),
      WORKGROUP_SIZE
    );
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length > 0)
        getBoundedDispatchLayout(
          id,
          chunk.length,
          WORKGROUP_SIZE,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
    }
    for (const [name, elements] of [
      ['scratch', maximumBatchRowCount * (model.outputCount + 2)],
      ['packed raster features', props.rasterStack ? maximumBatchRowCount * model.featureCount : 0]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        elements * 4,
        'split rows into smaller chunks'
      );
      validateMaximumCount(
        id,
        `${name} buffer byteLength (${elements * Float32Array.BYTES_PER_ELEMENT})`,
        elements * Float32Array.BYTES_PER_ELEMENT,
        graph.device.limits.maxBufferSize
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
    if (chunkPlan.nonemptyChunkCount === 0) return commandNodes;
    // Reuse aligned batch scratch in declaration order instead of allocating total-row storage.
    const scratchStorage = createTransientView(
      graph,
      `${id}-scratch`,
      'float32',
      maximumBatchRowCount * (model.outputCount + 2)
    );
    const rasterFeatures = props.rasterStack
      ? createTransientView(
          graph,
          `${id}-features`,
          'float32',
          maximumBatchRowCount * model.featureCount
        )
      : undefined;
    const featureChunks = props.features ? getGraphViewChunks(props.features) : undefined;
    const maskChunks = props.mask ? getGraphViewChunks(props.mask) : undefined;
    const predictions = getGraphViewChunks(output.predictions);
    const validity = getGraphViewChunks(output.validity);
    const scores = output.scores ? getGraphViewChunks(output.scores) : undefined;
    const leaves = output.leafIndices ? getGraphViewChunks(output.leafIndices) : undefined;
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length === 0) continue;
      const features =
        featureChunks?.[chunk.chunkIndex] ??
        graph.createDataView(rasterFeatures!.buffer, {
          format: 'float32',
          length: chunk.length * model.featureCount
        });
      const scratch = graph.createDataView(scratchStorage.buffer, {
        format: 'float32',
        length: chunk.length * (model.outputCount + 2)
      });
      this.rasterChunks[chunk.chunkIndex].forEach((band, featureIndex) => {
        const plane = graph.createDataView(features.buffer, {
          format: 'float32',
          byteOffset: featureIndex * chunk.length * 4,
          length: chunk.length
        });
        const packBindings = [
          read('bandValues', band.values, 'f32'),
          ...(band.validity ? [read('bandValidity', band.validity, 'u32')] : []),
          write('features', plane, 'f32')
        ];
        commandNodes.push(
          createGraphChunkGroupKernelNode<Parameters>(graph, {
            id: `${id}-pack-${featureIndex}-chunk-${chunk.chunkIndex}`,
            operation: OPERATION,
            variant: 'pack-band',
            invocationCount: chunk.length,
            workgroupSize: WORKGROUP_SIZE,
            bindings: packBindings,
            declarations: GET_NAN_WGSL,
            body: `if (index >= getChunkParameter(1u)) { return; }
let bandValuesBase = arrayLength(&bandValues) - getChunkParameter(1u);
let featuresBase = arrayLength(&features) - getChunkParameter(1u);
var value = bandValues[bandValuesBase + index];
${band.validity ? 'let bandValidityBase = arrayLength(&bandValidity) - getChunkParameter(1u); if (bandValidity[bandValidityBase + index] == 0u) { value = getNaN(); }' : ''}
features[featuresBase + index] = value;`
          })
        );
      });
      const featureIndex =
        props.rasterStack || props.featureLayout === 'feature-major'
          ? `feature * batchRowCount + index`
          : `index * ${model.featureCount}u + feature`;
      commandNodes.push(
        createGraphChunkGroupKernelNode<Parameters>(graph, {
          id: `${id}-infer-chunk-${chunk.chunkIndex}`,
          operation: OPERATION,
          variant: 'infer',
          invocationCount: chunk.length,
          workgroupSize: WORKGROUP_SIZE,
          bindings: [
            read('features', features, 'f32'),
            read('modelNodes', props.nodes, 'u32'),
            read('modelTrees', props.trees, 'u32'),
            read('leafValues', props.leafValues, 'f32'),
            read('parameters', props.parameters, 'f32'),
            ...(props.mask ? [read('rowMask', maskChunks![chunk.chunkIndex], 'u32')] : []),
            write('scratch', scratch, 'f32'),
            ...(output.leafIndices ? [write('leafIndices', leaves![chunk.chunkIndex], 'u32')] : [])
          ],
          declarations: `${GET_NAN_WGSL}
${IS_NAN_VALUE_WGSL}
${IS_FINITE_VALUE_WGSL}`,
          body: `let batchRowCount = arrayLength(&scratch) / ${model.outputCount + 2}u;
if (index >= batchRowCount) { return; }
let featuresBase = arrayLength(&features) - batchRowCount * ${model.featureCount}u;
${props.mask ? 'let rowMaskBase = arrayLength(&rowMask) - batchRowCount;' : ''}
${output.leafIndices ? `let leafIndicesBase = arrayLength(&leafIndices) - batchRowCount * ${model.treeCount}u;` : ''}
let nullValue = getNaN();
var valid = ${props.mask ? 'rowMask[rowMaskBase + index] != 0u' : 'true'};
var scores: array<f32, ${model.outputCount}>;
for (var channel = 0u; channel < ${model.outputCount}u; channel++) {
  scores[channel] = parameters[parametersOffset + 1u + channel];
}
for (var treeIndex = 0u; treeIndex < ${model.treeCount}u; treeIndex++) {
  ${output.leafIndices ? `leafIndices[leafIndicesBase + index * ${model.treeCount}u + treeIndex] = 0xffffffffu;` : ''}
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
      ${output.leafIndices ? `leafIndices[leafIndicesBase + index * ${model.treeCount}u + treeIndex] = modelNodes[nodeOffset + 5u];` : ''}
      foundLeaf = true;
      break;
    }
    if (feature >= ${model.featureCount}u) { break; }
    var value = features[featuresBase + ${featureIndex}];
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
  for (var treeIndex = 0u; treeIndex < ${model.treeCount}u; treeIndex++) { leafIndices[leafIndicesBase + index * ${model.treeCount}u + treeIndex] = 0xffffffffu; }
}`
    : ''
}`
        })
      );
      commandNodes.push(
        createGraphChunkGroupKernelNode<Parameters>(graph, {
          id: `${id}-publish-chunk-${chunk.chunkIndex}`,
          operation: OPERATION,
          variant: 'publish',
          invocationCount: chunk.length,
          workgroupSize: WORKGROUP_SIZE,
          bindings: [
            read('scratch', scratch, 'f32'),
            write('predictions', predictions[chunk.chunkIndex], 'f32'),
            write('validity', validity[chunk.chunkIndex], 'u32'),
            ...(output.scores ? [write('scores', scores![chunk.chunkIndex], 'f32')] : [])
          ],
          body: `let batchRowCount = arrayLength(&scratch) / ${model.outputCount + 2}u;
if (index >= batchRowCount) { return; }
let predictionsBase = arrayLength(&predictions) - batchRowCount;
let validityBase = arrayLength(&validity) - batchRowCount;
${output.scores ? `let scoresBase = arrayLength(&scores) - batchRowCount * ${model.outputCount}u;` : ''}
let rowOffset = scratchOffset + index * ${model.outputCount + 2}u;
predictions[predictionsBase + index] = scratch[rowOffset + ${model.outputCount}u];
validity[validityBase + index] = u32(scratch[rowOffset + ${model.outputCount + 1}u]);
${output.scores ? `for (var channel = 0u; channel < ${model.outputCount}u; channel++) { scores[scoresBase + index * ${model.outputCount}u + channel] = scratch[rowOffset + channel]; }` : ''}`
        })
      );
    }
    return commandNodes;
  }
}
