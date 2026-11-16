// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getBoundedDispatchLayout,
  GPUReduction,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateMaximumCount,
  validateSinglePackedView,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from '../../analysis-core/kernels/graph-chunk-groups';
import {
  planGraphChunkDispatch,
  type GraphChunkDispatchPlan,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

/** Row indices and the valid-row reduction use uint32; packed bindings are checked on the device. */
const MAXIMUM_ROW_COUNT = 0xffffffff;
/** Four published reductions need at most five hierarchy levels plus publication each; includes empty clears. */
const REDUCTION_COMMAND_NODE_ALLOWANCE = 128;
/** Fixed across row chunks and global elementwise passes. */
const WORKGROUP_SIZE = 256;

/** GeoDa variable transformations; MAD is mean absolute deviation about the mean. */
export type GPUVariableStandardizationMethod =
  | 'deviation-from-mean'
  | 'z-score'
  | 'mad'
  | 'range-adjust'
  | 'range-standardize';

/** Borrows a row range of a packed transient view. */
function sliceView<Format extends 'float32' | 'uint32', Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  start: number,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(view.buffer, {
    format: view.format,
    length,
    byteOffset: view.byteOffset + start * view.byteStride,
    byteStride: view.byteStride,
    rowByteLength: view.rowByteLength
  });
}

type Column<Format extends 'float32' | 'uint32'> = GraphDataView<Format> | GraphVectorView<Format>;

/** Topology is fixed; input and mask buffer contents may change between encodings. */
export type GPUVariableStandardizationProps = {
  id?: string;
  /** Scalar float32 column with packed chunks. 0 to 2^32 - 1 rows, bounded by the device storage binding size. */
  values: Column<'float32'>;
  /** Nonzero includes the row. Chunk lengths must match values and output. */
  mask?: Column<'uint32'>;
  method: GPUVariableStandardizationMethod;
  /** Z-score convention. Defaults to sample (GeoDa), dividing squared deviations by n - 1. */
  variance?: 'sample' | 'population';
  /** Caller-owned packed or chunked float32 output with the same logical length as values. */
  output: Column<'float32'>;
};

/**
 * Named standardization recipes composed from shared GPUReduction and elementwise passes.
 * Finite, unmasked rows define one global population across all chunks. Global scratch must fit one storage binding. Excluded rows output NaN.
 * MAD and range transformations preserve constant input values; deviation and z-score produce 0
 * for constant populations (sample z-score needs at least two rows, otherwise NaN).
 * No compilation, submission, reduction on the CPU, or readback is performed by this producer.
 */
export class GPUVariableStandardization implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVariableStandardizationProps;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: GPUVariableStandardizationProps) {
    this.id = props.id ?? 'variable-standardization';
    this.props = props;
    validateCapacity(this.id, 'values length', props.values.length, 0, MAXIMUM_ROW_COUNT);
    if (
      !['deviation-from-mean', 'z-score', 'mad', 'range-adjust', 'range-standardize'].includes(
        props.method
      )
    ) {
      throw new Error(`${this.id} invalid method`);
    }
    if (props.variance !== undefined && !['sample', 'population'].includes(props.variance)) {
      throw new Error(`${this.id} invalid variance convention`);
    }
    for (const [name, column, format] of [
      ['values', props.values, 'float32'],
      ['output', props.output, 'float32'],
      ['mask', props.mask, 'uint32']
    ] as const) {
      if (!column) continue;
      if (column.length !== props.values.length)
        throw new Error(`${this.id} ${name} length mismatch`);

      for (const chunk of getGraphViewChunks(column))
        validateSinglePackedView(chunk, [format], `${this.id} ${name}`);
    }
    validateDisjointWritableViews(this.id, {output: props.output});
    this.chunkPlan = planGraphChunkDispatch(this.id, {
      values: props.values,
      output: props.output,
      mask: props.mask
    });
    this.estimatedCommandNodeCount =
      REDUCTION_COMMAND_NODE_ALLOWANCE +
      2 *
        (getGraphViewChunks(props.values).length +
          getGraphViewChunks(props.output).length +
          (props.mask ? getGraphViewChunks(props.mask).length : 0));
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(this.id, getGraphViewChunks(props.output), [
      ...getGraphViewChunks(props.values),
      ...(props.mask ? getGraphViewChunks(props.mask) : [])
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphChunkBindingRanges(graph, id, {
      values: props.values,
      output: props.output,
      mask: props.mask
    });
    validateMaximumCount(
      id,
      'row scratch byteLength',
      Math.max(props.values.length, 2) * 4,
      graph.device.limits.maxBufferSize
    );
    // Every row-proportional scratch plane is scalar; reduction partials are smaller.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'row scratch',
      Math.max(props.values.length, 2) * 4,
      'reduce the packed values row count'
    );
    validateGraphChunkGroupKernelSupport(graph, id, 7, WORKGROUP_SIZE);
    const spans = this.chunkPlan.chunks.map(
      chunk => [chunk.views['values'], chunk.views['output'], chunk.views['mask']] as const
    );
    for (const [values] of spans) {
      if (values.length > 0)
        getBoundedDispatchLayout(
          id,
          values.length,
          WORKGROUP_SIZE,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
    }
    const totalLength = props.values.length;
    if (totalLength > 0)
      getBoundedDispatchLayout(
        id,
        totalLength,
        WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      );
    const clean = createTransientView(graph, `${id}-clean`, 'float32', totalLength);
    const valid = createTransientView(graph, `${id}-valid`, 'uint32', totalLength);
    const count = createTransientView(graph, `${id}-count`, 'uint32', 1);
    const sum = createTransientView(graph, `${id}-sum`, 'float32', 1);
    const extent = createTransientView(graph, `${id}-extent`, 'float32', 2);
    const dispersion = createTransientView(graph, `${id}-dispersion`, 'float32', 1);
    const needsMean = !props.method.startsWith('range-');
    const needsDispersion = props.method === 'mad' || props.method === 'z-score';
    const nodes: GPUCommandNode<Parameters>[] = [];
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
    let rowStart = 0;
    const prepared = spans
      .filter(([values]) => values.length > 0)
      .map(([values, output, mask], spanIndex) => {
        const cleanSpan = sliceView(graph, clean, rowStart, values.length);
        const validSpan = sliceView(graph, valid, rowStart, values.length);
        rowStart += values.length;
        nodes.push(
          createGraphChunkGroupKernelNode(graph, {
            id: `${id}-prepare-${spanIndex}`,
            operation: 'GPUVariableStandardization',
            variant: 'prepare',
            invocationCount: values.length,
            workgroupSize: WORKGROUP_SIZE,
            bindings: [
              read('values', values, 'f32'),
              write('clean', cleanSpan, 'f32'),
              write('valid', validSpan, 'u32'),
              ...(mask ? [read('mask', mask, 'u32')] : [])
            ],
            declarations: IS_FINITE_BITS_WGSL,
            body: `let value = values[valuesOffset + index];
  let included = isFiniteBits(value)${props.mask ? ' && mask[maskOffset + index] != 0u' : ''};
  clean[cleanOffset + index] = select(0.0, value, included);
  valid[validOffset + index] = select(0u, 1u, included);`
          })
        );
        return {values, output, valid: validSpan};
      });
    nodes.push(
      ...new GPUReduction({
        id: `${id}-count`,
        input: valid,
        output: count,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    nodes.push(
      ...new GPUReduction({
        id: `${id}-extent`,
        input: clean,
        mask: valid,
        output: extent,
        operation: 'extent'
      }).getCommandNodes(graph)
    );
    const normalized = needsMean
      ? createTransientView(graph, `${id}-normalized`, 'float32', totalLength)
      : undefined;
    if (normalized) {
      if (totalLength > 0)
        nodes.push(
          createGraphChunkGroupKernelNode(graph, {
            id: `${id}-normalize`,
            operation: 'GPUVariableStandardization',
            variant: 'normalized-mean',
            invocationCount: totalLength,
            bindings: [
              read('values', clean, 'f32'),
              read('valid', valid, 'u32'),
              read('extent', extent, 'f32'),
              write('normalized', normalized, 'f32')
            ],
            declarations: RATIO_WGSL,
            body: `${NORMALIZATION_WGSL}
  let value = values[valuesOffset + index];
  normalized[normalizedOffset + index] = select(0.0, getRatio(value - origin, radius), radius > 0.0 && valid[validOffset + index] != 0u);`
          })
        );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-mean-sum`,
          input: normalized,
          output: sum,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    if (needsDispersion && normalized) {
      const terms = createTransientView(graph, `${id}-terms`, 'float32', totalLength);
      if (totalLength > 0)
        nodes.push(
          createGraphChunkGroupKernelNode(graph, {
            id: `${id}-center`,
            operation: 'GPUVariableStandardization',
            variant: props.method,
            invocationCount: totalLength,
            bindings: [
              read('values', normalized, 'f32'),
              read('valid', valid, 'u32'),
              read('sum', sum, 'f32'),
              read('count', count, 'u32'),
              write('term', terms, 'f32')
            ],
            body: `let mean = sum[sumOffset] / f32(max(count[countOffset], 1u));
  let difference = values[valuesOffset + index] - mean;
  term[termOffset + index] = select(0.0, ${props.method === 'mad' ? 'abs(difference)' : 'difference * difference'}, valid[validOffset + index] != 0u);`
          })
        );
      nodes.push(
        ...new GPUReduction({
          id: `${id}-dispersion`,
          input: terms,
          output: dispersion,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    for (const [spanIndex, span] of prepared.entries()) {
      nodes.push(
        createGraphChunkGroupKernelNode(graph, {
          id: `${id}-apply-${spanIndex}`,
          operation: 'GPUVariableStandardization',
          variant: props.method,
          invocationCount: span.values.length,
          bindings: [
            read('values', span.values, 'f32'),
            read('valid', span.valid, 'u32'),
            read('count', count, 'u32'),
            write('output', span.output, 'f32'),
            read('extent', extent, 'f32'),
            ...(needsMean ? [read('sum', sum, 'f32')] : []),
            ...(needsDispersion ? [read('dispersion', dispersion, 'f32')] : [])
          ],
          declarations: GET_NAN_WGSL + IS_FINITE_BITS_WGSL + RATIO_WGSL,
          body: getApplyBody(props)
        })
      );
    }

    return nodes;
  }
}

function getApplyBody(props: GPUVariableStandardizationProps): string {
  let transformation: string;
  switch (props.method) {
    case 'deviation-from-mean':
      transformation = 'result = (normalizedValue - mean) * radius;';
      break;
    case 'z-score': {
      const correction = props.variance === 'population' ? 0 : 1;
      transformation = `if (countValue > ${correction}u) {
    let scale = sqrt(dispersion[dispersionOffset] / f32(countValue - ${correction}u));
    if (isFiniteBits(scale)) { result = select(0.0, (normalizedValue - mean) / scale, scale > 0.0); }
  }`;
      break;
    }
    case 'mad':
      transformation = `let scale = dispersion[dispersionOffset] / f32(max(countValue, 1u));
  if (isFiniteBits(scale)) { result = select(value, (normalizedValue - mean) / scale, scale > 0.0); }`;
      break;
    default:
      transformation = `let lower = getRatio(minimum - origin, radius);
  let upper = getRatio(maximum - origin, radius);
  let normalizedRange = upper - lower;
  result = value;
  if (radius > 0.0) { result = ${props.method === 'range-adjust' ? 'getRatio(value, radius)' : '(normalizedValue - lower)'} / normalizedRange; }`;
  }
  return `${NORMALIZATION_WGSL}
  let value = values[valuesOffset + index];
  let normalizedValue = select(0.0, getRatio(value - origin, radius), radius > 0.0);
  let countValue = count[countOffset];
  ${!props.method.startsWith('range-') ? 'let mean = sum[sumOffset] / f32(max(countValue, 1u));' : ''}
  var result = getNaN();
  if (valid[validOffset + index] != 0u) { ${transformation} }
  output[outputOffset + index] = select(getNaN(), result, isFiniteBits(result));`;
}

/** Origin/radius normalization bounds finite coordinates before moments are reduced. */
const NORMALIZATION_WGSL = /* wgsl */ `let minimum = extent[extentOffset];
  let maximum = extent[extentOffset + 1u];
  let sameSign = (minimum >= 0.0) == (maximum >= 0.0);
  let origin = select(0.0, minimum, sameSign);
  let radius = max(abs(minimum - origin), abs(maximum - origin));`;

/** Exponent-scaled division keeps extreme normal results under Metal reciprocal fast math. */
const RATIO_WGSL = /* wgsl */ `${SCALED_QUOTIENT_WGSL}
fn getRatio(numerator: f32, denominator: f32) -> f32 {
  if (denominator == 0.0) { return 0.0; }
  return getScaledQuotient(numerator, denominator, 1.0, 0);
}`;
