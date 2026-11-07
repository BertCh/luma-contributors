// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  alignGraphVectorViews,
  createTransientView,
  getGraphVectorData,
  GPUReduction,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';

/** GeoDa variable transformations; MAD is mean absolute deviation about the mean. */
export type GPUVariableStandardizationMethod =
  | 'deviation-from-mean'
  | 'z-score'
  | 'mad'
  | 'range-adjust'
  | 'range-standardize';

type Column<Format extends 'float32' | 'uint32'> = GraphDataView<Format> | GraphVectorView<Format>;

/** Topology is fixed; input and mask buffer contents may change between encodings. */
export type GPUVariableStandardizationProps = {
  id?: string;
  /** Packed scalar float32 column, optionally chunked. 0 to 2^24 rows. */
  values: Column<'float32'>;
  /** Nonzero includes the row. Chunk boundaries may differ from values and output. */
  mask?: Column<'uint32'>;
  method: GPUVariableStandardizationMethod;
  /** Z-score convention. Defaults to sample (GeoDa), dividing squared deviations by n - 1. */
  variance?: 'sample' | 'population';
  /** Caller-owned packed float32 output with the same logical length as values. */
  output: Column<'float32'>;
};

/**
 * Named standardization recipes composed from shared GPUReduction and elementwise passes.
 * Finite, unmasked rows define one global population across all chunks. Excluded rows output NaN.
 * MAD and range transformations preserve constant input values; deviation and z-score produce 0
 * for constant populations (sample z-score needs at least two rows, otherwise NaN).
 * No compilation, submission, reduction on the CPU, or readback is performed by this producer.
 */
export class GPUVariableStandardization implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVariableStandardizationProps;

  constructor(props: GPUVariableStandardizationProps) {
    this.id = props.id ?? 'variable-standardization';
    this.props = props;
    validateCapacity(this.id, 'row count', props.values.length, 0, 2 ** 24);
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
      for (const chunk of getGraphVectorData(column))
        validatePackedView(chunk, [format], `${this.id} ${name}`);
    }
    const outputChunks = getGraphVectorData(props.output);
    for (let index = 0; index < outputChunks.length; index++) {
      const current = outputChunks[index];
      for (const previous of outputChunks.slice(0, index)) {
        if (
          current.buffer === previous.buffer &&
          current.length > 0 &&
          previous.length > 0 &&
          current.byteOffset < previous.byteOffset + previous.length * 4 &&
          previous.byteOffset < current.byteOffset + current.length * 4
        ) {
          throw new Error(`${this.id} output chunks must not overlap`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(this.id, getGraphVectorData(props.output), [
      ...getGraphVectorData(props.values),
      ...(props.mask ? getGraphVectorData(props.mask) : [])
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGraphVectorData(props.values),
      ...getGraphVectorData(props.output),
      ...(props.mask ? getGraphVectorData(props.mask) : [])
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const spans = props.mask
      ? alignGraphVectorViews(graph, [props.values, props.output, props.mask] as const)
      : alignGraphVectorViews(graph, [props.values, props.output] as const);
    const cleanChunks: GraphDataView<'float32'>[] = [];
    const validChunks: GraphDataView<'uint32'>[] = [];
    const count = createTransientView(graph, `${id}-count`, 'uint32', 1);
    const sum = createTransientView(graph, `${id}-sum`, 'float32', 1);
    const extent = createTransientView(graph, `${id}-extent`, 'float32', 2);
    const dispersion = createTransientView(graph, `${id}-dispersion`, 'float32', 1);
    const needsMean = !props.method.startsWith('range-');
    const needsDispersion = props.method === 'mad' || props.method === 'z-score';
    for (const [spanIndex, [values, , mask]] of spans.entries()) {
      const clean = createTransientView(
        graph,
        `${id}-clean-${spanIndex}`,
        'float32',
        values.length
      );
      const valid = createTransientView(graph, `${id}-valid-${spanIndex}`, 'uint32', values.length);
      cleanChunks.push(clean);
      validChunks.push(valid);
      const bindings: WGSLKernelBinding[] = [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'clean', view: clean, type: 'f32', access: 'read_write'},
        {name: 'valid', view: valid, type: 'u32', access: 'read_write'}
      ];
      if (mask) bindings.push({name: 'mask', view: mask, type: 'u32', access: 'read'});
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-prepare-${spanIndex}`,
          operation: 'GPUVariableStandardization',
          variant: 'prepare',
          invocationCount: values.length,
          bindings,
          declarations: IS_FINITE_BITS_WGSL,
          body: `let value = values[valuesOffset + index];
  let included = isFiniteBits(value)${mask ? ' && mask[maskOffset + index] != 0u' : ''};
  clean[cleanOffset + index] = select(0.0, value, included);
  valid[validOffset + index] = select(0u, 1u, included);`
        })
      );
    }
    const clean = makeVector(`${id}-clean`, 'float32', cleanChunks);
    const valid = makeVector(`${id}-valid`, 'uint32', validChunks);
    nodes.push(
      ...new GPUReduction({
        id: `${id}-count`,
        input: valid,
        operation: 'sum',
        output: count
      }).getCommandNodes(graph)
    );
    nodes.push(
      ...new GPUReduction({
        id: `${id}-extent`,
        input: clean,
        mask: valid,
        operation: 'extent',
        output: extent
      }).getCommandNodes(graph)
    );
    const normalizedChunks: GraphDataView<'float32'>[] = [];
    if (needsMean) {
      for (const [spanIndex, values] of cleanChunks.entries()) {
        const normalized = createTransientView(
          graph,
          `${id}-normalized-${spanIndex}`,
          'float32',
          values.length
        );
        normalizedChunks.push(normalized);
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-normalize-${spanIndex}`,
            operation: 'GPUVariableStandardization',
            variant: 'normalized-mean',
            declarations: RATIO_WGSL,
            invocationCount: values.length,
            bindings: [
              {name: 'values', view: values, type: 'f32', access: 'read'},
              {name: 'valid', view: validChunks[spanIndex], type: 'u32', access: 'read'},
              {name: 'extent', view: extent, type: 'f32', access: 'read'},
              {name: 'normalized', view: normalized, type: 'f32', access: 'read_write'}
            ],
            body: `${NORMALIZATION_WGSL}
  let value = values[valuesOffset + index];
  normalized[normalizedOffset + index] = select(0.0, getRatio(value - origin, radius), radius > 0.0 && valid[validOffset + index] != 0u);`
          })
        );
      }
      nodes.push(
        ...new GPUReduction({
          id: `${id}-mean-sum`,
          input: makeVector(`${id}-normalized`, 'float32', normalizedChunks),
          operation: 'sum',
          output: sum
        }).getCommandNodes(graph)
      );
    }
    if (needsDispersion) {
      const terms: GraphDataView<'float32'>[] = [];
      for (const [spanIndex, values] of normalizedChunks.entries()) {
        const term = createTransientView(
          graph,
          `${id}-term-${spanIndex}`,
          'float32',
          values.length
        );
        terms.push(term);
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-center-${spanIndex}`,
            operation: 'GPUVariableStandardization',
            variant: props.method,
            invocationCount: values.length,
            bindings: [
              {name: 'values', view: values, type: 'f32', access: 'read'},
              {name: 'valid', view: validChunks[spanIndex], type: 'u32', access: 'read'},
              {name: 'sum', view: sum, type: 'f32', access: 'read'},
              {name: 'count', view: count, type: 'u32', access: 'read'},
              {name: 'term', view: term, type: 'f32', access: 'read_write'}
            ],
            body: `let mean = sum[sumOffset] / f32(max(count[countOffset], 1u));
  let difference = values[valuesOffset + index] - mean;
  term[termOffset + index] = select(0.0, ${props.method === 'mad' ? 'abs(difference)' : 'difference * difference'}, valid[validOffset + index] != 0u);`
          })
        );
      }
      nodes.push(
        ...new GPUReduction({
          id: `${id}-dispersion`,
          input: makeVector(`${id}-terms`, 'float32', terms),
          operation: 'sum',
          output: dispersion
        }).getCommandNodes(graph)
      );
    }
    for (const [spanIndex, [values, output]] of spans.entries()) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'valid', view: validChunks[spanIndex], type: 'u32', access: 'read'},
        {name: 'count', view: count, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'},
        {
          name: 'extent',
          view: extent,
          type: 'f32',
          access: 'read'
        }
      ];
      if (needsMean) bindings.push({name: 'sum', view: sum, type: 'f32', access: 'read'});
      if (needsDispersion)
        bindings.push({name: 'dispersion', view: dispersion, type: 'f32', access: 'read'});
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-apply-${spanIndex}`,
          operation: 'GPUVariableStandardization',
          variant: props.method,
          invocationCount: values.length,
          bindings,
          declarations: GET_NAN_WGSL + IS_FINITE_BITS_WGSL + RATIO_WGSL,
          body: getApplyBody(props)
        })
      );
    }
    return nodes;
  }
}

function makeVector<Format extends 'float32' | 'uint32'>(
  id: string,
  format: Format,
  data: readonly GraphDataView<Format>[]
): GraphVectorView<Format> {
  const length = data.reduce((total, chunk) => total + chunk.length, 0);
  return new GraphVectorView({
    id,
    name: id,
    format,
    length,
    valueLength: length,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data
  });
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

/** Rescale extreme divisors to avoid a flushed subnormal reciprocal on GPU implementations. */
const RATIO_WGSL = /* wgsl */ `fn getRatio(numerator: f32, denominator: f32) -> f32 {
  if (denominator == 0.0) { return 0.0; }
  if (denominator > 1e19) { return (numerator * 1e-19) / (denominator * 1e-19); }
  if (denominator < 1e-19) { return (numerator * 1e19) / (denominator * 1e19); }
  return numerator / denominator;
}`;
