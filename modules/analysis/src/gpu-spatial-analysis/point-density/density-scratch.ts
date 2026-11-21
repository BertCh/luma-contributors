// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  isGraphVectorView,
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {createDensityPointPass} from './density-point-passes';
import {
  createPointDensityScaleNodes,
  POINT_DENSITY_WORKGROUP_SUM_SIZE
} from './point-density-kernels';
import {
  createFixedPointClearNode,
  createFixedPointDecodeNode,
  FIXED_POINT_WORD_COUNT,
  FIXED_POINT_WGSL,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';

/** Scalar point scratch follows the input topology; no binding spans two input chunks. @internal */
export function createDensityScalarScratch<Parameters, Format extends 'float32' | 'uint32'>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  template: GraphDataView | GraphVectorView,
  format: Format,
  packed = false
): GraphDataView<Format> | GraphVectorView<Format> {
  if (packed || !isGraphVectorView(template)) {
    return createTransientView(graph, id, format, template.length);
  }
  const data = template.data.map((chunk, index) =>
    createTransientView(graph, `${id}-${index}`, format, chunk.length)
  );
  return new GraphVectorView({
    id,
    name: id,
    format,
    length: template.length,
    valueLength: template.length,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data
  });
}

/** Finite scalar extrema across chunks. Runtime dispatch metadata keeps the reduction's shader
 * and pipeline independent of chunk lengths and binding prefixes. Ordered integer keys select
 * the original float32 value exactly, including negative coordinates; no moment arithmetic occurs.
 */
export function createDensityExtremumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GraphDataView<'float32'> | GraphVectorView<'float32'>,
  output: GraphDataView<'float32'>,
  operation: 'min' | 'max'
): GPUCommandNode<Parameters>[] {
  if (!isGraphVectorView(input)) {
    return [...new GPUReduction({id, input, output, operation}).getCommandNodes(graph)];
  }
  const extremum = createTransientView(graph, `${id}-ordered-key`, 'uint32', 1);
  const initial = operation === 'min' ? '0xffffffffu' : '0u';
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-clear`,
      operation: 'GPUKernelDensity',
      variant: `extremum-${operation}-clear`,
      invocationCount: 1,
      bindings: [{name: 'extremum', view: extremum, type: 'u32', access: 'read_write'}],
      body: `extremum[extremumOffset] = ${initial};`
    }),
    ...getGraphViewChunks(input).flatMap((chunk, chunkIndex) =>
      chunk.length === 0
        ? []
        : [
            createDensityPointPass(graph, {
              id: `${id}-${chunkIndex}`,
              operation: 'GPUKernelDensity',
              variant: `extremum-${operation}`,
              invocationCount: chunk.length,
              workgroupSize: POINT_DENSITY_WORKGROUP_SUM_SIZE,
              guardIndex: false,
              bindings: [
                {name: 'terms', view: chunk, type: 'f32', access: 'read'},
                {name: 'extremum', view: extremum, type: 'atomic<u32>', access: 'read_write'}
              ],
              declarations: `var<workgroup> extremumKeys: array<u32, ${POINT_DENSITY_WORKGROUP_SUM_SIZE}>;`,
              body: `let lane = localInvocationIndex;
  var key = ${initial};
  if (index < INVOCATION_COUNT) {
    let bits = bitcast<u32>(terms[termsOffset + index]);
    key = select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
  }
  extremumKeys[lane] = key;
  workgroupBarrier();
  var stride = ${POINT_DENSITY_WORKGROUP_SUM_SIZE / 2}u;
  loop {
    if (lane < stride) { extremumKeys[lane] = ${operation}(extremumKeys[lane], extremumKeys[lane + stride]); }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (lane == 0u) { atomic${operation === 'min' ? 'Min' : 'Max'}(&extremum[extremumOffset], extremumKeys[0]); }`
            })
          ]
    ),
    createWGSLKernelNode(graph, {
      id: `${id}-decode`,
      operation: 'GPUKernelDensity',
      variant: 'extremum-decode',
      invocationCount: 1,
      bindings: [
        {name: 'extremum', view: extremum, type: 'u32', access: 'read'},
        {name: 'outputValue', view: output, type: 'f32', access: 'read_write'}
      ],
      body: `let key = extremum[extremumOffset];
let bits = select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u);
outputValue[outputValueOffset] = bitcast<f32>(bits);`
    })
  ];
}

/** Preflights the widest per-point scratch binding and allocation before declarations. */
export function validateDensityPointScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  template: GraphDataView | GraphVectorView,
  bytesPerRow: number
): void {
  const chunks = getGraphViewChunks(template);
  for (const chunk of chunks) {
    const byteLength = Math.max(1, chunk.length) * bytesPerRow;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'point scratch chunk',
      byteLength,
      'reduce the point chunk length'
    );
    validateCapacity(
      id,
      'point scratch chunk byteLength',
      byteLength,
      0,
      graph.device.limits.maxBufferSize
    );
  }
}

/**
 * Borrow matching key/weight spans for the deterministic point-density accumulator. Weight and
 * position boundaries may differ; no rows are repacked and every binding prefix stays runtime.
 */
export function getDensityWeightKeySpans<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  weights: GraphDataView<'float32'> | GraphVectorView<'float32'>,
  keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>
): {weights: GraphDataView<'float32'>; keys: GraphDataView<'uint32'>}[] {
  const weightChunks = getGraphViewChunks(weights).filter(chunk => chunk.length > 0);
  const keyChunks = getGraphViewChunks(keys).filter(chunk => chunk.length > 0);
  const spans: {weights: GraphDataView<'float32'>; keys: GraphDataView<'uint32'>}[] = [];
  let keyIndex = 0;
  let keyOffset = 0;
  for (const weight of weightChunks) {
    let weightOffset = 0;
    while (weightOffset < weight.length) {
      const key = keyChunks[keyIndex];
      const length = Math.min(weight.length - weightOffset, key.length - keyOffset);
      spans.push({
        weights: graph.createDataView(weight.buffer, {
          format: 'float32',
          length,
          byteOffset: weight.byteOffset + weightOffset * 4
        }),
        keys: graph.createDataView(key.buffer, {
          format: 'uint32',
          length,
          byteOffset: key.byteOffset + keyOffset * 4
        })
      });
      weightOffset += length;
      keyOffset += length;
      if (keyOffset === key.length) {
        keyIndex++;
        keyOffset = 0;
      }
    }
  }
  return spans;
}

/**
 * Density weight/moment totals reuse the deterministic point-density accumulator. One cell
 * holds the total across all chunks, so neither row count nor chunk count grows a binding.
 * Non-finite terms retain IEEE sum propagation instead of being silently dropped.
 */
export function createDensityWeightSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  input: GraphDataView<'float32'> | GraphVectorView<'float32'>,
  output: GraphDataView<'float32'>
): GPUCommandNode<Parameters>[] {
  // Preserve the established packed reduction and its float32 rounding order.
  if (!isGraphVectorView(input)) {
    return [...new GPUReduction({id, input, output, operation: 'sum'}).getCommandNodes(graph)];
  }
  const scale = createTransientView(graph, `${id}-scale`, 'uint32', 1);
  const words = createTransientView(graph, `${id}-words`, 'uint32', FIXED_POINT_WORD_COUNT);
  const nonfinite = createTransientView(graph, `${id}-nonfinite`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    ...createPointDensityScaleNodes(graph, {
      id: `${id}-scale`,
      operation: 'GPUKernelDensity',
      values: input,
      scale
    }),
    createFixedPointClearNode(graph, {id: `${id}-clear`, operation: 'GPUKernelDensity', words}),
    createDensityPointPass(graph, {
      id: `${id}-nonfinite-clear`,
      operation: 'GPUKernelDensity',
      invocationCount: 1,
      bindings: [{name: 'nonfinite', view: nonfinite, type: 'u32', access: 'read_write'}],
      body: 'nonfinite[nonfiniteOffset] = 0u;'
    })
  ];
  for (const [chunkIndex, chunk] of getGraphViewChunks(input).entries()) {
    if (chunk.length === 0) continue;
    nodes.push(
      createDensityPointPass(graph, {
        id: `${id}-accumulate-${chunkIndex}`,
        operation: 'GPUKernelDensity',
        invocationCount: chunk.length,
        guardIndex: false,
        workgroupSize: POINT_DENSITY_WORKGROUP_SUM_SIZE,
        bindings: [
          {name: 'terms', view: chunk, type: 'f32', access: 'read'},
          {name: 'scale', view: scale, type: 'u32', access: 'read'},
          {name: 'sums', view: words, type: 'atomic<u32>', access: 'read_write'},
          {name: 'nonfinite', view: nonfinite, type: 'atomic<u32>', access: 'read_write'}
        ],
        constants: {HEADROOM_BITS: getFixedPointHeadroomBits(input.length)},
        declarations: `override HEADROOM_BITS: u32;
${FIXED_POINT_WGSL}
${getFixedPointAtomicAddWGSL('sums')}
var<workgroup> termSums: array<vec3u, ${POINT_DENSITY_WORKGROUP_SUM_SIZE}>;
var<workgroup> nonfiniteFlags: atomic<u32>;`,
        body: `let lane = localInvocationIndex;
  if (lane == 0u) { atomicStore(&nonfiniteFlags, 0u); }
  workgroupBarrier();
  var contribution = vec3u(0u);
  if (index < INVOCATION_COUNT) {
    let value = terms[termsOffset + index];
    let bits = bitcast<u32>(value);
    if ((bits & 0x7fffffffu) < 0x7f800000u) {
      let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], HEADROOM_BITS);
      contribution = quantizeFixedPoint(value, unitExponent);
    } else {
      let flags = select(select(2u, 4u, (bits >> 31u) != 0u), 1u, (bits & 0x7fffffu) != 0u);
      atomicOr(&nonfiniteFlags, flags);
    }
  }
  termSums[lane] = contribution;
  workgroupBarrier();
  var stride = ${POINT_DENSITY_WORKGROUP_SUM_SIZE / 2}u;
  loop {
    if (lane < stride) { termSums[lane] = addFixedPoint(termSums[lane], termSums[lane + stride]); }
    workgroupBarrier();
    if (stride == 1u) { break; }
    stride /= 2u;
  }
  if (lane == 0u) {
    addFixedPointToSums(0u, termSums[0]);
    atomicOr(&nonfinite[nonfiniteOffset], atomicLoad(&nonfiniteFlags));
  }`
      })
    );
  }
  nodes.push(
    createFixedPointDecodeNode(graph, {
      id: `${id}-decode`,
      operation: 'GPUKernelDensity',
      words,
      scale,
      maximumTermCount: input.length,
      output
    }),
    createDensityPointPass(graph, {
      id: `${id}-nonfinite-finish`,
      operation: 'GPUKernelDensity',
      invocationCount: 1,
      bindings: [
        {name: 'nonfinite', view: nonfinite, type: 'u32', access: 'read'},
        {name: 'result', view: output, type: 'f32', access: 'read_write'}
      ],
      body: `let flags = nonfinite[nonfiniteOffset];
  // Non-finite f32 constants are invalid WGSL, including bitcasts inside runtime branches.
  // Select integer bits from runtime flags, then bitcast the mutable local value once.
  var nonfiniteBits = 0u;
  if ((flags & 1u) != 0u || (flags & 6u) == 6u) {
    nonfiniteBits = 0x7fc00000u;
  } else if ((flags & 2u) != 0u) {
    nonfiniteBits = 0x7f800000u;
  } else if ((flags & 4u) != 0u) {
    nonfiniteBits = 0xff800000u;
  }
  if (nonfiniteBits != 0u) {
    result[resultOffset] = bitcast<f32>(nonfiniteBits);
  }`
    })
  );
  return nodes;
}

/** Exact uint32 exclusion status, with one global atomic per workgroup and runtime chunk sizes. */
export function createDensityInvalidCountNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  excluded: GraphDataView<'uint32'> | GraphVectorView<'uint32'>,
  output: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  if (!isGraphVectorView(excluded)) {
    return [
      ...new GPUReduction({id, input: excluded, output, operation: 'sum'}).getCommandNodes(graph)
    ];
  }
  return [
    createDensityPointPass(graph, {
      id: `${id}-clear`,
      operation: 'GPUKernelDensity',
      invocationCount: 1,
      bindings: [{name: 'invalidCount', view: output, type: 'u32', access: 'read_write'}],
      body: 'invalidCount[invalidCountOffset] = 0u;'
    }),
    ...getGraphViewChunks(excluded).flatMap((chunk, chunkIndex) =>
      chunk.length === 0
        ? []
        : [
            createDensityPointPass(graph, {
              id: `${id}-${chunkIndex}`,
              operation: 'GPUKernelDensity',
              invocationCount: chunk.length,
              guardIndex: false,
              workgroupSize: POINT_DENSITY_WORKGROUP_SUM_SIZE,
              bindings: [
                {name: 'excluded', view: chunk, type: 'u32', access: 'read'},
                {name: 'invalidCount', view: output, type: 'atomic<u32>', access: 'read_write'}
              ],
              declarations: 'var<workgroup> excludedTotal: atomic<u32>;',
              body: `if (localInvocationIndex == 0u) { atomicStore(&excludedTotal, 0u); }
  workgroupBarrier();
  if (index < INVOCATION_COUNT && excluded[excludedOffset + index] != 0u) {
    atomicAdd(&excludedTotal, 1u);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&invalidCount[invalidCountOffset], atomicLoad(&excludedTotal));
  }`
            })
          ]
    )
  ];
}

/** Packed graphs keep their established reductions; split only when global scalar scratch cannot bind. */
export function needsDensitySplitScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  rows: number
): boolean {
  const byteLength = Math.max(1, rows) * 4;
  return (
    byteLength > graph.device.limits.maxStorageBufferBindingSize ||
    byteLength > graph.device.limits.maxBufferSize
  );
}
