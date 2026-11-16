// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Leaves headroom for rounding in the feature area reduction; keeps the existing quantization. */
const AREA_FIXED_POINT_TOTAL = '4.0e9';

/**
 * Inclusive, saturated fixed-point triangle CDF for consecutive feature ranges. @internal
 *
 * Ordering proof: polygon-triangulation-layout scans one capacity per polygon into
 * polygonTriangleOffsets, and both triangulation routes write only that polygon's slice.
 * The sampler maps featureOffsets[f] through those polygon offsets. A GeoArrow feature owns
 * consecutive polygons, so its triangles form one interval; empty polygons/features give equal
 * offsets and cannot interleave triangles of other features. Upper-bound lookup skips all such
 * empty intervals. Padding after the terminal offset starts independent zero segments.
 *
 * Quantization is unchanged. Integer addition is associative, so the published segmented scan
 * preserves every prefix of the serial route when the sum fits uint32. A second segmented scan
 * propagates carry detection, saturating the CDF after any wrap instead of making it decrease.
 * This also protects against underestimated feature areas; no floating-point headroom assumption
 * is needed for monotonicity. Carry counts fit uint32 because there are at most uint32 triangles.
 */
export function getDotTriangleCumulativeAreaNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    triangleAreas: GraphDataView<'float32'>;
    featureTriangleOffsets: GraphDataView<'uint32'>;
    featureValid: GraphDataView<'uint32'>;
    featureAreas: GraphDataView<'float32'>;
    cumulativeAreas: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, triangleAreas, featureTriangleOffsets, featureValid, featureAreas} = props;
  const triangleCount = triangleAreas.length;
  validateGraphChunkBindingRanges(graph, id, {
    triangleAreas,
    featureTriangleOffsets,
    featureValid,
    featureAreas,
    cumulativeAreas: props.cumulativeAreas
  });
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'triangle CDF scratch',
    Math.max(triangleCount, 1) * 4,
    'reduce the triangle count'
  );
  const quantizedAreas = createTransientView(graph, `${id}-quantized`, 'uint32', triangleCount);
  const segmentFlags = createTransientView(graph, `${id}-segments`, 'uint32', triangleCount);
  const carryFlags = createTransientView(graph, `${id}-carries`, 'uint32', triangleCount);
  const carryPrefixes = createTransientView(graph, `${id}-carry-prefixes`, 'uint32', triangleCount);
  // The caller may store mask maxima after the triangles. Scans must leave that tail alone.
  const cumulativeAreas = graph.createDataView(props.cumulativeAreas.buffer, {
    format: 'uint32',
    byteOffset: props.cumulativeAreas.byteOffset,
    length: triangleCount
  });
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-quantize`,
      operation,
      variant: 'triangle-cdf-quantize',
      bindings: [
        {name: 'triangleAreas', view: triangleAreas, type: 'f32', access: 'read'},
        {name: 'featureTriangleOffsets', view: featureTriangleOffsets, type: 'u32', access: 'read'},
        {name: 'featureValid', view: featureValid, type: 'u32', access: 'read'},
        {name: 'featureAreas', view: featureAreas, type: 'f32', access: 'read'},
        {name: 'quantizedAreas', view: quantizedAreas, type: 'u32', access: 'read_write'},
        {name: 'segmentFlags', view: segmentFlags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: triangleCount,
      declarations: IS_FINITE_BITS_WGSL,
      body: `let featureCount = arrayLength(&featureAreas) - featureAreasOffset;
  var low = 0u;
  var high = featureCount;
  // Last feature start at or before this triangle, including repeated offsets.
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (featureTriangleOffsets[featureTriangleOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  var quantized = 0u;
  var startsSegment = 1u;
  if (low < featureCount && index >= featureTriangleOffsets[featureTriangleOffsetsOffset]) {
    startsSegment = select(0u, 1u, index == featureTriangleOffsets[featureTriangleOffsetsOffset + low]);
    let featureArea = featureAreas[featureAreasOffset + low];
    let area = triangleAreas[triangleAreasOffset + index];
    if (featureValid[featureValidOffset + low] == 1u && isFiniteBits(featureArea) && featureArea > 0.0 && isFiniteBits(area)) {
      quantized = u32(clamp(area / featureArea, 0.0, 1.0) * ${AREA_FIXED_POINT_TOTAL});
    }
  }
  quantizedAreas[quantizedAreasOffset + index] = quantized;
  segmentFlags[segmentFlagsOffset + index] = startsSegment;`
    }),
    ...new GPUScan({
      id: `${id}-scan`,
      input: quantizedAreas,
      output: cumulativeAreas,
      mode: 'inclusive',
      segmentFlags
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-detect-carry`,
      operation,
      variant: 'triangle-cdf-detect-carry',
      bindings: [
        {name: 'quantizedAreas', view: quantizedAreas, type: 'u32', access: 'read'},
        {name: 'cumulativeAreas', view: cumulativeAreas, type: 'u32', access: 'read'},
        {name: 'carryFlags', view: carryFlags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: triangleCount,
      body: `carryFlags[carryFlagsOffset + index] = select(0u, 1u, cumulativeAreas[cumulativeAreasOffset + index] < quantizedAreas[quantizedAreasOffset + index]);`
    }),
    ...new GPUScan({
      id: `${id}-carry-scan`,
      input: carryFlags,
      output: carryPrefixes,
      mode: 'inclusive',
      segmentFlags
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id,
      operation,
      variant: 'triangle-cdf-saturate',
      bindings: [
        {name: 'carryPrefixes', view: carryPrefixes, type: 'u32', access: 'read'},
        {name: 'cumulativeAreas', view: cumulativeAreas, type: 'u32', access: 'read_write'}
      ],
      invocationCount: triangleCount,
      body: `if (carryPrefixes[carryPrefixesOffset + index] != 0u) {
    cumulativeAreas[cumulativeAreasOffset + index] = 0xffffffffu;
  }`
    })
  ];
}
