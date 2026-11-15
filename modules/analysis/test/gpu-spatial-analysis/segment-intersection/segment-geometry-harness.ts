// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUSegmentGeometry} from '../../../src/gpu-spatial-analysis/segment-intersection/index';
import {createInputBuffer} from '../../utils/gpu-contributor-test-utils';
import {buildArrays, type OracleSegmentFeature} from './segment-intersection-oracle';

/** Uploads oracle features as GeoArrow-layout graph views; pushes the buffers for later cleanup. */
export function createSegmentGeometry(
  device: Device,
  graph: GPUCommandGraph,
  name: string,
  features: OracleSegmentFeature[],
  buffers: Buffer[],
  padding?: {leading: number; trailing: number; value: number}
): GPUSegmentGeometry {
  const arrays = buildArrays(features);
  if (padding) {
    const positions = new Float32Array(
      arrays.positions.length + 2 * (padding.leading + padding.trailing)
    ).fill(padding.value);
    positions.set(arrays.positions, 2 * padding.leading);
    arrays.positions = positions;
    if (arrays.kind === 'lines') {
      arrays.lineOffsets = Uint32Array.from(arrays.lineOffsets, offset => offset + padding.leading);
    } else {
      arrays.ringOffsets = Uint32Array.from(arrays.ringOffsets, offset => offset + padding.leading);
    }
  }
  const upload = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const positions = importGraphBuffer(
    graph,
    `${name}-positions`,
    upload(arrays.positions),
    'float32x2',
    arrays.positions.length / 2
  );
  const offsets = (suffix: string, data: Uint32Array) =>
    importGraphBuffer(graph, `${name}-${suffix}`, upload(data), 'uint32', data.length);
  if (arrays.kind === 'lines') {
    return {kind: 'lines', positions, lineOffsets: offsets('line-offsets', arrays.lineOffsets)};
  }
  return {
    kind: 'polygons',
    positions,
    featureOffsets: offsets('feature-offsets', arrays.featureOffsets),
    polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets),
    ringOffsets: offsets('ring-offsets', arrays.ringOffsets)
  };
}
