// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {SpatialPredicateSide} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createMultipartCPUGeometry,
  type MultipartFixtureGeometry
} from './multipart-predicates-test-utils';

export function runMultipartScalar(
  leftInput: MultipartFixtureGeometry,
  rightInput: MultipartFixtureGeometry,
  declarations: (left: SpatialPredicateSide, right: SpatialPredicateSide) => string,
  expression: string
): number {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    const bindings: WGSLKernelBinding[] = [];
    const describe = (
      prefix: 'left' | 'right',
      input: MultipartFixtureGeometry
    ): SpatialPredicateSide => {
      const geometry = createMultipartCPUGeometry(input);
      const bind = (suffix: string, values: Float32Array | Uint32Array) => {
        const name = prefix + suffix,
          positions = suffix === 'Positions';
        const view = createTransientView(
          graph,
          name,
          positions ? 'float32x2' : 'uint32',
          Math.max(1, values.length / (positions ? 2 : 1))
        );
        memory.write(view, values);
        bindings.push({name, view, type: positions ? 'f32' : 'u32', access: 'read'});
      };
      bind('Positions', geometry.positions);
      if (geometry.kind === 'polygons') {
        bind('RingOffsets', geometry.ringOffsets!);
        bind('FeatureRings', Uint32Array.of(0, geometry.ringOffsets!.length - 1));
      } else {
        if (geometry.lineOffsets) bind('RingOffsets', geometry.lineOffsets);
        if (geometry.featureOffsets) bind('FeatureOffsets', geometry.featureOffsets);
      }
      return {
        prefix,
        kind: geometry.kind,
        vertexCount: geometry.positions.length / 2,
        ...(geometry.kind !== 'polygons' && geometry.featureOffsets
          ? {
              partCount:
                geometry.kind === 'points'
                  ? geometry.positions.length / 2
                  : geometry.lineOffsets!.length - 1
            }
          : {})
      };
    };
    const left = describe('left', leftInput),
      right = describe('right', rightInput);
    const output = createTransientView(graph, 'result', 'uint32', 1);
    const node = createWGSLKernelNode(graph, {
      id: 'review-scalar',
      operation: 'MultipartReview',
      bindings: [...bindings, {name: 'result', view: output, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      declarations: declarations(left, right),
      body: `result[resultOffset] = ${expression};`
    });
    executeCommandNodesOnCPU([node], memory);
    return memory.readUint32(output)[0];
  } finally {
    device.destroy();
  }
}
