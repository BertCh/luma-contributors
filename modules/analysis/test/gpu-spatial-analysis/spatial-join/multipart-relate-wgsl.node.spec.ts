// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getSpatialRelateWGSL} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-wgsl';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import type {SpatialPredicateSide} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {MULTIPART_PREDICATE_FIXTURES} from './multipart-predicates-fixtures';
import {
  createMultipartCPUGeometry,
  type MultipartFixtureGeometry
} from './multipart-predicates-test-utils';

// This executes the actual scalar relate shader. The headless companion checks cooperative
// lane merging and graph wiring on WebGPU; this interpreter result makes no hardware claim.
for (const fixture of MULTIPART_PREDICATE_FIXTURES) {
  it(`scalar WGSL matches Shapely: ${fixture.name}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const bindings: WGSLKernelBinding[] = [];
      const describe = (
        prefix: 'left' | 'right',
        input: MultipartFixtureGeometry
      ): SpatialPredicateSide => {
        const geometry = createMultipartCPUGeometry(input);
        const bind = (suffix: string, values: Float32Array | Uint32Array) => {
          const name = prefix + suffix;
          const view = createTransientView(
            graph,
            name,
            suffix === 'Positions' ? 'float32x2' : 'uint32',
            Math.max(1, values.length / (suffix === 'Positions' ? 2 : 1))
          );
          memory.write(view, values);
          bindings.push({name, view, type: suffix === 'Positions' ? 'f32' : 'u32', access: 'read'});
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
      const left = describe('left', fixture.left),
        right = describe('right', fixture.right);
      const output = createTransientView(graph, 'matrix', 'uint32', 1);
      const node = createWGSLKernelNode(graph, {
        id: 'matrix',
        operation: 'MultipartRelateTest',
        variant: fixture.name,
        bindings: [...bindings, {name: 'matrix', view: output, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        declarations: getSpatialRelateWGSL(left, right),
        body: 'matrix[matrixOffset] = pairRelate(0u, 0u);'
      });
      executeCommandNodesOnCPU([node], memory);
      const actual = memory.readUint32(output)[0];
      if (!fixture.left.positions.length || !fixture.right.positions.length) expect(actual).toBe(0);
      else {
        expect(actual >>> 31).toBe(0);
        expect(formatGPUSpatialRelate(actual)).toBe(fixture.matrix);
      }
    } finally {
      device.destroy();
    }
  });
}
