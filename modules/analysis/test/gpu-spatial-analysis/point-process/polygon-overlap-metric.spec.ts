// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {EXACT_ORIENTATION_WGSL} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';
import {POLYGON_OVERLAP_METRIC_WGSL} from '../../../src/gpu-spatial-analysis/point-process/polygon-overlap-metric';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('polygon clipping determinant retains unit residuals below rounded product precision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  // Exact integer determinants are +1,-1,+1,-1. The first arrangement forces
  // cancellation of products near1e10; the second is the actual clipping orientation.
  const source = device.createBuffer({
    data: new Float32Array([
      100000, 99999, 100001, 100000, 0, 0, 100000, 99999, 99999, 99998, 0, 0, 0, 0, 100000, 99999,
      100001, 100000, 0, 0, 100000, 99999, 99999, 99998
    ]),
    usage: Buffer.STORAGE
  });
  const output = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE | Buffer.COPY_SRC});
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'overlap-metric-test',
      operation: 'GPUInhomogeneousPointPattern',
      variant: 'metric-regression',
      invocationCount: 4,
      bindings: [
        {
          name: 'source',
          view: importGraphBuffer(graph, 'source', source, 'float32', 24),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'output',
          view: importGraphBuffer(graph, 'output', output, 'float32', 4),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `${EXACT_ORIENTATION_WGSL}\n${POLYGON_OVERLAP_METRIC_WGSL}`,
      body: `let base = sourceOffset + index * 6u;
  output[outputOffset + index] = overlapDeterminant(vec2f(source[base],source[base+1u]),vec2f(source[base+2u],source[base+3u]),vec2f(source[base+4u],source[base+5u]));`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(output, 4)).toEqual([1, -1, 1, -1]);
  } finally {
    compiled.destroy();
    source.destroy();
    output.destroy();
  }
});
