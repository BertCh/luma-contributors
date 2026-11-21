// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {EXACT_ORIENTATION_WGSL} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';
import {POLYGON_OVERLAP_METRIC_WGSL} from '../../../src/gpu-spatial-analysis/point-process/polygon-overlap-metric';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputBuffer, readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('R19 overlap metric retains determinant magnitude at near-collinearity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffer = createOutputBuffer(device, 10);
  const output = importGraphBuffer(graph, 'determinants', buffer, 'float32', 10);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'overlap-metric-probe',
      operation: 'OverlapMetricProbe',
      invocationCount: 5,
      bindings: [{name: 'result', view: output, type: 'f32', access: 'read_write'}],
      declarations: `${EXACT_ORIENTATION_WGSL}\n${POLYGON_OVERLAP_METRIC_WGSL}`,
      body: `let first=vec2f(0.0);
 var second=vec2f(8192.0,8191.0); var third=vec2f(8191.0,8190.0);
 if(index==1u){second=vec2f(8191.0,8190.0);third=vec2f(8192.0,8191.0);}
 if(index==2u){second=vec2f(8192.0);third=vec2f(8191.0);}
 if(index==3u){second=vec2f(16384.0,16382.0);third=vec2f(16382.0,16380.0);}
 if(index==4u){second=vec2f(16382.0,16380.0);third=vec2f(16384.0,16382.0);}
 result[resultOffset+index*2u]=overlapExactDeterminant(first,second,third);
 result[resultOffset+index*2u+1u]=f32(orientSign(first,second,third));`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Exact integer arithmetic gives determinants -1,+1,0. A sign-only substitute would
    // need a new shared magnitude API before replacing the overlap metric's integer sums.
    // The scaled pair has exact determinant -4 (reverse +4), distinguishing magnitude
    // from orientSign's -1/+1, unlike the unit-determinant fixtures alone (review B6).
    expect(await readFloat32(buffer, 10)).toEqual([-1, -1, 1, 1, 0, 0, -4, -1, 4, 1]);
  } finally {
    compiled.destroy();
    buffer.destroy();
  }
});
