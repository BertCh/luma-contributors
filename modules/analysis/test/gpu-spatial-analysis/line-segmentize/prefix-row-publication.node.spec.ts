// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createLinePrefixNode} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {COMPENSATED_SUM_WGSL} from '../../../src/analysis-core/kernels/compensated-sum-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();
it('B1 publishes the last moving row with the tile-prefix addition error', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rowMeasures = createTransientView(graph, 'measures', 'float32', 193);
  const node = createLinePrefixNode(graph, {
    id: 'prefix',
    operation: 'GPULineSegmentize',
    coordinateSystem: 'planar',
    radius: 1,
    positions: createTransientView(graph, 'positions', 'float32x2', 193),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    rowMeasures
  });
  const source = getGPUCommandNodeKernelSources([node])[0].source;
  // Execute the actual generated publication block in isolation: the interpreter cannot run
  // the cooperative barriers. Carry at the start of tile three is (1024, 2^-14), and the
  // final moving lane adds another 64 * 2^-20 = 2^-14. Their exact total is one f32 ULP.
  const publication = source
    .slice(source.indexOf('tileLengths[lane] += addedLength;'))
    .match(/if \(row < lineEnd\) \{([\s\S]*?)\n        \}/)![1];
  const probe = createWGSLKernelNode(graph, {
    id: 'publication',
    operation: 'test',
    invocationCount: 1,
    bindings: [{name: 'rowMeasures', view: rowMeasures, type: 'f32', access: 'read_write'}],
    declarations: COMPENSATED_SUM_WGSL,
    body: `var prefixCarry: array<f32, 3>;
      prefixCarry[0] = 1024.0;
      prefixCarry[1] = 0.00006103515625;
      var tileLengths: array<f32, 64>;
      let lane = 63u;
      let row = 192u;
      tileLengths[lane] = 0.00006103515625;
      ${publication}`
  });
  try {
    const memory = createCPUGraphMemory();
    executeGeometryNodesOnCPU([probe], memory);
    const actual = memory.readFloat32(rowMeasures)[192];
    expect(Number.isFinite(actual)).toBe(true);
    expect(actual).toBe(1024.0001220703125);
  } finally {
    device.destroy();
  }
});
