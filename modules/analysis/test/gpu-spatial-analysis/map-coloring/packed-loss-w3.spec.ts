// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUMapColoring,
  GPU_MAP_COLORING_UNCOLORED
} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {GPUMapColoringOperation} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation';
import {createFillNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('AMG-54 restores packed scalar outputs across clear set cleared loss frames', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const packed = rig.output('uint32', 2);
  const colorCount = rig.graph.createDataView(packed.view.buffer, {format: 'uint32', length: 1});
  const conflictCount = rig.graph.createDataView(packed.view.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 1
  });
  const colors = rig.output('uint32', 2);
  const converged = rig.output('uint32', 1);
  const iterationCount = rig.output('uint32', 1);
  const overflow = rig.input(Uint32Array.of(0), 'uint32', 1);
  const weights = rig.uploadWeights({
    offsets: [0, 1, 2],
    neighbors: [1, 0],
    weights: [1, 1],
    distances: []
  });
  const ports: Record<string, GraphDataView> = {
    offsets: weights.offsets,
    neighbors: weights.neighbors,
    weights: weights.weights,
    overflow,
    parameters: rig.input(Uint32Array.of(0), 'uint32', 1),
    colors: colors.view,
    colorCount,
    conflictCount,
    converged: converged.view,
    iterationCount: iterationCount.view
  };
  const operation = new GPUMapColoringOperation({
    rowCount: 2,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'loss'}
    },
    parameters: 'parameters',
    output: {
      colors: 'colors',
      colorCount: 'colorCount',
      conflictCount: 'conflictCount',
      status: {converged: 'done', iterationCount: 'rounds'}
    }
  });
  const contributor = operation.createContributor(name => ports[name] as never);
  if (!(contributor instanceof GPUMapColoring))
    throw new Error('Expected a map coloring contributor');
  const nodes = [...contributor.getCommandNodes(rig.graph)];
  const firstSuppression = nodes.findIndex(node => node.id.includes('-incomplete-'));
  // Isolate diagnostic suppression: a nonzero word must be cleared, not merely left untouched.
  nodes.splice(
    firstSuppression,
    0,
    createFillNode(rig.graph, {
      id: 'poison-conflicts',
      operation: 'test',
      view: conflictCount,
      type: 'u32',
      value: '23u'
    })
  );
  rig.graph.add(nodes);
  const compiled = rig.graph.compile();
  try {
    for (const loss of [0, 1, 0]) {
      rig.bufferOf(overflow).write(Uint32Array.of(loss));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(packed.buffer, 2)).toEqual(loss ? [0, 0] : [2, 23]);
      const actualColors = await readUint32(colors.buffer, 2);
      // An edge needs exactly two distinct colors; either priority order is valid.
      expect(actualColors.sort((left, right) => left - right)).toEqual(
        loss ? [GPU_MAP_COLORING_UNCOLORED, GPU_MAP_COLORING_UNCOLORED] : [0, 1]
      );
      expect(await readUint32(converged.buffer, 1)).toEqual([loss ? 0 : 1]);
      // Atomic select/apply can finish the lower-priority row in the same round or the next.
      // Suppression must retain this real work count, rather than zeroing it with convergence.
      expect([1, 2]).toContain((await readUint32(iterationCount.buffer, 1))[0]);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
