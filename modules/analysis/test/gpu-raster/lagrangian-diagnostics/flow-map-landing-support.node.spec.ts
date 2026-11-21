// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULagrangianFlowMap} from '../../../src/raster-analysis/lagrangian-diagnostics/gpu-lagrangian-flow-map';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('RT-124 FlowMap checks the last landing at the end time before publishing it', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const contributor = new GPULagrangianFlowMap({
    id: 'landing',
    seeds: createTransientView(graph, 'seeds', 'float32x2', 1),
    velocitiesStart: createTransientView(graph, 'start', 'float32x2', 4),
    velocitiesEnd: createTransientView(graph, 'end', 'float32x2', 4),
    fieldWidth: 4,
    fieldHeight: 1,
    stepCount: 1,
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    output: {positions: createTransientView(graph, 'positions', 'float32x2', 1)}
  });
  const nodes = contributor.getCommandNodes(graph);
  const integration = nodes
    .map(node => getWGSLKernelNodeRecord(graph, node.id))
    .find(record => record?.body.includes('position = position + timeStep * second.xy;'));
  expect(integration).toBeDefined();
  // Identical slices [3,3,3,NaN]: x=.5 -> midpoint 2 -> landing 3.5. Bounds alone
  // accept 3.5, but that cell has no velocity. The time of this sample must be t+h.
  expect(integration!.body).toMatch(
    /position = position \+ timeStep \* second.xy;[\s\S]*sampleVelocity\(position, extent, weightEnd\)\.z == 0\.0/
  );
  expect(integration!.body).toContain('time + timeStep - sliceStart');
});
