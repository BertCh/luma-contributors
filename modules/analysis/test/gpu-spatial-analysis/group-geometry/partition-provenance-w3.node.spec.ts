// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGroupGeometry} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-geometry';
import {GPUGroupConvexHull} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull';
import {GPUGroupConcaveHull} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {GPULabelPoint} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('CPL-14 publishes provisional and complete partition words without changing geometric overflow', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length = 1) =>
    createTransientView(graph, `provenance-${sequence++}`, format, length);
  const positions = view('float32x2', 4),
    labels = view('uint32', 4),
    labelsComplete = view('uint32');
  const outputComplete = view('uint32');
  const producers = [
    new GPUGroupGeometry({
      positions,
      labels,
      labelsComplete,
      groupCount: 4,
      parameters: view('float32', 8),
      output: {bounds: view('float32x4', 4), labelsComplete: outputComplete}
    }),
    new GPUGroupConvexHull({
      positions,
      labels,
      labelsComplete,
      groupCount: 4,
      maximumVerticesPerGroup: 4,
      vertexCapacity: 4,
      output: {
        vertexIndices: view('uint32', 4),
        offsets: view('uint32', 5),
        counts: view('uint32', 4),
        labelsComplete: outputComplete,
        status: {overflow: view('uint32')}
      }
    }),
    new GPUMinimumBounds({
      positions,
      labels,
      labelsComplete,
      groupCount: 4,
      output: {
        sourceBounds: view('float32x4', 4),
        labelsComplete: outputComplete,
        precisionLimited: view('uint32', 4),
        status: {overflow: view('uint32')}
      }
    }),
    new GPUGroupConcaveHull({
      positions,
      groupIds: labels,
      labelsComplete,
      groupCount: 4,
      maximumCircumradius: 2,
      output: {
        triangles: view('uint32x3', 12),
        triangleValidity: view('uint32', 12),
        boundaryEdges: view('uint32x2', 36),
        edgeValidity: view('uint32', 36),
        labelsComplete: outputComplete,
        status: {converged: view('uint32', 4)}
      }
    })
  ];
  try {
    for (const producer of producers) {
      expect(producer.partitionProvenance.labelsComplete).toBe(outputComplete);
      const nodes = producer.getCommandNodes(graph);
      const publish = nodes.find(node => node.id === `${producer.id}-partition-provenance`);
      expect(publish).toBeDefined();
      const memory = createCPUGraphMemory();
      for (const word of [0, 1, 7, 0]) {
        memory.write(labelsComplete, new Uint32Array([word]));
        memory.write(outputComplete, new Uint32Array([99]));
        executeCommandNodesOnCPU([publish!], memory);
        expect(memory.readUint32(outputComplete)[0]).toBe(word === 0 ? 0 : 1);
      }
      for (const source of getGPUCommandNodeKernelSources(
        nodes.filter(
          node => node.id.includes('partition-provenance') || node.id.includes('source-precision')
        )
      ))
        expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('UGE-04 and UGE-05 reject retired aliases and declare publication kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length = 1) =>
    createTransientView(graph, `aliases-${sequence++}`, format, length);
  try {
    const positions = view('float32x2', 4),
      labels = view('uint32', 4);
    const sourceBounds = view('float32x4');
    const props = {
      positions,
      labels,
      groupCount: 1,
      output: {sourceBounds, status: {overflow: view('uint32')}}
    };
    const legacyBounds = {...props, output: {...props.output, bounds: sourceBounds}};
    expect(() => new GPUMinimumBounds(legacyBounds)).toThrow(/bounds.*sourceBounds/);
    expect(() => new GPUMinimumBounds(props)).not.toThrow();
    const publishedCenterClearance = view('float32');
    const label = {
      positions,
      ringOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      featureOffsets: view('uint32', 2),
      output: {
        points: view('float32x2'),
        publishedCenterClearance,
        precisionLimited: view('uint32')
      }
    };
    const legacyLabel = {...label, output: {...label.output, distances: publishedCenterClearance}};
    expect(() => new GPULabelPoint(legacyLabel)).toThrow(/distances.*publishedCenterClearance/);
    const nodes = new GPULabelPoint(label).getCommandNodes(graph);
    const publish = nodes.find(node => node.id.endsWith('-published-clearance'));
    expect(publish).toBeDefined();
    for (const source of getGPUCommandNodeKernelSources([publish!]))
      expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
  } finally {
    device.destroy();
  }
});
