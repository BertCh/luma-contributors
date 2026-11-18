// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {expect, it, vi} from 'vitest';
import {GPUNetworkSnapping} from '../../../src/gpu-network/network-accessibility/gpu-network-snapping';
import {
  createKernelAuditContext,
  getDeclaredNodes
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getParsedKernelSources} from '../../utils/gpu-contributor-test-utils';

function createFixture(
  points: number,
  edges: number,
  candidateCapacity: number,
  greatCircle = false
) {
  const context = createKernelAuditContext('snapping-preflight');
  const {view, device} = context;
  Object.assign(device.limits, {maxStorageBufferBindingSize: 512});
  const contributor = new GPUNetworkSnapping({
    positions: view('float32x2', points),
    topology: {
      offsets: view('uint32', edges + 2),
      neighbors: view('uint32', edges),
      nodePositions: view('float32x2', edges + 1)
    },
    maximumSnapDistance: view('float32', 1),
    candidateCapacity,
    spatialContext: greatCircle
      ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      : undefined,
    output: {snappedEdges: view('uint32', points), status: {overflow: view('uint32', 1)}}
  });
  return {context, contributor};
}

it('snapping treats candidateCapacity as total pairs, independently of the point count', () => {
  const {context, contributor} = createFixture(33, 4, 32);
  try {
    const nodes = contributor.getCommandNodes(context.graph);
    for (const {id: name, source, constants} of getParsedKernelSources(nodes).configurations) {
      expect(auditWGSLModule(source, constants), name).toEqual([]);
    }
    expect(() => nodes.forEach(node => context.graph.add(node))).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

for (const [name, points, edges, capacity, greatCircle, expected] of [
  ['candidate geometry', 33, 4, 65, false, /candidates needs 520 bytes/],
  ['rounded BVH', 2, 30, 4, false, /boundsAndChildren needs 504 bytes/],
  ['great-circle node table', 50, 4, 4, true, /nodeTable needs 540 bytes/]
] as const) {
  it(`snapping checks ${name} scratch before any graph mutation`, () => {
    const {context, contributor} = createFixture(points, edges, capacity, greatCircle);
    if (name === 'rounded BVH')
      Object.assign(context.device.limits, {maxStorageBufferBindingSize: 500});
    const create = vi.spyOn(context.graph, 'createTransientBuffer');
    try {
      expect(() => context.graph.add(contributor)).toThrow(expected);
      expect(create).not.toHaveBeenCalled();
      expect(getDeclaredNodes(context)).toHaveLength(0);
    } finally {
      context.device.destroy();
    }
  });
}

it('the great-circle route does not preflight unused planar candidate storage', () => {
  const {context, contributor} = createFixture(2, 4, 10000, true);
  try {
    expect(() => context.graph.add(contributor)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

it('snapping ignores unused optional weights without STORAGE usage', () => {
  const {context} = createFixture(2, 4, 4);
  const {view, graph} = context;
  const unused = graph.createTransientBuffer({
    id: 'unused-weights',
    byteLength: 16,
    usage: Buffer.COPY_DST
  });
  try {
    const contributor = new GPUNetworkSnapping({
      positions: view('float32x2', 2),
      topology: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 4),
        nodePositions: view('float32x2', 3),
        weights: graph.createDataView(unused, {format: 'float32', length: 4})
      },
      output: {snappedEdges: view('uint32', 2)}
    });
    expect(() => graph.add(contributor)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});
