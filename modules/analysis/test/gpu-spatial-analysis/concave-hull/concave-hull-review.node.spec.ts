// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGroupConcaveHull} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createConcaveHullValidationProps} from './concave-hull-validation-cases';

function createPackedProps(graph: GPUCommandGraph) {
  const base = createConcaveHullValidationProps(graph, 6, 2, 7);
  return {
    ...base,
    layout: 'packed' as const,
    groupCount: 2,
    groupIds: createTransientView(graph, 'labels', 'uint32', 6),
    output: {
      ...base.output,
      status: {
        ...base.output.status,
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      },
      groups: {
        triangleOffsets: createTransientView(graph, 'triangle-offsets', 'uint32', 2),
        triangleCounts: createTransientView(graph, 'triangle-counts', 'uint32', 2),
        edgeOffsets: createTransientView(graph, 'edge-offsets', 'uint32', 2),
        edgeCounts: createTransientView(graph, 'edge-counts', 'uint32', 2)
      }
    }
  };
}

for (const failure of ['duplicate', 'input', 'missing', 'usage'] as const) {
  it(`R1 validates packed group ranges: ${failure}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const props = createPackedProps(graph);
      if (failure === 'duplicate')
        props.output.groups.triangleCounts = props.output.groups.triangleOffsets;
      if (failure === 'input')
        props.output.groups.edgeCounts = graph.createDataView(props.groupIds.buffer, {
          format: 'uint32',
          length: 2
        });
      if (failure === 'missing') Reflect.deleteProperty(props.output.groups, 'edgeCounts');
      if (failure === 'usage') {
        const buffer = graph.createTransientBuffer({
          id: 'non-storage',
          byteLength: 8,
          usage: Buffer.COPY_SRC | Buffer.COPY_DST
        });
        props.output.groups.edgeCounts = graph.createDataView(buffer, {
          format: 'uint32',
          length: 2
        });
      }
      const construct = () => new GPUGroupConcaveHull(props);
      if (failure === 'usage') {
        const contributor = construct();
        expect(() => contributor.getCommandNodes(graph)).toThrow(
          /^group-concave-hull .*buffer usage must include Buffer.STORAGE/
        );
      } else {
        const expected =
          failure === 'duplicate'
            ? /distinct buffers/
            : failure === 'input'
              ? /must not share buffers with inputs/
              : /output.groups.edgeCounts is required/;
        expect(construct).toThrow(expected);
        expect(construct).toThrow(/^group-concave-hull /);
      }
    } finally {
      device.destroy();
    }
  });
}

it('R6 snapshots packed layout and nested output maps while preserving GPU view identities', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createPackedProps(graph);
    const originalGroups = {...props.output.groups};
    const originalStatus = {...props.output.status};
    const originalTriangles = props.output.triangles;
    const contributor = new GPUGroupConcaveHull(props);
    Reflect.set(props, 'layout', 'fixed');
    props.output.groups.edgeCounts = createTransientView(graph, 'replacement-range', 'uint32', 2);
    props.output.status.count = createTransientView(graph, 'replacement-count', 'uint32', 1);
    props.output.triangles = createTransientView(graph, 'replacement-triangles', 'uint32x3', 14);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.some(node => node.id.endsWith('-publish-packed-status'))).toBe(true);
    expect(nodes.some(node => node.id.endsWith('-triangles-publish'))).toBe(false);
    expect(contributor.props.output.triangles).toBe(originalTriangles);
    const ranges = nodes.find(node => node.id.endsWith('-edge-group-ranges'))!;
    const status = nodes.find(node => node.id.endsWith('-publish-packed-status'))!;
    expect(
      ranges.resources?.some(
        resource => 'buffer' in resource && resource.buffer === originalGroups.edgeCounts
      )
    ).toBe(true);
    expect(
      status.resources?.some(
        resource => 'buffer' in resource && resource.buffer === originalStatus.count
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});
