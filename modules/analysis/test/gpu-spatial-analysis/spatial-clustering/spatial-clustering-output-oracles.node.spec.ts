// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUDBSCAN} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-clustering';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const alias of ['labels', 'parameters', 'clusterCount', 'count'] as const) {
  it(`clustering refuses candidateOverflow aliasing ${alias}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const labels = createTransientView(graph, 'labels', 'uint32', 4);
      const parameters = createTransientView(graph, 'parameters', 'float32', 8);
      const count = createTransientView(graph, 'count', 'uint32', 1);
      const clusterCount = createTransientView(graph, 'clusterCount', 'uint32', 1);
      const base =
        alias === 'labels'
          ? labels
          : alias === 'parameters'
            ? parameters
            : alias === 'count'
              ? count
              : clusterCount;
      const candidateOverflow = graph.createDataView(base.buffer, {
        format: 'uint32',
        byteOffset: alias === 'labels' ? 8 : 0,
        length: 1
      });
      // Labels [0,0,1,1] must survive publication of candidate loss zero: a flag
      // alias at byte8 would silently replace the third label with zero.
      expect(
        () =>
          new GPUDBSCAN({
            id: 'clustering-alias-oracle',
            positions: createTransientView(graph, 'positions', 'float32x2', 4),
            parameters,
            gridSize: [4, 1],
            output: {
              labels,
              clusterCount,
              clusters: {
                ids: createTransientView(graph, 'ids', 'uint32', 2),
                status: {
                  count,
                  overflow: createTransientView(graph, 'overflow', 'uint32', 1),
                  candidateOverflow
                }
              }
            }
          })
      ).toThrow(/overlap|different physical buffers|share buffers with inputs/);
    } finally {
      device.destroy();
    }
  });
}

it('clustering retains the validated candidate flag after caller replaces it with a label alias', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const labels = createTransientView(graph, 'labels', 'uint32', 4);
    const originalFlag = createTransientView(graph, 'original-flag', 'uint32', 1);
    const status = {
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1),
      candidateOverflow: originalFlag
    };
    const clusters = {ids: createTransientView(graph, 'ids', 'uint32', 2), status};
    const output = {labels, clusters};
    const props = {
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      gridSize: [4, 1] as const,
      output
    };
    const contributor = new GPUDBSCAN(props);
    const alias = graph.createDataView(labels.buffer, {format: 'uint32', byteOffset: 8, length: 1});
    status.candidateOverflow = alias;
    clusters.status = {...status};
    output.clusters = {...clusters};
    props.output = {...output};
    const resources = contributor.getCommandNodes(graph).flatMap(node => node.resources ?? []);
    expect(
      resources.some(
        resource =>
          'buffer' in resource &&
          resource.buffer === originalFlag &&
          resource.usage === 'storage-read-write'
      )
    ).toBe(true);
    expect(resources.some(resource => 'buffer' in resource && resource.buffer === alias)).toBe(
      false
    );
  } finally {
    device.destroy();
  }
});
