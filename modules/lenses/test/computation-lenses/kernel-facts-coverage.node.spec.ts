// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {
  GPUPointCloudFeatures,
  GPUPointCloudNeighborhoods,
  GPUPointCloudStatisticalFilter
} from '@luma-contributors/analysis/pointcloud-analysis';
import {describeGPUCommandGraph} from '@luma-contributors/analysis/computation-record';
import {LensCommandGraph} from '../../src/computation-lenses/lens-view-snapshot';
import {declareLensScenario} from './scenario-declaration.test-utils';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import '../../src/computation-lenses/scenarios/gpu-dataframe/class-breaks.lens';

test('classification collective kernels retain facts from successful compilation', async () => {
  const scenario = getLensScenario('gpu-dataframe.class-breaks');
  expect(scenario).toBeDefined();
  if (!scenario) throw new Error('Missing class-breaks scenario');
  const {record} = await declareLensScenario(scenario);
  const missing = record.graphs
    .flatMap(graph => graph.nodes)
    .filter(node => !node.source && (node.op ?? node.workload?.operation) === 'GPUClassBreaks')
    .map(node => node.id);
  expect(missing).toEqual([]);
});

test('the point-cloud filter identifies every upstream kernel without recorded facts', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
  });
  const graph = new LensCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x3', 5);
  const indices = createTransientView(graph, 'indices', 'uint32', 20);
  const distances = createTransientView(graph, 'distances', 'float32', 20);
  const counts = createTransientView(graph, 'counts', 'uint32', 5);
  graph.add(
    new GPUPointCloudNeighborhoods({
      positions,
      radius: 3,
      capacity: 4,
      output: {indices, squaredDistances: distances, counts}
    })
  );
  graph.add(
    new GPUPointCloudFeatures({
      positions,
      indices,
      counts,
      capacity: 4,
      output: {
        normals: createTransientView(graph, 'normals', 'float32x3', 5),
        eigenvalues: createTransientView(graph, 'eigenvalues', 'float32x3', 5),
        shapes: createTransientView(graph, 'shapes', 'float32x3', 5),
        status: createTransientView(graph, 'status', 'uint32', 5)
      }
    })
  );
  graph.add(
    new GPUPointCloudStatisticalFilter({
      squaredDistances: distances,
      counts,
      capacity: 4,
      output: {
        meanDistances: createTransientView(graph, 'means', 'float32', 5),
        validity: createTransientView(graph, 'validity', 'uint32', 5)
      }
    })
  );
  const compiled = graph.compile();
  try {
    const {graph: described} = describeGPUCommandGraph(compiled, {graph});
    const missing = described.nodes
      .filter(node => node.kind === 'compute' && !node.source)
      .map(node => `${node.op ?? node.workload?.operation ?? 'unnamed'}:${node.id}`);
    // Published reduction stages already have the same explicit exemption in kernel-records.spec.ts.
    expect(missing.filter(name => !/^unnamed:.*-(level-\d+|finalize)$/.test(name))).toEqual([]);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});
