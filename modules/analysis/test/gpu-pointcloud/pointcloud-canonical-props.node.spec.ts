// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudICP,
  type GPUPointCloudICPProps
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-icp';
import {
  GPUPointCloudSMRF,
  type GPUPointCloudSMRFProps
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-smrf';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('CORE-14: maximumIterations resolves once and sizes history and actual iteration nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x3' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `icp-alias-${serial++}`, format, length);
  const create = (
    settings: Pick<GPUPointCloudICPProps, 'maximumIterations'>,
    historyLength: number
  ) =>
    new GPUPointCloudICP({
      id: `icp-alias-${serial++}`,
      source: view('float32x3', 3),
      target: view('float32x3', 3),
      maximumDistance: 1,
      ...settings,
      output: {
        transform: view('float32', 16),
        history: view('float32x3', historyLength),
        inlierMask: view('uint32', 3),
        status: view('uint32', 1)
      }
    });
  try {
    const settings = {maximumIterations: 2};
    const contributorWithBudget = create(settings, 2);
    expect(contributorWithBudget.maximumIterations).toBe(2);
    const nodes = contributorWithBudget.getCommandNodes(graph);
    expect(nodes.filter(node => /-moments-\d+$/.test(node.id))).toHaveLength(2);
    expect(nodes.filter(node => /-solve-\d+-chunk-0$/.test(node.id))).toHaveLength(2);
    expect(() => create(settings, 1)).toThrow(/history/);
    expect(() => create(settings, 3)).toThrow(/history/);
    expect(create({}, 20).maximumIterations).toBe(20);
    expect(create({maximumIterations: 1}, 1).maximumIterations).toBe(1);
    expect(create({maximumIterations: 100}, 100).maximumIterations).toBe(100);
    const retiredBudget = {iterations: 2, maximumIterations: 2};
    expect(() => create(retiredBudget, 2)).toThrow(
      /icp-alias-\d+.*iterations.*removed.*use maximumIterations/
    );
    for (const value of [0, 101, 1.5]) {
      expect(() => create({maximumIterations: value}, 1)).toThrow(/maximumIterations.*1.*100/);
    }
    let reads = 0;
    // The getter must be on the final constructor input, after any spreading has finished.
    const input = {
      id: `icp-alias-${serial++}`,
      source: view('float32x3', 3),
      target: view('float32x3', 3),
      maximumDistance: 1,
      get maximumIterations() {
        reads++;
        return 2;
      },
      output: {
        transform: view('float32', 16),
        history: view('float32x3', 2),
        inlierMask: view('uint32', 3),
        status: view('uint32', 1)
      }
    };
    expect(Object.getOwnPropertyDescriptor(input, 'maximumIterations')?.get).toBeTypeOf('function');
    expect(reads).toBe(0);
    const contributor = new GPUPointCloudICP(input);
    contributor.getCommandNodes(graph);
    expect(reads).toBe(1);
  } finally {
    device.destroy();
  }
});

it('CORE-14: SMRF mask binds both consumers with nonzero-keeps semantics', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x3' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `smrf-alias-${serial++}`, format, length);
  const mask = view('uint32', 4);
  try {
    const contributor = new GPUPointCloudSMRF({
      id: 'smrf-mask',
      positions: view('float32x3', 4),
      classifications: view('uint32', 4),
      grid: {origin: [0, 0], cellSize: 1, width: 2, height: 2},
      window: 1,
      mask,
      output: {
        groundSurface: view('float32', 4),
        groundMask: view('uint32', 4),
        outputClassifications: view('uint32', 4),
        status: view('uint32', 4)
      }
    });
    expect(contributor.props.mask).toBe(mask);
    const records = contributor
      .getCommandNodes(graph)
      .map(node => getWGSLKernelNodeRecord(graph, node.id));
    const selected = records.filter(record =>
      record?.bindings.some(binding => binding.name === 'selection')
    );
    expect(selected).toHaveLength(2); // minimum raster and classification both consume the resolved mask
    for (const record of selected) {
      expect(record!.bindings.find(binding => binding.name === 'selection')?.view).toBe(mask);
      expect(record!.source).toMatch(/readSelection\([^)]*\)\s*==\s*0u/);
    }
  } finally {
    device.destroy();
  }
});

it('S3: SMRF diagnostics name the effective mask spelling for lengths and packed views', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x3' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `mask-diagnostic-${serial++}`, format, length);
  const create = (settings: Pick<GPUPointCloudSMRFProps, 'mask'>) =>
    new GPUPointCloudSMRF({
      positions: view('float32x3', 5),
      classifications: view('uint32', 5),
      grid: {origin: [0, 0], cellSize: 1, width: 2, height: 2},
      window: 1,
      ...settings,
      output: {
        groundSurface: view('float32', 4),
        groundMask: view('uint32', 5),
        outputClassifications: view('uint32', 5),
        status: view('uint32', 5)
      }
    });
  try {
    expect(() => create({mask: view('uint32', 4)})).toThrow(
      'gpu-pointcloud-smrf mask must have 5 rows'
    );
    const strided = graph.createDataView(view('uint32', 10).buffer, {
      format: 'uint32',
      length: 5,
      byteStride: 8
    });
    expect(() => create({mask: strided})).toThrow(/gpu-pointcloud-smrf mask.*packed/);
  } finally {
    device.destroy();
  }
});
