// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma-contributors/graph/gpu-core';
import {GPURasterWatershedSegmentation} from '../../src/gpu-raster/gpu-raster-watershed-segmentation';
import {makeGPURasterMetadata} from '../../src/gpu-raster/raster-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {calculateWatershedForestReference} from './watershed-forest-reference';

function createFixture(count: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  return {
    device,
    graph,
    props: {
      grid: makeGPURasterMetadata({width: count, height: 1}),
      input: {
        id: 'elevation',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: createTransientView(graph, 'elevation', 'float32', count)
        }
      },
      markers: createTransientView(graph, 'markers', 'uint32', count),
      output: {
        labels: createTransientView(graph, 'labels', 'uint32', count),
        status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
      }
    }
  };
}

it('forest preserves fixed markers and distinguishes its static plateau tie order from FIFO', () => {
  expect(
    calculateWatershedForestReference({
      width: 5,
      height: 1,
      values: new Float32Array(5),
      markers: Uint32Array.of(1, 0, 0, 0, 2)
    })
  ).toEqual([1, 1, 1, 1, 2]);
  expect(
    calculateWatershedForestReference({
      width: 5,
      height: 1,
      values: new Float32Array(5),
      markers: Uint32Array.of(1, 0, 0xffffffff, 0, 0)
    })
  ).toEqual([1, 1, 0, 0, 0]);
});

it('forest declares bounded parallel kernels beyond the FIFO cap without per-pixel graph unrolling', () => {
  const {device, graph, props} = createFixture(65536);
  try {
    expect(() => new GPURasterWatershedSegmentation(props)).toThrow();
    const contributor = new GPURasterWatershedSegmentation({
      ...props,
      method: 'minimum-spanning-forest'
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThan(500);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(1);
    const checked = new Set<string>();
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      if (checked.has(kernel.source)) continue;
      checked.add(kernel.source);
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('watershed rejects mixed method budgets and invalid forest rounds', () => {
  const {device, props} = createFixture(16);
  try {
    expect(
      () =>
        new GPURasterWatershedSegmentation({
          ...props,
          method: 'minimum-spanning-forest',
          maximumSteps: 16
        })
    ).toThrow();
    expect(() => new GPURasterWatershedSegmentation({...props, maximumIterations: 3})).toThrow();
    for (const maximumIterations of [0, -1, 1.5, 33])
      expect(
        () =>
          new GPURasterWatershedSegmentation({
            ...props,
            method: 'minimum-spanning-forest',
            maximumIterations
          })
      ).toThrow();
  } finally {
    device.destroy();
  }
});
