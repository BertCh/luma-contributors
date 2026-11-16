// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPULabelPoint,
  type GPULabelPointProps
} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createProps(graph: GPUCommandGraph, featureCount = 1): GPULabelPointProps {
  return {
    id: 'label',
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    featureOffsets: createTransientView(graph, 'features', 'uint32', featureCount + 1),
    output: {points: createTransientView(graph, 'points', 'float32x2', featureCount)}
  };
}

it('rejects overflowing dispatch and squared-grid counts at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, 67_108_864);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => new GPULabelPoint(props)).toThrow(
    /label feature count must be an integer in \[1, 67108863\]/
  );
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  const small = createProps(new GPUCommandGraph(device));
  expect(() => new GPULabelPoint({...small, initialGridSize: 65536})).toThrow(
    /initialGridSize.*65535/
  );
  expect(() => new GPULabelPoint({...small, refinementGridSize: 65536})).toThrow(
    /refinementGridSize.*65535/
  );
  device.destroy();
});

it('validates actual storage bindings before node creation and accepts a packed input', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  expect(() => graph.add(new GPULabelPoint(props).getCommandNodes(graph))).not.toThrow();
  const backing = graph.createTransientBuffer({
    id: 'non-storage',
    byteLength: 32,
    usage: Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const rejected = new GPULabelPoint({
    ...props,
    positions: graph.createDataView(backing, {format: 'float32x2', length: 4})
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => rejected.getCommandNodes(graph)).toThrow(/buffer usage must include Buffer.STORAGE/);
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});
