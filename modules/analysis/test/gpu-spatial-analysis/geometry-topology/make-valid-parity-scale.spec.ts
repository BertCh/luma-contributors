// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createMakeValidParityProbe} from './make-valid-parity-harness';

async function runParityProbe(device: Device, groupBits: number) {
  const graph = new GPUCommandGraph(device, {id: `parity-scale-${groupBits}`});
  const probe = createMakeValidParityProbe(graph, groupBits);
  const buffers: Buffer[] = [];
  for (const [index, input] of probe.inputs.entries()) {
    const buffer = createInputBuffer(device, input.values);
    buffers.push(buffer);
    const source = importGraphBuffer(
      graph,
      `input-${index}`,
      buffer,
      input.view.format,
      input.view.length
    );
    const type = input.values instanceof Float32Array ? 'f32' : 'u32';
    graph.add(
      createWGSLKernelNode(graph, {
        id: `upload-${index}`,
        operation: 'test',
        invocationCount: input.values.length,
        bindings: [
          {name: 'source', view: source, type, access: 'read'},
          {name: 'destination', view: input.view, type, access: 'read_write'}
        ],
        body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
      })
    );
  }
  for (const node of probe.nodes) graph.add(node);
  const buffer = createOutputBuffer(device, probe.diagnostics.length);
  buffers.push(buffer);
  const destination = importGraphBuffer(
    graph,
    'diagnostics',
    buffer,
    'uint32',
    probe.diagnostics.length
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'read-diagnostics',
      operation: 'test',
      invocationCount: probe.diagnostics.length,
      bindings: [
        {name: 'source', view: probe.diagnostics, type: 'u32', access: 'read'},
        {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
      ],
      body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {scene: probe.scene, diagnostics: await readUint32(buffer, probe.diagnostics.length)};
  } finally {
    compiled.destroy();
    for (const allocated of buffers) allocated.destroy();
  }
}

it('make-valid parity stays near-linear for 4097 bit-reversed square IDs and a uint32 outlier', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    let previousVisits = 0;
    for (const groupBits of [11, 12]) {
      const {scene, diagnostics} = await runParityProbe(device, groupBits);
      const perFaceLimit = 2 * Math.ceil(Math.log2(scene.segmentCapacity)) + 9;
      let visits = 0;
      for (let feature = 0; feature < scene.featureCount; feature++) {
        const base = 3 * feature;
        visits += diagnostics[base];
        expect(diagnostics[base], `feature ${feature} node visits`).toBeLessThanOrEqual(
          perFaceLimit
        );
        expect(diagnostics[base + 1], `feature ${feature} candidate count`).toBe(3);
        expect(diagnostics[base + 2], `feature ${feature} kept square`).toBe(1);
      }
      expect(diagnostics.slice(3 * scene.featureCount)).toEqual([0, 0, 0]);
      // Count actual traversal iterations, not wall time. The original Morton group key required
      // about 64 million visits at 4096 small groups and roughly quadrupled on doubling.
      expect(visits).toBeLessThan(scene.featureCount * perFaceLimit);
      if (previousVisits > 0) expect(visits).toBeLessThan(2.3 * previousVisits);
      previousVisits = visits;
    }
  } finally {
    device.destroy?.();
  }
}, 120_000);
