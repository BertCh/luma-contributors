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
import {createRingShellIndexProbe} from './ring-shell-index-harness';

async function runShellProbe(device: Device, annulusBits: number, transpose = false) {
  const graph = new GPUCommandGraph(device, {id: `shell-index-scale-${annulusBits}-${transpose}`});
  const probe = createRingShellIndexProbe(graph, annulusBits, transpose);
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

it('shell traversal stays near-linear on 4096 bit-reversed annuli and a distant outlier', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    let previousVisits = 0;
    for (const annulusBits of [10, 11, 12]) {
      const {scene, diagnostics} = await runShellProbe(device, annulusBits);
      const perHoleLimit = 2 * Math.ceil(Math.log2(scene.ringCapacity)) + 1;
      let visits = 0;
      for (let annulus = 0; annulus < scene.annulusCount; annulus++) {
        const shell = 2 * annulus;
        const base = 3 * (shell + 1);
        expect(diagnostics[3 * shell + 2], `shell ${shell} owns itself`).toBe(shell);
        expect(diagnostics[base], `hole ${shell + 1} visits`).toBeLessThanOrEqual(perHoleLimit);
        expect(diagnostics[base + 1], `hole ${shell + 1} candidates`).toBe(1);
        expect(diagnostics[base + 2], `hole ${shell + 1} shell`).toBe(shell);
        visits += diagnostics[base];
      }
      expect(diagnostics.slice(3 * scene.ringCount)).toEqual([0, 0, 0xffffffff, 0, 0, 0xffffffff]);
      // The unrefined key needed 1,403,220 / 5,602,644 / 22,390,100 visits here.
      // Count production traversal iterations, independently of shader compilation/GPU time.
      expect(visits).toBeLessThanOrEqual(scene.annulusCount * perHoleLimit);
      if (previousVisits > 0) expect(visits).toBeLessThan(2.3 * previousVisits);
      previousVisits = visits;
    }
    const {scene, diagnostics} = await runShellProbe(device, 6, true);
    for (let annulus = 0; annulus < scene.annulusCount; annulus++) {
      const base = 3 * (2 * annulus + 1);
      expect(diagnostics[base + 1]).toBe(1);
      expect(diagnostics[base + 2]).toBe(2 * annulus);
    }
  } finally {
    device.destroy?.();
  }
}, 120_000);
