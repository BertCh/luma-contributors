// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createMakeValidParityProbe, createMakeValidParityScene} from './make-valid-parity-harness';

useCPUExecutorTestHooks();

it('parity uses exact group ranges on reversed-bit IDs with a uint32 outlier', () => {
  let previousVisits = 0;
  for (const groupBits of [11, 12]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const probe = createMakeValidParityProbe(graph, groupBits);
    const memory = createCPUGraphMemory();
    for (const input of probe.inputs) memory.write(input.view, input.values);
    executeCommandNodesOnCPU(probe.nodes, memory);
    const diagnostics = memory.readUint32(probe.diagnostics);
    const keep = memory.readUint32(probe.keep);
    let visits = 0;
    const perFaceLimit = 2 * Math.ceil(Math.log2(probe.scene.segmentCapacity)) + 9;
    for (let feature = 0; feature < probe.scene.featureCount; feature++) {
      const base = 3 * feature;
      visits += diagnostics[base];
      expect(diagnostics[base], `feature ${feature} visited nodes`).toBeLessThanOrEqual(
        perFaceLimit
      );
      // Bottom, left and right edges touch y=0. The top edge at y=1 is pruned.
      expect(diagnostics[base + 1], `feature ${feature} candidates`).toBe(3);
      expect(diagnostics[base + 2], `feature ${feature} measured parity`).toBe(1);
      expect(keep[feature], `feature ${feature} production parity`).toBe(1);
    }
    expect(Array.from(diagnostics.slice(3 * probe.scene.featureCount))).toEqual([0, 0, 0]);
    if (previousVisits > 0) expect(visits).toBeLessThan(2.5 * previousVisits);
    previousVisits = visits;
  }
}, 120_000);

it('the scale fixture contains the exact 4097-square adversarial ID order', () => {
  const scene = createMakeValidParityScene(12);
  expect(scene.featureCount).toBe(4097);
  expect(Array.from(scene.sourceIds.slice(0, 8))).toEqual([
    0, 2048, 1024, 3072, 512, 2560, 1536, 3584
  ]);
  expect(scene.sourceIds[4096]).toBe(0xfffffffe);
  expect(Array.from(scene.probes.slice(4 * 4096, 4 * 4097))).toEqual([12288, 0, 12289, 0]);
});
