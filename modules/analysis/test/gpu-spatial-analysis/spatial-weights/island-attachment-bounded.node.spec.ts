// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPUIslandAttachment,
  type GPUIslandAttachmentNearestSearch
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {
  createNullWebGPUDevice,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

function createAttachment(
  rows: number,
  nearestSearch: GPUIslandAttachmentNearestSearch = 'grid',
  uncertainCount = true
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const contributor = new GPUIslandAttachment({
    id: 'bounded-islands',
    nearestSearch,
    weights: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', rows),
      weights: view('float32', rows)
    },
    positions: view('float32x2', rows),
    output: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', 3 * rows),
      weights: view('float32', 3 * rows),
      attachment: view('uint32', rows),
      added: view('uint32', 3 * rows),
      status: {
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1),
        ...(uncertainCount ? {uncertainCount: view('uint32', 1)} : {})
      }
    }
  });
  return {device, graph, contributor};
}

describe('GPUIslandAttachment bounded cooperative fallback', () => {
  for (const rows of [17, 1025, 65535]) {
    for (const uncertainCount of [false, true]) {
      it(`${rows} rows, uncertainty port ${uncertainCount}: bounded lanes and portable storage`, () => {
        const {device, graph, contributor} = createAttachment(rows, 'grid', uncertainCount);
        try {
          contributor.getCommandNodes(graph);
          const nearest = getWGSLKernelNodeRecord(graph, 'bounded-islands-nearest');
          const fallback = getWGSLKernelNodeRecord(graph, 'bounded-islands-resolve-unproven');
          expect(nearest).toBeDefined();
          expect(fallback).toBeDefined();
          // Regression: the grid's nearest invocation used to embed the complete row scan.
          expect(nearest?.body).not.toContain(`other < ${rows}u`);
          expect(nearest?.body).toContain('unresolved[unresolvedOffset + index] = 1u');
          expect(fallback?.workgroupSize).toEqual([256, 1, 1]);
          expect(fallback?.invocationCount).toBe(rows * 256);
          expect(fallback?.declarations).toContain('MAXIMUM_REPRESENTATIVES_PER_LANE: u32 = 256u');
          expect(fallback?.body).toContain('round < MAXIMUM_REPRESENTATIVES_PER_LANE');
          expect(fallback?.body).toContain('other >= ROWS');
          expect(fallback?.body).toContain('other < nearest');
          // Workgroup memory is 256 u32 rows and 256 vec2<f32> differences (3 KiB).
          const reflected = new WgslReflect(fallback!.source);
          expect(reflected.entry.compute.map(entry => entry.name)).toEqual(['main']);
          expect(getWorkgroupMemoryBytes(fallback!.source)).toBe(3072);
          expect(auditWGSLModule(fallback!.source)).toEqual([]);
          expect(fallback!.bindings.length).toBeLessThanOrEqual(8);
          expect(nearest!.bindings.length).toBeLessThanOrEqual(8);
          expect(rows).toBeLessThanOrEqual(256 * 256);
        } finally {
          device.destroy();
        }
      });
    }
  }

  it('rejects rows outside the declared coverage at construction', () => {
    expect(() => createAttachment(65536)).toThrow(/^bounded-islands rows must be.*65535/);
  });

  it('auto uses the bounded pass above the cut-over; explicit scan retains its path', () => {
    for (const nearestSearch of ['auto', 'scan'] as const) {
      const {device, graph, contributor} = createAttachment(4097, nearestSearch);
      try {
        contributor.getCommandNodes(graph);
        expect(Boolean(getWGSLKernelNodeRecord(graph, 'bounded-islands-resolve-unproven'))).toBe(
          nearestSearch === 'auto'
        );
      } finally {
        device.destroy();
      }
    }
  });
});
