// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {AnalysisOLSOutputs} from '../../../examples/deck/kepler-compute/src/analysis-ols-outputs';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
it('UCO-17 real chunked OLS allocation and imports use the documented active layout', () => {
  const device = createNullWebGPUDevice();
  const release = vi.fn();
  const reserveMemory = vi.fn(() => ({release}));
  const session = {device, registry: {reserveMemory}} as unknown as ConstructorParameters<
    typeof AnalysisOLSOutputs
  >[0];
  const output = new AnalysisOLSOutputs(session, [2, 3]);
  try {
    // 5 fitted floats + 3 * 16 coefficient floats + 22 summary floats + one uint32 = 304 bytes.
    expect(reserveMemory).toHaveBeenCalledWith(304, 'gpu', 'output');
    expect(output.fitted.data.map(chunk => [chunk.byteOffset, chunk.length, chunk.format])).toEqual(
      [
        [0, 2, 'float32'],
        [8, 3, 'float32']
      ]
    );
    const graph = new GPUCommandGraph(device, {id: 'ols-layout-probe'});
    const residuals = createTransientView(graph, 'residuals', 'float32', 5);
    const views = output.importToGraph(graph, residuals, 2);
    for (const [view, length, format] of [
      [views.fitted, 5, 'float32'],
      [views.coefficients, 3, 'float32'],
      [views.standardErrors, 3, 'float32'],
      [views.tStatistics, 3, 'float32'],
      [views.summary, 22, 'float32'],
      [views.status?.code, 1, 'uint32']
    ] as const) {
      expect(view?.length).toBe(length);
      expect(view?.format).toBe(format);
    }
  } finally {
    output.destroy();
  }
  expect(release).toHaveBeenCalledOnce();
});

import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {runCatalogLineGeometry} from '../src/analysis/catalog-platform4-geometry-common';
it('UCO-17 line output allocation retains terminal offsets and typed count status', async () => {
  const device = createNullWebGPUDevice();
  const session = new CatalogGpuSession(device, 'line-layout');
  const captured = new Error('checked output views before dispatch');
  try {
    await expect(
      runCatalogLineGeometry(
        session,
        {
          positions: Float32Array.of(0, 0, 1, 0, 2, 0),
          lineOffsets: Uint32Array.of(0, 3)
        },
        {vertices: 6, lines: 2},
        (_inputs, output) => {
          expect([output.positions.format, output.positions.length]).toEqual(['float32x2', 6]);
          expect([output.lineOffsets.format, output.lineOffsets.length]).toEqual(['uint32', 3]);
          expect([output.sourceLineIndices?.format, output.sourceLineIndices?.length]).toEqual([
            'uint32',
            2
          ]);
          expect([output.status.count.format, output.status.count.length]).toEqual(['uint32', 1]);
          expect([output.status.overflow.format, output.status.overflow.length]).toEqual([
            'uint32',
            1
          ]);
          throw captured;
        }
      )
    ).rejects.toBe(captured);
  } finally {
    await session.dispose();
    device.destroy();
  }
});
