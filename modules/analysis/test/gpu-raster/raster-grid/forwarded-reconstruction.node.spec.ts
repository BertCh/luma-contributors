// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {GPURasterReconstruction} from '../../../src/raster-analysis/gpu-raster-reconstruction';

for (const kind of ['buffer', 'texture'] as const) {
  it(`captures inherited direct reconstruction ${kind} storage fields`, () => {
    const context = createKernelAuditContext('forwarded-reconstruction');
    try {
      const storage =
        kind === 'buffer'
          ? {kind, values: context.view('float32', 4)}
          : {
              kind,
              view: context.graph.createTextureView(context.texture('r32float', 2, 2)),
              channel: 0 as const
            };
      const contributor = new GPURasterReconstruction({
        grid: {width: 2, height: 2, affine: [1, 0, 0, 0, -1, 2], pixelInterpretation: 'area'},
        marker: {id: 'marker', format: 'float32', storage: Object.create(storage)},
        mask: context.view('float32', 4),
        maximumIterations: 1,
        output: {
          values: context.view('float32', 4),
          validity: context.view('uint32', 4),
          status: {converged: context.view('uint32', 1)}
        }
      });
      expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
    } finally {
      context.device.destroy();
    }
  });
}

it('rejects an unknown direct reconstruction method at construction', () => {
  const context = createKernelAuditContext('forwarded-reconstruction');
  try {
    expect(
      () =>
        new GPURasterReconstruction({
          grid: {width: 2, height: 2, affine: [1, 0, 0, 0, -1, 2], pixelInterpretation: 'area'},
          marker: context.view('float32', 4),
          mask: context.view('float32', 4),
          method: 'errosion' as never,
          output: {
            values: context.view('float32', 4),
            validity: context.view('uint32', 4),
            status: {converged: context.view('uint32', 1)}
          }
        })
    ).toThrow(/method.*dilation.*erosion.*errosion/);
  } finally {
    context.device.destroy();
  }
});
