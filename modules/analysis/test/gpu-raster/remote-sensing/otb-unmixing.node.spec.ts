// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPURasterSpectralUnmixing} from '../../../src/raster-analysis/remote-sensing/index';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it.each([
  {members: 5, unconstrainedBands: 256, constrainedBands: 256},
  {members: 6, unconstrainedBands: 256, constrainedBands: 245},
  {members: 7, unconstrainedBands: 247, constrainedBands: 217},
  {members: 8, unconstrainedBands: 219, constrainedBands: 195}
])('audits and bounds maximum unmixing workspaces for $members endmembers', entry => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `unmixing-boundary-${serial++}`, format, length);
  try {
    for (const constraint of ['unconstrained', 'nonnegative', 'simplex'] as const) {
      const props = {
        cellCount: 1,
        bandCount:
          constraint === 'unconstrained' ? entry.unconstrainedBands : entry.constrainedBands,
        endmemberCount: entry.members,
        constraint,
        values: view('float32', 257),
        mask: view('uint32', 257),
        endmembers: view('float32', 257 * entry.members),
        maximumIterations: 4096,
        output: {
          abundances: view('float32', entry.members),
          residual: view('float32', 1),
          status: view('uint32', 1),
          iterationCounts: view('uint32', 1)
        }
      };
      const contributor = new GPURasterSpectralUnmixing(props);
      for (const source of getParsedKernelSources(contributor.getCommandNodes(graph)).values()) {
        expect(auditWGSLModule(source)).toEqual([]);
      }
      expect(
        () => new GPURasterSpectralUnmixing({...props, bandCount: props.bandCount + 1})
      ).toThrow();
    }
  } finally {
    device.destroy();
  }
});
