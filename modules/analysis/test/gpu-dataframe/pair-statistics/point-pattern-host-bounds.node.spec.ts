// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPointPatternIndices} from '../../../src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('CORE-118 named host limits preserve gamma and six-field guard kernel bytes', () => {
  const device = createNullWebGPUDevice();
  const hashes: Record<string, string> = {};
  try {
    for (const quadratGrid of [
      [3, 2],
      [16, 16]
    ] as const) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPUPointPatternIndices({
        id: 'bounds',
        positions: createTransientView(graph, 'positions', 'float32x2', 3),
        parameters: createTransientView(graph, 'parameters', 'float32', 8),
        gridSize: [2, 2],
        quadratGrid,
        output: {
          clarkEvans: createTransientView(graph, 'clark-evans', 'float32', 7),
          quadratStatistics: createTransientView(graph, 'statistics', 'float32', 7)
        }
      });
      for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
        if (!['bounds-quadrat-statistics', 'bounds-guard-clark-evans'].includes(kernel.id))
          continue;
        // CORE-118 added this bound to the WGSL comment. Keep the pre-change hash for all
        // executable bytes, including the centered gamma path and the six-field guard.
        const source = kernel.source.replace(
          '  // LOG_GAMMA_MAXIMUM_SHIFT_COUNT = 9, including eager shape -0.5.\n',
          ''
        );
        hashes[`${quadratGrid[0] * quadratGrid[1]}:${kernel.id}`] = createHash('sha256')
          .update(source)
          .digest('hex');
      }
    }
    // Captured before replacing host literals; these protect emitted WGSL, including
    // the centered gamma calculation, independently of how the host expresses its bounds.
    expect(hashes).toEqual({
      '256:bounds-guard-clark-evans':
        '378d4b46527b4f19ca95d53c4dec472478e94d795e9762b2137dbda9e8859bd4',
      '256:bounds-quadrat-statistics':
        'd28f6e3766ee4eb695abf0e3fe426e7e581b30a21c6bbe4948787de65ad7df3c',
      '6:bounds-guard-clark-evans':
        '378d4b46527b4f19ca95d53c4dec472478e94d795e9762b2137dbda9e8859bd4',
      '6:bounds-quadrat-statistics':
        '8a2341c037d55586b754cf5632ef139713fbfff31f5fbfbe261905de0ac4cc95'
    });
  } finally {
    device.destroy();
  }
});
