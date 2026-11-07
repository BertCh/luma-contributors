// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram';
import type {KernelAuditCase} from './kernel-audit-context';

export const DORLING_CARTOGRAM_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPUDorlingCartogram',
    name: 'maximum-capacity',
    build: ({graph, view}) => {
      graph.add(
        new GPUDorlingCartogram({
          positions: view('float32x2', 1024),
          values: view('float32', 1024),
          sourceIds: view('uint32', 1024),
          maximumIterations: 2,
          output: {
            positions: view('float32x2', 1024),
            radii: view('float32', 1024),
            sourceIds: view('uint32', 1024),
            metrics: view('float32x2', 1),
            status: view('uint32', 4)
          }
        })
      );
    }
  }
];
