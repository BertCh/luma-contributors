// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContinuousSpatialSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample';
import type {KernelAuditCase} from './kernel-audit-context';

export const CONTINUOUS_SAMPLING_CASES: KernelAuditCase[] = (['line', 'area'] as const).map(
  type => ({
    exportName: 'GPUContinuousSpatialSample',
    name: `${type} frame, legacy sites, resident parameters, replacement waves and bounded capacity`,
    build({graph, view}) {
      const positions = view('float32x2', type === 'line' ? 4096 : 4095);
      // Largest frames within the 2^24 serial work bound per wave (PTS-111):
      // line (1024 + 2) * (4095 + 2 * 4095 + 1) + 4096 + 4096 = 12613628;
      // area, an 81 x 81 lattice, (6561 + 256) * (1365 + 256 + 1) + 4095 + 4096 = 11065365.
      graph.add(
        new GPUContinuousSpatialSample({
          frame:
            type === 'line'
              ? {type, positions, legacySites: view('float32', 2)}
              : {type, positions, bounds: [0, 0, 100, 100], legacySites: view('float32x2', 256)},
          spacing: 1.25,
          seed: 7,
          parameters: view('uint32', 2),
          waves: 8,
          capacity: 4096,
          maximumCandidates: type === 'line' ? 1024 : 1048576,
          output: {
            positions: view('float32x2', 32768),
            designWeights: view('float32', 32768),
            provenance: view('uint32x4', 32768),
            status: view('uint32x4', 8)
          }
        })
      );
    }
  })
);
