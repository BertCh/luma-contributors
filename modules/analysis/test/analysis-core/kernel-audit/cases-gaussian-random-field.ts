// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGaussianRandomField} from '../../../src/gpu-spatial-analysis/spatial-interpolation/index';
import type {KernelAuditCase} from './kernel-audit-context';

export const GAUSSIAN_RANDOM_FIELD_KERNEL_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUGaussianRandomField',
    name: 'maximum-fft-grid',
    build({graph, view}) {
      graph.add(
        new GPUGaussianRandomField({
          grid: {
            width: 1024,
            height: 1024,
            affine: [1, 0, 0, 0, -1, 0],
            pixelInterpretation: 'point'
          },
          parameters: view('float32', 16),
          output: {
            values: view('float32', 1024 * 1024),
            validity: view('uint32', 1024 * 1024),
            status: {approximation: view('uint32', 1)},
            clippedEigenvalueMass: view('float32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUGaussianRandomField',
    name: 'maximum-spectral-modes-and-local-conditioning',
    build({graph, view}) {
      const rowCount = 2 ** 18;
      graph.add(
        new GPUGaussianRandomField({
          positions: view('float32x2', rowCount),
          modeCount: 4096,
          parameters: view('float32', 16),
          conditioning: {
            indices: view('uint32', 65536),
            values: view('float32', 65536),
            mask: view('uint32', 65536),
            neighborhoods: view('uint32', rowCount * 32),
            neighborCount: 32
          },
          output: {
            values: view('float32', rowCount + 1),
            validity: view('uint32', rowCount + 1),
            status: {approximation: view('uint32', 1)},
            clippedEigenvalueMass: view('float32', 1)
          }
        })
      );
    }
  }
];
