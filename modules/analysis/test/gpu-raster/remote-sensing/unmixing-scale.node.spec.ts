// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPURasterSpectralUnmixing} from '../../../src/raster-analysis/remote-sensing/gpu-raster-spectral-unmixing';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

for (const constraint of ['nonnegative', 'simplex'] as const) {
  it(`RT-140 ${constraint} normalizes before products and restores squared residual units`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floats = (length: number) =>
      createTransientView(graph, `float-${serial++}`, 'float32', length);
    const integers = (length: number) =>
      createTransientView(graph, `integer-${serial++}`, 'uint32', length);
    try {
      const contributor = new GPURasterSpectralUnmixing({
        cellCount: 1,
        bandCount: 3,
        endmemberCount: 3,
        constraint,
        values: floats(3),
        endmembers: floats(9),
        output: {
          abundances: floats(3),
          residual: floats(1),
          status: integers(1),
          iterationCounts: integers(1)
        }
      });
      const source = [...getParsedKernelSources(contributor.getCommandNodes(graph)).values()][0];
      expect(auditWGSLModule(source)).toEqual([]);
      const normalization = source.indexOf('design[entry] /= designScale');
      const products = source.indexOf('lipschitz += value * value');
      expect(normalization).toBeGreaterThanOrEqual(0);
      expect(products).toBeGreaterThanOrEqual(0);
      expect(normalization).toBeLessThan(products);
      expect(source).toContain('response[band] /= designScale');
      expect(source).toContain('let originalError = error * designScale');
      expect(source).not.toContain('max(1.0, gradientScale)');
      expect(source).toContain('code == 0u || code == 4u');
    } finally {
      device.destroy();
    }
  });
}

for (const division of ['direct', 'reciprocal'] as const) {
  it(`B2 dark-pixel simplex solves with ${division} division`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floats = (length: number) =>
      createTransientView(graph, `float-${serial++}`, 'float32', length);
    const integers = () => createTransientView(graph, `integer-${serial++}`, 'uint32', 1);
    const contributor = new GPURasterSpectralUnmixing({
      cellCount: 1,
      bandCount: 2,
      endmemberCount: 2,
      constraint: 'simplex',
      tolerance: 1e-7,
      values: floats(2),
      endmembers: floats(4),
      output: {
        abundances: floats(2),
        residual: floats(1),
        status: integers(),
        iterationCounts: integers()
      }
    });
    const buffers: Record<string, Float32Array | Uint32Array> = {
      values: new Float32Array([0, 0]),
      endmembers: new Float32Array([1, 0.125, 0.625, 1]),
      abundances: new Float32Array(2),
      residual: new Float32Array(1),
      status: new Uint32Array([99]),
      iterationCounts: new Uint32Array([99])
    };
    const errors = vi.spyOn(console, 'error').mockImplementation(message => {
      if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
    });
    try {
      const kernel = getParsedKernelSources(contributor.getCommandNodes(graph)).configurations[0];
      const original = kernel.source;
      // The CPU interpreter requires explicit private-array copy types. Expand copies only;
      // all floating-point expressions remain the generated kernel's expressions.
      const compatible = original
        .replace(
          'var sorted = candidate;',
          'var sorted: array<f32, 2>; for (var element = 0u; element < 2u; element++) { sorted[element] = candidate[element]; }'
        )
        .replace(
          'coefficients = candidate;',
          'for (var element = 0u; element < 2u; element++) { coefficients[element] = candidate[element]; }'
        );
      const source =
        division === 'direct'
          ? compatible
          : compatible.replace('gradient / lipschitz', 'gradient * (1.0 / lipschitz)');
      const bindings = Object.fromEntries(
        new WgslReflect(source).storage.map(declaration => [
          declaration.binding,
          buffers[declaration.name]
        ])
      );
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {0: bindings},
        {constants: kernel.constants}
      );
      // Let a be the first abundance; sum-one gives the second as 1-a. The squared
      // norm is (.625+.375*a)^2 + (1-.875*a)^2, minimized at a=41/58.
      expect(buffers['status'][0]).toBe(0);
      expect(buffers['iterationCounts'][0]).toBeLessThan(256);
      expect(Array.from(buffers['abundances'])).toHaveLength(2);
      for (const [index, expected] of [41 / 58, 17 / 58].entries()) {
        expect(Number.isFinite(buffers['abundances'][index])).toBe(true);
        expect(buffers['abundances'][index]).toBeCloseTo(expected, 6);
      }
    } finally {
      errors.mockRestore();
      device.destroy();
    }
  });
}
