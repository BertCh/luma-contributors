// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {computeSpatialCrossValidationOnCPU} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeSpatialAuditKernel} from './spatial-design-audit.test-utils';

const fixtures = [
  {
    name: 'B1 residual centroid',
    features: [0, 1, 1 + 2 ** -23, 1 + 2 ** -22, 1 + 2 ** -22, 1 + 2 ** -22, 1 + 2 ** -21],
    folds: [0, 2, 2, 1, 1, 1, 1],
    objective: 7 * 2 ** -47
  },
  {
    name: 'B2 tiny seed separation',
    features: [0, 2 ** -78, 1, 0, 2 ** -78, 1],
    folds: [0, 2, 1, 0, 2, 1],
    objective: 0
  }
];

it.each(fixtures)('$name retains all rows on both forms', fixture => {
  for (const standardize of [false, true]) {
    const rows = fixture.features.length;
    const positions = Float32Array.from({length: rows * 2}, (_, index) =>
      index % 2 === 0 ? index / 2 : 0
    );
    const cpu = computeSpatialCrossValidationOnCPU(
      {positions, blocks: new Uint32Array(rows), features: Float32Array.from(fixture.features)},
      {foldCount: 3, seed: 0, design: {mode: 'environmental', featureCount: 1, standardize}}
    );
    // B1: cluster means are 0, 1+2.5u, 1+0.5u; SSE=(3+.5)u².
    // B2: each of three distinct constant groups has zero SSE, however small h is.
    expect(Array.from(cpu.folds)).toEqual(fixture.folds);
    expect(Array.from(cpu.status)).toEqual([0, 0, 0]);
    for (const reciprocalDivision of [false, true]) {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        format: Format,
        length: number
      ) => createTransientView(graph, `review1-${serial++}`, format, length);
      try {
        const contributor = new GPUSpatialCrossValidation({
          positions: view('float32x2', rows),
          blocks: view('uint32', rows),
          foldCount: 3,
          seed: 0,
          design: {
            mode: 'environmental',
            features: view('float32', rows),
            featureCount: 1,
            standardize
          },
          output: {
            folds: view('uint32', rows),
            training: view('uint32', rows * 3),
            testing: view('uint32', rows * 3),
            counts: view('uint32', 9),
            status: view('uint32', 3)
          }
        });
        const values = {
          positions,
          features: Float32Array.from(fixture.features),
          centeredFeatures: new Float32Array(rows),
          folds: new Uint32Array(rows),
          state: new Float32Array(4),
          roundingZero: new Uint32Array(1)
        };
        for (const kernel of getGPUCommandNodeKernelSources(
          contributor
            .getCommandNodes(graph)
            .filter(node => node.id.endsWith('-center-features') || node.id.endsWith('-design'))
        ))
          executeSpatialAuditKernel(kernel, values, {
            expandArrayIncrement: true,
            reciprocalDivision
          });
        expect(values.state[0]).toBe(0);
        expect(
          Array.from(values.folds),
          `standardize=${standardize}, reciprocal=${reciprocalDivision}`
        ).toEqual(fixture.folds);
        expect(Number.isFinite(values.state[2])).toBe(true);
        if (!standardize)
          expect(Math.abs(values.state[2] - fixture.objective)).toBeLessThanOrEqual(
            fixture.objective * 1e-6
          );
        if (fixture.objective === 0) expect(values.state[2]).toBe(0);
      } finally {
        device.destroy();
      }
    }
  }
});
