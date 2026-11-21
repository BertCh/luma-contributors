// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUPointPatternEnvelope} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope';
import {GPUContinuousSpatialSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample';
import {computeContinuousSpatialSampleOnCPU} from '../../../src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation';
import {validatePointProcessView} from '../../../src/gpu-spatial-analysis/point-process/point-process-utils';

it('B5 ignores unused legacy views in ownership, alias and binding-range checks', () => {
  for (const statistic of ['maximum-deviation', 'extreme-rank'] as const) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice()),
      foreign = new GPUCommandGraph(createNullWebGPUDevice());
    const lower = createTransientView(graph, 'lower', 'float32', 2);
    const sample = new GPUPointPatternEnvelope({
      statistic,
      observed: createTransientView(graph, 'observed', 'float32', 2),
      simulations: createTransientView(graph, 'simulations', 'float32', 4),
      reference:
        statistic === 'extreme-rank'
          ? lower
          : createTransientView(graph, 'reference', 'float32', 2),
      integrationWeights: createTransientView(foreign, 'unused', 'float32', 1),
      nullModel: 'specified',
      fittedParameters: 'none',
      output: {
        lower,
        upper: createTransientView(graph, 'upper', 'float32', 2),
        statistics: createTransientView(graph, 'statistics', 'float32', 4),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    });
    expect(() => sample.getCommandNodes(graph)).not.toThrow();
  }
});
it('S2 carries actual and expected row counts in structured validation errors', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  try {
    validatePointProcessView(
      'my-park',
      'output.designWeights',
      createTransientView(graph, 'weights', 'float32', 2),
      ['float32'],
      3
    );
    expect.fail('short view accepted');
  } catch (error) {
    expect(error).toMatchObject({
      code: 'VIEW_LENGTH',
      id: 'my-park',
      path: 'output.designWeights',
      received: 2,
      expected: {rows: 3}
    });
  }
});
it('S2 names a short continuous output through the contributor constructor', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  try {
    new GPUContinuousSpatialSample({
      id: 'my-park',
      frame: {type: 'line', positions: createTransientView(graph, 'positions', 'float32x2', 2)},
      spacing: 1,
      capacity: 3,
      maximumCandidates: 3,
      seed: 17,
      output: {
        positions: createTransientView(graph, 'output', 'float32x2', 3),
        designWeights: createTransientView(graph, 'weights', 'float32', 2),
        provenance: createTransientView(graph, 'provenance', 'uint32x4', 3),
        status: createTransientView(graph, 'status', 'uint32x4', 1)
      }
    });
    expect.fail('short view accepted');
  } catch (error) {
    expect(error).toMatchObject({
      code: 'VIEW_LENGTH',
      id: 'my-park',
      path: 'output.designWeights',
      received: 2,
      expected: {rows: 3}
    });
    expect(String(error)).toMatch(/my-park.*output.designWeights.*3.*2/);
  }
});
it('S3 rejects an unknown continuous CPU frame selector', () => {
  expect(() =>
    computeContinuousSpatialSampleOnCPU(
      {positions: Float32Array.of(0, 0, 2, 0)},
      {frame: {type: 'linestring' as 'line'}, spacing: 1, seed: 17, capacity: 2}
    )
  ).toThrow(/frame.type.*line.*area.*linestring/);
});
it('S6 supplied-intensity example makes its intensity input mandatory', () => {
  const page = readFileSync('docs/api/gpu-spatial-analysis/point-process.md', 'utf8');
  const snippet = page.split('```typescript')[1].split('```')[0];
  expect(snippet).toContain(
    "Required<Pick<GPUInhomogeneousPointPatternProps, 'suppliedIntensity'>>"
  );
});
