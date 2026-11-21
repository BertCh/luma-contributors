// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNearestFeatureJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {expectNearestDistance} from '../geometry-refinement/geometry-output-review-test-utils';
import {NEAREST_RANK_BOUNDARY_CASES} from './nearest-rank-boundary-test-utils';

it.each(NEAREST_RANK_BOUNDARY_CASES)('CPU nearest $name', fixture => {
  const result = computeNearestFeatureJoinOnCPU(
    {kind: 'points', positions: Float32Array.of(0, 0)},
    {
      kind: 'points',
      positions: Float32Array.from(fixture.positions),
      sourceIds: Uint32Array.of(900, 100)
    }
  );
  expect(Array.from(result.neighborCounts)).toEqual([1]);
  expect(Array.from(result.neighborRows)).toEqual([fixture.row]);
  expect(Array.from(result.neighborIds)).toEqual([fixture.order[0]]);
  expectNearestDistance(result.neighborDistances);
});

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {adaptMultipartNearestUniforms} from './multipart-nearest-executor';
import {
  createNearestFinalWaveCases,
  expectNearestResultValues
} from './nearest-rank-boundary-test-utils';

useCPUExecutorTestHooks();
for (const entry of createNearestFinalWaveCases())
  it(entry.name, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const input = Object.fromEntries(
      Object.entries(entry.inputs).map(([name, value]) => {
        const view = createTransientView(
          graph,
          `in-${name}`,
          value.format,
          value.values.length / (value.format === 'float32x2' ? 2 : 1)
        );
        memory.write(view, value.values);
        return [name, view];
      })
    );
    const output = Object.fromEntries(
      Object.entries(entry.outputs).map(([name, value]) => [
        name,
        createTransientView(graph, `out-${name}`, value.format, value.length)
      ])
    );
    // A skipped witness write must not satisfy an exact zero-foot expectation.
    if (output['foot']) memory.write(output['foot'], Float32Array.of(123, 456));
    try {
      executeCommandNodesOnCPU(
        adaptMultipartNearestUniforms(entry.declare(graph, input, output)),
        memory
      );
      for (const [name, expected] of Object.entries(entry.expected)) {
        const actual =
          output[name].format === 'uint32'
            ? memory.readUint32(output[name])
            : memory.readFloat32(output[name]);
        // The wgsl_reflect interpreter rounds vec2u components through float32, so multiply24's
        // low word loses bits (probe: 0x800001 * 0xc65d40 gives low 2697354496, exact 2697354560).
        // Only the nonzero-foot case carries a nonzero exact residual through it; its exact (4,4)
        // is asserted by the WebGPU sibling. Here it must still land within 2^-16 of (4,4).
        const interpreterInexact =
          name === 'foot' && entry.name.includes('asymmetric nonzero foot');
        expectNearestResultValues(
          Array.from(actual),
          expected,
          !interpreterInexact &&
            (output[name].format === 'uint32' || name === 'foot' || name === 'queryPoint'),
          interpreterInexact ? 2 ** -16 : (entry.tolerance ?? 0)
        );
        if (name === 'foot') {
          const witness = memory.readFloat32(output['queryPoint']);
          expect(Math.hypot(witness[0] - actual[0], witness[1] - actual[1])).toBeCloseTo(
            entry.expected['distances'][0],
            -1
          );
        }
      }
    } finally {
      device.destroy();
    }
  });

import {
  NEAREST_REVIEW_CASES,
  createReviewNearestFixture,
  assertReviewNearestFixture
} from './multipart-nearest-review-fixture';
for (const definition of NEAREST_REVIEW_CASES.filter(entry =>
  entry.name.startsWith('R2 reversed parallel edges')
))
  it(`KI-rr-joins-R4 paired-witness control ${definition.name}`, async () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const fixture = createReviewNearestFixture(
      graph,
      (name, format, length, values) => {
        const view = createTransientView(graph, name, format, length);
        memory.write(view, values);
        return view;
      },
      definition
    );
    try {
      executeCommandNodesOnCPU(
        adaptMultipartNearestUniforms(fixture.join.getCommandNodes(graph)),
        memory
      );
      await assertReviewNearestFixture(fixture, async view =>
        Array.from(view.format === 'uint32' ? memory.readUint32(view) : memory.readFloat32(view))
      );
    } finally {
      device.destroy();
    }
  });

it('S3 nearest numeric assertions reject surplus distance rows', () => {
  expect(() => expectNearestResultValues([1, 999], [1], false, 0)).toThrow();
});
