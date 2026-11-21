// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPUIslandAttachment,
  GPU_ISLAND_ATTACHMENT_PLANNER
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';

it('SCL-12 island planning refuses the same unsupported distances as construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `forward-plan-${serial++}`, format, length);
  const props = {
    weights: {
      offsets: view('uint32', 7),
      neighbors: view('uint32', 6),
      weights: view('float32', 6)
    },
    positions: view('float32x2', 6),
    output: {
      offsets: view('uint32', 7),
      neighbors: view('uint32', 18),
      weights: view('float32', 18),
      attachment: view('uint32', 6),
      added: view('uint32', 18),
      status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
    }
  };
  try {
    expect(GPU_ISLAND_ATTACHMENT_PLANNER.getExecutionPlan(props).feasible).toBe(true);
    expect(() => new GPUIslandAttachment(props)).not.toThrow();
    const unsupported = {...props, output: {...props.output, distances: view('float32', 18)}};
    expect(() => new GPUIslandAttachment(unsupported)).toThrow(/distances/);
    expect(GPU_ISLAND_ATTACHMENT_PLANNER.getExecutionPlan(unsupported).feasible).toBe(false);
  } finally {
    device.destroy();
  }
});

it('trace exact preflight validates explicit probeCount even though exact probes use rows', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `forward-trace-${serial++}`, format, length);
  const props = {
    weights: {
      offsets: view('uint32', 7),
      neighbors: view('uint32', 6),
      weights: view('float32', 6)
    },
    maximumPower: 1,
    output: {powerTraces: view('float32', 1), powerSums: view('float32', 1)}
  };
  try {
    for (const mode of ['exact', 'stochastic'] as const) {
      for (const probeCount of [undefined, 1]) {
        expect(
          GPUSpatialWeightsTraces.getExecutionPlan({...props, mode, probeCount}).feasible
        ).toBe(true);
        expect(() => new GPUSpatialWeightsTraces({...props, mode, probeCount})).not.toThrow();
      }
    }
    expect(() => new GPUSpatialWeightsTraces({...props, probeCount: 0})).toThrow(/probeCount/);
    expect(() => GPUSpatialWeightsTraces.getExecutionPlan({...props, probeCount: 0})).toThrow(
      /probeCount/
    );
  } finally {
    device.destroy();
  }
});
