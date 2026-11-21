// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUGraphVectorFormat,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {CPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {GPUProfilePeaksOperation} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const INPUTS = {
  id: 'review-profile',
  values: 'values',
  offsets: 'offsets',
  parameters: 'parameters',
  sampleCount: 3,
  window: 1,
  minimumSide: 1
};

for (const Operation of [CPUProfilePeaksOperation, GPUProfilePeaksOperation]) {
  it(`S3 ${Operation.name} names missing compact status in JavaScript requests`, () => {
    for (const status of [undefined, null, 0, 'invalid']) {
      const peaks = status === undefined ? {} : {status};
      expect(() => Reflect.construct(Operation, [{...INPUTS, output: {peaks}}])).toThrow(
        /review-profile.*output\.peaks\.status/
      );
    }
  });

  it(`N1 ${Operation.name} rejects a zero declared count with only a status terminal`, () => {
    expect(
      () => new Operation({...INPUTS, sampleCount: 0, output: {status: {converged: 'done'}}})
    ).toThrow(/review-profile.*sampleCount/);
  });
}

for (const terminal of ['converged', 'iterationLimitReached'] as const) {
  for (const length of [2, 5]) {
    it(`N1 GPU status-only ${terminal} checks declared count 2 against ${length} resolved rows`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const output =
        terminal === 'converged'
          ? {status: {converged: 'done'}}
          : {status: {iterationLimitReached: 'done'}};
      const operation = new GPUProfilePeaksOperation({...INPUTS, sampleCount: 2, output});
      const views = new Map<string, GraphDataView>([
        ['values', createTransientView(graph, 'values', 'float32', length)],
        ['offsets', createTransientView(graph, 'offsets', 'uint32', 2)],
        ['parameters', createTransientView(graph, 'parameters', 'float32', 1)],
        [terminal, createTransientView(graph, 'done', 'uint32', 1)]
      ]);
      try {
        const lower = () =>
          operation.createContributor(<Format extends GPUGraphVectorFormat>(port: string) => {
            const view = views.get(port);
            if (!view) throw new Error(`Missing fixture port ${port}`);
            return view as GraphDataView<Format>;
          });
        if (length === 2) expect(lower).not.toThrow();
        else expect(lower).toThrow(/review-profile.*values.*sampleCount.*2.*5/);
      } finally {
        operation.destroy();
        device.destroy();
      }
    });
  }
}
