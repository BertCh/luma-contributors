// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {RecipeTestFixture} from './recipe-harness';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('compiles once and resubmits the same graph for loss and recovery runs', () => {
  const device = createNullWebGPUDevice();
  const fixture = new RecipeTestFixture(device, 'recipe-reuse');
  const compile = vi.spyOn(fixture.graph, 'compile');
  const encode = vi.spyOn(CompiledGPUCommandGraph.prototype, 'encode');
  const submit = vi.spyOn(device, 'submit');
  try {
    for (const run of [1, 2, 3]) {
      fixture.run();
      expect(encode).toHaveBeenCalledTimes(run);
      expect(encode.mock.contexts[run - 1]).toBe(compile.mock.results[0].value);
      expect(submit).toHaveBeenCalledTimes(run);
    }
    expect(compile).toHaveBeenCalledTimes(1);
  } finally {
    submit.mockRestore();
    encode.mockRestore();
    compile.mockRestore();
    fixture.destroy();
    device.destroy();
  }
});
