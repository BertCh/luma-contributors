// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {RecipeTestFixture} from './recipe-harness';
import {createNullWebGPUDevice, submitGraph} from '../../utils/gpu-contributor-test-utils';

// Inspect the real graph lifecycle without encoding or submitting GPU commands.
vi.mock('../../utils/gpu-contributor-test-utils', async importOriginal => ({
  ...(await importOriginal<typeof import('../../utils/gpu-contributor-test-utils')>()),
  submitGraph: vi.fn()
}));

it('compiles once and resubmits the same graph for loss and recovery runs', () => {
  const device = createNullWebGPUDevice();
  const fixture = new RecipeTestFixture(device, 'recipe-reuse');
  const compile = vi.spyOn(fixture.graph, 'compile');
  try {
    fixture.run();
    fixture.run();
    fixture.run();
    expect(compile).toHaveBeenCalledTimes(1);
    expect(submitGraph).toHaveBeenCalledTimes(3);
    for (const call of vi.mocked(submitGraph).mock.calls)
      expect(call[1]).toBe(compile.mock.results[0].value);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});
