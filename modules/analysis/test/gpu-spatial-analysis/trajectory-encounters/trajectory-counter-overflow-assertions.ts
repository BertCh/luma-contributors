// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Shared independent expectations for the production-counter node and WebGPU regressions. */
export function expectTrajectoryCounterOverflowReadback(output: {
  state: readonly number[];
  pairs: readonly number[];
  overflow: readonly number[];
}): void {
  expect(output.state).toEqual([0xffffffff]);
  expect(output.pairs).toEqual([99, 99, 99, 99]);
  expect(output.overflow).toEqual([1]);
}
