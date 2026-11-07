// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {RATE_INVALID_CAUSES} from '../../../examples/deck/kepler-compute/src/analysis-numeric-diagnostics';
import {KEPLER_RATE_CAUSES} from '../src/analysis/rate-cause-counts';

test('GPU rate cause slots match the CPU classifier priority order', () => {
  expect([...KEPLER_RATE_CAUSES]).toEqual([...RATE_INVALID_CAUSES]);
});
