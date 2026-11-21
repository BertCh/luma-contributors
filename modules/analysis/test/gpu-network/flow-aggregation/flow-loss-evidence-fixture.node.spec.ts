// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getFlowLossEvidenceFrame} from './flow-loss-evidence-fixture';

it('B4 isolates pair-table loss from final-output truncation and recovers', () => {
  // Device fixture rows are (0,1,2), (1,0,1), (0,0,4): key = origin*2 + destination.
  const pairKeys = [1, 2, 0];
  const pairCapacity = 1;
  const finalCapacity = 2;
  for (const loss of [false, true, false]) {
    const frame = getFlowLossEvidenceFrame(loss);
    const enabledKeys = new Set(pairKeys.filter((_, index) => frame.mask[index] !== 0));
    expect(enabledKeys.size).toBeLessThanOrEqual(finalCapacity);
    expect(enabledKeys.size > pairCapacity).toBe(loss);
    expect([...enabledKeys].sort()).toEqual([...frame.retainedKeys].sort());
    if (!loss) expect(frame.mask).toEqual([1, 0, 0]);
  }
});
