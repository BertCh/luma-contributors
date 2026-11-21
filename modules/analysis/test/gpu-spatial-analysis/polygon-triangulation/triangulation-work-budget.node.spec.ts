// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {getTriangulationSpendWGSL, WORK_BUDGET_CASES} from './triangulation-work-budget.test-utils';

for (const fixture of WORK_BUDGET_CASES) {
  it(`bounds production Earcut work before uint32 addition: ${fixture.name}`, () => {
    const source = `${getTriangulationSpendWGSL(fixture.limit)}
@group(0) @binding(0) var<storage, read_write> result: array<u32>;
@compute @workgroup_size(1) fn main() {
  workDone = ${fixture.start}u; failed = false;
  ${fixture.amounts.map(amount => `spend(${amount}u);`).join('\n')}
  result[0] = workDone; result[1] = select(0u, 1u, failed);
}`;
    const result = new Uint32Array(2);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
      0: {0: result}
    });
    // Hand integer budget accounting: accept exactly the remaining work; refuse excess before
    // addition, so a uint32 limit cannot wrap and rejected work never changes the counter.
    expect(Array.from(result)).toEqual(fixture.expected);
  });
}
