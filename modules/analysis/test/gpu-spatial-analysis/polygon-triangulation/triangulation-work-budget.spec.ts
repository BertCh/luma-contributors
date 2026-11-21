// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {getTriangulationSpendWGSL, WORK_BUDGET_CASES} from './triangulation-work-budget.test-utils';

for (const fixture of WORK_BUDGET_CASES) {
  it(`production Earcut budget refuses overflowing work: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const resultBuffer = createOutputBuffer(device, 2);
    const result = importGraphBuffer(graph, 'result', resultBuffer, 'uint32', 2);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'spend-boundary',
        operation: 'GPUPolygonTriangulation',
        variant: 'budget-boundary-control',
        invocationCount: 1,
        bindings: [{name: 'result', view: result, type: 'u32', access: 'read_write'}],
        declarations: getTriangulationSpendWGSL(fixture.limit),
        body: `workDone = ${fixture.start}u; failed = false;
${fixture.amounts.map(amount => `spend(${amount}u);`).join('\n')}
result[resultOffset] = workDone; result[resultOffset + 1u] = select(0u, 1u, failed);`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(Array.from(await readUint32(resultBuffer, 2))).toEqual(fixture.expected);
    } finally {
      compiled.destroy();
      resultBuffer.destroy();
    }
  });
}
