// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLineOracleHarness} from '../line-clip/line-output-oracle-harness';
import {createSimplificationOracleFixture} from './line-simplification-output-oracle-fixture';
import {GPULineSimplificationOperation} from '../../../src/gpu-spatial-analysis/line-simplification/index';
import {GPULineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification/index';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {CPULineSimplificationOperation} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  getSimplificationOperationProps,
  SIMPLIFICATION_INPUT
} from './line-simplification-output-oracle-fixture';

it('simplification operation retains rows zero and two and candidate overflow is zero without candidates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['operation', 'candidate', 'absent'] as const) {
    const harness = createLineOracleHarness(device);
    const fixture = createSimplificationOracleFixture(harness.view, mode);
    harness.graph.add(fixture.contributor);
    const destroy = await harness.run();
    try {
      expect(await harness.read(fixture.importance)).toEqual([Infinity, 0.125, Infinity]);
      expect(await harness.read(fixture.ids)).toEqual([0, 2]);
      expect(await harness.read(fixture.count)).toEqual([2]);
      expect(await harness.read(fixture.overflow)).toEqual([0]);
      if (fixture.candidateOverflow)
        expect(await harness.read(fixture.candidateOverflow)).toEqual([0]);
    } finally {
      destroy();
      fixture.operation?.destroy();
      harness.destroy();
    }
  }
});

it('both simplification operations distinguish final truncation from candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const fixture = createSimplificationOracleFixture(harness.view, 'operation', 1, true, true);
  harness.graph.add(fixture.contributor);
  const destroy = await harness.run();
  try {
    expect(await harness.read(fixture.ids)).toEqual([0]);
    expect(await harness.read(fixture.count)).toEqual([1]);
    expect(await harness.read(fixture.overflow)).toEqual([1]);
    expect(await harness.read(fixture.candidateOverflow!)).toEqual([0]);
  } finally {
    destroy();
    fixture.operation?.destroy();
    harness.destroy();
  }
  for (const getterCandidate of [false, true])
    for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
      const props = getSimplificationOperationProps(true, 1, true, getterCandidate);
      const operation = new Operation(props);
      // The declared one-row capacity and registered diagnostic must survive edits
      // to the original settings and nested status descriptor before graph lowering.
      props.capacity = 2;
      if (!getterCandidate) delete props.output!.status!.candidateOverflow;
      const result = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: SIMPLIFICATION_INPUT.positions},
          lineOffsets: {format: 'uint32', values: SIMPLIFICATION_INPUT.lineOffsets},
          parameters: {format: 'float32', values: SIMPLIFICATION_INPUT.parameters}
        },
        operations: [operation],
        outputs: ['ids', 'count', 'overflow', 'candidateOverflow']
      });
      expect(Array.from(result['ids'])).toEqual([0]);
      expect(Array.from(result['count'])).toEqual([1]);
      expect(Array.from(result['overflow'])).toEqual([1]);
      expect(Array.from(result['candidateOverflow'])).toEqual([0]);
    }
});

it('direct simplification preserves captured destinations across two submissions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const {graph, view, read} = harness;
  const lineOffsets = view('uint32', 2, [1, 4]);
  const candidateOverflow = view('uint32', 1);
  const ids = view('uint32', 2);
  const status = {count: view('uint32', 1), overflow: view('uint32', 1), candidateOverflow};
  const contributor = new GPULineSimplification({
    positions: view('float32x2', 4, [99, 99, 0, 0, 1, 0.125, 2, 0]),
    lineOffsets,
    importance: view('float32', 4, [0, Infinity, 0.125, Infinity]),
    computeImportance: false,
    parameters: view('float32', 4, Array.from(SIMPLIFICATION_INPUT.parameters)),
    output: {ids, status}
  });
  status.candidateOverflow = graph.createDataView(lineOffsets.buffer, {
    format: 'uint32',
    length: 1
  });
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      submitGraph(device, compiled, undefined);
      expect(await read(lineOffsets)).toEqual([1, 4]);
      expect(await read(candidateOverflow)).toEqual([0]);
      expect(await read(ids)).toEqual([1, 3]);
      expect(await read(status.count)).toEqual([2]);
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('simplification operation preserves the importance oracle with optional selection absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createLineOracleHarness(device);
  const {view, graph, read} = harness;
  const ports = {
    positions: view('float32x2', 3, Array.from(SIMPLIFICATION_INPUT.positions)),
    lineOffsets: view('uint32', 2, [0, 3]),
    importance: view('float32', 3)
  };
  const operation = new GPULineSimplificationOperation(getSimplificationOperationProps(false));
  graph.add(operation.createContributor(port => ports[port as keyof typeof ports] as never));
  const destroy = await harness.run();
  try {
    expect(await read(ports.importance)).toEqual([Infinity, 0.125, Infinity]);
  } finally {
    destroy();
    operation.destroy();
    harness.destroy();
  }
});
