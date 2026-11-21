// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {CPUTransitionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-transition-diagnostics-operation';
import {
  GPUTransitionDiagnosticsOperation,
  type TransitionDiagnosticsOperationProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics-operation';
import {
  GPUTransitionDiagnostics,
  type GPUTransitionDiagnosticsProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const OPERATION_PROPS = {
  counts: 'counts',
  classCount: 1,
  output: {stateSparsity: 'sparsity'}
} as const;

/** Props whose `minimumTransitions` lives on the prototype, as a class instance's would. */
function inherit<Props extends object>(own: Props, minimumTransitions: number): Props {
  return Object.assign(Object.create({minimumTransitions}), own) as Props;
}

/** Own accessor alternating between `values`; counts how often it was read. */
function alternate<Props extends object>(own: Props, values: number[]) {
  let reads = 0;
  const props = {...own} as Props & {minimumTransitions: number};
  Object.defineProperty(props, 'minimumTransitions', {
    enumerable: true,
    get: () => values[reads++ % values.length]
  });
  return {props, getReads: () => reads};
}

async function executeSparsity(operation: CPUTransitionDiagnosticsOperation, count: number) {
  const values = Uint32Array.of(count);
  const actual = await operation.execute({
    inputs: {
      counts: {format: 'uint32', length: 1, chunks: [{values, length: 1}], pack: () => values}
    }
  });
  return Array.from(actual['sparsity'] as Uint32Array);
}

it('rejects an inherited GPU threshold above uint32 in the operation and the contributor', () => {
  // GPU bound: a uint32 shader constant, so [1, 2^32 - 1].
  for (const value of [2 ** 32, 0]) {
    expect(
      () => new GPUTransitionDiagnosticsOperation(inherit({...OPERATION_PROPS}, value))
    ).toThrow(/minimumTransitions/);
  }
  expect(
    () => new GPUTransitionDiagnosticsOperation(inherit({...OPERATION_PROPS}, 2 ** 32 - 1))
  ).not.toThrow();
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const own: GPUTransitionDiagnosticsProps = {
      counts: createTransientView(graph, 'counts', 'uint32', 1),
      classCount: 1,
      output: {stateSparsity: createTransientView(graph, 'sparsity', 'uint32', 2)}
    };
    expect(() => new GPUTransitionDiagnostics(inherit(own, 2 ** 32))).toThrow(/minimumTransitions/);
    expect(new GPUTransitionDiagnostics(inherit(own, 2 ** 32 - 1)).minimumTransitions).toBe(
      2 ** 32 - 1
    );
    const {props, getReads} = alternate(own, [7, 2 ** 32]);
    const contributor = new GPUTransitionDiagnostics(props);
    expect(getReads()).toBe(1);
    expect(contributor.minimumTransitions).toBe(7);
    expect(contributor.props.minimumTransitions).toBe(7);
  } finally {
    device.destroy();
  }
});

it('validates and uses the inherited CPU threshold over its documented safe-integer range', async () => {
  // CPU range: [1, Number.MAX_SAFE_INTEGER].
  expect(
    () =>
      new CPUTransitionDiagnosticsOperation(
        inherit({...OPERATION_PROPS}, Number.MAX_SAFE_INTEGER + 1)
      )
  ).toThrow(/minimumTransitions/);
  expect(() => new CPUTransitionDiagnosticsOperation(inherit({...OPERATION_PROPS}, 0))).toThrow(
    /minimumTransitions/
  );
  // Six transitions: sparse under an inherited threshold of 7, not under the default of 5.
  const inherited = new CPUTransitionDiagnosticsOperation(inherit({...OPERATION_PROPS}, 7));
  expect(await executeSparsity(inherited, 6)).toEqual([1, 0]);
  const boundary = new CPUTransitionDiagnosticsOperation(
    inherit({...OPERATION_PROPS}, Number.MAX_SAFE_INTEGER)
  );
  expect(await executeSparsity(boundary, 6)).toEqual([1, 0]);
  const own = new CPUTransitionDiagnosticsOperation({...OPERATION_PROPS, minimumTransitions: 7});
  expect(await executeSparsity(own, 6)).toEqual([1, 0]);
});

it('reads an alternating threshold accessor once in both operation forms', async () => {
  const gpu = alternate<TransitionDiagnosticsOperationProps>({...OPERATION_PROPS}, [7, 2 ** 32]);
  expect(() => new GPUTransitionDiagnosticsOperation(gpu.props)).not.toThrow();
  expect(gpu.getReads()).toBe(1);
  const invalid = alternate<TransitionDiagnosticsOperationProps>({...OPERATION_PROPS}, [
    2 ** 32,
    7
  ]);
  expect(() => new GPUTransitionDiagnosticsOperation(invalid.props)).toThrow(/minimumTransitions/);
  expect(invalid.getReads()).toBe(1);
  const cpu = alternate<TransitionDiagnosticsOperationProps>({...OPERATION_PROPS}, [7, 5]);
  const operation = new CPUTransitionDiagnosticsOperation(cpu.props);
  expect(cpu.getReads()).toBe(1);
  // The validated first value (7) is the one used: six transitions are sparse.
  expect(await executeSparsity(operation, 6)).toEqual([1, 0]);
});
