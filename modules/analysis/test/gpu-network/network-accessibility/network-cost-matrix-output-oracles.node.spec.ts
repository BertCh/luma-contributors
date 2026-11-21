// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  computeNetworkCostMatrixOnCPU,
  CPUNetworkCostMatrixOperation
} from '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation';
import {GPUNetworkCostMatrixOperation} from '../../../src/gpu-network/network-accessibility/gpu-network-cost-matrix-operation';
import {
  expectOracleRows,
  getOracleCPUVectors,
  getOracleOperationNodes
} from '../alternative-routes/output-oracle-harness';
import {
  createCostMatrixOutputOracle,
  createCostMatrixSpillOracle,
  createCostMatrixDefaultOracle,
  createCostMatrixRewriteOracle
} from './network-cost-matrix-output-oracles';

it('R4 shares matrix operation budget maxima across CPU and GPU', () => {
  for (const Operation of [CPUNetworkCostMatrixOperation, GPUNetworkCostMatrixOperation]) {
    const {props, inputs} = createCostMatrixOutputOracle(false);
    expect(() => new Operation({...props, maximumIterations: 1025})).toThrow(
      /maximumIterations.*1024/
    );
    expect(() => new Operation({...props, localIterations: 65})).toThrow(/localIterations.*64/);
    expect(
      () => new Operation({...props, maximumIterations: 1024, localIterations: 64})
    ).not.toThrow();
    if (Operation === GPUNetworkCostMatrixOperation)
      expect(() =>
        getOracleOperationNodes(
          new GPUNetworkCostMatrixOperation({
            ...props,
            maximumIterations: 1024,
            localIterations: 64
          }),
          inputs
        )
      ).not.toThrow();
  }
});

it('R9 changes the same matrix operation seed and rejects a cached first answer', async () => {
  const {props, inputs, first, second} = createCostMatrixRewriteOracle();
  const metadata = new GPUNetworkCostMatrixOperation(props);
  const operation = new CPUNetworkCostMatrixOperation(props);
  const initial = await operation.execute({inputs: getOracleCPUVectors(inputs, metadata.inputs)});
  for (const [port, rows] of Object.entries(first))
    expectOracleRows(initial[port] as Float32Array | Uint32Array, rows);
  // Reproduce the old harness's weakness: replaying the initial answer passes run equality.
  const cachedReplay = initial;
  expect(cachedReplay).toEqual(initial);
  expect(() => expectOracleRows(cachedReplay['costs'] as Float32Array, second.costs)).toThrow();
  inputs['starts'].values = Uint32Array.of(2);
  const updated = await operation.execute({inputs: getOracleCPUVectors(inputs, metadata.inputs)});
  for (const [port, rows] of Object.entries(second))
    expectOracleRows(updated[port] as Float32Array | Uint32Array, rows);
});

it('R5 captures GPU budgets before lowering', () => {
  const {props, inputs} = createCostMatrixOutputOracle(false);
  props.maximumIterations = 1;
  const operation = new GPUNetworkCostMatrixOperation(props);
  props.maximumIterations = 8;
  const nodes = getOracleOperationNodes(operation, inputs);
  // Two batches, one round each. Mutating the caller must not add relaxation commands.
  expect(nodes.filter(node => node.id.match(/-relax-\d+$/))).toHaveLength(2);
});

it('R5 reads nested matrix status once on each backend', () => {
  for (const Operation of [CPUNetworkCostMatrixOperation, GPUNetworkCostMatrixOperation]) {
    const fixture = createCostMatrixOutputOracle(false);
    let reads = 0;
    Object.defineProperty(fixture.props.output, 'status', {
      get() {
        reads++;
        return {converged: `status${reads}`};
      }
    });
    const captured = new Operation(fixture.props);
    expect(reads).toBe(1);
    if (captured instanceof GPUNetworkCostMatrixOperation) {
      expect(captured.outputs['converged'].value).toBe('status1');
      getOracleOperationNodes(captured, fixture.inputs);
      expect(reads).toBe(1);
    }
  }
});

it('R4 pins exactly sixteen default local hops', async () => {
  const {props, inputs, expected} = createCostMatrixDefaultOracle();
  const metadata = new GPUNetworkCostMatrixOperation(props);
  const result = await new CPUNetworkCostMatrixOperation(props).execute({
    inputs: getOracleCPUVectors(inputs, metadata.inputs)
  });
  for (const [port, rows] of Object.entries(expected))
    expectOracleRows(result[port] as Float32Array | Uint32Array, rows);
});

for (const gather of [false, true])
  it(`pins enumerated chain matrix and convergence (gather ${gather})`, () => {
    const {input, expected} = createCostMatrixOutputOracle(gather);
    const result = computeNetworkCostMatrixOnCPU(input, {rowCount: 3});
    expectOracleRows(result.costs, expected);
    expectOracleRows(result.converged, [1]);
    expectOracleRows(result.iterationLimitReached, [0]);
  });

for (const neighborCount of [256, 257] as const)
  it(`R1 preserves local queue spill across matrix rounds (${neighborCount} neighbors)`, async () => {
    const {props, inputs, expected} = createCostMatrixSpillOracle(neighborCount);
    const metadata = new GPUNetworkCostMatrixOperation(props);
    const result = await new CPUNetworkCostMatrixOperation(props).execute({
      inputs: getOracleCPUVectors(inputs, metadata.inputs)
    });
    for (const [port, rows] of Object.entries(expected))
      expectOracleRows(result[port] as Float32Array | Uint32Array, rows);
  });

it('binds an explicit one-hop operation budget and publishes incomplete status', async () => {
  const {props, inputs} = createCostMatrixOutputOracle(false);
  props.nodeCount = 3;
  props.rowCount = 1;
  props.laneCount = 1;
  props.maximumIterations = 1;
  inputs['offsets'].values = Uint32Array.of(0, 1, 2, 2);
  inputs['starts'].values = Uint32Array.of(0);
  inputs['startCosts'].values = Float32Array.of(1);
  inputs['startRows'].values = Uint32Array.of(0);
  props.output.status = {converged: 'converged', iterationLimitReached: 'iterationLimitReached'};
  const metadata = new GPUNetworkCostMatrixOperation(props);
  const result = await new CPUNetworkCostMatrixOperation(props).execute({
    inputs: getOracleCPUVectors(inputs, metadata.inputs)
  });
  // Only 0->1 is examined in row zero; the distance-6 label at node 2 is beyond the budget.
  expectOracleRows(result['costs'] as Float32Array, [1, 3, Infinity]);
  expectOracleRows(result['converged'] as Uint32Array, [0]);
  expectOracleRows(result['iterationLimitReached'] as Uint32Array, [1]);
});

it('keeps omitted matrix budgets exact and captures operation budgets', async () => {
  const {input, props, inputs} = createCostMatrixOutputOracle(false);
  const bounded = computeNetworkCostMatrixOnCPU(input, {
    rowCount: 3,
    maximumIterations: 1,
    localIterations: 1
  });
  expectOracleRows(bounded.costs, [
    1,
    3,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    0.5,
    Infinity,
    Infinity,
    Infinity,
    Infinity,
    Infinity
  ]);
  expectOracleRows(bounded.converged, [0]);
  const defaultHops = computeNetworkCostMatrixOnCPU(input, {rowCount: 3, maximumIterations: 1});
  expectOracleRows(defaultHops.costs, createCostMatrixOutputOracle(false).expected);
  expectOracleRows(defaultHops.converged, [1]);
  props.maximumIterations = 1;
  props.output.status = {iterationLimitReached: 'limited'};
  const metadata = new GPUNetworkCostMatrixOperation(props);
  const operation = new CPUNetworkCostMatrixOperation(props);
  props.maximumIterations = 8;
  props.localIterations = 16;
  props.output.status.iterationLimitReached = 'changed';
  const result = await operation.execute({inputs: getOracleCPUVectors(inputs, metadata.inputs)});
  expectOracleRows(result['costs'] as Float32Array, Array.from(bounded.costs));
  expectOracleRows(result['limited'] as Uint32Array, [1]);
});
