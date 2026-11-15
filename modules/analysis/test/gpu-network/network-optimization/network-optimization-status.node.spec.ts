// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {
  computeNetworkRationalAccessOnCPU,
  CPUNetworkRationalAccessOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-rational-access-operation';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {GPUNetworkTimeRouting} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {
  computeNetworkFlowOnCPU,
  CPUNetworkFlowOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {
  computeNetworkTimeRoutingOnCPU,
  CPUNetworkTimeRoutingOperation
} from '../../../src/gpu-network/network-optimization/cpu-network-time-routing-operation';
import {GPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {GPUNetworkTimeRoutingOperation} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing-operation';
import {GPUNetworkRationalAccessOperation} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createNetworkOptimizationStatusNodes} from '../../../src/gpu-network/network-optimization/network-optimization-status';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

for (const name of ['overflow', 'summary', 'constructor']) {
  it(`CPU and GPU operation forms reject unsupported status.${name} before declaring outputs`, () => {
    const status = {invalidCount: 'invalid', [name]: 'extra'};
    const flowProps = {
      id: 'extra-flow',
      nodeCount: 2,
      edgeCount: 1,
      edges: 'edges',
      capacities: 'capacities',
      parameters: 'parameters',
      output: {flow: 'flow', cut: 'cut', summary: 'summary', status}
    };
    const timeProps = {
      id: 'extra-time',
      stateCount: 2,
      transitionCount: 1,
      stateVertices: 'stateVertices',
      transitions: 'transitions',
      costs: 'costs',
      endpoints: 'endpoints',
      departure: 'departure',
      output: {
        arrivals: 'arrivals',
        predecessorTransitions: 'predecessors',
        summary: 'summary',
        status
      }
    };
    const accessProps = {
      id: 'extra-access',
      originCount: 1,
      facilityCount: 1,
      costs: 'costs',
      demand: 'demand',
      supply: 'supply',
      parameters: 'parameters',
      output: {
        assignments: 'assignments',
        scores: 'scores',
        unservedDemand: 'unservedDemand',
        summary: 'summary',
        status
      }
    };
    for (const construct of [
      () => new CPUNetworkFlowOperation(flowProps),
      () => new GPUNetworkFlowOperation(flowProps),
      () => new CPUNetworkTimeRoutingOperation(timeProps),
      () => new GPUNetworkTimeRoutingOperation(timeProps),
      () => new CPUNetworkRationalAccessOperation(accessProps),
      () => new GPUNetworkRationalAccessOperation(accessProps)
    ]) {
      expect
        .soft(construct)
        .toThrow(new RegExp(`^extra-[a-z]+ output.status.${name} must be a supported field`));
    }
  });
}

it('CPU flow executes supported scalar outputs and preserves the eight-word primary summary', async () => {
  const operation = new CPUNetworkFlowOperation({
    nodeCount: 2,
    edgeCount: 1,
    edges: 'edges',
    capacities: 'capacities',
    parameters: 'parameters',
    output: {
      flow: 'flow',
      cut: 'cut',
      summary: 'summary',
      status: {
        invalidCount: 'invalid',
        iterationCount: 'iterations',
        converged: 'converged',
        iterationLimitReached: 'limit'
      }
    }
  });
  const vector = (format: 'uint32' | 'uint32x2', values: Uint32Array) => {
    const length = values.length / (format === 'uint32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  expect(operation.outputs['summary']).toMatchObject({format: 'uint32', length: 8});
  for (const name of ['invalid', 'iterations', 'converged', 'limit']) {
    expect(operation.outputs[name]).toMatchObject({format: 'uint32', length: 1});
  }
  const result = await operation.execute({
    inputs: {
      edges: vector('uint32x2', Uint32Array.of(0, 1)),
      capacities: vector('uint32', Uint32Array.of(3)),
      parameters: vector('uint32', Uint32Array.of(0, 1, 5))
    }
  });
  expect(result['flow']).toEqual(Uint32Array.of(3));
  expect(result['summary']).toHaveLength(8);
  expect(Array.from(result['summary'] as Uint32Array).slice(0, 5)).toEqual([1, 1, 3, 2, 1]);
  expect(result['invalid']).toEqual(Uint32Array.of(0));
  expect(result['iterations']).toEqual(Uint32Array.of(1));
  expect(result['converged']).toEqual(Uint32Array.of(1));
  expect(result['limit']).toEqual(Uint32Array.of(0));
  expect(Object.values(result).every(value => value !== undefined)).toBe(true);
});

it('CPU flow completion distinguishes an infeasible request from invalid inputs', () => {
  for (const invalid of [false, true]) {
    const result = computeNetworkFlowOnCPU(
      {
        edges: Uint32Array.of(0, 1),
        capacities: Uint32Array.of(3),
        costs: Uint32Array.of(2),
        parameters: Uint32Array.of(0, invalid ? 0 : 1, 5)
      },
      {nodeCount: 2}
    );
    expect(result.status.invalidCount).toEqual(Uint32Array.of(Number(invalid)));
    expect(result.status.iterationCount).toEqual(Uint32Array.of(invalid ? 0 : 1));
    expect(result.status.converged).toEqual(Uint32Array.of(Number(!invalid)));
    expect(result.status.iterationLimitReached).toEqual(Uint32Array.of(0));
    if (!invalid) {
      expect(result.flow).toEqual(Uint32Array.of(3));
      expect(result.summary[3]).toBe(2);
    }
  }
});

it('CPU time routing counts sweeps through the final fixpoint check', () => {
  for (const departure of [0, -1]) {
    const result = computeNetworkTimeRoutingOnCPU({
      stateVertices: Uint32Array.of(0, 1, 2),
      transitions: Uint32Array.of(1, 2, 0, 1),
      costs: Float32Array.of(1, 0, 1, 0),
      endpoints: Uint32Array.of(0, 2),
      departure: Float32Array.of(departure)
    });
    const valid = departure === 0;
    expect(result.status.invalidCount).toEqual(Uint32Array.of(Number(!valid)));
    expect(result.status.iterationCount).toEqual(Uint32Array.of(valid ? 3 : 0));
    expect(result.status.converged).toEqual(Uint32Array.of(Number(valid)));
    expect(result.status.iterationLimitReached).toEqual(Uint32Array.of(0));
    expect(result.arrivals).toEqual(
      valid ? Float32Array.of(0, 1, 2) : new Float32Array(3).fill(Infinity)
    );
  }
});

it('optimizer status ports parse with at most eight bindings, even with all optional outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `optimizer-status-${sequence++}`, format, length);
  const status = {
    invalidCount: view('uint32', 1),
    iterationCount: view('uint32', 1),
    converged: view('uint32', 1),
    iterationLimitReached: view('uint32', 1)
  };
  const flowProps = {
    id: 'flow-status',
    nodeCount: 2,
    edges: view('uint32x2', 1),
    capacities: view('uint32', 1),
    costs: view('uint32', 1),
    parameters: view('uint32', 3),
    output: {
      flow: view('uint32', 1),
      cut: view('uint32', 2),
      summary: view('uint32', 8),
      exactCost: view('uint32x2', 1),
      status
    }
  };
  const timeProps = {
    id: 'time-status',
    stateVertices: view('uint32', 3),
    transitions: view('uint32x2', 2),
    costs: view('float32x2', 2),
    endpoints: view('uint32', 2),
    departure: view('float32', 1),
    output: {
      arrivals: view('float32', 3),
      predecessorTransitions: view('uint32', 3),
      summary: view('uint32', 4),
      status
    }
  };
  const rationalProps = {
    id: 'rational-status',
    costs: view('float32', 2),
    demand: view('float32', 1),
    supply: view('float32', 2),
    parameters: view('float32', 6),
    output: {
      assignments: view('float32', 2),
      scores: view('float32', 1),
      unservedDemand: view('float32', 1),
      summary: view('uint32', 4),
      status
    }
  };
  try {
    expect(
      () =>
        new GPUNetworkRationalAccess({
          ...rationalProps,
          output: {
            ...rationalProps.output,
            status: {converged: status.invalidCount, invalidCount: status.invalidCount}
          }
        })
    ).toThrow(/distinct/);
    for (const contributor of [
      new GPUNetworkFlow(flowProps),
      new GPUNetworkTimeRouting(timeProps),
      new GPUNetworkRationalAccess(rationalProps)
    ]) {
      const nodes = contributor.getCommandNodes(graph);
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
    }
    for (const name of [
      'invalidCount',
      'iterationCount',
      'converged',
      'iterationLimitReached'
    ] as const) {
      for (const props of [flowProps, timeProps]) {
        const output = {...props.output, status: {[name]: view('uint32', 2)}};
        expect(() =>
          'nodeCount' in props
            ? new GPUNetworkFlow({...props, output: {...flowProps.output, status: output.status}})
            : new GPUNetworkTimeRouting({
                ...props,
                output: {...timeProps.output, status: output.status}
              })
        ).toThrow(new RegExp(`output.status.${name} must hold one row`));
      }
    }
    expect(
      () =>
        new GPUNetworkFlow({
          ...flowProps,
          output: {...flowProps.output, status: {converged: flowProps.output.flow}}
        })
    ).toThrow(/distinct/);
    expect(
      () =>
        new GPUNetworkTimeRouting({
          ...timeProps,
          output: {
            ...timeProps.output,
            status: {converged: status.invalidCount, invalidCount: status.invalidCount}
          }
        })
    ).toThrow(/distinct/);
    const otherGraph = new GPUCommandGraph(device);
    const foreign = createTransientView(otherGraph, 'foreign-status', 'uint32', 1);
    expect(() =>
      new GPUNetworkTimeRouting({
        ...timeProps,
        output: {...timeProps.output, status: {converged: foreign}}
      }).getCommandNodes(graph)
    ).toThrow(/time-status.*graph/);
    expect(
      new GPUNetworkTimeRouting({
        ...timeProps,
        output: {...timeProps.output, status: {converged: status.converged}}
      }).getCommandNodes(graph)
    ).toHaveLength(2);
    expect(
      new GPUNetworkTimeRouting({
        ...timeProps,
        output: {...timeProps.output, status: undefined}
      }).getCommandNodes(graph)
    ).toHaveLength(1);
  } finally {
    device.destroy();
  }
});

it('rational access reports equilibrium, a rounding stall and invalid inputs with named ports', () => {
  for (const kind of ['equilibrium', 'stall', 'invalid'] as const) {
    const result = computeNetworkRationalAccessOnCPU(
      {
        costs: Float32Array.of(0, 0),
        demand: Float32Array.of(kind === 'equilibrium' ? 0 : 1),
        supply: Float32Array.of(kind === 'invalid' ? 0 : 1, 1),
        parameters: Float32Array.of(1, 1, 1, 1, 50, 0)
      },
      {
        maximumCycles: 0,
        initialLimitCycles: 0,
        demandMode: kind === 'equilibrium' ? 'fractional' : 'integral'
      }
    );
    expect(result.status.invalidCount).toEqual(Uint32Array.of(Number(kind === 'invalid')));
    expect(result.status.iterationCount).toEqual(Uint32Array.of(0));
    expect(result.status.converged).toEqual(Uint32Array.of(Number(kind === 'equilibrium')));
    expect(result.status.iterationLimitReached).toEqual(Uint32Array.of(Number(kind === 'stall')));
  }
});

it('rejects unsupported own status keys carried by structurally compatible variables for every optimizer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `status-key-${sequence++}`, format, length);
  try {
    for (const name of ['overflow', 'constructor']) {
      const status = {invalidCount: view('uint32', 1), [name]: view('uint32', 1)};
      // Variables with extra fields are assignable to every optimizer's status port in TypeScript.
      const constructors = [
        () =>
          new GPUNetworkFlow({
            id: 'extra-flow',
            nodeCount: 2,
            edges: view('uint32x2', 1),
            capacities: view('uint32', 1),
            parameters: view('uint32', 3),
            output: {
              flow: view('uint32', 1),
              cut: view('uint32', 2),
              summary: view('uint32', 8),
              status
            }
          }),
        () =>
          new GPUNetworkTimeRouting({
            id: 'extra-time',
            stateVertices: view('uint32', 2),
            transitions: view('uint32x2', 1),
            costs: view('float32x2', 1),
            endpoints: view('uint32', 2),
            departure: view('float32', 1),
            output: {
              arrivals: view('float32', 2),
              predecessorTransitions: view('uint32', 2),
              summary: view('uint32', 4),
              status
            }
          }),
        () =>
          new GPUNetworkRationalAccess({
            id: 'extra-access',
            costs: view('float32', 1),
            demand: view('float32', 1),
            supply: view('float32', 1),
            parameters: view('float32', 6),
            output: {
              assignments: view('float32', 1),
              scores: view('float32', 1),
              unservedDemand: view('float32', 1),
              summary: view('uint32', 4),
              status
            }
          }),
        () =>
          new GPUNetworkFacilityOptimization({
            id: 'extra-facility',
            operation: 'capacitated-p-median',
            facilityCount: 1,
            selectCount: 1,
            costs: view('float32', 1),
            demand: view('float32', 1),
            capacities: view('float32', 1),
            threshold: view('float32', 1),
            output: {
              selectedFacilities: view('uint32', 1),
              assignments: view('uint32', 1),
              objective: view('float32', 1),
              summary: view('uint32', 4),
              status
            }
          })
      ];
      for (const construct of constructors) {
        expect(construct).toThrow(
          new RegExp(`^extra-[a-z]+ output.status.${name} must be a supported field`)
        );
      }
    }
  } finally {
    device.destroy();
  }
});

it('publishes only supported expression keys when a status bag reaches the shader helper directly', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const summary = createTransientView(graph, 'summary', 'uint32', 4);
    const status = {
      invalidCount: createTransientView(graph, 'invalid', 'uint32', 1),
      overflow: createTransientView(graph, 'unsupported-overflow', 'uint32', 1)
    };
    const nodes = createNetworkOptimizationStatusNodes(
      graph,
      'status-projection',
      'GPUNetworkFlow',
      summary,
      status,
      {
        invalidCount: '1u - summary[summaryOffset]',
        iterationCount: '0u',
        converged: 'summary[summaryOffset + 1u]',
        iterationLimitReached: '0u'
      }
    );
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(kernels).toHaveLength(1);
    expect(getStorageBindingCount(kernels[0].source)).toBe(2);
    expect(kernels[0].source).not.toMatch(/\bundefined\b|\boverflow\b/);
    expect(() => new WgslReflect(kernels[0].source)).not.toThrow();
  } finally {
    device.destroy();
  }
});
