// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPULineSimplification,
  GPULineSimplificationOperation
} from '../../../src/gpu-spatial-analysis/line-simplification/index';
import {
  computeLineSimplificationOnCPU,
  CPULineSimplificationOperation
} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';
import {
  createSimplificationOracleFixture,
  getSimplificationOperationProps,
  SIMPLIFICATION_INPUT
} from './line-simplification-output-oracle-fixture';

useCPUExecutorTestHooks();

function createMutationMemory() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: OracleView = (format, length, values) => {
    const result = createTransientView(graph, `mutation-${serial++}`, format, length);
    const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  return {graph, memory, view};
}

for (const method of ['douglas-peucker', 'visvalingam'] as const) {
  it(`clamps the right anchor to the last backing row (${method})`, () => {
    const {graph, memory, view} = createMutationMemory();
    try {
      const importance = view('float32', 4);
      const contributor = new GPULineSimplification({
        positions: view('float32x2', 4, [99, 99, 88, 88, 0, 0, 1, 0]),
        lineOffsets: view('uint32', 2, [2, 14]),
        importance,
        method,
        maximumIterations: 1
      });
      executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
      // [2, 14) clamps to [2, 4) and both rows are endpoints.
      expect(memory.readFloat32(importance)).toEqual(Float32Array.of(0, 0, Infinity, Infinity));
    } finally {
      graph.device.destroy();
    }
  });
}

function getChordCPUInputs() {
  const {positions, lineOffsets, parameters} = SIMPLIFICATION_INPUT;
  return {
    positions: {
      format: 'float32x2' as const,
      length: 3,
      chunks: [{values: positions, length: 3}],
      pack: () => positions
    },
    lineOffsets: {
      format: 'uint32' as const,
      length: 2,
      chunks: [{values: lineOffsets, length: 2}],
      pack: () => lineOffsets
    },
    parameters: {
      format: 'float32' as const,
      length: 4,
      chunks: [{values: parameters, length: 4}],
      pack: () => parameters
    }
  };
}

for (const field of ['count', 'overflow', 'requiredCount', 'candidateOverflow']) {
  for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
    it(`${Operation.name} refuses importance-only selection diagnostic ${field}`, () => {
      expect(
        () =>
          new Operation({
            ...getSimplificationOperationProps(false),
            output: {status: {converged: 'converged', [field]: field}}
          })
      ).toThrow(/selection|requires.*ids|output.ids/);
    });
  }
  it(`direct simplification refuses importance-only selection diagnostic ${field}`, () => {
    const {view} = createMutationMemory();
    const status = {converged: view('uint32', 1), [field]: view('uint32', 1)};
    expect(
      () =>
        new GPULineSimplification({
          positions: view('float32x2', 3),
          lineOffsets: view('uint32', 2),
          importance: view('float32', 3),
          maximumIterations: 2,
          output: {status}
        })
    ).toThrow(/selection|requires.*ids|output.ids/);
  });
}

for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
  it(`${Operation.name} refuses unsupported approximation status`, () => {
    const props = getSimplificationOperationProps(true, 1);
    const status = {...props.output!.status, approximation: 'approximation'};
    expect(() => new Operation({...props, output: {...props.output, status}})).toThrow(
      /approximation/
    );
  });
}

it('direct simplification refuses unsupported approximation status', () => {
  const {view} = createMutationMemory();
  const status = {converged: view('uint32', 1), approximation: view('uint32', 1)};
  expect(
    () =>
      new GPULineSimplification({
        positions: view('float32x2', 3),
        lineOffsets: view('uint32', 2),
        importance: view('float32', 3),
        output: {status}
      })
  ).toThrow(/approximation/);
});

it('GPU operation captures getter-backed candidate status for registration and lowering', () => {
  const {graph, memory, view} = createMutationMemory();
  const fixture = createSimplificationOracleFixture(view, 'operation', 1, true, true);
  expect(fixture.operation!.outputs['candidateOverflow']).toBeDefined();
  executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(fixture.ids))).toEqual([0]);
  expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([1]);
  expect(Array.from(memory.readUint32(fixture.candidateOverflow!))).toEqual([0]);
});

it('CPU operation captures getter-backed status and its declared capacity before mutation', async () => {
  const props = getSimplificationOperationProps(true, 1, true, true);
  const operation = new CPULineSimplificationOperation(props);
  props.capacity = 2;
  const result = await operation.execute({inputs: getChordCPUInputs()});
  expect(result).toMatchObject({
    ids: Uint32Array.of(0),
    count: Uint32Array.of(1),
    overflow: Uint32Array.of(1),
    candidateOverflow: Uint32Array.of(0)
  });
});

it('GPU operation preserves registered candidate destination when caller deletes it', () => {
  const {graph, memory, view} = createMutationMemory();
  const props = getSimplificationOperationProps(true, 1, true);
  const operation = new GPULineSimplificationOperation(props);
  delete props.output!.status!.candidateOverflow;
  const ports = {
    positions: view('float32x2', 3, Array.from(SIMPLIFICATION_INPUT.positions)),
    lineOffsets: view('uint32', 2, [0, 3]),
    parameters: view('float32', 4, Array.from(SIMPLIFICATION_INPUT.parameters)),
    importance: view('float32', 3),
    ids: view('uint32', 1),
    count: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  };
  const contributor = operation.createContributor(
    port => ports[port as keyof typeof ports] as never
  );
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(ports.candidateOverflow))).toEqual([0]);
  expect(Array.from(memory.readUint32(ports.overflow))).toEqual([1]);
});

it('direct simplification captures a diagnostic before caller rebinds it to input offsets', () => {
  const {graph, memory, view} = createMutationMemory();
  const lineOffsets = view('uint32', 2, [1, 4]);
  const original = view('uint32', 1);
  const status = {
    count: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: original
  };
  const contributor = new GPULineSimplification({
    positions: view('float32x2', 4, [99, 99, 0, 0, 1, 0.125, 2, 0]),
    lineOffsets,
    importance: view('float32', 4, [0, Infinity, 0.125, Infinity]),
    computeImportance: false,
    parameters: view('float32', 4, Array.from(SIMPLIFICATION_INPUT.parameters)),
    output: {ids: view('uint32', 2), status}
  });
  status.candidateOverflow = graph.createDataView(lineOffsets.buffer, {
    format: 'uint32',
    length: 1
  });
  const nodes = contributor.getCommandNodes(graph);
  for (let submission = 0; submission < 2; submission++) {
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(lineOffsets))).toEqual([1, 4]);
    expect(Array.from(memory.readUint32(original))).toEqual([0]);
  }
});

it('CPU simplification and its operation retain the two endpoints by the chord distance oracle', async () => {
  const pure = computeLineSimplificationOnCPU(SIMPLIFICATION_INPUT, {
    capacity: 2,
    maximumIterations: 2
  });
  expect(Array.from(pure.ids)).toEqual([0, 2]);
  expect(Array.from(pure.importance)).toEqual([Infinity, 0.125, Infinity]);
  const operation = new CPULineSimplificationOperation(getSimplificationOperationProps());
  const {positions, lineOffsets, parameters} = SIMPLIFICATION_INPUT;
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 3,
        chunks: [{values: positions, length: 3}],
        pack: () => positions
      },
      lineOffsets: {
        format: 'uint32',
        length: 2,
        chunks: [{values: lineOffsets, length: 2}],
        pack: () => lineOffsets
      },
      parameters: {
        format: 'float32',
        length: 4,
        chunks: [{values: parameters, length: 4}],
        pack: () => parameters
      }
    }
  });
  expect(result).toMatchObject({
    ids: Uint32Array.of(0, 2),
    count: Uint32Array.of(2),
    overflow: Uint32Array.of(0)
  });
});

for (const mode of ['operation', 'candidate', 'absent'] as const) {
  it(`simplification executes the chord oracle in ${mode} mode`, () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view: OracleView = (format, length, values) => {
      const result = createTransientView(graph, `simplify-${serial++}`, format, length);
      const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
      memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
      return result;
    };
    const fixture = createSimplificationOracleFixture(view, mode);
    executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(fixture.ids))).toEqual([0, 2]);
    expect(Array.from(memory.readUint32(fixture.count))).toEqual([2]);
    expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([0]);
    expect(Array.from(memory.readFloat32(fixture.importance))).toEqual([Infinity, 0.125, Infinity]);
    if (fixture.candidateOverflow)
      expect(Array.from(memory.readUint32(fixture.candidateOverflow))).toEqual([0]);
    fixture.operation?.destroy();
  });
}

it('simplification GPU operation publishes candidate zero separately from final overflow', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view: OracleView = (format, length, values) => {
    const result = createTransientView(graph, `candidate-${serial++}`, format, length);
    const data = values ?? new Array<number>(length * (format === 'float32x2' ? 2 : 1)).fill(93);
    memory.write(result, format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data));
    return result;
  };
  const fixture = createSimplificationOracleFixture(view, 'operation', 1, true);
  executeCommandNodesOnCPU(fixture.contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(fixture.ids))).toEqual([0]);
  expect(Array.from(memory.readUint32(fixture.count))).toEqual([1]);
  expect(Array.from(memory.readUint32(fixture.overflow))).toEqual([1]);
  expect(Array.from(memory.readUint32(fixture.candidateOverflow!))).toEqual([0]);
  fixture.operation?.destroy();
});

it('simplification CPU operation publishes candidate zero separately from final overflow', async () => {
  const operation = new CPULineSimplificationOperation(
    getSimplificationOperationProps(true, 1, true)
  );
  const {positions, lineOffsets, parameters} = SIMPLIFICATION_INPUT;
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 3,
        chunks: [{values: positions, length: 3}],
        pack: () => positions
      },
      lineOffsets: {
        format: 'uint32',
        length: 2,
        chunks: [{values: lineOffsets, length: 2}],
        pack: () => lineOffsets
      },
      parameters: {
        format: 'float32',
        length: 4,
        chunks: [{values: parameters, length: 4}],
        pack: () => parameters
      }
    }
  });
  expect(result).toMatchObject({
    ids: Uint32Array.of(0),
    count: Uint32Array.of(1),
    overflow: Uint32Array.of(1),
    candidateOverflow: Uint32Array.of(0)
  });
});

it('simplification refuses candidateOverflow aliased to input line offsets', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view: OracleView = (format, length) =>
    createTransientView(graph, `alias-${serial++}`, format, length);
  const lineOffsets = view('uint32', 2);
  const candidateOverflow = graph.createDataView(lineOffsets.buffer, {format: 'uint32', length: 1});
  expect(
    () =>
      new GPULineSimplification({
        positions: view('float32x2', 4),
        lineOffsets,
        importance: view('float32', 4),
        computeImportance: false,
        parameters: view('float32', 4),
        output: {
          ids: view('uint32', 2),
          status: {count: view('uint32', 1), overflow: view('uint32', 1), candidateOverflow}
        }
      })
  ).toThrow(/^line-simplification outputs must not share buffers with inputs/);
});

for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
  it(`${Operation.name} requires selection for candidateOverflow`, () => {
    expect(
      () =>
        new Operation({
          ...getSimplificationOperationProps(false),
          output: {status: {candidateOverflow: 'candidateOverflow'}}
        })
    ).toThrow(/candidateOverflow|selection columns require output.ids/);
  });
}

for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
  it(`${Operation.name} rejects a non-enumerable inherited unsupported status getter once`, () => {
    const props = getSimplificationOperationProps(true, 1);
    let reads = 0;
    const prototype = Object.defineProperty({}, 'approximation', {
      get() {
        reads++;
        return 'approximation';
      }
    });
    const status = {...props.output!.status};
    Object.setPrototypeOf(status, prototype);
    expect(() => new Operation({...props, output: {...props.output, status}})).toThrow(
      /approximation/
    );
    expect(reads).toBeLessThanOrEqual(1);
  });
}

for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
  it(`B6 ${Operation.name} accepts supported status getters beside prototype methods`, async () => {
    let reads = 0;
    class Status {
      get iterationCount() {
        reads++;
        return 'iterations';
      }
      toJSON() {
        return {};
      }
    }
    const props = {...getSimplificationOperationProps(false), output: {status: new Status()}};
    const operation = new Operation(props);
    expect(reads).toBe(1);
    if (operation instanceof CPULineSimplificationOperation) {
      const result = await operation.execute({inputs: getChordCPUInputs()});
      expect(result['iterations']).toEqual(Uint32Array.of(1));
    } else {
      const {graph, memory, view} = createMutationMemory();
      const ports = {
        positions: view('float32x2', 3, [0, 0, 1, 0.125, 2, 0]),
        lineOffsets: view('uint32', 2, [0, 3]),
        importance: view('float32', 3),
        iterationCount: view('uint32', 1)
      };
      const contributor = operation.createContributor(
        port => ports[port as keyof typeof ports] as never
      );
      executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
      expect(Array.from(memory.readUint32(ports.iterationCount))).toEqual([1]);
    }
    expect(reads).toBe(1);
  });
}
