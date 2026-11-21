// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  CPUSpatialWeightsValidationOperation,
  computeSpatialWeightsValidationOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {getGPUSpatialWeightsKernelValue} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {planGPUSpatialWeightsChunks} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import {AUDIT_TRANSFORM_CASES} from './audit-weights-a-transform-fixture';
import {AUDIT_ALGEBRA_LEFT} from './audit-weights-a-algebra-fixture';
import {AUDIT_BUILDER_NAMES, createAuditBuilder} from './audit-weights-a-builders-fixture';

it('audits all CSR violation predicates against literal independent row flags with empty rows', () => {
  const actual = computeSpatialWeightsValidationOnCPU(
    Uint32Array.of(0, 3, 4, 5, 6, 6, 8, 9, 9),
    Uint32Array.of(0, 1, 1, 2, 9, 3, 2, 1, 0),
    Float32Array.of(1, 1, NaN, 0, -1, 1, 1, 1, Infinity)
  );
  // audit-weights-a-helpers.oracle.py evaluates each mathematical predicate independently.
  expect(Array.from(actual.flags)).toEqual([28, 0, 18, 8, 0, 4, 16, 0]);
  expect(Array.from(actual.invalidCount)).toEqual([5]);
});

for (const mutation of ['rowCount', 'outputNames'] as const) {
  it(`CPU validation snapshots ${mutation} before execution`, () => {
    const props = {
      rowCount: 1,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {flags: 'flags', status: {invalidCount: 'invalid'}}
    };
    const operation = new CPUSpatialWeightsValidationOperation(props);
    if (mutation === 'rowCount') props.rowCount = 2;
    else {
      props.output.flags = 'renamedFlags';
      props.output.status.invalidCount = 'renamedInvalid';
    }
    const vector = (format: 'uint32' | 'float32', values: Uint32Array | Float32Array) => ({
      format,
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    // One self slot has the independent self-violation bit 8; exactly one row is invalid.
    const actual = operation.execute({
      inputs: {
        offsets: vector('uint32', Uint32Array.of(0, 1)),
        neighbors: vector('uint32', Uint32Array.of(0)),
        weights: vector('float32', Float32Array.of(1))
      }
    });
    expect(actual).toEqual({flags: Uint32Array.of(8), invalid: Uint32Array.of(1)});
  });
}

describe('small builder kernels against independent literal oracles', () => {
  useCPUExecutorTestHooks();
  // The interpreter cannot execute load-balanced lag's workgroup barriers; the headless case
  // exercises those kernels against the same independent dense-multiplication literals.
  for (const name of AUDIT_BUILDER_NAMES.filter(
    name => name !== 'lag' && name !== 'normalizedLag'
  )) {
    it(`audits ${name} in the CPU shader interpreter`, () => {
      for (const [scale, shift] of [
        [1, 0],
        [8, 1048576]
      ]) {
        const device = createNullWebGPUDevice();
        const graph = new GPUCommandGraph(device);
        const memory = createCPUGraphMemory();
        const getBuffer = memory.getBuffer;
        memory.getBuffer = logical => {
          const buffer = getBuffer(logical);
          if (buffer.backing.byteLength > logical.byteLength)
            buffer.backing = buffer.backing.slice(0, logical.byteLength);
          return buffer;
        };
        let serial = 0;
        const checks: (() => void)[] = [];
        const contributor = createAuditBuilder(
          {
            input: (values, format) => {
              const view = createTransientView(
                graph,
                `builder-input-${serial++}`,
                format,
                values.length / (format === 'float32x2' ? 2 : 1)
              );
              memory.write(view, values);
              return view;
            },
            output: (port, format, expected) => {
              const view = createTransientView(
                graph,
                `builder-output-${serial++}`,
                format,
                expected.length
              );
              memory.write(
                view,
                format === 'float32'
                  ? new Float32Array(expected.length).fill(999)
                  : new Uint32Array(expected.length).fill(99)
              );
              checks.push(() => {
                const actual = Array.from(
                  format === 'float32' ? memory.readFloat32(view) : memory.readUint32(view)
                );
                if (format === 'uint32') expect(actual, port).toEqual(expected);
                else
                  for (const [row, value] of actual.entries()) {
                    expect(Number.isFinite(value), port).toBe(true);
                    expect(Math.abs(value - expected[row]), port).toBeLessThanOrEqual(2e-6);
                  }
              });
              return view;
            }
          },
          name,
          scale,
          shift
        );
        try {
          executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
          for (const check of checks) check();
        } finally {
          device.destroy();
        }
      }
    });
  }
});

describe('island attachment inherited loss', () => {
  useCPUExecutorTestHooks();
  it('publishes upstream incompleteness and clears candidate status on recovery', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `island-audit-${serial++}`, format, length);
    const inputLoss = view('uint32', 1);
    const inputCandidateLoss = view('uint32', 1);
    const weights = {
      offsets: view('uint32', 2),
      neighbors: view('uint32', 1),
      weights: view('float32', 1),
      status: {overflow: inputLoss, candidateOverflow: inputCandidateLoss}
    };
    const output = {
      offsets: view('uint32', 2),
      neighbors: view('uint32', 1),
      weights: view('float32', 1),
      attachment: view('uint32', 1),
      added: view('uint32', 1),
      status: {
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1),
        candidateOverflow: view('uint32', 1)
      }
    };
    const positions = view('float32x2', 1);
    const memory = createCPUGraphMemory();
    memory.write(weights.offsets, Uint32Array.of(0, 0));
    memory.write(positions, Float32Array.of(0, 0));
    memory.write(output.status.candidateOverflow, Uint32Array.of(99));
    const contributor = new GPUIslandAttachment({
      weights,
      positions,
      reciprocal: false,
      nearestSearch: 'scan',
      output
    });
    try {
      const nodes = contributor.getCommandNodes(graph);
      for (const [loss, candidateLoss] of [
        [1, 0],
        [0, 1],
        [0, 0]
      ]) {
        memory.write(inputLoss, Uint32Array.of(loss));
        memory.write(inputCandidateLoss, Uint32Array.of(candidateLoss));
        executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readUint32(output.status.candidateOverflow))).toEqual([
          Number(Boolean(loss || candidateLoss))
        ]);
        expect(Array.from(memory.readUint32(output.status.overflow))).toEqual([0]);
        expect(Array.from(memory.readUint32(output.status.requiredCount))).toEqual([0]);
        expect(Array.from(memory.readUint32(output.offsets))).toEqual([0, 0]);
        expect(Array.from(memory.readUint32(output.attachment))).toEqual([0xffffffff]);
      }
    } finally {
      device.destroy();
    }
  });

  it('owns a candidate loss destination when the caller omits it', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `owned-island-${serial++}`, format, length);
    const candidateLoss = view('uint32', 1);
    const offsets = view('uint32', 2);
    const positions = view('float32x2', 1);
    const contributor = new GPUIslandAttachment({
      reciprocal: false,
      nearestSearch: 'scan',
      positions,
      weights: {
        offsets,
        neighbors: view('uint32', 1),
        weights: view('float32', 1),
        status: {candidateOverflow: candidateLoss}
      },
      output: {
        offsets: view('uint32', 2),
        neighbors: view('uint32', 1),
        weights: view('float32', 1),
        attachment: view('uint32', 1),
        added: view('uint32', 1),
        status: {overflow: view('uint32', 1)}
      }
    });
    const memory = createCPUGraphMemory();
    memory.write(offsets, Uint32Array.of(0, 0));
    memory.write(positions, Float32Array.of(0, 0));
    try {
      const destination = contributor.outputWeights.status?.candidateOverflow;
      expect(destination).toBeDefined();
      const nodes = contributor.getCommandNodes(graph);
      for (const loss of [1, 0]) {
        memory.write(candidateLoss, Uint32Array.of(loss));
        executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readUint32(destination!))).toEqual([loss]);
      }
    } finally {
      device.destroy();
    }
  });
});

it('audits exported kernel profiles against independent fixed-bandwidth literals', () => {
  for (const {options, expected} of AUDIT_TRANSFORM_CASES) {
    if (options.operation !== 'kernel' || typeof options.bandwidth !== 'number') continue;
    for (const [slot, distance] of AUDIT_ALGEBRA_LEFT.distances.entries()) {
      const actual = getGPUSpatialWeightsKernelValue(options.kernel!, distance / options.bandwidth);
      expect(Number.isFinite(actual)).toBe(true);
      expect(Math.abs(actual - expected[slot])).toBeLessThanOrEqual(1e-14);
    }
    for (const invalid of [-1, NaN, Infinity])
      expect(getGPUSpatialWeightsKernelValue(options.kernel!, invalid)).toBe(0);
  }
});

it('audits chunk planning against hand-counted consecutive capacity ranges including zero and oversized rows', () => {
  // Greedy consecutive groups, cutting before exceeding four slots; an oversized row stands alone.
  expect(planGPUSpatialWeightsChunks([0, 2, 0, 3, 8, 0, 1, 0], 4)).toEqual([
    {rowStart: 0, rowCount: 3, slotCount: 2},
    {rowStart: 3, rowCount: 1, slotCount: 3},
    {rowStart: 4, rowCount: 1, slotCount: 8},
    {rowStart: 5, rowCount: 3, slotCount: 1}
  ]);
});
