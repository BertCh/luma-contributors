// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPUTransitionDiagnostics} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  AUDIT_SUPPORT_CASES,
  AUDIT_TRANSITIONS,
  AUDIT_PARALLEL_DIAGNOSTICS
} from './audit-dynamics-fixture';

/** Execute owned diagnostic WGSL; supply public reduction semantics where the serial VM cannot synchronize barriers. */
function executeDiagnostics(
  counts: number[],
  classCount: number,
  conditionCount: number,
  minimumTransitions = 5,
  includeSummaries = true
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `audit-${serial++}`, format, length);
  const input = view('uint32', counts.length);
  const output = {
    homogeneity: conditionCount > 1 && includeSummaries ? view('float32', 5) : undefined,
    steadyStates: view('float32', (conditionCount > 1 ? conditionCount + 1 : 1) * classCount),
    stateSparsity: includeSummaries ? view('uint32', 2) : undefined,
    status: {converged: view('uint32', 1)}
  };
  const memory = new Map<GraphDataView['buffer'], ArrayBuffer>();
  const getMemory = (field: GraphDataView) => {
    let bytes = memory.get(field.buffer);
    if (!bytes) {
      bytes = new ArrayBuffer(field.buffer.byteLength);
      new Uint32Array(bytes).fill(0x7f7f7f7f);
      memory.set(field.buffer, bytes);
    }
    return bytes;
  };
  new Uint32Array(getMemory(input)).set(counts);
  try {
    const contributor = new GPUTransitionDiagnostics({
      counts: input,
      classCount,
      conditionCount,
      minimumTransitions,
      output
    });
    const nodes = contributor.getCommandNodes(graph);
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      if (!record) {
        // The independent expected values remain SciPy literals. These host sums stand in for
        // published primitives only; the headless oracle spec executes the actual GPU reductions.
        const resources = node.resources!.filter(resource => 'buffer' in resource);
        const fields = resources.map(resource => resource.buffer as GraphDataView);
        const values = (field: GraphDataView) =>
          field.format === 'float32'
            ? new Float32Array(getMemory(field), field.byteOffset, field.length)
            : new Uint32Array(getMemory(field), field.byteOffset, field.length);
        if (node.workload?.operation === 'GPUSegmentedReduction') {
          const input = values(fields[0]);
          const offsets = values(fields[1]);
          const output = values(fields[2]);
          for (let segment = 0; segment < output.length; segment++) {
            let total = 0;
            for (let row = offsets[segment]; row < offsets[segment + 1]; row++) total += input[row];
            output[segment] = total;
          }
        } else {
          expect(fields).toHaveLength(2);
          const input = values(fields[0]);
          const output = values(fields[1]);
          if (node.id.endsWith('-finalize')) output[0] = input[0];
          else
            for (let block = 0; block < output.length; block++) {
              let total = 0;
              for (const value of input.slice(block * 256, (block + 1) * 256)) total += value;
              output[block] = total;
            }
        }
        continue;
      }
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => {
          const range = getViewBindingRange(binding.view);
          const bytes = getMemory(binding.view);
          return [
            index,
            binding.type === 'f32'
              ? new Float32Array(bytes, range.offset, range.size / 4)
              : new Uint32Array(bytes, range.offset, range.size / 4)
          ];
        })
      );
      // WgslExec needs the explicit three-axis attribute and does not handle phony assignments.
      const source = record.source
        .replace(/^\s*_\s*=[^;]*;/gm, '')
        // VM lacks ldexp. This helper's exponent is >= -126, so exp2 is an exact normal
        // power of two; this test-only substitution supplies the builtin's scaling semantics.
        .replace(
          'ldexp(f32(errorUnits), i32(exponentSum) - 300)',
          'f32(errorUnits) * exp2(f32(i32(exponentSum) - 300))'
        )
        .replace(/@workgroup_size\((\d+)\)/g, '@workgroup_size($1, 1, 1)');
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: bindings},
        {constants: {...record.constants}}
      );
    }
    return {
      homogeneity: output.homogeneity
        ? new Float32Array(getMemory(output.homogeneity), output.homogeneity.byteOffset, 5)
        : undefined,
      steadyStates: new Float32Array(
        getMemory(output.steadyStates),
        output.steadyStates.byteOffset,
        output.steadyStates.length
      ),
      stateSparsity: output.stateSparsity
        ? new Uint32Array(getMemory(output.stateSparsity), output.stateSparsity.byteOffset, 2)
        : new Uint32Array(),
      converged: new Uint32Array(
        getMemory(output.status.converged),
        output.status.converged.byteOffset,
        1
      )
    };
  } finally {
    device.destroy();
  }
}

it('executes actual diagnostic WGSL against the SciPy oracle on every port', () => {
  for (const expected of AUDIT_TRANSITIONS) {
    const actual = executeDiagnostics(expected.counts, 3, 2);
    expectNumericArrayClose(actual.homogeneity!, expected.homogeneity, {
      absoluteTolerance: 2e-4,
      relativeTolerance: 1e-5
    });
    expectNumericArrayClose(actual.steadyStates, expected.steadyStates, {absoluteTolerance: 2e-6});
    expectNumericArrayClose(actual.stateSparsity, expected.stateSparsity);
    expectNumericArrayClose(actual.converged, expected.converged);
  }
  for (const expected of AUDIT_SUPPORT_CASES) {
    const actual = executeDiagnostics(expected.counts, 2, 1);
    expectNumericArrayClose(actual.steadyStates, expected.steadyStates);
    expectNumericArrayClose(actual.stateSparsity, expected.stateSparsity);
    expectNumericArrayClose(actual.converged, expected.converged);
  }
});

it('uses published reductions for every condition-axis summary, rather than a single-invocation row scan', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  try {
    const contributor = new GPUTransitionDiagnostics({
      counts: createTransientView(graph, 'counts', 'uint32', 257 * 3 * 3),
      classCount: 3,
      conditionCount: 257,
      output: {
        homogeneity: createTransientView(graph, 'homogeneity', 'float32', 5),
        stateSparsity: createTransientView(graph, 'sparsity', 'uint32', 2),
        steadyStates: createTransientView(graph, 'steady-states', 'float32', 258 * 3),
        status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.filter(node => node.workload?.operation === 'GPUSegmentedReduction')).toHaveLength(
      0
    );
    expect(nodes.some(node => node.id.includes('-sum-pearson-'))).toBe(true);
    expect(nodes.some(node => node.id.includes('-sum-sparse-'))).toBe(true);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
    for (const node of nodes) {
      const record = getWGSLKernelNodeRecord(graph, node.id);
      if (record) expect(record.source).not.toMatch(/(?:<|<=)\s*C\b/);
    }
  } finally {
    device.destroy();
  }
});

it('checks every diagnostic port across the 256-condition reduction boundary against SciPy', () => {
  const expected = AUDIT_PARALLEL_DIAGNOSTICS;
  const actual = executeDiagnostics(expected.counts, 3, 257);
  for (const [index, value] of expected.homogeneity.entries()) {
    expectNumericArrayClose([actual.homogeneity![index]], [value], {
      absoluteTolerance: index === 1 || index === 3 ? 0 : 1e-5,
      relativeTolerance: index === 1 ? 2e-3 : 1e-5
    });
  }
  expectNumericArrayClose(actual.stateSparsity, expected.stateSparsity);
  expectNumericArrayClose(actual.steadyStates, expected.steadyStates, {absoluteTolerance: 2e-6});
  expectNumericArrayClose(actual.converged, expected.converged);
  const standalone = executeDiagnostics(expected.counts, 3, 257, 5, false);
  expectNumericArrayClose(standalone.steadyStates, expected.steadyStates, {
    absoluteTolerance: 2e-6
  });
  expectNumericArrayClose(standalone.converged, expected.converged);
});

it('executes large-count and overflowing-weight WGSL regression cases', () => {
  const large = executeDiagnostics(
    [0xffffffff, 1, 1, 0xffffffff, 0xffffffff, 1, 1, 0xffffffff],
    2,
    2
  );
  expectNumericArrayClose(large.homogeneity!, [0, 1, 0, 1, 2], {absoluteTolerance: 1e-5});
  expectNumericArrayClose(large.steadyStates, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], {
    absoluteTolerance: 2e-6
  });
  expectNumericArrayClose(large.stateSparsity, [0, 0]);
  expectNumericArrayClose(large.converged, [1]);
  const counts = new Array<number>(64).fill(0);
  for (let state = 0; state < 8; state++) {
    if (state < 7) counts[state * 8 + state + 1] = 1_000_000_000;
    if (state > 0) counts[state * 8 + state - 1] = 1;
  }
  counts[63] = 1_000_000_000;
  const biased = executeDiagnostics(counts, 8, 1);
  expectNumericArrayClose(biased.steadyStates, new Array<number>(8).fill(NaN));
  expectNumericArrayClose(biased.stateSparsity, [0, 0]);
  expectNumericArrayClose(biased.converged, [0]);
});

it('preserves cross-product differences below a float32 expected-count ulp', () => {
  const actual = executeDiagnostics([4721, 4722, 0, 0, 4720, 4721, 0, 0], 2, 2);
  expectNumericArrayClose([actual.homogeneity![2]], [2.375953939555561e-12], {
    relativeTolerance: 2e-5
  });
  expectNumericArrayClose([actual.homogeneity![3]], [0.9999987701303906], {
    absoluteTolerance: 1e-7
  });
});

it('bounds the exact integer sparsity threshold before shader generation', () => {
  expect(() => executeDiagnostics([1, 0, 0, 1], 2, 1, 0x100000000)).toThrow(
    /transition-diagnostics minimumTransitions/
  );
  const actual = executeDiagnostics([0xffffffff, 1, 1, 0xffffffff], 2, 1, 0xffffffff);
  expectNumericArrayClose(actual.stateSparsity, [0, 0]);
  expectNumericArrayClose(actual.steadyStates, [0.5, 0.5]);
  expectNumericArrayClose(actual.converged, [1]);
});

for (const [first, second, likelihood, probability] of [
  [4143, 4141, 0.000965717054287385, 0.9752089389095286],
  [574, 528, 3.8414063913019163, 0.05000156344204883]
]) {
  it(`preserves centered likelihood-ratio evidence for nearby regimes ${first}/${second}`, () => {
    // scipy.special.xlogy and chi2.sf, df=1; every expected cell is (first+second)/2.
    const actual = executeDiagnostics([first, second, 0, 0, second, first, 0, 0], 2, 2);
    expectNumericArrayClose([actual.homogeneity![2]], [likelihood], {relativeTolerance: 2e-5});
    expectNumericArrayClose([actual.homogeneity![3]], [probability], {absoluteTolerance: 1e-6});
    expect(actual.homogeneity![3]).toBeGreaterThan(0.05);
  });
}

for (const [count, likelihood] of [
  [2 ** 27, 1.0339757656912846e-25],
  [2 ** 23, 4.235164736271532e-22]
]) {
  it(`retains exact integer cross-products for GPU uint32 counts near ${count}`, () => {
    // Decimal(110), same contingency table as the independent pure CPU regression.
    const actual = executeDiagnostics([count, count + 1, 0, 0, count - 1, count, 0, 0], 2, 2)
      .homogeneity!;
    expect(actual[2]).toBeGreaterThan(0);
    expectNumericArrayClose([actual[2]], [likelihood], {relativeTolerance: 2e-5});
    expect(actual[3]).toBe(1);
  });
}
