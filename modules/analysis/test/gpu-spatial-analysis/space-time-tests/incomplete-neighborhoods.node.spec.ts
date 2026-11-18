// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUKnoxTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-knox-test';
import {GPUMantelTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test';
import {GPUJacquezTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-jacquez-test';
import {GPUSpaceTimeLocalMoran} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-local-moran';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const kind of ['knox', 'mantel', 'jacquez', 'moran', 'lag', 'chunked-lag'] as const) {
  it(`${kind} consumes both neighborhood loss flags and suppresses incomplete results`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <F extends 'uint32' | 'float32'>(format: F, length: number) =>
      createTransientView(graph, `loss-${serial++}`, format, length);
    try {
      const overflow = view('uint32', 1),
        candidateOverflow = view('uint32', 1);
      const weights = {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 4),
        weights: view('float32', 4),
        distances: view('float32', 4),
        status: {overflow, candidateOverflow}
      };
      const parameters = view('uint32', 4),
        times = view('float32', 4);
      const summary = view('float32', 13),
        statistics = view('float32', 4);
      const counts = view('uint32', 4),
        lag = view('float32', 4);
      const completeness = view('uint32', 1);
      const permutationCount = view('uint32', 2);
      const contributor =
        kind === 'knox'
          ? new GPUKnoxTest({
              pairs: weights,
              times,
              parameters,
              maximumPermutations: 3,
              stopAfterExceedances: 3,
              output: {statistics: counts, summary, permutationCount}
            })
          : kind === 'mantel'
            ? new GPUMantelTest({
                pairs: weights,
                times,
                parameters,
                maximumPermutations: 3,
                stopAfterExceedances: 3,
                output: {statistics, summary, permutationCount}
              })
            : kind === 'jacquez'
              ? new GPUJacquezTest({
                  neighbors: weights,
                  times,
                  parameters,
                  k: 1,
                  maximumPermutations: 3,
                  stopAfterExceedances: 3,
                  output: {statistics: counts, summary, permutationCount}
                })
              : kind === 'moran'
                ? new GPUSpaceTimeLocalMoran({
                    weights,
                    cube: view('float32', 8),
                    layout: 'period-major',
                    periodCount: 2,
                    period: 1,
                    variant: 'lagged',
                    maximumPermutations: 3,
                    parameters,
                    output: {
                      exceedances: counts,
                      pseudoPValues: statistics,
                      status: {overflow: completeness}
                    }
                  })
                : new GPUSpatialLag({
                    weights:
                      kind === 'lag'
                        ? weights
                        : {
                            rowCount: 4,
                            status: weights.status,
                            chunks: [
                              {rowStart: 0, rowCount: 4, weights: {...weights, status: undefined}}
                            ]
                          },
                    values: times,
                    output: lag
                  });
      const nodes = contributor.getCommandNodes(graph);
      for (const flag of [overflow, candidateOverflow]) {
        expect(
          nodes.some(node =>
            getWGSLKernelNodeRecord(graph, node.id)?.bindings.some(
              binding => binding.view === flag && binding.access === 'read'
            )
          ),
          `${kind} must read ${flag.buffer.id}`
        ).toBe(true);
      }
      if (kind === 'moran') {
        const gates = nodes.filter(node => node.id.includes('-incomplete-analysis-'));
        expect(gates.length).toBe(2);
        const analysisFlags = getWGSLKernelNodeRecord(graph, gates[0].id)!.bindings.find(
          binding => binding.name === 'analysisFlags'
        )!.view;
        const memory = createCPUGraphMemory();
        for (const active of [overflow, candidateOverflow]) {
          memory.write(overflow, Uint32Array.of(0));
          memory.write(candidateOverflow, Uint32Array.of(0));
          memory.write(active, Uint32Array.of(1));
          memory.write(analysisFlags, Uint32Array.of(1, 1, 1, 1));
          executeCommandNodesOnCPU(gates, memory);
          expect(Array.from(memory.readUint32(analysisFlags))).toEqual([0, 0, 0, 0]);
          memory.write(completeness, Uint32Array.of(0));
          executeCommandNodesOnCPU(
            nodes.filter(node => node.id.includes('-weights-overflow')),
            memory
          );
          expect(Array.from(memory.readUint32(completeness))).toEqual([1]);
        }
      } else {
        const guards = nodes.filter(node => node.id.includes('-incomplete-'));
        expect(guards.length).toBeGreaterThan(0);
        for (const guard of guards) {
          const record = getWGSLKernelNodeRecord(graph, guard.id)!;
          expect(record.source).not.toContain('\\n');
          const quietNaN = new WgslReflect(record.source).functions.find(
            declaration => declaration.name === 'getQuietNaN'
          );
          expect(quietNaN?.arguments.length, `${kind} getQuietNaN(index) call arity`).toBe(1);
        }
        const memory = createCPUGraphMemory();
        const result = kind.endsWith('lag') ? lag : summary;
        for (const active of [overflow, candidateOverflow]) {
          memory.write(overflow, new Uint32Array([0]));
          memory.write(candidateOverflow, new Uint32Array([0]));
          memory.write(active, new Uint32Array([1]));
          memory.write(result, new Float32Array(result.length).fill(2));
          if (!kind.endsWith('lag')) {
            memory.write(
              summary,
              Float32Array.from({length: 13}, (_, index) => (index === 12 ? 73 : 2))
            );
            memory.write(permutationCount, Uint32Array.of(0, 71));
          }
          executeCommandNodesOnCPU(guards, memory);
          expect(
            Array.from(memory.readFloat32(result))
              .slice(0, kind.endsWith('lag') ? result.length : 12)
              .every(Number.isNaN)
          ).toBe(true);
          if (!kind.endsWith('lag')) {
            expect(memory.readFloat32(summary)[12]).toBe(73);
            expect(memory.readUint32(permutationCount)[1]).toBe(71);
          }
          memory.write(active, new Uint32Array([0]));
          memory.write(result, new Float32Array(result.length).fill(10));
          if (!kind.endsWith('lag'))
            memory.write(
              summary,
              Float32Array.from({length: 13}, (_, index) => (index === 12 ? 73 : 10))
            );
          executeCommandNodesOnCPU(guards, memory);
          expect(Array.from(memory.readFloat32(result))).toEqual(
            Array.from({length: result.length}, (_, index) =>
              !kind.endsWith('lag') && index === 12 ? 73 : 10
            )
          );
        }
      }
    } finally {
      device.destroy();
    }
  });
}

it('G6 Moran retains input loss and output completeness maps before deferred lowering', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <F extends 'uint32' | 'float32'>(format: F, length: number) =>
    createTransientView(graph, `moran-map-${serial++}`, format, length);
  try {
    const loss = view('uint32', 1),
      completeness = view('uint32', 1);
    const status: {candidateOverflow?: typeof loss} = {candidateOverflow: loss};
    const output = {
      exceedances: view('uint32', 4),
      pseudoPValues: view('float32', 4),
      status: {overflow: completeness}
    };
    const contributor = new GPUSpaceTimeLocalMoran({
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 4),
        weights: view('float32', 4),
        status
      },
      cube: view('float32', 8),
      layout: 'period-major',
      periodCount: 2,
      period: 1,
      variant: 'lagged',
      maximumPermutations: 3,
      parameters: view('uint32', 4),
      output
    });
    delete status.candidateOverflow;
    output.status.overflow = view('uint32', 1);
    expect(contributor.props.output.status.overflow).toBe(completeness);
    const nodes = contributor
      .getCommandNodes(graph)
      .filter(node => node.id.includes('-weights-overflow'));
    const memory = createCPUGraphMemory();
    memory.write(loss, Uint32Array.of(1));
    memory.write(completeness, Uint32Array.of(0));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(completeness))).toEqual([1]);
  } finally {
    device.destroy();
  }
});

for (const kind of ['knox', 'mantel', 'jacquez'] as const) {
  for (const mutate of [false, true])
    it(`G5 ${kind} snapshots getter summary and never writes an extra auxiliary output, mutate=${mutate}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      let serial = 0;
      const view = <F extends 'float32' | 'uint32'>(format: F, length: number) =>
        createTransientView(graph, `getter-${serial++}`, format, length);
      try {
        const loss = view('uint32', 1),
          summary = view('float32', 12),
          auxiliary = view('float32', 1);
        let target = summary;
        const output = Object.assign(
          Object.create({
            get summary() {
              return target;
            }
          }),
          {statistics: view(kind === 'mantel' ? 'float32' : 'uint32', 4), auxiliary}
        );
        const weights = {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 3),
          weights: view('float32', 3),
          distances: view('float32', 3),
          status: {candidateOverflow: loss}
        };
        const common = {
          times: view('float32', 3),
          parameters: view('uint32', 4),
          maximumPermutations: 3,
          output
        };
        const contributor =
          kind === 'knox'
            ? new GPUKnoxTest({...common, pairs: weights})
            : kind === 'mantel'
              ? new GPUMantelTest({...common, pairs: weights})
              : new GPUJacquezTest({...common, neighbors: weights, k: 1});
        if (mutate) {
          target = view('float32', 12);
          delete (weights.status as {candidateOverflow?: typeof loss}).candidateOverflow;
        }
        const guards = contributor
          .getCommandNodes(graph)
          .filter(node => node.id.includes('-incomplete-'));
        const memory = createCPUGraphMemory();
        memory.write(loss, Uint32Array.of(1));
        memory.write(summary, new Float32Array(12).fill(0.25));
        memory.write(auxiliary, Float32Array.of(42));
        executeCommandNodesOnCPU(guards, memory);
        expect(Array.from(memory.readFloat32(summary))).toEqual(new Array(12).fill(NaN));
        expect(Array.from(memory.readFloat32(auxiliary))).toEqual([42]);
      } finally {
        device.destroy();
      }
    });
}
