// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const Contributor of [
  GPUSpatialErrorGM,
  GPUSpatialTwoStageLeastSquares,
  GPUSpatialWeightsTraces,
  GPUSpatialLagMaximumLikelihood,
  GPUSpatialErrorMaximumLikelihood
]) {
  for (const chunked of [false, true]) {
    if (chunked && Contributor === GPUSpatialTwoStageLeastSquares) continue;
    it(`${Contributor.name} suppresses only logical outputs for each ${chunked ? 'root/chunk' : 'whole'} loss and recovers`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `rereview-${serial++}`, format, length);
      const flags = Array.from({length: chunked ? 4 : 2}, () => view('uint32', 1));
      const whole = {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 8),
        weights: view('float32', 8),
        status: {overflow: flags[0], candidateOverflow: flags[1]}
      };
      const weights = chunked
        ? {
            rowCount: 8,
            chunks: [{rowStart: 0, rowCount: 8, weights: whole}],
            status: {overflow: flags[2], candidateOverflow: flags[3]}
          }
        : whole;
      // These are the published logical lengths for eight rows, one predictor and two powers.
      // The oracle is the loss contract: NaN for each dependent float, code 3, converged 0.
      // Spare words 98/99 belong to the caller and no result writer may change them.
      const lengths =
        Contributor === GPUSpatialWeightsTraces
          ? {
              powerTraces: 2,
              powerTraceStandardErrors: 2,
              powerSums: 2,
              infinityNorm: 1,
              ...(!chunked ? {gramTraces: 4} : {})
            }
          : {
              table: 12,
              summary:
                Contributor === GPUSpatialErrorGM
                  ? 13
                  : Contributor === GPUSpatialTwoStageLeastSquares
                    ? 16
                    : 22,
              residuals: 8,
              ...(Contributor === GPUSpatialLagMaximumLikelihood
                ? {powerTraces: 2, powerSums: 2, infinityNorm: 1}
                : {}),
              ...(Contributor === GPUSpatialLagMaximumLikelihood ||
              Contributor === GPUSpatialErrorMaximumLikelihood
                ? {powerTraceStandardErrors: 2}
                : {})
            };
      const outputs = Object.fromEntries(
        Object.entries(lengths).map(([name, length]) => [name, view('float32', length + 2)])
      ) as Record<string, GraphDataView<'float32'>>;
      const status: Record<string, GraphDataView<'uint32'>> = Contributor ===
      GPUSpatialWeightsTraces
        ? {approximation: view('uint32', 3)}
        : Contributor === GPUSpatialLagMaximumLikelihood ||
            Contributor === GPUSpatialErrorMaximumLikelihood
          ? {code: view('uint32', 3), converged: view('uint32', 3)}
          : {code: view('uint32', 3)};
      const covariance = view('float32', 9);
      const robustStatus = view('uint32', 1);
      const callerStatus: Record<string, GraphDataView<'uint32'>> = {...status};
      const callerOutput: Record<string, unknown> = {...outputs, status: callerStatus};
      try {
        const contributor = new Contributor({
          weights: weights as typeof whole,
          predictors: view('float32', 8),
          response: view('float32', 8),
          predictorCount: 1,
          maximumPower: 2,
          mode: 'stochastic',
          probeCount: 1,
          estimator: 'kelejian-prucha',
          traceMode: 'stochastic',
          traceProbeCount: 1,
          output: callerOutput as never,
          ...(Contributor === GPUSpatialTwoStageLeastSquares
            ? {robust: {method: 'HC0' as const, output: {covariance, status: {code: robustStatus}}}}
            : {})
        });
        // Replacing caller maps must not redirect the validated result or loss bindings.
        for (const [name, length] of Object.entries(lengths))
          callerOutput[name] = view('float32', length + 2);
        for (const name of Object.keys(callerStatus)) callerStatus[name] = view('uint32', 3);
        whole.status.overflow = view('uint32', 1);
        whole.status.candidateOverflow = view('uint32', 1);
        weights.status.overflow = view('uint32', 1);
        weights.status.candidateOverflow = view('uint32', 1);
        const nodes = contributor
          .getCommandNodes(graph)
          .filter(node => /weights-loss|suppress-incomplete/.test(node.id));
        const memory = createCPUGraphMemory();
        for (const selected of flags) {
          for (const loss of [0, 9, 0]) {
            for (const flag of flags)
              memory.write(flag, Uint32Array.of(flag === selected ? loss : 0));
            for (const [name, length] of Object.entries(lengths))
              memory.write(
                outputs[name],
                Float32Array.from([...new Array(length).fill(7), 98, 99])
              );
            for (const scalar of Object.values(status))
              memory.write(scalar, Uint32Array.of(1, 98, 99));
            memory.write(covariance, Float32Array.of(7, 7, 7, 7, 7, 7, 7, 7, 7));
            memory.write(robustStatus, Uint32Array.of(1));
            executeCommandNodesOnCPU(nodes, memory);
            for (const [name, length] of Object.entries(lengths)) {
              expect(Array.from(memory.readFloat32(outputs[name])), name).toEqual([
                ...new Array(length).fill(loss ? NaN : 7),
                98,
                99
              ]);
            }
            for (const [name, scalar] of Object.entries(status))
              expect(Array.from(memory.readUint32(scalar)), name).toEqual([
                loss && name !== 'approximation' ? (name === 'code' ? 3 : 0) : 1,
                98,
                99
              ]);
            if (Contributor === GPUSpatialTwoStageLeastSquares) {
              expect(Array.from(memory.readFloat32(covariance))).toEqual([
                ...new Array(9).fill(loss ? NaN : 7)
              ]);
              expect(Array.from(memory.readUint32(robustStatus))).toEqual([loss ? 3 : 1]);
            }
          }
        }
      } finally {
        device.destroy();
      }
    });
  }
}

for (const chunked of [false, true]) {
  it(`executes the complete ${chunked ? 'chunked' : 'whole'} three-row trace graph against a hand-derived oracle`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    let serial = 0;
    const input = <Format extends 'float32' | 'uint32'>(
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const view = createTransientView(graph, `oracle-input-${serial++}`, format, values.length);
      memory.write(view, values);
      return view;
    };
    const result = (length: number) =>
      input('float32', Float32Array.from([...new Array(length).fill(77), 98, 99]));
    const flags = Array.from({length: chunked ? 4 : 2}, () => input('uint32', Uint32Array.of(0)));
    const offsets = input('uint32', Uint32Array.of(0, 1, 3, 4));
    const whole = {
      offsets,
      neighbors: input('uint32', Uint32Array.of(1, 0, 2, 1)),
      weights: input('float32', Float32Array.of(1, 1, 1, 1)),
      status: {overflow: flags[0], candidateOverflow: flags[1]}
    };
    const laterOffsets = input('uint32', Uint32Array.of(0, 2, 3));
    const laterNeighbors = input('uint32', Uint32Array.of(0, 2, 1));
    const firstChild = {
      rowStart: 0,
      rowCount: 1,
      weights: {
        offsets: input('uint32', Uint32Array.of(0, 1)),
        neighbors: input('uint32', Uint32Array.of(1)),
        weights: input('float32', Float32Array.of(1))
      }
    };
    const laterChild = {
      rowStart: 1,
      rowCount: 2,
      weights: {
        offsets: laterOffsets,
        neighbors: laterNeighbors,
        weights: input('float32', Float32Array.of(1, 1, 1)),
        status: whole.status
      }
    };
    // One power keeps this interpreter-only fixture within the first aligned probe block.
    // wgsl_reflect TypedData discards a bound typed array's byteOffset; the GPU spec
    // covers two powers and the nonzero aligned binding offsets without that limitation.
    const powerTraces = result(1);
    const powerSums = result(1);
    const powerTraceStandardErrors = result(1);
    const infinityNorm = result(1);
    const approximation = input('uint32', Uint32Array.of(77, 98, 99));
    // For the three-vertex path W, ||W||inf=2. A=W/2 has diagonal zero and
    // A1=[1/2,1,1/2].
    // Exact basis probes have no sampling error; approximation is zero.
    const expectations = [
      {view: powerTraces, values: [0]},
      {view: powerSums, values: [2]},
      {view: powerTraceStandardErrors, values: [0]},
      {view: infinityNorm, values: [2]}
    ];
    try {
      const nodes = new GPUSpatialWeightsTraces({
        weights: chunked
          ? {
              rowCount: 3,
              chunks: [firstChild, laterChild],
              status: {overflow: flags[2], candidateOverflow: flags[3]}
            }
          : whole,
        maximumPower: 1,
        mode: 'exact',
        output: {
          powerTraces,
          powerSums,
          powerTraceStandardErrors,
          infinityNorm,
          status: {approximation}
        }
      }).getCommandNodes(graph);
      for (const selected of flags) {
        for (const loss of [0, 9, 0]) {
          for (const flag of flags)
            memory.write(flag, Uint32Array.of(flag === selected ? loss : 0));
          memory.write(offsets, Uint32Array.from(loss ? [0, 1, 2, 3] : [0, 1, 3, 4]));
          memory.write(laterOffsets, Uint32Array.from(loss ? [0, 1, 2] : [0, 2, 3]));
          memory.write(laterNeighbors, Uint32Array.from(loss ? [0, 1, 1] : [0, 2, 1]));
          executeCommandNodesOnCPU(nodes, memory);
          for (const {view, values} of expectations) {
            const actual = memory.readFloat32(view);
            expect(Array.from(actual.slice(values.length))).toEqual([98, 99]);
            for (const [index, expected] of values.entries()) {
              if (loss) expect(Number.isNaN(actual[index])).toBe(true);
              else {
                expect(Number.isFinite(actual[index])).toBe(true);
                // Float32 reduction allowance for this fixed three-row fixture.
                expect(
                  Math.abs(actual[index] - expected),
                  JSON.stringify({actual: Array.from(actual), expected: values})
                ).toBeLessThanOrEqual(1e-5);
              }
            }
          }
          expect(Array.from(memory.readUint32(approximation))).toEqual([0, 98, 99]);
        }
      }
    } finally {
      device.destroy();
    }
  });
}
