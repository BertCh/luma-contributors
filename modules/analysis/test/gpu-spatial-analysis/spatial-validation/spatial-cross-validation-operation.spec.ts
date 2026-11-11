// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {CPUSpatialCrossValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {
  GPUSpatialCrossValidationOperation,
  type SpatialCrossValidationOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {constructSamplingContributor} from '../point-process/validation-sampling-operation-test-utils';
import fixtures from './spatial-cross-validation-operation-fixtures.json';

for (const mode of [
  'blocks',
  'groups',
  'environmental',
  'nndm',
  'knndm',
  'knndm-generated'
] as const)
  for (const empty of [false, true]) {
    it(`compares every cross-validation output, design=${mode}, empty=${empty}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = fixtures[mode === 'environmental' ? 'environmental' : 'knndm'];
      const positions =
        mode === 'nndm'
          ? new Float32Array([0, 0, 1, 0, 3, 0, 7, 0])
          : mode === 'knndm-generated'
            ? new Float32Array([0, 0, 1, 1, 2, 0, 8, 0, 9, 1, 10, 0])
            : new Float32Array(fixture.positions);
      const rowCount = positions.length / 2;
      const design: SpatialCrossValidationOperationProps['design'] =
        mode === 'blocks'
          ? undefined
          : mode === 'groups'
            ? {mode, diagnostics: 'diagnostics'}
            : mode === 'environmental'
              ? {mode, features: 'features', featureCount: 3, diagnostics: 'diagnostics'}
              : mode === 'nndm'
                ? {
                    mode,
                    maximumDistance: 4,
                    predictionPositions: 'predictionPositions',
                    predictionRowCount: 2,
                    diagnostics: 'diagnostics'
                  }
                : mode === 'knndm'
                  ? {
                      mode,
                      predictionPositions: 'predictionPositions',
                      predictionRowCount:
                        fixture === fixtures.knndm
                          ? fixtures.knndm.predictionPositions.length / 2
                          : 0,
                      candidates: 'candidates',
                      candidateCount: 4,
                      diagnostics: 'diagnostics'
                    }
                  : {
                      mode: 'knndm',
                      predictionPositions: 'predictionPositions',
                      predictionRowCount: 3,
                      candidateCount: 3,
                      diagnostics: 'diagnostics'
                    };
      const props: SpatialCrossValidationOperationProps = {
        id: 'cross-validation-comparison',
        rowCount,
        positions: 'positions',
        blocks: 'blocks',
        mask: 'mask',
        seed: 1,
        bufferDistance: 0.6,
        foldCount: mode === 'nndm' ? rowCount : mode === 'blocks' || mode === 'groups' ? 3 : 2,
        design,
        output: {
          folds: 'folds',
          training: 'training',
          testing: 'testing',
          counts: 'counts',
          status: 'status'
        }
      };
      const inputs: Record<string, OperationGraphInput> = {
        positions: {format: 'float32x2', values: positions},
        blocks: {
          format: 'uint32',
          values: Uint32Array.from(
            {length: rowCount},
            (_, row) => [100, 7, 100, 999, 7, 42, 9, 11][row]
          )
        },
        mask: {format: 'uint32', values: new Uint32Array(rowCount).fill(empty ? 0 : 1)}
      };
      if (mode === 'environmental')
        inputs['features'] = {
          format: 'float32',
          values: new Float32Array(fixtures.environmental.features)
        };
      if (mode === 'nndm' || mode === 'knndm' || mode === 'knndm-generated')
        inputs['predictionPositions'] = {
          format: 'float32x2',
          values: new Float32Array(
            mode === 'nndm'
              ? [20, 0, 25, 0]
              : mode === 'knndm'
                ? fixtures.knndm.predictionPositions
                : [0, 2, 4, 2, 8, 2]
          )
        };
      if (mode === 'knndm')
        inputs['candidates'] = {
          format: 'uint32',
          values: new Uint32Array(fixtures.knndm.candidates)
        };
      const outputs = [...Object.values(props.output), ...(design ? ['diagnostics'] : [])];
      const gpu = await runOperationGraph(device, {
        inputs,
        operations: [new GPUSpatialCrossValidationOperation(props)],
        outputs
      });
      const cpu = await runOperationGraph(device, {
        inputs,
        operations: [new CPUSpatialCrossValidationOperation(props)],
        outputs
      });
      for (const output of Object.values(props.output))
        expect(Array.from(gpu[output]), output).toEqual(Array.from(cpu[output]));
      if (design)
        for (const [index, expected] of cpu['diagnostics'].entries()) {
          const actual = gpu['diagnostics'][index];
          if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
          // Status, iterations, selected candidate, and the NNDM finite sentinel compare exactly.
          else if (
            index === 0 ||
            index === 1 ||
            index === 3 ||
            expected === Math.fround(3.402823e38)
          )
            expect(actual).toBe(expected);
          else
            expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
              2e-6 + 2e-5 * Math.abs(expected)
            );
        }
      if (empty) expect(cpu['status'].every(status => status === 3)).toBe(true);
      else if (mode === 'environmental' || mode === 'knndm')
        expect(Array.from(cpu['folds'])).toEqual(fixture.expectedFolds);
      else if (mode === 'nndm')
        expect(Array.from(cpu['counts'])).toEqual([0, 1, 3, 0, 1, 3, 0, 1, 3, 1, 1, 2]);
      else expect(new Set(cpu['folds']).size).toBe(props.foldCount);
    });
  }

it('keeps the last GPU-supported candidate index in range and rejects the reported excessive count', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    id: 'last-candidate-comparison',
    rowCount: 2,
    positions: 'positions',
    blocks: 'blocks',
    foldCount: 2,
    design: {
      mode: 'knndm',
      candidateCount: 64,
      candidates: 'candidates',
      predictionPositions: 'predictionPositions',
      predictionRowCount: 1,
      diagnostics: 'diagnostics'
    },
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  const candidates = new Uint32Array(128).fill(0xffffffff);
  candidates.set([0, 1], 126);
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 1, 0])},
    blocks: {format: 'uint32' as const, values: new Uint32Array(2)},
    predictionPositions: {format: 'float32x2' as const, values: new Float32Array([0, 0])},
    candidates: {format: 'uint32' as const, values: candidates}
  };
  const outputs = [...Object.values(props.output), 'diagnostics'];
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUSpatialCrossValidationOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUSpatialCrossValidationOperation(props)],
    outputs
  });
  for (const output of outputs) expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
  expect(Array.from(cpu['diagnostics'])).toEqual([0, 64, 1, 63]);
  // The reported 16,777,220-candidate input is outside the GPU constructor's 64-candidate
  // domain. Exercise that rejection without uploading a vector larger than a core binding.
  const rejectedProps: SpatialCrossValidationOperationProps = {
    ...props,
    design: {
      mode: 'knndm',
      candidateCount: 16777220,
      candidates: 'candidates',
      predictionPositions: 'predictionPositions',
      predictionRowCount: 1,
      diagnostics: 'diagnostics'
    }
  };
  expect(() => new CPUSpatialCrossValidationOperation(rejectedProps)).toThrow(/candidateCount/);
  expect(() =>
    constructSamplingContributor(new GPUSpatialCrossValidationOperation(rejectedProps), {
      positions: {format: 'float32x2', length: 2},
      blocks: {format: 'uint32', length: 2},
      predictionPositions: {format: 'float32x2', length: 1},
      candidates: {format: 'uint32', length: 33554440}
    })
  ).toThrow(/candidateCount/);
});

it('compares generated kNNDM restarts when candidate one beats candidate zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = fixtures.generated;
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array(fixture.positions)},
    blocks: {format: 'uint32' as const, values: new Uint32Array(6)},
    predictionPositions: {
      format: 'float32x2' as const,
      values: new Float32Array(fixture.predictionPositions)
    }
  };
  for (const candidateCount of [1, 3]) {
    const props: SpatialCrossValidationOperationProps = {
      id: 'competing-restarts',
      rowCount: 6,
      positions: 'positions',
      blocks: 'blocks',
      foldCount: 2,
      seed: 1,
      bufferDistance: 0.6,
      design: {
        mode: 'knndm',
        candidateCount,
        predictionPositions: 'predictionPositions',
        predictionRowCount: 2,
        diagnostics: 'diagnostics'
      },
      output: {
        folds: 'folds',
        training: 'training',
        testing: 'testing',
        counts: 'counts',
        status: 'status'
      }
    };
    const outputs = [...Object.values(props.output), 'diagnostics'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSpatialCrossValidationOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUSpatialCrossValidationOperation(props)],
      outputs
    });
    for (const output of Object.values(props.output))
      expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
    const selected = candidateCount === 1 ? 0 : 1;
    expect(Array.from(cpu['folds'])).toEqual(fixture.partitions[selected]);
    for (const rows of [gpu, cpu]) {
      expect(rows['diagnostics'][0]).toBe(0);
      expect(rows['diagnostics'][1]).toBe(candidateCount);
      expect(rows['diagnostics'][3]).toBe(selected);
      // Positive Wasserstein objectives: 2e-5 relative, with no absolute floor.
      expect(Math.abs(rows['diagnostics'][2] - fixture.objectives[selected])).toBeLessThanOrEqual(
        2e-5 * fixture.objectives[selected]
      );
    }
  }
});

it('compares two NNDM submissions sharing a diagnostics buffer, including unwritten radii on failure', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: SpatialCrossValidationOperationProps = {
    id: 'reused-nndm-diagnostics',
    rowCount: 4,
    positions: 'positions',
    blocks: 'blocks',
    mask: 'mask',
    foldCount: 4,
    design: {
      mode: 'nndm',
      maximumDistance: 4,
      predictionPositions: 'predictionPositions',
      predictionRowCount: 2,
      diagnostics: 'diagnostics'
    },
    output: {
      folds: 'folds',
      training: 'training',
      testing: 'testing',
      counts: 'counts',
      status: 'status'
    }
  };
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 1, 0, 3, 0, 7, 0])},
    blocks: {format: 'uint32' as const, values: new Uint32Array(4)},
    mask: {format: 'uint32' as const, values: new Uint32Array(4).fill(1)},
    predictionPositions: {format: 'float32x2' as const, values: new Float32Array([20, 0, 25, 0])}
  };
  // runOperationGraph destroys its graph after one run. Bind the GPU operation once to
  // caller-owned buffers here so both submissions really reuse the same diagnostics view.
  const graph = new GPUCommandGraph(device);
  const operation = new GPUSpatialCrossValidationOperation(props);
  const inputBuffers = Object.fromEntries(
    Object.entries(inputs).map(([port, input]) => [port, createInputBuffer(device, input.values)])
  );
  const outputBuffers = Object.fromEntries(
    Object.entries(operation.outputs).map(([port, output]) => [
      port,
      createOutputBuffer(device, output.length)
    ])
  );
  try {
    const views = Object.fromEntries([
      ...Object.entries(inputs).map(([port, input]) => [
        port,
        importGraphBuffer(
          graph,
          port,
          inputBuffers[port],
          input.format,
          input.values.length / (input.format === 'float32x2' ? 2 : 1)
        )
      ]),
      ...Object.entries(operation.outputs).map(([port, output]) => [
        port,
        importGraphBuffer(graph, port, outputBuffers[port], output.format, output.length)
      ])
    ]);
    graph.add(operation.createContributor(port => views[port] as never));
    const compiled = graph.compile();
    try {
      let precedingRadii: number[] = [];
      for (const empty of [false, true]) {
        inputs.mask.values.fill(empty ? 0 : 1);
        inputBuffers['mask'].write(inputs.mask.values);
        submitGraph(device, compiled, undefined);
        const gpu: Record<string, number[]> = {};
        for (const [port, buffer] of Object.entries(outputBuffers))
          gpu[port] = port === 'diagnostics' ? await readFloat32(buffer) : await readUint32(buffer);
        const cpu = await runOperationGraph(device, {
          inputs,
          operations: [new CPUSpatialCrossValidationOperation(props)],
          outputs: [...Object.values(props.output), 'diagnostics']
        });
        for (const output of Object.values(props.output))
          expect(gpu[output]).toEqual(Array.from(cpu[output]));
        for (let index = 0; index < 4; index++) {
          const expected = cpu['diagnostics'][index];
          if (Number.isNaN(expected)) expect(Number.isNaN(gpu['diagnostics'][index])).toBe(true);
          else expect(gpu['diagnostics'][index]).toBe(expected);
        }
        if (empty) {
          expect(Array.from(cpu['diagnostics'])).toEqual([1, 0, NaN, NaN, 0, 0, 0, 0]);
          // Kernel hypothesis to confirm on-device: an early return leaves these four words
          // unchanged. This is a documented difference from the stateless CPU result.
          expect(gpu['diagnostics'].slice(4)).toEqual(precedingRadii);
        } else {
          precedingRadii = gpu['diagnostics'].slice(4);
          const expectedRadii = [7, 6, Math.fround(3.402823e38), 6];
          expect(Array.from(cpu['diagnostics'].subarray(4))).toEqual(expectedRadii);
          for (const [index, expected] of expectedRadii.entries()) {
            const actual = precedingRadii[index];
            // The sentinel is a copied literal. Finite radii come from WGSL distance(),
            // which need not return an exact integer even on this axis-aligned frame.
            // Apply the documented 2e-5 relative diagnostic tolerance to those positive
            // distances only; the next submission must retain the actual GPU words exactly.
            if (expected === Math.fround(3.402823e38)) expect(actual).toBe(expected);
            else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-5 * expected);
          }
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    operation.destroy();
    for (const buffer of [...Object.values(inputBuffers), ...Object.values(outputBuffers)])
      buffer.destroy();
  }
});
