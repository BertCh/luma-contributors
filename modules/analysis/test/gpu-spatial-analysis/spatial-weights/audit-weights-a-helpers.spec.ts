// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import {GPUChunkedSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-validation';
import {
  createFloat32ValuesNodes,
  validateFloat32OrUint32Values
} from '../../../src/gpu-spatial-analysis/spatial-weights/float32-values';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {GPUSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transform';
import {GPUChunkedSpatialWeightsTransform} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transform';
import {AUDIT_ALGEBRA_LEFT} from './audit-weights-a-algebra-fixture';
import {AUDIT_TRANSFORM_CASES} from './audit-weights-a-transform-fixture';
import {AUDIT_BUILDER_NAMES, createAuditBuilder} from './audit-weights-a-builders-fixture';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {CPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('both named validators retain construction shape and output names with literal eight-row diagnostics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const Operation of [
    CPUSpatialWeightsValidationOperation,
    GPUSpatialWeightsValidationOperation
  ]) {
    const props = {
      rowCount: 8,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {flags: 'flags', status: {invalidCount: 'invalid'}}
    };
    const operation = new Operation(props);
    props.rowCount = 1;
    props.output.flags = 'renamedFlags';
    props.output.status.invalidCount = 'renamedInvalid';
    const actual = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 3, 4, 5, 6, 6, 8, 9, 9)},
        neighbors: {format: 'uint32', values: Uint32Array.of(0, 1, 1, 2, 9, 3, 2, 1, 0)},
        weights: {format: 'float32', values: Float32Array.of(1, 1, NaN, 0, -1, 1, 1, 1, Infinity)}
      },
      operations: [operation],
      outputs: ['flags', 'invalid']
    });
    // Independent per-row predicates from audit-weights-a-helpers.oracle.py.
    expect(Array.from(actual['flags'])).toEqual([28, 0, 18, 8, 0, 4, 16, 0]);
    expect(Array.from(actual['invalid'])).toEqual([5]);
  }
});

it('audits whole and chunked validation against literal independent flags for every violation bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const offsets = [0, 3, 4, 5, 6, 6, 8, 9, 9];
  const neighbors = [0, 1, 1, 2, 9, 3, 2, 1, 0];
  const values = [1, 1, NaN, 0, -1, 1, 1, 1, Infinity];
  for (const chunkRows of [[8], [2, 3, 3]]) {
    const rig = new WeightsRig(device);
    let rowStart = 0;
    const chunks = chunkRows.map(rowCount => {
      const begin = offsets[rowStart];
      const end = offsets[rowStart + rowCount];
      const chunk = {
        rowStart,
        rowCount,
        weights: rig.uploadWeights(
          {
            offsets: offsets.slice(rowStart, rowStart + rowCount + 1).map(value => value - begin),
            neighbors: neighbors.slice(begin, end),
            weights: values.slice(begin, end),
            distances: []
          },
          1
        )
      };
      rowStart += rowCount;
      return chunk;
    });
    const flags = rig.output('uint32', 8);
    const invalidCount = rig.output('uint32', 1);
    const output = {flags: flags.view, status: {invalidCount: invalidCount.view}};
    try {
      rig.run(
        chunkRows.length === 1
          ? new GPUSpatialWeightsValidation({weights: chunks[0].weights, output})
          : new GPUChunkedSpatialWeightsValidation({weights: {rowCount: 8, chunks}, output})
      );
      expect(await readUint32(flags.buffer, 8)).toEqual([28, 0, 18, 8, 0, 4, 16, 0]);
      expect(await readUint32(invalidCount.buffer, 1)).toEqual([5]);
    } finally {
      rig.destroy();
    }
  }
});

it('audits fuzzy/island/lattice builders and multicolumn lag against independent eight-row literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const name of AUDIT_BUILDER_NAMES) {
    for (const [scale, shift] of [
      [1, 0],
      [8, 1048576]
    ]) {
      for (const nearestSearch of name.startsWith('islands')
        ? (['scan', 'grid'] as const)
        : (['scan'] as const)) {
        const rig = new WeightsRig(device);
        const checks: (() => Promise<void>)[] = [];
        const contributor = createAuditBuilder(
          {
            input: (values, format) =>
              rig.input(values, format, values.length / (format === 'float32x2' ? 2 : 1)),
            output: (port, format, expected) => {
              const output = rig.output(format, expected.length);
              checks.push(async () => {
                const actual =
                  format === 'float32'
                    ? await readFloat32(output.buffer, expected.length)
                    : await readUint32(output.buffer, expected.length);
                if (format === 'uint32') expect(actual, `${name} ${port}`).toEqual(expected);
                else
                  for (const [row, value] of actual.entries()) {
                    // 2e-6 covers f32 division in normalized two-column lag; integer/dyadic ports are exact.
                    expect(Number.isFinite(value), `${name} ${port}`).toBe(true);
                    expect(Math.abs(value - expected[row]), `${name} ${port}`).toBeLessThanOrEqual(
                      2e-6
                    );
                  }
              });
              return output.view;
            }
          },
          name,
          scale,
          shift,
          nearestSearch
        );
        try {
          rig.run(contributor);
          for (const check of checks) await check();
        } finally {
          rig.destroy();
        }
      }
    }
  }
});

it('island attachment reports inherited loss and clears its candidate word on the same compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const inputLoss = rig.input(Uint32Array.of(1), 'uint32', 1);
  const inputCandidateLoss = rig.input(Uint32Array.of(0), 'uint32', 1);
  const output = rig.weightsOutput(1, 1);
  const candidate = rig.output('uint32', 1);
  const overflow = rig.output('uint32', 1);
  const required = rig.output('uint32', 1);
  const attachment = rig.output('uint32', 1);
  const added = rig.output('uint32', 1);
  rig.graph.add(
    new GPUIslandAttachment({
      weights: {
        ...rig.uploadWeights({offsets: [0, 0], neighbors: [], weights: [], distances: []}, 1),
        status: {overflow: inputLoss, candidateOverflow: inputCandidateLoss}
      },
      positions: rig.input(Float32Array.of(0, 0), 'float32x2', 1),
      reciprocal: false,
      nearestSearch: 'scan',
      output: {
        ...output.spatialWeights,
        attachment: attachment.view,
        added: added.view,
        status: {
          overflow: overflow.view,
          requiredCount: required.view,
          candidateOverflow: candidate.view
        }
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const [loss, candidateLoss] of [
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      rig.bufferOf(inputLoss).write(Uint32Array.of(loss));
      rig.bufferOf(inputCandidateLoss).write(Uint32Array.of(candidateLoss));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(candidate.buffer, 1)).toEqual([
        Number(Boolean(loss || candidateLoss))
      ]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(required.buffer, 1)).toEqual([0]);
      expect(await readUint32(attachment.buffer, 1)).toEqual([0xffffffff]);
      expect((await output.read()).offsets).toEqual([0, 0]);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('audits whole/chunked transforms and all kernel bandwidths against direct dense-row formulas', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {options, expected} of AUDIT_TRANSFORM_CASES) {
    for (const chunkRows of [[8], [2, 3, 3]]) {
      const rig = new WeightsRig(device);
      let rowStart = 0;
      const chunks = chunkRows.map(rowCount => {
        const begin = AUDIT_ALGEBRA_LEFT.offsets[rowStart];
        const end = AUDIT_ALGEBRA_LEFT.offsets[rowStart + rowCount];
        const information = {
          rowCount,
          transform: 'raw' as const,
          symmetric: 'unknown' as const,
          provenance: {source: 'audit', criterion: 'knn'}
        };
        const chunk = {
          rowStart,
          rowCount,
          weights: {
            ...rig.uploadWeights(
              {
                offsets: Array.from(
                  AUDIT_ALGEBRA_LEFT.offsets.slice(rowStart, rowStart + rowCount + 1),
                  value => value - begin
                ),
                neighbors: Array.from(AUDIT_ALGEBRA_LEFT.neighbors.slice(begin, end)),
                weights: Array.from(AUDIT_ALGEBRA_LEFT.weights.slice(begin, end)),
                distances: Array.from(AUDIT_ALGEBRA_LEFT.distances.slice(begin, end))
              },
              1
            ),
            information
          }
        };
        rowStart += rowCount;
        return chunk;
      });
      const outputs = chunks.map(chunk => rig.output('float32', chunk.weights.neighbors.length));
      try {
        rig.run(
          chunkRows.length === 1
            ? new GPUSpatialWeightsTransform({
                ...options,
                weights: chunks[0].weights,
                output: outputs[0].view
              })
            : new GPUChunkedSpatialWeightsTransform({
                ...options,
                weights: {rowCount: 8, chunks},
                output: outputs.map(output => output.view)
              })
        );
        const actual: number[] = [];
        for (const [index, chunk] of chunks.entries()) {
          const used =
            AUDIT_ALGEBRA_LEFT.offsets[chunk.rowStart + chunk.rowCount] -
            AUDIT_ALGEBRA_LEFT.offsets[chunk.rowStart];
          actual.push(...(await readFloat32(outputs[index].buffer, used)));
        }
        expect(actual.length).toBe(expected.length);
        for (const [slot, value] of actual.entries()) {
          // 2e-6 covers f32 reductions and adaptive epsilon rounding (1e-7 -> 2^-23).
          expect(Number.isFinite(value)).toBe(true);
          expect(Math.abs(value - expected[slot]), JSON.stringify(options)).toBeLessThanOrEqual(
            2e-6
          );
          // A zeroed adaptive boundary weight must fail even below the absolute tolerance.
          if (options.bandwidth === 'adaptive') expect(value).toBeGreaterThan(0);
        }
      } finally {
        rig.destroy();
      }
    }
  }
});

it('audits uint32-to-float32 rounding at 2^24 and the uint32 ceiling and preserves float32 input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  const input = rig.input(
    Uint32Array.of(0, 1, 16777215, 16777216, 16777217, 16777219, 2147483647, 4294967295),
    'uint32',
    8
  );
  const output = rig.output('float32', 8);
  try {
    expect(validateFloat32OrUint32Values(input, 'audit counts')).toBe(8);
    const conversion = createFloat32ValuesNodes(rig.graph, 'audit-convert', input);
    const identity = createFloat32ValuesNodes(rig.graph, 'audit-identity', conversion.view);
    expect(identity.view).toBe(conversion.view);
    expect(identity.nodes).toEqual([]);
    rig.run({
      getCommandNodes: graph => [
        ...conversion.nodes,
        createWGSLKernelNode(graph, {
          id: 'audit-conversion-copy',
          operation: 'AuditConversion',
          invocationCount: 8,
          bindings: [
            {name: 'input', view: conversion.view, type: 'f32', access: 'read'},
            {name: 'output', view: output.view, type: 'f32', access: 'read_write'}
          ],
          body: 'output[outputOffset + index] = input[inputOffset + index];'
        })
      ]
    });
    // NumPy float32 conversion in audit-weights-a-helpers.oracle.py. Exact binary results.
    expect(await readFloat32(output.buffer, 8)).toEqual([
      0, 1, 16777215, 16777216, 16777216, 16777220, 2147483648, 4294967296
    ]);
  } finally {
    rig.destroy();
  }
});
