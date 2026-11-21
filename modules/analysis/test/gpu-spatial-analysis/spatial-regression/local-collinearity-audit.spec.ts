// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import type {GPUChunkedSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  LOCAL_COLLINEARITY_AUDIT_CASES,
  LOCAL_AUDIT_NEIGHBORS,
  LOCAL_AUDIT_WEIGHTS
} from './local-collinearity-audit-fixtures';

for (const fixture of LOCAL_COLLINEARITY_AUDIT_CASES) {
  for (const chunked of [false, true]) {
    it(`matches independent local diagnostics and recovers after incomplete weights: transformed=${fixture.transformed}, chunked=${chunked}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const input = <Format extends 'float32' | 'uint32'>(
        name: string,
        values: number[],
        format: Format
      ) => {
        const buffer = createInputBuffer(
          device,
          format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
        );
        buffers.push(buffer);
        return {buffer, view: importGraphBuffer(graph, name, buffer, format, values.length)};
      };
      const output = <Format extends 'float32' | 'uint32'>(
        name: string,
        format: Format,
        length: number
      ) => {
        const buffer = createOutputBuffer(device, length);
        buffers.push(buffer);
        return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
      };
      const createCSR = (start: number, end: number): GPUSpatialWeights => {
        const neighborhoods = LOCAL_AUDIT_NEIGHBORS.slice(start, end);
        const offsets = [0];
        for (const neighbors of neighborhoods)
          offsets.push(offsets[offsets.length - 1] + neighbors.length);
        return {
          offsets: input(`offsets-${start}`, offsets, 'uint32').view,
          neighbors: input(`neighbors-${start}`, neighborhoods.flat(), 'uint32').view,
          weights: input(
            `weights-${start}`,
            LOCAL_AUDIT_WEIGHTS.slice(start, end).flat(),
            'float32'
          ).view
        };
      };
      const loss = input('loss', [0], 'uint32');
      const weights: GPUSpatialWeights | GPUChunkedSpatialWeights = chunked
        ? {
            rowCount: 9,
            chunks: [
              {rowStart: 0, rowCount: 5, weights: createCSR(0, 5)},
              {
                rowStart: 5,
                rowCount: 4,
                weights: {...createCSR(5, 9), status: {candidateOverflow: loss.view}}
              }
            ]
          }
        : {...createCSR(0, 9), status: {overflow: loss.view}};
      const outputs = {
        varianceInflationFactors: output('inflation', 'float32', 18),
        conditionNumbers: output('condition', 'float32', 9),
        designConditionNumbers: output('designCondition', 'float32', 9),
        varianceDecomposition: output('proportions', 'float32', 27),
        status: output('status', 'uint32', 9)
      };
      const props = {
        weights,
        predictors: input('predictors', Array.from(fixture.predictors), 'float32').view,
        predictorCount: 2,
        mask: input('mask', [1, 1, 1, 1, 1, 1, 1, 1, 0], 'uint32').view,
        output: {
          varianceInflationFactors: outputs.varianceInflationFactors.view,
          conditionNumbers: outputs.conditionNumbers.view,
          designConditionNumbers: outputs.designConditionNumbers.view,
          varianceDecomposition: outputs.varianceDecomposition.view,
          status: outputs.status.view
        }
      };
      const contributor = new GPULocalCollinearity(props);
      props.predictorCount = 1;
      graph.add(contributor);
      const compiled = graph.compile();
      try {
        // Normal -> loss -> normal on one graph proves suppression does not leave stale status.
        for (const incomplete of [false, true, false]) {
          loss.buffer.write(Uint32Array.of(Number(incomplete)));
          submitGraph(device, compiled);
          for (const port of [
            'varianceInflationFactors',
            'conditionNumbers',
            'designConditionNumbers',
            'varianceDecomposition',
            'status'
          ] as const) {
            const actual =
              port === 'status'
                ? await readUint32(outputs[port].buffer)
                : await readFloat32(outputs[port].buffer);
            const expected = fixture[port];
            for (let index = 0; index < expected.length; index++) {
              const value = incomplete ? (port === 'status' ? 1 : NaN) : expected[index];
              if (Number.isNaN(value)) expect(actual[index], `${port}[${index}]`).toBeNaN();
              else {
                expect(Number.isFinite(actual[index]), `${port}[${index}]`).toBe(true);
                // LAPACK float64 SVD versus bounded float32 QR/Jacobi on these small designs.
                const tolerance = port === 'status' ? 0 : 3e-4 * Math.max(1, Math.abs(value));
                expect(Math.abs(actual[index] - value), `${port}[${index}]`).toBeLessThanOrEqual(
                  tolerance
                );
              }
            }
          }
        }
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    });
  }
}
