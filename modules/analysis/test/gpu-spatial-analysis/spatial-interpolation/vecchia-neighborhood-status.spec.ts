// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  expectVecchiaMarginalTotals,
  expectVecchiaSuppressedTotals
} from './vecchia-marginal-expectations';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GPUVecchiaLogLikelihood} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-log-likelihood';
import {
  GPUVecchiaSpaceTimeLogLikelihood,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-space-time-log-likelihood';
import {GPUVecchiaFisherScoring} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-vecchia-fisher-scoring';
import {getGPUVecchiaLogLikelihoodParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/vecchia-parameters';

for (const kind of ['spatial', 'space-time', 'fisher'] as const) {
  it(`CPL-01 ${kind} suppresses candidate loss and recovers on the same compiled graph`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: ReturnType<typeof createInputBuffer>[] = [];
    const input = <Format extends GPUVectorFormat>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format,
      count: number
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, count)};
    };
    const output = <Format extends GPUVectorFormat>(
      name: string,
      format: Format,
      count: number,
      components = 1
    ) => {
      const buffer = createOutputBuffer(device, count * components);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, count)};
    };
    const covariance = {model: 'exponential' as const, sill: 1, range: 3, nugget: 0.1};
    const parameters =
      kind === 'space-time'
        ? getGPUVecchiaSpaceTimeLogLikelihoodParameterValues({
            type: 'metric',
            joint: covariance,
            anisotropy: 1
          })
        : getGPUVecchiaLogLikelihoodParameterValues(covariance);
    const loss = input('candidate-loss', Uint32Array.of(0), 'uint32', 1);
    const evidence = output('loss-evidence', 'uint32', 3);
    let evidenceReads = 0;
    // B2: validation and publication must use the same first captured destination.
    const captureEvidence = () => (++evidenceReads === 1 ? evidence.view : undefined);
    const totals = output('totals', 'float32', kind === 'fisher' ? 20 : 2);
    const status = output('status', 'uint32', 3);
    const siteTerms = output('terms', 'float32', kind === 'fisher' ? 20 : 2);
    const siteStatus = output('sites', 'uint32', 2);
    const common = {
      positions: input('positions', Float32Array.of(0, 0, 1, 0), 'float32x2', 2).view,
      observations: input('observations', Float32Array.of(1, 0, 0, 1, 1, 0, 0, 1), 'float32x4', 2)
        .view,
      neighborhoodOffsets: input('offsets', Uint32Array.of(0, 0, 0), 'uint32', 3).view,
      neighborhoods: input('neighbors', Uint32Array.of(0), 'uint32', 1).view,
      neighborCount: 1,
      neighborhoodStatus: {candidateOverflow: loss.view},
      parameters: input('parameters', parameters, 'float32', parameters.length).view
    };
    const likelihoodOutput = {
      conditionals: output('conditionals', 'float32x4', 2, 4).view,
      conditionalStatus: output('conditional-status', 'uint32', 2).view,
      siteTerms: siteTerms.view,
      siteStatus: siteStatus.view,
      logLikelihood: totals.view,
      status: status.view,
      get neighborhoodStatus() {
        return captureEvidence();
      }
    };
    graph.add(
      kind === 'fisher'
        ? new GPUVecchiaFisherScoring({
            ...common,
            output: {
              siteTerms: siteTerms.view,
              siteStatus: siteStatus.view,
              totals: totals.view,
              status: status.view,
              get neighborhoodStatus() {
                return captureEvidence();
              }
            }
          })
        : kind === 'space-time'
          ? new GPUVecchiaSpaceTimeLogLikelihood({
              ...common,
              times: input('times', new Float32Array(2), 'float32', 2).view,
              temporalContext: {
                unit: 'second',
                epoch: 0n,
                encoding: 'float32-relative',
                axis: {kind: 'irregular'}
              },
              output: likelihoodOutput
            })
          : new GPUVecchiaLogLikelihood({...common, output: likelihoodOutput})
    );
    expect(evidenceReads).toBe(1);
    const compiled = graph.compile();
    try {
      for (const flag of [0, 1, 0]) {
        loss.buffer.write(Uint32Array.of(flag));
        submitGraph(device, compiled, undefined);
        const values = await readFloat32(totals.buffer, kind === 'fisher' ? 20 : 2);
        expect(await readUint32(status.buffer, 3)).toEqual([0, 0, 2]);
        expect(await readUint32(evidence.buffer, 3)).toEqual([1 - flag, 0, flag]);
        if (flag) expectVecchiaSuppressedTotals(values, kind === 'fisher');
        else expectVecchiaMarginalTotals(values, kind === 'fisher');
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
