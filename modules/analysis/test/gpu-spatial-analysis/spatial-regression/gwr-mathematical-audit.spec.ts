// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUGeographicallyWeightedRegression,
  type GPUGeographicallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createGWRAuditScene, GWR_AUDIT_CASES} from './gwr-mathematical-audit-fixtures';

for (const fixture of GWR_AUDIT_CASES) {
  for (const indexed of [false, true]) {
    it(`matches every independent GWR port: ${fixture.kernel}/${fixture.bandwidthMode}/shift=${fixture.transformed}/golden=${'golden' in fixture}/refined=${'refinementPassCount' in fixture}/grid=${indexed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        name: string,
        values: Float32Array | Uint32Array,
        format: Format,
        length = values.length
      ) => {
        const buffer = createInputBuffer(device, values);
        buffers.push(buffer);
        return importGraphBuffer(graph, name, buffer, format, length);
      };
      const expected: Record<string, readonly number[]> = {...fixture.expected};
      const hasIsland =
        indexed && fixture.kernel === 'bisquare' && fixture.bandwidthMode === 'fixed';
      if (hasIsland) {
        // Unmask the distant ninth row. Its local design contains only itself (rank 1 < 3),
        // while it has exactly zero weight in every other fit. Those eight fits and their
        // scores remain the NumPy literals, with one singular row and nine included rows.
        expected['localStatus'] = [...fixture.expected.localStatus.slice(0, 8), 1];
        expected['bandwidthSingularRowCounts'] = [1, 1, 0];
        const summary = [...fixture.expected.summary] as number[];
        summary[6] = 1;
        summary[11] = 9;
        expected['summary'] = summary;
        const table = [...fixture.expected.candidateTable] as number[];
        for (const base of [0, 7]) {
          table[base + 2] = 1;
          table[base + 6] = 9;
        }
        expected['candidateTable'] = table;
      }
      // A cutoff of 8 discards no fixture neighbor (all d/h < 3), but must set approximation.
      if (indexed && fixture.kernel === 'gaussian') expected['approximation'] = [1];
      const integerPorts = new Set([
        'localStatus',
        'significant',
        'bandwidthSingularRowCounts',
        'approximation'
      ]);
      const outputBuffers: Record<string, Buffer> = {};
      const outputViews: Record<string, GraphDataView> = {};
      for (const [name, values] of Object.entries(expected)) {
        const buffer = createOutputBuffer(device, values.length);
        buffers.push(buffer);
        outputBuffers[name] = buffer;
        outputViews[name] = importGraphBuffer(
          graph,
          name,
          buffer,
          integerPorts.has(name) ? 'uint32' : 'float32',
          values.length
        );
      }
      const scene = createGWRAuditScene(fixture.transformed);
      if (hasIsland) scene.mask[8] = 1;
      const props: GPUGeographicallyWeightedRegressionProps = {
        positions: input('positions', scene.positions, 'float32x2', 9),
        predictors: input('predictors', scene.predictors, 'float32'),
        predictorCount: 2,
        response: input('response', scene.response, 'float32'),
        mask: input('mask', scene.mask, 'uint32'),
        parameters: input(
          'parameters',
          getGPUGeographicallyWeightedRegressionParameterValues(
            {
              kernel: fixture.kernel,
              bandwidthMode: fixture.bandwidthMode,
              bandwidths: fixture.bandwidths
            },
            3
          ),
          'float32'
        ),
        maximumBandwidthCount: 3,
        refinement: 'refinementPassCount' in fixture ? fixture.refinementPassCount : false,
        bandwidthSearch: 'golden' in fixture ? 'golden-section' : 'ladder',
        maximumSearchIterations: 1,
        indexGridSize: indexed ? [2, 2] : false,
        gaussianCutoff: indexed && fixture.kernel === 'gaussian' ? 8 : undefined,
        output: {
          ...outputViews,
          coefficients: outputViews['coefficients'] as GraphDataView<'float32'>,
          status: {approximation: outputViews['approximation'] as GraphDataView<'uint32'>}
        }
      };
      const contributor = new GPUGeographicallyWeightedRegression(props);
      // This used to change shader PREDICTOR_COUNT after shape validation (and redirect the output).
      props.predictorCount = 1;
      props.output.coefficients = outputViews['fitted'] as GraphDataView<'float32'>;
      graph.add(contributor);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled);
        for (const [port, values] of Object.entries(expected)) {
          const actual = integerPorts.has(port)
            ? await readUint32(outputBuffers[port])
            : await readFloat32(outputBuffers[port]);
          for (let index = 0; index < values.length; index++) {
            const label = `${port}[${index}]`;
            if (Number.isNaN(values[index])) expect(actual[index], label).toBeNaN();
            else {
              expect(Number.isFinite(actual[index]), label).toBe(true);
              // Small, well-conditioned float32 solves and reductions; critical t inversion and
              // adaptive bisquare tails allow 0.02% relative error, with a 0.0002 absolute floor.
              const tolerance = integerPorts.has(port)
                ? 0
                : 2e-4 * Math.max(1, Math.abs(values[index]));
              expect(Math.abs(actual[index] - values[index]), label).toBeLessThanOrEqual(tolerance);
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
