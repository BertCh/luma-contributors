// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  computeSpaceTimeHawkesOnCPU,
  type GPUSpaceTimeHawkes
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import * as sessions from '../src/analysis/catalog-gpu-session';
import {runSpacetimeEvents} from '../src/analysis/catalog-spacetime-events';
import {SPACETIME_TEST_PARAMETERS, SPACETIME_TEST_POINTS} from './catalog-spacetime-fixture';

it('publishes the fitted one-update Hawkes likelihood beside the Kepler estimates', async () => {
  const fixture = computeSpaceTimeHawkesOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 1),
      times: Float32Array.of(2, 2),
      parameters: Float32Array.of(1, 10, 10, 10, 1e-8, 1, 0.5, 1, 1)
    },
    {maximumIterations: 1}
  );
  expect(fixture.fittedLogLikelihood[0]).toBeCloseTo(-5.218876, 5);
  expect(fixture.logLikelihood[0]).toBeCloseTo(-10.999664, 5);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  let contributor: GPUSpaceTimeHawkes;
  const fakeSession = {
    graph: {
      add: (producer: GPUSpaceTimeHawkes) => {
        contributor = producer;
      }
    },
    input: (_name: string, values: Float32Array, format: 'float32' | 'float32x2') =>
      createTransientView(
        graph,
        `input-${serial++}`,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      ),
    output: (_name: string, format: 'float32' | 'uint32', length: number) => ({
      view: createTransientView(graph, `output-${serial++}`, format, length)
    }),
    run: () => {},
    readScalar: async () => 0,
    readFloat32: async (output: {view: GraphDataView}) => {
      if (output.view === contributor.props.output.estimates) return fixture.estimates;
      if (output.view === contributor.props.output.backgroundProbability)
        return fixture.backgroundProbability;
      if (output.view === contributor.props.output.fittedLogLikelihood)
        return fixture.fittedLogLikelihood;
      if (output.view === contributor.props.output.logLikelihood) return fixture.logLikelihood;
      throw new Error('Unexpected Hawkes catalog output');
    }
  };
  const spy = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockImplementation(async (_device, _id, run) =>
      run(fakeSession as unknown as sessions.CatalogGpuSession)
    );
  try {
    const result = await runSpacetimeEvents(
      'hawkes',
      {device},
      {points: SPACETIME_TEST_POINTS},
      {...SPACETIME_TEST_PARAMETERS, maximumIterations: 1}
    );
    expect(result.summary['fittedLogLikelihood']).toBe(fixture.fittedLogLikelihood[0]);
    expect(result.summary['backgroundRate']).toBe(fixture.estimates[0]);
    expect(result.summary['logLikelihood']).toBeUndefined();
    const catalog = readFileSync('website/src/catalog/entries/space-time-events.ts', 'utf8');
    expect(catalog.includes("name: 'fittedLogLikelihood'")).toBe(true);
    expect(catalog.includes('entering the last update')).toBe(true);
  } finally {
    spy.mockRestore();
  }
});
