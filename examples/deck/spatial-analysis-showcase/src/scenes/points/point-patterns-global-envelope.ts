// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUPointPatternEnvelope} from '@luma-contributors/graph/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';

export type GlobalEnvelopeStatistic =
  | 'maximum-deviation'
  | 'integrated-squared-deviation'
  | 'extreme-rank';

export type GlobalEnvelopeResult = {
  lower: Float32Array;
  upper: Float32Array;
  statistic: number;
  pValue: number;
  pResolution: number;
  simulationCount: number;
  status: number;
  pInterval: [number, number] | null;
};

export type GlobalEnvelopeGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  lower: Buffer;
  upper: Buffer;
  result: GlobalEnvelopeResult | null;
};

let globalEnvelopeBuildCount = 0;

/** Builds the GPU global test over the complete observed and replicate-major curve set. */
export async function buildGlobalEnvelopeGraph(
  device: Device,
  observed: Float32Array,
  simulations: Float32Array,
  radii: Float32Array,
  statistic: GlobalEnvelopeStatistic,
  significanceLevel: number,
  onResult: (result: GlobalEnvelopeResult) => void
): Promise<GlobalEnvelopeGraph> {
  const radiusCount = observed.length;
  const simulationCount = simulations.length / radiusCount;
  const buildId = globalEnvelopeBuildCount++;
  const resources = new SpatialAnalysisResources(
    device,
    `point-patterns-global-envelope-${buildId}`
  );
  try {
    const observedBuffer = resources.createBuffer('observed-l-minus-r', observed);
    const simulationBuffer = resources.createBuffer('simulation-l-minus-r', simulations);
    const referenceBuffer = resources.createBuffer(
      'fixed-zero-reference',
      new Float32Array(radiusCount)
    );
    const step = radii.length > 1 ? radii[1] - radii[0] : 1;
    const integrationBuffer = resources.createBuffer(
      'radius-integration-weights',
      new Float32Array(radiusCount).fill(step)
    );
    const lower = resources.createBuffer('lower-envelope', radiusCount * 4);
    const upper = resources.createBuffer('upper-envelope', radiusCount * 4);
    const statistics = resources.createBuffer('global-statistics', 4 * 4);
    const status = resources.createBuffer('global-status', 4);
    const pValueInterval =
      statistic === 'extreme-rank' ? resources.createBuffer('p-value-interval', 2 * 4) : null;
    const graph = new GPUCommandGraph<void>(device, {
      id: `point-patterns-global-${buildId}-${statistic}`
    });
    const observedView = importGraphBuffer(
      graph,
      'observed-l-minus-r',
      observedBuffer,
      'float32',
      radiusCount
    );
    const simulationsView = importGraphBuffer(
      graph,
      'simulation-l-minus-r',
      simulationBuffer,
      'float32',
      radiusCount * simulationCount
    );
    const referenceView = importGraphBuffer(
      graph,
      'fixed-zero-reference',
      referenceBuffer,
      'float32',
      radiusCount
    );
    const integrationView = importGraphBuffer(
      graph,
      'radius-integration-weights',
      integrationBuffer,
      'float32',
      radiusCount
    );
    const lowerView = importGraphBuffer(graph, 'lower-envelope', lower, 'float32', radiusCount);
    const upperView = importGraphBuffer(graph, 'upper-envelope', upper, 'float32', radiusCount);
    const statisticsView = importGraphBuffer(graph, 'global-statistics', statistics, 'float32', 4);
    const statusView = importGraphBuffer(graph, 'global-status', status, 'uint32', 1);
    const intervalView = pValueInterval
      ? importGraphBuffer(graph, 'p-value-interval', pValueInterval, 'float32', 2)
      : undefined;
    graph.add(
      new GPUPointPatternEnvelope({
        id: 'global-point-pattern-test',
        observed: observedView,
        simulations: simulationsView,
        reference: referenceView,
        integrationWeights: integrationView,
        simulationCount,
        statistic,
        significanceLevel,
        nullModel: 'fixed-count uniform conditional CSR in the selected study window',
        fittedParameters: 'none',
        output: {
          lower: lowerView,
          upper: upperView,
          statistics: statisticsView,
          status: statusView,
          ...(intervalView ? {pValueInterval: intervalView} : {})
        }
      })
    );
    const compiled = resources.track(await graph.compileAsync());
    let graphResult: GlobalEnvelopeGraph | null = null;
    const reader = new SummaryReader(
      resources,
      'global-envelope-result',
      [
        {buffer: lower, size: radiusCount * 4},
        {buffer: upper, size: radiusCount * 4},
        {buffer: statistics, size: 4 * 4},
        {buffer: status, size: 4},
        ...(pValueInterval ? [{buffer: pValueInterval, size: 2 * 4}] : [])
      ],
      bytes => {
        const values = new Float32Array(bytes);
        const lowerOffset = 0;
        const upperOffset = radiusCount;
        const statisticsOffset = radiusCount * 2;
        const statusOffset = statisticsOffset + 4;
        const intervalOffset = statusOffset + 1;
        const statusValue = new Uint32Array(bytes)[statusOffset];
        const result: GlobalEnvelopeResult = {
          lower: values.slice(lowerOffset, upperOffset),
          upper: values.slice(upperOffset, statisticsOffset),
          statistic: values[statisticsOffset],
          pValue: values[statisticsOffset + 1],
          pResolution: values[statisticsOffset + 2],
          simulationCount: values[statisticsOffset + 3],
          status: statusValue,
          pInterval:
            statistic === 'extreme-rank'
              ? [values[intervalOffset], values[intervalOffset + 1]]
              : null
        };
        if (graphResult) {
          graphResult.result = result;
          onResult(result);
        }
      }
    );
    graphResult = {
      resources,
      compiled,
      reader,
      lower,
      upper,
      result: null
    };
    return graphResult;
  } catch (error) {
    resources.destroy();
    throw error;
  }
}
