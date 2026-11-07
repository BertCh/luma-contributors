// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUStudyWindowPreparation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';
import {runExample as runInhomogeneous} from './inhomogeneous-point-pattern.example';
import {runExample as runEnvelope} from './point-pattern-envelope.example';
import {runExample as runSample} from './spatial-balanced-sample.example';
import {runExample as runOPTICS} from './optics.example';

export async function buildPointProcessScenario(
  element: string,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Promise<Record<string, S2CaptureSpec>> {
  const captures: Record<string, S2CaptureSpec> = {};
  const capture = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    view: GraphDataView<Format>,
    role: 'input' | 'output'
  ) => {
    captures[name] = {view, role, keySpace: 'entry'};
    return view;
  };
  const floats = (name: string, values: number[]) =>
    capture(name, uploads.view(name, Float32Array.from(values), 'float32'), 'input');
  const integers = (name: string, values: number[]) =>
    capture(name, uploads.view(name, Uint32Array.from(values), 'uint32'), 'input');
  const positions = (values: number[]) =>
    capture(
      'positions',
      uploads.view('positions', Float32Array.from(values), 'float32x2'),
      'input'
    );
  const output = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) =>
    capture(
      name,
      uploads.view(
        name,
        format === 'uint32' ? new Uint32Array(length) : new Float32Array(length),
        format
      ),
      'output'
    );
  if (element === 'GPUInhomogeneousPointPattern') {
    const studyWindow = uploads.own(
      new GPUStudyWindowPreparation({
        positions: uploads.view(
          'window-positions',
          new Float32Array([0, 0, 10, 0, 10, 10, 0, 10, 0, 0]),
          'float32x2'
        ),
        ringOffsets: integers('ringOffsets', [0, 5]),
        polygonOffsets: integers('polygonOffsets', [0, 1]),
        featureOffsets: integers('featureOffsets', [0, 1]),
        samplingGrid: [4, 4]
      })
    );
    graph.add(studyWindow);
    await runInhomogeneous(graph, {
      studyWindow,
      positions: positions([4, 5, 5, 5, 6, 5]),
      suppliedIntensity: floats('suppliedIntensity', [0.5, 1, 2]),
      radii: floats('radii', [1, 2]),
      output: {
        intensity: output('intensity', 'float32', 3),
        k: output('k', 'float32', 2),
        pairCorrelation: output('pairCorrelation', 'float32', 2),
        status: output('status', 'uint32', 2)
      }
    });
  } else if (element === 'GPUPointPatternEnvelope') {
    await runEnvelope(graph, {
      observed: floats('observed', [2, 3]),
      simulations: floats('simulations', [1, 1, 2, 2, 3, 3]),
      reference: floats('reference', [0, 0]),
      integrationWeights: floats('integrationWeights', [1, 1]),
      simulationCount: 3,
      statistic: 'integrated-squared-deviation',
      nullModel:
        'Three supplied illustrative exchangeable curves; no calibrated scientific null is asserted',
      fittedParameters: 'none',
      output: {
        lower: output('lower', 'float32', 2),
        upper: output('upper', 'float32', 2),
        statistics: output('statistics', 'float32', 4),
        status: output('status', 'uint32', 1)
      }
    });
  } else if (element === 'GPUSpatialBalancedSample') {
    await runSample(graph, {
      positions: positions([0, 0, 1, 0, 0, 1, 1, 1]),
      frameIds: integers('frameIds', [10, 11, 12, 13]),
      bounds: [0, 0, 1, 1],
      seed: 7,
      allocations: [{stratumId: 0, sampleCount: 2}],
      output: {
        frameIds: output('selectedFrameIds', 'uint32', 2),
        inclusionProbabilities: output('inclusionProbabilities', 'float32', 2),
        designWeights: output('designWeights', 'float32', 2),
        status: output('status', 'uint32', 2)
      }
    });
  } else if (element === 'GPUOPTICS') {
    await runOPTICS(graph, {
      positions: positions([0, 0, 1, 0, 4, 0, 5, 0]),
      minimumSamples: 2,
      maximumDistance: 2,
      extractionDistance: 1.5,
      output: {
        ordering: output('ordering', 'uint32', 4),
        coreDistances: output('coreDistances', 'float32', 4),
        reachability: output('reachability', 'float32', 4),
        predecessors: output('predecessors', 'uint32', 4),
        labels: output('labels', 'uint32', 4),
        status: output('status', 'uint32', 3)
      }
    });
  } else {
    throw new Error(`Unknown point-process scenario ${element}`);
  }
  return captures;
}
