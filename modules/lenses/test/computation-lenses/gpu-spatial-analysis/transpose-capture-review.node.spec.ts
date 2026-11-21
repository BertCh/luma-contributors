// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {getLensScenario} from '../../../src/computation-lenses/index';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-transpose.example';
import {runExample as runContiguity} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/contiguity-weights.example';
import {runExample as runSummary} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-summary.example';
import {validateSpatialWeightsCaptures} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {getTractWeightsCaptureViews} from '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {declareLensScenario} from '../scenario-declaration.test-utils';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-weights-transpose.lens';

test.each([
  ['contiguity', runContiguity],
  ['summary', runSummary]
] as const)('R3 %s captures the actual distinct candidate-loss view', async (_name, run) => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const execution = await run(device, {
      graph,
      capacity: 4,
      pairCapacity: 1,
      tracts: {
        kind: 'polygons',
        positions: createTransientView(graph, 'positions', 'float32x2', 12),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 4),
        polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 4),
        featureOffsets: createTransientView(graph, 'features', 'uint32', 4)
      },
      values: createTransientView(graph, 'values', 'float32', 3),
      searchParameters: createTransientView(graph, 'search', 'float32', 8),
      colorParameters: createTransientView(graph, 'colors', 'uint32', 2)
    });
    expect(execution.weights.status?.candidateOverflow).toBeDefined();
    expect(execution.views.weightsCandidateOverflow).toBe(
      execution.weights.status?.candidateOverflow
    );
    expect(execution.views.weightsCandidateOverflow).not.toBe(execution.views.weightsOverflow);
    // The shared boundary must retain the real producer ports even if a presentation omits them.
    const captured = getTractWeightsCaptureViews({...execution, views: {}});
    expect(captured.weightsOverflow).toBe(execution.weights.status?.overflow);
    expect(captured.weightsCandidateOverflow).toBe(execution.weights.status?.candidateOverflow);
  } finally {
    device.destroy();
  }
});

test('packed transpose example captures inherited loss diagnostics instead of unwritten destination status', async () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const execution = await runExample(device, {
      graph,
      tracts: {
        kind: 'polygons',
        positions: createTransientView(graph, 'positions', 'float32x2', 12),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 4),
        polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 4),
        featureOffsets: createTransientView(graph, 'features', 'uint32', 4)
      },
      values: createTransientView(graph, 'values', 'float32', 3),
      searchParameters: createTransientView(graph, 'search', 'float32', 8),
      colorParameters: createTransientView(graph, 'colors', 'uint32', 2)
    });
    expect(execution.views.weightsOverflow).toBe(execution.views.sourceWeightsOverflow);
    expect(execution.views.weightsOverflow).toBe(execution.weights.status?.overflow);
    expect(execution.views.weightsCandidateOverflow).toBeDefined();
    expect(execution.views.weightsCandidateOverflow).toBe(
      execution.weights.status?.candidateOverflow
    );
  } finally {
    device.destroy();
  }
});

test('transpose scenario retains both loss flags and refuses candidate-only loss before publishing', async () => {
  const declared = await declareLensScenario(
    getLensScenario('gpu-spatial-analysis.spatial-weights-transpose')!
  );
  expect(declared.missingChoices).toEqual([]);
  expect(declared.captureNames).toContain('weightsOverflow');
  expect(declared.captureNames).toContain('weightsCandidateOverflow');
  const captures = declared.captureNames.map(name => ({
    name,
    values: Uint32Array.of(name === 'weightsCandidateOverflow' ? 1 : 0)
  }));
  expect(() => validateSpatialWeightsCaptures(captures)).toThrow(/weightsCandidateOverflow/);
  for (const capture of captures) capture.values.fill(0);
  expect(() => validateSpatialWeightsCaptures(captures)).not.toThrow();
});
