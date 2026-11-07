// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUInhomogeneousPointPattern} from '../../../src/gpu-spatial-analysis/point-process/gpu-inhomogeneous-point-pattern';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('translation allocation and work budgets reject insufficient or unsupported capacities', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `capacity-${serial++}`, format, length);
  const studyWindow = new GPUStudyWindowPreparation({
    positions: view('float32x2', 5),
    ringOffsets: view('uint32', 2),
    polygonOffsets: view('uint32', 2),
    featureOffsets: view('uint32', 2)
  });
  const props = {
    positions: view('float32x2', 3),
    studyWindow,
    suppliedIntensity: view('float32', 3),
    radii: view('float32', 2),
    edgeCorrection: 'translation' as const,
    output: {
      intensity: view('float32', 3),
      k: view('float32', 2),
      pairCorrelation: view('float32', 2),
      status: view('uint32', 2)
    }
  };
  expect(() => new GPUInhomogeneousPointPattern({...props, maximumPairCount: 2})).toThrow();
  expect(() => new GPUInhomogeneousPointPattern({...props, maximumPairCount: 3})).not.toThrow();
  expect(() => new GPUInhomogeneousPointPattern({...props, maximumOverlapEdgePairs: 0})).toThrow();
  expect(
    () => new GPUInhomogeneousPointPattern({...props, maximumOverlapEdgePairs: 1048577})
  ).toThrow();
  expect(
    () =>
      new GPUInhomogeneousPointPattern({
        ...props,
        output: {...props.output, intensity: view('float32', 2)}
      })
  ).toThrow();
});
