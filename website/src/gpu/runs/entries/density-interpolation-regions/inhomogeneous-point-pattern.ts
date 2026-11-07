// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUInhomogeneousPointPattern estimates the local intensity at every event (dots, colored by the
// `intensity` output) and the intensity-reweighted K function. The inset plots K(r) against the
// complete-spatial-randomness curve pi r^2 (grey).

import {
  GPUInhomogeneousPointPattern,
  GPUStudyWindowPreparation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './inhomogeneous-point-pattern.parameters';

const RADII = Float32Array.from({length: 24}, (_, index) => 0.01 + index * 0.01);
const INSET = [0.6, 0.06, 0.97, 0.42] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Event intensity (color) and the K function (inset) against complete spatial randomness.',
  createInputs: ({random, world}) => ({
    world,
    events: createClusteredPoints(random, world, 260, 3, 0.25)
  }),
  build(builder, {world, events}, values) {
    const window = new GPUStudyWindowPreparation({
      positions: builder.upload(
        'window-positions',
        'float32x2',
        Float32Array.of(0, 0, world.width, 0, world.width, 1, 0, 1, 0, 0)
      ).view,
      ringOffsets: builder.upload('window-rings', 'uint32', Uint32Array.of(0, 5)).view,
      polygonOffsets: builder.upload('window-polygons', 'uint32', Uint32Array.of(0, 1)).view,
      featureOffsets: builder.upload('window-features', 'uint32', Uint32Array.of(0, 1)).view,
      samplingGrid: [32, 32]
    });
    builder.graph.add(window);
    const positions = builder.upload('events', 'float32x2', events.positions);
    const intensity = builder.allocate('intensity', 'float32', events.count);
    const k = builder.allocate('k', 'float32', RADII.length);
    builder.graph.add(
      new GPUInhomogeneousPointPattern({
        positions: positions.view,
        studyWindow: window,
        bandwidth: values.bandwidth,
        edgeCorrection: values.edgeCorrection === 0 ? 'border' : 'translation',
        selfContribution: values.selfContribution === 0 ? 'include' : 'exclude',
        radii: builder.upload('radii', 'float32', RADII).view,
        output: {
          intensity: intensity.view,
          k: k.view,
          pairCorrelation: builder.allocate('pair-correlation', 'float32', RADII.length).view,
          status: builder.allocate('status', 'uint32', RADII.length).view
        }
      })
    );
    const random = Float32Array.from(RADII, radius => Math.PI * radius * radius);
    return {outputs: {positions, intensity, k, random: builder.upload('csr', 'float32', random)}};
  },
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.intensity, radius: 4},
    {
      layer: 'chart',
      kind: 'line',
      context: true,
      values: outputs.random,
      range: [0, 0.2],
      rectangle: INSET,
      color: [0.7, 0.7, 0.7, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.k,
      range: [0, 0.2],
      rectangle: INSET,
      color: [1, 0.85, 0.3, 1]
    }
  ]
});
