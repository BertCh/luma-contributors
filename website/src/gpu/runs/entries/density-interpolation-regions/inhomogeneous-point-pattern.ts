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
import {findParameter, snapToParameter} from '../../interaction';
import {findNearestSamples} from './support/interpolation';
import {parameters} from './inhomogeneous-point-pattern.parameters';

const RADII = Float32Array.from({length: 24}, (_, index) => 0.01 + index * 0.01);
const INSET = [0.04, 0.08, 0.4, 0.44] as const;
// The bandwidths the pointer can settle on; the default is the middle one.
const BANDWIDTHS = [0.04, 0.1, 0.16, 0.24, 0.4] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Event intensity (color) and the K function (inset) against complete spatial randomness.',
  createInputs: ({random, world}) => {
    const events = createClusteredPoints(random, world, 260, 3, 0.25);
    // The event nearest the middle of the right half anchors the bandwidth ring.
    const [anchor] = findNearestSamples(events.positions, world.width * 0.7, 0.5, 1);
    return {world, events, anchor};
  },
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
  buildCache: 5,
  rebuildOn: ['bandwidth', 'edgeCorrection', 'selfContribution'],
  // The distance from the anchor event to the pointer is the kernel bandwidth; the intensity
  // estimate is rebuilt once the pointer rests on one of five bandwidths.
  interaction: {
    kind: 'reach',
    hint: 'stretch the bandwidth',
    rest: ({inputs, toUnit}) => {
      const {positions} = inputs.events;
      return toUnit([positions[inputs.anchor * 2] + 0.25, positions[inputs.anchor * 2 + 1]]);
    },
    resolve(pointer, {inputs}) {
      const {positions} = inputs.events;
      const at: [number, number] = [positions[inputs.anchor * 2], positions[inputs.anchor * 2 + 1]];
      const distance = Math.hypot(pointer.x - at[0], pointer.y - at[1]);
      // Pointer distance 0 to 0.5 in five bands of 0.1.
      const bandwidth = BANDWIDTHS[Math.min(BANDWIDTHS.length - 1, Math.floor(distance / 0.1))];
      return {
        settle: {bandwidth: snapToParameter(findParameter(parameters, 'bandwidth'), bandwidth)},
        glyphs: [
          {shape: 'ring', at, radius: bandwidth},
          {shape: 'dot', at}
        ],
        readout: `bandwidth ${bandwidth.toFixed(2)}`
      };
    }
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
