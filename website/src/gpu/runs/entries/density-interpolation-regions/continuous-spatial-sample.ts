// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUContinuousSpatialSample lays a randomly shifted square lattice over an area frame (here a
// connected set of Chicago Census tracts), keeping the lattice sites that fall inside the frame.

import {GPUContinuousSpatialSample} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {parameters} from './continuous-spatial-sample.parameters';

const CAPACITY = 4000;
// The spacings the pointer can settle on; the default is the middle one.
const SPACINGS = [0.03, 0.045, 0.06, 0.09, 0.12] as const;

export default defineOperationRun({
  parameters,
  caption: 'Random-shift lattice sites inside the Chicago tract boundaries.',
  createInputs: ({world}) => {
    const tracts = createTracts(() => 0.5, world, 6, 0);
    const frame = new Float32Array(tracts.triangleIndices.length * 2);
    tracts.triangleIndices.forEach((vertex, index) => {
      frame[index * 2] = tracts.vertices[vertex * 2];
      frame[index * 2 + 1] = tracts.vertices[vertex * 2 + 1];
    });
    let [minX, minY, maxX, maxY] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let index = 0; index < frame.length; index += 2) {
      minX = Math.min(minX, frame[index]);
      maxX = Math.max(maxX, frame[index]);
      minY = Math.min(minY, frame[index + 1]);
      maxY = Math.max(maxY, frame[index + 1]);
    }
    return {
      frame,
      outline: tracts.outline,
      bounds: [minX - 0.01, minY - 0.01, maxX + 0.01, maxY + 0.01] as const
    };
  },
  build(builder, {frame, outline, bounds}, values) {
    // Unwritten slots stay NaN so only the sites the operation wrote are drawn.
    const sites = builder.upload('sites', 'float32x2', new Float32Array(CAPACITY * 2).fill(NaN));
    builder.graph.add(
      new GPUContinuousSpatialSample({
        frame: {type: 'area', positions: builder.upload('frame', 'float32x2', frame).view, bounds},
        spacing: values.spacing,
        seed: values.seed,
        capacity: CAPACITY,
        output: {
          positions: sites.view,
          designWeights: builder.allocate('design-weights', 'float32', CAPACITY).view,
          provenance: builder.allocate('provenance', 'uint32x4', CAPACITY).view,
          status: builder.allocate('status', 'uint32x4', 1).view
        }
      })
    );
    return {
      outputs: {
        sites,
        outline: builder.upload('outline', 'float32x4', outline),
        outlineCount: outline.length / 4
      }
    };
  },
  buildCache: 5,
  rebuildOn: ['spacing', 'seed'],
  // The pointer chooses the lattice spacing: the square shows one lattice cell, and the sites
  // are rebuilt once the pointer rests on a spacing.
  interaction: {
    kind: 'reach',
    hint: 'choose the spacing',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs}) {
      const spacing =
        SPACINGS[Math.min(SPACINGS.length - 1, Math.floor(pointer.unitX * SPACINGS.length))];
      const [minimumX, minimumY, maximumX, maximumY] = inputs.bounds;
      const corner: [number, number] = [(minimumX + maximumX) / 2, (minimumY + maximumY) / 2];
      return {
        settle: {spacing},
        glyphs: [
          {
            shape: 'box',
            minimum: corner,
            maximum: [corner[0] + spacing, corner[1] + spacing]
          }
        ],
        readout: `spacing ${spacing}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: outputs.outlineCount,
      radius: 1.6,
      color: [...theme.muted, 0.9]
    },
    // Ink ring under an accent fill: legible on the dark and the light background alike.
    {layer: 'dots', positions: outputs.sites, radius: 6.5, color: [...theme.ink, 1]},
    {layer: 'dots', positions: outputs.sites, radius: 4.5, color: [...theme.categories[1], 1]}
  ]
});
