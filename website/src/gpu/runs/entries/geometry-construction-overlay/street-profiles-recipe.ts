// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addStreetProfilesRecipe measures the street cross-section at samples along a street: the distance
// to the facade on each side, the building heights there and the height-to-width ratio of the
// canyon. Rays (blue) run from every sample to the facades; the dot color is the canyon ratio.
// Samples with no facade within `maximumDistance` on both sides stay muted.

import {addStreetProfilesRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {createCityLayout, getBuildings, sampleStreet} from './shared/urban';
import {parameters} from './street-profiles-recipe.parameters';

const SAMPLE_COUNT = 40;

export default defineOperationRun({
  parameters,
  caption:
    'Street cross-sections: distance to the facade on each side, colored by height-to-width.',
  createInputs: ({world}) => {
    const layout = createCityLayout(world, 24);
    const samples = sampleStreet(layout, SAMPLE_COUNT);
    const normals = Float32Array.from({length: SAMPLE_COUNT * 2}, (_, index) => index % 2);
    return {layout, samples, normals, index: createPointIndex(samples), focus: {row: -1}};
  },
  build(builder, inputs, values) {
    const walls = getBuildings(inputs.layout, 1, values.streetWidth);
    const positions = builder.upload('samples', 'float32x2', inputs.samples);
    const normals = builder.upload('normals', 'float32x2', inputs.normals);
    const segments = builder.upload('walls', 'float32x4', walls.segments);
    const owners = builder.upload('wall-owners', 'uint32', walls.segmentBuildingIds);
    const heights = builder.upload('heights', 'float32', inputs.layout.heights);
    const result = addStreetProfilesRecipe(builder.graph, {
      positions: positions.view,
      normals: normals.view,
      segments: segments.view,
      segmentBuildingIds: owners.view,
      buildingHeights: heights.view,
      maximumDistance: values.maximumDistance,
      sampleSpacing: inputs.samples[2] - inputs.samples[0],
      tolerance: 0.0005
    });
    const rays = builder.allocate('profile-rays', 'float32x4', SAMPLE_COUNT * 2);
    const ratios = builder.allocate('canyon-ratios', 'float32', SAMPLE_COUNT);
    addKernelPass(builder, {
      id: 'profile-publish-pass',
      invocationCount: SAMPLE_COUNT,
      bindings: [
        {name: 'positions', buffer: positions, access: 'read'},
        {name: 'normals', buffer: normals, access: 'read'},
        {name: 'profiles', buffer: getViewBuffer(result.outputs.profiles), access: 'read'},
        {name: 'canyon', buffer: getViewBuffer(result.outputs.canyon), access: 'read'},
        {name: 'validity', buffer: getViewBuffer(result.outputs.validity), access: 'read'},
        {name: 'rays', buffer: rays, access: 'read_write'},
        {name: 'ratios', buffer: ratios, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let origin = positions[positionsOffset + index];
  let normal = normals[normalsOffset + index];
  let profile = profiles[profilesOffset + index];
  let isFinitePair = validity[validityOffset + index] == 3u;
  rays[raysOffset + 2u * index] = select(vec4f(-9.0), vec4f(origin, origin + normal * profile.x), isFinitePair);
  rays[raysOffset + 2u * index + 1u] = select(vec4f(-9.0), vec4f(origin, origin - normal * profile.y), isFinitePair);
  ratios[ratiosOffset + index] = select(1.0e30, canyon[canyonOffset + index].z, isFinitePair);`
    });
    const wallSegments = builder.allocate('wall-segments', 'float32x4', walls.segments.length / 4);
    return {
      outputs: {positions, rays, ratios, wallSegments},
      update: current => {
        const next = getBuildings(inputs.layout, 1, current.streetWidth);
        segments.write(next.segments);
        wallSegments.write(next.segments);
      }
    };
  },
  rebuildOn: ['maximumDistance'],
  // The pointer picks a street sample; its two cross-section rays, facade distances and canyon
  // ratio (the operation's output) are drawn bright.
  interaction: {
    kind: 'probe',
    hint: 'hover along the street',
    rest: ({inputs, toUnit}) => toUnit([inputs.samples[20 * 2], inputs.samples[20 * 2 + 1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, inputs.samples[1], inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const rays = peek(outputs.rays);
      const ratios = peek(outputs.ratios);
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: [inputs.samples[row * 2], inputs.samples[row * 2 + 1]], radius: 0.03},
        {
          shape: 'line',
          from: [inputs.samples[row * 2], inputs.samples[row * 2 + 1] - values.maximumDistance],
          to: [inputs.samples[row * 2], inputs.samples[row * 2 + 1] + values.maximumDistance],
          dashed: true
        }
      ];
      let readout = `sample ${row} · no facade in reach`;
      if (rays && ratios && rays[row * 8] > -8) {
        const left = Math.hypot(
          rays[row * 8 + 2] - rays[row * 8],
          rays[row * 8 + 3] - rays[row * 8 + 1]
        );
        const right = Math.hypot(
          rays[row * 8 + 6] - rays[row * 8 + 4],
          rays[row * 8 + 7] - rays[row * 8 + 5]
        );
        for (const slot of [0, 1]) {
          const base = row * 8 + slot * 4;
          glyphs.push(
            {
              shape: 'line',
              from: [rays[base], rays[base + 1]],
              to: [rays[base + 2], rays[base + 3]]
            },
            {shape: 'dot', at: [rays[base + 2], rays[base + 3]]}
          );
        }
        readout = `left ${left.toFixed(2)} · right ${right.toFixed(2)} · H/W ${ratios[row] > 1e29 ? '-' : ratios[row].toFixed(1)}`;
      }
      return {focus: row, glyphs, readout};
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.wallSegments, radius: 3},
    {layer: 'lines', segments: outputs.rays, color: [0.38, 0.68, 1, 0.3], radius: 1},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.ratios,
      radius: 6.5,
      range: [0, 2]
    }
  ]
});
