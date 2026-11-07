// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addStreetProfilesRecipe measures the street cross-section at samples along a street: the distance
// to the facade on each side, the building heights there and the height-to-width ratio of the
// canyon. Rays (blue) run from every sample to the facades; the dot color is the canyon ratio.
// Samples with no facade within `maximumDistance` on both sides stay muted.

import {addStreetProfilesRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addKernelPass, getViewBuffer} from './shared/kernel';
import {parameters} from './street-profiles-recipe.parameters';

const SLOT_COUNT = 7;
const SAMPLE_COUNT = 40;
const STREET_Y = 0.5;
const DEPTH = 0.32;

export default defineOperationRun({
  parameters,
  caption:
    'Street cross-sections: distance to the facade on each side, colored by height-to-width.',
  createInputs: ({random, world}) => {
    const heights = Float32Array.from({length: SLOT_COUNT * 2}, () => 0.03 + 0.22 * random());
    const slotWidth = (world.width - 0.08) / SLOT_COUNT;
    const getWalls = (streetWidth: number) => {
      const segments: number[] = [];
      const owners: number[] = [];
      for (let building = 0; building < SLOT_COUNT * 2; building++) {
        const slot = building % SLOT_COUNT;
        const isAbove = building >= SLOT_COUNT;
        const x0 = 0.04 + slot * slotWidth + 0.025;
        const x1 = 0.04 + (slot + 1) * slotWidth - 0.025;
        const nearY = STREET_Y + (isAbove ? 1 : -1) * (streetWidth / 2);
        const farY = nearY + (isAbove ? 1 : -1) * DEPTH;
        const y0 = Math.min(nearY, farY);
        const y1 = Math.max(nearY, farY);
        segments.push(x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0);
        owners.push(building, building, building, building);
      }
      return {segments: Float32Array.from(segments), owners: Uint32Array.from(owners)};
    };
    const samples = Float32Array.from({length: SAMPLE_COUNT * 2}, (_, slot) =>
      slot % 2 === 0 ? 0.06 + ((world.width - 0.12) * (slot / 2)) / (SAMPLE_COUNT - 1) : STREET_Y
    );
    const normals = Float32Array.from({length: SAMPLE_COUNT * 2}, (_, slot) =>
      slot % 2 === 0 ? 0 : 1
    );
    return {heights, getWalls, samples, normals};
  },
  build(builder, inputs, values) {
    const walls = inputs.getWalls(values.streetWidth);
    const positions = builder.upload('samples', 'float32x2', inputs.samples);
    const normals = builder.upload('normals', 'float32x2', inputs.normals);
    const segments = builder.upload('walls', 'float32x4', walls.segments);
    const owners = builder.upload('wall-owners', 'uint32', walls.owners);
    const heights = builder.upload('heights', 'float32', inputs.heights);
    const result = addStreetProfilesRecipe(builder.graph, {
      positions: positions.view,
      normals: normals.view,
      segments: segments.view,
      segmentBuildingIds: owners.view,
      buildingHeights: heights.view,
      maximumDistance: values.maximumDistance,
      sampleSpacing: 0.02,
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
        {name: 'rays', buffer: rays, access: 'read_write'},
        {name: 'ratios', buffer: ratios, access: 'read_write'}
      ],
      body: /* wgsl */ `
  let origin = positions[positionsOffset + index];
  let normal = normals[normalsOffset + index];
  let profile = profiles[profilesOffset + index];
  let isFinitePair = profile.x == profile.x && profile.y == profile.y && profile.x < ${values.maximumDistance * 1.001} && profile.y < ${values.maximumDistance * 1.001};
  rays[raysOffset + 2u * index] = select(vec4f(-9.0), vec4f(origin, origin + normal * profile.x), isFinitePair);
  rays[raysOffset + 2u * index + 1u] = select(vec4f(-9.0), vec4f(origin, origin - normal * profile.y), isFinitePair);
  ratios[ratiosOffset + index] = select(1.0e30, canyon[canyonOffset + index].z, isFinitePair);`
    });
    const wallSegments = builder.allocate('wall-segments', 'float32x4', SLOT_COUNT * 2 * 4);
    return {
      outputs: {positions, rays, ratios, wallSegments},
      update: current => {
        const next = inputs.getWalls(current.streetWidth);
        segments.write(next.segments);
        wallSegments.write(next.segments);
      }
    };
  },
  rebuildOn: ['maximumDistance'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.wallSegments, radius: 3},
    {layer: 'lines', segments: outputs.rays, color: [0.38, 0.68, 1, 0.6], radius: 1.4},
    {layer: 'dots', positions: outputs.positions, values: outputs.ratios, radius: 5, range: [0, 2]}
  ]
});
