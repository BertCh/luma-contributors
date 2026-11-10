// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialJoinPrepared builds the zone index once and reuses it while the points change every
// frame. Move `time`: the points slide across the static zones and the matched pairs (lines, from
// the prepared join) follow them.

import {
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPairSegments} from './support/kernels';
import {
  createBoundsSegments,
  createRandomPoints,
  createZones,
  getZoneBounds,
  uploadZones
} from './support/join';
import {parameters} from './spatial-join-prepared.parameters';

const POINT_COUNT = 360;
const PAIR_CAPACITY = 4096;

export default defineOperationRun({
  parameters,
  caption: 'The zone index is prepared once; moving points are joined against it every frame.',
  createInputs: ({random, world}) => ({
    zones: createZones(world),
    points: createRandomPoints(random, world, POINT_COUNT),
    world,
    pointer: [world.width / 2, 0.5] as [number, number]
  }),
  build(builder, inputs) {
    const points = builder.allocate('points', 'float32x2', POINT_COUNT);
    const zones = uploadZones(builder, inputs.zones);
    const prepared = new GPUSpatialJoinPrepared({id: 'zones', geometry: zones.geometry});
    builder.graph.add(prepared);
    const leftIds = builder.allocate('left-ids', 'uint32', PAIR_CAPACITY);
    const rightIds = builder.allocate('right-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: points.view},
        right: zones.geometry,
        predicate: 'within',
        prepared,
        candidateCapacity: PAIR_CAPACITY,
        output: {
          pairs: {
            leftIds: leftIds.view,
            rightIds: rightIds.view,
            status: {count: count.view, overflow: overflow.view}
          }
        }
      })
    );
    return {
      outputs: {
        points,
        zones,
        count,
        // The prepared index: one bounding box per zone, built once.
        index: builder.upload(
          'index-bounds',
          'float32x4',
          createBoundsSegments(getZoneBounds(inputs.zones, inputs.zones.positions))
        ),
        pairs: addPairSegments(builder, 'pairs', {
          leftPositions: points,
          rightPositions: zones.centers,
          leftIds,
          rightIds,
          count
        })
      },
      update: current =>
        points.write(
          Float32Array.from(inputs.points, (value, index) => {
            const row = Math.floor(index / 2);
            const phase = current.time * 2 * Math.PI + row * 0.37;
            // A swarm centered on the pointer; `time` orbits each point around its own spot.
            if (index % 2 === 0) {
              const x = inputs.pointer[0] + (value - inputs.world.width / 2) * current.spread;
              return x + Math.cos(phase) * 0.06;
            }
            return inputs.pointer[1] + (value - 0.5) * current.spread + Math.sin(phase) * 0.04;
          })
        )
    };
  },
  // Drag: the swarm of points follows the pointer over the zones; the index is not rebuilt.
  interaction: {
    kind: 'drag',
    hint: 'carry the swarm',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const count = peek(outputs.count);
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: inputs.pointer}],
        readout: `index built once · ${count ? count[0] : '…'} pairs this frame`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.index, radius: 1, color: [1, 0.85, 0.3, 0.7]},
    {
      layer: 'lines',
      context: true,
      segments: outputs.zones.outline,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    },
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'lines', segments: outputs.pairs, radius: 1, color: [0.4, 0.9, 0.6, 0.8]}
  ]
});
