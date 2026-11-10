// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMapMatching: noisy GPS fixes (muted polylines) are matched to the road network with a hidden
// Markov model. The bright dots are the matched positions on the roads, colored by snap distance.

import {
  GPUMapMatching,
  getGPUMapMatchingParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {uploadNetworkTopology} from '../../inputs';
import {addPairSegments} from './shared/display';
import {createRoads, getRoadContext, uploadRoads} from './shared/roads';
import {parameters} from './map-matching.parameters';

const TRACK_COUNT = 6;
const STEP_COUNT = 24;
const POINT_COUNT = TRACK_COUNT * STEP_COUNT;

export default defineOperationRun({
  parameters,
  caption: 'GPS fixes (muted) and where the matcher puts them on the road network.',
  createInputs: ({random, world}) => {
    const network = createRoads(random, world, 14);
    // Random walks along roads that never step straight back.
    const truth = new Float32Array(POINT_COUNT * 2);
    for (let track = 0; track < TRACK_COUNT; track++) {
      let node = network.findNearestNode(random() * world.width, random());
      let previous = -1;
      for (let step = 0; step < STEP_COUNT; step++) {
        const slotStart = network.offsets[node];
        const options: number[] = [];
        for (let slot = slotStart; slot < network.offsets[node + 1]; slot++) {
          if (network.neighbors[slot] !== previous) options.push(network.neighbors[slot]);
        }
        const index = track * STEP_COUNT + step;
        truth[index * 2] = network.nodes[node * 2];
        truth[index * 2 + 1] = network.nodes[node * 2 + 1];
        const next = options.length ? options[Math.floor(random() * options.length)] : previous;
        previous = node;
        node = next;
      }
    }
    const noise = Float32Array.from(
      {length: POINT_COUNT * 2},
      () => (random() + random() + random() - 1.5) * 1.3
    );
    const trackOffsets = Uint32Array.from(
      {length: TRACK_COUNT + 1},
      (_, index) => index * STEP_COUNT
    );
    const defaultNoise = parameters.find(parameter => parameter.id === 'noise')?.default ?? 0;
    const defaultFixes = Float32Array.from(
      truth,
      (value, index) => value + noise[index] * defaultNoise
    );
    return {
      network,
      truth,
      noise,
      trackOffsets,
      fixes: new Float32Array(POINT_COUNT * 2),
      world,
      defaultFixes,
      restFix: Math.floor(POINT_COUNT / 2),
      drag: {index: -1, x: 0, y: 0}
    };
  },
  build(builder, inputs) {
    const {network, truth, noise, fixes, trackOffsets, world} = inputs;
    const {topology} = uploadNetworkTopology(builder, network);
    const nodePositions = builder.upload('node-positions', 'float32x2', network.nodes);
    const positions = builder.allocate('fixes', 'float32x2', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const matchedEdges = builder.allocate('matched-edges', 'uint32', POINT_COUNT);
    const snapDistances = builder.allocate('snap-distances', 'float32', POINT_COUNT);
    const snappedPositions = builder.allocate('snapped-positions', 'float32x2', POINT_COUNT);
    builder.graph.add(
      new GPUMapMatching({
        positions: positions.view,
        trackOffsets: builder.upload('track-offsets', 'uint32', trackOffsets).view,
        topology: {...topology, nodePositions: nodePositions.view},
        parameters: parameterBuffer.view,
        candidateCount: 8,
        cellSize: 0.06,
        bounds: {minimum: [-0.2, -0.2], maximum: [world.width + 0.2, 1.2]},
        output: {
          matchedEdges: matchedEdges.view,
          snapDistances: snapDistances.view,
          snappedPositions: snappedPositions.view
        }
      })
    );
    const segments = new Float32Array((POINT_COUNT - TRACK_COUNT) * 4);
    const polylines = builder.upload('polylines', 'float32x4', segments);
    return {
      outputs: {
        ...uploadRoads(builder, network),
        positions,
        snapDistances,
        snappedPositions,
        polylines,
        connectors: addPairSegments(builder, {id: 'match', from: positions, to: snappedPositions})
      },
      update: values => {
        for (let index = 0; index < fixes.length; index++)
          fixes[index] = truth[index] + noise[index] * values.noise;
        if (inputs.drag.index >= 0) {
          fixes[inputs.drag.index * 2] = inputs.drag.x;
          fixes[inputs.drag.index * 2 + 1] = inputs.drag.y;
        }
        positions.write(fixes);
        let segment = 0;
        for (let point = 0; point < POINT_COUNT - 1; point++) {
          if ((point + 1) % STEP_COUNT === 0) continue;
          segments.set(
            [fixes[point * 2], fixes[point * 2 + 1], fixes[point * 2 + 2], fixes[point * 2 + 3]],
            segment * 4
          );
          segment++;
        }
        polylines.write(segments);
        parameterBuffer.write(
          getGPUMapMatchingParameterValues({
            sigma: values.sigma,
            beta: 0.05,
            searchRadius: values.searchRadius
          })
        );
      }
    };
  },
  // Drag: the GPS fix nearest the pointer follows it; the matcher re-snaps it within searchRadius.
  interaction: {
    kind: 'drag',
    hint: 'drag a GPS fix',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.defaultFixes[inputs.restFix * 2],
        inputs.defaultFixes[inputs.restFix * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const distanceTo = (fix: number) =>
        Math.hypot(
          inputs.defaultFixes[fix * 2] - pointer.x,
          inputs.defaultFixes[fix * 2 + 1] - pointer.y
        );
      let fix = 0;
      for (let candidate = 1; candidate < POINT_COUNT; candidate++) {
        if (distanceTo(candidate) < distanceTo(fix)) fix = candidate;
      }
      if (inputs.drag.index >= 0 && distanceTo(fix) > distanceTo(inputs.drag.index) * 0.85) {
        fix = inputs.drag.index;
      }
      const changed =
        fix !== inputs.drag.index || pointer.x !== inputs.drag.x || pointer.y !== inputs.drag.y;
      inputs.drag = {index: fix, x: pointer.x, y: pointer.y};
      const distances = peek(outputs.snapDistances);
      const snap = distances ? distances[fix] : Number.NaN;
      return {
        rerun: changed,
        glyphs: [
          {shape: 'ring', at: [pointer.x, pointer.y], radius: values.searchRadius, dashed: true},
          {shape: 'crosshair', at: [pointer.x, pointer.y], size: 10}
        ],
        readout: `fix ${fix} · searchRadius ${values.searchRadius.toFixed(3)} · ${
          Number.isFinite(snap) && snap >= 0 && snap < 1
            ? `snapped ${snap.toFixed(3)}`
            : 'unmatched'
        }`
      };
    }
  },
  draw: ({outputs, theme}) => [
    getRoadContext(outputs),
    {layer: 'lines', segments: outputs.connectors, radius: 1, color: [...theme.ink, 0.5]},
    // Raw GPS: a muted track, drawn thicker so the zigzag off the roads is visible.
    {
      layer: 'lines',
      context: true,
      segments: outputs.polylines,
      radius: 1.6,
      color: [...theme.categories[1], 0.75]
    },
    // Matched positions: an ink ring under a ramp-colored dot, on top of the roads.
    {layer: 'dots', positions: outputs.snappedPositions, radius: 8, color: [...theme.ink, 1]},
    {
      layer: 'dots',
      positions: outputs.snappedPositions,
      values: outputs.snapDistances,
      range: [-0.012, 0.03],
      radius: 6
    }
  ]
});
