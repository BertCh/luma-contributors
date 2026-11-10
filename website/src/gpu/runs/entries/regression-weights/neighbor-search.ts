// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNeighborSearch over clustered points. In knn mode each point links to its k nearest; in
// radius mode to every point within the radius. Points are colored by the neighbor count the
// operation wrote.

import {
  GPUNeighborSearch,
  getGPUNeighborSearchParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './neighbor-search.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  drawLinks,
  stretchToWorld
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Neighbor links found by the GPU grid search.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 260, 5, 0.12);
    const positions = stretchToWorld(points.positions, world);
    return {...points, positions, index: createPointIndex(positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const isRadius = values.mode === 1;
    const weights = allocateWeights(
      builder,
      'search',
      points.count,
      points.count * (isRadius ? 24 : values.k),
      true
    );
    builder.graph.add(
      new GPUNeighborSearch({
        mode: isRadius ? 'radius' : 'knn',
        k: isRadius ? undefined : values.k,
        positions: positions.view,
        gridSize: [24, 24],
        parameters: parameterBuffer.view,
        output: weights.port
      })
    );
    const cardinality = addCardinality(builder, 'search', weights);
    const links = addLinkSegments(builder, 'search', weights, positions);
    return {
      outputs: {positions, cardinality, links},
      update: current =>
        parameterBuffer.write(
          getGPUNeighborSearchParameterValues({
            bounds: [0, 0, ...worldBounds(points.positions)],
            radius: current.radius
          })
        )
    };
  },
  rebuildOn: ['mode', 'k'],
  // The pointer picks a point; the links the operation found for it are drawn bright.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const x = inputs.positions[row * 2];
      const y = inputs.positions[row * 2 + 1];
      const glyphs: RunGlyph[] = [];
      let neighborCount = 0;
      let reach = 0;
      // One segment per CSR slot, written by the link kernel: (row x, row y, neighbor x, neighbor y).
      const segments = peek(outputs.links);
      if (segments) {
        for (let slot = 0; slot < segments.length; slot += 4) {
          if (segments[slot] !== x || segments[slot + 1] !== y) continue;
          const to = [segments[slot + 2], segments[slot + 3]] as const;
          glyphs.push({shape: 'line', from: [x, y], to}, {shape: 'dot', at: to});
          reach = Math.max(reach, Math.hypot(to[0] - x, to[1] - y));
          neighborCount++;
        }
      }
      const isRadius = values.mode === 1;
      glyphs.push({
        shape: 'ring',
        at: [x, y],
        radius: isRadius ? values.radius : reach,
        dashed: true
      });
      return {
        focus: row,
        glyphs,
        readout: isRadius
          ? `point ${row} · ${neighborCount} neighbors within radius ${values.radius}`
          : `point ${row} · k ${values.k} nearest, farthest at ${reach.toFixed(3)}`
      };
    }
  },
  draw: ({outputs}) => [
    drawLinks(outputs.links, [0.95, 0.55, 0.2, 0.3]),
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.cardinality,
      radius: 3.2
    }
  ]
});

function worldBounds(positions: Float32Array): [number, number] {
  let maximumX = 0;
  let maximumY = 0;
  for (let index = 0; index < positions.length; index += 2) {
    maximumX = Math.max(maximumX, positions[index]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  return [maximumX + 0.01, maximumY + 0.01];
}
