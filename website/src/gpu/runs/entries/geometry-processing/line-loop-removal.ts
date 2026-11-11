// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineLoopRemoval erases the self-loops of a meandering curve. The red path is the tangled
// input; the bright path is what remains, with the vertex count the operation wrote.

import {GPULineLoopRemoval} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createRiverLine} from '../../../data/hydrography';
import {defineOperationRun} from '../../define-run';
import {findParameter} from '../../interaction';
import {createLegend} from '../../text';
import {createIdentityIds} from './shared/polygon-set';
import {parameters} from './line-loop-removal.parameters';

const VERTEX_COUNT = 240;
const SPREAD = 10;
const HOLD = 0.3;
const HERO_VERTEX = 120;
const TANGLE = findParameter(parameters, 'tangle');

type CurveDrag = {vertex: number; target: readonly [number, number]};

/** Pulls the carried vertex and its neighbors toward the target with a cosine falloff. */
function applyCurveDrag(base: Float32Array, drag: CurveDrag | null): Float32Array {
  const positions = Float32Array.from(base);
  if (!drag) return positions;
  const deltaX = drag.target[0] - base[drag.vertex * 2];
  const deltaY = drag.target[1] - base[drag.vertex * 2 + 1];
  for (let offset = -SPREAD; offset <= SPREAD; offset++) {
    const vertex = drag.vertex + offset;
    if (vertex < 0 || vertex >= VERTEX_COUNT) continue;
    const weight = 0.5 * (1 + Math.cos((Math.PI * offset) / (SPREAD + 1)));
    positions[vertex * 2] += deltaX * weight;
    positions[vertex * 2 + 1] += deltaY * weight;
  }
  return positions;
}

/** Add four controllable survey loops to a sampled Missouri reach. */
function createTangledLine(reach: number[][], radius: number, tangle: number) {
  return Float32Array.from(
    reach.flatMap(([x, y], vertex) => {
      const angle = (vertex / (VERTEX_COUNT - 1)) * 8 * Math.PI;
      return [x - radius * tangle * Math.sin(angle), y - radius * tangle * Math.cos(angle)];
    })
  );
}

export default defineOperationRun({
  parameters,
  caption: 'Loops erased from a meandering curve in drawing order.',
  createInputs: ({world}) => {
    const reach = createRiverLine(0, VERTEX_COUNT, world);
    const radius = 0.04 * world.width;
    return {
      reach,
      radius,
      base: createTangledLine(reach, radius, TANGLE.default),
      drag: null as CurveDrag | null
    };
  },
  build(builder, inputs, values) {
    const positions = builder.upload(
      'positions',
      'float32x2',
      createTangledLine(inputs.reach, inputs.radius, values.tangle)
    );
    const lineOffsets = builder.upload('line-offsets', 'uint32', Uint32Array.of(0, VERTEX_COUNT));
    const cleaned = builder.allocate('cleaned', 'float32x2', VERTEX_COUNT * 2);
    const cleanedOffsets = builder.allocate('cleaned-offsets', 'uint32', 2);
    const sourceEdges = builder.allocate('source-edges', 'uint32', VERTEX_COUNT * 2);
    const status = builder.allocate('status', 'uint32x4', 1);
    builder.graph.add(
      new GPULineLoopRemoval({
        lines: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        intersectionCapacity: 512,
        output: {
          positions: cleaned.view,
          lineOffsets: cleanedOffsets.view,
          sourceEdges: sourceEdges.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {
        positions,
        cleaned,
        status,
        ids: builder.upload('ids', 'uint32', createIdentityIds(VERTEX_COUNT * 2))
      },
      update: current => {
        inputs.base = createTangledLine(inputs.reach, inputs.radius, current.tangle);
        positions.write(applyCurveDrag(inputs.base, inputs.drag));
      }
    };
  },
  // The pointer carries the nearest curve vertex with its neighbors, pulling a loop open or closed.
  interaction: {
    kind: 'drag',
    hint: 'pull the curve',
    rest: ({inputs, toUnit}) =>
      toUnit([inputs.base[HERO_VERTEX * 2], inputs.base[HERO_VERTEX * 2 + 1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      let nearest = inputs.drag?.vertex ?? HERO_VERTEX;
      let nearestDistance = Infinity;
      for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
        const distance = Math.hypot(
          inputs.base[vertex * 2] - pointer.x,
          inputs.base[vertex * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = vertex;
        }
      }
      const held = inputs.drag?.vertex;
      if (
        held !== undefined &&
        nearestDistance > 0.02 &&
        Math.hypot(inputs.base[held * 2] - pointer.x, inputs.base[held * 2 + 1] - pointer.y) <= HOLD
      ) {
        nearest = held;
      }
      const target: [number, number] = [pointer.x, pointer.y];
      inputs.drag = {vertex: nearest, target};
      const base: [number, number] = [inputs.base[nearest * 2], inputs.base[nearest * 2 + 1]];
      const kept = peek(outputs.status)?.[0];
      return {
        rerun: true,
        glyphs: [
          {shape: 'line', from: base, to: target, dashed: true},
          {shape: 'crosshair', at: target}
        ],
        readout: `${VERTEX_COUNT} -> ${kept ?? '...'} vertices after loop removal`
      };
    }
  },
  // The input path is drawn in red under the cleaned path, so what shows red is what the operation
  // removed; the tangle slider and the drag make more of it.
  draw: ({outputs, theme}) => [
    {
      layer: 'paths',
      context: true,
      nodes: outputs.positions,
      ids: outputs.ids,
      count: VERTEX_COUNT,
      radius: 2,
      color: [...theme.categories[2], 0.9]
    },
    {layer: 'paths', nodes: outputs.cleaned, ids: outputs.ids, count: outputs.status, radius: 2.6},
    {
      layer: 'text',
      ...createLegend({
        corner: 'bottom-left',
        entries: [
          {label: 'kept by the operation', swatch: [...theme.ink]},
          {label: 'loop removed', swatch: 2}
        ]
      })
    }
  ]
});
