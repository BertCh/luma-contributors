// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNormalizePolygons gives every polygon one canonical form. Each ring of the input starts at a
// different vertex and every other polygon winds the other way; the white dots are the input ring
// starts, the orange dots the canonical starts the operation wrote.

import {GPUNormalizePolygons} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, restartRings, uploadPolygonSet} from './shared/polygon-set';
import {addKernel, addOutputEdges, addRingArrows} from './shared/shape-segments';
import {findParameter, snapToParameter} from '../../interaction';
import {parameters} from './normalize-polygons.parameters';

const STARTS = findParameter(parameters, 'start');

export default defineOperationRun({
  parameters,
  caption: 'Rings re-started at a canonical vertex and wound one way.',
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {hasHoles: true, rows: 2, columns: 4});
    return {set};
  },
  build(builder, {set}) {
    const polygons = uploadPolygonSet(builder, set);
    const ringCount = set.ringOffsets.length - 1;
    const vertexCount = set.positions.length / 2;
    const normalized = builder.allocate('normalized', 'float32x2', vertexCount);
    const normalizedRings = builder.allocate('normalized-rings', 'uint32', ringCount + 1);
    const normalizedPolygons = builder.allocate(
      'normalized-polygons',
      'uint32',
      set.polygonOffsets.length
    );
    builder.graph.add(
      new GPUNormalizePolygons({
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        output: {
          positions: normalized.view,
          ringOffsets: normalizedRings.view,
          polygonOffsets: normalizedPolygons.view
        }
      })
    );
    const firstVertices = (
      id: string,
      positions: typeof normalized,
      offsets: typeof normalizedRings
    ) => {
      const result = builder.allocate(id, 'float32x2', ringCount);
      addKernel(builder, {
        id: `${id}-kernel`,
        invocationCount: ringCount,
        bindings: [
          {name: 'positions', buffer: positions, access: 'read'},
          {name: 'offsets', buffer: offsets, access: 'read'},
          {name: 'result', buffer: result, access: 'read_write'}
        ],
        body: `let vertex = offsets[offsetsOffset + index];
  result[resultOffset + index * 2u] = positions[positionsOffset + vertex * 2u];
  result[resultOffset + index * 2u + 1u] = positions[positionsOffset + vertex * 2u + 1u];`
      });
      return result;
    };
    const input = builder.allocate('input', 'float32x2', vertexCount);
    return {
      outputs: {
        input,
        normalized,
        edges: addOutputEdges(builder, 'normalized', normalizedRings, vertexCount, true),
        inputEdges: polygons.edges,
        inputStarts: firstVertices('input-starts', input, polygons.ringOffsets),
        normalizedStarts: firstVertices('normalized-starts', normalized, normalizedRings),
        arrows: addRingArrows(builder, 'normalized', normalized, normalizedRings, ringCount),
        inputArrows: addRingArrows(builder, 'input', input, polygons.ringOffsets, ringCount),
        polygons
      },
      update: values => {
        input.write(
          restartRings(
            set,
            ring => ring * 2 + values.start,
            ring => ring % 3 === 1
          )
        );
        polygons.positions.write(
          restartRings(
            set,
            ring => ring * 2 + values.start,
            ring => ring % 3 === 1
          )
        );
      }
    };
  },
  // The pointer's angle around the picture center is where every input ring starts (eight vertices around).
  interaction: {
    kind: 'aim',
    hint: 'aim the ring start',
    rest: ({world, toUnit}) => {
      const angle = (STARTS.default / 8) * Math.PI * 2;
      return toUnit([world.width / 2 + Math.cos(angle) * 0.3, 0.5 + Math.sin(angle) * 0.3]);
    },
    resolve(pointer, {world}) {
      const center: [number, number] = [world.width / 2, 0.5];
      const turn = Math.atan2(pointer.y - center[1], pointer.x - center[0]) / (Math.PI * 2);
      const start = snapToParameter(STARTS, ((turn + 1) % 1) * 8);
      return {
        values: {start: start % 8},
        glyphs: [
          {shape: 'arrow', from: center, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: center}
        ],
        readout: `input start ${start % 8} · canonical start fixed`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {layer: 'segments', context: true, nodes: outputs.input, edges: outputs.inputEdges},
    {layer: 'segments', nodes: outputs.normalized, edges: outputs.edges, radius: 1.8},
    {layer: 'lines', context: true, segments: outputs.inputArrows, radius: 1.4},
    {layer: 'lines', segments: outputs.arrows, radius: 1.4, alpha: 0.8},
    {layer: 'dots', positions: outputs.inputStarts, radius: 8, color: [...theme.ink, 1]},
    {layer: 'dots', positions: outputs.normalizedStarts, radius: 4, color: [1, 0.6, 0.2, 1]}
  ]
});
