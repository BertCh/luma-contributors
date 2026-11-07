// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNormalizePolygons gives every polygon one canonical form. Each ring of the input starts at a
// different vertex and every other polygon winds the other way; the white dots are the input ring
// starts, the orange dots the canonical starts the operation wrote.

import {GPUNormalizePolygons} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, restartRings, uploadPolygonSet} from './shared/polygon-set';
import {addKernel, addOutputEdges} from './shared/shape-segments';
import {parameters} from './normalize-polygons.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Rings re-started at a canonical vertex and wound one way.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {hasHoles: true, rows: 2, columns: 4})
  }),
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
  draw: ({outputs, theme}) => [
    {layer: 'segments', context: true, nodes: outputs.input, edges: outputs.inputEdges},
    {layer: 'segments', nodes: outputs.normalized, edges: outputs.edges, radius: 1.8},
    {layer: 'dots', positions: outputs.inputStarts, radius: 5, color: [...theme.ink, 1]},
    {layer: 'dots', positions: outputs.normalizedStarts, radius: 4, color: [1, 0.6, 0.2, 1]}
  ]
});
