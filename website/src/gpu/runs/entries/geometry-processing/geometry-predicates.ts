// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryPredicates over linestrings: counter-clockwise rings, clockwise rings, bow-ties
// (self-crossing) and open lines. Edges take the color of the chosen written column.

import {GPUGeometryPredicates} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {chooseSticky, createPointIndex} from '../../pick';
import {createLineEdges, createPolygonSet, createRingOfVertex} from './shared/polygon-set';
import {findRingCrossings} from './shared/pointer-support';
import {addFeatureValues} from './shared/shape-segments';
import {parameters} from './geometry-predicates.parameters';

/** One closed or open line per polygon: kind 0 ccw ring, 1 cw ring, 2 bow-tie ring, 3 open line. */
function createLines(random: () => number, world: {width: number; height: number}) {
  const set = createPolygonSet(random, world, {
    rows: 3,
    columns: 5,
    vertexCount: 8,
    irregularity: 0.2
  });
  const positions: number[] = [];
  const lineOffsets = [0];
  for (let feature = 0; feature < set.count; feature++) {
    const first = set.ringOffsets[feature];
    const ring: number[][] = [];
    for (let vertex = 0; vertex < 8; vertex++) {
      ring.push([set.positions[(first + vertex) * 2], set.positions[(first + vertex) * 2 + 1]]);
    }
    const kind = feature % 4;
    if (kind === 1) ring.reverse();
    if (kind === 2) [ring[2], ring[3]] = [ring[3], ring[2]];
    if (kind !== 3) ring.push(ring[0]);
    for (const [x, y] of ring) positions.push(x, y);
    lineOffsets.push(positions.length / 2);
  }
  const lineCenters = Float32Array.from({length: set.count * 2}, (_, component) => {
    let sum = 0;
    for (
      let vertex = lineOffsets[component >> 1];
      vertex < lineOffsets[(component >> 1) + 1];
      vertex++
    ) {
      sum += positions[vertex * 2 + (component & 1)];
    }
    return sum / (lineOffsets[(component >> 1) + 1] - lineOffsets[component >> 1]);
  });
  return {
    count: set.count,
    positions: Float32Array.from(positions),
    lineOffsets: Uint32Array.from(lineOffsets),
    lineCenters,
    index: createPointIndex(lineCenters),
    focus: {line: -1}
  };
}

export default defineOperationRun({
  parameters,
  caption: 'Per-line flags: simple, ring, closed and counter-clockwise.',
  createInputs: ({seed, world}) => createLines(makeRandom(seed), world),
  build(builder, lines) {
    const positions = builder.upload('positions', 'float32x2', lines.positions);
    const lineOffsets = builder.upload('line-offsets', 'uint32', lines.lineOffsets);
    const columns = {
      isSimple: builder.allocate('is-simple', 'uint32', lines.count),
      isRing: builder.allocate('is-ring', 'uint32', lines.count),
      isClosed: builder.allocate('is-closed', 'uint32', lines.count),
      isCcw: builder.allocate('is-ccw', 'uint32', lines.count)
    };
    builder.graph.add(
      new GPUGeometryPredicates({
        geometry: {kind: 'lines', positions: positions.view, lineOffsets: lineOffsets.view},
        intersectionCapacity: 4096,
        output: {
          isSimple: columns.isSimple.view,
          isRing: columns.isRing.view,
          isClosed: columns.isClosed.view,
          isCcw: columns.isCcw.view,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    const lineOfVertex = builder.upload(
      'line-of-vertex',
      'uint32',
      createRingOfVertex(lines.lineOffsets)
    );
    return {
      // `output` only chooses which column `draw` shows, so it never recompiles.
      update: () => undefined,
      outputs: {
        columns,
        positions,
        edges: builder.upload('edges', 'uint32x2', createLineEdges(lines.lineOffsets)),
        flags: [
          addFeatureValues(builder, 'simple', columns.isSimple, lineOfVertex),
          addFeatureValues(builder, 'ring', columns.isRing, lineOfVertex),
          addFeatureValues(builder, 'closed', columns.isClosed, lineOfVertex),
          addFeatureValues(builder, 'ccw', columns.isCcw, lineOfVertex)
        ]
      }
    };
  },
  // The pointer picks a line; its four written flags are read back and the line is outlined.
  interaction: {
    kind: 'probe',
    hint: 'hover a line',
    rest: ({inputs, toUnit}) => toUnit([inputs.lineCenters[2 * 2], inputs.lineCenters[2 * 2 + 1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      const line = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.line);
      inputs.focus.line = line;
      if (line < 0) return {};
      const first = inputs.lineOffsets[line];
      const last = inputs.lineOffsets[line + 1];
      const flags = [
        ['isSimple', peek(outputs.columns.isSimple)],
        ['isRing', peek(outputs.columns.isRing)],
        ['isClosed', peek(outputs.columns.isClosed)],
        ['isCcw', peek(outputs.columns.isCcw)]
      ] as const;
      const crossings = findRingCrossings(inputs.positions, first, last - first - 1);
      return {
        focus: line,
        glyphs: [
          {shape: 'polyline', points: inputs.positions.subarray(first * 2, last * 2)},
          ...(flags[0][1]?.[line] === 0 ? crossings.map(at => ({shape: 'dot' as const, at})) : [])
        ],
        readout: flags
          .map(([name, column]) => `${name} ${column ? column[line] === 1 : '...'}`)
          .join(' · ')
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'segments',
      nodes: outputs.positions,
      edges: outputs.edges,
      values: outputs.flags[values.output],
      radius: 2.8
    }
  ]
});
