// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUIsobandRings chains the boundary edges of every isoband into closed rings (shells and holes).
// The rings it wrote are drawn as closed outlines, shells and holes in two colors, over the surface.

import {
  getGPUIsobandsParameterValues,
  GPUIsobandRings
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {isInsideRing} from '../../pick';
import {parameters} from './isoband-rings.parameters';
import {createIsoInputs, getLevels, LEVEL_CAPACITY} from './shared/iso';

const RING_CAPACITY = 600;
const VERTEX_CAPACITY = 20000;
const SHELL_COLOR = [1, 0.85, 0.35, 1] as const;
const HOLE_COLOR = [0.4, 0.85, 1, 1] as const;

/** Closed outlines of the rings picked by `isRingWanted`, as (x0, y0, x1, y1) segments. */
function getRingSegments(
  offsets: Uint32Array,
  positions: Float32Array,
  ringCount: number,
  isRingWanted: (ring: number) => boolean
): Float32Array {
  const segments: number[] = [];
  for (let ring = 0; ring < ringCount; ring++) {
    if (!isRingWanted(ring)) continue;
    for (let vertex = offsets[ring]; vertex + 1 < offsets[ring + 1]; vertex++) {
      segments.push(
        positions[vertex * 2],
        positions[vertex * 2 + 1],
        positions[vertex * 2 + 2],
        positions[vertex * 2 + 3]
      );
    }
  }
  // An empty buffer cannot be bound: one NaN segment stands in and is never counted.
  return segments.length > 0 ? Float32Array.from(segments) : new Float32Array(4).fill(Number.NaN);
}

function readUint32(bytes: Uint8Array): Uint32Array {
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
}

export default defineOperationRun({
  parameters,
  caption: 'Closed shells and holes assembled from the band boundaries.',
  createInputs: context => createIsoInputs(context, 32),
  build(builder, inputs) {
    const ringOffsets = builder.allocate('ring-offsets', 'uint32', RING_CAPACITY + 1);
    const ringIsHole = builder.allocate('ring-is-hole', 'uint32', RING_CAPACITY);
    const ringShells = builder.allocate('ring-shells', 'uint32', RING_CAPACITY);
    const ringGroups = builder.allocate('ring-groups', 'uint32', RING_CAPACITY);
    const ringCount = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const edgeOverflow = builder.allocate('edge-overflow', 'uint32', 1);
    const values = builder.upload('values', 'float32', inputs.valuesBottomUp);
    const display = builder.upload('display', 'float32', inputs.valuesTopDown);
    const breaks = builder.allocate('breaks', 'float32', LEVEL_CAPACITY);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const positions = builder.allocate('positions', 'float32x2', VERTEX_CAPACITY);
    builder.graph.add(
      new GPUIsobandRings({
        grid: inputs.grid,
        values: values.view,
        breaks: breaks.view,
        parameters: parameterBuffer.view,
        edgeCapacity: VERTEX_CAPACITY,
        output: {
          ringOffsets: ringOffsets.view,
          positions: positions.view,
          ringIsHole: ringIsHole.view,
          ringShells: ringShells.view,
          ringGroups: ringGroups.view,
          status: {
            count: ringCount.view,
            overflow: overflow.view,
            edgeOverflow: edgeOverflow.view
          }
        }
      })
    );
    return {
      outputs: {
        display,
        positions,
        ringOffsets,
        ringIsHole,
        ringShells,
        ringGroups,
        ringCount,
        overflow,
        edgeOverflow
      },
      update: next => {
        breaks.write(getLevels(next['interval'], next['interval']));
        parameterBuffer.write(
          getGPUIsobandsParameterValues(
            {breakCount: Math.round(next['breakCount']), extent: inputs.extent},
            inputs.grid
          )
        );
        // Vertices past the written ones stay out of the picture.
        positions.write(new Float32Array(VERTEX_CAPACITY * 2).fill(Number.NaN));
      }
    };
  },
  async afterRun({outputs, upload}) {
    const [offsetBytes, positionBytes, holeBytes, countBytes, overflowBytes, edgeOverflowBytes] =
      await Promise.all([
        outputs.ringOffsets.buffer.readAsync(),
        outputs.positions.buffer.readAsync(),
        outputs.ringIsHole.buffer.readAsync(),
        outputs.ringCount.buffer.readAsync(),
        outputs.overflow.buffer.readAsync(),
        outputs.edgeOverflow.buffer.readAsync()
      ]);
    const offsets = readUint32(offsetBytes);
    const holes = readUint32(holeBytes);
    const positions = new Float32Array(
      positionBytes.buffer.slice(
        positionBytes.byteOffset,
        positionBytes.byteOffset + positionBytes.byteLength
      )
    );
    const incomplete = readUint32(overflowBytes)[0] !== 0 || readUint32(edgeOverflowBytes)[0] !== 0;
    const count = incomplete ? 0 : Math.min(readUint32(countBytes)[0], RING_CAPACITY);
    const shells = getRingSegments(offsets, positions, count, ring => holes[ring] === 0);
    const holeSegments = getRingSegments(offsets, positions, count, ring => holes[ring] === 1);
    return {
      shells: upload('shell-segments', 'float32x4', shells),
      shellCount: shells[0] === shells[0] ? shells.length / 4 : 0,
      holes: upload('hole-segments', 'float32x4', holeSegments),
      holeCount: holeSegments[0] === holeSegments[0] ? holeSegments.length / 4 : 0
    };
  },
  // The pointer picks the innermost ring around it: the ring is outlined thick, and the holes of a
  // shell are dashed.
  interaction: {
    kind: 'probe',
    hint: 'hover a ring',
    rest: [0.5, 0.45],
    resolve(pointer, {outputs, peek}) {
      const offsets = peek(outputs.ringOffsets) as Uint32Array | null;
      const positions = peek(outputs.positions) as Float32Array | null;
      const holes = peek(outputs.ringIsHole) as Uint32Array | null;
      const shells = peek(outputs.ringShells) as Uint32Array | null;
      const groups = peek(outputs.ringGroups) as Uint32Array | null;
      const counts = peek(outputs.ringCount) as Uint32Array | null;
      const overflow = peek(outputs.overflow) as Uint32Array | null;
      const edgeOverflow = peek(outputs.edgeOverflow) as Uint32Array | null;
      if (!overflow || !edgeOverflow) return {};
      if (edgeOverflow[0]) return {readout: 'Incomplete rings: boundary edge capacity exceeded'};
      if (overflow[0]) return {readout: 'Incomplete rings: ring or vertex capacity exceeded'};
      if (!offsets || !positions || !holes || !shells || !groups || !counts) return {};
      const ringCount = Math.min(counts[0], RING_CAPACITY);
      let picked = -1;
      let smallest = Infinity;
      for (let ring = 0; ring < ringCount; ring++) {
        const start = offsets[ring];
        const length = offsets[ring + 1] - start;
        if (length < 4 || !isInsideRing(positions, pointer.x, pointer.y, start, length)) continue;
        let minimumX = Infinity;
        let maximumX = -Infinity;
        let minimumY = Infinity;
        let maximumY = -Infinity;
        for (let vertex = start; vertex < start + length; vertex++) {
          minimumX = Math.min(minimumX, positions[vertex * 2]);
          maximumX = Math.max(maximumX, positions[vertex * 2]);
          minimumY = Math.min(minimumY, positions[vertex * 2 + 1]);
          maximumY = Math.max(maximumY, positions[vertex * 2 + 1]);
        }
        const area = (maximumX - minimumX) * (maximumY - minimumY);
        if (area < smallest) {
          smallest = area;
          picked = ring;
        }
      }
      if (picked < 0) return {readout: `${ringCount} rings`};
      const getOutline = (ring: number) =>
        positions.subarray(offsets[ring] * 2, offsets[ring + 1] * 2);
      const glyphs: RunGlyph[] = [{shape: 'polyline', points: getOutline(picked)}];
      let holeCount = 0;
      if (holes[picked] === 0) {
        for (let ring = 0; ring < ringCount; ring++) {
          if (holes[ring] === 1 && shells[ring] === picked) {
            holeCount++;
            glyphs.push({shape: 'polyline', points: getOutline(ring), dashed: true});
          }
        }
      }
      return {
        glyphs,
        readout: `ring ${picked} · ${holes[picked] === 1 ? 'hole' : `shell, ${holeCount} holes`} · band ${groups[picked]}`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.display,
      columns: inputs.columns,
      rows: inputs.rows,
      smoothing: true
    },
    {
      layer: 'lines',
      segments: extra.shells,
      count: extra.shellCount,
      radius: 1.5,
      color: SHELL_COLOR
    },
    {layer: 'lines', segments: extra.holes, count: extra.holeCount, radius: 1.5, color: HOLE_COLOR}
  ]
});
