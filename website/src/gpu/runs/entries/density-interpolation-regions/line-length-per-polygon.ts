// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULineLengthPerPolygon splits every line at tract boundaries and sums the length inside each
// tract; tracts are filled by that `lengths` output with the lines on top.

import {GPULineLengthPerPolygon} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import commuteLines from '../../../data/chicago-commute-lines.json';
import {parameters} from './line-length-per-polygon.parameters';
import {getPackedPolygons, uploadPackedPolygons, uploadTractMesh} from './support/geometry';

const DEFAULT_TRACT_ROWS = 7;
// Most commuting flows in the extract end in this tract (the Loop).
const DOMINANT_WORK_TRACT = 35;

export default defineOperationRun({
  parameters,
  caption: 'Length of straight home-to-work flow lines inside each Chicago tract.',
  createInputs: ({world}) => ({
    world,
    // The workplace end of the flows into the dominant work tract, owned by the pointer; until
    // the pointer moves, the flows end at that tract's center, as in the data.
    destination: {x: 0, y: 0, isMoved: false},
    defaultTracts: createTracts(() => 0.5, world, DEFAULT_TRACT_ROWS)
  }),
  build(builder, {world, destination}, values) {
    const tracts = createTracts(() => 0.5, world, values.tractRows);
    const flows = commuteLines
      .filter(([home, work]) => home < tracts.count && work < tracts.count)
      .slice(0, values.lineCount);
    const vertices = Float32Array.from(
      flows.flatMap(([home, work]) => [
        tracts.centers[home * 2],
        tracts.centers[home * 2 + 1],
        tracts.centers[work * 2],
        tracts.centers[work * 2 + 1]
      ])
    );
    const positionBuffer = builder.upload('positions', 'float32x2', vertices);
    const segmentBuffer = builder.upload('segments', 'float32x4', vertices);
    const offsets = Uint32Array.from({length: flows.length + 1}, (_, row) => row * 2);
    const lengths = builder.allocate('lengths', 'float32', tracts.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPULineLengthPerPolygon({
        positions: positionBuffer.view,
        lineOffsets: builder.upload('line-offsets', 'uint32', offsets).view,
        polygons: {kind: 'polygons', ...uploadPackedPolygons(builder, getPackedPolygons(tracts))},
        output: {clippedLengths: lengths.view, status: {overflow: overflow.view}}
      })
    );
    return {
      outputs: {
        lengths,
        ...uploadTractMesh(builder, tracts),
        segments: segmentBuffer,
        segmentCount: flows.length,
        triangleCount: tracts.triangleCount,
        outlineCount: tracts.outlineCount
      },
      // The pointer moves the workplace end of every flow that goes to the dominant tract.
      update: () => {
        const moved = Float32Array.from(vertices);
        flows.forEach(([, work], flow) => {
          if (work !== DOMINANT_WORK_TRACT || !destination.isMoved) return;
          moved[flow * 4 + 2] = destination.x;
          moved[flow * 4 + 3] = destination.y;
        });
        positionBuffer.write(moved);
        segmentBuffer.write(moved);
      }
    };
  },
  rebuildOn: ['lineCount', 'tractRows'],
  // The pointer is the workplace of the flows into the dominant tract: the length inside every
  // tract re-forms as the flows swing around it.
  interaction: {
    kind: 'drag',
    hint: 'move the workplace',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.defaultTracts.centers[DOMINANT_WORK_TRACT * 2],
        inputs.defaultTracts.centers[DOMINANT_WORK_TRACT * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.destination.x = pointer.x;
      inputs.destination.y = pointer.y;
      inputs.destination.isMoved = true;
      const lengths = peek(outputs.lengths);
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y], size: 8}],
        readout: lengths
          ? `${outputs.segmentCount} flows · longest in one tract ${Math.max(...lengths).toFixed(2)}`
          : `${outputs.segmentCount} flows`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.lengths,
      triangleCount: outputs.triangleCount
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: outputs.outlineCount,
      alpha: 0.5
    },
    {
      layer: 'lines',
      context: true,
      color: [1, 1, 1, 1],
      segments: outputs.segments,
      count: outputs.segmentCount
    }
  ]
});
