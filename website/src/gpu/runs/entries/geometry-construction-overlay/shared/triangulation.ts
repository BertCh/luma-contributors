// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared body of the two polygon triangulation runs (ear clipping and constrained Delaunay): the
// same star-shaped polygons, with the triangles the operation wrote filled by polygon.

import {
  getPolygonTriangulationIndexCount,
  GPUConstrainedDelaunayTriangulation,
  GPUPolygonTriangulation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../../define-run';
import type {GpuSceneParameter} from '../../../../types';
import {clamp} from '../../../interaction';
import {addKernelPass} from './kernel';
import {findNearestVertex, OUTPUT_COLOR} from './lines';
import {
  createBlobSet,
  createRingSegments,
  getPolygonFields,
  uploadPolygonSet,
  type PolygonSet
} from './polygons';

export function createSeededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Minimum interior angle of every output triangle, in degrees: slivers are low, good triangles near 60. */
function addMinimumAngles(
  builder: RunBuilder,
  positions: RunBuffer<'float32x2'>,
  indices: RunBuffer<'uint32'>,
  triangleCapacity: number
): RunBuffer<'float32'> {
  const minimumAngles = builder.allocate('minimum-angles', 'float32', triangleCapacity);
  addKernelPass(builder, {
    id: 'minimum-angle-pass',
    invocationCount: triangleCapacity,
    bindings: [
      {name: 'positions', buffer: positions, access: 'read'},
      {name: 'indices', buffer: indices, access: 'read'},
      {name: 'minimumAngles', buffer: minimumAngles, access: 'read_write'}
    ],
    declarations: /* wgsl */ `
fn cornerAngle(at: vec2f, first: vec2f, second: vec2f) -> f32 {
  let u = normalize(first - at);
  let v = normalize(second - at);
  return degrees(acos(clamp(dot(u, v), -1.0, 1.0)));
}`,
    body: /* wgsl */ `
  let a = positions[positionsOffset + indices[indicesOffset + index * 3u]];
  let b = positions[positionsOffset + indices[indicesOffset + index * 3u + 1u]];
  let c = positions[positionsOffset + indices[indicesOffset + index * 3u + 2u]];
  let area = abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
  var result = 0.0;
  if (area > 1.0e-9) {
    result = min(cornerAngle(a, b, c), min(cornerAngle(b, a, c), cornerAngle(c, a, b)));
  }
  minimumAngles[minimumAnglesOffset + index] = result;`
  });
  return minimumAngles;
}

export function createTriangulationRun(options: {
  parameters: readonly GpuSceneParameter[];
  constrained: boolean;
  caption: string;
}) {
  return defineOperationRun({
    parameters: options.parameters,
    caption: options.caption,
    createInputs: ({world}) => {
      const getSet = (irregularity: number, hasHoles: boolean): PolygonSet =>
        createBlobSet(createSeededRandom(77), world, {
          rows: 2,
          columns: 3,
          vertexCount: 18,
          irregularity,
          fill: 0.82,
          hasHoles
        });
      // The pointer carries one polygon vertex: its offset is applied on top of the slider's set.
      const dragged = {vertex: -1, offset: [0, 0] as [number, number]};
      const getMovedSet = (irregularity: number, hasHoles: boolean): PolygonSet => {
        const set = getSet(irregularity, hasHoles);
        if (dragged.vertex >= 0) {
          set.positions[dragged.vertex * 2] += dragged.offset[0];
          set.positions[dragged.vertex * 2 + 1] += dragged.offset[1];
        }
        return set;
      };
      return {getSet, getMovedSet, dragged};
    },
    build(builder, inputs, values) {
      const set = inputs.getMovedSet(values.irregularity, values.hasHoles === 1);
      const polygons = uploadPolygonSet(builder, 'polygons', set);
      const polygonCount = set.polygonOffsets.length - 1;
      const ringCount = set.ringOffsets.length - 1;
      const triangleCapacity =
        getPolygonTriangulationIndexCount(set.positions.length / 2, ringCount, polygonCount) / 3;
      const indices = builder.allocate('indices', 'uint32', triangleCapacity * 3);
      const valid = builder.allocate('valid', 'uint32', polygonCount);
      const triangleOffsets = builder.allocate('triangle-offsets', 'uint32', polygonCount + 1);
      const triangleCounts = builder.allocate('triangle-counts', 'uint32', polygonCount);
      const converged = builder.allocate('converged', 'uint32', polygonCount);
      const output = {
        indices: indices.view,
        valid: valid.view,
        triangleCount: triangleCounts.view,
        triangleOffsets: triangleOffsets.view
      };
      builder.graph.add(
        options.constrained
          ? new GPUConstrainedDelaunayTriangulation({
              ...getPolygonFields(polygons),
              maximumFlips: values.maximumFlips,
              output: {...output, status: {converged: converged.view}}
            })
          : new GPUPolygonTriangulation({
              ...getPolygonFields(polygons),
              largePolygonMethod: 'monotone',
              output
            })
      );
      const polygonIds = builder.upload(
        'polygon-ids',
        'uint32',
        Uint32Array.from({length: polygonCount}, (_, index) => index)
      );
      const minimumAngles = addMinimumAngles(
        builder,
        polygons.positions,
        indices,
        triangleCapacity
      );
      const triangleIds = builder.upload(
        'triangle-ids',
        'uint32',
        Uint32Array.from({length: triangleCapacity}, (_, index) => index)
      );
      const outline = builder.allocate('outline', 'float32x4', createRingSegments(set).length / 4);
      return {
        outputs: {
          positions: polygons.positions,
          indices,
          minimumAngles,
          triangleIds,
          polygonIds,
          outline,
          triangleCapacity
        },
        update: current => {
          const next = inputs.getMovedSet(current.irregularity, values.hasHoles === 1);
          polygons.positions.write(next.positions);
          outline.write(createRingSegments(next));
        }
      };
    },
    rebuildOn: ['hasHoles', 'maximumFlips'],
    // The pointer carries the nearest polygon vertex (it leans toward the cursor and lets go at a
    // distance) and the triangulation re-forms around it. Triangles are colored by minimum angle.
    interaction: {
      kind: 'drag',
      hint: 'drag a vertex',
      rest: ({inputs, values, toUnit}) => {
        const set = inputs.getSet(values.irregularity, values.hasHoles === 1);
        return toUnit([set.positions[0], set.positions[1]]);
      },
      resolve(pointer, {inputs, values}) {
        const set = inputs.getSet(values.irregularity, values.hasHoles === 1);
        const nearest = findNearestVertex(set.positions, pointer.x, pointer.y);
        if (!nearest) return {};
        const previous = inputs.dragged.vertex;
        let vertex = nearest.row;
        if (previous >= 0 && previous !== vertex) {
          const previousDistance = Math.hypot(
            set.positions[previous * 2] - pointer.x,
            set.positions[previous * 2 + 1] - pointer.y
          );
          if (nearest.distance > previousDistance * 0.85) vertex = previous;
        }
        const baseX = set.positions[vertex * 2];
        const baseY = set.positions[vertex * 2 + 1];
        const deltaX = pointer.x - baseX;
        const deltaY = pointer.y - baseY;
        const weight = clamp((0.25 - Math.hypot(deltaX, deltaY)) / 0.15, 0, 1);
        const offset: [number, number] = [deltaX * weight, deltaY * weight];
        const changed =
          vertex !== previous ||
          offset[0] !== inputs.dragged.offset[0] ||
          offset[1] !== inputs.dragged.offset[1];
        inputs.dragged.vertex = vertex;
        inputs.dragged.offset = offset;
        return {
          rerun: changed,
          glyphs: [
            {shape: 'ring', at: [baseX + offset[0], baseY + offset[1]], radius: 0.025},
            {
              shape: 'line',
              from: [baseX, baseY],
              to: [baseX + offset[0], baseY + offset[1]],
              dashed: true
            }
          ],
          readout:
            values.maximumFlips === undefined
              ? `vertex ${vertex} carried`
              : `maximumFlips ${values.maximumFlips} · vertex ${vertex} carried`
        };
      }
    },
    draw: ({outputs}) => [
      {
        layer: 'polygons',
        positions: outputs.positions,
        indices: outputs.indices,
        owners: outputs.triangleIds,
        values: outputs.minimumAngles,
        ramp: 'sequential',
        range: [0, 60],
        triangleCount: outputs.triangleCapacity,
        alpha: 0.85
      },
      {
        layer: 'triangles',
        positions: outputs.positions,
        triangles: outputs.indices,
        count: outputs.triangleCapacity,
        color: [1, 1, 1, 0.55],
        radius: 0.7
      },
      {layer: 'lines', segments: outputs.outline, color: OUTPUT_COLOR, radius: 1.5}
    ]
  });
}
