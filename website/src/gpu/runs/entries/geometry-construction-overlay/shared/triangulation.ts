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
import {createPackedSegments, findNearestVertex, HIGHLIGHT_COLOR, OUTPUT_COLOR} from './lines';
import {createLegend} from '../../../text';
import {createBlobSet, getPolygonFields, uploadPolygonSet, type PolygonSet} from './polygons';

// The edge-flip limits the constrained triangulation's pointer height picks among.
const FLIP_LEVELS = [3, 8, 16, 24, 36] as const;

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

/**
 * Ring edges of a polygon set as segments, split into outer rings (the first ring of each polygon)
 * and holes. Both lists are padded to at least one off-screen segment so they can be buffers.
 */
function createSplitRingSegments(set: PolygonSet): {shells: Float32Array; holes: Float32Array} {
  const shells: number[] = [];
  const holes: number[] = [];
  for (let polygon = 0; polygon + 1 < set.polygonOffsets.length; polygon++) {
    for (let ring = set.polygonOffsets[polygon]; ring < set.polygonOffsets[polygon + 1]; ring++) {
      const segments = createPackedSegments(
        set.positions,
        Uint32Array.of(set.ringOffsets[ring], set.ringOffsets[ring + 1]),
        true
      );
      (ring === set.polygonOffsets[polygon] ? shells : holes).push(...segments);
    }
  }
  const pad = (segments: number[]) =>
    Float32Array.from(segments.length > 0 ? segments : [-9, -9, -9, -9]);
  return {shells: pad(shells), holes: pad(holes)};
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
      const rings = createSplitRingSegments(set);
      const outline = builder.allocate('outline', 'float32x4', rings.shells.length / 4);
      const holeOutline = builder.allocate('hole-outline', 'float32x4', rings.holes.length / 4);
      return {
        outputs: {
          positions: polygons.positions,
          indices,
          minimumAngles,
          triangleIds,
          polygonIds,
          outline,
          holeOutline,
          converged,
          triangleCapacity
        },
        update: current => {
          const next = inputs.getMovedSet(current.irregularity, values.hasHoles === 1);
          polygons.positions.write(next.positions);
          const nextRings = createSplitRingSegments(next);
          outline.write(nextRings.shells);
          holeOutline.write(nextRings.holes);
        }
      };
    },
    rebuildOn: ['hasHoles', 'maximumFlips'],
    buildCache: options.constrained ? 6 : undefined,
    // The pointer carries the nearest polygon vertex (it leans toward the cursor and lets go at a
    // distance) and the triangulation re-forms around it. Triangles are colored by minimum angle.
    interaction: options.constrained
      ? {
          // The pointer's height picks maximumFlips among five limits (applied once it rests): more
          // flips turn more slivers into well-shaped triangles in every polygon.
          kind: 'level',
          hint: 'up: allow more edge flips',
          rest: [0.5, 0.95 - 0.9 * 0.1],
          resolve(pointer, {outputs, values, peek}) {
            const fraction = clamp((0.95 - pointer.unitY) / 0.9, 0, 0.9999);
            const level = Math.floor(fraction * FLIP_LEVELS.length);
            const converged = peek(outputs.converged);
            let convergedCount = 0;
            for (let row = 0; converged && row < converged.length; row++) {
              if (converged[row]) convergedCount++;
            }
            const glyphs = FLIP_LEVELS.map((_, index) => {
              const y = 1 - (0.95 - 0.9 * ((index + 0.5) / FLIP_LEVELS.length));
              return {
                shape: 'line' as const,
                from: [0.01, y] as const,
                to: [index === level ? 0.1 : 0.04, y] as const
              };
            });
            return {
              settle: {maximumFlips: FLIP_LEVELS[level]},
              glyphs,
              readout: `maximumFlips ${FLIP_LEVELS[level]} · ${
                converged && values.maximumFlips === FLIP_LEVELS[level]
                  ? `${convergedCount} of ${converged.length} polygons converged`
                  : 'flipping…'
              }`
            };
          }
        }
      : {
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
    draw: ({outputs, theme}) => [
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
      {layer: 'lines', segments: outputs.outline, color: OUTPUT_COLOR, radius: 1.8},
      {layer: 'lines', segments: outputs.holeOutline, color: HIGHLIGHT_COLOR, radius: 1.8},
      {
        layer: 'text',
        ...createLegend({
          corner: options.constrained ? 'top-right' : 'top-left',
          entries: [
            {
              label: options.constrained ? 'outer ring (constraint)' : 'outer ring',
              swatch: [OUTPUT_COLOR[0], OUTPUT_COLOR[1], OUTPUT_COLOR[2]]
            },
            {
              label: options.constrained ? 'hole ring (constraint)' : 'hole ring',
              swatch: [HIGHLIGHT_COLOR[0], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[2]]
            },
            {
              label: 'triangle color: smallest angle',
              swatch: theme.ramp[Math.floor(theme.ramp.length * 0.6)]
            }
          ]
        })
      }
    ]
  });
}
