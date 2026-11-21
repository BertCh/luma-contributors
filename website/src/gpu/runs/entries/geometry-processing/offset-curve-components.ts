// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOffsetCurveComponents cleans the raw one-sided offset that GPUOffsetCurve produces. The
// muted path is the raw offset, with its loops on the inside of the turns; the colored paths are
// the clean components the operation wrote, one color each.

import {
  getGPUOffsetCurveRowsPerVertex,
  GPUOffsetCurve,
  GPUOffsetCurveComponents
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../define-run';
import {findNearestOnPolyline} from '../../pick';
import {createRiverBend} from '../../../data/hydrography';
import {createIdentityIds} from './shared/polygon-set';
import {findPointAtDistance, getNearestValue} from './shared/pointer-support';
import {addKernel} from './shared/shape-segments';
import {parameters} from './offset-curve-components.parameters';

const JOIN_STYLES = ['round', 'mitre', 'bevel'] as const;
const QUAD_SEGMENTS = 8;
const OUTPUT_VERTICES = 1024;
/** distance is a constructor prop: the pointer settles on one of these prebuilt values. */
const DISTANCES = [0.03, 0.06, 0.09, 0.12, 0.15];
/** The offset distance is this fraction of the pointer's distance from the river. */
const DISTANCE_SCALE = 0.5;
const COMPONENT_CAPACITY = 128;

/** Links vertices in the readable component prefix and makes all other edges degenerate. */
export function addOffsetCurveComponentEdges(
  builder: RunBuilder,
  components: RunBuffer<'uint32x4'>,
  status: RunBuffer<'uint32x4'>,
  edges: RunBuffer<'uint32x2'>,
  categories: RunBuffer<'uint32'>
): void {
  addKernel(builder, {
    id: 'component-edges',
    invocationCount: edges.length,
    bindings: [
      {name: 'components', buffer: components, access: 'read'},
      {name: 'status', buffer: status, access: 'read'},
      {name: 'edges', buffer: edges, access: 'read_write'},
      {name: 'categories', buffer: categories, access: 'read_write'}
    ],
    body: `let writtenComponents = status[statusOffset];
  let writtenVertices = status[statusOffset + 1u];
  let requiredComponents = status[statusOffset + 2u];
  let requiredVertices = status[statusOffset + 3u];
  let loss = status[statusOffset + 4u] | status[statusOffset + 5u] | status[statusOffset + 6u];
  let complete = loss == 0u && writtenComponents == requiredComponents &&
    writtenVertices == requiredVertices && writtenVertices <= ${edges.length}u;
  let componentCount = select(0u, min(writtenComponents, ${components.length}u),
    complete && index < writtenVertices);
  var nextVertex = index;
  var category = 0xffffffffu;
  for (var component = 0u; component < componentCount; component++) {
    let first = components[componentsOffset + component * 4u];
    let count = components[componentsOffset + component * 4u + 1u];
    if (first <= writtenVertices && count <= writtenVertices - first &&
        index >= first && index - first < count) {
      category = component;
      if (index - first + 1u < count) { nextVertex = index + 1u; }
      break;
    }
  }
  edges[edgesOffset + index * 2u] = index;
  edges[edgesOffset + index * 2u + 1u] = nextVertex;
  categories[categoriesOffset + index] = category;`
  });
}

export default defineOperationRun({
  parameters,
  caption: 'The raw offset of a river reach, cleaned into valid components.',
  buildCache: 5,
  createInputs: ({world}) => {
    const vertexCount = 24;
    const positions = Float32Array.from(createRiverBend(1, vertexCount, world).flat());
    return {
      positions,
      vertexCount,
      restPoint: findPointAtDistance(
        (x, y) => findNearestOnPolyline(positions, x, y)?.distance ?? null,
        0.09 / DISTANCE_SCALE,
        world
      )
    };
  },
  build(builder, {positions, vertexCount}, values) {
    const joinStyle = JOIN_STYLES[values.joinStyle];
    const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(joinStyle, QUAD_SEGMENTS, 'buffer');
    const rawCount = vertexCount * rowsPerVertex;
    const original = builder.upload('original', 'float32x2', positions);
    const originalOffsets = builder.upload(
      'original-offsets',
      'uint32',
      Uint32Array.of(0, vertexCount)
    );
    const raw = builder.allocate('raw', 'float32x2', rawCount);
    const rawOffsets = builder.upload('raw-offsets', 'uint32', Uint32Array.of(0, rawCount));
    const parameterBuffer = builder.upload(
      'parameters',
      'float32',
      Float32Array.of(values.distance, 5, 0, 0)
    );
    builder.graph.add(
      new GPUOffsetCurve({
        positions: original.view,
        lineOffsets: originalOffsets.view,
        joinStyle,
        quadSegments: QUAD_SEGMENTS,
        insideTurns: 'buffer',
        parameters: parameterBuffer.view,
        output: {positions: raw.view}
      })
    );
    const cleaned = builder.allocate('cleaned', 'float32x2', OUTPUT_VERTICES);
    const components = builder.allocate('components', 'uint32x4', COMPONENT_CAPACITY);
    const status = builder.allocate('status', 'uint32x4', 2);
    builder.graph.add(
      new GPUOffsetCurveComponents({
        original: {positions: original.view, lineOffsets: originalOffsets.view},
        raw: {positions: raw.view, lineOffsets: rawOffsets.view},
        rowsPerVertex,
        distance: values.distance,
        tolerance: values.distance * 0.01,
        segmentCapacity: 512,
        intersectionCapacity: 256,
        output: {
          positions: cleaned.view,
          sourceEdges: builder.allocate('source-edges', 'uint32', OUTPUT_VERTICES).view,
          components: components.view,
          status: status.view
        }
      })
    );
    const edges = builder.allocate('component-edges', 'uint32x2', OUTPUT_VERTICES);
    const categories = builder.allocate('component-categories', 'uint32', OUTPUT_VERTICES);
    addOffsetCurveComponentEdges(builder, components, status, edges, categories);
    return {
      outputs: {
        original,
        raw,
        cleaned,
        edges,
        categories,
        originalIds: builder.upload('original-ids', 'uint32', createIdentityIds(vertexCount)),
        rawIds: builder.upload('raw-ids', 'uint32', createIdentityIds(rawCount))
      }
    };
  },
  rebuildOn: ['distance', 'joinStyle'],
  // The pointer's distance from the river (halved) is the offset distance; its ring is drawn at the nearest river point.
  interaction: {
    kind: 'reach',
    hint: 'move away from the river',
    rest: ({inputs, toUnit}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, values}) {
      const hit = findNearestOnPolyline(inputs.positions, pointer.x, pointer.y);
      const distance = getNearestValue(DISTANCES, (hit?.distance ?? 0) * DISTANCE_SCALE);
      return {
        settle: {distance},
        glyphs: hit
          ? [
              {shape: 'ring', at: hit.point, radius: distance, dashed: true},
              {shape: 'line', from: [pointer.x, pointer.y], to: hit.point, dashed: true}
            ]
          : [],
        readout:
          distance === values.distance
            ? `distance ${distance.toFixed(3)} · ${values.joinStyle === 0 ? 'round' : values.joinStyle === 1 ? 'mitre' : 'bevel'} joins`
            : `distance ${values.distance.toFixed(3)} -> ${distance.toFixed(3)}`
      };
    }
  },
  draw: ({outputs, inputs, theme}) => [
    {
      layer: 'paths',
      context: true,
      nodes: outputs.raw,
      ids: outputs.rawIds,
      count: outputs.rawIds.length,
      radius: 1.2
    },
    {
      layer: 'segments',
      nodes: outputs.cleaned,
      edges: outputs.edges,
      values: outputs.categories,
      radius: 2.8
    },
    {
      layer: 'paths',
      nodes: outputs.original,
      ids: outputs.originalIds,
      count: inputs.vertexCount,
      color: [...theme.ink, 1],
      radius: 1.4
    }
  ]
});
