// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCanonicalPolygonEquality compares two polygon sets feature by feature after each is
// normalized. The right set restarts every ring at another vertex and winds some the other way,
// and displaces one vertex of each polygon by a growing amount; polygons are filled by the
// equality flag the operation wrote, so a larger tolerance accepts more of them.

import {
  GPUCanonicalPolygonEquality,
  GPUNormalizePolygons
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPointIndex} from '../../pick';
import {createPolygonSet, restartRings, uploadPolygonSet} from './shared/polygon-set';
import {createStaleGuard, getBinCenter, getBinIndex, readFresh} from './shared/pointer-support';
import {addCircleSegments, addPairSegments} from './shared/shape-segments';
import {parameters} from './canonical-polygon-equality.parameters';

/** tolerance is a constructor prop: the pointer settles on one of these prebuilt values. */
const TOLERANCES = [0.002, 0.004, 0.006, 0.009, 0.012];
/** Displacements and the tolerance ring are drawn this many times larger. */
const MAGNIFICATION = 12;
const STEP = 0.0008;

export default defineOperationRun({
  parameters,
  caption:
    'Polygons equal within the tolerance, after canonical normalization (displacements drawn 12x).',
  buildCache: 5,
  createInputs: ({seed, world}) => {
    const set = createPolygonSet(makeRandom(seed), world, {rows: 3, columns: 5});
    // The vertex that is displaced in every polygon of the right set.
    const vertices = Float32Array.from({length: set.count * 2}, (_, component) => {
      const feature = component >> 1;
      const ring = set.polygonOffsets[set.featureOffsets[feature]];
      const first = set.ringOffsets[ring];
      // restartRings starts ring r at vertex 3r + 2, and that first vertex is the displaced one.
      const row = first + ((ring * 3 + 2) % (set.ringOffsets[ring + 1] - first));
      return set.positions[row * 2 + (component & 1)];
    });
    return {
      set,
      vertices,
      index: createPointIndex(set.centers),
      guard: createStaleGuard()
    };
  },
  build(builder, inputs, values) {
    const {set, vertices} = inputs;
    const left = uploadPolygonSet(builder, set);
    const right = builder.upload('right-input', 'float32x2', set.positions);
    const vertexCount = set.positions.length / 2;
    const normalize = (id: string, positions: typeof right) => {
      const output = {
        positions: builder.allocate(`${id}-positions`, 'float32x2', vertexCount),
        ringOffsets: builder.allocate(`${id}-rings`, 'uint32', set.ringOffsets.length),
        polygonOffsets: builder.allocate(`${id}-polygons`, 'uint32', set.polygonOffsets.length)
      };
      builder.graph.add(
        new GPUNormalizePolygons({
          id,
          positions: positions.view,
          featureOffsets: left.featureOffsets.view,
          polygonOffsets: left.polygonOffsets.view,
          ringOffsets: left.ringOffsets.view,
          output: {
            positions: output.positions.view,
            ringOffsets: output.ringOffsets.view,
            polygonOffsets: output.polygonOffsets.view
          }
        })
      );
      return {
        positions: output.positions.view,
        ringOffsets: output.ringOffsets.view,
        polygonOffsets: output.polygonOffsets.view,
        featureOffsets: left.featureOffsets.view
      };
    };
    const equal = builder.allocate('equal', 'uint32', set.count);
    builder.graph.add(
      new GPUCanonicalPolygonEquality({
        left: normalize('left', left.positions),
        right: normalize('right', right),
        tolerance: values.tolerance,
        output: equal.view
      })
    );
    // A ring of the tolerance at each displaced vertex, and a line to where that vertex went.
    const rings = builder.upload(
      'tolerance-rings',
      'float32x3',
      Float32Array.from({length: set.count * 3}, (_, component) =>
        component % 3 === 2
          ? values.tolerance * MAGNIFICATION
          : vertices[Math.floor(component / 3) * 2 + (component % 3)]
      )
    );
    const anchors = builder.upload('anchors', 'float32x2', vertices);
    const ghosts = builder.allocate('ghosts', 'float32x2', set.count);
    return {
      outputs: {
        ...left,
        equal,
        ghosts,
        rings: addCircleSegments(builder, 'tolerance', rings),
        links: addPairSegments(builder, 'displacement', anchors, ghosts)
      },
      update: current => {
        ghosts.write(
          Float32Array.from({length: set.count * 2}, (_, component) =>
            component & 1
              ? vertices[component]
              : vertices[component] + (component >> 1) * STEP * current.displacement * MAGNIFICATION
          )
        );
        const restarted = restartRings(
          set,
          ring => ring * 3 + 2,
          ring => ring % 2 === 1
        );
        // Polygon f has its first vertex moved f * 0.0008 * displacement (the ring layout is unchanged).
        for (let feature = 0; feature < set.count; feature++) {
          const vertex = set.ringOffsets[set.polygonOffsets[set.featureOffsets[feature]]];
          restarted[vertex * 2] += feature * STEP * current.displacement;
        }
        right.write(restarted);
      }
    };
  },
  rebuildOn: ['tolerance'],
  // The pointer's x picks the tolerance (five prebuilt values); the hovered polygon's ring shows the test.
  interaction: {
    kind: 'level',
    hint: 'loosen the tolerance',
    rest: [getBinCenter(TOLERANCES.indexOf(0.006), 1, TOLERANCES.length), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const tolerance = TOLERANCES[getBinIndex(pointer.unitX, 1, TOLERANCES.length)];
      const feature = inputs.index.nearest(pointer.x, pointer.y);
      const equal = readFresh(inputs.guard, String(values.tolerance), peek(outputs.equal));
      let equalCount = 0;
      if (equal) for (const flag of equal) equalCount += flag;
      const anchor: [number, number] = [
        inputs.vertices[feature * 2],
        inputs.vertices[feature * 2 + 1]
      ];
      return {
        settle: {tolerance},
        glyphs:
          feature >= 0
            ? [
                {shape: 'ring', at: anchor, radius: tolerance * MAGNIFICATION},
                {shape: 'dot', at: anchor}
              ]
            : [],
        readout:
          tolerance === values.tolerance
            ? `tolerance ${tolerance.toFixed(3)} · ${equal ? equalCount : '...'} of ${inputs.set.count} equal`
            : `tolerance ${values.tolerance.toFixed(3)} -> ${tolerance.toFixed(3)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.equal,
      triangleCount: inputs.set.fillTriangleCount,
      categories: [
        [1, 0.55, 0.25],
        [0.45, 0.55, 0.72]
      ]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.rings, radius: 1.2},
    {layer: 'lines', segments: outputs.links, radius: 1.6},
    {layer: 'dots', positions: outputs.ghosts, radius: 3.5, color: [1, 0.55, 0.25, 1]}
  ]
});
