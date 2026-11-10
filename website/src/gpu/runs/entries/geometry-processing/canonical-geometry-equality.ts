// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCanonicalGeometryEquality compares two normalized collections leaf by leaf. The right tree
// lists its children in another order, rotates the ring starts and displaces leaf k by a growing
// amount. Each leaf is colored by the equality flag the operation wrote for its pair, and the
// tolerance decides how many leaves still count as equal.

import {
  GPUCanonicalGeometryEquality,
  GPUNormalizeGeometry
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createPointIndex} from '../../pick';
import {createGeometryTree} from './shared/geometry-tree';
import {createStaleGuard, getBinCenter, getBinIndex, readFresh} from './shared/pointer-support';
import {addCircleSegments, addKernel, addPairSegments} from './shared/shape-segments';
import {parameters} from './canonical-geometry-equality.parameters';

/** tolerance is a constructor prop: the pointer settles on one of these prebuilt values. */
const TOLERANCES = [0.002, 0.004, 0.006, 0.01, 0.015];
/** Displacements and the tolerance ring are drawn this many times larger. */
const MAGNIFICATION = 12;

export default defineOperationRun({
  parameters,
  caption:
    'Leaves equal within the tolerance, after both trees are normalized (displacements drawn 12x).',
  buildCache: 5,
  createInputs: ({world}) => {
    const left = createGeometryTree(world, {shuffleSeed: 1, start: 0});
    // The first vertex of every leaf: where the tolerance ring and the displaced ghost are drawn.
    const anchors = Float32Array.from({length: left.leafCount * 2}, (_, component) => {
      const leaf = component >> 1;
      return left.positions[left.nodes[leaf * 4 + 1] * 2 + (component & 1)];
    });
    return {
      world,
      left,
      anchors,
      index: createPointIndex(anchors),
      guard: createStaleGuard()
    };
  },
  build(builder, {world, left, anchors}, values) {
    const normalize = (
      id: string,
      tree: typeof left,
      positions: ReturnType<typeof builder.upload<'float32x2'>>
    ) => {
      const nodes = builder.upload(`${id}-nodes`, 'uint32x4', tree.nodes);
      const children = builder.upload(`${id}-children`, 'uint32', tree.children);
      const output = {
        positions: builder.allocate(`${id}-normalized`, 'float32x2', tree.positions.length / 2),
        children: builder.allocate(`${id}-normalized-children`, 'uint32', tree.children.length)
      };
      builder.graph.add(
        new GPUNormalizeGeometry({
          id,
          positions: positions.view,
          nodes: nodes.view,
          children: children.view,
          output: {
            positions: output.positions.view,
            children: output.children.view,
            status: builder.allocate(`${id}-status`, 'uint32', tree.nodes.length / 4).view
          }
        })
      );
      return {
        tree: {positions: output.positions.view, nodes: nodes.view, children: output.children.view},
        positions: output.positions
      };
    };
    const leftPositions = builder.upload('left-positions', 'float32x2', left.positions);
    const rightPositions = builder.upload('right-positions', 'float32x2', left.positions);
    const leftTree = normalize('left', left, leftPositions);
    const rightTree = normalize('right', left, rightPositions);
    // One pair per leaf (same leaf row on both sides), then the roots.
    const pairs = Uint32Array.from([
      ...Array.from({length: left.leafCount}, (_, leaf) => [leaf, leaf]).flat(),
      left.rootId,
      left.rootId
    ]);
    const equal = builder.allocate('equal', 'uint32', left.leafCount + 1);
    builder.graph.add(
      new GPUCanonicalGeometryEquality({
        left: leftTree.tree,
        right: rightTree.tree,
        pairs: builder.upload('pairs', 'uint32x2', pairs).view,
        tolerance: values.tolerance,
        output: equal.view
      })
    );
    const leafOfVertex = builder.upload('leaf-of-vertex', 'uint32', left.leafOfVertex);
    const vertexEqual = builder.allocate('vertex-equal', 'uint32', left.leafOfVertex.length);
    addKernel(builder, {
      id: 'vertex-equal',
      invocationCount: left.leafOfVertex.length,
      bindings: [
        {name: 'equal', buffer: equal, access: 'read'},
        {name: 'leafOfVertex', buffer: leafOfVertex, access: 'read'},
        {name: 'vertexEqual', buffer: vertexEqual, access: 'read_write'}
      ],
      body: 'vertexEqual[vertexEqualOffset + index] = equal[equalOffset + leafOfVertex[leafOfVertexOffset + index]];'
    });
    const anchorBuffer = builder.upload('anchors', 'float32x2', anchors);
    const ghosts = builder.allocate('ghosts', 'float32x2', left.leafCount);
    const rings = builder.upload(
      'tolerance-rings',
      'float32x3',
      Float32Array.from({length: left.leafCount * 3}, (_, component) =>
        component % 3 === 2
          ? values.tolerance * MAGNIFICATION
          : anchors[Math.floor(component / 3) * 2 + (component % 3)]
      )
    );
    return {
      outputs: {
        equal,
        ghosts,
        rings: addCircleSegments(builder, 'tolerance', rings),
        links: addPairSegments(builder, 'displacement', anchorBuffer, ghosts),
        positions: leftPositions,
        vertexEqual,
        edges: builder.upload('edges', 'uint32x2', left.edges)
      },
      update: current => {
        ghosts.write(
          Float32Array.from({length: left.leafCount * 2}, (_, component) =>
            component & 1
              ? anchors[component]
              : anchors[component] + (component >> 1) * current.displacement * MAGNIFICATION
          )
        );
        const right = createGeometryTree(world, {
          shuffleSeed: 5,
          start: 3,
          displacement: current.displacement
        });
        rightPositions.write(right.positions);
      }
    };
  },
  rebuildOn: ['tolerance'],
  // The pointer's x picks the tolerance (five prebuilt values); the hovered leaf's ring shows the test.
  interaction: {
    kind: 'level',
    hint: 'loosen the tolerance',
    rest: [getBinCenter(TOLERANCES.indexOf(0.006), 1, TOLERANCES.length), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const tolerance = TOLERANCES[getBinIndex(pointer.unitX, 1, TOLERANCES.length)];
      const leaf = inputs.index.nearest(pointer.x, pointer.y);
      const equal = readFresh(inputs.guard, String(values.tolerance), peek(outputs.equal));
      let equalCount = 0;
      if (equal)
        for (let index = 0; index < inputs.left.leafCount; index++) equalCount += equal[index];
      const anchor: [number, number] = [inputs.anchors[leaf * 2], inputs.anchors[leaf * 2 + 1]];
      return {
        settle: {tolerance},
        glyphs:
          leaf >= 0
            ? [
                {shape: 'ring', at: anchor, radius: tolerance * MAGNIFICATION},
                {shape: 'dot', at: anchor}
              ]
            : [],
        readout:
          tolerance === values.tolerance
            ? `tolerance ${tolerance.toFixed(3)} · ${equal ? equalCount : '...'} of ${inputs.left.leafCount} equal`
            : `tolerance ${values.tolerance.toFixed(3)} -> ${tolerance.toFixed(3)}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'segments',
      nodes: outputs.positions,
      edges: outputs.edges,
      values: outputs.vertexEqual,
      categories: [
        [1, 0.55, 0.25],
        [0.45, 0.55, 0.72]
      ],
      radius: 3
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.vertexEqual,
      categories: [
        [1, 0.55, 0.25],
        [0.45, 0.55, 0.72]
      ],
      radius: 5
    },
    {layer: 'lines', segments: outputs.rings, radius: 1.2},
    {layer: 'lines', segments: outputs.links, radius: 1.6},
    {layer: 'dots', positions: outputs.ghosts, radius: 3.5, color: [1, 0.55, 0.25, 1]}
  ]
});
