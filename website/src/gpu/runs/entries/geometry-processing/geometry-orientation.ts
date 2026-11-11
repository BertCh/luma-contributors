// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryOrientation forces polygon winding. Half the input rings wind each way; a large arrow
// on every ring shows its winding, and rings are colored by the `reversedRings` flag the operation
// wrote (orange: wound against the request and reversed). To the left of the curtain the arrows
// disagree; to the right they all point the requested way.

import {
  getGPUGeometryOrientationParameterValues,
  GPUGeometryOrientation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {createLegend} from '../../text';
import {createPolygonSet, createRingOfVertex, uploadPolygonSet} from './shared/polygon-set';
import {addKernel, addRingArrows} from './shared/shape-segments';
import {parameters} from './geometry-orientation.parameters';

const ARROW_SCALE = 1;

export default defineOperationRun({
  parameters,
  caption: 'Rings the operation had to reverse are orange; rings it kept are blue.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {mixedWinding: true, hasHoles: true})
  }),
  build(builder, inputs) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const ringCount = inputs.set.ringOffsets.length - 1;
    const oriented = builder.allocate('oriented', 'float32x2', inputs.set.positions.length / 2);
    const reversedRings = builder.allocate('reversed-rings', 'uint32', ringCount);
    builder.graph.add(
      new GPUGeometryOrientation({
        mode: 'orient-polygons',
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        parameters: parameterBuffer.view,
        output: {positions: oriented.view, reversedRings: reversedRings.view}
      })
    );
    // Spread each ring's flag onto its vertices so the edges can be colored by it.
    const ringOfVertex = builder.upload(
      'ring-of-vertex',
      'uint32',
      createRingOfVertex(inputs.set.ringOffsets)
    );
    const reversedVertices = builder.allocate('reversed-vertices', 'uint32', ringOfVertex.length);
    addKernel(builder, {
      id: 'spread-reversed',
      invocationCount: ringOfVertex.length,
      bindings: [
        {name: 'rings', buffer: reversedRings, access: 'read'},
        {name: 'ringOfVertex', buffer: ringOfVertex, access: 'read'},
        {name: 'flags', buffer: reversedVertices, access: 'read_write'}
      ],
      body: 'flags[flagsOffset + index] = rings[ringsOffset + ringOfVertex[ringOfVertexOffset + index]];'
    });
    return {
      outputs: {
        ...polygons,
        oriented,
        reversedRings,
        reversedVertices,
        inputArrows: addRingArrows(
          builder,
          'input',
          polygons.positions,
          polygons.ringOffsets,
          ringCount,
          ARROW_SCALE
        ),
        orientedArrows: addRingArrows(
          builder,
          'oriented',
          oriented,
          polygons.ringOffsets,
          ringCount,
          ARROW_SCALE
        )
      },
      update: values =>
        parameterBuffer.write(
          getGPUGeometryOrientationParameterValues({
            exteriorClockwise: values.exteriorClockwise === 1
          })
        )
    };
  },
  // The curtain splits the picture: the input rings and their winding to the left, the oriented output to the right.
  interaction: {
    kind: 'swipe',
    hint: 'swipe input to output',
    rest: [0.5, 0.5],
    resolve(pointer, {world, outputs, values, peek}) {
      const curtain = clamp(pointer.x, 0, world.width);
      const reversed = peek(outputs.reversedRings);
      let reversedCount = 0;
      if (reversed) for (const flag of reversed) reversedCount += flag;
      return {
        curtain,
        glyphs: [{shape: 'curtain', x: curtain}],
        readout: `exteriorClockwise ${values.exteriorClockwise === 1} · ${reversed ? reversedCount : '...'} rings reversed`
      };
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      {
        layer: 'lines',
        context: true,
        segments: outputs.outline,
        count: inputs.set.outlineCount,
        radius: 1.4,
        clip: {side: 'left', x: curtain}
      },
      // The input rings, colored by whether the operation had to reverse them.
      {
        layer: 'segments',
        nodes: outputs.positions,
        edges: outputs.edges,
        values: outputs.reversedVertices,
        radius: 2.4,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'lines',
        segments: outputs.inputArrows,
        radius: 2.2,
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'segments',
        nodes: outputs.oriented,
        edges: outputs.edges,
        values: outputs.reversedVertices,
        radius: 2.4,
        clip: {side: 'right', x: curtain}
      },
      {
        layer: 'lines',
        segments: outputs.orientedArrows,
        radius: 2.2,
        clip: {side: 'right', x: curtain}
      },
      {
        layer: 'text',
        ...createLegend({
          corner: 'bottom-left',
          inset: 2,
          entries: [
            {label: 'reversed', swatch: 1},
            {label: 'kept', swatch: 0}
          ]
        })
      }
    ];
  }
});
