// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCoverageSimplification simplifies the shared boundaries of a polygon coverage once, so
// neighbors stay gap-free. The jagged original is muted; the colored rings are the simplified
// polygons. With `simplifyBoundary` false the outer boundary is left untouched.

import {
  getGPULineSimplificationParameterValues,
  GPUCoverageSimplification
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {findParameter, snapToParameter} from '../../interaction';
import {createWigglyCoverage} from './shared/tract-coverage';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments, findNearestLine, findRestBesideLines} from './shared/lines';
import {createRingSegments, getPolygonFields, uploadPolygonSet} from './shared/polygons';
import {parameters} from './coverage-simplification.parameters';

export default defineOperationRun({
  parameters,
  caption: 'A jagged coverage simplified along shared boundaries; no gaps open between neighbors.',
  createInputs: ({world}) => ({
    ...createWigglyCoverage(world, Math.max(3, Math.round(world.width * 2.5)), 3, 0.03),
    world
  }),
  build(builder, coverage, values) {
    const polygons = uploadPolygonSet(builder, 'coverage', coverage);
    const outputPositions = builder.allocate(
      'simplified-positions',
      'float32x2',
      coverage.positions.length / 2
    );
    const outputRingOffsets = builder.allocate(
      'simplified-ring-offsets',
      'uint32',
      coverage.ringOffsets.length
    );
    const overflow = builder.allocate('simplified-overflow', 'uint32', 1);
    const parameterBuffer = builder.allocate('tolerance', 'float32', 4);
    builder.graph.add(
      new GPUCoverageSimplification({
        ...getPolygonFields(polygons),
        parameters: parameterBuffer.view,
        simplifyBoundary: values.simplifyBoundary === 1,
        output: {
          positions: outputPositions.view,
          ringOffsets: outputRingOffsets.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const drawing = addPackedSegments(builder, 'simplified', outputPositions, outputRingOffsets, {
      closed: true
    });
    const original = builder.upload('original', 'float32x4', createRingSegments(coverage));
    return {
      outputs: {original, ...drawing},
      update: current =>
        parameterBuffer.write(
          getGPULineSimplificationParameterValues({tolerance: current.tolerance})
        )
    };
  },
  rebuildOn: ['simplifyBoundary'],
  // The distance from the pointer to the nearest original boundary is the tolerance: boundaries
  // that deviate less than that from a straight run are simplified away.
  interaction: {
    kind: 'reach',
    hint: 'move off a boundary',
    rest: ({inputs, toUnit}) => {
      const lines = {positions: inputs.positions, lineOffsets: inputs.ringOffsets};
      return toUnit(
        findRestBesideLines(
          lines,
          findParameter(parameters, 'tolerance').default,
          (x, y) => findNearestLine(lines, x, y, true)?.distance ?? 0,
          inputs.world,
          1e-3
        )
      );
    },
    resolve(pointer, {inputs, values}) {
      const hit = findNearestLine(
        {positions: inputs.positions, lineOffsets: inputs.ringOffsets},
        pointer.x,
        pointer.y,
        true
      );
      if (!hit) return {};
      const tolerance = snapToParameter(findParameter(parameters, 'tolerance'), hit.distance);
      return {
        values: {tolerance},
        glyphs: [
          {shape: 'ring', at: hit.point, radius: tolerance, dashed: true},
          {shape: 'line', from: hit.point, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: hit.point}
        ],
        readout: `tolerance ${tolerance.toFixed(3)} · boundary ${values.simplifyBoundary === 1 ? 'simplified' : 'kept'}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 2},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2.6
    }
  ]
});
