// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCoverageSimplification simplifies the shared boundaries of a polygon coverage once, so
// neighbors stay gap-free. The jagged original is muted; the colored rings are the simplified
// polygons. With `simplifyBoundary` false the outer boundary is left untouched.

import {
  getGPULineSimplificationParameterValues,
  GPUCoverageSimplification
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments} from './shared/lines';
import {
  createCoverage,
  createRingSegments,
  getPolygonFields,
  uploadPolygonSet
} from './shared/polygons';
import {parameters} from './coverage-simplification.parameters';

export default defineOperationRun({
  parameters,
  caption: 'A jagged coverage simplified along shared boundaries; no gaps open between neighbors.',
  createInputs: ({random, world}) =>
    createCoverage(random, world, {
      rows: 3,
      columns: 4,
      edgeVertexCount: 10,
      wiggle: 0.1,
      wiggleBoundary: true
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
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 3},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 1.8
    }
  ]
});
