// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCoverageDissolve merges the cells of a coverage that share a label into one boundary per
// district by dropping every internal shared edge. Cell outlines are muted; the colored rings are
// the dissolved districts.

import {GPUCoverageDissolve} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments} from './shared/lines';
import {
  createCoverage,
  createRingSegments,
  getPolygonFields,
  uploadPolygonSet
} from './shared/polygons';
import {parameters} from './coverage-dissolve.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Cells merged by district label; internal boundaries disappear.',
  createInputs: ({random, world}) =>
    createCoverage(random, world, {rows: 4, columns: 6, edgeVertexCount: 3, wiggle: 0.08}),
  build(builder, coverage, values) {
    const polygons = uploadPolygonSet(builder, 'coverage', coverage);
    const labels = builder.upload('labels', 'uint32', coverage.labels);
    const ringCapacity = coverage.ringOffsets.length;
    const vertexCapacity = coverage.positions.length / 2 + ringCapacity;
    const ringOffsets = builder.allocate('dissolved-ring-offsets', 'uint32', ringCapacity + 1);
    const positions = builder.allocate('dissolved-positions', 'float32x2', vertexCapacity);
    const ringAreas = builder.allocate('dissolved-areas', 'float32', ringCapacity);
    const ringGroups = builder.allocate('dissolved-groups', 'uint32', ringCapacity);
    const count = builder.allocate('dissolved-count', 'uint32', 1);
    const overflow = builder.allocate('dissolved-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUCoverageDissolve({
        ...getPolygonFields(polygons),
        labels: labels.view,
        snapTolerance: 1e-5,
        output: {
          ringOffsets: ringOffsets.view,
          positions: positions.view,
          ringAreas: ringAreas.view,
          ringGroups: ringGroups.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    const drawing = addPackedSegments(builder, 'dissolved', positions, ringOffsets, {
      closed: true,
      lineCount: count,
      lineValues: ringGroups
    });
    const original = builder.upload('original', 'float32x4', createRingSegments(coverage));
    return {
      outputs: {original, ...drawing},
      update: current => {
        const size = current.districtSize;
        labels.write(
          Uint32Array.from(coverage.labels, (_, cell) => {
            const row = Math.floor(cell / coverage.columns);
            const column = cell % coverage.columns;
            return Math.floor(row / size) * 3 + Math.floor(column / size);
          })
        );
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 1.5},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 3
    }
  ]
});
