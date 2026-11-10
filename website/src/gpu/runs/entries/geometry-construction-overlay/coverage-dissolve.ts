// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCoverageDissolve merges the cells of a coverage that share a label into one boundary per
// district by dropping every internal shared edge. Cell outlines are muted; the colored rings are
// the dissolved districts.

import {GPUCoverageDissolve} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createTractCoverage, groupNeighborTracts} from './shared/tract-coverage';
import {defineOperationRun} from '../../define-run';
import {addPackedSegments} from './shared/lines';
import type {RunGlyph} from '../../interaction';
import {isInsideRing} from '../../pick';
import {createRingSegments, getPolygonFields, uploadPolygonSet} from './shared/polygons';
import {parameters} from './coverage-dissolve.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Cells merged by district label; internal boundaries disappear.',
  createInputs: ({random, world}) => createTractCoverage(random, world, 5),
  build(builder, coverage, values) {
    const polygons = uploadPolygonSet(builder, 'coverage', coverage);
    const labels = builder.allocate('labels', 'uint32', coverage.featureCount);
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
        labels.write(groupNeighborTracts(coverage.contiguity, current.districtSize));
      }
    };
  },
  // The pointer picks a tract: its district's member tracts are outlined (their shared edges are
  // the internal boundaries the operation drops) and the hovered tract is solid.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    resolve(pointer, {inputs, values}) {
      let row = -1;
      for (let feature = 0; feature < inputs.featureCount && row < 0; feature++) {
        const first = inputs.ringOffsets[feature];
        const count = inputs.ringOffsets[feature + 1] - first;
        if (isInsideRing(inputs.positions, pointer.x, pointer.y, first, count)) row = feature;
      }
      if (row < 0) return {};
      const labels = groupNeighborTracts(inputs.contiguity, values.districtSize);
      const glyphs: RunGlyph[] = [];
      let memberCount = 0;
      let internalEdgeCount = 0;
      for (let feature = 0; feature < inputs.featureCount; feature++) {
        if (labels[feature] !== labels[row]) continue;
        memberCount++;
        glyphs.push({
          shape: 'polyline',
          points: Array.from(
            inputs.positions.subarray(
              inputs.ringOffsets[feature] * 2,
              inputs.ringOffsets[feature + 1] * 2
            )
          ),
          closed: true,
          dashed: feature !== row
        });
        for (
          let slot = inputs.contiguity.offsets[feature];
          slot < inputs.contiguity.offsets[feature + 1];
          slot++
        ) {
          if (labels[inputs.contiguity.neighbors[slot]] === labels[row]) internalEdgeCount++;
        }
      }
      return {
        focus: row,
        glyphs,
        readout: `district ${labels[row]} · ${memberCount} tracts · ${internalEdgeCount / 2} borders dropped`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.original, radius: 1.5},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 4
    }
  ]
});
