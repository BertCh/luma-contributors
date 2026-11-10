// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCoverageValidity flags ring edges that break a polygon coverage: overlaps, crossings,
// mismatched shared boundaries and gaps narrower than `gapWidth`. Every ring edge is muted; the
// flagged edges are drawn thick in orange. The cells are shrunk by "Data: gap" (negative values
// overlap), so raising `gapWidth` above the real gap flags the strips between cells.

import {
  getGPUCoverageValidityParameterValues,
  GPUCoverageValidity
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {findParameter, snapToParameter} from '../../interaction';
import {createTractCoverage} from './shared/tract-coverage';
import {defineOperationRun} from '../../define-run';
import {
  addPackedSegments,
  findNearestLine,
  findRestBesideLines,
  HIGHLIGHT_COLOR
} from './shared/lines';
import {getPolygonFields, uploadPolygonSet} from './shared/polygons';
import {parameters} from './coverage-validity.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Coverage edges flagged as invalid (orange): overlaps and gaps narrower than gapWidth.',
  createInputs: ({random, world}) => {
    const coverage = createTractCoverage(random, world, 4);
    // Only a few cells are defective: a sliver gap, a wide gap and an overlap, scaled by "Data: gap".
    // Every other cell touches its neighbors exactly.
    const defects = new Map([
      [Math.floor(coverage.featureCount * 0.2), 1],
      [Math.floor(coverage.featureCount * 0.5), 4],
      [Math.floor(coverage.featureCount * 0.8), -0.8]
    ]);
    const getSet = (gap: number) => {
      const positions = new Float32Array(coverage.positions);
      for (let feature = 0; feature < coverage.featureCount; feature++) {
        if (!defects.has(feature)) continue;
        const shrink = gap * defects.get(feature)!;
        const x = coverage.centers[feature * 2];
        const y = coverage.centers[feature * 2 + 1];
        const start = coverage.ringOffsets[feature];
        const end = coverage.ringOffsets[feature + 1];
        let extent = 0;
        for (let row = start; row < end; row++) {
          extent = Math.max(extent, Math.hypot(positions[row * 2] - x, positions[row * 2 + 1] - y));
        }
        const scale = 1 - shrink / (2 * extent);
        for (let row = start; row < end; row++) {
          positions[row * 2] = x + (positions[row * 2] - x) * scale;
          positions[row * 2 + 1] = y + (positions[row * 2 + 1] - y) * scale;
        }
      }
      return {...coverage, positions};
    };
    return {getSet, world};
  },
  build(builder, inputs, values) {
    const initial = inputs.getSet(values.gap);
    const polygons = uploadPolygonSet(builder, 'coverage', initial);
    const vertexCount = initial.positions.length / 2;
    const segmentFlags = builder.allocate('segment-flags', 'uint32', vertexCount);
    const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
    const isValid = builder.allocate('is-valid', 'uint32', 1);
    const parameterBuffer = builder.allocate('gap-width', 'float32', 4);
    builder.graph.add(
      new GPUCoverageValidity({
        ...getPolygonFields(polygons),
        parameters: parameterBuffer.view,
        output: {
          segmentFlags: segmentFlags.view,
          status: {invalidCount: invalidCount.view},
          isValid: isValid.view
        }
      })
    );
    const drawing = addPackedSegments(
      builder,
      'coverage',
      polygons.positions,
      polygons.ringOffsets,
      {
        closed: true,
        flags: segmentFlags
      }
    );
    return {
      outputs: {...drawing, flagged: drawing.segments},
      update: current => {
        polygons.positions.write(inputs.getSet(current.gap).positions);
        parameterBuffer.write(getGPUCoverageValidityParameterValues({gapWidth: current.gapWidth}));
      }
    };
  },
  // The distance from the pointer to the nearest coverage edge, doubled, is gapWidth: a disc of that
  // diameter touching the edge shows which gaps are narrower than it (those edges turn orange).
  interaction: {
    kind: 'reach',
    hint: 'move beside an edge',
    rest: ({inputs, values, toUnit}) => {
      const set = inputs.getSet(values.gap);
      const lines = {positions: set.positions, lineOffsets: set.ringOffsets};
      return toUnit(
        findRestBesideLines(
          lines,
          findParameter(parameters, 'gapWidth').default / 2,
          (x, y) => findNearestLine(lines, x, y, true)?.distance ?? 0,
          inputs.world,
          1e-3
        )
      );
    },
    resolve(pointer, {inputs, values}) {
      const set = inputs.getSet(values.gap);
      const hit = findNearestLine(
        {positions: set.positions, lineOffsets: set.ringOffsets},
        pointer.x,
        pointer.y,
        true
      );
      if (!hit) return {};
      const gapWidth = snapToParameter(findParameter(parameters, 'gapWidth'), hit.distance * 2);
      return {
        values: {gapWidth},
        glyphs: [
          {shape: 'ring', at: [pointer.x, pointer.y], radius: hit.distance},
          {shape: 'dot', at: hit.point}
        ],
        readout: `gapWidth ${gapWidth.toFixed(3)} · gap ${values.gap.toFixed(3)}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'segments', context: true, nodes: outputs.points, edges: outputs.edges, radius: 1.2},
    {layer: 'lines', segments: outputs.flagged, color: HIGHLIGHT_COLOR, radius: 3.5}
  ]
});
