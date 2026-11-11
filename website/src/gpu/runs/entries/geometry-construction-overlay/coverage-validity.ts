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
import {createTriangleListDrawing, getPolygonFields, uploadPolygonSet} from './shared/polygons';
import {clipEars} from './shared/ear-clip';
import {createLegend} from '../../text';
import {parameters} from './coverage-validity.parameters';

/**
 * The cells as a triangle list for drawing the input, from vertex rows triangulated once on the
 * undistorted cells (shrinking a cell scales it about its center, which keeps those triangles
 * valid). A gap between two cells then stays the background color and an overlap is covered twice.
 */
function createCellTriangles(
  positions: Float32Array,
  triangleRows: readonly number[]
): Float32Array {
  const triangles = new Float32Array(triangleRows.length * 2);
  triangleRows.forEach((row, slot) => {
    triangles[slot * 2] = positions[row * 2];
    triangles[slot * 2 + 1] = positions[row * 2 + 1];
  });
  return triangles;
}

export default defineOperationRun({
  parameters,
  caption: 'Coverage edges flagged as invalid (orange): overlaps and gaps narrower than gapWidth.',
  createInputs: ({random, world}) => {
    const coverage = createTractCoverage(random, world, 4);
    // Only a few cells are defective: gaps of graded widths and an overlap, scaled by "Data: gap".
    // Every other cell touches its neighbors exactly.
    const defects = new Map<number, number>();
    [
      [0.05, 0.5],
      [0.2, 1],
      [0.35, 2],
      [0.5, 4],
      [0.65, 3],
      [0.8, -0.8],
      [0.95, 1.5]
    ].forEach(([share, multiplier]) =>
      defects.set(Math.floor(coverage.featureCount * share), multiplier)
    );
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
    const triangleRows: number[] = [];
    for (let ring = 0; ring + 1 < coverage.ringOffsets.length; ring++) {
      triangleRows.push(
        ...clipEars(coverage.positions, coverage.ringOffsets[ring], coverage.ringOffsets[ring + 1])
      );
    }
    return {getSet, world, triangleRows};
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
    const fill = builder.upload(
      'cell-fill',
      'float32x2',
      createCellTriangles(initial.positions, inputs.triangleRows)
    );
    const fillMesh = createTriangleListDrawing(builder, 'cell-fill', fill.length);
    return {
      outputs: {...drawing, flagged: drawing.segments, fill, fillMesh},
      update: current => {
        const set = inputs.getSet(current.gap);
        polygons.positions.write(set.positions);
        fill.write(createCellTriangles(set.positions, inputs.triangleRows));
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
          {shape: 'ring', at: [pointer.x, pointer.y], radius: gapWidth / 2},
          {shape: 'dot', at: hit.point}
        ],
        readout: `gapWidth ${gapWidth.toFixed(3)} · gap ${values.gap.toFixed(3)}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'polygons',
      context: true,
      positions: outputs.fill,
      indices: outputs.fillMesh.indices,
      owners: outputs.fillMesh.owners,
      values: outputs.fillMesh.values,
      triangleCount: outputs.fillMesh.triangleCount,
      color: [...theme.ink, 1],
      alpha: 0.2
    },
    {layer: 'segments', context: true, nodes: outputs.points, edges: outputs.edges, radius: 1.5},
    {layer: 'lines', segments: outputs.flagged, color: HIGHLIGHT_COLOR, radius: 3.5},
    {
      layer: 'text',
      ...createLegend({
        corner: 'bottom-left',
        entries: [
          {label: 'cell (gaps between cells show the background)', swatch: theme.muted},
          {
            label: 'edge flagged by GPUCoverageValidity',
            swatch: [HIGHLIGHT_COLOR[0], HIGHLIGHT_COLOR[1], HIGHLIGHT_COLOR[2]]
          }
        ]
      })
    }
  ]
});
