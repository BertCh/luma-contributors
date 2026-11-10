// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeometryCleanup snaps a densely sampled line to a grid and drops the repeated vertices that
// creates. The muted path is the input; the bright path is the compacted output, drawn with the
// vertex count the operation wrote.

import {
  getGPUGeometryCleanupParameterValues,
  GPUGeometryCleanup
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp, findParameter, snapToParameter} from '../../interaction';
import {findNearestOnPolyline} from '../../pick';
import {parameters} from './geometry-cleanup.parameters';
import {addJitter, createIdentityIds, createSingleLine} from './shared/polygon-set';
import {findPointAtDistance, getLatticeGlyphs} from './shared/pointer-support';
import {addMaskedPoints, allocateLattice, createLatticeSegments} from './shared/shape-segments';

const GRID_SIZE = findParameter(parameters, 'gridSize');
/** Distance from the line that maps to the largest gridSize. */
const REACH = 0.3;

export default defineOperationRun({
  parameters,
  caption: 'A dense line snapped to a grid, with repeated vertices removed.',
  createInputs: ({random, world}) => {
    const line = createSingleLine(random, world, 160);
    return {
      line,
      world,
      noisy: addJitter(line.positions, random, 0.002),
      restPoint: findPointAtDistance(
        (x, y) => findNearestOnPolyline(line.positions, x, y)?.distance ?? null,
        (GRID_SIZE.default / GRID_SIZE.max) * REACH,
        world
      )
    };
  },
  build(builder, {line, noisy, world}) {
    const lattice = allocateLattice(builder, 'lattice', GRID_SIZE.default, world);
    const positions = builder.upload('positions', 'float32x2', noisy);
    const lineOffsets = builder.upload('line-offsets', 'uint32', line.lineOffsets);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const cleaned = builder.allocate('cleaned', 'float32x2', line.count);
    const cleanedOffsets = builder.allocate('cleaned-offsets', 'uint32', 2);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const requiredCount = builder.allocate('required-count', 'uint32', 1);
    builder.graph.add(
      new GPUGeometryCleanup({
        geometryType: 'lines',
        positions: positions.view,
        lineOffsets: lineOffsets.view,
        parameters: parameterBuffer.view,
        output: {
          positions: cleaned.view,
          lineOffsets: cleanedOffsets.view,
          status: {count: count.view, overflow: overflow.view, requiredCount: requiredCount.view}
        }
      })
    );
    return {
      outputs: {
        positions,
        cleaned,
        count,
        lattice,
        kept: addMaskedPoints(builder, 'kept', cleaned, count),
        ids: builder.upload('ids', 'uint32', createIdentityIds(line.count))
      },
      update: values => {
        parameterBuffer.write(
          getGPUGeometryCleanupParameterValues({
            gridSize: values.gridSize,
            tolerance: values.tolerance,
            removeRepeatedPoints: true
          })
        );
        lattice.write(createLatticeSegments(values.gridSize, world));
      }
    };
  },
  // The pointer's distance from the line is the gridSize, and the lattice shows where the pointer lands.
  interaction: {
    kind: 'reach',
    hint: 'move away from the line',
    rest: ({inputs, toUnit}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, outputs, peek}) {
      const hit = findNearestOnPolyline(inputs.line.positions, pointer.x, pointer.y);
      const gridSize = snapToParameter(
        GRID_SIZE,
        clamp((hit?.distance ?? 0) / REACH, 0, 1) * GRID_SIZE.max
      );
      const kept = peek(outputs.count)?.[0];
      const point = [pointer.x, pointer.y] as const;
      return {
        values: {gridSize},
        glyphs: [
          ...(gridSize > 0 ? getLatticeGlyphs(point, gridSize) : []),
          ...(hit ? [{shape: 'line' as const, from: point, to: hit.point, dashed: true}] : [])
        ],
        readout: `gridSize ${gridSize.toFixed(3)} · ${inputs.line.count} -> ${kept ?? '...'} vertices`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.lattice, radius: 0.6, alpha: 0.3},
    {
      layer: 'paths',
      context: true,
      nodes: outputs.positions,
      ids: outputs.ids,
      count: inputs.line.count
    },
    {layer: 'paths', nodes: outputs.cleaned, ids: outputs.ids, count: outputs.count, radius: 2},
    {layer: 'dots', positions: outputs.kept, radius: 3.2}
  ]
});
