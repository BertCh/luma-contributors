// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCurvature on a Grand Canyon DEM: the chosen curvature of the surface per cell. Profile
// curvature is along the slope (acceleration), plan curvature across it (convergence); the
// differencing method is a construction prop, the z factor is written per frame.

import {
  getGPUTerrainCurvatureParameterValues,
  GPUTerrainCurvature,
  type GPUTerrainCurvatureKind,
  type GPUTerrainCurvatureMethod,
  type GPUTerrainCurvatureOutput
} from '@luma-contributors/analysis/gpu-terrain';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {defineOperationRun} from '../../define-run';
import {
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getCellWorldPosition,
  getWorldPerCell,
  uploadTerrain
} from './support/terrain';
import {parameters} from './terrain-curvature.parameters';

const METHODS: GPUTerrainCurvatureMethod[] = ['evans-young', 'zevenbergen-thorne', 'florinsky'];
const KINDS: GPUTerrainCurvatureKind[] = ['profile', 'plan', 'mean', 'gaussian'];

/** One selected curvature output; the class requires at least one named kind. */
function getCurvatureOutput(
  kind: GPUTerrainCurvatureKind,
  view: GraphDataView<'float32'>
): GPUTerrainCurvatureOutput {
  switch (kind) {
    case 'profile':
      return {curvatures: {profile: view}};
    case 'plan':
      return {curvatures: {plan: view}};
    case 'mean':
      return {curvatures: {mean: view}};
    case 'gaussian':
      return {curvatures: {gaussian: view}};
    default:
      throw new Error(`terrain-curvature run: unsupported curvature kind ${String(kind)}`);
  }
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['method', 'curvature'],
  caption: 'Elevation in; the surface curvature out: convex ridges against concave valleys.',
  createInputs: ({world}) => createTerrainInput(world),
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const curvature = builder.allocate('curvature', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainCurvature({
        grid,
        elevation: band,
        parameters: parameterBuffer.view,
        method: METHODS[values.method],
        output: getCurvatureOutput(KINDS[values.curvature], curvature.view)
      })
    );
    return {
      outputs: {elevation, curvature},
      update: current =>
        parameterBuffer.write(
          getGPUTerrainCurvatureParameterValues({zFactor: current.zFactor}, grid)
        )
    };
  },
  // The pointer is a cell: a short line lies along the slope (profile, mean, Gaussian) or across
  // it (plan), the direction in which the reported curvature is measured.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const at = (columnOffset: number, rowOffset: number) =>
        inputs.elevation[
          Math.min(inputs.rows - 1, Math.max(0, row + rowOffset)) * inputs.columns +
            Math.min(inputs.columns - 1, Math.max(0, column + columnOffset))
        ];
      // Uphill direction in world coordinates (x right, y up).
      const gradientX = at(1, 0) - at(-1, 0);
      const gradientY = at(0, -1) - at(0, 1);
      const length = Math.hypot(gradientX, gradientY) || 1;
      const isPlan = KINDS[values.curvature] === 'plan';
      const directionX = isPlan ? -gradientY / length : gradientX / length;
      const directionY = isPlan ? gradientX / length : gradientY / length;
      const center = getCellWorldPosition(inputs, world, column, row);
      const reach = 2.5 * getWorldPerCell(inputs, world);
      const curvature = peek(outputs.curvature);
      const value = curvature ? curvature[row * inputs.columns + column] : Number.NaN;
      return {
        glyphs: [
          {
            shape: 'line',
            from: [center[0] - directionX * reach, center[1] - directionY * reach],
            to: [center[0] + directionX * reach, center[1] + directionY * reach]
          },
          {shape: 'dot', at: center}
        ],
        readout: Number.isFinite(value)
          ? `${KINDS[values.curvature]} ${value.toFixed(4)} /m · ${value > 0 ? 'convex' : 'concave'}`
          : `${KINDS[values.curvature]} curvature`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.curvature, {
      ramp: 'diverging',
      range: [-0.003, 0.003],
      smoothing: true
    })
  ]
});
