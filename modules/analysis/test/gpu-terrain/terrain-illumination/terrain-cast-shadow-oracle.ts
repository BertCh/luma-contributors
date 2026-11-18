// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getTerrainSweepGroundLineGeometry,
  getTerrainSweepLineGeometry,
  TERRAIN_SWEEP_SLOPE_SCALE
} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {
  getRasterGroundCellSize,
  type GPURasterCellSizeMode
} from '../../../src/analysis-core/contracts/raster-grid';
import {
  computeTerrainHorizonSweepBruteForce,
  type SweepOracleOptions
} from './terrain-horizon-sweep-oracle';
import {getVisibleDiskFraction} from './terrain-horizon-oracle';

/** CPU oracle of the cast-shadow settings-source sweep, reading the same packed bytes. */
export function computeTerrainCastShadowOracle(
  options: {
    width: number;
    height: number;
    elevation: Float32Array;
    parameters: Float32Array;
    maximumRadius: number;
    parameterLayout?: 'standard' | 'sweep';
    cellSizeMode?: GPURasterCellSizeMode;
    rowDirection?: 'south' | 'north';
  },
  computeSweep = computeTerrainHorizonSweepBruteForce
): {horizonAngle: number[]; sunVisibility: number[]; validity: number[]} {
  const {width, height, parameters} = options;
  const shared = options.parameterLayout === 'sweep';
  if (parameters.length < (shared ? 32 : 16))
    throw new Error('Cast shadow oracle parameters are too short');
  const origin: readonly [number, number] = shared ? [parameters[18], parameters[19]] : [0, 0];
  const referenceCellSize: readonly [number, number] | undefined = shared
    ? [parameters[16], parameters[17]]
    : undefined;
  const mode = options.cellSizeMode ?? 'uniform';
  const northRows = options.rowDirection === 'north';
  const groundDirection: readonly [number, number] = [
    parameters[14],
    parameters[15] * (northRows ? -1 : 1)
  ];
  const reference =
    referenceCellSize ??
    getRasterGroundCellSize(
      mode,
      {
        cellSize: [parameters[0], parameters[1]],
        northEdge: parameters[4],
        southEdge: parameters[5]
      },
      (mode === 'uniform' ? 0 : Math.floor(height / 2)) + 0.5,
      height
    );
  const xMajor = parameters[10] !== 0;
  const slope = (parameters[11] * (northRows ? -1 : 1)) / TERRAIN_SWEEP_SLOPE_SCALE;
  const travel = parameters[12] * (northRows && !xMajor ? -1 : 1);
  // Preserve the packed integer slope on square cells; anisotropic cells use the packed ground
  // direction instead, exactly as the settings-source GPU kernel does after mirroring rows.
  let geometry = getTerrainSweepLineGeometry(
    xMajor ? [travel, slope * travel] : [slope * travel, travel],
    width,
    height,
    origin
  );
  if (Math.fround(reference[0]) !== Math.fround(reference[1])) {
    geometry = getTerrainSweepGroundLineGeometry(groundDirection, reference, width, height, origin);
  }
  const fixedSlope = geometry.slopeFixed / TERRAIN_SWEEP_SLOPE_SCALE;
  const pixelDirection: readonly [number, number] = geometry.xMajor
    ? [geometry.travelSign, fixedSlope * geometry.travelSign]
    : [fixedSlope * geometry.travelSign, geometry.travelSign];
  const sweep: SweepOracleOptions = {
    width,
    height,
    elevation: options.elevation,
    directionCount: 1,
    directions: [pixelDirection],
    cellSize: [parameters[0], parameters[1]],
    northEdge: parameters[4],
    southEdge: parameters[5],
    cellSizeMode: mode,
    maximumRadius: options.maximumRadius,
    zFactor: parameters[2],
    curvatureCoefficient: parameters[3],
    maximumDistance: parameters[6],
    origin,
    referenceCellSize,
    sourceGeoreference: shared
      ? {
          cellSize: [parameters[20], parameters[21]],
          northEdge: parameters[22],
          southEdge: parameters[23],
          height: parameters[24],
          rowOffset: parameters[25]
        }
      : undefined
  };
  const result = computeSweep(sweep);
  return {
    horizonAngle: result.horizon,
    validity: result.validity,
    sunVisibility: result.horizon.map(angle =>
      !Number.isFinite(angle)
        ? NaN
        : parameters[8] + parameters[9] <= 0
          ? 0
          : getVisibleDiskFraction(parameters[8], angle, parameters[9])
    )
  };
}
