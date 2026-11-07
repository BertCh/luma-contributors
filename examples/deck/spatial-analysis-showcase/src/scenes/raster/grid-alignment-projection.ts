// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {compileProjectionPlan, evaluateProjectionPlan, evaluateProjectionProgram, type ProjectionPlan, type ProjectionProgram} from '@luma.gl/graph/gpu-project';
import {
  GPU_WGS84_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getMercatorY} from '../../engine/web-mercator';
import {utmToLngLat} from './b16-common';
import {toRadians} from '@math.gl/core';

const PROJECTION_RADIUS = GPU_WGS84_SEMI_MAJOR_AXIS_METERS;

/**
 * Build a bounded double-single adaptive plan from local Mercator offsets to local UTM offsets.
 * Large absolute UTM coordinates are removed in the binary64 provider before values reach GPU
 * interpolation, keeping both axes near the 3.84 km source window.
 */
export function createGridAlignmentProjection(
  destinationCenter: readonly [number, number],
  sourceNorthwest: readonly [number, number],
  halfWidth: number
): {
  program: ProjectionProgram;
  plan: ProjectionPlan;
  evaluatePlan: (coordinates: readonly [number, number]) => [number, number];
  evaluateProvider: (coordinates: readonly [number, number]) => [number, number];
  patchCount: number;
  sampledTolerance: number;
  maximumError: number;
} {
  const longitudeRadians = toRadians(destinationCenter[0]);
  const latitudeRadians = toRadians(destinationCenter[1]);
  const cosine = Math.cos(latitudeRadians);
  const mercatorOrigin = [
    PROJECTION_RADIUS * longitudeRadians,
    getMercatorY(destinationCenter[1], PROJECTION_RADIUS)
  ] as const;
  const analyticProgram: ProjectionProgram = {
    precision: 'double-single',
    operations: [
      {type: 'affine', scale: [1 / cosine, 1 / cosine], offset: mercatorOrigin},
      {type: 'web-mercator', arithmetic: 'float32', radius: PROJECTION_RADIUS, inverse: true},
      {type: 'affine', scale: [1, 1], offset: [toRadians(123), 0]},
      {
        type: 'transverse-mercator',
        arithmetic: 'float32',
        semiMajorAxis: PROJECTION_RADIUS,
        semiMinorAxis: PROJECTION_RADIUS * (1 - GPU_WGS84_FLATTENING),
        scaleFactor: 0.9996,
        latitudeOrigin: 0
      },
      {
        type: 'affine',
        scale: [1, 1],
        offset: [500000 - sourceNorthwest[0], -sourceNorthwest[1]]
      }
    ]
  };
  const evaluateProvider = (coordinates: readonly [number, number]): [number, number] => {
    const result = evaluateProjectionProgram(analyticProgram, [coordinates[0], coordinates[1]]);
    // All destination centers lie within the bounded valid UTM projection domain.
    assert(result.valid);
    return [result.position[0], result.position[1]];
  };
  const plan = compileProjectionPlan({
    projection: coordinates => evaluateProvider([coordinates[0], coordinates[1]]),
    bounds: [-halfWidth, -halfWidth, halfWidth, halfWidth],
    precision: 'double-single',
    tolerance: 0.001,
    degree: 3,
    maxDepth: 8,
    maxPatches: 4096
  });
  return {
    program: {
      precision: 'local-f32',
      destinationOrigin: [0, 0],
      operations: [{type: 'adaptive', plan}]
    },
    plan,
    evaluatePlan: coordinates => evaluateProjectionPlan(plan, coordinates),
    evaluateProvider,
    patchCount: plan.patches.length,
    sampledTolerance: plan.tolerance,
    maximumError: plan.maxError
  };
}

/** Uses the same crop-center inverse UTM conversion as the live scene. */
export function getGridAlignmentDestinationCenter(
  cropWest: number,
  cropNorth: number,
  gridSize: number,
  cellSize: number
): [number, number] {
  return utmToLngLat(
    cropWest + (gridSize * cellSize) / 2,
    cropNorth - (gridSize * cellSize) / 2,
    10
  );
}
