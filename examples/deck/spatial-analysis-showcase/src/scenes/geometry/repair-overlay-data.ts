// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {buildBoundaryForensicsGeometry} from './boundary-forensics-data';

export const REPAIR_REFERENCE_BUILDING_ID = 814558;
/** A synthetic scenario identifier, not a City of Chicago feature ID. */
export const SIMULATED_WORK_AREA_SCENARIO_ID = 900001;

export type PackedPolygonData = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
  sourceIds: Uint32Array;
};

export type RepairOverlayData = {
  footprints: ReturnType<typeof buildBoundaryForensicsGeometry>;
  source: PackedPolygonData;
  simulatedCrossing: PackedPolygonData;
  workArea: PackedPolygonData;
  sourcePath: [number, number][];
  simulatedCrossingPath: [number, number][];
  workAreaPath: [number, number][];
  sourceBuildingId: number;
};

/** Keep the source footprint and make a separate, explicitly simulated bow-tie ring. */
export function buildRepairOverlayData(
  dataset: LoadedDataset,
  origin: readonly [number, number]
): RepairOverlayData {
  const footprints = buildBoundaryForensicsGeometry(dataset, origin);
  const sourceFeature = Array.from(footprints.sourceIds).indexOf(REPAIR_REFERENCE_BUILDING_ID);
  // This selected source is a single simple four-corner ring in the bundled extract.
  if (sourceFeature < 0)
    throw new Error('Expected source building is missing from the bundled extract');
  const firstPolygon = footprints.featureOffsets[sourceFeature];
  const sourceRing = footprints.polygonOffsets[firstPolygon];
  const firstVertex = footprints.ringOffsets[sourceRing];
  const lastVertex = footprints.ringOffsets[sourceRing + 1];
  const sourcePositions = footprints.positions.slice(firstVertex * 2, lastVertex * 2);
  const source = makeSinglePolygon(sourcePositions, REPAIR_REFERENCE_BUILDING_ID);
  const uniquePoints: [number, number][] = [];
  for (let vertex = 0; vertex < sourcePositions.length / 2; vertex++) {
    const point: [number, number] = [sourcePositions[vertex * 2], sourcePositions[vertex * 2 + 1]];
    const firstPoint = uniquePoints[0];
    if (
      vertex === sourcePositions.length / 2 - 1 &&
      firstPoint &&
      point[0] === firstPoint[0] &&
      point[1] === firstPoint[1]
    )
      break;
    uniquePoints.push(point);
  }
  // Swapping the two inner corners crosses the diagonals while retaining the source coordinates.
  const crossingPoints = [
    uniquePoints[0],
    uniquePoints[2],
    uniquePoints[1],
    uniquePoints[3],
    uniquePoints[0]
  ];
  const simulatedCrossingPositions = Float32Array.from(crossingPoints.flat());
  const simulatedCrossing = makeSinglePolygon(
    simulatedCrossingPositions,
    REPAIR_REFERENCE_BUILDING_ID
  );
  const workAreaPositions = Float32Array.of(35, -20, 130, -20, 130, 90, 35, 90, 35, -20);
  const workArea = makeSinglePolygon(workAreaPositions, SIMULATED_WORK_AREA_SCENARIO_ID);
  const projection = dataset.getProjection(origin);
  const unprojectPath = (positions: Float32Array): [number, number][] =>
    Array.from({length: positions.length / 2}, (_, vertex) =>
      projection.unproject(positions[vertex * 2], positions[vertex * 2 + 1])
    );
  return {
    footprints,
    source,
    simulatedCrossing,
    workArea,
    sourcePath: unprojectPath(sourcePositions),
    simulatedCrossingPath: unprojectPath(simulatedCrossingPositions),
    workAreaPath: unprojectPath(workAreaPositions),
    sourceBuildingId: REPAIR_REFERENCE_BUILDING_ID
  };
}

function makeSinglePolygon(positions: Float32Array, sourceId: number): PackedPolygonData {
  return {
    positions,
    ringOffsets: Uint32Array.of(0, positions.length / 2),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1),
    sourceIds: Uint32Array.of(sourceId)
  };
}
