// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  computeTrajectorySplitOnCPU,
  computeTrajectoryThinningOnCPU,
  computeTrajectoryStaypointsOnCPU,
  computeTrajectorySegmentsOnCPU,
  computeTrajectoryLocationsOnCPU,
  computeTrajectoryRecursionsOnCPU,
  computeMobilityIndicatorsOnCPU,
  computeModalSplitOnCPU,
  computeBrownianBridgeUtilizationOnCPU,
  getGPUTrajectorySplitParameterValues,
  getGPUTrajectoryThinningParameterValues,
  getGPUTrajectoryStaypointsParameterValues,
  getGPUTrajectorySegmentsParameterValues,
  getGPUTrajectoryLocationsParameterValues,
  getGPUBrownianBridgeUtilizationParameterValues,
  type GPUTrajectorySplitMode,
  type GPUTrajectoryThinningMode
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  appendCatalogRow,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogPosition
} from './catalog-geojson';
import {
  getTrajectoryTrackRows,
  getTrajectoryFixPosition,
  makeTrajectoryFixes,
  makeTrajectoryTracks,
  makeTrajectoryOutput,
  type CatalogTrajectory,
  type TrajectoryValues
} from './catalog-trajectory-common';

const MISSING_IDENTIFIER = 0xffffffff;
const getNumber = (values: TrajectoryValues, name: string) => values[name] as number;
const getRows = (start: number, end: number) =>
  Array.from({length: end - start}, (_, index) => start + index);

export function computeCatalogStaypoints(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  return computeTrajectoryStaypointsOnCPU(
    {
      ...trajectory,
      parameters: getGPUTrajectoryStaypointsParameterValues({
        distanceThreshold: getNumber(values, 'distanceThreshold'),
        timeThreshold: getNumber(values, 'timeThreshold') * 1000,
        maximumGap: getNumber(values, 'maximumGap') * 1000,
        includeLast: values['includeLast'] as boolean
      })
    },
    {capacity: trajectory.fixes.length}
  );
}

export function computeCatalogSegments(
  trajectory: CatalogTrajectory,
  values: TrajectoryValues,
  stays = computeCatalogStaypoints(trajectory, values)
) {
  return computeTrajectorySegmentsOnCPU(
    {
      ...trajectory,
      staypointIds: stays.staypointIds,
      parameters: getGPUTrajectorySegmentsParameterValues({
        maximumGap: getNumber(values, 'maximumGap') * 1000,
        activityMinimumDuration: getNumber(values, 'activityMinimumDuration') * 1000
      }),
      staypoints: {
        tracks: stays.ids,
        startRows: stays.startRows,
        finishRows: stays.finishRows,
        durations: stays.durations,
        count: stays.count
      }
    },
    {triplegCapacity: trajectory.fixes.length, tripCapacity: trajectory.fixes.length}
  );
}

function computeCatalogLocations(
  trajectory: CatalogTrajectory,
  values: TrajectoryValues,
  stays: ReturnType<typeof computeCatalogStaypoints>
) {
  return computeTrajectoryLocationsOnCPU({
    centroids: stays.centroids,
    tracks: stays.ids,
    count: stays.count,
    parameters: getGPUTrajectoryLocationsParameterValues({epsilon: getNumber(values, 'epsilon')})
  });
}

function makeStaypointTable(
  trajectory: CatalogTrajectory,
  stays: ReturnType<typeof computeCatalogStaypoints>,
  locations?: ReturnType<typeof computeCatalogLocations>
) {
  const table = makeCatalogTable(
    'trajectory-staypoints',
    'Trajectory staypoints',
    {
      staypoint_id: 'int32',
      track_id: 'string',
      source_index: 'int32',
      vertex_index: 'int32',
      start_time: 'float64',
      end_time: 'float64',
      duration_seconds: 'float64',
      location_id: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  for (let row = 0; row < stays.count[0]; row++) {
    const fix = trajectory.fixes[stays.startRows[row]];
    appendCatalogRow(table, {
      staypoint_id: row,
      track_id: trajectory.trackIds[stays.ids[row]],
      source_index: fix.sourceIndex,
      vertex_index: fix.vertexIndex,
      start_time: fix.timestamp,
      end_time: trajectory.fixes[stays.finishRows[row]].timestamp,
      duration_seconds: stays.durations[row] / 1000,
      location_id: locations?.locationIds[row] ?? null,
      geometry: makeCatalogGeometryText(
        'Point',
        trajectory.frame.inverse(stays.centroids[row * 2], stays.centroids[row * 2 + 1])
      )
    });
  }
  return table;
}

export function runTrajectorySplit(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const mode = values['mode'] as GPUTrajectorySplitMode;
  const categories = new Map<unknown, number>();
  const categoryValues = Uint32Array.from(trajectory.fixes, fix => {
    const value = fix.properties[values['valueProperty'] as string];
    if (
      mode === 'value-change' &&
      typeof value !== 'string' &&
      typeof value !== 'number' &&
      typeof value !== 'boolean'
    )
      throw new Error('Trajectory value-change requires a scalar valueProperty on every fix.');
    if (!categories.has(value)) categories.set(value, categories.size);
    return categories.get(value)!;
  });
  const result = computeTrajectorySplitOnCPU(
    {
      ...trajectory,
      parameters: getGPUTrajectorySplitParameterValues({
        maximumGap: getNumber(values, 'maximumGap') * 1000
      }),
      ...(mode === 'stop'
        ? {staypointIds: computeCatalogStaypoints(trajectory, values).staypointIds}
        : {}),
      ...(mode === 'value-change' ? {values: categoryValues} : {})
    },
    {mode, segmentCapacity: trajectory.fixes.length, rowCapacity: trajectory.fixes.length * 2}
  );
  if (result.overflow[0])
    throw new Error('Trajectory split capacity exceeded; no partial result published.');
  const groups = Array.from({length: result.count[0]}, (_, index) =>
    Array.from(
      result.rowIds.subarray(result.segmentOffsets[index], result.segmentOffsets[index + 1])
    )
  );
  return makeTrajectoryOutput(
    trajectory,
    [makeTrajectoryTracks(trajectory, groups), makeTrajectoryFixes(trajectory, {}, groups.flat())],
    {segmentCount: groups.length}
  );
}

export function runTrajectoryThinning(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const mode = values['mode'] as GPUTrajectoryThinningMode;
  const result = computeTrajectoryThinningOnCPU(
    {
      ...trajectory,
      parameters: getGPUTrajectoryThinningParameterValues({
        tolerance: getNumber(values, 'tolerance') * (mode === 'minimum-time' ? 1000 : 1)
      })
    },
    {mode, capacity: trajectory.fixes.length}
  );
  const rows = Array.from(result.rowIds.subarray(0, result.count[0]));
  const groups = getTrajectoryTrackRows(trajectory).map(group =>
    group.filter(row => result.validity[row] !== 0)
  );
  return makeTrajectoryOutput(
    trajectory,
    [makeTrajectoryTracks(trajectory, groups), makeTrajectoryFixes(trajectory, {}, rows)],
    {keptFixCount: rows.length}
  );
}

export function runTrajectoryStaypoints(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const stays = computeCatalogStaypoints(trajectory, values);
  return makeTrajectoryOutput(trajectory, [makeStaypointTable(trajectory, stays)], {
    staypointCount: stays.count[0]
  });
}

export function runTrajectorySegments(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const stays = computeCatalogStaypoints(trajectory, values);
  const segments = computeCatalogSegments(trajectory, values, stays);
  const groups = Array.from({length: segments.triplegCount[0]}, (_, index) =>
    getRows(segments.triplegStartRows[index], segments.triplegEndRows[index])
  );
  const trips = makeTrajectoryTracks(
    trajectory,
    Array.from({length: segments.tripCount[0]}, (_, index) =>
      getRows(segments.tripStartRows[index], segments.tripFinishRows[index] + 1)
    )
  );
  trips.id = 'trajectory-trips';
  trips.label = 'Trajectory trips';
  return makeTrajectoryOutput(
    trajectory,
    [makeTrajectoryTracks(trajectory, groups), trips, makeStaypointTable(trajectory, stays)],
    {triplegCount: groups.length, tripCount: trips.rowCount}
  );
}

export function runTrajectoryLocations(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const stays = computeCatalogStaypoints(trajectory, values);
  const locations = computeCatalogLocations(trajectory, values, stays);
  const table = makeCatalogTable(
    'trajectory-locations',
    'Trajectory locations',
    {location_id: 'int32', track_id: 'string', staypoint_count: 'int32', geometry: 'string'},
    'geometry'
  );
  const members = new Map<number, number[]>();
  for (let row = 0; row < stays.count[0]; row++) {
    const location = locations.locationIds[row];
    if (!members.has(location)) members.set(location, []);
    members.get(location)!.push(row);
  }
  for (const [location, rows] of members) {
    const row = rows[0];
    appendCatalogRow(table, {
      location_id: location,
      track_id: trajectory.trackIds[stays.ids[row]],
      staypoint_count: rows.length,
      geometry: makeCatalogGeometryText(
        'Point',
        trajectory.frame.inverse(
          locations.centers[location * 2],
          locations.centers[location * 2 + 1]
        )
      )
    });
  }
  return makeTrajectoryOutput(
    trajectory,
    [table, makeStaypointTable(trajectory, stays, locations)],
    {locationCount: table.rowCount}
  );
}

export function runTrajectoryRecursions(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const radii = values['radii'] as readonly number[];
  if (!radii.length || radii.some(radius => !(radius > 0)))
    throw new Error('Trajectory recursion radii must be positive.');
  const result = computeTrajectoryRecursionsOnCPU({...trajectory, radii: Float32Array.from(radii)});
  const table = makeCatalogTable(
    'trajectory-recursions',
    'Trajectory recursions',
    {
      fix_index: 'int32',
      source_index: 'int32',
      vertex_index: 'int32',
      track_id: 'string',
      radius_index: 'int32',
      radius: 'float64',
      revisits: 'int32',
      residence_seconds: 'float64',
      first_passage_seconds: 'float64',
      first_passage_censored: 'boolean',
      geometry: 'string'
    },
    'geometry'
  );
  trajectory.fixes.forEach((fix, row) =>
    radii.forEach((radius, radiusIndex) => {
      const index = row * radii.length + radiusIndex;
      appendCatalogRow(table, {
        fix_index: row,
        source_index: fix.sourceIndex,
        vertex_index: fix.vertexIndex,
        track_id: fix.trackId,
        radius_index: radiusIndex,
        radius,
        revisits: result.revisits[index],
        residence_seconds: result.residenceTimes[index] / 1000,
        first_passage_seconds: result.firstPassageCensored[index]
          ? null
          : result.firstPassageTimes[index] / 1000,
        first_passage_censored: Boolean(result.firstPassageCensored[index]),
        geometry: makeCatalogGeometryText('Point', getTrajectoryFixPosition(fix))
      });
    })
  );
  return makeTrajectoryOutput(trajectory, [table]);
}

export function runTrajectoryMobility(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const stays = computeCatalogStaypoints(trajectory, values);
  const locations = computeCatalogLocations(trajectory, values, stays);
  const segments = computeCatalogSegments(trajectory, values, stays);
  const result = computeMobilityIndicatorsOnCPU(
    {
      timestamps: trajectory.timestamps,
      staypoints: {
        tracks: stays.ids,
        centroids: stays.centroids,
        startRows: stays.startRows,
        durations: stays.durations,
        locationIds: locations.locationIds,
        count: stays.count
      },
      triplegs: {
        tracks: segments.triplegIds,
        startRows: segments.triplegStartRows,
        endRows: segments.triplegEndRows,
        count: segments.triplegCount,
        staypointFinishRows: stays.finishRows
      }
    },
    {trackCount: trajectory.trackIds.length}
  );
  const table = makeCatalogTable(
    'trajectory-mobility',
    'Mobility indicators',
    {
      track_id: 'string',
      radius_of_gyration: 'float64',
      random_entropy: 'float64',
      uncorrelated_entropy: 'float64',
      real_entropy: 'float64',
      home_location: 'int32',
      work_location: 'int32',
      tracking_quality: 'float64'
    },
    undefined
  );
  trajectory.trackIds.forEach((trackId, index) =>
    appendCatalogRow(table, {
      track_id: trackId,
      radius_of_gyration: result.radiusOfGyration[index],
      random_entropy: result.randomEntropy[index],
      uncorrelated_entropy: result.uncorrelatedEntropy[index],
      real_entropy: result.realEntropy[index],
      home_location:
        result.homeLocations[index] === MISSING_IDENTIFIER ? null : result.homeLocations[index],
      work_location:
        result.workLocations[index] === MISSING_IDENTIFIER ? null : result.workLocations[index],
      tracking_quality: result.trackingQuality[index]
    })
  );
  const stayTable = makeStaypointTable(trajectory, stays, locations);
  stayTable.columnTypes['jump_length'] = 'float64';
  stayTable.columns['jump_length'] = Array.from(result.jumpLengths.subarray(0, stays.count[0]));
  stayTable.columnTypes['waiting_seconds'] = 'float64';
  stayTable.columns['waiting_seconds'] = Array.from(
    result.waitingTimes.subarray(0, stays.count[0]),
    value => value / 1000
  );
  return makeTrajectoryOutput(trajectory, [table, stayTable]);
}

export function runTrajectoryModalSplit(trajectory: CatalogTrajectory, values: TrajectoryValues) {
  const segments = computeCatalogSegments(trajectory, values);
  const modeSpeedBreaks = values['modeSpeedBreaks'] as readonly number[];
  if (!modeSpeedBreaks.length || modeSpeedBreaks.some(value => value <= 0))
    throw new Error('Modal speed breaks must be positive and strictly ascending.');
  const metric = values['metric'] as 'count' | 'distance' | 'duration';
  const result = computeModalSplitOnCPU(
    {
      ...trajectory,
      triplegs: {
        tracks: segments.triplegIds,
        startRows: segments.triplegStartRows,
        endRows: segments.triplegEndRows,
        count: segments.triplegCount
      }
    },
    {
      modeCount: modeSpeedBreaks.length,
      metric,
      modeSpeedBreaks: modeSpeedBreaks.map(value => value / 1000)
    }
  );
  const table = makeCatalogTable('trajectory-modal-split', 'Modal split', {
    track_id: 'string',
    mode: 'int32',
    metric: 'string',
    value: 'float64'
  });
  trajectory.trackIds.forEach((trackId, trackIndex) =>
    modeSpeedBreaks.forEach((_, mode) =>
      appendCatalogRow(table, {
        track_id: trackId,
        mode,
        metric,
        value:
          result.modalSplit[trackIndex * modeSpeedBreaks.length + mode] /
          (metric === 'duration' ? 1000 : 1)
      })
    )
  );
  const triplegs = makeTrajectoryTracks(
    trajectory,
    Array.from({length: segments.triplegCount[0]}, (_, index) =>
      getRows(segments.triplegStartRows[index], segments.triplegEndRows[index])
    )
  );
  triplegs.columnTypes['mode'] = 'int32';
  triplegs.columns['mode'] = Array.from(
    result.predictedModes.subarray(0, triplegs.rowCount),
    value => (value === MISSING_IDENTIFIER ? null : value)
  );
  return makeTrajectoryOutput(trajectory, [table, triplegs]);
}

export function prepareTrajectoryHomeRange(
  trajectory: CatalogTrajectory,
  values: TrajectoryValues
) {
  if (trajectory.fixes.length < 2) throw new Error('Brownian bridge requires at least two fixes.');
  const padding = getNumber(values, 'padding');
  const horizontal = Array.from(trajectory.positions.filter((_, index) => index % 2 === 0));
  const vertical = Array.from(trajectory.positions.filter((_, index) => index % 2 === 1));
  const extent: [number, number, number, number] = [
    Math.min(...horizontal) - padding,
    Math.min(...vertical) - padding,
    Math.max(...horizontal) + padding,
    Math.max(...vertical) + padding
  ];
  const gridSize: [number, number] = [
    getNumber(values, 'gridWidth'),
    getNumber(values, 'gridHeight')
  ];
  const input = {
    ...trajectory,
    motionVariances: new Float32Array(trajectory.fixes.length).fill(
      getNumber(values, 'motionVariance') / 1000
    ),
    locationErrors: new Float32Array(trajectory.fixes.length).fill(
      getNumber(values, 'locationError')
    ),
    parameters: getGPUBrownianBridgeUtilizationParameterValues({extent})
  };
  const {density} = computeBrownianBridgeUtilizationOnCPU(input, {
    gridSize,
    subStepCount: getNumber(values, 'subStepCount')
  });
  return {extent, gridSize, density};
}

export function makeTrajectoryHomeRange(
  trajectory: CatalogTrajectory,
  grid: ReturnType<typeof prepareTrajectoryHomeRange>,
  volume?: Float32Array,
  inside?: Uint32Array
) {
  const table = makeCatalogTable(
    'trajectory-home-range',
    volume ? 'Home range isopleth cells' : 'Brownian bridge density cells',
    {
      cell_id: 'int32',
      density: 'float64',
      ...(volume ? {volume: 'float64' as const, inside: 'boolean' as const} : {}),
      geometry: 'string'
    },
    'geometry'
  );
  const [width, height] = grid.gridSize;
  const [minimumX, minimumY, maximumX, maximumY] = grid.extent;
  const cellWidth = (maximumX - minimumX) / width;
  const cellHeight = (maximumY - minimumY) / height;
  grid.density.forEach((density, index) => {
    if (inside && !inside[index]) return;
    const left = minimumX + (index % width) * cellWidth;
    const bottom = minimumY + Math.floor(index / width) * cellHeight;
    const ring: CatalogPosition[] = [
      [left, bottom],
      [left + cellWidth, bottom],
      [left + cellWidth, bottom + cellHeight],
      [left, bottom + cellHeight],
      [left, bottom]
    ].map(([horizontal, vertical]) => trajectory.frame.inverse(horizontal, vertical));
    appendCatalogRow(table, {
      cell_id: index,
      density,
      ...(volume ? {volume: volume[index], inside: Boolean(inside?.[index])} : {}),
      geometry: makeCatalogGeometryText('Polygon', [ring])
    });
  });
  return makeTrajectoryOutput(trajectory, [table], {cellCount: table.rowCount});
}

export function runTrajectoryBrownianBridge(
  trajectory: CatalogTrajectory,
  values: TrajectoryValues
) {
  return makeTrajectoryHomeRange(trajectory, prepareTrajectoryHomeRange(trajectory, values));
}
