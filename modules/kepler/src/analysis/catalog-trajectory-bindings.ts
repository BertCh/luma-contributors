// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogOperationBinding,
  CatalogOutputSpec,
  CatalogParameterSpec
} from './catalog-binding-types';
import {
  MAXIMUM_TRAJECTORY_FIXES,
  TRAJECTORY_PARAMETERS,
  numberParameter,
  enumParameter,
  stringParameter,
  prepareCatalogTrajectory,
  validateTrajectoryPublication,
  type CatalogTrajectory,
  type TrajectoryValues
} from './catalog-trajectory-common';
import {
  runTrajectorySplit,
  runTrajectoryThinning,
  runTrajectoryStaypoints,
  runTrajectorySegments,
  runTrajectoryLocations,
  runTrajectoryRecursions,
  runTrajectoryMobility,
  runTrajectoryModalSplit,
  runTrajectoryBrownianBridge
} from './catalog-trajectory-cpu';
import {
  runTrajectoryHampel,
  runTrajectoryOutlierFlags,
  runTrajectoryRelativeTimes,
  runTrajectoryIsopleths
} from './catalog-trajectory-gpu';

function integerParameter(
  name: string,
  label: string,
  defaultValue: number,
  minimum: number,
  maximum: number
): CatalogParameterSpec {
  return {...numberParameter(name, label, defaultValue, minimum, maximum), kind: 'integer'};
}
function output(
  tableId: string,
  label: string,
  geometry: CatalogOutputSpec['geometry'],
  identityColumns: string[]
): CatalogOutputSpec {
  return {tableId, label, geometry, identityColumns, maximumRows: 16384};
}
const FIXES = output('trajectory-fixes', 'Trajectory fixes', 'point', [
  'source_index',
  'vertex_index',
  'fix_index',
  'output_index',
  'track_id'
]);
const TRACKS = output('trajectory-tracks', 'Trajectory tracks', 'line', ['track_id', 'part_index']);
const STAYPOINTS = output('trajectory-staypoints', 'Trajectory staypoints', 'point', [
  'track_id',
  'staypoint_id',
  'source_index',
  'vertex_index'
]);
const HOME_RANGE = output('trajectory-home-range', 'Home range cells', 'polygon', ['cell_id']);
const STAY_PARAMETERS: CatalogParameterSpec[] = [
  numberParameter(
    'distanceThreshold',
    'Staypoint distance threshold',
    50,
    0.000001,
    1e7,
    'm or planar units'
  ),
  numberParameter('timeThreshold', 'Minimum stay duration', 60, 0, 1e9, 'seconds'),
  numberParameter('maximumGap', 'Maximum time gap (0 disables)', 0, 0, 1e9, 'seconds'),
  {name: 'includeLast', label: 'Include final staypoint', kind: 'boolean', defaultValue: true}
];
const SEGMENT_PARAMETERS: CatalogParameterSpec[] = [
  ...STAY_PARAMETERS,
  numberParameter('activityMinimumDuration', 'Minimum activity duration', 300, 0, 1e9, 'seconds')
];
const LOCATION_PARAMETERS: CatalogParameterSpec[] = [
  ...STAY_PARAMETERS,
  numberParameter('epsilon', 'Location clustering radius', 100, 0.000001, 1e7, 'm or planar units')
];
const HOME_PARAMETERS: CatalogParameterSpec[] = [
  integerParameter('gridWidth', 'Grid width', 16, 2, 64),
  integerParameter('gridHeight', 'Grid height', 16, 2, 64),
  numberParameter('padding', 'Grid padding', 100, 0.001, 1e7, 'm or planar units'),
  numberParameter(
    'motionVariance',
    'Brownian motion variance',
    10,
    0,
    1e9,
    'length squared / second'
  ),
  numberParameter(
    'locationError',
    'Location error standard deviation',
    10,
    0,
    1e7,
    'm or planar units'
  ),
  integerParameter('subStepCount', 'Bridge integration midpoints', 8, 1, 256)
];
const COMMON_BEHAVIOR = [
  'Accepts Point fixes with track/time properties, or timed LineStrings with [x,y,z,time] vertices (or a timestamp property array). Native point tables map explicit longitude/latitude columns. Track IDs fall back to feature IDs.',
  'Fixes are grouped in first-seen track order and sorted by time. Null geometries are skipped and counted. Missing IDs, invalid non-null coordinates/times, duplicate track timestamps, more than 128 tracks or 2,048 fixes reject the whole run. Timestamps resolve to safe integer epoch milliseconds; sub-millisecond input is refused.',
  'Source feature and path vertex identities survive sorting. Geographic positions use one local azimuthal equidistant frame in meters; planar positions use shifted coordinate units. Threshold durations and speeds are entered in seconds. Absolute output timestamps are epoch milliseconds.',
  'Line outputs retain four-component timed coordinates and can be published as native Trips. Point and polygon outputs publish as native GeoJSON layers. Singleton tracks have point outputs but cannot form a path.'
];

type CpuRunner = (
  trajectory: CatalogTrajectory,
  values: TrajectoryValues
) => ReturnType<typeof runTrajectorySplit>;
type GpuRunner = typeof runTrajectoryHampel;
function binding(
  id: string,
  title: string,
  contributorNames: string[],
  parameters: CatalogParameterSpec[],
  outputs: CatalogOutputSpec[],
  summary: string,
  run?: CpuRunner,
  runGpu?: GpuRunner,
  extraBehavior: string[] = []
): CatalogOperationBinding {
  return {
    id,
    title,
    domain: 'analysis',
    summary,
    capabilityIds: ['movement-analysis'],
    contributorNames,
    gpuContributorNames: runGpu
      ? [id === 'trajectory-isopleths' ? 'GPUUtilizationVolumeIsopleths' : contributorNames[0]]
      : [],
    engine: runGpu ? (run ? 'gpu-or-cpu' : 'gpu') : 'cpu-reference',
    inputs: [
      {
        name: 'tracks',
        label: 'Trajectory observations',
        accepts: ['point', 'line'],
        maximumFeatures: MAXIMUM_TRAJECTORY_FIXES,
        maximumVertices: MAXIMUM_TRAJECTORY_FIXES
      }
    ],
    parameters: [...TRAJECTORY_PARAMETERS, ...parameters],
    outputs,
    cleanup:
      'Results are copied typed tables. GPU graphs and buffers are destroyed after bounded readback; published native Sources own their independent resources.',
    declaredBehavior: [...COMMON_BEHAVIOR, ...extraBehavior],
    limitation:
      'Bounded local-frame trajectory analysis; distances away from the frame origin are approximate. CPU bindings use existing library operation twins, with their float64 threshold semantics and uncapped searches. No new numerical implementation. Native Trips publication has an explicit Float32 animation time error budget.',
    ...(run
      ? {
          run: (inputs, values) =>
            run(preparePublishedTrajectory(inputs['tracks'], values, outputs), values)
        }
      : {}),
    ...(runGpu
      ? {
          runGpu: (context, inputs, values) =>
            runGpu(context, preparePublishedTrajectory(inputs['tracks'], values, outputs), values)
        }
      : {})
  };
}

/** Existing contributor entry points, exposed through one isolated registry addition. */
export const TRAJECTORY_BINDINGS: readonly CatalogOperationBinding[] = [
  binding(
    'trajectory-hampel',
    'Trajectory Hampel filter',
    ['GPUTrajectoryHampel'],
    [
      enumParameter('target', 'Filter target', ['x', 'y', 'property']),
      stringParameter('valueProperty', 'Scalar value property', 'value'),
      integerParameter('windowSize', 'Hampel window size', 5, 1, 129),
      numberParameter('numberOfSigmas', 'Hampel deviation threshold', 3, 0, 100)
    ],
    [FIXES, TRACKS],
    'Filters one local coordinate or scalar per track with the existing GPU Hampel contributor.',
    undefined,
    runTrajectoryHampel,
    [
      'Window endpoints pass through unchanged. Coordinate targets replace that coordinate in cleaned paths; property target publishes filtered_value and leaves path positions unchanged. Even window sizes follow the contributor: 2*floor(windowSize/2)+1 rows.'
    ]
  ),
  binding(
    'trajectory-outlier-flags',
    'Trajectory outlier flags',
    ['GPUTrajectoryOutlierFlags'],
    [
      numberParameter('maximumSpeed', 'Maximum speed (0 disables)', 50, 0, 1e7, 'length / second'),
      numberParameter(
        'maximumAcceleration',
        'Maximum acceleration (0 disables)',
        0,
        0,
        1e7,
        'length / second squared'
      )
    ],
    [FIXES, TRACKS],
    'Flags displaced fixes by speed and acceleration and publishes tracks with flagged fixes removed.',
    undefined,
    runTrajectoryOutlierFlags,
    [
      'The point dataset retains all fixes and bit flags; cleaned paths join remaining fixes within each track. This is the contributor’s point-centered single pass, not a sequential last-kept-fix cleaner.'
    ]
  ),
  binding(
    'trajectory-split',
    'Trajectory split',
    ['GPUTrajectorySplit'],
    [
      ...STAY_PARAMETERS,
      enumParameter('mode', 'Split mode', ['gap', 'stop', 'value-change']),
      stringParameter('valueProperty', 'Split value property', 'value')
    ],
    [TRACKS, FIXES],
    'Splits trajectories at time gaps, computed stops or property changes.',
    runTrajectorySplit,
    undefined,
    [
      'Runs shorter than two fixes are dropped. Value-change mode repeats the boundary fix in adjacent paths, matching the library wrapper.'
    ]
  ),
  binding(
    'trajectory-thinning',
    'Trajectory thinning',
    ['GPUTrajectoryThinning'],
    [
      enumParameter('mode', 'Thinning mode', ['minimum-distance', 'minimum-time']),
      numberParameter('tolerance', 'Thinning tolerance (distance or seconds)', 25, 0, 1e9)
    ],
    [TRACKS, FIXES],
    'Keeps fixes separated by a minimum distance or time, including both endpoints.',
    runTrajectoryThinning
  ),
  binding(
    'trajectory-track-relative-times',
    'Trajectory track-relative times',
    ['GPUTrajectoryTrackRelativeTimes'],
    [
      numberParameter(
        'maximumTimeErrorMilliseconds',
        'Maximum relative time error',
        0,
        0,
        1e6,
        'milliseconds'
      )
    ],
    [FIXES, TRACKS],
    'Computes Float32 milliseconds from each track’s own origin with exact Int64 subtraction.',
    undefined,
    runTrajectoryRelativeTimes
  ),
  binding(
    'trajectory-staypoints',
    'Trajectory staypoints',
    ['GPUTrajectoryStaypoints'],
    STAY_PARAMETERS,
    [STAYPOINTS],
    'Detects stays using distance, minimum duration and maximum observation gap.',
    runTrajectoryStaypoints
  ),
  binding(
    'trajectory-segments',
    'Trajectory segments and trips',
    ['GPUTrajectorySegments'],
    SEGMENT_PARAMETERS,
    [
      TRACKS,
      output('trajectory-trips', 'Trajectory trips', 'line', ['track_id', 'part_index']),
      STAYPOINTS
    ],
    'Computes staypoints, moving triplegs and activity-to-activity trips with the library event state machine.',
    runTrajectorySegments
  ),
  binding(
    'trajectory-locations',
    'Trajectory locations',
    ['GPUTrajectoryLocations'],
    LOCATION_PARAMETERS,
    [
      output('trajectory-locations', 'Trajectory locations', 'point', ['track_id', 'location_id']),
      STAYPOINTS
    ],
    'Clusters computed staypoints into connected locations independently within each track.',
    runTrajectoryLocations
  ),
  binding(
    'trajectory-recursions',
    'Trajectory recursions',
    ['GPUTrajectoryRecursions'],
    [
      {
        name: 'radii',
        label: 'Recursion radii',
        kind: 'number-list',
        defaultValue: [100],
        minimum: 0.000001,
        maximum: 1e7,
        maximumLength: 8
      }
    ],
    [
      output('trajectory-recursions', 'Trajectory recursions', 'point', [
        'fix_index',
        'radius_index',
        'track_id',
        'source_index',
        'vertex_index',
        'radius'
      ])
    ],
    'Computes revisits, residence and first-passage times per fix and radius.',
    runTrajectoryRecursions,
    undefined,
    ['Censored first-passage times are null with an explicit censor flag.']
  ),
  binding(
    'trajectory-mobility-indicators',
    'Trajectory mobility indicators',
    ['GPUMobilityIndicators'],
    [
      ...SEGMENT_PARAMETERS,
      numberParameter('epsilon', 'Location clustering radius', 100, 0.000001, 1e7)
    ],
    [output('trajectory-mobility', 'Mobility indicators', 'none', ['track_id']), STAYPOINTS],
    'Computes gyration, jump lengths, waiting times, entropies, home/work locations and tracking quality from computed stays, locations and triplegs.',
    runTrajectoryMobility,
    undefined,
    [
      'Home/work are the two locations with longest cumulative stay duration, not time-of-day classification.'
    ]
  ),
  binding(
    'trajectory-modal-split',
    'Trajectory modal split',
    ['GPUModalSplit'],
    [
      ...SEGMENT_PARAMETERS,
      enumParameter('metric', 'Modal split metric', ['count', 'distance', 'duration']),
      {
        name: 'modeSpeedUpperBoundsFloat32',
        renamedFrom: 'modeSpeedBreaks',
        label: 'Mode speed upper bounds',
        kind: 'number-list',
        defaultValue: [1.5, 8, 50],
        minimum: 0.000001,
        maximum: 1e7,
        maximumLength: 8
      }
    ],
    [output('trajectory-modal-split', 'Modal split', 'none', ['track_id', 'mode']), TRACKS],
    'Classifies moving triplegs by average speed and aggregates count, distance or duration per track and mode.',
    runTrajectoryModalSplit,
    undefined,
    [
      'Speed upper bounds are exclusive and strictly ascending; the final mode also catches speeds above the last bound. Output mode numbers are classes, not inferred transport labels.'
    ]
  ),
  binding(
    'trajectory-brownian-bridge',
    'Trajectory Brownian bridge utilization',
    ['GPUBrownianBridgeUtilization'],
    HOME_PARAMETERS,
    [HOME_RANGE],
    'Computes a time-weighted bridge density grid from tracks with constant motion variance and observation error.',
    runTrajectoryBrownianBridge,
    undefined,
    [
      'All tracks contribute to one pooled distribution. Motion variance is entered in length squared per second; location error is a standard deviation. Density is per local-frame area, not geographic degree area.'
    ]
  ),
  binding(
    'trajectory-isopleths',
    'Trajectory home range isopleths',
    ['GPUUtilizationVolumeIsopleths'],
    [...HOME_PARAMETERS, numberParameter('level', 'Home range probability', 0.95, 0.000001, 1)],
    [HOME_RANGE],
    'Uses the existing CPU Brownian bridge wrapper then GPU volume isopleths to publish selected home-range grid cells.',
    undefined,
    runTrajectoryIsopleths,
    [
      'Publishes the smallest stable density-ordered set reaching the requested mass, including the crossing cell. Output is individual polygon cells, not a dissolved contour. Density preparation uses the CPU bridge wrapper; only GPUUtilizationVolumeIsopleths is dispatched on the GPU.'
    ]
  )
];

function preparePublishedTrajectory(
  input: Parameters<typeof prepareCatalogTrajectory>[0],
  values: TrajectoryValues,
  outputs: readonly CatalogOutputSpec[]
): CatalogTrajectory {
  const trajectory = prepareCatalogTrajectory(input, values);
  if (outputs.some(output => output.identityColumns.includes('track_id')))
    validateTrajectoryPublication(trajectory);
  return trajectory;
}
