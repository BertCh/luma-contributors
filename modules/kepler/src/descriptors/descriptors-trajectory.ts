// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_TRAJECTORY_FIXES,
  MAXIMUM_TRAJECTORY_TRACKS
} from '../analysis/catalog-trajectory-common';
import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor,
  KeplerOperationOutputDescriptor,
  KeplerOperationParameterDescriptor,
  KeplerOperationRequirement
} from './operation-descriptor-types';

const INPUT_NOTE =
  'Point fixes read trackProperty and timeProperty; track IDs may fall back to feature IDs. Track columns are chosen explicitly, never inferred from a shared category. Timed LineStrings read [x,y,z,time] vertices or a timestamp property array. Track/time roles are optional mappings to preserve embedded alternatives: preparation still requires an ID and time for every retained fix. Native point-table coordinates follow declared pointColumns or explicit longitudeProperty/latitudeProperty mappings; GeoJSON uses geometry coordinates.';
const TIME_NOTE =
  'timestampEncoding declares unix-millisecond, unix-second or ISO8601 input. Complete ISO text resolves automatically; numeric, mixed or sampled timestamps require an explicit encoding choice. Absolute output timestamps are epoch milliseconds. Safe integer milliseconds are required; sub-millisecond input is refused. Fixes are sorted within each track, retaining source_index and vertex_index. Duplicate track timestamps reject the run. Durations are seconds, speeds length/second and accelerations length/second squared.';
const FRAME_NOTE =
  'Geographic coordinates use one local azimuthal equidistant frame in metres; planar coordinates use shifted source units. Distances, radii, padding and location error use that length unit. Distances away from the frame origin are approximate. Altitude is retained for publication but numerical analysis is two dimensional.';
const PROFILE_NOTE =
  'The profile cannot certify embedded vertex times, fallback feature IDs, per-track timestamp uniqueness, finite Float32 scalars or exact retained fix/track counts after null geometry removal. Preparation checks these before execution, including the 2,048-fix and 128-track limits. These specialist operations stay in Browse all until the profile can certify track/time readiness. Singleton tracks produce points but no path.';
const STAY_NOTE =
  'Stay duration and maximumGap are seconds; maximumGap=0 disables the gap limit. includeLast controls final stays at the observation boundary, where departure is unobserved. Stops depend on sampling cadence, distanceThreshold and timeThreshold; they do not establish visit purpose.';
const HOME_NOTE =
  'All tracks contribute to one pooled utilization distribution. motionVariance is length squared/second; locationError is a standard deviation in the frame length unit. Density is per local-frame area. Neither uncertainty parameter is estimated from the profile. Grid dimensions and integration midpoints control numerical resolution; compare resolutions before interpreting a home range. Padding starts at one tenth of the extent diagonal, falling back to the binding constant when unavailable; it is a grid margin, not an estimated motion scale.';

function describeConstant(
  name: string,
  level: 'basic' | 'advanced' = 'advanced',
  apply: 'live' | 'rebuild' = 'rebuild'
): KeplerOperationParameterDescriptor {
  return {name, defaultPolicy: {from: 'constant'}, level, apply};
}

function describeFields(): KeplerOperationFieldDescriptor[] {
  return [
    {
      role: 'track',
      label: 'Track identifier (or feature ID)',
      parameter: 'trackProperty',
      autoAssign: 'explicit',
      accepts: ['number', 'category', 'text'],
      measurement: ['identifier', 'nominal'],
      required: false
    },
    {
      role: 'time',
      label: 'Timestamp (or embedded vertex time)',
      parameter: 'timeProperty',
      accepts: ['number', 'time', 'text'],
      required: false
    },
    {
      role: 'longitude',
      label: 'Native point horizontal coordinate',
      parameter: 'longitudeProperty',
      autoAssign: 'point-longitude',
      accepts: ['number'],
      required: false
    },
    {
      role: 'latitude',
      label: 'Native point vertical coordinate',
      parameter: 'latitudeProperty',
      autoAssign: 'point-latitude',
      accepts: ['number'],
      required: false
    }
  ];
}

function describeRequirements(): KeplerOperationRequirement[] {
  return [
    {
      input: 'tracks',
      id: 'minimum-rows',
      severity: 'block',
      limit: 1,
      remedy: 'Choose at least one nonempty Point fix or timed LineString'
    },
    {
      input: 'tracks',
      id: 'maximum-rows',
      severity: 'block',
      limit: MAXIMUM_TRAJECTORY_FIXES,
      remedy: 'Filter to at most 2,048 fixes across all tracks'
    },
    {
      input: 'tracks',
      id: 'maximum-vertices',
      severity: 'block',
      limit: MAXIMUM_TRAJECTORY_FIXES,
      remedy: 'Filter or thin timed paths to at most 2,048 total fixes'
    },
    {
      input: 'tracks',
      role: 'track',
      id: 'maximum-distinct-values',
      severity: 'block',
      limit: MAXIMUM_TRAJECTORY_TRACKS,
      remedy: 'Filter to at most 128 track IDs; preparation also checks fallback feature IDs'
    },
    {
      input: 'tracks',
      id: 'declared-coordinates',
      severity: 'block',
      remedy: 'Declare geographic longitude/latitude or planar coordinates and confirm length units'
    }
  ];
}

function describeOutput(
  tableId: string,
  kind: KeplerOperationOutputDescriptor['kind'],
  column: string,
  scale: 'categorical' | 'sequential' = 'sequential'
): KeplerOperationOutputDescriptor {
  return {
    tableId,
    kind,
    presentation: {column, scale},
    interpretation: {summary: 'row-count', cautions: []},
    next: ['how-it-moves']
  };
}
function describeTracks(tableId = 'trajectory-tracks'): KeplerOperationOutputDescriptor {
  return describeOutput(tableId, 'trajectories', 'track_id', 'categorical');
}
function describeStaypoints(): KeplerOperationOutputDescriptor {
  return describeOutput('trajectory-staypoints', 'points', 'duration_seconds');
}
function describeStayParameters(): KeplerOperationParameterDescriptor[] {
  return [
    describeConstant('distanceThreshold', 'basic'),
    describeConstant('timeThreshold', 'basic'),
    describeConstant('maximumGap'),
    describeConstant('includeLast')
  ];
}
function describeSegmentParameters(): KeplerOperationParameterDescriptor[] {
  return [...describeStayParameters(), describeConstant('activityMinimumDuration', 'basic')];
}
function describeHomeParameters(): KeplerOperationParameterDescriptor[] {
  return [
    describeConstant('gridWidth'),
    describeConstant('gridHeight'),
    {
      name: 'padding',
      defaultPolicy: {from: 'data', input: 'tracks', rule: 'extent-diagonal', factor: 0.1},
      level: 'advanced',
      apply: 'rebuild'
    },
    describeConstant('motionVariance', 'basic'),
    describeConstant('locationError', 'basic'),
    describeConstant('subStepCount')
  ];
}
function describeTrajectory(
  bindingId: string,
  action: string,
  parameters: KeplerOperationParameterDescriptor[],
  outputs: KeplerOperationOutputDescriptor[],
  notes: readonly string[],
  scalar?: KeplerOperationFieldDescriptor
): KeplerOperationDescriptor {
  return {
    bindingId,
    questions: ['how-it-moves'],
    action,
    prominence: 3,
    inputs: [
      {
        name: 'tracks',
        kind: 'trajectories',
        fields: [...describeFields(), ...(scalar ? [scalar] : [])]
      }
    ],
    parameters: [
      describeConstant('coordinateSystem', 'basic'),
      {
        name: 'timestampEncoding',
        defaultPolicy: {from: 'time-encoding', input: 'tracks', role: 'time'},
        level: 'basic',
        apply: 'rebuild'
      },
      ...parameters
    ],
    requirements: [
      ...describeRequirements(),
      ...(scalar
        ? [
            {
              input: 'tracks',
              role: scalar.role,
              id: 'no-missing-values' as const,
              statisticsScope: 'retained-geometry' as const,
              severity: 'block' as const,
              activeWhen: typeof scalar.required === 'object' ? scalar.required : undefined,
              remedy: 'Choose a scalar property present on every retained fix'
            }
          ]
        : [])
    ],
    outputs,
    notes: [INPUT_NOTE, TIME_NOTE, FRAME_NOTE, PROFILE_NOTE, ...notes]
  };
}

/** Hand-written semantics for all thirteen bounded trajectory catalog bindings. */
export const TRAJECTORY_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  describeTrajectory(
    'trajectory-hampel',
    'Filter trajectory spikes',
    [
      describeConstant('target', 'basic'),
      describeConstant('windowSize', 'basic'),
      describeConstant('numberOfSigmas', 'basic', 'live')
    ],
    [describeOutput('trajectory-fixes', 'points', 'outlier', 'categorical'), describeTracks()],
    [
      'The selected x or y is a local-frame coordinate; property mode requires a finite Float32 scalar on every fix and leaves path positions unchanged. filtered_value retains the selected scalar unit. Window endpoints pass through; an even windowSize uses 2*floor(windowSize/2)+1 fixes. Outlier marks are filter diagnostics, not significance tests.'
    ],
    {
      role: 'value',
      label: 'Scalar value to filter',
      parameter: 'valueProperty',
      accepts: ['number'],
      required: {parameter: 'target', equals: ['property']}
    }
  ),
  describeTrajectory(
    'trajectory-outlier-flags',
    'Flag implausible movement',
    [
      describeConstant('maximumSpeed', 'basic', 'live'),
      describeConstant('maximumAcceleration', 'basic', 'live')
    ],
    [
      describeOutput('trajectory-fixes', 'points', 'outlier_flags', 'categorical'),
      describeTracks()
    ],
    [
      'Zero disables each speed or acceleration limit. outlier_flags is a bit mask, not a ranked score. This point-centred single pass retains all fixes in the point table and joins unflagged fixes in cleaned paths; it is not a sequential last-kept-fix cleaner.'
    ]
  ),
  describeTrajectory(
    'trajectory-split',
    'Split tracks into parts',
    [
      ...describeStayParameters().map(parameter => ({
        ...parameter,
        activeWhen: {
          parameter: 'mode',
          equals: parameter.name === 'maximumGap' ? ['gap', 'stop'] : ['stop']
        }
      })),
      describeConstant('mode', 'basic')
    ],
    [describeTracks(), describeOutput('trajectory-fixes', 'points', 'track_id', 'categorical')],
    [
      STAY_NOTE,
      'Gap mode splits by maximumGap, stop mode uses computed stays, and value-change mode compares scalar strings, numbers or booleans as categories. Parts shorter than two fixes are dropped. Value-change boundaries repeat the boundary fix in adjacent paths.'
    ],
    {
      role: 'category',
      label: 'Split category',
      parameter: 'valueProperty',
      accepts: ['number', 'category', 'text', 'boolean'],
      measurement: ['nominal', 'ordinal', 'identifier'],
      required: {parameter: 'mode', equals: ['value-change']}
    }
  ),
  describeTrajectory(
    'trajectory-thinning',
    'Thin trajectory observations',
    [describeConstant('mode', 'basic'), describeConstant('tolerance', 'basic')],
    [describeTracks(), describeOutput('trajectory-fixes', 'points', 'track_id', 'categorical')],
    [
      'tolerance is a distance in the frame length unit for minimum-distance mode, or seconds for minimum-time mode. Both track endpoints are retained. The constant starting tolerance is not inferred from sampling cadence.'
    ]
  ),
  describeTrajectory(
    'trajectory-track-relative-times',
    'Compute time since track start',
    [describeConstant('maximumTimeErrorMilliseconds', 'basic')],
    [describeOutput('trajectory-fixes', 'points', 'relative_milliseconds'), describeTracks()],
    [
      'relative_milliseconds is Float32 milliseconds since each track origin, computed after exact Int64 subtraction. maximumTimeErrorMilliseconds=0 requires exact representation; an error above the budget refuses publication. Absolute path timestamps remain epoch milliseconds.'
    ]
  ),
  describeTrajectory(
    'trajectory-staypoints',
    'Find stops and dwell time',
    describeStayParameters(),
    [describeStaypoints()],
    [STAY_NOTE]
  ),
  describeTrajectory(
    'trajectory-segments',
    'Find triplegs and activity trips',
    describeSegmentParameters(),
    [describeTracks(), describeTracks('trajectory-trips'), describeStaypoints()],
    [
      STAY_NOTE,
      'activityMinimumDuration is seconds. Moving triplegs and activity-to-activity trips follow the existing event state machine; trip endpoints are determined by detected stays.'
    ]
  ),
  describeTrajectory(
    'trajectory-locations',
    'Group stops into locations',
    [...describeStayParameters(), describeConstant('epsilon', 'basic')],
    [describeOutput('trajectory-locations', 'points', 'staypoint_count'), describeStaypoints()],
    [
      STAY_NOTE,
      'epsilon is the location clustering radius in the frame length unit. Connected stays are grouped independently per track; location IDs are not shared place IDs across tracks.'
    ]
  ),
  describeTrajectory(
    'trajectory-recursions',
    'Measure revisits and residence',
    [describeConstant('radii', 'basic')],
    [describeOutput('trajectory-recursions', 'points', 'revisits')],
    [
      'radii contains one to eight positive distances in the frame length unit. Each fix/radius pair has revisits, residence_seconds and first_passage_seconds. A censored first passage is null with first_passage_censored=true; do not treat it as zero or a completed passage. Observation boundaries limit residence and revisit interpretation.'
    ]
  ),
  describeTrajectory(
    'trajectory-mobility-indicators',
    'Summarize track mobility',
    [...describeSegmentParameters(), describeConstant('epsilon', 'basic')],
    [describeOutput('trajectory-mobility', 'summary', 'tracking_quality'), describeStaypoints()],
    [
      STAY_NOTE,
      'Read the per-track indicator table alongside the stays. radius_of_gyration and jump_length use the frame length unit; waiting_seconds uses seconds. Home/work are the two locations with longest cumulative dwell, not time-of-day or visit-purpose classification. Missing home/work IDs are null.'
    ]
  ),
  describeTrajectory(
    'trajectory-modal-split',
    'Compare movement speed classes',
    [
      ...describeSegmentParameters(),
      describeConstant('metric', 'basic'),
      describeConstant('modeSpeedBreaks', 'basic')
    ],
    [
      describeOutput('trajectory-modal-split', 'summary', 'value'),
      {...describeTracks(), presentation: {column: 'mode', scale: 'categorical'}}
    ],
    [
      STAY_NOTE,
      'modeSpeedBreaks are positive, strictly ascending, exclusive speed upper bounds in length/second; the final class also catches speeds above the final bound. Mode numbers are speed classes, not inferred transport labels. The table value is count, distance in the frame length unit, or duration in seconds according to metric.'
    ]
  ),
  describeTrajectory(
    'trajectory-brownian-bridge',
    'Estimate track utilization density',
    describeHomeParameters(),
    [describeOutput('trajectory-home-range', 'cells', 'density')],
    [HOME_NOTE, 'This binding uses the existing CPU Brownian bridge wrapper.']
  ),
  describeTrajectory(
    'trajectory-isopleths',
    'Map home range probability cells',
    [...describeHomeParameters(), describeConstant('level', 'basic', 'live')],
    [describeOutput('trajectory-home-range', 'cells', 'density')],
    [
      HOME_NOTE,
      'level is included probability mass, not statistical confidence. Selected cells include the cell crossing that mass. These are individual grid polygons, not a dissolved contour. Density preparation uses the CPU bridge wrapper; only utilization-volume isopleths run on the GPU.'
    ]
  )
];
