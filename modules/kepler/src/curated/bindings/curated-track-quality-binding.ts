// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  GPU_PLANAR_SPATIAL_CONTEXT,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {
  addTrackQualityRecipe,
  GPU_TRACK_QUALITY_FLAGS
} from '@luma-contributors/analysis/gpu-network';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {
  appendCatalogRow,
  makeCatalogTable,
  type CatalogFeatureCollection
} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {
  MAXIMUM_TRAJECTORY_FIXES,
  MAXIMUM_TRAJECTORY_TRACKS,
  TRAJECTORY_PARAMETERS,
  prepareCatalogTrajectory,
  validateTrajectoryPublication,
  makeTrajectoryFixes,
  makeTrajectoryOutput,
  numberParameter,
  enumParameter,
  type CatalogTrajectory
} from '../../analysis/catalog-trajectory-common';

const BINDING_ID = 'curated-track-quality';
/** Interactive policy bounds, keeping thresholds finite after float32 upload. */
const MAXIMUM_SPEED = 1e7;
const MAXIMUM_GAP_SECONDS = 1e9;
const UNKNOWN_FLAGS = 0xffffffff;

export type TrackQualitySettings = {
  maximumSpeed: number;
  maximumGap: number;
  speedOutlierMode: 'step' | 'point-centred';
  coordinateSystem: 'EPSG:4326' | 'planar';
};

/** Reuse stable track/time packing and exact signed Int64 epoch words from catalog preparation. */
export function prepareTrackQuality(
  collection: CatalogFeatureCollection,
  parameters: Readonly<Record<string, CatalogParameterValue>> = {}
): {trajectory: CatalogTrajectory; settings: TrackQualitySettings; thresholds: Float32Array} {
  const values = Object.fromEntries(
    CURATED_TRACK_QUALITY_BINDING.parameters.map(parameter => [
      parameter.name,
      parameters[parameter.name] ?? parameter.defaultValue
    ])
  );
  const settings = values as TrackQualitySettings;
  validateRequirement(
    BINDING_ID,
    typeof settings.maximumSpeed === 'number',
    'maximumSpeed must be numeric in [0,10000000]'
  );
  validateRequirement(
    BINDING_ID,
    typeof settings.maximumGap === 'number',
    'maximumGap must be numeric in [0,1000000000]'
  );
  validateFiniteNumber(BINDING_ID, 'maximumSpeed', settings.maximumSpeed, 0, MAXIMUM_SPEED);
  validateFiniteNumber(BINDING_ID, 'maximumGap', settings.maximumGap, 0, MAXIMUM_GAP_SECONDS);
  validateRequirement(
    BINDING_ID,
    ['step', 'point-centred'].includes(settings.speedOutlierMode),
    'speedOutlierMode must be step or point-centred'
  );
  validateRequirement(
    BINDING_ID,
    ['EPSG:4326', 'planar'].includes(settings.coordinateSystem),
    'coordinateSystem must be EPSG:4326 or planar'
  );
  validateRequirement(
    BINDING_ID,
    ['unix-millisecond', 'unix-second', 'iso8601'].includes(String(values['timestampEncoding'])),
    'timestampEncoding must be unix-millisecond, unix-second or iso8601'
  );
  for (const name of ['trackProperty', 'timeProperty', 'longitudeProperty', 'latitudeProperty'])
    validateRequirement(
      BINDING_ID,
      typeof values[name] === 'string' && values[name].length <= 128,
      `${name} must be a string of at most 128 characters`
    );
  const trajectory = prepareCatalogTrajectory(collection, values);
  validateTrajectoryPublication(trajectory);
  return {
    trajectory,
    settings,
    // Recipe distances are per millisecond; interface speeds and gaps are per second.
    thresholds: new Float32Array([settings.maximumSpeed / 1000, settings.maximumGap * 1000])
  };
}

export type TrackQualityReadback = {
  stepSpeeds: Float32Array;
  trackLengths: Float32Array;
  trackDurations: Float32Array;
  maximumSpeeds: Float32Array;
  flags: Uint32Array;
  flaggedCounts: Uint32Array;
};

/** Keep every raw fix and source identity; flags describe checks, not edits or significance. */
export function publishTrackQuality(
  trajectory: CatalogTrajectory,
  settings: TrackQualitySettings,
  result: TrackQualityReadback
): CatalogRunOutput {
  const finite = (value: number): number | null => (Number.isFinite(value) ? value : null);
  const fixes = makeTrajectoryFixes(
    trajectory,
    {
      step_speed: 'float64',
      time_gap_seconds: 'float64',
      has_previous_step: 'boolean',
      quality_flags: 'int32',
      quality_valid: 'boolean',
      flagged: 'boolean',
      excessive_speed: 'boolean',
      time_gap: 'boolean'
    },
    undefined,
    row => {
      const fix = trajectory.fixes[row];
      const hasPrevious = row > trajectory.trackOffsets[fix.trackIndex];
      const flags = result.flags[row];
      const known = flags !== UNKNOWN_FLAGS;
      return {
        step_speed: finite(result.stepSpeeds[row] * 1000),
        time_gap_seconds: hasPrevious
          ? (fix.timestamp - trajectory.fixes[row - 1].timestamp) / 1000
          : null,
        has_previous_step: hasPrevious,
        quality_flags: known ? flags : null,
        quality_valid: known,
        flagged: known ? flags !== 0 : null,
        excessive_speed: known ? Boolean(flags & GPU_TRACK_QUALITY_FLAGS.excessiveSpeed) : null,
        time_gap: known ? Boolean(flags & GPU_TRACK_QUALITY_FLAGS.timeGap) : null
      };
    }
  );
  fixes.coordinateSystem = settings.coordinateSystem;
  const tracks = makeCatalogTable('track-quality-summary', 'Track quality summary', {
    track_index: 'int32',
    track_id: 'string',
    fix_count: 'int32',
    flagged_count: 'int32',
    track_length: 'float64',
    duration_seconds: 'float64',
    maximum_speed: 'float64',
    start_time: 'float64',
    end_time: 'float64'
  });
  trajectory.trackIds.forEach((trackId, trackIndex) => {
    const begin = trajectory.trackOffsets[trackIndex];
    const end = trajectory.trackOffsets[trackIndex + 1];
    appendCatalogRow(tracks, {
      track_index: trackIndex,
      track_id: trackId,
      fix_count: end - begin,
      flagged_count: result.flaggedCounts[trackIndex],
      track_length: finite(result.trackLengths[trackIndex]),
      duration_seconds: finite(result.trackDurations[trackIndex] / 1000),
      maximum_speed: finite(result.maximumSpeeds[trackIndex] * 1000),
      start_time: trajectory.fixes[begin].timestamp,
      end_time: trajectory.fixes[end - 1].timestamp
    });
  });
  const unknownCount = result.flags.filter(flags => flags === UNKNOWN_FLAGS).length;
  const output = makeTrajectoryOutput(trajectory, [fixes, tracks], {
    flaggedCount: result.flaggedCounts.reduce((total, count) => total + count, 0),
    unknownCount,
    maximumSpeed: settings.maximumSpeed,
    maximumGap: settings.maximumGap,
    effectiveMaximumSpeed: Math.fround(settings.maximumSpeed / 1000) * 1000,
    effectiveMaximumGap: Math.fround(settings.maximumGap * 1000) / 1000,
    speedOutlierMode: settings.speedOutlierMode,
    coordinateSystem: settings.coordinateSystem,
    lengthUnit: settings.coordinateSystem === 'EPSG:4326' ? 'metres' : 'coordinate units',
    speedUnit:
      settings.coordinateSystem === 'EPSG:4326' ? 'metres/second' : 'coordinate units/second',
    durationUnit: 'seconds',
    timestampUnit: 'epoch milliseconds',
    flagBits: JSON.stringify(GPU_TRACK_QUALITY_FLAGS)
  });
  if (unknownCount)
    output.diagnostics.push({
      code: 'track-quality-unknown',
      severity: 'error',
      count: unknownCount,
      message:
        'Unwritten recipe flags are unknown; affected observations must not be treated as clean.'
    });
  return output;
}

export const CURATED_TRACK_QUALITY_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Track quality',
  domain: 'analysis',
  summary:
    'Check raw observations for excessive step speeds and time gaps, with per-track metrics.',
  capabilityIds: ['movement-analysis'],
  contributorNames: ['GPUTrajectoryMetrics'],
  gpuContributorNames: ['GPUTrajectoryMetrics'],
  engine: 'gpu',
  inputs: [
    {
      name: 'tracks',
      label: 'Trajectory observations',
      accepts: ['point', 'line'],
      maximumFeatures: MAXIMUM_TRAJECTORY_FIXES,
      maximumVertices: MAXIMUM_TRAJECTORY_FIXES
    }
  ],
  parameters: [
    ...TRAJECTORY_PARAMETERS,
    numberParameter('maximumSpeed', 'Maximum speed', 50, 0, MAXIMUM_SPEED, 'length / second'),
    numberParameter('maximumGap', 'Maximum observation gap', 60, 0, MAXIMUM_GAP_SECONDS, 'seconds'),
    enumParameter('speedOutlierMode', 'Speed check', ['step', 'point-centred'])
  ],
  outputs: [
    {
      tableId: 'trajectory-fixes',
      label: 'Observation quality',
      geometry: 'point',
      identityColumns: ['source_index', 'vertex_index', 'fix_index', 'track_id'],
      maximumRows: MAXIMUM_TRAJECTORY_FIXES
    },
    {
      tableId: 'track-quality-summary',
      label: 'Track metrics',
      geometry: 'none',
      identityColumns: ['track_index', 'track_id'],
      maximumRows: MAXIMUM_TRAJECTORY_TRACKS
    }
  ],
  cleanup:
    'The session releases buffers and compiled graphs in finally; returned tables own plain arrays.',
  declaredBehavior: [
    'Reuses catalog trajectory preparation: Point fixes or timed LineStrings, stable first-seen track order and time sorting, original source and vertex identities, and exact signed Int64 epoch millisecond words. Track IDs must have distinct string spellings before GPU allocation.',
    'Missing IDs, invalid positions/times and duplicate timestamps reject preparation; null geometries are skipped and counted. At most 2048 fixes and 128 tracks.',
    'Geographic observations use the existing local azimuthal equidistant frame in metres; planar data uses shifted coordinate units. Metrics are two dimensional; source altitude survives publication.',
    'Speeds and gap thresholds use seconds at the interface and milliseconds in the recipe. Zero is a literal threshold, not a disable switch. Recipe comparisons are strictly greater than the uploaded float32 thresholds.',
    'Step mode flags the step ending at a fix, including the valid fix after a displaced observation. Point-centred mode uses the recipe single pass over adjacent steps. All observations remain in the output.',
    'First fixes have speed zero and no elapsed step (null time_gap_seconds). Singleton tracks have zero length, duration and maximum speed. Flags are bits: invalidPosition=1, invalidTime=2, excessiveSpeed=4, timeGap=8, invalidThreshold=16; an unwritten word is unknown.',
    'Per-track flagged_count counts observations with any flag, not the number of bits. Turning angles, acceleration, reference-route comparison and map matching are not computed.'
  ],
  limitation:
    'Requires WebGPU; no CPU fallback. Float32 metrics and thresholds in one bounded local frame are approximate away from its origin. Invalid or duplicate observations are rejected before the recipe; flags are plausibility checks, not evidence of an erroneous observation or statistical significance.',
  async runGpu(context, inputs, parameters) {
    const {trajectory, settings, thresholds} = prepareTrackQuality(inputs['tracks'], parameters);
    return withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const recipe = addTrackQualityRecipe(session.graph, {
        id: BINDING_ID,
        trajectory: {
          positions: session.input('positions', trajectory.positions, 'float32x2'),
          timestamps: session.input('timestamps', trajectory.timestamps, 'uint32x2'),
          trackOffsets: session.input('track-offsets', trajectory.trackOffsets, 'uint32'),
          spatialContext: trajectory.frame.gpuFrame?.spatialContext ?? GPU_PLANAR_SPATIAL_CONTEXT
        },
        thresholds: session.input('thresholds', thresholds, 'float32'),
        speedOutlierMode: settings.speedOutlierMode
      });
      // Recipe-owned views are transient. Copy terminal ports into host-owned readback buffers.
      function capture<Format extends 'float32' | 'uint32'>(
        name: string,
        source: GraphDataView<Format>
      ) {
        const destination = session.output(name, source.format, source.length);
        const type = source.format === 'float32' ? 'f32' : 'u32';
        session.graph.add(
          createWGSLKernelNode(session.graph, {
            id: `${BINDING_ID}-${name}`,
            operation: BINDING_ID,
            invocationCount: source.length,
            bindings: [
              {name: 'source', view: source, type, access: 'read'},
              {name: 'destination', view: destination.view, type, access: 'read_write'}
            ],
            body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
          })
        );
        return destination;
      }
      const {metrics, flags, flaggedCounts} = recipe.outputs;
      const captured = {
        stepSpeeds: capture('step-speeds', metrics.stepSpeeds),
        trackLengths: capture('track-lengths', metrics.trackLengths),
        trackDurations: capture('track-durations', metrics.trackDurations),
        maximumSpeeds: capture('maximum-speeds', metrics.maximumSpeeds),
        flags: capture('flags', flags),
        flaggedCounts: capture('flagged-counts', flaggedCounts)
      };
      session.run();
      const [stepSpeeds, trackLengths, trackDurations, maximumSpeeds, flagValues, counts] =
        await Promise.all([
          session.readFloat32(captured.stepSpeeds),
          session.readFloat32(captured.trackLengths),
          session.readFloat32(captured.trackDurations),
          session.readFloat32(captured.maximumSpeeds),
          session.readUint32(captured.flags),
          session.readUint32(captured.flaggedCounts)
        ]);
      return {
        ...publishTrackQuality(trajectory, settings, {
          stepSpeeds,
          trackLengths,
          trackDurations,
          maximumSpeeds,
          flags: flagValues,
          flaggedCounts: counts
        }),
        dispatchedContributors: ['GPUTrajectoryMetrics']
      };
    });
  }
};
