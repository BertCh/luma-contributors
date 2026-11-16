// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerOperationDescriptor} from './operation-descriptor-types';
import {FLEET_DWELL_LIMITS} from '../curated/bindings/curated-fleet-dwell-binding';

export const CURATED_FLEET_DWELL_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: 'curated-fleet-dwell',
  questions: ['how-it-moves'],
  action: 'Map vehicle stop dwell',
  prominence: 2,
  inputs: [
    {
      name: 'tracks',
      kind: 'trajectories',
      fields: [
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
      ]
    },
    {name: 'zones', kind: 'polygons', fields: []}
  ],
  parameters: [
    ...[
      'coordinateSystem',
      'stopCapacity',
      'candidateCapacity',
      'stopSpeedThreshold',
      'stopMinimumDuration'
    ].map(name => ({
      name,
      apply: 'rebuild' as const,
      level:
        name.startsWith('stop') && name !== 'stopCapacity'
          ? ('basic' as const)
          : ('advanced' as const),
      defaultPolicy: {from: 'constant' as const}
    })),
    {
      name: 'timestampEncoding',
      apply: 'rebuild',
      level: 'basic',
      defaultPolicy: {from: 'time-encoding', input: 'tracks', role: 'time'}
    }
  ],
  requirements: [
    {
      input: 'tracks',
      id: 'minimum-rows',
      severity: 'block',
      limit: 1,
      remedy:
        'Supply point fixes or timed LineStrings with a track identity and timestamp for every fix'
    },
    {
      input: 'tracks',
      id: 'maximum-rows',
      severity: 'block',
      limit: FLEET_DWELL_LIMITS.fixes,
      remedy: 'Filter to at most 2,048 fixes'
    },
    {
      input: 'tracks',
      id: 'maximum-vertices',
      severity: 'block',
      limit: FLEET_DWELL_LIMITS.fixes,
      remedy: 'Thin timed paths to at most 2,048 fixes'
    },
    {
      input: 'tracks',
      role: 'track',
      id: 'maximum-distinct-values',
      severity: 'block',
      limit: FLEET_DWELL_LIMITS.tracks,
      remedy: 'Filter to at most 128 tracks; preparation also counts fallback feature IDs'
    },
    {
      input: 'zones',
      id: 'minimum-rows',
      severity: 'block',
      limit: 1,
      remedy: 'Supply at least one polygon zone'
    },
    {
      input: 'zones',
      id: 'maximum-rows',
      severity: 'block',
      limit: FLEET_DWELL_LIMITS.zones,
      remedy: 'Filter to at most 128 zones'
    },
    {
      input: 'zones',
      id: 'maximum-vertices',
      severity: 'block',
      limit: FLEET_DWELL_LIMITS.zoneVertices,
      remedy: 'Simplify zones to at most 16,384 vertices'
    },
    ...['tracks', 'zones'].map(input => ({
      input,
      id: 'declared-coordinates' as const,
      severity: 'block' as const,
      remedy: 'Confirm a shared coordinate system for tracks and zones and the speed length unit'
    }))
  ],
  notes: [
    'Point fixes read explicit track/time columns; timed LineStrings may use feature IDs and vertex times or timestamp arrays. Embedded alternatives are checked during preparation, beyond the profile. Fixes sort by track/time, retaining original row and vertex identities. Duplicate track timestamps reject the run.',
    'Speed defaults to a stated 1 length unit/second and minimum dwell to 20 seconds; neither is estimated from the data. Geographic positions use a shared local frame in metres; planar coordinates keep their declared units. All reported dwell is seconds and absolute timestamps are epoch milliseconds.',
    'Slow-step runs define stops; zone totals use the arithmetic stop centroid, not time spent inside zone boundaries. Half-open boundaries and the smallest containing feature row resolve overlap. Holes and multipart zones are retained. Outside stops remain visible; empty zones have zero total and null mean/maximum.',
    'Sampling cadence, GPS error, long gaps and observation endpoints affect dwell. Gaps are not capped; clean or split tracks first. No stop-purpose inference or uncertainty interval. Float32/local projection is approximate. Any overflow or uncertain join candidate prevents publication; padding stop slots also contribute to the recipe uncertainty count.'
  ],
  outputs: [
    {
      tableId: 'fleet-stops',
      kind: 'points',
      presentation: {column: 'duration_seconds', scale: 'sequential'},
      interpretation: {summary: 'value-range', cautions: []},
      next: ['how-it-moves']
    },
    {
      tableId: 'fleet-zone-dwell',
      kind: 'polygons',
      kindFrom: 'zones',
      presentation: {column: 'total_dwell_seconds', scale: 'sequential'},
      interpretation: {summary: 'value-range', cautions: []},
      next: ['how-it-moves']
    }
  ]
};
