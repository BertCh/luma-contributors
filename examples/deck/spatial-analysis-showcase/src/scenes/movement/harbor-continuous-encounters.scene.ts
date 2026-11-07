// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {HarborContinuousOptions} from './harbor-continuous-encounters.compute';

export default defineScene<HarborContinuousOptions>({
  id: 'harbor-continuous-encounters',
  title: 'Which close passes do sparse clocks miss?',
  chapter: 'movement',
  order: 7,
  summary:
    'Compare shared-clock AIS encounters with continuous closest approach and a gap-aware pairwise restriction on eight audited passenger-vessel paths.',
  contributors: [
    'GPUTrajectoryClosestApproach',
    'addClockEncounters',
    'GPUTrajectoryEncounters',
    'GPUTrajectoryRestriction'
  ],
  datasets: [{id: 'ais-vessels', role: 'NY/NJ Harbor AIS tracks, 12 June 2024'}],
  initialView: {longitude: -74.015, latitude: 40.695, zoom: 11.2},
  basemap: ground('night', {labels: 'above'}),
  furniture: {
    title: {
      subtitle: 'Eight passenger-vessel tracks · 14:00–15:00 UTC',
      chips: ['Audited observation support']
    },
    scaleBar: {units: 'metric', latitude: 40.69},
    credit: 'NOAA Office for Coastal Management / U.S. Coast Guard Navigation Center · CC0 1.0',
    caveat:
      'Continuous approach assumes linear motion between observed fixes. No values are extrapolated beyond each path. Pairwise restriction marks support gaps above the chosen limit unknown; closest approach has no maximum-gap option.'
  },
  options: [
    {
      kind: 'slider',
      id: 'distanceMeters',
      label: 'Encounter distance',
      group: 'Search',
      apply: 'param',
      min: 50,
      max: 500,
      step: 25,
      default: 250,
      unit: 'm',
      help: 'Pairs match at or below this planar distance. The exact and minute-sampled analyses share the same threshold.'
    },
    {
      kind: 'slider',
      id: 'maximumGapSeconds',
      label: 'Restriction gap limit',
      group: 'Support',
      apply: 'param',
      min: 90,
      max: 300,
      step: 10,
      default: 300,
      unit: 's',
      help: 'Pairwise restriction stops interpolation across longer AIS gaps and records the interval as unknown support.'
    },
    {
      kind: 'select',
      id: 'clockStep',
      label: 'Shared sampling clock',
      group: 'Sampling comparison',
      apply: 'param',
      default: '60',
      options: [
        {value: '60', label: '60 seconds'},
        {value: '300', label: '5 minutes'},
        {value: '600', label: '10 minutes'}
      ],
      help: 'Compare the exact continuous result with co-located fixes sampled on a common UTC-aligned clock. All three clock variants are computed by the GPU.'
    },
    {
      kind: 'select',
      id: 'timeBoundary',
      label: 'Time endpoints',
      group: 'Restriction',
      apply: 'param',
      default: 'closed',
      options: [
        {value: 'closed', label: 'Include both'},
        {value: 'open', label: 'Exclude both'},
        {value: 'left-closed', label: 'Include start'},
        {value: 'right-closed', label: 'Include end'}
      ],
      help: 'Select how exact 14:00 and 15:00 endpoints participate in pairwise time support.'
    },
    {
      kind: 'select',
      id: 'geometryBoundary',
      label: 'Distance endpoint',
      group: 'Restriction',
      apply: 'param',
      default: 'closed',
      options: [
        {value: 'closed', label: 'At or within distance'},
        {value: 'open', label: 'Strictly within distance'}
      ],
      help: 'Choose whether equality with the encounter radius counts as inside.'
    }
  ],
  story: [
    {
      id: 'audited-window',
      title: 'Start with eight real paths and known support',
      headline: 'Eight sampled tracks bound the motion comparison.',
      body: 'The analysis follows public AIS path rows 221, 370, 509, 562, 595, 712, 726 and 735 during 14:00–15:00 UTC on 12 June 2024. Each has at least 40 fixes in this window, and the largest observed interval is under five minutes.',
      evidence: '**{{selectedTracks}}**',
      caveat:
        'The paths are an audited sample of passenger vessels, not all harbor traffic. AIS reporting intervals still vary, so the fix sequence is not continuous observation.',
      textAlternative:
        'Eight passenger paths cross the Upper Bay and East River during one UTC hour.',
      controls: [],
      readouts: ['selectedTracks', 'samplingAudit', 'modelAssumption']
    },
    {
      id: 'clock-vs-continuous',
      title: 'Sparse clocks can miss short passes',
      headline: 'A shared clock misses some between-fix passes.',
      body: 'A UTC-aligned shared clock compares interpolated positions every 60 seconds, 5 minutes or 10 minutes. The separate closest-approach contributor checks minimum separation while each pair of observed segments overlaps.',
      evidence: '**{{encounterCounts}}** · **{{samplingDifference}}**.',
      caveat:
        'Clock encounters are a discrete approximation. Closest approach is exact for linear interpolation between these fixes, while any real motion between fixes is unknown.',
      textAlternative:
        'Readouts compare several shared-clock sampling intervals with continuous linear-segment closest approach.',
      controls: ['distanceMeters', 'clockStep'],
      readouts: ['encounterCounts', 'samplingDifference', 'closestDistances', 'closestPairs']
    },
    {
      id: 'gap-aware-restriction',
      title: 'Treat long intervals as unknown support',
      headline: 'Long source gaps become unknown, not bridges.',
      body: 'Pairwise trajectory restriction independently computes the time intervals where each selected pair lies within the distance. It preserves the exact endpoint policy and stops bridging when a source interval exceeds the chosen maximum gap.',
      evidence: '**{{restrictionResult}}**.',
      caveat:
        'The interval layer shows GPU-clipped segments. Open and closed choices affect equality at time and distance boundaries; they do not improve the underlying AIS sampling.',
      textAlternative:
        'Cyan line segments show pairwise time intervals inside the chosen radius, with endpoint and maximum-gap controls.',
      controls: ['maximumGapSeconds', 'timeBoundary', 'geometryBoundary'],
      readouts: ['restrictionResult', 'solverStatus']
    },
    {
      id: 'boundary-semantics',
      title: 'Boundary choices change equality only',
      headline: 'Endpoint rules define equality, not certainty.',
      body: 'The interval output follows the selected open or closed time and distance boundaries. A boundary choice can include or exclude a fix exactly at 14:00, 15:00 or the selected radius; it cannot fill an unsupported interval or reveal unobserved motion.',
      evidence: '**{{restrictionResult}}** · **{{samplingAudit}}**.',
      caveat:
        'Intervals are derived from straight-line interpolation only where source gaps pass the configured support limit. A missing interval remains unknown, not evidence that vessels were apart.',
      textAlternative:
        'The GPU restriction marks within-radius intervals and reports gaps withheld by the support limit; endpoint controls affect equality cases.',
      controls: ['timeBoundary', 'geometryBoundary', 'maximumGapSeconds'],
      readouts: ['restrictionResult', 'samplingAudit']
    }
  ],
  legends: () => [
    {
      kind: 'line',
      title: 'Selected passenger-vessel tracks',
      entries: [
        {color: [89, 117, 136, 150], widthPixels: 2, label: 'Observed path'},
        {color: [35, 210, 203, 225], widthPixels: 4, label: 'Within-distance interval'}
      ],
      note: 'Interval rendering is withheld while a parameter update is running.'
    }
  ],
  readouts: [
    {id: 'selectedTracks', label: 'Source paths / MMSIs'},
    {id: 'samplingAudit', label: 'Sampling support'},
    {id: 'modelAssumption', label: 'Motion model'},
    {id: 'encounterCounts', label: 'Continuous vs sampled pairs'},
    {id: 'samplingDifference', label: 'Pairs missed by clock'},
    {id: 'closestPairs', label: 'Closest continuous pairs'},
    {id: 'closestDistances', label: 'Continuous minimum distances'},
    {id: 'restrictionResult', label: 'Pairwise restriction'},
    {id: 'solverStatus', label: 'GPU capacities'}
  ],
  snippet: state =>
    `new GPUTrajectoryClosestApproach({distance: ${state.distanceMeters}, cellSize: 1000, timeCellSize: 120});`,
  about: {
    what: 'Continuous and sampled encounter analysis for eight actual NY/NJ Harbor AIS paths.',
    why: 'A shared clock can miss passes between sample instants; linear closest approach answers a different, explicit interpolation question.',
    howToRead:
      'Compare pair counts, inspect the cyan pairwise-restriction intervals, and use the support-gap limit to see where straight-line interpolation is withheld. The sample is bounded to fixes in the selected hour.'
  },
  create: async ctx =>
    (await import('./harbor-continuous-encounters.compute')).createHarborContinuousEncounters(ctx)
});
