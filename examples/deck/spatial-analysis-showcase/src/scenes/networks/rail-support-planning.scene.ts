// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {RailSupportPlanningOptions} from './rail-support-planning.compute';

export default defineScene<RailSupportPlanningOptions>({
  id: 'rail-support-planning',
  title: 'Where should rail support desks serve?',
  chapter: 'networks',
  order: 11,
  summary:
    'Compare Dutch rail service areas, travel-time catchments and station assignments for four hypothetical support desks.',
  contributors: ['GPUNetworkCostMatrix', 'GPUNetworkServiceAreas', 'GPUNetworkCostWeights'],
  datasets: [{id: 'gtfs-nl-rail-graph', role: 'Dutch scheduled rail links and station locations'}],
  initialView: {longitude: 5.25, latitude: 52.15, zoom: 7.05},
  basemap: ground('paperSheet', {world: 'land', labels: 'above', labelPreset: 'orientation'}),
  furniture: {
    title: {
      subtitle: 'Hypothetical support desks · scheduled median in-vehicle minutes',
      chips: ['One unit of demand per station']
    },
    scaleBar: {units: 'metric', latitude: 52},
    credit: 'Bundled GTFS-NL rail graph · station geometry and median scheduled link times',
    caveat:
      'Desk locations and station demand are planning assumptions. The routing model omits waiting, transfers, staffing capacity and time-of-day variation.'
  },
  options: [
    {
      kind: 'select',
      id: 'direction',
      label: 'Service direction',
      group: 'Catchment',
      apply: 'param',
      default: 'inbound',
      help: 'Inbound assigns each station to the nearest desk by station-to-desk rail cost. Outbound asks which stations each desk can reach.',
      options: [
        {value: 'inbound', label: 'Station to desk'},
        {value: 'outbound', label: 'Desk to station'}
      ]
    },
    {
      kind: 'slider',
      id: 'costLimitMinutes',
      label: 'Travel-time catchment',
      group: 'Catchment',
      apply: 'param',
      min: 30,
      max: 240,
      step: 15,
      default: 120,
      unit: 'min',
      help: 'Stations beyond this cost stay unassigned. The same cutoff bounds the GPU distance-decay weights.'
    },
    {
      kind: 'select',
      id: 'decay',
      label: 'Catchment decay',
      group: 'Service weights',
      apply: 'param',
      default: 'exponential',
      options: [
        {value: 'none', label: 'Uniform'},
        {value: 'exponential', label: 'Exponential'},
        {value: 'power', label: 'Power'},
        {value: 'gaussian', label: 'Gaussian'}
      ],
      help: 'Transforms in-catchment minutes to a dimensionless weight; this is not passenger demand.'
    },
    {
      kind: 'slider',
      id: 'beta',
      label: 'Decay rate',
      group: 'Service weights',
      apply: 'param',
      min: 0,
      max: 0.08,
      step: 0.005,
      default: 0.02,
      help: 'For exponential decay, beta is per minute. For power decay it is dimensionless; Gaussian uses the selected catchment as its scale.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map measure',
      group: 'Map',
      apply: 'param',
      default: 'assignments',
      help: 'Compare the nearest-desk categorical assignment with the sum of distance-decay weights from all four candidate desks.',
      options: [
        {value: 'assignments', label: 'Nearest desk'},
        {value: 'weightedReach', label: 'Weighted desk reach'}
      ]
    }
  ],
  story: [
    {
      id: 'hypothetical-desks',
      headline: 'Four hypothetical desks divide the rail network into service catchments.',
      title: 'Four candidate desks, equal station demand',
      body: 'The four candidate hubs are **Amsterdam, Utrecht, Rotterdam and Zwolle**. Each station contributes one unit to make assignments easy to compare; this is not a passenger estimate.',
      evidence: '**{{stationsServed}}** are assigned within the current catchment.',
      caveat:
        'The desk locations are hypothetical planning candidates, not current staffed facilities.',
      textAlternative:
        'Dutch stations are colored by their nearest candidate desk under scheduled rail costs.',
      controls: ['direction'],
      readouts: ['stationsServed', 'deskSummary']
    },
    {
      id: 'network-service-area',
      headline: 'Directed station-to-desk costs can differ from outward desk reach.',
      title: 'The nearest desk follows the rail graph',
      body: 'GPU service areas route through directed rail links and assign each station to the lowest-cost reachable candidate. Toggle **Service direction** to compare inbound station-to-desk routing with outward desk reach.',
      evidence: 'The cost model uses **{{travelModel}}**.',
      caveat:
        'The bundled graph represents median scheduled in-vehicle link times; it assumes perfect transfers and no waiting.',
      textAlternative:
        'Station colors show the nearest reachable desk; markers show the four candidate hubs.',
      controls: ['direction'],
      readouts: ['travelModel', 'solver']
    },
    {
      id: 'catchment',
      headline: 'A stricter time budget leaves more stations without an assigned desk.',
      title: 'A time budget defines the catchment',
      body: 'Reduce **Travel-time catchment** to see which stations fall outside a two-hour service area. Unassigned stations can be too far away or disconnected in the selected direction.',
      evidence: '**{{unassigned}}** are beyond the current limit or unreachable.',
      caveat: 'This cutoff is a planning threshold, not a promise of response time.',
      textAlternative: 'Stations outside the chosen rail-cost limit are gray.',
      controls: ['costLimitMinutes'],
      readouts: ['unassigned', 'solver']
    },
    {
      id: 'decay-weights',
      headline: 'The decay assumption changes how much nearby and distant desks contribute.',
      title: 'Distance decay adds a service-weight view',
      body: 'GPU cost weights convert each desk-to-station cost row into a sparse catchment with selectable uniform, exponential, power or Gaussian decay. Adjust **Decay rate** to change the relative weight of farther stations, then switch the map to **Weighted desk reach**.',
      evidence: '**{{weightedPairs}}** are summarized within the selected catchment.',
      caveat:
        'Weights express the chosen planning decay and equal-unit station model; they are not observed service usage. Exponential beta is per minute; power beta is dimensionless.',
      textAlternative:
        'The weighted map sums the distance-decay weights from all four candidate desks at each station.',
      controls: ['decay', 'beta', 'costLimitMinutes', 'display'],
      readouts: ['weightedPairs', 'solver']
    }
  ],
  legends: state =>
    state.display === 'weightedReach'
      ? [
          {
            kind: 'ramp',
            title: 'Summed desk weights',
            ramp: 'cividis',
            extent: [0, 4],
            unit: 'dimensionless',
            note: 'Sum across four candidate desks; only costs inside the selected time limit contribute.'
          }
        ]
      : [
          {
            kind: 'categories',
            title: 'Assigned support desk',
            entries: [
              {color: [36, 132, 173, 255], label: 'Amsterdam'},
              {color: [238, 132, 43, 255], label: 'Utrecht'},
              {color: [135, 91, 177, 255], label: 'Rotterdam'},
              {color: [45, 153, 112, 255], label: 'Zwolle'},
              {color: [150, 154, 158, 180], label: 'Unassigned'}
            ],
            note: 'All marked desks and one-unit station demands are hypothetical.'
          }
        ],
  readouts: [
    {id: 'stationsServed', label: 'Stations assigned'},
    {id: 'deskSummary', label: 'Candidate hubs'},
    {id: 'unassigned', label: 'Outside catchment'},
    {id: 'weightedPairs', label: 'Weighted station-desk pairs'},
    {id: 'travelModel', label: 'Routing assumption'},
    {id: 'solver', label: 'GPU status'}
  ],
  snippet: state =>
    `new GPUNetworkServiceAreas({topology, facilities, costLimit}); // ${state.direction} · ${state.costLimitMinutes} min`,
  about: {
    what: 'A GPU network allocation view for four hypothetical rail-support desks.',
    why: 'It shows how direction, a travel-time cutoff and distance decay change station coverage.',
    howToRead:
      'Nearest desk colors show the lowest-cost reachable hub. Weighted reach sums dimensionless distance-decay values from all hubs. Gray means no desk lies within the selected cost limit. Weights are scenario values, not observed demand.'
  },
  create: async ctx =>
    (await import('./rail-support-planning.compute')).createRailSupportPlanning(ctx)
});
