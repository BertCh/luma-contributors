// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec, type OptionSpec} from '../scene';
import type {ChicagoRouteWorkbenchOptions} from './chicago-route-workbench.compute';

const CHICAGO_ROUTE_WORKBENCH_OPTIONS: readonly OptionSpec<ChicagoRouteWorkbenchOptions>[] = [
  {
    kind: 'select',
    id: 'routePair',
    label: 'Edge-position query',
    group: 'Route inquiry',
    apply: 'param',
    default: 'east-north',
    options: [
      {value: 'east-north', label: 'East → north extract edge'},
      {value: 'east-west', label: 'East → west extract edge'},
      {value: 'north-west', label: 'North → west extract edge'}
    ],
    help: 'Queries use retained links near the east, north and west boundaries of the capped extract; they are not routes to landmarks outside it. The links define both the point-on-edge query and the source and destination vertices of the time-dependent route.',
    disabledWhen: state => state.turnRestriction === 'hypothetical-ban'
  },
  {
    kind: 'select',
    id: 'turnRestriction',
    label: 'Directed turn scenario',
    group: 'Hypothetical route restriction',
    apply: 'param',
    default: 'none',
    options: [
      {value: 'none', label: 'No sequence restriction'},
      {value: 'hypothetical-ban', label: 'Forbid directed links 66 → 38'}
    ],
    help: 'The second choice forbids the consecutive directed edge-ID sequence 66→38 on the default east-to-north route. The bundled OSM extract has no turn-restriction relations; this is a hypothetical routing rule, not an observed road sign or OSM restriction.'
  },
  {
    kind: 'slider',
    id: 'originFraction',
    label: 'Origin position on link',
    group: 'Route inquiry',
    apply: 'param',
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.2,
    help: 'Fraction measured in the selected edge direction. Point routing charges only the remaining edge fraction.'
  },
  {
    kind: 'slider',
    id: 'destinationFraction',
    label: 'Destination position on link',
    group: 'Route inquiry',
    apply: 'param',
    min: 0,
    max: 1,
    step: 0.05,
    default: 0.8,
    help: 'Fraction measured in the selected edge direction. The destination pays its leading fraction; reverse travel requires a reverse directed edge.'
  },
  {
    kind: 'slider',
    id: 'departureMinute',
    label: 'Departure clock time',
    group: 'Time-dependent routing',
    apply: 'param',
    min: 0,
    max: 720,
    step: 30,
    default: 240,
    unit: 'min after 08:00',
    format: value =>
      `${String(Math.floor((8 * 60 + value) / 60) % 24).padStart(2, '0')}:${String(Math.floor((8 * 60 + value) % 60)).padStart(2, '0')}`,
    help: 'Changes the departure time used by both FIFO earliest-arrival models. Times are local Chicago clock time relative to 08:00.'
  },
  {
    kind: 'select',
    id: 'timeModel',
    label: 'Arrival model',
    group: 'Time-dependent routing',
    apply: 'param',
    default: 'affine-duration',
    display: 'segmented',
    options: [
      {value: 'affine-duration', label: 'Affine duration'},
      {value: 'piecewise-arrival', label: 'Piecewise arrivals'}
    ],
    help: 'Both variants are computed. This selects which result is summarized. Piecewise mode interpolates a hypothetical FIFO arrival envelope; it is not a transit timetable or measured traffic forecast.'
  },
  {
    kind: 'select',
    id: 'firstSampleMinute',
    label: 'First modeled departure',
    group: 'Time-dependent routing',
    apply: 'param',
    default: '0',
    options: [
      {value: '0', label: '08:00 · no initial opening wait'},
      {value: '15', label: '08:15 · wait if departing earlier'},
      {value: '60', label: '09:00 · wait if departing earlier'}
    ],
    help: 'Shifts every piecewise departure and arrival sample together. Before its first sample, an edge waits for that sample; after its last, duration is held constant. This is a hypothetical availability envelope on real street geometry, not a recorded opening restriction.'
  },
  {
    kind: 'slider',
    id: 'affineSlope',
    label: 'Affine duration slope',
    group: 'Time-dependent routing',
    apply: 'param',
    min: 0,
    max: 0.004,
    step: 0.00025,
    default: 0.001,
    format: value => `${value.toFixed(5)} min / departure-min`,
    disabledWhen: state => state.timeModel !== 'affine-duration',
    help: 'Sets the hypothetical nonnegative duration slope in minutes per departure minute for every edge. A nonnegative slope preserves FIFO order.'
  },
  {
    kind: 'toggle',
    id: 'showAlternatives',
    label: 'Draw computed alternatives',
    group: 'Display',
    apply: 'param',
    default: true
  },
  {
    kind: 'slider',
    id: 'costLimitMinutes',
    label: 'Reusable cost-table limit',
    group: 'Cost-table reuse',
    apply: 'param',
    min: 2,
    max: 30,
    step: 1,
    default: 12,
    unit: 'min',
    help: 'The all-pairs table stores directed pairs whose modeled walk cost is at most this limit. A zero limit pauses reuse to self-pairs only.'
  },
  {
    kind: 'toggle',
    id: 'showCostTable',
    label: 'Reuse bounded cost table',
    group: 'Cost-table reuse',
    apply: 'param',
    default: true,
    help: 'When off, the contributor receives a zero-minute bound and retains only self-pairs; when on, the selected limit is applied.'
  }
];

export default defineScene<ChicagoRouteWorkbenchOptions>({
  id: 'chicago-route-workbench',
  title: 'Chicago route workbench',
  chapter: 'networks',
  order: 90,
  summary:
    'Inspect point-on-link routes, alternative paths, departure-time effects and bounded cost-table reuse on an actual OpenStreetMap pedestrian extract.',
  contributors: [
    'GPUNetworkPointRouting',
    'GPUNetworkTimeRouting',
    'GPUAlternativeRoutes',
    'GPUNetworkCostTable',
    'prepareGPUNetworkRestrictionStates'
  ],
  datasets: [{id: 'chicago-walk', role: 'directed pedestrian links and source geometry'}],
  options: CHICAGO_ROUTE_WORKBENCH_OPTIONS,
  initialView: {longitude: -87.62984, latitude: 41.88302, zoom: 16.8},
  basemap: ground('paperCity', {labelPreset: 'streets'}),
  furniture: {
    title: {title: 'A route is a model of a walk'},
    credit: '© OpenStreetMap contributors · ODbL 1.0',
    caveat: 'Street geometry is real; speed and time-of-day effects are modeled scenarios.',
    scaleBar: {units: 'metric'}
  },
  readouts: [
    {
      id: 'networkExtent',
      label: 'Extract topology',
      help: 'Retained graph nodes and directed links, with directed links excluded by the deterministic local edge cap.',
      hood: true
    },
    {
      id: 'pointRoute',
      label: 'Point-on-link route cost',
      help: 'GPUNetworkPointRouting result; edge fractions are measured along each selected directed link.'
    },
    {
      id: 'alternativeCosts',
      label: 'Alternative route costs',
      help: 'The GPU uniform-cost search returns simple directed paths in ascending modeled travel minutes.'
    },
    {
      id: 'routeOverlap',
      label: 'Shared links',
      help: 'Count of directed links shared by at least one returned alternative and route 1.'
    },
    {
      id: 'alternativeStatus',
      label: 'Alternative search status',
      help: 'A false complete flag means the expansion/frontier bound stopped the search; returned routes are incumbents, not certified top-k.'
    },
    {
      id: 'arrivalTime',
      label: 'Selected arrival clock time',
      help: 'Local Chicago clock time from the selected FIFO earliest-arrival model.'
    },
    {
      id: 'affineArrival',
      label: 'Affine model arrival',
      help: 'Arrival offset from 08:00 under a hypothetical affine duration function.'
    },
    {
      id: 'piecewiseArrival',
      label: 'Piecewise model arrival',
      help: 'Arrival offset from 08:00 from interpolation of a hypothetical arrival envelope.'
    },
    {
      id: 'sampleSupport',
      label: 'Piecewise first sample and waiting',
      help: 'The first sample is a modeled input. The piecewise GPU arrival includes any initial wait needed before its envelope begins.'
    },
    {
      id: 'costReuse',
      label: 'Reusable cost table',
      help: 'GPU cost table overflow and convergence statuses; incomplete tables cannot answer a missing-pair query conclusively.'
    },
    {
      id: 'restrictionStatus',
      label: 'Expanded-state route status',
      help: 'Validity and convergence of the GPU route search after adding a finite-state automaton for the prohibited edge sequence.'
    },
    {
      id: 'restrictionComparison',
      label: 'Baseline vs turn-rule arrival',
      help: 'GPU arrival-minute comparison using the same departure, affine duration model and original edge costs. This comparison remains affine even while the main route model selector is set to piecewise.'
    }
  ],
  legends: (state): readonly LegendSpec[] => [
    {
      kind: 'line',
      title: 'Computed directed alternatives',
      entries: [
        {color: [220, 91, 54, 235], widthPixels: 5, label: 'Route 1 · lowest returned cost'},
        {color: [44, 121, 133, 220], widthPixels: 3.3, label: 'Route 2'},
        {color: [127, 91, 158, 220], widthPixels: 3.3, label: 'Route 3'},
        {color: [255, 255, 255, 230], widthPixels: 2, label: 'Directed links shared with route 1'},
        ...(state.turnRestriction === 'hypothetical-ban'
          ? [
              {
                color: [50, 157, 108, 245] as const,
                widthPixels: 5,
                label: 'GPU route with hypothetical sequence ban'
              },
              {
                color: [206, 59, 63, 235] as const,
                widthPixels: 6.2,
                label: 'Links in the prohibited 66→38 sequence'
              }
            ]
          : [])
      ],
      note: state.showAlternatives
        ? 'Map lines follow bundled OSM path vertices; graph links are directed topology edges.'
        : 'Alternative path overlay hidden; the turn-rule overlay can still be shown.'
    },
    {
      kind: 'categories',
      title: 'Query positions',
      entries: [
        {color: [207, 82, 49, 255], label: 'Origin point on directed link'},
        {color: [39, 115, 132, 255], label: 'Destination point on directed link'}
      ],
      note: 'Ring markers move with the fractional link positions.'
    }
  ],
  about: {
    what: 'The workbench runs point-on-edge routing, FIFO earliest arrival under two time-cost models, a bounded top-k alternative search, a bounded cost table and a hypothetical forbidden-edge-sequence scenario on one small Chicago walk graph.',
    why: 'A turn restriction applies to an ordered sequence of directed links. The automaton expands routing state so the route can avoid a sequence without deleting each link from the entire graph.',
    howToRead:
      'Colored lines are returned GPU paths over real OSM polylines. Street intersections and directed edges define topology; the line geometry only draws them. Costs use modeled minutes, and clock results are local Chicago time.'
  },
  snippet: state => `graph.add(new GPUNetworkPointRouting({
  turns, costs, queries, fractions,
  output: {costs: routeCost, status: routeStatus, predecessorEdges}
}));
graph.add(new GPUNetworkTimeRouting({
  stateVertices, transitions, costs: ${state.timeModel === 'affine-duration' ? 'affineDurations' : 'piecewiseArrivalSamples'},
  costModel: '${state.timeModel}',${state.timeModel === 'piecewise-arrival' ? '\n  samplesPerTransition: 3,' : ''}
  endpoints, departure, output: {arrivals, predecessorTransitions, summary}
}));
const restriction = prepareGPUNetworkRestrictionStates({
  nodeCount,
  edges: directedEdges, // [source, target] pairs in original edge order
  restrictions: [[66, 38]]
});
const restrictionStateVertices = view(
  resources.createBuffer('restriction-state-vertices', restriction.stateVertices),
  'uint32',
  restriction.stateCount
);
const restrictionTransitions = view(
  resources.createBuffer('restriction-transitions', restriction.transitions),
  'uint32x2',
  restriction.transitionCount
);
graph.add(new GPUNetworkTimeRouting({
  id: 'chicago-route-workbench-hypothetical-turn-restriction',
  stateVertices: restrictionStateVertices, transitions: restrictionTransitions,
  costs: affineDurationCostsMappedBy(restriction.originalEdges), endpoints, departure, output
}));
graph.add(new GPUAlternativeRoutes({topology, sources, targets, routeCount: 3, maximumExpansions: 100000, maximumFrontierSize: 1024, output}));
graph.add(new GPUNetworkCostTable({topology, costLimit, laneCount, output: {offsets, targets, costs, status}}));`,
  create: async ctx =>
    (await import('./chicago-route-workbench.compute')).createChicagoRouteWorkbench(ctx),
  story: [
    {
      id: 'real-links',
      title: 'The map shows actual walk links',
      headline: 'Street geometry and graph topology play different roles',
      textAlternative:
        'A compact real pedestrian street extract underlies a set of three modeled query endpoints.',
      body: 'This view introduces the selected Loop extract. Its OSM path vertices draw the street lines; directed source and target IDs define which transitions a routing algorithm may use. The route extract is chosen reproducibly by breadth-first expansion from a documented central Loop coordinate and stops at the contributor capacity.',
      evidence: 'The bundled extract retains **{{networkExtent}}**.',
      caveat:
        'Links omitted by the capacity bound lie outside this workbench graph; an unavailable route here is not evidence that the city has no walkable route.',
      controls: [],
      readouts: ['networkExtent'],
      optionsMode: 'fresh'
    },
    {
      id: 'edge-positions',
      title: 'A trip can begin along a link',
      headline: 'Endpoint fractions change the route cost',
      textAlternative:
        'A selected directed link pair is routed with fractional start and destination positions.',
      body: 'GPUNetworkPointRouting charges the remaining source-link fraction and the leading destination-link fraction, then searches legal directed turns. Change either fraction or choose another pair of links at the extract boundary to change the query.',
      evidence:
        'The selected edge-position route costs **{{pointRoute}}** under the bundled pedestrian travel-time assumption.',
      caveat:
        'The path geometry overlay displays alternative vertex-to-vertex paths; the edge-position route contributor supplies its own fractional query cost.',
      controls: ['routePair', 'originFraction', 'destinationFraction'],
      readouts: ['pointRoute'],
      optionsMode: 'fresh',
      camera: {
        bounds: [-87.6322, 41.8818, -87.6275, 41.8843],
        padding: 56,
        zoom: 17.2,
        transitionMs: 900
      }
    },
    {
      id: 'departure-time',
      title: 'Departure time changes arrival',
      headline: 'FIFO arrival models use different assumptions',
      textAlternative:
        'A clock-time departure drives affine and piecewise earliest-arrival summaries.',
      body: 'The affine model uses nonnegative duration slope. The piecewise model interpolates a FIFO arrival envelope with a wait-until-first-sample policy. Both outputs are computed; the model selector chooses the featured clock readout. Set departure to 08:00 and the first modeled departure to 08:15 or 09:00 to expose waiting before the envelope begins. Shifting both sample clocks preserves their travel durations.',
      evidence:
        'At this departure, the selected model arrives at **{{arrivalTime}}**. The two summaries are **{{affineArrival}}** and **{{piecewiseArrival}}**.',
      caveat:
        'These are explicitly hypothetical time-of-day scenarios. The contributor interpolates arrival envelopes; it does not consume a transit timetable or measured traffic forecast.',
      controls: ['departureMinute', 'timeModel', 'affineSlope', 'firstSampleMinute'],
      readouts: ['arrivalTime', 'affineArrival', 'piecewiseArrival', 'sampleSupport'],
      optionsMode: 'fresh',
      options: {departureMinute: 240, timeModel: 'piecewise-arrival'},
      camera: {
        bounds: [-87.6322, 41.8818, -87.6275, 41.8843],
        padding: 56,
        zoom: 17.2,
        transitionMs: 900
      }
    },
    {
      id: 'route-alternatives',
      title: 'Alternatives expose shared corridors',
      headline: 'Several paths can have similar modeled costs',
      textAlternative:
        'Three returned route candidates are drawn on street geometry, with shared links highlighted.',
      body: 'GPUAlternativeRoutes searches simple directed paths in nondecreasing cost order. The map draws each returned path and marks directed links shared by any alternative with route one. Costs and the shared-link count come from the same returned edge lists.',
      evidence: 'Returned costs are **{{alternativeCosts}}**; **{{routeOverlap}}**.',
      caveat:
        'The contributor bounds expansions and frontier size. If the complete flag is false, candidates are incumbents and are not a certified top three.',
      controls: ['showAlternatives', 'routePair'],
      readouts: ['alternativeCosts', 'routeOverlap', 'alternativeStatus'],
      optionsMode: 'fresh',
      options: {showAlternatives: true},
      camera: {
        bounds: [-87.6322, 41.8818, -87.6275, 41.8843],
        padding: 56,
        zoom: 17.2,
        transitionMs: 900
      }
    },
    {
      id: 'forbidden-turn-sequence',
      title: 'A rule can ban a turn without deleting streets',
      headline: 'Routing state remembers the last directed edge',
      textAlternative:
        'A hypothetical ban on the first two directed links of the east-to-north route causes a green detour; the two links are highlighted in red.',
      body: 'Enable the hypothetical rule for the default east-to-north query. The unrestricted shortest route begins with directed links **66 → 38**. `prepareGPUNetworkRestrictionStates` expands each physical vertex into the small set of link-history states needed to forbid only that consecutive pair. The GPU then routes across the expanded transitions while retaining the original street links and travel-minute costs.',
      evidence:
        '**{{restrictionStatus}}**. **{{restrictionComparison}}** under the selected departure and affine duration assumption.',
      caveat:
        'The bundled OSM graph contains directed ways and path geometry, but no OSM turn-restriction relations. Link IDs 66→38 are a hypothetical scenario chosen from this clipped extract; this does not claim a real intersection restriction or traffic rule.',
      controls: ['turnRestriction', 'routePair', 'departureMinute', 'affineSlope'],
      readouts: ['restrictionStatus', 'restrictionComparison'],
      optionsMode: 'fresh',
      options: {
        routePair: 'east-north',
        turnRestriction: 'hypothetical-ban',
        departureMinute: 240,
        timeModel: 'affine-duration',
        affineSlope: 0.001
      },
      camera: {
        bounds: [-87.6322, 41.8818, -87.6275, 41.8843],
        padding: 56,
        zoom: 17.2,
        transitionMs: 900
      }
    },
    {
      id: 'cost-reuse',
      title: 'A bounded table can be reused',
      headline: 'Precomputed pairs trade coverage for lookup speed',
      textAlternative:
        'A bounded all-pairs cost table reports capacity and convergence before reuse.',
      body: 'GPUNetworkCostTable computes all directed pairs within a minute limit. The workbench keeps the table in GPU buffers for reuse and reports overflow plus convergence. Turning reuse off sends a zero-minute bound, leaving self-pairs only.',
      evidence: 'Current table status: **{{costReuse}}**.',
      caveat:
        'A truncated or unconverged table cannot establish that a missing pair has no feasible path. This small extract teaches reuse rather than citywide routing.',
      controls: ['showCostTable', 'costLimitMinutes'],
      readouts: ['costReuse', 'networkExtent'],
      optionsMode: 'fresh',
      options: {showCostTable: true, costLimitMinutes: 12},
      camera: {
        bounds: [-87.6322, 41.8818, -87.6275, 41.8843],
        padding: 56,
        zoom: 17.2,
        transitionMs: 900
      }
    }
  ]
});
