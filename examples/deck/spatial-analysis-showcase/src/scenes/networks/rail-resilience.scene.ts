// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {ground} from '../../cartography/grounds';
import {RAIL_ORIENTATION, RANDSTAD_RAIL_CREDIT} from './randstad-network-cartography';
import type {RailResilienceOptions} from './rail-resilience.compute';

const NETHERLANDS_VIEW = {longitude: 5.25, latitude: 52.15, zoom: 7.05};

export default defineScene<RailResilienceOptions>({
  id: 'rail-resilience',
  title: 'When a Dutch rail link fails',
  chapter: 'networks',
  order: 10,
  summary:
    'Compare station-to-station travel costs before and after a hypothetical corridor closure, and see which stations lose access inside a chosen time budget.',
  contributors: [
    'GPUNetworkCostMatrix',
    'GPUNetworkAccessibility',
    'GPUNetworkSubgraphFilter',
    'GPUNetworkStructure',
    'addNetworkDisruptionRecipe'
  ],
  datasets: [
    {id: 'gtfs-nl-rail-graph', role: 'Dutch scheduled rail links and station locations'},
    {id: 'natural-earth', role: 'national map context'}
  ],
  initialView: NETHERLANDS_VIEW,
  basemap: ground('paperSheet', {world: 'land', labels: 'above', labelPreset: 'orientation'}),
  furniture: {
    title: {
      subtitle: 'Network stress test · scheduled median in-vehicle time',
      chips: ['Hypothetical disruption']
    },
    scaleBar: {units: 'metric', latitude: 52},
    credit: `${RANDSTAD_RAIL_CREDIT} · Made with Natural Earth`,
    caveat:
      'The bundled Friday timetable is represented by median link times. Transfers are perfect, with no waiting; closures remove only the selected direct graph links.'
  },
  annotations: RAIL_ORIENTATION,
  options: [
    {
      kind: 'select',
      id: 'origin',
      label: 'Starting hub',
      group: 'Network test',
      apply: 'param',
      default: 'Amsterdam Centraal',
      help: 'Choose the station from which station-to-station costs are compared. This is a structural reachability experiment, not a passenger demand forecast.',
      options: [
        'Amsterdam Centraal',
        'Utrecht Centraal',
        'Rotterdam Centraal',
        'Den Haag Centraal',
        'Gouda'
      ].map(value => ({value, label: value.replace(' Centraal', '')}))
    },
    {
      kind: 'select',
      id: 'trainTypes',
      label: 'Train types',
      group: 'Network test',
      apply: 'param',
      default: 'all',
      help: 'Filter the directed graph by service class. Excluded links are removed from both the route cost matrix and the induced subgraph used for structural connectivity.',
      options: [
        {value: 'all', label: 'All trains'},
        {value: 'fast', label: 'Intercity and express'},
        {value: 'stopping', label: 'Stopping trains'}
      ]
    },
    {
      kind: 'slider',
      id: 'dwellSeconds',
      label: 'Stop allowance per link',
      group: 'Network test',
      apply: 'param',
      min: 0,
      max: 180,
      step: 15,
      default: 0,
      unit: 's',
      help: 'Add an assumed stop allowance to every traversed link. It is a cost sensitivity, not a transfer wait; this model assumes zero waiting between services.'
    },
    {
      kind: 'select',
      id: 'corridor',
      label: 'Closure site',
      group: 'Network test',
      apply: 'param',
      default: 'Zwolle interchange offline',
      help: 'Choose a station interchange to remove all incident links, or one direct connection. These are structural edge removals in the bundled timetable graph.',
      options: [
        'Zwolle interchange offline',
        'Utrecht interchange offline',
        'Bodegraven–Alphen a/d Rijn',
        'Utrecht–Amsterdam',
        'Rotterdam–Delft',
        'Gouda–Utrecht'
      ].map(value => ({value, label: value}))
    },
    {
      kind: 'select',
      id: 'event',
      label: 'Disruption',
      group: 'Network test',
      apply: 'param',
      default: 'closure',
      help: 'Close the selected link completely or impose a severe delay on it. The link remains in the original CSR topology so station and edge identities stay stable.',
      disabledWhen: state => state.corridor.includes('interchange offline'),
      options: [
        {value: 'closure', label: 'Full link closure'},
        {value: 'slowdown', label: 'Severe slowdown'}
      ]
    },
    {
      kind: 'slider',
      id: 'slowdownMinutes',
      label: 'Added link delay',
      group: 'Network test',
      apply: 'param',
      min: 15,
      max: 120,
      step: 15,
      default: 60,
      unit: 'min',
      disabledWhen: state =>
        state.event !== 'slowdown' || state.corridor.includes('interchange offline'),
      help: 'Extra time on the affected link when the slowdown scenario is selected. Increase it to test when a detour becomes preferable.'
    },
    {
      kind: 'slider',
      id: 'budgetMinutes',
      label: 'Access time budget',
      group: 'Access measure',
      apply: 'param',
      min: 30,
      max: 240,
      step: 15,
      default: 90,
      unit: 'min',
      help: 'Count stations reachable within this many scheduled minutes. The cost matrix retains the full finite route costs; this threshold changes only the GPU accessibility pass.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map measure',
      group: 'Display',
      apply: 'param',
      default: 'delta',
      help: 'Show baseline travel time, disrupted travel time, the added cost, or stations that become unreachable from the selected hub.',
      options: [
        {value: 'baseline', label: 'Before · minutes'},
        {value: 'scenario', label: 'After · minutes'},
        {value: 'delta', label: 'Added cost · minutes'},
        {value: 'unreachable', label: 'Newly unreachable'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showNetwork',
      label: 'Show rail links',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draw the full scheduled graph as quiet context; the selected stressed link is highlighted.'
    },
    {
      kind: 'toggle',
      id: 'showStations',
      label: 'Show stations',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draw the stations above the cost surface so every affected location remains legible.'
    }
  ],
  story: [
    {
      id: 'baseline',
      headline: 'The time budget turns scheduled rail costs into a count of reachable stations.',
      title: 'Start with the network before disruption',
      body: 'Which stations lose useful rail access when a Dutch interchange closes? The GPU network scenario recipe runs matched shortest-path searches from the selected hub and scores reachable stations with **GPUNetworkAccessibility**. Choose the **Starting hub**, filter **Train types**, and adjust **Stop allowance per link** and **Access time budget** below.',
      evidence: 'The baseline reaches **{{baselineReach}}** stations within the current budget.',
      caveat:
        'Median in-vehicle times omit transfer waits and service frequency; connections are treated as perfect.',
      textAlternative:
        'Stations are drawn over the national rail network and colored by baseline travel time from the selected hub.',
      camera: {longitude: 5.25, latitude: 52.15, zoom: 7.05, pitch: 0, transitionMs: 1400},
      options: {display: 'baseline'},
      controls: ['origin', 'trainTypes', 'dwellSeconds', 'budgetMinutes'],
      readouts: ['baselineReach', 'stationCount']
    },
    {
      id: 'close-a-link',
      headline: 'Removing an interchange can disconnect stations far beyond that interchange.',
      title: 'Close a major interchange',
      body: 'The default test disables every scheduled edge touching Zwolle, which removes that interchange from the network. Choose **Closure site** below to test Zwolle, Utrecht or one direct connection; orange edges show the affected links.',
      evidence:
        '**{{newlyUnreachable}}** that were reachable from the hub become unreachable after the disruption.',
      caveat:
        'This edge-level test does not remove all possible trains passing through an intermediate station.',
      textAlternative:
        'The selected direct rail link is orange; stations disconnected from the hub after the closure are orange.',
      options: {corridor: 'Zwolle interchange offline', event: 'closure', display: 'unreachable'},
      controls: ['corridor'],
      readouts: ['newlyUnreachable', 'scenarioReach', 'components']
    },
    {
      id: 'detour-cost',
      headline: 'Still-connected stations can bear substantial extra travel costs.',
      title: 'A detour can hurt before it disconnects',
      body: 'Some stations remain connected but take longer to reach. Switch **Map measure** below to Added cost: darker marks small increases, and brighter values mark larger added travel time.',
      evidence:
        '**{{affectedStations}}** still-reachable stations have a higher cost; their median increase is **{{medianAdded}}**.',
      caveat:
        'This is a shortest-path comparison on scheduled median link costs, not observed disruption-day journeys.',
      textAlternative:
        'A sequential station-color map shows positive differences in shortest-path travel time.',
      options: {display: 'delta'},
      controls: ['display'],
      readouts: ['medianAdded', 'newlyUnreachable']
    },
    {
      id: 'slowdown-sensitivity',
      headline: 'A costly link can shrink useful access while the network stays connected.',
      title: 'Test a severe delay before a full closure',
      body: 'A major slowdown can redirect the fastest route before a connection disappears. Choose a **Closure site** and **Disruption** below, then set **Added link delay**; the contributor compares scenario costs with a matched baseline search.',
      evidence:
        'Within budget, the scenario retains **{{scenarioReach}}** stations against **{{baselineReach}}** at baseline.',
      caveat: 'The added delay is a stress-test assumption, not a measured engineering estimate.',
      textAlternative:
        'Station colors show travel time after adding a delay to the selected rail link.',
      options: {
        corridor: 'Utrecht–Amsterdam',
        event: 'slowdown',
        slowdownMinutes: 60,
        display: 'scenario'
      },
      controls: ['corridor', 'event', 'slowdownMinutes'],
      readouts: ['scenarioReach', 'baselineReach']
    },
    {
      id: 'compare',
      headline: 'Disruption impact depends on the starting hub and the chosen time budget.',
      title: 'Read the result as a network vulnerability map',
      body: 'The cost matrix keeps the matched before and after costs; the GPU recipe writes safe differences and newly-unreachable flags. Try another **Starting hub** and **Access time budget** below to see how the impact moves across the country.',
      evidence: 'Stations leaving the chosen time budget: **{{reachLoss}}**. **{{components}}**.',
      caveat:
        'The result describes structural access from one hub under the bundled timetable snapshot, not national population coverage.',
      textAlternative:
        'The map marks stations that become unreachable and the chart compares baseline, post-event and disconnected station counts.',
      options: {display: 'unreachable'},
      controls: ['origin', 'budgetMinutes'],
      readouts: ['reachLoss', 'medianAdded', 'components', 'solver']
    }
  ],
  legends: state => [
    state.display === 'unreachable'
      ? {
          kind: 'categories',
          title: 'Station status',
          entries: [
            {color: [224, 86, 44, 255], label: 'Newly cut off'},
            {color: [42, 112, 149, 255], label: 'Other stations'}
          ],
          note: 'Other stations can include locations already disconnected or outside the chosen time budget.'
        }
      : {
          kind: 'ramp',
          title: state.display === 'delta' ? 'Added scheduled time' : 'Scheduled time',
          ramp: state.display === 'delta' ? 'magma' : 'viridis',
          extent: state.display === 'delta' ? [0, 30] : [0, 240],
          unit: 'minutes',
          note: state.display === 'delta' ? 'Only positive differences are colored.' : undefined
        }
  ],
  readouts: [
    {id: 'stationCount', label: 'Network'},
    {id: 'baselineReach', label: 'Stations within budget before'},
    {id: 'scenarioReach', label: 'Stations within budget after'},
    {id: 'newlyUnreachable', label: 'Newly unreachable'},
    {id: 'reachLoss', label: 'Lost from time budget'},
    {id: 'medianAdded', label: 'Median added time'},
    {id: 'affectedStations', label: 'Stations with higher cost'},
    {id: 'solver', label: 'Routing status'},
    {id: 'components', label: 'Directed network components'},
    {id: 'impact', label: 'Access comparison', kind: 'chart'}
  ],
  snippet: state =>
    `addNetworkDisruptionRecipe(graph, { search: { topology, startNodes, targetNodes, ... }, edgeMask, opportunityWeights, parameters }); // ${state.corridor}: ${state.event}`,
  about: {
    what: 'A paired GPU shortest-path and cumulative-accessibility analysis of Dutch rail stations.',
    why: 'Network planners can compare a corridor closure or severe delay with the same origin, timetable costs and access-time budget.',
    howToRead:
      'Point color shows the selected cost measure. In the newly-unreachable view, orange stations were reachable before the event but have no route afterward; muted blue marks every other station, including places already unreachable before the scenario.'
  },
  create: async ctx => (await import('./rail-resilience.compute')).createRailResilience(ctx)
});
