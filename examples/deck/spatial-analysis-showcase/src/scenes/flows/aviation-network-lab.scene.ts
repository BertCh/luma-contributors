// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {AviationNetworkOptions} from './aviation-network-lab.compute';

const WESTERN_VIEW = {longitude: -114, latitude: 38, zoom: 3.6};

export default defineScene<AviationNetworkOptions>({
  id: 'aviation-network-lab',
  title: 'Why flight hubs tell different stories',
  chapter: 'flows',
  order: 8,
  summary:
    'Compare scheduled flight volume, shortest-path brokerage and flow communities in a small induced regional network of US airports.',
  contributors: [
    'GPUNetworkCentrality',
    'GPUNetworkLouvain',
    'GPUNetworkStructure',
    'GPUNetworkCoarsening'
  ],
  datasets: [
    {id: 'us-airline-flows', role: 'directed scheduled airport-pair flows reported in July 2023'}
  ],
  initialView: WESTERN_VIEW,
  basemap: ground('night', {labels: 'above'}),
  furniture: {
    title: {
      subtitle: 'July 2023 · top 24 airports in the selected region',
      chips: ['Scheduled flight legs, not passengers']
    },
    scaleBar: {units: 'metric', latitude: 38},
    credit:
      'U.S. Bureau of Transportation Statistics On-Time Reporting Carrier data · airport positions OpenFlights.org (ODbL)',
    caveat:
      'The network contains reported directional airport pairs in one region. Centrality measures structural brokerage; it does not estimate passengers, seats, disruption impact or feasible itineraries.'
  },
  options: [
    {
      kind: 'select',
      id: 'region',
      label: 'Regional subgraph',
      group: 'Network',
      apply: 'compile',
      default: 'west',
      help: 'Rank airports inside this regional box by all-source incoming plus outgoing flight legs, select the top 24, then retain only observed directed links between those airports.',
      options: [
        {value: 'west', label: 'West'},
        {value: 'northeast', label: 'Northeast'},
        {value: 'south', label: 'South'}
      ]
    },
    {
      kind: 'select',
      id: 'measure',
      label: 'Airport measure',
      group: 'Map',
      apply: 'param',
      default: 'traffic',
      help: 'Compare observed traffic scale with two shortest-path brokerage definitions, GPU communities, or directed strong components.',
      options: [
        {value: 'traffic', label: 'Scheduled flow volume'},
        {value: 'distanceBroker', label: 'Distance-based broker'},
        {value: 'hopBroker', label: 'Fewest-leg broker'},
        {value: 'communities', label: 'Flow communities'},
        {value: 'components', label: 'Strong components'}
      ]
    },
    {
      kind: 'slider',
      id: 'gamma',
      label: 'Community resolution',
      group: 'Flow communities',
      apply: 'param',
      min: 0.5,
      max: 2,
      step: 0.1,
      default: 1,
      help: 'Louvain resolution gamma changes the tradeoff between fewer larger flow communities and more smaller groups.'
    },
    {
      kind: 'select',
      id: 'communityModel',
      label: 'Community assumption',
      group: 'Flow communities',
      apply: 'param',
      default: 'directed',
      help: 'Directed keeps the observed origin-to-destination weights. Symmetric uses (Wuv + Wvu) / 2 in each direction: this removes orientation while preserving the total scheduled-count weight. It does not recover reciprocal service.',
      options: [
        {value: 'directed', label: 'Directed observed flows'},
        {value: 'symmetric', label: 'Symmetric mean flow'}
      ]
    }
  ],
  story: [
    {
      id: 'regional-network',
      headline: 'Regional boundaries determine which airports can act as brokers.',
      title: 'Start with observed directed flight links',
      body: 'The selected region keeps 24 airports with the largest sum of incoming and outgoing scheduled flight legs, then retains the observed directional links between them. Change **Regional subgraph** to compare three compact parts of the country.',
      evidence: 'The induced network contains **{{networkShape}}** and **{{scheduledFlights}}**.',
      caveat:
        'This regional slice is not a national network ranking; routes leaving the selected airports are outside the analysis. The map draws links without arrowheads; direction is preserved in the GPU graph and tooltips.',
      textAlternative:
        'A map of regional airports and directed July 2023 flight arcs, with airport size showing all scheduled legs touching each airport.',
      camera: {longitude: -114, latitude: 38, zoom: 3.6, transitionMs: 1200},
      controls: ['region'],
      readouts: ['networkShape', 'scheduledFlights']
    },
    {
      id: 'traffic-hubs',
      headline: 'A busy airport need not lie on the shortest paths between other airports.',
      title: 'Traffic volume finds the busiest airports',
      body: 'Choose **Scheduled flow volume**. Large circles mark airports with many incoming and outgoing flight legs in the source table. These counts are scheduled operations, not passenger totals.',
      evidence:
        'Hover an airport for its all-source flight volume and regional outgoing link count.',
      caveat:
        'Volume does not measure network indispensability: a busy airport can have many alternatives.',
      textAlternative: 'Circle size follows total scheduled legs touching each airport.',
      options: {measure: 'traffic'},
      controls: ['measure'],
      readouts: ['networkShape', 'scheduledFlights']
    },
    {
      id: 'brokerage-distance',
      headline: 'Distance costs identify geometric shortcuts through the network.',
      title: 'Longer routes define one kind of broker',
      body: 'Select **Distance-based broker**. GPU betweenness counts how often an airport lies inside shortest routes when each directed flight leg costs its great-circle distance in kilometres.',
      evidence: 'Top distance brokers: **{{topBrokers}}**.',
      caveat:
        'Great-circle distance is a geometric proxy, not schedule time, fares, transfers or passenger itineraries.',
      textAlternative:
        'Airport color shows directed shortest-distance betweenness; a chart ranks the top airports.',
      options: {measure: 'distanceBroker'},
      controls: ['measure'],
      readouts: ['topBrokers', 'centralityChart', 'solver']
    },
    {
      id: 'brokerage-hops',
      headline: 'Counting legs changes the ranking of regional brokers.',
      title: 'Fewest legs produce a different broker list',
      body: 'Switch to **Fewest-leg broker**. The GPU centrality contributor now assigns unit cost to every observed directional link, so routes compete by number of legs instead of distance.',
      evidence: 'Top fewest-leg brokers: **{{topBrokers}}**.',
      caveat: 'A leg-count path is not a bookable connection and ignores schedule coordination.',
      textAlternative: 'The map and ranking chart update to unit-hop betweenness.',
      options: {measure: 'hopBroker'},
      controls: ['measure'],
      readouts: ['topBrokers', 'centralityChart', 'solver']
    },
    {
      id: 'communities',
      headline: 'A flow partition and a reachability component answer different questions.',
      title: 'Communities group dense scheduled flows',
      body: 'Select **Flow communities** to color airports by Louvain groups. Compare **Directed observed flows** with **Symmetric mean flow**. The symmetric model gives each direction half the combined count, so total weight stays comparable while orientation is removed. The GPU coarsener then aggregates the chosen groups into between-group links and within-group totals.',
      evidence: '**{{communities}}**; **{{communityCoarsening}}**.',
      caveat:
        'Louvain is a bounded heuristic, not a globally optimal partition. Scheduled-count fields include canceled flights in this bundle; they are neither realized operations nor link capacity. Symmetrization is a modeling assumption, not observed reciprocal service.',
      textAlternative:
        'Airports are colored by GPU Louvain community and the readout reports completed levels and sweeps.',
      options: {measure: 'communities'},
      controls: ['communityModel', 'gamma'],
      readouts: ['communities', 'communityCoarsening', 'communityCoarseningChart', 'solver']
    }
  ],
  legends: state =>
    state.measure === 'communities' || state.measure === 'components'
      ? [
          {
            kind: 'categories',
            title:
              state.measure === 'communities'
                ? 'Louvain flow community'
                : 'Directed strong component',
            entries: [
              {color: [52, 180, 214, 255], label: 'Community or component ID'},
              {color: [165, 172, 178, 180], label: 'No published result'}
            ],
            note:
              state.measure === 'communities'
                ? `${state.communityModel === 'directed' ? 'Directed origin-to-destination' : 'Symmetric mean-flow'} Louvain group; circle area remains all-source scheduled count.`
                : 'Circle area continues to represent scheduled leg volume.'
          }
        ]
      : [
          {
            kind: 'ramp',
            title:
              state.measure === 'traffic' ? 'Airport scheduled volume' : 'Betweenness brokerage',
            ramp: 'magma',
            extent:
              state.measure === 'traffic'
                ? [
                    0,
                    state.region === 'west' ? 35041 : state.region === 'northeast' ? 45584 : 59834
                  ]
                : state.measure === 'distanceBroker'
                  ? [0, state.region === 'west' ? 73 : state.region === 'northeast' ? 65 : 70]
                  : [0, state.region === 'west' ? 34 : state.region === 'northeast' ? 14 : 35],
            unit:
              state.measure === 'traffic' ? 'all-source touching flight legs' : 'GPU broker score',
            note: 'Circle area also shows scheduled leg volume.'
          }
        ],
  readouts: [
    {id: 'networkShape', label: 'Regional graph'},
    {id: 'scheduledFlights', label: 'Scheduled flight legs'},
    {id: 'topBrokers', label: 'Highest broker airports'},
    {id: 'components', label: 'Directed reachability'},
    {id: 'communities', label: 'Louvain run'},
    {id: 'communityMembers', label: 'GPU community membership'},
    {id: 'communityCoarsening', label: 'GPU community-link aggregate'},
    {id: 'communityCoarseningChart', label: 'Within- and between-group weight', kind: 'chart'},
    {id: 'solver', label: 'GPU status'},
    {id: 'centralityChart', label: 'Broker ranking', kind: 'chart'}
  ],
  snippet: state =>
    state.measure === 'communities'
      ? `new GPUNetworkLouvain({directed: ${state.communityModel === 'directed'}, weights, parameters, output});\nnew GPUNetworkCoarsening({directed: ${state.communityModel === 'directed'}, topology, labels: output.communities, output}); // ${state.region} regional subgraph`
      : `new GPUNetworkCentrality({topology, reverseTopology, pivotNodes, output}); // ${state.measure} · ${state.region} regional subgraph`,
  about: {
    what: 'A GPU comparison of observed flight volume, directed shortest-path centrality, flow communities and strong components.',
    why: 'A high-volume airport can rank differently from a shortest-path broker; communities and reachability describe other parts of the same network.',
    howToRead:
      'Circle area always reflects scheduled legs touching an airport. Color reflects the selected measure. Lines retain observed directions through their graph rows, although the map draws them as a shared arc.'
  },
  create: async ctx => (await import('./aviation-network-lab.compute')).createAviationNetwork(ctx)
});
