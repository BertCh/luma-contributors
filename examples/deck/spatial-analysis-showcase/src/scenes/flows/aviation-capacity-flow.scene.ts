// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {AviationCapacityFlowOptions} from './aviation-capacity-flow.compute';

const WESTERN_VIEW = {longitude: -114, latitude: 38, zoom: 3.55};
const ENDPOINTS = ['SEA', 'SFO', 'PHX', 'LAS', 'LAX', 'SAN', 'PDX', 'SLC'] as const;

export default defineScene<AviationCapacityFlowOptions>({
  id: 'aviation-capacity-flow',
  create: async context =>
    (await import('./aviation-capacity-flow.compute')).createAviationCapacityFlow(context),
  title: 'How much flow can a regional flight network carry?',
  chapter: 'flows',
  order: 12,
  summary:
    'Use GPU integral network flow to compare illustrative link budgets derived from scheduled-count values, hypothetical network stress and a fixed-budget delay-score allocation.',
  contributors: ['GPUNetworkFlow', 'prepareGPUNetworkFlowDemands'],
  datasets: [
    {
      id: 'us-airline-flows',
      role: 'directed BTS scheduled-count values and mean departure delay, July 2023; counts include cancelled flights'
    }
  ],
  initialView: WESTERN_VIEW,
  basemap: ground('night', {labels: 'above'}),
  furniture: {
    title: {
      subtitle: 'Western US · July 2023 reported airport-pair operations',
      chips: ['Illustrative flight-count link budgets']
    },
    scaleBar: {units: 'metric', latitude: 38},
    credit:
      'U.S. Bureau of Transportation Statistics On-Time Reporting Carrier data · airport positions OpenFlights.org (ODbL)',
    caveat:
      'Link budgets are illustrative values derived from scheduled counts that include cancelled flights. They are not realized operations or physical capacity. The model has no passenger demand, ticketing, carrier compatibility, departure times, connection windows or schedule-feasible itineraries.'
  },
  options: [
    {
      kind: 'select',
      id: 'source',
      label: 'Source airport',
      group: 'Flow scenario',
      apply: 'param',
      default: 'SEA',
      disabledWhen: state => state.terminalMode === 'balanced4',
      help: 'Select a western airport as the source in single-pair mode. Balanced mode fixes supplies at SEA and PDX.',
      options: ENDPOINTS.map(value => ({value, label: value}))
    },
    {
      kind: 'select',
      id: 'sink',
      label: 'Sink airport',
      group: 'Flow scenario',
      apply: 'param',
      default: 'SAN',
      disabledWhen: state => state.terminalMode === 'balanced4',
      help: 'Select a distinct western airport as the sink in single-pair mode. Balanced mode fixes demands at LAX and SAN.',
      options: ENDPOINTS.map(value => ({value, label: value}))
    },
    {
      kind: 'select',
      id: 'objective',
      label: 'Flow objective',
      group: 'Flow scenario',
      apply: 'param',
      default: 'maximum',
      disabledWhen: state => state.terminalMode === 'balanced4',
      help: 'Each scheduled-count value, which includes cancelled flights, is used as an illustrative integer link budget, not realized operations or measured passenger capacity. Maximum mode finds the largest feasible integral flow. Fixed-budget mode routes the requested units with minimum aggregate nonnegative mean-departure-delay score.',
      options: [
        {value: 'maximum', label: 'Maximum throughput'},
        {value: 'fixedBudget', label: 'Serve fixed budget at minimum delay score'}
      ]
    },
    {
      kind: 'select',
      id: 'terminalMode',
      label: 'Terminal pattern',
      group: 'Flow scenario',
      apply: 'param',
      default: 'singlePair',
      help: 'Single-pair mode routes between selected airports. Balanced mode uses equal hypothetical supplies at SEA and PDX and equal demands at LAX and SAN, through the public flow-demand adapter.',
      options: [
        {value: 'singlePair', label: 'One source and one sink'},
        {value: 'balanced4', label: 'Balanced two-supply / two-demand'}
      ]
    },
    {
      kind: 'slider',
      id: 'flowBudget',
      label: 'Requested flow budget',
      group: 'Flow scenario',
      apply: 'param',
      min: 500,
      max: 4000,
      step: 500,
      default: 1000,
      unit: 'units',
      help: 'The requested integral flow in fixed-budget mode. If the residual cut proves this amount infeasible, the scene reports the maximum amount served and the remaining shortfall.'
    },
    {
      kind: 'select',
      id: 'capacityShare',
      label: 'Hypothetical link budget share',
      group: 'Flow scenario',
      apply: 'param',
      default: '100',
      help: 'Uniformly scale every illustrative link budget, based on its directed scheduled-count value, to test sensitivity. This is a synthetic network-wide stress, not a historical disruption record.',
      options: [
        {value: '100', label: '100% of reported July counts'},
        {value: '75', label: '75% scenario'},
        {value: '50', label: '50% scenario'},
        {value: '25', label: '25% scenario'}
      ]
    }
  ],
  story: [
    {
      id: 'regional-capacities',
      title: 'Aggregate links make a small directed network',
      headline: 'Scheduled counts define illustrative link budgets.',
      body: 'For each airport, the study ranks source records by incoming plus outgoing scheduled-count values, keeps the western top 24 and retains directed airport pairs between them. These illustrative link budgets come from scheduled counts, including cancelled flights; they are not realized operations or physical capacity. Choose a source/sink pair or use the balanced terminal experiment.',
      evidence: 'This induced graph has **{{networkShape}}** and **{{regionalCapacity}}**.',
      caveat:
        'Counts cover reported operating carriers in this bundled BTS snapshot and do not include all national or marketing totals. Scheduled-count values include cancelled flights. A path through the graph is a mathematical flow path, not a bookable itinerary.',
      textAlternative:
        'A regional directed airport network with source and sink markers and observed scheduled flight-pair links.',
      camera: {bounds: [-124.2, 31, -105.4, 48.5], padding: 44, zoom: 4.2, transitionMs: 1100},
      controls: ['terminalMode', 'source', 'sink'],
      readouts: ['networkShape', 'regionalCapacity']
    },
    {
      id: 'maximum-throughput',
      title: 'Residual paths reveal a throughput ceiling',
      headline: 'A residual cut certifies the modeled flow ceiling.',
      body: 'Select **Maximum throughput**. Each directed airport-pair link receives an illustrative budget equal to its July scheduled-count value. The GPU solver repeatedly augments an integral source-to-sink flow, then exposes the residual source-side cut and its crossing budget when it certifies the maximum.',
      evidence: '**{{flowValue}}**; **{{flowCut}}**. The solver reports **{{flowStatus}}**.',
      caveat:
        'This abstraction uses scheduled-count values, including cancelled flights; it is not realized operations or measured passenger or aircraft capacity. It does not represent passenger transfers, flight times, fleet availability, or synchronized schedules.',
      textAlternative:
        'Gold links show positive GPU-allocated flow; cyan airports are on the residual source side of a certified cut.',
      options: {objective: 'maximum', capacityShare: '100'},
      controls: ['objective'],
      readouts: ['flowStatus', 'flowValue', 'flowCut', 'flowLinks']
    },
    {
      id: 'capacity-stress',
      title: 'Uniform capacity stress changes the cut',
      headline: 'Uniform budget reductions change the cut.',
      body: 'Scale every retained illustrative link budget to 75%, 50% or 25% of its scheduled-count value. This hypothetical stress test shows how total source-to-sink throughput and the residual minimum cut respond when all local airport-pair budgets shrink together.',
      evidence: 'At this setting the GPU finds **{{flowValue}}**; **{{flowCut}}**.',
      caveat:
        'Uniformly scaling every link is deliberately stylized; it does not describe a specific airline, airport closure or observed month.',
      textAlternative:
        'The highlighted flow routes and residual source-side cut update after link capacities are scaled.',
      controls: ['capacityShare'],
      readouts: ['flowStatus', 'flowValue', 'flowCut', 'flowLinks']
    },
    {
      id: 'delay-weighted-budget',
      title: 'A fixed flow budget changes the objective',
      headline: 'A delay proxy changes which links carry flow.',
      body: 'Choose **Serve fixed budget at minimum delay score**. The solver routes the requested integral units over link budgets, minimizing the sum of each allocated unit times that link’s nonnegative rounded mean departure delay in minutes. Change the budget to see when the cut prevents service.',
      evidence:
        '**{{flowStatus}}**. The allocated links carry **{{flowValue}}** with score **{{delayScore}}**.',
      caveat:
        'Mean departure delay is a link-level aggregate, not a transfer-time model. The flow-unit × rounded delay-minute score is an abstract proxy objective, not expected traveler delay.',
      textAlternative:
        'A bar chart ranks the links carrying the largest part of the fixed flow budget; the map shows their illustrative budget utilization.',
      options: {objective: 'fixedBudget', flowBudget: 1000},
      controls: ['objective', 'flowBudget'],
      readouts: ['flowStatus', 'flowValue', 'delayScore', 'flowLinks']
    },
    {
      id: 'limits-and-certificate',
      title: 'Read the cut and the model boundary',
      headline: 'Balanced terminals add virtual gates to the certificate.',
      body: 'Switch to the balanced terminal pattern to route 1,000 units from SEA and PDX together to LAX and SAN, split evenly at 500 units per terminal. The preparation helper adds virtual source and sink gates; those virtual nodes never appear as airports on the map. Its cut certificate covers the augmented graph, while the flow map and link chart include only original airport-pair records.',
      evidence: '**{{flowStatus}}**; **{{flowValue}}**; **{{delayScore}}**.',
      caveat:
        'When requested balanced units cannot be routed, the residual cut certifies the augmented graph; when the request is met, the solver stops without seeking a maximum-flow cut. Link budgets derive from scheduled counts, including cancelled flights; neither budgets nor delay score describe realized capacity, passenger access or disruption impact.',
      textAlternative:
        'The map emphasizes allocated links and the cut partition, with endpoint names and link allocations in tooltips and charts.',
      options: {terminalMode: 'balanced4', objective: 'fixedBudget', flowBudget: 1000},
      controls: ['terminalMode', 'flowBudget', 'capacityShare'],
      readouts: ['flowStatus', 'flowValue', 'flowCut', 'delayScore']
    }
  ],
  legends: () => [
    {
      kind: 'categories',
      title: 'Flow and cut',
      entries: [
        {
          color: [252, 184, 66, 230] as const,
          label: 'Allocated directed link flow',
          shape: 'line' as const
        },
        {
          color: [39, 205, 185, 255] as const,
          label: 'Residual source side of cut',
          shape: 'dot' as const
        },
        {
          color: [246, 104, 64, 255] as const,
          label: 'Source / supply terminal',
          shape: 'dot' as const
        },
        {
          color: [66, 220, 237, 255] as const,
          label: 'Sink / demand terminal',
          shape: 'dot' as const
        }
      ],
      note: 'Link opacity is allocated flow divided by its illustrative scaled scheduled-count budget. Virtual gates are omitted from the map.'
    }
  ],
  readouts: [
    {id: 'networkShape', label: 'Regional graph'},
    {id: 'regionalCapacity', label: 'Retained scheduled-count sum'},
    {id: 'flowStatus', label: 'GPU flow and cut status'},
    {id: 'flowValue', label: 'Allocated monthly flow'},
    {id: 'flowCut', label: 'Residual cut'},
    {id: 'delayScore', label: 'Aggregate delay score'},
    {id: 'augmentations', label: 'Solver augmentations'},
    {id: 'flowLinks', label: 'Largest allocated airport-pair links', kind: 'chart'}
  ],
  snippet: state =>
    `new GPUNetworkFlow({ nodeCount: 26, edges: preparedEdges, capacities: linkBudgets, costs: delayMinutes, parameters: flowRequest, output: {flow, cut, summary, exactCost} }); // ${state.objective} · ${state.capacityShare}% illustrative budgets`,
  about: {
    what: 'A bounded GPU integral-flow solver over real directed July 2023 airport-pair records.',
    why: 'The residual cut identifies a finite-graph throughput ceiling under chosen illustrative link budgets, while delay-score optimization compares one fixed allocation objective.',
    howToRead:
      'Gold line opacity follows allocated utilization. Cyan nodes mark the residual source side only when a valid complete solver result is available. The virtual gates added for flow are not among the 24 mapped airports; when a balanced request is infeasible, any cut certificate covers the augmented graph. Flow units are abstract scheduled-link allocations, not passengers.'
  }
});
