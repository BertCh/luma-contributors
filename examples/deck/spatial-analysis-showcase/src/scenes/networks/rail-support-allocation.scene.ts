// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {RailSupportAllocationOptions} from './rail-support-allocation.compute';

export default defineScene<RailSupportAllocationOptions>({
  id: 'rail-support-allocation',
  title: 'How many rail support desks are enough?',
  chapter: 'networks',
  order: 12,
  summary:
    'Compare small GPU facility-location objectives and rational-access allocations on eight explicitly sampled Dutch stations.',
  contributors: [
    'GPUNetworkCostMatrix',
    'GPUNetworkFacilityOptimization',
    'addFacilityScenarioRecipe',
    'GPUNetworkRationalAccess'
  ],
  datasets: [{id: 'gtfs-nl-rail-graph', role: 'Dutch scheduled rail links and station locations'}],
  initialView: {longitude: 5.25, latitude: 52.15, zoom: 7.05},
  basemap: ground('paperSheet', {world: 'land', labels: 'above', labelPreset: 'orientation'}),
  furniture: {
    title: {
      subtitle: 'Eight unit-demand stations · four candidate desks',
      chips: ['Hypothetical optimization']
    },
    scaleBar: {units: 'metric', latitude: 52},
    credit: 'Bundled GTFS-NL rail graph · median scheduled in-vehicle link times',
    caveat:
      'This compact sample is a method demonstration. One unit per selected station is not passenger demand; desk supply is a tunable scenario scale, not measured staffing capacity.'
  },
  options: [
    {
      kind: 'select',
      id: 'objective',
      label: 'Facility objective',
      group: 'Placement',
      apply: 'param',
      default: 'pmedian',
      help: 'Compare exact two-desk capacitated p-median, two-desk backup coverage, and p-dispersion over symmetric worst-direction travel times.',
      options: [
        {value: 'pmedian', label: 'Capacitated p-median'},
        {value: 'backup', label: 'Backup coverage'},
        {value: 'dispersion', label: 'P-dispersion'}
      ]
    },
    {
      kind: 'slider',
      id: 'thresholdMinutes',
      label: 'Backup threshold',
      group: 'Placement',
      apply: 'param',
      min: 30,
      max: 120,
      step: 15,
      default: 60,
      unit: 'min',
      help: 'Used by backup coverage only; maximize unit-demand stations with at least two selected desks inside the threshold.'
    },
    {
      kind: 'slider',
      id: 'scenarioLimitMinutes',
      label: 'Scenario access limit',
      group: 'Facility scenario',
      apply: 'param',
      min: 30,
      max: 180,
      step: 15,
      default: 120,
      unit: 'min',
      help: 'Reports how much of the selected sample lies within the selected desk-pair travel-time limit.'
    },
    {
      kind: 'select',
      id: 'rationalMode',
      label: 'Demand allocation',
      group: 'Rational access',
      apply: 'param',
      default: 'integral',
      options: [
        {value: 'integral', label: 'Integral station units'},
        {value: 'fractional', label: 'Fractional units'}
      ],
      help: 'The GPU allocation shifts illustrative demand among all four candidates; Sample demand units scales the amount assigned to each of the eight selected stations.'
    },
    {
      kind: 'select',
      id: 'stepMode',
      label: 'Shift limit units',
      group: 'Rational access',
      apply: 'param',
      default: 'fraction',
      options: [
        {value: 'fraction', label: 'Fraction of station demand'},
        {value: 'absolute', label: 'Absolute demand units'}
      ],
      help: 'Choose whether the decaying per-cycle shift cap is a fraction of each station’s demand or an absolute amount.'
    },
    {
      kind: 'slider',
      id: 'demandMultiplier',
      label: 'Sample demand units',
      group: 'Rational access',
      apply: 'param',
      min: 1,
      max: 10,
      step: 1,
      default: 4,
      help: 'Scales the illustrative one-unit-per-station sample so fractional and absolute shift limits have different meanings.'
    },
    {
      kind: 'slider',
      id: 'travelScaleMinutes',
      label: 'Travel scale tau',
      group: 'Rational access',
      apply: 'param',
      min: 15,
      max: 120,
      step: 15,
      default: 60,
      unit: 'min',
      help: 'Travel-cost normalization used by rational access; larger tau makes differences between route times less dominant.'
    },
    {
      kind: 'slider',
      id: 'supplyScale',
      label: 'Desk supply scale rho',
      group: 'Rational access',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.25,
      default: 1,
      help: 'Scales equal baseline supply at all four candidates. This is a normalization parameter, not a count of staff or service slots.'
    },
    {
      kind: 'select',
      id: 'allocationDisplay',
      label: 'Station map',
      group: 'Map',
      apply: 'param',
      default: 'placement',
      options: [
        {value: 'placement', label: 'Placement assignment'},
        {value: 'scenario', label: 'Nearest selected desk'},
        {value: 'rational', label: 'Rational allocation'}
      ],
      help: 'Placement colors show the optimizer assignment; nearest-desk mode evaluates every selected pair; rational colors show the largest GPU allocated share.'
    }
  ],
  story: [
    {
      id: 'sampled-costs',
      headline: 'Eight named stations make every allocation decision inspectable.',
      title: 'Start with a small, named station sample',
      body: 'Eight stations stand in for possible support needs: Hilversum, Leiden Centraal, Gouda, Amersfoort Centraal, Apeldoorn, Alkmaar, Delft and Deventer. The full rail graph computes their costs to four candidate desks.',
      evidence: '**{{sampleCount}}** · **{{candidateCount}}**.',
      caveat:
        'The sample is deliberately small for exhaustive GPU optimization and has equal unit weights; it does not represent Dutch population or ridership.',
      textAlternative:
        'The map marks eight sampled demand stations and four candidate support desks.',
      controls: [],
      readouts: ['sampleCount', 'candidateCount', 'travelModel']
    },
    {
      id: 'placement-objectives',
      headline:
        'Minimizing travel, preserving backup access and separating desks choose different pairs.',
      title: 'Different objectives choose different pairs',
      body: 'Compare two-desk **Capacitated p-median**, **Backup coverage**, and **P-dispersion**. Median placement minimizes unit-weight station travel subject to four-unit desk capacities. Backup coverage maximizes stations with two desks inside the chosen threshold. Dispersion maximizes the minimum symmetric worst-direction time between selected desks.',
      evidence: '**{{objectiveResult}}**.',
      caveat:
        'P-dispersion uses max(forward, reverse) travel time for each desk pair, making its input matrix exactly symmetric. Exhaustive search status is reported.',
      textAlternative: 'Selected desk markers update with the chosen two-facility objective.',
      controls: ['objective', 'thresholdMinutes'],
      readouts: ['objectiveResult', 'optimizerStatus']
    },
    {
      id: 'facility-scenario',
      headline: 'A selected desk pair can be evaluated against the same station costs.',
      title: 'Evaluate the chosen two-desk scenario',
      body: 'The facility scenario recipe reuses the GPU station-to-desk cost table and selected objective’s desk pair to report nearest desk and demand accessibility for the eight-unit sample.',
      evidence: '**{{scenarioSummary}}**.',
      caveat:
        'The recipe evaluates only the selected sampled stations and one unit each; its access score is not a national coverage estimate.',
      textAlternative:
        'Sample station points are colored by selected nearest desk and readouts report sample coverage.',
      options: {objective: 'pmedian', allocationDisplay: 'scenario'},
      controls: ['objective', 'scenarioLimitMinutes', 'allocationDisplay'],
      readouts: ['scenarioSummary', 'optimizerStatus']
    },
    {
      id: 'rational-allocation',
      headline: 'Congestion-sensitive allocation redistributes the eight station demand units.',
      title: 'Rational access redistributes the station units',
      body: 'The rational-access solver lets each station shift demand among all four desks according to travel cost and current supply. Compare integral with fractional allocation and change whether the shrinking shift cap is fractional or absolute.',
      evidence: '**{{rationalSummary}}**.',
      caveat:
        'Integral demand can stall between feasible assignments; solver status distinguishes a completed equilibrium from a rounded or bounded run.',
      textAlternative:
        'Readouts show the GPU demand allocation status, score range and unserved demand across the candidate desks.',
      options: {allocationDisplay: 'rational'},
      controls: [
        'rationalMode',
        'stepMode',
        'demandMultiplier',
        'travelScaleMinutes',
        'allocationDisplay'
      ],
      readouts: ['rationalSummary', 'rationalStatus']
    }
  ],
  legends: state =>
    state.allocationDisplay === 'rational' || state.objective !== 'dispersion'
      ? [
          {
            kind: 'categories',
            title:
              state.allocationDisplay === 'rational'
                ? 'Largest rational allocation share'
                : state.objective === 'backup'
                  ? 'Nearest selected desk'
                  : 'Placement assignment',
            entries: [
              {color: [36, 132, 173, 255], label: 'Amsterdam'},
              {color: [238, 132, 43, 255], label: 'Utrecht'},
              {color: [135, 91, 177, 255], label: 'Rotterdam'},
              {color: [45, 153, 112, 255], label: 'Zwolle'},
              {color: [150, 154, 158, 180], label: 'No selected desk'}
            ],
            note:
              state.allocationDisplay === 'rational'
                ? 'Each sample station is colored by the candidate receiving its largest GPU allocation share.'
                : 'Sample stations have equal unit demand.'
          }
        ]
      : [
          {
            kind: 'categories',
            title: 'Station assignment',
            entries: [
              {color: [36, 132, 173, 255], label: 'Amsterdam'},
              {color: [238, 132, 43, 255], label: 'Utrecht'},
              {color: [135, 91, 177, 255], label: 'Rotterdam'},
              {color: [45, 153, 112, 255], label: 'Zwolle'},
              {color: [150, 154, 158, 180], label: 'Unassigned'}
            ],
            note: 'P-dispersion assignment is not a demand allocation.'
          }
        ],
  readouts: [
    {id: 'sampleCount', label: 'Sampled stations'},
    {id: 'candidateCount', label: 'Desk candidates'},
    {id: 'travelModel', label: 'Travel-cost model'},
    {id: 'objectiveResult', label: 'Placement objective'},
    {id: 'optimizerStatus', label: 'Exhaustive search'},
    {id: 'scenarioSummary', label: 'Selected facility scenario'},
    {id: 'rationalSummary', label: 'Rational allocation'},
    {id: 'rationalLoads', label: 'Demand by desk'},
    {id: 'rationalStatus', label: 'Allocation solver'}
  ],
  snippet: state =>
    `new GPUNetworkFacilityOptimization({operation: '${state.objective === 'pmedian' ? 'capacitated-p-median' : state.objective === 'backup' ? 'backup-coverage' : 'p-dispersion'}', facilityCount: 4, selectCount: 2}); // eight sampled stations`,
  about: {
    what: 'A small GPU facility-location and rational-access comparison on named Dutch rail stations.',
    why: 'Median travel, backup coverage, separation and congestion-aware allocation answer different placement questions.',
    howToRead:
      'This exact-optimization sample is not a population model. Its travel times are scheduled medians with perfect transfers; demand multiplier, travel scale and desk supply are explicit assumptions.'
  },
  create: async ctx =>
    (await import('./rail-support-allocation.compute')).createRailSupportAllocation(ctx)
});
