// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {BIXI_ASSIGNMENT_PALETTE} from '../flows/bixi-model-data';
import {defineScene} from '../scene';
import type {BixiServiceLocationsOptions} from './bixi-service-locations.compute';

const CREDIT = joinCredits(
  'BIXI Montréal open data, trip history 2024; August station activity',
  CREDITS.openStreetMap
);
const SITE_LABELS = [
  'Site A',
  'Site B',
  'Site C',
  'Site D',
  'Site E',
  'Site F',
  'Site G',
  'Site H'
];

export default defineScene<BixiServiceLocationsOptions>({
  id: 'bixi-service-locations',
  title: 'Where could BIXI support sites go?',
  chapter: 'networks',
  order: 18,
  summary:
    'A hypothetical facility study uses observed BIXI station activity as demand and real stations as candidate sites. Compare distance objectives, radius-truncated costs, exact small p-center, and order-sensitive capacity allocation.',
  contributors: ['GPULocationAllocation', 'GPUBoundedFacilityAllocation'],
  datasets: [{id: 'bixi-flows', role: 'BIXI station locations and August 2024 ride activity'}],
  initialView: {longitude: -73.57774, latitude: 45.51244, zoom: 12.4},
  options: [
    {
      kind: 'select',
      id: 'operation',
      label: 'Location objective',
      group: 'Hypothetical sites',
      apply: 'param',
      default: 'p-median',
      display: 'segmented',
      help: 'Choose demand-weighted distance, activity covered within the radius, set covering, exact bounded p-center, or capacity-constrained p-median. The operation changes which precomputed GPU result is drawn.',
      options: [
        {value: 'p-median', label: 'p-median'},
        {value: 'maximum-coverage', label: 'Max coverage'},
        {value: 'set-covering', label: 'Set covering'},
        {value: 'p-center', label: 'p-center'},
        {value: 'capacitated-p-median', label: 'Capacity'}
      ]
    },
    {
      kind: 'slider',
      id: 'coverageRadiusKm',
      label: 'Coverage radius',
      group: 'Coverage',
      apply: 'param',
      min: 0.5,
      max: 3,
      step: 0.25,
      default: 1.5,
      unit: 'km',
      disabledWhen: state =>
        state.operation === 'p-center' || state.operation === 'capacitated-p-median',
      help: 'Maximum straight-line distance for maximum-coverage and set-covering demand. A change reruns the bounded GPU allocation graph.'
    },
    {
      kind: 'select',
      id: 'costSupport',
      label: 'Cost support',
      group: 'Coverage',
      apply: 'param',
      default: 'dense',
      display: 'segmented',
      disabledWhen: state =>
        state.operation === 'p-center' || state.operation === 'capacitated-p-median',
      help: 'Use every demand-to-candidate distance, or use a sparse CSR table that treats distances above 3 km as unreachable. The 3 km truncation is fixed while coverage radius varies below it.',
      options: [
        {value: 'dense', label: 'Dense full matrix'},
        {value: 'sparse-3km', label: 'Sparse, ≤3 km'}
      ]
    },
    {
      kind: 'select',
      id: 'demandOrder',
      label: 'Capacity assignment order',
      group: 'Capacity scenario',
      apply: 'param',
      default: 'station-id',
      display: 'segmented',
      disabledWhen: state => state.operation !== 'capacitated-p-median',
      help: 'The capacitated contributor assigns indivisible activity weights in this order. Compare source station-ID order with descending observed activity to see the effect of its greedy assignment rule.',
      options: [
        {value: 'station-id', label: 'Station ID'},
        {value: 'activity-descending', label: 'Highest activity first'}
      ]
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Station encoding',
      group: 'Display',
      apply: 'param',
      default: 'assignments',
      display: 'segmented',
      help: 'Color demand points by their assigned candidate and show only assignment spokes, or size and color stations by observed activity.',
      options: [
        {value: 'assignments', label: 'Assignments'},
        {value: 'activity', label: 'Observed activity'}
      ]
    }
  ],
  readouts: [
    {id: 'objectiveValue', label: 'Objective value'},
    {id: 'objectiveUnits', label: 'Objective units'},
    {id: 'solverStatus', label: 'Solver certificate'},
    {id: 'selectedSites', label: 'Selected candidate sites'},
    {id: 'selectedStationNames', label: 'Selected station names'},
    {id: 'uncoveredDemand', label: 'Uncovered or unassigned points'},
    {id: 'swapStatus', label: 'Swap improvement status'},
    {id: 'capacityStatus', label: 'Capacity assignment status'},
    {id: 'costSupport', label: 'Cost support'},
    {id: 'coverageRadius', label: 'Coverage radius'}
  ],
  legends: state =>
    state.display === 'activity'
      ? [
          {
            kind: 'ramp',
            title: 'August station activity',
            ramp: 'cividis',
            extent: [0, 50000],
            unit: 'departures + arrivals'
          }
        ]
      : [
          {
            kind: 'categories',
            title: 'Demand assigned to candidate site',
            entries: [
              ...SITE_LABELS.map((label, index) => ({
                label,
                color: BIXI_ASSIGNMENT_PALETTE[index]
              })),
              {label: 'Unassigned or uncovered', color: [183, 55, 48, 255]}
            ],
            note: 'A spoke is drawn only to the facility chosen for that demand station.'
          }
        ],
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Hypothetical support-site experiment'},
    scaleBar: {units: 'metric'},
    credit: CREDIT,
    caveat: 'Straight-line kilometres and ride activity are proxies, not routes or staffing needs.'
  },
  snippet: state =>
    state.operation === 'p-center' || state.operation === 'capacitated-p-median'
      ? `graph.add(new GPUBoundedFacilityAllocation({
  operation: '${state.operation}', facilityCount: 8, selectCount: 3,
  costs: denseDemandMajorKilometres,
  demand: observedStationActivity, capacities: hypotheticalCapacities,
  output: {selectedFacilities, assignments, objective, status}
}));`
      : `graph.add(new GPULocationAllocation({
  operation: '${state.operation}', facilityCount: 8,
  maximumFacilityCount: ${state.operation === 'set-covering' ? 8 : 3},
  demandWeights: observedStationActivity,
  ${state.costSupport === 'dense' ? 'costs: denseDemandMajorKilometres' : 'sparseCosts: sparseCsrCostsTruncatedAt3Km'},
  ${state.operation !== 'p-median' ? 'coverageDistance: coverageRadiusKm,' : ''}
  maximumSwapIterations: ${state.operation === 'set-covering' ? 0 : 16},
  output: {selectedFacilities, assignments, assignedCosts, objective, status}
}));`,
  about: {
    what: 'Previously: BIXI monthly gravity models (bixi-demand-model). This story uses `GPULocationAllocation` for p-median, maximum-coverage and set-covering, then `GPUBoundedFacilityAllocation` for exact small p-center and capacity-constrained p-median.',
    why: 'The candidate locations are eight existing BIXI stations chosen from the same high-activity sample. The experiment asks how a hypothetical support network changes when its objective or cost support changes; it does not recommend operational sites.',
    howToRead:
      'Filled dots are the sixteen demand stations, colored by the assigned candidate. A colored spoke ends at that candidate. Dark rings are selected sites; red demand points are unassigned or outside coverage. The exact p-center result is over the supplied eight candidates and three-site subsets. Capacity assignment is greedy, indivisible and order-sensitive; failure to find a feasible assignment is not an infeasibility proof. Costs are straight-line kilometres, and demand weights are August departures plus arrivals.'
  },
  create: async context =>
    (await import('./bixi-service-locations.compute')).createBixiServiceLocations(context),
  story: [
    {
      id: 'activity-demand',
      title: 'Observed ride activity becomes a hypothetical demand weight',
      headline: 'Sixteen observed stations become demand points',
      textAlternative:
        'A map of 16 high-activity BIXI stations, colored by their August departures plus arrivals, with eight candidate stations marked as possible sites.',
      body: 'The same sixteen high-activity stations used in the gravity model are demand points. Each weight is departures plus arrivals in August 2024. Candidate sites are eight other sampled BIXI stations selected with a deterministic spread rule.',
      evidence: 'The map shows observed station activity before choosing a service objective.',
      caveat:
        'Ride activity is a comparison weight, not crew workload, maintenance demand or staffing data.',
      camera: {
        bounds: [-73.607, 45.486, -73.548, 45.539],
        padding: 56,
        zoom: 13.2,
        transitionMs: 1200
      },
      options: {display: 'activity', operation: 'p-median'},
      controls: ['display'],
      readouts: ['objectiveValue', 'objectiveUnits']
    },
    {
      id: 'coverage-tradeoffs',
      title: 'Coverage trades distance for reachable activity',
      headline: 'Coverage objectives choose different reachable activity',
      textAlternative:
        'A map marks selected candidate sites, colors each sampled station by its assigned site, and draws assignment spokes.',
      body: '**Location objective** below switches between p-median and maximum coverage. **Coverage radius** determines which stations count as reachable. **Cost support** compares the full distance table with costs truncated beyond 3 km.',
      evidence: 'The map highlights chosen candidate sites and the stations assigned to them.',
      caveat:
        'Both candidate sites and straight-line distance are hypothetical proxies for a support network.',
      options: {display: 'assignments', operation: 'maximum-coverage'},
      controls: ['operation', 'coverageRadiusKm', 'costSupport'],
      readouts: ['objectiveValue', 'selectedSites', 'uncoveredDemand', 'costSupport']
    },
    {
      id: 'set-cover',
      title: 'Set covering adds sites until all demand is within range',
      headline: 'The CPU reference needs six sites for full coverage',
      textAlternative:
        'A map shows set-covering candidate sites, colored station assignments and any uncovered demand points at the chosen radius.',
      body: 'The set-covering variant adds candidates until every sampled station lies within the chosen radius, subject to the candidate limit. Widen or tighten **Coverage radius** below to change the coverage requirement.',
      evidence:
        '**{{selectedSites}}** sites are selected; **{{uncoveredDemand}}** demand points remain uncovered.',
      caveat:
        'The source locations are candidate stations, not surveyed support facilities or eligible parcels.',
      options: {operation: 'set-covering'},
      controls: ['coverageRadiusKm', 'costSupport'],
      readouts: ['solverStatus', 'selectedSites', 'uncoveredDemand', 'coverageRadius']
    },
    {
      id: 'bounded-distance',
      title: 'P-center protects the worst straight-line distance',
      headline: 'P-center minimizes the worst station distance',
      textAlternative:
        'A map connects each sampled station to its closest selected site and reports the maximum straight-line distance.',
      body: 'The bounded p-center solver checks every three-site subset of the eight candidates, then assigns each demand station to its closest selected site. Compare it with **Location objective** set to p-median to see how the distance objective changes the map.',
      evidence: 'The p-center objective is **{{objectiveValue}} {{objectiveUnits}}**.',
      caveat: 'The exactness certificate applies only to this reduced candidate and demand set.',
      options: {operation: 'p-center'},
      controls: ['operation'],
      readouts: ['objectiveValue', 'objectiveUnits', 'solverStatus', 'selectedSites']
    },
    {
      id: 'capacity-order',
      title: 'Capacity makes assignment order visible',
      headline: 'Greedy capacity assignments change with order',
      textAlternative:
        'A map shows capacity-limited assignment spokes to selected sites, with any unassigned stations highlighted red.',
      body: 'This invented scenario gives each candidate capacity equal to 40% of total sample activity. **Capacity assignment order** changes the sequence of indivisible demand rows. Compare station-ID order with highest-activity-first.',
      evidence:
        'Solver status: **{{solverStatus}}**; unassigned demand points: **{{uncoveredDemand}}**.',
      caveat:
        'Capacitated assignments are greedy and order-sensitive. A failed assignment is not a proof that no feasible solution exists.',
      options: {operation: 'capacitated-p-median', demandOrder: 'station-id'},
      controls: ['demandOrder'],
      readouts: ['objectiveValue', 'capacityStatus', 'solverStatus', 'uncoveredDemand']
    }
  ]
});
