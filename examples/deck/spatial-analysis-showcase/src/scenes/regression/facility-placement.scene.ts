// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import {FACILITY_PLACEMENT_PALETTE} from './facility-placement-palette';
import type {FacilityPlacementOptions} from './facility-placement.compute';

/** Compares hypothetical library network locations under average-distance and worst-case objectives. */
export default defineScene<FacilityPlacementOptions>({
  id: 'facility-placement',
  title: 'Where could a three-site Chicago library network go?',
  chapter: 'regression',
  order: 5,
  summary:
    'GPU p-center and capacitated p-median compare hypothetical three-site library networks across eight tract-centroid candidates. Existing branches provide geographic context but are not credited toward demand service.',
  contributors: ['GPUBoundedFacilityAllocation'],
  datasets: [
    {id: 'chicago-tracts', role: 'population demand and candidate neighborhoods'},
    {id: 'chicago-facilities', role: 'existing public library branches'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.8},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'A hypothetical three-site library network',
      subtitle: 'Chicago tract population and straight-line access'
    },
    scaleBar: {units: 'metric'},
    credit: 'US Census Bureau ACS; City of Chicago',
    caveat:
      'Existing library branches are context only and do not serve demand in this scenario. Distances are straight-line local projected meters, not road distance or travel time; capacities are illustrative.'
  },
  options: [
    {
      kind: 'select',
      id: 'map',
      label: 'Show on the map',
      group: 'Map',
      apply: 'param',
      default: 'demand',
      help: 'Start with tract population demand, then show each objective’s assignments. This changes the mapped layer, not the compiled GPU calculations.',
      options: [
        {value: 'demand', label: 'Population demand'},
        {value: 'assignments', label: 'Site assignments'}
      ]
    },
    {
      kind: 'select',
      id: 'objective',
      label: 'Optimize for',
      group: 'Planning objective',
      apply: 'param',
      default: 'average',
      help: 'Both GPU solvers choose three of the same eight candidate tract centroids for a hypothetical network. Existing branches are shown for context but are not included in the allocation. Average minimizes demand-weighted distance with illustrative site capacities; worst case minimizes the farthest tract distance without capacity limits.',
      options: [
        {value: 'average', label: 'Average access (capacity-limited)'},
        {value: 'worst', label: 'Worst-served tract (p-center)'}
      ]
    },
    {
      kind: 'slider',
      id: 'capacity',
      label: 'Illustrative capacity per site',
      group: 'Capacity sensitivity',
      apply: 'param',
      min: 100000,
      max: 1200000,
      step: 100000,
      default: 600000,
      unit: 'people',
      disabledWhen: state => state.objective !== 'average',
      help: 'A tract is assigned whole to its nearest selected site with enough remaining capacity. The fixed demand order is largest-population tracts first; change this assumption when interpreting unassigned demand.'
    },
    {
      kind: 'toggle',
      id: 'showExisting',
      label: 'Existing library branches',
      group: 'Map',
      apply: 'param',
      default: true,
      help: 'Shows Chicago Public Library branches for geographic context only. They do not count as service locations in either hypothetical network solution.'
    }
  ],
  readouts: [
    {id: 'candidates', label: 'Candidate set'},
    {id: 'demand', label: 'Population demand'},
    {id: 'selected', label: 'Selected candidate sites'},
    {
      id: 'averageDistance',
      label: 'Capacity-limited mean distance',
      help: 'Population-weighted straight-line distance among assigned tracts. Unassigned demand is excluded from this mean and shown separately.'
    },
    {id: 'unassigned', label: 'Population unassigned'},
    {
      id: 'worstDistance',
      label: 'Worst-case maximum distance',
      help: 'Exact p-center objective over the bounded candidate subsets; no capacity constraint.'
    },
    {id: 'comparison', label: 'Mean-distance comparison', kind: 'chart'}
  ],
  legends: state => {
    const siteSymbols: LegendSpec = {
      kind: 'categories',
      title: 'Candidate and existing locations',
      entries: [
        {color: [91, 105, 122, 255], label: 'Candidate tract centroid', shape: 'ring'},
        ...(state.map === 'assignments'
          ? [
              {
                color: [226, 153, 54, 255] as const,
                label: 'Selected site',
                shape: 'dot' as const
              }
            ]
          : []),
        ...(state.showExisting
          ? [
              {
                color: [76, 129, 210, 255] as const,
                label: 'Existing library branch',
                shape: 'dot' as const
              }
            ]
          : [])
      ],
      ...(state.map === 'assignments'
        ? {note: 'Selected site locations are orange stars on the map.'}
        : {})
    };
    if (state.map === 'demand') {
      return [
        {
          kind: 'ramp',
          title: 'Population per census tract',
          ramp: 'cividis',
          extent: [0, 12000],
          unit: 'residents per tract',
          labels: ['fewer residents', 'more residents']
        },
        siteSymbols
      ];
    }
    const entries: LegendSpec[] = [
      {
        kind: 'categories',
        title: 'Assignment to selected site',
        entries: [
          ...FACILITY_PLACEMENT_PALETTE.slice(0, 3).map((color, index) => ({
            color,
            label: `Site ${index + 1}`
          })),
          ...(state.objective === 'average'
            ? [{color: [80, 86, 96, 105] as const, label: 'Unassigned (capacity)'}]
            : [])
        ],
        note:
          state.objective === 'average'
            ? 'Whole tract populations go to their nearest selected site with spare illustrative capacity.'
            : 'Tracts are assigned to their nearest p-center site; demand capacity is not applied.'
      }
    ];
    entries.push(siteSymbols);
    return entries;
  },
  snippet: () => `graph.add(new GPUBoundedFacilityAllocation({
  operation: 'p-center', // exact farthest-tract objective over bounded subsets
  facilityCount: 8, selectCount: 3, costs, demand, capacities, output: worstCase
}));
graph.add(new GPUBoundedFacilityAllocation({
  operation: 'capacitated-p-median', // deterministic demand-order greedy allocation per subset
  facilityCount: 8, selectCount: 3, costs, demand, capacities, output: averageCase
}));`,
  about: {
    what: '`GPUBoundedFacilityAllocation` evaluates every three-site subset of eight candidate neighborhoods for a hypothetical library network. Existing branches are displayed as context but are not counted as service locations. One run minimizes the farthest tract distance (p-center); the other minimizes total population-weighted distance under a per-site capacity assumption.',
    why: 'Average access tends to favor dense demand, while a worst-case objective protects the tract farthest from a selected site. Capacity limits add a practical constraint and expose which population cannot be assigned under the assumed site size.',
    howToRead:
      'The tract colors show assignment to the selected hypothetical sites for the current objective. Stars are the chosen candidate centroids; rings show the full bounded candidate set, and blue dots are existing library branches for geographic context only. Existing branches are not included as service locations, so the scenario assigns demand only to its selected sites. Candidates are population-tract centroids selected by deterministic farthest-point sampling. The capacitated p-median uses largest tracts first and a greedy nearest-feasible assignment; when capacity is insufficient, subset scoring first minimizes the number of unassigned tracts, then weighted distance. It is a heuristic, not a global optimality certificate. Neither distance calculation follows streets.'
  },
  create: async ctx => (await import('./facility-placement.compute')).createFacilityPlacement(ctx),
  story: [
    {
      id: 'question',
      title: 'Where could this library network go?',
      headline: 'Eight neighborhood candidates make the planning choice concrete',
      textAlternative:
        'Chicago census tracts show population demand. Rings mark candidate library locations and blue dots mark existing branches.',
      evidence:
        'The tract fills show where population demand is concentrated; rings mark the bounded candidate set and blue dots mark current branches.',
      caveat:
        'Candidate points are tract centroids chosen by a spatial sampling rule, not available property listings.',
      body: 'The map opens with **Population demand**: each tract’s residents are one demand total. The rings show the bounded candidate set, spread across Chicago by deterministic farthest-point sampling. These are scenario points, not an inventory of vacant properties. Existing branches are geographic context only; the model does not credit them toward demand. Toggle **Existing library branches** below to compare their locations with the hypothetical candidates.',
      options: {map: 'demand', objective: 'average', capacity: 600000, showExisting: true},
      camera: {longitude: -87.68, latitude: 41.84, zoom: 9.8},
      controls: ['map', 'showExisting'],
      readouts: ['candidates', 'demand']
    },
    {
      id: 'average',
      title: 'Minimize average distance under capacity',
      headline: 'The average objective concentrates service near more people',
      textAlternative:
        'A population-weighted average-distance solution assigns whole tracts to selected candidate sites subject to illustrative capacity.',
      evidence:
        'Selected stars and assignment colors show the capacitated p-median result; the readouts report its weighted mean distance and unassigned demand.',
      caveat:
        'Assignments use whole tract populations and largest-tract-first greedy order. If capacity is insufficient, the solver first minimizes the number of unassigned tracts.',
      body: '`capacitated-p-median` compares the subsets listed under **Candidate set** for this hypothetical network. Existing library branches are context only and do not serve demand in the calculation. For each subset, it assigns largest-population tracts first to the nearest selected site that still has room, then minimizes population × straight-line distance. **Show on the map** below reveals the assignment colors. **Capacity-limited mean distance** is averaged across assigned population; **Population unassigned** reports the demand that did not fit.',
      options: {objective: 'average', map: 'assignments'},
      controls: ['map', 'capacity'],
      readouts: ['averageDistance', 'unassigned']
    },
    {
      id: 'capacity',
      title: 'Capacity changes who can be assigned',
      headline: 'Smaller assumed sites leave more tract population unassigned',
      textAlternative:
        'The capacity slider changes the illustrative per-site limit and recomputes the capacitated assignment on the GPU.',
      evidence:
        'As the assumed capacity falls, the unassigned population readout shows tracts that cannot fit whole at any selected site.',
      caveat:
        'Capacity values are illustrative scenario settings, not measured library throughput. When capacity is insufficient, unassigned tract count takes priority over unassigned population in subset scoring.',
      body: 'Try a smaller **Illustrative capacity per site** below. Demand is indivisible at tract scale, so a tract only fits when its full population fits the site’s remaining capacity. This makes the result sensitive to both capacity and the largest-tract-first processing order. The slider rewrites a GPU parameter buffer and recomputes the allocations.',
      options: {capacity: 300000},
      controls: ['capacity'],
      readouts: ['averageDistance', 'unassigned']
    },
    {
      id: 'worst',
      title: 'Protect the farthest neighborhood',
      headline: 'The p-center solution minimizes the maximum tract distance',
      textAlternative:
        'Selected sites minimize the greatest straight-line distance from any tract centroid to its assigned site.',
      evidence:
        'The star locations and maximum-distance readout change when the objective changes from weighted average to the farthest tract.',
      caveat:
        'Both objectives use straight-line distances between centroids and candidate sites, not road or travel time.',
      body: 'Switch **Optimize for** to *Worst-served tract*. The exact bounded p-center run tests each subset and chooses the one with the smallest maximum centroid-to-site distance. It gives every tract equal standing in the worst-case objective, regardless of population. Compare the selected stars and **Worst-case maximum distance** with the average-distance solution.',
      options: {objective: 'worst'},
      controls: ['objective'],
      readouts: ['selected', 'worstDistance']
    },
    {
      id: 'limits',
      title: 'Compare the two hypothetical networks',
      headline: 'Average access and worst-case access can prefer different sites',
      textAlternative:
        'The proposed sites are tract-centroid candidates and distances are straight-line; roads, travel times, eligibility, and current branch service are not modeled.',
      evidence:
        'The average and worst-case objectives use one candidate set; the comparison chart measures both solutions on the population assigned by the capacity-limited one.',
      caveat:
        'The capacitated p-median is a demand-order greedy heuristic; the p-center result is exact only for this bounded candidate set and distance matrix.',
      body: 'These are **straight-line distances in locally projected meters between tract centroids**, not walking or driving distance. The candidate set is deliberately small so subset enumeration stays transparent. The chart compares population-weighted mean distance over the same tracts assigned by the capacity-limited solution, while **Worst-case maximum distance** reports the p-center objective across all tracts. Real siting would need parcel availability, roads and transit, service eligibility, existing branch capacity, and more flexible tract-level demand. Use **Optimize for** and **Show on the map** below to compare the hypothetical networks and continue exploring the trade-off.',
      options: {objective: 'average', map: 'assignments'},
      controls: ['objective', 'map'],
      readouts: ['averageDistance', 'worstDistance', 'comparison', 'unassigned']
    }
  ]
});
