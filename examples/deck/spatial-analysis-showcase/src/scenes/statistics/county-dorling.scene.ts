// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {CountyDorlingOptions} from './county-dorling.compute';

type CountyDorlingLegendData = {
  entries: {radiusPixels: number; label: string}[];
  maximumRadiusPixels: number;
};

export default defineScene<CountyDorlingOptions>({
  id: 'county-dorling',
  title: 'What changes when New Jersey county positions move?',
  chapter: 'statistics',
  order: 91,
  summary:
    'A GPU Dorling cartogram uses observed 1990 population counts for all 21 historical NCOVR New Jersey county units, with geographic anchors and leaders retained for context.',
  contributors: ['GPUDorlingCartogram'],
  datasets: [
    {
      id: 'ncovr-new-jersey',
      role: 'historical county-unit geometry, source FIPS identifiers and 1990 population counts'
    }
  ],
  initialView: {longitude: -74.55, latitude: 40.05, zoom: 7.1},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'New Jersey 1990 population as Dorling circles',
      subtitle: 'Displaced centers · geographic anchors retained',
      sample: '21 historical county units · observed census counts',
      chips: ['GPU relaxation', 'Not geographic positions']
    },
    scaleBar: {units: 'metric', ticks: [10_000]},
    credit: 'NCOVR; Messner et al. (2000); GeoDa archive',
    caveat:
      'Circle centers are displaced in a local planar frame. Rings and leaders mark authored polygon-vertex-mean display anchors; they are not guaranteed interior points.'
  },
  options: [
    {
      id: 'display',
      label: 'Map display',
      kind: 'select',
      display: 'segmented',
      default: 'anchors',
      apply: 'param',
      options: [
        {value: 'anchors', label: 'Anchors'},
        {value: 'population-at-origins', label: 'Population at origins'},
        {value: 'displaced', label: 'Displaced'}
      ]
    },
    {
      id: 'attraction',
      label: 'Attraction to origin',
      kind: 'select',
      display: 'segmented',
      default: '0.01',
      apply: 'compile',
      options: [
        {value: '0', label: '0'},
        {value: '0.01', label: '0.01'},
        {value: '0.05', label: '0.05'}
      ]
    },
    {
      id: 'iterationBudget',
      label: 'Iteration budget',
      kind: 'select',
      display: 'segmented',
      default: '100',
      apply: 'compile',
      options: [
        {value: '0', label: '0'},
        {value: '25', label: '25'},
        {value: '100', label: '100'},
        {value: '256', label: '256'}
      ]
    }
  ],
  story: [
    {
      id: 'geographic-origins',
      title: 'Start with authored geographic anchors',
      headline: 'Rings show the source display anchors',
      body: 'The archived NCOVR geometry supplies 21 historical county units. Each ring is the arithmetic mean of all authored polygon vertices in its rings, including closure vertices and holes. This reproducible display anchor is not a polygon centroid or guaranteed interior point. Coordinates are projected into a local azimuthal-equidistant metre frame.',
      textAlternative:
        'Twenty-one rings mark authored vertex-mean display anchors for historical New Jersey county units; population circles are hidden.',
      evidence: 'The rings identify the projected anchor locations for the source FIPS rows.',
      caveat:
        'These are historical common-support units; the archive does not establish that present county identities or boundaries are unchanged.',
      options: {display: 'anchors'},
      controls: ['display'],
      readouts: ['solverStatus', 'circleCount']
    },
    {
      id: 'population-at-origins',
      title: 'Encode population area at the anchors',
      headline: 'Circle area follows the observed count',
      body: 'The origin circles use the same equation as the GPU output: r = √(population × areaScale / π), with areaScale = π × (20,000 m)² / 1,000,000 people. Thus πr² is proportional to actual 1990 population, with radii and coordinates both in metres.',
      textAlternative:
        'Blue circles at the authored geographic anchors have areas proportional to observed 1990 population counts.',
      evidence: 'The scale key shows actual metre radii converted at the current camera scale.',
      caveat: 'Population is an extensive count, not density or a rate.',
      options: {display: 'population-at-origins'},
      controls: ['display'],
      readouts: ['circleCount']
    },
    {
      id: 'zero-budget-start',
      title: 'Inspect the solver’s actual starting overlap',
      headline: 'A zero budget leaves circles at their inputs',
      body: 'This stage runs the GPU solver with an iteration budget of zero. The circles are the returned starting positions and returned radii; the overlap readout is computed from that output, not drawn from a fabricated baseline. Rings remain the source anchors.',
      textAlternative:
        'Population circles remain at their authored anchors with zero relaxation iterations; the map reports measured starting overlap.',
      evidence: 'The GPU reports overlap, displacement, excluded rows and terminal budget status.',
      caveat:
        'A zero-iteration run measures the initial layout and makes no claim that overlap was resolved.',
      options: {display: 'displaced', iterationBudget: '0'},
      controls: ['iterationBudget', 'display'],
      readouts: ['maximumOverlap', 'maximumDisplacement', 'solverStatus', 'circleCount']
    },
    {
      id: 'bounded-relaxation',
      title: 'Relax overlaps with a finite budget',
      headline: 'The GPU moves circles and retains their origins',
      body: '`GPUDorlingCartogram` applies pairwise overlap relaxation and attraction toward each source anchor. Leaders connect the returned circle centers to those anchors. Change the finite iteration budget or attraction setting to compare actual GPU runs.',
      textAlternative:
        'GPU-returned population circles appear at displaced centers, with leaders connecting them to geographic anchor rings.',
      evidence:
        'Readouts show returned circle overlap, center displacement, terminal status and excluded rows.',
      caveat:
        'The bounded solver reports its run status; even a tolerance result is not a globally optimal cartogram.',
      options: {display: 'displaced', iterationBudget: '100', attraction: '0.01'},
      controls: ['iterationBudget', 'attraction', 'display'],
      readouts: ['maximumOverlap', 'maximumDisplacement', 'solverStatus', 'circleCount']
    },
    {
      id: 'sensitivity-and-reading',
      title: 'Compare attraction and read both locations',
      headline: 'Attraction changes how far circles drift',
      body: 'The attraction control changes the force pulling each circle toward its anchor; the iteration budget controls how many relaxation passes are allowed. Each rebuild replaces the active graph and its owned buffers together. Read a ring as an authored display anchor, a blue circle as a cartogram mark, and a leader as their link.',
      textAlternative:
        'The map shows GPU-relaxed population circles, source rings and leaders; attraction and iteration budget remain adjustable.',
      evidence:
        'Tooltips hit-test the displayed circle using GPU-returned centers, radii and exact source FIPS mapping.',
      caveat:
        'Center spacing belongs to the cartogram display frame and does not represent geographic distance.',
      options: {display: 'displaced', iterationBudget: '100', attraction: '0.01'},
      controls: ['attraction', 'iterationBudget', 'display'],
      readouts: ['maximumOverlap', 'maximumDisplacement', 'solverStatus', 'circleCount']
    }
  ],
  legends: (_state, data) => {
    const legend = data['countyDorling'] as CountyDorlingLegendData | undefined;
    if (!legend) return [];
    return [
      {
        kind: 'size',
        title: '1990 population',
        layout: 'nested',
        entries: legend.entries,
        color: [0, 114, 178, 215],
        unit: 'people',
        note: `Area ∝ count; 20 km maximum radius = ${legend.maximumRadiusPixels.toFixed(1)} px at current zoom.`
      }
    ];
  },
  readouts: [
    {id: 'maximumOverlap', label: 'Maximum circle overlap', unit: 'm'},
    {id: 'maximumDisplacement', label: 'Maximum center displacement', unit: 'm'},
    {id: 'solverStatus', label: 'GPU relaxation terminal status', hood: true},
    {id: 'circleCount', label: 'Returned circles and source ID mapping', hood: true}
  ],
  about: {
    what: 'A bounded GPU Dorling relaxation of population circles anchored to historical county geometry.',
    why: 'Circle area can show total population without implying that displaced marks occupy their source polygons.',
    howToRead:
      'Read the rings as authored polygon-vertex-mean display anchors, blue circles as population marks, and leaders as the link between anchors and displaced centers. The returned GPU status, overlap and displacement readouts describe the selected run.'
  },
  snippet: state =>
    `new GPUDorlingCartogram({positions: countyAnchorsMeters, values: population1990, sourceIds: sourceFips, areaScale: Math.PI * 20_000 ** 2 / 1_000_000, maximumIterations: ${state.iterationBudget}, overlapTolerance: 100, attraction: ${state.attraction}, output})`,
  create: async ctx => (await import('./county-dorling.compute')).createCountyDorling(ctx)
});
