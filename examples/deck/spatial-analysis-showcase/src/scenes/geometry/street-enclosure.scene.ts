// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';

type StreetEnclosureOptions = {
  view: 'footprints' | 'isovist' | 'profile' | 'limits';
  observerEast: number;
  observerNorth: number;
  profileBearing: number;
};

/** Why does one Chicago street feel enclosed? A compact real-footprint experiment. */
export default defineScene<StreetEnclosureOptions>({
  id: 'street-enclosure',
  title: 'Why does this street feel enclosed?',
  chapter: 'geometry',
  order: 8,
  summary:
    'Use real Loop and Near West Side building footprints to compare a street-level isovist and a side-to-side street profile, then inspect what footprint geometry can and cannot explain.',
  contributors: ['addIsovistsRecipe', 'addStreetProfilesRecipe'],
  datasets: [
    {id: 'chicago-buildings', role: 'building footprint boundaries and available story counts'}
  ],
  initialView: {longitude: -87.635, latitude: 41.8825, zoom: 15.3},
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Street enclosure', subtitle: 'Chicago Loop · building footprints'},
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cityOfChicago),
    caveat:
      'Footprints describe plan shape; they do not establish street-level visibility or complete building heights.'
  },
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Evidence',
      group: 'Street enclosure',
      apply: 'param',
      default: 'footprints',
      display: 'segmented',
      options: [
        {value: 'footprints', label: 'Footprints'},
        {value: 'isovist', label: 'Visibility'},
        {value: 'profile', label: 'Street profile'},
        {value: 'limits', label: 'Limits'}
      ],
      help: 'Change the map overlay while the same Chicago building sample stays in view.'
    },
    {
      kind: 'slider',
      id: 'observerEast',
      label: 'Observer east offset',
      group: 'Observer',
      apply: 'param',
      min: -220,
      max: 220,
      step: 5,
      default: 0,
      unit: 'm',
      help: 'Move the observer east or west from its clear initial position. Positions inside a footprint are rejected.'
    },
    {
      kind: 'slider',
      id: 'observerNorth',
      label: 'Observer north offset',
      group: 'Observer',
      apply: 'param',
      min: -220,
      max: 220,
      step: 5,
      default: 0,
      unit: 'm',
      help: 'Move the observer north or south from its clear initial position. Positions inside a footprint are rejected.'
    },
    {
      kind: 'slider',
      id: 'profileBearing',
      label: 'Profile direction',
      group: 'Observer',
      apply: 'param',
      min: 0,
      max: 355,
      step: 5,
      default: 90,
      unit: '°',
      help: 'Rotate the two-sided profile normal. This rewrites a small input buffer and keeps the graph compiled.'
    }
  ],
  story: [
    {
      id: 'context',
      title: 'Why does this street feel enclosed?',
      headline: 'Begin with the shape of the street wall',
      textAlternative:
        'Active Chicago building footprint outlines are shown around a Loop and Near West Side sample area, with the selected observer marked separately.',
      body: 'These outlines are **{{buildingSample}} active building footprints** from the City of Chicago Data Portal, clipped to a compact Loop and Near West Side window. A continuous pair of street walls can narrow the view even when the street itself is unchanged.',
      evidence: 'The map shows measured building boundaries from the city footprint dataset.',
      caveat:
        'The extract is a bounded neighborhood sample; the map does not show every approach into the district.',
      camera: {longitude: -87.635, latitude: 41.8825, zoom: 15.3, transitionMs: 800},
      options: {view: 'footprints'},
      controls: ['view'],
      readouts: ['buildingSample']
    },
    {
      id: 'visibility',
      title: 'What can an observer see?',
      headline: 'Ray intersections trace a local isovist',
      textAlternative:
        'Teal GPU-generated rays fan out from the marked observer and stop at a building boundary or the analysis range; east and north controls move the observer through clear map space.',
      body: '`addIsovistsRecipe` casts evenly spaced rays from the observer until a building boundary or the **{{analysisRange}} range** stops each ray. The teal lines trace the visible boundary from this point. Move **Observer east offset** or **Observer north offset** below to test another clear point; the map marks the observer.',
      evidence:
        'The GPU-generated rays terminate at the closest intersecting footprint segment or at the stated range.',
      caveat:
        'This is a 2D plan view. It ignores windows, doors, vegetation, terrain and eye height.',
      options: {view: 'isovist'},
      controls: ['view', 'observerEast', 'observerNorth'],
      readouts: ['analysisInputs', 'analysisRange']
    },
    {
      id: 'width',
      title: 'How wide is the street canyon?',
      headline: 'A cross-street profile samples both sides',
      textAlternative:
        'An orange two-sided profile extends from the marked observer to the first building hit in each direction, and a GPU readout reports the combined plan width when both hits are present.',
      body: '`addStreetProfilesRecipe` measures the first building hit on each side of a sample cross-section. The GPU readout **{{profileWidth}}** is the combined plan distance when both sides are observed; the orange line ends at the two first-hit points. Turn **Profile direction** below to rotate the section. Available story counts are incomplete, so no height-to-width ratio is inferred.',
      evidence:
        'The profile graph uses the same real boundary segments as the isovist and keeps a missing side as range-censored.',
      caveat:
        'One cross-section does not summarize a whole block, and the source does not provide complete measured heights.',
      options: {view: 'profile'},
      controls: ['view', 'profileBearing', 'observerEast', 'observerNorth'],
      readouts: ['profileWidth', 'analysisInputs', 'analysisRange']
    },
    {
      id: 'limits',
      title: 'What does “enclosed” leave out?',
      headline: 'Footprints explain plan geometry, not the whole experience',
      textAlternative:
        'The map returns to building footprints and the marked observer without the analytical rays, illustrating that plan geometry omits height, facade openings, vegetation, traffic and direction of view.',
      body: 'Switch back to **Footprints** below to remove the analytical overlays. The analysis depends on one observer and its range. Real enclosure also changes with facade height, openings, trees, traffic and the direction a person is facing.',
      evidence: 'The same observer and boundary set underlie both GPU analyses.',
      caveat:
        'Building story counts are shown only as source attributes; they are not complete enough to support a citywide height model.',
      options: {view: 'limits'},
      controls: ['view'],
      readouts: ['buildingSample']
    },
    {
      id: 'explore',
      title: 'Try another street question',
      headline: 'The footprint sample is ready to explore',
      textAlternative:
        'The visibility layer can be restored and the observer moved, or the street profile can be rotated to ask a different plan-geometry question of the same footprint sample.',
      body: 'Switch between **Visibility** and **Street profile** below to see two questions asked of the same geometry. A broader study should add surveyed street centerlines, facade openings and reliable height measurements.',
      evidence: 'Both outputs are generated from the bounded Chicago building sample on the GPU.',
      caveat: 'This is an illustrative local analysis, not a pedestrian visibility survey.',
      options: {view: 'isovist'},
      controls: ['view'],
      readouts: ['analysisInputs']
    }
  ],
  legends: state => [
    {
      kind: 'categories',
      title: 'Map layers',
      entries: [
        {color: [104, 116, 124, 155], label: 'City building footprints'},
        ...(state.view === 'isovist'
          ? [{color: [22, 135, 139, 235] as const, label: 'GPU isovist rays'}]
          : []),
        ...(state.view === 'profile'
          ? [{color: [205, 113, 48, 235] as const, label: 'GPU profile sample'}]
          : [])
      ]
    }
  ],
  readouts: [
    {id: 'buildingSample', label: 'Active footprints in sample'},
    {id: 'analysisInputs', label: 'Building boundary segments used'},
    {id: 'profileWidth', label: 'Footprint-defined cross-section width'},
    {id: 'analysisRange', label: 'Maximum analysis range'}
  ],
  about: {
    what: 'A compact urban morphology example built on the City of Chicago’s building footprint dataset.',
    why: 'The analysis separates visible plan geometry from the additional height and facade information needed to describe street enclosure.',
    howToRead:
      'Building outlines are source geometry. The teal visibility rays and orange first-hit profile endpoints are generated by GPU recipes from those outlines.'
  },
  create: async ctx => (await import('./street-enclosure.compute')).createStreetEnclosure(ctx)
});
