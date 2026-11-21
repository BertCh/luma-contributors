// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {SamplingFramesOptions} from './sampling-frames.compute';

const GREEN = [80, 137, 112, 255] as const;
const COPPER = [183, 105, 73, 255] as const;
const INK = [50, 65, 56, 255] as const;

export default defineScene<SamplingFramesOptions>({
  id: 'sampling-frames',
  title: 'What population can this sample represent?',
  chapter: 'points',
  order: 13,
  summary:
    'Compare a finite facility inventory with continuous line and area frames, then set an IID line draw beside a random-shift lattice. Inspect inclusion weights, replay keys, source provenance and explicit failure flags.',
  contributors: [
    'GPUSpatialBalancedSample',
    'GPUContinuousSpatialSample',
    'GPURandomPointsOnLine',
    'GPUPolygonTriangulation'
  ],
  datasets: [
    {id: 'chicago-facilities', role: 'Chicago libraries and fire stations; finite inventory frame'},
    {id: 'chicago-parks', role: 'Northerly Island OSM park polygon; mapped boundary and area frame'}
  ],
  initialView: {longitude: -87.7, latitude: 41.86, zoom: 10.7},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'What population can this sample represent?',
      subtitle: 'Finite inventory and continuous spatial frames',
      sample: 'Chicago facilities · Northerly Island OSM',
      chips: ['Design study']
    },
    scaleBar: {units: 'metric'},
    credit: true,
    caveat:
      'Frame membership and analyst-chosen designs do not measure demand, service quality, access or coverage.'
  },
  options: [
    {
      kind: 'select',
      id: 'design',
      label: 'Sampling frame',
      group: 'Frame',
      apply: 'compile',
      default: 'finite',
      display: 'chips',
      help: 'Finite inventory selects distinct source records. Continuous line and area modes apply a random-shift spatial lattice. IID line uses independent uniform arc-length draws.',
      options: [
        {value: 'finite', label: 'Finite inventory'},
        {value: 'line', label: 'Continuous boundary'},
        {value: 'area', label: 'Continuous park area'},
        {value: 'iid-line', label: 'IID line baseline'}
      ]
    },
    {
      kind: 'select',
      id: 'sampleCount',
      label: 'Finite sample size',
      group: 'Finite design',
      disabledWhen: state => state.design !== 'finite',
      apply: 'compile',
      default: '10',
      help: 'Number selected from the finite frame. In stratified mode it is split evenly across libraries and fire stations.',
      options: [
        {value: '10', label: '10 sites'},
        {value: '20', label: '20 sites'}
      ]
    },
    {
      kind: 'select',
      id: 'iidCount',
      label: 'IID draw count',
      group: 'IID line baseline',
      disabledWhen: state => state.design !== 'iid-line',
      apply: 'compile',
      default: '16',
      help: 'Fixed number of independent line draws. Raising the count appends to the same seeded stream, preserving the existing prefix.',
      options: [
        {value: '8', label: '8 draws'},
        {value: '16', label: '16 draws'},
        {value: '32', label: '32 draws'}
      ]
    },
    {
      kind: 'select',
      id: 'strata',
      label: 'Allocation',
      group: 'Finite design',
      disabledWhen: state => state.design !== 'finite',
      apply: 'compile',
      default: 'category',
      help: 'Keep equal allocations in the two source categories or treat all 174 records as one stratum.',
      options: [
        {value: 'category', label: 'Equal by category'},
        {value: 'one', label: 'One pooled stratum'}
      ]
    },
    {
      kind: 'toggle',
      id: 'unequal',
      label: 'Unequal ticket masses',
      group: 'Finite design',
      disabledWhen: state => state.design !== 'finite',
      apply: 'compile',
      default: false,
      help: 'Assign integer ticket masses from an explicit source-row parity rule. These are analyst-chosen inclusion masses, not observed importance.'
    },
    {
      kind: 'toggle',
      id: 'includeLegacy',
      label: 'Include fixed legacy site',
      group: 'Continuous design',
      apply: 'compile',
      default: true,
      disabledWhen: state => state.design !== 'line' && state.design !== 'area',
      help: 'Include one fixed hypothetical site in the continuous frame. Its point is drawn with zero weight; excluding it removes its spacing-sized neighborhood.'
    },
    {
      kind: 'select',
      id: 'spacing',
      label: 'Lattice spacing',
      group: 'Continuous design',
      disabledWhen: state => state.design !== 'line' && state.design !== 'area',
      apply: 'compile',
      default: '200',
      help: 'Arc-length spacing on the mapped boundary; Euclidean spacing over the GPU-triangulated park area. The random phase moves each complete lattice.',
      options: [
        {value: '100', label: '100 m'},
        {value: '200', label: '200 m'},
        {value: '400', label: '400 m'}
      ]
    },
    {
      kind: 'select',
      id: 'waves',
      label: 'Independent waves',
      group: 'Continuous design',
      disabledWhen: state => state.design !== 'line' && state.design !== 'area',
      apply: 'compile',
      default: '1',
      help: 'Draw complete independent designs at successive replicate keys. Do not pool wave weights or interpret a selectively retained replacement as unbiased.',
      options: [
        {value: '1', label: '1 wave'},
        {value: '2', label: '2 waves'},
        {value: '3', label: '3 waves'}
      ]
    },
    {
      kind: 'select',
      id: 'capacity',
      label: 'Output capacity',
      group: 'Validity gates',
      disabledWhen: state => state.design !== 'line' && state.design !== 'area',
      apply: 'compile',
      default: '2048',
      help: 'Fixed output slots per continuous wave. If required output exceeds this capacity, the sample is withheld and the GPU capacity flag is shown.',
      options: [
        {value: '1', label: '1 · trigger flag'},
        {value: '256', label: '256 slots'},
        {value: '2048', label: '2,048 slots'}
      ]
    },
    {
      kind: 'select',
      id: 'candidateBudget',
      label: 'Candidate budget',
      group: 'Validity gates',
      disabledWhen: state => state.design !== 'line' && state.design !== 'area',
      apply: 'compile',
      default: '65536',
      help: 'Maximum lattice candidates examined per wave. On the line design a deliberately tiny budget raises the GPU candidate flag (4) at run time; incomplete output is never shown as a valid sample. The area design knows its lattice size from the polygon bounds, so a budget smaller than that lattice is refused when the design is built and the error is shown instead.',
      options: [
        {value: '1', label: '1 · flag 4 on line, refused on area'},
        {value: '4096', label: '4,096'},
        {value: '65536', label: '65,536'}
      ]
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'Seed',
      group: 'Replay',
      apply: 'param',
      min: 1,
      max: 99999,
      step: 1,
      default: 47021,
      help: 'Resident random key. Returning to the same seed and replicate reproduces the same GPU output.'
    },
    {
      kind: 'slider',
      id: 'replicate',
      label: 'Replicate',
      group: 'Replay',
      apply: 'param',
      min: 0,
      max: 99,
      step: 1,
      default: 0,
      help: 'Resident replicate index. It selects another independent stream without changing the source frame.'
    }
  ],
  readouts: [
    {id: 'frame', label: 'Eligible frame'},
    {id: 'selected', label: 'Valid sample rows', emphasis: 'tile'},
    {id: 'ids', label: 'Selected IDs / provenance', layout: 'block'},
    {id: 'weights', label: 'Inclusion weights / units', layout: 'block'},
    {id: 'estimate', label: 'Weight sum / frame length', emphasis: 'tile'},
    {id: 'status', label: 'Design status', layout: 'block'}
  ],
  story: [
    {
      id: 'inventory',
      title: 'Start with a finite inventory',
      headline: 'The frame is 174 named facility records',
      textAlternative:
        'A sample of Chicago library and fire-station records is drawn from a finite source inventory.',
      body: '**GPUSpatialBalancedSample** selects distinct records from the Chicago facilities table. Each mark is a source row in the inventory; categories and IDs identify what was eligible before selection.',
      evidence:
        'The loaded frame contains **{{selected}}** selected rows from 174 library and fire-station records.',
      caveat:
        'This inventory supports statements about its listed records only; it says nothing about facility demand, quality or coverage.',
      options: {design: 'finite', sampleCount: '10', strata: 'category', unequal: false},
      controls: ['sampleCount', 'strata'],
      readouts: ['selected', 'frame', 'status'],
      camera: {longitude: -87.7, latitude: 41.86, zoom: 10.8},
      basemap: ground('paperCity')
    },
    {
      id: 'allocation',
      title: 'Allocation changes what is represented',
      headline: 'Strata preserve both inventory categories',
      textAlternative:
        'Copper and green points show selected library and fire-station source records.',
      body: 'Choose **Allocation** below to split the finite sample evenly between libraries and fire stations or pool the frame. The IDs and returned first-order inclusion probabilities describe the chosen design.',
      evidence:
        '**{{selected}}** distinct IDs carry inverse-probability weights; their sum is a realized Horvitz–Thompson frame-total estimate.',
      caveat:
        'Unequal-probability estimation needs a design-based variance method before uncertainty can be reported; this scene does not provide one.',
      options: {design: 'finite', sampleCount: '20', strata: 'category', unequal: true},
      controls: ['strata', 'unequal'],
      readouts: ['ids', 'weights', 'estimate'],
      camera: {longitude: -87.7, latitude: 41.86, zoom: 10.8}
    },
    {
      id: 'line-frame',
      title: 'A boundary is a different population',
      headline: 'Arc length defines the eligible line',
      textAlternative:
        'A random-shift lattice places points along the Northerly Island mapped polygon boundary.',
      body: '**GPUContinuousSpatialSample** translates an arc-length lattice by a random phase. The mapped OSM polygon boundary is the eligible frame; the points do not describe an accessible path or walkway.',
      evidence:
        '**{{selected}}** valid boundary points use spacing **{{weights}}** and identify source segments, stream replicates and candidate ordinals.',
      caveat:
        'With **Include fixed legacy site** on, a hypothetical point at arc 0 has zero weight and excludes its spacing-sized neighborhood. Turn it off to inspect the changed frame. Waves are independent complete designs; do not pool their weights.',
      options: {
        design: 'line',
        spacing: '200',
        waves: '1',
        capacity: '2048',
        candidateBudget: '65536'
      },
      controls: ['spacing', 'waves', 'includeLegacy'],
      readouts: ['selected', 'weights', 'status'],
      camera: {longitude: -87.6, latitude: 41.86, zoom: 14.3},
      basemap: ground('paperCity')
    },
    {
      id: 'area-frame',
      title: 'Area support requires valid topology',
      headline: 'Valid triangles define area support',
      textAlternative:
        'Area samples fall only within GPU-triangulated Northerly Island park triangles.',
      body: 'The source Polygon is triangulated by **GPUPolygonTriangulation** before **GPUContinuousSpatialSample** places the square lattice. Triangles carry source provenance, and the sampler accepts points only in their union.',
      evidence:
        'Valid points carry inverse-intensity weight **spacing² m²** and a source triangle, replicate and candidate ordinal.',
      caveat:
        'The synthetic fixed legacy point is the centroid of the first GPU triangle and has zero weight; its spacing-sized neighborhood is excluded when enabled. This source feature has one ring and no holes, and it is not a citywide green-space frame.',
      options: {
        design: 'area',
        spacing: '200',
        waves: '1',
        capacity: '2048',
        candidateBudget: '65536'
      },
      controls: ['spacing', 'candidateBudget', 'includeLegacy'],
      readouts: ['selected', 'weights', 'status'],
      camera: {longitude: -87.6, latitude: 41.86, zoom: 14.3}
    },
    {
      id: 'iid-baseline',
      title: 'Independent draws behave differently',
      headline: 'IID points need not spread evenly',
      textAlternative:
        'Independent uniform arc-length draws are compared with the regular random-shift boundary lattice.',
      body: '**GPURandomPointsOnLine** draws a fixed number of independent uniforms over cumulative boundary length. Raise the count: existing draws remain a stable prefix. This is a separate IID baseline, not a minimum-distance design.',
      evidence:
        '**{{selected}}** IID draws report their line ID and uniform fraction along the **{{estimate}} m** source boundary.',
      caveat:
        'The draw count is fixed and the positions are independent; close neighbors are allowed. No minimum Euclidean separation is claimed.',
      options: {design: 'iid-line', iidCount: '16', seed: 47021, replicate: 0},
      controls: ['iidCount', 'seed'],
      readouts: ['selected', 'ids', 'estimate'],
      camera: {longitude: -87.6, latitude: 41.86, zoom: 14.3}
    },
    {
      id: 'validity-and-replay',
      title: 'Validity is part of the result',
      headline: 'Incomplete designs are withheld',
      textAlternative:
        'A low candidate budget triggers an explicit sampler flag and hides incomplete points.',
      body: 'Set **Output capacity** below to 1 to exercise flag 2; then set **Candidate budget** to 1 to exercise flag 4. The readout shows rows written and required so far, not a guessed full count. Restore both controls, then change **Replicate** and return to replay.',
      evidence:
        'Status reports per-wave flags: 1 invalid, 2 capacity, 4 candidate budget. Only complete valid waves contribute visible marks or weight totals.',
      caveat:
        'Replacement waves are independent complete designs. Selective replacement after failures is not asserted to retain the original design weights.',
      options: {
        design: 'line',
        spacing: '100',
        waves: '2',
        capacity: '1',
        candidateBudget: '65536'
      },
      controls: ['capacity', 'candidateBudget', 'replicate'],
      readouts: ['status', 'selected', 'weights'],
      camera: {longitude: -87.6, latitude: 41.86, zoom: 14.3}
    },
    {
      id: 'what-it-represents',
      title: 'Name the frame before the estimate',
      headline: 'Weights follow the eligible frame',
      textAlternative:
        'The final map distinguishes source inventory records, line positions and area positions.',
      body: 'The sample can represent only the population encoded by its frame and design. Use **Sampling frame** below to revisit a facility inventory, mapped boundary or park-area triangle union, then inspect the frame ID or provenance attached to each result.',
      evidence:
        'Finite weights are returned as `1/π`; continuous weights are **spacing m** on a line or **spacing² m²** over area.',
      caveat:
        'No GRTS equivalence, spacing guarantee, design-based variance estimate, accessibility claim or service-coverage inference is made here.',
      options: {design: 'finite', sampleCount: '10', strata: 'category', unequal: false},
      controls: ['design', 'sampleCount'],
      readouts: ['frame', 'ids', 'weights'],
      camera: {longitude: -87.7, latitude: 41.86, zoom: 10.8}
    }
  ],
  legends: state => {
    const items: LegendSpec[] = [];
    if (state.design === 'finite') {
      items.push({
        kind: 'categories',
        title: 'Finite source category',
        entries: [
          {label: 'Library · source row', color: COPPER},
          {label: 'Fire station · source row', color: GREEN}
        ],
        note: 'Copper and green identify source categories. Selection status is reported in the readout.'
      });
    } else {
      items.push({
        kind: 'categories',
        title: 'Point role',
        entries:
          state.design === 'iid-line'
            ? [
                {label: 'IID arc-length draw', color: GREEN},
                {label: 'Mapped frame boundary', color: INK}
              ]
            : [
                {label: 'Primary wave', color: GREEN},
                {label: 'Replacement wave', color: [151, 163, 129, 245]},
                ...(state.includeLegacy
                  ? [{label: 'Synthetic legacy site · zero weight', color: COPPER}]
                  : []),
                {label: 'Mapped frame boundary', color: INK}
              ],
        note: 'GPU provenance includes source segment/triangle, stream replicate, candidate ordinal and role.'
      });
    }
    return items;
  },
  about: {
    what: 'Three explicit sampling frames connect finite records, continuous line support and continuous area support. GPU outputs expose sample positions, IDs, inclusion probabilities or intensity weights, replay keys and status flags.',
    why: 'An estimate can only refer to the population represented by the eligible frame. A list of facilities, a mapped boundary and a polygon interior have different support and different units.',
    howToRead:
      'Finite copper/green points are source records. Continuous points report source geometry and lattice provenance. IID line points are independent arc-length draws; close pairs are expected. Weights are design metadata, not observed facility importance.'
  },
  snippet: state =>
    state.design === 'finite'
      ? `new GPUSpatialBalancedSample({ positions, frameIds, strata, allocations, inclusionTickets, bounds, seed, parameters, output })`
      : state.design === 'iid-line'
        ? `new GPURandomPointsOnLine({ positions, lineOffsets, counts, parameters, output })`
        : `new GPUContinuousSpatialSample({ frame: {type: '${state.design}'}, spacing, seed, waves, capacity, maximumCandidates, parameters, output })`,
  create: async ctx => (await import('./sampling-frames.compute')).createSamplingFrames(ctx)
});
