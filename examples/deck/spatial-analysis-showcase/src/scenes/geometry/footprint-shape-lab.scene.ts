// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {FootprintShapeOptions} from './footprint-shape-lab.compute';

const COURTYARD = {longitude: -87.64917, latitude: 41.8837, zoom: 18.2};

/** A close survey-paper reading of how a mapped shell changes under planar operations. */
export default defineScene<FootprintShapeOptions>({
  id: 'footprint-shape-lab',
  create: async context =>
    (await import('./footprint-shape-lab.compute')).createFootprintShapeLab(context),
  title: 'How does a moving boundary change shape?',
  chapter: 'geometry',
  order: 16,
  summary:
    'Follow one real Chicago courtyard and one convex footprint through GPU wavefront, interior-circle, clearance and boundary-offset operations.',
  contributors: [
    'GPUVectorStraightSkeleton',
    'GPUConvexStraightSkeleton',
    'GPUMaximumInscribedCircle',
    'GPUMinimumClearance',
    'GPUBufferSurface',
    'GPUOffsetCurve',
    'GPUOffsetCurveComponents',
    'GPUOffsetExpansion'
  ],
  datasets: [{id: 'chicago-buildings', role: 'archived footprints 354712 and 814558'}],
  initialView: COURTYARD,
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Footprint shape lab', subtitle: 'A boundary moves; interior geometry responds'},
    scaleBar: {units: 'metric', ticks: [5, 10]},
    credit: joinCredits(CREDITS.cityOfChicago, CREDITS.openStreetMap),
    caveat:
      'Local Web Mercator-scaled meters around each source shell centroid support footprint-scale comparison; this is not cadastral precision or a code-compliant setback.'
  },
  legends: () => [
    {
      kind: 'categories',
      title: 'Geometry marks',
      entries: [
        {color: [45, 56, 53, 225], label: 'Archived source boundary', shape: 'line'},
        {color: [190, 91, 62, 240], label: 'GPU boundary or wavefront result', shape: 'line'},
        {
          color: [194, 133, 92, 190],
          label: 'Raw offset curve before component selection',
          shape: 'line'
        },
        {color: [46, 125, 151, 240], label: 'GPU circle or witness', shape: 'line'}
      ]
    }
  ],
  options: [
    {
      kind: 'select',
      id: 'footprint',
      label: 'Source footprint',
      group: 'Study',
      apply: 'compile',
      default: 'courtyard',
      help: '354712 preserves a courtyard ring. 814558 is a four-corner strictly convex shell.',
      options: [
        {value: 'courtyard', label: 'Courtyard · 354712'},
        {value: 'convex', label: 'Convex · 814558'}
      ]
    },
    {
      kind: 'select',
      id: 'question',
      label: 'Geometry question',
      group: 'Study',
      apply: 'compile',
      default: 'wavefront',
      help: 'Only the selected question compiles. Its graph is cached by inputs and settings.',
      options: [
        {value: 'wavefront', label: 'Wavefront'},
        {value: 'circle', label: 'Largest interior circle'},
        {value: 'clearance', label: 'Minimum clearance'},
        {value: 'move', label: 'Move boundary'}
      ]
    },
    {
      kind: 'slider',
      id: 'offsetDistance',
      label: 'Offset distance',
      group: 'Boundary movement',
      apply: 'compile',
      min: 0.5,
      max: 12,
      step: 0.5,
      default: 4,
      unit: 'm',
      disabledWhen: state => state.question !== 'wavefront' && state.question !== 'move',
      help: 'Skeleton distance moves its wavefront inward. Polygon buffer sign selects inward or outward; directed line offsets select left or right.'
    },
    {
      kind: 'select',
      id: 'join',
      label: 'Corner join',
      group: 'Boundary movement',
      apply: 'compile',
      default: 'round',
      disabledWhen: state => state.question !== 'move',
      options: [
        {value: 'round', label: 'Round'},
        {value: 'mitre', label: 'Mitre'},
        {value: 'bevel', label: 'Bevel'}
      ]
    },
    {
      kind: 'select',
      id: 'bufferDirection',
      label: 'Polygon buffer sign',
      group: 'Boundary movement',
      apply: 'compile',
      default: 'outward',
      disabledWhen: state => state.question !== 'move' || state.footprint !== 'convex',
      help: 'Negative polygon distance contracts; positive distance expands before the offset topology is resolved.',
      options: [
        {value: 'outward', label: 'Outward'},
        {value: 'inward', label: 'Inward'}
      ]
    },
    {
      kind: 'select',
      id: 'curveSide',
      label: 'Offset side',
      group: 'Boundary movement',
      apply: 'compile',
      default: 'left',
      disabledWhen: state => state.question !== 'move' || state.footprint !== 'courtyard',
      help: 'Left means positive distance and right means negative distance relative to the stored direction of the selected exterior boundary chain.',
      options: [
        {value: 'left', label: 'Left of source direction'},
        {value: 'right', label: 'Right of source direction'}
      ]
    },
    {
      kind: 'slider',
      id: 'quadSegments',
      label: 'Quarter-circle segments',
      group: 'Boundary movement',
      apply: 'compile',
      min: 1,
      max: 16,
      step: 1,
      default: 8,
      disabledWhen: state => state.question !== 'move' || state.join !== 'round',
      help: 'Controls the chord resolution of round offset corners.'
    },
    {
      kind: 'slider',
      id: 'mitreLimit',
      label: 'Mitre limit',
      group: 'Boundary movement',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 0.5,
      default: 5,
      disabledWhen: state => state.question !== 'move' || state.join !== 'mitre',
      help: 'Clips a mitre when its length exceeds this multiple of the offset distance.'
    },
    {
      kind: 'select',
      id: 'budget',
      label: 'Work budget',
      group: 'Search and events',
      apply: 'compile',
      default: 'resolved',
      disabledWhen: state => state.question !== 'wavefront' && state.question !== 'circle',
      options: [
        {value: 'quick', label: 'Quick'},
        {value: 'resolved', label: 'Resolved'}
      ],
      help: 'Quick uses a deliberately small event or cell budget; status and bounded partial results stay visible.'
    },
    {
      kind: 'slider',
      id: 'tolerance',
      label: 'Circle search tolerance',
      group: 'Search and events',
      apply: 'compile',
      min: 0.05,
      max: 1,
      step: 0.05,
      default: 0.2,
      unit: 'm',
      disabledWhen: state => state.question !== 'circle',
      help: 'Branch-and-bound stops when the remaining geometric search error meets this threshold, or reports budget exhaustion.'
    }
  ],
  story: [
    {
      id: 'trace',
      title: 'The source is already a topology',
      headline: 'One mapped shell encloses a real void',
      textAlternative:
        'A dark two-ring building footprint is centered on Chicago; the inner ring encloses a courtyard.',
      body: 'Building **354712** stores one polygon with an exterior ring and a courtyard ring. The coordinates use a local meter frame centered at the source location.',
      evidence: '{{shapeSource}}',
      caveat:
        'Mapped footprints describe a source boundary; they do not describe roof structure or access.',
      options: {footprint: 'courtyard', question: 'wavefront'},
      optionsMode: 'fresh',
      controls: ['footprint'],
      readouts: ['shapeSource'],
      camera: {longitude: -87.64917, latitude: 41.8837, zoom: 18.2},
      basemap: ground('paperCity')
    },
    {
      id: 'arcs',
      title: 'Wavefront traces leave arcs',
      headline: 'Event traces record changes in the courtyard wavefront',
      textAlternative:
        'Terracotta skeleton arcs connect moving footprint vertices; the remaining offset contour is shown inside the courtyard shell.',
      body: 'The **wavefront** moves each edge inward at unit speed. Its event traces retain source-edge IDs; linked active vertices show the requested offset when the budget reaches it.',
      evidence: '{{shapeEvidence}}',
      caveat:
        'The vector skeleton is an event trace, not a medial axis or a maximum-inscribed-circle solution.',
      options: {question: 'wavefront', offsetDistance: 4, budget: 'resolved'},
      optionsMode: 'fresh',
      controls: ['offsetDistance', 'budget'],
      readouts: ['shapeStatus', 'shapeEvidence'],
      camera: {longitude: -87.64917, latitude: 41.8837, zoom: 18.2},
      basemap: ground('paperCity')
    },
    {
      id: 'convex',
      title: 'A convex algorithm has a narrower contract',
      headline: 'The four-corner shell uses the convex solver',
      textAlternative:
        'A four-corner convex footprint produces GPU skeleton arcs and a requested inward offset contour.',
      body: 'Select **Convex · 814558** to run the strictly convex solver. It handles simultaneous collapses, while reflex corners and holes stay with the vector solver.',
      evidence: '{{shapeEvidence}}',
      caveat: 'The convex solver rejects nonconvex rings, holes and collinear corners.',
      options: {footprint: 'convex', question: 'wavefront', offsetDistance: 4, budget: 'resolved'},
      optionsMode: 'fresh',
      controls: ['footprint', 'offsetDistance'],
      readouts: ['shapeStatus', 'shapeEvidence'],
      basemap: ground('paperCity')
    },
    {
      id: 'circle',
      title: 'A different question finds a circle',
      headline: 'Largest circle balances distance to every edge',
      textAlternative:
        'A blue circle and nearest-boundary witness sit inside the selected footprint.',
      body: 'The **interior circle** uses branch-and-bound over the polygon and its holes. Read its center, radius, closest boundary witness and remaining search error together.',
      evidence: '{{shapeEvidence}}',
      caveat:
        'SUCCESS means the geometric search tolerance was met; budget exhaustion keeps a bounded answer but is reported separately. Float32 rounding is not certified.',
      options: {footprint: 'courtyard', question: 'circle', tolerance: 0.2, budget: 'resolved'},
      optionsMode: 'fresh',
      controls: ['tolerance', 'budget'],
      readouts: ['shapeStatus', 'shapeEvidence'],
      camera: {longitude: -87.64917, latitude: 41.8837, zoom: 18.2},
      basemap: ground('paperCity')
    },
    {
      id: 'clearance',
      title: 'Minimum clearance measures vertex displacement',
      headline: 'The witness measures invalidity sensitivity',
      textAlternative:
        'A blue segment joins the fragile boundary vertex to the nearest non-adjacent segment or vertex.',
      body: '**Minimum clearance** is the smallest displacement of one vertex that makes the geometry invalid. The GPU draws the actual fragile-vertex witness.',
      evidence: '{{shapeEvidence}}',
      caveat: 'This is not courtyard width, sidewalk width or free-space radius.',
      options: {question: 'clearance'},
      optionsMode: 'fresh',
      controls: ['footprint'],
      readouts: ['shapeStatus', 'shapeEvidence'],
      camera: {longitude: -87.64917, latitude: 41.8837, zoom: 18.2},
      basemap: ground('paperCity')
    },
    {
      id: 'offset',
      title: 'The raw curve needs resolution',
      headline: 'A selected boundary chain can split',
      textAlternative:
        'Terracotta connected offset components follow one selected directed shell boundary chain.',
      body: 'For **Move boundary**, the first exterior ring is treated as an open, directed line chain. Its stored closing coordinate is dropped, so the closing edge is omitted. The raw GPU offset is noded and resolved into selected components.',
      evidence: '{{shapeEvidence}}',
      caveat:
        'Positive line offset is left of the directed chain and negative is right; the result is a boundary chain, not a path. The convex option uses polygon buffering, where positive grows the filled shell and negative contracts it.',
      options: {
        question: 'move',
        footprint: 'courtyard',
        offsetDistance: 4,
        curveSide: 'left',
        bufferDirection: 'outward',
        join: 'round',
        quadSegments: 8
      },
      optionsMode: 'fresh',
      controls: ['footprint', 'offsetDistance', 'bufferDirection', 'curveSide', 'join'],
      readouts: ['shapeStatus', 'shapeEvidence'],
      camera: {longitude: -87.64917, latitude: 41.8837, zoom: 18.2},
      basemap: ground('paperCity')
    },
    {
      id: 'summary',
      title: 'Shape depends on the operation',
      headline: 'Boundary movement is not one universal answer',
      textAlternative:
        'The source boundary remains visible alongside the selected GPU output and its concise status.',
      body: 'Compare the **event trace**, largest interior circle, minimum displacement and topology-resolved offset surface as different geometric questions. Provenance expands stored vertices to ring and polygon ownership.',
      evidence: '{{shapeStatus}} · {{shapeEvidence}}',
      caveat:
        'These footprint-scale geometry operations make no engineering, roof-height, access or setback-compliance claim.',
      options: {question: 'move'},
      optionsMode: 'fresh',
      controls: ['question'],
      readouts: ['shapeStatus', 'shapeSource'],
      camera: {longitude: -87.64917, latitude: 41.8837, zoom: 18.2},
      basemap: ground('paperCity')
    }
  ],
  readouts: [
    {id: 'shapeStatus', label: 'GPU result', unit: 'status', emphasis: 'tile'},
    {id: 'shapeEvidence', label: 'Geometric evidence', unit: 'detail'},
    {id: 'shapeSource', label: 'Source and topology', unit: 'records'}
  ],
  snippet: state =>
    `GPU question: ${state.question}\nSource building: ${state.footprint === 'courtyard' ? 354712 : 814558}\nOffset: ${state.offsetDistance} m · ${state.join} joins`,
  about: {
    what: 'An interactive investigation of geometric sensitivity in two archived Chicago building footprints.',
    why: 'Wavefront traces, inscribed circles, minimum clearance and offsets answer related but distinct questions about a boundary.',
    howToRead:
      'Dark lines are the archived source. Terracotta marks moving-boundary results; blue marks geometric witnesses. All analysis uses a local planar meter frame centered at the selected source shell centroid.'
  }
});
