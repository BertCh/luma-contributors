// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SphericalBoundariesOptions} from './spherical-boundaries.compute';

export default defineScene<SphericalBoundariesOptions>({
  id: 'spherical-boundaries',
  title: 'Fiji on a sphere',
  chapter: 'earth',
  order: 8,
  summary:
    'Inspect one generalized Fiji shell with spherical containment, Boolean overlays and adaptive mixed-level S2 covers beside a synthetic analytic window.',
  contributors: [
    'GPUSphericalPolygonContains',
    'GPUSphericalPolygonOverlay',
    'GPUS2PolygonCovering',
    'GPUCellGeometry',
    'GPUS2CellMeasures'
  ],
  datasets: [{id: 'natural-earth', role: 'Natural Earth 1:110m generalized country boundaries'}],
  initialView: {longitude: 178.15, latitude: -17.3, zoom: 7.15},
  basemap: ground('paperSheet', {labels: 'none', graticule: {stepDegrees: 1, opacity: 0.35}}),
  furniture: {
    title: {
      title: 'Fiji on a sphere',
      subtitle: 'Viti Levu · generalized 1:110m geography'
    },
    credit: 'Natural Earth 1:110m countries · Public domain',
    caveat:
      'Natural Earth 1:110m · Great-circle edges between generalized vertices; the amber window is synthetic.'
  },
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Analysis view',
      group: 'Reading the boundary',
      apply: 'param',
      default: 'source',
      options: [
        {value: 'source', label: 'Source geometry'},
        {value: 'contains', label: 'Spherical containment'},
        {value: 'overlay', label: 'Polygon Boolean'},
        {value: 'cover', label: 'Adaptive S2 cover'}
      ],
      help: 'Switches among cached outputs from the GPU graph.'
    },
    {
      kind: 'select',
      id: 'boundary',
      label: 'Boundary rule',
      group: 'Containment predicate',
      apply: 'param',
      default: 'closed',
      options: [
        {value: 'closed', label: 'Closed: include boundary'},
        {value: 'open', label: 'Open: exclude boundary'}
      ],
      help: 'Chooses whether the spherical winding predicate accepts probes classified on the polygon arc within tolerance.'
    },
    {
      kind: 'toggle',
      id: 'complement',
      label: 'Complement selected shell',
      group: 'Containment predicate',
      apply: 'param',
      default: false,
      help: 'Reads the selected Fiji shell as sphere-minus-shell. This option applies to one shell, so it does not complement each part of Fiji independently.'
    },
    {
      kind: 'select',
      id: 'operation',
      label: 'Boolean operation',
      group: 'Analytic window',
      apply: 'param',
      default: 'intersection',
      options: [
        {value: 'intersection', label: 'Intersection'},
        {value: 'union', label: 'Union'},
        {value: 'difference', label: 'Source minus window'},
        {value: 'symmetric-difference', label: 'Symmetric difference'}
      ],
      help: 'Selects one precomputed snap-rounded spherical overlay between the source shell and the amber synthetic window.'
    },
    {
      kind: 'select',
      id: 'coverMode',
      label: 'Cover guarantee',
      group: 'Adaptive S2 cells',
      apply: 'param',
      default: 'exterior',
      options: [
        {value: 'exterior', label: 'Exterior: include intersections'},
        {value: 'interior', label: 'Interior: retain contained cells'}
      ],
      help: 'Shows one adaptive mixed-level cover of the selected source shell. Cells are candidate supports, not exact boundary geometry.'
    },
    {
      kind: 'select',
      id: 'minimumLevel',
      label: 'Minimum S2 level',
      group: 'Adaptive S2 cells',
      apply: 'compile',
      default: '3',
      options: [
        {value: '3', label: '3 · coarser starting cells'},
        {value: '5', label: '5 · finer starting cells'}
      ],
      help: 'Rebuilds the cover graph lazily. Every available maximum is at least 6, so minimum never exceeds maximum.'
    },
    {
      kind: 'select',
      id: 'maximumLevel',
      label: 'Maximum S2 level',
      group: 'Adaptive S2 cells',
      apply: 'compile',
      default: '8',
      options: [
        {value: '6', label: '6 · broader boundary cells'},
        {value: '8', label: '8 · medium boundary cells'},
        {value: '10', label: '10 · finer boundary cells'}
      ],
      help: 'Sets the finest hierarchy depth and lazily compiles that cover graph variant.'
    }
  ],
  story: [
    {
      id: 'source-split',
      title: 'The source keeps three separate shells',
      headline: 'Only Viti Levu enters the calculations',
      textAlternative:
        'A bright selected-shell outline marks Viti Levu; two lighter outlines show the other published Fiji shells, and an amber rectangle marks the synthetic analytic window.',
      body: 'Natural Earth stores Fiji as three separate generalized shells. Viti Levu is the selected input for the calculations; the two Vanua Levu pieces remain visible as context and are not joined or analyzed. The amber four-corner window is a synthetic geometry probe.',
      evidence:
        'The source file stores 8, 9 and 5 positions including closure across Fiji’s three parts. The GPU input is only the middle, Viti Levu shell.',
      caveat:
        'Natural Earth 1:110m is generalized cartography. It is neither survey coastline nor a coastal hazard delineation, and this view does not reconstruct the source’s split shells.',
      options: {view: 'source', boundary: 'closed', complement: false},
      controls: ['view'],
      camera: {longitude: 178.15, latitude: -17.3, zoom: 7.15, transitionMs: 1200}
    },
    {
      id: 'winding-rule',
      title: 'Containment follows minor great-circle arcs',
      headline: 'The predicate tests unit vectors on the sphere',
      textAlternative:
        'A synthetic grid of gray outside probes and amber inside probes surrounds the Fiji shell; exact source-vertex probes are teal boundary marks under the selected closed or open rule.',
      body: '`GPUSphericalPolygonContains` applies a spherical winding test to unit XYZ probes and minor great-circle edges. The deterministic grid is a mathematical probe, not a sample of people or observations. Exact source vertices expose the `Boundary rule`; `Complement selected shell` flips the region for this one shell.',
      evidence:
        '**{{insideCount}}** probes are accepted, **{{boundaryCount}}** lie on the boundary, and **{{invalidCount}}** are invalid for the selected rule.',
      caveat:
        'The tolerance is 1e-6 in unit-vector coordinates. It is an angular numerical tolerance, not a ground distance or a statement about shoreline accuracy.',
      options: {view: 'contains', boundary: 'closed', complement: false},
      controls: ['boundary', 'complement'],
      readouts: ['insideCount', 'boundaryCount', 'invalidCount'],
      camera: {longitude: 177.95, latitude: -17.8, zoom: 8.1, transitionMs: 900}
    },
    {
      id: 'intersection',
      title: 'Intersect the source with an analytic window',
      headline: 'The teal outline is computed output geometry',
      textAlternative:
        'The Fiji source is dark ink, the synthetic analytic window is amber, and the GPU-computed intersection is teal; selected source and window edges retain separate provenance colors.',
      body: '`GPUSphericalPolygonOverlay` projects both operands into one gnomonic hemisphere chart. Minor great-circle arcs become straight chart segments, where the snap-rounded arrangement builds the requested Boolean result. The teal outline is read back from the GPU output; amber and teal thin segments indicate source-edge and window-edge provenance.',
      evidence:
        '**{{overlayFeatures}}** computed polygons. The surviving outline separates the overlap from the rest of the two inputs.',
      caveat:
        'The snap tolerance `2⁻¹⁸` is in chart units, not metres. The window is synthetic, and the chart rejects vertices closer to its horizon than minimum cosine 0.5.',
      options: {view: 'overlay', operation: 'intersection'},
      controls: ['operation'],
      readouts: ['overlayFeatures'],
      camera: {longitude: 178.15, latitude: -17.3, zoom: 7.15, transitionMs: 900}
    },
    {
      id: 'boolean-family',
      title: 'Boolean rules change which edges survive',
      headline: 'Four operations share one pair of boundaries',
      textAlternative:
        'The selected Boolean output is teal while amber and teal edge segments show source and analytic-window provenance; changing the operation changes the actual GPU-computed boundary.',
      body: 'Choose `Boolean operation` to keep both inputs, subtract the window from Viti Levu, or retain only their non-overlapping parts. Each choice draws its own GPU-produced boundary. Thin amber and teal segments identify which input supplied each surviving edge.',
      evidence:
        '**{{overlayFeatures}}** computed polygons. Changing the rule changes which source and window edges survive; diagnostic counts remain in Details.',
      caveat:
        'All four operations use the same generalized shell and explicitly synthetic window. These results describe their supplied polygon inputs, not Fiji’s authoritative territory or coastline.',
      options: {view: 'overlay', operation: 'symmetric-difference'},
      controls: ['operation'],
      readouts: ['overlayFeatures'],
      camera: {longitude: 178.15, latitude: -17.3, zoom: 7.15, transitionMs: 900}
    },
    {
      id: 'adaptive-cover',
      title: 'S2 cells bracket the polygon at mixed levels',
      headline: 'Adaptive cells are candidate support units',
      textAlternative:
        'Blue exterior or teal interior S2 outlines have mixed cell sizes around the dark Fiji shell; the readout reports GPU-written cells, visited cells and summed spherical cell area.',
      body: 'S2 refines cells near the boundary while retaining larger cells where possible. Exterior cells include every intersection with the shell; interior cells must fit entirely inside it. Increase the maximum level to see these two area bounds tighten around Viti Levu.',
      evidence:
        '**{{coverCells}}** cells represent **{{coverArea}}** of spherical support. Compare exterior and interior to see the area left uncertain by the selected cell scale.',
      caveat:
        'A nonzero status indicates invalid input, output capacity exhaustion or traversal-budget exhaustion. Written count is not a complete cover when the status reports exhaustion; cell outlines are candidates, not the exact boundary.',
      options: {view: 'cover', coverMode: 'exterior'},
      controls: ['coverMode', 'minimumLevel', 'maximumLevel'],
      readouts: ['coverCells', 'coverArea'],
      camera: {longitude: 178.3, latitude: -17.65, zoom: 7.15, transitionMs: 900}
    }
  ],
  legends: state => [
    {
      id: 'boundary-plate',
      title: 'Boundary plate',
      kind: 'categories',
      entries: [
        {color: [18, 91, 122, 255], label: 'selected Viti Levu shell'},
        {color: [116, 138, 144, 170], label: 'Vanua Levu source shells (context)'},
        ...(state.view === 'source' || state.view === 'overlay'
          ? [
              {
                color: [231, 167, 76, 210] as [number, number, number, number],
                label: 'synthetic analytic window'
              }
            ]
          : []),
        ...(state.view === 'contains'
          ? [
              {
                color: [230, 163, 71, 230] as [number, number, number, number],
                label: 'accepted probes'
              },
              {
                color: [119, 150, 162, 150] as [number, number, number, number],
                label: 'rejected probes'
              },
              {
                color: [13, 139, 141, 255] as [number, number, number, number],
                label: 'boundary probes'
              },
              {
                color: [196, 75, 64, 255] as [number, number, number, number],
                label: 'invalid probes'
              }
            ]
          : []),
        ...(state.view === 'overlay'
          ? [
              {
                color: [18, 142, 139, 245] as [number, number, number, number],
                label: 'GPU Boolean result'
              },
              {
                color: [237, 176, 75, 255] as [number, number, number, number],
                label: 'source boundary provenance'
              },
              {
                color: [18, 142, 139, 255] as [number, number, number, number],
                label: 'window boundary provenance'
              }
            ]
          : []),
        ...(state.view === 'cover'
          ? [
              {
                color: (state.coverMode === 'exterior'
                  ? [42, 125, 152, 175]
                  : [12, 145, 137, 190]) as [number, number, number, number],
                label: `${state.coverMode} adaptive S2 cell boundaries`
              }
            ]
          : [])
      ],
      note: 'Natural Earth shell segmentation is retained. The analysis uses Viti Levu alone; the window and probes are synthetic.'
    }
  ],
  readouts: [
    {id: 'insideCount', label: 'Accepted spherical probes', unit: 'probes', emphasis: 'tile'},
    {id: 'boundaryCount', label: 'Boundary probes', unit: 'probes'},
    {id: 'invalidCount', label: 'Invalid probes', unit: 'probes'},
    {
      id: 'overlayFeatures',
      label: 'GPU Boolean output polygons',
      unit: 'polygons',
      emphasis: 'tile'
    },
    {
      id: 'overlayStatus',
      label: 'Overlay status and provenance counts',
      layout: 'block',
      hood: true
    },
    {id: 'coverCells', label: 'Adaptive S2 cells written', unit: 'cells', emphasis: 'tile'},
    {id: 'coverArea', label: 'Summed S2 support area'},
    {id: 'coverStatus', label: 'Cover traversal and capacity status', layout: 'block', hood: true},
    {id: 'computeStatus', label: 'Selected GPU result', layout: 'block', hood: true}
  ],
  snippet: state =>
    state.view === 'contains'
      ? `// Schematic: source/query views and caller-owned output ports are abbreviated.\ngraph.add(new GPUSphericalPolygonContains({ boundary: '${state.boundary}', complement: ${state.complement}, tolerance: 1e-6, positions: unitXYZ, queries, output }));`
      : state.view === 'overlay'
        ? `// Schematic: left/right polygon views, chart basis and full output ports are abbreviated.\ngraph.add(new GPUSphericalPolygonOverlay({ operation: '${state.operation}', chartCenter, vertexTolerance: 2 ** -18, minimumCosine: 0.5, capacity, output }));`
        : state.view === 'cover'
          ? `// Schematic: prepared polygon views and caller-owned output ports are abbreviated.\ngraph.add(new GPUS2PolygonCovering({ minimumLevel: ${state.minimumLevel}, maximumLevel: ${state.maximumLevel}, mode: '${state.coverMode}', capacity: 4096, maximumVisitedCells: 65536, output }));`
          : '// The selected Natural Earth source shell is rendered directly; no GPU operation runs in this view.',
  about: {
    what: '`GPUSphericalPolygonContains` applies nonconvex spherical winding containment to unit XYZ probes. `GPUSphericalPolygonOverlay` performs shell Boolean operations in a shared gnomonic hemisphere chart. `GPUS2PolygonCovering` emits a bounded adaptive mixed-level cover, while `GPUCellGeometry` and `GPUS2CellMeasures` decode cell boundaries and areas.',
    why: 'Polygon predicates, Boolean geometry and spatial indexing answer different boundary questions. This demonstration uses one generalized Fiji shell and a disclosed synthetic window; it does not join Fiji’s split source shells.',
    howToRead:
      'The blue outline is the selected generalized Viti Levu shell; gray outlines retain the other two published shells as context. Amber marks the synthetic window. Teal marks computed output or boundary probes. S2 cells are candidate support, not an authoritative coast. Read each status row with its count.'
  },
  create: async ctx =>
    (await import('./spherical-boundaries.compute')).createSphericalBoundaries(ctx)
});
