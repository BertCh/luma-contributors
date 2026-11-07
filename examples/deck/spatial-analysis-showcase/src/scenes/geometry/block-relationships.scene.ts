// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {BlockRelationshipsOptions} from './block-relationships-options';

const ERIE_BLOCK_VIEW = {longitude: -87.62887, latitude: 41.89431, zoom: 19.1};

function blockLegend(state: BlockRelationshipsOptions): LegendSpec[] {
  const entries =
    state.view === 'bounds'
      ? [
          {
            color: [52, 147, 157, 215] as const,
            label: 'Selected archived footprints',
            shape: 'swatch' as const
          },
          {
            color: [202, 114, 70, 255] as const,
            label: 'Analyst query rectangle',
            shape: 'line' as const
          },
          {
            color: [247, 187, 115, 105] as const,
            label: 'GPU bounds-filter candidates',
            shape: 'swatch' as const
          }
        ]
      : state.view === 'adjacency'
        ? [
            {
              color: [52, 147, 157, 215] as const,
              label: 'Archived source footprints',
              shape: 'swatch' as const
            },
            {
              color: [221, 126, 61, 255] as const,
              label: 'GPU shared mapped boundary',
              shape: 'line' as const
            },
            {
              color: [255, 251, 242, 255] as const,
              label: 'GPU adjacency component centers',
              shape: 'dot' as const
            }
          ]
        : state.view === 'tessellation'
          ? [
              {
                color: [215, 133, 72, 215] as const,
                label: 'Boundary-sampled cell for building 342227',
                shape: 'swatch' as const
              },
              {
                color: [52, 147, 157, 215] as const,
                label: 'Boundary-sampled cell for building 815550',
                shape: 'swatch' as const
              },
              {
                color: [124, 130, 183, 215] as const,
                label: 'Boundary-sampled cell for building 815551',
                shape: 'swatch' as const
              }
            ]
          : state.view === 'visibility'
            ? [
                {
                  color: [52, 147, 157, 215] as const,
                  label: 'Archived obstacle footprints',
                  shape: 'swatch' as const
                },
                {
                  color: [105, 125, 121, 170] as const,
                  label: 'Synthetic convex study enclosure',
                  shape: 'line' as const
                },
                {
                  color: [46, 129, 145, 220] as const,
                  label: 'GPU visibility edge',
                  shape: 'line' as const
                },
                {
                  color: [217, 115, 63, 255] as const,
                  label: 'Synthetic outside observer',
                  shape: 'dot' as const
                }
              ]
            : [
                {
                  color: [215, 133, 72, 215] as const,
                  label: 'Archived footprint · source row 21',
                  shape: 'swatch' as const
                },
                {
                  color: [52, 147, 157, 215] as const,
                  label: 'Archived footprint · source row 1118',
                  shape: 'swatch' as const
                },
                {
                  color: [124, 130, 183, 215] as const,
                  label: 'Archived footprint · source row 1119',
                  shape: 'swatch' as const
                }
              ];
  return [{kind: 'categories', title: 'Mapped and constructed marks', entries}];
}

/** A bounded inquiry into mapped contact, boundary-sampled cells and hypothetical sightlines. */
export default defineScene<BlockRelationshipsOptions>({
  id: 'block-relationships',
  title: 'How are footprints related?',
  chapter: 'geometry',
  order: 13,
  summary:
    'Use three archived Chicago footprints to compare GPU bounds candidates, unique shared-boundary runs, boundary-sampled proximity cells and synthetic observer visibility.',
  contributors: [
    'GPUBoundsFilter',
    'GPUSharedPaths',
    'addBuildingAdjacencyRecipe',
    'GPUBufferSurface',
    'addBuildingTessellationRecipe',
    'addVisibilityGraphRecipe'
  ],
  datasets: [
    {id: 'chicago-buildings', role: 'archived source footprints and building identifiers'}
  ],
  initialView: ERIE_BLOCK_VIEW,
  basemap: ground('paperSheet'),
  furniture: {
    title: {
      title: 'How are footprints related?',
      subtitle: 'Three archived buildings · West Erie Street · Chicago'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cityOfChicago),
    caveat:
      'The bundled building layer is an archived footprint selection. Mapped boundary contact does not verify a physical party wall, ownership, occupancy or present-day conditions. The inset, enclosure and observers are analyst-defined scenarios; the planar frame and boundary sampling affect constructed outputs.'
  },
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Evidence view',
      group: 'GPU evidence',
      apply: 'param',
      default: 'source',
      options: [
        {value: 'source', label: 'Archived source footprints'},
        {value: 'bounds', label: 'Bounds-filter candidates'},
        {value: 'adjacency', label: 'Shared boundary + adjacency'},
        {value: 'tessellation', label: 'Boundary-sampled cells'},
        {value: 'visibility', label: 'Synthetic observer graph'}
      ],
      help: 'Selects the map overlay made from the corresponding GPU result. Incomplete or uncertain computed geometry is withheld.'
    },
    {
      kind: 'slider',
      id: 'queryHalfExtent',
      label: 'Query half-width',
      group: 'Bounds filter',
      apply: 'param',
      min: 8,
      max: 48,
      step: 4,
      default: 20,
      unit: 'm',
      disabledWhen: state => state.view !== 'bounds',
      help: 'Expands a square centered on the local analysis origin. GPUBoundsFilter compares each GPU-measured footprint bounding box with this rectangle; it is a candidate relation, not an exact polygon intersection.'
    },
    {
      kind: 'select',
      id: 'boundsRelation',
      label: 'Bounds relation',
      group: 'Bounds filter',
      apply: 'compile',
      default: 'intersects',
      disabledWhen: state => state.view !== 'bounds',
      options: [
        {value: 'intersects', label: 'Intersects query envelope'},
        {value: 'within', label: 'Within query envelope'},
        {value: 'contains', label: 'Contains query envelope'}
      ],
      help: 'Applies the selected relation to GPU-measured axis-aligned footprint bounds.'
    },
    {
      kind: 'select',
      id: 'inset',
      label: 'Synthetic inset',
      group: 'Tessellation scenario',
      apply: 'compile',
      default: '0.4',
      disabledWhen: state => state.view !== 'tessellation',
      options: [
        {value: '0', label: '0 m · retain mapped boundaries'},
        {value: '0.4', label: '0.4 m · hypothetical shrink'}
      ],
      help: 'Changes GPUBufferSurface preprocessing. The zero-distance run retains touching boundary samples, which can create duplicate sites.'
    },
    {
      kind: 'select',
      id: 'sampleSpacing',
      label: 'Boundary sample spacing',
      group: 'Tessellation scenario',
      apply: 'param',
      default: '8',
      disabledWhen: state => state.view !== 'tessellation',
      options: [
        {value: '4', label: '4 m · denser samples'},
        {value: '8', label: '8 m · balanced'},
        {value: '16', label: '16 m · coarser samples'}
      ],
      help: 'Rebuilds the boundary-sampled Voronoi partition from GPU-readback boundaries. The selected inset and sample capacity define the input support.'
    },
    {
      kind: 'select',
      id: 'sampleCapacity',
      label: 'Site budget',
      group: 'Tessellation scenario',
      apply: 'param',
      default: '128',
      disabledWhen: state => state.view !== 'tessellation',
      options: [
        {value: '16', label: '16 sites'},
        {value: '128', label: '128 sites'}
      ],
      help: 'Sets the GPU tessellation site capacity. A smaller budget may produce an incomplete result, which is withheld rather than presented as a full partition.'
    },
    {
      kind: 'select',
      id: 'visibilityDistance',
      label: 'Maximum sightline distance',
      group: 'Visibility scenario',
      apply: 'compile',
      default: '55',
      disabledWhen: state => state.view !== 'visibility',
      options: [
        {value: '25', label: '25 m'},
        {value: '55', label: '55 m'},
        {value: '100', label: '100 m'}
      ],
      help: 'Limits analyst-defined probe pairs before the dense GPU graph is analyzed.'
    },
    {
      kind: 'select',
      id: 'visibilityEdgeCapacity',
      label: 'Displayed edge capacity',
      group: 'Visibility scenario',
      apply: 'compile',
      default: '120',
      disabledWhen: state => state.view !== 'visibility',
      options: [
        {value: '8', label: '8 edges · compact list'},
        {value: '120', label: '120 edges · full list'}
      ],
      help: 'Bounds the copied/rendered edge list only. Dense visibility, shortest paths and graph metrics still use every valid probe pair; status reports truncation.'
    },
    {
      kind: 'select',
      id: 'visibilityMetric',
      label: 'Probe metric',
      group: 'Visibility scenario',
      apply: 'param',
      default: 'degree',
      disabledWhen: state => state.view !== 'visibility',
      options: [
        {value: 'degree', label: 'Visible degree'},
        {value: 'reachability', label: 'Reachable probes'},
        {value: 'mean-hop-distance', label: 'Mean shortest-path distance'},
        {value: 'hop-closeness', label: 'Reachable-hop closeness'},
        {value: 'choice', label: 'Betweenness choice'},
        {value: 'normalized-choice', label: 'Normalized choice'},
        {value: 'relative-asymmetry', label: 'Relative asymmetry'},
        {value: 'real-relative-asymmetry', label: 'Real relative asymmetry'},
        {value: 'hillier-integration', label: 'Hillier-Hanson integration'},
        {value: 'pyramid', label: 'Pyramid index'},
        {value: 'teklenburg', label: 'Teklenburg index'},
        {value: 'penn', label: 'Penn index'},
        {value: 'maximum-depth', label: 'Maximum shortest-path depth'}
      ],
      help: 'Selects an actual dense-graph output copied from GPUVisibilityGraph. Undefined values stay undefined for isolated or insufficiently connected probes.'
    }
  ],
  story: [
    {
      id: 'three-footprints',
      title: 'Start with three mapped buildings',
      headline: 'Nearby addresses are not a relationship model',
      textAlternative:
        'Three muted building footprints appear on West Erie Street, each labeled by an archived source building identifier. The scene isolates source rows 21, 1118 and 1119 from the bundled Chicago building archive and preserves their complete polygon rings.',
      body: 'The bounded selection is building IDs **342227** (source row 21, 16 W Erie St), **815550** (row 1118, 14 W Erie St) and **815551** (row 1119, 22 W Erie St). The address labels are copied from the archive. GPU operators use their complete rings projected to a local planar metre frame centered near the three footprints.',
      evidence: '**{{blockSource}}**',
      caveat:
        'This is an archived geometry sample, not a check of current buildings or a statement about occupants, parcel ownership or construction.',
      options: {
        view: 'source',
        queryHalfExtent: 20,
        boundsRelation: 'intersects',
        inset: '0.4',
        sampleSpacing: '8',
        sampleCapacity: '128',
        visibilityDistance: '55',
        visibilityEdgeCapacity: '120',
        visibilityMetric: 'degree'
      },
      controls: [],
      readouts: ['blockSource'],
      camera: {...ERIE_BLOCK_VIEW, transitionMs: 900}
    },
    {
      id: 'bounds-candidates',
      title: 'Filter measured bounding boxes',
      headline: 'A rectangle narrows the geometry candidates',
      textAlternative:
        'A warm query rectangle surrounds a subset of the three source footprints. Filled candidate footprints and the rectangle correspond to the GPU bounds-filter result and current half-width control; the readout reports written, required and overflow counts.',
      body: '`GPUGeometryMeasures` writes each source footprint’s bounds, and `GPUBoundsFilter` tests their intersection with the centered square. Change the half-width to see GPU candidate IDs change. A bounding-box match can include a footprint even when its polygon would not intersect the query.',
      evidence: '**{{blockBounds}}**',
      caveat:
        'The operator answers a bounds relation. It does not test exact footprint interiors or infer building adjacency.',
      options: {
        view: 'bounds',
        queryHalfExtent: 12,
        boundsRelation: 'intersects',
        inset: '0.4',
        sampleSpacing: '8',
        sampleCapacity: '128',
        visibilityDistance: '55',
        visibilityEdgeCapacity: '120',
        visibilityMetric: 'degree'
      },
      controls: ['view', 'queryHalfExtent', 'boundsRelation'],
      readouts: ['blockBounds'],
      camera: {...ERIE_BLOCK_VIEW, transitionMs: 700}
    },
    {
      id: 'mapped-adjacency',
      title: 'Trace unique shared mapped runs',
      headline: 'Boundary overlap becomes pairwise evidence',
      textAlternative:
        'Warm linework highlights unique GPU shared-path runs between the three mapped footprints. A bar chart compares GPU-computed shared boundary lengths for each unordered building pair, and colored center markers show the GPU adjacency component labels.',
      body: '`GPUSharedPaths` compares the selected source rings, including self-query pairs. The scene keeps each distinct cross-building line pair once, then feeds those actual GPU runs into `addBuildingAdjacencyRecipe` with GPU-derived axial orientations. The chart reports mapped contact length in metres; component labels use the recipe’s positive-length adjacency rule.',
      evidence: '**{{blockAdjacency}}**',
      caveat:
        'Shared mapped boundary is a topological footprint relation. It is not evidence of a physical party wall, common ownership or direct access.',
      options: {
        view: 'adjacency',
        queryHalfExtent: 20,
        boundsRelation: 'intersects',
        inset: '0.4',
        sampleSpacing: '8',
        sampleCapacity: '128',
        visibilityDistance: '55',
        visibilityEdgeCapacity: '120',
        visibilityMetric: 'degree'
      },
      controls: ['view'],
      readouts: ['blockAdjacency', 'blockAdjacencyChart'],
      camera: {...ERIE_BLOCK_VIEW, transitionMs: 700}
    },
    {
      id: 'sampled-support',
      title: 'Build cells from inset boundaries',
      headline: 'Sampling and preprocessing define the support',
      textAlternative:
        'Three softly colored polygons show GPU boundary-sampled Voronoi cells inside a synthetic convex enclosure. The cells are computed from GPUBufferSurface output using the selected synthetic inset, boundary spacing and site budget, with the per-building area chart visible.',
      body: '`GPUBufferSurface` preprocesses each footprint using the selected zero or hypothetical 0.4 m inset. The scene reads those resulting segments back and passes them to `addBuildingTessellationRecipe`, which samples boundaries and clips nearest-site cells to the displayed convex enclosure. Compare 4, 8 or 16 m spacing and the 16- or 128-site budget. Zero inset retains touching boundaries, so duplicate boundary sites can make the recipe incomplete.',
      evidence: '**{{blockTessellation}}**',
      caveat:
        'These are modelled proximity cells, not parcel boundaries, catchments, access zones or property areas. Inset distance, boundary sampling, enclosure and planar projection are assumptions that shape the result.',
      options: {
        view: 'tessellation',
        queryHalfExtent: 20,
        boundsRelation: 'intersects',
        inset: '0.4',
        sampleSpacing: '8',
        sampleCapacity: '128',
        visibilityDistance: '55',
        visibilityEdgeCapacity: '120',
        visibilityMetric: 'degree'
      },
      controls: ['view', 'inset', 'sampleSpacing', 'sampleCapacity'],
      readouts: ['blockTessellation', 'blockTessellationChart'],
      camera: {...ERIE_BLOCK_VIEW, transitionMs: 700}
    },
    {
      id: 'synthetic-sightlines',
      title: 'Place hypothetical observers outside',
      headline: 'A visibility graph connects unobstructed probes',
      textAlternative:
        'Sixteen orange synthetic observer points sit outside three archived footprint obstacles inside a dashed analyst-defined enclosure. Teal GPU lines show the copied edge list up to its selected capacity; the chart reports the selected metric for each probe from the full dense visibility graph.',
      body: '`addVisibilityGraphRecipe` classifies these analyst-placed probes against the filled building rings, traces sightlines through mapped obstacles, and derives graph metrics. Choose the maximum sightline distance and a compact or full copied edge capacity. Edge-list truncation changes only which lines are drawn; dense visibility, shortest paths and metrics still use every valid probe pair. Select degree, reachability, hop integration, choice or an extended integration metric for the per-probe chart.',
      evidence: '**{{blockVisibility}}**',
      caveat:
        'The graph has no building heights, doors, street network, pedestrian rules or measured observer locations. Hop-based metrics count graph steps, not metres, and this is not an access or visibility study of people.',
      options: {
        view: 'visibility',
        queryHalfExtent: 20,
        boundsRelation: 'intersects',
        inset: '0.4',
        sampleSpacing: '8',
        sampleCapacity: '128',
        visibilityDistance: '55',
        visibilityEdgeCapacity: '120',
        visibilityMetric: 'degree'
      },
      controls: ['view', 'visibilityDistance', 'visibilityEdgeCapacity', 'visibilityMetric'],
      readouts: ['blockVisibility', 'blockVisibilityChart'],
      camera: {...ERIE_BLOCK_VIEW, transitionMs: 700}
    },
    {
      id: 'compare-supports',
      title: 'Compare the constructed relationships',
      headline: 'Each overlay answers a different geometric question',
      textAlternative:
        'The three source footprints remain on the map while the selected GPU overlay changes among bounds candidates, shared mapped boundaries, inset boundary-sampled cells and synthetic visibility edges. Readouts and charts report the active operation’s output and status.',
      body: 'Use the evidence selector to return to each GPU result. Bounds candidates summarize measured envelopes; adjacency describes unique shared line runs; cells depend on the inset and sampling interval; visibility metrics describe a synthetic observer graph. The source footprints anchor all four interpretations.',
      evidence:
        '**{{blockBounds}}** · **{{blockAdjacency}}** · **{{blockTessellation}}** · **{{blockVisibility}}**',
      caveat:
        'Every derived layer inherits the archived source geometry and the stated planar or synthetic assumptions. None verifies a current physical relationship between buildings.',
      options: {
        view: 'source',
        queryHalfExtent: 20,
        boundsRelation: 'intersects',
        inset: '0.4',
        sampleSpacing: '8',
        sampleCapacity: '128',
        visibilityDistance: '55',
        visibilityEdgeCapacity: '120',
        visibilityMetric: 'degree'
      },
      controls: ['view', 'inset', 'sampleSpacing', 'sampleCapacity', 'visibilityMetric'],
      readouts: ['blockBounds', 'blockAdjacency', 'blockTessellation', 'blockVisibility'],
      camera: {...ERIE_BLOCK_VIEW, transitionMs: 700}
    }
  ],
  readouts: [
    {id: 'blockSource', label: 'Archived source selection', layout: 'block'},
    {id: 'blockBounds', label: 'GPU bounds-filter result', layout: 'block'},
    {id: 'blockAdjacency', label: 'GPU shared paths and adjacency', layout: 'block'},
    {id: 'blockAdjacencyChart', label: 'Shared mapped boundary length (m)', kind: 'chart'},
    {id: 'blockTessellation', label: 'Inset boundary sampling and cell status', layout: 'block'},
    {id: 'blockTessellationChart', label: 'Boundary-sampled cell area (m²)', kind: 'chart'},
    {id: 'blockVisibility', label: 'Synthetic observer graph status', layout: 'block'},
    {id: 'blockVisibilityChart', label: 'Selected dense-graph metric by probe', kind: 'chart'}
  ],
  legends: blockLegend,
  snippet: state =>
    `graph.add(new GPUBoundsFilter({bounds: footprintBounds, parameters: queryBounds, mode: 'intersects', output: boundsCandidates}));\n` +
    `graph.add(new GPUSharedPaths({left: buildingRings, right: buildingRings, intersectionCapacity: 256, output: sharedRuns}));\n` +
    `// Keep each cross-building unordered line pair once before addBuildingAdjacencyRecipe.\n` +
    `addBuildingAdjacencyRecipe(graph, {sharedPaths: sharedRuns, lineBuildingIds, orientations});\n` +
    `// Boundary sample spacing: ${state.sampleSpacing} m; synthetic inset: ${state.inset} m; site budget: ${state.sampleCapacity}.`,
  about: {
    what: '`GPUBoundsFilter` selects footprint-envelope candidates, `GPUSharedPaths` finds common line runs, `addBuildingAdjacencyRecipe` measures those runs and components, `GPUBufferSurface` creates a synthetic inset for boundary-sampled cells, and `addVisibilityGraphRecipe` evaluates a small synthetic observer graph.',
    why: 'Footprint proximity, shared mapped edges, sampled nearest-site partitions and unobstructed graph connections express different relationships. Comparing their GPU outputs makes their inputs and assumptions visible.',
    howToRead:
      'All three source features are retained from the archived Chicago building dataset and projected near their source extent into a local planar metre frame. GPU shared paths are reduced to unique unordered cross-building line pairs before adjacency analysis. Tessellation uses GPU-readback inset segments and the displayed convex enclosure. The dashed enclosure and observer coordinates are analyst-defined. Incomplete, overflowing or uncertain geometric outputs are withheld.'
  },
  create: async context =>
    (await import('./block-relationships.compute')).createBlockRelationships(context)
});
