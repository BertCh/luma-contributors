// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {SiteMeshesOptions} from './site-meshes.compute';

const CHICAGO = {longitude: -87.7, latitude: 41.86, zoom: 9.6};
const COURTYARD = {longitude: -87.64917, latitude: 41.8837, zoom: 18.3};
function siteLegend(state: SiteMeshesOptions): LegendSpec[] {
  const entries =
    state.view === 'courtyard'
      ? [
          {
            color: [34, 104, 113, 235] as const,
            label: 'Source shell and courtyard ring',
            shape: 'line' as const
          },
          {
            color: [65, 129, 135, 190] as const,
            label: 'GPU polygon triangulation',
            shape: 'line' as const
          },
          {
            color: [189, 122, 73, 215] as const,
            label: 'GPU constrained Delaunay mesh',
            shape: 'line' as const
          }
        ]
      : state.view === 'mesh'
        ? [
            {
              color: [59, 110, 121, 190] as const,
              label: 'GPU Delaunay triangles',
              shape: 'line' as const
            },
            {
              color: [181, 126, 87, 220] as const,
              label: 'GPU clipped Voronoi edges',
              shape: 'line' as const
            },
            {
              color: [190, 103, 67, 215] as const,
              label: 'Circumcentres derived from GPU triangles',
              shape: 'dot' as const
            },
            {
              color: [97, 104, 100, 150] as const,
              label: 'Voronoi clipping box',
              shape: 'line' as const
            },
            {
              color: [37, 114, 123, 230] as const,
              label: 'Archived library location',
              shape: 'dot' as const
            }
          ]
        : state.view === 'groups'
          ? [
              {
                color: [54, 122, 135, 210] as const,
                label: 'Library group boundary',
                shape: 'line' as const
              },
              {
                color: [187, 110, 72, 220] as const,
                label: 'Fire-station group boundary',
                shape: 'line' as const
              },
              {
                color: [37, 114, 123, 230] as const,
                label: 'Archived library location',
                shape: 'dot' as const
              },
              {
                color: [194, 107, 71, 225] as const,
                label: 'Archived fire station',
                shape: 'dot' as const
              }
            ]
          : [
              {
                color: [178, 111, 72, 225] as const,
                label: 'GPU constructed hull boundary',
                shape: 'line' as const
              },
              {
                color: [37, 114, 123, 230] as const,
                label: 'Archived library location',
                shape: 'dot' as const
              }
            ];
  return [{kind: 'categories', title: 'Map marks', entries}];
}

/** Site meshes derived from archived public facility locations. */
export default defineScene<SiteMeshesOptions>({
  id: 'site-meshes',
  create: async context => (await import('./site-meshes.compute')).createSiteMeshes(context),
  title: 'What boundary can sites justify?',
  chapter: 'geometry',
  order: 12,
  summary:
    'Compare Chicago facility-site triangulations and constructed hulls, then inspect two GPU meshes of a real building with a courtyard.',
  contributors: [
    'GPUDelaunayTessellation',
    'GPUVoronoiDiagram',
    'GPUConcaveHull',
    'GPURatioConcaveHull',
    'GPUGroupConcaveHull',
    'GPUPolygonTriangulation',
    'GPUConstrainedDelaunayTriangulation'
  ],
  datasets: [
    {id: 'chicago-facilities', role: 'archived library and fire-station locations'},
    {id: 'chicago-buildings', role: 'building 354712 with a courtyard ring'}
  ],
  initialView: CHICAGO,
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Site meshes',
      subtitle: 'Constructed edges, bounded cells and a real courtyard'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cityOfChicago, CREDITS.openStreetMap),
    caveat:
      'These meshes describe geometry around archived mapped sites; they are not service areas, travel catchments or claims about facility access.'
  },
  legends: state => siteLegend(state),
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Geometry',
      group: 'Tool',
      apply: 'compile',
      default: 'mesh',
      help: 'Variants compile on first use. Each output is generated from the selected archived coordinates or building rings.',
      options: [
        {value: 'mesh', label: 'Delaunay and clipped Voronoi'},
        {value: 'alpha', label: 'Maximum-circumradius hull'},
        {value: 'ratio', label: 'Ratio concave hull'},
        {value: 'groups', label: 'Separate facility groups'},
        {value: 'courtyard', label: 'Building courtyard triangulation'}
      ]
    },
    {
      kind: 'slider',
      id: 'alphaRadius',
      label: 'Maximum circumradius',
      group: 'Concave hull',
      apply: 'compile',
      min: 1000,
      max: 12000,
      step: 500,
      default: 5000,
      unit: 'm',
      disabledWhen: state => state.view !== 'alpha' && state.view !== 'groups',
      help: 'The alpha-complex rule retains Delaunay triangles whose circumradius is at most this many metres. A radius is a length, not the ratio-hull control.'
    },
    {
      kind: 'slider',
      id: 'ratio',
      label: 'Edge-length ratio',
      group: 'Ratio hull',
      apply: 'compile',
      min: 0,
      max: 1,
      step: 0.05,
      default: 0.45,
      disabledWhen: state => state.view !== 'ratio',
      help: 'The ratio hull uses a dimensionless edge threshold; at 1 it becomes the convex hull. It is not a distance in metres and does not stand for a probability.'
    },
    {
      kind: 'toggle',
      id: 'allowHoles',
      label: 'Allow holes',
      group: 'Ratio hull',
      apply: 'compile',
      default: true,
      disabledWhen: state => state.view !== 'ratio',
      help: 'Whether the ratio hull may retain interior holes in the constructed boundary.'
    },
    {
      kind: 'slider',
      id: 'maximumFlips',
      label: 'Maximum edge flips',
      group: 'Courtyard mesh',
      apply: 'compile',
      min: 1,
      max: 20001,
      step: 1000,
      default: 8001,
      disabledWhen: state => state.view !== 'courtyard',
      help: 'Work budget for constrained Delaunay refinement. Validity and convergence are reported separately.'
    }
  ],
  story: [
    {
      id: 'delaunay-voronoi',
      title: 'Edges between mapped sites',
      headline: 'A dual mesh is not a walking boundary',
      textAlternative:
        'The map shows 82 distinct archived Chicago library sites, GPU Delaunay triangles, and clipped GPU Voronoi segments inside a measured rectangular box; Voronoi site-pair IDs and completion or overflow flags are reported. These straight-line constructions do not measure routes or access.',
      body: `The archived facility extract contains **82 library locations**, with no exact duplicate coordinates. The blue mesh comes from **GPUDelaunayTessellation**: each triangle joins neighbouring sites. The rust lines are the dual **GPUVoronoiDiagram**, clipped to the visible rectangle; the readout lists site pairs that share those boundaries.

Triangle circumcentres are derived on the CPU from the GPU triangle indices and shown as dots. They are construction points, not sites. The local frame is in metres; these straight-line edges say nothing about street travel, service reach or access.`,
      camera: {...CHICAGO, transitionMs: 1000},
      options: {view: 'mesh'},
      highlight: {readout: 'meshCounts'},
      controls: ['view'],
      readouts: ['meshStatus', 'meshCounts', 'voronoiPairs', 'meshCountsChart']
    },
    {
      id: 'alpha-hull',
      title: 'Filter triangles by circumradius',
      headline: 'A length threshold changes the boundary',
      textAlternative:
        'The warm boundary and retained-triangle counts come from the GPU alpha-complex hull at the selected maximum circumradius in metres. Increasing the radius can join components and fill gaps; the resulting outline is a geometric support, not a catchment.',
      body: `**GPUConcaveHull** filters the Delaunay triangles by their **maximum circumradius**. Small radii can leave isolated sites and disconnected pieces; larger radii can connect them and fill gaps. The teal points remain the complete input, even where a radius does not retain a surrounding triangle.

This is one explicit geometric rule for a boundary. It does not model travel time or facility service. Move **Maximum circumradius** to see the retained triangles and boundary edges change.`,
      options: {view: 'alpha', alphaRadius: 5000},
      highlight: {readout: 'hullCounts'},
      controls: ['alphaRadius'],
      readouts: ['hullParameter', 'hullStatus', 'hullCounts', 'hullCountsChart']
    },
    {
      id: 'ratio-hull',
      title: 'A different definition of concavity',
      headline: 'A unitless ratio is not a radius',
      textAlternative:
        'The ratio hull is computed by GPURatioConcaveHull from the same 82 library sites; a dimensionless edge ratio and hole toggle change its retained triangles and boundary. Ratio 1 gives the convex hull, while all original points remain visible.',
      body: `**GPURatioConcaveHull** is another construction, with a **dimensionless edge-length ratio** and an optional hole rule. At ratio 1 it reaches the convex hull. The parameter is not measured in metres, and its scale should not be compared numerically with the previous alpha radius.

The count and mask readbacks describe what the GPU retained. Every input location is still drawn, so a point outside the chosen boundary remains visible as evidence of the construction's limits.`,
      options: {view: 'ratio', ratio: 0.45, allowHoles: true},
      highlight: {readout: 'hullCounts'},
      controls: ['ratio', 'allowHoles'],
      readouts: ['hullParameter', 'hullStatus', 'hullCounts', 'hullCountsChart']
    },
    {
      id: 'group-hulls',
      title: 'Keep two source groups distinct',
      headline: 'Category hulls describe two point sets',
      textAlternative:
        'Blue points and teal boundary edges represent 82 archived library locations; rust points and edges represent 92 archived fire stations. GPUGroupConcaveHull computes two separate radius-filtered groups with per-group completion flags; neither boundary represents a service area.',
      body: `The optional comparison adds **92 archived fire-station locations** to the 82 libraries. **GPUGroupConcaveHull** computes an alpha-complex boundary for each source category independently, using the same radius in the same projected metre frame.

Blue is libraries; rust is fire stations. The categories stay separate, so nearby sites from different categories do not become one group's triangles. These polygons summarize point arrangements only; no facility capacity, travel network, hospitals or catchment model is included.`,
      options: {view: 'groups', alphaRadius: 5000},
      highlight: {readout: 'hullCounts'},
      controls: ['alphaRadius'],
      readouts: ['hullParameter', 'hullStatus', 'hullCounts', 'hullCountsChart']
    },
    {
      id: 'courtyard-mesh',
      title: 'A building with an actual hole',
      headline: 'Constrained triangles respect the courtyard',
      textAlternative:
        'Building 354712 is zoomed in as its source outline, with two input rings including a true courtyard hole. Teal GPU polygon triangulation and rust constrained Delaunay triangles are compared by triangle counts; separate validity and convergence flags show when the refinement budget stops early.',
      body: `This zoom uses actual Chicago building **354712**: one polygon with an exterior and a courtyard ring. The pale source outline remains visible. Teal is **GPUPolygonTriangulation**; rust is **GPUConstrainedDelaunayTriangulation**, which refines an interior mesh by edge flips while respecting the boundary.

The GPU reports polygon validity, triangle counts and whether the flip work converged. A valid mesh can still stop before convergence when **Maximum edge flips** is too small; those are separate facts.`,
      camera: {...COURTYARD, transitionMs: 1500},
      options: {view: 'courtyard', maximumFlips: 8001},
      highlight: {readout: 'courtyardCounts'},
      controls: ['maximumFlips'],
      readouts: ['courtyardSource', 'courtyardStatus', 'courtyardCounts', 'courtyardCountsChart']
    },
    {
      id: 'explore',
      title: 'Choose the construction that fits the question',
      headline: 'A boundary inherits its rule',
      textAlternative:
        'The selected output is accompanied by GPU completion, overflow, validity and retained-count readouts; controls switch among facility-site meshes, radius and ratio hulls, group comparison and the source courtyard. Geometry alone does not establish access or a service boundary.',
      body: `The map now exposes the choices directly: the output geometry, its source points or footprint, and the GPU's counts and status flags. Switch **Geometry** to revisit the constructions and adjust only the parameter meaningful for that method.

The facility extract is an archived set of public locations. The hulls are not walkability, capacity or service areas; the building mesh is not a claim about building use.`,
      camera: {...CHICAGO, transitionMs: 900},
      options: {view: 'groups', alphaRadius: 5000},
      controls: ['view', 'alphaRadius', 'ratio', 'allowHoles', 'maximumFlips'],
      readouts: ['hullParameter', 'hullStatus', 'hullCounts', 'hullCountsChart']
    }
  ],
  readouts: [
    {id: 'librarySites', label: 'Library sites'},
    {id: 'fireSites', label: 'Fire-station sites'},
    {id: 'duplicateCoordinates', label: 'Exact duplicate library sites'},
    {id: 'courtyardSource', label: 'Courtyard source geometry'},
    {id: 'meshStatus', label: 'Delaunay / Voronoi status'},
    {id: 'meshCounts', label: 'Delaunay / Voronoi output'},
    {id: 'voronoiPairs', label: 'Voronoi site pairs'},
    {id: 'meshCountsChart', label: 'Mesh output counts', kind: 'chart'},
    {id: 'hullParameter', label: 'Hull rule'},
    {id: 'hullStatus', label: 'Hull status'},
    {id: 'hullCounts', label: 'Hull output'},
    {id: 'hullCountsChart', label: 'Hull output counts', kind: 'chart'},
    {id: 'courtyardStatus', label: 'Courtyard triangulation status'},
    {id: 'courtyardCounts', label: 'Courtyard triangle counts'},
    {id: 'courtyardCountsChart', label: 'Courtyard output counts', kind: 'chart'}
  ]
});
