// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {RouteWindowingOptions} from './route-windowing.types';

const ROUTE_VIEW = {longitude: -87.65, latitude: 41.88, zoom: 10.2};

export default defineScene<RouteWindowingOptions>({
  id: 'route-windowing',
  create: async ctx => {
    const compute = await import('./route-windowing.compute');
    return compute.createRouteWindowing(ctx);
  },
  title: 'How does a retained Red Line shape meet the Loop boundary?',
  chapter: 'geometry',
  order: 13,
  summary:
    'Compare the retained CTA Red Line shape’s minimum bounds, two Loop subdivision methods and GPU inside/outside clipping against the City of Chicago community-area boundary.',
  contributors: ['GPUMinimumBounds', 'GPUSubdividePolygons', 'GPULineClipByPolygon'],
  datasets: [
    {id: 'cta-transit', role: 'Red Line source shape'},
    {id: 'chicago-community-areas', role: 'the Loop polygon and place names'}
  ],
  initialView: ROUTE_VIEW,
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'A route and a polygon window',
      subtitle: 'Bounds, subdivision and inside/outside clipping'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cta, CREDITS.cityOfChicago),
    caveat:
      'Geometric intersection with a community-area boundary does not imply transit service coverage.'
  },
  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Analysis view',
      group: 'Geometry',
      apply: 'param',
      default: 'inside',
      options: [
        {value: 'bounds', label: 'Minimum bounds of the route'},
        {value: 'subdivision', label: 'Selected partition of the Loop'},
        {value: 'inside', label: 'Red Line inside the Loop'},
        {value: 'outside', label: 'Red Line outside the Loop'}
      ],
      help: 'All four outputs are computed from the same retained route and boundary; this selection changes the displayed result.'
    },
    {
      kind: 'select',
      id: 'focus',
      label: 'Map extent',
      group: 'Display',
      apply: 'param',
      default: 'route',
      options: [
        {value: 'route', label: 'Retained shape extent'},
        {value: 'loop', label: 'Loop detail'}
      ],
      help: 'Changes the map camera only.'
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Partition method',
      group: 'Subdivision',
      apply: 'param',
      default: 'convex',
      options: [
        {value: 'convex', label: 'Convex pieces'},
        {value: 'rectilinear', label: 'Rectilinear pieces'}
      ],
      help: 'Both bounded GPU partitions are computed from the same Loop polygon; this selects which actual GPU output is read and displayed.'
    }
  ],
  story: [
    {
      id: 'route-extent',
      title: 'Start with the route’s extent',
      headline: 'Bounds summarize extent, not the path.',
      textAlternative:
        'An oriented rectangle and longest-line diameter overlay the retained, city-bounded CTA Red Line shape.',
      body: 'GPUMinimumBounds computes an axis-aligned envelope, minimum-area rotated rectangle and longest vertex pair from the retained CTA route-shape coordinates. These summaries reduce the line to its coordinate extent.',
      evidence:
        'The retained city-bounded source shape has {{sourceVertices}} vertices and a GPU diameter of {{routeDiameter}}. The oriented rectangle spans {{minimumBox}}.',
      caveat:
        'The supplied CTA shape is already clipped to a city bounding area and contains one stored direction. The longest line joins hull vertices; it is not a drivable route length or a transit distance.',
      options: {view: 'bounds', focus: 'route'},
      controls: ['view', 'focus'],
      readouts: ['sourceVertices', 'routeDiameter', 'minimumBox'],
      stage: 'bounds'
    },
    {
      id: 'loop-polygon',
      title: 'Inspect the actual Loop boundary',
      headline: 'The mask is a named community area.',
      textAlternative: 'The Loop polygon is highlighted on the retained Red Line shape.',
      body: 'The clipping mask is Chicago community-area feature 32, named Loop. Its published single exterior ring supplies the polygon boundary for each GPU operation. Measurements are computed in a shared local metric projection before coordinates are converted for map display.',
      evidence:
        'The published Loop ring encloses {{loopSourceArea}} square kilometres in the shared local metric frame.',
      caveat:
        'The published Loop polygon is a roughly 4 m simplified community-area boundary. It is a study-area definition, not a rail right-of-way or station catchment.',
      options: {view: 'inside', focus: 'loop'},
      controls: ['view', 'focus'],
      readouts: ['loopSourceArea'],
      stage: 'window'
    },
    {
      id: 'subdivide',
      title: 'Split the area into bounded pieces',
      headline: 'Subdivision partitions a polygon interior.',
      textAlternative:
        'The Loop interior is partitioned into coloured pieces by the selected convex or rectilinear GPU method.',
      body: 'Choose convex pieces, produced by triangulation and edge-adjacent merging, or rectilinear pieces, produced by bounded axis-aligned subdivision. Both use 16-vertex piece limits and GPU status outputs. The method control selects the matching GPU output. The area summary is calculated from GPU coordinates in the shared local metric frame.',
      evidence:
        '{{subdivisionPieces}} pieces were written of {{subdivisionRequired}} required; output area is {{subdivisionArea}} square kilometres, a difference of {{subdivisionAreaDifference}} from the source-ring CPU reference.',
      caveat:
        'These are computational partitions, not new community boundaries. Piece capacity, invalid polygon or incomplete splitting statuses suppress the fill; read the status before comparing areas.',
      options: {view: 'subdivision', method: 'convex', focus: 'loop'},
      controls: ['view', 'method', 'focus'],
      readouts: [
        'subdivisionPieces',
        'subdivisionArea',
        'subdivisionAreaDifference',
        'subdivisionStatus'
      ],
      stage: 'subdivide'
    },
    {
      id: 'inside-outside',
      title: 'Clip the line on both sides',
      headline: 'The line is split where it meets the polygon.',
      textAlternative:
        'The selected inside or outside fragments of the Red Line are highlighted over the Loop mask.',
      body: 'GPULineClipByPolygon builds sorted crossings and classifies the resulting line pieces. Choose inside to retain portions in or on the boundary, or outside for portions strictly outside it.',
      evidence:
        'Across the two results, the inside line has {{insidePieces}} pieces totalling {{insideLength}} kilometres; the outside line totals {{outsideLength}} kilometres. Inside status: {{insideStatus}}; outside status: {{outsideStatus}}.',
      caveat:
        'These lengths use the shared local planar projection. Uncertain classifications or capacity overflow can make clipped output incomplete, and a boundary clip is not a measure of service.',
      options: {view: 'inside', focus: 'loop'},
      controls: ['view', 'focus'],
      readouts: ['insideLength', 'outsideLength', 'insideStatus', 'outsideStatus'],
      stage: 'clip'
    }
  ],
  readouts: [
    {id: 'sourceVertices', label: 'Red Line vertices', format: 'integer'},
    {
      id: 'routeLength',
      label: 'Stored line length',
      format: 'decimal',
      unit: 'km',
      help: 'Summed from source coordinates in the local metric projection.'
    },
    {
      id: 'routeDiameter',
      label: 'Hull diameter',
      format: 'decimal',
      unit: 'km',
      help: 'GPU longest-line result between hull vertices, not route length.'
    },
    {id: 'routeHullVertices', label: 'Hull vertices', format: 'integer'},
    {id: 'minimumBox', label: 'Minimum rotated rectangle'},
    {
      id: 'loopSourceArea',
      label: 'Loop ring area',
      format: 'decimal',
      unit: 'km²',
      help: 'Planar shoelace area from the published Loop ring in the shared local frame.'
    },
    {id: 'subdivisionPieces', label: 'Subdivision pieces', format: 'integer'},
    {id: 'subdivisionRequired', label: 'Required pieces', format: 'integer'},
    {
      id: 'subdivisionArea',
      label: 'Emitted piece area sum',
      format: 'decimal',
      unit: 'km²',
      help: 'Planar area sum from GPU subdivision vertices read back in the shared local metric frame.'
    },
    {
      id: 'subdivisionAreaDifference',
      label: 'Area difference from source',
      format: 'decimal',
      unit: 'km²',
      help: 'Difference calculated in the shared local metric frame from GPU subdivision coordinates before map conversion.'
    },
    {id: 'subdivisionStatus', label: 'Subdivision status'},
    {id: 'insidePieces', label: 'Inside line pieces', format: 'integer'},
    {id: 'insideLength', label: 'Inside line length', format: 'decimal', unit: 'km'},
    {
      id: 'insideStatus',
      label: 'Inside clip status',
      help: 'GPU overflow, uncertain crossing/classification and candidate overflow.'
    },
    {id: 'outsidePieces', label: 'Outside line pieces', format: 'integer'},
    {id: 'outsideLength', label: 'Outside line length', format: 'decimal', unit: 'km'},
    {
      id: 'outsideStatus',
      label: 'Outside clip status',
      help: 'GPU overflow, uncertain crossing/classification and candidate overflow.'
    }
  ],
  about: {
    what: 'GPU bounds, subdivision and line clipping operate on the retained CTA Red route shape and the Loop community-area ring. Distance and area summaries use the shared local metric projection before coordinates are converted for map display.',
    why: 'A polygon boundary can divide a line geometrically. These operations show how bounds, partitions and clipping represent that relationship.',
    howToRead:
      'The source is CTA Red route shape row 16958, a retained 96-vertex directional shape already clipped to the prepared city bounding area. The boundary is Chicago community-area feature 32 (Loop), published as a roughly 4 m simplified ring. Clipped length is geometric intersection and does not establish service coverage. Place labels mark the study area.'
  },
  legends: state => [
    {
      kind: 'categories',
      title: 'Geometry',
      entries: [
        {color: [145, 162, 173, 190] as const, label: 'Stored CTA Red Line'},
        {color: [53, 98, 121, 75] as const, label: 'Chicago Loop boundary'},
        {
          color: [248, 170, 87, 245] as const,
          label:
            state.view === 'outside'
              ? 'Outside clip'
              : state.view === 'inside'
                ? 'Inside clip'
                : 'GPU result'
        }
      ]
    }
  ],
  snippet: state =>
    state.view === 'bounds'
      ? `graph.add(new GPUMinimumBounds({positions, labels, groupCount: 1,
  maximumHullVertices: 256, output: {rectangleCorners, longestLines, diameters, bounds, hullSizes, status: {overflow}}}));`
      : state.view === 'subdivision'
        ? `graph.add(new GPUSubdividePolygons({positions, ringOffsets, polygonOffsets, featureOffsets,
  method: '${state.method}', maximumVertices: 16, output: {positions: pieces, pieces: pieceMetadata, status}}));`
        : `graph.add(new GPULineClipByPolygon({
  lines: {kind: 'lines', positions, lineOffsets}, polygons: loopPolygon,
  mode: '${state.view}', intersectionCapacity: 256, candidateCapacity: 2048,
  output: {pieces: {geometry: {positions: clippedPositions, lineOffsets: clippedOffsets}, sourceIds, status}}
}));`
});
