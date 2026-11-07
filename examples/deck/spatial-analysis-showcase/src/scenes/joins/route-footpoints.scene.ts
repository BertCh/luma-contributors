// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {RouteFootpointsOptions} from './route-footpoints.compute';

const RAIL_COLORS = [
  {color: [198, 12, 48, 255], label: 'Red'},
  {color: [0, 161, 222, 255], label: 'Blue'},
  {color: [98, 54, 27, 255], label: 'Brown'},
  {color: [0, 155, 58, 255], label: 'Green'}
] as const;

export default defineScene<RouteFootpointsOptions>({
  id: 'route-footpoints',
  title: 'What is a station’s distance from a route?',
  chapter: 'joins',
  order: 14,
  summary:
    'An investigation of CTA rail stations, clipped GTFS route shapes, and GPU-computed planar footpoints. Geometry distance is not entrance, accessibility, network, or travel distance.',
  contributors: [
    'GPUPairwisePointLinestringNearest',
    'GPUPairwisePointSegmentDistance',
    'GPUPairwisePointDistance',
    'GPUHaversineDistance',
    'GPUPairwisePointInPolygon',
    'GPUPointSpatialQuery'
  ],
  datasets: [
    {id: 'cta-transit', role: 'archived GTFS-derived stations, assigned routes and route shapes'}
  ],
  initialView: {longitude: -87.66, latitude: 41.87, zoom: 10.4},
  basemap: ground('paperCity', {labelPreset: 'streets'}),
  furniture: {
    title: {
      title: 'Route footpoints',
      subtitle: 'Station to archived route geometry · planar metres'
    },
    scaleBar: {units: 'metric'},
    credit: 'Data provided by Chicago Transit Authority · derived from archived CTA GTFS',
    caveat:
      'A geometric offset is not distance to an entrance, access, route-network distance, or a diagnosis of source error.'
  },
  options: [
    {
      kind: 'select',
      id: 'route',
      label: 'Rail route',
      group: 'Source geometry',
      apply: 'compile',
      default: 'Blue',
      help: 'Selects stations assigned to this route and every direction 0 archived route shape part. The route ID is resolved from manifest.properties.routes.',
      options: [
        {value: 'Red', label: 'Red'},
        {value: 'Blue', label: 'Blue'},
        {value: 'Brn', label: 'Brown'},
        {value: 'G', label: 'Green'}
      ]
    },
    {
      kind: 'slider',
      id: 'sample',
      label: 'Selected station row',
      group: 'Station detail',
      apply: 'param',
      min: 0,
      max: 134,
      step: 1,
      default: 0,
      help: 'Selects a zero-based row in the active route subset; out-of-range values normalize to that subset’s final row. The readout gives the active subset size. Click a station on the map to select it.'
    },
    {
      kind: 'select',
      id: 'framing',
      label: 'Evidence framing',
      group: 'Map view',
      apply: 'param',
      default: 'overview',
      help: 'Chooses which current evidence the map follows: overview, selected station plus GPU footpoint, station plus declared route reference, synthetic workframe, or current query support. Selection framing updates after a valid GPU readback.',
      options: [
        {value: 'overview', label: 'Route overview'},
        {value: 'nearest', label: 'Station and GPU footpoint'},
        {value: 'reference', label: 'Station and route reference'},
        {value: 'workframe', label: 'Synthetic workframe'},
        {value: 'query', label: 'Current query support'}
      ]
    },
    {
      kind: 'select',
      id: 'query',
      label: 'Station query predicate',
      group: 'GPU query',
      apply: 'compile',
      default: 'radius',
      help: 'Runs GPUPointSpatialQuery against the station rows. Radius is planar metres; bounds and polygon use the same projected analysis frame.',
      options: [
        {value: 'radius', label: 'Radius'},
        {value: 'bounds', label: 'Bounds'},
        {value: 'polygon', label: 'Synthetic polygon'}
      ]
    },
    {
      kind: 'slider',
      id: 'queryRadius',
      label: 'Query radius',
      group: 'GPU query',
      apply: 'compile',
      min: 100,
      max: 5000,
      step: 100,
      default: 1200,
      unit: 'm',
      help: 'Changes the planar radius parameter consumed by GPUPointSpatialQuery.'
    },
    {
      kind: 'select',
      id: 'haversineUnits',
      label: 'Haversine output units',
      group: 'Distance comparison',
      apply: 'compile',
      default: 'meters',
      help: 'Changes GPUHaversineDistance constructor units and its sphere radius. The readout shows the contributor output in the selected unit and includes a metre comparison.',
      options: [
        {value: 'meters', label: 'Meters'},
        {value: 'kilometers', label: 'Kilometers'}
      ]
    }
  ],
  readouts: [
    {
      id: 'routeCoverage',
      label: 'Route geometry coverage',
      help: 'Counts finite GPU nearest results and no-segment rows. Route shapes are clipped to the city bounding box plus one kilometre.'
    },
    {id: 'stationDetail', label: 'Selected station / nearest result', layout: 'block'},
    {id: 'selectionStatus', label: 'Active route selection'},
    {id: 'segmentDistanceCheck', label: 'GPU distance to returned segment'},
    {id: 'footpointDistanceCheck', label: 'Point distance to returned footpoint'},
    {id: 'offsetDistribution', label: 'Station-to-route offset distribution', kind: 'chart'},
    {id: 'referenceDistance', label: 'Planar distance to route reference'},
    {id: 'haversineDistance', label: 'Great-circle distance to same reference'},
    {id: 'polygonStatus', label: 'Synthetic workframe classification'},
    {id: 'queryStatus', label: 'Station query output'}
  ],
  legends: state => [
    {
      kind: 'categories',
      title: 'CTA rail route accent',
      entries: RAIL_COLORS,
      note: `Active source route: ${state.route}. Route strokes use the manifest route color; pale blue dashes show the active query support centered on the first route station; the gray dashed ±400 m workframe uses the same anchor.`
    },
    {
      kind: 'categories',
      title: 'GPU polygon classification and query matches',
      entries: [
        {color: [133, 143, 150, 220], label: 'Outside source-anchored synthetic workframe'},
        {color: [39, 153, 194, 255], label: 'Inside synthetic workframe'},
        {color: [248, 190, 76, 255], label: 'Workframe boundary'},
        {color: [166, 112, 190, 255], label: 'Classification uncertain'},
        {color: [211, 55, 142, 255], label: 'GPU query match (ring)'},
        {color: [35, 143, 183, 255], label: 'Selected GPU nearest segment / footpoint'},
        {color: [248, 158, 76, 255], label: 'Selected station'}
      ],
      note: 'Station fill is the GPU polygon category. Query rings use compact GPU IDs; rings are hidden when the bounded query overflows. A no-segment row has no fabricated nearest location.'
    }
  ],
  story: [
    {
      id: 'question',
      title: 'What is the distance?',
      headline: 'Start with the route geometry question',
      textAlternative:
        'CTA parent-merged rail stations over a muted archive route map; the selected station is highlighted.',
      body: 'This uses the archived CTA GTFS-derived stop and shape columns. **Rail route** below changes which assigned stations and source geometry are paired.',
      evidence:
        'The sample is archive-derived CTA schedule data and clipped route shapes, not a live service feed.',
      caveat:
        'A track shape omits station entrances, pedestrian paths, vertical circulation and street crossings.',
      controls: ['route'],
      readouts: ['routeCoverage'],
      options: {framing: 'overview'},
      camera: {longitude: -87.66, latitude: 41.87, zoom: 10.4}
    },
    {
      id: 'multipart',
      title: 'Keep every part separate',
      headline: 'Multipart boundaries stay meaningful',
      textAlternative: 'Direction 0 paths remain split at their source part boundaries.',
      body: 'Each station is paired with the route’s source path parts. **Rail route** below changes the path collection. Empty or singleton parts provide no segment; parts are not connected and lines are not implicitly closed.',
      evidence:
        'The source offsets preserve path-part boundaries and GPU output indices identify the winning local part and segment.',
      caveat:
        'The source shapes are clipped to the city bounding box plus one kilometre, so some assigned stations can have no valid segment.',
      controls: ['route'],
      readouts: ['routeCoverage'],
      options: {framing: 'overview'},
      camera: {longitude: -87.66, latitude: 41.87, zoom: 10.4}
    },
    {
      id: 'nearest',
      title: 'Project a perpendicular footpoint',
      headline: 'Nearest means shortest planar segment distance',
      textAlternative:
        'The selected station’s actual GPU nearest location is reported with part and segment indices.',
      body: '`GPUPairwisePointLinestringNearest` searches every aligned multipart feature and returns distance, nearest position, local part index and local segment index. **Selected station row** below changes which output detail is shown.',
      evidence:
        'Nearest positions and distances come from the GPU contributor; invalid rows keep NaN and sentinel indices. Returned indices select the displayed segment, which a second GPU graph checks.',
      caveat:
        'Equal-distance ties keep the first route part and segment. The map uses a local projected metre frame.',
      controls: ['sample', 'framing'],
      options: {framing: 'nearest'},
      readouts: ['stationDetail', 'segmentDistanceCheck', 'footpointDistanceCheck', 'routeCoverage']
    },
    {
      id: 'other-questions',
      title: 'Choose a different geometry question',
      headline: 'A reference point is not the route',
      textAlternative: 'The paired reference point yields distinct planar and spherical distances.',
      body: '`GPUPairwisePointDistance` compares a station to a declared first route vertex reference point. `GPUHaversineDistance` measures the same pair in longitude and latitude. **Haversine output units** below changes the contributor units.',
      evidence:
        'Both readouts are actual contributor outputs. Haversine is displayed in the selected output unit with a metre comparison.',
      caveat:
        'This comparison point is not the nearest endpoint or a station entrance; Chicago-scale differences may be small.',
      controls: ['haversineUnits', 'framing'],
      options: {framing: 'reference'},
      readouts: ['referenceDistance', 'haversineDistance']
    },
    {
      id: 'classification',
      title: 'Classify against a declared polygon',
      headline: 'The workframe polygon is synthetic',
      textAlternative:
        'Each station is classified against a square workframe centered on the first station in the active route subset.',
      body: '`GPUPairwisePointInPolygon` returns inside, boundary, outside or uncertain. The source-anchored ±400 m square is centered on the first station in the active route subset; it is an explicit synthetic analysis frame, not a real service or construction boundary. Station colors show the GPU category.',
      evidence: 'Classification is read from the GPU result for the selected source station.',
      caveat:
        'This source-anchored synthetic polygon is a teaching geometry; it makes no claim about CTA operations.',
      controls: ['framing'],
      options: {framing: 'workframe'},
      readouts: ['polygonStatus']
    },
    {
      id: 'query',
      title: 'Bound the station rows',
      headline: 'A spatial query returns matching row IDs',
      textAlternative:
        'The query readout reports the contributor’s compact count, required count and overflow flag.',
      body: '`GPUPointSpatialQuery` applies a planar radius, bounds or source-anchored synthetic polygon predicate to the actual station rows. Compact GPU row IDs appear as pink rings when the bounded query completes without overflow. **Station query predicate** and **Query radius** below change the contributor variant and parameter.',
      evidence: 'The compact IDs and cardinality status come from the GPU query output.',
      caveat:
        'This is a bounded point query, not an all-pairs join; overflow status remains visible.',
      controls: ['query', 'queryRadius', 'framing'],
      options: {framing: 'query'},
      readouts: ['queryStatus', 'selectionStatus']
    },
    {
      id: 'interpretation',
      title: 'Interpret the offset carefully',
      headline: 'Geometry is only one distance definition',
      textAlternative:
        'The histogram shows GPU nearest-route offsets while the selected detail identifies the source station row.',
      body: 'Use the histogram to see the archived station offsets, then click a station or change **Selected station row** below. Route geometry can answer a map-matching question; it cannot establish door-to-door access.',
      evidence:
        'The distribution is derived from the current GPU nearest-distance rows, with no-segment values excluded.',
      caveat:
        'The CTA feed vintage is September 16, 2026; no conclusion here describes later service.',
      controls: ['sample', 'framing'],
      options: {framing: 'nearest'},
      readouts: ['offsetDistribution', 'stationDetail']
    }
  ],
  snippet: state =>
    `new GPUPairwisePointLinestringNearest({\n  queryPositions: stations,\n  positions: ${state.route}Direction0Parts,\n  featureOffsets, lineOffsets,\n  output: {distances, nearestPositions, lineIndices, segmentIndices},\n  spatialContext: ctaLocalMeters\n});`,
  about: {
    what: 'This scene investigates what geometric distance from a CTA rail station to a route shape means. The archived inputs are CTA GTFS-derived parent-merged rail station positions, assigned route indices, and direction 0 clipped shape parts. Route IDs and official route colors are resolved from manifest.properties.routes.',
    why: 'A route-footpoint offset can support geometry inspection or map matching, but it is not a station entrance distance, accessibility measure, network path, travel time, or evidence of bad source data.',
    howToRead:
      'Colored strokes are archived source paths. Select a station to inspect its GPU distance and local part/segment indices. A bounded readback gathers the returned segment endpoints on the CPU for a second GPU distance check; CPU does not search for the nearest segment. The synthetic workframe classification and query are separate demonstrations. CTA GTFS stop and shape semantics are documented at https://gtfs.org/documentation/schedule/reference/; the source feed is https://www.transitchicago.com/developers/gtfs/. The planar point-to-linestring question follows the family of geospatial functions documented by cuSpatial: https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/.'
  },
  create: async ctx => (await import('./route-footpoints.compute')).createRouteFootpoints(ctx)
});
