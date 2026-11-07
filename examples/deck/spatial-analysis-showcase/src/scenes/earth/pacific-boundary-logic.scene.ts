// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {PacificBoundaryLogicOptions} from './pacific-boundary-logic.compute';

const WINDOW_A = 'corners 145°E / 150°W · 15°S / 55°N';
const WINDOW_B = 'corners 170°E / 115°W · 35°S / 30°N';

export default defineScene<PacificBoundaryLogicOptions>({
  id: 'pacific-boundary-logic',
  title: 'Boundaries across the Pacific',
  chapter: 'earth',
  order: 17,
  summary:
    'Use two analyst-defined Pacific windows to compare convex spherical containment, intersection and intrinsic arc Boolean geometry without a longitude seam.',
  contributors: [
    'GPUSphericalConvexContains',
    'GPUSphericalConvexIntersection',
    'GPUSphericalArcOverlay'
  ],
  datasets: [
    {id: 'poopdeck-earthquakes', role: 'Archived USGS ComCat event snapshot'},
    {id: 'natural-earth', role: 'Natural Earth 1:110m coastline orientation'}
  ],
  initialView: {longitude: 180, latitude: 8, zoom: 2.6, pitch: 0, bearing: 0},
  basemap: ground('abyss', {labels: 'none', graticule: {stepDegrees: 20, opacity: 0.28}}),
  furniture: {
    title: {
      title: 'Boundaries across the Pacific',
      subtitle: 'spherical regions · no longitude seam'
    },
    credit: 'USGS ComCat via poopdeck.gl · Natural Earth 1:110m coastlines',
    caveat:
      'Both colored windows are synthetic analyst-defined examples, not tectonic, hazard, or warning boundaries. Archived ComCat selection ends 2024-12-30 and is not a complete global catalog.'
  },
  options: [
    {
      kind: 'select',
      id: 'analysis',
      label: 'GPU investigation',
      group: 'Method',
      apply: 'param',
      default: 'contains',
      options: [
        {value: 'contains', label: 'Convex containment'},
        {value: 'convex', label: 'Convex intersection'},
        {value: 'arc', label: 'Intrinsic arc overlay'}
      ],
      help: 'Selects one family of GPU-computed boundary evidence. Each family has distinct region limits.'
    },
    {
      kind: 'select',
      id: 'window',
      label: 'Containment window',
      group: 'Convex containment',
      apply: 'param',
      default: 'A',
      disabledWhen: state => state.analysis !== 'contains',
      options: [
        {value: 'A', label: `A · ${WINDOW_A}`},
        {value: 'B', label: `B · ${WINDOW_B}`}
      ],
      help: 'Runs every type-0 archived earthquake against the selected four-edge spherical window. Listed latitudes describe corners; great-circle edges bow away from latitude parallels.'
    },
    {
      kind: 'select',
      id: 'boundary',
      label: 'Boundary rule',
      group: 'Convex containment',
      apply: 'param',
      default: 'closed',
      disabledWhen: state => state.analysis !== 'contains',
      options: [
        {value: 'closed', label: 'Closed · include boundary band'},
        {value: 'open', label: 'Open · exclude boundary band'}
      ],
      help: 'Exact source-vertex probes are appended separately; they expose status 2 and never enter earthquake totals.'
    },
    {
      kind: 'toggle',
      id: 'complement',
      label: 'Complement selected region',
      group: 'Arc arrangement',
      apply: 'param',
      default: false,
      disabledWhen: state => state.analysis === 'convex',
      help: 'For containment this complements the selected window. For arc overlay it sets A reference membership at 0°E, 0°N. Convex intersection always uses ordinary A and B.'
    },
    {
      kind: 'select',
      id: 'operation',
      label: 'Boolean operation',
      group: 'Arc arrangement',
      apply: 'param',
      default: 'intersection',
      disabledWhen: state => state.analysis !== 'arc',
      options: [
        {value: 'intersection', label: 'Intersection'},
        {value: 'union', label: 'Union'},
        {value: 'difference', label: 'A minus B'},
        {value: 'symmetric-difference', label: 'Symmetric difference'}
      ],
      help: 'Selects the GPU-assembled boundary and a CPU Boolean combination of the two GPU containment arrays for archive-event membership.'
    },
    {
      kind: 'select',
      id: 'capacity',
      label: 'Intersection vertex capacity',
      group: 'Convex intersection',
      apply: 'param',
      default: '16',
      disabledWhen: state => state.analysis !== 'convex',
      options: [
        {value: '3', label: '3 · bounded control'},
        {value: '16', label: '16 · full result'}
      ],
      help: 'The bounded control reports the required vertex count. Geometry and measures are withheld when status is nonzero.'
    }
  ],
  story: [
    {
      id: 'two-windows',
      title: 'Two boundaries cross the date line',
      headline: 'A longitude wrap is a map convention',
      textAlternative:
        'Natural Earth coastlines orient the Pacific map. Cyan window A and coral window B are four-edge analyst examples whose source corners cross ±180°. Muted earthquake marks show every type-0 archive row.',
      body: 'The windows are defined by four source corners each and joined by minor great-circle arcs on a unit sphere. Their longitudes cross the conventional ±180° map cut, but that cut is not part of either boundary. Coordinates are checked on the sphere before they enter the graph: orientation comes from the signed unit-vector cross products around the normalized vertex center, and both rings fit strictly inside an open hemisphere.',
      evidence:
        '**{{archiveEvidence}}** The display uses a fixed 1:1 event mark for every earthquake row; calculations also use every row.',
      caveat:
        'This archived ComCat snapshot includes only the exported M4+ selection, ends on 2024-12-30, and must not be read as a complete global catalog. Windows A and B are hypothetical analysis boundaries, not hazard, tectonic, or warning polygons.',
      options: {analysis: 'contains', window: 'A'},
      controls: ['analysis'],
      camera: {longitude: 180, latitude: 8, zoom: 2.6, transitionMs: 900}
    },
    {
      id: 'convex-membership',
      title: 'Membership is tested against spherical edge planes',
      headline: 'The GPU classifies unit vectors, not map pixels',
      textAlternative:
        'The selected cyan or coral window remains visible while brighter dots show archive earthquakes accepted by GPUSphericalConvexContains; exact corner probes are counted in a separate boundary readout.',
      body: '`GPUSphericalConvexContains` evaluates each float32 source longitude/latitude after conversion to a float32 unit XYZ direction. Its oriented great-circle edge planes define the convex interior. The open/closed rule only changes points in the angular tolerance band; four exact corners are extra test queries, excluded from all earthquake counts.',
      evidence:
        '**{{acceptedEvents}}** archived earthquakes are accepted; **{{boundaryEvents}}** source rows fall in the boundary band and **{{invalidEvents}}** report invalid status. {{probeStatus}}.',
      caveat:
        'The tolerance is 1e-6 in unit-vector coordinates. It is a numerical angular band, not a ground-distance or source-boundary accuracy claim.',
      options: {analysis: 'contains', window: 'A', boundary: 'closed', complement: false},
      controls: ['window', 'boundary', 'complement'],
      readouts: ['acceptedEvents', 'boundaryEvents', 'invalidEvents', 'probeStatus'],
      camera: {longitude: 180, latitude: 8, zoom: 2.6, transitionMs: 900}
    },
    {
      id: 'convex-overlap',
      title: 'Convex intersection returns its own vertices',
      headline: 'The overlap geometry comes back from the GPU',
      textAlternative:
        'The differently colored synthetic windows frame a vivid teal geodesic outline from GPUSphericalConvexIntersection. Brighter dots show archive events accepted by a CPU intersection of the two GPU containment arrays; a valid empty geometry result is labelled rather than drawn as a false polygon.',
      body: '`GPUSphericalConvexIntersection` compares A and B through their oriented 3D great-circle planes and writes a bounded list of unit-vector vertices. The output drives the teal line directly after readback; map coordinates are only produced for display by spherical interpolation. Optional GPU measures report intersection steradians, source and target fractions, and union steradians.',
      evidence:
        '**{{intersectionVertices}}** · **{{intersectionArea}}** · **{{intersectionFractions}}** · **{{combinedEvents}}** · {{computeStatus}}.',
      caveat:
        'This contributor requires each input to be convex and contained in an open hemisphere. The results do not support complementary or otherwise nonconvex regions; use the separate arc arrangement for those.',
      options: {analysis: 'convex', capacity: '16'},
      controls: ['capacity'],
      readouts: ['intersectionVertices', 'intersectionArea', 'combinedEvents', 'computeStatus'],
      camera: {longitude: 180, latitude: 8, zoom: 2.6, transitionMs: 900}
    },
    {
      id: 'bounded-capacity',
      title: 'A bounded result reports what it needed',
      headline: 'Capacity failure withholds partial geometry',
      textAlternative:
        'The capacity selector switches between a three-vertex control and a sixteen-vertex result. Readouts show written and required counts and state when the GPU status prevents drawing or trusting measures.',
      body: 'The capacity-3 graph and capacity-16 graph both execute against the same boundaries. A status of 2 means the required output exceeded the selected capacity. The readout keeps the required count visible, while the map suppresses the incomplete ring and its measures. An empty or boundary-contact result is valid when status is zero.',
      evidence:
        '**{{intersectionVertices}}** · **{{intersectionArea}}** · {{computeStatus}}. A nonzero status is never converted to zero area.',
      caveat:
        'The small capacity is a bounded-output demonstration, not an alternate geometric definition. Only a successful full output is rendered as intersection geometry.',
      options: {analysis: 'convex', capacity: '3'},
      controls: ['capacity'],
      readouts: ['intersectionVertices', 'intersectionArea', 'computeStatus'],
      camera: {longitude: 180, latitude: 8, zoom: 2.6, transitionMs: 900}
    },
    {
      id: 'intrinsic-arcs',
      title: 'Arc arrangements have no hemisphere cap',
      headline: 'Parity plus an anchor determines the region',
      textAlternative:
        'The vivid selected Boolean boundary is read back from GPUSphericalArcOverlay. Text marks the reference point at 0°E, 0°N and states whether the chosen A/B operation contains it.',
      body: '`GPUSphericalArcOverlay` intersects and assembles the same input arcs on the sphere, with no longitude projection or convexity assumption. Its parity membership is anchored at the known reference point: ordinary A and B both exclude 0°E, 0°N; complementary A includes it. The shown lines are densified from the returned unit-vector rings along great circles.',
      evidence:
        '**{{arcCounts}}** · {{arcReference}} · **{{combinedEvents}}** · {{computeStatus}}.',
      caveat:
        'Archive membership here is an explicit CPU Boolean combination of the two GPUSphericalConvexContains result arrays; it is not claimed to be GPUSphericalArcOverlay point classification. Complementary A covers more than a hemisphere, so the map may need a world view to show its region interpretation.',
      options: {analysis: 'arc', operation: 'intersection', complement: false},
      controls: ['operation', 'complement'],
      readouts: ['arcCounts', 'arcReference', 'combinedEvents', 'computeStatus'],
      camera: {longitude: 180, latitude: 8, zoom: 2.6, transitionMs: 900}
    },
    {
      id: 'four-boolean-rules',
      title: 'Four operations share the same source arcs',
      headline: 'Changing a rule changes the assembled boundary',
      textAlternative:
        'The selected intersection, union, A-minus-B, or symmetric-difference result is outlined in a vivid color; the cyan A and coral B input boundaries remain distinct and unchanged.',
      body: 'The four results are computed from identical A and B source rings. Change the operation to see a different assembled output and a corresponding archive-event membership count from the two GPU containment arrays. Toggle complementary A to examine a region larger than a hemisphere; its boundary ring is still the same source ring, while its inside/outside interpretation changes.',
      evidence:
        '**{{arcCounts}}** · {{arcReference}} · **{{combinedEvents}}** · {{computeStatus}}.',
      caveat:
        'These are analyst-defined geometry experiments over an archived event selection. They do not imply physical plate edges, seismic hazard, or event completeness. Numerical arc topology may return invalid, capacity, or unresolved status; nonzero output is withheld.',
      options: {analysis: 'arc', operation: 'symmetric-difference', complement: true},
      controls: ['operation', 'complement'],
      readouts: ['arcCounts', 'arcReference', 'combinedEvents', 'computeStatus'],
      camera: {longitude: 0, latitude: 10, zoom: 0.55, transitionMs: 1100}
    }
  ],
  legends: state => [
    {
      id: 'pacific-boundary-evidence',
      title: 'Boundary evidence',
      kind: 'categories',
      entries: [
        {color: [61, 213, 232, 230], label: 'synthetic window A'},
        {color: [238, 123, 81, 225], label: 'synthetic window B'},
        {color: [103, 123, 131, 85], label: 'all archived type-0 earthquakes'},
        {
          color:
            state.analysis === 'convex'
              ? [76, 235, 196, 255]
              : state.complement
                ? [231, 90, 222, 255]
                : [255, 189, 74, 255],
          label:
            state.analysis === 'convex'
              ? 'A ∩ B events / valid intersection boundary'
              : state.complement
                ? 'Complementary events / valid boundary'
                : 'Accepted events / valid arc boundary'
        }
      ],
      note: 'Bright event marks use GPU containment or the stated Boolean combination of its arrays. Output boundaries require complete status; event membership can remain visible when a bounded boundary output is withheld.'
    }
  ],
  readouts: [
    {id: 'archiveEvidence', label: 'Source rows and analysis extent', layout: 'block'},
    {
      id: 'acceptedEvents',
      label: 'Accepted archived earthquakes',
      unit: 'events',
      emphasis: 'tile'
    },
    {id: 'boundaryEvents', label: 'Archive points in boundary band', unit: 'events'},
    {id: 'invalidEvents', label: 'Invalid archive point statuses', unit: 'events'},
    {id: 'probeStatus', label: 'Separate synthetic corner probes', layout: 'block'},
    {id: 'intersectionVertices', label: 'GPU intersection vertices', layout: 'block'},
    {id: 'intersectionArea', label: 'Intersection spherical area', emphasis: 'tile'},
    {id: 'intersectionFractions', label: 'Input fractions and union area', layout: 'block'},
    {id: 'arcCounts', label: 'GPU arc output counts', layout: 'block'},
    {id: 'arcReference', label: 'Reference membership at 0°E, 0°N', layout: 'block'},
    {id: 'combinedEvents', label: 'Archive events by A/B Boolean membership', layout: 'block'},
    {id: 'computeStatus', label: 'Selected GPU result status', layout: 'block', hood: true}
  ],
  snippet: state =>
    state.analysis === 'contains'
      ? `graph.add(new GPUSphericalConvexContains({boundary: '${state.boundary}', complement: ${state.complement}, tolerance: 1e-6, positions: unitXYZ, offsets, queries: archiveUnitXYZ, polygonIndices, output}));`
      : state.analysis === 'convex'
        ? `graph.add(new GPUSphericalConvexIntersection({capacity: ${state.capacity}, tolerance: 1e-6, firstPositions: windowAUnitXYZ, secondPositions: windowBUnitXYZ, firstOffsets, secondOffsets, output: {positions, counts, status, measures}}));`
        : `graph.add(new GPUSphericalArcOverlay({operation: '${state.operation}', left: {positions: windowAUnitXYZ, ringOffsets: offsetsA, referenceInside: ${state.complement}}, right: {positions: windowBUnitXYZ, ringOffsets: offsetsB, referenceInside: false}, referencePoint: [1, 0, 0], output}));`,
  about: {
    what: '`GPUSphericalConvexContains` classifies unit-vector queries against convex spherical half-spaces. `GPUSphericalConvexIntersection` constructs a bounded intersection and optional spherical measures. `GPUSphericalArcOverlay` assembles spherical Boolean boundaries using anchored parity, including complementary non-hemisphere regions.',
    why: 'Longitude is a coordinate convention with a discontinuity at ±180°. The paired Pacific windows make the map cut visible while the GPU operations work directly with unit vectors and great-circle arcs.',
    howToRead:
      'Cyan and coral outlines are synthetic source rings; muted marks are all archived type-0 ComCat events. Brighter marks use GPU containment outputs or the explicitly stated CPU Boolean combination of GPU arrays. The teal intersection and selected arc result are actual GPU readback geometry densified along sphere arcs for display. Read status with counts; nonzero status withholds partial geometry and optional measures. Projection is only for map display, while all tests and areas use unit XYZ sphere geometry.'
  },
  create: async ctx =>
    (await import('./pacific-boundary-logic.compute')).createPacificBoundaryLogic(ctx)
});
