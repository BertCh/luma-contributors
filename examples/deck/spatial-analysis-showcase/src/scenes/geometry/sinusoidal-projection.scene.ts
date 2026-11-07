// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {SinusoidalProjectionOptions} from './sinusoidal-projection.compute';

const HUBS = [
  {value: 'ORD', label: "Chicago O'Hare (ORD)"},
  {value: 'HND', label: 'Tokyo Haneda (HND)'},
  {value: 'SYD', label: 'Sydney (SYD)'}
] as const;

/** Coordinate-contract diagnostic for the cuSpatial-compatible sinusoidal operator. */
export default defineScene<SinusoidalProjectionOptions>({
  id: 'sinusoidal-projection',
  title: 'What does this projection actually compute?',
  chapter: 'geometry',
  order: 6,
  summary:
    'GPUSinusoidalProjection transforms all 3,257 frozen OpenFlights airport coordinates around a real hub. Compare its unwrapped and wrapped longitude delta, read actual GPU coordinates for the hub and LAX, and inspect the output extent.',
  contributors: ['GPUSinusoidalProjection', 'GPUColumnProfile'],
  datasets: [
    {id: 'openflights', role: '3,257 airport locations in the frozen OpenFlights snapshot'}
  ],
  initialView: {longitude: 0, latitude: 15, zoom: 0.55},
  basemap: ground('space', {labels: 'none', graticule: true}),
  furniture: {
    title: {
      title: 'Projection contract, not map distance',
      subtitle: 'OpenFlights airports · GPU coordinates in kilometres',
      sample: '3,257 source locations',
      chips: ['coordinate diagnostic', 'not a global distance metric']
    },
    credit: joinCredits(CREDITS.naturalEarth, 'OpenFlights community data'),
    caveat:
      'This cuSpatial-compatible formula uses a 40,000 km sphere and origin-minus-point axes. RAPIDS describes this projection as approximate near the origin, with error growing farther away ([API note](https://docs.rapids.ai/api/libcuspatial/stable/group__spatial__join)). It is neither a global metric-distance projection nor an equal-area claim. Map locations stay in source longitude/latitude.'
  },
  options: [
    {
      kind: 'select',
      id: 'hub',
      label: 'Projection origin',
      group: 'Coordinate formula',
      apply: 'compile',
      default: 'ORD',
      display: 'segmented',
      options: HUBS,
      help: 'The source airport’s longitude and latitude are compile-time origin coordinates for GPUSinusoidalProjection.'
    },
    {
      kind: 'toggle',
      id: 'wrapLongitude',
      label: 'Wrap longitude delta',
      group: 'Coordinate formula',
      apply: 'param',
      default: false,
      help: 'Compare the contributor’s default raw delta with the optional [-180°, 180°) wrap. The projection output is GPU-computed in both variants.'
    },
    {
      kind: 'select',
      id: 'axis',
      label: 'Map colour',
      group: 'Read the coordinates',
      apply: 'param',
      default: 'x',
      display: 'segmented',
      options: [
        {value: 'x', label: 'x · west/east'},
        {value: 'y', label: 'y · south/north'}
      ],
      help: 'Colours each airport by one projected GPU coordinate. Map positions remain the original geographic coordinates.'
    }
  ],
  legends: () => [
    {
      kind: 'ramp',
      id: 'projected-coordinate',
      title: 'Projected coordinate · km',
      ramp: 'diverging',
      midpoint: 0,
      extent: 'gpu',
      unit: 'km',
      format: (value: number) => `${Math.round(value).toLocaleString()} km`,
      note: 'Fixed axis extents: x ±40,000 km; y ±20,000 km.'
    },
    {
      kind: 'categories',
      title: 'Reference airports',
      entries: [
        {color: [182, 133, 255], label: 'Selected origin', shape: 'ring'},
        {color: [255, 157, 53], label: 'LAX probe', shape: 'ring'}
      ],
      note: 'The marker positions are geographic source coordinates.'
    }
  ],
  readouts: [
    {id: 'projectionExtent', label: 'GPU coordinate support', kind: 'value', emphasis: 'tile'},
    {id: 'projectionProbe', label: 'Origin and LAX coordinates', kind: 'value', layout: 'block'}
  ],
  story: [
    {
      id: 'formula',
      title: 'A coordinate transform has a direction',
      headline: 'The formula points back toward its origin',
      textAlternative:
        'All 3,257 OpenFlights airport markers stay at their source longitude and latitude. Their colour is the GPU-computed x coordinate around Chicago O’Hare; the violet origin and orange LAX markers identify the two GPU-probed rows.',
      body: '`GPUSinusoidalProjection` writes x and y in kilometres, following [the RAPIDS cuSpatial projection formula](https://raw.githubusercontent.com/rapidsai/cuspatial/branch-25.04/cpp/include/cuspatial/detail/projection/sinusoidal_projection.cuh). The API describes this style of projection as approximate near the origin, with error increasing farther away ([RAPIDS spatial API](https://docs.rapids.ai/api/libcuspatial/stable/group__spatial__join)). Its x is `(origin longitude − point longitude) × 111.111 × cos(midpoint latitude)`; y is `(origin latitude − point latitude) × 111.111`. Positive x points west and positive y points south, so the axes point from a point toward the selected origin. The map deliberately retains each airport’s source longitude/latitude and uses a projected coordinate only for colour.',
      camera: {bounds: [-180, -56, 180, 80], padding: 20, transitionMs: 1000},
      options: {hub: 'ORD', axis: 'x', wrapLongitude: false},
      controls: ['hub', 'axis'],
      readouts: ['projectionExtent', 'projectionProbe']
    },
    {
      id: 'branch-cut',
      title: 'The antimeridian exposes the wrap choice',
      headline: 'Wrapping changes far-side longitude deltas',
      textAlternative:
        'The same GPU projection is recomputed with longitude wrapping enabled. For the fixed LAX row relative to Tokyo Haneda, the raw x is about 23,572 km and the wrapped x about −9,295 km; airport map positions do not move.',
      body: 'The default leaves `originLongitude − longitude` unwrapped. A target across the antimeridian can therefore receive a longitude delta near 360°, even when the wrapped alternative is nearby. Turn on **Wrap longitude delta** and select **Tokyo Haneda**: the GPU-probed LAX x coordinate changes from about +23,572 km to about −9,295 km. This is a change to the coordinate formula, not a change in airport location or a claim that either global coordinate is a reliable distance.',
      camera: {bounds: [-180, -56, 180, 80], padding: 20, transitionMs: 1000},
      options: {hub: 'HND', axis: 'x', wrapLongitude: true},
      controls: ['hub', 'wrapLongitude'],
      readouts: ['projectionExtent', 'projectionProbe'],
      highlight: {readout: 'projectionProbe'}
    },
    {
      id: 'local-approximation',
      title: 'A local approximation is not a global metric',
      headline: 'The coordinate contract has a limited purpose',
      textAlternative:
        'The x and y output extents and two fixed GPU-probed airports are shown for a real hub. The operator uses 111.111 kilometres per degree on a 40,000-kilometre sphere and is approximate away from its origin; the map is not a global distance or equal-area result.',
      body: 'The contributor follows [cuSpatial’s coordinate formula](https://raw.githubusercontent.com/rapidsai/cuspatial/branch-25.04/cpp/include/cuspatial/detail/projection/sinusoidal_projection.cuh): a 40,000 km sphere, midpoint-latitude cosine for x, and origin-minus-point signs. Its 111.111 km/degree convention differs by about 0.0755% from the 111.195 km/degree mean-Earth convention used by other operators here; that is a sphere convention difference, not a bound on projection error. Its outputs can support a local coordinate diagnostic, but error grows with distance from the origin. Select **Sydney** and switch between x and y to inspect actual GPU output ranges. `GPUColumnProfile` reports the computed coordinate support, and the bounded probe reads the selected origin and LAX outputs directly from the GPU. Neither measure is airport activity, route density, global distance accuracy, or equal-area distortion.',
      camera: {bounds: [-180, -56, 180, 80], padding: 20, transitionMs: 1000},
      options: {hub: 'SYD', axis: 'y', wrapLongitude: true},
      controls: ['hub', 'axis', 'wrapLongitude'],
      readouts: ['projectionExtent', 'projectionProbe']
    }
  ],
  create: async ctx =>
    (await import('./sinusoidal-projection.compute')).createSinusoidalProjection(ctx)
});
