// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fetchJson} from '../../data/loaders';
import {toRadians} from '@math.gl/core';

type CountryCollection = {
  features: Array<{
    properties: {NAME?: string; ISO_A3?: string};
    geometry: {type: string; coordinates: number[][][][]};
  }>;
};

export type SphericalBoundarySource = {
  label: string;
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  lonLat: [number, number][];
};

export type SphericalBoundaryInputs = {
  source: SphericalBoundarySource;
  sourceParts: SphericalBoundarySource[];
  probePositions: Float32Array;
  probeLonLat: [number, number][];
  window: SphericalBoundarySource;
};

function unitPosition(longitude: number, latitude: number): [number, number, number] {
  const longitudeRadians = toRadians(longitude);
  const latitudeRadians = toRadians(latitude);
  const cosineLatitude = Math.cos(latitudeRadians);
  return [
    cosineLatitude * Math.cos(longitudeRadians),
    cosineLatitude * Math.sin(longitudeRadians),
    Math.sin(latitudeRadians)
  ];
}

function makePolygon(longitudeRings: number[][][], label: string): SphericalBoundarySource {
  const positions: number[] = [];
  const lonLat: [number, number][] = [];
  const ringOffsets = [0];
  for (const inputRing of longitudeRings) {
    const ring = inputRing.slice();
    if (ring.length > 1 && ring[0][0] === ring.at(-1)![0] && ring[0][1] === ring.at(-1)![1])
      ring.pop();
    const vectors = ring.map(([longitude, latitude]) => unitPosition(longitude, latitude));
    const center = vectors.reduce(
      (sum, vector) => [sum[0] + vector[0], sum[1] + vector[1], sum[2] + vector[2]],
      [0, 0, 0]
    );
    let orientation = 0;
    for (let index = 0; index < vectors.length; index++) {
      const first = vectors[index];
      const second = vectors[(index + 1) % vectors.length];
      orientation +=
        center[0] * (first[1] * second[2] - first[2] * second[1]) +
        center[1] * (first[2] * second[0] - first[0] * second[2]) +
        center[2] * (first[0] * second[1] - first[1] * second[0]);
    }
    const ordered = orientation < 0 ? ring.slice().reverse() : ring;
    for (const [longitude, latitude] of ordered) {
      positions.push(...unitPosition(longitude, latitude));
      lonLat.push([longitude, latitude]);
    }
    ringOffsets.push(positions.length / 3);
  }
  return {
    label,
    positions: Float32Array.from(positions),
    ringOffsets: Uint32Array.from(ringOffsets),
    polygonOffsets: Uint32Array.of(0, ringOffsets.length - 1),
    lonLat
  };
}

function makeProbeGrid(source: SphericalBoundarySource): {
  positions: Float32Array;
  lonLat: [number, number][];
} {
  const longitudes = source.lonLat.map(point => point[0]);
  const latitudes = source.lonLat.map(point => point[1]);
  const west = Math.min(...longitudes);
  const east = Math.max(...longitudes);
  const south = Math.min(...latitudes);
  const north = Math.max(...latitudes);
  const lonLat: [number, number][] = [];
  for (let row = 0; row <= 18; row++) {
    for (let column = 0; column <= 22; column++) {
      lonLat.push([west + ((east - west) * column) / 22, south + ((north - south) * row) / 18]);
    }
  }
  // Exact source vertices make the open/closed boundary rule visible in the same result set.
  lonLat.push(...source.lonLat);
  return {positions: Float32Array.from(lonLat.flatMap(point => unitPosition(...point))), lonLat};
}

/** Loads the bundled, generalized Natural Earth Fiji shell and an explicitly synthetic window. */
export async function loadSphericalBoundaryInputs(
  dataset: {fileUrl: (file: string) => string},
  signal: AbortSignal
): Promise<SphericalBoundaryInputs> {
  const collection = await fetchJson<CountryCollection>(
    dataset.fileUrl('ne_110m_admin_0_countries.geojson'),
    signal
  );
  const fiji = collection.features.find(
    feature => feature.properties.NAME === 'Fiji' && feature.properties.ISO_A3 === 'FJI'
  );
  if (!fiji || fiji.geometry.type !== 'MultiPolygon')
    throw new Error(
      'The bundled Natural Earth country file does not contain Fiji as a MultiPolygon.'
    );
  // Natural Earth stores Fiji as three antimeridian split shells. The middle shell is Viti Levu;
  // this scene retains the published segmentation and does not infer unsplit coastline geometry.
  const partLabels = [
    'western Vanua Levu source shell',
    'Viti Levu source shell',
    'eastern Vanua Levu source shell'
  ];
  const sourceParts = fiji.geometry.coordinates.map((coordinates, index) =>
    makePolygon(coordinates, `Fiji / ${partLabels[index] ?? `source shell ${index + 1}`}`)
  );
  const source = sourceParts[1];
  const window = makePolygon(
    [
      [
        [178.18, -17.65],
        [179.52, -17.65],
        [179.52, -16.2],
        [178.18, -16.2],
        [178.18, -17.65]
      ]
    ],
    'Synthetic analytic window'
  );
  const probes = makeProbeGrid(source);
  return {
    source,
    sourceParts,
    probePositions: probes.positions,
    probeLonLat: probes.lonLat,
    window
  };
}
