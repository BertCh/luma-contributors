// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Buffer} from '@luma.gl/core';
import {
  addClusterAndOutlineRecipe,
  GPUSpatialClustering
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {describe, expect, it, vi} from 'vitest';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING} from '../../src/curated/bindings/curated-clusters-and-footprints-binding';
import {prepareCuratedClustersGpu} from '../../src/curated/bindings/curated-clusters-and-footprints-gpu';
import {
  measurePathLength,
  measureRingSignedArea,
  type CatalogFeatureCollection,
  type CatalogPosition
} from '../../src/analysis/catalog-geojson';
import {getConvexHull} from '../../src/analysis/catalog-platform3-a-shared';
import {readClusterSamples} from '../../src/analysis/catalog-platform3-b-clusters';
import {DBSCAN_GEOGRAPHIC, DBSCAN_PLANAR} from './curated-bindings-oracle-data';
import {
  expectCuratedClusterFootprintMeasures,
  expectCuratedClusterRingAdjacency
} from './curated-bindings-clusters-test-support';
import {
  getNumbers,
  getResultTable,
  makePoints,
  recordBindingMargin,
  runCuratedBinding
} from './curated-bindings-test-support';

// Oracles: /usr/bin/python3 .typecheck/b1-oracles.py. DBSCAN_PLANAR: scikit-learn DBSCAN on 220
// planar points (three blobs and uniform noise; no border point is within reach of two clusters,
// so the partition is unique) and Shapely convex hulls of each cluster. DBSCAN_GEOGRAPHIC: 88
// longitude and latitude points clustered in the azimuthal equidistant frame, with hull area and
// perimeter on a sphere of the library's radius (pyproj laea and Geod).

const ID = 'curated-clusters-and-footprints';

describe('curated clusters GPU preparation and execution contract (no GPU submission)', () => {
  it('packs both coordinate systems in the existing local frame and reserves the complete worst case', () => {
    for (const coordinateSystem of ['planar', 'EPSG:4326'] as const) {
      const points =
        coordinateSystem === 'planar'
          ? makePoints([1e12, 1e12 + 8, 1e12 + 2], [1e12, 1e12, 1e12 + 4])
          : makePoints(DBSCAN_GEOGRAPHIC.longitudes, DBSCAN_GEOGRAPHIC.latitudes);
      const {sample} = readClusterSamples(points, coordinateSystem);
      const prepared = prepareCuratedClustersGpu(sample, 200, 4);
      expect(Array.from(prepared.positions)).toEqual(
        Array.from({length: sample.count * 2}, (_, component) =>
          Math.fround(
            component % 2
              ? sample.y[Math.floor(component / 2)]
              : sample.x[Math.floor(component / 2)]
          )
        )
      );
      expect(prepared.maximumClusterCount).toBe(sample.count);
      expect(prepared.hullCapacity).toBe(sample.count);
      expect(prepared.maximumVerticesPerHull).toBe(Math.max(3, sample.count));
      const [minimumX, minimumY, maximumX, maximumY, epsilon, minimumPoints] =
        prepared.clusteringParameters;
      expect([epsilon, minimumPoints]).toEqual([200, 4]);
      for (let index = 0; index < sample.count; index++) {
        expect(prepared.positions[2 * index]).toBeGreaterThanOrEqual(minimumX);
        expect(prepared.positions[2 * index]).toBeLessThanOrEqual(maximumX);
        expect(prepared.positions[2 * index + 1]).toBeGreaterThanOrEqual(minimumY);
        expect(prepared.positions[2 * index + 1]).toBeLessThanOrEqual(maximumY);
      }
    }
  });

  it('accepts a single repeated location and rejects named limits before allocation', () => {
    const {sample} = readClusterSamples(makePoints([5], [5]), 'planar');
    const prepared = prepareCuratedClustersGpu(sample, 1e-9, 1000);
    expect(Array.from(prepared.clusteringParameters.slice(0, 4))).toEqual([0, 0, 1, 1]);
    expect(prepared.maximumVerticesPerHull).toBe(3);
    for (const count of [0, 6001])
      expect(() => prepareCuratedClustersGpu({...sample, count}, 1, 4)).toThrow(
        /curated-clusters-and-footprints pointCount.*6000/
      );
    expect(() => prepareCuratedClustersGpu(sample, 0, 4)).toThrow(
      /curated-clusters-and-footprints epsilon/
    );
    expect(() => prepareCuratedClustersGpu(sample, 1, 1001)).toThrow(
      /curated-clusters-and-footprints minimumPoints.*1000/
    );
    expect(() =>
      prepareCuratedClustersGpu({...sample, x: new Float64Array([1e100])}, 1, 4)
    ).toThrow(/curated-clusters-and-footprints positions.*float32/);
  });

  it('forwards the optional core-flag view to the recipe clustering contributor', () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const coreFlags = createTransientView(graph, 'core-flags', 'uint32', 3);
      const recipe = addClusterAndOutlineRecipe(graph, {
        points: {kind: 'points', positions: createTransientView(graph, 'points', 'float32x2', 3)},
        clusteringParameters: createTransientView(graph, 'parameters', 'float32', 8),
        geometryParameters: createTransientView(graph, 'geometry-parameters', 'float32', 8),
        gridSize: [2, 2],
        maximumClusterCount: 3,
        maximumVerticesPerHull: 3,
        hullCapacity: 3,
        outputs: {coreFlags}
      });
      const clustering = recipe.contributors.find(
        contributor => contributor instanceof GPUSpatialClustering
      ) as GPUSpatialClustering;
      expect(clustering.props.output.coreFlags).toBe(coreFlags);
    } finally {
      device.destroy();
    }
  });

  for (const failure of [
    'none',
    'cluster-overflow',
    'hull-overflow',
    'invalid-count',
    'readback',
    'spherical-none',
    'spherical-invalid-count',
    'spherical-readback'
  ] as const) {
    it(`publishes GPU readbacks or reports ${failure} with resource cleanup`, async () => {
      const device = createNullWebGPUDevice();
      const buffers: Buffer[] = [];
      const createBuffer = device.createBuffer.bind(device);
      vi.spyOn(device, 'createBuffer').mockImplementation(properties => {
        const buffer = createBuffer(properties);
        buffers.push(buffer);
        vi.spyOn(buffer, 'destroy');
        return buffer;
      });
      vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
      vi.spyOn(CatalogGpuSession.prototype, 'readScalar').mockImplementation(async output => {
        if (
          failure === 'spherical-invalid-count' &&
          output.buffer.id.endsWith('-spherical-measures-invalid-count')
        )
          return 1;
        if (output.buffer.id.endsWith(`-${failure}`)) return 1;
        return output.buffer.id.endsWith('-cluster-count') ? 1 : 0;
      });
      vi.spyOn(CatalogGpuSession.prototype, 'readUint32').mockImplementation(async output => {
        if (failure === 'readback') throw new Error(`${ID}: simulated readback failure`);
        const values = new Uint32Array(output.length);
        if (output.buffer.id.endsWith('-core-flags')) values.fill(1);
        if (output.buffer.id.endsWith('-counts')) values[0] = 3;
        if (output.buffer.id.endsWith('-hull-counts')) values[0] = 3;
        if (output.buffer.id.endsWith('-hull-offsets')) values.set([0, 3, 3, 3]);
        if (output.buffer.id.endsWith('-hull-indices')) values.set([0, 1, 2]);
        return values;
      });
      vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockImplementation(async output => {
        if (failure === 'spherical-readback' && output.buffer.id.includes('-spherical-measures-'))
          throw new Error(`${ID}: simulated spherical readback failure`);
        const values = new Float32Array(output.length * output.components);
        if (output.buffer.id.endsWith('-mean-centers')) values.set([1 / 3, 1 / 3]);
        if (output.buffer.id.endsWith('-areas')) values[0] = 0.5;
        if (output.buffer.id.endsWith('-perimeters')) values[0] = 2 + Math.SQRT2;
        if (output.buffer.id.endsWith('-spherical-measures-areas')) values[0] = 1234;
        return values;
      });
      try {
        const coordinateSystem = failure.startsWith('spherical') ? 'EPSG:4326' : 'planar';
        const parameters = {
          coordinateSystem,
          epsilon: coordinateSystem === 'planar' ? 2 : 200000,
          minimumPoints: 1
        };
        const points = makePoints([0, 1, 0], [0, 0, 1]);
        const result = await runCatalogOperationAsync(
          CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING,
          parameters,
          {points},
          {device}
        );
        if (failure === 'none' || failure === 'spherical-none') {
          expect(result.engine).toBe('gpu');
          expect(result.dispatchedContributors).toEqual(
            CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING.gpuContributorNames
          );
          expect(getResultTable(result, 'points').columns['is_core']).toEqual([true, true, true]);
          expect(getResultTable(result, 'footprints').columns['area']).toEqual([
            failure === 'spherical-none' ? 1234 : 0.5
          ]);
          expect(result.summary['footprintCount']).toBe(1);
        } else {
          expect(result.engine).toBe('cpu-reference');
          expect(result.dispatchedContributors).toEqual([]);
          expect(result.fallbackReason).toMatch(
            failure.includes('readback')
              ? /simulated .*readback failure/
              : failure === 'spherical-invalid-count'
                ? /spherical hull invalidCount must be 0/
                : /recipe clusterOverflow, hullOverflow and invalidCount must be 0/
          );
          expect(result.diagnostics.some(diagnostic => diagnostic.code === 'gpu-fallback')).toBe(
            true
          );
          const reference = runCuratedBinding(ID, parameters, {points});
          expect(getResultTable(result, 'footprints')).toEqual(
            getResultTable(reference, 'footprints')
          );
        }
        expect(CatalogGpuSession.prototype.run).toHaveBeenCalledTimes(
          failure.startsWith('spherical') ? 2 : 1
        );
        expect(buffers.length).toBeGreaterThan(0);
        for (const buffer of buffers) expect(buffer.destroy).toHaveBeenCalledOnce();
      } finally {
        vi.restoreAllMocks();
        device.destroy();
      }
    });
  }
});

describe('GPU center overflow readback regression', () => {
  const finiteCenter = Math.fround(1e38);
  const fixtures = [
    {
      name: 'finite occupied center and NaN empty centers',
      center: [finiteCenter, finiteCenter],
      fallback: false
    },
    ...[Infinity, -Infinity, NaN].flatMap(value => [
      {name: `non-finite x center (${value})`, center: [value, finiteCenter], fallback: true},
      {name: `non-finite y center (${value})`, center: [finiteCenter, value], fallback: true}
    ])
  ];
  for (const fixture of fixtures) {
    it(`handles ${fixture.name} before publication and releases buffers`, async () => {
      const points = makePoints([0, 1e38, 1e38, 1e38, 1e38], [0, 1e38, 1e38, 1e38, 1e38]);
      const {sample} = readClusterSamples(points, 'planar');
      // The reported counterexample is legal at upload; only its GPU sum overflows.
      expect(
        Array.from(prepareCuratedClustersGpu(sample, 1, 4).positions).every(Number.isFinite)
      ).toBe(true);
      expect(Math.fround(finiteCenter * 4)).toBe(Infinity);
      const device = createNullWebGPUDevice();
      const buffers: Buffer[] = [];
      const createBuffer = device.createBuffer.bind(device);
      vi.spyOn(device, 'createBuffer').mockImplementation(properties => {
        const buffer = createBuffer(properties);
        buffers.push(buffer);
        vi.spyOn(buffer, 'destroy');
        return buffer;
      });
      vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {});
      vi.spyOn(CatalogGpuSession.prototype, 'readScalar').mockImplementation(async output =>
        output.buffer.id.endsWith('-cluster-count') ? 1 : 0
      );
      vi.spyOn(CatalogGpuSession.prototype, 'readUint32').mockImplementation(async output => {
        const values = new Uint32Array(output.length);
        if (output.buffer.id.endsWith('-labels')) values[0] = 0xffffffff;
        if (output.buffer.id.endsWith('-core-flags')) values.set([0, 1, 1, 1, 1]);
        if (output.buffer.id.endsWith('-counts')) values[0] = 4;
        if (output.buffer.id.endsWith('-hull-counts')) values[0] = 1;
        if (output.buffer.id.endsWith('-hull-offsets')) values.set([0, 1, 1, 1, 1, 1]);
        if (output.buffer.id.endsWith('-hull-indices')) values[0] = 1;
        return values;
      });
      vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockImplementation(async output => {
        const values = new Float32Array(output.length * output.components);
        if (output.buffer.id.endsWith('-mean-centers')) {
          values.fill(NaN);
          values.set(fixture.center);
        }
        return values;
      });
      try {
        const result = await runCatalogOperationAsync(
          CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING,
          {coordinateSystem: 'planar', epsilon: 1, minimumPoints: 4},
          {points},
          {device}
        );
        expect(result.engine).toBe(fixture.fallback ? 'cpu-reference' : 'gpu');
        const footprints = getResultTable(result, 'footprints');
        expect(footprints.rowCount).toBe(1);
        expect(footprints.columns['point_count']).toEqual([4]);
        expect(footprints.columns['centroid_x']).toEqual([fixture.fallback ? 1e38 : finiteCenter]);
        expect(footprints.columns['centroid_y']).toEqual([fixture.fallback ? 1e38 : finiteCenter]);
        if (fixture.fallback) {
          expect(result.fallbackReason).toMatch(
            /curated-clusters-and-footprints meanCenters\[0\] must be finite for an occupied cluster/
          );
          expect(result.dispatchedContributors).toEqual([]);
          expect(result.diagnostics.some(diagnostic => diagnostic.code === 'gpu-fallback')).toBe(
            true
          );
        } else expect(result.fallbackReason).toBeUndefined();
        expect(buffers.length).toBeGreaterThan(0);
        for (const buffer of buffers) expect(buffer.destroy).toHaveBeenCalledOnce();
      } finally {
        vi.restoreAllMocks();
        device.destroy();
      }
    });
  }
});

describe('headless footprint parity assertion regressions', () => {
  const square: CatalogPosition[] = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]
  ];

  it('accepts every cyclic rotation and reversed winding with unchanged measurements', () => {
    for (const vertices of [square.slice(0, -1), square.slice(0, -1).reverse()]) {
      for (let offset = 0; offset < vertices.length; offset++) {
        const rotated = [...vertices.slice(offset), ...vertices.slice(0, offset)];
        const ring = [...rotated, rotated[0]];
        expectCuratedClusterRingAdjacency(ring, square, 'square permutation');
        expectCuratedClusterFootprintMeasures(
          ring,
          {area: 1, perimeter: 4},
          'planar',
          'square permutation'
        );
      }
    }
  });

  it('rejects a closed bow-tie with exactly the same vertices as the square', () => {
    const bowTie: CatalogPosition[] = [
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ];
    const sortVertices = (ring: CatalogPosition[]) =>
      ring
        .slice(0, -1)
        .map(position => JSON.stringify(position))
        .sort();
    expect(sortVertices(bowTie)).toEqual(sortVertices(square));
    expect(measureRingSignedArea(square, 'planar')).toBe(1);
    expect(measureRingSignedArea(bowTie, 'planar')).toBe(0);
    expect(() => expectCuratedClusterRingAdjacency(bowTie, square, 'corrupted hull')).toThrow(
      /hull ring adjacency/
    );
    expect(() =>
      expectCuratedClusterFootprintMeasures(
        bowTie,
        {area: 1, perimeter: 4},
        'planar',
        'stale measurement buffers'
      )
    ).toThrow(/published geometry area/);
  });

  for (const measurements of [
    {area: 0, perimeter: 4},
    {area: 1, perimeter: 3}
  ]) {
    it(`rejects stale measurement buffers ${JSON.stringify(measurements)} for a valid square`, () => {
      expect(() =>
        expectCuratedClusterFootprintMeasures(square, measurements, 'planar', 'mismatched buffers')
      ).toThrow(/published geometry/);
    });
  }
});

describe('AX-87: differential hull measurement before removing the extra translation', () => {
  it('matches the previous measurements on every scikit-learn fixture cluster', () => {
    const points = getPoints();
    const {sample, frame} = readClusterSamples(points, 'planar');
    const result = runPlanar(points, DBSCAN_PLANAR.epsilon, DBSCAN_PLANAR.minimumPoints);
    const members = getMembers(getNumbers(getResultTable(result, 'points'), 'cluster'));
    let maximumAreaDelta = 0;
    let maximumPerimeterDelta = 0;
    let maximumInverseAreaDelta = 0;
    for (const indices of members.values()) {
      const hull = getConvexHull(indices.map(index => [sample.x[index], sample.y[index]]));
      const frameRing = [...hull, hull[0]];
      const previousRing: CatalogPosition[] = hull.map(([horizontal, vertical]) => [
        horizontal - hull[0][0],
        vertical - hull[0][1]
      ]);
      previousRing.push(previousRing[0]);
      const area = measureRingSignedArea(frameRing, 'planar');
      const perimeter = measurePathLength(frameRing, 'planar');
      const areaDelta = Math.abs(area - measureRingSignedArea(previousRing, 'planar'));
      const perimeterDelta = Math.abs(perimeter - measurePathLength(previousRing, 'planar'));
      maximumAreaDelta = Math.max(maximumAreaDelta, areaDelta);
      maximumPerimeterDelta = Math.max(maximumPerimeterDelta, perimeterDelta);
      maximumInverseAreaDelta = Math.max(
        maximumInverseAreaDelta,
        Math.abs(
          area -
            measureRingSignedArea(
              frameRing.map(([horizontal, vertical]) => frame.inverse(horizontal, vertical)),
              'planar'
            )
        )
      );
      expect(areaDelta).toBe(0);
      expect(perimeterDelta / perimeter).toBeLessThan(1e-14);
    }
    recordBindingMargin('AX-87 fixture hull maximum area delta', maximumAreaDelta);
    recordBindingMargin('AX-87 fixture hull maximum perimeter delta', maximumPerimeterDelta);
    recordBindingMargin('AX-87 fixture inverse-ring maximum area delta', maximumInverseAreaDelta);
    expect(maximumInverseAreaDelta).toBeGreaterThan(0);
  });

  for (const offset of [0, 1e9, 1e12]) {
    it(`preserves area and perimeter at offset ${offset}, including a remote frame origin`, () => {
      // Shapely: Polygon([(0,0),(8,0),(10,4),(2,4)]).area == 32.
      const points = makePoints(
        [offset, offset + 8, offset + 10, offset + 2, 0],
        [offset, offset, offset + 4, offset + 4, -100]
      );
      const {sample} = readClusterSamples(points, 'planar');
      const hull = getConvexHull(
        Array.from({length: 4}, (_, index) => [sample.x[index], sample.y[index]])
      );
      const previousRing: CatalogPosition[] = hull.map(([horizontal, vertical]) => [
        horizontal - hull[0][0],
        vertical - hull[0][1]
      ]);
      previousRing.push(previousRing[0]);
      const candidateRing = [...hull, hull[0]];
      const area = Math.abs(measureRingSignedArea(candidateRing, 'planar'));
      const perimeter = measurePathLength(candidateRing, 'planar');
      const areaDelta = area - Math.abs(measureRingSignedArea(previousRing, 'planar'));
      const perimeterDelta = perimeter - measurePathLength(previousRing, 'planar');
      recordBindingMargin(`AX-87 hull offset ${offset} area delta`, areaDelta);
      recordBindingMargin(`AX-87 hull offset ${offset} perimeter delta`, perimeterDelta);
      expect(areaDelta).toBe(0);
      expect(perimeterDelta).toBe(0);
      expect(area).toBe(32);
      const footprints = getResultTable(runPlanar(points, 12, 3), 'footprints');
      expect(footprints.columns['area']).toEqual([area]);
      expect(footprints.columns['perimeter']).toEqual([perimeter]);
    });
  }

  it('preserves the dateline area while retaining short output geometry edges', () => {
    const points = makePoints([179.9, -179.9, -179.9, 179.9], [40, 40, 40.1, 40.1]);
    const footprints = getResultTable(
      runCuratedBinding(
        ID,
        {coordinateSystem: 'EPSG:4326', epsilon: 30000, minimumPoints: 3},
        {points}
      ),
      'footprints'
    );
    const ring = JSON.parse(String(footprints.columns['geometry'][0]))
      .coordinates[0] as CatalogPosition[];
    const wrappedRing = ring.map(
      ([longitude, latitude]) =>
        [longitude > 180 ? longitude - 360 : longitude, latitude] as CatalogPosition
    );
    const areaDelta =
      measureRingSignedArea(ring, 'EPSG:4326') - measureRingSignedArea(wrappedRing, 'EPSG:4326');
    recordBindingMargin('AX-87 dateline wrapped versus unwrapped area delta', areaDelta);
    expect(areaDelta).toBe(0);
    expect(footprints.columns['area'][0]).toBe(Math.abs(measureRingSignedArea(ring, 'EPSG:4326')));
    expect(
      Math.max(...ring.map(position => position[0])) -
        Math.min(...ring.map(position => position[0]))
    ).toBeLessThan(1);
    expect(ring.some(position => Math.abs(position[0]) > 180)).toBe(true);
  });
});

function getPoints(): CatalogFeatureCollection {
  return makePoints(
    DBSCAN_PLANAR.points.map(point => point[0]),
    DBSCAN_PLANAR.points.map(point => point[1])
  );
}

function runPlanar(points: CatalogFeatureCollection, epsilon: number, minimumPoints: number) {
  return runCuratedBinding(ID, {coordinateSystem: 'planar', epsilon, minimumPoints}, {points});
}

/** Groups point indices by label, in order of the smallest member. */
function getMembers(labels: readonly number[]): Map<number, number[]> {
  const members = new Map<number, number[]>();
  labels.forEach((label, index) => {
    if (label < 0) return;
    members.set(label, [...(members.get(label) ?? []), index]);
  });
  return members;
}

describe('curated-clusters-and-footprints against scikit-learn and Shapely (planar)', () => {
  const result = runPlanar(getPoints(), DBSCAN_PLANAR.epsilon, DBSCAN_PLANAR.minimumPoints);
  const pointTable = getResultTable(result, 'points');
  const footprints = getResultTable(result, 'footprints');

  it('gives the same partition, noise and core flags as scikit-learn, up to cluster numbering', () => {
    const labels = getNumbers(pointTable, 'cluster');
    // Labels agree up to a relabelling exactly when the pairs of points that share a label agree.
    const mapping = new Map<number, number>();
    labels.forEach((label, index) => {
      const expected = DBSCAN_PLANAR.labels[index];
      expect(label < 0).toBe(expected < 0);
      if (expected < 0) return;
      const mapped = mapping.get(expected);
      if (mapped === undefined) mapping.set(expected, label);
      else expect(label).toBe(mapped);
    });
    expect(new Set(mapping.values()).size).toBe(mapping.size);
    expect(pointTable.columns['is_core']).toEqual(DBSCAN_PLANAR.core);
    expect(result.summary['clusterCount']).toBe(DBSCAN_PLANAR.clusters.length);
    expect(result.summary['noiseCount']).toBe(
      DBSCAN_PLANAR.labels.filter(label => label < 0).length
    );
  });

  it('draws the Shapely convex hull of every cluster: area and perimeter to 1e-9, same vertices, same count', () => {
    expect(footprints.rowCount).toBe(DBSCAN_PLANAR.clusters.length);
    const labels = getNumbers(pointTable, 'cluster');
    const members = getMembers(labels);
    const ours = [...members.entries()].map(([cluster, indices]) => ({
      cluster,
      minimum: Math.min(...indices)
    }));
    for (const expected of DBSCAN_PLANAR.clusters) {
      const cluster = ours.find(item => item.minimum === expected.minimumMember)!.cluster;
      const row = (footprints.columns['cluster'] as number[]).indexOf(cluster);
      expect(footprints.columns['point_count'][row]).toBe(expected.pointCount);
      const area = footprints.columns['area'][row] as number;
      const perimeter = footprints.columns['perimeter'][row] as number;
      const areaError = Math.abs(area - expected.area) / expected.area;
      const perimeterError = Math.abs(perimeter - expected.perimeter) / expected.perimeter;
      recordBindingMargin(`footprint area relative error cluster ${cluster}`, areaError);
      recordBindingMargin(`footprint perimeter relative error cluster ${cluster}`, perimeterError);
      expect(areaError).toBeLessThan(1e-9);
      expect(perimeterError).toBeLessThan(1e-9);
      expect(footprints.columns['hull_vertex_count'][row]).toBe(expected.vertices.length);
      const ring = (
        JSON.parse(String(footprints.columns['geometry'][row])).coordinates[0] as number[][]
      ).slice(0, -1);
      for (const vertex of expected.vertices)
        expect(
          ring.some(
            candidate => Math.hypot(candidate[0] - vertex[0], candidate[1] - vertex[1]) < 1e-9
          ),
          `vertex ${vertex}`
        ).toBe(true);
      expect(footprints.columns['has_footprint'][row]).toBe(true);
    }
  });

  it('reports noise in the points table and a diagnostic, and hulls nothing for it', () => {
    const noise = DBSCAN_PLANAR.labels.filter(label => label < 0).length;
    expect(noise).toBeGreaterThan(0);
    expect(result.diagnostics.find(item => item.code === 'noise-points')?.count).toBe(noise);
    const counted = (footprints.columns['point_count'] as number[]).reduce(
      (sum, count) => sum + count,
      0
    );
    expect(counted + noise).toBe(DBSCAN_PLANAR.points.length);
  });

  it('is deterministic', () => {
    expect(runPlanar(getPoints(), DBSCAN_PLANAR.epsilon, DBSCAN_PLANAR.minimumPoints)).toEqual(
      result
    );
  });
});

describe('curated-clusters-and-footprints on longitude and latitude', () => {
  const points = makePoints(DBSCAN_GEOGRAPHIC.longitudes, DBSCAN_GEOGRAPHIC.latitudes);
  const result = runCuratedBinding(
    ID,
    {
      coordinateSystem: 'EPSG:4326',
      epsilon: DBSCAN_GEOGRAPHIC.epsilon,
      minimumPoints: DBSCAN_GEOGRAPHIC.minimumPoints
    },
    {points}
  );

  it('finds the clusters of the sphere oracle and measures hulls in square metres and metres', () => {
    const footprints = getResultTable(result, 'footprints');
    const labels = getNumbers(getResultTable(result, 'points'), 'cluster');
    expect(footprints.rowCount).toBe(DBSCAN_GEOGRAPHIC.clusters.length);
    const members = getMembers(labels);
    for (const expected of DBSCAN_GEOGRAPHIC.clusters) {
      const cluster = [...members.entries()].find(
        ([, indices]) => Math.min(...indices) === expected.minimumMember
      )![0];
      const row = (footprints.columns['cluster'] as number[]).indexOf(cluster);
      expect(footprints.columns['point_count'][row]).toBe(expected.pointCount);
      const areaError =
        Math.abs((footprints.columns['area'][row] as number) - expected.area) / expected.area;
      const perimeterError =
        Math.abs((footprints.columns['perimeter'][row] as number) - expected.perimeter) /
        expected.perimeter;
      recordBindingMargin(`geographic footprint area relative error cluster ${cluster}`, areaError);
      recordBindingMargin(
        `geographic footprint perimeter relative error cluster ${cluster}`,
        perimeterError
      );
      // The hull vertices are chosen in the equidistant frame and the area measured on the sphere
      // (Chamberlain-Duquette); the oracle hulls in an equal-area frame. Over 1 km they agree to 1e-4.
      expect(areaError).toBeLessThan(1e-4);
      expect(perimeterError).toBeLessThan(1e-4);
    }
    expect(Math.abs(Number(footprints.columns['area'][0]))).toBeGreaterThan(1000);
  });
});

describe('curated-clusters-and-footprints declared behaviour', () => {
  it('lists a collinear cluster and a two-point cluster without a polygon, keeps the triangle, and reports noise', () => {
    const points = makePoints([0, 1, 2, 10, 11, 20, 21, 20.5, 50], [0, 0, 0, 0, 0, 0, 0, 1, 50]);
    const result = runPlanar(points, 1.5, 2);
    const footprints = getResultTable(result, 'footprints');
    expect(footprints.rowCount).toBe(3);
    expect(footprints.columns['point_count']).toEqual([3, 2, 3]);
    expect(footprints.columns['has_footprint']).toEqual([false, false, true]);
    expect(footprints.columns['geometry'].slice(0, 2)).toEqual([null, null]);
    expect(footprints.columns['area'].slice(0, 2)).toEqual([null, null]);
    expect(footprints.columns['perimeter'].slice(0, 2)).toEqual([null, null]);
    expect(Math.abs(Number(footprints.columns['area'][2]) - 0.5)).toBeLessThan(1e-12);
    expect(
      Math.abs(Number(footprints.columns['perimeter'][2]) - (1 + 2 * Math.sqrt(1.25)))
    ).toBeLessThan(1e-12);
    expect(result.diagnostics.find(item => item.code === 'degenerate-footprint')?.count).toBe(2);
    expect(result.diagnostics.find(item => item.code === 'noise-points')?.count).toBe(1);
    expect(result.summary['footprintCount']).toBe(1);
    expect(result.summary['degenerateClusterCount']).toBe(2);
    // The centroid of an unhulled cluster is still reported.
    expect(footprints.columns['centroid_x'][0]).toBeCloseTo(1, 12);
    expect(footprints.columns['centroid_x'][1]).toBeCloseTo(10.5, 12);
  });

  it('puts every point in noise, and lists no footprint, when no point has enough neighbours', () => {
    const result = runPlanar(makePoints([0, 10, 20, 30], [0, 0, 0, 0]), 1, 3);
    expect(getResultTable(result, 'footprints').rowCount).toBe(0);
    expect(new Set(getNumbers(getResultTable(result, 'points'), 'cluster'))).toEqual(new Set([-1]));
    expect(result.summary['noiseCount']).toBe(4);
  });

  it('skips features without a point position, counted, and throws a named error when none is left', () => {
    const points = makePoints([0, 0.5, 1, 100], [0, 0, 0, 100]);
    points.features.push({id: 'none', geometry: null, properties: {}});
    const result = runPlanar(points, 2, 3);
    expect(result.diagnostics.find(item => item.code === 'feature-dropped')?.count).toBe(1);
    expect(result.summary['pointCount']).toBe(4);
    expect(() =>
      runPlanar(
        {type: 'FeatureCollection', features: [{id: 'n', geometry: null, properties: {}}]},
        2,
        3
      )
    ).toThrow(/curated-clusters-and-footprints: no feature of the input has a point position/);
  });

  it('treats one repeated point as one cluster of a single location without a polygon', () => {
    const result = runPlanar(makePoints([5, 5, 5, 5], [5, 5, 5, 5]), 1, 3);
    const footprints = getResultTable(result, 'footprints');
    expect(footprints.rowCount).toBe(1);
    expect(footprints.columns['point_count'][0]).toBe(4);
    expect(footprints.columns['has_footprint'][0]).toBe(false);
    expect(footprints.columns['hull_vertex_count'][0]).toBe(1);
  });

  it('rejects polygons and parameters outside their range', () => {
    expect(() =>
      runCuratedBinding(
        ID,
        {coordinateSystem: 'planar'},
        {
          points: {
            type: 'FeatureCollection',
            features: [
              {
                id: 'a',
                geometry: {
                  type: 'Polygon',
                  coordinates: [
                    [
                      [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 0]
                    ]
                  ]
                },
                properties: {}
              }
            ]
          }
        }
      )
    ).toThrow(/geometry kind not accepted/);
    expect(() => runPlanar(makePoints([0], [0]), 0, 3)).toThrow(/out of range/);
    expect(() => runPlanar(makePoints([0], [0]), 1, 0)).toThrow(/out of range/);
  });
});

it('keeps local hull area and perimeter accurate at large planar offsets (Shapely)', () => {
  // /usr/bin/python3: MultiPoint(points + offset).convex_hull.area/.length, Shapely.
  const coordinates = [
    [-3, 1],
    [-2, -4],
    [1, -3],
    [5, 1],
    [2, 4],
    [-1, 3],
    [0, 0],
    [2, 1]
  ];
  for (const offset of [0, 1e6, 1e9, 1e12]) {
    const points = makePoints(
      coordinates.map(point => point[0] + offset),
      coordinates.map(point => point[1] + offset)
    );
    const footprints = getResultTable(runPlanar(points, 20, 1), 'footprints');
    expect(footprints.columns['area']).toEqual([38]);
    expect(Number(footprints.columns['perimeter'][0])).toBeCloseTo(24.151496895287398, 12);
    expect(footprints.columns['has_footprint']).toEqual([true]);
  }
});

it('measures and represents a small dateline hull without a globe-spanning edge (pyproj)', () => {
  // /usr/bin/python3: Geod(a=6371008.8,b=6371008.8).polygon_area_perimeter(
  // [179.9,-179.9,-179.9,179.9],[10,10,10.1,10.1]) -> (243492701.43635178,66034.55280921928).
  const points = makePoints([179.9, -179.9, -179.9, 179.9], [10, 10, 10.1, 10.1]);
  const footprints = getResultTable(
    runCuratedBinding(
      ID,
      {coordinateSystem: 'EPSG:4326', epsilon: 30000, minimumPoints: 1},
      {points}
    ),
    'footprints'
  );
  // Chamberlain-Duquette integrates latitude edges; pyproj uses geodesic edges. At this extent
  // the relative area difference is under 1e-6, while perimeter uses the same great-circle metric.
  expect(Math.abs(Number(footprints.columns['area'][0]) / 243492701.43635178 - 1)).toBeLessThan(
    1e-6
  );
  expect(Number(footprints.columns['perimeter'][0])).toBeCloseTo(66034.55280921928, 6);
  const ring = JSON.parse(String(footprints.columns['geometry'][0])).coordinates[0] as number[][];
  for (let index = 1; index < ring.length; index++)
    expect(Math.abs(ring[index][0] - ring[index - 1][0])).toBeLessThan(1);
  expect(ring.at(-1)).toEqual(ring[0]);
});
