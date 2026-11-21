// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUDatumGridTransform,
  type GPUDatumGridTransformProps
} from '../../../src/geospatial-analysis/datum-grid-transform/index';
import areaReference from './fixtures/geodesic-area-adaptive.json';
import difficultReference from './fixtures/difficult-geodesics.json';
import datumReference from './fixtures/datum-horizontal.json';
import arcReference from './fixtures/spherical-arc-overlay.json';
import {GPUSphericalArcOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay';
import coverReference from './fixtures/s2-polygon-covering.json';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import overlayReference from './fixtures/spherical-polygon-overlay.json';
import {GPUSphericalPolygonOverlay} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-overlay';
import polygonReference from './fixtures/spherical-polygon-contains.json';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import fixtureReference from './fixtures/global-coordinates.json';
import {GPUS2CellMeasures} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-measures';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {GPUSphericalConvexContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-contains';
import {GPUS2CellHierarchy} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cell-hierarchy';
import {GPUSphericalConvexIntersection} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

function makeFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2'>(
    format: Format,
    values: number[]
  ) => {
    const components = format.endsWith('x4')
      ? 4
      : format.endsWith('x3')
        ? 3
        : format.endsWith('x2')
          ? 2
          : 1;
    const buffer = createInputBuffer(
      device,
      format.startsWith('float') ? new Float32Array(values) : new Uint32Array(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      format,
      values.length / components
    );
  };
  const output = <
    Format extends 'float32' | 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2'
  >(
    format: Format,
    length: number
  ) => {
    const components = format.endsWith('x4')
      ? 4
      : format.endsWith('x3')
        ? 3
        : format.endsWith('x2')
          ? 2
          : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, length)
    };
  };
  return {graph, input, output, destroy: () => buffers.forEach(buffer => buffer.destroy())};
}

async function runDatum(
  device: Device,
  positions: number[],
  options: {
    samples?: number[];
    validity?: number[];
    direction?: 'inverse';
    operation?: GPUDatumGridTransformProps['operation'];
    iterations?: number;
    heightUnitMeters?: number;
  } = {}
) {
  const fixture = makeFixture(device);
  const result = fixture.output('float32x3', positions.length / 3);
  const status = fixture.output('uint32', positions.length / 3);
  fixture.graph.add(
    new GPUDatumGridTransform({
      positions: fixture.input('float32x3', positions),
      samples: fixture.input(
        'float32x3',
        options.samples ?? [0.1, 0.2, -10, 0.2, 0.2, -20, 0.1, 0.3, -30, 0.2, 0.3, -40]
      ),
      mask: fixture.input('uint32', options.validity ?? [1, 1, 1, 1]),
      grids: [{width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]}],
      operation: options.operation ?? 'combined',
      direction: options.direction,
      maximumIterations: options.iterations,
      heightUnitMeters: options.heightUnitMeters,
      sourceVerticalReference: 'ellipsoid',
      targetVerticalReference: 'orthometric',
      output: {positions: result.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      positions: await readFloat32(result.buffer, positions.length),
      status: await readUint32(status.buffer, positions.length / 3)
    };
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}

it('datum grids interpolate affine fields, invert, reject nodata/coverage and finite arithmetic overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const forward = await runDatum(device, [0.5, 0.5, 100, 2, 2, 100, NaN, 0, 0]);
  expect(forward.status).toEqual([0, 2, 1]);
  expect(forward.positions[0]).toBeCloseTo(0.65, 5);
  expect(forward.positions[1]).toBeCloseTo(0.75, 5);
  expect(forward.positions[2]).toBeCloseTo(75, 5);
  expect(forward.positions[3]).toBeNaN();
  const inverse = await runDatum(device, forward.positions.slice(0, 3), {direction: 'inverse'});
  expect(inverse.status).toEqual([0]);
  expect(inverse.positions[0]).toBeCloseTo(0.5, 4);
  expect(inverse.positions[1]).toBeCloseTo(0.5, 4);
  expect(inverse.positions[2]).toBeCloseTo(100, 3);
  expect((await runDatum(device, [0.5, 0.5, 0], {validity: [1, 0, 1, 1]})).status).toEqual([4]);
  expect(
    (await runDatum(device, [0.65, 0.75, 75], {direction: 'inverse', iterations: 1})).status
  ).toEqual([8]);
  expect(
    (
      await runDatum(device, [0.5, 0.5, 3e38], {
        samples: Array(4).fill([0, 0, 3e38]).flat(),
        operation: 'vertical'
      })
    ).status
  ).toEqual([1]);
  expect(
    (await runDatum(device, [0.5, 0.5, 0], {samples: Array(4).fill([0, 100, 0]).flat()})).status
  ).toEqual([1]);
  expect((await runDatum(device, [1.2, 0.5, 0], {direction: 'inverse'})).status).toEqual([2]);
});

it('S2 integer hierarchy matches independent BigInt parent/children/range identities through level 30', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const keys = [
    0x1000000000000000n,
    0xb000000000000000n,
    0x89c25a3d00000000n,
    0x89c25a3d12345679n,
    0n,
    2n
  ];
  for (const operation of ['parent', 'children', 'range'] as const) {
    const fixture = makeFixture(device);
    const stride = operation === 'children' ? 4 : operation === 'range' ? 2 : 1;
    const output = fixture.output('uint32x2', keys.length * stride);
    const status = fixture.output('uint32', keys.length);
    fixture.graph.add(
      new GPUS2CellHierarchy({
        cells: fixture.input(
          'uint32x2',
          keys.flatMap(key => [Number(key & 0xffffffffn), Number(key >> 32n)])
        ),
        operation,
        level: 0,
        output: {cells: output.view, status: status.view}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const words = await readUint32(output.buffer, keys.length * stride * 2);
      const statuses = await readUint32(status.buffer, keys.length);
      for (let row = 0; row < keys.length; row++) {
        const key = keys[row];
        const marker = key & -key;
        const valid = row < 4;
        const unavailable = operation === 'children' && marker === 1n;
        expect(statuses[row]).toBe(!valid ? 1 : unavailable ? 2 : 0);
        for (let slot = 0; slot < stride; slot++) {
          const expected =
            !valid || unavailable
              ? 0n
              : operation === 'parent'
                ? (key & ~((1n << 61n) - 1n)) | (1n << 60n)
                : operation === 'children'
                  ? key - marker + marker / 4n + (BigInt(slot) * marker) / 2n
                  : key + (slot ? marker - 1n : 1n - marker);
          const offset = (row * stride + slot) * 2;
          expect((BigInt(words[offset + 1]) << 32n) | BigInt(words[offset])).toBe(expected);
        }
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('convex spherical intersection preserves the polar octant, reports capacities and rejects antipodal edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [2, 6]) {
    const fixture = makeFixture(device);
    const positions = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const output = fixture.output('float32x3', capacity * 2);
    const counts = fixture.output('uint32x2', 2);
    const status = fixture.output('uint32', 2);
    const measures = fixture.output('float32x4', 2);
    fixture.graph.add(
      new GPUSphericalConvexIntersection({
        firstPositions: fixture.input('float32x3', [...positions, 1, 0, 0, -1, 0, 0, 0, 0, 1]),
        secondPositions: fixture.input('float32x3', [...positions, ...positions]),
        firstOffsets: fixture.input('uint32', [0, 3, 6]),
        secondOffsets: fixture.input('uint32', [0, 3, 6]),
        capacity,
        output: {
          positions: output.view,
          counts: counts.view,
          status: status.view,
          measures: measures.view
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status.buffer, 2)).toEqual([capacity === 2 ? 2 : 0, 1]);
      expect(await readUint32(counts.buffer, 4)).toEqual([Math.min(capacity, 3), 3, 0, 0]);
      const overlap = await readFloat32(measures.buffer, 8);
      expect(overlap[4]).toBeNaN();
      if (capacity === 2) expect(overlap[0]).toBeNaN();
      if (capacity === 6) {
        expect(overlap[0]).toBeCloseTo(Math.PI / 2, 5);
        expect(overlap[1]).toBeCloseTo(1, 5);
        expect(overlap[2]).toBeCloseTo(1, 5);
        const vertices = await readFloat32(output.buffer, 9);
        expect(vertices.filter(value => Math.abs(value - 1) < 1e-6)).toHaveLength(3);
        expect(vertices.filter(value => Math.abs(value) < 1e-6)).toHaveLength(6);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('ellipsoidal measures expose unsupported near-antipodal edges in serial and cooperative paths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const cooperativeRingRows of [0, 1]) {
    const fixture = makeFixture(device);
    const lengths = fixture.output('float32', 1);
    const invalid = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUGeometryMeasures({
        geometryType: 'lines',
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          units: 'meters'
        },
        positions: fixture.input('float32x2', [0, 0, 179.7, 0.2]),
        lineOffsets: fixture.input('uint32', [0, 2]),
        cooperativeRingRows,
        output: {lengths: lengths.view, status: {invalidCount: invalid.view}}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect((await readFloat32(lengths.buffer, 1))[0]).toBeNaN();
      expect(await readUint32(invalid.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('S2 neighbors cross all low-level faces and retain exact high-level IDs against pinned s2sphere', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeFixture(device);
  const rows = fixtureReference.cells;
  const output = fixture.output('uint32x2', rows.length * 4);
  const status = fixture.output('uint32', rows.length);
  fixture.graph.add(
    new GPUS2CellHierarchy({
      cells: fixture.input(
        'uint32x2',
        rows.flatMap(row => row.cell)
      ),
      operation: 'neighbors',
      output: {cells: output.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, rows.length)).toEqual(rows.map(() => 0));
    const words = await readUint32(output.buffer, rows.length * 8);
    for (let row = 0; row < rows.length; row++) {
      const actual = Array.from({length: 4}, (_, slot) =>
        words.slice(row * 8 + slot * 2, row * 8 + slot * 2 + 2).join(':')
      ).sort();
      expect(actual).toEqual(rows[row].neighbors.map(key => key.join(':')).sort());
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('S2 measures agree with pinned spherical area/perimeter through level 30 with stable local increments', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeFixture(device);
  const rows = fixtureReference.cells;
  const output = fixture.output('float32x2', rows.length);
  const status = fixture.output('uint32', rows.length);
  fixture.graph.add(
    new GPUS2CellMeasures({
      cells: fixture.input(
        'uint32x2',
        rows.flatMap(row => row.cell)
      ),
      output: {measures: output.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const statuses = await readUint32(status.buffer, rows.length);
    const values = await readFloat32(output.buffer, rows.length * 2);
    for (let row = 0; row < rows.length; row++) {
      expect(statuses[row]).toBe(0);
      const tolerance = 2e-5;
      expect(Math.abs(values[row * 2] / rows[row].area - 1)).toBeLessThan(tolerance);
      expect(Math.abs(values[row * 2 + 1] / rows[row].perimeter - 1)).toBeLessThan(tolerance);
    }
    expect(
      values
        .filter((_, offset) => offset < 12 && offset % 2 === 0)
        .reduce((total, area) => total + area, 0)
    ).toBeCloseTo(4 * Math.PI, 4);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('S2 cap coverings distinguish exterior/interior, globe coverage, invalidity and capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['exterior', 'interior'] as const) {
    const fixture = makeFixture(device);
    const output = fixture.output('uint32x2', 12);
    const counts = fixture.output('uint32x2', 3);
    const status = fixture.output('uint32', 3);
    fixture.graph.add(
      new GPUS2CapCovering({
        caps: fixture.input('float32x4', [1, 0, 0, 0.1, 1, 0, 0, Math.PI, 0, 0, 0, 1]),
        level: 0,
        mode,
        capacity: 4,
        output: {cells: output.view, counts: counts.view, status: status.view}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status.buffer, 3)).toEqual([0, 2, 1]);
      // Interior's strict safety margin excludes the opposite root cell whose farthest point is pi.
      expect(await readUint32(counts.buffer, 6)).toEqual(
        mode === 'exterior' ? [1, 1, 4, 6, 0, 0] : [0, 0, 4, 5, 0, 0]
      );
      if (mode === 'exterior') expect(await readUint32(output.buffer, 2)).toEqual([0, 0x10000000]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('spherical containment distinguishes closed/open boundaries and convex-region complements at poles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const complement of [false, true])
    for (const boundary of ['closed', 'open'] as const) {
      const fixture = makeFixture(device);
      const output = fixture.output('uint32', 4);
      const status = fixture.output('uint32', 4);
      const direction = 1 / Math.sqrt(3);
      fixture.graph.add(
        new GPUSphericalConvexContains({
          positions: fixture.input('float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
          offsets: fixture.input('uint32', [0, 3]),
          queries: fixture.input('float32x3', [
            direction,
            direction,
            direction,
            -direction,
            -direction,
            -direction,
            0,
            0,
            1,
            0,
            0,
            0
          ]),
          polygonIndices: fixture.input('uint32', [0, 0, 0, 0]),
          boundary,
          complement,
          output: {contains: output.view, status: status.view}
        })
      );
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(status.buffer, 4)).toEqual([0, 0, 2, 1]);
        expect(await readUint32(output.buffer, 4)).toEqual([
          complement ? 0 : 1,
          complement ? 1 : 0,
          boundary === 'closed' ? 1 : 0,
          0
        ]);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
});

it('datum grid selection honors priority and signed height units without nodata fallthrough', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const firstValid of [0, 1]) {
    const fixture = makeFixture(device);
    const output = fixture.output('float32x3', 1);
    const status = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUDatumGridTransform({
        positions: fixture.input('float32x3', [0.5, 0.5, 100]),
        samples: fixture.input('float32x3', [
          ...Array(4).fill([0.1, 0.2, -10]).flat(),
          ...Array(4).fill([0.3, 0.4, -20]).flat()
        ]),
        mask: fixture.input('uint32', [firstValid, firstValid, firstValid, firstValid, 1, 1, 1, 1]),
        grids: [
          {width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]},
          {width: 2, height: 2, sampleOffset: 4, origin: [0, 0], spacing: [1, 1]}
        ],
        operation: 'combined',
        sourceVerticalReference: 'ellipsoid',
        targetVerticalReference: 'orthometric',
        heightUnitMeters: 0.3048,
        output: {positions: output.view, status: status.view}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status.buffer, 1)).toEqual([firstValid ? 0 : 4]);
      const values = await readFloat32(output.buffer, 3);
      if (firstValid) {
        expect(values[0]).toBeCloseTo(0.6, 5);
        expect(values[1]).toBeCloseTo(0.7, 5);
        expect(values[2]).toBeCloseTo(100 - 10 / 0.3048, 4);
      } else {
        expect(values[0]).toBeNaN();
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('supported ellipsoidal lengths match pinned GeographicLib pairs across poles and the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = fixtureReference.geodesics.slice(0, 3);
  const fixture = makeFixture(device);
  const lengths = fixture.output('float32', rows.length);
  const invalid = fixture.output('uint32', 1);
  fixture.graph.add(
    new GPUGeometryMeasures({
      geometryType: 'lines',
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal',
        units: 'meters'
      },
      positions: fixture.input(
        'float32x2',
        rows.flatMap(row => [...row.origin, ...row.target])
      ),
      lineOffsets: fixture.input('uint32', [0, 2, 4, 6]),
      output: {lengths: lengths.view, status: {invalidCount: invalid.view}}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(invalid.buffer, 1)).toEqual([0]);
    const values = await readFloat32(lengths.buffer, rows.length);
    for (let row = 0; row < rows.length; row++)
      expect(Math.abs(values[row] - rows[row].distance)).toBeLessThan(5);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('spherical crossing assembly and cross-family overlap weights match a composed independent oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeFixture(device);
  const reference = fixtureReference.overlap;
  const positions = fixture.output('float32x3', 16);
  const counts = fixture.output('uint32x2', 1);
  const status = fixture.output('uint32', 1);
  const measures = fixture.output('float32x4', 1);
  fixture.graph.add(
    new GPUSphericalConvexIntersection({
      firstPositions: fixture.input('float32x3', reference.first.flat()),
      secondPositions: fixture.input('float32x3', reference.second.flat()),
      firstOffsets: fixture.input('uint32', [0, reference.first.length]),
      secondOffsets: fixture.input('uint32', [0, reference.second.length]),
      capacity: 16,
      output: {
        positions: positions.view,
        counts: counts.view,
        status: status.view,
        measures: measures.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 1)).toEqual([0]);
    expect(await readUint32(counts.buffer, 2)).toEqual([
      reference.intersection.length,
      reference.intersection.length
    ]);
    const values = await readFloat32(measures.buffer, 4);
    expect(values[0]).toBeCloseTo(reference.area, 5);
    expect(values[1]).toBeCloseTo(reference.sourceFraction, 5);
    expect(values[2]).toBeCloseTo(reference.targetFraction, 5);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('vertical grid corrections match pinned PROJ vgridshift with explicit multiplier and signed samples', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const actual = await runDatum(device, fixtureReference.vertical.input, {operation: 'vertical'});
  expect(actual.status).toEqual([0]);
  actual.positions.forEach((value, index) =>
    expect(value).toBeCloseTo(fixtureReference.vertical.output[index], 5)
  );
});

it('S2 cap covers exclude perpendicular edge planes despite f32 normalization residues', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeFixture(device);
  const capacity = 1536;
  const output = fixture.output('uint32x2', capacity);
  const counts = fixture.output('uint32x2', 1);
  const status = fixture.output('uint32', 1);
  fixture.graph.add(
    new GPUS2CapCovering({
      caps: fixture.input('float32x4', [1, 0, 0, fixtureReference.covering.initialRadius]),
      level: 4,
      capacity,
      output: {cells: output.view, counts: counts.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 1)).toEqual([0]);
    expect(await readUint32(counts.buffer, 2)).toEqual([
      fixtureReference.covering.initialCells.length,
      fixtureReference.covering.initialCells.length
    ]);
    const cells = await readUint32(output.buffer, capacity * 2);
    expect(cells.slice(0, fixtureReference.covering.initialCells.length * 2)).toEqual(
      fixtureReference.covering.initialCells.flat()
    );
    expect(
      cells.slice(fixtureReference.covering.initialCells.length * 2).every(word => word === 0)
    ).toBe(true);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('datum transforms preserve small shifts and inverse residuals using split coordinate words', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeFixture(device);
  const high = fixture.output('float32x3', 1);
  const low = fixture.output('float32x3', 1);
  const status = fixture.output('uint32', 1);
  const restoredHigh = fixture.output('float32x3', 1);
  const restoredLow = fixture.output('float32x3', 1);
  const restoredStatus = fixture.output('uint32', 1);
  const original = [179.123456789, 45.123456789, 1000.123456789];
  const shift = [0.000000123, -0.000000234, 0.0000123];
  const samples = fixture.input('float32x3', [...shift, ...shift, ...shift, ...shift]);
  const validity = fixture.input('uint32', [1, 1, 1, 1]);
  const common = {
    samples,
    mask: validity,
    grids: [
      {
        width: 2,
        height: 2,
        sampleOffset: 0,
        origin: [179.123, 45.123] as const,
        spacing: [0.001, 0.001] as const
      }
    ],
    operation: 'combined' as const,
    sourceVerticalReference: 'source',
    targetVerticalReference: 'target',
    tolerance: 1e-10
  };
  fixture.graph.add(
    new GPUDatumGridTransform({
      ...common,
      positions: fixture.input('float32x3', original.map(Math.fround)),
      positionsLow: fixture.input(
        'float32x3',
        original.map(value => value - Math.fround(value))
      ),
      output: {positions: high.view, positionsLow: low.view, status: status.view}
    })
  );
  fixture.graph.add(
    new GPUDatumGridTransform({
      ...common,
      id: 'split-inverse',
      direction: 'inverse',
      positions: high.view,
      positionsLow: low.view,
      output: {
        positions: restoredHigh.view,
        positionsLow: restoredLow.view,
        status: restoredStatus.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 1)).toEqual([0]);
    expect(await readUint32(restoredStatus.buffer, 1)).toEqual([0]);
    const upper = await readFloat32(high.buffer, 3);
    const lower = await readFloat32(low.buffer, 3);
    const restoredUpper = await readFloat32(restoredHigh.buffer, 3);
    const restoredLower = await readFloat32(restoredLow.buffer, 3);
    for (let component = 0; component < 3; component++) {
      expect(
        Math.abs(upper[component] + lower[component] - original[component] - shift[component])
      ).toBeLessThan(3e-11);
      expect(
        Math.abs(restoredUpper[component] + restoredLower[component] - original[component])
      ).toBeLessThan(3e-11);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

// Math correctness review 2026-11-10, geometry-measures-05. With the default tolerance the
// inverse used to test the residual before its first update. That residual is the grid shift
// itself, so any shift under the old default of 1e-5 degrees (1.1 m) was reported as converged
// with the input returned unchanged. Expected values are closed forms:
// - a constant shift s inverts to q - s;
// - a shift a + b (x - x0) that is linear in longitude inverts to q - (a + b (q - x0)) / (1 + b):
//   with a = 2e-6, b = 4e-6 per degree and q - x0 = 0.5, the offset is -4e-6 / 1.000004.
it('datum inverse applies shifts smaller than a meter with the default tolerance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeFixture(device);
  const high = fixture.output('float32x3', 2);
  const low = fixture.output('float32x3', 2);
  const status = fixture.output('uint32', 2);
  const constantShift = [5e-6, -3e-6, 0];
  fixture.graph.add(
    new GPUDatumGridTransform({
      samples: fixture.input('float32x3', [
        ...constantShift,
        ...constantShift,
        ...constantShift,
        ...constantShift,
        ...[2e-6, 0, 0],
        ...[6e-6, 0, 0],
        ...[2e-6, 0, 0],
        ...[6e-6, 0, 0]
      ]),
      mask: fixture.input('uint32', [1, 1, 1, 1, 1, 1, 1, 1]),
      grids: [
        {width: 2, height: 2, sampleOffset: 0, origin: [10, 20], spacing: [1, 1]},
        {width: 2, height: 2, sampleOffset: 4, origin: [30, 20], spacing: [1, 1]}
      ],
      operation: 'horizontal',
      direction: 'inverse',
      positions: fixture.input('float32x3', [10.5, 20.5, 0, 30.5, 20.5, 0]),
      positionsLow: fixture.input('float32x3', [0, 0, 0, 0, 0, 0]),
      output: {positions: high.view, positionsLow: low.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 2)).toEqual([0, 0]);
    expect(await readFloat32(high.buffer, 6)).toEqual([10.5, 20.5, 0, 30.5, 20.5, 0]);
    const lower = await readFloat32(low.buffer, 6);
    expect(Math.abs(lower[0] - -5e-6)).toBeLessThan(1e-11);
    expect(Math.abs(lower[1] - 3e-6)).toBeLessThan(1e-11);
    // One update leaves a residual of b * shift = 1.6e-11 degrees, inside the default tolerance.
    expect(Math.abs(lower[3] - -4e-6 / 1.000004)).toBeLessThan(1e-10);
    expect(lower[4]).toBe(0);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('datum transforms reject sub-ulp latitude excursions at both poles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const pole of [-90, 90]) {
    for (const shift of [0, Math.sign(pole) * 1e-6]) {
      const fixture = makeFixture(device);
      const high = fixture.output('float32x3', 3);
      const low = fixture.output('float32x3', 3);
      const status = fixture.output('uint32', 3);
      const residual = Math.sign(pole) * 1e-6;
      fixture.graph.add(
        new GPUDatumGridTransform({
          positions: fixture.input('float32x3', [0, pole, 0, 0, pole, 0, 0, pole, 0]),
          positionsLow: fixture.input('float32x3', [0, residual, 0, 0, -residual, 0, 0, 0, 0]),
          samples: fixture.input('float32x3', Array.from({length: 4}, () => [0, shift, 0]).flat()),
          mask: fixture.input('uint32', [1, 1, 1, 1]),
          grids: [{width: 2, height: 2, sampleOffset: 0, origin: [-1, -90], spacing: [2, 180]}],
          operation: 'horizontal',
          output: {positions: high.view, positionsLow: low.view, status: status.view}
        })
      );
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const expected = [1, 0, shift === 0 ? 0 : 1];
        expect(await readUint32(status.buffer, 3)).toEqual(expected);
        const upper = await readFloat32(high.buffer, 9);
        const lower = await readFloat32(low.buffer, 9);
        for (let row = 0; row < 3; row++) {
          if (expected[row] === 1) {
            expect(upper.slice(row * 3, row * 3 + 3).every(Number.isNaN)).toBe(true);
            expect(lower.slice(row * 3, row * 3 + 3).every(Number.isNaN)).toBe(true);
          } else {
            expect(Math.abs(upper[row * 3 + 1] + lower[row * 3 + 1])).toBeLessThanOrEqual(90);
          }
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
  }
});

it('general spherical polygon winding matches direct S2 for nonconvex rings, holes, poles and complements', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of polygonReference.cases)
    for (const complement of [false, true]) {
      const fixture = makeFixture(device);
      const contains = fixture.output('uint32', reference.queries.length);
      const status = fixture.output('uint32', reference.queries.length);
      const offsets = [0];
      for (const ring of reference.rings) offsets.push(offsets[offsets.length - 1] + ring.length);
      fixture.graph.add(
        new GPUSphericalPolygonContains({
          positions: fixture.input('float32x3', reference.rings.flat(2)),
          ringOffsets: fixture.input('uint32', offsets),
          polygonOffsets: fixture.input('uint32', [0, reference.rings.length]),
          queries: fixture.input('float32x3', reference.queries.flat()),
          polygonIndices: fixture.input('uint32', new Array(reference.queries.length).fill(0)),
          complement,
          output: {contains: contains.view, status: status.view}
        })
      );
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(status.buffer, reference.queries.length)).toEqual(
          new Array(reference.queries.length).fill(0)
        );
        expect(await readUint32(contains.buffer, reference.queries.length)).toEqual(
          reference.contains.map(value => (complement ? 1 - value : value))
        );
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
});

it('spherical general overlay matches direct S2 nonconvex and holed Boolean areas', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of overlayReference.cases) {
    const fixture = makeFixture(device);
    const operand = (rings: number[][][]) => {
      const offsets = [0];
      for (const ring of rings) offsets.push(offsets[offsets.length - 1] + ring.length);
      return {
        kind: 'polygons' as const,
        positions: fixture.input('float32x3', rings.flat(2)),
        ringOffsets: fixture.input('uint32', offsets),
        polygonOffsets: fixture.input('uint32', [0, rings.length]),
        featureOffsets: fixture.input('uint32', [0, 1])
      };
    };
    const positions = fixture.output('float32x3', 128);
    const ringOffsets = fixture.output('uint32', 17);
    const polygonOffsets = fixture.output('uint32', 17);
    const featureOffsets = fixture.output('uint32', 17);
    const count = fixture.output('uint32', 1);
    const overflow = fixture.output('uint32', 1);
    const required = fixture.output('uint32', 1);
    const candidate = fixture.output('uint32', 1);
    const complement = fixture.output('uint32', 1);
    const boundedStatus = () => ({
      count: fixture.output('uint32', 1).view,
      requiredCount: fixture.output('uint32', 1).view,
      overflow: fixture.output('uint32', 1).view,
      candidateOverflow: fixture.output('uint32', 1).view
    });
    fixture.graph.add(
      new GPUSphericalPolygonOverlay({
        left: operand(overlayReference.left),
        right: operand(overlayReference.right),
        chartCenter: [0, 0, 1],
        operation: reference.operation as
          | 'intersection'
          | 'union'
          | 'difference'
          | 'symmetric-difference',
        vertexTolerance: 2 ** -18,
        capacity: {intersections: 64, nodedSegments: 128},
        output: {
          geometry: {
            kind: 'polygons',
            positions: positions.view,
            ringOffsets: ringOffsets.view,
            polygonOffsets: polygonOffsets.view,
            featureOffsets: featureOffsets.view
          },
          complement: complement.view,
          status: {
            count: count.view,
            requiredRingCount: required.view,
            overflow: overflow.view,
            candidateOverflow: candidate.view
          },
          boundary: {
            endpoints: fixture.output('float32x4', 128).view,
            operandIds: fixture.output('uint32', 128).view,
            sourceFeatureIds: fixture.output('uint32', 128).view,
            status: boundedStatus()
          }
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
      expect(await readUint32(complement.buffer, 1)).toEqual([0]);
      const features = (await readUint32(count.buffer, 1))[0];
      const featureValues = await readUint32(featureOffsets.buffer, 17);
      const polygonValues = await readUint32(polygonOffsets.buffer, 17);
      const ringValues = await readUint32(ringOffsets.buffer, 17);
      const values = await readFloat32(positions.buffer, 384);
      const rings = polygonValues[featureValues[features]];
      expect(rings).toBe(reference.rings.length);
      let area = 0;
      for (let ring = 0; ring < rings; ring++) {
        const start = ringValues[ring];
        const end = ringValues[ring + 1];
        for (let vertex = start; vertex < end; vertex++) {
          const first = values.slice(vertex * 3, vertex * 3 + 3);
          const next = vertex + 1 === end ? start : vertex + 1;
          const second = values.slice(next * 3, (next + 1) * 3);
          // Independent solid angle of (north pole, first, second); holes subtract by orientation.
          const numerator = first[0] * second[1] - first[1] * second[0];
          const denominator =
            1 +
            first[2] +
            second[2] +
            first.reduce((sum, value, component) => sum + value * second[component], 0);
          area += 2 * Math.atan2(numerator, denominator);
        }
      }
      expect(
        Math.abs(area - reference.area),
        JSON.stringify({
          operation: reference.operation,
          area,
          expected: reference.area,
          ringValues: ringValues.slice(0, rings + 1),
          values: values.slice(0, ringValues[rings] * 3)
        })
      ).toBeLessThan(3e-5);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
}, 60000);

it('adaptive S2 polygon covers satisfy direct S2 interior/exterior guarantees and explicit exhaustion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of coverReference.cases)
    for (const mode of ['exterior', 'interior'] as const) {
      const fixture = makeFixture(device);
      const cells = fixture.output('uint32x2', 2048);
      const counts = fixture.output('uint32x2', 1);
      const status = fixture.output('uint32', 1);
      const visited = fixture.output('uint32', 1);
      const offsets = [0];
      for (const ring of reference.rings) offsets.push(offsets[offsets.length - 1] + ring.length);
      fixture.graph.add(
        new GPUS2PolygonCovering({
          positions: fixture.input('float32x3', reference.rings.flat(2)),
          ringOffsets: fixture.input('uint32', offsets),
          maximumLevel: reference.level,
          mode,
          capacity: 2048,
          output: {
            cells: cells.view,
            counts: counts.view,
            status: status.view,
            visitedCount: visited.view
          }
        })
      );
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(status.buffer, 1)).toEqual([0]);
        const count = (await readUint32(counts.buffer, 2))[0];
        const keys = await readUint32(cells.buffer, count * 2);
        const identifiers = Array.from(
          {length: count},
          (_, index) => BigInt(keys[index * 2]) + (BigInt(keys[index * 2 + 1]) << 32n)
        );
        const covers = (words: number[]) => {
          const child = BigInt(words[0]) + (BigInt(words[1]) << 32n);
          return identifiers.some(parent => {
            const marker = parent & -parent;
            return child >= parent - marker + 1n && child <= parent + marker - 1n;
          });
        };
        let area = 0;
        for (const cell of reference.cells) {
          const included = covers(cell.cell);
          if (included) area += cell.area;
          if (mode === 'interior' && included) expect(cell.contained).toBe(true);
          if (mode === 'exterior' && cell.intersects)
            expect(included, JSON.stringify(cell)).toBe(true);
        }
        if (mode === 'exterior') expect(area).toBeGreaterThanOrEqual(reference.area);
        else expect(area).toBeLessThanOrEqual(reference.area);
        expect((await readUint32(visited.buffer, 1))[0]).toBeLessThan(reference.cells.length);
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
});

it('intrinsic spherical arrangement agrees with direct S2 for non-hemisphere and complemented regions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reference of arcReference.cases) {
    const fixture = makeFixture(device);
    const positions = fixture.output('float32x3', 64);
    const rings = fixture.output('uint32', 17);
    const counts = fixture.output('uint32', 4);
    const status = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUSphericalArcOverlay({
        left: {
          positions: fixture.input('float32x3', arcReference.left.flat()),
          ringOffsets: fixture.input('uint32', [0, 4]),
          referenceInside: reference.leftInside
        },
        right: {
          positions: fixture.input('float32x3', arcReference.right.flat()),
          ringOffsets: fixture.input('uint32', [0, 4]),
          referenceInside: reference.rightInside
        },
        referencePoint: [0, 0, 1],
        operation: reference.operation as
          | 'intersection'
          | 'union'
          | 'difference'
          | 'symmetric-difference',
        output: {
          positions: positions.view,
          ringOffsets: rings.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(status.buffer, 1), JSON.stringify(reference)).toEqual([0]);
      const count = await readUint32(counts.buffer, 4);
      expect(count[1]).toBe(reference.rings);
      expect(count[3]).toBe(Number(reference.referenceInside));
      const offsets = await readUint32(rings.buffer, count[1] + 1);
      const values = await readFloat32(positions.buffer, count[0] * 3);
      let area = 0;
      for (let ring = 0; ring < count[1]; ring++)
        for (let vertex = offsets[ring]; vertex < offsets[ring + 1]; vertex++) {
          const next = vertex + 1 === offsets[ring + 1] ? offsets[ring] : vertex + 1;
          const first = values.slice(vertex * 3, vertex * 3 + 3);
          const second = values.slice(next * 3, next * 3 + 3);
          area +=
            2 *
            Math.atan2(
              first[0] * second[1] - first[1] * second[0],
              1 +
                first[2] +
                second[2] +
                first.reduce((sum, value, component) => sum + value * second[component], 0)
            );
        }
      if (area < 0) area += 4 * Math.PI;
      if (count[0] === 0 && count[3] === 1) area = 4 * Math.PI;
      expect(
        Math.abs(area - reference.area),
        JSON.stringify({reference, area, count})
      ).toBeLessThan(2e-5);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
}, 60000);

it('split horizontal grid priority, inverse and bounds agree with direct PROJ hgridshift', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const direction of ['forward', 'inverse'] as const) {
    const fixture = makeFixture(device);
    const rows = datumReference.rows;
    const source = rows.flatMap(row =>
      direction === 'forward' ? row.input : (row.forward ?? row.input)
    );
    const high = fixture.output('float32x3', rows.length);
    const low = fixture.output('float32x3', rows.length);
    const status = fixture.output('uint32', rows.length);
    let sampleOffset = 0;
    const grids = datumReference.grids.map(grid => {
      const result = {
        width: grid.width,
        height: grid.height,
        origin: [grid.origin[0], grid.origin[1]] as const,
        spacing: [grid.spacing[0], grid.spacing[1]] as const,
        sampleOffset
      };
      sampleOffset += grid.width * grid.height;
      return result;
    });
    fixture.graph.add(
      new GPUDatumGridTransform({
        positions: fixture.input('float32x3', source.map(Math.fround)),
        positionsLow: fixture.input(
          'float32x3',
          source.map(value => value - Math.fround(value))
        ),
        samples: fixture.input(
          'float32x3',
          datumReference.grids.flatMap(grid => grid.samples)
        ),
        mask: fixture.input('uint32', new Array(sampleOffset).fill(1)),
        grids,
        operation: 'horizontal',
        direction,
        tolerance: 1e-10,
        maximumIterations: 32,
        output: {positions: high.view, positionsLow: low.view, status: status.view}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const statuses = await readUint32(status.buffer, rows.length);
      const upper = await readFloat32(high.buffer, source.length);
      const lower = await readFloat32(low.buffer, source.length);
      for (let row = 0; row < rows.length; row++) {
        const expected = direction === 'forward' ? rows[row].forward : rows[row].inverse;
        if (expected === null) {
          expect(statuses[row]).not.toBe(0);
        } else {
          expect(statuses[row], JSON.stringify({direction, row})).toBe(0);
          for (let component = 0; component < 3; component++)
            expect(
              Math.abs(
                upper[row * 3 + component] + lower[row * 3 + component] - expected[component]
              ),
              JSON.stringify({direction, row, component})
            ).toBeLessThan(1e-9);
        }
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('safeguarded ellipsoidal inverse validates a dense direct GeographicLib near-antipodal sweep', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const cooperativeRingRows of [0, 1]) {
    const fixture = makeFixture(device);
    const rows = difficultReference.rows;
    const lengths = fixture.output('float32', rows.length);
    const invalid = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUGeometryMeasures({
        geometryType: 'lines',
        ellipsoidalInverse: 'safeguarded',
        cooperativeRingRows,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'ellipsoidal',
          units: 'meters'
        },
        positions: fixture.input(
          'float32x2',
          rows.flatMap(row => [...row.first, ...row.second])
        ),
        lineOffsets: fixture.input(
          'uint32',
          Array.from({length: rows.length + 1}, (_, index) => index * 2)
        ),
        output: {lengths: lengths.view, status: {invalidCount: invalid.view}}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const values = await readFloat32(lengths.buffer, rows.length);
      const failures = values.filter(value => !Number.isFinite(value)).length;
      expect((await readUint32(invalid.buffer, 1))[0]).toBe(failures);
      expect(failures).toBeLessThan(8);
      for (let row = 0; row < rows.length; row++)
        if (Number.isFinite(values[row]))
          expect(
            Math.abs(values[row] - rows[row].distance),
            JSON.stringify({row, reference: rows[row], actual: values[row]})
          ).toBeLessThan(10);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
}, 60000);

it('adaptive ellipsoidal area validates GeographicLib holes and reports exhausted quadrature', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const cooperativeRingRows of [0, 1])
    for (const exhausted of [false, true]) {
      const fixture = makeFixture(device);
      const rows = areaReference.rows;
      const areas = fixture.output('float32', rows.length);
      const invalid = fixture.output('uint32', 1);
      fixture.graph.add(
        new GPUGeometryMeasures({
          geometryType: 'polygons',
          ellipsoidalInverse: 'safeguarded',
          geodesicAreaTolerance: exhausted ? 1e-20 : 1e6,
          geodesicAreaMaximumSegments: exhausted ? 2 : 256,
          cooperativeRingRows,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'ellipsoidal',
            units: 'meters'
          },
          positions: fixture.input(
            'float32x2',
            rows.flatMap(row => row.rings.flat(2))
          ),
          ringOffsets: fixture.input(
            'uint32',
            Array.from({length: rows.length * 2 + 1}, (_, index) => index * 4)
          ),
          polygonOffsets: fixture.input(
            'uint32',
            Array.from({length: rows.length + 1}, (_, index) => index * 2)
          ),
          featureOffsets: fixture.input(
            'uint32',
            Array.from({length: rows.length + 1}, (_, index) => index)
          ),
          output: {areas: areas.view, status: {invalidCount: invalid.view}}
        })
      );
      const compiled = fixture.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(areas.buffer, rows.length);
        const failures = actual.filter(value => !Number.isFinite(value)).length;
        expect((await readUint32(invalid.buffer, 1))[0]).toBe(failures);
        if (exhausted) {
          expect(failures).toBe(rows.length);
        } else {
          expect(failures).toBe(0);
          for (let row = 0; row < rows.length; row++)
            expect(
              Math.abs(actual[row] - rows[row].area),
              JSON.stringify({row, actual: actual[row], reference: rows[row].area})
            ).toBeLessThan(Math.max(2e6, rows[row].area * 3e-6));
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    }
}, 60000);

it('explicit hemisphere poles preserve containment after uneven great-circle densification', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeFixture(device);
  const chart = [
    [1, -5, -0.2],
    [1, 5, -0.2],
    ...Array.from({length: 31}, (_, index) => [1, 5, -0.2 + (0.4 * (index + 1)) / 32]),
    [1, 5, 0.2],
    [1, -5, 0.2]
  ];
  const vertices = chart.flatMap(point => {
    const length = Math.hypot(...point);
    return point.map(value => value / length);
  });
  const contains = fixture.output('uint32', 1);
  const status = fixture.output('uint32', 1);
  fixture.graph.add(
    new GPUSphericalPolygonContains({
      positions: fixture.input('float32x3', vertices),
      ringOffsets: fixture.input('uint32', [0, chart.length]),
      polygonOffsets: fixture.input('uint32', [0, 1]),
      queries: fixture.input('float32x3', [1, 0, 0]),
      polygonIndices: fixture.input('uint32', [0]),
      hemisphereCenter: [1, 0, 0],
      output: {contains: contains.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 1)).toEqual([0]);
    expect(await readUint32(contains.buffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('adaptive covers and intrinsic overlays expose incomplete capacities and traversal budgets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const budget of [6, 65536]) {
    const fixture = makeFixture(device);
    const status = fixture.output('uint32', 1);
    const counts = fixture.output('uint32x2', 1);
    fixture.graph.add(
      new GPUS2PolygonCovering({
        positions: fixture.input('float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
        ringOffsets: fixture.input('uint32', [0, 3]),
        maximumLevel: 5,
        maximumVisitedCells: budget,
        capacity: 1,
        output: {
          cells: fixture.output('uint32x2', 1).view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const failure = (await readUint32(status.buffer, 1))[0];
      expect(failure & (budget === 6 ? 4 : 2)).not.toBe(0);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
  const fixture = makeFixture(device);
  const status = fixture.output('uint32', 1);
  const counts = fixture.output('uint32', 4);
  const positions = arcReference.left.flat(2);
  const other = arcReference.right.flat(2);
  fixture.graph.add(
    new GPUSphericalArcOverlay({
      left: {
        positions: fixture.input('float32x3', positions),
        ringOffsets: fixture.input('uint32', [0, positions.length / 3]),
        referenceInside: true
      },
      right: {
        positions: fixture.input('float32x3', other),
        ringOffsets: fixture.input('uint32', [0, other.length / 3]),
        referenceInside: true
      },
      referencePoint: [0, 0, 1],
      operation: 'symmetric-difference',
      output: {
        positions: fixture.output('float32x3', 3).view,
        ringOffsets: fixture.output('uint32', 2).view,
        counts: counts.view,
        status: status.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect((await readUint32(status.buffer, 1))[0] & 2).toBe(2);
    expect((await readUint32(counts.buffer, 4)).slice(0, 2)).toEqual([0, 0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('spherical predicates and adaptive covers reject self-crossing and touching rings before traversal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Gnomonic projection preserves straight segments as minor great-circle arcs.
  const rings = [
    [
      [-1, -1],
      [1, 1],
      [-1, 1],
      [1, -1]
    ],
    [
      [-1, -1],
      [0, 0],
      [-1, 1],
      [1, 1],
      [0, 0],
      [1, -1]
    ],
    [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1]
    ]
  ];
  for (const [ringIndex, ring] of rings.entries()) {
    const fixture = makeFixture(device);
    const positions = fixture.input(
      'float32x3',
      ring.flatMap(([horizontal, vertical]) => {
        const magnitude = Math.hypot(1, horizontal, vertical);
        return [1 / magnitude, horizontal / magnitude, vertical / magnitude];
      })
    );
    const ringOffsets = fixture.input('uint32', [0, ring.length]);
    const contains = fixture.output('uint32', 1);
    const containsStatus = fixture.output('uint32', 1);
    const cells = fixture.output('uint32x2', 128);
    const counts = fixture.output('uint32x2', 1);
    const status = fixture.output('uint32', 1);
    const visited = fixture.output('uint32', 1);
    fixture.graph.add(
      new GPUSphericalPolygonContains({
        positions,
        ringOffsets,
        polygonOffsets: fixture.input('uint32', [0, 1]),
        queries: fixture.input('float32x3', [1, 0, 0]),
        polygonIndices: fixture.input('uint32', [0]),
        hemisphereCenter: [1, 0, 0],
        output: {contains: contains.view, status: containsStatus.view}
      })
    );
    fixture.graph.add(
      new GPUS2PolygonCovering({
        positions,
        ringOffsets,
        hemisphereCenter: [1, 0, 0],
        maximumLevel: 2,
        capacity: 128,
        output: {
          cells: cells.view,
          counts: counts.view,
          status: status.view,
          visitedCount: visited.view
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const invalid = ringIndex < 2;
      expect(await readUint32(containsStatus.buffer, 1)).toEqual([invalid ? 1 : 0]);
      expect(await readUint32(contains.buffer, 1)).toEqual([invalid ? 0 : 1]);
      expect(await readUint32(status.buffer, 1)).toEqual([invalid ? 1 : 0]);
      if (invalid) {
        expect(await readUint32(counts.buffer, 2)).toEqual([0, 0]);
        expect(await readUint32(visited.buffer, 1)).toEqual([0]);
        expect(await readUint32(cells.buffer, 256)).toEqual(Array(256).fill(0));
      } else {
        expect((await readUint32(counts.buffer, 2))[0]).toBeGreaterThan(0);
        expect((await readUint32(visited.buffer, 1))[0]).toBeGreaterThan(0);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});

it('ENT-02 spherical containment preserves public-shape vector chunks across module formats', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kind of ['convex', 'polygon'] as const) {
    const fixture = makeFixture(device);
    // A metadata-only object with the published vector shape has no realm-specific prototype.
    // The Node companion uses actual CommonJS constructors; here we execute the same shape.
    const vector = <Format extends 'float32x3' | 'uint32'>(data: GraphDataView<Format>[]) => {
      const rows = new GraphVectorView({
        id: 'rows',
        name: 'rows',
        format: data[0].format,
        length: data.length,
        valueLength: (data.length * data[0].rowByteLength) / 4,
        stride: data[0].rowByteLength / 4,
        byteStride: data[0].byteStride,
        rowByteLength: data[0].rowByteLength,
        data
      });
      return Object.assign(Object.create(null), rows) as GraphVectorView<Format>;
    };
    const direction = 1 / Math.sqrt(3);
    const contains = [fixture.output('uint32', 1), fixture.output('uint32', 1)];
    const status = [fixture.output('uint32', 1), fixture.output('uint32', 1)];
    for (const field of [...contains, ...status]) field.buffer.write(Uint32Array.of(0xffffffff));
    const props = {
      positions: fixture.input('float32x3', [1, 0, 0, 0, 1, 0, 0, 0, 1]),
      queries: vector([
        fixture.input('float32x3', [direction, direction, direction]),
        fixture.input('float32x3', [-direction, -direction, -direction])
      ]),
      polygonIndices: vector([fixture.input('uint32', [0]), fixture.input('uint32', [0])]),
      output: {
        contains: vector(contains.map(field => field.view)),
        status: vector(status.map(field => field.view))
      }
    };
    fixture.graph.add(
      kind === 'convex'
        ? new GPUSphericalConvexContains({...props, offsets: fixture.input('uint32', [0, 3])})
        : new GPUSphericalPolygonContains({
            ...props,
            ringOffsets: fixture.input('uint32', [0, 3]),
            polygonOffsets: fixture.input('uint32', [0, 1])
          })
    );
    const compiled = fixture.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // The positive octant triangle contains its normalized center, excludes its antipode.
      expect(await readUint32(contains[0].buffer, 1)).toEqual([1]);
      expect(await readUint32(contains[1].buffer, 1)).toEqual([0]);
      for (const field of status) expect(await readUint32(field.buffer, 1)).toEqual([0]);
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  }
});
