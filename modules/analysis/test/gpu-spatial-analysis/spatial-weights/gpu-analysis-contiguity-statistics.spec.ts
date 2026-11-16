// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUAnalysisContiguityStatistics,
  decodeGPUAnalysisContiguityStatistics,
  type GPUContiguityCriterion
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {flattenFeatures, flattenPolygons, type PolygonLayout} from './spatial-weights-oracle';

function square(left: number, bottom: number, size = 1): [number, number][] {
  return [
    [left, bottom],
    [left + size, bottom],
    [left + size, bottom + size],
    [left, bottom + size]
  ];
}

async function runProfile(
  device: Device,
  polygons: PolygonLayout,
  options: {
    criterion: GPUContiguityCriterion;
    capacity?: number;
    pairCapacity?: number;
    snapTolerance?: number;
    coordinateExtent?: number;
    /** Same-shaped replacement, submitted on the same compiled graph. */
    nextPositions?: Float32Array;
  }
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const input = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = track(createInputBuffer(device, values));
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        name,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    length: number,
    format: Format
  ) => {
    const buffer = track(createOutputBuffer(device, length));
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    const featureCount = polygons.featureOffsets.length - 1;
    const capacity = options.capacity ?? featureCount * 8;
    const positions = input('positions', polygons.positions, 'float32x2');
    const ringOffsets = input('ring-offsets', polygons.ringOffsets, 'uint32');
    const polygonOffsets = input('polygon-offsets', polygons.polygonOffsets, 'uint32');
    const featureOffsets = input('feature-offsets', polygons.featureOffsets, 'uint32');
    const offsets = output('offsets', featureCount + 1, 'uint32');
    const neighbors = output('neighbors', capacity, 'uint32');
    const weights = output('weights', capacity, 'float32');
    const overflow = output('overflow', 1, 'uint32');
    const requiredCount = output('required-count', 1, 'uint32');
    const candidateOverflow = output('candidate-overflow', 1, 'uint32');
    const cardinality = output('cardinality', featureCount, 'uint32');
    const counts = output('counts', 6, 'uint32');
    const median = output('median', 1, 'float32');
    const validity = output('validity', 1, 'uint32');
    graph.add(
      new GPUAnalysisContiguityStatistics({
        ...options,
        polygons: {
          kind: 'polygons',
          positions: positions.view,
          ringOffsets: ringOffsets.view,
          polygonOffsets: polygonOffsets.view,
          featureOffsets: featureOffsets.view
        },
        output: {
          weights: {
            offsets: offsets.view,
            neighbors: neighbors.view,
            weights: weights.view,
            status: {
              overflow: overflow.view,
              requiredCount: requiredCount.view,
              candidateOverflow: candidateOverflow.view
            }
          },
          cardinality: cardinality.view,
          counts: counts.view,
          median: median.view,
          validity: validity.view
        }
      })
    );
    compiled = graph.compile();
    const read = async () => {
      const [
        countValues,
        medianValues,
        validityValues,
        overflowValues,
        candidateValues,
        cardinalityValues,
        requiredValues,
        offsetValues
      ] = await Promise.all([
        readUint32(counts.buffer),
        readFloat32(median.buffer),
        readUint32(validity.buffer),
        readUint32(overflow.buffer),
        readUint32(candidateOverflow.buffer),
        readUint32(cardinality.buffer),
        readUint32(requiredCount.buffer),
        readUint32(offsets.buffer)
      ]);
      return {
        summary: decodeGPUAnalysisContiguityStatistics({
          counts: countValues,
          median: medianValues,
          validity: validityValues,
          overflow: overflowValues,
          candidateOverflow: candidateValues
        }),
        counts: countValues,
        median: medianValues[0],
        validity: validityValues[0],
        cardinality: cardinalityValues,
        requiredCount: requiredValues[0],
        offsets: offsetValues
      };
    };
    submitGraph(device, compiled, undefined);
    const first = await read();
    if (options.nextPositions) {
      positions.buffer.write(options.nextPositions);
      submitGraph(device, compiled, undefined);
      return [first, await read()];
    }
    return [first];
  } finally {
    compiled?.destroy();
    for (const buffer of buffers.reverse()) buffer.destroy();
  }
}

it('reports queen/rook counts, islands and extrema on a hand-counted grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = flattenPolygons(
    [square(0, 0), square(1, 0), square(0, 1), square(1, 1), square(10, 10)].map(ring => [ring])
  );
  for (const criterion of ['queen', 'rook'] as const) {
    const degree = criterion === 'queen' ? 3 : 2;
    const [result] = await runProfile(device, polygons, {criterion, pairCapacity: 128});
    const report = `${criterion} grid profile: ${JSON.stringify(result)}`;
    expect(result.cardinality, report).toEqual([degree, degree, degree, degree, 0]);
    expect(result.counts, report).toEqual([degree * 4, 0, 1, 0, degree, 1]);
    expect(result.summary, report).toEqual({
      exact: true,
      overflow: false,
      candidateOverflow: false,
      slots: degree * 4,
      islands: 1,
      minimum: 0,
      median: degree,
      maximum: degree
    });
  }
});

it('interpolates the even median, handles all islands and reuses one compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const chain = flattenPolygons([0, 1, 2, 3].map(left => [square(left, 0)]));
  const separated = flattenPolygons([0, 3, 6, 9].map(left => [square(left, 0)]));
  for (const criterion of ['queen', 'rook'] as const) {
    const [joined, islands] = await runProfile(device, chain, {
      criterion,
      pairCapacity: 64,
      nextPositions: separated.positions
    });
    expect(joined.cardinality, JSON.stringify(joined)).toEqual([1, 2, 2, 1]);
    expect(joined.summary, JSON.stringify(joined)).toMatchObject({
      exact: true,
      minimum: 1,
      median: 1.5,
      maximum: 2,
      islands: 0
    });
    expect(islands.cardinality, JSON.stringify(islands)).toEqual([0, 0, 0, 0]);
    expect(islands.summary, JSON.stringify(islands)).toMatchObject({
      exact: true,
      slots: 0,
      minimum: 0,
      median: 0,
      maximum: 0,
      islands: 4
    });
  }
  const single = flattenPolygons([[square(0, 0)]]);
  // A feature with an empty ring: stored positions exist but belong to no ring.
  single.ringOffsets = new Uint32Array([0, 0]);
  const [emptyFeature] = await runProfile(device, single, {criterion: 'queen'});
  expect(emptyFeature.summary, JSON.stringify(emptyFeature)).toMatchObject({
    exact: true,
    slots: 0,
    islands: 1,
    median: 0
  });
});

it('counts multipart features once and includes hole boundaries and corner-only contacts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const topology = flattenFeatures([
    [[square(0, 0, 4), square(1, 1)], [square(10, 0)]],
    [[square(1, 1)]],
    [[square(11, 0)]],
    [[square(4, 4)]],
    [[square(20, 0)]]
  ]);
  for (const criterion of ['queen', 'rook'] as const) {
    const [result] = await runProfile(device, topology, {criterion, pairCapacity: 128});
    const expected = criterion === 'queen' ? [3, 1, 1, 1, 0] : [2, 1, 1, 0, 0];
    expect(result.cardinality, `${criterion} multipart/hole: ${JSON.stringify(result)}`).toEqual(
      expected
    );
    expect(result.summary, JSON.stringify(result)).toMatchObject({
      exact: true,
      islands: criterion === 'queen' ? 1 : 2,
      minimum: 0,
      median: 1,
      maximum: expected[0]
    });
  }
});

it('applies the inherited local-origin snap keys for both criteria', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Dyadic gap and tolerance avoid uncertain f32 half-cell boundaries.
  const topology = flattenPolygons([
    [square(10, 20)],
    [
      [
        [11.0625, 20],
        [12, 20],
        [12, 21],
        [11.0625, 21]
      ]
    ]
  ]);
  for (const criterion of ['queen', 'rook'] as const) {
    for (const snapTolerance of [0, 0.25]) {
      const [result] = await runProfile(device, topology, {
        criterion,
        snapTolerance,
        coordinateExtent: 2,
        pairCapacity: 32
      });
      const connected = snapTolerance > 0;
      expect(
        result.cardinality,
        `${criterion}/${snapTolerance}: ${JSON.stringify(result)}`
      ).toEqual(connected ? [1, 1] : [0, 0]);
      expect(result.summary, JSON.stringify(result)).toMatchObject({
        exact: true,
        islands: connected ? 0 : 2,
        median: connected ? 1 : 0
      });
    }
  }
});

it('refuses exact claims and poisons profile fields on candidate or final-slot overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const topology = flattenPolygons(
    [square(0, 0), square(1, 0), square(0, 1), square(1, 1)].map(ring => [ring])
  );
  const separated = flattenPolygons([0, 3, 6, 9].map(left => [square(left, 0)]));
  for (const criterion of ['queen', 'rook'] as const) {
    for (const capacities of [
      {capacity: 1, pairCapacity: 128},
      {capacity: 32, pairCapacity: 1},
      {capacity: 1, pairCapacity: 1},
      {capacity: 1, pairCapacity: 4}
    ]) {
      const [result, recovered] = await runProfile(device, topology, {
        criterion,
        ...capacities,
        nextPositions: separated.positions
      });
      const report = `${criterion}/${JSON.stringify(capacities)}: ${JSON.stringify(result)}`;
      expect(result.summary, report).toEqual({
        exact: false,
        overflow: result.requiredCount > capacities.capacity,
        candidateOverflow: capacities.pairCapacity < 128
      });
      expect(result.validity, report).toBe(0);
      expect(result.median, report).toBeNaN();
      expect(result.counts, report).toEqual(Array(6).fill(0xffffffff));
      expect(result.cardinality, report).toEqual(Array(4).fill(0xffffffff));
      if (capacities.pairCapacity === 128) {
        expect(result.requiredCount, report).toBe(criterion === 'queen' ? 12 : 8);
        expect(result.summary, report).toMatchObject({overflow: true, candidateOverflow: false});
      } else if (capacities.pairCapacity === 1) {
        // The one retained candidate fits both final capacities: intermediate overflow must
        // not be folded into the final-slot status, even though the profile remains invalid.
        expect(result.requiredCount, report).toBe(1);
        expect(result.summary, report).toMatchObject({overflow: false, candidateOverflow: true});
      } else {
        expect(result.summary, report).toMatchObject({overflow: true, candidateOverflow: true});
      }
      expect(result.offsets.at(-1), report).toBeLessThanOrEqual(capacities.capacity);
      expect(recovered.summary, `recovered ${report}: ${JSON.stringify(recovered)}`).toEqual({
        exact: true,
        overflow: false,
        candidateOverflow: false,
        slots: 0,
        islands: 4,
        minimum: 0,
        median: 0,
        maximum: 0
      });
      expect(recovered.cardinality, JSON.stringify(recovered)).toEqual([0, 0, 0, 0]);
    }
  }
});
