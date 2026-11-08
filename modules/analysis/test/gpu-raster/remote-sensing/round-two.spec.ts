// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device, Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  GPURasterConservativeRemapping,
  GPURasterTemporal,
  GPURasterTexture,
  GPURasterRegistration,
  GPURasterSeasonalBreak,
  GPURasterRegionGraph,
  addRasterRegionGraphTiles
} from '../../../src/raster-analysis/remote-sensing/index';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/raster-round-two.json';
import terraReference from '../../oracles/fixtures/raster-terra.json';

function close(actual: number[], expected: number[], tolerance = 2e-5) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index++)
    expect(Math.abs(actual[index] - expected[index]), `row ${index}`).toBeLessThanOrEqual(
      tolerance * Math.max(1, Math.abs(expected[index]))
    );
}

async function texture(
  device: Device,
  values: number[],
  validity: number[],
  width: number,
  height: number,
  offset: number[],
  symmetric: boolean
) {
  const harness = createRemoteSensingHarness(device);
  return harness.run(
    new GPURasterTexture({
      grid: makeGPURasterMetadata({width, height}),
      values: harness.input(values),
      validity: harness.mask(validity),
      levels: fixture.texture.levels,
      range: [0, fixture.texture.levels],
      radius: fixture.texture.radius,
      offset: [offset[0], offset[1]],
      symmetric,
      features: harness.output('features', 'float32', width * height * 7),
      pairCount: harness.output('counts', 'uint32', width * height)
    })
  );
}

for (const entry of fixture.texture.cases) {
  it(`RS04 direct scikit-image properties and halo equivalence: ${entry.offset}, symmetric ${entry.symmetric}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const source = fixture.texture;
    const whole = await texture(
      device,
      source.values,
      source.validity,
      source.width,
      source.height,
      entry.offset,
      entry.symmetric
    );
    close(whole['features'], entry.features);
    expect(whole['counts']).toEqual(entry.counts);
    // Split inside the masked image; only core outputs are compared, never duplicate halo cells.
    for (const [start, end] of [
      [0, 5],
      [5, source.width]
    ]) {
      const lower = Math.max(0, start - source.radius),
        upper = Math.min(source.width, end + source.radius);
      const extract = (values: number[]) =>
        Array.from({length: source.height}, (_, row) =>
          values.slice(row * source.width + lower, row * source.width + upper)
        ).flat();
      const tile = await texture(
        device,
        extract(source.values),
        extract(source.validity),
        upper - lower,
        source.height,
        entry.offset,
        entry.symmetric
      );
      for (let row = 0; row < source.height; row++)
        for (let column = start; column < end; column++) {
          const local = row * (upper - lower) + column - lower,
            global = row * source.width + column;
          close(
            tile['features'].slice(local * 7, local * 7 + 7),
            whole['features'].slice(global * 7, global * 7 + 7)
          );
          expect(tile['counts'][local]).toBe(whole['counts'][global]);
        }
    }
  }, 120000);
}

for (const [index, entry] of fixture.registration.cases.entries()) {
  it(`RS07 direct scikit-image periodic subpixel displacement ${index}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegistration({
        grid: makeGPURasterMetadata({
          width: fixture.registration.size,
          height: fixture.registration.size
        }),
        reference: harness.input(fixture.registration.reference),
        moving: harness.input(entry.values),
        upsampleFactor: 20,
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['status']).toEqual([0]);
    close(result['shift'], entry.shift, 0.025001);
  }, 120000);
}

it('RS11 selected harmonic level-break model matches pinned NumPy least squares and BIC', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device),
    source = fixture.seasonal;
  const result = await harness.run(
    new GPURasterSeasonalBreak({
      cellCount: source.cases.length,
      sliceCount: source.slices,
      period: source.period,
      minimumSegmentLength: source.minimumSegmentLength,
      values: harness.input(source.cases.flatMap(entry => entry.values)),
      validity: harness.mask(source.cases.flatMap(entry => entry.validity)),
      output: {
        coefficients: harness.output('coefficients', 'float32', source.cases.length * 5),
        components: harness.output(
          'components',
          'float32',
          source.cases.length * source.slices * 3
        ),
        breakIndex: harness.output('breakIndex', 'uint32', source.cases.length),
        improvement: harness.output('improvement', 'float32', source.cases.length),
        status: harness.output('status', 'uint32', source.cases.length)
      }
    })
  );
  expect(result['status']).toEqual(source.cases.map(() => 0));
  expect(result['breakIndex']).toEqual(source.cases.map(entry => entry.breakIndex));
  close(
    result['coefficients'],
    source.cases.flatMap(entry => entry.coefficients)
  );
  close(
    result['improvement'],
    source.cases.map(entry => entry.improvement)
  );
  source.cases.forEach((entry, cell) =>
    entry.validity.forEach((valid, slice) => {
      if (!valid) expect(result['components'][(cell * source.slices + slice) * 3 + 2]).toBeNaN();
    })
  );
}, 120000);

it('RS08 explicit tiled local-label reconciliation preserves scikit-image hierarchical partition and seam edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const source = fixture.regions;
  for (const tiled of [false, true]) {
    const harness = createRemoteSensingHarness(device);
    const props = {
      grid: makeGPURasterMetadata({width: source.width, height: source.height}),
      regionCount: 5,
      mergeMode: 'adaptive' as const,
      mergeThreshold: 0.8,
      maximumIterations: 5,
      output: {
        adjacency: harness.output('adjacency', 'uint32', 25),
        weights: harness.output('weights', 'float32', 25),
        mapping: harness.output('mapping', 'uint32', 5),
        status: harness.output('status', 'uint32', 1)
      }
    };
    if (tiled) {
      addRasterRegionGraphTiles(harness.graph, {
        ...props,
        tiles: [
          [0, 2, 0, 2],
          [2, 5, 0, 2],
          [0, 2, 2, 4],
          [2, 5, 2, 4]
        ].map(([start, end, firstRow, lastRow]) => ({
          grid: makeGPURasterMetadata({width: end - start, height: lastRow - firstRow}),
          origin: [start, firstRow] as const,
          labels: harness.mask(
            Array.from({length: lastRow - firstRow}, () =>
              Array.from({length: end - start}, (_, index) => index)
            ).flat()
          ),
          values: harness.input(
            Array.from({length: lastRow - firstRow}, (_, row) =>
              source.values.slice(
                (row + firstRow) * source.width + start,
                (row + firstRow) * source.width + end
              )
            ).flat()
          ),
          sourceRegionIds: harness.mask(
            Array.from({length: end - start}, (_, index) => start + index)
          )
        }))
      });
    }
    const result = await harness.run(
      tiled
        ? {getCommandNodes: () => []}
        : new GPURasterRegionGraph({
            ...props,
            labels: harness.mask(source.labels),
            values: harness.input(source.values)
          })
    );
    expect(result['adjacency']).toEqual(source.adjacency);
    expect(result['mapping']).toEqual(source.mapping);
    expect(result['status']).toEqual([0]);
  }
}, 120000);

it('RS02 executes 32768 dense categories with exact area and deterministic ties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device),
    count = 32768;
  const result = await harness.run(
    new GPURasterFractionalCategoricalStatistics({
      grid: makeGPURasterMetadata({width: count, height: 1}),
      input: {
        id: 'classes',
        format: 'uint32',
        storage: {
          kind: 'buffer',
          values: harness.mask(Array.from({length: count}, (_, index) => index))
        }
      },
      zoneCount: 1,
      categoryCapacity: count,
      coverage: {
        zoneIds: harness.mask(Array(count).fill(0)),
        cellIds: harness.mask(Array.from({length: count}, (_, index) => index)),
        fractions: harness.input(Array(count).fill(0.5)),
        zoneValidity: harness.mask([1]),
        status: {
          count: harness.mask([count]),
          requiredCount: harness.mask([count]),
          overflow: harness.mask([0])
        }
      },
      output: {
        coverageSums: harness.output('coverageSums', 'float32', count),
        classAreas: harness.output('classAreas', 'float32', count),
        proportions: harness.output('proportions', 'float32', count),
        coveredArea: harness.output('coveredArea', 'float32', 1),
        validArea: harness.output('validArea', 'float32', 1),
        missingArea: harness.output('missingArea', 'float32', 1),
        majority: harness.output('majority', 'uint32', 1),
        minority: harness.output('minority', 'uint32', 1),
        validity: harness.output('validity', 'uint32', 1)
      }
    })
  );
  expect(result['validity']).toEqual([1]);
  expect(result['majority']).toEqual([count - 1]);
  expect(result['minority']).toEqual([0]);
  expect(result['coveredArea']).toEqual([count / 2]);
  expect(result['missingArea']).toEqual([0]);
  expect(result['classAreas'].every(value => value === 0.5)).toBe(true);
  expect(result['proportions'].every(value => value === 1 / count)).toBe(true);
}, 120000);

for (const size of [128, 256, 512]) {
  it(`RS03 persistent prepared weights across graphs and repeated dates: ${size} squared source`, async () => {
    const testDevice = await getWebGPUTestDevice();
    requireTestDevice(testDevice);
    const device = testDevice;
    const count = size * size,
      destinationCount = count / 4,
      stride = 9;
    const buffers: Buffer[] = [];
    const allocate = (length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return buffer;
    };
    const sourceValues = Float32Array.from({length: count}, (_, index) => 1 + (index % 17) / 16);
    const sourceBuffer = createInputBuffer(device, sourceValues),
      validityBuffer = createInputBuffer(device, new Uint32Array(count).fill(1));
    buffers.push(sourceBuffer, validityBuffer);
    const indices = allocate(destinationCount * stride),
      areas = allocate(destinationCount * stride),
      values = allocate(destinationCount),
      covered = allocate(destinationCount),
      status = allocate(destinationCount);
    function build(weightMode: 'prepare' | 'reuse') {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      function view<Format extends 'uint32' | 'float32'>(
        buffer: Buffer,
        format: Format,
        length: number
      ) {
        return graph.createDataView(
          graph.importBuffer(
            {id: `prepared-${serial++}`, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          ),
          {format, length}
        );
      }
      const contributor = new GPURasterConservativeRemapping({
        source: {width: size, height: size, origin: [0, 0], spacing: [1, 1]},
        destination: {width: size / 2, height: size / 2, origin: [0, 0], spacing: [2, 2]},
        values: view(sourceBuffer, 'float32', count),
        validity: view(validityBuffer, 'uint32', count),
        normalization: 'extensive',
        preparedWeights: {
          indices: view(indices, 'uint32', destinationCount * stride),
          areas: view(areas, 'float32', destinationCount * stride)
        },
        weightMode,
        output: {
          values: view(values, 'float32', destinationCount),
          coveredArea: view(covered, 'float32', destinationCount),
          status: view(status, 'uint32', destinationCount)
        }
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBe(weightMode === 'reuse' ? 1 : 2);
      nodes.forEach(node => graph.add(node));
      return graph.compile();
    }
    const prepared = build('prepare'),
      reused = build('reuse');
    try {
      submitGraph(device, prepared, undefined);
      const original = await readFloat32(values, destinationCount);
      close(
        [original.reduce((sum, value) => sum + value, 0)],
        [sourceValues.reduce((sum, value) => sum + value, 0)],
        1e-7
      );
      const before = await readFloat32(areas, destinationCount * stride);
      const start = performance.now();
      for (let date = 1; date <= 3; date++) {
        const changed = sourceValues.map(value => value * (date + 1));
        const mask = Uint32Array.from({length: count}, (_, index) => Number(index % 11 !== date));
        sourceBuffer.write(changed);
        validityBuffer.write(mask);
        submitGraph(device, reused, undefined);
        const result = await readFloat32(values, destinationCount);
        close(
          [result.reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0)],
          [changed.reduce((sum, value, index) => sum + value * mask[index], 0)],
          1e-7
        );
        expect((await readUint32(status, destinationCount)).every(value => value === 0)).toBe(true);
      }
      expect(await readFloat32(areas, destinationCount * stride)).toEqual(before);
      console.info(
        JSON.stringify({
          rasterScale: 'prepared-remap',
          sourceCells: count,
          destinationCells: destinationCount,
          reusedDates: 3,
          wallMillisecondsIncludingReadback: performance.now() - start,
          preparedBytes: destinationCount * stride * 8
        })
      );
    } finally {
      prepared.destroy();
      reused.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  }, 120000);
}

it('RS04 maximum 32-level, radius-eight texture runs on 16384 cells with exact analytic metrics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device),
    size = 128,
    count = size * size;
  const result = await harness.run(
    new GPURasterTexture({
      grid: makeGPURasterMetadata({width: size, height: size}),
      values: harness.input(
        Array.from(
          {length: count},
          (_, index) => (((index % size) + Math.floor(index / size)) % 2) * 31
        )
      ),
      levels: 32,
      range: [0, 32],
      radius: 8,
      offset: [1, 0],
      symmetric: true,
      features: harness.output('features', 'float32', count * 7),
      pairCount: harness.output('pairCount', 'uint32', count)
    })
  );
  const center = 64 * size + 64;
  close(result['features'].slice(center * 7, center * 7 + 7), [
    961,
    31,
    1 / 962,
    0.5,
    Math.sqrt(0.5),
    -1,
    Math.log(2)
  ]);
  expect(result['pairCount'][center]).toBe(544);
  expect(result['pairCount'][0]).toBe(144);
}, 120000);

it('RS08 64-region adjacency and bounded propagation scale to 16384 source cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device),
    size = 128,
    count = size * size;
  const labels = Array.from(
    {length: count},
    (_, index) => Math.floor((index % size) / 16) + 8 * Math.floor(Math.floor(index / size) / 16)
  );
  const result = await harness.run(
    new GPURasterRegionGraph({
      grid: makeGPURasterMetadata({width: size, height: size}),
      labels: harness.mask(labels),
      values: harness.input(labels.map(label => label % 8)),
      regionCount: 64,
      mergeThreshold: 0,
      maximumIterations: 8,
      output: {
        adjacency: harness.output('adjacency', 'uint32', 4096),
        weights: harness.output('weights', 'float32', 4096),
        mapping: harness.output('mapping', 'uint32', 64),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(result['mapping']).toEqual(Array.from({length: 64}, (_, index) => index % 8));
  expect(result['status']).toEqual([0]);
  for (let first = 0; first < 64; first++)
    for (let second = 0; second < 64; second++) {
      const adjacent =
        (Math.floor(first / 8) === Math.floor(second / 8) && Math.abs(first - second) === 1) ||
        Math.abs(first - second) === 8;
      expect(result['adjacency'][first * 64 + second]).toBe(adjacent ? 16 : 0);
    }
}, 120000);

for (const operation of ['mean', 'minimum', 'maximum', 'median'] as const) {
  it(`RS06 direct terra stack composite: ${operation}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device),
      source = terraReference.temporal;
    const result = await harness.run(
      new GPURasterTemporal({
        cellCount: source.cellCount,
        sliceCount: source.sliceCount,
        operation,
        values: harness.input(source.values.map(value => value ?? NaN)),
        output: {
          values: harness.output('values', 'float32', source.cellCount),
          counts: harness.output('counts', 'uint32', source.cellCount),
          chosenSlice: harness.output('chosenSlice', 'uint32', source.cellCount),
          status: harness.output('status', 'uint32', source.cellCount)
        }
      })
    );
    close(result['values'], source.expected[operation]);
    expect(result['counts']).toEqual([3, 3, 4]);
    expect(result['status']).toEqual([0, 0, 0]);
  });
}
