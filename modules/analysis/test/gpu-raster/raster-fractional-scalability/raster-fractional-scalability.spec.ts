// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues,
  type GPURasterFractionalCoverageProps
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  const inputBuffers = new Map<GraphDataView, ReturnType<typeof createInputBuffer>>();
  let identifier = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const view = importGraphBuffer(
      graph,
      `fractional-input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
    inputBuffers.set(view, buffer);
    return view;
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    const view = importGraphBuffer(
      graph,
      `fractional-output-${identifier++}`,
      buffer,
      format,
      length
    );
    return {
      view,
      buffer,
      read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
    };
  };
  const coverageOutput = (zoneCount: number, capacity: number) => {
    const result = {
      zoneIds: output(capacity, 'uint32'),
      cellIds: output(capacity, 'uint32'),
      fractions: output(capacity, 'float32'),
      zoneValidity: output(zoneCount, 'uint32'),
      status: {
        count: output(1, 'uint32'),
        requiredCount: output(1, 'uint32'),
        overflow: output(1, 'uint32'),
        candidateOverflow: output(1, 'uint32')
      }
    };
    const views = {
      zoneIds: result.zoneIds.view,
      cellIds: result.cellIds.view,
      fractions: result.fractions.view,
      zoneValidity: result.zoneValidity.view,
      status: {
        count: result.status.count.view,
        requiredCount: result.status.requiredCount.view,
        overflow: result.status.overflow.view,
        candidateOverflow: result.status.candidateOverflow.view
      }
    };
    return {result, views};
  };
  const categoryOutput = (zoneCount: number) => {
    const result = {
      coverageSums: output(zoneCount * 2, 'float32'),
      classAreas: output(zoneCount * 2, 'float32'),
      proportions: output(zoneCount * 2, 'float32'),
      coveredArea: output(zoneCount, 'float32'),
      validArea: output(zoneCount, 'float32'),
      missingArea: output(zoneCount, 'float32'),
      majority: output(zoneCount, 'uint32'),
      minority: output(zoneCount, 'uint32'),
      validity: output(zoneCount, 'uint32')
    };
    const views = {
      coverageSums: result.coverageSums.view,
      classAreas: result.classAreas.view,
      proportions: result.proportions.view,
      coveredArea: result.coveredArea.view,
      validArea: result.validArea.view,
      missingArea: result.missingArea.view,
      majority: result.majority.view,
      minority: result.minority.view,
      validity: result.validity.view
    };
    return {result, views};
  };
  return {
    graph,
    input,
    output,
    coverageOutput,
    categoryOutput,
    writeInput: (view: GraphDataView, values: Uint32Array | Float32Array, byteOffset = 0) =>
      inputBuffers.get(view)!.write(values, byteOffset),
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

it('100k single-cell polygons over 4096x4096 have exact support and explicit missing area', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const context = createContext(device);
  const {graph, input} = context;
  const grid: GPURasterFractionalCoverageProps['grid'] = {
    width: 4096,
    height: 4096,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  const zoneCount = 100000;
  const positions = new Float32Array(zoneCount * 8);
  const offsets = Uint32Array.from({length: zoneCount + 1}, (_, index) => index);
  const ringOffsets = Uint32Array.from({length: zoneCount + 1}, (_, index) => index * 4);
  const cells = new Uint32Array(zoneCount);
  const categories = new Uint32Array(grid.width * grid.height);
  for (let zone = 0; zone < zoneCount; zone++) {
    const column = (zone % 320) * 12 + 2;
    const row = Math.floor(zone / 320) * 12 + 2;
    positions.set([column, row, column + 1, row, column + 1, row + 1, column, row + 1], zone * 8);
    cells[zone] = row * grid.width + column;
    categories[cells[zone]] = zone % 5 === 0 ? 0xffffffff : zone % 2;
  }
  const coverage = context.coverageOutput(zoneCount, zoneCount);
  const parameters = input(getGPURasterFractionalCoverageParameterValues(grid), 'float32');
  graph.add(
    new GPURasterFractionalCoverage({
      grid,
      parameters,
      zones: {
        positions: input(positions, 'float32x2'),
        featureOffsets: input(offsets, 'uint32'),
        polygonOffsets: input(offsets, 'uint32'),
        ringOffsets: input(ringOffsets, 'uint32')
      },
      output: coverage.views
    })
  );
  const statistics = context.categoryOutput(zoneCount);
  graph.add(
    new GPURasterFractionalCategoricalStatistics({
      grid,
      parameters,
      zoneCount,
      categoryCapacity: 2,
      coverage: coverage.views,
      input: {
        id: 'categories',
        format: 'uint32',
        noDataValue: 0xffffffff,
        storage: {kind: 'buffer', values: input(categories, 'uint32')}
      },
      output: statistics.views
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await coverage.result.status.count.read()).toEqual([zoneCount]);
    expect(await coverage.result.status.requiredCount.read()).toEqual([zoneCount]);
    expect(await coverage.result.status.overflow.read()).toEqual([0]);
    expect(await coverage.result.status.candidateOverflow.read()).toEqual([0]);
    expect(await coverage.result.cellIds.read()).toEqual(Array.from(cells));
    expect(await coverage.result.zoneIds.read()).toEqual(
      Array.from(offsets.subarray(0, zoneCount))
    );
    expect((await coverage.result.fractions.read()).every(value => value === 1)).toBe(true);
    expect((await coverage.result.zoneValidity.read()).every(value => value === 1)).toBe(true);
    const [covered, observed, missing, majority, validity, supports] = await Promise.all([
      statistics.result.coveredArea.read(),
      statistics.result.validArea.read(),
      statistics.result.missingArea.read(),
      statistics.result.majority.read(),
      statistics.result.validity.read(),
      statistics.result.coverageSums.read()
    ]);
    for (let zone = 0; zone < zoneCount; zone++) {
      const isMissing = zone % 5 === 0;
      expect(covered[zone]).toBe(1);
      expect(observed[zone]).toBe(isMissing ? 0 : 1);
      expect(missing[zone]).toBe(isMissing ? 1 : 0);
      expect(validity[zone]).toBe(isMissing ? 0 : 1);
      expect(majority[zone]).toBe(isMissing ? 0xffffffff : zone % 2);
      expect(supports[zone * 2]).toBe(!isMissing && zone % 2 === 0 ? 1 : 0);
      expect(supports[zone * 2 + 1]).toBe(!isMissing && zone % 2 === 1 ? 1 : 0);
    }
  } finally {
    compiled.destroy();
    context.destroy();
  }
}, 120000);

it('more than 1M unique tuples merge missing and categorical support before normalisation', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const context = createContext(device);
  const {graph, input} = context;
  const grid: GPURasterFractionalCoverageProps['grid'] = {
    width: 2048,
    height: 513,
    affine: [2, 0, 0, 0, 3, 0],
    pixelInterpretation: 'area'
  };
  const tupleCount = 1048579;
  const values = Uint32Array.from({length: grid.width * grid.height}, (_, index) =>
    index % 3 === 0 ? 0xffffffff : index % 2
  );
  const cells = Uint32Array.from({length: tupleCount}, (_, index) => index);
  const fractions = new Float32Array(tupleCount).fill(0.5);
  const status = {
    count: input(Uint32Array.of(tupleCount), 'uint32'),
    requiredCount: input(Uint32Array.of(tupleCount), 'uint32'),
    overflow: input(Uint32Array.of(0), 'uint32')
  };
  const coverage = {
    zoneIds: input(new Uint32Array(tupleCount), 'uint32'),
    cellIds: input(cells, 'uint32'),
    fractions: input(fractions, 'float32'),
    zoneValidity: input(Uint32Array.of(1), 'uint32'),
    status
  };
  const categoryValues = input(values, 'uint32');
  const statistics = context.categoryOutput(1);
  graph.add(
    new GPURasterFractionalCategoricalStatistics({
      grid,
      zoneCount: 1,
      categoryCapacity: 2,
      coverage,
      input: {
        id: 'categories',
        format: 'uint32',
        noDataValue: 0xffffffff,
        storage: {kind: 'buffer', values: categoryValues}
      },
      output: statistics.views
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Six consecutive cells contain two missing cells and two of each category.
    const completeCycles = Math.floor(tupleCount / 6);
    const tail = tupleCount % 6;
    const categoryZero = completeCycles * 2 + Number(tail > 2) + Number(tail > 4);
    const categoryOne = completeCycles * 2 + Number(tail > 1) + Number(tail > 5);
    const missing = completeCycles * 2 + Number(tail > 0) + Number(tail > 3);
    expect(await statistics.result.coverageSums.read()).toEqual([
      categoryZero * 0.5,
      categoryOne * 0.5
    ]);
    expect(await statistics.result.classAreas.read()).toEqual([categoryZero * 3, categoryOne * 3]);
    expect(await statistics.result.missingArea.read()).toEqual([missing * 3]);
    expect(await statistics.result.coveredArea.read()).toEqual([tupleCount * 3]);
    expect(await statistics.result.validArea.read()).toEqual([(categoryZero + categoryOne) * 3]);
    const proportions = await statistics.result.proportions.read();
    expect(proportions[0]).toBeCloseTo(categoryZero / (categoryZero + categoryOne), 6);
    expect(proportions[1]).toBeCloseTo(categoryOne / (categoryZero + categoryOne), 6);
    expect(await statistics.result.majority.read()).toEqual([categoryZero > categoryOne ? 0 : 1]);
    expect(await statistics.result.minority.read()).toEqual([categoryZero <= categoryOne ? 0 : 1]);
    expect(await statistics.result.validity.read()).toEqual([1]);
    // A malformed active tuple in the final partial tile must invalidate every zone.
    context.writeInput(
      coverage.cellIds,
      Uint32Array.of(grid.width * grid.height),
      (tupleCount - 1) * 4
    );
    submitGraph(device, compiled, undefined);
    expect(await statistics.result.validity.read()).toEqual([0]);
    expect((await statistics.result.coverageSums.read()).every(Number.isNaN)).toBe(true);
    expect((await statistics.result.missingArea.read()).every(Number.isNaN)).toBe(true);
    expect((await statistics.result.proportions.read()).every(Number.isNaN)).toBe(true);
    expect(await statistics.result.majority.read()).toEqual([0xffffffff]);
    expect(await statistics.result.minority.read()).toEqual([0xffffffff]);
    // Restore the tuple and change the frame to all category 1, with no missing support.
    context.writeInput(coverage.cellIds, Uint32Array.of(tupleCount - 1), (tupleCount - 1) * 4);
    context.writeInput(categoryValues, new Uint32Array(values.length).fill(1));
    submitGraph(device, compiled, undefined);
    expect(await statistics.result.coverageSums.read()).toEqual([0, tupleCount * 0.5]);
    expect(await statistics.result.missingArea.read()).toEqual([0]);
    expect(await statistics.result.validArea.read()).toEqual([tupleCount * 3]);
    expect(await statistics.result.proportions.read()).toEqual([0, 1]);
    expect(await statistics.result.majority.read()).toEqual([1]);
    expect(await statistics.result.minority.read()).toEqual([1]);
    expect(await statistics.result.validity.read()).toEqual([1]);
  } finally {
    compiled.destroy();
    context.destroy();
  }
}, 120000);

it('indexed and brute-force clipping match a closed-form partial-cell rectangle and report exhaustion separately', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const context = createContext(device);
  const {graph, input} = context;
  const grid: GPURasterFractionalCoverageProps['grid'] = {
    width: 3,
    height: 2,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  const zones = {
    positions: input(Float32Array.of(0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1), 'float32x2'),
    featureOffsets: input(Uint32Array.of(0, 1), 'uint32'),
    polygonOffsets: input(Uint32Array.of(0, 1), 'uint32'),
    ringOffsets: input(Uint32Array.of(0, 4), 'uint32')
  };
  const parameters = input(getGPURasterFractionalCoverageParameterValues(grid), 'float32');
  const outputs = [
    context.coverageOutput(1, 6),
    context.coverageOutput(1, 6),
    context.coverageOutput(1, 6),
    context.coverageOutput(1, 1)
  ];
  for (const [index, searchStructure] of ['bounds', 'brute-force', 'bounds', 'bounds'].entries()) {
    graph.add(
      new GPURasterFractionalCoverage({
        id: `rectangle-${index}`,
        grid,
        zones,
        parameters,
        searchStructure: searchStructure as 'bounds' | 'brute-force',
        candidateCapacity: index === 2 ? 1 : undefined,
        output: outputs[index].views
      })
    );
  }
  const categoryResults = ['tiled', 'exact'].map((mode, index) => {
    const statistics = context.categoryOutput(1);
    graph.add(
      new GPURasterFractionalCategoricalStatistics({
        id: `rectangle-categories-${index}`,
        grid,
        parameters,
        zoneCount: 1,
        categoryCapacity: 2,
        mode: mode as 'tiled' | 'exact',
        tupleTileSize: mode === 'tiled' ? 1 : undefined,
        coverage: outputs[0].views,
        input: {
          id: 'rectangle-values',
          format: 'uint32',
          noDataValue: 0xffffffff,
          storage: {
            kind: 'buffer',
            values: input(Uint32Array.of(0, 0xffffffff, 1, 0, 0, 0), 'uint32')
          }
        },
        output: statistics.views
      })
    );
    return statistics.result;
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const {result} of outputs.slice(0, 2)) {
      expect(await result.status.count.read()).toEqual([2]);
      expect(await result.status.requiredCount.read()).toEqual([2]);
      expect((await result.cellIds.read()).slice(0, 2)).toEqual([0, 1]);
      expect((await result.fractions.read()).slice(0, 2)).toEqual([0.5, 0.5]);
      expect(await result.zoneValidity.read()).toEqual([1]);
      expect(await result.status.candidateOverflow.read()).toEqual([0]);
    }
    for (const statistics of categoryResults) {
      expect(await statistics.coverageSums.read()).toEqual([0.5, 0]);
      expect(await statistics.classAreas.read()).toEqual([0.5, 0]);
      const proportions = await statistics.proportions.read();
      expect(proportions).toEqual([1, 0]);
      expect(await statistics.coveredArea.read()).toEqual([1]);
      expect(await statistics.validArea.read()).toEqual([0.5]);
      expect(await statistics.missingArea.read()).toEqual([0.5]);
      expect(await statistics.majority.read()).toEqual([0]);
      expect(await statistics.minority.read()).toEqual([0]);
      expect(await statistics.validity.read()).toEqual([1]);
    }
    const exhausted = outputs[2].result;
    expect(await exhausted.status.candidateOverflow.read()).toEqual([1]);
    expect(await exhausted.status.overflow.read()).toEqual([0]);
    expect(await exhausted.status.requiredCount.read()).toEqual([0xffffffff]);
    expect(await exhausted.zoneValidity.read()).toEqual([0]);
    const truncated = outputs[3].result;
    expect(await truncated.status.count.read()).toEqual([1]);
    expect(await truncated.status.requiredCount.read()).toEqual([2]);
    expect(await truncated.status.overflow.read()).toEqual([1]);
    expect(await truncated.status.candidateOverflow.read()).toEqual([0]);
    expect(await truncated.cellIds.read()).toEqual([0]);
    expect(await truncated.fractions.read()).toEqual([0.5]);
  } finally {
    compiled.destroy();
    context.destroy();
  }
});

it('saturates a bounds index with 2^32 candidates without wrapping its total to zero', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const context = createContext(device);
  const {graph, input} = context;
  const grid: GPURasterFractionalCoverageProps['grid'] = {
    width: 65536,
    height: 32768,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  const rectangle = [0, 0, 65536, 0, 65536, 32768, 0, 32768];
  const coverage = context.coverageOutput(2, 1);
  graph.add(
    new GPURasterFractionalCoverage({
      grid,
      candidateCapacity: 1,
      parameters: input(getGPURasterFractionalCoverageParameterValues(grid), 'float32'),
      zones: {
        positions: input(Float32Array.from([...rectangle, ...rectangle]), 'float32x2'),
        featureOffsets: input(Uint32Array.of(0, 1, 2), 'uint32'),
        polygonOffsets: input(Uint32Array.of(0, 1, 2), 'uint32'),
        ringOffsets: input(Uint32Array.of(0, 4, 8), 'uint32')
      },
      output: coverage.views
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await coverage.result.status.candidateOverflow.read()).toEqual([1]);
    expect(await coverage.result.status.requiredCount.read()).toEqual([0xffffffff]);
    expect(await coverage.result.zoneValidity.read()).toEqual([0, 0]);
  } finally {
    compiled.destroy();
    context.destroy();
  }
});

function createGeometryViews(
  context: ReturnType<typeof createContext>,
  positions: readonly number[],
  ringOffsets: readonly number[],
  polygonOffsets: readonly number[],
  featureOffsets: readonly number[]
) {
  return {
    positions: context.input(Float32Array.from(positions), 'float32x2'),
    ringOffsets: context.input(Uint32Array.from(ringOffsets), 'uint32'),
    polygonOffsets: context.input(Uint32Array.from(polygonOffsets), 'uint32'),
    featureOffsets: context.input(Uint32Array.from(featureOffsets), 'uint32')
  };
}

for (const dimensionCase of [
  {
    width: 16777217,
    height: 1,
    rectangle: [16777216, 0, 16777218, 0, 16777218, 1, 16777216, 1],
    candidateCapacity: 8,
    capacity: 4,
    cells: [16777216],
    required: 1,
    exhausted: 0,
    validity: 1
  },
  {
    width: 1431655765,
    height: 3,
    rectangle: [0, 0, 1431655765, 0, 1431655765, 3, 0, 3],
    candidateCapacity: 128,
    capacity: 128,
    cells: Array.from({length: 128}, (_, index) => index),
    required: 0xffffffff,
    exhausted: 1,
    validity: 0
  }
]) {
  it(`clamps integer bounds for ${dimensionCase.width}x${dimensionCase.height} without dropping cells or wrapping counts`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const context = createContext(device);
    const grid: GPURasterFractionalCoverageProps['grid'] = {
      width: dimensionCase.width,
      height: dimensionCase.height,
      affine: [1, 0, 0, 0, 1, 0],
      pixelInterpretation: 'area'
    };
    const coverage = context.coverageOutput(1, dimensionCase.capacity);
    context.graph.add(
      new GPURasterFractionalCoverage({
        grid,
        candidateCapacity: dimensionCase.candidateCapacity,
        parameters: context.input(getGPURasterFractionalCoverageParameterValues(grid), 'float32'),
        zones: createGeometryViews(context, dimensionCase.rectangle, [0, 4], [0, 1], [0, 1]),
        output: coverage.views
      })
    );
    const compiled = context.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await coverage.result.status.count.read()).toEqual([dimensionCase.cells.length]);
      expect(await coverage.result.status.requiredCount.read()).toEqual([dimensionCase.required]);
      expect(await coverage.result.status.candidateOverflow.read()).toEqual([
        dimensionCase.exhausted
      ]);
      expect(await coverage.result.status.overflow.read()).toEqual([0]);
      expect(await coverage.result.zoneValidity.read()).toEqual([dimensionCase.validity]);
      expect((await coverage.result.cellIds.read()).slice(0, dimensionCase.cells.length)).toEqual(
        dimensionCase.cells
      );
      if (dimensionCase.exhausted === 0)
        expect((await coverage.result.fractions.read())[0]).toBe(1);
    } finally {
      compiled.destroy();
      context.destroy();
    }
  });
}

for (const candidateCount of [262143, 262144, 262145]) {
  it(`clips ${candidateCount} candidates across the scratch tile boundary with stable truncation`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const context = createContext(device);
    const grid: GPURasterFractionalCoverageProps['grid'] = {
      width: candidateCount,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0],
      pixelInterpretation: 'area'
    };
    const capacity = Math.min(candidateCount, 262144);
    const coverage = context.coverageOutput(1, capacity);
    const zones = createGeometryViews(
      context,
      [0, 0, candidateCount, 0, candidateCount, 1, 0, 1],
      [0, 4],
      [0, 1],
      [0, 1]
    );
    context.graph.add(
      new GPURasterFractionalCoverage({
        grid,
        candidateCapacity: candidateCount,
        parameters: context.input(getGPURasterFractionalCoverageParameterValues(grid), 'float32'),
        zones,
        output: coverage.views
      })
    );
    const compiled = context.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await coverage.result.status.count.read()).toEqual([capacity]);
      expect(await coverage.result.status.requiredCount.read()).toEqual([candidateCount]);
      expect(await coverage.result.status.overflow.read()).toEqual([
        Number(candidateCount > capacity)
      ]);
      expect(await coverage.result.status.candidateOverflow.read()).toEqual([0]);
      expect(await coverage.result.zoneValidity.read()).toEqual([1]);
      expect(await coverage.result.cellIds.read()).toEqual(
        Array.from({length: capacity}, (_, index) => index)
      );
      const fractions = await coverage.result.fractions.read();
      expect(fractions.every(value => Math.abs(value - 1) <= 1e-5)).toBe(true);
      // Reuse must clear the previous full tile's flags before its new partial support is scanned.
      context.writeInput(zones.positions, Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1));
      submitGraph(device, compiled, undefined);
      expect(await coverage.result.status.count.read()).toEqual([1]);
      expect(await coverage.result.status.requiredCount.read()).toEqual([1]);
      expect(await coverage.result.status.overflow.read()).toEqual([0]);
      expect((await coverage.result.cellIds.read())[0]).toBe(0);
      expect((await coverage.result.fractions.read())[0]).toBe(1);
    } finally {
      compiled.destroy();
      context.destroy();
    }
  }, 120000);
}

it('keeps empty and failed zones between valid zones when candidate offsets repeat', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const context = createContext(device);
  const grid: GPURasterFractionalCoverageProps['grid'] = {
    width: 4,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0],
    pixelInterpretation: 'area'
  };
  const coverage = context.coverageOutput(5, 4);
  const zones = createGeometryViews(
    context,
    [0, 0, 1, 0, 1, 1, 0, 1, NaN, 0, 2, 0, 2, 1, 1, 1, 3, 0, 4, 0, 4, 1, 3, 1],
    [0, 4, 8, 12],
    [0, 1, 2, 3],
    [0, 1, 1, 2, 2, 3]
  );
  context.graph.add(
    new GPURasterFractionalCoverage({
      grid,
      zones,
      parameters: context.input(getGPURasterFractionalCoverageParameterValues(grid), 'float32'),
      output: coverage.views
    })
  );
  const compiled = context.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await coverage.result.status.requiredCount.read()).toEqual([2]);
    expect(await coverage.result.status.candidateOverflow.read()).toEqual([0]);
    expect(await coverage.result.zoneValidity.read()).toEqual([1, 1, 0, 1, 1]);
    expect((await coverage.result.zoneIds.read()).slice(0, 2)).toEqual([0, 4]);
    expect((await coverage.result.cellIds.read()).slice(0, 2)).toEqual([0, 3]);
    expect((await coverage.result.fractions.read()).slice(0, 2)).toEqual([1, 1]);
  } finally {
    compiled.destroy();
    context.destroy();
  }
});

for (const geometryMode of ['area', 'point', 'rotated'] as const) {
  it(`preserves holes and multipart ${geometryMode} geometry against closed forms and brute force`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const context = createContext(device);
    const grid: GPURasterFractionalCoverageProps['grid'] = {
      width: 4,
      height: 4,
      affine: geometryMode === 'rotated' ? [2, 1, 10, -1, 2, 20] : [1, 0, 0, 0, 1, 0],
      pixelInterpretation: geometryMode === 'point' ? 'point' : 'area'
    };
    const pixelPositions = [
      0, 0, 3, 0, 3, 3, 0, 3, 1, 1, 1, 2, 2, 2, 2, 1, 3, 0, 4, 0, 4, 1, 3, 1, 0, 3, 1, 3, 1, 4, 0,
      4, 0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75
    ];
    const positions: number[] = [];
    for (let vertex = 0; vertex < pixelPositions.length; vertex += 2) {
      const column = pixelPositions[vertex] - (geometryMode === 'point' ? 0.5 : 0);
      const row = pixelPositions[vertex + 1] - (geometryMode === 'point' ? 0.5 : 0);
      positions.push(
        grid.affine[0] * column + grid.affine[1] * row + grid.affine[2],
        grid.affine[3] * column + grid.affine[4] * row + grid.affine[5]
      );
    }
    const zones = createGeometryViews(
      context,
      positions,
      [0, 4, 8, 12, 16, 20],
      [0, 2, 3, 4, 5],
      [0, 1, 3, 4]
    );
    const parameters = context.input(
      getGPURasterFractionalCoverageParameterValues(grid),
      'float32'
    );
    const outputs = (['bounds', 'brute-force'] as const).map(searchStructure => {
      const coverage = context.coverageOutput(3, 48);
      context.graph.add(
        new GPURasterFractionalCoverage({
          id: `geometry-${searchStructure}`,
          grid,
          zones,
          parameters,
          searchStructure,
          output: coverage.views
        })
      );
      return coverage.result;
    });
    const compiled = context.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // A 3x3 shell minus its center; two disjoint one-cell parts; one quarter-cell square.
      const expectedCells = [0, 1, 2, 4, 6, 8, 9, 10, 3, 12, 0];
      const expectedZones = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2];
      for (const coverage of outputs) {
        expect(await coverage.status.count.read()).toEqual([11]);
        expect(await coverage.status.requiredCount.read()).toEqual([11]);
        expect(await coverage.status.candidateOverflow.read()).toEqual([0]);
        expect(await coverage.zoneValidity.read()).toEqual([1, 1, 1]);
        expect((await coverage.cellIds.read()).slice(0, 11)).toEqual(expectedCells);
        expect((await coverage.zoneIds.read()).slice(0, 11)).toEqual(expectedZones);
        const fractions = (await coverage.fractions.read()).slice(0, 11);
        for (let tuple = 0; tuple < 11; tuple++)
          expect(fractions[tuple]).toBeCloseTo(tuple === 10 ? 0.25 : 1, 5);
      }
    } finally {
      compiled.destroy();
      context.destroy();
    }
  });
}
