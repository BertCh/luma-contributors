// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import reference from '../../oracles/terrain-upstream.json';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainBreach} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {GPUTerrainDepressionHierarchy} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const grid = (width: number, height = 1) => ({
  width,
  height,
  affine: [1, 0, 0, 0, 1, 0] as const,
  pixelInterpretation: 'area' as const
});
async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  const input = <Format extends 'float32' | 'uint32'>(values: number[], format: Format) => {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? Float32Array.from(values) : Uint32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${identifier++}`, buffer, format, values.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length),
      async read() {
        return Array.from(
          await (format === 'float32' ? readFloat32(buffer, length) : readUint32(buffer, length))
        );
      }
    };
  };
  return {
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

function makeRouting(width: number, height: number) {
  const count = width * height;
  const fractions = Array(count * 8).fill(0);
  const directions = Array(count).fill(0);
  for (let row = 0; row < height; row++)
    for (let column = 0; column < width; column++) {
      const cell = row * width + column;
      if (column < width - 1) fractions[cell * 8] = row === height - 1 ? 1 : 0.375;
      if (row < height - 1) fractions[cell * 8 + 2] = column === width - 1 ? 1 : 0.625;
      directions[cell] = column < width - 1 ? 1 : row < height - 1 ? 4 : 0;
    }
  return {fractions, directions};
}

async function compareTileRouting(
  width: number,
  height: number,
  routing: 'd8' | 'branching',
  expected?: number[]
) {
  const context = await setup();
  const {graph, input, output, run} = context;
  const count = width * height;
  const prepared = makeRouting(width, height);
  const whole = output(count, 'float32');
  const wholeStatus = output(count, 'uint32');
  const wholeFractions =
    routing === 'branching'
      ? prepared.fractions
      : prepared.fractions.map((_, offset) =>
          Number(prepared.directions[Math.floor(offset / 8)] === 1 << (offset % 8))
        );
  graph.add(
    new GPUTerrainBranchingFlow({
      id: 'whole',
      grid: grid(width, height),
      fractions: input(wholeFractions, 'float32'),
      values: input(Array(count).fill(1), 'float32'),
      mode: 'transport',
      output: {
        values: whole.view,
        status: wholeStatus.view,
        unresolvedCount: output(1, 'uint32').view
      }
    })
  );
  const captures: {
    column: number;
    row: number;
    width: number;
    height: number;
    values: ReturnType<typeof output>;
    status: ReturnType<typeof output>;
    reach: ReturnType<typeof output>;
  }[] = [];
  const tiles = [];
  for (let tileRow = 0; tileRow < 2; tileRow++)
    for (let tileColumn = 0; tileColumn < 2; tileColumn++) {
      const ownedWidth = width / 2,
        ownedHeight = height / 2;
      const column = tileColumn * ownedWidth - 1,
        row = tileRow * ownedHeight - 1;
      const tileWidth = ownedWidth + 2,
        tileHeight = ownedHeight + 2;
      const tileCount = tileWidth * tileHeight;
      const fractions = Array(tileCount * 8).fill(Number.NaN);
      const directions = Array(tileCount).fill(0xffffffff);
      const runoff = Array(tileCount).fill(1000000);
      const streams = Array(tileCount).fill(0xffffffff);
      for (let localRow = 1; localRow <= ownedHeight; localRow++)
        for (let localColumn = 1; localColumn <= ownedWidth; localColumn++) {
          const local = localRow * tileWidth + localColumn;
          const global = (row + localRow) * width + column + localColumn;
          directions[local] = prepared.directions[global];
          runoff[local] = 1;
          streams[local] = 1;
          for (let direction = 0; direction < 8; direction++)
            fractions[local * 8 + direction] = prepared.fractions[global * 8 + direction];
        }
      const values = output(tileCount, 'float32'),
        status = output(tileCount, 'uint32'),
        reach = output(tileCount, 'uint32');
      captures.push({column, row, width: tileWidth, height: tileHeight, values, status, reach});
      tiles.push({
        column,
        row,
        width: tileWidth,
        height: tileHeight,
        owned: [1, 1, ownedWidth, ownedHeight] as const,
        flowDirections: input(directions, 'uint32'),
        ...(routing === 'branching' ? {fractions: input(fractions, 'float32')} : {}),
        runoff: input(runoff, 'float32'),
        streams: input(streams, 'uint32'),
        output: {accumulation: values.view, status: status.view, reachIds: reach.view}
      });
    }
  graph.add(new GPUTerrainTiledFlow({id: 'tiled', grid: grid(width, height), routing, tiles}));
  const started = performance.now();
  const destroy = run();
  try {
    const wholeValues = await whole.read();
    expect((await wholeStatus.read()).every(value => value === 1)).toBe(true);
    expect(Math.abs(wholeValues[count - 1] - count) / count).toBeLessThan(0.000005);
    for (let cell = 0; cell < count; cell++) {
      if (expected)
        expect(Math.abs(wholeValues[cell] - expected[cell])).toBeLessThan(
          0.0001 * Math.max(1, expected[cell])
        );
      else if (routing === 'd8')
        expect(wholeValues[cell]).toBe(
          cell % width === width - 1 ? (Math.floor(cell / width) + 1) * width : (cell % width) + 1
        );
    }
    for (const capture of captures) {
      const values = await capture.values.read(),
        status = await capture.status.read(),
        reach = await capture.reach.read();
      for (let localRow = 0; localRow < capture.height; localRow++)
        for (let localColumn = 0; localColumn < capture.width; localColumn++) {
          const local = localRow * capture.width + localColumn;
          if (
            localRow === 0 ||
            localColumn === 0 ||
            localRow === capture.height - 1 ||
            localColumn === capture.width - 1
          ) {
            expect(status[local]).toBe(0);
            expect(Number.isNaN(values[local])).toBe(true);
            expect(reach[local]).toBe(0xffffffff);
          } else {
            const global = (capture.row + localRow) * width + capture.column + localColumn;
            expect(status[local]).toBe(1);
            expect(values[local]).toBe(wholeValues[global]);
            expect(reach[local]).toBeLessThan(count);
          }
        }
    }
    console.info(
      `terrain-scale ${routing} ${count} cells: compile/submit/readback/assertions ${Math.round(performance.now() - started)} ms`
    );
  } finally {
    destroy();
  }
}

it('matches executed pysheds D8 and MFD accumulation across four poisoned-halo tiles', async () => {
  await compareTileRouting(
    reference.routing.width,
    reference.routing.height,
    'd8',
    reference.routing.d8
  );
  await compareTileRouting(
    reference.routing.width,
    reference.routing.height,
    'branching',
    reference.routing.branching
  );
}, 120000);

it('preserves whole-domain D8 and branching mass on 65536 cells with four tile seams', async () => {
  await compareTileRouting(256, 256, 'd8');
  await compareTileRouting(256, 256, 'branching');
}, 120000);

it('matches executed xDEM median vertical shift and residual NMAD on masked outliers', async () => {
  const {graph, input, output, run} = await setup();
  const count = 1024;
  const referenceDEM = Array.from({length: count}, (_, index) => index * 0.125);
  const residual = Array.from(
    {length: count},
    (_, index) => (((index * 7) % 17) - 8) * 0.25 + 3 + (index % 61 === 0 ? 30 : 0)
  );
  const fit = output(4, 'float32'),
    differences = output(count * 2, 'float32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      grid: grid(32, 32),
      reference: input(referenceDEM, 'float32'),
      moving: input(
        referenceDEM.map((value, index) => value + residual[index]),
        'float32'
      ),
      stableMask: input(
        residual.map((_, index) => Number(index % 13 !== 0)),
        'uint32'
      ),
      differenceVariance: input(Array(count).fill(1), 'float32'),
      output: {
        fit: fit.view,
        differences: differences.view,
        validity: output(count, 'uint32').view,
        volume: volume.view
      }
    })
  );
  const destroy = run();
  try {
    const actualFit = await fit.read(),
      actualDifference = await differences.read();
    expect(actualFit[0]).toBe(reference.vertical.bias);
    expect(actualFit[1]).toBeCloseTo(reference.vertical.nmad, 5);
    expect(actualFit[2]).toBe(reference.vertical.stableCount);
    expect(actualDifference.filter((_, index) => index % 2 === 0)).toEqual(
      reference.vertical.corrected
    );
    expect((await volume.read())[0]).toBe(reference.vertical.volume);
  } finally {
    destroy();
  }
});

it('matches SciPy sublevel connected-component areas and volumes for a nested 768-cell hierarchy', async () => {
  const {graph, input, output, run} = await setup();
  const fixture = reference.hierarchy;
  const count = fixture.elevation.length;
  const nodes = output(count * 4, 'uint32'),
    levels = output(count * 2, 'float32'),
    curves = output(count * fixture.stages.length * 2, 'float32'),
    status = output(3, 'uint32');
  graph.add(
    new GPUTerrainDepressionHierarchy({
      grid: grid(fixture.width, fixture.height),
      elevation: input(fixture.elevation, 'float32'),
      outlets: input(Array(count).fill(0), 'uint32'),
      stages: input(fixture.stages, 'float32'),
      nodeCapacity: count,
      output: {
        nodes: nodes.view,
        levels: levels.view,
        curves: curves.view,
        cellNodes: output(count, 'uint32').view,
        status: status.view
      }
    })
  );
  const destroy = run();
  try {
    const actualStatus = await status.read(),
      actualLevels = await levels.read(),
      actualCurves = await curves.read();
    expect(actualStatus[2]).toBe(0);
    for (let stageIndex = 0; stageIndex < fixture.stages.length; stageIndex++) {
      const stage = fixture.stages[stageIndex],
        actual = [];
      for (let node = 0; node < actualStatus[0]; node++)
        if (actualLevels[node * 2] <= stage && actualLevels[node * 2 + 1] > stage) {
          const offset = (node * fixture.stages.length + stageIndex) * 2;
          actual.push(actualCurves.slice(offset, offset + 2));
        }
      actual.sort((first, second) => first[0] - second[0] || first[1] - second[1]);
      expect(actual).toEqual(fixture.components[stageIndex]);
    }
  } finally {
    destroy();
  }
});

it('agrees with the executed Whitebox escape corridor while retaining the explicit no-pit-fill objective', async () => {
  const {graph, input, output, run} = await setup();
  const fixture = reference.breach;
  const elevation = output(fixture.source.length, 'float32'),
    paths = output(4, 'uint32'),
    status = output(1, 'uint32');
  graph.add(
    new GPUTerrainBreach({
      grid: grid(fixture.width, fixture.height),
      elevation: input(fixture.source, 'float32'),
      pitCells: input([fixture.pit], 'uint32'),
      maximumSteps: 3,
      maximumCutDepth: 10,
      elevationDecrement: 0.5,
      output: {
        elevation: elevation.view,
        cutDepth: output(fixture.source.length, 'float32').view,
        modificationMask: output(fixture.source.length, 'uint32').view,
        pathCells: paths.view,
        pathLengths: output(1, 'uint32').view,
        pitStatus: status.view,
        unresolvedCount: output(1, 'uint32').view
      }
    })
  );
  const destroy = run();
  try {
    expect(await status.read()).toEqual([1]);
    expect(await paths.read()).toEqual([15, 16, 17, 18]);
    const actual = await elevation.read();
    expect(actual.slice(15, 19)).toEqual([2, 1.5, 1, 0]);
    // Whitebox lowers the starting pit by its flat increment; compare the escape corridor, not unlike objectives.
    expect(fixture.output.slice(15, 19)).toEqual([1.5, 1, 0.5, 0]);
    for (let cell = 0; cell < actual.length; cell++)
      if (cell < 15 || cell > 18) expect(actual[cell]).toBe(fixture.output[cell]);
  } finally {
    destroy();
  }
});

it('matches a composed pysheds catchment and independent elevation reductions', async () => {
  const {graph, input, output, run} = await setup();
  const {width, height} = reference.routing,
    count = width * height;
  const statistics = output(6, 'float32'),
    hypsometry = output(8, 'float32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: grid(width, height),
      elevation: input(
        Array.from(
          {length: count},
          (_, cell) => (height - Math.floor(cell / width)) * 2 + width - (cell % width)
        ),
        'float32'
      ),
      basinIds: input(Array(count).fill(0), 'uint32'),
      basinCount: 1,
      streams: input(Array(count).fill(1), 'uint32'),
      flowDirections: input(makeRouting(width, height).directions, 'uint32'),
      histogramBins: 8,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: output(count * 3, 'float32').view,
        profileStatus: output(count, 'uint32').view
      }
    })
  );
  const destroy = run();
  try {
    const actual = await statistics.read();
    expect(actual.slice(0, 4)).toEqual([
      reference.catchment.area,
      reference.catchment.mean,
      reference.catchment.minimum,
      reference.catchment.maximum
    ]);
    (await hypsometry.read()).forEach((value, index) =>
      expect(value).toBeCloseTo(reference.catchment.hypsometry[index] * reference.catchment.area, 5)
    );
  } finally {
    destroy();
  }
});

it('composes global depression filling and flat resolution into D8 and MFD tile consumers without host readback', async () => {
  const {graph, input, output, run} = await setup();
  const width = 32,
    height = 16,
    count = width * height;
  const metadata = grid(width, height);
  const source = Array.from({length: count}, (_, cell) => {
    const column = cell % width,
      row = Math.floor(cell / width);
    if (column === width - 1 && row === height / 2) return 1;
    if (column === 0 || row === 0 || column === width - 1 || row === height - 1) return 10;
    if (column === width / 2) return 4;
    return column < width / 2 ? 0 : 2;
  });
  const parameters = input(
    Array.from(getGPUTerrainFlowParameterValues({streamThreshold: 1}, metadata)),
    'float32'
  );
  const results = [];
  for (const routing of ['d8', 'mfd-freeman'] as const) {
    const filled = output(count, 'float32'),
      directions = output(count, 'uint32'),
      accumulation = output(count, 'float32'),
      streams = output(count, 'uint32');
    const fillConverged = output(1, 'uint32'),
      flatsConverged = output(1, 'uint32');
    graph.add(
      new GPUTerrainFlow({
        id: `${routing}-prepare`,
        grid: metadata,
        elevation: makeGPURasterBand(input(source, 'float32'), metadata),
        parameters,
        fillDepressions: true,
        resolveFlats: true,
        maximumFlatIterations: 128,
        maximumAccumulationIterations: 128,
        flowRouting: routing,
        output: {
          filledElevation: filled.view,
          flowDirections: directions.view,
          accumulation: accumulation.view,
          streams: streams.view,
          status: {fillConverged: fillConverged.view, flatsConverged: flatsConverged.view}
        }
      })
    );
    const fractions = output(count * 8, 'float32');
    if (routing !== 'd8')
      graph.add(
        new GPUTerrainFlowFractions({
          id: `${routing}-fractions`,
          grid: metadata,
          elevation: filled.view,
          flowDirections: directions.view,
          parameters,
          routing,
          output: {fractions: fractions.view, edgeLengths: output(count * 8, 'float32').view}
        })
      );
    const tiles = [0, 1].map(side => {
      const values = output(count, 'float32'),
        status = output(count, 'uint32');
      results.push({side, values, status, accumulation, fillConverged, flatsConverged, filled});
      return {
        column: 0,
        row: 0,
        width,
        height,
        owned: [(side * width) / 2, 0, width / 2, height] as const,
        flowDirections: directions.view,
        ...(routing === 'd8' ? {} : {fractions: fractions.view}),
        runoff: input(Array(count).fill(1), 'float32'),
        streams: streams.view,
        output: {
          accumulation: values.view,
          status: status.view,
          reachIds: output(count, 'uint32').view
        }
      };
    });
    graph.add(
      new GPUTerrainTiledFlow({
        id: `${routing}-tiles`,
        grid: metadata,
        routing: routing === 'd8' ? 'd8' : 'branching',
        tiles
      })
    );
  }
  const destroy = run();
  try {
    for (const result of results) {
      expect(await result.fillConverged.read()).toEqual([1]);
      expect(await result.flatsConverged.read()).toEqual([1]);
      expect((await result.filled.read()).some((value, index) => value > source[index])).toBe(true);
      const values = await result.values.read(),
        expected = await result.accumulation.read(),
        statuses = await result.status.read();
      for (let cell = 0; cell < count; cell++)
        if (Math.floor((cell % width) / (width / 2)) === result.side) {
          expect(statuses[cell]).toBe(1);
          expect(values[cell]).toBeCloseTo(expected[cell], 3);
        }
    }
  } finally {
    destroy();
  }
}, 120000);

it('fits 16384-cell DEMs and builds 16384-cell hierarchies with sorted linear reductions', async () => {
  const {graph, input, output, run} = await setup();
  const width = 128,
    count = width * width;
  const elevations = Array.from(
    {length: count},
    (_, cell) => ((cell % width) - 63) ** 2 + (Math.floor(cell / width) - 63) ** 2
  );
  const treeStatus = output(3, 'uint32'),
    curves = output(count * 6, 'float32');
  graph.add(
    new GPUTerrainDepressionHierarchy({
      grid: grid(width, width),
      elevation: input(elevations, 'float32'),
      outlets: input(Array(count).fill(0), 'uint32'),
      stages: input([0, 1, 10000], 'float32'),
      nodeCapacity: count,
      output: {
        nodes: output(count * 4, 'uint32').view,
        levels: output(count * 2, 'float32').view,
        cellNodes: output(count, 'uint32').view,
        curves: curves.view,
        status: treeStatus.view
      }
    })
  );
  const fit = output(4, 'float32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      grid: grid(width, width),
      reference: input(elevations, 'float32'),
      moving: input(
        elevations.map(value => value + 3),
        'float32'
      ),
      stableMask: input(Array(count).fill(1), 'uint32'),
      differenceVariance: input(Array(count).fill(1), 'float32'),
      output: {
        fit: fit.view,
        volume: volume.view,
        differences: output(count * 2, 'float32').view,
        validity: output(count, 'uint32').view
      }
    })
  );
  const started = performance.now(),
    destroy = run();
  try {
    expect(await treeStatus.read()).toEqual([1, 1, 0]);
    const actualCurves = await curves.read();
    expect(actualCurves.slice(0, 4)).toEqual([1, 0, 5, 1]);
    expect(actualCurves[4]).toBe(count);
    expect(actualCurves[5]).toBe(elevations.reduce((sum, elevation) => sum + 10000 - elevation, 0));
    expect(await fit.read()).toEqual([3, 0, count, 0]);
    expect(await volume.read()).toEqual([0, width, count]);
    console.info(
      `terrain-scale hierarchy/DEM ${count} cells: compile/submit/readback/assertions ${Math.round(performance.now() - started)} ms`
    );
  } finally {
    destroy();
  }
}, 120000);

it('resolves 16384-cell transport and reverse distances with bounded successor walks', async () => {
  const {graph, input, output, run} = await setup();
  const count = 16384;
  const fractions = Array(count * 8).fill(0);
  for (let cell = 0; cell < count - 1; cell++) fractions[cell * 8] = 1;
  const results = [];
  for (const mode of ['transport', 'expected-distance'] as const) {
    const values = output(count, 'float32'),
      unresolved = output(1, 'uint32');
    results.push({values, unresolved});
    graph.add(
      new GPUTerrainBranchingFlow({
        id: mode,
        grid: grid(count),
        fractions: input(fractions, 'float32'),
        values: input(Array(count).fill(1), 'float32'),
        mode,
        maximumWalkLength: 256,
        maximumIterations: 128,
        output: {
          values: values.view,
          status: output(count, 'uint32').view,
          unresolvedCount: unresolved.view
        }
      })
    );
  }
  const destroy = run();
  try {
    for (const result of results) expect(await result.unresolved.read()).toEqual([0]);
    expect((await results[0].values.read())[count - 1]).toBe(count);
    expect((await results[1].values.read())[0]).toBe(count - 1);
  } finally {
    destroy();
  }
}, 120000);

it('breaches 64 selected pits in a 65536-cell domain using local bounded search storage', async () => {
  const {graph, input, output, run} = await setup();
  const width = 256,
    count = width * width;
  const source = Array(count).fill(50),
    pits: number[] = [];
  for (let pit = 0; pit < 64; pit++) {
    const cell = (8 + Math.floor(pit / 16) * 32) * width + 4 + (pit % 16) * 16;
    pits.push(cell);
    source.splice(cell, 5, 2, 2, 5, 0, -1);
  }
  const elevation = output(count, 'float32'),
    status = output(pits.length, 'uint32'),
    unresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainBreach({
      grid: grid(width, width),
      elevation: input(source, 'float32'),
      pitCells: input(pits, 'uint32'),
      maximumSteps: 3,
      maximumCutDepth: 10,
      elevationDecrement: 0.5,
      output: {
        elevation: elevation.view,
        cutDepth: output(count, 'float32').view,
        modificationMask: output(count, 'uint32').view,
        pathCells: output(pits.length * 4, 'uint32').view,
        pathLengths: output(pits.length, 'uint32').view,
        pitStatus: status.view,
        unresolvedCount: unresolved.view
      }
    })
  );
  const started = performance.now(),
    destroy = run();
  try {
    expect(await unresolved.read()).toEqual([0]);
    expect((await status.read()).every(value => value === 1)).toBe(true);
    const actual = await elevation.read();
    for (const pit of pits) expect(actual.slice(pit, pit + 5)).toEqual([2, 1.5, 1, 0, -1]);
    console.info(
      `terrain-scale breach ${count} cells / 64 pits: compile/submit/readback/assertions ${Math.round(performance.now() - started)} ms`
    );
  } finally {
    destroy();
  }
}, 120000);

it('propagates the full sheared-grid covariance matrix for 1024 heteroskedastic DEM differences', async () => {
  const {graph, input, output, run} = await setup();
  const count = 1024,
    expected = reference.vertical.correlated;
  const residual = Array.from(
    {length: count},
    (_, index) => (((index * 7) % 17) - 8) * 0.25 + 3 + (index % 61 === 0 ? 30 : 0)
  );
  const fit = output(4, 'float32'),
    differences = output(count * 2, 'float32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      grid: {...grid(32, 32), affine: [2, 0.5, 0, -0.25, 3, 0]},
      reference: input(Array(count).fill(0), 'float32'),
      moving: input(residual, 'float32'),
      stableMask: input(
        residual.map((_, index) => Number(index % 13 !== 0)),
        'uint32'
      ),
      differenceVariance: input(
        residual.map((_, index) => 1 + (index % 5) * 0.25),
        'float32'
      ),
      correlationLength: expected.length,
      output: {
        fit: fit.view,
        differences: differences.view,
        validity: output(count, 'uint32').view,
        volume: volume.view
      }
    })
  );
  const destroy = run();
  try {
    expect((await fit.read())[3]).toBeCloseTo(expected.biasStandardError, 5);
    const actual = await differences.read();
    expected.cellStandardDeviations.forEach((value, index) =>
      expect(actual[index * 2 + 1]).toBeCloseTo(value, 5)
    );
    expect(
      Math.abs((await volume.read())[1] - expected.volumeStandardDeviation) /
        expected.volumeStandardDeviation
    ).toBeLessThan(0.00001);
  } finally {
    destroy();
  }
});

it('detects overlapping breach proposals that destroy one path after deterministic minimum edits', async () => {
  const {graph, input, output, run} = await setup();
  const elevation = output(5, 'float32'),
    status = output(2, 'uint32'),
    unresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainBreach({
      grid: grid(5),
      elevation: input([2, 5, 4, 5, 0], 'float32'),
      pitCells: input([0, 2], 'uint32'),
      maximumSteps: 4,
      maximumCutDepth: 4,
      elevationDecrement: 0.25,
      output: {
        elevation: elevation.view,
        cutDepth: output(5, 'float32').view,
        modificationMask: output(5, 'uint32').view,
        pathCells: output(10, 'uint32').view,
        pathLengths: output(2, 'uint32').view,
        pitStatus: status.view,
        unresolvedCount: unresolved.view
      }
    })
  );
  const destroy = run();
  try {
    expect(await elevation.read()).toEqual([2, 1.75, 1.5, 1.25, 0]);
    expect(await status.read()).toEqual([1, 2]);
    expect(await unresolved.read()).toEqual([1]);
  } finally {
    destroy();
  }
});

it('retains unresolved donor dependencies when the incoming fraction itself is unknown', async () => {
  const {graph, input, output, run} = await setup();
  const fractions = Array(16).fill(0);
  fractions[0] = Number.NaN;
  const status = output(2, 'uint32'),
    values = output(2, 'float32');
  graph.add(
    new GPUTerrainBranchingFlow({
      grid: grid(2),
      fractions: input(fractions, 'float32'),
      values: input([1, 1], 'float32'),
      mode: 'transport',
      output: {values: values.view, status: status.view, unresolvedCount: output(1, 'uint32').view}
    })
  );
  const destroy = run();
  try {
    expect(await status.read()).toEqual([0, 2]);
    expect((await values.read()).every(Number.isNaN)).toBe(true);
  } finally {
    destroy();
  }
});

it('reduces 256 basins and stitches 65536 stream profiles without basin-by-raster or path-length scans', async () => {
  const {graph, input, output, run} = await setup();
  const width = 256,
    count = width * width;
  const statistics = output(width * 6, 'float32'),
    hypsometry = output(width * 8, 'float32'),
    profiles = output(count * 3, 'float32'),
    status = output(count, 'uint32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: grid(width, width),
      elevation: input(
        Array.from({length: count}, (_, cell) => width - (cell % width)),
        'float32'
      ),
      basinIds: input(
        Array.from({length: count}, (_, cell) => Math.floor(cell / width)),
        'uint32'
      ),
      basinCount: width,
      streams: input(Array(count).fill(1), 'uint32'),
      flowDirections: input(
        Array.from({length: count}, (_, cell) => (cell % width === width - 1 ? 0 : 1)),
        'uint32'
      ),
      histogramBins: 8,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: profiles.view,
        profileStatus: status.view
      }
    })
  );
  const started = performance.now(),
    destroy = run();
  try {
    const actualStatistics = await statistics.read(),
      actualHypsometry = await hypsometry.read(),
      actualProfiles = await profiles.read();
    expect((await status.read()).every(value => value === 1)).toBe(true);
    for (let basin = 0; basin < width; basin++) {
      expect(actualStatistics.slice(basin * 6, basin * 6 + 6)).toEqual([
        256,
        128.5,
        1,
        256,
        255,
        255 / 256
      ]);
      expect(actualHypsometry.slice(basin * 8, basin * 8 + 8)).toEqual([
        32, 64, 96, 128, 160, 192, 224, 256
      ]);
    }
    for (let cell = 0; cell < count; cell++) {
      expect(actualProfiles[cell * 3]).toBe(width - (cell % width) - 1);
      expect(actualProfiles[cell * 3 + 2]).toBe(cell % width === width - 1 ? 0 : 1);
    }
    console.info(
      `terrain-scale catchments ${count} cells / 256 basins: compile/submit/readback/assertions ${Math.round(performance.now() - started)} ms`
    );
  } finally {
    destroy();
  }
}, 120000);

it('preserves exact profile walk budgets after pointer jumping and excludes cyclic chains', async () => {
  const {graph, input, output, run} = await setup();
  const status = output(6, 'uint32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: grid(6),
      elevation: input([6, 5, 4, 3, 2, 1], 'float32'),
      basinIds: input([0, 0, 0, 0, 0, 0], 'uint32'),
      basinCount: 1,
      streams: input([1, 1, 1, 1, 1, 1], 'uint32'),
      flowDirections: input([1, 1, 1, 0, 1, 16], 'uint32'),
      histogramBins: 2,
      maximumSteps: 3,
      output: {
        statistics: output(6, 'float32').view,
        hypsometry: output(2, 'float32').view,
        profiles: output(18, 'float32').view,
        profileStatus: status.view
      }
    })
  );
  const destroy = run();
  try {
    expect(await status.read()).toEqual([2, 1, 1, 1, 2, 2]);
  } finally {
    destroy();
  }
});

it('retains correlated uncertainty for large and small representable variances', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const variance of [1e20, 1e-24]) {
    const differences = output(4, 'float32'),
      volume = output(3, 'float32');
    results.push({variance, differences, volume});
    graph.add(
      new GPUTerrainDEMCoregistration({
        id: `variance-${variance}`,
        grid: grid(2),
        reference: input([0, 0], 'float32'),
        moving: input([0, 0], 'float32'),
        stableMask: input([1, 1], 'uint32'),
        differenceVariance: input([variance, variance], 'float32'),
        correlationLength: 1,
        output: {
          fit: output(4, 'float32').view,
          differences: differences.view,
          validity: output(2, 'uint32').view,
          volume: volume.view
        }
      })
    );
  }
  const destroy = run();
  try {
    for (const result of results) {
      const expected = Math.sqrt(result.variance * (2 + 2 * Math.exp(-1)));
      const actual = (await result.volume.read())[1];
      expect(
        Number.isFinite(actual) && actual > 0,
        `variance ${result.variance}, standard deviation ${actual}`
      ).toBe(true);
      expect(Math.abs(actual / expected - 1)).toBeLessThan(0.00001);
      expect(
        Math.abs((await result.differences.read())[1] / Math.sqrt(result.variance) - 1)
      ).toBeLessThan(0.00001);
    }
  } finally {
    destroy();
  }
});
