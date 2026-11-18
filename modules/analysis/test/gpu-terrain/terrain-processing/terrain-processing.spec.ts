// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import reference from '../../oracles/terrain-processing.json';
import {GPUTerrainFlowFractions} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-flow-fractions';
import {GPUTerrainBranchingFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow';
import {GPUTerrainBreach} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {GPUTerrainDepressionHierarchy} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/contracts/raster-grid';
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

/** The readable half of a graph output, which is all the assertions use. */
type TerrainOutput = {read(): Promise<number[]>};

it('branching diamond conserves transport and distinguishes expected, minimum and maximum distance', async () => {
  const {graph, input, output, run} = await setup();
  // 0 -> 1 (1/4), 0 -> 2 (3/4), both -> 3. Node 3 is the sole outlet.
  const fractions = Array(32).fill(0);
  fractions[0] = 0.25;
  fractions[2] = 0.75;
  fractions[10] = 1;
  fractions[16] = 1;
  const lengths = Array(32).fill(0);
  lengths[0] = 1;
  lengths[2] = 3;
  lengths[10] = 1;
  lengths[16] = 1;
  const results = [];
  for (const mode of [
    'transport',
    'expected-distance',
    'minimum-distance',
    'maximum-distance',
    'fractional-catchment'
  ] as const) {
    const values = output(4, 'float32'),
      status = output(4, 'uint32'),
      unresolved = output(1, 'uint32');
    results.push(values);
    graph.add(
      new GPUTerrainBranchingFlow({
        id: mode,
        grid: grid(2, 2),
        fractions: input(fractions, 'float32'),
        values: input(mode === 'transport' ? [8, 0, 0, 0] : [0, 0, 0, 1], 'float32'),
        edgeLengths: input(lengths, 'float32'),
        mode,
        output: {values: values.view, status: status.view, unresolvedCount: unresolved.view}
      })
    );
  }
  const destroy = run();
  expect(await results[0].read()).toEqual(reference.diamond.transport);
  expect((await results[1].read())[0]).toBe(reference.diamond.expected);
  expect((await results[2].read())[0]).toBe(reference.diamond.minimum);
  expect((await results[3].read())[0]).toBe(reference.diamond.maximum);
  expect(await results[4].read()).toEqual([1, 1, 1, 1]);
  destroy();
});

it('branching cycles and short budgets remain unresolved', async () => {
  const {graph, input, output, run} = await setup();
  const fractions = Array(16).fill(0);
  fractions[0] = 1;
  fractions[12] = 1;
  const values = output(2, 'float32'),
    status = output(2, 'uint32'),
    unresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainBranchingFlow({
      grid: grid(2),
      fractions: input(fractions, 'float32'),
      values: input([1, 1], 'float32'),
      mode: 'transport',
      output: {values: values.view, status: status.view, unresolvedCount: unresolved.view}
    })
  );
  const destroy = run();
  expect(await status.read()).toEqual([2, 2]);
  expect(await unresolved.read()).toEqual([2]);
  expect((await values.read()).every(Number.isNaN)).toBe(true);
  destroy();
});

it('bounded breaching cuts an embankment and reports depth exhaustion without silently filling', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const maximumCutDepth of [4, 1]) {
    const elevation = output(3, 'float32'),
      depth = output(3, 'float32'),
      mask = output(3, 'uint32'),
      paths = output(3, 'uint32'),
      lengths = output(1, 'uint32'),
      status = output(1, 'uint32'),
      unresolved = output(1, 'uint32');
    results.push({elevation, status, unresolved});
    graph.add(
      new GPUTerrainBreach({
        id: `breach-${maximumCutDepth}`,
        grid: grid(3),
        elevation: input([2, 5, 0], 'float32'),
        pitCells: input([0], 'uint32'),
        maximumSteps: 2,
        maximumCutDepth,
        elevationDecrement: 0.5,
        output: {
          elevation: elevation.view,
          cutDepth: depth.view,
          modificationMask: mask.view,
          pathCells: paths.view,
          pathLengths: lengths.view,
          pitStatus: status.view,
          unresolvedCount: unresolved.view
        }
      })
    );
  }
  const destroy = run();
  expect(await results[0].elevation.read()).toEqual([2, 1.5, 0]);
  expect(await results[0].status.read()).toEqual([1]);
  expect(await results[1].elevation.read()).toEqual([2, 5, 0]);
  expect(await results[1].unresolved.read()).toEqual([1]);
  destroy();
});

it('nested sink ownership produces analytic stage areas and volumes and rejects capacity exhaustion', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const capacity of [3, 1]) {
    const nodes = output(capacity * 4, 'uint32'),
      levels = output(capacity * 2, 'float32'),
      ownership = output(3, 'uint32'),
      curves = output(capacity * 4, 'float32'),
      status = output(3, 'uint32'),
      requiredCount = output(1, 'uint32'),
      overflow = output(1, 'uint32');
    results.push({nodes, curves, status, requiredCount, overflow});
    graph.add(
      new GPUTerrainDepressionHierarchy({
        id: `hierarchy-${capacity}`,
        grid: grid(3),
        elevation: input([0, 2, 1], 'float32'),
        outlets: input([0, 0, 0], 'uint32'),
        stages: input([1, 3], 'float32'),
        nodeCapacity: capacity,
        output: {
          nodes: nodes.view,
          levels: levels.view,
          cellNodes: ownership.view,
          curves: curves.view,
          status: status.view,
          ports: {status: {requiredCount: requiredCount.view, overflow: overflow.view}}
        }
      })
    );
  }
  const destroy = run();
  expect(await results[0].status.read()).toEqual([3, 3, 0]);
  expect(await results[0].requiredCount.read()).toEqual([3]);
  expect(await results[0].overflow.read()).toEqual([0]);
  // Named ports beside the loose status words: required 3 nodes, capacity 1 overflowed.
  expect(await results[1].requiredCount.read()).toEqual([3]);
  expect(await results[1].overflow.read()).toEqual([1]);
  expect((await results[0].curves.read()).slice(8)).toEqual(reference.nested.rootCurves);
  expect((await results[0].nodes.read()).slice(0, 4)).toEqual([2, 0, 0xffffffff, 1]);
  expect(await results[1].status.read()).toEqual([1, 3, 1]);
  expect((await results[1].curves.read()).every(Number.isNaN)).toBe(true);
  destroy();
});

it('basin summaries have physical area, cumulative hypsometry and directed profiles', async () => {
  const {graph, input, output, run} = await setup();
  const statistics = output(12, 'float32'),
    hypsometry = output(4, 'float32'),
    profiles = output(9, 'float32'),
    status = output(3, 'uint32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: grid(3),
      elevation: input([3, 2, 1], 'float32'),
      basinIds: input([0, 0, 0], 'uint32'),
      basinCount: 2,
      streams: input([1, 1, 1], 'uint32'),
      flowDirections: input([1, 1, 0], 'uint32'),
      histogramBins: 2,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: profiles.view,
        profileStatus: status.view
      }
    })
  );
  const destroy = run();
  const actual = await statistics.read();
  expect(actual.slice(0, 5)).toEqual([3, 2, 1, 3, 2]);
  expect(actual[5]).toBeCloseTo(2 / 3);
  expect(actual[6]).toBe(0);
  expect(actual[7]).toBeNaN();
  expect(await hypsometry.read()).toEqual([2, 3, 0, 0]);
  expect(await profiles.read()).toEqual([2, 3, 1, 1, 2, 1, 0, 1, 0]);
  destroy();
});

it('vertical coregistration rejects outliers and propagates common bias and correlated measurement uncertainty', async () => {
  const {graph, input, output, run} = await setup();
  const fit = output(4, 'float32'),
    differences = output(10, 'float32'),
    validity = output(5, 'uint32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      grid: grid(5),
      reference: input([0, 0, 0, 0, 0], 'float32'),
      moving: input([2, 2, 2, 100, 5], 'float32'),
      stableMask: input([1, 1, 1, 1, 0], 'uint32'),
      differenceVariance: input([1, 1, 1, 1, 1], 'float32'),
      output: {
        fit: fit.view,
        differences: differences.view,
        validity: validity.view,
        volume: volume.view
      }
    })
  );
  const destroy = run();
  expect(await fit.read()).toEqual([2, 0, 4, 0]);
  expect((await differences.read()).filter((_, index) => index % 2 === 0)).toEqual(
    reference.vertical.corrected
  );
  expect((await volume.read())[0]).toBe(reference.vertical.volume);
  expect((await volume.read())[1]).toBeCloseTo(reference.vertical.independentStandardDeviation);
  destroy();
});

it('tiled D8 seams and halo exclusion equal complete-domain accumulation and reach IDs', async () => {
  const {graph, input, output, run} = await setup();
  const results: {
    description: {column: number; width: number; owned: readonly number[]};
    accumulation: TerrainOutput;
    status: TerrainOutput;
    reachIds: TerrainOutput;
  }[] = [];
  for (const tiled of [false, true]) {
    const tiles = (
      tiled
        ? [
            {column: 0, width: 3, owned: [0, 0, 2, 1] as const},
            {column: 1, width: 3, owned: [1, 0, 2, 1] as const}
          ]
        : [{column: 0, width: 4, owned: [0, 0, 4, 1] as const}]
    ).map(description => {
      const accumulation = output(description.width, 'float32'),
        status = output(description.width, 'uint32'),
        reachIds = output(description.width, 'uint32');
      results.push({description, accumulation, status, reachIds});
      return {
        ...description,
        row: 0,
        height: 1,
        flowDirections: input(
          Array.from({length: description.width}, (_, index) =>
            index + description.column === 3 ? 0 : 1
          ),
          'uint32'
        ),
        runoff: input(
          Array.from({length: description.width}, (_, index) =>
            index >= description.owned[0] && index < description.owned[0] + description.owned[2]
              ? 1
              : 999
          ),
          'float32'
        ),
        streams: input(Array(description.width).fill(1), 'uint32'),
        output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
      };
    });
    graph.add(
      new GPUTerrainTiledFlow({
        id: `tiled-${tiled}`,
        grid: {width: 4, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        tiles
      })
    );
  }
  const destroy = run();
  expect(await results[0].accumulation.read()).toEqual([1, 2, 3, 4]);
  expect((await results[1].accumulation.read()).slice(0, 2)).toEqual([1, 2]);
  expect((await results[2].accumulation.read()).slice(1)).toEqual([3, 4]);
  expect(await results[0].reachIds.read()).toEqual([0, 0, 0, 0]);
  expect((await results[2].reachIds.read()).slice(1)).toEqual([0, 0]);
  expect((await results[1].status.read())[2]).toBe(0);
  destroy();
});

it('existing routing fractions preserve D-infinity/MFD outputs and unresolved sentinels', async () => {
  const {graph, input, output, run} = await setup();
  const results = [];
  for (const routing of ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
    const fractions = output(24, 'float32'),
      lengths = output(24, 'float32');
    results.push(fractions);
    graph.add(
      new GPUTerrainFlowFractions({
        id: routing,
        grid: grid(3),
        elevation: input([3, 2, 1], 'float32'),
        flowDirections: input([1, 1, 0], 'uint32'),
        parameters: input([1, 1, 0, 1, 0, 0, 1.1, 0], 'float32'),
        routing,
        output: {fractions: fractions.view, edgeLengths: lengths.view}
      })
    );
  }
  const invalidFractions = output(24, 'float32'),
    invalidLengths = output(24, 'float32');
  graph.add(
    new GPUTerrainFlowFractions({
      id: 'unresolved',
      grid: grid(3),
      elevation: input([3, 2, 1], 'float32'),
      flowDirections: input([1, 0xfffffffe, 0], 'uint32'),
      parameters: input([1, 1, 0, 1, 0, 0, 1.1, 0], 'float32'),
      routing: 'd-infinity',
      output: {fractions: invalidFractions.view, edgeLengths: invalidLengths.view}
    })
  );
  const destroy = run();
  for (const result of results) {
    const actual = await result.read();
    expect(actual[0]).toBe(1);
    expect(actual[8]).toBe(1);
    expect(actual.slice(16)).toEqual(Array(8).fill(0));
  }
  expect((await invalidFractions.read()).slice(8, 16).every(Number.isNaN)).toBe(true);
  destroy();
});

it('fractional catchments preserve distinct outlets across bounded atomic sweeps', async () => {
  const {graph, input, output, run} = await setup();
  const fractions = Array(24).fill(0);
  fractions[8] = 0.8;
  fractions[12] = 0.2;
  const results = [];
  for (const iterations of [1, 3]) {
    const values = output(3, 'float32'),
      status = output(3, 'uint32'),
      unresolved = output(1, 'uint32');
    results.push({values, status});
    graph.add(
      new GPUTerrainBranchingFlow({
        id: `fractional-${iterations}`,
        grid: grid(3),
        fractions: input(fractions, 'float32'),
        values: input([0, 0, 1], 'float32'),
        mode: 'fractional-catchment',
        maximumIterations: iterations,
        output: {values: values.view, status: status.view, unresolvedCount: unresolved.view}
      })
    );
  }
  const destroy = run();
  const boundedStatus = await results[0].status.read();
  expect([boundedStatus[0], boundedStatus[2]]).toEqual([1, 1]);
  // Atomic successor walks can finish multiple dependency levels in one sweep.
  if (boundedStatus[1] === 1) expect((await results[0].values.read())[1]).toBeCloseTo(0.8);
  else {
    expect(boundedStatus[1]).toBe(2);
    expect(Number.isNaN((await results[0].values.read())[1])).toBe(true);
  }
  expect((await results[1].values.read())[1]).toBeCloseTo(0.8);
  destroy();
});

it('nodata barriers preserve disconnected closed sinks', async () => {
  const {graph, input, output, run} = await setup();
  const nodes = output(12, 'uint32'),
    levels = output(6, 'float32'),
    ownership = output(3, 'uint32'),
    curves = output(6, 'float32'),
    status = output(3, 'uint32');
  graph.add(
    new GPUTerrainDepressionHierarchy({
      grid: grid(3),
      elevation: input([0, NaN, 0], 'float32'),
      outlets: input([1, 0, 0], 'uint32'),
      stages: input([2], 'float32'),
      nodeCapacity: 3,
      output: {
        nodes: nodes.view,
        levels: levels.view,
        cellNodes: ownership.view,
        curves: curves.view,
        status: status.view
      }
    })
  );
  const destroy = run();
  expect(await status.read()).toEqual([2, 2, 0]);
  expect(await ownership.read()).toEqual([0, 0xffffffff, 1]);
  expect((await nodes.read()).slice(0, 8)).toEqual([
    0xffffffff, 0, 0, 1, 0xffffffff, 2, 0xffffffff, 1
  ]);
  expect((await curves.read()).slice(0, 4)).toEqual([1, 2, 1, 2]);
  destroy();
});

it('correlated DEM errors increase integrated uncertainty according to the declared covariance model', async () => {
  const {graph, input, output, run} = await setup();
  const fit = output(4, 'float32'),
    differences = output(10, 'float32'),
    validity = output(5, 'uint32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      grid: grid(5),
      reference: input([0, 0, 0, 0, 0], 'float32'),
      moving: input([2, 2, 2, 100, 5], 'float32'),
      stableMask: input([1, 1, 1, 1, 0], 'uint32'),
      differenceVariance: input([1, 1, 1, 1, 1], 'float32'),
      correlationLength: 2,
      output: {
        fit: fit.view,
        differences: differences.view,
        validity: validity.view,
        volume: volume.view
      }
    })
  );
  const destroy = run();
  expect((await volume.read())[1]).toBeCloseTo(reference.vertical.correlatedStandardDeviation);
  destroy();
});

it('a river can leave and reenter a tile without making tile-level dependency cycles into outlets', async () => {
  const {graph, input, output, run} = await setup();
  const directions = [1, 1, 1, 4, 0, 16, 16, 16];
  const results: {accumulation: TerrainOutput; status: TerrainOutput; reachIds: TerrainOutput}[] =
    [];
  const tiles = [0, 2].map(column => {
    const globalCells = [column, column + 1, column + 4, column + 5];
    const accumulation = output(4, 'float32'),
      status = output(4, 'uint32'),
      reachIds = output(4, 'uint32');
    results.push({accumulation, status, reachIds});
    return {
      column,
      row: 0,
      width: 2,
      height: 2,
      owned: [0, 0, 2, 2] as const,
      flowDirections: input(
        globalCells.map(cell => directions[cell]),
        'uint32'
      ),
      runoff: input([1, 1, 1, 1], 'float32'),
      streams: input([1, 1, 1, 1], 'uint32'),
      output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
    };
  });
  graph.add(
    new GPUTerrainTiledFlow({
      grid: {width: 4, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      tiles
    })
  );
  const destroy = run();
  expect(await results[0].accumulation.read()).toEqual([1, 2, 8, 7]);
  expect(await results[1].accumulation.read()).toEqual([3, 4, 6, 5]);
  expect(await results[0].reachIds.read()).toEqual([0, 0, 0, 0]);
  destroy();
});

it('insufficient stable ground leaves the DEM fit uncertainty and differencing unresolved', async () => {
  const {graph, input, output, run} = await setup();
  const fit = output(4, 'float32'),
    differences = output(4, 'float32'),
    validity = output(2, 'uint32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      grid: grid(2),
      reference: input([0, 0], 'float32'),
      moving: input([2, 3], 'float32'),
      stableMask: input([1, 0], 'uint32'),
      differenceVariance: input([1, 1], 'float32'),
      output: {
        fit: fit.view,
        differences: differences.view,
        validity: validity.view,
        volume: volume.view
      }
    })
  );
  const destroy = run();
  expect((await fit.read())[3]).toBeNaN();
  expect(await validity.read()).toEqual([0, 0]);
  expect((await differences.read()).every(Number.isNaN)).toBe(true);
  destroy();
});

it('explicit burning carries separate provenance and rejects invalid burn depths', async () => {
  const {graph, input, output, run} = await setup();
  const elevation = output(3, 'float32'),
    depth = output(3, 'float32'),
    mask = output(3, 'uint32'),
    paths = output(3, 'uint32'),
    lengths = output(1, 'uint32'),
    status = output(1, 'uint32'),
    unresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainBreach({
      grid: grid(3),
      elevation: input([2, 5, 0], 'float32'),
      pitCells: input([0], 'uint32'),
      maximumSteps: 2,
      maximumCutDepth: 4,
      elevationDecrement: 0.5,
      burnDepth: input([0.25, 0, -1], 'float32'),
      output: {
        elevation: elevation.view,
        cutDepth: depth.view,
        modificationMask: mask.view,
        pathCells: paths.view,
        pathLengths: lengths.view,
        pitStatus: status.view,
        unresolvedCount: unresolved.view
      }
    })
  );
  const destroy = run();
  expect((await elevation.read()).slice(0, 2)).toEqual([1.75, 1.5]);
  expect((await elevation.read())[2]).toBeNaN();
  expect(await mask.read()).toEqual([2, 1, 4]);
  expect(await status.read()).toEqual([2]);
  destroy();
});

it('explicit burning works without selected pits and preserves unselected depressions', async () => {
  const {graph, input, output, run} = await setup();
  const elevation = output(4, 'float32');
  const depth = output(4, 'float32');
  const mask = output(4, 'uint32');
  const unresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainBreach({
      grid: grid(4),
      elevation: input([5, 0, 5, NaN], 'float32'),
      pitCells: output(0, 'uint32').view,
      maximumSteps: 2,
      maximumCutDepth: 0,
      elevationDecrement: 0.5,
      burnDepth: input([1, 0, 0, 1], 'float32'),
      output: {
        elevation: elevation.view,
        cutDepth: depth.view,
        modificationMask: mask.view,
        pathCells: output(0, 'uint32').view,
        pathLengths: output(0, 'uint32').view,
        pitStatus: output(0, 'uint32').view,
        unresolvedCount: unresolved.view
      }
    })
  );
  const destroy = run();
  try {
    expect(await elevation.read()).toEqual([4, 0, 5, NaN]);
    expect(await depth.read()).toEqual([1, 0, 0, NaN]);
    expect(await mask.read()).toEqual([2, 0, 0, 0]);
    expect(await unresolved.read()).toEqual([0]);
  } finally {
    destroy();
  }
});

it('existing unresolved stream sentinels never become valid reach IDs', async () => {
  const {graph, input, output, run} = await setup();
  const accumulation = output(3, 'float32'),
    status = output(3, 'uint32'),
    reachIds = output(3, 'uint32');
  graph.add(
    new GPUTerrainTiledFlow({
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      tiles: [
        {
          column: 0,
          row: 0,
          width: 3,
          height: 1,
          owned: [0, 0, 3, 1],
          flowDirections: input([1, 1, 0], 'uint32'),
          runoff: input([1, 1, 1], 'float32'),
          streams: input([1, 0xfffffffe, 1], 'uint32'),
          output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
        }
      ]
    })
  );
  const destroy = run();
  expect(await status.read()).toEqual([1, 0, 2]);
  expect(await reachIds.read()).toEqual([0, 0xffffffff, 0xffffffff]);
  destroy();
});

it('branching split/rejoin seams equal the existing complete-domain consumer and conserve injected mass', async () => {
  const {graph, input, output, run} = await setup();
  const fractions = Array(32).fill(0);
  fractions[0] = 0.25;
  fractions[2] = 0.75;
  fractions[10] = 1;
  fractions[16] = 1;
  const injection = [8, 0, 0, 0];
  const complete = output(4, 'float32'),
    completeStatus = output(4, 'uint32'),
    unresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainBranchingFlow({
      id: 'complete-domain-diamond',
      grid: grid(2, 2),
      fractions: input(fractions, 'float32'),
      values: input(injection, 'float32'),
      mode: 'transport',
      output: {values: complete.view, status: completeStatus.view, unresolvedCount: unresolved.view}
    })
  );
  const results: {accumulation: TerrainOutput; status: TerrainOutput; reachIds: TerrainOutput}[] =
    [];
  const tiles = [0, 1].map(column => {
    const cells = [column, column + 2];
    const accumulation = output(2, 'float32'),
      status = output(2, 'uint32'),
      reachIds = output(2, 'uint32');
    results.push({accumulation, status, reachIds});
    return {
      column,
      row: 0,
      width: 1,
      height: 2,
      owned: [0, 0, 1, 2] as const,
      flowDirections: input([0, 0], 'uint32'),
      fractions: input(
        cells.flatMap(cell => fractions.slice(cell * 8, cell * 8 + 8)),
        'float32'
      ),
      runoff: input(
        cells.map(cell => injection[cell]),
        'float32'
      ),
      streams: input([1, 1], 'uint32'),
      output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
    };
  });
  graph.add(
    new GPUTerrainTiledFlow({
      id: 'tiled-diamond',
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      routing: 'branching',
      tiles
    })
  );
  const destroy = run();
  const left = await results[0].accumulation.read(),
    right = await results[1].accumulation.read();
  expect([left[0], right[0], left[1], right[1]]).toEqual(await complete.read());
  expect([left[0], right[0], left[1], right[1]]).toEqual(reference.diamond.transport);
  expect(right[1]).toBe(injection.reduce((sum, value) => sum + value, 0));
  expect(await results[0].reachIds.read()).toEqual([0, 2]);
  expect(await results[1].reachIds.read()).toEqual([1, 3]);
  expect(await results[0].status.read()).toEqual([1, 1]);
  expect(await results[1].status.read()).toEqual([1, 1]);
  destroy();
});

it('branching outlet sums conserve mass while contradictory halo routing and runoff are ignored', async () => {
  const {graph, input, output, run} = await setup();
  const globalFractions = Array(24).fill(0);
  globalFractions[8] = 0.7;
  globalFractions[12] = 0.3;
  const results: {accumulation: TerrainOutput; status: TerrainOutput}[] = [];
  const tiles = [0, 1, 2].map(ownedCell => {
    const accumulation = output(3, 'float32'),
      status = output(3, 'uint32'),
      reachIds = output(3, 'uint32');
    results.push({accumulation, status});
    const fractions = Array(24).fill(NaN);
    fractions.splice(ownedCell * 8, 8, ...globalFractions.slice(ownedCell * 8, ownedCell * 8 + 8));
    return {
      column: 0,
      row: 0,
      width: 3,
      height: 1,
      owned: [ownedCell, 0, 1, 1] as const,
      flowDirections: input([0, 0, 0], 'uint32'),
      fractions: input(fractions, 'float32'),
      runoff: input(
        [0, 1, 2].map(cell => (cell === ownedCell ? (cell === 1 ? 10 : 0) : 999)),
        'float32'
      ),
      streams: input([1, 1, 1], 'uint32'),
      output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
    };
  });
  graph.add(
    new GPUTerrainTiledFlow({
      grid: {width: 3, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      routing: 'branching',
      tiles
    })
  );
  const destroy = run();
  const west = (await results[0].accumulation.read())[0],
    east = (await results[2].accumulation.read())[2];
  expect(west).toBeCloseTo(3);
  expect(east).toBeCloseTo(7);
  expect(west + east).toBeCloseTo(10);
  expect(await results[1].status.read()).toEqual([0, 1, 0]);
  destroy();
});

it('positive fractional cycles spanning tiles remain unresolved and zero-weight edges do not create cycles', async () => {
  const {graph, input, output, run} = await setup();
  const results: {accumulation: TerrainOutput; status: TerrainOutput}[][] = [];
  for (const cyclic of [false, true]) {
    const tileResults: {accumulation: TerrainOutput; status: TerrainOutput}[] = [];
    const tiles = [0, 1].map(column => {
      const accumulation = output(1, 'float32'),
        status = output(1, 'uint32'),
        reachIds = output(1, 'uint32');
      tileResults.push({accumulation, status});
      const fractions = Array(8).fill(0);
      if (column === 0) {
        fractions[0] = 1;
      } else if (cyclic) {
        fractions[4] = 1;
      }
      return {
        column,
        row: 0,
        width: 1,
        height: 1,
        owned: [0, 0, 1, 1] as const,
        flowDirections: input([0], 'uint32'),
        fractions: input(fractions, 'float32'),
        runoff: input([1], 'float32'),
        streams: input([1], 'uint32'),
        output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
      };
    });
    results.push(tileResults);
    graph.add(
      new GPUTerrainTiledFlow({
        id: `fractional-cycle-${cyclic}`,
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        routing: 'branching',
        tiles
      })
    );
  }
  const destroy = run();
  expect(await results[0][1].accumulation.read()).toEqual([2]);
  expect(await results[1][0].status.read()).toEqual([2]);
  expect(await results[1][1].status.read()).toEqual([2]);
  destroy();
});

it('branching corner transfers use the diagonal owner rather than an adjacent tile edge outlet', async () => {
  const {graph, input, output, run} = await setup();
  const results: TerrainOutput[] = [];
  const tiles = [0, 1, 2, 3].map(cell => {
    const accumulation = output(1, 'float32'),
      status = output(1, 'uint32'),
      reachIds = output(1, 'uint32');
    results.push(accumulation);
    const fractions = Array(8).fill(0);
    if (cell === 0) fractions[1] = 1;
    return {
      column: cell % 2,
      row: Math.floor(cell / 2),
      width: 1,
      height: 1,
      owned: [0, 0, 1, 1] as const,
      flowDirections: input([0], 'uint32'),
      fractions: input(fractions, 'float32'),
      runoff: input([cell === 0 ? 5 : 0], 'float32'),
      streams: input([1], 'uint32'),
      output: {accumulation: accumulation.view, status: status.view, reachIds: reachIds.view}
    };
  });
  graph.add(
    new GPUTerrainTiledFlow({
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      routing: 'branching',
      tiles
    })
  );
  const destroy = run();
  expect(await results[3].read()).toEqual([5]);
  expect(await results[1].read()).toEqual([0]);
  expect(await results[2].read()).toEqual([0]);
  destroy();
});

it('a validity view counts exactly like NaN in coregistration, catchment summaries and breaching', async () => {
  const {graph, input, output, run} = await setup();
  const coregistrations = [false, true].map(useValidity => {
    const result = {
      fit: output(4, 'float32'),
      differences: output(10, 'float32'),
      validity: output(5, 'uint32'),
      volume: output(3, 'float32')
    };
    graph.add(
      new GPUTerrainDEMCoregistration({
        id: `coregistration-${useValidity}`,
        grid: grid(5),
        reference: input([0, 0, 0, 0, 0], 'float32'),
        moving: input([2, 2, 2, useValidity ? 100 : NaN, 5], 'float32'),
        ...(useValidity ? {validity: input([1, 1, 1, 0, 1], 'uint32')} : {}),
        stableMask: input([1, 1, 1, 1, 0], 'uint32'),
        differenceVariance: input([1, 1, 1, 1, 1], 'float32'),
        output: {
          fit: result.fit.view,
          differences: result.differences.view,
          validity: result.validity.view,
          volume: result.volume.view
        }
      })
    );
    return result;
  });
  const catchments = [false, true].map(useValidity => {
    const result = {
      statistics: output(6, 'float32'),
      hypsometry: output(2, 'float32'),
      profiles: output(9, 'float32'),
      profileStatus: output(3, 'uint32')
    };
    graph.add(
      new GPUTerrainCatchmentSummary({
        id: `catchment-${useValidity}`,
        grid: grid(3),
        elevation: input([3, 2, useValidity ? -9999 : NaN], 'float32'),
        ...(useValidity ? {validity: input([1, 1, 0], 'uint32')} : {}),
        basinIds: input([0, 0, 0], 'uint32'),
        basinCount: 1,
        streams: input([1, 1, 1], 'uint32'),
        flowDirections: input([1, 1, 0], 'uint32'),
        histogramBins: 2,
        output: {
          statistics: result.statistics.view,
          hypsometry: result.hypsometry.view,
          profiles: result.profiles.view,
          profileStatus: result.profileStatus.view
        }
      })
    );
    return result;
  });
  const breaches = [false, true].map(useValidity => {
    const result = {
      elevation: output(3, 'float32'),
      cutDepth: output(3, 'float32'),
      modificationMask: output(3, 'uint32'),
      pathCells: output(3, 'uint32'),
      pathLengths: output(1, 'uint32'),
      pitStatus: output(1, 'uint32'),
      unresolvedCount: output(1, 'uint32')
    };
    graph.add(
      new GPUTerrainBreach({
        id: `breach-${useValidity}`,
        grid: grid(3),
        elevation: input([2, 5, useValidity ? -9999 : NaN], 'float32'),
        ...(useValidity ? {validity: input([1, 1, 0], 'uint32')} : {}),
        pitCells: input([0], 'uint32'),
        maximumSteps: 2,
        maximumCutDepth: 4,
        elevationDecrement: 0.5,
        output: {
          elevation: result.elevation.view,
          cutDepth: result.cutDepth.view,
          modificationMask: result.modificationMask.view,
          pathCells: result.pathCells.view,
          pathLengths: result.pathLengths.view,
          pitStatus: result.pitStatus.view,
          unresolvedCount: result.unresolvedCount.view
        }
      })
    );
    return result;
  });
  const destroy = run();
  try {
    for (const group of [coregistrations, catchments, breaches]) {
      for (const key of Object.keys(group[0]) as (keyof (typeof group)[0])[]) {
        const withNaN = await (group[0][key] as TerrainOutput).read();
        const withValidity = await (group[1][key] as TerrainOutput).read();
        expect(withValidity, String(key)).toEqual(withNaN);
      }
    }
    // The masked breach has no outlet to drain into, so nothing is cut.
    expect(await (breaches[1].pitStatus as TerrainOutput).read()).toEqual([2]);
  } finally {
    destroy();
  }
});

it('cellSizeMode measures catchment areas, lengths and coregistration volume in ground meters', async () => {
  const {graph, input, output, run} = await setup();
  const geographicGrid = {
    width: 3,
    height: 2,
    affine: [0.001, 0, 10, 0, -0.001, 50] as const,
    pixelInterpretation: 'area' as const
  };
  const statistics = output(6, 'float32'),
    hypsometry = output(2, 'float32'),
    profiles = output(18, 'float32'),
    profileStatus = output(6, 'uint32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      id: 'geographic-catchment',
      grid: geographicGrid,
      cellSizeMode: 'geographic',
      elevation: input([6, 5, 4, 3, 2, 1], 'float32'),
      basinIds: input([0, 0, 0, 0, 0, 0], 'uint32'),
      basinCount: 1,
      // Row 0 flows east, then south on the last column; row 1 flows east.
      flowDirections: input([1, 1, 4, 1, 1, 0], 'uint32'),
      streams: input([1, 1, 1, 1, 1, 1], 'uint32'),
      histogramBins: 2,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: profiles.view,
        profileStatus: profileStatus.view
      }
    })
  );
  const fit = output(4, 'float32'),
    differences = output(12, 'float32'),
    validity = output(6, 'uint32'),
    volume = output(3, 'float32');
  graph.add(
    new GPUTerrainDEMCoregistration({
      id: 'geographic-coregistration',
      grid: geographicGrid,
      cellSizeMode: 'geographic',
      reference: input([0, 0, 0, 0, 0, 0], 'float32'),
      moving: input([1, 2, 2, 2, 2, 5], 'float32'),
      stableMask: input([1, 1, 1, 1, 1, 1], 'uint32'),
      differenceVariance: input([1, 1, 1, 1, 1, 1], 'float32'),
      output: {
        fit: fit.view,
        differences: differences.view,
        validity: validity.view,
        volume: volume.view
      }
    })
  );
  const destroy = run();
  try {
    const georeference = getRasterGridGeoreference(geographicGrid);
    const cellAreas = [0, 1].map(row => {
      const [cellX, cellY] = getRasterGroundCellSize('geographic', georeference, row + 0.5, 2);
      return cellX * cellY;
    });
    const expectedArea = 3 * (cellAreas[0] + cellAreas[1]);
    const actual = await statistics.read();
    expect(actual[0] / expectedArea).toBeCloseTo(1, 4);
    // Ground square meters (six cells of about 8,000 m2), not the 6e-6 square degrees of the affine.
    expect(actual[0]).toBeGreaterThan(1e4);
    expect((await hypsometry.read())[1] / expectedArea).toBeCloseTo(1, 4);
    // Stream length: three east moves of one cell width plus a south move in the last column.
    const [firstWidth, firstHeight] = getRasterGroundCellSize('geographic', georeference, 0.5, 2);
    const [secondWidth] = getRasterGroundCellSize('geographic', georeference, 1.5, 2);
    expect(actual[4] / (2 * firstWidth + firstHeight + 2 * secondWidth)).toBeCloseTo(1, 3);
    // The volume integrates the corrected differences over the ground area: median shift is 2.
    const volumes = await volume.read();
    expect(volumes[2] / expectedArea).toBeCloseTo(1, 4);
    expect(volumes[0] / (-cellAreas[0] + 3 * cellAreas[1])).toBeCloseTo(1, 3);
  } finally {
    destroy();
  }
});

it('depression hierarchy and branching flow honour validity and ground units', async () => {
  const {graph, input, output, run} = await setup();
  const geographicGrid = {
    width: 3,
    height: 1,
    affine: [0.001, 0, 10, 0, -0.001, 50] as const,
    pixelInterpretation: 'area' as const
  };
  const hierarchies = [false, true].map(useValidity => {
    const result = {
      nodes: output(12, 'uint32'),
      levels: output(6, 'float32'),
      cellNodes: output(3, 'uint32'),
      curves: output(12, 'float32'),
      status: output(3, 'uint32')
    };
    graph.add(
      new GPUTerrainDepressionHierarchy({
        id: `hierarchy-${useValidity}`,
        grid: geographicGrid,
        cellSizeMode: 'geographic',
        elevation: input([0, 2, useValidity ? -9999 : NaN], 'float32'),
        ...(useValidity ? {validity: input([1, 1, 0], 'uint32')} : {}),
        outlets: input([0, 0, 0], 'uint32'),
        stages: input([1, 3], 'float32'),
        nodeCapacity: 3,
        output: {
          nodes: result.nodes.view,
          levels: result.levels.view,
          cellNodes: result.cellNodes.view,
          curves: result.curves.view,
          status: result.status.view
        }
      })
    );
    return result;
  });
  const flows = [false, true].map(useValidity => {
    const result = {
      values: output(2, 'float32'),
      status: output(2, 'uint32'),
      unresolvedCount: output(1, 'uint32')
    };
    graph.add(
      new GPUTerrainBranchingFlow({
        id: `branching-${useValidity}`,
        grid: {...geographicGrid, width: 2},
        cellSizeMode: 'geographic',
        mode: 'expected-distance',
        // Cell 0 sends everything east to cell 1, which is a terminal.
        fractions: input([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'float32'),
        values: input([0, useValidity ? -1 : NaN], 'float32'),
        ...(useValidity ? {validity: input([1, 0], 'uint32')} : {}),
        output: {
          values: result.values.view,
          status: result.status.view,
          unresolvedCount: result.unresolvedCount.view
        }
      })
    );
    return result;
  });
  const distance = output(2, 'float32'),
    distanceStatus = output(2, 'uint32'),
    distanceUnresolved = output(1, 'uint32');
  graph.add(
    new GPUTerrainBranchingFlow({
      id: 'branching-distance',
      grid: {...geographicGrid, width: 2},
      cellSizeMode: 'geographic',
      mode: 'expected-distance',
      fractions: input([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'float32'),
      values: input([0, 0], 'float32'),
      output: {
        values: distance.view,
        status: distanceStatus.view,
        unresolvedCount: distanceUnresolved.view
      }
    })
  );
  const destroy = run();
  try {
    for (const group of [hierarchies, flows]) {
      for (const key of Object.keys(group[0]) as (keyof (typeof group)[0])[]) {
        expect(await (group[1][key] as TerrainOutput).read(), String(key)).toEqual(
          await (group[0][key] as TerrainOutput).read()
        );
      }
    }
    const georeference = getRasterGridGeoreference(geographicGrid);
    const [cellWidth, cellHeight] = getRasterGroundCellSize('geographic', georeference, 0.5, 1);
    // One finite basin of cells 0 and 1: stage 3 floods both, so the area is two ground cells.
    const curves = await hierarchies[0].curves.read();
    expect(
      Math.max(...curves.filter(Number.isFinite)) / (2 * cellWidth * cellHeight * 3)
    ).toBeLessThanOrEqual(1.0001);
    expect(curves.some(value => value > 1e3)).toBe(true);
    // Cell 0 is one ground cell width upstream of the terminal at cell 1.
    expect((await distance.read())[0] / cellWidth).toBeCloseTo(1, 4);
  } finally {
    destroy();
  }
});
