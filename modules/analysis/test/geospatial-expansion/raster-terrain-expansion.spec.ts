// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterFillNoData} from '../../src/raster-analysis/gpu-raster-fill-nodata';
import {GPURasterRankFilter} from '../../src/raster-analysis/gpu-raster-rank-filter';
import {GPURasterFillHoles} from '../../src/raster-analysis/gpu-raster-reconstruction';
import {GPUTerrainFlowDistance} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {GPUConcaveHull} from '../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {GPUConstrainedDelaunayTriangulation} from '../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

async function setup() {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  function input<Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  function output<Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2' | 'uint32x3'>(
    length: number,
    format: Format
  ) {
    const components = format.endsWith('x2') ? 2 : format.endsWith('x3') ? 3 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length)
    };
  }
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
function grid(width: number, height: number): GPURasterMetadata {
  return {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'};
}

it('raster gap fill preserves donors, limits search and exposes provenance', async () => {
  const {graph, input, output, run} = await setup();
  const values = output(5, 'float32'),
    validity = output(5, 'uint32'),
    filled = output(5, 'uint32'),
    unresolvedCount = output(1, 'uint32');
  graph.add(
    new GPURasterFillNoData({
      grid: grid(5, 1),
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: input([2, NaN, 4, NaN, NaN], 'float32')}
      },
      maximumDistance: 1,
      output: {
        values: values.view,
        validity: validity.view,
        filled: filled.view,
        unresolvedCount: unresolvedCount.view
      }
    })
  );
  const destroy = run();
  const actual = Array.from(await readFloat32(values.buffer, 5));
  expect(actual.slice(0, 4)).toEqual([2, 3, 4, 4]);
  expect(actual[4]).toBeNaN();
  expect(Array.from(await readUint32(filled.buffer, 5))).toEqual([0, 1, 0, 1, 0]);
  expect(Array.from(await readUint32(unresolvedCount.buffer, 1))).toEqual([1]);
  destroy();
});

it('raster rank filter implements upper median and hole reconstruction preserves border background', async () => {
  const {graph, input, output, run} = await setup();
  const values = output(9, 'float32'),
    validity = output(9, 'uint32');
  graph.add(
    new GPURasterRankFilter({
      grid: grid(3, 3),
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: input([1, 2, 3, 4, 99, 6, 7, 8, 9], 'float32')}
      },
      radius: 1,
      output: {values: values.view, validity: validity.view}
    })
  );
  const holes = output(9, 'uint32'),
    converged = output(1, 'uint32');
  graph.add(
    new GPURasterFillHoles({
      grid: grid(3, 3),
      input: input([1, 1, 1, 1, 0, 1, 1, 1, 1], 'uint32'),
      maximumIterations: 4,
      output: {values: holes.view, status: {converged: converged.view}}
    })
  );
  const destroy = run();
  expect((await readFloat32(values.buffer, 9))[4]).toBe(6);
  expect(Array.from(await readUint32(holes.buffer, 9))).toEqual(Array(9).fill(1));
  expect(Array.from(await readUint32(converged.buffer, 1))).toEqual([1]);
  destroy();
});

it('D8 distances and extracted reaches retain cell identities and bounded failure', async () => {
  const {graph, input, output, run} = await setup();
  const directions = input([1, 1, 0], 'uint32');
  const distance = output(3, 'float32'),
    validity = output(3, 'uint32'),
    terminals = output(3, 'uint32');
  graph.add(
    new GPUTerrainFlowDistance({
      grid: grid(3, 1),
      flowDirections: directions,
      output: {distance: distance.view, validity: validity.view, terminalCells: terminals.view}
    })
  );
  const sourceCells = output(3, 'uint32'),
    targetCells = output(3, 'uint32'),
    vertexOffsets = output(4, 'uint32'),
    positions = output(6, 'float32x2'),
    lengths = output(3, 'float32'),
    reachCount = output(1, 'uint32'),
    vertexCount = output(1, 'uint32'),
    overflow = output(1, 'uint32'),
    unresolvedCount = output(1, 'uint32');
  graph.add(
    new GPUTerrainStreamNetwork({
      grid: grid(3, 1),
      flowDirections: directions,
      streams: input([1, 1, 1], 'uint32'),
      output: {
        sourceCells: sourceCells.view,
        targetCells: targetCells.view,
        vertexOffsets: vertexOffsets.view,
        positions: positions.view,
        lengths: lengths.view,
        reachCount: reachCount.view,
        vertexCount: vertexCount.view,
        status: {overflow: overflow.view},
        unresolvedCount: unresolvedCount.view
      }
    })
  );
  const destroy = run();
  expect(Array.from(await readFloat32(distance.buffer, 3))).toEqual([2, 1, 0]);
  expect(Array.from(await readUint32(terminals.buffer, 3))).toEqual([2, 2, 2]);
  expect(Array.from(await readUint32(reachCount.buffer, 1))).toEqual([1]);
  expect((await readUint32(sourceCells.buffer, 1))[0]).toBe(0);
  expect((await readUint32(targetCells.buffer, 1))[0]).toBe(2);
  expect(Array.from(await readFloat32(positions.buffer, 6))).toEqual([
    0.5, 0.5, 1.5, 0.5, 2.5, 0.5
  ]);
  destroy();
});

it('alpha hull retains square boundary and constrained Delaunay preserves polygon mesh', async () => {
  const {graph, input, output, run} = await setup();
  const positions = input([0, 0, 2, 0, 2, 2, 0, 2], 'float32x2');
  const triangles = output(3, 'uint32x3'),
    triangleValidity = output(3, 'uint32'),
    edges = output(9, 'uint32x2'),
    edgeValidity = output(9, 'uint32'),
    complete = output(1, 'uint32');
  graph.add(
    new GPUConcaveHull({
      positions,
      maximumCircumradius: 2,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: edges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: complete.view}
      }
    })
  );
  const indices = output(6, 'uint32'),
    valid = output(1, 'uint32'),
    converged = output(1, 'uint32');
  graph.add(
    new GPUConstrainedDelaunayTriangulation({
      positions,
      featureOffsets: input([0, 1], 'uint32'),
      polygonOffsets: input([0, 1], 'uint32'),
      ringOffsets: input([0, 4], 'uint32'),
      output: {indices: indices.view, valid: valid.view, status: {converged: converged.view}}
    })
  );
  const destroy = run();
  expect(Array.from(await readUint32(complete.buffer, 1))).toEqual([1]);
  expect(
    Array.from(await readUint32(edgeValidity.buffer, 9)).reduce((sum, value) => sum + value, 0)
  ).toBe(4);
  expect(Array.from(await readUint32(converged.buffer, 1))).toEqual([1]);
  expect(Array.from(await readUint32(valid.buffer, 1))).toEqual([1]);
  destroy();
});

import {
  GPURasterFractionalCoverage,
  GPURasterFractionalZonalStatistics
} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {GPURasterWalkingCost} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {
  GPURasterSkeleton,
  GPURasterMedialAxis
} from '../../src/raster-analysis/gpu-raster-skeleton';
import {GPUGroupConcaveHull} from '../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';

it('fractional coverage preserves polygon area and intensive/extensive quantity meanings', async () => {
  const {graph, input, output, run} = await setup();
  const coverage = {
    zoneIds: output(2, 'uint32'),
    cellIds: output(2, 'uint32'),
    fractions: output(2, 'float32'),
    count: output(1, 'uint32'),
    requiredCount: output(1, 'uint32'),
    overflow: output(1, 'uint32'),
    zoneValidity: output(1, 'uint32')
  };
  const coverageViews = {
    zoneIds: coverage.zoneIds.view,
    cellIds: coverage.cellIds.view,
    fractions: coverage.fractions.view,
    status: {
      count: coverage.count.view,
      requiredCount: coverage.requiredCount.view,
      overflow: coverage.overflow.view
    },
    zoneValidity: coverage.zoneValidity.view
  };
  graph.add(
    new GPURasterFractionalCoverage({
      grid: grid(2, 1),
      zones: {
        positions: input([0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1], 'float32x2'),
        featureOffsets: input([0, 1], 'uint32'),
        polygonOffsets: input([0, 1], 'uint32'),
        ringOffsets: input([0, 4], 'uint32')
      },
      output: coverageViews
    })
  );
  const totals = output(1, 'float32'),
    means = output(1, 'float32'),
    coveredArea = output(1, 'float32'),
    validArea = output(1, 'float32'),
    validity = output(1, 'uint32');
  graph.add(
    new GPURasterFractionalZonalStatistics({
      grid: grid(2, 1),
      input: {
        id: 'values',
        format: 'float32',
        storage: {kind: 'buffer', values: input([10, 30], 'float32')}
      },
      zoneCount: 1,
      coverage: coverageViews,
      quantity: 'extensive',
      output: {
        totals: totals.view,
        means: means.view,
        coveredArea: coveredArea.view,
        validArea: validArea.view,
        validity: validity.view
      }
    })
  );
  const destroy = run();
  expect(Array.from(await readFloat32(coverage.fractions.buffer, 2))).toEqual([0.5, 0.5]);
  expect((await readFloat32(totals.buffer, 1))[0]).toBe(20);
  expect((await readFloat32(means.buffer, 1))[0]).toBe(20);
  expect((await readFloat32(coveredArea.buffer, 1))[0]).toBe(1);
  destroy();
});

it('destination grid resampling uses affine centers and retains outside nodata', async () => {
  const {graph, input, output, run} = await setup();
  const values = output(3, 'float32'),
    validity = output(3, 'uint32');
  graph.add(
    new GPURasterReprojection({
      sourceGrid: grid(2, 1),
      destinationGrid: grid(3, 1),
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: input([10, 20], 'float32')}
      },
      method: 'nearest',
      output: {values: values.view, validity: validity.view}
    })
  );
  const destroy = run();
  expect(Array.from(await readFloat32(values.buffer, 3)).slice(0, 2)).toEqual([10, 20]);
  expect((await readFloat32(values.buffer, 3))[2]).toBeNaN();
  expect(Array.from(await readUint32(validity.buffer, 3))).toEqual([1, 1, 0]);
  destroy();
});

it('walking cost treats uphill/downhill as directed edges', async () => {
  const {graph, input, output, run} = await setup();
  const elevation = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: input([0, 1], 'float32')}
  };
  const friction = {
    id: 'friction',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: input([0, 0], 'float32')}
  };
  const results = [];
  for (let source = 0; source < 2; source++) {
    const costs = output(2, 'float32'),
      validity = output(2, 'uint32'),
      converged = output(1, 'uint32');
    results.push(costs);
    graph.add(
      new GPURasterWalkingCost({
        id: `walk-${source}`,
        grid: grid(2, 1),
        elevation,
        friction,
        sources: input(source === 0 ? [1, 0] : [0, 1], 'uint32'),
        model: {kind: 'grass', coefficients: [1, 2, -1, -1], downhillSlopeThreshold: -0.5},
        maximumIterations: 3,
        output: {costs: costs.view, validity: validity.view, status: {converged: converged.view}}
      })
    );
  }
  const destroy = run();
  expect(Array.from(await readFloat32(results[0].buffer, 2))).toEqual([0, 3]);
  expect(Array.from(await readFloat32(results[1].buffer, 2))).toEqual([2, 0]);
  destroy();
});

it('priority watershed resolves plateaus by marker discovery and exposes truncation', async () => {
  const {graph, input, output, run} = await setup();
  const labels = output(3, 'uint32'),
    converged = output(1, 'uint32');
  graph.add(
    new GPURasterWatershedSegmentation({
      grid: grid(3, 1),
      input: {
        id: 'surface',
        format: 'float32',
        storage: {kind: 'buffer', values: input([0, 1, 0], 'float32')}
      },
      markers: input([1, 0, 2], 'uint32'),
      output: {labels: labels.view, status: {converged: converged.view}}
    })
  );
  const destroy = run();
  expect(Array.from(await readUint32(labels.buffer, 3))).toEqual([1, 1, 2]);
  expect((await readUint32(converged.buffer, 1))[0]).toBe(1);
  destroy();
});

it('skeletonization preserves a single connected center and medial radii use padded Euclidean distance', async () => {
  const {graph, input, output, run} = await setup();
  const source = input(Array(9).fill(1), 'uint32');
  const skeleton = output(9, 'uint32'),
    converged = output(1, 'uint32'),
    medialAxis = output(9, 'uint32'),
    radius = output(9, 'float32');
  graph.add(
    new GPURasterSkeleton({
      grid: grid(3, 3),
      input: source,
      maximumIterations: 5,
      output: {skeleton: skeleton.view, status: {converged: converged.view}}
    })
  );
  graph.add(
    new GPURasterMedialAxis({
      grid: grid(3, 3),
      input: source,
      output: {medialAxis: medialAxis.view, radius: radius.view}
    })
  );
  const destroy = run();
  expect(Array.from(await readUint32(skeleton.buffer, 9))).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 0]);
  expect(Array.from(await readFloat32(radius.buffer, 9))).toEqual([1, 1, 1, 1, 2, 1, 1, 1, 1]);
  expect((await readUint32(medialAxis.buffer, 9))[4]).toBe(1);
  destroy();
});

it('group alpha hulls preserve independent point IDs and empty groups', async () => {
  const {graph, input, output, run} = await setup();
  const positions = input([0, 0, 1, 0, 0, 1, 10, 0, 11, 0, 10, 1], 'float32x2');
  const triangles = output(21, 'uint32x3'),
    triangleValidity = output(21, 'uint32'),
    boundaryEdges = output(63, 'uint32x2'),
    edgeValidity = output(63, 'uint32'),
    converged = output(3, 'uint32');
  graph.add(
    new GPUGroupConcaveHull({
      positions,
      groupIds: input([0, 0, 0, 1, 1, 1], 'uint32'),
      groupCount: 3,
      maximumCircumradius: 1,
      output: {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: boundaryEdges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: converged.view}
      }
    })
  );
  const destroy = run();
  const actual = Array.from(await readUint32(edgeValidity.buffer, 63));
  expect(
    [actual.slice(0, 21), actual.slice(21, 42), actual.slice(42)].map(values =>
      values.reduce((sum, value) => sum + value, 0)
    )
  ).toEqual([3, 3, 0]);
  expect(Array.from(await readUint32(converged.buffer, 3))).toEqual([1, 1, 1]);
  destroy();
});

import reference from '../oracles/raster-terrain-expansion.json';
import {GPURasterReconstruction} from '../../src/raster-analysis/gpu-raster-reconstruction';

it('raster and terrain expansion agrees with SciPy, scikit-image and NetworkX reference fixtures', async () => {
  const {graph, input, output, run} = await setup();
  const median = output(9, 'float32'),
    medianValidity = output(9, 'uint32');
  graph.add(
    new GPURasterRankFilter({
      id: 'reference-median',
      grid: grid(3, 3),
      input: {
        id: 'median',
        format: 'float32',
        storage: {kind: 'buffer', values: input(reference.median.values, 'float32')}
      },
      radius: 1,
      output: {values: median.view, validity: medianValidity.view}
    })
  );
  const holes = output(25, 'uint32'),
    holesConverged = output(1, 'uint32');
  graph.add(
    new GPURasterFillHoles({
      id: 'reference-holes',
      grid: grid(5, 5),
      input: input(reference.holes.values, 'uint32'),
      maximumIterations: 12,
      output: {values: holes.view, status: {converged: holesConverged.view}}
    })
  );
  const reconstructed = output(9, 'float32'),
    reconstructionValidity = output(9, 'uint32'),
    reconstructionConverged = output(1, 'uint32');
  graph.add(
    new GPURasterReconstruction({
      id: 'reference-reconstruction',
      grid: grid(3, 3),
      marker: input(reference.reconstruction.marker, 'float32'),
      mask: input(reference.reconstruction.mask, 'float32'),
      maximumIterations: 12,
      output: {
        values: reconstructed.view,
        validity: reconstructionValidity.view,
        status: {converged: reconstructionConverged.view}
      }
    })
  );
  const distance = output(6, 'float32'),
    flowValidity = output(6, 'uint32'),
    terminals = output(6, 'uint32');
  graph.add(
    new GPUTerrainFlowDistance({
      id: 'reference-flow',
      grid: grid(3, 2),
      flowDirections: input(reference.flow.directions, 'uint32'),
      output: {distance: distance.view, validity: flowValidity.view, terminalCells: terminals.view}
    })
  );
  const labels = output(15, 'uint32'),
    segmentationConverged = output(1, 'uint32');
  graph.add(
    new GPURasterWatershedSegmentation({
      id: 'reference-segmentation',
      grid: grid(5, 3),
      input: {
        id: 'segmentation',
        format: 'float32',
        storage: {kind: 'buffer', values: input(reference.segmentation.values, 'float32')}
      },
      markers: input(reference.segmentation.markers, 'uint32'),
      output: {labels: labels.view, status: {converged: segmentationConverged.view}}
    })
  );
  const destroy = run();
  expect(Array.from(await readFloat32(median.buffer, 9))).toEqual(reference.median.expected);
  expect(Array.from(await readUint32(holes.buffer, 25))).toEqual(reference.holes.expected);
  expect(Array.from(await readFloat32(reconstructed.buffer, 9))).toEqual(
    reference.reconstruction.expected
  );
  expect(Array.from(await readFloat32(distance.buffer, 6))).toEqual(reference.flow.expected);
  expect(Array.from(await readUint32(labels.buffer, 15))).toEqual(reference.segmentation.expected);
  destroy();
});

it('Shapely fractional intersections preserve holes and required capacity evidence', async () => {
  for (const capacity of [9, 1]) {
    const {graph, input, output, run} = await setup();
    const zoneIds = output(capacity, 'uint32'),
      cellIds = output(capacity, 'uint32'),
      fractions = output(capacity, 'float32'),
      count = output(1, 'uint32'),
      requiredCount = output(1, 'uint32'),
      overflow = output(1, 'uint32'),
      zoneValidity = output(1, 'uint32');
    graph.add(
      new GPURasterFractionalCoverage({
        grid: grid(3, 3),
        zones: {
          positions: input(reference.fractional.positions, 'float32x2'),
          featureOffsets: input([0, 1], 'uint32'),
          polygonOffsets: input([0, 2], 'uint32'),
          ringOffsets: input(reference.fractional.ringOffsets, 'uint32')
        },
        output: {
          zoneIds: zoneIds.view,
          cellIds: cellIds.view,
          fractions: fractions.view,
          status: {count: count.view, requiredCount: requiredCount.view, overflow: overflow.view},
          zoneValidity: zoneValidity.view
        }
      })
    );
    const destroy = run();
    expect((await readUint32(requiredCount.buffer, 1))[0]).toBe(8);
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(capacity === 1 ? 1 : 0);
    if (capacity === 9) {
      const ids = await readUint32(cellIds.buffer, 8);
      const actual = await readFloat32(fractions.buffer, 8);
      for (let tuple = 0; tuple < 8; tuple++)
        expect(actual[tuple]).toBeCloseTo(reference.fractional.expected[ids[tuple]], 6);
      expect(Array.from(actual).reduce((sum, value) => sum + value, 0)).toBeCloseTo(
        reference.fractional.area,
        6
      );
    }
    destroy();
  }
});

it('flow stop sentinels, cycles, and bounded morphology remain unresolved', async () => {
  const {graph, input, output, run} = await setup();
  const distance = output(3, 'float32'),
    validity = output(3, 'uint32'),
    terminals = output(3, 'uint32');
  graph.add(
    new GPUTerrainFlowDistance({
      grid: grid(3, 1),
      flowDirections: input([1, 1, 0], 'uint32'),
      streams: input([0, 0xffffffff, 0], 'uint32'),
      output: {distance: distance.view, validity: validity.view, terminalCells: terminals.view}
    })
  );
  const cycleDistance = output(2, 'float32'),
    cycleValidity = output(2, 'uint32'),
    cycleTerminals = output(2, 'uint32');
  graph.add(
    new GPUTerrainFlowDistance({
      id: 'cycle',
      grid: grid(2, 1),
      flowDirections: input([1, 16], 'uint32'),
      output: {
        distance: cycleDistance.view,
        validity: cycleValidity.view,
        terminalCells: cycleTerminals.view
      }
    })
  );
  const reconstructed = output(6, 'float32'),
    reconstructionValidity = output(6, 'uint32'),
    complete = output(1, 'uint32'),
    limitReached = output(1, 'uint32'),
    iterationCount = output(1, 'uint32');
  graph.add(
    new GPURasterReconstruction({
      grid: grid(6, 1),
      marker: input([1, 0, 0, 0, 0, 0], 'float32'),
      mask: input([1, 1, 1, 1, 1, 1], 'float32'),
      maximumIterations: 1,
      output: {
        values: reconstructed.view,
        validity: reconstructionValidity.view,
        status: {
          converged: complete.view,
          iterationLimitReached: limitReached.view,
          iterationCount: iterationCount.view
        }
      }
    })
  );
  const destroy = run();
  expect(Array.from(await readUint32(terminals.buffer, 3))).toEqual([0xfffffffe, 0xfffffffe, 2]);
  expect(Array.from(await readUint32(cycleTerminals.buffer, 2))).toEqual([0xfffffffe, 0xfffffffe]);
  // An unconverged run publishes its current values (the front advanced two cells) with
  // converged = 0 and the limit flag set.
  expect((await readUint32(complete.buffer, 1))[0]).toBe(0);
  expect((await readUint32(limitReached.buffer, 1))[0]).toBe(1);
  expect((await readUint32(iterationCount.buffer, 1))[0]).toBe(2);
  expect(Array.from(await readUint32(reconstructionValidity.buffer, 6))).toEqual([
    1, 1, 1, 1, 1, 1
  ]);
  expect(Array.from(await readFloat32(reconstructed.buffer, 6))).toEqual([1, 1, 1, 0, 0, 0]);
  destroy();
});

import {compileProjectionProgram} from '@luma.gl/graph/gpu-project';

it('reprojection composes a GPU inverse program and restores its destination origin', async () => {
  const {graph, input, output, run} = await setup();
  const values = output(2, 'float32'),
    validity = output(2, 'uint32');
  const sourceGrid = {...grid(2, 1), affine: [1, 0, 10, 0, 1, 0] as const};
  const projection = compileProjectionProgram({
    precision: 'local-f32',
    destinationOrigin: [10, 0],
    operations: [{type: 'affine', scale: [1, 1], offset: [10, 0]}]
  });
  const contributor = new GPURasterReprojection({
    sourceGrid,
    destinationGrid: grid(2, 1),
    input: {
      id: 'source',
      format: 'float32',
      storage: {kind: 'buffer', values: input([10, 20], 'float32')}
    },
    inverseProjection: projection,
    method: 'bilinear',
    output: {values: values.view, validity: validity.view}
  });
  graph.add(contributor);
  const destroy = run();
  expect(Array.from(await readFloat32(values.buffer, 2))).toEqual([10, 20]);
  expect(Array.from(await readUint32(validity.buffer, 2))).toEqual([1, 1]);
  destroy();
  contributor.destroy();
});

it('constrained Lawson refinement selects the Shapely Delaunay diagonal', async () => {
  const {graph, input, output, run} = await setup();
  const indices = output(6, 'uint32'),
    valid = output(1, 'uint32'),
    converged = output(1, 'uint32');
  graph.add(
    new GPUConstrainedDelaunayTriangulation({
      positions: input([0, 0, 3, 0, 2, 1, 0, 2], 'float32x2'),
      featureOffsets: input([0, 1], 'uint32'),
      polygonOffsets: input([0, 1], 'uint32'),
      ringOffsets: input([0, 4], 'uint32'),
      output: {indices: indices.view, valid: valid.view, status: {converged: converged.view}}
    })
  );
  const destroy = run();
  const actual = Array.from(await readUint32(indices.buffer, 6));
  expect((await readUint32(converged.buffer, 1))[0]).toBe(1);
  expect([actual.slice(0, 3).sort(), actual.slice(3).sort()].sort()).toEqual([
    [0, 1, 2],
    [0, 2, 3]
  ]);
  destroy();
});

const TOBLER_FLAT_STEP_COST = 0.6 * Math.exp(3.5 * 0.05);

it('walking cost publishes partial costs with converged = 0 when the iteration limit is reached', async () => {
  const {graph, input, output, run} = await setup();
  const elevation = {
    id: 'elevation',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: input([0, 0, 0, 0, 0, 0], 'float32')}
  };
  const friction = {
    id: 'friction',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: input([0, 0, 0, 0, 0, 0], 'float32')}
  };
  const sources = input([1, 0, 0, 0, 0, 0], 'uint32');
  const runs = [2, 100].map(maximumIterations => {
    const result = {
      costs: output(6, 'float32'),
      validity: output(6, 'uint32'),
      converged: output(1, 'uint32'),
      limitReached: output(1, 'uint32'),
      iterationCount: output(1, 'uint32')
    };
    const walkingCost = new GPURasterWalkingCost({
      id: `walk-limit-${maximumIterations}`,
      grid: grid(6, 1),
      elevation,
      friction,
      sources,
      maximumIterations,
      output: {
        costs: result.costs.view,
        validity: result.validity.view,
        status: {
          converged: result.converged.view,
          iterationLimitReached: result.limitReached.view,
          iterationCount: result.iterationCount.view
        }
      }
    });
    expect(walkingCost.estimatedCommandNodeCount).toBeGreaterThan(maximumIterations);
    graph.add(walkingCost);
    return result;
  });
  const destroy = run();
  const [partial, complete] = runs;
  // Three rounds move the front three cells; the rest is still unreached (+Infinity).
  const partialCosts = Array.from(await readFloat32(partial.costs.buffer, 6));
  expect(partialCosts[0]).toBe(0);
  for (const cell of [1, 2, 3]) {
    expect(partialCosts[cell]).toBeCloseTo(cell * TOBLER_FLAT_STEP_COST, 4);
  }
  expect(partialCosts.slice(4)).toEqual([Infinity, Infinity]);
  expect(Array.from(await readUint32(partial.validity.buffer, 6))).toEqual([1, 1, 1, 1, 1, 1]);
  expect((await readUint32(partial.converged.buffer, 1))[0]).toBe(0);
  expect((await readUint32(partial.limitReached.buffer, 1))[0]).toBe(1);
  expect((await readUint32(partial.iterationCount.buffer, 1))[0]).toBe(3);
  // With a generous limit the GPU gate stops dispatching after the verifying round: five rounds
  // move the front, one more changes nothing.
  const completeCosts = Array.from(await readFloat32(complete.costs.buffer, 6));
  expect(completeCosts[5]).toBeCloseTo(5 * TOBLER_FLAT_STEP_COST, 4);
  expect((await readUint32(complete.converged.buffer, 1))[0]).toBe(1);
  expect((await readUint32(complete.limitReached.buffer, 1))[0]).toBe(0);
  expect((await readUint32(complete.iterationCount.buffer, 1))[0]).toBe(6);
  destroy();
});

it('skeleton and fill holes publish their current values when the iteration limit is reached', async () => {
  const {graph, input, output, run} = await setup();
  const skeleton = output(81, 'uint32'),
    skeletonConverged = output(1, 'uint32'),
    skeletonLimit = output(1, 'uint32');
  graph.add(
    new GPURasterSkeleton({
      id: 'thin-limit',
      grid: grid(9, 9),
      input: input(Array(81).fill(1), 'uint32'),
      maximumIterations: 1,
      output: {
        skeleton: skeleton.view,
        status: {converged: skeletonConverged.view, iterationLimitReached: skeletonLimit.view}
      }
    })
  );
  // Background corridor that touches the border at both ends: two rounds reach only the cells
  // next to each end, so the middle is still reported as a hole.
  const holes = output(33, 'uint32'),
    holesConverged = output(1, 'uint32'),
    holesCount = output(1, 'uint32');
  const holeInput = Array(33).fill(1);
  for (let column = 0; column < 11; column++) holeInput[11 + column] = 0;
  graph.add(
    new GPURasterFillHoles({
      id: 'holes-limit',
      grid: grid(11, 3),
      input: input(holeInput, 'uint32'),
      maximumIterations: 1,
      output: {
        values: holes.view,
        status: {converged: holesConverged.view, iterationCount: holesCount.view}
      }
    })
  );
  const destroy = run();
  const thinned = Array.from(await readUint32(skeleton.buffer, 81));
  expect((await readUint32(skeletonConverged.buffer, 1))[0]).toBe(0);
  expect((await readUint32(skeletonLimit.buffer, 1))[0]).toBe(1);
  expect(thinned.every(value => value === 0 || value === 1)).toBe(true);
  expect(thinned.some(value => value === 0)).toBe(true);
  expect(thinned.some(value => value === 1)).toBe(true);
  expect((await readUint32(holesConverged.buffer, 1))[0]).toBe(0);
  expect((await readUint32(holesCount.buffer, 1))[0]).toBe(2);
  const filled = Array.from(await readUint32(holes.buffer, 33));
  expect(filled.every(value => value === 0 || value === 1)).toBe(true);
  // Corridor cells beside the border are reached (not holes); the middle one is not reached.
  expect(filled[11 + 1]).toBe(0);
  expect(filled[11 + 5]).toBe(1);
  destroy();
});
