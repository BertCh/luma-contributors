// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {computeCostDistanceOnCPU} from '../../src/raster-analysis/cost-distance/cpu-cost-distance-operation';
import {computeCostDistancePathOnCPU} from '../../src/raster-analysis/cost-distance/cpu-cost-distance-path-operation';
import {getGPUCostDistanceParameterValues} from '../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {computeIsolinesOnCPU} from '../../src/raster-analysis/isolines/cpu-isolines-operation';
import {getGPUIsolinesParameterValues} from '../../src/raster-analysis/isolines/isolines-parameters';
import {computeStreamlinesOnCPU} from '../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../src/raster-analysis/flow-texture/streamlines-parameters';
import {makeGPURasterMetadata} from '../../src/raster-analysis/raster-grid/index';
import {getGPUPolygonRasterizationParameterValues} from '../../src/raster-analysis/polygon-rasterization/polygon-rasterization-parameters';
import {decodeStreamlineCase, streamlineCases} from '../oracles/parity-wp4-round3-fixtures';
import {createCensusRasterJoinFixture, expectCensusValues} from './census-raster-oracle-utils';

it('census point-zone-only fixture distinguishes query rows, cells and zone IDs', () => {
  const {grid, points, zones} = createCensusRasterJoinFixture();
  // Closed form: a north-up row covering [0,2) x [0,1) has top edge y=1 and
  // packed bottom edge y=0. Its two cell centers are (.5,.5) and (1.5,.5).
  expectCensusValues(Array.from(getGPUPolygonRasterizationParameterValues(grid)), [0, 0, 1, -1]);
  expectCensusValues(points, [1.5, 0.5, 0.5, 0.5, 1.5, 0.5, 2.5, 0.5]);
  expectCensusValues(zones, [1, 0]);
  const [stepX, , originX, , stepY, originY] = grid.affine;
  expectCensusValues(
    points.map((value, index) =>
      index % 2 === 0 ? (value - originX) / stepX : (value - originY) / stepY
    ),
    [1.5, 0.5, 0.5, 0.5, 1.5, 0.5, 2.5, 0.5]
  );
  const pointZones = Array.from({length: points.length / 2}, (_, index) => {
    const column = Math.floor((points[index * 2] - originX) / stepX);
    const row = Math.floor((points[index * 2 + 1] - originY) / stepY);
    return column >= 0 && column < grid.width && row === 0 ? zones[column] : 0xffffffff;
  });
  expectCensusValues(pointZones, [0, 1, 0, 0xffffffff]);
});

it('census CPU cost/path forms match the independently integrated three-cell line', () => {
  const grid = makeGPURasterMetadata({width: 3, height: 1});
  const result = computeCostDistanceOnCPU(
    {
      friction: {id: 'friction', format: 'float32', values: Float32Array.of(1, 1, 1)},
      sources: Uint32Array.of(0),
      parameters: getGPUCostDistanceParameterValues({}, grid),
      costBreaks: Float32Array.of(0.5, 1.5, Infinity)
    },
    {grid}
  );
  // Unit edge integrals give costs 0,1,2, west links, and three singleton bands.
  expectCensusValues(result.costs, [0, 1, 2]);
  expectCensusValues(result.backLinks, [0, 16, 16]);
  expectCensusValues(result.bands, [0, 1, 2]);
  expectCensusValues(result.bandCounts, [1, 1, 1]);
  expectCensusValues(result.converged, [1]);
  expectCensusValues(result.iterationLimitReached, [0]);
  for (const unresolved of [false, true]) {
    const path = computeCostDistancePathOnCPU(
      Uint32Array.of(0, 16, unresolved ? 0xfffffffe : 16),
      Uint32Array.of(2),
      grid,
      3
    );
    if (!unresolved) expectCensusValues(path.ids, [2, 1, 0]);
    expectCensusValues(path.count, [unresolved ? 0 : 3]);
    expectCensusValues(path.requiredCount, [unresolved ? 0 : 3]);
    expectCensusValues(path.overflow, [0]);
    expectCensusValues(path.iterationLimitReached, [unresolved ? 1 : 0]);
  }
});

for (const vertexCapacity of [2, 3]) {
  it(`census CPU stitched ramp matches exact geometry and capacity (vertices=${vertexCapacity})`, () => {
    const grid = makeGPURasterMetadata({width: 2, height: 3});
    const result = computeIsolinesOnCPU(
      {
        values: Float32Array.of(0, 2, 0, 2, 0, 2),
        levels: Float32Array.of(1),
        parameters: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 3]}, grid)
      },
      {grid, segmentCapacity: 2, vertexCapacity}
    );
    // The mid-ramp contour is the vertical line x=1 directed south, high values on its left.
    expectCensusValues(result.segments, [1, 1.5, 1, 0.5, 1, 2.5, 1, 1.5]);
    expectCensusValues(result.count, [2]);
    expectCensusValues(result.polylineCount, [vertexCapacity === 3 ? 1 : 0]);
    expectCensusValues(result.vertexCount, [vertexCapacity === 3 ? 3 : 0]);
    expectCensusValues(result.polylineOverflow, [vertexCapacity === 3 ? 0 : 1]);
    if (vertexCapacity === 3) {
      expectCensusValues(result.vertices, [1, 2.5, 1, 1.5, 1, 0.5]);
      expectCensusValues(result.polylineOffsets.slice(0, 2), [0, 3]);
      expectCensusValues(result.polylineLevels.slice(0, 1), [0]);
      expectCensusValues(result.polylineClosed.slice(0, 1), [0]);
    }
  });
}

it('census CPU streamline form matches the committed SciPy/Philox oracle', async () => {
  const scene = await decodeStreamlineCase(
    streamlineCases.find(
      candidate => candidate.name === 'horizontal-boundaries-and-competing-seeds'
    )!
  );
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.from(scene.field),
      parameters: getGPUStreamlinesParameterValues(scene.settings),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues(scene.settings)
    },
    {
      ...scene.config,
      fieldWidth: scene.fieldWidth,
      fieldHeight: scene.fieldHeight,
      maximumIterations: 64
    }
  );
  expectCensusValues(result.ids.slice(0, scene.ids.length), scene.ids);
  expectCensusValues(result.lineOffsets.slice(0, scene.offsets.length), scene.offsets);
  expectCensusValues(result.positions.slice(0, scene.points.length), scene.points, 5e-4);
  expectCensusValues(result.count, [scene.ids.length]);
  expectCensusValues(result.requiredCount, [scene.ids.length]);
  expectCensusValues(result.positionCount, [scene.points.length / 2]);
  expectCensusValues(result.overflow, [0]);
  expectCensusValues(result.candidateOverflow, [0]);
  expectCensusValues(result.iterationLimitReached, [0]);
  expectCensusValues(result.candidatePositions, scene.candidates, 5e-4);
});

it('census closed diamond distinguishes level index two from closure one', () => {
  const grid = makeGPURasterMetadata({width: 3, height: 3});
  const result = computeIsolinesOnCPU(
    {
      values: Float32Array.of(0, 0, 0, 0, 2, 0, 0, 0, 0),
      levels: Float32Array.of(-2, -1, 1),
      parameters: getGPUIsolinesParameterValues({levelCount: 3, extent: [0, 0, 3, 3]}, grid)
    },
    {grid, segmentCapacity: 4, vertexCapacity: 5}
  );
  // Only the third requested level crosses the four center-to-neighbor edges. The four
  // midpoint crossings close into one diamond, requiring four vertices plus its repeated start.
  expectCensusValues(result.count, [4]);
  expectCensusValues(result.polylineCount, [1]);
  expectCensusValues(result.vertexCount, [5]);
  expectCensusValues(result.polylineOffsets.slice(0, 2), [0, 5]);
  expectCensusValues(result.polylineLevels.slice(0, 1), [2]);
  expectCensusValues(result.polylineClosed.slice(0, 1), [1]);
});
