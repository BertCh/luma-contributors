// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_GRID_COORDINATE_WGSL} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from './gpu-contributor-test-utils';
import {
  createGridCoordinateFixture,
  getFloat32GridCoordinate,
  GRID_COORDINATE_CASES,
  type GridCoordinateReader
} from './grid-coordinate-bump-fixtures';
import {computeGridDensity} from '../gpu-spatial-analysis/point-density/point-density-oracle';
import {findFlowZone} from '../gpu-network/flow-aggregation/flow-aggregation-oracle';
import {computeEncountersOracle} from '../gpu-spatial-analysis/trajectory-encounters/encounters-oracle';
import {interpolateInverseDistanceWeightingOnCPU} from '../gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-oracle';
import {
  isInsideRectangle,
  computeRegionStatistics
} from '../gpu-spatial-analysis/region-statistics/region-statistics-fixtures';

/** Assert the actual compiled reader sources use the public helper once, without private reads. */
export function expectGridCoordinateReaderSource(reader: GridCoordinateReader): void {
  const device = createNullWebGPUDevice();
  const fixture = createGridCoordinateFixture(device, reader, GRID_COORDINATE_CASES[0]);
  try {
    const sources = getGPUCommandNodeKernelSources(fixture.readerNodes).filter(kernel =>
      kernel.source.includes('fn getCoordinate(')
    );
    expect(sources.length).toBeGreaterThan(0);
    for (const {source} of sources) {
      expect(source).toContain(GPU_GRID_COORDINATE_WGSL);
      expect(source.match(/\bfn\s+getCoordinate\s*\(/g)).toHaveLength(1);
    }
    // Validate the actual headless fixture's resource declarations and scheduling on a null device.
    Object.defineProperty(device, 'props', {
      value: {...device.props, _cacheShaders: false, _cachePipelines: false}
    });
    Object.defineProperty(device, 'createComputePipeline', {
      value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
    });
    const compiled = fixture.graph.compile();
    compiled.destroy();
  } finally {
    fixture.destroy();
    device.destroy();
  }
}

/** Retain one pin per reader, against generated shader text rather than private upstream files. */
export function testGridCoordinateReader(reader: GridCoordinateReader): void {
  it(`${reader} includes the public grid coordinate export verbatim and defines it exactly once`, () => {
    expectGridCoordinateReaderSource(reader);
  });

  it.each(
    GRID_COORDINATE_CASES
  )(`${reader} CPU reference finds the published builder's point: $name`, fixture => {
    const {value, minimum, maximum, size, cell} = fixture;
    const coordinate = getFloat32GridCoordinate(value, minimum, maximum, size);
    expect(coordinate).toBe(cell);
    // A reference CSR build with a single occupied cell. Reading a neighbouring cell loses row 0.
    const cells = Array.from({length: size}, () => [] as number[]);
    cells[cell].push(0);
    expect(cells[coordinate]).toEqual([0]);
    if (reader === 'density') {
      const result = computeGridDensity([value, 0.5], [7], [minimum, 0, maximum, 1], [size, 1]);
      expect(result.counts[cell]).toBe(1);
      expect(result.sums[cell]).toBe(7);
      expect(result.counts.reduce((total, count) => total + count, 0)).toBe(1);
    } else if (reader === 'flow') {
      expect(
        findFlowZone(
          {kind: 'grid', bounds: [minimum, 0, maximum, 1], gridSize: [size, 1]},
          value,
          0.5
        )
      ).toBe(cell);
    } else if (reader === 'encounters') {
      expect(
        computeEncountersOracle(Float32Array.of(value, 0.5, value, 0.5), 2, 1, 0, undefined, [
          minimum,
          0,
          maximum,
          1
        ])
      ).toEqual([{track: 0, partner: 1, firstBucket: 0, minimumDistance: 0, bucketCount: 1}]);
    } else if (reader === 'regions') {
      const point = Math.fround(value);
      const rows = isInsideRectangle(point, 0.5, [point, 0, point, 1]) ? [0] : [];
      expect(rows).toEqual([0]);
      expect(computeRegionStatistics(rows, Float32Array.of(7), 0).sum).toBe(7);
    } else if (reader === 'interpolation') {
      const center = Math.fround(value);
      const result = interpolateInverseDistanceWeightingOnCPU({
        positions: Float32Array.of(value, 0.5),
        values: Float32Array.of(7),
        width: 1,
        height: 1,
        indexBounds: [minimum, 0, maximum, 1],
        extent: [center - 0.5, 0, center + 0.5, 1],
        searchRadius: 0,
        power: 2,
        neighborCount: 0,
        minimumNeighborCount: 1
      });
      expect([...result.counts]).toEqual([1]);
      expect([...result.values]).toEqual([7]);
    }
  });
}
