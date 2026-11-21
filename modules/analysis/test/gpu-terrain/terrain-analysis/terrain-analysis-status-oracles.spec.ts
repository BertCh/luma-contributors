// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {GPUTerrainCumulativeViewshed} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-cumulative-viewshed';
import {getGPUTerrainLineOfSightParameterValues} from '../../../src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createTerrainReviewFixture} from '../terrain-processing/terrain-review-fixture';
import {
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';

for (const traversal of ['march', 'pyramid'] as const) {
  it(`cumulative visibility counts every marginal pixel and clears on replay: ${traversal}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 12, height: 1});
    const values = new Float32Array(12);
    values[6] = 0.5;
    const elevation = fixture.input('float32', values);
    const observers = fixture.input('float32', [0, 0, 0, 0]);
    const visible = fixture.output('uint32', 12);
    const marginal = fixture.output('uint32', 12);
    const parameters = fixture.input(
      'float32',
      getGPUTerrainLineOfSightParameterValues(
        {observerHeight: 1, targetHeight: 0, curvatureCoefficient: 0, toleranceMeters: 0.5},
        grid
      )
    );
    visible.buffer.write(new Uint32Array(12).fill(17));
    marginal.buffer.write(new Uint32Array(12).fill(17));
    fixture.graph.add(
      new GPUTerrainCumulativeViewshed({
        grid,
        traversal,
        observersPerDispatch: 1,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: elevation.view}
        },
        observers: fixture.graph.createDataView(observers.view.buffer, {
          format: 'float32x2',
          length: 2
        }),
        parameters: parameters.view,
        output: {visibleCount: visible.view, marginalCount: marginal.view}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      // Exact rational oracle: terrain-analysis-status-oracles.oracle.py (command in file).
      // The tolerance band also includes unobstructed rays. On flat ground the last
      // interior sample gives target-plane clearance 1/(j-1), so columns 4..11 are
      // marginal at tolerance 1/2. With the wall, clearance is 1/3 and 1/4 at 4/5,
      // 7/10 at target 6, and j/12-1 at targets 7..11. Visible means clearance >=
      // tolerance; marginal means -tolerance <= clearance < tolerance. Columns 0/1
      // have no interior sample. A separate flat, zero-tolerance frame clears all
      // prior marginal counts, then the original wall frame restores them.
      const frames = [
        {
          name: 'wall with tolerance',
          wallHeight: 0.5,
          toleranceMeters: 0.5,
          observerColumns: [0, 0],
          visible: [2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0],
          marginal: [0, 0, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2]
        },
        {
          name: 'flat with tolerance',
          wallHeight: 0,
          toleranceMeters: 0.5,
          observerColumns: [0, 0],
          visible: [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
          marginal: [0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]
        },
        {
          name: 'flat without tolerance',
          wallHeight: 0,
          toleranceMeters: 0,
          observerColumns: [0, 0],
          visible: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
          marginal: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        },
        {
          // A marginal count cannot be computed as 2-visible: at column 10,
          // observer 0 is hidden and observer 10 sees itself, giving (1,0).
          // At column 4, clearance from 0 is 1/3 (marginal), while observer 10
          // is hidden behind the 2m wall, giving (0,1). Full arrays use Fraction geometry.
          name: 'opposite observers with hidden and marginal rays',
          wallHeight: 2,
          toleranceMeters: 0.5,
          observerColumns: [0, 10],
          visible: [1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 1],
          marginal: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        }
      ];
      for (const frame of [...frames, frames[0]]) {
        values[6] = frame.wallHeight;
        elevation.buffer.write(values);
        observers.buffer.write(
          new Float32Array([frame.observerColumns[0], 0, frame.observerColumns[1], 0])
        );
        parameters.buffer.write(
          getGPUTerrainLineOfSightParameterValues(
            {
              observerHeight: 1,
              targetHeight: 0,
              curvatureCoefficient: 0,
              toleranceMeters: frame.toleranceMeters
            },
            grid
          )
        );
        fixture.submit(compiled);
        expect(await visible.read(), frame.name).toEqual(frame.visible);
        expect(await marginal.read(), frame.name).toEqual(frame.marginal);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}

for (const algorithm of ['march', 'sweep'] as const) {
  it(`horizon validity writes every finite centre and clears invalid centres: ${algorithm}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const grid = makeGPURasterMetadata({width: 3, height: 3, cellSize: 1});
    const values = new Float32Array(9);
    values[4] = NaN;
    const elevation = fixture.input('float32', values);
    const validity = fixture.output('uint32', 9);
    validity.buffer.write(new Uint32Array(9).fill(17));
    fixture.graph.add(
      new GPUTerrainHorizon({
        grid,
        algorithm,
        directionCount: 4,
        maximumRadius: 1,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: elevation.view}
        },
        parameters: fixture.input('float32', getGPUTerrainHorizonParameterValues({}, grid)).view,
        output: {outputValidity: validity.view}
      })
    );
    const compiled = fixture.graph.compile();
    try {
      // A missing centre is invalid; missing neighbours only remove ray samples. Edges
      // with finite centres are valid even when a ray leaves the raster immediately.
      for (const excluded of [false, true, false]) {
        elevation.buffer.write(excluded ? new Float32Array(9).fill(NaN) : values);
        fixture.submit(compiled);
        expect(await validity.read()).toEqual(
          excluded ? [0, 0, 0, 0, 0, 0, 0, 0, 0] : [1, 1, 1, 1, 0, 1, 1, 1, 1]
        );
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
