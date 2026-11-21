// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {checkTerrainGPUStatus} from '../terrain-status-oracles-test-utils';
import {getTerrainHydrologyStatusOracles} from './terrain-hydrology-status-oracles-fixtures';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect} from 'vitest';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {GPUTerrainHeightAboveDrainage} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {addDrainageNetworkRecipe} from '../../../src/gpu-spatial-analysis/recipes/drainage-network-recipe';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createTerrainReviewFixture} from '../terrain-processing/terrain-review-fixture';

for (const oracle of getTerrainHydrologyStatusOracles()) {
  it(`GPU hydrology independent status oracle: ${oracle.name}`, async () => {
    await checkTerrainGPUStatus(oracle);
  });
}

for (const maximumFillIterations of [1, 2]) {
  it(`drainage recipe exposes incomplete fill through its flat-routing stage: ${maximumFillIterations}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const grid = {
      width: 3,
      height: 3,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const elevation = fixture.input('float32', [2, 2, 2, 2, 0, 2, 2, 2, 2]);
    const recipe = addDrainageNetworkRecipe(fixture.graph, {
      flow: {
        grid,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: elevation.view}
        },
        parameters: fixture.input(
          'float32',
          getGPUTerrainFlowParameterValues({fillEpsilon: 0.25}, grid)
        ).view,
        fillDepressions: true,
        resolveFlats: true,
        maximumFillIterations,
        maximumFlatIterations: 2
      },
      maximumSteps: 9
    });
    const flag = fixture.output('uint32', 1),
      directions = fixture.output('uint32', 9);
    flag.buffer.write(new Uint32Array([17]));
    expect(
      recipe.status.stages.find(stage => stage.stage === 'flat-routing')?.status
    ).toMatchObject({converged: recipe.intermediates.flowStatus.flatsConverged});
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'publish-flat-routing',
        operation: 'TerrainStatusOracle',
        invocationCount: 9,
        bindings: [
          {
            name: 'source',
            view: recipe.intermediates.flowStatus.flatsConverged!,
            type: 'u32',
            access: 'read'
          },
          {name: 'directions', view: recipe.outputs.flowDirections, type: 'u32', access: 'read'},
          {name: 'flagOut', view: flag.view, type: 'u32', access: 'read_write'},
          {name: 'directionsOut', view: directions.view, type: 'u32', access: 'read_write'}
        ],
        body: 'directionsOut[directionsOutOffset + index] = directions[directionsOffset + index]; if (index == 0u) { flagOut[flagOutOffset] = source[sourceOffset]; }'
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (const bowl of [true, false, true]) {
        elevation.buffer.write(new Float32Array([2, 2, 2, 2, bowl ? 0 : NaN, 2, 2, 2, 2]));
        fixture.submit(compiled);
        expect(await flag.read()).toEqual([bowl && maximumFillIterations === 1 ? 0 : 1]);
        expect(await directions.read()).toEqual(
          !bowl
            ? [0, 0, 0, 0, 0xffffffff, 0, 0, 0, 0]
            : maximumFillIterations === 1
              ? [
                  0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe,
                  0xfffffffe, 0xfffffffe, 0xfffffffe
                ]
              : [0, 0, 0, 0, 1, 0, 0, 0, 0]
        );
      }
    } finally {
      compiled.destroy();
      for (const contributor of recipe.contributors)
        if ('destroy' in contributor && typeof contributor.destroy === 'function')
          contributor.destroy();
      fixture.destroy();
    }
  });
}

for (const [directionEncoding, directions] of [
  ['esri', [1, 16]],
  ['whitebox', [2, 32]],
  ['taudem', [1, 5]]
] as const) {
  for (const includeDrainageCells of [true, false]) {
    it(`D8 HAND cycle is unresolved until a stream intercepts it: ${directionEncoding}/${includeDrainageCells}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createTerrainReviewFixture(device);
      const streams = fixture.input('uint32', [0, 0]);
      const drainage = fixture.output('uint32', 2),
        heights = fixture.output('float32', 2);
      const converged = fixture.output('uint32', 1),
        limit = fixture.output('uint32', 1);
      for (const output of [drainage, converged, limit])
        output.buffer.write(new Uint32Array(output.view.length).fill(17));
      const contributor = new GPUTerrainHeightAboveDrainage({
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
        maximumIterations: 2,
        directionEncoding,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {kind: 'buffer', values: fixture.input('float32', [5, 3]).view}
        },
        flowDirections: fixture.input('uint32', directions).view,
        streams: streams.view,
        output: {
          drainageCells: includeDrainageCells ? drainage.view : undefined,
          heightAboveDrainage: heights.view,
          status: {converged: converged.view, iterationLimitReached: limit.view}
        }
      });
      fixture.graph.add(contributor);
      const compiled = fixture.graph.compile();
      try {
        // 0↔1 has no stream or terminal despite stable doubled pointers. A stream
        // at 0 cuts that cycle: both cells reach 0, with height drops 5-5 and 3-5.
        for (const intercepted of [false, true, false]) {
          streams.buffer.write(new Uint32Array(intercepted ? [1, 0] : [0, 0]));
          fixture.submit(compiled);
          expect(await converged.read()).toEqual([intercepted ? 1 : 0]);
          expect(await limit.read()).toEqual([0]);
          if (includeDrainageCells)
            expect(await drainage.read()).toEqual(intercepted ? [0, 0] : [0xfffffffe, 0xfffffffe]);
          expect(await heights.read()).toEqual(intercepted ? [0, -2] : [NaN, NaN]);
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}

for (const {reachCapacity, vertexCapacity, storedReaches, expectedOverflow} of [
  {reachCapacity: 1, vertexCapacity: 2, storedReaches: 1, expectedOverflow: 1},
  {reachCapacity: 2, vertexCapacity: 3, storedReaches: 1, expectedOverflow: 1},
  {reachCapacity: 2, vertexCapacity: 4, storedReaches: 2, expectedOverflow: 0}
]) {
  it(`stream network overflow counts two independently drawn disjoint reaches: reaches ${reachCapacity}, vertices ${vertexCapacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createTerrainReviewFixture(device);
    const grid = {
      width: 3,
      height: 2,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const streams = fixture.input('uint32', [1, 1, 0, 1, 1, 0]);
    const reachCount = fixture.output('uint32', 1),
      vertexCount = fixture.output('uint32', 1),
      overflow = fixture.output('uint32', 1),
      unresolved = fixture.output('uint32', 1);
    for (const output of [reachCount, vertexCount, overflow, unresolved])
      output.buffer.write(new Uint32Array([17]));
    const positionStorage = fixture.output('float32', vertexCapacity * 2);
    fixture.graph.add(
      new GPUTerrainStreamNetwork({
        grid,
        flowDirections: fixture.input('uint32', [1, 0, 0, 1, 0, 0]).view,
        streams: streams.view,
        output: {
          sourceCells: fixture.output('uint32', reachCapacity).view,
          targetCells: fixture.output('uint32', reachCapacity).view,
          vertexOffsets: fixture.output('uint32', reachCapacity + 1).view,
          positions: fixture.graph.createDataView(positionStorage.view.buffer, {
            format: 'float32x2',
            length: vertexCapacity
          }),
          lengths: fixture.output('float32', reachCapacity).view,
          reachCount: reachCount.view,
          vertexCount: vertexCount.view,
          unresolvedCount: unresolved.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      // Exactly two disjoint reaches, 0→1 and 3→4, each with two vertices. The
      // two non-stream cells are excluded. Three vertices can store only the first
      // complete reach, even with two reach slots. Removing streams empties all counts.
      for (const present of [true, false, true]) {
        streams.buffer.write(new Uint32Array(present ? [1, 1, 0, 1, 1, 0] : [0, 0, 0, 0, 0, 0]));
        fixture.submit(compiled);
        expect(await reachCount.read()).toEqual([present ? storedReaches : 0]);
        expect(await vertexCount.read()).toEqual([present ? storedReaches * 2 : 0]);
        expect(await overflow.read()).toEqual([present ? expectedOverflow : 0]);
        expect(await unresolved.read()).toEqual([0]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
