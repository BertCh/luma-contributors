// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUElementwise, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {addDrainageNetworkRecipe} from '../../../src/gpu-spatial-analysis/recipes/drainage-network-recipe';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {RecipeTestFixture} from './recipe-harness';

function publish<Format extends 'float32' | 'uint32'>(
  fixture: RecipeTestFixture,
  id: string,
  source: GraphDataView<Format>
) {
  const output = fixture.output(id, source.format, source.length);
  fixture.graph.add(
    new GPUElementwise({id, operation: 'copy', input: source, output: output.view})
  );
  return output;
}

it('measures geographic drainage distances and reach lengths in ground metres', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'geographic-drainage');
  try {
    const grid: GPURasterMetadata = {
      width: 2,
      height: 1,
      affine: [1, 0, 0, 0, -1, 0.5],
      pixelInterpretation: 'area'
    };
    const {outputs} = addDrainageNetworkRecipe(fixture.graph, {
      flow: {
        grid,
        elevation: {
          id: 'dem',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: fixture.input('dem', new Float32Array([2, 1]), 'float32', 2)
          }
        },
        parameters: fixture.parameters(
          'flow-parameters',
          'float32',
          getGPUTerrainFlowParameterValues({streamThreshold: 1}, grid)
        ),
        cellSizeMode: 'geographic'
      }
    });

    const standaloneDistance = fixture.output('standalone-distance', 'float32', 2);
    fixture.graph.add(
      new GPUTerrainFlowDistance({
        grid,
        flowDirections: outputs.flowDirections,
        cellSizeMode: 'geographic',
        output: {
          distance: standaloneDistance.view,
          validity: fixture.output('standalone-validity', 'uint32', 2).view,
          terminalCells: fixture.output('standalone-terminals', 'uint32', 2).view
        }
      })
    );
    const standaloneLength = fixture.output('standalone-length', 'float32', 2);
    const standaloneReachCount = fixture.output('standalone-reach-count', 'uint32', 1);
    fixture.graph.add(
      new GPUTerrainStreamNetwork({
        grid,
        flowDirections: outputs.flowDirections,
        streams: outputs.streams,
        cellSizeMode: 'geographic',
        output: {
          sourceCells: fixture.output('standalone-sources', 'uint32', 2).view,
          targetCells: fixture.output('standalone-targets', 'uint32', 2).view,
          vertexOffsets: fixture.output('standalone-offsets', 'uint32', 3).view,
          positions: fixture.output('standalone-positions', 'float32x2', 4).view,
          lengths: standaloneLength.view,
          reachCount: standaloneReachCount.view,
          vertexCount: fixture.output('standalone-vertex-count', 'uint32', 1).view,
          status: {overflow: fixture.output('standalone-overflow', 'uint32', 1).view},
          unresolvedCount: fixture.output('standalone-unresolved', 'uint32', 1).view
        }
      })
    );

    const distance = publish(fixture, 'recipe-distance', outputs.distance.distance);
    const reachCount = publish(fixture, 'recipe-reach-count', outputs.network.reachCount);
    const reachLength = publish(fixture, 'recipe-length', outputs.network.lengths);
    const summaryCount = publish(fixture, 'recipe-summary-count', outputs.summary.count);
    const meanDistance = publish(
      fixture,
      'recipe-summary-distance',
      outputs.summary.meanDownstreamDistance
    );
    fixture.run();

    // The recipe's dense distance view covers exactly the two grid cells.
    expect(outputs.distance.distance.length).toBe(2);
    expect(await fixture.readUint32(reachCount, 1)).toEqual([1]);
    expect(await fixture.readUint32(standaloneReachCount, 1)).toEqual([1]);
    expect(await fixture.readUint32(summaryCount, 1)).toEqual([1]);
    const actualDistance = await fixture.readFloat32(distance, 2);
    const actualReachLength = await fixture.readFloat32(reachLength, 1);
    expect(actualDistance).toEqual(await fixture.readFloat32(standaloneDistance, 2));
    expect(actualReachLength).toEqual(await fixture.readFloat32(standaloneLength, 1));

    // WGS84 east-west scale N(latitude) * cos(latitude) * pi / 180 at the midpoint
    // of the step. Both cell centres are on the equator, where N is the semi-major axis.
    // /usr/bin/python3:
    // from math import pi
    // length = 6378137 * pi / 180
    // print(length, length / 2)
    const [actualMeanDistance] = await fixture.readFloat32(meanDistance, 1);
    expect(actualDistance[1]).toBe(0);
    for (const [actual, expected] of [
      [actualDistance[0], 111319.49079327358],
      [actualReachLength[0], 111319.49079327358],
      [actualMeanDistance, 55659.74539663679]
    ]) {
      expect(Number.isFinite(actual)).toBe(true);
      expect(Math.abs(actual - expected)).toBeLessThan(0.05);
    }
  } finally {
    fixture.destroy();
  }
});
