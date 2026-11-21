// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {cellsToDirectedEdge, gridDisk, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addCellFlowTopologyRecipe} from '../../../src/gpu-spatial-analysis/recipes/cell-flow-topology-recipe';
import {RecipeTestFixture} from './recipe-harness';

async function runFlow(origins: Uint32Array, destinations: Uint32Array, values: Float32Array) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'cell-flow-chunks');
  try {
    const first = latLngToCell(37.77, -122.42, 9);
    const second = gridDisk(first, 1).find(key => key !== first)!;
    const third = latLngToCell(40.71, -74, 9);
    const cells = Uint32Array.from(
      [first, second, third].flatMap(key => {
        const bits = BigInt(`0x${key}`);
        return [Number(bits & 0xffffffffn), Number(bits >> 32n)];
      })
    );
    const outgoing = fixture.output('outgoing', 'float32', 3);
    const incoming = fixture.output('incoming', 'float32', 3);
    const flows = fixture.output('flows', 'float32', 18);
    const invalid = fixture.output('invalid', 'uint32', 1);
    addCellFlowTopologyRecipe(fixture.graph, {
      cells: fixture.input('cells', cells, 'uint32x2', 3),
      origins: fixture.input('origins', origins, 'uint32', values.length),
      destinations: fixture.input('destinations', destinations, 'uint32', values.length),
      values: fixture.input('values', values, 'float32', values.length),
      output: {
        outgoingTotals: outgoing.view,
        incomingTotals: incoming.view,
        edgeFlows: flows.view,
        status: {invalidCount: invalid.view}
      }
    });
    fixture.run();
    // H3 directed-edge reserved bits encode direction 1..6; recipe slots are direction-1.
    // Derive from h3-js independently of the recipe's neighbor output.
    const edgeSlot = Number((BigInt(`0x${cellsToDirectedEdge(first, second)}`) >> 56n) & 7n) - 1;
    expect(edgeSlot).toBeGreaterThanOrEqual(0);
    expect(edgeSlot).toBeLessThan(6);
    return {
      outgoing: await fixture.readFloat32(outgoing, 3),
      incoming: await fixture.readFloat32(incoming, 3),
      flows: await fixture.readFloat32(flows, 18),
      edgeSlot,
      invalid: await fixture.readUint32(invalid, 1)
    };
  } finally {
    fixture.destroy();
  }
}

it.each([
  4096, 4097
])('RCP-105 retains all %s pairs, including self and nonneighbor pairs', async count => {
  const origins = new Uint32Array(count);
  const destinations = new Uint32Array(count).fill(1);
  const values = new Float32Array(count).fill(1);
  origins[0] = 1;
  values[0] = 4; // self
  destinations[1] = 2;
  values[1] = 8; // nonneighbor
  origins[2] = 3; // invalid cell row
  const result = await runFlow(origins, destinations, values);
  // count-3 ordinary neighbor rows, one self flow 4 and one nonneighbor flow 8.
  expect(result.outgoing).toEqual([count + 5, 4, 0]);
  expect(result.incoming).toEqual([0, count + 1, 8]);
  const expectedFlows = new Array<number>(18).fill(0);
  expectedFlows[result.edgeSlot] = count - 3;
  expect(result.flows).toEqual(expectedFlows);
  expect(result.invalid).toEqual([1]);
});

it.each([
  1, 2
])('RCP-105 preserves sequential rounding across a pass boundary (destination %s)', async destination => {
  const values = new Float32Array(4098);
  values[0] = 16777216;
  values[4096] = 1;
  values[4097] = 1;
  const destinations = new Uint32Array(4098).fill(1);
  destinations[4096] = destination;
  destinations[4097] = destination;
  const result = await runFlow(new Uint32Array(4098), destinations, values);
  // f32 spacing at 2^24 is two: each separate +1 ties to even, leaving 2^24.
  // Summing the second pass first would give 2, then incorrectly 16777218.
  expect(result.outgoing).toEqual([16777216, 0, 0]);
  expect(result.incoming).toEqual(destination === 1 ? [0, 16777216, 0] : [0, 16777216, 2]);
  const expectedFlows = new Array<number>(18).fill(0);
  expectedFlows[result.edgeSlot] = 16777216;
  expect(result.flows).toEqual(expectedFlows);
  expect(result.invalid).toEqual([0]);
});
