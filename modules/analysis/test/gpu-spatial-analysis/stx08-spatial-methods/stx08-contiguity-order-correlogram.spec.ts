// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  addContiguityOrderCorrelogramRecipe,
  GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS
} from '../../../src/gpu-spatial-analysis/recipes/index';
import fixture from '../../oracles/fixtures/stx08-spatial-methods.json';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

const columns = GPU_CONTIGUITY_ORDER_CORRELOGRAM_COLUMNS;

async function runCorrelogram(
  csr: {offsets: number[]; neighbors: number[]},
  values: number[],
  maximumOrder: number,
  slotCapacity?: number
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const correlogram = rig.output('float32', maximumOrder * columns.length);
  const overflow = rig.output('uint32', maximumOrder);
  addContiguityOrderCorrelogramRecipe(rig.graph, {
    weights: rig.weights({...csr, weights: csr.neighbors.map(() => 1)}),
    values: rig.input(new Float32Array(values), 'float32'),
    maximumOrder,
    slotCapacity,
    outputs: {correlogram, incompleteOrders: overflow}
  });
  const compiled = rig.graph.compile();
  const {submitGraph} = await import('../../utils/gpu-contributor-test-utils');
  submitGraph(device, compiled, undefined);
  const result = {table: await rig.readFloat(correlogram), overflow: await rig.readUint(overflow)};
  compiled.destroy();
  rig.destroy();
  return result;
}

describe('addContiguityOrderCorrelogramRecipe', () => {
  for (const reference of fixture.correlogram) {
    it(`matches the binary64 ring and Moran reference on ${reference.name}`, async () => {
      const {table, overflow} = await runCorrelogram(
        reference,
        reference.values,
        reference.maximumOrder
      );
      expect(overflow).toEqual(new Array(reference.maximumOrder).fill(0));
      reference.orders.forEach((expected, order) => {
        const row = table.slice(order * columns.length, (order + 1) * columns.length);
        expect(row[columns.count], `order ${order + 1} count`).toBe(expected.count);
        expect(row[columns.islandCount]).toBe(expected.islandCount);
        expect(
          Math.abs(row[columns.statistic] - expected.statistic),
          `I order ${order + 1}`
        ).toBeLessThan(1e-4);
        expect(Math.abs(row[columns.expected] - expected.expected)).toBeLessThan(1e-6);
        expect(row[columns.varianceNormality] / expected.varianceNormality).toBeCloseTo(1, 3);
        expect(row[columns.varianceRandomization] / expected.varianceRandomization).toBeCloseTo(
          1,
          3
        );
      });
    });
  }

  it('reports slot capacity overflow per order instead of a wrong ring', async () => {
    const reference = fixture.correlogram[0];
    const {overflow} = await runCorrelogram(
      reference,
      reference.values,
      3,
      reference.neighbors.length
    );
    // Order 1 fits exactly; the rings of orders 2 and 3 are larger on a lattice.
    expect(overflow[0]).toBe(0);
    expect(overflow[1]).toBe(1);
    expect(overflow[2]).toBe(1);
  });
});
