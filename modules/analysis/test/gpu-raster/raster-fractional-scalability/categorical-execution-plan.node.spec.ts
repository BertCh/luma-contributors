// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createScalabilityProps} from './raster-fractional-scalability-fixture';

it('SCL-09/16 plans the default tiled route without declaring graph resources', () => {
  const device = createNullWebGPUDevice();
  for (const [zones, tuples] of [
    [30000, 100],
    [2, 1048577]
  ]) {
    const graph = new GPUCommandGraph(device);
    const {categorical} = createScalabilityProps(graph, zones, tuples);
    const resources = vi.spyOn(graph, 'createTransientBuffer');
    const plan = GPURasterFractionalCategoricalStatistics.getExecutionPlan(
      categorical,
      device.limits
    );
    expect(plan.feasible).toBe(true);
    expect(plan.route).toBe('tiled');
    expect(
      new GPURasterFractionalCategoricalStatistics(categorical).estimatedCommandNodeCount
    ).toBe(plan.estimatedCommandNodeCount);
    expect(plan.estimatedCommandNodeCount).toBe(tuples === 100 ? 264 : 520);
    expect(resources).not.toHaveBeenCalled();
    expect(
      GPURasterFractionalCategoricalStatistics.getExecutionPlan(
        {...categorical, mode: 'exact'},
        device.limits
      ).feasible
    ).toBe(false);
    resources.mockRestore();
  }
});

it('SCL-09/16 plans output bindings and the exact 127-tile command ceiling', () => {
  const device = createNullWebGPUDevice();
  for (const tiles of [127, 128]) {
    const graph = new GPUCommandGraph(device);
    const {categorical} = createScalabilityProps(graph, 2, tiles);
    const plan = GPURasterFractionalCategoricalStatistics.getExecutionPlan(
      {...categorical, tupleTileSize: 1},
      device.limits
    );
    expect(plan.feasible).toBe(tiles === 127);
    expect(plan.estimatedCommandNodeCount).toBe(8 + 256 * tiles);
  }
});

it('S1 identifies an oversized caller backing allocation when every scratch binding fits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {categorical} = createScalabilityProps(graph, 2, 2);
  const buffer = graph.createTransientBuffer({
    id: 'oversized-class-areas',
    byteLength: 1024,
    usage: Buffer.STORAGE
  });
  const classAreas = graph.createDataView(buffer, {format: 'float32', length: 4});
  const plan = GPURasterFractionalCategoricalStatistics.getExecutionPlan(
    {...categorical, output: {...categorical.output, classAreas}},
    {...device.limits, maxStorageBufferBindingSize: 256, maxBufferSize: 512}
  );
  const failed = plan.constraints.filter(constraint => !constraint.satisfied);
  expect(plan.feasible).toBe(false);
  expect(failed).toHaveLength(1);
  expect(failed[0].quantity).toBe('oversized-class-areas buffer bytes');
  expect(failed[0].received).toBe(1024);
  expect(failed[0].maximum).toBe(512);
});
