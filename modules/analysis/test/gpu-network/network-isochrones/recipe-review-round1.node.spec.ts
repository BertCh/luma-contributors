// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {Buffer} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createIsochroneOracleHarness} from './isochrones-output-oracle-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createRecipeCellReviewProps} from './recipe-cell-review-fixture';

const foreign = createRequire(import.meta.url)(
  '@luma.gl/graph/gpu-core'
) as typeof import('@luma.gl/graph/gpu-core');

it.each(['raster', 'cells'] as const)('declares a recipe with foreign %s outputs (B1)', route => {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  try {
    const props = createRecipeCellReviewProps(harness);
    if (route === 'raster') {
      const classes = harness.output('classes', 'uint32', 4);
      props.isochrones!.cellOutline = undefined;
      props.isochrones!.raster = {
        width: 2,
        height: 2,
        output: {
          bandClasses: new foreign.GraphDataView(classes.buffer, classes)
        }
      };
    } else {
      const output = props.isochrones!.cellOutline!.output;
      output.rows = new foreign.GraphDataView(output.rows.buffer, output.rows);
    }
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).not.toThrow();
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('exposes each supplied cell-loss and ring-completeness stage view (B2)', () => {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  try {
    const props = createRecipeCellReviewProps(harness);
    const result = addDriveTimeCatchmentRecipe(harness.graph, props);
    const cell = props.isochrones!.cellOutline!;
    for (const [name, status] of [
      ['isochrone-cells', cell.table!.status],
      ['isochrone-outline', cell.output.status],
      ['isochrone-rings', cell.rings!.output.status]
    ] as const) {
      const stage = result.status.stages.find(entry => entry.stage === name)!;
      expect(stage.status.candidateOverflow === status.candidateOverflow, name).toBe(true);
    }
    const ringStage = result.status.stages.find(entry => entry.stage === 'isochrone-rings')!;
    expect(ringStage.status.complete === cell.rings!.output.status.complete).toBe(true);
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('keeps the demand join loss evidence independent of its own overflow (STA-08)', () => {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  try {
    const props = createRecipeCellReviewProps(harness);
    props.isochrones!.joinDemand = {candidateCapacity: 8};
    const uncertainCount = harness.output('join-uncertain', 'uint32', 1);
    const candidateOverflow = harness.output('join-source-loss', 'uint32', 1);
    props.output = {
      isochroneDemand: {
        status: {
          overflow: harness.output('join-overflow', 'uint32', 1),
          uncertainCount,
          candidateOverflow
        }
      }
    };
    const recipe = addDriveTimeCatchmentRecipe(harness.graph, props);
    const stage = recipe.status.stages.find(entry => entry.stage === 'isochrone-demand')!;
    expect(stage.status.candidateOverflow === candidateOverflow).toBe(true);
    expect(stage.status.uncertainCount === uncertainCount).toBe(true);
    expect(stage.status.candidateOverflow === recipe.isochroneDemand!.status.overflow).toBe(false);
    expect(
      stage.status.candidateOverflow === recipe.isochroneDemand!.status.candidateOverflow
    ).toBe(true);
    const rings = recipe.status.stages.find(entry => entry.stage === 'isochrone-rings')!;
    const memory = createCPUGraphMemory();
    const nodes = (harness.graph as unknown as {nodes: {id: string}[]}).nodes.filter(
      node =>
        node.id === 'review-cell-recipe-isochrone-join-source-loss' ||
        node.id === 'review-cell-recipe-isochrone-join-suppress-incomplete'
    );
    expect(nodes.length).toBe(2);
    for (const [candidateLoss, complete, ownOverflow] of [
      [1, 0, 0],
      [0, 0, 0],
      [0, 1, 1],
      [0, 1, 0]
    ] as const) {
      const incomplete = candidateLoss !== 0 || complete === 0;
      memory.write(rings.status.candidateOverflow!, Uint32Array.of(candidateLoss));
      memory.write(rings.status.complete!, Uint32Array.of(complete));
      // Demand lies strictly within the complete first tile: the join would return feature 0.
      memory.write(recipe.isochroneDemand!.pointFeatureIds, Uint32Array.of(0));
      memory.write(stage.status.overflow!, Uint32Array.of(ownOverflow));
      memory.write(uncertainCount, Uint32Array.of(13));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(stage.status.candidateOverflow!))).toEqual([
        incomplete ? 1 : 0
      ]);
      expect(Array.from(memory.readUint32(stage.status.overflow!))).toEqual([ownOverflow]);
      expect(Array.from(memory.readUint32(uncertainCount))).toEqual([13]);
      expect(Array.from(memory.readUint32(recipe.isochroneDemand!.pointFeatureIds))).toEqual([
        incomplete ? 0xffffffff : 0
      ]);
    }
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('forwards explicit planar context to the cell contributor and preserves omitted compatibility (context parity)', () => {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  try {
    const props = createRecipeCellReviewProps(harness);
    props.spatialContext = {coordinateSpace: 'planar', metric: 'native'};
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).toThrow(
      /cellOutline needs longitude-latitude positions/
    );
    props.spatialContext = undefined;
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).not.toThrow();
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it.each([
  'missing-join',
  'wrong-format',
  'wrong-length'
] as const)('validates the new inherited-loss destination (%s, B3)', invalid => {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  try {
    const props = createRecipeCellReviewProps(harness);
    if (invalid !== 'missing-join') props.isochrones!.joinDemand = {candidateCapacity: 8};
    const candidateOverflow =
      invalid === 'wrong-format'
        ? (harness.output('invalid-loss', 'float32', 1) as unknown as GraphDataView<'uint32'>)
        : harness.output('invalid-loss', 'uint32', invalid === 'wrong-length' ? 2 : 1);
    props.output = {
      isochroneDemand: {
        status: {
          overflow: harness.output('join-own-overflow', 'uint32', 1),
          candidateOverflow
        }
      }
    };
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).toThrow(
      /review-cell-recipe.*candidateOverflow/
    );
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('TSC-04 identifies the inherited-loss packed destination without weakening its prerequisite', () => {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  try {
    const props = createRecipeCellReviewProps(harness);
    const buffer = harness.graph.createTransientBuffer({
      id: 'loss-strided',
      byteLength: 8,
      usage: Buffer.STORAGE
    });
    const candidateOverflow = harness.graph.createDataView(buffer, {
      format: 'uint32',
      length: 1,
      byteStride: 8
    });
    props.output = {
      isochroneDemand: {
        status: {overflow: harness.output('own-overflow', 'uint32', 1), candidateOverflow}
      }
    };
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).toThrow(
      /candidateOverflow requires isochrones.joinDemand/
    );
    props.isochrones!.joinDemand = {candidateCapacity: 8};
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).toThrowError(
      expect.objectContaining({
        id: 'review-cell-recipe',
        contributor: 'GPUDriveTimeCatchmentRecipe',
        code: 'VIEW_FORMAT',
        path: 'output.isochroneDemand.status.candidateOverflow'
      })
    );
    props.output.isochroneDemand!.status!.candidateOverflow = harness.output(
      'loss-packed',
      'uint32',
      1
    );
    expect(() => addDriveTimeCatchmentRecipe(harness.graph, props)).not.toThrow();
  } finally {
    harness.destroy();
    device.destroy();
  }
});
