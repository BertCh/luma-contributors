// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {test} from 'vitest';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {createRasterHandoffFixture} from './workflow-raster-handoff-fixtures';

// The lead runs Chromium; mismatches include evidence because console logs are not collected.
test('WebGPU workflow fills planar elevation and hands its retained raster to drainage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {host, registry, document, elevation} = createRasterHandoffFixture(device);
  const runner = new WorkflowRunner({host, registry});
  try {
    runner.setDocument(document);
    const record = await runner.run();
    const steps = runner.getState().steps;
    const preparation = steps['prepare'].result;
    const drainage = steps['drainage'].result;
    const prepared = preparation?.outputs['prepared-surface'];
    const cells = drainage?.outputs['catchment-cells'];
    const report = JSON.stringify({
      record,
      preparation: steps['prepare'],
      drainage: steps['drainage']
    });
    if (
      record.status !== 'done' ||
      preparation?.engine !== 'gpu' ||
      drainage?.engine !== 'gpu' ||
      !preparation.dispatchedContributors.includes('GPURasterFillNoData') ||
      !drainage.dispatchedContributors.length ||
      prepared?.kind !== 'raster' ||
      !prepared.raster ||
      prepared.raster.crs !== 'planar' ||
      prepared.raster.cellSizeMode !== 'uniform' ||
      JSON.stringify(prepared.raster.grid) !== JSON.stringify(elevation.grid) ||
      prepared.raster.band.index !== elevation.band.index ||
      prepared.table.geometryColumn !== undefined ||
      prepared.raster.band.validity[4] === 0 ||
      !Number.isFinite(prepared.raster.band.values[4]) ||
      cells?.table.rowCount !== 9 ||
      cells.table.geometryColumn !== 'geometry' ||
      cells.table.coordinateSystem !== 'planar' ||
      elevation.band.values[4] !== -999 ||
      elevation.band.validity[4] !== 0
    ) {
      throw new Error(`Prepared raster workflow hand-off failed: ${report}`);
    }
  } finally {
    await runner.destroy();
  }
}, 60000);
