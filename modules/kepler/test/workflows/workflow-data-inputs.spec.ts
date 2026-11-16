// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterReclassify,
  getGPURasterReclassifyParameterValues,
  makeGPURasterBand
} from '@luma-contributors/analysis/raster-analysis';
import {test} from 'vitest';
import type {CatalogOperationBinding} from '../../src/analysis/catalog-binding-types';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {withCatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../../examples/deck/kepler-compute/src/raster-tile-ingest';
import {readWorkbenchRasterInput} from '../../../../examples/deck/kepler-compute/src/workflow/workbench-raster-input';
import {makeArrowTableFromCatalogTable} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-table';

// Boundary qualification, not recipe qualification: both fixtures use decoded bands from the
// existing ingestion path, real GPURasterReclassify dispatch and the existing Arrow table adapter.
// It does not establish native raster-layer publication or a file-upload browser journey.
test('ingested numeric rasters reach a real GPU contributor with validity and cell identity intact', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of [
    {width: 2, height: 2, values: [1, 4, -999, 2], expected: [0, 1, null, 0]},
    {width: 3, height: 2, values: [8, 0, 3, -999, 7, 1], expected: [2, 0, 1, null, 2, 0]}
  ]) {
    const tile = createNumericRasterTile({
      id: 'tile',
      width: fixture.width,
      height: fixture.height,
      extent: [0, 0, 30, 20],
      bands: [{values: new Float32Array(fixture.values), noDataValue: -999}]
    });
    const source = createNumericRasterSource({
      id: 'loaded',
      kind: 'single-raster',
      crs: 'EPSG:3857',
      tiles: [tile]
    });
    const raster = readWorkbenchRasterInput(source, {bandIndex: 1, maximumCells: 6});
    const binding: CatalogOperationBinding = {
      id: 'typed-raster-gpu-boundary',
      title: 'Raster GPU boundary',
      domain: 'raster',
      summary: 'Qualification fixture',
      capabilityIds: [],
      contributorNames: ['GPURasterReclassify'],
      gpuContributorNames: ['GPURasterReclassify'],
      engine: 'gpu',
      inputs: [
        {
          name: 'raster',
          label: 'Raster',
          dataKind: 'raster',
          accepts: [],
          maximumFeatures: 0,
          maximumVertices: 0,
          maximumCells: 6
        }
      ],
      parameters: [],
      outputs: [
        {
          tableId: 'classes',
          label: 'Classes',
          geometry: 'none',
          identityColumns: ['cell_id'],
          maximumRows: 6
        }
      ],
      cleanup: 'Session releases graph and storage in finally',
      declaredBehavior: ['Mask and native nodata survive ingestion'],
      limitation: 'Boundary qualification only',
      runDataGpu: async (context, inputs) => {
        const input = inputs['raster'];
        if (input.type !== 'raster')
          throw new Error('Raster GPU boundary: raster band/grid required');
        return withCatalogGpuSession(context.device, 'typed-raster-boundary', async session => {
          const count = input.grid.width * input.grid.height;
          const values = session.input('values', Float32Array.from(input.band.values), 'float32');
          const validity = session.input(
            'validity',
            Uint32Array.from(input.band.validity),
            'uint32'
          );
          const band = makeGPURasterBand(values, input.grid, {
            validity,
            ...(input.band.noDataValue === null ? {} : {noDataValue: input.band.noDataValue})
          });
          const classes = session.output('classes', 'uint32', count);
          session.graph.add(
            new GPURasterReclassify({
              values: band,
              grid: input.grid,
              breaks: session.input('breaks', new Float32Array([3, 7]), 'float32'),
              parameters: session.input(
                'parameters',
                getGPURasterReclassifyParameterValues({breakCount: 2}),
                'float32'
              ),
              output: {classes: classes.view}
            })
          );
          session.run();
          const result = await session.readUint32(classes);
          return {
            tables: [
              {
                id: 'classes',
                label: 'Classes',
                rowCount: count,
                columns: {
                  cell_id: Array.from({length: count}, (_, index) => index),
                  class_id: Array.from(result, value => (value === 0xffffffff ? null : value))
                },
                columnTypes: {cell_id: 'int32', class_id: 'int32'}
              }
            ],
            diagnostics: [],
            summary: {cellCount: count},
            dispatchedContributors: ['GPURasterReclassify']
          };
        });
      }
    };
    const result = await runCatalogOperationAsync(binding, {}, {raster}, {device});
    const table = result.tables[0];
    const publishedTable = makeArrowTableFromCatalogTable(table);
    const actual = Array.from(publishedTable.getChild('class_id')!);
    if (
      result.engine !== 'gpu' ||
      JSON.stringify(actual) !== JSON.stringify(fixture.expected) ||
      JSON.stringify(result.dispatchedContributors) !== JSON.stringify(['GPURasterReclassify']) ||
      JSON.stringify(table.columns['cell_id']) !==
        JSON.stringify(fixture.values.map((_, index) => index))
    ) {
      throw new Error(
        `Typed raster GPU boundary mismatch: ${JSON.stringify({shape: [fixture.width, fixture.height], engine: result.engine, contributors: result.dispatchedContributors, expected: fixture.expected, actual, cellIds: table.columns['cell_id'], affine: raster.grid.affine})}`
      );
    }
  }
});
