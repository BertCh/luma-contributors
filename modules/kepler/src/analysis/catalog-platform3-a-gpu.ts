// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUHilbertKeys} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import type {CatalogFeatureCollection} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {getHilbertItems, makeHilbertOutput} from './catalog-platform3-a-measure';

/**
 * GPU Hilbert order: GPUHilbertKeys quantizes the float32 box centers of every feature against the
 * collection bounds and sorts the rows by key on the GPU; identity and output assembly mirror the
 * CPU reference.
 */
export async function runGpuHilbertOrder(
  context: CatalogGpuContext,
  features: CatalogFeatureCollection,
  order: number
): Promise<CatalogRunOutput> {
  const items = getHilbertItems(features);
  // GPUHilbertKeys needs at least one finite feature box to derive the bounds.
  if (!items.validCount) throw new CatalogGpuUnsupportedError('No feature has finite geometry');
  const count = features.features.length;
  const {keys, rows} = await withCatalogGpuSession(
    context.device,
    'catalog-hilbert-order',
    async session => {
      const keysOutput = session.output('keys', 'uint32', count);
      const rowsOutput = session.output('sorted-rows', 'uint32', count);
      session.graph.add(
        new GPUHilbertKeys({
          order,
          minima: session.input('minima', items.minima, 'float32x2'),
          maxima: session.input('maxima', items.maxima, 'float32x2'),
          bounds: session.input('bounds', items.bounds, 'float32'),
          output: {keys: keysOutput.view, sortedRows: rowsOutput.view}
        })
      );
      session.run();
      return {
        keys: await session.readUint32(keysOutput),
        rows: await session.readUint32(rowsOutput)
      };
    }
  );
  return {
    ...makeHilbertOutput(features, keys, Array.from(rows), order),
    dispatchedContributors: ['GPUHilbertKeys']
  };
}
