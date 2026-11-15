// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUPointCloudGround,
  GPUPointCloudSMRF,
  GPUPointCloudSurfaces,
  GPUPointCloudHeight
} from '@luma-contributors/analysis/pointcloud-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import type {CatalogOutputTable} from './catalog-geojson';
import {withCatalogGpuSession} from './catalog-gpu-session';
import {
  makePointCloudCatalogGrid,
  makePointCloudTable,
  validatePointCloudRadii,
  type PointCloudParameters,
  type PreparedPointCloudInputs
} from './catalog-pointcloud-input';
import {makePointCloudDiagnostics} from './catalog-pointcloud-neighborhoods';

export async function runPointCloudRaster(
  context: CatalogGpuContext,
  operation: string,
  prepared: PreparedPointCloudInputs,
  parameters: PointCloudParameters
): Promise<CatalogRunOutput> {
  const cloud = prepared.clouds['points'];
  const count = cloud.collection.features.length;
  const grid = makePointCloudCatalogGrid(cloud, Number(parameters['cellSize']));
  const cells = grid.width * grid.height;
  if (operation === 'ground')
    validatePointCloudRadii(parameters['radii'] as readonly number[], true);
  return withCatalogGpuSession(context.device, `catalog-pointcloud-${operation}`, async session => {
    const positions = session.input('positions', cloud.positions, 'float32x3');
    const classifications = session.input('classifications', cloud.classifications, 'uint32');
    const terrain = session.output('terrain', 'float32', cells);
    const surface = session.output('surface', 'float32', cells);
    const canopy = session.output('canopy', 'float32', cells);
    const coverage = session.output('coverage', 'uint32', cells);
    const mask = session.output('ground-mask', 'uint32', count);
    const outputClassifications = session.output('output-classifications', 'uint32', count);
    const status = session.output('status', 'uint32', count);
    const heights = session.output('heights', 'float32', count);
    let contributors: string[];
    if (operation === 'ground') {
      // Ground consumes a precomputed unclassified minimum surface and nonzero coverage.
      // Mark every point eligible only for this preparation pass; original LAS classes are
      // passed separately to Ground so classification cannot restrict the terrain support.
      session.graph.add(
        new GPUPointCloudSurfaces({
          positions,
          classifications: session.input(
            'minimum-surface-classifications',
            new Uint32Array(count).fill(1),
            'uint32'
          ),
          groundClass: 1,
          grid,
          terrain: terrain.view,
          surface: surface.view,
          canopy: canopy.view,
          coverage: coverage.view
        })
      );
      session.graph.add(
        new GPUPointCloudGround({
          positions,
          classifications,
          grid,
          minimumSurface: terrain.view,
          coverage: coverage.view,
          radii: parameters['radii'] as readonly number[],
          initialThreshold: Number(parameters['initialThreshold']),
          slope: Number(parameters['slope']),
          maximumThreshold: Number(parameters['maximumThreshold']),
          groundMask: mask.view,
          outputClassifications: outputClassifications.view,
          status: status.view
        })
      );
      contributors = ['GPUPointCloudSurfaces', 'GPUPointCloudGround'];
    } else if (operation === 'smrf') {
      session.graph.add(
        new GPUPointCloudSMRF({
          positions,
          classifications,
          grid,
          slope: Number(parameters['slope']),
          scalar: Number(parameters['scalar']),
          threshold: Number(parameters['threshold']),
          window: Number(parameters['windowCells']) * grid.cellSize,
          cut: 0,
          groundSurface: terrain.view,
          groundMask: mask.view,
          outputClassifications: outputClassifications.view,
          status: status.view
        })
      );
      contributors = ['GPUPointCloudSMRF'];
    } else {
      session.graph.add(
        new GPUPointCloudSurfaces({
          positions,
          classifications,
          grid,
          terrain: terrain.view,
          surface: surface.view,
          canopy: canopy.view,
          coverage: coverage.view
        })
      );
      contributors = ['GPUPointCloudSurfaces'];
      if (operation === 'height') {
        session.graph.add(
          new GPUPointCloudHeight({
            positions,
            grid,
            terrain: terrain.view,
            coverage: coverage.view,
            coverageBit: 2,
            ...(parameters['interpolation'] === 'bilinear'
              ? {interpolation: {method: 'bilinear' as const}}
              : {}),
            heights: heights.view,
            status: status.view
          })
        );
        contributors.push('GPUPointCloudHeight');
      }
    }
    session.run();
    const columns: Record<string, (number | null)[]> = {};
    const tables: CatalogOutputTable[] = [];
    if (operation === 'ground' || operation === 'smrf') {
      columns['source_classification'] = Array.from(cloud.classifications);
      columns['ground_mask'] = Array.from(await session.readUint32(mask));
      columns['classification'] = Array.from(await session.readUint32(outputClassifications));
      columns['ground_status'] = Array.from(await session.readUint32(status));
    } else if (operation === 'height') {
      const flags = await session.readUint32(status);
      columns['height_status'] = Array.from(flags);
      columns['height_above_ground'] = Array.from(
        await session.readFloat32(heights),
        (value, row) => (flags[row] === 0 && Number.isFinite(value) ? value : null)
      );
    } else {
      const terrainValues = await session.readFloat32(terrain);
      const surfaceValues = await session.readFloat32(surface);
      const canopyValues = await session.readFloat32(canopy);
      const coverageValues = await session.readUint32(coverage);
      const sample = (row: number) => {
        const horizontal = Math.floor(
          Math.fround(
            Math.fround(cloud.positions[row * 3] - grid.origin[0]) / Math.fround(grid.cellSize)
          )
        );
        const vertical = Math.floor(
          Math.fround(
            Math.fround(cloud.positions[row * 3 + 1] - grid.origin[1]) / Math.fround(grid.cellSize)
          )
        );
        return (grid.height - 1 - vertical) * grid.width + horizontal;
      };
      columns['coverage'] = Array.from({length: count}, (_, row) => coverageValues[sample(row)]);
      columns['terrain'] = Array.from({length: count}, (_, row) =>
        coverageValues[sample(row)] & 2 ? terrainValues[sample(row)] + prepared.origin[2] : null
      );
      columns['surface'] = Array.from({length: count}, (_, row) =>
        coverageValues[sample(row)] & 1 ? surfaceValues[sample(row)] + prepared.origin[2] : null
      );
      columns['canopy'] = Array.from({length: count}, (_, row) =>
        coverageValues[sample(row)] === 3 ? canopyValues[sample(row)] : null
      );
      const coordinates = Array.from({length: cells}, (_, cell) =>
        prepared.frame.inverse(
          grid.origin[0] + ((cell % grid.width) + 0.5) * grid.cellSize + prepared.origin[0],
          grid.origin[1] +
            (grid.height - 1 - Math.floor(cell / grid.width) + 0.5) * grid.cellSize +
            prepared.origin[1]
        )
      );
      const gridColumns = {
        cell_id: Array.from({length: cells}, (_, cell) => cell),
        longitude: coordinates.map(position => position[0]),
        latitude: coordinates.map(position => position[1]),
        altitude: Array.from(terrainValues, (value, cell) =>
          coverageValues[cell] & 2 ? value + prepared.origin[2] : null
        ),
        terrain: Array.from(terrainValues, (value, cell) =>
          coverageValues[cell] & 2 ? value + prepared.origin[2] : null
        ),
        surface: Array.from(surfaceValues, (value, cell) =>
          coverageValues[cell] & 1 ? value + prepared.origin[2] : null
        ),
        canopy: Array.from(canopyValues, (value, cell) =>
          coverageValues[cell] === 3 ? value : null
        ),
        coverage: Array.from(coverageValues)
      };
      tables.push({
        id: 'grid',
        label: 'Point-cloud surfaces',
        rowCount: cells,
        columns: gridColumns,
        columnTypes: Object.fromEntries(
          Object.keys(gridColumns).map(name => [name, 'float64' as const])
        )
      });
    }
    const diagnostics = makePointCloudDiagnostics(columns);
    if ((operation === 'height' || operation === 'surfaces') && !cloud.classifications.includes(2))
      diagnostics.push({
        code: 'no-ground-class',
        severity: 'warning',
        count,
        message:
          'No LAS class 2 support; run ground classification and use its published classification column first.'
      });
    return {
      tables: [makePointCloudTable(cloud, operation, columns), ...tables],
      diagnostics,
      summary: {
        pointCount: count,
        gridWidth: grid.width,
        gridHeight: grid.height,
        gridCellSize: grid.cellSize,
        originX: prepared.origin[0],
        originY: prepared.origin[1],
        originZ: prepared.origin[2]
      },
      dispatchedContributors: contributors
    };
  });
}
