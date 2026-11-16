// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  makeCatalogRasterInput,
  makeWorkflowRasterData,
  profileKeplerWorkbenchData,
  type CatalogRasterInput,
  type WorkflowData
} from '@luma-contributors/kepler';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {NumericRasterSource} from '../raster-tile-ingest';

export type WorkbenchRasterInputOptions = {
  /** One-based decoded band index. */
  bandIndex: number;
  /** Required for a multi-tile source; no implicit mosaic, crop or resampling. */
  tileId?: string;
  maximumCells: number;
};

/** Reuses the host's decoded numeric samples and validity without copying or decoding. */
export function readWorkbenchRasterInput(
  source: NumericRasterSource,
  options: WorkbenchRasterInputOptions
): CatalogRasterInput {
  const id = `workbench-raster-input: ${source.id}`;
  validateRequirement(
    id,
    options.tileId !== undefined || source.tiles.length === 1,
    'tileId must select one bounded tile when source.tiles does not contain exactly one tile'
  );
  const matches =
    options.tileId === undefined
      ? source.tiles
      : source.tiles.filter(tile => tile.id === options.tileId);
  validateRequirement(id, matches.length === 1, 'tileId must name exactly one decoded tile');
  const tile = matches[0];
  const bands = tile.bands.filter(band => band.index === options.bandIndex);
  validateRequirement(id, bands.length === 1, 'bandIndex must name exactly one decoded band');
  const [west, south, east, north] = tile.extent;
  validateRequirement(
    id,
    tile.extent.every(Number.isFinite) && west < east && south < north,
    'tile.extent must be a finite north-up box with positive area'
  );
  return makeCatalogRasterInput(
    {
      id: `${source.id}/${tile.id}/b${options.bandIndex}`,
      grid: makeGPURasterMetadata({
        width: tile.width,
        height: tile.height,
        cellSize: [(east - west) / tile.width, (north - south) / tile.height],
        origin: [west, north],
        rowDirection: 'south'
      }),
      band: bands[0],
      crs: source.crs,
      cellSizeMode: source.crs === 'EPSG:4326' ? 'geographic' : 'web-mercator'
    },
    options.maximumCells
  );
}

/** Host readDataset adapters can hand this directly to the existing workflow executor. */
export function makeWorkbenchRasterWorkflowData(
  source: NumericRasterSource,
  options: WorkbenchRasterInputOptions
): WorkflowData {
  return makeWorkflowRasterData(readWorkbenchRasterInput(source, options), options.maximumCells);
}

/** Each discovery entry selects one tile and band explicitly, including for multi-tile sources. */
export type WorkbenchRasterSource = {source: NumericRasterSource; label: string};

export function createWorkbenchRasterDatasetAdapter(
  getSources: () => readonly WorkbenchRasterSource[]
) {
  const revisions = new WeakMap<NumericRasterSource, number>();
  const entries = new WeakMap<NumericRasterSource, ReturnType<typeof makeEntries>>();
  let revision = 0;
  function makeEntries({source, label}: WorkbenchRasterSource) {
    if (!revisions.has(source)) revisions.set(source, ++revision);
    return source.tiles.flatMap(tile =>
      tile.bands.map(band => {
        const id = `workbench-raster:${JSON.stringify([source.id, tile.id, band.index])}`;
        const data = makeWorkbenchRasterWorkflowData(source, {
          tileId: tile.id,
          bandIndex: band.index,
          maximumCells: tile.width * tile.height
        });
        data.table = {
          ...data.table,
          id,
          label: `${label} · tile ${tile.id} · band ${band.index}${band.name ? ` (${band.name})` : ''}`
        };
        data.raster = {...data.raster!, id};
        data.analysisProfile = {
          ...profileKeplerWorkbenchData(data),
          revision: revisions.get(source)
        };
        return {
          data,
          profile: {
            id,
            label: data.table.label,
            rowCount: tile.width * tile.height,
            fields: [],
            geometryFields: [],
            resident: false
          },
          revision: String(revisions.get(source))
        };
      })
    );
  }
  return {
    list() {
      return getSources().flatMap(source => {
        let cached = entries.get(source.source);
        // Labels are user-editable metadata and must refresh alongside their profiles.
        if (!cached || !cached[0]?.profile.label.startsWith(`${source.label} · tile `)) {
          cached = makeEntries(source);
          entries.set(source.source, cached);
        }
        return cached;
      });
    }
  };
}
