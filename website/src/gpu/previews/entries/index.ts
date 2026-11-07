// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registry of function-specific previews, one file per catalog group (src/catalog/entries).
// Entries without one fall back to the generic scene of their preview kind (builders.ts).

import type {EntryPreview} from '../helpers.ts';
import {CELLS_INDEX_JOIN_PREVIEWS} from './cells-index-join.ts';
import {DATAFRAME_POINTCLOUD_RECORDS_PREVIEWS} from './dataframe-pointcloud-records.ts';
import {DENSITY_INTERPOLATION_REGIONS_PREVIEWS} from './density-interpolation-regions.ts';
import {GEOMETRY_CONSTRUCTION_OVERLAY_PREVIEWS} from './geometry-construction-overlay.ts';
import {GEOMETRY_PROCESSING_PREVIEWS} from './geometry-processing.ts';
import {NETWORK_ROUTING_ACCESS_PREVIEWS} from './network-routing-access.ts';
import {RASTER_BANDS_INDICES_PREVIEWS} from './raster-bands-indices.ts';
import {RECIPES_TRAJECTORIES_PREVIEWS} from './recipes-trajectories.ts';
import {REGRESSION_WEIGHTS_PREVIEWS} from './regression-weights.ts';
import {SPATIAL_STATISTICS_PREVIEWS} from './spatial-statistics.ts';
import {TERRAIN_HYDROLOGY_FLOW_PREVIEWS} from './terrain-hydrology-flow.ts';
import {TERRAIN_RELIEF_LIGHT_PREVIEWS} from './terrain-relief-light.ts';

const ENTRY_PREVIEWS: ReadonlyMap<string, EntryPreview> = new Map(
  Object.entries({
    ...CELLS_INDEX_JOIN_PREVIEWS,
    ...DATAFRAME_POINTCLOUD_RECORDS_PREVIEWS,
    ...DENSITY_INTERPOLATION_REGIONS_PREVIEWS,
    ...GEOMETRY_CONSTRUCTION_OVERLAY_PREVIEWS,
    ...GEOMETRY_PROCESSING_PREVIEWS,
    ...NETWORK_ROUTING_ACCESS_PREVIEWS,
    ...RASTER_BANDS_INDICES_PREVIEWS,
    ...RECIPES_TRAJECTORIES_PREVIEWS,
    ...REGRESSION_WEIGHTS_PREVIEWS,
    ...SPATIAL_STATISTICS_PREVIEWS,
    ...TERRAIN_HYDROLOGY_FLOW_PREVIEWS,
    ...TERRAIN_RELIEF_LIGHT_PREVIEWS
  })
);

/** The function-specific preview of a catalog entry, if it has one. */
export function getEntryPreview(entryId: string): EntryPreview | undefined {
  return ENTRY_PREVIEWS.get(entryId);
}
