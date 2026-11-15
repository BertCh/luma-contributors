// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COREGIONALIZATION_FIT_REJECTIONS} from './rejections-coregionalization-fit';
import type {KernelAuditRejection} from './kernel-audit-rejection-type';
import {CELLS_AND_JOINS_REJECTIONS} from './rejections-cells-and-joins';
import {DATAFRAME_REJECTIONS} from './rejections-dataframe';
import {GEOMETRY_REJECTIONS} from './rejections-geometry';
import {GEOMETRY_LINES_REJECTIONS} from './rejections-geometry-lines';
import {NETWORK_REJECTIONS} from './rejections-network';
import {POINTCLOUD_REJECTIONS} from './rejections-pointcloud';
import {POINTS_AND_REGIONS_REJECTIONS} from './rejections-points-and-regions';
import {RASTER_REJECTIONS} from './rejections-raster';
import {RTS_TIME_SERIES_REJECTIONS} from './rejections-rts-time-series';
import {RTS04_RTS07_MONITORING_EOF_REJECTIONS} from './rejections-rts04-rts07-monitoring-eof';
import {RTS05_COMPOSITING_REJECTIONS} from './rejections-rts05-compositing';
import {RTS_01_03_EMBEDDING_FDR_REJECTIONS} from './rejections-rts-01-03-embedding-fdr';
import {STATISTICS_REJECTIONS} from './rejections-statistics';
import {TERRAIN_REJECTIONS} from './rejections-terrain';
import {MOVEMENT_SEMANTICS_REJECTIONS} from './rejections-movement-semantics';
import {TIME_GLUE_REJECTIONS} from './rejections-time-glue';
import {STX06_CHANGE_POINTS_STX07_SCAN_REJECTIONS} from './rejections-stx06-change-points-stx07-scan';
import {STM08_STM09_HAWKES_AND_SINKHORN_REJECTIONS} from './rejections-stm08-stm09-hawkes-and-sinkhorn';
import {STX04_STX05_SPACE_TIME_DENSITY_AND_K_REJECTIONS} from './rejections-stx04-stx05-space-time-density-and-k';
import {VALUE_CUBE_REJECTIONS} from './rejections-value-cube';
import {URBAN_ECOLOGY_REJECTIONS} from './rejections-urban-ecology';

export type {KernelAuditRejection} from './kernel-audit-rejection-type';

/**
 * Cases one step above a contributor's documented maximum: rule 16 asks that they fail at
 * declaration with a short `${id}`-prefixed message naming the prop and the limit, before anything
 * compiles. Each has a passing companion case at exactly the documented maximum. The maxima are
 * sized to the core limits (128 MiB storage bindings, `capacity x lanes <= 2^32 - 1`).
 */
export const KERNEL_AUDIT_REJECTIONS: readonly KernelAuditRejection[] = [
  ...DATAFRAME_REJECTIONS,
  ...COREGIONALIZATION_FIT_REJECTIONS,
  ...STX06_CHANGE_POINTS_STX07_SCAN_REJECTIONS,
  ...STX04_STX05_SPACE_TIME_DENSITY_AND_K_REJECTIONS,
  ...STM08_STM09_HAWKES_AND_SINKHORN_REJECTIONS,
  ...TIME_GLUE_REJECTIONS,
  ...VALUE_CUBE_REJECTIONS,
  ...MOVEMENT_SEMANTICS_REJECTIONS,
  ...RASTER_REJECTIONS,
  ...RTS_TIME_SERIES_REJECTIONS,
  ...RTS04_RTS07_MONITORING_EOF_REJECTIONS,
  ...RTS05_COMPOSITING_REJECTIONS,
  ...RTS_01_03_EMBEDDING_FDR_REJECTIONS,
  ...TERRAIN_REJECTIONS,
  ...NETWORK_REJECTIONS,
  ...POINTCLOUD_REJECTIONS,
  ...STATISTICS_REJECTIONS,
  ...GEOMETRY_REJECTIONS,
  ...GEOMETRY_LINES_REJECTIONS,
  ...CELLS_AND_JOINS_REJECTIONS,
  ...POINTS_AND_REGIONS_REJECTIONS,
  ...URBAN_ECOLOGY_REJECTIONS
];

/**
 * Messages of the generic declaration guards (`createWGSLKernelNode`, `GPUCommandGraph` buffers,
 * dispatch layout). They are backstops: a documented maximum that only they refuse is a finding,
 * because their message names neither the contributor ID nor the prop.
 */
export const GENERIC_REFUSAL_PATTERNS: readonly RegExp[] = [
  /: kernel \S+ binding \w+ needs \d+ bytes/,
  /GPUCommandGraph buffer "[^"]+" exceeds/,
  /element count must be a non-negative uint32/,
  /exceeds (one bounded|bounded one-dimensional|device dispatch)/
];
