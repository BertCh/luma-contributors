// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {QUANTILE_LISA_AUDIT_CASES} from './cases-quantile-lisa';
import {EXACT_NATURAL_BREAKS_AUDIT_CASES} from './cases-exact-natural-breaks';
import {DORLING_CARTOGRAM_AUDIT_CASES} from './cases-dorling-cartogram';
import {VARIABLE_STANDARDIZATION_AUDIT_CASES} from './cases-variable-standardization';
import {RATE_MAPPING_KERNEL_AUDIT_CASES} from './cases-rate-mapping';
import {COVERAGE_GLOBAL_CASES} from './cases-coverage-global';
import {COVERAGE_NETWORK_MOVEMENT_CASES} from './cases-coverage-network-movement';
import {COVERAGE_INTERPOLATION_KERNEL_CASES} from './cases-coverage-interpolation';
import {COVERAGE_GEOMETRY_CASES} from './cases-coverage-geometry';
import {COVERAGE_TERRAIN_CASES} from './cases-coverage-terrain';
import {COVERAGE_POINT_PROCESS_CASES} from './cases-coverage-point-process';
import {CONTINUOUS_SAMPLING_CASES} from './cases-continuous-sampling';
import {SPATIAL_VALIDATION_ADVANCED_KERNEL_CASES} from './cases-spatial-validation-advanced';
import {COVERAGE_STATISTICS_CASES} from './cases-coverage-statistics';
import {COVERAGE_URBAN_ECOLOGY_CASES} from './cases-coverage-urban-ecology';
import {COVERAGE_RASTER_KERNEL_CASES} from './cases-coverage-raster';
import {BACKLOG_CASES} from './cases-backlog';
import {COVERAGE_TRANCHE_ONE_CASES} from './cases-coverage-tranche-one';
import {EXPANSION_CASES} from './cases-expansion';
import type {KernelAuditCase} from './kernel-audit-context';
import {GEOSPATIAL_CASES} from './cases-geospatial';
import {DATAFRAME_CASES} from './cases-dataframe';
import {RASTER_CASES} from './cases-raster';
import {RASTER_OPERATIONS_CASES} from './cases-raster-operations';
import {TERRAIN_CASES} from './cases-terrain';
import {NETWORK_CASES} from './cases-network';
import {STATISTICS_CASES} from './cases-statistics';
import {RECIPES_CASES} from './cases-recipes';
import {GEOMETRY_CASES} from './cases-geometry';
import {CELLS_AND_JOINS_CASES} from './cases-cells-and-joins';
import {POINTS_AND_REGIONS_CASES} from './cases-points-and-regions';

/** Every kernel-audit case, by area file. */
export const KERNEL_AUDIT_CASES: Readonly<Record<string, readonly KernelAuditCase[]>> = {
  'coverage-global': COVERAGE_GLOBAL_CASES,
  'coverage-network-movement': COVERAGE_NETWORK_MOVEMENT_CASES,
  'coverage-interpolation': COVERAGE_INTERPOLATION_KERNEL_CASES,
  'coverage-geometry': COVERAGE_GEOMETRY_CASES,
  'coverage-terrain': COVERAGE_TERRAIN_CASES,
  'coverage-point-process': [...COVERAGE_POINT_PROCESS_CASES, ...CONTINUOUS_SAMPLING_CASES],
  'coverage-statistics': [
    ...COVERAGE_STATISTICS_CASES,
    ...SPATIAL_VALIDATION_ADVANCED_KERNEL_CASES
  ],
  'coverage-urban-ecology': COVERAGE_URBAN_ECOLOGY_CASES,
  'coverage-raster': COVERAGE_RASTER_KERNEL_CASES,
  backlog: BACKLOG_CASES,
  'coverage-tranche-one': COVERAGE_TRANCHE_ONE_CASES,
  expansion: EXPANSION_CASES,
  geospatial: GEOSPATIAL_CASES,
  dataframe: [
    ...DATAFRAME_CASES,
    ...VARIABLE_STANDARDIZATION_AUDIT_CASES,
    ...EXACT_NATURAL_BREAKS_AUDIT_CASES
  ],
  raster: RASTER_CASES,
  'raster-operations': RASTER_OPERATIONS_CASES,
  terrain: TERRAIN_CASES,
  network: NETWORK_CASES,
  statistics: STATISTICS_CASES,
  recipes: [...RECIPES_CASES, ...RATE_MAPPING_KERNEL_AUDIT_CASES, ...QUANTILE_LISA_AUDIT_CASES],
  geometry: [...GEOMETRY_CASES, ...DORLING_CARTOGRAM_AUDIT_CASES],
  'cells-and-joins': CELLS_AND_JOINS_CASES,
  'points-and-regions': POINTS_AND_REGIONS_CASES
};
