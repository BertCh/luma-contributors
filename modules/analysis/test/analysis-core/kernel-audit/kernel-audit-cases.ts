// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES} from '../../gpu-network/network-centrality/network-centrality-kernel-audit-cases';
import {FITTING_KERNEL_AUDIT_CASES} from './cases-fitting-kernels';
import {QUANTILE_LISA_AUDIT_CASES} from './cases-quantile-lisa';
import {COREGIONALIZATION_FIT_AUDIT_CASES} from './cases-coregionalization-fit';
import {EXACT_NATURAL_BREAKS_AUDIT_CASES} from './cases-exact-natural-breaks';
import {DORLING_CARTOGRAM_AUDIT_CASES} from './cases-dorling-cartogram';
import {MOVEMENT_SEMANTICS_AUDIT_CASES} from './cases-movement-semantics';
import {TIME_GLUE_AUDIT_CASES} from './cases-time-glue';
import {VALUE_CUBE_AUDIT_CASES} from './cases-value-cube';
import {SPACE_TIME_STATISTICS_AUDIT_CASES} from './cases-space-time-statistics';
import {SPACE_TIME_LOCAL_STATISTICS_AUDIT_CASES} from './cases-space-time-local-statistics';
import {STX06_CHANGE_POINTS_STX07_SCAN_AUDIT_CASES} from './cases-stx06-change-points-stx07-scan';
import {STX08_SPATIAL_METHODS_AUDIT_CASES} from './cases-stx08-spatial-methods';
import {STX08_SECOND_GROUP_AUDIT_CASES} from './cases-stx08-colocation-licd-adaptive-density';
import {STM08_STM09_HAWKES_AND_SINKHORN_AUDIT_CASES} from './cases-stm08-stm09-hawkes-and-sinkhorn';
import {STX04_STX05_SPACE_TIME_DENSITY_AND_K_AUDIT_CASES} from './cases-stx04-stx05-space-time-density-and-k';
import {SPACE_TIME_CLUSTERING_AUDIT_CASES} from './cases-space-time-clustering';
import {POINT_NEIGHBOR_GRID_UNIFICATION_AUDIT_CASES} from './cases-point-neighbor-grid-unification';
import {TRACK_SIMILARITY_MEASURE_AUDIT_CASES} from './cases-track-similarity-measures';
import {MOB04_PAIR_MERGE_TOP_K_AUDIT_CASES} from './cases-mob04-pair-merge-top-k';
import {MOB06_LAGRANGIAN_DIAGNOSTICS_AUDIT_CASES} from './cases-mob06-lagrangian-diagnostics';
import {MOB07_HOME_RANGE_RECURSION_AUDIT_CASES} from './cases-mob07-home-range-recursion';
import {MOB10_FLOW_MODELS_AUDIT_CASES} from './cases-mob10-flow-models';
import {MOB05_MOB09_AUDIT_CASES} from './cases-mob05-mob09-edge-volumes-time-geography';
import {SCORING_RULES_AUDIT_CASES} from './cases-scoring-rules';
import {CONFORMAL_PREDICTION_INTERVALS_AUDIT_CASES} from './cases-conformal-prediction-intervals';
import {VECCHIA_LIKELIHOOD_AUDIT_CASES} from './cases-vecchia-likelihood';
import {AREA_OF_APPLICABILITY_AUDIT_CASES} from './cases-area-of-applicability';
import {GTWR_AUDIT_CASES} from './cases-gtwr';
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
import {RTS_TIME_SERIES_AUDIT_CASES} from './cases-rts-time-series';
import {RTS04_RTS07_MONITORING_EOF_AUDIT_CASES} from './cases-rts04-rts07-monitoring-eof';
import {RTS05_COMPOSITING_AUDIT_CASES} from './cases-rts05-compositing';
import {RTS_01_03_EMBEDDING_FDR_AUDIT_CASES} from './cases-rts-01-03-embedding-fdr';
import {BACKLOG_CASES} from './cases-backlog';
import {COVERAGE_TRANCHE_ONE_CASES} from './cases-coverage-tranche-one';
import {EXPANSION_CASES} from './cases-expansion';
import type {KernelAuditCase} from './kernel-audit-context';
import {DATAFRAME_CASES} from './cases-dataframe';
import {FOREST_INFERENCE_AUDIT_CASES} from './cases-forest-inference';
import {WIDE_NUMERIC_SCAN_CASES} from './cases-wide-numeric-scan';
import {RASTER_CASES} from './cases-raster';
import {TERRAIN_CASES} from './cases-terrain';
import {NETWORK_CASES} from './cases-network';
import {STATISTICS_CASES} from './cases-statistics';
import {RECIPES_CASES} from './cases-recipes';
import {GEOMETRY_CASES} from './cases-geometry';
import {CELLS_AND_JOINS_CASES} from './cases-cells-and-joins';
import {POINTS_AND_REGIONS_CASES} from './cases-points-and-regions';
import {POINTCLOUD_AND_DATUM_CASES} from './cases-pointcloud-and-datum';

/** Every kernel-audit case, by area file. */
export const KERNEL_AUDIT_CASES: Readonly<Record<string, readonly KernelAuditCase[]>> = {
  'coregionalization-fit': COREGIONALIZATION_FIT_AUDIT_CASES,
  'coverage-global': COVERAGE_GLOBAL_CASES,
  'coverage-network-movement': COVERAGE_NETWORK_MOVEMENT_CASES,
  'movement-semantics': MOVEMENT_SEMANTICS_AUDIT_CASES,
  'space-time-statistics': SPACE_TIME_STATISTICS_AUDIT_CASES,
  'space-time-local-statistics': SPACE_TIME_LOCAL_STATISTICS_AUDIT_CASES,
  'stx06-change-points-stx07-scan': STX06_CHANGE_POINTS_STX07_SCAN_AUDIT_CASES,
  'stx08-spatial-methods': STX08_SPATIAL_METHODS_AUDIT_CASES,
  'stx08-colocation-licd-adaptive-density': STX08_SECOND_GROUP_AUDIT_CASES,
  'stx04-stx05-space-time-density-and-k': STX04_STX05_SPACE_TIME_DENSITY_AND_K_AUDIT_CASES,
  'stm08-stm09-hawkes-and-sinkhorn': STM08_STM09_HAWKES_AND_SINKHORN_AUDIT_CASES,
  'space-time-clustering': SPACE_TIME_CLUSTERING_AUDIT_CASES,
  'point-neighbor-grid-unification': POINT_NEIGHBOR_GRID_UNIFICATION_AUDIT_CASES,
  'track-similarity-measures': TRACK_SIMILARITY_MEASURE_AUDIT_CASES,
  'mob04-pair-merge-top-k': MOB04_PAIR_MERGE_TOP_K_AUDIT_CASES,
  'mob06-lagrangian-diagnostics': MOB06_LAGRANGIAN_DIAGNOSTICS_AUDIT_CASES,
  'mob07-home-range-recursion': MOB07_HOME_RANGE_RECURSION_AUDIT_CASES,
  'mob10-flow-models': MOB10_FLOW_MODELS_AUDIT_CASES,
  'mob05-mob09-edge-volumes-time-geography': MOB05_MOB09_AUDIT_CASES,
  'scoring-rules': SCORING_RULES_AUDIT_CASES,
  'conformal-prediction-intervals': CONFORMAL_PREDICTION_INTERVALS_AUDIT_CASES,
  'vecchia-likelihood': VECCHIA_LIKELIHOOD_AUDIT_CASES,
  'area-of-applicability': AREA_OF_APPLICABILITY_AUDIT_CASES,
  gtwr: GTWR_AUDIT_CASES,
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
  'rts-time-series': RTS_TIME_SERIES_AUDIT_CASES,
  'rts04-rts07-monitoring-eof': RTS04_RTS07_MONITORING_EOF_AUDIT_CASES,
  'rts05-compositing': RTS05_COMPOSITING_AUDIT_CASES,
  'rts-01-03-embedding-fdr': RTS_01_03_EMBEDDING_FDR_AUDIT_CASES,
  backlog: BACKLOG_CASES,
  'coverage-tranche-one': COVERAGE_TRANCHE_ONE_CASES,
  expansion: EXPANSION_CASES,
  'wide-numeric-scan': WIDE_NUMERIC_SCAN_CASES,
  dataframe: [
    ...DATAFRAME_CASES,
    ...FOREST_INFERENCE_AUDIT_CASES,
    ...TIME_GLUE_AUDIT_CASES,
    ...VALUE_CUBE_AUDIT_CASES,
    ...VARIABLE_STANDARDIZATION_AUDIT_CASES,
    ...EXACT_NATURAL_BREAKS_AUDIT_CASES
  ],
  'analysis-core': FITTING_KERNEL_AUDIT_CASES,
  raster: RASTER_CASES,
  terrain: TERRAIN_CASES,
  network: [...NETWORK_CASES, ...NETWORK_CENTRALITY_SCHEDULE_KERNEL_AUDIT_CASES],
  statistics: STATISTICS_CASES,
  recipes: [...RECIPES_CASES, ...RATE_MAPPING_KERNEL_AUDIT_CASES, ...QUANTILE_LISA_AUDIT_CASES],
  geometry: [...GEOMETRY_CASES, ...DORLING_CARTOGRAM_AUDIT_CASES],
  'cells-and-joins': CELLS_AND_JOINS_CASES,
  'points-and-regions': POINTS_AND_REGIONS_CASES,
  'pointcloud-and-datum': POINTCLOUD_AND_DATUM_CASES
};
