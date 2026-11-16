// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CURATED_DRIVE_TIME_BANDS_BINDING} from './curated-drive-time-bands-binding';
import {CURATED_CELL_FLOW_TOPOLOGY_BINDING} from './curated-cell-flow-topology-binding';
import {CURATED_NETWORK_DISRUPTION_BINDING} from './curated-network-disruption-binding';
import {CURATED_REGIONAL_RASTER_CHANGE_BINDING} from './curated-regional-raster-change-binding';
import {CURATED_ZONAL_EXPOSURE_BINDING} from './curated-zonal-exposure-binding';
import {CURATED_SURFACE_PREPARATION_BINDING} from './curated-surface-preparation-binding';
import {CURATED_DRAINAGE_NETWORK_BINDING} from './curated-drainage-network-binding';
import {CURATED_LANDSCAPE_METRICS_BINDING} from './curated-landscape-metrics-binding';
import {CURATED_EMERGING_HOT_SPOTS_BINDING} from './curated-emerging-hot-spots-binding';
import {CURATED_FLEET_DWELL_BINDING} from './curated-fleet-dwell-binding';
import {CURATED_TRACK_QUALITY_BINDING} from './curated-track-quality-binding';
import {CURATED_DASYMETRIC_REDISTRIBUTION_BINDING} from './curated-dasymetric-redistribution-binding';

import type {CatalogOperationBinding} from '../../analysis/catalog-binding-types';
import {CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING} from './curated-clusters-and-footprints-binding';
import {CURATED_CONTIGUITY_CORRELOGRAM_BINDING} from './curated-contiguity-correlogram-binding';
import {CURATED_EXCESS_RISK_BINDING} from './curated-excess-risk-binding';
import {CURATED_KRIGING_WITH_DIAGNOSTICS_BINDING} from './curated-kriging-with-diagnostics-binding';
import {CURATED_RATE_CLUSTER_MAP_BINDING} from './curated-rate-cluster-map-binding';
import {CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING} from './curated-regression-with-diagnostics-binding';

/**
 * The curated bindings that compose analysis recipes and existing operations.
 * The catalog registry spreads this list into `KEPLER_CATALOG_BINDINGS`; none of
 * these files imports the registry.
 */
export const KEPLER_CURATED_BINDINGS: readonly CatalogOperationBinding[] = [
  CURATED_RATE_CLUSTER_MAP_BINDING,
  CURATED_EXCESS_RISK_BINDING,
  CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING,
  CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING,
  CURATED_KRIGING_WITH_DIAGNOSTICS_BINDING,
  CURATED_CONTIGUITY_CORRELOGRAM_BINDING,
  CURATED_EMERGING_HOT_SPOTS_BINDING,
  CURATED_DASYMETRIC_REDISTRIBUTION_BINDING,
  CURATED_TRACK_QUALITY_BINDING,
  CURATED_LANDSCAPE_METRICS_BINDING,
  CURATED_DRAINAGE_NETWORK_BINDING,
  CURATED_SURFACE_PREPARATION_BINDING,
  CURATED_ZONAL_EXPOSURE_BINDING,
  CURATED_REGIONAL_RASTER_CHANGE_BINDING,
  CURATED_NETWORK_DISRUPTION_BINDING,
  CURATED_CELL_FLOW_TOPOLOGY_BINDING,
  CURATED_DRIVE_TIME_BANDS_BINDING,
  CURATED_FLEET_DWELL_BINDING
];

export * from './curated-emerging-hot-spots-binding';
export * from './curated-dasymetric-redistribution-binding';
export * from './curated-track-quality-binding';
export * from './curated-landscape-metrics-binding';
export * from './curated-fleet-dwell-binding';

export * from './curated-drainage-network-binding';

export * from './curated-surface-preparation-binding';

export * from './curated-zonal-exposure-binding';

export * from './curated-regional-raster-change-binding';

export * from './curated-network-disruption-binding';

export * from './curated-cell-flow-topology-binding';

export * from './curated-drive-time-bands-binding';
