// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogOperationBinding} from '../../analysis/catalog-binding-types';
import {CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING} from './curated-clusters-and-footprints-binding';
import {CURATED_CONTIGUITY_CORRELOGRAM_BINDING} from './curated-contiguity-correlogram-binding';
import {CURATED_EXCESS_RISK_BINDING} from './curated-excess-risk-binding';
import {CURATED_KRIGING_WITH_DIAGNOSTICS_BINDING} from './curated-kriging-with-diagnostics-binding';
import {CURATED_RATE_CLUSTER_MAP_BINDING} from './curated-rate-cluster-map-binding';
import {CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING} from './curated-regression-with-diagnostics-binding';

/**
 * The six curated bindings that fuse operations the catalog already computes on feature
 * collections. The catalog registry spreads this list into `KEPLER_CATALOG_BINDINGS`; none of
 * these files imports the registry.
 */
export const KEPLER_CURATED_BINDINGS: readonly CatalogOperationBinding[] = [
  CURATED_RATE_CLUSTER_MAP_BINDING,
  CURATED_EXCESS_RISK_BINDING,
  CURATED_CLUSTERS_AND_FOOTPRINTS_BINDING,
  CURATED_REGRESSION_WITH_DIAGNOSTICS_BINDING,
  CURATED_KRIGING_WITH_DIAGNOSTICS_BINDING,
  CURATED_CONTIGUITY_CORRELOGRAM_BINDING
];
