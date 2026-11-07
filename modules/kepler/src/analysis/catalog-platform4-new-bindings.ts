// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogOperationBinding} from './catalog-binding-types';
import {PLATFORM4_DGGS_BINDINGS} from './catalog-platform4-dggs';
import {PLATFORM4_RASTER_BINDINGS} from './catalog-platform4-raster';
import {PLATFORM4_REGRESSION_BINDINGS} from './catalog-platform4-regression';
import {PLATFORM4_SPHERICAL_BINDINGS} from './catalog-platform4-spherical';
import {PLATFORM4_WEIGHTS_BINDINGS} from './catalog-platform4-weights';

/** PLATFORM4 newly bound families (DGGS, S2, regression, weights algebra, raster-only, workflow recipes). */
export const PLATFORM4_NEW_BINDINGS: readonly CatalogOperationBinding[] = [
  ...PLATFORM4_DGGS_BINDINGS,
  ...PLATFORM4_WEIGHTS_BINDINGS,
  ...PLATFORM4_REGRESSION_BINDINGS,
  ...PLATFORM4_SPHERICAL_BINDINGS,
  ...PLATFORM4_RASTER_BINDINGS
];
