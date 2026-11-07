// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogOperationBinding} from './catalog-binding-types';
import {PLATFORM5_LISA_BINDINGS} from './catalog-platform5-lisa';
import {PLATFORM5_RASTER_BINDINGS} from './catalog-platform5-raster';
import {PLATFORM5_RECIPE_BINDINGS} from './catalog-platform5-recipes';
import {PLATFORM5_REGRESSION_BINDINGS} from './catalog-platform5-regression';
import {PLATFORM5_S2_BINDINGS} from './catalog-platform5-s2';

/** PLATFORM5 newly bound families: workflow recipes, S2 contributors, regression families and raster-tile contributors. */
export const PLATFORM5_NEW_BINDINGS: readonly CatalogOperationBinding[] = [
  ...PLATFORM5_LISA_BINDINGS,
  ...PLATFORM5_RECIPE_BINDINGS,
  ...PLATFORM5_S2_BINDINGS,
  ...PLATFORM5_REGRESSION_BINDINGS,
  ...PLATFORM5_RASTER_BINDINGS
];
