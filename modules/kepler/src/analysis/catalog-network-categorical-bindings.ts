// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogOperationBinding} from './catalog-binding-types';
import {CATEGORY_BINDINGS} from './catalog-network-categorical-categories';
import {FLOW_BINDINGS} from './catalog-network-categorical-flows';
import {NETWORK_BINDINGS} from './catalog-network-categorical-network';
import {CELL_BINDINGS} from './catalog-network-categorical-cells';
import {DATUM_BINDING} from './catalog-network-categorical-datum';
import {TRANSPORT_BINDING} from './catalog-network-categorical-transport';

/** Existing analysis contributors exposed through the Kepler catalog; no numerical implementations. */
export const NETWORK_CATEGORICAL_BINDINGS: readonly CatalogOperationBinding[] = [
  ...FLOW_BINDINGS,
  ...NETWORK_BINDINGS,
  ...CATEGORY_BINDINGS,
  ...CELL_BINDINGS,
  DATUM_BINDING,
  TRANSPORT_BINDING
];
