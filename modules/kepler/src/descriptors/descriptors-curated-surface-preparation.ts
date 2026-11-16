// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  CURATED_SURFACE_PREPARATION_BINDING,
  MAXIMUM_SURFACE_PREPARATION_CELLS
} from '../curated/bindings/curated-surface-preparation-binding';
import type {KeplerOperationDescriptor} from './operation-descriptor-types';

/** The selected decoded band is the input itself; there is no string column parameter to infer. */
export const CURATED_SURFACE_PREPARATION_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: CURATED_SURFACE_PREPARATION_BINDING.id,
  questions: ['prepare-data'],
  action: 'Fill surface gaps',
  prominence: 2,
  inputs: [{name: 'surface', kind: 'surface', fields: []}],
  parameters: CURATED_SURFACE_PREPARATION_BINDING.parameters.map(parameter => ({
    name: parameter.name,
    apply: 'rebuild',
    defaultPolicy: {from: 'constant'},
    level: ['maximumDistance', 'method', 'filter'].includes(parameter.name) ? 'basic' : 'advanced',
    ...(['radius', 'quantile', 'borderMode', 'noDataPolicy'].includes(parameter.name)
      ? {activeWhen: {parameter: 'filter', equals: [true]}}
      : {})
  })),
  requirements: [
    {
      input: 'surface',
      id: 'maximum-rows',
      severity: 'block',
      limit: MAXIMUM_SURFACE_PREPARATION_CELLS,
      remedy:
        'Crop the decoded grid to at most 65536 cells; the binding also checks the search-work limit.'
    }
  ],
  outputs: [
    {
      tableId: 'prepared-surface',
      kind: 'surface',
      presentation: {column: 'value', scale: 'sequential'},
      interpretation: {summary: 'value-range', cautions: []},
      next: ['shape-of-the-surface', 'what-changed']
    }
  ],
  notes: [
    'Filled cells are estimates, not observations. Inspect original_valid, filled, filtered and valid independently; filled and filtered can both be true.',
    'unresolvedCount counts cells invalid after filtering. A propagating filter can invalidate an observed or filled center; the status and filterInvalidatedCount say so.',
    'Search distance and rank-filter radius are in pixels. Constants describe a repair policy rather than a physical scale inferred from the extent.',
    'Requires a complete decoded grid and WebGPU. EPSG:4326 outputs have polygon footprints; projected outputs are tables pending host CRS transformation.'
  ]
};
