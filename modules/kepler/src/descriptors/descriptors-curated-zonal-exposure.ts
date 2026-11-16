// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerOperationDescriptor} from './operation-descriptor-types';
import {MAXIMUM_ZONAL_EXPOSURE_CELLS} from '../curated/bindings/curated-zonal-exposure-binding';

/** Metadata for the genuine raster recipes; registration is owned by the integration stream. */
export const CURATED_ZONAL_EXPOSURE_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: 'curated-zonal-exposure',
  questions: ['how-much-per-area'],
  action: 'Summarise hazard and exposure by zone',
  prominence: 2,
  inputs: [
    {name: 'zones', kind: 'polygons', rowSemantics: ['objects'], fields: []},
    {name: 'surface', kind: 'surface', rowSemantics: ['surface-samples'], fields: []},
    {
      name: 'cells',
      kind: 'attributes',
      fields: [
        {
          role: 'cell-index',
          label: 'Zero-based raster cell index',
          accepts: ['number', 'integer'],
          measurement: ['identifier'],
          required: true,
          parameter: 'cellIndexProperty',
          autoAssign: false
        },
        {
          role: 'zone',
          label: 'Whole-cell zone ID (null for unassigned)',
          accepts: ['number', 'integer'],
          measurement: ['identifier'],
          required: true,
          parameter: 'zoneProperty',
          autoAssign: false
        },
        {
          role: 'population',
          label: 'Population count or density',
          accepts: ['number', 'integer'],
          measurement: ['count', 'rate', 'amount'],
          required: {parameter: 'weightMeaning', equals: ['count', 'density']},
          parameter: 'populationProperty',
          autoAssign: false
        },
        {
          role: 'cell-area',
          label: 'Cell area in the density denominator unit',
          accepts: ['number', 'integer'],
          measurement: ['amount'],
          required: {parameter: 'weightMeaning', equals: ['density']},
          parameter: 'cellAreaProperty',
          autoAssign: false
        }
      ]
    }
  ],
  parameters: [
    {name: 'probabilities', level: 'basic', apply: 'rebuild', defaultPolicy: {from: 'constant'}},
    {name: 'weightMeaning', level: 'basic', apply: 'rebuild', defaultPolicy: {from: 'constant'}},
    {
      name: 'threshold',
      level: 'basic',
      apply: 'live',
      defaultPolicy: {from: 'constant'},
      activeWhen: {parameter: 'weightMeaning', equals: ['count', 'density']}
    },
    {
      name: 'densityAreaUnit',
      level: 'basic',
      apply: 'rebuild',
      defaultPolicy: {from: 'constant'},
      activeWhen: {parameter: 'weightMeaning', equals: ['density']}
    },
    {
      name: 'cellAreaUnit',
      level: 'basic',
      apply: 'rebuild',
      defaultPolicy: {from: 'constant'},
      activeWhen: {parameter: 'weightMeaning', equals: ['density']}
    }
  ],
  requirements: [
    {
      input: 'cells',
      id: 'maximum-rows',
      severity: 'block',
      limit: MAXIMUM_ZONAL_EXPOSURE_CELLS,
      remedy: 'Use a complete raster of at most 65,536 cells and one ownership row per cell'
    },
    {
      input: 'cells',
      id: 'positive-values',
      severity: 'warn',
      role: 'cell-area',
      activeWhen: {parameter: 'weightMeaning', equals: ['density']},
      remedy:
        'Supply positive cell areas in the density denominator unit; invalid areas are omitted'
    },
    {
      input: 'cells',
      id: 'non-negative-values',
      severity: 'warn',
      role: 'population',
      activeWhen: {parameter: 'weightMeaning', equals: ['count', 'density']},
      remedy: 'Supply nonnegative population weights; invalid weights are omitted'
    }
  ],
  outputs: [
    {
      tableId: 'zone-summary',
      kind: 'polygons',
      kindFrom: 'zones',
      presentation: {column: 'median', scale: 'sequential'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['how-much-per-area']
    },
    {
      tableId: 'zone-quantiles',
      kind: 'summary',
      interpretation: {summary: 'row-count', cautions: []}
    },
    {
      tableId: 'recipe-status',
      kind: 'summary',
      interpretation: {summary: 'row-count', cautions: []}
    }
  ],
  notes: [
    'Whole-cell ownership must be supplied explicitly as zero-based indices into the supplied zone polygons. Boundary fractions are not inferred. The map shows median hazard; exposure and coverage are separate report columns.',
    'Distribution statistics are unweighted. Exposure is an optional separate branch; choose count per cell or density and supply matching area units explicitly.',
    'Exposed share is exposed population divided by observed population, not total population. Coverage is observed population divided by total population; missing hazards do not remove people from the total.',
    'Raster cells, ownership rows and dense zone IDs must meet the binding shape and capacity checks. These checks run before GPU allocation; this descriptor does not certify alignment from column summaries.'
  ]
};
