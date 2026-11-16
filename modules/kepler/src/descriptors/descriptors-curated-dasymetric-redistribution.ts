// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerOperationDescriptor} from './operation-descriptor-types';
import {
  MAXIMUM_DASYMETRIC_ZONES,
  MAXIMUM_DASYMETRIC_VERTICES
} from '../curated/bindings/curated-dasymetric-redistribution-binding';

export const CURATED_DASYMETRIC_REDISTRIBUTION_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: 'curated-dasymetric-redistribution',
  questions: ['prepare-data', 'how-much-per-area'],
  action: 'Redistribute source totals',
  prominence: 2,
  inputs: [
    {
      name: 'sources',
      kind: 'polygons',
      rowSemantics: ['objects'],
      fields: [
        {
          role: 'total',
          label: 'Extensive count or amount',
          accepts: ['number', 'integer'],
          measurement: ['count', 'amount'],
          required: true,
          parameter: 'totalProperty'
        }
      ]
    },
    {name: 'targets', kind: 'polygons', fields: []},
    {name: 'ancillary', kind: 'surface', fields: []}
  ],
  parameters: [
    {name: 'targetCellOwners', defaultPolicy: {from: 'constant'}, apply: 'rebuild', level: 'basic'},
    {name: 'denominator', defaultPolicy: {from: 'constant'}, apply: 'rebuild', level: 'basic'},
    {name: 'pairCapacity', defaultPolicy: {from: 'constant'}, apply: 'rebuild', level: 'advanced'},
    {
      name: 'coordinatesMatchRaster',
      defaultPolicy: {from: 'constant'},
      apply: 'rebuild',
      level: 'basic',
      confirmation: {
        message: 'The zones and raster share one planar coordinate system and unit.',
        remedy: 'Reproject the zones and raster together and confirm their CRS and units.'
      }
    },
    {
      name: 'targetOwnershipConfirmed',
      defaultPolicy: {from: 'constant'},
      apply: 'rebuild',
      level: 'basic',
      confirmation: {
        message: 'Each assigned cell belongs wholly to its target row, in target input order.',
        remedy:
          'Supply disjoint whole-cell ownership; split cells or prepare an aligned grid before confirming.'
      }
    }
  ],
  requirements: [
    ...['sources', 'targets'].flatMap(input => [
      {
        input,
        id: 'maximum-rows' as const,
        severity: 'block' as const,
        limit: MAXIMUM_DASYMETRIC_ZONES,
        remedy: 'Reduce the number of source or target zones.'
      },
      {
        input,
        id: 'maximum-vertices' as const,
        severity: 'block' as const,
        limit: MAXIMUM_DASYMETRIC_VERTICES,
        remedy: 'Simplify the polygon geometry.'
      }
    ]),
    {
      input: 'sources',
      id: 'non-negative-values',
      role: 'total',
      severity: 'block',
      remedy:
        'Use nonnegative extensive counts or amounts; rates cannot be redistributed as totals.'
    }
  ],
  outputs: [
    {
      tableId: 'dasymetric-targets',
      kind: 'polygons',
      kindFrom: 'targets',
      presentation: {column: 'total', scale: 'sequential'},
      interpretation: {summary: 'value-range', cautions: []},
      next: ['how-much-per-area', 'how-related']
    },
    {
      tableId: 'dasymetric-ledger',
      kind: 'summary',
      interpretation: {summary: 'row-count', cautions: []}
    },
    {
      tableId: 'dasymetric-status',
      kind: 'summary',
      interpretation: {summary: 'row-count', cautions: []}
    }
  ],
  notes: [
    'Ancillary density is per square affine-coordinate unit. Ingestion validity and nodata are preserved; zero or negative samples provide no support. Finite density conversion overflow is rejected.',
    'Source polygon coverage is fractional. Targets own whole cells by explicit row index; agreement between ownership and target geometry requires confirmation.',
    'The zone default preserves uncovered residual mass. overlap redistributes supported mass among assigned targets. Zero-support sources retain their entire totals.',
    'Map only complete totals. Incomplete coverage, invalid totals, arithmetic failure or pair overflow suppress derived values to null; stage evidence explains failure.',
    'Dense source-cell and target-tuple work cannot exceed 4194304. Prepare larger domains using disjoint cores before normalization; this adapter does not normalize tiles separately.'
  ]
};
