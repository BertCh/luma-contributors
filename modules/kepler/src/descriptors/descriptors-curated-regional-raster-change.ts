// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerOperationDescriptor} from './operation-descriptor-types';
import {CURATED_REGIONAL_RASTER_CHANGE_BINDING} from '../curated/bindings/curated-regional-raster-change-binding';

export const CURATED_REGIONAL_RASTER_CHANGE_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: CURATED_REGIONAL_RASTER_CHANGE_BINDING.id,
  questions: ['what-changed'],
  action: 'Compare regional raster change',
  prominence: 2,
  inputs: [
    {name: 'before', kind: 'surface', fields: []},
    {name: 'after', kind: 'surface', fields: []},
    {name: 'zones', kind: 'polygons', fields: []}
  ],
  parameters: [
    {name: 'threshold', level: 'basic', apply: 'rebuild', defaultPolicy: {from: 'constant'}},
    {
      name: 'confirmComparable',
      level: 'basic',
      apply: 'rebuild',
      defaultPolicy: {from: 'constant'},
      confirmation: {
        message:
          'Confirm both bands are calibrated in the same measurement units and zones use the raster CRS and linear units.',
        remedy: 'Calibrate/resample upstream and map a zone layer in the same planar coordinates.'
      }
    }
  ],
  requirements: [
    {
      input: 'zones',
      id: 'minimum-rows',
      severity: 'block',
      limit: 1,
      remedy: 'Supply at least one polygon zone.'
    }
  ],
  overlappingInputs: [
    ['before', 'after'],
    ['before', 'zones']
  ],
  outputs: [
    {
      tableId: 'regions',
      kind: 'polygons',
      presentation: {column: 'mean_change', scale: 'diverging'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['what-changed', 'how-related']
    },
    {
      tableId: 'differences',
      kind: 'attributes',
      interpretation: {summary: 'row-count', cautions: []}
    }
  ],
  notes: CURATED_REGIONAL_RASTER_CHANGE_BINDING.declaredBehavior
};
