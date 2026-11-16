// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  KeplerOperationDescriptor,
  KeplerOperationParameterDescriptor
} from './operation-descriptor-types';
import {EMERGING_HOT_SPOTS_LIMITS} from '../curated/bindings/curated-emerging-hot-spots-binding';

const constant = (
  name: string,
  level: 'basic' | 'advanced' = 'advanced'
): KeplerOperationParameterDescriptor => ({
  name,
  level,
  apply: 'rebuild',
  defaultPolicy: {from: 'constant'}
});

export const CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: 'curated-emerging-hot-spots',
  questions: ['what-changed'],
  action: 'Map emerging hot spots',
  prominence: 1,
  inputs: [
    {
      name: 'points',
      kind: 'points',
      rowSemantics: ['events'],
      fields: [
        {
          role: 'time',
          label: 'Event time (ISO timestamp or numeric epoch)',
          accepts: ['time', 'number', 'integer'],
          required: true,
          parameter: 'timeProperty'
        }
      ]
    }
  ],
  parameters: [
    constant('coordinateSystem'),
    constant('timeUnit'),
    {
      name: 'cellSize',
      level: 'basic',
      apply: 'rebuild',
      defaultPolicy: {
        from: 'data',
        input: 'points',
        rule: 'extent-shorter-side-over-250',
        factor: 50
      }
    },
    constant('timeStep', 'basic'),
    constant('timeInterval'),
    constant('sliceCount', 'basic'),
    constant('radius', 'basic'),
    constant('temporalWindow', 'basic'),
    constant('confidenceLevel'),
    constant('multipleTesting')
  ],
  requirements: [
    {
      input: 'points',
      id: 'maximum-rows',
      severity: 'block',
      limit: EMERGING_HOT_SPOTS_LIMITS.events,
      remedy: 'Aggregate or filter events before materializing this bounded cube'
    },
    {
      input: 'points',
      id: 'maximum-vertices',
      severity: 'block',
      limit: EMERGING_HOT_SPOTS_LIMITS.events,
      remedy: 'Supply one Point per event'
    },
    {
      input: 'points',
      id: 'minimum-rows',
      severity: 'block',
      limit: 1,
      remedy: 'Supply timestamped events'
    },
    {
      input: 'points',
      id: 'no-missing-values',
      role: 'time',
      severity: 'block',
      remedy: 'Supply finite numeric epochs or valid ISO timestamps (missing timezone means UTC)'
    },
    {
      input: 'points',
      id: 'row-semantics',
      severity: 'block',
      expects: ['events'],
      remedy: 'Confirm that each row is one event; values are not summed'
    },
    {
      input: 'points',
      id: 'declared-coordinates',
      severity: 'block',
      remedy: 'Confirm the coordinate system and cell-size units'
    }
  ],
  notes: [
    'Counts events, not values or rates. Calendar bins use UTC; ISO timestamps without an offset are read as UTC, matching profiling; empty cells and empty periods remain zero. Cell size defaults to one fifth of the shorter extent: a bounded exploratory grid, not an optimized hot spot rule.',
    'Benjamini-Hochberg adjusts valid space-time bin p-values by default. Normal-approximation Gi* significance and Mann–Kendall trends require the full population; trends use at least ten slices and no serial-correlation correction. Short series can be UNDETERMINED.',
    'The map uses the exact library category names. The shared categorical palette colours seven classes separately and groups additional classes into Other; exact category names remain in the table. These colours do not encode a hot/cold confidence scale. Trend p-values are not corrected across cells. Derived cell sizes are raised to fit the cell and cube capacities; automatic temporal coverage reserves the maximum 256 slices, while an explicit slice count permits a finer grid. Unsafe explicit cell sizes require confirmation. Capacities are checked again during preparation.'
  ],
  outputs: [
    {
      tableId: 'cells',
      inference: true,
      kind: 'polygons',
      presentation: {column: 'category_name', scale: 'categorical'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['where-significant', 'how-related']
    },
    {
      tableId: 'cube',
      inference: true,
      kind: 'attributes',
      interpretation: {summary: 'row-count', cautions: []}
    }
  ]
};
