// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerOperationDescriptor} from './operation-descriptor-types';
import {CELL_FLOW_TOPOLOGY_LIMITS} from '../curated/bindings/curated-cell-flow-topology-binding';

export const CURATED_CELL_FLOW_TOPOLOGY_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: 'curated-cell-flow-topology',
  questions: ['how-it-moves'],
  action: 'Map neighboring cell flows',
  prominence: 2,
  inputs: [
    {
      name: 'flows',
      kind: 'attributes',
      fields: [
        {
          role: 'origin',
          label: 'Origin H3 cell',
          accepts: ['text', 'category'],
          required: true,
          parameter: 'originProperty'
        },
        {
          role: 'destination',
          label: 'Destination H3 cell',
          accepts: ['text', 'category'],
          required: true,
          parameter: 'destinationProperty'
        },
        {
          role: 'flow',
          label: 'Flow',
          accepts: ['number', 'integer'],
          measurement: ['count', 'amount'],
          required: true,
          parameter: 'flowProperty'
        }
      ]
    }
  ],
  parameters: [
    {name: 'resolution', level: 'basic', apply: 'rebuild', defaultPolicy: {from: 'constant'}}
  ],
  requirements: [
    {
      input: 'flows',
      id: 'minimum-rows',
      severity: 'block',
      limit: 1,
      remedy: 'Load a geometry-free OD table with exact H3 hex strings and a numeric flow column'
    },
    {
      input: 'flows',
      id: 'maximum-rows',
      severity: 'block',
      limit: CELL_FLOW_TOPOLOGY_LIMITS.pairs,
      remedy: 'Preaggregate repeated OD pairs or filter the table before running'
    },
    {
      input: 'flows',
      id: 'non-negative-values',
      severity: 'warn',
      role: 'flow',
      remedy: 'Negative flows are retained as rejected rows and excluded from all totals'
    },
    {
      input: 'flows',
      id: 'no-missing-values',
      severity: 'warn',
      role: 'flow',
      remedy: 'Missing and nonfinite float32 flows are retained as rejected rows'
    }
  ],
  outputs: [
    {tableId: 'pairs', kind: 'attributes', interpretation: {summary: 'row-count', cautions: []}},
    {tableId: 'cells', kind: 'attributes', interpretation: {summary: 'row-count', cautions: []}},
    {
      tableId: 'edges',
      kind: 'lines',
      presentation: {column: 'flow', scale: 'sequential'},
      interpretation: {summary: 'value-range', cautions: []},
      next: ['how-it-moves']
    }
  ],
  notes: [
    'Supply H3 cells as exact hex text at the selected resolution, never decimal JavaScript numbers. Semantic H3 validation and pair validity run on the GPU. Invalid keys, wrong resolutions and nonnegative-flow failures remain visible in the original row table.',
    'Totals include every valid OD pair, including self and distant pairs. Only immediate neighbors appear as boundary flows; the sum on the map can be smaller than total flow. No routes are inferred.',
    'Opposite directions share a boundary and remain separate rows. Zero-flow boundaries are included. Geometry declares EPSG:4326 degrees, and antimeridian boundaries are split into MultiLineString parts within one directed-edge row. Flows keep input units and are rounded to float32 before summation; boundary_count retains the original recipe vertex count before splitting.',
    `The direct-scan recipe accepts at most ${CELL_FLOW_TOPOLOGY_LIMITS.cells} distinct decoded cells and ${CELL_FLOW_TOPOLOGY_LIMITS.cellPairWork} cell/pair combinations; preparation checks these limits before allocation. Nonfinite aggregates stop publication.`
  ]
};
