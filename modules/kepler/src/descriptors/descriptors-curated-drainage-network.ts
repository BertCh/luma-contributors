// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerOperationDescriptor} from './operation-descriptor-types';

export const CURATED_DRAINAGE_NETWORK_DESCRIPTOR: KeplerOperationDescriptor = {
  bindingId: 'curated-drainage-network',
  questions: ['shape-of-the-surface'],
  action: 'Trace drainage and catchments',
  prominence: 1,
  // The raster input already carries one selected numeric band; it is not a feature column.
  inputs: [{name: 'elevation', kind: 'surface', fields: []}],
  parameters: [
    'streamThreshold',
    'fillDepressions',
    'resolveFlats',
    'maximumFillIterations',
    'maximumFlatIterations',
    'maximumAccumulationIterations',
    'maximumSteps',
    'reachCapacity',
    'vertexCapacity'
  ].map((name, index) => ({
    name,
    defaultPolicy: {from: 'constant'},
    apply: 'rebuild',
    level: index < 3 ? 'basic' : 'advanced'
  })),
  requirements: [
    {
      input: 'elevation',
      id: 'planar-coordinates',
      severity: 'block',
      remedy: 'Supply a complete axis-aligned elevation grid in declared planar linear units.'
    },
    {
      input: 'elevation',
      id: 'declared-coordinates',
      severity: 'block',
      remedy: 'Confirm the planar coordinate system and use consistent horizontal linear units.'
    }
  ],
  outputs: [
    {
      tableId: 'streams',
      kind: 'lines',
      presentation: {column: 'stream_order', scale: 'sequential'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['shape-of-the-surface']
    },
    {
      tableId: 'catchment-cells',
      kind: 'polygons',
      presentation: {column: 'catchment_id', scale: 'categorical'},
      interpretation: {summary: 'row-count', cautions: []},
      next: ['how-much-per-area']
    },
    {tableId: 'catchments', kind: 'summary', interpretation: {summary: 'row-count', cautions: []}}
  ],
  notes: [
    'Contributing area is counted in cells. Stream threshold defaults to three cells, a display choice rather than a hydrologically calibrated threshold.',
    'Only the supplied complete planar D8 domain contributes flow. Distances use its linear coordinate units. Elevations are evaluated in float32.',
    'Catchment IDs are terminal outlet or pit cell indices. The map shows cell footprints, not dissolved catchment boundaries or a separate catchment per stream reach.',
    'Filling and flat routing default on. Nonconvergence, unresolved paths or output overflow rejects publication with named stage status. Capacity and path settings of zero derive from the grid cell count.'
  ]
};
