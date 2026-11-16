// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeCatalogTableInput, type CatalogTableInput} from '../../analysis/catalog-data-input';

export const CELL_FLOW_TOPOLOGY_SAMPLE_LICENSE = {
  license: 'MIT',
  attribution:
    'vis.gl contributors; repository-authored synthetic OD weights, no external observations',
  reference: 'THIRD-PARTY-DATA.md#curated-cell-flow-topology-synthetic-samples'
} as const;

/** The pentagon fixture of analysis/test/geospatial-expansion/network-expansion.spec.ts. */
export function createCellFlowTopologySample(): CatalogTableInput {
  const origin = '81083ffffffffff';
  const neighbor = '8108bffffffffff';
  const remote = '81e6bffffffffff';
  return makeCatalogTableInput(
    {
      id: 'cell-flow-topology',
      label: 'Synthetic pentagon OD flows',
      rowCount: 6,
      columnTypes: {row_id: 'string', origin: 'string', destination: 'string', flow: 'float64'},
      columns: {
        row_id: ['a', 'b', 'c', 'd', 'e', 'invalid'],
        origin: [origin, origin, neighbor, origin, origin, null],
        destination: [neighbor, neighbor, origin, remote, origin, neighbor],
        flow: [2, 3, 4, 7, 11, 17]
      }
    },
    {maximumRows: 6, identityColumn: 'row_id'}
  );
}

/** Independent antimeridian dataset: renamed fields, different weights, order, and bad rows. */
export function createSecondCellFlowTopologySample(): CatalogTableInput {
  const origin = '827eb7fffffffff';
  const neighbor = '827e87fffffffff';
  const remote = '82e697fffffffff';
  return makeCatalogTableInput(
    {
      id: 'independent-freight',
      label: 'Synthetic antimeridian freight',
      rowCount: 11,
      columnTypes: {shipment: 'string', from_hex: 'string', to_hex: 'string', tonnes: 'float64'},
      columns: {
        shipment: [
          'remote',
          'repeat',
          'repeat',
          'return',
          'distant',
          'self',
          'missing',
          'negative',
          'overflow',
          'wrong-resolution',
          'invalid-h3'
        ],
        from_hex: [
          remote,
          origin.toUpperCase(),
          `0x0${origin}`,
          neighbor,
          origin,
          origin,
          origin,
          origin,
          origin,
          '81083ffffffffff',
          '820000000000000'
        ],
        to_hex: [
          origin,
          neighbor,
          neighbor,
          origin,
          remote,
          origin,
          neighbor,
          neighbor,
          neighbor,
          neighbor,
          neighbor
        ],
        tonnes: [3, 6, 1, 8, 2, 5, null, -1, 1e40, 4, 9]
      }
    },
    {maximumRows: 11, identityColumn: 'shipment'}
  );
}
