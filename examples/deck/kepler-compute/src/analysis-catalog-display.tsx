// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogEntry, CatalogInputSpec} from '@luma-contributors/kepler';

export function AnalysisCatalogBlockedEntries({entries}: {entries: readonly CatalogEntry[]}) {
  const firstReason = entries[0]?.reason;
  const sharedReason = entries.every(entry => entry.reason === firstReason)
    ? firstReason
    : undefined;
  return (
    <div id="catalog-blocked-list" data-testid="catalog-blocked" style={{fontSize: 12}}>
      {sharedReason && <p>{sharedReason}</p>}
      <ul style={{maxHeight: 160, overflow: 'auto'}}>
        {entries.map(entry => (
          <li
            key={JSON.stringify([entry.identity.package, entry.identity.exportName])}
            title={sharedReason ? undefined : entry.reason}
          >
            {entry.identity.exportName} · {entry.identity.package} · {entry.capabilityId}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function describeCatalogInput(input: CatalogInputSpec, tableSelected: boolean): string {
  const maximumRows = input.maximumFeatures.toLocaleString('en-US');
  if (tableSelected || input.dataKind === 'table')
    return `${input.label} (table rows, ≤ ${maximumRows} rows; no geometry required)`;
  return `${input.label} (${input.accepts.join('/')} features: ≤ ${maximumRows} features and ≤ ${input.maximumVertices.toLocaleString('en-US')} vertices${input.acceptsTable ? `; or table rows: ≤ ${maximumRows} rows` : ''})`;
}

/** Capacity text follows the selected representation even before profiling has finished. */
export function describeCatalogCapacity(
  capacity: {
    label: string;
    rows?: number;
    maximumRows?: number;
    vertices?: number;
    maximumVertices: number;
  },
  tableSelected: boolean
): string {
  const rows = `${capacity.label}: ${capacity.rows ?? 'unknown'} / ${capacity.maximumRows ?? 'unknown'}`;
  return tableSelected
    ? `${rows} table rows; no geometry required.`
    : `${rows} rows; ${capacity.vertices ?? 'unknown'} / ${capacity.maximumVertices} vertices.`;
}
