// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import type KeplerTable from '@kepler.gl/table';
import {
  listCatalogLensFieldRoles,
  mapCatalogLensUserData,
  type CatalogLensFieldRole,
  type CatalogLensUserData
} from '@luma-contributors/kepler';

/** Rows a lens scenario reads from a native table; larger tables must be filtered first. */
export const MAXIMUM_LENS_USER_ROWS = 2_000_000;

export type LensMappingDraft = {datasetId: string; mapping: Record<string, string>};
export const EMPTY_LENS_MAPPING: LensMappingDraft = {datasetId: '', mapping: {}};

type NativeTableLike = {
  id: string;
  label?: string;
  fields: readonly {name: string; type: string}[];
  dataContainer: {numRows(): number; valueAt(rowIndex: number, columnIndex: number): unknown};
};

/** Numeric (integer or real) fields of a native table, the only fields a role accepts. */
export function listNumericFieldNames(dataset: NativeTableLike): string[] {
  return dataset.fields
    .filter(field => field.type === 'integer' || field.type === 'real')
    .map(field => field.name);
}

/** Explicit, bounded read of the mapped native columns into the lens user table. */
export function readLensUserData(
  dataset: NativeTableLike,
  mapping: Readonly<Record<string, string>>
): CatalogLensUserData {
  const rowCount = dataset.dataContainer.numRows();
  // Row bound exceeded; filter or sample the dataset before running a scenario on it.
  if (rowCount < 1 || rowCount > MAXIMUM_LENS_USER_ROWS)
    throw new Error(`Choose a table with 1 to ${MAXIMUM_LENS_USER_ROWS} rows`);
  const numeric = new Set(listNumericFieldNames(dataset));
  return mapCatalogLensUserData(mapping, rowCount, fieldName => {
    const columnIndex = dataset.fields.findIndex(field => field.name === fieldName);
    if (columnIndex < 0 || !numeric.has(fieldName)) return undefined;
    const values = new Array<number | null>(rowCount);
    for (let row = 0; row < rowCount; row++) {
      const value = Number(dataset.dataContainer.valueAt(row, columnIndex));
      values[row] = Number.isFinite(value) ? value : null;
    }
    return values;
  });
}

/**
 * Typed field mapping for a lens scenario that accepts user data: pick a native table, then map
 * each scenario role (an embedded column name) to a numeric field. Roles the scenario does not
 * read may stay unmapped; the run names any role it needed but did not get.
 */
export function AnalysisCatalogLensMapping({
  scenarioId,
  datasets,
  draft,
  onChange
}: {
  scenarioId: string;
  datasets: readonly KeplerTable[];
  draft: LensMappingDraft;
  onChange(draft: LensMappingDraft): void;
}) {
  const [roles, setRoles] = useState<CatalogLensFieldRole[]>([]);
  useEffect(() => {
    let current = true;
    listCatalogLensFieldRoles(scenarioId).then(
      next => current && setRoles(next),
      () => current && setRoles([])
    );
    return () => {
      current = false;
    };
  }, [scenarioId]);
  const dataset = datasets.find(candidate => candidate.id === draft.datasetId) as
    | NativeTableLike
    | undefined;
  const fieldNames = dataset ? listNumericFieldNames(dataset) : [];
  if (!roles.length) return null;
  return (
    <fieldset data-testid="catalog-lens-mapping" style={{marginTop: 8}}>
      <legend>Run on your data</legend>
      <select
        aria-label="Lens data table"
        value={draft.datasetId}
        onChange={event => onChange({datasetId: event.target.value, mapping: {}})}
        style={{width: '100%'}}
      >
        <option value="">Embedded dataset</option>
        {datasets.map(candidate => (
          <option key={candidate.id} value={candidate.id}>
            {candidate.label}
          </option>
        ))}
      </select>
      {dataset &&
        roles.map(role => (
          <div key={role.role}>
            <label htmlFor={`lens-role-${role.role}`}>
              {role.role}
              {role.example === null ? '' : ` (e.g. ${role.example})`}
            </label>
            <select
              id={`lens-role-${role.role}`}
              value={draft.mapping[role.role] ?? ''}
              onChange={event =>
                onChange({
                  ...draft,
                  mapping: {...draft.mapping, [role.role]: event.target.value}
                })
              }
              style={{width: '100%'}}
            >
              <option value="">Not mapped</option>
              {fieldNames.map(name => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        ))}
    </fieldset>
  );
}
