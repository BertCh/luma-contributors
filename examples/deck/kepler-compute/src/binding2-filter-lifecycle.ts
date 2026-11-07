// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerComputeFilter, KeplerComputeTableBinding} from './kepler-table-bridge';

type FieldLookup = readonly {name: string}[];

export type FilterRevisionDiff = {
  added: string[];
  removed: string[];
  /** Same filters in a different order; resident scalar topology is order independent. */
  reordered: boolean;
  /** Filter now reads another field of this dataset. */
  reassigned: string[];
  valueChanged: string[];
  inversionChanged: string[];
  enabledChanged: string[];
  /** Fields read by more than one enabled filter, which share one dimension. */
  duplicateFields: Record<string, string[]>;
};

/** The field a native filter reads in one dataset, or undefined when it does not target it. */
export function getFilterFieldName(
  filter: KeplerComputeFilter,
  datasetId: string,
  fields: FieldLookup
): string | undefined {
  const datasetIds = typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId;
  const index = datasetIds.indexOf(datasetId);
  if (index < 0) return undefined;
  return filter.name?.[index] ?? fields[filter.fieldIdx?.[index] ?? -1]?.name;
}

/**
 * Classifies how one native filter revision differs from the previous one for a dataset, so
 * deletion, reordering, field reassignment, inversion and multiple filters on one field are
 * explicit instead of being inferred from identical-looking plans.
 */
export function diffNativeFilters(
  previous: readonly KeplerComputeFilter[],
  next: readonly KeplerComputeFilter[],
  datasetId: string,
  fields: FieldLookup
): FilterRevisionDiff {
  const targets = (filter: KeplerComputeFilter) =>
    (typeof filter.dataId === 'string' ? [filter.dataId] : filter.dataId).includes(datasetId);
  const before = previous.filter(targets);
  const after = next.filter(targets);
  const beforeById = new Map(before.map(filter => [filter.id, filter]));
  const afterById = new Map(after.map(filter => [filter.id, filter]));
  const diff: FilterRevisionDiff = {
    added: after.filter(filter => !beforeById.has(filter.id)).map(filter => filter.id),
    removed: before.filter(filter => !afterById.has(filter.id)).map(filter => filter.id),
    reordered: false,
    reassigned: [],
    valueChanged: [],
    inversionChanged: [],
    enabledChanged: [],
    duplicateFields: {}
  };
  const commonBefore = before.filter(filter => afterById.has(filter.id)).map(filter => filter.id);
  const commonAfter = after.filter(filter => beforeById.has(filter.id)).map(filter => filter.id);
  diff.reordered = commonBefore.join('\u0000') !== commonAfter.join('\u0000');
  for (const filter of after) {
    const old = beforeById.get(filter.id);
    if (!old) continue;
    if (
      getFilterFieldName(old, datasetId, fields) !== getFilterFieldName(filter, datasetId, fields)
    )
      diff.reassigned.push(filter.id);
    if (JSON.stringify(old.value) !== JSON.stringify(filter.value))
      diff.valueChanged.push(filter.id);
    if (Boolean(old.inverted) !== Boolean(filter.inverted)) diff.inversionChanged.push(filter.id);
    if ((old.enabled !== false) !== (filter.enabled !== false)) diff.enabledChanged.push(filter.id);
  }
  const byField = new Map<string, string[]>();
  for (const filter of after) {
    if (filter.enabled === false || filter.type === 'polygon') continue;
    const field = getFilterFieldName(filter, datasetId, fields);
    if (field) byField.set(field, [...(byField.get(field) ?? []), filter.id]);
  }
  for (const [field, ids] of byField) if (ids.length > 1) diff.duplicateFields[field] = ids;
  return diff;
}

export type DictionaryCoverage = {
  filterId: string;
  datasetId: string;
  field: string;
  /** Selected labels this dataset's dictionary does not contain; they match no rows there. */
  missingLabels: (string | number | boolean)[];
  /** This dataset's dictionary indices the selection resolves to. */
  selectedIndices: number[];
};

/**
 * Resolves one categorical native filter against each targeted dataset's own dictionary. Datasets
 * encode the same label as different indices, so each is mapped independently; labels absent from
 * a dataset are reported (they select nothing there) rather than failing the others.
 */
export function planDictionaryCoverage(
  filter: KeplerComputeFilter,
  datasets: readonly {
    datasetId: string;
    fields: FieldLookup;
    bindings: KeplerComputeTableBinding['fields'];
  }[]
): DictionaryCoverage[] {
  if (filter.type !== 'multiSelect' && filter.type !== 'select') return [];
  const labels = filter.type === 'select' ? [filter.value] : filter.value;
  if (!Array.isArray(labels)) return [];
  return datasets.flatMap(({datasetId, fields, bindings}) => {
    const field = getFilterFieldName(filter, datasetId, fields);
    const binding = field ? bindings[field] : undefined;
    if (!field || binding?.kind !== 'category') return [];
    const dictionary = binding.categories ?? binding.domain;
    return [
      {
        filterId: filter.id,
        datasetId,
        field,
        missingLabels: labels.filter(label => !dictionary.some(entry => Object.is(entry, label))),
        selectedIndices: dictionary.flatMap((entry, index) =>
          labels.some(label => Object.is(entry, label)) ? [index] : []
        )
      }
    ];
  });
}

/** Clear failure text for a filter whose field was never prepared for resident execution. */
export function describeUnpreparedFilterField(
  filter: KeplerComputeFilter,
  datasetId: string,
  field: string | undefined
): string {
  return `Filter ${filter.id} (${filter.type ?? 'unknown'}) reads ${
    field ? `field "${field}"` : 'an unknown field'
  } of dataset ${datasetId}, which is not prepared for resident execution; prepare that column or remove the filter.`;
}
