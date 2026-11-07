// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerComputeFilter, KeplerComputeTableBinding} from './kepler-table-bridge';
import {
  diffNativeFilters,
  getFilterFieldName,
  planDictionaryCoverage,
  type FilterRevisionDiff
} from './binding2-filter-lifecycle';

type FieldLookup = readonly {name: string}[];

/** What one filter revision cost the resident scalar selection of one dataset. */
export type FilterRevisionReport = {
  datasetId: string;
  diff: FilterRevisionDiff;
  /** True when the scalar topology changed, so every dimension was rebuilt; false when only controls changed. */
  scalarRebuilt: boolean;
  /** Short reason for the rebuild, from the diff. */
  rebuildReason: string;
};

/** Explains in one phrase which edit forced (or avoided) a scalar rebuild. */
export function describeScalarRebuildReason(diff: FilterRevisionDiff, rebuilt: boolean): string {
  if (!rebuilt) {
    if (diff.inversionChanged.length) return 'inversion toggled in place';
    if (diff.valueChanged.length) return 'value updated in place';
    if (diff.reordered) return 'reordered; topology is order independent';
    return 'unchanged';
  }
  const reasons: string[] = [];
  if (diff.added.length) reasons.push(`added ${diff.added.join(', ')}`);
  if (diff.removed.length) reasons.push(`removed ${diff.removed.join(', ')}`);
  if (diff.reassigned.length) reasons.push(`reassigned ${diff.reassigned.join(', ')}`);
  if (diff.enabledChanged.length) reasons.push(`enabled changed ${diff.enabledChanged.join(', ')}`);
  return reasons.join('; ') || 'dimension layout changed';
}

export function makeFilterRevisionReport(
  previous: readonly KeplerComputeFilter[],
  next: readonly KeplerComputeFilter[],
  datasetId: string,
  fields: FieldLookup,
  scalarRebuilt: boolean
): FilterRevisionReport {
  const diff = diffNativeFilters(previous, next, datasetId, fields);
  return {
    datasetId,
    diff,
    scalarRebuilt,
    rebuildReason: describeScalarRebuildReason(diff, scalarRebuilt)
  };
}

export type CategoryCoverageWarning = {
  datasetId: string;
  filterId: string;
  field: string;
  missingLabels: (string | number | boolean)[];
  message: string;
};

/**
 * Categorical filters whose selected labels are absent from this dataset's own dictionary. Datasets
 * encode one label as different indices, so the check is per dataset; the labels select nothing here.
 */
export function listCategoryCoverageWarnings(
  filters: readonly KeplerComputeFilter[],
  datasetId: string,
  fields: FieldLookup,
  bindings: KeplerComputeTableBinding['fields']
): CategoryCoverageWarning[] {
  const warnings: CategoryCoverageWarning[] = [];
  for (const filter of filters) {
    if (filter.enabled === false) continue;
    for (const coverage of planDictionaryCoverage(filter, [{datasetId, fields, bindings}])) {
      if (!coverage.missingLabels.length) continue;
      const labels = coverage.missingLabels.map(String).join(', ');
      warnings.push({
        datasetId,
        filterId: filter.id,
        field: coverage.field,
        missingLabels: coverage.missingLabels,
        message: `Filter ${filter.id} selects ${labels} on ${coverage.field}, which dataset ${datasetId} does not contain; ${
          coverage.selectedIndices.length
            ? `only its other ${coverage.selectedIndices.length} selected categories match`
            : 'no rows match'
        } there.`
      });
    }
  }
  return warnings;
}

/** Collects per-dataset category coverage warnings for the UI; datasets replace only their own entries. */
export class CategoryCoverageRegistry {
  private readonly byDataset = new Map<
    string,
    {owner: object; warnings: readonly CategoryCoverageWarning[]}
  >();
  private readonly listeners = new Set<() => void>();

  /** The owner is the binding writing the entry; a replaced binding must not clear its successor's. */
  set(datasetId: string, warnings: readonly CategoryCoverageWarning[], owner: object): void {
    const previous = this.byDataset.get(datasetId)?.warnings;
    if (
      (previous?.length ?? 0) === warnings.length &&
      (previous ?? []).every((entry, index) => entry.message === warnings[index].message)
    ) {
      if (previous) this.byDataset.set(datasetId, {owner, warnings});
      return;
    }
    if (warnings.length) this.byDataset.set(datasetId, {owner, warnings});
    else this.byDataset.delete(datasetId);
    for (const listener of this.listeners) listener();
  }

  clear(datasetId: string, owner: object): void {
    if (this.byDataset.get(datasetId)?.owner === owner) this.set(datasetId, [], owner);
  }

  list(): CategoryCoverageWarning[] {
    return [...this.byDataset.values()].flatMap(entry => [...entry.warnings]);
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  }
}

/** Process-wide registry read by the sandbox UI; bindings write their dataset's entry. */
export const categoryCoverageRegistry = new CategoryCoverageRegistry();

export {getFilterFieldName};
