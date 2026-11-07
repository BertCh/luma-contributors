// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {FACETS, FACET_TITLES, type FacetCounts, type FacetKind} from '../../catalog';
import {FacetIcon} from '../brand/facet-icon';

export type FacetSelection = Partial<Record<FacetKind, string>>;

/** Three rows of facet chips with live counts. Selecting a chip again clears it. */
export function FacetPicker({
  selection,
  counts,
  onSelect
}: {
  selection: FacetSelection;
  counts: FacetCounts;
  onSelect: (kind: FacetKind, id: string | undefined) => void;
}) {
  return (
    <div className="facet-picker">
      {(Object.keys(FACETS) as FacetKind[]).map(kind => (
        <fieldset key={kind} className="facet-row">
          <legend>{FACET_TITLES[kind]}</legend>
          <div className="facet-chips">
            {[...FACETS[kind]]
              .filter(facet => (counts[kind][facet.id] ?? 0) > 0 || selection[kind] === facet.id)
              .map(facet => {
                const selected = selection[kind] === facet.id;
                const count = counts[kind][facet.id] ?? 0;
                return (
                  <button
                    key={facet.id}
                    type="button"
                    className="facet-chip"
                    aria-pressed={selected}
                    disabled={count === 0 && !selected}
                    title={facet.description}
                    onClick={() => onSelect(kind, selected ? undefined : facet.id)}
                  >
                    <FacetIcon id={facet.id} size={16} />
                    <span>{facet.label}</span>
                    <span className="facet-count">{count}</span>
                  </button>
                );
              })}
          </div>
        </fieldset>
      ))}
    </div>
  );
}
