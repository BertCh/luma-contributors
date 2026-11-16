// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useState} from 'react';
import {FACETS, FACET_TITLES, type FacetCounts, type FacetKind} from '../../catalog/index';
import {FacetIcon} from '../brand/facet-icon';

export type FacetSelection = Partial<Record<FacetKind, string>>;

/** Shared questions and three rows of facet chips with live counts. Selecting a chip again clears it. */
export function FacetPicker({
  selection,
  counts,
  onSelect
}: {
  selection: FacetSelection;
  counts: FacetCounts;
  onSelect: (kind: FacetKind, id: string | undefined) => void;
}) {
  // The data row starts open; use and domain start collapsed (and so does data on phones).
  const [collapsed, setCollapsed] = useState<Partial<Record<FacetKind, boolean>>>({
    use: true,
    domain: true
  });
  useEffect(() => {
    if (window.matchMedia('(max-width: 640px)').matches) {
      setCollapsed({question: true, data: true, use: true, domain: true});
    }
  }, []);
  return (
    <div className="facet-picker">
      {(Object.keys(FACETS) as FacetKind[]).map(kind => (
        <details
          key={kind}
          className="facet-row"
          open={!collapsed[kind]}
          onToggle={event => {
            const isOpen = event.currentTarget.open;
            setCollapsed(previous =>
              previous[kind] === !isOpen ? previous : {...previous, [kind]: !isOpen}
            );
          }}
        >
          <summary>
            <span>{FACET_TITLES[kind]}</span>
            {selection[kind] ? (
              <span className="facet-selected">
                {FACETS[kind].find(facet => facet.id === selection[kind])?.label}
              </span>
            ) : null}
          </summary>
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
                    {kind !== 'question' ? <FacetIcon id={facet.id} size={16} /> : null}
                    <span>{facet.label}</span>
                    <span className="facet-count">{count}</span>
                  </button>
                );
              })}
          </div>
        </details>
      ))}
    </div>
  );
}
