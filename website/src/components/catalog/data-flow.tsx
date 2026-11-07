// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {getFacet} from '../../catalog';
import type {DataTypeId} from '../../catalog/taxonomy';
import {Chip} from '../ui/chip';

function DataTypeChips({ids, compact}: {ids: DataTypeId[]; compact?: boolean}) {
  return (
    <span className="data-flow-types">
      {(compact ? ids.slice(0, 1) : ids).map(id => {
        const label = getFacet('data', id)?.label ?? id;
        return compact ? (
          <Chip key={id}>{label}</Chip>
        ) : (
          <Link key={id} to="/explore/data/$id" params={{id}} className="chip-link">
            <Chip>{label}</Chip>
          </Link>
        );
      })}
      {compact && ids.length > 1 ? <Chip>+{ids.length - 1}</Chip> : null}
    </span>
  );
}

/** Data in -> data out, as chips joined by an arrow. */
export function DataFlow({
  input,
  output,
  compact = false
}: {
  input: DataTypeId[];
  output: DataTypeId[];
  compact?: boolean;
}) {
  return (
    <span className="data-flow" aria-label="Data in, data out">
      <DataTypeChips ids={input} compact={compact} />
      <svg
        className="data-flow-arrow"
        viewBox="0 0 32 12"
        width="28"
        height="12"
        aria-hidden="true"
      >
        <path
          d="M1 6h27M22 1l6 5-6 5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <DataTypeChips ids={output} compact={compact} />
    </span>
  );
}
