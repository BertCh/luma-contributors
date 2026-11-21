// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatResourceBytes, formatResourceCount} from '../../lens/format';
import type {LensViewModel} from '../../lens/types';
import './meta.css';

/** The largest resources of a recorded run, as proportional bars. */
export function ResourceBars({model, limit = 6}: {model: LensViewModel; limit?: number}) {
  const rows = model.resources.largest.slice(0, limit);
  const maximum = Math.max(1, ...rows.map(row => row.bytes));
  return (
    <div className="meta-panel">
      <div className="meta-panel__bar">
        <span className="meta-panel__title">{model.title}: largest resources</span>
        <ul className="meta-stats">
          <li>
            <strong>
              {formatResourceBytes(model.resources.totalBytes, model.resources.totalBytesKind)}
            </strong>{' '}
            in {formatResourceCount(model.resources)}
          </li>
        </ul>
      </div>
      <ul className="resource-bars">
        {rows.map(row => (
          <li key={row.id}>
            <span className="resource-bars__name" title={row.id}>
              {shortenResourceName(row.id)}
            </span>
            <span className="resource-bars__track">
              <span
                className="resource-bars__fill"
                style={{width: `${(100 * row.bytes) / maximum}%`}}
              />
            </span>
            <span className="resource-bars__size">
              {formatResourceBytes(row.bytes, row.bytesKind)}
              {row.count > 1 ? ` x${row.count}` : ''}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function shortenResourceName(id: string): string {
  const name = id.includes('.') ? id.slice(id.lastIndexOf('.') + 1) : id;
  return name;
}
