// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatBytes} from '../../lens/format';
import type {LensViewModel} from '../../lens/types';
import './meta.css';

/** The largest buffers of a recorded run, as proportional bars. */
export function ResourceBars({model, limit = 6}: {model: LensViewModel; limit?: number}) {
  const rows = model.resources.largest.slice(0, limit);
  const maximum = Math.max(1, ...rows.map(row => row.bytes));
  return (
    <div className="meta-panel">
      <div className="meta-panel__bar">
        <span className="meta-panel__title">{model.title}: largest buffers</span>
        <ul className="meta-stats">
          <li>
            <strong>{formatBytes(model.resources.totalBytes)}</strong> in {model.resources.count}{' '}
            buffers
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
              {formatBytes(row.bytes)}
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
