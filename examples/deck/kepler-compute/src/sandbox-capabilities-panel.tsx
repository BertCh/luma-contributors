// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useId, useState} from 'react';
import {
  filterSandboxCapabilities,
  SANDBOX_CAPABILITY_INVENTORY,
  SANDBOX_CAPABILITY_STATUS_LABELS
} from './sandbox-capabilities';
import './sandbox-capabilities.css';

const groups = [...new Set(SANDBOX_CAPABILITY_INVENTORY.capabilities.map(item => item.group))];

/** Read-only coverage guide: no buffer access, execution flags or implicit backend changes. */
export function SandboxCapabilities() {
  const identifier = useId();
  const [query, setQuery] = useState('');
  const [group, setGroup] = useState('');
  const [status, setStatus] = useState('');
  const capabilities = filterSandboxCapabilities(query, group, status);
  return (
    <details className="sandbox-capabilities" data-testid="sandbox-capabilities">
      <summary>Explore feature coverage</summary>
      <p>
        Kepler {SANDBOX_CAPABILITY_INVENTORY.keplerVersion} · reviewed{' '}
        {SANDBOX_CAPABILITY_INVENTORY.reviewDate}. This guide distinguishes connected controls from
        reusable contributors and remaining work. It is not a live compatibility check.
      </p>
      <label htmlFor={`${identifier}-search`}>Find a feature or contributor</label>
      <input
        id={`${identifier}-search`}
        type="search"
        placeholder="Try raster, export, or weights"
        value={query}
        onChange={event => setQuery(event.target.value)}
      />
      <label htmlFor={`${identifier}-group`}>Workflow</label>
      <select
        id={`${identifier}-group`}
        value={group}
        onChange={event => setGroup(event.target.value)}
      >
        <option value="">All workflows</option>
        {groups.map(name => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
      <label htmlFor={`${identifier}-status`}>Integration stage</label>
      <select
        id={`${identifier}-status`}
        value={status}
        onChange={event => setStatus(event.target.value)}
      >
        <option value="">All stages</option>
        {Object.entries(SANDBOX_CAPABILITY_STATUS_LABELS).map(([value, label]) => (
          <option key={value} value={value}>
            {label}
          </option>
        ))}
      </select>
      <p role="status">
        {capabilities.length} of {SANDBOX_CAPABILITY_INVENTORY.capabilities.length} features
      </p>
      <div className="sandbox-capabilities-results">
        {capabilities.map(capability => (
          <details key={capability.id}>
            <summary>{capability.title}</summary>
            <p className="sandbox-capabilities-stage">
              {capability.group} · {SANDBOX_CAPABILITY_STATUS_LABELS[capability.status]}
            </p>
            <p>
              <strong>Foundation:</strong> {capability.foundation}
            </p>
            <p>
              <strong>Remaining work:</strong> {capability.gap}
            </p>
            <p>
              <strong>Acceptance:</strong> {capability.acceptance}
            </p>
          </details>
        ))}
        {capabilities.length === 0 && (
          <p>No matching features. Try a broader search or clear the filters.</p>
        )}
      </div>
    </details>
  );
}
