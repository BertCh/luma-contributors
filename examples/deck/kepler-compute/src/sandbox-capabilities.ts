// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import inventory from './sandbox-capabilities.json';

export const SANDBOX_CAPABILITY_INVENTORY = inventory;
export const SANDBOX_CAPABILITY_STATUS_LABELS: Readonly<Record<string, string>> = {
  partial: 'Partial host integration',
  adapter: 'Reusable building blocks',
  planned: 'Needs integration',
  host: 'Host or service workflow',
  integrated: 'Integrated and verified'
};

/** This reviewed ledger describes integration work; it does not authorize runtime execution. */
export function filterSandboxCapabilities(query: string, group = '', status = '') {
  const terms = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  return inventory.capabilities.filter(capability => {
    if (group && capability.group !== group) return false;
    if (status && capability.status !== status) return false;
    const text = [
      capability.id,
      capability.title,
      capability.group,
      capability.foundation,
      capability.gap,
      capability.acceptance,
      SANDBOX_CAPABILITY_STATUS_LABELS[capability.status]
    ]
      .join(' ')
      .toLocaleLowerCase();
    return terms.every(term => text.includes(term));
  });
}
