// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Columns align only within the same role and declared key space. Unkeyed columns each have
 * their own domain. Groups and columns retain first-occurrence order from the supplied panels.
 */
export function groupTableColumns<Panel extends {role: string; capture: {keySpace?: string}}>(
  panels: readonly Panel[]
): Panel[][] {
  const groups: Panel[][] = [];
  const roles = new Map<string, Map<string, Panel[]>>();
  for (const panel of panels) {
    const keySpace = panel.capture.keySpace;
    if (!keySpace) {
      groups.push([panel]);
      continue;
    }
    let domains = roles.get(panel.role);
    if (!domains) {
      domains = new Map();
      roles.set(panel.role, domains);
    }
    let columns = domains.get(keySpace);
    if (!columns) {
      columns = [];
      domains.set(keySpace, columns);
      groups.push(columns);
    }
    columns.push(panel);
  }
  return groups;
}
