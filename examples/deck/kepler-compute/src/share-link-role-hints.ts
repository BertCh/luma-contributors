// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerRoleInput} from './auto-prepare-plan';

/** Roles a share link can pin; other recorded roles are reported, never silently ignored. */
const APPLICABLE_ROLES = ['longitude', 'latitude', 'geometry', 'h3'] as const;

const pendingHints = new Map<string, Readonly<Record<string, string>>>();

/** Remembers link roles for a file about to enter native Kepler; the label equals the file name. */
export function registerLinkedRoleHint(
  fileName: string,
  roles: Readonly<Record<string, string>> | undefined
): void {
  if (roles && Object.keys(roles).length) pendingHints.set(fileName, roles);
}

/** Consumed once by automatic preparation when that dataset arrives. */
export function takeLinkedRoleHint(label: string): Readonly<Record<string, string>> | undefined {
  const hints = pendingHints.get(label);
  pendingHints.delete(label);
  return hints;
}

/**
 * Turns link roles into one preferred synthetic layer for the auto-prepare plan, so the linked roles
 * override whatever columns Kepler guessed. Returns notes for roles that cannot be applied.
 */
export function makeLayerInputsFromLinkedRoles(
  roles: Readonly<Record<string, string>>,
  fieldNames: readonly string[]
): {layers: LayerRoleInput[]; notes: string[]} {
  const notes: string[] = [];
  const layers: LayerRoleInput[] = [];
  const lookup = (role: string): string | undefined => {
    const column = roles[role];
    if (column === undefined) return undefined;
    if (fieldNames.includes(column)) return column;
    notes.push(`Linked ${role} column "${column}" was not found in the data`);
    return undefined;
  };
  for (const role of Object.keys(roles))
    if (!(APPLICABLE_ROLES as readonly string[]).includes(role))
      notes.push(`Linked role "${role}" has no automatic-preparation meaning and was not applied`);
  const longitude = lookup('longitude');
  const latitude = lookup('latitude');
  const geometry = lookup('geometry');
  const h3 = lookup('h3');
  if (geometry)
    layers.push({
      id: 'linked-geometry',
      type: 'geojson',
      preferred: true,
      columns: {geojson: geometry}
    });
  else if (h3)
    layers.push({id: 'linked-h3', type: 'hexagonId', preferred: true, columns: {hex_id: h3}});
  else if (longitude && latitude)
    layers.push({
      id: 'linked-points',
      type: 'point',
      preferred: true,
      columns: {lng: longitude, lat: latitude}
    });
  else if (longitude || latitude)
    notes.push('Linked longitude and latitude must be given together');
  return {layers, notes};
}
