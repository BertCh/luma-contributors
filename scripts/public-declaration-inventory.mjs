// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * One public identity is (package entry, export spelling, defining declaration). Source consumers
 * use a resolved file#declaration origin; runtime consumers use the actual binding value, which
 * preserves aliases without conflating distinct functions that happen to have the same name.
 * This module deliberately has no Node or compiler dependency: audits also run in the browser.
 *
 * @template Declaration
 * @param {Iterable<{entryPoint: string, sourceTarget?: string, exports: Iterable<[string, Declaration]>}>} entries
 * @returns {{entryPoint: string, exportName: string, declaration: Declaration, sourceTarget?: string}[]}
 */
export function createPublicDeclarationInventory(entries) {
  return [...entries].flatMap(({entryPoint, sourceTarget, exports}) =>
    [...exports].map(([exportName, declaration]) => ({
      entryPoint,
      exportName,
      declaration,
      sourceTarget
    }))
  );
}

/**
 * Keep qualified identities in storage. A consumer may coalesce aliases by declaration, never
 * merely by export spelling. In particular two entries can export different same-name packers.
 * @template Declaration
 * @param {{entryPoint: string, exportName: string, declaration: Declaration}[]} inventory
 * @returns {Map<string, Map<string, Set<Declaration>>>}
 */
export function groupPublicEntries(inventory) {
  const entries = new Map();
  for (const {entryPoint, exportName, declaration} of inventory) {
    if (!entries.has(entryPoint)) entries.set(entryPoint, new Map());
    const declarations = entries.get(entryPoint).get(exportName) ?? new Set();
    declarations.add(declaration);
    entries.get(entryPoint).set(exportName, declarations);
  }
  return entries;
}
