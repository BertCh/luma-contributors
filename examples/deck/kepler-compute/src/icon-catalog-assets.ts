// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  prepareNativeIconCatalog,
  type NativeIconDefinition,
  type NativeIconLimits
} from './icon-source';

export const ICON_CATALOG_FORMAT = 'luma-kepler-icon-catalog';

/** Embeddable custom icon meshes: the same triangle-mesh definitions as native svg-icons.json. */
export type IconCatalogDocument = {
  format: typeof ICON_CATALOG_FORMAT;
  version: 1;
  icons: NativeIconDefinition[];
};

export const DEFAULT_ICON_CATALOG_LIMITS: Required<NativeIconLimits> & {
  maximumDocumentBytes: number;
} = {
  maximumIconCount: 512,
  maximumVertexCount: 200_000,
  maximumDocumentBytes: 8 * 1024 * 1024
};

/**
 * Validates definitions with the same mesh checks the renderer applies and returns a JSON-safe
 * document. Throws with the failing icon id so the host can surface it instead of silently
 * dropping custom icons.
 */
export function makeIconCatalogDocument(
  definitions: readonly NativeIconDefinition[],
  limits = DEFAULT_ICON_CATALOG_LIMITS
): IconCatalogDocument {
  for (const definition of definitions) {
    const id = definition?.id;
    if (
      typeof id !== 'string' ||
      !id ||
      !Array.isArray(definition?.mesh?.positions) ||
      !Array.isArray(definition?.mesh?.cells)
    )
      throw new Error(`Invalid custom icon mesh: ${String(id)}`);
    // JSON serialization would silently replace nonfinite unused coordinates with null. Validate
    // every retained position, including vertices not referenced by the rendered triangles.
    for (const position of definition.mesh.positions)
      if (
        !Array.isArray(position) ||
        position.length !== 3 ||
        position.some(value => typeof value !== 'number' || !Number.isFinite(Math.fround(value)))
      )
        throw new Error(`Custom icon ${id} has invalid coordinates`);
    for (const cell of definition.mesh.cells)
      if (!Array.isArray(cell) || cell.length !== 3)
        throw new Error(`Custom icon ${id} must contain triangles`);
  }
  // Reuses the renderer validation (triangles, finite Float32 coordinates, count limits).
  prepareNativeIconCatalog([], definitions, limits);
  const unique = new Map<string, NativeIconDefinition>();
  for (const definition of definitions)
    unique.set(definition.id, {
      id: definition.id,
      mesh: {
        positions: definition.mesh.positions.map(position => [...position]),
        cells: definition.mesh.cells.map(cell => [...cell])
      }
    });
  const document: IconCatalogDocument = {
    format: ICON_CATALOG_FORMAT,
    version: 1,
    icons: [...unique.values()]
  };
  assertDocumentSize(JSON.stringify(document), limits);
  return document;
}

/** Parses an embedded catalog; any malformed document or mesh rejects the whole catalog. */
export function parseIconCatalogDocument(
  value: unknown,
  limits = DEFAULT_ICON_CATALOG_LIMITS
): NativeIconDefinition[] {
  if (
    !value ||
    typeof value !== 'object' ||
    (value as {format?: unknown}).format !== ICON_CATALOG_FORMAT ||
    (value as {version?: unknown}).version !== 1 ||
    !Array.isArray((value as {icons?: unknown}).icons)
  )
    throw new Error('Unsupported custom icon catalog');
  const document = value as IconCatalogDocument;
  assertDocumentSize(JSON.stringify(document), limits);
  for (const definition of document.icons)
    if (!definition || typeof definition.id !== 'string' || !definition.id)
      throw new Error('Custom icon catalog entry has no id');
  return makeIconCatalogDocument(document.icons, limits).icons;
}

/**
 * Portable-catalog rule for saved bundles: only custom definitions travel with a workspace; the
 * default CDN catalog is never embedded. Returns undefined when there is nothing to embed.
 */
export function selectEmbeddedIconCatalog(
  customIcons: readonly NativeIconDefinition[] | undefined
): IconCatalogDocument | undefined {
  return customIcons?.length ? makeIconCatalogDocument(customIcons) : undefined;
}

function assertDocumentSize(serialized: string, limits: {maximumDocumentBytes: number}): void {
  if (new TextEncoder().encode(serialized).byteLength > limits.maximumDocumentBytes)
    throw new Error('Custom icon catalog exceeds its document budget');
}
