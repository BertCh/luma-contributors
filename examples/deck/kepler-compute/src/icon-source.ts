// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Native Kepler svg-icons.json entries contain triangle meshes, not texture-atlas frames. */
export type NativeIconDefinition = {
  id: string;
  mesh: {positions: readonly (readonly number[])[]; cells: readonly (readonly number[])[]};
};
export type NativeIconCatalog = {
  meshes: ReadonlyMap<string, Float32Array>;
  typedStorageByteLength: number;
};
export type NativeIconLimits = {
  maximumIconCount?: number;
  maximumVertexCount?: number;
};
export type NativeIconSourceBatch = {
  id: string;
  icons: readonly unknown[];
  /** Optional upload-time coordinate validity; invalid rows are absent just as in native Icon. */
  validity?: ArrayLike<number>;
};
export type PreparedNativeIconGroup = {
  iconId: string;
  /** Already Y-flipped native unit geometry; never normalized, rescaled, or rasterized. */
  vertices: Float32Array;
  /** Global source-row indices, one chunk per original source batch including empty chunks. */
  sourceRowIds: readonly Uint32Array[];
  resolveSourceRow(primitiveIndex: number): number;
};
export type PreparedNativeIcons = {
  batches: readonly {id: string; sourceRowOffset: number; rowCount: number}[];
  groups: readonly PreparedNativeIconGroup[];
  rowCount: number;
  iconRowCount: number;
  /** Native Icon excludes non-string marker values; those rows retain their original identity. */
  omittedRowCount: number;
  typedStorageByteLength: number;
};

/** Native custom definitions override default IDs without changing their insertion order. */
export function prepareNativeIconCatalog(
  definitions: readonly NativeIconDefinition[],
  customDefinitions: readonly NativeIconDefinition[] = [],
  limits: NativeIconLimits = {}
): NativeIconCatalog {
  const definitionsById = new Map<string, NativeIconDefinition>();
  for (const definition of [...definitions, ...customDefinitions])
    definitionsById.set(definition.id, definition);
  const geometry: Record<string, Float32Array> = Object.create(null);
  let vertexCount = 0;
  for (const [id, definition] of definitionsById) {
    if (
      typeof id !== 'string' ||
      !id ||
      !Array.isArray(definition.mesh?.cells) ||
      !Array.isArray(definition.mesh?.positions)
    )
      throw new Error(`Invalid native icon mesh: ${String(id)}`);
    vertexCount += definition.mesh.cells.length * 3;
    if (
      vertexCount > (limits.maximumVertexCount ?? 1_000_000) ||
      definitionsById.size > (limits.maximumIconCount ?? 4096)
    )
      throw new Error('Native icon catalog exceeds its declared geometry limits');
    const vertices = new Float32Array(definition.mesh.cells.length * 9);
    let vertexOffset = 0;
    for (const cell of definition.mesh.cells) {
      if (cell.length !== 3) throw new Error(`Native icon ${id} must contain triangles`);
      for (const index of cell) {
        const position = definition.mesh.positions[index];
        if (
          !Number.isSafeInteger(index) ||
          index < 0 ||
          !position ||
          position.length !== 3 ||
          position.some(
            (value: unknown) => typeof value !== 'number' || !Number.isFinite(Math.fround(value))
          )
        )
          throw new Error(`Native icon ${id} has invalid triangle coordinates`);
        // This is exactly Kepler IconLayer.flatterIconPositions; native radius/billboard is applied later.
        vertices.set([position[0], -position[1], position[2]], vertexOffset);
        vertexOffset += 3;
      }
    }
    geometry[id] = vertices;
  }
  return prepareNativeIconGeometry(geometry, limits);
}

/** Accept a native layer's already flattened iconGeometry without applying a second Y flip. */
export function prepareNativeIconGeometry(
  geometry: object,
  limits: NativeIconLimits = {}
): NativeIconCatalog {
  const meshes = new Map<string, Float32Array>();
  let vertexCount = 0;
  for (const [id, value] of Object.entries(geometry)) {
    if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || !('length' in value))
      throw new Error(`Native icon ${id} has invalid triangle coordinates`);
    // The native boundary declares geometry as object; validate its arrays centrally here.
    const positions = value as ArrayLike<unknown>;
    vertexCount += positions.length / 3;
    if (
      meshes.size + 1 > (limits.maximumIconCount ?? 4096) ||
      vertexCount > (limits.maximumVertexCount ?? 1_000_000)
    )
      throw new Error('Native icon catalog exceeds its declared geometry limits');
    if (!id || positions.length === 0 || positions.length % 9 !== 0)
      throw new Error(`Native icon ${id} must contain nonempty triangle geometry`);
    const vertices = new Float32Array(positions.length);
    for (let index = 0; index < positions.length; index++) {
      const coordinate = positions[index];
      if (typeof coordinate !== 'number' || !Number.isFinite(Math.fround(coordinate)))
        throw new Error(`Native icon ${id} has invalid triangle coordinates`);
      vertices[index] = coordinate;
    }
    meshes.set(id, vertices);
  }
  return {
    meshes,
    typedStorageByteLength: [...meshes.values()].reduce(
      (total, vertices) => total + vertices.byteLength,
      0
    )
  };
}

/** CPU preparation runs once at the upload boundary; interaction consumes only GPU row maps. */
export function prepareNativeIconSource(
  batches: readonly NativeIconSourceBatch[],
  catalog: NativeIconCatalog
): PreparedNativeIcons {
  const groups = new Map<string, number[][]>();
  let rowCount = 0;
  let iconRowCount = 0;
  const sourceBatches = batches.map((batch, batchIndex) => {
    const sourceRowOffset = rowCount;
    if (batch.validity && batch.validity.length !== batch.icons.length)
      throw new Error('Native icon validity must match source rows');
    if (
      !Number.isSafeInteger(rowCount + batch.icons.length) ||
      rowCount + batch.icons.length > 0xffffffff
    )
      throw new Error('Native icon source exceeds uint32 row identity');
    batch.icons.forEach((value, rowIndex) => {
      if (typeof value !== 'string' || (batch.validity && batch.validity[rowIndex] === 0)) return;
      if (!catalog.meshes.has(value))
        throw new Error(
          `Native icon "${value}" at source row ${sourceRowOffset + rowIndex} is unavailable`
        );
      let sourceRows = groups.get(value);
      if (!sourceRows) {
        sourceRows = batches.map(() => []);
        groups.set(value, sourceRows);
      }
      sourceRows[batchIndex].push(sourceRowOffset + rowIndex);
      iconRowCount++;
    });
    rowCount += batch.icons.length;
    return {id: batch.id, sourceRowOffset, rowCount: batch.icons.length};
  });
  // Native SvgIconLayer enumerates an object: numeric IDs precede other IDs in numeric order.
  const orderedIds = Object.keys(Object.fromEntries([...groups.keys()].map(id => [id, true])));
  const preparedGroups = orderedIds.map(iconId => {
    const sourceRowIds = groups.get(iconId)!.map(values => Uint32Array.from(values));
    return {
      iconId,
      vertices: catalog.meshes.get(iconId)!,
      sourceRowIds,
      resolveSourceRow(primitiveIndex: number): number {
        if (!Number.isSafeInteger(primitiveIndex) || primitiveIndex < 0) return -1;
        for (const chunk of sourceRowIds) {
          if (primitiveIndex < chunk.length) return chunk[primitiveIndex];
          primitiveIndex -= chunk.length;
        }
        return -1;
      }
    };
  });
  return {
    batches: sourceBatches,
    groups: preparedGroups,
    rowCount,
    iconRowCount,
    omittedRowCount: rowCount - iconRowCount,
    typedStorageByteLength: preparedGroups.reduce(
      (total, group) =>
        total +
        group.vertices.byteLength +
        group.sourceRowIds.reduce((sum, chunk) => sum + chunk.byteLength, 0),
      0
    )
  };
}

/** Unlike native's console-only fallback, catalog fetch/parse failures reject visibly to the host. */
export async function loadNativeIconCatalog(options: {
  defaultUrl: string;
  customUrl?: string;
  customDefinitions?: readonly NativeIconDefinition[];
  limits?: NativeIconLimits;
  fetch?: typeof fetch;
}): Promise<NativeIconCatalog> {
  const fetchCatalog = async (url: string): Promise<NativeIconDefinition[]> => {
    const response = await (options.fetch ?? fetch)(url, {
      method: 'GET',
      mode: 'cors',
      cache: 'no-cache'
    });
    if (!response.ok)
      throw new Error(`Native icon catalog request failed (${response.status}): ${url}`);
    const parsed: unknown = await response.json();
    if (
      !parsed ||
      typeof parsed !== 'object' ||
      !('svgIcons' in parsed) ||
      !Array.isArray(parsed.svgIcons)
    )
      throw new Error(`Native icon catalog has no svgIcons array: ${url}`);
    return parsed.svgIcons;
  };
  const [defaults, custom] = await Promise.all([
    fetchCatalog(options.defaultUrl),
    options.customUrl ? fetchCatalog(options.customUrl) : []
  ]);
  return prepareNativeIconCatalog(
    defaults,
    [...custom, ...(options.customDefinitions ?? [])],
    options.limits
  );
}
