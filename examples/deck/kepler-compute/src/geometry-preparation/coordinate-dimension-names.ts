// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Data,
  DataType,
  DenseUnion,
  Field,
  FixedSizeList,
  List,
  SparseUnion,
  Struct,
  Vector,
  type DataType as ArrowDataType
} from 'apache-arrow';

const COORDINATE_DIMENSION_NAMES = new Set(['xy', 'xyz', 'xym', 'xyzm']);

/**
 * Published @math.gl/arrow rejects an interleaved 3-wide coordinate list whose child field is not
 * named by its dimension ("Ambiguous three-component coordinates"). The old fork treated such
 * vectors as XYZ. This helper keeps that Kepler input contract: it renames the child of every
 * 3-wide interleaved coordinate list to the explicit `dimension` (default 'xyz') so the data
 * reaches the published adapter unambiguous. Storage buffers are shared, not copied.
 * Vectors without an unnamed 3-wide coordinate list are returned unchanged.
 */
export function nameKeplerCoordinateDimension<T extends ArrowDataType>(
  vector: Vector<T>,
  dimension?: string
): Vector<T> {
  const coordinateName = dimension && dimension.length === 3 ? dimension : 'xyz';
  let changed = false;
  const chunks = vector.data.map(data => {
    const renamed = renameCoordinateData(data, coordinateName);
    if (renamed !== data) changed = true;
    return renamed;
  });
  return changed ? (new Vector(chunks) as Vector<T>) : vector;
}

function renameCoordinateData(data: Data, coordinateName: string): Data {
  const type = data.type;
  if (DataType.isFixedSizeList(type)) {
    const child = type.children[0];
    if (
      type.listSize === 3 &&
      !COORDINATE_DIMENSION_NAMES.has(child.name.toLowerCase()) &&
      DataType.isFloat(child.type)
    ) {
      const renamedType = new FixedSizeList(
        3,
        new Field(coordinateName, child.type, child.nullable, child.metadata)
      );
      return data.clone(renamedType as ArrowDataType);
    }
  }
  if (data.children.length === 0) return data;
  const children = data.children.map(child => renameCoordinateData(child, coordinateName));
  if (children.every((child, index) => child === data.children[index])) return data;
  const fields = type.children.map(
    (field, index) => new Field(field.name, children[index].type, field.nullable, field.metadata)
  );
  let nextType: ArrowDataType | null = null;
  if (DataType.isList(type)) nextType = new List(fields[0]);
  else if (DataType.isFixedSizeList(type)) nextType = new FixedSizeList(type.listSize, fields[0]);
  else if (DataType.isStruct(type)) nextType = new Struct(fields);
  else if (DataType.isUnion(type))
    nextType =
      type.mode === 1
        ? new DenseUnion(Array.from(type.typeIds), fields)
        : new SparseUnion(Array.from(type.typeIds), fields);
  if (!nextType) return data;
  return data.clone(nextType, data.offset, data.length, data.nullCount, undefined, children);
}
