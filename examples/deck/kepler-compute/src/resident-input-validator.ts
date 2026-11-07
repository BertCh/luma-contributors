// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {validateInputData} from '@kepler.gl/utils';
import type * as arrow from 'apache-arrow';

const EXPLICIT_ARROW_FIELDS = Symbol('resident-explicit-arrow-fields');
type NativeInput = Parameters<typeof validateInputData>[0];

/** Only publication after Arrow preparation opts into this schema-preserving ingestion path. */
export function markResidentArrowInput<Input extends NativeInput & {arrowTable: arrow.Table}>(
  input: Input
): Input & {[EXPLICIT_ARROW_FIELDS]: true} {
  return {...input, [EXPLICIT_ARROW_FIELDS]: true};
}

/** Pinned native validation otherwise re-infers every column when any analyzerType is absent. */
export function createResidentInputValidator(
  nativeValidator: typeof validateInputData
): typeof validateInputData {
  return data => {
    if (
      !data ||
      typeof data !== 'object' ||
      !(EXPLICIT_ARROW_FIELDS in data) ||
      data[EXPLICIT_ARROW_FIELDS] !== true
    )
      return nativeValidator(data);
    const table = data.arrowTable;
    const columns = data.cols;
    if (
      !table ||
      !columns ||
      columns.length !== data.fields.length ||
      table.schema.fields.length !== data.fields.length ||
      data.fields.some(
        (field, index) => !field.type || field.name !== table.schema.fields[index].name
      )
    )
      throw new Error('Resident Arrow field metadata does not match its source schema');
    return {
      ...data,
      fields: data.fields.map((field, fieldIndex) => ({
        ...field,
        id: field.name,
        displayName: field.displayName || field.name,
        fieldIdx: fieldIndex,
        type: field.type!,
        format: field.format ?? '',
        analyzerType: field.analyzerType || getAnalyzerType(field.type!),
        valueAccessor: (row: {index: number}) => columns[fieldIndex].get(row.index)
      }))
    };
  };
}

function getAnalyzerType(type: string): string {
  switch (type) {
    case 'integer':
      return 'INT';
    case 'real':
      return 'FLOAT';
    case 'timestamp':
      return 'DATETIME';
    case 'date':
      return 'DATE';
    case 'boolean':
      return 'BOOLEAN';
    case 'geojson':
      return 'GEOMETRY_FROM_STRING';
    case 'geoarrow':
      return 'GEOMETRY';
    case 'h3':
      return 'H3';
    default:
      return 'STRING';
  }
}
