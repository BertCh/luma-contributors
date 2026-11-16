// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {profileKeplerWorkbenchData} from '../descriptors/workbench-adapters';
import {applyKeplerDatasetProfileDeclarations} from '../profile/dataset-profile';
import type {KeplerDatasetProfileDeclarations} from '../profile/dataset-profile-types';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import type {
  CatalogColumnType,
  CatalogDiagnostic,
  CatalogOutputTable
} from '../analysis/catalog-geojson';
import {makeWorkflowData} from './workflow-data';
import {parseWorkflowExpression, type WorkflowExpression} from './workflow-expression';
import {normalizeWorkflowParameters, parseWorkflowColumnList} from './workflow-parameters';
import {
  WorkflowError,
  type WorkflowData,
  type WorkflowOperation,
  type WorkflowOperationOutput,
  type WorkflowParameterSpec,
  type WorkflowPortKind
} from './workflow-types';

const MAXIMUM_ROWS = 100000;
const MAXIMUM_VERTICES = 2000000;
const ALL_KINDS: readonly WorkflowPortKind[] = ['point', 'line', 'polygon', 'features', 'table'];
type ColumnValue = string | number | boolean | null;
type Parameters = Readonly<Record<string, CatalogParameterValue>>;
type TableResult = {
  declarations?: KeplerDatasetProfileDeclarations;
  table: CatalogOutputTable;
  identityColumn?: string;
  diagnostics?: CatalogDiagnostic[];
};
type TableOperationDefinition = {
  id: string;
  title: string;
  inputNames?: readonly string[];
  parameters?: readonly WorkflowParameterSpec[];
  output: string;
  kind?: WorkflowPortKind;
  execute(inputs: Readonly<Record<string, WorkflowData>>, parameters: Parameters): TableResult;
};

function failParameter(title: string, parameter: string, label: string, problem: string): never {
  throw new WorkflowError('parameter-out-of-range', `${title}: ${label} ${problem}.`, {parameter});
}

function requireColumn(
  title: string,
  parameter: string,
  label: string,
  name: string,
  data: WorkflowData
): void {
  if (!Object.hasOwn(data.table.columns, name)) {
    failParameter(
      title,
      parameter,
      label,
      `names column ${JSON.stringify(name)}, which the input does not have. Columns: ${Object.keys(data.table.columns).slice(0, 12).join(', ')}`
    );
  }
}

function requireColumns(
  title: string,
  parameter: string,
  label: string,
  value: string,
  data: WorkflowData
): string[] {
  const names = [...new Set(parseWorkflowColumnList(value))];
  for (const name of names) requireColumn(title, parameter, label, name, data);
  return names;
}

function parseExpression(title: string, source: string, data?: WorkflowData): WorkflowExpression {
  let expression: WorkflowExpression;
  try {
    expression = parseWorkflowExpression(source);
  } catch (cause) {
    if (!(cause instanceof WorkflowError)) throw cause;
    throw new WorkflowError('parameter-out-of-range', `${title}: Expression ${cause.message}.`, {
      parameter: 'expression',
      cause
    });
  }
  if (data) {
    for (const name of expression.columns)
      requireColumn(title, 'expression', 'Expression', name, data);
  }
  return expression;
}

function parseRename(title: string, value: string): Map<string, string> {
  const renames = new Map<string, string>();
  if (!value.trim()) return renames;
  for (const pair of value.split(',')) {
    const parts = pair.split('=').map(part => part.trim());
    if (parts.length !== 2 || !parts[0] || !parts[1] || renames.has(parts[0])) {
      failParameter(
        title,
        'rename',
        'Rename',
        'must contain unique old=new pairs separated by commas'
      );
    }
    renames.set(parts[0], parts[1]);
  }
  return renames;
}

function copyTable(
  data: WorkflowData,
  indices?: readonly number[],
  names = Object.keys(data.table.columns)
): CatalogOutputTable {
  return {
    ...data.table,
    rowCount: indices?.length ?? data.table.rowCount,
    columns: Object.assign(
      Object.create(null) as CatalogOutputTable['columns'],
      Object.fromEntries(
        names.map(name => [
          name,
          indices
            ? indices.map(index => data.table.columns[name][index])
            : [...data.table.columns[name]]
        ])
      )
    ),
    columnTypes: Object.assign(
      Object.create(null) as CatalogOutputTable['columnTypes'],
      Object.fromEntries(names.map(name => [name, data.table.columnTypes[name]]))
    ),
    ...(data.table.geometryColumn !== undefined && !names.includes(data.table.geometryColumn)
      ? {geometryColumn: undefined}
      : {})
  };
}

function makeDiagnostic(
  code: string,
  count: number,
  message: string,
  severity: CatalogDiagnostic['severity'] = 'warning'
): CatalogDiagnostic[] {
  return count > 0 ? [{code, severity, count, message}] : [];
}

function evaluateRows(expression: WorkflowExpression, data: WorkflowData): ColumnValue[] {
  return Array.from({length: data.table.rowCount}, (_, index) =>
    expression.evaluate(
      Object.fromEntries(expression.columns.map(name => [name, data.table.columns[name][index]]))
    )
  );
}

type TableDeclarationSource = {
  data: WorkflowData;
  /** Output name to original name; absent means unchanged names. Only listed columns survive. */
  columns?: ReadonlyMap<string, string>;
};

/** Carry confirmed meanings through value-preserving copies, never old statistics or guesses. */
function getTableProfileDeclarations(
  table: CatalogOutputTable,
  sources: readonly TableDeclarationSource[],
  rowSources = sources.map(source => source.data)
): KeplerDatasetProfileDeclarations {
  const measurement: NonNullable<KeplerDatasetProfileDeclarations['measurement']> =
    Object.fromEntries(
      Object.keys(table.columns).flatMap(name => {
        const contributors = sources.flatMap(source => {
          const original = source.columns ? source.columns.get(name) : name;
          if (original === undefined || !Object.hasOwn(source.data.table.columns, original))
            return [];
          const column = source.data.analysisProfile?.columns.find(
            entry => entry.name === original
          );
          return [
            {
              compatible: source.data.table.columnTypes[original] === table.columnTypes[name],
              declared: column?.measurementSource === 'declared' ? column.measurement : undefined
            }
          ];
        });
        const declared = contributors[0]?.declared;
        return declared &&
          contributors.every(entry => entry.compatible && entry.declared === declared)
          ? [[name, declared]]
          : [];
      })
    );
  const rowSemantics = rowSources[0]?.analysisProfile?.rowSemantics;
  const geometrySources = rowSources.filter(source => source.table.geometryColumn !== undefined);
  const coordinates = geometrySources[0]?.analysisProfile?.coordinates;
  return {
    measurement,
    ...(rowSemantics?.source === 'declared' &&
    rowSources.every(
      source =>
        source.analysisProfile?.rowSemantics?.source === 'declared' &&
        source.analysisProfile.rowSemantics.value === rowSemantics.value
    )
      ? {rowSemantics: rowSemantics.value}
      : {}),
    ...(table.geometryColumn !== undefined &&
    coordinates?.source === 'declared' &&
    geometrySources.every(
      source =>
        source.analysisProfile?.coordinates?.source === 'declared' &&
        source.analysisProfile.coordinates.model === coordinates.model
    )
      ? {coordinateModel: coordinates.model}
      : {})
  };
}

function makeOperation(definition: TableOperationDefinition): WorkflowOperation {
  const parameters = definition.parameters ?? [];
  const inputNames = definition.inputNames ?? ['rows'];
  const operation: WorkflowOperation = {
    id: definition.id,
    title: definition.title,
    category: 'table',
    summary: `${definition.title} on workflow tables.`,
    origin: 'builtin',
    version: '2',
    engine: 'cpu-reference',
    inputs: inputNames.map(name => ({
      name,
      label: name[0].toUpperCase() + name.slice(1),
      accepts: ALL_KINDS,
      maximumRows: MAXIMUM_ROWS,
      maximumVertices: MAXIMUM_VERTICES
    })),
    parameters,
    outputs: [
      {
        name: definition.output,
        label: definition.title,
        kind: definition.kind ?? 'features',
        identityColumns: [],
        maximumRows: MAXIMUM_ROWS
      }
    ],
    capabilityIds: [],
    contributorNames: [],
    declaredBehavior: [
      'Copies input columns; preserves row order unless sorting or grouping.',
      ...(definition.id === 'table-concatenate'
        ? [
            'Merges geometry into the first geometry column; refuses geometry/attribute name collisions.',
            'Reserves source_table for provenance; rename an existing source_table before concatenating.',
            'Reconciles differing identities into a new workflow_identity column, with a numeric suffix on collision; rows without an identity use their input row index.'
          ]
        : [])
    ],
    limitation:
      'CPU reference execution; at most 100000 rows and 2000000 geometry vertices per input.',
    normalizeParameters(raw) {
      const normalized = normalizeWorkflowParameters(parameters, raw, {
        operationTitle: definition.title
      });
      if (Object.hasOwn(normalized, 'expression'))
        parseExpression(definition.title, normalized['expression'] as string);
      if (Object.hasOwn(normalized, 'columnName') && !(normalized['columnName'] as string).trim()) {
        failParameter(
          definition.title,
          'columnName',
          'Column name',
          'must contain 1 to 64 characters'
        );
      }
      if (Object.hasOwn(normalized, 'rename'))
        parseRename(definition.title, normalized['rename'] as string);
      return normalized;
    },
    async run(context, inputs, rawParameters): Promise<WorkflowOperationOutput> {
      if (context.signal.aborted)
        throw new WorkflowError('cancelled', `${definition.title}: Run was cancelled.`, {
          stepId: context.stepId
        });
      let inputRows = 0;
      for (const name of inputNames) {
        const input = inputs[name];
        if (!input)
          throw new WorkflowError(
            'input-required',
            `${definition.title}: ${name} requires an input.`,
            {port: name}
          );
        if (input.table.rowCount > MAXIMUM_ROWS)
          throw new WorkflowError(
            'capacity-exceeded',
            `${definition.title}: ${name} must have at most ${MAXIMUM_ROWS} rows; got ${input.table.rowCount}.`,
            {port: name}
          );
        inputRows += input.table.rowCount;
      }
      const result = definition.execute(inputs, operation.normalizeParameters(rawParameters));
      const table = {...result.table, id: definition.output, label: definition.title};
      const identityColumn = result.identityColumn;
      const data = makeWorkflowData(
        table,
        identityColumn !== undefined && Object.hasOwn(table.columns, identityColumn)
          ? {identityColumn}
          : {}
      );
      // Reprofile the actual output, then restore only declarations justified by its lineage.
      const declarations = result.declarations;
      if (
        Object.values(inputs).some(input => input.analysisProfile) ||
        declarations?.coordinateModel ||
        declarations?.rowSemantics ||
        Object.keys(declarations?.measurement ?? {}).length > 0
      ) {
        data.analysisProfile = applyKeplerDatasetProfileDeclarations(
          profileKeplerWorkbenchData(data),
          declarations ?? {}
        );
      }
      return {
        outputs: {[definition.output]: data},
        diagnostics: result.diagnostics ?? [],
        summary: {inputRows, outputRows: table.rowCount},
        engine: 'cpu-reference',
        dispatchedContributors: []
      };
    }
  };
  return operation;
}

function columnParameter(
  name: string,
  label: string,
  columnsFrom = 'rows',
  defaultValue = ''
): WorkflowParameterSpec {
  return {name, label, kind: 'column', columnsFrom, defaultValue};
}
function columnsParameter(
  name: string,
  label: string,
  columnsFrom = 'rows'
): WorkflowParameterSpec {
  return {name, label, kind: 'column-list', columnsFrom, defaultValue: ''};
}
function enumParameter(
  name: string,
  label: string,
  options: readonly string[],
  defaultValue = options[0]
): WorkflowParameterSpec {
  return {name, label, kind: 'enum', options, defaultValue};
}
const EXPRESSION_PARAMETER: WorkflowParameterSpec = {
  name: 'expression',
  label: 'Expression',
  kind: 'expression',
  columnsFrom: 'rows',
  defaultValue: ''
};

function joinAttributes(
  inputs: Readonly<Record<string, WorkflowData>>,
  parameters: Parameters
): TableResult {
  const title = 'Join attributes';
  const {target, attributes} = inputs;
  const targetKey = (parameters['targetKey'] as string) || target.identityColumn;
  if (targetKey === undefined)
    failParameter(
      title,
      'targetKey',
      'Target key',
      'requires a column because Target has no identity column'
    );
  const attributesKey = parameters['attributesKey'] as string;
  requireColumn(title, 'targetKey', 'Target key', targetKey, target);
  requireColumn(title, 'attributesKey', 'Attributes key', attributesKey, attributes);
  const chosen = parameters['columns']
    ? requireColumns(title, 'columns', 'Columns', parameters['columns'] as string, attributes)
    : Object.keys(attributes.table.columns).filter(
        name => name !== attributesKey && name !== attributes.table.geometryColumn
      );
  const names = chosen.filter(name => name !== attributes.table.geometryColumn);
  for (const name of names) {
    const added = `${parameters['prefix']}${name}`;
    if (Object.hasOwn(target.table.columns, added))
      failParameter(
        title,
        'prefix',
        'Prefix',
        `must make added column ${JSON.stringify(added)} unique in Target`
      );
  }
  const lookup = new Map<string, number>();
  let duplicates = 0;
  attributes.table.columns[attributesKey].forEach((value, index) => {
    if (value === null) return;
    const key = String(value);
    if (lookup.has(key)) duplicates++;
    else lookup.set(key, index);
  });
  const indices: number[] = [];
  const matches: Array<number | undefined> = [];
  let unmatched = 0;
  target.table.columns[targetKey].forEach((value, index) => {
    const match = value === null ? undefined : lookup.get(String(value));
    if (match === undefined) unmatched++;
    if (match !== undefined || parameters['unmatched'] === 'keep') {
      indices.push(index);
      matches.push(match);
    }
  });
  const table = copyTable(target, indices);
  for (const name of names) {
    const added = `${parameters['prefix']}${name}`;
    table.columns[added] = matches.map(match =>
      match === undefined ? null : attributes.table.columns[name][match]
    );
    table.columnTypes[added] = attributes.table.columnTypes[name];
  }
  return {
    table,
    declarations: getTableProfileDeclarations(
      table,
      [
        {data: target},
        {
          data: attributes,
          columns: new Map(names.map(name => [`${parameters['prefix']}${name}`, name]))
        }
      ],
      [target]
    ),
    identityColumn: target.identityColumn,
    diagnostics: [
      ...makeDiagnostic(
        'duplicate-attribute-key',
        duplicates,
        `${title}: ${duplicates} duplicate attribute keys; the first row wins.`
      ),
      ...makeDiagnostic(
        'unmatched-target-rows',
        unmatched,
        `${title}: ${unmatched} target rows have no matching attributes.`
      )
    ]
  };
}

function filterRows(
  inputs: Readonly<Record<string, WorkflowData>>,
  parameters: Parameters
): TableResult {
  const {rows} = inputs;
  const values = evaluateRows(
    parseExpression('Filter rows', parameters['expression'] as string, rows),
    rows
  );
  const indices = values.flatMap((value, index) => (value ? [index] : []));
  const nullCount = values.filter(value => value === null).length;
  const table = copyTable(rows, indices);
  return {
    table,
    declarations: getTableProfileDeclarations(table, [{data: rows}]),
    identityColumn: rows.identityColumn,
    diagnostics: makeDiagnostic(
      'null-condition',
      nullCount,
      `Filter rows: Expression evaluated to null for ${nullCount} rows.`,
      'info'
    )
  };
}

function calculateColumn(
  inputs: Readonly<Record<string, WorkflowData>>,
  parameters: Parameters
): TableResult {
  const {rows} = inputs;
  const name = parameters['columnName'] as string;
  if (name === rows.table.geometryColumn)
    failParameter(
      'Calculate column',
      'columnName',
      'Column name',
      'must not name the geometry column'
    );
  const values = evaluateRows(
    parseExpression('Calculate column', parameters['expression'] as string, rows),
    rows
  );
  const present = values.filter(value => value !== null);
  const type: CatalogColumnType = present.every(value => typeof value === 'number')
    ? 'float64'
    : present.every(value => typeof value === 'boolean')
      ? 'boolean'
      : 'string';
  const table = copyTable(rows);
  table.columns[name] = values.map(value =>
    type === 'string' && value !== null ? String(value) : value
  );
  table.columnTypes[name] = type;
  return {
    table,
    declarations: getTableProfileDeclarations(table, [
      {
        data: rows,
        columns: new Map(
          Object.keys(rows.table.columns)
            .filter(column => column !== name)
            .map(column => [column, column])
        )
      }
    ]),
    identityColumn: rows.identityColumn,
    diagnostics: Object.hasOwn(rows.table.columns, name)
      ? makeDiagnostic(
          'column-replaced',
          rows.table.rowCount,
          `Calculate column: Column ${JSON.stringify(name)} was replaced for ${rows.table.rowCount} rows.`,
          'info'
        )
      : []
  };
}

function selectColumns(
  inputs: Readonly<Record<string, WorkflowData>>,
  parameters: Parameters
): TableResult {
  const {rows} = inputs;
  const selected = new Set(
    requireColumns('Select columns', 'columns', 'Columns', parameters['columns'] as string, rows)
  );
  const renames = parseRename('Select columns', parameters['rename'] as string);
  for (const [oldName] of renames) {
    requireColumn('Select columns', 'rename', 'Rename', oldName, rows);
    if (oldName === rows.table.geometryColumn)
      failParameter('Select columns', 'rename', 'Rename', 'must not rename the geometry column');
  }
  const names = Object.keys(rows.table.columns).filter(
    name =>
      name === rows.table.geometryColumn ||
      (parameters['mode'] === 'keep' ? selected.has(name) : !selected.has(name))
  );
  const resulting = names.map(name => renames.get(name) ?? name);
  if (new Set(resulting).size !== resulting.length)
    failParameter('Select columns', 'rename', 'Rename', 'must produce unique column names');
  const table = copyTable(rows, undefined, names);
  table.columns = Object.fromEntries(
    names.map((name, index) => [resulting[index], table.columns[name]])
  );
  table.columnTypes = Object.fromEntries(
    names.map((name, index) => [resulting[index], table.columnTypes[name]])
  );
  return {
    table,
    declarations: getTableProfileDeclarations(table, [
      {data: rows, columns: new Map(names.map((name, index) => [resulting[index], name]))}
    ]),
    identityColumn:
      rows.identityColumn === undefined || !names.includes(rows.identityColumn)
        ? undefined
        : (renames.get(rows.identityColumn) ?? rows.identityColumn)
  };
}

function sortRows(
  inputs: Readonly<Record<string, WorkflowData>>,
  parameters: Parameters
): TableResult {
  const {rows} = inputs;
  const name = parameters['column'] as string;
  requireColumn('Sort rows', 'column', 'Column', name, rows);
  const values = rows.table.columns[name];
  const direction = parameters['order'] === 'ascending' ? 1 : -1;
  const indices = Array.from({length: rows.table.rowCount}, (_, index) => index);
  const isLast = (value: ColumnValue): boolean =>
    value === null || (typeof value === 'number' && !Number.isFinite(value));
  indices.sort((first, second) => {
    const firstValue = values[first];
    const secondValue = values[second];
    if (isLast(firstValue) || isLast(secondValue))
      return Number(isLast(firstValue)) - Number(isLast(secondValue)) || first - second;
    const firstComparable =
      rows.table.columnTypes[name] === 'string' ? String(firstValue) : firstValue!;
    const secondComparable =
      rows.table.columnTypes[name] === 'string' ? String(secondValue) : secondValue!;
    const comparison =
      firstComparable < secondComparable ? -1 : firstComparable > secondComparable ? 1 : 0;
    return comparison * direction || first - second;
  });
  const table = copyTable(
    rows,
    parameters['limit'] ? indices.slice(0, parameters['limit'] as number) : indices
  );
  return {
    table,
    declarations: getTableProfileDeclarations(table, [{data: rows}]),
    identityColumn: rows.identityColumn
  };
}

// Binary compensation retains low bits exactly and allows cancellation after partial overflow.
class CompensatedSum {
  private units = 0n;
  private readonly buffer = new DataView(new ArrayBuffer(8));

  add(value: number): void {
    this.buffer.setFloat64(0, value);
    const bits = this.buffer.getBigUint64(0);
    const exponent = Number((bits >> 52n) & 0x7ffn);
    const fraction = bits & 0xfffffffffffffn;
    const significand = exponent === 0 ? fraction : fraction | 0x10000000000000n;
    // Every finite double is an integer multiple of the smallest subnormal, 2^-1074.
    const units = significand << BigInt(Math.max(0, exponent - 1));
    this.units += bits >> 63n ? -units : units;
  }

  value(divisor = 1): number {
    const negative = this.units < 0n;
    const magnitude = negative ? -this.units : this.units;
    const denominator = BigInt(divisor);
    const shift = Math.max(0, (magnitude / denominator).toString(2).length - 53);
    const scaledDenominator = denominator << BigInt(shift);
    let significand = magnitude / scaledDenominator;
    const remainder = magnitude % scaledDenominator;
    // Round once to nearest, with ties to even, including means below the subnormal spacing.
    if (
      remainder * 2n > scaledDenominator ||
      (remainder * 2n === scaledDenominator && significand % 2n === 1n)
    )
      significand++;
    const result = Number(significand) * 2 ** (shift - 1074);
    return negative ? -result : result;
  }
}

function summarizeRows(
  inputs: Readonly<Record<string, WorkflowData>>,
  parameters: Parameters
): TableResult {
  const {rows} = inputs;
  const groupBy = requireColumns(
    'Summarise',
    'groupBy',
    'Group by',
    parameters['groupBy'] as string,
    rows
  );
  const valueColumn = parameters['valueColumn'] as string;
  if (valueColumn) {
    requireColumn('Summarise', 'valueColumn', 'Value column', valueColumn, rows);
    if (!['int32', 'float64'].includes(rows.table.columnTypes[valueColumn]))
      failParameter('Summarise', 'valueColumn', 'Value column', 'must name a numeric column');
  }
  const aggregateNames = [
    'row_count',
    ...(valueColumn
      ? ['value_count', 'value_sum', 'value_mean', 'value_minimum', 'value_maximum']
      : [])
  ];
  if (groupBy.some(name => aggregateNames.includes(name)))
    failParameter(
      'Summarise',
      'groupBy',
      'Group by',
      'must not use an aggregate output column name'
    );
  type Group = {
    values: ColumnValue[];
    count: number;
    valueCount: number;
    sum: CompensatedSum;
    minimum: number | null;
    maximum: number | null;
  };
  const groups = new Map<string, Group>();
  function getGroup(values: ColumnValue[]): Group {
    // Encode types and non-finite values separately so groups never collide with null or strings.
    const key = JSON.stringify(
      values.map(value => [
        typeof value,
        typeof value === 'number' ? String(value === 0 ? 0 : value) : value
      ])
    );
    let group = groups.get(key);
    if (!group) {
      group = {
        values,
        count: 0,
        valueCount: 0,
        sum: new CompensatedSum(),
        minimum: null,
        maximum: null
      };
      groups.set(key, group);
    }
    return group;
  }
  if (groupBy.length === 0) getGroup([]);
  for (let index = 0; index < rows.table.rowCount; index++) {
    const group = getGroup(groupBy.map(name => rows.table.columns[name][index]));
    group.count++;
    const value = valueColumn ? rows.table.columns[valueColumn][index] : null;
    if (typeof value === 'number' && Number.isFinite(value)) {
      group.valueCount++;
      group.sum.add(value);
      group.minimum = Math.min(group.minimum ?? value, value);
      group.maximum = Math.max(group.maximum ?? value, value);
    }
  }
  const table: CatalogOutputTable = {
    id: 'summary',
    label: 'Summarise',
    rowCount: groups.size,
    columns: Object.fromEntries([...groupBy, ...aggregateNames].map(name => [name, []])),
    columnTypes: Object.fromEntries([
      ...groupBy.map(name => [name, rows.table.columnTypes[name]]),
      ...aggregateNames.map(name => [name, name.endsWith('count') ? 'int32' : 'float64'])
    ])
  };
  const columns = table.columns as Record<string, ColumnValue[]>;
  for (const group of groups.values()) {
    groupBy.forEach((name, index) => columns[name].push(group.values[index]));
    columns['row_count'].push(group.count);
    if (valueColumn) {
      columns['value_count'].push(group.valueCount);
      columns['value_sum'].push(group.valueCount ? group.sum.value() : null);
      columns['value_mean'].push(group.valueCount ? group.sum.value(group.valueCount) : null);
      columns['value_minimum'].push(group.minimum);
      columns['value_maximum'].push(group.maximum);
    }
  }
  return {
    table,
    declarations: getTableProfileDeclarations(
      table,
      [{data: rows, columns: new Map(groupBy.map(name => [name, name]))}],
      []
    ),
    identityColumn:
      rows.identityColumn && groupBy.includes(rows.identityColumn) ? rows.identityColumn : undefined
  };
}

function concatenateRows(inputs: Readonly<Record<string, WorkflowData>>): TableResult {
  const {first, second} = inputs;
  const rowCount = first.table.rowCount + second.table.rowCount;
  if (rowCount > MAXIMUM_ROWS)
    throw new WorkflowError(
      'capacity-exceeded',
      `Concatenate: Combined rows must have at most ${MAXIMUM_ROWS} rows; got ${rowCount}.`
    );
  const geometryColumn = first.table.geometryColumn ?? second.table.geometryColumn;
  for (const [port, data] of [
    ['first', first],
    ['second', second]
  ] as const) {
    if (Object.hasOwn(data.table.columns, 'source_table')) {
      throw new WorkflowError(
        'incompatible-port',
        `Concatenate: ${port === 'first' ? 'First' : 'Second'} column "source_table" is reserved for output provenance; rename the column before concatenating.`,
        {port}
      );
    }
    if (
      geometryColumn !== undefined &&
      data.table.geometryColumn !== geometryColumn &&
      Object.hasOwn(data.table.columns, geometryColumn)
    ) {
      throw new WorkflowError(
        'incompatible-port',
        `Concatenate: ${port === 'first' ? 'First' : 'Second'} column ${JSON.stringify(geometryColumn)} is an attribute but is also the output geometry column; rename the attribute before concatenating.`,
        {port}
      );
    }
  }
  const names = [
    ...new Set(
      [first, second].flatMap(data =>
        Object.keys(data.table.columns).map(name =>
          name === data.table.geometryColumn ? geometryColumn! : name
        )
      )
    )
  ];
  names.push('source_table');
  const columns: CatalogOutputTable['columns'] = Object.create(null);
  const columnTypes: CatalogOutputTable['columnTypes'] = Object.create(null);
  for (const name of names) {
    if (name === 'source_table') {
      columns[name] = [
        ...Array<string>(first.table.rowCount).fill('first'),
        ...Array<string>(second.table.rowCount).fill('second')
      ];
      columnTypes[name] = 'string';
      continue;
    }
    const firstName =
      name === geometryColumn
        ? first.table.geometryColumn
        : name === first.table.geometryColumn
          ? undefined
          : name;
    const secondName =
      name === geometryColumn
        ? second.table.geometryColumn
        : name === second.table.geometryColumn
          ? undefined
          : name;
    const firstType =
      firstName !== undefined && Object.hasOwn(first.table.columnTypes, firstName)
        ? first.table.columnTypes[firstName]
        : undefined;
    const secondType =
      secondName !== undefined && Object.hasOwn(second.table.columnTypes, secondName)
        ? second.table.columnTypes[secondName]
        : undefined;
    const type =
      firstType && secondType && firstType !== secondType
        ? 'string'
        : (firstType ?? secondType ?? 'string');
    columnTypes[name] = type;
    columns[name] = [first, second].flatMap((data, index) => {
      const sourceName = index === 0 ? firstName : secondName;
      const values =
        sourceName !== undefined && Object.hasOwn(data.table.columns, sourceName)
          ? data.table.columns[sourceName]
          : Array<null>(data.table.rowCount).fill(null);
      return values.map(value => (value !== null && type === 'string' ? String(value) : value));
    });
  }
  let identityColumn = first.identityColumn;
  if (first.identityColumn !== second.identityColumn) {
    // A shared column name can be an unrelated attribute on the other input.
    identityColumn = 'workflow_identity';
    let suffix = 2;
    while (Object.hasOwn(columns, identityColumn)) identityColumn = `workflow_identity_${suffix++}`;
    columns[identityColumn] = [first, second].flatMap(data =>
      Array.from({length: data.table.rowCount}, (_, index) => {
        const value =
          data.identityColumn === undefined
            ? index
            : data.table.columns[data.identityColumn][index];
        return value === null ? null : String(value);
      })
    );
    columnTypes[identityColumn] = 'string';
  }
  const table: CatalogOutputTable = {
    id: 'combined',
    label: 'Concatenate',
    rowCount,
    columns,
    columnTypes,
    ...(geometryColumn === undefined ? {} : {geometryColumn})
  };
  return {
    table,
    declarations: getTableProfileDeclarations(
      table,
      [first, second].map(data => ({
        data,
        columns: new Map(
          Object.keys(data.table.columns).map(name => [
            name === data.table.geometryColumn ? geometryColumn! : name,
            name
          ])
        )
      }))
    ),
    identityColumn
  };
}

function pointsFromColumns(
  inputs: Readonly<Record<string, WorkflowData>>,
  parameters: Parameters
): TableResult {
  const {rows} = inputs;
  const longitudeColumn = parameters['longitudeColumn'] as string;
  const latitudeColumn = parameters['latitudeColumn'] as string;
  for (const [parameter, label, name] of [
    ['longitudeColumn', 'Longitude column', longitudeColumn],
    ['latitudeColumn', 'Latitude column', latitudeColumn]
  ]) {
    requireColumn('Points from columns', parameter, label, name, rows);
    if (!['int32', 'float64'].includes(rows.table.columnTypes[name]))
      failParameter('Points from columns', parameter, label, 'must name a numeric column');
  }
  if (longitudeColumn === latitudeColumn)
    failParameter(
      'Points from columns',
      'latitudeColumn',
      'Latitude column',
      'must differ from Longitude column'
    );
  const indices: number[] = [];
  const geometry: string[] = [];
  for (let index = 0; index < rows.table.rowCount; index++) {
    const longitude = rows.table.columns[longitudeColumn][index];
    const latitude = rows.table.columns[latitudeColumn][index];
    if (
      typeof longitude !== 'number' ||
      !Number.isFinite(longitude) ||
      longitude < -180 ||
      longitude > 180 ||
      typeof latitude !== 'number' ||
      !Number.isFinite(latitude) ||
      latitude < -90 ||
      latitude > 90
    )
      continue;
    indices.push(index);
    geometry.push(JSON.stringify({type: 'Point', coordinates: [longitude, latitude]}));
  }
  const table = copyTable(
    rows,
    indices,
    Object.keys(rows.table.columns).filter(
      name => name !== rows.table.geometryColumn || name === 'geometry'
    )
  );
  table.columns['geometry'] = geometry;
  table.columnTypes['geometry'] = 'string';
  table.geometryColumn = 'geometry';
  const invalid = rows.table.rowCount - indices.length;
  return {
    table,
    declarations: {
      ...getTableProfileDeclarations(table, [
        {
          data: rows,
          columns: new Map(
            Object.keys(rows.table.columns)
              .filter(name => name !== 'geometry')
              .map(name => [name, name])
          )
        }
      ]),
      coordinateModel: 'geographic'
    },
    identityColumn: rows.identityColumn,
    diagnostics: makeDiagnostic(
      'invalid-coordinate',
      invalid,
      `Points from columns: ${invalid} rows have null, non-finite or out-of-range coordinates.`
    )
  };
}

/** Glue operations on tables: join, filter, calculate, select, sort, summarise, concatenate. */
export function getWorkflowTableOperations(): readonly WorkflowOperation[] {
  return [
    makeOperation({
      id: 'table-join-attributes',
      title: 'Join attributes',
      inputNames: ['target', 'attributes'],
      parameters: [
        columnParameter('targetKey', 'Target key', 'target'),
        columnParameter('attributesKey', 'Attributes key', 'attributes', 'source_id'),
        columnsParameter('columns', 'Columns', 'attributes'),
        {name: 'prefix', label: 'Prefix', kind: 'string', defaultValue: ''},
        enumParameter('unmatched', 'Unmatched', ['keep', 'drop'])
      ],
      output: 'joined',
      execute: joinAttributes
    }),
    makeOperation({
      id: 'table-filter-rows',
      title: 'Filter rows',
      parameters: [EXPRESSION_PARAMETER],
      output: 'filtered',
      execute: filterRows
    }),
    makeOperation({
      id: 'table-calculate-column',
      title: 'Calculate column',
      parameters: [
        {
          name: 'columnName',
          label: 'Column name',
          kind: 'string',
          maximumLength: 64,
          defaultValue: ''
        },
        EXPRESSION_PARAMETER
      ],
      output: 'calculated',
      execute: calculateColumn
    }),
    makeOperation({
      id: 'table-select-columns',
      title: 'Select columns',
      parameters: [
        columnsParameter('columns', 'Columns'),
        enumParameter('mode', 'Mode', ['keep', 'drop']),
        {name: 'rename', label: 'Rename', kind: 'string', maximumLength: 512, defaultValue: ''}
      ],
      output: 'selected',
      execute: selectColumns
    }),
    makeOperation({
      id: 'table-sort-rows',
      title: 'Sort rows',
      parameters: [
        columnParameter('column', 'Column'),
        enumParameter('order', 'Order', ['ascending', 'descending']),
        {
          name: 'limit',
          label: 'Limit',
          kind: 'integer',
          minimum: 0,
          maximum: MAXIMUM_ROWS,
          defaultValue: 0
        }
      ],
      output: 'sorted',
      execute: sortRows
    }),
    makeOperation({
      id: 'table-summarize',
      title: 'Summarise',
      parameters: [
        columnsParameter('groupBy', 'Group by'),
        columnParameter('valueColumn', 'Value column')
      ],
      output: 'summary',
      kind: 'table',
      execute: summarizeRows
    }),
    makeOperation({
      id: 'table-concatenate',
      title: 'Concatenate',
      inputNames: ['first', 'second'],
      output: 'combined',
      execute: concatenateRows
    }),
    makeOperation({
      id: 'table-points-from-columns',
      title: 'Points from columns',
      parameters: [
        columnParameter('longitudeColumn', 'Longitude column'),
        columnParameter('latitudeColumn', 'Latitude column')
      ],
      output: 'points',
      kind: 'point',
      execute: pointsFromColumns
    }),
    makeOperation({
      id: 'table-drop-geometry',
      title: 'Drop geometry',
      output: 'table',
      kind: 'table',
      execute: ({rows}) => {
        const table = copyTable(
          rows,
          undefined,
          Object.keys(rows.table.columns).filter(name => name !== rows.table.geometryColumn)
        );
        return {
          table,
          declarations: getTableProfileDeclarations(table, [{data: rows}]),
          identityColumn: rows.identityColumn
        };
      }
    })
  ];
}
