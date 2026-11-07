// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {GPUDataFrame} from '@luma-contributors/graph/gpu-dataframe';
import {GPUCommandGraph, type GraphDataView} from '@luma-contributors/graph/gpu-core';
import {GPUTable} from '@luma-contributors/graph/gpu-tables';
import {addResidentJoinPass} from './resident-join-gpu-kernels';
import {
  getResidentSqlCastConversionBody,
  getResidentSqlCastKernelBody,
  getResidentSqlValueLookupBody
} from './resident-sql-cast-kernel';
import {getResidentSqlNumericExpressionBody} from './resident-sql-expressions';
import type {ResidentSqlDerivedColumn, ResidentSqlScalarFormat} from './resident-sql-strings';

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;

/** A dataframe extended with derived GPU columns, plus the scratch it owns. */
export type ResidentSqlDerivedFrame = {
  readonly frame: GPUDataFrame;
  /** Bytes of GPU scratch allocated for the derived columns. */
  readonly scratchByteLength: number;
  /** Frees derived buffers; call once the compiled query no longer reads them. */
  destroy(): void;
};

/** Rows of GPU scratch the derived columns need, for memory admission before allocation. */
export function estimateResidentSqlDerivedByteLength(
  specs: readonly ResidentSqlDerivedColumn[],
  rowCount: number
): number {
  let bytes = 0;
  for (const spec of specs) {
    if (spec.kind === 'alias') continue;
    bytes += Math.max(4, rowCount * 4);
    if (spec.kind === 'cast' || spec.kind === 'expression' || spec.kind === 'pairLookup')
      bytes += Math.max(4, rowCount * 4);
    // Many-column lookups fold two columns per pass through one running-entry buffer each.
    if (spec.kind === 'pairLookup' && spec.extraColumns?.length)
      bytes += Math.ceil((spec.extraColumns.length + 2) / 2) * Math.max(4, rowCount * 4);
    if (spec.kind === 'lookup' || spec.kind === 'pairLookup')
      bytes += Math.max(4, spec.words.byteLength);
    else if (spec.kind === 'valueLookup')
      bytes += Math.max(4, spec.keys.byteLength) + Math.max(4, spec.codes.byteLength);
  }
  return bytes;
}

/** Wraps scalar resident vectors as a borrowed GPU dataframe with nullable flags set. */
export function assembleSqlFrame(
  vectors: Record<string, GPUVector>,
  validity: Record<string, GPUVector<'uint32'>>,
  dictionaries: Record<string, readonly unknown[]>
): GPUDataFrame {
  const table = new GPUTable({vectors});
  for (const schema of [table.schema, ...table.batches.map(batch => batch.schema)])
    for (const field of schema.fields) field.nullable = Boolean(validity[field.name]);
  return new GPUDataFrame({table, validity, dictionaries});
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

/**
 * Adds derived columns (dictionary lookups, aliases, casts, expressions) to a base frame. Each
 * derived column is computed per source chunk by one small compute pass, submitted immediately in
 * queue order, so chunk boundaries stay identical to the base table. Sources are only read.
 */
export function buildResidentSqlDerivedFrame(props: {
  device: Device;
  base: GPUDataFrame;
  specs: readonly ResidentSqlDerivedColumn[];
}): ResidentSqlDerivedFrame {
  const {device, base, specs} = props;
  const vectors: Record<string, GPUVector> = Object.create(null);
  for (const name of base.columnNames) vectors[name] = base.table.gpuVectors[name] as GPUVector;
  const validity: Record<string, GPUVector<'uint32'>> = Object.create(null);
  for (const [name, vector] of Object.entries(base.validity))
    if (vector) validity[name] = vector as GPUVector<'uint32'>;
  const dictionaries: Record<string, readonly unknown[]> = Object.create(null);
  for (const [name, values] of Object.entries(base.dictionaries))
    if (values) dictionaries[name] = values as readonly unknown[];

  const owned: Buffer[] = [];
  let scratchByteLength = 0;
  const graph = new GPUCommandGraph<undefined>(device, {id: 'resident-sql-derived'});
  const handles = new Map<Buffer, ReturnType<typeof graph.importBuffer>>();
  let passIndex = 0;
  const view = (buffer: Buffer, length: number, byteOffset: number): GraphDataView<'uint32'> => {
    let handle = handles.get(buffer);
    if (!handle) {
      handle = graph.importBuffer(
        {id: `derived-buffer-${handles.size}`, usage: buffer.usage, byteLength: buffer.byteLength},
        buffer
      );
      handles.set(buffer, handle);
    }
    return graph.createDataView(handle, {format: 'uint32', length, byteOffset});
  };
  const makeOutput = (label: string, length: number): Buffer => {
    const buffer = device.createBuffer({
      id: `sql-derived-${label}`,
      byteLength: Math.max(4, length * 4),
      usage: STORAGE_USAGE
    });
    owned.push(buffer);
    scratchByteLength += buffer.byteLength;
    return buffer;
  };
  const wrap = (
    name: string,
    format: ResidentSqlScalarFormat | 'uint32',
    buffers: Buffer[],
    lengths: number[]
  ) =>
    new GPUVector({
      name,
      type: 'data',
      format,
      ownsData: false,
      data: buffers.map(
        (buffer, index) => new GPUData({format, buffer, length: lengths[index], ownsBuffer: false})
      )
    }) as GPUVector;

  const makeTable = (name: string, words: Uint32Array): Buffer => {
    const buffer = device.createBuffer({
      id: `sql-derived-${name}-table`,
      data: words.length ? words : new Uint32Array(1),
      usage: STORAGE_USAGE
    });
    owned.push(buffer);
    scratchByteLength += buffer.byteLength;
    return buffer;
  };
  for (const spec of specs) {
    if (spec.kind === 'alias') {
      const source = vectors[spec.sourceColumn];
      if (!source)
        throw new Error(`Derived column source "${spec.sourceColumn}" is not on the GPU`);
      vectors[spec.name] = new GPUVector({
        name: spec.name,
        type: 'data',
        format: source.format,
        data: source.data,
        ownsData: false
      });
      const sourceValidity = validity[spec.sourceColumn];
      if (sourceValidity) validity[spec.name] = sourceValidity;
      const dictionary = dictionaries[spec.sourceColumn];
      if (dictionary) dictionaries[spec.name] = dictionary;
      continue;
    }
    const sourceNames =
      spec.kind === 'expression'
        ? spec.plan.columns.map(column => column.name)
        : spec.kind === 'pairLookup'
          ? [spec.sourceColumn, spec.secondColumn, ...(spec.extraColumns ?? [])]
          : [spec.sourceColumn];
    const sources = sourceNames.map(name => {
      const vector = vectors[name];
      if (!vector) throw new Error(`Derived column source "${name}" is not on the GPU`);
      return vector;
    });
    const sourceValidities = sourceNames.map(name => validity[name]);
    const lengths = sources[0].data.map(chunk => chunk.length);
    sources.forEach((source, sourceIndex) => {
      const batched =
        source.data.length === lengths.length &&
        source.data.every((chunk, chunkIndex) => chunk.length === lengths[chunkIndex]);
      const validityVector = sourceValidities[sourceIndex];
      if (!batched || (validityVector && validityVector.data.length !== lengths.length))
        throw new Error(`Column "${sourceNames[sourceIndex]}" is batched differently`);
    });
    const producesValidity =
      spec.kind === 'cast' ||
      spec.kind === 'expression' ||
      (spec.kind === 'pairLookup' && sourceValidities.some(Boolean));
    const outputs = lengths.map((length, chunkIndex) =>
      makeOutput(`${spec.name}-${chunkIndex}`, length)
    );
    const validityOutputs = producesValidity
      ? lengths.map((length, chunkIndex) =>
          makeOutput(`${spec.name}-validity-${chunkIndex}`, length)
        )
      : [];
    const tableBuffer =
      spec.kind === 'lookup' || spec.kind === 'pairLookup'
        ? makeTable(`${spec.name}`, spec.words)
        : undefined;
    const keysBuffer =
      spec.kind === 'valueLookup' ? makeTable(`${spec.name}-keys`, spec.keys) : undefined;
    const codesBuffer =
      spec.kind === 'valueLookup' ? makeTable(`${spec.name}-codes`, spec.codes) : undefined;
    lengths.forEach((length, chunkIndex) => {
      if (!length) return;
      const id = `sql-derived-${passIndex++}`;
      const sourceViews = sources.map(source => {
        const chunk = source.data[chunkIndex];
        return view(getBuffer(chunk), length, chunk.byteOffset);
      });
      const validityViews = sourceValidities.map(vector => {
        const chunk = vector?.data[chunkIndex];
        return chunk ? view(getBuffer(chunk), length, chunk.byteOffset) : undefined;
      });
      const output = {view: view(outputs[chunkIndex], length, 0), access: 'write' as const};
      const validityOut = producesValidity
        ? {view: view(validityOutputs[chunkIndex], length, 0), access: 'write' as const}
        : undefined;
      const read = (v: GraphDataView<'uint32'>) => ({view: v, access: 'read' as const});
      if (spec.kind === 'lookup') {
        addResidentJoinPass(graph, {
          id,
          count: length,
          constants: `const LOOKUP_COUNT: u32 = ${spec.words.length}u;`,
          bindings: {
            source: read(sourceViews[0]),
            lookupTable: read(view(tableBuffer!, Math.max(1, spec.words.length), 0)),
            output
          },
          body: `  let code = source[index];
  if (code < LOOKUP_COUNT) { output[index] = lookupTable[code]; } else { output[index] = 0u; }`
        });
      } else if (spec.kind === 'valueLookup') {
        addResidentJoinPass(graph, {
          id,
          count: length,
          constants: `const KEY_COUNT: u32 = ${spec.keys.length}u;`,
          bindings: {
            source: read(sourceViews[0]),
            keysTable: read(view(keysBuffer!, Math.max(1, spec.keys.length), 0)),
            codesTable: read(view(codesBuffer!, Math.max(1, spec.codes.length), 0)),
            output
          },
          body: getResidentSqlValueLookupBody(spec.sourceFormat)
        });
      } else if (spec.kind === 'pairLookup') {
        // Mixed radix over every column's dictionary code, last column fastest.
        const sizes = [spec.firstSize, spec.secondSize, ...(spec.extraSizes ?? [])];
        const validityCount = validityViews.filter(Boolean).length;
        const columnLines = (columnIndex: number, entryName: string, validName: string) =>
          `  let code${columnIndex} = source${columnIndex}[index];
${validityViews[columnIndex] ? `  if (validityIn${columnIndex}[index] == 0u) { ${validName} = 0u; }\n` : ''}  if (code${columnIndex} >= SIZE_${columnIndex}) { ${validName} = 0u; }
  ${entryName} = ${entryName} * SIZE_${columnIndex} + min(code${columnIndex}, SIZE_${columnIndex} - 1u);`;
        const sizeConstants = sizes.map((size, i) => `const SIZE_${i}: u32 = ${size}u;`).join('\n');
        const sourceBindings = (indices: number[]) => ({
          ...Object.fromEntries(indices.map(i => [`source${i}`, read(sourceViews[i])])),
          ...Object.fromEntries(
            indices.flatMap(i =>
              validityViews[i] ? [[`validityIn${i}`, read(validityViews[i]!)]] : []
            )
          )
        });
        if (sizes.length + validityCount + 3 <= 8) {
          // One pass fits the storage binding limit (sources, validities, table, output, validity).
          const indices = sizes.map((_, columnIndex) => columnIndex);
          addResidentJoinPass(graph, {
            id,
            count: length,
            constants: sizeConstants,
            bindings: {
              ...sourceBindings(indices),
              lookupTable: read(view(tableBuffer!, Math.max(1, spec.words.length), 0)),
              output,
              ...(validityOut ? {validityOut} : {})
            },
            body: `  var valid: u32 = 1u;
  var entry: u32 = 0u;
${indices.map(i => columnLines(i, 'entry', 'valid')).join('\n')}
  if (valid != 0u) { output[index] = lookupTable[entry]; } else { output[index] = 0u; }
${validityOut ? '  validityOut[index] = valid;' : ''}`
          });
        } else {
          // Many columns: fold two columns per pass into a running entry (0xffffffff = invalid),
          // then look the final entry up. Keeps every pass within eight storage bindings.
          let previous: Buffer | undefined;
          for (let first = 0; first < sizes.length; first += 2) {
            const indices = [first, first + 1].filter(i => i < sizes.length);
            const last = first + 2 >= sizes.length;
            const entryBuffer = last
              ? undefined
              : makeOutput(`${spec.name}-entry-${first}`, length);
            addResidentJoinPass(graph, {
              id: `${id}-fold-${first}`,
              count: length,
              constants: sizeConstants,
              bindings: {
                ...sourceBindings(indices),
                ...(previous ? {previousEntry: read(view(previous, length, 0))} : {}),
                ...(last
                  ? {
                      lookupTable: read(view(tableBuffer!, Math.max(1, spec.words.length), 0)),
                      output,
                      ...(validityOut ? {validityOut} : {})
                    }
                  : {entryOut: {view: view(entryBuffer!, length, 0), access: 'write' as const}})
              },
              body: `  var valid: u32 = 1u;
  var entry: u32 = 0u;
${previous ? '  entry = previousEntry[index];\n  if (entry == 0xffffffffu) { valid = 0u; entry = 0u; }' : ''}
${indices.map(i => columnLines(i, 'entry', 'valid')).join('\n')}
${
  last
    ? `  if (valid != 0u) { output[index] = lookupTable[entry]; } else { output[index] = 0u; }
${validityOut ? '  validityOut[index] = valid;' : ''}`
    : '  if (valid != 0u) { entryOut[index] = entry; } else { entryOut[index] = 0xffffffffu; }'
}`
            });
            previous = entryBuffer;
          }
        }
      } else if (spec.kind === 'expression') {
        const hasValidity = validityViews.map(Boolean);
        addResidentJoinPass(graph, {
          id,
          count: length,
          bindings: {
            ...Object.fromEntries(sourceViews.map((v, i) => [`source${i}`, read(v)])),
            ...Object.fromEntries(
              validityViews.flatMap((v, i) => (v ? [[`validityIn${i}`, read(v)]] : []))
            ),
            output,
            validityOut: validityOut!
          },
          body: `${getResidentSqlNumericExpressionBody(spec.plan, hasValidity)}
  var result: u32 = 0u;
${getResidentSqlCastConversionBody(spec.plan.resultFormat, spec.target)}
  output[index] = select(0u, result, valid != 0u);
  validityOut[index] = valid;`
        });
      } else {
        addResidentJoinPass(graph, {
          id,
          count: length,
          bindings: {
            source: read(sourceViews[0]),
            ...(validityViews[0] ? {validityIn: read(validityViews[0])} : {}),
            output,
            validityOut: validityOut!
          },
          body: getResidentSqlCastKernelBody(
            spec.sourceFormat,
            spec.target,
            Boolean(validityViews[0])
          )
        });
      }
    });
    const outputFormat: ResidentSqlScalarFormat =
      spec.kind === 'lookup' || spec.kind === 'pairLookup'
        ? spec.format
        : spec.kind === 'valueLookup'
          ? 'uint32'
          : spec.target === 'bool'
            ? 'uint32'
            : spec.target === 'date' || spec.target === 'timestamp'
              ? 'sint32'
              : spec.target;
    vectors[spec.name] = wrap(spec.name, outputFormat, outputs, lengths);
    if (producesValidity) {
      validity[spec.name] = wrap(
        `${spec.name}-validity`,
        'uint32',
        validityOutputs,
        lengths
      ) as GPUVector<'uint32'>;
    } else if (sourceValidities[0]) {
      validity[spec.name] = sourceValidities[0];
    }
    const labels = 'dictionary' in spec ? spec.dictionary : undefined;
    if (labels) dictionaries[spec.name] = labels;
  }

  const compiled = graph.compile();
  const encoder = device.createCommandEncoder({id: 'resident-sql-derived-encode'});
  compiled.encode(encoder, {parameters: undefined});
  device.submit(encoder.finish());

  let destroyed = false;
  return {
    frame: assembleSqlFrame(vectors, validity, dictionaries),
    scratchByteLength,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      compiled.destroy();
      for (const buffer of owned) buffer.destroy();
    }
  };
}
