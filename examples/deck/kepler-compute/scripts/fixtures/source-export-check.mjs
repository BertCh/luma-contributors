// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import * as arrow from 'apache-arrow';
import {serializeSelectedSource} from '../../src/source-export';
window.sourceExportChecks = (async () => {
  const words = BigInt64Array.from([0n, -1n, 1700000000123456789n, 123n, 1700000000987654321n, 999n]);
  const original = new arrow.Table({
    amount: arrow.vectorFromArray([0, 0.123456789012345, null, 3, 4.25, 5], new arrow.Float64()),
    label: arrow.vectorFromArray(['zero', 'comma, quote"', 'line\nbreak', 'three', '北', 'five']),
    instant: arrow.makeVector(arrow.makeData({type: new arrow.TimestampNanosecond('UTC'), data: words})),
    geometry: arrow.vectorFromArray(Array.from({length: 6}, (_, index) => ({type: 'Point', coordinates: [-74 + index, 40]}))),
    __luma_source_row_index: arrow.makeVector(new Uint32Array([10, 11, 12, 13, 14, 15]))
  });
  original.schema.metadata.set('example', 'preserve source metadata');
  const source = original.slice(0, 3).concat(original.slice(3));
  const identities = new Uint32Array([1, 2, 4]);
  const metadata = {datasetId: 'original', datasetRevision: 2, selectionRevision: 4, selectedCount: 3, offset: 0};
  const options = {format: 'arrow', maximumOutputBytes: 65536, assertCurrent() {}};
  const exported = await serializeSelectedSource(source, identities, metadata, options);
  const restored = arrow.tableFromIPC(exported.bytes);
  const csv = await serializeSelectedSource(source, identities, metadata, {...options, format: 'csv'});
  const separateDictionaries = arrow.tableFromArrays({region: ['north', 'south']}).concat(arrow.tableFromArrays({region: ['west', 'north']}));
  const dictionaryExport = arrow.tableFromIPC((await serializeSelectedSource(separateDictionaries, new Uint32Array([1, 2]), {...metadata, selectedCount: 2}, options)).bytes);
  const empty = arrow.tableFromIPC((await serializeSelectedSource(source, new Uint32Array(), metadata, options)).bytes);
  const rejects = async operation => {try {await operation(); return false;} catch {return true;}};
  const large = new arrow.Table({value: arrow.makeVector(new Uint32Array(1024))});
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 0);
  return {
    separateDictionaryValues: [...dictionaryExport.getChild('region')],
    amount: [...restored.getChild('amount')],
    labels: [...restored.getChild('label')],
    words: restored.getChild('instant').data.flatMap(data => Array.from(data.values, String)),
    geometry: [...restored.getChild('geometry')].map(value => ({type: value.type, coordinates: Array.from(value.coordinates)})),
    indices: [...restored.getChild(exported.sourceRowColumn)],
    originalIndexColumn: [...restored.getChild('__luma_source_row_index')],
    sourceRowColumn: exported.sourceRowColumn,
    batches: restored.batches.map(batch => batch.numRows),
    metadata: restored.schema.metadata.get('example'),
    exportMetadata: JSON.parse(restored.schema.metadata.get('luma:source-export')),
    csv: new TextDecoder().decode(csv.bytes),
    empty: {rows: empty.numRows, fields: empty.schema.fields.map(field => field.name)},
    oversized: await rejects(() => serializeSelectedSource(source, identities, metadata, {...options, maximumOutputBytes: 64})),
    invalid: await rejects(() => serializeSelectedSource(source, new Uint32Array([2, 1]), metadata, options)),
    stale: await rejects(() => serializeSelectedSource(source, identities, metadata, {...options, assertCurrent() {throw new Error('stale');}})),
    canceled: await rejects(() => serializeSelectedSource(large, Uint32Array.from({length: 512}, (_, index) => index * 2), metadata, {...options, maximumOutputBytes: 1024 * 1024, assertCurrent: () => controller.signal.throwIfAborted()}))
  };
})();
