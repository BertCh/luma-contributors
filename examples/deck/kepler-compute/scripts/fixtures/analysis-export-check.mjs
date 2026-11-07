// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import * as arrow from 'apache-arrow';
import {serializeAnalysisExport} from '../../src/analysis-export';
window.analysisExportChecks = (async () => {
  const data = {datasetId: 'uploaded', datasetRevision: 2, selectionRevision: 7, analysisRevision: '4:7',
    recipe: {operation: {kind: 'excess-risk', events: 'cases', populations: 'population'}, population: 'all'},
    mode: 'page', offset: 5, selectedCount: 12, sourceRowIndices: new Uint32Array([10, 13, 99]),
    columns: {score: new Float32Array([0.25, 99, NaN]), class: new Uint32Array([1, 2, 3]), __luma_source_row_index: new Int32Array([-1, 2, 3])},
    validity: {score: new Uint32Array([1, 0, 1])}};
  const options = {format: 'arrow', maximumOutputBytes: 65536, assertCurrent() {}};
  const output = await serializeAnalysisExport(data, options);
  const table = arrow.tableFromIPC(output.bytes);
  const json = JSON.parse(new TextDecoder().decode((await serializeAnalysisExport(data, {...options, format: 'json'})).bytes));
  const csv = await serializeAnalysisExport(data, {...options, format: 'csv'});
  const empty = arrow.tableFromIPC((await serializeAnalysisExport({...data, sourceRowIndices: new Uint32Array(), columns: {score: new Float32Array()}, validity: {score: new Uint32Array()}}, options)).bytes);
  const original = new arrow.Table({
    score: arrow.makeVector(Float64Array.from({length: 100}, (_, row) => 0.123456789012345 + row)),
    score_analysis: arrow.makeVector(new Int32Array(100).fill(12)),
    __luma_source_row_index_: arrow.makeVector(Uint32Array.from({length: 100}, (_, row) => row + 100)),
    instant: arrow.makeVector(arrow.makeData({type: new arrow.TimestampNanosecond('UTC'), data: BigInt64Array.from({length: 100}, (_, row) => 1700000000123456789n + BigInt(row))})),
    label: arrow.vectorFromArray(Array.from({length: 100}, (_, row) => row % 2 ? 'odd' : 'even'))
  });
  const source = original.slice(0, 12).concat(original.slice(12, 50), original.slice(50));
  const combined = await serializeAnalysisExport(data, {...options, source});
  const joined = arrow.tableFromIPC(combined.bytes);
  const combinedCSV = await serializeAnalysisExport(data, {...options, source, format: 'csv'});
  const rejects = async operation => {try {await operation(); return false;} catch {return true;}};
  const controller = new AbortController(); setTimeout(() => controller.abort(), 0);
  return {combined: {
    sourceValues: [...joined.getChild('score')], sourceLabels: [...joined.getChild('label')],
    sourceWords: joined.getChild('instant').data.flatMap(chunk => Array.from(chunk.values, String)),
    derivedValues: [...joined.getChild('score_analysis_analysis')].map(value => Number.isNaN(value) ? 'NaN' : value),
    sourceRowColumn: combined.sourceRowColumn, indices: [...joined.getChild(combined.sourceRowColumn)],
    mapping: combined.derivedColumns, metadata: JSON.parse(joined.schema.metadata.get('luma:analysis-export')),
    csv: new TextDecoder().decode(combinedCSV.bytes), csvMetadata: JSON.parse(combinedCSV.metadata),
    sourceBatches: source.batches.map(batch => batch.numRows), outputBatches: joined.batches.map(batch => batch.numRows)
  }, identities: [...table.getChild(output.sourceRowColumn)], sourceRowColumn: output.sourceRowColumn,
    score: [...table.getChild('score')].map(value => Number.isNaN(value) ? 'NaN' : value),
    fields: table.schema.fields.map(field => ({name: field.name, type: field.type.toString()})),
    metadata: JSON.parse(table.schema.metadata.get('luma:analysis-export')), json,
    csv: new TextDecoder().decode(csv.bytes), csvMetadata: JSON.parse(csv.metadata), emptyRows: empty.numRows,
    invalid: await rejects(() => serializeAnalysisExport({...data, validity: {score: new Uint32Array(2)}}, options)),
    oversized: await rejects(() => serializeAnalysisExport(data, {...options, maximumOutputBytes: 16})),
    stale: await rejects(() => serializeAnalysisExport(data, {...options, assertCurrent() {throw new Error('stale');}})),
    canceled: await rejects(() => serializeAnalysisExport({...data, sourceRowIndices: new Uint32Array(1024), columns: {score: new Float32Array(1024)}, validity: {}}, {...options, format: 'json', assertCurrent: () => controller.signal.throwIfAborted()}))};
})();
