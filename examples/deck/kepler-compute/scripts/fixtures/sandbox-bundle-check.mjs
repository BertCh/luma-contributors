// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import * as arrow from 'apache-arrow';
import {serializeSandboxBundle, parseSandboxBundle} from '../../src/sandbox-bundle';
window.bundleChecks = (async () => {
  const original = new arrow.Table({
    'cost €': arrow.vectorFromArray([0.1, null, 30.25], new arrow.Float64()),
    region: arrow.vectorFromArray(['北', 'East', '北']),
    time: arrow.vectorFromArray([1700000000000.125, null, 1700000000000.75], new arrow.TimestampMicrosecond('UTC'))
  });
  const table = original.slice(0, 2).concat(original.slice(2));
  const result = new arrow.Table({total: arrow.vectorFromArray([30.35], new arrow.Float64())});
  const sources = [
    {datasetId: 'local source', revision: 3, table, isCurrent: () => true},
    {datasetId: 'SQL result', revision: 2, table: result, isCurrent: () => true}
  ];
  const state = {schemaVersion: 1, sources: sources.map(source => ({
    dataset: {datasetId: source.datasetId, revision: source.revision},
    reference: {kind: 'application', id: source.datasetId, version: 1, parameters: null},
    schema: source.table.schema.fields.map(field => ({name: field.name, type: field.type.toString(), nullable: field.nullable})),
    normalization: {float64: {mode: 'round', maximumAbsoluteError: 0.00001, maximumRelativeError: 0}}
  })), nativeConfig: {layers: [{dataId: 'local source'}, {dataId: 'SQL result'}], mapState: {mapSplitMode: 'SWIPE_COMPARE', swipeComparePercentage: 37.5, isViewportSynced: false, splitMapViewports: [{zoom: 11, maxBounds: [[null, -90], [null, 90]]}, {zoom: 12, maxBounds: [[null, -90], [null, 90]]}]}, splitMaps: [{layers: {'local source': true}}, {layers: {'SQL result': true}}]}, selections: [], analyses: []};
  const limits = {maximumDocumentBytes: 65536, maximumSourceBytes: 32768, maximumRows: 10, maximumSources: 2};
  const serialized = await serializeSandboxBundle(JSON.stringify(state), sources, limits);
  const restored = await parseSandboxBundle(serialized, limits);
  const read = restored.sources[0].table;
  const rejects = async operation => {try {await operation(); return false;} catch {return true;}};
  const corrupt = JSON.parse(serialized);
  corrupt.sources[0].data = `${corrupt.sources[0].data.startsWith('A') ? 'B' : 'A'}${corrupt.sources[0].data.slice(1)}`;
  const wrongSchema = structuredClone(state);
  wrongSchema.sources[0].schema[0].type = 'Float32';
  const duplicate = JSON.parse(serialized);
  duplicate.sources[1] = duplicate.sources[0];
  const wrongRevision = JSON.parse(serialized);
  wrongRevision.sources[0].revision++;
  const missingReference = structuredClone(state);
  missingReference.nativeConfig.layers.push({dataId: 'unsaved'});
  let currentChecks = 0;
  return {
    batches: read.batches.map(batch => batch.numRows),
    fieldNames: read.schema.fields.map(field => field.name),
    costs: [...read.getChild('cost €')],
    categories: [...read.getChild('region')],
    timeType: String(read.getChild('time').type),
    timeWords: read.getChild('time').data.flatMap(data => Array.from(data.values, value => value.toString())),
    originalTimeWords: table.getChild('time').data.flatMap(data => Array.from(data.values, value => value.toString())),
    stateMatches: JSON.stringify(restored.state) === JSON.stringify(state),
    sourceCount: restored.sources.length,
    corrupt: await rejects(() => parseSandboxBundle(JSON.stringify(corrupt), limits)),
    duplicate: await rejects(() => parseSandboxBundle(JSON.stringify(duplicate), limits)),
    wrongRevision: await rejects(() => parseSandboxBundle(JSON.stringify(wrongRevision), limits)),
    wrongSchema: await rejects(() => serializeSandboxBundle(JSON.stringify(wrongSchema), sources, limits)),
    missingReference: await rejects(() => serializeSandboxBundle(JSON.stringify(missingReference), sources, limits)),
    stale: await rejects(() => serializeSandboxBundle(JSON.stringify(state), [{...sources[0], isCurrent: () => ++currentChecks === 1}, sources[1]], limits)),
    documentLimit: await rejects(() => parseSandboxBundle(serialized, {...limits, maximumDocumentBytes: 200})),
    sourceLimit: await rejects(() => parseSandboxBundle(serialized, {...limits, maximumSourceBytes: 100})),
    rowLimit: await rejects(() => parseSandboxBundle(serialized, {...limits, maximumRows: 2})),
    countLimit: await rejects(() => parseSandboxBundle(serialized, {...limits, maximumSources: 1}))
  };
})();
