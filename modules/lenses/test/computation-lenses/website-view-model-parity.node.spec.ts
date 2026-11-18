// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync, statSync} from 'node:fs';
import {expect, test} from 'vitest';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {extractLensViewModel} from '../../../../website/src/lens/extract-lens-view-model';
import {getResourceBytesKind} from '../../../../website/src/lens/format';
import type {LensSummary, LensViewModel} from '../../../../website/src/lens/types';
import {
  LOCAL_MORAN_RECORD,
  WORKFLOWS
} from '../../../../website/src/components/meta/pages/workflows/workflows';

const recordsDirectory = new URL('../../../../docs/lens-records/', import.meta.url);
const modelsDirectory = new URL('../../../../website/src/lens/generated/', import.meta.url);
const recordFiles = readdirSync(recordsDirectory)
  .filter(file => file.endsWith('.json'))
  .sort();

function readRecord(file: string) {
  return fromRecordedLensFile(JSON.parse(readFileSync(new URL(file, recordsDirectory), 'utf8')))
    .record;
}

function getModelComparison(model: LensViewModel) {
  return {
    recordedNodes: model.recordedNodeCount,
    recordedEdges: model.recordedEdgeCount,
    nodes: model.nodes,
    edges: model.edges,
    map: model.map,
    totalBytesKind: model.resources.totalBytesKind
  };
}

type SummaryComparisonInput = Pick<
  LensSummary,
  'recordCount' | 'packs' | 'bufferBytesKind' | 'recordBytes' | 'viewModelBytes'
>;

function getSummaryComparison(summary: SummaryComparisonInput) {
  return {
    recordCount: summary.recordCount,
    packs: summary.packs,
    bufferBytesKind: summary.bufferBytesKind,
    recordBytes: summary.recordBytes,
    viewModelBytes: summary.viewModelBytes
  };
}

function extractSummaryComparison(): SummaryComparisonInput {
  const packs: Record<string, number> = {};
  let bufferBytesKind: LensSummary['bufferBytesKind'] = 'logical';
  let recordBytes = 0;
  let viewModelBytes = 0;
  for (const file of recordFiles) {
    const record = readRecord(file);
    recordBytes += statSync(new URL(file, recordsDirectory)).size;
    viewModelBytes += statSync(new URL(file, modelsDirectory)).size;
    const graph =
      record.graphs.find(candidate => candidate.form === 'compiled') ?? record.graphs[0];
    const pack = file.slice(0, file.indexOf('.'));
    packs[pack] = (packs[pack] ?? 0) + 1;
    bufferBytesKind = getResourceBytesKind([
      {bytesKind: bufferBytesKind},
      ...(graph.resources ?? [])
    ]);
  }
  return {recordCount: recordFiles.length, packs, bufferBytesKind, recordBytes, viewModelBytes};
}

test('website lens/generated scenario membership matches the committed records exactly', () => {
  expect(recordFiles.length).toBeGreaterThan(0);
  const modelFiles = readdirSync(modelsDirectory)
    .filter(file => file.endsWith('.json'))
    .sort();
  expect(modelFiles).toEqual(recordFiles);
  for (const file of modelFiles) {
    const model: LensViewModel = JSON.parse(readFileSync(new URL(file, modelsDirectory), 'utf8'));
    expect(model.id, file).toBe(file.slice(0, -5));
  }
});

test('X3 segment map samples only the 1,802 published points and aligned kinds', () => {
  const id = 'gpu-spatial-analysis.segment-intersection';
  const record = readRecord(`${id}.json`);
  const points = record.captures!.snapshots.find(capture => capture.name === 'intersections')!;
  const count = record.captures!.snapshots.find(capture => capture.name === 'count')!;
  expect(count.values[0]).toBe(1802);
  const actual = extractLensViewModel(id, record).map!;
  expect(actual.layers.find(layer => layer.name === 'intersections')!.positions).toHaveLength(
    901 * 2
  );
  const geometry = points.geometry;
  if (geometry?.kind !== 'points') throw new Error('intersections must declare point geometry');
  const values = record.captures!.snapshots.find(
    capture => capture.name === geometry.valuesCapture
  )!;
  points.values = Array.from(points.values).slice(0, 1802 * 2);
  values.values = Array.from(values.values).slice(0, 1802);
  expect(extractLensViewModel(id, record).map).toEqual(actual);
});

test('X3 zero count and finite spare-slot outliers cannot affect the map', () => {
  const id = 'gpu-spatial-analysis.segment-intersection';
  const record = readRecord(`${id}.json`);
  const points = record.captures!.snapshots.find(capture => capture.name === 'intersections')!;
  const count = record.captures!.snapshots.find(capture => capture.name === 'count')!;
  const expected = extractLensViewModel(id, record).map;
  const coordinates = Array.from(points.values);
  coordinates[1802 * 2] = 1e20;
  coordinates[1802 * 2 + 1] = -1e20;
  points.values = coordinates;
  expect(extractLensViewModel(id, record).map).toEqual(expected);
  count.values = [0];
  expect(
    extractLensViewModel(id, record).map!.layers.find(layer => layer.name === 'intersections')!
      .positions
  ).toEqual([]);
});

test('website lens/generated graphs, maps and byte precision match fresh extraction', () => {
  const stale: unknown[] = [];
  for (const file of recordFiles) {
    const expected = extractLensViewModel(file.slice(0, -5), readRecord(file));
    const modelPath = new URL(file, modelsDirectory);
    const actual: LensViewModel | undefined = existsSync(modelPath)
      ? JSON.parse(readFileSync(modelPath, 'utf8'))
      : undefined;
    const counts = (model: LensViewModel) => ({
      nodes: model.recordedNodeCount,
      edges: model.recordedEdgeCount,
      displayedNodes: model.nodes.length,
      displayedEdges: model.edges.length,
      pointMarks: model.map?.layers.map(layer => [layer.name, layer.positions.length / 2])
    });
    if (
      !actual ||
      JSON.stringify(getModelComparison(actual)) !== JSON.stringify(getModelComparison(expected))
    ) {
      stale.push({
        scenarioId: expected.id,
        shipped: actual ? counts(actual) : 'missing',
        extracted: counts(expected)
      });
    }
  }
  expect(
    stale,
    `${stale.length} missing or stale website graphs/maps; regenerate-view-models`
  ).toEqual([]);
});

test('website lens summary counts and byte precision match fresh generation', () => {
  const summary = JSON.parse(
    readFileSync(new URL('../../../../website/src/lens/summary.json', import.meta.url), 'utf8')
  );
  expect(getSummaryComparison(summary)).toEqual(getSummaryComparison(extractSummaryComparison()));
});

test('R3 generated model parity rejects the lower_bound byte-precision mutation', () => {
  const expected = extractLensViewModel(
    'gpu-spatial-analysis.local-moran',
    readRecord('gpu-spatial-analysis.local-moran.json')
  );
  expect(expected.resources.totalBytesKind).toBe('lower-bound');
  const malformed = JSON.parse(JSON.stringify(expected));
  malformed.resources.totalBytesKind = 'lower_bound';
  expect(getModelComparison(malformed)).not.toEqual(getModelComparison(expected));
});

test('R3 generated summary parity rejects the lower_bound byte-precision mutation', () => {
  const expected = extractSummaryComparison();
  expect(expected.bufferBytesKind).toBe('lower-bound');
  const malformed = JSON.parse(JSON.stringify(expected));
  malformed.bufferBytesKind = 'lower_bound';
  expect(getSummaryComparison(malformed)).not.toEqual(getSummaryComparison(expected));
});

test('R3 displayed workflow byte precision matches fresh extraction', () => {
  const records = [LOCAL_MORAN_RECORD, ...WORKFLOWS.map(workflow => workflow.record)];
  for (const actual of records) {
    const expected = extractLensViewModel(actual.scenario, readRecord(`${actual.scenario}.json`));
    expect(actual.bytesKind, actual.scenario).toBe(expected.resources.totalBytesKind);
  }
});

test('nearest weights recording captures candidate loss and emits from neighbor rows', () => {
  const record = readRecord('gpu-spatial-analysis.nearest-feature-weights.json');
  expect(record.captures?.snapshots.map(capture => capture.name)).toContain('candidateOverflow');
  const graph = record.graphs.find(candidate => candidate.form === 'compiled')!;
  const emit = graph.nodes.find(node => node.id === 'nearest-feature-weights-emit')!;
  expect(emit.reads).toContain('neighborRows');
  expect(emit.reads).not.toContain('neighborIds');
});

function readLensesPageData() {
  return JSON.parse(
    readFileSync(
      new URL(
        '../../../../website/src/components/meta/pages/lenses/lenses-data.json',
        import.meta.url
      ),
      'utf8'
    )
  ) as {
    bins: {label: string; count: number}[];
    median: {id: string; kernels: number};
    featured: {id: string; recordBytes: number; viewModelBytes: number}[];
  };
}

test('X1 lens page histogram accounts for every shipped model', () => {
  expect(readLensesPageData().bins.reduce((total, bin) => total + bin.count, 0)).toBe(
    recordFiles.length
  );
});

test('X1 lens page median scenario and compute count match shipped models', () => {
  const counts = recordFiles
    .map(file => {
      const model: LensViewModel = JSON.parse(readFileSync(new URL(file, modelsDirectory), 'utf8'));
      return {
        id: file.slice(0, -5),
        kernels: model.nodes
          .filter(node => node.kind === 'compute')
          .reduce((total, node) => total + node.count, 0)
      };
    })
    .sort((left, right) => left.kernels - right.kernels);
  expect(readLensesPageData().median).toEqual(counts[Math.floor(counts.length / 2)]);
});

test('X1 featured record and model byte sizes match the shipped files', () => {
  const featured = readLensesPageData().featured;
  expect(featured.map(entry => entry.id)).toEqual([
    'gpu-spatial-analysis.kernel-density',
    'gpu-spatial-analysis.local-moran',
    'gpu-network.network-service-areas'
  ]);
  for (const entry of featured) {
    expect(entry.recordBytes, entry.id).toBe(
      statSync(new URL(`${entry.id}.json`, recordsDirectory)).size
    );
    expect(entry.viewModelBytes, entry.id).toBe(
      statSync(new URL(`${entry.id}.json`, modelsDirectory)).size
    );
  }
});
