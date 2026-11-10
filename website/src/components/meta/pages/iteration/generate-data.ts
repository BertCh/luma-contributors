// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Writes the compact data of the iteration page from the recorded reachability run and the
// generated view models. Run from website/:
//   node --experimental-strip-types src/components/meta/pages/iteration/generate-data.ts
// Outputs (committed): reachability-search.json (the road network, per-node round stamps, final
// costs, the frontier series and the gate buffers) and gated-records.json (records ranked by
// gated kernels).

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {
  GPU_MAXIMUM_COMMAND_NODE_COUNT,
  validateCapacity,
  validateCommandNodeBudget
} from '@luma-contributors/analysis/analysis-core';
import {GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS} from '@luma-contributors/analysis/gpu-network';
import {fromRecordedLensFile} from '@luma-contributors/lenses/computation-lenses/registry';

const pageDirectory = path.dirname(fileURLToPath(import.meta.url));
const websiteDirectory = path.resolve(pageDirectory, '../../../../..');
const recordPath = path.resolve(
  websiteDirectory,
  '../docs/lens-records/gpu-network.network-reachability.json'
);
const viewModelDirectory = path.join(websiteDirectory, 'src/lens/generated');

type RecordedValues = {name: string; values: Record<string, number> | number[]};
type RecordedNode = {
  id: string;
  kind: string;
  reads: string[];
  writes: string[];
  workload?: {maximumWorkgroupCount?: number};
  condition?: {source: string; mode: string; resource: string};
};
type ReachabilityRecord = {
  graphs: {nodes: RecordedNode[]; attrs: Record<string, unknown>}[];
  captures: {snapshots: RecordedValues[]; stamps: (RecordedValues & {meaning: string})[]};
  metrics: {series: {name: string; values: number[]}[]}[];
};

const envelope = fromRecordedLensFile(JSON.parse(fs.readFileSync(recordPath, 'utf8')));
const record = envelope.record as unknown as ReachabilityRecord;

function toArray(values: RecordedValues['values']): number[] {
  return Array.isArray(values) ? values : Object.values(values);
}

function getSnapshot(name: string): number[] {
  const snapshot = record.captures.snapshots.find(candidate => candidate.name === name);
  if (!snapshot) throw new Error(`reachability record has no snapshot ${name}`);
  return toArray(snapshot.values);
}

const round3 = (value: number) => Math.round(value * 1000) / 1000;

// Road network: lon/lat positions projected with cos(latitude), normalised to 0..1, north up.
const lonLat = getSnapshot('nodePositions');
const nodeCount = lonLat.length / 2;
let minimumLatitude = Infinity;
let maximumLatitude = -Infinity;
for (let node = 0; node < nodeCount; node++) {
  minimumLatitude = Math.min(minimumLatitude, lonLat[node * 2 + 1]);
  maximumLatitude = Math.max(maximumLatitude, lonLat[node * 2 + 1]);
}
const longitudeScale = Math.cos((((minimumLatitude + maximumLatitude) / 2) * Math.PI) / 180);
const projected: [number, number][] = [];
for (let node = 0; node < nodeCount; node++) {
  projected.push([lonLat[node * 2] * longitudeScale, -lonLat[node * 2 + 1]]);
}
const minimumX = Math.min(...projected.map(point => point[0]));
const maximumX = Math.max(...projected.map(point => point[0]));
const minimumY = Math.min(...projected.map(point => point[1]));
const maximumY = Math.max(...projected.map(point => point[1]));
const positions = projected.flatMap(([x, y]) => [
  round3((x - minimumX) / (maximumX - minimumX)),
  round3((y - minimumY) / (maximumY - minimumY))
]);
const aspectRatio = round3((maximumX - minimumX) / (maximumY - minimumY));

// Undirected edges from the CSR, each pair once.
const offsets = getSnapshot('offsets');
const neighbors = getSnapshot('neighbors');
const edgeKeys = new Set<number>();
const edges: number[] = [];
for (let node = 0; node < nodeCount; node++) {
  for (let edge = offsets[node]; edge < offsets[node + 1]; edge++) {
    const neighbor = neighbors[edge];
    const low = Math.min(node, neighbor);
    const high = Math.max(node, neighbor);
    const key = low * nodeCount + high;
    if (low === high || edgeKeys.has(key)) continue;
    edgeKeys.add(key);
    edges.push(low, high);
  }
}

// Per-node stamp: the round whose queue the node last joined (see the stamp's recorded meaning).
const stampCapture = record.captures.stamps.find(stamp => stamp.name === 'lastImprovedRound');
if (!stampCapture) throw new Error('reachability record has no lastImprovedRound stamp');
const lastImprovedRound = toArray(stampCapture.values);

const costs = getSnapshot('costs').map(cost => (Number.isFinite(cost) ? Math.round(cost) : null));
const startNodes = costs.flatMap((cost, node) => (cost === 0 ? [node] : []));
const frontier = record.metrics[0]?.series.find(series => series.name === 'frontier-size');
if (!frontier) throw new Error('reachability record has no frontier-size series');

// The unrolled relax rounds and the dispatch buffer each one is gated on.
const graph = record.graphs[0];
const relaxNodes = graph.nodes.filter(node => /-relax-\d+$/.test(node.id));
const gateBuffers = [...new Set(relaxNodes.map(node => node.condition?.resource ?? ''))];
relaxNodes.forEach((node, round) => {
  const gate = node.condition;
  if (!node.id.endsWith(`-relax-${round}`)) throw new Error(`unexpected round order at ${node.id}`);
  if (gate?.source !== 'gpu' || gate.mode !== 'indirect')
    throw new Error(`${node.id} is not gated`);
  if (gate.resource !== gateBuffers[round % 2])
    throw new Error(`${node.id} breaks the alternation`);
  const written = gateBuffers[(round + 1) % 2];
  if (!node.writes.includes(written) || node.writes.includes(gate.resource)) {
    throw new Error(`${node.id} does not write the other dispatch buffer`);
  }
});
const seed = graph.nodes.find(node => node.id.endsWith('-seed'));

const search = {
  scenario: 'gpu-network.network-reachability',
  stampMeaning: stampCapture.meaning,
  nodeCount,
  directedEdgeCount: neighbors.length,
  aspectRatio,
  positions,
  edges,
  lastImprovedRound,
  costs,
  startNodes,
  frontier: frontier.values,
  maximumIterations: Number(graph.attrs.maximumIterations),
  localIterations: Number(graph.attrs.localIterations),
  relaxRoundCount: relaxNodes.length,
  relaxMaximumWorkgroups: relaxNodes[0]?.workload?.maximumWorkgroupCount ?? 0,
  converged: getSnapshot('converged')[0],
  iterationCount: getSnapshot('iterationCount')[0],
  gateBuffers,
  seedWrites: gateBuffers.filter(buffer => seed?.writes.includes(buffer)),
  graphNodeCount: graph.nodes.length,
  commandNodeBudget: GPU_MAXIMUM_COMMAND_NODE_COUNT,
  iterationsLimit: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  // The real construction-time messages, thrown by the published validators.
  iterationsMessage: getThrownMessage(() =>
    validateCapacity(
      'network-reachability',
      'maximumIterations',
      GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS * 2,
      1,
      GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS
    )
  ),
  budgetMessage: getThrownMessage(() =>
    validateCommandNodeBudget('network-cost-table', GPU_MAXIMUM_COMMAND_NODE_COUNT + 1)
  )
};

function getThrownMessage(validate: () => void): string {
  try {
    validate();
  } catch (error) {
    return (error as Error).message;
  }
  throw new Error('validator did not throw');
}

// Records ranked by GPU-gated kernels, from the generated view models.
type ViewModelLike = {
  id: string;
  title: string;
  gatedNodeCount?: number;
  nodes: {kind: string; count: number}[];
};
const gatedRecords: {id: string; title: string; gated: number; kernels: number}[] = [];
let recordCount = 0;
for (const file of fs.readdirSync(viewModelDirectory).sort()) {
  if (!file.endsWith('.json')) continue;
  recordCount++;
  const model = JSON.parse(
    fs.readFileSync(path.join(viewModelDirectory, file), 'utf8')
  ) as ViewModelLike;
  if (!model.gatedNodeCount) continue;
  const kernels = model.nodes
    .filter(node => node.kind === 'compute')
    .reduce((total, node) => total + node.count, 0);
  gatedRecords.push({id: model.id, title: model.title, gated: model.gatedNodeCount, kernels});
}
gatedRecords.sort((left, right) => right.gated - left.gated || left.id.localeCompare(right.id));
const gated = {
  recordCount,
  recordsWithGates: gatedRecords.length,
  top: gatedRecords.slice(0, 10),
  reachability: gatedRecords.find(entry => entry.id === search.scenario) ?? null,
  reachabilityRank: gatedRecords.findIndex(entry => entry.id === search.scenario) + 1
};

fs.writeFileSync(path.join(pageDirectory, 'reachability-search.json'), JSON.stringify(search));
fs.writeFileSync(path.join(pageDirectory, 'gated-records.json'), JSON.stringify(gated));
process.stdout.write(
  `reachability-search.json: ${nodeCount} nodes, ${edges.length / 2} edges, ` +
    `${frontier.values.length} rounds; gated-records.json: ${gatedRecords.length} records\n`
);
