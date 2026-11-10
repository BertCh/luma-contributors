// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Build-time only (called by generate-view-models.ts). Turns a decoded lens record into the compact
// LensViewModel. Records are decoded by @luma-contributors/lenses (fromRecordedLensFile) first.

import {deflateSync, crc32} from 'node:zlib';
import {sampleValueRamp} from './ramp.ts';
import type {
  LensMap,
  LensNodeKind,
  LensParameter,
  LensPointLayer,
  LensRaster,
  LensResourceLifetime,
  LensResourceSummary,
  LensSeries,
  LensViewEdge,
  LensViewModel,
  LensViewNode
} from './types.ts';

// Minimal structural view of the record; the full types live in @luma-contributors/analysis.
type RecordNode = {
  id: string;
  label: string;
  kind: string;
  op?: string;
  reads: readonly string[];
  writes: readonly string[];
  dispatch?: {workgroups: readonly [number, number, number]};
  workload?: {
    variant?: string;
    maximumWorkgroupCount?: number;
    maximumInvocationCount?: number;
    readByteLength?: number;
    writeByteLength?: number;
  };
  condition?: {source?: string; mode?: string};
};
type RecordGraph = {
  id: string;
  form: string;
  nodes: readonly RecordNode[];
  edges: readonly {from: string; to: string; kind?: string; resource?: string}[];
  resources?: readonly {
    id: string;
    bytes: number;
    ownership: 'imported' | 'transient';
    allocation?: string;
    allocationBytes?: number;
  }[];
};
type RecordCapture = {
  id: string;
  name: string;
  role?: string;
  keySpace?: string;
  elementType: string;
  components?: number;
  values: ArrayLike<number>;
  geometry?: {
    kind: string;
    shape?: readonly [number, number];
    origin?: readonly [number, number];
    cellSize?: number;
    rowDirection?: string;
    offsetsCapture?: string;
    valuesCapture?: string;
  };
};
export type LensRecordLike = {
  runId: string;
  adapter?: {backend?: string; description?: string};
  graphs: readonly RecordGraph[];
  choices?: readonly {name: string; label?: string; value?: unknown; unit?: string}[];
  refs?: {formulas?: readonly {id: string; ascii?: string}[]};
  captures?: {snapshots: readonly RecordCapture[]};
  metrics?: readonly {
    graph: string;
    timebase: 'ns' | 'ms';
    nodes: Record<string, {gpuBegin?: number; gpuEnd?: number}>;
    series?: readonly {name: string; unit?: string; values: ArrayLike<number>}[];
  }[];
};

const MAXIMUM_NODES = 30;
const MAXIMUM_RASTER_CELLS = 2000;
const MAXIMUM_POINTS = 1200;
const MAXIMUM_SERIES_VALUES = 200;
const MAXIMUM_ROUND_VALUES = 512;
/** A record that lists more formulas than this carries its whole pack's set, not its own. */
const PACK_FORMULA_COUNT = 8;

export function extractLensViewModel(scenarioId: string, record: LensRecordLike): LensViewModel {
  const graph = record.graphs.find(candidate => candidate.form === 'compiled') ?? record.graphs[0];
  const resourceById = new Map((graph.resources ?? []).map(resource => [resource.id, resource]));
  const timings = readTimings(record, graph.id);
  const grouping = groupNodes(graph.nodes, scenarioId.slice(scenarioId.indexOf('.') + 1));

  const nodes: LensViewNode[] = grouping.groups.map(group => {
    const members = group.members;
    const operators = [...new Set(members.map(member => member.op).filter(Boolean))].slice(0, 3);
    let milliseconds: number | undefined;
    for (const member of members) {
      const value = timings.get(member.id);
      if (value !== undefined) milliseconds = (milliseconds ?? 0) + value;
    }
    return {
      id: group.id,
      label: group.label,
      kind: getNodeKind(members),
      count: members.length,
      operators: operators as string[],
      workgroups: sum(members, member => getWorkgroups(member)),
      readBytes: sum(members, member => member.workload?.readByteLength ?? 0),
      writeBytes: sum(members, member => member.workload?.writeByteLength ?? 0),
      ...(milliseconds === undefined ? {} : {milliseconds}),
      ...describeMembers(members, grouping.collapsed)
    };
  });

  const order = new Map(nodes.map((node, index) => [node.id, index]));
  const edgeByKey = new Map<
    string,
    {from: string; to: string; resources: Set<string>; hazards: Set<'raw' | 'war' | 'waw'>}
  >();
  for (const edge of graph.edges) {
    const from = grouping.groupOfNode.get(edge.from);
    const to = grouping.groupOfNode.get(edge.to);
    if (from === undefined || to === undefined || from === to) continue;
    // Groups are ordered by first member, so keeping forward edges keeps the quotient graph acyclic.
    if ((order.get(from) ?? 0) >= (order.get(to) ?? 0)) continue;
    const key = `${from}\u0000${to}`;
    const entry = edgeByKey.get(key) ?? {
      from,
      to,
      resources: new Set<string>(),
      hazards: new Set<'raw' | 'war' | 'waw'>()
    };
    if (edge.resource !== undefined) entry.resources.add(edge.resource);
    if (edge.kind === 'raw' || edge.kind === 'war' || edge.kind === 'waw')
      entry.hazards.add(edge.kind);
    edgeByKey.set(key, entry);
  }
  const edges: LensViewEdge[] = [...edgeByKey.values()].map(entry => {
    const names = [...entry.resources].sort(
      (a, b) => (resourceById.get(b)?.bytes ?? 0) - (resourceById.get(a)?.bytes ?? 0)
    );
    return {
      from: entry.from,
      to: entry.to,
      bytes: sum(names, name => resourceById.get(name)?.bytes ?? 0),
      resources: names.slice(0, 3),
      resourceCount: names.length,
      hazards: [...entry.hazards].sort()
    };
  });

  const formulas = selectFormulas(scenarioId, record.refs?.formulas ?? []).map(formula => ({
    id: formula.id,
    ascii: truncate(formula.ascii ?? '', 320)
  }));
  const rounds = (record.metrics ?? [])
    .flatMap(metrics => (metrics.graph === graph.id ? (metrics.series ?? []) : []))
    .filter(series => series.values.length > 1 && series.values.length <= MAXIMUM_ROUND_VALUES)
    .map(series => ({
      name: series.name,
      ...(series.unit === undefined ? {} : {unit: series.unit}),
      values: toArray(series.values)
    }));
  const gatedNodeCount = graph.nodes.filter(isGated).length;
  const bufferBytes: Record<string, number> = {};
  if (!grouping.collapsed) {
    for (const node of graph.nodes) {
      for (const id of [...node.reads, ...node.writes]) {
        bufferBytes[id] = resourceById.get(id)?.bytes ?? 0;
      }
    }
  }
  const parameters: LensParameter[] = (record.choices ?? [])
    .filter(choice => choice.value !== undefined && typeof choice.value !== 'object')
    .slice(0, 6)
    .map(choice => ({
      name: choice.name,
      label: choice.label ?? choice.name,
      value: String(choice.value),
      ...(choice.unit === undefined ? {} : {unit: choice.unit})
    }));

  const snapshots = record.captures?.snapshots ?? [];
  return {
    id: scenarioId,
    title: scenarioId.slice(scenarioId.indexOf('.') + 1),
    pack: scenarioId.slice(0, scenarioId.indexOf('.')),
    adapter: record.adapter?.backend ?? 'webgpu',
    form: graph.form,
    recordedNodeCount: graph.nodes.length,
    recordedEdgeCount: graph.edges.length,
    collapsed: grouping.collapsed,
    nodes,
    edges,
    resources: summarizeResources(graph),
    ...summarizeLifetimes(graph, grouping.groups),
    hazards: countHazards(graph),
    totals: {
      workgroups: sum(nodes, node => node.workgroups),
      readBytes: sum(nodes, node => node.readBytes),
      writeBytes: sum(nodes, node => node.writeBytes)
    },
    formulas,
    parameters,
    ...extractMap(scenarioId, snapshots),
    series: extractSeries(snapshots),
    ...(rounds.length > 0 ? {rounds} : {}),
    ...(gatedNodeCount > 0 ? {gatedNodeCount} : {}),
    ...(grouping.collapsed ? {} : {bufferBytes})
  };
}

function isGated(node: RecordNode): boolean {
  return node.condition?.source === 'gpu' && node.condition.mode === 'indirect';
}

/** Workload facts of a group; read and write sets only while one node is one kernel. */
function describeMembers(
  members: readonly RecordNode[],
  collapsed: boolean
): Pick<LensViewNode, 'invocations' | 'gated' | 'variant' | 'reads' | 'writes'> {
  const invocations = sum(members, member => member.workload?.maximumInvocationCount ?? 0);
  const gated = members.filter(isGated).length;
  const single = collapsed ? undefined : members[0];
  return {
    ...(invocations > 0 ? {invocations} : {}),
    ...(gated > 0 ? {gated} : {}),
    ...(single?.workload?.variant ? {variant: single.workload.variant} : {}),
    ...(single ? {reads: [...single.reads], writes: [...single.writes]} : {})
  };
}

/**
 * The formulas of this operation. Network records list every formula of their pack, so those keep
 * only the ones whose topic (the id part after the pack) occurs in the scenario name.
 */
function selectFormulas<T extends {id: string}>(scenarioId: string, formulas: readonly T[]): T[] {
  if (formulas.length <= PACK_FORMULA_COUNT) return formulas.slice(0, 3);
  const slug = scenarioId.slice(scenarioId.indexOf('.') + 1);
  return formulas
    .filter(formula => {
      const topic = formula.id.split('.')[1];
      return topic !== undefined && slug.includes(topic);
    })
    .slice(0, 3);
}

// Graph grouping

type Group = {id: string; label: string; members: RecordNode[]};

function groupNodes(
  nodes: readonly RecordNode[],
  slug: string
): {
  groups: Group[];
  groupOfNode: Map<string, string>;
  collapsed: boolean;
} {
  const readSharedLabel = makeLabelReader(nodes);
  const labelOf = (node: RecordNode): string => {
    if (node.label.endsWith('snapshot-tap')) return 'snapshot-tap';
    const label = readSharedLabel(node);
    return label.startsWith(`${slug}-`) ? label.slice(slug.length + 1) : label;
  };
  const identity = (): ReturnType<typeof groupNodes> => ({
    groups: nodes.map(node => ({id: node.id, label: shortenLabel(labelOf(node)), members: [node]})),
    groupOfNode: new Map(nodes.map(node => [node.id, node.id])),
    collapsed: false
  });
  if (nodes.length <= MAXIMUM_NODES) return identity();
  // Fold unrolled copies: replace numbers by '#', then keep fewer leading tokens until it fits.
  for (let keep = 99; keep >= 1; keep--) {
    const groups = new Map<string, Group>();
    for (const node of nodes) {
      const tokens = labelOf(node).replace(/\d+/g, '#').split(/[-/.]/).filter(Boolean);
      const key = tokens.slice(0, keep).join('-');
      const group = groups.get(key) ?? {id: key, label: key, members: []};
      group.members.push(node);
      groups.set(key, group);
    }
    if (groups.size <= MAXIMUM_NODES || keep === 1) {
      const list = [...groups.values()];
      const groupOfNode = new Map<string, string>();
      for (const group of list) {
        group.label = shortenLabel(group.label);
        for (const member of group.members) groupOfNode.set(member.id, group.id);
      }
      return {groups: list, groupOfNode, collapsed: true};
    }
  }
  return identity();
}

function shortenLabel(label: string): string {
  return label.length > 26 ? `${label.slice(0, 25)}…` : label;
}

function getNodeKind(members: readonly RecordNode[]): LensNodeKind {
  if (members.some(member => member.kind === 'compute')) return 'compute';
  if (members.some(member => member.kind === 'render')) return 'render';
  if (members.every(member => member.kind === 'copy')) return 'copy';
  return 'other';
}

function getWorkgroups(node: RecordNode): number {
  const workgroups = node.dispatch?.workgroups;
  if (workgroups) return workgroups[0] * workgroups[1] * workgroups[2];
  return node.workload?.maximumWorkgroupCount ?? 0;
}

function readTimings(record: LensRecordLike, graphId: string): Map<string, number> {
  const timings = new Map<string, number>();
  for (const metrics of record.metrics ?? []) {
    if (metrics.graph !== graphId) continue;
    const scale = metrics.timebase === 'ns' ? 1e-6 : 1;
    for (const [id, node] of Object.entries(metrics.nodes)) {
      if (node.gpuBegin !== undefined && node.gpuEnd !== undefined)
        timings.set(id, (node.gpuEnd - node.gpuBegin) * scale);
    }
  }
  return timings;
}

function summarizeResources(graph: RecordGraph): LensResourceSummary {
  const resources = graph.resources ?? [];
  const allocations = new Map<string, number>();
  for (const resource of resources) {
    if (resource.ownership === 'transient')
      allocations.set(
        resource.allocation ?? resource.id,
        resource.allocationBytes ?? resource.bytes
      );
  }
  return {
    count: resources.length,
    totalBytes: sum(resources, resource => resource.bytes),
    importedBytes: sum(
      resources.filter(resource => resource.ownership === 'imported'),
      resource => resource.bytes
    ),
    transientBytes: sum(
      resources.filter(resource => resource.ownership === 'transient'),
      resource => resource.bytes
    ),
    allocatedBytes: [...allocations.values()].reduce((total, bytes) => total + bytes, 0),
    largest: summarizeLargest(resources)
  };
}

const MAXIMUM_LIFETIMES = 28;

/** Per buffer first write and last use in group steps, for the largest buffers. */
function summarizeLifetimes(
  graph: RecordGraph,
  groups: readonly Group[]
): {lifetimes?: LensResourceLifetime[]; lifetimesOmitted?: number} {
  const resources = graph.resources ?? [];
  if (resources.length === 0) return {};
  const firstWrite = new Map<string, number>();
  const firstUse = new Map<string, number>();
  const lastUse = new Map<string, number>();
  groups.forEach((group, step) => {
    for (const member of group.members) {
      for (const id of member.writes) {
        if (!firstWrite.has(id)) firstWrite.set(id, step);
        if (!firstUse.has(id)) firstUse.set(id, step);
        lastUse.set(id, step);
      }
      for (const id of member.reads) {
        if (!firstUse.has(id)) firstUse.set(id, step);
        lastUse.set(id, step);
      }
    }
  });
  const used = resources.filter(resource => lastUse.has(resource.id));
  const largest = [...used]
    .sort((a, b) => b.bytes - a.bytes)
    .slice(0, MAXIMUM_LIFETIMES)
    .sort(
      (a, b) =>
        (firstWrite.get(a.id) ?? firstUse.get(a.id) ?? 0) -
        (firstWrite.get(b.id) ?? firstUse.get(b.id) ?? 0)
    );
  const slots = new Map<string, number>();
  const lifetimes = largest.map(resource => {
    const lifetime: LensResourceLifetime = {
      id: resource.id,
      bytes: resource.bytes,
      ownership: resource.ownership,
      first: firstWrite.get(resource.id) ?? firstUse.get(resource.id) ?? 0,
      last: lastUse.get(resource.id) ?? 0
    };
    if (resource.ownership === 'transient' && resource.allocation !== undefined) {
      if (!slots.has(resource.allocation)) slots.set(resource.allocation, slots.size);
      lifetime.slot = slots.get(resource.allocation);
    }
    return lifetime;
  });
  return {lifetimes, lifetimesOmitted: Math.max(0, used.length - lifetimes.length)};
}

function summarizeLargest(
  resources: NonNullable<RecordGraph['resources']>
): LensResourceSummary['largest'] {
  const bySize = new Map<
    number,
    {id: string; bytes: number; ownership: 'imported' | 'transient'; count: number}
  >();
  for (const resource of resources) {
    const entry = bySize.get(resource.bytes);
    if (entry) entry.count++;
    else
      bySize.set(resource.bytes, {
        id: resource.id,
        bytes: resource.bytes,
        ownership: resource.ownership,
        count: 1
      });
  }
  return [...bySize.values()].sort((a, b) => b.bytes - a.bytes).slice(0, 6);
}

// Snapshots

function extractMap(scenarioId: string, snapshots: readonly RecordCapture[]): {map?: LensMap} {
  const byName = new Map(snapshots.map(capture => [capture.name, capture]));
  const byId = new Map(snapshots.map(capture => [capture.id, capture]));
  const geometric = snapshots.filter(capture => capture.geometry);
  const outputRaster = geometric.find(
    capture => capture.geometry?.kind === 'raster' && capture.role === 'output'
  );
  const anyRaster = outputRaster ?? geometric.find(capture => capture.geometry?.kind === 'raster');

  let frame: Frame | undefined;
  let raster: LensRaster | undefined;
  if (anyRaster?.geometry?.shape && anyRaster.geometry.origin && anyRaster.geometry.cellSize) {
    const [width, height] = anyRaster.geometry.shape;
    const cellSize = anyRaster.geometry.cellSize;
    const origin = anyRaster.geometry.origin;
    const north = anyRaster.geometry.rowDirection === 'north';
    frame = {
      minimumX: origin[0],
      maximumX: origin[0] + width * cellSize,
      minimumY: north ? origin[1] : origin[1] - height * cellSize,
      maximumY: north ? origin[1] + height * cellSize : origin[1],
      flipY: true
    };
    raster = makeRaster(anyRaster, width, height, north);
  }

  const layers: LensPointLayer[] = [];
  const pointSources: {capture: RecordCapture; values?: RecordCapture}[] = [];
  for (const capture of geometric) {
    const kind = capture.geometry?.kind;
    if (kind === 'points' && capture.role !== 'context') {
      const valuesCapture = capture.geometry?.valuesCapture;
      pointSources.push({
        capture,
        values: valuesCapture ? (byName.get(valuesCapture) ?? byId.get(valuesCapture)) : undefined
      });
    }
  }
  // Polygon outputs are drawn as one point per polygon (ring centroid) colored by the value.
  const polygonOutput =
    geometric.find(capture => capture.geometry?.kind === 'polygons' && capture.role === 'output') ??
    geometric.find(capture => capture.geometry?.kind === 'polygons');
  const polygonLayer = polygonOutput
    ? makePolygonCentroidLayer(polygonOutput, byName, byId)
    : undefined;

  const allPoints: {name: string; xy: number[]; values?: number[]}[] = [];
  for (const source of pointSources.slice(0, 3)) {
    allPoints.push({
      name: source.capture.name,
      xy: toArray(source.capture.values),
      values: source.values ? toArray(source.values.values) : undefined
    });
  }
  if (polygonLayer) allPoints.push(polygonLayer);
  if (!raster && allPoints.length === 0) return {};

  if (!frame) frame = fitFrame(allPoints.map(entry => entry.xy));
  for (const entry of allPoints) layers.push(makePointLayer(entry, frame));
  const frameWidth = frame.maximumX - frame.minimumX;
  const frameHeight = frame.maximumY - frame.minimumY;
  return {
    map: {
      title: raster ? raster.name : (layers[0]?.name ?? scenarioId),
      aspectRatio: Math.min(2, Math.max(0.5, frameWidth / frameHeight || 1)),
      ...(raster ? {raster} : {}),
      layers
    }
  };
}

type Frame = {
  minimumX: number;
  maximumX: number;
  minimumY: number;
  maximumY: number;
  flipY: boolean;
};

function fitFrame(lists: readonly number[][]): Frame {
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (const list of lists) {
    for (let index = 0; index + 1 < list.length; index += 2) {
      const x = list[index];
      const y = list[index + 1];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
    }
  }
  if (!Number.isFinite(minimumX))
    return {minimumX: 0, maximumX: 1, minimumY: 0, maximumY: 1, flipY: true};
  const padX = (maximumX - minimumX) * 0.03 || 1;
  const padY = (maximumY - minimumY) * 0.03 || 1;
  return {
    minimumX: minimumX - padX,
    maximumX: maximumX + padX,
    minimumY: minimumY - padY,
    maximumY: maximumY + padY,
    flipY: true
  };
}

function makePointLayer(
  entry: {name: string; xy: number[]; values?: number[]},
  frame: Frame
): LensPointLayer {
  const count = Math.floor(entry.xy.length / 2);
  const stride = Math.max(1, Math.ceil(count / MAXIMUM_POINTS));
  const positions: number[] = [];
  const picked: number[] = [];
  for (let index = 0; index < count; index += stride) {
    const x = (entry.xy[index * 2] - frame.minimumX) / (frame.maximumX - frame.minimumX);
    const y = (entry.xy[index * 2 + 1] - frame.minimumY) / (frame.maximumY - frame.minimumY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    positions.push(round(x, 4), round(frame.flipY ? 1 - y : y, 4));
    if (entry.values && entry.values.length === count) picked.push(entry.values[index]);
  }
  if (picked.length === 0) return {name: entry.name, positions};
  const finite = picked.filter(Number.isFinite).sort((a, b) => a - b);
  const minimum = finite[Math.floor(finite.length * 0.02)] ?? 0;
  const maximum = finite[Math.floor(finite.length * 0.98)] ?? 1;
  const range = maximum - minimum || 1;
  return {
    name: entry.name,
    positions,
    values: picked.map(value =>
      Number.isFinite(value) ? round(Math.min(1, Math.max(0, (value - minimum) / range)), 3) : 0
    ),
    min: round(minimum, 4),
    max: round(maximum, 4)
  };
}

function makePolygonCentroidLayer(
  geometry: RecordCapture,
  byName: Map<string, RecordCapture>,
  byId: Map<string, RecordCapture>
): {name: string; xy: number[]; values?: number[]} | undefined {
  const offsets = lookup(byName, byId, geometry.geometry?.offsetsCapture);
  if (!offsets) return undefined;
  const vertices = toArray(geometry.values);
  const ringOffsets = toArray(offsets.values);
  const valuesCapture = lookup(byName, byId, geometry.geometry?.valuesCapture);
  const xy: number[] = [];
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    let x = 0;
    let y = 0;
    const start = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    for (let vertex = start; vertex < end; vertex++) {
      x += vertices[vertex * 2];
      y += vertices[vertex * 2 + 1];
    }
    const count = Math.max(1, end - start);
    xy.push(x / count, y / count);
  }
  return {
    name: geometry.name.replace(/\.geometry$/, ''),
    xy,
    values: valuesCapture ? toArray(valuesCapture.values) : undefined
  };
}

function lookup(
  byName: Map<string, RecordCapture>,
  byId: Map<string, RecordCapture>,
  key: string | undefined
): RecordCapture | undefined {
  return key === undefined ? undefined : (byName.get(key) ?? byId.get(key));
}

function makeRaster(
  capture: RecordCapture,
  width: number,
  height: number,
  flipRows: boolean
): LensRaster {
  const source = toArray(capture.values);
  const factor = Math.max(1, Math.ceil(Math.sqrt((width * height) / MAXIMUM_RASTER_CELLS)));
  const outWidth = Math.ceil(width / factor);
  const outHeight = Math.ceil(height / factor);
  const cells = new Float64Array(outWidth * outHeight);
  const counts = new Float64Array(outWidth * outHeight);
  for (let row = 0; row < height; row++) {
    const sourceRow = flipRows ? height - 1 - row : row;
    for (let column = 0; column < width; column++) {
      const value = source[sourceRow * width + column];
      if (!Number.isFinite(value)) continue;
      const target = Math.floor(row / factor) * outWidth + Math.floor(column / factor);
      cells[target] += value;
      counts[target] += 1;
    }
  }
  const means = Array.from(cells, (total, index) => (counts[index] ? total / counts[index] : 0));
  const sorted = means.filter(Number.isFinite).sort((a, b) => a - b);
  const minimum = sorted[Math.floor(sorted.length * 0.01)] ?? 0;
  const maximum = sorted[Math.floor(sorted.length * 0.99)] ?? 1;
  const range = maximum - minimum || 1;
  const bytes = new Uint8Array(means.length);
  means.forEach((value, index) => {
    bytes[index] = Math.round(255 * Math.min(1, Math.max(0, (value - minimum) / range)));
  });
  return {
    name: capture.name,
    width: outWidth,
    height: outHeight,
    grid: Buffer.from(bytes).toString('base64'),
    min: round(minimum, 6),
    max: round(maximum, 6),
    png: `data:image/png;base64,${encodeColoredPng(bytes, outWidth, outHeight)}`
  };
}

function extractSeries(snapshots: readonly RecordCapture[]): LensSeries[] {
  const candidates = snapshots.filter(
    capture =>
      !capture.geometry &&
      (capture.components ?? 1) === 1 &&
      capture.values.length >= 4 &&
      capture.values.length <= 20000 &&
      capture.role !== 'context' &&
      !/offset|neighbors|ids?$/i.test(capture.name)
  );
  candidates.sort((a, b) => rankRole(a.role) - rankRole(b.role));
  return candidates.slice(0, 3).map(capture => {
    const source = toArray(capture.values);
    const block = Math.max(1, Math.ceil(source.length / MAXIMUM_SERIES_VALUES));
    const values: number[] = [];
    for (let start = 0; start < source.length; start += block) {
      let total = 0;
      let count = 0;
      for (let index = start; index < Math.min(source.length, start + block); index++) {
        if (Number.isFinite(source[index])) {
          total += source[index];
          count++;
        }
      }
      values.push(count ? round(total / count, 6) : 0);
    }
    return {
      name: capture.name,
      role: capture.role ?? 'intermediate',
      values,
      length: source.length,
      elementType: capture.elementType,
      min: Math.min(...values),
      max: Math.max(...values)
    };
  });
}

function rankRole(role: string | undefined): number {
  return role === 'output' ? 0 : role === 'input' ? 1 : 2;
}

// Helpers

function sum<T>(items: readonly T[], read: (item: T) => number): number {
  let total = 0;
  for (const item of items) total += read(item);
  return total;
}

function toArray(values: ArrayLike<number>): number[] {
  return Array.from(values);
}

function round(value: number, digits: number): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

// A minimal PNG encoder (8-bit RGBA) so the static fallback needs no image library. Low values are
// transparent, matching the live shader that blends them into the page surface.
function encodeColoredPng(bytes: Uint8Array, width: number, height: number): string {
  const row = width * 4 + 1;
  const raw = Buffer.alloc(row * height);
  for (let y = 0; y < height; y++) {
    raw[y * row] = 0;
    for (let x = 0; x < width; x++) {
      const value = bytes[y * width + x] / 255;
      const [red, green, blue] = sampleValueRamp(value);
      const offset = y * row + 1 + x * 4;
      raw[offset] = Math.round(red * 255);
      raw[offset + 1] = Math.round(green * 255);
      raw[offset + 2] = Math.round(blue * 255);
      raw[offset + 3] = Math.round(255 * smoothstep(0.02, 0.28, value));
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  const chunk = (type: string, data: Buffer): Buffer => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const checksum = Buffer.alloc(4);
    checksum.writeUInt32BE(crc32(body));
    return Buffer.concat([length, body, checksum]);
  };
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0))
  ]).toString('base64');
}

/** Drops the leading tokens every label shares, such as the operation name. */
function makeLabelReader(nodes: readonly RecordNode[]): (node: RecordNode) => string {
  const tokenLists = nodes.map(node => node.label.split('-'));
  let shared = 0;
  const first = tokenLists[0] ?? [];
  while (
    shared < first.length - 1 &&
    tokenLists.every(tokens => tokens.length > shared + 1 && tokens[shared] === first[shared])
  )
    shared++;
  return node => node.label.split('-').slice(shared).join('-');
}

function countHazards(graph: RecordGraph): {raw: number; war: number; waw: number} {
  const hazards = {raw: 0, war: 0, waw: 0};
  for (const edge of graph.edges) {
    if (edge.kind === 'raw' || edge.kind === 'war' || edge.kind === 'waw') hazards[edge.kind]++;
  }
  return hazards;
}

function smoothstep(edgeStart: number, edgeEnd: number, value: number): number {
  const t = Math.min(1, Math.max(0, (value - edgeStart) / (edgeEnd - edgeStart)));
  return t * t * (3 - 2 * t);
}
