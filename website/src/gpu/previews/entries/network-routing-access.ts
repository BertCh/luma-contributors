// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/network-routing-access.ts, keyed by entry id.

import {
  clamp,
  createArc,
  createCircle,
  createClusterPoints,
  createRoadNetwork,
  createScatterPoints,
  evaluateField,
  getDistance,
  getPathTo,
  mixNumbers,
  type Random,
  type RoadNetwork
} from '../data.ts';
import {
  CLASS_ROLES,
  clampPoint,
  findNearestIndex,
  getClassRole,
  pickClosedEdges,
  smoothPath,
  toPaths,
  toSprites,
  type EntryPreview
} from '../helpers.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type FillLayer,
  type PathItem,
  type Point,
  type PolygonItem,
  type PreviewLayer,
  type Role,
  type SpriteItem,
  type SpriteLayer,
  type StrokeLayer
} from '../types.ts';

// ---------------------------------------------------------------------------------------------
// Layer shorthands

type StrokeOptions = Partial<Omit<StrokeLayer, 'type' | 'items' | 'animation'>>;

function createStrokes(
  items: PathItem[],
  animation: StrokeLayer['animation'] = 'static',
  options: StrokeOptions = {}
): StrokeLayer {
  return {type: 'strokes', items, animation, ...options};
}

function createSprites(items: SpriteItem[], pulse = 0): SpriteLayer {
  return {type: 'sprites', items, pulse};
}

function createFills(
  items: PolygonItem[],
  animation: FillLayer['animation'] = 'static',
  blend: FillLayer['blend'] = 'add'
): FillLayer {
  return {type: 'fills', items, animation, blend};
}

function createRectangle(x: number, y: number, width: number, height: number): Point[] {
  return [
    [x, y],
    [x + width, y],
    [x + width, y + height],
    [x, y + height]
  ];
}

function getHeatRole(value: number): Role {
  return CLASS_ROLES[clamp(Math.floor(value * 5), 0, 4)];
}

const OWNER_ROLES: Role[] = ['warm', 'cool', 'violet', 'primary', 'hot'];

// ---------------------------------------------------------------------------------------------
// Road network helpers

type Arc = {to: number; length: number; edge: number};
type CostFunction = (arc: Arc, from: number, time: number) => number;
type SearchResult = {
  distances: number[];
  previous: number[];
  previousEdge: number[];
  owner: number[];
  order: number[];
};

function createHeap() {
  const keys: number[] = [];
  const values: number[] = [];
  return {
    get size() {
      return keys.length;
    },
    push(key: number, value: number) {
      let index = keys.length;
      keys.push(key);
      values.push(value);
      while (index > 0) {
        const parent = (index - 1) >> 1;
        if (keys[parent] <= keys[index]) break;
        [keys[parent], keys[index]] = [keys[index], keys[parent]];
        [values[parent], values[index]] = [values[index], values[parent]];
        index = parent;
      }
    },
    pop(): [number, number] {
      const top: [number, number] = [keys[0], values[0]];
      const lastKey = keys.pop() as number;
      const lastValue = values.pop() as number;
      if (keys.length > 0) {
        keys[0] = lastKey;
        values[0] = lastValue;
        let index = 0;
        for (;;) {
          const left = index * 2 + 1;
          const right = left + 1;
          let smallest = index;
          if (left < keys.length && keys[left] < keys[smallest]) smallest = left;
          if (right < keys.length && keys[right] < keys[smallest]) smallest = right;
          if (smallest === index) break;
          [keys[smallest], keys[index]] = [keys[index], keys[smallest]];
          [values[smallest], values[index]] = [values[index], values[smallest]];
          index = smallest;
        }
      }
      return top;
    }
  };
}

function createAdjacency(net: RoadNetwork, closed?: ReadonlySet<number>): Arc[][] {
  const adjacency: Arc[][] = Array.from({length: net.nodes.length}, () => []);
  net.edges.forEach(([first, second], edge) => {
    if (closed?.has(edge)) return;
    const length = getDistance(net.nodes[first], net.nodes[second]);
    adjacency[first].push({to: second, length, edge});
    adjacency[second].push({to: first, length, edge});
  });
  return adjacency;
}

/** Dijkstra from one or many sources; `cost` makes it time dependent or penalised. */
function runSearch(adjacency: Arc[][], sources: number[], cost?: CostFunction): SearchResult {
  const count = adjacency.length;
  const distances = new Array<number>(count).fill(Infinity);
  const previous = new Array<number>(count).fill(-1);
  const previousEdge = new Array<number>(count).fill(-1);
  const owner = new Array<number>(count).fill(-1);
  const settled = new Array<boolean>(count).fill(false);
  const order: number[] = [];
  const heap = createHeap();
  sources.forEach((source, index) => {
    distances[source] = 0;
    owner[source] = index;
    heap.push(0, source);
  });
  while (heap.size > 0) {
    const [distance, node] = heap.pop();
    if (settled[node]) continue;
    settled[node] = true;
    order.push(node);
    for (const arc of adjacency[node]) {
      const next = distance + (cost ? cost(arc, node, distance) : arc.length);
      if (next < distances[arc.to]) {
        distances[arc.to] = next;
        previous[arc.to] = node;
        previousEdge[arc.to] = arc.edge;
        owner[arc.to] = owner[node];
        heap.push(next, arc.to);
      }
    }
  }
  return {distances, previous, previousEdge, owner, order};
}

/** Number of shortest-path descendants of each node, itself included. */
function getSubtreeCounts(result: SearchResult): number[] {
  const counts = result.distances.map(distance => (distance < Infinity ? 1 : 0));
  for (let index = result.order.length - 1; index >= 0; index--) {
    const node = result.order[index];
    if (result.previous[node] >= 0) counts[result.previous[node]] += counts[node];
  }
  return counts;
}

function getMaxFinite(values: number[]): number {
  let maximum = 0;
  for (const value of values) if (value < Infinity && value > maximum) maximum = value;
  return maximum || 1;
}

function getNodeIndex(net: RoadNetwork, row: number, column: number): number {
  return row * net.columns + column;
}

function pickNode(net: RoadNetwork, pointer: Point | null, row: number, column: number): number {
  return pointer ? findNearestIndex(net.nodes, pointer) : getNodeIndex(net, row, column);
}

function getPathPoints(net: RoadNetwork, nodes: number[]): Point[] {
  return nodes.map(node => net.nodes[node]);
}

function getEdgePaths(net: RoadNetwork, edges: Iterable<number>): Point[][] {
  const paths: Point[][] = [];
  for (const edge of edges) {
    paths.push([net.nodes[net.edges[edge][0]], net.nodes[net.edges[edge][1]]]);
  }
  return paths;
}

function getAllEdges(net: RoadNetwork): number[] {
  return net.edges.map((_, index) => index);
}

/** Network drawn into a sub box of the scene, so a chart or matrix can share the picture. */
function placeNet(
  net: RoadNetwork,
  box: {x: number; y: number; width: number; height: number}
): RoadNetwork {
  const nodes = net.nodes.map(
    ([x, y]): Point => [
      box.x + ((x - 12) / (SCENE_WIDTH - 24)) * box.width,
      box.y + ((y - 12) / (SCENE_HEIGHT - 24)) * box.height
    ]
  );
  return {...net, nodes};
}

function createBackdrop(
  net: RoadNetwork,
  options: {closed?: ReadonlySet<number>; alpha?: number; width?: number} = {}
): StrokeLayer {
  const {closed, alpha = 0.5, width = 0.5} = options;
  const edges = getAllEdges(net).filter(edge => !closed?.has(edge));
  return createStrokes(toPaths(getEdgePaths(net, edges), 'dim', width, alpha));
}

function findNearestSegmentPoint(
  net: RoadNetwork,
  point: Point
): {point: Point; distance: number; edge: number} {
  let best = {point: net.nodes[0], distance: Infinity, edge: 0};
  net.edges.forEach(([first, second], edge) => {
    const start = net.nodes[first];
    const end = net.nodes[second];
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const amount = clamp(
      ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy || 1),
      0,
      1
    );
    const projected: Point = [start[0] + dx * amount, start[1] + dy * amount];
    const distance = getDistance(point, projected);
    if (distance < best.distance) best = {point: projected, distance, edge};
  });
  return best;
}

/** Points sampled along a polyline at a fixed spacing. */
function samplePolyline(points: Point[], spacing: number): Point[] {
  const samples: Point[] = [points[0]];
  let carried = 0;
  for (let index = 0; index + 1 < points.length; index++) {
    const start = points[index];
    const end = points[index + 1];
    const length = getDistance(start, end);
    let position = spacing - carried;
    while (position <= length) {
      const amount = position / length;
      samples.push([mixNumbers(start[0], end[0], amount), mixNumbers(start[1], end[1], amount)]);
      position += spacing;
    }
    carried = length - (position - spacing);
  }
  return samples;
}

function createCostField(
  net: RoadNetwork,
  distances: number[],
  limit: number,
  columns = 48,
  rows = 30
): Float32Array {
  return evaluateField(columns, rows, (u, v) => {
    const x = u * SCENE_WIDTH;
    const y = v * SCENE_HEIGHT;
    let best = Infinity;
    for (let index = 0; index < net.nodes.length; index++) {
      if (distances[index] === Infinity) continue;
      const cost =
        distances[index] + 1.25 * Math.hypot(net.nodes[index][0] - x, net.nodes[index][1] - y);
      if (cost < best) best = cost;
    }
    return Math.min(best, limit) / limit;
  });
}

function getCombinations(count: number, size: number): number[][] {
  const result: number[][] = [];
  const current: number[] = [];
  const visit = (start: number) => {
    if (current.length === size) {
      result.push([...current]);
      return;
    }
    for (let index = start; index < count; index++) {
      current.push(index);
      visit(index + 1);
      current.pop();
    }
  };
  visit(0);
  return result;
}

/** Network-constrained distances between every pair of nodes. */
function getAllPairs(adjacency: Arc[][]): number[][] {
  return adjacency.map((_, source) => runSearch(adjacency, [source]).distances);
}

/** Seeded node subset of a given size; the weights bias which nodes are likely. */
function pickWeightedNodes(random: Random, weights: number[], count: number): number[] {
  return weights
    .map((weight, index) => ({index, key: Math.pow(random.next(), 1 / Math.max(weight, 1e-6))}))
    .sort((first, second) => second.key - first.key)
    .slice(0, count)
    .map(entry => entry.index);
}

type CommunityGraph = {points: Point[]; group: number[]; edges: [number, number][]};

const COMMUNITY_CENTERS: Point[] = [
  [42, 34],
  [116, 32],
  [46, 70],
  [114, 68],
  [80, 50]
];

/** A small graph with planted communities, laid out as rings around cluster centers. */
function createCommunityGraph(random: Random, clusterCount: number, size: number): CommunityGraph {
  const points: Point[] = [];
  const group: number[] = [];
  const edges: [number, number][] = [];
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    const [centerX, centerY] = COMMUNITY_CENTERS[cluster];
    const base = points.length;
    for (let index = 0; index < size; index++) {
      const angle = (index / size) * Math.PI * 2 + random.range(-0.2, 0.2);
      const radius = random.range(9, 14);
      points.push([centerX + Math.cos(angle) * radius, centerY + Math.sin(angle) * radius * 0.85]);
      group.push(cluster);
    }
    for (let index = 0; index < size; index++) {
      edges.push([base + index, base + ((index + 1) % size)]);
      for (let other = index + 2; other < size; other++) {
        if (random.next() < 0.3 && !(index === 0 && other === size - 1)) {
          edges.push([base + index, base + other]);
        }
      }
    }
  }
  for (let cluster = 0; cluster < clusterCount; cluster++) {
    const next = (cluster + 1) % clusterCount;
    if (next === cluster) continue;
    edges.push([cluster * size + random.integer(size), next * size + random.integer(size)]);
    if (random.next() < 0.6) {
      edges.push([cluster * size + random.integer(size), next * size + random.integer(size)]);
    }
  }
  return {points, group, edges};
}

function shuffleIndices(random: Random, count: number): number[] {
  const order = Array.from({length: count}, (_, index) => index);
  for (let index = count - 1; index > 0; index--) {
    const other = random.integer(index + 1);
    [order[index], order[other]] = [order[other], order[index]];
  }
  return order;
}

/** Clustered events on network nodes: the nodes closest to a few seeded centers are most likely. */
function pickClusteredEvents(
  net: RoadNetwork,
  random: Random,
  count: number,
  centers: Point[],
  spread: number
): number[] {
  const weights = net.nodes.map(node => {
    let best = 0;
    for (const center of centers) {
      best = Math.max(best, Math.exp(-Math.pow(getDistance(node, center) / spread, 2)));
    }
    return best + 0.02;
  });
  return pickWeightedNodes(random, weights, count);
}

const RADIUS_BINS = 12;

/** Cumulative count of event pairs within each radius bin, from a node-to-node distance table. */
function countPairsWithin(
  table: number[][],
  nodes: number[],
  maximumRadius: number,
  weight?: (first: number, second: number) => number
): number[] {
  const counts = new Array<number>(RADIUS_BINS).fill(0);
  for (let first = 0; first < nodes.length; first++) {
    for (let second = first + 1; second < nodes.length; second++) {
      const distance = table[nodes[first]][nodes[second]];
      const bin = Math.ceil((distance / maximumRadius) * RADIUS_BINS) - 1;
      if (bin >= RADIUS_BINS) continue;
      counts[Math.max(bin, 0)] += weight ? weight(nodes[first], nodes[second]) : 1;
    }
  }
  for (let index = 1; index < RADIUS_BINS; index++) counts[index] += counts[index - 1];
  return counts;
}

/** A small chart box in scene units: maps (radius bin, value) to a point. */
function createChartMapper(box: {x: number; y: number; width: number; height: number}) {
  return {
    box,
    at(bin: number, value: number): Point {
      return [
        box.x + ((bin + 1) / RADIUS_BINS) * box.width,
        box.y + box.height - clamp(value, 0, 1) * box.height
      ];
    },
    axes(): PathItem[] {
      return toPaths(
        [
          [
            [box.x, box.y],
            [box.x, box.y + box.height],
            [box.x + box.width, box.y + box.height]
          ]
        ],
        'dim',
        0.5,
        0.7
      );
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Reachability, service areas, isochrones

const reachability: EntryPreview = {
  pointer: 'sticky',
  parameters: [
    {id: 'budget', label: 'Cost budget', min: 20, max: 100, step: 5, default: 70, unit: '%'},
    {id: 'starts', label: 'Start nodes', min: 1, max: 3, step: 1, default: 2}
  ],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const starts = [
      pickNode(net, pointer, 2, 2),
      getNodeIndex(net, 4, 6),
      getNodeIndex(net, 1, 7)
    ].slice(0, Math.round(p.starts));
    const result = runSearch(adjacency, starts);
    const limit = getMaxFinite(result.distances) * (p.budget / 100);
    const bandRoles: Role[] = ['cool', 'primary', 'warm', 'hot'];
    const getBand = (node: number) => Math.min(3, Math.floor((result.distances[node] / limit) * 4));
    const tree: PathItem[] = [];
    const nodes: SpriteItem[] = [];
    net.nodes.forEach(([x, y], node) => {
      const reached = result.distances[node] <= limit;
      if (reached && result.previous[node] >= 0) {
        tree.push({
          points: [net.nodes[result.previous[node]], net.nodes[node]],
          role: bandRoles[getBand(node)],
          width: 1.2,
          alpha: 0.95,
          phase: result.distances[node] / limit
        });
      }
      nodes.push({
        x,
        y,
        radius: reached ? 1.7 : 1.1,
        role: reached ? bandRoles[getBand(node)] : 'dim',
        alpha: reached ? 0.95 : 0.4,
        phase: reached ? result.distances[node] / limit : 0
      });
    });
    const startSprites = toSprites(
      starts.map(node => net.nodes[node]),
      'ink',
      4.2
    );
    return [
      createBackdrop(net),
      createStrokes(tree, 'flow', {speed: 0.35, trail: 1.2, blend: 'add'}),
      createSprites(nodes, 0.35),
      createSprites(startSprites, 0.3)
    ];
  }
};

const serviceAreas: EntryPreview = {
  pointer: 'hover',
  parameters: [{id: 'sites', label: 'Facilities', min: 2, max: 5, step: 1, default: 4}],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const sites = [
      pickNode(net, pointer, 1, 1),
      getNodeIndex(net, 1, 6),
      getNodeIndex(net, 4, 2),
      getNodeIndex(net, 4, 7),
      getNodeIndex(net, 2, 4)
    ].slice(0, Math.round(p.sites));
    const result = runSearch(adjacency, sites);
    const owned: PathItem[] = [];
    const boundary: Point[] = [];
    net.edges.forEach(([first, second]) => {
      const points: Point[] = [net.nodes[first], net.nodes[second]];
      if (result.owner[first] === result.owner[second]) {
        owned.push({
          points,
          role: OWNER_ROLES[result.owner[first]],
          width: 1,
          alpha: 0.85,
          phase: result.distances[first] / 120
        });
      } else {
        boundary.push([(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2]);
      }
    });
    return [
      createBackdrop(net, {alpha: 0.3}),
      createStrokes(owned, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1.5,
          role: OWNER_ROLES[result.owner[node]] ?? 'dim',
          alpha: 0.9,
          phase: result.distances[node] / 120
        })),
        0.3
      ),
      createSprites(toSprites(boundary, 'ink', 1.6, undefined, 0.95), 0.5),
      createSprites(
        sites.map((node, index) => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 4.4,
          role: OWNER_ROLES[index],
          phase: index / 5
        })),
        0.3
      )
    ];
  }
};

const isochrones: EntryPreview = {
  pointer: 'sticky',
  parameters: [
    {id: 'budget', label: 'Travel budget', min: 20, max: 100, step: 5, default: 75, unit: '%'},
    {id: 'bands', label: 'Contour bands', min: 3, max: 12, step: 1, default: 7}
  ],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const source = pickNode(net, pointer, 3, 3);
    const result = runSearch(createAdjacency(net), [source]);
    const limit = 200 * (p.budget / 100);
    const values = createCostField(net, result.distances, limit).map(value => 1 - value);
    return [
      {
        type: 'field',
        columns: 48,
        rows: 30,
        values,
        ramp: 'heat',
        style: 'contours',
        levels: Math.round(p.bands),
        strength: 0.8,
        speed: 0.12
      },
      createStrokes(toPaths(getEdgePaths(net, getAllEdges(net)), 'ink', 0.35, 0.2)),
      createSprites(
        [{x: net.nodes[source][0], y: net.nodes[source][1], radius: 4, role: 'ink'}],
        0.3
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Cost matrix, cost table, cost weights

const costMatrix: EntryPreview = {
  pointer: 'hover',
  parameters: [
    {id: 'bound', label: 'Cost bound', min: 20, max: 100, step: 5, default: 75, unit: '%'}
  ],
  build: ({random, p, pointer}) => {
    const net = placeNet(createRoadNetwork(random), {x: 4, y: 6, width: 100, height: 88});
    const adjacency = createAdjacency(net);
    const origins = [
      pickNode(net, pointer, 1, 1),
      getNodeIndex(net, 4, 2),
      getNodeIndex(net, 2, 5),
      getNodeIndex(net, 5, 7)
    ];
    const searches = origins.map(origin => runSearch(adjacency, [origin]));
    const columns = Array.from({length: 14}, (_, index) => Math.round(2 + index * 3.6));
    const bound = getMaxFinite(searches[0].distances) * (p.bound / 100) * 0.8;
    const trees: PathItem[] = [];
    searches.forEach((search, row) => {
      net.nodes.forEach((_, node) => {
        if (search.previous[node] >= 0 && search.distances[node] <= bound) {
          trees.push({
            points: [net.nodes[search.previous[node]], net.nodes[node]],
            role: OWNER_ROLES[row],
            width: 0.7,
            alpha: 0.5,
            phase: search.distances[node] / bound
          });
        }
      });
    });
    const cells: PolygonItem[] = [];
    searches.forEach((search, row) => {
      columns.forEach((node, column) => {
        const cost = search.distances[node];
        const reached = cost <= bound;
        cells.push({
          ring: createRectangle(112 + column * 3.4, 34 + row * 8, 2.8, 6.4),
          role: reached ? getHeatRole(cost / bound) : 'dim',
          alpha: reached ? 0.9 : 0.22,
          phase: (row * 14 + column) / 56
        });
      });
    });
    return [
      createBackdrop(net, {alpha: 0.35}),
      createStrokes(trees, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(
        columns.map(node => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 1.5,
          role: 'ink' as Role,
          alpha: 0.7
        }))
      ),
      createSprites(
        origins.map((node, row) => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 3.8,
          role: OWNER_ROLES[row],
          phase: row / 4
        })),
        0.3
      ),
      createFills(cells, 'pulse')
    ];
  }
};

const costTable: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'bound', label: 'Cost bound', min: 10, max: 90, step: 5, default: 35, unit: '%'}
  ],
  build: ({random, p}) => {
    const net = placeNet(createRoadNetwork(random), {x: 4, y: 6, width: 82, height: 88});
    const adjacency = createAdjacency(net);
    const sampled = Array.from({length: 14}, (_, index) => Math.round(index * 3.8));
    const table = sampled.map(node => runSearch(adjacency, [node]).distances);
    const bound = getMaxFinite(table[0]) * (p.bound / 100) * 1.2;
    const pitch = 3.6;
    const originX = 96;
    const originY = 22;
    const lit: PolygonItem[] = [];
    const empty: Point[] = [];
    const rowTicks: PathItem[] = [];
    table.forEach((row, rowIndex) => {
      let reachable = 0;
      sampled.forEach((node, columnIndex) => {
        const x = originX + columnIndex * pitch;
        const y = originY + rowIndex * pitch;
        if (row[node] <= bound) {
          reachable++;
          lit.push({
            ring: createRectangle(x, y, pitch - 0.7, pitch - 0.7),
            role: getHeatRole(row[node] / bound),
            alpha: 0.95,
            phase: (rowIndex + columnIndex) / 28
          });
        } else {
          empty.push([x + pitch / 2 - 0.3, y + pitch / 2 - 0.3]);
        }
      });
      rowTicks.push({
        points: [
          [originX + 14 * pitch + 2, originY + rowIndex * pitch + pitch / 2 - 0.3],
          [originX + 14 * pitch + 2 + reachable * 0.9, originY + rowIndex * pitch + pitch / 2 - 0.3]
        ],
        role: 'warm',
        width: 1.3,
        alpha: 0.9
      });
    });
    return [
      createBackdrop(net, {alpha: 0.4}),
      createSprites(
        sampled.map((node, index) => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 2.4,
          role: getClassRole(index, 14),
          phase: index / 14
        })),
        0.3
      ),
      createSprites(toSprites(empty, 'dim', 0.5, undefined, 0.5)),
      createFills(lit, 'pulse'),
      createStrokes(rowTicks, 'pulse', {speed: 0.3, blend: 'add'})
    ];
  }
};

const costWeights: EntryPreview = {
  pointer: 'sticky',
  parameters: [
    {id: 'bandwidth', label: 'Decay scale', min: 10, max: 80, step: 2, default: 32, unit: 'px'}
  ],
  build: ({random, p, pointer}) => {
    const net = placeNet(createRoadNetwork(random), {x: 4, y: 4, width: 112, height: 92});
    const origin = pickNode(net, pointer, 2, 3);
    const result = runSearch(createAdjacency(net), [origin]);
    const links: PathItem[] = [];
    const nodes: SpriteItem[] = [];
    net.nodes.forEach(([x, y], node) => {
      const weight = Math.exp(-result.distances[node] / p.bandwidth);
      if (node !== origin && weight > 0.05) {
        links.push({
          points: [net.nodes[origin], [x, y]],
          role: weight > 0.5 ? 'warm' : weight > 0.22 ? 'primary' : 'cool',
          width: 0.5 + weight * 1.8,
          alpha: 0.35 + weight * 0.6,
          phase: weight
        });
      }
      nodes.push({
        x,
        y,
        radius: 1 + weight * 3,
        role: weight > 0.5 ? 'warm' : weight > 0.15 ? 'primary' : 'dim',
        alpha: 0.4 + weight * 0.6
      });
    });
    const curve: Point[] = Array.from({length: 24}, (_, index): Point => {
      const amount = index / 23;
      return [124 + amount * 30, 90 - Math.exp(-(amount * 120) / p.bandwidth) * 32];
    });
    return [
      createBackdrop(net, {alpha: 0.3}),
      createStrokes(links, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(nodes),
      createSprites(
        [{x: net.nodes[origin][0], y: net.nodes[origin][1], radius: 4, role: 'ink'}],
        0.3
      ),
      createStrokes(
        [
          {
            points: [
              [124, 56],
              [124, 90],
              [154, 90]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          }
        ],
        'static'
      ),
      createStrokes([{points: curve, role: 'violet', width: 1.4, alpha: 1}], 'draw', {
        speed: 0.25,
        blend: 'add'
      })
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Accessibility, catchments, Huff

const accessibility: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'decay', label: 'Cost scale', min: 10, max: 80, step: 2, default: 36, unit: 'px'},
    {
      id: 'measure',
      label: 'Measure',
      min: 0,
      max: 1,
      step: 1,
      default: 1,
      options: ['Cumulative', 'Gravity']
    }
  ],
  build: ({random, fork, p}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const opportunityRandom = fork('opportunities');
    const opportunities = Array.from({length: 6}, () =>
      opportunityRandom.integer(net.nodes.length)
    );
    const scores = new Array<number>(net.nodes.length).fill(0);
    for (const opportunity of opportunities) {
      const {distances} = runSearch(adjacency, [opportunity]);
      distances.forEach((distance, node) => {
        scores[node] +=
          p.measure === 0 ? (distance <= p.decay * 1.6 ? 1 : 0) : Math.exp(-distance / p.decay);
      });
    }
    const maximum = Math.max(...scores, 1e-6);
    return [
      createBackdrop(net, {alpha: 0.35}),
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1 + Math.sqrt(scores[node] / maximum) * 3.6,
          role: getHeatRole(scores[node] / maximum),
          alpha: 0.5 + 0.5 * (scores[node] / maximum),
          phase: scores[node] / maximum
        })),
        0.3
      ),
      createStrokes(
        opportunities.flatMap(node => [
          {
            points: createCircle(net.nodes[node][0], net.nodes[node][1], 6, 24),
            role: 'violet' as Role,
            width: 0.8,
            alpha: 0.9
          }
        ]),
        'pulse',
        {speed: 0.35, blend: 'add'}
      ),
      createSprites(
        opportunities.map(node => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 2.4,
          role: 'violet' as Role
        })),
        0.2
      )
    ];
  }
};

const catchmentAccessibility: EntryPreview = {
  pointer: 'hover',
  parameters: [
    {id: 'radius', label: 'Catchment', min: 12, max: 52, step: 1, default: 28, unit: 'px'}
  ],
  build: ({fork, p, pointer}) => {
    const supplyRandom = fork('supply');
    const facilities: Point[] = [
      pointer ? clampPoint(pointer, 10) : [40, 38],
      [114, 32],
      [90, 76],
      [32, 78]
    ];
    const supply = facilities.map(() => supplyRandom.range(0.8, 2));
    const demand = createClusterPoints(
      fork('demand'),
      [
        [48, 52],
        [100, 44],
        [76, 74],
        [118, 74]
      ],
      9,
      10
    );
    const populationRandom = fork('population');
    const population = demand.map(() => populationRandom.range(0.6, 1.6));
    const ratios = facilities.map((facility, index) => {
      let covered = 0;
      demand.forEach((point, zone) => {
        if (getDistance(point, facility) <= p.radius) covered += population[zone];
      });
      return covered > 0 ? supply[index] / covered : 0;
    });
    const access = demand.map(point =>
      facilities.reduce(
        (sum, facility, index) =>
          sum + (getDistance(point, facility) <= p.radius ? ratios[index] : 0),
        0
      )
    );
    const maximum = Math.max(...access, 1e-6);
    return [
      createStrokes(
        facilities.map(facility => ({
          points: createCircle(facility[0], facility[1], p.radius, 48),
          role: 'warm' as Role,
          width: 0.7,
          alpha: 0.7
        })),
        'pulse',
        {speed: 0.3, blend: 'add'}
      ),
      createSprites(
        demand.map(([x, y], zone) => ({
          x,
          y,
          radius: 1 + population[zone] * 0.9,
          role: access[zone] > 0 ? getHeatRole(access[zone] / maximum) : ('dim' as Role),
          alpha: access[zone] > 0 ? 0.95 : 0.4,
          phase: zone / demand.length
        })),
        0.25
      ),
      createSprites(
        facilities.map(([x, y], index) => ({
          x,
          y,
          radius: 2 + supply[index] * 1.6,
          role: 'ink' as Role,
          phase: index / 4
        })),
        0.25
      )
    ];
  }
};

const huffTradeAreas: EntryPreview = {
  pointer: 'hover',
  parameters: [{id: 'beta', label: 'Distance decay', min: 0.5, max: 4, step: 0.1, default: 2}],
  build: ({fork, p, pointer}) => {
    const facilities: Point[] = [pointer ? clampPoint(pointer, 10) : [44, 42], [118, 30], [92, 78]];
    const attractiveness = [1.4, 1, 0.8];
    const demand = createClusterPoints(
      fork('demand'),
      [
        [52, 66],
        [100, 52],
        [78, 24]
      ],
      5,
      9
    );
    const links: PathItem[] = [];
    const load = [0, 0, 0];
    demand.forEach((point, zone) => {
      const utility = facilities.map(
        (facility, index) =>
          attractiveness[index] / Math.pow(Math.max(getDistance(point, facility), 4), p.beta)
      );
      const total = utility.reduce((sum, value) => sum + value, 0);
      utility.forEach((value, index) => {
        const probability = value / total;
        load[index] += probability;
        if (probability > 0.05) {
          links.push({
            points: createArc(point, facilities[index], 0.08),
            role: OWNER_ROLES[index],
            width: 0.4 + probability * 2.2,
            alpha: 0.25 + probability * 0.7,
            phase: (zone * 0.17 + index * 0.31) % 1
          });
        }
      });
    });
    const loadTotal = load.reduce((sum, value) => sum + value, 0);
    return [
      createStrokes(links, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(toSprites(demand, 'cool', 1.2, undefined, 0.85)),
      createSprites(
        facilities.map(([x, y], index) => ({
          x,
          y,
          radius: 2.4 + 6 * Math.sqrt(load[index] / loadTotal),
          role: OWNER_ROLES[index],
          phase: index / 3
        })),
        0.2
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Facility selection

type Objective = 'median' | 'center' | 'coverage';

function selectFacilities(
  candidates: Point[],
  demand: Point[],
  count: number,
  objective: Objective,
  coverageRadius: number
): number[] {
  const distances = candidates.map(candidate => demand.map(point => getDistance(point, candidate)));
  const score = (chosen: number[]) => {
    let total = 0;
    let worst = 0;
    let covered = 0;
    demand.forEach((_, zone) => {
      let nearest = Infinity;
      for (const facility of chosen) nearest = Math.min(nearest, distances[facility][zone]);
      total += nearest;
      worst = Math.max(worst, nearest);
      if (nearest <= coverageRadius) covered++;
    });
    return objective === 'median'
      ? total
      : objective === 'center'
        ? worst
        : -covered * 1000 + total;
  };
  const chosen: number[] = [];
  while (chosen.length < count) {
    let best = -1;
    let bestScore = Infinity;
    candidates.forEach((_, index) => {
      if (chosen.includes(index)) return;
      const value = score([...chosen, index]);
      if (value < bestScore) {
        bestScore = value;
        best = index;
      }
    });
    chosen.push(best);
  }
  return chosen;
}

const locationAllocation: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'count', label: 'Facilities (p)', min: 1, max: 5, step: 1, default: 3},
    {
      id: 'objective',
      label: 'Objective',
      min: 0,
      max: 2,
      step: 1,
      default: 0,
      options: ['p-median', 'p-center', 'Coverage']
    }
  ],
  build: ({fork, p}) => {
    const demand = createClusterPoints(
      fork('demand'),
      [
        [40, 36],
        [112, 34],
        [76, 74],
        [120, 76]
      ],
      8,
      9
    );
    const candidates = createScatterPoints(fork('candidates'), 11, 12);
    const objective: Objective = (['median', 'center', 'coverage'] as const)[
      Math.round(p.objective)
    ];
    const chosen = selectFacilities(candidates, demand, Math.round(p.count), objective, 26);
    const assignment: PathItem[] = [];
    const reach = chosen.map(() => 0);
    demand.forEach((point, zone) => {
      let nearest = 0;
      chosen.forEach((facility, index) => {
        if (
          getDistance(point, candidates[facility]) < getDistance(point, candidates[chosen[nearest]])
        )
          nearest = index;
      });
      const distance = getDistance(point, candidates[chosen[nearest]]);
      reach[nearest] = Math.max(reach[nearest], distance);
      assignment.push({
        points: [point, candidates[chosen[nearest]]],
        role: OWNER_ROLES[nearest],
        width: 0.6,
        alpha: objective === 'coverage' && distance > 26 ? 0.1 : 0.7,
        phase: zone / demand.length
      });
    });
    const rings = chosen
      .filter(() => objective !== 'median')
      .map((facility, index) => ({
        points: createCircle(
          candidates[facility][0],
          candidates[facility][1],
          objective === 'center' ? reach[index] : 26,
          48
        ),
        role: OWNER_ROLES[index],
        width: 0.7,
        alpha: 0.8
      }));
    return [
      createStrokes(rings, 'pulse', {speed: 0.3, blend: 'add'}),
      createStrokes(assignment, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(toSprites(demand, 'cool', 1.1, undefined, 0.85)),
      createSprites(toSprites(candidates, 'dim', 1.8, undefined, 0.5)),
      createSprites(
        chosen.map((facility, index) => ({
          x: candidates[facility][0],
          y: candidates[facility][1],
          radius: 4,
          role: OWNER_ROLES[index],
          phase: index / 5
        })),
        0.3
      )
    ];
  }
};

type SubsetMode = 'all' | 'batches';

/** Map on the left, one cell per candidate subset on the right. */
function buildSubsetScene(
  context: {fork: (label: string) => Random; p: Record<string, number>},
  mode: SubsetMode
): PreviewLayer[] {
  const {fork, p} = context;
  const demandRandom = fork('demand');
  const demand = Array.from(
    {length: 22},
    (): Point => [demandRandom.range(12, 92), demandRandom.range(14, 88)]
  );
  const candidateRandom = fork('candidates');
  const candidates = Array.from({length: 8}, (_, index): Point => {
    const angle = (index / 8) * Math.PI * 2 + 0.3;
    return [
      52 + Math.cos(angle) * 36 + candidateRandom.range(-5, 5),
      50 + Math.sin(angle) * 33 + candidateRandom.range(-4, 4)
    ];
  });
  const subsets = getCombinations(8, Math.round(p.size));
  const nearest = candidates.map(candidate => demand.map(point => getDistance(point, candidate)));
  const scores = subsets.map(subset => {
    let total = 0;
    let worst = 0;
    demand.forEach((_, zone) => {
      let closest = Infinity;
      for (const facility of subset) closest = Math.min(closest, nearest[facility][zone]);
      total += closest;
      worst = Math.max(worst, closest);
    });
    return mode === 'all' ? worst : total;
  });
  const minimum = Math.min(...scores);
  const maximum = Math.max(...scores);
  const best = scores.indexOf(minimum);
  const chosen = subsets[best];
  const assignment: PathItem[] = [];
  const load = chosen.map(() => 0);
  demand.forEach((point, zone) => {
    let target = 0;
    chosen.forEach((facility, index) => {
      if (nearest[facility][zone] < nearest[chosen[target]][zone]) target = index;
    });
    load[target]++;
    assignment.push({
      points: [point, candidates[chosen[target]]],
      role: 'warm',
      width: 0.5,
      alpha: 0.6,
      phase: zone / demand.length
    });
  });
  const layers: PreviewLayer[] = [
    createStrokes(assignment, 'flow', {speed: 0.3, blend: 'add'}),
    createSprites(toSprites(demand, 'cool', 1.1, undefined, 0.8)),
    createSprites(toSprites(candidates, 'dim', 1.8, undefined, 0.6))
  ];
  const cells: PolygonItem[] = [];
  const extras: SpriteItem[] = [];
  const snake: Point[] = [];
  const total = subsets.length;
  if (mode === 'all') {
    const columns = 7;
    const rows = Math.ceil(total / columns);
    const pitch = 5.4;
    const originX = 114;
    const originY = 50 - (rows * pitch) / 2;
    subsets.forEach((_, index) => {
      const x = originX + (index % columns) * pitch;
      const y = originY + Math.floor(index / columns) * pitch;
      const value = (scores[index] - minimum) / (maximum - minimum || 1);
      cells.push({
        ring: createRectangle(x, y, pitch - 1, pitch - 1),
        role: index === best ? 'warm' : getHeatRole(Math.min(0.99, value)),
        alpha: index === best ? 1 : 0.8,
        phase: (index * 0.61) % 1
      });
      if (index === best) extras.push({x: x + 2, y: y + 2, radius: 3.2, role: 'ink'});
    });
  } else {
    const batch = Math.round(p.batch);
    const rows = Math.ceil(total / batch);
    const pitch = Math.min(5.4, 36 / batch, 82 / rows);
    const originX = 112;
    const originY = 50 - (rows * pitch) / 2;
    for (let row = 0; row < rows; row++) {
      const first = row * batch;
      const count = Math.min(batch, total - first);
      let rowBest = first;
      for (let column = 0; column < count; column++) {
        const index = first + column;
        if (scores[index] < scores[rowBest]) rowBest = index;
        const value = (scores[index] - minimum) / (maximum - minimum || 1);
        cells.push({
          ring: createRectangle(
            originX + column * pitch,
            originY + row * pitch,
            pitch * 0.8,
            pitch * 0.8
          ),
          role: index === best ? 'warm' : getHeatRole(Math.min(0.99, value)),
          alpha: index === best ? 1 : 0.75,
          phase: (row * 0.13 + column * 0.04) % 1
        });
      }
      const y = originY + row * pitch + pitch * 0.4;
      const ends: [number, number] = [originX - 1, originX + batch * pitch];
      snake.push([row % 2 === 0 ? ends[0] : ends[1], y], [row % 2 === 0 ? ends[1] : ends[0], y]);
      extras.push({
        x: originX + batch * pitch + 2.2,
        y,
        radius: rowBest === best ? 2.4 : 1.2,
        role: rowBest === best ? 'ink' : 'hot'
      });
    }
  }
  layers.push(
    createSprites(
      chosen.map((facility, index) => ({
        x: candidates[facility][0],
        y: candidates[facility][1],
        radius: mode === 'batches' ? 2.8 + Math.sqrt(load[index]) * 1.2 : 4,
        role: mode === 'batches' ? 'violet' : 'warm',
        phase: index / 4
      })),
      0.3
    ),
    createFills(cells, 'pulse')
  );
  if (mode === 'batches') {
    layers.push(
      createStrokes([{points: snake, role: 'primary', width: 0.6, alpha: 0.9}], 'draw', {
        speed: 0.18,
        blend: 'add'
      })
    );
  }
  layers.push(createSprites(extras, 0.3));
  return layers;
}

const boundedFacilityAllocation: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'size', label: 'Facilities (p)', min: 2, max: 4, step: 1, default: 3}],
  build: context => buildSubsetScene(context, 'all')
};

const networkFacilityOptimization: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'size', label: 'Facilities (p)', min: 2, max: 4, step: 1, default: 4},
    {id: 'batch', label: 'Batch size', min: 4, max: 14, step: 1, default: 8}
  ],
  build: context => buildSubsetScene(context, 'batches')
};

const facilityScenario: EntryPreview = {
  pointer: 'sticky',
  parameters: [
    {id: 'coverage', label: 'Coverage cutoff', min: 20, max: 100, step: 5, default: 55, unit: '%'}
  ],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const sites = [pickNode(net, pointer, 3, 4), getNodeIndex(net, 1, 1), getNodeIndex(net, 4, 7)];
    const result = runSearch(createAdjacency(net), sites);
    const maximum = getMaxFinite(result.distances);
    const cutoff = maximum * (p.coverage / 100);
    const tree: PathItem[] = [];
    const halos: SpriteItem[] = [];
    net.nodes.forEach(([x, y], node) => {
      if (result.previous[node] >= 0) {
        tree.push({
          points: [net.nodes[result.previous[node]], [x, y]],
          role: getHeatRole(result.distances[node] / maximum),
          width: 0.9,
          alpha: 0.8,
          phase: result.distances[node] / maximum
        });
      }
      if (result.distances[node] <= cutoff) {
        halos.push({x, y, radius: 3, role: 'warm', alpha: 0.28, phase: node / 54});
      }
    });
    return [
      createBackdrop(net, {alpha: 0.3}),
      createSprites(halos, 0.4),
      createStrokes(tree, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1.4,
          role:
            result.distances[node] <= cutoff
              ? getHeatRole(result.distances[node] / maximum)
              : 'dim',
          alpha: result.distances[node] <= cutoff ? 0.95 : 0.45
        }))
      ),
      createSprites(
        sites.map((node, index) => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: index === 0 ? 5 : 3.8,
          role: index === 0 ? 'warm' : 'ink',
          phase: index / 3
        })),
        0.3
      )
    ];
  }
};

const networkDisruption: EntryPreview = {
  pointer: 'sticky',
  parameters: [{id: 'closures', label: 'Closed roads', min: 1, max: 10, step: 1, default: 4}],
  build: ({random, fork, p, pointer}) => {
    const net = createRoadNetwork(random);
    const source = pickNode(net, pointer, 3, 0);
    const target = getNodeIndex(net, 2, net.columns - 1);
    const baseline = runSearch(createAdjacency(net), [source]);
    const counts = getSubtreeCounts(baseline);
    const jitter = fork('closures');
    const closed = new Set(
      baseline.order
        .filter(node => baseline.previousEdge[node] >= 0)
        .map(node => ({
          edge: baseline.previousEdge[node],
          load: counts[node] * (1 + jitter.next() * 0.6)
        }))
        .sort((first, second) => second.load - first.load)
        .slice(0, Math.round(p.closures))
        .map(entry => entry.edge)
    );
    const disrupted = runSearch(createAdjacency(net, closed), [source]);
    const delta = net.nodes.map((_, node) => disrupted.distances[node] - baseline.distances[node]);
    const maximumDelta = Math.max(...delta.filter(value => value < Infinity), 1);
    const detour =
      disrupted.distances[target] < Infinity
        ? getPathPoints(net, getPathTo(disrupted.previous, target))
        : [];
    return [
      createBackdrop(net, {closed, alpha: 0.4}),
      createStrokes(toPaths(getEdgePaths(net, closed), 'hot', 1.2, 0.8), 'pulse', {
        speed: 0.5,
        blend: 'add'
      }),
      createStrokes(
        [
          {
            points: getPathPoints(net, getPathTo(baseline.previous, target)),
            role: 'cool',
            width: 1.4,
            alpha: 0.55
          }
        ],
        'static'
      ),
      createStrokes([{points: detour, role: 'warm', width: 2.2, alpha: 1}], 'flow', {
        speed: 0.3,
        blend: 'add'
      }),
      createSprites(
        net.nodes.map(([x, y], node) =>
          delta[node] === Infinity
            ? {x, y, radius: 3, role: 'hot' as Role, alpha: 1, phase: node / 54}
            : {
                x,
                y,
                radius: 1.2 + 3.2 * (delta[node] / maximumDelta),
                role: delta[node] > 1 ? ('warm' as Role) : ('dim' as Role),
                alpha: delta[node] > 1 ? 0.85 : 0.4,
                phase: node / 54
              }
        ),
        0.4
      ),
      createSprites(
        [
          {x: net.nodes[source][0], y: net.nodes[source][1], radius: 4, role: 'ink'},
          {x: net.nodes[target][0], y: net.nodes[target][1], radius: 4, role: 'ink'}
        ],
        0.3
      )
    ];
  }
};

const driveTimeCatchment: EntryPreview = {
  pointer: 'hover',
  parameters: [{id: 'bands', label: 'Time bands', min: 2, max: 7, step: 1, default: 4}],
  build: ({random, fork, p, pointer}) => {
    const net = createRoadNetwork(random);
    const sites = [pickNode(net, pointer, 2, 2), getNodeIndex(net, 3, 7)];
    const result = runSearch(createAdjacency(net), sites);
    const demandRandom = fork('demand');
    const bands = Math.round(p.bands);
    const bandWidth = 120 / bands;
    const stubs: PathItem[] = [];
    const dots: SpriteItem[] = [];
    for (let index = 0; index < 34; index++) {
      const point: Point = [demandRandom.range(10, 150), demandRandom.range(10, 90)];
      const node = findNearestIndex(net.nodes, point);
      const cost = result.distances[node] + getDistance(point, net.nodes[node]);
      const band = Math.floor(cost / bandWidth);
      const inside = band < bands;
      stubs.push({
        points: [point, net.nodes[node]],
        role: inside ? getClassRole(band, bands) : 'dim',
        width: 0.35,
        alpha: inside ? 0.5 : 0.15
      });
      dots.push({
        x: point[0],
        y: point[1],
        radius: 1.7,
        role: inside ? getClassRole(band, bands) : 'dim',
        alpha: inside ? 0.95 : 0.35,
        phase: band / bands
      });
    }
    return [
      createBackdrop(net, {alpha: 0.4}),
      createStrokes(
        net.nodes
          .map((_, node) => node)
          .filter(node => result.previous[node] >= 0 && result.distances[node] < bandWidth * bands)
          .map(node => ({
            points: [net.nodes[result.previous[node]], net.nodes[node]] as Point[],
            role: getClassRole(Math.floor(result.distances[node] / bandWidth), bands),
            width: 0.9,
            alpha: 0.7,
            phase: result.distances[node] / 120
          })),
        'flow',
        {speed: 0.3, blend: 'add'}
      ),
      createStrokes(stubs),
      createSprites(dots, 0.3),
      createSprites(
        sites.map(node => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 4.4,
          role: 'ink' as Role
        })),
        0.3
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Snapping, map matching, track quality

const networkSnapping: EntryPreview = {
  pointer: 'hover',
  parameters: [
    {id: 'maximum', label: 'Snap distance', min: 3, max: 30, step: 1, default: 14, unit: 'px'}
  ],
  build: ({random, fork, p, pointer}) => {
    const net = createRoadNetwork(random);
    const pointRandom = fork('points');
    const points: Point[] = Array.from({length: 24}, () => {
      const [first, second] = net.edges[pointRandom.integer(net.edges.length)];
      const amount = pointRandom.next();
      const angle = pointRandom.range(0, Math.PI * 2);
      const offset = pointRandom.range(2, 18);
      return clampPoint(
        [
          mixNumbers(net.nodes[first][0], net.nodes[second][0], amount) + Math.cos(angle) * offset,
          mixNumbers(net.nodes[first][1], net.nodes[second][1], amount) + Math.sin(angle) * offset
        ],
        3
      );
    });
    const probe: Point = pointer ?? [74, 44];
    const all = [...points, probe];
    const stubs: PathItem[] = [];
    const snaps: SpriteItem[] = [];
    const rejected: SpriteItem[] = [];
    const usedEdges = new Set<number>();
    all.forEach((point, index) => {
      const snap = findNearestSegmentPoint(net, point);
      const isProbe = index === points.length;
      if (snap.distance <= p.maximum) {
        usedEdges.add(snap.edge);
        stubs.push({
          points: [point, snap.point],
          role: isProbe ? 'ink' : 'warm',
          width: isProbe ? 0.9 : 0.6,
          alpha: 0.95,
          phase: index / all.length
        });
        snaps.push({
          x: snap.point[0],
          y: snap.point[1],
          radius: isProbe ? 2.6 : 1.7,
          role: 'warm',
          phase: index / all.length
        });
      } else {
        rejected.push({x: point[0], y: point[1], radius: 1.6, role: 'hot', alpha: 0.9});
      }
    });
    return [
      createBackdrop(net, {alpha: 0.35}),
      createStrokes(toPaths(getEdgePaths(net, usedEdges), 'primary', 0.9, 0.55)),
      createStrokes(stubs, 'draw', {speed: 0.35, blend: 'add'}),
      createSprites(snaps, 0.3),
      createSprites(toSprites(points, 'cool', 1.2, undefined, 0.85)),
      createSprites(rejected, 0.5),
      createSprites([{x: probe[0], y: probe[1], radius: 3.4, role: 'ink'}], 0.3)
    ];
  }
};

const mapMatching: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'noise', label: 'GPS noise', min: 0, max: 10, step: 0.5, default: 4.5, unit: 'px'}
  ],
  build: ({random, fork, p}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const start = getNodeIndex(net, 4, 0);
    const end = getNodeIndex(net, 1, net.columns - 1);
    const via = getNodeIndex(net, 3, 4);
    const first = runSearch(adjacency, [start]);
    const second = runSearch(adjacency, [via]);
    const route = [...getPathTo(first.previous, via), ...getPathTo(second.previous, end).slice(1)];
    const routePoints = getPathPoints(net, route);
    const truth = samplePolyline(routePoints, 7);
    const noiseRandom = fork('noise');
    const fixes = truth.map(
      ([x, y]): Point => [
        x + noiseRandom.gaussian() * p.noise,
        y + noiseRandom.gaussian() * p.noise
      ]
    );
    const candidates: SpriteItem[] = [];
    const residuals: PathItem[] = [];
    fixes.forEach((fix, index) => {
      const ranked = net.edges
        .map(([a, b]) => ({
          a,
          b,
          snap: findNearestSegmentPoint({...net, edges: [[a, b]]}, fix)
        }))
        .sort((x, y) => x.snap.distance - y.snap.distance)
        .slice(0, 3);
      ranked.slice(1).forEach(entry => {
        candidates.push({
          x: entry.snap.point[0],
          y: entry.snap.point[1],
          radius: 1,
          role: 'violet',
          alpha: 0.6
        });
      });
      residuals.push({
        points: [fix, truth[index]],
        role: 'dim',
        width: 0.35,
        alpha: 0.6
      });
    });
    return [
      createBackdrop(net, {alpha: 0.3}),
      createSprites(candidates, 0.2),
      createStrokes(residuals),
      createStrokes([{points: routePoints, role: 'warm', width: 2.2, alpha: 1}], 'flow', {
        speed: 0.25,
        trail: 1.4,
        blend: 'add'
      }),
      createStrokes([{points: fixes, role: 'cool', width: 0.5, alpha: 0.5}]),
      createSprites(toSprites(fixes, 'cool', 1.5, undefined, 0.95), 0.2)
    ];
  }
};

const trackQuality: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'limit', label: 'Speed limit', min: 5, max: 20, step: 0.5, default: 9, unit: 'px/s'}
  ],
  build: ({fork, p}) => {
    const trackRandom = fork('track');
    const phase = trackRandom.range(0, Math.PI);
    const wander = Array.from(
      {length: 35},
      (_, index): Point => [
        8 + index * 4.3,
        clamp(52 + 22 * Math.sin(index * 0.2 + phase) + trackRandom.range(-1.2, 1.2), 12, 88)
      ]
    );
    const sampled: {point: Point; interval: number}[] = [];
    wander.forEach((point, index) => {
      if (index >= 21 && index <= 24) return;
      const displaced: Point =
        index === 12
          ? [point[0] + 6, point[1] - 19]
          : index === 29
            ? [point[0] - 5, point[1] + 17]
            : point;
      sampled.push({point: displaced, interval: index === 25 ? 4 : 1});
    });
    const good: PathItem[] = [];
    const bad: PathItem[] = [];
    const gaps: PathItem[] = [];
    const flagged = new Set<number>();
    for (let index = 1; index < sampled.length; index++) {
      const points = [sampled[index - 1].point, sampled[index].point];
      const speed = getDistance(points[0], points[1]) / sampled[index].interval;
      const entry: PathItem = {
        points,
        role: 'cool',
        width: 1,
        alpha: 0.9,
        phase: index / sampled.length
      };
      if (sampled[index].interval > 2) {
        gaps.push({...entry, role: 'violet', width: 0.8, alpha: 0.8});
        flagged.add(index - 1);
        flagged.add(index);
      } else if (speed > p.limit) {
        bad.push({...entry, role: 'hot', width: 1.2});
        flagged.add(index - 1);
        flagged.add(index);
      } else {
        good.push(entry);
      }
    }
    return [
      createStrokes(good, 'flow', {speed: 0.3, blend: 'add'}),
      createStrokes(bad, 'pulse', {speed: 0.6, blend: 'add'}),
      createStrokes(gaps, 'pulse', {speed: 0.3, blend: 'add'}),
      createSprites(
        sampled.map(({point}, index) => ({
          x: point[0],
          y: point[1],
          radius: flagged.has(index) ? 2.6 : 1.4,
          role: flagged.has(index)
            ? sampled[index].interval > 2 || sampled[index + 1]?.interval > 2
              ? 'violet'
              : 'hot'
            : 'cool',
          alpha: 0.95,
          phase: index / sampled.length
        })),
        0.3
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Routing

const networkPathExtraction: EntryPreview = {
  pointer: 'sticky',
  parameters: [{id: 'targets', label: 'Targets', min: 1, max: 8, step: 1, default: 4}],
  build: ({random, fork, p, pointer}) => {
    const net = createRoadNetwork(random);
    const source = pickNode(net, pointer, 3, 1);
    const result = runSearch(createAdjacency(net), [source]);
    const targetRandom = fork('targets');
    const targets = Array.from({length: 8}, () => targetRandom.integer(net.nodes.length)).slice(
      0,
      Math.round(p.targets)
    );
    const tree: PathItem[] = [];
    net.nodes.forEach((point, node) => {
      if (result.previous[node] >= 0) {
        tree.push({
          points: [net.nodes[result.previous[node]], point],
          role: 'primary',
          width: 0.5,
          alpha: 0.4
        });
      }
    });
    const paths: PathItem[] = [];
    const ordered: SpriteItem[] = [];
    targets.forEach((target, index) => {
      const nodes = getPathTo(result.previous, target);
      paths.push({
        points: getPathPoints(net, nodes),
        role: index % 2 === 0 ? 'warm' : 'hot',
        width: 1.8,
        alpha: 0.95,
        phase: index / targets.length
      });
      nodes.forEach((node, position) => {
        ordered.push({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 1 + (position / Math.max(nodes.length - 1, 1)) * 1.8,
          role: index % 2 === 0 ? 'warm' : 'hot',
          alpha: 0.9,
          phase: position / Math.max(nodes.length, 1)
        });
      });
    });
    return [
      createBackdrop(net, {alpha: 0.25}),
      createStrokes(tree),
      createStrokes(paths, 'flow', {speed: 0.3, trail: 1.5, blend: 'add'}),
      createSprites(ordered, 0.6),
      createSprites(
        [{x: net.nodes[source][0], y: net.nodes[source][1], radius: 4.6, role: 'ink'}],
        0.3
      ),
      createSprites(
        targets.map(node => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 3.4,
          role: 'hot' as Role
        })),
        0.3
      )
    ];
  }
};

const alternativeRoutes: EntryPreview = {
  pointer: 'sticky',
  parameters: [
    {id: 'routes', label: 'Routes per query', min: 1, max: 4, step: 1, default: 3},
    {id: 'penalty', label: 'Overlap penalty', min: 0.3, max: 3, step: 0.1, default: 1.2}
  ],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const source = pickNode(net, pointer, 4, 0);
    const target = getNodeIndex(net, 1, net.columns - 1);
    const usage = new Array<number>(net.edges.length).fill(0);
    const seen = new Set<string>();
    const roles: Role[] = ['warm', 'primary', 'violet', 'cool'];
    const widths = [2.6, 1.8, 1.8, 1.8];
    const routes: StrokeLayer[] = [];
    for (let attempt = 0; attempt < 8 && routes.length < Math.round(p.routes); attempt++) {
      const result = runSearch(
        adjacency,
        [source],
        arc => arc.length * (1 + p.penalty * usage[arc.edge])
      );
      const nodes = getPathTo(result.previous, target);
      const key = nodes.join(',');
      for (let index = 1; index < nodes.length; index++) {
        const edge = adjacency[nodes[index - 1]].find(arc => arc.to === nodes[index])?.edge ?? 0;
        usage[edge] += 1.5;
      }
      if (seen.has(key)) continue;
      seen.add(key);
      routes.push(
        createStrokes(
          [
            {
              points: getPathPoints(net, nodes),
              role: roles[routes.length],
              width: widths[routes.length],
              alpha: 0.95,
              phase: routes.length * 0.25
            }
          ],
          'flow',
          {speed: 0.22 + routes.length * 0.04, trail: 1.3, blend: 'add'}
        )
      );
    }
    return [
      createBackdrop(net, {alpha: 0.4}),
      ...routes,
      createSprites(
        [
          {x: net.nodes[source][0], y: net.nodes[source][1], radius: 4.4, role: 'ink'},
          {x: net.nodes[target][0], y: net.nodes[target][1], radius: 4.4, role: 'ink'}
        ],
        0.3
      )
    ];
  }
};

type TurnKind = 'straight' | 'left' | 'right' | 'uturn';

function classifyTurn(previous: Point, node: Point, next: Point): TurnKind {
  const ax = node[0] - previous[0];
  const ay = node[1] - previous[1];
  const bx = next[0] - node[0];
  const by = next[1] - node[1];
  const length = Math.hypot(ax, ay) * Math.hypot(bx, by) || 1;
  const cosine = (ax * bx + ay * by) / length;
  const sine = (ax * by - ay * bx) / length;
  if (cosine < -0.9) return 'uturn';
  if (Math.abs(sine) < 0.45) return 'straight';
  return sine < 0 ? 'left' : 'right';
}

/** Dijkstra over (previous node, node) states, so turns can cost something or be banned. */
function findTurnRoute(
  net: RoadNetwork,
  adjacency: Arc[][],
  source: number,
  target: number,
  getTurnCost: (kind: TurnKind) => number
): number[] {
  const count = net.nodes.length;
  const distances = new Float64Array(count * count).fill(Infinity);
  const parents = new Int32Array(count * count).fill(-1);
  const heap = createHeap();
  const start = source * count + source;
  distances[start] = 0;
  heap.push(0, start);
  while (heap.size > 0) {
    const [distance, state] = heap.pop();
    if (distance > distances[state]) continue;
    const node = state % count;
    const previous = (state - node) / count;
    if (node === target) {
      const nodes: number[] = [];
      for (let cursor = state; cursor >= 0; cursor = parents[cursor]) {
        nodes.unshift(cursor % count);
        if (parents[cursor] === cursor) break;
      }
      return nodes.filter((entry, index) => index === 0 || entry !== nodes[index - 1]);
    }
    for (const arc of adjacency[node]) {
      const turn =
        previous === node
          ? 0
          : getTurnCost(classifyTurn(net.nodes[previous], net.nodes[node], net.nodes[arc.to]));
      if (turn === Infinity) continue;
      const next = node * count + arc.to;
      const cost = distance + arc.length + turn;
      if (cost < distances[next]) {
        distances[next] = cost;
        parents[next] = state;
        heap.push(cost, next);
      }
    }
  }
  return [];
}

const networkPointRouting: EntryPreview = {
  pointer: 'hover',
  parameters: [
    {id: 'penalty', label: 'Left-turn penalty', min: 0, max: 80, step: 2, default: 34, unit: 'px'}
  ],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const pairs: [number, number][] = [
      [pickNode(net, pointer, 1, 0), getNodeIndex(net, 4, 8)],
      [getNodeIndex(net, 5, 1), getNodeIndex(net, 0, 7)],
      [getNodeIndex(net, 3, 0), getNodeIndex(net, 1, 8)]
    ];
    const roles: Role[] = ['warm', 'primary', 'violet'];
    const costs = (kind: TurnKind) =>
      kind === 'uturn'
        ? Infinity
        : kind === 'left'
          ? p.penalty
          : kind === 'right'
            ? p.penalty * 0.2
            : 0;
    const plain: PathItem[] = [];
    const aware: PathItem[] = [];
    const turns: SpriteItem[] = [];
    pairs.forEach(([source, target], index) => {
      const free = findTurnRoute(net, adjacency, source, target, () => 0);
      plain.push({points: getPathPoints(net, free), role: 'dim', width: 1.6, alpha: 0.6});
      const route = findTurnRoute(net, adjacency, source, target, costs);
      aware.push({
        points: getPathPoints(net, route),
        role: roles[index],
        width: 1.8,
        alpha: 0.95,
        phase: index / 3
      });
      for (let step = 1; step + 1 < route.length; step++) {
        if (
          classifyTurn(
            net.nodes[route[step - 1]],
            net.nodes[route[step]],
            net.nodes[route[step + 1]]
          ) !== 'straight'
        ) {
          turns.push({
            x: net.nodes[route[step]][0],
            y: net.nodes[route[step]][1],
            radius: 2.2,
            role: 'ink',
            alpha: 0.9,
            phase: step / route.length
          });
        }
      }
    });
    return [
      createBackdrop(net, {alpha: 0.35}),
      createStrokes(plain),
      createStrokes(aware, 'flow', {speed: 0.25, trail: 1.3, blend: 'add'}),
      createSprites(turns, 0.4),
      createSprites(
        pairs.flatMap(([source, target], index) => [
          {
            x: net.nodes[source][0],
            y: net.nodes[source][1],
            radius: 3.8,
            role: roles[index],
            phase: 0
          },
          {
            x: net.nodes[target][0],
            y: net.nodes[target][1],
            radius: 3.8,
            role: roles[index],
            phase: 0.5
          }
        ]),
        0.3
      )
    ];
  }
};

const networkTimeRouting: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'depart', label: 'Departure time', min: 0, max: 1, step: 0.02, default: 0.45}],
  build: ({random, p}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const source = getNodeIndex(net, 3, 0);
    const target = getNodeIndex(net, 2, net.columns - 1);
    // A congestion blob drifts east across the middle corridor, so the earliest-arrival route
    // must bend around it while the free-flow route runs straight through.
    const getCenter = (time: number) => -10 + (time % 1.4) * 110;
    const corridorY = (net.nodes[source][1] + net.nodes[target][1]) / 2;
    const getFactor = (edge: number, time: number) => {
      const [first, second] = net.edges[edge];
      const middleX = (net.nodes[first][0] + net.nodes[second][0]) / 2;
      const middleY = (net.nodes[first][1] + net.nodes[second][1]) / 2;
      const across = (middleX - getCenter(time)) / 24;
      const along = (middleY - corridorY) / 20;
      return 1 + 7 * Math.exp(-(across * across + along * along));
    };
    const timed = runSearch(
      adjacency,
      [source],
      (arc, _from, elapsed) => arc.length * getFactor(arc.edge, p.depart + elapsed / 160)
    );
    const free = runSearch(adjacency, [source]);
    const slow: PathItem[] = [];
    net.edges.forEach((_, edge) => {
      const factor = getFactor(edge, p.depart);
      if (factor > 1.35) {
        slow.push({
          points: getEdgePaths(net, [edge])[0],
          role: 'hot',
          width: 0.5 + factor * 0.25,
          alpha: Math.min(0.9, 0.25 + (factor - 1) * 0.3)
        });
      }
    });
    const centerX = getCenter(p.depart);
    return [
      createBackdrop(net, {alpha: 0.4}),
      createStrokes(slow, 'pulse', {speed: 0.5, blend: 'add'}),
      createStrokes(
        [
          {
            points: [
              [centerX, 4],
              [centerX, 96]
            ],
            role: 'hot',
            width: 0.5,
            alpha: 0.25
          }
        ],
        'pulse',
        {speed: 0.3, blend: 'add'}
      ),
      createStrokes([
        {
          points: getPathPoints(net, getPathTo(free.previous, target)),
          role: 'cool',
          width: 1.3,
          alpha: 0.5
        }
      ]),
      createStrokes(
        [
          {
            points: getPathPoints(net, getPathTo(timed.previous, target)),
            role: 'warm',
            width: 2.4,
            alpha: 1
          }
        ],
        'flow',
        {speed: 0.25, trail: 1.4, blend: 'add'}
      ),
      createSprites(
        [
          {x: net.nodes[source][0], y: net.nodes[source][1], radius: 4.2, role: 'ink'},
          {x: net.nodes[target][0], y: net.nodes[target][1], radius: 4.2, role: 'ink'}
        ],
        0.3
      )
    ];
  }
};

const networkLineGraph: EntryPreview = {
  pointer: 'hover',
  parameters: [{id: 'hops', label: 'Junction radius', min: 0, max: 2, step: 1, default: 1}],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const focus = pickNode(net, pointer, 3, 4);
    const reach = runSearch(
      adjacency.map(arcs => arcs.map(arc => ({...arc, length: 1}))),
      [focus]
    );
    const junctions = net.nodes
      .map((_, node) => node)
      .filter(node => reach.distances[node] <= Math.round(p.hops));
    const middle = (edge: number): Point => {
      const [first, second] = net.edges[edge];
      return [
        (net.nodes[first][0] + net.nodes[second][0]) / 2,
        (net.nodes[first][1] + net.nodes[second][1]) / 2
      ];
    };
    const transitions: PathItem[] = [];
    for (const junction of junctions) {
      for (const incoming of adjacency[junction]) {
        for (const outgoing of adjacency[junction]) {
          const kindReversed = classifyTurn(
            net.nodes[incoming.to],
            net.nodes[junction],
            net.nodes[outgoing.to]
          );
          transitions.push({
            points: smoothPath(
              [middle(incoming.edge), net.nodes[junction], middle(outgoing.edge)],
              3
            ),
            role: kindReversed === 'uturn' ? 'hot' : kindReversed === 'left' ? 'warm' : 'violet',
            width: kindReversed === 'uturn' ? 0.6 : 1.4,
            alpha: kindReversed === 'uturn' ? 0.35 : 0.85,
            phase: (incoming.edge * 0.13 + outgoing.edge * 0.29) % 1
          });
        }
      }
    }
    return [
      createBackdrop(net, {alpha: 0.35}),
      createStrokes(
        transitions.filter(entry => entry.role === 'hot'),
        'pulse',
        {speed: 0.5, blend: 'add'}
      ),
      createStrokes(
        transitions.filter(entry => entry.role !== 'hot'),
        'flow',
        {speed: 0.3, trail: 0.8, blend: 'add'}
      ),
      createSprites(
        net.edges.map((_, edge) => ({
          x: middle(edge)[0],
          y: middle(edge)[1],
          radius: 1.6,
          role: 'violet' as Role,
          alpha: 0.75,
          phase: edge / net.edges.length
        })),
        0.3
      ),
      createSprites(
        [{x: net.nodes[focus][0], y: net.nodes[focus][1], radius: 3.4, role: 'ink'}],
        0.3
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Network construction and analytics

type Segment = {start: Point; end: Point};

function intersectSegments(first: Segment, second: Segment): [number, number] | null {
  const ax = first.end[0] - first.start[0];
  const ay = first.end[1] - first.start[1];
  const bx = second.end[0] - second.start[0];
  const by = second.end[1] - second.start[1];
  const denominator = ax * by - ay * bx;
  if (Math.abs(denominator) < 1e-9) return null;
  const dx = second.start[0] - first.start[0];
  const dy = second.start[1] - first.start[1];
  const along = (dx * by - dy * bx) / denominator;
  const across = (dx * ay - dy * ax) / denominator;
  return along > 0.01 && along < 0.99 && across > 0.01 && across < 0.99 ? [along, across] : null;
}

const networkNoding: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'lines', label: 'Raw lines', min: 3, max: 16, step: 1, default: 8}],
  build: ({fork, p}) => {
    const lineRandom = fork('lines');
    const segments: Segment[] = Array.from({length: 16}, (_, index): Segment => {
      const horizontal = index % 2 === 0;
      return horizontal
        ? {
            start: [lineRandom.range(6, 30), lineRandom.range(10, 90)],
            end: [lineRandom.range(120, 154), lineRandom.range(10, 90)]
          }
        : {
            start: [lineRandom.range(20, 140), lineRandom.range(6, 22)],
            end: [lineRandom.range(20, 140), lineRandom.range(78, 94)]
          };
    }).slice(0, Math.round(p.lines));
    const cuts: number[][] = segments.map(() => []);
    const joints: Point[] = [];
    for (let first = 0; first < segments.length; first++) {
      for (let second = first + 1; second < segments.length; second++) {
        const hit = intersectSegments(segments[first], segments[second]);
        if (!hit) continue;
        cuts[first].push(hit[0]);
        cuts[second].push(hit[1]);
        joints.push([
          mixNumbers(segments[first].start[0], segments[first].end[0], hit[0]),
          mixNumbers(segments[first].start[1], segments[first].end[1], hit[0])
        ]);
      }
    }
    const pieces: PathItem[] = [];
    segments.forEach((segment, index) => {
      const stops = [0, ...cuts[index].sort((a, b) => a - b), 1];
      for (let piece = 0; piece + 1 < stops.length; piece++) {
        const length = getDistance(segment.start, segment.end);
        const shrink = Math.min(1.8 / length, (stops[piece + 1] - stops[piece]) / 3);
        const from = stops[piece] + (piece === 0 ? 0 : shrink);
        const to = stops[piece + 1] - (piece + 2 === stops.length ? 0 : shrink);
        pieces.push({
          points: [
            [
              mixNumbers(segment.start[0], segment.end[0], from),
              mixNumbers(segment.start[1], segment.end[1], from)
            ],
            [
              mixNumbers(segment.start[0], segment.end[0], to),
              mixNumbers(segment.start[1], segment.end[1], to)
            ]
          ],
          role: OWNER_ROLES[(index + piece) % 4],
          width: 1.3,
          alpha: 0.95,
          phase: ((index + piece) * 0.19) % 1
        });
      }
    });
    return [
      createStrokes(
        segments.map(segment => ({
          points: [segment.start, segment.end],
          role: 'dim' as Role,
          width: 2.6,
          alpha: 0.18
        }))
      ),
      createStrokes(pieces, 'pulse', {speed: 0.25, blend: 'add'}),
      createSprites(toSprites(joints, 'ink', 2.2, undefined, 0.95), 0.5)
    ];
  }
};

const networkCentrality: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'pivots', label: 'Pivots', min: 3, max: 54, step: 1, default: 16}],
  build: ({random, fork, p}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const pivots = shuffleIndices(fork('pivots'), net.nodes.length).slice(0, Math.round(p.pivots));
    const betweenness = new Array<number>(net.nodes.length).fill(0);
    const closeness = new Array<number>(net.nodes.length).fill(0);
    const load = new Array<number>(net.edges.length).fill(0);
    for (const pivot of pivots) {
      const result = runSearch(adjacency, [pivot]);
      const counts = getSubtreeCounts(result);
      result.order.forEach(node => {
        if (node !== pivot) {
          betweenness[node] += counts[node] - 1;
          load[result.previousEdge[node]] += counts[node];
        }
        closeness[node] += result.distances[node];
      });
    }
    const maximum = Math.max(...betweenness, 1);
    const maximumLoad = Math.max(...load, 1);
    const farthest = Math.max(...closeness, 1);
    const nearest = Math.min(...closeness);
    const edges: PathItem[] = [];
    load.forEach((value, edge) => {
      if (value / maximumLoad > 0.3) {
        edges.push({
          points: getEdgePaths(net, [edge])[0],
          role: 'warm',
          width: 0.6 + (value / maximumLoad) * 2.4,
          alpha: 0.9,
          phase: (edge * 0.37) % 1
        });
      }
    });
    return [
      createBackdrop(net, {alpha: 0.4}),
      createStrokes(edges, 'flow', {speed: 0.28, blend: 'add'}),
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1 + 5 * Math.sqrt(betweenness[node] / maximum),
          role: getHeatRole(
            Math.min(0.99, 1 - (closeness[node] - nearest) / (farthest - nearest || 1))
          ),
          alpha: 0.55 + 0.45 * Math.sqrt(betweenness[node] / maximum),
          phase: node / net.nodes.length
        })),
        0.25
      )
    ];
  }
};

function findCutVertices(net: RoadNetwork, closed: ReadonlySet<number>) {
  const adjacency = createAdjacency(net, closed);
  const count = net.nodes.length;
  const discovery = new Array<number>(count).fill(-1);
  const low = new Array<number>(count).fill(0);
  const component = new Array<number>(count).fill(-1);
  const bridges = new Set<number>();
  const articulation = new Set<number>();
  let clock = 0;
  let components = 0;
  const visit = (node: number, parentEdge: number, label: number) => {
    discovery[node] = low[node] = clock++;
    component[node] = label;
    let children = 0;
    for (const arc of adjacency[node]) {
      if (arc.edge === parentEdge) continue;
      if (discovery[arc.to] >= 0) {
        low[node] = Math.min(low[node], discovery[arc.to]);
      } else {
        children++;
        visit(arc.to, arc.edge, label);
        low[node] = Math.min(low[node], low[arc.to]);
        if (low[arc.to] > discovery[node]) bridges.add(arc.edge);
        if (parentEdge >= 0 && low[arc.to] >= discovery[node]) articulation.add(node);
      }
    }
    if (parentEdge < 0 && children > 1) articulation.add(node);
  };
  for (let node = 0; node < count; node++) {
    if (discovery[node] < 0) visit(node, -1, components++);
  }
  return {component, components, bridges, articulation};
}

const networkStructure: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'removed', label: 'Removed links', min: 0, max: 34, step: 1, default: 14}],
  build: ({random, fork, p}) => {
    const net = createRoadNetwork(random);
    const closed = pickClosedEdges(fork('removed'), net.edges.length, p.removed);
    const {component, bridges, articulation} = findCutVertices(net, closed);
    const sizes = new Map<number, number>();
    component.forEach(label => sizes.set(label, (sizes.get(label) ?? 0) + 1));
    const ranked = [...sizes.entries()]
      .sort((first, second) => second[1] - first[1])
      .map(entry => entry[0]);
    const roleOf = (node: number): Role =>
      (['primary', 'violet', 'cool', 'warm'] as const)[
        Math.min(ranked.indexOf(component[node]), 3)
      ];
    const open = getAllEdges(net).filter(edge => !closed.has(edge));
    return [
      createStrokes(
        toPaths(
          getEdgePaths(
            net,
            open.filter(edge => !bridges.has(edge))
          ),
          'dim',
          0.7,
          0.55
        )
      ),
      createStrokes(toPaths(getEdgePaths(net, [...bridges]), 'hot', 1.8, 0.95), 'pulse', {
        speed: 0.5,
        blend: 'add'
      }),
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1.7,
          role: roleOf(node),
          alpha: 0.9,
          phase: node / 54
        })),
        0.2
      ),
      createStrokes(
        [...articulation].map(node => ({
          points: createCircle(net.nodes[node][0], net.nodes[node][1], 3.8, 20),
          role: 'ink' as Role,
          width: 0.9,
          alpha: 0.95
        })),
        'pulse',
        {speed: 0.4, blend: 'add'}
      )
    ];
  }
};

const networkStatistics: EntryPreview = {
  pointer: 'none',
  build: ({random}) => {
    const net = createRoadNetwork(random);
    const placed = placeNet(net, {x: 4, y: 8, width: 94, height: 84});
    const degree = new Array<number>(net.nodes.length).fill(0);
    net.edges.forEach(([first, second]) => {
      degree[first]++;
      degree[second]++;
    });
    const histogram = [0, 0, 0, 0, 0];
    degree.forEach(value => histogram[Math.min(value, 4)]++);
    const peak = Math.max(...histogram);
    const centers: Point[] = [
      [20, 20],
      [60, 70],
      [84, 30]
    ].map(([x, y]) => [x, y]);
    const sizes = [0, 0, 0];
    placed.nodes.forEach(point => {
      let best = 0;
      centers.forEach((center, index) => {
        if (getDistance(point, center) < getDistance(point, centers[best])) best = index;
      });
      sizes[best]++;
    });
    const bars: PolygonItem[] = histogram.slice(1).map((count, index) => {
      const height = (count / peak) * 46;
      return {
        ring: createRectangle(108 + index * 11, 90 - height, 8, height),
        role: getClassRole(index, 4),
        alpha: 0.9,
        phase: index / 4
      };
    });
    return [
      createBackdrop(placed, {alpha: 0.45}),
      createSprites(
        placed.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1.2 + degree[node] * 0.35,
          role: getClassRole(degree[node] - 1, 4),
          alpha: 0.9,
          phase: node / 54
        })),
        0.3
      ),
      createFills(bars, 'pulse'),
      createStrokes([
        {
          points: [
            [104, 91],
            [154, 91]
          ],
          role: 'dim',
          width: 0.5,
          alpha: 0.8
        }
      ]),
      createSprites(
        sizes.map((size, index) => ({
          x: 118 + index * 14,
          y: 16,
          radius: 1.4 + Math.sqrt(size) * 0.7,
          role: OWNER_ROLES[index],
          phase: index / 3
        })),
        0.3
      )
    ];
  }
};

const networkAnalyticsColumns: EntryPreview = {
  pointer: 'hover',
  build: ({random, fork, pointer}) => {
    const net = createRoadNetwork(random);
    const placed = placeNet(net, {x: 4, y: 8, width: 100, height: 84});
    const orientation = fork('orientation');
    const degree = new Array<number>(net.nodes.length).fill(0);
    const inDegree = new Array<number>(net.nodes.length).fill(0);
    net.edges.forEach(([first, second]) => {
      degree[first]++;
      degree[second]++;
      inDegree[orientation.next() > 0.5 ? first : second]++;
    });
    const focus = pointer ? findNearestIndex(placed.nodes, pointer) : -1;
    const columns = [
      (node: number) => degree[node] / 4,
      (node: number) => inDegree[node] / 3,
      (node: number) => (node % 9 < 5 ? 0.15 : 0.7)
    ];
    const cells: PolygonItem[] = [];
    const marks: SpriteItem[] = [];
    columns.forEach((getValue, column) => {
      net.nodes.forEach((_, node) => {
        const x = 114 + (node % 9) * 4.4;
        const y = 10 + column * 29 + Math.floor(node / 9) * 4.4;
        const value = getValue(node);
        cells.push({
          ring: createRectangle(x, y, 3.6, 3.6),
          role:
            column === 2
              ? value > 0.5
                ? 'violet'
                : 'primary'
              : getHeatRole(Math.min(0.99, value)),
          alpha: 0.85,
          phase: (node * 0.07 + column * 0.3) % 1
        });
        if (node === focus) marks.push({x: x + 1.8, y: y + 1.8, radius: 2.4, role: 'ink'});
      });
    });
    return [
      createBackdrop(placed, {alpha: 0.4}),
      createSprites(
        placed.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1 + degree[node] * 0.5,
          role: getHeatRole(Math.min(0.99, degree[node] / 4)),
          alpha: 0.9,
          phase: node / 54
        })),
        0.3
      ),
      createFills(cells, 'pulse'),
      createSprites(
        focus >= 0
          ? [
              {
                x: placed.nodes[focus][0],
                y: placed.nodes[focus][1],
                radius: 4,
                role: 'ink' as Role
              },
              ...marks
            ]
          : [],
        0.3
      )
    ];
  }
};

const networkNeighborhood: EntryPreview = {
  pointer: 'sticky',
  parameters: [{id: 'hops', label: 'Hops (k)', min: 0, max: 6, step: 1, default: 3}],
  build: ({random, p, pointer}) => {
    const net = createRoadNetwork(random);
    const source = pickNode(net, pointer, 3, 4);
    const hops = Math.round(p.hops);
    const result = runSearch(
      createAdjacency(net).map(arcs => arcs.map(arc => ({...arc, length: 1}))),
      [source]
    );
    const outward: PathItem[] = [];
    const inside = new Set<number>();
    net.edges.forEach(([first, second], edge) => {
      const near = Math.min(result.distances[first], result.distances[second]);
      const far = Math.max(result.distances[first], result.distances[second]);
      if (far <= hops) {
        inside.add(edge);
        const from = result.distances[first] <= result.distances[second] ? first : second;
        const to = from === first ? second : first;
        outward.push({
          points: [net.nodes[from], net.nodes[to]],
          role: getClassRole(near, hops + 1),
          width: 1.3,
          alpha: 0.9,
          phase: near / (hops + 1)
        });
      }
    });
    return [
      createBackdrop(net, {alpha: 0.3}),
      createStrokes(outward, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(
        net.nodes.map(([x, y], node) => {
          const hop = result.distances[node];
          return hop <= hops
            ? {
                x,
                y,
                radius: node === source ? 4.6 : 2.2,
                role: node === source ? ('ink' as Role) : getClassRole(hop, hops + 1),
                alpha: 0.95,
                phase: hop / (hops + 1)
              }
            : {x, y, radius: 1, role: 'dim' as Role, alpha: 0.4};
        }),
        0.4
      )
    ];
  }
};

const networkSubgraphFilter: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'minimum', label: 'Attribute min', min: 0, max: 1, step: 0.02, default: 0.28},
    {id: 'maximum', label: 'Attribute max', min: 0, max: 1, step: 0.02, default: 0.85},
    {id: 'time', label: 'Edge time', min: 0, max: 1, step: 0.02, default: 0.8}
  ],
  build: ({random, fork, p}) => {
    const net = createRoadNetwork(random);
    const noise = fork('attributes');
    const attribute = net.nodes.map(([x, y]) =>
      clamp(
        0.5 +
          0.34 * Math.sin(x * 0.05 + 1.3) +
          0.24 * Math.cos(y * 0.075) +
          noise.range(-0.08, 0.08),
        0,
        1
      )
    );
    const edgeTime = net.edges.map(() => noise.next());
    const keepNode = attribute.map(value => value >= p.minimum && value <= p.maximum);
    const kept: PathItem[] = [];
    const ghosts: PathItem[] = [];
    net.edges.forEach(([first, second], edge) => {
      const points: Point[] = [net.nodes[first], net.nodes[second]];
      if (keepNode[first] && keepNode[second] && edgeTime[edge] <= p.time) {
        kept.push({points, role: 'primary', width: 1.1, alpha: 0.95, phase: edgeTime[edge]});
      } else {
        ghosts.push({points, role: 'dim', width: 0.4, alpha: 0.2});
      }
    });
    return [
      createStrokes(ghosts),
      createStrokes(kept, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(
        net.nodes.map(([x, y], node) =>
          keepNode[node]
            ? {
                x,
                y,
                radius: 2.2,
                role: getHeatRole(Math.min(0.99, attribute[node])),
                alpha: 0.95,
                phase: attribute[node]
              }
            : {x, y, radius: 1, role: 'dim' as Role, alpha: 0.3}
        ),
        0.3
      )
    ];
  }
};

const networkCoarsening: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'groups', label: 'Groups', min: 2, max: 8, step: 1, default: 5}],
  build: ({random, fork, p}) => {
    const net = createRoadNetwork(random);
    const centerRandom = fork('centers');
    const groupCount = Math.round(p.groups);
    const seeds = Array.from({length: 8}, (_, index): Point => {
      const column = index % 4;
      const row = Math.floor(index / 4);
      return [
        24 + column * 38 + centerRandom.range(-8, 8),
        28 + row * 44 + centerRandom.range(-8, 8)
      ];
    }).slice(0, groupCount);
    const group = net.nodes.map(point => findNearestIndex(seeds, point));
    const centroids: Point[] = seeds.map((_, index) => {
      const members = net.nodes.filter((_, node) => group[node] === index);
      if (members.length === 0) return seeds[index];
      return [
        members.reduce((sum, point) => sum + point[0], 0) / members.length,
        members.reduce((sum, point) => sum + point[1], 0) / members.length
      ];
    });
    const sizes = seeds.map((_, index) => group.filter(value => value === index).length);
    const weights = new Map<string, number>();
    net.edges.forEach(([first, second]) => {
      if (group[first] === group[second]) return;
      const key = `${Math.min(group[first], group[second])}:${Math.max(group[first], group[second])}`;
      weights.set(key, (weights.get(key) ?? 0) + 1);
    });
    const summary: PathItem[] = [...weights.entries()].map(([key, weight], index) => {
      const [first, second] = key.split(':').map(Number);
      return {
        points: [centroids[first], centroids[second]],
        role: 'warm',
        width: 0.7 + weight * 0.7,
        alpha: 0.9,
        phase: (index * 0.23) % 1
      };
    });
    return [
      createBackdrop(net, {alpha: 0.18}),
      createStrokes(
        net.nodes.map((point, node) => ({
          points: [point, centroids[group[node]]],
          role: OWNER_ROLES[group[node] % 5],
          width: 0.35,
          alpha: 0.28
        })),
        'static'
      ),
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1.3,
          role: OWNER_ROLES[group[node] % 5],
          alpha: 0.85,
          phase: node / 54
        })),
        0.2
      ),
      createStrokes(summary, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(
        centroids.map(([x, y], index) => ({
          x,
          y,
          radius: 3 + Math.sqrt(sizes[index]) * 1.2,
          role: 'ink' as Role,
          alpha: 0.9,
          phase: index / groupCount
        })),
        0.25
      ),
      createSprites(
        centroids.map(([x, y], index) => ({
          x,
          y,
          radius: 2 + Math.sqrt(sizes[index]) * 0.7,
          role: OWNER_ROLES[index % 5],
          alpha: 1
        })),
        0
      )
    ];
  }
};

const networkLouvain: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'communities', label: 'Communities', min: 2, max: 5, step: 1, default: 4}],
  build: ({fork, p}) => {
    const clusters = Math.round(p.communities);
    const graph = createCommunityGraph(fork('graph'), clusters, Math.floor(28 / clusters));
    const inside: PathItem[] = [];
    const between: PathItem[] = [];
    graph.edges.forEach(([first, second], index) => {
      const points = [graph.points[first], graph.points[second]];
      if (graph.group[first] === graph.group[second]) {
        inside.push({
          points,
          role: OWNER_ROLES[graph.group[first]],
          width: 0.9,
          alpha: 0.9,
          phase: (index * 0.17) % 1
        });
      } else {
        between.push({points, role: 'ink', width: 0.5, alpha: 0.35});
      }
    });
    const halos: PolygonItem[] = Array.from({length: clusters}, (_, cluster) => ({
      ring: createCircle(COMMUNITY_CENTERS[cluster][0], COMMUNITY_CENTERS[cluster][1], 22, 36),
      role: OWNER_ROLES[cluster],
      alpha: 0.12,
      phase: cluster / clusters
    }));
    return [
      createFills(halos, 'pulse'),
      createStrokes(between),
      createStrokes(inside, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(
        graph.points.map(([x, y], node) => ({
          x,
          y,
          radius: 2.2,
          role: OWNER_ROLES[graph.group[node]],
          phase: node / graph.points.length
        })),
        0.3
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Adjacency matrices

function createMatrixFrame(x: number, y: number, size: number): PathItem {
  return {
    points: [...createRectangle(x, y, size, size), [x, y]],
    role: 'dim',
    width: 0.5,
    alpha: 0.75
  };
}

const adjacencyMatrix: EntryPreview = {
  pointer: 'hover',
  parameters: [{id: 'resolution', label: 'Matrix size (R)', min: 4, max: 24, step: 1, default: 24}],
  build: ({fork, p, pointer}) => {
    const graph = createCommunityGraph(fork('graph'), 4, 6);
    const count = graph.points.length;
    const order = shuffleIndices(fork('labels'), count);
    const resolution = Math.round(p.resolution);
    const placed = graph.points.map(
      ([x, y]): Point => [4 + ((x - 24) / 114) * 72, 12 + ((y - 14) / 72) * 76]
    );
    const focus = pointer ? findNearestIndex(placed, pointer) : -1;
    const side = 66;
    const originX = 88;
    const originY = 16;
    const cell = side / resolution;
    const bin = (node: number) =>
      Math.min(resolution - 1, Math.floor((order[node] / count) * resolution));
    const counts = new Map<string, number>();
    const add = (row: number, column: number) =>
      counts.set(`${row}:${column}`, (counts.get(`${row}:${column}`) ?? 0) + 1);
    graph.edges.forEach(([first, second]) => {
      add(bin(first), bin(second));
      add(bin(second), bin(first));
    });
    const peak = Math.max(...counts.values(), 1);
    const cells: PolygonItem[] = [...counts.entries()].map(([key, value], index) => {
      const [row, column] = key.split(':').map(Number);
      return {
        ring: createRectangle(
          originX + column * cell + 0.15,
          originY + row * cell + 0.15,
          cell - 0.3,
          cell - 0.3
        ),
        role: value / peak > 0.6 ? 'hot' : value / peak > 0.3 ? 'warm' : 'primary',
        alpha: 0.9,
        phase: (index * 0.13) % 1
      };
    });
    const highlight: PolygonItem[] = [];
    const incident: PathItem[] = [];
    if (focus >= 0) {
      highlight.push(
        {
          ring: createRectangle(originX, originY + bin(focus) * cell, side, cell),
          role: 'ink',
          alpha: 0.2
        },
        {
          ring: createRectangle(originX + bin(focus) * cell, originY, cell, side),
          role: 'ink',
          alpha: 0.2
        }
      );
      graph.edges.forEach(([first, second]) => {
        if (first === focus || second === focus)
          incident.push({
            points: [placed[first], placed[second]],
            role: 'warm',
            width: 1.4,
            alpha: 1
          });
      });
    }
    return [
      createStrokes(
        graph.edges.map(([first, second]) => ({
          points: [placed[first], placed[second]],
          role: 'primary' as Role,
          width: 0.5,
          alpha: 0.4
        }))
      ),
      createStrokes(incident, 'flow', {speed: 0.35, blend: 'add'}),
      createSprites(
        placed.map(([x, y], node) => ({
          x,
          y,
          radius: node === focus ? 3.2 : 1.8,
          role: node === focus ? ('ink' as Role) : getClassRole(graph.group[node], 4),
          phase: node / count
        })),
        0.3
      ),
      createStrokes([createMatrixFrame(originX, originY, side)]),
      createFills(cells, 'pulse'),
      createFills(highlight, 'static')
    ];
  }
};

const adjacencyMatrixOrder: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'groups', label: 'Groups', min: 2, max: 5, step: 1, default: 4}],
  build: ({fork, p}) => {
    const groups = Math.round(p.groups);
    const size = Math.round(24 / groups);
    const graph = createCommunityGraph(fork('graph'), groups, size);
    const count = graph.points.length;
    const shuffled = shuffleIndices(fork('labels'), count);
    const pitch = 2.05;
    const span = count * pitch;
    const leftX = 8;
    const rightX = SCENE_WIDTH - 8 - span;
    const originY = 50 - span / 2;
    const draw = (position: number[], originX: number): PolygonItem[] => {
      const cells: PolygonItem[] = [];
      graph.edges.forEach(([first, second], index) => {
        const same = graph.group[first] === graph.group[second];
        for (const [row, column] of [
          [first, second],
          [second, first]
        ]) {
          cells.push({
            ring: createRectangle(
              originX + position[column] * pitch,
              originY + position[row] * pitch,
              pitch - 0.25,
              pitch - 0.25
            ),
            role: same ? OWNER_ROLES[graph.group[first]] : 'dim',
            alpha: same ? 0.95 : 0.55,
            phase: (index * 0.11) % 1
          });
        }
      });
      return cells;
    };
    const sortedPosition = graph.points.map((_, node) => node);
    const permutation: PathItem[] = graph.points.map((_, node) => {
      const startY = originY + shuffled[node] * pitch + pitch / 2;
      const endY = originY + sortedPosition[node] * pitch + pitch / 2;
      return {
        points: Array.from({length: 14}, (_, step): Point => {
          const amount = step / 13;
          const eased = amount * amount * (3 - 2 * amount);
          return [
            leftX + span + 3 + amount * (rightX - leftX - span - 6),
            mixNumbers(startY, endY, eased)
          ];
        }),
        role: OWNER_ROLES[graph.group[node]],
        width: 0.7,
        alpha: 0.7,
        phase: (node * 0.07) % 1
      };
    });
    return [
      createStrokes([
        createMatrixFrame(leftX, originY, span),
        createMatrixFrame(rightX, originY, span)
      ]),
      createFills(draw(shuffled, leftX), 'pulse'),
      createStrokes(permutation, 'flow', {speed: 0.3, blend: 'add'}),
      createFills(draw(sortedPosition, rightX), 'pulse')
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Flows

const edgeBundling: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'bundling', label: 'Bundling', min: 0, max: 1, step: 0.05, default: 0.85}],
  build: ({fork, p}) => {
    const centers: Point[] = [
      [26, 28],
      [30, 76],
      [134, 24],
      [132, 74]
    ];
    const pairs: [number, number][] = [
      [0, 2],
      [0, 3],
      [1, 2],
      [1, 3],
      [0, 1],
      [2, 3]
    ];
    const edgeRandom = fork('edges');
    const items: PathItem[] = [];
    const ends: Point[] = [];
    for (let index = 0; index < 36; index++) {
      const [from, to] = pairs[index % pairs.length];
      const swap = edgeRandom.next() > 0.5;
      const source = swap ? to : from;
      const target = swap ? from : to;
      const start: Point = [
        centers[source][0] + edgeRandom.gaussian() * 5,
        centers[source][1] + edgeRandom.gaussian() * 7
      ];
      const end: Point = [
        centers[target][0] + edgeRandom.gaussian() * 5,
        centers[target][1] + edgeRandom.gaussian() * 7
      ];
      ends.push(start, end);
      const axisA = centers[source];
      const axisB = centers[target];
      const fractions = [0, 0.2, 0.5, 0.8, 1];
      const bundled: Point[] = [
        start,
        [mixNumbers(axisA[0], axisB[0], 0.22), mixNumbers(axisA[1], axisB[1], 0.22)],
        [
          mixNumbers(axisA[0], axisB[0], 0.5),
          mixNumbers(axisA[1], axisB[1], 0.5) + (from === 0 && to === 3 ? 6 : 0)
        ],
        [mixNumbers(axisA[0], axisB[0], 0.78), mixNumbers(axisA[1], axisB[1], 0.78)],
        end
      ];
      const controls = bundled.map(
        (point, step): Point => [
          mixNumbers(mixNumbers(start[0], end[0], fractions[step]), point[0], p.bundling),
          mixNumbers(mixNumbers(start[1], end[1], fractions[step]), point[1], p.bundling)
        ]
      );
      items.push({
        points: smoothPath(controls, 3),
        role: OWNER_ROLES[(index % pairs.length) % 4],
        width: 0.8,
        alpha: 0.7,
        phase: (index * 0.137) % 1
      });
    }
    return [
      createStrokes(items, 'flow', {speed: 0.3, trail: 1.1, blend: 'add'}),
      createSprites(toSprites(ends, 'ink', 1.1, undefined, 0.7)),
      createSprites(toSprites(centers, 'dim', 0.1, undefined, 0))
    ];
  }
};

const flowAggregation: EntryPreview = {
  pointer: 'none',
  parameters: [{id: 'top', label: 'Top K flows', min: 1, max: 12, step: 1, default: 6}],
  build: ({fork, p}) => {
    const zones: Point[] = [
      [24, 52],
      [58, 22],
      [96, 34],
      [134, 24],
      [122, 70],
      [78, 80],
      [40, 82]
    ];
    const tripRandom = fork('trips');
    const weights = new Map<string, number>();
    const raw: PathItem[] = [];
    const totals = zones.map(() => 0);
    for (let index = 0; index < 90; index++) {
      const from = Math.floor(Math.pow(tripRandom.next(), 1.7) * zones.length);
      let to = Math.floor(Math.pow(tripRandom.next(), 1.7) * zones.length);
      if (to === from) to = (to + 1) % zones.length;
      const key = `${from}:${to}`;
      weights.set(key, (weights.get(key) ?? 0) + 1);
      totals[from]++;
      totals[to]++;
      if (index < 50) {
        raw.push({
          points: createArc(zones[from], zones[to], 0.2 + tripRandom.range(-0.08, 0.08), 16),
          role: 'dim',
          width: 0.3,
          alpha: 0.28
        });
      }
    }
    const ranked = [...weights.entries()]
      .sort((first, second) => second[1] - first[1])
      .slice(0, Math.round(p.top));
    const peak = ranked[0][1];
    const top: PathItem[] = ranked.map(([key, weight], index) => {
      const [from, to] = key.split(':').map(Number);
      return {
        points: createArc(zones[from], zones[to], 0.2, 24),
        role: weight / peak > 0.6 ? 'warm' : 'primary',
        width: 0.8 + (weight / peak) * 2.8,
        alpha: 0.95,
        phase: (index * 0.21) % 1
      };
    });
    return [
      createStrokes(raw),
      createStrokes(top, 'flow', {speed: 0.3, trail: 1.3, blend: 'add'}),
      createSprites(
        zones.map(([x, y], index) => ({
          x,
          y,
          radius: 1.8 + Math.sqrt(totals[index]) * 0.5,
          role: 'ink' as Role,
          phase: index / 7
        })),
        0.25
      )
    ];
  }
};

type FlowArc = {to: number; capacity: number; cost: number; reverse: number; edge: number};

const networkFlow: EntryPreview = {
  pointer: 'sticky',
  parameters: [{id: 'capacity', label: 'Max capacity', min: 1, max: 4, step: 1, default: 2}],
  build: ({random, fork, p, pointer}) => {
    const net = createRoadNetwork(random);
    const capacityRandom = fork('capacity');
    const capacities = net.edges.map(
      () => 1 + Math.floor(capacityRandom.next() * Math.round(p.capacity))
    );
    const arcs: FlowArc[][] = net.nodes.map(() => []);
    const forwardArcs: FlowArc[][] = net.edges.map(() => []);
    const addArc = (from: number, to: number, capacity: number, cost: number, edge: number) => {
      const arc = {to, capacity, cost, reverse: arcs[to].length, edge};
      arcs[from].push(arc);
      arcs[to].push({to: from, capacity: 0, cost: -cost, reverse: arcs[from].length - 1, edge});
      forwardArcs[edge].push(arc);
    };
    net.edges.forEach(([first, second], edge) => {
      const cost = getDistance(net.nodes[first], net.nodes[second]);
      addArc(first, second, capacities[edge], cost, edge);
      addArc(second, first, capacities[edge], cost, edge);
    });
    const source = getNodeIndex(net, 2, 0);
    const sink =
      pointer && pointer[0] > 90
        ? findNearestIndex(net.nodes, pointer)
        : getNodeIndex(net, 3, net.columns - 1);
    for (let round = 0; round < 40; round++) {
      const distance = new Array<number>(net.nodes.length).fill(Infinity);
      const via = new Array<[number, number]>(net.nodes.length).fill([-1, -1]);
      distance[source] = 0;
      const queue = [source];
      for (let head = 0; head < queue.length; head++) {
        const node = queue[head];
        arcs[node].forEach((arc, index) => {
          if (arc.capacity > 0 && distance[node] + arc.cost < distance[arc.to] - 1e-9) {
            distance[arc.to] = distance[node] + arc.cost;
            via[arc.to] = [node, index];
            if (!queue.includes(arc.to, head + 1)) queue.push(arc.to);
          }
        });
      }
      if (distance[sink] === Infinity) break;
      let push = Infinity;
      for (let node = sink; node !== source; node = via[node][0])
        push = Math.min(push, arcs[via[node][0]][via[node][1]].capacity);
      for (let node = sink; node !== source; node = via[node][0]) {
        const arc = arcs[via[node][0]][via[node][1]];
        arc.capacity -= push;
        arcs[node][arc.reverse].capacity += push;
      }
    }
    const side = new Array<boolean>(net.nodes.length).fill(false);
    const stack = [source];
    side[source] = true;
    while (stack.length > 0) {
      const node = stack.pop() as number;
      for (const arc of arcs[node]) {
        if (arc.capacity > 0 && !side[arc.to]) {
          side[arc.to] = true;
          stack.push(arc.to);
        }
      }
    }
    const flows: PathItem[] = [];
    const cut: PathItem[] = [];
    net.edges.forEach(([first, second], edge) => {
      const used = capacities[edge] - forwardArcs[edge][0].capacity;
      const usedBack = capacities[edge] - forwardArcs[edge][1].capacity;
      const netFlow = used - usedBack;
      if (netFlow !== 0) {
        const points =
          netFlow > 0
            ? [net.nodes[first], net.nodes[second]]
            : [net.nodes[second], net.nodes[first]];
        flows.push({
          points,
          role: Math.abs(netFlow) >= capacities[edge] ? 'hot' : 'warm',
          width: 0.8 + Math.abs(netFlow) * 0.7,
          alpha: 0.95,
          phase: (edge * 0.19) % 1
        });
      }
      if (side[first] !== side[second])
        cut.push({
          points: [net.nodes[first], net.nodes[second]],
          role: 'violet',
          width: 2.4,
          alpha: 0.95
        });
    });
    return [
      createBackdrop(net, {alpha: 0.35}),
      createStrokes(flows, 'flow', {speed: 0.35, blend: 'add'}),
      createStrokes(cut, 'pulse', {speed: 0.5, blend: 'add'}),
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 1.2,
          role: side[node] ? ('warm' as Role) : ('cool' as Role),
          alpha: side[node] ? 0.8 : 0.55
        })),
        0
      ),
      createSprites(
        [
          {x: net.nodes[source][0], y: net.nodes[source][1], radius: 4.6, role: 'ink'},
          {x: net.nodes[sink][0], y: net.nodes[sink][1], radius: 4.6, role: 'ink', phase: 0.5}
        ],
        0.3
      )
    ];
  }
};

const networkRationalAccess: EntryPreview = {
  pointer: 'hover',
  parameters: [
    {id: 'equalization', label: 'Congestion', min: 0, max: 1, step: 0.05, default: 0.85}
  ],
  build: ({fork, p, pointer}) => {
    const facilities: Point[] = [pointer ? clampPoint(pointer, 10) : [36, 38], [118, 30], [86, 78]];
    const capacity = [1, 1, 1];
    const demand = createClusterPoints(
      fork('demand'),
      [
        [44, 70],
        [104, 54],
        [80, 22]
      ],
      6,
      9
    );
    const populationRandom = fork('population');
    const population = demand.map(() => populationRandom.range(0.7, 1.5));
    const total = population.reduce((sum, value) => sum + value, 0);
    const multiplier = [1, 1, 1];
    let shares: number[][] = [];
    let load = [0, 0, 0];
    for (let round = 0; round < 16; round++) {
      shares = demand.map(point => {
        const utility = facilities.map(
          (facility, index) =>
            multiplier[index] / Math.pow(Math.max(getDistance(point, facility), 5), 2)
        );
        const sum = utility.reduce((accumulated, value) => accumulated + value, 0);
        return utility.map(value => value / sum);
      });
      load = facilities.map((_, index) =>
        shares.reduce((sum, share, zone) => sum + share[index] * population[zone], 0)
      );
      facilities.forEach((_, index) => {
        multiplier[index] *= Math.pow(
          ((capacity[index] / 3) * total) / Math.max(load[index], 1e-6),
          0.6 * p.equalization
        );
      });
    }
    const links: PathItem[] = [];
    demand.forEach((point, zone) => {
      shares[zone].forEach((share, index) => {
        if (share > 0.06) {
          links.push({
            points: createArc(point, facilities[index], 0.08),
            role: OWNER_ROLES[index],
            width: 0.35 + share * population[zone] * 1.6,
            alpha: 0.3 + share * 0.65,
            phase: (zone * 0.19 + index * 0.33) % 1
          });
        }
      });
    });
    const scale = (value: number) => 4 + 13 * Math.sqrt(value);
    return [
      createStrokes(
        facilities.map(([x, y]) => ({
          points: createCircle(x, y, scale(1 / 3), 40),
          role: 'ink' as Role,
          width: 0.6,
          alpha: 0.6
        })),
        'static'
      ),
      createStrokes(links, 'flow', {speed: 0.3, blend: 'add'}),
      createSprites(toSprites(demand, 'cool', 1.2, undefined, 0.85)),
      createSprites(
        facilities.map(([x, y], index) => ({
          x,
          y,
          radius: scale(load[index] / total) * 0.75,
          role: load[index] / total > 0.4 ? ('hot' as Role) : OWNER_ROLES[index],
          alpha: 0.8,
          phase: index / 3
        })),
        0.2
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Point patterns on networks

function buildKFunctionScene(
  context: {
    random: Random;
    fork: (label: string) => Random;
    p: Record<string, number>;
    pointer: Point | null;
  },
  inhomogeneous: boolean
): PreviewLayer[] {
  const {random, fork, p, pointer} = context;
  const net = placeNet(createRoadNetwork(random), {x: 4, y: 6, width: 106, height: 88});
  const adjacency = createAdjacency(net);
  const table = getAllPairs(adjacency);
  const maximumRadius = 70;
  const count = 16;
  const intensity = net.nodes.map(([x, y]) =>
    inhomogeneous
      ? 0.15 +
        Math.exp(-Math.pow((x - 38) / 26, 2) - Math.pow((y - 38) / 30, 2)) +
        0.5 * Math.exp(-Math.pow((x - 86) / 18, 2))
      : 1
  );
  const eventRandom = fork('events');
  const events = inhomogeneous
    ? pickWeightedNodes(
        eventRandom,
        intensity.map(value => value * value),
        count
      )
    : pickClusteredEvents(
        net,
        eventRandom,
        count,
        [
          [30, 30],
          [78, 72]
        ],
        15
      );
  if (inhomogeneous && p.clustering > 0) {
    const extra = Math.round(p.clustering * 6);
    const parents = events.slice(0, extra);
    parents.forEach((parent, index) => {
      const near = adjacency[parent].map(arc => arc.to).find(node => !events.includes(node));
      if (near !== undefined) events[count - 1 - index] = near;
    });
  }
  const simulationRandom = fork('simulations');
  const weight = (first: number, second: number) => 1 / (intensity[first] * intensity[second]);
  const observed = countPairsWithin(
    table,
    events,
    maximumRadius,
    inhomogeneous ? weight : undefined
  );
  const simulations: number[][] = [];
  for (let run = 0; run < 12; run++) {
    const nodes = inhomogeneous
      ? pickWeightedNodes(
          simulationRandom,
          intensity.map(value => value * value),
          count
        )
      : pickWeightedNodes(simulationRandom, intensity, count);
    simulations.push(
      countPairsWithin(table, nodes, maximumRadius, inhomogeneous ? weight : undefined)
    );
  }
  const uniform: number[][] = [];
  for (let run = 0; run < 5; run++) {
    uniform.push(
      countPairsWithin(
        table,
        pickWeightedNodes(
          simulationRandom,
          net.nodes.map(() => 1),
          count
        ),
        maximumRadius
      )
    );
  }
  const mean = (runs: number[][], bin: number) =>
    runs.reduce((sum, row) => sum + row[bin], 0) / runs.length + 1e-6;
  const ratio = (value: number, reference: number) => clamp(value / reference / 3, 0, 1);
  const chart = createChartMapper({x: 120, y: 18, width: 34, height: 62});
  const toLine = (values: number[]): Point[] => values.map((value, bin) => chart.at(bin, value));
  const observedRatios = observed.map((value, bin) => ratio(value, mean(simulations, bin)));
  const upper = simulations[0].map((_, bin) =>
    ratio(Math.max(...simulations.map(row => row[bin])), mean(simulations, bin))
  );
  const lower = simulations[0].map((_, bin) =>
    ratio(Math.min(...simulations.map(row => row[bin])), mean(simulations, bin))
  );
  const band: Point[] = [...toLine(upper), ...toLine(lower).reverse()];
  const naive = observed.map((value, bin) => ratio(value, mean(uniform, bin)));
  const origin = pointer ? findNearestIndex(net.nodes, pointer) : events[0];
  const search = runSearch(adjacency, [origin]);
  const radius = p.radius;
  const ball: PathItem[] = [];
  net.edges.forEach(([first, second], edge) => {
    if (search.distances[first] <= radius && search.distances[second] <= radius) {
      ball.push({
        points: [net.nodes[first], net.nodes[second]],
        role: 'warm',
        width: 1.2,
        alpha: 0.9,
        phase: (edge * 0.07) % 1
      });
    }
  });
  const markerBin = clamp((radius / maximumRadius) * RADIUS_BINS - 1, 0, RADIUS_BINS - 1);
  const markerX = chart.box.x + ((markerBin + 1) / RADIUS_BINS) * chart.box.width;
  const layers: PreviewLayer[] = [];
  if (inhomogeneous) {
    layers.push(
      createSprites(
        net.nodes.map(([x, y], node) => ({
          x,
          y,
          radius: 4 + intensity[node] * 5,
          role: 'violet' as Role,
          alpha: 0.05 + intensity[node] * 0.13,
          phase: node / 54
        })),
        0.35
      )
    );
  }
  layers.push(
    createBackdrop(net, {alpha: 0.3}),
    createStrokes(ball, 'flow', {speed: 0.3, blend: 'add'}),
    createSprites(
      events.map(node => ({
        x: net.nodes[node][0],
        y: net.nodes[node][1],
        radius: search.distances[node] <= radius ? 3.2 : 2.2,
        role: search.distances[node] <= radius ? ('warm' as Role) : ('hot' as Role),
        phase: node / 54
      })),
      0.3
    ),
    createSprites(
      [{x: net.nodes[origin][0], y: net.nodes[origin][1], radius: 4.4, role: 'ink'}],
      0.3
    ),
    createStrokes(chart.axes()),
    createFills([{ring: band, role: 'violet', alpha: 0.35}], 'static'),
    createStrokes([
      {
        points: [
          [markerX, chart.box.y],
          [markerX, chart.box.y + chart.box.height]
        ],
        role: 'ink',
        width: 0.5,
        alpha: 0.6
      }
    ]),
    createStrokes(
      [
        ...(inhomogeneous
          ? [{points: toLine(naive), role: 'hot' as Role, width: 1, alpha: 0.7}]
          : []),
        {points: toLine(observedRatios), role: 'warm' as Role, width: 1.6, alpha: 1}
      ],
      'draw',
      {speed: 0.2, blend: 'add'}
    )
  );
  return layers;
}

const networkKFunction: EntryPreview = {
  pointer: 'sticky',
  parameters: [{id: 'radius', label: 'Radius', min: 8, max: 70, step: 2, default: 34, unit: 'px'}],
  build: context => buildKFunctionScene(context, false)
};

const networkInhomogeneousK: EntryPreview = {
  pointer: 'sticky',
  parameters: [
    {id: 'radius', label: 'Radius', min: 8, max: 70, step: 2, default: 34, unit: 'px'},
    {id: 'clustering', label: 'Extra clustering', min: 0, max: 1, step: 0.1, default: 0}
  ],
  build: context => buildKFunctionScene(context, true)
};

const networkKernelDensity: EntryPreview = {
  pointer: 'none',
  parameters: [
    {id: 'bandwidth', label: 'Bandwidth', min: 8, max: 50, step: 1, default: 22, unit: 'px'}
  ],
  build: ({random, fork, p}) => {
    const net = createRoadNetwork(random);
    const adjacency = createAdjacency(net);
    const events = pickClusteredEvents(
      net,
      fork('events'),
      22,
      [
        [36, 36],
        [112, 66]
      ],
      16
    );
    const density = new Array<number>(net.nodes.length).fill(0);
    for (const event of events) {
      const {distances} = runSearch(adjacency, [event]);
      distances.forEach((distance, node) => {
        density[node] += Math.exp(-(distance * distance) / (2 * p.bandwidth * p.bandwidth));
      });
    }
    const peak = Math.max(...density, 1e-6);
    const lixels: PathItem[] = [];
    const pieces = 4;
    net.edges.forEach(([first, second], edge) => {
      for (let piece = 0; piece < pieces; piece++) {
        const a = piece / pieces;
        const b = (piece + 1) / pieces;
        const value = mixNumbers(density[first], density[second], (a + b) / 2) / peak;
        lixels.push({
          points: [
            [
              mixNumbers(net.nodes[first][0], net.nodes[second][0], a),
              mixNumbers(net.nodes[first][1], net.nodes[second][1], a)
            ],
            [
              mixNumbers(net.nodes[first][0], net.nodes[second][0], b),
              mixNumbers(net.nodes[first][1], net.nodes[second][1], b)
            ]
          ],
          role: getHeatRole(Math.min(0.99, Math.pow(value, 0.7))),
          width: 0.5 + value * 2.6,
          alpha: 0.35 + value * 0.65,
          phase: (edge * 0.11 + piece * 0.05) % 1
        });
      }
    });
    return [
      createStrokes(
        lixels.filter(item => item.role !== 'hot' && item.role !== 'warm'),
        'static',
        {blend: 'add'}
      ),
      createStrokes(
        lixels.filter(item => item.role === 'warm'),
        'pulse',
        {speed: 0.25, blend: 'add'}
      ),
      createStrokes(
        lixels.filter(item => item.role === 'hot'),
        'pulse',
        {speed: 0.4, blend: 'add'}
      ),
      createSprites(
        events.map(node => ({
          x: net.nodes[node][0],
          y: net.nodes[node][1],
          radius: 1.4,
          role: 'ink' as Role,
          alpha: 0.9
        })),
        0.3
      )
    ];
  }
};

export const NETWORK_ROUTING_ACCESS_PREVIEWS: Record<string, EntryPreview> = {
  'network-reachability': reachability,
  'network-service-areas': serviceAreas,
  'network-isochrones': isochrones,
  'network-cost-matrix': costMatrix,
  'network-cost-table': costTable,
  'network-cost-weights': costWeights,
  'network-accessibility': accessibility,
  'catchment-accessibility': catchmentAccessibility,
  'huff-trade-areas': huffTradeAreas,
  'location-allocation': locationAllocation,
  'bounded-facility-allocation': boundedFacilityAllocation,
  'network-facility-optimization': networkFacilityOptimization,
  'facility-scenario': facilityScenario,
  'network-disruption': networkDisruption,
  'drive-time-catchment': driveTimeCatchment,
  'network-snapping': networkSnapping,
  'map-matching': mapMatching,
  'track-quality': trackQuality,
  'network-path-extraction': networkPathExtraction,
  'alternative-routes': alternativeRoutes,
  'network-point-routing': networkPointRouting,
  'network-time-routing': networkTimeRouting,
  'network-line-graph': networkLineGraph,
  'network-noding': networkNoding,
  'network-centrality': networkCentrality,
  'network-structure': networkStructure,
  'network-statistics': networkStatistics,
  'network-analytics-columns': networkAnalyticsColumns,
  'network-neighborhood': networkNeighborhood,
  'network-subgraph-filter': networkSubgraphFilter,
  'network-coarsening': networkCoarsening,
  'network-louvain': networkLouvain,
  'adjacency-matrix': adjacencyMatrix,
  'adjacency-matrix-order': adjacencyMatrixOrder,
  'edge-bundling': edgeBundling,
  'flow-aggregation': flowAggregation,
  'network-flow': networkFlow,
  'network-rational-access': networkRationalAccess,
  'network-k-function': networkKFunction,
  'network-inhomogeneous-k': networkInhomogeneousK,
  'network-kernel-density': networkKernelDensity
};
