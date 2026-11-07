// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/regression-weights.ts, keyed by entry id.

import type {GpuSceneParameter} from '../../types.ts';
import {
  clamp,
  createCircle,
  createScatterPoints,
  evaluateField,
  FIELD_COLUMNS,
  FIELD_ROWS,
  createSmoothField,
  getDistance,
  mixNumbers,
  type Random
} from '../data.ts';
import {findNearestIndex, getClassRole, toPaths, toSprites, type EntryPreview} from '../helpers.ts';
import type {
  FillLayer,
  PathItem,
  PolygonItem,
  Point,
  PreviewLayer,
  Role,
  SpriteItem,
  StrokeLayer
} from '../types.ts';

// ---------------------------------------------------------------------------------------------
// Local helpers

type Edge = [number, number];

function slider(
  id: string,
  label: string,
  min: number,
  max: number,
  step: number,
  defaultValue: number,
  unit?: string
): GpuSceneParameter {
  return {id, label, min, max, step, default: defaultValue, unit};
}

function choice(id: string, label: string, options: string[], defaultValue = 0): GpuSceneParameter {
  return {
    id,
    label,
    min: 0,
    max: options.length - 1,
    step: 1,
    default: defaultValue,
    options
  };
}

function path(points: Point[], role: Role, width: number, alpha = 0.9, phase = 0): PathItem {
  return {points, role, width, alpha, phase};
}

function strokes(
  items: PathItem[],
  animation: StrokeLayer['animation'] = 'static',
  speed = 0.3,
  blend: 'add' | 'alpha' = 'add',
  trail = 1
): StrokeLayer {
  return {type: 'strokes', items, animation, speed, blend, trail};
}

function fills(
  items: PolygonItem[],
  animation: FillLayer['animation'] = 'static',
  blend: 'add' | 'alpha' = 'add'
): FillLayer {
  return {type: 'fills', items, animation, blend};
}

function dots(items: SpriteItem[], pulse = 0.25): PreviewLayer {
  return {type: 'sprites', items, pulse};
}

function closeRing(ring: Point[]): Point[] {
  return [...ring, ring[0]];
}

function arrowHead(from: Point, to: Point, size = 2.6): Point[] {
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  return [
    [to[0] - Math.cos(angle - 0.5) * size, to[1] - Math.sin(angle - 0.5) * size],
    to,
    [to[0] - Math.cos(angle + 0.5) * size, to[1] - Math.sin(angle + 0.5) * size]
  ];
}

/** Segment from a to b, pulled back from both ends so arrows stop before the node glow. */
function shorten(from: Point, to: Point, startMargin: number, endMargin: number): [Point, Point] {
  const length = getDistance(from, to) || 1;
  const dx = (to[0] - from[0]) / length;
  const dy = (to[1] - from[1]) / length;
  return [
    [from[0] + dx * startMargin, from[1] + dy * startMargin],
    [to[0] - dx * endMargin, to[1] - dy * endMargin]
  ];
}

function createSpreadPoints(
  random: Random,
  columns: number,
  rows: number,
  left = 12,
  right = 148,
  top = 12,
  bottom = 88,
  jitter = 0.36
): Point[] {
  const cellWidth = (right - left) / columns;
  const cellHeight = (bottom - top) / rows;
  const points: Point[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      points.push([
        left + (column + 0.5) * cellWidth + random.range(-1, 1) * jitter * cellWidth,
        top + (row + 0.5) * cellHeight + random.range(-1, 1) * jitter * cellHeight
      ]);
    }
  }
  return points;
}

/** Directed nearest neighbor edges, source to neighbor. */
function getNearestEdges(points: Point[], neighbors: number): Edge[] {
  const edges: Edge[] = [];
  points.forEach((point, index) => {
    const nearest = points
      .map((other, otherIndex) => ({
        otherIndex,
        distance: getDistance(point, other)
      }))
      .filter(entry => entry.otherIndex !== index)
      .sort((first, second) => first.distance - second.distance)
      .slice(0, neighbors);
    for (const entry of nearest) edges.push([index, entry.otherIndex]);
  });
  return edges;
}

function getEdgeKey(first: number, second: number): number {
  return Math.min(first, second) * 1000 + Math.max(first, second);
}

function getUniqueEdges(edges: Edge[]): Edge[] {
  const seen = new Set<number>();
  const unique: Edge[] = [];
  for (const edge of edges) {
    const key = getEdgeKey(edge[0], edge[1]);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(edge);
  }
  return unique;
}

function edgesToPaths(points: Point[], edges: Edge[]): Point[][] {
  return edges.map(([first, second]) => [points[first], points[second]]);
}

function getAdjacency(count: number, edges: Edge[]): number[][] {
  const adjacency: number[][] = Array.from({length: count}, () => []);
  for (const [first, second] of edges) {
    adjacency[first].push(second);
    adjacency[second].push(first);
  }
  return adjacency;
}

function getHops(adjacency: number[][], source: number): number[] {
  const hops = new Array<number>(adjacency.length).fill(-1);
  hops[source] = 0;
  const queue = [source];
  for (let head = 0; head < queue.length; head++) {
    for (const neighbor of adjacency[queue[head]]) {
      if (hops[neighbor] >= 0) continue;
      hops[neighbor] = hops[queue[head]] + 1;
      queue.push(neighbor);
    }
  }
  return hops;
}

function getValueRole(value: number): Role {
  return getClassRole(Math.round(clamp(value, 0, 1) * 4), 5);
}

/** A smooth value in 0..1 that varies across the scene. */
function getSurfaceValue(x: number, y: number, phase = 0): number {
  return clamp(
    0.5 + 0.34 * Math.sin(x * 0.05 + phase) + 0.26 * Math.cos(y * 0.07 - phase * 0.7),
    0,
    1
  );
}

function fitLine(points: Point[]): {
  slope: number;
  intercept: number;
  meanX: number;
  meanY: number;
} {
  const meanX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
  let covariance = 0;
  let variance = 0;
  for (const [x, y] of points) {
    covariance += (x - meanX) * (y - meanY);
    variance += (x - meanX) ** 2;
  }
  const slope = covariance / (variance || 1);
  return {slope, intercept: meanY - slope * meanX, meanX, meanY};
}

const PLOT_LEFT = 18;
const PLOT_RIGHT = 148;
const PLOT_TOP = 10;
const PLOT_BOTTOM = 90;

function createAxes(
  left = PLOT_LEFT,
  right = PLOT_RIGHT,
  top = PLOT_TOP,
  bottom = PLOT_BOTTOM
): StrokeLayer {
  return strokes([
    path(
      [
        [left, top],
        [left, bottom],
        [right, bottom]
      ],
      'dim',
      0.5,
      0.7
    )
  ]);
}

function createNormalCdf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * (1 + sign * erf);
}

type CellMap = {
  vertices: Point[][];
  cells: {column: number; row: number; ring: Point[]; center: Point}[];
};

function createCellMap(random: Random, columns: number, rows: number, jitter: number): CellMap {
  const left = 14;
  const right = 146;
  const top = 10;
  const bottom = 90;
  const vertices: Point[][] = [];
  for (let row = 0; row <= rows; row++) {
    const line: Point[] = [];
    for (let column = 0; column <= columns; column++) {
      const interior = column > 0 && column < columns && row > 0 && row < rows;
      line.push([
        left + ((right - left) * column) / columns + (interior ? random.range(-1, 1) * jitter : 0),
        top + ((bottom - top) * row) / rows + (interior ? random.range(-1, 1) * jitter : 0)
      ]);
    }
    vertices.push(line);
  }
  const cells: CellMap['cells'] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const ring: Point[] = [
        vertices[row][column],
        vertices[row][column + 1],
        vertices[row + 1][column + 1],
        vertices[row + 1][column]
      ];
      const center: Point = [
        ring.reduce((sum, point) => sum + point[0], 0) / 4,
        ring.reduce((sum, point) => sum + point[1], 0) / 4
      ];
      cells.push({column, row, ring, center});
    }
  }
  return {vertices, cells};
}

// ---------------------------------------------------------------------------------------------
// Weights built from geometry

const contiguityWeights: EntryPreview = {
  parameters: [choice('rule', 'Rule', ['Queen', 'Rook'])],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const columns = 6;
    const map = createCellMap(fork('map'), columns, 4, 4.5);
    const focus = pointer
      ? findNearestIndex(
          map.cells.map(cell => cell.center),
          pointer
        )
      : 8;
    const focusCell = map.cells[focus];
    const rook = p.rule === 1;
    const edgeNeighbors: number[] = [];
    const cornerNeighbors: number[] = [];
    const sharedEdges: Point[][] = [];
    const sharedVertices: Point[] = [];
    map.cells.forEach((cell, index) => {
      const dc = cell.column - focusCell.column;
      const dr = cell.row - focusCell.row;
      if (index === focus || Math.abs(dc) > 1 || Math.abs(dr) > 1) return;
      const {column, row} = focusCell;
      if (Math.abs(dc) + Math.abs(dr) === 1) {
        edgeNeighbors.push(index);
        const a: Point =
          dc === 0
            ? map.vertices[row + (dr === 1 ? 1 : 0)][column]
            : map.vertices[row][column + (dc === 1 ? 1 : 0)];
        const b: Point =
          dc === 0
            ? map.vertices[row + (dr === 1 ? 1 : 0)][column + 1]
            : map.vertices[row + 1][column + (dc === 1 ? 1 : 0)];
        sharedEdges.push([a, b]);
      } else if (!rook) {
        cornerNeighbors.push(index);
        sharedVertices.push(map.vertices[row + (dr === 1 ? 1 : 0)][column + (dc === 1 ? 1 : 0)]);
      }
    });
    const neighborItems = [...edgeNeighbors, ...cornerNeighbors];
    return [
      fills(
        map.cells.map(cell => ({
          ring: cell.ring,
          role: 'primary' as Role,
          alpha: 0.07
        })),
        'static'
      ),
      fills(
        neighborItems.map((index, order) => ({
          ring: map.cells[index].ring,
          role: 'warm' as Role,
          alpha: 0.32,
          phase: order / 8
        })),
        'pulse'
      ),
      fills([{ring: focusCell.ring, role: 'hot', alpha: 0.5}], 'pulse'),
      strokes(map.cells.map(cell => path(closeRing(cell.ring), 'dim', 0.5, 0.75))),
      strokes(
        sharedEdges.map(edge => path(edge, 'warm', 1.8, 1)),
        'pulse',
        0.6
      ),
      strokes(
        neighborItems.map((index, order) =>
          path([focusCell.center, map.cells[index].center], 'warm', 0.9, 0.9, order / 8)
        ),
        'flow',
        0.5
      ),
      dots(
        [
          ...sharedVertices.map(
            (point): SpriteItem => ({
              x: point[0],
              y: point[1],
              radius: 2.4,
              role: 'violet',
              alpha: 1
            })
          ),
          {
            x: focusCell.center[0],
            y: focusCell.center[1],
            radius: 3,
            role: 'hot',
            alpha: 1
          }
        ],
        0.3
      )
    ];
  }
};

const fuzzyContiguity: EntryPreview = {
  parameters: [slider('tolerance', 'Tolerance', 0, 8, 0.25, 5.5, 'px')],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const columns = 6;
    const rows = 4;
    const random = fork('gaps');
    const cells = Array.from({length: columns * rows}, (_, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      const gap = random.range(1, 4.6);
      return {
        column,
        row,
        center: [14 + 22 * (column + 0.5), 10 + 20 * (row + 0.5)] as Point,
        halfWidth: 11 - gap,
        halfHeight: 10 - gap
      };
    });
    const focus = pointer
      ? findNearestIndex(
          cells.map(cell => cell.center),
          pointer
        )
      : 9;
    const focusCell = cells[focus];
    const getRect = (cell: (typeof cells)[number], grow = 0): Point[] => [
      [cell.center[0] - cell.halfWidth - grow, cell.center[1] - cell.halfHeight - grow],
      [cell.center[0] + cell.halfWidth + grow, cell.center[1] - cell.halfHeight - grow],
      [cell.center[0] + cell.halfWidth + grow, cell.center[1] + cell.halfHeight + grow],
      [cell.center[0] - cell.halfWidth - grow, cell.center[1] + cell.halfHeight + grow]
    ];
    const neighbors: number[] = [];
    cells.forEach((cell, index) => {
      if (
        index === focus ||
        Math.abs(cell.column - focusCell.column) > 1 ||
        Math.abs(cell.row - focusCell.row) > 1
      )
        return;
      const dx = Math.max(
        0,
        Math.abs(cell.center[0] - focusCell.center[0]) - cell.halfWidth - focusCell.halfWidth
      );
      const dy = Math.max(
        0,
        Math.abs(cell.center[1] - focusCell.center[1]) - cell.halfHeight - focusCell.halfHeight
      );
      if (Math.hypot(dx, dy) <= p.tolerance) neighbors.push(index);
    });
    return [
      fills(
        cells.map(cell => ({
          ring: getRect(cell),
          role: 'primary' as Role,
          alpha: 0.1
        })),
        'static'
      ),
      fills(
        neighbors.map((index, order) => ({
          ring: getRect(cells[index]),
          role: 'warm' as Role,
          alpha: 0.35,
          phase: order / 8
        })),
        'pulse'
      ),
      fills([{ring: getRect(focusCell), role: 'hot', alpha: 0.5}], 'pulse'),
      strokes(cells.map(cell => path(closeRing(getRect(cell)), 'dim', 0.5, 0.8))),
      strokes([path(closeRing(getRect(focusCell, p.tolerance)), 'cool', 1, 0.95)], 'pulse', 0.5),
      strokes(
        neighbors.map((index, order) =>
          path([focusCell.center, cells[index].center], 'warm', 0.9, 0.9, order / 8)
        ),
        'flow',
        0.5
      ),
      dots([
        {
          x: focusCell.center[0],
          y: focusCell.center[1],
          radius: 3,
          role: 'hot',
          alpha: 1
        }
      ])
    ];
  }
};

const latticeWeights: EntryPreview = {
  parameters: [choice('rule', 'Rule', ['Queen', 'Rook']), slider('order', 'Order', 1, 3, 1, 2)],
  pointer: 'hover',
  build: ({p, pointer}) => {
    const columns = 14;
    const rows = 8;
    const getCenter = (column: number, row: number): Point => [13 + column * 10.3, 12 + row * 10.9];
    const focusColumn = pointer ? clamp(Math.round((pointer[0] - 13) / 10.3), 0, columns - 1) : 6;
    const focusRow = pointer ? clamp(Math.round((pointer[1] - 12) / 10.9), 0, rows - 1) : 3;
    const roles: Role[] = ['hot', 'warm', 'violet', 'cool'];
    const groups: PolygonItem[][] = [[], [], [], []];
    const dim: PolygonItem[] = [];
    const sprites: SpriteItem[] = [];
    const links: PathItem[] = [];
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const [x, y] = getCenter(column, row);
        const dc = Math.abs(column - focusColumn);
        const dr = Math.abs(row - focusRow);
        const distance = p.rule === 1 ? dc + dr : Math.max(dc, dr);
        const ring: Point[] = [
          [x - 4.3, y - 4.6],
          [x + 4.3, y - 4.6],
          [x + 4.3, y + 4.6],
          [x - 4.3, y + 4.6]
        ];
        if (distance <= p.order) {
          groups[distance].push({
            ring,
            role: roles[distance],
            alpha: 0.42 - distance * 0.08,
            phase: distance / 4
          });
          sprites.push({
            x,
            y,
            radius: distance === 0 ? 3 : 1.4,
            role: roles[distance],
            alpha: 1
          });
          if (distance === 1)
            links.push(
              path(
                [getCenter(focusColumn, focusRow), [x, y]],
                'warm',
                0.8,
                0.9,
                (column + row) / 10
              )
            );
        } else {
          dim.push({ring, role: 'primary', alpha: 0.07});
          sprites.push({x, y, radius: 0.8, role: 'primary', alpha: 0.5});
        }
      }
    }
    return [
      fills(dim),
      ...groups.map(group => fills(group, 'pulse')),
      strokes(links, 'flow', 0.5),
      dots(sprites, 0.15)
    ];
  }
};

const neighborSearch: EntryPreview = {
  parameters: [slider('neighbors', 'Neighbors (k)', 1, 8, 1, 4)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const points = createScatterPoints(fork('points'), 64, 8);
    const origin: Point = pointer ?? [76, 46];
    const ranked = points
      .map((point, index) => ({index, distance: getDistance(point, origin)}))
      .sort((first, second) => first.distance - second.distance);
    const selected = ranked.slice(0, Math.round(p.neighbors));
    const radius = selected[selected.length - 1].distance;
    const chosen = new Set(selected.map(entry => entry.index));
    return [
      strokes(
        [
          path(createCircle(origin[0], origin[1], radius * 0.5, 36), 'dim', 0.4, 0.5),
          path(createCircle(origin[0], origin[1], radius * 0.78, 36), 'dim', 0.4, 0.4)
        ],
        'static'
      ),
      strokes(
        [path(createCircle(origin[0], origin[1], radius, 48), 'cool', 1, 0.95)],
        'pulse',
        0.5
      ),
      strokes(
        selected.map((entry, order) =>
          path([origin, points[entry.index]], 'warm', 1, 0.9, order / 8)
        ),
        'flow',
        0.55
      ),
      dots(
        points.map((point, index) => ({
          x: point[0],
          y: point[1],
          radius: chosen.has(index) ? 2.4 : 1.3,
          role: chosen.has(index) ? ('warm' as Role) : ('primary' as Role),
          alpha: chosen.has(index) ? 1 : 0.55,
          phase: index / 64
        }))
      ),
      dots([{x: origin[0], y: origin[1], radius: 4, role: 'hot', alpha: 1}], 0.3)
    ];
  }
};

const islandAttachment: EntryPreview = {
  parameters: [slider('islands', 'Islands', 1, 3, 1, 2)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const random = fork('clusters');
    const centers: Point[] = [
      [45, 50],
      [108, 46]
    ];
    const members: Point[] = [];
    centers.forEach(center => {
      for (let index = 0; index < 16; index++)
        members.push([
          clamp(center[0] + random.gaussian() * 11, 8, 152),
          clamp(center[1] + random.gaussian() * 12, 8, 92)
        ]);
    });
    const islandSpots: Point[] = [pointer ?? [140, 86], [22, 12], [84, 90]];
    const islands = islandSpots.slice(0, Math.round(p.islands));
    const links: PathItem[] = [];
    members.forEach((point, index) => {
      members.forEach((other, otherIndex) => {
        if (otherIndex > index && getDistance(point, other) < 16)
          links.push(path([point, other], 'primary', 0.5, 0.55));
      });
    });
    const attachments = islands.map((island, order) =>
      path([island, members[findNearestIndex(members, island)]], 'warm', 1.3, 1, order / 3)
    );
    return [
      strokes(links, 'static'),
      strokes(attachments, 'draw', 0.25),
      dots(toSprites(members, 'primary', 1.6)),
      dots(
        islands.map(
          (island, order): SpriteItem => ({
            x: island[0],
            y: island[1],
            radius: 3.4,
            role: 'hot',
            alpha: 1,
            phase: order / 3
          })
        ),
        0.4
      ),
      strokes(
        islands.map(island => path(createCircle(island[0], island[1], 7, 24), 'hot', 0.6, 0.6)),
        'pulse',
        0.5
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Weights operations

const spatialWeightsTransform: EntryPreview = {
  parameters: [slider('standardise', 'Row standardise', 0, 1, 0.05, 1)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('rows');
    const rows: {center: Point; count: number; radius: number}[] = [
      {center: [42, 44], count: 2, radius: 22},
      {center: [112, 44], count: 6, radius: 25}
    ];
    const layers: PreviewLayer[] = [];
    const links: PathItem[] = [];
    const heads: PathItem[] = [];
    const neighborSprites: SpriteItem[] = [];
    const bars: PathItem[] = [];
    rows.forEach((row, rowIndex) => {
      const weight = mixNumbers(1, 1 / row.count, p.standardise);
      const phase = random.range(0, 1);
      for (let index = 0; index < row.count; index++) {
        const angle = phase + (index / row.count) * Math.PI * 2;
        const target: Point = [
          row.center[0] + Math.cos(angle) * row.radius,
          row.center[1] + Math.sin(angle) * row.radius * 0.8
        ];
        const [start, end] = shorten(row.center, target, 4, 3);
        links.push(path([start, end], 'warm', 0.4 + 3.2 * weight, 0.9, index / row.count));
        neighborSprites.push({
          x: target[0],
          y: target[1],
          radius: 2,
          role: 'primary',
          alpha: 0.9,
          phase: index / row.count
        });
      }
      const sum = weight * row.count;
      bars.push(
        path(
          [
            [row.center[0] - 20, 88],
            [row.center[0] - 20 + sum * 40, 88]
          ],
          rowIndex === 0 ? 'cool' : 'violet',
          3,
          0.95
        )
      );
      heads.push(
        path(
          [
            [row.center[0] - 20, 82],
            [row.center[0] - 20, 94]
          ],
          'dim',
          0.5,
          0.7
        ),
        path(
          [
            [row.center[0] + 20, 82],
            [row.center[0] + 20, 94]
          ],
          'dim',
          0.5,
          0.7
        )
      );
    });
    layers.push(strokes(heads), strokes(bars, 'pulse', 0.4));
    layers.push(
      strokes(links, 'flow', 0.4),
      dots(neighborSprites),
      dots(
        rows.map(
          (row): SpriteItem => ({
            x: row.center[0],
            y: row.center[1],
            radius: 4,
            role: 'hot',
            alpha: 1
          })
        ),
        0.3
      )
    );
    return layers;
  }
};

const spatialWeightsAlgebra: EntryPreview = {
  parameters: [choice('operation', 'Operation', ['Union', 'Intersection', 'Difference'])],
  pointer: 'none',
  build: ({fork, p}) => {
    const points = createSpreadPoints(fork('points'), 8, 5);
    const edgesA = getUniqueEdges(getNearestEdges(points, 2));
    const edgesB: Edge[] = [];
    points.forEach((point, index) =>
      points.forEach((other, otherIndex) => {
        if (
          otherIndex > index &&
          Math.abs(point[1] - other[1]) < 9 &&
          getDistance(point, other) < 28
        )
          edgesB.push([index, otherIndex]);
      })
    );
    const keysA = new Set(edgesA.map(edge => getEdgeKey(edge[0], edge[1])));
    const keysB = new Set(edgesB.map(edge => getEdgeKey(edge[0], edge[1])));
    const result = getUniqueEdges([...edgesA, ...edgesB]).filter(([first, second]) => {
      const key = getEdgeKey(first, second);
      if (p.operation === 1) return keysA.has(key) && keysB.has(key);
      if (p.operation === 2) return keysA.has(key) && !keysB.has(key);
      return true;
    });
    return [
      strokes(toPaths(edgesToPaths(points, edgesA), 'cool', 3, 0.5)),
      strokes(toPaths(edgesToPaths(points, edgesB), 'violet', 3, 0.5)),
      strokes(toPaths(edgesToPaths(points, result), 'warm', 0.9, 1), 'flow', 0.45),
      dots(toSprites(points, 'primary', 1.5))
    ];
  }
};

const spatialWeightsTranspose: EntryPreview = {
  parameters: [slider('neighbors', 'Neighbors (k)', 1, 3, 1, 2)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const base = createSpreadPoints(fork('points'), 5, 5, 8, 72, 12, 88);
    const edges = getNearestEdges(base, Math.round(p.neighbors));
    const shift = (point: Point, offset: number): Point => [point[0] + offset, point[1]];
    const focus = pointer
      ? findNearestIndex(
          base.map(point => [point[0] + (pointer[0] > 80 ? 80 : 0), point[1]] as Point),
          pointer
        )
      : 12;
    const makePanel = (offset: number, reverse: boolean) => {
      const regular: PathItem[] = [];
      const picked: PathItem[] = [];
      const heads: PathItem[] = [];
      edges.forEach(([from, to], order) => {
        const source = shift(base[reverse ? to : from], offset);
        const target = shift(base[reverse ? from : to], offset);
        const [start, end] = shorten(source, target, 2, 2.6);
        const isPicked = from === focus;
        (isPicked ? picked : regular).push(
          path(
            [start, end],
            isPicked ? 'warm' : 'primary',
            isPicked ? 1.2 : 0.6,
            isPicked ? 1 : 0.6,
            order / edges.length
          )
        );
        heads.push(
          path(arrowHead(start, end, 2.2), isPicked ? 'warm' : 'primary', 0.6, isPicked ? 1 : 0.6)
        );
      });
      return [
        strokes(regular, 'flow', 0.3, 'add', 0.6),
        strokes(picked, 'flow', 0.45),
        strokes(heads)
      ];
    };
    return [
      strokes([
        path(
          [
            [80, 8],
            [80, 92]
          ],
          'dim',
          0.4,
          0.5
        )
      ]),
      ...makePanel(0, false),
      ...makePanel(80, true),
      dots(
        toSprites(
          base.map(point => shift(point, 0)),
          'primary',
          1.6
        )
      ),
      dots(
        toSprites(
          base.map(point => shift(point, 80)),
          'primary',
          1.6
        )
      ),
      dots(
        [
          {
            x: base[focus][0],
            y: base[focus][1],
            radius: 3.4,
            role: 'hot',
            alpha: 1
          },
          {
            x: base[focus][0] + 80,
            y: base[focus][1],
            radius: 3.4,
            role: 'hot',
            alpha: 1
          }
        ],
        0.3
      )
    ];
  }
};

const spatialWeightsSummary: EntryPreview = {
  parameters: [slider('radius', 'Link radius', 12, 26, 0.5, 18, 'px')],
  pointer: 'none',
  build: ({fork, p}) => {
    const points = createSpreadPoints(fork('points'), 6, 5, 8, 76, 12, 88, 0.3);
    const edges: Edge[] = [];
    points.forEach((point, index) =>
      points.forEach((other, otherIndex) => {
        if (otherIndex > index && getDistance(point, other) <= p.radius)
          edges.push([index, otherIndex]);
      })
    );
    const degrees = getAdjacency(points.length, edges).map(list => list.length);
    const binCount = 9;
    const counts = new Array<number>(binCount).fill(0);
    degrees.forEach(degree => counts[Math.min(binCount - 1, degree)]++);
    const tallest = Math.max(...counts, 1);
    const mean = degrees.reduce((sum, degree) => sum + degree, 0) / degrees.length;
    const barWidth = 7;
    const bars: PolygonItem[] = counts.map((count, bin) => {
      const x = 88 + bin * (barWidth + 0.8);
      const height = (count / tallest) * 62;
      return {
        ring: [
          [x, 88],
          [x + barWidth, 88],
          [x + barWidth, 88 - height],
          [x, 88 - height]
        ],
        role: getClassRole(Math.min(4, Math.round((bin / (binCount - 1)) * 4)), 5),
        alpha: 0.75,
        phase: bin / binCount
      };
    });
    const meanX = 88 + (mean + 0.5) * (barWidth + 0.8);
    return [
      strokes(toPaths(edgesToPaths(points, edges), 'dim', 0.5, 0.6)),
      dots(
        points.map((point, index) => ({
          x: point[0],
          y: point[1],
          radius: 1.2 + degrees[index] * 0.35,
          role: getClassRole(Math.min(4, Math.round((degrees[index] / (binCount - 1)) * 4)), 5),
          alpha: 0.95,
          phase: index / points.length
        }))
      ),
      fills(bars, 'pulse'),
      strokes(
        [
          path(
            [
              [84, 88],
              [152, 88]
            ],
            'dim',
            0.5,
            0.8
          ),
          path(
            [
              [meanX, 22],
              [meanX, 90]
            ],
            'warm',
            1.1,
            1
          )
        ],
        'pulse',
        0.5
      )
    ];
  }
};

const spatialWeightsValidation: EntryPreview = {
  parameters: [slider('defects', 'Problems', 0, 4, 1, 3)],
  pointer: 'none',
  build: ({fork, p}) => {
    const points = createSpreadPoints(fork('points'), 9, 5);
    const all = getUniqueEdges(getNearestEdges(points, 2));
    const random = fork('defects');
    const defects = Math.round(p.defects);
    const isolated = defects >= 1 ? random.integer(points.length) : -1;
    const selfLoop = defects >= 2 ? (isolated + 7) % points.length : -1;
    const oneWay =
      defects >= 3
        ? all.filter(edge => edge[0] !== isolated && edge[1] !== isolated)[
            random.integer(all.length - 8)
          ]
        : null;
    const bad =
      defects >= 4
        ? all.filter(edge => edge !== oneWay && edge[0] !== isolated && edge[1] !== isolated)[
            random.integer(all.length - 12)
          ]
        : null;
    const good = all.filter(
      edge => edge[0] !== isolated && edge[1] !== isolated && edge !== oneWay && edge !== bad
    );
    const scan: Point[] = [];
    for (let row = 0; row < 5; row++) {
      const rowPoints = points.slice(row * 9, row * 9 + 9);
      scan.push(...(row % 2 === 0 ? rowPoints : [...rowPoints].reverse()));
    }
    const flagged: SpriteItem[] = [];
    if (isolated >= 0)
      flagged.push({
        x: points[isolated][0],
        y: points[isolated][1],
        radius: 4,
        role: 'hot',
        alpha: 1
      });
    const extras: PathItem[] = [];
    if (selfLoop >= 0)
      extras.push(
        path(
          createCircle(points[selfLoop][0] + 3.2, points[selfLoop][1] - 3.2, 3.4, 18),
          'hot',
          0.9,
          1
        )
      );
    if (oneWay) {
      const [start, end] = shorten(points[oneWay[0]], points[oneWay[1]], 2, 3);
      extras.push(
        path([start, end], 'violet', 1.3, 1),
        path(arrowHead(start, end, 3), 'violet', 1, 1)
      );
    }
    if (bad) extras.push(path([points[bad[0]], points[bad[1]]], 'hot', 1.6, 1));
    return [
      strokes(toPaths(edgesToPaths(points, good), 'primary', 0.6, 0.6)),
      strokes([path(scan, 'cool', 0.8, 0.9)], 'flow', 0.12, 'add', 1.6),
      strokes(extras, 'pulse', 0.6),
      dots(toSprites(points, 'primary', 1.5)),
      dots(flagged, 0.4),
      strokes(
        isolated >= 0
          ? [path(createCircle(points[isolated][0], points[isolated][1], 8, 28), 'hot', 0.6, 0.7)]
          : [],
        'pulse',
        0.5
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Using weights

const spatialLag: EntryPreview = {
  parameters: [slider('neighbors', 'Neighbors (k)', 2, 7, 1, 4)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const random = fork('values');
    const points = createSpreadPoints(fork('points'), 9, 5);
    const values = points.map(([x, y]) =>
      clamp(getSurfaceValue(x, y, 0.6) + random.gaussian() * 0.08, 0, 1)
    );
    const focus = pointer ? findNearestIndex(points, pointer) : 22;
    const neighbors = points
      .map((point, index) => ({
        index,
        distance: getDistance(point, points[focus])
      }))
      .filter(entry => entry.index !== focus)
      .sort((first, second) => first.distance - second.distance)
      .slice(0, Math.round(p.neighbors));
    const lag = neighbors.reduce((sum, entry) => sum + values[entry.index], 0) / neighbors.length;
    const chosen = new Set(neighbors.map(entry => entry.index));
    return [
      strokes(
        neighbors.map((entry, order) =>
          path(
            [points[entry.index], points[focus]],
            getValueRole(values[entry.index]),
            1.1,
            0.9,
            order / 8
          )
        ),
        'flow',
        0.5
      ),
      dots(
        points.map((point, index) => ({
          x: point[0],
          y: point[1],
          radius: chosen.has(index) ? 2.6 : 1.5,
          role: getValueRole(values[index]),
          alpha: chosen.has(index) ? 1 : 0.55,
          phase: index / points.length
        })),
        0.15
      ),
      dots([
        {
          x: points[focus][0],
          y: points[focus][1],
          radius: 3.4,
          role: getValueRole(values[focus]),
          alpha: 1
        }
      ]),
      strokes(
        [
          path(createCircle(points[focus][0], points[focus][1], 7.5, 28), getValueRole(lag), 1.6, 1)
        ],
        'pulse',
        0.5
      )
    ];
  }
};

const neighborhoodSummary: EntryPreview = {
  parameters: [slider('radius', 'Window radius', 1, 3, 1, 2, 'cells')],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const columns = 24;
    const rows = 15;
    const random = fork('cells');
    const smooth = createSmoothField(fork('surface'), 5, 1.4, columns, rows);
    const values = new Float32Array(columns * rows);
    for (let index = 0; index < values.length; index++)
      values[index] = clamp(smooth[index] * 0.8 + random.next() * 0.2, 0, 1);
    const cellWidth = 160 / columns;
    const cellHeight = 100 / rows;
    const focusColumn = pointer ? clamp(Math.floor(pointer[0] / cellWidth), 0, columns - 1) : 11;
    const focusRow = pointer ? clamp(Math.floor(pointer[1] / cellHeight), 0, rows - 1) : 7;
    const radius = Math.round(p.radius);
    const windowCells: PolygonItem[] = [];
    const flows: PathItem[] = [];
    const center: Point = [(focusColumn + 0.5) * cellWidth, (focusRow + 0.5) * cellHeight];
    for (let row = focusRow - radius; row <= focusRow + radius; row++) {
      for (let column = focusColumn - radius; column <= focusColumn + radius; column++) {
        if (
          row < 0 ||
          column < 0 ||
          row >= rows ||
          column >= columns ||
          (row === focusRow && column === focusColumn)
        )
          continue;
        const x = column * cellWidth;
        const y = row * cellHeight;
        windowCells.push({
          ring: [
            [x, y],
            [x + cellWidth, y],
            [x + cellWidth, y + cellHeight],
            [x, y + cellHeight]
          ],
          role: 'warm',
          alpha: 0.35
        });
        flows.push(
          path(
            [[x + cellWidth / 2, y + cellHeight / 2], center],
            'warm',
            0.5,
            0.8,
            (row + column) / 12
          )
        );
      }
    }
    const x0 = focusColumn * cellWidth;
    const y0 = focusRow * cellHeight;
    return [
      {
        type: 'field',
        columns,
        rows,
        values,
        ramp: 'spectral',
        style: 'ramp',
        filter: 'nearest',
        strength: 0.04,
        speed: 0.3
      },
      fills(windowCells, 'pulse'),
      strokes([
        path(
          closeRing([
            [x0 - radius * cellWidth, y0 - radius * cellHeight],
            [x0 + (radius + 1) * cellWidth, y0 - radius * cellHeight],
            [x0 + (radius + 1) * cellWidth, y0 + (radius + 1) * cellHeight],
            [x0 - radius * cellWidth, y0 + (radius + 1) * cellHeight]
          ]),
          'ink',
          0.8,
          0.9
        )
      ]),
      strokes(flows, 'flow', 0.5),
      dots([{x: center[0], y: center[1], radius: 3.4, role: 'hot', alpha: 1}], 0.3)
    ];
  }
};

const pairGather: EntryPreview = {
  parameters: [slider('pairs', 'Pairs', 3, 10, 1, 7)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('table');
    const rows = 8;
    const table: Point[] = Array.from({length: rows}, (_, index) => [
      26,
      12 + (index * 76) / (rows - 1)
    ]);
    const values = table.map(() => random.next());
    const pairCount = Math.round(p.pairs);
    const pickRandom = fork('pairs');
    const lookups = Array.from({length: 10}, () => pickRandom.integer(rows));
    const outputs: Point[] = Array.from({length: pairCount}, (_, index) => [
      134,
      12 + (index * 76) / Math.max(1, pairCount - 1)
    ]);
    const arcs: PathItem[] = [];
    outputs.forEach((output, index) => {
      const source = table[lookups[index]];
      const midX = 80;
      const midY = (source[1] + output[1]) / 2;
      const points: Point[] = Array.from({length: 24}, (_, step) => {
        const t = step / 23;
        const eased = t * t * (3 - 2 * t);
        return [
          mixNumbers(source[0], output[0], t),
          mixNumbers(source[1], output[1], eased) +
            Math.sin(t * Math.PI) * (midX - 80 + (midY - 50) * 0)
        ] as Point;
      });
      arcs.push(path(points, getValueRole(values[lookups[index]]), 1.1, 0.9, index / pairCount));
    });
    return [
      strokes([
        path(
          [
            [26, 6],
            [26, 94]
          ],
          'dim',
          0.4,
          0.5
        ),
        path(
          [
            [134, 6],
            [134, 94]
          ],
          'dim',
          0.4,
          0.5
        )
      ]),
      strokes(arcs, 'flow', 0.4),
      dots(
        table.map(
          (point, index): SpriteItem => ({
            x: point[0],
            y: point[1],
            radius: 3,
            role: getValueRole(values[index]),
            alpha: 0.95,
            phase: index / rows
          })
        )
      ),
      dots(
        outputs.map(
          (point, index): SpriteItem => ({
            x: point[0],
            y: point[1],
            radius: 2.6,
            role: getValueRole(values[lookups[index]]),
            alpha: 1,
            phase: index / pairCount
          })
        ),
        0.3
      )
    ];
  }
};

const spatialWeightsTraces: EntryPreview = {
  parameters: [slider('powers', 'Powers of W', 1, 5, 1, 4)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const points = createSpreadPoints(fork('points'), 7, 5, 8, 106, 12, 88);
    const edges = getUniqueEdges(getNearestEdges(points, 3));
    const adjacency = getAdjacency(points.length, edges);
    const focus = pointer ? findNearestIndex(points, pointer) : 17;
    const hops = getHops(adjacency, focus);
    const powers = Math.round(p.powers);
    const roles: Role[] = ['hot', 'warm', 'violet', 'cool', 'primary', 'dim'];
    const reached = edges.filter(
      ([first, second]) =>
        hops[first] >= 0 &&
        hops[second] >= 0 &&
        Math.max(hops[first], hops[second]) <= powers &&
        Math.abs(hops[first] - hops[second]) === 1
    );
    const directed = reached.map(
      ([first, second]) =>
        (hops[first] < hops[second]
          ? [points[first], points[second]]
          : [points[second], points[first]]) as Point[]
    );
    const bars: PolygonItem[] = Array.from({length: powers}, (_, index) => {
      const x = 118 + index * 6.4;
      const height = 66 * Math.pow(0.62, index);
      return {
        ring: [
          [x, 88],
          [x + 5, 88],
          [x + 5, 88 - height],
          [x, 88 - height]
        ],
        role: roles[index + 1],
        alpha: 0.8,
        phase: index / powers
      };
    });
    return [
      strokes(toPaths(edgesToPaths(points, edges), 'dim', 0.4, 0.45)),
      strokes(
        directed.map((segment, order) => path(segment, 'warm', 0.9, 0.85, order / directed.length)),
        'flow',
        0.4
      ),
      dots(
        points.map((point, index) => ({
          x: point[0],
          y: point[1],
          radius: hops[index] >= 0 && hops[index] <= powers ? 2.8 - hops[index] * 0.3 : 1.2,
          role: hops[index] >= 0 && hops[index] <= powers ? roles[hops[index]] : 'primary',
          alpha: hops[index] >= 0 && hops[index] <= powers ? 1 - hops[index] * 0.12 : 0.4,
          phase: hops[index] / 6
        }))
      ),
      strokes([
        path(
          [
            [114, 88],
            [152, 88]
          ],
          'dim',
          0.5,
          0.8
        )
      ]),
      fills(bars, 'pulse')
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Chunked weights

function buildChunked(
  mode: 'algebra' | 'transform' | 'transpose' | 'validation'
): EntryPreview['build'] {
  return ({fork, p}) => {
    const chunks = Math.round(p.chunks);
    const points = createSpreadPoints(fork('points'), 10, 5, 8, 152, 14, 90);
    const chunkOf = (point: Point) =>
      Math.min(chunks - 1, Math.floor(((point[0] - 4) / 152) * chunks));
    const width = 152 / chunks;
    const bands: PolygonItem[] = Array.from({length: chunks}, (_, index) => ({
      ring: [
        [4 + index * width + 0.8, 4],
        [4 + (index + 1) * width - 0.8, 4],
        [4 + (index + 1) * width - 0.8, 96],
        [4 + index * width + 0.8, 96]
      ],
      role: index % 2 === 0 ? 'cool' : 'violet',
      alpha: 0.1,
      phase: index / chunks
    }));
    const layers: PreviewLayer[] = [fills(bands, 'pulse')];
    const frames = bands.map(band => path(closeRing(band.ring), 'dim', 0.5, 0.6));
    layers.push(strokes(frames));
    const directed = getNearestEdges(points, 2);
    const undirected = getUniqueEdges(directed);
    const crossing = (edge: Edge) => chunkOf(points[edge[0]]) !== chunkOf(points[edge[1]]);
    if (mode === 'algebra') {
      const other = getUniqueEdges(getNearestEdges(points, 3)).filter(
        edge =>
          !undirected.some(
            known => getEdgeKey(known[0], known[1]) === getEdgeKey(edge[0], edge[1])
          ) || edge[0] % 3 === 0
      );
      layers.push(
        strokes(toPaths(edgesToPaths(points, undirected), 'cool', 3, 0.45)),
        strokes(toPaths(edgesToPaths(points, other), 'violet', 3, 0.45))
      );
      layers.push(
        strokes(
          toPaths(
            edgesToPaths(points, getUniqueEdges([...undirected, ...other])),
            'warm',
            0.8,
            1
          ).map((item, index) => ({
            ...item,
            phase: chunkOf(item.points[0]) / chunks + index * 0.001
          })),
          'flow',
          0.35
        )
      );
    } else if (mode === 'transform') {
      const degree = new Array<number>(points.length).fill(0);
      const outgoing = getNearestEdges(points, 1).concat(
        directed.filter((_, index) => index % 5 === 0)
      );
      outgoing.forEach(([from]) => degree[from]++);
      const items = outgoing.map(([from, to]) =>
        path(
          shorten(points[from], points[to], 1.5, 1.5),
          'warm',
          0.4 + 2.8 / degree[from],
          0.95,
          chunkOf(points[from]) / chunks
        )
      );
      layers.push(strokes(items, 'flow', 0.35));
    } else if (mode === 'transpose') {
      const local = directed.filter(edge => !crossing(edge));
      const across = directed.filter(crossing);
      const reversedItems = (list: Edge[], role: Role, widthValue: number) =>
        list.map(([from, to], order) =>
          path(
            shorten(points[to], points[from], 1.8, 2.2),
            role,
            widthValue,
            0.95,
            order / list.length
          )
        );
      const items = reversedItems(local, 'warm', 1);
      const heads = [...local, ...across].map(([from, to]) => {
        const [start, end] = shorten(points[to], points[from], 1.8, 2.2);
        return path(arrowHead(start, end, 2.2), crossing([from, to]) ? 'violet' : 'warm', 0.7, 1);
      });
      layers.push(
        strokes(items, 'flow', 0.35),
        strokes(reversedItems(across, 'violet', 1.3), 'flow', 0.5),
        strokes(heads)
      );
    } else {
      const flagged = Math.min(chunks - 1, 1);
      const scan: Point[] = Array.from(
        {length: chunks},
        (_, index) => [4 + (index + 0.5) * width, 10] as Point
      );
      const trouble = points.findIndex(point => chunkOf(point) === flagged && point[1] > 40);
      layers.push(strokes(toPaths(edgesToPaths(points, undirected), 'primary', 0.6, 0.55)));
      layers.push(strokes([path(scan, 'cool', 1, 0.9)], 'flow', 0.2, 'add', 1.5));
      layers.push(
        dots(
          scan.map(
            (point, index): SpriteItem => ({
              x: point[0],
              y: 10,
              radius: 3,
              role: index === flagged ? 'hot' : 'cool',
              alpha: 1,
              phase: index / chunks
            })
          ),
          0.4
        )
      );
      layers.push(
        strokes(
          [path(createCircle(points[trouble][0], points[trouble][1], 7, 24), 'hot', 0.9, 0.9)],
          'pulse',
          0.6
        )
      );
    }
    layers.push(dots(toSprites(points, 'primary', 1.5, undefined, 0.85)));
    return layers;
  };
}

const CHUNK_PARAMETER = [slider('chunks', 'Chunks', 2, 6, 1, 3)];

// ---------------------------------------------------------------------------------------------
// Regression on a scatter

const ordinaryLeastSquares: EntryPreview = {
  parameters: [
    slider('slope', 'Slope', -0.6, 0.6, 0.05, 0.4),
    slider('noise', 'Noise', 1, 20, 0.5, 7, 'px')
  ],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const random = fork('points');
    const base: Point[] = Array.from({length: 46}, () => {
      const x = random.range(PLOT_LEFT + 4, PLOT_RIGHT - 2);
      return [x, clamp(50 - p.slope * (x - 83) + random.gaussian() * p.noise, 12, 88)] as Point;
    });
    const extra: Point[] = pointer
      ? [[clamp(pointer[0], PLOT_LEFT + 2, PLOT_RIGHT), clamp(pointer[1], 10, 90)]]
      : [];
    const fit = fitLine([...base, ...extra]);
    const plain = fitLine(base);
    const lineAt = (line: {slope: number; intercept: number}, x: number): Point => [
      x,
      line.intercept + line.slope * x
    ];
    const all = [...base, ...extra];
    return [
      createAxes(),
      strokes(
        all.map(point =>
          path([point, [point[0], fit.intercept + fit.slope * point[0]]], 'hot', 0.45, 0.65)
        ),
        'pulse',
        0.4
      ),
      strokes(
        extra.length
          ? [path([lineAt(plain, PLOT_LEFT), lineAt(plain, PLOT_RIGHT)], 'dim', 0.8, 0.6)]
          : []
      ),
      dots(toSprites(base, 'primary', 1.5, fork('sizes'))),
      dots(
        extra.map(
          (point): SpriteItem => ({
            x: point[0],
            y: point[1],
            radius: 3.2,
            role: 'warm',
            alpha: 1
          })
        ),
        0.3
      ),
      strokes(
        [path([lineAt(fit, PLOT_LEFT), lineAt(fit, PLOT_RIGHT)], 'warm', 1.8, 1)],
        'draw',
        0.18
      ),
      dots([{x: fit.meanX, y: fit.meanY, radius: 2.6, role: 'cool', alpha: 1}])
    ];
  }
};

const spatialRegressionDiagnostics: EntryPreview = {
  parameters: [slider('dependence', 'Spatial dependence', 0, 1, 0.05, 0.75)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('noise');
    const points = createSpreadPoints(fork('points'), 6, 5, 8, 76, 12, 88, 0.3);
    const raw = points.map(
      ([x, y]) =>
        p.dependence * Math.sin(x * 0.11 + 1) * Math.cos(y * 0.09) * 1.4 +
        (1 - p.dependence) * random.gaussian() * 0.7 +
        random.gaussian() * 0.12
    );
    const std = Math.sqrt(raw.reduce((sum, value) => sum + value * value, 0) / raw.length) || 1;
    const residuals = raw.map(value => value / std);
    const edges = getUniqueEdges(getNearestEdges(points, 3));
    const adjacency = getAdjacency(points.length, getNearestEdges(points, 3));
    const lags = residuals.map(
      (_, index) =>
        adjacency[index].reduce((sum, other) => sum + residuals[other], 0) /
        (adjacency[index].length || 1)
    );
    const scatter = residuals.map(
      (value, index) =>
        [120 + clamp(value, -2.2, 2.2) * 14, 50 - clamp(lags[index], -2.2, 2.2) * 14] as Point
    );
    const fit = fitLine(scatter);
    return [
      strokes(toPaths(edgesToPaths(points, edges), 'dim', 0.45, 0.5)),
      dots(
        points.map((point, index) => ({
          x: point[0],
          y: point[1],
          radius: 1.4 + Math.min(1, Math.abs(residuals[index])) * 2,
          role: residuals[index] > 0 ? ('hot' as Role) : ('cool' as Role),
          alpha: 0.95,
          phase: index / points.length
        }))
      ),
      fills(
        [
          {
            ring: [
              [120, 50],
              [152, 50],
              [152, 12],
              [120, 12]
            ],
            role: 'hot',
            alpha: 0.06
          },
          {
            ring: [
              [88, 50],
              [120, 50],
              [120, 88],
              [88, 88]
            ],
            role: 'cool',
            alpha: 0.06
          }
        ],
        'static'
      ),
      strokes([
        path(
          [
            [88, 50],
            [152, 50]
          ],
          'dim',
          0.5,
          0.7
        ),
        path(
          [
            [120, 12],
            [120, 88]
          ],
          'dim',
          0.5,
          0.7
        )
      ]),
      dots(
        scatter.map(
          (point, index): SpriteItem => ({
            x: point[0],
            y: point[1],
            radius: 1.4,
            role: residuals[index] * lags[index] > 0 ? 'warm' : 'primary',
            alpha: 0.9,
            phase: index / scatter.length
          })
        ),
        0.2
      ),
      strokes(
        [
          path(
            [
              [88, fit.intercept + fit.slope * 88],
              [152, fit.intercept + fit.slope * 152]
            ],
            'warm',
            1.6,
            1
          )
        ],
        'draw',
        0.2
      )
    ];
  }
};

const probitRegression: EntryPreview = {
  parameters: [slider('steepness', 'Steepness', 0.5, 5, 0.1, 2.4)],
  pointer: 'sticky',
  build: ({fork, p, pointer}) => {
    const center = pointer ? clamp(pointer[0], 40, 126) : 83;
    const random = fork('points');
    const yTop = 20;
    const yBottom = 80;
    const probabilityAt = (x: number) => createNormalCdf((p.steepness * (x - center)) / 30);
    const toY = (probability: number) => yBottom - probability * (yBottom - yTop);
    const observed = Array.from({length: 54}, () => {
      const x = random.range(PLOT_LEFT + 3, PLOT_RIGHT - 3);
      const one = random.next() < probabilityAt(x);
      return {
        x,
        one,
        y: (one ? yTop : yBottom) + random.range(-3, 3) * (one ? -1 : 1) - (one ? 2 : -2)
      };
    });
    const curve: Point[] = Array.from({length: 60}, (_, index) => {
      const x = PLOT_LEFT + (index / 59) * (PLOT_RIGHT - PLOT_LEFT);
      return [x, toY(probabilityAt(x))] as Point;
    });
    const bell: Point[] = Array.from({length: 60}, (_, index) => {
      const x = PLOT_LEFT + (index / 59) * (PLOT_RIGHT - PLOT_LEFT);
      const z = (p.steepness * (x - center)) / 30;
      return [x, 88 - Math.exp(-0.5 * z * z) * 24] as Point;
    });
    return [
      createAxes(),
      strokes([
        path(
          [
            [PLOT_LEFT, toY(0.5)],
            [PLOT_RIGHT, toY(0.5)]
          ],
          'dim',
          0.4,
          0.5
        ),
        path(
          [
            [center, 12],
            [center, 90]
          ],
          'dim',
          0.4,
          0.5
        )
      ]),
      strokes([path(bell, 'violet', 1, 0.8)], 'pulse', 0.4),
      dots(
        observed.map(
          (entry, index): SpriteItem => ({
            x: entry.x,
            y: entry.y,
            radius: 1.5,
            role: entry.one ? 'hot' : 'cool',
            alpha: 0.9,
            phase: index / 54
          })
        )
      ),
      strokes([path(curve, 'warm', 1.9, 1)], 'draw', 0.2),
      dots([{x: center, y: toY(0.5), radius: 3, role: 'warm', alpha: 1}], 0.3)
    ];
  }
};

const seeminglyUnrelatedRegression: EntryPreview = {
  parameters: [slider('correlation', 'Error correlation', -0.95, 0.95, 0.05, 0.85)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('errors');
    const count = 28;
    const xs = Array.from(
      {length: count},
      (_, index) =>
        PLOT_LEFT +
        6 +
        (index / (count - 1)) * (PLOT_RIGHT - PLOT_LEFT - 10) +
        random.range(-1.5, 1.5)
    );
    const first: Point[] = [];
    const second: Point[] = [];
    const matches: PathItem[] = [];
    xs.forEach(x => {
      const z1 = random.gaussian();
      const z2 = random.gaussian();
      const e1 = z1;
      const e2 = p.correlation * z1 + Math.sqrt(1 - p.correlation ** 2) * z2;
      const y1 = clamp(34 - 0.2 * (x - 83) + e1 * 6, 8, 46);
      const y2 = clamp(72 + 0.16 * (x - 83) + e2 * 6, 54, 92);
      first.push([x, y1]);
      second.push([x, y2]);
      matches.push(
        path(
          [
            [x, y1],
            [x, y2]
          ],
          e1 * e2 > 0 ? 'warm' : 'dim',
          0.6,
          e1 * e2 > 0 ? 0.5 : 0.25,
          x / 160
        )
      );
    });
    const fitFirst = fitLine(first);
    const fitSecond = fitLine(second);
    return [
      strokes([
        path(
          [
            [PLOT_LEFT, 8],
            [PLOT_LEFT, 92]
          ],
          'dim',
          0.5,
          0.7
        ),
        path(
          [
            [PLOT_LEFT, 50],
            [PLOT_RIGHT, 50]
          ],
          'dim',
          0.4,
          0.5
        )
      ]),
      strokes(matches, 'flow', 0.3),
      dots(toSprites(first, 'cool', 1.6, undefined, 0.95)),
      dots(toSprites(second, 'violet', 1.6, undefined, 0.95)),
      strokes(
        [
          path(
            [
              [PLOT_LEFT, fitFirst.intercept + fitFirst.slope * PLOT_LEFT],
              [PLOT_RIGHT, fitFirst.intercept + fitFirst.slope * PLOT_RIGHT]
            ],
            'cool',
            1.6,
            1
          ),
          path(
            [
              [PLOT_LEFT, fitSecond.intercept + fitSecond.slope * PLOT_LEFT],
              [PLOT_RIGHT, fitSecond.intercept + fitSecond.slope * PLOT_RIGHT]
            ],
            'violet',
            1.6,
            1
          )
        ],
        'draw',
        0.18
      )
    ];
  }
};

const instrumentalVariablesCovariance: EntryPreview = {
  parameters: [slider('clustering', 'Error clustering', 0, 1, 0.05, 0.7)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('clusters');
    const centers = [30, 56, 84, 112, 138];
    const points: Point[] = [];
    const halos: PathItem[] = [];
    centers.forEach(center => {
      const shock = random.gaussian() * 9 * p.clustering;
      const y = 52 - 0.12 * (center - 83) + shock;
      halos.push(path(createCircle(center, clamp(y, 18, 82), 9, 20), 'dim', 0.4, 0.45));
      for (let index = 0; index < 8; index++)
        points.push([center + random.gaussian() * 4, clamp(y + random.gaussian() * 4.5, 10, 90)]);
    });
    const fit = fitLine(points);
    const lineAt = (x: number) => fit.intercept + fit.slope * x;
    const half = (x: number, factor: number) => 5 * factor * Math.sqrt(1 + ((x - 83) / 50) ** 2);
    const xs = Array.from(
      {length: 30},
      (_, index) => PLOT_LEFT + (index / 29) * (PLOT_RIGHT - PLOT_LEFT)
    );
    const wide = 1 + 2.6 * p.clustering;
    const band: Point[] = [
      ...xs.map(x => [x, lineAt(x) - half(x, wide)] as Point),
      ...[...xs].reverse().map(x => [x, lineAt(x) + half(x, wide)] as Point)
    ];
    const narrow = (sign: number) => xs.map(x => [x, lineAt(x) + sign * half(x, 1)] as Point);
    return [
      createAxes(),
      fills([{ring: band, role: 'violet', alpha: 0.2}], 'pulse'),
      strokes([path(narrow(-1), 'dim', 0.7, 0.8), path(narrow(1), 'dim', 0.7, 0.8)]),
      strokes(halos),
      dots(toSprites(points, 'primary', 1.4, undefined, 0.9)),
      strokes(
        [
          path(
            [
              [PLOT_LEFT, lineAt(PLOT_LEFT)],
              [PLOT_RIGHT, lineAt(PLOT_RIGHT)]
            ],
            'warm',
            1.7,
            1
          )
        ],
        'draw',
        0.18
      )
    ];
  }
};

const spatialTwoStageLeastSquares: EntryPreview = {
  parameters: [slider('strength', 'Instrument strength', 0.2, 1, 0.05, 0.7)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('data');
    const count = 30;
    const instrument = Array.from({length: count}, () => random.next());
    const lagged = instrument.map(z =>
      clamp(0.15 + 0.7 * z + random.gaussian() * 0.32 * (1.15 - p.strength), 0.03, 0.97)
    );
    const outcome = lagged.map(w => clamp(0.2 + 0.6 * w + random.gaussian() * 0.07, 0.03, 0.97));
    const stageOne = fitLine(instrument.map((z, index) => [z, lagged[index]] as Point));
    const fitted = instrument.map(z => clamp(stageOne.intercept + stageOne.slope * z, 0.03, 0.97));
    const toLeft = (u: number, v: number): Point => [10 + u * 62, 88 - v * 74];
    const toRight = (u: number, v: number): Point => [90 + u * 62, 88 - v * 74];
    const raw = instrument.map((z, index) => toLeft(z, lagged[index]));
    const hat = instrument.map((z, index) => toLeft(z, fitted[index]));
    const second = fitted.map((w, index) => toRight(w, outcome[index]));
    const stageTwo = fitLine(fitted.map((w, index) => [w, outcome[index]] as Point));
    return [
      strokes([
        path(
          [
            [10, 10],
            [10, 88],
            [74, 88]
          ],
          'dim',
          0.5,
          0.7
        ),
        path(
          [
            [90, 10],
            [90, 88],
            [154, 88]
          ],
          'dim',
          0.5,
          0.7
        )
      ]),
      strokes(raw.map((point, index) => path([point, hat[index]], 'dim', 0.4, 0.5))),
      strokes(
        [
          path(
            [toLeft(0, stageOne.intercept), toLeft(1, stageOne.intercept + stageOne.slope)],
            'cool',
            1.5,
            1
          )
        ],
        'draw',
        0.18
      ),
      dots(toSprites(raw, 'hot', 1.4, undefined, 0.8)),
      dots(toSprites(hat, 'cool', 1.5)),
      strokes(
        [
          path(
            [
              [77, 50],
              [86, 50]
            ],
            'violet',
            1.5,
            1
          ),
          path(arrowHead([77, 50], [86, 50], 3), 'violet', 1.3, 1)
        ],
        'flow',
        0.6
      ),
      dots(toSprites(second, 'primary', 1.5)),
      strokes(
        [
          path(
            [toRight(0, stageTwo.intercept), toRight(1, stageTwo.intercept + stageTwo.slope)],
            'warm',
            1.8,
            1
          )
        ],
        'draw',
        0.18
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Spatial lag and error models

const spatialLagMaximumLikelihood: EntryPreview = {
  parameters: [slider('rho', 'Spillover (rho)', 0, 0.95, 0.05, 0.55)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const field = createSmoothField(fork('field'), 7, mixNumbers(3.2, 0.9, p.rho));
    const points = createSpreadPoints(fork('points'), 7, 4, 8, 108, 12, 70);
    const focus = pointer ? findNearestIndex(points, pointer) : 16;
    const neighbors = points
      .map((point, index) => ({
        index,
        distance: getDistance(point, points[focus])
      }))
      .filter(entry => entry.index !== focus)
      .sort((first, second) => first.distance - second.distance)
      .slice(0, 4);
    const likelihood = (rho: number) => -Math.pow((rho - 0.55) * 2.1, 2);
    const curve: Point[] = Array.from({length: 26}, (_, index) => {
      const rho = index / 25;
      return [114 + rho * 36, 92 - (1 + likelihood(rho)) * 18] as Point;
    });
    const marker = curve[Math.round(p.rho * 25)];
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values: field,
        ramp: 'heat',
        style: 'ramp',
        strength: 0.06,
        speed: 0.25
      },
      strokes(
        neighbors.map((entry, order) =>
          path([points[entry.index], points[focus]], 'ink', 1.3, 0.95, order / 4)
        ),
        'flow',
        0.5
      ),
      dots(toSprites(points, 'primary', 1.5, undefined, 0.75)),
      dots(
        [
          {
            x: points[focus][0],
            y: points[focus][1],
            radius: 4,
            role: 'warm',
            alpha: 1
          },
          ...neighbors.map(
            (entry): SpriteItem => ({
              x: points[entry.index][0],
              y: points[entry.index][1],
              radius: 2.6,
              role: 'ink',
              alpha: 1
            })
          )
        ],
        0.3
      ),
      strokes([
        path(
          [
            [112, 94],
            [152, 94]
          ],
          'dim',
          0.5,
          0.9
        ),
        path(curve, 'ink', 1.3, 1)
      ]),
      dots([{x: marker[0], y: marker[1], radius: 3, role: 'hot', alpha: 1}], 0.3)
    ];
  }
};

const spatialLagImpacts: EntryPreview = {
  parameters: [slider('rho', 'Spillover (rho)', 0, 0.9, 0.05, 0.6)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const points = createSpreadPoints(fork('points'), 7, 5, 8, 108, 12, 88);
    const edges = getUniqueEdges(getNearestEdges(points, 3));
    const adjacency = getAdjacency(points.length, edges);
    const focus = pointer ? findNearestIndex(points, pointer) : 17;
    const hops = getHops(adjacency, focus);
    const impactAt = (hop: number, rho: number) => (hop < 0 ? 0 : Math.pow(rho, hop));
    const direct = 28;
    const indirect = Math.min(58, (direct * p.rho) / (1 - p.rho));
    const directed = edges
      .filter(([first, second]) => hops[first] >= 0 && Math.abs(hops[first] - hops[second]) === 1)
      .map(
        ([first, second]) =>
          (hops[first] < hops[second]
            ? [points[first], points[second]]
            : [points[second], points[first]]) as Point[]
      );
    const bar = (x: number, from: number, length: number, role: Role) =>
      path(
        [
          [x, 90 - from],
          [x, 90 - from - Math.max(length, 0.1)]
        ],
        role,
        7,
        0.9
      );
    return [
      strokes(toPaths(edgesToPaths(points, edges), 'dim', 0.4, 0.4)),
      strokes(
        directed.map((segment, order) =>
          path(segment, 'warm', 1, 0.5 + 0.4 * p.rho, order / directed.length)
        ),
        'flow',
        0.45
      ),
      dots(
        points.map((point, index) => ({
          x: point[0],
          y: point[1],
          radius: 1.3 + 4 * impactAt(hops[index], p.rho),
          role: index === focus ? ('hot' as Role) : ('cool' as Role),
          alpha: 0.4 + 0.6 * impactAt(hops[index], p.rho),
          phase: hops[index] / 6
        }))
      ),
      strokes([
        path(
          [
            [118, 92],
            [152, 92]
          ],
          'dim',
          0.5,
          0.8
        )
      ]),
      strokes(
        [
          bar(124, 0, direct, 'hot'),
          bar(136, direct, indirect, 'cool'),
          bar(148, 0, direct, 'hot'),
          bar(148, direct, indirect, 'cool')
        ],
        'pulse',
        0.4
      )
    ];
  }
};

const spatialErrorMaximumLikelihood: EntryPreview = {
  parameters: [slider('lambda', 'Error dependence (lambda)', 0, 0.95, 0.05, 0.8)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('errors');
    const base = createSpreadPoints(fork('points'), 7, 4, 8, 72, 14, 86, 0.3);
    const edges = getNearestEdges(base, 3);
    const adjacency = getAdjacency(base.length, edges);
    const errors = base.map(
      ([x, y]) => Math.sin(x * 0.09 + 0.4) * Math.cos(y * 0.1) * 1.2 + random.gaussian() * 0.25
    );
    const lagged = errors.map(
      (_, index) =>
        adjacency[index].reduce((sum, other) => sum + errors[other], 0) /
        (adjacency[index].length || 1)
    );
    const whitened = errors.map((value, index) => value - p.lambda * lagged[index]);
    const normalize = (values: number[]) => {
      const std =
        Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length) || 1;
      return values.map(value => value / std);
    };
    const left = normalize(errors);
    const right = normalize(whitened);
    const makeDots = (values: number[], offset: number) =>
      base.map(
        (point, index): SpriteItem => ({
          x: point[0] + offset,
          y: point[1],
          radius: 1.2 + Math.min(1, Math.abs(values[index])) * 2.6,
          role: values[index] > 0 ? 'hot' : 'cool',
          alpha: 0.95,
          phase: index / base.length
        })
      );
    return [
      strokes(toPaths(edgesToPaths(base, getUniqueEdges(edges)), 'dim', 0.4, 0.4)),
      dots(makeDots(left, 0)),
      strokes(
        [
          path(
            [
              [77, 50],
              [84, 50]
            ],
            'violet',
            1.6,
            1
          ),
          path(arrowHead([77, 50], [84, 50], 3), 'violet', 1.3, 1)
        ],
        'flow',
        0.6
      ),
      dots(makeDots(right, 80)),
      strokes([
        path(
          [
            [80, 8],
            [80, 92]
          ],
          'dim',
          0.3,
          0.3
        )
      ])
    ];
  }
};

const spatialErrorGm: EntryPreview = {
  parameters: [slider('lambda', 'True lambda', 0.2, 0.9, 0.05, 0.6)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const objectiveAt = (lambda: number) =>
      74 - 62 * (1 - 4.4 * Math.pow(lambda - p.lambda, 2) * (1 + 0.5 * (lambda - p.lambda)));
    const toX = (lambda: number) => 20 + lambda * 100;
    const curve: Point[] = Array.from({length: 50}, (_, index) => {
      const lambda = -0.15 + (index / 49) * 1.15;
      return [toX(lambda), clamp(objectiveAt(lambda), 8, 90)] as Point;
    });
    const trial = pointer ? clamp((pointer[0] - 20) / 100, -0.1, 0.95) : 0.12;
    const trialPoint: Point = [toX(trial), clamp(objectiveAt(trial), 8, 90)];
    const best: Point = [toX(p.lambda), clamp(objectiveAt(p.lambda), 8, 90)];
    const random = fork('moments');
    const pairs: Point[] = Array.from({length: 26}, () => {
      const e = random.gaussian();
      return [
        134 + clamp(e, -2, 2) * 8,
        24 - clamp(p.lambda * e + random.gaussian() * 0.5, -2, 2) * 8
      ] as Point;
    });
    const pairFit = fitLine(pairs);
    const trace: Point[] = Array.from({length: 14}, (_, index) => {
      const t = index / 13;
      const lambda = mixNumbers(trial, p.lambda, t);
      return [toX(lambda), clamp(objectiveAt(lambda), 8, 90)] as Point;
    });
    return [
      strokes([
        path(
          [
            [20, 8],
            [20, 92],
            [124, 92]
          ],
          'dim',
          0.5,
          0.7
        )
      ]),
      strokes([path(curve, 'violet', 1.6, 0.95)]),
      strokes([path(trace, 'warm', 1.4, 1)], 'flow', 0.35),
      strokes([path([trialPoint, [trialPoint[0], 92]], 'dim', 0.6, 0.7)]),
      dots(
        [
          {
            x: trialPoint[0],
            y: trialPoint[1],
            radius: 3,
            role: 'primary',
            alpha: 1
          },
          {x: best[0], y: best[1], radius: 3.6, role: 'warm', alpha: 1}
        ],
        0.3
      ),
      strokes(
        [
          path(
            [
              [best[0], best[1]],
              [best[0], 92]
            ],
            'warm',
            0.6,
            0.6
          )
        ],
        'pulse',
        0.5
      ),
      strokes([
        path(
          [
            [118, 24],
            [152, 24]
          ],
          'dim',
          0.3,
          0.5
        ),
        path(
          [
            [134, 8],
            [134, 40]
          ],
          'dim',
          0.3,
          0.5
        )
      ]),
      dots(toSprites(pairs, 'primary', 1.2, undefined, 0.8)),
      strokes(
        [
          path(
            [
              [118, pairFit.intercept + pairFit.slope * 118],
              [152, pairFit.intercept + pairFit.slope * 152]
            ],
            'warm',
            1.2,
            1
          )
        ],
        'draw',
        0.2
      )
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Regimes

const spatialRegimeRegression: EntryPreview = {
  parameters: [slider('difference', 'Regime difference', 0, 1, 0.05, 0.85)],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('points');
    const boundary = 83;
    const slopes = [-0.5 * p.difference, 0.55 * p.difference];
    const offsets = [60 - 10 * p.difference, 40 + 8 * p.difference];
    const regions = [
      {
        from: PLOT_LEFT,
        to: boundary - 2,
        slope: slopes[0],
        offset: offsets[0],
        role: 'cool' as Role
      },
      {
        from: boundary + 2,
        to: PLOT_RIGHT,
        slope: slopes[1],
        offset: offsets[1],
        role: 'warm' as Role
      }
    ];
    const layers: PreviewLayer[] = [
      fills(
        [
          {
            ring: [
              [PLOT_LEFT, 8],
              [boundary, 8],
              [boundary, 92],
              [PLOT_LEFT, 92]
            ],
            role: 'cool',
            alpha: 0.06
          },
          {
            ring: [
              [boundary, 8],
              [PLOT_RIGHT + 4, 8],
              [PLOT_RIGHT + 4, 92],
              [boundary, 92]
            ],
            role: 'warm',
            alpha: 0.06
          }
        ],
        'pulse'
      ),
      createAxes(),
      strokes(
        [
          path(
            [
              [boundary, 8],
              [boundary, 92]
            ],
            'violet',
            0.8,
            0.8
          )
        ],
        'pulse',
        0.4
      )
    ];
    const everything: Point[] = [];
    const lines: PathItem[] = [];
    regions.forEach(region => {
      const points: Point[] = Array.from({length: 24}, () => {
        const x = random.range(region.from + 2, region.to);
        return [
          x,
          clamp(
            region.offset -
              region.slope * (x - (region.from + region.to) / 2) +
              random.gaussian() * 6,
            10,
            90
          )
        ] as Point;
      });
      everything.push(...points);
      layers.push(dots(toSprites(points, region.role, 1.5)));
      const fit = fitLine(points);
      lines.push(
        path(
          [
            [region.from, fit.intercept + fit.slope * region.from],
            [region.to, fit.intercept + fit.slope * region.to]
          ],
          region.role,
          1.9,
          1
        )
      );
    });
    const pooled = fitLine(everything);
    layers.push(
      strokes([
        path(
          [
            [PLOT_LEFT, pooled.intercept + pooled.slope * PLOT_LEFT],
            [PLOT_RIGHT, pooled.intercept + pooled.slope * PLOT_RIGHT]
          ],
          'dim',
          0.8,
          0.7
        )
      ])
    );
    layers.push(strokes(lines, 'draw', 0.2));
    return layers;
  }
};

const restrictedRegimeRegression: EntryPreview = {
  parameters: [
    slider('gap', 'Intercept gap', 0, 1, 0.05, 0.7),
    slider('slope', 'Shared slope', -0.6, 0.6, 0.05, 0.35)
  ],
  pointer: 'none',
  build: ({fork, p}) => {
    const random = fork('points');
    const roles: Role[] = ['cool', 'violet', 'warm'];
    const layers: PreviewLayer[] = [createAxes()];
    const lines: PathItem[] = [];
    const markers: PathItem[] = [];
    roles.forEach((role, index) => {
      const intercept = 50 + (index - 1) * 24 * p.gap;
      const lineAt = (x: number): Point => [x, intercept - p.slope * (x - 83)];
      const points: Point[] = Array.from({length: 18}, () => {
        const x = random.range(PLOT_LEFT + 4, PLOT_RIGHT - 2);
        return [x, clamp(lineAt(x)[1] + random.gaussian() * 4.5, 8, 92)] as Point;
      });
      layers.push(dots(toSprites(points, role, 1.4, undefined, 0.85)));
      lines.push(path([lineAt(PLOT_LEFT), lineAt(PLOT_RIGHT)], role, 1.7, 1));
      if (index > 0) {
        const previous = 50 + (index - 2) * 24 * p.gap - p.slope * (PLOT_LEFT - 83);
        const here = lineAt(PLOT_LEFT)[1];
        markers.push(
          path(
            [
              [PLOT_LEFT + 5, previous],
              [PLOT_LEFT + 5, here]
            ],
            'ink',
            0.9,
            0.9
          ),
          path(arrowHead([PLOT_LEFT + 5, previous], [PLOT_LEFT + 5, here], 2.4), 'ink', 0.8, 0.9)
        );
      }
    });
    layers.push(strokes(markers, 'pulse', 0.5), strokes(lines, 'draw', 0.2));
    return layers;
  }
};

// ---------------------------------------------------------------------------------------------
// Geographically weighted regression

const GLYPH_COLUMNS = 10;
const GLYPH_ROWS = 6;

function getGlyphCenters(): Point[] {
  const centers: Point[] = [];
  for (let row = 0; row < GLYPH_ROWS; row++) {
    for (let column = 0; column < GLYPH_COLUMNS; column++)
      centers.push([14 + column * 14.6, 14 + row * 14.4]);
  }
  return centers;
}

function getLocalCoefficient(x: number, y: number): number {
  return 1.35 * Math.sin(x * 0.036 + 0.5) * Math.cos(y * 0.052 - 0.3);
}

const geographicallyWeightedRegression: EntryPreview = {
  parameters: [slider('bandwidth', 'Bandwidth', 10, 40, 1, 22, 'px')],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const origin: Point = pointer ?? [62, 46];
    const weightAt = (x: number, y: number) =>
      Math.exp(-0.5 * (getDistance([x, y], origin) / p.bandwidth) ** 2);
    const buckets: PathItem[][] = [[], [], []];
    let weightSum = 0;
    let coefficientSum = 0;
    getGlyphCenters().forEach(([x, y]) => {
      const beta = getLocalCoefficient(x, y);
      const angle = Math.atan(-beta);
      const weight = weightAt(x, y);
      weightSum += weight;
      coefficientSum += weight * beta;
      const dx = Math.cos(angle) * 4.2;
      const dy = Math.sin(angle) * 4.2;
      buckets[weight > 0.5 ? 2 : weight > 0.18 ? 1 : 0].push(
        path(
          [
            [x - dx, y - dy],
            [x + dx, y + dy]
          ],
          weight > 0.5 ? 'warm' : weight > 0.18 ? 'primary' : 'dim',
          weight > 0.5 ? 1.5 : 1,
          0.25 + 0.75 * Math.sqrt(weight)
        )
      );
    });
    const localSlope = coefficientSum / (weightSum || 1);
    const localAngle = Math.atan(-localSlope);
    const random = fork('observations');
    const observations: SpriteItem[] = Array.from({length: 70}, () => {
      const x = random.range(8, 152);
      const y = random.range(8, 92);
      const weight = weightAt(x, y);
      return {
        x,
        y,
        radius: 0.8 + 1.4 * weight,
        role: weight > 0.3 ? 'warm' : 'primary',
        alpha: 0.2 + 0.7 * weight,
        phase: random.next()
      };
    });
    return [
      dots(observations, 0.15),
      strokes(buckets[0]),
      strokes(buckets[1]),
      strokes(buckets[2]),
      strokes(
        [
          path(createCircle(origin[0], origin[1], p.bandwidth, 44), 'cool', 1, 0.9),
          path(createCircle(origin[0], origin[1], p.bandwidth * 2, 56), 'cool', 0.5, 0.35)
        ],
        'pulse',
        0.4
      ),
      strokes(
        [
          path(
            [
              [origin[0] - Math.cos(localAngle) * 13, origin[1] - Math.sin(localAngle) * 13],
              [origin[0] + Math.cos(localAngle) * 13, origin[1] + Math.sin(localAngle) * 13]
            ],
            'hot',
            2.2,
            1
          )
        ],
        'draw',
        0.3
      ),
      dots([{x: origin[0], y: origin[1], radius: 2.6, role: 'hot', alpha: 1}], 0.3)
    ];
  }
};

const generalizedGwr: EntryPreview = {
  parameters: [slider('bandwidth', 'Bandwidth', 10, 40, 1, 24, 'px')],
  pointer: 'hover',
  build: ({p, pointer}) => {
    const origin: Point = pointer ?? [100, 50];
    const weightAt = (x: number, y: number) =>
      Math.exp(-0.5 * (getDistance([x, y], origin) / p.bandwidth) ** 2);
    const getSteepness = (x: number, y: number) =>
      clamp(0.5 + 0.5 * Math.sin(x * 0.04 - 0.6) * Math.cos(y * 0.05 + 0.4) * 1.5, 0, 1);
    const buildCurve = (cx: number, cy: number, halfWidth: number, steepness: number): Point[] =>
      Array.from({length: 12}, (_, index) => {
        const t = index / 11;
        const z = (t - 0.5) * 2;
        const logistic = 1 / (1 + Math.exp(-(0.6 + steepness * 7) * z));
        return [cx + z * halfWidth, cy + halfWidth * 0.8 - logistic * halfWidth * 1.6] as Point;
      });
    const items: Record<Role, PathItem[]> = {
      primary: [],
      warm: [],
      hot: [],
      cool: [],
      violet: [],
      ink: [],
      dim: []
    };
    let weightSum = 0;
    let steepSum = 0;
    getGlyphCenters().forEach(([x, y]) => {
      const steepness = getSteepness(x, y);
      const weight = weightAt(x, y);
      weightSum += weight;
      steepSum += weight * steepness;
      const role: Role = steepness > 0.66 ? 'hot' : steepness > 0.33 ? 'violet' : 'cool';
      items[role].push(
        path(buildCurve(x, y, 5, steepness), role, 0.7 + weight, 0.2 + 0.8 * Math.sqrt(weight))
      );
    });
    const local = steepSum / (weightSum || 1);
    return [
      strokes(items.cool),
      strokes(items.violet),
      strokes(items.hot),
      strokes(
        [path(createCircle(origin[0], origin[1], p.bandwidth, 44), 'ink', 0.9, 0.85)],
        'pulse',
        0.4
      ),
      strokes([path(buildCurve(origin[0], origin[1], 14, local), 'warm', 2.2, 1)], 'draw', 0.3),
      dots([{x: origin[0], y: origin[1], radius: 2.4, role: 'warm', alpha: 1}], 0.3)
    ];
  }
};

const multiscaleGwr: EntryPreview = {
  parameters: [slider('rough', 'Local share', 0, 1, 0.05, 0.45)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const smooth = createSmoothField(fork('smooth'), 4, 0.7);
    const fine = createSmoothField(fork('fine'), 6, 3.6);
    const values = new Float32Array(smooth.length);
    for (let index = 0; index < values.length; index++)
      values[index] = mixNumbers(smooth[index], fine[index], p.rough);
    const origin: Point = pointer ?? [80, 50];
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values,
        ramp: 'spectral',
        style: 'ramp',
        strength: 0.05,
        speed: 0.25
      },
      strokes(
        [
          path(createCircle(origin[0], origin[1], 38, 56), 'cool', 1.1, 0.95),
          path(createCircle(origin[0], origin[1], 21, 44), 'violet', 1.1, 0.95),
          path(createCircle(origin[0], origin[1], 8, 30), 'hot', 1.2, 1)
        ],
        'pulse',
        0.4
      ),
      dots([{x: origin[0], y: origin[1], radius: 2.4, role: 'ink', alpha: 1}], 0.3)
    ];
  }
};

const gwrNonstationarityTest: EntryPreview = {
  parameters: [slider('observed', 'Observed spread', 0.4, 3, 0.05, 2.1)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const random = fork('permutations');
    const bins = 26;
    const maxValue = 3.2;
    const counts = new Array<number>(bins).fill(0);
    for (let draw = 0; draw < 240; draw++) {
      let sum = 0;
      for (let index = 0; index < 3; index++) sum += random.gaussian() ** 2;
      const value = (sum / 3) * 0.95;
      counts[Math.min(bins - 1, Math.floor((value / maxValue) * bins))]++;
    }
    const tallest = Math.max(...counts, 1);
    const left = 18;
    const right = 150;
    const binWidth = (right - left) / bins;
    const observed = pointer
      ? clamp(((pointer[0] - left) / (right - left)) * maxValue, 0.2, maxValue)
      : p.observed;
    const observedX = left + (observed / maxValue) * (right - left);
    const normal: PolygonItem[] = [];
    const tail: PolygonItem[] = [];
    counts.forEach((count, bin) => {
      const x = left + bin * binWidth;
      const height = (count / tallest) * 60;
      const item: PolygonItem = {
        ring: [
          [x + 0.4, 90],
          [x + binWidth - 0.4, 90],
          [x + binWidth - 0.4, 90 - height],
          [x + 0.4, 90 - height]
        ],
        role: x >= observedX ? 'hot' : 'cool',
        alpha: 0.7,
        phase: bin / bins
      };
      (x >= observedX ? tail : normal).push(item);
    });
    const shuffleLeft: Point[] = Array.from({length: 6}, (_, index) => [114, 10 + index * 5]);
    const order = [3, 0, 5, 1, 4, 2];
    return [
      strokes([
        path(
          [
            [left, 92],
            [right, 92]
          ],
          'dim',
          0.5,
          0.8
        )
      ]),
      fills(normal, 'pulse'),
      fills(tail, 'pulse'),
      strokes(
        [
          path(
            [
              [observedX, 24],
              [observedX, 92]
            ],
            'warm',
            1.5,
            1
          )
        ],
        'draw',
        0.3
      ),
      dots([{x: observedX, y: 24, radius: 3.4, role: 'warm', alpha: 1}], 0.3),
      strokes(
        shuffleLeft.map((point, index) =>
          path([point, [146, 10 + order[index] * 5]], 'violet', 0.6, 0.8, index / 6)
        ),
        'flow',
        0.4
      ),
      dots(
        shuffleLeft.flatMap((point, index): SpriteItem[] => [
          {
            x: point[0],
            y: point[1],
            radius: 1.1,
            role: 'primary',
            alpha: 0.9
          },
          {
            x: 146,
            y: 10 + index * 5,
            radius: 1.1,
            role: 'primary',
            alpha: 0.9
          }
        ]),
        0.1
      )
    ];
  }
};

const localCollinearity: EntryPreview = {
  parameters: [slider('threshold', 'Flag above', 0.3, 0.9, 0.05, 0.6)],
  pointer: 'hover',
  build: ({fork, p, pointer}) => {
    const getCollinearity = (x: number, y: number) =>
      clamp(0.5 + 0.7 * Math.sin(x * 0.042 + 1.2) * Math.cos(y * 0.06 + 0.4), 0, 1);
    const values = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) =>
      getCollinearity(u * 160, v * 100)
    );
    const buildEllipse = (cx: number, cy: number, size: number, collinearity: number): Point[] => {
      const major = size;
      const minor = size * (1 - 0.88 * collinearity);
      const cos = Math.SQRT1_2;
      return Array.from({length: 22}, (_, index) => {
        const angle = (index / 21) * Math.PI * 2;
        const a = Math.cos(angle) * major;
        const b = Math.sin(angle) * minor;
        return [cx + a * cos - b * cos, cy - (a * cos + b * cos)] as Point;
      });
    };
    const flagged: PathItem[] = [];
    const calm: PathItem[] = [];
    getGlyphCenters().forEach(([x, y]) => {
      const collinearity = getCollinearity(x, y);
      (collinearity > p.threshold ? flagged : calm).push(
        path(
          buildEllipse(x, y, 5.6, collinearity),
          collinearity > p.threshold ? 'hot' : 'cool',
          0.9,
          0.85
        )
      );
    });
    const origin: Point = pointer ?? [60, 40];
    const here = getCollinearity(origin[0], origin[1]);
    const random = fork('scatter');
    const cloud: SpriteItem[] = Array.from({length: 26}, () => {
      const along = random.gaussian() * 7.5;
      const across = random.gaussian() * 7.5 * (1 - 0.88 * here);
      return {
        x: origin[0] + (along - across) * Math.SQRT1_2,
        y: origin[1] - (along + across) * Math.SQRT1_2,
        radius: 0.9,
        role: 'ink',
        alpha: 0.95,
        phase: random.next()
      };
    });
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values,
        ramp: 'heat',
        style: 'ramp',
        strength: 0.05,
        speed: 0.25
      },
      strokes(calm),
      strokes(flagged, 'pulse', 0.5),
      strokes(
        [
          path(createCircle(origin[0], origin[1], 15, 36), 'ink', 0.9, 0.9),
          path(
            buildEllipse(origin[0], origin[1], 11, here),
            here > p.threshold ? 'hot' : 'warm',
            1.6,
            1
          )
        ],
        'pulse',
        0.4
      ),
      dots(cloud, 0.2)
    ];
  }
};

// ---------------------------------------------------------------------------------------------
// Panels: places stacked through time

type PanelMode = 'regression' | 'lag' | 'error';

function buildPanel(mode: PanelMode): EntryPreview['build'] {
  return ({fork, p}) => {
    const periods = Math.round(p.periods);
    const columns = 6;
    const rows = 3;
    const step = 56 / periods;
    const project = (u: number, v: number, period: number): Point => [
      14 + u * 76 + v * 34 + period * 3,
      90 - period * step - v * 20
    ];
    const random = fork('places');
    const places = Array.from({length: columns * rows}, (_, index) => ({
      u: ((index % columns) + 0.5) / columns,
      v: (Math.floor(index / columns) + 0.5) / rows,
      effect: random.next()
    }));
    const focus = 8;
    const layers: PreviewLayer[] = [];
    const slabs: PolygonItem[] = [];
    const slabEdges: PathItem[] = [];
    for (let period = periods - 1; period >= 0; period--) {
      const ring = [
        project(0, 0, period),
        project(1, 0, period),
        project(1, 1, period),
        project(0, 1, period)
      ];
      slabs.push({ring, role: 'primary', alpha: 0.06 + 0.02 * period});
      slabEdges.push(path(closeRing(ring), 'dim', 0.5, 0.65));
    }
    layers.push(fills(slabs), strokes(slabEdges));
    const links: PathItem[] = [];
    const spatial: PathItem[] = [];
    const sprites: SpriteItem[] = [];
    const noise = fork('residuals');
    for (let period = 0; period < periods; period++) {
      places.forEach((place, index) => {
        const point = project(place.u, place.v, period);
        const residual =
          Math.sin(place.u * 7 + period * 1.3) * Math.cos(place.v * 5 - period) +
          noise.gaussian() * 0.25;
        if (period > 0) {
          const below = project(place.u, place.v, period - 1);
          const picked = mode === 'regression' && index === focus;
          const colored = mode === 'error';
          links.push(
            path(
              [below, point],
              picked ? 'warm' : colored ? getClassRole(Math.floor(place.effect * 5), 5) : 'dim',
              picked ? 1.3 : colored ? 0.8 : 0.4,
              picked ? 1 : colored ? 0.55 : 0.4,
              period / periods
            )
          );
        }
        if (mode === 'lag') {
          const column = index % columns;
          const row = Math.floor(index / columns);
          if (column + 1 < columns)
            spatial.push(
              path(
                [point, project(places[index + 1].u, places[index + 1].v, period)],
                'warm',
                0.5 + p.rho * 1.2,
                0.35 + 0.5 * p.rho,
                (period + column) / 10
              )
            );
          if (row + 1 < rows)
            spatial.push(
              path(
                [point, project(places[index + columns].u, places[index + columns].v, period)],
                'warm',
                0.5 + p.rho * 1.2,
                0.35 + 0.5 * p.rho,
                (period + row) / 10
              )
            );
        }
        sprites.push({
          x: point[0],
          y: point[1],
          radius:
            mode === 'error'
              ? 1.2 + Math.min(1, Math.abs(residual)) * 1.8
              : index === focus && mode === 'regression'
                ? 2.6
                : 1.5,
          role:
            mode === 'error'
              ? residual > 0
                ? 'hot'
                : 'cool'
              : index === focus && mode === 'regression'
                ? 'warm'
                : 'primary',
          alpha: 0.95,
          phase: (index + period) / 20
        });
      });
    }
    layers.push(strokes(links, 'flow', 0.3, 'add', 0.8));
    if (mode === 'lag') layers.push(strokes(spatial, 'flow', 0.5));
    if (mode === 'regression') {
      const arrowStart: Point = [128, 80];
      const arrowEnd: Point = [150, 44];
      layers.push(
        strokes(
          [
            path([arrowStart, arrowEnd], 'warm', 1.6, 1),
            path(arrowHead(arrowStart, arrowEnd, 3.4), 'warm', 1.3, 1)
          ],
          'flow',
          0.4
        )
      );
    }
    layers.push(dots(sprites, 0.2));
    return layers;
  };
}

const PANEL_PARAMETERS = [slider('periods', 'Periods', 2, 6, 1, 4)];

export const REGRESSION_WEIGHTS_PREVIEWS: Record<string, EntryPreview> = {
  'contiguity-weights': contiguityWeights,
  'fuzzy-contiguity': fuzzyContiguity,
  'lattice-weights': latticeWeights,
  'neighbor-search': neighborSearch,
  'island-attachment': islandAttachment,
  'spatial-weights-transform': spatialWeightsTransform,
  'spatial-weights-algebra': spatialWeightsAlgebra,
  'spatial-weights-transpose': spatialWeightsTranspose,
  'spatial-weights-summary': spatialWeightsSummary,
  'spatial-weights-validation': spatialWeightsValidation,
  'spatial-lag': spatialLag,
  'neighborhood-summary': neighborhoodSummary,
  'pair-gather': pairGather,
  'spatial-weights-traces': spatialWeightsTraces,
  'chunked-weights-algebra': {
    parameters: CHUNK_PARAMETER,
    pointer: 'none',
    build: buildChunked('algebra')
  },
  'chunked-weights-transform': {
    parameters: CHUNK_PARAMETER,
    pointer: 'none',
    build: buildChunked('transform')
  },
  'chunked-weights-transpose': {
    parameters: CHUNK_PARAMETER,
    pointer: 'none',
    build: buildChunked('transpose')
  },
  'chunked-weights-validation': {
    parameters: CHUNK_PARAMETER,
    pointer: 'none',
    build: buildChunked('validation')
  },
  'ordinary-least-squares': ordinaryLeastSquares,
  'spatial-regression-diagnostics': spatialRegressionDiagnostics,
  'probit-regression': probitRegression,
  'seemingly-unrelated-regression': seeminglyUnrelatedRegression,
  'instrumental-variables-covariance': instrumentalVariablesCovariance,
  'spatial-two-stage-least-squares': spatialTwoStageLeastSquares,
  'spatial-lag-maximum-likelihood': spatialLagMaximumLikelihood,
  'spatial-lag-impacts': spatialLagImpacts,
  'spatial-error-maximum-likelihood': spatialErrorMaximumLikelihood,
  'spatial-error-gm': spatialErrorGm,
  'spatial-regime-regression': spatialRegimeRegression,
  'restricted-regime-regression': restrictedRegimeRegression,
  'geographically-weighted-regression': geographicallyWeightedRegression,
  'generalized-gwr': generalizedGwr,
  'multiscale-gwr': multiscaleGwr,
  'gwr-nonstationarity-test': gwrNonstationarityTest,
  'local-collinearity': localCollinearity,
  'spatial-panel-regression': {
    parameters: PANEL_PARAMETERS,
    pointer: 'none',
    build: buildPanel('regression')
  },
  'spatial-panel-lag-ml': {
    parameters: [...PANEL_PARAMETERS, slider('rho', 'Spillover (rho)', 0, 0.95, 0.05, 0.6)],
    pointer: 'none',
    build: buildPanel('lag')
  },
  'spatial-panel-error-ml': {
    parameters: PANEL_PARAMETERS,
    pointer: 'none',
    build: buildPanel('error')
  }
};
