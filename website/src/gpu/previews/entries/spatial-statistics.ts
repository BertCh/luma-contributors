// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/spatial-statistics.ts, keyed by entry id.

import {
  clamp,
  createArc,
  createCircle,
  createClusterPoints,
  createScatterPoints,
  createSmoothField,
  getDistance,
  type Random
} from '../data.ts';
import {closeRing} from '../geometry.ts';
import {getClassRole, toSprites, type EntryPreview} from '../helpers.ts';
import {
  createChoice,
  createFillLayer,
  createSlider,
  createSpriteLayer,
  createStrokeLayer
} from '../layers.ts';
import type {
  FillLayer,
  PathItem,
  Point,
  PolygonItem,
  PreviewLayer,
  Role,
  SpriteItem,
  StrokeLayer
} from '../types.ts';

// ---------------------------------------------------------------------------------------------
// Local helpers: lattices of areal units, small plots and statistics shared by several entries.

type Layout = {
  columns: number;
  rows: number;
  size: number;
  x: number;
  y: number;
};

/** A 16 by 10 lattice of areal units covering the scene. */
const MAP_LAYOUT: Layout = {columns: 16, rows: 10, size: 9, x: 8, y: 5};

const ROLE_CYCLE: Role[] = ['primary', 'warm', 'violet', 'cool', 'hot'];

/** Rectangle ring from its top-left corner and size. */
function createSizedRectangle(x: number, y: number, width: number, height: number): Point[] {
  return [
    [x, y],
    [x + width, y],
    [x + width, y + height],
    [x, y + height]
  ];
}

function getCellCenter(layout: Layout, index: number): Point {
  const column = index % layout.columns;
  const row = Math.floor(index / layout.columns);
  return [layout.x + (column + 0.5) * layout.size, layout.y + (row + 0.5) * layout.size];
}

function getCellRing(layout: Layout, index: number, inset = 0.5): Point[] {
  const column = index % layout.columns;
  const row = Math.floor(index / layout.columns);
  return createSizedRectangle(
    layout.x + column * layout.size + inset,
    layout.y + row * layout.size + inset,
    layout.size - inset * 2,
    layout.size - inset * 2
  );
}

function getCellAt(layout: Layout, point: Point | null): number | null {
  if (!point) return null;
  const column = Math.floor((point[0] - layout.x) / layout.size);
  const row = Math.floor((point[1] - layout.y) / layout.size);
  if (column < 0 || row < 0 || column >= layout.columns || row >= layout.rows) return null;
  return row * layout.columns + column;
}

/** Cells within `radius` rings (Chebyshev), excluding the cell itself; radius 1 is queen contiguity. */
function getQueenNeighbors(layout: Layout, index: number, radius = 1): number[] {
  const column = index % layout.columns;
  const row = Math.floor(index / layout.columns);
  const neighbors: number[] = [];
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const neighborColumn = column + dx;
      const neighborRow = row + dy;
      if (
        (dx || dy) &&
        neighborColumn >= 0 &&
        neighborRow >= 0 &&
        neighborColumn < layout.columns &&
        neighborRow < layout.rows
      ) {
        neighbors.push(neighborRow * layout.columns + neighborColumn);
      }
    }
  }
  return neighbors;
}

function standardize(values: number[]): number[] {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const deviation =
    Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length) || 1;
  return values.map(value => (value - mean) / deviation);
}

/** Spatially smooth z-scores, one per lattice cell. */
function createLatticeValues(random: Random, layout: Layout, frequency = 1.2): number[] {
  return standardize(
    Array.from(createSmoothField(random, 5, frequency, layout.columns, layout.rows))
  );
}

/** Flips the sign of a few cells so the map has local outliers (HL and LH). */
function injectContrast(random: Random, values: number[], count: number, amount = 1.6): void {
  for (let step = 0; step < count; step++) {
    const index = random.integer(values.length);
    values[index] = -values[index] * amount + (random.next() - 0.5);
  }
}

function getSpatialLag(layout: Layout, values: number[]): number[] {
  return values.map((_, index) => {
    const neighbors = getQueenNeighbors(layout, index);
    return neighbors.reduce((sum, neighbor) => sum + values[neighbor], 0) / neighbors.length;
  });
}

function getQuadrantRole(value: number, lag: number): Role {
  if (value >= 0) return lag >= 0 ? 'hot' : 'warm';
  return lag < 0 ? 'cool' : 'violet';
}

function getIndexOfMaximum(values: number[]): number {
  let best = 0;
  values.forEach((value, index) => {
    if (value > values[best]) best = index;
  });
  return best;
}

/** Ranks values into 0..1 so a class ramp always uses its whole range. */
function getRanks(values: number[]): number[] {
  const order = values
    .map((value, index) => ({value, index}))
    .sort((first, second) => first.value - second.value);
  const ranks: number[] = [];
  order.forEach((entry, rank) => {
    ranks[entry.index] = rank / Math.max(1, values.length - 1);
  });
  return ranks;
}

function getAxes(left: number, top: number, right: number, bottom: number): PathItem {
  return {
    points: [
      [left, top],
      [left, bottom],
      [right, bottom]
    ],
    role: 'dim',
    width: 0.5,
    alpha: 0.7
  };
}

/** Links from one cell to its neighbors, thickness by weight; the shape of a spatial weights row. */
function createCellLinks(
  layout: Layout,
  index: number,
  neighbors: number[],
  widths?: number[]
): PreviewLayer[] {
  const center = getCellCenter(layout, index);
  return [
    createStrokeLayer(
      neighbors.map((neighbor, position) => ({
        points: [center, getCellCenter(layout, neighbor)],
        role: 'warm' as Role,
        width: widths ? widths[position] : 0.8,
        alpha: 0.95,
        phase: position / neighbors.length
      })),
      {animation: 'flow', speed: 0.5, blend: 'add'}
    ),
    createStrokeLayer(
      [
        {
          points: closeRing(getCellRing(layout, index, 0)),
          role: 'warm',
          width: 0.9,
          alpha: 1
        }
      ],
      {animation: 'pulse', speed: 0.5, blend: 'add'}
    )
  ];
}

function createPulsingFillLayer(items: PolygonItem[]): FillLayer {
  return createFillLayer(items, {animation: 'pulse', blend: 'add'});
}

function createOutlineLayer(rings: Point[][], role: Role = 'dim', alpha = 0.5): StrokeLayer {
  return createStrokeLayer(
    rings.map(ring => ({
      points: closeRing(ring),
      role,
      width: 0.35,
      alpha
    })),
    {animation: 'static'}
  );
}

function createLatticeOutline(layout: Layout, alpha = 0.35): StrokeLayer {
  const rings: Point[][] = [];
  for (let index = 0; index < layout.columns * layout.rows; index++) {
    rings.push(getCellRing(layout, index, 0.3));
  }
  return createOutlineLayer(rings, 'dim', alpha);
}

const HOTSPOT_THRESHOLD_PARAMETER = createSlider(
  'threshold',
  'Significance',
  0.2,
  3,
  0.1,
  0.5,
  '|I|'
);

const AUTOCORRELATION_PREVIEWS: Record<string, EntryPreview> = {
  'local-morans-i': {
    pointer: 'hover',
    parameters: [
      HOTSPOT_THRESHOLD_PARAMETER,
      createSlider('outliers', 'Local outliers', 0, 14, 1, 7)
    ],
    build: ({fork, p, pointer}) => {
      const layout = MAP_LAYOUT;
      const values = createLatticeValues(fork('values'), layout);
      injectContrast(fork('outliers'), values, Math.round(p.outliers));
      const lag = getSpatialLag(layout, values);
      const localI = values.map((value, index) => value * lag[index]);
      const items: PolygonItem[] = values.map((value, index) => {
        const significant = Math.abs(localI[index]) > p.threshold;
        return {
          ring: getCellRing(layout, index),
          role: significant ? getQuadrantRole(value, lag[index]) : 'ink',
          alpha: significant ? 0.35 + Math.min(0.5, Math.abs(localI[index]) * 0.18) : 0.07,
          phase: (index % layout.columns) / layout.columns
        };
      });
      const probe = getCellAt(layout, pointer) ?? getIndexOfMaximum(localI);
      return [
        createPulsingFillLayer(items),
        createLatticeOutline(layout),
        ...createCellLinks(layout, probe, getQueenNeighbors(layout, probe))
      ];
    }
  },

  'local-gearys-c': {
    pointer: 'hover',
    parameters: [createSlider('contrasts', 'Contrasts', 0, 14, 1, 6)],
    build: ({fork, p, pointer}) => {
      const layout = MAP_LAYOUT;
      const values = createLatticeValues(fork('values'), layout, 1);
      const contrast = fork('contrast');
      for (let step = 0; step < p.contrasts; step++) {
        values[contrast.integer(values.length)] += contrast.range(-2.8, 2.8);
      }
      const scores = values.map((value, index) => {
        const neighbors = getQueenNeighbors(layout, index);
        return (
          neighbors.reduce((sum, neighbor) => sum + (value - values[neighbor]) ** 2, 0) /
          neighbors.length
        );
      });
      const ranks = getRanks(scores);
      const items: PolygonItem[] = scores.map((_, index) => ({
        ring: getCellRing(layout, index),
        role: getClassRole(Math.min(4, Math.floor(ranks[index] * 5)), 5),
        // Dissimilarity: only the ends of the scale glow, similar places and contrasts.
        alpha: 0.08 + Math.abs(ranks[index] - 0.5) * 1.1,
        phase: ranks[index]
      }));
      const probe = getCellAt(layout, pointer) ?? getIndexOfMaximum(scores);
      const neighbors = getQueenNeighbors(layout, probe);
      // Link thickness is the squared difference to that neighbor, the term Geary sums.
      const widths = neighbors.map(
        neighbor => 0.3 + Math.min(3, (values[probe] - values[neighbor]) ** 2 * 0.5)
      );
      return [createPulsingFillLayer(items), ...createCellLinks(layout, probe, neighbors, widths)];
    }
  },

  'hot-spot-analysis-gi-star': {
    pointer: 'hover',
    parameters: [createSlider('window', 'Neighborhood', 1, 3, 1, 2, 'cells')],
    build: ({fork, p, pointer}) => {
      const layout = MAP_LAYOUT;
      const smooth = createLatticeValues(fork('values'), layout, 1);
      const noise = fork('noise');
      const values = standardize(smooth.map(value => value + noise.gaussian() * 0.5));
      const radius = Math.round(p.window);
      const scores = values.map((_, index) => {
        // Gi* includes the focal cell itself; with z-scored values the z-score is the sum over sqrt(n).
        const members = [index, ...getQueenNeighbors(layout, index, radius)];
        return members.reduce((sum, member) => sum + values[member], 0) / Math.sqrt(members.length);
      });
      const items: PolygonItem[] = scores.map((score, index) => {
        const bin =
          Math.abs(score) >= 2.58
            ? 3
            : Math.abs(score) >= 1.96
              ? 2
              : Math.abs(score) >= 1.65
                ? 1
                : 0;
        return {
          ring: getCellRing(layout, index),
          role: bin === 0 ? 'ink' : score > 0 ? 'hot' : 'cool',
          alpha: bin === 0 ? 0.06 : 0.2 + bin * 0.25,
          phase: (index % layout.columns) / layout.columns
        };
      });
      const probe = getCellAt(layout, pointer) ?? getIndexOfMaximum(scores);
      const column = probe % layout.columns;
      const row = Math.floor(probe / layout.columns);
      const left = Math.max(0, column - radius);
      const right = Math.min(layout.columns - 1, column + radius);
      const top = Math.max(0, row - radius);
      const bottom = Math.min(layout.rows - 1, row + radius);
      const window = createSizedRectangle(
        layout.x + left * layout.size,
        layout.y + top * layout.size,
        (right - left + 1) * layout.size,
        (bottom - top + 1) * layout.size
      );
      return [
        createPulsingFillLayer(items),
        createLatticeOutline(layout, 0.2),
        createStrokeLayer(
          [
            {
              points: closeRing(window),
              role: 'warm',
              width: 1.1,
              alpha: 1
            }
          ],
          {animation: 'pulse', speed: 0.5, blend: 'add'}
        ),
        createSpriteLayer([{...toCenter(layout, probe), radius: 3, role: 'warm', alpha: 0.9}], {
          pulse: 0.5
        })
      ];
    }
  },

  'global-spatial-statistics': {
    parameters: [createSlider('clustering', 'Clustering', 0, 1, 0.05, 0.8)],
    build: ({fork, p}) => {
      const layout: Layout = {columns: 10, rows: 8, size: 6, x: 8, y: 26};
      const smooth = createLatticeValues(fork('values'), layout, 1);
      const noise = fork('noise');
      const values = smooth.map(
        value => value * p.clustering + noise.gaussian() * (1 - p.clustering)
      );
      const items: PolygonItem[] = values.map((value, index) => ({
        ring: getCellRing(layout, index, 0.3),
        role: value >= 0 ? 'hot' : 'cool',
        alpha: 0.12 + Math.min(0.55, Math.abs(value) * 0.3),
        phase: (index % layout.columns) / layout.columns
      }));
      // One summary bar per statistic: Moran, Geary (inverted: clustered gives c below 1), G, bivariate, join count.
      const baseline = 52;
      const magnitudes = [1, -0.85, 0.7, 0.45, 0.6].map(value => value * p.clustering * 34);
      const bars: PolygonItem[] = [];
      const caps: SpriteItem[] = [];
      magnitudes.forEach((magnitude, index) => {
        const x = 84 + index * 13;
        const height = clamp(magnitude, -34, 34);
        const outside = Math.abs(height) > 11;
        bars.push({
          ring: createSizedRectangle(
            x,
            Math.min(baseline, baseline - height),
            8,
            Math.abs(height) + 0.6
          ),
          role: outside ? (height > 0 ? 'hot' : 'cool') : 'dim',
          alpha: outside ? 0.7 : 0.4,
          phase: index / 5
        });
        caps.push({
          x: x + 4,
          y: baseline - height,
          radius: 2.2,
          role: outside ? 'warm' : 'dim',
          alpha: 0.95,
          phase: index / 5
        });
      });
      const flows: PathItem[] = [20, 32, 44].map((y, index) => ({
        points: [
          [70, y + 12],
          [74, y + 12 - index * 2],
          [80, baseline + (index - 1) * 5]
        ],
        role: 'primary',
        width: 0.7,
        alpha: 0.7,
        phase: index / 3
      }));
      return [
        createPulsingFillLayer(items),
        createOutlineLayer([createSizedRectangle(layout.x, layout.y, 60, 48)], 'dim', 0.5),
        // The no-evidence band: z within about two standard errors of zero.
        createFillLayer(
          [{ring: createSizedRectangle(80, baseline - 11, 68, 22), role: 'ink', alpha: 0.1}],
          {animation: 'static'}
        ),
        createStrokeLayer(
          [
            {
              points: [
                [80, baseline],
                [150, baseline]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.8
            },
            ...flows
          ],
          {animation: 'flow', speed: 0.4, blend: 'add'}
        ),
        createPulsingFillLayer(bars),
        createSpriteLayer(caps, {pulse: 0.4})
      ];
    }
  }
};

function toCenter(layout: Layout, index: number): {x: number; y: number} {
  const [x, y] = getCellCenter(layout, index);
  return {x, y};
}

// Per-variable local Geary values on a lattice: mean squared difference to the queen neighbors.
function getLocalGeary(layout: Layout, values: number[]): number[] {
  return values.map((value, index) => {
    const neighbors = getQueenNeighbors(layout, index);
    return (
      neighbors.reduce((sum, neighbor) => sum + (value - values[neighbor]) ** 2, 0) /
      neighbors.length
    );
  });
}

/** Three related variables sharing latent structure and a few contrast cells. */
function createVariables(
  fork: (label: string) => Random,
  layout: Layout,
  count: number
): number[][] {
  const shared = createLatticeValues(fork('shared'), layout, 1);
  const spikes = fork('spikes');
  const cells = Array.from({length: 6}, () => spikes.integer(shared.length));
  return Array.from({length: count}, (_, variable) => {
    const own = createLatticeValues(fork(`variable-${variable}`), layout, 1.1);
    const values = shared.map((value, index) => value * 0.8 + own[index] * 0.5);
    for (const cell of cells) values[cell] += (variable % 2 ? -1 : 1) * 2.2;
    return values;
  });
}

const BELL_HEIGHTS = [0.06, 0.18, 0.4, 0.7, 0.95, 0.95, 0.7, 0.4, 0.18, 0.06];

const WEIGHTS_AND_TESTS_PREVIEWS: Record<string, EntryPreview> = {
  'spatial-pearson': {
    parameters: [createSlider('correlation', 'Correlation', -1, 1, 0.05, 0.7)],
    build: ({fork, p}) => {
      const random = fork('pairs');
      const centerX = 112;
      const centerY = 50;
      const correlation = p.correlation;
      const points: SpriteItem[] = [];
      for (let index = 0; index < 56; index++) {
        const x = clamp(random.gaussian(), -2.4, 2.4);
        const y = clamp(
          correlation * x + Math.sqrt(1 - correlation * correlation) * random.gaussian(),
          -2.4,
          2.4
        );
        // Pairs that agree with the overall sign of the association glow; the rest stay quiet.
        const agrees = x * y * (correlation >= 0 ? 1 : -1) > 0;
        points.push({
          x: centerX + x * 14,
          y: centerY - y * 14,
          radius: 1.5,
          role: agrees ? 'warm' : 'dim',
          alpha: agrees ? 0.9 : 0.55,
          phase: random.next()
        });
      }
      // A small neighbor graph: x is read at the focal place i, y at its neighbors j.
      const focal: Point[] = [
        [14, 26],
        [14, 52],
        [14, 78]
      ];
      const neighbors: Point[] = [
        [46, 14],
        [46, 36],
        [46, 56],
        [46, 74],
        [46, 90]
      ];
      const links = [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4]
      ];
      return [
        createStrokeLayer(
          [
            getAxes(centerX - 40, 8, centerX + 40, 92),
            {
              points: [
                [centerX - 40, centerY],
                [centerX + 40, centerY]
              ],
              role: 'dim',
              width: 0.3,
              alpha: 0.4
            },
            {
              points: [
                [centerX, 8],
                [centerX, 92]
              ],
              role: 'dim',
              width: 0.3,
              alpha: 0.4
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(
          links.map(([from, to], index) => ({
            points: [neighbors[to], focal[from]],
            role: 'violet' as Role,
            width: 0.8,
            alpha: 0.9,
            phase: index / links.length
          })),
          {animation: 'flow', speed: 0.35, blend: 'add'}
        ),
        createSpriteLayer(
          [...toSprites(focal, 'primary', 3.2), ...toSprites(neighbors, 'violet', 2.4)],
          {pulse: 0.3}
        ),
        createSpriteLayer(points, {pulse: 0.3}),
        createStrokeLayer(
          [
            {
              points: [
                [centerX - 34, centerY + correlation * 34],
                [centerX + 34, centerY - correlation * 34]
              ],
              role: 'hot',
              width: 1.1,
              alpha: 0.95
            }
          ],
          {animation: 'pulse', speed: 0.4, blend: 'add'}
        )
      ];
    }
  },

  'gamma-statistic': {
    parameters: [
      createSlider('bandwidth', 'Neighbor reach', 1, 4, 1, 2),
      createChoice('similarity', 'Similarity', ['Cross-product', 'Squared difference'])
    ],
    build: ({fork, p}) => {
      const size = 10;
      const cell = 4.4;
      const top = 10;
      const origins = [6, 58, 110];
      const random = fork('values');
      const phase = random.range(0, 6);
      const values = standardize(
        Array.from(
          {length: size},
          (_, index) => Math.sin(index * 0.8 + phase) + random.gaussian() * 0.25
        )
      );
      const squared = p.similarity >= 1;
      const weights = new Array<boolean>(size * size);
      const similarity = new Array<number>(size * size);
      let total = 0;
      let edges = 0;
      for (let row = 0; row < size; row++) {
        for (let column = 0; column < size; column++) {
          const index = row * size + column;
          weights[index] = row !== column && Math.abs(row - column) <= p.bandwidth;
          similarity[index] = squared
            ? (values[row] - values[column]) ** 2
            : values[row] * values[column];
          if (weights[index]) {
            total += similarity[index];
            edges++;
          }
        }
      }
      const matrix = (origin: number, fill: (index: number) => {role: Role; alpha: number}) => {
        const items: PolygonItem[] = [];
        for (let index = 0; index < size * size; index++) {
          const {role, alpha} = fill(index);
          items.push({
            ring: createSizedRectangle(
              origin + (index % size) * cell + 0.25,
              top + Math.floor(index / size) * cell + 0.25,
              cell - 0.5,
              cell - 0.5
            ),
            role,
            alpha,
            phase: (index % size) / size
          });
        }
        return items;
      };
      const similarityFill = (index: number) =>
        squared
          ? {
              role: 'warm' as Role,
              alpha: 0.08 + Math.min(0.7, similarity[index] * 0.16)
            }
          : {
              role: (similarity[index] >= 0 ? 'hot' : 'cool') as Role,
              alpha: 0.08 + Math.min(0.7, Math.abs(similarity[index]) * 0.35)
            };
      const mean = edges ? total / edges : 0;
      const length = squared
        ? clamp(mean / 3, 0.04, 1) * 136
        : clamp(0.5 + mean * 0.35, 0.04, 1) * 136;
      const arrows: PathItem[] = [
        {
          points: [
            [origins[0] + 46, top + 22],
            [origins[1] - 3, top + 22]
          ],
          role: 'primary',
          width: 0.7,
          alpha: 0.9
        },
        {
          points: [
            [origins[1] + 46, top + 22],
            [origins[2] - 3, top + 22]
          ],
          role: 'primary',
          width: 0.7,
          alpha: 0.9
        },
        {
          points: [
            [origins[2] + 22, top + 48],
            [origins[2] + 22, 72],
            [18, 72],
            [18, 79]
          ],
          role: 'warm',
          width: 0.7,
          alpha: 0.9
        }
      ];
      return [
        createFillLayer(
          [
            ...matrix(origins[0], index => ({
              role: 'primary',
              alpha: weights[index] ? 0.7 : 0.06
            })),
            ...matrix(origins[1], similarityFill),
            ...matrix(origins[2], index =>
              weights[index] ? similarityFill(index) : {role: 'ink', alpha: 0.04}
            )
          ],
          {animation: 'pulse', blend: 'add'}
        ),
        createOutlineLayer(
          origins.map(origin => createSizedRectangle(origin, top, size * cell, size * cell)),
          'dim',
          0.5
        ),
        createStrokeLayer(arrows, {animation: 'flow', speed: 0.45, blend: 'add'}),
        createStrokeLayer(
          [
            {
              points: [
                [12, 84],
                [148, 84]
              ],
              role: 'dim',
              width: 1,
              alpha: 0.5
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(
          [
            {
              points: [
                [12, 84],
                [12 + length, 84]
              ],
              role: 'warm',
              width: 2.2,
              alpha: 1
            }
          ],
          {animation: 'draw', speed: 0.25, blend: 'add'}
        )
      ];
    }
  },

  'local-spatial-heteroscedasticity': {
    parameters: [createSlider('noise', 'Noisy area', 0, 1.5, 0.05, 1)],
    build: ({fork, p}) => {
      const layout = MAP_LAYOUT;
      const base = createLatticeValues(fork('values'), layout, 0.9);
      const noise = fork('noise');
      const centers: Point[] = [
        [noise.range(9, 13), noise.range(2.5, 7)],
        [noise.range(2, 5), noise.range(1.5, 3)]
      ];
      const values = base.map((value, index) => {
        const column = index % layout.columns;
        const row = Math.floor(index / layout.columns);
        const near = Math.min(
          ...centers.map(([centerX, centerY]) =>
            Math.hypot(column - centerX, (row - centerY) * 1.1)
          )
        );
        const patch = Math.max(0, 1 - near / 4);
        return value * 0.5 + noise.gaussian() * patch * p.noise * 1.6;
      });
      const scores = values.map((value, index) => {
        const members = [index, ...getQueenNeighbors(layout, index)];
        const mean = members.reduce((sum, member) => sum + values[member], 0) / members.length;
        return Math.sqrt(
          members.reduce((sum, member) => sum + (values[member] - mean) ** 2, 0) / members.length
        );
      });
      const ranks = getRanks(scores);
      const items: PolygonItem[] = scores.map((_, index) => ({
        ring: getCellRing(layout, index),
        role: ranks[index] > 0.85 ? 'hot' : ranks[index] > 0.6 ? 'warm' : 'cool',
        alpha: ranks[index] > 0.6 ? 0.2 + (ranks[index] - 0.6) * 1.8 : 0.04 + ranks[index] * 0.12,
        phase: ranks[index]
      }));
      // Each cell's own residual is a small dot; LOSH is how unevenly those dots vary around it.
      const dots: SpriteItem[] = values.map((value, index) => {
        const [x, y] = getCellCenter(layout, index);
        return {
          x,
          y,
          radius: 0.3 + Math.min(1.4, Math.abs(value) * 0.45),
          role: 'ink',
          alpha: 0.5,
          phase: ranks[index]
        };
      });
      return [
        createPulsingFillLayer(items),
        createLatticeOutline(layout, 0.2),
        createSpriteLayer(dots, {pulse: 0.2})
      ];
    }
  },

  'multivariate-local-gearys-c': {
    parameters: [createSlider('variables', 'Variables', 1, 3, 1, 3)],
    build: ({fork, p}) => {
      const panel: Layout = {columns: 10, rows: 7, size: 3.6, x: 6, y: 6};
      const result: Layout = {columns: 10, rows: 7, size: 8, x: 72, y: 22};
      const variables = createVariables(fork, panel, 3);
      const active = Math.round(p.variables);
      const scores = new Array<number>(panel.columns * panel.rows).fill(0);
      variables.slice(0, active).forEach(values => {
        getLocalGeary(panel, values).forEach((score, index) => {
          scores[index] += score / active;
        });
      });
      const ranks = getRanks(scores);
      const panels: PolygonItem[] = [];
      const arcs: PathItem[] = [];
      variables.forEach((values, variable) => {
        const offset = {...panel, y: panel.y + variable * 30};
        values.forEach((value, index) => {
          panels.push({
            ring: getCellRing(offset, index, 0.2),
            role: value >= 0 ? 'hot' : 'cool',
            alpha: variable < active ? 0.1 + Math.min(0.6, Math.abs(value) * 0.25) : 0.05,
            phase: variable / 3
          });
        });
        arcs.push({
          points: createArc(
            [panel.x + 36.5, offset.y + 12.6],
            [70, 50],
            variable === 1 ? 0 : variable === 0 ? 0.15 : -0.15,
            16
          ),
          role: 'violet',
          width: 0.7,
          alpha: variable < active ? 0.9 : 0.15,
          phase: variable / 3
        });
      });
      const items: PolygonItem[] = scores.map((_, index) => ({
        ring: getCellRing(result, index),
        role: ranks[index] < 0.5 ? 'cool' : 'warm',
        // Low means neighbors agree on every variable at once, so the cool end glows most.
        alpha:
          ranks[index] < 0.5 ? 0.1 + (0.5 - ranks[index]) * 1.2 : 0.04 + (ranks[index] - 0.5) * 0.3,
        phase: ranks[index]
      }));
      return [
        createPulsingFillLayer(panels),
        createOutlineLayer(
          [0, 1, 2].map(variable =>
            createSizedRectangle(panel.x, panel.y + variable * 30, 36, 25.2)
          ),
          'dim',
          0.5
        ),
        createStrokeLayer(arcs, {animation: 'flow', speed: 0.4, blend: 'add'}),
        createPulsingFillLayer(items),
        createLatticeOutline(result, 0.25)
      ];
    }
  }
};

// ----------------------------------------------------------------------------- permutation
const PERMUTATION_PREVIEWS: Record<string, EntryPreview> = {
  'global-permutation-test': {
    pointer: 'sticky',
    parameters: [createSlider('observed', 'Observed statistic', -1, 4.5, 0.1, 2, 'σ')],
    build: ({fork, p, pointer}) => {
      const bins = 26;
      const centerX = 80;
      const scale = 17;
      const baseline = 82;
      const noise = fork('histogram');
      const observed = pointer ? clamp((pointer[0] - centerX) / scale, -3.8, 4.5) : p.observed;
      const observedX = centerX + observed * scale;
      const bars: PolygonItem[] = [];
      const width = 136 / bins;
      for (let bin = 0; bin < bins; bin++) {
        const x = 12 + bin * width;
        const sigma = (x + width / 2 - centerX) / scale;
        const height = Math.exp(-(sigma * sigma) / 2) * 50 * (0.9 + noise.next() * 0.2);
        const inTail = sigma >= observed;
        bars.push({
          ring: createSizedRectangle(x + 0.4, baseline - height, width - 0.8, height),
          role: inTail ? 'hot' : 'cool',
          alpha: inTail ? 0.85 : 0.4,
          phase: bin / bins
        });
      }
      // The 999 shuffled maps as a rug of tiny dots under the histogram.
      const rug = fork('rug');
      const dots: SpriteItem[] = Array.from({length: 90}, () => ({
        x: clamp(centerX + rug.gaussian() * scale, 12, 148),
        y: 90 + rug.range(-1.5, 1.5),
        radius: 0.8,
        role: 'ink' as Role,
        alpha: 0.45,
        phase: rug.next()
      }));
      return [
        createPulsingFillLayer(bars),
        createStrokeLayer(
          [
            {
              points: [
                [10, baseline],
                [150, baseline]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.8
            }
          ],
          {animation: 'static'}
        ),
        createSpriteLayer(dots, {pulse: 0.5}),
        createStrokeLayer(
          [
            {
              points: [
                [observedX, 10],
                [observedX, baseline + 3]
              ],
              role: 'warm',
              width: 1.2,
              alpha: 1
            }
          ],
          {animation: 'pulse', speed: 0.5, blend: 'add'}
        ),
        createSpriteLayer([{x: observedX, y: 10, radius: 3.4, role: 'warm', alpha: 1}], {
          pulse: 0.4
        })
      ];
    }
  },

  'local-permutation-test': {
    pointer: 'sticky',
    parameters: [
      createSlider('shuffles', 'Shuffles shown', 1, 12, 1, 3),
      createSlider('threshold', 'Significance', 0.2, 3, 0.1, 1, '|I|')
    ],
    build: ({fork, p, pointer}) => {
      const layout: Layout = {columns: 12, rows: 9, size: 8, x: 6, y: 14};
      const values = createLatticeValues(fork('values'), layout);
      injectContrast(fork('outliers'), values, 4);
      const lag = getSpatialLag(layout, values);
      const localI = values.map((value, index) => value * lag[index]);
      const items: PolygonItem[] = values.map((value, index) => {
        const survives = Math.abs(localI[index]) > p.threshold;
        return {
          ring: getCellRing(layout, index),
          role: survives ? getQuadrantRole(value, lag[index]) : 'ink',
          alpha: survives ? 0.55 : 0.07,
          phase: (index % layout.columns) / layout.columns
        };
      });
      const probe = getCellAt(layout, pointer) ?? getIndexOfMaximum(localI);
      const center = getCellCenter(layout, probe);
      const shuffle = fork('shuffle');
      const fans: PathItem[] = [];
      for (let permutation = 0; permutation < p.shuffles; permutation++) {
        // Each permutation hands the focal cell a random set of neighbors.
        for (let link = 0; link < 3; link++) {
          fans.push({
            points: createArc(
              center,
              getCellCenter(layout, shuffle.integer(values.length)),
              shuffle.range(-0.15, 0.15),
              12
            ),
            role: 'violet',
            width: 0.5,
            alpha: 0.8,
            phase: permutation / p.shuffles
          });
        }
      }
      // The null distribution of this one cell's local statistic, with the observed value marked.
      const observed = clamp(133 + localI[probe] * 8, 114, 150);
      const bars: PolygonItem[] = BELL_HEIGHTS.map((height, index) => {
        const x = 114 + index * 4.2;
        return {
          ring: createSizedRectangle(x, 74 - height * 36, 3.6, height * 36),
          role: x + 2 >= observed ? 'hot' : 'violet',
          alpha: x + 2 >= observed ? 0.85 : 0.45,
          phase: index / 10
        };
      });
      return [
        createPulsingFillLayer(items),
        createLatticeOutline(layout, 0.2),
        createStrokeLayer(fans, {animation: 'flow', speed: 0.5, trail: 0.7, blend: 'add'}),
        createStrokeLayer(
          [
            {
              points: closeRing(getCellRing(layout, probe, 0)),
              role: 'warm',
              width: 1,
              alpha: 1
            }
          ],
          {animation: 'pulse', blend: 'add'}
        ),
        createPulsingFillLayer(bars),
        createStrokeLayer(
          [
            {
              points: [
                [112, 75],
                [158, 75]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.8
            },
            {
              points: [
                [observed, 30],
                [observed, 76]
              ],
              role: 'warm',
              width: 1,
              alpha: 1
            }
          ],
          {animation: 'static'}
        )
      ];
    }
  },

  'losh-residual-permutation-test': {
    parameters: [
      createSlider('shuffles', 'Shuffled rows', 1, 3, 1, 2),
      createSlider('noise', 'Noisy area', 0, 1.5, 0.05, 1)
    ],
    build: ({fork, p}) => {
      const layout: Layout = {columns: 10, rows: 8, size: 8, x: 6, y: 18};
      const noise = fork('noise');
      const base = createLatticeValues(fork('values'), layout, 0.9);
      const center: Point = [noise.range(5, 8), noise.range(2.5, 5)];
      const values = base.map((value, index) => {
        const near = Math.hypot(
          (index % layout.columns) - center[0],
          Math.floor(index / layout.columns) - center[1]
        );
        return value * 0.5 + noise.gaussian() * Math.max(0, 1 - near / 3.2) * p.noise * 1.6;
      });
      const scores = values.map((value, index) => {
        const members = [index, ...getQueenNeighbors(layout, index)];
        const mean = members.reduce((sum, member) => sum + values[member], 0) / members.length;
        return Math.sqrt(
          members.reduce((sum, member) => sum + (values[member] - mean) ** 2, 0) / members.length
        );
      });
      const ranks = getRanks(scores);
      const items: PolygonItem[] = scores.map((_, index) => ({
        ring: getCellRing(layout, index),
        role: ranks[index] > 0.88 ? 'hot' : 'ink',
        alpha: ranks[index] > 0.88 ? 0.75 : 0.05 + ranks[index] * 0.08,
        phase: ranks[index]
      }));
      // Residuals of twelve neighbors: the observed arrangement, then shuffled copies of the same bars.
      const residuals = Array.from(
        {length: 12},
        (_, index) => 4 + Math.abs(values[index * 3 + 1]) * 7
      );
      const shuffle = fork('shuffle');
      const rows = Math.round(p.shuffles);
      const bars: PolygonItem[] = [];
      const links: PathItem[] = [];
      let previous = residuals.map((height, index) => ({height, index}));
      const barX = (slot: number) => 98 + slot * 4.3;
      previous.forEach(({height}, slot) => {
        bars.push({
          ring: createSizedRectangle(barX(slot), 38 - height, 3, height),
          role: 'warm',
          alpha: 0.8,
          phase: slot / 12
        });
      });
      for (let row = 1; row <= rows; row++) {
        const baseline = 38 + row * 22;
        const order = previous
          .map(entry => ({entry, key: shuffle.next()}))
          .sort((first, second) => first.key - second.key)
          .map(({entry}) => entry);
        order.forEach(({height, index}, slot) => {
          bars.push({
            ring: createSizedRectangle(barX(slot), baseline - height, 3, height),
            role: 'violet',
            alpha: 0.6,
            phase: slot / 12
          });
          const source = previous.findIndex(entry => entry.index === index);
          links.push({
            points: [
              [barX(source) + 1.5, baseline - 22 + 1],
              [barX(slot) + 1.5, baseline - height - 1]
            ],
            role: 'violet',
            width: 0.3,
            alpha: 0.7,
            phase: slot / 12
          });
        });
        previous = order;
      }
      return [
        createPulsingFillLayer(items),
        createLatticeOutline(layout, 0.2),
        createStrokeLayer(links, {animation: 'flow', speed: 0.4, trail: 0.6, blend: 'add'}),
        createPulsingFillLayer(bars),
        createStrokeLayer(
          [38, 60, 82].slice(0, rows + 1).map(y => ({
            points: [[96, y] as Point, [150, y] as Point],
            role: 'dim' as Role,
            width: 0.4,
            alpha: 0.7
          })),
          {animation: 'static'}
        )
      ];
    }
  },

  'multivariate-local-geary-permutation-test': {
    pointer: 'sticky',
    build: ({fork, pointer}) => {
      const layout: Layout = {columns: 12, rows: 9, size: 8, x: 6, y: 14};
      const variables = createVariables(fork, layout, 3);
      const perVariable = variables.map(values => getLocalGeary(layout, values));
      const scores = perVariable[0].map(
        (_, index) => (perVariable[0][index] + perVariable[1][index] + perVariable[2][index]) / 3
      );
      const ranks = getRanks(scores);
      const items: PolygonItem[] = scores.map((_, index) => {
        const rank = ranks[index];
        // Classified like the esda output: similar clusters, contrasts, in between, not significant.
        const role: Role =
          rank < 0.14 ? 'cool' : rank > 0.88 ? 'hot' : rank < 0.3 ? 'violet' : 'ink';
        return {
          ring: getCellRing(layout, index),
          role,
          alpha: role === 'ink' ? 0.05 : role === 'violet' ? 0.3 : 0.7,
          phase: rank
        };
      });
      const probe = getCellAt(layout, pointer) ?? ranks.indexOf(Math.min(...ranks));
      const markers: SpriteItem[] = [];
      const bands: PolygonItem[] = [];
      const guides: PathItem[] = [];
      perVariable.forEach((scoresOfVariable, variable) => {
        const y = 28 + variable * 22;
        // Where this cell's statistic falls in the spread of all cells stands in for its place in the null.
        const offset = clamp((getRanks(scoresOfVariable)[probe] - 0.5) * 44, -20, 20);
        bands.push({
          ring: createSizedRectangle(118, y - 4, 24, 8),
          role: 'ink',
          alpha: 0.18
        });
        guides.push({
          points: [
            [130, y - 7],
            [130, y + 7]
          ],
          role: 'dim',
          width: 0.4,
          alpha: 0.7
        });
        markers.push({
          x: 130 + offset,
          y,
          radius: 2.4,
          role: Math.abs(offset) > 12 ? 'warm' : 'primary',
          alpha: 1,
          phase: variable / 3
        });
      });
      const ring = getCellRing(layout, probe, 0);
      return [
        createPulsingFillLayer(items),
        createLatticeOutline(layout, 0.2),
        createStrokeLayer([{points: closeRing(ring), role: 'warm', width: 1, alpha: 1}], {
          animation: 'pulse',
          blend: 'add'
        }),
        createFillLayer(bands, {animation: 'static'}),
        createStrokeLayer(guides, {animation: 'static'}),
        createSpriteLayer(markers, {pulse: 0.5})
      ];
    }
  }
};
// ------------------------------------------------------------- inference, rates, space-time
const RATES_AND_SPACE_TIME_PREVIEWS: Record<string, EntryPreview> = {
  'multiple-testing-adjustment': {
    parameters: [
      createSlider('alpha', 'Alpha', 0.01, 0.3, 0.01, 0.12),
      createChoice('method', 'Method', ['None', 'Bonferroni', 'Holm', 'Benjamini-Hochberg'], 3)
    ],
    build: ({fork, p}) => {
      const random = fork('values');
      const count = 40;
      const left = 12;
      const width = 136;
      const values = Array.from({length: count}, (_, index) =>
        index < 18 ? 0.0025 * (index + 1) * random.range(0.35, 1.2) : random.range(0.01, 1)
      ).sort((first, second) => first - second);
      const getX = (index: number) => left + ((index + 0.5) * width) / count;
      // Square-root scale so both tiny p-values and the threshold lines stay readable.
      const getY = (value: number) => 92 - Math.sqrt(value) * 78;
      const alpha = p.alpha;
      const lineAt = [
        () => alpha,
        () => alpha / count,
        (index: number) => alpha / (count - index),
        (index: number) => (alpha * (index + 1)) / count
      ];
      const method = Math.round(p.method);
      let rejected = 0;
      if (method === 0) rejected = values.filter(value => value <= alpha).length;
      if (method === 1) rejected = values.filter(value => value <= alpha / count).length;
      if (method === 2) {
        while (rejected < count && values[rejected] <= lineAt[2](rejected)) rejected++;
      }
      if (method === 3) {
        values.forEach((value, index) => {
          if (value <= lineAt[3](index)) rejected = index + 1;
        });
      }
      const lines: PathItem[] = lineAt.map((line, which) => ({
        points: values.map((_, index) => [getX(index), getY(line(index))] as Point),
        role: which === method ? ('warm' as Role) : ('dim' as Role),
        width: which === method ? 1.2 : 0.4,
        alpha: which === method ? 1 : 0.45
      }));
      // Draw the active threshold last so it sits on top of the faint ones.
      lines.push(lines.splice(method, 1)[0]);
      const dots: SpriteItem[] = values.map((value, index) => ({
        x: getX(index),
        y: getY(value),
        radius: index < rejected ? 2 : 1.4,
        role: index < rejected ? 'hot' : 'primary',
        alpha: index < rejected ? 1 : 0.55,
        phase: index / count
      }));
      return [
        createStrokeLayer([getAxes(10, 8, 150, 92)], {animation: 'static'}),
        createStrokeLayer(lines, {animation: 'static', blend: 'add'}),
        createSpriteLayer(dots, {pulse: 0.5})
      ];
    }
  },

  'assuncao-reis-standardization': {
    parameters: [createSlider('areas', 'Areas', 20, 80, 2, 46)],
    build: ({fork, p}) => {
      const random = fork('areas');
      const effects = fork('effects');
      const rawItems: SpriteItem[] = [];
      const standardized: SpriteItem[] = [];
      const upper: Point[] = [];
      const lower: Point[] = [];
      const logMinimum = Math.log(15);
      const logMaximum = Math.log(600);
      for (let step = 0; step <= 16; step++) {
        const population = Math.exp(logMinimum + (step / 16) * (logMaximum - logMinimum));
        const x = 8 + (step / 16) * 62;
        const half = Math.min(42, (2 * 90) / Math.sqrt(population));
        upper.push([x, 50 - half]);
        lower.push([x, 50 + half]);
      }
      for (let index = 0; index < p.areas; index++) {
        const population = Math.exp(random.range(logMinimum, logMaximum));
        const x = 8 + ((Math.log(population) - logMinimum) / (logMaximum - logMinimum)) * 62;
        let score = random.gaussian();
        if (effects.next() < 0.12) score += effects.next() < 0.6 ? 2.6 : -2.6;
        const hot = score > 2;
        const cold = score < -2;
        const role: Role = hot ? 'hot' : cold ? 'cool' : 'primary';
        const alpha = hot || cold ? 1 : 0.6;
        // Raw rates fan out where populations are small; standardized scores share one spread.
        rawItems.push({
          x,
          y: clamp(50 - (score * 90) / Math.sqrt(population), 6, 94),
          radius: 1.4,
          role,
          alpha,
          phase: random.next()
        });
        standardized.push({
          x: x + 84,
          y: clamp(50 - score * 14, 6, 94),
          radius: 1.4,
          role,
          alpha,
          phase: random.next()
        });
      }
      return [
        createStrokeLayer(
          [
            {
              points: [
                [8, 50],
                [70, 50]
              ],
              role: 'dim',
              width: 0.4,
              alpha: 0.6
            },
            {
              points: [
                [92, 50],
                [154, 50]
              ],
              role: 'dim',
              width: 0.4,
              alpha: 0.6
            },
            {points: upper, role: 'dim', width: 0.5, alpha: 0.7},
            {points: lower, role: 'dim', width: 0.5, alpha: 0.7},
            {
              points: [
                [92, 22],
                [154, 22]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.7
            },
            {
              points: [
                [92, 78],
                [154, 78]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.7
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(
          [
            {
              points: [
                [74, 50],
                [88, 50]
              ],
              role: 'primary',
              width: 1.2,
              alpha: 1
            }
          ],
          {animation: 'flow', speed: 0.5, blend: 'add'}
        ),
        createSpriteLayer([...rawItems, ...standardized], {pulse: 0.35})
      ];
    }
  },

  'empirical-bayes-rates': {
    parameters: [createSlider('spread', 'Between-area spread', 0.3, 3, 0.1, 1)],
    build: ({fork, p}) => {
      const random = fork('areas');
      const rows = Array.from({length: 14}, () => ({
        population: random.range(25, 600),
        score: random.gaussian()
      })).sort((first, second) => second.population - first.population);
      const prior = 10 * p.spread;
      const rawItems: SpriteItem[] = [];
      const shrunk: SpriteItem[] = [];
      const arrows: PathItem[] = [];
      rows.forEach(({population, score}, index) => {
        const y = 10 + index * 6.2;
        const noise = 38 / Math.sqrt(population / 25);
        const weight = (prior * prior) / (prior * prior + noise * noise);
        const rawX = clamp(80 + score * noise, 10, 150);
        const shrunkX = 80 + (rawX - 80) * weight;
        const radius = 0.8 + (population / 600) * 1.1;
        rawItems.push({
          x: rawX,
          y,
          radius,
          role: rawX > 80 ? 'hot' : 'cool',
          alpha: 0.6,
          phase: index / 14
        });
        shrunk.push({
          x: shrunkX,
          y,
          radius,
          role: 'primary',
          alpha: 1,
          phase: index / 14
        });
        arrows.push({
          points: [
            [rawX, y],
            [shrunkX, y]
          ],
          role: 'warm',
          width: 0.7,
          alpha: 0.9,
          phase: index / 14
        });
      });
      return [
        createStrokeLayer(
          [
            {
              points: [
                [80, 4],
                [80, 96]
              ],
              role: 'cool',
              width: 0.8,
              alpha: 0.75
            }
          ],
          {animation: 'pulse', speed: 0.3, blend: 'add'}
        ),
        createStrokeLayer(arrows, {animation: 'flow', speed: 0.45, trail: 0.8, blend: 'add'}),
        createSpriteLayer(rawItems, {pulse: 0.2}),
        createSpriteLayer(shrunk, {pulse: 0.3})
      ];
    }
  },

  'spatial-empirical-bayes-rates': {
    pointer: 'hover',
    parameters: [createSlider('smoothing', 'Smoothing', 0, 1, 0.05, 0.8)],
    build: ({fork, p, pointer}) => {
      const left: Layout = {columns: 8, rows: 7, size: 8, x: 10, y: 22};
      const right: Layout = {...left, x: 86};
      const trend = createLatticeValues(fork('trend'), left, 1);
      const noise = fork('noise');
      const population = trend.map(() => noise.range(0.1, 1));
      const raw = trend.map(
        (value, index) => value + noise.gaussian() * 3.2 * (1.1 - population[index])
      );
      // Local mean of the neighbors, applied twice so the reference is a genuinely local level.
      const average = (values: number[]) =>
        values.map((_, index) => {
          const members = [index, ...getQueenNeighbors(left, index)];
          return members.reduce((sum, member) => sum + values[member], 0) / members.length;
        });
      const local = average(average(raw));
      const smoothed = raw.map((value, index) => {
        const amount = p.smoothing * (1 - 0.1 * population[index]);
        return value * (1 - amount) + local[index] * amount;
      });
      const getRole = (value: number): Role =>
        value < -0.8 ? 'cool' : value < 0 ? 'primary' : value < 0.8 ? 'warm' : 'hot';
      const toItems = (layout: Layout, values: number[]) =>
        values.map(
          (value, index): PolygonItem => ({
            // Small populations draw smaller: their rates are the unreliable ones.
            ring: getCellRing(layout, index, 0.4 + (1 - population[index]) * 1.7),
            role: getRole(value),
            alpha: 0.6,
            phase: (index % layout.columns) / layout.columns
          })
        );
      const probe = getCellAt(left, pointer) ?? getCellAt(right, pointer) ?? 3 + 3 * left.columns;
      const probeColumn = probe % left.columns;
      const probeRow = Math.floor(probe / left.columns);
      const windows = [left, right].map(layout => {
        const column0 = Math.max(0, probeColumn - 1);
        const firstRow = Math.max(0, probeRow - 1);
        const column1 = Math.min(layout.columns - 1, probeColumn + 1);
        const lastRow = Math.min(layout.rows - 1, probeRow + 1);
        const ring = createSizedRectangle(
          layout.x + column0 * layout.size,
          layout.y + firstRow * layout.size,
          (column1 - column0 + 1) * layout.size,
          (lastRow - firstRow + 1) * layout.size
        );
        return {
          points: closeRing(ring),
          role: 'warm' as Role,
          width: 1,
          alpha: 1
        };
      });
      return [
        createPulsingFillLayer([...toItems(left, raw), ...toItems(right, smoothed)]),
        createOutlineLayer(
          [
            createSizedRectangle(left.x, left.y, 64, 56),
            createSizedRectangle(right.x, right.y, 64, 56)
          ],
          'dim',
          0.5
        ),
        createStrokeLayer(
          [
            {
              points: [
                [76, 50],
                [84, 50]
              ],
              role: 'primary',
              width: 1.2,
              alpha: 1
            }
          ],
          {animation: 'flow', speed: 0.5, blend: 'add'}
        ),
        createStrokeLayer(windows, {animation: 'pulse', speed: 0.5, blend: 'add'})
      ];
    }
  },

  'emerging-hot-spots': {
    parameters: [createSlider('radius', 'Hot spot size', 1.2, 3.2, 0.1, 2)],
    build: ({fork, p}) => {
      const columns = 8;
      const rows = 6;
      const random = fork('centers');
      const seeds: Point[] = [
        [1.5, 1.4],
        [5.6, 1.2],
        [3.4, 3.5],
        [6.3, 4.3],
        [1.2, 4.4]
      ].map(([x, y]) => [x + random.range(-0.4, 0.4), y + random.range(-0.3, 0.3)]);
      const categories = Array.from({length: columns * rows}, (_, index) => {
        const column = (index % columns) + 0.5;
        const row = Math.floor(index / columns) + 0.5;
        let best = -1;
        let bestDistance = p.radius * 0.7;
        seeds.forEach(([x, y], category) => {
          const distance = Math.hypot(column - x, row - y);
          if (distance < bestDistance) {
            bestDistance = distance;
            best = category;
          }
        });
        return best;
      });
      // Gi* intensity of each category through four time slices.
      const intensity = (category: number, slice: number): number =>
        [
          slice === 3 ? 1 : 0, // new
          0.75, // persistent
          0.3 + slice * 0.24, // intensifying
          1 - slice * 0.28, // diminishing
          slice % 2 ? 0.7 : 0 // sporadic
        ][category] ?? 0;
      const roles: Role[] = ['violet', 'warm', 'hot', 'cool', 'primary'];
      const cell = 3.4;
      const slices: PolygonItem[] = [];
      const frames: Point[][] = [];
      for (let slice = 3; slice >= 0; slice--) {
        const x = 6 + slice * 14;
        const y = 58 - slice * 11;
        frames.push(createSizedRectangle(x, y, columns * cell, rows * cell));
        slices.push({
          ring: createSizedRectangle(x, y, columns * cell, rows * cell),
          role: 'ink',
          alpha: 0.06
        });
        categories.forEach((category, index) => {
          const level = category < 0 ? 0 : intensity(category, slice);
          if (level < 0.15) return;
          slices.push({
            ring: createSizedRectangle(
              x + (index % columns) * cell + 0.2,
              y + Math.floor(index / columns) * cell + 0.2,
              cell - 0.4,
              cell - 0.4
            ),
            role: roles[category],
            alpha: 0.2 + level * 0.65,
            phase: slice / 4
          });
        });
      }
      // One intensifying cell followed through time: the series Mann-Kendall tests for a trend.
      const trackedCell = Math.max(0, categories.indexOf(2));
      const trendLine: Point[] = [0, 1, 2, 3].map(slice => [
        6 + slice * 14 + ((trackedCell % columns) + 0.5) * cell,
        58 - slice * 11 + (Math.floor(trackedCell / columns) + 0.5) * cell
      ]);
      const result: Layout = {columns, rows, size: 9, x: 84, y: 24};
      const items: PolygonItem[] = categories.map((category, index) => ({
        ring: getCellRing(result, index),
        role: category < 0 ? 'ink' : roles[category],
        alpha: category < 0 ? 0.05 : 0.75,
        phase: (index % columns) / columns
      }));
      return [
        createFillLayer(slices, {animation: 'pulse', blend: 'add'}),
        createOutlineLayer(frames, 'dim', 0.55),
        createStrokeLayer(
          [
            {points: trendLine, role: 'violet', width: 0.9, alpha: 1},
            {
              points: createArc([77, 34], [83, 46], 0.2, 10),
              role: 'primary',
              width: 1.1,
              alpha: 1
            }
          ],
          {animation: 'flow', speed: 0.35, blend: 'add'}
        ),
        createPulsingFillLayer(items),
        createLatticeOutline(result, 0.25)
      ];
    }
  },

  'knox-test': {
    parameters: [
      createSlider('distance', 'Near in space', 6, 30, 1, 11, 'px'),
      createSlider('window', 'Near in time', 0.04, 0.5, 0.02, 0.16)
    ],
    build: ({fork, p}) => {
      const random = fork('cases');
      const outbreaks: Array<{x: number; y: number; time: number}> = [
        {x: 30, y: 30, time: 0.2},
        {x: 74, y: 66, time: 0.5},
        {x: 46, y: 78, time: 0.82}
      ];
      const cases: Array<{x: number; y: number; time: number}> = [];
      outbreaks.forEach(outbreak => {
        for (let index = 0; index < 10; index++) {
          cases.push({
            x: clamp(outbreak.x + random.gaussian() * 7, 8, 100),
            y: clamp(outbreak.y + random.gaussian() * 7, 8, 92),
            time: clamp(outbreak.time + random.gaussian() * 0.05, 0, 1)
          });
        }
      });
      for (let index = 0; index < 12; index++) {
        cases.push({
          x: random.range(8, 100),
          y: random.range(8, 92),
          time: random.next()
        });
      }
      const links: PathItem[] = [];
      const counts = [0, 0, 0, 0]; // near both, near space only, near time only, neither
      for (let first = 0; first < cases.length; first++) {
        for (let second = first + 1; second < cases.length; second++) {
          const nearSpace =
            getDistance([cases[first].x, cases[first].y], [cases[second].x, cases[second].y]) <=
            p.distance;
          const nearTime = Math.abs(cases[first].time - cases[second].time) <= p.window;
          counts[(nearSpace ? 0 : 2) + (nearTime ? 0 : 1)]++;
          if (nearSpace && nearTime) {
            links.push({
              points: [
                [cases[first].x, cases[first].y],
                [cases[second].x, cases[second].y]
              ],
              role: 'warm',
              width: 0.6,
              alpha: 0.95,
              phase: first / cases.length
            });
          }
        }
      }
      const peak = Math.log1p(Math.max(...counts));
      const table: PolygonItem[] = counts.map((count, index) => ({
        ring: createSizedRectangle(116 + (index % 2) * 17, 34 + Math.floor(index / 2) * 17, 15, 15),
        role: index === 0 ? 'warm' : 'primary',
        alpha: 0.12 + (Math.log1p(count) / peak) * (index === 0 ? 0.85 : 0.5),
        phase: index / 4
      }));
      return [
        createStrokeLayer(
          [
            {
              points: [...createSizedRectangle(6, 6, 98, 88), [6, 6]],
              role: 'dim',
              width: 0.4,
              alpha: 0.5
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(links, {animation: 'flow', speed: 0.5, trail: 0.7, blend: 'add'}),
        createSpriteLayer(
          cases.map(
            (entry, index): SpriteItem => ({
              x: entry.x,
              y: entry.y,
              radius: 1.6,
              role: getClassRole(Math.min(4, Math.floor(entry.time * 5)), 5),
              alpha: 0.95,
              phase: index / cases.length
            })
          ),
          {pulse: 0.3}
        ),
        createPulsingFillLayer(table),
        createOutlineLayer([createSizedRectangle(116, 34, 32, 32)], 'dim', 0.6)
      ];
    }
  },

  'mantel-test': {
    parameters: [createSlider('correlation', 'Space-time link', 0, 1, 0.05, 0.8)],
    build: ({fork, p}) => {
      const size = 10;
      const random = fork('places');
      const positions = Array.from({length: size}, () => random.next()).sort(
        (first, second) => first - second
      );
      const times = positions.map(
        position => p.correlation * position + (1 - p.correlation) * random.next()
      );
      const shuffled = [...times];
      const permute = fork('permute');
      for (let index = size - 1; index > 0; index--) {
        const other = permute.integer(index + 1);
        [shuffled[index], shuffled[other]] = [shuffled[other], shuffled[index]];
      }
      const cell = 3.7;
      const matrices: PolygonItem[] = [];
      const pairs: Array<[number, number]> = [];
      const nullPairs: Array<[number, number]> = [];
      for (let row = 0; row < size; row++) {
        for (let column = 0; column < size; column++) {
          const space = Math.abs(positions[row] - positions[column]);
          const time = Math.abs(times[row] - times[column]);
          matrices.push({
            ring: createSizedRectangle(8 + column * cell, 8 + row * cell, cell - 0.3, cell - 0.3),
            role: 'primary',
            alpha: 0.08 + space * 0.8,
            phase: space
          });
          matrices.push({
            ring: createSizedRectangle(8 + column * cell, 54 + row * cell, cell - 0.3, cell - 0.3),
            role: 'violet',
            alpha: 0.08 + time * 0.8,
            phase: time
          });
          if (column > row) {
            pairs.push([space, time]);
            nullPairs.push([space, Math.abs(shuffled[row] - shuffled[column])]);
          }
        }
      }
      const toPoint = ([space, time]: [number, number]): Point => [68 + space * 78, 88 - time * 76];
      const meanSpace = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
      const meanTime = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
      let covariance = 0;
      let variance = 0;
      for (const [space, time] of pairs) {
        covariance += (space - meanSpace) * (time - meanTime);
        variance += (space - meanSpace) ** 2;
      }
      const slope = covariance / (variance || 1);
      const fit = (space: number): Point =>
        toPoint([space, meanTime + slope * (space - meanSpace)]);
      return [
        createPulsingFillLayer(matrices),
        createOutlineLayer(
          [createSizedRectangle(8, 8, 37, 37), createSizedRectangle(8, 54, 37, 37)],
          'dim',
          0.55
        ),
        createStrokeLayer(
          [
            getAxes(64, 8, 150, 90),
            {
              points: [
                [46, 26],
                [60, 44]
              ],
              role: 'primary',
              width: 0.8,
              alpha: 0.9
            },
            {
              points: [
                [46, 72],
                [60, 58]
              ],
              role: 'violet',
              width: 0.8,
              alpha: 0.9
            }
          ],
          {animation: 'flow', speed: 0.4, blend: 'add'}
        ),
        createSpriteLayer(toSprites(nullPairs.map(toPoint), 'ink', 1, undefined, 0.3)),
        createSpriteLayer(toSprites(pairs.map(toPoint), 'warm', 1.5, undefined, 0.9), {pulse: 0.3}),
        createStrokeLayer([{points: [fit(0), fit(1)], role: 'hot', width: 1.1, alpha: 1}], {
          animation: 'pulse',
          speed: 0.4,
          blend: 'add'
        })
      ];
    }
  },

  'spatial-scan-statistic': {
    pointer: 'hover',
    parameters: [createSlider('radius', 'Window radius', 8, 34, 1, 17, 'px')],
    build: ({fork, p, pointer}) => {
      const random = fork('points');
      const hot: Point = [random.range(64, 108), random.range(34, 66)];
      const points = [
        ...createScatterPoints(random, 80, 6),
        ...createClusterPoints(random, [hot], 34, 7)
      ];
      const center = pointer ?? hot;
      const inside = points.map(point => getDistance(point, center) <= p.radius);
      const count = inside.filter(Boolean).length;
      // Observed over expected under a uniform risk surface: the likelihood-ratio idea of the scan.
      const expected = (points.length * Math.PI * p.radius * p.radius) / (160 * 100);
      const significant = count / expected > 1.8 && count > 8;
      const windowRole: Role = significant ? 'hot' : 'warm';
      const candidates: PathItem[] = Array.from({length: 7}, (_, index) => {
        const candidateRandom = fork(`candidate-${index}`);
        return {
          points: createCircle(
            candidateRandom.range(14, 146),
            candidateRandom.range(14, 86),
            p.radius * candidateRandom.range(0.6, 1.4),
            32
          ),
          role: 'dim' as Role,
          width: 0.4,
          alpha: 0.7,
          phase: index / 7
        };
      }).map(item => ({...item, points: closeRing(item.points)}));
      const ring = createCircle(center[0], center[1], p.radius, 48);
      return [
        createStrokeLayer(candidates, {animation: 'pulse', speed: 0.35, blend: 'add'}),
        createFillLayer([{ring, role: windowRole, alpha: 0.16}], {
          animation: 'pulse',
          blend: 'add'
        }),
        createSpriteLayer(
          points.map(
            ([x, y], index): SpriteItem => ({
              x,
              y,
              radius: inside[index] ? 1.9 : 1.2,
              role: inside[index] ? windowRole : 'primary',
              alpha: inside[index] ? 1 : 0.5,
              phase: index / points.length
            })
          ),
          {pulse: 0.3}
        ),
        createStrokeLayer(
          [
            {
              points: closeRing(ring),
              role: windowRole,
              width: 1.2,
              alpha: 1
            }
          ],
          {animation: 'pulse', speed: 0.5, blend: 'add'}
        )
      ];
    }
  }
};

// ------------------------------------------------------------------ clustering and outliers
type Cluster = {center: Point; spread: number; count: number};

function createBlobPoints(random: Random, clusters: Cluster[]): Point[] {
  return clusters.flatMap(({center, spread, count}) =>
    createClusterPoints(random, [center], count, spread)
  );
}

function getNearestNeighbors(points: Point[], index: number, count: number): number[] {
  return points
    .map((point, other) => ({
      other,
      distance: getDistance(point, points[index])
    }))
    .filter(entry => entry.other !== index)
    .sort((first, second) => first.distance - second.distance)
    .slice(0, count)
    .map(entry => entry.other);
}

function runDbscan(
  points: Point[],
  epsilon: number,
  minimumSamples: number
): {labels: number[]; core: boolean[]; neighbors: number[][]} {
  const neighbors = points.map((point, index) => {
    const found: number[] = [];
    points.forEach((other, otherIndex) => {
      if (otherIndex !== index && getDistance(point, other) <= epsilon) found.push(otherIndex);
    });
    return found;
  });
  const core = neighbors.map(list => list.length + 1 >= minimumSamples);
  const labels = points.map(() => -1);
  let cluster = 0;
  points.forEach((_, start) => {
    if (labels[start] !== -1 || !core[start]) return;
    labels[start] = cluster;
    const queue = [start];
    for (let head = 0; head < queue.length; head++) {
      for (const neighbor of neighbors[queue[head]]) {
        if (labels[neighbor] !== -1) continue;
        labels[neighbor] = cluster;
        if (core[neighbor]) queue.push(neighbor);
      }
    }
    cluster++;
  });
  return {labels, core, neighbors};
}

const CLUSTERING_PREVIEWS: Record<string, EntryPreview> = {
  'dbscan-clustering': {
    pointer: 'hover',
    parameters: [
      createSlider('epsilon', 'Radius (eps)', 4, 14, 0.5, 7.5, 'px'),
      createSlider('minimumSamples', 'Min points', 3, 10, 1, 5)
    ],
    build: ({fork, p, pointer}) => {
      const random = fork('points');
      const points = [
        ...createBlobPoints(random, [
          {center: [36, 34], spread: 6, count: 34},
          {center: [96, 62], spread: 8, count: 40},
          {center: [124, 26], spread: 5, count: 24}
        ]),
        ...createScatterPoints(random, 22, 6)
      ];
      const {labels, core, neighbors} = runDbscan(points, p.epsilon, p.minimumSamples);
      const links = new Map<Role, PathItem[]>();
      const items: SpriteItem[] = points.map(([x, y], index) => {
        const role: Role =
          labels[index] < 0 ? 'dim' : ROLE_CYCLE[labels[index] % ROLE_CYCLE.length];
        if (core[index]) {
          for (const neighbor of neighbors[index]) {
            if (neighbor > index && core[neighbor]) {
              const list = links.get(role) ?? [];
              list.push({
                points: [[x, y], points[neighbor]],
                role,
                width: 0.3,
                alpha: 0.4
              });
              links.set(role, list);
            }
          }
        }
        return {
          x,
          y,
          radius: core[index] ? 1.9 : labels[index] < 0 ? 1 : 1.3,
          role,
          alpha: core[index] ? 1 : labels[index] < 0 ? 0.5 : 0.7,
          phase: random.next()
        };
      });
      // Epsilon rings: one around the pointer, otherwise around a core point of the first cluster.
      const anchor =
        pointer ??
        points[
          Math.max(
            0,
            labels.findIndex((label, index) => label === 0 && core[index])
          )
        ];
      const ring = createCircle(anchor[0], anchor[1], p.epsilon, 40);
      return [
        ...[...links.values()].map(
          (list): StrokeLayer => createStrokeLayer(list, {animation: 'static', blend: 'add'})
        ),
        createSpriteLayer(items, {pulse: 0.3}),
        createStrokeLayer([{points: closeRing(ring), role: 'warm', width: 0.9, alpha: 1}], {
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        })
      ];
    }
  },

  'hdbscan-clustering': {
    parameters: [createSlider('minimumSize', 'Min cluster size', 3, 30, 1, 10)],
    build: ({fork, p}) => {
      const groups: Cluster[] = [
        {center: [34, 32], spread: 3.5, count: 22},
        {center: [86, 30], spread: 10, count: 34},
        {center: [122, 66], spread: 4.5, count: 16},
        {center: [56, 72], spread: 12, count: 30},
        {center: [136, 22], spread: 4, count: 6}
      ];
      const random = fork('points');
      const halos: PolygonItem[] = [];
      const rings: PathItem[] = [];
      const items: SpriteItem[] = [];
      groups.forEach((group, index) => {
        const kept = group.count >= p.minimumSize;
        const role: Role = kept ? ROLE_CYCLE[index % ROLE_CYCLE.length] : 'dim';
        if (kept) {
          // Soft halo and stability ring: tight and loose clusters are both found, each at its own scale.
          halos.push({
            ring: createCircle(group.center[0], group.center[1], group.spread * 2.3, 32),
            role,
            alpha: 0.1,
            phase: index / 5
          });
          const ring = createCircle(group.center[0], group.center[1], group.spread * 1.2, 32);
          rings.push({
            points: closeRing(ring),
            role,
            width: 0.5,
            alpha: 0.8,
            phase: index / 5
          });
        }
        for (const point of createClusterPoints(
          random,
          [group.center],
          group.count,
          group.spread
        )) {
          const membership = Math.exp(
            -(getDistance(point, group.center) ** 2) / (2 * (group.spread * 1.6) ** 2)
          );
          items.push({
            x: point[0],
            y: point[1],
            radius: kept ? 1.2 + membership * 0.9 : 1,
            role,
            alpha: kept ? 0.35 + membership * 0.65 : 0.5,
            phase: random.next()
          });
        }
      });
      for (const [x, y] of createScatterPoints(random, 20, 6)) {
        items.push({
          x,
          y,
          radius: 1,
          role: 'dim',
          alpha: 0.5,
          phase: random.next()
        });
      }
      return [
        createFillLayer(halos, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer(rings, {animation: 'pulse', speed: 0.35, blend: 'add'}),
        createSpriteLayer(items, {pulse: 0.3})
      ];
    }
  },

  'optics-clustering': {
    parameters: [createSlider('threshold', 'Reachability cut', 6, 32, 1, 18)],
    build: ({fork, p}) => {
      const random = fork('points');
      const groups: Cluster[] = [
        {center: [30, 28], spread: 5, count: 26},
        {center: [84, 26], spread: 8, count: 30},
        {center: [128, 30], spread: 4.5, count: 22}
      ];
      const baseline = 94;
      type Entry = {point: Point; height: number; cluster: number};
      const order: Entry[] = [];
      const noise = (count: number) => {
        for (let index = 0; index < count; index++) {
          order.push({
            point: [random.range(8, 152), random.range(8, 48)],
            height: random.range(28, 34),
            cluster: -1
          });
        }
      };
      noise(3);
      groups.forEach((group, cluster) => {
        createClusterPoints(random, [group.center], group.count, group.spread).forEach(
          (point, index) => {
            order.push({
              point: [point[0], clamp(point[1], 8, 48)],
              height:
                index === 0
                  ? random.range(29, 34)
                  : 3 + group.spread * 1.3 * random.range(0.4, 1.5),
              cluster
            });
          }
        );
        noise(2);
      });
      const width = 144 / order.length;
      const bars: PolygonItem[] = [];
      const sprites: SpriteItem[] = [];
      const profile: Point[] = [];
      order.forEach((entry, index) => {
        const inside = entry.height < p.threshold && entry.cluster >= 0;
        const role: Role = inside ? ROLE_CYCLE[entry.cluster % ROLE_CYCLE.length] : 'dim';
        bars.push({
          ring: createSizedRectangle(
            8 + index * width,
            baseline - entry.height,
            width * 0.8,
            entry.height
          ),
          role,
          alpha: inside ? 0.85 : 0.35,
          phase: index / order.length
        });
        sprites.push({
          x: entry.point[0],
          y: entry.point[1],
          radius: inside ? 1.6 : 1,
          role,
          alpha: inside ? 1 : 0.5,
          phase: index / order.length
        });
        profile.push([8 + (index + 0.4) * width, baseline - entry.height]);
      });
      return [
        createSpriteLayer(sprites, {pulse: 0.3}),
        createFillLayer(bars, {animation: 'pulse', blend: 'add'}),
        createStrokeLayer([{points: profile, role: 'primary', width: 0.6, alpha: 0.9}], {
          animation: 'flow',
          speed: 0.25,
          trail: 1.5,
          blend: 'add'
        }),
        createStrokeLayer(
          [
            {
              points: [
                [6, baseline],
                [154, baseline]
              ],
              role: 'dim',
              width: 0.5,
              alpha: 0.8
            },
            {
              points: [
                [6, baseline - p.threshold],
                [154, baseline - p.threshold]
              ],
              role: 'warm',
              width: 0.8,
              alpha: 0.9
            }
          ],
          {animation: 'static'}
        )
      ];
    }
  },

  'k-means-clustering': {
    parameters: [
      createSlider('clusters', 'Centers (k)', 2, 6, 1, 4),
      createSlider('iterations', 'Iterations', 0, 8, 1, 5)
    ],
    build: ({fork, p}) => {
      const random = fork('points');
      const points = createBlobPoints(random, [
        {center: [38, 32], spread: 8, count: 30},
        {center: [110, 30], spread: 8, count: 30},
        {center: [48, 74], spread: 8, count: 30},
        {center: [116, 70], spread: 8, count: 30}
      ]);
      const init = fork('init');
      const count = Math.round(p.clusters);
      let centers: Point[] = Array.from(
        {length: count},
        () => [...points[init.integer(points.length)]] as Point
      );
      const history: Point[][] = centers.map(center => [center]);
      const assign = (current: Point[]) =>
        points.map(point => {
          let best = 0;
          current.forEach((center, index) => {
            if (getDistance(point, center) < getDistance(point, current[best])) best = index;
          });
          return best;
        });
      let labels = assign(centers);
      for (let iteration = 0; iteration < p.iterations; iteration++) {
        centers = centers.map((center, index) => {
          const members = points.filter((_, pointIndex) => labels[pointIndex] === index);
          if (members.length === 0) return center;
          return [
            members.reduce((sum, point) => sum + point[0], 0) / members.length,
            members.reduce((sum, point) => sum + point[1], 0) / members.length
          ] as Point;
        });
        centers.forEach((center, index) => history[index].push(center));
        labels = assign(centers);
      }
      const spokes: PathItem[] = points.map((point, index) => ({
        points: [point, centers[labels[index]]],
        role: ROLE_CYCLE[labels[index] % ROLE_CYCLE.length],
        width: 0.25,
        alpha: 0.35
      }));
      return [
        createStrokeLayer(spokes, {animation: 'static', blend: 'add'}),
        createSpriteLayer(
          points.map(
            ([x, y], index): SpriteItem => ({
              x,
              y,
              radius: 1.4,
              role: ROLE_CYCLE[labels[index] % ROLE_CYCLE.length],
              alpha: 0.9,
              phase: random.next()
            })
          ),
          {pulse: 0.25}
        ),
        createStrokeLayer(
          history
            .filter(path => path.length > 1)
            .map((path, index) => ({
              points: path,
              role: 'ink' as Role,
              width: 0.9,
              alpha: 0.9,
              phase: index / count
            })),
          {animation: 'flow', speed: 0.4, blend: 'add'}
        ),
        createSpriteLayer(
          centers.map(
            ([x, y], index): SpriteItem => ({
              x,
              y,
              radius: 2.8,
              role: 'ink',
              alpha: 0.9,
              phase: index / count
            })
          ),
          {pulse: 0.5}
        ),
        createSpriteLayer(
          centers.map(
            ([x, y], index): SpriteItem => ({
              x,
              y,
              radius: 2,
              role: ROLE_CYCLE[index % ROLE_CYCLE.length],
              alpha: 1
            })
          )
        )
      ];
    }
  },

  'local-outlier-factor': {
    pointer: 'hover',
    parameters: [createSlider('neighbors', 'Neighbors (k)', 3, 15, 1, 6)],
    build: ({fork, p, pointer}) => {
      const random = fork('points');
      const points: Point[] = [
        ...createBlobPoints(random, [
          {center: [48, 40], spread: 4.5, count: 34},
          {center: [110, 60], spread: 10, count: 36}
        ]),
        [20, 86],
        [82, 12],
        [142, 14],
        [84, 54],
        [28, 10],
        [146, 92]
      ];
      const k = Math.round(p.neighbors);
      const nearest = points.map((_, index) => getNearestNeighbors(points, index, k));
      const kDistance = points.map((point, index) =>
        getDistance(point, points[nearest[index][k - 1]])
      );
      const density = points.map((point, index) => {
        const reach = nearest[index].reduce(
          (sum, other) => sum + Math.max(kDistance[other], getDistance(point, points[other])),
          0
        );
        return k / (reach || 1);
      });
      const factor = points.map(
        (_, index) =>
          nearest[index].reduce((sum, other) => sum + density[other], 0) / k / density[index]
      );
      let probe = getIndexOfMaximum(factor);
      if (pointer) {
        let best = Infinity;
        points.forEach((point, index) => {
          const distance = getDistance(point, pointer);
          if (distance < best) {
            best = distance;
            probe = index;
          }
        });
      }
      const roleOf = (value: number): Role =>
        value > 1.7 ? 'hot' : value > 1.3 ? 'warm' : 'primary';
      const outliers = points.map((_, index) => index).filter(index => factor[index] > 1.3);
      const ring = createCircle(points[probe][0], points[probe][1], kDistance[probe], 40);
      return [
        createSpriteLayer(
          outliers.map(
            (index): SpriteItem => ({
              x: points[index][0],
              y: points[index][1],
              radius: 3 + Math.min(5, (factor[index] - 1) * 3),
              role: roleOf(factor[index]),
              alpha: 0.3,
              phase: index / points.length
            })
          ),
          {pulse: 0.6}
        ),
        createStrokeLayer(
          nearest[probe].map((other, index) => ({
            points: [points[probe], points[other]],
            role: 'violet' as Role,
            width: 0.6,
            alpha: 0.9,
            phase: index / k
          })),
          {animation: 'flow', speed: 0.5, blend: 'add'}
        ),
        createStrokeLayer(
          [
            {
              points: closeRing(ring),
              role: 'warm',
              width: 0.5,
              alpha: 0.4
            }
          ],
          {animation: 'pulse', speed: 0.4, blend: 'add'}
        ),
        createSpriteLayer(
          points.map(
            ([x, y], index): SpriteItem => ({
              x,
              y,
              radius: 1.2 + Math.min(1.8, Math.max(0, factor[index] - 1) * 1.2),
              role: roleOf(factor[index]),
              alpha: factor[index] > 1.3 ? 1 : 0.6,
              phase: index / points.length
            })
          ),
          {pulse: 0.2}
        )
      ];
    }
  },

  'similar-locations': {
    pointer: 'sticky',
    parameters: [createSlider('matches', 'Matches', 1, 10, 1, 5)],
    build: ({fork, p, pointer}) => {
      const columns = 32;
      const rows = 20;
      const attributes = [0, 1, 2].map(index =>
        createSmoothField(fork(`attribute-${index}`), 5, 1.2, columns, rows)
      );
      const referenceDefault = fork('reference');
      const reference =
        pointer !== null
          ? clamp(Math.floor(pointer[1] / 5), 0, rows - 1) * columns +
            clamp(Math.floor(pointer[0] / 5), 0, columns - 1)
          : (3 + referenceDefault.integer(5)) * columns + 8 + referenceDefault.integer(14);
      const distances = new Float32Array(columns * rows);
      let maximum = 0;
      for (let index = 0; index < distances.length; index++) {
        let sum = 0;
        for (const attribute of attributes) sum += (attribute[index] - attribute[reference]) ** 2;
        distances[index] = Math.sqrt(sum);
        maximum = Math.max(maximum, distances[index]);
      }
      const similarity = distances.map(distance => (1 - distance / (maximum || 1)) ** 4);
      const center = (index: number): Point => [
        ((index % columns) + 0.5) * 5,
        (Math.floor(index / columns) + 0.5) * 5
      ];
      const ranked = Array.from(similarity.keys()).sort(
        (first, second) => similarity[second] - similarity[first]
      );
      const matches: number[] = [];
      for (const index of ranked) {
        if (matches.length >= p.matches) break;
        if (getDistance(center(index), center(reference)) < 14) continue;
        if (matches.every(other => getDistance(center(index), center(other)) > 14))
          matches.push(index);
      }
      return [
        {
          type: 'field',
          columns,
          rows,
          values: similarity,
          ramp: 'ice',
          style: 'ramp',
          filter: 'nearest',
          strength: 0.05
        },
        createStrokeLayer(
          matches.map((index, order) => ({
            points: createArc(center(reference), center(index), 0.18, 20),
            role: 'warm' as Role,
            width: 0.7,
            alpha: 0.9,
            phase: order / matches.length
          })),
          {animation: 'flow', speed: 0.35, blend: 'add'}
        ),
        createSpriteLayer(
          matches.map(
            (index): SpriteItem => ({
              x: center(index)[0],
              y: center(index)[1],
              radius: 3,
              role: 'hot',
              alpha: 0.9
            })
          ),
          {pulse: 0.4}
        ),
        createSpriteLayer(
          [
            {
              x: center(reference)[0],
              y: center(reference)[1],
              radius: 4.4,
              role: 'warm',
              alpha: 1
            }
          ],
          {pulse: 0.6}
        )
      ];
    }
  },

  'segregation-indices': {
    parameters: [createSlider('segregation', 'Segregation', 0, 1, 0.05, 0.7)],
    build: ({fork, p}) => {
      const layout: Layout = {columns: 8, rows: 4, size: 17, x: 12, y: 6};
      const smooth = createSmoothField(fork('structure'), 4, 0.8, layout.columns, layout.rows);
      const random = fork('residents');
      const fills: PolygonItem[] = [];
      const dots: SpriteItem[] = [];
      let totalA = 0;
      let totalB = 0;
      const tracts: Array<{groupA: number; groupB: number}> = [];
      smooth.forEach((value, index) => {
        const share = clamp(
          0.5 + (value - 0.5) * p.segregation * 2.6 + random.gaussian() * 0.08,
          0.03,
          0.97
        );
        let groupACount = 0;
        const [centerX, centerY] = getCellCenter(layout, index);
        for (let dotIndex = 0; dotIndex < 9; dotIndex++) {
          const isGroupA = random.next() < share;
          if (isGroupA) groupACount++;
          dots.push({
            x: centerX + ((dotIndex % 3) - 1) * 4.6,
            y: centerY + (Math.floor(dotIndex / 3) - 1) * 4.6,
            radius: 1.3,
            role: isGroupA ? 'hot' : 'cool',
            alpha: 0.95,
            phase: random.next()
          });
        }
        tracts.push({groupA: groupACount, groupB: 9 - groupACount});
        totalA += groupACount;
        totalB += 9 - groupACount;
        fills.push({
          ring: getCellRing(layout, index, 0.6),
          role: share > 0.5 ? 'hot' : 'cool',
          alpha: 0.05 + Math.abs(share - 0.5) * 0.3,
          phase: (index % layout.columns) / layout.columns
        });
      });
      // Dissimilarity index: half the summed gap between each group's share of the tract.
      const dissimilarity = clamp(
        0.5 *
          tracts.reduce(
            (sum, tract) =>
              sum + Math.abs(tract.groupA / (totalA || 1) - tract.groupB / (totalB || 1)),
            0
          ),
        0.02,
        1
      );
      return [
        createPulsingFillLayer(fills),
        createLatticeOutline(layout, 0.4),
        createSpriteLayer(dots, {pulse: 0.2}),
        createStrokeLayer(
          [
            {
              points: [
                [12, 88],
                [148, 88]
              ],
              role: 'dim',
              width: 1,
              alpha: 0.5
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(
          [
            {
              points: [
                [12, 88],
                [12 + dissimilarity * 136, 88]
              ],
              role: 'warm',
              width: 2.4,
              alpha: 1
            }
          ],
          {animation: 'draw', speed: 0.25, blend: 'add'}
        )
      ];
    }
  }
};

// ------------------------------------------------------------------- models and validation
const MODEL_PREVIEWS: Record<string, EntryPreview> = {
  'spatial-interaction': {
    pointer: 'hover',
    parameters: [createSlider('decay', 'Distance decay', 0.3, 3.5, 0.1, 1.4)],
    build: ({fork, p, pointer}) => {
      const random = fork('places');
      const places: Array<{point: Point; mass: number}> = Array.from({length: 8}, () => ({
        point: [random.range(16, 144), random.range(16, 84)] as Point,
        mass: random.range(0.5, 2)
      }));
      if (pointer) places.push({point: pointer, mass: 1.6});
      const flows: Array<{from: number; to: number; amount: number}> = [];
      for (let from = 0; from < places.length; from++) {
        for (let to = from + 1; to < places.length; to++) {
          const distance = getDistance(places[from].point, places[to].point) / 30;
          flows.push({
            from,
            to,
            amount: (places[from].mass * places[to].mass) / distance ** p.decay
          });
        }
      }
      flows.sort((first, second) => second.amount - first.amount);
      const top = flows.slice(0, 14);
      const peak = top[0].amount;
      const arcs: PathItem[] = top.map(({from, to, amount}, index) => {
        const share = amount / peak;
        return {
          points: createArc(places[from].point, places[to].point, 0.2, 22),
          role: share > 0.55 ? 'hot' : share > 0.25 ? 'warm' : 'primary',
          width: 0.35 + share * 2.4,
          alpha: 0.4 + share * 0.6,
          phase: index / top.length
        };
      });
      return [
        createStrokeLayer(arcs, {animation: 'flow', speed: 0.3, blend: 'add'}),
        createSpriteLayer(
          places.map(
            ({point, mass}, index): SpriteItem => ({
              x: point[0],
              y: point[1],
              radius: 1.6 + mass * 1.6,
              role: pointer && index === places.length - 1 ? 'warm' : 'primary',
              alpha: 1,
              phase: index / places.length
            })
          ),
          {pulse: 0.35}
        )
      ];
    }
  },

  'spatial-cross-validation': {
    pointer: 'sticky',
    parameters: [
      createSlider('fold', 'Held-out block', 0, 7, 1, 5),
      createSlider('buffer', 'Buffer', 0, 2, 1, 1, 'cells')
    ],
    build: ({p, pointer}) => {
      const layout = MAP_LAYOUT;
      const blockColumns = 4;
      const blockRows = 5;
      const cell = getCellAt(layout, pointer);
      const fold =
        cell === null
          ? Math.round(p.fold)
          : Math.floor(Math.floor(cell / layout.columns) / blockRows) * 4 +
            Math.floor((cell % layout.columns) / blockColumns);
      const testColumn = (fold % 4) * blockColumns;
      const testRow = Math.floor(fold / 4) * blockRows;
      const buffer = Math.round(p.buffer);
      const items: PolygonItem[] = [];
      for (let index = 0; index < layout.columns * layout.rows; index++) {
        const column = index % layout.columns;
        const row = Math.floor(index / layout.columns);
        const dx = Math.max(testColumn - column, 0, column - (testColumn + blockColumns - 1));
        const dy = Math.max(testRow - row, 0, row - (testRow + blockRows - 1));
        const distance = Math.max(dx, dy);
        const role: Role = distance === 0 ? 'hot' : distance <= buffer ? 'ink' : 'primary';
        items.push({
          ring: getCellRing(layout, index, 0.6),
          role,
          alpha: distance === 0 ? 0.75 : distance <= buffer ? 0.09 : 0.3,
          phase: (index % layout.columns) / layout.columns
        });
      }
      const blocks: Point[][] = [];
      for (let block = 0; block < 8; block++) {
        blocks.push(
          createSizedRectangle(
            layout.x + (block % 4) * blockColumns * layout.size,
            layout.y + Math.floor(block / 4) * blockRows * layout.size,
            blockColumns * layout.size,
            blockRows * layout.size
          )
        );
      }
      const test = blocks[fold];
      return [
        createPulsingFillLayer(items),
        createOutlineLayer(blocks, 'dim', 0.8),
        createStrokeLayer([{points: closeRing(test), role: 'warm', width: 1.2, alpha: 1}], {
          animation: 'pulse',
          speed: 0.5,
          blend: 'add'
        })
      ];
    }
  },

  'spatial-cross-validated-ols': {
    parameters: [createSlider('noise', 'Prediction error', 0.02, 0.3, 0.01, 0.1)],
    build: ({fork, p}) => {
      const random = fork('points');
      const foldBias = Array.from({length: 5}, () => random.gaussian() * 0.04);
      const toPoint = (observed: number, predicted: number): Point => [
        44 + observed * 84,
        90 - predicted * 80
      ];
      const dots: SpriteItem[] = [];
      const whiskers: PathItem[] = [];
      for (let index = 0; index < 50; index++) {
        const fold = index % 5;
        const observed = random.range(0.05, 0.95);
        // Out-of-fold predictions shrink toward the mean and carry a per-fold bias.
        const predicted = clamp(
          0.5 + (observed - 0.5) * 0.85 + foldBias[fold] + random.gaussian() * p.noise,
          0.02,
          0.98
        );
        const [x, y] = toPoint(observed, predicted);
        dots.push({
          x,
          y,
          radius: 1.6,
          role: ROLE_CYCLE[fold],
          alpha: 0.95,
          phase: random.next()
        });
        whiskers.push({
          points: [[x, y], toPoint(observed, observed)],
          role: ROLE_CYCLE[fold],
          width: 0.3,
          alpha: 0.5
        });
      }
      const chips: PolygonItem[] = [0, 1, 2, 3, 4].map(fold => ({
        ring: createSizedRectangle(10, 14 + fold * 15, 14, 9),
        role: ROLE_CYCLE[fold],
        alpha: 0.7,
        phase: fold / 5
      }));
      return [
        createPulsingFillLayer(chips),
        createStrokeLayer(
          [
            getAxes(40, 8, 132, 92),
            {
              points: [toPoint(0, 0), toPoint(1, 1)],
              role: 'dim',
              width: 0.6,
              alpha: 0.9
            }
          ],
          {animation: 'static'}
        ),
        createStrokeLayer(whiskers, {animation: 'static', blend: 'add'}),
        createSpriteLayer(dots, {pulse: 0.3})
      ];
    }
  },

  'nested-interpolation-validation': {
    parameters: [createSlider('roughness', 'Surface roughness', 0, 1, 0.05, 0.5)],
    build: ({fork, p}) => {
      const random = fork('samples');
      const held: Point[][] = [createSizedRectangle(60, 8, 36, 40)];
      const items: SpriteItem[] = [];
      for (const [x, y] of createScatterPoints(random, 58, 10).map(
        ([x, y]) => [6 + ((x - 10) / 140) * 90, 8 + ((y - 10) / 80) * 84] as Point
      )) {
        const outer = x >= 60 && y <= 48;
        // Inside training, three inner folds (stripes) choose among the candidates.
        const role: Role = outer
          ? 'warm'
          : (['primary', 'violet', 'cool'] as const)[Math.floor(((x - 6) / 91) * 3) % 3];
        items.push({
          x,
          y,
          radius: outer ? 2 : 1.5,
          role,
          alpha: 0.95,
          phase: random.next()
        });
      }
      // Errors of three candidates on five outer blocks; the roughest surface favors the last candidate.
      const errors = fork('errors');
      const grid: number[][] = [0, 1, 2].map(candidate =>
        Array.from(
          {length: 5},
          () => Math.abs(candidate - p.roughness * 2) * 0.28 + 0.15 + errors.next() * 0.1
        )
      );
      const means = grid.map(row => row.reduce((sum, value) => sum + value, 0) / 5);
      const winner = means.indexOf(Math.min(...means));
      const cells: PolygonItem[] = [];
      grid.forEach((row, candidate) => {
        row.forEach((error, block) => {
          cells.push({
            ring: createSizedRectangle(110 + block * 8, 26 + candidate * 17, 7, 14),
            role: getClassRole(Math.min(4, Math.floor((error / 0.8) * 5)), 5),
            alpha: 0.7,
            phase: block / 5
          });
        });
      });
      const winnerRow = createSizedRectangle(109, 25.5 + winner * 17, 42, 15);
      return [
        createFillLayer([{ring: held[0], role: 'warm', alpha: 0.16}], {
          animation: 'pulse',
          blend: 'add'
        }),
        createOutlineLayer([createSizedRectangle(6, 8, 90, 84)], 'dim', 0.5),
        createStrokeLayer(
          [
            {
              points: closeRing(held[0]),
              role: 'warm',
              width: 0.9,
              alpha: 1
            },
            {
              points: createArc([96, 28], [107, 34], 0.25, 10),
              role: 'warm',
              width: 0.9,
              alpha: 1
            }
          ],
          {animation: 'flow', speed: 0.4, blend: 'add'}
        ),
        createSpriteLayer(items, {pulse: 0.3}),
        createPulsingFillLayer(cells),
        createStrokeLayer(
          [
            {
              points: closeRing(winnerRow),
              role: 'warm',
              width: 1.1,
              alpha: 1
            }
          ],
          {animation: 'pulse', speed: 0.5, blend: 'add'}
        ),
        createSpriteLayer(
          [
            {
              x: 104,
              y: 33 + winner * 17,
              radius: 2.6,
              role: 'warm',
              alpha: 1
            }
          ],
          {pulse: 0.5}
        )
      ];
    }
  }
};

export const SPATIAL_STATISTICS_PREVIEWS: Record<string, EntryPreview> = {
  ...AUTOCORRELATION_PREVIEWS,
  ...WEIGHTS_AND_TESTS_PREVIEWS,
  ...PERMUTATION_PREVIEWS,
  ...RATES_AND_SPACE_TIME_PREVIEWS,
  ...CLUSTERING_PREVIEWS,
  ...MODEL_PREVIEWS
};
