// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/dataframe-pointcloud-records.ts, keyed by entry id.

import type {GpuSceneParameter} from '../../types.ts';
import {
  clamp,
  createArc,
  createCircle,
  createClusterPoints,
  createSmoothField,
  createScatterPoints,
  evaluateField,
  FIELD_COLUMNS,
  FIELD_ROWS,
  getDistance,
  sampleField,
  type Random
} from '../data.ts';
import {closeRing, createRectangleRing} from '../geometry.ts';
import {getClassRole, getRoleCycle, type EntryPreview} from '../helpers.ts';
import {
  createChoice,
  createDot,
  createFillLayer,
  createPath,
  createSlider,
  createSpriteLayer,
  createStrokeLayer
} from '../layers.ts';
import type {
  FillLayer,
  PathItem,
  Point,
  PolygonItem,
  Role,
  SpriteItem,
  SpriteLayer,
  StrokeLayer
} from '../types.ts';

// Local layer builders with this group's defaults (explicit animation, speed, trail and pulse).

function createDotAt(
  x: number,
  y: number,
  radius: number,
  role: Role,
  alpha = 0.9,
  phase = 0
): SpriteItem {
  return createDot([x, y], role, radius, alpha, phase);
}

function createPlotPath(
  points: Point[],
  role: Role,
  width = 0.6,
  alpha = 0.8,
  phase = 0
): PathItem {
  return createPath(points, role, width, alpha, phase);
}

function createStrokes(
  items: PathItem[],
  animation: StrokeLayer['animation'] = 'static',
  speed = 0.3,
  trail = 1,
  blend: 'add' | 'alpha' = 'add'
): StrokeLayer {
  return createStrokeLayer(items, {animation, speed, trail, blend});
}

function createSprites(items: SpriteItem[], pulse = 0.3): SpriteLayer {
  return createSpriteLayer(items, {pulse});
}

function createFills(
  items: PolygonItem[],
  animation: FillLayer['animation'] = 'static',
  blend: 'add' | 'alpha' = 'add'
): FillLayer {
  return createFillLayer(items, {animation, blend});
}

/** The dim L-shaped axes of a small plot. */
function getAxes(left = 14, right = 150, top = 12, bottom = 86): StrokeLayer {
  return createStrokes(
    [
      createPlotPath(
        [
          [left, top],
          [left, bottom],
          [right, bottom]
        ],
        'dim',
        0.5,
        0.6
      )
    ],
    'static'
  );
}

function getCurve(
  left: number,
  right: number,
  bottom: number,
  height: number,
  evaluate: (amount: number) => number,
  steps = 36
): Point[] {
  return Array.from({length: steps + 1}, (_, index): Point => {
    const amount = index / steps;
    return [left + amount * (right - left), bottom - evaluate(amount) * height];
  });
}

function getBars(
  values: number[],
  left: number,
  right: number,
  bottom: number,
  height: number,
  getRole: (index: number) => Role,
  alpha = 0.85
): PathItem[] {
  const step = (right - left) / values.length;
  return values.map((value, index) => {
    const x = left + (index + 0.5) * step;
    return createPlotPath(
      [
        [x, bottom],
        [x, bottom - Math.max(value * height, 0.4)]
      ],
      getRole(index),
      step * 0.78,
      alpha,
      index / values.length
    );
  });
}

/** A right-skewed column (log-normal) scaled into 0..1. */
function getSkewedColumn(random: Random, count: number, skew = 0.7): number[] {
  const raw = Array.from({length: count}, () => Math.exp(random.gaussian() * skew));
  const sorted = [...raw].sort((first, second) => first - second);
  const top = sorted[Math.floor(count * 0.97)] || 1;
  return raw.map(value => clamp(value / top, 0, 1));
}

/** Counts per bin over 0..1, scaled so the tallest bin is 1. */
function getHistogram(values: number[], binCount: number, low = 0, high = 1): number[] {
  const counts = new Array<number>(binCount).fill(0);
  for (const value of values) {
    const bin = Math.floor(((value - low) / (high - low)) * binCount);
    if (bin >= 0 && bin < binCount) counts[bin]++;
  }
  const tallest = Math.max(...counts, 1);
  return counts.map(count => count / tallest);
}

function getMean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / (values.length || 1);
}

function getQuantile(sorted: number[], fraction: number): number {
  return sorted[clamp(Math.round(fraction * (sorted.length - 1)), 0, sorted.length - 1)];
}

// ---------------------------------------------------------------------------------------------
// Variogram helpers
// ---------------------------------------------------------------------------------------------

type VariogramModel = 'spherical' | 'exponential' | 'gaussian';
const VARIOGRAM_MODELS: VariogramModel[] = ['spherical', 'exponential', 'gaussian'];

function getVariogramValue(
  model: VariogramModel,
  lag: number,
  range: number,
  nugget: number,
  sill: number
): number {
  const ratio = lag / range;
  const partial = sill - nugget;
  if (model === 'spherical') {
    return nugget + partial * (ratio >= 1 ? 1 : 1.5 * ratio - 0.5 * ratio ** 3);
  }
  if (model === 'exponential') return nugget + partial * (1 - Math.exp(-3 * ratio));
  return nugget + partial * (1 - Math.exp(-3 * ratio * ratio));
}

const PLOT = {left: 18, right: 148, bottom: 86, height: 68};

function getPlotPoint(lagFraction: number, gamma: number): Point {
  return [
    PLOT.left + lagFraction * (PLOT.right - PLOT.left),
    PLOT.bottom - clamp(gamma, 0, 1.15) * PLOT.height
  ];
}

// ---------------------------------------------------------------------------------------------
// Point cloud helpers (3D turntable scenes, local coordinates around x = 80)
// ---------------------------------------------------------------------------------------------

type CloudKind = 'ground' | 'canopy' | 'roof' | 'wall';
type CloudPoint = {x: number; z: number; height: number; ground: number; kind: CloudKind};

const TREE_CENTERS: Point[] = [
  [-36, -18],
  [-14, 16],
  [8, 22],
  [40, 12],
  [-48, 10],
  [32, -30]
];
const BUILDING = {left: 14, right: 32, front: -16, back: 0, roof: 14};

function getTerrainHeight(x: number, z: number): number {
  return 6 * Math.sin(x / 18) * Math.cos(z / 14) + 3 * Math.sin(z / 7 + 1);
}

function createCloudPoints(random: Random, count: number): CloudPoint[] {
  const points: CloudPoint[] = [];
  const buildingBase = getTerrainHeight(23, -8);
  for (let index = 0; index < count; index++) {
    const roll = random.next();
    if (roll < 0.55) {
      const x = random.range(-60, 60);
      const z = random.range(-40, 40);
      const ground = getTerrainHeight(x, z);
      points.push({x, z, height: ground, ground, kind: 'ground'});
    } else if (roll < 0.8) {
      const [centerX, centerZ] = TREE_CENTERS[random.integer(TREE_CENTERS.length)];
      const angle = random.range(0, Math.PI * 2);
      const radius = 8 * Math.sqrt(random.next());
      const x = centerX + Math.cos(angle) * radius;
      const z = centerZ + Math.sin(angle) * radius;
      const ground = getTerrainHeight(x, z);
      const rise = 6 + random.range(0, 10) * (1 - radius / 10);
      points.push({x, z, height: ground + rise, ground, kind: 'canopy'});
    } else if (random.next() < 0.6) {
      const x = random.range(BUILDING.left, BUILDING.right);
      const z = random.range(BUILDING.front, BUILDING.back);
      points.push({
        x,
        z,
        height: buildingBase + BUILDING.roof,
        ground: buildingBase,
        kind: 'roof'
      });
    } else {
      const alongX = random.next() < 0.5;
      const x = alongX
        ? random.range(BUILDING.left, BUILDING.right)
        : random.next() < 0.5
          ? BUILDING.left
          : BUILDING.right;
      const z = alongX
        ? random.next() < 0.5
          ? BUILDING.front
          : BUILDING.back
        : random.range(BUILDING.front, BUILDING.back);
      points.push({
        x,
        z,
        height: buildingBase + random.range(0, BUILDING.roof),
        ground: buildingBase,
        kind: 'wall'
      });
    }
  }
  return points;
}

function createCloudDot(
  point: {x: number; z: number},
  height: number,
  role: Role,
  radius = 0.9,
  alpha = 0.85,
  phase = 0
): SpriteItem {
  return {x: 80 + point.x, y: 74 - height * 1.9, z: point.z, radius, role, alpha, phase};
}

function createCloudLayer(items: SpriteItem[], depth: number, pulse = 0.2): SpriteLayer {
  return {
    type: 'sprites',
    items,
    pulse,
    turntable: {angle: 0.5, speed: 0, perspective: depth / 1000}
  };
}

const SPIN_PARAMETERS: GpuSceneParameter[] = [
  createSlider('spin', 'Spin', -90, 90, 5, 15, '°/s'),
  createSlider('depth', 'Perspective', 0, 8, 0.5, 4.5)
];

// ---------------------------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------------------------

export const DATAFRAME_POINTCLOUD_RECORDS_PREVIEWS: Record<string, EntryPreview> = {
  // ---- Column statistics and classification -------------------------------------------------

  // A column strip is scanned and summarized into a histogram with a mean marker.
  'column-profile': {
    parameters: [createSlider('bins', 'Bins', 6, 28, 1, 16)],
    pointer: 'none',
    build: ({fork, p}) => {
      const values = getSkewedColumn(fork('column'), 320);
      const counts = getHistogram(values, Math.round(p.bins));
      const left = 50;
      const right = 148;
      const mean = getMean(values);
      const strip = Array.from({length: 12}, (_, index) =>
        createPlotPath(
          [
            [10, 16 + index * 6],
            [24, 16 + index * 6]
          ],
          'primary',
          4,
          0.15 + 0.7 * values[index],
          index / 12
        )
      );
      const step = (right - left) / counts.length;
      const arcs = [1, 4, 7, 10].map((row, index) => {
        const bin = Math.floor(((index + 0.5) / 4) * counts.length * 0.8);
        return createPlotPath(
          createArc(
            [26, 16 + row * 6],
            [left + (bin + 0.5) * step, 84 - counts[bin] * 60],
            0.25,
            20
          ),
          'primary',
          0.5,
          0.8,
          index / 4
        );
      });
      return [
        createStrokes(strip, 'pulse', 0.4),
        createStrokes(
          getBars(
            counts,
            left,
            right,
            84,
            60,
            index => (counts[index] > 0.85 ? 'warm' : 'cool'),
            0.8
          ),
          'pulse',
          0.2
        ),
        createStrokes(arcs, 'flow', 0.35, 1.2),
        createStrokes(
          [
            createPlotPath(
              [
                [left + mean * (right - left), 84],
                [left + mean * (right - left), 20]
              ],
              'warm',
              0.7,
              0.9
            )
          ],
          'pulse',
          0.5
        ),
        createSprites([
          createDotAt(left, 86, 1.8, 'cool'),
          createDotAt(right, 86, 1.8, 'hot'),
          createDotAt(left + mean * (right - left), 18, 2.4, 'warm')
        ])
      ];
    }
  },

  // Histogram shaded by the trimmed band, with quartile lines and a box plot underneath.
  'column-quantiles': {
    parameters: [createSlider('trim', 'Trim each tail', 0, 20, 1, 5, '%')],
    pointer: 'none',
    build: ({fork, p}) => {
      const values = getSkewedColumn(fork('column'), 400);
      const sorted = [...values].sort((first, second) => first - second);
      const counts = getHistogram(values, 30);
      const left = 14;
      const right = 148;
      const toX = (value: number) => left + value * (right - left);
      const lowCut = getQuantile(sorted, p.trim / 100);
      const highCut = getQuantile(sorted, 1 - p.trim / 100);
      const [q1, median, q3] = [0.25, 0.5, 0.75].map(fraction => getQuantile(sorted, fraction));
      const line = (value: number, role: Role, bottom = 70, top = 18) =>
        createPlotPath(
          [
            [toX(value), bottom],
            [toX(value), top]
          ],
          role,
          0.7,
          0.9
        );
      return [
        createStrokes(
          getBars(counts, left, right, 66, 46, index => {
            const center = (index + 0.5) / counts.length;
            return center >= lowCut && center <= highCut ? 'primary' : 'dim';
          }),
          'pulse',
          0.2
        ),
        createFills(
          [{ring: createRectangleRing(toX(q1), 72, toX(q3), 88), role: 'violet', alpha: 0.35}],
          'pulse'
        ),
        createStrokes(
          [
            createPlotPath(
              [
                [toX(sorted[0]), 80],
                [toX(sorted[sorted.length - 1]), 80]
              ],
              'dim',
              0.4,
              0.6
            ),
            createPlotPath(
              [
                [toX(lowCut), 80],
                [toX(highCut), 80]
              ],
              'primary',
              0.8,
              0.9
            )
          ],
          'static'
        ),
        createStrokes(
          [line(q1, 'warm'), line(q3, 'warm'), line(median, 'hot', 90, 18)],
          'pulse',
          0.4
        ),
        createSprites([
          createDotAt(toX(lowCut), 80, 1.8, 'cool'),
          createDotAt(toX(highCut), 80, 1.8, 'hot')
        ])
      ];
    }
  },

  // Natural-breaks classes: bars coloured by class, break lines between, class ribbon below.
  'class-breaks': {
    parameters: [createSlider('classes', 'Classes', 2, 7, 1, 5)],
    pointer: 'none',
    build: ({fork, p}) => {
      const values = getSkewedColumn(fork('column'), 400);
      const binCount = 40;
      const counts = getHistogram(values, binCount);
      const classes = Math.round(p.classes);
      // One dimensional k-means over the bin centers, weighted by count.
      let centers = Array.from({length: classes}, (_, index) => (index + 0.5) / classes);
      for (let iteration = 0; iteration < 8; iteration++) {
        const sums = new Array<number>(classes).fill(0);
        const weights = new Array<number>(classes).fill(0);
        counts.forEach((count, bin) => {
          const center = (bin + 0.5) / binCount;
          let best = 0;
          centers.forEach((candidate, index) => {
            if (Math.abs(candidate - center) < Math.abs(centers[best] - center)) best = index;
          });
          sums[best] += center * count;
          weights[best] += count;
        });
        centers = centers.map((center, index) =>
          weights[index] ? sums[index] / weights[index] : center
        );
      }
      centers.sort((first, second) => first - second);
      const getClass = (center: number) => {
        let best = 0;
        centers.forEach((candidate, index) => {
          if (Math.abs(candidate - center) < Math.abs(centers[best] - center)) best = index;
        });
        return best;
      };
      const left = 14;
      const right = 148;
      const breaks = centers.slice(1).map((center, index) => (center + centers[index]) / 2);
      const ribbon: PathItem[] = [];
      let start = 0;
      [...breaks, 1].forEach((end, index) => {
        ribbon.push(
          createPlotPath(
            [
              [left + start * (right - left) + 0.5, 82],
              [left + end * (right - left) - 0.5, 82]
            ],
            getClassRole(index, classes),
            4,
            0.9
          )
        );
        start = end;
      });
      return [
        createStrokes(
          getBars(counts, left, right, 70, 52, index =>
            getClassRole(getClass((index + 0.5) / binCount), classes)
          ),
          'pulse',
          0.2
        ),
        createStrokes(
          breaks.map(value =>
            createPlotPath(
              [
                [left + value * (right - left), 74],
                [left + value * (right - left), 14]
              ],
              'ink',
              0.5,
              0.9
            )
          ),
          'pulse',
          0.5
        ),
        createStrokes(ribbon, 'flow', 0.25, 1.4)
      ];
    }
  },

  // A number line of values, threshold ticks, and a stepped palette they are painted with.
  'color-scale': {
    parameters: [createSlider('steps', 'Palette steps', 3, 7, 1, 5)],
    pointer: 'none',
    build: ({fork, p}) => {
      const values = getSkewedColumn(fork('column'), 44, 0.6);
      const steps = Math.round(p.steps);
      const sorted = [...values].sort((first, second) => first - second);
      const thresholds = Array.from({length: steps - 1}, (_, index) =>
        getQuantile(sorted, (index + 1) / steps)
      );
      const getStep = (value: number) => thresholds.filter(threshold => value >= threshold).length;
      const toX = (value: number) => 14 + value * 132;
      const items = values.map((value, index) =>
        createDotAt(
          toX(value),
          30 + (index % 4) * 6,
          1.7,
          getClassRole(getStep(value), steps),
          0.9,
          index / 44
        )
      );
      const swatches: PathItem[] = [];
      const edges = [0, ...thresholds, 1];
      for (let index = 0; index < steps; index++) {
        swatches.push(
          createPlotPath(
            [
              [toX(edges[index]) + 0.5, 76],
              [toX(edges[index + 1]) - 0.5, 76]
            ],
            getClassRole(index, steps),
            6,
            0.85,
            index / steps
          )
        );
      }
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [14, 58],
                [146, 58]
              ],
              'dim',
              0.4,
              0.6
            )
          ],
          'static'
        ),
        createStrokes(
          thresholds.map(threshold =>
            createPlotPath(
              [
                [toX(threshold), 64],
                [toX(threshold), 22]
              ],
              'dim',
              0.5,
              0.8
            )
          ),
          'pulse',
          0.4
        ),
        createSprites(items, 0.4),
        createStrokes(swatches, 'flow', 0.2, 1.6)
      ];
    }
  },

  // Proportional symbols: domain ticks map to a range of radii.
  'scalar-scale': {
    parameters: [
      createSlider('maxRadius', 'Max radius', 2, 9, 0.5, 6.5, 'px'),
      createChoice('scale', 'Scale', ['linear', 'sqrt'], 1)
    ],
    pointer: 'none',
    build: ({p}) => {
      const count = 8;
      const items: SpriteItem[] = [];
      const links: PathItem[] = [];
      for (let index = 0; index < count; index++) {
        const amount = index / (count - 1);
        const scaled = p.scale >= 0.5 ? Math.sqrt(amount) : amount;
        const x = 20 + amount * 120;
        const radius = 1 + scaled * (p.maxRadius - 1);
        items.push(createDotAt(x, 62, radius, getClassRole(index, count), 0.85, amount));
        links.push(
          createPlotPath(
            [
              [x, 24],
              [x, 62 - radius]
            ],
            'dim',
            0.4,
            0.8,
            amount
          )
        );
      }
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [20, 20],
                [140, 20]
              ],
              'dim',
              0.5,
              0.7
            ),
            ...Array.from({length: count}, (_, index) =>
              createPlotPath(
                [
                  [20 + (index / (count - 1)) * 120, 17],
                  [20 + (index / (count - 1)) * 120, 23]
                ],
                'ink',
                0.5,
                0.8
              )
            )
          ],
          'static'
        ),
        createStrokes(links, 'flow', 0.35, 1),
        createSprites(items, 0.3),
        createStrokes(
          [
            createPlotPath(
              [
                [20, 86],
                [20 + 120, 86]
              ],
              'dim',
              0.4,
              0.6
            )
          ],
          'static'
        )
      ];
    }
  },

  // A 3 by 3 grid whose cells mix two hues; the scatter falls into it.
  'bivariate-classification': {
    parameters: [createSlider('correlation', 'Correlation', 0, 1, 0.05, 0.55)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('scatter');
      const left = 42;
      const top = 14;
      const cellWidth = 25;
      const cellHeight = 22;
      const cells: PolygonItem[] = [];
      const lines: PathItem[] = [];
      for (let row = 0; row < 3; row++) {
        for (let column = 0; column < 3; column++) {
          const ring = createRectangleRing(
            left + column * cellWidth + 1,
            top + row * cellHeight + 1,
            left + (column + 1) * cellWidth - 1,
            top + (row + 1) * cellHeight - 1
          );
          const high = 2 - row;
          cells.push({ring, role: 'cool', alpha: 0.05 + column * 0.17});
          cells.push({ring, role: 'warm', alpha: 0.05 + high * 0.17});
          lines.push(createPlotPath(closeRing(ring), 'dim', 0.3, 0.5));
        }
      }
      const points = Array.from({length: 38}, (_, index) => {
        const valueA = random.next();
        const valueB = clamp(
          p.correlation * valueA + (1 - p.correlation) * random.next(),
          0,
          0.999
        );
        const column = Math.min(2, Math.floor(valueA * 3));
        const row = Math.min(2, Math.floor(valueB * 3));
        const role: Role =
          column === 2 && row === 2
            ? 'hot'
            : column === 0 && row === 0
              ? 'dim'
              : column > row
                ? 'cool'
                : row > column
                  ? 'warm'
                  : 'violet';
        return createDotAt(
          left + valueA * cellWidth * 3,
          top + (1 - valueB) * cellHeight * 3,
          1.4,
          role,
          0.95,
          index / 38
        );
      });
      return [
        createFills(cells, 'pulse'),
        createStrokes(lines, 'static'),
        createStrokes(
          [
            createPlotPath(
              [
                [left, 86],
                [left + 75, 86]
              ],
              'cool',
              0.9,
              0.9
            ),
            createPlotPath(
              [
                [36, top + 66],
                [36, top]
              ],
              'warm',
              0.9,
              0.9
            )
          ],
          'flow',
          0.3
        ),
        createSprites(points, 0.4)
      ];
    }
  },

  // A skewed histogram recentered and rescaled: shape stays, position and spread change.
  'variable-standardization': {
    parameters: [createSlider('skew', 'Skew', 0.2, 1.2, 0.05, 0.7)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('column');
      const raw = Array.from({length: 360}, () => Math.exp(random.gaussian() * p.skew));
      const mean = getMean(raw);
      const deviation = Math.sqrt(getMean(raw.map(value => (value - mean) ** 2))) || 1;
      const top = Math.max(...raw) * 0.6;
      const original = raw.map(value => clamp(value / (top * 1.6), 0, 0.999));
      const standardized = raw.map(value => clamp(((value - mean) / deviation + 2) / 6, 0, 0.999));
      const rawCounts = getHistogram(original, 32);
      const zCounts = getHistogram(standardized, 32);
      const zeroX = 14 + (2 / 6) * 134;
      const sigmaX = (1 / 6) * 134;
      return [
        createStrokes(
          getBars(rawCounts, 14, 148, 44, 28, () => 'dim', 0.7),
          'static'
        ),
        createStrokes(
          getBars(zCounts, 14, 148, 90, 30, index =>
            Math.abs(index / 32 - 2 / 6) < 1 / 6 ? 'primary' : 'cool'
          ),
          'pulse',
          0.2
        ),
        createStrokes(
          [
            createPlotPath(
              [
                [zeroX, 92],
                [zeroX, 54]
              ],
              'warm',
              0.7,
              0.9
            ),
            createPlotPath(
              [
                [zeroX - sigmaX, 90],
                [zeroX - sigmaX, 62]
              ],
              'ink',
              0.4,
              0.6
            ),
            createPlotPath(
              [
                [zeroX + sigmaX, 90],
                [zeroX + sigmaX, 62]
              ],
              'ink',
              0.4,
              0.6
            ),
            createPlotPath(
              [
                [14 + (mean / (top * 1.6)) * 134, 46],
                [14 + (mean / (top * 1.6)) * 134, 14]
              ],
              'warm',
              0.5,
              0.6
            )
          ],
          'pulse',
          0.4
        ),
        createStrokes(
          [
            createPlotPath(
              createArc([14 + (mean / (top * 1.6)) * 134, 34], [zeroX, 60], -0.35, 20),
              'warm',
              0.8,
              0.9
            )
          ],
          'flow',
          0.4,
          1.2
        )
      ];
    }
  },

  // Four indicator grids fold into one composite grid; the first weight is adjustable.
  'composite-score': {
    parameters: [createSlider('weight', 'First weight', 0, 1, 0.05, 0.5)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('indicators');
      const waves = Array.from({length: 4}, () => ({
        kx: random.range(2, 6),
        ky: random.range(2, 6),
        phase: random.range(0, 6.28)
      }));
      const sample = (index: number, u: number, v: number) =>
        0.5 + 0.5 * Math.sin(waves[index].kx * u + waves[index].ky * v + waves[index].phase);
      const roles: Role[] = ['cool', 'violet', 'primary', 'warm'];
      const weights = [p.weight, 0.5, 0.5, 0.5];
      const weightTotal = weights.reduce((sum, weight) => sum + weight, 0) || 1;
      const miniItems: SpriteItem[] = [];
      const flows: PathItem[] = [];
      waves.forEach((_, index) => {
        const originY = 10 + index * 22;
        for (let row = 0; row < 4; row++) {
          for (let column = 0; column < 4; column++) {
            miniItems.push(
              createDotAt(
                14 + column * 4,
                originY + row * 4,
                1.1,
                roles[index],
                0.2 + 0.75 * sample(index, column / 3, row / 3),
                index / 4
              )
            );
          }
        }
        flows.push(
          createPlotPath(
            createArc([32, originY + 6], [90, 48], index < 2 ? 0.12 : -0.12, 24),
            roles[index],
            0.4 + weights[index] * 1.2,
            0.8,
            index / 4
          )
        );
      });
      const output: SpriteItem[] = [];
      for (let row = 0; row < 9; row++) {
        for (let column = 0; column < 9; column++) {
          const u = column / 8;
          const v = row / 8;
          let value = 0;
          for (let index = 0; index < 4; index++)
            value += (weights[index] / weightTotal) * sample(index, u, v);
          output.push(
            createDotAt(
              100 + column * 6,
              22 + row * 6,
              2.3,
              getClassRole(Math.min(4, Math.floor(value * 5)), 5),
              0.35 + 0.6 * value,
              value
            )
          );
        }
      }
      return [
        createStrokes(flows, 'flow', 0.3, 1.2),
        createSprites(miniItems, 0.15),
        createSprites(output, 0.35)
      ];
    }
  },

  // Lorenz curves bowing under the equality line; the gap is the Gini.
  inequality: {
    parameters: [createSlider('gini', 'Gini', 0.1, 0.85, 0.05, 0.5)],
    pointer: 'none',
    build: ({p}) => {
      const left = 24;
      const right = 134;
      const bottom = 86;
      const height = 70;
      const exponentFor = (gini: number) => (1 + gini) / (1 - gini);
      const lorenz = (gini: number) =>
        getCurve(left, right, bottom, height, amount => amount ** exponentFor(gini), 40);
      const main = lorenz(p.gini);
      const diagonal = getCurve(left, right, bottom, height, amount => amount, 2);
      const gap = [...diagonal, ...[...main].reverse()];
      return [
        getAxes(left, right + 6, 12, bottom),
        createFills([{ring: gap, role: 'hot', alpha: 0.28}], 'pulse'),
        createStrokes([createPlotPath(diagonal, 'dim', 0.5, 0.8)], 'static'),
        createStrokes(
          [
            createPlotPath(lorenz(Math.max(0.1, p.gini - 0.3)), 'cool', 0.8, 0.8),
            createPlotPath(lorenz(Math.min(0.9, p.gini + 0.2)), 'violet', 0.8, 0.8)
          ],
          'draw',
          0.15
        ),
        createStrokes([createPlotPath(main, 'hot', 1.3, 1)], 'flow', 0.3, 1.6),
        createSprites(
          [createDotAt(left, bottom, 1.6, 'dim'), createDotAt(right, bottom - height, 1.6, 'dim')],
          0
        )
      ];
    }
  },

  // Dots in four group columns with a mean bar and a standard deviation whisker per group.
  'group-statistics': {
    parameters: [createSlider('spread', 'Spread', 0.4, 2, 0.05, 1)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('rows');
      const means = [58, 38, 66, 30];
      const deviations = [7, 11, 5, 9];
      const roles: Role[] = ['cool', 'primary', 'violet', 'warm'];
      const dots: SpriteItem[] = [];
      const bars: PathItem[] = [];
      const whiskers: PathItem[] = [];
      means.forEach((mean, group) => {
        const x = 30 + group * 33;
        const deviation = deviations[group] * p.spread;
        for (let index = 0; index < 14; index++) {
          dots.push(
            createDotAt(
              x + random.range(-7, 7),
              clamp(mean + random.gaussian() * deviation, 8, 92),
              1.2,
              roles[group],
              0.75,
              random.next()
            )
          );
        }
        bars.push(
          createPlotPath(
            [
              [x - 9, mean],
              [x + 9, mean]
            ],
            'hot',
            1.3,
            1,
            group / 4
          )
        );
        whiskers.push(
          createPlotPath(
            [
              [x, mean - deviation],
              [x, mean + deviation]
            ],
            'ink',
            0.6,
            0.9,
            group / 4
          )
        );
        whiskers.push(
          createPlotPath(
            [
              [x - 3, mean - deviation],
              [x + 3, mean - deviation]
            ],
            'ink',
            0.6,
            0.9
          )
        );
        whiskers.push(
          createPlotPath(
            [
              [x - 3, mean + deviation],
              [x + 3, mean + deviation]
            ],
            'ink',
            0.6,
            0.9
          )
        );
      });
      return [
        createSprites(dots, 0.35),
        createStrokes(whiskers, 'pulse', 0.4),
        createStrokes(bars, 'flow', 0.25, 1.6)
      ];
    }
  },

  // Two tables linked by key: right rows are gathered onto left rows; a left join keeps orphans.
  'key-join': {
    parameters: [createChoice('joinKind', 'Join kind', ['left', 'inner'])],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('keys');
      const rightKeys = [0, 1, 2, 3, 4, 5];
      const leftKeys = Array.from({length: 9}, () => random.integer(8));
      const getY = (index: number, count: number) => 14 + (index / (count - 1)) * 72;
      const left: SpriteItem[] = [];
      const result: SpriteItem[] = [];
      const links: PathItem[] = [];
      const orphans: SpriteItem[] = [];
      const inner = p.joinKind >= 0.5;
      leftKeys.forEach((key, index) => {
        const y = getY(index, leftKeys.length);
        const rightIndex = rightKeys.indexOf(key);
        left.push(createDotAt(24, y, 2, getRoleCycle(key), 0.9, index / 9));
        if (rightIndex >= 0) {
          const rightY = getY(rightIndex, rightKeys.length);
          links.push(
            createPlotPath(
              [
                [134, rightY],
                [108, rightY],
                [56, y],
                [30, y]
              ],
              getRoleCycle(key),
              0.6,
              0.85,
              index / 9
            )
          );
          result.push(createDotAt(76, y, 1.8, getRoleCycle(key), 0.9, index / 9));
        } else if (!inner) {
          orphans.push(createDotAt(76, y, 1.4, 'dim', 0.55));
          orphans.push(createDotAt(24, y, 4, 'hot', 0.25, index / 9));
        }
      });
      const right = rightKeys.map((key, index) =>
        createDotAt(136, getY(index, rightKeys.length), 2.6, getRoleCycle(key), 0.9, index / 6)
      );
      return [
        createStrokes(links, 'flow', 0.3, 1.2),
        createStrokes(
          [
            createPlotPath(
              [
                [76, 8],
                [76, 92]
              ],
              'dim',
              0.3,
              0.35
            )
          ],
          'static'
        ),
        createSprites([...left, ...right, ...result], 0.3),
        createSprites(orphans, 0.5)
      ];
    }
  },

  // ---- Time ----------------------------------------------------------------------------------

  // Events on a timeline folded into a weekday by hour grid.
  'calendar-buckets': {
    parameters: [createSlider('events', 'Events', 10, 60, 2, 36)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('events');
      const columns = 7;
      const rows = 5;
      const counts = new Array<number>(columns * rows).fill(0);
      const ticks: SpriteItem[] = [];
      const flows: PathItem[] = [];
      const cellX = (column: number) => 38 + column * 14;
      const cellY = (row: number) => 38 + row * 11;
      const pool = Array.from({length: 60}, () => ({
        time: random.next(),
        row: random.integer(rows)
      }));
      pool.slice(0, Math.round(p.events)).forEach((event, index) => {
        const column = Math.min(columns - 1, Math.floor(event.time * columns));
        counts[event.row * columns + column]++;
        const x = 18 + event.time * 124;
        ticks.push(createDotAt(x, 12, 0.9, 'primary', 0.9, index / 60));
        flows.push(
          createPlotPath(
            [
              [x, 14],
              [cellX(column), cellY(event.row) - 4]
            ],
            'primary',
            0.3,
            0.55,
            random.next()
          )
        );
      });
      const maximum = Math.max(...counts, 1);
      const cells = counts.map((count, index) => {
        const amount = count / maximum;
        return createDotAt(
          cellX(index % columns),
          cellY(Math.floor(index / columns)),
          4.3,
          getClassRole(Math.min(4, Math.round(amount * 4)), 5),
          0.12 + 0.85 * amount,
          amount
        );
      });
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [16, 12],
                [144, 12]
              ],
              'dim',
              0.4,
              0.6
            ),
            ...Array.from({length: columns - 1}, (_, index) =>
              createPlotPath(
                [
                  [18 + ((index + 1) / columns) * 124, 9],
                  [18 + ((index + 1) / columns) * 124, 15]
                ],
                'ink',
                0.4,
                0.7
              )
            )
          ],
          'static'
        ),
        createStrokes(flows, 'flow', 0.3, 1),
        createSprites(cells, 0.3),
        createSprites(ticks, 0.2)
      ];
    }
  },

  // A window slides over bursty events; those inside are copied down to a selected strip.
  'time-window-filter': {
    parameters: [createSlider('width', 'Window', 10, 90, 2, 40, 'px')],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('events');
      const centers = [28, 62, 104, 132];
      const events = Array.from({length: 56}, () =>
        clamp(centers[random.integer(4)] + random.gaussian() * 9, 8, 152)
      );
      const center = pointer ? clamp(pointer[0], 8, 152) : 72;
      const half = p.width / 2;
      const inside = (x: number) => Math.abs(x - center) <= half;
      const top: SpriteItem[] = events.map((x, index) =>
        createDotAt(
          x,
          30 + (index % 3) * 3,
          1.4,
          inside(x) ? 'warm' : 'dim',
          inside(x) ? 0.95 : 0.45,
          index / 56
        )
      );
      const picked: SpriteItem[] = events
        .filter(inside)
        .map((x, index) => createDotAt(x, 74 + (index % 3) * 3, 1.4, 'warm', 0.95, index / 20));
      const drops: PathItem[] = events.filter(inside).map((x, index) =>
        createPlotPath(
          [
            [x, 38],
            [x, 72]
          ],
          'warm',
          0.3,
          0.5,
          index / 20
        )
      );
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [8, 44],
                [152, 44]
              ],
              'dim',
              0.4,
              0.6
            ),
            createPlotPath(
              [
                [8, 86],
                [152, 86]
              ],
              'dim',
              0.3,
              0.4
            )
          ],
          'static'
        ),
        createFills(
          [
            {
              ring: createRectangleRing(center - half, 22, center + half, 50),
              role: 'warm',
              alpha: 0.12
            }
          ],
          'pulse'
        ),
        createStrokes(
          [
            createPlotPath(
              [
                [center - half, 20],
                [center - half, 52]
              ],
              'warm',
              0.8,
              0.9
            ),
            createPlotPath(
              [
                [center + half, 20],
                [center + half, 52]
              ],
              'warm',
              0.8,
              0.9
            )
          ],
          'pulse',
          0.5
        ),
        createStrokes(drops, 'flow', 0.4, 0.8),
        createSprites(top, 0.3),
        createSprites(picked, 0.3)
      ];
    }
  },

  // Raw samples per hour reduced to a minimum and maximum, joined into two envelope lines.
  'temporal-reduction': {
    parameters: [createSlider('samples', 'Samples per hour', 3, 12, 1, 6)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('samples');
      const hours = 24;
      const raw: SpriteItem[] = [];
      const whiskers: PathItem[] = [];
      const lows: Point[] = [];
      const highs: Point[] = [];
      for (let hour = 0; hour < hours; hour++) {
        const x = 14 + (hour / (hours - 1)) * 132;
        const base = 0.5 + 0.28 * Math.sin((hour / hours) * Math.PI * 2 - 1.2);
        const samples = Array.from({length: 12}, () =>
          clamp(base + random.gaussian() * 0.1, 0.05, 0.98)
        );
        const used = samples.slice(0, Math.round(p.samples));
        const low = Math.min(...used);
        const high = Math.max(...used);
        used.forEach(value => raw.push(createDotAt(x, 90 - value * 76, 0.7, 'dim', 0.5)));
        whiskers.push(
          createPlotPath(
            [
              [x, 90 - low * 76],
              [x, 90 - high * 76]
            ],
            'violet',
            0.5,
            0.7,
            hour / hours
          )
        );
        lows.push([x, 90 - low * 76]);
        highs.push([x, 90 - high * 76]);
      }
      return [
        createSprites(raw, 0),
        createStrokes(whiskers, 'pulse', 0.3),
        createStrokes(
          [createPlotPath(lows, 'cool', 1, 0.95), createPlotPath(highs, 'hot', 1, 0.95)],
          'flow',
          0.25,
          1.4
        ),
        createSprites(
          [
            ...lows.map(([x, y], index) => createDotAt(x, y, 1.4, 'cool', 0.9, index / hours)),
            ...highs.map(([x, y], index) => createDotAt(x, y, 1.4, 'hot', 0.9, index / hours))
          ],
          0.3
        )
      ];
    }
  },

  // ---- Point pattern statistics -------------------------------------------------------------

  // Clustered points and a K curve above the random expectation; the ring sets the radius.
  'ripley-k': {
    parameters: [createSlider('radius', 'Radius', 4, 34, 1, 16, 'px')],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('points');
      const points = createClusterPoints(
        random,
        [
          [28, 30],
          [60, 66],
          [38, 74]
        ],
        14,
        6
      ).map(([x, y]): Point => [clamp(x, 6, 88), y]);
      const focal: Point = pointer && pointer[0] < 92 ? pointer : points[3];
      const inRing = points.map(point => getDistance(point, focal) <= p.radius);
      const t = clamp(p.radius / 34, 0, 1);
      const observed = (amount: number) =>
        clamp(amount * amount * (1 + 2.4 * (1 - amount) ** 1.5), 0, 1);
      const expected = (amount: number) => amount * amount;
      const left = 104;
      const right = 150;
      const markerObserved: Point = [left + t * (right - left), 84 - observed(t) * 64];
      const markerExpected: Point = [left + t * (right - left), 84 - expected(t) * 64];
      return [
        getAxes(left, right, 18, 84),
        createStrokes(
          [
            createPlotPath(
              createCircle(focal[0], focal[1], p.radius, 36).concat([
                [focal[0] + p.radius, focal[1]]
              ]),
              'warm',
              0.7,
              0.9
            )
          ],
          'pulse',
          0.5
        ),
        createSprites(
          points.map(([x, y], index) =>
            createDotAt(
              x,
              y,
              1.5,
              inRing[index] ? 'warm' : 'primary',
              inRing[index] ? 1 : 0.7,
              index / 40
            )
          ),
          0.3
        ),
        createSprites([createDotAt(focal[0], focal[1], 2.6, 'hot')], 0.5),
        createStrokes(
          [createPlotPath(getCurve(left, right, 84, 64, expected, 30), 'dim', 0.6, 0.8)],
          'static'
        ),
        createStrokes(
          [createPlotPath(getCurve(left, right, 84, 64, observed, 30), 'hot', 1.1, 0.95)],
          'flow',
          0.3,
          1.4
        ),
        createStrokes(
          [createPlotPath([markerExpected, markerObserved], 'warm', 0.6, 0.9)],
          'pulse',
          0.5
        ),
        createSprites(
          [
            createDotAt(markerObserved[0], markerObserved[1], 2, 'hot'),
            createDotAt(markerExpected[0], markerExpected[1], 1.5, 'dim')
          ],
          0.3
        )
      ];
    }
  },

  // The observed curve against a band drawn from simulated random patterns.
  'ripley-envelope': {
    parameters: [createSlider('simulations', 'Simulations', 5, 30, 1, 16)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('simulations');
      const left = 16;
      const right = 148;
      const bottom = 86;
      const height = 68;
      const steps = 40;
      const waves = Array.from({length: 30}, () => ({
        amplitudeA: random.range(-0.35, 0.35),
        phaseA: random.range(0, 6.28),
        amplitudeB: random.range(-0.2, 0.2),
        phaseB: random.range(0, 6.28)
      }));
      const used = waves.slice(0, Math.round(p.simulations));
      const expected = (amount: number) => amount * amount;
      const simulate = (wave: (typeof waves)[number], amount: number) =>
        expected(amount) *
        (1 +
          wave.amplitudeA * Math.sin(amount * 9 + wave.phaseA) +
          wave.amplitudeB * Math.sin(amount * 17 + wave.phaseB));
      const upper: Point[] = [];
      const lower: Point[] = [];
      for (let step = 0; step <= steps; step++) {
        const amount = step / steps;
        const values = used.map(wave => simulate(wave, amount));
        upper.push([left + amount * (right - left), bottom - Math.max(...values) * height]);
        lower.push([left + amount * (right - left), bottom - Math.min(...values) * height]);
      }
      const observed = (amount: number) =>
        expected(amount) * (1 + 1.5 * Math.exp(-(((amount - 0.38) / 0.17) ** 2)));
      const observedPoints = getCurve(left, right, bottom, height, observed, steps);
      const hotRuns: Point[][] = [];
      let run: Point[] = [];
      observedPoints.forEach((point, index) => {
        if (point[1] < upper[index][1] - 0.2) run.push(point);
        else {
          if (run.length > 1) hotRuns.push(run);
          run = [];
        }
      });
      if (run.length > 1) hotRuns.push(run);
      return [
        getAxes(left, right + 2, 10, bottom),
        createFills(
          [{ring: [...upper, ...[...lower].reverse()], role: 'cool', alpha: 0.28}],
          'pulse'
        ),
        createStrokes(
          used.map(wave =>
            createPlotPath(
              getCurve(left, right, bottom, height, amount => simulate(wave, amount), steps),
              'dim',
              0.25,
              0.4
            )
          ),
          'static'
        ),
        createStrokes([createPlotPath(observedPoints, 'warm', 1.1, 0.95)], 'flow', 0.3, 1.5),
        createStrokes(
          hotRuns.map(points => createPlotPath(points, 'hot', 1.8, 1)),
          'pulse',
          0.6
        )
      ];
    }
  },

  // Nearest-neighbor distances (G) and empty-space distances (F) from probe locations.
  'ripley-distance-functions': {
    parameters: [createSlider('points', 'Points', 12, 60, 2, 28)],
    pointer: 'none',
    build: ({fork, p}) => {
      const points = createScatterPoints(fork('points'), 60, 8)
        .map(([x, y]): Point => [8 + ((x - 8) / 144) * 78, y])
        .slice(0, Math.round(p.points));
      const probes: Point[] = [];
      for (let row = 0; row < 4; row++)
        for (let column = 0; column < 5; column++) probes.push([15 + column * 15.5, 20 + row * 20]);
      const nearestLinks: PathItem[] = [];
      const gDistances = points.map((point, index) => {
        let best = Infinity;
        let target = point;
        points.forEach((other, otherIndex) => {
          const distance = getDistance(point, other);
          if (otherIndex !== index && distance < best) {
            best = distance;
            target = other;
          }
        });
        nearestLinks.push(createPlotPath([point, target], 'cool', 0.4, 0.7, index / points.length));
        return best;
      });
      const fLinks: PathItem[] = [];
      const fDistances = probes.map((probe, index) => {
        let best = Infinity;
        let target = probe;
        points.forEach(point => {
          const distance = getDistance(probe, point);
          if (distance < best) {
            best = distance;
            target = point;
          }
        });
        fLinks.push(createPlotPath([probe, target], 'warm', 0.4, 0.7, index / probes.length));
        return best;
      });
      const maximum = Math.max(...gDistances, ...fDistances, 1);
      const curve = (distances: number[]): Point[] => {
        const sorted = [...distances].sort((first, second) => first - second);
        const path: Point[] = [[104, 84]];
        sorted.forEach((distance, index) => {
          path.push([104 + (distance / maximum) * 46, 84 - (index / sorted.length) * 64]);
          path.push([104 + (distance / maximum) * 46, 84 - ((index + 1) / sorted.length) * 64]);
        });
        return path;
      };
      const intensity = points.length / (78 * 72);
      return [
        getAxes(104, 150, 18, 84),
        createStrokes(
          [
            createPlotPath(
              getCurve(
                104,
                150,
                84,
                64,
                amount => 1 - Math.exp(-intensity * Math.PI * (amount * maximum) ** 2),
                30
              ),
              'dim',
              0.5,
              0.8
            )
          ],
          'static'
        ),
        createStrokes(fLinks, 'flow', 0.3, 0.8),
        createStrokes(nearestLinks, 'static'),
        createSprites(
          probes.map(([x, y]) => createDotAt(x, y, 1, 'warm', 0.8)),
          0
        ),
        createSprites(
          points.map(([x, y], index) => createDotAt(x, y, 1.5, 'primary', 0.95, index / 30)),
          0.3
        ),
        createStrokes(
          [
            createPlotPath(curve(gDistances), 'cool', 1, 0.95),
            createPlotPath(curve(fDistances), 'warm', 1, 0.95)
          ],
          'flow',
          0.3,
          1.4
        )
      ];
    }
  },

  // An 8 by 8 quadrat grid with counts, plus nearest-neighbor links.
  'point-pattern-indices': {
    parameters: [createSlider('clustering', 'Clustering', 0, 1, 0.05, 0.6)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('points');
      const uniform = createScatterPoints(fork('uniform'), 64, 6);
      const clustered = createClusterPoints(
        random,
        [
          [34, 30],
          [112, 66],
          [100, 24]
        ],
        22,
        7
      );
      const count = 64;
      const mixed = Array.from({length: count}, (_, index) =>
        index / count < p.clustering ? clustered[index] : uniform[index]
      );
      const columns = 8;
      const rows = 8;
      const cellWidth = 160 / columns;
      const cellHeight = 100 / rows;
      const counts = new Array<number>(columns * rows).fill(0);
      for (const [x, y] of mixed)
        counts[
          Math.min(rows - 1, Math.floor(y / cellHeight)) * columns +
            Math.min(columns - 1, Math.floor(x / cellWidth))
        ]++;
      const maximum = Math.max(...counts, 1);
      const cells: PolygonItem[] = [];
      counts.forEach((cellCount, index) => {
        if (cellCount === 0) return;
        const column = index % columns;
        const row = Math.floor(index / columns);
        cells.push({
          ring: createRectangleRing(
            column * cellWidth + 0.8,
            row * cellHeight + 0.8,
            (column + 1) * cellWidth - 0.8,
            (row + 1) * cellHeight - 0.8
          ),
          role: 'violet',
          alpha: 0.1 + 0.5 * (cellCount / maximum)
        });
      });
      const links = mixed.map((point, index) => {
        let best = Infinity;
        let target = point;
        mixed.forEach((other, otherIndex) => {
          const distance = getDistance(point, other);
          if (otherIndex !== index && distance < best) {
            best = distance;
            target = other;
          }
        });
        return createPlotPath([point, target], 'cool', 0.4, 0.8, index / count);
      });
      const grid: PathItem[] = [];
      for (let column = 1; column < columns; column++)
        grid.push(
          createPlotPath(
            [
              [column * cellWidth, 0],
              [column * cellWidth, 100]
            ],
            'dim',
            0.15,
            0.25
          )
        );
      for (let row = 1; row < rows; row++)
        grid.push(
          createPlotPath(
            [
              [0, row * cellHeight],
              [160, row * cellHeight]
            ],
            'dim',
            0.15,
            0.25
          )
        );
      return [
        createFills(cells, 'pulse'),
        createStrokes(grid, 'static'),
        createStrokes(links, 'flow', 0.25, 0.8),
        createSprites(
          mixed.map(([x, y], index) => createDotAt(x, y, 1.4, 'primary', 0.95, index / count)),
          0.3
        )
      ];
    }
  },

  // Two point types; links from a chosen type A point to type B points within the ring.
  'marked-point-statistics': {
    parameters: [createSlider('attraction', 'Attraction', 0, 1, 0.05, 0.7)],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('points');
      const centers: Point[] = [
        [26, 28],
        [66, 62],
        [34, 74],
        [78, 24]
      ];
      const typeA = createClusterPoints(random, centers, 6, 6).map(
        ([x, y]): Point => [clamp(x, 6, 94), y]
      );
      const scatter = createScatterPoints(fork('scatter'), typeA.length, 6).map(
        ([x, y]): Point => [6 + ((x - 6) / 148) * 88, y]
      );
      const typeB = typeA.map((point, index): Point => {
        const jitter = fork('jitter' + index);
        return [
          clamp(
            mixNumbersLocal(scatter[index][0], point[0] + jitter.gaussian() * 7, p.attraction),
            6,
            94
          ),
          clamp(
            mixNumbersLocal(scatter[index][1], point[1] + jitter.gaussian() * 7, p.attraction),
            6,
            94
          )
        ];
      });
      const focal: Point = pointer && pointer[0] < 98 ? pointer : typeA[8];
      let nearest = 0;
      typeA.forEach((point, index) => {
        if (getDistance(point, focal) < getDistance(typeA[nearest], focal)) nearest = index;
      });
      const center = typeA[nearest];
      const radius = 16;
      const links = typeB
        .filter(point => getDistance(point, center) <= radius)
        .map((point, index) => createPlotPath([center, point], 'violet', 0.6, 0.95, index / 6));
      const t = 0.5;
      const observed = (amount: number) =>
        clamp(amount * amount * (1 + 2.6 * p.attraction * (1 - amount) ** 1.5), 0, 1);
      const left = 112;
      const right = 152;
      return [
        getAxes(left, right, 24, 80),
        createStrokes(
          [
            createPlotPath(
              createCircle(center[0], center[1], radius, 32).concat([
                [center[0] + radius, center[1]]
              ]),
              'violet',
              0.6,
              0.8
            )
          ],
          'pulse',
          0.5
        ),
        createStrokes(links, 'flow', 0.5, 1),
        createSprites(
          [
            ...typeA.map(([x, y], index) => createDotAt(x, y, 1.6, 'cool', 0.9, index / 24)),
            ...typeB.map(([x, y], index) => createDotAt(x, y, 1.6, 'warm', 0.9, index / 24)),
            createDotAt(center[0], center[1], 3, 'cool', 0.6)
          ],
          0.3
        ),
        createStrokes(
          [
            createPlotPath(
              getCurve(left, right, 80, 48, amount => amount * amount, 24),
              'dim',
              0.5,
              0.8
            )
          ],
          'static'
        ),
        createStrokes(
          [createPlotPath(getCurve(left, right, 80, 48, observed, 24), 'violet', 1, 0.95)],
          'flow',
          0.3,
          1.4
        ),
        createSprites(
          [createDotAt(left + t * (right - left), 80 - observed(t) * 48, 1.8, 'violet')],
          0.4
        )
      ];
    }
  },

  // ---- Variograms ---------------------------------------------------------------------------

  // Empirical semivariance by lag: dots sized by pair count rising to a sill, nugget at the origin.
  variogram: {
    parameters: [createSlider('range', 'Range', 20, 95, 5, 50, '%')],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('bins');
      const range = p.range / 100;
      const nugget = 0.12;
      const bins = Array.from({length: 14}, (_, index) => {
        const lag = (index + 0.5) / 14;
        const noise = random.gaussian() * 0.05 * (1.3 - lag * 0.6);
        return {
          lag,
          gamma: getVariogramValue('spherical', lag, range, nugget, 1) + noise,
          count: 0.6 + random.next() * 1.4
        };
      });
      const rangeX = getPlotPoint(Math.min(range, 1), 0)[0];
      return [
        getAxes(PLOT.left, PLOT.right + 2, 12, PLOT.bottom),
        createStrokes(
          [
            createPlotPath([getPlotPoint(0, 1), getPlotPoint(1, 1)], 'warm', 0.5, 0.7),
            createPlotPath(
              [
                [rangeX, PLOT.bottom],
                [rangeX, getPlotPoint(0, 1)[1]]
              ],
              'cool',
              0.5,
              0.7
            )
          ],
          'pulse',
          0.3
        ),
        createStrokes(
          [
            createPlotPath(
              bins.map(bin => getPlotPoint(bin.lag, bin.gamma)),
              'primary',
              0.5,
              0.5
            )
          ],
          'flow',
          0.3,
          1.2
        ),
        createSprites(
          bins.map((bin, index) => {
            const [x, y] = getPlotPoint(bin.lag, bin.gamma);
            return createDotAt(x, y, 1.2 + bin.count * 0.8, 'primary', 0.9, index / 14);
          }),
          0.3
        ),
        createSprites([createDotAt(...getPlotPoint(0, nugget), 2.4, 'hot')], 0.5)
      ];
    }
  },

  // The same bins with three candidate models; the best fit by squared error is bright.
  'variogram-model-fit': {
    parameters: [
      createChoice('truth', 'Underlying model', ['spherical', 'exponential', 'gaussian']),
      createSlider('range', 'Range', 20, 95, 5, 55, '%')
    ],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('bins');
      const range = p.range / 100;
      const nugget = 0.1;
      const truth = VARIOGRAM_MODELS[Math.round(p.truth)];
      const bins = Array.from({length: 14}, (_, index) => {
        const lag = (index + 0.5) / 14;
        return {
          lag,
          gamma: getVariogramValue(truth, lag, range, nugget, 1) + random.gaussian() * 0.04
        };
      });
      const roles: Role[] = ['cool', 'violet', 'warm'];
      const errors = VARIOGRAM_MODELS.map(model =>
        bins.reduce(
          (sum, bin) =>
            sum + (getVariogramValue(model, bin.lag, range, nugget, 1) - bin.gamma) ** 2,
          0
        )
      );
      const best = errors.indexOf(Math.min(...errors));
      const curve = (model: VariogramModel) =>
        getCurve(
          PLOT.left,
          PLOT.right,
          PLOT.bottom,
          PLOT.height,
          amount => getVariogramValue(model, amount, range, nugget, 1),
          40
        );
      return [
        getAxes(PLOT.left, PLOT.right + 2, 12, PLOT.bottom),
        createStrokes(
          VARIOGRAM_MODELS.map((model, index) =>
            createPlotPath(
              curve(model),
              roles[index],
              index === best ? 1.4 : 0.6,
              index === best ? 1 : 0.4
            )
          ),
          'static'
        ),
        createStrokes(
          [createPlotPath(curve(VARIOGRAM_MODELS[best]), roles[best], 1.8, 1)],
          'flow',
          0.3,
          1.5
        ),
        createSprites(
          bins.map((bin, index) =>
            createDotAt(...getPlotPoint(bin.lag, bin.gamma), 1.5, 'primary', 0.95, index / 14)
          ),
          0.25
        )
      ];
    }
  },

  // Two direct variograms and the cross variogram between them, with a zero line.
  'cross-variogram': {
    parameters: [createSlider('correlation', 'Correlation', -1, 1, 0.05, 0.6)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('bins');
      const bins = Array.from({length: 14}, (_, index) => ({
        lag: (index + 0.5) / 14,
        noise: random.gaussian() * 0.03
      }));
      const gammaA = (lag: number) => getVariogramValue('spherical', lag, 0.55, 0.08, 0.9);
      const gammaB = (lag: number) => getVariogramValue('exponential', lag, 0.5, 0.1, 0.6);
      const cross = (lag: number) => p.correlation * Math.sqrt(gammaA(lag) * gammaB(lag));
      const base = 62;
      const scale = 56;
      const toPoint = (lag: number, value: number): Point => [
        PLOT.left + lag * (PLOT.right - PLOT.left),
        base - value * scale
      ];
      const curve = (evaluate: (lag: number) => number) =>
        Array.from({length: 41}, (_, index) => toPoint(index / 40, evaluate(index / 40)));
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [PLOT.left, 14],
                [PLOT.left, 90]
              ],
              'dim',
              0.5,
              0.6
            ),
            createPlotPath(
              [
                [PLOT.left, base],
                [PLOT.right + 2, base]
              ],
              'dim',
              0.5,
              0.8
            )
          ],
          'static'
        ),
        createStrokes(
          [
            createPlotPath(curve(gammaA), 'cool', 0.7, 0.6),
            createPlotPath(curve(gammaB), 'warm', 0.7, 0.6)
          ],
          'pulse',
          0.25
        ),
        createStrokes([createPlotPath(curve(cross), 'violet', 1.5, 1)], 'flow', 0.3, 1.6),
        createSprites(
          bins.map((bin, index) =>
            createDotAt(
              ...toPoint(bin.lag, cross(bin.lag) + bin.noise),
              1.4,
              'violet',
              0.95,
              index / 14
            )
          ),
          0.3
        )
      ];
    }
  },

  // Semivariance surface over space lag (right) and time lag (up); the probe reads one pair of lags.
  'space-time-variogram': {
    parameters: [
      createSlider('spaceRange', 'Space range', 0.3, 2, 0.05, 1),
      createSlider('timeRange', 'Time range', 0.3, 2, 0.05, 0.85)
    ],
    pointer: 'hover',
    build: ({p, pointer}) => {
      const values = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
        const spaceLag = u / p.spaceRange;
        const timeLag = (1 - v) / p.timeRange;
        return 0.1 + 0.9 * (1 - Math.exp(-2 * Math.hypot(spaceLag, timeLag * 0.9)));
      });
      const probe: Point = pointer ?? [92, 38];
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'heat',
          style: 'contours',
          levels: 14,
          strength: 0.05,
          speed: 0.4
        },
        createStrokes(
          [
            createPlotPath(
              [
                [probe[0], probe[1]],
                [probe[0], 98]
              ],
              'warm',
              0.5,
              0.8
            ),
            createPlotPath(
              [
                [probe[0], probe[1]],
                [0, probe[1]]
              ],
              'cool',
              0.5,
              0.8
            )
          ],
          'flow',
          0.4,
          1.2
        ),
        createSprites(
          [
            createDotAt(probe[0], probe[1], 2.6, 'ink'),
            createDotAt(probe[0], 96, 1.8, 'warm'),
            createDotAt(4, probe[1], 1.8, 'cool')
          ],
          0.5
        )
      ];
    }
  },

  // Moran's I per distance band: bars fall from positive through zero, with a significance band.
  'spatial-correlogram': {
    parameters: [createSlider('decay', 'Decay', 0.1, 1, 0.05, 0.45)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('bands');
      const count = 12;
      const zeroY = 56;
      const values = Array.from(
        {length: count},
        (_, index) =>
          0.9 * Math.exp(-index * p.decay) - 0.12 * Math.sin(index / 2) + random.gaussian() * 0.04
      );
      const step = 128 / count;
      const bars = values.map((value, index): PathItem => {
        const x = 18 + (index + 0.5) * step;
        return createPlotPath(
          [
            [x, zeroY],
            [x, zeroY - value * 42]
          ],
          Math.abs(value) < 0.1 ? 'dim' : value > 0 ? 'hot' : 'cool',
          step * 0.7,
          0.85,
          index / count
        );
      });
      return [
        createFills([
          {ring: createRectangleRing(16, zeroY - 5, 148, zeroY + 5), role: 'dim', alpha: 0.18}
        ]),
        createStrokes(
          [
            createPlotPath(
              [
                [16, zeroY],
                [148, zeroY]
              ],
              'dim',
              0.5,
              0.8
            ),
            createPlotPath(
              [
                [16, 12],
                [16, 90]
              ],
              'dim',
              0.5,
              0.6
            )
          ],
          'static'
        ),
        createStrokes(bars, 'pulse', 0.25),
        createStrokes(
          [
            createPlotPath(
              values.map((value, index): Point => [18 + (index + 0.5) * step, zeroY - value * 42]),
              'warm',
              0.9,
              0.95
            )
          ],
          'flow',
          0.3,
          1.4
        ),
        createSprites(
          values.map((value, index) =>
            createDotAt(
              18 + (index + 0.5) * step,
              zeroY - value * 42,
              1.6,
              'warm',
              0.95,
              index / count
            )
          ),
          0.3
        )
      ];
    }
  },

  // ---- Point clouds ---------------------------------------------------------------------------

  // Radius neighborhoods: three probe points light up the lidar points within a sphere.
  'pointcloud-neighborhoods': {
    parameters: [createSlider('neighborhood', 'Radius', 4, 20, 1, 12), ...SPIN_PARAMETERS],
    build: ({fork, p}) => {
      const points = createCloudPoints(fork('cloud'), 1300);
      const focalIndices = [
        points.findIndex(point => point.kind === 'ground' && point.x > -20 && point.x < 0),
        points.findIndex(point => point.kind === 'canopy'),
        points.findIndex(point => point.kind === 'roof')
      ];
      const focalRoles: Role[] = ['warm', 'violet', 'hot'];
      const items = points.map((point, index): SpriteItem => {
        let role: Role = 'primary';
        let alpha = 0.3;
        let radius = 0.7;
        focalIndices.forEach((focalIndex, order) => {
          const focal = points[focalIndex];
          if (index === focalIndex) {
            role = focalRoles[order];
            alpha = 1;
            radius = 3.6;
          } else if (
            alpha < 1 &&
            Math.hypot(point.x - focal.x, point.z - focal.z, point.height - focal.height) <=
              p.neighborhood
          ) {
            role = focalRoles[order];
            alpha = 0.95;
            radius = 1.1;
          }
        });
        return createCloudDot(point, point.height, role, radius, alpha, index / 1300);
      });
      return [createCloudLayer(items, p.depth, 0.2)];
    }
  },

  // Points coloured by local geometry: planar ground, vertical walls, scattered canopy.
  'pointcloud-features': {
    parameters: SPIN_PARAMETERS,
    build: ({fork, p}) => {
      const points = createCloudPoints(fork('cloud'), 1100);
      const roles: Record<CloudKind, Role> = {
        ground: 'cool',
        roof: 'primary',
        wall: 'violet',
        canopy: 'hot'
      };
      const edge = (point: CloudPoint) => point.kind === 'wall' && point.height > point.ground + 12;
      return [
        createCloudLayer(
          points.map((point, index) =>
            createCloudDot(
              point,
              point.height,
              edge(point) ? 'warm' : roles[point.kind],
              0.9,
              0.85,
              index / 1100
            )
          ),
          p.depth
        )
      ];
    }
  },

  // A dense surface with floating noise; noise past the threshold is rejected (pulsing hot).
  'pointcloud-statistical-filter': {
    parameters: [createSlider('ratio', 'Std ratio', 0.5, 3, 0.1, 1.2), ...SPIN_PARAMETERS],
    build: ({fork, p}) => {
      const surface = createCloudPoints(fork('cloud'), 800).filter(
        point => point.kind !== 'canopy'
      );
      const random = fork('noise');
      const items: SpriteItem[] = surface.map((point, index) =>
        createCloudDot(
          point,
          point.height,
          point.kind === 'ground' ? 'cool' : 'primary',
          0.8,
          0.8,
          index / 800
        )
      );
      for (let index = 0; index < 90; index++) {
        const x = random.range(-60, 60);
        const z = random.range(-40, 40);
        const isolation = random.range(1, 32);
        const point = {x, z};
        const removed = isolation > 3 + 5 * p.ratio;
        items.push(
          createCloudDot(
            point,
            getTerrainHeight(x, z) + isolation,
            removed ? 'hot' : 'warm',
            removed ? 1.7 : 1,
            removed ? 1 : 0.7,
            random.next()
          )
        );
      }
      return [createCloudLayer(items, p.depth, 0.35)];
    }
  },

  // Tiles of points each reduce to voxel centroids; voxels spanning the seam (hot) merge both tiles.
  'pointcloud-voxel-merge': {
    parameters: [createSlider('voxel', 'Voxel size', 4, 18, 1, 9), ...SPIN_PARAMETERS],
    build: ({fork, p}) => {
      const points = createCloudPoints(fork('cloud'), 1000);
      const voxel = Math.round(p.voxel);
      const cells = new Map<
        number,
        {x: number; z: number; height: number; count: number; tiles: number}
      >();
      points.forEach(point => {
        const key =
          (Math.floor((point.x + 100) / voxel) * 1000 + Math.floor((point.height + 50) / voxel)) *
            1000 +
          Math.floor((point.z + 100) / voxel);
        const tiles = point.x < -6 ? 1 : point.x > 6 ? 2 : 3;
        const cell = cells.get(key) ?? {x: 0, z: 0, height: 0, count: 0, tiles: 0};
        cell.x += point.x;
        cell.z += point.z;
        cell.height += point.height;
        cell.count++;
        cell.tiles |= tiles;
        cells.set(key, cell);
      });
      const items: SpriteItem[] = points.map((point, index) =>
        createCloudDot(point, point.height, point.x < 0 ? 'cool' : 'warm', 0.55, 0.28, index / 1000)
      );
      cells.forEach(cell => {
        const merged = cell.tiles === 3;
        items.push(
          createCloudDot(
            {x: cell.x / cell.count, z: cell.z / cell.count},
            cell.height / cell.count,
            merged ? 'hot' : 'primary',
            merged ? 2.4 : 1.8,
            0.95,
            0.5
          )
        );
      });
      return [createCloudLayer(items, p.depth, 0.3)];
    }
  },

  // Top-down ground and canopy surfaces rasterized from the points.
  'pointcloud-surfaces': {
    parameters: [createSlider('canopy', 'Canopy', 0, 1, 0.05, 0.75)],
    pointer: 'none',
    build: ({fork, p}) => {
      const ground = createSmoothField(fork('ground'), 4, 0.9);
      const random = fork('trees');
      const getGround = (u: number, v: number) =>
        0.1 + 0.5 * sampleField(ground, FIELD_COLUMNS, FIELD_ROWS, u, v);
      // Trees in panel space (u, v in 0..1): the first four sit on the profile line v = 0.5.
      const trees = Array.from({length: 13}, (_, index) => ({
        u: index < 4 ? 0.12 + index * 0.25 + random.range(-0.05, 0.05) : random.range(0.06, 0.94),
        v: index < 4 ? 0.5 + random.range(-0.04, 0.04) : random.range(0.08, 0.92),
        size: random.range(0.07, 0.12),
        rise: random.range(0.55, 1) * p.canopy
      }));
      const getTreeHeight = (u: number, v: number) => {
        let height = 0;
        for (const tree of trees) {
          const distance = Math.hypot(u - tree.u, v - tree.v);
          height = Math.max(height, tree.rise * Math.exp(-((distance / tree.size) ** 2)));
        }
        return height;
      };
      // Three small cell rasters: DSM (cell maximum) minus DTM (cell minimum) gives canopy height.
      const cells = 14;
      const cellSize = 3;
      const panelSize = cells * cellSize;
      const panelLeft = [9, 59, 109];
      const panelTop = 8;
      const dsmItems: PolygonItem[] = [];
      const dtmItems: PolygonItem[] = [];
      const chmItems: PolygonItem[] = [];
      for (let row = 0; row < cells; row++) {
        for (let column = 0; column < cells; column++) {
          const u = (column + 0.5) / cells;
          const v = (row + 0.5) / cells;
          const base = getGround(u, v);
          const height = getTreeHeight(u, v);
          const phase = (row * cells + column) / (cells * cells);
          const top = panelTop + row * cellSize + 0.2;
          const bottom = top + cellSize - 0.4;
          const addCell = (items: PolygonItem[], panel: number, role: Role, alpha: number) => {
            const left = panelLeft[panel] + column * cellSize + 0.2;
            items.push({
              ring: createRectangleRing(left, top, left + cellSize - 0.4, bottom),
              role,
              alpha,
              phase
            });
          };
          const surface = clamp(base + height * 0.5, 0, 1);
          addCell(dsmItems, 0, getClassRole(Math.round(surface * 4), 5), 0.45 + 0.5 * surface);
          addCell(
            dtmItems,
            1,
            getClassRole(Math.round(clamp(base, 0, 1) * 4), 5),
            0.4 + 0.5 * base
          );
          if (height > 0.04) {
            addCell(
              chmItems,
              2,
              height > 0.55 ? 'hot' : height > 0.3 ? 'warm' : 'primary',
              0.5 + 0.5 * clamp(height, 0, 1)
            );
          } else {
            addCell(chmItems, 2, 'dim', 0.35);
          }
        }
      }
      const panelOutlines: PathItem[] = panelLeft.map(left =>
        createPlotPath(
          closeRing(
            createRectangleRing(
              left - 0.6,
              panelTop - 0.6,
              left + panelSize + 0.6,
              panelTop + panelSize + 0.6
            )
          ),
          'dim',
          0.4,
          0.7
        )
      );
      // Minus between DSM and DTM, equals before the canopy height panel.
      const middleY = panelTop + panelSize / 2;
      const operators: PathItem[] = [
        createPlotPath(
          [
            [53.6, middleY],
            [57.6, middleY]
          ],
          'ink',
          0.9,
          0.95
        ),
        createPlotPath(
          [
            [104.2, middleY - 1.6],
            [108.2, middleY - 1.6]
          ],
          'ink',
          0.9,
          0.95
        ),
        createPlotPath(
          [
            [104.2, middleY + 1.6],
            [108.2, middleY + 1.6]
          ],
          'ink',
          0.9,
          0.95
        )
      ];
      // Cross-section below: ground points on the terrain line, canopy points above it.
      const baseline = 94;
      const profileLeft = 9;
      const profileWidth = 142;
      const groundScale = 16;
      const treeScale = 26;
      const samples = 72;
      const terrainLine: Point[] = [];
      const surfaceLine: Point[] = [];
      for (let index = 0; index <= samples; index++) {
        const u = index / samples;
        const x = profileLeft + u * profileWidth;
        const base = getGround(u, 0.5);
        terrainLine.push([x, baseline - base * groundScale]);
        surfaceLine.push([x, baseline - base * groundScale - getTreeHeight(u, 0.5) * treeScale]);
      }
      const pointRandom = fork('points');
      const groundPoints: SpriteItem[] = [];
      const canopyPoints: SpriteItem[] = [];
      for (let index = 0; index < 90; index++) {
        const u = pointRandom.next();
        const x = profileLeft + u * profileWidth;
        const base = getGround(u, 0.5);
        groundPoints.push(
          createDotAt(
            x,
            baseline - base * groundScale + pointRandom.range(0.2, 1.2),
            0.8,
            'cool',
            0.85,
            pointRandom.next()
          )
        );
        const height = getTreeHeight(u, 0.5);
        if (height > 0.06) {
          const lift = pointRandom.range(0.3, 1);
          canopyPoints.push(
            createDotAt(
              x,
              baseline - base * groundScale - height * treeScale * lift,
              0.9,
              lift > 0.85 ? 'hot' : 'warm',
              0.9,
              pointRandom.next()
            )
          );
        }
      }
      // Vertical brackets at the tallest crowns: surface minus terrain is the canopy height.
      const brackets: PathItem[] = [];
      for (let index = 0; index <= samples; index++) {
        const [x, surfaceY] = surfaceLine[index];
        const terrainY = terrainLine[index][1];
        const isPeak =
          terrainY - surfaceY > 8 &&
          surfaceY <= (surfaceLine[index - 1]?.[1] ?? Infinity) &&
          surfaceY <= (surfaceLine[index + 1]?.[1] ?? Infinity);
        if (isPeak)
          brackets.push(
            createPlotPath(
              [
                [x, surfaceY],
                [x, terrainY]
              ],
              'violet',
              0.7,
              0.9,
              index / samples
            )
          );
      }
      return [
        createFills(dsmItems, 'pulse'),
        createFills(dtmItems, 'pulse'),
        createFills(chmItems, 'pulse'),
        createStrokes([...panelOutlines, ...operators]),
        createStrokes([createPlotPath(terrainLine, 'cool', 0.9, 0.9)]),
        createStrokes([createPlotPath(surfaceLine, 'hot', 1.1, 0.95)], 'flow', 0.25, 1.4),
        createStrokes(brackets, 'pulse', 0.5),
        createSprites(groundPoints, 0.3),
        createSprites(canopyPoints, 0.4)
      ];
    }
  },

  // Points lifted off a terrain by their height above ground, as stems coloured by height.
  'pointcloud-height': {
    parameters: [createSlider('scale', 'Height scale', 0.4, 1.6, 0.05, 1)],
    pointer: 'none',
    build: ({fork, p}) => {
      const ground = createSmoothField(fork('ground'), 5, 1.1);
      const random = fork('points');
      const stems: PathItem[] = [];
      const tops: SpriteItem[] = [];
      const feet: SpriteItem[] = [];
      for (let index = 0; index < 30; index++) {
        const x = random.range(10, 150);
        const y = random.range(30, 94);
        const rise = random.range(3, 26) * p.scale;
        const amount = clamp(rise / 26, 0, 1);
        const role: Role = amount > 0.66 ? 'hot' : amount > 0.33 ? 'warm' : 'cool';
        stems.push(
          createPlotPath(
            [
              [x, y],
              [x, y - rise]
            ],
            role,
            0.7,
            0.9,
            random.next()
          )
        );
        tops.push(createDotAt(x, y - rise, 1.8, role, 0.95, random.next()));
        feet.push(createDotAt(x, y, 1, 'dim', 0.6));
      }
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: ground,
          ramp: 'terrain',
          style: 'ramp',
          strength: 0.03,
          speed: 0.3
        },
        createSprites(feet, 0),
        createStrokes(stems, 'flow', 0.4, 1),
        createSprites(tops, 0.4)
      ];
    }
  },

  // Ground classification: points within a tolerance of the local terrain are ground.
  'pointcloud-ground': {
    parameters: [
      createSlider('tolerance', 'Height tolerance', 0.2, 6, 0.2, 1.5),
      ...SPIN_PARAMETERS
    ],
    build: ({fork, p}) => {
      const points = createCloudPoints(fork('cloud'), 1200);
      return [
        createCloudLayer(
          points.map((point, index) => {
            const isGround = point.height - point.ground <= p.tolerance;
            return createCloudDot(
              point,
              point.height,
              isGround ? 'cool' : 'warm',
              isGround ? 0.85 : 1,
              isGround ? 0.7 : 0.95,
              index / 1200
            );
          }),
          p.depth
        )
      ];
    }
  },

  // SMRF in profile: an opening of the lowest returns builds the ground line under trees and a roof.
  'pointcloud-smrf': {
    parameters: [createSlider('window', 'Window', 3, 26, 1, 14, 'cells')],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const random = fork('profile');
      const columns = 80;
      const terrain = (index: number) =>
        14 + 7 * Math.sin(index / 12) + 3 * Math.sin(index / 5) * 0.4 + index * 0.05;
      const buildingHeight = 16;
      const isBuilding = (index: number) => index >= 48 && index < 58;
      const isTree = (index: number) => (index >= 14 && index < 24) || (index >= 64 && index < 73);
      const lowest: number[] = [];
      const items: SpriteItem[] = [];
      for (let index = 0; index < columns; index++) {
        const x = 12 + (index / (columns - 1)) * 136;
        const base = terrain(index);
        if (isBuilding(index)) {
          lowest.push(base + buildingHeight);
          items.push(createDotAt(x, 88 - base - buildingHeight, 1.1, 'warm', 0.95, random.next()));
          items.push(
            createDotAt(
              x,
              88 - base - buildingHeight + random.range(0, 3),
              0.8,
              'warm',
              0.6,
              random.next()
            )
          );
        } else {
          lowest.push(base);
          if (index % 2 === 0 || !isTree(index))
            items.push(createDotAt(x, 88 - base, 1.1, 'cool', 0.9, random.next()));
        }
        if (isTree(index)) {
          for (let leaf = 0; leaf < 3; leaf++)
            items.push(
              createDotAt(
                x + random.range(-0.6, 0.6),
                88 - base - random.range(5, 14),
                1,
                'warm',
                0.85,
                random.next()
              )
            );
        }
      }
      const half = Math.max(1, Math.round(p.window / 2));
      const erode = lowest.map((_, index) =>
        Math.min(...lowest.slice(Math.max(0, index - half), index + half + 1))
      );
      const opened = erode.map((_, index) =>
        Math.max(...erode.slice(Math.max(0, index - half), index + half + 1))
      );
      const line = opened.map(
        (value, index): Point => [12 + (index / (columns - 1)) * 136, 88 - value]
      );
      const center = pointer ? clamp(pointer[0], 12 + half * 1.7, 148 - half * 1.7) : 80;
      const halfWidth = half * 1.72;
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [12, 88],
                [148, 88]
              ],
              'dim',
              0.3,
              0.4
            )
          ],
          'static'
        ),
        createFills(
          [
            {
              ring: createRectangleRing(center - halfWidth, 8, center + halfWidth, 92),
              role: 'violet',
              alpha: 0.1
            }
          ],
          'pulse'
        ),
        createStrokes(
          [
            createPlotPath(
              [
                [center - halfWidth, 8],
                [center - halfWidth, 92]
              ],
              'violet',
              0.5,
              0.7
            ),
            createPlotPath(
              [
                [center + halfWidth, 8],
                [center + halfWidth, 92]
              ],
              'violet',
              0.5,
              0.7
            )
          ],
          'pulse',
          0.5
        ),
        createSprites(items, 0.25),
        createStrokes([createPlotPath(line, 'hot', 1.4, 1)], 'flow', 0.3, 1.8)
      ];
    }
  },

  // ICP: the warm scan converges onto the cool reference as iterations increase.
  'pointcloud-icp': {
    parameters: [createSlider('iteration', 'Iteration', 0, 10, 1, 3), ...SPIN_PARAMETERS],
    build: ({fork, p}) => {
      const reference = createCloudPoints(fork('cloud'), 520).filter(
        point => point.kind !== 'canopy'
      );
      const noise = fork('noise');
      const remaining = 0.7 ** p.iteration;
      const angle = 0.45 * remaining;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const items: SpriteItem[] = [];
      reference.forEach((point, index) => {
        items.push(createCloudDot(point, point.height, 'cool', 0.9, 0.85, index / 520));
        const jitter = noise.gaussian() * 0.25;
        const moved = {
          x: point.x * cosine + point.z * sine + 18 * remaining,
          z: -point.x * sine + point.z * cosine + 10 * remaining
        };
        items.push(
          createCloudDot(
            moved,
            point.height + 6 * remaining + jitter,
            'warm',
            0.9,
            0.85,
            index / 520
          )
        );
      });
      return [createCloudLayer(items, p.depth, 0.2)];
    }
  },

  // Change detection: signed change along the surface, hot where it grew and cool where it was cut.
  'pointcloud-change': {
    parameters: [createSlider('threshold', 'Min change', 0.5, 6, 0.25, 2), ...SPIN_PARAMETERS],
    build: ({fork, p}) => {
      const points = createCloudPoints(fork('cloud'), 1100).filter(
        point => point.kind === 'ground' || point.kind === 'roof'
      );
      return [
        createCloudLayer(
          points.map((point, index) => {
            const delta =
              8 * Math.exp(-(((point.x - 20) ** 2 + (point.z + 14) ** 2) / 220)) -
              7 * Math.exp(-(((point.x + 30) ** 2 + (point.z - 8) ** 2) / 260));
            const changed = Math.abs(delta) >= p.threshold;
            return createCloudDot(
              point,
              point.height + delta,
              changed ? (delta > 0 ? 'hot' : 'cool') : 'primary',
              changed ? 1.1 : 0.7,
              changed ? 0.95 : 0.35,
              index / 1100
            );
          }),
          p.depth
        )
      ];
    }
  },

  // A cloud with a huge world offset (quantised, jagged) recentered to a clean local frame.
  'pointcloud-local-positions': {
    parameters: [createSlider('offset', 'Precision loss', 1, 9, 0.5, 4)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('cloud');
      const surface = (amount: number) =>
        46 + 9 * Math.sin(amount * 8) + 5 * Math.sin(amount * 3 + 1);
      const quantised: SpriteItem[] = [];
      const clean: SpriteItem[] = [];
      for (let index = 0; index < 90; index++) {
        const amount = random.next();
        const jitter = random.gaussian() * 2.2;
        const x = amount * 48;
        const y = surface(amount) + jitter;
        clean.push(createDotAt(104 + x - 24, y, 1.1, 'primary', 0.9, random.next()));
        quantised.push(
          createDotAt(
            10 + Math.round(x / p.offset) * p.offset,
            Math.round(y / p.offset) * p.offset,
            1.1,
            'hot',
            0.85,
            random.next()
          )
        );
      }
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [10, 94],
                [10, 90]
              ],
              'dim',
              0.3,
              0.3
            ),
            createPlotPath(
              [
                [82, 50],
                [78, 50]
              ],
              'dim',
              0.3,
              0.3
            )
          ],
          'static'
        ),
        createStrokes(
          [createPlotPath(createArc([60, 30], [100, 30], -0.3, 18), 'warm', 1, 0.95)],
          'flow',
          0.4,
          1.4
        ),
        createStrokes(
          [
            createPlotPath(
              [
                [6, 96],
                [6, 70],
                [10, 70]
              ],
              'violet',
              0.5,
              0.6
            ),
            createPlotPath(
              [
                [2, 92],
                [12, 92]
              ],
              'violet',
              0.5,
              0.6
            )
          ],
          'pulse',
          0.4
        ),
        createSprites(quantised, 0.3),
        createSprites(clean, 0.3),
        createSprites([createDotAt(104, 20, 2.4, 'violet', 0.9)], 0.5)
      ];
    }
  },

  // A shift grid: old lattice nodes (dim) move to datum-shifted positions (warm); a probe reads one.
  'datum-grid-transform': {
    parameters: [createSlider('magnify', 'Shift scale', 1, 9, 0.5, 4.5, '×')],
    pointer: 'hover',
    build: ({p, pointer}) => {
      const shift = (x: number, y: number): Point => [
        (Math.sin(y / 22 + 0.5) + 0.5 * Math.cos(x / 30)) * p.magnify,
        (0.8 * Math.cos(x / 26) + 0.4 * Math.sin(y / 18 + 1)) * p.magnify
      ];
      const values = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
        const [dx, dy] = shift(u * 160, v * 100);
        return clamp(Math.hypot(dx, dy) / (2.2 * p.magnify), 0, 1);
      });
      const oldNodes: SpriteItem[] = [];
      const newNodes: SpriteItem[] = [];
      const links: PathItem[] = [];
      for (let row = 0; row < 6; row++) {
        for (let column = 0; column < 9; column++) {
          const x = 14 + column * 16.5;
          const y = 12 + row * 15.2;
          const [dx, dy] = shift(x, y);
          oldNodes.push(createDotAt(x, y, 0.9, 'dim', 0.8));
          newNodes.push(createDotAt(x + dx, y + dy, 1.3, 'warm', 0.95, (row * 9 + column) / 54));
          links.push(
            createPlotPath(
              [
                [x, y],
                [x + dx, y + dy]
              ],
              'warm',
              0.6,
              0.8
            )
          );
        }
      }
      const probe: Point = pointer ?? [92, 52];
      const [probeX, probeY] = shift(probe[0], probe[1]);
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'ice',
          style: 'ramp',
          strength: 0.05,
          speed: 0.4
        },
        createStrokes(links, 'flow', 0.3, 1.2),
        createSprites(oldNodes, 0),
        createSprites(newNodes, 0.3),
        createStrokes(
          [createPlotPath([probe, [probe[0] + probeX, probe[1] + probeY]], 'hot', 1, 1)],
          'pulse',
          0.6
        ),
        createSprites(
          [
            createDotAt(probe[0], probe[1], 2, 'ink'),
            createDotAt(probe[0] + probeX, probe[1] + probeY, 2.6, 'hot')
          ],
          0.5
        )
      ];
    }
  },

  // A circle of one radius around every point, with the radius spoke on the first.
  'point-buffer': {
    parameters: [createSlider('distance', 'Distance', 2, 24, 1, 12, 'px')],
    pointer: 'hover',
    build: ({fork, p, pointer}) => {
      const points = createScatterPoints(fork('points'), 11, 18);
      const all: Point[] = pointer ? [...points, pointer] : points;
      const circles = all.map(([x, y]) => createCircle(x, y, p.distance, 28));
      const first = all[0];
      return [
        createFills(
          circles.map(ring => ({ring, role: 'cool', alpha: 0.2})),
          'pulse'
        ),
        createStrokes(
          circles.map((ring, index) =>
            createPlotPath(
              closeRing(ring),
              index === all.length - 1 && pointer ? 'warm' : 'primary',
              0.55,
              0.9,
              index / all.length
            )
          ),
          'draw',
          0.2
        ),
        createStrokes(
          [createPlotPath([first, [first[0] + p.distance, first[1]]], 'warm', 0.9, 1)],
          'flow',
          0.5,
          1
        ),
        createSprites(
          all.map(([x, y], index) =>
            createDotAt(
              x,
              y,
              1.7,
              index === all.length - 1 && pointer ? 'warm' : 'ink',
              1,
              index / all.length
            )
          ),
          0.3
        )
      ];
    }
  },

  // ---- Computation records -------------------------------------------------------------------

  // A command graph grows node by node as it is recorded; the newest node blinks like a REC light.
  'computation-recorder': {
    parameters: [createSlider('nodes', 'Recorded nodes', 2, 10, 1, 7)],
    pointer: 'none',
    build: ({p}) => {
      const lanes = [50, 50, 30, 70, 50, 30, 70, 50, 50, 50];
      const parents: number[][] = [[], [0], [1], [1], [2, 3], [4], [4], [5, 6], [7], [8]];
      const count = Math.round(p.nodes);
      const position = (index: number): Point => [16 + index * 14.5, lanes[index]];
      const edges: PathItem[] = [];
      const ghosts: SpriteItem[] = [];
      const recorded: SpriteItem[] = [];
      for (let index = 0; index < 10; index++) {
        const [x, y] = position(index);
        if (index >= count) {
          ghosts.push(createDotAt(x, y, 1.3, 'dim', 0.6));
          continue;
        }
        parents[index].forEach(parent =>
          edges.push(createPlotPath([position(parent), [x, y]], 'primary', 0.7, 0.85, index / 10))
        );
        recorded.push(
          createDotAt(
            x,
            y,
            index === count - 1 ? 3 : 2.1,
            index === count - 1 ? 'hot' : 'primary',
            0.95,
            index / 10
          )
        );
      }
      const newest = parents[count - 1].map(parent =>
        createPlotPath([position(parent), position(count - 1)], 'hot', 1.1, 1)
      );
      return [
        createSprites(ghosts, 0),
        createStrokes(edges, 'static'),
        createStrokes(newest, 'draw', 0.5),
        createSprites(recorded, 0.35),
        createSprites([createDotAt(148, 10, 3, 'hot', 1)], 0.8)
      ];
    }
  },

  // One record frames the graph, its adapter and the data version together.
  'computation-record': {
    parameters: [createSlider('version', 'Data version', 1, 6, 1, 3)],
    pointer: 'none',
    build: ({p}) => {
      const frame = closeRing(createRectangleRing(8, 10, 152, 90));
      const graphNodes: Point[] = [
        [22, 36],
        [34, 50],
        [22, 64],
        [46, 36],
        [46, 64]
      ];
      const graphEdges: Point[][] = [
        [graphNodes[0], graphNodes[1]],
        [graphNodes[2], graphNodes[1]],
        [graphNodes[1], graphNodes[3]],
        [graphNodes[1], graphNodes[4]]
      ];
      const pins: PathItem[] = [];
      for (let index = 0; index < 4; index++) {
        const offset = 70 + index * 7 + 2;
        pins.push(
          createPlotPath(
            [
              [offset, 38],
              [offset, 42]
            ],
            'warm',
            0.6,
            0.9
          ),
          createPlotPath(
            [
              [offset, 58],
              [offset, 62]
            ],
            'warm',
            0.6,
            0.9
          )
        );
      }
      const layers = Math.round(p.version);
      const stack: PathItem[] = [];
      for (let index = 0; index < layers; index++)
        stack.push(
          createPlotPath(
            [
              [112, 66 - index * 7],
              [136, 66 - index * 7]
            ],
            index === layers - 1 ? 'hot' : 'cool',
            4,
            index === layers - 1 ? 1 : 0.55,
            index / layers
          )
        );
      return [
        createStrokes([createPlotPath(frame, 'violet', 0.8, 0.8)], 'pulse', 0.25),
        createStrokes(
          graphEdges.map(points => createPlotPath(points, 'primary', 0.6, 0.85)),
          'static'
        ),
        createSprites(
          graphNodes.map(([x, y], index) => createDotAt(x, y, 2, 'primary', 0.95, index / 5)),
          0.3
        ),
        createStrokes(
          [createPlotPath(closeRing(createRectangleRing(66, 36, 98, 64)), 'warm', 0.9, 0.95)],
          'static'
        ),
        createStrokes(pins, 'static'),
        createFills(
          [{ring: createRectangleRing(72, 42, 92, 58), role: 'warm', alpha: 0.25}],
          'pulse'
        ),
        createStrokes(stack, 'pulse', 0.35),
        createStrokes(
          [
            createPlotPath(
              [
                [50, 50],
                [66, 50]
              ],
              'primary',
              0.8,
              0.9
            ),
            createPlotPath(
              [
                [98, 50],
                [112, 50],
                [124, 50 - 0]
              ],
              'warm',
              0.8,
              0.9
            )
          ],
          'flow',
          0.4,
          1
        ),
        createSprites([createDotAt(124, 26, 1.6, 'cool', 0.8)], 0.4)
      ];
    }
  },

  // Node durations on a GPU timeline, with the short CPU encode time above each node.
  'computation-metrics': {
    parameters: [createSlider('nodes', 'Nodes', 4, 12, 1, 8)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('durations');
      const durations = Array.from({length: 12}, () => 0.4 + random.next() ** 2 * 3);
      const count = Math.round(p.nodes);
      const used = durations.slice(0, count);
      const total = used.reduce((sum, value) => sum + value, 0);
      const maximum = Math.max(...used);
      const gpu: PathItem[] = [];
      const cpu: PathItem[] = [];
      const marks: SpriteItem[] = [];
      let cursor = 14;
      used.forEach((duration, index) => {
        const width = (duration / total) * 126;
        const share = duration / maximum;
        gpu.push(
          createPlotPath(
            [
              [cursor + 0.6, 60],
              [cursor + width - 0.6, 60]
            ],
            share > 0.75 ? 'hot' : share > 0.45 ? 'warm' : share > 0.2 ? 'primary' : 'cool',
            8,
            0.9,
            index / count
          )
        );
        cpu.push(
          createPlotPath(
            [
              [cursor + 0.6, 34],
              [cursor + 0.6 + Math.min(width * 0.35, 4), 34]
            ],
            'violet',
            3,
            0.7,
            index / count
          )
        );
        marks.push(createDotAt(cursor, 76, 0.8, 'dim', 0.7));
        cursor += width;
      });
      marks.push(createDotAt(cursor, 76, 0.8, 'dim', 0.7));
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [14, 76],
                [140, 76]
              ],
              'dim',
              0.3,
              0.5
            ),
            createPlotPath(
              [
                [14, 34],
                [140, 34]
              ],
              'dim',
              0.3,
              0.3
            )
          ],
          'static'
        ),
        createStrokes(cpu, 'static'),
        createStrokes(gpu, 'pulse', 0.3),
        createStrokes(
          [
            createPlotPath(
              [
                [14, 60],
                [140, 60]
              ],
              'ink',
              1.2,
              0.8
            )
          ],
          'flow',
          0.25,
          0.6
        ),
        createSprites(marks, 0)
      ];
    }
  },

  // Snapshot frames hang off the nodes of the run that produced them.
  'computation-captures': {
    parameters: [createSlider('captures', 'Captures', 1, 6, 1, 3)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('frames');
      const count = Math.round(p.captures);
      const columns = 6;
      const nodes: SpriteItem[] = [];
      const hangers: PathItem[] = [];
      const frameOutlines: PathItem[] = [];
      const pixels: SpriteItem[] = [];
      for (let index = 0; index < columns; index++) {
        const x = 20 + index * 24;
        const captured = index < count;
        nodes.push(
          createDotAt(
            x,
            24,
            captured ? 2.6 : 2,
            captured ? 'warm' : 'primary',
            0.95,
            index / columns
          )
        );
        if (!captured) continue;
        hangers.push(
          createPlotPath(
            [
              [x, 27],
              [x, 50]
            ],
            'warm',
            0.6,
            0.9,
            index / columns
          )
        );
        frameOutlines.push(
          createPlotPath(closeRing(createRectangleRing(x - 8, 50, x + 8, 70)), 'violet', 0.7, 0.9)
        );
        for (let row = 0; row < 4; row++) {
          for (let column = 0; column < 3; column++) {
            pixels.push(
              createDotAt(
                x - 5 + column * 5,
                54 + row * 4.4,
                1.3,
                getRoleCycle(index + 1),
                0.15 + 0.8 * random.next(),
                random.next()
              )
            );
          }
        }
      }
      return [
        createStrokes(
          [
            createPlotPath(
              [
                [10, 24],
                [150, 24]
              ],
              'primary',
              0.8,
              0.7
            )
          ],
          'flow',
          0.3,
          1.2
        ),
        createStrokes(hangers, 'flow', 0.4, 0.8),
        createStrokes(frameOutlines, 'static'),
        createSprites(pixels, 0.35),
        createSprites(nodes, 0.3)
      ];
    }
  },

  // One arena buffer divided into slots; frames from the graph nodes fill the slots in order.
  'snapshot-arena': {
    parameters: [createSlider('frames', 'Frames stored', 1, 12, 1, 7)],
    pointer: 'none',
    build: ({p}) => {
      const slotCount = 12;
      const filled = Math.round(p.frames);
      const slotWidth = 128 / slotCount;
      const chain: Point[] = Array.from({length: 6}, (_, index): Point => [24 + index * 22, 22]);
      const slots: PolygonItem[] = [];
      const lines: PathItem[] = [];
      const links: PathItem[] = [];
      for (let slot = 0; slot < slotCount; slot++) {
        const left = 16 + slot * slotWidth;
        const ring = createRectangleRing(left + 0.6, 62, left + slotWidth - 0.6, 82);
        lines.push(
          createPlotPath(
            closeRing(ring),
            slot < filled ? 'warm' : 'dim',
            0.5,
            slot < filled ? 0.9 : 0.4
          )
        );
        if (slot < filled) {
          slots.push({ring, role: getRoleCycle(slot), alpha: 0.4, phase: slot / slotCount});
          const source = chain[slot % chain.length];
          links.push(
            createPlotPath(
              [
                [source[0], 26],
                [source[0], 40],
                [left + slotWidth / 2, 48],
                [left + slotWidth / 2, 62]
              ],
              getRoleCycle(slot),
              0.4,
              0.55,
              slot / slotCount
            )
          );
        }
      }
      return [
        createFills(slots, 'pulse'),
        createStrokes(lines, 'static'),
        createStrokes(links, 'flow', 0.3, 0.7),
        createStrokes(
          chain
            .slice(1)
            .map((point, index) => createPlotPath([chain[index], point], 'primary', 0.7, 0.85)),
          'static'
        ),
        createSprites(
          chain.map(([x, y], index) => createDotAt(x, y, 2.2, 'primary', 0.95, index / 6)),
          0.3
        ),
        createSprites(
          [createDotAt(16 + Math.min(filled, slotCount) * slotWidth, 88, 1.4, 'hot')],
          0.7
        )
      ];
    }
  },

  // A tap on the edge between two nodes copies a view off the main flow into a captured square.
  'snapshot-tap': {
    parameters: [createSlider('afterNode', 'Tap after node', 0, 2, 1, 1)],
    pointer: 'none',
    build: ({fork, p}) => {
      const random = fork('view');
      const nodes: Point[] = [
        [22, 32],
        [62, 32],
        [102, 32],
        [142, 32]
      ];
      const index = Math.round(p.afterNode);
      const tapX = (nodes[index][0] + nodes[index + 1][0]) / 2;
      const view = createRectangleRing(tapX - 12, 62, tapX + 12, 86);
      const pixels: SpriteItem[] = [];
      for (let row = 0; row < 4; row++)
        for (let column = 0; column < 5; column++)
          pixels.push(
            createDotAt(
              tapX - 8 + column * 4,
              67 + row * 5,
              1.5,
              'cool',
              0.2 + 0.8 * random.next(),
              random.next()
            )
          );
      return [
        createStrokes([createPlotPath(nodes, 'primary', 0.9, 0.7)], 'flow', 0.35, 1.4),
        createStrokes(
          [
            createPlotPath(
              [
                [tapX, 32],
                [tapX, 62]
              ],
              'violet',
              1,
              0.95
            )
          ],
          'flow',
          0.5,
          0.9
        ),
        createStrokes([createPlotPath(closeRing(view), 'violet', 0.8, 0.9)], 'static'),
        createSprites(pixels, 0.3),
        createSprites(
          [
            ...nodes.map(([x, y], order) =>
              createDotAt(
                x,
                y,
                order === index || order === index + 1 ? 2.8 : 2,
                order === index ? 'warm' : 'primary',
                0.95,
                order / 4
              )
            ),
            createDotAt(tapX, 32, 2.6, 'violet', 1)
          ],
          0.35
        )
      ];
    }
  },

  // A selection mask carried across a relation: selected child rows select their parent rows.
  'transfer-mask': {
    parameters: [createSlider('width', 'Selection', 8, 60, 2, 26, 'px')],
    pointer: 'sticky',
    build: ({fork, p, pointer}) => {
      const random = fork('relation');
      const leftCount = 10;
      const rightCount = 5;
      const leftY = (index: number) => 12 + (index / (leftCount - 1)) * 76;
      const rightY = (index: number) => 20 + (index / (rightCount - 1)) * 60;
      const center = pointer ? clamp(pointer[1], 10, 90) : 38;
      const parentOf = Array.from({length: leftCount}, (_, index) =>
        clamp(Math.floor(index / 2) + (random.next() < 0.25 ? 1 : 0), 0, rightCount - 1)
      );
      const selected = Array.from(
        {length: leftCount},
        (_, index) => Math.abs(leftY(index) - center) <= p.width / 2
      );
      const carried = new Array<boolean>(rightCount).fill(false);
      parentOf.forEach((parent, index) => {
        if (selected[index]) carried[parent] = true;
      });
      const quiet: PathItem[] = [];
      const active: PathItem[] = [];
      parentOf.forEach((parent, index) => {
        const link = createPlotPath(
          [
            [28, leftY(index)],
            [60, leftY(index)],
            [100, rightY(parent)],
            [132, rightY(parent)]
          ],
          selected[index] ? 'warm' : 'dim',
          selected[index] ? 0.8 : 0.4,
          selected[index] ? 0.95 : 0.4,
          index / leftCount
        );
        (selected[index] ? active : quiet).push(link);
      });
      return [
        createFills(
          [
            {
              ring: createRectangleRing(10, center - p.width / 2, 36, center + p.width / 2),
              role: 'warm',
              alpha: 0.1
            }
          ],
          'pulse'
        ),
        createStrokes(quiet, 'static'),
        createStrokes(active, 'flow', 0.4, 1),
        createSprites(
          Array.from({length: leftCount}, (_, index) =>
            createDotAt(
              24,
              leftY(index),
              selected[index] ? 2.4 : 1.7,
              selected[index] ? 'warm' : 'dim',
              selected[index] ? 1 : 0.6,
              index / leftCount
            )
          ),
          0.3
        ),
        createSprites(
          Array.from({length: rightCount}, (_, index) =>
            createDotAt(
              136,
              rightY(index),
              carried[index] ? 3.4 : 2.4,
              carried[index] ? 'warm' : 'dim',
              carried[index] ? 1 : 0.55,
              index / rightCount
            )
          ),
          0.4
        )
      ];
    }
  }
};

function mixNumbersLocal(first: number, second: number, amount: number): number {
  return first + (second - first) * amount;
}
