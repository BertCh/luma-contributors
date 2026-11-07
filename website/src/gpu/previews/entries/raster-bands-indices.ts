// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Function-specific previews for the catalog group src/catalog/entries/raster-bands-indices.ts, keyed by entry id.

import {
  FIELD_COLUMNS,
  FIELD_ROWS,
  clamp,
  createBlob,
  createCircle,
  createSmoothField,
  createWanderPath,
  getDistance,
  sampleField,
  type Random
} from '../data.ts';
import {
  clampPoint,
  getClassRole,
  getRoleCycle,
  smoothPath,
  toPaths,
  toSprites,
  type EntryPreview
} from '../helpers.ts';
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
// Local toolkit: small square-cell panels, drawn as additive fills, so one scene can hold several
// rasters side by side (inputs, operator, output).
// ---------------------------------------------------------------------------------------------

const CELL_SIZE = 4;

type Panel = {
  x: number;
  y: number;
  columns: number;
  rows: number;
  size: number;
};
type Cell = {role: Role; alpha: number; phase?: number};
type CellPainter = (u: number, v: number, column: number, row: number) => Cell | null;

function makePanel(x: number, y: number, columns: number, rows: number, size = CELL_SIZE): Panel {
  return {x, y, columns, rows, size};
}

/** The 128 by 80 panel that fills the scene's middle. */
const FULL_PANEL = makePanel(16, 10, 32, 20);

function getPanelWidth(panel: Panel): number {
  return panel.columns * panel.size;
}

function getPanelHeight(panel: Panel): number {
  return panel.rows * panel.size;
}

function paintPanel(panel: Panel, paint: CellPainter): PolygonItem[] {
  const items: PolygonItem[] = [];
  const gap = panel.size > 3 ? 0.35 : 0.2;
  for (let row = 0; row < panel.rows; row++) {
    for (let column = 0; column < panel.columns; column++) {
      const cell = paint((column + 0.5) / panel.columns, (row + 0.5) / panel.rows, column, row);
      if (!cell) continue;
      const left = panel.x + column * panel.size + gap;
      const top = panel.y + row * panel.size + gap;
      const right = left + panel.size - gap * 2;
      const bottom = top + panel.size - gap * 2;
      items.push({
        ring: [
          [left, top],
          [right, top],
          [right, bottom],
          [left, bottom]
        ],
        role: cell.role,
        alpha: cell.alpha,
        phase: cell.phase ?? 0
      });
    }
  }
  return items;
}

function getPanelOutline(panel: Panel, role: Role = 'dim', alpha = 0.65): PathItem {
  const right = panel.x + getPanelWidth(panel);
  const bottom = panel.y + getPanelHeight(panel);
  return {
    points: [
      [panel.x - 1, panel.y - 1],
      [right + 1, panel.y - 1],
      [right + 1, bottom + 1],
      [panel.x - 1, bottom + 1],
      [panel.x - 1, panel.y - 1]
    ],
    role,
    width: 0.5,
    alpha
  };
}

function getPanelEdge(panel: Panel, side: 'left' | 'right'): Point {
  return [
    side === 'left' ? panel.x - 2 : panel.x + getPanelWidth(panel) + 2,
    panel.y + getPanelHeight(panel) / 2
  ];
}

function getPanelCell(panel: Panel, pointer: Point | null): [number, number] | null {
  if (!pointer) return null;
  const column = Math.floor((pointer[0] - panel.x) / panel.size);
  const row = Math.floor((pointer[1] - panel.y) / panel.size);
  if (column < 0 || row < 0 || column >= panel.columns || row >= panel.rows) return null;
  return [column, row];
}

function getCellCenter(panel: Panel, column: number, row: number): Point {
  return [panel.x + (column + 0.5) * panel.size, panel.y + (row + 0.5) * panel.size];
}

function getCellRing(panel: Panel, column: number, row: number, inset = 0): Point[] {
  const left = panel.x + column * panel.size + inset;
  const top = panel.y + row * panel.size + inset;
  const right = left + panel.size - inset * 2;
  const bottom = top + panel.size - inset * 2;
  return [
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom],
    [left, top]
  ];
}

/** Low to high values read cool, teal, amber, pink. */
function getValueRole(value: number): Role {
  if (value < 0.25) return 'cool';
  if (value < 0.5) return 'primary';
  if (value < 0.75) return 'warm';
  return 'hot';
}

function makeCell(role: Role, value: number, floor = 0.12): Cell {
  return {role, alpha: floor + (1 - floor) * clamp(value, 0, 1)};
}

function createLandscape(random: Random, waves = 6, frequency = 1.4): Float32Array {
  return createSmoothField(random, waves, frequency);
}

function readField(values: Float32Array, u: number, v: number): number {
  return sampleField(values, FIELD_COLUMNS, FIELD_ROWS, u, v);
}

function createLink(from: Point, to: Point, role: Role = 'warm', alpha = 0.8): PathItem {
  return {points: [from, to], role, width: 0.7, alpha};
}

function createFlowLayer(items: PathItem[], speed = 0.35, trail = 1): StrokeLayer {
  return {
    type: 'strokes',
    items,
    animation: 'flow',
    speed,
    trail,
    blend: 'add'
  };
}

function createStaticLayer(items: PathItem[]): StrokeLayer {
  return {type: 'strokes', items, animation: 'static'};
}

function createFillLayer(
  items: PolygonItem[],
  animation: 'static' | 'pulse' = 'static'
): FillLayer {
  return {type: 'fills', items, animation, blend: 'add'};
}

function createNode(x: number, y: number, role: Role = 'warm', radius = 3): SpriteItem {
  return {x, y, radius, role, alpha: 0.95, phase: 0};
}

function isInsideRing(point: Point, ring: Point[]): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[index];
    const [bx, by] = ring[previous];
    if (
      ay > point[1] !== by > point[1] &&
      point[0] < ((bx - ax) * (point[1] - ay)) / (by - ay) + ax
    ) {
      inside = !inside;
    }
  }
  return inside;
}

/** Shared "bands in, derived raster out" picture. */
function buildCombine(options: {
  inputA: (u: number, v: number) => number;
  inputB: (u: number, v: number) => number;
  roleA: Role;
  roleB: Role;
  paintOutput: (a: number, b: number, u: number, v: number) => Cell | null;
  operatorRole?: Role;
}): PreviewLayer[] {
  const inputTop = makePanel(8, 10, 11, 9);
  const inputBottom = makePanel(8, 54, 11, 9);
  const output = makePanel(86, 10, 17, 20);
  const {inputA, inputB, roleA, roleB, paintOutput, operatorRole = 'warm'} = options;
  const outputItems = paintPanel(output, (u, v, column) => {
    const cell = paintOutput(inputA(u, v), inputB(u, v), u, v);
    return cell ? {...cell, phase: column / output.columns} : null;
  });
  const centerA = getPanelEdge(inputTop, 'right');
  const centerB = getPanelEdge(inputBottom, 'right');
  const operator: Point = [68, 50];
  return [
    createFillLayer(paintPanel(inputTop, (u, v) => makeCell(roleA, inputA(u, v)))),
    createFillLayer(paintPanel(inputBottom, (u, v) => makeCell(roleB, inputB(u, v)))),
    createFillLayer(outputItems, 'pulse'),
    createStaticLayer([
      getPanelOutline(inputTop, roleA, 0.5),
      getPanelOutline(inputBottom, roleB, 0.5),
      getPanelOutline(output, 'dim', 0.6)
    ]),
    createFlowLayer([
      createLink(centerA, operator, roleA),
      createLink(centerB, operator, roleB),
      createLink(operator, getPanelEdge(output, 'left'), operatorRole)
    ]),
    {
      type: 'sprites',
      items: [createNode(operator[0], operator[1], operatorRole, 3.6)],
      pulse: 0.4
    }
  ];
}

/** Connected-component labels of equal-valued cells (4-connected). */
function labelComponents(classes: Int8Array, columns: number, rows: number) {
  const labels = new Int32Array(columns * rows).fill(-1);
  const sizes: number[] = [];
  const stack: number[] = [];
  for (let start = 0; start < labels.length; start++) {
    if (labels[start] >= 0 || classes[start] < 0) continue;
    const label = sizes.length;
    let size = 0;
    stack.push(start);
    labels[start] = label;
    while (stack.length > 0) {
      const index = stack.pop() as number;
      size++;
      const column = index % columns;
      const row = (index - column) / columns;
      const neighbors = [
        column > 0 ? index - 1 : -1,
        column < columns - 1 ? index + 1 : -1,
        row > 0 ? index - columns : -1,
        row < rows - 1 ? index + columns : -1
      ];
      for (const neighbor of neighbors) {
        if (neighbor >= 0 && labels[neighbor] < 0 && classes[neighbor] === classes[start]) {
          labels[neighbor] = label;
          stack.push(neighbor);
        }
      }
    }
    sizes.push(size);
  }
  return {labels, sizes};
}

/** Samples a scene-wide field onto a panel's cells. */
function createPanelValues(field: Float32Array, panel: Panel): Float32Array {
  const values = new Float32Array(panel.columns * panel.rows);
  for (let row = 0; row < panel.rows; row++) {
    for (let column = 0; column < panel.columns; column++) {
      values[row * panel.columns + column] = readField(
        field,
        (column + 0.5) / panel.columns,
        (row + 0.5) / panel.rows
      );
    }
  }
  return values;
}

/** Nearest of a few seeded centers, giving Voronoi-like zones. */
function createZoneCenters(random: Random, count: number): Point[] {
  return Array.from({length: count}, (_, index) => [
    ((index % 4) + 0.5 + random.range(-0.25, 0.25)) * (1 / 4),
    (Math.floor(index / 4) + 0.5 + random.range(-0.25, 0.25)) * (1 / Math.ceil(count / 4))
  ]);
}

function getZoneIndex(centers: Point[], u: number, v: number): number {
  let best = 0;
  let bestDistance = Infinity;
  centers.forEach(([cx, cy], index) => {
    const distance = Math.hypot(u - cx, (v - cy) * 0.8);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  });
  return best;
}

function createBarPolygon(left: number, bottom: number, width: number, height: number): Point[] {
  return [
    [left, bottom],
    [left, bottom - height],
    [left + width, bottom - height],
    [left + width, bottom]
  ];
}

export const RASTER_BANDS_INDICES_PREVIEWS: Record<string, EntryPreview> = {
  // Grids and bands ---------------------------------------------------------------------------

  'make-raster-grid': {
    pointer: 'hover',
    parameters: [
      {
        id: 'columns',
        label: 'Cells across',
        min: 6,
        max: 24,
        step: 1,
        default: 14
      },
      {
        id: 'direction',
        label: 'Row direction',
        min: 0,
        max: 1,
        step: 1,
        default: 0,
        options: ['north up', 'south up']
      }
    ],
    build: ({p, pointer}) => {
      const columns = Math.round(p.columns);
      const size = 120 / columns;
      const rows = Math.max(2, Math.floor(76 / size));
      const northUp = p.direction < 0.5;
      const left = 24;
      const width = columns * size;
      const top = 12 + (76 - rows * size) / 2;
      const bottom = top + rows * size;
      const right = left + width;
      // Row zero sits at the origin corner; rows count away from it.
      const originY = northUp ? top : bottom;
      const rowStep = northUp ? size : -size;
      const cellRing = (column: number, row: number, inset: number): Point[] => {
        const y0 = originY + row * rowStep;
        const y1 = y0 + rowStep;
        const yNear = y0 + (northUp ? inset : -inset);
        const yFar = y1 + (northUp ? -inset : inset);
        return [
          [left + column * size + inset, yNear],
          [left + (column + 1) * size - inset, yNear],
          [left + (column + 1) * size - inset, yFar],
          [left + column * size + inset, yFar]
        ];
      };
      // Cells fill in from the origin row by row: the pulse phase is the row-major cell order.
      const cells: PolygonItem[] = [];
      const total = columns * rows;
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const order = (row * columns + column) / total;
          const checker = (row + column) % 2 === 0 ? 0.5 : 0.34;
          cells.push({
            ring: cellRing(column, row, 0.25),
            role: getClassRole(Math.min(4, Math.floor((row / rows) * 5)), 5),
            alpha: checker,
            phase: order
          });
        }
      }
      const lines: PathItem[] = [];
      for (let column = 0; column <= columns; column++) {
        lines.push({
          points: [
            [left + column * size, top],
            [left + column * size, bottom]
          ],
          role: 'dim',
          width: 0.25,
          alpha: 0.45
        });
      }
      for (let row = 0; row <= rows; row++) {
        lines.push({
          points: [
            [left, top + row * size],
            [right, top + row * size]
          ],
          role: 'dim',
          width: 0.25,
          alpha: 0.45
        });
      }
      // Extent box, with the cell size marked as a bracket on the first cell.
      const extent: PathItem = {
        points: [
          [left, top],
          [right, top],
          [right, bottom],
          [left, bottom],
          [left, top]
        ],
        role: 'violet',
        width: 1,
        alpha: 0.95
      };
      const tickY = northUp ? top - 4 : bottom + 4;
      const sizeBracket: PathItem[] = [
        {
          points: [
            [left, tickY],
            [left + size, tickY]
          ],
          role: 'ink',
          width: 0.8,
          alpha: 0.95
        },
        {
          points: [
            [left, tickY - 1.5],
            [left, tickY + 1.5]
          ],
          role: 'ink',
          width: 0.6,
          alpha: 0.95
        },
        {
          points: [
            [left + size, tickY - 1.5],
            [left + size, tickY + 1.5]
          ],
          role: 'ink',
          width: 0.6,
          alpha: 0.95
        }
      ];
      // Affine frame: pixel axes from the origin, world axes in the scene corner, and a link.
      const worldOrigin: Point = [8, northUp ? 92 : 8];
      const worldUp = northUp ? -1 : 1;
      const worldAxes: PathItem[] = [
        {points: [worldOrigin, [worldOrigin[0] + 11, worldOrigin[1]]], role: 'cool', width: 0.8},
        {
          points: [worldOrigin, [worldOrigin[0], worldOrigin[1] + worldUp * 11]],
          role: 'primary',
          width: 0.8
        }
      ];
      const hovered: PolygonItem[] = [];
      if (pointer) {
        const column = Math.floor((pointer[0] - left) / size);
        const row = Math.floor((pointer[1] - originY) / rowStep);
        if (column >= 0 && column < columns && row >= 0 && row < rows) {
          hovered.push({ring: cellRing(column, row, 0.2), role: 'hot', alpha: 0.95});
        }
      }
      return [
        createFillLayer(cells, 'pulse'),
        createStaticLayer([...lines, ...sizeBracket, ...worldAxes]),
        createFillLayer(
          [{ring: cellRing(0, 0, 0.2), role: 'warm', alpha: 0.9}, ...hovered],
          'pulse'
        ),
        createFlowLayer(
          [
            {
              points: [
                [left, originY],
                [right, originY]
              ],
              role: 'cool',
              width: 1.5,
              alpha: 0.95
            },
            {
              points: [
                [left, originY],
                [left, originY + rowStep * rows]
              ],
              role: 'primary',
              width: 1.5,
              alpha: 0.95
            },
            {
              points: [worldOrigin, [left, originY]],
              role: 'warm',
              width: 0.6,
              alpha: 0.7
            }
          ],
          0.3
        ),
        createStaticLayer([extent]),
        {
          type: 'sprites',
          items: [
            createNode(left, originY, 'warm', 3.6),
            createNode(worldOrigin[0], worldOrigin[1], 'ink', 2.2),
            createNode(right, originY + rowStep * rows, 'violet', 2.4)
          ],
          pulse: 0.5
        }
      ];
    }
  },

  'make-raster-band': {
    parameters: [
      {
        id: 'nodata',
        label: 'Nodata share',
        min: 0,
        max: 45,
        step: 5,
        default: 20,
        unit: '%'
      }
    ],
    build: ({fork, p}) => {
      const field = createLandscape(fork('values'), 5, 1.3);
      const noise = fork('validity');
      const valid = makePanel(60, 34, 10, 8);
      const values = makePanel(6, 34, 10, 8);
      const band = makePanel(114, 34, 10, 8);
      const mask = Array.from(
        {length: valid.columns * valid.rows},
        () => noise.next() >= p.nodata / 100
      );
      const isValid = (column: number, row: number) => mask[row * valid.columns + column];
      return [
        createFillLayer(paintPanel(values, (u, v) => makeCell('cool', readField(field, u, v)))),
        createFillLayer(
          paintPanel(valid, (_u, _v, column, row) =>
            isValid(column, row) ? {role: 'primary', alpha: 0.75} : {role: 'hot', alpha: 0.25}
          )
        ),
        createFillLayer(
          paintPanel(band, (u, v, column, row) =>
            isValid(column, row)
              ? {
                  ...makeCell('warm', readField(field, u, v)),
                  phase: column / band.columns
                }
              : null
          ),
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(values, 'cool', 0.5),
          getPanelOutline(valid, 'primary', 0.5),
          getPanelOutline(band, 'warm', 0.5)
        ]),
        createFlowLayer([
          createLink([46, 50], [58, 50], 'cool'),
          createLink([100, 50], [112, 50], 'primary')
        ]),
        {
          type: 'sprites',
          items: [createNode(52, 50, 'warm', 2.6), createNode(106, 50, 'warm', 2.6)],
          pulse: 0.4
        }
      ];
    }
  },

  // Spectral indices --------------------------------------------------------------------------

  'normalized-difference-water-index': {
    parameters: [
      {
        id: 'threshold',
        label: 'Water threshold',
        min: 0.1,
        max: 0.9,
        step: 0.05,
        default: 0.5
      },
      {
        id: 'variant',
        label: 'Variant',
        min: 0,
        max: 1,
        step: 1,
        default: 0,
        options: ['McFeeters', 'Gao']
      }
    ],
    build: ({fork, p}) => {
      const field = createLandscape(fork('land'), 6, 1.5);
      const water = (u: number, v: number) => clamp((0.42 - readField(field, u, v)) * 6, 0, 1);
      const gao = p.variant > 0.5;
      // McFeeters: green is bright over water, NIR dark. Gao: NIR against SWIR.
      const first = (u: number, v: number) =>
        gao ? 0.7 - 0.5 * water(u, v) : 0.35 + 0.35 * water(u, v);
      const second = (u: number, v: number) =>
        gao ? 0.25 + 0.2 * (1 - water(u, v)) : 0.7 - 0.5 * water(u, v);
      return buildCombine({
        inputA: first,
        inputB: second,
        roleA: gao ? 'violet' : 'primary',
        roleB: gao ? 'hot' : 'violet',
        paintOutput: (a, b) => {
          const index = (a - b) / (a + b + 1e-6);
          const scaled = clamp((index + 0.6) / 1.2, 0, 1);
          return scaled >= p.threshold ? makeCell('cool', scaled, 0.5) : {role: 'dim', alpha: 0.18};
        },
        operatorRole: 'cool'
      });
    }
  },

  'normalized-burn-ratio': {
    pointer: 'hover',
    parameters: [
      {
        id: 'severity',
        label: 'Burn severity',
        min: 0.2,
        max: 1,
        step: 0.05,
        default: 0.85
      }
    ],
    build: ({fork, p, pointer}) => {
      const field = createLandscape(fork('land'), 5, 1.2);
      const center = pointer ? clampPoint(pointer, 8) : ([48, 40] as Point);
      const scar = (u: number, v: number) => {
        const distance = Math.hypot(u * 160 - center[0], (v * 100 - center[1]) * 1.1);
        const edge = 18 + 8 * readField(field, v, u);
        return clamp((edge - distance) / 6, 0, 1) * p.severity;
      };
      return buildCombine({
        inputA: (u, v) => 0.75 - 0.55 * scar(u, v) + 0.1 * (readField(field, u, v) - 0.5),
        inputB: (u, v) => 0.2 + 0.6 * scar(u, v) + 0.08 * (readField(field, u, v) - 0.5),
        roleA: 'primary',
        roleB: 'warm',
        paintOutput: (nir, swir) => {
          const nbr = (nir - swir) / (nir + swir + 1e-6);
          return nbr < 0.25
            ? makeCell('hot', 0.4 + (0.25 - nbr), 0.4)
            : makeCell('primary', nbr, 0.15);
        },
        operatorRole: 'hot'
      });
    }
  },

  'enhanced-vegetation-index': {
    parameters: [
      {
        id: 'canopy',
        label: 'Canopy density',
        min: 0.3,
        max: 1.6,
        step: 0.1,
        default: 1.1
      }
    ],
    build: ({fork, p}) => {
      const field = createLandscape(fork('canopy'), 6, 1.3);
      const haze = createLandscape(fork('haze'), 4, 2.2);
      const cover = (u: number, v: number) => clamp(readField(field, u, v) * p.canopy, 0, 1.4);
      const inputTop = makePanel(8, 10, 11, 9);
      const inputBottom = makePanel(8, 54, 11, 9);
      const ndviPanel = makePanel(86, 10, 17, 9);
      const eviPanel = makePanel(86, 54, 17, 9);
      return [
        createFillLayer(
          paintPanel(inputTop, (u, v) => makeCell('violet', 0.25 + 0.55 * clamp(cover(u, v), 0, 1)))
        ),
        createFillLayer(
          paintPanel(inputBottom, (u, v) => makeCell('hot', 0.7 - 0.4 * clamp(cover(u, v), 0, 1)))
        ),
        // The blue band: an aerosol haze EVI corrects for.
        createFillLayer(
          paintPanel(inputBottom, (u, v) => makeCell('cool', readField(haze, u, v) * 0.5, 0.0))
        ),
        // NDVI saturates over dense canopy; EVI keeps a gradient.
        createFillLayer(
          paintPanel(ndviPanel, (u, v) => makeCell('dim', Math.min(1, cover(u, v) * 2.6), 0.2))
        ),
        createFillLayer(
          paintPanel(eviPanel, (u, v, column) => ({
            ...makeCell('primary', (cover(u, v) / 1.4) ** 0.7, 0.1),
            phase: column / eviPanel.columns
          })),
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(inputTop, 'violet', 0.5),
          getPanelOutline(inputBottom, 'hot', 0.5),
          getPanelOutline(ndviPanel, 'dim', 0.6),
          getPanelOutline(eviPanel, 'primary', 0.7)
        ]),
        createFlowLayer([
          createLink([52, 28], [68, 50], 'violet'),
          createLink([52, 72], [68, 50], 'hot'),
          createLink([68, 50], [84, 28], 'dim', 0.5),
          createLink([68, 50], [84, 72], 'primary')
        ]),
        {
          type: 'sprites',
          items: [createNode(68, 50, 'warm', 3.6)],
          pulse: 0.4
        }
      ];
    }
  },

  'soil-adjusted-vegetation-index': {
    parameters: [
      {
        id: 'soil',
        label: 'Soil factor L',
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.5
      }
    ],
    build: ({fork, p}) => {
      const soil = fork('soil');
      const plants = fork('plants');
      const originX = 12 - p.soil * 8;
      const originY = 86 + p.soil * 10;
      const soilLine: Point[] = [
        [12, 86],
        [72, 26]
      ];
      const soilPoints: Point[] = Array.from({length: 34}, () => {
        const amount = soil.next();
        return [12 + amount * 56 + soil.gaussian() * 1.2, 86 - amount * 56 + soil.gaussian() * 1.4];
      });
      const vegetationPoints: Point[] = Array.from({length: 26}, () => {
        const amount = plants.next();
        return [
          14 + amount * 40 + plants.gaussian() * 2,
          82 - amount * 36 - plants.range(2, 26) * (0.4 + amount)
        ];
      });
      const fan = [0.25, 0.5, 0.75, 1].map(
        (amount): PathItem => ({
          points: [
            [originX, originY],
            [originX + 70, originY - 70 * (0.25 + amount * 1.3)]
          ],
          role: 'warm',
          width: 0.5,
          alpha: 0.7
        })
      );
      const raster = makePanel(88, 12, 16, 20, 4);
      const cover = createLandscape(fork('cover'), 5, 1.8);
      return [
        createStaticLayer([
          {
            points: [
              [10, 14],
              [10, 88],
              [76, 88]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          },
          {points: soilLine, role: 'cool', width: 0.8, alpha: 0.6}
        ]),
        createFlowLayer(fan, 0.25, 0.6),
        {
          type: 'sprites',
          items: [
            ...toSprites(soilPoints, 'cool', 1.2, soil, 0.7),
            ...toSprites(vegetationPoints, 'primary', 1.3, plants, 0.9),
            createNode(originX, originY, 'warm', 2.6)
          ],
          pulse: 0.3
        },
        // Sparse cover over bare ground: soil cells stay dark.
        createFillLayer(
          paintPanel(raster, (u, v, column) => {
            const amount = clamp((readField(cover, u, v) - 0.55) * 3, 0, 1);
            return amount > 0.05
              ? {
                  ...makeCell('primary', amount, 0.3),
                  phase: column / raster.columns
                }
              : {role: 'cool', alpha: 0.12};
          }),
          'pulse'
        ),
        createStaticLayer([getPanelOutline(raster, 'primary', 0.5)])
      ];
    }
  },

  // Map algebra -------------------------------------------------------------------------------

  'raster-arithmetic': {
    parameters: [
      {
        id: 'operator',
        label: 'Operator',
        min: 0,
        max: 3,
        step: 1,
        default: 0,
        options: ['A + B', 'A - B', 'A × B', 'A / B']
      }
    ],
    build: ({fork, p}) => {
      const fieldA = createLandscape(fork('a'), 5, 1.2);
      const fieldB = createLandscape(fork('b'), 5, 1.6);
      const operator = Math.round(p.operator);
      return buildCombine({
        inputA: (u, v) => readField(fieldA, u, v),
        inputB: (u, v) => readField(fieldB, u, v),
        roleA: 'cool',
        roleB: 'warm',
        paintOutput: (a, b) => {
          const value = [(a + b) / 2, (a - b) / 2 + 0.5, a * b, a / (b + 0.35) / 2.5][operator];
          return makeCell(getValueRole(value), value, 0.2);
        }
      });
    }
  },

  'cell-statistics': {
    pointer: 'hover',
    parameters: [
      {
        id: 'statistic',
        label: 'Statistic',
        min: 0,
        max: 3,
        step: 1,
        default: 0,
        options: ['mean', 'min', 'max', 'range']
      }
    ],
    build: ({fork, p, pointer}) => {
      const layerCount = 5;
      const fields = Array.from({length: layerCount}, (_, index) =>
        createLandscape(fork(`layer-${index}`), 5, 1.4)
      );
      const stackPanels = fields.map((_, index) =>
        makePanel(8 + index * 7, 6 + index * 13, 8, 6, 3.6)
      );
      const output = makePanel(86, 10, 17, 20);
      const statistic = Math.round(p.statistic);
      const reduce = (u: number, v: number): number => {
        const values = fields.map(field => readField(field, u, v));
        const minimum = Math.min(...values);
        const maximum = Math.max(...values);
        return [
          values.reduce((sum, value) => sum + value, 0) / layerCount,
          minimum,
          maximum,
          (maximum - minimum) * 1.6
        ][statistic];
      };
      const selected = getPanelCell(output, pointer) ?? ([9, 8] as [number, number]);
      const target = getCellCenter(output, selected[0], selected[1]);
      const links: PathItem[] = stackPanels.map(panel => ({
        points: [
          [panel.x + getPanelWidth(panel) / 2 + 4, panel.y + getPanelHeight(panel) / 2],
          target
        ],
        role: 'warm',
        width: 0.35,
        alpha: 0.55
      }));
      return [
        ...stackPanels.map((panel, index) =>
          createFillLayer(
            paintPanel(panel, (u, v) => makeCell('cool', readField(fields[index], u, v), 0.1))
          )
        ),
        createStaticLayer(stackPanels.map(panel => getPanelOutline(panel, 'cool', 0.45))),
        createFillLayer(
          paintPanel(output, (u, v, column) => {
            const value = reduce(u, v);
            return {
              ...makeCell(getValueRole(value), value, 0.15),
              phase: column / output.columns
            };
          }),
          'pulse'
        ),
        createFlowLayer(links, 0.3, 0.8),
        createFillLayer(
          [
            {
              ring: getCellRing(output, selected[0], selected[1]),
              role: 'ink',
              alpha: 0.9
            }
          ],
          'pulse'
        ),
        createStaticLayer([getPanelOutline(output, 'dim', 0.6)])
      ];
    }
  },

  'raster-conditional': {
    parameters: [
      {
        id: 'threshold',
        label: 'Condition threshold',
        min: 0.2,
        max: 0.8,
        step: 0.05,
        default: 0.5
      }
    ],
    build: ({fork, p}) => {
      const conditionField = createLandscape(fork('condition'), 5, 1.3);
      const fieldA = createLandscape(fork('a'), 4, 1.8);
      const fieldB = createLandscape(fork('b'), 4, 1.8);
      const condition = makePanel(8, 6, 9, 6, 4);
      const inputA = makePanel(8, 38, 9, 6, 4);
      const inputB = makePanel(8, 70, 9, 6, 4);
      const output = makePanel(86, 10, 17, 20);
      const passes = (u: number, v: number) => readField(conditionField, u, v) > p.threshold;
      return [
        createFillLayer(
          paintPanel(condition, (u, v) =>
            passes(u, v) ? {role: 'primary', alpha: 0.85} : {role: 'dim', alpha: 0.2}
          )
        ),
        createFillLayer(paintPanel(inputA, (u, v) => makeCell('warm', readField(fieldA, u, v)))),
        createFillLayer(paintPanel(inputB, (u, v) => makeCell('cool', readField(fieldB, u, v)))),
        createFillLayer(
          paintPanel(output, (u, v, column) =>
            passes(u, v)
              ? {
                  ...makeCell('warm', readField(fieldA, u, v), 0.3),
                  phase: column / output.columns
                }
              : {
                  ...makeCell('cool', readField(fieldB, u, v), 0.3),
                  phase: column / output.columns
                }
          ),
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(condition, 'primary', 0.5),
          getPanelOutline(inputA, 'warm', 0.5),
          getPanelOutline(inputB, 'cool', 0.5),
          getPanelOutline(output, 'dim', 0.6)
        ]),
        createFlowLayer([
          createLink([42, 18], [68, 50], 'primary'),
          createLink([42, 50], [68, 50], 'warm'),
          createLink([42, 82], [68, 50], 'cool'),
          createLink([68, 50], [84, 50], 'ink')
        ]),
        {
          type: 'sprites',
          items: [createNode(68, 50, 'primary', 3.6)],
          pulse: 0.4
        }
      ];
    }
  },

  'raster-reclassify': {
    parameters: [{id: 'classes', label: 'Classes', min: 2, max: 5, step: 1, default: 4}],
    build: ({fork, p}) => {
      const field = createLandscape(fork('values'), 6, 1.2);
      const classes = Math.round(p.classes);
      const source = makePanel(6, 10, 15, 20);
      const target = makePanel(88, 10, 15, 20);
      const classOf = (value: number) => Math.min(classes - 1, Math.floor(value * classes));
      const tableLeft = 72;
      const bars: PolygonItem[] = Array.from({length: classes}, (_, index) => ({
        ring: createBarPolygon(tableLeft, 90 - (index * 80) / classes, 8, 80 / classes - 1.2),
        role: getClassRole(index, classes),
        alpha: 0.85,
        phase: index / classes
      }));
      const breaks: PathItem[] = Array.from({length: classes - 1}, (_, index) => ({
        points: [
          [tableLeft - 5, 90 - ((index + 1) * 80) / classes],
          [tableLeft + 13, 90 - ((index + 1) * 80) / classes]
        ],
        role: 'ink',
        width: 0.4,
        alpha: 0.8
      }));
      return [
        createFillLayer(
          paintPanel(source, (u, v) => makeCell('primary', readField(field, u, v), 0.08))
        ),
        createFillLayer(
          paintPanel(target, (u, v, column) => {
            const bin = classOf(readField(field, u, v));
            return {
              role: getClassRole(bin, classes),
              alpha: 0.75,
              phase: column / target.columns
            };
          }),
          'pulse'
        ),
        createFillLayer(bars, 'pulse'),
        createStaticLayer([
          getPanelOutline(source, 'primary', 0.5),
          getPanelOutline(target, 'dim', 0.6),
          ...breaks
        ]),
        createFlowLayer(
          [createLink([68, 50], [71, 50], 'primary'), createLink([82, 50], [86, 50], 'warm')],
          0.3
        )
      ];
    }
  },

  'weighted-overlay': {
    parameters: [
      {
        id: 'first',
        label: 'Slope weight',
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.5
      },
      {
        id: 'second',
        label: 'Road weight',
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.3
      },
      {
        id: 'third',
        label: 'Soil weight',
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.2
      }
    ],
    build: ({fork, p}) => {
      const fields = [fork('first'), fork('second'), fork('third')].map(random =>
        createLandscape(random, 5, 1.5)
      );
      const weights = [p.first, p.second, p.third];
      const total = weights.reduce((sum, weight) => sum + weight, 0) || 1;
      const roles: Role[] = ['cool', 'violet', 'warm'];
      const inputs = [6, 38, 70].map(y => makePanel(8, y, 9, 6, 4));
      const output = makePanel(86, 10, 17, 20);
      const links: PathItem[] = inputs.map((panel, index) => ({
        points: [
          [panel.x + getPanelWidth(panel) + 3, panel.y + 12],
          [68, 50]
        ],
        role: roles[index],
        width: 0.3 + (weights[index] / total) * 3,
        alpha: 0.8
      }));
      return [
        ...inputs.map((panel, index) =>
          createFillLayer(
            paintPanel(panel, (u, v) =>
              makeCell(roles[index], readField(fields[index], u, v), 0.08)
            )
          )
        ),
        createFillLayer(
          paintPanel(output, (u, v, column) => {
            const score =
              fields.reduce(
                (sum, field, index) => sum + weights[index] * readField(field, u, v),
                0
              ) / total;
            return {
              ...makeCell(getValueRole(score), score, 0.2),
              phase: column / output.columns
            };
          }),
          'pulse'
        ),
        createStaticLayer([
          ...inputs.map((panel, index) => getPanelOutline(panel, roles[index], 0.5)),
          getPanelOutline(output, 'dim', 0.6)
        ]),
        createFlowLayer([...links, createLink([68, 50], [84, 50], 'ink')]),
        {
          type: 'sprites',
          items: [createNode(68, 50, 'ink', 3.6)],
          pulse: 0.4
        }
      ];
    }
  },

  // Zones -------------------------------------------------------------------------------------

  'polygon-rasterization': {
    parameters: [
      {
        id: 'columns',
        label: 'Cells across',
        min: 10,
        max: 40,
        step: 2,
        default: 24
      },
      {
        id: 'boundary',
        label: 'Boundary flag',
        min: 0,
        max: 1,
        step: 1,
        default: 1,
        options: ['off', 'on']
      }
    ],
    build: ({fork, p}) => {
      const random = fork('polygons');
      const columns = Math.round(p.columns);
      const size = 128 / columns;
      const rows = Math.floor(80 / size);
      const panel = makePanel(16, 10, columns, rows, size);
      const polygons = [
        createBlob(random, 50, 36, 22, 12, 0.3),
        createBlob(random, 108, 34, 17, 12, 0.3),
        createBlob(random, 86, 70, 18, 12, 0.3)
      ];
      const zoneOf = (column: number, row: number): number => {
        if (column < 0 || row < 0 || column >= columns || row >= rows) return -1;
        const point = getCellCenter(panel, column, row);
        return polygons.findIndex(ring => isInsideRing(point, ring));
      };
      const items = paintPanel(panel, (_u, _v, column, row) => {
        const zone = zoneOf(column, row);
        if (zone < 0) return null;
        const edge = [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1]
        ].some(([dx, dy]) => zoneOf(column + dx, row + dy) !== zone);
        return p.boundary > 0.5 && edge
          ? {role: 'ink', alpha: 0.85, phase: zone / 3}
          : {role: getRoleCycle(zone), alpha: 0.5, phase: zone / 3};
      });
      return [
        createFillLayer(items, 'pulse'),
        {
          type: 'strokes',
          items: toPaths(
            polygons.map(ring => [...ring, ring[0]]),
            'ink',
            0.5,
            0.7
          ),
          animation: 'draw',
          speed: 0.15,
          blend: 'add'
        }
      ];
    }
  },

  'raster-zonal-statistics': {
    parameters: [
      {
        id: 'statistic',
        label: 'Statistic',
        min: 0,
        max: 2,
        step: 1,
        default: 0,
        options: ['mean', 'max', 'sum']
      }
    ],
    build: ({fork, p}) => {
      const field = createLandscape(fork('values'), 6, 1.5);
      const centers = createZoneCenters(fork('zones'), 5);
      const raw = makePanel(8, 10, 17, 20);
      const zonal = makePanel(84, 10, 17, 20);
      const statistic = Math.round(p.statistic);
      const sums = new Float32Array(centers.length);
      const counts = new Float32Array(centers.length);
      const maxima = new Float32Array(centers.length);
      const zones: number[] = [];
      const values: number[] = [];
      for (let row = 0; row < raw.rows; row++) {
        for (let column = 0; column < raw.columns; column++) {
          const u = (column + 0.5) / raw.columns;
          const v = (row + 0.5) / raw.rows;
          const zone = getZoneIndex(centers, u, v);
          const value = readField(field, u, v);
          zones.push(zone);
          values.push(value);
          sums[zone] += value;
          counts[zone]++;
          maxima[zone] = Math.max(maxima[zone], value);
        }
      }
      const summary = (zone: number) =>
        [sums[zone] / counts[zone], maxima[zone], Math.min(1, sums[zone] / 60)][statistic];
      return [
        createFillLayer(
          paintPanel(raw, (_u, _v, column, row) =>
            makeCell('cool', values[row * raw.columns + column], 0.1)
          )
        ),
        createFillLayer(
          paintPanel(zonal, (_u, _v, column, row) => {
            const value = summary(zones[row * zonal.columns + column]);
            return {
              ...makeCell(getValueRole(value), value, 0.25),
              phase: zones[row * zonal.columns + column] / centers.length
            };
          }),
          'pulse'
        ),
        createStaticLayer([getPanelOutline(raw, 'cool', 0.5), getPanelOutline(zonal, 'dim', 0.6)]),
        createFlowLayer(
          [
            createLink([72, 30], [82, 30], 'warm'),
            createLink([72, 50], [82, 50], 'warm'),
            createLink([72, 70], [82, 70], 'warm')
          ],
          0.3
        ),
        {type: 'sprites', items: [createNode(76, 50, 'warm', 3)], pulse: 0.4}
      ];
    }
  },

  'raster-join': {
    pointer: 'hover',
    parameters: [{id: 'count', label: 'Points', min: 10, max: 140, step: 5, default: 60}],
    build: ({fork, p, pointer}) => {
      const centers = createZoneCenters(fork('zones'), 4);
      const panel = makePanel(8, 10, 28, 20);
      const random = fork('points');
      const hub = pointer ? clampPoint(pointer, 12) : ([60, 44] as Point);
      const points: Point[] = Array.from({length: Math.round(p.count)}, (_, index) =>
        index % 3 === 0
          ? ([hub[0] + random.gaussian() * 9, hub[1] + random.gaussian() * 8] as Point)
          : ([random.range(10, 118), random.range(12, 88)] as Point)
      ).map(point => [clamp(point[0], 9, 119), clamp(point[1], 11, 89)] as Point);
      const counts = new Map<number, number>();
      const zoneCounts = new Array<number>(centers.length).fill(0);
      for (const [x, y] of points) {
        const column = Math.floor((x - panel.x) / panel.size);
        const row = Math.floor((y - panel.y) / panel.size);
        counts.set(
          row * panel.columns + column,
          (counts.get(row * panel.columns + column) ?? 0) + 1
        );
        zoneCounts[
          getZoneIndex(centers, (column + 0.5) / panel.columns, (row + 0.5) / panel.rows)
        ]++;
      }
      const tallest = Math.max(...zoneCounts, 1);
      const bars: PolygonItem[] = zoneCounts.map((count, index) => ({
        ring: createBarPolygon(132 + index * 7, 88, 4.5, 3 + (count / tallest) * 66),
        role: getRoleCycle(index),
        alpha: 0.85,
        phase: index / 4
      }));
      return [
        createFillLayer(
          paintPanel(panel, (u, v) => ({
            role: getRoleCycle(getZoneIndex(centers, u, v)),
            alpha: 0.16
          }))
        ),
        createFillLayer(
          paintPanel(panel, (_u, _v, column, row) => {
            const count = counts.get(row * panel.columns + column);
            return count
              ? {
                  role: 'ink',
                  alpha: Math.min(1, 0.35 + count * 0.2),
                  phase: (column + row) / 40
                }
              : null;
          }),
          'pulse'
        ),
        createFillLayer(bars, 'pulse'),
        {
          type: 'sprites',
          items: toSprites(points, 'warm', 1.1, random, 0.9),
          pulse: 0.3
        },
        createStaticLayer([
          getPanelOutline(panel, 'dim', 0.5),
          {
            points: [
              [130, 90],
              [160, 90]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.6
          }
        ]),
        createFlowLayer([createLink([121, 50], [129, 50], 'warm')], 0.3)
      ];
    }
  },

  'fractional-coverage': {
    pointer: 'hover',
    parameters: [
      {
        id: 'columns',
        label: 'Cells across',
        min: 8,
        max: 32,
        step: 2,
        default: 16
      }
    ],
    build: ({fork, p, pointer}) => {
      const columns = Math.round(p.columns);
      const size = 128 / columns;
      const rows = Math.floor(80 / size);
      const panel = makePanel(16, 10, columns, rows, size);
      const center = pointer ? clampPoint(pointer, 24) : ([84, 50] as Point);
      const ring = createBlob(fork('shape'), center[0], center[1], 30, 14, 0.3);
      const samples = 4;
      const items = paintPanel(panel, (_u, _v, column, row) => {
        let inside = 0;
        for (let sy = 0; sy < samples; sy++) {
          for (let sx = 0; sx < samples; sx++) {
            const point: Point = [
              panel.x + (column + (sx + 0.5) / samples) * size,
              panel.y + (row + (sy + 0.5) / samples) * size
            ];
            if (isInsideRing(point, ring)) inside++;
          }
        }
        const fraction = inside / (samples * samples);
        if (fraction === 0) return {role: 'dim', alpha: 0.08};
        return fraction === 1
          ? {role: 'primary', alpha: 0.8}
          : {role: 'warm', alpha: 0.2 + fraction * 0.7, phase: fraction};
      });
      return [
        createFillLayer(items, 'pulse'),
        {
          type: 'strokes',
          items: toPaths([[...ring, ring[0]]], 'ink', 0.5, 0.85),
          animation: 'flow',
          speed: 0.2,
          blend: 'add'
        }
      ];
    }
  },

  'fractional-zonal-statistics': {
    pointer: 'hover',
    parameters: [],
    build: ({fork, pointer}) => {
      const field = createLandscape(fork('values'), 6, 1.6);
      const panel = makePanel(6, 10, 20, 20);
      const center = pointer ? clampPoint(pointer, 24) : ([46, 50] as Point);
      const ring = createBlob(fork('shape'), center[0], center[1], 24, 12, 0.35);
      let weightSum = 0;
      let weighted = 0;
      let hardSum = 0;
      let hardCount = 0;
      const items = paintPanel(panel, (u, v, column, row) => {
        let inside = 0;
        for (let sy = 0; sy < 3; sy++) {
          for (let sx = 0; sx < 3; sx++) {
            if (
              isInsideRing(
                [panel.x + (column + (sx + 0.5) / 3) * 4, panel.y + (row + (sy + 0.5) / 3) * 4],
                ring
              )
            )
              inside++;
          }
        }
        const weight = inside / 9;
        if (weight === 0) return {role: 'cool', alpha: 0.06};
        const value = readField(field, u, v);
        weightSum += weight;
        weighted += weight * value;
        if (weight >= 0.5) {
          hardSum += value;
          hardCount++;
        }
        return {
          role: weight < 1 ? 'warm' : 'primary',
          alpha: 0.2 + value * 0.7 * (0.3 + 0.7 * weight),
          phase: weight
        };
      });
      const weightedMean = weighted / (weightSum || 1);
      const hardMean = hardSum / (hardCount || 1);
      return [
        createFillLayer(items, 'pulse'),
        createStaticLayer([
          ...toPaths([[...ring, ring[0]]], 'ink', 0.5, 0.8),
          {
            points: [
              [100, 30],
              [100, 84]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.6
          }
        ]),
        // Two means: the hard-edged one (cell centre inside) against the coverage-weighted one.
        createFlowLayer(
          [
            {
              points: [
                [100, 44],
                [100 + hardMean * 52, 44]
              ],
              role: 'cool',
              width: 5,
              alpha: 0.85
            },
            {
              points: [
                [100, 66],
                [100 + weightedMean * 52, 66]
              ],
              role: 'warm',
              width: 5,
              alpha: 0.95
            }
          ],
          0.3,
          0.8
        )
      ];
    }
  },

  'fractional-zonal-distribution': {
    parameters: [
      {
        id: 'quantile',
        label: 'Quantile',
        min: 0.05,
        max: 0.95,
        step: 0.05,
        default: 0.5
      }
    ],
    build: ({fork, p}) => {
      const random = fork('values');
      const samples: {value: number; weight: number}[] = Array.from({length: 240}, () => ({
        value: clamp(0.42 + random.gaussian() * 0.14 + (random.next() < 0.2 ? 0.25 : 0), 0, 0.999),
        weight: random.next() < 0.3 ? random.range(0.15, 0.9) : 1
      }));
      const bins = 24;
      const counts = new Array<number>(bins).fill(0);
      let total = 0;
      let mean = 0;
      for (const {value, weight} of samples) {
        counts[Math.floor(value * bins)] += weight;
        total += weight;
        mean += value * weight;
      }
      mean /= total;
      let variance = 0;
      for (const {value, weight} of samples) variance += weight * (value - mean) ** 2;
      const deviation = Math.sqrt(variance / total);
      const sorted = [...samples].sort((first, second) => first.value - second.value);
      const getQuantile = (fraction: number) => {
        let running = 0;
        for (const {value, weight} of sorted) {
          running += weight;
          if (running >= fraction * total) return value;
        }
        return 1;
      };
      const tallest = Math.max(...counts);
      const toX = (value: number) => 26 + value * 124;
      const bars: PolygonItem[] = counts.map((count, index) => ({
        ring: createBarPolygon(
          26 + index * (124 / bins) + 0.4,
          88,
          124 / bins - 0.8,
          2 + (count / tallest) * 64
        ),
        role: 'primary',
        alpha: 0.55,
        phase: index / bins
      }));
      const selected = getQuantile(p.quantile);
      return [
        createFillLayer([
          {
            ring: [
              [toX(mean - deviation), 20],
              [toX(mean + deviation), 20],
              [toX(mean + deviation), 88],
              [toX(mean - deviation), 88]
            ],
            role: 'cool',
            alpha: 0.12
          }
        ]),
        createFillLayer(bars, 'pulse'),
        createStaticLayer([
          {
            points: [
              [24, 88],
              [152, 88]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          },
          ...[0.1, 0.5, 0.9].map(
            (fraction): PathItem => ({
              points: [
                [toX(getQuantile(fraction)), 14],
                [toX(getQuantile(fraction)), 88]
              ],
              role: 'dim',
              width: 0.4,
              alpha: 0.6
            })
          )
        ]),
        createFlowLayer(
          [
            {
              points: [
                [toX(selected), 8],
                [toX(selected), 88]
              ],
              role: 'warm',
              width: 1.4,
              alpha: 1
            }
          ],
          0.5
        ),
        {
          type: 'sprites',
          items: [createNode(toX(mean), 12, 'cool', 2.4), createNode(toX(selected), 8, 'warm', 3)],
          pulse: 0.4
        }
      ];
    }
  },

  'fractional-category-composition': {
    parameters: [{id: 'classes', label: 'Classes', min: 2, max: 5, step: 1, default: 4}],
    build: ({fork, p}) => {
      const classes = Math.round(p.classes);
      const field = createLandscape(fork('mosaic'), 7, 2);
      const centers = createZoneCenters(fork('zones'), 3);
      const panel = makePanel(8, 10, 17, 20);
      const shares = Array.from({length: centers.length}, () => new Array<number>(classes).fill(0));
      const items = paintPanel(panel, (u, v) => {
        const bin = Math.min(classes - 1, Math.floor(readField(field, u, v) * classes));
        const zone = getZoneIndex(centers, u, v);
        shares[zone][bin]++;
        return {
          role: getClassRole(bin, classes),
          alpha: 0.35 + 0.15 * zone,
          phase: zone / 3
        };
      });
      const bars: PolygonItem[] = [];
      shares.forEach((counts, zone) => {
        const total = counts.reduce((sum, count) => sum + count, 0) || 1;
        let top = 88;
        counts.forEach((count, bin) => {
          const height = (count / total) * 72;
          if (height > 0.4)
            bars.push({
              ring: createBarPolygon(88 + zone * 22, top, 15, height - 0.6),
              role: getClassRole(bin, classes),
              alpha: 0.85,
              phase: bin / classes
            });
          top -= height;
        });
      });
      return [
        createFillLayer(items),
        createFillLayer(bars, 'pulse'),
        createStaticLayer([
          getPanelOutline(panel, 'dim', 0.55),
          {
            points: [
              [84, 88],
              [158, 88]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          },
          {
            points: [
              [84, 16],
              [84, 88]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          }
        ]),
        createFlowLayer([createLink([72, 50], [82, 50], 'warm')], 0.3)
      ];
    }
  },

  // Sampling and enhancement ------------------------------------------------------------------

  'raster-sampling': {
    pointer: 'hover',
    parameters: [
      {
        id: 'method',
        label: 'Interpolation',
        min: 0,
        max: 2,
        step: 1,
        default: 1,
        options: ['nearest', 'bilinear', 'bicubic']
      }
    ],
    build: ({fork, p, pointer}) => {
      const method = Math.round(p.method);
      const columns = 16;
      const rows = 10;
      const coarse = createSmoothField(fork('surface'), 7, 1.7, columns, rows);
      const values =
        method === 0
          ? coarse
          : createSmoothField(fork('surface'), 7, 1.7, FIELD_COLUMNS, FIELD_ROWS);
      const probe = pointer ?? ([78, 46] as Point);
      const taps: Point[] = [];
      const cellWidth = 160 / columns;
      const cellHeight = 100 / rows;
      const span = method === 0 ? 0 : method === 1 ? 1 : 2;
      const baseColumn = Math.floor(probe[0] / cellWidth - 0.5);
      const baseRow = Math.floor(probe[1] / cellHeight - 0.5);
      if (method === 0)
        taps.push([
          (Math.floor(probe[0] / cellWidth) + 0.5) * cellWidth,
          (Math.floor(probe[1] / cellHeight) + 0.5) * cellHeight
        ]);
      else {
        for (let dy = 1 - span; dy <= span; dy++) {
          for (let dx = 1 - span; dx <= span; dx++)
            taps.push([(baseColumn + dx + 0.5) * cellWidth, (baseRow + dy + 0.5) * cellHeight]);
        }
      }
      const random = fork('points');
      const samplePoints = Array.from(
        {length: 12},
        () => [random.range(10, 150), random.range(10, 90)] as Point
      );
      return [
        {
          type: 'field',
          columns: method === 0 ? columns : FIELD_COLUMNS,
          rows: method === 0 ? rows : FIELD_ROWS,
          values,
          ramp: 'terrain',
          style: 'ramp',
          filter: method === 0 ? 'nearest' : 'linear',
          strength: 0.05
        },
        createFlowLayer(
          taps.map(tap => createLink(tap, probe, 'warm', 0.8)),
          0.4,
          0.8
        ),
        {
          type: 'sprites',
          items: [
            ...samplePoints.map(
              (point): SpriteItem => ({
                x: point[0],
                y: point[1],
                radius:
                  1.2 +
                  2.6 *
                    (method === 0
                      ? coarse[
                          Math.min(rows - 1, Math.floor(point[1] / cellHeight)) * columns +
                            Math.min(columns - 1, Math.floor(point[0] / cellWidth))
                        ]
                      : readField(values, point[0] / 160, point[1] / 100)),
                role: 'ink',
                alpha: 0.75,
                phase: random.next()
              })
            ),
            ...taps.map(
              (tap): SpriteItem => ({
                x: tap[0],
                y: tap[1],
                radius: 1.4,
                role: 'warm',
                alpha: 0.9
              })
            ),
            createNode(probe[0], probe[1], 'hot', 3.2)
          ],
          pulse: 0.35
        }
      ];
    }
  },

  'raster-profile': {
    parameters: [
      {
        id: 'spacing',
        label: 'Sample spacing',
        min: 3,
        max: 16,
        step: 1,
        default: 7,
        unit: 'px'
      }
    ],
    build: ({fork, p}) => {
      const surface = createSmoothField(fork('surface'), 8, 2);
      const path = smoothPath(
        [
          [12, 18],
          [48, 40],
          [88, 22],
          [126, 44],
          [148, 30]
        ],
        3
      );
      const lengths: number[] = [0];
      for (let index = 1; index < path.length; index++)
        lengths.push(lengths[index - 1] + getDistance(path[index - 1], path[index]));
      const total = lengths[lengths.length - 1];
      const samplesOnPath: {point: Point; distance: number}[] = [];
      for (let distance = 0; distance <= total; distance += p.spacing) {
        const index = lengths.findIndex(length => length >= distance);
        samplesOnPath.push({point: path[Math.max(0, index)], distance});
      }
      const elevation = samplesOnPath.map(({point}) =>
        readField(surface, point[0] / 160, point[1] / 100)
      );
      const profile: Point[] = samplesOnPath.map(({distance}, index) => [
        12 + (distance / total) * 136,
        92 - elevation[index] * 26
      ]);
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: surface,
          ramp: 'terrain',
          style: 'hillshade',
          strength: 4,
          light: -2.3
        },
        createFlowLayer(toPaths([path], 'warm', 1.1), 0.25),
        {
          type: 'sprites',
          items: samplesOnPath.map(
            ({point}, index): SpriteItem => ({
              x: point[0],
              y: point[1],
              radius: 1.2,
              role: 'ink',
              alpha: 0.9,
              phase: index / samplesOnPath.length
            })
          ),
          pulse: 0.3
        },
        createStaticLayer([
          {
            points: [
              [12, 94],
              [148, 94]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.7
          }
        ]),
        createFlowLayer(toPaths([profile], 'primary', 1.1), 0.35, 1.4),
        {
          type: 'sprites',
          items: profile.map(
            ([x, y], index): SpriteItem => ({
              x,
              y,
              radius: 1.1,
              role: 'primary',
              alpha: 0.9,
              phase: index / profile.length
            })
          ),
          pulse: 0.3
        }
      ];
    }
  },

  'contrast-stretch': {
    parameters: [
      {
        id: 'clip',
        label: 'Clip percentile',
        min: 0,
        max: 20,
        step: 1,
        default: 4,
        unit: '%'
      }
    ],
    build: ({fork, p}) => {
      const random = fork('values');
      const bins = 36;
      const values = Array.from({length: 600}, () =>
        clamp(0.5 + random.gaussian() * 0.07, 0, 0.999)
      );
      const sorted = [...values].sort((first, second) => first - second);
      const low = sorted[Math.floor((p.clip / 100) * (sorted.length - 1))];
      const high = sorted[Math.floor((1 - p.clip / 100) * (sorted.length - 1))];
      const makeBars = (
        mapValue: (value: number) => number,
        left: number,
        width: number,
        bottom: number,
        height: number,
        role: Role
      ): PolygonItem[] => {
        const counts = new Array<number>(bins).fill(0);
        for (const value of values) counts[Math.floor(clamp(mapValue(value), 0, 0.999) * bins)]++;
        const tallest = Math.max(...counts);
        return counts.map((count, index) => ({
          ring: createBarPolygon(
            left + (index * width) / bins + 0.3,
            bottom,
            width / bins - 0.6,
            1 + (count / tallest) * height
          ),
          role,
          alpha: 0.7,
          phase: index / bins
        }));
      };
      const beforeLeft = 8;
      const beforeWidth = 144;
      const toX = (value: number) => beforeLeft + value * beforeWidth;
      return [
        createFillLayer(makeBars(value => value, beforeLeft, beforeWidth, 44, 34, 'cool')),
        createFillLayer(
          makeBars(
            value => (value - low) / (high - low + 1e-6),
            beforeLeft,
            beforeWidth,
            92,
            34,
            'primary'
          ),
          'pulse'
        ),
        createStaticLayer([
          {
            points: [
              [8, 44],
              [152, 44]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.7
          },
          {
            points: [
              [8, 92],
              [152, 92]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.7
          },
          {
            points: [
              [toX(low), 8],
              [toX(low), 46]
            ],
            role: 'warm',
            width: 0.8,
            alpha: 0.9
          },
          {
            points: [
              [toX(high), 8],
              [toX(high), 46]
            ],
            role: 'warm',
            width: 0.8,
            alpha: 0.9
          }
        ]),
        createFlowLayer(
          [
            createLink([toX(low), 46], [8, 56], 'warm', 0.8),
            createLink([toX(high), 46], [152, 56], 'warm', 0.8)
          ],
          0.35
        )
      ];
    }
  },

  // Focal and morphology ----------------------------------------------------------------------

  'rank-filter': {
    pointer: 'hover',
    parameters: [
      {
        id: 'rank',
        label: 'Rank',
        min: 0,
        max: 2,
        step: 1,
        default: 1,
        options: ['min', 'median', 'max']
      }
    ],
    build: ({fork, p, pointer}) => {
      const noise = fork('speckle');
      const noisy = makePanel(6, 10, 15, 20);
      const filtered = makePanel(94, 10, 15, 20);
      const columns = noisy.columns;
      const rows = noisy.rows;
      const rank = Math.round(p.rank);
      const field = createLandscape(fork('scene'), 4, 1.1);
      const values = new Float32Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const base =
            readField(field, (column + 0.5) / columns, (row + 0.5) / rows) > 0.5 ? 0.8 : 0.25;
          const speckle = noise.next();
          values[row * columns + column] = speckle < 0.08 ? 1 : speckle > 0.92 ? 0 : base;
        }
      }
      const windowValues = (column: number, row: number): number[] => {
        const result: number[] = [];
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const x = clamp(column + dx, 0, columns - 1);
            const y = clamp(row + dy, 0, rows - 1);
            result.push(values[y * columns + x]);
          }
        }
        return result.sort((first, second) => first - second);
      };
      const pick = (column: number, row: number) => windowValues(column, row)[[0, 4, 8][rank]];
      const selected = getPanelCell(noisy, pointer) ?? ([8, 9] as [number, number]);
      const sorted = windowValues(selected[0], selected[1]);
      const windowRing = [
        ...getCellRing(noisy, selected[0] - 1, selected[1] - 1).slice(0, 1),
        getCellRing(noisy, selected[0] + 1, selected[1] - 1)[1],
        getCellRing(noisy, selected[0] + 1, selected[1] + 1)[2],
        getCellRing(noisy, selected[0] - 1, selected[1] + 1)[3],
        getCellRing(noisy, selected[0] - 1, selected[1] - 1)[0]
      ];
      const sortedBars: PolygonItem[] = sorted.map((value, index) => ({
        ring: createBarPolygon(69 + index * 2.4, 70, 1.8, 3 + value * 34),
        role: index === [0, 4, 8][rank] ? 'warm' : 'cool',
        alpha: index === [0, 4, 8][rank] ? 1 : 0.55,
        phase: index / 9
      }));
      return [
        createFillLayer(
          paintPanel(noisy, (_u, _v, column, row) =>
            makeCell('cool', values[row * columns + column], 0.08)
          )
        ),
        createFillLayer(
          paintPanel(filtered, (_u, _v, column, row) => ({
            ...makeCell('primary', pick(column, row), 0.08),
            phase: column / columns
          })),
          'pulse'
        ),
        createFillLayer(sortedBars, 'pulse'),
        createStaticLayer([
          getPanelOutline(noisy, 'cool', 0.5),
          getPanelOutline(filtered, 'primary', 0.55),
          {
            points: [
              [68, 70],
              [92, 70]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.7
          }
        ]),
        createFlowLayer(
          [
            {points: windowRing, role: 'warm', width: 1, alpha: 1},
            createLink([68, 54], [80, 36], 'warm', 0.5),
            createLink([80, 36], [92, 30], 'warm', 0.5)
          ],
          0.4,
          1.2
        )
      ];
    }
  },

  'focal-correlation-regression': {
    pointer: 'hover',
    parameters: [
      {
        id: 'window',
        label: 'Window',
        min: 1,
        max: 3,
        step: 1,
        default: 2,
        unit: 'cells'
      }
    ],
    build: ({fork, p, pointer}) => {
      const fieldA = createLandscape(fork('a'), 5, 1.4);
      const slopeField = createLandscape(fork('relation'), 3, 0.9);
      const noise = fork('noise');
      const first = makePanel(6, 6, 8, 6, 4);
      const second = makePanel(6, 40, 8, 6, 4);
      const map = makePanel(94, 10, 15, 18);
      const columns = map.columns;
      const rows = map.rows;
      const a = new Float32Array(columns * rows);
      const b = new Float32Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const u = (column + 0.5) / columns;
          const v = (row + 0.5) / rows;
          const index = row * columns + column;
          a[index] = readField(fieldA, u, v);
          const relation = (readField(slopeField, u, v) - 0.5) * 3;
          b[index] = 0.5 + relation * (a[index] - 0.5) + noise.gaussian() * 0.12;
        }
      }
      const radius = Math.round(p.window);
      const correlationAt = (column: number, row: number) => {
        let count = 0,
          meanA = 0,
          meanB = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const index =
              clamp(row + dy, 0, rows - 1) * columns + clamp(column + dx, 0, columns - 1);
            meanA += a[index];
            meanB += b[index];
            count++;
          }
        }
        meanA /= count;
        meanB /= count;
        let covariance = 0,
          varianceA = 0,
          varianceB = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const index =
              clamp(row + dy, 0, rows - 1) * columns + clamp(column + dx, 0, columns - 1);
            covariance += (a[index] - meanA) * (b[index] - meanB);
            varianceA += (a[index] - meanA) ** 2;
            varianceB += (b[index] - meanB) ** 2;
          }
        }
        return {
          correlation: covariance / (Math.sqrt(varianceA * varianceB) + 1e-6),
          slope: covariance / (varianceA + 1e-6),
          meanA,
          meanB
        };
      };
      const selected = getPanelCell(map, pointer) ?? ([4, 9] as [number, number]);
      const target = correlationAt(selected[0], selected[1]);
      const scatter: SpriteItem[] = [];
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const index =
            clamp(selected[1] + dy, 0, rows - 1) * columns +
            clamp(selected[0] + dx, 0, columns - 1);
          scatter.push({
            x: 46 + clamp(a[index], 0, 1) * 34,
            y: 84 - clamp(b[index], 0, 1) * 34,
            radius: 1.5,
            role: 'ink',
            alpha: 0.9,
            phase: scatter.length / 25
          });
        }
      }
      const fitStart: Point = [
        46,
        84 - clamp(target.meanB - target.slope * target.meanA, 0, 1) * 34
      ];
      const fitEnd: Point = [
        80,
        84 - clamp(target.meanB + target.slope * (1 - target.meanA), 0, 1.2) * 34
      ];
      return [
        createFillLayer(
          paintPanel(first, (_u, _v, column, row) =>
            makeCell('cool', a[row * 2 * columns + Math.min(columns - 1, column * 2)], 0.1)
          )
        ),
        createFillLayer(
          paintPanel(second, (_u, _v, column, row) =>
            makeCell('violet', b[row * 2 * columns + Math.min(columns - 1, column * 2)], 0.1)
          )
        ),
        createFillLayer(
          paintPanel(map, (_u, _v, column, row) => {
            const {correlation} = correlationAt(column, row);
            return {
              role: correlation >= 0 ? 'hot' : 'cool',
              alpha: 0.12 + 0.8 * Math.abs(correlation),
              phase: column / columns
            };
          }),
          'pulse'
        ),
        createFillLayer(
          [
            {
              ring: getCellRing(map, selected[0], selected[1]),
              role: 'ink',
              alpha: 0.9
            }
          ],
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(first, 'cool', 0.5),
          getPanelOutline(second, 'violet', 0.5),
          getPanelOutline(map, 'dim', 0.6),
          {
            points: [
              [44, 84],
              [44, 50],
              [44, 84],
              [82, 84]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.7
          }
        ]),
        createFlowLayer(
          [
            {
              points: [fitStart, fitEnd],
              role: target.correlation >= 0 ? 'hot' : 'cool',
              width: 1.1,
              alpha: 0.95
            },
            createLink(getCellCenter(map, selected[0], selected[1]), [82, 66], 'warm', 0.5)
          ],
          0.35
        ),
        {type: 'sprites', items: scatter, pulse: 0.3}
      ];
    }
  },

  'glcm-texture': {
    pointer: 'hover',
    parameters: [
      {
        id: 'distance',
        label: 'Pixel distance',
        min: 1,
        max: 3,
        step: 1,
        default: 1
      }
    ],
    build: ({fork, p, pointer}) => {
      const smooth = createLandscape(fork('smooth'), 3, 0.7);
      const image = makePanel(8, 10, 18, 20);
      const levels = 8;
      const distance = Math.round(p.distance);
      const bandWidth = image.columns / 3;
      const level = (column: number, row: number): number => {
        const band = Math.floor(column / bandWidth);
        // Same average brightness in each band; only the spatial arrangement differs.
        if (band === 0)
          return (
            readField(smooth, (column + 0.5) / image.columns, (row + 0.5) / image.rows) * 0.3 + 0.35
          );
        if (band === 1) return Math.floor(column / 2) % 2 === 0 ? 0.85 : 0.15;
        return (column + row) % 2 === 0 ? 0.85 : 0.15;
      };
      const hovered = getPanelCell(image, pointer);
      const selectedBand = hovered ? Math.floor(hovered[0] / bandWidth) : 1;
      const counts = new Float32Array(levels * levels);
      for (let row = 0; row < image.rows; row++) {
        for (
          let column = Math.floor(selectedBand * bandWidth);
          column < Math.floor((selectedBand + 1) * bandWidth) - distance;
          column++
        ) {
          const from = Math.min(levels - 1, Math.floor(level(column, row) * levels));
          const to = Math.min(levels - 1, Math.floor(level(column + distance, row) * levels));
          counts[from * levels + to]++;
          counts[to * levels + from]++;
        }
      }
      const peak = Math.max(...counts, 1);
      const matrix = makePanel(88, 18, levels, levels, 8);
      return [
        createFillLayer(
          paintPanel(image, (_u, _v, column, row) => makeCell('cool', level(column, row), 0.05))
        ),
        createFillLayer(
          paintPanel(matrix, (_u, _v, column, row) => {
            const share = counts[row * levels + column] / peak;
            return share > 0.02
              ? {
                  ...makeCell(getValueRole(share), Math.sqrt(share), 0.3),
                  phase: (column + row) / 16
                }
              : {role: 'dim', alpha: 0.08};
          }),
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(image, 'dim', 0.5),
          getPanelOutline(matrix, 'dim', 0.6),
          {
            points: getCellRing(image, Math.floor(selectedBand * bandWidth), 0)
              .slice(0, 1)
              .concat([
                [image.x + (selectedBand + 1) * bandWidth * 4, image.y],
                [image.x + (selectedBand + 1) * bandWidth * 4, image.y + 80],
                [image.x + selectedBand * bandWidth * 4, image.y + 80],
                [image.x + selectedBand * bandWidth * 4, image.y]
              ]),
            role: 'warm',
            width: 1,
            alpha: 1
          }
        ]),
        createFlowLayer(
          [createLink([image.x + (selectedBand + 1) * bandWidth * 4, 50], [86, 50], 'warm')],
          0.35
        )
      ];
    }
  },

  'morphological-reconstruction': {
    parameters: [{id: 'rounds', label: 'Rounds', min: 0, max: 16, step: 1, default: 10}],
    build: ({fork, p}) => {
      const field = createLandscape(fork('mask'), 7, 1.7);
      const panel = FULL_PANEL;
      const columns = panel.columns;
      const rows = panel.rows;
      const mask = new Uint8Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++)
          mask[row * columns + column] =
            readField(field, (column + 0.5) / columns, (row + 0.5) / rows) > 0.46 ? 1 : 0;
      }
      const distances = new Int32Array(columns * rows).fill(-1);
      const queue: number[] = [];
      const labels = labelComponents(
        Int8Array.from(mask, value => (value ? 0 : -1)),
        columns,
        rows
      );
      // One marker per second component, so some regions stay unreached.
      const seeded = new Set<number>();
      const seeds: number[] = [];
      for (let index = 0; index < mask.length; index++) {
        const label = labels.labels[index];
        if (label >= 0 && label % 2 === 0 && !seeded.has(label) && labels.sizes[label] > 4) {
          seeded.add(label);
          seeds.push(index);
          distances[index] = 0;
          queue.push(index);
        }
      }
      for (let head = 0; head < queue.length; head++) {
        const index = queue[head];
        const column = index % columns;
        const row = (index - column) / columns;
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1]
        ]) {
          const x = column + dx;
          const y = row + dy;
          if (x < 0 || y < 0 || x >= columns || y >= rows) continue;
          const next = y * columns + x;
          if (mask[next] && distances[next] < 0) {
            distances[next] = distances[index] + 1;
            queue.push(next);
          }
        }
      }
      const rounds = Math.round(p.rounds);
      return [
        createFillLayer(
          paintPanel(panel, (_u, _v, column, row) => {
            const index = row * columns + column;
            if (!mask[index]) return null;
            const distance = distances[index];
            if (distance >= 0 && distance <= rounds)
              return distance >= rounds - 1 && rounds > 0
                ? {role: 'hot', alpha: 0.95, phase: 0}
                : {role: 'warm', alpha: 0.7, phase: distance / 16};
            return {role: 'cool', alpha: 0.28, phase: 0};
          }),
          'pulse'
        ),
        {
          type: 'sprites',
          items: seeds.map(index =>
            createNode(
              getCellCenter(panel, index % columns, Math.floor(index / columns))[0],
              getCellCenter(panel, index % columns, Math.floor(index / columns))[1],
              'ink',
              2.4
            )
          ),
          pulse: 0.5
        }
      ];
    }
  },

  'fill-holes': {
    parameters: [
      {
        id: 'connectivity',
        label: 'Connectivity',
        min: 0,
        max: 1,
        step: 1,
        default: 0,
        options: ['4-way', '8-way']
      }
    ],
    build: ({fork, p}) => {
      const field = createLandscape(fork('mask'), 9, 2.4);
      const panel = FULL_PANEL;
      const columns = panel.columns;
      const rows = panel.rows;
      const land = new Uint8Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++)
          land[row * columns + column] =
            readField(field, (column + 0.5) / columns, (row + 0.5) / rows) > 0.3 ? 1 : 0;
      }
      // Ponds carved into the land: the ones the sea cannot reach are the holes to fill.
      const ponds = fork('ponds');
      for (let pond = 0; pond < 9; pond++) {
        const pondColumn = ponds.range(4, columns - 4);
        const pondRow = ponds.range(3, rows - 3);
        const pondRadius = ponds.range(1.4, 3.4);
        for (let row = 0; row < rows; row++) {
          for (let column = 0; column < columns; column++) {
            if (Math.hypot(column - pondColumn, (row - pondRow) * 1.1) < pondRadius)
              land[row * columns + column] = 0;
          }
        }
      }
      // Flood the background from the raster edge: whatever it reaches is open water.
      const open = new Uint8Array(columns * rows);
      const stack: number[] = [];
      const push = (column: number, row: number) => {
        const index = row * columns + column;
        if (!land[index] && !open[index]) {
          open[index] = 1;
          stack.push(index);
        }
      };
      for (let column = 0; column < columns; column++) {
        push(column, 0);
        push(column, rows - 1);
      }
      for (let row = 0; row < rows; row++) {
        push(0, row);
        push(columns - 1, row);
      }
      const offsets =
        p.connectivity > 0.5
          ? [
              [1, 0],
              [-1, 0],
              [0, 1],
              [0, -1],
              [1, 1],
              [-1, -1],
              [1, -1],
              [-1, 1]
            ]
          : [
              [1, 0],
              [-1, 0],
              [0, 1],
              [0, -1]
            ];
      while (stack.length > 0) {
        const index = stack.pop() as number;
        const column = index % columns;
        const row = (index - column) / columns;
        for (const [dx, dy] of offsets) {
          if (column + dx >= 0 && column + dx < columns && row + dy >= 0 && row + dy < rows)
            push(column + dx, row + dy);
        }
      }
      return [
        createFillLayer(
          paintPanel(panel, (_u, _v, column, row) => {
            const index = row * columns + column;
            if (land[index]) return {role: 'primary', alpha: 0.6, phase: 0};
            if (open[index]) return {role: 'cool', alpha: 0.12, phase: 0};
            return {role: 'warm', alpha: 0.95, phase: (column + row) / 50};
          }),
          'pulse'
        )
      ];
    }
  },

  'fill-nodata': {
    pointer: 'hover',
    parameters: [
      {
        id: 'radius',
        label: 'Search radius',
        min: 1,
        max: 8,
        step: 1,
        default: 4,
        unit: 'cells'
      }
    ],
    build: ({fork, p, pointer}) => {
      const field = createLandscape(fork('values'), 6, 1.3);
      const panel = FULL_PANEL;
      const columns = panel.columns;
      const rows = panel.rows;
      const holes: Point[] = [
        [8, 6, 4],
        [22, 14, 5],
        [14, 15, 3],
        [26, 5, 3]
      ].map(([c, r]) => [c, r] as Point);
      const radii = [4, 5, 3, 3];
      const movable = pointer ? getPanelCell(panel, pointer) : null;
      if (movable) holes[0] = [movable[0], movable[1]];
      const isHole = (column: number, row: number) =>
        holes.some(([hc, hr], index) => Math.hypot(column - hc, (row - hr) * 1.1) < radii[index]);
      const searchRadius = Math.round(p.radius);
      return [
        createFillLayer(
          paintPanel(panel, (u, v, column, row) => {
            if (!isHole(column, row)) return makeCell('cool', readField(field, u, v), 0.1);
            let weightSum = 0;
            let total = 0;
            for (let dy = -searchRadius; dy <= searchRadius; dy++) {
              for (let dx = -searchRadius; dx <= searchRadius; dx++) {
                const x = column + dx;
                const y = row + dy;
                const distance = Math.hypot(dx, dy);
                if (
                  distance === 0 ||
                  distance > searchRadius ||
                  x < 0 ||
                  y < 0 ||
                  x >= columns ||
                  y >= rows ||
                  isHole(x, y)
                )
                  continue;
                const weight = 1 / (distance * distance);
                weightSum += weight;
                total += weight * readField(field, (x + 0.5) / columns, (y + 0.5) / rows);
              }
            }
            if (weightSum === 0) return {role: 'hot', alpha: 0.16};
            return {
              ...makeCell('warm', total / weightSum, 0.2),
              phase: (column + row) / 50
            };
          }),
          'pulse'
        )
      ];
    }
  },

  // Patches and shapes ------------------------------------------------------------------------

  'patch-metrics': {
    pointer: 'hover',
    parameters: [
      {
        id: 'detail',
        label: 'Patch detail',
        min: 3,
        max: 8,
        step: 1,
        default: 5
      }
    ],
    build: ({fork, p, pointer}) => {
      const field = createLandscape(fork('patches'), 7, 1 + p.detail * 0.28);
      const panel = makePanel(16, 6, 32, 18);
      const columns = panel.columns;
      const rows = panel.rows;
      const classes = Int8Array.from({length: columns * rows}, (_, index) => {
        const value = readField(
          field,
          ((index % columns) + 0.5) / columns,
          (Math.floor(index / columns) + 0.5) / rows
        );
        return value > 0.55 ? 1 : -1;
      });
      const {labels, sizes} = labelComponents(classes, columns, rows);
      const hovered = getPanelCell(panel, pointer);
      let selected = hovered ? labels[hovered[1] * columns + hovered[0]] : -1;
      if (selected < 0) selected = sizes.indexOf(Math.max(...sizes));
      const isEdge = (column: number, row: number, label: number) =>
        [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1]
        ].some(([dx, dy]) => {
          const x = column + dx;
          const y = row + dy;
          return x < 0 || y < 0 || x >= columns || y >= rows || labels[y * columns + x] !== label;
        });
      let perimeter = 0;
      const items = paintPanel(panel, (_u, _v, column, row) => {
        const label = labels[row * columns + column];
        if (label < 0) return null;
        const edge = isEdge(column, row, label);
        if (label === selected && edge) perimeter++;
        const active = label === selected;
        return {
          role: active ? 'warm' : getRoleCycle(label),
          alpha: edge ? (active ? 1 : 0.7) : active ? 0.55 : 0.22,
          phase: label / 10
        };
      });
      const area = sizes[selected] ?? 0;
      return [
        createFillLayer(items, 'pulse'),
        createFlowLayer(
          [
            {
              points: [
                [16, 90],
                [16 + Math.min(1, area / 220) * 120, 90]
              ],
              role: 'primary',
              width: 2,
              alpha: 0.95
            },
            {
              points: [
                [16, 96],
                [16 + Math.min(1, perimeter / 120) * 120, 96]
              ],
              role: 'hot',
              width: 2,
              alpha: 0.95
            }
          ],
          0.3,
          0.8
        )
      ];
    }
  },

  sieve: {
    parameters: [
      {
        id: 'minimum',
        label: 'Minimum size',
        min: 1,
        max: 40,
        step: 1,
        default: 12,
        unit: 'cells'
      }
    ],
    build: ({fork, p}) => {
      const field = createLandscape(fork('classes'), 6, 1.3);
      const noise = fork('speckle');
      const panel = FULL_PANEL;
      const columns = panel.columns;
      const rows = panel.rows;
      const classes = Int8Array.from({length: columns * rows}, (_, index) => {
        const value = readField(
          field,
          ((index % columns) + 0.5) / columns,
          (Math.floor(index / columns) + 0.5) / rows
        );
        const base = Math.min(2, Math.floor(value * 3));
        return noise.next() < 0.1 ? Math.floor(noise.next() * 3) : base;
      });
      const {labels, sizes} = labelComponents(classes, columns, rows);
      const merged = Int8Array.from(classes);
      for (let index = 0; index < classes.length; index++) {
        if (sizes[labels[index]] >= p.minimum) continue;
        const column = index % columns;
        const row = (index - column) / columns;
        const votes = [0, 0, 0];
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1]
        ]) {
          const x = column + dx;
          const y = row + dy;
          if (
            x >= 0 &&
            y >= 0 &&
            x < columns &&
            y < rows &&
            sizes[labels[y * columns + x]] >= p.minimum
          )
            votes[classes[y * columns + x]]++;
        }
        const best = votes.indexOf(Math.max(...votes));
        if (votes[best] > 0) merged[index] = best;
      }
      return [
        createFillLayer(
          paintPanel(panel, (_u, _v, column, row) => {
            const index = row * columns + column;
            const changed = merged[index] !== classes[index];
            const small = sizes[labels[index]] < p.minimum;
            if (small && !changed) return {role: 'hot', alpha: 0.9, phase: 0};
            return changed
              ? {role: 'ink', alpha: 0.95, phase: 0.5}
              : {role: getClassRole(merged[index], 3), alpha: 0.5, phase: 0};
          }),
          'pulse'
        )
      ];
    }
  },

  skeleton: {
    parameters: [
      {
        id: 'thickness',
        label: 'Thickness',
        min: 2,
        max: 12,
        step: 0.5,
        default: 7,
        unit: 'px'
      }
    ],
    build: ({fork, p}) => {
      const random = fork('shape');
      const trunk = smoothPath(
        createWanderPath(random, [20, 56], -0.15, 11, 8, 0.3).map(point => clampPoint(point, 14)),
        2
      );
      const joint = trunk[Math.floor(trunk.length * 0.45)];
      const branch = smoothPath(
        [
          joint,
          [joint[0] + 14, joint[1] - 16],
          [joint[0] + 30, joint[1] - 28],
          [Math.min(146, joint[0] + 42), joint[1] - 32]
        ],
        2
      );
      const spine = [trunk, branch];
      const half = p.thickness / 2;
      const panel = FULL_PANEL;
      const distanceToSpine = (point: Point) => {
        let best = Infinity;
        for (const path of spine) {
          for (let index = 0; index + 1 < path.length; index++) {
            const dx = path[index + 1][0] - path[index][0];
            const dy = path[index + 1][1] - path[index][1];
            const lengthSquared = dx * dx + dy * dy || 1;
            const amount = clamp(
              ((point[0] - path[index][0]) * dx + (point[1] - path[index][1]) * dy) / lengthSquared,
              0,
              1
            );
            best = Math.min(
              best,
              Math.hypot(
                point[0] - path[index][0] - dx * amount,
                point[1] - path[index][1] - dy * amount
              )
            );
          }
        }
        return best;
      };
      return [
        createFillLayer(
          paintPanel(panel, (_u, _v, column, row) => {
            const distance = distanceToSpine(getCellCenter(panel, column, row));
            return distance < half
              ? {
                  role: 'cool',
                  alpha: 0.18 + 0.2 * (1 - distance / half),
                  phase: 0
                }
              : null;
          })
        ),
        createFlowLayer(toPaths(spine, 'primary', 1.5, 1), 0.35, 1.6),
        {
          type: 'sprites',
          items: [
            createNode(joint[0], joint[1], 'warm', 2.6),
            createNode(trunk[0][0], trunk[0][1], 'warm', 2),
            createNode(branch[branch.length - 1][0], branch[branch.length - 1][1], 'warm', 2),
            createNode(trunk[trunk.length - 1][0], trunk[trunk.length - 1][1], 'warm', 2)
          ],
          pulse: 0.45
        }
      ];
    }
  },

  'medial-axis': {
    pointer: 'hover',
    parameters: [{id: 'taper', label: 'Taper', min: 0, max: 1, step: 0.05, default: 0.6}],
    build: ({fork, p, pointer}) => {
      const axis = smoothPath(
        [
          [18, 62],
          [48, 38],
          [88, 56],
          [124, 36],
          [146, 46]
        ],
        3
      );
      const halfWidthAt = (index: number) => {
        const t = index / (axis.length - 1);
        return 4 + 13 * (0.35 + 0.65 * Math.sin(t * Math.PI * 1.5 + 0.4) ** 2) * (1 - p.taper * t);
      };
      fork('unused');
      const panel = FULL_PANEL;
      const nearestAxis = (point: Point): number => {
        let best = 0;
        let bestDistance = Infinity;
        for (let index = 0; index < axis.length; index++) {
          const distance = getDistance(point, axis[index]);
          if (distance < bestDistance) {
            bestDistance = distance;
            best = index;
          }
        }
        return best;
      };
      const selected = pointer ? nearestAxis(pointer) : Math.floor(axis.length * 0.35);
      const discs = axis
        .map((point, index) => ({point, radius: halfWidthAt(index)}))
        .filter((_, index) => index % 6 === 0 || index === selected);
      return [
        createFillLayer(
          paintPanel(panel, (_u, _v, column, row) => {
            const center = getCellCenter(panel, column, row);
            const index = nearestAxis(center);
            return getDistance(center, axis[index]) < halfWidthAt(index)
              ? {role: 'cool', alpha: 0.22, phase: 0}
              : null;
          })
        ),
        createStaticLayer(
          discs.map(({point, radius}) => ({
            points: createCircle(point[0], point[1], radius, 24).concat([
              [point[0] + radius, point[1]]
            ]),
            role: 'violet' as Role,
            width: 0.4,
            alpha: 0.5
          }))
        ),
        createFlowLayer(
          [
            {
              points: createCircle(
                axis[selected][0],
                axis[selected][1],
                halfWidthAt(selected),
                28
              ).concat([[axis[selected][0] + halfWidthAt(selected), axis[selected][1]]]),
              role: 'warm',
              width: 1,
              alpha: 1
            }
          ],
          0.3
        ),
        createFlowLayer(toPaths([axis], 'primary', 1.4), 0.3, 1.5),
        {
          type: 'sprites',
          items: axis
            .filter((_, index) => index % 4 === 0)
            .map(
              (point, index): SpriteItem => ({
                x: point[0],
                y: point[1],
                radius: 0.8 + halfWidthAt(index * 4) * 0.18,
                role: 'primary',
                alpha: 0.9,
                phase: index / 10
              })
            ),
          pulse: 0.3
        }
      ];
    }
  },

  'region-adjacency-graph': {
    pointer: 'hover',
    parameters: [
      {
        id: 'regions',
        label: 'Regions',
        min: 6,
        max: 24,
        step: 1,
        default: 14
      }
    ],
    build: ({fork, p, pointer}) => {
      const random = fork('regions');
      const count = Math.round(p.regions);
      const centers: Point[] = Array.from({length: count}, () => [
        random.range(0.04, 0.96),
        random.range(0.06, 0.94)
      ]);
      const panel = makePanel(0, 0, 40, 25);
      const columns = panel.columns;
      const rows = panel.rows;
      const owner = new Int16Array(columns * rows);
      const sums = centers.map(() => [0, 0, 0]);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const u = (column + 0.5) / columns;
          const v = (row + 0.5) / rows;
          let best = 0;
          let bestDistance = Infinity;
          centers.forEach(([cx, cy], index) => {
            const distance = (u - cx) ** 2 + ((v - cy) * 0.625) ** 2;
            if (distance < bestDistance) {
              bestDistance = distance;
              best = index;
            }
          });
          owner[row * columns + column] = best;
          sums[best][0] += column;
          sums[best][1] += row;
          sums[best][2]++;
        }
      }
      const edgeKeys = new Set<number>();
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const here = owner[row * columns + column];
          for (const [dx, dy] of [
            [1, 0],
            [0, 1]
          ]) {
            if (column + dx >= columns || row + dy >= rows) continue;
            const there = owner[(row + dy) * columns + column + dx];
            if (there !== here) edgeKeys.add(Math.min(here, there) * 100 + Math.max(here, there));
          }
        }
      }
      const nodes: Point[] = sums.map(([x, y, n]) => [
        (x / (n || 1) + 0.5) * 4,
        (y / (n || 1) + 0.5) * 4
      ]);
      const hovered = pointer
        ? nodes.reduce(
            (best, node, index) =>
              getDistance(node, pointer) < getDistance(nodes[best], pointer) ? index : best,
            0
          )
        : -1;
      const edges = [...edgeKeys].map((key): [number, number] => [
        Math.floor(key / 100),
        key % 100
      ]);
      const quiet = edges
        .filter(([first, second]) => first !== hovered && second !== hovered)
        .map(([first, second]) => [nodes[first], nodes[second]]);
      const active = edges
        .filter(([first, second]) => first === hovered || second === hovered)
        .map(([first, second]) => [nodes[first], nodes[second]]);
      return [
        createFillLayer(
          paintPanel(panel, (_u, _v, column, row) => ({
            role: getRoleCycle(owner[row * columns + column]),
            alpha: owner[row * columns + column] === hovered ? 0.35 : 0.15
          }))
        ),
        createFlowLayer(toPaths(quiet, 'primary', 0.8, 0.8), 0.25),
        createFlowLayer(toPaths(active, 'warm', 1.4, 1), 0.4),
        {
          type: 'sprites',
          items: nodes.map(
            (node, index): SpriteItem => ({
              x: node[0],
              y: node[1],
              radius: index === hovered ? 3.4 : 2.2,
              role: index === hovered ? 'warm' : 'ink',
              alpha: 0.95,
              phase: index / count
            })
          ),
          pulse: 0.35
        }
      ];
    }
  },

  'watershed-segmentation': {
    parameters: [
      {
        id: 'level',
        label: 'Flood level',
        min: 0.05,
        max: 1,
        step: 0.05,
        default: 0.55
      }
    ],
    build: ({fork, p}) => {
      const surface = createLandscape(fork('relief'), 6, 1.5);
      const columns = 40;
      const rows = 25;
      const panel = makePanel(0, 0, columns, rows);
      const heights = createPanelValues(surface, panel);
      const label = new Int16Array(columns * rows).fill(-1);
      const basinOf = (index: number): number => {
        if (label[index] >= 0) return label[index];
        const trail: number[] = [];
        let current = index;
        while (label[current] < 0) {
          trail.push(current);
          const column = current % columns;
          const row = (current - column) / columns;
          let best = current;
          for (const [dx, dy] of [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
          ]) {
            const x = column + dx;
            const y = row + dy;
            if (
              x >= 0 &&
              y >= 0 &&
              x < columns &&
              y < rows &&
              heights[y * columns + x] < heights[best]
            )
              best = y * columns + x;
          }
          if (best === current) {
            label[current] = current;
            break;
          }
          current = best;
        }
        for (const visited of trail) label[visited] = label[current];
        return label[index];
      };
      const owners = Array.from({length: columns * rows}, (_, index) => basinOf(index));
      const seedIds = [...new Set(owners)];
      const minimum = Math.min(...heights);
      const maximum = Math.max(...heights);
      const level = minimum + (maximum - minimum) * p.level;
      const items = paintPanel(panel, (_u, _v, column, row) => {
        const index = row * columns + column;
        if (heights[index] > level) return null;
        const border = [
          [1, 0],
          [0, 1]
        ].some(
          ([dx, dy]) =>
            column + dx < columns &&
            row + dy < rows &&
            owners[(row + dy) * columns + column + dx] !== owners[index] &&
            heights[(row + dy) * columns + column + dx] <= level
        );
        const basin = seedIds.indexOf(owners[index]);
        return border
          ? {role: 'ink', alpha: 0.7, phase: 0}
          : {
              role: getRoleCycle(basin),
              alpha: 0.35 + 0.4 * (1 - (heights[index] - minimum) / (level - minimum + 1e-6)),
              phase: basin / seedIds.length
            };
      });
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: surface,
          ramp: 'terrain',
          style: 'hillshade',
          strength: 3,
          light: -2.2
        },
        createFillLayer(items, 'pulse'),
        {
          type: 'sprites',
          items: seedIds.map(
            (seed, index): SpriteItem => ({
              x: ((seed % columns) + 0.5) * 4,
              y: (Math.floor(seed / columns) + 0.5) * 4,
              radius: 1.8,
              role: 'ink',
              alpha: 0.9,
              phase: index / seedIds.length
            })
          ),
          pulse: 0.5
        }
      ];
    }
  },

  'extrema-pyramid': {
    parameters: [
      {
        id: 'level',
        label: 'Pyramid level',
        min: 1,
        max: 4,
        step: 1,
        default: 2
      }
    ],
    build: ({fork, p}) => {
      const values = createSmoothField(fork('relief'), 9, 2.2);
      const across = 32 / 2 ** Math.round(p.level);
      const down = Math.max(1, Math.round(across * 0.625));
      const blockColumns = FIELD_COLUMNS / across;
      const blockRows = FIELD_ROWS / down;
      const lows: SpriteItem[] = [];
      const highs: SpriteItem[] = [];
      for (let blockRow = 0; blockRow < down; blockRow++) {
        for (let blockColumn = 0; blockColumn < across; blockColumn++) {
          let low = Infinity;
          let high = -Infinity;
          let lowAt = 0;
          let highAt = 0;
          for (
            let row = Math.floor(blockRow * blockRows);
            row < Math.floor((blockRow + 1) * blockRows);
            row++
          ) {
            for (
              let column = Math.floor(blockColumn * blockColumns);
              column < Math.floor((blockColumn + 1) * blockColumns);
              column++
            ) {
              const index = row * FIELD_COLUMNS + column;
              if (values[index] < low) {
                low = values[index];
                lowAt = index;
              }
              if (values[index] > high) {
                high = values[index];
                highAt = index;
              }
            }
          }
          const toPoint = (index: number): Point => [
            ((index % FIELD_COLUMNS) + 0.5) * 1.25,
            (Math.floor(index / FIELD_COLUMNS) + 0.5) * 1.25
          ];
          lows.push({
            x: toPoint(lowAt)[0],
            y: toPoint(lowAt)[1],
            radius: 1.5,
            role: 'cool',
            alpha: 0.95,
            phase: blockColumn / across
          });
          highs.push({
            x: toPoint(highAt)[0],
            y: toPoint(highAt)[1],
            radius: 1.5,
            role: 'hot',
            alpha: 0.95,
            phase: blockRow / down
          });
        }
      }
      const grid: PathItem[] = [];
      for (let column = 1; column < across; column++)
        grid.push({
          points: [
            [(column * 160) / across, 0],
            [(column * 160) / across, 100]
          ],
          role: 'ink',
          width: 0.3,
          alpha: 0.35
        });
      for (let row = 1; row < down; row++)
        grid.push({
          points: [
            [0, (row * 100) / down],
            [160, (row * 100) / down]
          ],
          role: 'ink',
          width: 0.3,
          alpha: 0.35
        });
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values,
          ramp: 'terrain',
          style: 'hillshade',
          strength: 4,
          light: -2.3
        },
        createStaticLayer(grid),
        {type: 'sprites', items: [...lows, ...highs], pulse: 0.35}
      ];
    }
  },

  // Cost, reprojection and time ---------------------------------------------------------------

  'walking-cost': {
    pointer: 'sticky',
    parameters: [
      {
        id: 'budget',
        label: 'Walking budget',
        min: 10,
        max: 100,
        step: 5,
        default: 60,
        unit: '%'
      }
    ],
    build: ({fork, p, pointer}) => {
      const columns = 64;
      const rows = 40;
      const relief = createSmoothField(fork('relief'), 8, 2, columns, rows);
      const cost = new Float32Array(columns * rows);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const dx =
            relief[row * columns + Math.min(columns - 1, column + 1)] -
            relief[row * columns + Math.max(0, column - 1)];
          const dy =
            relief[Math.min(rows - 1, row + 1) * columns + column] -
            relief[Math.max(0, row - 1) * columns + column];
          // Steeper ground is slower to cross.
          cost[row * columns + column] = 1 + 7 * Math.hypot(dx, dy);
        }
      }
      const source = pointer ?? ([36, 52] as Point);
      const start =
        clamp(Math.floor(source[1] / 2.5), 0, rows - 1) * columns +
        clamp(Math.floor(source[0] / 2.5), 0, columns - 1);
      const time = new Float32Array(columns * rows).fill(Infinity);
      const heap: [number, number][] = [[0, start]];
      time[start] = 0;
      const pushHeap = (entry: [number, number]) => {
        heap.push(entry);
        let child = heap.length - 1;
        while (child > 0) {
          const parent = (child - 1) >> 1;
          if (heap[parent][0] <= heap[child][0]) break;
          [heap[parent], heap[child]] = [heap[child], heap[parent]];
          child = parent;
        }
      };
      while (heap.length > 0) {
        const [distance, index] = heap[0];
        const last = heap.pop() as [number, number];
        if (heap.length > 0) {
          heap[0] = last;
          let parent = 0;
          for (;;) {
            const left = parent * 2 + 1;
            const right = left + 1;
            let smallest = parent;
            if (left < heap.length && heap[left][0] < heap[smallest][0]) smallest = left;
            if (right < heap.length && heap[right][0] < heap[smallest][0]) smallest = right;
            if (smallest === parent) break;
            [heap[parent], heap[smallest]] = [heap[smallest], heap[parent]];
            parent = smallest;
          }
        }
        if (distance > time[index]) continue;
        const column = index % columns;
        const row = (index - column) / columns;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (
              (dx === 0 && dy === 0) ||
              column + dx < 0 ||
              column + dx >= columns ||
              row + dy < 0 ||
              row + dy >= rows
            )
              continue;
            const next = (row + dy) * columns + column + dx;
            const step = (dx !== 0 && dy !== 0 ? 1.414 : 1) * 0.5 * (cost[index] + cost[next]);
            if (distance + step < time[next]) {
              time[next] = distance + step;
              pushHeap([time[next], next]);
            }
          }
        }
      }
      const values = Float32Array.from(time, value => Math.min(value, 80) / 80);
      return [
        {
          type: 'field',
          columns: FIELD_COLUMNS,
          rows: FIELD_ROWS,
          values: createSmoothField(fork('relief'), 8, 2),
          ramp: 'terrain',
          style: 'hillshade',
          strength: 3,
          light: -2.3
        },
        {
          type: 'field',
          columns,
          rows,
          values,
          ramp: 'heat',
          style: 'expand',
          levels: 8,
          cutoff: p.budget / 100,
          speed: 0.12
        },
        {
          type: 'sprites',
          items: [createNode(source[0], source[1], 'ink', 3.4)],
          pulse: 0.5
        }
      ];
    }
  },

  'raster-reprojection': {
    parameters: [
      {
        id: 'warp',
        label: 'Distortion',
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.6
      }
    ],
    build: ({fork, p}) => {
      const image = createLandscape(fork('image'), 6, 1.8);
      const warp = (u: number, v: number): Point => [
        u + p.warp * 0.09 * Math.sin(v * Math.PI * 2 + 0.6) + p.warp * 0.08 * (v - 0.5),
        v + p.warp * 0.12 * Math.sin(u * Math.PI * 1.5 + 1)
      ];
      const panel = FULL_PANEL;
      // The source graticule, bent by the same mapping: the footprint of the target grid in source space.
      const lines: PathItem[] = [];
      for (let index = 1; index < 8; index++) {
        const u = index / 8;
        lines.push({
          points: Array.from({length: 14}, (_, step): Point => {
            const [wu, wv] = warp(u, step / 13);
            return [
              16 + clamp(2 * u - wu, 0, 1) * 128,
              10 + clamp((step / 13) * 2 - wv, 0, 1) * 80
            ];
          }),
          role: 'cool',
          width: 0.5,
          alpha: 0.7
        });
      }
      for (let index = 1; index < 5; index++) {
        const v = index / 5;
        lines.push({
          points: Array.from({length: 18}, (_, step): Point => {
            const [wu, wv] = warp(step / 17, v);
            return [
              16 + clamp((step / 17) * 2 - wu, 0, 1) * 128,
              10 + clamp(2 * v - wv, 0, 1) * 80
            ];
          }),
          role: 'cool',
          width: 0.5,
          alpha: 0.7
        });
      }
      return [
        createFillLayer(
          paintPanel(panel, (u, v, column) => {
            const [sourceU, sourceV] = warp(u, v);
            return {
              ...makeCell(
                'primary',
                readField(image, clamp(sourceU, 0, 1), clamp(sourceV, 0, 1)),
                0.08
              ),
              phase: column / panel.columns
            };
          }),
          'pulse'
        ),
        createFlowLayer(lines, 0.2, 0.9),
        createStaticLayer([getPanelOutline(panel, 'warm', 0.7)])
      ];
    }
  },

  'tile-mosaic-reprojection': {
    parameters: [{id: 'join', label: 'Join', min: 0, max: 1, step: 0.05, default: 0.55}],
    build: ({fork, p}) => {
      const image = createLandscape(fork('image'), 6, 1.4);
      const gap = (1 - p.join) * 16;
      const tiles = [0, 1, 2, 3].map(index => {
        const tx = index % 2;
        const ty = Math.floor(index / 2);
        return {
          tx,
          ty,
          panel: makePanel(
            26 + tx * 56 + (tx - 0.5) * gap,
            14 + ty * 36 + (ty - 0.5) * gap + (tx === 1 ? gap * 0.15 : 0),
            14,
            9
          )
        };
      });
      return [
        ...tiles.map(({tx, ty, panel}, index) =>
          createFillLayer(
            paintPanel(panel, (u, v) =>
              makeCell(
                getRoleCycle(index) === 'primary'
                  ? 'primary'
                  : (['primary', 'warm', 'cool', 'violet'][index] as Role),
                readField(image, (tx + u) / 2, (ty + v) / 2),
                0.14
              )
            ),
            'pulse'
          )
        ),
        createStaticLayer(
          tiles.map(({panel}, index) =>
            getPanelOutline(panel, (['primary', 'warm', 'cool', 'violet'] as Role[])[index], 0.8)
          )
        ),
        createFlowLayer(
          tiles.map(({panel}) => createLink([panel.x + 28, panel.y + 18], [82, 50], 'ink', 0.4)),
          0.3,
          0.6
        ),
        {
          type: 'sprites',
          items: [createNode(82, 50, 'ink', 2.2)],
          pulse: 0.4
        }
      ];
    }
  },

  'conservative-remapping': {
    pointer: 'hover',
    parameters: [
      {
        id: 'factor',
        label: 'Coarsening',
        min: 0,
        max: 1,
        step: 1,
        default: 0,
        options: ['2×', '4×']
      }
    ],
    build: ({fork, p, pointer}) => {
      const field = createLandscape(fork('fine'), 6, 1.8);
      const fine = makePanel(8, 10, 16, 20);
      const factor = p.factor > 0.5 ? 4 : 2;
      const coarse = makePanel(88, 10, fine.columns / factor, fine.rows / factor, 4 * factor);
      const sums = new Float32Array(coarse.columns * coarse.rows);
      const values = createPanelValues(field, fine);
      for (let row = 0; row < fine.rows; row++) {
        for (let column = 0; column < fine.columns; column++)
          sums[Math.floor(row / factor) * coarse.columns + Math.floor(column / factor)] +=
            values[row * fine.columns + column];
      }
      const peak = Math.max(...sums);
      const hovered = getPanelCell(fine, pointer);
      const selected: [number, number] = hovered
        ? [Math.floor(hovered[0] / factor), Math.floor(hovered[1] / factor)]
        : [1, 1];
      const blockCenter: Point = getCellCenter(coarse, selected[0], selected[1]);
      const blockLeft = fine.x + selected[0] * factor * 4;
      const blockTop = fine.y + selected[1] * factor * 4;
      return [
        createFillLayer(
          paintPanel(fine, (_u, _v, column, row) =>
            makeCell('cool', values[row * fine.columns + column], 0.1)
          )
        ),
        createFillLayer(
          paintPanel(coarse, (_u, _v, column, row) => ({
            ...makeCell('warm', sums[row * coarse.columns + column] / peak, 0.12),
            phase: column / coarse.columns
          })),
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(fine, 'cool', 0.5),
          getPanelOutline(coarse, 'warm', 0.5),
          {
            points: [
              [blockLeft, blockTop],
              [blockLeft + factor * 4, blockTop],
              [blockLeft + factor * 4, blockTop + factor * 4],
              [blockLeft, blockTop + factor * 4],
              [blockLeft, blockTop]
            ],
            role: 'ink',
            width: 0.9,
            alpha: 1
          }
        ]),
        createFlowLayer(
          [
            createLink(
              [blockLeft + factor * 4, blockTop + factor * 2],
              [blockCenter[0] - coarse.size / 2, blockCenter[1]],
              'ink',
              0.9
            )
          ],
          0.35
        ),
        createFillLayer(
          [
            {
              ring: getCellRing(coarse, selected[0], selected[1], 0.3),
              role: 'ink',
              alpha: 0.35
            }
          ],
          'pulse'
        )
      ];
    }
  },

  'raster-registration': {
    parameters: [
      {id: 'shiftX', label: 'Shift X', min: -6, max: 6, step: 1, default: 4},
      {id: 'shiftY', label: 'Shift Y', min: -4, max: 4, step: 1, default: -2}
    ],
    build: ({fork, p}) => {
      const shape = createBlob(fork('shape'), 0, 0, 12, 12, 0.4);
      const shiftScale = 2.5;
      const placeShape = (cx: number, cy: number): Point[] =>
        shape
          .map(([x, y]) => [x + cx, y + cy] as Point)
          .concat([[shape[0][0] + cx, shape[0][1] + cy]]);
      const before = [
        placeShape(26, 50),
        placeShape(26 + p.shiftX * shiftScale, 50 + p.shiftY * shiftScale)
      ];
      const after = [placeShape(134, 50), placeShape(134, 50)];
      const correlation = makePanel(58, 30, 11, 11, 4);
      const peakColumn = 5 + Math.round(p.shiftX);
      const peakRow = 5 + Math.round(p.shiftY);
      const peak = getCellCenter(correlation, clamp(peakColumn, 0, 10), clamp(peakRow, 0, 10));
      const origin = getCellCenter(correlation, 5, 5);
      return [
        createFillLayer(
          paintPanel(correlation, (_u, _v, column, row) => {
            const distance = Math.hypot(column - peakColumn, row - peakRow);
            const value =
              Math.exp((-distance * distance) / 2.2) +
              0.08 * Math.sin(column * 1.3) * Math.cos(row * 1.1) +
              0.1;
            return {
              ...makeCell(getValueRole(value), value, 0.08),
              phase: distance / 10
            };
          }),
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(correlation, 'dim', 0.6),
          {
            points: [
              [origin[0] - 3, origin[1]],
              [origin[0] + 3, origin[1]]
            ],
            role: 'ink',
            width: 0.4,
            alpha: 0.8
          },
          {
            points: [
              [origin[0], origin[1] - 3],
              [origin[0], origin[1] + 3]
            ],
            role: 'ink',
            width: 0.4,
            alpha: 0.8
          }
        ]),
        createFlowLayer(
          [
            {points: before[0], role: 'cool', width: 1, alpha: 0.9},
            {points: before[1], role: 'hot', width: 1, alpha: 0.9},
            {points: after[0], role: 'cool', width: 1, alpha: 0.9},
            {points: after[1], role: 'hot', width: 1, alpha: 0.9},
            createLink(origin, peak, 'warm', 0.95),
            createLink([44, 50], [56, 50], 'dim', 0.5),
            createLink([104, 50], [116, 50], 'warm', 0.8)
          ],
          0.3,
          1.4
        ),
        {
          type: 'sprites',
          items: [createNode(peak[0], peak[1], 'warm', 2.4)],
          pulse: 0.5
        }
      ];
    }
  },

  'spectral-unmixing': {
    pointer: 'hover',
    parameters: [],
    build: ({fork, pointer}) => {
      const mixture = createLandscape(fork('mixture'), 5, 1.3);
      const mixtureTwo = createLandscape(fork('mixture-two'), 5, 1.3);
      const image = makePanel(8, 10, 17, 20);
      const roles: Role[] = ['primary', 'warm', 'cool'];
      const fractionsAt = (u: number, v: number): number[] => {
        const raw = [
          Math.exp(6 * readField(mixture, u, v)),
          Math.exp(6 * readField(mixtureTwo, u, v)),
          Math.exp(6 * (1 - readField(mixture, v, u)))
        ];
        const total = raw[0] + raw[1] + raw[2];
        return raw.map(value => value / total);
      };
      const selected = getPanelCell(image, pointer) ?? ([9, 8] as [number, number]);
      const fractions = fractionsAt(
        (selected[0] + 0.5) / image.columns,
        (selected[1] + 0.5) / image.rows
      );
      const spectra: Point[][] = [
        [0.1, 0.12, 0.2, 0.75, 0.8, 0.7],
        [0.2, 0.3, 0.45, 0.55, 0.6, 0.62],
        [0.5, 0.35, 0.2, 0.1, 0.05, 0.03]
      ].map(values => values.map((value, index): Point => [86 + index * 12, 52 - value * 36]));
      const blended: Point[] = spectra[0].map(
        (_, index): Point => [
          86 + index * 12,
          spectra.reduce(
            (sum, curve, curveIndex) => sum + fractions[curveIndex] * curve[index][1],
            0
          )
        ]
      );
      return [
        // Three additive layers per pixel: the colour mix is the abundance mix.
        ...roles.map((role, endmember) =>
          createFillLayer(
            paintPanel(image, (u, v) => ({
              role,
              alpha: 0.7 * fractionsAt(u, v)[endmember],
              phase: 0
            }))
          )
        ),
        createFillLayer(
          [
            {
              ring: getCellRing(image, selected[0], selected[1]),
              role: 'ink',
              alpha: 0.9
            }
          ],
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(image, 'dim', 0.5),
          {
            points: [
              [84, 14],
              [84, 54],
              [160, 54]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.7
          },
          ...spectra.map(
            (points, index): PathItem => ({
              points,
              role: roles[index],
              width: 0.5,
              alpha: 0.55
            })
          )
        ]),
        createFlowLayer([{points: blended, role: 'ink', width: 1.4, alpha: 0.95}], 0.3, 1.5),
        createFillLayer(
          fractions.map(
            (fraction, index): PolygonItem => ({
              ring: createBarPolygon(84, 70 + index * 9 + 5, 4 + fraction * 66, 4),
              role: roles[index],
              alpha: 0.9,
              phase: index / 3
            })
          ),
          'pulse'
        )
      ];
    }
  },

  'change-detection': {
    pointer: 'hover',
    parameters: [
      {
        id: 'significance',
        label: 'Significance',
        min: 0.1,
        max: 0.9,
        step: 0.05,
        default: 0.45
      }
    ],
    build: ({fork, p, pointer}) => {
      const trend = createLandscape(fork('trend'), 5, 1.3);
      const noise = fork('noise');
      const map = makePanel(8, 10, 17, 20);
      const slopeAt = (u: number, v: number) => (readField(trend, u, v) - 0.5) * 2;
      const selected = getPanelCell(map, pointer) ?? ([11, 6] as [number, number]);
      const slope = slopeAt((selected[0] + 0.5) / map.columns, (selected[1] + 0.5) / map.rows);
      const series: Point[] = Array.from(
        {length: 24},
        (_, index): Point => [
          86 + index * 3,
          44 - (0.5 + slope * (index / 23 - 0.5) + noise.gaussian() * 0.06) * 30
        ]
      );
      const fit: Point[] = [
        [86, 44 - (0.5 - slope * 0.5) * 30],
        [86 + 23 * 3, 44 - (0.5 + slope * 0.5) * 30]
      ];
      const significant = Math.abs(slope) > p.significance;
      return [
        createFillLayer(
          paintPanel(map, (u, v, column) => {
            const value = slopeAt(u, v);
            const passes = Math.abs(value) > p.significance;
            return {
              role: value >= 0 ? 'hot' : 'cool',
              alpha: passes ? 0.35 + 0.6 * Math.abs(value) : 0.07,
              phase: column / map.columns
            };
          }),
          'pulse'
        ),
        createFillLayer(
          [
            {
              ring: getCellRing(map, selected[0], selected[1]),
              role: 'ink',
              alpha: 0.9
            }
          ],
          'pulse'
        ),
        createStaticLayer([
          getPanelOutline(map, 'dim', 0.55),
          {
            points: [
              [84, 10],
              [84, 46],
              [160, 46]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.7
          },
          {
            points: [
              [86, 86],
              [150, 86]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.6
          }
        ]),
        createFlowLayer(
          [
            {points: series, role: 'ink', width: 0.9, alpha: 0.9},
            {
              points: fit,
              role: significant ? 'warm' : 'dim',
              width: 1.5,
              alpha: 1
            }
          ],
          0.3,
          1.2
        ),
        createFlowLayer(
          [
            {
              points: [
                [86, 78],
                [86 + Math.min(1, Math.abs(slope) / 0.9) * 64, 78]
              ],
              role: significant ? 'warm' : 'dim',
              width: 3,
              alpha: 0.95
            }
          ],
          0.3,
          0.8
        ),
        {
          type: 'strokes',
          items: [
            {
              points: [
                [86 + (p.significance / 0.9) * 64, 72],
                [86 + (p.significance / 0.9) * 64, 84]
              ],
              role: 'hot',
              width: 0.8,
              alpha: 1
            }
          ],
          animation: 'static',
          blend: 'add'
        },
        {
          type: 'sprites',
          items: series.map(
            ([x, y], index): SpriteItem => ({
              x,
              y,
              radius: 0.9,
              role: 'ink',
              alpha: 0.8,
              phase: index / 24
            })
          ),
          pulse: 0.2
        }
      ];
    }
  },

  'seasonal-break-detection': {
    parameters: [
      {
        id: 'magnitude',
        label: 'Break size',
        min: 0,
        max: 1,
        step: 0.05,
        default: 0.6
      },
      {
        id: 'time',
        label: 'Break time',
        min: 0.25,
        max: 0.8,
        step: 0.05,
        default: 0.6
      }
    ],
    build: ({fork, p}) => {
      const noise = fork('noise');
      const count = 72;
      const toX = (index: number) => 10 + (index / (count - 1)) * 140;
      const seasonal = (index: number) => 0.5 + 0.22 * Math.sin((index / 12) * Math.PI * 2);
      const breakIndex = Math.round(p.time * (count - 1));
      const observed = Array.from(
        {length: count},
        (_, index) =>
          seasonal(index) - (index >= breakIndex ? p.magnitude * 0.3 : 0) + noise.gaussian() * 0.035
      );
      let cumulative = 0;
      let detection = -1;
      const cusum = observed.map((value, index) => {
        cumulative += seasonal(index) - value;
        if (detection < 0 && Math.abs(cumulative) > 0.45) detection = index;
        return cumulative;
      });
      const peak = Math.max(0.6, ...cusum.map(Math.abs));
      const detectedAt = detection >= 0 ? detection : breakIndex;
      return [
        createStaticLayer([
          {
            points: [
              [10, 60],
              [150, 60]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.6
          },
          {
            points: [
              [10, 94],
              [150, 94]
            ],
            role: 'dim',
            width: 0.4,
            alpha: 0.6
          },
          {
            points: Array.from(
              {length: count},
              (_, index): Point => [toX(index), 56 - seasonal(index) * 48]
            ),
            role: 'cool',
            width: 0.5,
            alpha: 0.55
          },
          {
            points: [
              [10, 77],
              [150, 77]
            ],
            role: 'dim',
            width: 0.3,
            alpha: 0.4
          }
        ]),
        createFlowLayer(
          [
            {
              points: observed.map((value, index): Point => [toX(index), 56 - value * 48]),
              role: 'primary',
              width: 1,
              alpha: 0.95
            }
          ],
          0.25,
          1.4
        ),
        createFlowLayer(
          [
            {
              points: cusum.map((value, index): Point => [toX(index), 77 + (value / peak) * 15]),
              role: 'violet',
              width: 1,
              alpha: 0.9
            }
          ],
          0.3,
          1.2
        ),
        createFlowLayer(
          [
            {
              points: [
                [toX(detection >= 0 ? detectedAt : breakIndex), 6],
                [toX(detection >= 0 ? detectedAt : breakIndex), 94]
              ],
              role: 'warm',
              width: 0.9,
              alpha: 1
            }
          ],
          0.4
        ),
        {
          type: 'sprites',
          items: [createNode(toX(detectedAt), 56 - observed[detectedAt] * 48, 'warm', 3)],
          pulse: 0.5
        }
      ];
    }
  },

  'temporal-composite': {
    parameters: [
      {id: 'dates', label: 'Dates', min: 2, max: 6, step: 1, default: 5},
      {
        id: 'cloud',
        label: 'Cloud cover',
        min: 10,
        max: 70,
        step: 5,
        default: 45,
        unit: '%'
      }
    ],
    build: ({fork, p}) => {
      const surface = createLandscape(fork('surface'), 6, 1.5);
      const clouds = createLandscape(fork('clouds'), 4, 2.4);
      const dates = Math.round(p.dates);
      const panels = Array.from({length: dates}, (_, index) =>
        makePanel(6 + index * 6, 4 + index * (78 / dates), 8, 6, 3.4)
      );
      const output = makePanel(88, 10, 17, 20);
      const isCloudy = (date: number, u: number, v: number) =>
        readField(clouds, (u + date * 0.173) % 1, (v + date * 0.29) % 1) < (p.cloud / 100) * 0.9;
      const links: PathItem[] = panels.map(panel =>
        createLink([panel.x + 32, panel.y + 10], [72, 50], 'cool', 0.4)
      );
      return [
        ...panels.map((panel, date) =>
          createFillLayer(
            paintPanel(panel, (u, v) =>
              isCloudy(date, u, v)
                ? {role: 'ink', alpha: 0.85}
                : makeCell('primary', readField(surface, u, v), 0.15)
            )
          )
        ),
        createStaticLayer(panels.map(panel => getPanelOutline(panel, 'cool', 0.4))),
        createFillLayer(
          paintPanel(output, (u, v, column) => {
            const clearDate = Array.from({length: dates}, (_, date) => date).find(
              date => !isCloudy(date, u, v)
            );
            return clearDate === undefined
              ? {role: 'hot', alpha: 0.25, phase: 0}
              : {
                  ...makeCell('primary', readField(surface, u, v), 0.15),
                  phase: column / output.columns
                };
          }),
          'pulse'
        ),
        createStaticLayer([getPanelOutline(output, 'dim', 0.6)]),
        createFlowLayer([...links, createLink([72, 50], [86, 50], 'warm')], 0.3, 0.8),
        {
          type: 'sprites',
          items: [createNode(72, 50, 'warm', 3.4)],
          pulse: 0.4
        }
      ];
    }
  }
};
