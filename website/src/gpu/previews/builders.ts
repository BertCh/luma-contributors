// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// One small builder per PreviewKind. Each returns layers in draw order.

import type {PreviewKind} from '../../catalog/schema.ts';
import {
  FIELD_COLUMNS,
  FIELD_ROWS,
  clamp,
  createArc,
  createBlob,
  createCircle,
  createClusterPoints,
  createDensityField,
  createRandom,
  createRegularPolygon,
  createRoadNetwork,
  createScatterPoints,
  createSmoothField,
  createWanderPath,
  evaluateField,
  findShortestPaths,
  getDistance,
  getPathTo,
  normalizeValues,
  type Random,
  type RoadNetwork
} from './data.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type PathItem,
  type Point,
  type PolygonItem,
  type PreviewData,
  type PreviewLayer,
  type Role,
  type SpriteItem
} from './types.ts';

type Builder = (random: Random) => PreviewLayer[];

function createGridLayer(): PreviewLayer {
  const items: PathItem[] = [];
  for (let x = 20; x < SCENE_WIDTH; x += 20) {
    items.push({
      points: [
        [x, 0],
        [x, SCENE_HEIGHT]
      ],
      role: 'dim',
      width: 0.18,
      alpha: 0.14
    });
  }
  for (let y = 20; y < SCENE_HEIGHT; y += 20) {
    items.push({
      points: [
        [0, y],
        [SCENE_WIDTH, y]
      ],
      role: 'dim',
      width: 0.18,
      alpha: 0.14
    });
  }
  return {type: 'strokes', items, animation: 'static'};
}

function toSprites(
  points: Point[],
  role: Role,
  radius: number,
  random?: Random,
  alpha = 0.9
): SpriteItem[] {
  return points.map(([x, y]) => ({
    x,
    y,
    radius: radius * (random ? random.range(0.7, 1.3) : 1),
    role,
    alpha,
    phase: random ? random.next() : 0
  }));
}

function toPaths(
  paths: Point[][],
  role: Role,
  width: number,
  alpha = 0.9,
  random?: Random
): PathItem[] {
  return paths.map(points => ({points, role, width, alpha, phase: random ? random.next() : 0}));
}

function clampPoint([x, y]: Point, margin = 6): Point {
  return [clamp(x, margin, SCENE_WIDTH - margin), clamp(y, margin, SCENE_HEIGHT - margin)];
}

const CLUSTER_CENTERS: Point[] = [
  [42, 36],
  [100, 62],
  [122, 28],
  [60, 74]
];

function getClusterCenters(random: Random): Point[] {
  return CLUSTER_CENTERS.map(([x, y]) => [x + random.range(-8, 8), y + random.range(-6, 6)]);
}

function getRoleCycle(index: number): Role {
  return (['primary', 'warm', 'hot', 'cool', 'violet'] as const)[index % 5];
}

function buildNetworkLayers(network: RoadNetwork): PreviewLayer[] {
  const segments = network.edges.map(([first, second]) => [
    network.nodes[first],
    network.nodes[second]
  ]);
  const center = network.nodes[Math.floor(network.nodes.length / 2)];
  const nodes = network.nodes.map(
    ([x, y]): SpriteItem => ({
      x,
      y,
      radius: 1.6,
      role: 'primary',
      alpha: 0.9,
      phase: clamp(getDistance([x, y], center) / 90, 0, 1)
    })
  );
  return [
    {type: 'strokes', items: toPaths(segments, 'dim', 0.5, 0.55), animation: 'static'},
    {type: 'sprites', items: nodes, pulse: 0.5}
  ];
}

const builders: Record<PreviewKind, Builder> = {
  points: random => {
    const points = [
      ...createClusterPoints(random, getClusterCenters(random), 28, 11),
      ...createScatterPoints(random, 40)
    ];
    return [{type: 'sprites', items: toSprites(points, 'primary', 1.5, random), pulse: 0.35}];
  },

  'point-clusters': random => {
    const centers = getClusterCenters(random);
    const items: SpriteItem[] = [];
    centers.forEach(([x, y], index) => {
      const role = getRoleCycle(index);
      items.push({x, y, radius: 17, role, alpha: 0.12, phase: random.next()});
      items.push(...toSprites(createClusterPoints(random, [[x, y]], 26, 7), role, 1.5, random));
    });
    items.push(...toSprites(createScatterPoints(random, 22), 'dim', 1.1, random, 0.6));
    return [{type: 'sprites', items, pulse: 0.3}];
  },

  hotspots: random => {
    const hot: Point = [random.range(88, 108), random.range(34, 48)];
    const cold: Point = [random.range(36, 52), random.range(52, 68)];
    const items: SpriteItem[] = [
      {x: hot[0], y: hot[1], radius: 24, role: 'hot', alpha: 0.16, phase: 0},
      {x: cold[0], y: cold[1], radius: 20, role: 'cool', alpha: 0.14, phase: 0.5},
      ...toSprites(createScatterPoints(random, 70), 'dim', 1.2, random, 0.6),
      ...toSprites(createClusterPoints(random, [hot], 34, 8), 'hot', 1.7, random),
      ...toSprites(createClusterPoints(random, [cold], 24, 7), 'cool', 1.6, random)
    ];
    return [{type: 'sprites', items, pulse: 0.4}];
  },

  'density-surface': random => {
    const points = createClusterPoints(random, getClusterCenters(random), 36, 11);
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values: createDensityField(points, 9),
        ramp: 'heat',
        style: 'ramp',
        strength: 0.06
      },
      {type: 'sprites', items: toSprites(points, 'ink', 0.7, random, 0.35)}
    ];
  },

  interpolation: random => {
    const samples = createScatterPoints(random, 13, 14);
    const sampleValues = samples.map(() => random.next());
    const values = normalizeValues(
      evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
        let weightSum = 0;
        let sum = 0;
        for (let index = 0; index < samples.length; index++) {
          const distance = Math.hypot(
            samples[index][0] - u * SCENE_WIDTH,
            samples[index][1] - v * SCENE_HEIGHT
          );
          const weight = 1 / (distance * distance + 4);
          weightSum += weight;
          sum += weight * sampleValues[index];
        }
        return sum / weightSum;
      })
    );
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values,
        ramp: 'spectral',
        style: 'ramp',
        strength: 0.05
      },
      {type: 'sprites', items: toSprites(samples, 'ink', 1.7, random, 0.95), pulse: 0.2}
    ];
  },

  raster: random => {
    const columns = 32;
    const rows = 20;
    return [
      {
        type: 'field',
        columns,
        rows,
        values: createSmoothField(random, 6, 1.4, columns, rows),
        ramp: 'ice',
        style: 'ramp',
        filter: 'nearest',
        strength: 0.1
      }
    ];
  },

  terrain: random => [
    {
      type: 'field',
      columns: FIELD_COLUMNS,
      rows: FIELD_ROWS,
      values: createSmoothField(random, 10, 2.1),
      ramp: 'terrain',
      style: 'hillshade',
      strength: 5,
      speed: 0.12
    }
  ],

  'flow-accumulation': random => {
    const values = createSmoothField(random, 9, 1.9);
    const trunk = createWanderPath(random, [150, 62], Math.PI, 7, 20, 0.5).map(point =>
      clampPoint(point, 8)
    );
    const streams: PathItem[] = [
      {points: trunk, role: 'primary', width: 2.6, alpha: 0.95, phase: 0}
    ];
    trunk.forEach((junction, index) => {
      if (index < 3 || index % 3 !== 0) return;
      const side = index % 2 === 0 ? 1 : -1;
      const branch = createWanderPath(random, junction, Math.PI + side * 1.3, 6, 8, 0.6)
        .map(point => clampPoint(point))
        .reverse();
      streams.push({points: branch, role: 'cool', width: 1.4, alpha: 0.85, phase: random.next()});
      const twig = createWanderPath(random, branch[2] ?? junction, side * 1.2, 5, 5, 0.6)
        .map(point => clampPoint(point))
        .reverse();
      streams.push({points: twig, role: 'cool', width: 0.8, alpha: 0.7, phase: random.next()});
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
        speed: 0.05
      },
      {type: 'strokes', items: streams, animation: 'flow', speed: 0.35, blend: 'add'}
    ];
  },

  contours: random => [
    {
      type: 'field',
      columns: FIELD_COLUMNS,
      rows: FIELD_ROWS,
      values: createSmoothField(random, 8, 1.1),
      ramp: 'ice',
      style: 'contours',
      levels: 14,
      speed: 0.12
    }
  ],

  network: random => [createGridLayer(), ...buildNetworkLayers(createRoadNetwork(random))],

  'shortest-path': random => {
    const network = createRoadNetwork(random);
    const source = network.columns * 1 + 1;
    const target = network.columns * (network.rows - 2) + network.columns - 2;
    const {previous} = findShortestPaths(network, source);
    const path = getPathTo(previous, target).map(index => network.nodes[index]);
    return [
      createGridLayer(),
      ...buildNetworkLayers(network).map(
        (layer): PreviewLayer => (layer.type === 'sprites' ? {...layer, pulse: 0} : layer)
      ),
      {
        type: 'strokes',
        items: [{points: path, role: 'warm', width: 2.4, alpha: 1}],
        animation: 'draw',
        speed: 0.3,
        blend: 'add'
      },
      {
        type: 'sprites',
        items: [
          {
            x: network.nodes[source][0],
            y: network.nodes[source][1],
            radius: 4.5,
            role: 'warm',
            phase: 0
          },
          {
            x: network.nodes[target][0],
            y: network.nodes[target][1],
            radius: 4.5,
            role: 'hot',
            phase: 0.5
          }
        ],
        pulse: 0.3
      }
    ];
  },

  isochrones: random => {
    const network = createRoadNetwork(random);
    const source = network.columns * 3 + 3;
    const {distances} = findShortestPaths(network, source);
    const values = normalizeValues(
      evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
        const x = u * SCENE_WIDTH;
        const y = v * SCENE_HEIGHT;
        let best = Infinity;
        for (let index = 0; index < network.nodes.length; index++) {
          best = Math.min(
            best,
            distances[index] +
              1.25 * Math.hypot(network.nodes[index][0] - x, network.nodes[index][1] - y)
          );
        }
        return Math.min(best, 220);
      })
    );
    const segments = network.edges.map(([first, second]) => [
      network.nodes[first],
      network.nodes[second]
    ]);
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values,
        ramp: 'heat',
        style: 'expand',
        levels: 8,
        speed: 0.12
      },
      {type: 'strokes', items: toPaths(segments, 'ink', 0.35, 0.2), animation: 'static'},
      {
        type: 'sprites',
        items: [
          {
            x: network.nodes[source][0],
            y: network.nodes[source][1],
            radius: 4,
            role: 'ink',
            phase: 0
          }
        ],
        pulse: 0.3
      }
    ];
  },

  flows: random => {
    const hubs: Point[] = [
      [30, 52],
      [78, 24],
      [124, 36],
      [104, 78],
      [58, 82],
      [140, 70]
    ];
    const items: PathItem[] = [];
    for (let index = 0; index < 18; index++) {
      const from = hubs[random.integer(hubs.length)];
      const to = hubs[random.integer(hubs.length)];
      if (from === to) continue;
      items.push({
        points: createArc(from, to, random.range(0.15, 0.3) * (random.next() > 0.5 ? 1 : -1)),
        role: index % 3 === 0 ? 'warm' : 'primary',
        width: random.range(0.6, 2),
        alpha: 0.9,
        phase: random.next()
      });
    }
    return [
      createGridLayer(),
      {type: 'strokes', items, animation: 'flow', speed: 0.3, blend: 'add'},
      {type: 'sprites', items: toSprites(hubs, 'ink', 3, random), pulse: 0.25}
    ];
  },

  lines: random => {
    const roles: Role[] = ['primary', 'warm', 'cool', 'violet', 'hot'];
    const items: PathItem[] = Array.from({length: 9}, (_, index) => ({
      points: createWanderPath(
        random,
        [random.range(8, 40), random.range(10, 90)],
        random.range(-0.4, 0.4),
        6,
        22,
        0.5
      ).map(point => clampPoint(point)),
      role: roles[index % roles.length],
      width: random.range(0.9, 1.8),
      alpha: 0.95,
      phase: index / 9
    }));
    return [
      createGridLayer(),
      {type: 'strokes', items, animation: 'draw', speed: 0.22, blend: 'add'}
    ];
  },

  polygons: random => {
    const items: PolygonItem[] = [];
    const outlines: PathItem[] = [];
    const roles: Role[] = ['primary', 'cool', 'violet', 'warm', 'hot'];
    for (let row = 0; row < 3; row++) {
      for (let column = 0; column < 5; column++) {
        const ring = createBlob(
          random,
          20 + column * 30 + random.range(-3, 3),
          20 + row * 30 + random.range(-3, 3),
          random.range(10, 13),
          10,
          0.25
        );
        const value = random.next();
        items.push({
          ring,
          role: roles[Math.floor(value * 5)],
          alpha: 0.2 + value * 0.3,
          phase: random.next()
        });
        outlines.push({points: [...ring, ring[0]], role: 'ink', width: 0.35, alpha: 0.5});
      }
    }
    return [
      {type: 'fills', items, animation: 'pulse', blend: 'add'},
      {type: 'strokes', items: outlines, animation: 'static'}
    ];
  },

  'polygon-overlay': random => {
    const first = createBlob(random, 62, 50, 30, 18, 0.22, 0.9);
    const second = createBlob(random, 98, 50, 30, 18, 0.22, 0.9);
    return [
      createGridLayer(),
      {
        type: 'fills',
        items: [
          {ring: first, role: 'primary', alpha: 0.32, phase: 0},
          {ring: second, role: 'warm', alpha: 0.32, phase: 0.5}
        ],
        animation: 'pulse',
        blend: 'add'
      },
      {
        type: 'strokes',
        items: [
          {points: [...first, first[0]], role: 'primary', width: 0.9, alpha: 0.95},
          {points: [...second, second[0]], role: 'warm', width: 0.9, alpha: 0.95}
        ],
        animation: 'draw',
        speed: 0.18,
        blend: 'add'
      }
    ];
  },

  buffers: random => {
    const river = createWanderPath(random, [10, 62], -0.2, 8, 18, 0.35).map(
      ([x, y]): Point => [x, clamp(y, 18, 82)]
    );
    const sites = [
      [40, 28],
      [88, 78],
      [128, 30]
    ].map(([x, y]): Point => [x + random.range(-4, 4), y]);
    return [
      createGridLayer(),
      {
        type: 'strokes',
        items: [{points: river, role: 'cool', width: 15, alpha: 0.14, phase: 0}],
        animation: 'pulse',
        speed: 0.2,
        blend: 'add'
      },
      {
        type: 'fills',
        items: sites.map(([x, y], index) => ({
          ring: createCircle(x, y, 17),
          role: 'warm' as Role,
          alpha: 0.14,
          phase: index / 3
        })),
        animation: 'pulse',
        blend: 'add'
      },
      {
        type: 'strokes',
        items: [{points: river, role: 'cool', width: 1.4, alpha: 0.95}],
        animation: 'static',
        blend: 'add'
      },
      {type: 'sprites', items: toSprites(sites, 'warm', 2.6), pulse: 0.2}
    ];
  },

  tessellation: random => {
    const ring = createBlob(random, 80, 50, 36, 12, 0.35, 0.85);
    const center: Point = [80, 50];
    const triangles: PolygonItem[] = ring.map((point, index) => ({
      ring: [center, point, ring[(index + 1) % ring.length]],
      role: getRoleCycle(index),
      alpha: 0.4,
      phase: index / ring.length
    }));
    const edges: PathItem[] = ring.flatMap((point, index) => [
      {
        points: [point, ring[(index + 1) % ring.length]] as Point[],
        role: 'ink' as Role,
        width: 0.5,
        alpha: 0.8
      },
      {points: [center, point] as Point[], role: 'ink' as Role, width: 0.3, alpha: 0.4}
    ]);
    return [
      createGridLayer(),
      {type: 'fills', items: triangles, animation: 'pulse', blend: 'add'},
      {type: 'strokes', items: edges, animation: 'static'},
      {type: 'sprites', items: toSprites(ring, 'ink', 1.3)}
    ];
  },

  cells: random => {
    const radius = 7.2;
    const width = Math.sqrt(3) * radius;
    const field = createSmoothField(random, 6, 1.5);
    const items: PolygonItem[] = [];
    const roles: Role[] = ['cool', 'primary', 'violet', 'warm', 'hot'];
    for (let row = -1; row < 15; row++) {
      for (let column = -1; column < 14; column++) {
        const x = column * width + (row % 2 ? width / 2 : 0);
        const y = row * radius * 1.5;
        const fieldRow = clamp(Math.floor((y / SCENE_HEIGHT) * FIELD_ROWS), 0, FIELD_ROWS - 1);
        const fieldColumn = clamp(
          Math.floor((x / SCENE_WIDTH) * FIELD_COLUMNS),
          0,
          FIELD_COLUMNS - 1
        );
        const value = field[fieldRow * FIELD_COLUMNS + fieldColumn];
        items.push({
          ring: createRegularPolygon(x, y, radius * 0.93, 6, Math.PI / 6),
          role: roles[Math.min(4, Math.floor(value * 5))],
          alpha: 0.18 + value * 0.5,
          phase: x / SCENE_WIDTH
        });
      }
    }
    return [{type: 'fills', items, animation: 'pulse', blend: 'add'}];
  },

  'spatial-weights': random => {
    const points = createScatterPoints(random, 46, 10);
    const background: PathItem[] = [];
    const focusLinks: PathItem[] = [];
    const focus = 18;
    const neighbors: number[] = [];
    points.forEach((point, index) => {
      const nearest = points
        .map((other, otherIndex) => ({otherIndex, distance: getDistance(point, other)}))
        .filter(entry => entry.otherIndex !== index)
        .sort((first, second) => first.distance - second.distance)
        .slice(0, 3);
      for (const entry of nearest) {
        if (index === focus) {
          neighbors.push(entry.otherIndex);
          focusLinks.push({
            points: [point, points[entry.otherIndex]],
            role: 'warm',
            width: 1.2,
            alpha: 0.95,
            phase: random.next()
          });
        } else if (index < entry.otherIndex) {
          background.push({
            points: [point, points[entry.otherIndex]],
            role: 'dim',
            width: 0.4,
            alpha: 0.5
          });
        }
      }
    });
    return [
      {type: 'strokes', items: background, animation: 'static'},
      {type: 'strokes', items: focusLinks, animation: 'flow', speed: 0.5, blend: 'add'},
      {type: 'sprites', items: toSprites(points, 'primary', 1.7, random), pulse: 0.25},
      {
        type: 'sprites',
        items: [
          {x: points[focus][0], y: points[focus][1], radius: 4.5, role: 'warm', phase: 0},
          ...neighbors.map(index => ({
            x: points[index][0],
            y: points[index][1],
            radius: 2.6,
            role: 'warm' as Role,
            phase: random.next()
          }))
        ],
        pulse: 0.3
      }
    ];
  },

  regression: random => {
    const left = 18;
    const right = 148;
    const slope = random.range(-0.45, -0.3);
    const lineAt = (x: number): Point => [x, 70 + slope * (x - left)];
    const points = Array.from({length: 54}, () => {
      const x = random.range(left, right);
      return [x, clamp(lineAt(x)[1] + random.gaussian() * 7, 8, 92)] as Point;
    });
    const residuals: PathItem[] = points.map(point => ({
      points: [point, [point[0], lineAt(point[0])[1]]],
      role: 'hot',
      width: 0.3,
      alpha: 0.5
    }));
    return [
      {
        type: 'strokes',
        items: [
          {
            points: [
              [left, 8],
              [left, 92],
              [right + 6, 92]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          }
        ],
        animation: 'static'
      },
      {type: 'strokes', items: residuals, animation: 'static'},
      {type: 'sprites', items: toSprites(points, 'primary', 1.5, random), pulse: 0.2},
      {
        type: 'strokes',
        items: [{points: [lineAt(left), lineAt(right)], role: 'warm', width: 1.8, alpha: 1}],
        animation: 'draw',
        speed: 0.18,
        blend: 'add'
      }
    ];
  },

  histogram: random => {
    const binCount = 22;
    const width = 128 / binCount;
    const items: PolygonItem[] = Array.from({length: binCount}, (_, index) => {
      const center = (index + 0.5) / binCount;
      const height =
        6 +
        70 * Math.exp(-(((center - 0.45) / 0.22) ** 2)) * random.range(0.8, 1.15) +
        4 * random.next();
      const left = 16 + index * width + 0.5;
      return {
        ring: [
          [left, 90],
          [left, 90 - height],
          [left + width - 1, 90 - height],
          [left + width - 1, 90]
        ],
        role: index > binCount * 0.75 ? 'hot' : 'primary',
        alpha: 0.7,
        phase: index / binCount
      };
    });
    return [
      {
        type: 'strokes',
        items: [
          {
            points: [
              [14, 10],
              [14, 90],
              [148, 90]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          }
        ],
        animation: 'static'
      },
      {type: 'fills', items, animation: 'pulse', blend: 'add'}
    ];
  },

  classification: random => {
    const values = createSmoothField(random, 6, 1.4, 16, 10);
    const roles: Role[] = ['cool', 'primary', 'violet', 'warm', 'hot'];
    const items: PolygonItem[] = [];
    for (let row = 0; row < 10; row++) {
      for (let column = 0; column < 16; column++) {
        const bin = Math.min(4, Math.floor(values[row * 16 + column] * 5));
        const left = 12 + column * 8.5;
        const top = 6 + row * 7.8;
        items.push({
          ring: [
            [left, top],
            [left + 7.6, top],
            [left + 7.6, top + 7],
            [left, top + 7]
          ],
          role: roles[bin],
          alpha: 0.5,
          phase: bin / 5
        });
      }
    }
    roles.forEach((role, index) => {
      const left = 12 + index * 27;
      items.push({
        ring: [
          [left, 88],
          [left + 25, 88],
          [left + 25, 93],
          [left, 93]
        ],
        role,
        alpha: 0.85,
        phase: index / 5
      });
    });
    return [{type: 'fills', items, animation: 'pulse', blend: 'add'}];
  },

  'time-series': random => {
    const roles: Role[] = ['primary', 'warm', 'violet'];
    const strokes: PathItem[] = [];
    const areas: PolygonItem[] = [];
    const heads: SpriteItem[] = [];
    roles.forEach((role, seriesIndex) => {
      let value = random.range(40, 60);
      const points: Point[] = [];
      for (let index = 0; index <= 40; index++) {
        value = clamp(
          value + random.gaussian() * 6 + Math.sin(index / 5 + seriesIndex) * 2.5,
          18,
          86
        );
        points.push([14 + index * 3.35, 100 - value]);
      }
      strokes.push({points, role, width: 1.3, alpha: 0.95, phase: seriesIndex * 0.1});
      if (seriesIndex === 0)
        areas.push({ring: [[14, 92], ...points, [148, 92]], role, alpha: 0.12});
      heads.push({x: points[40][0], y: points[40][1], radius: 2.6, role, phase: seriesIndex / 3});
    });
    return [
      {
        type: 'strokes',
        items: [
          {
            points: [
              [14, 8],
              [14, 92],
              [150, 92]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.7
          }
        ],
        animation: 'static'
      },
      {type: 'fills', items: areas, animation: 'static', blend: 'add'},
      {type: 'strokes', items: strokes, animation: 'draw', speed: 0.2, blend: 'add'},
      {type: 'sprites', items: heads, pulse: 0.35}
    ];
  },

  trajectories: random => {
    const roles: Role[] = ['primary', 'warm', 'cool', 'hot', 'violet'];
    const items: PathItem[] = Array.from({length: 16}, (_, index) => ({
      points: createWanderPath(
        random,
        [random.range(14, 50), random.range(14, 86)],
        random.range(-0.5, 0.5),
        5.5,
        26,
        0.45
      ).map(point => clampPoint(point)),
      role: roles[index % roles.length],
      width: 1.1,
      alpha: 0.9,
      phase: random.next()
    }));
    return [
      createGridLayer(),
      {type: 'strokes', items, animation: 'flow', speed: 0.25, blend: 'add'}
    ];
  },

  'point-cloud': random => {
    const items: SpriteItem[] = [];
    for (let index = 0; index < 1400; index++) {
      const x = random.range(-60, 60);
      const z = random.range(-40, 40);
      const hill = 14 * Math.sin(x / 14) * Math.cos(z / 11) + 9 * Math.sin(z / 7 + 1);
      const tower =
        (Math.abs(x - 22) < 9 && Math.abs(z + 8) < 8 ? -22 : 0) +
        (Math.abs(x + 26) < 7 && Math.abs(z - 14) < 7 ? -16 : 0);
      const height = hill + tower;
      const level = clamp((-height + 20) / 50, 0, 1);
      items.push({
        x: 80 + x,
        y: 62 + height * 0.9,
        z,
        radius: 0.9,
        role: level < 0.33 ? 'cool' : level < 0.66 ? 'primary' : 'warm',
        alpha: 0.85,
        phase: random.next()
      });
    }
    return [{type: 'sprites', items, turntable: {angle: 0.5, speed: 0.25, perspective: 1 / 220}}];
  },

  pipeline: random => {
    const boxes: {x: number; y: number; role: Role}[] = [
      {x: 8, y: 40, role: 'cool'},
      {x: 46, y: 18, role: 'primary'},
      {x: 46, y: 62, role: 'primary'},
      {x: 86, y: 40, role: 'violet'},
      {x: 124, y: 40, role: 'warm'}
    ];
    const edges: [number, number][] = [
      [0, 1],
      [0, 2],
      [1, 3],
      [2, 3],
      [3, 4]
    ];
    const boxWidth = 28;
    const boxHeight = 20;
    const getCorners = (box: {x: number; y: number}): Point[] => [
      [box.x, box.y],
      [box.x + boxWidth, box.y],
      [box.x + boxWidth, box.y + boxHeight],
      [box.x, box.y + boxHeight]
    ];
    const items: PolygonItem[] = boxes.map((box, index) => ({
      ring: getCorners(box),
      role: box.role,
      alpha: 0.28,
      phase: index / boxes.length
    }));
    const outlines: PathItem[] = boxes.map(box => {
      const corners = getCorners(box);
      return {points: [...corners, corners[0]], role: box.role, width: 0.7, alpha: 0.95};
    });
    const wires: PathItem[] = edges.map(([from, to], index) => {
      const start: Point = [boxes[from].x + boxWidth, boxes[from].y + boxHeight / 2];
      const end: Point = [boxes[to].x, boxes[to].y + boxHeight / 2];
      const middle = (start[0] + end[0]) / 2;
      return {
        points: [start, [middle, start[1]], [middle, end[1]], end],
        role: 'primary',
        width: 1.2,
        alpha: 0.95,
        phase: index * 0.17 + random.range(0, 0.05)
      };
    });
    return [
      createGridLayer(),
      {type: 'fills', items, animation: 'pulse', blend: 'add'},
      {type: 'strokes', items: outlines, animation: 'static'},
      {type: 'strokes', items: wires, animation: 'flow', speed: 0.45, blend: 'add'}
    ];
  }
};

function hashText(text: string): number {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index++) {
    hash = Math.imul(hash ^ text.charCodeAt(index), 16777619);
  }
  return hash >>> 0;
}

/** Deterministic data for one preview kind. */
export function buildPreview(kind: PreviewKind, seed = 1): PreviewData {
  return {layers: builders[kind](createRandom(hashText(kind) ^ Math.imul(seed, 2654435761)))};
}
