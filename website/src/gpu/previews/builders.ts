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
  getPathTo
} from './data.ts';
import {getDefaultParameterValues, type PreviewParameterValues} from './parameters.ts';
import {
  SCENE_HEIGHT,
  SCENE_WIDTH,
  type PathItem,
  type Point,
  type PolygonItem,
  type PreviewData,
  type Role,
  type SpriteItem
} from './types.ts';
import {
  BUFFER_FIELD_RANGE,
  buildNetworkLayers,
  clampPoint,
  createGridLayer,
  findNearestIndex,
  getClassRole,
  getClusterCenters,
  getLightAngle,
  getRoleCycle,
  getSegmentDistance,
  getTrailingMean,
  pickClosedEdges,
  smoothPath,
  toPaths,
  toSprites,
  type Builder
} from './helpers.ts';
import {getEntryPreview} from './entries/index.ts';

const builders: Record<PreviewKind, Builder> = {
  points: ({random, p}) => {
    const centers = getClusterCenters(random);
    const poolSize = 400;
    const pool: Point[] = Array.from({length: poolSize}, (_, index) => {
      if (index % 10 >= 7) return createScatterPoints(random, 1)[0];
      return createClusterPoints(random, [centers[index % 4]], 1, 11)[0];
    });
    const items = toSprites(pool, 'primary', p.size, random).slice(0, Math.round(p.count));
    return [{type: 'sprites', items, pulse: 0.35}];
  },

  'point-clusters': ({random, fork, p}) => {
    const centers = getClusterCenters(random);
    const items: SpriteItem[] = [];
    centers.slice(0, Math.round(p.clusters)).forEach(([x, y], index) => {
      const role = getRoleCycle(index);
      const local = fork(`cluster-${index}`);
      items.push({x, y, radius: p.radius * 2.4, role, alpha: 0.12, phase: local.next()});
      items.push(
        ...toSprites(createClusterPoints(local, [[x, y]], 26, p.radius), role, 1.5, local)
      );
    });
    items.push(...toSprites(createScatterPoints(fork('scatter'), 22), 'dim', 1.1, random, 0.6));
    return [{type: 'sprites', items, pulse: 0.3}];
  },

  hotspots: ({random, p}) => {
    const hot: Point = [random.range(88, 108), random.range(34, 48)];
    const cold: Point = [random.range(36, 52), random.range(52, 68)];
    const points = [
      ...createScatterPoints(random, 70),
      ...createClusterPoints(random, [hot], 34, 8),
      ...createClusterPoints(random, [cold], 24, 7)
    ];
    // Getis-Ord style score: neighbors within the radius, standardized across all points.
    const counts = points.map(
      point => points.filter(other => getDistance(point, other) <= p.radius).length
    );
    const mean = counts.reduce((sum, count) => sum + count, 0) / counts.length;
    const deviation =
      Math.sqrt(counts.reduce((sum, count) => sum + (count - mean) ** 2, 0) / counts.length) || 1;
    const items = points.map(([x, y], index): SpriteItem => {
      const score = (counts[index] - mean) / deviation;
      const role: Role = score > p.threshold ? 'hot' : score < -p.threshold ? 'cool' : 'dim';
      return {
        x,
        y,
        radius: role === 'dim' ? 1.2 : 1.8,
        role,
        alpha: role === 'dim' ? 0.55 : 0.95,
        phase: random.next()
      };
    });
    return [{type: 'sprites', items, pulse: 0.4}];
  },

  'density-surface': ({random, fork, p, pointer}) => {
    const points: Point[] = [];
    getClusterCenters(random).forEach((center, index) => {
      points.push(
        ...createClusterPoints(fork(`density-${index}`), [center], Math.round(p.events), 11)
      );
    });
    if (pointer) points.push(...createClusterPoints(fork('probe'), [pointer], 28, 5));
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values: createDensityField(points, p.bandwidth),
        ramp: 'heat',
        style: 'ramp',
        strength: 0.06
      },
      {type: 'sprites', items: toSprites(points, 'ink', 0.7, random, 0.35)}
    ];
  },

  interpolation: ({random, p, pointer}) => {
    const samplePool = createScatterPoints(random, 26, 14);
    const valuePool = samplePool.map(() => random.next());
    const count = Math.round(p.samples);
    const samples = samplePool.slice(0, count);
    const sampleValues = valuePool.slice(0, count);
    if (pointer) {
      samples.push(pointer);
      sampleValues.push(1);
    }
    const values = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
      let weightSum = 0;
      let sum = 0;
      for (let index = 0; index < samples.length; index++) {
        const distance = Math.hypot(
          samples[index][0] - u * SCENE_WIDTH,
          samples[index][1] - v * SCENE_HEIGHT
        );
        const weight = 1 / (distance ** p.power + 4);
        weightSum += weight;
        sum += weight * sampleValues[index];
      }
      return sum / weightSum;
    });
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

  raster: ({random, p}) => {
    const columns = Math.round(p.resolution);
    const rows = Math.max(3, Math.round(columns * 0.625));
    const field = createSmoothField(random, 6, 1.4, columns, rows);
    return [
      {
        type: 'field',
        columns,
        rows,
        values: field.map(value => value ** p.contrast),
        ramp: 'ice',
        style: 'ramp',
        filter: 'nearest',
        strength: 0.1
      }
    ];
  },

  terrain: ({random, p, pointer}) => [
    {
      type: 'field',
      columns: FIELD_COLUMNS,
      rows: FIELD_ROWS,
      values: createSmoothField(random, 10, 2.1),
      ramp: 'terrain',
      style: 'hillshade',
      strength: p.exaggeration,
      light: getLightAngle(pointer ? getPointerAzimuth(pointer) : p.azimuth)
    }
  ],

  'flow-accumulation': ({random, p, pointer}) => {
    const values = createSmoothField(random, 9, 1.9);
    const trunk = createWanderPath(random, [150, 62], Math.PI, 7, 20, 0.5).map(point =>
      clampPoint(point, 8)
    );
    // Each stream carries an accumulation (share of the largest); the threshold hides small ones.
    const streams: (PathItem & {accumulation: number})[] = [
      {points: trunk, role: 'primary', width: 2.6, alpha: 0.95, phase: 0, accumulation: 100}
    ];
    trunk.forEach((junction, index) => {
      if (index < 3 || index % 3 !== 0) return;
      const side = index % 2 === 0 ? 1 : -1;
      const branch = createWanderPath(random, junction, Math.PI + side * 1.3, 6, 8, 0.6)
        .map(point => clampPoint(point))
        .reverse();
      const branchAccumulation = random.range(34, 70);
      streams.push({
        points: branch,
        role: 'cool',
        width: 1.4,
        alpha: 0.85,
        phase: random.next(),
        accumulation: branchAccumulation
      });
      const twig = createWanderPath(random, branch[2] ?? junction, side * 1.2, 5, 5, 0.6)
        .map(point => clampPoint(point))
        .reverse();
      streams.push({
        points: twig,
        role: 'cool',
        width: 0.8,
        alpha: 0.7,
        phase: random.next(),
        accumulation: random.range(6, 30)
      });
    });
    const visible = streams
      .filter(stream => stream.accumulation >= p.threshold)
      .map(({accumulation, ...stream}) => ({
        ...stream,
        width: 0.5 + 2.2 * Math.sqrt(accumulation / 100)
      }));
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values,
        ramp: 'terrain',
        style: 'hillshade',
        strength: 4,
        light: getLightAngle(pointer ? getPointerAzimuth(pointer) : p.azimuth)
      },
      {type: 'strokes', items: visible, animation: 'flow', speed: 0.35, blend: 'add'}
    ];
  },

  contours: ({random, p}) => [
    {
      type: 'field',
      columns: FIELD_COLUMNS,
      rows: FIELD_ROWS,
      values: createSmoothField(random, 8, p.roughness),
      ramp: 'ice',
      style: 'contours',
      levels: Math.round(p.levels),
      speed: 0.12
    }
  ],

  network: ({random, fork, p, pointer}) => {
    const network = createRoadNetwork(random);
    const closed = pickClosedEdges(fork('closures'), network.edges.length, p.closures);
    const source = pointer
      ? findNearestIndex(network.nodes, pointer)
      : Math.floor(network.nodes.length / 2);
    return [
      createGridLayer(),
      ...buildNetworkLayers(network, {closed, source, hops: Math.round(p.hops)})
    ];
  },

  'shortest-path': ({random, fork, p, pointer}) => {
    const network = createRoadNetwork(random);
    const closed = pickClosedEdges(fork('closures'), network.edges.length, p.closures);
    const source = pointer ? findNearestIndex(network.nodes, pointer) : network.columns + 1;
    const target = network.columns * (network.rows - 2) + network.columns - 2;
    const open = {...network, edges: network.edges.filter((_, index) => !closed.has(index))};
    const {distances, previous} = findShortestPaths(open, source);
    const path =
      distances[target] < Infinity
        ? getPathTo(previous, target).map(index => network.nodes[index])
        : [];
    return [
      createGridLayer(),
      ...buildNetworkLayers(network, {closed, pulse: 0}),
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

  isochrones: ({random, p, pointer}) => {
    const network = createRoadNetwork(random);
    const source = pointer ? findNearestIndex(network.nodes, pointer) : network.columns * 3 + 3;
    const {distances} = findShortestPaths(network, source);
    const values = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
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
      return Math.min(best, 200) / 200;
    });
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
        levels: Math.round(p.bands),
        cutoff: p.budget / 100,
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

  flows: ({random, p}) => {
    const hubs: Point[] = [
      [30, 52],
      [78, 24],
      [124, 36],
      [104, 78],
      [58, 82],
      [140, 70]
    ];
    const pool: PathItem[] = [];
    for (let index = 0; index < 44; index++) {
      const from = hubs[random.integer(hubs.length)];
      const to = hubs[random.integer(hubs.length)];
      const bend = random.range(0.15, 0.3) * (random.next() > 0.5 ? 1 : -1);
      const width = random.range(0.6, 2);
      const phase = random.next();
      if (from === to) continue;
      pool.push({
        points: createArc(from, to, bend),
        role: index % 3 === 0 ? 'warm' : 'primary',
        width,
        alpha: 0.9,
        phase
      });
    }
    return [
      createGridLayer(),
      {
        type: 'strokes',
        items: pool.slice(0, Math.round(p.arcs)),
        animation: 'flow',
        speed: p.speed,
        blend: 'add'
      },
      {type: 'sprites', items: toSprites(hubs, 'ink', 3, random), pulse: 0.25}
    ];
  },

  lines: ({random, p}) => {
    const roles: Role[] = ['primary', 'warm', 'cool', 'violet', 'hot'];
    const pool: PathItem[] = Array.from({length: 24}, (_, index) => ({
      points: createWanderPath(
        random,
        [random.range(8, 40), random.range(10, 90)],
        random.range(-0.4, 0.4),
        6,
        22,
        0.5
      ).map(point => clampPoint(point)),
      role: roles[index % roles.length],
      width: random.range(0.7, 1.3) * p.width,
      alpha: 0.95,
      phase: (index * 0.37) % 1
    }));
    return [
      createGridLayer(),
      {
        type: 'strokes',
        items: pool.slice(0, Math.round(p.count)),
        animation: 'draw',
        speed: 0.22,
        blend: 'add'
      }
    ];
  },

  polygons: ({fork, p}) => {
    const items: PolygonItem[] = [];
    const outlines: PathItem[] = [];
    const roles: Role[] = ['primary', 'cool', 'violet', 'warm', 'hot'];
    for (let row = 0; row < 3; row++) {
      for (let column = 0; column < 5; column++) {
        const local = fork(`cell-${row}-${column}`);
        const ring = createBlob(
          local,
          20 + column * 30 + local.range(-3, 3),
          20 + row * 30 + local.range(-3, 3),
          local.range(10, 13),
          Math.round(p.vertices),
          p.roughness
        );
        const value = local.next();
        items.push({
          ring,
          role: roles[Math.floor(value * 5)],
          alpha: 0.2 + value * 0.3,
          phase: local.next()
        });
        outlines.push({points: [...ring, ring[0]], role: 'ink', width: 0.35, alpha: 0.5});
      }
    }
    return [
      {type: 'fills', items, animation: 'pulse', blend: 'add'},
      {type: 'strokes', items: outlines, animation: 'static'}
    ];
  },

  'polygon-overlay': ({fork, p, pointer}) => {
    const first = createBlob(fork('first'), 80 - p.separation / 2, 50, p.size, 18, 0.22, 0.9);
    const secondCenter: Point = pointer
      ? [clamp(pointer[0], 10, 150), clamp(pointer[1], 14, 86)]
      : [80 + p.separation / 2, 50];
    const second = createBlob(
      fork('second'),
      secondCenter[0],
      secondCenter[1],
      p.size,
      18,
      0.22,
      0.9
    );
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

  buffers: ({random, p, pointer}) => {
    const river = smoothPath(
      createWanderPath(random, [10, 62], -0.2, 8, 18, 0.35).map(
        ([x, y]): Point => [x, clamp(y, 18, 82)]
      ),
      2
    );
    const sites = [
      [40, 28],
      [88, 78],
      [128, 30]
    ].map(([x, y]): Point => [x + random.range(-4, 4), y]);
    if (pointer) sites.push(pointer);
    // A distance field to the river and the sites; the buffer is where it is below the distance.
    const values = evaluateField(FIELD_COLUMNS, FIELD_ROWS, (u, v) => {
      const point: Point = [u * SCENE_WIDTH, v * SCENE_HEIGHT];
      let best = Infinity;
      for (let index = 0; index + 1 < river.length; index++) {
        best = Math.min(best, getSegmentDistance(point, river[index], river[index + 1]));
      }
      for (const site of sites) best = Math.min(best, getDistance(point, site));
      return Math.min(best, BUFFER_FIELD_RANGE) / BUFFER_FIELD_RANGE;
    });
    return [
      {
        type: 'field',
        columns: FIELD_COLUMNS,
        rows: FIELD_ROWS,
        values,
        ramp: 'ice',
        style: 'expand',
        levels: 3,
        cutoff: p.distance / BUFFER_FIELD_RANGE,
        speed: 0.15
      },
      createGridLayer(),
      {
        type: 'strokes',
        items: [{points: river, role: 'cool', width: 1.4, alpha: 0.95}],
        animation: 'static',
        blend: 'add'
      },
      {type: 'sprites', items: toSprites(sites, 'warm', 2.6), pulse: 0.2}
    ];
  },

  tessellation: ({random, p}) => {
    const ring = createBlob(random, 80, 50, 36, Math.round(p.vertices), p.roughness, 0.85);
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

  cells: ({random, p}) => {
    const radius = p.size;
    const classes = Math.round(p.classes);
    const width = Math.sqrt(3) * radius;
    const field = createSmoothField(random, 6, 1.5);
    const items: PolygonItem[] = [];
    const rowCount = Math.ceil(SCENE_HEIGHT / (radius * 1.5)) + 1;
    const columnCount = Math.ceil(SCENE_WIDTH / width) + 1;
    for (let row = -1; row < rowCount; row++) {
      for (let column = -1; column < columnCount; column++) {
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
          role: getClassRole(Math.min(classes - 1, Math.floor(value * classes)), classes),
          alpha: 0.18 + value * 0.5,
          phase: x / SCENE_WIDTH
        });
      }
    }
    return [{type: 'fills', items, animation: 'pulse', blend: 'add'}];
  },

  'spatial-weights': ({random, p, pointer}) => {
    const points = createScatterPoints(random, 46, 10);
    const background: PathItem[] = [];
    const focusLinks: PathItem[] = [];
    const focus = pointer ? findNearestIndex(points, pointer) : 18;
    const neighbors: number[] = [];
    const seen = new Set<number>();
    points.forEach((point, index) => {
      const nearest = points
        .map((other, otherIndex) => ({otherIndex, distance: getDistance(point, other)}))
        .filter(entry => entry.otherIndex !== index)
        .sort((first, second) => first.distance - second.distance)
        .slice(0, Math.round(p.neighbors));
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
          continue;
        }
        const key = Math.min(index, entry.otherIndex) * 1000 + Math.max(index, entry.otherIndex);
        if (seen.has(key) || entry.otherIndex === focus) continue;
        seen.add(key);
        background.push({
          points: [point, points[entry.otherIndex]],
          role: 'dim',
          width: 0.4,
          alpha: 0.5
        });
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

  regression: ({random, p}) => {
    const left = 18;
    const right = 148;
    const trueLine = (x: number) => 50 - p.slope * (x - 83);
    const points = Array.from({length: 54}, () => {
      const x = random.range(left, right);
      return [x, clamp(trueLine(x) + random.gaussian() * p.noise, 8, 92)] as Point;
    });
    // Ordinary least squares fit of the points, the line the picture is about.
    const meanX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
    const meanY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
    let covariance = 0;
    let variance = 0;
    for (const [x, y] of points) {
      covariance += (x - meanX) * (y - meanY);
      variance += (x - meanX) ** 2;
    }
    const fitSlope = covariance / (variance || 1);
    const lineAt = (x: number): Point => [x, meanY + fitSlope * (x - meanX)];
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

  histogram: ({random, p}) => {
    const binCount = Math.round(p.bins);
    const counts = new Array<number>(binCount).fill(0);
    for (let index = 0; index < 700; index++) {
      const tail = random.next() < 0.12;
      const value = tail ? random.range(0.5, 1) : 0.45 + random.gaussian() * p.spread;
      counts[Math.min(binCount - 1, Math.max(0, Math.floor(value * binCount)))]++;
    }
    const tallest = Math.max(...counts, 1);
    const width = 128 / binCount;
    const gap = Math.min(1, width * 0.2);
    const items: PolygonItem[] = counts.map((count, index) => {
      const height = 2 + (count / tallest) * 74;
      const left = 16 + index * width + gap / 2;
      return {
        ring: [
          [left, 90],
          [left, 90 - height],
          [left + width - gap, 90 - height],
          [left + width - gap, 90]
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

  classification: ({random, p}) => {
    const values = createSmoothField(random, 6, 1.4, 16, 10);
    const classes = Math.round(p.classes);
    const items: PolygonItem[] = [];
    for (let row = 0; row < 10; row++) {
      for (let column = 0; column < 16; column++) {
        const value = values[row * 16 + column] ** p.skew;
        const bin = Math.min(classes - 1, Math.floor(value * classes));
        const left = 12 + column * 8.5;
        const top = 6 + row * 7.8;
        items.push({
          ring: [
            [left, top],
            [left + 7.6, top],
            [left + 7.6, top + 7],
            [left, top + 7]
          ],
          role: getClassRole(bin, classes),
          alpha: 0.5,
          phase: bin / classes
        });
      }
    }
    const legendWidth = 136 / classes;
    for (let index = 0; index < classes; index++) {
      const left = 12 + index * legendWidth;
      items.push({
        ring: [
          [left, 88],
          [left + legendWidth - 2, 88],
          [left + legendWidth - 2, 93],
          [left, 93]
        ],
        role: getClassRole(index, classes),
        alpha: 0.85,
        phase: index / classes
      });
    }
    return [{type: 'fills', items, animation: 'pulse', blend: 'add'}];
  },

  'time-series': ({random, p}) => {
    const roles: Role[] = ['primary', 'warm', 'violet'];
    const strokes: PathItem[] = [];
    const areas: PolygonItem[] = [];
    const heads: SpriteItem[] = [];
    const windowSize = Math.round(p.window);
    const visibleSeries = Math.round(p.series);
    roles.forEach((role, seriesIndex) => {
      let value = random.range(40, 60);
      const raw: number[] = [];
      for (let index = 0; index <= 40; index++) {
        value = clamp(
          value + random.gaussian() * 6 + Math.sin(index / 5 + seriesIndex) * 2.5,
          18,
          86
        );
        raw.push(value);
      }
      if (seriesIndex >= visibleSeries) return;
      const toPoints = (series: number[]): Point[] =>
        series.map((entry, index) => [14 + index * 3.35, 100 - entry]);
      const smooth = toPoints(getTrailingMean(raw, windowSize));
      if (windowSize > 1) {
        strokes.push({
          points: toPoints(raw),
          role,
          width: 0.5,
          alpha: 0.35,
          phase: seriesIndex * 0.1
        });
      }
      strokes.push({points: smooth, role, width: 1.3, alpha: 0.95, phase: seriesIndex * 0.1});
      if (seriesIndex === 0)
        areas.push({ring: [[14, 92], ...smooth, [148, 92]], role, alpha: 0.12});
      heads.push({x: smooth[40][0], y: smooth[40][1], radius: 2.6, role, phase: seriesIndex / 3});
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

  trajectories: ({random, p}) => {
    const roles: Role[] = ['primary', 'warm', 'cool', 'hot', 'violet'];
    const pool: PathItem[] = Array.from({length: 32}, (_, index) => ({
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
      {
        type: 'strokes',
        items: pool.slice(0, Math.round(p.count)),
        animation: 'flow',
        speed: 0.25,
        trail: p.trail,
        blend: 'add'
      }
    ];
  },

  'point-cloud': ({random, p}) => {
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
    // The angle is advanced on the CPU by the runtime (spin and pointer), so speed stays 0.
    return [
      {
        type: 'sprites',
        items,
        turntable: {angle: 0.5, speed: 0, perspective: p.depth / 1000}
      }
    ];
  },

  'affine-transform': ({p}) => {
    // An asymmetric "F" so rotation, skew and mirroring are readable. Units are scene units.
    const shape: Point[] = [
      [-10, -20],
      [14, -20],
      [14, -13],
      [-3, -13],
      [-3, -4],
      [10, -4],
      [10, 3],
      [-3, 3],
      [-3, 20],
      [-10, 20]
    ];
    const radians = (p.rotate * Math.PI) / 180;
    const cosine = Math.cos(radians) * p.scale;
    const sine = Math.sin(radians) * p.scale;
    // [a b; d e] = Rotation * Shear(x += skew * y) * uniform scale.
    const a = cosine;
    const b = cosine * p.skew - sine;
    const d = sine;
    const e = sine * p.skew + cosine;
    const apply = ([x, y]: Point, centerX: number, matrix: boolean): Point =>
      matrix ? [centerX + a * x + b * y, 50 + d * x + e * y] : [centerX + x, 50 + y];
    const getGridLines = (centerX: number, matrix: boolean): PathItem[] => {
      const lines: PathItem[] = [];
      for (let offset = -30; offset <= 30; offset += 10) {
        lines.push({
          points: [apply([offset, -30], centerX, matrix), apply([offset, 30], centerX, matrix)],
          role: 'dim',
          width: 0.25,
          alpha: matrix ? 0.55 : 0.3
        });
        lines.push({
          points: [apply([-30, offset], centerX, matrix), apply([30, offset], centerX, matrix)],
          role: 'dim',
          width: 0.25,
          alpha: matrix ? 0.55 : 0.3
        });
      }
      return lines;
    };
    const sourceRing = shape.map(point => apply(point, 40, false));
    const resultRing = shape.map(point => apply(point, 118, true));
    // Basis vectors: the columns of the matrix, drawn from the shape's origin.
    const origin: Point = [118, 50];
    const basisX: Point = [118 + a * 24, 50 + d * 24];
    const basisY: Point = [118 + b * 24, 50 + e * 24];
    return [
      {type: 'strokes', items: getGridLines(40, false), animation: 'static'},
      {type: 'strokes', items: getGridLines(118, true), animation: 'static'},
      {type: 'fills', items: [{ring: sourceRing, role: 'dim', alpha: 0.35}], animation: 'static'},
      {
        type: 'fills',
        items: [{ring: resultRing, role: 'primary', alpha: 0.4}],
        animation: 'static',
        blend: 'add'
      },
      {
        type: 'strokes',
        items: [
          {points: [...sourceRing, sourceRing[0]], role: 'ink', width: 0.6, alpha: 0.6},
          {points: [...resultRing, resultRing[0]], role: 'primary', width: 0.9, alpha: 0.95},
          {points: [origin, basisX], role: 'warm', width: 1.4, alpha: 0.95},
          {points: [origin, basisY], role: 'cool', width: 1.4, alpha: 0.95},
          {
            points: [
              [62, 50],
              [92, 50]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.9
          },
          {
            points: [
              [87, 46],
              [92, 50],
              [87, 54]
            ],
            role: 'dim',
            width: 0.5,
            alpha: 0.9
          }
        ],
        animation: 'static'
      },
      {
        type: 'sprites',
        items: [
          {x: basisX[0], y: basisX[1], radius: 1.8, role: 'warm', phase: 0},
          {x: basisY[0], y: basisY[1], radius: 1.8, role: 'cool', phase: 0.5}
        ],
        pulse: 0.2
      }
    ];
  },

  'cell-cover': ({fork, p}) => {
    const polygon = createBlob(fork('polygon'), 80, 50, 34, Math.round(p.vertices), 0.3, 0.85);
    const contains = ([x, y]: Point): boolean => {
      let inside = false;
      for (
        let index = 0, previous = polygon.length - 1;
        index < polygon.length;
        previous = index++
      ) {
        const [xi, yi] = polygon[index];
        const [xj, yj] = polygon[previous];
        if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
      return inside;
    };
    // Resolution maps to cell radius: each level is a finer grid.
    const radius = 22 / 1.35 ** (p.resolution - 3);
    const width = Math.sqrt(3) * radius;
    const lit: PolygonItem[] = [];
    const quiet: PathItem[] = [];
    const rowCount = Math.ceil(SCENE_HEIGHT / (radius * 1.5)) + 1;
    const columnCount = Math.ceil(SCENE_WIDTH / width) + 1;
    for (let row = -1; row < rowCount; row++) {
      for (let column = -1; column < columnCount; column++) {
        const x = column * width + (row % 2 ? width / 2 : 0);
        const y = row * radius * 1.5;
        const ring = createRegularPolygon(x, y, radius * 0.95, 6, Math.PI / 6);
        if (contains([x, y])) {
          lit.push({ring, role: 'primary', alpha: 0.55, phase: (x + y) / 260});
        } else {
          quiet.push({points: [...ring, ring[0]], role: 'dim', width: 0.2, alpha: 0.35});
        }
      }
    }
    return [
      {type: 'strokes', items: quiet, animation: 'static'},
      {type: 'fills', items: lit, animation: 'pulse', blend: 'add'},
      {
        type: 'strokes',
        items: [{points: [...polygon, polygon[0]], role: 'warm', width: 1.1, alpha: 0.95}],
        animation: 'static'
      }
    ];
  },

  pipeline: ({random, p}) => {
    const branches = Math.round(p.branches);
    const middleRows = [[40], [18, 62], [8, 40, 72]][branches - 1];
    const boxes: {x: number; y: number; role: Role}[] = [
      {x: 8, y: 40, role: 'cool'},
      ...middleRows.map(y => ({x: 46, y, role: 'primary' as Role})),
      {x: 86, y: 40, role: 'violet'},
      {x: 124, y: 40, role: 'warm'}
    ];
    const mergeIndex = branches + 1;
    const edges: [number, number][] = [
      ...middleRows.map((_, index): [number, number] => [0, index + 1]),
      ...middleRows.map((_, index): [number, number] => [index + 1, mergeIndex]),
      [mergeIndex, mergeIndex + 1]
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
      {type: 'strokes', items: wires, animation: 'flow', speed: p.speed, blend: 'add'}
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

/** Compass azimuth in degrees of a scene point seen from the scene center. */
function getPointerAzimuth(pointer: Point): number {
  const degrees =
    (Math.atan2(pointer[0] - SCENE_WIDTH / 2, -(pointer[1] - SCENE_HEIGHT / 2)) * 180) / Math.PI;
  return (degrees + 360) % 360;
}

/**
 * Deterministic data for one preview. When `entryId` has a function-specific builder
 * (entries/*.ts) it draws instead of the generic scene of `kind`. `parameters` override the
 * defaults; `pointer` is a scene-space position for previews that react to it.
 */
export function buildPreview(
  kind: PreviewKind,
  seed = 1,
  parameters: PreviewParameterValues = {},
  pointer: Point | null = null,
  entryId?: string
): PreviewData {
  const entryPreview = entryId ? getEntryPreview(entryId) : undefined;
  const base = hashText(entryPreview ? entryId! : kind) ^ Math.imul(seed, 2654435761);
  const values = {...getDefaultParameterValues(kind, entryId), ...parameters};
  const build = entryPreview?.build ?? builders[kind];
  return {
    layers: build({
      random: createRandom(base),
      fork: label => createRandom(base ^ hashText(label)),
      p: values,
      pointer
    })
  };
}
