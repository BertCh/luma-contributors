// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs of the spatial statistics runs: points with a spatial field, k-nearest-neighbor
// weights, and helpers to mix noise into the field and to draw the neighbor graph.

import events from '../../../../data/kilauea-events.json';
import {projectObservedPositions} from '../../../../data/observed-positions';
import {getRainStations} from '../../../../data/point-observations';
import type {RunLayer} from '../../../define-run';
import type {Random, World} from '../../../inputs';
import {createDistanceBandWeights, createKnnWeights, type CsrWeights} from '../../../inputs';
import type {RunGlyph, RunPointer} from '../../../interaction';
import {chooseSticky, createPointIndex, type PointIndex} from '../../../pick';

/** Share of the world height (from the bottom) reserved for a chart under the map. */
export const CHART_BAND = 0.3;

export type PointField = {
  count: number;
  /** Width of the world the points were generated in. */
  worldWidth: number;
  positions: Float32Array;
  /** Normalized measured rainfall in [0, 1]. */
  field: Float32Array;
  /** Independent noise in [0, 1], mixed in by `mixNoise`. */
  noise: Float32Array;
};

export type NeighborGraph = {
  weights: CsrWeights;
  /** Neighbor graph as node pairs (uint32x2), one per weights slot. */
  edges: Uint32Array;
  edgeCount: number;
};

/** k-nearest-neighbor weights of `positions` plus the graph edges for drawing. */
export function createNeighborGraph(positions: Float32Array, neighborCount: number): NeighborGraph {
  const weights = createKnnWeights(positions, neighborCount);
  const edges = createWeightEdges(weights);
  return {weights, edges, edgeCount: edges.length / 2};
}

/** Distance-band weights (pairs closer than `radius`) plus the graph edges; sparse areas become islands. */
export function createBandGraph(positions: Float32Array, radius: number): NeighborGraph {
  const weights = createDistanceBandWeights(positions, radius);
  const edges = createWeightEdges(weights);
  return {weights, edges, edgeCount: edges.length / 2};
}

/** Pairs (row, neighbor) for every slot of the weights, for drawing the neighbor graph. */
export function createWeightEdges(weights: CsrWeights): Uint32Array {
  const edges = new Uint32Array(weights.neighbors.length * 2);
  for (let row = 0; row < weights.rowCount; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      edges[slot * 2] = row;
      edges[slot * 2 + 1] = weights.neighbors[slot];
    }
  }
  return edges;
}

export type PointProbeInputs = PointField & {index: PointIndex; focus: {row: number}};

/** A point field plus the lookups a hover probe needs. */
export function createProbeInputs(points: PointField): PointProbeInputs {
  return {...points, index: createPointIndex(points.positions), focus: {row: -1}};
}

/** The point nearest the pointer, kept while the pointer stays close to it; -1 when none. */
export function pickProbeRow(
  inputs: {index: PointIndex; focus: {row: number}},
  pointer: RunPointer
): number {
  inputs.focus.row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
  return inputs.focus.row;
}

/** Value of `row` after `noiseShare` of the field was replaced by noise (see `mixNoise`). */
export function getMixedValue(points: PointField, row: number, noiseShare: number): number {
  return points.field[row] * (1 - noiseShare) + points.noise[row] * noiseShare;
}

/** Neighbors of `row` in the weights (a view, not a copy). */
export function getNeighborRows(weights: CsrWeights, row: number): Uint32Array {
  return weights.neighbors.subarray(weights.offsets[row], weights.offsets[row + 1]);
}

/** Links from `row` to each of its neighbors, with a dot on the point and on every neighbor. */
export function createNeighborGlyphs(
  positions: Float32Array,
  weights: CsrWeights,
  row: number
): RunGlyph[] {
  const from = [positions[row * 2], positions[row * 2 + 1]] as const;
  const glyphs: RunGlyph[] = [];
  for (const neighbor of getNeighborRows(weights, row)) {
    const to = [positions[neighbor * 2], positions[neighbor * 2 + 1]] as const;
    glyphs.push({shape: 'line', from, to}, {shape: 'dot', at: to});
  }
  glyphs.push({shape: 'dot', at: from});
  return glyphs;
}

/**
 * Carolina rain stations over the map part of the world (the lower `chartBand` is left for a
 * chart; 0 uses the whole world). The window is the Carolinas, so the stations fill the frame, and
 * the field is rainfall scaled so its 95th percentile is 1.
 */
export function createPointField(
  random: Random,
  world: World,
  count = 700,
  chartBand = CHART_BAND
): PointField {
  const mapHeight = world.height - chartBand;
  const stations = getRainStations(random, {width: world.width, height: mapHeight}, count, true);
  const offset = world.height - mapHeight;
  const positions = Float32Array.from(stations.positions, (value, index) =>
    index % 2 === 1 ? value + offset : value
  );
  const sorted = Float32Array.from(stations.rain).sort();
  const scale = 1 / Math.max(sorted[Math.floor(count * 0.95)], 1);
  const field = Float32Array.from(stations.rain, rain => Math.min(1, rain * scale));
  const noise = Float32Array.from({length: count}, () => random());
  return {count, worldWidth: world.width, positions, field, noise};
}

/** `field` blended with `noise`: share 0 is the smooth field, 1 is pure noise. */
export function mixNoise(field: Float32Array, noise: Float32Array, share: number): Float32Array {
  return Float32Array.from(field, (value, index) => value * (1 - share) + noise[index] * share);
}

export type SpaceTimeEvents = {
  count: number;
  /** True local coordinates in kilometers (equirectangular at the swarm's mean latitude): what the operations read. */
  positions: Float32Array;
  /** The stretched layout the dots, glyphs and picks use; never given to an operation. */
  drawnPositions: Float32Array;
  observedDays: Float32Array;
  shuffledOrder: Uint32Array;
  index: PointIndex;
  focus: {row: number};
};

/**
 * M3.5+ HVO epicenters during the 2018 Kīlauea eruption. `positions` are true local kilometers, so
 * distances and the weights band are in km. The swarm is dense in a small core, so for drawing
 * only, `drawnPositions` blend the linear position (2nd to 98th percentile) with the rank on each
 * axis, spreading the core across the map above the chart band.
 */
export function createSpaceTimeEvents(random: Random, world: World): SpaceTimeEvents {
  const count = events.length;
  const projected = projectObservedPositions(events, {
    width: world.width,
    height: world.height - CHART_BAND
  });
  const drawnPositions = zoomToQuantiles(projected, world.width, world.height - CHART_BAND);
  for (let row = 0; row < count; row++) drawnPositions[row * 2 + 1] += CHART_BAND;
  const meanLatitude = events.reduce((sum, [, latitude]) => sum + latitude, 0) / count;
  const kilometersPerDegree = 111.32;
  const positions = Float32Array.from(
    events.flatMap(([longitude, latitude]) => [
      longitude * Math.cos((meanLatitude * Math.PI) / 180) * kilometersPerDegree,
      latitude * kilometersPerDegree
    ])
  );
  const observedDays = Float32Array.from(events, event => event[2]);
  const shuffledOrder = Uint32Array.from({length: count}, (_, row) => row);
  for (let row = count - 1; row > 0; row--) {
    const other = Math.floor(random() * (row + 1));
    [shuffledOrder[row], shuffledOrder[other]] = [shuffledOrder[other], shuffledOrder[row]];
  }
  return {
    count,
    positions,
    drawnPositions,
    observedDays,
    shuffledOrder,
    index: createPointIndex(drawnPositions),
    focus: {row: -1}
  };
}

/** Shuffle the selected share of times among the observed sites, preserving the time distribution. */
export function getEventTimes(events: SpaceTimeEvents, observedShare: number): Float32Array {
  const days = events.observedDays.slice();
  const shuffleCount = Math.round(events.count * (1 - observedShare));
  for (let index = 0; index < shuffleCount; index++) {
    days[events.shuffledOrder[index]] =
      events.observedDays[events.shuffledOrder[(index + 1) % shuffleCount]];
  }
  return days;
}

function zoomToQuantiles(positions: Float32Array, width: number, height: number): Float32Array {
  const count = positions.length / 2;
  const zoomed = new Float32Array(positions.length);
  for (const [axis, size] of [
    [0, width],
    [1, height]
  ] as const) {
    const sorted = Float32Array.from({length: count}, (_, row) => positions[row * 2 + axis]).sort();
    const minimum = sorted[Math.floor(count * 0.02)];
    const maximum = sorted[Math.floor(count * 0.98)];
    for (let row = 0; row < count; row++) {
      const value = positions[row * 2 + axis];
      const linear = Math.min(1, Math.max(0, (value - minimum) / (maximum - minimum)));
      // Rank in the sorted values: spreads the dense core that the linear fit squeezes together.
      let low = 0;
      let high = count;
      while (low < high) {
        const middle = (low + high) >> 1;
        if (sorted[middle] < value) low = middle + 1;
        else high = middle;
      }
      const rank = low / count;
      zoomed[row * 2 + axis] = size * (0.03 + 0.94 * (0.4 * linear + 0.6 * rank));
    }
  }
  return zoomed;
}

/**
 * The point under the pointer with its neighbor links (and, when `ringRadius` is given, a ring of
 * the weights' reach). `row` is -1 and `glyphs` empty when no point is near.
 */
export function probeNeighbors(
  inputs: PointProbeInputs,
  pointer: RunPointer,
  weights: CsrWeights,
  ringRadius?: number
): {row: number; neighborCount: number; glyphs: RunGlyph[]} {
  const row = pickProbeRow(inputs, pointer);
  if (row < 0) return {row, neighborCount: 0, glyphs: []};
  const glyphs = createNeighborGlyphs(inputs.positions, weights, row);
  glyphs.push({
    shape: 'crosshair',
    at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
    size: 12
  });
  if (ringRadius !== undefined) {
    glyphs.push({
      shape: 'ring',
      at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
      radius: ringRadius,
      dashed: true
    });
  }
  return {row, neighborCount: getNeighborRows(weights, row).length, glyphs};
}

/**
 * Variable `variable` of `row` in the multivariate runs: the smooth field with a share `noiseShare`
 * of a per-variable pseudo-random value mixed in, so the variables share one spatial pattern.
 */
export function getVariableValue(
  field: Float32Array,
  row: number,
  variable: number,
  noiseShare: number
): number {
  const jitter = Math.sin((row * 7 + variable * 131) * 12.9898) * 43758.5453;
  return field[row] * (1 - noiseShare) + (jitter - Math.floor(jitter)) * noiseShare;
}

/** The values of every variable at every row, row-major, for the operation's `values` view. */
export function createInterleavedValues(
  field: Float32Array,
  variableCount: number,
  noiseShare: number
): Float32Array {
  const interleaved = new Float32Array(field.length * variableCount);
  for (let row = 0; row < field.length; row++) {
    for (let variable = 0; variable < variableCount; variable++) {
      interleaved[row * variableCount + variable] = getVariableValue(
        field,
        row,
        variable,
        noiseShare
      );
    }
  }
  return interleaved;
}

/**
 * Links from event `row` to its spatial neighbors: solid when the neighbor's time is within
 * `timeWindow` of the event's, dashed otherwise. Also returns the pair counts and the mean time
 * difference to the neighbors.
 */
export function createTimedNeighborGlyphs(
  positions: Float32Array,
  weights: CsrWeights,
  row: number,
  times: Float32Array,
  timeWindow: number,
  ringRadius?: number
): {glyphs: RunGlyph[]; neighborCount: number; closeCount: number; meanDifference: number} {
  const from = [positions[row * 2], positions[row * 2 + 1]] as const;
  const glyphs: RunGlyph[] = [];
  let closeCount = 0;
  let totalDifference = 0;
  const neighbors = getNeighborRows(weights, row);
  for (const neighbor of neighbors) {
    const to = [positions[neighbor * 2], positions[neighbor * 2 + 1]] as const;
    const difference = Math.abs(times[neighbor] - times[row]);
    const isClose = difference <= timeWindow;
    if (isClose) closeCount++;
    totalDifference += difference;
    glyphs.push({shape: 'line', from, to, dashed: !isClose}, {shape: 'dot', at: to});
  }
  glyphs.push({shape: 'crosshair', at: from, size: 12});
  if (ringRadius !== undefined)
    glyphs.push({shape: 'ring', at: from, radius: ringRadius, dashed: true});
  return {
    glyphs,
    neighborCount: neighbors.length,
    closeCount,
    meanDifference: neighbors.length > 0 ? totalDifference / neighbors.length : 0
  };
}

/** `[layer]` when `count` is positive and no layers otherwise: a dots layer with `count` 0 still draws one dot. */
export function whenShown(count: number, layer: RunLayer): RunLayer[] {
  return count > 0 ? [layer] : [];
}
