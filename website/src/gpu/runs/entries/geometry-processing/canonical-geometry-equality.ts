// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCanonicalGeometryEquality answers "is this the same geometry?" for pairs of trees. Six
// features (a lake polygon, its river and a gauge point) arrive in two deliveries, drawn on top of
// each other: A wide and muted, B thin and colored by the flag the operation wrote for its pair.
// B stores the top row differently (part order, ring start and winding, line direction), which
// GPUNormalizeGeometry removes, so those pairs are equal. B edits the bottom row: a vertex moved
// (unequal once it is farther than the tolerance), a vertex added and a part missing (unequal at
// any tolerance). The pointer carries the nearest vertex of B: the pair stays equal while the
// vertex is inside the tolerance box around its place in A.

import {
  GPUCanonicalGeometryEquality,
  GPUNormalizeGeometry
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getSwatchWidth, getTextWidth, type RunLabel} from '../../text';
import {
  createFeatureDeliveries,
  DELIVERY_VARIANTS,
  FEATURE_COUNT,
  getDeliveryVertex,
  writeDeliveryPositions,
  type DeliveryTree,
  type FeatureDeliveries,
  type VertexOverride
} from './shared/feature-deliveries';
import {createStaleGuard, readFresh, type StaleGuard} from './shared/pointer-support';
import {addKernel} from './shared/shape-segments';
import {parameters} from './canonical-geometry-equality.parameters';

/** A feature is a collection holding a polygon holding a ring: three nodes deep. */
const NESTING_DEPTH = 3;
/** Radius of the mark at the first stored vertex of a line or ring (world units). */
const START_RADIUS = 0.017;
/** Radius of the ring around what delivery B added or left out (world units). */
const CALLOUT_RADIUS = 0.026;
const RING_SEGMENT_COUNT = 20;
/** Segments per start: the direction chevron's two wings. */
const CHEVRON_SEGMENT_COUNT = 2;
const VERDICTS = ['not equal', 'equal', 'invalid'];

type Hand = {
  /** The vertex of delivery B the pointer carries, and where to. */
  carry: VertexOverride | null;
  pointer: readonly [number, number] | null;
  shift: number;
  /**
   * The shift slider moved under a parked pointer: the vertex is let go until the pointer moves,
   * or the slider would seem to do nothing. A tolerance change keeps the carry, so a reader can
   * park a vertex and sweep the tolerance across it.
   */
  isReleased: boolean;
};

type Inputs = {
  deliveries: FeatureDeliveries;
  /** Lines and rings that start differently in the two deliveries: see `findRestartedLeaves`. */
  restarted: number[][];
  hand: Hand;
  guard: StaleGuard;
};

/** The vertices of delivery B that are not where delivery A has them. */
function getOverrides({deliveries, hand}: Inputs, shift: number): VertexOverride[] {
  const {homes, movedVertex} = deliveries;
  const moved: VertexOverride = {
    vertex: movedVertex,
    at: [homes[movedVertex * 2] + shift, homes[movedVertex * 2 + 1]]
  };
  return hand.carry ? [hand.carry, moved] : [moved];
}

/** The vertex of delivery B nearest the pointer, where B stores it; the carried one is sticky. */
function findCarriedVertex(inputs: Inputs, x: number, y: number, shift: number): number {
  const {deliveries, hand} = inputs;
  const stored = getOverrides({...inputs, hand: {...hand, carry: null}}, shift);
  const getDistance = (vertex: number) => {
    const [vertexX, vertexY] = getDeliveryVertex(deliveries, vertex, stored);
    return Math.hypot(vertexX - x, vertexY - y);
  };
  let nearest = -1;
  let nearestDistance = Infinity;
  for (const vertex of deliveries.carriable) {
    const distance = getDistance(vertex);
    if (distance < nearestDistance) {
      nearest = vertex;
      nearestDistance = distance;
    }
  }
  const previous = hand.carry?.vertex ?? -1;
  if (previous < 0 || previous === nearest) return nearest;
  return nearestDistance > getDistance(previous) * 0.8 ? previous : nearest;
}

/**
 * The lines and rings whose stored vertex sequence starts differently in the two deliveries, as
 * the first and second stored row in each: `[leftFirst, leftSecond, rightFirst, rightSecond]`.
 */
function findRestartedLeaves({left, right}: FeatureDeliveries): number[][] {
  const restarted: number[][] = [];
  for (const leaf of left.leaves) {
    const twin = right.leaves.find(
      candidate => candidate.feature === leaf.feature && candidate.part === leaf.part
    );
    if (!twin || leaf.rowCount < 2) continue;
    const isSameStart = [0, 1].every(
      offset =>
        left.vertexOfRow[leaf.firstRow + offset] === right.vertexOfRow[twin.firstRow + offset]
    );
    if (!isSameStart) {
      restarted.push([leaf.firstRow, leaf.firstRow + 1, twin.firstRow, twin.firstRow + 1]);
    }
  }
  return restarted;
}

/** The first stored vertex of the given lines and rings, and a chevron along the first edge. */
function getStartMarks(
  rows: readonly (readonly [number, number])[],
  positions: Float32Array
): {starts: Float32Array; chevrons: Float32Array} {
  const starts = new Float32Array(rows.length * 2);
  const chevrons = new Float32Array(rows.length * CHEVRON_SEGMENT_COUNT * 4);
  rows.forEach(([first, second], mark) => {
    const x = positions[first * 2];
    const y = positions[first * 2 + 1];
    const deltaX = positions[second * 2] - x;
    const deltaY = positions[second * 2 + 1] - y;
    const length = Math.hypot(deltaX, deltaY) || 1;
    const directionX = deltaX / length;
    const directionY = deltaY / length;
    const tipX = x + directionX * START_RADIUS * 3.2;
    const tipY = y + directionY * START_RADIUS * 3.2;
    starts.set([x, y], mark * 2);
    [1, -1].forEach((side, wing) => {
      chevrons.set(
        [
          tipX,
          tipY,
          tipX - directionX * START_RADIUS * 1.3 - side * directionY * START_RADIUS,
          tipY - directionY * START_RADIUS * 1.3 + side * directionX * START_RADIUS
        ],
        (mark * CHEVRON_SEGMENT_COUNT + wing) * 4
      );
    });
  });
  return {starts, chevrons};
}

/** Circle outlines around points, as segments for a `lines` layer. */
function getRingSegments(centers: ArrayLike<number>, radius: number): Float32Array {
  const count = centers.length / 2;
  const segments = new Float32Array(count * RING_SEGMENT_COUNT * 4);
  for (let center = 0; center < count; center++) {
    for (let step = 0; step < RING_SEGMENT_COUNT; step++) {
      const from = (step / RING_SEGMENT_COUNT) * Math.PI * 2;
      const to = ((step + 1) / RING_SEGMENT_COUNT) * Math.PI * 2;
      segments.set(
        [
          centers[center * 2] + Math.cos(from) * radius,
          centers[center * 2 + 1] + Math.sin(from) * radius,
          centers[center * 2] + Math.cos(to) * radius,
          centers[center * 2 + 1] + Math.sin(to) * radius
        ],
        (center * RING_SEGMENT_COUNT + step) * 4
      );
    }
  }
  return segments;
}

/** Uploads one delivery and adds its GPUNormalizeGeometry; returns the tree the comparison reads. */
function addNormalizedTree(
  builder: RunBuilder,
  id: string,
  tree: DeliveryTree,
  positions: RunBuffer<'float32x2'>
) {
  const nodes = builder.upload(`${id}-nodes`, 'uint32x4', tree.nodes);
  const children = builder.upload(`${id}-children`, 'uint32', tree.children);
  const normalized = builder.allocate(`${id}-normalized`, 'float32x2', tree.rowCount);
  const ordered = builder.allocate(`${id}-normalized-children`, 'uint32', tree.children.length);
  builder.graph.add(
    new GPUNormalizeGeometry({
      id: `normalize-${id}`,
      positions: positions.view,
      nodes: nodes.view,
      children: children.view,
      maximumNestingDepth: NESTING_DEPTH,
      output: {
        positions: normalized.view,
        children: ordered.view,
        status: builder.allocate(`${id}-status`, 'uint32', tree.nodes.length / 4).view
      }
    })
  );
  return {positions: normalized.view, nodes: nodes.view, children: ordered.view};
}

/**
 * The largest per-coordinate difference between a feature's vertices in the two deliveries, as
 * the float32 difference the operation compares with the tolerance.
 */
function getFeatureOffset(inputs: Inputs, feature: number, shift: number): number {
  const {deliveries} = inputs;
  const overrides = getOverrides(inputs, shift);
  let offset = 0;
  for (let vertex = 0; vertex < deliveries.sharedCount; vertex++) {
    if (deliveries.featureOfVertex[vertex] !== feature) continue;
    const [x, y] = getDeliveryVertex(deliveries, vertex, overrides);
    offset = Math.max(
      offset,
      Math.abs(Math.fround(Math.fround(x) - deliveries.homes[vertex * 2])),
      Math.abs(Math.fround(Math.fround(y) - deliveries.homes[vertex * 2 + 1]))
    );
  }
  return offset;
}

/** Why a pair is or is not equal, from what delivery B did to the feature. */
function describeFeature(inputs: Inputs, feature: number, values: Record<string, number>): string {
  const {left, right} = inputs.deliveries;
  const variant = DELIVERY_VARIANTS[feature];
  if (variant.id === 'missing') {
    return `${left.partOrder[feature].length} vs ${right.partOrder[feature].length} parts`;
  }
  // A part with another vertex count is unequal whatever the coordinates are.
  for (const leaf of left.leaves) {
    const twin = right.leaves.find(
      candidate => candidate.feature === feature && candidate.part === leaf.part
    );
    if (leaf.feature === feature && twin && twin.rowCount !== leaf.rowCount) {
      return `${leaf.part} has ${leaf.rowCount} vs ${twin.rowCount} vertices`;
    }
  }
  const offset = getFeatureOffset(inputs, feature, values.shift);
  if (offset === 0) return variant.label;
  const isOutside = offset > Math.fround(values.tolerance);
  return `offset ${offset.toFixed(3)} ${isOutside ? '>' : '≤'} tolerance ${values.tolerance.toFixed(3)}`;
}

export default defineOperationRun({
  parameters,
  caption:
    "Two deliveries overlaid, B colored by its pair's flag: stored differently is equal; edits past the tolerance or to the structure are not.",
  buildCache: 5,
  createInputs: ({world}): Inputs => {
    const deliveries = createFeatureDeliveries(world);
    const maximumTolerance = parameters[0].max;
    // A ring whose two westernmost vertices are closer than the tolerance could get another
    // canonical start from an edit inside the tolerance; the lakes are sampled to rule that out.
    if (deliveries.startGap <= maximumTolerance) {
      throw new Error(
        `canonical-geometry-equality: ring start gap ${deliveries.startGap} must exceed the maximum tolerance ${maximumTolerance}`
      );
    }
    return {
      deliveries,
      restarted: findRestartedLeaves(deliveries),
      hand: {carry: null, pointer: null, shift: Number.NaN, isReleased: false},
      guard: createStaleGuard()
    };
  },
  build(builder, inputs, values) {
    const {deliveries} = inputs;
    const {left, right} = deliveries;
    const leftStored = writeDeliveryPositions(deliveries, left);
    const leftPositions = builder.upload('left-positions', 'float32x2', leftStored);
    const rightPositions = builder.allocate('right-positions', 'float32x2', right.rowCount);
    // One pair per feature: its collection node in each delivery (the node ids differ).
    const pairs = Uint32Array.from({length: FEATURE_COUNT * 2}, (_, component) =>
      component & 1 ? right.roots[component >> 1] : left.roots[component >> 1]
    );
    const equal = builder.allocate('equal', 'uint32', FEATURE_COUNT);
    builder.graph.add(
      new GPUCanonicalGeometryEquality({
        left: addNormalizedTree(builder, 'left', left, leftPositions),
        right: addNormalizedTree(builder, 'right', right, rightPositions),
        pairs: builder.upload('pairs', 'uint32x2', pairs).view,
        tolerance: values.tolerance,
        output: equal.view
      })
    );
    // Every stored row of delivery B takes the flag of its feature's pair.
    const featureOfRow = builder.upload('right-feature-of-row', 'uint32', right.featureOfRow);
    const rowEqual = builder.allocate('right-row-equal', 'uint32', right.rowCount);
    addKernel(builder, {
      id: 'right-row-equal',
      invocationCount: right.rowCount,
      bindings: [
        {name: 'equal', buffer: equal, access: 'read'},
        {name: 'featureOfRow', buffer: featureOfRow, access: 'read'},
        {name: 'rowEqual', buffer: rowEqual, access: 'read_write'}
      ],
      body: 'rowEqual[rowEqualOffset + index] = equal[equalOffset + featureOfRow[featureOfRowOffset + index]];'
    });
    const leftMarks = getStartMarks(
      inputs.restarted.map(rows => [rows[0], rows[1]] as const),
      leftStored
    );
    const gauges = left.leaves.filter(leaf => leaf.part === 'gauge');
    const gaugePositions = Float32Array.from(
      gauges.flatMap(leaf => [leftStored[leaf.firstRow * 2], leftStored[leaf.firstRow * 2 + 1]])
    );
    // What delivery B left out stays where delivery A has it: a ring around every such point.
    const missing = gauges.filter(
      leaf => !right.leaves.some(twin => twin.feature === leaf.feature && twin.part === leaf.part)
    );
    const missingRings = getRingSegments(
      missing.flatMap(leaf => [leftStored[leaf.firstRow * 2], leftStored[leaf.firstRow * 2 + 1]]),
      CALLOUT_RADIUS
    );
    const rightStarts = builder.allocate('right-starts', 'float32x2', inputs.restarted.length);
    const rightChevrons = builder.allocate(
      'right-chevrons',
      'float32x4',
      inputs.restarted.length * CHEVRON_SEGMENT_COUNT
    );
    const callouts = builder.allocate(
      'callouts',
      'float32x4',
      missingRings.length / 4 + RING_SEGMENT_COUNT
    );
    return {
      outputs: {
        equal,
        rowEqual,
        leftPositions,
        leftEdges: builder.upload('left-edges', 'uint32x2', left.edges),
        leftGauges: builder.upload('left-gauges', 'float32x2', gaugePositions),
        leftStartRings: builder.upload(
          'left-start-rings',
          'float32x4',
          getRingSegments(leftMarks.starts, START_RADIUS)
        ),
        leftChevrons: builder.upload('left-chevrons', 'float32x4', leftMarks.chevrons),
        rightPositions,
        rightEdges: builder.upload('right-edges', 'uint32x2', right.edges),
        rightStarts,
        rightChevrons,
        callouts
      },
      update: current => {
        const overrides = getOverrides(inputs, current.shift);
        const stored = writeDeliveryPositions(deliveries, right, overrides);
        rightPositions.write(stored);
        const marks = getStartMarks(
          inputs.restarted.map(rows => [rows[2], rows[3]] as const),
          stored
        );
        rightStarts.write(marks.starts);
        rightChevrons.write(marks.chevrons);
        // The vertex only delivery B has follows the edge it lies on.
        const added = getDeliveryVertex(deliveries, deliveries.addedVertex.vertex, overrides);
        const rings = new Float32Array(callouts.length * 4);
        rings.set(missingRings);
        rings.set(getRingSegments(added, CALLOUT_RADIUS), missingRings.length);
        callouts.write(rings);
      }
    };
  },
  rebuildOn: ['tolerance'],
  // The pointer carries the nearest vertex of delivery B. At rest it holds the moved vertex where
  // delivery B stores it, so the poster is the two deliveries as they arrived.
  interaction: {
    kind: 'drag',
    hint: 'move a vertex',
    rest: ({inputs, values, toUnit}) => {
      const {homes, movedVertex} = inputs.deliveries;
      return toUnit([homes[movedVertex * 2] + values.shift, homes[movedVertex * 2 + 1]]);
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {deliveries, hand} = inputs;
      const hasMoved =
        !hand.pointer || hand.pointer[0] !== pointer.x || hand.pointer[1] !== pointer.y;
      if (hasMoved) hand.isReleased = false;
      else if (hand.shift !== values.shift) hand.isReleased = true;
      hand.pointer = [pointer.x, pointer.y];
      hand.shift = values.shift;
      // Parked under a slider, the hand lets go: the marks then follow the moved vertex to where
      // delivery B stores it.
      hand.carry = hand.isReleased
        ? null
        : {
            vertex: findCarriedVertex(inputs, pointer.x, pointer.y, values.shift),
            at: [pointer.x, pointer.y]
          };
      const {vertex, at} = hand.carry ?? getOverrides(inputs, values.shift)[0];
      const equal = readFresh(inputs.guard, String(values.tolerance), peek(outputs.equal));
      const feature = deliveries.featureOfVertex[vertex];
      const home: [number, number] = [
        deliveries.homes[vertex * 2],
        deliveries.homes[vertex * 2 + 1]
      ];
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [home[0] - values.tolerance, home[1] - values.tolerance],
          maximum: [home[0] + values.tolerance, home[1] + values.tolerance]
        },
        {shape: 'line', from: home, to: at, dashed: true},
        {shape: 'dot', at}
      ];
      return {
        rerun: true,
        glyphs,
        readout: `${equal ? VERDICTS[Math.min(equal[feature], 2)] : '...'} · ${describeFeature(inputs, feature, values)}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => {
    // Flag 0 is not equal, 1 is equal, 2 is invalid input.
    const categories = [theme.categories[1], theme.categories[0], theme.muted];
    const {keyAnchor, labelAnchors, left, partAnchors, right} = inputs.deliveries;
    const labels: RunLabel[] = DELIVERY_VARIANTS.map((variant, feature) => ({
      text: variant.label,
      at: [labelAnchors[feature * 2], labelAnchors[feature * 2 + 1]],
      role: 'label',
      priority: 2
    }));
    // Where a feature's parts are stored in another order: each part's place in A, then in B.
    left.partOrder.forEach((order, feature) => {
      const stored = right.partOrder[feature];
      if (order.length !== stored.length || order.every((part, place) => part === stored[place])) {
        return;
      }
      order.forEach((part, place) =>
        labels.push({
          text: `${place + 1}→${stored.indexOf(part) + 1}`,
          at: partAnchors[feature][part],
          anchor: part === 'gauge' ? 'right' : 'center',
          role: 'value'
        })
      );
    });
    let keyOffset = 0;
    for (const entry of [
      {text: 'A', swatch: theme.muted},
      {text: 'B, equal', swatch: 0},
      {text: 'B, not equal', swatch: 1}
    ]) {
      labels.push({
        ...entry,
        at: keyAnchor,
        anchor: 'top-left',
        offset: [keyOffset, -4],
        role: 'label',
        priority: 4
      });
      keyOffset += getSwatchWidth('label') + getTextWidth(entry.text, 'label') + 12;
    }
    // Layer radii are canvas pixels and the canvas has `devicePixelRatio` of them per CSS pixel:
    // without this the muted band of delivery A all but disappears on a dense display.
    const density = globalThis.devicePixelRatio || 1;
    return [
      {
        layer: 'segments',
        context: true,
        nodes: outputs.leftPositions,
        edges: outputs.leftEdges,
        radius: 4.5 * density
      },
      {layer: 'dots', context: true, positions: outputs.leftGauges, radius: 6.5 * density},
      {
        layer: 'segments',
        nodes: outputs.rightPositions,
        edges: outputs.rightEdges,
        values: outputs.rowEqual,
        categories,
        radius: 1.8 * density
      },
      {
        layer: 'dots',
        positions: outputs.rightPositions,
        values: outputs.rowEqual,
        categories,
        radius: 3.1 * density
      },
      {layer: 'lines', segments: outputs.callouts, radius: 1.2 * density},
      // Where a line or ring starts and which way it runs: a ring in A, a dot in B.
      {layer: 'lines', segments: outputs.leftStartRings, radius: 1.2 * density},
      {layer: 'lines', segments: outputs.leftChevrons, radius: 1.2 * density},
      {layer: 'lines', segments: outputs.rightChevrons, radius: 1.2 * density},
      {
        layer: 'dots',
        positions: outputs.rightStarts,
        radius: 3.4 * density,
        color: [...theme.ink, 1]
      },
      {layer: 'text', labels}
    ];
  }
});
