// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getWGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUNetworkFlowProps} from './gpu-network-flow';

/** Limits global relabel phases (each contains bounded BFS, push and excess-return rounds).
 * The shared command budget also bounds the product with maximumRelaxationRounds. */
export const MAXIMUM_NETWORK_FLOW_RELABELS = 4096;
/** Three blocking preflow phases, followed by a final cut search, fit the default node budget. */
export const DEFAULT_NETWORK_FLOW_RELABELS = 3;
/** Cost phases bound recomputed dual labels and feasible preflow batches; the command budget
 * further bounds their product with maximumRelaxationRounds. */
export const MAXIMUM_NETWORK_FLOW_COST_PHASES = 4096;
/** Three dual-label phases fit 1024 relaxation/forward/return rounds under the command budget. */
export const DEFAULT_NETWORK_FLOW_COST_PHASES = 3;

/** Global-relabel preflow push variant. Within a phase all admissible arcs strictly decrease
 * the immutable BFS height. Each vertex owns its discharge; atomic excess additions accept concurrent arrivals. Before committing, reverse
 * pushes return stranded excess along this phase's DAG, so every published flow is feasible.
 * A height-D phase needs at most D push rounds and D return rounds; each round is a separate
 * dispatch, never a workgroup barrier pretending to synchronize the device. Relabelling occurs
 * globally between phases instead of locally changing heights underneath the return DAG.
 * This sends all 99,998 paths of the 100k-vertex fan in one phase, not one path per search.
 * With costs, labels are lexicographic (signed cost-to-sink, hop count). Tightness is checked
 * on push and undo, so each batch is a sum of minimum-cost paths. Those labels are feasible dual
 * potentials on sink-reachable vertices: residual reduced costs are nonnegative. Committing or
 * undoing tight arcs introduces only zero reduced-cost reverse arcs, preserving optimality for
 * the committed value. The hop component removes zero-cost cycles from the admissible graph.
 * @internal */
export function getNetworkFlowPushRelabelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUNetworkFlowProps,
  views: {
    state: GraphDataView<'uint32'>;
    offsets: GraphDataView<'uint32'>;
    arcs: GraphDataView<'uint32'>;
    records: GraphDataView<'uint32'>;
    maximumRounds: number;
  }
): GPUCommandNode<Parameters>[] {
  const {state, offsets, arcs, records, maximumRounds} = views;
  const nodeCount = props.nodeCount;
  const edgeCount = props.edges.length;
  const costed = Boolean(props.costs);
  const maximumRelabels = costed
    ? (props.maximumCostPhases ?? DEFAULT_NETWORK_FLOW_COST_PHASES)
    : (props.maximumRelabels ?? DEFAULT_NETWORK_FLOW_RELABELS);
  // Costed labels pack signed distance bits then unsigned hop count. Excess temporarily holds
  // normalized labels before it is reset to vertex supplies, so it has the same binding size.
  const labelWords = costed ? 2 : 1;
  const word = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const heights = [
    word('height-first', nodeCount * labelWords),
    word('height-second', nodeCount * labelWords)
  ];
  const excess = word('excess', nodeCount * labelWords);
  const increments = word('phase-increments', Math.max(edgeCount, 1));
  const previousFlow = word('previous-flow', Math.max(edgeCount, 1));
  const read = (name: string, view: GraphDataView) => getWGSLKernelBinding(name, view);
  const write = (name: string, view: GraphDataView) =>
    getWGSLKernelBinding(name, view, 'read_write');
  const atomic = (name: string, view: GraphDataView) => getWGSLKernelBinding(name, view, 'atomic');
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-push-relabel-dispatch`,
    operation: 'GPUNetworkFlow',
    shapes: [[nodeCount], [edgeCount]]
  });
  const kernel = (
    name: string,
    count: number,
    bindings: ReturnType<typeof getWGSLKernelBinding>[],
    body: string,
    slot?: number
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation: 'GPUNetworkFlow',
      variant: costed ? 'primal-dual' : 'push-relabel',
      invocationCount: count,
      bindings,
      declarations: `const NONE: u32 = 0xffffffffu;`,
      body,
      ...(slot === undefined ? {} : gate.getGate(slot))
    });
  const nodes: GPUCommandNode<Parameters>[] = [];
  // State words 5/6/7/17 are reused as changed/searchEnabled/parity/searchConverged. Word 18
  // enables a phase, 19 is the number of global relabels and 20 flags stranded interior excess.
  for (let phase = 0; phase <= maximumRelabels; phase++) {
    const prefix = `relabel-${phase}`;
    nodes.push(
      kernel(
        `${prefix}-start`,
        1,
        [write('state', state), gate.binding],
        `
state[stateOffset + 18u] = 0u;
state[stateOffset + 6u] = state[stateOffset];
if (state[stateOffset] != 0u) {
  state[stateOffset + 5u] = 0u; state[stateOffset + 7u] = 0u; state[stateOffset + 17u] = 0u;
  ${gate.armWGSL}
} else { ${gate.disarmWGSL} }`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-initialize-height`,
        nodeCount,
        [read('state', state), write('height', heights[0])],
        `
${
  costed
    ? `height[heightOffset + 2u * index] = bitcast<u32>(select(0x7fffffff, 0, index == state[stateOffset + 15u]));
height[heightOffset + 2u * index + 1u] = select(NONE, 0u, index == state[stateOffset + 15u]);`
    : 'height[heightOffset + index] = select(NONE, 0u, index == state[stateOffset + 15u]);'
}`,
        0
      )
    );
    for (let round = 0; round < maximumRounds; round++) {
      const input = heights[round % 2];
      const output = heights[1 - (round % 2)];
      nodes.push(
        kernel(
          `${prefix}-bfs-${round}`,
          nodeCount,
          [
            read('offsets', offsets),
            read('arcs', arcs),
            read('records', records),
            read('flow', props.output.flow),
            read('inputHeight', input),
            write('outputHeight', output),
            atomic('state', state)
          ],
          `
${
  costed
    ? `var distance = bitcast<i32>(inputHeight[inputHeightOffset + 2u * index]);
var hops = inputHeight[inputHeightOffset + 2u * index + 1u];
let originalDistance = distance; let originalHops = hops;`
    : `var distance = inputHeight[inputHeightOffset + index];
let originalDistance = distance;`
}
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let arc = arcs[arcsOffset + slot]; let edge = arc / 2u; let reverse = (arc & 1u) != 0u;
  let amount = flow[flowOffset + edge]; let capacity = records[recordsOffset + 4u * edge + 2u];
  if ((!reverse && amount == capacity) || (reverse && amount == 0u)) { continue; }
  let neighbor = records[recordsOffset + 4u * edge + select(1u, 0u, reverse)];
  ${
    costed
      ? `let neighborDistance = bitcast<i32>(inputHeight[inputHeightOffset + 2u * neighbor]);
  if (neighborDistance == 0x7fffffff) { continue; }
  let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let candidate = neighborDistance + select(unitCost, -unitCost, reverse);
  let candidateHops = inputHeight[inputHeightOffset + 2u * neighbor + 1u] + 1u;
  if (candidate < distance || (candidate == distance && candidateHops < hops)) {
    distance = candidate; hops = candidateHops;
  }`
      : `let neighborDistance = inputHeight[inputHeightOffset + neighbor];
  if (neighborDistance != NONE) { distance = min(distance, neighborDistance + 1u); }`
  }
}
${
  costed
    ? `outputHeight[outputHeightOffset + 2u * index] = bitcast<u32>(distance);
outputHeight[outputHeightOffset + 2u * index + 1u] = hops;
if (distance != originalDistance || hops != originalHops) { atomicOr(&state[stateOffset + 5u], 1u); }`
    : `outputHeight[outputHeightOffset + index] = distance;
if (distance != originalDistance) { atomicOr(&state[stateOffset + 5u], 1u); }`
}`,
          0
        )
      );
      nodes.push(
        kernel(
          `${prefix}-bfs-control-${round}`,
          1,
          [write('state', state), gate.binding],
          `
if (state[stateOffset + 6u] == 0u) { return; }
state[stateOffset + 7u] = 1u - state[stateOffset + 7u];
if (state[stateOffset + 5u] == 0u) {
  state[stateOffset + 6u] = 0u; state[stateOffset + 17u] = 1u; ${gate.disarmWGSL}
}
state[stateOffset + 5u] = 0u;`
        )
      );
    }
    // Normalize into excess temporarily. This avoids reading and writing the same height array
    // in one dispatch, and respects the runtime parity of the last BFS that actually ran.
    nodes.push(
      kernel(
        `${prefix}-normalize`,
        nodeCount,
        [
          read('state', state),
          read('first', heights[0]),
          read('second', heights[1]),
          write('excess', excess)
        ],
        `
if (state[stateOffset] == 0u) { return; }
${
  costed
    ? `excess[excessOffset + 2u * index] = select(first[firstOffset + 2u * index], second[secondOffset + 2u * index], state[stateOffset + 7u] != 0u);
excess[excessOffset + 2u * index + 1u] = select(first[firstOffset + 2u * index + 1u], second[secondOffset + 2u * index + 1u], state[stateOffset + 7u] != 0u);`
    : 'excess[excessOffset + index] = select(first[firstOffset + index], second[secondOffset + index], state[stateOffset + 7u] != 0u);'
}`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-height-copy`,
        nodeCount,
        [read('state', state), read('excess', excess), write('height', heights[0])],
        `
if (state[stateOffset] != 0u) {
${
  costed
    ? `height[heightOffset + 2u * index] = excess[excessOffset + 2u * index];
height[heightOffset + 2u * index + 1u] = excess[excessOffset + 2u * index + 1u];`
    : 'height[heightOffset + index] = excess[excessOffset + index];'
}
}`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-finish-search`,
        1,
        [write('state', state), read('height', heights[0]), gate.binding],
        `
${gate.disarmWGSL}
if (state[stateOffset] == 0u) { return; }
if (state[stateOffset + 3u] == state[stateOffset + 16u]) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 17u] == 0u) { state[stateOffset] = 0u; return; }
if (height[heightOffset + ${costed ? '2u * state[stateOffset + 14u] + 1u' : 'state[stateOffset + 14u]'}] == NONE) {
  state[stateOffset + 2u] = 1u; state[stateOffset] = 0u; return;
}
if (state[stateOffset + 19u] == ${maximumRelabels}u) { state[stateOffset] = 0u; return; }
state[stateOffset + 18u] = 1u; state[stateOffset + 20u] = 0u;
state[stateOffset + 19u] += 1u;
${gate.armWGSL}`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-initialize-excess`,
        nodeCount,
        [read('state', state), write('excess', excess)],
        `
excess[excessOffset + index] = select(0u, state[stateOffset + 16u] - state[stateOffset + 3u], index == state[stateOffset + 14u]);`,
        0
      )
    );
    nodes.push(
      kernel(
        `${prefix}-initialize-increments`,
        edgeCount,
        [
          read('flow', props.output.flow),
          write('previousFlow', previousFlow),
          write('increments', increments)
        ],
        `
previousFlow[previousFlowOffset + index] = flow[flowOffset + index];
increments[incrementsOffset + index] = 0u;`,
        1
      )
    );
    // Admissibility is immutable within a phase: an original edge can only be pushed in one
    // direction. Each vertex owns its outgoing pushes and each admissible edge has exactly one writer.
    for (let round = 0; round < maximumRounds; round++) {
      nodes.push(
        kernel(
          `${prefix}-push-${round}`,
          nodeCount,
          [
            read('offsets', offsets),
            read('arcs', arcs),
            read('records', records),
            read('height', heights[0]),
            atomic('flow', props.output.flow),
            atomic('excess', excess),
            write('increments', increments),
            atomic('state', state)
          ],
          `
if (index == atomicLoad(&state[stateOffset + 15u])) { return; }
var available = atomicLoad(&excess[excessOffset + index]); let originalAvailable = available;
let vertexHeight = height[heightOffset + ${costed ? '2u * index + 1u' : 'index'}]; if (vertexHeight == NONE || available == 0u) { return; }
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let arc = arcs[arcsOffset + slot]; let edge = arc / 2u; let reverse = (arc & 1u) != 0u;
  let neighbor = records[recordsOffset + 4u * edge + select(1u, 0u, reverse)];
  let neighborHeight = height[heightOffset + ${costed ? '2u * neighbor + 1u' : 'neighbor'}];
  if (neighborHeight == NONE || vertexHeight != neighborHeight + 1u) { continue; }
  ${
    costed
      ? `let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let vertexDistance = bitcast<i32>(height[heightOffset + 2u * index]);
  let neighborDistance = bitcast<i32>(height[heightOffset + 2u * neighbor]);
  if (vertexDistance != neighborDistance + select(unitCost, -unitCost, reverse)) { continue; }`
      : ''
  }
  let amount = atomicLoad(&flow[flowOffset + edge]);
  let residual = select(records[recordsOffset + 4u * edge + 2u] - amount, amount, reverse);
  let increment = min(available, residual); if (increment == 0u) { continue; }
  if (reverse) { atomicSub(&flow[flowOffset + edge], increment); }
  else { atomicAdd(&flow[flowOffset + edge], increment); }
  increments[incrementsOffset + edge] += increment;
  atomicAdd(&excess[excessOffset + neighbor], increment); available -= increment;
  if (available == 0u) { break; }
}
let sent = originalAvailable - available;
if (sent != 0u) { atomicSub(&excess[excessOffset + index], sent); atomicOr(&state[stateOffset + 5u], 1u); }`,
          0
        )
      );
      nodes.push(
        kernel(
          `${prefix}-push-control-${round}`,
          1,
          [write('state', state), gate.binding],
          `
if (state[stateOffset + 18u] == 0u) { return; }
if (state[stateOffset + 5u] == 0u) { ${gate.disarmWGSL} }
state[stateOffset + 5u] = 0u;`
        )
      );
    }
    nodes.push(
      kernel(
        `${prefix}-return-start`,
        1,
        [read('state', state), gate.binding],
        `
if (state[stateOffset + 18u] != 0u) { ${gate.armWGSL} } else { ${gate.disarmWGSL} }`
      )
    );
    for (let round = 0; round < maximumRounds; round++) {
      nodes.push(
        kernel(
          `${prefix}-return-${round}`,
          nodeCount,
          [
            read('offsets', offsets),
            read('arcs', arcs),
            read('records', records),
            read('height', heights[0]),
            atomic('flow', props.output.flow),
            atomic('excess', excess),
            write('increments', increments),
            atomic('state', state)
          ],
          `
if (index == atomicLoad(&state[stateOffset + 15u]) || index == atomicLoad(&state[stateOffset + 14u])) { return; }
var available = atomicLoad(&excess[excessOffset + index]); let originalAvailable = available;
let vertexHeight = height[heightOffset + ${costed ? '2u * index + 1u' : 'index'}]; if (available == 0u || vertexHeight == NONE) { return; }
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let outgoing = arcs[arcsOffset + slot]; let edge = outgoing / 2u; let reverse = (outgoing & 1u) != 0u;
  let neighbor = records[recordsOffset + 4u * edge + select(1u, 0u, reverse)];
  let neighborHeight = height[heightOffset + ${costed ? '2u * neighbor + 1u' : 'neighbor'}];
  if (neighborHeight == NONE || neighborHeight != vertexHeight + 1u) { continue; }
  ${
    costed
      ? `let unitCost = i32(records[recordsOffset + 4u * edge + 3u]);
  let vertexDistance = bitcast<i32>(height[heightOffset + 2u * index]);
  let neighborDistance = bitcast<i32>(height[heightOffset + 2u * neighbor]);
  if (neighborDistance != vertexDistance + select(-unitCost, unitCost, reverse)) { continue; }`
      : ''
  }
  let decrement = min(available, increments[incrementsOffset + edge]); if (decrement == 0u) { continue; }
  // Undo the incoming push: outgoing forward means the original phase push was reversed.
  if (reverse) { atomicSub(&flow[flowOffset + edge], decrement); }
  else { atomicAdd(&flow[flowOffset + edge], decrement); }
  increments[incrementsOffset + edge] -= decrement;
  atomicAdd(&excess[excessOffset + neighbor], decrement); available -= decrement;
  if (available == 0u) { break; }
}
let returned = originalAvailable - available;
if (returned != 0u) { atomicSub(&excess[excessOffset + index], returned); atomicOr(&state[stateOffset + 5u], 1u); }`,
          0
        )
      );
      nodes.push(
        kernel(
          `${prefix}-return-control-${round}`,
          1,
          [write('state', state), gate.binding],
          `
if (state[stateOffset + 18u] == 0u) { return; }
if (state[stateOffset + 5u] == 0u) { ${gate.disarmWGSL} }
state[stateOffset + 5u] = 0u;`
        )
      );
    }
    nodes.push(
      kernel(
        `${prefix}-verify`,
        nodeCount,
        [read('excess', excess), atomic('state', state)],
        `
if (atomicLoad(&state[stateOffset + 18u]) != 0u && index != atomicLoad(&state[stateOffset + 14u]) && index != atomicLoad(&state[stateOffset + 15u]) && excess[excessOffset + index] != 0u) {
  atomicOr(&state[stateOffset + 20u], 1u);
}`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-restore-if-needed`,
        edgeCount,
        [
          read('state', state),
          read('previousFlow', previousFlow),
          write('flow', props.output.flow)
        ],
        `
if (state[stateOffset + 18u] != 0u && state[stateOffset + 20u] != 0u) { flow[flowOffset + index] = previousFlow[previousFlowOffset + index]; }`
      )
    );
    nodes.push(
      kernel(
        `${prefix}-commit`,
        1,
        [write('state', state), read('excess', excess)],
        `
if (state[stateOffset + 18u] == 0u) { return; }
if (state[stateOffset + 20u] != 0u) { state[stateOffset] = 0u; state[stateOffset + 17u] = 0u; return; }
let increment = excess[excessOffset + state[stateOffset + 15u]];
state[stateOffset + 3u] += increment; state[stateOffset + 4u] += 1u;
if (increment == 0u) { state[stateOffset] = 0u; state[stateOffset + 17u] = 0u; }`
      )
    );
  }
  // The final source-side cut is a separate forward BFS: reverse BFS to the sink does not
  // identify the minimum source-side cut. It uses the same bounded globally synchronized rounds.
  nodes.push(
    kernel(
      'cut-start',
      1,
      [write('state', state), gate.binding],
      `
state[stateOffset + 6u] = state[stateOffset + 1u]; state[stateOffset + 7u] = 0u;
state[stateOffset + 17u] = 0u; state[stateOffset + 5u] = 0u;
if (state[stateOffset + 1u] != 0u) { ${gate.armWGSL} } else { ${gate.disarmWGSL} }`
    )
  );
  nodes.push(
    kernel(
      'cut-initialize',
      nodeCount,
      [read('state', state), write('height', heights[0])],
      `
height[heightOffset + index] = select(NONE, 0u, index == state[stateOffset + 14u]);`,
      0
    )
  );
  for (let round = 0; round < maximumRounds; round++) {
    nodes.push(
      kernel(
        `cut-bfs-${round}`,
        nodeCount,
        [
          read('offsets', offsets),
          read('arcs', arcs),
          read('records', records),
          read('flow', props.output.flow),
          read('inputHeight', heights[round % 2]),
          write('outputHeight', heights[1 - (round % 2)]),
          atomic('state', state)
        ],
        `
var distance = inputHeight[inputHeightOffset + index]; let originalDistance = distance;
for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
  let incoming = arcs[arcsOffset + slot] ^ 1u; let edge = incoming / 2u; let reverse = (incoming & 1u) != 0u;
  let amount = flow[flowOffset + edge]; let capacity = records[recordsOffset + 4u * edge + 2u];
  if ((!reverse && amount == capacity) || (reverse && amount == 0u)) { continue; }
  let neighbor = records[recordsOffset + 4u * edge + select(0u, 1u, reverse)];
  let neighborDistance = inputHeight[inputHeightOffset + neighbor];
  if (neighborDistance != NONE) { distance = min(distance, neighborDistance + 1u); }
}
outputHeight[outputHeightOffset + index] = distance;
if (distance != originalDistance) { atomicOr(&state[stateOffset + 5u], 1u); }`,
        0
      )
    );
    nodes.push(
      kernel(
        `cut-control-${round}`,
        1,
        [write('state', state), gate.binding],
        `
if (state[stateOffset + 6u] == 0u) { return; }
state[stateOffset + 7u] = 1u - state[stateOffset + 7u];
if (state[stateOffset + 5u] == 0u) { state[stateOffset + 6u] = 0u; state[stateOffset + 17u] = 1u; ${gate.disarmWGSL} }
state[stateOffset + 5u] = 0u;`
      )
    );
  }
  nodes.push(
    kernel(
      'cut-publish',
      nodeCount,
      [
        read('state', state),
        read('first', heights[0]),
        read('second', heights[1]),
        write('cut', props.output.cut)
      ],
      `
if (state[stateOffset + 1u] == 0u) { return; }
let distance = select(first[firstOffset + index], second[secondOffset + index], state[stateOffset + 7u] != 0u);
cut[cutOffset + index] = select(0u, 1u, distance != NONE);`
    )
  );
  nodes.push(
    kernel(
      'cut-finish',
      1,
      [write('state', state), read('cut', props.output.cut)],
      `
if (state[stateOffset + 1u] != 0u && state[stateOffset + 17u] != 0u && cut[cutOffset + state[stateOffset + 15u]] == 0u) { state[stateOffset + 2u] = 1u; }`
    )
  );
  return nodes;
}
