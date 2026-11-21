// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getRegressionTileLayout, getRowTileSumNodes} from './regression-row-tiles';
import {validateRegressionAllocations} from './spatial-likelihood-allocations';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

/** CSR row IDs are uint32; a binary search halves its interval on every iteration. */
export const MAXIMUM_DIAGNOSTICS_CSR_SEARCH_ITERATIONS = 32;
/** Selected-row slot work per invocation, independent of neighborhood length. */
export const MAXIMUM_DIAGNOSTICS_REFERENCE_TILE_SLOTS = 64;

/** Preserve a selected-row reference without a single invocation scanning its neighborhood.
 * origin: [runtime zero, selected row then centered lag, symmetric link, fitted origin, row sum].
 * @internal
 */
export function getDiagnosticsObservedOriginNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    slotCount: number;
    constants: string;
    rowScaleWGSL: string;
    weightBindings: WGSLKernelBinding[];
    table: WGSLKernelBinding;
    rowScaleBindings: WGSLKernelBinding[];
    origin: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, slotCount} = props;
  const layout = getRegressionTileLayout(slotCount, MAXIMUM_DIAGNOSTICS_REFERENCE_TILE_SLOTS);
  validateRegressionAllocations(graph.device, id, {
    'selected reference partials': 3 * layout.tileCount
  });
  const reduced = createTransientView(graph, `${id}-selected-reference`, 'float32', 3);
  const selectedWeights = createTransientView(
    graph,
    `${id}-selected-reference-weights`,
    'float32',
    2
  );
  const origin: WGSLKernelBinding = {
    name: 'observedOrigin',
    view: props.origin,
    type: 'f32',
    access: 'read'
  };
  const declarations = `${props.constants}\n${props.rowScaleWGSL}\n${ROUNDED_DIFFERENCE_WGSL}`;
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-observed-origin-start`,
      operation: 'GPUSpatialRegressionDiagnostics',
      invocationCount: 1,
      bindings: [props.table, {...origin, access: 'read_write'}],
      declarations: `${props.constants}\n${ROUNDED_DIFFERENCE_WGSL}`,
      body: `observedOrigin[observedOriginOffset + 2u] = 0.0;
observedOrigin[observedOriginOffset + 3u] = 0.0;
observedOrigin[observedOriginOffset + 4u] = 0.0;
let selected = observedOrigin[observedOriginOffset + 1u];
if (selected > 0.0) {
  let base = tableOffset + (ROWS - u32(selected)) * TABLE_WIDTH;
  observedOrigin[observedOriginOffset + 3u] = getRoundedDifference(table[base + P], table[base + P + 1u], bitcast<u32>(observedOrigin[observedOriginOffset]));
}`
    }),
    ...getRowTileSumNodes(graph, {
      id: `${id}-observed-origin-slots`,
      operation: 'GPUSpatialRegressionDiagnostics',
      variant: 'observed-origin-slots',
      rowCount: slotCount,
      layout,
      sumCount: 2,
      maximumCount: 1,
      bindings: [...props.weightBindings, props.table, ...props.rowScaleBindings, origin],
      partial: createTransientView(
        graph,
        `${id}-selected-reference-partials`,
        'float32',
        3 * layout.tileCount
      ),
      output: reduced,
      declarations: `${declarations}
fn accumulateRow(slot: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let selected = observedOrigin[observedOriginOffset + 1u];
  if (!(selected > 0.0)) { return; }
  let row = ROWS - u32(selected);
  if (slot < offsets[offsetsOffset + row] || slot >= offsets[offsetsOffset + row + 1u]) { return; }
  let neighbor = neighbors[neighborsOffset + slot];
  if (neighbor >= ROWS || table[tableOffset + neighbor * TABLE_WIDTH] < 0.5) { return; }
  let runtimeZero = bitcast<u32>(observedOrigin[observedOriginOffset]);
  let base = tableOffset + neighbor * TABLE_WIDTH;
  let fitted = getRoundedDifference(table[base + P], table[base + P + 1u], runtimeZero);
  let weight = weights[weightsOffset + slot] * getRowScale(row);
  (*terms)[0] += weight * getRoundedDifference(fitted, observedOrigin[observedOriginOffset + 3u], runtimeZero);
  (*terms)[1] += weight;
  // Sorted unique row IDs preserve the first valid off-diagonal reference exactly.
  if (neighbor != row) { (*terms)[2] = max((*terms)[2], f32(ROWS - neighbor)); }
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-observed-origin-weights`,
      operation: 'GPUSpatialRegressionDiagnostics',
      invocationCount: 2,
      bindings: [
        ...props.weightBindings,
        ...props.rowScaleBindings,
        {name: 'reference', view: reduced, type: 'f32', access: 'read'},
        origin,
        {name: 'selectedWeights', view: selectedWeights, type: 'f32', access: 'read_write'}
      ],
      declarations: `${declarations}
const MAXIMUM_SEARCH_ITERATIONS: u32 = ${MAXIMUM_DIAGNOSTICS_CSR_SEARCH_ITERATIONS}u;
fn referenceWeight(row: u32, targetRow: u32) -> f32 {
  var low = offsets[offsetsOffset + row]; var high = offsets[offsetsOffset + row + 1u];
  for (var iteration = 0u; iteration < MAXIMUM_SEARCH_ITERATIONS; iteration++) {
    if (low >= high) { break; }
    let middle = low + (high - low) / 2u; let neighbor = neighbors[neighborsOffset + middle];
    if (neighbor == targetRow) { return weights[weightsOffset + middle] * getRowScale(row); }
    if (neighbor < targetRow) { low = middle + 1u; } else { high = middle; }
  }
  return 0.0;
}`,
      body: `selectedWeights[selectedWeightsOffset + index] = 0.0;
let selected = observedOrigin[observedOriginOffset + 1u];
let selectedNeighbor = reference[referenceOffset + 2u];
if (selected > 0.0 && selectedNeighbor > 0.0) {
  let row = ROWS - u32(selected); let neighbor = ROWS - u32(selectedNeighbor);
  selectedWeights[selectedWeightsOffset + index] = referenceWeight(select(row, neighbor, index == 1u), select(neighbor, row, index == 1u));
}`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-observed-origin`,
      operation: 'GPUSpatialRegressionDiagnostics',
      invocationCount: 1,
      bindings: [
        {name: 'reference', view: reduced, type: 'f32', access: 'read'},
        {name: 'selectedWeights', view: selectedWeights, type: 'f32', access: 'read'},
        {...origin, access: 'read_write'}
      ],
      body: `observedOrigin[observedOriginOffset + 1u] = reference[referenceOffset];
observedOrigin[observedOriginOffset + 4u] = reference[referenceOffset + 1u];
observedOrigin[observedOriginOffset + 2u] = 0.5 * selectedWeights[selectedWeightsOffset] + 0.5 * selectedWeights[selectedWeightsOffset + 1u];`
    })
  ];
}
