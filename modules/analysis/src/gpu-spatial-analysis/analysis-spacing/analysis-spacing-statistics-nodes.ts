// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  DECODE_ORDERED_KEY_WGSL,
  GET_ORDERED_KEY_WGSL,
  IS_FINITE_FLOAT_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {createSortedSegmentOffsetsNode} from '../../analysis-core/kernels/sorted-segment-offsets';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {GPUNeighborSearch} from '../neighbor-search/index';
import {NEIGHBOR_SEARCH_CANONICAL_LONGITUDE_WGSL} from '../neighbor-search/neighbor-search-wgsl';
import type {GPUAnalysisSpacing} from './gpu-analysis-spacing';

/** GPU composition only; all allocations are transient and all data-sized work is parallel. @internal */
export function createAnalysisSpacingStatisticsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUAnalysisSpacing
): readonly GPUCommandNode<Parameters>[] {
  const {id, props, neighborRank, neighborCapacity, gridSize} = contributor;
  const rowCount = props.positions.length;
  const geographic = contributor.spatialContext.coordinateSpace === 'longitude-latitude';
  const nodes: GPUCommandNode<Parameters>[] = [];
  const unsigned = (name: string, length = rowCount) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const floating = (name: string, length = rowCount) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);
  const positions = createTransientView(graph, `${id}-canonical`, 'float32x2', rowCount);
  const valid = unsigned('valid');
  const invalid = unsigned('invalid');
  const sourceX = floating('source-x');
  const sourceY = floating('source-y');
  const canonicalX = floating('canonical-x');
  const state = unsigned('state', 64);
  const bind = (name: string, view: GraphDataView, writable = false): WGSLKernelBinding => ({
    name,
    view,
    type: view.format === 'float32x2' || view.format === 'float32' ? 'f32' : 'u32',
    access: writable ? 'read_write' : 'read'
  });
  const kernel = (
    name: string,
    bindings: WGSLKernelBinding[],
    body: string,
    invocationCount = rowCount,
    declarations = ''
  ) => {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-${name}`,
        operation: 'GPUAnalysisSpacing',
        bindings,
        body,
        invocationCount,
        declarations
      })
    );
  };
  const readState = bind('state', state);
  const writeState = bind('state', state, true);
  const copyScalar = (name: string, input: GraphDataView, slot: number) => {
    kernel(
      `${name}-store`,
      [bind('value', input), writeState],
      `state[stateOffset + ${slot}u] = ${input.format === 'float32' ? 'bitcast<u32>(value[valueOffset])' : 'value[valueOffset]'};`,
      1
    );
  };
  const reduceFloat = (
    name: string,
    input: GraphDataView<'float32'>,
    operation: 'sum' | 'min' | 'max',
    slot: number,
    mask?: GraphDataView<'uint32'>
  ) => {
    const output = floating(`${name}-result`, 1);
    nodes.push(
      ...new GPUReduction({id: `${id}-${name}`, input, operation, output, mask}).getCommandNodes(
        graph
      )
    );
    copyScalar(name, output, slot);
  };
  const reduceUnsigned = (
    name: string,
    input: GraphDataView<'uint32'>,
    operation: 'sum' | 'min',
    slot: number
  ) => {
    const output = unsigned(`${name}-result`, 1);
    nodes.push(
      ...new GPUReduction({id: `${id}-${name}`, input, operation, output}).getCommandNodes(graph)
    );
    copyScalar(name, output, slot);
  };
  kernel('clear-state', [writeState], 'state[stateOffset + index] = 0u;', 64);
  kernel(
    'prepare',
    [
      bind('source', props.positions),
      ...(props.mask ? [bind('mask', props.mask)] : []),
      bind('positions', positions, true),
      bind('valid', valid, true),
      bind('invalid', invalid, true),
      bind('sourceX', sourceX, true),
      bind('sourceY', sourceY, true),
      bind('canonicalX', canonicalX, true)
    ],
    `
    let point = vec2f(source[sourceOffset + index * 2u], source[sourceOffset + index * 2u + 1u]);
    let included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
    let finite = isFiniteFloat(point.x) && isFiniteFloat(point.y)${geographic ? ' && abs(point.y) <= 90.0' : ''};
    let keep = included && finite;
    valid[validOffset + index] = select(0u, 1u, keep);
    invalid[invalidOffset + index] = select(0u, 1u, included && !finite);
    sourceX[sourceXOffset + index] = select(0.0, point.x, keep);
    sourceY[sourceYOffset + index] = select(0.0, point.y, keep);
    var canonical = select(vec2f(0.0), point, keep);
    ${
      geographic
        ? `
    // Share exact stored-f32 reduction with neighbor search, retaining tiny signed offsets
    // and large-angle remainders. +180 and -180 and every pole use one physical identity.
    canonical.x = getNeighborSearchCanonicalLongitude(canonical.x);
    if (canonical.x <= -180.0) { canonical.x += 360.0; }
    if (abs(canonical.y) == 90.0) { canonical.x = 0.0; }`
        : ''
    }
    if (canonical.x == 0.0) { canonical.x = 0.0; }
    if (canonical.y == 0.0) { canonical.y = 0.0; }
    canonicalX[canonicalXOffset + index] = canonical.x;
    positions[positionsOffset + index * 2u] = canonical.x;
    positions[positionsOffset + index * 2u + 1u] = canonical.y;`,
    rowCount,
    [IS_FINITE_FLOAT_WGSL, ...(geographic ? [NEIGHBOR_SEARCH_CANONICAL_LONGITUDE_WGSL] : [])].join(
      '\n'
    )
  );
  const prefix = unsigned('valid-prefix');
  nodes.push(
    ...new GPUScan({
      id: `${id}-valid-scan`,
      input: valid,
      output: prefix,
      mode: 'inclusive'
    }).getCommandNodes(graph)
  );
  kernel(
    'count',
    [bind('prefix', prefix), writeState],
    `
    state[stateOffset] = prefix[prefixOffset + ${rowCount - 1}u];
    state[stateOffset + 5u] = ${neighborRank}u;
    state[stateOffset + 7u] = 1u;
    state[stateOffset + 26u] = ${geographic ? 1 : 0}u;`,
    1
  );
  reduceUnsigned('invalid-count', invalid, 'sum', 1);
  reduceFloat('minimum-x', sourceX, 'min', 8, valid);
  reduceFloat('minimum-y', sourceY, 'min', 9, valid);
  reduceFloat('maximum-x', sourceX, 'max', 10, valid);
  reduceFloat('maximum-y', sourceY, 'max', 11, valid);
  if (geographic) {
    reduceFloat('minimum-canonical-longitude', canonicalX, 'min', 38, valid);
    reduceFloat('maximum-canonical-longitude', canonicalX, 'max', 39, valid);
  }

  const indices = unsigned('indices');
  const longitudeKeys = unsigned('longitude-keys');
  const locationX = unsigned('location-x');
  const locationY = unsigned('location-y');
  kernel(
    'keys',
    [
      bind('positions', positions),
      bind('valid', valid),
      bind('sourceX', sourceX),
      bind('indices', indices, true),
      bind('longitudeKeys', longitudeKeys, true),
      bind('locationX', locationX, true),
      bind('locationY', locationY, true)
    ],
    `
    let keep = valid[validOffset + index] != 0u;
    let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    indices[indicesOffset + index] = index;
    locationX[locationXOffset + index] = select(0xffffffffu, getOrderedKey(point.x), keep);
    locationY[locationYOffset + index] = select(0xffffffffu, getOrderedKey(point.y), keep);
    var longitude = getNeighborSearchCanonicalLongitude(sourceX[sourceXOffset + index]);
    if (longitude <= -180.0) { longitude += 360.0; }
    if (longitude == 0.0) { longitude = 0.0; }
    // Rotate the signed ordered key: positive longitudes precede negatives, exactly as on
    // [0, 360), while the key still retains every bit of a tiny negative endpoint.
    longitudeKeys[longitudeKeysOffset + index] = select(0xffffffffu, getOrderedKey(longitude) ^ 0x80000000u, keep);`,
    rowCount,
    [GET_ORDERED_KEY_WGSL, NEIGHBOR_SEARCH_CANONICAL_LONGITUDE_WGSL].join('\n')
  );
  const locationOrder = createStableLexicographicIndexSortNodes(graph, {
    id: `${id}-location-sort`,
    operation: 'GPUAnalysisSpacing',
    indices,
    indicesAreIdentity: true,
    keys: [
      {view: locationX},
      {view: locationY},
      ...(props.identities ? [{view: props.identities}] : [])
    ]
  });
  nodes.push(...locationOrder.nodes);
  const distinctMask = unsigned('distinct-mask');
  const distinctFlags = unsigned('distinct-flags');
  kernel(
    'distinct',
    [
      bind('order', locationOrder.sortedIndices),
      bind('positions', positions),
      bind('valid', valid),
      bind('distinctMask', distinctMask, true),
      bind('distinctFlags', distinctFlags, true)
    ],
    `
    let row = order[orderOffset + index];
    var first = valid[validOffset + row] != 0u;
    if (index > 0u) {
      let previous = order[orderOffset + index - 1u];
      first = first && any(vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]) != vec2f(positions[positionsOffset + previous * 2u], positions[positionsOffset + previous * 2u + 1u]));
    }
    distinctMask[distinctMaskOffset + row] = select(0u, 1u, first);
    distinctFlags[distinctFlagsOffset + index] = select(0u, 1u, first);`
  );
  reduceUnsigned('distinct-count', distinctFlags, 'sum', 25);
  const locationPrefix = unsigned('location-prefix');
  nodes.push(
    ...new GPUScan({
      id: `${id}-location-scan`,
      input: distinctFlags,
      output: locationPrefix,
      mode: 'inclusive'
    }).getCommandNodes(graph)
  );
  const locationGroups = unsigned('location-groups');
  kernel(
    'location-groups',
    [bind('prefix', locationPrefix), readState, bind('groups', locationGroups, true)],
    `
    groups[groupsOffset + index] = select(${rowCount}u, prefix[prefixOffset + index] - 1u, index < state[stateOffset]);`
  );
  const locationOffsets = unsigned('location-offsets', rowCount + 1);
  nodes.push(
    createSortedSegmentOffsetsNode(graph, {
      id: `${id}-location-offsets`,
      operation: 'GPUAnalysisSpacing',
      segmentCount: rowCount,
      sortedKeys: locationGroups,
      segmentOffsets: locationOffsets
    })
  );
  const multiplicities = unsigned('multiplicities');
  const queryMask = unsigned('full-query-mask');
  kernel(
    'multiplicities',
    [
      bind('groups', locationGroups),
      bind('offsets', locationOffsets),
      bind('order', locationOrder.sortedIndices),
      bind('multiplicities', multiplicities, true),
      bind('queryMask', queryMask, true)
    ],
    `
    let group = groups[groupsOffset + index]; let row = order[orderOffset + index]; var count = 0u;
    if (group < ${rowCount}u) { count = offsets[offsetsOffset + group + 1u] - offsets[offsetsOffset + group]; }
    multiplicities[multiplicitiesOffset + row] = count;
    // With rank + 1 identical rows, every requested distance is already known to be zero.
    // Keep these rows as targets for other queries, but never search their own dense cells.
    queryMask[queryMaskOffset + row] = select(0u, 1u, count > 0u && count <= ${neighborRank}u);`
  );
  kernel(
    'duplicate-count',
    [writeState],
    'state[stateOffset + 4u] = state[stateOffset] - state[stateOffset + 25u];',
    1
  );

  // Keep the planar centre relative to the extent origin in a separate pass. A centre at
  // 10000001.5 is not representable as one f32 even when every input coordinate is exact.
  const centerPositions = geographic
    ? positions
    : createTransientView(graph, `${id}-relative-positions`, 'float32x2', rowCount);
  if (!geographic) {
    kernel(
      'relative-positions',
      [bind('positions', positions), readState, bind('relative', centerPositions, true)],
      `
      relative[relativeOffset + index * 2u] = positions[positionsOffset + index * 2u] - bitcast<f32>(state[stateOffset + 8u]);
      relative[relativeOffset + index * 2u + 1u] = positions[positionsOffset + index * 2u + 1u] - bitcast<f32>(state[stateOffset + 9u]);`
    );
  }
  // Five scalar means: relative/canonical x/y, then sphere x/y/z for geographic inputs.
  for (let component = 0; component < (geographic ? 5 : 2); component++) {
    const contributions = floating(`mean-${component}`);
    const value =
      component < 2
        ? `point.${component === 0 ? 'x' : 'y'}`
        : `geodesicUnitVector(point)[${component - 2}]`;
    kernel(
      `mean-${component}-map`,
      [
        bind('positions', centerPositions),
        bind('valid', valid),
        readState,
        bind('contributions', contributions, true)
      ],
      `
      let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
      var value = 0.0;
      if (valid[validOffset + index] != 0u) { value = (${value}) / f32(max(1u, state[stateOffset])); }
      contributions[contributionsOffset + index] = value;`,
      rowCount,
      geographic ? GEODESIC_WGSL : ''
    );
    reduceFloat(`mean-${component}`, contributions, 'sum', 32 + component);
  }
  kernel(
    'center',
    [writeState, bind('order', locationOrder.sortedIndices), bind('positions', positions)],
    `
    var center = vec2f(bitcast<f32>(state[stateOffset + 32u]), bitcast<f32>(state[stateOffset + 33u]));
    ${
      geographic
        ? `let vector = vec3f(bitcast<f32>(state[stateOffset + 34u]), bitcast<f32>(state[stateOffset + 35u]), bitcast<f32>(state[stateOffset + 36u]));
    // f32 trig cancellation is larger than the CPU's f64 threshold at antipodes.
    if (length(vector) > 0.000001) { center = geodesicFromUnitVector(vector); }`
        : `center += vec2f(bitcast<f32>(state[stateOffset + 8u]), bitcast<f32>(state[stateOffset + 9u]));`
    }
    if (state[stateOffset + 25u] == 1u) {
      let row = order[orderOffset];
      center = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
    }
    state[stateOffset + 12u] = bitcast<u32>(center.x);
    state[stateOffset + 13u] = bitcast<u32>(center.y);`,
    1,
    geographic ? GEODESIC_WGSL : ''
  );

  if (geographic) {
    const sortedLongitudes = unsigned('sorted-longitudes');
    const sortedRows = unsigned('longitude-rows');
    nodes.push(
      ...new GPUSort({
        id: `${id}-longitude-sort`,
        keys: longitudeKeys,
        values: indices,
        outputKeys: sortedLongitudes,
        outputValues: sortedRows
      }).getCommandNodes(graph)
    );
    const gaps = floating('longitude-gaps');
    kernel(
      'gaps',
      [bind('longitudes', sortedLongitudes), readState, bind('gaps', gaps, true)],
      `
      let count = state[stateOffset]; var gap = -1.0;
      if (index < count) {
        let next = select(index + 1u, 0u, index + 1u == count);
        let nextLongitude = decodeOrderedKey(longitudes[longitudesOffset + next] ^ 0x80000000u);
        let longitude = decodeOrderedKey(longitudes[longitudesOffset + index] ^ 0x80000000u);
        gap = nextLongitude - longitude;
        if (gap < 0.0 || (gap == 0.0 && index + 1u == count)) { gap += 360.0; }
      }
      gaps[gapsOffset + index] = gap;`,
      rowCount,
      DECODE_ORDERED_KEY_WGSL
    );
    reduceFloat('maximum-gap', gaps, 'max', 28);
    const candidates = unsigned('gap-candidates');
    kernel(
      'gap-candidates',
      [bind('gaps', gaps), readState, bind('candidates', candidates, true)],
      `
      candidates[candidatesOffset + index] = select(0xffffffffu, index,
        index < state[stateOffset] && gaps[gapsOffset + index] == bitcast<f32>(state[stateOffset + 28u]));`
    );
    reduceUnsigned('gap-index', candidates, 'min', 29);
    kernel(
      'covering-arc',
      [bind('longitudes', sortedLongitudes), writeState],
      `
      let count = state[stateOffset]; let gap = bitcast<f32>(state[stateOffset + 28u]);
      let plainWidth = bitcast<f32>(state[stateOffset + 10u]) - bitcast<f32>(state[stateOffset + 8u]);
      if (count > 0u && gap > 0.0 && 360.0 - gap < plainWidth - 0.000000001) {
        let row = state[stateOffset + 29u]; let next = select(row + 1u, 0u, row + 1u == count);
        let west = decodeOrderedKey(longitudes[longitudesOffset + next] ^ 0x80000000u);
        let east = decodeOrderedKey(longitudes[longitudesOffset + row] ^ 0x80000000u);
        state[stateOffset + 8u] = bitcast<u32>(west); state[stateOffset + 10u] = bitcast<u32>(east);
      }`,
      1,
      DECODE_ORDERED_KEY_WGSL
    );
  }

  const parameters = floating('search-parameters', 12);
  kernel(
    'search-parameters',
    [readState, bind('parameters', parameters, true)],
    `
    for (var word = 0u; word < 12u; word++) { parameters[parametersOffset + word] = 0.0; }
    ${
      geographic
        ? `var west = -180.0; var east = 180.0; var south = -90.0; var north = 90.0;
    if (state[stateOffset] > 0u) {
      west = bitcast<f32>(state[stateOffset + 38u]); east = bitcast<f32>(state[stateOffset + 39u]);
      south = bitcast<f32>(state[stateOffset + 9u]); north = bitcast<f32>(state[stateOffset + 11u]);
      // GPUNeighborSearch uses the same positions for indexing and distance evaluation.
      // Keep canonical f32 positions unchanged; a broad/seam-crossing set uses world bounds.
      if (east - west > 180.0) { west = -180.0; east = 180.0; }
    }
    parameters[parametersOffset] = west; parameters[parametersOffset + 1u] = south;
    parameters[parametersOffset + 2u] = east; parameters[parametersOffset + 3u] = north;`
        : `
    var minimum = vec2f(0.0); var maximum = vec2f(1.0);
    if (state[stateOffset] > 0u) {
      minimum = vec2f(bitcast<f32>(state[stateOffset + 8u]), bitcast<f32>(state[stateOffset + 9u]));
      maximum = vec2f(bitcast<f32>(state[stateOffset + 10u]), bitcast<f32>(state[stateOffset + 11u]));
      // Degenerate axes need a nonzero lattice width, without excluding any source row.
      if (maximum.x == minimum.x) { maximum.x = minimum.x + max(1.0, abs(minimum.x) * 0.000001); }
      if (maximum.y == minimum.y) { maximum.y = minimum.y + max(1.0, abs(minimum.y) * 0.000001); }
    }
    parameters[parametersOffset] = minimum.x; parameters[parametersOffset + 1u] = minimum.y;
    parameters[parametersOffset + 2u] = maximum.x; parameters[parametersOffset + 3u] = maximum.y;`
    }
    parameters[parametersOffset + 4u] = 0.0; // Unbounded kNN radius.
    parameters[parametersOffset + 6u] = 1.0;
    parameters[parametersOffset + 7u] = 0.000001;`,
    1
  );

  const sortDistances = (name: string, keys: GraphDataView<'uint32'>) => {
    const sorted = unsigned(`${name}-sorted`);
    const sortedRows = unsigned(`${name}-rows`);
    nodes.push(
      ...new GPUSort({
        id: `${id}-${name}-sort`,
        keys,
        values: indices,
        outputKeys: sorted,
        outputValues: sortedRows
      }).getCommandNodes(graph)
    );
    return sorted;
  };
  // A fixed-size helper uses only O(1) order statistics. Missing rows sort after every finite distance.
  const quantileWGSL = `fn quantile(fraction: f32, count: u32) -> f32 {
    if (count == 0u) { return 0.0; }
    let rank = fraction * f32(count - 1u); let lower = u32(floor(rank)); let upper = min(lower + 1u, count - 1u);
    return mix(bitcast<f32>(sorted[sortedOffset + lower]), bitcast<f32>(sorted[sortedOffset + upper]), rank - f32(lower));
  }`;
  const centerKeys = unsigned('center-distance-keys');
  const squared = floating('squared-center-distance');
  kernel(
    'center-distances',
    [
      bind('positions', centerPositions),
      bind('valid', valid),
      readState,
      bind('keys', centerKeys, true),
      bind('squared', squared, true)
    ],
    `
    let center = vec2f(bitcast<f32>(state[stateOffset + ${geographic ? 12 : 32}u]), bitcast<f32>(state[stateOffset + ${geographic ? 13 : 33}u]));
    let point = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    var distance = 0.0;
    let keep = valid[validOffset + index] != 0u;
    if (keep) { distance = ${geographic ? 'geodesicHaversineDistance(point, center, 6371008.8)' : 'length(point - center)'}; }
    keys[keysOffset + index] = select(0xffffffffu, bitcast<u32>(distance), keep);
    squared[squaredOffset + index] = distance * distance / f32(max(1u, state[stateOffset]));`,
    rowCount,
    geographic ? GEODESIC_WGSL : ''
  );
  reduceFloat('squared-distance', squared, 'sum', 14);
  const sortedCenter = sortDistances('center-distance', centerKeys);
  kernel(
    'center-summary',
    [bind('sorted', sortedCenter), writeState],
    `
    state[stateOffset + 14u] = bitcast<u32>(sqrt(bitcast<f32>(state[stateOffset + 14u])));
    state[stateOffset + 15u] = bitcast<u32>(quantile(0.5, state[stateOffset]));`,
    1,
    quantileWGSL
  );

  const nearestKeys = unsigned('nearest-keys');
  const kthKeys = unsigned('kth-keys');
  const nearestContributions = floating('nearest-contributions');
  const distinctKeys = unsigned('distinct-distance-keys');
  for (const distinct of [false, true]) {
    const name = distinct ? 'distinct-search' : 'full-search';
    const rank = distinct ? 1 : neighborRank;
    const capacity = distinct ? rowCount : neighborCapacity;
    const offsets = unsigned(`${name}-offsets`, rowCount + 1);
    const neighbors = unsigned(`${name}-neighbors`, capacity);
    const distances = floating(`${name}-distances`, capacity);
    const weights = floating(`${name}-weights`, capacity);
    const overflow = unsigned(`${name}-overflow`, 1);
    const requiredCount = unsigned(`${name}-required`, 1);
    nodes.push(
      ...new GPUNeighborSearch({
        id: `${id}-${name}`,
        positions,
        mask: distinct ? distinctMask : valid,
        queryMask: distinct ? undefined : queryMask,
        mode: 'knn',
        k: rank,
        gridSize,
        parameters,
        spatialContext: contributor.spatialContext,
        rowTileSize: 16384,
        output: {offsets, neighbors, distances, weights, status: {overflow, requiredCount}}
      }).getCommandNodes(graph)
    );
    kernel(
      `${name}-status`,
      [bind('overflow', overflow), bind('required', requiredCount), writeState],
      `
      state[stateOffset + 27u] |= overflow[overflowOffset];
      ${distinct ? '' : 'state[stateOffset + 3u] = required[requiredOffset];'}`,
      1
    );
    kernel(
      `${name}-distances`,
      [
        bind('offsets', offsets),
        bind('distances', distances),
        readState,
        ...(distinct
          ? [bind('keys', distinctKeys, true)]
          : [
              bind('keys', nearestKeys, true),
              bind('kth', kthKeys, true),
              bind('contributions', nearestContributions, true),
              bind('multiplicities', multiplicities)
            ])
      ],
      `
      let start = offsets[offsetsOffset + index]; let end = offsets[offsetsOffset + index + 1u];
      var nearest = 3.402823466e38; var farthest = 0.0;
      // CSR rows are ID ordered, so both extrema must visit every bounded rank slot.
      for (var offset = 0u; offset < ${rank}u; offset++) {
        let slot = start + offset;
        if (slot < end && slot < ${capacity}u) {
          let distance = distances[distancesOffset + slot];
          nearest = min(nearest, distance); farthest = max(farthest, distance);
        }
      }
      keys[keysOffset + index] = select(0xffffffffu, bitcast<u32>(nearest), end > start);
      ${
        distinct
          ? ''
          : `kth[kthOffset + index] = select(0xffffffffu, bitcast<u32>(farthest), end - start == ${rank}u);
      contributions[contributionsOffset + index] = select(0.0, nearest / f32(max(1u, state[stateOffset])), end > start);
      if (multiplicities[multiplicitiesOffset + index] > ${rank}u) {
        keys[keysOffset + index] = 0u; kth[kthOffset + index] = 0u;
      }`
      }`
    );
  }
  reduceFloat('nearest-mean', nearestContributions, 'sum', 20);
  const sortedNearest = sortDistances('nearest', nearestKeys);
  const sortedDistinct = sortDistances('distinct-distance', distinctKeys);
  const sortedKth = sortDistances('kth', kthKeys);
  kernel(
    'nearest-summary',
    [bind('sorted', sortedNearest), writeState],
    `
    let count = state[stateOffset];
    if (count >= 2u) {
      let first = quantile(0.0, count); let lower = quantile(0.25, count); let middle = quantile(0.5, count);
      let upper = quantile(0.75, count); let last = quantile(1.0, count);
      state[stateOffset + 16u] = bitcast<u32>(first); state[stateOffset + 17u] = bitcast<u32>(lower);
      state[stateOffset + 18u] = bitcast<u32>(middle); state[stateOffset + 19u] = bitcast<u32>(upper);
      state[stateOffset + 21u] = bitcast<u32>(last);
      state[stateOffset + 22u] = bitcast<u32>(min(last, upper + 3.0 * (upper - lower)));
    }`,
    1,
    quantileWGSL
  );
  kernel(
    'distinct-summary',
    [bind('sorted', sortedDistinct), writeState],
    `
    if (state[stateOffset + 25u] >= 2u) { state[stateOffset + 24u] = bitcast<u32>(quantile(0.5, state[stateOffset + 25u])); }`,
    1,
    quantileWGSL
  );
  const aboveFence = unsigned('above-fence');
  kernel(
    'fence-count',
    [bind('keys', nearestKeys), readState, bind('above', aboveFence, true)],
    `
    let key = keys[keysOffset + index];
    above[aboveOffset + index] = select(0u, 1u, key != 0xffffffffu && bitcast<f32>(key) > bitcast<f32>(state[stateOffset + 22u]));`
  );
  reduceUnsigned('above-fence-count', aboveFence, 'sum', 23);
  kernel(
    'curve-count',
    [writeState],
    `
    state[stateOffset + 6u] = select(0u, min(state[stateOffset], 512u), state[stateOffset] > ${neighborRank}u);
    // Preserve the relative planar centre so decoding can add the origin in host f64.
    state[stateOffset + 28u] = ${geographic ? '0u' : 'state[stateOffset + 32u]'};
    state[stateOffset + 29u] = ${geographic ? '0u' : 'state[stateOffset + 33u]'};
    state[stateOffset + 30u] = 0u; state[stateOffset + 31u] = 0u;`,
    1
  );
  kernel(
    'publish-summary',
    [readState, bind('summary', props.output.summary, true)],
    'summary[summaryOffset + index] = state[stateOffset + index];',
    32
  );
  kernel(
    'publish-curve',
    [bind('sorted', sortedKth), readState, bind('summary', props.output.summary, true)],
    `
    let length = state[stateOffset + 6u]; var distance = 0.0;
    if (index < length) {
      // The CPU display curve selects the nearest order statistic, without interpolation.
      let denominator = max(1u, length - 1u);
      // At most 511 * 999999: exact u32 arithmetic avoids f32 half-rank rounding.
      let rank = (index * (state[stateOffset] - 1u) + denominator / 2u) / denominator;
      distance = bitcast<f32>(sorted[sortedOffset + min(rank, state[stateOffset] - 1u)]);
    }
    summary[summaryOffset + 32u + index] = bitcast<u32>(distance);`,
    512
  );
  return nodes;
}
