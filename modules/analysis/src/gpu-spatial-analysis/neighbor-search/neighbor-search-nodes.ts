// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSaturatedCountScanNodes} from '../../analysis-core/kernels/scan-total';
import {getGPUNeighborSearchExecutionPlan} from './neighbor-search-planning';
import {getGPUGreatCircleRadius} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  getViewBindingRange,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTiledWGSLKernelNodes} from '../../analysis-core/kernels/wgsl-kernel-tiles';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUGridIndexMaximumCellCount,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getBVHNearestNeighborSearchWGSL,
  getBVHNeighborSearchDeclarationsWGSL,
  getBVHRadiusNeighborLoopWGSL,
  getBVHThreadToRowWGSL,
  getGeodesicNearestNeighborSearchWGSL,
  getGeodesicRadiusNeighborLoopWGSL,
  getInsertNeighborWGSL,
  getNearestNeighborSearchWGSL,
  getNeighborSearchLatticeWGSL,
  getRadiusNeighborLoopWGSL,
  NEIGHBOR_SEARCH_BVH_NODE_WORDS,
  NEIGHBOR_WEIGHT_WGSL,
  type NeighborSearchTimeAxis
} from './neighbor-search-kernels';
import {getNearestBVHNodes} from '../spatial-join/nearest-nodes';
import {
  createPointNeighborGridIndexNodes,
  POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL
} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON} from '../spatial-weights/gpu-spatial-weights-transform';

import type {GPUNeighborSearch} from './gpu-neighbor-search';

const OPERATION = 'GPUNeighborSearch';

/** Lowers a validated neighbor search without changing its kernel text or graph order. @internal */
export function getGPUNeighborSearchNodes<Parameters>(
  contributor: GPUNeighborSearch,
  graph: GPUCommandGraph<Parameters>
): readonly GPUCommandNode<Parameters>[] {
  const {id, props} = contributor;
  const {positions, queryPositions, parameters, gridSize, output: weights, mode} = props;
  validateGraphViewsBelongToGraph(id, graph, [
    positions,
    queryPositions,
    props.mask,
    props.queryMask,
    parameters,
    props.times,
    props.queryTimes,
    weights.offsets,
    weights.neighbors,
    weights.weights,
    weights.distances,
    weights.status.overflow,
    weights.status.candidateOverflow,
    weights.status.requiredCount,
    weights.status.invalidCount,
    weights.neighborCounts
  ]);
  const {times} = props;
  const timeBucketCount = times ? (props.timeBucketCount ?? 1) : 1;
  const timeAxis: NeighborSearchTimeAxis | undefined = times
    ? {bucketCount: timeBucketCount}
    : undefined;
  const bucketed = timeBucketCount > 1;
  const plan = getGPUNeighborSearchExecutionPlan(
    {
      mode,
      searchStructure: contributor.searchStructure,
      coordinateSpace: contributor.spatialContext.coordinateSpace,
      hasTimes: Boolean(times),
      targetRowCount: positions.length,
      queryRowCount: contributor.getQueryRows(),
      hasQueryPositions: Boolean(queryPositions),
      hasQueryMask: Boolean(props.queryMask),
      includeInvalidCount: Boolean(weights.status.invalidCount),
      distanceCapacity: weights.distances ? undefined : weights.neighbors.length,
      gridSize,
      timeBucketCount: times ? timeBucketCount : undefined,
      neighborCount: props.neighborCount
    },
    graph.device.limits
  );
  for (const constraint of plan.constraints) {
    validateRequirement(
      id,
      constraint.satisfied,
      `${constraint.name} needs ${constraint.received} bytes, above the device limit of ${constraint.maximum}; reduce rows, grid size or neighborCount, or use smaller independent queries`
    );
  }
  const cellCount = plan.gridCellCount;
  validateMaximumCount(
    id,
    'gridSize cells',
    cellCount,
    getGPUGridIndexMaximumCellCount(graph.device)
  );
  const targetRows = positions.length;
  const queryRows = contributor.getQueryRows();
  const capacity = weights.neighbors.length;
  if (weights.status.candidateOverflow) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'output.status.candidateOverflow',
      getViewBindingRange(weights.status.candidateOverflow).size,
      'use a smaller packed status view'
    );
  }
  const neighborCount = props.neighborCount ?? 1;
  if (mode === 'knn') {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'k-nearest lists',
      plan.nearestListLength * 4,
      'lower neighborCount or the query row count'
    );
  }
  const {spatialContext} = contributor;
  const geodesic = spatialContext.coordinateSpace === 'longitude-latitude';
  // Sphere radius in the context's length unit.
  const sphereRadius = getGPUGreatCircleRadius(spatialContext);
  const latticeWGSL = `${getNeighborSearchLatticeWGSL(mode, gridSize, sphereRadius, timeAxis)}
${POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL}
${
  timeAxis
    ? `fn getGridUnitTime(lattice: Lattice, time: f32) -> f32 {
  // The grid drops a coordinate above 1, and float32 rounding can put the latest time there.
  return min(1.0, (time - lattice.minimumTime) / (lattice.timeBucketWidth * f32(TIME_BUCKETS)));
}`
    : ''
}`;
  const radiusLoop = geodesic ? getGeodesicRadiusNeighborLoopWGSL : getRadiusNeighborLoopWGSL;
  const crossJoin = Boolean(queryPositions);
  const queryBinding: WGSLKernelBinding[] = queryPositions
    ? [{name: 'queryPositions', view: queryPositions, type: 'f32', access: 'read'}]
    : [];
  // A target or query record carries its time as a fourth word.
  const recordWords = plan.recordWords;
  const queryTimes = props.queryTimes ?? times;
  const timeBindings = (name: 'times' | 'queryTimes'): WGSLKernelBinding[] => {
    const view = name === 'times' ? times : queryTimes;
    return view ? [{name, view, type: 'f32', access: 'read'}] : [];
  };
  const readTimeWGSL = (name: 'times' | 'queryTimes', row: string) =>
    `${name}[${name}Offset + ${row}]`;
  const queryTimeName: 'times' | 'queryTimes' = crossJoin ? 'queryTimes' : 'times';
  const queryTimeBinding = timeBindings(queryTimeName);
  // A cross-join query needs finite coordinates and time (and a spherical latitude); self-join rows
  // are its target rows, which stay bound by the bounds.
  const queryPointInBoundsWGSL = crossJoin ? 'isCrossJoinQueryInBounds' : 'isPointInBounds';
  const queryPointValidWGSL = crossJoin ? 'isCrossJoinQueryValid' : 'isPointValid';
  const queryTimeInBoundsWGSL = (row: string) =>
    crossJoin
      ? `isCrossJoinQueryTimeInBounds(${readTimeWGSL(queryTimeName, row)})`
      : `isTimeInBounds(lattice, ${readTimeWGSL(queryTimeName, row)})`;
  const queryTimeValidWGSL = (row: string) => (times ? ` && ${queryTimeInBoundsWGSL(row)}` : '');
  const queryName = crossJoin ? 'queryPositions' : 'positions';
  const readQueryWGSL = (row: string) => `let x = ${queryName}[${queryName}Offset + ${row} * 2u];
  let y = ${queryName}[${queryName}Offset + ${row} * 2u + 1u];`;
  const selfCondition = crossJoin ? 'true' : 'neighbor != queryIndex';
  // A self join applies the target mask to the query rows too unless a query mask is given.
  const queryMask = props.queryMask ?? (crossJoin ? undefined : props.mask);
  // Threads of the search kernels walk the valid queries in grid-cell order, so neighboring
  // threads search neighboring cells (coalesced reads, coherent control flow) however the
  // input is ordered. The thread's query row (`row`) is where its results are written, so
  // outputs are identical to launching in row order. A self join whose query set equals the
  // target set reuses the target index; otherwise the queries get an index of their own.
  const sharesTargetIndex = !crossJoin && !props.queryMask;

  // Resolve before allocating/building: the measured clustered and uniform workloads favor
  // the grid. An unused BVH must not add a sort/build to every auto encoding.
  const useBvh = plan.route === 'bvh';
  const useGrid = !useBvh;
  // Grid-only scratch is one element when the grid is not used.
  const gridSized = (length: number) => (useGrid ? length : 1);

  // Time is a third lattice dimension only when the time axis is bucketed.
  const gridDimension = plan.gridDimension;
  const gridFormat = bucketed ? ('float32x3' as const) : ('float32x2' as const);
  // Typed as two components because the BVH path, which never buckets, reads it that way.
  const gridPositions = createTransientView(
    graph,
    `${id}-grid-positions`,
    gridFormat,
    targetRows
  ) as GraphDataView<'float32x2'>;
  const fullGridSize = bucketed
    ? ([gridSize[0], gridSize[1], timeBucketCount] as const)
    : ([gridSize[0], gridSize[1]] as const);
  const targetIndex = useGrid
    ? createPointNeighborGridIndexNodes<Parameters>(graph, {
        id,
        positions: gridPositions,
        gridSize: fullGridSize,
        rowCount: targetRows
      })
    : undefined;
  const cellOffsets =
    targetIndex?.cellOffsets ?? createTransientView(graph, `${id}-cell-offsets`, 'uint32', 1);
  const sortedRows =
    targetIndex?.sortedRows ?? createTransientView(graph, `${id}-sorted-rows`, 'uint32', 1);
  const gridCount =
    targetIndex?.gridCount ?? createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
  // Target `[x, y, row]` records (coordinates as float bits; with times a fourth time word) in
  // cell order: the search loops read them sequentially instead of gathering, and one binding
  // carries every field.
  const sortedTargets = createTransientView(
    graph,
    `${id}-sorted-targets`,
    'uint32',
    gridSized(targetRows * recordWords)
  );
  const counts =
    weights.neighborCounts ?? createTransientView(graph, `${id}-counts`, 'uint32', queryRows);
  const starts = createTransientView(graph, `${id}-starts`, 'uint32', queryRows);
  const distances =
    weights.distances ?? createTransientView(graph, `${id}-distances`, 'float32', capacity);
  // Replaced by the query index below when the queries are not the targets.
  let queryOrder: GraphDataView<'uint32'> = sortedRows;
  // Valid query records in cell order; a shared index reuses the target records.
  const sortedQueries = sharesTargetIndex
    ? sortedTargets
    : createTransientView(
        graph,
        `${id}-sorted-queries`,
        'uint32',
        gridSized(queryRows * recordWords)
      );
  let queryCount: GraphDataView<'uint32'> = gridCount;

  const nodes: GPUCommandNode<Parameters>[] = [
    ...(weights.status.candidateOverflow
      ? [
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-candidate-status`,
            operation: OPERATION,
            invocationCount: 1,
            bindings: [
              {
                name: 'candidateOverflow',
                view: weights.status.candidateOverflow,
                type: 'u32',
                access: 'read_write'
              }
            ],
            body: 'candidateOverflow[candidateOverflowOffset] = 0u;'
          })
        ]
      : []),
    // The grid index has no mask and its domain is wider than the valid extent, so excluded
    // targets become NaN points, which it ignores. The BVH build treats a NaN point as an empty
    // box, so this one array carries target validity (mask, finite, bounds) into both structures.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-grid-positions`,
      operation: OPERATION,
      variant: 'grid-positions',
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        ...(props.mask
          ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        ...timeBindings('times'),
        {name: 'gridPositions', view: gridPositions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: targetRows,
      declarations: latticeWGSL,
      body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'} && isPointValid(lattice, x, y)${times ? ` && isTimeInBounds(lattice, ${readTimeWGSL('times', 'index')})` : ''};
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  ${useGrid ? 'let unit = getGridUnitPosition(lattice, x, y);' : 'let unit = vec2f(x, y);'}
  gridPositions[gridPositionsOffset + index * ${gridDimension}u] = select(invalid, unit.x, included);
  gridPositions[gridPositionsOffset + index * ${gridDimension}u + 1u] = select(invalid, unit.y, included);${
    bucketed
      ? `
  gridPositions[gridPositionsOffset + index * 3u + 2u] = select(invalid, getGridUnitTime(lattice, ${readTimeWGSL('times', 'index')}), included);`
      : ''
  }`
    }),
    // Buckets targets into the lattice: `cellOffsets` and `sortedRows` (target rows by cell, in
    // unspecified order within a cell; every consumer below is order independent or sorts).
    ...(targetIndex?.nodes ?? []),
    // Query validity, stored in `counts` until the search pass overwrites it with the count.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-query-valid`,
      operation: OPERATION,
      variant: 'query-valid',
      bindings: [
        ...(crossJoin
          ? queryBinding
          : [{name: 'positions', view: positions, type: 'f32' as const, access: 'read' as const}]),
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        ...(queryMask
          ? [
              {
                name: 'queryMask',
                view: queryMask,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...queryTimeBinding,
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: queryRows,
      declarations: latticeWGSL,
      body: `let lattice = readLattice();
  ${readQueryWGSL('index')}
  let included = ${queryMask ? 'queryMask[queryMaskOffset + index] != 0u' : 'true'};
  counts[countsOffset + index] = select(0u, 1u, included && ${queryPointValidWGSL}(lattice, x, y)${queryTimeValidWGSL('index')});`
    })
  ];

  if (useGrid && !sharesTargetIndex) {
    const queryGridPositions = createTransientView(
      graph,
      `${id}-query-grid-positions`,
      gridFormat,
      queryRows
    ) as GraphDataView<'float32x2'>;
    // Only the cell order of the queries is used; the cell offsets are scratch.
    const queryIndex = createPointNeighborGridIndexNodes<Parameters>(graph, {
      id: `${id}-query`,
      positions: queryGridPositions,
      gridSize: fullGridSize,
      rowCount: queryRows
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-query-grid-positions`,
        operation: OPERATION,
        variant: 'query-grid-positions',
        bindings: [
          ...(crossJoin
            ? queryBinding
            : [
                {
                  name: 'positions',
                  view: positions,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]),
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...(queryMask
            ? [
                {
                  name: 'queryMask',
                  view: queryMask,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          ...queryTimeBinding,
          {
            name: 'queryGridPositions',
            view: queryGridPositions,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: queryRows,
        declarations: latticeWGSL,
        body: `let lattice = readLattice();
  ${readQueryWGSL('index')}
  let included = ${queryMask ? 'queryMask[queryMaskOffset + index] != 0u' : 'true'} && ${queryPointValidWGSL}(lattice, x, y)${queryTimeValidWGSL('index')};
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  // Only the cell order of the queries is used, so a query outside the lattice sorts with the
  // nearest edge cell; the grid index would otherwise skip it.
  let unit = clamp(getGridUnitPosition(lattice, x, y), vec2f(0.0), vec2f(1.0));
  queryGridPositions[queryGridPositionsOffset + index * ${gridDimension}u] = select(invalid, unit.x, included);
  queryGridPositions[queryGridPositionsOffset + index * ${gridDimension}u + 1u] = select(invalid, unit.y, included);${
    bucketed
      ? `
  queryGridPositions[queryGridPositionsOffset + index * 3u + 2u] = select(invalid, max(0.0, getGridUnitTime(lattice, ${readTimeWGSL(queryTimeName, 'index')})), included);`
      : ''
  }`
      }),
      ...queryIndex.nodes
    );
    queryOrder = queryIndex.sortedRows;
    queryCount = queryIndex.gridCount;
  }
  // Kernels bind at most 8 storage buffers, so the target and query triples are single buffers
  // and a shared index binds each of them once under the target names.
  const queryBufferName = sharesTargetIndex ? 'sortedTargets' : 'sortedQueries';
  const queryCountName = sharesTargetIndex ? 'gridCount' : 'queryCount';
  const searchBindings: WGSLKernelBinding[] = [
    {name: 'sortedTargets', view: sortedTargets, type: 'u32', access: 'read'},
    ...(sharesTargetIndex
      ? []
      : [
          {
            name: 'sortedQueries',
            view: sortedQueries,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]),
    {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
    {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
    {name: queryCountName, view: queryCount, type: 'u32', access: 'read'}
  ];
  // Maps the thread to the valid query at that cell-order slot and reads its point; surplus
  // threads (invalid queries have no work and a zero count from the query-valid pass) return.
  const threadToRowWGSL = `if (index >= ${queryCountName}[${queryCountName}Offset]) {
    return;
  }
  let queryIndex = ${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS + 2u];
  let x = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS]);
  let y = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS + 1u]);${
    times
      ? `
  let queryTime = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * TARGET_WORDS + 3u]);`
      : ''
  }`;

  if (useGrid) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sorted-targets`,
        operation: OPERATION,
        variant: 'sorted-targets',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...timeBindings('times'),
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'gridCount', view: gridCount, type: 'u32', access: 'read'},
          {name: 'sortedTargets', view: sortedTargets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: targetRows,
        body: `if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u] = bitcast<u32>(positions[positionsOffset + row * 2u]);
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u + 1u] = bitcast<u32>(positions[positionsOffset + row * 2u + 1u]);
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u + 2u] = row;${
    times
      ? `
  sortedTargets[sortedTargetsOffset + index * ${recordWords}u + 3u] = bitcast<u32>(${readTimeWGSL('times', 'row')});`
      : ''
  }`
      })
    );
  }
  if (useGrid && !sharesTargetIndex) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sorted-queries`,
        operation: OPERATION,
        variant: 'sorted-queries',
        bindings: [
          {
            name: 'queryPositions',
            view: queryPositions ?? positions,
            type: 'f32',
            access: 'read'
          },
          ...timeBindings('queryTimes'),
          {name: 'queryOrder', view: queryOrder, type: 'u32', access: 'read'},
          {name: 'queryCount', view: queryCount, type: 'u32', access: 'read'},
          {name: 'sortedQueries', view: sortedQueries, type: 'u32', access: 'read_write'}
        ],
        invocationCount: queryRows,
        body: `if (index >= queryCount[queryCountOffset]) {
    return;
  }
  let row = queryOrder[queryOrderOffset + index];
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u]);
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u + 1u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u + 1u]);
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u + 2u] = row;${
    times
      ? `
  sortedQueries[sortedQueriesOffset + index * ${recordWords}u + 3u] = bitcast<u32>(${readTimeWGSL('queryTimes', 'row')});`
      : ''
  }`
      })
    );
  }

  // BVH over the valid targets (NaN points of `gridPositions` become empty boxes), packed with
  // the leaf rows into one node table so a search kernel binds the whole tree once.
  let bvhNodeData: GraphDataView<'uint32'> | undefined;
  let bvhDeclarations = '';
  if (useBvh) {
    const {bvh, nodes: bvhNodes} = getNearestBVHNodes(graph, {
      id: `${id}-bvh`,
      operation: OPERATION,
      features: {kind: 'points', positions: gridPositions},
      featureCount: targetRows,
      leafCapacity: plan.bvhLeafCapacity,
      spatialSort: true
    });
    nodes.push(...bvhNodes);
    bvhNodeData = createTransientView(
      graph,
      `${id}-bvh-node-data`,
      'uint32',
      plan.bvhNodeCount * NEIGHBOR_SEARCH_BVH_NODE_WORDS
    );
    bvhDeclarations = getBVHNeighborSearchDeclarationsWGSL({
      targetRows,
      internalNodeCount: bvh.internalNodeCount,
      levelCount: bvh.levelCount
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bvh-node-data`,
        operation: OPERATION,
        variant: 'bvh-node-data',
        bindings: [
          {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
          {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
          {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
          {name: 'nodeData', view: bvhNodeData, type: 'u32', access: 'read_write'}
        ],
        invocationCount: bvh.nodeCount,
        declarations: `const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const NO_ROW: u32 = 0xffffffffu;`,
        body: `let base = nodeDataOffset + index * ${NEIGHBOR_SEARCH_BVH_NODE_WORDS}u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_ROW, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`
      })
    );
  }

  // BVH kernels bind the tree, the query points and the parameters (and the validity in `counts`).
  const bvhSearchBindings: WGSLKernelBinding[] = bvhNodeData
    ? [
        {name: 'nodeData', view: bvhNodeData, type: 'u32', access: 'read'},
        {name: queryName, view: queryPositions ?? positions, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'}
      ]
    : [];
  const bvhThreadToRowWGSL = getBVHThreadToRowWGSL(crossJoin);

  let knnIds: GraphDataView<'uint32'> | undefined;
  let knnDistances: GraphDataView<'float32'> | undefined;
  if (mode === 'knn') {
    knnIds = createTransientView(graph, `${id}-knn-ids`, 'uint32', plan.nearestListLength);
    knnDistances = createTransientView(
      graph,
      `${id}-knn-distances`,
      'float32',
      plan.nearestListLength
    );
    const knnTailWGSL = `// Re-sort the selected neighbors by ID (insertion sort over at most K entries).
  for (var slot = 1u; slot < found; slot++) {
    let neighborId = bestIds[slot];
    let distanceSquared = bestDistances[slot];
    var destination = slot;
    while (destination > 0u && bestIds[destination - 1u] > neighborId) {
      bestIds[destination] = bestIds[destination - 1u];
      bestDistances[destination] = bestDistances[destination - 1u];
      destination--;
    }
    bestIds[destination] = neighborId;
    bestDistances[destination] = distanceSquared;
  }
  for (var slot = 0u; slot < found; slot++) {
    knnIds[knnIdsOffset + queryIndex * K + slot] = bestIds[slot];
    knnDistances[knnDistancesOffset + queryIndex * K + slot] = ${
      geodesic ? 'bestDistances[slot]' : 'sqrt(bestDistances[slot])'
    };
  }
  counts[countsOffset + queryIndex] = found;`;
    const knnColumns: WGSLKernelBinding[] = [
      {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
      {name: 'knnIds', view: knnIds, type: 'u32', access: 'read_write'},
      {name: 'knnDistances', view: knnDistances, type: 'f32', access: 'read_write'}
    ];
    if (useGrid) {
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-knn`,
            operation: OPERATION,
            variant: 'knn',
            bindings: [...searchBindings, ...knnColumns],
            invocationCount: queryRows,
            declarations: `${latticeWGSL}
${getInsertNeighborWGSL(neighborCount)}`,
            body: `${threadToRowWGSL}
  var found = 0u;
  var bestDistances: array<f32, ${neighborCount}>;
  var bestIds: array<u32, ${neighborCount}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${
      geodesic
        ? getGeodesicNearestNeighborSearchWGSL(selfCondition, timeAxis)
        : getNearestNeighborSearchWGSL(selfCondition, timeAxis)
    }
  }
  ${knnTailWGSL}`
          },
          props.rowTileSize
        )
      );
    }
    if (useBvh) {
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-bvh-knn`,
            operation: OPERATION,
            variant: 'bvh-knn',
            bindings: [...bvhSearchBindings, ...knnColumns],
            invocationCount: queryRows,
            declarations: `${latticeWGSL}
${getInsertNeighborWGSL(neighborCount)}
${bvhDeclarations}`,
            body: `${bvhThreadToRowWGSL}
  var found = 0u;
  var bestDistances: array<f32, ${neighborCount}>;
  var bestIds: array<u32, ${neighborCount}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    ${getBVHNearestNeighborSearchWGSL(selfCondition)}
  }
  ${knnTailWGSL}`
          },
          props.rowTileSize
        )
      );
    }
  } else {
    if (useGrid) {
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-count`,
            operation: OPERATION,
            variant: 'radius-count',
            bindings: [
              ...searchBindings,
              {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
            ],
            invocationCount: queryRows,
            declarations: latticeWGSL,
            body: `${threadToRowWGSL}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${radiusLoop(
      `if (${selfCondition}) {
          count++;
        }`,
      timeAxis
    )}
  }
  counts[countsOffset + queryIndex] = count;`
          },
          props.rowTileSize
        )
      );
    }
    if (useBvh) {
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-bvh-count`,
            operation: OPERATION,
            variant: 'bvh-radius-count',
            bindings: [
              ...bvhSearchBindings,
              {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
            ],
            invocationCount: queryRows,
            declarations: `${latticeWGSL}
${bvhDeclarations}`,
            body: `${bvhThreadToRowWGSL}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    ${getBVHRadiusNeighborLoopWGSL(`if (${selfCondition}) {
        count++;
      }`)}
  }
  counts[countsOffset + queryIndex] = count;`
          },
          props.rowTileSize
        )
      );
    }
  }

  const scanTotal = createTransientView(graph, `${id}-scan-total`, 'uint32', 1);
  nodes.push(
    ...createSaturatedCountScanNodes<Parameters>(graph, {
      id: `${id}-scan`,
      operation: OPERATION,
      counts,
      offsets: starts,
      rowCount: queryRows,
      total: scanTotal
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation: OPERATION,
      variant: 'offsets',
      bindings: [
        {name: 'scanTotal', view: scanTotal, type: 'u32', access: 'read'},
        {name: 'starts', view: starts, type: 'u32', access: 'read'},
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'},
        {name: 'overflow', view: weights.status.overflow, type: 'u32', access: 'read_write'},
        ...(weights.status.requiredCount
          ? [
              {
                name: 'totalNeighbors',
                view: weights.status.requiredCount,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: queryRows + 1,
      declarations: `const QUERY_ROWS: u32 = ${queryRows}u;
const CAPACITY: u32 = ${capacity}u;`,
      body: `if (index < QUERY_ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = scanTotal[scanTotalOffset];
    offsets[offsetsOffset + QUERY_ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${weights.status.requiredCount ? 'totalNeighbors[totalNeighborsOffset] = total;' : ''}
  }`
    })
  );

  if (knnIds && knnDistances) {
    nodes.push(
      ...createTiledWGSLKernelNodes<Parameters>(
        graph,
        {
          id: `${id}-emit`,
          operation: OPERATION,
          variant: 'knn-emit',
          bindings: [
            {name: 'knnIds', view: knnIds, type: 'u32', access: 'read'},
            {name: 'knnDistances', view: knnDistances, type: 'f32', access: 'read'},
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          declarations: `const K: u32 = ${neighborCount}u;`,
          body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin; slot < end; slot++) {
    let source = index * K + slot - begin;
    neighbors[neighborsOffset + slot] = knnIds[knnIdsOffset + source];
    distances[distancesOffset + slot] = knnDistances[knnDistancesOffset + source];
  }`
        },
        props.rowTileSize
      )
    );
  } else {
    const emitColumns: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
      {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
    ];
    if (useBvh) {
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-bvh-emit`,
            operation: OPERATION,
            variant: 'bvh-radius-emit',
            bindings: [...bvhSearchBindings, ...emitColumns],
            invocationCount: queryRows,
            declarations: `${latticeWGSL}
${bvhDeclarations}`,
            body: `${bvhThreadToRowWGSL}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    var next = begin;
    ${getBVHRadiusNeighborLoopWGSL(`if (${selfCondition} && next < end) {
        neighbors[neighborsOffset + next] = neighbor;
        distances[distancesOffset + next] = sqrt(distanceSquared);
        next++;
      }`)}
  }`
          },
          props.rowTileSize
        )
      );
    }
    if (useGrid) {
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-emit`,
            operation: OPERATION,
            variant: 'radius-emit',
            bindings: [...searchBindings, ...emitColumns],
            invocationCount: queryRows,
            declarations: latticeWGSL,
            body: `${threadToRowWGSL}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    var next = begin;
    ${radiusLoop(
      `if (${selfCondition} && next < end) {
          neighbors[neighborsOffset + next] = neighbor;
          distances[distancesOffset + next] = ${geodesic ? 'distance' : 'sqrt(distanceSquared)'};
          next++;
        }`,
      timeAxis
    )}
  }`
          },
          props.rowTileSize
        )
      );
    }
    nodes.push(
      ...createTiledWGSLKernelNodes<Parameters>(
        graph,
        {
          id: `${id}-sort-rows`,
          operation: OPERATION,
          variant: 'radius-sort-rows',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let neighborId = neighbors[neighborsOffset + slot];
    let distance = distances[distancesOffset + slot];
    var destination = slot;
    while (destination > begin && neighbors[neighborsOffset + destination - 1u] > neighborId) {
      neighbors[neighborsOffset + destination] = neighbors[neighborsOffset + destination - 1u];
      distances[distancesOffset + destination] = distances[distancesOffset + destination - 1u];
      destination--;
    }
    neighbors[neighborsOffset + destination] = neighborId;
    distances[distancesOffset + destination] = distance;
  }`
        },
        props.rowTileSize
      )
    );
  }

  if (weights.status.invalidCount) {
    // Included rows (mask nonzero) with a non-finite coordinate or outside the bounds, summed
    // with a scan so no atomics are needed. Self joins count targets; cross joins add queries.
    const flagRows = targetRows + (crossJoin ? queryRows : 0);
    const invalidFlags = createTransientView(graph, `${id}-invalid-flags`, 'uint32', flagRows);
    const invalidStarts = createTransientView(graph, `${id}-invalid-starts`, 'uint32', flagRows);
    const readPoint = (name: string, row: string) => `let x = ${name}[${name}Offset + ${row} * 2u];
    let y = ${name}[${name}Offset + ${row} * 2u + 1u];`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-invalid-flags`,
        operation: OPERATION,
        variant: 'invalid-flags',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          ...queryBinding,
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...timeBindings('times'),
          ...(crossJoin ? timeBindings('queryTimes') : []),
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          ...(crossJoin && props.queryMask
            ? [
                {
                  name: 'queryMask',
                  view: props.queryMask,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: flagRows,
        declarations: `${latticeWGSL}
const TARGET_ROWS: u32 = ${targetRows}u;`,
        body: `let lattice = readLattice();
  var included = true;
  var inside = false;
  if (index < TARGET_ROWS) {
    ${readPoint('positions', 'index')}
    included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
    inside = isPointInBounds(lattice, x, y)${times ? ` && isTimeInBounds(lattice, ${readTimeWGSL('times', 'index')})` : ''};
  } else {
    ${
      crossJoin
        ? `let queryRow = index - TARGET_ROWS;
    ${readPoint('queryPositions', 'queryRow')}
    included = ${crossJoin && props.queryMask ? 'queryMask[queryMaskOffset + queryRow] != 0u' : 'true'};
    inside = ${queryPointInBoundsWGSL}(lattice, x, y)${times ? ` && ${queryTimeInBoundsWGSL('queryRow')}` : ''};`
        : ''
    }
  }
  invalidFlags[invalidFlagsOffset + index] = select(0u, 1u, included && !inside);`
      }),
      ...new GPUScan({
        id: `${id}-invalid-scan`,
        input: invalidFlags,
        output: invalidStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-invalid-count`,
        operation: OPERATION,
        variant: 'invalid-count',
        bindings: [
          {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read'},
          {name: 'invalidStarts', view: invalidStarts, type: 'u32', access: 'read'},
          {
            name: 'invalidCount',
            view: weights.status.invalidCount,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `invalidCount[invalidCountOffset] =
    invalidStarts[invalidStartsOffset + ${flagRows - 1}u] + invalidFlags[invalidFlagsOffset + ${flagRows - 1}u];`
      })
    );
  }

  const bandwidthWGSL =
    mode === 'radius'
      ? 'let bandwidth = readParameter(4u);'
      : // Adaptive bandwidth: the k-th neighbor distance (the row maximum) times libpysal's epsilon.
        `var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }
  bandwidth = bandwidth * ${getWGSLFloatLiteral(GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON)};`;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-weights`,
      operation: OPERATION,
      variant: 'weights',
      bindings: [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'distances', view: distances, type: 'f32', access: 'read'},
        {name: 'weights', view: weights.weights, type: 'f32', access: 'read_write'}
      ],
      invocationCount: queryRows,
      declarations: `${latticeWGSL}
${NEIGHBOR_WEIGHT_WGSL}`,
      body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${bandwidthWGSL}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let weight = getNeighborWeight(distances[distancesOffset + slot], bandwidth);
    weights[weightsOffset + slot] = weight;
    sum += weight;
  }
  if (${
    props.rowStandardize === undefined ? 'readParameter(9u) != 0.0' : String(props.rowStandardize)
  } && sum > 0.0 && isFiniteFloat(sum)) {
    for (var slot = begin; slot < end; slot++) {
      weights[weightsOffset + slot] = weights[weightsOffset + slot] / sum;
    }
  }`
    })
  );
  return nodes;
}
