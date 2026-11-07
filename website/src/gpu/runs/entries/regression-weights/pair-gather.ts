// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPairGather flattens per-query neighbor lists into join rows and gathers a right-side column.
// Matched pairs are drawn as lines from the left row to the right row the operation selected; query
// points are colored by the gathered target value. With `how: 'left'`, unmatched queries stay
// (drawn in the muted invalid color).

import {GPUPairGather} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createUniformPoints} from '../../inputs';
import {addKernelPass} from './shared/kernel';
import {parameters} from './pair-gather.parameters';

const NO_ROW = 0xffffffff;
const QUERY_COUNT = 50;
const TARGET_COUNT = 60;

const PAIR_BODY = /* wgsl */ `
  let rowCount = joinedCount[joinedCountOffset];
  let rightRow = rightRows[rightRowsOffset + index];
  if (index >= rowCount) {
    segments[segmentsOffset + index * 4u] = -10.0;
    segments[segmentsOffset + index * 4u + 1u] = -10.0;
    segments[segmentsOffset + index * 4u + 2u] = -10.0;
    segments[segmentsOffset + index * 4u + 3u] = -10.0;
    queryDots[queryDotsOffset + index * 2u] = -10.0;
    queryDots[queryDotsOffset + index * 2u + 1u] = -10.0;
    return;
  }
  let leftRow = leftRows[leftRowsOffset + index];
  let start = vec2f(queries[queriesOffset + leftRow * 2u], queries[queriesOffset + leftRow * 2u + 1u]);
  queryDots[queryDotsOffset + index * 2u] = start.x;
  queryDots[queryDotsOffset + index * 2u + 1u] = start.y;
  var finish = start;
  if (rightRow != ${NO_ROW}u) {
    finish = vec2f(targets[targetsOffset + rightRow * 2u], targets[targetsOffset + rightRow * 2u + 1u]);
  }
  segments[segmentsOffset + index * 4u] = start.x;
  segments[segmentsOffset + index * 4u + 1u] = start.y;
  segments[segmentsOffset + index * 4u + 2u] = finish.x;
  segments[segmentsOffset + index * 4u + 3u] = finish.y;
`;

export default defineOperationRun({
  parameters,
  caption: 'Join rows as lines from each left row to the right row it was paired with.',
  createInputs: ({random, world}) => ({
    queries: createUniformPoints(random, world, QUERY_COUNT, 0.1),
    targets: createUniformPoints(random, world, TARGET_COUNT, 0.3)
  }),
  build(builder, {queries, targets}, values) {
    const capacity = values.capacity;
    const queryPositions = builder.upload('queries', 'float32x2', queries.positions);
    const targetPositions = builder.upload('targets', 'float32x2', targets.positions);
    const targetValues = builder.upload('target-values', 'float32', targets.values);
    const ids = builder.allocate('neighbor-ids', 'uint32', QUERY_COUNT * capacity);
    const counts = builder.allocate('neighbor-counts', 'uint32', QUERY_COUNT);
    const rowCapacity = QUERY_COUNT * capacity;
    const leftRows = builder.allocate('left-rows', 'uint32', rowCapacity);
    const rightRows = builder.allocate('right-rows', 'uint32', rowCapacity);
    const joinedCount = builder.allocate('row-count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const gathered = builder.allocate('gathered-value', 'float32', rowCapacity);
    builder.graph.add(
      new GPUPairGather({
        how: values.how === 0 ? 'inner' : 'left',
        neighbors: {ids: ids.view, counts: counts.view, capacity},
        rightColumns: [{source: targetValues.view, output: gathered.view}],
        output: {
          leftRows: leftRows.view,
          rightRows: rightRows.view,
          rowCount: joinedCount.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const segments = builder.allocate('segments', 'float32x4', rowCapacity);
    const queryDots = builder.allocate('query-dots', 'float32x2', rowCapacity);
    addKernelPass(builder, {
      id: 'join-pairs',
      invocationCount: rowCapacity,
      bindings: [
        {name: 'joinedCount', buffer: joinedCount, access: 'read'},
        {name: 'leftRows', buffer: leftRows, access: 'read'},
        {name: 'rightRows', buffer: rightRows, access: 'read'},
        {name: 'queries', buffer: queryPositions, access: 'read'},
        {name: 'targets', buffer: targetPositions, access: 'read'},
        {name: 'segments', buffer: segments, access: 'read_write'},
        {name: 'queryDots', buffer: queryDots, access: 'read_write'}
      ],
      body: PAIR_BODY
    });
    return {
      outputs: {queryPositions, targetPositions, targetValues, segments, queryDots, gathered},
      // The nearest targets of every query inside the radius, as dense per-query lists.
      update: current => {
        const denseIds = new Uint32Array(rowCapacity).fill(NO_ROW);
        const denseCounts = new Uint32Array(QUERY_COUNT);
        for (let query = 0; query < QUERY_COUNT; query++) {
          const candidates: [number, number][] = [];
          for (let target = 0; target < TARGET_COUNT; target++) {
            const distance = Math.hypot(
              queries.positions[query * 2] - targets.positions[target * 2],
              queries.positions[query * 2 + 1] - targets.positions[target * 2 + 1]
            );
            if (distance <= current.radius) candidates.push([distance, target]);
          }
          candidates.sort((first, second) => first[0] - second[0]);
          candidates.slice(0, capacity).forEach(([, target], slot) => {
            denseIds[query * capacity + slot] = target;
          });
          denseCounts[query] = Math.min(candidates.length, capacity);
        }
        ids.write(denseIds);
        counts.write(denseCounts);
      }
    };
  },
  rebuildOn: ['how', 'capacity'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.targetPositions,
      values: outputs.targetValues,
      radius: 3
    },
    {layer: 'lines', segments: outputs.segments, color: [1, 0.55, 0.2, 0.85], radius: 1.4},
    {
      layer: 'dots',
      positions: outputs.queryDots,
      values: outputs.gathered,
      radius: 6,
      range: [0, 1]
    }
  ]
});
