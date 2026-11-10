// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getRecipeView as getOrCreateView} from './expansion-recipe-utils';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUTerrainFlow,
  type GPUTerrainFlowProps
} from '../../gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainStreamOrder} from '../../gpu-terrain/hydrology/gpu-terrain-stream-order';
import {GPUTerrainWatersheds} from '../../gpu-terrain/hydrology/gpu-terrain-watersheds';
import {GPUTerrainFlowDistance} from '../../gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../gpu-terrain/hydrology/gpu-terrain-stream-network';
import {GPUGroupStatistics} from '../../dataframe-analysis/group-statistics/index';
import {addRecipeCountNodes, RecipeBuilder} from './recipe-utils';

export type GPUDrainageNetworkRecipeProps = {
  id?: string;
  flow: Omit<GPUTerrainFlowProps, 'id' | 'output' | 'flowRouting'>;
  pourPoints?: GraphDataView<'uint32'>;
  maximumSteps?: number;
  /** Capacities default to cellCount reaches and twice cellCount vertices. */
  reachCapacity?: number;
  vertexCapacity?: number;
};
/** One complete planar D8 domain; graph-owned outputs can feed rendering or explicit readback copies. */
export function addDrainageNetworkRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUDrainageNetworkRecipeProps
) {
  const id = props.id ?? 'drainage-network-recipe';
  const builder = new RecipeBuilder(graph);
  const grid = props.flow.grid;
  const count = grid.width * grid.height;
  const flowDirections = getOrCreateView(graph, `${id}-directions`, 'uint32', count);
  const streams = getOrCreateView(graph, `${id}-streams`, 'uint32', count);
  const accumulation = getOrCreateView(graph, `${id}-accumulation`, 'float32', count);
  const flowStatus = {
    fillConverged: props.flow.fillDepressions
      ? getOrCreateView(graph, `${id}-fill-converged`, 'uint32', 1)
      : undefined,
    accumulationConverged: getOrCreateView(graph, `${id}-accumulation-converged`, 'uint32', 1),
    flatsConverged: props.flow.resolveFlats
      ? getOrCreateView(graph, `${id}-flats-converged`, 'uint32', 1)
      : undefined
  };
  builder.add(
    new GPUTerrainFlow({
      ...props.flow,
      id: `${id}-flow`,
      flowRouting: 'd8',
      output: {flowDirections, accumulation, streams, status: flowStatus}
    })
  );
  const streamOrder = getOrCreateView(graph, `${id}-order`, 'uint32', count);
  const orderStatus = {converged: getOrCreateView(graph, `${id}-order-converged`, 'uint32', 1)};
  builder.add(
    new GPUTerrainStreamOrder({
      id: `${id}-order`,
      grid,
      flowDirections,
      streams,
      output: {streamOrder, status: orderStatus}
    })
  );
  const catchments = getOrCreateView(graph, `${id}-catchments`, 'uint32', count);
  const catchmentStatus = {
    converged: getOrCreateView(graph, `${id}-catchment-converged`, 'uint32', 1)
  };
  builder.add(
    new GPUTerrainWatersheds({
      id: `${id}-catchments`,
      grid,
      flowDirections,
      pourPoints: props.pourPoints,
      output: {labels: catchments, status: catchmentStatus}
    })
  );
  const distance = {
    distance: getOrCreateView(graph, `${id}-distance`, 'float32', count),
    validity: getOrCreateView(graph, `${id}-distance-validity`, 'uint32', count),
    terminalCells: getOrCreateView(graph, `${id}-terminals`, 'uint32', count)
  };
  builder.add(
    new GPUTerrainFlowDistance({
      id: `${id}-distance`,
      grid,
      flowDirections,
      maximumSteps: props.maximumSteps,
      output: distance
    })
  );
  const unresolvedDistanceCount = getOrCreateView(graph, `${id}-distance-unresolved`, 'uint32', 1);
  addRecipeCountNodes(builder, {
    id: `${id}-distance-status`,
    operation: id,
    rowCount: count,
    bindings: [{name: 'terminals', view: distance.terminalCells, type: 'u32', access: 'read'}],
    counts: [
      {
        name: 'unresolved',
        output: unresolvedDistanceCount,
        value: 'u32(terminals[terminalsOffset + index] == 0xfffffffeu)'
      }
    ]
  });
  const reachCapacity = props.reachCapacity ?? count;
  const vertexCapacity = props.vertexCapacity ?? 2 * count;
  const network = {
    sourceCells: getOrCreateView(graph, `${id}-sources`, 'uint32', reachCapacity),
    targetCells: getOrCreateView(graph, `${id}-targets`, 'uint32', reachCapacity),
    vertexOffsets: getOrCreateView(graph, `${id}-offsets`, 'uint32', reachCapacity + 1),
    positions: getOrCreateView(graph, `${id}-positions`, 'float32x2', vertexCapacity),
    lengths: getOrCreateView(graph, `${id}-lengths`, 'float32', reachCapacity),
    reachCount: getOrCreateView(graph, `${id}-reach-count`, 'uint32', 1),
    vertexCount: getOrCreateView(graph, `${id}-vertex-count`, 'uint32', 1),
    status: {overflow: getOrCreateView(graph, `${id}-overflow`, 'uint32', 1)},
    unresolvedCount: getOrCreateView(graph, `${id}-unresolved`, 'uint32', 1)
  };
  builder.add(
    new GPUTerrainStreamNetwork({
      id: `${id}-network`,
      grid,
      flowDirections,
      streams,
      maximumSteps: props.maximumSteps,
      output: network
    })
  );
  const summary = {
    catchmentIds: getOrCreateView(graph, `${id}-summary-ids`, 'uint32', count),
    cellCounts: getOrCreateView(graph, `${id}-summary-counts`, 'uint32', count),
    meanDownstreamDistance: getOrCreateView(graph, `${id}-summary-distance`, 'float32', count),
    count: getOrCreateView(graph, `${id}-summary-count`, 'uint32', 1),
    overflow: getOrCreateView(graph, `${id}-summary-overflow`, 'uint32', 1)
  };
  builder.add(
    new GPUGroupStatistics({
      id: `${id}-summary`,
      keys: catchments,
      mask: distance.validity,
      columns: [
        {
          values: distance.distance,
          statistics: ['mean'],
          output: {means: summary.meanDownstreamDistance}
        }
      ],
      output: {
        keys: summary.catchmentIds,
        counts: summary.cellCounts,
        status: {count: summary.count, overflow: summary.overflow}
      }
    })
  );
  return {
    contributors: builder.contributors,
    outputs: {
      flowDirections,
      streams,
      accumulation,
      streamOrder,
      catchments,
      distance,
      network,
      summary
    },
    intermediates: {grid, flowStatus},
    status: {
      stages: [
        ...(flowStatus.fillConverged
          ? [{stage: 'depression-fill', status: {converged: flowStatus.fillConverged}}]
          : []),
        ...(flowStatus.flatsConverged
          ? [{stage: 'flat-routing', status: {converged: flowStatus.flatsConverged}}]
          : []),
        {stage: 'accumulation', status: {converged: flowStatus.accumulationConverged}},
        {stage: 'stream-order', status: orderStatus},
        {stage: 'catchments', status: catchmentStatus},
        {stage: 'downstream-distance', status: {uncertainCount: unresolvedDistanceCount}},
        {
          stage: 'stream-network',
          status: {
            count: network.reachCount,
            overflow: network.status.overflow,
            uncertainCount: network.unresolvedCount
          }
        },
        {stage: 'catchment-summary', status: {count: summary.count, overflow: summary.overflow}}
      ]
    }
  };
}
