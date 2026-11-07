// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OperatorDescriptor} from '../../computation-views/types';
import {NETWORK_POINT_PROCESS_OPERATORS} from './network-point-process-operators';
import {NETWORK_OPTIMIZATION_OPERATORS} from './network-optimization-metadata';

/** Scope and reduction of all public network elements, from their contributor contracts. */
export const NETWORK_OPERATORS: Record<string, OperatorDescriptor> = {
  ...NETWORK_OPTIMIZATION_OPERATORS,
  ...NETWORK_POINT_PROCESS_OPERATORS,
  GPUNetworkReachability: {scope: 'frontier', reduction: 'argmin'},
  GPUNetworkServiceAreas: {scope: 'frontier', scopeSecondary: 'zonal', reduction: 'argmin'},
  GPUNetworkPathExtraction: {scope: 'frontier', reduction: 'select'},
  GPUNetworkNeighborhood: {scope: 'frontier', reduction: 'select'},
  GPUNetworkIsochrones: {scope: 'frontier', scopeSecondary: 'local', reduction: 'argmin'},
  addDriveTimeCatchmentRecipe: {scope: 'frontier', scopeSecondary: 'zonal', reduction: 'argmin'},
  GPUNetworkCostTable: {scope: 'frontier', scopeSecondary: 'pairwise', reduction: 'argmin'},
  GPUNetworkSnapping: {scope: 'pairwise', reduction: 'argmin'},
  GPUNetworkCostMatrix: {scope: 'frontier', scopeSecondary: 'pairwise', reduction: 'argmin'},
  GPUNetworkAccessibility: {scope: 'pairwise', reduction: 'weighted-mean'},
  GPUNetworkCostWeights: {scope: 'pairwise', reduction: 'weighted-mean'},
  GPUNetworkCentrality: {scope: 'frontier', reduction: 'sum'},
  GPUNetworkAnalyticsColumns: {scope: 'focal', scopeSecondary: 'global', reduction: 'none'},
  GPUNetworkStatistics: {scope: 'global', reduction: 'histogram'},
  GPUNetworkSubgraphFilter: {scope: 'local', reduction: 'select'},
  GPUNetworkCoarsening: {scope: 'zonal', reduction: 'sum'},
  GPUAdjacencyMatrix: {scope: 'pairwise', reduction: 'sum'},
  GPUAdjacencyMatrixOrder: {scope: 'global', reduction: 'scan-sort'},
  GPUNetworkNoding: {scope: 'pairwise', reduction: 'select'},
  GPUNetworkLineGraph: {scope: 'focal', reduction: 'none'},
  GPUMapMatching: {scope: 'solver', reduction: 'argmin'},
  GPUNetworkKFunction: {scope: 'frontier', scopeSecondary: 'pairwise', reduction: 'count'},
  GPUEdgeBundling: {scope: 'focal', reduction: 'weighted-mean'},
  GPUNetworkStructure: {scope: 'global', reduction: 'select'},
  GPUAlternativeRoutes: {scope: 'solver', reduction: 'argmin'},
  addNetworkDisruptionRecipe: {scope: 'frontier', scopeSecondary: 'pairwise', reduction: 'argmin'},
  addTrackQualityRecipe: {scope: 'local', reduction: 'none'},
  addFacilityScenarioRecipe: {scope: 'pairwise', reduction: 'sum'},
  GPUFlowAggregation: {scope: 'zonal', reduction: 'sum'}
};
