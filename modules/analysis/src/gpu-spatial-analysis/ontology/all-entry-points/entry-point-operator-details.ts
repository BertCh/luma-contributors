// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisPackage, GPUSpatialAnalysisOperator} from '../spatial-analysis-ontology';

/** What cannot be inherited from the capability family for an operator of another entry point. */
export type EntryPointOperatorDetail = {
  package: GPUAnalysisPackage;
  /** First sentence of the class or function documentation. */
  summary: string;
  /** Repository-relative spec that exercises the operator. */
  tests: string;
  /** Repository-relative documentation page. */
  documentation: string;
  dimensionality?: GPUSpatialAnalysisOperator['dimensionality'];
};

const PACKAGE = '@luma-contributors/analysis';

/** Operators of the entry points other than `gpu-spatial-analysis`, by export name. */
export const ENTRY_POINT_OPERATOR_DETAILS: Readonly<Record<string, EntryPointOperatorDetail>> = {
  GPUForestInference: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Predicts with imported scikit-learn, XGBoost and LightGBM numerical forests, preserving original leaf ids and missing-value routing.',
    tests: 'modules/analysis/test/gpu-dataframe/forest-inference/gpu-forest-inference.spec.ts',
    documentation: 'docs/api/forest-inference.md'
  },
  GPUComputationRecorder: {
    package: `${PACKAGE}/computation-record`,
    summary: 'Analysis-side record of one command graph under construction.',
    tests: 'modules/analysis/test/computation-record/computation-recorder.node.spec.ts',
    documentation: 'docs/api/computation-record.md'
  },
  GPUSnapshotArena: {
    package: `${PACKAGE}/computation-record`,
    summary: 'Copies frames into one bounded readback buffer, then maps that buffer once.',
    tests: 'modules/analysis/test/computation-record/gpu-snapshot-arena.node.spec.ts',
    documentation: 'docs/api/computation-record.md'
  },
  GPUSnapshotTap: {
    package: `${PACKAGE}/computation-record`,
    summary:
      'A copy node that snapshots views of a graph into a GPUSnapshotArena at a chosen point of the authored sequence.',
    tests: 'modules/analysis/test/computation-record/computation-record-capture.spec.ts',
    documentation: 'docs/api/computation-record.md'
  },
  GPUTransferMask: {
    package: `${PACKAGE}/computation-record`,
    summary: 'Clears and scatters a selection mask across a GPU-resident relation.',
    tests: 'modules/analysis/test/computation-record/gpu-transfer-mask.node.spec.ts',
    documentation: 'docs/api/computation-record.md'
  },
  GPUNetworkNoding: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Builds a routable planar network from linestrings.',
    tests: 'modules/analysis/test/gpu-network/network-spatial-context.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkSnapping: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Snaps points onto the nearest edge of a road network, so they can enter a shortest-path search as virtual nodes between the edge endpoints.',
    tests:
      'modules/analysis/test/gpu-network/network-accessibility/gpu-network-accessibility.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkLineGraph: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Builds the turn-restriction line graph (edge-based graph) of a directed road CSR.',
    tests:
      'modules/analysis/test/gpu-network/network-line-graph/gpu-network-line-graph.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkSubgraphFilter: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Turns attribute and time predicates on node and edge rows into a consistent induced subgraph.',
    tests:
      'modules/analysis/test/gpu-network/network-subgraph-filter/gpu-network-subgraph-filter.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkCoarsening: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Coarsens a CSR network by a node group label into a summary graph for zoomed-out compound nodes: per-group node counts, centroids, bounds.',
    tests: 'modules/analysis/test/gpu-network/network-coarsening/gpu-network-coarsening.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkReachability: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Computes single- or multi-source shortest-path costs over a directed CSR road network, with isochrone bands, shortest-path predecessors, and a GPU convergence flag.',
    tests:
      'modules/analysis/test/gpu-network/network-reachability/gpu-network-reachability.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkPathExtraction: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Extracts ordered source-to-target node and edge lists from a shortest-path predecessor array on the GPU, ready for a path layer or highlight.',
    tests:
      'modules/analysis/test/gpu-network/network-analysis/gpu-network-path-extraction.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUKShortestPaths: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Yen k-shortest simple edge paths on CSR through parallel GPU shortest-path spur searches, with path, candidate and iteration budgets (Yen 1971).',
    tests: 'modules/analysis/test/gpu-network/k-shortest-paths/gpu-k-shortest-paths.spec.ts',
    documentation: 'docs/api/gpu-network/k-shortest-paths.md',
    dimensionality: 'topology'
  },
  GPUAlternativeRoutes: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Bounded uniform-cost search over simple edge paths, independent queries on the GPU.',
    tests: 'modules/analysis/test/gpu-network/alternative-routes/gpu-alternative-routes.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    dimensionality: 'topology'
  },
  GPUNetworkPointRouting: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Batched static routing between directed edge positions, with explicit incoming turn state.',
    tests:
      'modules/analysis/test/gpu-network/network-optimization/gpu-network-optimization.spec.ts',
    documentation: 'docs/api/gpu-network/network-movement-coverage.md',
    dimensionality: 'topology'
  },
  GPUNetworkTimeRouting: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'FIFO earliest-arrival routing on caller-prepared restriction states.',
    tests:
      'modules/analysis/test/gpu-network/network-optimization/network-movement-advanced.spec.ts',
    documentation: 'docs/api/gpu-network/network-movement-coverage.md',
    dimensionality: '2d+time'
  },
  addNetworkDisruptionRecipe: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Compare directed routing scenarios without compaction or Infinity - Infinity arithmetic.',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/recipes/backlog-recipe-boundaries.node.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    dimensionality: 'topology'
  },
  GPUNetworkCostMatrix: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      "Computes a bounded many-to-all cost matrix: one shortest-path search per matrix row, from that row's starts to every node of a directed CSR network.",
    tests:
      'modules/analysis/test/gpu-network/network-accessibility/gpu-network-accessibility.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkCostTable: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Precomputes a bounded all-pairs cost table of a static directed network, in the spirit of the UBODT of fast map matching (Yang and Gidofalvi 2018): for every node `u`.',
    tests:
      'modules/analysis/test/gpu-network/network-cost-table/gpu-network-cost-table.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkAccessibility: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Scores per-origin accessibility from a retained network cost matrix: cumulative opportunities within a per-frame cost threshold.',
    tests:
      'modules/analysis/test/gpu-network/network-accessibility/gpu-network-accessibility.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkCostWeights: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Turns a network cost matrix or reachability result into GPUSpatialWeights with a distance decay, so any weights consumer runs on network costs.',
    tests:
      'modules/analysis/test/gpu-network/network-cost-weights/gpu-network-cost-weights.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkServiceAreas: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Assigns every node of a directed CSR road network to its nearest facility ("Voronoi on the network", the evaluation step of location-allocation).',
    tests:
      'modules/analysis/test/gpu-network/network-analysis/gpu-network-service-areas.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkIsochrones: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Network isochrone polygons: edge-interpolated travel costs splatted to a raster and contoured into bands, or the outline of the cells that contain reached nodes.',
    tests:
      'modules/analysis/test/gpu-network/network-isochrones/gpu-network-isochrones.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkRationalAccess: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      "Bounded integral/fractional rational-agent allocation with RAAM's travel/supply normalization, pairwise congestion equalization and decaying demand-shift cap.",
    tests:
      'modules/analysis/test/gpu-network/network-optimization/network-movement-advanced.spec.ts',
    documentation: 'docs/api/gpu-network/network-movement-coverage.md',
    dimensionality: 'topology'
  },
  addDriveTimeCatchmentRecipe: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Drive-time catchment recipe: which demand is within each drive-time band of the facilities.',
    tests: 'modules/analysis/test/gpu-network/recipes/drive-time-catchment-recipe.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkStronglyConnectedComponents: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Exact minimum-ID directed SCCs with linear scratch, GPU trim/colour/backward marking and a fail-closed step budget; CPU reference: computeNetworkStructureOnCPU (Tarjan).',
    tests:
      'modules/analysis/test/gpu-network/network-structure/network-strongly-connected-components.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    dimensionality: 'topology'
  },
  GPUNetworkStructure: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Exact bounded graph structure on GPU-resident CSR.',
    tests:
      'modules/analysis/test/gpu-network/network-structure/network-structure-capacity.node.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    dimensionality: 'topology'
  },
  GPUNetworkStatistics: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Reduces a CSR network to a compact statistics summary for a graph stats panel: live node/edge/slot counts, weak component count and largest size, isolated nodes.',
    tests: 'modules/analysis/test/gpu-network/network-statistics/gpu-network-statistics.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkAnalyticsColumns: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Publishes node-aligned `float32` and `uint32` analytics columns for a CSR road network, ready for deck.gl styling: degree, in-degree and weak component labels.',
    tests:
      'modules/analysis/test/gpu-network/network-analysis/gpu-network-analytics-columns.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkNeighborhood: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Computes the k-hop ego network around per-frame start nodes: per-node hop distance, a node mask, a mask of induced edges.',
    tests:
      'modules/analysis/test/gpu-network/network-analysis/gpu-network-neighborhood.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkCentrality: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Computes radius-bounded betweenness and closeness centrality of a directed weighted CSR network from a set of pivot nodes, exactly or from a sample.',
    tests: 'modules/analysis/test/gpu-network/network-centrality/network-centrality.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkLouvain: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Weighted directed/undirected multilevel Louvain with sparse parallel CSR moving, conflict resolution and quotient aggregation; legacy dense reference retained.',
    tests: 'modules/analysis/test/gpu-network/network-optimization/network-louvain-sparse.spec.ts',
    documentation: 'modules/analysis/docs/gpu-network-louvain.md',
    dimensionality: 'topology'
  },
  GPUNetworkFlow: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Successive shortest augmenting paths with explicit reverse arcs.',
    tests:
      'modules/analysis/test/gpu-network/network-optimization/gpu-network-optimization.spec.ts',
    documentation: 'docs/api/gpu-network/network-movement-coverage.md',
    dimensionality: 'topology'
  },
  GPUNetworkFacilityOptimization: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Exhaustive facility candidates evaluated independently in fixed-size GPU batches.',
    tests:
      'modules/analysis/test/gpu-network/network-optimization/gpu-network-optimization.spec.ts',
    documentation: 'docs/api/gpu-network/network-movement-coverage.md',
    dimensionality: 'topology'
  },
  addFacilityScenarioRecipe: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Evaluate one live facility scenario against a reusable matrix.',
    tests: 'modules/analysis/test/geospatial-expansion/recipe-review.node.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    dimensionality: 'topology'
  },
  GPUNetworkKFunction: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Network-constrained Ripley K function (Okabe and Yamada; spaghetti `GlobalAutoK`) of events on a road network.',
    tests:
      'modules/analysis/test/gpu-network/network-k-function/gpu-network-k-function.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUNetworkInhomogeneousK: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Inhomogeneous network K = sum_{i != j,dij <= r} e(i,j)/(lambda_i lambda_j) / denominator.',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/point-process/network-inhomogeneous-k.node.spec.ts',
    documentation: 'docs/api/gpu-network/point-process.md',
    dimensionality: 'topology'
  },
  GPUNetworkKernelDensity: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      "Triangular shortest-path network KDE corrected by each event's total kernel mass over all branches.",
    tests: 'modules/analysis/test/gpu-spatial-analysis/point-process/point-process.node.spec.ts',
    documentation: 'docs/api/gpu-network/point-process.md',
    dimensionality: 'topology'
  },
  GPUMapMatching: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Hidden Markov model map matching (Newson and Krumm 2009) of GPS tracks onto a road network.',
    tests: 'modules/analysis/test/gpu-network/map-matching/gpu-map-matching.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: '2d+time'
  },
  GPUMapMatchedEdgeVolumes: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Traversal counts per network edge, and the busiest edge-to-edge movements, from map-matched traversed routes.',
    tests:
      'modules/analysis/test/gpu-network/flow-aggregation/gpu-map-matched-edge-volumes.spec.ts',
    documentation: 'docs/api/gpu-network/map-matched-edge-volumes-mob05.md',
    dimensionality: '2d+time'
  },
  GPUNetworkPotentialPathArea: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Network potential path area of a space-time prism from a forward and a reverse shortest-path tree.',
    tests:
      'modules/analysis/test/gpu-network/network-time-geography/gpu-network-time-geography.spec.ts',
    documentation: 'docs/api/gpu-network/network-time-geography-mob09.md',
    dimensionality: '2d+time'
  },
  GPUNetworkTravelTimePercentiles: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Travel-time percentiles over a departure window on FIFO time-dependent routing (r5r time_window analogue).',
    tests:
      'modules/analysis/test/gpu-network/network-time-geography/gpu-network-time-geography.spec.ts',
    documentation: 'docs/api/gpu-network/network-time-geography-mob09.md',
    dimensionality: '2d+time'
  },
  GPUFlowAggregation: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Aggregates origin-destination movements into a weight-sorted top-K flow list and per-zone totals.',
    tests: 'modules/analysis/test/gpu-network/flow-aggregation/gpu-flow-aggregation.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPURadiationFlowModel: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Radiation model of human mobility (Simini et al. 2012): the expected flow between zones from populations and distances, with no fitted parameter.',
    tests: 'modules/analysis/test/gpu-network/flow-aggregation/flow-models-mob10.spec.ts',
    documentation: 'docs/api/gpu-network/flow-models-mob10.md',
    dimensionality: 'topology'
  },
  GPUKernelFlowSmoothing: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Kernel smoothing of origin-destination flows over neighbouring origins and destinations (after Guo and Zhu 2014), from a matrix or the pairs of GPUFlowAggregation.',
    tests: 'modules/analysis/test/gpu-network/flow-aggregation/flow-models-mob10.spec.ts',
    documentation: 'docs/api/gpu-network/flow-models-mob10.md',
    dimensionality: 'topology'
  },
  addTrackQualityRecipe: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'Raw quality flags, optional route distances and optional map-matching completeness.',
    tests: 'modules/analysis/test/geospatial-expansion/recipe-review.spec.ts',
    documentation: 'docs/api/gpu-network/expansion.md',
    dimensionality: '2d+time'
  },
  GPUAdjacencyMatrix: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Bins a graph into an `R x R` adjacency-matrix image on the GPU, for a matrix view linked to a node-link view.',
    tests: 'modules/analysis/test/gpu-network/adjacency-matrix/gpu-adjacency-matrix.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUAdjacencyMatrixOrder: {
    package: `${PACKAGE}/gpu-network`,
    summary: 'GPU counterpart of computeAdjacencyMatrixOrder.',
    tests: 'modules/analysis/test/gpu-network/adjacency-matrix/gpu-adjacency-matrix.node.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUEdgeBundling: {
    package: `${PACKAGE}/gpu-network`,
    summary:
      'Kernel-density edge bundling (KDEEB) as WebGPU compute, producing renderable polylines.',
    tests: 'modules/analysis/test/gpu-network/edge-bundling/gpu-edge-bundling.spec.ts',
    documentation: 'docs/api/gpu-network/operators.md',
    dimensionality: 'topology'
  },
  GPUTerrainFlow: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Terrain hydrology on one raster tile: optional depression filling, D8 flow directions, flow accumulation, and a stream mask.',
    tests: 'modules/analysis/test/gpu-terrain/hydrology/gpu-terrain-flow.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainFlowFractions: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Materializes the existing D-infinity/MFD routing kernel for branching consumers, without duplicating routing mathematics.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.spec.ts',
    documentation: 'docs/api/gpu-terrain/terrain-processing.md',
    dimensionality: '2d'
  },
  GPUTerrainBranchingFlow: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Deterministic DAG pulls with atomic value/readiness publication and GPU-gated successor walks.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.node.spec.ts',
    documentation: 'docs/api/gpu-terrain/terrain-processing.md',
    dimensionality: '2d'
  },
  GPUTerrainTiledFlow: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Owned-cell GPU exchange into a resident global DAG, with atomic transport and pointer-jumped reach identity.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.spec.ts',
    documentation: 'docs/api/gpu-terrain/terrain-processing.md',
    dimensionality: '2d'
  },
  GPUTerrainFlowDistance: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Downstream D8 path length, preserving invalid directions and bounded-search failures.',
    tests: 'modules/analysis/test/gpu-terrain/hydrology/terrain-expansion-validation.spec.ts',
    documentation: 'docs/api/gpu-terrain/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPUTerrainWatersheds: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Watershed delineation on one raster tile from D8 flow directions.',
    tests: 'modules/analysis/test/gpu-terrain/hydrology/parity-wp5-watersheds.parity.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainStreamOrder: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Strahler stream order (Strahler 1957) on one raster tile.',
    tests: 'modules/analysis/test/gpu-terrain/hydrology/gpu-terrain-drainage.node.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainStreamNetwork: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Extracts compact directed reaches between headwaters, confluences and terminals.',
    tests: 'modules/analysis/test/gpu-terrain/hydrology/terrain-expansion-validation.spec.ts',
    documentation: 'docs/api/gpu-terrain/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPUTerrainHeightAboveDrainage: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Height above drainage over D8 or D-infinity fractions (TauDEM vertical average).',
    tests: 'modules/analysis/test/gpu-terrain/hydrology/terrain-dinf-hand.spec.ts',
    documentation: 'modules/analysis/docs/terrain-dinf-hand.md',
    dimensionality: '2d'
  },
  GPUTerrainHydrologicIndices: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Flow-accumulation-based terrain indices on one raster tile: specific catchment area.',
    tests:
      'modules/analysis/test/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices.node.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainCatchmentSummary: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Composes sorted basin reductions and pointer-jumped stream profiles.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.spec.ts',
    documentation: 'docs/api/gpu-terrain/terrain-processing.md',
    dimensionality: '2d'
  },
  GPUTerrainBreach: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Bounded least total excavation-depth path on a step-layered D8 graph, with deterministic simultaneous minimum-elevation edits.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.node.spec.ts',
    documentation: 'docs/api/gpu-terrain/terrain-processing.md',
    dimensionality: '2d'
  },
  GPUTerrainDepressionHierarchy: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Stable GPU-sorted sublevel-set merge tree with compressed component roots and linear stage reductions.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.node.spec.ts',
    documentation: 'docs/api/gpu-terrain/terrain-processing.md',
    dimensionality: '2d'
  },
  GPUTerrainSpikeRepair: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Repairs +/-1 errors in the high byte of a Terrarium-style elevation encoding, an opt-in port of mt-image `validateTile` without its out-of-range fill.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-decode/gpu-terrain-spike-repair.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainRGBDecode: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Decodes RGB-encoded elevation (Terrarium or Mapbox terrain-RGB) into float32 heights plus uint32 validity.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-decode/terrain-rgb-decode.node.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainDEMCoregistration: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: "Robust vertical-shift model, matching xDEM's median VerticalShift objective.",
    tests: 'modules/analysis/test/gpu-terrain/terrain-processing/terrain-processing.spec.ts',
    documentation: 'docs/api/gpu-terrain/terrain-processing.md',
    dimensionality: '2d'
  },
  GPUTerrainDerivatives: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: "Computes slope, aspect, and hillshade for an elevation tile with Horn's 3x3 method.",
    tests: 'modules/analysis/test/gpu-terrain/terrain-analysis/gpu-terrain-derivatives.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainCurvature: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Computes second-order terrain curvatures (profile, plan, tangential, mean, Gaussian and others) from an elevation raster.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-curvature/terrain-curvature.node.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainRuggedness: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Computes gdaldem-compatible topographic position index (TPI), terrain ruggedness index (TRI), and roughness over a 3x3 window in one fused kernel.',
    tests: 'modules/analysis/test/gpu-terrain/topographic-position/gpu-terrain-ruggedness.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainVectorRuggedness: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Computes the vector ruggedness measure (VRM; Sappington, Longshore and Thompson 2007).',
    tests:
      'modules/analysis/test/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainContours: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Extracts marching-squares contour segments for several elevation levels into bounded buffers.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-analysis/gpu-terrain-contours.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainTopographicPosition: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Multiscale topographic position: TPI over square annuli, deviation from mean elevation (DEV), and the maximum-deviation scale (DEVmax, Lindsay, Cockburn & Russell 2015).',
    tests:
      'modules/analysis/test/gpu-terrain/topographic-position/gpu-terrain-topographic-position.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainWeissLandforms: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Weiss (2001) topographic-position landform classification into ten classes from TPI at a small and a large scale plus slope.',
    tests: 'modules/analysis/test/gpu-terrain/parity-wp5-round8-composites.parity.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUGeomorphons: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Classifies each cell into one of ten landforms with the geomorphon method of Jasiewicz and Stepinski (2013), using the GRASS `r.geomorphon` angle-comparison variants.',
    tests: 'modules/analysis/test/gpu-terrain/geomorphons/gpu-geomorphons.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainCriticalPoints: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Classifies every pixel as regular, peak, pit, saddle, boundary or nodata by counting sign changes of the height difference around its ring.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-features/gpu-terrain-critical-points.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainSummits: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Finds DEM summits: pixels that are the highest point of a metric disc and stand out from its boundary ring.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-features/gpu-terrain-summits.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainPeakSnap: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Snaps caller-supplied candidate points (for example catalogue peaks) to the DEM summit near them.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-features/gpu-terrain-peak-snap.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUProfilePeaks: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Local maxima of many 1-D profiles with windowed topographic prominence, parabolic sub-sample refinement and non-maximum suppression.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-features/gpu-profile-peaks.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainViewshed: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Computes a line-of-sight viewshed from one observer over an elevation tile.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-analysis/gpu-terrain-viewshed.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainLineOfSight: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Tests explicit observer-target pairs against an elevation tile.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainCumulativeViewshed: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Counts, for every pixel, how many observers see it (the analogue of GDAL `gdal_viewshed` cumulative mode).',
    tests: 'modules/analysis/test/gpu-terrain/parity-wp5-round8-visibility.parity.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUPointHorizonProfile: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Computes the skyline (horizon) profile seen from many observers over one terrain window.',
    tests: 'modules/analysis/test/gpu-terrain/point-horizon/gpu-point-horizon-profile.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUPointHorizonVisibility: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      "Classifies targets (peaks, towers, points of interest) as hidden, marginal or visible from observers, and flags those on the skyline, following mt-image's `classifyPeak`.",
    tests: 'modules/analysis/test/gpu-terrain/parity-wp5-round8-visibility.parity.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUSolarPosition: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Computes the sun azimuth and altitude for every row of a longitude/latitude column at one per-frame instant, with the NOAA solar calculator formulas evaluated in float32.',
    tests:
      'modules/analysis/test/gpu-terrain/terrain-illumination/parity-wp5-solar-position.parity.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainHorizon: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Computes horizon-angle maps for `directionCount` azimuth sectors by bounded ray marching, plus sky-view factor.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-illumination/gpu-terrain-horizon.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainCastShadow: {
    package: `${PACKAGE}/gpu-terrain`,
    summary: 'Single-sun terrain cast shadow without a horizon map.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUSolarShadowMask: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Turns a precomputed horizon map and a per-frame sun into soft or hard terrain shadows.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUSolarIrradiance: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Accumulates sun-hours and insolation over a day or date range from a precomputed horizon map.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-illumination/gpu-solar-irradiance.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d+time'
  },
  GPUSimpleLocalRelief: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Simple local relief model (SLRM): the elevation minus its local mean, which removes the large-scale trend and leaves small features such as banks, ditches and mounds.',
    tests: 'modules/analysis/test/gpu-terrain/relief-visualization/gpu-simple-local-relief.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUMultiScaleRelief: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Multi-scale relief model (MSRM): the mean of the differences between consecutive low-pass filtered surfaces.',
    tests: 'modules/analysis/test/gpu-terrain/relief-visualization/gpu-multi-scale-relief.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPULocalDominance: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Local dominance: how much an observer standing on the surface at a fixed height rises above the terrain in a ring of distances, which brings out mounds, banks and pits.',
    tests: 'modules/analysis/test/gpu-terrain/relief-visualization/gpu-local-dominance.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUReliefBlend: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Blends up to five single-band visualizations into one greyscale image, the Visualization for Archaeological Topography (VAT) workflow of Kokalj and Somrak (2019).',
    tests:
      'modules/analysis/test/gpu-terrain/relief-visualization/parity-wp5-relief-blend.parity.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUReliefShading: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Computes a multidirectional hillshade and a Swiss/Imhof-style cartographic relief blend.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-illumination/gpu-relief-shading.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTextureShading: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      "Leland Brown's texture shading, a fractional Laplacian of order `alpha` (`|f|^alpha` in the frequency domain).",
    tests: 'modules/analysis/test/gpu-terrain/terrain-illumination/gpu-texture-shading.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPUTerrainFlowField: {
    package: `${PACKAGE}/gpu-terrain`,
    summary:
      'Terrain-following wind: a uniform horizontal wind deflected by the DEM so it flows around and over relief, ready for the particle-advection and flow-texture contributors.',
    tests: 'modules/analysis/test/gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.spec.ts',
    documentation: 'docs/api/gpu-terrain/operators.md',
    dimensionality: '2d'
  },
  GPURasterArithmetic: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'A small per-cell raster calculator: `op(a * scaleA + offsetA, b * scaleB + offsetB)` then an optional clamp, with the operation chosen per frame.',
    tests: 'modules/analysis/test/gpu-raster/raster-algebra/gpu-raster-local-operations.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterConditional: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Per-cell `where(condition, a, b)` (ArcGIS `Con`, numpy `where`).',
    tests: 'modules/analysis/test/gpu-raster/raster-algebra/gpu-raster-local-operations.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterReclassify: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Reclassifies a float32 raster or column through a per-frame break table.',
    tests: 'modules/analysis/test/gpu-raster/raster-algebra/raster-algebra.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUWeightedOverlay: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Weighted overlay (suitability analysis) of up to 16 float32 rasters.',
    tests: 'modules/analysis/test/gpu-raster/raster-algebra/raster-algebra.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterCellStatistics: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Local (per-cell) statistics across a stack of rasters, like ArcGIS "Cell Statistics" or GDAL `gdal_calc` reductions: minimum, maximum, range, sum, mean.',
    tests:
      'modules/analysis/test/gpu-raster/raster-algebra/parity-wp4-cell-statistics-modes.parity.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterContrastEqualization: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'CLAHE with reflected edge tiles and bilinear tile mappings, or quantized CDF histogram matching.',
    tests:
      'modules/analysis/test/gpu-raster/raster-contrast-equalization/gpu-raster-contrast-equalization.spec.ts',
    documentation: 'docs/api/gpu-raster/contrast-equalization.md',
    dimensionality: '2d'
  },
  GPURasterStretch: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Raster contrast stretch and colormap helper: linear, percentile and histogram-equalization stretches of a float32 raster.',
    tests: 'modules/analysis/test/gpu-raster/raster-stretch/raster-stretch.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterNDWI: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Calibrated `(green - NIR)/(green + NIR)` or `(NIR - SWIR)/(NIR + SWIR)`.',
    tests: 'modules/analysis/test/gpu-raster/gpu-raster-spectral-indices.node.spec.ts',
    documentation: 'docs/api/gpu-raster/spectral-indices.md',
    dimensionality: '2d'
  },
  GPURasterNBR: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Calibrated `(NIR - SWIR2)/(NIR + SWIR2)`; outputs the ratio, not a scaled integer product.',
    tests: 'modules/analysis/test/gpu-raster/gpu-raster-spectral-indices.node.spec.ts',
    documentation: 'docs/api/gpu-raster/spectral-indices.md',
    dimensionality: '2d'
  },
  GPURasterEVI: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Enhanced vegetation index `2.5 * (NIR - red)/(NIR + 6 * red - 7.5 * blue + 1)`.',
    tests: 'modules/analysis/test/gpu-raster/gpu-raster-spectral-indices.node.spec.ts',
    documentation: 'docs/api/gpu-raster/spectral-indices.md',
    dimensionality: '2d'
  },
  GPURasterSAVI: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Soil-adjusted vegetation index `(1 + L) * (NIR - red)/(NIR + red + L)`.',
    tests: 'modules/analysis/test/analysis-core/kernel-audit-coverage-tranche-one.node.spec.ts',
    documentation: 'docs/api/gpu-raster/spectral-indices.md',
    dimensionality: '2d'
  },
  GPURasterSpectralUnmixing: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Small QR least squares (up to 256 bands/8 endmembers), or bounded projected-gradient NNLS.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/remote-sensing.node.spec.ts',
    documentation: 'docs/api/gpu-raster/remote-sensing.md',
    dimensionality: '2d'
  },
  GPURasterFocalPairs: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Unweighted focal covariance, Pearson correlation and simple OLS with an intercept.',
    tests: 'modules/analysis/test/gpu-raster/focal-pairs/raster-focal-pairs.node.spec.ts',
    documentation: 'docs/api/gpu-raster/focal-pairs.md',
    dimensionality: '2d'
  },
  GPURasterRankFilter: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Raster-native rank selection.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterTexture: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Local normalized GLCM; natural-log entropy, constant-window correlation 1, empty features NaN.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/remote-sensing.node.spec.ts',
    documentation: 'docs/api/gpu-raster/remote-sensing.md',
    dimensionality: '2d'
  },
  GPURasterFillNoData: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Bounded, deterministic nodata interpolation on the GPU.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-fill-four-quadrant.node.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterReconstruction: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Geodesic reconstruction under an explicit mask, using synchronous bounded GPU rounds.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterFillHoles: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Fills background components not connected to the boundary, using the selected background connectivity.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterSkeleton: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Zhang–Suen two-subiteration thinning.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterMedialAxis: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Discrete maximal-inscribed-disk centers and their Euclidean radii in pixel units.',
    tests: 'modules/analysis/test/gpu-raster/medial-axis/gpu-raster-medial-axis.node.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterWatershedSegmentation: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Marker-controlled segmentation with explicit FIFO or rooted minimum-spanning-forest semantics.',
    tests: 'modules/analysis/test/geospatial-expansion/watershed-forest.node.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterRegionGraph: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Four-neighbor region adjacency with bounded fixed or adaptive mean-difference merging.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/remote-sensing.node.spec.ts',
    documentation: 'docs/api/gpu-raster/remote-sensing.md',
    dimensionality: '2d'
  },
  GPURasterPatchMetrics: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Per-patch metrics of a dense label raster: area, perimeter and bounding box (the `r.clump` / FRAGSTATS companions to `GPURasterConnectedComponents`).',
    tests: 'modules/analysis/test/gpu-raster/raster-patches/gpu-raster-patches.node.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-patches.md',
    dimensionality: '2d'
  },
  GPURasterSieve: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Sieve filter for a dense patch label raster (GDAL `gdal_sieve`, GRASS `r.reclass.area`).',
    tests: 'modules/analysis/test/gpu-raster/raster-patches/gpu-raster-patches.node.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-patches.md',
    dimensionality: '2d'
  },
  GPUDistanceField: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Computes a Euclidean distance transform and nearest-seed allocation (Voronoi zones) on a raster grid, from seed points snapped to cells and/or a seed raster.',
    tests: 'modules/analysis/test/gpu-raster/distance-field/gpu-distance-field.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUCostDistance: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Computes least-accumulated-cost distance over an 8-connected raster friction surface, with optional back-links, isochrone bands, per-cell settling stamps.',
    tests: 'modules/analysis/test/gpu-raster/cost-distance/gpu-cost-distance.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUCostDistancePath: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Extracts the least-cost path from a target cell back to its source by following the back-links written by GPUCostDistance.',
    tests: 'modules/analysis/test/gpu-raster/cost-distance/gpu-cost-distance.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterWalkingCost: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Directed 8-connected least-cost movement.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPUPolygonRasterization: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Scan-converts polygons into a dense zone-ID raster on the GPU, for raster joins (`GPURasterJoin`) and raster zonal statistics (`GPURasterZonalStatistics`).',
    tests:
      'modules/analysis/test/gpu-raster/polygon-rasterization/polygon-rasterization.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterJoin: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Approximate polygon aggregation of points (a raster join).',
    tests:
      'modules/analysis/test/gpu-raster/polygon-rasterization/polygon-rasterization.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterZonalStatistics: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Computes per-zone count, sum, mean, minimum, and maximum of a raster band, where zones are a pre-rasterized grid of dense zone IDs.',
    tests:
      'modules/analysis/test/gpu-raster/raster-zonal-statistics/gpu-raster-zonal-statistics.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterFractionalCoverage: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Exact planar triangle/pixel clipping followed by compact (zone,cell,coverage) tuples.',
    tests:
      'modules/analysis/test/gpu-raster/fractional-zonal-distribution/fractional-coverage-rounding.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterFractionalZonalStatistics: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Area-aware fractional aggregation, preserving missing support and incomplete coverage.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterFractionalCategoricalStatistics: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Fractional categorical composition using stable, sorted zone/category sums.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/round-two.spec.ts',
    documentation: 'docs/api/gpu-raster/fractional-categorical-statistics.md',
    dimensionality: '2d'
  },
  GPURasterFractionalZonalDistribution: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Coverage times observation-weight distribution of calibrated finite raster values.',
    tests:
      'modules/analysis/test/gpu-raster/fractional-zonal-distribution/fractional-zonal-distribution.node.spec.ts',
    documentation: 'docs/api/gpu-raster/fractional-zonal-distribution.md',
    dimensionality: '2d'
  },
  GPUIsolines: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Marching-squares isolines of a raster at several levels, as ordered segments and optionally stitched polylines.',
    tests: 'modules/analysis/test/gpu-raster/isolines/gpu-isolines.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUIsobands: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Filled contours (isobands) of a float32 raster with per-frame breaks.',
    tests: 'modules/analysis/test/gpu-raster/isolines/gpu-isobands.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUIsobandRings: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Closed polygon rings of every isoband: shells and holes in GeoArrow layout.',
    tests: 'modules/analysis/test/gpu-raster/isolines/parity-wp4-isoband-rings.parity.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUParticleAdvection: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Advects particles through a 2D vector field (wind, currents, flow directions), one frame per encoding, with exact replays.',
    tests:
      'modules/analysis/test/gpu-raster/particle-advection/gpu-particle-advection.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPULagrangianFlowMap: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Integrates a particle per seed through a time-varying planar or internally projected longitude-latitude vector field (two slices, linear in time) for FTLE, connectivity and residence time.',
    tests:
      'modules/analysis/test/gpu-raster/lagrangian-diagnostics/lagrangian-diagnostics-mob06.spec.ts',
    documentation: 'docs/api/gpu-raster/lagrangian-diagnostics-mob06.md',
    dimensionality: '2d+time'
  },
  GPUScalarFieldRidges: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Local height-ridge mask and downward curvature of utilization volumes or FTLE, using a 3 by 3 Hessian stencil and a half-open subcell Newton test (Peikert and Sadlo, 2008).',
    tests:
      'modules/analysis/test/gpu-raster/lagrangian-diagnostics/flow-field-extensions-mob06.spec.ts',
    documentation: 'docs/api/gpu-raster/lagrangian-diagnostics-mob06.md',
    dimensionality: '2d'
  },
  GPUFiniteTimeLyapunovExponent: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Finite-time Lyapunov exponent of a flow map on a regular seed grid, from the largest eigenvalue of the Cauchy-Green tensor.',
    tests:
      'modules/analysis/test/gpu-raster/lagrangian-diagnostics/lagrangian-diagnostics-mob06.spec.ts',
    documentation: 'docs/api/gpu-raster/lagrangian-diagnostics-mob06.md',
    dimensionality: '2d+time'
  },
  GPULineIntegralConvolution: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Line integral convolution (Cabral and Leedom 1993): a texture that smears white noise along the streamlines of a vector field.',
    tests:
      'modules/analysis/test/gpu-raster/flow-texture/gpu-line-integral-convolution.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUStreamlines: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Evenly spaced streamlines (after Jobard and Lefer 1997) resolved by priority, so the result does not depend on GPU scheduling.',
    tests: 'modules/analysis/test/gpu-raster/flow-texture/gpu-streamlines.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterReprojection: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Destination-driven grid alignment/reprojection.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-terrain-expansion.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterTileReprojection: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Explicit GPU mosaic assembly followed by destination-driven reprojection.',
    tests: 'modules/analysis/test/geospatial-expansion/raster-tile-reprojection.spec.ts',
    documentation: 'docs/api/gpu-raster/expansion-lenses.md',
    dimensionality: '2d'
  },
  GPURasterConservativeRemapping: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Exact rectangle overlap integration, bounded to 256 candidate source cells per destination.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/remote-sensing.node.spec.ts',
    documentation: 'docs/api/gpu-raster/remote-sensing.md',
    dimensionality: '2d'
  },
  GPURasterSpectralFilter: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Periodic spectral raster filtering composed from the published GPUFFT2D.',
    tests: 'modules/analysis/test/gpu-raster/spectral-filter/gpu-raster-spectral-filter.spec.ts',
    documentation: 'docs/api/gpu-raster/spectral-filter.md',
    dimensionality: '2d'
  },
  GPURasterRegistration: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Periodic unmasked phase correlation with separable localized upsampled inverse DFT.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/remote-sensing.node.spec.ts',
    documentation: 'docs/api/gpu-raster/remote-sensing.md',
    dimensionality: '2d'
  },
  GPURasterExtremaPyramid: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Builds a min/max pyramid ("maximum mipmap", Tevs, Ihrke and Seidel 2008) of a raster band.',
    tests: 'modules/analysis/test/gpu-raster/raster-pyramid/raster-pyramid.node.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterSampling: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Samples a float32 raster at points ("extract raster values to points").',
    tests: 'modules/analysis/test/gpu-raster/raster-sampling/gpu-raster-sampling.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPURasterProfile: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Elevation profiles: samples a raster along polylines at a per-frame spacing.',
    tests: 'modules/analysis/test/gpu-raster/raster-sampling/gpu-raster-profile.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d'
  },
  GPUChangeDetection: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Per-cell change detection over a dense stack of raster time slices.',
    tests: 'modules/analysis/test/gpu-raster/change-detection/gpu-change-detection.spec.ts',
    documentation: 'docs/api/gpu-raster/operations-analysis.md',
    dimensionality: '2d+time'
  },
  GPURasterTemporal: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Composites write one row per cell; SG writes cellCount*sliceCount rows.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/remote-sensing.node.spec.ts',
    documentation: 'docs/api/gpu-raster/remote-sensing.md',
    dimensionality: '2d+time'
  },
  GPURasterSeasonalBreak: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Offline single-harmonic trend plus at most one level shift.',
    tests: 'modules/analysis/test/gpu-raster/remote-sensing/remote-sensing.node.spec.ts',
    documentation: 'docs/api/gpu-raster/remote-sensing.md',
    dimensionality: '2d+time'
  },
  GPURasterEmbeddingDequantize: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Decodes AlphaEarth int8, TESSERA per-pixel scaled int8 and IEEE float16 embedding rasters.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingQuantize: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Encodes float32 embedding rasters as packed AlphaEarth int8 with pixel-wide NoData.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/emb01-embedding-codecs.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embedding-codecs.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingSimilarity: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Cosine or dot similarity of every pixel to up to 64 query vectors, best query and threshold mask.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingNearestPixels: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Exact top-k most similar pixels per query by cosine, on the published similarity search.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingChange: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Year-to-year cosine, angle and change-vector length of two embedding rasters.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingKMeans: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'K-means of an embedding raster on the published high-dimensional k-means.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingKNearestClassifier: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Brute-force k-nearest-neighbour classification of an embedding raster against a labelled set.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingLinearProbe: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Softmax inference of a host-trained linear probe on an embedding raster.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingCovariance: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Mean vector and sample covariance of an embedding raster, the GPU half of PCA.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingProjection: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Projection of an embedding raster on principal components with an optional RGB stretch.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingZoneMean: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Coverage-weighted mean embedding vector per zone, renormalized to unit length.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterEmbeddingPyramidLevel: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Overview level of an embedding raster that averages decoded vectors and renormalizes them.',
    tests: 'modules/analysis/test/gpu-raster/raster-embeddings/rts03-embedding-rasters.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-embeddings.md',
    dimensionality: '2d'
  },
  GPURasterStreamingAccumulator: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Chunked per-cell, per-slot counts, fixed-point sums and moments over raster stacks with no slice limit.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/tc07-streaming-accumulator.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterRollingMean: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Trailing rolling mean over a chunked stream, the aggregation SPI and SPEI apply.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts06-climate-indices.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterWhittakerSmoother: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Weighted Whittaker smoother and gap filling on regular or irregular dates with fixed robust reweighting.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts02-whittaker-hants.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterHANTS: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Harmonic time-series fit with bounded reject-and-refit of outliers on irregular dates.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts02-whittaker-hants.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterTemporalResample: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Linear interpolation of irregular observations onto a regular time grid.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts02-whittaker-hants.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterMedoidComposite: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Medoid composite per cell: the real observation with the smallest sum of distances to the others, with its source slice.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts05-compositing.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterGeometricMedianComposite: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Geometric median composite per cell by a fixed number of Weiszfeld iterations over all bands.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts05-compositing.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterBestAvailablePixel: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Best-available-pixel composite: highest day-of-year, year and cloud-distance score per cell.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts05-compositing.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterPhenologyMetrics: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Up to two seasons per cell from amplitude threshold crossings on a smoothed curve: start, end, peak, slopes, integral.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts05-compositing.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterStandardizedIndexCalibration: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Streaming per-slot gamma or log-logistic fit by the APP method for SPI and SPEI.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts06-climate-indices.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterStandardizedIndex: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'SPI and SPEI of a chunk from a streamed calibration fit, clipped to plus or minus 8.21.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts06-climate-indices.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterClimateRunAccumulator: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Streaming spell lengths, exceedance counts and degree days per cell.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts06-climate-indices.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPURasterHarmonicRegression: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Trend and harmonic least squares per cell on an irregular date axis.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts04-change-monitoring.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-change-monitoring.md',
    dimensionality: '2d+time'
  },
  GPURasterCOLDMonitor: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Simplified multi-band COLD monitor (ordinary least squares, no Tmask) on an irregular date axis.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts04-change-monitoring.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-change-monitoring.md',
    dimensionality: '2d+time'
  },
  GPURasterBFASTMonitor: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'BFAST Monitor: harmonic history fit and OLS-MOSUM monitoring on an irregular date axis.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts04-change-monitoring.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-change-monitoring.md',
    dimensionality: '2d+time'
  },
  GPURasterSnapshotCovariance: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Area-weighted temporal Gram matrix for EOF analysis by the method of snapshots, in fixed point.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts07-snapshot-eof.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-eof-analysis.md',
    dimensionality: '2d+time'
  },
  GPURasterEOFProjection: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Spatial EOF maps from the temporal modes of the CPU eigendecomposition.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts07-snapshot-eof.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-eof-analysis.md',
    dimensionality: '2d+time'
  },
  GPURasterLaggedCorrelation: {
    package: `${PACKAGE}/raster-analysis`,
    summary:
      'Pearson correlation of each cell with an index at fixed slice lags, pairwise complete.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/rts07-snapshot-eof.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-eof-analysis.md',
    dimensionality: '2d+time'
  },
  GPURasterClimateAnomaly: {
    package: `${PACKAGE}/raster-analysis`,
    summary: 'Climatology and anomalies from the streaming accumulator state.',
    tests: 'modules/analysis/test/gpu-raster/raster-time-series/tc07-streaming-accumulator.spec.ts',
    documentation: 'docs/api/gpu-raster/raster-time-series.md',
    dimensionality: '2d+time'
  },
  GPUColumnQuantiles: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Exact quantiles of a float32 column by GPU radix selection, plus a per-frame percentile-range filter mask, with no readback and no sort.',
    tests: 'modules/analysis/test/gpu-dataframe/column-classification/gpu-column-quantiles.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUClassBreaks: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Computes choropleth class breaks of one column on the GPU every frame, with no readback: equal interval, quantile, standard deviation, head/tail, box plot.',
    tests:
      'modules/analysis/test/gpu-dataframe/column-classification/gpu-class-breaks.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUColorScale: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Applies a colour scale to a column on the GPU, with the scale type, domain, and palette all per-frame data, so switching them never rebuilds or recompiles the graph.',
    tests: 'modules/analysis/test/gpu-dataframe/column-classification/gpu-color-scale.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUBivariateClassification: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Classifies two columns into an `n x n` bivariate class grid and colours each row from a bivariate palette, optionally fading by a third variable (value-by-alpha).',
    tests:
      'modules/analysis/test/gpu-dataframe/column-classification/gpu-bivariate-classification.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUScalarScale: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Native-style, source-aligned scalar mapping with no source readback or row compaction.',
    tests:
      'modules/analysis/test/gpu-dataframe/column-classification/gpu-scalar-scale.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUVariableStandardization: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Named standardization recipes composed from shared GPUReduction and elementwise passes.',
    tests:
      'modules/analysis/test/gpu-dataframe/variable-standardization/gpu-variable-standardization.node.spec.ts',
    documentation: 'docs/api/variable-standardization.md'
  },
  GPUGroupStatistics: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and column entirely on the GPU, into a capacity-bounded table with ascending keys.',
    tests: 'modules/analysis/test/gpu-dataframe/group-statistics/gpu-group-statistics.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUKeyJoin: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary: 'Attribute join of a left table onto a right table by key, entirely on the GPU.',
    tests: 'modules/analysis/test/gpu-dataframe/key-join/key-join.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUColumnProfile: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Fused per-column statistics panel (the kepler and Studio dataset stats panel), recomputed on the GPU every frame: count, null count, minimum, maximum, sum, mean.',
    tests: 'modules/analysis/test/gpu-dataframe/column-profile/gpu-column-profile.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUCompositeScore: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      "Builds a composite indicator score per row from several indicator columns, as in CARTO's composite-indicator procedures and ArcGIS index builders.",
    tests:
      'modules/analysis/test/gpu-dataframe/composite-indicators/gpu-composite-score.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUInequality: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Per-zone income inequality indices from a column of non-negative values: Gini, Theil T and L, Atkinson, Hoover, Palma, the Lorenz curve.',
    tests: 'modules/analysis/test/gpu-dataframe/composite-indicators/gpu-inequality.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md'
  },
  GPUCalendarBuckets: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Decodes epoch timestamps into calendar columns such as year, month, day, hour and weekday.',
    tests: 'modules/analysis/test/gpu-dataframe/calendar-buckets/gpu-calendar-buckets.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUTemporalReduction: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Reduces timestamped rows into per-(cell, coarse time bucket) minimum, maximum, first, last, and count.',
    tests: 'modules/analysis/test/gpu-dataframe/temporal-reduction/gpu-temporal-reduction.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUTimeWindowFilter: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary: 'Filters timestamped rows or time intervals against a per-frame time window.',
    tests:
      'modules/analysis/test/gpu-dataframe/time-window-filter/gpu-time-window-filter-int64.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUCubeLayoutTranspose: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Moves a dense cell by period cube between cell-major and period-major layouts, bit for bit.',
    tests: 'modules/analysis/test/gpu-dataframe/cube-layout/gpu-cube-layout-transpose.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUPeriodIndex: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Maps Int64 timestamps to calendar or fixed-width period indices from an origin, exactly in integer arithmetic.',
    tests: 'modules/analysis/test/gpu-dataframe/period-index/gpu-period-index.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUTimeWordsRelative: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Converts Int64 time words to float32 offsets from an origin, taking the difference exactly before rounding.',
    tests:
      'modules/analysis/test/gpu-dataframe/time-words-relative/gpu-time-words-relative.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUValueCube: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Dense cell by period cube of count, exact fixed-point sum and mean of a value column, in the cell-major layout of the hot spot cube.',
    tests: 'modules/analysis/test/gpu-dataframe/value-cube/gpu-value-cube.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUCellPeriodKeys: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Packs a 64-bit H3, S2 or Quadbin cell key and a period index into one group key for GPUGroupStatistics, rejecting cells too fine for the period bits.',
    tests: 'modules/analysis/test/gpu-dataframe/value-cube/gpu-cell-period-keys.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUVariogram: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Empirical semivariogram (gstat `variogram`, scikit-gstat `Variogram`) of planar point values, optionally directional.',
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/gpu-variogram.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPUCoregionalizationFit: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Goulard–Voltz weighted least squares with PSD projection for supplied local co-kriging structures.',
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.spec.ts',
    documentation: 'modules/analysis/docs/coregionalization-fit.md'
  },
  GPUVariogramModelFit: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Constrained nugget/sill fitting over a log-range ladder and golden-section refinement, entirely on GPU.',
    tests:
      'modules/analysis/test/gpu-dataframe/pair-statistics/variogram-model-fit-parallel.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPUCrossVariogram: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary: 'Deterministic GPU pair accumulation of (x_i-x_j)(y_i-y_j)/2.',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPUSpaceTimeVariogram: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Empirical space-time semivariogram on the indexed pair histogram: signed `sum (x_i-x_j)(y_i-y_j) / (2 N)` per bin.',
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/gpu-space-time-variogram.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d+time'
  },
  GPUSpatialCorrelogram: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      "Global Moran's I at a ladder of distance bands (spatial correlogram), with analytic z-scores.",
    tests:
      'modules/analysis/test/gpu-dataframe/pair-statistics/gpu-spatial-correlogram.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPURipley: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      "Ripley's K, Besag's L and the annulus pair-correlation function of a planar point pattern in a rectangle or prepared polygon window (spatstat `Kest`, `Lest`, `pcf`).",
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/gpu-ripley.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPURipleyEnvelope: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary: "Monte Carlo complete-spatial-randomness (CSR) envelopes of Ripley's K.",
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/gpu-ripley-envelope.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPURipleyDistanceFunctions: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      "Ripley's distance functions of a planar point pattern: the nearest-neighbor function G, the empty-space function F and J.",
    tests:
      'modules/analysis/test/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions.node.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPUPointPatternIndices: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Clark-Evans nearest-neighbor index and quadrat-count dispersion statistics of a planar point pattern in a rectangle or prepared polygon window.',
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/gpu-point-pattern-indices.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPUMarkedPointStatistics: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary: 'Cross-type K/L, inhomogeneous K and annular pair/mark correlation on a rectangle.',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/spatial-regression/geospatial-expansion.spec.ts',
    documentation: 'docs/api/gpu-dataframe-analysis.md',
    dimensionality: '2d'
  },
  GPUSpaceTimeKFunction: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Space-time K, pair correlation, Diggle D and local (LISTA) statistics of a space-time point pattern in a rectangle and time interval, on the indexed pair histogram with isotropic edge weights.',
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/space-time-k-function.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md',
    dimensionality: '2d+time'
  },
  GPUSpaceTimeDiggleTest: {
    package: `${PACKAGE}/dataframe-analysis`,
    summary:
      'Diggle, Chetwynd, Haggkvist and Morris (1995) Monte Carlo test for space-time interaction: the sum of D(u, v) over the grid of distance and time classes, compared with random relabellings of the event times.',
    tests: 'modules/analysis/test/gpu-dataframe/pair-statistics/space-time-diggle-test.spec.ts',
    documentation: 'docs/api/gpu-spatial-analysis/space-time-density-and-k-function.md',
    dimensionality: '2d+time'
  },
  GPUPointCloudNeighborhoods: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary:
      'Exact 3D neighborhoods with all-pairs or caller-bounded sorted-grid search and observable truncation.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-neighborhoods-cooperative.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudFeatures: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary: 'Centered 3D covariance and bounded symmetric Jacobi eigensolve on GPU neighborhoods.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-analysis.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudStatisticalFilter: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary: 'Global mean-neighbor-distance outlier filter (PDAL `filters.outlier`, statistical).',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-advanced.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudGround: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary: 'Bounded progressive morphological filter; a square-window PMF, not SMRF equivalence.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-analysis.node.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudSMRF: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary:
      'Simple morphological filtering following the PDAL 2.8.4 SMRF stages: diamond openings, low outliers, optional net cuts.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-smrf.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudSurfaces: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary:
      'Explicit cell-min DTM and cell-max DSM, with no interpolation or implicit datum conversion.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-advanced.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudHeight: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary:
      'Height above ground using the containing terrain cell, explicitly preserving negative heights.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-advanced.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudICP: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary:
      "Bounded GPU point-to-point ICP with distance rejection and Horn's rigid least-squares solve.",
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-advanced.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudNormalScale: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary: 'Select the minimum-roughness normal over a bounded set of spherical scales.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-normal-scale.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudChange: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary:
      'M3C2-style cylindrical mean differences with explicit orientation, alignment and support.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-analysis.node.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUPointCloudVoxelMerge: {
    package: `${PACKAGE}/pointcloud-analysis`,
    summary:
      'Explicit associative sufficient-statistic merge; no concatenation or halo deduplication is inferred.',
    tests: 'modules/analysis/test/gpu-pointcloud/gpu-pointcloud-advanced.spec.ts',
    documentation: 'docs/api/pointcloud-analysis.md',
    dimensionality: '3d'
  },
  GPUDatumGridTransform: {
    package: `${PACKAGE}/geospatial-analysis`,
    summary: 'GPU-resident bilinear datum shift with ordered grids and bounded inverse iteration.',
    tests:
      'modules/analysis/test/gpu-spatial-analysis/global-coordinates/global-coordinates.spec.ts',
    documentation: 'docs/api/geospatial.md',
    dimensionality: '2d'
  },
  GPUPointBuffer: {
    package: `${PACKAGE}/geospatial-analysis`,
    summary: 'Generates inscribed circle vertices around points.',
    tests: 'modules/analysis/test/gpu-geometry/gpu-point-buffer.spec.ts',
    documentation: 'docs/api/geospatial.md',
    dimensionality: '2d'
  }
};
