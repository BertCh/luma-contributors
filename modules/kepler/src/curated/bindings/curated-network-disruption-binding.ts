// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createWGSLKernelNode,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {
  addNetworkDisruptionRecipe,
  getGPUNetworkAccessibilityParameterValues
} from '@luma-contributors/analysis/gpu-network';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {NETWORK_PARAMETERS, getNetworkParameters} from '../../analysis/catalog-binding-types';
import {buildNetworkGraph, type NetworkGraph} from '../../analysis/catalog-network';
import {buildCatalogDirectedCsr} from '../../analysis/catalog-gpu-network';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {
  appendCatalogRow,
  countFeatureVertices,
  getFeatureIdentity,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureDistance,
  type CatalogFeatureCollection
} from '../../analysis/catalog-geojson';

const BINDING_ID = 'curated-network-disruption';
/** Bound CPU preparation, publication, routing scratch and two routing command graphs separately. */
export const NETWORK_DISRUPTION_LIMITS = {
  features: 2048,
  vertices: 16384,
  edges: 4096,
  nodes: 4096,
  arcs: 8192,
  points: 128,
  pairs: 16384
} as const;

/**
 * Non-negative shortest paths need at most nodeCount - 1 arcs. Allow one further relaxation,
 * with a factor-of-four margin below float32 overflow. This power-of-two ceiling also bounds
 * rounded sums: each of at most 4096 additions is at most this exactly representable value.
 */
export const MAXIMUM_NETWORK_DISRUPTION_ARC_COST = 2 ** 126 / NETWORK_DISRUPTION_LIMITS.nodes;

const PARAMETERS: CatalogOperationBinding['parameters'] = [
  ...NETWORK_PARAMETERS.filter(parameter => parameter.name !== 'crossings'),
  {
    name: 'closureProperty',
    label: 'Closed street column (true or 1 closes both directions)',
    kind: 'string',
    maximumLength: 64,
    defaultValue: 'closed'
  },
  {
    name: 'weightProperty',
    label: 'Facility opportunity weight column (empty counts facilities)',
    kind: 'string',
    maximumLength: 64,
    defaultValue: ''
  },
  {
    name: 'maximumSnapDistance',
    label: 'Maximum distance to a routing endpoint',
    kind: 'number',
    minimum: 0,
    maximum: 1e6,
    unit: 'crs',
    defaultValue: 100
  },
  {
    name: 'maximumCost',
    label: 'Accessibility threshold in routing cost units',
    kind: 'number',
    minimum: 0,
    maximum: 1e12,
    defaultValue: 100
  },
  {
    name: 'costUnit',
    label: 'Property cost unit',
    kind: 'enum',
    options: ['cost-unit', 'seconds', 'minutes', 'hours'],
    defaultValue: 'cost-unit'
  }
];

export type PreparedCuratedNetworkDisruption = {
  graph: NetworkGraph;
  topology: ReturnType<typeof buildCatalogDirectedCsr>;
  /** One entry per CSR slot, including parallel edges; never compacted for a closure. */
  arcIdentities: {edge: number; from: number; to: number; forward: boolean}[];
  edgeMask: Uint32Array;
  demand: CatalogFeatureCollection;
  facilities: CatalogFeatureCollection;
  demandNodes: Uint32Array;
  facilityNodes: Uint32Array;
  demandSnapDistances: Float64Array;
  facilitySnapDistances: Float64Array;
  opportunityWeights: Float32Array;
  maximumCost: number;
  costUnit: string;
};

/** Construction-time preparation uses the existing network builder; it never removes closed arcs. */
export function prepareCuratedNetworkDisruption(
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  parameters: Readonly<Record<string, CatalogParameterValue>>
): PreparedCuratedNetworkDisruption {
  const values = {
    ...Object.fromEntries(PARAMETERS.map(parameter => [parameter.name, parameter.defaultValue!])),
    ...parameters
  };
  for (const specification of PARAMETERS) {
    const value = values[specification.name];
    if (specification.kind === 'number') {
      validateRequirement(
        BINDING_ID,
        typeof value === 'number',
        `${specification.name} must be numeric`
      );
      validateFiniteNumber(
        BINDING_ID,
        specification.name,
        value as number,
        specification.minimum,
        specification.maximum
      );
    } else if (specification.kind === 'boolean') {
      validateRequirement(
        BINDING_ID,
        typeof value === 'boolean',
        `${specification.name} must be boolean`
      );
    } else {
      validateRequirement(
        BINDING_ID,
        typeof value === 'string' &&
          value.length <= (specification.maximumLength ?? Infinity) &&
          (!specification.options || specification.options.includes(value)),
        `${specification.name} must be a string within its declared length and allowed values`
      );
    }
  }
  const network = inputs['network'];
  for (const [name, maximumFeatures, maximumVertices, geometryTypes] of [
    [
      'network',
      NETWORK_DISRUPTION_LIMITS.features,
      NETWORK_DISRUPTION_LIMITS.vertices,
      ['LineString', 'MultiLineString']
    ],
    ['facilities', NETWORK_DISRUPTION_LIMITS.points, NETWORK_DISRUPTION_LIMITS.points, ['Point']],
    ['demand', NETWORK_DISRUPTION_LIMITS.points, NETWORK_DISRUPTION_LIMITS.points, ['Point']]
  ] as const) {
    const input = inputs[name];
    validateRequirement(
      BINDING_ID,
      input?.type === 'FeatureCollection' && Array.isArray(input.features),
      `${name} must be a FeatureCollection`
    );
    validateCapacity(
      BINDING_ID,
      `${name} feature count`,
      input.features.length,
      1,
      maximumFeatures
    );
    validateCapacity(
      BINDING_ID,
      `${name} vertex count`,
      input.features.reduce((count, feature) => count + countFeatureVertices(feature), 0),
      1,
      maximumVertices
    );
    validateRequirement(
      BINDING_ID,
      input.features.every(feature =>
        (geometryTypes as readonly string[]).includes(feature.geometry?.type ?? '')
      ),
      `${name} geometry must be ${geometryTypes.join(' or ')}`
    );
  }
  validateRequirement(
    BINDING_ID,
    Boolean(values['closureProperty']),
    'closureProperty must name a boolean or 0/1 closure column'
  );
  const closureValues = network.features.map(
    feature => feature.properties?.[String(values['closureProperty'])]
  );
  validateRequirement(
    BINDING_ID,
    closureValues.every(value => value === true || value === false || value === 0 || value === 1),
    'closureProperty must contain only boolean or numeric 0/1 values on every street'
  );
  const graph = buildNetworkGraph(
    network,
    getNetworkParameters(values),
    NETWORK_DISRUPTION_LIMITS.edges,
    BINDING_ID
  );
  validateCapacity(BINDING_ID, 'nodeCount', graph.nodes.length, 1, NETWORK_DISRUPTION_LIMITS.nodes);
  const arcIdentities = graph.outgoing.flatMap((row, from) =>
    row.map(arc => ({edge: arc.edge, from, to: arc.to, forward: arc.forward}))
  );
  validateCapacity(BINDING_ID, 'arcCount', arcIdentities.length, 1, NETWORK_DISRUPTION_LIMITS.arcs);
  const topology = buildCatalogDirectedCsr(
    graph.nodes.length,
    graph.outgoing.flatMap((row, from) => row.map(arc => ({from, to: arc.to, cost: arc.cost})))
  );
  validateRequirement(
    BINDING_ID,
    topology.weights.every(weight => Number.isFinite(weight) && weight >= 0),
    'arc costs must be finite non-negative float32 values'
  );
  validateFiniteNumber(
    BINDING_ID,
    'maximumArcCost (rescale routing costs to keep supported path sums finite)',
    Math.max(...topology.weights),
    0,
    MAXIMUM_NETWORK_DISRUPTION_ARC_COST
  );
  const edgeMask = Uint32Array.from(arcIdentities, arc =>
    closureValues[graph.edges[arc.edge].sourceIndex] ? 0 : 1
  );
  // The builder retains interior vertices in nodes even when they are not split into routing
  // endpoints. Snapping to those unused CSR rows would invent a disconnected demand/facility.
  const endpointNodes = new Set(graph.edges.flatMap(edge => [edge.fromNode, edge.toNode]));
  const snap = (name: 'demand' | 'facilities') => {
    const distances = new Float64Array(inputs[name].features.length);
    const nodes = Uint32Array.from(inputs[name].features, (feature, row) => {
      const position = getPointPositions(feature.geometry)[0];
      validateRequirement(
        BINDING_ID,
        Boolean(position) && position.every(Number.isFinite),
        `${name} coordinates must be finite Point coordinates`
      );
      let nearestNode = 0;
      let minimumDistance = Infinity;
      graph.nodes.forEach((node, index) => {
        if (!endpointNodes.has(index)) return;
        const distance = measureDistance(position, node, graph.coordinateSystem);
        if (distance < minimumDistance) {
          minimumDistance = distance;
          nearestNode = index;
        }
      });
      validateRequirement(
        BINDING_ID,
        minimumDistance <= Number(values['maximumSnapDistance']),
        `${name} row ${row} distance to a routing endpoint exceeds maximumSnapDistance ${values['maximumSnapDistance']}; supply endpoint-aligned points or raise the distance`
      );
      distances[row] = minimumDistance;
      return nearestNode;
    });
    return {nodes, distances};
  };
  const demand = inputs['demand'];
  const facilities = inputs['facilities'];
  validateCapacity(
    BINDING_ID,
    'pairCount',
    demand.features.length * facilities.features.length,
    1,
    NETWORK_DISRUPTION_LIMITS.pairs
  );
  const demandSnaps = snap('demand');
  const facilitySnaps = snap('facilities');
  const opportunityWeights = Float32Array.from(facilities.features, feature => {
    const weight = values['weightProperty']
      ? feature.properties?.[String(values['weightProperty'])]
      : 1;
    validateRequirement(
      BINDING_ID,
      typeof weight === 'number',
      'weightProperty must contain finite non-negative numeric facility weights'
    );
    validateFiniteNumber(BINDING_ID, 'weightProperty', weight as number, 0, 1e20);
    return weight as number;
  });
  return {
    graph,
    topology,
    arcIdentities,
    edgeMask,
    demand,
    facilities,
    demandNodes: demandSnaps.nodes,
    facilityNodes: facilitySnaps.nodes,
    demandSnapDistances: demandSnaps.distances,
    facilitySnapDistances: facilitySnaps.distances,
    opportunityWeights,
    maximumCost: Number(values['maximumCost']),
    costUnit:
      values['costMode'] === 'property'
        ? String(values['costUnit'])
        : graph.coordinateSystem === 'EPSG:4326'
          ? 'meters'
          : 'coordinate-unit'
  };
}

export type CuratedNetworkDisruptionReadback = {
  baselineCosts: ArrayLike<number>;
  scenarioCosts: ArrayLike<number>;
  costDifferences: ArrayLike<number>;
  newlyUnreachable: ArrayLike<number>;
  baselineAccessibility: ArrayLike<number>;
  scenarioAccessibility: ArrayLike<number>;
  accessibilityDifferences: ArrayLike<number>;
  stages: readonly {stage: string; converged: number}[];
};

/** Only converged full-network results can become native publication tables. */
export function publishCuratedNetworkDisruption(
  prepared: PreparedCuratedNetworkDisruption,
  readback: CuratedNetworkDisruptionReadback
): CatalogRunOutput {
  for (const stage of ['baseline-routing', 'scenario-routing'])
    validateRequirement(
      BINDING_ID,
      readback.stages.some(status => status.stage === stage && status.converged === 1),
      `${stage} did not converge within 64 routing iterations; reduce the network before publishing`
    );
  const demandCount = prepared.demand.features.length;
  const facilityCount = prepared.facilities.features.length;
  for (const name of [
    'baselineCosts',
    'scenarioCosts',
    'costDifferences',
    'newlyUnreachable',
    'baselineAccessibility',
    'scenarioAccessibility',
    'accessibilityDifferences'
  ] as const)
    validateRequirement(
      BINDING_ID,
      readback[name].length ===
        (['baselineAccessibility', 'scenarioAccessibility', 'accessibilityDifferences'].includes(
          name
        )
          ? demandCount
          : demandCount * facilityCount),
      `${name} readback length must match the prepared matrix dimensions`
    );
  const pairs = makeCatalogTable('pairs', 'Demand to facility costs', {
    demand_index: 'int32',
    demand_source_id: 'string',
    facility_index: 'int32',
    facility_source_id: 'string',
    demand_node: 'int32',
    facility_node: 'int32',
    demand_snap_distance: 'float64',
    facility_snap_distance: 'float64',
    baseline_cost: 'float64',
    scenario_cost: 'float64',
    cost_difference: 'float64',
    newly_unreachable: 'boolean',
    status: 'string',
    cost_unit: 'string'
  });
  const demand = makeCatalogTable(
    'demand-access',
    'Demand access loss',
    {
      demand_index: 'int32',
      demand_source_id: 'string',
      node_index: 'int32',
      snap_distance: 'float64',
      baseline_accessibility: 'float64',
      scenario_accessibility: 'float64',
      accessibility_difference: 'float64',
      newly_unreachable_count: 'int32',
      cut_off: 'boolean',
      status: 'string',
      cost_unit: 'string',
      maximum_cost: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let newlyUnreachableCount = 0;
  let cutOffCount = 0;
  prepared.demand.features.forEach((feature, row) => {
    let baselineReached = 0;
    let scenarioReached = 0;
    let rowUnreachable = 0;
    prepared.facilities.features.forEach((facility, column) => {
      const index = row * facilityCount + column;
      const baseline = readback.baselineCosts[index];
      const scenario = readback.scenarioCosts[index];
      validateRequirement(
        BINDING_ID,
        ((Number.isFinite(baseline) && baseline >= 0) || baseline === Infinity) &&
          ((Number.isFinite(scenario) && scenario >= 0) || scenario === Infinity),
        'routing costs must be non-negative finite values or positive Infinity'
      );
      const before = Number.isFinite(baseline);
      const after = Number.isFinite(scenario);
      const newlyUnreachable = before && !after;
      validateRequirement(
        BINDING_ID,
        readback.newlyUnreachable[index] === Number(newlyUnreachable) &&
          (before && after
            ? Number.isFinite(readback.costDifferences[index])
            : Number.isNaN(readback.costDifferences[index])),
        'pair differences or newlyUnreachable status are inconsistent with routing costs'
      );
      baselineReached += Number(before);
      scenarioReached += Number(after);
      rowUnreachable += Number(newlyUnreachable);
      appendCatalogRow(pairs, {
        demand_index: row,
        demand_source_id: getFeatureIdentity(feature),
        facility_index: column,
        facility_source_id: getFeatureIdentity(facility),
        demand_node: prepared.demandNodes[row],
        facility_node: prepared.facilityNodes[column],
        demand_snap_distance: prepared.demandSnapDistances[row],
        facility_snap_distance: prepared.facilitySnapDistances[column],
        baseline_cost: before ? baseline : null,
        scenario_cost: after ? scenario : null,
        cost_difference: before && after ? readback.costDifferences[index] : null,
        newly_unreachable: newlyUnreachable,
        status: newlyUnreachable
          ? 'newly-unreachable'
          : !before
            ? 'baseline-unreachable'
            : 'reachable',
        cost_unit: prepared.costUnit
      });
    });
    for (const name of [
      'baselineAccessibility',
      'scenarioAccessibility',
      'accessibilityDifferences'
    ] as const)
      validateRequirement(
        BINDING_ID,
        Number.isFinite(readback[name][row]),
        `${name} must be finite after convergence`
      );
    const cutOff = baselineReached > 0 && scenarioReached === 0;
    cutOffCount += Number(cutOff);
    newlyUnreachableCount += rowUnreachable;
    appendCatalogRow(demand, {
      demand_index: row,
      demand_source_id: getFeatureIdentity(feature),
      node_index: prepared.demandNodes[row],
      snap_distance: prepared.demandSnapDistances[row],
      baseline_accessibility: readback.baselineAccessibility[row],
      scenario_accessibility: readback.scenarioAccessibility[row],
      accessibility_difference: readback.accessibilityDifferences[row],
      newly_unreachable_count: rowUnreachable,
      cut_off: cutOff,
      status: cutOff
        ? 'cut-off'
        : baselineReached === 0
          ? 'baseline-unreachable'
          : rowUnreachable > 0
            ? 'lost-facilities'
            : 'reachable',
      cost_unit: prepared.costUnit,
      maximum_cost: prepared.maximumCost,
      geometry: JSON.stringify(feature.geometry)
    });
  });
  const arcs = makeCatalogTable(
    'scenario-arcs',
    'Stable directed scenario arcs',
    {
      csr_slot: 'int32',
      edge_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      from_node: 'int32',
      to_node: 'int32',
      direction: 'string',
      baseline_cost: 'float64',
      scenario_cost: 'float64',
      closed: 'boolean',
      cost_unit: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  prepared.arcIdentities.forEach((arc, slot) => {
    const edge = prepared.graph.edges[arc.edge];
    appendCatalogRow(arcs, {
      csr_slot: slot,
      edge_id: edge.edgeId,
      source_index: edge.sourceIndex,
      source_id: edge.sourceId,
      part_index: edge.partIndex,
      from_node: arc.from,
      to_node: arc.to,
      direction: arc.forward ? 'forward' : 'reverse',
      baseline_cost: prepared.topology.weights[slot],
      scenario_cost: prepared.edgeMask[slot] ? prepared.topology.weights[slot] : null,
      closed: prepared.edgeMask[slot] === 0,
      cost_unit: prepared.costUnit,
      geometry: makeCatalogGeometryText(
        'LineString',
        arc.forward ? edge.coordinates : [...edge.coordinates].reverse()
      )
    });
  });
  return {
    tables: [demand, pairs, arcs],
    diagnostics: [
      ...prepared.graph.diagnostics,
      {
        code: 'endpoint-snapping',
        severity: 'info',
        count: demandCount + facilityCount,
        message:
          'Demand routes toward facilities on the original directed network. Points snap to the nearest routing endpoint; connector distance is reported but adds no travel cost. Crossings without a shared vertex are not noded.'
      },
      {
        code: 'unreachable-costs',
        severity: 'info',
        count: newlyUnreachableCount,
        message:
          'Unreachable costs and undefined Infinity differences are null; cut-off means no directed path to any facility, independently of the accessibility threshold or facility weight.'
      }
    ],
    summary: {
      demandCount,
      facilityCount,
      nodeCount: prepared.graph.nodes.length,
      edgeCount: prepared.graph.edges.length,
      arcCount: arcs.rowCount,
      closedArcCount: Array.from(prepared.edgeMask).filter(value => value === 0).length,
      newlyUnreachableCount,
      cutOffCount,
      maximumCost: prepared.maximumCost,
      costUnit: prepared.costUnit,
      converged: 1
    }
  };
}

export const CURATED_NETWORK_DISRUPTION_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'What does a closure cut off?',
  domain: 'network',
  summary:
    'Compare directed demand-to-facility routing and cumulative accessibility before and after street closures without changing CSR identities.',
  capabilityIds: ['gpu-network'],
  contributorNames: [
    'addNetworkDisruptionRecipe',
    'GPUNetworkCostMatrix',
    'GPUNetworkAccessibility'
  ],
  gpuContributorNames: [
    'GPUNetworkCostMatrix',
    'GPUNetworkReachability',
    'GPUNetworkAccessibility'
  ],
  engine: 'gpu',
  inputs: [
    {
      name: 'network',
      label: 'Street network',
      accepts: ['line'],
      maximumFeatures: NETWORK_DISRUPTION_LIMITS.features,
      maximumVertices: NETWORK_DISRUPTION_LIMITS.vertices
    },
    ...['facilities', 'demand'].map(name => ({
      name,
      label: name === 'facilities' ? 'Facilities' : 'Demand points',
      accepts: ['point'] as const,
      maximumFeatures: NETWORK_DISRUPTION_LIMITS.points,
      maximumVertices: NETWORK_DISRUPTION_LIMITS.points
    }))
  ],
  parameters: PARAMETERS,
  outputs: [
    {
      tableId: 'demand-access',
      label: 'Demand access loss',
      geometry: 'point',
      identityColumns: ['demand_index', 'demand_source_id', 'node_index'],
      maximumRows: NETWORK_DISRUPTION_LIMITS.points
    },
    {
      tableId: 'pairs',
      label: 'Demand to facility costs',
      geometry: 'none',
      identityColumns: ['demand_index', 'demand_source_id', 'facility_index', 'facility_source_id'],
      maximumRows: NETWORK_DISRUPTION_LIMITS.pairs
    },
    {
      tableId: 'scenario-arcs',
      label: 'Directed scenario arcs',
      geometry: 'line',
      identityColumns: ['csr_slot', 'edge_id', 'source_index', 'source_id', 'part_index'],
      maximumRows: NETWORK_DISRUPTION_LIMITS.arcs
    }
  ],
  cleanup:
    'One catalog GPU session owns uploads, readbacks and compiled graphs and releases them on success or failure; returned tables are plain arrays.',
  declaredBehavior: [
    'Closures mask every allowed direction of a source feature without compacting nodes, physical edges, parallel edges or CSR slots.',
    'All three nonempty inputs and separate physical-edge, node, directed-arc and matrix bounds are checked before GPU allocation.',
    'Nonconverged baseline or scenario routing refuses publication; unreachable costs and undefined cost differences are null.',
    `Each float32 arc cost must be at most ${MAXIMUM_NETWORK_DISRUPTION_ARC_COST}; rescale larger routing costs before execution to keep supported path sums finite.`
  ],
  limitation:
    'Nearest-endpoint snapping only; no partial-edge connectors, turn restrictions, crossing noding, retained-baseline cache or replacement costs. Searches use 16 lanes and 64 iterations; threshold settings rebuild the graph. Property cost units must be declared by the analyst.',
  async runGpu(context, inputs, parameters) {
    const prepared = prepareCuratedNetworkDisruption(inputs, parameters);
    return withCatalogGpuSession(context.device, BINDING_ID, async session => {
      const pairCount = prepared.demandNodes.length * prepared.facilityNodes.length;
      const baselineCosts = session.output('baseline-costs', 'float32', pairCount);
      const scenarioCosts = session.output('scenario-costs', 'float32', pairCount);
      const costDifferences = session.output('cost-differences', 'float32', pairCount);
      const newlyUnreachable = session.output('newly-unreachable', 'uint32', pairCount);
      const baselineAccessibility = session.output(
        'baseline-accessibility',
        'float32',
        prepared.demandNodes.length
      );
      const scenarioAccessibility = session.output(
        'scenario-accessibility',
        'float32',
        prepared.demandNodes.length
      );
      const accessibilityDifferences = session.output(
        'accessibility-differences',
        'float32',
        prepared.demandNodes.length
      );
      const recipe = addNetworkDisruptionRecipe(session.graph, {
        search: {
          topology: {
            offsets: session.input('offsets', prepared.topology.offsets, 'uint32'),
            neighbors: session.input('neighbors', prepared.topology.neighbors, 'uint32'),
            weights: session.input('weights', prepared.topology.weights, 'float32')
          },
          startNodes: session.input('demand-nodes', prepared.demandNodes, 'uint32'),
          targetNodes: session.input('facility-nodes', prepared.facilityNodes, 'uint32'),
          laneCount: Math.min(16, prepared.demandNodes.length),
          maximumIterations: 64
        },
        rowCount: prepared.demandNodes.length,
        edgeMask: session.input('closures', prepared.edgeMask, 'uint32'),
        opportunityWeights: session.input('opportunities', prepared.opportunityWeights, 'float32'),
        parameters: session.input(
          'parameters',
          getGPUNetworkAccessibilityParameterValues({threshold: prepared.maximumCost}),
          'float32'
        ),
        outputs: {
          baselineCosts: baselineCosts.view,
          scenarioCosts: scenarioCosts.view,
          costDifferences: costDifferences.view,
          newlyUnreachable: newlyUnreachable.view,
          baselineAccessibility: baselineAccessibility.view,
          scenarioAccessibility: scenarioAccessibility.view,
          accessibilityDifferences: accessibilityDifferences.view
        }
      });
      const statuses = recipe.status.stages.map(stage => {
        const output = session.output(`${stage.stage}-converged`, 'uint32', 1);
        session.graph.add(
          createWGSLKernelNode(session.graph, {
            id: `publish-${stage.stage}`,
            operation: BINDING_ID,
            invocationCount: 1,
            bindings: [
              {name: 'source', view: stage.status.converged!, type: 'u32', access: 'read'},
              {name: 'destination', view: output.view, type: 'u32', access: 'read_write'}
            ],
            body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
          })
        );
        return {stage: stage.stage, output};
      });
      session.run();
      const readback: CuratedNetworkDisruptionReadback = {
        baselineCosts: await session.readFloat32(baselineCosts),
        scenarioCosts: await session.readFloat32(scenarioCosts),
        costDifferences: await session.readFloat32(costDifferences),
        newlyUnreachable: await session.readUint32(newlyUnreachable),
        baselineAccessibility: await session.readFloat32(baselineAccessibility),
        scenarioAccessibility: await session.readFloat32(scenarioAccessibility),
        accessibilityDifferences: await session.readFloat32(accessibilityDifferences),
        stages: await Promise.all(
          statuses.map(async status => ({
            stage: status.stage,
            converged: await session.readScalar(status.output)
          }))
        )
      };
      return {
        ...publishCuratedNetworkDisruption(prepared, readback),
        dispatchedContributors: CURATED_NETWORK_DISRUPTION_BINDING.gpuContributorNames
      };
    });
  }
};
