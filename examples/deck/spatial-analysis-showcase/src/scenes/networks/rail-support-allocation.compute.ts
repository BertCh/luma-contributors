// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addFacilityScenarioRecipe,
  GPUNetworkCostMatrix,
  GPUNetworkFacilityOptimization,
  GPUNetworkRationalAccess,
  GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
  getGPUNetworkAccessibilityParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {findStation, loadRailGraph} from './transit-data';

export type RailSupportAllocationOptions = {
  objective: 'pmedian' | 'backup' | 'dispersion';
  thresholdMinutes: number;
  rationalMode: 'integral' | 'fractional';
  stepMode: 'fraction' | 'absolute';
  supplyScale: number;
  demandMultiplier: number;
  travelScaleMinutes: number;
  scenarioLimitMinutes: number;
  allocationDisplay: 'placement' | 'scenario' | 'rational';
};

const HUB_NAMES = [
  'Amsterdam Centraal',
  'Utrecht Centraal',
  'Rotterdam Centraal',
  'Zwolle'
] as const;
const DEMAND_NAMES = [
  'Hilversum',
  'Leiden Centraal',
  'Gouda',
  'Amersfoort Centraal',
  'Apeldoorn',
  'Alkmaar',
  'Delft',
  'Deventer'
] as const;
const HUB_COLORS: [number, number, number, number][] = [
  [36, 132, 173, 255],
  [238, 132, 43, 255],
  [135, 91, 177, 255],
  [45, 153, 112, 255]
];

function reverseGraph(
  offsets: Uint32Array,
  targets: Uint32Array,
  weights: Float32Array,
  nodeCount: number
) {
  const reversedOffsets = new Uint32Array(nodeCount + 1);
  for (const target of targets) reversedOffsets[target + 1]++;
  for (let node = 0; node < nodeCount; node++) reversedOffsets[node + 1] += reversedOffsets[node];
  const cursor = reversedOffsets.slice(0, nodeCount),
    reversedTargets = new Uint32Array(targets.length),
    reversedWeights = new Float32Array(weights.length);
  for (let source = 0; source < nodeCount; source++)
    for (let edge = offsets[source]; edge < offsets[source + 1]; edge++) {
      const slot = cursor[targets[edge]]++;
      reversedTargets[slot] = source;
      reversedWeights[slot] = weights[edge];
    }
  return {offsets: reversedOffsets, targets: reversedTargets, weights: reversedWeights};
}

/** Compare exact small facility objectives and four rational-access parameter variants. */
export async function createRailSupportAllocation(
  ctx: SceneContext<RailSupportAllocationOptions>
): Promise<SceneInstance<RailSupportAllocationOptions>> {
  const rail = loadRailGraph(ctx.datasets.get('gtfs-nl-rail-graph'));
  const hubNodes = HUB_NAMES.map(name => findStation(rail, name));
  const demandNodes = DEMAND_NAMES.map(name => findStation(rail, name));
  const {device} = ctx;
  const {nodeCount, edgeCount} = rail;
  const facilityCount = hubNodes.length,
    demandCount = demandNodes.length;
  const resources = new SpatialAnalysisResources(device, 'rail-support-allocation');
  const hubNodesBuffer = resources.createBuffer('candidate-hubs', Uint32Array.from(hubNodes));
  const rationalCandidateMaskBuffer = resources.createBuffer(
    'rational-active-candidates',
    new Uint32Array(facilityCount).fill(1)
  );
  const demandNodesBuffer = resources.createBuffer(
    'demand-stations',
    Uint32Array.from(demandNodes)
  );
  const demandBuffer = resources.createBuffer(
    'sampled-unit-demand',
    new Float32Array(demandCount).fill(1)
  );
  const rationalDemandBuffer = resources.createBuffer(
    'rational-sample-demand',
    new Float32Array(demandCount).fill(ctx.options.demandMultiplier)
  );
  const capacitiesBuffer = resources.createBuffer(
    'desk-capacities',
    new Float32Array(facilityCount).fill(4)
  );
  const supplyBuffer = resources.createBuffer(
    'desk-supply',
    new Float32Array(facilityCount).fill(1)
  );
  const thresholdBuffer = resources.createBuffer(
    'backup-threshold',
    Float32Array.of(ctx.options.thresholdMinutes)
  );
  const rationalParametersBuffer = resources.createBuffer(
    'rational-parameters',
    Float32Array.of(ctx.options.travelScaleMinutes, ctx.options.supplyScale, 0.4, 0.02, 30, 0.01)
  );
  const accessibilityParametersBuffer = resources.createBuffer(
    'facility-scenario-parameters',
    getGPUNetworkAccessibilityParameterValues({
      threshold: ctx.options.scenarioLimitMinutes,
      decay: 'exponential',
      beta: 0.03
    })
  );
  const nodePositionsBuffer = resources.createBuffer(
    'all-station-positions',
    rail.nodePositions,
    Buffer.VERTEX
  );
  const demandPositions = Float32Array.from(
    demandNodes.flatMap(node => [...rail.nodePositions.subarray(node * 2, node * 2 + 2)])
  );
  const hubPositions = Float32Array.from(
    hubNodes.flatMap(node => [...rail.nodePositions.subarray(node * 2, node * 2 + 2)])
  );
  const demandPositionsBuffer = resources.createBuffer(
    'sampled-station-positions',
    demandPositions,
    Buffer.VERTEX
  );
  const hubPositionsBuffer = resources.createBuffer(
    'candidate-hub-positions',
    hubPositions,
    Buffer.VERTEX
  );
  const railSegmentsBuffer = resources.createBuffer(
    'rail-network-segments',
    rail.segments,
    Buffer.VERTEX
  );
  const topologyForward = {
    offsets: resources.createBuffer('forward-offsets', rail.offsets),
    neighbors: resources.createBuffer('forward-neighbors', rail.targets),
    weights: resources.createBuffer(
      'forward-travel-minutes',
      Float32Array.from(rail.travelTime, seconds => seconds / 60)
    )
  };
  const reverse = reverseGraph(
    rail.offsets,
    rail.targets,
    Float32Array.from(rail.travelTime, seconds => seconds / 60),
    nodeCount
  );
  const topologyReverse = {
    offsets: resources.createBuffer('reverse-offsets', reverse.offsets),
    neighbors: resources.createBuffer('reverse-neighbors', reverse.targets),
    weights: resources.createBuffer('reverse-travel-minutes', reverse.weights)
  };
  const demandCostFacilityMajor = resources.createBuffer(
    'sample-costs-facility-major',
    facilityCount * demandCount * 4
  );
  const demandCosts = resources.createBuffer(
    'sample-costs-demand-major',
    demandCount * facilityCount * 4
  );
  const hubDistances = resources.createBuffer(
    'hub-pair-directed-costs',
    facilityCount * facilityCount * 4
  );
  const symmetricHubDistances = resources.createBuffer(
    'hub-pair-symmetric-costs',
    facilityCount * facilityCount * 4
  );
  const demandCostStatus = resources.createBuffer('sample-cost-status', 4);
  const hubDistanceStatus = resources.createBuffer('hub-distance-status', 4);
  const outputBuffers = (prefix: string) => ({
    selected: resources.createBuffer(`${prefix}-selected`, facilityCount * 4),
    assignments: resources.createBuffer(`${prefix}-assignments`, demandCount * 4),
    objective: resources.createBuffer(`${prefix}-objective`, 4),
    status: resources.createBuffer(`${prefix}-status`, 16)
  });
  const pmedian = outputBuffers('pmedian');
  const backup = outputBuffers('backup-coverage');
  const dispersion = outputBuffers('dispersion');
  const scenarioOutputs = [0, 1, 2].map(index => ({
    accessibility: resources.createBuffer(`scenario-${index}-accessibility`, demandCount * 4),
    nearestCosts: resources.createBuffer(`scenario-${index}-nearest-costs`, demandCount * 4),
    nearestFacilities: resources.createBuffer(
      `scenario-${index}-nearest-facilities`,
      demandCount * 4
    ),
    covered: resources.createBuffer(`scenario-${index}-covered-demand`, 4),
    uncovered: resources.createBuffer(`scenario-${index}-uncovered-demand`, 4),
    invalid: resources.createBuffer(`scenario-${index}-invalid-demand`, 4)
  }));
  const rationalOutputs = Array.from({length: 4}, (_, index) => ({
    assignments: resources.createBuffer(
      `rational-${index}-assignments`,
      demandCount * facilityCount * 4
    ),
    scores: resources.createBuffer(`rational-${index}-scores`, demandCount * 4),
    winners: resources.createBuffer(`rational-${index}-winners`, demandCount * 4),
    unserved: resources.createBuffer(`rational-${index}-unserved`, demandCount * 4),
    status: resources.createBuffer(`rational-${index}-status`, 16)
  }));

  const graph = new GPUCommandGraph<void>(device, {id: 'rail-support-allocation'});
  const view = createGraphBufferImporter(graph);
  const inputTopology = (topology: typeof topologyForward) => ({
    offsets: view(topology.offsets, 'uint32', nodeCount + 1),
    neighbors: view(topology.neighbors, 'uint32', edgeCount),
    weights: view(topology.weights, 'float32', edgeCount)
  });
  graph.add(
    new GPUNetworkCostMatrix({
      id: 'sample-station-to-hubs',
      topology: inputTopology(topologyReverse),
      startNodes: view(hubNodesBuffer, 'uint32', facilityCount),
      targetNodes: view(demandNodesBuffer, 'uint32', demandCount),
      laneCount: facilityCount,
      maximumIterations: 32,
      localIterations: 32,
      output: {
        costs: view(demandCostFacilityMajor, 'float32', facilityCount * demandCount),
        status: {converged: view(demandCostStatus, 'uint32', 1)}
      }
    })
  );
  graph.add(
    new GPUNetworkCostMatrix({
      id: 'directed-hub-pair-times',
      topology: inputTopology(topologyForward),
      startNodes: view(hubNodesBuffer, 'uint32', facilityCount),
      targetNodes: view(hubNodesBuffer, 'uint32', facilityCount),
      laneCount: facilityCount,
      maximumIterations: 32,
      localIterations: 32,
      output: {
        costs: view(hubDistances, 'float32', facilityCount * facilityCount),
        status: {converged: view(hubDistanceStatus, 'uint32', 1)}
      }
    })
  );
  addKernelPass(graph, {
    id: 'transpose-sampled-facility-costs',
    invocationCount: facilityCount * demandCount,
    bindings: [
      {
        name: 'facilityMajor',
        view: view(demandCostFacilityMajor, 'float32', facilityCount * demandCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'demandMajor',
        view: view(demandCosts, 'float32', demandCount * facilityCount),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let facility = index % ${facilityCount}u; let demand = index / ${facilityCount}u; demandMajor[demandMajorOffset + demand * ${facilityCount}u + facility] = facilityMajor[facilityMajorOffset + facility * ${demandCount}u + demand];`
  });
  addKernelPass(graph, {
    id: 'symmetric-worst-direction-hub-distances',
    invocationCount: facilityCount * facilityCount,
    bindings: [
      {
        name: 'directed',
        view: view(hubDistances, 'float32', facilityCount * facilityCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'symmetric',
        view: view(symmetricHubDistances, 'float32', facilityCount * facilityCount),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let first = index / ${facilityCount}u; let second = index % ${facilityCount}u; let forward = directed[directedOffset + first * ${facilityCount}u + second]; let reverse = directed[directedOffset + second * ${facilityCount}u + first]; symmetric[symmetricOffset + index] = max(forward, reverse);`
  });

  function addOptimization(
    id: string,
    operation: 'capacitated-p-median' | 'backup-coverage' | 'p-dispersion',
    outputs: ReturnType<typeof outputBuffers>,
    costs: Buffer,
    costLength: number
  ) {
    graph.add(
      new GPUNetworkFacilityOptimization({
        id,
        operation,
        facilityCount,
        selectCount: 2,
        costs: view(costs, 'float32', costLength),
        demand: view(demandBuffer, 'float32', demandCount),
        capacities: view(capacitiesBuffer, 'float32', facilityCount),
        threshold: view(thresholdBuffer, 'float32', 1),
        maximumCandidates: operation === 'capacitated-p-median' ? 65536 : undefined,
        output: {
          selectedFacilities: view(outputs.selected, 'uint32', facilityCount),
          assignments: view(outputs.assignments, 'uint32', demandCount),
          objective: view(outputs.objective, 'float32', 1),
          status: view(outputs.status, 'uint32', 4)
        }
      })
    );
  }
  addOptimization(
    'capacitated-p-median',
    'capacitated-p-median',
    pmedian,
    demandCosts,
    demandCount * facilityCount
  );
  addOptimization(
    'backup-coverage',
    'backup-coverage',
    backup,
    demandCosts,
    demandCount * facilityCount
  );
  addOptimization(
    'symmetric-p-dispersion',
    'p-dispersion',
    dispersion,
    symmetricHubDistances,
    facilityCount * facilityCount
  );
  for (const [index, outputs] of scenarioOutputs.entries()) {
    const selected =
      index === 0 ? pmedian.selected : index === 1 ? backup.selected : dispersion.selected;
    addFacilityScenarioRecipe(graph, {
      id: `facility-scenario-${index}`,
      costs: view(demandCostFacilityMajor, 'float32', facilityCount * demandCount),
      demand: view(demandBuffer, 'float32', demandCount),
      supply: view(supplyBuffer, 'float32', facilityCount),
      facilityMask: view(selected, 'uint32', facilityCount),
      parameters: view(
        accessibilityParametersBuffer,
        'float32',
        GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH
      ),
      outputs: {
        accessibility: view(outputs.accessibility, 'float32', demandCount),
        nearestCosts: view(outputs.nearestCosts, 'float32', demandCount),
        nearestFacilities: view(outputs.nearestFacilities, 'uint32', demandCount),
        coveredDemand: view(outputs.covered, 'float32', 1),
        uncoveredDemand: view(outputs.uncovered, 'float32', 1),
        invalidDemandCount: view(outputs.invalid, 'uint32', 1)
      }
    });
  }
  for (const [index, modes] of (['integral', 'fractional'] as const)
    .flatMap(demandMode =>
      (['fraction', 'absolute'] as const).map(stepMode => [demandMode, stepMode] as const)
    )
    .entries()) {
    const [demandMode, stepMode] = modes;
    const output = rationalOutputs[index];
    graph.add(
      new GPUNetworkRationalAccess({
        id: `rational-access-${demandMode}-${stepMode}`,
        costs: view(demandCosts, 'float32', demandCount * facilityCount),
        demand: view(rationalDemandBuffer, 'float32', demandCount),
        supply: view(supplyBuffer, 'float32', facilityCount),
        parameters: view(rationalParametersBuffer, 'float32', 6),
        maximumCycles: 120,
        demandMode,
        stepMode,
        output: {
          assignments: view(output.assignments, 'float32', demandCount * facilityCount),
          scores: view(output.scores, 'float32', demandCount),
          unservedDemand: view(output.unserved, 'float32', demandCount),
          status: view(output.status, 'uint32', 4)
        }
      })
    );
    addKernelPass(graph, {
      id: `rational-winner-${index}`,
      invocationCount: demandCount,
      bindings: [
        {
          name: 'assignments',
          view: view(output.assignments, 'float32', demandCount * facilityCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'winners',
          view: view(output.winners, 'uint32', demandCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `var winner = 0xffffffffu; var maximum = 0.0; for (var facility = 0u; facility < ${facilityCount}u; facility++) { let value = assignments[assignmentsOffset + index * ${facilityCount}u + facility]; if (value > maximum) { maximum = value; winner = facility; } } winners[winnersOffset + index] = winner;`
    });
  }

  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());
  let generation = 1,
    pendingGeneration = 0,
    dirty = true,
    destroyed = false,
    outputReady = false;
  let lastPmedianSelected: Uint32Array = new Uint32Array(facilityCount),
    lastBackupSelected: Uint32Array = new Uint32Array(facilityCount),
    lastDispersionSelected: Uint32Array = new Uint32Array(facilityCount);
  let lastPmedianAssignments = new Uint32Array(demandCount);
  let lastScenarioCosts: Float32Array = new Float32Array(demandCount),
    lastScenarioFacilities: Uint32Array = new Uint32Array(demandCount),
    lastRationalScore = NaN;
  let lastRationalWinners: Uint32Array = new Uint32Array(demandCount);
  type RationalResult = {
    score: number;
    unserved: number;
    status: Uint32Array;
    winners: Uint32Array;
    assignments: Float32Array;
  };
  let lastRationalResults: RationalResult[] = [];
  let placementResults: {
    pmedianObjective: number;
    pmedianStatus: Uint32Array;
    backupObjective: number;
    backupStatus: Uint32Array;
    dispersionObjective: number;
    dispersionStatus: Uint32Array;
    costsConverged: boolean;
  } | null = null;
  type ScenarioResult = {
    covered: number;
    uncovered: number;
    invalid: number;
    costs: Float32Array;
    facilities: Uint32Array;
  };
  let parsedScenarioResults: ScenarioResult[] = [];
  let rationalReady = false;
  let rationalRoutingReady = false;
  function publishPlacement(): void {
    if (!placementResults) return;
    const {
      pmedianObjective,
      pmedianStatus,
      backupObjective,
      backupStatus,
      dispersionObjective,
      dispersionStatus,
      costsConverged
    } = placementResults;
    const selected =
      ctx.options.objective === 'pmedian'
        ? lastPmedianSelected
        : ctx.options.objective === 'backup'
          ? lastBackupSelected
          : lastDispersionSelected;
    const status =
      ctx.options.objective === 'pmedian'
        ? pmedianStatus
        : ctx.options.objective === 'backup'
          ? backupStatus
          : dispersionStatus;
    const selectedNames = HUB_NAMES.filter((_, index) => selected[index] !== 0)
      .map(name => name.replace(' Centraal', ''))
      .join(' + ');
    const objective =
      ctx.options.objective === 'pmedian'
        ? `${pmedianObjective.toFixed(1)} station-minutes`
        : ctx.options.objective === 'backup'
          ? `${backupObjective.toFixed(0)} / ${demandCount} double-covered stations`
          : `${dispersionObjective.toFixed(1)} min minimum separation`;
    const converged = costsConverged && status[0] === 1 && status[1] === 1 && status[2] === 1;
    outputReady = converged;
    ctx.setReadout(
      'objectiveResult',
      converged
        ? `${selectedNames} · ${objective}`
        : costsConverged && status[0] === 1 && status[2] === 1
          ? 'No feasible placement under this threshold'
          : 'withheld until exact search completes'
    );
    ctx.setReadout(
      'optimizerStatus',
      converged
        ? `enumerated ${status[3].toLocaleString()} candidates · feasible incumbent`
        : status[0] === 1 && status[2] === 1
          ? `search complete · no feasible incumbent`
          : 'cost search or optimizer incomplete; result withheld'
    );
  }
  function publishRational(): void {
    if (lastRationalResults.length !== 4) return;
    const rational =
      lastRationalResults[
        (ctx.options.rationalMode === 'fractional' ? 2 : 0) +
          (ctx.options.stepMode === 'absolute' ? 1 : 0)
      ];
    rationalReady = rationalRoutingReady && rational.status[0] === 1;
    if (!rationalReady) {
      lastRationalScore = NaN;
      ctx.setReadout('rationalSummary', 'invalid or incomplete allocation; result withheld');
      ctx.setReadout(
        'rationalStatus',
        `${rational.status[2]} cycles · invalid input or bounded run`
      );
      return;
    }
    lastRationalScore = rational.score;
    lastRationalWinners = rational.winners;
    const facilityLoads = Array.from({length: facilityCount}, (_, facility) =>
      rational.assignments.reduce(
        (sum, amount, index) => sum + (index % facilityCount === facility ? amount : 0),
        0
      )
    );
    ctx.setReadout(
      'rationalLoads',
      facilityLoads
        .map((load, index) => `${HUB_NAMES[index].replace(' Centraal', '')} ${load.toFixed(1)}`)
        .join(' · ')
    );
    ctx.setReadout(
      'rationalSummary',
      `${rational.score.toFixed(2)} mean access score · ${rational.unserved.toFixed(1)} unserved units`
    );
    const maximumGap = new Float32Array(rational.status.buffer)[3];
    ctx.setReadout(
      'rationalStatus',
      `${rational.status[2]} cycles · maximum gap ${maximumGap.toFixed(3)} · ${rational.status[1] ? 'within tolerance' : 'bounded or rounded before tolerance'}`
    );
  }
  function publishScenario(): void {
    if (!placementResults || parsedScenarioResults.length !== 3) return;
    const index =
      ctx.options.objective === 'backup' ? 1 : ctx.options.objective === 'dispersion' ? 2 : 0;
    const status =
      index === 0
        ? placementResults.pmedianStatus
        : index === 1
          ? placementResults.backupStatus
          : placementResults.dispersionStatus;
    const scenario = parsedScenarioResults[index];
    lastScenarioCosts = scenario.costs;
    lastScenarioFacilities = scenario.facilities;
    ctx.setReadout(
      'scenarioSummary',
      placementResults.costsConverged && status[0] === 1 && status[1] === 1 && status[2] === 1
        ? `${scenario.covered.toFixed(0)} units within ${ctx.options.scenarioLimitMinutes} min · ${scenario.uncovered.toFixed(0)} outside · ${scenario.invalid} invalid`
        : 'no feasible selected pair'
    );
  }
  const summaryBuffers = [
    demandCostStatus,
    hubDistanceStatus,
    pmedian.selected,
    pmedian.assignments,
    pmedian.objective,
    pmedian.status,
    backup.selected,
    backup.assignments,
    backup.objective,
    backup.status,
    dispersion.selected,
    dispersion.assignments,
    dispersion.objective,
    dispersion.status,
    ...scenarioOutputs.flatMap(output => [
      output.covered,
      output.uncovered,
      output.invalid,
      output.nearestCosts,
      output.nearestFacilities
    ]),
    ...rationalOutputs.flatMap(output => [
      output.scores,
      output.unserved,
      output.status,
      output.winners,
      output.assignments
    ])
  ];
  const reader = new SummaryReader(
    resources,
    'rail-support-allocation-summary',
    summaryBuffers.map(buffer => ({buffer, size: buffer.byteLength})),
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      let offset = 0;
      const readWords = (length: number) => {
        const result = new Uint32Array(bytes).slice(offset / 4, offset / 4 + length);
        offset += length * 4;
        return result;
      };
      const readFloats = (length: number) => {
        const result = new Float32Array(bytes).slice(offset / 4, offset / 4 + length);
        offset += length * 4;
        return result;
      };
      const demandCostsConverged = readWords(1)[0] === 1;
      const hubsConverged = readWords(1)[0] === 1;
      lastPmedianSelected = readWords(facilityCount);
      lastPmedianAssignments = readWords(demandCount);
      const pmedianObjective = readFloats(1)[0];
      const pmedianStatus = readWords(4);
      lastBackupSelected = readWords(facilityCount);
      readWords(demandCount);
      const backupObjective = readFloats(1)[0];
      const backupStatus = readWords(4);
      lastDispersionSelected = readWords(facilityCount);
      readWords(demandCount);
      const dispersionObjective = readFloats(1)[0];
      const dispersionStatus = readWords(4);
      parsedScenarioResults = scenarioOutputs.map(
        (): ScenarioResult => ({
          covered: 0,
          uncovered: 0,
          invalid: 0,
          costs: new Float32Array(demandCount),
          facilities: new Uint32Array(demandCount)
        })
      );
      for (const result of parsedScenarioResults) {
        result.covered = readFloats(1)[0];
        result.uncovered = readFloats(1)[0];
        result.invalid = readWords(1)[0];
        result.costs = readFloats(demandCount);
        result.facilities = readWords(demandCount);
      }
      rationalRoutingReady = demandCostsConverged;
      lastRationalResults = [];
      for (let index = 0; index < 4; index++) {
        const scores = readFloats(demandCount);
        const unserved = readFloats(demandCount);
        const status = readWords(4);
        const winners = readWords(demandCount);
        const assignments = readFloats(demandCount * facilityCount);
        const finiteScores = scores.filter(Number.isFinite);
        lastRationalResults.push({
          score:
            finiteScores.reduce((sum, value) => sum + value, 0) / Math.max(1, finiteScores.length),
          unserved: unserved.reduce((sum, value) => sum + value, 0),
          status,
          winners,
          assignments
        });
      }
      placementResults = {
        pmedianObjective,
        pmedianStatus,
        backupObjective,
        backupStatus,
        dispersionObjective,
        dispersionStatus,
        costsConverged: demandCostsConverged && hubsConverged
      };
      publishPlacement();
      publishScenario();
      publishRational();
      ctx.requestLayers();
    }
  );
  const accessSettings = () => {
    const values = rationalParametersBuffer;
    values.write(
      Float32Array.of(ctx.options.travelScaleMinutes, ctx.options.supplyScale, 0.4, 0.02, 30, 0.01)
    );
    rationalDemandBuffer.write(new Float32Array(demandCount).fill(ctx.options.demandMultiplier));
    accessibilityParametersBuffer.write(
      getGPUNetworkAccessibilityParameterValues({
        threshold: ctx.options.scenarioLimitMinutes,
        decay: 'exponential',
        beta: 0.03
      })
    );
  };
  function update() {
    thresholdBuffer.write(Float32Array.of(ctx.options.thresholdMinutes));
    accessSettings();
    generation++;
    dirty = true;
    outputReady = false;
    rationalReady = false;
    rationalRoutingReady = false;
    placementResults = null;
    lastRationalResults = [];
    parsedScenarioResults = [];
    reader.markStale();
    ctx.setReadout('objectiveResult', 'updating GPU analysis');
    ctx.setReadout('scenarioSummary', 'updating GPU evaluation');
    ctx.setReadout('rationalSummary', 'updating GPU allocation');
    ctx.setReadout('rationalLoads', 'waiting for GPU allocation');
    ctx.setReadout('rationalStatus', 'waiting for GPU allocation');
    ctx.setReadout('optimizerStatus', 'waiting for GPU analysis');
  }
  ctx.setReadout('sampleCount', `${demandCount} named stations · equal unit demand`);
  ctx.setReadout('candidateCount', `${facilityCount} hypothetical desks`);
  ctx.setReadout('travelModel', 'Directed scheduled median time · minutes · perfect transfers');
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (!reader.isPending && generation !== pendingGeneration) pendingGeneration = generation;
      reader.flush(commandEncoder);
    },
    setOption(id) {
      if (
        id === 'thresholdMinutes' ||
        id === 'supplyScale' ||
        id === 'scenarioLimitMinutes' ||
        id === 'demandMultiplier' ||
        id === 'travelScaleMinutes'
      )
        update();
      else {
        if (id === 'objective') {
          publishPlacement();
          publishScenario();
        }
        if (id === 'rationalMode' || id === 'stepMode') publishRational();
        ctx.requestLayers();
      }
    },
    getLayers() {
      const selectedBuffer =
        ctx.options.allocationDisplay === 'rational'
          ? rationalCandidateMaskBuffer
          : ctx.options.objective === 'pmedian'
            ? pmedian.selected
            : ctx.options.objective === 'backup'
              ? backup.selected
              : dispersion.selected;
      const rationalIndex =
        (ctx.options.rationalMode === 'fractional' ? 2 : 0) +
        (ctx.options.stepMode === 'absolute' ? 1 : 0);
      const assignmentBuffer =
        ctx.options.allocationDisplay === 'rational'
          ? rationalOutputs[rationalIndex].winners
          : ctx.options.allocationDisplay === 'scenario' || ctx.options.objective === 'backup'
            ? scenarioOutputs[
                ctx.options.objective === 'backup'
                  ? 1
                  : ctx.options.objective === 'dispersion'
                    ? 2
                    : 0
              ].nearestFacilities
            : pmedian.assignments;
      const context = new SpatialAnalysisPointLayer({
        id: 'rail-support-all-stations-context',
        analysisFrame: rail.frame,
        positions: nodePositionsBuffer,
        instanceCount: nodeCount,
        radiusPixels: 2.5,
        color: [122, 134, 144, 95]
      });
      const network = new SpatialAnalysisSegmentLayer({
        id: 'rail-support-all-rail-links',
        analysisFrame: rail.frame,
        segments: railSegmentsBuffer,
        instanceCount: edgeCount,
        widthPixels: 0.65,
        color: [115, 132, 147, 45]
      });
      const showDemandAssignments =
        ctx.options.allocationDisplay === 'rational'
          ? rationalReady
          : outputReady &&
            (ctx.options.allocationDisplay === 'scenario' ||
              ctx.options.objective !== 'dispersion');
      const demandLayer = showDemandAssignments
        ? new SpatialAnalysisPointLayer({
            id: 'rail-support-sampled-assignments',
            analysisFrame: rail.frame,
            positions: demandPositionsBuffer,
            instanceCount: demandCount,
            values: assignmentBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: HUB_COLORS,
            noDataColor: [150, 154, 158, 180],
            radiusPixels: 7,
            outlineColor: [255, 255, 255, 200],
            outlineWidthPixels: 1
          })
        : new SpatialAnalysisPointLayer({
            id: 'rail-support-sampled-pending',
            analysisFrame: rail.frame,
            positions: demandPositionsBuffer,
            instanceCount: demandCount,
            radiusPixels: 6,
            color: [139, 146, 153, 190],
            outlineColor: [255, 255, 255, 180],
            outlineWidthPixels: 1
          });
      const desksReady = ctx.options.allocationDisplay === 'rational' ? rationalReady : outputReady;
      const desks = desksReady
        ? new SpatialAnalysisPointLayer({
            id: 'rail-support-candidate-desks',
            analysisFrame: rail.frame,
            positions: hubPositionsBuffer,
            instanceCount: facilityCount,
            values: selectedBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [251, 194, 69, 255],
            noDataColor: [106, 116, 125, 170],
            radiusPixels: 9,
            outlineColor: [32, 39, 45, 255],
            outlineWidthPixels: 1.6,
            shape: 'star'
          })
        : new SpatialAnalysisPointLayer({
            id: 'rail-support-candidate-desks-pending',
            analysisFrame: rail.frame,
            positions: hubPositionsBuffer,
            instanceCount: facilityCount,
            radiusPixels: 7,
            color: [150, 159, 168, 190],
            outlineColor: [32, 39, 45, 255],
            outlineWidthPixels: 1.4,
            shape: 'star'
          });
      return [network, context, demandLayer, desks] as Layer[];
    },
    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (
        !viewport ||
        (ctx.options.allocationDisplay === 'rational' ? !rationalReady : !outputReady)
      )
        return null;
      let nearest = -1,
        closest = 15 * 15;
      for (let row = 0; row < demandCount; row++) {
        const point = viewport.project([
          rail.nodeLngLat[demandNodes[row] * 2],
          rail.nodeLngLat[demandNodes[row] * 2 + 1]
        ]);
        const distance = (point[0] - event.pixel[0]) ** 2 + (point[1] - event.pixel[1]) ** 2;
        if (distance < closest) {
          closest = distance;
          nearest = row;
        }
      }
      if (nearest < 0) return null;
      const rationalIndex =
        (ctx.options.rationalMode === 'fractional' ? 2 : 0) +
        (ctx.options.stepMode === 'absolute' ? 1 : 0);
      const isRational = ctx.options.allocationDisplay === 'rational';
      const showScenario =
        ctx.options.allocationDisplay === 'scenario' || ctx.options.objective === 'backup';
      const assignment = isRational
        ? lastRationalWinners[nearest]
        : showScenario
          ? lastScenarioFacilities[nearest]
          : lastPmedianAssignments[nearest];
      return {
        title: DEMAND_NAMES[nearest],
        subtitle: `Sample station · ${isRational ? ctx.options.demandMultiplier : 1} illustrative demand units`,
        rows: [
          {
            label: isRational
              ? 'Largest allocation share'
              : showScenario
                ? 'Nearest selected desk'
                : 'Placement assignment',
            value:
              ctx.options.objective === 'dispersion' && !isRational && !showScenario
                ? 'not applicable to p-dispersion'
                : assignment < facilityCount
                  ? HUB_NAMES[assignment].replace(' Centraal', '')
                  : 'not assigned'
          },
          {label: 'Selected objective', value: ctx.options.objective},
          {
            label: 'Facility scenario nearest cost',
            value: Number.isFinite(lastScenarioCosts[nearest])
              ? `${lastScenarioCosts[nearest].toFixed(1)} min`
              : 'unreachable'
          },
          {
            label: 'Rational-access mean score',
            value: Number.isFinite(lastRationalScore) ? lastRationalScore.toFixed(2) : 'pending'
          },
          {
            label: 'Allocation by desk',
            value:
              isRational && lastRationalResults.length
                ? HUB_NAMES.map(
                    (name, facility) =>
                      `${name.replace(' Centraal', '')} ${lastRationalResults[rationalIndex].assignments[nearest * facilityCount + facility].toFixed(2)}`
                  ).join(' · ')
                : 'switch map to rational allocation'
          }
        ],
        note: 'Station costs come from the GPU rail cost matrix. Demand and supply are explicit illustrative assumptions, and perfect transfers are assumed.'
      };
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
