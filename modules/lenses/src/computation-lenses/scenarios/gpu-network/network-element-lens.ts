// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, Buffer, type Device} from '@luma.gl/core';
import {
  createGPUComputationCaptures,
  describeGPUCommandGraph,
  GPUComputationRecorder,
  createWGSLKernelNodeAnnotation,
  type GPUAnalysisChoice,
  type GPUComputationCapture,
  type GPUComputationKeySpace,
  type GPUComputationRelation,
  type GPUComputationRecordWithSidecars
} from '@luma-contributors/analysis/computation-record';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphComputeNode,
  type GPUCommandGraphCopyNode,
  type GraphBufferDescriptor,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {NETWORK_FORMULA_REFS} from '../../../gpu-network/computation-pack/network-formulas';
import {
  assembleNetworkLensRecord,
  createNetworkRelations,
  getNetworkFinalCapture,
  loadChicagoLoopRoads,
  uploadNetworkTopology,
  uploadNetworkView
} from './network-lens-support';
import {assertValidLensSidecars} from '../../validate-lens-record';
import {
  addLensViewSnapshot,
  LENS_RECORD_SOURCE_TEXT,
  readLensRunMetrics,
  readLensViewSnapshot,
  submitTimedLensGraph,
  type LensViewSnapshot
} from '../../lens-view-snapshot';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '@luma-contributors/analysis/analysis-core';

/** The prepared Chicago Loop road extract, or a deterministic transformation of it. */
export type NetworkRoads = ReturnType<typeof loadChicagoLoopRoads>;

/** Everything a scenario's `build` receives; uploads are tracked and destroyed by the runner. */
export type NetworkElementContext = {
  device: Device;
  graph: GPUCommandGraph;
  /** Road data after `prepareRoads`. */
  data: NetworkRoads;
  /**
   * Uploaded CSR, impedance and node positions of `data`. Undefined at run time when the scenario
   * sets `withoutRoads`.
   */
  topology: ReturnType<typeof uploadNetworkTopology>;
  /** True on a null device: pipelines are not created and nothing executes. */
  structureOnly: boolean;
  /** Finds a transient the contributor allocated (for example its scratch counts) by its id. */
  findTransient(id: string): GraphBufferHandle;
  /** Uploads a caller-owned packed input that can be read back. */
  upload<Format extends 'float32' | 'float32x2' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ): GraphDataView<Format>;
};

/** Reads the final value of a named captured view. */
export type NetworkFinalReader = (name: string) => GPUComputationCapture;

/** One element's scenario: what to build, what to read back and how to draw it. */
export type NetworkElementScenario = {
  /** Scenario id, `gpu-network.<kebab>`; also the graph id. */
  id: string;
  /** Impedance uploaded as `topology.weights`. */
  weights: 'travelTimeSeconds' | 'lengthMeters';
  /** Deterministic data preparation (reverse CSR, local projection); omitted: the file as is. */
  prepareRoads?: (data: NetworkRoads) => NetworkRoads;
  /** Coordinates of the uploaded node positions. Defaults to `geographic`. */
  coordinateModel?: 'geographic' | 'projected';
  /**
   * The element does not run on the road graph (noding, bundling, flow aggregation): nothing of
   * the roads is uploaded, read back or related, and `context.topology` is undefined.
   */
  withoutRoads?: boolean;
  /** Seed recorded with the run (a start node or similar). */
  seed: number;
  /** Dataset version recorded with the run. Defaults to the Chicago Loop roads version. */
  datasetVersion?: string;
  /** Transient ids to observe; they receive copy access in the lens graph only. */
  retain: readonly string[];
  /** Formula ids from the network pack; the element's own formula first. */
  formulas: readonly string[];
  /** Key spaces beyond the defaults declared by the scenario. */
  keySpaces: readonly GPUComputationKeySpace[];
  /** Key space of the data axis (the datum rung's entity). */
  entityKeySpace: string;
  /** Additional relations for workflows with multiple separately owned frontiers. */
  relations?: readonly GPUComputationRelation[];
  /** Rename a contributor-local relation by node ID prefix when composing several searches. */
  frontierRelationAliases?: Readonly<Record<string, string>>;
  /** Queue resource of the nested reachability search, for the `frontier-queue` relation. */
  queuesResource?: string;
  /**
   * False leaves the road-graph captures (context and its three readbacks) out of the record, for
   * results that are not drawn on the map. Defaults to true.
   */
  roadContext?: boolean;
  /** Analysis choices recorded with the run. */
  choices: readonly GPUAnalysisChoice[];
  /** Adds the element to the graph (through the shown example) and names the views to read back. */
  build(
    context: NetworkElementContext
  ): Promise<{views: Readonly<Record<string, GraphDataView | GraphBufferHandle>>}>;
  /**
   * Input and output captures from final readbacks; throws when a status readback says the run
   * failed. `captured` holds the end-of-run readback of every view.
   */
  capture(
    final: NetworkFinalReader,
    context: NetworkElementContext,
    captured: readonly GPUComputationCapture[]
  ): GPUComputationCapture[];
};

// COPY_SRC is an observation-only allocation choice; structure-only runs drop compile callbacks
// because null devices cannot create compute pipelines.
class ElementLensGraph extends GPUCommandGraph {
  /** Explicit declaration annotations for the null-device adapter, whose callbacks cannot compile. */
  readonly structureRecorder = new GPUComputationRecorder({sourceText: LENS_RECORD_SOURCE_TEXT});
  readonly structureOnly: boolean;
  /** Every transient by id, so a scenario can observe a contributor's own scratch resource. */
  readonly transients = new Map<string, GraphBufferHandle>();

  readonly observed: ReadonlySet<string>;

  constructor(device: Device, id: string, retain: readonly string[], structureOnly: boolean) {
    super(device, {id});
    this.observed = new Set(retain);
    this.structureOnly = structureOnly;
    // Kernel facts come from createWGSLKernelNode, so structure-only records carry them too.
    recordWGSLKernelNodes(this);
  }

  override createTransientBuffer(descriptor: GraphBufferDescriptor): GraphBufferHandle {
    const handle = super.createTransientBuffer(
      this.observed.has(descriptor.id)
        ? {...descriptor, usage: descriptor.usage | Buffer.COPY_SRC}
        : descriptor
    );
    this.transients.set(descriptor.id, handle);
    return handle;
  }

  override addComputePass(node: Omit<GPUCommandGraphComputeNode<void>, 'type'>): void {
    super.addComputePass(this.structureOnly ? withoutCompilation(node) : node);
    if (this.structureOnly) {
      const kernel = getWGSLKernelNodeRecord(this, node.id);
      if (kernel)
        this.structureRecorder.recordNode(
          {...node, type: 'compute'},
          createWGSLKernelNodeAnnotation(kernel, LENS_RECORD_SOURCE_TEXT)
        );
    }
  }

  override addCopyPass(node: Omit<GPUCommandGraphCopyNode<void>, 'type'>): void {
    super.addCopyPass(this.structureOnly ? withoutCompilation(node) : node);
  }
}

function withoutCompilation<Node extends {compile: unknown; compileAsync?: unknown}>(
  node: Node
): Node {
  const {compile: _compile, compileAsync: _compileAsync, ...structure} = node;
  const unavailable = () => ({
    encode: () => {
      throw new Error('Structure-only graph cannot execute');
    }
  });
  return {...structure, compile: unavailable} as unknown as Node;
}

async function buildNetworkElement(
  device: Device,
  scenario: NetworkElementScenario,
  structureOnly: boolean
) {
  const graph = new ElementLensGraph(device, scenario.id, scenario.retain, structureOnly);
  const data = scenario.prepareRoads?.(loadChicagoLoopRoads()) ?? loadChicagoLoopRoads();
  const ownedBuffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  let snapshot: LensViewSnapshot | undefined;
  try {
    const topology = scenario.withoutRoads
      ? undefined
      : uploadNetworkTopology(device, graph, data, {weights: scenario.weights});
    if (topology) ownedBuffers.push(...topology.ownedBuffers);
    const context: NetworkElementContext = {
      device,
      graph,
      data,
      topology: topology as NetworkElementContext['topology'],
      structureOnly,
      findTransient: id => {
        const handle = graph.transients.get(id);
        // The scenario names a transient its contributor allocates while the graph is built.
        assert(handle);
        return handle;
      },
      upload: (id, values, format) => {
        const uploaded = uploadNetworkView(device, graph, id, values, format);
        ownedBuffers.push(uploaded.buffer);
        return uploaded.view;
      }
    };
    const {views} = await scenario.build(context);
    const {topology: roadTopology} = context;
    snapshot = addLensViewSnapshot(graph, device, {
      id: scenario.id,
      views: {
        ...views,
        ...(roadTopology && {
          offsets: roadTopology.offsets,
          neighbors: roadTopology.neighbors,
          nodePositions: roadTopology.nodePositions
        })
      }
    });
    compiled = structureOnly ? graph.compile() : await graph.compileAsync();
    const own = scenario.formulas.map(id => NETWORK_FORMULA_REFS[id]);
    const description = describeGPUCommandGraph(compiled, {
      domain: 'gpu-network',
      graph,
      recorder: structureOnly ? graph.structureRecorder : undefined,
      sourceText: LENS_RECORD_SOURCE_TEXT
    });
    const graphs = [
      {
        ...description,
        graph: {
          ...description.graph,
          nodes: description.graph.nodes.map(node => {
            const alias =
              node.datum?.relation === 'frontier-queue'
                ? Object.entries(scenario.frontierRelationAliases ?? {}).find(([prefix]) =>
                    node.id.startsWith(prefix)
                  )?.[1]
                : undefined;
            return alias ? {...node, datum: {...node.datum!, relation: alias}} : node;
          })
        }
      }
    ];
    const record = assembleNetworkLensRecord(device, {
      id: scenario.id,
      seed: scenario.seed,
      ...(scenario.datasetVersion && {datasetVersion: scenario.datasetVersion}),
      graphs,
      formulas: [...own, ...Object.values(NETWORK_FORMULA_REFS).filter(ref => !own.includes(ref))],
      choices: scenario.choices,
      axes: {
        axes: [
          {id: 'entity', label: 'element datum', kind: 'data', keySpace: scenario.entityKeySpace}
        ],
        keySpaces: scenario.keySpaces,
        relations: [
          ...(topology ? createNetworkRelations(topology, scenario.queuesResource) : []),
          ...(scenario.relations ?? [])
        ]
      }
    });
    return {compiled, record, context, snapshot, ownedBuffers};
  } catch (error) {
    snapshot?.arena.destroy();
    compiled?.destroy();
    for (const buffer of ownedBuffers) buffer.destroy();
    throw error;
  }
}

/** Builds the scenario graph without executing it; works on `createNullWebGPUDevice()`. */
export async function describeNetworkElement(device: Device, scenario: NetworkElementScenario) {
  const built = await buildNetworkElement(device, scenario, true);
  built.snapshot.arena.destroy();
  built.compiled.destroy();
  for (const buffer of built.ownedBuffers) buffer.destroy();
  return built.record;
}

/**
 * Executes the scenario once, reads back the road context and the scenario's views at the end of
 * the run, and returns the validated record. Everything shown is a GPU readback; resources are
 * destroyed afterwards.
 */
export async function recordNetworkElement(
  device: Device,
  scenario: NetworkElementScenario
): Promise<GPUComputationRecordWithSidecars> {
  const {compiled, record, context, snapshot, ownedBuffers} = await buildNetworkElement(
    device,
    scenario,
    false
  );
  try {
    const {topology} = context;
    const submission = submitTimedLensGraph(device, compiled, {
      parameters: undefined,
      coalesceComputePasses: false
    });
    const timedMetrics = await readLensRunMetrics(submission, {
      runId: record.runId,
      executionId: record.executionId,
      graphId: record.graphs[0].id,
      taps: 1,
      retained: scenario.retain
    });
    const captured = await readLensViewSnapshot(snapshot, {graphId: record.graphs[0].id});
    const final: NetworkFinalReader = name => getNetworkFinalCapture(captured, name);
    const hasRoadContext = topology && scenario.roadContext !== false;
    const roadContext: GPUComputationCapture[] = hasRoadContext
      ? [
          final('nodePositions'),
          final('offsets'),
          final('neighbors'),
          {
            ...final('nodePositions'),
            id: 'road-context',
            name: 'road-context',
            role: 'context',
            keySpace: 'node',
            meaning:
              'context: uploaded road graph (offsets, neighbors and node positions read back)',
            geometry: {
              kind: 'graph',
              offsetsCapture: 'offsets',
              targetsCapture: 'neighbors',
              layoutCapture: 'nodePositions',
              coordinateModel: scenario.coordinateModel ?? 'geographic'
            }
          }
        ]
      : [];
    const snapshots: GPUComputationCapture[] = [
      ...roadContext,
      ...scenario.capture(final, context, captured)
    ];
    const captures = createGPUComputationCaptures({
      runId: record.runId,
      executionId: record.executionId,
      snapshots
    });
    const metrics = [timedMetrics];
    const result = {...record, captures, metrics};
    assertValidLensSidecars(record, result);
    return result;
  } finally {
    snapshot.arena.destroy();
    compiled.destroy();
    for (const buffer of ownedBuffers) buffer.destroy();
  }
}
