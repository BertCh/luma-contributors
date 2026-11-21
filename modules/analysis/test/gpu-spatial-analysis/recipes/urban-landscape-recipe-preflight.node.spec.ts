// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VertexFormat} from '@luma.gl/core';
import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import * as overlayPlanning from '../../../src/gpu-spatial-analysis/polygon-overlay/overlay-planning';
import type {GPUPolygonOverlayOutput} from '../../../src/gpu-spatial-analysis/polygon-overlay/index';
import {
  addBuildingAdjacencyRecipe,
  addBuildingTessellationRecipe,
  addIsovistsRecipe,
  addStreetProfilesRecipe,
  addVisibilityGraphRecipe,
  type GPUBuildingTessellationProps
} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {
  addLandscapeMetricsRecipe,
  addPatchProximityRecipe,
  addPopulationSupportAlignmentRecipe,
  addRasterResistanceConnectivityRecipe,
  addResistanceConnectivityRecipe,
  addSegregationCounterfactualRecipe,
  addSegregationDecompositionRecipe,
  addSparseResistanceConnectivityRecipe
} from '../../../src/gpu-spatial-analysis/landscape-ecology/index';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {resolveRecipeDestinations} from '../../../src/gpu-spatial-analysis/recipes/recipe-utils';
import type {GPUCompletenessStatusPort} from '../../../src/analysis-core/contracts/status';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

// RCP-101 and RCP-102: the 13 public urban and landscape recipes sit outside the every-stage
// rollback matrix in recipe-registry-rollback.node.spec.ts, so each validates by its own
// preflight. These cases pin the property the matrix pins for the recipe barrels: a rejected
// call leaves the caller graph without any added command node.

type ViewFactory = <Format extends VertexFormat>(
  format: Format,
  length: number
) => GraphDataView<Format>;
type Recipe = (graph: GPUCommandGraph, props: never) => unknown;
type Fixture = {
  recipe: Recipe;
  props: (view: ViewFactory) => Record<string, unknown>;
  /** Caller-owned output views the recipe accepts: path below `output`, format and valid length. */
  outputs?: readonly (readonly [string, VertexFormat])[];
};

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const foreignGraph = new GPUCommandGraph(device);
  let serial = 0;
  const view: ViewFactory = (format, length) =>
    createTransientView(graph, `preflight-${serial++}`, format, length);
  const foreignView: ViewFactory = (format, length) =>
    createTransientView(foreignGraph, `foreign-${serial++}`, format, length);
  return {device, graph, view, foreignView};
}

function getNodeCount(graph: GPUCommandGraph): number {
  return (graph as unknown as {nodes: unknown[]}).nodes.length;
}

function createReferenceTessellationProps(
  view: ViewFactory,
  enclosureMode: 'convex' | 'simple' = 'convex'
): GPUBuildingTessellationProps {
  return {
    mode: 'reference',
    segments: view('float32x4', 4),
    segmentBuildingIds: view('uint32', 4),
    buildingCount: 2,
    enclosure: view('float32x2', 4),
    enclosureMode,
    sampleCapacity: 1,
    sampleSpacing: 1,
    vertexTolerance: 1 / 1024,
    capacity: {intersections: 1, nodedSegments: 32, vertices: 32, rings: 2}
  };
}

function createSuppliedOverlay(view: ViewFactory): GPUPolygonOverlayOutput {
  const status = () => ({
    count: view('uint32', 1),
    overflow: view('uint32', 1),
    candidateOverflow: view('uint32', 1)
  });
  return {
    geometry: {
      kind: 'polygons',
      positions: view('float32x2', 8),
      featureOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      ringOffsets: view('uint32', 3)
    },
    status: {...status(), requiredRingCount: view('uint32', 1)},
    boundary: {
      endpoints: view('float32x4', 32),
      operandIds: view('uint32', 32),
      sourceFeatureIds: view('uint32', 32),
      status: {...status(), requiredCount: view('uint32', 1)}
    }
  };
}

function isView(value: unknown): value is {format: VertexFormat; length: number} {
  return (
    value instanceof GraphVectorView ||
    (typeof value === 'object' &&
      value !== null &&
      'buffer' in value &&
      'format' in value &&
      'byteOffset' in value)
  );
}

type Mutation = {name: string; apply: (props: Record<string, unknown>, context: Context) => void};
type Context = ReturnType<typeof createContext>;

const NUMBER_VALUES = [-1, 0, 0.5, Number.NaN, Number.POSITIVE_INFINITY, 2 ** 32 + 1];

function getParent(root: Record<string, unknown>, path: readonly (string | number)[]) {
  let parent = root as Record<string | number, unknown>;
  for (const key of path.slice(0, -1)) {
    // Copy on write keeps tuples and nested option objects unshared between mutations.
    const child = parent[key] as Record<string | number, unknown>;
    parent[key] = Array.isArray(child) ? [...child] : {...child};
    parent = parent[key] as Record<string | number, unknown>;
  }
  return {parent, key: path[path.length - 1]};
}

function setAt(root: Record<string, unknown>, path: readonly (string | number)[], value: unknown) {
  const {parent, key} = getParent(root, path);
  parent[key] = value;
}

/** One invalid option per mutation: every number, string, view and tuple the fixture supplies. */
function getMutations(sample: Record<string, unknown>, fixture: Fixture): Mutation[] {
  const mutations: Mutation[] = [];
  const visit = (value: unknown, path: (string | number)[]) => {
    const name = path.join('.');
    if (isView(value)) {
      const {format, length} = value;
      const other: VertexFormat = format === 'uint32' ? 'float32' : 'uint32';
      mutations.push(
        {
          name: `${name} has one more row`,
          apply: (props, c) => setAt(props, path, c.view(format, length + 1))
        },
        {
          name: `${name} has another format`,
          apply: (props, c) => setAt(props, path, c.view(other, length))
        },
        {
          name: `${name} belongs to another graph`,
          apply: (props, c) => setAt(props, path, c.foreignView(format, length))
        }
      );
      return;
    }
    if (typeof value === 'number') {
      for (const number of NUMBER_VALUES) {
        mutations.push({name: `${name} = ${number}`, apply: props => setAt(props, path, number)});
      }
      return;
    }
    if (typeof value === 'string') {
      for (const text of ['', 'unknown-option']) {
        mutations.push({name: `${name} = '${text}'`, apply: props => setAt(props, path, text)});
      }
      return;
    }
    if (Array.isArray(value)) {
      mutations.push({
        name: `${name} without its last entry`,
        apply: props => setAt(props, path, value.slice(0, -1))
      });
      value.forEach((child, index) => visit(child, [...path, index]));
      return;
    }
    if (value && typeof value === 'object') {
      for (const [key, child] of Object.entries(value)) visit(child, [...path, key]);
    }
  };
  visit(sample, []);
  // Caller-owned outputs: a wrong row count, a wrong format, and a view aliasing an input.
  for (const [outputPath, format] of fixture.outputs ?? []) {
    const path = ['output', ...outputPath.split('.')];
    const other: VertexFormat = format === 'uint32' ? 'float32' : 'uint32';
    mutations.push(
      {
        name: `output.${outputPath} has 997 rows`,
        apply: (props, c) => {
          props['output'] ??= {};
          setAt(props, path, c.view(format, 997));
        }
      },
      {
        name: `output.${outputPath} has another format`,
        apply: (props, c) => {
          props['output'] ??= {};
          setAt(props, path, c.view(other, 1));
        }
      },
      {
        name: `output.${outputPath} belongs to another graph`,
        apply: (props, c) => {
          props['output'] ??= {};
          setAt(props, path, c.foreignView(format, 1));
        }
      }
    );
  }
  return mutations;
}

const SHARED_PATHS = (view: ViewFactory) => ({
  runs: {
    positions: view('float32x2', 2),
    offsets: view('uint32', 2),
    leftSourceIds: view('uint32', 1),
    rightSourceIds: view('uint32', 1),
    forward: view('uint32', 1)
  },
  status: {count: view('uint32', 1)}
});

const FIXTURES: Record<string, Fixture> = {
  isovists: {
    recipe: addIsovistsRecipe as Recipe,
    props: view => ({
      positions: view('float32x2', 2),
      segments: view('float32x4', 4),
      rayCount: 8,
      maximumDistance: 10
    }),
    outputs: [
      ['endpoints', 'float32x2'],
      ['distances', 'float32'],
      ['segments', 'uint32'],
      ['validity', 'uint32'],
      ['localMeasures', 'float32x4'],
      ['angles', 'float32'],
      ['viewpointClassifications', 'uint32']
    ]
  },
  'visibility graph reference': {
    recipe: addVisibilityGraphRecipe as Recipe,
    props: view => ({
      positions: view('float32x2', 4),
      segments: view('float32x4', 4),
      maximumDistance: 10,
      edgeCapacity: 8
    }),
    outputs: [
      ['adjacency', 'uint32'],
      ['edges', 'uint32x2'],
      ['metrics', 'float32x4'],
      ['choice', 'float32'],
      ['integration', 'float32x4'],
      ['normalizedChoice', 'float32'],
      ['viewpointClassifications', 'uint32'],
      ['extendedIntegration', 'float32x4'],
      ['status', 'uint32x4']
    ]
  },
  'visibility graph csr': {
    recipe: addVisibilityGraphRecipe as Recipe,
    props: view => ({
      mode: 'csr',
      positions: view('float32x2', 4),
      segments: view('float32x4', 4),
      maximumDistance: 10,
      edgeCapacity: 16
    }),
    outputs: [
      ['offsets', 'uint32'],
      ['neighbors', 'uint32'],
      ['weights', 'float32'],
      ['degree', 'uint32'],
      ['gridVisits', 'uint32'],
      ['viewpointClassifications', 'uint32'],
      ['status.count', 'uint32'],
      ['status.requiredCount', 'uint32'],
      ['status.overflow', 'uint32'],
      ['status.candidateOverflow', 'uint32'],
      ['status.invalidCount', 'uint32']
    ]
  },
  'building tessellation reference': {
    recipe: addBuildingTessellationRecipe as Recipe,
    props: view => ({
      segments: view('float32x4', 4),
      segmentBuildingIds: view('uint32', 4),
      buildingCount: 2,
      enclosure: view('float32x2', 4),
      sampleCapacity: 8,
      sampleSpacing: 1,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 64, rings: 16}
    }),
    outputs: [
      ['samples', 'float32x2'],
      ['owners', 'uint32'],
      ['fragments', 'float32x2'],
      ['fragmentCounts', 'uint32'],
      ['samplingStatus', 'uint32x4'],
      ['complete', 'uint32']
    ]
  },
  'building tessellation simple enclosure': {
    recipe: addBuildingTessellationRecipe as Recipe,
    props: view => ({
      segments: view('float32x4', 4),
      segmentBuildingIds: view('uint32', 4),
      buildingCount: 2,
      enclosure: view('float32x2', 4),
      enclosureMode: 'simple',
      sampleCapacity: 8,
      sampleSpacing: 1,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 64, nodedSegments: 128, vertices: 64, rings: 16}
    }),
    outputs: [
      ['samples', 'float32x2'],
      ['owners', 'uint32'],
      ['fragments', 'float32x2'],
      ['fragmentCounts', 'uint32'],
      ['samplingStatus', 'uint32x4'],
      ['complete', 'uint32']
    ]
  },
  'building tessellation local': {
    recipe: addBuildingTessellationRecipe as Recipe,
    props: view => ({
      segments: view('float32x4', 8),
      segmentBuildingIds: view('uint32', 8),
      enclosure: view('float32x2', 4),
      buildingCount: 2,
      sampleSpacing: 1,
      sampleCapacity: 8,
      candidateRadius: 2.75,
      vertexTolerance: 1 / 1024
    }),
    outputs: [
      ['positions', 'float32x2'],
      ['cells', 'uint32x4'],
      ['buildingStatus', 'uint32x4'],
      ['samplingStatus', 'uint32x4'],
      ['status.count', 'uint32'],
      ['status.requiredCount', 'uint32'],
      ['status.overflow', 'uint32'],
      ['status.candidateOverflow', 'uint32'],
      ['status.invalidCount', 'uint32']
    ]
  },
  'street profiles': {
    recipe: addStreetProfilesRecipe as Recipe,
    props: view => ({
      positions: view('float32x2', 2),
      normals: view('float32x2', 2),
      segments: view('float32x4', 8),
      segmentBuildingIds: view('uint32', 8),
      buildingHeights: view('float32', 2),
      maximumDistance: 10,
      sampleSpacing: 1,
      streetSegment: view('float32x4', 1)
    }),
    outputs: [
      ['profiles', 'float32x4'],
      ['buildings', 'uint32x2'],
      ['validity', 'uint32'],
      ['canyon', 'float32x4'],
      ['frontage', 'float32x2'],
      ['facadeLengths', 'float32x2'],
      ['facadeFrontage', 'float32x2']
    ]
  },
  'building adjacency': {
    recipe: addBuildingAdjacencyRecipe as Recipe,
    props: view => ({
      sharedPaths: SHARED_PATHS(view),
      lineBuildingIds: view('uint32', 1),
      orientations: view('float32', 2)
    }),
    outputs: [
      ['pairs', 'float32x4'],
      ['lengths', 'float32'],
      ['morphology', 'float32x4'],
      ['components', 'uint32'],
      ['complete', 'uint32'],
      ['structurePerimeters', 'float32']
    ]
  },
  'building adjacency with structures': {
    recipe: addBuildingAdjacencyRecipe as Recipe,
    props: view => ({
      sharedPaths: SHARED_PATHS(view),
      lineBuildingIds: view('uint32', 1),
      orientations: view('float32', 2),
      contiguity: view('uint32x2', 1),
      structures: {
        geometry: {
          kind: 'polygons',
          positions: view('float32x2', 8),
          featureOffsets: view('uint32', 3),
          polygonOffsets: view('uint32', 3),
          ringOffsets: view('uint32', 3)
        },
        vertexTolerance: 1 / 1024,
        capacity: {intersections: 16, nodedSegments: 32, vertices: 32, rings: 4}
      }
    }),
    outputs: [
      ['components', 'uint32'],
      ['morphology', 'float32x4'],
      ['structurePerimeters', 'float32']
    ]
  },
  'landscape metrics': {
    recipe: addLandscapeMetricsRecipe as Recipe,
    props: view => ({
      patches: view('uint32', 4),
      patchClasses: view('uint32', 2),
      width: 2,
      height: 2,
      classCount: 2,
      cellSize: [1, 1],
      connectivity: 4
    }),
    outputs: [
      ['patchMetrics', 'float32x4'],
      ['classMetrics', 'float32x4'],
      ['classAdjacency', 'uint32'],
      ['aggregation', 'float32'],
      ['landscape', 'float32x4'],
      ['landscapeAggregation', 'float32'],
      ['landscapeCohesion', 'float32'],
      ['reconciledPatches', 'uint32'],
      ['patchParents', 'uint32']
    ]
  },
  'patch proximity reference': {
    recipe: addPatchProximityRecipe as Recipe,
    props: view => ({
      patches: view('uint32', 4),
      patchClasses: view('uint32', 2),
      width: 2,
      height: 2,
      cellSize: [1, 1],
      edgeDepth: 1,
      searchRadius: 1
    }),
    outputs: [
      ['core', 'uint32'],
      ['distances', 'float32'],
      ['metrics', 'float32x4'],
      ['horizontal', 'float32'],
      ['fields', 'float32'],
      ['reconciledPatches', 'uint32'],
      ['patchParents', 'uint32']
    ]
  },
  'patch proximity boundary': {
    recipe: addPatchProximityRecipe as Recipe,
    props: view => ({
      mode: 'boundary',
      patches: view('uint32', 4),
      patchClasses: view('uint32', 2),
      width: 2,
      height: 2,
      cellSize: [1, 1],
      edgeDepth: 1,
      searchRadius: 1
    }),
    outputs: [
      ['core', 'uint32'],
      ['metrics', 'float32x4'],
      ['validity', 'uint32'],
      ['reconciledPatches', 'uint32'],
      ['patchParents', 'uint32'],
      ['status.count', 'uint32'],
      ['status.requiredCount', 'uint32'],
      ['status.overflow', 'uint32'],
      ['status.candidateOverflow', 'uint32']
    ]
  },
  'segregation decomposition': {
    recipe: addSegregationDecompositionRecipe as Recipe,
    props: view => ({
      populations: [
        view('float32x2', 4),
        view('float32x2', 4),
        view('float32x2', 4),
        view('float32x2', 4)
      ],
      supportId: 'common',
      counterfactualMethod: 'supplied',
      randomization: {replicates: 8, seed: 1, swaps: view('uint32', 32)}
    }),
    outputs: [
      ['indices', 'float32'],
      ['validity', 'uint32'],
      ['decomposition', 'float32x4'],
      ['distribution', 'float32'],
      ['inference', 'float32x4']
    ]
  },
  'segregation counterfactual': {
    recipe: addSegregationCounterfactualRecipe as Recipe,
    props: view => ({
      baseline: view('float32x2', 4),
      final: view('float32x2', 4),
      supportId: 'common',
      method: 'share'
    }),
    outputs: [
      ['indices', 'float32'],
      ['validity', 'uint32'],
      ['decomposition', 'float32x4'],
      ['compositionOnly', 'float32x2'],
      ['spatialOnly', 'float32x2']
    ]
  },
  'population support alignment': {
    recipe: addPopulationSupportAlignmentRecipe as Recipe,
    props: view => ({
      populations: view('float32x2', 2),
      sourceIds: view('uint32', 2),
      targetIds: view('uint32', 2)
    }),
    outputs: [
      ['populations', 'float32x2'],
      ['alignmentCode', 'uint32'],
      ['matchedMask', 'uint32'],
      ['status', 'uint32x4']
    ]
  },
  'resistance connectivity': {
    recipe: addResistanceConnectivityRecipe as Recipe,
    props: view => ({
      edges: view('uint32x2', 2),
      conductances: view('float32', 2),
      vertexCount: 3,
      source: 0,
      sink: 2
    }),
    outputs: [
      ['potentials', 'float32'],
      ['currents', 'float32'],
      ['diagnostics', 'float32x4'],
      ['status', 'uint32x4']
    ]
  },
  'sparse resistance connectivity': {
    recipe: addSparseResistanceConnectivityRecipe as Recipe,
    props: view => ({
      edges: view('uint32x2', 2),
      conductances: view('float32', 2),
      vertexCount: 3,
      source: 0,
      sink: 2
    }),
    outputs: [
      ['potentials', 'float32'],
      ['currents', 'float32'],
      ['vertexCurrents', 'float32'],
      ['diagnostics', 'float32x4'],
      ['status', 'uint32x4']
    ]
  },
  'raster resistance connectivity': {
    recipe: addRasterResistanceConnectivityRecipe as Recipe,
    props: view => ({
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
      resistance: {
        id: 'resistance',
        format: 'float32',
        storage: {kind: 'buffer', values: view('float32', 4)},
        validity: view('uint32', 4)
      },
      parameters: view('float32', 8),
      source: 0,
      sink: 3,
      output: {
        potentials: view('float32', 4),
        vertexCurrents: view('float32', 4),
        diagnostics: view('float32x4', 1),
        solverStatus: view('uint32x4', 1)
      }
    }),
    outputs: [
      ['currents', 'float32'],
      ['edges', 'uint32x2'],
      ['conductances', 'float32']
    ]
  }
};

const DESTINATIONS = [
  ['isovists', 'localMeasures', 'float32x4', 2, 'outputs.localMeasures'],
  ['visibility graph reference', 'adjacency', 'uint32', 16, 'outputs.adjacency'],
  ['visibility graph csr', 'offsets', 'uint32', 5, 'outputs.offsets'],
  ['building tessellation reference', 'samples', 'float32x2', 8, 'intermediates.samples'],
  ['building tessellation local', 'samplingStatus', 'uint32x4', 1, 'status.sampling'],
  ['street profiles', 'profiles', 'float32x4', 2, 'outputs.profiles'],
  ['building adjacency', 'pairs', 'float32x4', 4, 'outputs.pairs'],
  ['landscape metrics', 'landscape', 'float32x4', 1, 'outputs.landscape'],
  ['patch proximity reference', 'metrics', 'float32x4', 2, 'outputs.metrics'],
  ['patch proximity boundary', 'metrics', 'float32x4', 2, 'outputs.metrics'],
  ['segregation decomposition', 'indices', 'float32', 4, 'outputs.indices'],
  ['segregation counterfactual', 'compositionOnly', 'float32x2', 4, 'outputs.compositionOnly'],
  ['resistance connectivity', 'potentials', 'float32', 3, 'outputs.potentials'],
  ['sparse resistance connectivity', 'potentials', 'float32', 3, 'outputs.potentials'],
  ['population support alignment', 'matchedMask', 'uint32', 2, 'outputs.matchedMask'],
  ['raster resistance connectivity', 'potentials', 'float32', 4, 'outputs.potentials']
] as const;

it('R1 S5 raster resistance rejects required destinations split across maps', () => {
  const context = createContext();
  try {
    const fixture = FIXTURES['raster resistance connectivity'];
    const props = fixture.props(context.view);
    const output = props['output'] as Record<string, unknown>;
    props['output'] = {potentials: output['potentials'], diagnostics: output['diagnostics']};
    props['outputs'] = {
      vertexCurrents: output['vertexCurrents'],
      solverStatus: output['solverStatus']
    };
    expect(() => fixture.recipe(context.graph, props as never)).toThrow(/outputs.*use output/);
  } finally {
    context.device.destroy();
  }
});

it('R1 B3 S2 captures shared inherited status destinations once', () => {
  const context = createContext();
  try {
    const fixture = FIXTURES['patch proximity boundary'];
    const count = context.view('uint32', 1);
    let reads = 0;
    const status = Object.create(
      Object.defineProperty({}, 'count', {
        get() {
          reads++;
          return count;
        }
      })
    );
    fixture.recipe(context.graph, {
      ...fixture.props(context.view),
      output: {status}
    } as never);
    expect(reads).toBe(1);
  } finally {
    context.device.destroy();
  }
});

it('R1 B1 visibility preflights output storage before mutation', () => {
  const context = createContext();
  try {
    const fixture = FIXTURES['visibility graph reference'];
    const props = fixture.props(context.view);
    const buffer = context.graph.createTransientBuffer({
      id: 'copy-only',
      byteLength: 64,
      usage: Buffer.COPY_DST
    });
    props['output'] = {
      adjacency: context.graph.createDataView(buffer, {format: 'uint32', length: 16})
    };
    const before = [
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ];
    expect(() => fixture.recipe(context.graph, props as never)).toThrow();
    expect(getNodeCount(context.graph)).toBe(0);
    expect([
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ]).toEqual(before);
  } finally {
    context.device.destroy();
  }
});

it('R1 B1 retired viewpoint destinations reject before BVH construction', () => {
  const context = createContext();
  try {
    const fixture = FIXTURES['visibility graph reference'];
    const props = {
      ...fixture.props(context.view),
      obstacleIndex: 'bvh',
      output: {viewpointClassifications: context.view('uint32', 4)},
      outputs: {viewpointClassifications: context.view('uint32', 4)}
    };
    expect(() => fixture.recipe(context.graph, props as never)).toThrow(/outputs.*use output/);
    expect(getNodeCount(context.graph)).toBe(0);
  } finally {
    context.device.destroy();
  }
});

it.each([
  'landscape metrics',
  'patch proximity reference',
  'patch proximity boundary'
])('R1 B3 B4 %s retains inherited inputs and publishes zero-based provenance', name => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    const original = fixture.props(context.view);
    delete original['patches'];
    const converged = context.view('uint32', 1),
      overflow = context.view('uint32', 1);
    const labels = context.view('uint32', 4);
    const denseLabels = Object.create({
      grid: Object.create({
        width: 2,
        height: 2,
        affine: [1, 0, 0, 0, 1, 0],
        pixelInterpretation: 'area'
      }),
      labels,
      status: Object.create({converged, overflow})
    });
    const props = Object.create({...original, denseLabels});
    const result = fixture.recipe(context.graph, props as never) as {
      inputStatus: {converged: unknown};
      provenance: {patches: GraphDataView; denseLabels: unknown};
    };
    expect(result.inputStatus.converged === converged).toBe(true);
    expect(result.provenance.patches === labels).toBe(false);
    expect(result.provenance.patches.buffer.id).toContain('dense-input-owners');
    expect((result.provenance.denseLabels as {labels: unknown}).labels === labels).toBe(true);
    const kernels = getGPUCommandNodeKernelSources(
      (context.graph as unknown as {nodes: unknown[]}).nodes
    );
    expect(kernels.some(kernel => kernel.id.includes('unresolved'))).toBe(true);
  } finally {
    context.device.destroy();
  }
});

it('R1 B3 legacy inherited landscape patchClasses remains accepted', () => {
  const context = createContext();
  try {
    const fixture = FIXTURES['landscape metrics'];
    expect(() =>
      fixture.recipe(context.graph, Object.create(fixture.props(context.view)) as never)
    ).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

it('R1 B6 dense boundary labels preserve disjoint packed destinations', () => {
  const context = createContext();
  try {
    const fixture = FIXTURES['patch proximity boundary'];
    const props = fixture.props(context.view);
    const labels = props['patches'];
    delete props['patches'];
    props['denseLabels'] = {
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      labels
    };
    const buffer = context.graph.createTransientBuffer({
      id: 'packed',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const core = context.graph.createDataView(buffer, {format: 'uint32', length: 4});
    const count = context.graph.createDataView(buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    props['output'] = {core, status: {count}};
    expect(() => fixture.recipe(context.graph, props as never)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

it.each([
  'building tessellation reference',
  'building tessellation local',
  'visibility graph csr'
])('R1 B7 %s accepts inherited recipe settings', name => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    expect(() =>
      fixture.recipe(context.graph, Object.create(fixture.props(context.view)) as never)
    ).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

it.each(['inherited', 'nonenumerable'])('R1 B7 isovists captures %s destinations', kind => {
  const context = createContext();
  try {
    const endpoints = context.view('float32x2', 16);
    const output =
      kind === 'inherited'
        ? Object.create({endpoints})
        : Object.defineProperty({}, 'endpoints', {value: endpoints});
    const result = addIsovistsRecipe(context.graph, {
      ...FIXTURES['isovists'].props(context.view),
      output
    } as never);
    expect(result.outputs.endpoints === endpoints).toBe(true);
  } finally {
    context.device.destroy();
  }
});

it('R1 S1 cellSize rejects extra entries before copying', () => {
  const context = createContext();
  try {
    const fixture = FIXTURES['landscape metrics'];
    expect(() =>
      fixture.recipe(context.graph, {...fixture.props(context.view), cellSize: [1, 1, 99]} as never)
    ).toThrow(/cellSize/);
    expect(getNodeCount(context.graph)).toBe(0);
  } finally {
    context.device.destroy();
  }
});

it.each([
  ['building adjacency', 'structurePerimeters', 'float32', 2],
  ['street profiles', 'facadeLengths', 'float32x2', 8],
  ['landscape metrics', 'patchParents', 'uint32', 2],
  ['patch proximity reference', 'patchParents', 'uint32', 2],
  ['patch proximity boundary', 'patchParents', 'uint32', 2],
  ['segregation decomposition', 'distribution', 'float32', 8]
] as const)('R1 S2 %s rejects inactive %s conflicts', (name, port, format, length) => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    const props = fixture.props(context.view);
    delete props['streetSegment'];
    delete props['randomization'];
    props['output'] = {[port]: context.view(format, length)};
    props['outputs'] = {[port]: context.view(format, length)};
    expect(() => fixture.recipe(context.graph, props as never)).toThrow(/outputs.*use output/);
    expect(getNodeCount(context.graph)).toBe(0);
  } finally {
    context.device.destroy();
  }
});

it.each([
  ['visibility graph reference', 'searchStructure'],
  ['isovists', 'obstacleIndex'],
  ['building tessellation reference', 'enclosureMode'],
  ['building tessellation reference', 'searchStructure']
])('R1 S4 %s rejects unknown %s with the received value', (name, option) => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    expect(() =>
      fixture.recipe(context.graph, {
        ...fixture.props(context.view),
        [option]: 'unknown-option'
      } as never)
    ).toThrow(/unknown-option/);
    expect(getNodeCount(context.graph)).toBe(0);
  } finally {
    context.device.destroy();
  }
});

it('R1 unfinished population alignment preserves output.matchedMask', () => {
  const context = createContext();
  try {
    const matchedMask = context.view('uint32', 2);
    const result = addPopulationSupportAlignmentRecipe(context.graph, {
      populations: context.view('float32x2', 2),
      sourceIds: context.view('uint32', 2),
      targetIds: context.view('uint32', 2),
      output: {matchedMask}
    } as never);
    expect(result.outputs.matchedMask === matchedMask).toBe(true);
  } finally {
    context.device.destroy();
  }
});

it.each([
  ['visibility graph reference', 'mode', 'reference.*csr'],
  ['building tessellation reference', 'mode', 'reference.*local'],
  ['patch proximity reference', 'mode', 'reference.*boundary'],
  ['patch proximity reference', 'distanceModel', 'cell-boundary.*cell-center'],
  ['patch proximity reference', 'coreModel', 'physical.*raster-erosion'],
  ['patch proximity reference', 'coreBoundary', 'edge.*open'],
  ['landscape metrics', 'contagionWeighting', 'area.*adjacency']
])('W3-08 %s rejects unknown %s values before side effects', (name, option, accepted) => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    const props = {...fixture.props(context.view), [option]: 'unknown-option'};
    const before = [
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ];
    expect(() => fixture.recipe(context.graph, props as never)).toThrow(new RegExp(accepted));
    expect(getNodeCount(context.graph)).toBe(0);
    expect([
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ]).toEqual(before);
  } finally {
    context.device.destroy();
  }
});

function getResultPath(value: unknown, path: string): unknown {
  for (const field of path.split('.')) value = (value as Record<string, unknown>)[field];
  return value;
}

it('W3-08 the shared resolver preserves bare view identity and rejects the removed prop', () => {
  const context = createContext();
  try {
    const view = context.view('float32', 4);
    expect(resolveRecipeDestinations('bare', {output: view})).toBe(view);
    const retired = {output: view, outputs: context.view('float32', 4)};
    expect(() => resolveRecipeDestinations('bare', retired)).toThrow(
      /outputs was removed; use output/
    );
  } finally {
    context.device.destroy();
  }
});

it('STA-12 shared-path completeness retains the caller certificate view', () => {
  const context = createContext();
  try {
    const complete = context.view('uint32', 1);
    const result = addBuildingAdjacencyRecipe(context.graph, {
      sharedPaths: SHARED_PATHS(context.view),
      lineBuildingIds: context.view('uint32', 1),
      orientations: context.view('float32', 2),
      output: {complete}
    });
    const sharedStatus: GPUCompletenessStatusPort = result.status;
    expect(sharedStatus.complete).toBe(complete);
  } finally {
    context.device.destroy();
  }
});

it.each(
  DESTINATIONS
)('W3-08 %s preserves the caller destination identity', (name, port, format, length, resultPath) => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    const props = fixture.props(context.view);
    const destination = context.view(format, length);
    const supplied = {...(props['output'] as object), [port]: destination};
    props['output'] = supplied;
    const result = fixture.recipe(context.graph, props as never) as Record<string, unknown>;
    expect(getResultPath(result, resultPath) === destination).toBe(true);
  } finally {
    context.device.destroy();
  }
});

it.each(
  DESTINATIONS
)('W3-08 %s rejects conflicting destinations before side effects', (name, port, format, length) => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    const props = fixture.props(context.view);
    props['output'] = {...(props['output'] as object), [port]: context.view(format, length)};
    props['outputs'] = {[port]: context.view(format, length)};
    const before = [
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ];
    expect(() => fixture.recipe(context.graph, props as never)).toThrow(/outputs.*use output/);
    expect(getNodeCount(context.graph)).toBe(0);
    expect([
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ]).toEqual(before);
  } finally {
    context.device.destroy();
  }
});

it.each([
  'landscape metrics',
  'patch proximity reference',
  'patch proximity boundary'
])('UCT-01 %s accepts guarded one-based dense raster labels', name => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    const props = fixture.props(context.view);
    delete props['patches'];
    props['reconcilePatches'] = true;
    props['denseLabels'] = {
      grid: {width: 2, height: 2, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      labels: context.view('uint32', 4),
      status: {
        converged: context.view('uint32', 1),
        overflow: context.view('uint32', 1),
        componentCount: context.view('uint32', 1)
      }
    };
    expect(() => fixture.recipe(context.graph, props as never)).not.toThrow();
    const kernels = getGPUCommandNodeKernelSources(
      (context.graph as unknown as {nodes: unknown[]}).nodes
    );
    expect(kernels.some(kernel => kernel.id.includes('dense-input'))).toBe(true);
    for (const kernel of kernels.filter(kernel => /dense-input|unresolved/.test(kernel.id))) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    context.device.destroy();
  }
});

it.each(Object.keys(FIXTURES))('the %s fixture is accepted', name => {
  const fixture = FIXTURES[name];
  const context = createContext();
  try {
    fixture.recipe(context.graph, fixture.props(context.view) as never);
    expect(getNodeCount(context.graph)).toBeGreaterThan(0);
  } finally {
    context.device.destroy();
  }
});

it.each(
  Object.keys(FIXTURES)
)('every rejected %s option leaves the caller graph without nodes or buffer changes', name => {
  const fixture = FIXTURES[name];
  const sampleContext = createContext();
  const mutations = getMutations(fixture.props(sampleContext.view), fixture);
  sampleContext.device.destroy();
  const leaks: string[] = [];
  let rejected = 0;
  for (const mutation of mutations) {
    const context = createContext();
    try {
      const props = fixture.props(context.view);
      mutation.apply(props, context);
      const beforeBuffers = [
        ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
      ];
      try {
        fixture.recipe(context.graph, props as never);
      } catch (error) {
        rejected++;
        const nodeCount = getNodeCount(context.graph);
        const afterBuffers = [
          ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
        ];
        if (nodeCount > 0 || JSON.stringify(afterBuffers) !== JSON.stringify(beforeBuffers)) {
          leaks.push(
            `${mutation.name}: ${nodeCount} nodes, ${afterBuffers.length - beforeBuffers.length} buffers; ${(error as Error).message}`
          );
        }
      }
    } finally {
      context.device.destroy();
    }
  }
  expect(rejected).toBeGreaterThan(0);
  expect(leaks).toEqual([]);
});

it('RCP-101 a rejected randomization adds no segregation decomposition node', () => {
  const context = createContext();
  try {
    const population = context.view('float32x2', 4);
    // Before the fix this rejection came after the four scenario indices and the Shapley node
    // were added (the review counted 61 graph.add calls; this graph held 45 nodes).
    expect(() =>
      addSegregationDecompositionRecipe(context.graph, {
        populations: [population, population, population, population],
        supportId: 'common',
        counterfactualMethod: 'supplied',
        randomization: {replicates: 0, seed: 1}
      })
    ).toThrow(
      /^segregation-decomposition randomization.replicates must be an integer in \[1, 65536\]$/
    );
    expect(getNodeCount(context.graph)).toBe(0);
    // Three tables used to fail with a TypeError after 45 nodes.
    expect(() =>
      addSegregationDecompositionRecipe(context.graph, {
        populations: [population, population, population] as never,
        supportId: 'common',
        counterfactualMethod: 'supplied'
      })
    ).toThrow(/^segregation-decomposition populations must have 4 entries$/);
    expect(getNodeCount(context.graph)).toBe(0);
  } finally {
    context.device.destroy();
  }
});

it('URB-101 and URB-103 kernels parse and pass the WGSL audit', () => {
  const context = createContext();
  try {
    const population = context.view('float32x2', 4);
    for (const swaps of [undefined, context.view('uint32', 32)]) {
      addSegregationDecompositionRecipe(context.graph, {
        id: swaps ? 'supplied-swaps' : 'philox-swaps',
        populations: [population, population, population, population],
        supportId: 'common',
        counterfactualMethod: 'supplied',
        randomization: {replicates: 8, seed: 1, swaps}
      });
    }
    addVisibilityGraphRecipe(context.graph, {
      mode: 'csr',
      positions: context.view('float32x2', 4),
      segments: context.view('float32x4', 4),
      maximumDistance: 10,
      edgeCapacity: 16
    });
    const kernels = getGPUCommandNodeKernelSources(
      (context.graph as unknown as {nodes: unknown[]}).nodes
    ).filter(kernel =>
      /paired-randomization|randomization-flags|finalize-offsets|csr-scatter/.test(kernel.id)
    );
    expect(kernels.length).toBeGreaterThanOrEqual(6);
    for (const kernel of kernels) {
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    context.device.destroy();
  }
});

it.each([
  ['building tessellation reference', ['building-0-dissolve', 'building-1-dissolve']],
  [
    'building tessellation simple enclosure',
    [
      'building-0-dissolve',
      'building-0-clip-enclosure',
      'building-1-dissolve',
      'building-1-clip-enclosure'
    ]
  ],
  ['building adjacency with structures', ['structure-0-union', 'structure-1-union']]
] as const)('RCP-102 plans every %s overlay before recipe resources', (name, suffixes) => {
  const context = createContext();
  const plan = vi.spyOn(overlayPlanning, 'planGPUPolygonOverlay');
  const resources = vi.spyOn(GPUCommandGraph.prototype, 'createTransientBuffer');
  try {
    const props = {...FIXTURES[name].props(context.view), id: 'all-plans'};
    FIXTURES[name].recipe(context.graph, props as never);
    expect(
      plan.mock.calls.slice(0, suffixes.length).map(([, dimensions]) => dimensions.id)
    ).toEqual(suffixes.map(suffix => `all-plans-${suffix}`));
    const firstResource = resources.mock.calls.findIndex(([descriptor]) =>
      descriptor.id?.startsWith('all-plans')
    );
    expect(firstResource).toBeGreaterThanOrEqual(0);
    expect(plan.mock.invocationCallOrder[suffixes.length - 1]).toBeLessThan(
      resources.mock.invocationCallOrder[firstResource]
    );
  } finally {
    resources.mockRestore();
    plan.mockRestore();
    context.device.destroy();
  }
});

it.each([
  {vertices: 2049, rings: 2},
  {vertices: 8, rings: 4096}
])('B2 accepts a supplied overlay when only unused defaults exceed the device limit (%j)', capacity => {
  const context = createContext();
  const plan = vi.spyOn(overlayPlanning, 'planGPUPolygonOverlay');
  try {
    const supplied = createSuppliedOverlay(context.view);
    const props: GPUBuildingTessellationProps = {
      ...createReferenceTessellationProps(context.view),
      id: 'unused-default',
      buildingCount: 1,
      capacity: {intersections: 1, nodedSegments: 32, ...capacity},
      output: {overlays: [supplied]}
    };
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: 16384});
    // Actual output: 8 positions (64 bytes), 2 rings (12 offset bytes).
    // Unused defaults: 2049*8=16392 positions bytes or (4096+1)*4=16388 offsets bytes.
    expect(() => {
      const result = addBuildingTessellationRecipe(context.graph, props);
      expect(result.outputs[0]).toBe(supplied);
    }).not.toThrow();
    expect(plan.mock.calls[0][1].output).toEqual({
      vertexCapacity: 8,
      ringCapacity: 2,
      boundaryCapacity: 32
    });
  } finally {
    plan.mockRestore();
    context.device.destroy();
  }
});

it('RCP-102 plans a later supplied overlay and preserves its identity on retry', () => {
  const context = createContext();
  const plan = vi.spyOn(overlayPlanning, 'planGPUPolygonOverlay');
  const resources = vi.spyOn(GPUCommandGraph.prototype, 'createTransientBuffer');
  try {
    const props: GPUBuildingTessellationProps = {
      ...createReferenceTessellationProps(context.view),
      id: 'later-building'
    };
    const status = () => ({
      count: context.view('uint32', 1),
      overflow: context.view('uint32', 1),
      candidateOverflow: context.view('uint32', 1)
    });
    const supplied: GPUPolygonOverlayOutput = {
      geometry: {
        kind: 'polygons',
        positions: context.view('float32x2', 32),
        featureOffsets: context.view('uint32', 258),
        polygonOffsets: context.view('uint32', 258),
        ringOffsets: context.view('uint32', 258)
      },
      status: {...status(), requiredRingCount: context.view('uint32', 1)},
      boundary: {
        endpoints: context.view('float32x4', 32),
        operandIds: context.view('uint32', 32),
        sourceFeatureIds: context.view('uint32', 32),
        status: {...status(), requiredCount: context.view('uint32', 1)}
      }
    };
    props.output = {overlays: [undefined, supplied]};
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: 4096});
    resources.mockClear();
    // First building uses two rings; the second destination's 257 padded rings require
    // 257*16 = 4112 bytes of ring bounds even though every resident view itself fits.
    expect(() => addBuildingTessellationRecipe(context.graph, props)).toThrow(
      /later-building-building-1-dissolve.*ringBounds/
    );
    expect(plan.mock.calls.map(([, dimensions]) => dimensions.id)).toEqual([
      'later-building-building-0-dissolve',
      'later-building-building-1-dissolve'
    ]);
    expect(
      resources.mock.calls.filter(([descriptor]) => descriptor.id?.startsWith('later-building'))
    ).toEqual([]);
    expect(getNodeCount(context.graph)).toBe(0);
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: 16384});
    const result = addBuildingTessellationRecipe(context.graph, props);
    expect(result.outputs[1] === supplied).toBe(true);
  } finally {
    resources.mockRestore();
    plan.mockRestore();
    context.device.destroy();
  }
});

it('RCP-102 plans a later enclosure clip before creating any recipe scratch', () => {
  const context = createContext();
  const plan = vi.spyOn(overlayPlanning, 'planGPUPolygonOverlay');
  const resources = vi.spyOn(GPUCommandGraph.prototype, 'createTransientBuffer');
  try {
    const props: GPUBuildingTessellationProps = {
      ...createReferenceTessellationProps(context.view, 'simple'),
      id: 'later-clip'
    };
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: 4096});
    resources.mockClear();
    // Dissolve: 5 padded vertices + 1 ring + 1 intersection = 7 hot pixels (896 bytes).
    // Clip: 32 + 4 padded vertices + 2 + 1 rings + 1 intersection = 40 (5120 bytes).
    expect(() => addBuildingTessellationRecipe(context.graph, props)).toThrow(
      /later-clip-building-0-clip-enclosure.*hotPixelSegmentTables/
    );
    expect(plan.mock.calls.map(([, dimensions]) => dimensions.id)).toEqual([
      'later-clip-building-0-dissolve',
      'later-clip-building-0-clip-enclosure'
    ]);
    expect(
      resources.mock.calls.filter(([descriptor]) => descriptor.id?.startsWith('later-clip'))
    ).toEqual([]);
    expect(getNodeCount(context.graph)).toBe(0);
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: 16384});
    expect(() => addBuildingTessellationRecipe(context.graph, props)).not.toThrow();
    expect(getNodeCount(context.graph)).toBeGreaterThan(0);
  } finally {
    resources.mockRestore();
    plan.mockRestore();
    context.device.destroy();
  }
});

// A power-of-two hierarchy can exceed the device limit even when flat noded storage fits.
// Dimension planning must reject it before either recipe allocates resources.
it.each([
  'building tessellation reference',
  'building tessellation simple enclosure',
  'building adjacency with structures'
])('overlay device limits leave %s unchanged and allow the same ID to retry', name => {
  const context = createContext();
  try {
    const fixture = FIXTURES[name];
    const props = fixture.props(context.view);
    props['id'] = 'overlay-retry';
    const capacity = {intersections: 1, nodedSegments: 129, vertices: 8, rings: 2};
    if (name.startsWith('building tessellation')) {
      props['sampleCapacity'] = 1;
      props['capacity'] = capacity;
    } else {
      (props['structures'] as Record<string, unknown>)['capacity'] = capacity;
    }
    Object.assign(context.device.limits, {maxStorageBufferBindingSize: 3000});
    const before = [
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ];
    let rejection: unknown;
    try {
      fixture.recipe(context.graph, props as never);
    } catch (error) {
      rejection = error;
    }
    expect(String(rejection)).toMatch(/groupHierarchy/);
    expect(getNodeCount(context.graph)).toBe(0);
    expect([
      ...(context.graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()
    ]).toEqual(before);
    capacity.nodedSegments = 64;
    fixture.recipe(context.graph, props as never);
    expect(getNodeCount(context.graph)).toBeGreaterThan(0);
  } finally {
    context.device.destroy();
  }
});

it.each([
  'brute-force',
  'bvh'
] as const)('URB-104 facade kernels parse within portable storage limits (%s)', obstacleIndex => {
  const context = createContext();
  try {
    addStreetProfilesRecipe(context.graph, {
      positions: context.view('float32x2', 1),
      normals: context.view('float32x2', 1),
      segments: context.view('float32x4', 2),
      segmentBuildingIds: context.view('uint32', 2),
      buildingHeights: context.view('float32', 2),
      streetSegment: context.view('float32x4', 1),
      maximumDistance: 5,
      sampleSpacing: 1,
      obstacleIndex
    });
    const kernels = getGPUCommandNodeKernelSources(
      (context.graph as unknown as {nodes: unknown[]}).nodes
    ).filter(kernel => /facade-/.test(kernel.id));
    expect(kernels).toHaveLength(3);
    for (const kernel of kernels) {
      const reflection = new WgslReflect(kernel.source);
      expect(reflection.storage.length).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    }
  } finally {
    context.device.destroy();
  }
});
