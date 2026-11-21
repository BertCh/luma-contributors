// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Website evidence attached to five shared curated templates: each step in the order its recipe adds it, the data that flows to
// the next step, and the recorded run behind it. Source lines are cited per step so the order can be
// checked; the recorded numbers come from the lens view models (named JSON imports keep only the
// fields used here in the page bundle).

import {KEPLER_CURATED_ANALYSES} from '../../../../../../modules/kepler/src/curated/curated-analyses';
import type {KeplerCuratedAnalysis} from '../../../../../../modules/kepler/src/curated/curated-analysis-types';
import type {LensResourceBytesKind} from '../../../../lens/types';

import {
  recordedEdgeCount as driveTimeEdges,
  recordedNodeCount as driveTimeNodeCount,
  resources as driveTimeResources,
  title as driveTimeTitle
} from '../../../../lens/generated/gpu-network.drive-time-catchment-recipe.json';
import {
  recordedEdgeCount as drainageEdges,
  recordedNodeCount as drainageNodeCount,
  resources as drainageResources,
  title as drainageTitle
} from '../../../../lens/generated/gpu-spatial-analysis.drainage-network-recipe.json';
import {
  recordedEdgeCount as hotSpotEdges,
  recordedNodeCount as hotSpotNodeCount,
  resources as hotSpotResources,
  title as hotSpotTitle
} from '../../../../lens/generated/gpu-spatial-analysis.hot-spot-analysis-recipe.json';
import {
  recordedEdgeCount as localMoranEdges,
  recordedNodeCount as localMoranNodeCount,
  resources as localMoranResources,
  title as localMoranTitle
} from '../../../../lens/generated/gpu-spatial-analysis.local-moran.json';
import {
  recordedEdgeCount as rateClusterEdges,
  recordedNodeCount as rateClusterNodeCount,
  resources as rateClusterResources,
  title as rateClusterTitle
} from '../../../../lens/generated/gpu-spatial-analysis.rate-cluster-map-recipe.json';
import {
  recordedEdgeCount as spaceTimeEdges,
  recordedNodeCount as spaceTimeNodeCount,
  resources as spaceTimeResources,
  title as spaceTimeTitle
} from '../../../../lens/generated/gpu-spatial-analysis.space-time-hot-spots-recipe.json';

/** What one recorded run compiled to. */
export type WorkflowRecord = {
  scenario: string;
  title: string;
  /** All recorded graph nodes, including compute and copy nodes. */
  nodeCount: number;
  dependencies: number;
  resourceCount: number;
  inventoryKnown?: boolean;
  bytes: number;
  bytesKind?: LensResourceBytesKind;
};

/** JSON imports widen string literals; unknown precision must stay explicitly unknown. */
function readRecordBytesKind(bytesKind: string | undefined): LensResourceBytesKind | undefined {
  switch (bytesKind) {
    case undefined:
    case 'logical':
    case 'lower-bound':
    case 'estimated':
    case 'unspecified':
    case 'unavailable':
      return bytesKind;
    default:
      return 'unspecified';
  }
}

function makeRecord(
  scenario: string,
  title: string,
  nodeCount: number,
  dependencies: number,
  resources: {
    count: number;
    inventoryKnown?: boolean;
    totalBytes: number;
    totalBytesKind?: string;
  }
): WorkflowRecord {
  return {
    scenario,
    title,
    nodeCount,
    dependencies,
    resourceCount: resources.count,
    inventoryKnown: resources.inventoryKnown,
    bytes: resources.totalBytes,
    bytesKind: readRecordBytesKind(resources.totalBytesKind)
  };
}

export const LOCAL_MORAN_RECORD = makeRecord(
  'gpu-spatial-analysis.local-moran',
  localMoranTitle,
  localMoranNodeCount,
  localMoranEdges,
  localMoranResources
);

/**
 * One step of a chain. `entry` is a catalog id (poster and function page); steps without one are
 * adapter kernels a recipe adds with `addGlue` or `graph.add`, or a contributor with no page.
 */
export type WorkflowStep = {
  entry?: string;
  /** Label when there is no catalog entry. */
  label?: string;
  /** What the step passes on, in a few words. */
  passes: string;
  /** Format of what it passes on. */
  format: string;
  /** Not in the recorded run: added only when the named option is given. */
  optional?: string;
  /** Recipe source line that adds the step. */
  source: string;
};

export type WorkflowRecording = {
  templateId: string;
  id: string;
  recipe: string;
  /** Catalog id of the recipe's function page. */
  recipeEntry: string;
  /** Name of the page example (data.ts). */
  example: string;
  /** Where the example comes from, shown as its label. */
  exampleSource: string;
  steps: readonly WorkflowStep[];
  record: WorkflowRecord;
  /** `status.stages` names the recorded configuration reports, in order. */
  stages: readonly string[];
  /** Where `status.stages` is assembled. */
  stagesSource: string;
  choice: string;
  /** Said next to the record when its inputs are not a realistic dataset. */
  recordNote?: string;
};

const HOT_SPOTS = 'modules/analysis/src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe.ts';
const RATE_CLUSTERS =
  'modules/analysis/src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe.ts';
const SPACE_TIME =
  'modules/analysis/src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe.ts';
const DRIVE_TIME = 'modules/analysis/src/gpu-network/recipes/drive-time-catchment-recipe.ts';
const DRAINAGE = 'modules/analysis/src/gpu-spatial-analysis/recipes/drainage-network-recipe.ts';

const WORKFLOW_RECORDINGS: readonly WorkflowRecording[] = [
  {
    id: 'hot-spots',
    templateId: 'hot-spot-map',
    recipe: 'addHotSpotAnalysisRecipe',
    recipeEntry: 'hot-spot-analysis-recipe',
    example: 'hot-spots',
    exampleSource:
      'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/hot-spot-analysis-recipe.example.ts, the code behind the recorded run',
    steps: [
      {
        entry: 'cell-cover',
        passes: 'study-area cells',
        format: 'uint32x2 keys',
        source: `${HOT_SPOTS}:743`
      },
      {
        entry: 'cell-aggregation',
        passes: 'sorted study table',
        format: 'uint32x2 keys',
        source: `${HOT_SPOTS}:801`
      },
      {
        entry: 'cell-aggregation',
        passes: 'points per cell',
        format: 'uint32 counts',
        source: `${HOT_SPOTS}:862`
      },
      {
        label: 'Join kernel',
        passes: 'count per study cell, zeros kept',
        format: 'uint32',
        source: `${HOT_SPOTS}:909`
      },
      {
        entry: 'cell-geometry',
        passes: 'cell centres',
        format: 'float32x2',
        source: `${HOT_SPOTS}:999`
      },
      {
        label: 'Area kernel',
        passes: 'density and mask',
        format: 'float32, uint32',
        source: `${HOT_SPOTS}:1030`
      },
      {
        entry: 'cell-topology',
        passes: 'grid-disk neighbours',
        format: 'CSR: uint32 offsets, neighbours; float32 weights',
        source: `${HOT_SPOTS}:1100`
      },
      {
        entry: 'hot-spot-analysis-gi-star',
        passes: 'Gi* z-scores and bins',
        format: 'float32, sint32',
        source: `${HOT_SPOTS}:1191`
      },
      {
        entry: 'local-permutation-test',
        passes: 'significant mask',
        format: 'uint32',
        optional: 'permutation',
        source: `${HOT_SPOTS}:1312`
      },
      {
        entry: 'class-breaks',
        passes: 'class per cell',
        format: 'uint32',
        optional: 'color',
        source: `${HOT_SPOTS}:1398`
      },
      {
        entry: 'color-scale',
        passes: 'colour per cell',
        format: 'uint32 rgba8',
        optional: 'color',
        source: `${HOT_SPOTS}:1421`
      }
    ],
    record: makeRecord(
      'gpu-spatial-analysis.hot-spot-analysis-recipe',
      hotSpotTitle,
      hotSpotNodeCount,
      hotSpotEdges,
      hotSpotResources
    ),
    recordNote:
      'This recording counts events on a study-area DGGS cover. The curated template uses a square lattice and its own inference settings; the recording is a related recipe, not the template execution.',
    stages: ['study-area', 'study-table', 'point-aggregation', 'weights', 'minimum-neighbors'],
    stagesSource: `${HOT_SPOTS}:509`,
    // Defaults: neighborRings 1 for Quadbin, 2 for H3 (hot-spot-analysis-recipe.ts:618),
    // minimumNeighbors 8 (hot-spot-analysis-recipe.ts:84).
    choice:
      'Cell size sets the scale of a hot spot. Neighbours are grid rings (one ring for Quadbin, two for H3 by default), and cells with fewer than eight neighbours get no score.'
  },
  {
    id: 'rate-clusters',
    templateId: 'rate-cluster-map',
    recipe: 'addRateClusterMapRecipe',
    recipeEntry: 'rate-cluster-map-recipe',
    example: 'rate-clusters',
    exampleSource:
      'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/rate-cluster-map-recipe.example.ts, the code behind the recorded run',
    steps: [
      {
        entry: 'empirical-bayes-rates',
        passes: 'standardized rates',
        format: 'float32',
        source: `${RATE_CLUSTERS}:319`
      },
      {
        entry: 'contiguity-weights',
        passes: 'shared-edge neighbours',
        format: 'CSR: uint32 offsets, neighbours; float32 weights',
        source: `${RATE_CLUSTERS}:351`
      },
      {
        entry: 'spatial-weights-transform',
        passes: 'row-standardized weights, in place',
        format: 'CSR',
        source: `${RATE_CLUSTERS}:361`
      },
      {
        entry: 'local-morans-i',
        passes: 'local I, z-scores, quadrants',
        format: 'float32, uint32',
        source: `${RATE_CLUSTERS}:402`
      },
      {
        entry: 'local-permutation-test',
        passes: 'significant mask',
        format: 'uint32',
        source: `${RATE_CLUSTERS}:465`
      },
      {
        label: 'Quadrant colour kernel',
        passes: 'colour per polygon',
        format: 'uint32 rgba8',
        source: `${RATE_CLUSTERS}:495`
      }
    ],
    record: makeRecord(
      'gpu-spatial-analysis.rate-cluster-map-recipe',
      rateClusterTitle,
      rateClusterNodeCount,
      rateClusterEdges,
      rateClusterResources
    ),
    stages: ['weights', 'permutation'],
    stagesSource: `${RATE_CLUSTERS}:584`,
    // 'queen' or 'rook' criterion; 999 permutations recommended (rate-cluster-map-recipe.ts:265).
    choice:
      'Queen or rook contiguity decides who counts as a neighbour. Without permutations the test is analytic, a preview; the recipe recommends 999 permutations for the reference answer.'
  },
  {
    id: 'emerging',
    templateId: 'emerging-hot-spots',
    recipe: 'addSpaceTimeHotSpotsRecipe',
    recipeEntry: 'space-time-hot-spots-recipe',
    example: 'space-time',
    exampleSource:
      'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/space-time-hot-spots-recipe.example.ts, the code behind the recorded run',
    steps: [
      {
        entry: 'calendar-buckets',
        passes: 'year, month, day per event',
        format: 'sint32, uint32',
        source: `${SPACE_TIME}:444`
      },
      {
        label: 'GPUPeriodIndex',
        passes: 'time slice per event',
        format: 'uint32',
        source: `${SPACE_TIME}:479`
      },
      {
        label: 'Cube kernels',
        passes: 'dense cell by slice cube of counts',
        format: 'uint32',
        source: `${SPACE_TIME}:546`
      },
      {
        entry: 'emerging-hot-spots',
        passes: 'category per cell',
        format: 'uint32, 17 classes',
        source: `${SPACE_TIME}:737`
      },
      {
        entry: 'color-scale',
        passes: 'colour per cell',
        format: 'uint32 rgba8',
        optional: 'color',
        source: `${SPACE_TIME}:797`
      }
    ],
    record: makeRecord(
      'gpu-spatial-analysis.space-time-hot-spots-recipe',
      spaceTimeTitle,
      spaceTimeNodeCount,
      spaceTimeEdges,
      spaceTimeResources
    ),
    stages: ['slice-range'],
    stagesSource: `${SPACE_TIME}:577`,
    choice:
      'The slice step (a month here) and the neighbourhood radius in cells set what a trend means: short slices are noisy, long ones hide change.'
  },
  {
    id: 'drive-time',
    templateId: 'drive-time-catchments',
    recipe: 'addDriveTimeCatchmentRecipe',
    recipeEntry: 'drive-time-catchment',
    example: 'drive-time',
    exampleSource:
      'modules/lenses/src/computation-lenses/scenarios/gpu-network/drive-time-catchment-recipe.example.ts, the code behind the recorded run',
    steps: [
      {
        entry: 'network-snapping',
        passes: 'start nodes and their costs',
        format: 'uint32, float32',
        source: `${DRIVE_TIME}:438`
      },
      {
        entry: 'network-service-areas',
        passes: 'cost and facility per node',
        format: 'float32, uint32',
        source: `${DRIVE_TIME}:482`
      },
      {
        entry: 'network-isochrones',
        passes: 'isochrone polygons',
        format: 'triangles or rings',
        optional: 'isochrones',
        source: `${DRIVE_TIME}:506`
      },
      {
        entry: 'point-in-polygon-join',
        passes: 'demand inside each ring',
        format: 'uint32',
        optional: 'isochrones.joinDemand',
        source: `${DRIVE_TIME}:552`
      },
      {
        entry: 'network-snapping',
        passes: 'demand on its nearest edge',
        format: 'uint32',
        source: `${DRIVE_TIME}:593`
      },
      {
        label: 'Demand time kernels',
        passes: 'time and band per demand point',
        format: 'float32, uint32',
        source: `${DRIVE_TIME}:618`
      },
      {
        entry: 'group-statistics',
        passes: 'demand per band',
        format: 'uint32 keys, counts',
        source: `${DRIVE_TIME}:726`
      }
    ],
    record: makeRecord(
      'gpu-network.drive-time-catchment-recipe',
      driveTimeTitle,
      driveTimeNodeCount,
      driveTimeEdges,
      driveTimeResources
    ),
    stages: ['service-areas', 'band-statistics'],
    stagesSource: `${DRIVE_TIME}:783`,
    recordNote:
      'This recording runs the drive-time band recipe. The curated template computes one demand total per facility, without bands or isochrones; its method and binding are listed above.',
    choice:
      'maximumIterations bounds the cost search at compile time; if service-areas reports not converged, raise it or set a cost limit.'
  },
  {
    id: 'drainage',
    templateId: 'drainage-network',
    recipe: 'addDrainageNetworkRecipe',
    recipeEntry: 'drainage-network-recipe',
    example: 'drainage',
    exampleSource:
      'drainage-network.example.ts, a typechecked page example; the record ran the same recipe on a test grid',
    steps: [
      {
        entry: 'terrain-flow',
        passes: 'D8 directions, accumulation, streams',
        format: 'uint32, float32, uint32',
        source: `${DRAINAGE}:48`
      },
      {
        entry: 'terrain-stream-order',
        passes: 'Strahler order per stream cell',
        format: 'uint32',
        source: `${DRAINAGE}:58`
      },
      {
        entry: 'terrain-watersheds',
        passes: 'catchment label per cell',
        format: 'uint32',
        source: `${DRAINAGE}:71`
      },
      {
        entry: 'terrain-flow-distance',
        passes: 'distance to the outlet',
        format: 'float32',
        source: `${DRAINAGE}:85`
      },
      {
        entry: 'terrain-stream-network',
        passes: 'stream reaches as lines',
        format: 'float32x2 vertices, uint32 offsets',
        source: `${DRAINAGE}:121`
      },
      {
        entry: 'group-statistics',
        passes: 'cells and mean distance per catchment',
        format: 'uint32, float32',
        source: `${DRAINAGE}:138`
      }
    ],
    record: makeRecord(
      'gpu-spatial-analysis.drainage-network-recipe',
      drainageTitle,
      drainageNodeCount,
      drainageEdges,
      drainageResources
    ),
    // The record runs without fillDepressions and resolveFlats, so those two stages are absent
    // (expansion-examples.ts:288-313 in the lenses scenarios).
    stages: [
      'accumulation',
      'stream-order',
      'catchments',
      'downstream-distance',
      'stream-network',
      'catchment-summary'
    ],
    stagesSource: `${DRAINAGE}:171`,
    // expansion.lens.ts:104-106: 'explicit illustrative inputs', a 3 x 3 grid.
    recordNote: 'recorded on a 3 by 3 test grid, so the buffers are tiny',
    choice:
      'The stream threshold, the upstream area at which a cell counts as a stream, sets how dense the network is; filling depressions first keeps pits from ending streams early.'
  }
];

/** All workflow descriptions are the shared records, including honest availability and typed slots. */
export const WORKFLOW_TEMPLATES = KEPLER_CURATED_ANALYSES;

export type Workflow = WorkflowRecording & {
  template: KeplerCuratedAnalysis;
  title: string;
  question: string;
  start: string;
};

/** Attach only website evidence to the shared template; never author its question or method here. */
export const WORKFLOWS: readonly Workflow[] = WORKFLOW_TEMPLATES.flatMap(template => {
  const recording = WORKFLOW_RECORDINGS.find(candidate => candidate.templateId === template.id);
  return recording
    ? [
        {
          ...recording,
          template,
          title: template.name,
          question: template.title,
          start: template.slots.map(slot => slot.label).join('; ')
        }
      ]
    : [];
});
