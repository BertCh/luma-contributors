// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisOperator, GPUAnalysisOperatorIdentity} from '../spatial-analysis-ontology';

const SPATIAL = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
const identity = (exportName: string): GPUAnalysisOperatorIdentity => ({
  package: SPATIAL,
  exportName
});
const chunked = (exportName: string, differences: readonly string[]) => ({
  ...identity(exportName),
  relationship: 'chunked',
  differences
});

/** Audited relationships; absence does not imply that no variants or recipe stages exist. */
export const OPERATOR_FORM_RELATIONSHIPS: Readonly<Record<string, GPUAnalysisOperator['forms']>> = {
  [`${SPATIAL}:GPUSpatialWeightsValidation`]: {
    variants: [
      chunked('GPUChunkedSpatialWeightsValidation', [
        'Same implementation; accepts chunked CSR. Flags use global row order. Read invalidCount before downstream submission.'
      ])
    ]
  },
  [`${SPATIAL}:GPUSpatialWeightsTranspose`]: {
    variants: [
      chunked('GPUChunkedSpatialWeightsTranspose', [
        'Accepts chunked input and packed or chunked output. For chunked output inspect outputWeights.status.overflow for any dropped output slots and outputWeights.status.candidateOverflow for inherited input loss; returned chunk descriptions omit loss flags. Caller-owned output chunk ports receive local overflow and reset candidateOverflow to zero, so chunk diagnostics do not certify matrix completeness.'
      ])
    ]
  },
  [`${SPATIAL}:GPUSpatialWeightsTransform`]: {
    variants: [
      chunked('GPUChunkedSpatialWeightsTransform', [
        'Input and output are chunked CSR; inspect each chunk loss status.'
      ])
    ]
  },
  [`${SPATIAL}:GPUSpatialWeightsAlgebra`]: {
    variants: [
      chunked('GPUChunkedSpatialWeightsAlgebra', [
        'Row-local operations only: binary merge, self weights and subgraph; this is not the whole unchunked operation set. Output ranges follow operand chunks; capacities and overflow are per chunk.'
      ])
    ]
  },
  [`${SPATIAL}:GPUSpatialPredicateJoin`]: {
    variants: [
      {
        ...identity('GPUSpatialJoinPrepared'),
        relationship: 'prepared-input',
        differences: [
          'Preparation produces reusable geometry/index resources, not join pairs. Whole-domain BVH bindings still must fit the device.'
        ]
      }
    ]
  },
  [`${SPATIAL}:GPUNearestFeatureJoin`]: {
    variants: [
      {
        ...identity('GPUSpatialJoinPrepared'),
        relationship: 'prepared-input',
        differences: [
          'Preparation produces reusable geometry/index resources, not nearest-feature results. Whole-domain BVH bindings still must fit the device.'
        ]
      }
    ]
  },
  [`${SPATIAL}:GPUContiguityWeights`]: {
    variants: [
      {
        ...identity('computePolygonContiguityOnCPU'),
        relationship: 'prepared-topology-array',
        differences: [
          'Uses prepared polygon topology; see the array signature and source contract notes.'
        ]
      }
    ]
  },
  [`${SPATIAL}:addClusterAndOutlineRecipe`]: {
    recipeStages: [
      'GPUDBSCAN',
      'GPUGroupGeometry',
      'GPUGroupConvexHull',
      'GPUGeometryMeasures',
      'GPUGroupConcaveHull'
    ].map(exportName => ({
      ...identity(exportName),
      appliesWhen:
        exportName === 'GPUGroupConvexHull'
          ? 'outline convex (default)'
          : exportName === 'GPUGroupConcaveHull'
            ? 'outline alpha'
            : exportName === 'GPUGeometryMeasures'
              ? "props.outline?.kind !== 'alpha' (convex; always computes area, perimeter and centroid)"
              : 'always'
    }))
  }
};

/** Additional prepared and tiled contracts, keyed by their existing contributor census identity. */
export const PREPARED_TILED_FORM_RELATIONSHIPS: Readonly<
  Record<string, GPUAnalysisOperator['forms']>
> = {
  [`${SPATIAL}:GPUPointInPolygonJoin`]: {
    variants: [
      {
        ...identity('GPUSpatialJoinPrepared'),
        relationship: 'prepared-input',
        differences: [
          'Optional prepared right polygons must describe the same geometry views and be declared before the join. Preparation does not change predicate semantics or remove whole-domain index bindings; update the handle after geometry mutations.'
        ]
      }
    ]
  },
  [`${SPATIAL}:GPUSpatialJoinCandidates`]: {
    variants: [
      {
        ...identity('GPUSpatialJoinPrepared'),
        relationship: 'prepared-input',
        differences: [
          'Optional prepared right geometry shares the right views; declare preparation first. Otherwise leafCapacity defaults to nextPowerOfTwo(right feature count). Candidate pairs are conservative bounding-box hits, not exact predicates; count bounds the prefix, overflow is pair loss and candidateOverflow is BVH leaf loss. Default pair capacity maximum is 16777216 at a 128 MiB binding.'
        ]
      }
    ]
  },
  [`${SPATIAL}:GPUContiguityWeights`]: {
    variants: [
      {
        ...identity('computePolygonContiguityOnCPU'),
        relationship: 'prepared-topology-array',
        differences: [
          'CPU arrays consume prepared polygon topology rather than GPU geometry resources.'
        ]
      },
      {
        ...identity('preparePolygonContiguityTopology'),
        relationship: 'host-preparation',
        differences: [
          'Validates contiguous source batch rows and creates reusable polygon topology. Invalid rows retain empty feature ranges; empty/nonpolygon rows have mask zero. GeometryCollection polygon parts participate; the borrowed source is not retained. No GPU preparation resource is produced.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/gpu-terrain:GPUTerrainFlow': {
    variants: [
      {
        package: '@luma-contributors/analysis/gpu-terrain',
        exportName: 'GPUTerrainTiledFlow',
        relationship: 'tiled-input',
        differences: [
          'Consumes explicit halo tiles of precomputed flow directions and runoff, not raw elevations. D8 is default; branching requires eight fractions per cell. It is not a replacement elevation-to-direction calculation. Execution defaults to auto; D8 uses tiled boundary routing when the resident 32-byte-per-cell plane cannot fit one buffer/binding. Resident/branching domains are at most 536870911 cells; D8 tiled domains at most 4294967295 cells, subject to owned forest/boundary/index budgets. All tiles must agree on domain grid and flow-direction encoding.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPUIsolines': {
    variants: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'GPUIsolines',
        relationship: 'tiled-input',
        differences: [
          'Optional tile uses half-open cell ownership with a one-sample right/top halo; grid dimensions are at least two. Use identical domain parameters and localOrigin for every tile. Counts and capacities are tile-local; polylines remain fragments across tiles. Match segmentEdges and segmentLevels to stitch domain chains without coordinate snapping. The array function has no tile option; caller assembles the field.'
        ]
      }
    ]
  },
  '@luma-contributors/analysis/raster-analysis:GPURasterRegionGraph': {
    variants: [
      {
        package: '@luma-contributors/analysis/raster-analysis',
        exportName: 'addRasterRegionGraphTiles',
        relationship: 'tiled-builder',
        differences: [
          'Assembles nonoverlapping tiles that exactly cover the domain, then runs GPURasterRegionGraph. Caller label equivalence IDs identify shared regions across tiles; absent mappings use supplied labels. No automatic cross-tile equivalence inference. Assembly still requires whole-domain labels/values and dense regionCount squared adjacency/weight outputs; source tiling does not remove those binding limits.',
          'Returns GPURasterRegionGraph & {readonly estimatedRecipeCommandNodeCount: number}. The recipe estimate adds one assembly command per tile and, when source completeness certificates are supplied, one initialization plus one command per certificate to the inner region-graph estimate. Source certificates require output.status.complete; convergence alone does not certify source labels.'
        ]
      }
    ]
  }
};
