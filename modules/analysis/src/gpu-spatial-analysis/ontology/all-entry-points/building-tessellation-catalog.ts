// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OperatorDescriptor} from './resolve-operator-descriptor';

const REFERENCE_SOURCE =
  'modules/analysis/src/gpu-spatial-analysis/urban-morphology/building-tessellation.ts';
const LOCAL_SOURCE =
  'modules/analysis/src/gpu-spatial-analysis/urban-morphology/building-tessellation-local.ts';

/** Named public routes added after the historical 261; no GPU constructors are imported. */
export const BUILDING_TESSELLATION_OPERATORS: readonly OperatorDescriptor[] = [
  {
    exportName: 'addBuildingTessellationReferenceRecipe',
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    primaryCapability: 'coverage-urban-morphology',
    tags: [],
    inputs: ['lines', 'polygons', 'attributes'],
    outputs: ['polygons', 'attributes', 'summary'],
    dimensionality: '2d',
    coordinateModels: ['planar'],
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: 'order-dependent',
    readback: 'none',
    role: 'operator',
    tier: 'reference',
    summary:
      'Boundary-sampled building tessellation with dissolved polygon rings, one overlay per building.',
    status: [
      'count',
      'requiredCount',
      'overflow',
      'candidateOverflow',
      'requiredRingCount',
      'ringCount',
      'uncertainCount',
      'nodedSegmentRequiredCount'
    ],
    limitation:
      'The default reference layout exposes result.outputs[i].status.{count,overflow,candidateOverflow,requiredRingCount} per building overlay. A caller-owned overlay may omit requiredRingCount and supply optional ringCount, uncertainCount and nodedSegmentRequiredCount. result.outputs[i].boundary.status.requiredCount is the selected boundary-edge requirement, separate from the geometry ring requirement and feature count. result.status contains packed sampling and per-building complete flags. Per call: at most 64 buildings, 32 total enclosure vertices and 128 total boundary samples; overlay capacities and device storage/vertex limits also apply.',
    limits: [
      {
        name: 'buildingCount (per call)',
        maximum: 64,
        reason: 'Reference per-building dissolved overlays.',
        source: REFERENCE_SOURCE
      },
      {
        name: 'enclosure.length (per call)',
        maximum: 32,
        reason: 'Total reference enclosure vertices.',
        source: REFERENCE_SOURCE
      },
      {
        name: 'sampleCapacity (per call)',
        maximum: 128,
        reason: 'Total reference boundary samples.',
        source: REFERENCE_SOURCE
      }
    ],
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/gpu-spatial-analysis/ontology/catalog-reference-targets.node.spec.ts',
      additionalTests: [
        'modules/analysis/test/gpu-spatial-analysis/urban-ecology/building-tessellation-routes.node.spec.ts',
        'modules/analysis/test/gpu-spatial-analysis/urban-ecology/urban-ecology.spec.ts'
      ],
      documentation: 'docs/api/gpu-spatial-analysis/urban-morphology.md',
      contract: 'GPUBuildingTessellationProps'
    }
  },
  {
    exportName: 'addLocalBuildingTessellationRecipe',
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    primaryCapability: 'coverage-urban-morphology',
    tags: [],
    inputs: ['lines', 'polygons', 'attributes', 'parameters'],
    outputs: ['polygons', 'attributes', 'summary'],
    dimensionality: '2d',
    coordinateModels: ['planar'],
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'approximate',
    deterministic: 'order-dependent',
    readback: 'none',
    role: 'operator',
    tier: 'scale',
    summary:
      'Indexed local building tessellation with convex polygon pieces and fixed-stride cells.',
    status: ['count', 'requiredCount', 'overflow', 'candidateOverflow', 'invalidCount'],
    limitation:
      'Local result.status.{count,requiredCount,overflow,candidateOverflow,invalidCount} is recipe-level; outputs.positions/cells encode fixed-stride convex polygon pieces with variable vertex count and retained internal edges, not dissolved shell/hole rings. Read the first word of each cells record for the piece vertex count. Packed sampling and per-building buildingStatus remain separate. Per call: at most 1,000,000 buildings, 16,777,216 total samples, 32 total enclosure vertices and 32 rings. Per building: 32 own samples and 256 candidates. Per convex piece: 128 vertices. Each indexed query attempt visits at most 8,192 rows. Device storage limits may lower these bounds; inspect certification and overflow.',
    limits: [
      {
        name: 'buildingCount (per call)',
        maximum: 1000000,
        reason: 'Indexed local building rows.',
        source: LOCAL_SOURCE
      },
      {
        name: 'sampleCapacity (per call)',
        maximum: 16777216,
        reason: 'Total boundary samples with exact float32 row counts.',
        source: LOCAL_SOURCE
      },
      {
        name: 'enclosure.length (per call)',
        maximum: 32,
        reason: 'Total enclosure vertices, including all rings.',
        source: LOCAL_SOURCE
      },
      {
        name: 'enclosure rings (per call)',
        maximum: 32,
        reason: 'Bounds enclosure triangulation and ring slots.',
        source: LOCAL_SOURCE
      },
      {
        name: 'maximumSamplesPerBuilding',
        maximum: 32,
        reason: 'Own samples per building.',
        source: LOCAL_SOURCE
      },
      {
        name: 'candidateCapacity (per building)',
        maximum: 256,
        reason: 'Private candidate table per building.',
        source: LOCAL_SOURCE
      },
      {
        name: 'cellVertexCapacity (per convex piece)',
        maximum: 128,
        reason: 'Private clipping array per convex piece.',
        source: LOCAL_SOURCE
      },
      {
        name: 'maximumGridVisits (per indexed query attempt)',
        maximum: 8192,
        reason: 'Row visits per grid or sorted-interval search attempt.',
        source: LOCAL_SOURCE
      }
    ],
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/gpu-spatial-analysis/ontology/catalog-reference-targets.node.spec.ts',
      additionalTests: [
        'modules/analysis/test/gpu-spatial-analysis/urban-ecology/building-tessellation-routes.node.spec.ts',
        'modules/analysis/test/gpu-spatial-analysis/urban-ecology/building-tessellation-local.spec.ts'
      ],
      documentation: 'docs/api/gpu-spatial-analysis/urban-morphology.md',
      contract: 'GPUBuildingTessellationLocalProps'
    }
  }
];
