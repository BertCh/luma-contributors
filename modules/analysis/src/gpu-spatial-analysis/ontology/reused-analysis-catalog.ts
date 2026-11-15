// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisOperator} from './spatial-analysis-ontology';

/** Public analysis operations extracted from the Kepler host; arithmetic/upload helpers stay helpers. */
export const REUSED_ANALYSIS_OPERATORS: readonly GPUAnalysisOperator[] = [
  {
    summary:
      'Runs parallel packed-word phases for exact wide-column summaries and narrowing diagnostics.',
    exportName: 'GPUWideNumericScan',
    role: 'helper',
    primaryCapability: 'dataframe-aggregation',
    tags: [],
    package: '@luma-contributors/analysis/dataframe-analysis',
    inputs: ['attributes'],
    outputs: ['summary'],
    dimensionality: 'unspecified',
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'none',
    tier: 'scale',
    limitation:
      'Fewer than 2^32 rows for exact counters and carry sums; packed words and matching validity chunks. Float64 GPU error is a lower bound within relative 2^-31; integer error is exact bigint.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/dataframe-analysis/column-profile/wide-numeric-contract.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'GPUWideNumericScanProps'
    }
  },
  {
    summary:
      'Reads exact Int64 or Uint64 counts, extrema and carry-based bigint sums across all chunks.',
    exportName: 'aggregateInteger64OnGpu',
    primaryCapability: 'dataframe-aggregation',
    tags: [],
    package: '@luma-contributors/analysis/dataframe-analysis',
    inputs: ['attributes'],
    outputs: ['summary'],
    dimensionality: 'unspecified',
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'explicit-adapter',
    tier: 'scale',
    limitation:
      'Fewer than 2^32 rows for exact counters and carry sums; packed words and matching validity chunks. Float64 GPU error is a lower bound within relative 2^-31; integer error is exact bigint.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/dataframe-analysis/column-profile/wide-numeric-contract.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'WideInteger64Summary'
    }
  },
  {
    summary:
      'Aggregates exact 64-bit integers within explicit inclusive or exclusive bigint bounds.',
    exportName: 'aggregateInteger64Range',
    primaryCapability: 'dataframe-aggregation',
    tags: [],
    package: '@luma-contributors/analysis/dataframe-analysis',
    inputs: ['attributes'],
    outputs: ['summary'],
    dimensionality: 'unspecified',
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'explicit-adapter',
    tier: 'scale',
    limitation:
      'Fewer than 2^32 rows for exact counters and carry sums; packed words and matching validity chunks. Float64 GPU error is a lower bound within relative 2^-31; integer error is exact bigint.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/dataframe-analysis/column-profile/wide-numeric-contract.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'WideInteger64Summary'
    }
  },
  {
    summary: 'Reads exact signed Decimal128 extrema, counts, unscaled bigint sums and scale.',
    exportName: 'aggregateDecimal128OnGpu',
    primaryCapability: 'dataframe-aggregation',
    tags: [],
    package: '@luma-contributors/analysis/dataframe-analysis',
    inputs: ['attributes'],
    outputs: ['summary'],
    dimensionality: 'unspecified',
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'explicit-adapter',
    tier: 'scale',
    limitation:
      'Fewer than 2^32 rows for exact counters and carry sums; packed words and matching validity chunks. Float64 GPU error is a lower bound within relative 2^-31; integer error is exact bigint.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/dataframe-analysis/column-profile/wide-numeric-contract.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'Decimal128Summary'
    }
  },
  {
    summary:
      'Uploads an Arrow Decimal128 column, reads its exact aggregate and frees temporary vectors.',
    exportName: 'aggregateDecimal128Column',
    primaryCapability: 'dataframe-aggregation',
    tags: [],
    package: '@luma-contributors/analysis/dataframe-analysis',
    inputs: ['attributes'],
    outputs: ['summary'],
    dimensionality: 'unspecified',
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'explicit-adapter',
    tier: 'scale',
    limitation:
      'Fewer than 2^32 rows for exact counters and carry sums; packed words and matching validity chunks. Float64 GPU error is a lower bound within relative 2^-31; integer error is exact bigint.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/dataframe-analysis/column-profile/wide-numeric-contract.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'Decimal128Summary'
    }
  },
  {
    summary:
      'Scans every resident word for Float32 narrowing, returning exact bigint integer errors or bounded Float64 error estimates.',
    exportName: 'scanWideNumericColumnOnGpu',
    primaryCapability: 'dataframe-aggregation',
    tags: [],
    package: '@luma-contributors/analysis/dataframe-analysis',
    inputs: ['attributes'],
    outputs: ['summary'],
    dimensionality: 'unspecified',
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'explicit-adapter',
    tier: 'scale',
    limitation:
      'Fewer than 2^32 rows for exact counters and carry sums; packed words and matching validity chunks. Float64 GPU error is a lower bound within relative 2^-31; integer error is exact bigint.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/dataframe-analysis/column-profile/wide-numeric-contract.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'WideInteger64Summary'
    }
  },
  {
    summary:
      'Classifies Float32 narrowing on the CPU, subtracting integer errors exactly before any Number conversion.',
    exportName: 'diagnoseFloat32Narrowing',
    primaryCapability: 'dataframe-aggregation',
    tags: [],
    package: '@luma-contributors/analysis/dataframe-analysis',
    inputs: ['attributes'],
    outputs: ['summary'],
    dimensionality: 'unspecified',
    coordinateModels: ['none'],
    topology: 'fixed',
    cardinality: 'fixed-sized',
    semantics: 'approximate',
    deterministic: true,
    status: [],
    readback: 'none',
    tier: 'scale',
    limitation:
      'Fewer than 2^32 rows for exact counters and carry sums; packed words and matching validity chunks. Float64 GPU error is a lower bound within relative 2^-31; integer error is exact bigint.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/dataframe-analysis/column-profile/wide-numeric-contract.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'NumericNarrowingReport'
    }
  },
  {
    summary:
      'Interns original binary64 XY into equality-only tokens while retaining source rows, batches and invalid diagnostics.',
    exportName: 'preparePolygonContiguityTopology',
    primaryCapability: 'spatial-neighborhoods',
    tags: [],
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    inputs: ['polygons'],
    outputs: ['polygons'],
    dimensionality: 'topology',
    coordinateModels: ['topological'],
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'none',
    tier: 'reference',
    limitation:
      '4096 source rows; 65536 inspected vertices; exact-source XY equality tokens, zero snap tolerance.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/gpu-spatial-analysis/spatial-weights/polygon-contiguity-preparation.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'PreparedPolygonContiguityTopology'
    }
  },
  {
    summary:
      'Builds queen or whole-edge rook CSR adjacency from exact-source equality tokens with zero snap tolerance.',
    exportName: 'computePolygonContiguityOnCPU',
    primaryCapability: 'spatial-neighborhoods',
    tags: [],
    package: '@luma-contributors/analysis/gpu-spatial-analysis',
    inputs: ['polygons'],
    outputs: ['spatial-weights'],
    dimensionality: 'topology',
    coordinateModels: ['topological'],
    topology: 'fixed',
    cardinality: 'capacity-bounded',
    semantics: 'exact',
    deterministic: true,
    status: [],
    readback: 'none',
    tier: 'reference',
    limitation:
      '4096 source rows; 65536 inspected vertices; exact-source XY equality tokens, zero snap tolerance.',
    evidence: {
      status: 'partial',
      tests:
        'modules/analysis/test/gpu-spatial-analysis/spatial-weights/polygon-contiguity-preparation.node.spec.ts',
      documentation: 'docs/api/reused-analysis-preparation.md',
      contract: 'PreparedPolygonContiguityTopology'
    }
  }
];
