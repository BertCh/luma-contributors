// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../analysis/catalog-binding-types';

import type {
  GPUAnalysisCoordinateModel,
  GPUAnalysisDataProfile,
  GPUAnalysisMeasurementLevel,
  GPUAnalysisRowSemantics,
  GPUAnalysisSpacingStatistics
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';

// What the Kepler host knows about one loaded dataset when it chooses analyses for it: the
// host-neutral analysis profile plus how the dataset is addressed in Kepler.

/** Exact graph counts tied to source revision and resolved catalog topology settings. */
export type KeplerDatasetNetworkProfile = {
  datasetId: string;
  revision?: number;
  topologyKey: string;
  /** Includes isolated interior vertices retained by the catalog builder. */
  nodeCount: number;
  /** Physical pieces counted once, independent of allowed traversal directions. */
  edgeCount: number;
  /** Directed traversals with finite allowed costs; zero to twice the physical edge count. */
  arcCount: number;
};

/** Describes whether resident positions cover every location or only feature anchors. */
export type KeplerDatasetProfileSpacing = GPUAnalysisSpacingStatistics & {
  locationCoverage?: 'all-locations' | 'feature-anchors';
};

/** Progress is published synchronously for metadata and before/after each deferred stage. */
export type KeplerDatasetProfileProgress = {
  datasetId: string;
  revision?: number;
  stage:
    | 'metadata'
    | 'columns'
    | 'spacing'
    | 'contiguity'
    | 'complete'
    | 'cancelled'
    | 'stale'
    | 'failed';
  elapsedMilliseconds: number;
  timings: Partial<Record<'columns' | 'spacing' | 'contiguity', number>>;
  profile?: KeplerDatasetProfile;
  error?: unknown;
};

/** Profile of one Kepler dataset. Computed once per dataset revision and cached. */
export type KeplerDatasetProfile = GPUAnalysisDataProfile & {
  spacing?: KeplerDatasetProfileSpacing;
  /** Native track/time readiness validated by the catalog preparer for this revision. */
  trajectory?: {
    parameters: Readonly<Record<string, CatalogParameterValue>>;
    fixCount: number;
    trackCount: number;
  };
  scalarGrid?: {width: number; height: number; valueColumn: string};
  /** Lazy facts for one topology; never inferred from rows or vertices. */
  network?: KeplerDatasetNetworkProfile;
  /** Exact seed collisions for one execution grid and both source revisions. */
  catchmentSeeds?: {
    facilityRevision?: number;
    observationsDatasetId: string;
    observationsRevision?: number;
    coordinateSystem: string;
    cellSize: number;
    collisionCount: number;
  };
  datasetId: string;
  label: string;
  /** Underlying dataset identity when a caller aliases this profile to an input port. */
  sourceDatasetId?: string;
  /** Dataset revision the profile was computed for; the profile is stale when it differs. */
  revision?: number;
  /** Geometry kind the catalog can read from this dataset; absent for a table without geometry. */
  catalogGeometry?: 'point' | 'line' | 'polygon';
  /**
   * Rows per geometry family when the dataset holds more than one. The catalog rejects an input
   * whose features are not all of an accepted kind, so such a dataset must be split first.
   */
  mixedGeometry?: {point: number; line: number; polygon: number};
  /** Exact geometry-family counts when profiling inspected geometry on every row. */
  geometryFeatureCounts?: {point: number; line: number; polygon: number; withoutGeometry: number};
  /** Exact feature counts accepted by catalog packers, excluding empty point/part lists. */
  geometryPackedFeatureCounts?: {point: number; line: number; polygon: number};
  /** Columns that hold geometry as GeoJSON text. */
  geometryColumns: readonly string[];
  /** Coordinate columns, when the rows are points described by two numeric columns. */
  pointColumns?: {longitude: string; latitude: string};
};

/** What a user declared about a dataset; each entry replaces the corresponding guess. */
export type KeplerDatasetProfileDeclarations = {
  /** Measurement level by column name. */
  measurement?: Readonly<Record<string, GPUAnalysisMeasurementLevel>>;
  rowSemantics?: GPUAnalysisRowSemantics;
  coordinateModel?: GPUAnalysisCoordinateModel;
};
