// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUAnalysisCoordinateModel,
  GPUAnalysisDataProfile,
  GPUAnalysisMeasurementLevel,
  GPUAnalysisRowSemantics
} from '@luma-contributors/analysis/gpu-spatial-analysis';

// What the Kepler host knows about one loaded dataset when it chooses analyses for it: the
// host-neutral analysis profile plus how the dataset is addressed in Kepler.

/** Profile of one Kepler dataset. Computed once per dataset revision and cached. */
export type KeplerDatasetProfile = GPUAnalysisDataProfile & {
  datasetId: string;
  label: string;
  /** Dataset revision the profile was computed for; the profile is stale when it differs. */
  revision?: number;
  /** Geometry kind the catalog can read from this dataset; absent for a table without geometry. */
  catalogGeometry?: 'point' | 'line' | 'polygon';
  /**
   * Rows per geometry family when the dataset holds more than one. The catalog rejects an input
   * whose features are not all of an accepted kind, so such a dataset must be split first.
   */
  mixedGeometry?: {point: number; line: number; polygon: number};
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
