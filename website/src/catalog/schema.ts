// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The function catalog contract. Every user-facing operation gets one CatalogEntry, written in
// src/catalog/entries/<group>.ts. Keep the prose very concise.

import type {DataTypeId, DomainId, UseCaseId} from './taxonomy';
import type {CatalogAnalysisMetadata, CatalogInterpretation} from './analysis-metadata';

export type CatalogEntry = Partial<CatalogAnalysisMetadata> & {
  /** How to read the output; omitted prose is supplied from the existing explanation. */
  interpretation?: CatalogInterpretation;
  /** Unique kebab-case id, used as the URL slug: /fn/<id>. */
  id: string;
  /** Exported symbol, for example `GPULocalMoran` or `addDasymetricRedistributionRecipe`. */
  symbol: string;
  kind: 'class' | 'function' | 'recipe';
  /** Public entry point, for example `@luma-contributors/analysis/gpu-spatial-analysis`. */
  importPath: string;
  /** Human title, for example `Local Moran's I`. */
  title: string;
  /** One sentence, at most 120 characters. What it does, not how. */
  summary: string;
  /** At most three short sentences: when to use it and the key idea. Inline `code` allowed. */
  explanation: string;
  /** Legacy authoring field; metadata attachment replaces it with the shared kind projection. */
  dataTypes: {input: DataTypeId[]; output: DataTypeId[]};
  /** One to three. */
  useCases: UseCaseId[];
  /** One to three. */
  domains: DomainId[];
  /** The props that matter, at most eight. Names and types must match the source. */
  params: ParamDoc[];
  /**
   * Names of required props left out of `params` to stay within the eight-row cap. The function page
   * adds a final "and N more required" row; the catalog checker verifies each name is a required prop.
   */
  moreRequired?: string[];
  outputs: OutputDoc[];
  /** TypeScript, at most 25 lines, using the real public API and import path. */
  example: string;
  /** Visual preview shown at the top of the function page and on cards. */
  preview: PreviewSpec;
  /** Ids of related entries. */
  related?: string[];
  /** Lens scenario id in docs/lens-records (file name without .json), if one exists. */
  lensScenario?: string;
  /** Reference docs slug, for example `api/gpu-spatial-analysis/operators-statistics`. */
  reference?: string;
  maturity: 'stable' | 'partial' | 'experimental';
};

export type ParamDoc = {
  name: string;
  type: string;
  required?: boolean;
  default?: string;
  /** At most 80 characters. */
  description: string;
};

export type OutputDoc = {
  name: string;
  type: string;
  /** At most 80 characters. */
  description: string;
};

/**
 * Illustrative visual for an entry. Each kind has a live luma.gl scene (src/gpu/previews) and a
 * static SVG fallback rendered during prerendering, both drawn from the same seeded data.
 */
export type PreviewKind =
  | 'points'
  | 'point-clusters'
  | 'hotspots'
  | 'density-surface'
  | 'interpolation'
  | 'raster'
  | 'terrain'
  | 'flow-accumulation'
  | 'contours'
  | 'network'
  | 'shortest-path'
  | 'isochrones'
  | 'flows'
  | 'lines'
  | 'polygons'
  | 'polygon-overlay'
  | 'buffers'
  | 'tessellation'
  | 'cells'
  | 'spatial-weights'
  | 'regression'
  | 'histogram'
  | 'classification'
  | 'time-series'
  | 'trajectories'
  | 'point-cloud'
  | 'pipeline'
  | 'affine-transform'
  | 'cell-cover';

export type PreviewSpec = {
  kind: PreviewKind;
  /** Short caption under the preview, at most 80 characters. */
  caption?: string;
  /** Deterministic seed so the live scene and the static fallback match. */
  seed?: number;
};
