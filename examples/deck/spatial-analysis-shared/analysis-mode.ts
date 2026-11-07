// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CommandEncoder} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import type {SpatialAnalysisColor} from './layer-constants';

/**
 * Common shape of one spatial analysis, shared by the explorer (`modes/*`) and the showcase
 * (`scenes/**`). An analysis is split in three parts that the apps combine differently:
 *
 * - {@link AnalysisModeCompute}: the GPU compute graph. Owns buffers, compiled graphs, parameter
 *   writes, summary readback and decoding. Knows nothing about Deck, the DOM or either app.
 * - {@link AnalysisModeStyle}: legend entries, explanatory text and palette selection, as plain
 *   data derived from the current parameters.
 * - {@link AnalysisModeSceneHooks}: the only calls back into an app. Each app keeps a thin adapter
 *   that wires controls to `setParameters`, reads the style, and forwards the hooks to its own
 *   scene, readouts and layers.
 */

/** Legend entry: a swatch color and its label. */
export type AnalysisLegendEntry = {color: SpatialAnalysisColor; label: string};

/** Legend: a title and entries, independent of how an app draws them. */
export type AnalysisLegend = {title: string; entries: readonly AnalysisLegendEntry[]};

/** Calls from a compute back into the hosting app. */
export type AnalysisModeSceneHooks<Summary> = {
  /** A readback completed. `summary` is decoded; the app formats it as readouts or charts. */
  publishSummary: (summary: Summary) => void;
};

/** The GPU compute graph of an analysis. */
export type AnalysisModeCompute<Parameters> = {
  /** Compiled graphs this compute encodes; the set never changes while parameters move. */
  getCompiledGraphs: () => readonly CompiledGPUCommandGraph<never>[];
  /** Current parameters. */
  getParameters: () => Readonly<Parameters>;
  /**
   * Merges parameter changes. Per-frame parameters are buffer writes and compile-time options only
   * select between graphs that are already compiled; nothing is recompiled. Marks results stale.
   */
  setParameters: (parameters: Partial<Parameters>) => void;
  /** Encodes the graphs that need to run and schedules a summary readback. Never compiles. */
  encode: (commandEncoder: CommandEncoder, frameIndex: number) => void;
  /** Releases every GPU resource the compute created. */
  destroy: () => void;
};

/** Presentation of an analysis as data. */
export type AnalysisModeStyle<Parameters> = {
  /** Legends valid for the parameters. */
  getLegends: (parameters: Readonly<Parameters>) => readonly AnalysisLegend[];
  /** One-paragraph explanation of what the current parameters compute. */
  describe: (parameters: Readonly<Parameters>) => string;
};
