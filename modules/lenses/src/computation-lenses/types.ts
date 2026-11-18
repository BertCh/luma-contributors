// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  GPUComputationRecord,
  GPUComputationRecordWithSidecars
} from '@luma-contributors/analysis/computation-record';
import type {LensRungId} from '../computation-views/types';
import type {LensRecordEncoding} from './record-encoding';

/** Scenario identifier: `<entry>.<element>[.<variant>]`, with kebab-case segments. */
export type LensScenarioId = string;
/** One real execution with optional measurement and capture sidecars. */
export type LensScenarioResult = GPUComputationRecordWithSidecars;

/** Lazily loaded GPU scenario. Register descriptors with registerLensScenario at module scope. */
export type LensScenarioDescriptor = {
  /** `<entry>.<element-kebab>[.<variant>]`, e.g. `gpu-spatial-analysis.local-moran`. */
  id: LensScenarioId;
  /** Documented entry; must equal the first id segment, e.g. `gpu-spatial-analysis`. */
  entry: string;
  /**
   * Documented export this scenario covers: the exported class name (`GPULocalMoran`), or the
   * function name for recipes. Pack `operators` and `formulas` use the same key.
   */
  element: string;
  title: string;
  audience: 'analyst' | 'engineer';
  rungs: readonly LensRungId[];
  defaultRung: LensRungId;
  /** Defaults to getLensExamplePath(id); required by the coverage spec once scenarios exist. */
  example?: {path: string};
  dataset?: {id: string; licence: string; source: string};
  /** Result presentation shared by live hosts and static twins. */
  presentation?: {
    /** Final output capture ID used for the headline and result grammar; absent selects the last output. */
    primaryOutputCaptureId?: string;
  };
  /** Imports GPU code only when explicitly activated; never during a docs build. */
  load(): Promise<{
    run(device: Device, options?: {signal?: AbortSignal}): Promise<LensScenarioResult>;
    describeStructure?(device: Device): Promise<GPUComputationRecord>;
  }>;
};

/** Provenance shared by encoded files and decoded recordings. */
export type LensRecordingMetadata = {
  schema: 'luma.lens-record/1';
  scenarioId: LensScenarioId;
  recordedAt: string;
  adapterLabel: string;
  variant?: string;
};

/**
 * JSON-safe file returned by toRecordedLensFile and captureLensScenarioRecord. Its opaque
 * payload contains compact graphs and encoded capture values, not a LensScenarioResult.
 * Pass it (or parsed legacy JSON) to fromRecordedLensFile before inspecting or rendering data.
 */
export type RecordedLensFile = LensRecordingMetadata & {
  encoding?: LensRecordEncoding;
  record: unknown;
};

/** In-memory recording returned by fromRecordedLensFile, with graph objects and numeric values. */
export type DecodedLensFile = LensRecordingMetadata & {
  record: LensScenarioResult;
};
