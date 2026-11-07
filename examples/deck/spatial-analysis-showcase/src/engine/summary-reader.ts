// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {
  SummaryReader as SharedSummaryReader,
  type SummaryReaderResources
} from '../../../spatial-analysis-shared/summary-reader';

/** Scene id for log lines, read from the test hook the shell publishes. */
function activeSceneId(): string {
  return globalThis.spatialAnalysisShowcase?.sceneId ?? 'no-scene';
}

/** Shared summary readback that logs callback and readback failures with the active scene id. */
export class SummaryReader extends SharedSummaryReader {
  constructor(
    resources: SummaryReaderResources,
    id: string,
    sources: readonly {buffer: Buffer; size: number}[],
    onResult: (bytes: ArrayBuffer) => void
  ) {
    super(resources, id, sources, onResult, {logLabel: `${activeSceneId()}/${id}`});
  }
}
