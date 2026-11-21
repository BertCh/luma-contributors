// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPULineSmooth} from '../../../src/gpu-spatial-analysis/line-segmentize/index';
import type {OracleView} from '../line-merge/line-merge-output-oracle-fixture';

export function createSmoothOracleFixture(
  view: OracleView,
  optional: 'required' | 'sources' | 'absent'
) {
  const positions = view('float32x2', 8);
  const lineOffsets = view('uint32', 3);
  const count = view('uint32', 1);
  const overflow = view('uint32', 1);
  const requiredCount = optional === 'required' ? view('uint32', 1) : undefined;
  const sourceLineIndices = optional === 'sources' ? view('uint32', 2) : undefined;
  const contributor = new GPULineSmooth({
    positions: view('float32x2', 4, [0, 0, 4, 0, 10, 2, 14, 2]),
    lineOffsets: view('uint32', 3, [0, 2, 4]),
    parameters: view('float32', 4, [0.25, 0, 0, 0]),
    iterationCount: 1,
    output: {positions, lineOffsets, sourceLineIndices, status: {count, overflow, requiredCount}}
  });
  // Chaikin inserts quarter/three-quarter points and keeps endpoints, independently
  // for each two-point input line. It preserves two lines with input rows 0 and 1.
  return {contributor, positions, lineOffsets, count, overflow, requiredCount, sourceLineIndices};
}
