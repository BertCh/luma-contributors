// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateLineNodingBounds} from '../geometry-topology/gpu-line-noding';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Largest finite f32: shader literals must survive their construction-time conversion. */
export const MAXIMUM_REFINEMENT_SHADER_MAGNITUDE = 3.4028234663852886e38;
/** WebGPU may flush subnormals; positive divisors and stopping tolerances must remain nonzero. */
export const MINIMUM_REFINEMENT_SHADER_POSITIVE_VALUE = 2 ** -126;

export function validateRefinementShaderNumber(
  id: string,
  name: string,
  value: number,
  positive = false
): void {
  const minimum = positive ? MINIMUM_REFINEMENT_SHADER_POSITIVE_VALUE : 0;
  validateFiniteNumber(id, name, value, 0, MAXIMUM_REFINEMENT_SHADER_MAGNITUDE);
  validateFiniteNumber(
    id,
    `${name} converted to float32`,
    Math.fround(value),
    minimum,
    MAXIMUM_REFINEMENT_SHADER_MAGNITUDE
  );
}

/** Checks every aligned derived allocation, including nested contributors, before declaration. */
export function validateRefinementScratchByteLengths<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  byteLengths: Record<string, number>
): void {
  for (const [name, bytes] of Object.entries(byteLengths)) {
    const byteLength = Math.max(4, bytes);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'split the input or reduce capacity'
    );
    validateCapacity(id, `${name} byteLength`, byteLength, 4, graph.device.limits.maxBufferSize);
  }
}

/** The nested splitter's largest tables; scans and sort columns are smaller than these tables. */
export function validateRefinementLineNodingScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  vertexCount: number,
  lineCount: number,
  intersectionCapacity: number
): void {
  validateLineNodingBounds(id, graph.device, vertexCount, lineCount, intersectionCapacity);
  validateRefinementScratchByteLengths(graph, id, {
    segmentTableAndHierarchy: Math.max(vertexCount, 1) * 32,
    splitPieceTable: (lineCount + intersectionCapacity * 4) * 32,
    nodedPiecePositions: (vertexCount + intersectionCapacity * 8) * 8
  });
}
