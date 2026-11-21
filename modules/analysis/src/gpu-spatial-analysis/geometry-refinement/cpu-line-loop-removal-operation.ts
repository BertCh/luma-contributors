// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateCPULineFeatureLayout} from '../../analysis-core/contracts/line-feature-provenance';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {nodeCPURefinementLines, type CPURefinementPoint} from './cpu-refinement-line-noding';
import {
  GPULineLoopRemovalOperation,
  snapshotLineLoopRemovalOperationProps,
  type LineLoopRemovalOperationProps
} from './gpu-line-loop-removal-operation';

/** Packed planar open curves; repeated closure erases the entire closed loop. */
export type CPULineLoopRemovalInput = {
  positions: Float32Array;
  lineOffsets: Uint32Array;
  featureOffsets?: Uint32Array;
};

/** Caller-sized output and optional declared planar context. */
export type CPULineLoopRemovalOptions = Pick<
  LineLoopRemovalOperationProps,
  'vertexCapacity' | 'spatialContext'
> & {
  intersectionCapacity?: number /** Publish identity feature offsets for ungrouped input. */;
  includeFeatureOffsets?: boolean;
};

/** The contributor's four output graph values, with zeroed unused rows. */
export type CPULineLoopRemovalResult = {
  positions: Float32Array;
  lineOffsets: Uint32Array;
  featureOffsets?: Uint32Array;
  sourceEdges: Uint32Array;
  /** [writtenVertices,requiredVertices,outputOverflow,incompleteNoding]. */
  status: Uint32Array;
};

/**
 * Nodes open planar curves at crossings, touches and overlap endpoints, then erases loops in
 * traversal order using exact visit keys. A repeat replaces the retained vertex's outgoing source
 * edge. The final vertex inherits its predecessor's source edge, matching the contributor. Lines
 * shorter than two retained vertices disappear; overflow drops whole lines in input order, and a
 * later shorter line can still fit.
 *
 * What differs from the GPU contributor:
 * - An explicit intersectionCapacity binds the noding pair list. Exhaustion suppresses CPU
 *   geometry and sets incompleteNoding=1; the GPU can keep an incomplete prefix instead.
 *   Omitted capacity is uncapped. GPU intermediate binding and successor-scan limits are dropped.
 * - Predicates and visit identities use exact rationals of float32 input coordinates. Coordinates
 *   use float64 and round once at output; the GPU rounds crossings before loop erasure. Distinct
 *   exact crossings that round to the same float32 point remain distinct in the twin. The twin
 *   never reports uncertain predicates.
 * - Unused positions/sourceEdges are zeroed; the GPU leaves them untouched on reused buffers.
 * - Empty coordinate arrays with valid zero offsets are accepted and return zero-filled outputs
 *   with all-zero status; the GPU line-split delegate requires at least one coordinate row.
 * - Inconsistent offsets throw instead of relying on out-of-bounds shader reads. Nonfinite
 *   vertices set incompleteNoding=1, their edges are not intersected, and their original values
 *   are retained in the traversal (NaNs never count as repeated visits).
 *
 * @throws Error naming lines.positions, lines.lineOffsets or vertexCapacity for inconsistent input.
 */
export function computeLineLoopRemovalOnCPU(
  input: CPULineLoopRemovalInput,
  options: CPULineLoopRemovalOptions
): CPULineLoopRemovalResult {
  validateCapacity('line-loop-removal', 'vertexCapacity', options.vertexCapacity, 0);
  resolveGPUSpatialContext('line-loop-removal', options.spatialContext);
  if (options.intersectionCapacity !== undefined)
    validateCapacity('line-loop-removal', 'intersectionCapacity', options.intersectionCapacity);
  const activeParts = validateCPULineFeatureLayout(
    'line-loop-removal',
    input.positions,
    input.lineOffsets,
    input.featureOffsets,
    'lines',
    true
  );
  const noded = nodeCPURefinementLines(
    input.positions,
    input.lineOffsets.subarray(0, activeParts + 1),
    {
      intersectionCapacity: options.intersectionCapacity
    }
  );
  const result: CPULineLoopRemovalResult = {
    positions: new Float32Array(options.vertexCapacity * 2),
    lineOffsets: new Uint32Array(input.lineOffsets.length),
    sourceEdges: new Uint32Array(options.vertexCapacity),
    status: new Uint32Array(4),
    ...(input.featureOffsets || options.includeFeatureOffsets
      ? {
          featureOffsets:
            input.featureOffsets?.slice() ??
            Uint32Array.from({length: input.lineOffsets.length}, (_, row) => row)
        }
      : {})
  };
  let required = 0;
  let written = 0;
  for (let line = 0; line < noded.lines.length; line++) {
    const traversal = noded.lines[line];
    const stack: {point: CPURefinementPoint; source: number; key: string}[] = [];
    const visits = new Map<string, number>();
    if (traversal.length >= 2) {
      for (let vertex = 0; vertex < traversal.length; vertex++) {
        const {point, source} = traversal[vertex];
        const key = Number.isNaN(point.x) || Number.isNaN(point.y) ? `nan:${vertex}` : point.key;
        const found = visits.get(key);
        if (found !== undefined) {
          while (stack.length > found + 1) visits.delete(stack.pop()!.key);
        } else {
          visits.set(key, stack.length);
          stack.push({point, source: 0, key});
        }
        const current = stack[stack.length - 1];
        current.point = point;
        if (vertex + 1 < traversal.length) current.source = source & 0x7fffffff;
        else if (stack.length > 1) current.source = stack[stack.length - 2].source;
      }
    }
    const length = stack.length >= 2 ? stack.length : 0;
    result.lineOffsets[line] = written;
    required += length;
    if (written + length > options.vertexCapacity) continue;
    for (const entry of length === 0 ? [] : stack) {
      result.positions.set([entry.point.x, entry.point.y], written * 2);
      result.sourceEdges[written] = entry.source;
      written++;
    }
  }
  result.lineOffsets.fill(written, activeParts);
  result.status.set([written, required, Number(written < required), noded.incomplete]);
  return result;
}

/** CPU loop erasure twin; explicit intersectionCapacity caps exact noding. */
export class CPULineLoopRemovalOperation extends GPUCPUOperation {
  constructor(props: LineLoopRemovalOperationProps) {
    props = snapshotLineLoopRemovalOperationProps(props);
    const id = props.id ?? 'line-loop-removal';
    validateCapacity(id, 'lineCount', props.lineCount);
    validateCapacity(id, 'vertexCapacity', props.vertexCapacity, 0);
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    resolveGPUSpatialContext(id, props.spatialContext);
    const operation = new GPULineLoopRemovalOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const lineOffsets = getCPUOperationUint32Rows(id, inputs, 'inputOffsets');
        if (lineOffsets.length !== props.lineCount + 1)
          throw new Error(`${id} lines.lineOffsets must have lineCount + 1 rows`);
        const featureOffsets = props.lines.featureOffsets
          ? getCPUOperationUint32Rows(id, inputs, 'inputFeatures')
          : undefined;
        if (featureOffsets && featureOffsets.length !== props.featureCount! + 1)
          throw new Error(`${id} lines.featureOffsets must have featureCount + 1 rows`);
        const result = computeLineLoopRemovalOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'inputPositions', 'float32x2'),
            lineOffsets,
            ...(featureOffsets ? {featureOffsets} : {})
          },
          {...props, includeFeatureOffsets: Boolean(props.output.featureOffsets)}
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPULineLoopRemovalResult]!
          ])
        );
      }
    });
  }
}
