// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getTypedArrayName,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Compile-time dimensions of the query-major, padded GPUKShortestPaths output. */
export type GPUKShortestPathsOutputLayoutOptions = {
  /** ID used in validation messages. */
  id?: string;
  queryCount: number;
  routeCount: number;
  /** Vertex slots per route; defaults to 64, including for the equally sized edge output. */
  maximumPathLength?: number;
};

/**
 * Length and scalar format of every output, with batch scalars nested under status.
 * Counts and products must fit uint32 row counts. GPU execution budgets and device limits
 * are checked by GPUKShortestPaths; this layout also describes the CPU twin's larger budgets.
 */
export function getGPUKShortestPathsOutputLayout(options: GPUKShortestPathsOutputLayoutOptions) {
  const {
    id = 'getGPUKShortestPathsOutputLayout',
    queryCount,
    routeCount,
    maximumPathLength = 64
  } = options;
  validateCapacity(id, 'queryCount', queryCount);
  validateCapacity(id, 'routeCount', routeCount);
  validateCapacity(id, 'maximumPathLength', maximumPathLength);
  const routeRows = queryCount * routeCount;
  const pathRows = routeRows * maximumPathLength;
  validateCapacity(id, 'queryCount * routeCount', routeRows);
  validateCapacity(id, 'queryCount * routeCount * maximumPathLength', pathRows);
  const unsigned = (length: number) => ({format: 'uint32' as const, length});
  return {
    vertices: unsigned(pathRows),
    edges: unsigned(pathRows),
    lengths: unsigned(routeRows),
    costs: {format: 'float32' as const, length: routeRows},
    counts: unsigned(queryCount),
    status: {
      converged: unsigned(1),
      iterationLimitReached: unsigned(1),
      candidateOverflow: unsigned(1),
      iterationCount: unsigned(1)
    }
  };
}

export type GPUKShortestPathsOutputLayout = ReturnType<typeof getGPUKShortestPathsOutputLayout>;

/** Readback arrays for GPUKShortestPaths; views may have byte offsets and spare capacity. */
export type GPUKShortestPathsReadback = {
  vertices: Uint32Array;
  edges: Uint32Array;
  lengths: Uint32Array;
  costs: Float32Array;
  counts: Uint32Array;
  status: {
    converged: Uint32Array;
    iterationLimitReached: Uint32Array;
    candidateOverflow: Uint32Array;
    iterationCount: Uint32Array;
  };
};

/** Only accepted routes, in query and route order; status still describes the whole batch. */
export type GPUKShortestPathsDecodedResult = {
  queries: {
    count: number;
    routes: {length: number; cost: number; vertices: number[]; edges: number[]}[];
  }[];
  status: {
    converged: boolean;
    iterationLimitReached: boolean;
    candidateOverflow: boolean;
    iterationCount: number;
  };
};

/**
 * Decodes retained routes without their padding, checking formats, required lengths, accepted
 * counts, path lengths and nonnegative costs (including positive infinity from CPU float32
 * overflow; NaN is rejected). Uses the supplied typed-array views,
 * including their byte offsets, and ignores capacity beyond the declared layout. Returned
 * arrays are copies. Incomplete batches retain their routes and expose all interruption flags;
 * a caller must inspect status before treating them as a complete enumeration.
 */
export function decodeGPUKShortestPaths(
  output: GPUKShortestPathsReadback,
  options: GPUKShortestPathsOutputLayoutOptions
): GPUKShortestPathsDecodedResult {
  const {id = 'decodeGPUKShortestPaths', queryCount, routeCount, maximumPathLength = 64} = options;
  const {status: statusLayout, ...columnLayout} = getGPUKShortestPathsOutputLayout({
    id,
    queryCount,
    routeCount,
    maximumPathLength
  });
  validateRequirement(
    id,
    output.status !== null && typeof output.status === 'object',
    'output.status must contain the required status arrays'
  );
  for (const [name, descriptor] of Object.entries({...columnLayout, ...statusLayout})) {
    const isStatus = Object.hasOwn(statusLayout, name);
    const array = isStatus
      ? output.status[name as keyof typeof output.status]
      : output[name as keyof typeof columnLayout];
    const path = `output.${isStatus ? 'status.' : ''}${name}`;
    const ArrayType = descriptor.format === 'float32' ? Float32Array : Uint32Array;
    validateRequirement(
      id,
      getTypedArrayName(array) === ArrayType.name,
      `${path} must be a ${ArrayType.name}`
    );
    validateCapacity(id, `${path}.length`, array.length, descriptor.length);
  }
  for (const name of ['converged', 'iterationLimitReached', 'candidateOverflow'] as const) {
    validateCapacity(id, `output.status.${name}[0]`, output.status[name][0], 0, 1);
  }
  validateCapacity(
    id,
    'output.status.iterationCount[0]',
    output.status.iterationCount[0],
    0,
    routeCount
  );
  const queries: GPUKShortestPathsDecodedResult['queries'] = [];
  for (let query = 0; query < queryCount; query++) {
    const count = output.counts[query];
    validateCapacity(id, `output.counts[${query}]`, count, 0, routeCount);
    const routes: GPUKShortestPathsDecodedResult['queries'][number]['routes'] = [];
    for (let route = 0; route < count; route++) {
      const row = query * routeCount + route;
      const length = output.lengths[row];
      const cost = output.costs[row];
      validateCapacity(id, `output.lengths[${row}]`, length, 1, maximumPathLength);
      validateRequirement(id, cost >= 0, `output.costs[${row}] must be nonnegative and not NaN`);
      const start = row * maximumPathLength;
      routes.push({
        length,
        cost,
        vertices: Array.from(output.vertices.subarray(start, start + length)),
        edges: Array.from(output.edges.subarray(start, start + length - 1))
      });
    }
    queries.push({count, routes});
  }
  return {
    queries,
    status: {
      converged: output.status.converged[0] !== 0,
      iterationLimitReached: output.status.iterationLimitReached[0] !== 0,
      candidateOverflow: output.status.candidateOverflow[0] !== 0,
      iterationCount: output.status.iterationCount[0]
    }
  };
}
