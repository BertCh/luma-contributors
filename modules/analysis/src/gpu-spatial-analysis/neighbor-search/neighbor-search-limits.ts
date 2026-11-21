// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Largest kNN list: two private arrays use 2 KiB; queryRows * k * 4 bytes must also fit one storage binding. */
export const GPU_NEIGHBOR_SEARCH_MAXIMUM_K = 256;

/** Largest temporal bucket count; gridSize[0] * gridSize[1] * timeBucketCount + 1 offsets must fit one binding. */
export const GPU_NEIGHBOR_SEARCH_MAXIMUM_TIME_BUCKETS = 65536;
