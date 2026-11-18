// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Private per-vertex arrays bound the reference route. @internal */
export const NETWORK_FLOW_REFERENCE_MAXIMUM_NODE_COUNT = 128;
/** The reference invocation scans every original edge per relaxation. @internal */
export const NETWORK_FLOW_REFERENCE_MAXIMUM_EDGE_COUNT = 1024;
/** Reference flow arithmetic preserves exact float32 integers through this total. @internal */
export const NETWORK_FLOW_REFERENCE_MAXIMUM_TOTAL_CAPACITY = 0xffffff;
/** Parallel CSR storage is further bounded by device bindings and command schedules. @internal */
export const NETWORK_FLOW_PARALLEL_MAXIMUM_NODE_COUNT = 30_000_000;
/** Residual arcs use two slots per original edge. Device bindings also apply. @internal */
export const NETWORK_FLOW_PARALLEL_MAXIMUM_EDGE_COUNT = 30_000_000;
/** Flow and cut outputs occupy one unsigned word. @internal */
export const NETWORK_FLOW_MAXIMUM_TOTAL_CAPACITY = 0xffffffff;
/** Both routes bound nonnegative original costs to this value. @internal */
export const NETWORK_FLOW_MAXIMUM_UNIT_COST = 32767;
/** Signed residual distances reserve INT_MAX for unreachable. @internal */
export const NETWORK_FLOW_MAXIMUM_DISTANCE = 0x7ffffffe;
