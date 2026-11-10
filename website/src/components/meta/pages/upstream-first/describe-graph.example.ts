// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describeGPUCommandGraph} from '@luma-contributors/analysis/computation-record';

// example:hide-start
/** Any graph, built by any code, compiled by published luma.gl. */
declare const compiled: CompiledGPUCommandGraph<void>;
// example:hide-end
// luma.gl publishes what the compiler decided: the order, and per node its buffers and usage.
const order = compiled.stats.nodeOrder;
const firstNode = compiled.preflight.nodes[0];
const firstNodeBuffers = (firstNode.resources ?? []).map(resource => resource.id);

// The description reads only those fields. No hook, no private field, no patched graph.
const {graph} = describeGPUCommandGraph(compiled);
const hazards = graph.edges.filter(edge => ['raw', 'war', 'waw'].includes(edge.kind));
const bufferBytes = (graph.resources ?? []).reduce((total, resource) => total + resource.bytes, 0);

// example:hide-start
export {order, firstNodeBuffers, hazards, bufferBytes};
// example:hide-end
