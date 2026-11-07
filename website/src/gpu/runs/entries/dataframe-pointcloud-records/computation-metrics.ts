// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// createGPUComputationMetrics over a real encoding. The relaxation graph (with `taps` snapshot
// copies between iterations) is encoded again with the slider's `coalesced` flag, 400 times, and
// each node's mean CPU encode time becomes its `cpuEncodeMs` (one encode of a small node is far
// below timer resolution; the device here has no GPU timestamps, so no gpuBegin or gpuEnd).
// The sidecar is drawn as one bar per node in scheduled order, colored by passIndex, and the total
// time of every physical pass on the right. Coalescing merges runs of compute nodes into one pass;
// a tap is a copy node and ends the pass.

import {
  createGPUComputationMetrics,
  describeGPUCommandGraph,
  getGPUCommandGraphPassIndices,
  GPUSnapshotArena
} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun} from '../../define-run';
import {buildRelaxation, createRecordTracts, ShapeBatch} from '../../record-demo';
import {parameters} from './computation-metrics.parameters';

const ENCODE_COUNT = 400;

export default defineOperationRun({
  parameters,
  caption: 'Per-node timing of a real encoding, colored by the pass each node ran in.',
  createInputs: ({random, world}) => ({tracts: createRecordTracts(random), world}),
  build(builder, {tracts}, values) {
    // Taps copy into an arena that is never read here; it has room for every repeated encoding.
    const arena = builder.own(
      new GPUSnapshotArena(builder.device, {
        id: 'metrics-taps',
        slots: [{name: 'state', byteLength: tracts.count * 4, elementType: 'f32'}],
        capacity: Math.max(values.taps, 1) * (ENCODE_COUNT + 2)
      })
    );
    const tapEvery = values.taps === 0 ? 0 : Math.ceil(values.iterations / values.taps);
    buildRelaxation(builder, tracts, {
      iterations: values.iterations,
      tap: iteration =>
        tapEvery > 0 && iteration.iteration % tapEvery === 0
          ? {arena, views: {state: iteration.state}}
          : undefined
    });
    return {outputs: {}};
  },
  async afterRun({compiled, device, inputs, values, upload}) {
    const {width} = inputs.world;
    const isCoalesced = values.coalesced === 1;
    const sums = new Map<string, number>();
    let encoding: ReturnType<typeof compiled.encode> | undefined;
    for (let repeat = 0; repeat < ENCODE_COUNT; repeat++) {
      const encoder = device.createCommandEncoder({id: 'metrics-encoder'});
      encoding = compiled.encode(encoder, {
        parameters: undefined,
        coalesceComputePasses: isCoalesced
      });
      encoder.finish();
      for (const node of encoding.stats.nodes) {
        sums.set(node.id, (sums.get(node.id) ?? 0) + node.cpuEncodeTimeMilliseconds);
      }
    }
    const stats = encoding!.stats;
    const timing = {
      nodes: stats.nodes.map(node => ({
        ...node,
        cpuEncodeTimeMilliseconds: (sums.get(node.id) ?? 0) / ENCODE_COUNT
      }))
    };
    const described = describeGPUCommandGraph(compiled).graph;
    const metrics = createGPUComputationMetrics({
      runId: 'run-metrics',
      graph: described.id,
      timing,
      encoding: {
        coalesced: stats.coalescedComputeNodeCount > 0,
        timestamped: false,
        taps: stats.nodes.filter(node => node.type === 'copy').length,
        lensTwin: false
      },
      passIndices: getGPUCommandGraphPassIndices(stats)
    });

    // One bar per node, from the left, as long as its cpuEncodeMs.
    const rows = stats.nodes.map(({id}) => ({id, node: metrics.nodes[id]}));
    const spans = rows.map(({node}) => ({begin: 0, end: node.cpuEncodeMs, pass: node.passIndex}));
    const total = Math.max(...spans.map(span => span.end), 1e-6);
    const area = {x0: 0.05 * width, x1: 0.64 * width, y0: 0.06, y1: 0.95};
    const rowHeight = (area.y1 - area.y0) / Math.max(rows.length, 1);
    const bars = new ShapeBatch();
    const passTotals = new Map<number, number>();
    spans.forEach((span, row) => {
      const x0 = area.x0 + ((area.x1 - area.x0) * span.begin) / total;
      const x1 = Math.max(area.x0 + ((area.x1 - area.x0) * span.end) / total, x0 + 0.004 * width);
      bars.addRectangle(
        {x0, x1, y1: area.y1 - row * rowHeight, y0: area.y1 - (row + 0.8) * rowHeight},
        span.pass === undefined ? 0 : 1 + (span.pass % 7)
      );
      if (span.pass !== undefined) {
        passTotals.set(span.pass, (passTotals.get(span.pass) ?? 0) + span.end - span.begin);
      }
    });
    const passes = [...passTotals.entries()].sort((first, second) => first[0] - second[0]);
    const passBars = Float32Array.from(passes.length ? passes.map(([, time]) => time) : [0]);
    const passColors = new ShapeBatch();
    passes.forEach(([pass], index) => {
      // A strip of swatches under the chart, in the same pass colors as the timeline.
      const slot = (0.27 * width) / Math.max(passes.length, 1);
      passColors.addRectangle(
        {
          x0: 0.7 * width + index * slot + slot * 0.1,
          x1: 0.7 * width + (index + 1) * slot - slot * 0.1,
          y0: 0.03,
          y1: 0.06
        },
        1 + (pass % 7)
      );
    });
    return {
      bars: bars.uploadShapes(upload, 'timeline'),
      passBars: upload('pass-bars', 'float32', passBars),
      passMaximum: Math.max(...passBars, 1e-6),
      swatches: passColors.uploadShapes(upload, 'pass-swatches')
    };
  },
  draw: ({extra, theme}) => [
    {
      layer: 'polygons',
      positions: extra.bars.positions,
      indices: extra.bars.indices,
      owners: extra.bars.owners,
      values: extra.bars.categories,
      triangleCount: extra.bars.triangleCount,
      categories: [theme.muted, ...theme.categories]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: extra.passBars,
      range: [0, extra.passMaximum],
      rectangle: [0.7, 0.08, 0.97, 0.9],
      color: [...theme.categories[3], 0.9]
    },
    {
      layer: 'polygons',
      positions: extra.swatches.positions,
      indices: extra.swatches.indices,
      owners: extra.swatches.owners,
      values: extra.swatches.categories,
      triangleCount: extra.swatches.triangleCount,
      categories: [theme.muted, ...theme.categories]
    }
  ]
});
