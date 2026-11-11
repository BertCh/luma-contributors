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
import type {RunGlyph} from '../../interaction';
import {buildRelaxation, createRecordTracts, ShapeBatch} from '../../record-demo';
import {createChartAxes, getChartRectangle, mergeText, type RunLabel} from '../../text';
import {getBoxGlyph, type Box} from './_lib/record-probe';
import {parameters} from './computation-metrics.parameters';

const ENCODE_COUNT = 400;
const PASS_RECTANGLE = [0.7, 0.08, 0.97, 0.84] as const;

/** What the last run drew, for the pointer: one row per node, with its bar's extent. */
type Latest = {
  area: Box;
  rowHeight: number;
  rows: {id: string; pass: number | undefined; milliseconds: number; x1: number}[];
  /** Pass numbers in bar order and each pass's total milliseconds, for the right chart. */
  passes: [number, number][];
  passMaximum: number;
};

export default defineOperationRun({
  parameters,
  caption: 'Per-node timing of a real encoding, colored by the pass each node ran in.',
  createInputs: ({random, world}) => ({
    tracts: createRecordTracts(random),
    world,
    latest: null as Latest | null
  }),
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
    const area = {x0: 0.05 * width, x1: 0.64 * width, y0: 0.08, y1: 0.89};
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
    inputs.latest = {
      passes,
      passMaximum: Math.max(...passBars, 1e-6),
      area,
      rowHeight,
      rows: rows.map(({id, node}, row) => ({
        id,
        pass: node.passIndex,
        milliseconds: node.cpuEncodeMs,
        x1: Math.max(
          area.x0 + ((area.x1 - area.x0) * spans[row].end) / total,
          area.x0 + 0.004 * width
        )
      }))
    };
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
      passes,
      total,
      bars: bars.uploadShapes(upload, 'timeline'),
      passBars: upload('pass-bars', 'float32', passBars),
      passMaximum: Math.max(...passBars, 1e-6),
      swatches: passColors.uploadShapes(upload, 'pass-swatches')
    };
  },
  // The pointer is a node: its bar is outlined and its pass is marked on the right chart (the colour of the
  // bars already groups the nodes of a pass), and the readout gives its pass and encode time.
  interaction: {
    kind: 'probe',
    hint: 'hover a node',
    rest: ({inputs}) => {
      const latest = inputs.latest;
      if (!latest) return [0.3, 0.2];
      return [0.3, 1 - (latest.area.y1 - latest.rowHeight * 0.4)];
    },
    resolve(pointer, {inputs}) {
      const latest = inputs.latest;
      if (!latest || latest.rows.length === 0) return {readout: 'hover a node'};
      const row = Math.min(
        latest.rows.length - 1,
        Math.max(0, Math.floor((latest.area.y1 - pointer.y) / latest.rowHeight))
      );
      const hovered = latest.rows[row];
      const barBox = (index: number): Box => ({
        x0: latest.area.x0,
        x1: latest.rows[index].x1,
        y1: latest.area.y1 - index * latest.rowHeight,
        y0: latest.area.y1 - (index + 0.8) * latest.rowHeight
      });
      const members = latest.rows.filter(other => other.pass === hovered.pass).length;
      // The same pass on the right chart: its slot is outlined and a rule marks its total.
      const passSlot = latest.passes.findIndex(([pass]) => pass === hovered.pass);
      const chart = [
        PASS_RECTANGLE[0] * inputs.world.width,
        PASS_RECTANGLE[1],
        PASS_RECTANGLE[2] * inputs.world.width,
        PASS_RECTANGLE[3]
      ] as const;
      const slotWidth = (chart[2] - chart[0]) / Math.max(latest.passes.length, 1);
      const passGlyphs: RunGlyph[] =
        passSlot < 0
          ? []
          : [
              {
                shape: 'box',
                minimum: [chart[0] + passSlot * slotWidth, chart[1]],
                maximum: [chart[0] + (passSlot + 1) * slotWidth, chart[3]],
                dashed: true
              },
              {
                shape: 'line',
                from: [
                  chart[0],
                  chart[1] +
                    (latest.passes[passSlot][1] / latest.passMaximum) * (chart[3] - chart[1])
                ],
                to: [
                  chart[2],
                  chart[1] +
                    (latest.passes[passSlot][1] / latest.passMaximum) * (chart[3] - chart[1])
                ],
                dashed: true
              }
            ];
      return {
        spotlight: null,
        glyphs: [getBoxGlyph(barBox(row)), ...passGlyphs],
        readout: `${hovered.id} · ${
          hovered.pass === undefined ? 'copy, no pass' : `pass ${hovered.pass} (${members} nodes)`
        } · ${hovered.milliseconds.toFixed(4)} ms`
      };
    }
  },
  draw: ({extra, theme, world}) => {
    const rectangle = getChartRectangle(world, PASS_RECTANGLE);
    const axes = createChartAxes({
      rectangle,
      y: {range: [0, extra.passMaximum], ticks: 3}
    });
    const axisLabels = axes.labels ?? [];
    // The value labels go left of the bars (the default writes them inside the plot, on the bars).
    const topTick = Math.max(
      ...axisLabels.filter(label => label.group === 'y-ticks').map(label => label.at[1])
    );
    const tickLabels = axisLabels
      .filter(label => label.group === 'y-ticks')
      .map(label => ({
        ...label,
        text: label.at[1] === topTick ? `${label.text} ms` : label.text,
        anchor: 'right' as const,
        offset: [-4, 0] as const
      }));
    const axisText = {
      rules: axes.rules,
      labels: [...axisLabels.filter(label => label.group !== 'y-ticks'), ...tickLabels]
    };
    const area = {x0: 0.05 * world.width, x1: 0.64 * world.width};
    const slot = (0.27 * world.width) / Math.max(extra.passes.length, 1);
    const labels: RunLabel[] = [
      {
        text: 'encode time of each node (scheduled order), colour = pass',
        at: [area.x0, 0.995],
        anchor: 'top-left',
        role: 'title'
      },
      {text: '0', at: [area.x0, 0.075], anchor: 'top-left', role: 'tick'},
      {
        text: `${extra.total.toFixed(4)} ms`,
        at: [area.x1, 0.075],
        anchor: 'top-right',
        role: 'tick'
      },
      {
        text: 'total time per pass',
        at: [PASS_RECTANGLE[0] * world.width, 0.995],
        anchor: 'top-left',
        role: 'title'
      },
      ...extra.passes.map(([pass], index) => ({
        text: String(pass),
        at: [0.7 * world.width + (index + 0.5) * slot, 0.045] as const,
        anchor: 'center' as const,
        role: 'tick' as const,
        group: 'pass-numbers'
      })),
      {
        text: 'pass',
        at: [0.7 * world.width - 0.01, 0.045],
        anchor: 'right',
        role: 'tick'
      }
    ];
    return [
      {layer: 'text', ...axes.grid},
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
        rectangle: PASS_RECTANGLE,
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
      },
      {layer: 'text', ...mergeText({labels}, axisText)}
    ];
  }
});
