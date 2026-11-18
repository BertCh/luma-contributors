// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// createGPUComputationRecord over a recorded spatial-lag relaxation. The top shows graphs[0] of the
// finished record (nodes colored by source id), the bottom left the resource lifetimes it holds
// and the bars on the right count what went in and what the record kept: sources passed twice are
// kept once (`refs.sources`). The corner square is validateGPUComputationRecord.

import {
  createGPUComputationRecord,
  describeGPUCommandGraph,
  getGPUComputationAdapter,
  GPUComputationRecorder,
  validateGPUComputationRecord,
  type GPUComputationFormulaRef,
  type GPUComputationSourceRef
} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun} from '../../define-run';
import {
  buildRelaxation,
  createDiagram,
  createRecordTracts,
  getDiagramLayers,
  ShapeBatch
} from '../../record-demo';
import {createChartAxes, getChartRectangle, mergeText, type RunLabel} from '../../text';
import {findBoxAt, getBoxGlyph, getCenter, type Box} from './_lib/record-probe';
import {parameters} from './computation-record.parameters';

/** What the last run drew, for the pointer: the node boxes and the lifetime rows. */
type Latest = {
  boxes: ReturnType<typeof createDiagram>['layout']['boxes'];
  order: string[];
  lifetimes: {box: Box; first: number; last: number}[];
  /** The lifetime area and the number of nodes across it, for the position marker. */
  area: {x0: number; x1: number; y0: number; y1: number};
  span: number;
};

const BAR_NAMES = ['nodes', 'edges', 'buffers', 'passed', 'kept', 'formulas'];
// Bars stop at this share of the plot, leaving room for the count above the tallest.
const BAR_HEADROOM = 1.2;
const BAR_RECTANGLE = [0.66, 0.06, 0.97, 0.46] as const;

export default defineOperationRun({
  parameters,
  caption: 'graphs[0] of the record, its resource lifetimes and what the record kept.',
  createInputs: ({random, world}) => ({
    tracts: createRecordTracts(random),
    world,
    latest: null as Latest | null,
    focus: {id: ''}
  }),
  build(builder, {tracts}, values) {
    const recorder = new GPUComputationRecorder({sourceText: true});
    buildRelaxation(builder, tracts, {
      iterations: values.iterations,
      recorder,
      deviation: true,
      scope: iteration => ({namespace: `iteration-${iteration}`, op: 'GPUSpatialLag'})
    });
    // The extra source and formula refs only change the record, which `afterRun` rebuilds, so the
    // graph is not rebuilt when they move.
    return {outputs: {recorder}, update: () => undefined};
  },
  rebuildOn: ['iterations'],
  afterRun({compiled, device, outputs, inputs, values, upload}) {
    const {width} = inputs.world;
    const described = describeGPUCommandGraph(compiled, {recorder: outputs.recorder});
    const extraSources: GPUComputationSourceRef[] = Array.from(
      {length: values.sources},
      (_, index) => ({
        id: `expression:extra-${index}`,
        language: 'expression',
        moduleId: `extra/${index}`,
        hash: `extra-${index}`
      })
    );
    const formulas: GPUComputationFormulaRef[] = Array.from(
      {length: values.formulas},
      (_, index) => ({
        id: `formula-${index}`,
        tex: 'x',
        ascii: 'x',
        symbols: [],
        derivation: 'curated'
      })
    );
    // The graph's own sources are passed again: the record keeps each id once.
    const passed = [...described.sources, ...extraSources];
    const record = createGPUComputationRecord({
      id: 'relaxation',
      adapter: getGPUComputationAdapter(device, 'portable'),
      domains: ['gpu-core'],
      graphs: [described],
      sources: passed,
      formulas,
      datasetVersion: 'tracts-v1',
      seed: 1
    });
    const report = validateGPUComputationRecord(record);
    const isValid = report.valid;
    const graph = record.graphs[0];

    const sourceIds = record.refs.sources.map(source => source.id);
    const diagram = createDiagram(
      upload,
      'record-graph',
      graph,
      {x0: 0.05 * width, y0: 0.52, x1: 0.95 * width, y1: 0.96},
      node => (node.source ? 1 + (sourceIds.indexOf(node.source) % 7) : 0)
    );

    // Resource lifetimes: one row per resource, from its first to its last node.
    const order = new Map(graph.nodes.map((node, index) => [node.id, index]));
    const lifetimes = new ShapeBatch();
    const resources = [...(graph.resources ?? [])].sort(
      (first, second) =>
        (order.get(first.firstNode ?? '') ?? 0) - (order.get(second.firstNode ?? '') ?? 0)
    );
    const area = {x0: 0.05 * width, x1: 0.6 * width, y0: 0.06, y1: 0.43};
    const rowHeight = (area.y1 - area.y0) / Math.max(resources.length, 1);
    const lifetimeRows: Latest['lifetimes'] = [];
    resources.forEach((resource, row) => {
      const first = order.get(resource.firstNode ?? '') ?? 0;
      const last = order.get(resource.lastNode ?? '') ?? first;
      const span = Math.max(graph.nodes.length, 1);
      const rectangle = {
        x0: area.x0 + ((area.x1 - area.x0) * first) / span,
        x1: area.x0 + ((area.x1 - area.x0) * (last + 1)) / span,
        y1: area.y1 - row * rowHeight,
        y0: area.y1 - (row + 0.78) * rowHeight
      };
      lifetimeRows.push({box: rectangle, first, last});
      lifetimes.addRectangle(
        rectangle,
        resource.ownership === 'transient' ? 2 : resource.ownership === 'imported' ? 1 : 0
      );
    });
    inputs.latest = {
      boxes: diagram.layout.boxes,
      order: graph.nodes.map(node => node.id),
      lifetimes: lifetimeRows,
      area,
      span: Math.max(graph.nodes.length, 1)
    };
    const marker = new ShapeBatch();
    marker.addRectangle(
      {x0: 0.94 * width, y0: 0.965, x1: 0.97 * width, y1: 0.995},
      isValid ? 1 : 2
    );
    // Bar heights are log2(1 + count): a node count of 40 does not hide a formula count of 2.
    const countValues = [
      graph.nodes.length,
      graph.edges.length,
      graph.resources?.length ?? 0,
      passed.length,
      record.refs.sources.length,
      record.refs.formulas.length
    ];
    const counts = Float32Array.from(countValues, count => Math.log2(1 + count));
    // The sources passed (index 3) and the sources kept (index 4) stand out from the other counts.
    const dedup = Float32Array.from(counts, (count, index) =>
      index === 3 || index === 4 ? count : 0
    );
    return {
      dedup: upload('dedup', 'float32', dedup),
      diagram,
      lifetimes: lifetimes.uploadShapes(upload, 'lifetimes'),
      marker: marker.uploadShapes(upload, 'valid'),
      counts: upload('counts', 'float32', counts),
      maximum: Math.max(...counts, 1),
      countValues,
      isValid
    };
  },
  // The pointer is a node of the recorded graph: it is outlined, and the resources alive at that
  // node are outlined in the lifetime rows below.
  interaction: {
    kind: 'probe',
    hint: 'hover a node',
    rest: ({inputs}) => {
      const box = inputs.latest?.boxes[0];
      if (!box) return [0.2, 0.3];
      const [x, y] = getCenter(box);
      return [x / inputs.world.width, 1 - y];
    },
    resolve(pointer, {inputs}) {
      const latest = inputs.latest;
      if (!latest || latest.boxes.length === 0) return {readout: 'hover a node'};
      const under = findBoxAt(latest.boxes, pointer.x, pointer.y);
      if (under) inputs.focus.id = under.node.id;
      const box =
        latest.boxes.find(candidate => candidate.node.id === inputs.focus.id) ?? latest.boxes[0];
      const position = latest.order.indexOf(box.node.id);
      const live = latest.lifetimes.filter(row => row.first <= position && position <= row.last);
      // A vertical rule through the lifetime rows at the hovered node's place in the order.
      const ruleX =
        latest.area.x0 + ((latest.area.x1 - latest.area.x0) * (position + 0.5)) / latest.span;
      return {
        glyphs: [
          getBoxGlyph(box),
          {shape: 'line', from: [ruleX, latest.area.y0], to: [ruleX, latest.area.y1]},
          ...live.map(row => getBoxGlyph(row.box, true))
        ],
        readout: `node ${position + 1} of ${latest.order.length}: ${box.node.label}${box.node.op ? ` (${box.node.op})` : ''} · ${live.length} resources live`
      };
    }
  },
  draw: ({extra, theme, world}) => {
    const rectangle = getChartRectangle(world, BAR_RECTANGLE);
    const axes = createChartAxes({
      rectangle,
      x: {categories: BAR_NAMES}
    });
    // The real count above each bar (the bars themselves are log2(1 + count)).
    const countLabels = {
      labels: extra.countValues.map((count, index) => ({
        text: String(count),
        at: [
          rectangle[0] + ((index + 0.5) / extra.countValues.length) * (rectangle[2] - rectangle[0]),
          rectangle[1] +
            (Math.log2(1 + count) / (extra.maximum * BAR_HEADROOM)) * (rectangle[3] - rectangle[1])
        ] as const,
        anchor: 'bottom' as const,
        offset: [0, 2] as const,
        role: 'value' as const
      }))
    };
    const labels: RunLabel[] = [
      {
        text: 'graphs[0]: one box per node, colour = source',
        at: [0.05 * world.width, 0.975],
        anchor: 'left',
        role: 'title'
      },
      {
        text: 'resource lifetimes: one row each, first to last node',
        at: [0.05 * world.width, 0.475],
        anchor: 'left',
        role: 'title'
      },
      {
        text: 'record counts (orange: sources passed, kept)',
        at: [BAR_RECTANGLE[0] * world.width, 0.475],
        anchor: 'left',
        role: 'title'
      },
      {
        text: extra.isValid ? 'record valid' : 'record invalid',
        at: [0.935 * world.width, 0.98],
        anchor: 'right',
        role: 'label'
      }
    ];
    return [
      ...getDiagramLayers(extra.diagram, {categories: [theme.muted, ...theme.categories]}),
      {
        layer: 'polygons',
        positions: extra.lifetimes.positions,
        indices: extra.lifetimes.indices,
        owners: extra.lifetimes.owners,
        values: extra.lifetimes.categories,
        triangleCount: extra.lifetimes.triangleCount,
        categories: [theme.muted, theme.categories[3], theme.categories[1]]
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: extra.counts,
        range: [0, extra.maximum * BAR_HEADROOM],
        rectangle: BAR_RECTANGLE,
        color: [...theme.categories[6], 1]
      },
      {
        layer: 'chart',
        kind: 'bars',
        values: extra.dedup,
        range: [0, extra.maximum * BAR_HEADROOM],
        rectangle: BAR_RECTANGLE,
        color: [...theme.categories[1], 1]
      },
      {
        layer: 'polygons',
        positions: extra.marker.positions,
        indices: extra.marker.indices,
        owners: extra.marker.owners,
        values: extra.marker.categories,
        triangleCount: extra.marker.triangleCount,
        categories: [theme.muted, theme.categories[4], theme.categories[2]]
      },
      {layer: 'text', ...mergeText({labels}, axes, countLabels)}
    ];
  }
});
