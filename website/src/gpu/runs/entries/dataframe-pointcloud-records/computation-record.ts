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
import {parameters} from './computation-record.parameters';

export default defineOperationRun({
  parameters,
  caption: 'graphs[0] of the record, its resource lifetimes and what the record kept.',
  createInputs: ({random, world}) => ({tracts: createRecordTracts(random), world}),
  build(builder, {tracts}, values) {
    const recorder = new GPUComputationRecorder({sourceText: true});
    buildRelaxation(builder, tracts, {
      iterations: values.iterations,
      recorder,
      deviation: true,
      scope: iteration => ({namespace: `iteration-${iteration}`, op: 'GPUSpatialLag'})
    });
    return {outputs: {recorder}};
  },
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
    const area = {x0: 0.05 * width, x1: 0.6 * width, y0: 0.06, y1: 0.46};
    const rowHeight = (area.y1 - area.y0) / Math.max(resources.length, 1);
    resources.forEach((resource, row) => {
      const first = order.get(resource.firstNode ?? '') ?? 0;
      const last = order.get(resource.lastNode ?? '') ?? first;
      const span = Math.max(graph.nodes.length, 1);
      lifetimes.addRectangle(
        {
          x0: area.x0 + ((area.x1 - area.x0) * first) / span,
          x1: area.x0 + ((area.x1 - area.x0) * (last + 1)) / span,
          y1: area.y1 - row * rowHeight,
          y0: area.y1 - (row + 0.78) * rowHeight
        },
        resource.ownership === 'transient' ? 2 : 1
      );
    });
    const marker = new ShapeBatch();
    marker.addRectangle(
      {x0: 0.94 * width, y0: 0.965, x1: 0.97 * width, y1: 0.995},
      isValid ? 1 : 2
    );
    // Bar heights are log2(1 + count): a node count of 40 does not hide a formula count of 2.
    const counts = Float32Array.from(
      [
        graph.nodes.length,
        graph.edges.length,
        graph.resources?.length ?? 0,
        passed.length,
        record.refs.sources.length,
        record.refs.formulas.length
      ],
      count => Math.log2(1 + count)
    );
    return {
      diagram,
      lifetimes: lifetimes.uploadShapes(upload, 'lifetimes'),
      marker: marker.uploadShapes(upload, 'valid'),
      counts: upload('counts', 'float32', counts),
      maximum: Math.max(...counts, 1)
    };
  },
  draw: ({extra, theme}) => [
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
      range: [0, extra.maximum],
      rectangle: [0.66, 0.06, 0.97, 0.46],
      color: [...theme.categories[6], 1]
    },
    {
      layer: 'polygons',
      positions: extra.marker.positions,
      indices: extra.marker.indices,
      owners: extra.marker.owners,
      values: extra.marker.categories,
      triangleCount: extra.marker.triangleCount,
      categories: [theme.muted, theme.categories[4], theme.categories[2]]
    }
  ]
});
