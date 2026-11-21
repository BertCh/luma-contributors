// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUComputationRecorder watching a real analysis being built. A spatial-lag relaxation is added
// to a command graph through the recorder. The top graph is what the compiled graph publishes by
// itself; the bottom one is described with the recorder, whose withSynchronousScope namespaces color the
// nodes. The strip below shows the kernel sources the recorder kept: full text only with
// `sourceText`, otherwise just a hash.

import {
  describeGPUCommandGraph,
  GPUComputationRecorder
} from '@luma-contributors/analysis/computation-record';
import {defineOperationRun} from '../../define-run';
import {
  buildRelaxation,
  createDiagram,
  createNamespaceCategories,
  createRecordTracts,
  getDiagramLayers,
  ShapeBatch
} from '../../record-demo';
import {findBoxAt, getBoxGlyph, getCenter} from './_lib/record-probe';
import {parameters} from './computation-recorder.parameters';

const SOURCE_BAR_CHARACTERS = 600;

type Latest = {
  published: ReturnType<typeof createDiagram>['layout']['boxes'];
  recorded: ReturnType<typeof createDiagram>['layout']['boxes'];
  /** Characters of retained text per source id (undefined: only a hash was kept). */
  textLength: Map<string, number | undefined>;
};

export default defineOperationRun({
  parameters,
  caption: 'The same compiled graph: published facts only (top), recorded (bottom).',
  createInputs: ({random, world}) => ({
    tracts: createRecordTracts(random),
    world,
    latest: null as Latest | null,
    focus: {id: ''}
  }),
  build(builder, {tracts}, values) {
    const recorder = new GPUComputationRecorder({sourceText: values.sourceText === 1});
    buildRelaxation(builder, tracts, {
      iterations: values.iterations,
      recorder,
      deviation: true,
      scope: iteration =>
        values.scopes === 0
          ? undefined
          : {
              namespace: values.scopes === 1 ? 'relaxation' : `iteration-${iteration}`,
              op: 'GPUSpatialLag'
            }
    });
    return {outputs: {recorder}};
  },
  afterRun({compiled, outputs, inputs, upload}) {
    const {width} = inputs.world;
    const published = describeGPUCommandGraph(compiled).graph;
    const recorded = describeGPUCommandGraph(compiled, {recorder: outputs.recorder});
    const publishedDiagram = createDiagram(
      upload,
      'published',
      published,
      {x0: 0.05 * width, y0: 0.58, x1: 0.95 * width, y1: 0.96},
      () => 0
    );
    const recordedDiagram = createDiagram(
      upload,
      'recorded',
      recorded.graph,
      {x0: 0.05 * width, y0: 0.17, x1: 0.95 * width, y1: 0.55},
      createNamespaceCategories(recorded.graph)
    );
    // One bar per distinct source: its length is the retained text; without text only a hash tick.
    const bars = new ShapeBatch();
    const barHeight = 0.05;
    recorded.sources.forEach((source, index) => {
      const y0 = 0.02 + index * (barHeight + 0.012);
      const length = source.text === undefined ? 12 : Math.min(source.text.length, 1200);
      bars.addRectangle(
        {
          x0: 0.05 * width,
          y0,
          x1: 0.05 * width + (length / SOURCE_BAR_CHARACTERS) * 0.8 * width,
          y1: y0 + barHeight
        },
        source.text === undefined ? 0 : 1
      );
    });
    inputs.latest = {
      published: publishedDiagram.layout.boxes,
      recorded: recordedDiagram.layout.boxes,
      textLength: new Map(recorded.sources.map(source => [source.id, source.text?.length]))
    };
    return {
      publishedDiagram,
      recordedDiagram,
      bars: bars.uploadShapes(upload, 'source-bars')
    };
  },
  // The pointer is a node of either graph: it is outlined with its twin in the other graph, and
  // the nodes of the same recorded scope are outlined too.
  interaction: {
    kind: 'probe',
    hint: 'hover a node',
    rest: ({inputs}) => {
      const box = inputs.latest?.recorded[0];
      if (!box) return [0.2, 0.7];
      const [x, y] = getCenter(box);
      return [x / inputs.world.width, 1 - y];
    },
    resolve(pointer, {inputs}) {
      const latest = inputs.latest;
      if (!latest || latest.recorded.length === 0) return {readout: 'hover a node'};
      const under = findBoxAt([...latest.recorded, ...latest.published], pointer.x, pointer.y);
      if (under) inputs.focus.id = under.node.id;
      const recorded =
        latest.recorded.find(box => box.node.id === inputs.focus.id) ?? latest.recorded[0];
      const published = latest.published.find(box => box.node.id === recorded.node.id);
      const siblings = latest.recorded.filter(
        box =>
          box !== recorded &&
          recorded.node.namespace &&
          box.node.namespace === recorded.node.namespace
      );
      const length = recorded.node.source ? latest.textLength.get(recorded.node.source) : undefined;
      return {
        glyphs: [
          getBoxGlyph(recorded),
          ...siblings.map(box => getBoxGlyph(box, true)),
          ...(published
            ? [
                getBoxGlyph(published),
                {
                  shape: 'line' as const,
                  from: getCenter(recorded),
                  to: getCenter(published),
                  dashed: true
                }
              ]
            : [])
        ],
        readout: `${recorded.node.id} · scope ${recorded.node.namespace ?? 'none'} · ${
          length === undefined ? 'hash only' : `${(length / 1000).toFixed(1)} kB text`
        }`
      };
    }
  },
  draw: ({extra, theme}) => [
    ...getDiagramLayers(extra.publishedDiagram, {context: true}),
    ...getDiagramLayers(extra.recordedDiagram, {
      categories: [theme.muted, ...theme.categories]
    }),
    {
      layer: 'polygons',
      positions: extra.bars.positions,
      indices: extra.bars.indices,
      owners: extra.bars.owners,
      values: extra.bars.values,
      triangleCount: extra.bars.triangleCount,
      range: [0, 1],
      ramp: 'sequential',
      color: [...theme.categories[3], 0.9]
    }
  ]
});
