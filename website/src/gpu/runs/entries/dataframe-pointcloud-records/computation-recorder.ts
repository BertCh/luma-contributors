// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUComputationRecorder watching a real analysis being built. A spatial-lag relaxation is added
// to a command graph through the recorder. The top graph is what the compiled graph publishes by
// itself; the bottom one is described with the recorder, whose withScope namespaces color the
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
import {parameters} from './computation-recorder.parameters';

const SOURCE_BAR_CHARACTERS = 600;

export default defineOperationRun({
  parameters,
  caption: 'The same compiled graph: published facts only (top), recorded (bottom).',
  createInputs: ({random, world}) => ({tracts: createRecordTracts(random), world}),
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
    return {
      publishedDiagram,
      recordedDiagram,
      bars: bars.uploadShapes(upload, 'source-bars')
    };
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
