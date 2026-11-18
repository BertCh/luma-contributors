// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingKMeans: Lloyd k-means on the per-pixel vectors, with no labels given; each
// color is one cluster. Colors follow the clusters' mean NIR (low to high), so they stay put when
// `clusterCount` changes. Pointing at a pixel lightens every pixel of its cluster and reads its share
// from `counts`; few `maximumIterations` leave the seeding visible. Vectors are six
// standardized Sentinel-2 bands, not a learned embedding.

import {GPURasterEmbeddingKMeans} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createEmbeddingRaster, DIMENSIONS, getPixelAt, getPixelBox} from './_lib/embedding';
import {parameters} from './raster-embedding-kmeans.parameters';

const ITERATIONS = [1, 2, 4, 16];
const FOCUS_CATEGORIES = [
  [0, 0, 0, 0],
  [1, 1, 1, 0.3]
] as const;
/** Band whose centroid value orders the cluster colors (NIR). */
const ORDER_BAND = 3;

export default defineOperationRun({
  parameters,
  rebuildOn: ['clusterCount', 'maximumIterations'],
  caption:
    'Clusters found from the vectors alone, colored by mean NIR; point to lighten one. Six Sentinel-2 bands stand in for a learned embedding.',
  createInputs: ({aspect}) => ({
    ...createEmbeddingRaster(aspect),
    focus: {cluster: -1, build: -1},
    orderedBuild: -1
  }),
  build(builder, raster, values) {
    const clusterCount = Math.round(values.clusterCount);
    const vectors = builder.upload('vectors', 'float32', raster.vectors);
    const labels = builder.allocate('labels', 'uint32', raster.pixelCount);
    const counts = builder.allocate('counts', 'uint32', clusterCount);
    const status = builder.allocate('status', 'uint32', 3);
    const centroids = builder.allocate('centroids', 'float32', clusterCount * DIMENSIONS);
    builder.graph.add(
      new GPURasterEmbeddingKMeans({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        vectors: vectors.view,
        clusterCount,
        maximumIterations: ITERATIONS[Math.round(values.maximumIterations)],
        output: {
          labels: labels.view,
          centroids: centroids.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    const mask = builder.allocate('focus-mask', 'uint32', raster.pixelCount);
    const ordered = builder.allocate('ordered-labels', 'uint32', raster.pixelCount);
    return {outputs: {labels, counts, status, centroids, mask, ordered}};
  },
  interaction: {
    kind: 'probe',
    hint: 'point at a cluster',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, world, peek, build}) {
      const {index} = getPixelAt(inputs, world, pointer.x, pointer.y);
      const labels = peek(outputs.labels);
      const counts = peek(outputs.counts);
      const status = peek(outputs.status);
      const centroids = peek(outputs.centroids);
      if (labels && centroids && build !== inputs.orderedBuild) {
        inputs.orderedBuild = build;
        const clusterCount = centroids.length / DIMENSIONS;
        const byBand = Array.from({length: clusterCount}, (_, cluster) => cluster).sort(
          (left, right) =>
            centroids[left * DIMENSIONS + ORDER_BAND] - centroids[right * DIMENSIONS + ORDER_BAND]
        );
        const rank = new Uint32Array(clusterCount);
        byBand.forEach((cluster, position) => {
          rank[cluster] = position;
        });
        outputs.ordered.write(
          Uint32Array.from(labels, label => (label < clusterCount ? rank[label] : label))
        );
      }
      const cluster = labels ? labels[index] : -1;
      if (labels && (cluster !== inputs.focus.cluster || build !== inputs.focus.build)) {
        inputs.focus = {cluster, build};
        outputs.mask.write(Uint32Array.from(labels, label => (label === cluster ? 1 : 0)));
      }
      const size = counts && cluster >= 0 ? counts[cluster] : 0;
      const clusterTotal = counts ? counts.length : 0;
      return {
        spotlight: null,
        focus: cluster,
        glyphs: [getPixelBox(inputs, world, index)],
        readout: `${((100 * size) / inputs.pixelCount).toFixed(0)}% of pixels · cluster of ${clusterTotal} · ${status && status[2] ? 'converged' : 'stopped'} in ${status ? status[0] : '-'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'raster', values: outputs.ordered, columns: inputs.columns, rows: inputs.rows},
    {
      layer: 'raster',
      values: outputs.mask,
      columns: inputs.columns,
      rows: inputs.rows,
      categories: FOCUS_CATEGORIES
    }
  ]
});
