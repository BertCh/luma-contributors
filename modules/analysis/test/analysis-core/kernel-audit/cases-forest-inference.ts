// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUForestInference,
  type GPUForestInferenceModelShape
} from '../../../src/dataframe-analysis/forest-inference/index';
import type {KernelAuditCase} from './kernel-audit-context';

/** Every inference variant at shape and storage limits, including the eight-binding variant. */
export const FOREST_INFERENCE_AUDIT_CASES: readonly KernelAuditCase[] = [
  ...(['identity', 'sigmoid', 'softmax'] as const).map(
    (transform): KernelAuditCase => ({
      exportName: 'GPUForestInference',
      name: `${transform}, maximum forest shape, every optional view`,
      build: ({graph, view}) => {
        const outputCount = transform === 'sigmoid' ? 2 : 32;
        const model: GPUForestInferenceModelShape = {
          featureCount: 256,
          treeCount: 1024,
          nodeCount: 1048576,
          outputCount,
          maximumDepth: 64,
          task: 'classification',
          transform
        };
        const rowCount = 513;
        graph.add(
          new GPUForestInference({
            model,
            rowCount,
            features: view('float32', rowCount * model.featureCount),
            mask: view('uint32', rowCount),
            nodes: view('uint32', model.nodeCount * 6),
            trees: view('uint32', model.treeCount * 4),
            leafValues: view('float32', model.nodeCount * outputCount),
            parameters: view('float32', outputCount + 1),
            output: {
              predictions: view('float32', rowCount),
              validity: view('uint32', rowCount),
              scores: view('float32', rowCount * outputCount),
              leafIndices: view('uint32', rowCount * model.treeCount)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUForestInference',
    name: 'binding-limited regression rows, feature-major, minimal outputs',
    build: ({graph, view, maximumRows}) => {
      const rowCount = maximumRows(12);
      graph.add(
        new GPUForestInference({
          model: {
            featureCount: 1,
            treeCount: 1,
            nodeCount: 1,
            outputCount: 1,
            maximumDepth: 0,
            task: 'regression',
            transform: 'identity'
          },
          rowCount,
          featureLayout: 'feature-major',
          features: view('float32', rowCount),
          nodes: view('uint32', 6),
          trees: view('uint32', 4),
          leafValues: view('float32', 1),
          parameters: view('float32', 2),
          output: {predictions: view('float32', rowCount), validity: view('uint32', rowCount)}
        })
      );
    }
  },
  {
    exportName: 'GPUForestInference',
    name: '16 buffer bands and validity, every output',
    build: ({graph, view}) => {
      const rowCount = 300;
      graph.add(
        new GPUForestInference({
          model: {
            featureCount: 16,
            treeCount: 2,
            nodeCount: 6,
            outputCount: 1,
            maximumDepth: 1,
            task: 'regression',
            transform: 'identity'
          },
          rowCount,
          nodes: view('uint32', 36),
          trees: view('uint32', 8),
          leafValues: view('float32', 6),
          parameters: view('float32', 2),
          mask: view('uint32', rowCount),
          rasterStack: {
            bands: Array.from({length: 16}, (_, index) => ({
              id: `band-${index}`,
              format: 'float32',
              storage: {kind: 'buffer', values: view('float32', rowCount)},
              validity: view('uint32', rowCount)
            })),
            grid: {width: 20, height: 15, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'}
          },
          output: {
            predictions: view('float32', rowCount),
            validity: view('uint32', rowCount),
            scores: view('float32', rowCount),
            leafIndices: view('uint32', rowCount * 2)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUForestInference',
    name: 'one row and one leaf',
    build: ({graph, view}) =>
      graph.add(
        new GPUForestInference({
          model: {
            featureCount: 1,
            treeCount: 1,
            nodeCount: 1,
            outputCount: 1,
            maximumDepth: 0,
            task: 'regression',
            transform: 'identity'
          },
          rowCount: 1,
          features: view('float32', 1),
          nodes: view('uint32', 6),
          trees: view('uint32', 4),
          leafValues: view('float32', 1),
          parameters: view('float32', 2),
          output: {predictions: view('float32', 1), validity: view('uint32', 1)}
        })
      )
  }
];

/** Empty batches are valid no-ops and are checked separately from the nonempty kernel sweep. */
export const FOREST_INFERENCE_EMPTY_AUDIT_CASE: KernelAuditCase = {
  exportName: 'GPUForestInference',
  name: 'empty rows and one leaf',
  build: ({graph, view}) =>
    graph.add(
      new GPUForestInference({
        model: {
          featureCount: 1,
          treeCount: 1,
          nodeCount: 1,
          outputCount: 1,
          maximumDepth: 0,
          task: 'regression',
          transform: 'identity'
        },
        rowCount: 0,
        features: view('float32', 0),
        nodes: view('uint32', 6),
        trees: view('uint32', 4),
        leafValues: view('float32', 1),
        parameters: view('float32', 2),
        output: {predictions: view('float32', 0), validity: view('uint32', 0)}
      })
    )
};
