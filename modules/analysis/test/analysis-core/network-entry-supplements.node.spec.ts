// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createSourceFile, isExportDeclaration, isNamedExports, ScriptTarget} from 'typescript';
import {expect, expectTypeOf, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkCostMatrixOperation,
  MAXIMUM_PATH_OUTPUT_ROW_COUNT,
  decodeGPUNetworkPathExtractionOutput,
  type GPUNetworkCostMatrixOperationProps,
  type NetworkCostMatrixOperationStatus,
  type NetworkCostMatrixOperationProps,
  type NetworkPathExtractionCompactOutput
} from '@luma-contributors/analysis/gpu-network';
import {MAXIMUM_PATH_OUTPUT_ROW_COUNT as producerPathRowCount} from '../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('CMP-06 exposes directed noded matrix props and shared status beside legacy props', () => {
  const source = createSourceFile(
    'index.ts',
    readFileSync(
      new URL('../../src/gpu-network/network-accessibility/index.ts', import.meta.url),
      'utf8'
    ),
    ScriptTarget.Latest
  );
  const names = source.statements.flatMap(statement =>
    isExportDeclaration(statement) &&
    statement.exportClause &&
    isNamedExports(statement.exportClause)
      ? statement.exportClause.elements.map(element => element.name.text)
      : []
  );
  expect(names).toEqual(
    expect.arrayContaining([
      'GPUNetworkCostMatrixOperationProps',
      'NetworkCostMatrixOperationStatus',
      'NetworkCostMatrixOperationProps'
    ])
  );
  const status: NetworkCostMatrixOperationStatus = {
    converged: 'converged',
    iterationLimitReached: 'iteration-limit',
    sourceComplete: 'source-complete'
  };
  const legacy: NetworkCostMatrixOperationProps = {
    nodeCount: 4,
    rowCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    startNodes: 'starts',
    output: {costs: 'costs', status}
  };
  const props: GPUNetworkCostMatrixOperationProps = {
    ...legacy,
    direction: 'target-to-start',
    nodedTopology: {
      nodeCount: 'live-node-count',
      pieceCount: 'piece-count',
      overflow: 'overflow',
      uncertainCount: 'uncertain-count',
      piecesOverflow: 'piece-overflow',
      candidateOverflow: 'candidate-overflow'
    }
  };
  const operation = new GPUNetworkCostMatrixOperation(props);
  expect(operation.inputs['noded-overflow']).toBe('overflow');
  expect(operation.inputs['noded-uncertainCount']).toBe('uncertain-count');
  expect(operation.inputs['noded-nodeCount']).toBe('live-node-count');
  expect(operation.inputs['noded-pieceCount']).toBe('piece-count');
  expect(operation.inputs['noded-piecesOverflow']).toBe('piece-overflow');
  expect(operation.inputs['noded-candidateOverflow']).toBe('candidate-overflow');
  expect(operation.outputs['sourceComplete']).toEqual({
    value: 'source-complete',
    format: 'uint32',
    length: 1
  });
  expect(new GPUNetworkCostMatrixOperation(legacy)).toBeInstanceOf(GPUNetworkCostMatrixOperation);
});

it('OUT-10 exposes the uint32 path output bound beside the delivered reader', () => {
  expect(MAXIMUM_PATH_OUTPUT_ROW_COUNT).toBe(4294967295);
  expect(MAXIMUM_PATH_OUTPUT_ROW_COUNT).toBe(producerPathRowCount);
  expect(typeof decodeGPUNetworkPathExtractionOutput).toBe('function');
});

it('B1 exports the shared compact path output including unrepresentable status', () => {
  const source = createSourceFile(
    'index.ts',
    readFileSync(
      new URL('../../src/gpu-network/network-analysis/index.ts', import.meta.url),
      'utf8'
    ),
    ScriptTarget.Latest
  );
  const names = source.statements.flatMap(statement =>
    isExportDeclaration(statement) &&
    statement.exportClause &&
    isNamedExports(statement.exportClause)
      ? statement.exportClause.elements.map(element => element.name.text)
      : []
  );
  expect(names).toContain('NetworkPathExtractionCompactOutput');
  expectTypeOf<NetworkPathExtractionCompactOutput['status']>().toHaveProperty('unrepresentable');
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output: NetworkPathExtractionCompactOutput = {
      ids: createTransientView(graph, 'path-ids', 'uint32', 2),
      status: {
        count: createTransientView(graph, 'path-count', 'uint32', 1),
        overflow: createTransientView(graph, 'path-overflow', 'uint32', 1),
        unrepresentable: createTransientView(graph, 'path-unrepresentable', 'uint32', 1)
      }
    };
    expect(output.status.unrepresentable?.length).toBe(1);
  } finally {
    device.destroy();
  }
});
