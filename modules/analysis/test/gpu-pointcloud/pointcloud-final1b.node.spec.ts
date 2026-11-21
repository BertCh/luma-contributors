// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudSurfaces,
  GPUPointCloudGround,
  GPUPointCloudSMRF,
  GPUPointCloudHeight,
  GPUPointCloudCoverageValidity,
  getRasterMetadataFromPointCloudGrid
} from '@luma-contributors/analysis/pointcloud-analysis';
import {constructPointCloudShowcaseSnippet} from './pointcloud-showcase-snippets';
import {makeGPURasterBand} from '../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {makeSpatialFixture} from '../spatial-test-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {addPointCloudSupportVoxelPart} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-voxel-parts';
import {
  createGroundSeedCoverageNode,
  createSupportedGroundClassesNode,
  createGroundDisplayClassesNode
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-ground-support';
import {
  createPointCloudDisplayCopyNode,
  getPointCloudHeightSupportText,
  getPointCloudICPSupportText,
  summarizePointCloudGround
} from '../../../../website/src/gpu/runs/entries/dataframe-pointcloud-records/_lib/pointcloud-display';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

const scenes = 'examples/deck/spatial-analysis-showcase/src/scenes/terrain/';
const entries = 'website/src/gpu/runs/entries/dataframe-pointcloud-records/';

// Execute only the actual headless fixtures' declarations on a null device. No compilation,
// encoding or readback: catch setup failures without duplicating their graph inputs here.
it.each([
  'F2 Surfaces',
  'F4 suppresses provisional',
  'F4 / final2 G4 displayed lidar height'
])('%s headless graph declares successfully', prefix => {
  const source = typescript.createSourceFile(
    'pointcloud-website.spec.ts',
    readFileSync('modules/analysis/test/gpu-pointcloud/pointcloud-website.spec.ts', 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  const declaration = source.statements.find(
    statement =>
      typescript.isExpressionStatement(statement) &&
      typescript.isCallExpression(statement.expression) &&
      statement.expression.expression.getText(source) === 'it' &&
      typescript.isStringLiteral(statement.expression.arguments[0]) &&
      statement.expression.arguments[0].text.startsWith(prefix)
  );
  if (
    !declaration ||
    !typescript.isExpressionStatement(declaration) ||
    !typescript.isCallExpression(declaration.expression)
  )
    throw new Error(`Missing fixture ${prefix}`);
  const callback = declaration.expression.arguments[1];
  if (!typescript.isArrowFunction(callback) || !typescript.isBlock(callback.body))
    throw new Error(`Missing fixture body ${prefix}`);
  const block = callback.body.statements.find(typescript.isTryStatement)!.tryBlock;
  const submission = block.statements.findIndex(
    statement =>
      typescript.isExpressionStatement(statement) &&
      typescript.isAwaitExpression(statement.expression)
  );
  expect(submission).toBeGreaterThan(0);
  const setup = block.statements
    .slice(0, submission)
    .map(statement => statement.getText(source))
    .join('\n');
  const javascript = typescript.transpileModule(setup, {
    compilerOptions: {target: typescript.ScriptTarget.ES2022}
  }).outputText;
  const fixture = makeSpatialFixture(createNullWebGPUDevice());
  try {
    expect(() =>
      runInNewContext(javascript, {
        fixture,
        Float32Array,
        Uint32Array,
        GPUPointCloudSurfaces,
        GPUPointCloudGround,
        GPUPointCloudSMRF,
        GPUPointCloudHeight,
        GPUPointCloudCoverageValidity,
        makeGPURasterBand,
        getRasterMetadataFromPointCloudGrid,
        constructPointCloudShowcaseSnippet,
        createGroundSeedCoverageNode,
        createSupportedGroundClassesNode,
        createGroundDisplayClassesNode,
        createPointCloudDisplayCopyNode
      })
    ).not.toThrow();
  } finally {
    fixture.destroy();
  }
});

it('F2 refuses occupancy and provisional classifications as terrain seeds', () => {
  const source = readFileSync(`${scenes}lidar-ground.compute.ts`, 'utf8');
  expect(source.includes('coverage: firstCoverage,')).toBe(false);
  expect(source.includes('classifications: classified,')).toBe(false);
});

it.each(['height', 'icp', 'ground', 'smrf'])('F4 %s retains its status for display', name => {
  const source = readFileSync(`${entries}pointcloud-${name}.ts`, 'utf8');
  expect(/status: builder.allocate\('status'/.test(source)).toBe(false);
  expect(/peek\(outputs.status\)/.test(source)).toBe(true);
});

it('F10 never declares selected positions as voxel means', () => {
  const source = readFileSync(`${scenes}point-cloud-support.compute.ts`, 'utf8');
  expect(source.includes('centroids: partOutput,')).toBe(false);
});

it('F2 declares seed-bit extraction and rejects every nonzero ground status', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const input = createTransientView(graph, 'input', 'uint32', 2);
  const status = createTransientView(graph, 'status', 'uint32', 2);
  const output = createTransientView(graph, 'output', 'uint32', 2);
  const sources = getGPUCommandNodeKernelSources([
    ...createGroundSeedCoverageNode(graph, input, output).getCommandNodes(graph),
    createSupportedGroundClassesNode(graph, input, status, output)
  ]);
  expect(sources).toHaveLength(2);
  expect(sources[0].source).toContain('& 2u) != 0u');
  expect(sources[0].source).toContain('writeTerrainValidity(index, ground)');
  expect(sources[1].source).toContain('classification == 2u && status[statusOffset + index] != 0u');
  for (const source of sources)
    expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
});

it('F10 actual showcase part builder uploads means, including a nonzero origin and empty tile', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const samples = new Float32Array([1.03125, 2.05, 3.05, 1.15625, 2.05, 3.05]);
  for (const sourcePositions of [samples, new Float32Array()]) {
    let uploaded: Float32Array | undefined;
    const positions = createTransientView(
      graph,
      `positions-${sourcePositions.length}`,
      'float32x3',
      sourcePositions.length / 3
    );
    const counts = createTransientView(graph, `counts-${sourcePositions.length}`, 'uint32', 1);
    const representatives = createTransientView(
      graph,
      `rows-${sourcePositions.length}`,
      'uint32',
      1
    );
    const selectedPositions = createTransientView(
      graph,
      `selected-${sourcePositions.length}`,
      'float32x3',
      1
    );
    const means = createTransientView(graph, `means-${sourcePositions.length}`, 'float32x3', 1);
    const part = addPointCloudSupportVoxelPart(graph, {
      id: `tile-${sourcePositions.length}`,
      positions,
      sourcePositions,
      origin: [1, 2, 3],
      voxelSize: 0.2,
      gridSize: [1, 1, 1],
      counts,
      representatives,
      selectedPositions,
      sourceRowOffset: 0,
      uploadMeans: values => {
        uploaded = values;
        return means;
      }
    });
    expect(part.centroids).toBe(means);
    expect(part.centroids).not.toBe(selectedPositions);
    expect(uploaded![0]).toBe(sourcePositions.length ? 1.09375 : 0);
  }
});

it('F4 distinguishes zero height from missing support and excludes provisional ground', () => {
  expect(getPointCloudHeightSupportText(0)).toBe('valid height');
  expect(getPointCloudHeightSupportText(2)).toBe('unsupported terrain');
  expect(getPointCloudICPSupportText(2)).toBe('insufficient matches');
  expect(getPointCloudICPSupportText(1)).toBe('iteration budget exhausted');
  expect(summarizePointCloudGround([1, 1, 1, 0], [0, 2, 8, 1])).toEqual({
    ground: 1,
    supported: 1,
    incomplete: 3
  });
});

it('F4 display kernels read integer masks numerically and broadcast whole-fit status', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const source = createTransientView(graph, 'mask', 'uint32', 2);
  const status = createTransientView(graph, 'fit-status', 'uint32', 1);
  const output = createTransientView(graph, 'display', 'float32', 2);
  const [shader] = getGPUCommandNodeKernelSources([
    createPointCloudDisplayCopyNode(graph, {
      id: 'broadcast-display',
      source,
      status,
      output,
      statusIndex: 0
    })
  ]);
  expect(shader.source).toContain('values: array<u32>');
  expect(shader.source).toContain('status[statusOffset + 0u] == 0u');
  expect(auditWGSLModule(shader.source)).toEqual([]);
});
