// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolve} from 'node:path';
import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import typescript from 'typescript';
import {entries} from '../../../../website/src/catalog/entries/dataframe-pointcloud-records';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {
  GPUPointCloudSMRF,
  GPUPointCloudVoxelMerge,
  GPUPointCloudGround,
  GPUPointCloudHeight
} from '@luma-contributors/analysis/pointcloud-analysis';
import {constructPointCloudShowcaseSnippet} from '../../../analysis/test/gpu-pointcloud/pointcloud-showcase-snippets';

test('final2 catalog-docs:G3 displayed SMRF and voxel merge use separate point and voxel statuses', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const unsigned = (id: string, count: number) => createTransientView(graph, id, 'uint32', count);
  const vector = (id: string, count: number) => createTransientView(graph, id, 'float32x3', count);
  const smrfStatus = unsigned('smrf-status', 4);
  const mergedStatus = unsigned('merged-status', 3);
  const bindings = {
    GPUPointCloudSMRF,
    GPUPointCloudVoxelMerge,
    positions: vector('positions', 4),
    classifications: unsigned('classes', 4),
    mask: unsigned('mask', 4),
    grid: {origin: [0, 0], cellSize: 1, width: 3, height: 2},
    groundSurface: createTransientView(graph, 'surface', 'float32', 6),
    groundMask: unsigned('ground-mask', 4),
    outputClassifications: unsigned('ground-classes', 4),
    countsA: unsigned('counts-a', 3),
    countsB: unsigned('counts-b', 3),
    representativesA: unsigned('representatives-a', 3),
    representativesB: unsigned('representatives-b', 3),
    meansA: vector('means-a', 3),
    meansB: vector('means-b', 3),
    sourceRowOffset: 2,
    mergedCounts: unsigned('merged-counts', 3),
    mergedRepresentatives: unsigned('merged-representatives', 3),
    mergedCentroids: vector('merged-centroids', 3),
    smrfStatus,
    mergedStatus,
    status: smrfStatus
  };
  try {
    const smrf = constructPointCloudShowcaseSnippet<GPUPointCloudSMRF>(
      'support',
      'GPUPointCloudSMRF',
      bindings
    );
    graph.add(smrf);
    const merged = constructPointCloudShowcaseSnippet<GPUPointCloudVoxelMerge>(
      'support',
      'GPUPointCloudVoxelMerge',
      bindings
    );
    graph.add(merged);
    expect(smrf.props.output.status).toBe(smrfStatus);
    expect(merged.props.output.status).toBe(mergedStatus);
    expect(() =>
      constructPointCloudShowcaseSnippet('support', 'GPUPointCloudVoxelMerge', {
        ...bindings,
        mergedStatus: smrfStatus
      })
    ).toThrow('status must have 3 rows');
  } finally {
    device.destroy();
  }
});

test('final2 catalog-docs:G4 displayed lidar ground and height require ground support', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const unsigned = (id: string) => createTransientView(graph, id, 'uint32', 2);
  const scalar = (id: string) => createTransientView(graph, id, 'float32', 2);
  const bindings = {
    GPUPointCloudGround,
    GPUPointCloudHeight,
    positions: createTransientView(graph, 'positions', 'float32x3', 2),
    classifications: unsigned('classes'),
    grid: {origin: [0, 0], cellSize: 1, width: 2, height: 1},
    minimumSurface: scalar('minimum'),
    coverage: unsigned('coverage'),
    groundSeedCoverage: unsigned('ground-seed-coverage'),
    terrain: scalar('terrain'),
    groundMask: unsigned('ground-mask'),
    outputClassifications: unsigned('ground-classes'),
    groundStatus: unsigned('ground-status'),
    heights: scalar('heights'),
    heightStatus: unsigned('height-status')
  };
  try {
    const ground = constructPointCloudShowcaseSnippet<GPUPointCloudGround>(
      'lidar',
      'GPUPointCloudGround',
      bindings
    );
    const height = constructPointCloudShowcaseSnippet<GPUPointCloudHeight>(
      'lidar',
      'GPUPointCloudHeight',
      bindings
    );
    expect(height.props.coverageBit).toBe(2);
    expect(ground.props.coverage).toBe(bindings.groundSeedCoverage);
  } finally {
    device.destroy();
  }
});

test('final1 catalog-docs:F9 showcase snippets retain nested local point-cloud outputs', () => {
  for (const scene of ['lidar-ground', 'point-cloud-support', 'point-cloud-change']) {
    const source = readFileSync(
      resolve(`examples/deck/spatial-analysis-showcase/src/scenes/terrain/${scene}.scene.ts`),
      'utf8'
    );
    const constructors = [
      ...source.matchAll(
        /new GPUPointCloud(?:Ground|Height|Features|SMRF|VoxelMerge|Change)\(\{[\s\S]*?\}\)\)/g
      )
    ];
    expect(constructors.length).toBeGreaterThan(0);
    for (const [construction] of constructors) expect(construction).toContain('output:');
  }
});

test('final1 catalog-docs:F9 authored point-cloud examples typecheck against the public entry point', () => {
  const ids = [
    'pointcloud-neighborhoods',
    'pointcloud-features',
    'pointcloud-voxel-merge',
    'pointcloud-ground',
    'pointcloud-statistical-filter',
    'pointcloud-surfaces',
    'pointcloud-height',
    'pointcloud-smrf',
    'pointcloud-icp',
    'pointcloud-change'
  ];
  const selected = entries.filter(entry => ids.includes(entry.id));
  expect(selected).toHaveLength(10);
  const prelude = `import type {Device} from '@luma.gl/core';
    import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
    import type {GPUPointCloudVoxelMergeProps} from '@luma-contributors/analysis/pointcloud-analysis';
    declare const device: Device, graph: GPUCommandGraph;
    declare const positions: GraphDataView<'float32x3'>;
    declare const normals: GraphDataView<'float32x3'>, eigenvalues: GraphDataView<'float32x3'>, shapes: GraphDataView<'float32x3'>, centroids: GraphDataView<'float32x3'>;
    declare const indices: GraphDataView<'uint32'>, counts: GraphDataView<'uint32'>, status: GraphDataView<'uint32'>, representatives: GraphDataView<'uint32'>;
    declare const classes: GraphDataView<'uint32'>, coverage: GraphDataView<'uint32'>, groundMask: GraphDataView<'uint32'>, outputClasses: GraphDataView<'uint32'>;
    declare const squaredDistances: GraphDataView<'float32'>, minimum: GraphDataView<'float32'>;
    declare const terrain: GraphDataView<'float32'>, surface: GraphDataView<'float32'>, canopy: GraphDataView<'float32'>, heights: GraphDataView<'float32'>, groundSurface: GraphDataView<'float32'>, meanDistances: GraphDataView<'float32'>, transform: GraphDataView<'float32'>;
    declare const validity: GraphDataView<'uint32'>, inlierMask: GraphDataView<'uint32'>;
    declare const source: GraphDataView<'float32x3'>, target: GraphDataView<'float32x3'>, reference: GraphDataView<'float32x3'>, comparison: GraphDataView<'float32x3'>, cores: GraphDataView<'float32x3'>, changes: GraphDataView<'float32x3'>, roughness: GraphDataView<'float32x3'>, history: GraphDataView<'float32x3'>;
    declare const tileA: Omit<GPUPointCloudVoxelMergeProps['parts'][number], 'sourceRowOffset'>, tileB: typeof tileA;`;
  const files = new Map(
    selected.map(entry => [
      resolve(import.meta.dirname, `pointcloud-catalog-${entry.id}.virtual.ts`),
      `${prelude}\n${entry.example}`
    ])
  );
  const parsed = typescript.readConfigFile(resolve('tsconfig.json'), typescript.sys.readFile);
  const configuration = typescript.parseJsonConfigFileContent(
    parsed.config,
    typescript.sys,
    process.cwd()
  );
  const options = {...configuration.options, noEmit: true};
  const host = typescript.createCompilerHost(options);
  const original = host.getSourceFile.bind(host);
  host.getSourceFile = (file, version, onError, shouldCreate) =>
    files.has(file)
      ? typescript.createSourceFile(file, files.get(file)!, version, true)
      : original(file, version, onError, shouldCreate);
  const program = typescript.createProgram([...files.keys()], options, host);
  const diagnostics = program
    .getSemanticDiagnostics()
    .filter(diagnostic => diagnostic.file && files.has(diagnostic.file.fileName));
  expect(
    diagnostics.map(
      diagnostic =>
        `${diagnostic.file!.fileName}: ${typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n')}`
    )
  ).toEqual([]);
  for (const entry of selected) {
    // The eight-row short list can omit output; the actual public call must keep it nested.
    const source = typescript.createSourceFile(
      'example.ts',
      entry.example,
      typescript.ScriptTarget.Latest,
      true
    );
    const constructions: typescript.NewExpression[] = [];
    const visit = (node: typescript.Node): void => {
      if (typescript.isNewExpression(node) && node.expression.getText(source) === entry.symbol)
        constructions.push(node);
      typescript.forEachChild(node, visit);
    };
    visit(source);
    expect(constructions, entry.id).toHaveLength(1);
    const props = constructions[0].arguments?.[0];
    if (!props || !typescript.isObjectLiteralExpression(props))
      throw new Error(`${entry.id}: expected inline constructor props`);
    const output = props.properties.find(property => property.name?.getText(source) === 'output');
    if (
      !output ||
      !typescript.isPropertyAssignment(output) ||
      !typescript.isObjectLiteralExpression(output.initializer)
    )
      throw new Error(`${entry.id}: expected nested output object`);
    expect(output.initializer.properties.length, entry.id).toBeGreaterThan(0);
    expect(entry.params.length, entry.id).toBeLessThanOrEqual(8);
    const writable = entry.outputs?.map(output => output.name) ?? [];
    expect(
      props.properties.filter(
        property => property.name && writable.includes(property.name.getText(source))
      ),
      entry.id
    ).toEqual([]);
    expect(entry.params?.filter(parameter => writable.includes(parameter.name))).toEqual([]);
    expect(entry.moreRequired?.filter(name => writable.includes(name)) ?? []).toEqual([]);
  }
});
