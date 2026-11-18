// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync} from 'node:fs';
import {basename, resolve} from 'node:path';
import typescript from 'typescript';
import {expect, test, vi} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {Buffer} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/index';
import {getLensScenario} from '../../src/computation-lenses/registry-implementation';
import {getSpatialOperatorEntry} from '../../src/gpu-spatial-analysis/computation-pack/pack';
import {LensCommandGraph} from '../../src/computation-lenses/lens-view-snapshot';
import {runSpatialScenario} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import * as spatialRuns from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/spatial-scenario-utils';
import {runAnalysisProfileExample} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/analysis-profile-run';
import * as s3Runs from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-run';

const DIRECTORY = resolve('modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis');

test('every registered S1/S3 scenario supplies a value for every declared choice template', () => {
  // Inspect the actual choices expressions without loading datasets, compiling graphs or submitting work.
  const files = new Set<string>();
  for (const filename of readdirSync(DIRECTORY)) {
    const path = resolve(DIRECTORY, filename);
    const source = readFileSync(path, 'utf8');
    if (/^s1-.*-index\.ts$/.test(filename) || filename === 's3-index.ts') {
      const syntax = typescript.createSourceFile(
        path,
        source,
        typescript.ScriptTarget.Latest,
        true
      );
      for (const statement of syntax.statements)
        if (
          typescript.isImportDeclaration(statement) &&
          typescript.isStringLiteral(statement.moduleSpecifier) &&
          statement.moduleSpecifier.text.endsWith('.lens')
        )
          files.add(resolve(DIRECTORY, `${statement.moduleSpecifier.text}.ts`));
    }
    // Also cover every direct scenario caller of the shared recording/regression paths.
    if (
      filename.endsWith('.lens.ts') &&
      /\b(?:registerRegressionLens|runRegressionScenario|runSpatialScenario|runTractWeightsScenario|runTractStatisticsScenario|runS3Lens)\s*\(/.test(
        source
      )
    ) {
      // This shared module registers several kinds, checked through their real run closures below.
      if (filename === 'trajectory-coverage.lens.ts') continue;
      files.add(path);
    }
  }
  const parsed = typescript.readConfigFile(resolve('tsconfig.json'), typescript.sys.readFile);
  const configuration = typescript.parseJsonConfigFileContent(
    parsed.config,
    typescript.sys,
    process.cwd()
  );
  const program = typescript.createProgram([...files], {...configuration.options, noEmit: true});
  const checker = program.getTypeChecker();
  const problems: string[] = [];
  let checked = 0;
  for (const file of [...files].sort()) {
    const id = `gpu-spatial-analysis.${basename(file, '.lens.ts')}`;
    const scenario = getLensScenario(id);
    expect(scenario, `${id}: imported S1/S3 registration must exist`).toBeDefined();
    const entry = getSpatialOperatorEntry(scenario!.element);
    expect(entry, `${id}: operator entry must exist`).toBeDefined();
    const maps: typescript.Type[] = [];
    const visit = (node: typescript.Node): void => {
      if (
        typescript.isPropertyAssignment(node) &&
        node.name.getText().replace(/^['"]|['"]$/g, '') === 'choices'
      ) {
        maps.push(checker.getTypeAtLocation(node.initializer));
        if (
          id === 'gpu-spatial-analysis.spatial-two-stage-least-squares' &&
          typescript.isObjectLiteralExpression(node.initializer)
        ) {
          const method = node.initializer.properties.find(
            (property): property is typescript.PropertyAssignment =>
              typescript.isPropertyAssignment(property) &&
              property.name.getText().replace(/^['"]|['"]$/g, '') === 'robust.method'
          );
          expect(
            method && typescript.isStringLiteral(method.initializer)
              ? method.initializer.text
              : undefined
          ).toBe('none');
        }
      }
      typescript.forEachChild(node, visit);
    };
    visit(program.getSourceFile(file)!);
    for (const template of entry!.choices) {
      if (!maps.length) problems.push(`${id}: missing choices.${template.name}`);
      for (const map of maps) {
        const property = checker.getPropertyOfType(map, template.name);
        const value =
          property &&
          checker.getTypeOfSymbolAtLocation(
            property,
            property.valueDeclaration ?? program.getSourceFile(file)!
          );
        const variants = value?.isUnion() ? value.types : value ? [value] : [];
        if (
          !property ||
          variants.some(
            type => type.flags & (typescript.TypeFlags.Undefined | typescript.TypeFlags.Null)
          )
        )
          problems.push(`${id}: missing choices.${template.name}`);
      }
    }
    checked++;
  }
  expect(checked).toBe(files.size);
  expect(checked).toBeGreaterThan(100);
  expect(problems).toEqual([]);
}, 20000);

test('missing spatial choice errors name the scenario and choice before recording begins', async () => {
  const device = createNullWebGPUDevice();
  const graph = new LensCommandGraph(device, {id: 'missing-choice'});
  const view = createTransientView(graph, 'input', 'float32', 1, Buffer.STORAGE | Buffer.COPY_SRC);
  const add = vi.spyOn(graph, 'add');
  const compile = vi.spyOn(graph, 'compileAsync');
  try {
    await expect(
      runSpatialScenario({
        id: 'gpu-spatial-analysis.spatial-two-stage-least-squares',
        graph,
        views: {input: view},
        encodeOptions: {parameters: undefined},
        axes: {},
        tags: {},
        datasetVersion: 'fixture',
        seed: 1,
        entry: getSpatialOperatorEntry('GPUSpatialTwoStageLeastSquares')!,
        choices: {weightsCriterion: 'queen', weightsTransform: 'row', instrumentOrder: 1}
      })
    ).rejects.toThrow(
      /gpu-spatial-analysis\.spatial-two-stage-least-squares.*choices\.robust\.method/
    );
    expect(compile).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    compile.mockRestore();
    add.mockRestore();
    device.destroy();
  }
});

test.each([
  'GPUAnalysisSpacing',
  'GPUAnalysisContiguityStatistics'
] as const)('%s profile supplies every declared choice before the recording boundary', async element => {
  const device = createNullWebGPUDevice();
  const boundary = new Error('Host-only recording boundary');
  const capture = vi.spyOn(spatialRuns, 'runSpatialScenario').mockImplementation(async props => {
    for (const template of props.entry.choices) {
      expect(
        Object.hasOwn(props.choices, template.name),
        `${props.id}: missing ${template.name}`
      ).toBe(true);
      expect(props.choices[template.name]).not.toBeUndefined();
    }
    throw boundary;
  });
  try {
    await expect(runAnalysisProfileExample(device, element)).rejects.toBe(boundary);
    expect(capture).toHaveBeenCalledOnce();
  } finally {
    capture.mockRestore();
    device.destroy();
  }
});

test('all aggregated S3 trajectory registrations supply their declared choices', async () => {
  const file = resolve(DIRECTORY, 'trajectory-coverage.lens.ts');
  const source = typescript.createSourceFile(
    file,
    readFileSync(file, 'utf8'),
    typescript.ScriptTarget.Latest,
    true
  );
  const declaration = source.statements.find(
    (statement): statement is typescript.TypeAliasDeclaration =>
      typescript.isTypeAliasDeclaration(statement) &&
      statement.name.text === 'TrajectoryCoverageKind'
  )!;
  if (!typescript.isUnionTypeNode(declaration.type))
    throw new Error('Expected trajectory kind alternatives');
  const device = createNullWebGPUDevice();
  const capture = vi.spyOn(s3Runs, 'runS3Lens').mockImplementation(async (_device, props) => {
    const entry = getSpatialOperatorEntry(props.element)!;
    for (const template of entry.choices) {
      expect(
        Object.hasOwn(props.choices ?? {}, template.name),
        `${props.scenarioId}: missing ${template.name}`
      ).toBe(true);
      expect(props.choices?.[template.name]).not.toBeUndefined();
    }
    return {} as never;
  });
  try {
    for (const kind of declaration.type.types) {
      if (!typescript.isLiteralTypeNode(kind) || !typescript.isStringLiteral(kind.literal))
        throw new Error('Expected a literal trajectory kind');
      const scenario = getLensScenario(`gpu-spatial-analysis.${kind.literal.text}`)!;
      expect(scenario).toBeDefined();
      await (await scenario.load()).run(device);
    }
    expect(capture).toHaveBeenCalledTimes(declaration.type.types.length);
  } finally {
    capture.mockRestore();
    device.destroy();
  }
});
