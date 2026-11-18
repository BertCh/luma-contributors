// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  readFileSync,
  readdirSync,
  mkdtempSync,
  mkdirSync,
  cpSync,
  existsSync,
  writeFileSync,
  rmSync
} from 'node:fs';
import {join, dirname, resolve} from 'node:path';
import {createRequire} from 'node:module';
import {tmpdir} from 'node:os';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {listLensUserInputRoles} from '../../src/computation-lenses/user-input';
import type {LensScenarioDescriptor} from '../../src/computation-lenses/types';
import {describe, expect, it} from 'vitest';
import {build, transform} from 'esbuild';
import typescript from 'typescript';
import {transformCommonJSModule} from '../../../../scripts/build-commonjs.mjs';
import {REPOSITORY_ROOT} from './lens-directives';

function collectBareJSONImports(source: string, filename: string): string[] {
  const parsed = typescript.createSourceFile(
    filename,
    source,
    typescript.ScriptTarget.Latest,
    true
  );
  const failures: string[] = [];
  function hasJSONAttributes(options: typescript.Expression | undefined): boolean {
    if (!options || !typescript.isObjectLiteralExpression(options)) return false;
    const attributes = options.properties.find(
      property =>
        typescript.isPropertyAssignment(property) && property.name.getText(parsed) === 'with'
    );
    if (
      !attributes ||
      !typescript.isPropertyAssignment(attributes) ||
      !typescript.isObjectLiteralExpression(attributes.initializer)
    )
      return false;
    return attributes.initializer.properties.some(
      property =>
        typescript.isPropertyAssignment(property) &&
        property.name.getText(parsed) === 'type' &&
        typescript.isStringLiteral(property.initializer) &&
        property.initializer.text === 'json'
    );
  }
  function visit(node: typescript.Node): void {
    if (
      (typescript.isImportDeclaration(node) || typescript.isExportDeclaration(node)) &&
      node.moduleSpecifier &&
      typescript.isStringLiteral(node.moduleSpecifier) &&
      node.moduleSpecifier.text.endsWith('.json')
    ) {
      const typeOnly = typescript.isImportDeclaration(node)
        ? node.importClause?.isTypeOnly
        : node.isTypeOnly;
      const attributed =
        node.attributes?.token === typescript.SyntaxKind.WithKeyword &&
        node.attributes.elements.some(
          element =>
            element.name.text === 'type' &&
            typescript.isStringLiteral(element.value) &&
            element.value.text === 'json'
        );
      if (!typeOnly && !attributed) failures.push(node.moduleSpecifier.text);
    }
    if (
      typescript.isCallExpression(node) &&
      node.expression.kind === typescript.SyntaxKind.ImportKeyword &&
      typescript.isStringLiteral(node.arguments[0]) &&
      node.arguments[0].text.endsWith('.json') &&
      !hasJSONAttributes(node.arguments[1])
    )
      failures.push(node.arguments[0].text);
    typescript.forEachChild(node, visit);
  }
  visit(parsed);
  return failures;
}

describe('lenses package registration', () => {
  it('rejects new bare runtime JSON imports outside the shared loader', () => {
    const root = join(REPOSITORY_ROOT, 'modules/lenses/src');
    const failures: string[] = [];
    for (const file of readdirSync(root, {recursive: true, encoding: 'utf8'})) {
      if (!/\.(ts|js|mjs)$/.test(file) || file === 'computation-lenses/embedded-dataset-loaders.js')
        continue;
      failures.push(
        ...collectBareJSONImports(readFileSync(join(root, file), 'utf8'), file).map(
          specifier => `${file}: ${specifier}`
        )
      );
    }
    expect(failures).toEqual([]);
    expect(
      collectBareJSONImports(
        `
      import type Data from './type.json';
      type Shape = typeof import('./shape.json');
      import value from './static.json';
      export {default} from './export.json';
      const loaded = import('./dynamic.json');
      const empty = import('./empty.json', {});
      const attributed = import('./valid.json', {with: {type: 'json'}});
    `,
        'fixture.ts'
      )
    ).toEqual(['./static.json', './export.json', './dynamic.json', './empty.json']);
  });
  it('documents the shared dataset loader and actual emission layout', () => {
    const text = readFileSync(
      join(REPOSITORY_ROOT, 'modules/lenses/src/computation-lenses/data/README.md'),
      'utf8'
    );
    expect(text).toContain("EMBEDDED_DATASET_LOADERS['s1-chicago-tracts.json']()");
    expect(text).toContain('modules/lenses/tsconfig.json');
    expect(text).not.toContain('modules/experimental/tsconfig.json');
    expect(text).not.toContain('esbuild) inlines');
  });
  it('bundles a browser role consumer with all dataset assets', async () => {
    const temporary = mkdtempSync(join(tmpdir(), 'lens-browser-bundle-'));
    try {
      const result = await build({
        entryPoints: [join(REPOSITORY_ROOT, 'modules/lenses/src/computation-lenses/user-input.ts')],
        outdir: temporary,
        bundle: true,
        platform: 'browser',
        format: 'esm',
        minify: true,
        write: false,
        metafile: true,
        logLevel: 'silent'
      });
      expect(result.warnings).toEqual([]);
      expect(
        Object.keys(result.metafile!.inputs).filter(file => file.endsWith('.json'))
      ).toHaveLength(33);
      for (const file of result.outputFiles) writeFileSync(file.path, file.contents);
      writeFileSync(join(temporary, 'package.json'), '{"type":"module"}');
      const output = execFileSync(
        process.execPath,
        [
          '-e',
          `
        const nodeProcess = process;
        globalThis.process = undefined;
        (async () => {
          const {listLensUserInputRoles} = await import(${JSON.stringify(pathToFileURL(join(temporary, 'user-input.js')).href)});
          const result = await listLensUserInputRoles({entry:'gpu-dataframe', dataset:{id:'f-chicago-tracts'}});
          nodeProcess.stdout.write(JSON.stringify(result));
        })().catch(error => {console.error(error); nodeProcess.exitCode = 1;});
      `
        ],
        {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']}
      );
      const roles = JSON.parse(output);
      const source = JSON.parse(
        readFileSync(
          join(REPOSITORY_ROOT, 'modules/lenses/src/computation-lenses/data/f-chicago-tracts.json'),
          'utf8'
        )
      );
      expect(roles.datasetId).toBe('f-chicago-tracts');
      expect(roles.roles.map((role: {role: string}) => role.role).sort()).toEqual(
        Object.keys(source.columns).sort()
      );
    } finally {
      rmSync(temporary, {recursive: true, force: true});
    }
  });
  it('loads attributed datasets through Vite source resolution', async () => {
    const result = await listLensUserInputRoles({
      entry: 'gpu-dataframe',
      dataset: {id: 'f-chicago-tracts'}
    } as LensScenarioDescriptor);
    expect(result?.datasetId).toBe('f-chicago-tracts');
    expect(result?.roles.length).toBeGreaterThan(0);
  });
  it('uses file entry points in lazy pack imports', () => {
    const root = join(REPOSITORY_ROOT, 'modules/lenses/src');
    const failures: string[] = [];
    for (const file of readdirSync(root, {recursive: true, encoding: 'utf8'})) {
      if (!file.endsWith('.ts')) continue;
      const source = readFileSync(join(root, file), 'utf8');
      for (const match of source.matchAll(/import\(['"]([^'"]+\/computation-pack)['"]\)/g))
        failures.push(`${file}: directory import ${match[1]}`);
    }
    expect(failures).toEqual([]);
  });

  it.runIf(process.env['LENS_VERIFY_BUILT'] === '1')(
    'checks emitted relative links and public role loading after a build',
    () => {
      const root = join(REPOSITORY_ROOT, 'modules/lenses/dist');
      expect(existsSync(root)).toBe(true);
      const failures: string[] = [];
      let checkedCount = 0;
      for (const file of readdirSync(root, {recursive: true, encoding: 'utf8'})) {
        if (!file.endsWith('.js')) continue;
        const source = readFileSync(join(root, file), 'utf8');
        for (const match of source.matchAll(/(?:from\s*|import\s*(?:\(\s*)?)['"](\.[^'"]+)['"]/g)) {
          checkedCount++;
          if (!existsSync(resolve(root, dirname(file), match[1])))
            failures.push(`${file}: ${match[1]}`);
        }
      }
      expect(checkedCount).toBeGreaterThan(0);
      expect(failures).toEqual([]);
      const output = execFileSync(
        process.execPath,
        [
          '--input-type=module',
          '-e',
          `import {listLensUserInputRoles} from '@luma-contributors/lenses/computation-lenses';
       import '@luma-contributors/lenses/computation-lenses/scenarios/index';
       const {getLensScenario} = await import('@luma-contributors/lenses/computation-lenses');
       for (const id of ['gpu-network.flow-aggregation', 'gpu-network.network-reachability']) {
         const loaded = await getLensScenario(id).load();
         if (typeof loaded.run !== 'function') throw new Error(id + ': missing run');
       }
       const result = await listLensUserInputRoles({entry:'gpu-dataframe', dataset:{id:'f-chicago-tracts'}});
       console.log(result.roles.length);`
        ],
        {cwd: REPOSITORY_ROOT, encoding: 'utf8'}
      );
      expect(Number(output.trim())).toBeGreaterThan(0);
      const commonJSOutput = execFileSync(
        process.execPath,
        [
          '-e',
          `const {listLensUserInputRoles} = require('@luma-contributors/lenses/computation-lenses');
       listLensUserInputRoles({entry:'gpu-dataframe', dataset:{id:'f-chicago-tracts'}})
         .then(result => console.log(result.roles.length));`
        ],
        {cwd: REPOSITORY_ROOT, encoding: 'utf8'}
      );
      expect(Number(commonJSOutput.trim())).toBeGreaterThan(0);
    }
  );

  it('loads public user-input roles in native Node ESM', async () => {
    const root = join(REPOSITORY_ROOT, 'modules/lenses/src/computation-lenses');
    const temporary = mkdtempSync(join(tmpdir(), 'lens-native-'));
    try {
      mkdirSync(join(temporary, 'data'));
      cpSync(join(root, 'data'), join(temporary, 'data'), {recursive: true});
      const loader = join(root, 'embedded-dataset-loaders.js');
      if (existsSync(loader)) cpSync(loader, join(temporary, 'embedded-dataset-loaders.js'));
      writeFileSync(join(temporary, 'package.json'), '{"type":"module"}');
      const result = await transform(readFileSync(join(root, 'user-input.ts'), 'utf8'), {
        loader: 'ts',
        format: 'esm',
        target: 'node22'
      });
      writeFileSync(join(temporary, 'user-input.mjs'), result.code);
      const output = execFileSync(
        process.execPath,
        [
          '--input-type=module',
          '-e',
          `import {listLensUserInputRoles} from ${JSON.stringify(pathToFileURL(join(temporary, 'user-input.mjs')).href)};
         const result = await listLensUserInputRoles({entry:'gpu-dataframe', dataset:{id:'f-chicago-tracts'}});
         console.log(JSON.stringify(result));`
        ],
        {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']}
      );
      const roles = JSON.parse(output);
      expect(roles.datasetId).toBe('f-chicago-tracts');
      expect(roles.roles.length).toBeGreaterThan(0);
      expect(roles.roles.every((role: {role: string}) => typeof role.role === 'string')).toBe(true);
    } finally {
      rmSync(temporary, {recursive: true, force: true});
    }
  });

  it.each([
    'esm',
    'cjs'
  ] as const)('loads every embedded dataset through production %s emission', async format => {
    const root = join(REPOSITORY_ROOT, 'modules/lenses/src/computation-lenses');
    const temporary = mkdtempSync(join(tmpdir(), 'lens-emission-'));
    try {
      cpSync(join(root, 'data'), join(temporary, 'data'), {recursive: true});
      writeFileSync(join(temporary, 'package.json'), '{"type":"module"}');
      const extension = format === 'esm' ? 'js' : 'cjs';
      for (const file of [
        'embedded-dataset-loaders.js',
        'user-input.ts',
        'scenarios/gpu-network/network-road-data.js',
        'scenarios/gpu-network/construction-data.js'
      ]) {
        const emitted = typescript.transpileModule(readFileSync(join(root, file), 'utf8'), {
          fileName: file,
          compilerOptions: {
            module: typescript.ModuleKind.ES2022,
            target: typescript.ScriptTarget.ES2022,
            allowJs: true
          }
        }).outputText;
        const code =
          format === 'esm' ? emitted : (await transformCommonJSModule(emitted, file)).code;
        const destination = join(temporary, file.replace(/\.(js|ts)$/, `.${extension}`));
        mkdirSync(dirname(destination), {recursive: true});
        writeFileSync(destination, code);
      }
      const loadModule = (file: string) =>
        format === 'esm'
          ? `await import(${JSON.stringify(pathToFileURL(join(temporary, file)).href)})`
          : `require(${JSON.stringify(join(temporary, file))})`;
      const output = execFileSync(
        process.execPath,
        [
          '-e',
          `
        (async () => {
          const {EMBEDDED_DATASET_LOADERS} = ${loadModule(`embedded-dataset-loaders.${extension}`)};
          const {listLensUserInputRoles} = ${loadModule(`user-input.${extension}`)};
          const roads = (${loadModule(`scenarios/gpu-network/network-road-data.${extension}`)}).default;
          const construction = ${loadModule(`scenarios/gpu-network/construction-data.${extension}`)};
          const roles = await listLensUserInputRoles({entry:'gpu-dataframe', dataset:{id:'f-chicago-tracts'}});
          const datasets = {};
          for (const [name, load] of Object.entries(EMBEDDED_DATASET_LOADERS)) datasets[name] = (await load()).default;
          console.log(JSON.stringify({roles, datasets, network: {roads, ...construction}}));
        })().catch(error => {console.error(error); process.exitCode = 1;});
      `
        ],
        {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 4 * 1024 * 1024}
      );
      const result = JSON.parse(output);
      expect(result.roles.datasetId).toBe('f-chicago-tracts');
      expect(result.roles.roles.length).toBeGreaterThan(0);
      const filenames = readdirSync(join(root, 'data'))
        .filter(file => file.endsWith('.json'))
        .sort();
      expect(Object.keys(result.datasets).sort()).toEqual(filenames);
      for (const filename of filenames) {
        expect(result.datasets[filename], filename).toEqual(
          JSON.parse(readFileSync(join(root, 'data', filename), 'utf8'))
        );
      }
      for (const [name, filename] of Object.entries({
        roads: 'n-chicago-loop-roads',
        crashes: 'n-chicago-loop-crashes',
        odFlows: 'n-chicago-od-flows',
        polylines: 'n-chicago-loop-polylines',
        traces: 'n-chicago-loop-traces'
      })) {
        expect(result.network[name], name).toEqual(result.datasets[`${filename}.json`]);
      }
    } finally {
      rmSync(temporary, {recursive: true, force: true});
    }
  });
  it('documents exactly the embedded dataset filenames', () => {
    const directory = join(REPOSITORY_ROOT, 'modules/lenses/src/computation-lenses/data');
    const inventory = readFileSync(join(directory, 'README.md'), 'utf8');
    const documented = [...inventory.matchAll(/^\| `?([a-z0-9-]+\.json)`? \|/gm)].map(
      match => match[1]
    );
    expect(documented.sort()).toEqual(
      readdirSync(directory)
        .filter(file => file.endsWith('.json'))
        .sort()
    );
  });
  it('retains the core result view through the optimized all-scenarios entry', async () => {
    const sourceRoot = join(REPOSITORY_ROOT, 'modules/lenses/src');
    const result = await build({
      stdin: {
        contents: `import './computation-lenses/scenarios/index';
          export {buildLensTwin} from './computation-views/twin/static-twin';`,
        resolveDir: sourceRoot
      },
      bundle: true,
      write: false,
      format: 'cjs',
      platform: 'node',
      packages: 'external',
      treeShaking: true,
      plugins: [
        {
          name: 'omit-domain-scenarios',
          setup(builder) {
            builder.onResolve({filter: /^\.\/gpu-/}, arguments_ =>
              arguments_.importer.endsWith('/computation-lenses/scenarios/index.ts')
                ? {path: arguments_.path, namespace: 'empty-scenarios'}
                : undefined
            );
            builder.onLoad({filter: /.*/, namespace: 'empty-scenarios'}, () => ({contents: ''}));
          }
        }
      ]
    });
    const compiled: {
      exports: {buildLensTwin?: (record: unknown, directive: unknown) => {viewId: string}};
    } = {exports: {}};
    new Function('require', 'module', 'exports', result.outputFiles[0].text)(
      createRequire(import.meta.url),
      compiled,
      compiled.exports
    );
    const record = JSON.parse(
      readFileSync(
        join(REPOSITORY_ROOT, 'docs/lens-records/gpu-dataframe.class-breaks.json'),
        'utf8'
      )
    );
    expect(
      compiled.exports.buildLensTwin!(record, {
        scenario: 'gpu-dataframe.class-breaks',
        rung: 'result'
      }).viewId
    ).toBe('gpu-core-array-result');
  });

  it('marks the distribution registration module as side effectful and stays private', () => {
    const manifest = JSON.parse(
      readFileSync(join(REPOSITORY_ROOT, 'modules/lenses/package.json'), 'utf8')
    );
    expect(manifest.private).toBe(true);
    expect(manifest.sideEffects).toContain('./dist/computation-views/packs/gpu-core/index.js');
    expect(manifest.sideEffects).toContain('./dist/computation-views/packs/gpu-core/index.cjs');
  });
});
