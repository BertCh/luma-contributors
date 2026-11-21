// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {afterEach, expect, it} from 'vitest';
import checkoutConfiguration from '../../../../vitest.config';
import {createServer} from 'vite';

const ROOT = fileURLToPath(new URL('../../../../', import.meta.url));
const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

function parseConfiguration(file: string) {
  const configuration = typescript.getParsedCommandLineOfConfigFile(
    file,
    {},
    {
      ...typescript.sys,
      onUnRecoverableConfigFileDiagnostic(diagnostic) {
        throw new Error(typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
      }
    }
  );
  expect(configuration?.errors.filter(error => error.code !== 18003)).toEqual([]);
  return configuration!.options;
}

it('ENT-11 maps exactly the manifest production entries and rejects private source paths', () => {
  const options = parseConfiguration(join(ROOT, 'tsconfig.json'));
  const manifest = JSON.parse(
    readFileSync(join(ROOT, 'modules/analysis/package.json'), 'utf8')
  ) as {
    exports: Record<string, {types: string}>;
  };
  const expected = Object.fromEntries(
    Object.entries(manifest.exports).map(([key, target]) => [
      `@luma-contributors/analysis${key === '.' ? '' : key.slice(1)}`,
      [target.types.replace('./dist/', './modules/analysis/src/').replace(/\.d\.ts$/, '.ts')]
    ])
  );
  expect(
    Object.fromEntries(
      Object.entries(options.paths ?? {}).filter(
        ([key]) =>
          key === '@luma-contributors/analysis' || key.startsWith('@luma-contributors/analysis/')
      )
    )
  ).toEqual(expected);
  const importer = join(ROOT, 'modules/kepler/src/production-probe.ts');
  for (const [specifier, [source]] of Object.entries(expected)) {
    expect(
      typescript.resolveModuleName(specifier, importer, options, typescript.sys).resolvedModule
        ?.resolvedFileName
    ).toBe(resolve(ROOT, source));
  }
  for (const suffix of [
    'gpu-spatial-analysis/spatial-interpolation',
    'testing',
    'test/utils/gpu-contributor-test-utils'
  ]) {
    expect(
      typescript.resolveModuleName(
        `@luma-contributors/analysis/${suffix}`,
        importer,
        options,
        typescript.sys
      ).resolvedModule,
      suffix
    ).toBeUndefined();
  }
});

it('ENT-11 keeps private analysis helpers available to the actual lenses test configuration', () => {
  const options = parseConfiguration(join(ROOT, 'modules/lenses/tsconfig.test.json'));
  for (const suffix of ['testing', 'test/utils/gpu-contributor-test-utils']) {
    const resolved = typescript.resolveModuleName(
      `@luma-contributors/analysis/${suffix}`,
      join(ROOT, 'modules/lenses/test/probe.ts'),
      options,
      typescript.sys
    ).resolvedModule;
    expect(resolved?.resolvedFileName).toBe(
      join(
        ROOT,
        suffix === 'testing'
          ? 'modules/analysis/src/testing/index.ts'
          : 'modules/analysis/test/utils/gpu-contributor-test-utils.ts'
      )
    );
  }
  expect(options.paths?.['@luma-contributors/kepler']).toBeDefined();
});

it.each([
  'plain',
  'comments',
  'extends-only'
])('ENT-11 generates test aliases while preserving inherited paths, ambient types, Vite and family stages (%s)', rootForm => {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'analysis-test-resolution-')));
  directories.push(directory);
  function write(file: string, contents: string) {
    mkdirSync(dirname(join(directory, file)), {recursive: true});
    writeFileSync(join(directory, file), contents);
  }
  for (const script of ['typecheck-tests.mjs', 'verification-policy.mjs'])
    write(`scripts/${script}`, readFileSync(join(ROOT, 'scripts', script), 'utf8'));
  // Explicit fixture settings keep this regression independent of the checkout root's JSONC
  // spelling and inheritance. The generator itself must resolve each supported root form.
  const originalRoot = JSON.stringify({
    compilerOptions: {
      types: ['@webgpu/types', 'node'],
      target: 'ES2022',
      module: 'ES2022',
      moduleResolution: 'bundler',
      strict: true,
      skipLibCheck: true,
      noEmit: true,
      paths: {'fixture/*': ['./fixtures/*']}
    },
    include: ['modules/*/src', 'modules/*/test']
  });
  write('tsconfig.parent.json', originalRoot);
  write(
    'tsconfig.base.json',
    rootForm === 'comments'
      ? `// Valid root JSONC\n${originalRoot}`
      : rootForm === 'extends-only'
        ? '{"extends":"./tsconfig.parent.json"}'
        : originalRoot
  );
  write(
    'tsconfig.json',
    '// JSONC and inherited compiler options\n{"extends":"./tsconfig.base.json"}'
  );
  write(
    'test/kepler-compute-example-specs.json',
    '{"excludedFromRuns":{},"excludedFromTypecheck":{}}'
  );
  const source = `import {testValue} from '@luma-contributors/analysis/testing';
    import {helperValue} from '@luma-contributors/analysis/test/utils/helper';
    import {inheritedValue} from 'fixture/inherited';
    export const result: number = testValue + helperValue + inheritedValue;
    export const mode: string = import.meta.env.MODE;`;
  for (const family of ['analysis/analysis-core', 'lenses/lens-family', 'kepler/kepler-family']) {
    const [packageName, testFamily] = family.split('/');
    write(`modules/${packageName}/test/${testFamily}/probe.node.spec.ts`, source);
  }
  write('modules/analysis/src/testing/index.ts', 'export const testValue = 1;');
  write('modules/analysis/test/utils/helper.ts', 'export const helperValue = 2;');
  write('fixtures/inherited.ts', 'export const inheritedValue = 3;');
  mkdirSync(join(directory, 'node_modules'), {recursive: true});
  for (const dependency of ['typescript', 'vite', '@types', '@webgpu'])
    symlinkSync(
      join(ROOT, 'node_modules', dependency),
      join(directory, 'node_modules', dependency),
      'dir'
    );
  const generated = spawnSync(
    process.execPath,
    ['scripts/typecheck-tests.mjs', '--generate-only'],
    {
      cwd: directory,
      encoding: 'utf8',
      timeout: 10000
    }
  );
  expect(generated.status, `${generated.stdout}${generated.stderr}`).toBe(0);
  const stages = JSON.parse(
    readFileSync(join(directory, '.typecheck/test-family-stages.json'), 'utf8')
  ) as {label: string; configuration: string; files: string[]}[];
  expect(
    stages
      .filter(stage => stage.files.some(file => file.endsWith('/probe.node.spec.ts')))
      .map(stage => stage.label)
  ).toEqual([
    'tests: analysis-analysis-core',
    'tests: lenses-lens-family',
    'tests: kepler-kepler-family'
  ]);
  for (const stage of stages.filter(stage =>
    stage.files.some(file => file.endsWith('/probe.node.spec.ts'))
  )) {
    const options = parseConfiguration(join(directory, stage.configuration));
    expect(options.types).toEqual(['@webgpu/types', 'node', 'vite/client']);
    expect(options.incremental).toBe(true);
    expect(options.tsBuildInfoFile).toContain(`${stage.label.slice('tests: '.length)}.tsbuildinfo`);
    // A tiny real program checks the imports and import.meta.env, rather than just inspecting JSON.
    const program = typescript.createProgram(
      stage.files
        .filter(file => file.endsWith('/probe.node.spec.ts'))
        .map(file => join(directory, file)),
      {...options, incremental: false}
    );
    expect(
      typescript
        .getPreEmitDiagnostics(program)
        .map(diagnostic => typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'))
    ).toEqual([]);
  }
  for (const configuration of [
    '.typecheck/tsconfig.example-specs.json',
    '.typecheck/tsconfig.website-specs.json'
  ]) {
    const generatedOptions = JSON.parse(
      readFileSync(join(directory, configuration), 'utf8')
    ).compilerOptions;
    expect(generatedOptions.paths['fixture/*']).toEqual([join(directory, 'fixtures/*')]);
    expect(generatedOptions.paths['@luma-contributors/analysis/testing']).toEqual([
      join(directory, 'modules/analysis/src/testing/index.ts')
    ]);
  }
}, 30000);

it('review B2 separates production roots from editor test projects', () => {
  const configuration = typescript.getParsedCommandLineOfConfigFile(
    join(ROOT, 'tsconfig.json'),
    {},
    {
      ...typescript.sys,
      onUnRecoverableConfigFileDiagnostic() {
        throw new Error('invalid root configuration');
      }
    }
  );
  expect(configuration!.fileNames.filter(file => /\/test\//.test(file)).length).toBe(0);
});

it('review B2 resolves private test imports through the real Vite resolver with the shared test config', async () => {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'analysis-vite-resolution-')));
  directories.push(directory);
  const configuration = checkoutConfiguration;
  const server = await createServer({
    configFile: false,
    root: ROOT,
    cacheDir: join(directory, 'vite-cache'),
    resolve: configuration.resolve,
    server: {middlewareMode: true, watch: null, hmr: false, preTransformRequests: false},
    optimizeDeps: {noDiscovery: true, include: []},
    logLevel: 'silent'
  });
  try {
    for (const [suffix, target] of [
      ['testing', 'modules/analysis/src/testing/index.ts'],
      [
        'test/utils/gpu-contributor-test-utils',
        'modules/analysis/test/utils/gpu-contributor-test-utils.ts'
      ]
    ]) {
      for (const importer of [
        'modules/lenses/test/computation-lenses/record-structure.node.spec.ts',
        'modules/kepler/test/curated/curated-bindings-clusters.node.spec.ts'
      ]) {
        const resolved = await server.environments['ssr']!.pluginContainer.resolveId(
          `@luma-contributors/analysis/${suffix}`,
          join(ROOT, importer)
        );
        expect(resolved?.id, importer).toBe(join(ROOT, target));
      }
    }
  } finally {
    await server.close();
  }
}, 15000);

it.each([
  'modules/lenses/test/computation-lenses/record-structure.node.spec.ts',
  'modules/kepler/test/curated/curated-bindings-clusters.node.spec.ts'
])('W5-01 discovers the editor test project for %s', importer => {
  const importerPath = join(ROOT, importer);
  const configurationPath = typescript.findConfigFile(
    dirname(importerPath),
    typescript.sys.fileExists
  );
  expect(configurationPath).toBe(
    join(ROOT, importer.split('/').slice(0, 3).join('/'), 'tsconfig.json')
  );
  const configuration = typescript.getParsedCommandLineOfConfigFile(
    configurationPath!,
    {},
    {
      ...typescript.sys,
      onUnRecoverableConfigFileDiagnostic() {
        throw new Error('invalid editor project');
      }
    }
  );
  expect(configuration!.errors).toEqual([]);
  expect(configuration!.fileNames).toContain(importerPath);
  for (const [suffix, target] of [
    ['testing', 'modules/analysis/src/testing/index.ts'],
    [
      'test/utils/gpu-contributor-test-utils',
      'modules/analysis/test/utils/gpu-contributor-test-utils.ts'
    ]
  ]) {
    expect(
      typescript.resolveModuleName(
        `@luma-contributors/analysis/${suffix}`,
        importerPath,
        configuration!.options,
        typescript.sys
      ).resolvedModule?.resolvedFileName
    ).toBe(join(ROOT, target));
  }
});

it('review S1 resolves checkout config aliases when launched from a package directory', () => {
  const directory = realpathSync(mkdtempSync(join(tmpdir(), 'analysis-package-cwd-')));
  directories.push(directory);
  const result = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `
    import assert from 'node:assert/strict';
    import {join} from 'node:path';
    import {loadConfigFromFile, createServer} from 'vite';
    const root = ${JSON.stringify(ROOT)};
    const loaded = await loadConfigFromFile({command: 'serve', mode: 'test'},
      join(root, 'vitest.config.ts'), root, 'silent', undefined, 'runner');
    assert.ok(loaded);
    const server = await createServer({configFile: false, root,
      cacheDir: ${JSON.stringify(join(directory, 'vite-cache'))}, resolve: loaded.config.resolve,
      server: {middlewareMode: true, watch: null, hmr: false, preTransformRequests: false},
      optimizeDeps: {noDiscovery: true, include: []}, logLevel: 'silent'});
    try {
      for (const importer of [
        'modules/lenses/test/computation-lenses/record-structure.node.spec.ts',
        'modules/kepler/test/curated/curated-bindings-clusters.node.spec.ts'
      ]) for (const [suffix, target] of [
        ['testing', 'modules/analysis/src/testing/index.ts'],
        ['test/utils/gpu-contributor-test-utils', 'modules/analysis/test/utils/gpu-contributor-test-utils.ts']
      ]) {
        const resolved = await server.environments.ssr.pluginContainer.resolveId(
          '@luma-contributors/analysis/' + suffix, join(root, importer));
        assert.equal(resolved?.id, join(root, target), importer + ': ' + suffix);
      }
      console.log('package-directory aliases passed');
    } finally {await server.close();}
  `
    ],
    {cwd: join(ROOT, 'modules/kepler'), encoding: 'utf8', timeout: 15000}
  );
  expect(result.status, `${result.stdout}${result.stderr}`).toBe(0);
  expect(result.stdout).toContain('package-directory aliases passed');
}, 20000);
