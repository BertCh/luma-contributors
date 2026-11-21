// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {afterEach, describe, expect, it} from 'vitest';
import {assertPackedFileList} from '../../../../scripts/packed-file-inventory.mjs';
import {createPackedConsumerContractFixture} from '../utils/packed-consumer-fixture';

const ROOT = fileURLToPath(new URL('../../../../', import.meta.url));
const CONTRACT = createPackedConsumerContractFixture();
const CHECKER_PATH = fileURLToPath(
  new URL('../../../../scripts/check-packed-consumer.mjs', import.meta.url)
);
const fixtureDirectories: string[] = [];
const SPATIAL_DECLARATION = [
  'export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: {wordLength: number};',
  "export declare const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA: {format: 'float32'; wordLength: number};",
  CONTRACT.declarationSource
].join('\n');

function writeFixtureFile(directory: string, path: string, text: string): void {
  const target = join(directory, path);
  mkdirSync(dirname(target), {recursive: true});
  writeFileSync(target, text);
}

function createFixture(): string {
  const directory = mkdtempSync(join(tmpdir(), 'packed-consumer-fixture-'));
  fixtureDirectories.push(directory);
  // This package fixture models only the resident declaration protocol exercised by the
  // packed checker. Numerical aggregation is covered by the contributor's independent specs.
  const zonalFixture = `class GPUZonalStatistics {
    constructor(props) {this.props = props; this.sumOrder = 'atomic';}
    getCommandNodes() {
      return this.props.values.data.map((chunk, index) => ({
        id: this.props.id + '-' + index, type: 'compute',
        resources: [{buffer: chunk, usage: 'storage-read'},
          {buffer: this.props.output.sums, usage: 'storage-read-write'}],
        compile: () => ({encode() {}})
      }));
    }
  }`;
  const computeFunction = `function computeVariogramModelFitOnCPU(bins, options) {
    if (bins.distances.length !== 20 || options.models[1] !== 'exponential') throw new Error('wrong documented input');
    return {parameters: Float32Array.of(0, 0, 0, 0, 0.2, 1.5, 2.2, 0, 0, 0, 0, 0), status: Uint32Array.of(0, 0, 0)};
  }`;
  const entrySource =
    '// SPDX-License-Identifier: MIT\n// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors\nexport const value = 1;';
  const sourceMap = JSON.stringify({
    version: 3,
    sources: ['../src/index.ts'],
    sourcesContent: [entrySource],
    mappings: 'AAAA'
  });
  const files: Record<string, string> = {
    'package/package.json': JSON.stringify({
      name: '@luma-contributors/analysis',
      version: '0.0.0',
      type: 'module',
      license: 'MIT',
      files: ['dist', 'LICENSES', 'THIRD-PARTY-NOTICES.md'],
      scripts: {prepack: 'node -e "throw new Error(\'prepack must not run\')"'},
      dependencies: {
        'fixture-dependency': '1.0.0',
        ...Object.fromEntries(
          ['@luma.gl/core', '@luma.gl/graph'].map(name => [
            name,
            JSON.parse(readFileSync(join(ROOT, 'node_modules', name, 'package.json'), 'utf8'))
              .version
          ])
        )
      },
      peerDependencies: {'fixture-peer': '1.0.0'},
      exports: {
        ...Object.fromEntries(
          CONTRACT.publicEntryKeys.map(key => [
            key,
            {
              types: './dist/index.d.ts',
              import: './dist/index.js',
              require: './dist/index.cjs'
            }
          ])
        ),
        '.': {types: './dist/index.d.ts', import: './dist/index.js', require: './dist/index.cjs'},
        './dataframe-analysis': {
          types: './dist/dataframe-analysis/index.d.ts',
          import: './dist/dataframe-analysis/index.js',
          require: './dist/dataframe-analysis/index.cjs'
        },
        './gpu-spatial-analysis': {
          types: './dist/gpu-spatial-analysis/index.d.ts',
          import: './dist/gpu-spatial-analysis/index.js',
          require: './dist/gpu-spatial-analysis/index.cjs'
        }
      }
    }),
    'package/README.md': '[licence](LICENSE)\n[web](https://example.com/docs)\n',
    'package/LICENSE': 'MIT fixture licence',
    'package/LICENSES/Apache-2.0.txt': 'Apache fixture licence',
    'package/LICENSES/BSD-3-Clause.txt': 'BSD fixture licence',
    'package/LICENSES/H3-NOTICE.txt': 'H3 fixture notice',
    'package/THIRD-PARTY-NOTICES.md': 'Default source expression: `MIT`.',
    'package/src/index.ts': entrySource,
    'package/dist/index.js.map': sourceMap,
    'package/dist/index.cjs.map': sourceMap,
    'package/dist/index.js': `import dependency from 'fixture-dependency'; import peer from 'fixture-peer'; export const value = dependency + peer;\n${CONTRACT.esmSource}\nexport {${CONTRACT.exportNames.join(',')}};`,
    'package/dist/index.cjs': `exports.value = require('fixture-dependency') + require('fixture-peer');\n${CONTRACT.commonjsSource}\n${CONTRACT.exportNames.map(name => `exports.${name} = ${name};`).join('\n')}`,
    'package/dist/index.d.ts': `export declare const value: number;\n${CONTRACT.declarationSource}`,
    'package/dist/gpu-spatial-analysis/index.js': `export ${zonalFixture}\nexport const GPU_LINE_DENSITY_PARAMETER_SCHEMA = {wordLength: 4}; export const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA = {format: 'float32', wordLength: 4};`,
    'package/dist/gpu-spatial-analysis/index.cjs': `${zonalFixture}\nexports.GPUZonalStatistics = GPUZonalStatistics; exports.GPU_LINE_DENSITY_PARAMETER_SCHEMA = {wordLength: 4}; exports.GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA = {format: 'float32', wordLength: 4};`,
    'package/dist/gpu-spatial-analysis/index.d.ts': SPATIAL_DECLARATION,
    'package/dist/dataframe-analysis/index.js': `export ${computeFunction}`,
    'package/dist/dataframe-analysis/index.cjs': `${computeFunction}\nexports.computeVariogramModelFitOnCPU = computeVariogramModelFitOnCPU;`,
    'package/dist/dataframe-analysis/index.d.ts': `export declare function computeVariogramModelFitOnCPU(...args: unknown[]): {parameters: Float32Array; status: Uint32Array};\n${CONTRACT.declarationSource}`
  };
  for (const name of ['fixture-dependency', 'fixture-peer']) {
    files[`node_modules/${name}/package.json`] = JSON.stringify({
      name,
      version: '1.0.0',
      main: 'index.cjs'
    });
    files[`node_modules/${name}/index.cjs`] = 'module.exports = 1;';
  }
  for (const [path, text] of Object.entries(files)) writeFixtureFile(directory, path, text);
  for (const name of ['@luma.gl/core', '@luma.gl/graph']) {
    mkdirSync(dirname(join(directory, 'node_modules', name)), {recursive: true});
    symlinkSync(join(ROOT, 'node_modules', name), join(directory, 'node_modules', name), 'dir');
  }
  return directory;
}

function countFixtureDeclarations(directory: string): number {
  // All dist files are packed; public entries deliberately share three declaration targets.
  return readdirSync(join(directory, 'package/dist'), {recursive: true}).filter(
    file => typeof file === 'string' && /\.d\.[cm]?ts$/.test(file)
  ).length;
}

function writeSpatialDeclaration(directory: string, declaration: string): void {
  writeFixtureFile(
    directory,
    'package/dist/gpu-spatial-analysis/index.d.ts',
    `${declaration}\n${CONTRACT.declarationSource}`
  );
}

function runChecker(
  directory: string,
  environment: NodeJS.ProcessEnv = {},
  executableArguments: string[] = []
) {
  const result = spawnSync(
    process.execPath,
    [
      ...executableArguments,
      CHECKER_PATH,
      '--package-directory',
      join(directory, 'package'),
      '--dependency-directory',
      join(directory, 'node_modules')
    ],
    {encoding: 'utf8', timeout: 30_000, env: {...process.env, ...environment}}
  );
  return {exitCode: result.status, output: `${result.stdout}${result.stderr}`};
}

type FixtureManifest = {
  files: string[];
  license: string;
  exports: Record<string, {types: string; import: string; require: string}>;
  dependencies: Record<string, string>;
  peerDependencies: Record<string, string>;
};

function editManifest(directory: string, edit: (manifest: FixtureManifest) => void): void {
  const path = join(directory, 'package/package.json');
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  edit(manifest);
  writeFileSync(path, JSON.stringify(manifest));
}

afterEach(() => {
  for (const directory of fixtureDirectories.splice(0)) {
    rmSync(directory, {recursive: true, force: true});
  }
});

describe('packed consumer checker', () => {
  it.each([
    'src/index.ts',
    'src/testing/helper.ts',
    'dist/testing/helper.js',
    'dist/.build-record.json'
  ])('Decision 28 rejects %s in a packed file list', path => {
    expect(() => assertPackedFileList(['package/dist/index.js', `package/${path}`])).toThrow(
      `packed file policy: forbidden path ${path}`
    );
  });

  it('Decision 28 permits compiled output, source maps and notices in a packed file list', () => {
    expect(() =>
      assertPackedFileList([
        'package/package.json',
        'package/README.md',
        'package/THIRD-PARTY-NOTICES.md',
        'package/dist/index.js',
        'package/dist/index.cjs',
        'package/dist/index.d.ts',
        'package/dist/index.js.map'
      ])
    ).not.toThrow();
  });

  it('rejects a fixture missing the zonal resident-vector declaration contract', () => {
    const directory = createFixture();
    const path = 'package/dist/gpu-spatial-analysis/index.js';
    const source = readFileSync(join(directory, path), 'utf8');
    writeFixtureFile(
      directory,
      path,
      source.replace('export class GPUZonalStatistics', 'class GPUZonalStatistics')
    );
    const commonjsPath = 'package/dist/gpu-spatial-analysis/index.cjs';
    writeFixtureFile(
      directory,
      commonjsPath,
      readFileSync(join(directory, commonjsPath), 'utf8').replace(
        'exports.GPUZonalStatistics = GPUZonalStatistics;',
        ''
      )
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('GPUZonalStatistics is not a constructor');
  });
  it('rejects different CJS names with the same count', () => {
    const directory = createFixture();
    const commonjsEntry = join(directory, 'package/dist/index.cjs');
    writeFileSync(
      commonjsEntry,
      readFileSync(commonjsEntry, 'utf8').replace('exports.value', 'exports.other')
    );
    const inventory = spawnSync(
      process.execPath,
      [
        '--input-type=module',
        '--eval',
        `
        import {createRequire} from 'node:module';
        const require = createRequire(import.meta.url);
        const imported = await import(${JSON.stringify(join(directory, 'package/dist/index.js'))});
        const required = require(${JSON.stringify(commonjsEntry)});
        console.log(JSON.stringify([Object.keys(imported).sort(), Object.keys(required).sort()]));
      `
      ],
      {encoding: 'utf8', timeout: 10000}
    );
    expect(inventory.status, inventory.stderr).toBe(0);
    const [importedNames, requiredNames]: string[][] = JSON.parse(inventory.stdout);
    const expectedNames = ['value', ...CONTRACT.exportNames].sort();
    expect(importedNames).toEqual(expectedNames);
    expect(requiredNames).toHaveLength(expectedNames.length);
    expect(requiredNames).toEqual(
      expectedNames.map(name => (name === 'value' ? 'other' : name)).sort()
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('export names differ');
  });

  it.each([
    'present',
    'missing',
    'outside'
  ] as const)('W2-01 checks a relative reference types edge whose target is %s', state => {
    const directory = createFixture();
    // Declaration edges are checked before dependencies are linked. Escape to an already
    // existing fixture file through the filesystem root, independently of the isolated temp path.
    const outsideTarget = join(directory, 'outside.d.ts');
    const reference =
      state === 'outside' ? `${'../'.repeat(64)}${outsideTarget.slice(1, -5)}` : './ambient';
    writeFixtureFile(
      directory,
      'package/dist/reference-types.d.cts',
      `/// <reference types="${reference}" preserve="true" />\nexport {};\n`
    );
    if (state === 'present') writeFixtureFile(directory, 'package/dist/ambient.d.ts', 'export {};');
    if (state === 'outside') {
      writeFixtureFile(directory, 'outside.d.ts', 'export {};');
      expect(
        resolve(
          '/independent/consumer/node_modules/@luma-contributors/analysis/dist',
          `${reference}.d.ts`
        )
      ).toBe(outsideTarget);
      expect(readFileSync(outsideTarget, 'utf8')).toBe('export {};');
    }
    const declaringPath = join(directory, 'package/dist/reference-types.d.cts');
    const options = {
      module: typescript.ModuleKind.NodeNext,
      moduleResolution: typescript.ModuleResolutionKind.NodeNext
    };
    const source = typescript.createSourceFile(
      declaringPath,
      readFileSync(declaringPath, 'utf8'),
      {
        languageVersion: typescript.ScriptTarget.Latest,
        impliedNodeFormat: typescript.getImpliedNodeFormatForFile(
          declaringPath,
          undefined,
          typescript.sys,
          options
        )
      },
      true
    );
    const directive = source.typeReferenceDirectives[0];
    const mode = typescript.getModeForFileReference(directive, source.impliedNodeFormat);
    const resolved = typescript.resolveTypeReferenceDirective(
      directive.fileName,
      declaringPath,
      options,
      typescript.sys,
      undefined,
      undefined,
      mode
    ).resolvedTypeReferenceDirective?.resolvedFileName;
    if (state === 'present' || state === 'outside') {
      expect(mode).toBe(typescript.ModuleKind.CommonJS);
      expect(
        resolved,
        'the target must resolve before package containment is tested'
      ).toBeDefined();
      if (resolved)
        expect(realpathSync(resolved)).toBe(
          realpathSync(
            state === 'present' ? join(directory, 'package/dist/ambient.d.ts') : outsideTarget
          )
        );
    } else expect(resolved).toBeUndefined();
    const result = runChecker(directory);
    if (state === 'present') {
      expect(result.exitCode, result.output).toBe(0);
      expect(result.output).toContain(
        `resolved 1 relative references in ${countFixtureDeclarations(directory)} packed declaration files`
      );
    } else {
      expect(result.exitCode, result.output).toBe(1);
      expect(result.output).toContain('1 of 1 relative declaration references do not resolve');
      expect(result.output).toContain('dist/reference-types.d.cts:1');
      expect(result.output).toContain(reference);
    }
  });

  it('W2-01 rejects extensionless reference types in an ESM declaration under native NodeNext', () => {
    const directory = createFixture();
    const declaringPath = join(directory, 'package/dist/reference-types.d.ts');
    writeFixtureFile(
      directory,
      'package/dist/reference-types.d.ts',
      '/// <reference types="./ambient" preserve="true" />\nexport {};\n'
    );
    writeFixtureFile(directory, 'package/dist/ambient.d.ts', 'export {};');
    const options = {
      module: typescript.ModuleKind.NodeNext,
      moduleResolution: typescript.ModuleResolutionKind.NodeNext,
      types: [],
      noEmit: true
    };
    const program = typescript.createProgram([declaringPath], options);
    const source = program.getSourceFile(declaringPath)!;
    const directive = source.typeReferenceDirectives[0];
    const mode = typescript.getModeForFileReference(directive, source.impliedNodeFormat);
    expect(mode).toBe(typescript.ModuleKind.ESNext);
    expect(
      typescript.resolveTypeReferenceDirective(
        directive.fileName,
        declaringPath,
        options,
        typescript.sys,
        undefined,
        undefined,
        mode
      ).resolvedTypeReferenceDirective
    ).toBeUndefined();
    expect(typescript.getPreEmitDiagnostics(program).map(diagnostic => diagnostic.code)).toContain(
      2688
    );
    // Omitting mode would incorrectly accept the extensionless ESM reference.
    expect(
      typescript.resolveTypeReferenceDirective(
        directive.fileName,
        declaringPath,
        options,
        typescript.sys
      ).resolvedTypeReferenceDirective
    ).toBeDefined();
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('dist/reference-types.d.ts:1');
  });

  it('final1 F8 preserves a 256 MiB heap in the isolated runtime while clearing module-loading options', () => {
    const directory = createFixture();
    const preload = join(directory, 'preload.cjs');
    writeFileSync(preload, 'globalThis.loadingOptionWasInherited = true;');
    const entry = join(directory, 'package/dist/index.js');
    writeFileSync(
      entry,
      `import {getHeapStatistics} from 'node:v8';\nconst heapMegabytes = getHeapStatistics().heap_size_limit / 1024 ** 2;\nprocess.stdout.write('packed-worker-heap-mib=' + heapMegabytes + '\\n');\nif (globalThis.loadingOptionWasInherited) throw new Error('module-loading option leaked');\nif (heapMegabytes >= 384) throw new Error('caller heap was dropped');\n${readFileSync(entry, 'utf8')}`
    );
    for (const environmentHeap of [256, 512]) {
      const result = runChecker(
        directory,
        {
          NODE_OPTIONS: `--max-old-space-size=${environmentHeap} --require=${preload}`
        },
        environmentHeap === 512 ? ['--max-old-space-size=256'] : []
      );
      expect(result.exitCode, result.output).toBe(0);
      const heapMegabytes = Number(result.output.match(/packed-worker-heap-mib=([\d.]+)/)?.[1]);
      expect(heapMegabytes).toBeGreaterThan(256);
      expect(heapMegabytes).toBeLessThan(384);
    }
  });
  it('rejects an external TMPDIR symlink that resolves into the workspace', () => {
    const directory = createFixture();
    const workspaceTemporaryDirectory = mkdtempSync(
      fileURLToPath(
        new URL('../../../../.typecheck/release-package-real-isolation-', import.meta.url)
      )
    );
    fixtureDirectories.push(workspaceTemporaryDirectory);
    const alias = join(directory, 'temporary-root');
    symlinkSync(workspaceTemporaryDirectory, alias, 'dir');
    const result = runChecker(directory, {TMPDIR: alias});
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('temporary directory must be outside the workspace');
  });

  it('accepts a later stable peer version allowed by the documented prerelease range', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.peerDependencies['fixture-peer'] = '~10.0.0-alpha.16';
    });
    writeFixtureFile(
      directory,
      'node_modules/fixture-peer/package.json',
      JSON.stringify({name: 'fixture-peer', version: '10.0.99', main: 'index.cjs'})
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
  });
  it('TOOL-102 rejects an extensionless relative import type even when its target declaration exists', () => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      'package/dist/gpu-spatial-analysis/schema.d.ts',
      "export type SegmentizeSchema = {format: 'float32'; wordLength: number};"
    );
    // The shape TypeScript emits for an inferred type: `moduleResolution: bundler` accepts it,
    // NodeNext does not, and `skipLibCheck` turns the type into `any` without a diagnostic.
    writeSpatialDeclaration(
      directory,
      'export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: {wordLength: number};\nexport declare const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA: import("./schema").SegmentizeSchema;\nexport declare const other: import("..").Missing;'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('2 of 2 relative declaration references do not resolve');
    expect(result.output).toContain('dist/gpu-spatial-analysis/index.d.ts:2 "./schema"');
    expect(result.output).toContain('dist/gpu-spatial-analysis/index.d.ts:3 ".."');
  });
  it('TOOL-102 accepts the same import type through its explicit .js specifier', () => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      'package/dist/gpu-spatial-analysis/schema.d.ts',
      "export type SegmentizeSchema = {format: 'float32'; wordLength: number};"
    );
    writeSpatialDeclaration(
      directory,
      'export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: {wordLength: number};\nexport declare const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA: import("./schema.js").SegmentizeSchema;'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
    expect(result.output).toContain(
      `resolved 1 relative references in ${countFixtureDeclarations(directory)} packed declaration files`
    );
  });
  it('TOOL-115 type-checks a CommonJS consumer beside the ES module consumer', () => {
    const directory = createFixture();
    writeSpatialDeclaration(
      directory,
      'export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: {wordLength: number};\nexport declare const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA: any;'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toMatch(/native-consumer\.cts\(\d+,\d+\): error TS2578/);
    expect(result.output).toMatch(/native-consumer\.mts\(\d+,\d+\): error TS2578/);
    expect([...result.output.matchAll(/error TS(\d+):/g)].map(match => match[1])).toEqual([
      '2578',
      '2578'
    ]);
  });
  it('rejects a declaration that erases the schema to any', () => {
    const directory = createFixture();
    writeSpatialDeclaration(
      directory,
      'export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: any;\nexport declare const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA: {format: "float32"; wordLength: number};'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    // The density invented-field sentinel is in the ES consumer; CommonJS checks segmentize.
    expect(result.output).toMatch(/native-consumer\.mts\(\d+,\d+\): error TS2578/);
    expect([...result.output.matchAll(/error TS(\d+):/g)].map(match => match[1])).toEqual(['2578']);
  });
  it('rejects a workspace-local TMPDIR before an undeclared workspace import can resolve', () => {
    const directory = createFixture();
    const workspaceTemporaryDirectory = mkdtempSync(
      fileURLToPath(new URL('../../../../.typecheck/release-package-isolation-', import.meta.url))
    );
    fixtureDirectories.push(workspaceTemporaryDirectory);
    writeFixtureFile(
      directory,
      'package/dist/index.js',
      "import 'typescript'; export const value = 1;"
    );
    writeFixtureFile(
      directory,
      'package/dist/index.cjs',
      "require('typescript'); exports.value = 1;"
    );
    const result = runChecker(directory, {TMPDIR: workspaceTemporaryDirectory});
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('temporary directory must be outside the workspace');
  });

  it.each(['md', 'js'])('checks SPDX headers in build source .%s files', extension => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      `package/src/guide.${extension}`,
      '// SPDX-License-Identifier: ISC\n'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('unlisted SPDX expression ISC');
  });

  it('checks a reference link defined inside a blockquote', () => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      'package/README.md',
      '> [guide][reference]\n>\n> [reference]: missing.md\n'
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('README link missing');
  });

  it('ignores links inside indented Markdown code blocks', () => {
    const directory = createFixture();
    writeFixtureFile(directory, 'package/README.md', '    [example](missing.md)\n');
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
  });

  it.each([
    'dependencies',
    'peerDependencies'
  ] as const)('rejects a present installed version outside its %s range', section => {
    const directory = createFixture();
    const name = section === 'dependencies' ? 'fixture-dependency' : 'fixture-peer';
    editManifest(directory, manifest => {
      manifest[section][name] = '>=2.0.0';
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(`${name}@1.0.0 does not satisfy >=2.0.0`);
  });
  it('packs offline, links only declared dependencies and peers, and checks both native formats', () => {
    const directory = createFixture();
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
    expect(result.output).toContain('verified 11 packed entry points');
    expect(result.output).toContain(
      `resolved 0 relative references in ${countFixtureDeclarations(directory)} packed declaration files`
    );
    expect(result.output).toContain('native NodeNext declaration consumer passed');
    expect(result.output).toContain('documented CPU variogram fit passed');
    expect(result.output).toContain('verified packed ESM and CommonJS root source maps');
  });

  it('requires the packed CommonJS map to embed the original entry-point source', () => {
    const directory = createFixture();
    writeFixtureFile(
      directory,
      'package/dist/index.cjs.map',
      JSON.stringify({version: 3, sources: ['../src/index.ts'], mappings: 'AAAA'})
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('entry-point source map ./dist/index.cjs.map must embed');
  });

  it('requires the ESM entry-point map to be in the tarball', () => {
    const directory = createFixture();
    editManifest(directory, manifest => manifest.files.push('!dist/index.js.map'));
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(
      'entry-point source map: missing packed file ./dist/index.js.map'
    );
  });

  it.each([
    ['missing declaration', 'package/dist/index.d.ts', '', 'types target'],
    [
      'broken native import',
      'package/dist/index.js',
      "export {value} from './workspace-only.js';",
      'workspace-only.js'
    ],
    [
      'relative README escape',
      'package/README.md',
      '[outside](../outside.md)',
      'README link escapes'
    ],
    [
      'missing README reference target',
      'package/README.md',
      '[guide][guide]\n\n[guide]: missing.md',
      'README link missing'
    ],
    ['missing HTML image', 'package/README.md', '<img src="missing.png">', 'README link missing'],
    [
      'nested Markdown label',
      'package/README.md',
      '[a [nested] label](missing.md)',
      'README link missing'
    ],
    [
      'nested parentheses in target',
      'package/README.md',
      '[guide](missing(a(b)).md)',
      'README link missing'
    ],
    ['unquoted HTML image', 'package/README.md', '<img src=missing.png>', 'README link missing'],
    [
      'missing percent-encoded link',
      'package/README.md',
      '[guide](missing%20guide.md)',
      'README link missing'
    ],
    [
      'unlisted SPDX expression',
      'package/src/index.ts',
      '// SPDX-License-Identifier: ISC',
      'unlisted SPDX expression'
    ],
    [
      'unlisted non-MIT file',
      'package/src/index.ts',
      '// SPDX-License-Identifier: BSD-3-Clause',
      'unlisted source notice'
    ],
    [
      'NaN CPU output',
      'package/dist/dataframe-analysis/index.js',
      'export function computeVariogramModelFitOnCPU() {return {parameters: new Float32Array(12).fill(NaN), status: new Uint32Array(3)}}',
      'CPU variogram parameter'
    ]
  ])('rejects %s', (_name, path, text, message) => {
    const directory = createFixture();
    if (text) writeFixtureFile(directory, path, text);
    else rmSync(join(directory, path));
    if (message === 'unlisted source notice') {
      writeFixtureFile(directory, 'package/THIRD-PARTY-NOTICES.md', '`MIT` and `BSD-3-Clause`');
      editManifest(directory, manifest => {
        manifest.license = 'MIT AND BSD-3-Clause';
      });
    }
    writeFixtureFile(directory, 'outside.md', 'This exists outside the tarball.');
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain(message);
  });

  it('refuses an empty exports map', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.exports = {};
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('missing required public entry "."');
  });

  it('allows packed destinations with spaces and parentheses and ignores links in code', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.files.push('guide (one).md', 'a(b(c)).md');
    });
    writeFixtureFile(directory, 'package/guide (one).md', 'guide');
    writeFixtureFile(directory, 'package/a(b(c)).md', 'guide');
    writeFixtureFile(
      directory,
      'package/README.md',
      [
        '[guide](<guide (one).md>)',
        '[guide](a(b(c)).md)',
        '[encoded](guide%20%28one%29.md#heading)',
        '`[code](missing.md)`',
        '```md',
        '[code](missing.md)',
        '```'
      ].join('\n')
    );
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(0);
  });

  it('rejects a declaration target outside the tarball even if the file exists', () => {
    const directory = createFixture();
    writeFixtureFile(directory, 'outside.d.ts', 'export const value: number;');
    editManifest(directory, manifest => {
      manifest.exports['.'].types = join(directory, 'outside.d.ts');
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('types target');
  });

  it('refuses wildcard entries instead of silently skipping them', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.exports['./*'] = {...manifest.exports['.']};
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('unsupported export');
  });

  it('requires every declared peer to be installed in the dependency directory', () => {
    const directory = createFixture();
    rmSync(join(directory, 'node_modules/fixture-peer'), {recursive: true});
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('missing installed dependency fixture-peer');
  });

  it('checks licence files in the tarball rather than their workspace presence', () => {
    const directory = createFixture();
    editManifest(directory, manifest => {
      manifest.files = ['dist', 'THIRD-PARTY-NOTICES.md'];
    });
    const result = runChecker(directory);
    expect(result.exitCode, result.output).toBe(1);
    expect(result.output).toContain('LICENSES/Apache-2.0.txt');
  });
});
