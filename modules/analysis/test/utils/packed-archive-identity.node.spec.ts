// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {afterEach, expect, it} from 'vitest';
import {createPackedConsumerContractFixture} from './packed-consumer-fixture';

const ROOT = fileURLToPath(new URL('../../../../', import.meta.url));
const directories: string[] = [];
function write(directory: string, file: string, contents: string) {
  mkdirSync(dirname(join(directory, file)), {recursive: true});
  writeFileSync(join(directory, file), contents);
}
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

it.each([
  ['unchanged', true],
  ['type-name-collision', true],
  ['vector-metadata', true],
  ['output-first', true],
  ['after-extraction', true],
  ['before-success', true],
  ['after-extraction', false],
  ['before-success', false],
  ['missing-core-entry', false],
  ['missing-geospatial-entry', false],
  ['missing-public-type', false],
  ['foreign-vector', false],
  ['foreign-record', false],
  ['foreign-operation', false],
  ['type-string-impostor', false],
  ['reverse-zonal', false],
  ['reverse-zonal-preparation-only', false],
  ['reverse-zonal-missing-second-chunk', false],
  ['reverse-operation-ports', false],
  ['reverse-foreign-vector', false],
  ['reverse-foreign-record', false],
  ['reverse-foreign-operation', false],
  ['truncated-fit', false],
  ['oversized-fit', false],
  ['truncated-fit-parameters', false],
  ['oversized-fit-parameters', false],
  ['truncated-fit-status', false],
  ['oversized-fit-status', false]
] as const)(
  'TOOL-105 binds the real packed producer to extracted bytes (%s, keep=%s)',
  (mode, keepArtifact) => {
    const directory = mkdtempSync(join(tmpdir(), 'packed-identity-'));
    directories.push(directory);
    const target = {
      types: './dist/index.d.ts',
      import: './dist/index.js',
      require: './dist/index.cjs'
    };
    const contracts = createPackedConsumerContractFixture(mode);
    write(
      directory,
      'package/package.json',
      JSON.stringify({
        name: '@luma-contributors/analysis',
        version: '0.0.0',
        type: 'module',
        license: 'MIT',
        files: ['dist', 'LICENSES', 'THIRD-PARTY-NOTICES.md'],
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
        exports: Object.fromEntries(
          contracts.publicEntryKeys
            .filter(
              key =>
                !(mode === 'missing-core-entry' && key === './analysis-core') &&
                !(mode === 'missing-geospatial-entry' && key === './geospatial-analysis')
            )
            .map(key => [key, target])
        )
      })
    );
    for (const file of [
      'LICENSE',
      'LICENSES/Apache-2.0.txt',
      'LICENSES/BSD-3-Clause.txt',
      'LICENSES/H3-NOTICE.txt'
    ])
      write(directory, `package/${file}`, 'fixture licence');
    write(directory, 'package/README.md', '[licence](LICENSE)');
    write(directory, 'package/THIRD-PARTY-NOTICES.md', '`MIT`');
    const entrySource =
      '// SPDX-License-Identifier: MIT\n// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors\nexport {};';
    write(directory, 'package/src/index.ts', entrySource);
    const sourceMap = JSON.stringify({
      version: 3,
      sources: ['../src/index.ts'],
      sourcesContent: [entrySource],
      mappings: 'AAAA'
    });
    for (const extension of ['js', 'cjs']) {
      write(directory, `package/dist/index.${extension}.map`, sourceMap);
    }
    const fitLayouts: Partial<Record<string, readonly [number, number]>> = {
      'truncated-fit': [8, 2],
      'oversized-fit': [13, 4],
      'truncated-fit-parameters': [8, 3],
      'oversized-fit-parameters': [13, 3],
      'truncated-fit-status': [12, 2],
      'oversized-fit-status': [12, 4]
    };
    const [parameterWordCount, statusWordCount] = fitLayouts[mode] ?? [12, 3];
    const computation = `function computeVariogramModelFitOnCPU() {
      return {parameters: Float32Array.from({length: ${parameterWordCount}},
        (_, index) => [0,0,0,0,0.2,1.5,2.2,0,0,0,0,0][index] ?? 0),
        status: new Uint32Array(${statusWordCount})};
    }`;
    const schemas = `const GPU_LINE_DENSITY_PARAMETER_SCHEMA = {wordLength: 4};
      const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA = {format: 'float32', wordLength: 4};`;
    const names = [
      'computeVariogramModelFitOnCPU',
      'GPU_LINE_DENSITY_PARAMETER_SCHEMA',
      'GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA',
      ...contracts.exportNames
    ];
    write(
      directory,
      'package/dist/index.js',
      `import 'fixture-dependency';\n${computation}\n${schemas}\n${contracts.esmSource}\nexport {${names.join(',')}};`
    );
    write(
      directory,
      'package/dist/index.cjs',
      `require('fixture-dependency');\n${computation}\n${schemas}\n${contracts.commonjsSource}\n${names.map(name => `exports.${name} = ${name};`).join('\n')}`
    );
    write(
      directory,
      'package/dist/index.d.ts',
      `${mode === 'type-name-collision' ? '/// <reference types="./node" resolution-mode="require" />\n' : ''}export declare function computeVariogramModelFitOnCPU(...args: unknown[]): {parameters: Float32Array; status: Uint32Array};
      export declare const GPU_LINE_DENSITY_PARAMETER_SCHEMA: ${mode === 'type-name-collision' ? 'PackedNode.DensitySchema' : '{wordLength: number}'};
      export declare const GPU_LINE_SEGMENTIZE_PARAMETER_SCHEMA: {format: 'float32'; wordLength: number};
      ${contracts.declarationSource}`
    );
    if (mode === 'type-name-collision')
      write(
        directory,
        'package/dist/node.d.ts',
        'declare namespace PackedNode { interface DensitySchema {wordLength: number} }'
      );
    write(
      directory,
      'node_modules/fixture-dependency/package.json',
      '{"name":"fixture-dependency","version":"1.0.0","main":"index.cjs"}'
    );
    for (const name of ['@luma.gl/core', '@luma.gl/graph']) {
      mkdirSync(dirname(join(directory, 'node_modules', name)), {recursive: true});
      symlinkSync(join(ROOT, 'node_modules', name), join(directory, 'node_modules', name), 'dir');
    }
    // Replace one byte, preserving length, after successful extraction or during native loading.
    const tamper = `const fileSystem = require('node:fs'), path = require('node:path');
      const parent = process.env.TMPDIR;
      const consumer = fileSystem.readdirSync(parent).find(name => name.startsWith('analysis-packed-consumer-'));
      const archiveDirectory = path.join(parent, consumer);
      const archive = path.join(archiveDirectory, fileSystem.readdirSync(archiveDirectory).find(name => name.endsWith('.tgz')));
      const bytes = fileSystem.readFileSync(archive); bytes[bytes.length - 1] ^= 1; fileSystem.writeFileSync(archive, bytes);`;
    write(
      directory,
      'node_modules/fixture-dependency/index.cjs',
      mode === 'before-success' ? tamper : ''
    );
    write(
      directory,
      'bin/tar',
      `#!${process.execPath}
      const fileSystem = require('node:fs'), childProcess = require('node:child_process');
      const arguments_ = process.argv.slice(2);
      if (arguments_[0] === '-tzf') {
        const listing = childProcess.spawnSync('/usr/bin/tar', arguments_, {stdio: 'inherit'});
        process.exit(listing.status ?? 1);
      }
      if (arguments_[0] !== '-xzf' || arguments_[1] !== '-') throw new Error('extraction must receive captured bytes on stdin');
      const bytes = fileSystem.readFileSync(0);
      fileSystem.writeFileSync(${JSON.stringify(join(directory, 'extracted.tgz'))}, bytes);
      const result = childProcess.spawnSync('/usr/bin/tar', arguments_, {input: bytes, stdio: ['pipe', 'inherit', 'inherit']});
      if (result.status !== 0) process.exit(result.status ?? 1);
      { ${mode === 'after-extraction' ? tamper : ''} }
    `
    );
    chmodSync(join(directory, 'bin/tar'), 0o755);
    mkdirSync(join(directory, 'temporary'));
    const result = spawnSync(
      process.execPath,
      [
        join(ROOT, 'scripts/check-packed-consumer.mjs'),
        '--package-directory',
        join(directory, 'package'),
        '--dependency-directory',
        join(directory, 'node_modules'),
        ...(keepArtifact ? ['--keep-artifact'] : [])
      ],
      {
        encoding: 'utf8',
        timeout: 30000,
        env: {
          ...process.env,
          NODE_OPTIONS: '--max-old-space-size=256',
          TMPDIR: join(directory, 'temporary'),
          PATH: `${join(directory, 'bin')}:${process.env['PATH']}`
        }
      }
    );
    const output = `${result.stdout}${result.stderr}`;
    const expectedFailures: Partial<Record<string, string>> = {
      'missing-core-entry': 'missing required public entry "./analysis-core"',
      'missing-geospatial-entry': 'missing required public entry "./geospatial-analysis"',
      'missing-public-type': 'GPUNetworkIsochronesOutput',
      'foreign-vector': 'ordered chunks',
      'foreign-record': 'opposite-format kernel was not recorded',
      'foreign-operation': 'is not a GPUContributorOperation',
      'type-string-impostor': 'Missing expected exception',
      'reverse-zonal': 'require-to-import: zonal aggregation declaration',
      'reverse-zonal-preparation-only':
        'require-to-import: zonal accumulation reads chunk 0 and updates sums',
      'reverse-zonal-missing-second-chunk':
        'require-to-import: zonal accumulation reads chunk 1 and updates sums',
      'reverse-operation-ports': 'require-to-import: selected input identity',
      'reverse-foreign-vector': 'require-to-import: ordered chunks',
      'reverse-foreign-record': 'require-to-import: opposite-format kernel was not recorded',
      'reverse-foreign-operation': 'operation-require-to-import is not a GPUContributorOperation',
      'truncated-fit': 'CPU variogram result lengths: expected parameters 12 and status 3',
      'oversized-fit': 'CPU variogram result lengths: expected parameters 12 and status 3',
      'truncated-fit-parameters':
        'CPU variogram result lengths: expected parameters 12 and status 3',
      'oversized-fit-parameters':
        'CPU variogram result lengths: expected parameters 12 and status 3',
      'truncated-fit-status': 'CPU variogram result lengths: expected parameters 12 and status 3',
      'oversized-fit-status': 'CPU variogram result lengths: expected parameters 12 and status 3'
    };
    const expectedFailure = expectedFailures[mode];
    if (expectedFailure) {
      expect(result.status, output).toBe(1);
      expect(output).toContain(expectedFailure);
      if (mode.startsWith('reverse-')) {
        expect(output).toContain('native cross-format import-to-require passed');
        expect(output).not.toContain('native cross-format require-to-import passed');
      }
      return;
    }
    if (mode === 'after-extraction' || mode === 'before-success') {
      expect(result.status, output).toBe(1);
      expect(output).toContain('differs from the extracted tarball');
      expect(output).not.toContain('checked tarball identity:');
      if (keepArtifact) {
        const retained = output.match(/^retained consumer directory: (.+)$/m)?.[1];
        expect(retained).toBeDefined();
        const archiveName = readdirSync(retained!).find(name => name.endsWith('.tgz'));
        expect(archiveName).toBeDefined();
        const replacement = readFileSync(join(retained!, archiveName!));
        const extracted = readFileSync(join(directory, 'extracted.tgz'));
        expect(replacement.byteLength).toBe(extracted.byteLength);
        expect(replacement).not.toEqual(extracted);
      } else {
        expect(output).not.toContain('retained consumer directory:');
        expect(
          readdirSync(join(directory, 'temporary')).filter(name =>
            name.startsWith('analysis-packed-consumer-')
          )
        ).toEqual([]);
      }
      if (mode === 'before-success')
        expect(output).toContain('native NodeNext declaration consumer passed');
      else {
        expect(output).not.toContain('packed README links');
        expect(output).not.toContain('verified packed notices against');
        expect(output).not.toContain('packed declaration files');
        expect(output).not.toContain('packed entry points');
        expect(output).not.toContain('native NodeNext declaration consumer passed');
      }
      return;
    }
    expect(result.status, output).toBe(0);
    expect(output).toContain('verified 11 packed entry points');
    expect(output).toContain(
      'native cross-format vectors, recording and lowering passed (both directions)'
    );
    const identityLine = output.match(/^checked tarball identity: (.+)$/m)?.[1];
    expect(identityLine, output).toBeDefined();
    // Independent hashlib oracle over exactly the bytes the real tar command received.
    const expected = spawnSync(
      '/usr/bin/python3',
      [
        '-c',
        'import hashlib,json,sys; archive=open(sys.argv[1],"rb").read(); print(json.dumps(dict(sha256=hashlib.sha256(archive).hexdigest(),byteLength=len(archive))))',
        join(directory, 'extracted.tgz')
      ],
      {encoding: 'utf8'}
    );
    expect(expected.status, expected.stderr).toBe(0);
    expect(JSON.parse(identityLine!)).toEqual(JSON.parse(expected.stdout));
    const archive = output.match(/^checked tarball: (.+)$/m)?.[1];
    expect(archive).toBeDefined();
    expect(readFileSync(archive!)).toEqual(readFileSync(join(directory, 'extracted.tgz')));
  },
  30000
);
