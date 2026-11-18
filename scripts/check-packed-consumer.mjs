// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// No install, registry access, build or lifecycle scripts: test precisely the packed bytes.
import {spawnSync} from 'node:child_process';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {tmpdir} from 'node:os';
import {createRequire} from 'node:module';
import {basename, dirname, isAbsolute, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import MarkdownIt from 'markdown-it';
import semver from 'semver';
import typescript from 'typescript';
import {getIsolatedNodeHeapArguments} from './verification-policy.mjs';

const WORKSPACE_DIRECTORY = fileURLToPath(new URL('../', import.meta.url));

function runCommand(command, arguments_, options) {
  const result = spawnSync(command, arguments_, {
    encoding: 'utf8',
    timeout: 60_000,
    maxBuffer: 16 * 1024 * 1024,
    ...options
  });
  if (result.error || result.status !== 0) {
    throw new Error(
      `${command} failed: ${result.error?.message ?? ''}\n${result.stdout ?? ''}${result.stderr ?? ''}`
    );
  }
  return result.stdout;
}

function resolvePackedFile(packageDirectory, target, description) {
  if (typeof target !== 'string' || !target.startsWith('./')) {
    throw new Error(`${description}: expected a relative package target, received ${target}`);
  }
  const path = resolve(packageDirectory, target);
  const offset = relative(packageDirectory, path);
  if (offset.startsWith('..') || isAbsolute(offset)) {
    throw new Error(`${description}: target escapes tarball: ${target}`);
  }
  let actualPath;
  try {
    actualPath = realpathSync(path);
    if (!statSync(actualPath).isFile()) throw new Error('not a file');
  } catch {
    throw new Error(`${description}: missing packed file ${target}`);
  }
  const actualOffset = relative(realpathSync(packageDirectory), actualPath);
  if (actualOffset.startsWith('..') || isAbsolute(actualOffset)) {
    throw new Error(`${description}: target escapes tarball: ${target}`);
  }
  return actualPath;
}

function checkReadmeLinks(packageDirectory) {
  const readme = readFileSync(resolvePackedFile(packageDirectory, './README.md', 'README'), 'utf8');
  const markdown = new MarkdownIt({html: true});
  const destinations = [];
  function collectDestinations(tokens) {
    for (const token of tokens) {
      if (token.type === 'link_open') destinations.push(token.attrGet('href'));
      if (token.type === 'image') destinations.push(token.attrGet('src'));
      if (token.type === 'html_inline' || token.type === 'html_block') {
        for (const match of token.content.matchAll(
          /\b(?:href|src)\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s"'=<>`]+))/gi
        )) {
          destinations.push(markdown.utils.unescapeAll(match[1] ?? match[2] ?? match[3]));
        }
      }
      if (token.children) collectDestinations(token.children);
    }
  }
  collectDestinations(markdown.parse(readme, {}));
  for (const destination of destinations) {
    const href = destination;
    if (/^[a-z][a-z\d+.-]*:/i.test(href)) {
      new URL(href);
      continue;
    }
    const target = fileURLToPath(new URL(href, pathToFileURL(join(packageDirectory, 'README.md'))));
    const offset = relative(packageDirectory, target);
    if (offset.startsWith('..') || isAbsolute(offset)) {
      throw new Error(`README link escapes tarball: ${href}`);
    }
    let actualPath;
    try {
      actualPath = realpathSync(target);
    } catch {
      throw new Error(`README link missing from tarball: ${href}`);
    }
    const actualOffset = relative(realpathSync(packageDirectory), actualPath);
    if (actualOffset.startsWith('..') || isAbsolute(actualOffset)) {
      throw new Error(`README link escapes tarball: ${href}`);
    }
  }
  process.stdout.write(`verified ${destinations.length} packed README links\n`);
}

function collectSourceFiles(directory) {
  return readdirSync(directory).flatMap(name => {
    const path = join(directory, name);
    return statSync(path).isDirectory() ? collectSourceFiles(path) : [path];
  });
}

function checkLicenseInventory(packageDirectory, manifest) {
  for (const file of [
    'LICENSE',
    'LICENSES/Apache-2.0.txt',
    'LICENSES/BSD-3-Clause.txt',
    'LICENSES/H3-NOTICE.txt'
  ]) {
    resolvePackedFile(packageDirectory, `./${file}`, 'licence');
  }
  const inventory = readFileSync(
    resolvePackedFile(packageDirectory, './THIRD-PARTY-NOTICES.md', 'notices'),
    'utf8'
  );
  const sources = collectSourceFiles(join(packageDirectory, 'src'));
  if (!sources.length) throw new Error('zero shipped source files checked for notices');
  for (const source of sources) {
    const text = readFileSync(source, 'utf8');
    const expression = text
      .match(/SPDX-License-Identifier: ([^\r\n]+)/)?.[1]
      ?.replace(/\s*(?:\*\/|-->)\s*$/, '')
      .trim();
    if (!expression && !/\.(?:[cm]?ts|tsx)$/.test(source)) continue;
    if (!expression || !inventory.includes(`\`${expression}\``)) {
      throw new Error(
        `unlisted SPDX expression ${expression}: ${relative(packageDirectory, source)}`
      );
    }
    for (const identifier of expression.split(/\s+(?:AND|OR)\s+/)) {
      if (!manifest.license?.split(/\s+(?:AND|OR)\s+/).includes(identifier)) {
        throw new Error(
          `manifest licence omits ${identifier}: ${relative(packageDirectory, source)}`
        );
      }
    }
    if (expression !== 'MIT') {
      const path = relative(packageDirectory, source).split('\\').join('/');
      const line = inventory.split('\n').find(candidate => candidate.includes(`\`${path}\``));
      if (!line?.includes(`\`${expression}\``)) throw new Error(`unlisted source notice: ${path}`);
      for (const holder of text.matchAll(/SPDX-FileCopyrightText: ([^\r\n]+)/g)) {
        if (!inventory.includes(holder[1]))
          throw new Error(`unlisted copyright holder: ${path}: ${holder[1]}`);
      }
    }
  }
  process.stdout.write(`verified notices for ${sources.length} packed source files\n`);
}

function checkPackedDeclarations(consumerDirectory, manifest) {
  const entries = Object.keys(manifest.exports ?? {});
  if (!entries.length || entries.some(key => key.includes('*'))) {
    throw new Error('zero exports or unsupported export patterns in the native type consumer');
  }
  const imports = entries.map((key, index) => {
    const specifier = key === '.' ? manifest.name : `${manifest.name}/${key.slice(2)}`;
    return `import * as entry${index} from ${JSON.stringify(specifier)}; void entry${index};`;
  });
  const source = readFileSync(new URL('./packed-consumer.mts', import.meta.url), 'utf8');
  const consumerPath = join(consumerDirectory, 'native-consumer.mts');
  writeFileSync(consumerPath, `${source}\n${imports.join('\n')}\n`);
  const program = typescript.createProgram([consumerPath], {
    module: typescript.ModuleKind.NodeNext,
    moduleResolution: typescript.ModuleResolutionKind.NodeNext,
    target: typescript.ScriptTarget.ES2022,
    skipLibCheck: true,
    types: [],
    strict: true,
    noEmit: true
  });
  const diagnostics = typescript.getPreEmitDiagnostics(program);
  if (diagnostics.length) {
    throw new Error(
      `native declaration consumer failed:\n${typescript.formatDiagnostics(diagnostics, {
        getCanonicalFileName: file => file,
        getCurrentDirectory: () => consumerDirectory,
        getNewLine: () => '\n'
      })}`
    );
  }
  process.stdout.write('native NodeNext declaration consumer passed\n');
}

async function checkExportEntries(packageDirectory, manifest) {
  const entries = Object.entries(manifest.exports ?? {});
  if (!entries.length) throw new Error('zero exports checked');
  const require = createRequire(import.meta.url);
  const failures = [];
  let checked = 0;
  for (const [key, target] of entries) {
    const specifier = key === '.' ? manifest.name : `${manifest.name}/${key.slice(2)}`;
    try {
      if (
        (key !== '.' && !key.startsWith('./')) ||
        key.includes('*') ||
        !target ||
        typeof target !== 'object'
      ) {
        throw new Error(
          `unsupported export ${key}; declare explicit types, import and require targets`
        );
      }
      resolvePackedFile(packageDirectory, target.types, `${specifier} types target`);
      resolvePackedFile(packageDirectory, target.import, `${specifier} import target`);
      resolvePackedFile(packageDirectory, target.require, `${specifier} require target`);
      const imported = await import(specifier);
      const required = require(specifier);
      const importedNames = Object.keys(imported).sort();
      const requiredNames = Object.keys(required).sort();
      if (!importedNames.length) throw new Error('zero exported names');
      if (JSON.stringify(importedNames) !== JSON.stringify(requiredNames)) {
        throw new Error(
          `export names differ: ESM ${importedNames.join(', ')}; CommonJS ${requiredNames.join(', ')}`
        );
      }
      checked++;
    } catch (error) {
      failures.push(`${specifier}: ${error.message}`);
    }
  }
  if (!checked || failures.length)
    throw new Error(
      `verified ${checked} of ${entries.length} packed entry points\n${failures.join('\n')}`
    );
  process.stdout.write(`verified ${checked} packed entry points (ESM, CommonJS, types)\n`);
}

async function runDocumentedComputation(manifest) {
  // docs/api/operation-forms-and-cpu-twins.md, "Variogram fitting without a graph".
  const {computeVariogramModelFitOnCPU} = await import(`${manifest.name}/dataframe-analysis`);
  const distances = Float32Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
  const result = computeVariogramModelFitOnCPU(
    {
      distances,
      semivariances: Float32Array.from(
        distances,
        distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
      ),
      pairCounts: Uint32Array.from({length: 20}, (_, index) => 10 + index)
    },
    {models: ['gaussian', 'exponential', 'spherical'], weighting: 'pairs'}
  );
  if (result.status[1] !== 0)
    throw new Error(`CPU variogram status: expected 0, received ${result.status[1]}`);
  const expected = [0.2, 1.5, 2.2, 0];
  for (let index = 0; index < expected.length; index++) {
    const actual = result.parameters[4 + index];
    if (!Number.isFinite(actual) || Math.abs(actual - expected[index]) > 1e-4) {
      throw new Error(
        `CPU variogram parameter ${index}: expected ${expected[index]}, received ${actual}`
      );
    }
  }
  process.stdout.write('documented CPU variogram fit passed\n');
}

function main() {
  const heapArguments = getIsolatedNodeHeapArguments(
    [
      process.env.NODE_OPTIONS ?? '',
      ...process.execArgv.map(argument => JSON.stringify(argument))
    ].join(' ')
  );
  let packageDirectory = join(WORKSPACE_DIRECTORY, 'modules/analysis');
  let dependencyDirectory = join(WORKSPACE_DIRECTORY, 'node_modules');
  let keepArtifact = false;
  for (let index = 2; index < process.argv.length; index++) {
    const argument = process.argv[index];
    if (argument === '--keep-artifact') keepArtifact = true;
    else if (argument === '--package-directory' || argument === '--dependency-directory') {
      const value = process.argv[++index];
      if (!value || value.startsWith('--')) throw new Error(`${argument} requires a directory`);
      if (argument === '--package-directory') packageDirectory = resolve(value);
      else dependencyDirectory = resolve(value);
    } else throw new Error(`unknown argument ${argument}`);
  }
  const temporaryRoot = realpathSync(tmpdir());
  const temporaryOffset = relative(realpathSync(WORKSPACE_DIRECTORY), temporaryRoot);
  if (
    !temporaryOffset ||
    (temporaryOffset.split(sep)[0] !== '..' && !isAbsolute(temporaryOffset))
  ) {
    throw new Error('temporary directory must be outside the workspace');
  }
  const temporaryDirectory = mkdtempSync(join(temporaryRoot, 'analysis-packed-consumer-'));
  try {
    const manifest = JSON.parse(readFileSync(join(packageDirectory, 'package.json'), 'utf8'));
    if (!/^(@[^/]+\/)?[^/]+$/.test(manifest.name) || manifest.name.includes('..'))
      throw new Error('invalid package name');
    const packed = JSON.parse(
      runCommand(
        'npm',
        [
          'pack',
          '--offline',
          '--ignore-scripts',
          '--json',
          '--workspaces=false',
          '--pack-destination',
          temporaryDirectory,
          '--cache',
          join(temporaryDirectory, 'npm-cache')
        ],
        {cwd: packageDirectory}
      )
    );
    if (
      packed.length !== 1 ||
      !packed[0].filename ||
      basename(packed[0].filename) !== packed[0].filename
    ) {
      throw new Error('npm pack did not produce exactly one tarball');
    }
    const tarball = join(temporaryDirectory, packed[0].filename);
    const consumerDirectory = join(temporaryDirectory, 'consumer');
    const installedPackage = join(consumerDirectory, 'node_modules', manifest.name);
    mkdirSync(installedPackage, {recursive: true});
    runCommand('tar', ['-xzf', tarball, '--strip-components=1', '-C', installedPackage]);
    const installedManifest = JSON.parse(
      readFileSync(join(installedPackage, 'package.json'), 'utf8')
    );
    checkReadmeLinks(installedPackage);
    checkLicenseInventory(installedPackage, installedManifest);
    const linkedDependencies = new Set();
    for (const [name, range] of [
      ...Object.entries(installedManifest.dependencies ?? {}),
      ...Object.entries(installedManifest.peerDependencies ?? {})
    ]) {
      if (!/^(@[^/]+\/)?[^/]+$/.test(name) || name.includes('..'))
        throw new Error(`invalid dependency name ${name}`);
      let installed;
      try {
        installed = realpathSync(join(dependencyDirectory, name));
        if (!statSync(installed).isDirectory()) throw new Error('not a directory');
      } catch {
        throw new Error(`missing installed dependency ${name} in ${dependencyDirectory}`);
      }
      const dependency = JSON.parse(readFileSync(join(installed, 'package.json'), 'utf8'));
      if (
        !semver.validRange(range) ||
        !semver.valid(dependency.version) ||
        !semver.satisfies(dependency.version, range)
      ) {
        throw new Error(`${name}@${dependency.version} does not satisfy ${range}`);
      }
      if (linkedDependencies.has(name)) continue;
      const destination = join(consumerDirectory, 'node_modules', name);
      mkdirSync(dirname(destination), {recursive: true});
      symlinkSync(installed, destination, 'dir');
      linkedDependencies.add(name);
    }
    const worker = join(consumerDirectory, 'check-consumer.mjs');
    writeFileSync(
      worker,
      `
      import {readFileSync, realpathSync, statSync} from 'node:fs';
      import {createRequire} from 'node:module';
      import {isAbsolute, join, relative, resolve} from 'node:path';
      ${resolvePackedFile.toString()}
      ${checkExportEntries.toString()}
      ${runDocumentedComputation.toString()}
      const packageDirectory = process.argv[2];
      const manifest = JSON.parse(readFileSync(join(packageDirectory, 'package.json'), 'utf8'));
      await checkExportEntries(packageDirectory, manifest);
      await runDocumentedComputation(manifest);
    `
    );
    process.stdout.write(
      runCommand(process.execPath, [...heapArguments, worker, installedPackage], {
        cwd: consumerDirectory,
        env: {...process.env, NODE_PATH: '', NODE_OPTIONS: ''}
      })
    );
    checkPackedDeclarations(consumerDirectory, installedManifest);
    if (keepArtifact) process.stdout.write(`checked tarball: ${tarball}\n`);
  } finally {
    if (!keepArtifact) rmSync(temporaryDirectory, {recursive: true, force: true});
    else process.stdout.write(`retained consumer directory: ${temporaryDirectory}\n`);
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`packed-consumer check failed: ${error.message}\n`);
  process.exitCode = 1;
}
