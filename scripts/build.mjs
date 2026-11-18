// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Build every package: ESM + declarations with tspc, then a CommonJS copy of each ESM file.
// CommonJS files are transformed one-to-one (not bundled) so every entry point shares one
// module graph and class identity, matching the luma.gl packaging convention.

import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync, realpathSync} from 'node:fs';
import {readdir, readFile, rm, stat} from 'node:fs/promises';
import {relative, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {transformCommonJSModule} from './build-commonjs.mjs';
import {getTypeScriptProjectInputs} from './typescript-project-inputs.mjs';
import {mapWithConcurrency} from './verification-policy.mjs';
import {
  captureBuildInputs,
  checkBuildRecord,
  validateBuildInputs,
  writeBuildRecord,
  writeFileAtomicAsync
} from './build-record.mjs';

const MODULES = ['analysis', 'lenses', 'kepler'];
const defaultRoot = fileURLToPath(new URL('../', import.meta.url));

/** Build one workspace; clean release snapshots supply their own root and the installed compiler. */
export async function buildPackages(
  repositoryDirectory = defaultRoot,
  compilerBinary = resolve(defaultRoot, 'node_modules/.bin/tspc'),
  environment = {...process.env}
) {
  const repositoryRoot = pathToFileURL(`${resolve(repositoryDirectory)}/`);
  const projects = MODULES.map(moduleName => {
    const moduleDirectory = fileURLToPath(new URL(`modules/${moduleName}/`, repositoryRoot));
    const project = getTypeScriptProjectInputs(resolve(moduleDirectory, 'tsconfig.json'));
    const compiledInputs = captureBuildInputs(moduleDirectory, project);
    const distributionDirectory = new URL(`modules/${moduleName}/dist/`, repositoryRoot);
    return {
      moduleName,
      moduleDirectory,
      project,
      distributionDirectory,
      previousRecord: checkBuildRecord(moduleDirectory, project, compiledInputs),
      compiledInputs,
      expected: getExpectedOutputs(project, moduleDirectory)
    };
  });
  // tsc's incremental cache may say "up to date" after somebody removed dist. Repair missing
  // emitted files without throwing away caches on the normal warm path. CJS is repaired below.
  const missingOutput = projects.some(({distributionDirectory, expected}) =>
    [...expected].some(
      file =>
        !(file.endsWith('.cjs') && expected.has(file.replace(/\.cjs$/, '.js'))) &&
        !existsSync(new URL(file, distributionDirectory))
    )
  );
  const corruptOutput = projects.some(
    ({previousRecord}) => previousRecord.present && !previousRecord.outputsValid
  );

  // Restored/backdated files can change content without making tsc's timestamp planner dirty.
  // Invalidate only affected caches; ordinary edits retain the compiler's incremental path.
  for (const {moduleDirectory, project, previousRecord} of projects) {
    const cache =
      project.options.tsBuildInfoFile ?? resolve(moduleDirectory, 'tsconfig.tsbuildinfo');
    const cacheTime = await stat(cache).then(
      information => information.mtime.getTime(),
      () => undefined
    );
    if (cacheTime === undefined) continue;
    // An unrecorded legacy cache has no trustworthy content baseline for existing outputs.
    if (!previousRecord.present) {
      await rm(cache, {force: true});
      continue;
    }
    for (const file of previousRecord.changedInputs ?? []) {
      const inputTime = await stat(resolve(moduleDirectory, file)).then(
        information => information.mtime.getTime(),
        () => undefined
      );
      if (inputTime === undefined || inputTime <= cacheTime) {
        await rm(cache, {force: true});
        break;
      }
    }
  }

  // Keep composite caches and outputs on warm builds. No package runner/network lookup is needed.
  execFileSync(
    compilerBinary,
    ['-b', 'tsconfig.build.json', ...(missingOutput || corruptOutput ? ['--force'] : [])],
    {
      cwd: fileURLToPath(repositoryRoot),
      env: environment,
      stdio: 'inherit'
    }
  );

  for (const {moduleName, distributionDirectory, expected, previousRecord} of projects) {
    await pruneRemovedOutputs(expected, distributionDirectory);
    const count = await writeCommonJSModules(
      distributionDirectory,
      !previousRecord.present || !previousRecord.outputsValid,
      previousRecord.outputHashes
    );
    process.stdout.write(`${moduleName}: wrote ${count} CommonJS modules\n`);
  }
  // A later package's processing can overlap edits to an earlier package. Check the whole
  // workspace after every asynchronous transform, before replacing any successful-build record.
  const completedProjects = projects.map(project => ({
    ...project,
    currentProject: getTypeScriptProjectInputs(resolve(project.moduleDirectory, 'tsconfig.json'))
  }));
  for (const {moduleDirectory, currentProject, compiledInputs} of completedProjects)
    validateBuildInputs(moduleDirectory, currentProject, compiledInputs);
  for (const {moduleDirectory, currentProject, expected, compiledInputs} of completedProjects)
    writeBuildRecord(moduleDirectory, currentProject, expected, compiledInputs);
}

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await buildPackages();
}

async function writeCommonJSModules(directory, force = false, outputHashes = {}) {
  const files = (await readdir(directory, {recursive: true})).filter(file => file.endsWith('.js'));
  const transformTime = (await stat(new URL('./build-commonjs.mjs', import.meta.url))).mtimeMs;
  let writtenCount = 0;
  // Two transforms keep esbuild input/output buffers bounded independently of package size.
  await mapWithConcurrency(files, 2, async file => {
    const target = new URL(file.replace(/\.js$/, '.cjs'), directory);
    const sourceTime = (await stat(new URL(file, directory))).mtimeMs;
    const targetTime = await stat(target).then(
      information => information.mtimeMs,
      () => 0
    );
    const source = await readFile(new URL(file, directory), 'utf8');
    const previousSourceHash = outputHashes[`dist/${file}`];
    const sourceUnchanged = previousSourceHash
      ? previousSourceHash === createHash('sha256').update(source).digest('hex')
      : targetTime >= sourceTime;
    if (!force && sourceUnchanged && targetTime >= transformTime) return;
    const {code} = await transformCommonJSModule(source, fileURLToPath(new URL(file, directory)));
    await writeFileAtomicAsync(fileURLToPath(target), code);
    writtenCount++;
  });
  return writtenCount;
}

function getExpectedOutputs(project, moduleDirectory) {
  const sourceDirectory = project.options.rootDir ?? resolve(moduleDirectory, 'src');
  const expected = new Set();
  for (const source of project.files) {
    const path = relative(sourceDirectory, source);
    if (path.startsWith('..') || /\.d\.[cm]?ts$/.test(path)) continue;
    if (path.endsWith('.json')) expected.add(path);
    else {
      const base = path.replace(/\.(?:[cm]?[jt]s|[jt]sx)$/, '');
      const javascriptSuffix = /\.[cm][jt]s$/.test(path)
        ? path.match(/\.([cm])[jt]s$/)[1] === 'm'
          ? '.mjs'
          : '.cjs'
        : /\.[jt]sx$/.test(path) && [1, 3].includes(project.options.jsx)
          ? '.jsx'
          : '.js';
      const declarationSuffix =
        javascriptSuffix === '.mjs' ? '.d.mts' : javascriptSuffix === '.cjs' ? '.d.cts' : '.d.ts';
      expected.add(`${base}${javascriptSuffix}`);
      expected.add(`${base}${declarationSuffix}`);
      if (javascriptSuffix === '.js') expected.add(`${base}.cjs`);
      if (project.options.sourceMap) expected.add(`${base}${javascriptSuffix}.map`);
      if (project.options.declarationMap) expected.add(`${base}${declarationSuffix}.map`);
    }
  }
  return expected;
}

async function pruneRemovedOutputs(expected, directory) {
  for (const file of await readdir(directory, {recursive: true})) {
    if (
      file !== '.build-record.json' &&
      /\.(?:[cm]?jsx?|json|d\.[cm]?ts)(?:\.map)?$/.test(file) &&
      !expected.has(file)
    ) {
      await rm(new URL(file, directory));
    }
  }
}
