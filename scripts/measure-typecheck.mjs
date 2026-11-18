// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {
  chmodSync,
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  statfsSync,
  statSync,
  symlinkSync,
  writeFileSync
} from 'node:fs';
import {arch, hostname, platform, release} from 'node:os';
import {dirname, join, relative, resolve} from 'node:path';
import {performance} from 'node:perf_hooks';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {getChildNodeOptions, readTreeRSS} from './verification-policy.mjs';

const ROOT = realpathSync(fileURLToPath(new URL('../', import.meta.url)));
const GIBIBYTE = 1024 ** 3;

function parseOptions(arguments_) {
  const options = {mode: 'check', timingOnly: false};
  for (let index = 0; index < arguments_.length; index++) {
    const argument = arguments_[index];
    if (argument === '--help') {
      process.stdout.write(
        'Usage: node scripts/measure-typecheck.mjs --project <tsconfig> [--mode check|declarations|build|packages] [--compiler <compiler.js>] [--timing-only]\nFresh isolated caches and outputs are retained under .typecheck/measurements/. Cold means compiler-cache cold, not OS-cache cold. Build measures the TypeScript phase; packages measures the existing three-package build engine, including CommonJS, in a private source snapshot.\n'
      );
      process.exit(0);
    }
    if (argument === '--timing-only') options.timingOnly = true;
    else if (['--project', '--mode', '--compiler'].includes(argument) && arguments_[index + 1]) {
      options[argument.slice(2)] = arguments_[++index];
    } else throw new Error(`Invalid option: ${argument}`);
  }
  if (!options.project) throw new Error('--project is required');
  if (!['check', 'declarations', 'build', 'packages'].includes(options.mode))
    throw new Error('Invalid --mode');
  return options;
}

function checkFreeDisk() {
  const information = statfsSync(ROOT);
  if ((information.bavail * information.bsize) / GIBIBYTE < 8) {
    throw new Error('Free disk is below the 8 GiB floor; no compiler started');
  }
}

function createRunDirectory() {
  const typecheckDirectory = join(ROOT, '.typecheck');
  mkdirSync(typecheckDirectory, {recursive: true});
  if (realpathSync(typecheckDirectory) !== typecheckDirectory) {
    throw new Error('.typecheck must not be a symlink');
  }
  const parent = join(typecheckDirectory, 'measurements');
  mkdirSync(parent, {recursive: true});
  if (realpathSync(parent) !== parent) throw new Error('measurements must not be a symlink');
  return mkdtempSync(join(parent, 'run-'));
}

// Discover source/configuration inputs without constructing a semantic TypeScript program.
// Reparse roots and resolve imports on each capture so additions and dependency changes count.
function captureInputFingerprint(configuration) {
  const entries = new Map();
  const contents = new Map();
  const visitedProjects = new Set();
  function readTrackedFile(file) {
    const absolute = resolve(file);
    if (contents.has(absolute)) return contents.get(absolute);
    if (!typescript.sys.fileExists(absolute)) return undefined;
    const bytes = readFileSync(absolute);
    const information = statSync(absolute);
    entries.set(absolute, [
      information.mtimeMs,
      bytes.length,
      createHash('sha256').update(bytes).digest('hex')
    ]);
    const source = bytes.toString('utf8');
    contents.set(absolute, source);
    return source;
  }
  const host = {...typescript.sys, readFile: readTrackedFile};
  function visitProject(project) {
    const original = realpathSync(project);
    if (visitedProjects.has(original)) return;
    visitedProjects.add(original);
    readTrackedFile(join(dirname(original), 'package.json'));
    const parsed = typescript.getParsedCommandLineOfConfigFile(
      original,
      {},
      {
        ...host,
        onUnRecoverableConfigFileDiagnostic: diagnostic => {
          throw new Error(typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
        }
      }
    );
    for (const reference of parsed.projectReferences ?? []) {
      visitProject(typescript.resolveProjectReferencePath(reference));
    }
    const moduleCache = typescript.createModuleResolutionCache(ROOT, file => file, parsed.options);
    const queue = [...parsed.fileNames];
    const visitedFiles = new Set();
    const libraryDirectory = dirname(typescript.getDefaultLibFilePath(parsed.options));
    queue.push(
      ...(parsed.options.lib ?? [typescript.getDefaultLibFileName(parsed.options)]).map(file =>
        join(libraryDirectory, file)
      )
    );
    function addTypeReference(name, containingFile) {
      const reference = typescript.resolveTypeReferenceDirective(
        name,
        containingFile,
        parsed.options,
        host
      ).resolvedTypeReferenceDirective;
      if (reference) queue.push(reference.resolvedFileName);
    }
    for (const name of typescript.getAutomaticTypeDirectiveNames(parsed.options, host))
      addTypeReference(name, original);
    for (let index = 0; index < queue.length; index++) {
      const file = resolve(queue[index]);
      if (visitedFiles.has(file)) continue;
      visitedFiles.add(file);
      const source = readTrackedFile(file);
      if (source === undefined || !/\.(?:[cm]?[jt]s|[jt]sx)$/.test(file)) continue;
      const dependencies = typescript.preProcessFile(source, true, true);
      const moduleSpecifiers = new Set(
        dependencies.importedFiles.map(imported => imported.fileName)
      );
      // preProcessFile omits JSDoc and automatic JSX imports. Parse files that can carry
      // those dependencies, using the compiler's own JSX option/pragma selection.
      const hasJSXRuntime =
        !/\.d\.[cm]?ts$/.test(file) &&
        (parsed.options.jsx === typescript.JsxEmit.ReactJSX ||
          parsed.options.jsx === typescript.JsxEmit.ReactJSXDev ||
          parsed.options.jsxImportSource ||
          source.includes('@jsx'));
      if (
        hasJSXRuntime ||
        source.includes('@import') ||
        (/\.[cm]?jsx?$/.test(file) && source.includes('/**'))
      ) {
        const sourceFile = typescript.createSourceFile(
          file,
          source,
          {
            languageVersion: typescript.ScriptTarget.Latest,
            jsDocParsingMode: typescript.JSDocParsingMode.ParseAll
          },
          true
        );
        function visitImports(node) {
          if (
            typescript.isJSDocImportTag(node) &&
            typescript.isStringLiteralLike(node.moduleSpecifier)
          ) {
            moduleSpecifiers.add(node.moduleSpecifier.text);
          }
          if (
            typescript.isImportTypeNode(node) &&
            typescript.isLiteralTypeNode(node.argument) &&
            typescript.isStringLiteralLike(node.argument.literal)
          ) {
            moduleSpecifiers.add(node.argument.literal.text);
          }
          for (const documentation of node.jsDoc ?? []) visitImports(documentation);
          typescript.forEachChild(node, visitImports);
        }
        visitImports(sourceFile);
        if (hasJSXRuntime) {
          const runtime = typescript.getJSXRuntimeImport(
            typescript.getJSXImplicitImportBase(parsed.options, sourceFile),
            parsed.options
          );
          if (runtime) moduleSpecifiers.add(runtime);
        }
      }
      for (const moduleSpecifier of moduleSpecifiers) {
        const resolved = typescript.resolveModuleName(
          moduleSpecifier,
          file,
          parsed.options,
          host,
          moduleCache
        ).resolvedModule;
        if (resolved) queue.push(resolved.resolvedFileName);
      }
      for (const reference of dependencies.referencedFiles)
        queue.push(resolve(dirname(file), reference.fileName));
      for (const reference of dependencies.typeReferenceDirectives)
        addTypeReference(reference.fileName, file);
      for (const reference of dependencies.libReferenceDirectives)
        queue.push(join(libraryDirectory, `lib.${reference.fileName.toLowerCase()}.d.ts`));
    }
  }
  visitProject(configuration);
  return createHash('sha256')
    .update(JSON.stringify([...entries].sort(([left], [right]) => left.localeCompare(right))))
    .digest('hex');
}

function resolveProjectTypeLookup(options, configuration, fallbackRoot) {
  const containingFile = join(dirname(configuration), '__inferred type names__.ts');
  const typeRoots = [...(typescript.getEffectiveTypeRoots(options, typescript.sys) ?? [])];
  // Explicit typeRoots disable secondary lookup for inferred types. Preserve the original
  // selected files too, including directly typed packages outside @types.
  const types = typescript.getAutomaticTypeDirectiveNames(options, typescript.sys).map(name => {
    const reference = typescript.resolveTypeReferenceDirective(
      name,
      containingFile,
      options,
      typescript.sys
    ).resolvedTypeReferenceDirective;
    if (!reference)
      throw new Error(`Cannot resolve original ambient type ${name} in ${configuration}`);
    return reference.resolvedFileName;
  });
  // TypeScript skips even an absolute type entry when every type-root directory is absent.
  // An empty private root enables those entries without adding named declaration candidates.
  if (types.length && !typeRoots.some(directory => typescript.sys.directoryExists(directory))) {
    mkdirSync(fallbackRoot, {recursive: true});
    typeRoots.push(fallbackRoot);
  }
  return {typeRoots, types};
}

function validatePackageWorkspace(workspace) {
  const root = realpathSync(workspace);
  const visited = new Set();
  function requirePrivatePath(path, label) {
    const absolute = resolve(path);
    // Resolve the nearest existing ancestor too, so an output under the linked node_modules
    // directory cannot pass a lexical containment check and write into installed dependencies.
    let ancestor = absolute;
    while (!existsSync(ancestor) && dirname(ancestor) !== ancestor) ancestor = dirname(ancestor);
    const canonical = resolve(realpathSync(ancestor), relative(ancestor, absolute));
    if (canonical !== root && !canonical.startsWith(`${root}/`)) {
      throw new Error(`${label} is outside private package workspace: ${absolute}`);
    }
  }
  function visit(configuration) {
    requirePrivatePath(configuration, 'referenced project');
    const original = realpathSync(configuration);
    if (visited.has(original)) return;
    visited.add(original);
    const parsed = typescript.getParsedCommandLineOfConfigFile(
      original,
      {},
      {
        ...typescript.sys,
        onUnRecoverableConfigFileDiagnostic: diagnostic => {
          throw new Error(typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
        }
      }
    );
    for (const property of ['outDir', 'declarationDir', 'outFile', 'rootDir']) {
      if (parsed.options[property]) requirePrivatePath(parsed.options[property], property);
    }
    // Check both the compiler's resolved cache and the build engine's legacy cache fallback
    // before the engine can invalidate either one, even if no emit would otherwise occur.
    const cache = typescript.getTsBuildInfoEmitOutputFilePath({...parsed.options, tscBuild: true});
    if (cache) requirePrivatePath(cache, 'tsBuildInfoFile');
    requirePrivatePath(
      parsed.options.tsBuildInfoFile ?? join(dirname(original), 'tsconfig.tsbuildinfo'),
      'build cache'
    );
    for (const file of parsed.fileNames) {
      requirePrivatePath(file, 'source root');
      for (const output of typescript.getOutputFileNames(
        parsed,
        file,
        !typescript.sys.useCaseSensitiveFileNames
      ))
        requirePrivatePath(output, 'compiler output');
    }
    for (const reference of parsed.projectReferences ?? [])
      visit(typescript.resolveProjectReferencePath(reference));
  }
  visit(join(root, 'tsconfig.build.json'));
}

function preparePackageWorkspace(project, runDirectory, compiler) {
  if (project !== join(ROOT, 'tsconfig.build.json')) {
    throw new Error('--mode packages requires --project tsconfig.build.json');
  }
  const workspace = join(runDirectory, 'workspace');
  mkdirSync(workspace);
  for (const file of ['package.json', 'tsconfig.json', 'tsconfig.build.json']) {
    cpSync(join(ROOT, file), join(workspace, file));
  }
  for (const packageName of ['analysis', 'lenses', 'kepler']) {
    const target = join(workspace, 'modules', packageName);
    mkdirSync(target, {recursive: true});
    for (const file of ['package.json', 'tsconfig.json', 'src']) {
      cpSync(join(ROOT, 'modules', packageName, file), join(target, file), {
        recursive: true,
        dereference: true
      });
    }
    const original = join(ROOT, 'modules', packageName, 'tsconfig.json');
    const parsed = typescript.getParsedCommandLineOfConfigFile(
      original,
      {},
      {
        ...typescript.sys,
        onUnRecoverableConfigFileDiagnostic: diagnostic => {
          throw new Error(typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
        }
      }
    );
    const configuration = typescript.readConfigFile(original, typescript.sys.readFile);
    if (configuration.error)
      throw new Error(
        typescript.flattenDiagnosticMessageText(configuration.error.messageText, '\n')
      );
    writeFileSync(
      join(target, 'tsconfig.json'),
      `${JSON.stringify(
        {
          ...configuration.config,
          compilerOptions: {
            ...configuration.config.compilerOptions,
            ...resolveProjectTypeLookup(
              parsed.options,
              original,
              join(target, '.typecheck-type-roots')
            )
          }
        },
        null,
        2
      )}\n`
    );
  }
  symlinkSync(join(ROOT, 'node_modules'), join(workspace, 'node_modules'), 'dir');
  validatePackageWorkspace(workspace);
  // The real build engine invokes an executable compiler. Preserve its argument list and add
  // diagnostics without changing the engine, compiler caches or CommonJS reuse policy.
  const wrapper = join(runDirectory, 'compiler.mjs');
  writeFileSync(
    wrapper,
    `#!${process.execPath}\n// SPDX-License-Identifier: MIT\n// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors\nimport {spawnSync} from 'node:child_process';\nconst result = spawnSync(${JSON.stringify(process.execPath)}, [${JSON.stringify(compiler)}, ...process.argv.slice(2), '--extendedDiagnostics'], {stdio: 'inherit', env: process.env});\nif (result.error) throw result.error;\nprocess.exit(result.status ?? 1);\n`
  );
  chmodSync(wrapper, 0o755);
  return join(workspace, 'tsconfig.build.json');
}

// Extend the original configs so include paths, aliases and inherited compiler options retain
// their original bases. Redirect every reference, output and cache into a fresh private tree.
function isolateProject(project, runDirectory, mode) {
  const projects = new Map();
  function visit(configuration) {
    const original = realpathSync(configuration);
    if (projects.has(original)) return projects.get(original);
    const parsed = typescript.getParsedCommandLineOfConfigFile(
      original,
      {},
      {
        ...typescript.sys,
        onUnRecoverableConfigFileDiagnostic: diagnostic => {
          throw new Error(typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n'));
        }
      }
    );
    if (parsed.errors.length) {
      throw new Error(
        typescript.formatDiagnosticsWithColorAndContext(parsed.errors, {
          getCanonicalFileName: file => file,
          getCurrentDirectory: () => ROOT,
          getNewLine: () => '\n'
        })
      );
    }
    if (mode === 'check' && parsed.projectReferences?.length) {
      throw new Error('A referenced project requires --mode declarations or --mode build');
    }
    const index = projects.size;
    const isolated = join(runDirectory, `project-${index}.json`);
    projects.set(original, isolated);
    const compilerOptions = {
      noEmit: mode === 'check',
      rootDir: parsed.options.rootDir ?? dirname(original),
      ...resolveProjectTypeLookup(
        parsed.options,
        original,
        join(runDirectory, `type-roots-${index}`)
      ),
      outDir: join(runDirectory, `output-${index}`),
      declarationDir: join(runDirectory, `output-${index}`),
      emitDeclarationOnly: mode === 'declarations'
    };
    // Preserve the project's incremental policy: enabling it on the monolithic source baseline
    // would make its warm time incomparable with the actual yarn typecheck command.
    if (mode !== 'check' || parsed.options.incremental || parsed.options.composite) {
      compilerOptions.tsBuildInfoFile = join(runDirectory, `cache-${index}.tsbuildinfo`);
    }
    if (mode === 'check') delete compilerOptions.declarationDir;
    else if (!parsed.options.declaration && !parsed.options.composite && mode === 'build') {
      delete compilerOptions.declarationDir;
    }
    if (mode === 'declarations') compilerOptions.declaration = true;
    if (parsed.options.outFile)
      throw new Error('outFile projects are unsupported; no compiler started');
    writeFileSync(
      isolated,
      `${JSON.stringify(
        {
          extends: original,
          compilerOptions,
          files: parsed.fileNames,
          include: [],
          references: (parsed.projectReferences ?? []).map(reference => ({
            path: visit(typescript.resolveProjectReferencePath(reference))
          }))
        },
        null,
        2
      )}\n`
    );
    return isolated;
  }
  return visit(project);
}

async function measure(label, compiler, configuration, options, runDirectory, environment) {
  checkFreeDisk();
  const arguments_ =
    options.mode === 'packages'
      ? [
          fileURLToPath(import.meta.url),
          '--package-worker',
          dirname(configuration),
          join(runDirectory, 'compiler.mjs')
        ]
      : [
          compiler,
          options.mode === 'check' ? '-p' : '-b',
          configuration,
          '--extendedDiagnostics',
          '--pretty',
          'false'
        ];
  const started = performance.now();
  const child = spawn(process.execPath, arguments_, {
    cwd: ROOT,
    env: environment,
    detached: true,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let output = '';
  let peakRSSBytes = options.timingOnly ? null : 0;
  let rssSampleCount = 0;
  let failure;
  const terminate = () => {
    try {
      if (child.pid) process.kill(-child.pid, 'SIGKILL');
    } catch {
      /* Already exited. */
    }
  };
  const interrupt = () => {
    failure = 'interrupted';
    terminate();
  };
  process.on('SIGINT', interrupt);
  process.on('SIGTERM', interrupt);
  function sample() {
    if (!child.pid || options.timingOnly) return;
    try {
      const residentBytes = readTreeRSS(child.pid);
      if (Number.isFinite(residentBytes) && residentBytes > 0) {
        peakRSSBytes = Math.max(peakRSSBytes, residentBytes);
        rssSampleCount++;
      }
      if (peakRSSBytes > 12 * GIBIBYTE) throw new Error('child-tree RSS exceeded 12 GiB');
    } catch (error) {
      failure = `cannot monitor process-tree RSS: ${error.message}`;
      terminate();
    }
  }
  child.stdout.on('data', chunk => {
    output += chunk;
  });
  child.stderr.on('data', chunk => {
    output += chunk;
  });
  const monitor = options.timingOnly ? undefined : setInterval(sample, 1000);
  sample();
  const code = await new Promise(resolveExit => {
    child.once('error', error => {
      failure = error.message;
      resolveExit(1);
    });
    child.once('close', status => resolveExit(status ?? 1));
  });
  clearInterval(monitor);
  process.off('SIGINT', interrupt);
  process.off('SIGTERM', interrupt);
  const seconds = (performance.now() - started) / 1000;
  writeFileSync(join(runDirectory, `${label}.log`), output);
  if (failure || code !== 0) {
    throw new Error(
      `${label} compiler exited with ${code}${failure ? `: ${failure}` : ''}\n${output}`
    );
  }
  if (!options.timingOnly && rssSampleCount === 0) {
    throw new Error(
      `no positive RSS sample observed for ${label} compiler; no comparison published`
    );
  }
  return {
    seconds,
    peakRSSBytes,
    rssSampleCount,
    files: [...output.matchAll(/^Files:\s+(\d+)/gm)].reduce(
      (total, match) => total + Number(match[1]),
      0
    )
  };
}

async function main() {
  const options = parseOptions(process.argv.slice(2));
  const environment = {...process.env, NODE_OPTIONS: getChildNodeOptions()};
  checkFreeDisk();
  if (!options.timingOnly) {
    try {
      readTreeRSS(process.pid);
    } catch (error) {
      throw new Error(
        `cannot monitor process-tree RSS: ${error.message}; use --timing-only for explicitly unmeasured RSS`
      );
    }
  }
  const runDirectory = createRunDirectory();
  process.stdout.write(
    `measure-typecheck: ${relative(ROOT, runDirectory)}; mode=${options.mode}; NODE_OPTIONS=${environment.NODE_OPTIONS}\n`
  );
  const compiler = resolve(
    ROOT,
    options.compiler ??
      (options.mode === 'packages'
        ? 'node_modules/ts-patch/bin/tspc.js'
        : 'node_modules/typescript/bin/tsc')
  );
  let originalConfiguration = resolve(ROOT, options.project);
  let sourceSnapshotFingerprint;
  if (options.mode === 'packages') {
    sourceSnapshotFingerprint = captureInputFingerprint(originalConfiguration);
    const snapshot = preparePackageWorkspace(originalConfiguration, runDirectory, compiler);
    if (captureInputFingerprint(originalConfiguration) !== sourceSnapshotFingerprint) {
      throw new Error('inputs changed while copying the package snapshot; no compiler started');
    }
    originalConfiguration = snapshot;
  }
  const inputFingerprint = captureInputFingerprint(originalConfiguration);
  const configuration =
    options.mode === 'packages'
      ? originalConfiguration
      : isolateProject(originalConfiguration, runDirectory, options.mode);
  const cold = await measure('cold', compiler, configuration, options, runDirectory, environment);
  if (captureInputFingerprint(originalConfiguration) !== inputFingerprint) {
    throw new Error('inputs changed during cold measurement; no comparison published');
  }
  if (!cold.files)
    throw new Error('cold compiler did not report any files; no comparison published');
  const warm = await measure('warm', compiler, configuration, options, runDirectory, environment);
  if (captureInputFingerprint(originalConfiguration) !== inputFingerprint) {
    throw new Error('inputs changed during warm measurement; no comparison published');
  }
  const report = {
    project: options.project,
    mode: options.mode,
    compiler,
    typescript: typescript.version,
    node: process.version,
    machine: `${hostname()} ${platform()} ${release()} ${arch()}`,
    nodeOptions: environment.NODE_OPTIONS,
    sampleIntervalMilliseconds: 1000,
    inputFingerprint,
    sourceSnapshotFingerprint,
    cold,
    warm
  };
  writeFileSync(join(runDirectory, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
  const peak = options.timingOnly
    ? 'unavailable (timing only)'
    : `${(Math.max(cold.peakRSSBytes, warm.peakRSSBytes) / GIBIBYTE).toFixed(3)} GiB`;
  process.stdout.write(
    `| Project | Files (cold program sum) | Cold s | Warm s | Peak child-tree RSS |\n| --- | ---: | ---: | ---: | --- |\n| ${options.project} | ${cold.files} | ${cold.seconds.toFixed(3)} | ${warm.seconds.toFixed(3)} | ${peak} |\nReport: ${relative(ROOT, join(runDirectory, 'report.json'))}\n`
  );
}

try {
  if (process.argv[2] === '--package-worker') {
    const workspace = realpathSync(process.argv[3]);
    const measurementDirectory = realpathSync(join(ROOT, '.typecheck/measurements'));
    if (!workspace.startsWith(`${measurementDirectory}/`) || !workspace.endsWith('/workspace')) {
      throw new Error('package worker requires a private measurement workspace');
    }
    validatePackageWorkspace(workspace);
    const {buildPackages} = await import('./build.mjs');
    await buildPackages(workspace, process.argv[4], process.env);
  } else await main();
} catch (error) {
  process.stderr.write(`measure-typecheck: FAILED ${error.message}\n`);
  process.exitCode = 1;
}
