// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {existsSync, readdirSync, readFileSync, realpathSync, lstatSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';

/** Sample resident bytes for a process and its descendants, using ps RSS in KiB. */
export function readTreeRSS(processId) {
  const records = execFileSync('ps', ['-axo', 'pid=,ppid=,rss='], {encoding: 'utf8'})
    .trim()
    .split('\n')
    .map(line => line.trim().split(/\s+/).map(Number));
  const descendants = new Set([processId]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const [childId, parentId] of records) {
      if (descendants.has(parentId) && !descendants.has(childId)) {
        descendants.add(childId);
        changed = true;
      }
    }
  }
  return records.reduce(
    (total, [childId, , residentKilobytes]) =>
      total + (descendants.has(childId) ? residentKilobytes * 1024 : 0),
    0
  );
}

export const BENCHMARK_PATTERNS = [
  'modules/**/*-bench.spec.{ts,js}',
  'modules/**/*-benchmark.spec.{ts,js}'
];
export const isBenchmark = file => /-(?:bench|benchmark)\.spec\.(?:ts|js)$/.test(file);

const OPT_IN_TEST_JOBS = [
  {
    file: 'modules/analysis/test/gpu-spatial-analysis/delaunay-tessellation/ab/delaunay-ab.spec.ts',
    flag: 'A5_AB',
    value: '1'
  },
  {
    file: 'modules/analysis/test/gpu-spatial-analysis/polygon-overlay/ab/overlay-ab.spec.ts',
    flag: 'A5_AB',
    value: '1'
  },
  {
    file: 'modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/vecchia-float32-gpu-study.spec.ts',
    flag: 'VITE_VECCHIA_FLOAT32_STUDY'
  },
  {
    file: 'modules/lenses/test/computation-lenses/capture-records.spec.ts',
    flag: 'VITE_CAPTURE_LENS_RECORDS',
    value: '1'
  }
];

/** Eligibility mirrors each job's explicit entry gate; ordinary correctness files never opt out. */
export function getOptInTestExclusions(environment = process.env) {
  return OPT_IN_TEST_JOBS.filter(job =>
    job.value ? environment[job.flag] !== job.value : !environment[job.flag]
  ).map(job => job.file);
}

export function getOptInBrowserDefines(environment = process.env) {
  return {'import.meta.env.LUMA_A5_AB': JSON.stringify(environment.A5_AB ?? '')};
}

function readNodeHeapOptions(callerOptions) {
  const tokens = [];
  let token = '';
  let quoted = false;
  for (let index = 0; index < callerOptions.length; index++) {
    const character = callerOptions[index];
    if (character === '"') quoted = !quoted;
    else if (character === '\\' && quoted && index + 1 < callerOptions.length)
      token += callerOptions[++index];
    else if (/\s/.test(character) && !quoted) {
      if (token) tokens.push(token);
      token = '';
    } else token += character;
  }
  if (token) tokens.push(token);
  const heapOptions = [];
  for (let index = 0; index < tokens.length; index++) {
    const separator = tokens[index].indexOf('=');
    const rawName = separator < 0 ? tokens[index] : tokens[index].slice(0, separator);
    const inlineValue = separator < 0 ? undefined : tokens[index].slice(separator + 1);
    const name = rawName.replaceAll('_', '-');
    if (!['--max-old-space-size', '--max-old-space-size-percentage'].includes(name)) continue;
    const value = Number(inlineValue ?? tokens[index + 1]);
    heapOptions.push({name, value});
  }
  return heapOptions;
}

/** Preserve Node's original option text; normalize only for heap-policy decisions. */
export function resolveNodeHeapOptions(callerOptions = '', maximumHeapMegabytes) {
  const heapOptions = readNodeHeapOptions(callerOptions);
  // Callers list NODE_OPTIONS before command-line options and Node applies the last heap option,
  // so only the effective limit meets the ceiling. A lower command-line limit overrides a larger
  // environment value. Percentage and malformed limits are refused wherever they appear.
  const effectiveHeapOption = heapOptions.at(-1);
  for (const heapOption of heapOptions) {
    const {name, value} = heapOption;
    if (
      maximumHeapMegabytes !== undefined &&
      (name.endsWith('-percentage') ||
        !Number.isSafeInteger(value) ||
        value < 1 ||
        (heapOption === effectiveHeapOption && value > maximumHeapMegabytes))
    ) {
      throw new Error(
        `verify allows a V8 heap of at most ${maximumHeapMegabytes} MiB; use an absolute heap limit in NODE_OPTIONS`
      );
    }
  }
  return heapOptions.length ? callerOptions : `${callerOptions} --max-old-space-size=8192`.trim();
}

/** Keep only a validated numeric heap argument when isolating a worker from loader options. */
export function getIsolatedNodeHeapArguments(callerOptions = '') {
  const validatedOptions = resolveNodeHeapOptions(callerOptions, 8192);
  const heap = readNodeHeapOptions(validatedOptions).at(-1);
  return [`--max-old-space-size=${heap.value}`];
}

/** Forward the effective wrapper heap, retaining environment options and CLI heap precedence. */
export function getChildNodeOptions(
  environmentOptions = process.env.NODE_OPTIONS ?? '',
  executableArguments = process.execArgv
) {
  const callerOptions = [
    environmentOptions,
    ...executableArguments.map(argument => JSON.stringify(argument))
  ].join(' ');
  return [environmentOptions, ...getIsolatedNodeHeapArguments(callerOptions)]
    .filter(Boolean)
    .join(' ');
}

/** Same exclusions as vitest.config.ts, without starting Vite or a browser. */
export function listTestFiles(rootDirectory, kind = 'headless', environment = process.env) {
  const exclusionFile = join(rootDirectory, 'test/kepler-compute-example-specs.json');
  const excluded = [
    ...getOptInTestExclusions(environment),
    ...(environment.LUMA_CONTRIBUTORS_RUN_EXCLUDED_SPECS === 'true' || !existsSync(exclusionFile)
      ? []
      : Object.keys(JSON.parse(readFileSync(exclusionFile, 'utf8')).excludedFromRuns ?? {}))
  ];
  function visit(directory) {
    if (!existsSync(join(rootDirectory, directory))) return [];
    return readdirSync(join(rootDirectory, directory), {withFileTypes: true}).flatMap(entry => {
      const path = `${directory}/${entry.name}`;
      if (['node_modules', 'dist', '.git'].includes(entry.name)) return [];
      if (entry.isDirectory()) return visit(path);
      if (
        !/\.spec\.(?:ts|js)$/.test(path) ||
        path.includes('.disabled.') ||
        excluded.includes(path)
      )
        return [];
      if (kind === 'benchmarks') return isBenchmark(path) ? [path] : [];
      if (kind === 'node')
        return path.endsWith('.node.spec.ts') || path.endsWith('.node.spec.js') ? [path] : [];
      return !/\.node\.spec\.(?:ts|js)$/.test(path) && !isBenchmark(path) ? [path] : [];
    });
  }
  return visit('modules').sort();
}

/** A fixed-size queue: at most concurrency transforms/read/write groups are in flight. */
export async function mapWithConcurrency(values, concurrency, callback) {
  if (!Number.isSafeInteger(concurrency) || concurrency < 1)
    throw new Error('concurrency must be a positive integer');
  let nextIndex = 0;
  await Promise.all(
    Array.from({length: Math.min(concurrency, values.length)}, async () => {
      while (nextIndex < values.length) {
        const index = nextIndex++;
        await callback(values[index], index);
      }
    })
  );
}

export const VERIFICATION_STEPS = [
  'lint',
  'check-upstream-first',
  'check-contributor-contract',
  'typecheck',
  'typecheck-tests',
  'build',
  'verify-package-exports',
  'check-packed-consumer',
  'test-node',
  'test-headless'
];

/** Reconcile a fresh Vitest JSON report. Node permits explicit skips; headless needs a pass/file. */
export function reconcileTestReport(rootDirectory, files, report, kind) {
  const label = kind === 'node' ? 'Node' : 'Headless';
  if (report.success === false || report.numFailedTests > 0 || report.numFailedTestSuites > 0)
    throw new Error(`${label} report records a failed run`);
  const actual = new Set();
  const suites = [];
  for (const suite of report.testResults ?? []) {
    const file = relative(rootDirectory, realpathSync(resolve(rootDirectory, suite.name)));
    if (actual.has(file)) throw new Error(`${label} duplicate suite: ${file}`);
    actual.add(file);
    if (suite.status === 'failed')
      throw new Error(`${label} report records a failed suite: ${file}`);
    const assertions = suite.assertionResults ?? [];
    if (!assertions.length) throw new Error(`${label} unexplained empty suite: ${file}`);
    const skipped = [];
    let passed = 0;
    for (const assertion of assertions) {
      if (assertion.status === 'passed') passed++;
      else if (['skipped', 'todo'].includes(assertion.status))
        skipped.push({
          name: [
            ...(assertion.ancestorTitles ?? []),
            assertion.title ?? assertion.fullName ?? 'unnamed'
          ].join(' > '),
          status: assertion.status,
          reason: assertion.failureMessages ?? []
        });
      else throw new Error(`${label} nonpassing assertion in ${file}: ${assertion.status}`);
    }
    if (kind === 'headless' && !passed) throw new Error(`No passing test executed in ${file}`);
    suites.push({file, passed, skipped, status: suite.status ?? 'unknown'});
  }
  const missing = files.filter(file => !actual.has(file));
  const unexpected = [...actual].filter(file => !files.includes(file));
  if (!files.length || missing.length || unexpected.length)
    throw new Error(
      `${label} inventory mismatch: missing [${missing.join(', ')}]; unexpected [${unexpected.join(', ')}]`
    );
  return {
    selected: files,
    suites,
    deviceEvidence: report.deviceEvidence ?? null,
    backendEvidence: report.backendEvidence ?? null
  };
}

const IDENTITY_PACKAGES = [
  '@luma.gl/core',
  '@luma.gl/graph',
  '@luma.gl/shaders',
  '@luma.gl/engine',
  'apache-arrow',
  'typescript',
  'ts-patch',
  '@vis.gl/ts-plugins',
  '@vis.gl/dev-tools',
  'esbuild',
  'vite',
  'vitest',
  '@vitest/browser',
  '@vitest/browser-playwright',
  'playwright',
  'playwright-core',
  '@biomejs/biome',
  'markdown-it',
  'semver'
];

/** Resolve package metadata with the same package-local lookup as the invoking module. */
function readResolvedPackage(rootDirectory, packageName) {
  const require = createRequire(join(rootDirectory, 'package.json'));
  const direct = join(rootDirectory, 'node_modules', packageName, 'package.json');
  if (existsSync(direct))
    return {directory: dirname(direct), metadata: JSON.parse(readFileSync(direct, 'utf8'))};
  try {
    let directory = dirname(require.resolve(packageName));
    while (dirname(directory) !== directory) {
      const file = join(directory, 'package.json');
      if (existsSync(file)) {
        const metadata = JSON.parse(readFileSync(file, 'utf8'));
        if (metadata.name === packageName) return {directory, metadata};
      }
      directory = dirname(directory);
    }
  } catch {
    /* Missing installations stay explicit, never replaced by accepted ranges. */
  }
  return null;
}
function readResolvedVersion(rootDirectory, packageName) {
  return readResolvedPackage(rootDirectory, packageName)?.metadata.version ?? null;
}

/** Peers actually resolved by each package, including package-local installations. */
export function captureResolvedWorkspaceDependencies(rootDirectory) {
  return Object.fromEntries(
    ['analysis', 'lenses', 'kepler']
      .filter(name => existsSync(join(rootDirectory, 'modules', name, 'package.json')))
      .map(name => [
        name,
        Object.fromEntries(
          IDENTITY_PACKAGES.slice(0, 5).map(packageName => [
            packageName,
            readResolvedVersion(join(rootDirectory, 'modules', name), packageName)
          ])
        )
      ])
  );
}

function readExecutableVersion(executable, rootDirectory) {
  try {
    const output = execFileSync(executable, ['--version'], {
      cwd: rootDirectory,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 5000
    }).trim();
    return /^\d+\.\d+\.\d+(?:[-+].*)?$/.test(output) ? output : null;
  } catch {
    return null;
  }
}

function captureBrowserDriver(rootDirectory) {
  const adapter = readResolvedPackage(rootDirectory, '@vitest/browser-playwright');
  const playwright = adapter ? readResolvedPackage(adapter.directory, 'playwright') : null;
  const core = playwright ? readResolvedPackage(playwright.directory, 'playwright-core') : null;
  const browsersFile = core ? join(core.directory, 'browsers.json') : null;
  return {
    playwright: playwright?.metadata.version ?? null,
    playwrightCore: core?.metadata.version ?? null,
    chromiumBuilds:
      browsersFile && existsSync(browsersFile)
        ? (JSON.parse(readFileSync(browsersFile, 'utf8')).browsers?.filter(browser =>
            browser.name.startsWith('chromium')
          ) ?? [])
        : null
  };
}

/** Matches vitest.config.ts's configured launch request; this is not observed adapter evidence. */
export function getVerificationBrowserConfiguration(
  environment = process.env,
  platform = process.platform
) {
  return {
    provider: 'playwright',
    browser: 'chromium',
    channel: 'chromium',
    launchArguments: [
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      ...(environment.CI && platform === 'linux'
        ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']
        : [])
    ],
    observedAdapter: null
  };
}

function listInputFiles(rootDirectory, directory = '') {
  const absolute = join(rootDirectory, directory);
  if (!existsSync(absolute)) return [];
  return readdirSync(absolute, {withFileTypes: true}).flatMap(entry => {
    if (
      ['.git', '.typecheck', 'node_modules', 'dist'].includes(entry.name) ||
      entry.name.endsWith('.tsbuildinfo')
    )
      return [];
    const file = join(directory, entry.name);
    return entry.isDirectory() ? listInputFiles(rootDirectory, file) : [file];
  });
}

/** Hash tested working-tree bytes (including untracked inputs), the lockfile and resolved tools. */
export function captureQualificationIdentity(rootDirectory) {
  let revision = null;
  let files;
  try {
    const checkout = execFileSync('git', ['rev-parse', '--show-toplevel'], {
      cwd: rootDirectory,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim();
    if (realpathSync(checkout) !== realpathSync(rootDirectory))
      throw new Error('not checkout root');
    revision = execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: rootDirectory,
      encoding: 'utf8'
    }).trim();
    files = execFileSync('git', ['ls-files', '-z', '--cached', '--others', '--exclude-standard'], {
      cwd: rootDirectory,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024
    })
      .split('\0')
      .filter(Boolean);
  } catch {
    // Exported trees have no Git inventory. Still include all ordinary inputs.
    files = listInputFiles(rootDirectory);
  }
  // Git ignores are not a test-input policy. Include source/spec/fixture trees regardless of
  // local excludes, including optional Kepler cached Arrow/config inputs when present.
  for (const directory of ['modules', 'examples', 'test', 'scripts', 'website'])
    files.push(...listInputFiles(rootDirectory, directory));
  files = [...new Set(files)]
    .filter(
      file =>
        !file
          .split('/')
          .some(part => ['.git', '.typecheck', 'node_modules', 'dist'].includes(part)) &&
        !file.endsWith('.tsbuildinfo')
    )
    .sort();
  const hash = createHash('sha256');
  for (const file of files) {
    const absolute = join(rootDirectory, file);
    if (!existsSync(absolute)) {
      hash.update(JSON.stringify([file, 'missing']));
      continue;
    }
    // Read symlink targets as bytes too; source symlinks must not hide a working-tree edit.
    if (!lstatSync(absolute).isFile() && !lstatSync(absolute).isSymbolicLink()) continue;
    const contents = readFileSync(absolute);
    hash.update(JSON.stringify([file, contents.byteLength]));
    hash.update(contents);
  }
  const resolvedDependencies = Object.fromEntries(
    IDENTITY_PACKAGES.map(name => [name, readResolvedVersion(rootDirectory, name)])
  );
  const lockfile = join(rootDirectory, 'yarn.lock');
  const lockfileSha256 = existsSync(lockfile)
    ? createHash('sha256').update(readFileSync(lockfile)).digest('hex')
    : null;
  const resolvedWorkspaceDependencies = captureResolvedWorkspaceDependencies(rootDirectory);
  const yarnVersion = readExecutableVersion('yarn', rootDirectory);
  const npmVersion = readExecutableVersion('npm', rootDirectory);
  const browserDriver = captureBrowserDriver(rootDirectory);
  const identity = {
    revision,
    resolvedWorkspaceDependencies,
    yarnVersion,
    npmVersion,
    browserDriver,
    contentSha256: hash.digest('hex'),
    lockfileSha256,
    resolvedDependencies,
    nodeVersion: process.version,
    platform: process.platform,
    architecture: process.arch
  };
  return {
    ...identity,
    identitySha256: createHash('sha256').update(JSON.stringify(identity)).digest('hex')
  };
}

/** Validate every command/chunk and reject malformed nested evidence without blocking builds. */
export function isFullQualificationReceipt(receipt, identity) {
  try {
    if (
      receipt?.schemaVersion !== 1 ||
      typeof receipt.id !== 'string' ||
      !receipt.id ||
      receipt.qualification !== 'full' ||
      receipt.failure ||
      !receipt.finishedAt ||
      !identity?.identitySha256 ||
      receipt.startedIdentity?.identitySha256 !== identity.identitySha256 ||
      receipt.finishedIdentity?.identitySha256 !== identity.identitySha256 ||
      JSON.stringify(receipt.requestedSteps) !== JSON.stringify(VERIFICATION_STEPS) ||
      JSON.stringify(receipt.executedSteps) !== JSON.stringify(VERIFICATION_STEPS) ||
      JSON.stringify(receipt.attemptedSteps) !== JSON.stringify(VERIFICATION_STEPS) ||
      !Array.isArray(receipt.commands) ||
      !receipt.commands.length ||
      !Array.isArray(receipt.inputChanges) ||
      receipt.inputChanges.length ||
      receipt.options?.maxWorkers !== 1
    )
      return false;
    for (const kind of ['node', 'headless']) {
      const inventory = receipt.inventories?.[kind];
      if (
        !Array.isArray(inventory?.selected) ||
        !inventory.selected.length ||
        inventory.selected.some(file => typeof file !== 'string') ||
        new Set(inventory.selected).size !== inventory.selected.length ||
        !Array.isArray(inventory.suites) ||
        inventory.suites.length !== inventory.selected.length ||
        inventory.suites.some(
          suite =>
            !suite ||
            !inventory.selected.includes(suite.file) ||
            suite.status === 'failed' ||
            !Number.isSafeInteger(suite.passed) ||
            suite.passed < 0 ||
            !Array.isArray(suite.skipped) ||
            suite.skipped.some(skip => !['skipped', 'todo'].includes(skip?.status)) ||
            (kind === 'headless' ? suite.passed < 1 : suite.passed < 1 && !suite.skipped.length)
        ) ||
        new Set(inventory.suites.map(suite => suite.file)).size !== inventory.selected.length
      )
        return false;
    }
    const chunks = [...Map.groupBy(receipt.inventories.headless.selected, file => dirname(file))];
    const expectedCommands = [
      ...VERIFICATION_STEPS.slice(0, -1).map(step => ({
        label: step,
        files: step === 'test-node' ? receipt.inventories.node.selected : null
      })),
      ...chunks.map(([directory, files]) => ({label: `test-headless ${directory}`, files}))
    ];
    if (receipt.commands.length !== expectedCommands.length) return false;
    for (const [index, expected] of expectedCommands.entries()) {
      const command = receipt.commands[index];
      if (
        !command ||
        command.label !== expected.label ||
        command.outcome !== 'passed' ||
        command.exitCode !== 0 ||
        !Number.isFinite(command.durationSeconds) ||
        command.durationSeconds < 0 ||
        !Number.isFinite(command.peakRSSBytes) ||
        command.peakRSSBytes < 0 ||
        !Array.isArray(command.arguments) ||
        command.inputIdentityBefore !== identity.identitySha256 ||
        command.inputIdentityAfter !== identity.identitySha256
      )
        return false;
      if (expected.files) {
        if (
          JSON.stringify(command.selectedFiles) !== JSON.stringify(expected.files) ||
          typeof command.reportFile !== 'string' ||
          !command.reportFile.includes(receipt.id) ||
          !command.arguments.includes('--maxWorkers=1') ||
          !command.arguments.includes(`--outputFile.json=${command.reportFile}`)
        )
          return false;
        const kind = expected.label === 'test-node' ? 'node' : 'headless';
        const report = receipt.inventories[kind].reports?.find(
          item => item?.file === command.reportFile
        );
        if (
          report?.outcome !== 'passed' ||
          typeof report.sha256 !== 'string' ||
          !Number.isSafeInteger(report.byteLength) ||
          report.byteLength < 1
        )
          return false;
        if (
          kind === 'headless' &&
          JSON.stringify(
            command.arguments.filter(argument => /\.spec\.(?:ts|js)$/.test(argument))
          ) !== JSON.stringify(expected.files)
        )
          return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}

/** Bind the stage archive bytes to full matching input evidence, leaving independent builds usable. */
export function associateReleaseQualification(receipt, identity, tarball) {
  if (!isFullQualificationReceipt(receipt, identity)) return {kind: 'build-package'};
  return {
    kind: 'full',
    receiptId: receipt.id,
    identitySha256: identity.identitySha256,
    tarballSha256: tarball.sha256,
    tarballByteLength: tarball.byteLength
  };
}
