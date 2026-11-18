// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawn} from 'node:child_process';
import {mkdirSync, readFileSync, realpathSync, rmSync, statfsSync, writeFileSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {getChildNodeOptions, listTestFiles, readTreeRSS} from './verification-policy.mjs';

const ROOT = realpathSync(fileURLToPath(new URL('../', import.meta.url)));
const GIBIBYTE = 1024 ** 3;
const MINIMUM_FREE_DISK_GIBIBYTES = 8;
const MAXIMUM_RSS_GIBIBYTES = 12;
const STEPS = [
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
const HEAVY_STEPS = new Set(
  STEPS.filter(
    step => !['lint', 'check-upstream-first', 'check-contributor-contract'].includes(step)
  )
);
const summaries = [];

function parseOptions(arguments_) {
  const options = {};
  for (let index = 0; index < arguments_.length; index++) {
    const option = arguments_[index];
    if (option === '--help') {
      process.stdout.write(
        `Usage: yarn verify [--from <step> | --only <step>] [--minimum-free-disk-gib <GiB>]\nSteps: ${STEPS.join(', ')}\nDisk floor: 8 GiB; sampled process-tree RSS ceiling: 12 GiB; default V8 heap: 8192 MiB.\n`
      );
      process.exit(0);
    }
    if (!['--from', '--only', '--minimum-free-disk-gib'].includes(option) || !arguments_[index + 1])
      throw new Error(`Invalid option: ${option}`);
    if (options[option]) throw new Error(`Duplicate option: ${option}`);
    options[option] = arguments_[++index];
  }
  if (options['--from'] && options['--only']) throw new Error('Use either --from or --only');
  for (const option of ['--from', '--only']) {
    if (options[option] && !STEPS.includes(options[option]))
      throw new Error(`Unknown step: ${options[option]}`);
  }
  const minimumDisk = Number(options['--minimum-free-disk-gib'] ?? MINIMUM_FREE_DISK_GIBIBYTES);
  if (!Number.isFinite(minimumDisk) || minimumDisk < MINIMUM_FREE_DISK_GIBIBYTES)
    throw new Error('minimum free disk must be at least 8 GiB');
  return {
    steps: options['--only']
      ? [options['--only']]
      : STEPS.slice(options['--from'] ? STEPS.indexOf(options['--from']) : 0),
    minimumDisk
  };
}

function checkFreeDisk(minimumDisk) {
  const information = statfsSync(ROOT);
  const available = (information.bavail * information.bsize) / GIBIBYTE;
  if (available < minimumDisk)
    throw new Error(
      `Free disk ${available.toFixed(2)} GiB is below the ${minimumDisk} GiB floor; no heavy command started`
    );
}

async function runCommand(label, arguments_, environment, checkResult) {
  const started = Date.now();
  process.stdout.write(`\nverify: START ${label}\n`);
  // Yarn prepends its executable shim directory to PATH; npm_execpath may be a shell script.
  const child = spawn('yarn', arguments_, {
    cwd: ROOT,
    stdio: 'inherit',
    env: environment,
    detached: true
  });
  let peakRSS = 0;
  let failure;
  function terminate() {
    if (child.pid) {
      try {
        process.kill(-child.pid, 'SIGKILL');
      } catch {
        /* Already exited. */
      }
    }
  }
  function sample() {
    if (!child.pid) return;
    try {
      peakRSS = Math.max(peakRSS, readTreeRSS(process.pid));
      if (peakRSS > MAXIMUM_RSS_GIBIBYTES * GIBIBYTE) {
        failure = `process-tree RSS exceeded ${MAXIMUM_RSS_GIBIBYTES} GiB`;
        terminate();
      }
    } catch (error) {
      failure = `cannot monitor process-tree RSS: ${error.message}`;
      terminate();
    }
  }
  const interrupt = signal => {
    failure = `interrupted (${signal})`;
    terminate();
  };
  const onInterrupt = () => interrupt('SIGINT');
  const onTerminate = () => interrupt('SIGTERM');
  process.on('SIGINT', onInterrupt);
  process.on('SIGTERM', onTerminate);
  const monitor = setInterval(sample, 1000);
  sample();
  const code = await new Promise(resolveExit => {
    child.once('error', error => {
      failure = error.message;
      resolveExit(1);
    });
    child.once('close', status => resolveExit(status ?? 1));
  });
  clearInterval(monitor);
  process.off('SIGINT', onInterrupt);
  process.off('SIGTERM', onTerminate);
  if (code === 0 && !failure && checkResult) {
    try {
      checkResult();
    } catch (error) {
      failure = error.message;
    }
  }
  const summary = `${label}: ${code === 0 && !failure ? 'PASS' : 'FAIL'} ${(Date.now() - started) / 1000}s; sampled peak child-tree RSS ${(peakRSS / GIBIBYTE).toFixed(2)} GiB`;
  summaries.push(summary);
  process.stdout.write(`verify: ${summary}\n`);
  if (failure || code !== 0) throw new Error(`${label}: ${failure ?? `exit ${code}`}`);
}

function checkReport(files, reportFile) {
  const report = JSON.parse(readFileSync(reportFile, 'utf8'));
  if (report.success === false) throw new Error('Headless report records a failed run');
  const actual = new Set();
  for (const suite of report.testResults ?? []) {
    const file = relative(ROOT, realpathSync(resolve(ROOT, suite.name)));
    actual.add(file);
    if (suite.status === 'failed')
      throw new Error(`Headless report records a failed suite: ${file}`);
    if (!(suite.assertionResults ?? []).some(assertion => assertion.status === 'passed'))
      throw new Error(`No passing test executed in ${file}`);
  }
  const missing = files.filter(file => !actual.has(file));
  const unexpected = [...actual].filter(file => !files.includes(file));
  if (missing.length || unexpected.length || !files.length)
    throw new Error(
      `Headless inventory mismatch: missing [${missing.join(', ')}]; unexpected [${unexpected.join(', ')}]`
    );
}

try {
  const {steps, minimumDisk} = parseOptions(process.argv.slice(2));
  const environment = {
    ...process.env,
    NODE_OPTIONS: getChildNodeOptions()
  };
  delete environment.LUMA_CONTRIBUTORS_BENCHMARKS;
  process.stdout.write(
    `verify: sequential; disk floor ${minimumDisk} GiB; sampled child-tree RSS ceiling ${MAXIMUM_RSS_GIBIBYTES} GiB; NODE_OPTIONS=${environment.NODE_OPTIONS}\n`
  );
  for (const step of steps) {
    if (HEAVY_STEPS.has(step)) checkFreeDisk(minimumDisk);
    if (step !== 'test-headless') {
      await runCommand(
        step,
        [step, ...(step === 'test-node' ? ['--maxWorkers=1'] : [])],
        environment
      );
      continue;
    }
    const files = listTestFiles(ROOT, 'headless', environment);
    if (!files.length) throw new Error('No headless specs found');
    const chunks = Map.groupBy(files, file => dirname(file));
    const reportDirectory = join(ROOT, '.typecheck/verify');
    mkdirSync(reportDirectory, {recursive: true});
    writeFileSync(
      join(reportDirectory, 'headless-files.json'),
      `${JSON.stringify(files, null, 2)}\n`
    );
    let chunkIndex = 0;
    for (const [directory, chunk] of chunks) {
      checkFreeDisk(minimumDisk);
      const reportFile = join(reportDirectory, `headless-${++chunkIndex}.json`);
      rmSync(reportFile, {force: true});
      await runCommand(
        `test-headless ${directory}`,
        [
          'test-headless',
          ...chunk,
          '--maxWorkers=1',
          '--reporter=default',
          '--reporter=json',
          `--outputFile.json=${reportFile}`
        ],
        environment,
        () => checkReport(chunk, reportFile)
      );
    }
    const currentFiles = listTestFiles(ROOT, 'headless', environment);
    if (JSON.stringify(currentFiles) !== JSON.stringify(files))
      throw new Error('Headless files changed during verification; rerun test-headless');
    process.stdout.write(
      `verify: headless inventory matched ${files.length} files in ${chunks.size} directory chunks\n`
    );
  }
  process.stdout.write(
    `\nverify summary (${steps.length === STEPS.length ? 'full' : 'partial'}):\n${summaries.join('\n')}\n`
  );
} catch (error) {
  process.stderr.write(
    `\nverify: FAILED ${error.message}\n${summaries.join('\n')}\nRemaining steps were not run.\n`
  );
  process.exitCode = 1;
}
