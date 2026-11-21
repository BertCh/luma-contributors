// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash, randomUUID} from 'node:crypto';
import {spawn} from 'node:child_process';
import {
  appendFileSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  statfsSync,
  writeFileSync
} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {
  captureQualificationIdentity,
  getChildNodeOptions,
  getVerificationBrowserConfiguration,
  isFullQualificationReceipt,
  listTestFiles,
  readTreeRSS,
  reconcileTestReport,
  VERIFICATION_STEPS
} from './verification-policy.mjs';

const ROOT = realpathSync(fileURLToPath(new URL('../', import.meta.url)));
const GIBIBYTE = 1024 ** 3;
const MINIMUM_FREE_DISK_GIBIBYTES = 8;
const MAXIMUM_RSS_GIBIBYTES = 12;
const STEPS = VERIFICATION_STEPS;
const HEAVY_STEPS = new Set(
  STEPS.filter(
    step => !['lint', 'check-upstream-first', 'check-contributor-contract'].includes(step)
  )
);
const summaries = [];
let receipt;
const reportDirectory = join(ROOT, '.typecheck/verify');
function publishReceipt() {
  mkdirSync(reportDirectory, {recursive: true});
  const contents = `${JSON.stringify(receipt, null, 2)}\n`;
  // Keep a run-specific record as well as the latest pointer. A crash leaves "running", not full.
  for (const file of [
    join(reportDirectory, `${receipt.id}.json`),
    join(reportDirectory, 'latest.json')
  ]) {
    const temporary = `${file}.${randomUUID()}.tmp`;
    try {
      writeFileSync(temporary, contents, {flag: 'wx'});
      renameSync(temporary, file);
    } finally {
      rmSync(temporary, {force: true});
    }
  }
}

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

function checkpointInputs(label) {
  const identity = captureQualificationIdentity(ROOT);
  if (identity.identitySha256 !== receipt.startedIdentity.identitySha256) {
    receipt.inputChanges.push({checkpoint: label, observedAt: new Date().toISOString(), identity});
    publishReceipt();
    if (receipt.requestedSteps.length === STEPS.length)
      throw new Error(`tested inputs changed at ${label}; receipt cannot qualify a release`);
  }
  return identity.identitySha256;
}

async function runCommand(label, arguments_, environment, checkResult, metadata = {}) {
  const started = Date.now();
  const command = {
    label,
    arguments: arguments_,
    startedAt: new Date(started).toISOString(),
    outcome: 'running',
    durationSeconds: null,
    peakRSSBytes: 0,
    ...metadata,
    inputIdentityBefore: checkpointInputs(`before ${label}`)
  };
  command.outputFile = join(
    reportDirectory,
    `${receipt.id}-command-${receipt.commands.length + 1}.log`
  );
  command.deviceBackendOutput = [];
  receipt.commands.push(command);
  publishReceipt();
  process.stdout.write(`\nverify: START ${label}\n`);
  // Yarn prepends its executable shim directory to PATH; npm_execpath may be a shell script.
  const child = spawn('yarn', arguments_, {
    cwd: ROOT,
    stdio: ['inherit', 'pipe', 'pipe'],
    env: environment,
    detached: true
  });
  for (const [stream, output] of [
    [child.stdout, process.stdout],
    [child.stderr, process.stderr]
  ]) {
    stream.on('data', chunk => {
      output.write(chunk);
      appendFileSync(command.outputFile, chunk);
      for (const line of chunk.toString().split('\n')) {
        if (
          /webgpu|adapter|backend|device|chromium|metal|vulkan|swiftshader/i.test(line) &&
          command.deviceBackendOutput.length < 200
        )
          command.deviceBackendOutput.push(line.slice(0, 2000));
      }
    });
  }
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
  if (checkResult) {
    try {
      checkResult();
    } catch (error) {
      failure ??= error.message;
    }
  }
  try {
    command.inputIdentityAfter = checkpointInputs(`after ${label}`);
  } catch (error) {
    failure ??= error.message;
  }
  const summary = `${label}: ${code === 0 && !failure ? 'PASS' : 'FAIL'} ${(Date.now() - started) / 1000}s; sampled peak child-tree RSS ${(peakRSS / GIBIBYTE).toFixed(2)} GiB`;
  Object.assign(command, {
    outcome: code === 0 && !failure ? 'passed' : 'failed',
    exitCode: code,
    failure: failure ?? null,
    durationSeconds: (Date.now() - started) / 1000,
    peakRSSBytes: peakRSS
  });
  publishReceipt();
  summaries.push(summary);
  process.stdout.write(`verify: ${summary}\n`);
  if (failure || code !== 0) throw new Error(`${label}: ${failure ?? `exit ${code}`}`);
}

function checkReport(files, reportFile, kind) {
  const inventory = receipt.inventories[kind];
  const evidence = {file: reportFile, outcome: 'failed'};
  inventory.reports.push(evidence);
  try {
    const contents = readFileSync(reportFile);
    Object.assign(evidence, {
      sha256: createHash('sha256').update(contents).digest('hex'),
      byteLength: contents.byteLength
    });
    const report = JSON.parse(contents.toString('utf8'));
    evidence.summary = {
      success: report.success ?? null,
      failedTests: report.numFailedTests ?? null,
      failedSuites: report.numFailedTestSuites ?? null,
      reportedFiles: (report.testResults ?? []).map(suite => ({
        name: suite.name,
        status: suite.status
      }))
    };
    const result = reconcileTestReport(ROOT, files, report, kind);
    inventory.suites.push(...result.suites);
    Object.assign(evidence, {
      outcome: 'passed',
      deviceEvidence: result.deviceEvidence,
      backendEvidence: result.backendEvidence
    });
  } catch (error) {
    evidence.error = error.message;
    throw error;
  } finally {
    publishReceipt();
  }
}

try {
  const {steps, minimumDisk} = parseOptions(process.argv.slice(2));
  const environment = {
    ...process.env,
    NODE_OPTIONS: getChildNodeOptions()
  };
  delete environment.LUMA_CONTRIBUTORS_BENCHMARKS;
  receipt = {
    schemaVersion: 1,
    id: `run-${randomUUID()}`,
    startedAt: new Date().toISOString(),
    finishedAt: null,
    qualification: 'running',
    startedIdentity: captureQualificationIdentity(ROOT),
    finishedIdentity: null,
    requestedSteps: steps,
    executedSteps: [],
    attemptedSteps: [],
    inputChanges: [],
    commands: [],
    inventories: {},
    options: {
      maxWorkers: 1,
      configuredBrowser: getVerificationBrowserConfiguration(environment),
      minimumFreeDiskGiB: minimumDisk,
      maximumRSSGiB: MAXIMUM_RSS_GIBIBYTES,
      rssSamplingMilliseconds: 1000,
      nodeOptions: environment.NODE_OPTIONS,
      optIns: Object.fromEntries(
        [
          'A5_AB',
          'VITE_VECCHIA_FLOAT32_STUDY',
          'VITE_CAPTURE_LENS_RECORDS',
          'LUMA_CONTRIBUTORS_RUN_EXCLUDED_SPECS'
        ].map(name => [name, environment[name] ?? null])
      )
    },
    deviceEvidence: 'Not reported by Vitest JSON unless supplied by the runner'
  };
  publishReceipt();
  process.stdout.write(
    `verify: sequential; disk floor ${minimumDisk} GiB; sampled child-tree RSS ceiling ${MAXIMUM_RSS_GIBIBYTES} GiB; NODE_OPTIONS=${environment.NODE_OPTIONS}\n`
  );
  for (const step of steps) {
    if (HEAVY_STEPS.has(step)) checkFreeDisk(minimumDisk);
    receipt.attemptedSteps.push(step);
    publishReceipt();
    if (step === 'test-node') {
      const files = listTestFiles(ROOT, 'node', environment);
      const reportFile = join(reportDirectory, `${receipt.id}-node.json`);
      receipt.inventories.node = {selected: files, suites: [], reports: []};
      publishReceipt();
      rmSync(reportFile, {force: true});
      await runCommand(
        step,
        [
          step,
          '--maxWorkers=1',
          '--reporter=default',
          '--reporter=json',
          `--outputFile.json=${reportFile}`
        ],
        environment,
        () => {
          checkReport(files, reportFile, 'node');
          if (JSON.stringify(listTestFiles(ROOT, 'node', environment)) !== JSON.stringify(files))
            throw new Error('Node files changed during verification; rerun test-node');
        },
        {selectedFiles: files, reportFile}
      );
      receipt.executedSteps.push(step);
      publishReceipt();
      continue;
    }
    if (step !== 'test-headless') {
      await runCommand(step, [step], environment);
      receipt.executedSteps.push(step);
      publishReceipt();
      continue;
    }
    const files = listTestFiles(ROOT, 'headless', environment);
    if (!files.length) throw new Error('No headless specs found');
    const chunks = Map.groupBy(files, file => dirname(file));
    receipt.inventories.headless = {selected: files, suites: [], reports: []};
    writeFileSync(
      join(reportDirectory, `${receipt.id}-headless-files.json`),
      `${JSON.stringify(files, null, 2)}\n`
    );
    let chunkIndex = 0;
    for (const [directory, chunk] of chunks) {
      checkFreeDisk(minimumDisk);
      const reportFile = join(reportDirectory, `${receipt.id}-headless-${++chunkIndex}.json`);
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
        () => checkReport(chunk, reportFile, 'headless'),
        {selectedFiles: chunk, reportFile}
      );
    }
    const currentFiles = listTestFiles(ROOT, 'headless', environment);
    if (JSON.stringify(currentFiles) !== JSON.stringify(files))
      throw new Error('Headless files changed during verification; rerun test-headless');
    receipt.executedSteps.push(step);
    publishReceipt();
    process.stdout.write(
      `verify: headless inventory matched ${files.length} files in ${chunks.size} directory chunks\n`
    );
  }
  receipt.finishedIdentity = captureQualificationIdentity(ROOT);
  receipt.finishedAt = new Date().toISOString();
  if (
    steps.length === STEPS.length &&
    receipt.finishedIdentity.identitySha256 !== receipt.startedIdentity.identitySha256
  )
    throw new Error('tested inputs changed during verification; receipt cannot qualify a release');
  receipt.qualification = steps.length === STEPS.length ? 'full' : 'partial';
  if (
    receipt.qualification === 'full' &&
    !isFullQualificationReceipt(receipt, receipt.startedIdentity)
  )
    throw new Error('incomplete full-run evidence');
  publishReceipt();
  process.stdout.write(
    `\nverify summary (${steps.length === STEPS.length ? 'full' : 'partial'}):\n${summaries.join('\n')}\n`
  );
} catch (error) {
  if (receipt) {
    receipt.qualification = 'failed';
    receipt.failure = error.message;
    receipt.finishedAt = new Date().toISOString();
    try {
      receipt.finishedIdentity = captureQualificationIdentity(ROOT);
    } catch (identityError) {
      receipt.identityError = identityError.message;
    }
    publishReceipt();
  }
  process.stderr.write(
    `\nverify: FAILED ${error.message}\n${summaries.join('\n')}\nRemaining steps were not run.\n`
  );
  process.exitCode = 1;
}
