// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Supervise one shared compiler, and query its live completed cycle instead of trusting old logs.
// Usage: node scripts/typecheck-status.mjs --watch | ['<path regex>']
import {spawn} from 'node:child_process';
import {randomUUID} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {captureTypeScriptProjectInputs} from './typescript-project-inputs.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const LOG = new URL('../.typecheck/watch.log', import.meta.url);
const OWNER = new URL('../.typecheck/watch.owner.json', import.meta.url);
const LOCK = new URL('../.typecheck/watch.lock/', import.meta.url);
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

function isAlive(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function readOwner() {
  try {
    return JSON.parse(fs.readFileSync(OWNER, 'utf8'));
  } catch {
    return null;
  }
}

const captureInputs = () =>
  captureTypeScriptProjectInputs(
    path.join(ROOT, 'tsconfig.check-src.json'),
    ['analysis', 'lenses', 'kepler'].map(name => path.join(ROOT, 'modules', name, 'src'))
  );

async function queryOwner(owner) {
  if (!owner || !isAlive(owner.pid) || !isAlive(owner.compilerPid))
    throw new Error('No live supervised typecheck owner/compiler');
  // Require a heartbeat produced after this request. A reused PID and old metadata cannot
  // certify a dead supervisor, and this works without opening a listening network socket.
  const requestedAt = Date.now();
  while (Date.now() - requestedAt < 3000) {
    const state = readOwner();
    if (state?.token !== owner.token || !isAlive(state.pid) || !isAlive(state.compilerPid))
      throw new Error('Typecheck owner or compiler is no longer live');
    if (state.heartbeatAt > requestedAt) return state;
    await sleep(50);
  }
  throw new Error('Typecheck owner heartbeat is stale');
}

async function watch() {
  fs.mkdirSync(new URL('../.typecheck/', import.meta.url), {recursive: true});
  try {
    fs.mkdirSync(LOCK);
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    const previous = readOwner();
    const live = previous && (isAlive(previous.pid) || isAlive(previous.compilerPid));
    throw new Error(
      live
        ? 'A shared typecheck owner is already running'
        : 'Stale watch.lock; confirm its compiler is stopped and remove the empty lock directory before restarting'
    );
  }
  const state = {
    pid: process.pid,
    token: randomUUID(),
    compilerPid: 0,
    heartbeatAt: 0,
    cycleId: 0,
    startedAt: 0,
    completedAt: 0,
    inputs: null,
    cycle: ''
  };
  const publishOwner = () => {
    state.heartbeatAt = Date.now();
    const temporary = new URL(`../.typecheck/watch.owner.${state.token}.tmp`, import.meta.url);
    fs.writeFileSync(temporary, JSON.stringify(state));
    fs.renameSync(temporary, OWNER);
  };
  publishOwner();
  let compiler;
  const heartbeat = setInterval(publishOwner, 500);
  const release = () => {
    if (readOwner()?.token === state.token) {
      fs.unlinkSync(OWNER);
      fs.rmdirSync(LOCK);
    }
  };
  try {
    fs.writeFileSync(LOG, '');
    compiler = spawn(
      process.execPath,
      [
        path.join(ROOT, 'node_modules/typescript/bin/tsc'),
        '-p',
        'tsconfig.check-src.json',
        '--noEmit',
        '--watch',
        '--preserveWatchOutput',
        '--pretty',
        'false'
      ],
      {cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe']}
    );
    state.compilerPid = compiler.pid;
    publishOwner();
    let pending = '';
    const receive = chunk => {
      fs.appendFileSync(LOG, chunk);
      pending += chunk.toString();
      while (pending.includes('\n')) {
        const newline = pending.indexOf('\n');
        const line = pending.slice(0, newline + 1);
        pending = pending.slice(newline + 1);
        if (/Starting (?:compilation in watch mode|incremental compilation)/.test(line)) {
          state.cycleId++;
          state.startedAt = Date.now();
          // Invalid or partially written configuration cannot certify this cycle, and must
          // not crash the supervisor and orphan its compiler.
          try {
            state.inputs = captureInputs();
          } catch {
            state.inputs = null;
          }
          state.completedAt = 0;
          state.cycle = '';
        }
        state.cycle += line;
        if (/Found \d+ errors?\. Watching for file changes\./.test(line))
          state.completedAt = Date.now();
      }
      publishOwner();
    };
    compiler.stdout.on('data', receive);
    compiler.stderr.on('data', receive);
    const stop = signal => compiler.kill(signal);
    const interrupt = () => stop('SIGINT');
    const terminate = () => stop('SIGTERM');
    process.on('SIGINT', interrupt);
    process.on('SIGTERM', terminate);
    try {
      process.exitCode = await new Promise((resolve, reject) => {
        compiler.once('error', reject);
        compiler.once('exit', code => resolve(code ?? 1));
      });
    } finally {
      process.off('SIGINT', interrupt);
      process.off('SIGTERM', terminate);
    }
  } finally {
    clearInterval(heartbeat);
    compiler?.kill();
    release();
  }
}

async function status() {
  const filter = process.argv[2] ? new RegExp(process.argv[2]) : null;
  const deadline = Date.now() + 10 * 60 * 1000;
  for (;;) {
    const state = await queryOwner(readOwner());
    // Scan after the requested heartbeat, at the point of acceptance. Comparing the full
    // input set captured at cycle start detects deletions as well as additions and edits.
    const currentInputs = captureInputs();
    const current =
      state.inputs?.fingerprint === currentInputs.fingerprint &&
      state.startedAt + 1 >= currentInputs.latestEdit;
    if (state.completedAt && current) {
      const errors = state.cycle.split('\n').filter(line => /error TS\d+/.test(line));
      const selected = filter ? errors.filter(line => filter.test(line)) : errors;
      process.stdout.write(selected.join('\n') + (selected.length ? '\n' : ''));
      process.stdout.write(
        `${selected.length} ${filter ? 'filtered matching' : 'matching'} errors (${errors.length} total), live cycle ${state.cycleId} completed ${new Date(state.completedAt).toISOString()}\n`
      );
      process.exitCode = selected.length ? 1 : 0;
      return;
    }
    if (state.completedAt && !current)
      throw new Error('Completed typecheck cycle is stale relative to current source/config edits');
    if (Date.now() > deadline) {
      process.exitCode = 3;
      throw new Error('Timed out waiting for the shared typecheck to settle');
    }
    await sleep(1000);
  }
}

try {
  if (process.argv[2] === '--watch') await watch();
  else await status();
} catch (error) {
  process.stderr.write(
    `${error.message}. Start the supervisor with: node scripts/typecheck-status.mjs --watch\n`
  );
  process.exitCode ??= 2;
}
