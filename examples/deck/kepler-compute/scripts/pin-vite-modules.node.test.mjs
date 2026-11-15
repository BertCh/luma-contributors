// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {test} from 'node:test';
import {openKeplerComputePage, discardStaleImportErrors} from './pin-vite-modules.mjs';

const ADDRESS = 'http://localhost/?rows=4096';
const ENTRY_URL = 'http://localhost/src/main.tsx';
const MODULE_URL = 'http://localhost/node_modules/.vite/deps/dependency.js?v=old';
const importFailure = moduleUrl => `Failed to fetch dynamically imported module: ${moduleUrl}`;
function makeResponse(url, status, statusText) {
  return {
    url: () => url,
    status: () => status,
    statusText: () => statusText,
    request: () => ({resourceType: () => 'script'})
  };
}
const OUTDATED = makeResponse(MODULE_URL, 504, 'Outdated Optimize Dep');
const ENTRY = makeResponse(ENTRY_URL, 200, 'OK');

function makePage({
  responses = [],
  readinessError,
  hostError = '',
  onBoot = () => {},
  onBlank = () => {}
} = {}) {
  const page = new EventEmitter();
  page.navigations = [];
  page.blankNavigations = 0;
  page.goto = async url => {
    if (url === 'about:blank') {
      page.blankNavigations++;
      onBlank(page);
      return;
    }
    page.navigations.push(url);
    for (const response of [responses[page.navigations.length - 1]].flat().filter(Boolean)) {
      page.emit('response', response);
    }
    onBoot(page);
  };
  const evaluate = predicate => {
    const previousWindow = globalThis.window;
    globalThis.window = {
      __keplerComputeTest: {
        getState: () => ({selectedCount: 4096, error: hostError}),
        source: () => ({length: 4096})
      }
    };
    try {
      return predicate();
    } finally {
      globalThis.window = previousWindow;
    }
  };
  page.evaluate = async predicate => evaluate(predicate);
  page.waitForFunction = async predicate => {
    if (readinessError) throw readinessError;
    assert(evaluate(predicate), 'boot accepts the caller-supplied source count');
  };
  return page;
}
function observeFailures(page, failures) {
  page.on('pageerror', error => failures.push(error.message));
  page.on('console', message => failures.push(message.text()));
}
function cleanup(failures) {
  const startIndex = failures.length;
  return ({staleImportErrors}) => discardStaleImportErrors(failures, startIndex, staleImportErrors);
}

test('boot accepts a URL with a nondefault sample size and removes listeners', async () => {
  const page = makePage();
  await openKeplerComputePage(page, ADDRESS);
  assert.equal(page.navigations.length, 1);
  for (const event of ['response', 'pageerror', 'console'])
    assert.equal(page.listenerCount(event), 0);
});

test('boot recovers only identified imports, including late entry rejection after a dependency 504', async () => {
  const failures = ['failure before boot'];
  const page = makePage({
    responses: [[ENTRY, OUTDATED]],
    onBoot(page) {
      if (page.navigations.length === 1)
        page.emit('pageerror', new Error(importFailure(MODULE_URL)));
    },
    onBlank(page) {
      page.emit('pageerror', new Error(importFailure(ENTRY_URL)));
    }
  });
  observeFailures(page, failures);
  await openKeplerComputePage(page, ADDRESS, {onAbandonedAttempt: cleanup(failures)});
  assert.equal(page.navigations.length, 2);
  assert.equal(page.blankNavigations, 1);
  page.emit('pageerror', new Error('failure after boot'));
  assert.deepEqual(failures, ['failure before boot', 'failure after boot']);
});

test('the last failed optimizer attempt is never discarded', async () => {
  const page = makePage({
    responses: [OUTDATED, OUTDATED, OUTDATED],
    onBoot(page) {
      page.emit('pageerror', new Error(importFailure(MODULE_URL)));
    }
  });
  const failures = [];
  observeFailures(page, failures);
  await assert.rejects(
    openKeplerComputePage(page, ADDRESS, {
      onAbandonedAttempt: cleanup(failures)
    }),
    /all three boots/
  );
  assert.equal(page.navigations.length, 3);
  assert.equal(page.blankNavigations, 2);
  assert.deepEqual(failures, [importFailure(MODULE_URL)]);
});

test('unrelated HTTP failures are not retried', async () => {
  const page = makePage({responses: [makeResponse(MODULE_URL, 504, 'Gateway Timeout')]});
  await assert.rejects(openKeplerComputePage(page, ADDRESS), /returned HTTP 504/);
  assert.equal(page.navigations.length, 1);
});

test('real page errors from the final boot remain failures', async () => {
  const page = makePage({
    responses: [OUTDATED],
    onBoot(page) {
      page.emit(
        'pageerror',
        new Error(
          page.navigations.length === 1 ? importFailure(MODULE_URL) : 'real final-boot failure'
        )
      );
    }
  });
  const failures = ['earlier failure'];
  observeFailures(page, failures);
  await assert.rejects(
    openKeplerComputePage(page, ADDRESS, {
      onAbandonedAttempt: cleanup(failures)
    }),
    /real final-boot failure/
  );
  assert.deepEqual(failures, ['earlier failure', 'real final-boot failure']);
});

test('a mixed optimizer/GPU failure aborts without discarding any errors', async () => {
  const page = makePage({
    responses: [OUTDATED],
    onBoot(page) {
      page.emit('pageerror', new Error(importFailure(MODULE_URL)));
      page.emit('console', {
        type: () => 'error',
        text: () => 'WebGPU validation error: invalid pipeline'
      });
    }
  });
  const failures = [];
  observeFailures(page, failures);
  let abandoned = false;
  await assert.rejects(
    openKeplerComputePage(page, ADDRESS, {
      onAbandonedAttempt() {
        abandoned = true;
      }
    }),
    /WebGPU validation error/
  );
  assert.equal(abandoned, false);
  assert.equal(page.navigations.length, 1);
  assert.deepEqual(failures, [
    importFailure(MODULE_URL),
    'WebGPU validation error: invalid pipeline'
  ]);
});

test('a host state error outranks an optimizer 504', async () => {
  const page = makePage({responses: [OUTDATED], hostError: 'selection computation failed'});
  await assert.rejects(openKeplerComputePage(page, ADDRESS), /selection computation failed/);
  assert.equal(page.blankNavigations, 0);
});

test('a readiness failure outranks an optimizer 504 even with no host error field', async () => {
  const page = makePage({responses: [OUTDATED], readinessError: new Error('application failed')});
  await assert.rejects(openKeplerComputePage(page, ADDRESS), /application failed/);
  assert.equal(page.blankNavigations, 0);
});

test('a late GPU error while retiring a document aborts without stale-error cleanup', async () => {
  const page = makePage({
    responses: [OUTDATED],
    onBlank(page) {
      page.emit('console', {
        type: () => 'error',
        text: () => 'WebGPU validation error: late buffer failure'
      });
    }
  });
  let abandoned = false;
  await assert.rejects(
    openKeplerComputePage(page, ADDRESS, {
      onAbandonedAttempt() {
        abandoned = true;
      }
    }),
    /late buffer failure/
  );
  assert.equal(abandoned, false);
  assert.equal(page.navigations.length, 1);
});

test('an unrelated dynamic import failure is retained alongside an optimizer 504', async () => {
  const page = makePage({
    responses: [[ENTRY, OUTDATED]],
    onBoot(page) {
      page.emit('pageerror', new Error(importFailure('http://localhost/unrelated.js')));
    }
  });
  await assert.rejects(openKeplerComputePage(page, ADDRESS), /unrelated.js/);
  assert.equal(page.blankNavigations, 0);
});

test('stale-import cleanup preserves other errors and matching messages before its checkpoint', () => {
  const stale = importFailure(MODULE_URL);
  const failures = [stale, 'GPU validation failure', stale, 'real runtime failure'];
  discardStaleImportErrors(failures, 1, [stale]);
  assert.deepEqual(failures, [stale, 'GPU validation failure', 'real runtime failure']);
});
