// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Pin loaded development modules while other contributors edit the shared checkout. */
export async function pinViteModules(page) {
  await page.routeWebSocket('**/*', socket => {
    const server = socket.connectToServer();
    server.onMessage(message => {
      if (typeof message === 'string') {
        try {
          const packet = JSON.parse(message);
          if (packet.type === 'update' || packet.type === 'full-reload') return;
        } catch {
          // Non-JSON application messages still pass through.
        }
      }
      socket.send(message);
    });
    socket.onMessage(message => server.send(message));
  });
}

/** Removes only identified stale-import messages, leaving earlier work and other failures intact. */
export function discardStaleImportErrors(failures, startIndex, staleImportErrors) {
  for (const message of staleImportErrors) {
    const index = failures.indexOf(message, startIndex);
    if (index >= 0) failures.splice(index, 1);
  }
}

const APPLICATION_FAILURE_PATTERN =
  /WGSL|Invalid.*(?:Pipeline|Buffer)|validation error|constant attributes|invalid commandbuffer|models is not iterable|drawing.*ResidentScenegraph/i;
const IMPORT_FAILURE_PREFIX = 'Failed to fetch dynamically imported module: ';

/** Boots against the requested sample size, recovering only Vite's stale optimizer responses. */
export async function openKeplerComputePage(
  page,
  url,
  {timeout = 60000, onAbandonedAttempt = () => {}} = {}
) {
  // bootstrap.ts has exactly one dynamic entry. Its import may reject because one of its
  // dependencies returned an optimizer 504 even when main.tsx itself was fetched successfully.
  const entryModuleUrl = new URL('./src/main.tsx', url).href;
  for (let attempt = 0; attempt < 3; attempt++) {
    const outdatedModuleUrls = new Set();
    const pageErrors = [];
    const applicationFailures = [];
    let entryModuleFetched = false;
    let readinessError;
    let reportOutdated;
    const outdated = new Promise(resolve => {
      reportOutdated = resolve;
    });
    const onResponse = response => {
      const moduleUrl = response.url();
      if (moduleUrl === entryModuleUrl && response.status() === 200) entryModuleFetched = true;
      // HMR is pinned above, so the test performs Vite's normal optimizer recovery reload.
      if (response.status() === 504 && response.statusText() === 'Outdated Optimize Dep') {
        outdatedModuleUrls.add(moduleUrl);
        reportOutdated(false);
      } else if (response.status() >= 400 && response.request().resourceType() === 'script') {
        applicationFailures.push(`Module ${moduleUrl} returned HTTP ${response.status()}`);
      }
    };
    const onPageError = error => {
      pageErrors.push(error);
    };
    const onConsole = message => {
      if (message.type() === 'error' && APPLICATION_FAILURE_PATTERN.test(message.text())) {
        applicationFailures.push(message.text());
      }
    };
    const isStaleImport = error => {
      if (!error.message.startsWith(IMPORT_FAILURE_PREFIX)) return false;
      const moduleUrl = error.message.slice(IMPORT_FAILURE_PREFIX.length).trim();
      return (
        outdatedModuleUrls.has(moduleUrl) ||
        (outdatedModuleUrls.size > 0 && entryModuleFetched && moduleUrl === entryModuleUrl)
      );
    };
    const checkApplicationFailures = hostError => {
      if (hostError) throw new Error(hostError);
      if (readinessError) throw readinessError;
      if (applicationFailures.length) throw new Error(applicationFailures.join('\n'));
      const applicationError = pageErrors.find(error => !isStaleImport(error));
      if (applicationError) throw applicationError;
    };
    page.on('response', onResponse);
    page.on('pageerror', onPageError);
    page.on('console', onConsole);
    try {
      await page.goto(url, {waitUntil: 'domcontentloaded'});
      const ready = page
        .waitForFunction(
          () => {
            const acceptance = window.__keplerComputeTest;
            const state = acceptance?.getState();
            if (state?.error) throw new Error(state.error);
            return (
              acceptance &&
              acceptance.source().length > 0 &&
              state.selectedCount === acceptance.source().length
            );
          },
          undefined,
          {timeout}
        )
        .then(
          () => true,
          error => {
            readinessError = error;
            return false;
          }
        );
      const succeeded = await Promise.race([ready, outdated]);
      // An optimizer response must never win over a genuine host or GPU failure. Inspect
      // both the current host state and all observed failures before retiring its document.
      const hostError = await page.evaluate(() => window.__keplerComputeTest?.getState().error);
      checkApplicationFailures(hostError);
      if (succeeded) return;
      if (attempt === 2)
        throw new Error('Vite optimizer returned Outdated Optimize Dep on all three boots');
      // Dispose the old document before classifying late import/page/GPU errors. Only
      // identified stale imports may be removed; unrelated failures abort without cleanup.
      await page.goto('about:blank');
      checkApplicationFailures(hostError);
      await onAbandonedAttempt({
        staleImportErrors: pageErrors.filter(isStaleImport).map(error => error.message)
      });
    } finally {
      page.off('response', onResponse);
      page.off('pageerror', onPageError);
      page.off('console', onConsole);
    }
  }
}
