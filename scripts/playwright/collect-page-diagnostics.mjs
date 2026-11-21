// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Keep JSON-serializable evidence and detach only the listeners installed by this collector. */
export function collectPageDiagnostics(page) {
  const diagnostics = {consoleMessages: [], pageErrors: []};
  const collectConsole = message => {
    diagnostics.consoleMessages.push({
      type: message.type(),
      text: message.text(),
      location: message.location()
    });
  };
  const collectError = error => {
    diagnostics.pageErrors.push({name: error.name, message: error.message, stack: error.stack});
  };
  page.on('console', collectConsole);
  page.on('pageerror', collectError);
  return {
    diagnostics,
    dispose() {
      page.off('console', collectConsole);
      page.off('pageerror', collectError);
    }
  };
}

/** Probe availability without creating a device or changing the application's device lifecycle. */
export async function probeWebGPU(page) {
  return page.evaluate(async () => {
    const gpu = globalThis.navigator?.gpu;
    if (!gpu) return {supported: false, adapter: null};
    const adapter = await gpu.requestAdapter();
    if (!adapter) return {supported: true, adapter: null};
    const information = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
    const info = Object.fromEntries(
      ['vendor', 'architecture', 'device', 'description'].map(name => [
        name,
        information[name] ?? ''
      ])
    );
    // WebIDL limit attributes live on the prototype; object spread silently loses them.
    const limits = {};
    for (const name in adapter.limits) {
      if (typeof adapter.limits[name] === 'number') limits[name] = adapter.limits[name];
    }
    return {
      supported: true,
      adapter: {
        info,
        isFallbackAdapter: Boolean(information.isFallbackAdapter ?? adapter.isFallbackAdapter),
        features: [...adapter.features].sort(),
        limits
      }
    };
  });
}
