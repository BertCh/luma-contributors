// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';

test('computation-views and computation-lenses import in Node without a DOM', async () => {
  expect(typeof (globalThis as {document?: unknown}).document).toBe('undefined');
  const views = await import('@luma-contributors/lenses/computation-views');
  const lenses = await import('@luma-contributors/lenses/computation-lenses');
  const publicFunctions = [
    'buildDecodedLensTwin',
    'buildLensTwin',
    'createDefaultViewRegistry',
    'LadderCoordinator',
    'mountLadderLens',
    'defineLadderLensElement',
    'LensController'
  ];
  for (const name of publicFunctions) {
    expect(typeof (views as Record<string, unknown>)[name], name).toBe('function');
  }
  // The retained API is smaller; internal implementations still import without a DOM.
  const implementations: Record<string, unknown> = Object.assign(
    {},
    views,
    ...(await Promise.all([
      import('../../src/computation-views/twin/static-twin'),
      import('../../src/computation-views/twin/description'),
      import('../../src/computation-views/registry/default-registry'),
      import('../../src/computation-views/registry/view-registry'),
      import('../../src/computation-views/coordinator/ladder-coordinator'),
      import('../../src/computation-views/scales/scale-registry'),
      import('../../src/computation-views/scales/scale-validator'),
      import('../../src/computation-views/hosts/url-state'),
      import('../../src/computation-views/hosts/navigation-graph'),
      import('../../src/computation-views/hosts/page-scheduler'),
      import('../../src/computation-views/hosts/device-profile'),
      import('../../src/computation-views/views/formula-card'),
      import('../../src/computation-views/views/datum-receipt'),
      import('../../src/computation-views/views/buffer-strip'),
      import('../../src/computation-views/views/dispatch-view'),
      import('../../src/computation-views/views/skeleton-view'),
      import('../../src/computation-views/views/realized-view'),
      import('../../src/computation-views/views/command-graph-view'),
      import('../../src/computation-views/views/timing-view'),
      import('../../src/computation-views/views/compare-view'),
      import('../../src/computation-views/views/scope-diagram')
    ]))
  );
  for (const name of [
    'buildLensTwin',
    'renderLensTwinHtml',
    'generateLensDescription',
    'createDefaultViewRegistry',
    'registerDefaultViewPack',
    'ViewRegistry',
    'LadderCoordinator',
    'getPageCoordinator',
    'ScaleRegistry',
    'validateViewEncodings',
    'mountLadderLens',
    'defineLadderLensElement',
    'LensController',
    'encodeLensUrlState',
    'navigateLadder',
    'PageScheduler',
    'getAnalysisDevice',
    'mountFormulaCard',
    'mountDatumReceipt',
    'mountBufferStrip',
    'mountDispatchView',
    'mountSkeletonView',
    'mountRealizedView',
    'mountCommandGraphView',
    'mountTimingView',
    'mountCompareView',
    'mountScopeDiagram'
  ]) {
    expect(typeof implementations[name], name).toBe('function');
    expect(typeof (views as Record<string, unknown>)[name], name).toBe(
      publicFunctions.includes(name) ? 'function' : 'undefined'
    );
  }
  for (const name of ['registerLensScenario', 'fromRecordedLensFile', 'toRecordedLensFile']) {
    expect(typeof (lenses as Record<string, unknown>)[name], name).toBe('function');
  }
  const root = await import('@luma.gl/graph');
  expect('buildLensTwin' in root).toBe(false);
  expect(typeof (globalThis as {document?: unknown}).document).toBe('undefined');
});
