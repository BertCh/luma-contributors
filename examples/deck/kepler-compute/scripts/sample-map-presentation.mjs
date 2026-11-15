// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';

function normalizeVisibility(value) {
  return typeof value === 'boolean' ? value : value?.isVisible;
}
/** Check original encodings even when a documented fallback hides their original layer. */
export function assertSamplePresentation(
  actualSaved,
  expectedSaved,
  limitations = [],
  label = 'sample'
) {
  const actual = actualSaved.config.visState;
  const expected = expectedSaved.config.visState;
  assert.deepEqual(
    actual.layers.map(layer => layer.id).sort(),
    [
      ...expected.layers.map(layer => layer.id),
      ...limitations.map(value => value.fallbackLayerId)
    ].sort(),
    `${label}: complete layer identities`
  );
  assert.deepEqual(
    actual.layers
      .filter(layer => expected.layers.some(original => original.id === layer.id))
      .map(layer => layer.id),
    expected.layers.map(layer => layer.id),
    `${label}: original layer order`
  );
  for (const layer of expected.layers) {
    const restored = actual.layers.find(candidate => candidate.id === layer.id);
    assert.ok(restored, `${label}: missing layer ${layer.id}`);
    assert.equal(restored.type, layer.type, `${label}: ${layer.id}.type`);
    const limitation = limitations.find(candidate => candidate.layerId === layer.id);
    const actualConfig = {...restored.config};
    // A missing saved constant color consumes Kepler's module-global color cycle. Its
    // value is unspecified by the sample; all saved colors remain part of this exact check.
    if (!('color' in layer.config)) delete actualConfig.color;
    assert.deepEqual(
      actualConfig,
      {...layer.config, ...(limitation ? {isVisible: false} : {})},
      `${label}: ${layer.id}.config`
    );
    assert.deepEqual(
      restored.visualChannels,
      layer.visualChannels ?? {},
      `${label}: ${layer.id}.visualChannels`
    );
  }
  assert.deepEqual(actual.filters ?? [], expected.filters ?? [], `${label}: complete filters`);
  const actualViews = actual.splitMaps ?? [];
  const expectedViews = expected.splitMaps ?? [];
  assert.equal(actualViews.length, expectedViews.length, `${label}: split view count`);
  expectedViews.forEach((view, index) => {
    const expectedLayers = Object.fromEntries(
      Object.entries(view.layers ?? {}).map(([layerId, visibility]) => [
        layerId,
        normalizeVisibility(visibility)
      ])
    );
    for (const limitation of limitations) {
      if (limitation.layerId in expectedLayers)
        expectedLayers[limitation.fallbackLayerId] = expectedLayers[limitation.layerId];
    }
    assert.deepEqual(
      Object.fromEntries(
        Object.entries(actualViews[index].layers ?? {}).map(([layerId, visibility]) => [
          layerId,
          normalizeVisibility(visibility)
        ])
      ),
      expectedLayers,
      `${label}: view ${index} complete layer visibility`
    );
  });
  // Canvas dimensions depend on the driver's viewport; camera and split camera values do not.
  for (const [key, value] of Object.entries(expectedSaved.config.mapState ?? {})) {
    assert.deepEqual(actualSaved.config.mapState?.[key], value, `${label}: camera.${key}`);
  }
}

export function assertSampleRenderedLayers(
  rendered,
  expectedSaved,
  limitations = [],
  label = 'sample'
) {
  for (const original of expectedSaved.config.visState.layers) {
    const limitation = limitations.find(candidate => candidate.layerId === original.id);
    const layerId = limitation?.fallbackLayerId ?? original.id;
    const views = expectedSaved.config.visState.splitMaps ?? [];
    const visible =
      original.config.isVisible !== false &&
      (!views.length ||
        views.some(view => normalizeVisibility(view.layers?.[original.id]) === true));
    const matches = rendered.filter(
      layer => layer.id === layerId || layer.renderId?.includes(layerId)
    );
    if (visible)
      assert.ok(
        matches.some(layer => layer.visible),
        `${label}: expected visible rendered layer ${layerId}`
      );
    else
      assert.equal(
        matches.some(layer => layer.visible),
        false,
        `${label}: hidden rendered layer ${layerId}`
      );
    if (limitation)
      assert.equal(
        rendered.some(
          layer =>
            (layer.id === original.id || layer.renderId?.includes(original.id)) &&
            !layer.renderId?.includes(limitation.fallbackLayerId) &&
            layer.visible
        ),
        false,
        `${label}: original limited layer must remain hidden`
      );
  }
}

/** Publication finishes before asynchronous integer style widening/quantiles can draw. */
export async function assertSampleRenderedLayersWhenReady(
  readRendered,
  expectedSaved,
  limitations = [],
  label = 'sample',
  {timeout = 180000, interval = 250} = {}
) {
  return waitForSampleRendering(
    readRendered,
    rendered => assertSampleRenderedLayers(rendered, expectedSaved, limitations, label),
    {timeout, interval}
  );
}

/** Every split view must draw its enabled layers and exclude all layers hidden in that view. */
export function assertSampleRenderedViews(
  renderedViews,
  expectedSaved,
  limitations = [],
  label = 'sample'
) {
  const splitMaps = expectedSaved.config.visState.splitMaps ?? [];
  assert.equal(
    renderedViews.length,
    Math.max(1, splitMaps.length),
    `${label}: rendered viewport count`
  );
  for (let index = 0; index < Math.max(1, splitMaps.length); index++) {
    const view = splitMaps.length
      ? renderedViews.find(candidate => candidate.id === `view-${index}`)
      : renderedViews[0];
    assert.ok(view, `${label}: missing rendered view-${index}`);
    assertSampleRenderedLayers(
      view.layers,
      {
        ...expectedSaved,
        config: {
          ...expectedSaved.config,
          visState: {
            ...expectedSaved.config.visState,
            splitMaps: splitMaps.length ? [splitMaps[index]] : []
          }
        }
      },
      limitations,
      `${label} view ${index}`
    );
  }
}

export async function assertSampleRenderedViewsWhenReady(
  readRendered,
  expectedSaved,
  limitations = [],
  label = 'sample',
  options = {}
) {
  return waitForSampleRendering(
    readRendered,
    rendered => assertSampleRenderedViews(rendered, expectedSaved, limitations, label),
    options
  );
}

async function waitForSampleRendering(
  readRendered,
  assertRendered,
  {timeout = 180000, interval = 250} = {}
) {
  const deadline = Date.now() + timeout;
  for (;;) {
    const rendered = await readRendered();
    try {
      assertRendered(rendered);
      return rendered;
    } catch (error) {
      if (Date.now() >= deadline) throw error;
    }
    await new Promise(resolve => setTimeout(resolve, interval));
  }
}
