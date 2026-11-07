// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getTileViewState,
  planImageTiles,
  validateTiledImageDimensions
} from '../../../examples/deck/kepler-compute/src/native-tiled-capture';
import {
  detectVideoExportSupport,
  interpolateKeyframes,
  planVideoFrames,
  validateVideoPlan
} from '../../../examples/deck/kepler-compute/src/native-video-export';
import {
  buildStandaloneHtml,
  buildStandaloneHtmlWithViewer,
  escapeJsonForScript,
  injectBundleIntoViewer,
  STANDALONE_VIEWER_PLACEHOLDER
} from '../../../examples/deck/kepler-compute/src/standalone-html-export';

test('tiled capture plans tiles and offsets the viewport centre', () => {
  expect(planImageTiles(5000, 3000, 2048, 2048)).toHaveLength(6);
  expect(() => validateTiledImageDimensions(32769, 10)).toThrow();
  expect(() => validateTiledImageDimensions(30000, 30000)).toThrow();
  const view = {longitude: 10, latitude: 20, zoom: 3, bearing: 0, pitch: 0};
  const middle = getTileViewState(view, {x: 400, y: 300, width: 200, height: 200}, 1000, 800);
  expect(middle.longitude).toBeCloseTo(10, 9);
  expect(middle.latitude).toBeCloseTo(20, 9);
  const right = getTileViewState(view, {x: 800, y: 300, width: 200, height: 200}, 1000, 800);
  expect(right.longitude).toBeGreaterThan(10);
  expect(() =>
    getTileViewState({...view, pitch: 30}, {x: 0, y: 0, width: 2, height: 2}, 4, 4)
  ).toThrow();
});

test('video planning, bounds, interpolation and feature detection', () => {
  expect(planVideoFrames(2, 30).frameCount).toBe(61);
  expect(() => validateVideoPlan(121, 30, 640, 480)).toThrow();
  expect(() => validateVideoPlan(100, 60, 640, 480)).toThrow();
  expect(() => validateVideoPlan(5, 30, 3840, 2160)).not.toThrow();
  const a = {longitude: 0, latitude: 0, zoom: 1, bearing: 350, pitch: 0};
  const b = {longitude: 10, latitude: 10, zoom: 3, bearing: 10, pitch: 0};
  const half = interpolateKeyframes(
    [
      {time: 0, viewState: a},
      {time: 2, viewState: b}
    ],
    1
  );
  expect(half.zoom).toBe(2);
  expect(half.bearing).toBeCloseTo(360, 9);
  expect(detectVideoExportSupport({}).supported).toBe(false);
  expect(
    detectVideoExportSupport({MediaRecorder: {isTypeSupported: type => type === 'video/webm'}})
      .mimeType
  ).toBe('video/webm');
});

test('standalone HTML embeds the bundle safely and states runtime limits', () => {
  const json = JSON.stringify({format: 'luma-kepler-sandbox', note: '</script><!--'});
  expect(escapeJsonForScript(json)).not.toContain('</script>');
  const html = buildStandaloneHtml(json, {title: '<b>x</b>'});
  expect(html).not.toContain('<b>x</b>');
  expect(html).toContain('runtime not included');
  expect(() => buildStandaloneHtml(json, {maximumBytes: 10})).toThrow();
});

test('viewer injection replaces the placeholder safely and falls back when absent', async () => {
  const template = `<html><title>t</title><body>${STANDALONE_VIEWER_PLACEHOLDER}</body></html>`;
  const json = JSON.stringify({format: 'luma-kepler-sandbox', note: '</script> $& $1'});
  const html = injectBundleIntoViewer(template, json, {title: 'x'})!;
  expect(html).toContain('id="luma-kepler-bundle"');
  expect(html).toContain('$& $1');
  expect(html).not.toContain(STANDALONE_VIEWER_PLACEHOLDER);
  expect(injectBundleIntoViewer('<html></html>', json)).toBeNull();
  const viaViewer = await buildStandaloneHtmlWithViewer(json, {fetchViewer: async () => template});
  expect(viaViewer.usedViewer).toBe(true);
  const fallback = await buildStandaloneHtmlWithViewer(json, {fetchViewer: async () => '<html/>'});
  expect(fallback.usedViewer).toBe(false);
  expect(fallback.html).toContain('runtime not included');
});
