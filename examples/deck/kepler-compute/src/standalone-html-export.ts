// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Standalone HTML packaging of the portable sandbox bundle (see sandbox-bundle.ts).
 *
 * What is packaged: the complete bundle JSON (state plus base64 Arrow IPC sources, with their sha256
 * digests) embedded in an inert JSON script element, and a small inline loader that checks the
 * format, verifies each source digest, lists the contents and offers the bundle for download.
 * What is NOT packaged: the luma.gl / Kepler WebGPU runtime. No standalone runtime artifact exists
 * in this repository, so the page does not draw the map by itself. If `runtimeModuleUrl` (for example a
 * version-pinned CDN URL for a runtime exposing `mountLumaKeplerBundle(element, bundle)`) is supplied, the
 * loader imports it and calls it; otherwise the page says the runtime is absent. Restore the downloaded
 * bundle in the kepler-compute app with "Load portable bundle" for the full map.
 */
export const MAXIMUM_STANDALONE_HTML_BYTES = 128 * 1024 * 1024;

export type StandaloneHtmlOptions = {
  title?: string;
  /** Optional ES module URL providing mountLumaKeplerBundle(element, bundle). Not fetched at build time. */
  runtimeModuleUrl?: string;
  maximumBytes?: number;
};

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, character => `&#${character.charCodeAt(0)};`);
}

/** Make JSON safe inside a script element: no `</script>`, `<!--`, or line separators. */
export function escapeJsonForScript(json: string): string {
  return json
    .replace(/</g, '\\u003c')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
}

const LOADER = `
const element = document.getElementById('luma-kepler-bundle');
const output = document.getElementById('summary');
const bundle = JSON.parse(element.textContent);
const runtimeUrl = document.documentElement.dataset.runtimeModule;
async function digest(base64) {
  const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0));
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, '0')).join('');
}
function addLine(text) { const item = document.createElement('li'); item.textContent = text; output.appendChild(item); }
if (bundle.format !== 'luma-kepler-sandbox' || bundle.version !== 1) addLine('Unsupported bundle format.');
else {
  for (const source of bundle.sources) {
    let integrity = 'digest not checked (needs a secure context)';
    if (globalThis.crypto?.subtle) integrity = (await digest(source.data)) === source.sha256 ? 'sha256 verified' : 'SHA-256 MISMATCH';
    addLine(source.datasetId + ' revision ' + source.revision + ': ' + source.byteLength.toLocaleString() + ' Arrow bytes, ' + integrity);
  }
  if (runtimeUrl) {
    try { (await import(runtimeUrl)).mountLumaKeplerBundle(document.getElementById('map'), bundle); }
    catch (error) { addLine('Runtime failed to load: ' + error.message); }
  } else addLine('No luma.gl runtime is embedded or referenced; load the bundle in the kepler-compute app to see the map.');
}
document.getElementById('download').onclick = () => {
  const url = URL.createObjectURL(new Blob([element.textContent], {type: 'application/json'}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'kepler-workspace.bundle.json'; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
};
`;

/** Build the document. `bundleJson` is the output of serializeSandboxBundle. */
export function buildStandaloneHtml(
  bundleJson: string,
  options: StandaloneHtmlOptions = {}
): string {
  const title = escapeHtml(options.title ?? 'luma.gl Kepler workspace');
  const runtime = options.runtimeModuleUrl
    ? ` data-runtime-module="${escapeHtml(options.runtimeModuleUrl)}"`
    : '';
  const html = `<!doctype html>
<html lang="en"${runtime}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title>
<style>body{font:14px system-ui,sans-serif;margin:24px;max-width:960px}#map{height:60vh;background:#10151d}</style>
</head>
<body>
<h1>${title}</h1>
<p>Portable luma.gl Kepler workspace. Bundle embedded below; runtime ${options.runtimeModuleUrl ? 'referenced by URL (not inlined)' : 'not included'}.</p>
<div id="map"></div>
<ul id="summary"></ul>
<button id="download">Download bundle JSON</button>
<script type="application/json" id="luma-kepler-bundle">${escapeJsonForScript(bundleJson)}</script>
<script type="module">${LOADER}</script>
</body>
</html>
`;
  // Bound the output: base64 Arrow inflates sources, so the cap is on the final document.
  if (html.length > (options.maximumBytes ?? MAXIMUM_STANDALONE_HTML_BYTES))
    throw new Error('The standalone HTML exceeds the size limit.');
  return html;
}

/** Marker comment in the prebuilt viewer (`yarn build:viewer` -> dist/standalone-viewer.html). */
export const STANDALONE_VIEWER_PLACEHOLDER = '<!--LUMA_KEPLER_BUNDLE_PLACEHOLDER-->';

/** Insert the bundle into a prebuilt self-contained viewer; null when the template is not a viewer. */
export function injectBundleIntoViewer(
  viewerHtml: string,
  bundleJson: string,
  options: StandaloneHtmlOptions = {}
): string | null {
  // The dev server serves the unbuilt template, which references external scripts; only inlined builds qualify.
  if (!viewerHtml.includes(STANDALONE_VIEWER_PLACEHOLDER) || /<script[^>]+\ssrc=/.test(viewerHtml))
    return null;
  const title = escapeHtml(options.title ?? 'luma.gl Kepler workspace');
  const data = `<script type="application/json" id="luma-kepler-bundle">${escapeJsonForScript(bundleJson)}</script>`;
  // Function replacers: bundle text may contain `$&` style replacement patterns.
  const html = viewerHtml
    .replace(STANDALONE_VIEWER_PLACEHOLDER, () => data)
    .replace(/<title>[^<]*<\/title>/, () => `<title>${title}</title>`);
  if (html.length > (options.maximumBytes ?? MAXIMUM_STANDALONE_HTML_BYTES))
    throw new Error('The standalone HTML exceeds the size limit.');
  return html;
}

/**
 * Offline-capable export: fetch the prebuilt viewer and inject the bundle. Falls back to the
 * bundle-only page (no map) with `usedViewer: false` when the viewer asset was not built or served.
 */
export async function buildStandaloneHtmlWithViewer(
  bundleJson: string,
  options: StandaloneHtmlOptions & {
    viewerUrl?: string;
    fetchViewer?: (url: string) => Promise<string | null>;
  } = {}
): Promise<{html: string; usedViewer: boolean}> {
  const viewerUrl = options.viewerUrl ?? 'standalone-viewer.html';
  const fetchViewer =
    options.fetchViewer ??
    (async (url: string) => {
      try {
        const response = await fetch(url);
        return response.ok ? await response.text() : null;
      } catch {
        return null;
      }
    });
  const viewerHtml = await fetchViewer(viewerUrl);
  const html = viewerHtml ? injectBundleIntoViewer(viewerHtml, bundleJson, options) : null;
  return html
    ? {html, usedViewer: true}
    : {html: buildStandaloneHtml(bundleJson, options), usedViewer: false};
}
