// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Run from the repository root: node website/scripts/snapshot-workflow-intermediates.mjs
// Requires a real WebGPU browser. Captures one chained run (not one operation per picture), then
// publishes all twelve pictures and their record together. Follow with the provenance node spec.
import {createHash} from 'node:crypto';
import {mkdir, readFile, rename, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {chromium} from 'playwright';
import {createServer} from 'vite';
import {PNG} from 'pngjs';
import {verifyWorkflowStagePixels} from '../src/gpu/runs/workflow-intermediates.ts';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const publicDirectory = path.join(websiteDirectory, 'public');
const manifestPath = path.join(websiteDirectory, 'src/gpu/runs/workflow-intermediates.manifest.json');
const port = 3304;
const hashBytes = bytes => createHash('sha256').update(bytes).digest('hex');
const server = await createServer({
  configFile: path.join(websiteDirectory, 'scripts/harness.vite.config.ts'),
  server: {port, strictPort: true},
  plugins: [{
    name: 'workflow-intermediate-capture-page',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url !== '/workflow-intermediates.html') return next();
        response.setHeader('Content-Type', 'text/html');
        response.end(`<!doctype html><html><body><canvas width="640" height="360"></canvas>
<script type="module">
import {captureWorkflowIntermediates} from '/src/gpu/runs/workflow-intermediates.ts';
window.workflowCapture = {status: 'loading'};
try {
  window.workflowCapture.result = await captureWorkflowIntermediates(document.querySelector('canvas'));
  window.workflowCapture.status = 'ready';
} catch (error) {
  window.workflowCapture.error = error.stack || String(error);
  window.workflowCapture.status = 'error';
}
</script></body></html>`);
      });
    }
  }]
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({
    args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist',
      ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])]
  });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://localhost:${port}/workflow-intermediates.html`);
  await page.waitForFunction(() => window.workflowCapture?.status !== 'loading' && window.workflowCapture, null, {timeout: 300000});
  const capture = await page.evaluate(() => window.workflowCapture);
  if (capture.status !== 'ready') throw new Error(capture.error);
  if (errors.length) throw new Error(errors.join('\n'));
  const {run, images} = capture.result;
  if (run.submissionCount !== 1 || images.length !== 12) throw new Error('Expected one submission and twelve pictures');
  const existing = JSON.parse(await readFile(manifestPath, 'utf8'));
  const directory = path.join(publicDirectory, 'workflow-intermediates', run.runId);
  await mkdir(directory, {recursive: true});
  for (const image of images) {
    const picture = run.pictures.find(candidate => candidate.path === image.path);
    if (!picture || !image.path.startsWith(`workflow-intermediates/${run.runId}/`)) throw new Error('Invalid picture identity');
    if (!image.dataUrl.startsWith('data:image/png;base64,')) throw new Error(`PNG encoding failed: ${image.path}`);
    const bytes = Buffer.from(image.dataUrl.split(',')[1], 'base64');
    if (hashBytes(bytes) !== picture.sha256) throw new Error(`Picture hash mismatch: ${image.path}`);
    const stage = run.stages.find(candidate => candidate.id === picture.stageId);
    if (!stage) throw new Error(`Missing stage ${picture.stageId}`);
    verifyWorkflowStagePixels(stage, picture.theme, run.input.cellSize, PNG.sync.read(bytes));
    await writeFile(path.join(publicDirectory, image.path), bytes, {flag: 'wx'});
  }
  const recordPath = `workflow-intermediates/${run.runId}/record.json`;
  const record = `${JSON.stringify(run)}\n`;
  await writeFile(path.join(publicDirectory, recordPath), record, {flag: 'wx'});
  const manifest = {
    schemaVersion: 1,
    runs: [
      ...existing.runs.filter(candidate => candidate.templateId !== run.templateId),
      {runId: run.runId, templateId: run.templateId, recordPath,
        recordSha256: hashBytes(record), pictures: run.pictures}
    ]
  };
  const temporaryManifest = `${manifestPath}.${process.pid}.tmp`;
  await writeFile(temporaryManifest, `${JSON.stringify(manifest, null, 2)}\n`, {flag: 'wx'});
  await rename(temporaryManifest, manifestPath);
  console.log(`Captured ${run.runId}: ${images.length} pictures from one chained submission.`);
  console.log('Run .ax/node-spec.sh modules/kepler/test/curated/workflow-intermediate-provenance.node.spec.ts, then review all six stages in both themes on /workflows.');
} finally {
  await browser?.close();
  await server.close();
}
