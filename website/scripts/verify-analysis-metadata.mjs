// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Run against an already served website: node website/scripts/verify-analysis-metadata.mjs <base-url>
// The base URL may include the deployment subpath. This checks metadata UI, not GPU pictures.
import {pathToFileURL, fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import {chromium} from 'playwright';

/** Read the same shared records as the page, without requiring a built package. */
export async function loadWorkflowTemplateIds() {
  const bundled = await build({
    entryPoints: [
      fileURLToPath(new URL('../src/components/meta/pages/workflows/workflows.ts', import.meta.url))
    ],
    bundle: true,
    platform: 'node',
    format: 'esm',
    write: false
  });
  const {WORKFLOW_TEMPLATES} = await import(
    `data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`
  );
  return WORKFLOW_TEMPLATES.map(template => template.id);
}

/** Count alone would accept a missing template replaced with a duplicate or an unknown id. */
export function verifyWorkflowTemplateIds(actualIds, expectedIds) {
  if (JSON.stringify([...actualIds].sort()) !== JSON.stringify([...expectedIds].sort())) {
    throw new Error(
      `Curated template ids differ: expected ${expectedIds.join(', ')}; rendered ${actualIds.join(', ')}`
    );
  }
}

export async function verifyAnalysisMetadata() {
  const baseUrl = new URL(`${(process.argv[2] ?? 'http://localhost:3000').replace(/\/$/, '')}/`);
  const browser = await chromium.launch();
  const page = await browser.newPage({viewport: {width: 1280, height: 900}});
  const failures = [];
  function check(condition, description) {
    if (!condition) failures.push(description);
  }
  try {
    await page.goto(new URL('explore?question=where-significant', baseUrl).href);
    const questionRow = page
      .locator('.facet-row')
      .filter({has: page.locator('summary > span', {hasText: /^Question$/})});
    await questionRow.locator('button[aria-pressed="true"]').waitFor();
    check(
      (await questionRow.locator('button[aria-pressed="true"]').count()) === 1,
      'question query selects exactly one shared facet'
    );
    const filteredCount = await page.locator('.explore-count').first().textContent();
    await page.reload();
    await questionRow.locator('button[aria-pressed="true"]').waitFor();
    check(
      (await page.locator('.explore-count').first().textContent()) === filteredCount,
      'question filter survives reload'
    );
    await questionRow.locator('button[aria-pressed="true"]').click();
    await page.waitForURL(url => !url.searchParams.has('question'));
    check(
      (await page.locator('.explore-count').first().textContent()) !== filteredCount,
      'clearing the question restores entries'
    );

    await page.goto(new URL('fn/local-morans-i', baseUrl).href);
    await page.getByRole('heading', {name: 'Reading the result'}).waitFor();
    check(
      await page
        .getByText('Neighbour weights set the meaning of local association.', {exact: false})
        .isVisible(),
      'named-operation interpretation is rendered'
    );
    await page.locator('.function-facets a[href*="question="]').first().click();
    await questionRow.locator('button[aria-pressed="true"]').waitFor();
    check(
      new URL(page.url()).searchParams.has('question'),
      'function-page question link opens filtered explore route'
    );

    await page.goto(new URL('how-it-works/workflows', baseUrl).href);
    await page.getByRole('heading', {name: 'Curated workflows', exact: true}).waitFor();
    verifyWorkflowTemplateIds(
      await page
        .locator('.workflows-index details')
        .evaluateAll(elements => elements.map(element => element.getAttribute('data-template-id'))),
      await loadWorkflowTemplateIds()
    );
    await page.getByRole('tab', {name: /Demand within reach of each facility/}).click();
    check(
      await page
        .getByText('This recording runs the drive-time band recipe.', {exact: false})
        .first()
        .isVisible(),
      'recipe/template execution differences are visible'
    );
    if (failures.length) throw new Error(`Website AX metadata failures:\n${failures.join('\n')}`);
    console.log(
      'Website AX metadata: question routing, filtering, interpretation and curated records passed.'
    );
  } finally {
    await browser.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await verifyAnalysisMetadata();
}
