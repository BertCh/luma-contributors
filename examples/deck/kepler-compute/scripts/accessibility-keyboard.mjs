// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Keyboard-only accessibility pass: disclosure toggles, modal dialog focus management,
// absence of keyboard traps on the page and presence of named status / alert regions.
// Uses only Tab, Shift+Tab, Enter, Space, Escape and arrow keys to drive the page.
// The keyboard-only workflow (checkKeyboardWorkflow) imports a CSV, adds a filter, runs an analysis
// and saves a portable bundle without any pointer event; the only non-key action is Playwright
// answering the native file chooser that the focused control opened with Enter/Space.
// Set KEPLER_A11Y_WORKFLOW=0 to skip it. No real screen reader is available in CI: speech output,
// reading order and live-region announcements still need a manual NVDA/VoiceOver pass.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const baseUrl = process.env.KEPLER_COMPUTE_URL ?? 'http://127.0.0.1:5318/?rows=256';
const maximumTabStops = Number(process.env.KEPLER_A11Y_TAB_STOPS ?? 120);
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const failures = [];

/** Describes document.activeElement; runs in the page. */
function describeActiveElement() {
  const element = document.activeElement;
  if (!element || element === document.body || element === document.documentElement) return null;
  const style = getComputedStyle(element);
  const name = (
    element.getAttribute('aria-label') ||
    element.labels?.[0]?.textContent ||
    element.textContent ||
    element.getAttribute('title') ||
    element.getAttribute('placeholder') ||
    ''
  ).trim();
  const labelledBy = element.getAttribute('aria-labelledby');
  const labelledText = labelledBy
    ? labelledBy
        .split(/\s+/)
        .map(id => document.getElementById(id)?.textContent ?? '')
        .join(' ')
        .trim()
    : '';
  const focusIndicator =
    (style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0) ||
    style.boxShadow !== 'none' ||
    element.matches(':focus-visible');
  return {
    tag: element.tagName.toLowerCase(),
    id: element.id,
    className: String(element.className).slice(0, 60),
    name: name || labelledText,
    focusIndicator,
    inDialog: Boolean(element.closest('[role="dialog"], .ReactModal__Content')),
    key: `${element.tagName}#${element.id}.${String(element.className).slice(0, 40)}|${(element.textContent ?? '').trim().slice(0, 30)}`
  };
}

async function focusByTab(page, predicate, limit = maximumTabStops, backwards = false) {
  for (let index = 0; index < limit; index++) {
    await page.keyboard.press(backwards ? 'Shift+Tab' : 'Tab');
    if (await page.evaluate(predicate)) return true;
  }
  return false;
}

async function checkDisclosure(page) {
  // (1) A disclosure (details/summary or aria-expanded button) is reachable by Tab and toggles
  // with both Enter and Space, reflecting the new state in `open` or `aria-expanded`.
  const readState = () => {
    const element = document.activeElement;
    if (element?.tagName === 'SUMMARY') return String(element.parentElement.open);
    return element?.getAttribute('aria-expanded');
  };
  const isDisclosure = () => {
    const element = document.activeElement;
    return (
      element?.tagName === 'SUMMARY' ||
      (element?.tagName === 'BUTTON' && element.hasAttribute('aria-expanded'))
    );
  };
  await page.evaluate(() => document.activeElement?.blur());
  const reached = await focusByTab(page, isDisclosure);
  if (!reached) {
    failures.push('disclosure: no summary or aria-expanded button reachable by Tab');
    return;
  }
  for (const key of ['Enter', 'Space']) {
    const before = await page.evaluate(readState);
    await page.keyboard.press(key);
    const after = await page.evaluate(readState);
    if (before === after) failures.push(`disclosure: ${key} did not change expanded state (${before})`);
    // Restore the original state for the next key.
    await page.keyboard.press(key);
    const restored = await page.evaluate(readState);
    if (restored !== before) failures.push(`disclosure: ${key} twice did not restore state`);
  }
}

async function checkDialog(page) {
  // (2) Open the Add Data dialog with the keyboard and verify focus management.
  await page.evaluate(() => document.activeElement?.blur());
  const found = await focusByTab(
    page,
    () => /^Add Data$/i.test((document.activeElement?.textContent ?? '').trim())
  );
  if (!found) {
    failures.push('dialog: Add Data button not reachable by Tab');
    return;
  }
  const opener = await page.evaluate(describeActiveElement);
  await page.keyboard.press('Enter');
  await page.locator('.ReactModal__Content').first().waitFor({timeout: 15000});
  const attributes = await page.evaluate(() => {
    const content = document.querySelector('.ReactModal__Content');
    return {role: content?.getAttribute('role'), modal: content?.getAttribute('aria-modal')};
  });
  if (attributes.role !== 'dialog') failures.push(`dialog: role is ${attributes.role}, expected dialog`);
  await page.waitForTimeout(300);
  let state = await page.evaluate(describeActiveElement);
  if (!state?.inDialog) {
    // Some dialogs do not autofocus; Tab must then enter the dialog on the first stop.
    await page.keyboard.press('Tab');
    state = await page.evaluate(describeActiveElement);
  }
  if (!state?.inDialog) failures.push('dialog: focus did not move inside the dialog');

  // Focus trap: forward and backward Tab never leaves the dialog.
  let escaped = 0;
  for (let index = 0; index < 40; index++) {
    await page.keyboard.press('Tab');
    const active = await page.evaluate(describeActiveElement);
    if (active && !active.inDialog) escaped++;
  }
  for (let index = 0; index < 10; index++) {
    await page.keyboard.press('Shift+Tab');
    const active = await page.evaluate(describeActiveElement);
    if (active && !active.inDialog) escaped++;
  }
  if (escaped) failures.push(`dialog: focus left the dialog ${escaped} times while tabbing`);

  await page.keyboard.press('Escape');
  await page.locator('.ReactModal__Overlay').first().waitFor({state: 'hidden', timeout: 15000}).catch(() => {
    failures.push('dialog: Escape did not close the dialog');
  });
  await page.waitForTimeout(300);
  const returned = await page.evaluate(describeActiveElement);
  if (!returned || returned.key !== opener.key)
    failures.push(
      `dialog: focus did not return to opener (${opener.key} -> ${returned?.key ?? 'body'})`
    );
}

async function checkNoKeyboardTrap(page) {
  // (3) Tab through the page: focus must keep advancing, reach the end or wrap, and every
  // stop needs a focus indicator and an accessible name.
  await page.evaluate(() => {
    document.activeElement?.blur();
    window.scrollTo(0, 0);
  });
  const seen = new Map();
  let wrapped = false;
  let repeats = 0;
  let previousKey = null;
  for (let index = 0; index < maximumTabStops; index++) {
    await page.keyboard.press('Tab');
    const active = await page.evaluate(describeActiveElement);
    if (!active) {
      wrapped = true; // focus left the document (browser chrome), i.e. the page end
      break;
    }
    if (active.inDialog) continue;
    if (active.key === previousKey) {
      repeats++;
      if (repeats >= 3) {
        failures.push(`trap: focus stuck on ${active.key}`);
        break;
      }
    } else repeats = 0;
    previousKey = active.key;
    if (seen.has(active.key)) {
      wrapped = true;
      break;
    }
    seen.set(active.key, active);
    if (!active.name) failures.push(`name: focused ${active.tag} ${active.id || active.className} has no accessible name`);
    if (!active.focusIndicator)
      failures.push(`focus: ${active.tag} ${active.id || active.className} has no focus indicator`);
  }
  if (seen.size < 5) failures.push(`trap: only ${seen.size} distinct focus stops reached`);
  if (!wrapped) console.log(`note: ${maximumTabStops} tab stops without wrapping (${seen.size} distinct)`);
  // Reverse direction must also move focus.
  const before = await page.evaluate(describeActiveElement);
  await page.keyboard.press('Shift+Tab');
  const after = await page.evaluate(describeActiveElement);
  if (before && after && before.key === after.key) failures.push('trap: Shift+Tab did not move focus');
}

async function checkLiveRegions(page) {
  // (4) Every status / alert region currently shown carries text; consumer notices expose
  // their consumer and state attributes so assistive technology gets a stable message.
  const regions = await page.evaluate(() =>
    [...document.querySelectorAll('[role="alert"], [role="status"]')]
      .filter(element => {
        const box = element.getBoundingClientRect();
        // Visually hidden (1px clipped) live regions such as dnd-kit's are empty until they announce.
        return getComputedStyle(element).display !== 'none' && box.width > 1 && box.height > 1;
      })
      .map(element => ({
        role: element.getAttribute('role'),
        text: (element.textContent ?? '').trim(),
        consumer: element.getAttribute('data-resident-consumer'),
        state: element.getAttribute('data-resident-consumer-state')
      }))
  );
  for (const region of regions) {
    if (!region.text) failures.push(`live region: empty visible role=${region.role}`);
    if (region.consumer && region.role !== (region.state === 'error' ? 'alert' : 'status'))
      failures.push(`live region: consumer ${region.consumer} state ${region.state} uses role=${region.role}`);
  }
  console.log(`live regions shown: ${regions.length}`);
}

/** Records a failure as `workflow step "<name>": <reason>` and keeps going with the next step. */
async function runWorkflowStep(name, action) {
  try {
    await action();
    console.log(`workflow step ok: ${name}`);
  } catch (error) {
    failures.push(`workflow step "${name}": ${error instanceof Error ? error.message : error}`);
  }
}

/** Presses Tab (up to limit times) until the in-page predicate passes; throws with the step name. */
async function tabUntil(page, description, predicate, limit = 200, keepFocus = false) {
  if (!keepFocus) await page.evaluate(() => document.activeElement?.blur());
  if (await focusByTab(page, predicate, limit)) return;
  throw new Error(`${description} is not reachable with Tab within ${limit} stops`);
}

async function checkKeyboardWorkflow(page) {
  const csv = Buffer.from(
    'latitude,longitude,value,note\n' +
      Array.from({length: 300}, (_, index) =>
        `${40.5 + (index % 5) / 32},${-74 + (index % 7) / 32},${index},row${index}`
      ).join('\n')
  );
  // Step 1: import. Enter opens Add Data, Enter/Space on the focused file control opens the chooser.
  await runWorkflowStep('import: open Add Data dialog', async () => {
    await tabUntil(page, 'Add Data button', () => /^Add Data$/i.test((document.activeElement?.textContent ?? '').trim()));
    await page.keyboard.press('Enter');
    await page.locator('.ReactModal__Content').first().waitFor({timeout: 15000});
  });
  await runWorkflowStep('import: choose a file from the keyboard', async () => {
    await tabUntil(
      page,
      'file chooser control inside the dialog',
      () => {
        const element = document.activeElement;
        return Boolean(
          element?.closest('.ReactModal__Content') &&
            (element.matches('input[type=file]') || /browse|upload|choose/i.test(element.textContent ?? ''))
        );
      },
      60
    );
    const chooser = page.waitForEvent('filechooser', {timeout: 10000});
    await page.keyboard.press('Enter');
    await (await chooser).setFiles({name: 'keyboard-observations.csv', mimeType: 'text/csv', buffer: csv});
    await page.locator('.ReactModal__Content').getByText('keyboard-observations.csv').first().waitFor({timeout: 15000});
    // The submit button stays disabled (and out of the Tab order) until the file is processed.
    await page.waitForFunction(
      () =>
        [...document.querySelectorAll('.ReactModal__Content button')].some(
          button => /^Add Data$/i.test((button.textContent ?? '').trim()) && !button.disabled
        ),
      undefined,
      {timeout: 30000}
    );
  });
  await runWorkflowStep('import: submit the dialog with Enter', async () => {
    await tabUntil(
      page,
      'dialog Add Data submit button',
      () => {
        const element = document.activeElement;
        return Boolean(
          element?.closest('.ReactModal__Content') &&
            element.tagName === 'BUTTON' &&
            /^Add Data$/i.test((element.textContent ?? '').trim())
        );
      },
      60,
      true
    );
    await page.keyboard.press('Enter');
    await page.waitForFunction(
      () =>
        window.__keplerComputeTest
          ?.getState()
          .publications.some(publication => publication.selectedCount === 300),
      undefined,
      {timeout: 60000}
    );
  });
  // Step 2: filter. Open the Filters tab, Add Filter, pick the first offered field with arrows.
  await runWorkflowStep('filter: add a filter by keyboard', async () => {
    const readFilterCount = () =>
      page.evaluate(() => window.__keplerComputeTest.saveNativeConfig().config.visState.filters.length);
    const before = await readFilterCount();
    await tabUntil(
      page,
      'Filters tab',
      () => /^filters?$/i.test((document.activeElement?.getAttribute('aria-label') ?? document.activeElement?.textContent ?? '').trim())
    );
    await page.keyboard.press('Enter');
    await tabUntil(page, 'Add Filter button', () => /add filter/i.test(document.activeElement?.textContent ?? ''), 60, true);
    await page.keyboard.press('Enter');
    // With more than one dataset the button opens a dataset picker whose search input takes focus.
    await page.waitForTimeout(500);
    if (await page.evaluate(() => document.activeElement?.classList.contains('typeahead__input'))) {
      for (const key of ['ArrowDown', 'Enter']) {
        await page.keyboard.press(key);
        await page.waitForTimeout(250);
      }
    }
    // The new filter shows an empty field selector: Enter opens it, ArrowDown highlights, Enter picks.
    await tabUntil(page, 'new filter field selector', () => /select a field/i.test(document.activeElement?.textContent ?? ''), 40, true);
    for (const key of ['Enter', 'ArrowDown', 'Enter']) {
      await page.keyboard.press(key);
      await page.waitForTimeout(250);
    }
    await page.waitForFunction(
      count => window.__keplerComputeTest.saveNativeConfig().config.visState.filters.length > count,
      before,
      {timeout: 10000}
    );
  });
  // Step 3: analyze. Run the analysis catalog's default invocation with Tab, ArrowDown and Enter.
  await runWorkflowStep('analyze: run a catalog analysis by keyboard', async () => {
    // The filter step left the Filters tab open; the analysis catalog lives under Layers.
    await tabUntil(page, 'Layers tab', () => (document.activeElement?.getAttribute('aria-label') ?? '') === 'Layers');
    await page.keyboard.press('Enter');
    await tabUntil(page, 'catalog Run button', () => document.activeElement?.getAttribute('data-testid') === 'catalog-run');
    await page.keyboard.press('Enter');
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="catalog-result"]') ||
        document.querySelector('[data-testid="catalog-error"]'),
      undefined,
      {timeout: 60000}
    );
    const failed = await page.locator('[data-testid="catalog-error"]').count();
    if (failed) throw new Error(`catalog reported: ${await page.locator('[data-testid="catalog-error"]').innerText()}`);
  });
  // Step 4: save. Enter on "Save portable bundle" downloads; the status region announces it.
  await runWorkflowStep('save: download a portable bundle by keyboard', async () => {
    // Kepler-native route: Save menu (#save-action) -> Export Map -> Save map region.
    await tabUntil(page, 'Save menu button', () => document.activeElement?.id === 'save-action');
    await page.keyboard.press('Enter');
    await tabUntil(page, 'Export Map menu item', () => /^Export Map$/i.test((document.activeElement?.getAttribute('aria-label') ?? '').trim()), 40, true);
    await page.keyboard.press('Enter');
    await page.getByRole('region', {name: 'Save map', exact: true}).waitFor({timeout: 15000});
    await tabUntil(page, 'Save portable bundle button', () => /save portable bundle/i.test(document.activeElement?.textContent ?? ''), 80, true);
    const download = page.waitForEvent('download', {timeout: 30000});
    await page.keyboard.press('Enter');
    const file = await download;
    if (!/\.bundle\.json$/.test(file.suggestedFilename()))
      throw new Error(`unexpected download name ${file.suggestedFilename()}`);
    await page.waitForFunction(
      () =>
        [...document.querySelectorAll('[role="status"]')].some(element =>
          /Portable workspace saved/.test(element.textContent ?? '')
        ),
      undefined,
      {timeout: 10000}
    );
  });
}

try {
  for (const [label, viewport] of [
    ['desktop 1440x1000', {width: 1440, height: 1000}],
    ['mobile 390x844', {width: 390, height: 844}]
  ]) {
    const context = await browser.newContext({viewport});
    const page = await context.newPage();
    await pinViteModules(page);
    const start = failures.length;
    page.on('pageerror', error => failures.push(`pageerror: ${error.message}`));
    await page.goto(baseUrl);
    await page.locator('.maplibregl-canvas').waitFor();
    await page.waitForTimeout(2500);
    await checkLiveRegions(page);
    await checkNoKeyboardTrap(page);
    await checkDisclosure(page);
    await checkDialog(page);
    if (process.env.KEPLER_A11Y_WORKFLOW !== '0' && label.startsWith('desktop'))
      await checkKeyboardWorkflow(page);
    for (let index = start; index < failures.length; index++)
      failures[index] = `${label}: ${failures[index]}`;
    await context.close();
  }
  assert.deepEqual(failures, [], failures.join('\n'));
  console.log('accessibility-keyboard.mjs checks completed');
} finally {
  await browser.close();
}
