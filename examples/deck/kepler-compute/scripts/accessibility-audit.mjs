// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Accessibility audit of host-added surfaces: accessible names, text contrast, visible keyboard
// focus, keyboard-only reachability and 390px / desktop map-control overlap.
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {getPlaywrightLaunchOptions} from '../../../../scripts/playwright/get-playwright-launch-options.mjs';
import {pinViteModules} from './pin-vite-modules.mjs';

const baseUrl = process.env.KEPLER_COMPUTE_URL ?? 'http://127.0.0.1:5318/';
const browser = await chromium.launch(
  getPlaywrightLaunchOptions({headless: true, backend: 'webgpu-core'})
);
const failures = [];

/** Runs in the page: returns findings for the host-added surfaces currently visible. */
function auditPage() {
  const findings = [];
  const surfaces = [
    ...document.querySelectorAll('.native-workflow, .sandbox-capabilities, .resident-series-view')
  ];
  const isVisible = element => {
    const style = getComputedStyle(element);
    const box = element.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' && box.width > 0 && box.height > 0;
  };
  const parseColor = value => {
    const match = value.match(/rgba?\(([^)]+)\)/);
    if (!match) return null;
    const [red, green, blue, alpha = 1] = match[1].split(/[ ,/]+/).filter(Boolean).map(Number);
    return {red, green, blue, alpha};
  };
  const luminance = ({red, green, blue}) => {
    const [r, g, b] = [red, green, blue].map(channel => {
      const value = channel / 255;
      return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const getBackground = element => {
    for (let node = element; node; node = node.parentElement) {
      const color = parseColor(getComputedStyle(node).backgroundColor);
      if (color && color.alpha > 0.9) return color;
    }
    return {red: 255, green: 255, blue: 255, alpha: 1};
  };
  const getName = element =>
    (
      element.getAttribute('aria-label') ||
      element.getAttribute('aria-labelledby') ||
      element.labels?.[0]?.textContent ||
      element.textContent ||
      element.getAttribute('title') ||
      ''
    ).trim();
  for (const surface of surfaces) {
    for (const control of surface.querySelectorAll(
      'button, input:not([type=hidden]), select, textarea, summary, [role=button], [tabindex="0"]'
    )) {
      if (!isVisible(control)) continue;
      if (!getName(control))
        findings.push(`unnamed ${control.tagName.toLowerCase()} ${control.id || control.className}`);
    }
    const walker = document.createTreeWalker(surface, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const element = node.parentElement;
      if (!element || !node.textContent.trim() || !isVisible(element)) continue;
      if (element.closest('[disabled], :disabled')) continue;
      const foreground = parseColor(getComputedStyle(element).color);
      if (!foreground) continue;
      const background = getBackground(element);
      const [light, dark] = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
      const ratio = (light + 0.05) / (dark + 0.05);
      const size = parseFloat(getComputedStyle(element).fontSize);
      if (ratio < (size >= 18.66 ? 3 : 4.5))
        findings.push(`contrast ${ratio.toFixed(2)} for "${node.textContent.trim().slice(0, 40)}"`);
    }
  }
  return findings;
}

async function auditViewport(label, viewport) {
  const context = await browser.newContext({viewport});
  const page = await context.newPage();
  await pinViteModules(page);
  page.on('pageerror', error => failures.push(`${label}: ${error.message}`));
  await page.goto(baseUrl);
  await page.locator('.maplibregl-canvas').waitFor();
  await page.waitForTimeout(2500);
  const findings = await page.evaluate(auditPage);
  for (const finding of findings) failures.push(`${label}: ${finding}`);

  // Keyboard-only: Tab must reach at least one control and show a visible focus indicator.
  let indicators = 0;
  for (let index = 0; index < 25; index++) {
    await page.keyboard.press('Tab');
    const visible = await page.evaluate(() => {
      const element = document.activeElement;
      if (!element || element === document.body) return null;
      const style = getComputedStyle(element);
      return style.outlineStyle !== 'none' || style.boxShadow !== 'none' || element.matches(':focus-visible');
    });
    if (visible) indicators++;
  }
  if (indicators < 5) failures.push(`${label}: only ${indicators} focus stops showed a focus indicator`);

  // Native map controls stay reachable and do not overlap each other.
  const overlaps = await page.evaluate(() => {
    const boxes = [...document.querySelectorAll('.map-control-button')]
      .filter(element => element.getBoundingClientRect().width > 0)
      .map(element => ({name: element.className, box: element.getBoundingClientRect()}));
    const result = [];
    for (let first = 0; first < boxes.length; first++) {
      const {box} = boxes[first];
      if (box.right > innerWidth || box.bottom > innerHeight || box.left < 0 || box.top < 0)
        result.push(`${boxes[first].name} is outside the viewport`);
      for (let second = first + 1; second < boxes.length; second++) {
        const other = boxes[second].box;
        if (box.left < other.right - 1 && other.left < box.right - 1 && box.top < other.bottom - 1 && other.top < box.bottom - 1)
          result.push(`${boxes[first].name} overlaps ${boxes[second].name}`);
      }
    }
    return result;
  });
  for (const overlap of overlaps) failures.push(`${label}: ${overlap}`);
  await context.close();
}

try {
  await auditViewport('desktop 1440x1000', {width: 1440, height: 1000});
  await auditViewport('mobile 390x844', {width: 390, height: 844});
  assert.deepEqual(failures, [], failures.join('\n'));
  console.log('accessibility-audit.mjs checks completed');
} finally {
  await browser.close();
}
