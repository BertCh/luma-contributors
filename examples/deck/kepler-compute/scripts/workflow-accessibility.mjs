// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {
  dock,
  frameGraph,
  openDock,
  runAcceptance,
  runAll,
  screenshot,
  TEST_IDS,
  useFirstTemplate,
  waitForStepStatus
} from './workflow-utils.mjs';

/** Self-contained so Playwright can serialize this function into the page. */
export function readControlNames(element) {
  function normalizeName(value) {
    return (value ?? '').replace(/\s+/g, ' ').trim();
  }
  function readName(control) {
    const labelledBy = control.getAttribute('aria-labelledby');
    if (labelledBy) {
      const name = normalizeName(
        labelledBy
          .split(/\s+/)
          .map(identifier => control.ownerDocument.getElementById(identifier)?.textContent ?? '')
          .join(' ')
      );
      if (name) return name;
    }
    const ariaName = normalizeName(control.ariaLabel);
    if (ariaName) return ariaName;
    const labelName = normalizeName(
      Array.from(control.labels ?? [], label => label.textContent).join(' ') ||
        control.closest('label')?.textContent
    );
    if (labelName) return labelName;
    // Form values and select options are content, not names. Only button-like controls name
    // themselves from their content (or from the value of a native input button).
    if (control.tagName === 'SELECT' || control.tagName === 'TEXTAREA') return '';
    if (control.tagName === 'INPUT') {
      return ['button', 'submit', 'reset'].includes(control.type)
        ? normalizeName(control.value) || normalizeName(control.getAttribute('title'))
        : '';
    }
    if (control.tagName === 'BUTTON' || ['button', 'tab'].includes(control.getAttribute('role'))) {
      const content = normalizeName(control.textContent);
      if (content) return content;
      return control.tagName === 'BUTTON' || control.getAttribute('role') === 'button'
        ? normalizeName(control.getAttribute('title'))
        : '';
    }
    return '';
  }
  return Array.from(
    element.querySelectorAll('button, input, select, textarea, [role="button"], [role="tab"]')
  )
    .filter(
      control =>
        !control.closest('[hidden], [aria-hidden="true"]') &&
        getComputedStyle(control).display !== 'none' &&
        getComputedStyle(control).visibility !== 'hidden'
    )
    .map(control => ({
      tag: control.tagName,
      id: control.id,
      testId: control.getAttribute('data-testid'),
      type: control.getAttribute('type'),
      text: control.textContent?.trim().slice(0, 100),
      name: readName(control)
    }));
}

/** Includes overlapping owned surfaces once, while excluding unrelated host panels and dialogs. */
export function readWorkbenchDuplicateIds({surfaceIds, dialogTitles}) {
  const selectors = surfaceIds.map(identifier => `[data-testid="${identifier}"]`);
  selectors.push('.side-panel__tab[data-for="workflows-nav"]');
  const roots = new Set(document.querySelectorAll(selectors.join(', ')));
  for (const dialog of document.querySelectorAll('[role="dialog"][aria-modal="true"]')) {
    const heading = (dialog.getAttribute('aria-labelledby') ?? '')
      .split(/\s+/)
      .map(identifier => document.getElementById(identifier)?.textContent ?? '')
      .join(' ')
      .trim();
    if (dialogTitles.includes(dialog.getAttribute('aria-label') || heading)) roots.add(dialog);
  }
  const elements = new Set();
  for (const surface of roots) {
    elements.add(surface);
    for (const element of surface.querySelectorAll('[id]')) elements.add(element);
  }
  const counts = new Map();
  for (const element of elements) {
    if (element.id) counts.set(element.id, (counts.get(element.id) ?? 0) + 1);
  }
  return Array.from(counts).filter(([, count]) => count > 1);
}

export function finishAccessibilityAudit(problems) {
  assert.deepEqual(
    problems,
    [],
    `Accessibility audit found ${problems.length} problems:\n` +
      `${problems.map(problem => problem.message ?? `${problem.surface}: ${problem.kind}`).join('\n')}\n` +
      JSON.stringify(problems, null, 2)
  );
}

/** Navigation and evaluation errors must not hide the remaining surfaces' accessibility issues. */
export async function collectSurfaceProblems(problems, surface, check) {
  try {
    return await check();
  } catch (error) {
    problems.push({
      surface,
      kind: 'surface-error',
      message: error instanceof Error ? error.message : String(error)
    });
    return undefined;
  }
}

export async function checkNames(page, root, context, problems) {
  const findings = problems ?? [];
  const controls = await root.evaluate(readControlNames);
  if (!controls.length)
    findings.push({
      surface: context,
      kind: 'empty-surface',
      message: `${context}: expected controls to audit`
    });
  const nameless = controls.filter(control => !control.name);
  if (nameless.length)
    findings.push({
      surface: context,
      kind: 'nameless-controls',
      controls: nameless,
      message: `${context}: every control must have a nonempty accessible name`
    });
  const duplicateIds = await page.evaluate(readWorkbenchDuplicateIds, {
    surfaceIds: [TEST_IDS.dock, TEST_IDS.palette, TEST_IDS.sidePanel],
    dialogTitles: ['Workflow templates', 'Import and export', 'Keyboard shortcuts']
  });
  if (duplicateIds.length)
    findings.push({
      surface: context,
      kind: 'duplicate-ids',
      ids: duplicateIds,
      message: `${context}: duplicate workbench ids: ${JSON.stringify(duplicateIds)}`
    });
  if (!problems) finishAccessibilityAudit(findings);
  return controls.length;
}

function readLuminance(color) {
  const linear = color.map(channel => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

export function calculateContrast(foreground, background) {
  const first = readLuminance(foreground);
  const second = readLuminance(background);
  return (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
}

/** Composite the two pixels (text and its background) through each CSS opacity group. */
export function readContrastSamples(elements) {
  function parseColor(text) {
    const components = text.match(/[\d.]+/g)?.map(Number);
    if (!components || components.length < 3 || !/^rgba?\(/.test(text)) {
      throw new Error(`Contrast sampling needs computed rgb/rgba; observed ${text}`);
    }
    return [components[0], components[1], components[2], components[3] ?? 1];
  }
  function compositeColor(front, back) {
    const alpha = front[3] + back[3] * (1 - front[3]);
    return [
      ...[0, 1, 2].map(index =>
        alpha === 0 ? 0 : (front[index] * front[3] + back[index] * back[3] * (1 - front[3])) / alpha
      ),
      alpha
    ];
  }
  return elements
    .filter(
      element =>
        element.getClientRects().length > 0 &&
        element.textContent?.trim() &&
        !element.closest(':disabled')
    )
    .slice(0, 12)
    .map(element => {
      let foreground = parseColor(getComputedStyle(element).color);
      let background = [0, 0, 0, 0];
      const opacities = [];
      // Opacity applies to a whole painted subtree, including its background, before the subtree
      // is composited over its parent. Multiplying text alpha alone gets nested surfaces wrong.
      for (let current = element; current; current = current.parentElement) {
        const style = getComputedStyle(current);
        if (style.backgroundImage !== 'none') {
          throw new Error(`Cannot calculate a solid background through ${style.backgroundImage}`);
        }
        const opacity = Number(style.opacity || '1');
        if (!Number.isFinite(opacity) || opacity < 0 || opacity > 1) {
          throw new Error(`Expected CSS opacity between 0 and 1; observed ${style.opacity}`);
        }
        const surface = parseColor(style.backgroundColor);
        foreground = compositeColor(foreground, surface);
        background = compositeColor(background, surface);
        foreground[3] *= opacity;
        background[3] *= opacity;
        opacities.push(opacity);
      }
      return {
        text: element.textContent.trim().slice(0, 80),
        opacities,
        foreground: compositeColor(foreground, [255, 255, 255, 1]).slice(0, 3),
        background: compositeColor(background, [255, 255, 255, 1]).slice(0, 3)
      };
    });
}

export async function checkContrast(root, context, selector, problems) {
  const findings = problems ?? [];
  const samples = await root.locator(selector).evaluateAll(readContrastSamples);
  if (!samples.length)
    findings.push({
      surface: context,
      kind: 'missing-contrast-samples',
      message: `${context}: expected computed color samples for ${selector}`
    });
  const measured = samples.map(sample => {
    const contrast = calculateContrast(sample.foreground, sample.background);
    if (!(contrast >= 4.5))
      findings.push({
        surface: context,
        kind: 'contrast',
        ...sample,
        contrast,
        message:
          `${context}: "${sample.text}" must have contrast >= 4.5:1; saw ${contrast.toFixed(3)}:1 ` +
          `(foreground ${sample.foreground}, background ${sample.background})`
      });
    return {...sample, contrast};
  });
  if (!problems) finishAccessibilityAudit(findings);
  return measured;
}

export async function auditTabbedSurface(page, root, context, problems) {
  const tabs = root.getByRole('tab');
  const count = await tabs.count();
  assert.equal(count > 0, true, `${context}: expected at least one tab`);
  const audited = {};
  for (let index = 0; index < count; index++) {
    const tab = tabs.nth(index);
    await collectSurfaceProblems(problems, `${context}/tab ${index + 1}`, async () => {
      const label = (await tab.textContent())?.trim() || `tab ${index + 1}`;
      const surface = `${context}/${label}`;
      await tab.click();
      audited[surface] = await checkNames(page, root, surface, problems);
    });
  }
  return audited;
}

export async function auditResultInspector(page, directory, problems) {
  const runCheck = (surface, check) =>
    problems ? collectSurfaceProblems(problems, surface, check) : check();
  const inspector = dock(page).getByTestId(TEST_IDS.inspector);
  const audited = {};
  const outputIds = await runCheck('Result inspector', async () => {
    await inspector.getByTestId(TEST_IDS.inspectorTab('result')).click();
    await inspector.getByTestId(TEST_IDS.result).waitFor();
    audited.result = await checkNames(page, inspector, 'Result inspector', problems);
    const identifiers = await inspector
      .locator('[data-testid^="workflow-result-output-"]')
      .evaluateAll(elements => elements.map(element => element.getAttribute('data-testid')));
    assert.equal(identifiers.length > 0, true, 'Result inspector must expose at least one output');
    return identifiers;
  });
  for (const outputId of outputIds ?? []) {
    await runCheck(outputId, async () => {
      const output = inspector.getByTestId(outputId);
      const disclosure = output.locator('details').first();
      if ((await disclosure.count()) && (await disclosure.getAttribute('open')) === null) {
        await disclosure
          .locator('summary')
          .first()
          .click({position: {x: 8, y: 8}});
      }
      const table = output.locator('table');
      await table.waitFor();
      const summaries = await table
        .getByRole('button', {name: /^Summarize /})
        .evaluateAll(elements => elements.map(element => element.getAttribute('aria-label')));
      assert.equal(summaries.length > 0, true, `${outputId}: expected column summary controls`);
      for (const name of summaries) {
        const column = name.slice('Summarize '.length);
        const surface = `${outputId}/${column} summary`;
        await runCheck(surface, async () => {
          await table.getByRole('button', {name, exact: true}).click();
          const dialog = inspector.getByRole('dialog', {name: `${column} summary`, exact: true});
          await dialog.waitFor();
          let completed = false;
          try {
            audited[`${outputId}/${column}`] = await checkNames(page, dialog, surface, problems);
            await screenshot(
              page,
              directory,
              `workflow-accessibility-${outputId}-${column.replace(/[^a-z0-9-]/gi, '-')}-summary`
            );
            completed = true;
          } finally {
            if (completed || problems) {
              await dialog
                .getByRole('button', {name: `Close ${column} summary`, exact: true})
                .click();
              await dialog.waitFor({state: 'hidden'});
            }
          }
        });
      }
      audited[outputId] = await checkNames(page, output, `${outputId} controls`, problems);
    });
  }
  await runCheck('Result inspector screenshot', () =>
    screenshot(page, directory, 'workflow-accessibility-result-inspector')
  );
  await runCheck('Restore Parameters tab', () =>
    inspector.getByTestId(TEST_IDS.inspectorTab('parameters')).click()
  );
  return audited;
}

async function runAccessibilityScenario(page, directory) {
  await openDock(page);
  await useFirstTemplate(page);
  await frameGraph(page);
  await page.evaluate(() => window.__keplerWorkflowTest.store.select(['clusters']));
  const problems = [];
  const audited = {};
  const contrast = {};
  await collectSurfaceProblems(problems, 'Dock', async () => {
    audited.dock = await checkNames(page, dock(page), 'Dock', problems);
    await screenshot(page, directory, 'workflow-accessibility-dock');
  });
  await collectSurfaceProblems(problems, 'Selected-step inspector tabs', async () => {
    Object.assign(
      audited,
      await auditTabbedSurface(
        page,
        dock(page).getByTestId(TEST_IDS.inspector),
        'Selected-step inspector',
        problems
      )
    );
  });
  await collectSurfaceProblems(problems, 'Workflow run', () => runAll(page));
  await collectSurfaceProblems(problems, 'Result after run', async () => {
    Object.assign(audited, await auditResultInspector(page, directory, problems));
  });
  for (const [name, root, selector] of [
    ['toolbar', dock(page).getByTestId(TEST_IDS.toolbar), 'button:not(:disabled)'],
    ['node', dock(page).getByTestId(TEST_IDS.node('clusters')), 'strong'],
    ['inspector', dock(page).getByTestId(TEST_IDS.inspector), 'label'],
    ['status', dock(page).getByTestId(TEST_IDS.canvas), '[data-testid^="workflow-node-status-"]']
  ]) {
    await collectSurfaceProblems(problems, `${name} contrast`, async () => {
      contrast[name] = await checkContrast(root, `${name} contrast`, selector, problems);
    });
  }
  await collectSurfaceProblems(problems, 'Palette', async () => {
    await dock(page).getByTestId(TEST_IDS.addStep).click();
    const palette = page.getByTestId(TEST_IDS.palette);
    await palette.waitFor();
    audited.palette = await checkNames(page, palette, 'Palette', problems);
    await screenshot(page, directory, 'workflow-accessibility-palette');
  });
  await collectSurfaceProblems(problems, 'Close palette', () =>
    page.evaluate(() => window.__keplerWorkflowTest.store.closePalette())
  );

  for (const [dialog, title] of [
    ['templates', 'Workflow templates'],
    ['import-export', 'Import and export'],
    ['shortcuts', 'Keyboard shortcuts']
  ]) {
    await collectSurfaceProblems(problems, title, async () => {
      await page.evaluate(name => window.__keplerWorkflowTest.store.setDialog(name), dialog);
      const modal = page.getByRole('dialog', {name: title, exact: true});
      await modal.waitFor();
      audited[dialog] = await checkNames(page, modal, title, problems);
      if (await modal.getByRole('tab').count()) {
        Object.assign(audited, await auditTabbedSurface(page, modal, title, problems));
      }
      await screenshot(page, directory, `workflow-accessibility-${dialog}`);
    });
    await collectSurfaceProblems(problems, `Close ${title}`, () =>
      page.evaluate(() => window.__keplerWorkflowTest.store.setDialog(null))
    );
  }

  await collectSurfaceProblems(problems, 'Workflows side panel', async () => {
    const sideTab = page.locator('.side-panel__tab[data-for="workflows-nav"]');
    await sideTab.click();
    const sidePanel = page.getByTestId(TEST_IDS.sidePanel);
    await sidePanel.waitFor();
    audited.sidePanel = await checkNames(page, sidePanel, 'Workflows side panel', problems);
    const sideTabName = await sideTab.evaluate(element =>
      (
        element.ariaLabel ||
        element
          .getAttribute('aria-labelledby')
          ?.split(/\s+/)
          .map(identifier => document.getElementById(identifier)?.textContent ?? '')
          .join(' ') ||
        element.textContent ||
        ''
      ).trim()
    );
    if (!sideTabName)
      problems.push({
        surface: 'Workflows side-panel tab',
        kind: 'nameless-controls',
        message: 'Workflows side-panel tab must have a nonempty accessible name'
      });
    await screenshot(page, directory, 'workflow-accessibility-side-panel');
  });

  const reached = new Set();
  await collectSurfaceProblems(problems, 'Tab order', async () => {
    if (!(await page.evaluate(() => window.__keplerWorkflowTest.store.getState().dock.logOpen))) {
      await dock(page).getByTestId(TEST_IDS.runLogToggle).click();
    }
    await dock(page).getByTestId(TEST_IDS.runLog).waitFor();
    await dock(page).getByTestId(TEST_IDS.toolbar).locator('button:not(:disabled)').first().focus();
    const visited = [];
    for (let index = 0; index < 150 && reached.size < 4; index++) {
      const focus = await page.evaluate(
        identifiers => {
          const element = document.activeElement;
          return {
            tag: element?.tagName,
            testId: element?.getAttribute('data-testid'),
            regions: identifiers.filter(identifier =>
              element?.closest(`[data-testid="${identifier}"]`)
            )
          };
        },
        [TEST_IDS.toolbar, TEST_IDS.canvas, TEST_IDS.inspector, TEST_IDS.runLog]
      );
      visited.push(focus);
      for (const region of focus.regions) reached.add(region);
      if (reached.size < 4) await page.keyboard.press('Tab');
    }
    assert.deepEqual(
      [...reached].sort(),
      [TEST_IDS.toolbar, TEST_IDS.canvas, TEST_IDS.inspector, TEST_IDS.runLog].sort(),
      `Tab order must reach toolbar, canvas, inspector and run log; visited ${JSON.stringify(visited)}`
    );
    await screenshot(page, directory, 'workflow-accessibility-tab-order');
  });
  await collectSurfaceProblems(problems, 'Close keyboard palette', () =>
    page.evaluate(() => window.__keplerWorkflowTest.store.closePalette())
  );
  await collectSurfaceProblems(problems, 'Disabled node', async () => {
    await dock(page).getByTestId(TEST_IDS.nodeDisable('clusters')).click();
    await waitForStepStatus(page, 'clusters', 'disabled');
    await screenshot(page, directory, 'workflow-accessibility-disabled-node');
    contrast.disabledNode = await checkContrast(
      dock(page).getByTestId(TEST_IDS.node('clusters')),
      'Disabled node status with group opacity',
      `[data-testid="${TEST_IDS.nodeStatus('clusters')}"]`,
      problems
    );
  });
  finishAccessibilityAudit(problems);
  return {audited, contrast, tabRegions: [...reached]};
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  await runAcceptance('workflow-accessibility', runAccessibilityScenario);
}
