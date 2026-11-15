// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Source checks own their fixture size, even when a driver supplies only the server URL. */
export function getSourceCheckUrl(argument, {rows, fixture, retry = false} = {}) {
  const url = new URL(argument?.startsWith('http') ? argument : 'http://127.0.0.1:5318/');
  // The old scripts navigated to the app instead of their fixture when passed a bare server URL,
  // or waited for 256/4096 rows on the app's default 100,000-row fixture. Neither tests ingestion.
  if (fixture && !url.pathname.endsWith('.html'))
    url.pathname = `${url.pathname.replace(/\/$/, '')}/scripts/fixtures/${fixture}.html`;
  if (rows !== undefined) url.searchParams.set('rows', String(rows));
  if (retry) {
    const policies = new Set((url.searchParams.get('automatic') ?? '').split(',').filter(Boolean));
    policies.add('retry');
    url.searchParams.set('automatic', [...policies].join(','));
  }
  return url.href;
}

/** Native dataset labels appear before asynchronous replacement with the resident publication. */
export async function waitForAutomaticPreparation(page, labels, timeout = 120000) {
  // The old saved-map/demo checks read config as soon as a label appeared, while preparation
  // could still replace its dataset/layers. Wait for each actual outcome, including failures.
  await page.waitForFunction(
    expected => {
      const entries = Array.from(
        document.querySelectorAll('[data-testid="auto-prepare-report"] details')
      );
      return expected.every(label => {
        const entry = entries.find(element => {
          const summary = element.querySelector('summary')?.textContent ?? '';
          const sourceLabel = summary.replace(
            /: (Waiting to prepare|Preparing|Prepared|Not prepared) \(.*$/,
            ''
          );
          // Native file labels may include their extension; saved-map labels are user supplied.
          return sourceLabel === label || sourceLabel.replace(/\.[^.]+$/, '') === label;
        });
        if (!entry) return false;
        const summary = entry.querySelector('summary').textContent;
        if (summary.includes(': Not prepared (')) throw new Error(entry.textContent);
        return summary.includes(': Prepared (');
      });
    },
    labels,
    {timeout}
  );
}
