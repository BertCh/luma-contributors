// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exercise the generated control through user actions, including folded rebuild settings.
 * @param {import('playwright').Locator} control
 * @param {string | number | boolean | readonly number[] | readonly string[]} value
 */
export async function setAnalysisSheetControl(control, value) {
  for (const disclosure of await control.locator('xpath=ancestor::details').all()) {
    if (!(await disclosure.evaluate(element => element.open)))
      await disclosure.locator(':scope > summary').click();
  }
  const kind = await control.evaluate(element => ({
    tag: element.tagName,
    type: element.type,
    role: element.getAttribute('role'),
    multiple: element.multiple
  }));
  if (kind.role === 'radiogroup') {
    for (const option of await control.getByRole('radio').all()) {
      if ((await option.inputValue()) === String(value)) {
        await option.check();
        return;
      }
    }
    throw new Error(`Analysis sheet radio option is unavailable: ${value}`);
  }
  if (kind.tag === 'SELECT') {
    const selected = kind.multiple
      ? Array.isArray(value)
        ? value.map(String)
        : String(value)
            .split(',')
            .map(name => name.trim())
            .filter(Boolean)
      : String(value);
    await control.selectOption(selected);
  } else if (kind.type === 'checkbox') {
    await control.setChecked(Boolean(value));
  } else {
    await control.fill(Array.isArray(value) ? value.join(', ') : String(value));
    // Rebuild text commits on blur; filling alone intentionally leaves the old parameter active.
    await control.press('Tab');
  }
}

/**
 * Stable test IDs avoid matching both the radio group and its individually labelled options.
 * @param {import('playwright').Locator} panel
 * @param {string} name
 * @param {string | number | boolean | readonly number[] | readonly string[]} value
 * @param {{optional?: boolean}} [options]
 */
export async function setAnalysisSheetParameter(panel, name, value, {optional = false} = {}) {
  const control = panel.getByTestId(`catalog-parameter-${name}`);
  if (optional && (await control.count()) === 0) return false;
  await setAnalysisSheetControl(control, value);
  return true;
}

/**
 * Field-role controls can select one column or several; inputValue reads only the first.
 * @param {import('playwright').Locator} panel
 * @param {string} name
 */
export async function readAnalysisSheetField(panel, name) {
  return panel
    .getByTestId(`catalog-parameter-${name}`)
    .evaluate(element =>
      element.tagName === 'SELECT' && element.multiple
        ? Array.from(element.selectedOptions, option => option.value).join(',')
        : element.value
    );
}
