// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export async function closeNativeDialog(page) {
  if (await page.locator('.ReactModal__Overlay').count()) {
    await page.keyboard.press('Escape');
    await page.locator('.ReactModal__Overlay').waitFor({state: 'hidden'});
  }
}

export async function openNativeData(page, method = 'Prepare data') {
  await closeNativeDialog(page);
  await page.getByRole('button', {name: 'Add Data', exact: true}).click();
  await page.getByText(method, {exact: true}).click();
}

export async function openNativeExport(page, workspace = false) {
  await closeNativeDialog(page);
  await page.locator('#save-action').click();
  await page.getByText(workspace ? 'Export Map' : 'Export Data', {exact: true}).click();
  await page.getByRole('region', {name: workspace ? 'Save map' : 'Export data', exact: true}).waitFor();
}
