// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Adds an "External style URL" input to Kepler's native map-style panel (`.map-style-panel`).
 *
 * Kepler renders the panel itself, so this module watches the document and appends a small form
 * to every panel instance that does not have one yet. Submitting validates the URL through the
 * host basemap context and adds it to the native picker (`basemap-picker.ts`); failures are
 * shown beside the field and the local background stays.
 */

export const BASEMAP_STYLE_INPUT_TEST_ID = 'basemap-style-url-input';

export type BasemapStyleUrlInputOptions = {
  addStyle(url: string): Promise<{status: string; errorMessage?: string | null}>;
};

function makeFormElement(options: BasemapStyleUrlInputOptions): HTMLFormElement {
  const form = document.createElement('form');
  form.dataset['testid'] = 'basemap-style-url-form';
  form.style.cssText = 'display:flex;flex-direction:column;gap:6px;padding:8px 0;font-size:11px';

  const label = document.createElement('label');
  label.textContent = 'External style URL';
  label.style.cssText = 'color:#a0a7b4;text-transform:uppercase;letter-spacing:.4px';

  const input = document.createElement('input');
  input.type = 'url';
  input.placeholder = 'https://example.com/style.json';
  input.dataset['testid'] = BASEMAP_STYLE_INPUT_TEST_ID;
  input.setAttribute('aria-label', 'External map style URL');
  input.style.cssText =
    'width:100%;box-sizing:border-box;padding:6px 8px;background:#29323c;color:#fff;border:1px solid #3a414c;border-radius:2px';

  const button = document.createElement('button');
  button.type = 'submit';
  button.textContent = 'Add style';
  button.dataset['testid'] = 'basemap-style-url-add';
  button.style.cssText =
    'align-self:flex-start;padding:5px 10px;background:#0f9668;color:#fff;border:0;border-radius:2px;cursor:pointer';

  const message = document.createElement('div');
  message.dataset['testid'] = 'basemap-style-url-message';
  message.setAttribute('role', 'status');
  message.style.cssText = 'min-height:14px;color:#a0a7b4';

  form.append(label, input, button, message);
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const url = input.value.trim();
    if (!url) return;
    button.disabled = true;
    message.textContent = 'Validating style...';
    message.style.color = '#a0a7b4';
    try {
      const state = await options.addStyle(url);
      const isFailure = state.status === 'failed';
      message.textContent = isFailure
        ? (state.errorMessage ?? 'Style failed to load')
        : 'Style added';
      message.style.color = isFailure ? '#f88' : '#6fd39c';
      if (!isFailure) input.value = '';
    } catch (error) {
      message.textContent = error instanceof Error ? error.message : String(error);
      message.style.color = '#f88';
    } finally {
      button.disabled = false;
    }
  });
  return form;
}

/** Starts watching the document; returns a function that stops watching and removes the forms. */
export function installBasemapStyleUrlInput(options: BasemapStyleUrlInputOptions): () => void {
  const ensureForms = () => {
    for (const panel of document.querySelectorAll('.map-style-panel')) {
      if (!panel.querySelector('[data-testid="basemap-style-url-form"]'))
        panel.append(makeFormElement(options));
    }
  };
  const observer = new MutationObserver(ensureForms);
  observer.observe(document.body, {childList: true, subtree: true});
  ensureForms();
  return () => {
    observer.disconnect();
    for (const form of document.querySelectorAll('[data-testid="basemap-style-url-form"]'))
      form.remove();
  };
}
