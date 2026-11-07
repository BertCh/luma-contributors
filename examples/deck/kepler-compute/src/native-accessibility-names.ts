// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Host accessibility patch for Kepler 3.3.0-alpha.18 controls that ship without names or keyboard
 * support. Like the basemap style input, it watches the document with a MutationObserver and
 * annotates the native DOM in place; it never changes Kepler behavior, only exposes it:
 * - layer/group header icons (svg with a tooltip id such as `tooltip.moreOptions_<layerId>`) get
 *   role=button and an aria-label derived from that tooltip key;
 * - the layer title input gets an aria-label;
 * - header dropdown entries (Export Map, Export Data) become focusable buttons;
 * - item selector dropdowns (field pickers) become focusable buttons;
 * - the dataset picker popup of Add Filter/Add Layer takes focus when it opens;
 * - the Add Filter button is put back in the Tab order;
 * - side panel tabs, the Share/Save header action and the Add Data "browse your files" control become focusable buttons that
 *   activate with Enter or Space.
 */

const KEYBOARD_CONTROL_ATTRIBUTE = 'data-luma-keyboard-control';

/** `tooltip.moreOptions_resident-points` -> `More options`. */
export function getTooltipKeyLabel(tooltipId: string): string {
  const key = tooltipId.split('_')[0].replace(/^tooltip\./, '');
  const words = key.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Visible text of a native control, without the inline <style> that react-tooltip appends. */
function getVisibleText(element: Element): string {
  let text = '';
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode())
    if (node.parentElement?.tagName.toLowerCase() !== 'style') text += node.textContent ?? '';
  return text.replace(/\s+/g, ' ').trim();
}

function makeKeyboardControl(element: Element, role: string, label?: string): void {
  if (!element.hasAttribute('role')) element.setAttribute('role', role);
  if (label && !element.hasAttribute('aria-label')) element.setAttribute('aria-label', label);
  if (!element.hasAttribute('tabindex')) element.setAttribute('tabindex', '0');
  element.setAttribute(KEYBOARD_CONTROL_ATTRIBUTE, 'true');
}

/** Annotates the native controls currently in the document. */
export function annotateNativeControls(root: ParentNode = document): void {
  for (const icon of root.querySelectorAll('svg[data-tip][data-for]:not([aria-label])')) {
    const label = getTooltipKeyLabel(icon.getAttribute('data-for') ?? '');
    if (label) makeKeyboardControl(icon, 'button', label);
  }
  for (const input of root.querySelectorAll(
    'input.layer__title__editor:not([aria-label]), input[data-testid="layer__title__editor"]:not([aria-label])'
  ))
    input.setAttribute('aria-label', 'Layer name');
  for (const tab of root.querySelectorAll('.side-panel__tab, div.side-panel__panel-header__action'))
    makeKeyboardControl(tab, 'button', getVisibleText(tab));
  // Kepler removes the Add Filter button from the Tab order (tabindex=-1).
  for (const button of root.querySelectorAll('button.add-filter-button[tabindex="-1"]'))
    button.setAttribute('tabindex', '0');
  // Header dropdown entries (Share -> Export Map, Export Data, Save) are clickable divs.
  for (const item of root.querySelectorAll('.panel-header-dropdown__inner .toolbar-item'))
    makeKeyboardControl(
      item,
      'button',
      getVisibleText(item.querySelector('.toolbar-item__title') ?? item)
    );
  // Kepler's item selector (field pickers) is a clickable div; expose it as a button.
  for (const selector of root.querySelectorAll('.item-selector__dropdown'))
    makeKeyboardControl(selector, 'button', getVisibleText(selector) || 'Select');
  // Interactive Tippy popups (the Add Filter/Add Layer dataset picker) are appended away from their
  // button and never receive focus; move it to the popup's search input when the popup opens.
  for (const input of root.querySelectorAll<HTMLInputElement>(
    '.add-layer-menu-dropdown .typeahead__input:not([data-luma-focused])'
  )) {
    input.setAttribute('data-luma-focused', 'true');
    input.focus();
  }
  // Kepler puts the click handler on the inner span and keeps the real file input hidden.
  for (const upload of root.querySelectorAll('.upload-button > .file-upload__upload-button-span'))
    makeKeyboardControl(upload, 'button');
}

/** Starts watching the document; returns a function that stops watching. */
export function installNativeAccessibilityNames(): () => void {
  let scheduled = false;
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      annotateNativeControls();
    });
  };
  const onKeyDown = (event: KeyboardEvent) => {
    const target = event.target;
    if (
      (event.key === 'Enter' || event.key === ' ') &&
      target instanceof Element &&
      target.hasAttribute(KEYBOARD_CONTROL_ATTRIBUTE)
    ) {
      event.preventDefault();
      target.dispatchEvent(new MouseEvent('click', {bubbles: true}));
    }
  };
  const observer = new MutationObserver(schedule);
  observer.observe(document.body, {childList: true, subtree: true});
  document.addEventListener('keydown', onKeyDown);
  annotateNativeControls();
  return () => {
    observer.disconnect();
    document.removeEventListener('keydown', onKeyDown);
  };
}
