// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewerFilterSpec} from './standalone-viewer-config';
import type {ViewerDataset} from './standalone-viewer-data';
import type {ViewerBuildResult, ViewerLegendEntry} from './standalone-viewer-layers';
import {isActiveFilter} from './standalone-viewer-filters';

/** DOM panels of the offline viewer: legends, filter controls and skipped-layer notices. */

const toCss = (color: readonly number[]) => `rgb(${color[0]},${color[1]},${color[2]})`;

function createElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  text?: string,
  className?: string
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toPrecision(4);
}

function renderLegendEntry(entry: ViewerLegendEntry): HTMLElement {
  const block = createElement('div', undefined, 'entry');
  block.appendChild(createElement('div', entry.title, 'title'));
  if (entry.kind === 'constant') {
    const swatch = createElement('span', undefined, 'swatch');
    swatch.style.background = toCss(entry.color);
    block.appendChild(swatch);
  } else if (entry.kind === 'ramp') {
    const ramp = createElement('div', undefined, 'ramp');
    ramp.style.background = `linear-gradient(to right, ${entry.colors.map(toCss).join(',')})`;
    block.appendChild(ramp);
    if (entry.domain.length >= 2) {
      const labels = createElement('div', undefined, 'labels');
      labels.append(
        createElement('span', formatNumber(entry.domain[0])),
        createElement('span', formatNumber(entry.domain[entry.domain.length - 1]))
      );
      block.appendChild(labels);
    }
  } else if (entry.kind === 'categories') {
    for (const item of entry.items) {
      const row = createElement('div', undefined, 'category');
      const swatch = createElement('span', undefined, 'swatch');
      swatch.style.background = toCss(item.color);
      row.append(swatch, createElement('span', item.label));
      block.appendChild(row);
    }
  } else {
    block.appendChild(
      createElement(
        'div',
        entry.ticks
          .map(tick => `${formatNumber(tick.value)} → ${formatNumber(tick.size)}`)
          .join(', '),
        'labels'
      )
    );
  }
  return block;
}

export function renderLegends(container: HTMLElement, result: ViewerBuildResult): void {
  container.replaceChildren();
  for (const legend of result.legends) {
    const section = createElement('section', undefined, 'legend');
    section.appendChild(createElement('h4', legend.label));
    for (const entry of legend.entries) section.appendChild(renderLegendEntry(entry));
    container.appendChild(section);
  }
  for (const skipped of result.skipped) {
    const section = createElement('section', undefined, 'legend skipped');
    section.append(
      createElement('h4', skipped.label),
      createElement('div', `Not drawn: ${skipped.reason}`, 'labels')
    );
    container.appendChild(section);
  }
}

/** Interactive controls for range, timeRange, select and multiSelect filters; edits the specs in place. */
export function renderFilterControls(
  container: HTMLElement,
  filters: ViewerFilterSpec[],
  datasets: ReadonlyMap<string, ViewerDataset>,
  onChange: () => void
): void {
  container.replaceChildren();
  for (const filter of filters) {
    const dataset = datasets.get(filter.datasetIds[0]);
    const fieldName = filter.fieldNames[0];
    if (!dataset || !fieldName || filter.type === 'polygon') continue;
    const section = createElement('section', undefined, 'legend');
    section.appendChild(createElement('h4', `Filter: ${fieldName}`));
    if (filter.type === 'multiSelect') {
      const values = Array.from(new Set(Array.from(dataset.readStrings(fieldName) ?? [])))
        .filter((value): value is string => value !== null)
        .sort();
      const selected = new Set(Array.isArray(filter.value) ? (filter.value as string[]) : []);
      for (const value of values.slice(0, 40)) {
        const label = createElement('label', undefined, 'category');
        const box = createElement('input');
        box.type = 'checkbox';
        box.checked = selected.has(value);
        box.onchange = () => {
          if (box.checked) selected.add(value);
          else selected.delete(value);
          filter.value = Array.from(selected);
          onChange();
        };
        label.append(box, createElement('span', value));
        section.appendChild(label);
      }
    } else if (filter.type === 'select') {
      const box = createElement('input');
      box.type = 'checkbox';
      box.checked = filter.value === true;
      box.onchange = () => {
        filter.value = box.checked;
        onChange();
      };
      section.appendChild(box);
    } else {
      const numbers = dataset.readNumbers(fieldName);
      if (!numbers) continue;
      let minimum = Infinity;
      let maximum = -Infinity;
      for (let row = 0; row < numbers.length; row++) {
        if (numbers[row] < minimum) minimum = numbers[row];
        if (numbers[row] > maximum) maximum = numbers[row];
      }
      if (!(minimum <= maximum)) continue;
      const current: [number, number] = isActiveFilter(filter)
        ? [...(filter.value as [number, number])]
        : [minimum, maximum];
      const readout = createElement('div', undefined, 'labels');
      const format = (value: number) =>
        filter.type === 'timeRange'
          ? new Date(value).toISOString().slice(0, 19)
          : formatNumber(value);
      const update = () => {
        readout.textContent = `${format(current[0])} – ${format(current[1])}`;
      };
      const makeSlider = (index: 0 | 1) => {
        const slider = createElement('input');
        slider.type = 'range';
        slider.min = String(minimum);
        slider.max = String(maximum);
        slider.step = String((maximum - minimum) / 1000 || 1);
        slider.value = String(current[index]);
        slider.oninput = () => {
          current[index] = Number(slider.value);
          if (current[0] > current[1]) current[index === 0 ? 1 : 0] = current[index];
          filter.value = [...current];
          update();
          onChange();
        };
        return slider;
      };
      update();
      section.append(makeSlider(0), makeSlider(1), readout);
    }
    container.appendChild(section);
  }
}
