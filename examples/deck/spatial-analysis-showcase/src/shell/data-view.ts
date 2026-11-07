// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './data-view.css';
import {loadDatasetInfos} from '../data/catalog';
import {findScenesByDataset} from '../scenes/registry';
import {formatBytes, h, icon} from './dom';
import {getStoryHash, replaceHash} from './router';

/** Searchable source catalog; descriptions retain each dataset's sampling and interpretation limits. */
export async function renderData(root: HTMLElement): Promise<void> {
  root.replaceChildren(h('div', {class: 'page-loading'}, 'Loading data catalog…'));
  const infos = (await loadDatasetInfos()).sort((first, second) =>
    first.title.localeCompare(second.title)
  );
  const usage = await Promise.all(infos.map(info => findScenesByDataset(info.id)));
  if (!root.isConnected) return;
  const parameters = new URLSearchParams(window.location.hash.split('?')[1] ?? '');
  const search = h('input', {
    id: 'data-search',
    type: 'search',
    placeholder: 'Search places, measurements, sources or licences…',
    value: parameters.get('q') ?? '',
    'aria-controls': 'data-results'
  });
  const usageFilter = h(
    'select',
    {id: 'data-usage', 'aria-controls': 'data-results'},
    h('option', {value: ''}, 'All datasets'),
    h('option', {value: 'used'}, 'Used in stories'),
    h('option', {value: 'unused'}, 'Available for new stories')
  );
  const selectedUsage = parameters.get('usage') ?? '';
  usageFilter.value = ['used', 'unused'].includes(selectedUsage) ? selectedUsage : '';
  const count = h('p', {class: 'muted small', role: 'status', 'aria-live': 'polite'});
  const results = h('div', {id: 'data-results', class: 'dataset-grid'});
  const reset = h('button', {class: 'btn btn-ghost', type: 'button'}, 'Clear filters');
  const total = infos.reduce((sum, info) => sum + info.approxBytes, 0);
  root.replaceChildren(
    h(
      'div',
      {class: 'page dataset-catalog'},
      h('p', {class: 'dataset-eyebrow'}, 'THE EVIDENCE LIBRARY'),
      h('h1', {}, 'Start with the source.'),
      h(
        'p',
        {class: 'lede'},
        `${infos.length} datasets connect these investigations to observations of the world. Explore the source, licence, coverage and studies before interpreting a result.`
      ),
      h(
        'p',
        {class: 'muted small'},
        `About ${formatBytes(total)} across the catalog; individual stories load the data they need. Descriptions distinguish observations, derived products and modeled scenarios.`
      ),
      h(
        'div',
        {class: 'dataset-tools'},
        h('div', {}, h('label', {for: 'data-search'}, 'Find source data'), search),
        h('div', {}, h('label', {for: 'data-usage'}, 'Story use'), usageFilter),
        reset
      ),
      count,
      results
    )
  );

  function renderResults(): void {
    const query = search.value.trim().toLocaleLowerCase();
    const selected = infos.flatMap((info, index) => {
      const stories = usage[index];
      const searchable = [
        info.id,
        info.title,
        info.description,
        info.license,
        info.attribution,
        ...stories.map(scene => scene.title)
      ]
        .join(' ')
        .toLocaleLowerCase();
      if (query && !searchable.includes(query)) return [];
      if (usageFilter.value === 'used' && !stories.length) return [];
      if (usageFilter.value === 'unused' && stories.length) return [];
      return [{info, stories}];
    });
    count.textContent = `${selected.length} of ${infos.length} datasets`;
    results.replaceChildren(
      ...selected.map(({info, stories}) =>
        h(
          'article',
          {class: 'dataset-card'},
          h(
            'div',
            {class: 'dataset-card-meta'},
            h('span', {}, info.license),
            h('span', {}, `~${formatBytes(info.approxBytes)}`)
          ),
          h('h2', {}, info.title),
          h('p', {}, info.description),
          h(
            'details',
            {class: 'dataset-source'},
            h('summary', {}, 'Attribution and geographic coverage'),
            h('p', {}, info.attribution),
            h('p', {class: 'small mono'}, info.id),
            h(
              'dl',
              {class: 'dataset-bounds'},
              ['West', 'South', 'East', 'North'].map((label, index) =>
                h('div', {}, h('dt', {}, label), h('dd', {}, `${info.bbox[index].toFixed(4)}°`))
              )
            )
          ),
          h(
            'a',
            {
              class: 'dataset-source-link',
              href: info.sourceUrl,
              target: '_blank',
              rel: 'noreferrer noopener'
            },
            'Read the source',
            icon('external', 13)
          ),
          h(
            'div',
            {class: 'dataset-studies'},
            h(
              'p',
              {class: 'small muted'},
              stories.length
                ? `${stories.length} ${stories.length === 1 ? 'investigation' : 'investigations'} using this source`
                : 'Available for a new investigation'
            ),
            stories.length
              ? h(
                  'ul',
                  {},
                  stories.map(scene =>
                    h('li', {}, h('a', {href: getStoryHash(scene.id)}, scene.title))
                  )
                )
              : null
          )
        )
      )
    );
    if (!selected.length)
      results.append(
        h(
          'p',
          {class: 'dataset-empty'},
          'No sources match these filters. Try a place, measurement or licence, or clear the filters.'
        )
      );
    const next = new URLSearchParams();
    if (search.value) next.set('q', search.value);
    if (usageFilter.value) next.set('usage', usageFilter.value);
    replaceHash(`#/data${next.size ? `?${next}` : ''}`);
  }
  search.addEventListener('input', renderResults);
  usageFilter.addEventListener('change', renderResults);
  reset.addEventListener('click', () => {
    search.value = '';
    usageFilter.value = '';
    renderResults();
    search.focus();
  });
  renderResults();
}
