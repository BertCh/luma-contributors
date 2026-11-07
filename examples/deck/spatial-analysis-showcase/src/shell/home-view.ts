// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {loadDatasetInfo} from '../data/catalog';
import {CHAPTERS} from '../scenes/chapters';
import {loadScenes} from '../scenes/registry';
import type {AnyScene} from '../scenes/scene';
import {ATLAS_ROUTES} from './atlas-routes';
import {h} from './dom';
import {getStoryHash, replaceHash} from './router';

/** Searchable field atlas for the full scene inventory. */
export async function renderHome(root: HTMLElement, focusChapter: string | null): Promise<void> {
  const loadingMarker = h('div', {class: 'page-loading'}, 'Loading the atlas…');
  root.replaceChildren(loadingMarker);
  const scenes = await loadScenes();
  const datasetInfos = new Map(
    await Promise.all(
      [...new Set(scenes.flatMap(scene => scene.datasets.map(dataset => dataset.id)))].map(
        async id => [id, await loadDatasetInfo(id)] as const
      )
    )
  );
  if (!root.isConnected || !root.contains(loadingMarker)) return;
  const parameters = new URLSearchParams(window.location.hash.split('?')[1] ?? '');
  const initialQuery = parameters.get('q') ?? '';
  const initialChapter = parameters.get('chapter') ?? focusChapter ?? '';
  const initialCapability = parameters.get('capability') ?? '';
  const search = h('input', {
    type: 'search',
    id: 'atlas-search',
    value: initialQuery,
    placeholder: 'Try a place, method, contributor or dataset ID',
    autocomplete: 'off',
    'aria-controls': 'atlas-results'
  });
  const chapter = h(
    'select',
    {id: 'atlas-chapter', 'aria-label': 'Filter by chapter'},
    h('option', {value: ''}, 'All chapters'),
    CHAPTERS.map(item => h('option', {value: item.id}, item.title))
  );
  chapter.value = initialChapter;
  const capability = h(
    'select',
    {id: 'atlas-capability', 'aria-label': 'Filter by capability'},
    h('option', {value: ''}, 'All capabilities'),
    h('option', {value: 'chart'}, 'Charts'),
    h('option', {value: 'time'}, 'Timeline'),
    h('option', {value: 'compare'}, 'Compare views')
  );
  capability.value = initialCapability;
  const count = h('p', {class: 'atlas-count', role: 'status', 'aria-live': 'polite'});
  const resetButton = h(
    'button',
    {class: 'btn btn-ghost btn-small atlas-reset', type: 'button', hidden: true},
    'Clear filters'
  );
  const resultRoot = h('div', {id: 'atlas-results', class: 'atlas-results'});
  const routeSections = ATLAS_ROUTES.map((route, index) => {
    const selected = route.stops.flatMap(([id, purpose]) => {
      const scene = scenes.find(candidate => candidate.id === id);
      return scene ? [{scene, purpose}] : [];
    });
    const routeDetails = h(
      'div',
      {class: 'atlas-route-details'},
      h(
        'span',
        {class: 'atlas-route-kicker'},
        `FIELD ROUTE · ${selected.length} ${selected.length === 1 ? 'stop' : 'stops'}`
      ),
      h('h3', {}, route.title),
      h(
        'ol',
        {class: 'atlas-route-sequence'},
        selected.map(({scene, purpose}) =>
          h('li', {}, h('a', {href: getStoryHash(scene.id)}, scene.title), h('span', {}, purpose))
        )
      )
    );
    const routeElement = h(
      'section',
      {
        class: `atlas-route ${index === 0 ? 'atlas-route-lead' : !route.previewId ? 'atlas-route-ledger' : 'atlas-route-compact'}`
      },
      routeDetails
    );
    const previewScene = scenes.find(scene => scene.id === route.previewId);
    if (previewScene) {
      const preview = h('img', {
        class: 'atlas-route-preview',
        src: `${import.meta.env.BASE_URL}images/story-previews/${route.previewId}.png`,
        alt: `Map view for ${previewScene.title}`,
        loading: 'lazy',
        decoding: 'async'
      });
      preview.addEventListener('error', () => preview.remove(), {once: true});
      const sourceTitle = previewScene.datasets
        .map(dataset => datasetInfos.get(dataset.id)?.title ?? dataset.id)
        .join(' · ');
      routeElement.append(
        h(
          'figure',
          {class: 'atlas-route-visual'},
          preview,
          h(
            'figcaption',
            {class: 'atlas-route-caption'},
            `Map preview · ${previewScene.title} · ${sourceTitle}`
          )
        )
      );
    }
    return routeElement;
  });
  const routeNav = h(
    'section',
    {class: 'atlas-routes', 'aria-labelledby': 'routes-heading'},
    h(
      'div',
      {class: 'atlas-section-heading'},
      h('p', {class: 'atlas-kicker'}, 'CURATED ROUTES'),
      h('h2', {id: 'routes-heading'}, 'Choose a line of inquiry')
    ),
    h('div', {class: 'atlas-route-grid'}, routeSections)
  );
  const page = h(
    'div',
    {class: 'page atlas-page'},
    h(
      'section',
      {class: 'atlas-masthead'},
      h('p', {class: 'atlas-kicker'}, 'LUMA.GL · FIELD ATLAS'),
      h('h1', {}, 'Spatial questions, mapped.'),
      h(
        'p',
        {class: 'atlas-intro'},
        'Explore GPU spatial analysis through short, evidence-led studies. Follow a route or search the full inventory by question, place, method or source data.'
      ),
      h(
        'div',
        {class: 'atlas-search-tools'},
        h('label', {class: 'visually-hidden', for: 'atlas-search'}, 'Search stories'),
        search,
        chapter,
        capability
      )
    ),
    routeNav,
    h(
      'section',
      {class: 'atlas-index', 'aria-labelledby': 'index-heading'},
      h(
        'div',
        {class: 'atlas-index-head'},
        h(
          'div',
          {},
          h('p', {class: 'atlas-kicker'}, 'THE INDEX'),
          h('h2', {id: 'index-heading'}, 'Browse every study')
        ),
        h('div', {class: 'atlas-index-actions'}, resetButton, count)
      ),
      resultRoot
    )
  );
  root.replaceChildren(page);

  const renderResults = () => {
    const query = search.value.trim().toLocaleLowerCase();
    const capabilityValue = capability.value;
    const filtersActive = Boolean(query || chapter.value || capabilityValue);
    routeNav.hidden = filtersActive;
    resetButton.hidden = !filtersActive;
    const matching = scenes.filter(scene => {
      if (chapter.value && scene.chapter !== chapter.value) return false;
      if (
        capabilityValue === 'chart' &&
        !(scene.readouts ?? []).some(readout => readout.kind === 'chart')
      )
        return false;
      if (capabilityValue === 'time' && !scene.timeline) return false;
      if (capabilityValue === 'compare' && !scene.story.some(step => Boolean(step.compare)))
        return false;
      if (!query) return true;
      const dataTerms = scene.datasets.flatMap(dataset => [
        dataset.id,
        dataset.role,
        datasetInfos.get(dataset.id)?.title ?? '',
        datasetInfos.get(dataset.id)?.description ?? ''
      ]);
      const chapterMetadata = CHAPTERS.find(item => item.id === scene.chapter);
      const terms = [
        scene.id,
        scene.title,
        scene.summary,
        scene.chapter,
        chapterMetadata?.title ?? '',
        chapterMetadata?.summary ?? '',
        ...scene.contributors,
        ...dataTerms
      ]
        .join(' ')
        .toLocaleLowerCase();
      return query.split(/\s+/).every(token => terms.includes(token));
    });
    count.textContent = `${matching.length} of ${scenes.length} studies`;
    const sections = CHAPTERS.map(item => {
      const entries = matching.filter(scene => scene.chapter === item.id);
      if (!entries.length) return null;
      return h(
        'section',
        {class: 'atlas-chapter', id: `chapter-${item.id}`},
        h(
          'header',
          {class: 'atlas-chapter-heading'},
          h(
            'span',
            {class: 'atlas-chapter-number'},
            String(CHAPTERS.indexOf(item) + 1).padStart(2, '0')
          ),
          h('div', {}, h('h3', {}, item.title), h('p', {}, item.summary)),
          h(
            'span',
            {class: 'atlas-chapter-count'},
            `${entries.length} ${entries.length === 1 ? 'study' : 'studies'}`
          )
        ),
        h(
          'div',
          {class: 'atlas-entries'},
          entries.map(scene => renderEntry(scene, datasetInfos))
        )
      );
    }).filter((section): section is HTMLElement => Boolean(section));
    resultRoot.replaceChildren(
      ...(sections.length
        ? sections
        : [
            h(
              'div',
              {class: 'atlas-empty'},
              h('h3', {}, 'No studies match this view'),
              h('p', {}, 'Clear the search or reset the filters to return to the full atlas.'),
              h(
                'button',
                {
                  class: 'btn',
                  type: 'button',
                  onclick: () => {
                    search.value = '';
                    chapter.value = '';
                    capability.value = '';
                    update();
                  }
                },
                'Reset search and filters'
              )
            )
          ])
    );
    const next = new URLSearchParams();
    if (search.value) next.set('q', search.value);
    if (chapter.value) next.set('chapter', chapter.value);
    if (capability.value) next.set('capability', capability.value);
    const hash = `#/${next.size ? `?${next}` : ''}`;
    replaceHash(hash);
  };
  const update = () => renderResults();
  resetButton.addEventListener('click', () => {
    search.value = '';
    chapter.value = '';
    capability.value = '';
    renderResults();
    search.focus();
  });
  search.addEventListener('input', update);
  chapter.addEventListener('change', update);
  capability.addEventListener('change', update);
  renderResults();
  if (focusChapter) document.getElementById(`chapter-${focusChapter}`)?.scrollIntoView();
}

function renderEntry(
  scene: AnyScene,
  datasetInfos: Map<string, Awaited<ReturnType<typeof loadDatasetInfo>>>
): HTMLElement {
  const datasets = scene.datasets.map(dataset => datasetInfos.get(dataset.id)?.title ?? dataset.id);
  const capabilities = [
    ...((scene.readouts ?? []).some(readout => readout.kind === 'chart') ? ['charts'] : []),
    scene.timeline ? ['timeline'] : [],
    scene.story.some(step => Boolean(step.compare)) ? ['compare'] : []
  ];
  return h(
    'a',
    {class: 'atlas-entry', href: getStoryHash(scene.id)},
    h('span', {class: 'atlas-entry-mark', 'aria-hidden': 'true'}, '↗'),
    h(
      'span',
      {class: 'atlas-entry-main'},
      h('strong', {}, scene.title),
      h('span', {class: 'atlas-entry-summary'}, scene.summary),
      h(
        'span',
        {class: 'atlas-entry-context'},
        `${datasets.join(' · ') || 'Dataset details in story'} · ${scene.contributors.join(', ')}`
      )
    ),
    h(
      'span',
      {class: 'atlas-entry-tags'},
      capabilities.map(label => h('span', {}, label))
    )
  );
}
