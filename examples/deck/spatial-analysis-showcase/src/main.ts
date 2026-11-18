// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference path="./showcase-globals.d.ts" />

import './styles.css';
import {createDataCatalog} from './data/catalog';
import type {DeckHost} from './engine/deck-host';
import {h, icon} from './shell/dom';
import {renderHome} from './shell/home-view';
import {getStoryHash, navigate, startRouter, type Route} from './shell/router';
import type {StoryView} from './shell/story-view';
import {getTheme, initializeTheme, onThemeChange, toggleTheme} from './shell/theme';

const searchParameters = new URLSearchParams(window.location.search);
const forceSynthetic = searchParameters.get('data') === 'synthetic';
const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);

// `?scene=<id>` deep link for headless captures: jump to the story when no hash is given.
const sceneParameter = searchParameters.get('scene');
if (sceneParameter && (!window.location.hash || window.location.hash === '#/')) {
  history.replaceState(
    null,
    '',
    `${window.location.pathname}${window.location.search}${getStoryHash(sceneParameter)}`
  );
}

initializeTheme();
const catalog = createDataCatalog({forceSynthetic});
let host: DeckHost | null = null;

const hooks = {
  sceneId: null as string | null,
  ready: false,
  error: null as string | null,
  rebuildCount: 0,
  selectScene: (id: string) => navigate(getStoryHash(id))
};
globalThis.spatialAnalysisShowcase = hooks;

function publishState(): void {
  const state = host?.state;
  hooks.sceneId = state?.sceneId ?? null;
  hooks.ready = Boolean(state?.ready);
  hooks.error = state?.error ?? null;
  hooks.rebuildCount = state?.rebuildCount ?? 0;
  const {dataset} = document.body;
  if (hooks.sceneId) dataset['showcaseScene'] = hooks.sceneId;
  else delete dataset['showcaseScene'];
  if (hooks.ready) dataset['showcaseReady'] = 'true';
  else delete dataset['showcaseReady'];
  if (hooks.error) dataset['showcaseError'] = hooks.error;
  else delete dataset['showcaseError'];
}

// Compile-time rebuilds are counted by new compiled-graph objects; keep the test hooks current.
setInterval(() => {
  if (host?.state.sceneId) {
    host.pollRebuilds();
    publishState();
  }
}, 500);

// Shell.
const navLinks = [
  {name: 'home', label: 'Stories', href: '#/'},
  {name: 'reference', label: 'Reference', href: '#/reference'},
  {name: 'data', label: 'Data', href: '#/data'},
  {name: 'about', label: 'About', href: '#/about'}
] as const;
const nav = h(
  'nav',
  {class: 'site-nav', 'aria-label': 'Main'},
  navLinks.map(link => h('a', {href: link.href, dataset: {route: link.name}}, link.label))
);
const themeButton = h('button', {
  class: 'btn btn-ghost icon-button',
  type: 'button',
  'aria-label': 'Toggle light and dark theme'
});
const updateThemeButton = () =>
  themeButton.replaceChildren(icon(getTheme() === 'dark' ? 'sun' : 'moon', 18));
themeButton.addEventListener('click', toggleTheme);
updateThemeButton();
onThemeChange(updateThemeButton);
const view = h('main', {id: 'view', class: 'view'});
view.tabIndex = -1;
const skipLink = h('a', {class: 'skip-link', href: '#view'}, 'Skip to content');
skipLink.addEventListener('click', event => {
  event.preventDefault();
  view.focus({preventScroll: true});
  window.scrollTo(0, 0);
});
document.body.append(
  skipLink,
  h(
    'header',
    {class: 'site-header'},
    h(
      'a',
      {class: 'logo', href: '#/'},
      h('span', {class: 'logo-mark', 'aria-hidden': 'true'}, icon('layers', 20)),
      h('span', {}, 'luma.gl ', h('strong', {}, 'Spatial Analysis'))
    ),
    nav,
    h('span', {class: 'spacer'}),
    themeButton
  ),
  view
);

let storyView: StoryView | null = null;
let activeKind: Route['name'] | null = null;
let renderToken = 0;

/** Browsing the atlas does not load the GPU host or the reference lens archive. */
async function loadStoryView(): Promise<StoryView> {
  const [{DeckHost}, {StoryView}] = await Promise.all([
    import('./engine/deck-host'),
    import('./shell/story-view')
  ]);
  storyView ??= new StoryView({
    getHost: () => {
      if (!host) {
        host = new DeckHost({
          theme: getTheme(),
          initialView: {longitude: 0, latitude: 20, zoom: 2}
        });
        host.onStateChange = publishState;
      }
      return host;
    },
    catalog,
    hasWebGPU
  });
  return storyView;
}

function leaveStory(): void {
  storyView?.unmount();
  document.body.classList.remove('is-story');
  publishState();
}

startRouter(async route => {
  const token = ++renderToken;
  try {
    for (const link of nav.querySelectorAll('a')) {
      const kind =
        route.name === 'story' || route.name === 'home'
          ? 'home'
          : route.name === 'guide'
            ? 'about'
            : route.name;
      link.classList.toggle('is-active', link.dataset['route'] === kind);
      if (link.dataset['route'] === kind) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    if (route.name === 'story') {
      document.body.classList.add('is-story');
      if (activeKind === 'story' && storyView?.sceneId === route.sceneId) {
        storyView.update(route);
        return;
      }
      if (activeKind === 'story') storyView?.unmount();
      activeKind = 'story';
      view.replaceChildren(h('div', {class: 'page-loading', role: 'status'}, 'Loading the study…'));
      const loadedStoryView = await loadStoryView();
      if (token !== renderToken) return;
      await loadedStoryView.mount(view, route);
      return;
    }
    if (activeKind === 'story') leaveStory();
    activeKind = route.name;
    window.scrollTo(0, 0);
    // Late async renders can only modify their own detached route container.
    const routeView = h('div', {class: 'route-content'});
    view.replaceChildren(routeView);
    switch (route.name) {
      case 'home':
        await renderHome(
          routeView,
          new URLSearchParams(window.location.hash.split('?')[1] ?? '').get('chapter')
        );
        break;
      case 'reference': {
        const {renderReference} = await import('./shell/reference-view');
        if (token !== renderToken) return;
        await renderReference(routeView, route.entry);
        break;
      }
      case 'guide': {
        const {renderGuidePage} = await import('./shell/about-view');
        if (token !== renderToken) return;
        renderGuidePage(routeView, route.guideId);
        break;
      }
      case 'data': {
        const {renderData} = await import('./shell/data-view');
        if (token !== renderToken) return;
        await renderData(routeView);
        break;
      }
      case 'about': {
        const {renderAbout} = await import('./shell/about-view');
        if (token !== renderToken) return;
        renderAbout(routeView);
        break;
      }
    }
    if (token !== renderToken) return;
  } catch (error) {
    if (token !== renderToken) return;
    if (activeKind === 'story') leaveStory();
    activeKind = null;
    view.replaceChildren(
      h(
        'div',
        {class: 'page narrow', role: 'alert'},
        h('h1', {}, 'Unable to load this page'),
        h('p', {}, 'Reload the page or return to the atlas.'),
        h('details', {}, h('summary', {}, 'Error details'), h('p', {}, String(error))),
        h('button', {class: 'btn', on: {click: () => window.location.reload()}}, 'Reload page'),
        ' ',
        h('button', {class: 'btn btn-ghost', on: {click: () => navigate('#/')}}, 'Back to stories')
      )
    );
  }
});
