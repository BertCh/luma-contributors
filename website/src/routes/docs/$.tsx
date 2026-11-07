// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Link, notFound} from '@tanstack/react-router';
import {getSidebar, loadDocPage, type SidebarItem} from '../../docs';
import {SITE_NAME} from '../__root';

export const Route = createFileRoute('/docs/$')({
  loader: async ({params}) => {
    const page = await loadDocPage(params._splat ?? '');
    if (!page) {
      throw notFound();
    }
    return page;
  },
  head: ({loaderData}) => ({
    meta: [{title: loaderData ? `${loaderData.title} | ${SITE_NAME}` : SITE_NAME}]
  }),
  component: DocPageView
});

const sidebar = getSidebar();

function DocPageView() {
  // The HTML is compiled at build time from repository markdown.
  const page = Route.useLoaderData();
  return (
    <div className="docs-layout">
      <aside className="docs-sidebar">
        <SidebarList items={sidebar} />
      </aside>
      <article className="docs-content" dangerouslySetInnerHTML={{__html: page.html}} />
    </div>
  );
}

function SidebarList({items}: {items: SidebarItem[]}) {
  return (
    <ul>
      {items.map(item =>
        item.type === 'page' ? (
          <li key={item.slug}>
            <Link
              to="/docs/$"
              params={{_splat: item.slug}}
              activeOptions={{exact: true}}
              activeProps={{className: 'active'}}
            >
              {item.title}
            </Link>
          </li>
        ) : (
          <li key={item.label}>
            <details open>
              <summary>{item.label}</summary>
              <SidebarList items={item.items} />
            </details>
          </li>
        )
      )}
    </ul>
  );
}
