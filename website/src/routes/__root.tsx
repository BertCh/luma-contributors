// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRootRoute, HeadContent, Link, Outlet, Scripts} from '@tanstack/react-router';
import type {ReactNode} from 'react';
import {SiteFooter} from '../components/brand/site-footer';
import {SiteHeader} from '../components/brand/site-header';
import {THEME_INIT_SCRIPT} from '../components/brand/theme-toggle';
import {PageHeader} from '../components/ui/page-header';
import baseCss from '../styles/base.css?url';
import componentsCss from '../styles/components.css?url';
import docsCss from '../styles/docs.css?url';
import homeCss from '../styles/home.css?url';
import tokensCss from '../styles/tokens.css?url';

export const SITE_NAME = 'Gridlight';
export const SITE_TAGLINE = 'GPU geospatial analysis on luma.gl';

const FONTS_URL =
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Space+Grotesk:wght@500;600;700&display=swap';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      {charSet: 'utf-8'},
      {name: 'viewport', content: 'width=device-width, initial-scale=1'},
      {title: `${SITE_NAME}: ${SITE_TAGLINE}`},
      {
        name: 'description',
        content:
          'Browse GPU spatial analysis operations by data type, use case and domain, each drawn live with luma.gl.'
      }
    ],
    links: [
      {rel: 'preconnect', href: 'https://fonts.googleapis.com'},
      {rel: 'preconnect', href: 'https://fonts.gstatic.com', crossOrigin: 'anonymous'},
      {rel: 'stylesheet', href: FONTS_URL},
      {rel: 'stylesheet', href: tokensCss},
      {rel: 'stylesheet', href: baseCss},
      {rel: 'stylesheet', href: componentsCss},
      {rel: 'stylesheet', href: homeCss},
      {rel: 'stylesheet', href: docsCss}
    ]
  }),
  shellComponent: RootDocument,
  component: RootLayout,
  notFoundComponent: NotFound
});

function RootDocument({children}: {children: ReactNode}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{__html: THEME_INIT_SCRIPT}} />
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

function RootLayout() {
  return (
    <>
      <SiteHeader />
      <Outlet />
      <SiteFooter />
    </>
  );
}

function NotFound() {
  return (
    <main className="page">
      <PageHeader
        eyebrow="404"
        title="Off the map"
        lede="That page does not exist."
        actions={
          <Link to="/" className="button">
            Back home
          </Link>
        }
      />
    </main>
  );
}
