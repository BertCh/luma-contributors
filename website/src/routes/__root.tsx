// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRootRoute, HeadContent, Link, Outlet, Scripts} from '@tanstack/react-router';
import type {ReactNode} from 'react';
import appCss from '../styles.css?url';

export const SITE_NAME = 'luma.gl contributors';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      {charSet: 'utf-8'},
      {name: 'viewport', content: 'width=device-width, initial-scale=1'},
      {title: SITE_NAME}
    ],
    links: [{rel: 'stylesheet', href: appCss}]
  }),
  shellComponent: RootDocument,
  component: RootLayout,
  notFoundComponent: NotFound
});

function RootDocument({children}: {children: ReactNode}) {
  return (
    <html lang="en">
      <head>
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
      <header className="site-header">
        <Link to="/" className="site-title">
          {SITE_NAME}
        </Link>
        <nav className="site-nav">
          <Link to="/docs/$" params={{_splat: ''}} activeProps={{className: 'active'}}>
            Docs
          </Link>
          <a href="https://github.com/BertCh/luma-contributors">GitHub</a>
        </nav>
      </header>
      <Outlet />
    </>
  );
}

function NotFound() {
  return (
    <main className="page">
      <h1>Page not found</h1>
      <p>
        <Link to="/">Back to the home page</Link>
      </p>
    </main>
  );
}
