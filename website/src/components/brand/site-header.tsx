// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {LogoMark, Wordmark} from './logo';
import {ThemeToggle} from './theme-toggle';

export const GITHUB_URL = 'https://github.com/BertCh/luma-contributors';

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-header__inner">
        <Link to="/" className="brand-link" aria-label="Gridlight home">
          <LogoMark />
          <Wordmark />
        </Link>
        <nav className="site-nav" aria-label="Main">
          <Link to="/explore" activeProps={{className: 'active'}}>
            Explore
          </Link>
          <Link to="/how-it-works" activeProps={{className: 'active'}}>
            <span className="site-nav__long">How it works</span>
            <span className="site-nav__short">How</span>
          </Link>
          <Link to="/docs/$" params={{_splat: ''}} activeProps={{className: 'active'}}>
            Reference
          </Link>
          <a className="site-nav__github" href={GITHUB_URL}>
            GitHub
          </a>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
