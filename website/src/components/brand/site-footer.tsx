// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Link} from '@tanstack/react-router';
import {GITHUB_URL} from './site-header';

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer__inner">
        <span>Gridlight · GPU geospatial analysis on luma.gl. Code licensed under MIT.</span>
        <nav className="site-footer__links" aria-label="Footer">
          <Link to="/explore">Explore</Link>
          <Link to="/how-it-works">How it works</Link>
          <Link to="/demo-data">Demo data</Link>
          <Link to="/docs/$" params={{_splat: ''}}>
            Reference
          </Link>
          <a href={GITHUB_URL}>GitHub</a>
          <a href="https://luma.gl">luma.gl</a>
        </nav>
      </div>
    </footer>
  );
}
