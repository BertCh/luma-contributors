// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFileRoute, Link} from '@tanstack/react-router';

// Placeholder home page. Content will follow the site content guidelines.
export const Route = createFileRoute('/')({
  component: HomePage
});

function HomePage() {
  return (
    <main className="page home">
      <h1>luma.gl contributors</h1>
      <p className="lead">GPU geospatial analysis operations built on luma.gl.</p>
      <p>
        <Link to="/docs/$" params={{_splat: ''}} className="button">
          Read the docs
        </Link>
      </p>
    </main>
  );
}
