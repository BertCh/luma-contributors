// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * URL of a file in `public/`, under the path the site is served from (Vite `base`). Router links
 * get the base path from the router; plain `src` and `href` attributes need it spelled out.
 */
export function getPublicUrl(publicPath: string): string {
  return `${import.meta.env.BASE_URL}${publicPath}`;
}
