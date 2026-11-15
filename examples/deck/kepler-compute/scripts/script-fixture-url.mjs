// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Keep the fixture required by the assertions when the driver supplies only a server URL.
 * The pre-port scripts replaced their entire default URL with that argument, losing rows=256,
 * SQL fixture installation and the native-CPU oracle's autoPrepare=0 setting. The app defaults
 * to 100,000 rows, so those scripts timed out before exercising the feature under test.
 * Unrelated query parameters, the path and the hash still come from the driver's URL.
 * @param {string | undefined} baseUrl
 * @param {Record<string, string>} parameters Required fixture parameters, overriding the URL.
 */
export function getFixtureUrl(baseUrl, parameters) {
  const fixtureUrl = new URL(baseUrl || 'http://127.0.0.1:5318/');
  for (const [name, value] of Object.entries(parameters)) fixtureUrl.searchParams.set(name, value);
  return fixtureUrl.href;
}
