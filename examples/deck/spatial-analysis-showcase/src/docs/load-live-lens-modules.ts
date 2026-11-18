// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Loads the modules used when a reader explores a recorded lens. */
export async function loadLiveLensModules() {
  const [views, lenses] = await Promise.all([
    import('@luma-contributors/lenses/computation-views'),
    import('@luma-contributors/lenses/computation-lenses'),
    import('@luma-contributors/lenses/computation-lenses/scenarios/index')
  ]);
  return {views, lenses};
}
