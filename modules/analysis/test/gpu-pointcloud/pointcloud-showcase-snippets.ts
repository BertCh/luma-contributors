// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import supportSource from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-support.scene.ts?raw';
import lidarSource from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/lidar-ground.scene.ts?raw';

/** Execute the actual displayed constructor against small caller-owned fixture views. */
export function constructPointCloudShowcaseSnippet<Result>(
  scene: 'support' | 'lidar',
  contributor: string,
  bindings: Record<string, unknown>
): Result {
  const source = scene === 'support' ? supportSource : lidarSource;
  const construction = source.match(new RegExp(`new ${contributor}\\(\\{[\\s\\S]*?\\}\\)`));
  if (!construction) throw new Error(`Missing displayed ${contributor} constructor`);
  // These are repository-authored JavaScript examples, never user-supplied expressions.
  return new Function(...Object.keys(bindings), `return ${construction[0]};`)(
    ...Object.values(bindings)
  ) as Result;
}
