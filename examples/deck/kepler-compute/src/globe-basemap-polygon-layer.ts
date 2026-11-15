// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SolidPolygonLayer} from '@deck.gl/layers';

/**
 * Keep deck's WGSL fill renderer and globe tessellation, with a binary-exact grid in tile space.
 * math.gl beta.8's grid cutter can round a fractional cell boundary back into its previous cell,
 * then dereference an empty cut half (`null.pos`). The z0 MVT step 1/72 reproduces this on a square.
 * Powers of two avoid that rounding in normalized tile coordinates. Always choose the next finer
 * grid, so this adaptation increases subdivision rather than relaxing globe resolution.
 */
export class GlobeBasemapPolygonLayer extends SolidPolygonLayer {
  static override layerName = 'GlobeBasemapPolygonLayer';

  override getSubdivisionStep(): number {
    const subdivisionStep = super.getSubdivisionStep();
    if (subdivisionStep <= 0) return subdivisionStep;
    const binaryStep = 2 ** Math.floor(Math.log2(subdivisionStep));
    // log2 can round up immediately below a power of two; keep the finer-grid guarantee there too.
    return binaryStep > subdivisionStep ? binaryStep / 2 : binaryStep;
  }
}
