// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {DynamicTexture} from '@luma.gl/core/dynamic';
import type {FontAtlas} from '@luma.gl/engine/text';

export type KeplerMipmappedTextResourcesProps = {
  /** Normalized font atlas whose pages become the layers of one 2d-array texture. */
  fontAtlas: FontAtlas;
  /** Debug label used for the atlas texture. */
  id?: string;
};

/**
 * Atlas resources for point labels with a full trilinear mip chain.
 *
 * Published `GPUTextResources` uploads one mip level and samples it without a mipmap filter, so a
 * 64 px bitmap atlas minified to label sizes aliases and paints about one pixel narrower than the
 * native deck.gl TextLayer, which samples a mipmapped atlas. This helper has the same surface the
 * label layer reads (`device`, `fontAtlas`, `atlasTexture`, `destroy`) and differs only in the
 * texture: `mipmaps` with `mipLevels: 'auto'` and a linear mipmap filter. Retire it when
 * `GPUTextResourcesProps` accepts mipmaps (dev-docs/upstream-proposals.md).
 */
export class KeplerMipmappedTextResources {
  readonly device: Device;
  readonly fontAtlas: FontAtlas;
  readonly atlasTexture: DynamicTexture;
  private destroyed = false;

  constructor(device: Device, props: KeplerMipmappedTextResourcesProps) {
    this.device = device;
    // Published atlases are cached per font and their canvas grows in place when another runtime
    // adds characters, while DynamicTexture reads the canvas asynchronously. Snapshot the pages in
    // the same synchronous turn that validated the dimensions and admitted the memory.
    this.fontAtlas = snapshotKeplerFontAtlas(props.fontAtlas);
    this.atlasTexture = new DynamicTexture(device, {
      id: `${props.id || 'text'}-atlas`,
      dimension: '2d-array',
      sampler: {minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear'},
      mipmaps: true,
      mipLevels: 'auto',
      data: [...this.fontAtlas.pages]
    });
  }

  /** Releases the atlas texture. Idempotent. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.atlasTexture.destroy();
  }
}

/** Bytes of a full RGBA8 mip chain for every page of an atlas (matches the uploaded texture). */
export function getKeplerMipChainByteLength(fontAtlas: FontAtlas): number {
  let byteLength = 0;
  for (let width = fontAtlas.width, height = fontAtlas.height; ; ) {
    byteLength += width * height * 4 * fontAtlas.pages.length;
    if (width === 1 && height === 1) break;
    width = Math.max(1, Math.floor(width / 2));
    height = Math.max(1, Math.floor(height / 2));
  }
  return byteLength;
}

type KeplerSnapshotCanvas = {
  width: number;
  height: number;
  getContext(kind: '2d'): {drawImage(source: unknown, x: number, y: number): void} | null;
};

/**
 * Copies every page of an atlas into a private canvas of exactly the atlas width and height, so
 * later in-place growth of a shared cached canvas cannot change what is uploaded. Pages that are
 * not canvases, or environments without a canvas factory, are returned unchanged.
 */
export function snapshotKeplerFontAtlas(
  fontAtlas: FontAtlas,
  createCanvas: (width: number, height: number) => KeplerSnapshotCanvas | null = createKeplerCanvas
): FontAtlas {
  const pages = fontAtlas.pages.map(page => {
    const source = page as unknown as Partial<KeplerSnapshotCanvas>;
    if (typeof source.getContext !== 'function') return page;
    const copy = createCanvas(fontAtlas.width, fontAtlas.height);
    const context = copy?.getContext('2d');
    if (!copy || !context) return page;
    context.drawImage(page, 0, 0);
    return copy as unknown as typeof page;
  });
  return {...fontAtlas, pages};
}

function createKeplerCanvas(width: number, height: number): KeplerSnapshotCanvas | null {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  return null;
}
