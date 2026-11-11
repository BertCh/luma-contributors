// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Lazy loading of the font atlas texels (./font-pixels.ts, about 120 KiB of source). Only pictures
// that draw text load them: the metrics that layout needs are small and load with the layout.

let pixels: Uint32Array | null = null;
let loading: Promise<Uint32Array> | null = null;

/** The atlas texels packed four to a 32-bit word (the shader reads array<u32>), once loaded. */
export function getFontPixels(): Uint32Array | null {
  return pixels;
}

/**
 * Loads the atlas texels; resolves at once when they are already loaded. A scene awaits this
 * before its first frame when it draws text, so neither a poster nor a first frame shows a picture
 * without its labels.
 */
export function loadFontPixels(): Promise<Uint32Array> {
  loading ??= import('./font-pixels').then(({FONT_PIXELS}) => {
    const binary = atob(FONT_PIXELS.data);
    const bytes = new Uint8Array(Math.ceil(binary.length / 4) * 4);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    pixels = new Uint32Array(bytes.buffer);
    return pixels;
  });
  return loading;
}
