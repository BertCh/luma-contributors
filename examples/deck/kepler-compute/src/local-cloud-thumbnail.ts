// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Converts the Kepler save-payload thumbnail Blob to a data URL; undefined when it cannot be read. */
export async function blobToDataUrl(blob: Blob): Promise<string | undefined> {
  try {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = '';
    for (let index = 0; index < bytes.length; index += 0x8000)
      binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
    return `data:${blob.type || 'image/png'};base64,${btoa(binary)}`;
  } catch {
    return undefined;
  }
}
