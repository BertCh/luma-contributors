// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {convertToPng} from '@kepler.gl/utils';

export const MAXIMUM_IMAGE_PIXELS = 16777216;
export const MAXIMUM_IMAGE_DIMENSION = 8192;

export function validateImageDimensions(width: number, height: number): void {
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > MAXIMUM_IMAGE_DIMENSION ||
    height > MAXIMUM_IMAGE_DIMENSION ||
    width * height > MAXIMUM_IMAGE_PIXELS
  ) {
    throw new Error('Choose an image up to 8192 pixels per side and 16 megapixels in total.');
  }
}

const EXCLUDED_CONTROLS = [
  '.maplibregl-control-container',
  '.mapboxgl-control-container',
  '.attrition-link',
  '.attrition-logo',
  '.map-control__panel-split-viewport-tools',
  '.map-navigation-control',
  '.map-popover'
].join(',');

/** Copy submitted pixels in the render task, before the browser discards its surface. */
export function captureCanvas(
  canvas: Pick<HTMLCanvasElement, 'width' | 'height' | 'toDataURL'>
): string {
  validateImageDimensions(canvas.width, canvas.height);
  const image = canvas.toDataURL('image/png');
  if (!image.startsWith('data:image/png;'))
    throw new Error('The browser could not capture this map canvas.');
  return image;
}

/** Freeze real rendered pixels and computed DOM styles before the asynchronous SVG conversion. */
export async function composeMapImage(
  source: HTMLElement,
  canvases: ReadonlyMap<HTMLCanvasElement, string>,
  width: number,
  height: number,
  isCurrent: () => boolean,
  escapeXhtmlForWebpack?: boolean
): Promise<string> {
  validateImageDimensions(width, height);
  let nodes = 0;
  function copyNode(node: Node): Node | null {
    if (++nodes > 20000) throw new Error('The map overlays exceed the image export size limit.');
    if (node instanceof Element && node.matches(EXCLUDED_CONTROLS)) return null;
    if (!(node instanceof Element)) return node.cloneNode(false);
    const style = getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') return null;
    let copy: Node;
    if (node instanceof HTMLCanvasElement) {
      const pixels = canvases.get(node);
      if (!pixels) throw new Error('A map canvas has not finished rendering. Retry image export.');
      const image = document.createElement('img');
      image.src = pixels;
      image.width = node.width;
      image.height = node.height;
      copy = image;
    } else {
      copy = node.cloneNode(false);
      for (const child of node.childNodes) {
        const childCopy = copyNode(child);
        if (childCopy) copy.appendChild(childCopy);
      }
    }
    if (copy instanceof HTMLElement || copy instanceof SVGElement) {
      for (const property of style)
        copy.style.setProperty(property, style.getPropertyValue(property));
      // Exported overlays are inert and never execute copied event handlers.
      for (const attribute of copy.attributes) {
        if (attribute.name.startsWith('on')) copy.removeAttribute(attribute.name);
      }
    }
    return copy;
  }
  const copy = copyNode(source);
  if (!(copy instanceof HTMLElement)) throw new Error('The map is unavailable for image export.');
  copy.style.position = 'relative';
  copy.style.left = '0';
  copy.style.top = '0';
  copy.style.width = `${width}px`;
  copy.style.height = `${height}px`;
  const holder = document.createElement('div');
  holder.style.cssText = 'position:fixed;left:-100000px;top:0;pointer-events:none;';
  holder.setAttribute('aria-hidden', 'true');
  holder.appendChild(copy);
  document.body.appendChild(holder);
  try {
    await Promise.all(Array.from(copy.querySelectorAll('img'), image => image.decode()));
    if (!isCurrent()) throw new Error('Image export was cancelled.');
    const image = await convertToPng(copy, {width, height, escapeXhtmlForWebpack});
    if (typeof image !== 'string' || !image.startsWith('data:image/png;'))
      throw new Error('The browser could not compose the map image.');
    return image;
  } finally {
    holder.remove();
  }
}
