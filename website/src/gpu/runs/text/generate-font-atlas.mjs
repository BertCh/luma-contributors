// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Generates the label font of picture text: ./font-metrics.ts (advances, kerning, atlas cells,
// loaded with the layout) and ./font-pixels.ts (the signed distance field atlas, loaded lazily).
// Run it again only to change the character set or the atlas size (from the repository root):
//
//   node website/src/gpu/runs/text/generate-font-atlas.mjs
//   npx biome check --write website/src/gpu/runs/text/font-metrics.ts website/src/gpu/runs/text/font-pixels.ts
//
// The glyphs come from KaTeX_SansSerif-Regular, with symbols it lacks from KaTeX_Main-Regular, in
// the `katex` package (declare it as a devDependency of the website workspace). The bytes are read
// from node_modules and handed to headless Chromium as an ArrayBuffer: nothing is fetched. Chromium
// rasterizes each glyph at four times the atlas resolution; the distance field is computed here
// with an exact Euclidean distance transform and sampled down.
//
// Licence. The KaTeX fonts are Font Software under the SIL Open Font License 1.1, with Reserved
// Font Names KaTeX_SansSerif and KaTeX_Main (name table records 0, 13 and 14 of each font). The
// atlas subsets the fonts and changes their format, which the OFL defines as a Modified Version.
// So the generated files are OFL-1.1 only (condition 5), carry the copyright notice and the full
// licence both as a header and as data that survives bundling (condition 2), and present the
// result under its own name, FONT_NAME below, never a Reserved Font Name (condition 3). The KaTeX
// fonts are acknowledged as the source, which condition 4 allows. Pictures drawn with the atlas
// (posters, screenshots) are documents created with the font and are not covered by the OFL.
// This script itself is MIT, like the rest of the website code.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(directory, '../../../../..');
const require = createRequire(path.join(repositoryRoot, 'package.json'));
const {chromium} = require('playwright');

/** Atlas pixels per em. */
const EM = 24;
/** Distance field reach on each side of the outline, in atlas pixels. */
const SPREAD = 6;
/** Name of the Modified Version: the OFL forbids the Reserved Font Names for it. */
const FONT_NAME = 'Gridlight Label Atlas';
const SANS_FILE = 'KaTeX_SansSerif-Regular.ttf';
const MAIN_FILE = 'KaTeX_Main-Regular.ttf';
/** Rasterization resolution relative to the atlas. */
const SUPERSAMPLE = 4;
const ATLAS_WIDTH = 512;

const CHARACTERS = [
  ...Array.from({length: 95}, (_, index) => String.fromCharCode(32 + index)),
  ...'°–—−×±·⋅…²³µμ′≤≥→←↑↓ΔΣσρλαβθχ√∞≈≠'
];

// The text of SIL OFL 1.1, which condition 2 requires with every copy of the Font Software.
const OFL_TEXT = `SIL OPEN FONT LICENSE

Version 1.1 - 26 February 2007

PREAMBLE

The goals of the Open Font License (OFL) are to stimulate worldwide development of collaborative font projects, to support the font creation efforts of academic and linguistic communities, and to provide a free and open framework in which fonts may be shared and improved in partnership with others.

The OFL allows the licensed fonts to be used, studied, modified and redistributed freely as long as they are not sold by themselves. The fonts, including any derivative works, can be bundled, embedded, redistributed and/or sold with any software provided that any reserved names are not used by derivative works. The fonts and derivatives, however, cannot be released under any other type of license. The requirement for fonts to remain under this license does not apply to any document created using the fonts or their derivatives.

DEFINITIONS

"Font Software" refers to the set of files released by the Copyright Holder(s) under this license and clearly marked as such. This may include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the copyright statement(s).

"Original Version" refers to the collection of Font Software components as distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting, or substituting — in part or in whole — any of the components of the Original Version, by changing formats or by porting the Font Software to a new environment.

"Author" refers to any designer, engineer, programmer, technical writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS

Permission is hereby granted, free of charge, to any person obtaining a copy of the Font Software, to use, study, copy, merge, embed, modify, redistribute, and sell modified and unmodified copies of the Font Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components, in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled, redistributed and/or sold with any software, provided that each copy contains the above copyright notice and this license. These can be included either as stand-alone text files, human-readable headers or in the appropriate machine-readable metadata fields within text or binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font Name(s) unless explicit written permission is granted by the corresponding Copyright Holder. This restriction only applies to the primary font name as presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font Software shall not be used to promote, endorse or advertise any Modified Version, except to acknowledge the contribution(s) of the Copyright Holder(s) and the Author(s) or with their explicit written permission.

5) The Font Software, modified or unmodified, in part or in whole, must be distributed entirely under this license, and must not be distributed under any other license. The requirement for fonts to remain under this license does not apply to any document created using the Font Software.

TERMINATION

This license becomes null and void if any of the above conditions are not met.

DISCLAIMER

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.`;

const katexDirectory = path.join(repositoryRoot, 'node_modules/katex');
const fontDirectory = path.join(katexDirectory, 'dist/fonts');
for (const file of [SANS_FILE, MAIN_FILE]) {
  if (!fs.existsSync(path.join(fontDirectory, file))) {
    console.error(
      `generate-font-atlas: ${path.join(fontDirectory, file)} is missing. The label font comes ` +
        'from the katex package: add it as a devDependency of the website workspace ' +
        '("katex": "0.16.47" in website/package.json), run yarn install, then run this again.'
    );
    process.exit(1);
  }
}
const sansBytes = fs.readFileSync(path.join(fontDirectory, SANS_FILE));
const mainBytes = fs.readFileSync(path.join(fontDirectory, MAIN_FILE));
const katexVersion = JSON.parse(
  fs.readFileSync(path.join(katexDirectory, 'package.json'), 'utf8')
).version;

/** The font's own name records (0 copyright, 1 family, 5 version, 13 licence, 14 licence URL). */
function readNames(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tableCount = view.getUint16(4);
  const names = {};
  for (let table = 0; table < tableCount; table++) {
    const record = 12 + table * 16;
    if (bytes.toString('latin1', record, record + 4) !== 'name') continue;
    const offset = view.getUint32(record + 8);
    const count = view.getUint16(offset + 2);
    const strings = offset + view.getUint16(offset + 4);
    for (let index = 0; index < count; index++) {
      const entry = offset + 6 + index * 12;
      const platform = view.getUint16(entry);
      const id = view.getUint16(entry + 6);
      const length = view.getUint16(entry + 8);
      const start = strings + view.getUint16(entry + 10);
      if (names[id] !== undefined) continue;
      const raw = bytes.subarray(start, start + length);
      names[id] =
        platform === 1
          ? raw.toString('latin1')
          : String.fromCharCode(
              ...Array.from({length: length / 2}, (_, char) => raw.readUInt16BE(char * 2))
            );
    }
  }
  return names;
}

const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const SOURCES = [
  {file: SANS_FILE, bytes: sansBytes},
  {file: MAIN_FILE, bytes: mainBytes}
].map(({file, bytes}) => {
  const names = readNames(bytes);
  if (!/Open Font License/.test(names[13] ?? '')) {
    throw new Error(`${file}: expected an SIL Open Font License notice in name record 13`);
  }
  return {file, version: names[5], copyright: names[0], license: names[13], sha256: sha256(bytes)};
});
// Copyright holders, one per line of the copyright records, without duplicates.
const HOLDERS = [
  ...new Set(SOURCES.flatMap(source => source.copyright.split('\n').map(line => line.trim())))
].filter(Boolean);

/** Runs in the page: rasterizes every character and measures kerning. */
async function rasterize({sans, main, characters, size}) {
  const toBuffer = values => new Uint8Array(values).buffer;
  const sansFace = new FontFace('AtlasSans', toBuffer(sans));
  const mainFace = new FontFace('AtlasMain', toBuffer(main));
  document.fonts.add(await sansFace.load());
  document.fonts.add(await mainFace.load());
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d', {willReadFrequently: true});
  const fontOf = fallback => `${size}px AtlasSans, AtlasMain, ${fallback}`;
  const draw = (character, fallback, width, height, originX, baseline) => {
    canvas.width = width;
    canvas.height = height;
    context.clearRect(0, 0, width, height);
    context.font = fontOf(fallback);
    context.fillStyle = '#fff';
    context.textBaseline = 'alphabetic';
    context.fillText(character, originX, baseline);
    const pixels = context.getImageData(0, 0, width, height).data;
    const alpha = new Array(width * height);
    for (let index = 0; index < width * height; index++) alpha[index] = pixels[index * 4 + 3];
    return alpha;
  };
  context.font = fontOf('serif');
  const fontMetrics = context.measureText('Hxg');
  const glyphs = [];
  const missing = [];
  for (const character of characters) {
    context.font = fontOf('serif');
    const metrics = context.measureText(character);
    const left = Math.ceil(metrics.actualBoundingBoxLeft) + 2;
    const right = Math.ceil(metrics.actualBoundingBoxRight) + 2;
    const ascent = Math.ceil(metrics.actualBoundingBoxAscent) + 2;
    const descent = Math.ceil(metrics.actualBoundingBoxDescent) + 2;
    const width = Math.max(1, left + right);
    const height = Math.max(1, ascent + descent);
    const serif = draw(character, 'serif', width, height, left, ascent);
    const monospace = draw(character, 'monospace', width, height, left, ascent);
    // A character neither KaTeX font has falls back to a system font: leave it out.
    if (character !== ' ' && serif.some((value, index) => value !== monospace[index])) {
      missing.push(character);
      continue;
    }
    glyphs.push({character, advance: metrics.width, left, ascent, width, height, alpha: serif});
  }
  context.font = fontOf('serif');
  context.fontKerning = 'normal';
  const kerning = [];
  const kernable = glyphs.map(glyph => glyph.character).filter(character => character !== ' ');
  const advances = new Map(glyphs.map(glyph => [glyph.character, glyph.advance]));
  for (const first of kernable) {
    for (const second of kernable) {
      const pair = context.measureText(first + second).width;
      const kern = pair - advances.get(first) - advances.get(second);
      if (Math.abs(kern) > size * 0.01) kerning.push([first, second, kern]);
    }
  }
  return {
    glyphs,
    missing,
    kerning,
    ascent: fontMetrics.fontBoundingBoxAscent,
    descent: fontMetrics.fontBoundingBoxDescent,
    capHeight: context.measureText('H').actualBoundingBoxAscent,
    xHeight: context.measureText('x').actualBoundingBoxAscent
  };
}

/** Squared distance transform of one row or column (Felzenszwalb and Huttenlocher). */
function transformLine(input, output, length, locations, boundaries) {
  const getIntersection = (position, location) =>
    (input[position] + position * position - (input[location] + location * location)) /
    (2 * position - 2 * location);
  let rightmost = 0;
  locations[0] = 0;
  boundaries[0] = -Infinity;
  boundaries[1] = Infinity;
  for (let position = 1; position < length; position++) {
    let intersection = getIntersection(position, locations[rightmost]);
    // boundaries[0] is -Infinity, so this stops at the first parabola.
    while (intersection <= boundaries[rightmost]) {
      rightmost--;
      intersection = getIntersection(position, locations[rightmost]);
    }
    rightmost++;
    locations[rightmost] = position;
    boundaries[rightmost] = intersection;
    boundaries[rightmost + 1] = Infinity;
  }
  rightmost = 0;
  for (let position = 0; position < length; position++) {
    while (boundaries[rightmost + 1] < position) rightmost++;
    const location = locations[rightmost];
    output[position] = (position - location) * (position - location) + input[location];
  }
}

/** Euclidean distance from every pixel to the nearest pixel where `isTarget` holds. */
function getDistances(width, height, isTarget) {
  const large = 1e12;
  const grid = new Float64Array(width * height);
  for (let index = 0; index < grid.length; index++) grid[index] = isTarget(index) ? 0 : large;
  const length = Math.max(width, height);
  const input = new Float64Array(length);
  const output = new Float64Array(length);
  const locations = new Int32Array(length);
  const boundaries = new Float64Array(length + 1);
  for (let column = 0; column < width; column++) {
    for (let row = 0; row < height; row++) input[row] = grid[row * width + column];
    transformLine(input, output, height, locations, boundaries);
    for (let row = 0; row < height; row++) grid[row * width + column] = output[row];
  }
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) input[column] = grid[row * width + column];
    transformLine(input, output, width, locations, boundaries);
    for (let column = 0; column < width; column++) grid[row * width + column] = output[column];
  }
  for (let index = 0; index < grid.length; index++) grid[index] = Math.sqrt(grid[index]);
  return grid;
}

/** Downsampled signed distance field of one glyph, with SPREAD texels of padding. */
function createField(glyph) {
  const padding = SPREAD * SUPERSAMPLE;
  // Pad the high-resolution bitmap so the field reaches beyond the outline, and align it to the
  // supersampling grid.
  const width = Math.ceil((glyph.width + 2 * padding) / SUPERSAMPLE) * SUPERSAMPLE;
  const height = Math.ceil((glyph.height + 2 * padding) / SUPERSAMPLE) * SUPERSAMPLE;
  const coverage = new Uint8Array(width * height);
  for (let row = 0; row < glyph.height; row++) {
    for (let column = 0; column < glyph.width; column++) {
      coverage[(row + padding) * width + column + padding] = glyph.alpha[row * glyph.width + column];
    }
  }
  const inside = getDistances(width, height, index => coverage[index] >= 128);
  const outside = getDistances(width, height, index => coverage[index] < 128);
  const fieldWidth = width / SUPERSAMPLE;
  const fieldHeight = height / SUPERSAMPLE;
  const field = new Uint8Array(fieldWidth * fieldHeight);
  for (let row = 0; row < fieldHeight; row++) {
    for (let column = 0; column < fieldWidth; column++) {
      // Average the signed distance over the supersampled block (texel centre).
      let sum = 0;
      for (let y = 0; y < SUPERSAMPLE; y++) {
        for (let x = 0; x < SUPERSAMPLE; x++) {
          const index = (row * SUPERSAMPLE + y) * width + column * SUPERSAMPLE + x;
          // Positive inside the glyph, in high-resolution pixels, measured to the outline.
          sum += outside[index] > 0 ? outside[index] - 0.5 : -(inside[index] - 0.5);
        }
      }
      const distance = sum / (SUPERSAMPLE * SUPERSAMPLE) / SUPERSAMPLE;
      const value = 0.5 + distance / (2 * SPREAD);
      field[row * fieldWidth + column] = Math.round(Math.min(1, Math.max(0, value)) * 255);
    }
  }
  return {
    field,
    fieldWidth,
    fieldHeight,
    // Quad origin relative to the pen position, in high-resolution pixels (y up for top).
    quadLeft: -(glyph.left + padding),
    quadTop: glyph.ascent + padding
  };
}

async function main() {
  const browser = await chromium.launch();
  const browserVersion = `Chromium ${browser.version()} (playwright)`;
  let result;
  try {
    const page = await browser.newPage();
    await page.setContent('<!doctype html><html><body></body></html>');
    result = await page.evaluate(rasterize, {
      sans: [...sansBytes],
      main: [...mainBytes],
      characters: CHARACTERS,
      size: EM * SUPERSAMPLE
    });
  } finally {
    await browser.close();
  }
  const scale = EM * SUPERSAMPLE;
  const fields = result.glyphs.map(glyph => ({glyph, ...createField(glyph)}));
  // Shelf packing, tallest first.
  const order = [...fields].sort((first, second) => second.fieldHeight - first.fieldHeight);
  let x = 0;
  let y = 0;
  let shelfHeight = 0;
  for (const item of order) {
    if (x + item.fieldWidth > ATLAS_WIDTH) {
      x = 0;
      y += shelfHeight;
      shelfHeight = 0;
    }
    item.x = x;
    item.y = y;
    x += item.fieldWidth;
    shelfHeight = Math.max(shelfHeight, item.fieldHeight);
  }
  const atlasHeight = y + shelfHeight;
  // Rows padded to whole 32-bit words: the shader reads the atlas as array<u32>.
  const pixels = new Uint8Array(ATLAS_WIDTH * atlasHeight);
  for (const item of fields) {
    for (let row = 0; row < item.fieldHeight; row++) {
      pixels.set(
        item.field.subarray(row * item.fieldWidth, (row + 1) * item.fieldWidth),
        (item.y + row) * ATLAS_WIDTH + item.x
      );
    }
  }
  const round = value => Math.round(value * 10000) / 10000;
  // One row per glyph: code point, atlas x, y, width, height, quad left and top (em, top is above
  // the baseline), advance (em).
  const glyphRows = fields.map(item => [
    item.glyph.character.codePointAt(0),
    item.x,
    item.y,
    item.fieldWidth,
    item.fieldHeight,
    round(item.quadLeft / scale),
    round(item.quadTop / scale),
    round(item.glyph.advance / scale)
  ]);
  const kerningRows = result.kerning.map(([first, second, kern]) => [
    first.codePointAt(0),
    second.codePointAt(0),
    round(kern / scale)
  ]);
  const pixelData = Buffer.from(pixels).toString('base64');
  const header = [
    '// SPDX-License-Identifier: OFL-1.1',
    ...HOLDERS.map(holder => `// SPDX-FileCopyrightText: ${holder}`),
    '',
    '// Generated by ./generate-font-atlas.mjs. Do not edit.',
    '//',
    `// ${FONT_NAME}: a Modified Version (subset, signed distance field) of Font Software licensed`,
    '// under the SIL Open Font License 1.1, distributed under that licence only. It does not use the',
    '// Reserved Font Names KaTeX_SansSerif and KaTeX_Main. The copyright notice and licence of the',
    '// fonts, as the font files state them:',
    '//',
    ...SOURCES.flatMap(source => [
      `// ${source.file} (${source.version}, sha256 ${source.sha256}):`,
      ...source.license.split('\n').map(line => `//   ${line}`.trimEnd())
    ]),
    '//',
    `// Source: katex ${katexVersion} (node_modules/katex/dist/fonts), rasterized by ${browserVersion}.`,
    `// ${fields.length} glyphs, ${ATLAS_WIDTH} x ${atlasHeight} texels, ${EM} texels per em, field reach ${SPREAD} texels.`,
    `// Characters not in either font: ${JSON.stringify(result.missing.join(''))}.`,
    '// The full licence text is in FONT_LICENSE below.'
  ].join('\n');
  const notice = {
    name: FONT_NAME,
    copyright: HOLDERS.join('\n'),
    reservedFontNames: ['KaTeX_SansSerif', 'KaTeX_Main'],
    source: SOURCES.map(({file, version, sha256: hash}) => ({file, version, sha256: hash})),
    package: `katex@${katexVersion}`,
    rasterizer: browserVersion
  };
  const metricsSource = `${header}

/** The licence of the font data (SIL OFL 1.1), kept as data so it ships with every bundle. */
export const FONT_LICENSE = ${JSON.stringify({...notice, text: OFL_TEXT}, null, 2)};

export type FontMetricsData = {
  /** Atlas texels per em. */
  em: number;
  /** Distance field reach on each side of an outline, in atlas texels (0.5 is the outline). */
  spread: number;
  /** Atlas size in texels; the pixels are in ./font-pixels.ts. */
  width: number;
  height: number;
  /** Line metrics in em. */
  ascent: number;
  descent: number;
  capHeight: number;
  xHeight: number;
  /** Code point, atlas x, y, width, height, quad left, quad top (above the baseline), advance. */
  glyphs: readonly (readonly number[])[];
  /** First code point, second code point, adjustment in em. */
  kerning: readonly (readonly number[])[];
  license: typeof FONT_LICENSE;
};

export const FONT_METRICS: FontMetricsData = {
  em: ${EM},
  spread: ${SPREAD},
  width: ${ATLAS_WIDTH},
  height: ${atlasHeight},
  ascent: ${round(result.ascent / scale)},
  descent: ${round(result.descent / scale)},
  capHeight: ${round(result.capHeight / scale)},
  xHeight: ${round(result.xHeight / scale)},
  glyphs: ${JSON.stringify(glyphRows)},
  kerning: ${JSON.stringify(kerningRows)},
  license: FONT_LICENSE
};
`;
  const pixelsSource = `${header.replace('in FONT_LICENSE below', 'in FONT_LICENSE in ./font-metrics.ts')}

/** The atlas texels, loaded on first use by ./font.ts. */
export const FONT_PIXELS = {
  name: ${JSON.stringify(FONT_NAME)},
  license: 'SIL Open Font License 1.1 (OFL-1.1); copyright ${HOLDERS.join('; ').replace(/'/g, "\\'")}',
  width: ${ATLAS_WIDTH},
  height: ${atlasHeight},
  /** Base64 of width x height bytes, row 0 at the top. */
  data:
    '${pixelData}'
};
`;
  fs.writeFileSync(path.join(directory, 'font-metrics.ts'), metricsSource);
  fs.writeFileSync(path.join(directory, 'font-pixels.ts'), pixelsSource);
  console.log(
    `${fields.length} glyphs, ${kerningRows.length} kerning pairs, ${ATLAS_WIDTH}x${atlasHeight}; ` +
      `metrics ${(metricsSource.length / 1024).toFixed(1)} KiB, pixels ${(pixelsSource.length / 1024).toFixed(1)} KiB; ` +
      `missing ${JSON.stringify(result.missing.join(''))}; ${browserVersion}`
  );
}

await main();
