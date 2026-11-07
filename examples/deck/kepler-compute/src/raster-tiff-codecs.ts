// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Pure-JS TIFF block codecs: DEFLATE (zlib or raw), TIFF LZW and PackBits. No dependencies. */

type Huffman = {count: Uint16Array; symbol: Uint16Array};

const LENGTH_BASE = [
  3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
  163, 195, 227, 258
];
const LENGTH_EXTRA = [
  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0
];
const DISTANCE_BASE = [
  1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049,
  3073, 4097, 6145, 8193, 12289, 16385, 24577
];
const DISTANCE_EXTRA = [
  0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13
];
const CODE_LENGTH_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function buildHuffman(lengths: ArrayLike<number>, symbolCount: number): Huffman {
  const count = new Uint16Array(16);
  for (let symbol = 0; symbol < symbolCount; symbol++) count[lengths[symbol]]++;
  const offsets = new Uint16Array(16);
  for (let length = 1; length < 15; length++) offsets[length + 1] = offsets[length] + count[length];
  const symbol = new Uint16Array(symbolCount);
  for (let index = 0; index < symbolCount; index++)
    if (lengths[index] !== 0) symbol[offsets[lengths[index]]++] = index;
  return {count, symbol};
}

let fixedTables: {literal: Huffman; distance: Huffman} | undefined;

function getFixedTables() {
  if (!fixedTables) {
    const lengths = new Uint8Array(288);
    lengths.fill(8, 0, 144);
    lengths.fill(9, 144, 256);
    lengths.fill(7, 256, 280);
    lengths.fill(8, 280, 288);
    fixedTables = {
      literal: buildHuffman(lengths, 288),
      distance: buildHuffman(new Uint8Array(30).fill(5), 30)
    };
  }
  return fixedTables;
}

/** Inflates a raw DEFLATE stream (RFC 1951). `expectedLength` presizes the output. */
export function inflateRaw(input: Uint8Array, expectedLength = 0): Uint8Array {
  let output = new Uint8Array(Math.max(expectedLength, 1024));
  let outputLength = 0;
  let position = 0;
  let bitBuffer = 0;
  let bitCount = 0;
  const readBits = (need: number): number => {
    while (bitCount < need) {
      // A truncated block is a corrupt file, not a recoverable state.
      if (position >= input.length) throw new Error('Inflate: unexpected end of data');
      bitBuffer |= input[position++] << bitCount;
      bitCount += 8;
    }
    const value = bitBuffer & ((1 << need) - 1);
    bitBuffer >>>= need;
    bitCount -= need;
    return value;
  };
  const decodeSymbol = (table: Huffman): number => {
    let code = 0;
    let first = 0;
    let index = 0;
    for (let length = 1; length <= 15; length++) {
      code |= readBits(1);
      const count = table.count[length];
      if (code - count < first) return table.symbol[index + (code - first)];
      index += count;
      first += count;
      first <<= 1;
      code <<= 1;
    }
    throw new Error('Inflate: invalid Huffman code');
  };
  const ensureCapacity = (extra: number): void => {
    if (outputLength + extra <= output.length) return;
    const grown = new Uint8Array(Math.max(output.length * 2, outputLength + extra));
    grown.set(output.subarray(0, outputLength));
    output = grown;
  };
  let isLast = false;
  while (!isLast) {
    isLast = readBits(1) === 1;
    const type = readBits(2);
    if (type === 0) {
      bitBuffer = 0;
      bitCount = 0;
      const length = input[position] | (input[position + 1] << 8);
      position += 4;
      ensureCapacity(length);
      output.set(input.subarray(position, position + length), outputLength);
      outputLength += length;
      position += length;
      continue;
    }
    let literal: Huffman;
    let distance: Huffman;
    if (type === 1) ({literal, distance} = getFixedTables());
    else if (type === 2) {
      const literalCount = readBits(5) + 257;
      const distanceCount = readBits(5) + 1;
      const codeLengthCount = readBits(4) + 4;
      const codeLengths = new Uint8Array(19);
      for (let index = 0; index < codeLengthCount; index++)
        codeLengths[CODE_LENGTH_ORDER[index]] = readBits(3);
      const codeLengthTable = buildHuffman(codeLengths, 19);
      const lengths = new Uint8Array(literalCount + distanceCount);
      for (let index = 0; index < lengths.length; ) {
        const symbol = decodeSymbol(codeLengthTable);
        if (symbol < 16) lengths[index++] = symbol;
        else {
          const previous = symbol === 16 ? lengths[index - 1] : 0;
          const repeat =
            symbol === 16 ? 3 + readBits(2) : symbol === 17 ? 3 + readBits(3) : 11 + readBits(7);
          for (let step = 0; step < repeat; step++) lengths[index++] = previous;
        }
      }
      literal = buildHuffman(lengths.subarray(0, literalCount), literalCount);
      distance = buildHuffman(lengths.subarray(literalCount), distanceCount);
    } else throw new Error('Inflate: invalid block type');
    for (;;) {
      const symbol = decodeSymbol(literal);
      if (symbol < 256) {
        ensureCapacity(1);
        output[outputLength++] = symbol;
      } else if (symbol === 256) break;
      else {
        const lengthIndex = symbol - 257;
        const length = LENGTH_BASE[lengthIndex] + readBits(LENGTH_EXTRA[lengthIndex]);
        const distanceSymbol = decodeSymbol(distance);
        const back = DISTANCE_BASE[distanceSymbol] + readBits(DISTANCE_EXTRA[distanceSymbol]);
        ensureCapacity(length);
        for (let step = 0; step < length; step++, outputLength++)
          output[outputLength] = output[outputLength - back];
      }
    }
  }
  return output.subarray(0, outputLength);
}

/** Inflates a zlib stream (TIFF compression 8 and 32946); tolerates a missing zlib header. */
export function inflateZlib(input: Uint8Array, expectedLength = 0): Uint8Array {
  const hasHeader = (input[0] & 0x0f) === 8 && ((input[0] << 8) | input[1]) % 31 === 0;
  return inflateRaw(hasHeader ? input.subarray(2) : input, expectedLength);
}

/** Decodes TIFF-flavoured LZW (MSB-first codes, 9 to 12 bits, early change). */
export function decodeTiffLzw(input: Uint8Array, expectedLength: number): Uint8Array {
  const output = new Uint8Array(expectedLength);
  const prefix = new Int16Array(4096);
  const suffix = new Uint8Array(4096);
  const stringLength = new Uint16Array(4096);
  for (let code = 0; code < 256; code++) {
    suffix[code] = code;
    stringLength[code] = 1;
    prefix[code] = -1;
  }
  let nextCode = 258;
  let width = 9;
  let bitBuffer = 0;
  let bitCount = 0;
  let position = 0;
  let outputLength = 0;
  const readCode = (): number => {
    while (bitCount < width) {
      if (position >= input.length) return 257;
      bitBuffer = (bitBuffer << 8) | input[position++];
      bitCount += 8;
    }
    bitCount -= width;
    const code = (bitBuffer >>> bitCount) & ((1 << width) - 1);
    bitBuffer &= (1 << bitCount) - 1;
    return code;
  };
  const writeString = (code: number): number => {
    const length = stringLength[code];
    let cursor = outputLength + length;
    let current = code;
    let first = 0;
    while (current !== -1) {
      cursor--;
      if (cursor < expectedLength) output[cursor] = suffix[current];
      first = suffix[current];
      current = prefix[current];
    }
    outputLength += length;
    return first;
  };
  const getFirstByte = (code: number): number => {
    let current = code;
    while (prefix[current] !== -1) current = prefix[current];
    return suffix[current];
  };
  let previous = -1;
  while (outputLength < expectedLength) {
    const code = readCode();
    if (code === 257) break;
    if (code === 256) {
      nextCode = 258;
      width = 9;
      previous = -1;
      continue;
    }
    let first: number;
    if (previous === -1) {
      first = writeString(code);
    } else if (code < nextCode) {
      first = writeString(code);
      prefix[nextCode] = previous;
      suffix[nextCode] = first;
      stringLength[nextCode] = stringLength[previous] + 1;
      nextCode++;
    } else {
      first = getFirstByte(previous);
      prefix[nextCode] = previous;
      suffix[nextCode] = first;
      stringLength[nextCode] = stringLength[previous] + 1;
      nextCode++;
      writeString(code);
    }
    width = nextCode >= 2047 ? 12 : nextCode >= 1023 ? 11 : nextCode >= 511 ? 10 : 9;
    previous = code;
  }
  return output;
}

/** Decodes PackBits (TIFF compression 32773). */
export function decodePackBits(input: Uint8Array, expectedLength: number): Uint8Array {
  const output = new Uint8Array(expectedLength);
  let position = 0;
  let outputLength = 0;
  while (position < input.length && outputLength < expectedLength) {
    const header = (input[position++] << 24) >> 24;
    if (header >= 0) {
      const count = header + 1;
      output.set(input.subarray(position, position + count), outputLength);
      position += count;
      outputLength += count;
    } else if (header !== -128) {
      const count = 1 - header;
      output.fill(input[position++], outputLength, outputLength + count);
      outputLength += count;
    }
  }
  return output;
}

/**
 * Undoes TIFF predictor 2 (horizontal differencing) in place for one block of `rowCount` rows.
 * Samples are 8, 16 or 32 bit in file byte order.
 */
export function undoHorizontalPredictor(
  block: Uint8Array,
  rowWidth: number,
  rowCount: number,
  samplesPerPixel: number,
  bitsPerSample: number,
  isLittleEndian: boolean
): void {
  const rowSamples = rowWidth * samplesPerPixel;
  if (bitsPerSample === 8) {
    for (let row = 0; row < rowCount; row++) {
      const base = row * rowSamples;
      for (let index = samplesPerPixel; index < rowSamples; index++)
        block[base + index] = (block[base + index] + block[base + index - samplesPerPixel]) & 255;
    }
    return;
  }
  const view = new DataView(block.buffer, block.byteOffset, block.byteLength);
  const bytes = bitsPerSample / 8;
  for (let row = 0; row < rowCount; row++) {
    const base = row * rowSamples * bytes;
    for (let index = samplesPerPixel; index < rowSamples; index++) {
      const offset = base + index * bytes;
      const previous = base + (index - samplesPerPixel) * bytes;
      if (bitsPerSample === 16)
        view.setUint16(
          offset,
          (view.getUint16(offset, isLittleEndian) + view.getUint16(previous, isLittleEndian)) &
            0xffff,
          isLittleEndian
        );
      else
        view.setUint32(
          offset,
          (view.getUint32(offset, isLittleEndian) + view.getUint32(previous, isLittleEndian)) >>> 0,
          isLittleEndian
        );
    }
  }
}

/**
 * Undoes TIFF predictor 3 (floating point). The result holds big-endian samples, so callers must
 * read them with `getFloat32(offset, false)` / `getFloat64(offset, false)`.
 */
export function undoFloatingPointPredictor(
  block: Uint8Array,
  rowWidth: number,
  rowCount: number,
  samplesPerPixel: number,
  bytesPerSample: number
): void {
  const sampleCount = rowWidth * samplesPerPixel;
  const rowBytes = sampleCount * bytesPerSample;
  const row = new Uint8Array(rowBytes);
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    const base = rowIndex * rowBytes;
    for (let index = samplesPerPixel; index < rowBytes; index++)
      block[base + index] = (block[base + index] + block[base + index - samplesPerPixel]) & 255;
    row.set(block.subarray(base, base + rowBytes));
    for (let sample = 0; sample < sampleCount; sample++)
      for (let byte = 0; byte < bytesPerSample; byte++)
        block[base + sample * bytesPerSample + byte] = row[byte * sampleCount + sample];
  }
}
