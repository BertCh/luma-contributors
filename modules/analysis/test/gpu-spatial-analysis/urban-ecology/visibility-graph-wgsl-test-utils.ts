// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

import {
  findConstantNonFiniteBitcasts,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

export function expectUrbanWGSLIdentifiers(source: string, name: string): void {
  const code = stripWGSLComments(source);
  const identifiers = code.match(/[a-zA-Z_][a-zA-Z_0-9]*/g) ?? [];
  expect(
    [...new Set(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier)))],
    name
  ).toEqual([]);
  expectUrbanWGSLNumericValues(code, name);
}

/** Reflection does not enforce WGSL literal ranges; reject oversized integer tokens separately. */
export function expectUrbanWGSLIntegerLiterals(source: string, name: string): void {
  // Tokenize identifiers and complete floats too, so their digits are never mistaken for integers.
  const tokens =
    stripWGSLComments(source).match(
      /0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?(?:[pP][+-]?\d+)?[iufh]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[iufh]?|[A-Za-z_]\w*/g
    ) ?? [];
  const invalid = tokens.filter(token => {
    const integer = /^(0[xX][0-9a-fA-F]+|\d+)([iu])?$/.exec(token);
    if (!integer) return false;
    const maximum =
      integer[2] === 'u' ? 0xffffffffn : integer[2] === 'i' ? 0x7fffffffn : 0x7fffffffffffffffn;
    return BigInt(integer[1]) > maximum;
  });
  expect(invalid, `${name}: integer literals must fit their WGSL type`).toEqual([]);
}

/** All generated urban floats materialize as f32; non-finite sentinels must be runtime values. */
export function expectUrbanWGSLNumericValues(source: string, name: string): void {
  const code = stripWGSLComments(source);
  expectUrbanWGSLIntegerLiterals(code, name);
  expect(
    findConstantNonFiniteBitcasts(code),
    `${name}: constant evaluation must stay finite`
  ).toEqual([]);
  const declaredNames = new Set(
    [...code.matchAll(/\b(?:const|let|var(?:<[^>]*>)?)\s+(\w+)/g)].map(match => match[1])
  );
  const invalidSpellings = (code.match(/\b(?:NaN|Infinity|nan|inf)\b/g) ?? []).filter(
    value => !declaredNames.has(value)
  );
  expect(invalidSpellings, `${name}: invalid number spellings`).toEqual([]);
  const tokens =
    code.match(
      /0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?(?:[pP][+-]?\d+)?[iufh]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[iufh]?|[A-Za-z_]\w*/g
    ) ?? [];
  const invalid = tokens.filter(token => {
    const hexadecimal = /^0[xX]/.test(token);
    if (hexadecimal ? !/[.pP]/.test(token) : !/^\d|^\./.test(token) || !/[.eEfh]/.test(token))
      return false;
    const literal = token.replace(/[fh]$/, '');
    let value: number;
    if (hexadecimal) {
      const [significand, exponent] = literal.slice(2).toLowerCase().split('p');
      const [whole, fraction = ''] = significand.split('.');
      value =
        (Number.parseInt(whole || '0', 16) +
          (fraction ? Number.parseInt(fraction, 16) / 16 ** fraction.length : 0)) *
        2 ** Number(exponent);
    } else value = Number(literal);
    return !Number.isFinite(Math.fround(value)) || (token.endsWith('h') && value > 65504);
  });
  expect(invalid, `${name}: float literals must stay finite in their WGSL type`).toEqual([]);
}
