// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

/** Finds reserved tokens in generated WGSL, independently of the node WGSL interpreter. */
export function getReservedWGSLWords(source: string): string[] {
  const withoutComments = source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, ' ');
  return [...new Set(withoutComments.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [])].filter(word =>
    WGSL_RESERVED_WORDS.has(word)
  );
}
