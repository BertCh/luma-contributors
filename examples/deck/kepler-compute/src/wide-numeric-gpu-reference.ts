// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compatibility entry: the implementation belongs to the public analysis package.
export {
  splitBigIntToWords,
  joinWordsToBigInt,
  splitDoubleToWords,
  addWords,
  subtractWords,
  negateWords,
  isUnsignedLess,
  isSignedLess,
  multiplyWords32,
  multiplyWordsLow,
  multiplyWords128,
  shiftWordsRight,
  shiftWordsLeft,
  countLeadingZeroWords,
  divideWordsBySmall,
  divideSignedWordsBySmall,
  convertWordsToFloat32,
  sumWordsWithCarry,
  combineWordSumAccumulators,
  classifyWordsNarrowing,
  decodeNarrowingError,
  roundBigIntToFloat32,
  summarizeInteger64Reference,
  WORD_RADIX
} from '@luma-contributors/analysis/dataframe-analysis';
export type {
  WordPair,
  WideNumericGpuFormat,
  WordNarrowingClassification,
  WideInteger64Summary
} from '@luma-contributors/analysis/dataframe-analysis';

import {summarizeNarrowingReference as summarizeLibraryNarrowingReference} from '@luma-contributors/analysis/dataframe-analysis';

/** Legacy example reports use a JSON-safe number; the public library retains exact bigint errors. */
export function summarizeNarrowingReference(
  ...parameters: Parameters<typeof summarizeLibraryNarrowingReference>
): Omit<ReturnType<typeof summarizeLibraryNarrowingReference>, 'maximumAbsoluteError'> & {
  maximumAbsoluteError: number;
} {
  const report = summarizeLibraryNarrowingReference(...parameters);
  return {...report, maximumAbsoluteError: Number(report.maximumAbsoluteError)};
}
