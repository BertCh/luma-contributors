// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Keep inspection resources/capacity charged until every sibling mapping settles, including errors. */
export async function readAllBuffers(
  reads: readonly (() => Promise<Uint8Array>)[]
): Promise<Uint8Array[]> {
  const results = await Promise.allSettled(reads.map(read => Promise.resolve().then(read)));
  const failure = results.find(result => result.status === 'rejected');
  if (failure?.status === 'rejected') throw failure.reason;
  return results.flatMap(result => (result.status === 'fulfilled' ? [result.value] : []));
}
