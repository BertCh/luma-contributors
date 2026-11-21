// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {execFileSync} from 'node:child_process';
import {posix} from 'node:path';

/** Checks npm archive member names, which normally start with "package/". */
export function assertPackedFileList(files) {
  for (const file of files) {
    const path = posix
      .normalize(file)
      .replace(/^package\//, '')
      .replace(/\/$/, '');
    if (
      path === 'src' ||
      path.startsWith('src/') ||
      path === 'dist/testing' ||
      path.startsWith('dist/testing/') ||
      path === 'dist/.build-record.json'
    ) {
      throw new Error(`packed file policy: forbidden path ${path}`);
    }
  }
}

/** Lists the captured archive bytes, using the same tar tool as extraction. */
export function checkPackedArchiveFiles(archiveBytes) {
  const listing = execFileSync('tar', ['-tzf', '-'], {
    input: archiveBytes,
    stdio: 'pipe',
    encoding: 'utf8',
    timeout: 60_000,
    maxBuffer: 16 * 1024 * 1024
  });
  assertPackedFileList(listing.split('\n'));
}
