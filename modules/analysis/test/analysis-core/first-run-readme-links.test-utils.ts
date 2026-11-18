// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {join, normalize, resolve, sep} from 'node:path';
import {REPOSITORY_ROOT} from './doc-program';

/** Validates repository-document destinations offline; the repository identity is owner-supplied. */
export function getReadmeLinkIssues(
  text: string,
  repositoryUrl: string,
  shippedFiles: string[]
): string[] {
  const repository = new URL(repositoryUrl);
  const packageRoot = join(REPOSITORY_ROOT, 'modules/analysis');
  const targets = [...text.matchAll(/\]\(([^)]+)\)/g)].map(match => match[1]);
  const issues: string[] = [];
  for (const target of targets) {
    if (target.startsWith('#')) continue;
    if (target.startsWith('https://')) {
      const url = new URL(target);
      if (url.hostname !== 'github.com') {
        if (url.origin !== 'https://luma.gl')
          issues.push(`${target}: unrecognized external origin`);
        continue;
      }
      const prefix = `${repository.pathname.replace(/\/$/, '')}/blob/`;
      if (url.origin !== repository.origin || !url.pathname.startsWith(prefix)) {
        issues.push(`${target}: repository or branch does not match ${repositoryUrl}`);
        continue;
      }
      const referenceAndPath = url.pathname.slice(prefix.length);
      const separator = referenceAndPath.indexOf('/');
      const reference = referenceAndPath.slice(0, separator);
      if (separator < 0 || (reference !== 'master' && !/^[a-f0-9]{40}$/.test(reference))) {
        issues.push(`${target}: repository or branch does not match ${repositoryUrl}`);
        continue;
      }
      const relativePath = decodeURIComponent(referenceAndPath.slice(separator + 1));
      const path = resolve(REPOSITORY_ROOT, relativePath);
      if (!path.startsWith(`${REPOSITORY_ROOT}${sep}`) || !existsSync(path)) {
        issues.push(`${target}: repository path does not exist`);
        continue;
      }
      if (url.hash) {
        const headings = [...readFileSync(path, 'utf8').matchAll(/^#{1,6}\s+(.+?)\s*#*$/gm)];
        const counts = new Map<string, number>();
        const anchors = headings.map(match => {
          const base = match[1]
            .toLowerCase()
            .replace(/[^\p{L}\p{N}_\s-]/gu, '')
            .replace(/\s/g, '-');
          const count = counts.get(base) ?? 0;
          counts.set(base, count + 1);
          return count ? `${base}-${count}` : base;
        });
        if (!anchors.includes(decodeURIComponent(url.hash.slice(1)))) {
          issues.push(`${target}: heading anchor does not exist`);
        }
      }
      continue;
    }
    const path = normalize(target.split('#')[0]);
    const matches = (entry: string) => path === entry || path.startsWith(`${entry}/`);
    const included = shippedFiles.filter(entry => !entry.startsWith('!')).some(matches);
    const excluded = shippedFiles
      .filter(entry => entry.startsWith('!'))
      .some(entry => matches(entry.slice(1)));
    if (!included || excluded || !existsSync(join(packageRoot, path))) {
      issues.push(`${target}: target is not shipped`);
    }
  }
  return issues;
}
