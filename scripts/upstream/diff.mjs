// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';

const [previousPath, currentPath] = process.argv.slice(2);
if (!previousPath || !currentPath) throw new Error('Usage: diff.mjs <old.json> <new.json>');
const previous = JSON.parse(readFileSync(previousPath, 'utf8'));
const current = JSON.parse(readFileSync(currentPath, 'utf8'));
for (const entry of new Set([...Object.keys(previous), ...Object.keys(current)])) {
  if (!Object.hasOwn(previous, entry)) {
    console.log('NEW ENTRY', entry, Object.keys(current[entry] || {}).length);
    continue;
  }
  if (!Object.hasOwn(current, entry)) {
    console.log('REMOVED ENTRY', entry);
    continue;
  }
  if (current[entry] === null || previous[entry] === null) {
    if (current[entry] !== previous[entry])
      console.log(current[entry] === null ? 'UNRESOLVED ENTRY' : 'RESOLVED ENTRY', entry);
    continue;
  }
  const added = Object.keys(current[entry]).filter(name => !(name in previous[entry]));
  const removed = Object.keys(previous[entry]).filter(name => !(name in current[entry]));
  if (added.length || removed.length)
    console.log(
      `${entry}\n  + ${added.map(name => `${name}(${current[entry][name].kind})`).join(', ')}\n  - ${removed.join(', ')}`
    );
}
