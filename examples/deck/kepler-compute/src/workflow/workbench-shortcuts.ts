// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WORKBENCH_SHORTCUTS, type WorkbenchShortcutAction} from './workbench-contract';

type ShortcutPlatform = 'mac' | 'other';
type ShortcutEvent = {
  key: string;
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
};

/** Matches the shared vocabulary with exact modifiers, including platform-specific Mod. */
export function matchWorkbenchShortcut(
  event: ShortcutEvent,
  platform: ShortcutPlatform
): WorkbenchShortcutAction | null {
  if (event.altKey) return null;
  const key = event.key === 'Backspace' ? 'delete' : event.key.toLowerCase();
  for (const shortcut of WORKBENCH_SHORTCUTS) {
    const parts = shortcut.keys.split('+');
    const hasModifier = parts.includes('Mod');
    if (
      event.shiftKey !== parts.includes('Shift') ||
      event.metaKey !== (hasModifier && platform === 'mac') ||
      event.ctrlKey !== (hasModifier && platform === 'other')
    )
      continue;
    if (parts.at(-1)?.toLowerCase() === key) return shortcut.action;
  }
  return null;
}

export function formatWorkbenchShortcut(keys: string, platform: ShortcutPlatform): string {
  if (platform === 'other') return keys.replace('Mod', 'Ctrl');
  const symbols: Record<string, string> = {
    Mod: '⌘',
    Shift: '⇧',
    Tab: '⇥',
    Enter: '↵',
    Escape: 'Esc',
    Delete: '⌫'
  };
  return keys
    .split('+')
    .map(key => symbols[key] ?? key)
    .join('');
}

/** Structural checks also work during server rendering, where Element does not exist. */
export function isEditableTarget(target: unknown): boolean {
  if (!target || typeof target !== 'object') return false;
  const element = target as {
    tagName?: string;
    isContentEditable?: boolean;
    getAttribute?: (name: string) => string | null;
    parentElement?: unknown;
  };
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(element.tagName?.toUpperCase() ?? '')) return true;
  if (element.isContentEditable) return true;
  const editable = element.getAttribute?.('contenteditable');
  if (editable !== null && editable !== undefined) return editable.toLowerCase() !== 'false';
  return isEditableTarget(element.parentElement);
}
