// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const THEME_STORAGE_KEY = 'gridlight-theme';

/** Inline script run before paint: applies a stored theme so there is no flash. */
export const THEME_INIT_SCRIPT = `try{var t=localStorage.getItem('${THEME_STORAGE_KEY}');if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-theme',t);}}catch(e){}`;

function readResolvedTheme(): 'light' | 'dark' {
  const explicit = document.documentElement.getAttribute('data-theme');
  if (explicit === 'light' || explicit === 'dark') {
    return explicit;
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function ThemeToggle() {
  function toggleTheme(): void {
    const next = readResolvedTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    window.dispatchEvent(new CustomEvent('gridlight-theme-change', {detail: next}));
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Storage blocked: the choice lasts for this page view only.
    }
  }

  // Both icons are rendered; CSS tokens show the one that matches the active theme (no hydration mismatch).
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggleTheme}
      aria-label="Toggle light and dark theme"
    >
      <svg
        className="theme-toggle__sun"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      >
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </svg>
      <svg
        className="theme-toggle__moon"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />
      </svg>
    </button>
  );
}
