# Frontend Changes: Dark/Light Mode Toggle

## Summary
Added a theme toggle button that allows users to switch between dark and light modes. The button features sun/moon icons with smooth transition animations.

## Files Modified

### 1. `frontend/index.html`
- Added a theme toggle button element with sun and moon SVG icons
- Button is positioned at the top of the body, outside the main container
- Includes proper accessibility attributes (`aria-label`, `title`)

### 2. `frontend/style.css`
- Added light theme CSS variables under `[data-theme="light"]` selector
- Added new `--code-bg` variable for code block backgrounds (works in both themes)
- Added `.theme-toggle` button styles:
  - Fixed position in top-right corner
  - Circular button design (44px diameter)
  - Hover, focus, and active states
  - Icon visibility transitions (sun/moon swap with rotation animation)
- Added responsive styles for mobile (smaller button size)
- Updated code block styles to use the new `--code-bg` variable

### 3. `frontend/script.js`
- Added `themeToggle` to DOM elements
- Added `initializeTheme()` function - loads saved theme from localStorage on page load
- Added `toggleTheme()` function - handles click events to switch themes
- Added `setTheme(theme)` function - applies theme and updates accessibility labels
- Theme preference persists across sessions via localStorage

## Features
- Toggle button positioned in top-right corner
- Sun icon shown in dark mode (click to switch to light)
- Moon icon shown in light mode (click to switch to dark)
- Smooth 0.3s transition animation when toggling
- Icons rotate during transition for visual polish
- Button scales on hover (1.05x) and shrinks on click (0.95x)
- Theme preference saved to localStorage
- Keyboard accessible (can tab to button and activate with Enter/Space)
- Screen reader friendly with dynamic aria-label
- Responsive design - smaller button on mobile devices
