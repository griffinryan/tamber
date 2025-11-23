import { createTokens } from '@tamagui/core';

export const tokens = createTokens({
  color: {
    background: '#050814',
    panel: 'rgba(14, 18, 32, 0.85)',
    accent: '#7ef1ff',
    accentAlt: '#7c8cff',
    text: '#f7f8ff',
    textMuted: '#9aa8d8',
    border: 'rgba(255, 255, 255, 0.08)',
    glow: '#4bd6ff',
    success: '#7df2a4',
    warn: '#ffdb7f',
    danger: '#ff8fb1'
  },
  radius: {
    0: 0,
    1: 6,
    2: 10,
    3: 14,
    4: 18,
  },
  space: {
    0: 0,
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    5: 20,
    6: 24,
    7: 32,
    8: 40,
  },
  size: {
    true: 32,
    0: 0,
    1: 20,
    2: 28,
    3: 36,
    4: 44,
    5: 52,
    6: 64,
    7: 72,
  },
  zIndex: {
    0: 0,
    1: 10,
    2: 50,
    3: 100,
  },
});
