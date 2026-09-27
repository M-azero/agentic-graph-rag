/** @type {import('tailwindcss').Config} */

// Colors are CSS variables (defined in index.css) rather than literal values,
// so light and dark are one set of class names with two sets of values —
// components never branch on theme.
const withVar = (name) => `rgb(var(--${name}) / <alpha-value>)`;
// Surfaces are glass: their alpha is the requested one scaled by a per-theme
// translucency, so `bg-surface` lets the sky through and `bg-surface/50` still
// means "half of that".
const glassVar = (name) =>
  `rgb(var(--${name}) / calc(<alpha-value> * var(--${name}-alpha, 1)))`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        canvas: withVar("canvas"),
        surface: glassVar("surface"),
        raised: glassVar("raised"),
        border: glassVar("border"),
        strong: withVar("text-strong"),
        body: withVar("text-body"),
        muted: withVar("text-muted"),
        accent: {
          DEFAULT: withVar("accent"),
          soft: withVar("accent-soft"),
          text: withVar("accent-text"),
        },
        positive: withVar("positive"),
        caution: withVar("caution"),
        danger: withVar("danger"),
      },
      fontFamily: {
        sans: ["Onest Variable", "Onest", "system-ui", "sans-serif"],
        display: ["Syne Variable", "Syne", "Onest Variable", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      // A named scale, so components stop reaching for text-[13px] / text-[15px]
      // literals. Tight steps: at 12/13/14/16 the difference between a label, a
      // control and body text is legible without any of them shouting.
      fontSize: {
        "2xs": ["11px", "15px"],
        xs: ["12px", "17px"],
        sm: ["13px", "20px"],
        base: ["14px", "22px"],
        lg: ["16px", "24px"],
        xl: ["20px", "28px"],
        "2xl": ["26px", "32px"],
      },
      borderRadius: { md: "10px", lg: "14px", xl: "20px", "2xl": "26px" },
      boxShadow: {
        // Glass floats on light, not on a grey smudge: a tinted, wide, soft
        // drop plus a one-pixel sheen along the top edge.
        card: "inset 0 1px 0 rgb(255 255 255 / var(--sheen)), 0 18px 40px -28px rgb(var(--shade) / 0.55)",
        pop: "inset 0 1px 0 rgb(255 255 255 / var(--sheen)), 0 24px 60px -20px rgb(var(--shade) / 0.6)",
        glow: "0 0 0 1px rgb(var(--accent) / 0.35), 0 8px 28px -8px rgb(var(--accent) / 0.55)",
      },
      keyframes: {
        drift: {
          "0%": { transform: "translate3d(-4%, -2%, 0) rotate(-2deg) scale(1)" },
          "50%": { transform: "translate3d(3%, 2%, 0) rotate(1deg) scale(1.08)" },
          "100%": { transform: "translate3d(-2%, 4%, 0) rotate(3deg) scale(1.02)" },
        },
        "fade-in": { from: { opacity: 0 }, to: { opacity: 1 } },
        "slide-up": {
          from: { opacity: 0, transform: "translateY(4px)" },
          to: { opacity: 1, transform: "translateY(0)" },
        },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: {
        "fade-in": "fade-in 150ms ease-out",
        "slide-up": "slide-up 200ms ease-out",
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};
