/** @type {import('tailwindcss').Config} */

// Colors are CSS variables (defined in index.css) rather than literal values,
// so light and dark are one set of class names with two sets of values —
// components never branch on theme.
const withVar = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        canvas: withVar("canvas"),
        surface: withVar("surface"),
        raised: withVar("raised"),
        border: withVar("border"),
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
        sans: ["Inter Variable", "Inter", "system-ui", "sans-serif"],
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
      borderRadius: { md: "8px", lg: "12px", xl: "16px" },
      boxShadow: {
        // Barely there. With hairline borders doing the separating, a shadow's
        // only job is to say "this floats" — anything heavier reads as a
        // 2015 material-design card.
        card: "0 1px 2px rgb(0 0 0 / 0.03)",
        pop: "0 4px 12px rgb(0 0 0 / 0.08), 0 12px 32px rgb(0 0 0 / 0.10)",
      },
      keyframes: {
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
