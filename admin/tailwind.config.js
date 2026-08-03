/** @type {import('tailwindcss').Config} */

// Deliberately a copy of ../frontend/tailwind.config.js rather than a shared
// package: the two apps are independent builds and the console is free to
// diverge. The *token names* are what must stay in step — they are the reason
// both apps read as one product — so if you rename one here, rename it there.
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
      // A console reads denser than the chat app: one step smaller across the
      // board, with line heights tight enough that a 40-row table fits on a
      // laptop screen.
      fontSize: {
        "2xs": ["11px", "14px"],
        xs: ["12px", "16px"],
        sm: ["13px", "18px"],
        base: ["14px", "20px"],
        lg: ["16px", "22px"],
        xl: ["20px", "26px"],
        "2xl": ["26px", "32px"],
      },
      spacing: { sidebar: "224px" },
      borderRadius: { md: "8px", lg: "12px", xl: "16px" },
      boxShadow: {
        card: "0 1px 2px rgb(0 0 0 / 0.03)",
        pop: "0 4px 12px rgb(0 0 0 / 0.08), 0 12px 32px rgb(0 0 0 / 0.10)",
      },
      keyframes: {
        "fade-in": { from: { opacity: 0 }, to: { opacity: 1 } },
        "slide-up": {
          from: { opacity: 0, transform: "translateY(4px)" },
          to: { opacity: 1, transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 150ms ease-out",
        "slide-up": "slide-up 160ms ease-out",
      },
    },
  },
  plugins: [],
};
