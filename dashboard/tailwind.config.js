/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
      },
      colors: {
        'rs-bg':     'var(--rs-bg)',
        'rs-panel':  'var(--rs-panel)',
        'rs-border': 'var(--rs-border)',
        'rs-green':  'var(--rs-green)',
        'rs-amber':  'var(--rs-amber)',
        'rs-blue':   'var(--rs-blue)',
        'rs-red':    'var(--rs-red)',
        'rs-text':   'var(--rs-text)',
        'rs-muted':  'var(--rs-muted)',
      },
      animation: {
        'pulse-red': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.2s ease-in',
      },
      keyframes: {
        fadeIn: { '0%': { opacity: 0 }, '100%': { opacity: 1 } }
      }
    }
  },
  plugins: []
}
