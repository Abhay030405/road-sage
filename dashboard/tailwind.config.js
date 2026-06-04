/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        'rs-bg': '#0a0a0f',
        'rs-panel': '#12121a',
        'rs-border': '#1e1e2e',
        'rs-green': '#22c55e',
        'rs-amber': '#f59e0b',
        'rs-blue': '#3b82f6',
        'rs-red': '#ef4444',
        'rs-text': '#e2e8f0',
        'rs-muted': '#64748b',
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
