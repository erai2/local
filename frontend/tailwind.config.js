
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{vue,js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',
        dark: '#191D24',
        gray: '#374151',
      },
    },
  },
  plugins: [],
}
