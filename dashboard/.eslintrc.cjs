module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: ["eslint:recommended", "plugin:react/recommended", "plugin:react/jsx-runtime"],
  settings: { react: { version: "detect" } },
  rules: {
    // Modern React doesn't require PropTypes
    "react/prop-types": "off",
    // Prefer replaceAll
    "prefer-regex-literals": "off",
    // Allow window for browser code
    "no-restricted-globals": "off",
    // Allow nested ternaries in JSX
    "no-nested-ternary": "off",
    "unicorn/no-nested-ternary": "off",
    "unicorn/prefer-global-this": "off",
    "unicorn/prefer-string-replace-all": "off",
  },
  parserOptions: { ecmaVersion: "latest", sourceType: "module", ecmaFeatures: { jsx: true } },
};
