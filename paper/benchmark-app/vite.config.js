import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@huggingface/transformers": path.resolve(
        __dirname,
        "../../packages/transformers/dist/transformers.web.js",
      ),
      "@transformers-src": path.resolve(__dirname, "../../packages/transformers/src"),
    },
  },
});
