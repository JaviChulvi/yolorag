import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const source = resolve(root, "../../../llm/js/chat.js");
const target = resolve(root, "../public/vendor/ultralytics-chat.js");

if (!existsSync(source)) {
  throw new Error(`Missing local LLM widget at ${source}`);
}

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log(`Synced ${source} -> ${target}`);
