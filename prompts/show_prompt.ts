#!/usr/bin/env tsx

import { constructSystemPrompt } from "./system_prompt.js";

async function main() {
  try {
    console.log("🔄 Generating system prompt...\n");

    const systemPrompt = await constructSystemPrompt();

    // ANSI escape code for yellow: \x1b[33m, reset: \x1b[0m
    console.log("\x1b[33m⭐️⭐️⭐️=== SYSTEM PROMPT ===⭐️⭐️⭐️\n\x1b[0m");
    console.log(systemPrompt);
    console.log("\n\x1b[33m⛔️⛔️⛔️=== END SYSTEM PROMPT ===⛔️⛔️⛔️\x1b[0m");
  } catch (error) {
    console.error("Error generating system prompt:", error);
    process.exit(1);
  }
}

main();
