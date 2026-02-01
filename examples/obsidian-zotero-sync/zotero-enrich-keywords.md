---
description: Enrich paper notes with AI-generated [[wikilink]] keywords
---

<%*
/**
 * Zotero Paper Keyword Enricher
 *
 * This Templater script enriches paper notes with AI-generated topic keywords.
 * It uses the bibtex-updater's keyword generator as a fallback when Zotero tags
 * are insufficient.
 *
 * REQUIREMENTS:
 * - bibtex-updater installed with: pip install bibtex-updater
 * - ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable set
 * - Templater plugin with shell command execution enabled
 *
 * CONFIGURATION:
 */

// AI backend: "claude", "openai", or "embedding"
const AI_BACKEND = "claude";

// Maximum keywords to generate
const MAX_KEYWORDS = 5;

// Minimum existing keywords before enrichment (skip if already has enough)
const MIN_EXISTING_THRESHOLD = 3;

// Path to topics file (optional, one topic per line)
const TOPICS_FILE = "";

// Papers folder pattern
const PAPERS_FOLDER = "Papers";

/**
 * Helper: Parse YAML frontmatter
 */
function parseFrontmatter(content) {
    if (!content.startsWith("---")) return { frontmatter: {}, body: content };
    const parts = content.split("---");
    if (parts.length < 3) return { frontmatter: {}, body: content };

    // Simple YAML parsing for keywords
    const yaml = parts[1];
    const body = parts.slice(2).join("---");

    const keywordsMatch = yaml.match(/keywords:\s*\n((?:\s+-\s+.+\n)*)/);
    const keywords = [];
    if (keywordsMatch) {
        const lines = keywordsMatch[1].split("\n");
        for (const line of lines) {
            const match = line.match(/^\s+-\s+"?\[\[([^\]]+)\]\]"?/);
            if (match) keywords.push(match[1]);
        }
    }

    return { keywords, body, yaml };
}

/**
 * Helper: Extract abstract from note content
 */
function extractAbstract(content) {
    const match = content.match(/>\s*\[!abstract\][^\n]*\n>\s*(.+?)(?=\n\n|\n>?\s*\[!)/s);
    if (match) {
        return match[1].replace(/^>\s*/gm, "").trim();
    }
    return "";
}

/**
 * Helper: Get title from frontmatter aliases
 */
function extractTitle(content) {
    const aliasMatch = content.match(/aliases:\s*\n-\s*(\S+)\n-\s*"([^"]+)"/);
    if (aliasMatch) return aliasMatch[2];

    const h1Match = content.match(/^#\s+(.+)$/m);
    if (h1Match) return h1Match[1];

    return "";
}

// Main execution
const file = tp.file.find_tfile(tp.file.path(true));
const content = await app.vault.read(file);

// Parse existing keywords
const { keywords: existingKeywords } = parseFrontmatter(content);

// Check if enrichment is needed
if (existingKeywords.length >= MIN_EXISTING_THRESHOLD) {
    new Notice(`Note already has ${existingKeywords.length} keywords. Skipping.`);
    return;
}

// Extract paper info
const title = extractTitle(content);
const abstract = extractAbstract(content);

if (!abstract) {
    new Notice("No abstract found in note. Cannot generate keywords.");
    return;
}

// Build command
let cmd = `bibtex-obsidian-keywords "${tp.file.path(true)}" --backend ${AI_BACKEND} --max-keywords ${MAX_KEYWORDS} --json`;
if (TOPICS_FILE) {
    cmd += ` --topics-file "${TOPICS_FILE}"`;
}

try {
    new Notice("Generating AI keywords...", 3000);

    // Execute keyword generation
    const result = await tp.system.cmd(cmd);
    const data = JSON.parse(result);

    if (data.length === 0 || data[0].action === "error") {
        new Notice(`Error: ${data[0]?.message || "Unknown error"}`);
        return;
    }

    const generated = data[0].generated_keywords || [];

    if (generated.length === 0) {
        new Notice("No new keywords generated.");
        return;
    }

    // Keywords were already written by the CLI, just notify
    new Notice(`Added ${generated.length} keywords: ${generated.join(", ")}`);

    // Reload the file to show changes
    await app.vault.adapter.read(tp.file.path(true));

} catch (error) {
    console.error("Keyword generation failed:", error);
    new Notice(`Failed to generate keywords: ${error.message}`);
}
%>
