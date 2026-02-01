# Obsidian Zotero Sync Templates

Templater scripts for automating Zotero → Obsidian annotation extraction.

## Overview

These scripts trigger Zotero Integration imports using your paper template. They make it easier to:
- Import new papers from Zotero
- Update existing paper notes with new annotations
- Bulk import multiple papers at once

## Prerequisites

### Required Obsidian Plugins
- **[Zotero Integration](https://github.com/mgmeyers/obsidian-zotero-integration)** - Core import functionality
- **[Templater](https://github.com/SilentVoid13/Templater)** - For running the sync scripts
- **[Obsidian Citation Plugin](https://github.com/hans/obsidian-citation-plugin)** - Exports CSL-JSON for bulk sync

### Required Zotero Plugins
- **[Better BibTeX](https://retorque.re/zotero-better-bibtex/)** - For citekey generation

## Files

| File | Purpose |
|------|---------|
| `zotero-sync.md` | Interactive single-paper sync |
| `zotero-bulk-sync.md` | Bulk import new papers |
| `zotero-paper-template.md` | Example paper import template with color-coded annotations |

## Installation

1. Copy `zotero-sync.md` and `zotero-bulk-sync.md` to your Obsidian `Templates/` folder
2. Configure Zotero Integration with an "Import" format pointing to your paper template
3. Configure Obsidian Citation Plugin to export CSL-JSON (for bulk sync)

## Configuration

Edit the configuration variables at the top of each sync script:

```javascript
const IMPORT_FORMAT = 'Import';      // Zotero Integration format name
const LIBRARY_ID = 1;                // 1 = My Library, 2+ = group libraries
const PAPERS_FOLDER = 'Papers';      // Where paper notes are stored
const CSL_JSON_PATH = 'Papers/zotero.json';  // CSL-JSON export path
```

## Setup Hotkeys

1. Open **Settings → Hotkeys**
2. Search for "Templater: Insert Templates/zotero-sync.md"
3. Assign hotkey (e.g., `Cmd+Shift+Z`)
4. Search for "Templater: Insert Templates/zotero-bulk-sync.md"
5. Assign hotkey (e.g., `Cmd+Shift+Alt+Z`)

## Usage

### Quick Sync (`zotero-sync.md`)
- **On a paper note**: Updates that paper with latest annotations
- **Elsewhere**: Shows menu to import by citekey or open Zotero dialog

### Bulk Sync (`zotero-bulk-sync.md`)
- Compares Zotero library against existing paper notes
- Imports all papers not yet in Obsidian (max 50 per run)
- Shows progress notifications

## Paper Template

The included `zotero-paper-template.md` uses color-coded callouts for annotations:

| Color | Meaning |
|-------|---------|
| Yellow (#ffd400) | Summary |
| Red (#ff6666) | Important |
| Green (#5fb236) | Notation |
| Blue (#2ea8e5) | Technical details/experiments |
| Purple (#a28ae5) | Contribution |
| Pink (#e56eee) | Connection to literature |
| Orange (#f19837) | Assumption |
| Gray (#aaaaaa) | Wrong?! |

## Startup Automation (Optional)

### Option A: QuickAdd Plugin
1. Install QuickAdd from Community Plugins
2. Create macro with Templater action → `zotero-bulk-sync.md`
3. Enable "Run on plugin load"

### Option B: macOS Launch Agent
Create `~/Library/LaunchAgents/com.obsidian.zotero-sync.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.obsidian.zotero-sync</string>
    <key>ProgramArguments</key>
    <array>
        <string>open</string>
        <string>obsidian://advanced-uri?vault=YourVaultName&commandid=templater-obsidian%3Ainsert-templater</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
```

## Troubleshooting

### "Zotero Integration plugin not found"
- Ensure `obsidian-zotero-desktop-connector` is enabled
- Restart Obsidian

### "CSL-JSON file not found"
- Open Settings → Citation Plugin
- Verify export path matches `CSL_JSON_PATH` in script
- Click refresh/rebuild cache

### Import fails for specific paper
- Ensure Zotero desktop is running
- Verify citekey exists in Zotero (check Better BibTeX)
- View console (Cmd+Opt+I) for detailed errors

## References

- [Zotero Integration Plugin](https://github.com/mgmeyers/obsidian-zotero-integration)
- [Templater Plugin](https://github.com/SilentVoid13/Templater)
- [Better BibTeX Auto Export](https://retorque.re/zotero-better-bibtex/exporting/auto/)
- [FeralFlora's runImport Template](https://gist.github.com/FeralFlora/78f494c1862ce4457cef28d9d9ba5a01)
