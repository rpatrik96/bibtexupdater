<%*
// Zotero Bulk Sync Script for Obsidian
// Imports all new papers from Zotero library that don't have notes yet
//
// Prerequisites:
//   - Obsidian Citation Plugin configured to export CSL-JSON to Papers/zotero.json
//   - Zotero Integration plugin configured with "Import" format
//
// Usage:
//   1. Assign a hotkey via Settings > Templater > Template Hotkeys
//   2. Or run via Templater: Insert Template command

// Configuration
const IMPORT_FORMAT = 'Import';  // Must match Zotero Integration export format name
const LIBRARY_ID = 1;  // 1 = "My Library", 2+ = group libraries
const PAPERS_FOLDER = 'Papers';  // Folder where paper notes are stored
const CSL_JSON_PATH = 'Papers/zotero.json';  // Path to CSL-JSON export from Citation plugin
const DELAY_MS = 800;  // Delay between imports to prevent API overload
const MAX_IMPORTS = 50;  // Maximum papers to import in one run (safety limit)

// Get plugins
const zoteroPlugin = app.plugins.getPlugin('obsidian-zotero-desktop-connector');

if (!zoteroPlugin) {
    new Notice('Zotero Integration plugin not found!', 5000);
    return;
}

try {
    // Read CSL-JSON file
    const cslFile = app.vault.getAbstractFileByPath(CSL_JSON_PATH);
    if (!cslFile) {
        new Notice(`CSL-JSON file not found at: ${CSL_JSON_PATH}`, 5000);
        new Notice('Make sure Obsidian Citation Plugin is configured to export zotero.json', 5000);
        return;
    }

    const cslContent = await app.vault.read(cslFile);
    const cslData = JSON.parse(cslContent);

    if (!Array.isArray(cslData)) {
        new Notice('Invalid CSL-JSON format', 5000);
        return;
    }

    // Extract citekeys from CSL-JSON
    // CSL-JSON uses 'id' field for citekey
    const allCitekeys = cslData
        .map(item => item.id)
        .filter(id => id && typeof id === 'string');

    new Notice(`Found ${allCitekeys.length} items in Zotero library`, 3000);

    // Get existing paper notes
    const existingNotes = app.vault.getMarkdownFiles()
        .filter(f => f.path.startsWith(PAPERS_FOLDER + '/'))
        .filter(f => f.basename.startsWith('@'))
        .map(f => f.basename.replace(/^@/, ''));

    new Notice(`Found ${existingNotes.length} existing paper notes`, 3000);

    // Find new items to import
    const newItems = allCitekeys.filter(key => !existingNotes.includes(key));

    if (newItems.length === 0) {
        new Notice('All papers already imported!', 3000);
        return;
    }

    // Prompt for confirmation
    const importCount = Math.min(newItems.length, MAX_IMPORTS);
    const proceed = await tp.system.suggester(
        [
            `Import ${importCount} new papers` + (newItems.length > MAX_IMPORTS ? ` (${newItems.length - MAX_IMPORTS} will be skipped)` : ''),
            'Show list of new papers',
            'Cancel'
        ],
        ['import', 'list', 'cancel'],
        false,
        `${newItems.length} new papers found`
    );

    if (proceed === 'cancel' || !proceed) {
        return;
    }

    if (proceed === 'list') {
        // Show list of new papers
        const selection = await tp.system.suggester(
            newItems.slice(0, 20).map(key => {
                const item = cslData.find(i => i.id === key);
                const title = item?.title || 'Unknown title';
                const shortTitle = title.length > 60 ? title.substring(0, 57) + '...' : title;
                return `${key}: ${shortTitle}`;
            }),
            newItems.slice(0, 20),
            false,
            'Select paper to import (showing first 20)'
        );

        if (selection) {
            new Notice(`Importing: ${selection}...`, 2000);
            try {
                await zoteroPlugin.runImport(IMPORT_FORMAT, selection, LIBRARY_ID);
                new Notice(`Imported: ${selection}`, 3000);
            } catch (e) {
                new Notice(`Failed to import ${selection}: ${e.message}`, 5000);
            }
        }
        return;
    }

    // Bulk import
    new Notice(`Importing ${importCount} new papers...`, 3000);
    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < importCount; i++) {
        const citekey = newItems[i];
        try {
            await zoteroPlugin.runImport(IMPORT_FORMAT, citekey, LIBRARY_ID);
            successCount++;
            new Notice(`[${i + 1}/${importCount}] Imported: ${citekey}`, 2000);
        } catch (e) {
            failCount++;
            console.error(`Failed to import ${citekey}:`, e);
        }
        // Delay between imports
        if (i < importCount - 1) {
            await new Promise(resolve => setTimeout(resolve, DELAY_MS));
        }
    }

    new Notice(`Import complete! Success: ${successCount}, Failed: ${failCount}`, 5000);

} catch (e) {
    new Notice(`Zotero bulk sync error: ${e.message}`, 5000);
    console.error('Zotero bulk sync error:', e);
}
-%>
