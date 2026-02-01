<%*
// Zotero Sync Script for Obsidian
// Imports new/updated papers from Zotero library
//
// Usage:
//   1. Assign a hotkey to this template via Settings > Templater > Template Hotkeys
//   2. Run manually via Templater: Insert Template command
//
// Configuration
const IMPORT_FORMAT = 'Import';  // Must match Zotero Integration export format name
const LIBRARY_ID = 1;  // 1 = "My Library", 2+ = group libraries
const PAPERS_FOLDER = 'Papers';  // Folder where paper notes are stored
const DELAY_MS = 500;  // Delay between imports to prevent API overload

// Get the Zotero Integration plugin
const zoteroPlugin = app.plugins.getPlugin('obsidian-zotero-desktop-connector');

if (!zoteroPlugin) {
    new Notice('Zotero Integration plugin not found!', 5000);
    return;
}

// Check if Zotero is running
try {
    // Get existing paper notes to determine what's already imported
    const existingNotes = app.vault.getMarkdownFiles()
        .filter(f => f.path.startsWith(PAPERS_FOLDER + '/'))
        .filter(f => f.basename.startsWith('@'))
        .map(f => f.basename.replace(/^@/, ''));

    new Notice(`Found ${existingNotes.length} existing paper notes`, 3000);

    // Option 1: If this template is applied to a file with a citekey in frontmatter, update that single paper
    const activeFile = app.workspace.getActiveFile();
    if (activeFile) {
        const cache = app.metadataCache.getFileCache(activeFile);
        const citekey = cache?.frontmatter?.citekey;

        if (citekey && typeof citekey === 'string') {
            new Notice(`Updating: ${citekey}...`, 2000);
            try {
                await zoteroPlugin.runImport(IMPORT_FORMAT, citekey, LIBRARY_ID);
                new Notice(`Updated: ${citekey}`, 3000);
            } catch (e) {
                new Notice(`Failed to update ${citekey}: ${e.message}`, 5000);
                console.error('Zotero sync error:', e);
            }
            return;
        }
    }

    // Option 2: Prompt user for action
    const actions = [
        'Import specific citekey',
        'Import from Zotero selection',
        'Cancel'
    ];

    const choice = await tp.system.suggester(actions, actions, false, 'Zotero Sync Action');

    if (!choice || choice === 'Cancel') {
        return;
    }

    if (choice === 'Import specific citekey') {
        const citekey = await tp.system.prompt('Enter citekey to import:');
        if (citekey && citekey.trim()) {
            new Notice(`Importing: ${citekey.trim()}...`, 2000);
            try {
                await zoteroPlugin.runImport(IMPORT_FORMAT, citekey.trim(), LIBRARY_ID);
                new Notice(`Imported: ${citekey.trim()}`, 3000);
            } catch (e) {
                new Notice(`Failed to import ${citekey}: ${e.message}`, 5000);
                console.error('Zotero sync error:', e);
            }
        }
        return;
    }

    if (choice === 'Import from Zotero selection') {
        // This triggers the standard Zotero Integration import dialog
        // which allows selecting items from Zotero
        new Notice('Opening Zotero import dialog...', 2000);
        app.commands.executeCommandById('obsidian-zotero-desktop-connector:zotero-import');
        return;
    }

} catch (e) {
    new Notice(`Zotero sync error: ${e.message}`, 5000);
    console.error('Zotero sync error:', e);
}
-%>
