Lumen Notes is a fictional product used to demo this project.

# Lumen Notes Guide

Lumen is a note-taking app for macOS, Windows and Linux. Notes are plain Markdown files stored in a folder you choose, so you can open them with any editor.

## Install

### macOS

Download Lumen from lumen.example and drag it to Applications.
```bash
# verify the install
lumen --version
```

### Linux

Lumen ships as an AppImage. Make it executable, then run it.

```bash
# make the AppImage executable
chmod +x Lumen-3.2.AppImage

# start Lumen
./Lumen-3.2.AppImage
```

### Windows

Run `LumenSetup.exe` and follow the installer. Lumen installs per user, so it does not need administrator rights.

## Sync

Open Settings > Sync and sign in. Notes sync every 5 minutes, and immediately when you press Ctrl+S.

### Devices and limits

| Plan | Devices | Storage | Version history |
|------|---------|---------|-----------------|
| Free | 3 | 1 GB | 7 days |
| Pro | 10 | 50 GB | 1 year |
| Team | Unlimited | 200 GB per seat | Forever |

Free accounts can sync up to 3 devices. Adding a fourth device signs out the device that was used least recently.

### Conflicts

A conflict happens when the same note is edited on two devices before either one syncs. Lumen never merges the two edits automatically, because an automatic merge can silently drop text. Instead, Lumen keeps both copies. The copy that synced first keeps the original name. The newer copy is renamed with the suffix "(conflict)" and the name of the device that created it, for example "Groceries (conflict, work laptop).md".

Conflicted copies appear in the sidebar with an orange marker. Open both copies side by side with View > Compare. Copy the text you want to keep into the original note, then delete the conflicted copy. Lumen does not delete conflicted copies on its own, even after the conflict is resolved, so check the sidebar now and then.

Conflicts are more likely when a device stays offline for a long time. A laptop that was closed for a week will sync all of its pending edits at once when it reconnects, and every note that was also edited elsewhere during that week becomes a conflict. If you know a device will be offline, close the notes you are editing on it first.

Some edits never cause conflicts. Renaming a note, moving it to another folder, or adding a tag is stored as metadata rather than as a change to the note's text, and Lumen applies metadata changes in the order they arrive. If two devices rename the same note, the rename that syncs last wins, and the other name is recorded in the note's history.

Attachments behave differently from notes. Images and PDFs are never edited inside Lumen, so two versions of the same attachment can only appear if the file was replaced on two devices. In that case Lumen keeps the larger file and moves the other one to the Trash, where it stays for 30 days.

If conflicts keep appearing for the same note, the usual cause is a second sync tool, such as a cloud drive client, watching the same folder. Two sync tools fighting over one folder will create a new conflict on almost every save. Keep the Lumen notes folder outside any folder that another sync tool manages.

To see the full conflict history for a note, open the note and choose File > History. Each entry shows which device made the change, when it synced, and whether it caused a conflict. History is kept for 7 days on the Free plan and for 1 year on Pro.

## Shortcuts

~~~text
Ctrl+N        new note
Ctrl+S        save and sync now
Ctrl+Shift+F  search all notes

Ctrl+K        insert link
~~~

## Pricing

Lumen Pro costs $4 per month or $40 per year. Team pricing is $8 per seat per month, billed annually. Students get Pro free with a school email address.
