; installer.iss — Inno Setup 6 script for Wikipedia Assistant
;
; Prerequisites:
;   1. Complete PyInstaller build: python -m PyInstaller wiki-offline.spec
;      (dist\WikiOffline\ must exist and be fully populated)
;   2. Install Inno Setup 6: https://jrsoftware.org/isdl.php
;
; Build (from project root, Windows):
;   iscc installer.iss
;   -- or open in Inno Setup GUI and press F9 --
;
; Output: installer\WikiOffline-Setup.exe  (~1-2 GB after lzma2 compression)

#define AppName      "Wikipedia Assistant"
#define AppVersion   "0.1.0"
#define AppPublisher "wiki-offline"
#define AppExeName   "WikiOffline.exe"
#define AppID        "{{A3F2B1C4-7E8D-4F9A-B2C3-D4E5F6A7B8C9}"

[Setup]
AppId={#AppID}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/ianbarjordan/offline-wikipedia
AppSupportURL=https://github.com/ianbarjordan/offline-wikipedia/issues
DefaultDirName={localappdata}\WikiOffline
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
OutputDir=installer
OutputBaseFilename=WikiOffline-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
    GroupDescription: "Additional icons:"; Flags: checkedonce

[Files]
Source: "dist\WikiOffline\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    Comment: "Offline Wikipedia assistant — no internet required"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; \
    Tasks: desktopicon; Comment: "Offline Wikipedia assistant"

[Run]
Filename: "{app}\{#AppExeName}"; \
    Description: "&Launch {#AppName} now"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
