[Setup]
AppName=Whisper GUI
AppVersion=1.0
DefaultDirName={autopf}\WhisperGUI
DefaultGroupName=Whisper GUI
OutputDir=installer
OutputBaseFilename=WhisperGUI-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\WhisperGUI.exe

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"

[Files]
Source: "dist\WhisperGUI\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Whisper GUI"; Filename: "{app}\WhisperGUI.exe"
Name: "{autodesktop}\Whisper GUI"; Filename: "{app}\WhisperGUI.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\WhisperGUI.exe"; Description: "Launch Whisper GUI"; Flags: nowait postinstall skipifsilent
