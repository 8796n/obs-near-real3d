; Inno Setup script for near Real 3D (obs-near-real3d).
; Built by package.ps1 / CI. The staged plugin tree must exist at
;   ..\dist\stage\obs-near-real3d\{bin\64bit, data, ...}
; Pass the version with:  ISCC.exe /DAppVersion=0.3.1 obs-near-real3d.iss
; Pass /DLite for the lightweight (392x224 model) build. Both variants share the
; same AppId and install path, so installing one replaces the other (the DLL and
; install dir are identical -- only the bundled model differs).

#ifndef AppVersion
  #define AppVersion "0.5.0-dev"
#endif
#ifdef Lite
  #define VariantSuffix "-lite"
  #define VariantLabel " Lite (392x224)"
#else
  #define VariantSuffix ""
  #define VariantLabel ""
#endif

[Setup]
AppId={{B7A3F1E2-4C5D-4A6B-9E8F-1A2B3C4D5E6F}
AppName=near Real 3D{#VariantLabel} (OBS plugin)
AppVersion={#AppVersion}
AppPublisher=8796n
AppPublisherURL=https://github.com/8796n/obs-near-real3d
; Per-user OBS plugin path scanned by OBS on Windows (no admin needed).
DefaultDirName={commonappdata}\obs-studio\plugins\obs-near-real3d
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=near-real3d-{#AppVersion}{#VariantSuffix}-windows-x64-installer
UninstallDisplayName=near Real 3D{#VariantLabel} (OBS plugin)
WizardStyle=modern
SolidCompression=yes
; Pick the wizard language from the OS UI language automatically (no prompt).
; English is listed first, so it's the fallback on any non-Japanese system.
ShowLanguageDialog=no

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"

[Files]
Source: "..\dist\stage\obs-near-real3d\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

; Translate the custom welcome line per language (prefix = the [Languages] Name).
[Messages]
en.WelcomeLabel2=This will install the near Real 3D filter for OBS Studio.%n%nClose OBS Studio before continuing, then restart it after install.
ja.WelcomeLabel2=OBS Studio 用の near Real 3D フィルターをインストールします。%n%n続行する前に OBS Studio を終了し、インストール後に再起動してください。
