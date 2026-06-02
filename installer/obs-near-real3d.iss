; Inno Setup script for near Real 3D (obs-near-real3d).
; Built by package.ps1 / CI. The staged plugin tree must exist at
;   ..\dist\stage\obs-near-real3d\{bin\64bit, data, ...}
; Pass the version with:  ISCC.exe /DAppVersion=0.2.3 obs-near-real3d.iss

#ifndef AppVersion
  #define AppVersion "0.2.3-dev"
#endif

[Setup]
AppId={{B7A3F1E2-4C5D-4A6B-9E8F-1A2B3C4D5E6F}
AppName=near Real 3D (OBS plugin)
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
OutputBaseFilename=near-real3d-{#AppVersion}-windows-x64-installer
UninstallDisplayName=near Real 3D (OBS plugin)
WizardStyle=modern
SolidCompression=yes

[Files]
Source: "..\dist\stage\obs-near-real3d\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Messages]
WelcomeLabel2=This will install the near Real 3D filter for OBS Studio.%n%nClose OBS Studio before continuing, then restart it after install.
