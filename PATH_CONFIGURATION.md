# Path configuration

Public scripts must not depend on an author's machine-specific path.

## Python scripts

Where a project root is needed, scripts use either their command-line
`--root`/`--project-root` argument or the environment variable:

`GIS_ANOMALY_PROJECT_ROOT`

If neither is supplied by a legacy script, the current working directory is
used as the portable fallback.

Example (PowerShell):

```powershell
$env:GIS_ANOMALY_PROJECT_ROOT = "D:\research\outlier-paper\Code"
```

Example (Linux):

```bash
export GIS_ANOMALY_PROJECT_ROOT=/data/outlier-paper/Code
```

## Historical notebooks

Some historical notebooks originally used a parent workspace directory.
Their personal path has been replaced with:

`GIS_ANOMALY_WORKSPACE_ROOT`

These notebooks are included for research provenance and are not the
authoritative final Pure scoring workflow.
