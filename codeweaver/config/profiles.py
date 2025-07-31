DEFAULT_IGNORE_PROFILES = {
    "Python (General)": [
        "__pycache__/", "venv/", ".venv/", "*/venv/*", "*.pyc", "*.egg-info/",
        "build/", "dist/", ".env", "data/", "notebooks/"
    ],
    "Node.js (React/Next)": [
        "node_modules/", ".next/", "build/", "dist/", "coverage/",
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", ".env*"
    ],
    "General Purpose": [
        ".git/", ".vscode/", ".idea/", "*.log", "*.tmp", "*.swp", ".DS_Store"
    ],
    "Media & Docs": [
        "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.svg", "*.ico",
        "*.mp3", "*.mp4", "*.avi", "*.mov", "*.webm",
        "*.pdf", "*.zip", "*.gz", "*.tar", "*.rar",
        "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx"
    ]
}