"""ExoSNN / astro TESS vetting pipeline.

Run everything from the project root (the directory that CONTAINS this `root/`
folder), always as a module, e.g.

    python -m root.main_pipeline --tic 172518755 --sector 21
    python -m root.train_all
    streamlit run root/ux/dashboard.py

All internal imports are absolute `root.*` imports. (An earlier revision put
`root/` itself on sys.path so bare `data`, `views`, `detection`, ... resolved;
that made every module importable under two names and let those generic names
shadow third-party packages, so it was removed.)
"""
