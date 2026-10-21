"""The scope-profiler tag of plasmacoll's profiling regions.

The regions are opt-in: :mod:`plasmacoll` registers the tag on import, so they
record nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["plasmacoll"])``.
"""

#: Tags of every plasmacoll region; the first is the opt-in tag.
PROFILE_TAGS = ("plasmacoll",)
