# Vendored — `scene_graph/`

This directory is **vendored** from the `sravani-scenegraph-demo` branch of
this same repository. Do not edit files in place — see "Resync" below.

## Why vendored, not imported

Both branches live in clones of `github.com/e-lab/Alex-robot.git`. The demo
branch is **local-only** (never pushed to `origin` as of this writing) and
contains the production scene-graph implementation that the May 15 demo
depends on.

We need the same scene graph for the autonomous-navigation work
(`PLAN/autonomous_navigation_plan.md` Phase 2+) — porting object detections
to a goal XYZ requires the dedup, IoU-merge, lock-on, and serialisation
machinery that already exists on the demo branch. Re-implementing it on
this branch would mean maintaining two divergent copies of the same code.

Vendoring lets us:
- Use the demo branch's tested implementation today (no waiting on a merge).
- Keep our autonomy work editable independently while the demo branch
  stabilises for May 15.
- Resync deterministically when the demo branch evolves.

## Source

- **Repo:** `github.com/e-lab/Alex-robot`
- **Branch (local clone):** `sravani-scenegraph-demo`
- **Source clone path:** `/home/sravani/E-Lab/Spring2026/repos/Alex-robot/`
- **Source commit:** `cfc14dfad311af99065ff69f46e14afa5827333c`
  (`feat(demo): build-stats banner at finalisation for the audience`,
  Apr 28 2026)

## What we copied

```
scene_graph/
├── __init__.py
├── graph/             # SceneGraph + node dataclasses + JSON (de)serialise
│   ├── node_types.py
│   ├── scene_graph.py
│   ├── serialize.py
│   └── __init__.py
├── geometry/          # pinhole unproject + AABB + point cloud helpers
│   ├── unprojection.py
│   ├── bbox.py
│   ├── pointcloud.py
│   └── __init__.py
├── detection/         # SAM3 wrapper + mask-pooled embeddings
│   ├── sam3_detector.py
│   ├── embeddings.py
│   └── __init__.py
├── layers/            # Object layer only
│   ├── object_layer.py
│   └── __init__.py
├── association/       # IoU + embedding-similarity dedup, periodic cleanup
│   ├── merge.py
│   ├── dedup_rules.py
│   ├── similarity.py
│   └── __init__.py
└── pipeline/          # Per-tick frame_loop + atomic snapshot manager
    ├── frame_loop.py
    ├── snapshot_manager.py
    └── __init__.py
```

## What we deliberately skipped (and why)

| Skipped         | Reason                                                              |
|-----------------|---------------------------------------------------------------------|
| `discovery/`    | VLM query (Anthropic API). Phase 5 of autonomy plan, deferred.      |
| `viz/`          | Rerun blueprints for the demo. Demo-only.                           |
| `data/`         | `fixture_recorder.py` + `occupancy_accumulator.py`. Replay-only.    |
| `labeling/`     | (empty stub on source branch.)                                      |
| `layers/place_layer.py`     | Free-space waypoints. Phase 3+ obstacle work may revisit. |
| `layers/room_layer{,_v2}.py` | Room segmentation. Not needed for object-targeted autonomy. |
| `layers/wall_layer.py`      | Wall detection. Not needed for navigation FSM.            |
| `layers/room_edges.py`      | Room adjacency. Not needed.                               |

The skipped subpackages have no inbound references from anything we kept,
so the prune is mechanical (no code edits needed).

## Tests

We also vendored the matching subset of unit tests at the same source
commit. Tests for skipped subpackages were dropped:

| Test file dropped                | Touched skipped subpackage |
|----------------------------------|----------------------------|
| `test_fixture_recorder.py`       | `data/`                    |
| `test_occupancy_accumulator.py`  | `data/`                    |
| `test_place_layer.py`            | `layers/place_layer`       |
| `test_room_edges.py`             | `layers/room_edges`        |
| `test_room_layer.py`             | `layers/room_layer`        |
| `test_room_layer_v2.py`          | `layers/room_layer_v2`     |
| `test_wall_layer.py`             | `layers/wall_layer`        |
| `test_vlm.py`                    | `discovery/vlm`            |
| `test_vocabulary_config.py`      | needs `scripts/run_pipeline_on_fixture.py` (not vendored) |

Run vendored tests:

```bash
cd isaac-sim-rl-bringup
python -m pytest tests/unit/ -q   # 138 tests
python -m pytest tests/ -q        # 194 tests (with autonomy tests)
```

At vendor time, all 138 vendored unit tests passed.

## Resync procedure

Before each Phase-2/3/4/5 work session, check whether the source branch
has moved. If you are about to make non-trivial changes that depend on
recent improvements upstream:

```bash
SRC=/home/sravani/E-Lab/Spring2026/repos/Alex-robot
DST=/home/sravani/E-Lab/Spring2026/repos/Door/Alex-robot

# 1. See what moved upstream since cfc14df
cd "$SRC"
git log cfc14df..sravani-scenegraph-demo --oneline -- isaac-sim-rl-bringup/scene_graph/

# 2. If anything is interesting, capture the new HEAD and re-vendor
NEW_REF=$(git rev-parse sravani-scenegraph-demo)
git archive "$NEW_REF" isaac-sim-rl-bringup/scene_graph | tar -x -C "$DST/"

# 3. Re-prune the skipped subpackages (see "What we deliberately skipped")
cd "$DST/isaac-sim-rl-bringup/scene_graph"
rm -rf discovery viz data labeling
rm -f layers/place_layer.py layers/room_layer.py layers/room_layer_v2.py \
      layers/room_edges.py layers/wall_layer.py

# 4. Re-vendor matching tests at the same commit, drop the same skip list
cd "$SRC"
git archive "$NEW_REF" isaac-sim-rl-bringup/tests/unit | tar -x -C "$DST/"
cd "$DST/isaac-sim-rl-bringup/tests/unit"
rm -f test_fixture_recorder.py test_occupancy_accumulator.py \
      test_place_layer.py test_room_edges.py test_room_layer.py \
      test_room_layer_v2.py test_wall_layer.py test_vlm.py \
      test_vocabulary_config.py

# 5. Verify
cd "$DST/isaac-sim-rl-bringup"
python -m pytest tests/ -q

# 6. Update the "Source commit" + "What we copied" sections of this file.
```

If a resync introduces breakage in our autonomy code that depends on this
package, **fix our adapter, not the vendored files**. Local edits to
vendored files defeat the purpose of pinning to a known commit.

## Sunset rule

This vendored copy exists because the source branch is local-only. Once
`sravani-scenegraph-demo` (or its descendants) lands on `main` and contains
the same `scene_graph/` package, **delete this directory** and import the
canonical version from `main` directly. Track the merge:

```bash
git log origin/main -- isaac-sim-rl-bringup/scene_graph/
```

Sunset when that command shows commits — the vendored directory is no
longer needed at that point.

## License / authorship

Same repo, same author (Sravani). This is internal code-sharing within a
single project; no external license headers needed.
