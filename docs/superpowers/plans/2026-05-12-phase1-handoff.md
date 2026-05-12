# GPU Pipeline Phase 1 — Handoff for Continuation

> **For the agent picking this up:** This is in-flight execution of the plan at `docs/superpowers/plans/2026-05-10-gpu-pipeline-latency-reduction-phase1.md`. The subagent-driven-development workflow was started, then the org hit a monthly usage limit on subagent dispatch, so the controller dropped into inline execution. Continue inline (no subagents) until completion.

## Branch and SHAs

- Branch: `qt-desktop-app`
- Latest commit at handoff time: `ba1f99d feat(stereo_align): add GPU warp_pair_gpu using cached matrices`
- Plan: `docs/superpowers/plans/2026-05-10-gpu-pipeline-latency-reduction-phase1.md`
- Spec: `docs/superpowers/specs/2026-05-10-gpu-pipeline-latency-reduction-design.md`

## Progress

| Task | Status | Commit |
| ---- | ------ | ------ |
| 1. Spike — verify cv2.cuda_GpuMat ROI semantics | ✅ done | `661b62e` |
| 2. Add `use_gpu_pipeline` config flag | ✅ done | `4f48474` |
| **Baseline restoration** (popped WIP from stash) | ✅ done | `d049d80` |
| 3. Implement `fill_holes_cross_gpu` | ✅ done | `fbbaba0` |
| 4. Add `warp_pair_gpu` to StereoAligner | ✅ done | `ba1f99d` |
| 5. Add `process_pair_gpu` to StereoProcessor | ⏳ pending | — |
| 6. Add `apply_nudge_gpu` to CalibrationOverlay | ⏳ pending | — |
| 7. Implement `GpuPipeline` orchestrator | ⏳ pending | — |
| 8. Wire `pipeline_worker.py` to use `GpuPipeline` | ⏳ pending | — |
| 9. Remove `_can_process_now` throttle | ⏳ pending | — |
| 10. Add `cv2.cuda_Event` GPU-side timing | ⏳ pending | — |
| 11. Performance regression test | ⏳ pending | — |
| Final code review | ⏳ pending | — |

## Critical gotchas discovered

These are not in the plan — discovered during execution and will trip you up if you trust the plan literally.

### 1. The plan was written against a WIP-state working tree, not the committed baseline

When the plan was authored, `git stash` had not yet happened, so the author read post-WIP file contents thinking they were committed. The WIP included a refactor of `pipeline_worker.py` (~500 lines added, with helpers like `_can_process_now`, `_fill_holes_cross`, `_compute_disparity_gpu_from_bgr`, `_warp_affine_fast`, `_cvtcolor_and_resize_fast`, `_apply_camera_flips`, etc.) and a `PerformanceCfg` dataclass in `src/config.py`. The WIP was reapplied in commit `d049d80` after Task 2.

Line numbers in the plan are off by ~5-30 because of this. Use Grep to find method definitions by name — don't trust the absolute line numbers.

### 2. cv2.cuda API quirks on this build (opencv-python-cuda 4.12.0 / CUDA 12.9)

Verified the hard way:
- `cv2.cuda.copyTo` (module-level function) — **does NOT exist.** Use `GpuMat.copyTo(mask, dst)` instance method instead.
- `cv2.cuda.compare` — requires both args to be matrices. For scalar comparison use `cv2.cuda.compareWithScalar(src, scalar, cmpop)`.
- `cv2.cuda.merge(channels)` — returns a host `numpy.ndarray`, **not** a `GpuMat`. Avoid using it; it forces a host round-trip. If you need to recombine channels on-device, use per-channel `copyTo` with masks instead.
- `cv2.cuda.split(gpu)` — returns a list of `cv2.cuda_GpuMat` (correct, GPU-resident). Verified by a guard test.
- `cv2.cuda.warpAffine` — output differs from CPU `cv2.warpAffine` at image borders (CPU interpolates with near-edge pixels; GPU treats them as out-of-bounds). Interior pixels match within ~4-5 on uint8 with bilinear interp for smooth inputs. **In tests, restrict per-pixel comparison to the interior (e.g., `frame[5:-5, 5:-5]`) and use smooth gradient inputs instead of random noise** (random noise amplifies any sub-pixel interp diff). The border discrepancy is harmless in the real pipeline because `fill_holes_cross_gpu` runs immediately after and substitutes the partner eye's pixel.

### 3. `GpuMat.copyTo(mask, dst)` argument order

The instance method signature is `src.copyTo(mask, dst)` — destination is the LAST positional. This is the opposite of some other APIs. The `gpu_l.clone()` pattern is used to snapshot a source before mutation when ordering matters (see `fill_holes_cross_gpu` for the canonical example).

## Workflow you should follow

The controller exhausted its subagent budget. Continue **inline** (controller does the work itself; no Task tool subagents). Per task:

1. **Read** the task verbatim from the plan file.
2. **Check the current state** of the files you'll touch — line numbers may have drifted from the plan.
3. **Write the test first** (TDD), run it, confirm it fails for the expected reason.
4. **Implement** the change.
5. **Run tests** including the full suite if integration-heavy: `.venv\Scripts\python.exe -m pytest tests\ -x -q --ignore=tests\test_recording_worker.py`
6. **Commit** with the message from the plan.

When you finish a task, update the table above (mark it done with its SHA) and update the TodoWrite. If the user wants you to also do the spec review + code quality review steps, those can happen after Task 11 in a single sweep, since you're no longer using subagents.

## Environment

- Working dir: `e:\Documents\University\year2\Semester1\Research\SurgeryRobot\piccolo`
- Shell: PowerShell (use `.venv\Scripts\python.exe`, not `python`)
- OpenCV: `opencv-python-cuda 4.12.0+a1a2e91` against CUDA 12.9
- The `tests/_cuda_helpers.py` file provides `skip_if_no_cuda()` — use it at the top of every CUDA-touching test.
- `pipeline_worker.py` is currently 593 lines after baseline restoration. The methods it has are listed in commit `d049d80`'s diff.

## What NOT to repeat

- Don't widen test tolerances to make failing tests pass. If output diverges by more than a few uint8 levels in the interior, investigate the API — the most likely culprit is one of the quirks listed above.
- Don't use `cv2.cuda.merge` + `gpu.upload()` to recombine channels — that's a host round-trip and defeats the whole point of GPU residency. Use `GpuMat.copyTo(mask, dst)` instead.
- Don't trust plan line numbers — Grep for method names.

## Where to resume

Resume at Task 5: `process_pair_gpu` in `src/stereo_processor.py`. The CPU `process_pair` is at `src/stereo_processor.py:190` (after baseline restoration). Add `process_pair_gpu` directly after it. See plan Task 5 for the exact code.

When you implement the test, **expect to need a smooth gradient and an interior-only comparison** for the same reason described in gotcha #2 above. The plan's literal `<= 2` tolerance over the full SBS frame will not pass; use `<= 5` over the interior and document why.
