# Repository script review: bugs and improvement points

## High-priority bugs

1. **`train_model.py` can crash when `training_record.csv` has string model numbers.**  
   `model_number = log_df['model_number'].max() + 1` assumes numeric dtype. If the CSV is read with mixed/object dtype (easy when manually edited), `max()` may be a string and `+ 1` will fail.  
   **Improvement:** coerce with `pd.to_numeric(..., errors='coerce')` before incrementing.

2. **`calculate_model_performance.py` writes a `Path` object directly to CSV row dict.**  
   `"out_dir": out_path` is not serialized explicitly and relies on implicit conversion.  
   **Improvement:** store `str(out_path)` for predictable CSV output.

3. **`run_cellpose_per_chunk.py` re-initializes the Cellpose model for every chunk.**  
   This is a major performance bug; model load is expensive and can dominate runtime for many chunks.  
   **Improvement:** instantiate `CellposeModel` once before the loop and reuse it.

4. **`run_cellpose_per_chunk.py` can silently overflow label IDs when saving masks.**  
   `predicted_masks.astype(np.uint8)` truncates labels above 255 and corrupts instance IDs.  
   **Improvement:** save as `uint16` (or keep original dtype).

5. **`utils.calculate_z_numbers` divides by zero for single-plane stacks.**  
   `step = distance / (no_z_planes - 1)` fails when `no_z_planes == 1`.  
   **Improvement:** return `[first_z_number]` early for one plane.

## Reliability and portability concerns

1. **Hard-coded GPU usage in multiple scripts (`gpu=True`).**  
   On non-GPU systems this can fail or degrade portability.  
   **Improvement:** expose a user `use_gpu` flag, or auto-detect availability.

2. **Windows-only path splitting in `train_model.py`.**  
   `i.split("\\")[-1]` is not cross-platform.  
   **Improvement:** use `Path(i).name`.

3. **`split_annotated_stacks.py` assumes strict filename schema and fixed indices.**  
   Index access like `split_stem[10]` can break with any naming variation.  
   **Improvement:** validate filename structure and fail with a descriptive error.

4. **`run_cellpose_per_image.py` shadows Python built-in name `input`.**  
   This is not fatal but makes debugging and extension harder.  
   **Improvement:** rename to `input_path`.

5. **Potential mismatch between configured threshold/normalize and saved metadata.**  
   `generate_cellpose_npy_dict` always writes `flow_threshold=0.4` and `normalize=True`, even when scripts run with different settings.  
   **Improvement:** pass active inference settings into this helper.

## Maintainability improvements

1. **Convert scripts to CLIs using `argparse`.**  
   Nearly all scripts require manual edits in source to change paths/params.

2. **Add basic logging and structured error messages.**  
   Most scripts use only `print`, making debugging and automation harder.

3. **Add minimal automated checks (e.g., `python -m compileall .`, unit tests for `utils.py`).**

4. **Document expected filename patterns explicitly in README.**  
   Several scripts depend on pattern-based parsing but this is only implicit.

5. **Consider extracting shared model-loading/eval helpers.**  
   Inference logic is duplicated across run scripts, making drift likely.

## Quick wins to implement first

1. Reuse model instance in `run_cellpose_per_chunk.py`.
2. Save masks as `uint16` in chunk workflow.
3. Fix `model_number` numeric coercion in `train_model.py`.
4. Handle one-plane stacks in `utils.calculate_z_numbers`.
5. Serialize `out_dir` as string in `calculate_model_performance.py`.
