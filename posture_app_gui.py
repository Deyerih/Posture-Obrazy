from __future__ import annotations

import argparse
import contextlib
import io
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from benchmark_charts import create_benchmark_figure
from benchmark_charts import create_placeholder_figure
from benchmark_charts import discover_benchmark_runs
from benchmark_charts import load_benchmark_run
from benchmark_charts import save_benchmark_figure
from benchmark_pose_models import benchmark
from predict_posture_images import predict_from_images


class PostureApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Posture Obrazy")
        self.root.geometry("1120x760")
        self.root.minsize(980, 660)

        self.worker_thread: threading.Thread | None = None
        self.current_figure_canvas: FigureCanvasTkAgg | None = None
        self.activity_after_id: str | None = None
        self.task_started_at: float | None = None
        self.current_task_name = ""
        self.action_widgets: list[tk.Widget] = []

        self.predict_input_var = tk.StringVar(value="test_images")
        self.predict_model_path_var = tk.StringVar(value="model/posture_level_model.joblib")
        self.predict_yolo_model_var = tk.StringVar(value="yolo11n-pose.pt")
        self.predict_conf_var = tk.StringVar(value="0.25")
        self.predict_save_visuals_var = tk.BooleanVar(value=True)
        self.predict_output_dir_var = tk.StringVar(value="outputs")

        self.benchmark_input_var = tk.StringVar(value="test_images")
        self.benchmark_model_path_var = tk.StringVar(value="model/posture_level_model.joblib")
        self.benchmark_engines_var = tk.StringVar(value="yolo,torchvision")
        self.benchmark_yolo_models_var = tk.StringVar(value="yolo11n-pose.pt")
        self.benchmark_torchvision_models_var = tk.StringVar(
            value="keypointrcnn_resnet50_fpn"
        )
        self.benchmark_warmup_var = tk.StringVar(value="1")
        self.benchmark_log_dir_var = tk.StringVar(value="benchmark_logs")

        self.camera_model_path_var = tk.StringVar(value="model/posture_level_model.joblib")
        self.camera_id_var = tk.StringVar(value="0")

        self.chart_log_dir_var = tk.StringVar(value="benchmark_logs")
        self.chart_run_var = tk.StringVar()
        self.chart_metadata_var = tk.StringVar(value="No benchmark selected.")

        self.status_var = tk.StringVar(value="Ready")
        self.phase_var = tk.StringVar(value="Idle")
        self.elapsed_var = tk.StringVar(value="Elapsed: 0.0s")
        self.last_result_var = tk.StringVar(value="No completed task yet.")

        self._build_ui()
        self.refresh_runs()

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=10)
        root_frame.pack(fill="both", expand=True)

        self._build_status_bar(root_frame)

        notebook = ttk.Notebook(root_frame)
        notebook.pack(fill="both", expand=True, pady=(10, 10))

        predict_tab = ttk.Frame(notebook, padding=10)
        benchmark_tab = ttk.Frame(notebook, padding=10)
        charts_tab = ttk.Frame(notebook, padding=10)

        notebook.add(predict_tab, text="Predict")
        notebook.add(benchmark_tab, text="Benchmark")
        notebook.add(charts_tab, text="Charts")

        self._build_predict_tab(predict_tab)
        self._build_benchmark_tab(benchmark_tab)
        self._build_charts_tab(charts_tab)

        log_frame = ttk.LabelFrame(root_frame, text="Output Log", padding=8)
        log_frame.pack(fill="both", expand=False)
        self.output_text = ScrolledText(log_frame, height=9, wrap="word")
        self.output_text.pack(fill="both", expand=True)

    def _build_status_bar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent)
        bar.pack(fill="x", expand=False)

        status_card = ttk.LabelFrame(bar, text="Status", padding=10)
        status_card.pack(side="left", fill="x", expand=True)

        ttk.Label(
            status_card,
            textvariable=self.status_var,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(status_card, textvariable=self.phase_var).pack(anchor="w", pady=(2, 0))
        ttk.Label(status_card, textvariable=self.elapsed_var).pack(anchor="w", pady=(2, 0))
        ttk.Label(status_card, textvariable=self.last_result_var, foreground="#555555").pack(
            anchor="w", pady=(4, 0)
        )

        self.progress_bar = ttk.Progressbar(bar, mode="indeterminate", length=220)
        self.progress_bar.pack(side="left", padx=(12, 0), pady=8)

    def _build_predict_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)

        form_card = ttk.LabelFrame(parent, text="Image Prediction", padding=10)
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._add_path_row(form_card, 0, "Input", self.predict_input_var, is_dir=True)
        self._add_path_row(
            form_card, 1, "Model", self.predict_model_path_var, is_dir=False
        )
        self._add_text_row(form_card, 2, "YOLO pose model", self.predict_yolo_model_var)
        self._add_text_row(form_card, 3, "Confidence", self.predict_conf_var)
        self._add_path_row(
            form_card, 4, "Output dir", self.predict_output_dir_var, is_dir=True
        )
        ttk.Checkbutton(
            form_card,
            text="Save annotated visuals",
            variable=self.predict_save_visuals_var,
        ).grid(row=5, column=1, sticky="w", pady=(8, 0))

        side_card = ttk.LabelFrame(parent, text="Actions", padding=10)
        side_card.grid(row=0, column=1, sticky="nsew")

        self._register_action_widget(
            ttk.Button(side_card, text="Run prediction", command=self.on_run_prediction)
        ).pack(fill="x")
        self._register_action_widget(
            ttk.Button(side_card, text="Open camera window", command=self.on_open_camera)
        ).pack(fill="x", pady=(8, 0))

        ttk.Separator(side_card).pack(fill="x", pady=12)

        ttk.Label(side_card, text="Camera model").pack(anchor="w")
        ttk.Entry(side_card, textvariable=self.camera_model_path_var).pack(
            fill="x", pady=(4, 8)
        )
        ttk.Label(side_card, text="Camera ID").pack(anchor="w")
        ttk.Entry(side_card, textvariable=self.camera_id_var).pack(fill="x", pady=(4, 0))

        ttk.Label(
            side_card,
            text=(
                "Use this tab for quick image prediction.\n"
                "The camera opens in a separate window and closes with Q."
            ),
            justify="left",
            foreground="#555555",
        ).pack(anchor="w", pady=(14, 0))

    def _build_benchmark_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=3)
        parent.columnconfigure(1, weight=2)

        form_card = ttk.LabelFrame(parent, text="Benchmark Setup", padding=10)
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self._add_path_row(form_card, 0, "Input", self.benchmark_input_var, is_dir=True)
        self._add_path_row(
            form_card, 1, "Model", self.benchmark_model_path_var, is_dir=False
        )
        self._add_text_row(form_card, 2, "Engines", self.benchmark_engines_var)
        self._add_text_row(form_card, 3, "YOLO models", self.benchmark_yolo_models_var)
        self._add_text_row(
            form_card, 4, "Torchvision models", self.benchmark_torchvision_models_var
        )
        self._add_text_row(form_card, 5, "Warmup runs", self.benchmark_warmup_var)
        self._add_path_row(
            form_card, 6, "Log dir", self.benchmark_log_dir_var, is_dir=True
        )

        side_card = ttk.LabelFrame(parent, text="Run", padding=10)
        side_card.grid(row=0, column=1, sticky="nsew")

        self._register_action_widget(
            ttk.Button(side_card, text="Run benchmark", command=self.on_run_benchmark)
        ).pack(fill="x")

        ttk.Label(
            side_card,
            text=(
                "Charts are now generated automatically together with the benchmark logs.\n"
                "Default comparison is YOLO vs Torchvision Keypoint R-CNN."
            ),
            justify="left",
            foreground="#7a4f01",
        ).pack(anchor="w", pady=(12, 0))

    def _build_charts_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="Chart Source", padding=10)
        controls.pack(fill="x", expand=False)

        self._add_path_row(controls, 0, "Log dir", self.chart_log_dir_var, is_dir=True)

        ttk.Label(controls, text="Run").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.run_combo = ttk.Combobox(
            controls, textvariable=self.chart_run_var, state="readonly", width=42
        )
        self.run_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _event: self.load_selected_run())

        chart_button_row = ttk.Frame(controls)
        chart_button_row.grid(row=1, column=2, sticky="e", padx=(8, 0), pady=(8, 0))
        self._register_action_widget(
            ttk.Button(chart_button_row, text="Refresh", command=self.refresh_runs)
        ).pack(side="left")
        self._register_action_widget(
            ttk.Button(chart_button_row, text="Save PNG", command=self.on_save_chart)
        ).pack(side="left", padx=(8, 0))

        controls.columnconfigure(1, weight=1)

        meta_card = ttk.LabelFrame(parent, text="Run Info", padding=10)
        meta_card.pack(fill="x", expand=False, pady=(10, 10))
        ttk.Label(meta_card, textvariable=self.chart_metadata_var, justify="left").pack(
            anchor="w"
        )

        self.chart_frame = ttk.LabelFrame(parent, text="Chart Preview", padding=6)
        self.chart_frame.pack(fill="both", expand=True)
        self._show_chart(
            create_placeholder_figure(
                title="No chart selected",
                message="Run a benchmark or choose an existing log from benchmark_logs.",
            )
        )

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        is_dir: bool,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Button(
            parent,
            text="Browse",
            command=lambda: self._browse_path(variable, is_dir=is_dir),
        ).grid(row=row, column=2, sticky="e", padx=(8, 0), pady=(6, 0))
        parent.columnconfigure(1, weight=1)

    def _add_text_row(
        self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=(6, 0)
        )
        parent.columnconfigure(1, weight=1)

    def _register_action_widget(self, widget: tk.Widget) -> tk.Widget:
        self.action_widgets.append(widget)
        return widget

    def _browse_path(self, variable: tk.StringVar, is_dir: bool) -> None:
        if is_dir:
            selected = filedialog.askdirectory(initialdir=variable.get() or ".")
        else:
            current = Path(variable.get() or ".")
            initial_dir = str(current.parent if current.parent.exists() else Path("."))
            selected = filedialog.askopenfilename(initialdir=initial_dir)
        if selected:
            variable.set(selected)

    def on_run_prediction(self) -> None:
        def task() -> None:
            predict_from_images(
                model_path=Path(self.predict_model_path_var.get()),
                image_input=Path(self.predict_input_var.get()),
                yolo_model_name=self.predict_yolo_model_var.get(),
                conf=float(self.predict_conf_var.get()),
                save_visuals=bool(self.predict_save_visuals_var.get()),
                output_dir=Path(self.predict_output_dir_var.get()),
            )

        self._run_background_task("Prediction", task)

    def on_run_benchmark(self) -> None:
        def task() -> None:
            args = argparse.Namespace(
                input=Path(self.benchmark_input_var.get()),
                model_path=Path(self.benchmark_model_path_var.get()),
                engines=self.benchmark_engines_var.get(),
                yolo_models=self.benchmark_yolo_models_var.get(),
                yolo_conf=0.25,
                torchvision_models=self.benchmark_torchvision_models_var.get(),
                torchvision_score_threshold=0.75,
                mediapipe_complexities="",
                mediapipe_min_detection_conf=0.5,
                mediapipe_min_tracking_conf=0.5,
                mediapipe_model_path=Path("models") / "pose_landmarker_full.task",
                warmup_runs=int(self.benchmark_warmup_var.get()),
                log_dir=Path(self.benchmark_log_dir_var.get()),
            )
            benchmark(args)

        self._run_background_task("Benchmark", task, refresh_runs=True)

    def on_open_camera(self) -> None:
        try:
            camera_id = int(self.camera_id_var.get())
        except ValueError:
            messagebox.showerror("Camera", "Camera ID must be an integer.")
            return

        command = [
            sys.executable,
            str(Path(__file__).with_name("live_posture_camera.py")),
            "--model-path",
            self.camera_model_path_var.get(),
            "--camera-id",
            str(camera_id),
        ]
        subprocess.Popen(command)
        self.status_var.set("Camera window started")
        self.phase_var.set("Live camera is running in a separate window")
        self.last_result_var.set("Close the camera window with Q.")
        self._append_output("Camera window started. Close it with Q.\n")

    def refresh_runs(self) -> None:
        log_dir = Path(self.chart_log_dir_var.get())
        run_paths = discover_benchmark_runs(log_dir)
        self.run_combo["values"] = [path.name for path in run_paths]
        if not run_paths:
            self.chart_run_var.set("")
            self.chart_metadata_var.set(f"No benchmark logs found in {log_dir}.")
            self._show_chart(
                create_placeholder_figure(
                    title="No logs found",
                    message="Run a benchmark first or choose another log directory.",
                )
            )
            return

        if self.chart_run_var.get() not in self.run_combo["values"]:
            self.chart_run_var.set(run_paths[0].name)
        self.load_selected_run()

    def load_selected_run(self) -> None:
        selected_name = self.chart_run_var.get()
        if not selected_name:
            return

        details_path = Path(self.chart_log_dir_var.get()) / selected_name
        try:
            run = load_benchmark_run(details_path)
            self.chart_metadata_var.set(
                f"Run: {run.run_id} | Input: {run.metadata.get('input', '-')} | "
                f"Engines: {run.metadata.get('engines', '-')} | "
                f"Images: {run.metadata.get('images_count', '-')} | "
                f"Records: {run.metadata.get('records_count', '-')}"
            )
            self._show_chart(create_benchmark_figure(run))
        except Exception as exc:  # noqa: BLE001
            self.chart_metadata_var.set(f"Could not load {selected_name}: {exc}")
            self._show_chart(
                create_placeholder_figure(
                    title="Chart unavailable",
                    message=f"Failed to read {selected_name}.\n{exc}",
                )
            )

    def on_save_chart(self) -> None:
        selected_name = self.chart_run_var.get()
        if not selected_name:
            self.status_var.set("No benchmark selected")
            self.phase_var.set("Choose a benchmark run before saving a chart")
            return

        default_path = Path(self.chart_log_dir_var.get()) / selected_name.replace(
            "_details.json", "_charts.png"
        )
        selected = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_path.name,
            initialdir=str(default_path.parent),
            filetypes=[("PNG image", "*.png")],
        )
        if not selected:
            return

        try:
            saved_path = save_benchmark_figure(
                details_path=Path(self.chart_log_dir_var.get()) / selected_name,
                output_path=Path(selected),
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Charts", f"Could not save chart:\n{exc}")
            self._append_output(f"[Charts] Save failed: {exc}\n")
            return

        self.status_var.set("Chart saved")
        self.phase_var.set("Chart export completed")
        self.last_result_var.set(str(saved_path))
        self._append_output(f"Saved chart image to: {saved_path}\n")

    def _show_chart(self, figure) -> None:
        self._clear_chart()
        canvas = FigureCanvasTkAgg(figure, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.current_figure_canvas = canvas

    def _clear_chart(self) -> None:
        if self.current_figure_canvas is not None:
            self.current_figure_canvas.get_tk_widget().destroy()
            self.current_figure_canvas = None

    def _run_background_task(
        self, task_name: str, callback, refresh_runs: bool = False
    ) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.status_var.set("Busy")
            self.phase_var.set("Wait for the current task to finish")
            return

        self._set_busy_state(task_name, True)
        self._append_output(f"[{task_name}] Started...\n")

        def runner() -> None:
            buffer = io.StringIO()
            error_text = ""
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                try:
                    callback()
                except Exception:  # noqa: BLE001
                    error_text = traceback.format_exc()
            self.root.after(
                0,
                lambda: self._finish_background_task(
                    task_name=task_name,
                    output=buffer.getvalue(),
                    error_text=error_text,
                    refresh_runs=refresh_runs,
                ),
            )

        self.worker_thread = threading.Thread(target=runner, daemon=True)
        self.worker_thread.start()

    def _set_busy_state(self, task_name: str, is_busy: bool) -> None:
        self.current_task_name = task_name if is_busy else ""
        if is_busy:
            self.task_started_at = time.time()
            self.status_var.set(f"{task_name} running")
            self.phase_var.set("Processing... controls are temporarily locked")
            self.elapsed_var.set("Elapsed: 0.0s")
            self.progress_bar.start(10)
            for widget in self.action_widgets:
                widget.configure(state="disabled")
            self._schedule_activity_tick()
            return

        if self.activity_after_id is not None:
            self.root.after_cancel(self.activity_after_id)
            self.activity_after_id = None
        self.progress_bar.stop()
        for widget in self.action_widgets:
            widget.configure(state="normal")
        self.task_started_at = None

    def _schedule_activity_tick(self) -> None:
        self._update_elapsed_label()
        self.activity_after_id = self.root.after(250, self._schedule_activity_tick)

    def _update_elapsed_label(self) -> None:
        if self.task_started_at is None:
            self.elapsed_var.set("Elapsed: 0.0s")
            return
        elapsed = time.time() - self.task_started_at
        self.elapsed_var.set(f"Elapsed: {elapsed:.1f}s")

    def _finish_background_task(
        self, task_name: str, output: str, error_text: str, refresh_runs: bool
    ) -> None:
        self._set_busy_state(task_name, False)
        if output:
            self._append_output(output)
        if error_text:
            self.status_var.set(f"{task_name} failed")
            self.phase_var.set("See the output log for details")
            self.last_result_var.set("Task finished with an error.")
            self._append_output(error_text)
            messagebox.showerror(task_name, f"{task_name} failed. See output below.")
            return

        self.status_var.set(f"{task_name} completed")
        self.phase_var.set("Finished successfully")
        self.last_result_var.set(f"{task_name} completed without errors.")
        self._append_output(f"[{task_name}] Finished.\n")
        if refresh_runs:
            self.refresh_runs()

    def _append_output(self, text: str) -> None:
        self.output_text.insert("end", text)
        self.output_text.see("end")


def main() -> None:
    root = tk.Tk()
    PostureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
