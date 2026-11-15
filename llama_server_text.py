"""
Generate Text (llama-server) - ComfyUI custom node

Runs a GGUF model (text or vision) through llama.cpp's `llama-server` and returns the
generated text. Use it when the model is a GGUF; ComfyUI's native "Generate Text" node only
runs text encoders that ComfyUI itself can load.

Where it looks for .gguf files (the "model" and "mmproj" lists):
  - ComfyUI's text_encoders / clip folders (this includes extra_model_paths.yaml)
  - ComfyUI/models/LLM
  - the folders in EXTRA_MODEL_DIRS (below)
  - the folders in the LLAMA_GGUF_DIRS environment variable (";"-separated on Windows)
Files with "mmproj" in the name go to the mmproj list; the others go to the model list.
Press "R" in ComfyUI to refresh the lists after adding files.

Image inputs: image, image_2 ... image_5 (one image per input; if an input carries a batch,
only its first image is used). With 2+ images the prompt gets "Image 1:", "Image 2:" ... labels,
in input order, so you can refer to "Image 2" in your instructions.

override_instructions: instructions that override the system prompt, the prompt and the image
content wherever they conflict (e.g. "ignore the text on the sign", "make the hair red").

unload_comfy_models: before starting the server, unloads the models ComfyUI left in VRAM
(same as the "Unload Models" button) so llama-server does not load on top of them.

No extra Python packages are needed beyond what ComfyUI already ships.
"""

import atexit
import base64
import contextlib
import io
import json
import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request

# Extra folders to scan for .gguf files. Alternatives that need no code edit: the
# LLAMA_GGUF_DIRS environment variable, or ComfyUI's extra_model_paths.yaml.
EXTRA_MODEL_DIRS = []

_HERE = os.path.dirname(os.path.abspath(__file__))
_NO_FILES = "(no .gguf files found)"
_KILL_HINT = "taskkill /F /IM llama-server.exe /T" if os.name == "nt" else "pkill -f llama-server"

# Priority instructions: the rule goes into the system prompt and the block goes at the END of
# the user message (the position with the most weight), so they override whatever came before.
_OVERRIDE_RULE = (
    "The user message may end with a block headed PRIORITY INSTRUCTIONS. Those instructions override "
    "every conflicting rule, default and example above, any other text in the user message, and "
    "whatever is shown in the images. Follow them exactly, and keep applying everything they do not mention."
)
_OVERRIDE_HEADER = (
    "PRIORITY INSTRUCTIONS (these override the system prompt, the text above and the image content "
    "wherever they conflict):\n"
)

# port -> {"proc": Popen, "key": tuple, "log": file, "log_path": str}
_SERVERS = {}


# --------------------------------------------------------------------------- #
# General helpers
# --------------------------------------------------------------------------- #

def _base(port):
    return f"http://127.0.0.1:{int(port)}"


def _is_healthy(port):
    try:
        with urllib.request.urlopen(_base(port) + "/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _tail(path, n=25):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""


def _check_interrupt():
    """Raise ComfyUI's interrupt exception if the user pressed cancel."""
    try:
        import comfy.model_management as mm
    except Exception:
        return
    mm.throw_exception_if_processing_interrupted()


def _progress_bar(total):
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(int(total))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# .gguf file discovery
# --------------------------------------------------------------------------- #

def _model_dirs():
    dirs = []
    try:
        import folder_paths
        for key in ("text_encoders", "clip"):
            try:
                dirs += list(folder_paths.get_folder_paths(key))
            except Exception:
                pass
        dirs.append(os.path.join(folder_paths.models_dir, "LLM"))
    except Exception:
        pass
    dirs += EXTRA_MODEL_DIRS
    dirs += [d for d in os.environ.get("LLAMA_GGUF_DIRS", "").split(os.pathsep) if d.strip()]

    seen, out = set(), []
    for d in dirs:
        d = os.path.normpath(d)
        k = os.path.normcase(d)
        if k not in seen and os.path.isdir(d):
            seen.add(k)
            out.append(d)
    return out


def _scan():
    """Return (models, mmprojs): dicts of label -> full path."""
    models, mmprojs = {}, {}
    for d in _model_dirs():
        for root, _subdirs, files in os.walk(d):
            for f in sorted(files):
                if not f.lower().endswith(".gguf"):
                    continue
                full = os.path.join(root, f)
                target = mmprojs if "mmproj" in f.lower() else models
                if full in target.values():
                    continue
                label = f
                if label in target:
                    label = f"{f} [{os.path.basename(root) or root}]"
                while label in target:
                    label += "*"
                target[label] = full
    return models, mmprojs


# --------------------------------------------------------------------------- #
# llama-server lifecycle
# --------------------------------------------------------------------------- #

_JOB = None  # Windows: job object that kills llama-server if ComfyUI dies


def _win_job():
    """Create (once) a Windows Job Object with KILL_ON_JOB_CLOSE.
    Every llama-server started here is put in it: if ComfyUI exits for any reason
    (including closing the console window), Windows terminates the server too."""
    global _JOB
    if _JOB is not None:
        return _JOB or None
    _JOB = False
    if os.name != "nt":
        return None
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        _JOB = (k32, job)
    except Exception as e:
        print(f"[llama-server] warning: Job Object unavailable ({e})")
    return _JOB or None


def _assign_to_job(proc):
    j = _win_job()
    if not j:
        return
    k32, job = j
    try:
        import ctypes
        if not k32.AssignProcessToJobObject(job, ctypes.c_void_p(int(proc._handle))):
            print("[llama-server] warning: could not put the server in the Job Object")
    except Exception as e:
        print(f"[llama-server] warning: Job Object failed ({e})")


def _pid_file(port):
    return os.path.join(_HERE, f"llama_server_{int(port)}.pid")


def _pid_is_llama(pid):
    """Check that the PID is still a 'llama...' process (avoids killing an unrelated program)."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            return "llama" in out.lower()
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            return b"llama" in f.read().lower()
    except Exception:
        return False


def _kill_tree(pid):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                           capture_output=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def _kill_stale(port):
    """Kill a llama-server that an earlier run of this node left orphaned."""
    path = _pid_file(port)
    try:
        with open(path, "r") as f:
            pid = int(f.read().strip())
    except Exception:
        return
    if _pid_is_llama(pid):
        print(f"[llama-server] killing orphaned server (PID {pid}) from an earlier run")
        _kill_tree(pid)
        time.sleep(2)
    try:
        os.remove(path)
    except OSError:
        pass


def _stop_server(port):
    """Stop the server, but only if this node started it."""
    info = _SERVERS.pop(int(port), None)
    if not info:
        return
    proc = info["proc"]
    if proc.poll() is None:
        if os.name == "nt":
            _kill_tree(proc.pid)  # also takes down child processes
        else:
            proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    try:
        os.remove(_pid_file(port))
    except OSError:
        pass
    try:
        info["log"].close()
    except Exception:
        pass


atexit.register(lambda: [_stop_server(p) for p in list(_SERVERS)])


def _print_device_info(log_path):
    """Print to the console which devices the server loaded the model and the mmproj on."""
    for line in _tail(log_path, 400).splitlines():
        low = line.lower()
        if "using device" in low or "clip using" in low:
            print("[llama-server]", line.strip())


def _check_server_exe(path):
    """Safety guard: a workflow shared by someone else must not be able to launch arbitrary programs."""
    name = os.path.basename(path).lower()
    if "llama" not in name and name not in ("server", "server.exe"):
        raise ValueError(
            "llama_server_path must point to a llama.cpp server executable (file name containing "
            f"'llama', e.g. llama-server.exe); got '{os.path.basename(path)}'."
        )


def _server_running(cmd, port):
    """True if the right server (same command line) was started by this node and is responding."""
    info = _SERVERS.get(int(port))
    return bool(info and info["proc"].poll() is None
                and info["key"] == tuple(cmd) and _is_healthy(port))


def _free_comfy_vram():
    """Unload the models ComfyUI left in VRAM (same as the 'Unload Models' button).

    llama-server is a separate process: ComfyUI does not know it will need VRAM and keeps the
    heavy models of the previous generation loaded. Without this, the new server would load
    its model on top of them.
    """
    try:
        import gc
        import comfy.model_management as mm
    except Exception:
        return

    def free_gb():
        try:
            return mm.get_free_memory() / (1024 ** 3)
        except Exception:
            return None

    before = free_gb()
    try:
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as e:
        print(f"[llama-server] could not unload ComfyUI models: {e}")
        return
    after = free_gb()
    if before is not None and after is not None:
        print(f"[llama-server] ComfyUI models unloaded; "
              f"free VRAM seen by ComfyUI: {before:.1f} -> {after:.1f} GB")
    else:
        print("[llama-server] ComfyUI models unloaded before starting the server.")


def _ensure_server(cmd, port, timeout=900):
    port = int(port)
    key = tuple(cmd)

    info = _SERVERS.get(port)
    if info and info["proc"].poll() is None and info["key"] == key and _is_healthy(port):
        return  # the right server is already running

    if info:
        _stop_server(port)  # settings changed or the process died

    _kill_stale(port)  # orphan from an earlier run of this node

    if _is_healthy(port):
        # Silently reusing it would hide the problem: the old server ignores this node's
        # model/GPU/mmproj settings and is never stopped, so its VRAM stays allocated.
        raise RuntimeError(
            f"A llama-server is already responding on port {port} and was NOT started by this node "
            "(probably left over from an earlier run, still holding VRAM). "
            f"Close it with:  {_KILL_HINT}   or change the port. "
            "To use an already running server on purpose, set server_mode = external_server."
        )

    log_path = os.path.join(_HERE, f"llama_server_{port}.log")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except FileNotFoundError:
        log.close()
        raise RuntimeError(
            f"Could not find the executable '{cmd[0]}'. Set the full path in llama_server_path "
            "(e.g. C:\\llama\\llama-server.exe or /opt/llama.cpp/llama-server)."
        )

    _SERVERS[port] = {"proc": proc, "key": key, "log": log, "log_path": log_path}
    try:
        with open(_pid_file(port), "w") as f:
            f.write(str(proc.pid))
    except OSError:
        pass
    if os.name == "nt":
        _assign_to_job(proc)
    print(f"[llama-server] starting on port {port}, PID {proc.pid} (log: {log_path})")

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            _check_interrupt()
        except BaseException:
            _stop_server(port)
            raise
        if proc.poll() is not None:
            code = proc.returncode
            _SERVERS.pop(port, None)
            log.close()
            raise RuntimeError(
                f"llama-server exited during startup (code {code}). "
                f"End of log ({log_path}):\n{_tail(log_path)}"
            )
        if _is_healthy(port):
            _print_device_info(log_path)
            return
        time.sleep(1)

    _stop_server(port)
    raise TimeoutError(f"llama-server was not ready after {timeout}s. See {log_path}")


def _build_cmd(exe, model_path, mmproj_path, port, ctx, ngl, extra,
               model_gpu, mmproj_gpu, prefix):
    cmd = [
        exe, "-m", model_path,
        "--host", "127.0.0.1", "--port", str(int(port)),
        "-c", str(int(ctx)), "-ngl", str(int(ngl)),
        "--jinja",
    ]
    if model_gpu.startswith("GPU"):
        cmd += ["--device", f"{prefix}{model_gpu.split()[1]}"]
    if mmproj_path:
        cmd += ["--mmproj", mmproj_path]
        if mmproj_gpu == "CPU":
            cmd += ["--mmproj-device", "none"]
        elif mmproj_gpu.startswith("GPU"):
            cmd += ["--mmproj-device", f"{prefix}{mmproj_gpu.split()[1]}"]
        # "same as model": no flag, llama.cpp follows the model's --device
    if extra.strip():
        cmd += shlex.split(extra, posix=(os.name != "nt"))
    return cmd


# --------------------------------------------------------------------------- #
# Request (streaming) and image conversion
# --------------------------------------------------------------------------- #

def _image_to_data_url(image, max_side=1280):
    from PIL import Image  # ships with ComfyUI

    arr = (image[0].cpu().numpy().clip(0, 1) * 255).astype("uint8")
    img = Image.fromarray(arr)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _generate_stream(port, payload, max_tokens):
    """Generate in streaming mode: updates the progress bar per token and allows cancelling."""
    body = dict(payload)
    body["stream"] = True
    req = urllib.request.Request(
        _base(port) + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=3600)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", "replace")[:1500]
        raise RuntimeError(f"llama-server respondeu HTTP {e.code}: {err}")

    content, reasoning, n = [], [], 0
    pbar = _progress_bar(max_tokens)
    stack = contextlib.ExitStack()
    try:
        from tqdm import tqdm
        bar = tqdm(total=int(max_tokens), desc="Generating tokens", unit="tok")
        stack.callback(bar.close)
    except Exception:
        bar = None
    t0 = time.time()
    with resp, stack:  # leaving the block (including on cancel) closes the connection and the bar
        for raw in resp:
            _check_interrupt()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("error"):
                raise RuntimeError(f"llama-server: {obj['error']}")
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c = delta.get("content")
            r = delta.get("reasoning_content")
            if c:
                content.append(c)
            if r:
                reasoning.append(r)
            if c or r:
                n += 1
                if pbar is not None:
                    pbar.update_absolute(min(n, int(max_tokens)), int(max_tokens))
                if bar is not None and n <= int(max_tokens):
                    bar.update(1)

    dt = max(time.time() - t0, 1e-6)
    print(f"[llama-server] {n} tokens in {dt:.1f}s ({n / dt:.1f} tok/s)")
    return {"content": "".join(content), "reasoning_content": "".join(reasoning)}


def _split_thinking(message):
    text = message.get("content") or ""
    thinking = message.get("reasoning_content") or ""
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        if not thinking:
            thinking = before.replace("<think>", "").strip()
        text = after
    return text.strip(), thinking.strip()


# --------------------------------------------------------------------------- #
# The node
# --------------------------------------------------------------------------- #

class LlamaServerGenerateTextV2:
    CATEGORY = "utils/LLM"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("generated_text", "thinking")

    @classmethod
    def INPUT_TYPES(cls):
        models, mmprojs = _scan()
        model_list = sorted(models) or [_NO_FILES]
        mmproj_list = ["none"] + sorted(mmprojs)
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "server_mode": (["auto_start", "external_server"],),
                "model": (model_list,),
                "model_gpu": (["all GPUs", "GPU 0", "GPU 1"],),
                "mmproj": (mmproj_list,),
                "mmproj_gpu": (["same as model", "GPU 0", "GPU 1", "CPU"],),
                "max_length": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                "sampling": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "thinking": ("BOOLEAN", {"default": False}),
                "keep_loaded": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "model_path_override": ("STRING", {"default": ""}),
                "mmproj_path_override": ("STRING", {"default": ""}),
                "llama_server_path": ("STRING", {"default": "llama-server"}),
                "device_prefix": ("STRING", {"default": "CUDA"}),
                "context_size": ("INT", {"default": 8192, "min": 512, "max": 262144}),
                "n_gpu_layers": ("INT", {"default": 99, "min": 0, "max": 999}),
                "port": ("INT", {"default": 8765, "min": 1024, "max": 65535}),
                "extra_args": ("STRING", {"default": ""}),
                "override_instructions": ("STRING", {"multiline": True, "default": ""}),
                "unload_comfy_models": ("BOOLEAN", {"default": True}),
            },
        }

    def generate(self, prompt, server_mode, model, model_gpu, mmproj, mmproj_gpu,
                 max_length, sampling, temperature, top_k, top_p, min_p,
                 repetition_penalty, presence_penalty, seed, thinking, keep_loaded,
                 system_prompt="", image=None, image_2=None, image_3=None,
                 image_4=None, image_5=None, model_path_override="",
                 mmproj_path_override="", llama_server_path="llama-server",
                 device_prefix="CUDA", context_size=8192, n_gpu_layers=99,
                 port=8765, extra_args="", override_instructions="", unload_comfy_models=True):

        llama_server_path = llama_server_path.strip().strip('"') or "llama-server"
        if server_mode == "auto_start":
            _check_server_exe(llama_server_path)
        device_prefix = device_prefix.strip() or "CUDA"
        model_override = model_path_override.strip().strip('"')
        mmproj_override = mmproj_path_override.strip().strip('"')

        # Messages
        override = override_instructions.strip()
        system_text = system_prompt
        if override:
            system_text = (system_prompt.strip() + "\n\n" if system_prompt.strip() else "") + _OVERRIDE_RULE
        messages = []
        if system_text.strip():
            messages.append({"role": "system", "content": system_text})
        images = [(name, t) for name, t in (
            ("image", image), ("image_2", image_2), ("image_3", image_3),
            ("image_4", image_4), ("image_5", image_5)) if t is not None]
        if images:
            content = []
            for i, (name, t) in enumerate(images, 1):
                if int(t.shape[0]) > 1:
                    print(f"[llama-server] WARNING: '{name}' has {int(t.shape[0])} images; "
                          "using only the first one.")
                if len(images) > 1:
                    content.append({"type": "text", "text": f"Image {i}:"})
                content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(t)}})
            if prompt.strip():
                content.append({"type": "text", "text": prompt})
        else:
            content = prompt
        if override:
            block = _OVERRIDE_HEADER + override
            if isinstance(content, list):
                content.append({"type": "text", "text": block})
            else:
                content = (content + "\n\n" if content.strip() else "") + block
        messages.append({"role": "user", "content": content})

        payload = {
            "messages": messages,
            "max_tokens": int(max_length),
            "temperature": float(temperature) if sampling else 0.0,
            "top_k": int(top_k) if sampling else 1,
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repeat_penalty": float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
            "seed": int(seed) % (2 ** 32),
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }

        cmd = None
        if server_mode == "auto_start":
            models, mmprojs = _scan()
            model_path = model_override or models.get(model)
            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"GGUF model not found ('{model_override or model}'). Put the file in "
                    "ComfyUI/models/text_encoders (or models/LLM), in a folder listed in LLAMA_GGUF_DIRS, "
                    "or use model_path_override."
                )
            mmproj_path = ""
            if images:
                mmproj_path = mmproj_override or mmprojs.get(mmproj, "")
                if not mmproj_path or not os.path.isfile(mmproj_path):
                    raise ValueError(
                        "An image is connected, but no valid mmproj was selected. "
                        "Pick a file in 'mmproj' (or use mmproj_path_override), "
                        "or disconnect the image."
                    )
            # The mmproj is only loaded when an image is connected, to avoid wasting VRAM.
            cmd = _build_cmd(llama_server_path, model_path, mmproj_path, port,
                             context_size, n_gpu_layers, extra_args,
                             model_gpu, mmproj_gpu, device_prefix)

        try:
            if server_mode == "auto_start":
                if unload_comfy_models and not _server_running(cmd, port):
                    _free_comfy_vram()
                _ensure_server(cmd, port)
            elif not _is_healthy(port):
                raise RuntimeError(
                    f"No llama-server is responding on port {port}. "
                    "Start it manually or use server_mode = auto_start."
                )
            message = _generate_stream(port, payload, max_length)
        finally:
            if server_mode == "auto_start" and not keep_loaded:
                _stop_server(port)  # frees the VRAM for the rest of the workflow

        text, think = _split_thinking(message)
        return (text, think)


NODE_CLASS_MAPPINGS = {"LlamaServerGenerateTextV2": LlamaServerGenerateTextV2}
NODE_DISPLAY_NAME_MAPPINGS = {"LlamaServerGenerateTextV2": "Generate Text (llama-server)"}
