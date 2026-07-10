"""SandboxRunner — run an arbitrary command inside the hardened `exec` container (D11).

The container is the isolation boundary; the D11 hardening flags live HERE at the call site
(not baked into the image) so the contract is auditable. The runner targets either local Docker
or a remote sandbox host over SSH (scp samples in, run, scp results out, clean up). Benchmarks
supply the inner command (e.g. `timeout Ns python -m evalplus.evaluate ...`).

Extracted from scripts/score_humaneval.py during the Phase-1 refactor — the docker argv and the
SSH step sequence are unchanged.
"""
from __future__ import annotations

import shlex
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path


class SandboxRunner:
    def __init__(self, image: str, *, cpus: str = "2", memory: str = "4g",
                 pids_limit: int = 256, read_only: bool = True, auto_confirm: bool = False,
                 ssh_host: str | None = None,
                 ssh_workdir: str = "/tmp", local: bool = False, dry_run: bool = False):
        self.image = image
        self.cpus = str(cpus)
        self.memory = memory
        self.pids_limit = int(pids_limit)
        self.read_only = read_only      # False = writable rootfs (looser profile for evaluators that write)
        self.auto_confirm = auto_confirm  # feed 'y' to interactive prompts (ssh path)
        self.ssh_host = ssh_host
        self.ssh_workdir = ssh_workdir
        self.local = local
        self.dry_run = dry_run

    def build_docker_argv(self, work_mount: str, inner_cmd: list[str], env: dict | None = None) -> list[str]:
        """The hardened `docker run` (D11). `work_mount` is the host dir bind-mounted to /work;
        `env` entries become `-e VAR=val` flags (e.g. an offline dataset override)."""
        argv = ["docker", "run", "--rm"]
        if self.auto_confirm:
            argv.append("-i")                 # keep stdin open so a piped 'y' answers prompts
        argv += [
            "--network=none",                 # untrusted code gets no network
            "--cap-drop=ALL",                 # drop every Linux capability
            "--security-opt=no-new-privileges",
            "--tmpfs", "/tmp:rw,size=512m",   # writable scratch
            "--pids-limit", str(self.pids_limit),
            "--cpus", self.cpus,
            "--memory", self.memory,
        ]
        if self.read_only:
            argv.append("--read-only")        # strict 'exec' profile; looser benchmarks turn it off
        for k, v in (env or {}).items():
            argv += ["-e", f"{k}={v}"]
        argv += ["-v", f"{work_mount}:/work", self.image, *inner_cmd]
        return argv

    def _run_steps(self, steps: list[list[str]], *, stdin_data: bytes | None = None) -> None:
        for argv in steps:
            print("  $ " + " ".join(shlex.quote(a) for a in argv))
            if not self.dry_run:
                subprocess.run(argv, check=True, input=stdin_data)

    def run_local(self, work_dir: Path, inner_cmd: list[str], env: dict | None = None) -> None:
        argv = self.build_docker_argv(str(work_dir.resolve()), inner_cmd, env)
        print("[sandbox] local Docker:")
        stdin_data = b"y\n" * 100 if self.auto_confirm else None
        self._run_steps([argv], stdin_data=stdin_data)

    def run_ssh(self, work_dir: Path, inner_cmd: list[str], *,
                in_files: list[str], out_files: list[str], env: dict | None = None) -> None:
        host = self.ssh_host
        remote = f"{self.ssh_workdir.rstrip('/')}/ocb-score-{work_dir.name}"
        docker_cmd = " ".join(shlex.quote(a) for a in self.build_docker_argv(remote, inner_cmd, env))
        if self.auto_confirm:
            docker_cmd = "yes | " + docker_cmd   # auto-answer the evaluator's interactive [Y/N] prompts
        print(f"[sandbox] over SSH: {host}  (remote work dir: {remote})")
        steps = [["ssh", host, f"mkdir -p {shlex.quote(remote)} && chmod 777 {shlex.quote(remote)}"]]
        steps += [["scp", str(work_dir / f), f"{host}:{remote}/{f}"] for f in in_files]
        steps.append(["ssh", host, docker_cmd])
        steps += [["scp", f"{host}:{remote}/{f}", str(work_dir / f)] for f in out_files]
        steps.append(["ssh", host, f"rm -rf {shlex.quote(remote)}"])
        self._run_steps(steps)

    # ---- directory-tree exec with captured output (agentic/multi-turn benchmarks, D14) ----
    # Single-shot scoring ships flat sample files and reads result JSON back (run_local/run_ssh).
    # AiderPolyglot instead needs, per model turn: ship a whole *exercise directory* (edited
    # solution files + hidden tests + build config), run the language's test command inside it,
    # and read back only the exit code (pass/fail) and combined output (to feed as the next-turn
    # prompt). No files come back. This method provides exactly that; infra failures (ssh/scp/tar)
    # raise, while a nonzero *test* exit is returned as data so the caller can distinguish a failed
    # test from an infra error (D12).
    def run_dir_capture(self, work_dir: Path, inner_cmd: list[str], *,
                        env: dict | None = None, timeout: int | None = None) -> tuple[int, str]:
        if self.local:
            argv = self.build_docker_argv(str(work_dir.resolve()), inner_cmd, env)
            print("[sandbox] local Docker (capture): $ " + " ".join(shlex.quote(a) for a in argv))
            if self.dry_run:
                return (0, "[dry-run] not executed")
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            return (p.returncode, (p.stdout or "") + (p.stderr or ""))

        # SSH: tar the tree locally (stdlib tarfile — no external `tar` needed on Windows), scp it,
        # extract remotely, run docker (capturing that step's exit as the *test* result), clean up.
        host = self.ssh_host
        # `work_dir.name` alone is NOT unique across concurrent calls: AiderPolyglot ships one
        # working dir per (task, attempt), and its leaf name is the bare exercise name (e.g.
        # "two-fer"), which the polyglot dataset reuses across all 6 language tracks. Keying the
        # remote scratch path on that name alone let two concurrent same-named exercises in
        # different languages collide on one remote dir and race each other's rm -rf/extract. A
        # per-call random suffix guarantees a distinct remote path regardless of what naming
        # convention the caller's work_dir happens to use.
        remote = f"{self.ssh_workdir.rstrip('/')}/ocb-exec-{work_dir.name}-{uuid.uuid4().hex[:12]}"
        docker_cmd = " ".join(shlex.quote(a) for a in self.build_docker_argv(remote, inner_cmd, env))
        print(f"[sandbox] over SSH (capture): {host}  (remote: {remote})")
        if self.dry_run:
            print("  $ tar+scp <work_dir> && ssh extract && " + docker_cmd)
            return (0, "[dry-run] not executed")
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tf:
            tgz = Path(tf.name)
        try:
            with tarfile.open(tgz, "w:gz") as tar:
                tar.add(work_dir, arcname=".")   # contents at the archive root
            rq = shlex.quote(remote)
            # infra steps: fail loudly (check=True) so the caller marks the attempt infra_error
            subprocess.run(["ssh", host, f"rm -rf {rq} && mkdir -p {rq} && chmod 777 {rq}"], check=True)
            subprocess.run(["scp", str(tgz), f"{host}:{remote}/_ocb.tgz"], check=True)
            subprocess.run(["ssh", host, f"tar xzf {rq}/_ocb.tgz -C {rq} && rm -f {rq}/_ocb.tgz"],
                           check=True)
            # test step: capture exit + output WITHOUT check — a nonzero exit here is a failing test
            p = subprocess.run(["ssh", host, docker_cmd], capture_output=True, text=True,
                               timeout=timeout)
            subprocess.run(["ssh", host, f"rm -rf {rq}"], check=False)   # best-effort cleanup
            return (p.returncode, (p.stdout or "") + (p.stderr or ""))
        finally:
            tgz.unlink(missing_ok=True)
