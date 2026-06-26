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
