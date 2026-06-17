# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Optional

from pydantic import Field

from cloudai.core import DockerImage, File, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class SlurmContainerCmdArgs(CmdArgs):
    """Command line arguments for a generic Slurm container test."""

    docker_image_url: str
    cmd: str
    exit_status_file: str | None = Field(
        default=None,
        description=(
            "Name of a file the workload writes its exit status into, inside the "
            "test output folder. 0 means it passed; any non-zero number means it "
            "failed. If you set this but the file is missing, the test counts as "
            "failed (the job died before writing it). If you leave it empty, the "
            "container is never checked and always counts as passed."
        ),
    )


class SlurmContainerTestDefinition(TestDefinition):
    """Test definition for a generic Slurm container test."""

    cmd_args: SlurmContainerCmdArgs
    extra_srun_args: list[str] = Field(default_factory=list)
    scripts: list[File] = Field(default_factory=list)
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, *self.git_repos, *self.scripts]

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        name = self.cmd_args.exit_status_file
        if not name:                                   # not a grading test -> unchanged
            return JobStatusResult(is_successful=True)

        exit_status_file = tr.output_path / name
        if not exit_status_file.is_file():                       # opted in but no verdict -> died
            return JobStatusResult(
                is_successful=False,
                error_message=f"{name} not found in {tr.output_path} (job died before writing a verdict?)",
            )

        return self._parse_exit_status(exit_status_file)

    def _parse_exit_status(self, exit_status_file: Path) -> JobStatusResult:
        content = exit_status_file.read_text(errors="replace").strip()
        try:
            code = int(content)
        except ValueError:
            return JobStatusResult(
                is_successful=False,
                error_message=f"non-integer status in {exit_status_file.name!r}: {content!r}",
            )

        if code == 0:
            return JobStatusResult(is_successful=True)

        return JobStatusResult(
            is_successful=False,
            error_message=f"{exit_status_file.name} reports non-zero exit status {code}",
        )
