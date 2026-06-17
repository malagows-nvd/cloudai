# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from cloudai.core import TestRun
from cloudai.workloads.slurm_container import SlurmContainerCmdArgs, SlurmContainerTestDefinition

EXIT_STATUS_FILE = "exit_status.txt"


def make_tr(tmp_path, exit_status_file: str | None) -> TestRun:
    tdef = SlurmContainerTestDefinition(
        name="sc",
        description="desc",
        test_template_name="SlurmContainer",
        cmd_args=SlurmContainerCmdArgs(
            docker_image_url="docker://url",
            cmd="bash /scripts/run.sh",
            exit_status_file=exit_status_file,
        ),
    )
    return TestRun(name="sc", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "output")


def write_exit_status(tr: TestRun, content: str) -> None:
    tr.output_path.mkdir(parents=True, exist_ok=True)
    (tr.output_path / EXIT_STATUS_FILE).write_text(content, encoding="utf-8")


class TestSlurmContainerStatusCheck:
    def test_no_exit_status_file_configured_is_always_successful(self, tmp_path) -> None:
        tr = make_tr(tmp_path, exit_status_file=None)  # generic container, no grading

        result = tr.test.was_run_successful(tr)

        assert result.is_successful
        assert result.error_message == ""

    def test_configured_but_exit_status_file_missing_is_reported(self, tmp_path) -> None:
        tr = make_tr(tmp_path, exit_status_file=EXIT_STATUS_FILE)
        tr.output_path.mkdir(parents=True, exist_ok=True)  # dir exists, verdict file does not

        result = tr.test.was_run_successful(tr)

        assert not result.is_successful
        assert EXIT_STATUS_FILE in result.error_message
        assert "not found" in result.error_message

    @pytest.mark.parametrize(
        ("content", "expected_success"),
        [
            pytest.param("0\n", True, id="zero-pass"),
            pytest.param("0", True, id="zero-no-newline"),
            pytest.param("1", False, id="one-fail"),
            pytest.param("137", False, id="sigkill-status"),
            pytest.param("-1", False, id="negative"),
            pytest.param("", False, id="empty"),
            pytest.param("PASS", False, id="non-integer-word"),
            pytest.param("0 ok", False, id="trailing-text-rejected"),
        ],
    )
    def test_exit_status_is_honored(self, tmp_path, content: str, expected_success: bool) -> None:
        tr = make_tr(tmp_path, exit_status_file=EXIT_STATUS_FILE)
        write_exit_status(tr, content)

        result = tr.test.was_run_successful(tr)

        assert result.is_successful is expected_success
        if not expected_success:
            assert result.error_message

    def test_failure_status_is_surfaced(self, tmp_path) -> None:
        tr = make_tr(tmp_path, exit_status_file=EXIT_STATUS_FILE)
        write_exit_status(tr, "137")

        result = tr.test.was_run_successful(tr)

        assert not result.is_successful
        assert "137" in result.error_message
