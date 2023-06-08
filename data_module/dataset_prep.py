# Copyright 2022 Mayo Clinic. All Rights Reserved.
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
# ==============================================================================


import os
import sys
import logging
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path


def resolve_gcs_uri_to_local_uri(cfg):
    """
    Use an existing gcsfuse mountpoint, or use gcsfuse to mount a bucket
    locally if not mountpoint exists, to get a local filepath for a file stored
    in a GCS bucket.
    Args:
        cfg:
    Returns:
        str: A string indicating a local filepath where gcs_file_uri can be
        found.
    """
    if not cfg.gcs_file_uri.startswith("gs:/"):
        raise Exception('Please use a "gs:/" GCS path.')

    # Pathlib isn't wholly apposite to use for URLs (specifically, it changes
    # 'gs://...' to 'gs:/...'), but it's useful in this case:
    gcs_path = Path(cfg.gcs_file_uri)
    bucket_name = gcs_path.parts[1]
    file_remaining_path = Path(*gcs_path.parts[2:])
    logging.info("Checking for existing mountpoint for %s...", bucket_name)
    existing_mountpoint_check = subprocess.run(
        f'findmnt --noheadings --output TARGET --source "{bucket_name}"',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    existing_mountpoint_check_stdout = existing_mountpoint_check.stdout.decode().strip()

    if existing_mountpoint_check.returncode == 127:
        raise Exception(
            f'Error when running findmnt: "{existing_mountpoint_check_stdout}"'
        )

    if (
        existing_mountpoint_check.returncode == 0
        and existing_mountpoint_check_stdout != ""
    ):
        # An existing mountpoint for the bucket exists; we will reuse it here,
        # if it contains the file we're looking for (it's possible that the
        # bucket is mounted, but to a different subdirectory within the bucket
        # than we need).
        for mountpoint in existing_mountpoint_check_stdout.split("\n"):
            logging.info(
                'Checking for file "%s" at "%s"...', file_remaining_path, mountpoint
            )
            if Path(mountpoint, file_remaining_path).exists():
                found_file_path = os.path.join(mountpoint, file_remaining_path)
                logging.info('Found file at "%s"', found_file_path)
                return found_file_path

    # No existing mountpoint exists, so we will mount the bucket with gcsfuse:
    import tempfile

    tmp_directory = tempfile.mkdtemp()
    logging.info('Creating new mountpoint at "%s"...', tmp_directory)
    mount_command = subprocess.run(
        # Unexpectedly, using `f"gcsfuse --implicit-dirs --only-dir {specific_bucket_directory} {bucket_name} {tmp_directory}"`
        # resulted in a mountpoint that was *unusably slow* when trying to,
        # e.g., using `.get_thumbnail()`. Thus, here, we are not using
        # `--only-dir.`
        f"gcsfuse --implicit-dirs {bucket_name} {tmp_directory}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    mount_command_stdout = mount_command.stdout.decode().strip()
    if "File system has been successfully mounted" not in mount_command_stdout:
        raise Exception(
            f'Error when running gcsfuse: "{mount_command_stdout}". Exit code was {existing_mountpoint_check.returncode}.'
        )

    return os.path.join(tmp_directory, file_remaining_path)

def rsna_ich_download():
    """_summary_
    """
    cmd = "kaggle competitions download -c rsna-intracranial-hemorrhage-detection"
    os.system(cmd)

def gcp_rsna_ich_prep(cfg):
    """_summary_

    Args:
        gcs_file_uri (_type_): _description_
        local_dataset_path (_type_): _description_
    """
    base_path = resolve_gcs_uri_to_local_uri(gcs_file_uri=cfg.gcs_file_uri)
    cmd = f"cp -r {cfg.local_dataset_path} {base_path}"
    os.system(cmd)

@hydra.main(version_base=None, config_path="configs", config_name="dataset")
def main(cfg: DictConfig) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_

    Returns:
        _type_: _description_
    """
    for key, value in cfg.items():
        if value == "None":
            cfg[key] = eval(value)
    gcp_rsna_ich_prep(cfg)


if __name__=="__main__":
    print("Start to prepare RSNA ICH Dataset")
    main()
