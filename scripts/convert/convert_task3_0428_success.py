#!/usr/bin/env python3
"""Convert official Task3 HDF5 files to one LeRobot dataset.

This wrapper is intentionally stricter than the tutorial's editable example:
it accepts multiple HDF5 files, skips demos with success=False, and defaults to
60 fps with no initial-frame drop so the first action stays aligned with the
recorded initial_state.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


BASE_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (16,),
        "names": [
            "J1_1.pos", "J1_2.pos", "J1_3.pos", "J1_4.pos",
            "J1_5.pos", "J1_6.pos", "J1_7.pos", "J1_8.pos",
            "J2_1.pos", "J2_2.pos", "J2_3.pos", "J2_4.pos",
            "J2_5.pos", "J2_6.pos", "J2_7.pos", "J2_8.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (16,),
        "names": [
            "J1_1.pos", "J1_2.pos", "J1_3.pos", "J1_4.pos",
            "J1_5.pos", "J1_6.pos", "J1_7.pos", "J1_8.pos",
            "J2_1.pos", "J2_2.pos", "J2_3.pos", "J2_4.pos",
            "J2_5.pos", "J2_6.pos", "J2_7.pos", "J2_8.pos",
        ],
    },
    "observation.images.top": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 60.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 60.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 60.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", default="local/task3_0428_success_fps60_noskip")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--skip-initial-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--task",
        default="Put items with Chinese labels into the red basket, and others into the blue basket.",
    )
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Feed frames directly to LeRobot's streaming video encoder instead of writing temporary PNGs.",
    )
    return parser.parse_args()


def features_with_fps(fps: int) -> dict:
    features = {}
    for key, value in BASE_FEATURES.items():
        item = dict(value)
        if item.get("dtype") == "video":
            item["video_info"] = dict(item["video_info"])
            item["video_info"]["video.fps"] = float(fps)
        features[key] = item
    return features


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    def key(name: str) -> tuple[int, str]:
        if name.startswith("demo_") and name.split("_")[-1].isdigit():
            return (int(name.split("_")[-1]), name)
        return (10**9, name)

    return sorted(data_group.keys(), key=key)


def process_demo(dataset: LeRobotDataset, task: str, demo: h5py.Group, demo_name: str, skip_initial: int) -> int:
    try:
        # HDF5 frame-by-frame image slicing is extremely slow for these files.
        # Read one demo at a time into RAM, then iterate over numpy arrays.
        actions = np.asarray(demo["actions"], dtype=np.float32)
        left_joint_pos = np.asarray(demo["obs/left_joint_pos_rel"], dtype=np.float32)
        right_joint_pos = np.asarray(demo["obs/right_joint_pos_rel"], dtype=np.float32)
        top_images = np.asarray(demo["obs/top"])
        left_images = np.asarray(demo["obs/left_wrist"])
        right_images = np.asarray(demo["obs/right_wrist"])
    except KeyError as exc:
        print(f"[skip] {demo_name}: missing {exc}", flush=True)
        return 0

    total = int(actions.shape[0])
    if total <= max(10, skip_initial):
        print(f"[skip] {demo_name}: too short ({total} frames)", flush=True)
        return 0

    for frame_index in tqdm(range(skip_initial, total), desc=f"{demo_name}", leave=False):
        joint_pos = np.concatenate([left_joint_pos[frame_index], right_joint_pos[frame_index]], axis=0)
        dataset.add_frame(
            {
                "action": actions[frame_index],
                "observation.state": joint_pos,
                "observation.images.top": top_images[frame_index],
                "observation.images.left_wrist": left_images[frame_index],
                "observation.images.right_wrist": right_images[frame_index],
                "task": task,
            }
        )
    return total - skip_initial


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} already exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        robot_type="xtrainer_follower",
        features=features_with_fps(args.fps),
        video_backend=args.video_backend,
        vcodec=args.vcodec,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        encoder_threads=args.encoder_threads,
        streaming_encoding=args.streaming_encoding,
    )

    saved = 0
    skipped_failed = 0
    skipped_invalid = 0
    total_frames = 0

    for input_file in args.input_files:
        input_path = Path(input_file)
        print(f"[file] {input_path}", flush=True)
        with h5py.File(input_path, "r") as h5:
            data = h5["data"]
            for demo_name in tqdm(sorted_demo_names(data), desc=input_path.name):
                demo = data[demo_name]
                if "success" in demo.attrs and not bool(demo.attrs["success"]):
                    skipped_failed += 1
                    print(f"[skip] {input_path.name}:{demo_name}: success=False", flush=True)
                    continue
                frames = process_demo(dataset, args.task, demo, demo_name, args.skip_initial_frames)
                if frames <= 0:
                    skipped_invalid += 1
                    continue
                dataset.save_episode()
                saved += 1
                total_frames += frames
                print(f"[save] episode={saved} source={input_path.name}:{demo_name} frames={frames}", flush=True)

    print(
        f"[done] saved_episodes={saved} frames={total_frames} "
        f"skipped_failed={skipped_failed} skipped_invalid={skipped_invalid} root={root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
