# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from pathlib import Path

# List of segmentations as Paths
files = list(
    map(
        Path,
        [
            f"coordinates/{i}"
            for i in [
                "cell1/34D-grid2-s3-acta1_001_16.txt",
                "cell2/34D-grid3-ActA1_007_16.txt",
                "cell2/34D-grid3-ActA1_013_16.txt",
                "cell3/34D-grid2-s2_002_16.txt",
                "cell3/34D-grid2-s5_005_16.txt",
                "cell3/34D-grid3-ActA1_020_16.txt",
                "cell3/34D-grid3-s6_005_16.txt",
                "cell4/34D-grid2-s3_028_16.txt",
                "cell5/34D-grid3-ActA1_001_16.txt",
                "cell5/34D-grid3-ActA1_002_16.txt",
                "cell5/34D-grid3-ActA1_003_16.txt",
                "cell5/34D-grid3-ActA1_004_16.txt",
            ]
        ],
    )
)

# Map of image key to microns per pixel
images = {
    "34D-grid2-s3-acta1_001_16": 0.012723,
    "34D-grid3-ActA1_007_16": 0.015904,
    "34D-grid3-ActA1_013_16": 0.015904,
    "34D-grid2-s2_002_16": 0.015904,
    "34D-grid2-s5_005_16": 0.015904,
    "34D-grid3-ActA1_020_16": 0.015904,
    "34D-grid3-s6_005_16": 0.015904,
    "34D-grid2-s3_028_16": 0.015904,
    "34D-grid3-ActA1_001_16": 0.015904,
    "34D-grid3-ActA1_002_16": 0.015904,
    "34D-grid3-ActA1_003_16": 0.015904,
    "34D-grid3-ActA1_004_16": 0.015904,
}
