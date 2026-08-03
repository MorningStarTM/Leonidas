"""Fetch mocap training data (.npz files) from S3.

Required in .env:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION      (optional, defaults to "us-east-1")
    S3_BUCKET               name of the bucket holding training data
    S3_PREFIX               (optional) default key prefix, e.g. "training_data/"
"""
import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

from src.utils.logger import logger

load_dotenv()


class DataFetcher:
    """Lists, downloads, and loads mocap .npz data from an S3 bucket."""

    def __init__(self, bucket: Optional[str] = None, region: Optional[str] = None,
                 prefix: str = ""):
        self.bucket = bucket or os.environ.get("S3_BUCKET")
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.prefix = prefix or os.environ.get("S3_PREFIX", "")

        if not self.bucket:
            raise ValueError(
                "No S3 bucket configured. Pass bucket=... or set S3_BUCKET in .env"
            )

        self.client = boto3.client("s3", region_name=self.region)

    def check_connection(self) -> bool:
        """Verify credentials and bucket access."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"Connected to s3://{self.bucket} ({self.region})")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"S3 connection failed: {e}")
            return False

    def list_files(self, prefix: Optional[str] = None,
                    suffix: Optional[str] = ".npz") -> List[Dict]:
        """List objects under `prefix`, optionally filtered by key suffix."""
        prefix = self.prefix if prefix is None else prefix
        paginator = self.client.get_paginator("list_objects_v2")
        files = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if suffix is None or obj["Key"].endswith(suffix):
                    files.append({"key": obj["Key"], "size": obj["Size"]})
        logger.info(f"Found {len(files)} files under s3://{self.bucket}/{prefix}")
        return files

    def download_file(self, key: str, local_path: str, overwrite: bool = False) -> str:
        """Download a single object to `local_path`."""
        local_path = Path(local_path)
        if local_path.exists() and not overwrite:
            logger.debug(f"Skip existing {local_path}")
            return str(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local_path))
        logger.info(f"Downloaded s3://{self.bucket}/{key} -> {local_path}")
        return str(local_path)

    def download_all(self, local_dir: str, prefix: Optional[str] = None,
                      suffix: Optional[str] = ".npz", overwrite: bool = False) -> List[str]:
        """Download every matching object under `prefix`, preserving the key
        layout relative to `prefix` inside `local_dir`."""
        prefix = self.prefix if prefix is None else prefix
        local_dir = Path(local_dir)
        paths = []
        for f in self.list_files(prefix=prefix, suffix=suffix):
            key = f["key"]
            rel = key[len(prefix):] if prefix and key.startswith(prefix) else key
            # A leading "/" (e.g. prefix given without a trailing slash) would
            # make Path join treat `rel` as drive-absolute and silently drop
            # `local_dir` entirely, so strip it. An empty `rel` (key == prefix
            # exactly) falls back to the key's basename.
            rel = rel.lstrip("/\\") or os.path.basename(key)
            paths.append(self.download_file(key, local_dir / rel, overwrite=overwrite))
        return paths

    def load_npz(self, key: str) -> Dict[str, np.ndarray]:
        """Load a .npz object directly into memory, without writing to disk."""
        buf = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, buf)
        buf.seek(0)
        with np.load(buf, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}


if __name__ == "__main__":
    fetcher = DataFetcher()
    if fetcher.check_connection():
        for f in fetcher.list_files()[:20]:
            logger.info(f"  {f['key']}  ({f['size'] / 1024:.0f} KB)")
