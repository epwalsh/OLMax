import uuid
from typing import Generator

import pytest

import olmax


@pytest.fixture
def bucket_name() -> str:
    return "ai2-olmo-testing"


@pytest.fixture
def gcs_bucket_name() -> str:
    return "olmo-core-testing"


@pytest.fixture
def unique_name() -> str:
    return uuid.uuid4().hex


@pytest.fixture
def s3_checkpoint_dir(bucket_name, unique_name) -> Generator[str, None, None]:
    from botocore.exceptions import NoCredentialsError

    folder = f"s3://{bucket_name}/checkpoints/{unique_name}"
    yield folder

    try:
        olmax.fs.clear_directory(folder, force=True)
    except NoCredentialsError:
        pass


@pytest.fixture
def gcs_checkpoint_dir(gcs_bucket_name, unique_name) -> Generator[str, None, None]:
    from google.auth.exceptions import DefaultCredentialsError

    folder = f"gs://{gcs_bucket_name}/checkpoints/{unique_name}"
    yield folder

    try:
        olmax.fs.clear_directory(folder, force=True)
    except DefaultCredentialsError:
        pass
