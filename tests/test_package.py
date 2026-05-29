"""Tests for image packaging API endpoint."""

import io
import zipfile
from datetime import UTC, datetime, timedelta

import pytest


@pytest.fixture(scope="module")
def client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient

    from inference.api import app

    return TestClient(app)


@pytest.fixture
def setup_test_images(tmp_path, monkeypatch):
    """Create test images with different class names."""
    from pydantic import TypeAdapter

    from inference.image_index import ImageEntry

    # Create temporary upload directory with subdirectories
    test_upload_dir = tmp_path / "upload"
    test_upload_dir.mkdir(parents=True, exist_ok=True)

    screen_dir = test_upload_dir / "screen_photo"
    screen_dir.mkdir(parents=True, exist_ok=True)
    normal_dir = test_upload_dir / "normal_photo"
    normal_dir.mkdir(parents=True, exist_ok=True)

    # Create test images in different directories
    image_files = []
    for i in range(3):
        if i < 2:
            # First 2 are screen_photo
            img_path = screen_dir / f"test_hash_{i}.jpg"
        else:
            # Last 1 is normal_photo
            img_path = normal_dir / f"test_hash_{i}.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        image_files.append(img_path)

    # Create index with different timestamps and class names
    now = datetime.now(UTC)
    ta = TypeAdapter(dict[str, ImageEntry])

    index = {}
    for i in range(3):
        class_name = "screen_photo" if i < 2 else "normal_photo"
        entry = ImageEntry(
            file_name=f"test_hash_{i}.jpg",
            file_hash=f"test_hash_{i}",
            class_name=class_name,
            created_at=now - timedelta(minutes=5 - i),
        )
        index[f"test_hash_{i}"] = entry

    index_file = test_upload_dir / "index.json"
    index_file.write_bytes(ta.dump_json(index))

    # Monkey patch the module-level constants
    monkeypatch.setattr("inference.image_index.UPLOAD_DIR", test_upload_dir)
    monkeypatch.setattr("inference.image_index.INDEX_FILE", index_file)
    monkeypatch.setattr("inference.api.router.INDEX_FILE", index_file)

    return {
        "upload_dir": test_upload_dir,
        "index_file": index_file,
        "now": now,
        "image_files": image_files,
    }


def test_package_images_success(client, setup_test_images):
    """Test successful image packaging."""
    test_data = setup_test_images
    now = test_data["now"]

    # Request images after now - 3.5 min (should get 1: test_hash_2 at 3 min ago)
    timestamp = (now - timedelta(minutes=3, seconds=30)).isoformat()

    response = client.post(
        "/api/package",
        json={"after_timestamp": timestamp},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "attachment" in response.headers.get("content-disposition", "")

    # Verify zip content
    zip_content = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_content, "r") as zf:
        file_list = zf.namelist()
        assert len(file_list) == 1
        assert file_list[0] == "non_screen_photo/test_hash_2.jpg"


def test_package_images_with_folders(client, setup_test_images):
    """Test that images are sorted into screen_photo and non_screen_photo folders."""
    test_data = setup_test_images
    now = test_data["now"]

    # Request all images
    timestamp = (now - timedelta(minutes=6)).isoformat()

    response = client.post(
        "/api/package",
        json={"after_timestamp": timestamp},
    )

    assert response.status_code == 200

    zip_content = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_content, "r") as zf:
        file_list = zf.namelist()

        # Check screen_photo folder
        screen_files = [f for f in file_list if f.startswith("screen_photo/")]
        assert len(screen_files) == 2
        assert "screen_photo/test_hash_0.jpg" in screen_files
        assert "screen_photo/test_hash_1.jpg" in screen_files

        # Check non_screen_photo folder
        non_screen_files = [f for f in file_list if f.startswith("non_screen_photo/")]
        assert len(non_screen_files) == 1
        assert "non_screen_photo/test_hash_2.jpg" in non_screen_files


def test_package_images_not_found(client, setup_test_images):
    """Test when no images match the timestamp filter."""
    test_data = setup_test_images
    now = test_data["now"]

    # Request images created after now (should get none)
    timestamp = (now + timedelta(minutes=1)).isoformat()

    response = client.post(
        "/api/package",
        json={"after_timestamp": timestamp},
    )

    assert response.status_code == 404
    assert "No images found" in response.json()["detail"]


def test_package_images_invalid_timestamp(client):
    """Test with invalid timestamp format."""
    response = client.post(
        "/api/package",
        json={"after_timestamp": "invalid-timestamp"},
    )

    assert response.status_code == 422  # Validation error


def test_package_images_response_structure(client, setup_test_images):
    """Test that the response is a valid zip file."""
    test_data = setup_test_images
    now = test_data["now"]

    timestamp = (now - timedelta(minutes=6)).isoformat()

    response = client.post(
        "/api/package",
        json={"after_timestamp": timestamp},
    )

    assert response.status_code == 200

    # Verify it's a valid zip file
    zip_content = io.BytesIO(response.content)
    assert zipfile.is_zipfile(zip_content)

    # Verify filename format
    content_disp = response.headers.get("content-disposition", "")
    assert "images_" in content_disp
    assert ".zip" in content_disp


def test_package_images_preserves_filenames(client, setup_test_images):
    """Test that original filenames are preserved in the zip."""
    test_data = setup_test_images
    now = test_data["now"]

    timestamp = (now - timedelta(minutes=6)).isoformat()

    response = client.post(
        "/api/package",
        json={"after_timestamp": timestamp},
    )

    assert response.status_code == 200

    zip_content = io.BytesIO(response.content)
    with zipfile.ZipFile(zip_content, "r") as zf:
        file_list = zf.namelist()
        # All test files should be present with folder prefixes
        for i in range(3):
            if i < 2:
                assert f"screen_photo/test_hash_{i}.jpg" in file_list
            else:
                assert f"non_screen_photo/test_hash_{i}.jpg" in file_list
