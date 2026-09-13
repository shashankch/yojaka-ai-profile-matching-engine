import os
import tempfile
from unittest.mock import patch, MagicMock
from agentic_profile_matching.celery_app import celery_app
from agentic_profile_matching.tasks import async_ingest_directory, async_deep_screen_candidate


def test_celery_task_registration():
    registered = list(celery_app.tasks.keys())
    assert "tasks.async_ingest_directory" in registered
    assert "tasks.async_deep_screen_candidate" in registered


def test_async_ingest_directory_eager():
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_file = os.path.join(tmpdir, "candidate.txt")
        with open(sample_file, "w") as f:
            f.write("Alice Smith\nPython Developer with 5 years experience.")

        with patch("agentic_profile_matching.tasks.IngestionService") as mock_service_cls:
            mock_service = MagicMock()
            mock_service.ingest_directory.return_value = {"success": True, "total_files": 1, "successful": 1}
            mock_service_cls.return_value = mock_service

            result = async_ingest_directory.delay(tmpdir)
            val = result.get()
            assert val["success"] is True
            assert val["total_files"] == 1
            mock_service.ingest_directory.assert_called_once_with(tmpdir)


def test_async_deep_screen_candidate_eager():
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write("Bob Jones\nReact and TypeScript frontend engineer.")
        tmp_path = tmp.name

    try:
        reqs = {"title": "Frontend Engineer", "must_have_skills": ["React"]}
        result = async_deep_screen_candidate.delay(tmp_path, "Bob Jones", reqs)
        val = result.get()
        assert val["candidate_id"] == tmp_path
        assert val["candidate_name"] == "Bob Jones"
        assert "screening_status" in val
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
