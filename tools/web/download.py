from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

from workspace.manager import WorkspaceManager


DEFAULT_TIMEOUT = 60
MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024


def download_file(
    workspace: WorkspaceManager,
    url: str,
    path: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """
    Download a remote resource into the agent workspace.
    """

    url = url.strip()
    path = path.strip()

    if not url:
        raise ValueError(
            "URL cannot be empty."
        )

    if not (
        url.startswith("http://")
        or url.startswith("https://")
    ):
        raise ValueError(
            "Only HTTP and HTTPS URLs are supported."
        )

    if not path:
        raise ValueError(
            "Destination path cannot be empty."
        )

    if timeout < 1:
        raise ValueError(
            "timeout must be >= 1 second."
        )

    if timeout > 300:
        raise ValueError(
            "timeout must be <= 300 seconds."
        )

    destination = workspace.write_path(path)

    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(
            f"Destination is a directory: {path}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/154.0 Safari/537.36"
            )
        },
    )

    total_bytes = 0

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:

            content_type = response.headers.get(
                "Content-Type",
                "",
            )

            final_url = response.geturl()

            with destination.open(
                "wb"
            ) as output:

                while True:
                    chunk = response.read(
                        64 * 1024
                    )

                    if not chunk:
                        break

                    total_bytes += len(chunk)

                    if total_bytes > MAX_DOWNLOAD_SIZE:
                        raise RuntimeError(
                            "Download exceeds the "
                            f"{MAX_DOWNLOAD_SIZE} byte limit."
                        )

                    output.write(chunk)

    except Exception as exc:
        if destination.exists():
            try:
                destination.unlink()
            except OSError:
                pass

        if isinstance(exc, RuntimeError):
            raise

        raise RuntimeError(
            f"Failed to download resource: {exc}"
        ) from exc

    return {
        "url": url,
        "final_url": final_url,
        "path": str(destination),
        "size": total_bytes,
        "content_type": content_type,
        "success": True,
    }