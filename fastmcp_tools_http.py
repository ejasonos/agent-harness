from __future__ import annotations

import os
from functools import wraps

from fastmcp import FastMCP
from dotenv import load_dotenv

from config.settings import load_settings
from workspace.manager import WorkspaceManager
from tools.math_fns import (
    addition as _addition,
    subtraction as _subtraction,
    multiply as _multiply,
    division as _division,
)

from tools.filesystem.read import read_file as _read_file
from tools.filesystem.write import write_file as _write_file
from tools.filesystem.edit import edit_file as _edit_file
from tools.filesystem.delete import delete_file as _delete_file
from tools.filesystem.search import search_files as _search_files
from tools.filesystem.tree import directory_tree as _directory_tree

from tools.project.detect import detect_project as _detect_project
from tools.project.framework import detect_framework as _detect_framework
from tools.project.commands import project_commands as _project_commands

from tools.git.status import git_status as _git_status
from tools.git.diff import git_diff as _git_diff
from tools.git.history import git_history as _git_history
from tools.git.operations import (
    git_add as _git_add,
    git_unstage as _git_unstage,
    git_commit as _git_commit,
    git_create_branch as _git_create_branch,
    git_switch_branch as _git_switch_branch,
    git_delete_branch as _git_delete_branch,
    git_restore as _git_restore,
)
from tools.shell.execute import (
    execute_command as _execute_command,
)
from tools.shell.process import (
    start_process as _start_process,
    process_status as _process_status,
    stop_process as _stop_process,
    remove_process as _remove_process,
)
from tools.shell.editing import (
    quote_argument as _quote_argument,
    build_command as _build_command,
    parse_command as _parse_command,
    inspect_command as _inspect_command,
)

from tools.web.search import web_search as _web_search
from tools.web.open import open_url as _open_url
from tools.web.extract import (
    extract_text as _extract_text,
    extract_links as _extract_links,
)
from tools.web.download import (
    download_file as _download_file,
)

from tools.browser.lifecycle import (
    browser_launch as _browser_launch,
    browser_close as _browser_close,
    browser_status as _browser_status,
)

from tools.browser.navigation import (
    browser_goto as _browser_goto,
    browser_back as _browser_back,
    browser_forward as _browser_forward,
    browser_reload as _browser_reload,
    browser_current_page as _browser_current_page,
)

from tools.browser.interaction import (
    browser_click as _browser_click,
    browser_double_click as _browser_double_click,
    browser_fill as _browser_fill,
    browser_type as _browser_type,
    browser_press as _browser_press,
    browser_select as _browser_select,
    browser_check as _browser_check,
    browser_uncheck as _browser_uncheck,
    browser_hover as _browser_hover,
)

from tools.browser.inspection import (
    browser_page_text as _browser_page_text,
    browser_element_text as _browser_element_text,
    browser_element_attribute as _browser_element_attribute,
    browser_element_exists as _browser_element_exists,
    browser_page_html as _browser_page_html,
)

from tools.browser.console import (
    browser_get_console as _browser_get_console,
    browser_clear_console as _browser_clear_console,
)

from tools.browser.network import (
    browser_get_requests as _browser_get_requests,
    browser_get_responses as _browser_get_responses,
    browser_clear_network as _browser_clear_network,
)

from tools.browser.viewport import (
    browser_screenshot as _browser_screenshot,
    browser_set_viewport as _browser_set_viewport,
    browser_scroll as _browser_scroll,
)

from tools.conversion.documents import (
    extract_document_text as _extract_document_text,
    convert_document as _convert_document,
)

from tools.conversion.images import (
    convert_image as _convert_image,
)

from tools.conversion.pdf import (
    extract_pdf_text as _extract_pdf_text,
    render_pdf_pages as _render_pdf_pages,
    create_pdf as _create_pdf,
)

from tools.conversion.web import (
    html_to_text as _html_to_text,
    html_to_pdf as _html_to_pdf,
)

from tools.screenshots.capture import (
    capture_screen as _capture_screen,
    capture_region as _capture_region,
    image_info as _image_info,
    crop_image as _crop_image,
    resize_image as _resize_image,
)

from tools.testing import (
    run_tests as _run_tests,
    run_test_file as _run_test_file,
    run_linter as _run_linter,
    run_formatter as _run_formatter,
    run_formatter_check as _run_formatter_check,
    run_typecheck as _run_typecheck,
    run_build as _run_build,
    collect_diagnostics as _collect_diagnostics,
)

load_dotenv()

mcp = FastMCP("Local Agent Tools")


# =========================================================
# WORKSPACE
# =========================================================

WORKSPACE = os.getenv(
    "AGENT_WORKSPACE",
    os.getcwd(),
)

settings = load_settings(WORKSPACE)

workspace = WorkspaceManager(
    settings.require_workspace(),
    allow_delete=settings.allow_delete,
    allow_write=True,
    allow_git_write=settings.allow_git_write,
    allow_shell=settings.allow_shell,
    allow_network=settings.allow_network,
)


def _network_required(function):
    @wraps(function)
    def checked(*args, **kwargs):
        workspace.check_network()
        return function(*args, **kwargs)

    return checked


# =========================================================
# TESTING TOOLS
# =========================================================

@mcp.tool()
def run_tests(
    command: str | None = None,
    cwd: str | None = None,
    timeout: int = 120,
) -> dict:
    return _run_tests(
        workspace,
        command,
        cwd=cwd,
        timeout=timeout,
    )


@mcp.tool()
def run_test_file(
    path: str,
    command: str | None = None,
    timeout: int = 120,
) -> dict:
    return _run_test_file(
        workspace,
        path,
        command=command,
        timeout=timeout,
    )


@mcp.tool()
def run_linter(
    command: str | None = None,
    timeout: int = 120,
) -> dict:
    return _run_linter(
        workspace,
        command,
        timeout=timeout,
    )


@mcp.tool()
def run_formatter_check(
    command: str | None = None,
    timeout: int = 120,
) -> dict:
    return _run_formatter_check(
        workspace,
        command,
        timeout=timeout,
    )


@mcp.tool()
def run_formatter(
    command: str | None = None,
    timeout: int = 120,
) -> dict:
    return _run_formatter(
        workspace,
        command,
        timeout=timeout,
    )


@mcp.tool()
def run_typecheck(
    command: str | None = None,
    timeout: int = 120,
) -> dict:
    return _run_typecheck(
        workspace,
        command,
        timeout=timeout,
    )


@mcp.tool()
def run_build(
    command: str | None = None,
    timeout: int = 300,
) -> dict:
    return _run_build(
        workspace,
        command,
        timeout=timeout,
    )


@mcp.tool()
def collect_diagnostics(
    result: dict,
) -> dict:
    return _collect_diagnostics(result)

# =========================================================
# SCREENSHOT / IMAGE TOOLS
# =========================================================

@mcp.tool()
def capture_screen(
    output_path: str,
) -> dict[str, object]:
    """Capture the primary screen."""

    return _capture_screen(
        workspace,
        output_path,
    )


@mcp.tool()
def capture_region(
    output_path: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> dict[str, object]:
    """Capture a rectangular screen region."""

    return _capture_region(
        workspace,
        output_path,
        x,
        y,
        width,
        height,
    )


@mcp.tool()
def image_info(
    path: str,
) -> dict[str, object]:
    """Inspect image dimensions and metadata."""

    return _image_info(
        workspace,
        path,
    )


@mcp.tool()
def crop_image(
    source_path: str,
    output_path: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> dict[str, object]:
    """Crop an image."""

    return _crop_image(
        workspace,
        source_path,
        output_path,
        left,
        top,
        right,
        bottom,
    )


@mcp.tool()
def resize_image(
    source_path: str,
    output_path: str,
    width: int,
    height: int | None = None,
) -> dict[str, object]:
    """Resize an image."""

    return _resize_image(
        workspace,
        source_path,
        output_path,
        width,
        height,
    )

# =========================================================
# CONVERSION TOOLS
# =========================================================

@mcp.tool()
def extract_document_text(
    path: str,
) -> dict[str, object]:
    """Extract readable text from a supported document."""

    return _extract_document_text(
        workspace,
        path,
    )


@mcp.tool()
def convert_document(
    source_path: str,
    output_path: str,
    output_format: str,
) -> dict[str, object]:
    """Convert a supported document to another format."""

    return _convert_document(
        workspace,
        source_path,
        output_path,
        output_format,
    )


@mcp.tool()
def convert_image(
    source_path: str,
    output_path: str,
    output_format: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, object]:
    """Convert, resize, or re-encode an image."""

    return _convert_image(
        workspace,
        source_path,
        output_path,
        output_format=output_format,
        width=width,
        height=height,
    )


@mcp.tool()
def extract_pdf_text(
    path: str,
) -> dict[str, object]:
    """Extract text from a PDF document."""

    return _extract_pdf_text(
        workspace,
        path,
    )


@mcp.tool()
def render_pdf_pages(
    path: str,
    output_directory: str,
    start_page: int = 1,
    end_page: int | None = None,
    dpi: int = 150,
) -> dict[str, object]:
    """Render PDF pages into PNG images."""

    return _render_pdf_pages(
        workspace,
        path,
        output_directory,
        start_page=start_page,
        end_page=end_page,
        dpi=dpi,
    )


@mcp.tool()
def create_pdf(
    text: str,
    output_path: str,
) -> dict[str, object]:
    """Create a PDF from plain text."""

    return _create_pdf(
        workspace,
        text,
        output_path,
    )


@mcp.tool()
def html_to_text(
    path: str,
) -> dict[str, object]:
    """Convert an HTML file into readable text."""

    return _html_to_text(
        workspace,
        path,
    )


@mcp.tool()
@_network_required
def html_to_pdf(
    source_path: str,
    output_path: str,
) -> dict[str, object]:
    """Convert an HTML file into a PDF."""

    return _html_to_pdf(
        workspace,
        source_path,
        output_path,
    )

# =========================================================
# BROWSER TOOLS
# =========================================================

@mcp.tool()
def browser_launch(
    headless: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 900,
) -> dict[str, object]:
    """Launch the persistent browser session."""

    return _browser_launch(
        headless=headless,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )


@mcp.tool()
def browser_close() -> dict[str, object]:
    """Close the persistent browser session."""

    return _browser_close()


@mcp.tool()
def browser_status() -> dict[str, object]:
    """Return browser session status."""

    return _browser_status()


@mcp.tool()
@_network_required
def browser_goto(
    url: str,
    timeout: int = 30_000,
) -> dict[str, object]:
    """Navigate the browser to a URL."""

    return _browser_goto(
        url,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_back(
    timeout: int = 30_000,
) -> dict[str, object]:
    """Navigate backward."""

    return _browser_back(
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_forward(
    timeout: int = 30_000,
) -> dict[str, object]:
    """Navigate forward."""

    return _browser_forward(
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_reload(
    timeout: int = 30_000,
) -> dict[str, object]:
    """Reload the current page."""

    return _browser_reload(
        timeout=timeout,
    )


@mcp.tool()
def browser_current_page() -> dict[str, object]:
    """Return the current URL and page title."""

    return _browser_current_page()


@mcp.tool()
@_network_required
def browser_click(
    selector: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Click an element."""

    return _browser_click(
        selector,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_double_click(
    selector: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Double-click an element."""

    return _browser_double_click(
        selector,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_fill(
    selector: str,
    value: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Fill an input or editable element."""

    return _browser_fill(
        selector,
        value,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_type(
    selector: str,
    text: str,
    delay: int = 0,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Type text into an element."""

    return _browser_type(
        selector,
        text,
        delay=delay,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_press(
    selector: str,
    key: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Press a keyboard key on an element."""

    return _browser_press(
        selector,
        key,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_select(
    selector: str,
    value: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Select an option from a select element."""

    return _browser_select(
        selector,
        value,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_check(
    selector: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Check a checkbox."""

    return _browser_check(
        selector,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_uncheck(
    selector: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Uncheck a checkbox."""

    return _browser_uncheck(
        selector,
        timeout=timeout,
    )


@mcp.tool()
@_network_required
def browser_hover(
    selector: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Hover over an element."""

    return _browser_hover(
        selector,
        timeout=timeout,
    )


@mcp.tool()
def browser_page_text(
    max_length: int = 100_000,
) -> dict[str, object]:
    """Read visible text from the current page."""

    return _browser_page_text(
        max_length=max_length,
    )


@mcp.tool()
def browser_element_text(
    selector: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Read text from an element."""

    return _browser_element_text(
        selector,
        timeout=timeout,
    )


@mcp.tool()
def browser_element_attribute(
    selector: str,
    attribute: str,
    timeout: int = 10_000,
) -> dict[str, object]:
    """Read an element attribute."""

    return _browser_element_attribute(
        selector,
        attribute,
        timeout=timeout,
    )


@mcp.tool()
def browser_element_exists(
    selector: str,
) -> dict[str, object]:
    """Check whether an element exists."""

    return _browser_element_exists(
        selector
    )


@mcp.tool()
def browser_page_html(
    selector: str | None = None,
    max_length: int = 200_000,
) -> dict[str, object]:
    """Inspect page or element HTML."""

    return _browser_page_html(
        selector,
        max_length=max_length,
    )


@mcp.tool()
def browser_get_console(
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return captured browser console messages."""

    return _browser_get_console(
        limit=limit
    )


@mcp.tool()
def browser_clear_console() -> dict[str, object]:
    """Clear captured browser console messages."""

    return _browser_clear_console()


@mcp.tool()
def browser_get_requests(
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return captured network requests."""

    return _browser_get_requests(
        limit=limit
    )


@mcp.tool()
def browser_get_responses(
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return captured network responses."""

    return _browser_get_responses(
        limit=limit
    )


@mcp.tool()
def browser_clear_network() -> dict[str, object]:
    """Clear captured network activity."""

    return _browser_clear_network()


@mcp.tool()
def browser_screenshot(
    path: str | None = None,
    full_page: bool = False,
) -> dict[str, object]:
    """Capture a screenshot of the current page."""

    output_path = (
        str(workspace.write_path(path))
        if path is not None
        else None
    )

    return _browser_screenshot(
        output_path,
        full_page=full_page,
    )


@mcp.tool()
@_network_required
def browser_set_viewport(
    width: int,
    height: int,
) -> dict[str, object]:
    """Change the browser viewport size."""

    return _browser_set_viewport(
        width,
        height,
    )


@mcp.tool()
@_network_required
def browser_scroll(
    x: int = 0,
    y: int = 0,
) -> dict[str, object]:
    """Scroll the current page."""

    return _browser_scroll(
        x,
        y,
    )

# =========================================================
# WEB TOOLS
# =========================================================

@mcp.tool()
@_network_required
def web_search(
    query: str,
    max_results: int = 5,
) -> dict[str, object]:
    """
    Search the public web with Tavily and return structured results.
    """

    return _web_search(
        query,
        max_results=max_results,
    )


@mcp.tool()
@_network_required
def web_open(
    url: str,
    timeout: int = 30,
) -> dict[str, object]:
    """
    Retrieve a public HTTP or HTTPS URL.
    """

    return _open_url(
        url,
        timeout=timeout,
    )


@mcp.tool()
def web_extract_text(
    html: str,
    max_length: int = 100_000,
) -> dict[str, object]:
    """
    Extract readable text from an HTML document.
    """

    return _extract_text(
        html,
        max_length=max_length,
    )


@mcp.tool()
def web_extract_links(
    html: str,
    max_results: int = 200,
) -> list[dict[str, str]]:
    """
    Extract hyperlinks from an HTML document.
    """

    return _extract_links(
        html,
        max_results=max_results,
    )


@mcp.tool()
@_network_required
def web_download(
    url: str,
    path: str,
    timeout: int = 60,
) -> dict[str, object]:
    """
    Download a remote resource into the agent workspace.
    """

    return _download_file(
        workspace,
        url,
        path,
        timeout=timeout,
    )

# =========================================================
# SHELL TOOLS
# =========================================================

@mcp.tool()
def execute_command(
    command: str,
    cwd: str | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """
    Execute a shell command inside the agent workspace.

    Returns stdout, stderr, return code, and execution status.
    """

    return _execute_command(
        workspace,
        command,
        cwd=cwd,
        timeout=timeout,
        env=env,
    )


@mcp.tool()
def start_process(
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    """
    Start a persistent background process.

    Returns a process_id that can be used with process_status
    and stop_process.
    """

    return _start_process(
        workspace,
        command,
        cwd=cwd,
        env=env,
    )


@mcp.tool()
def process_status(
    process_id: str,
) -> dict[str, object]:
    """Return the current status and captured output of a process."""

    return _process_status(
        process_id,
    )


@mcp.tool()
def stop_process(
    process_id: str,
    force: bool = False,
) -> dict[str, object]:
    """Stop a running managed process."""

    return _stop_process(
        process_id,
        force=force,
    )


@mcp.tool()
def remove_process(
    process_id: str,
) -> dict[str, object]:
    """Remove a completed process from the process registry."""

    return _remove_process(
        process_id,
    )


@mcp.tool()
def quote_shell_argument(
    value: str,
) -> str:
    """Safely quote one shell argument."""

    return _quote_argument(value)


@mcp.tool()
def build_shell_command(
    executable: str,
    arguments: list[str] | None = None,
) -> str:
    """Build a shell command with individually quoted arguments."""

    return _build_command(
        executable,
        arguments,
    )


@mcp.tool()
def parse_shell_command(
    command: str,
) -> list[str]:
    """Parse a shell command into executable and arguments."""

    return _parse_command(command)


@mcp.tool()
def inspect_shell_command(
    command: str,
) -> dict[str, object]:
    """Inspect a shell command without executing it."""

    return _inspect_command(command)

# =========================================================
# GIT TOOLS
# =========================================================

@mcp.tool()
def git_status() -> dict[str, object]:
    """Return the current Git repository status."""

    return _git_status(workspace)


@mcp.tool()
def git_diff(
    staged: bool = False,
    path: str | None = None,
) -> dict[str, object]:
    """
    Return Git changes.

    By default returns unstaged changes.
    Set staged=True to inspect staged changes.
    """

    return _git_diff(
        workspace,
        staged=staged,
        path=path,
    )


@mcp.tool()
def git_history(
    limit: int = 20,
    path: str | None = None,
) -> list[dict[str, str]]:
    """Return recent Git commit history."""

    return _git_history(
        workspace,
        limit=limit,
        path=path,
    )


@mcp.tool()
def git_add(
    paths: list[str],
) -> dict[str, object]:
    """Stage specific files or directories for a Git commit."""

    return _git_add(
        workspace,
        paths,
    )


@mcp.tool()
def git_unstage(
    paths: list[str],
) -> dict[str, object]:
    """Remove specific files from the Git staging area."""

    return _git_unstage(
        workspace,
        paths,
    )


@mcp.tool()
def git_commit(
    message: str,
) -> dict[str, object]:
    """
    Commit currently staged Git changes.

    Only changes already in the staging area are committed.
    """

    return _git_commit(
        workspace,
        message,
    )


@mcp.tool()
def git_create_branch(
    branch: str,
    checkout: bool = False,
) -> dict[str, object]:
    """
    Create a local Git branch.

    Set checkout=True to switch to it immediately.
    """

    return _git_create_branch(
        workspace,
        branch,
        checkout=checkout,
    )


@mcp.tool()
def git_switch_branch(
    branch: str,
) -> dict[str, object]:
    """Switch to an existing local Git branch."""

    return _git_switch_branch(
        workspace,
        branch,
    )


@mcp.tool()
def git_delete_branch(
    branch: str,
    force: bool = False,
) -> dict[str, object]:
    """
    Delete a local Git branch.

    force=False preserves Git's normal safety checks.
    """

    return _git_delete_branch(
        workspace,
        branch,
        force=force,
    )


@mcp.tool()
def git_restore(
    paths: list[str],
    staged: bool = False,
) -> dict[str, object]:
    """
    Restore Git changes for specific paths.

    By default restores working-tree changes.
    Set staged=True to restore the index.
    """

    return _git_restore(
        workspace,
        paths,
        staged=staged,
    )

# =========================================================
# PROJECT TOOLS
# =========================================================

@mcp.tool()
def detect_project(
    path: str = ".",
) -> dict[str, object]:
    """
    Detect the project type and configuration markers
    inside the agent workspace.
    """

    return _detect_project(
        workspace,
        path,
    )


@mcp.tool()
def detect_framework(
    path: str = ".",
) -> dict[str, object]:
    """
    Detect application frameworks and major technologies
    from project configuration files.
    """

    return _detect_framework(
        workspace,
        path,
    )


@mcp.tool()
def project_commands(
    path: str = ".",
) -> dict[str, object]:
    """
    Discover commands defined by the current project.

    This tool only discovers commands. It does not execute them.
    """

    return _project_commands(
        workspace,
        path,
    )

# =========================================================
# FILESYSTEM TOOLS
# =========================================================

@mcp.tool()
def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read a text file inside the agent workspace."""

    return _read_file(
        workspace,
        path,
        start_line,
        end_line,
    )


@mcp.tool()
def write_file(
    path: str,
    content: str,
) -> str:
    """Write text content to a file inside the agent workspace."""

    return _write_file(
        workspace,
        path,
        content,
    )


@mcp.tool()
def edit_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    """Replace text inside a file in the agent workspace."""

    return _edit_file(
        workspace,
        path,
        old_text,
        new_text,
        replace_all=replace_all,
    )


@mcp.tool()
def delete_file(
    path: str,
) -> str:
    """Delete a file or directory inside the agent workspace."""

    return _delete_file(
        workspace,
        path,
    )


@mcp.tool()
def search_files(
    query: str,
    path: str = ".",
    extensions: list[str] | None = None,
    max_results: int = 100,
) -> list[dict[str, object]]:
    """Search text files inside the agent workspace."""

    return _search_files(
        workspace,
        query,
        path,
        extensions=extensions,
        max_results=max_results,
    )


@mcp.tool()
def directory_tree(
    path: str = ".",
    max_depth: int = 4,
) -> str:
    """Return a readable directory tree."""

    return _directory_tree(
        workspace,
        path,
        max_depth=max_depth,
    )


# =========================================================
# MATH TOOLS
# =========================================================

@mcp.tool()
def addition(a: float, b: float) -> float:
    """Add two numbers."""
    return _addition(a, b)


@mcp.tool()
def subtraction(a: float, b: float) -> float:
    """Subtract b from a."""
    return _subtraction(a, b)


@mcp.tool()
def multiply(a: float, b: float) -> dict:
    """Multiply two numbers."""
    return _multiply(a, b)


@mcp.tool()
def division(a: float, b: float) -> dict:
    """Divide a by b."""
    return _division(a, b)

# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":
    print(f"Agent workspace: {workspace.root}")

    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8082,
    )