"""Small terminal progress displays and callback hooks without extra dependencies."""

import sys
import time


def format_duration(seconds):
    """Format elapsed wall-clock seconds for a compact progress display."""
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


class ProgressBar:
    """Display progress, or call progress(completed, total) on each update.

    progress=False disables output; True writes to stderr. Callback mode is
    useful for notebooks or forwarding worker progress to a parent process.
    Terminal output is throttled and uses one updating line; redirected output
    uses separate lines. Early stopping retains the actual completion count.
    """

    def __init__(self, total, description, progress=False):
        if not isinstance(progress, bool) and not callable(progress):
            raise TypeError('progress must be a boolean or a callable.')
        self.total = total
        self.description = description
        self.progress = progress
        self.completed = 0
        self.started = time.monotonic()
        self.last_render = float('-inf')
        self.last_width = 0

    def __enter__(self):
        self.update(0, force=True)
        return self

    def advance(self, count=1):
        """Record additional completed units."""
        self.update(self.completed + count)

    def update(self, completed=None, detail='', force=False):
        """Set the completed count, optionally adding a current-stage message."""
        if completed is not None:
            self.completed = completed
        if callable(self.progress):
            self.progress(self.completed, self.total)
            return
        if not self.progress:
            return
        now = time.monotonic()
        interval = 0.2 if sys.stderr.isatty() else 5.0
        if not force and now - self.last_render < interval:
            return
        fraction = min(1.0, self.completed / self.total) if self.total else 1.0
        filled = int(20 * fraction)
        bar = '#' * filled + '-' * (20 - filled)
        text = (f'{self.description} [{bar}] {self.completed}/{self.total}'
                f' elapsed {format_duration(now - self.started)}')
        if detail:
            text += ' | ' + detail
        if sys.stderr.isatty():
            sys.stderr.write('\r' + text.ljust(self.last_width))
            self.last_width = len(text)
        else:
            sys.stderr.write(text + '\n')
        sys.stderr.flush()
        self.last_render = now

    def __exit__(self, exc_type, exc_value, traceback):
        if self.progress and not callable(self.progress):
            detail = ('failed' if exc_type else
                      'complete' if self.completed == self.total else 'stopped early')
            self.update(detail=detail, force=True)
            if sys.stderr.isatty():
                sys.stderr.write('\n')
                sys.stderr.flush()
