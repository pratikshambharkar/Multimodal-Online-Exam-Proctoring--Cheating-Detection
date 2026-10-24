"""Site-wide template context processors."""
import time

# Set once when the Django process starts and reused for every request.
# Appended as a query string to static CSS/JS URLs (see templates/base.html
# and any template that loads its own script) so a server restart after an
# edit always forces browsers to fetch the new file instead of quietly
# reusing a stale cached copy - the exact failure mode that made the gaze-
# calibration marker invisible for a browser that had cached an older
# style.css from earlier in the session, weeks before the calibration CSS
# was added.
_ASSET_VERSION = str(int(time.time()))


def asset_version(request):
    return {"ASSET_VERSION": _ASSET_VERSION}
