"""Compatibility shim for Python 3.13+.

The classic Sphinx/Jupyter Book stack still imports ``imghdr``, which was
removed from the Python standard library in 3.13. This minimal local copy is
enough for Sphinx's image utilities.
"""

from __future__ import annotations


tests = []


def what(file, h=None):
    f = None
    try:
        if h is None:
            if isinstance(file, (str, bytes)):
                f = open(file, "rb")
                h = f.read(32)
            else:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
        for tf in tests:
            res = tf(h, f)
            if res:
                return res
    finally:
        if f is not None:
            f.close()
    return None


def test_jpeg(h, f):
    if h[6:10] in (b"JFIF", b"Exif") or h[:4] == b"\xff\xd8\xff\xdb":
        return "jpeg"


tests.append(test_jpeg)


def test_png(h, f):
    if h.startswith(b"\211PNG\r\n\032\n"):
        return "png"


tests.append(test_png)


def test_gif(h, f):
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"


tests.append(test_gif)


def test_tiff(h, f):
    if h[:2] in (b"MM", b"II"):
        return "tiff"


tests.append(test_tiff)


def test_rgb(h, f):
    if h.startswith(b"\001\332"):
        return "rgb"


tests.append(test_rgb)


def test_pbm(h, f):
    if len(h) >= 3 and h[0] == ord("P") and h[1] in b"14" and h[2] in b" \t\n\r":
        return "pbm"


tests.append(test_pbm)


def test_pgm(h, f):
    if len(h) >= 3 and h[0] == ord("P") and h[1] in b"25" and h[2] in b" \t\n\r":
        return "pgm"


tests.append(test_pgm)


def test_ppm(h, f):
    if len(h) >= 3 and h[0] == ord("P") and h[1] in b"36" and h[2] in b" \t\n\r":
        return "ppm"


tests.append(test_ppm)


def test_rast(h, f):
    if h.startswith(b"\x59\xA6\x6A\x95"):
        return "rast"


tests.append(test_rast)


def test_xbm(h, f):
    if h.startswith(b"#define "):
        return "xbm"


tests.append(test_xbm)


def test_bmp(h, f):
    if h.startswith(b"BM"):
        return "bmp"


tests.append(test_bmp)


def test_webp(h, f):
    if h.startswith(b"RIFF") and h[8:12] == b"WEBP":
        return "webp"


tests.append(test_webp)


def test_exr(h, f):
    if h.startswith(b"\x76\x2f\x31\x01"):
        return "exr"


tests.append(test_exr)
