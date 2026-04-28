#!/usr/bin/env python3
"""
Work around mbientlab MetaWare.connect() bug: USB MetaWearUSB.connect_async passes
Const.STATUS_ERROR_TIMEOUT (int) on failure, then MetaWear.connect() does raise result[0],
which raises TypeError (exceptions must derive from BaseException).
"""
from __future__ import print_function

import os
import pwd
import grp
from threading import Event

from mbientlab.metawear.cbindings import Const


def metawear_connect(device, **kwargs):
    """
    Same behavior as MetaWear.connect() but turns integer transport status codes into
    RuntimeError so callers get a normal exception.
    """
    e = Event()
    result = []

    def completed(error):
        result.append(error)
        e.set()

    device.connect_async(completed, **kwargs)
    e.wait()
    if not result:
        return
    err = result[0]
    if err is None:
        return
    if isinstance(err, BaseException):
        raise err
    if err == Const.STATUS_ERROR_TIMEOUT:
        # Most common reason on Linux: /dev/ttyACM* exists but user lacks dialout permissions,
        # or another process has the port open. Help the user diagnose quickly.
        usb_path = None
        try:
            usb_path = device.usb._device_path(device.address)
        except Exception:
            usb_path = None

        who = "unknown"
        groups = []
        try:
            who = pwd.getpwuid(os.getuid()).pw_name
            groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
        except Exception:
            pass

        perms = None
        if usb_path and os.path.exists(usb_path):
            try:
                st = os.stat(usb_path)
                perms = {
                    "path": usb_path,
                    "mode_octal": oct(st.st_mode & 0o777),
                    "uid": st.st_uid,
                    "gid": st.st_gid,
                    "user": pwd.getpwuid(st.st_uid).pw_name if st.st_uid is not None else None,
                    "group": grp.getgrgid(st.st_gid).gr_name if st.st_gid is not None else None,
                    "readable": os.access(usb_path, os.R_OK),
                    "writable": os.access(usb_path, os.W_OK),
                }
            except Exception:
                perms = {"path": usb_path}

        raise RuntimeError(
            "USB serial open failed (SDK status TIMEOUT).\n"
            "\n"
            "Diagnostics:\n"
            "- user: {}\n"
            "- groups: {}\n"
            "- usb_device_path: {}\n"
            "- usb_path_perms: {}\n"
            "\n"
            "Typical fixes:\n"
            "1) Permissions: add user to dialout, then re-login:\n"
            "   sudo usermod -aG dialout {}\n"
            "2) Close any app holding the port (/dev/ttyACM*), then unplug/replug the MetaMotionS.\n"
            "3) If permissions look fine, try running once with sudo to confirm it's permissions-related.\n"
            "   (If sudo works, it’s definitely groups/udev.)\n".format(
                who,
                ",".join(groups) if groups else "(unknown)",
                usb_path or "(none found)",
                perms or "(unavailable)",
                who,
            )
        )
    raise RuntimeError("MetaWear connect failed with non-exception status: {!r}".format(err))
