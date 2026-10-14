#!/usr/bin/env python3
"""Run a command with inherited Linux seccomp rules denying IPv4/IPv6 sockets.

AF_UNIX remains available for DataLoader IPC. This is a verification tool,
not a substitute for a site's firewall policy. It never falls back silently.
"""
import ctypes
import ctypes.util
import errno
import os
import socket
import sys


class ArgCompare(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def block_network():
    name = ctypes.util.find_library("seccomp")
    if not name:
        raise RuntimeError("libseccomp is required; use an OS network sandbox instead")
    lib = ctypes.CDLL(name, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                         ctypes.c_int, ctypes.c_uint,
                                         ctypes.POINTER(ArgCompare)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    try:
        syscall = lib.seccomp_syscall_resolve_name(b"socket")
        if syscall < 0:
            raise RuntimeError("socket syscall unavailable on this platform")
        for family in (socket.AF_INET, socket.AF_INET6):
            comparison = ArgCompare(0, 4, family, 0)  # SCMP_CMP_EQ
            result = lib.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM,
                                               syscall, 1, ctypes.byref(comparison))
            if result:
                raise RuntimeError(f"seccomp_rule_add failed: {result}")
        result = lib.seccomp_load(ctx)
        if result:
            raise RuntimeError(f"seccomp_load failed: {result}")
    finally:
        lib.seccomp_release(ctx)
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except PermissionError:
            continue
        else:
            sock.close()
            raise RuntimeError(f"Network socket unexpectedly allowed: {family}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
        pass
    print("Verified: IPv4/IPv6 sockets denied by seccomp; AF_UNIX available.",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/no_network.py COMMAND [ARG ...]")
    block_network()
    os.execvp(sys.argv[1], sys.argv[1:])
