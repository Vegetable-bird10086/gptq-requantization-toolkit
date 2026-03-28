#!/usr/bin/env python3
import argparse
import torch
import shutil
import os

KEYS_DEFAULT = ["tok_embeddings.weight", "output.weight"]


def process(path, keys=KEYS_DEFAULT, backup=True, dry=False):
    print(f"Loading: {path}")
    obj = torch.load(path, map_location="cpu")
    wrapper = None
    sd = obj
    if isinstance(obj, dict) and 'state_dict' in obj:
        wrapper = 'state_dict'
        sd = obj['state_dict']

    before = {k: tuple(v.shape) for k, v in sd.items() if k in keys}
    print("Before shapes:")
    for k, s in before.items():
        print(f"  {k}: {s}")

    # Group keys by underlying storage to preserve shared tensors and avoid
    # writing duplicate storages to disk (which can increase file size).
    groups = {}
    for k in keys:
        if k not in sd:
            print(f"Key not found, skip: {k}")
            continue
        v = sd[k]
        if not hasattr(v, 'shape') or v.dim() != 2:
            print(f"Key {k} is not 2-D tensor, skip (shape={getattr(v,'shape',None)})")
            continue
        if v.shape[0] < 1:
            print(f"Key {k} has zero rows, skip")
            continue
        try:
            ptr = v.storage().data_ptr()
        except Exception:
            # Fallback: use id() if storage pointer not available
            ptr = id(v.storage())
        if ptr not in groups:
            groups[ptr] = {
                'orig': v,
                'keys': [],
            }
        groups[ptr]['keys'].append(k)

    for ptr, info in groups.items():
        orig = info['orig']
        key_list = info['keys']
        # create a single cropped tensor for the whole shared group
        new_v = orig[:-1].clone()
        for k in key_list:
            sd[k] = new_v
            print(f"Shrunk {k}: {orig.shape} -> {new_v.shape} (shared group)")

    if dry:
        print("Dry run, no file changes.")
        return

    if backup:
        bak = path + ".backup"
        shutil.copy2(path, bak)
        print(f"Backup created: {bak}")

    # save preserving original wrapper structure
    if wrapper == 'state_dict':
        obj['state_dict'] = sd
        torch.save(obj, path)
    else:
        torch.save(sd, path)
    print(f"Saved modified file: {path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pth', required=True, help='Path to .pth to modify')
    p.add_argument('--keys', nargs='+', default=KEYS_DEFAULT, help='Keys to shrink')
    p.add_argument('--no-backup', dest='backup', action='store_false')
    p.add_argument('--dry', action='store_true')
    args = p.parse_args()
    process(args.pth, keys=args.keys, backup=args.backup, dry=args.dry)
