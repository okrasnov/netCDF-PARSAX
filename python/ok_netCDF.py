import tarfile
#import json
import io
import numpy as np
from pathlib import Path
import xarray as xr

# v.1.0 19.06.2026 OK@TUD & CoPilot

def index_tar(tar_path):
    """Build a byte-offset index of all NetCDF files in a TAR."""
    index = {}
    hf=tarfile.open(tar_path)
    with hf as tf:
        for member in tf.getmembers():
            #print(f"Found member: {member.name}, offset: {member.offset_data}, size: {member.size}")
            if member.name.endswith(".nc"):
                index[member.name] = {
                    "offset": member.offset_data,
                    "size": member.size
                }
    return index


# jump directly to any file by byte offset

def open_nc_from_tar(tar_path, index, filename):
    offset = index[filename]["offset"]
    size = index[filename]["size"]
    with open(tar_path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    return xr.open_dataset(io.BytesIO(data))


def combine_tar_nc_directory(
    data_dir,
    tar_glob="*.tar",
    nc_suffix=".nc",
    concat_dim="time",
    decode_times=True,
    open_kwargs=None,
    concat_kwargs=None,
):
    data_dir = Path(data_dir)
    tar_paths = sorted(data_dir.glob(tar_glob))
    if not tar_paths:
        raise FileNotFoundError(f"No tar files found in {data_dir!s}")

    open_kwargs = open_kwargs or {}
    concat_kwargs = concat_kwargs or {
        "dim": concat_dim,
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
    }

    datasets = []
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tf:
            members = [
                m
                for m in tf.getmembers()
                if m.isfile() and m.name.endswith(nc_suffix)
            ]
            members.sort(key=lambda m: m.name)
            for member in members:
                with tf.extractfile(member) as member_file:
                    data = member_file.read()
                ds = xr.open_dataset(
                    io.BytesIO(data),
                    decode_times=decode_times,
                    **open_kwargs,
                )
                datasets.append(ds)

    if not datasets:
        raise ValueError(f"No NetCDF members found in tar files under {data_dir!s}")

    combined = xr.concat(datasets, **concat_kwargs)
    return combined


def combine_nc_from_tar(
    tar_path,
    nc_suffix=".nc",
    concat_dim="time",
    decode_times=True,
    open_kwargs=None,
    concat_kwargs=None,
):
    """Read all NetCDF files from one hourly TAR and concatenate along time.

    Usage:
        combined_ds = combine_nc_from_tar(
            "/path/to/archive_2026-06-09T09.tar"
        )

        # with custom xarray open_dataset options
        combined_ds = combine_nc_from_tar(
            "/path/to/archive_2026-06-09T09.tar",
            decode_times=False,
            open_kwargs={"engine": "netcdf4"},
        )
    """
    open_kwargs = open_kwargs or {}
    concat_kwargs = concat_kwargs or {
        "dim": concat_dim,
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
    }

    datasets = []
    member_ranges = []  # collect (name, start, end, count)
    with tarfile.open(tar_path, "r") as tf:
        members = [
            m
            for m in tf.getmembers()
            if m.isfile() and m.name.endswith(nc_suffix)
        ]
        members.sort(key=lambda m: m.name)
        for member in members:
            with tf.extractfile(member) as member_file:
                data = member_file.read()
            ds = xr.open_dataset(
                io.BytesIO(data),
                decode_times=decode_times,
                **open_kwargs,
            )

            # ensure dataset has a `time` coordinate
            if "time" not in ds.coords and "time" not in ds:
                print(f"Skipping member without 'time' coord: {member.name}")
                continue

            # sort each dataset by time to ensure increasing order
            try:
                ds = ds.sortby("time")
            except Exception:
                # attempt a best-effort conversion and sort
                try:
                    ds = xr.decode_cf(ds)
                    ds = ds.sortby("time")
                except Exception:
                    print(f"Warning: could not sort member {member.name} by time; including as-is")

            # record member time range
            try:
                times = ds["time"].values
                if times.size:
                    member_ranges.append((member.name, times[0], times[-1], times.size))
            except Exception:
                member_ranges.append((member.name, None, None, 0))

            datasets.append(ds)

    if not datasets:
        raise ValueError(f"No NetCDF members found in {tar_path}")

    # concatenate along time and then ensure global sort by time
    combined = xr.concat(datasets, **concat_kwargs)
    try:
        combined = combined.sortby("time")
    except Exception:
        print("Warning: combined dataset could not be sorted by 'time'")

    # report overlapping member ranges
    if member_ranges:
        ranges = [r for r in member_ranges if r[1] is not None]
        if ranges:
            ranges.sort(key=lambda x: x[1])
            overlaps = []
            prev_name, prev_start, prev_end, prev_count = ranges[0]
            for name, start, end, cnt in ranges[1:]:
                if start <= prev_end:
                    overlaps.append((prev_name, prev_start, prev_end, name, start, end))
                    if end > prev_end:
                        prev_end = end
                        prev_name = name
                else:
                    prev_name, prev_start, prev_end, prev_count = name, start, end, cnt

            if overlaps:
                print(f"Detected {len(overlaps)} overlapping member time ranges in {tar_path}:")
                for a in overlaps:
                    print(f"  Overlap between {a[0]} [{a[1]} - {a[2]}] and {a[3]} [{a[4]} - {a[5]}]")
            else:
                print(f"No overlapping member ranges detected in {tar_path}.")

    # detect and remove duplicate timestamps in combined
    try:
        times = combined["time"].values
        unique_times, first_idx, counts = np.unique(times, return_index=True, return_counts=True)
        if np.any(counts > 1):
            num_dup = int(times.size - unique_times.size)
            print(f"Detected {num_dup} duplicate timestamps in combined dataset from {tar_path}.")
            dup_times = unique_times[counts > 1]
            print("Duplicated timestamps (sample):", dup_times[:10])
            # keep first occurrence of each unique time
            keep_idx = np.sort(first_idx)
            combined = combined.isel(time=keep_idx)
            print(f"Removed duplicates; new time length: {combined['time'].size}")
        else:
            print(f"No duplicate timestamps found in combined dataset from {tar_path}.")
    except Exception:
        print("Warning: could not inspect duplicate timestamps for combined dataset")

    return combined

