import objaverse
import objaverse.xl as oxl
from typing import Any, Dict, Hashable


def handle_found_object(local_path: str, file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    print("\n\n\n---HANDLE_FOUND_OBJECT CALLED---\n", f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")


def handle_modified_object(
    local_path: str,
    file_identifier: str,
    new_sha256: str,
    old_sha256: str,
    metadata: Dict[Hashable, Any],
) -> None:
    print("\n\n\n---HANDLE_MODIFIED_OBJECT CALLED---\n", f"  {local_path=}\n  {file_identifier=}\n  {old_sha256=}\n  {new_sha256}\n  {metadata=}\n\n\n")


def handle_missing_object(file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    print("\n\n\n---HANDLE_MISSING_OBJECT CALLED---\n", f"  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")


def handle_new_object(local_path: str, file_identifier: str, sha256: str, metadata: Dict[Hashable, Any]) -> None:
    print("\n\n\n---HANDLE_NEW_OBJECT CALLED---\n", f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n")


annotations = oxl.get_annotations(download_dir="data")  # default download directory
# sampled_df = annotations.groupby('source').apply(lambda x: x.sample(1)).reset_index(drop=True)
# sampled_df = annotations[annotations['source'] == 'sketchfab'].reset_index(drop=True)

download_quantity = 10  # 例如，下载100条记录
sampled_df = annotations[annotations['source'] == 'smithsonian'].head(download_quantity).reset_index(drop=True)

oxl.download_objects(
    objects=sampled_df,
    download_dir='data',
    handle_found_object=handle_found_object,
    handle_modified_object=handle_modified_object,
    handle_missing_object=handle_missing_object,
    handle_new_object=handle_new_object,
    save_repo_format="tar.gz",
)
