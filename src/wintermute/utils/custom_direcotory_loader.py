from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path
from typing import List, Callable, Type, Optional
import os

# Custom DirectoryLoader that allows excluding specific directories
class CustomDirectoryLoader(DirectoryLoader):
    def __init__(
        self,
        path: str,
        exclude_dirs: Optional[List[str]] = None,
        glob: str = "**/*",
        loader_cls: Type[TextLoader] = TextLoader,
        loader_kwargs: Optional[dict] = None,
        show_progress: bool = False,
        recursive: bool = True
    ):
        self.bast_path = Path(path).resolve()
        self.exclude_dirs = [self.bast_path / ed for ed in (exclude_dirs or [])]
        if loader_kwargs is None:
            loader_kwargs = {}
        super().__init__(
            path=path,
            glob=glob,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=show_progress,
            recursive=recursive
        )

    def _is_excluded(self, file_path: Path) -> bool:
        try:
            file_path = file_path.resolve()
            for ex in self.exclude_dirs:
                if ex in file_path.parents:
                    return True
        except FileNotFoundError:
            return True  # If the file does not exist, consider it excluded
        return False


    def load_exclude(self):  # ディレクトリを除外してドキュメントをロードするメソッド
        docs = []
        patterns = self.glob if isinstance(self.glob, (list, tuple)) else [self.glob]
        for pattern in patterns:
            for p in Path(self.path).rglob(pattern):
                if p.is_file() and not self._is_excluded(p):
                    loader = self.loader_cls(str(p), **self.loader_kwargs)
                    docs.extend(loader.load())
        return docs
