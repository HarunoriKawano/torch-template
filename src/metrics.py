from typing import Literal, Any

from states import BatchState

# TODO 評価指標を実装する
class Metrics:
    """
    評価指標を計算するためのインターフェース。
    バッチごとに状態を蓄積し、エポックの最後に計算・リセットする。
    """
    def update(self, batch_state: BatchState, mode: Literal["train", "val", "test"]) -> None: ...

    def compute(self, mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        """蓄積された状態から最終的な指標を計算して返す"""
        ...

    def reset(self) -> None:
        """内部状態を初期化する（次のエポック用）"""
        ...