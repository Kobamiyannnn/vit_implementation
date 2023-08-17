"""
Vision Transformer (ViT) の実装

ViTの構成要素
    - Input Layer
    - Encoder
    - MLP Head
"""

import torch
import torch.nn as nn


class VitInputLayer(nn.Module):
    """
    Input Layerの実装
        - 画像の入力
        - パッチに分割
        - パッチの埋め込み
        - クラストークンの結合
        - 位置埋め込み
    """

    def __init__(self,
        in_channels  : int = 3,
        emb_dim      : int = 384,
        num_patch_row: int = 2,
        image_size   : int = 32,
        ):
        """モジュールの部品の定義
        --

        モジュールのインスタンスを定義する

        Parameters
        ----------
        `in_channels` : `int`
            入力画像のチャンネル数
        `emb_dim` : `int`
            埋め込み後のベクトルの長さ
        `num_patch_row` : `int`
            高さ方向のパッチの数。デフォルトでは2*2に分割する
        `image_size` : `int`
            入力画像の一辺の長さ。高さと幅が同じだと仮定

        Returns
        -------
        None
        """
        super(VitInputLayer, self).__init__()
        self.in_channels   = in_channels
        self.emb_dim       = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size    = image_size

        # パッチの数
        # 例: 入力画像を2*2のパッチに分ける場合、num_patchは4となる
        self.num_patch = self.num_patch_row ** 2

        # 各パッチの大きさ
        # 例: 入力画像の1辺の長さが32の場合、patch_sizeは16となる
        self.patch_size = int(self.image_size // self.num_patch_row)

        # 入力画像をパッチに分割 & パッチの埋め込みを行う層
        # 畳み込み層のカーネルサイズとストライド幅をパッチと同じ大きさに設定することで同時に実現
        self.patch_emb_layer = nn.Conv2d(
            in_channels  = self.in_channels,
            out_channels = self.emb_dim,
            kernel_size  = self.patch_size,
            stride       = self.patch_size,
        )

        # クラストークン
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim)  # randnにより、1*1*emb_dimのテンソルを得る
        )

        # 位置埋め込み
        # クラストークンが先頭に結合されているため、長さemd_dimでパッチ数+1個用意する
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch + 1, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝搬の挙動の定義
        --
        Input Layerでの処理手順を定義する

        Parameters
        ----------
        `x` : `torch.Tensor`
            入力画像。形状は、(B, C, H, W)\n
            B: バッチサイズ、C: チャンネル数、H: 高さ、W: 幅
        Returns
        -------
        `z_0` : `torch.Tensor`
            Encoderへの入力。形状は、(B, N, D)\n
            B: バッチサイズ、N: トークン数、D: 埋め込みベクトルの長さ
        """

        # パッチの埋め込み (B, C, H, W) -> (B, D, H/P, W/P)
        # P: パッチ1辺の大きさ
        z_0 = self.patch_emb_layer(x)

        # パッチのflatten (B, D, H/P, W/P) -> (B, D, N_p)
        # N_p: パッチの数 (= H/P * W/P = H*W/P^2)
        z_0 = z_0.flatten(2)  # 引数はstart_dim H/P以降がflattenされるということ

        # 軸の入れ替え (B, D, N_p) -> (B, N_p, D)
        z_0 = z_0.transpose(1, 2)  # transposeにより転置

        # パッチの埋め込みの先頭にクラストークンを結合
        # (B, N_p, D) -> (B, (N_p + 1), D) -> (B, N, D)
        # cls_tokenが(1,1,D)だから、repeatメソッドによって(B,1,D)に変換してから結合
        z_0 = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1)

        # 位置埋め込みの加算 (B, N, D) -> (B, N, D)
        z_0 += self.pos_emb

        return z_0
    
if __name__ == "__main__":
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    print(x.shape)
    input_layer = VitInputLayer(in_channels=channel, image_size=height)
    print(input_layer)
    z_0 = input_layer(x)  # input_layer.forward(x)とした場合と同じ挙動 なぜ？

    print(z_0.shape)