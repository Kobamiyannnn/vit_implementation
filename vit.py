"""
Vision Transformer (ViT) の実装

ViTの構成要素
    - Input Layer
    - Encoder
    - MLP Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attentionの実装
        - 埋め込み
        - 内積
        - 加重和
    """

    def __init__(self,
                 emb_dim: int = 384,
                 head   : int = 3,
                 dropout: float = 0.,
        ):
        """
        Parameters
        ----------
        `emb_dim` : `int`
            埋め込み後のベクトルの長さ
        `head` : `int`
            ヘッドの数
        `dropout` : `float`
            ドロップアウト率

        Returns
        -------
        None
        """

        super(MultiHeadSelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.head = head
        self.sqrt_dh = self.head_dim ** 0.5  # D_hの二乗根。qk^Tを割るための係数

        # 入力をq、k、vに埋め込むための線形層
        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(p=dropout)  # ドロップアウト層

        # Multi-Head Self-Attentionの結果を出力に埋め込むための線形層
        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        `z` : `torch.Tensor`
            Multi-Head Self-Attentionへの入力。形状は、(B, N, D)\n
            B: バッチサイズ、N: トークン数、D: ベクトル長

        Returns
        -------
        `out` : `torch.Tensor`
            Multi-Head Self-Attentionの出力。形状は、(B, N, D)\n
            B: バッチサイズ、N: トークン数、D: 埋め込みベクトルの長さ
        """

        batch_size, num_patch, _ = z.size()

        # Self-Attentionの埋め込み
        # (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # q、k、vをヘッドに分ける
        # ベクトルをヘッドの個数（h）に分ける
        # (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)  # *.view(): Tensorの次元操作
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)
        # Self-Attentionのために、(B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Self-Attentionの内積
        # (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)
        # (B, h, N, D//h) * (B, h, D//h, N) -> (B, h, N, N)
        dots = (q @ k_T) / self.sqrt_dh
        # 列方向にソフトマックス関数
        attn = F.softmax(dots, dim=-1)
        # ドロップアウト
        attn = self.attn_drop(attn)

        # Self-Attentionの加重和
        # (B, h, N, N) * (B, h, N, D//h) -> (B, h, N, D//h)
        out = attn @ v
        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        # (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # 出力層
        # (B, N, D) -> (B, N, D)
        out = self.w_o(out)
        return out
    

class VitEncoderBlock(nn.Module):
    """
    Encoder Blockの実装
        - Multi-Head Self-Attention + Layer Normalization
        - MLP + Layer Normalization
    """

    def __init__(self,
                 emb_dim   : int = 384,
                 head      : int = 8,
                 hidden_dim: int = 384 * 4,
                 dropout   : float = 0.,
        ):
        """
        Parameters
        ----------
        `emb_dim` : `int`
            埋め込み後のベクトルの長さ
        `head` : `int`
            ヘッドの数
        `hidden_dim` : `int`
            Encoder BlockのMLPにおける中間層のベクトルの長さ\n
            原論文に従ってemb_dimの4倍をデフォルト値に 
        `dropout` : `float`
            ドロップアウト率

        Returns
        -------
        None
        """

        super(VitEncoderBlock, self).__init__()
        # 1つ目のLayer Normalization
        self.ln1 = nn.LayerNorm(emb_dim)
        # Multi-Head Self-Attention
        self.msa = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            head=head,
            dropout=dropout
        )
        # 2つ目のLayer Normalization
        self.ln2 = nn.LayerNorm(emb_dim)
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        `z` : `torch.Tensor`
            Encoder Blockへの入力。形状は、(B, N, D)\n
            B: バッチサイズ、N: トークン数、D: ベクトルの長さ

        Returns
        -------
        `out` : `torch.Tensor`
            Encoder Blockの出力。形状は、(B, N, D)\n
            B: バッチサイズ、N: トークン数、D: 埋め込みベクトルの長さ
        """

        # Encoder Blockの前半部分
        out = self.msa(self.ln1(z)) + z
        # Encder Blockの後半部分
        out = self.mlp(self.ln2(out)) + out
        return out
    

class Vit(nn.Module):
        """
        Vision Transformerの実装
            - Input Layer
            - Encoder
                - Encoder Block
            - MLP Head
        """
        def __init__(self,
                     in_channels   : int = 3,
                     num_classes   : int = 10,
                     emb_dim       : int = 384,
                     num_patch_row : int = 2,
                     image_size    : int = 32,
                     num_blocks    : int = 7,
                     head          : int = 8,
                     hidden_dim    : int = 384 * 4,
                     dropout       : float = 0.
            ):
            """
            Parameters
            ----------
            `in_channels` : `int`
                入力画像のチャンネル数
            `num_classes` : `int`
                画像分類のクラス数
            `emb_dim` : `int`
                埋め込み後のベクトルの長さ
            `num_patch_row` : `int`
                1辺のパッチの長さ
            `image_size` : `int`
                入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
            `num_blocks` : `int`
                Encoder Blockの数
            `head` : `int`
                ヘッドの数
            `hidden_dim` : `int`
                Encoder BlockのMLPにおける中間層のベクトルの長さ
            `dropout` : `float`
                ドロップアウト率
            
            Returns
            -------
            None
            """
            
            super(Vit, self).__init__()

            # Input Layer
            self.input_layer = VitInputLayer(
                in_channels=in_channels,
                emb_dim=emb_dim,
                num_patch_row=num_patch_row,
                image_size=image_size
            )

            # Encoder
            # Encoder Blockを多段に重ねる
            self.encoder = nn.Sequential(*[
                VitEncoderBlock(
                    emb_dim=emb_dim,
                    head=head,
                    hidden_dim=hidden_dim,
                    dropout=dropout
                )
                for _ in range(num_blocks)])  # *はアンパック
            
            # MLP Head
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, num_classes)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            `x` : `torch.Tensor`
                ViTへの入力画像。形状は、(B, C, H, W)\n
                B: バッチサイズ、C: チャンネル数、H: 高さ、W: 幅

            Returns
            -------
            `out` : `torch.Tensor`
                ViTの出力。形状は、(B, M)\n
                B: バッチサイズ、M: クラス数
            """

            # Input Layer
            # (B, C, H, W) -> (B, N, D)
            # N: トークン数（パッチ数+1）、D: ベクトルの長さ
            out = self.input_layer(x)
            # Encoder
            # (B, N, D) -> (B, N, D)
            out = self.encoder(out)
            # クラストークンのみ抜き出す
            # (B, N, D) -> (B, D)
            cls_token = out[:, 0]
            # MLP Head
            # (B, D) -> (B, M)
            pred = self.mlp_head(cls_token)
            return pred

if __name__ == "__main__":
    # Input Layerの確認
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    print("入力:", x.shape, "\n")
    input_layer = VitInputLayer(in_channels=channel, image_size=height)
    print(input_layer)
    z_0 = input_layer(x)  # input_layer.forward(x)とした場合と同じ挙動 なぜ？
    print("Input Layerの出力形状:               ", z_0.shape, "\n")

    # Multi-Head Self-Attentionの確認
    mhsa = MultiHeadSelfAttention()
    print(mhsa)
    out = mhsa(z_0) # (B, N, D)
    print("Multi-Head Self-Attentionの出力形状: ", out.shape, "\n")

    # Encoder Blockの確認
    vit_enc = VitEncoderBlock()
    print(vit_enc)
    z_1 = vit_enc(z_0)  # z_0は(B, N, D)
    print("Encoder Blockの出力形状: ", z_1.shape, "\n")
    print("-" * 50, "\n")

    # ViTの実装の確認
    num_classes = 10
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    vit = Vit(in_channels=channel, num_classes=num_classes)
    pred = vit(x)

    print("ViTの出力形状: ", pred.shape)  # (2, 10)(=(B, M))