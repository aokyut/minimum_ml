# minimum_ml

**Rustによる実験的機械学習ライブラリ**

`minimum_ml` は、教育目的および依存関係を最小限に抑えた環境向けに設計された、軽量で実験的な機械学習ライブラリです。シンプルさとフットプリントの小ささに重点を置き、ニューラルネットワークの基本的な構成要素を提供します。

## 特徴

- **最小限の依存関係**: コアライブラリは `getrandom`（シード生成用）とローカルの derive マクロのみに依存しています。`ndarray`、`tch`、`rand` などの重量級クレートは不要です。
- **カスタム乱数生成**: 独自の `XorShift64` 実装を使用しています。
- **量子化サポート**: 8ビット整数（i8）量子化レイヤーと、SIMD最適化（AVX2）された行列演算を含みます。
- **軽量ロギング**: 純粋な Rust で実装された TensorBoard 互換ロガー（Scalar のみ）を内蔵。
- **自動微分**: 基本的な誤差逆伝播エンジンを搭載しています。

## インストール

`Cargo.toml` に以下を追加してください：

```toml
[dependencies]
minimum_ml = { git = "https://github.com/aokyut/minimum_ml" }
```

### 機能フラグ (Features)

- `default`: 追加機能なし。
- `logging`: TensorBoard ロギングサポートを有効にします（標準ライブラリのみ使用）。
- `full`: 全機能を有効にします。

## 使用ガイド

### 1. `sequential!` マクロによるネットワーク定義

`sequential!` マクロを使うと、レイヤーを順番に接続して定義できます。

```rust
use minimum_ml::ml::{Graph, Tensor};
use minimum_ml::ml::params::MM;
use minimum_ml::ml::funcs::{ReLU, Softmax, CrossEntropyLoss};

fn main() {
    let mut g = Graph::new();
    
    // 1. 入力とターゲット（正解ラベル）のプレースホルダーを定義
    let input = g.push_placeholder();
    let target = g.push_placeholder();
    
    // 2. ネットワーク構造の定義
    let network_output = minimum_ml::sequential!(
        g,
        input,
        [
            MM::new(784, 128), // 線形層 (784 -> 128)
            ReLU::new(),       // 活性化関数
            MM::new(128, 10),  // 線形層 (128 -> 10)
            Softmax::new(),    // 出力層
        ]
    );

    // 3. 損失関数の定義
    // ネットワークの出力とターゲットを損失関数に接続
    let loss = g.add_layer(
        vec![network_output, target], 
        Box::new(CrossEntropyLoss::new())
    );
    
    // 4. グラフのターゲット設定
    // 計算すべき対象（loss）と、外部から与える入力（placeholder）を設定
    g.set_target(loss);
    g.set_placeholder(vec![input, target]);
    
    // (任意) オプティマイザの設定
    let optimizer = minimum_ml::ml::optim::Adam::new(0.001, 0.9, 0.999);
    g.optimizer = Some(Box::new(optimizer));
}
```

### 2. 学習ループ（フォワード・バックワード）

グラフ構築後は、以下のように学習ループを回します。

```rust
// 学習ループ内...
// let input_tensor = ...;
// let target_tensor = ...;

// フォワードパス（順伝播）
let loss_val = g.forward(vec![input_tensor, target_tensor]);
println!("Loss: {:?}", loss_val.as_f32_slice()[0]);

// バックワードパス（勾配計算）
g.backward();

// オプティマイズ（重み更新）
g.optimize();

// 次のイテレーションのために勾配と計算フローをリセット
g.reset();
```

### 3. Datasetの実装とDataloaderの使用

独自のデータセットは `Dataset` トレイトを実装することで、`Dataloader` を使ってバッチ化やシャッフルが容易に行えます。

```rust
use minimum_ml::dataset::{Dataset, Dataloader, Stackable};

// 1. データ項目の構造体定義
#[derive(Stackable)] // バッチ化のためのマクロ
pub struct MyData {
    x: Tensor,
    y: Tensor,
}

// 2. Dataset構造体の定義
struct MyDataset {
    data: Vec<MyData>,
}

// 3. Datasetトレイトの実装
impl Dataset for MyDataset {
    type Item = MyData;
    fn len(&self) -> usize { self.data.len() }
    fn get(&self, index: usize) -> Self::Item { 
        // 指定インデックスのデータを返す
        // ... 
    }
}

// 4. Dataloaderの使用
let dataset = MyDataset { ... };
let loader = Dataloader::new(dataset, 64 /* batch_size */, true /* strict_batch_size */);

for batch in loader.iter_batch() {
    let batch_x = batch.x; // バッチ化されたTensor
    let batch_y = batch.y;
    // ... グラフに入力 ...
}
```

### 4. モデルの保存と読み込み

学習したパラメータを保存・ロードできます。

```rust
// モデルの保存
// "my_model" ディレクトリを作成し、その中にパラメータを保存します
g.save("my_model");

// モデルの読み込み
// "my_model" ディレクトリからパラメータを読み込みます
g.load("my_model");
```

### 5. 利用可能なコンポーネント

| コンポーネント | 説明 |
|-----------|-------------|
| `ml.params.MM` | 行列積 (線形層) |
| `ml.params.Bias` | バイアス加算層 |
| `ml.params.Linear` | MM + Bias の組み合わせ (ヘルパー) |
| `ml.funcs.ReLU` | ReLU 活性化関数 |
| `ml.funcs.Sigmoid` | Sigmoid 活性化関数 |
| `ml.funcs.Softmax` | Softmax 活性化関数 |
| `ml.funcs.CrossEntropyLoss` | 交差エントロピー損失関数 |
| `ml.funcs.MSELoss` | 平均二乗誤差損失関数 |
| `quantize.funcs.Quantize` | 推論モード時の量子化 |
| `quantize.funcs.Dequantize` | 推論モード時の逆量子化 |
| `quantize.funcs.QReLU` | 量子化対応 ReLU |
| `quantize.params.QuantizedLinear` | int8量子化対応の線形層 |
| `quantize.params.QuantizedMM` | int8量子化対応の行列積 |

## テストの実行

テストスイートを実行するには：

```bash
cargo test --lib
```

## ライセンス

MIT
