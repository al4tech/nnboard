# nnboard
a GUI editor for neural network

chainer 向けの GUI エディタです。

## 使い方
### ダウンロード
```
git clone https://github.com/yos1up/nnboard.git
cd nnboard
```

### 実行
```
python server.py
```
`server.py` を実行すると、自動的に `index.html` が開きます。このページ上で、ニューラルネットワークをさくさく設計することができます。

### 終了
`index.html` の最下部にある `Shutdown Server` ボタンを押してください。すると、 `server.py` がシャットダウンします。

### 作業状態の保存と読み込み

- `index.html` の下の方にある `Download Canvas` リンクを押すと、json ファイルがローカルに保存されます。この中に、構築したネットワークの情報が入っています。

- `Load Canvas` ボタンを押して json ファイルを選択すると、構築したネットワークが復元されます。

- 構築したネットワークとともに、optimizerの設定なども保存されます。

- 学習結果は保存されません・・・


<!-- `server.py` automatically opens `index.html`; in this page you can edit neural networks on GUI.
Press `Shutdown Server` button in the page to shutdown `server.py`. Otherwise `server.py` continues running. -->

## 動作環境
- python 3.5.2 以降 （おそらく 3.5 以降なら動きます）
- chainer 2.0.2 以降 （おそらく 1.19 以降なら動きます）


## 詳しい使い方
### ネットワーク設計

- 層や結合を編集するときは、canvas にフォーカスが当たった状態にします（canvas内をどこかクリックすれば良いです）。

- `a` キーを押すと、層が作られます(`add`)。作った層は、クリックで選択でき、ドラッグで移動できます。

- ある層(a)を選択中に、`Shift` キーを押しながら別の層(b)をクリックすると、(a)から(b)に結合が生じます。結合もクリックで選択できます。

- `Del` キーを押すと、選択中の層や結合を削除できます。層を削除すると、層にくっついている結合も一緒に削除されます。

- ある単一の層を選択中に、様々な英字キーを押すことで、層のタイプを変更することができます。

    - 対応キー：`b`(batch Normalization),`c`(convolution),`C`(Concat),`e`(experience replay),`f`(full connected),`i`(input),`m`(mean_squared_loss),`o`(other;任意の関数),`p`(pooling),`r`(random),`R`(Reshape),`s`(softmax_cross_entropy),`T`(Transpose),`+`(足し算),`-`(引き算),`*`(掛け算)

    - `Options`から、オプション引数を設定できます。`o`の場合は、任意の関数を設定できます(lambda式も可)。

- ある単一の層を選択中に、`Ctrl` キーを押しながら様々な英字キーを押すことで、層の活性化関数を変更することができます。

    - 対応キー：`e`(elu),`i`(id),`l`(leaky_relu),`r`(relu),`s`(sigmoid),`t`(tanh)

    - `Options`から、（これらに限らない）任意の活性化関数に変更できます(lambda式も可)。


### 学習


### 構成
- `index.html`：編集画面(GUI)
- `server.py`：chainerでニューラルネットの計算を行うサーバー

### FAQ

- 起動時に `Address already in use` などと表示される

    - 以下を確かめてみてください：

    - `server.py` が前回から起動したままになっている。 → `index.html` を（手動で）開いてから、下の手順でシャットダウンしてください。

    - それ以外の何らかのプログラムが `localhost:8000` を使用している。 → `server.py` が使用するポート番号を変更しましょう。

        - 現状 `server.py`　と `index.html` の中にポート番号がハードコードされてますが、いずれ修正可能にしたいと思います
