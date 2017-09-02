# nnboard
a GUI editor for neural network

chainer 向けの GUI エディタです。

![](https://github.com/yos1up/nnboard/blob/master/nnboard_digest.gif)

## 動作環境
* python 3.5.2 以降 （おそらく 3.5 以降なら動きます）
* chainer 2.0.2 以降 （おそらく 1.19 以降なら動きます）

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

`server.py` を Ctrl-C で終了してください。

`index.html` の最下部にある `Shutdown Server` ボタンを押しても、 `server.py` が終了します。

### 作業状態の保存と読み込み

* `index.html` の下の方にある `Download Canvas` リンクを押すと、json ファイルがローカルに保存されます。この中に、構築したネットワークの情報が入っています。

* `Load Canvas` ボタンを押して json ファイルを選択すると、構築したネットワークが復元されます。

* 構築したネットワークとともに、optimizerの設定なども保存されます。

* 学習結果は保存されません・・・


<!-- `server.py` automatically opens `index.html`; in this page you can edit neural networks on GUI.
Press `Shutdown Server` button in the page to shutdown `server.py`. Otherwise `server.py` continues running. -->


## 詳しい使い方
### ネットワーク設計

* 層や結合を編集するときは、canvas にフォーカスが当たった状態にします（canvas内をどこかクリックすれば良いです）。

* `a` キーを押すと、層が作られます(`add`)。作った層は、クリックで選択でき、ドラッグで移動できます。

* ある層(a)を選択中に、`Shift` キーを押しながら別の層(b)をクリックすると、(a)から(b)に結合が生じます。結合もクリックで選択できます。

* `Del` キーを押すと、選択中の層や結合を削除できます。層を削除すると、層にくっついている結合も一緒に削除されます。

* ある単一の層を選択中に、様々な英字キーを押すことで、層のタイプを変更することができます。

    * 対応キー：`b`(batch Normalization),`c`(convolution),`C`(Concat),`e`(experience replay),`f`(full connected),`i`(input),`m`(mean_squared_loss),`o`(other;任意の関数),`p`(pooling),`r`(random),`R`(Reshape),`s`(softmax_cross_entropy),`T`(Transpose),`+`(足し算),`-`(引き算),`*`(掛け算)

    * `Options`から、オプション引数を設定できます。`o`の場合は、任意の関数を設定できます。(lambda式も指定可です。例えば `"func":"lambda x,y:F.softmax_cross_entropy(x,y)"`と書けば、タイプ`s`の層と実質的に同じになります。)
    
    * `Options` は json の書式で書く必要があります。 None は null で指定します。タプルを指定したいときは、(jsの)Array として書きます。例： `"shape":[-1,1,28,28]`

* ある単一の層を選択中に、`Ctrl` キーを押しながら様々な英字キーを押すことで、層の活性化関数を変更することができます。

    * 対応キー：`e`(elu),`i`(id),`l`(leaky_relu),`r`(relu),`s`(sigmoid),`t`(tanh)

    * `Options`から、（これらに限らない）任意の活性化関数に変更できます(lambda式も指定可：例えば `"act":"lambda x:F.relu(x)"` など)。


### 学習

* loss と optimizer を設定する必要があります。

    * 単一の層を選択中に数字キー(`0`-`9`)を押すと、層に「タグ」をつけることができます(層の中に `#0` などと表示されます)。
    
    * 一行目には `optimizee: #0, loss:#4` と書いてあります。これに従って、最適化したい loss の層に `#4` タグを指定します。その loss から計算される勾配に従って最適化したい重みをもつ層に `#0` タグを指定します。
    
    * 同時に4つまで複数のoptimizerを併用できます。
    
    * 複数のoptimizerを交互に動かしたい場合などには、`condition` の指定を行ってください。ここでは「非負整数 x を受け取り、（xイテレーション目にこのoptimizerを動かすか）を返す関数」を指定してください。
    
        * 例：optimizer 0 の condtition が `lambda x: x%6` で、optimizer 1 の condition が `lambda x: not(x%6)` の時、「0を5回」→「1を1回」→「0を5回」→・・・という動かし方になります。
* `Start Learning` ボタンを押すと、学習が始まります。

* 正常に学習が始まると、ボタンの表示が `Quit Learning` に変化します。`Quit Learning` ボタンを押すと、学習が終了し、ボタンの表示が `Start Learning` に戻ります。何らかのエラーが生じて学習が死んだ場合も、ボタンの表示が `Start Learning` に戻ります。

* 学習中の表示の見方

    * 各層の右下に表示されているのは shape です。右上は、現在の層の値のプレビューです。
    
    * 各層をダブルクリックすると、現在の層の値をいつでも可視化することができます
    
    * いい感じに可視化されない場合は、任意の層を使って、いい感じに shape を整形すると良いです。
    
    * 学習中の loss の変化が折れ線グラフで表示されます（google chart API を利用しているため、インターネット接続時のみの機能です）。
    
      * このグラフは15秒ごとに自動更新されます。手動で更新したい場合は `Update graph manually` ボタンを押してください。
      
      * softmax_cross_entropy の層で loss を集計している場合に限り、 accuracy の変化も折れ線グラフで表示されます。
      
* 学習中に任意コードの実行ができます。学習係数を途中で変えたりできます。 `Execute Code` にコードを入力し、`Execute` ボタンを押してください。

    * エラーが出た場合はダイアログで表示されます。
    
* 学習中にハイパーパラメータをスライダーで調節できます。 `Tuning Slider` の欄に調節したいハイパーパラメータの変数を入力し、 `GetValue` ボタンを押すと、スライダーに現在の値がセットされ、スライダーが操作可能となります。この状態でスライダーを操作すると、動的にハイパーパラメータの値を変更できます。
    
### 構成

* `index.html`：編集画面(GUI)

* `server.py`：chainerでニューラルネットの計算を行うサーバー

### FAQ

* `server.py` 起動時に `Address already in use` などと表示されて起動できない。

    * 以下を確かめてみてください：
    
    * `server.py` がバックグラウンドで起動したままになっている。 → `index.html` を（手動で）開いて、 `Shutdown Server` ボタンを押せば終了できます。
    
    * それ以外の何らかのプログラムが `localhost:8000` を使用している。 → 通信に使用するポート番号を変更しましょう。例えば、ポート 12345 番を使いたい場合は、サーバを `server.py -p 12345` で起動し、 `index.html` の最下部にある `Address of Server` を `http://localhost:12345` と変更してください。

* テストエラー見たい

    * 現状、そのような機能はありません
    
* 学習中にネットワークいじったらどうなるの

    * 現状、どうにもなりません。 ← 部分的に、ネットワークを学習中に動的にいじれるようにしました。ネットワークを変更してから、 `SendNetworkInfoToServer` ボタンを押すと、変更したことをサーバーの計算に反映できます。（link 層はいじれません）

* `Start Learning` ボタンを押しても `Quit Learning` に変化しない

    * おそらく一瞬で学習が落ちてます。






