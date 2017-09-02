# 層の一覧

* `b` : Batch Normalization

    * バッチ正規化層です。`size` に、出力チャネル数を指定してください。内部では `L.BatchNormalization(size, **kwargs)` がインスタンス化されます。

* `c`: Convolution 2D

    * 畳み込み層です。`out_channel` に、出力チャネル数を指定してください。内部では `L.Convolution2D(None, out_channel, **kwargs)` がインスタンス化されます。

* `C`: Concat

    * Variableを結合します。`type` には `"batch_dim"` （バッチ次元(chainerの場合0次元目)）か `"channel_dim"`（チャネル次元(chainerの場合1次元目)）を指定してください。内部では `F.concat(list_of_args, axis=<0 or 1>` が呼ばれます。引数の順番は、引数を生み出す層のx座標の昇順となります。

* `d`: Dropout

    * ドロップアウト層です。内部では `F.dropout(arg, **kwargs)` が呼ばれます。

* `e`: experience replay

    * 体験再生 (experience replay) を行う層です。過去に入力されたバッチをいくつか貯めておいて、その中からランダムに1つを選んで出力します。貯めておくバッチの個数を `size` で指定してください。

* `f`: full connected

    * 全結合層です。`out_channel` に、出力チャネル数を指定してください。内部では `L.Linear(None, out_channel, **kwargs)` がインスタンス化されます。

* `i`: input

    * 入力層です。`source` に、データのソースを指定してください（現時点では、`mnist_train_x` `mnist_train_t` `mnist_test_x` `mnist_test_t` のみが選択可能です）。

* `m`: mean squared error

    * 平均2乗和誤差を求める層です。内部では `F.mean_squared_error(arg0, arg1)` が呼ばれます。

* `o`: others

    * 任意の層を設計できます。`func` に指定したオブジェクトが、層が生成されたタイミングでインスタンス化され、学習中、そのインスタンスが引数付きで呼び出され続けます。引数の与えられる順番は、引数を生み出す層のx座標の昇順となります。（ここに記載している層の多くは、この `o` でも実装可能です。）

        * 例1："func":"L.Linear(None, 32)"` と指定すれば、`f` 層で `"out_channel":32` と指定するのと同じになります。

        * 例2："func":"F.relu"` と指定すれば、ReLU活性化のみを行う層となります。

        * 例3：`"func":"lambda x, y:x+2*y"` と指定すれば、x と y を受け取って x + 2*y を出力する層となります。

* `p`: pooling

    * プーリング層です。`type` には、`max` か `average` を指定してください。内部では `F.max_pooling_2d(arg, **kwargs)` または `F.average_pooling_2d(arg, **kwargs)` が呼ばれます。

* `r`: random

    * 乱数を生成する層です。現状、`type` 指定に関わらず、各要素が独立な正規乱数によって生成されます。`sample_shape` に、生成される Variable のサンプル部分の shape（つまり、0次元目のバッチサイズを除いたshape）を指定してください（例：`"shape":[3, 32, 32]`）。`mu` に正規分布の平均、`sigma` に正規分布の標準偏差を指定してください。

* `R`: Reshape

    * shape の変更を行う層です。（現状、`o` 層に `"func":"F.reshape"` と指定したものが出てきます。）`shape` に、変更後の shape を指定してください（例：`"shape":[-1, 1, 28, 28]`）。

* `s`: softmax cross entropy

    * ソフトマックス交差エントロピー誤差を求める層です。内部では `F.softmax_cross_entropy(arg0, arg1)` が呼ばれます。（arg0 と arg1 の順番は、
    ノードの x 座標の小さな方が arg0 となります。）なお、この層は正答率集計の対象となります（裏で `F.accuracy(arg0, arg1)` を呼んでいる）。

* `T`: Transpose

    * Variable を転置するための層です。最後の2つの軸が入れ替わります。（これ以外の軸の転置を行いたい場合は、`o` 層で `"func":"F.transpose"` としてください）

* `v`: value

    * 固定値を出力するための層です。`value` に値を指定してください。`type` には、"float32" "int32" など、型を指定してください（`np.dtype` の引数に渡せる文字列を指定してください）。

* `+`: 和

    * 引数を全て足し合わせる層です。

* `*`: 積

    * 引数を全て掛け合わせる層です。

* `-`: 負号

    * 引数にマイナスをつけます。（引数は 1 つしか受け取れません。引き算記号ではありません。）


# 活性化関数について

* 全ての層には活性化関数がついています。上記の計算を行った後に、活性化関数が適用され、その層の出力となります。

* 活性化関数は、`Ctrl`+{`e`,`i`,`l`,`r`,`s`,`t`} で切り替え可能です。また、Options の `act` を編集すれば、任意のものに変更可能です。









