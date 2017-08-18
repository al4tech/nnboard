# nnboard
a GUI editor for neural network

chainer 向けの GUI エディタです。

## 使い方
- ダウンロード
```
git clone https://github.com/yos1up/nnboard.git
```

- 実行
```
python server.py
```
`server.py` を実行すると、自動的に `index.html` が開きます。このページ上で、ニューラルネットワークをさくさく設計することができます。
終了するときは、ページの最下部にある `Shutdown Server` ボタンを押してください。すると、 `server.py` がシャットダウンします。

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
    - 対応キー：`b,c,C,e,f,i,n,o,p,r,R,s,T,+,-,*`
- ある単一の層を選択中に、`Ctrl` キーを押しながら様々な英字キーを押すことで、層の活性化関数を変更することができます。
    - 対応キー：`e,i,l,r,s,t`

    - 対応キーの説明書きます

### 学習


### 構成
- `index.html`：編集画面(GUI)
- `server.py`：chainerでニューラルネットの計算を行うサーバー
