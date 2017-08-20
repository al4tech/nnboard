# 参考：
# http://blog.sarabande.jp/post/81479479934
# http://coreblog.org/ats/stuff/minpy_web/13/03.html 
# http://d.hatena.ne.jp/matasaburou/touch/20151003/1443882557
import numpy as np
import argparse
import urllib
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import cgi

import json
from http.server import BaseHTTPRequestHandler

import threading, sys

from chainer import Chain, ChainList, cuda, gradient_check, Function, Link, optimizers, serializers, utils, Variable, datasets
from chainer import functions as F
from chainer import links as L

def float_if_float(x):
    # np.float系列がJSON serializableでないのを回避するため
    if isinstance(x, list):
        return [float_if_float(i) for i in x]
    return float(x) if isinstance(x, np.float) or isinstance(x, np.float32) or isinstance(x, np.float64) else x
    # もっと綺麗に書けるはず

def filter_dic(dic, filt=None, omit=None):
    ret = {}
    for k,v in dic.items():
        if (filt is None or (k in filt)) and (omit is None or (not k in omit)): ret[k] = v
    return ret

def cov2ovl(cov):
    # covmatをovlmatに
    norm = np.sqrt(np.diag(cov)).reshape(-1, 1)
    return np.maximum(np.minimum(1., cov / np.dot(norm, norm.T)), -1.)

def cov2deg(cov):
    # covmatをdegmatに
    return np.arccos(cov2ovl(cov)) * 180 / np.pi

def cov2maxovl(cov):
    return np.max(np.abs(cov2ovl(cov) * (np.ones(cov.shape) - np.eye(cov.shape[0]))))

def cov2minnorm(cov):
    return np.sqrt(np.min(np.diag(cov)))

class linkset(ChainList):
    def __init__(self, links):
        super(linkset, self).__init__(*links)
# class nnmdl(Chain): # 一つのlinkを複数のchainに登録できないようなので、全体を管理するchainはchainでなくした（chainにする必要もないので）
class nnmdl:
    def __init__(self, net_info):
        self.net_info = net_info
        self.l = {}
        self.h = {}
        self.bs = self.net_info['settings']['bs']
        self.aveintvl = self.net_info['settings']['aveintvl']
        # トポロジカルソートする
        self.torder, self.visited_for_tsort = [], set()
        for k in self.net_info['node'].keys():
            self.recur_for_tsort(k)
        self.torder.reverse()
        # print('self.torder =', self.torder) # トポロジカルソート完了
        # super(nnmdl, self).__init__()        # 必要な Link を用意する
        random_seed = np.random.randint(1000000007)
        for k,v in net_info['node'].items():
            if v['ltype']=='f':
                self.l[k] = L.Linear(None, int(v['opt']['out_channel']), **filter_dic(v['opt'], filt=['nobias', 'initialW', 'initial_bias']))
                # self.add_link(k, self.l[k])
            elif v['ltype']=='r':
                self.l[k] = Sampler(source='random_normal', bs=self.bs, opt=v['opt'], sample_shape=v['opt']['sample_shape'], random_seed=random_seed)
            elif v['ltype']=='i':
                self.l[k] = Sampler(source=v['opt']['source'], bs=self.bs, opt=v['opt'], sample_shape=None, random_seed=random_seed)
            elif v['ltype']=='o':
                if v['opt']['func'][:2] == 'L.':
                    self.l[k] = eval(v['opt']['func']) # optで指定できるのはインスタンス生成時引数でなく__call__時引数とする（インスタンス生成時引数はfuncの中に直接書いて。）
                else:
                    self.l[k] = eval(v['opt']['func']) # Fも同じ扱い
            elif v['ltype']=='b':
                self.l[k] = L.BatchNormalization(v['opt']['size'])
            elif v['ltype']=='e':
                self.l[k] = [] # 溜め込み場としてのリスト
            elif v['ltype'] in ['i','C','s','v','m','p','+','-','*','T']:
                pass
            else:
                print('Not Implemented:', v['ltype'])
                raise NotImplementedError

        # optimizer準備
        self.num_of_opt = 4
        self.optchain = [None] * self.num_of_opt
        self.opt = [None] * self.num_of_opt
        self.lossidx = [[] for i in range(self.num_of_opt)]
        linkflg = False
        for i in range(self.num_of_opt): # optimizer番号のループ
            optiee_tag = int(self.net_info['optiee'][i]) # タグ番号
            links = []
            for j in self.net_info['node'].keys():
                n = self.net_info['node'][j]
                if n['optiflg'][optiee_tag]:
                    if j in self.l and isinstance(self.l[j], Link):
                        links.append(self.l[j])
                        linkflg = True
            if len(links) > 0: # optimizeeがない場合はoptimizerはインスタンス化されないようにしてる。(self.opt[i]はNoneのままとなる)
                self.optchain[i] = linkset(links) # optimizeしたいlinksを束ねたchainを作る (opt.setupが引数にlinkしか受け付けないため...)
                self.opt[i] = eval(self.net_info['opti'][i]) # optimizerのインスタンス取得
                self.opt[i].setup(self.optchain[i])
                # loss調べとく
                loss_tag = int(self.net_info['loss'][i]) # タグ番号
                for j in self.net_info['node'].keys():
                    n = self.net_info['node'][j]
                    if n['optiflg'][loss_tag]:
                        self.lossidx[i].append(j)
                if len(self.lossidx[i])==0: print('WARNING: please specify at least one node as the loss. (optimizer '+str(i)+')')
        if not linkflg: print('WARNING: please specify at least one node as the optimizee.')

        self.lossfrac = [np.zeros(2) for i in range(self.num_of_opt)]
        self.accfrac = [np.zeros(2) for i in range(self.num_of_opt)]
        self.aveloss = None
        self.aveacc = None
        self.update_cnt = 0
    def recur_for_tsort(self, node_id):
        # self.net_infoを元にtsortするときの再帰用 self.visited_for_tsort を更新しつつ self.torder に結果入れてく
        if not(isinstance(node_id, str)): node_id = str(node_id)
        if node_id in self.visited_for_tsort: return
        self.visited_for_tsort.add(node_id)
        n = self.net_info['node'][node_id]
        for i in n['to']:
            to_node_id = str(self.net_info['edge'][str(i)]['post'])
            if not (to_node_id in self.visited_for_tsort):
                self.recur_for_tsort(to_node_id)
        self.torder.append(node_id)
    def __call__(self, mode='train'): # 順計算
        self.acc = {}
        for nid in self.torder:
            n = self.net_info['node'][str(nid)]
            n_ltype = n['ltype']
            if n_ltype in ['r', 'i']:
                self.h[nid] = self.l[nid]() # サンプル取得
            elif n_ltype in ['f', 'c']:
                vs = self.prevars(n, sort=True)
                assert(len(vs)==1)
                self.h[nid] = self.l[nid](vs[0])
            elif n_ltype == 'C':
                vs = self.prevars(n, sort=True)
                typ = n['opt']['type']
                if typ == 'batch_dim':
                    axis = 0
                elif typ == 'channel_dim':
                    axis = 1
                else:
                    raise ValueError
                self.h[nid] = F.concat(vs, axis=axis)
            elif n_ltype in ['+', '*']:
                vs = self.prevars(n)
                self.h[nid] = 0
                for v in vs:
                    if n_ltype == '+':
                        self.h[nid] += v
                    else:
                        self.h[nid] *= v
            elif n_ltype == '-':
                vs = self.prevars(n)
                assert(len(vs)==1)
                self.h[nid] = -vs[0]
            elif n_ltype == 'm':
                vs = self.prevars(n)
                assert(len(vs)==2)
                self.h[nid] = F.mean_squared_error(vs[0], vs[1])
            elif n_ltype == 's':
                vs = self.prevars(n, sort=True)
                assert(len(vs)==2)
                self.h[nid] = F.softmax_cross_entropy(vs[0], vs[1])
                self.acc[nid] = F.accuracy(vs[0], vs[1])
            elif n_ltype == 'd':
                vs = self.prevars(n)
                assert(len(vs)==1)
                self.h[nid] = F.dropout(vs[0], **filter_dic(n['opt'], filt=['ratio']))
            elif n_ltype == 'e':
                # experience replay やってきたサンプルを (リストself.l[nid]に) 溜め込む。そして、同じ数だけランダムに吐き出す。
                # Variableのitem assignmentが無いようなので、以下の「サンプル」は「ミニバッチ」に読み替えて実装することとする。
                vs = self.prevars(n)
                assert(len(vs)==1)
                mem = self.l[nid]
                insiz = 1 # vs[0].data.shape[0] # 入ってきたサンプル数
                maxsiz = n['opt']['size'] # 溜め込む最大サンプル数
                nowsiz = len(mem) # 現在溜め込んでるサンプル数
                rest = maxsiz - nowsiz
                if rest > 0: mem += [vs[0]]
                if rest < insiz:
                    # memに溜め込まれているサンプルをランダムに上書きする
                    mem[np.random.randint(maxsiz)] = vs[0]
                self.h[nid] = mem[np.random.randint(len(mem))] # memからランダムにinsiz個のサンプルを吐き出す
            elif n_ltype == 'v': # value
                typ = n['opt']['type'] if 'type' in n['opt'] else 'np.float32'
                self.h[nid] = Variable(np.array([n['opt']['value']]*self.bs, dtype=np.dtype(typ)))
                # ary = np.array([n['opt']['value']]*self.bs)
                # if ary.dtype in ['int64', 'int']:
                #     ary = ary.astype(np.int32)
                # elif ary.dtype in ['float64', 'float']:
                #     ary = ary.astype(np.float32)                
                # self.h[nid] = Variable(ary)
            elif n_ltype == 'T': # transpose (last 2 dim) それ以外やりたいなら F.transpose でやって
                vs = self.prevars(n)
                assert(len(vs)==1)
                nd = vs[0].data.ndim
                self.h[nid] = F.transpose(vs[0], axes=list(np.arange(nd-2))+[nd-1, nd-2])
                # self.h[nid] = F.transpose(vs[0], **filter_dic(n['opt'], filt=['axes']))
            elif n_ltype == 'o': # 任意の層
                vs = self.prevars(n, sort=True)
                self.h[nid] = eval(n['opt']['func'])(*vs, **filter_dic(n['opt'], omit=['act','func'])) 
                # print(self.update_cnt,'calculated! shape=',self.h[nid].data.shape)

            # あとは活性化
            a = n['opt']['act']
            if a == 'relu':
                self.h[nid] = F.relu(self.h[nid])
            elif a in ['sigm', 'sigmoid']:
                self.h[nid] = F.sigmoid(self.h[nid])
            elif a == 'elu':
                self.h[nid] = F.elu(self.h[nid])
            elif a in ['l_relu', 'leaky_relu']:
                self.h[nid] = F.leaky_relu(self.h[nid])
            elif a == 'tanh':
                self.h[nid] = F.tanh(self.h[nid])
            elif a in ['id', 'identity']:
                pass
            else:
                self.h[nid] = eval(a)(self.h[nid]) # 任意の活性化関数を使える
            # print('nid ==',nid,'self.h[nid].data.shape ==',self.h[nid].data.shape)
    def prevars(self, n, sort=False):
        vs = []
        for i in n['from']:
            pre_nid = self.net_info['edge'][str(i)]['pre']
            if sort:
                vs.append((self.net_info['node'][str(pre_nid)]['x'], self.h[pre_nid]))
            else:
                vs.append(self.h[pre_nid])
        if sort:
            vs.sort()
            return [v[1] for v in vs]
        return vs
    def update(self):
        self.update_cnt += 1
        self() # とりあえず順計算
        # ロスを計算
        self.loss = [0] * self.num_of_opt
        for i in range(self.num_of_opt):
            if not eval(self.net_info['cond'][i])(self.update_cnt): continue # conditionが合致してるかチェック
            for j in self.lossidx[i]: # まずはlossを計算（足し合わせ）accfracも計算
                self.loss[i] += self.h[j]
                if j in self.acc: self.accfrac[i] += np.array([self.acc[j].data, 1])
            if isinstance(self.loss[i], Variable) and self.opt[i] is not None:
                # あとは逆伝播 (TODO: loss指定が前と同じならわざわざ逆伝播し直す必要はないはず。)
                for j in range(self.num_of_opt):
                    if self.opt[j] is not None: self.optchain[j].cleargrads() # TODO: 一体どの範囲をcleargradsすれば十分なのかはよくわかっていない。毎回全部やっとくか、て感じになってる
                self.loss[i].grad = np.ones(self.loss[i].shape, dtype=np.float32)
                self.loss[i].backward()
                self.opt[i].update()
                self.lossfrac[i] += np.array([np.sum(self.loss[i].data), 1])
        if self.update_cnt % self.aveintvl == 0:
            self.aveloss = self.get_aveloss(clear=True)
            self.aveacc = self.get_aveacc(clear=True)
    def get_aveloss(self, clear=False):
        ret = [None for i in range(self.num_of_opt)]
        for i in range(self.num_of_opt):
            if self.lossfrac[i][1] == 0: continue
            ret[i] = self.lossfrac[i][0]/self.lossfrac[i][1]
        if clear: self.lossfrac = [np.zeros(2) for i in range(self.num_of_opt)]          
        return ret
    def get_aveacc(self, clear=False):
        ret = [None for i in range(self.num_of_opt)]
        for i in range(self.num_of_opt):
            if self.accfrac[i][1] == 0: continue
            ret[i] = self.accfrac[i][0]/self.accfrac[i][1]
        if clear: self.accfrac = [np.zeros(2) for i in range(self.num_of_opt)]          
        return ret
    def W(self, idx):
        return self.l[idx].W.data


class Sampler:
    def __init__(self, source, bs, opt, sample_shape, random_seed=None):
        _ = np.random.get_state() # 保存
        if random_seed is not None: np.random.seed(random_seed)
        self.random_state = np.random.get_state()
        np.random.set_state(_) # 復元
        self.source = source
        self.bs = bs
        self.opt = opt
        self.sample_shape = list(sample_shape) if isinstance(sample_shape, tuple) else sample_shape
        if self.source=='random_normal':
            self.sample_num = self.bs
        elif self.source in ['mnist_train_x', 'mnist_train_t', 'mnist_test_x', 'mnist_test_t']:
            # self.dataをロードする
            mnist_train, mnist_test = datasets.get_mnist()
            if self.source == 'mnist_train_x':
                self.data = np.array([d[0] for d in mnist_train], dtype=np.float32)
            if self.source == 'mnist_train_t':
                self.data = np.array([d[1] for d in mnist_train], dtype=np.int32)
            if self.source == 'mnist_test_x':
                self.data = np.array([d[0] for d in mnist_test], dtype=np.float32)
            if self.source == 'mnist_test_t':
                self.data = np.array([d[1] for d in mnist_test], dtype=np.int32)
            self.sample_num = len(self.data)
        else:
            raise NotImplementedError
        self.epoch = 0
        self.sample_cnt = 0
    def __call__(self):
        if self.sample_cnt == 0:
            self.epoch += 1
            if self.source == 'random_normal': # 新しいサンプルを作ろう
                self.data = (np.random.randn(*([self.bs] + self.sample_shape)) * self.opt['sigma'] + self.opt['mu']).astype(np.float32)
            else:
                self.data = self.shuffled(self.data) # 自身の random_state の下でシャッフルするだけ

        ret = Variable(self.data[self.sample_cnt:self.sample_cnt+self.bs])
        self.sample_cnt += self.bs
        if self.sample_cnt >= self.sample_num: self.sample_cnt = 0
        return ret
    def shuffled(self, ary):
        # こいつが持つrandom stateの下でaryをシャッフルしたものを返す
        # ary は np.array とする
        _ = np.random.get_state() # 保存
        np.random.set_state(self.random_state)
        # np.random.shuffle(ary) # TODO: shuffleが同じseedで必ず同じ順列に並べ替える（データ長以外のデータの性質に依存せず）ことを仮定しているので注意！！
        # データ非依存性がやや不安なので、indexをshuffleすることにする。
        perm = np.arange(len(ary))
        np.random.shuffle(perm)
        ret = ary[perm]
        self.random_state = np.random.get_state()
        np.random.set_state(_) # 復元
        return ret

class ComputationThreadManager():
# class ComputationThreadManager(threading.Thread):
    def __init__(self):
        # threading.Thread.__init__(self)
        self.start_event = threading.Event() # 計算を開始させるかのフラグ
        self.stop_event = threading.Event() # 計算を停止させるかのフラグ
        self.exit_event = threading.Event() # スレッドを終了させるイベント（不要？）
        self.mdl = None
        self.value = None
        self.value2 = None
    def target(self):
        """別スレッド"""
        self.computing = False
        # self.net_info = None
        if self.exit_event.is_set(): self.exit_event.clear()
        while not self.exit_event.is_set(): # 別スレッドのメインループ
            if self.start_event.is_set(): # 学習開始時の処理
                self.start_event.clear()
                self.computing = True
                # 計算のための補助情報を self.net_info から計算する
                # トポロジカルソート順、サンプラーのロード、LinkやChain作成など
                # Chain作成
                self.mdl = nnmdl(self.net_info)
            if self.stop_event.is_set(): # 学習終了時の処理
                self.stop_event.clear()
                self.computing = False
            if self.computing: # 学習中の処理
                self.mdl.update()
            else: # 学習してない時の処理
                time.sleep(0.1)
        self.exit_event.clear()
        self.mdl = None
        print('[end of thread]')
        # ここで別スレッド終了
    def start_computing(self, net_info):
        self.net_info = net_info
        self.thread = threading.Thread(target = self.target) # スレッド作成        
        self.thread.start() # スレッド開始！
        self.start_event.set()
        print('[computation thread started.]')
    def stop_computing(self):
        self.stop_event.set()
        self.exit_event.set()
        self.thread.join()    #スレッドが停止するのを待つ
        print('[computation thread stopped.]')
    def get_info(self, params):
        typ = params['type']
        dic = {}
        if self.mdl is not None:
            if typ=='shape':
                for k,v in self.mdl.h.items():
                    dic[k] = v.data.shape
            elif typ=='weight_summary':
                for k,v in self.mdl.l.items():
                    rec = {}
                    if hasattr(v, 'W') and isinstance(v.W, Variable) and v.W.data is not None:
                        rec['W_shape'] = v.W.data.shape
                        rec['W_norm'] = float_if_float(np.linalg.norm(v.W.data))
                        W_pre = np.dot(v.W.data.T, v.W.data)
                        rec['W_pre_maxovl'] = float_if_float(cov2maxovl(W_pre))
                        rec['W_pre_minnorm'] = float_if_float(cov2minnorm(W_pre))
                        W_post = np.dot(v.W.data, v.W.data.T)
                        rec['W_post_maxovl'] = float_if_float(cov2maxovl(W_post))
                        rec['W_post_minnorm'] = float_if_float(cov2minnorm(W_post))
                    if hasattr(v, 'b') and isinstance(v.b, Variable):
                        rec['b_shape'] = v.b.data.shape
                        rec['b_norm'] = float_if_float(np.linalg.norm(v.b.data))
                    dic[k] = rec
            elif typ=='learning_status':
                dic = {'aveloss': float_if_float(self.mdl.aveloss), 'aveacc': float_if_float(self.mdl.aveacc),'update_cnt': self.mdl.update_cnt, 'thread_alive': self.thread.is_alive()}
            elif typ=='image_sample':
                # image型なノードすべてから現在のイメージを送り返す ←image型に限らずにした！
                for k,v in self.mdl.h.items():
                    if np.ndim(v.data) >= 1: # もともと ==4 にしてた
                        dic[k] = [v.data[0].tolist()]
                        if v.data.shape[0]>1: # もう一個送ってやる
                            dic[k].append(v.data[1].tolist())
            elif typ=='activation_detail':
                # nid番のノードの現在のactivationを丸ごと送り返す
                v = self.mdl.h[params['id']]
                if np.ndim(v.data)==4:
                    dic = v.data.tolist()
                else:
                    dic = v.data.tolist()


        ret = json.dumps(dic)
        return ret
    def exec(self, com):
        try:
            exec(com)
            ret = 'executed successfully'
        except:
            ex, ms, tb = sys.exc_info()
            ret = '[Error]\n' + str(ex) + '\n' + str(ms)
        return ret



class MyHandler(BaseHTTPRequestHandler):
    def __init__(self, *initargs):
        # リクエスト来るたびにここから毎回実行されることに注意！！！
        super(BaseHTTPRequestHandler, self).__init__(*initargs)
    def do_GET(self):
        i=self.path.rfind('?')
        if i>=0:
            path, query=self.path[:i], self.path[i+1:]
        else:
            path=self.path
            query=''
        unquoted_query = urllib.parse.unquote(query)

        body = (str(np.random.randint(1000000007))+' Do nothing.<br>').encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*') # CORS解決
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        content_len = int(self.headers.get('content-length'))
        requestBody = self.rfile.read(content_len).decode('UTF-8')
        # print('requestBody=' + requestBody)
        jsonData = json.loads(requestBody)
        # print('**JSON**')
        # print(json.dumps(jsonData, sort_keys=False, indent=4, separators={',', ':'}))
        body = ''

        com = jsonData['command']
        if com == 'set':
            ctm.start_computing(net_info = jsonData['data'])
        elif com == 'getinfo':
            body = ctm.get_info(jsonData['params']) # 'data'でもいいんでは 中身 typeだけだし
        elif com == 'exec': # 文字通りexecする
            body = ctm.exec(jsonData['data']) # 文字列そのまま送って
        elif com == 'stop':
            ctm.stop_computing()
        elif com == 'shutdown':
            global httpd
            threading.Thread(target=httpd.shutdown).start() # 他のスレッドからじゃないと殺せないようだ。
        else:
            body = 'unknown command. computation thread is '+('alive' if ctm.thread.is_alive() else 'dead')
        body = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.send_header('Access-Control-Allow-Origin', '*') # CORS解決
        self.end_headers()
        self.wfile.write(body)


# index.htmlを開く
import webbrowser, os
webbrowser.open('file:///' + os.path.abspath(".") + '/index.html')

parser = argparse.ArgumentParser(description='server')
# parser.add_argument('-p', '--password',
#                default='', help='password to connect to sql')
commandargs = parser.parse_args()


ctm = ComputationThreadManager() # 計算用スレッドの開始
host = 'localhost'
port = 8000
httpd = HTTPServer((host, port), MyHandler)
print('serving at port', port)
httpd.serve_forever()