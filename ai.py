import torch as t
import math
from torch import nn,functional
import matplotlib.pyplot as plt
import random
import numpy as np
import copy
# X = t.rand([10000, 6])
# y = X.clone() + 1
# print(X, y)
#
# print(X)
# print(y)
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.b=random.randint(0,2)
#         print(self.b)
#         if(self.b==1):
#             self.layer = nn.Sequential(nn.Linear(6,6))
#         else:
#             self.layer=nn.Sequential(nn.Linear(6,6),
#                                      nn.Linear(6,1000),
#                                      nn.Linear(1000,6))
#
#     def forward(self, x):
#             logits = self.layer(x)
#             return logits
#
# model = Net()
# model2=Net()
# print(model)
# print(model2)
#
# loss_fn = nn.MSELoss()
# loss_fn2 = nn.MSELoss()
# optimizer = t.optim.SGD(model.parameters(),lr=0.001)
# optimizer2 = t.optim.SGD(model2.parameters(),lr=0.001)
#
# pl_y=[]
# pl_y_2=[]
# for i in range(10):
#     for j in range(10):
#         pred = model(X[i])
#         pred2 = model2(X[i])
#         loss = loss_fn(pred, y[i])
#         loss2 = loss_fn(pred2, y[i])
#
#     # Backpropagation
#         optimizer.zero_grad()
#         optimizer2.zero_grad()
#         loss.backward()
#         loss2.backward()
#         optimizer.step()
#         optimizer2.step()
#     pl_y.append(loss.item())
#     pl_y_2.append(loss2.item())
# print(loss.item())
# print(loss2.item())
# model.eval()
# model2.eval()
# with t.no_grad():
#     X2 = t.rand([6])
#     print(X2)
#     pred = model(X2)
#     pred2 = model2(X2)
#     print(X2+1,pred,(X2+1)-pred)
#     print(X2+1,pred2,(X2+1)-pred2)
#
# fig,ax=plt.subplots(2,1,figsize=(7, 7))
# ax[0].plot(pl_y)
# ax[0].set_title(model.b)
# ax[1].plot(pl_y_2)
# ax[1].set_title(model2.b)
# plt.show()





def check_game(b: list) -> bool:
    is_over = False
    for i in range(3):
        if ((b[i][0] == b[i][1] == b[i][2] and b[i][0] != 0) or (b[0][i] == b[1][i] == b[2][i] and b[0][i] != 0)):
            is_over = True
    if (b[0][0] == b[1][1] == b[2][2] and b[0][0] != 0 or b[2][0] == b[1][1] == b[0][2] and b[2][0] != 0):
        is_over = True
    return is_over


def transform(g):
    if (g == '1'): return [0, 0]
    if (g == '2'): return [0, 1]
    if (g == '3'): return [0, 2]
    if (g == '4'): return [1, 0]
    if (g == '5'): return [1, 1]
    if (g == '6'): return [1, 2]
    if (g == '7'): return [2, 0]
    if (g == '8'): return [2, 1]
    if (g == '9'): return [2, 2]
    else:
        return None
def print_board():
    global board
    for i in board:
        print(i)

def get_avail_indexces(board):
    av_moves = []
    ind = 0
    for i in board:
        for j in i:
            ind += 1
            if (j == 0):
                av_moves.append(ind)
    return av_moves


def random_move():
    global board
    av_moves=get_avail_indexces(board)
    res=random.choice(av_moves)
    return str(res)

def max_random_search(weight,p2):
    global board
    av_moves=get_avail_indexces()
    state=t.from_numpy(np.array(board,dtype=np.float).flatten())
    m=t.matmul(state,weight)
    action=t.argmax(m)
    if(action.item()<av_moves.__len__()):
        res=av_moves[action.item()]
    else:
        p2['reward']-=100
        res=random.choice(av_moves)
    return str(res)

def markov():
    pass

def iterat():
    global board
    av=get_avail_indexces(board)
    boards=[]
    for i in av:
        print('av ',i)
        bo=copy.deepcopy(board)
        t=transform(str(i))
        bo[t[0]][t[1]]=2
        print(t)
        print(bo)
        boards.append(bo)
    res=0
    dec=False
    for i in boards:
        res+=1
        if(check_game(i)):
            dec=True
            break
    if(not dec):res=random.choice(av)
    print('res',res)
    return str(res)
count=0
def minimax(b,p=1):
    global count
    count+=1
    best_move=''
    av=get_avail_indexces(b)
    over=check_game(b)
    if (av.__len__() == 0 or over):
        if (p == 1 and over):
           return [-1,'']
        elif(p == 2 and over):
            return [1,'']
        else:
            return [0,'']
    if p==1:
        best_value=-100
    else:
        best_value=100
    for i in av:
        board = copy.deepcopy(b)
        c=transform(str(i))
        board[c[0]][c[1]] = 1 if p == 1 else 2
        hv=minimax(board, 2 if p == 1 else 1)[0]
        if p==1 and hv>best_value:
            best_value=hv
            best_move=i
        if p==2 and hv<best_value:
            best_value=hv
            best_move=i
    return [best_value,best_move]


def make_move(p, c):
    global board
    if (p == 0):
        board[c[0]][c[1]] = 1
    else:
        board[c[0]][c[1]] = 2

best_reward=-10000
best_weights=None
def game_start():
    global board, move_num,p1,p2,best_reward,best_weights,noise_scale,count
    game_over = False
    #weight = t.rand([9, 8], dtype=t.double)+noise_scale*t.rand([9,8])
    while (not game_over and move_num < 9):
        if(move_num%2==0):
            v,g=minimax(board,1)
        else:
            # g=max_random_search (best_weights if(best_weights!=None) else weight,p2)
            # g=random_move()
            # g=iterat()
            count=0
            v,g=minimax(board,2)
            print(count)
        make_move(move_num % 2, transform(str(g)))
        print_board()
        game_over = check_game(board)



        move_num += 1
    if (game_over):
        if((move_num-1)%2==0):
            p1['wins']+=1
            p1['reward']+=20
            p2['reward']-=20
        else:
            p2['wins']+=1
            p2['reward'] += 20
            p1['reward'] -= 20
        # print('game over')
    else:
        p1['draws'] += 1
        p2['draws'] += 1
        p1['reward'] += 3
        p2['reward'] += 3
        # print('draw')
    p1['num_games']+=1
    p2['num_games']+=1


    # if(best_reward<=p2['reward']):
    #     best_reward=p2['reward']
    #     best_weights=weight
    #     noise_scale=max(noise_scale/2,1e-4)
    # else:
    #     noise_scale=min(noise_scale*2,2)


p1={'num_games':0,'wins':0,'draws':0,'reward':0,'total_reward':0,'strategy':'random'}
p2={'num_games':0,'wins':0,'draws':0,'reward':0,'total_reward':0,'strategy':'minimax'}
p2_rew=[]
noise_scale=0.01
while(p1['num_games']<30):
    board = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    move_num = 0
    p1['reward']=0
    p2['reward']=0
    game_start()
    p1['total_reward']+=p1['reward']
    p2['total_reward']+=p2['reward']
    print('Эпизод ',p1['num_games'],':','p1:',p1['reward'],', p2:', p2['reward'])
    p2_rew.append(p2['reward'])
print(p1,p2,sep='\n')
print(p2['total_reward']/p2['num_games'])
print(best_weights)
