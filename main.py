from datetime import datetime
import eel
from sqlalchemy import create_engine,MetaData,Table,String,Column, Integer,DATE

board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
move_num = 0
moves_record=''

engine=create_engine('sqlite:///replays.db')
meta=MetaData()

@eel.expose
def get_move_num():
    global move_num
    return move_num


def make_move(p, c):
    global board
    if (p == 0):
        board[c[0]][c[1]] = 1
    else:
        board[c[0]][c[1]] = 2


def check_game(b: list) -> bool:
    is_over = False
    for i in range(3):
        if ((b[i][0] == b[i][1] == b[i][2] and b[i][0] != 0) or (b[0][i] == b[1][i] == b[2][i] and b[0][i] != 0)):
            is_over = True
    if (b[0][0] == b[1][1] == b[2][2] and b[0][0] != 0 or b[2][0] == b[1][1] == b[0][2] and b[2][0] != 0):
        is_over = True
        ('here 2')
    return is_over


def transform(g):
    global moves_record
    moves_record+=' '+g
    if (g == '1'): return [0, 0]
    if (g == '2'): return [0, 1]
    if (g == '3'): return [0, 2]
    if (g == '4'): return [1, 0]
    if (g == '5'): return [1, 1]
    if (g == '6'): return [1, 2]
    if (g == '7'): return [2, 0]
    if (g == '8'): return [2, 1]
    if (g == '9'):
        return [2, 2]
    else:
        return None


def game_start():
    global board, move_num,moves_record
    game_over = False
    prev = 0
    while (not game_over and move_num < 9):
        print('here')
        res = True
        g = 0

        while (res):
            print('cycle')
            g = eel.f()()
            eel.sleep(0.1)
            print(g, prev)
            if (g != prev):
                prev = g
                res = False
        make_move(move_num % 2, transform(g))
        print(board)
        game_over = check_game(board)
        move_num += 1
    if (game_over):
        print('game over')
    else:
        print('draw')
    print(moves_record)
    add_to_db(moves_record)



def add_to_db(s:str):
    global engine,meta
    da=datetime.utcnow()
    engine.execute(f'INSERT INTO Replays(moves,date) VALUES (:moves,:date)',{'moves':s,'date':da})
    rs=engine.execute('SELECT * FROM Replays')
    for row in rs:
        print(row)



if __name__ == '__main__':
    eel.init('static')
    eel.spawn(game_start)
    eel.start('index.html', size=(420, 320))
