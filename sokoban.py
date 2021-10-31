import sys
import pygame
import numpy as np
import time
from queue import PriorityQueue
from types import ModuleType, FunctionType
from gc import get_referents


BLACKLIST = type, ModuleType, FunctionType

def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size
def transferToGameState(layout):
    """Chuyển test case thành ma trận 2 chiều để tính toán"""
    layout = [x.replace('\n', '') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ':
                layout[irow][icol] = 0  # ô trống
            elif layout[irow][icol] == '#':
                layout[irow][icol] = 1  # tường
            elif layout[irow][icol] == '&':
                layout[irow][icol] = 2  # người chơi
            elif layout[irow][icol] == 'B':
                layout[irow][icol] = 3  # thùng
            elif layout[irow][icol] == '.':
                layout[irow][icol] = 4  # đích
            elif layout[irow][icol] == 'X':
                layout[irow][icol] = 5  # thùng được đặt vào đích
            elif layout[irow][icol] == 'Y':
                layout[irow][icol] = 6  # human on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)])
    return np.array(layout)


def PosOfPlayer(gameState):
    return tuple(np.argwhere((gameState == 2) | (gameState == 6))[0])  # vd (2, 2)


def PosOfBoxes(gameState):
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5)))  # vd: ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))


def PosOfWalls(gameState):
    return tuple(tuple(x) for x in np.argwhere(gameState == 1))  


def PosOfGoals(gameState):
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5) | (gameState == 6)))  


def isEndState(posBox):
    return sorted(posBox) == sorted(posGoals)


def isLegalAction(action, posPlayer, posBox):
    """check tường"""
    xPlayer = posPlayer[0]
    yPlayer = posPlayer[1]
    if action[-1].isupper():  # push move
        # check vị trí mới của box
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]

    if (x1, y1) not in posBox + posWalls:
        return True
    else:
        return False


def legalActions(posPlayer, posBox):
    """ 
        xét tất cả các trạng thái có thể có của neighbor
        array bao gôm vector của hướng di chuyển và trạng thái là push hay không push
    """
    allActions = [[-1, 0, 'u', 'U'], [1, 0, 'd', 'D'],
                  [0, -1, 'l', 'L'], [0, 1, 'r', 'R']]
    legalActions = []
    for action in allActions:
        xNew, yNew = posPlayer[0] + action[0], posPlayer[1] + action[1]
        if (xNew, yNew) in posBox:  # nếu bước tiếp theo trùng với vị trí box bất kỳ thì đó là hành động đẩy
            action.pop(2)  # hành động đẩy -> bỏ little letter
        else:
            action.pop(-1)  # bỏ upper letter

        # check xem vị trí mới có chạm tường
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else:
            continue

    # vd: ((0, -1, 'l'), (0, 1, 'R'))
    neighbors = tuple(tuple(x) for x in legalActions)
    return neighbors


def updateState(posPlayer, posBox, action):
    """Trả về game state cập nhật sau mỗi bước di chuyển"""

    xPlayer, yPlayer = posPlayer # tọa độ của player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # cập nhật tọa độ mới của player theo action 

    # kiểm tra xem action có phải là push hay ko, nếu phải:
    if action[-1].isupper():        
        posBox = [list(x) for x in posBox]  # chuyển vị trí các box thành list 2 chiều
        posBox.remove(newPosPlayer)         # xóa vị trí của box bị push trong list
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]]) # cập nhật tọa độ mới cho nó

    # định dạng và return tọa độ mới của player, các box    
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox


def isFailed(posBox):
    """hàm này sử dụng để check có bị rới vào deadlock hay không"""
    """
        sử dụng một mẫu trạng thái chung
        vd:
        rotatePattern[0] có thứ tự các vị trí xung quanh là:
            0 1 2
            3 4 5
            6 7 8
        rotatePattern[1] xoay trái :
            2 5 8
            1 4 7
            0 3 6
        tương tự ta được dãy số thứ tự cho các vị trí xung quanh
    """
    rotatePattern = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6],
                     [8, 7, 6, 5, 4, 3, 2, 1, 0],
                     [6, 3, 0, 7, 4, 1, 8, 5, 2]]
    # lật ngược lại
    flipPattern = [[2, 1, 0, 5, 4, 3, 8, 7, 6],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8],
                   [6, 7, 8, 3, 4, 5, 0, 1, 2],
                   [8, 5, 2, 7, 4, 1, 6, 3, 0]]

    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            """
                board gồm các vị trí xung quanh của box hiện tại
                vd:        box[0] - 1:box[1] - 1 | box[0] - 1:box[1] | box[0] - 1 :box[1] + 1
                           ------------------------------------------------------------------
                           box[0]    :box[1] - 1 | box[0]    :box[1] | box[0]     :box[1] + 1
                           ------------------------------------------------------------------
                           box[0] + 1:box[1] - 1 | box[0] + 1:box[1] | box[0] + 1 :box[1] + 1

                => 
            """
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                     (box[0],     box[1] - 1), (box[0],     box[1]), (box[0],     box[1] + 1),
                     (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]

            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                """            #                         
                              [B] #
                                                           
                """
                if newBoard[1] in posWalls and newBoard[5] in posWalls:
                    return True
                #               B  #
                 #             [B] #
                 #               
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls:
                    return True
                #                B  #
                #               [B] B
                #
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox:
                    return True

                ##            B  B
                #            [B] B

                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox:
                    return True
                ##            B  #
                ##         # [B] 
                ##            

                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls:
                    return True
    return False


"""Hiện thực giải thuật"""


def breadthFirstSearch():
    """Hiện thực giải thuật tìm kiếm theo chiều rộng"""
    
    # Khởi tạo trạng thái bạn đầu (init state) gồm tọa độ player và list_box theo dạng tuple: 
    # ( (xPlayer,yPlayer), ( (xBox_1,yBox_1),(xBox_1,yBox_1), ... ) ) 
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)
    startState = (beginPlayer, beginBox) # e.g. ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))

    queue = [[startState]]  # lưu states
    actions = [[0]]         # lưu actions (gồm move và push)
    visitedStates = set()    # lưu các state đã duyệt

    # maxSize = getsize(queue) + getsize(actions) + getsize(visitedStates)
    
    while queue:
        node = queue.pop(0)             # lấy 1 state ra để duyệt
        node_action = actions.pop(0)    # list_actions

        # nếu state hiện lại là goal state thì dừng và in ra actions
        if isEndState(node[-1][-1]):
            # print("maxSize:", maxSize)
            return node_action[1:]
        
        # nếu state hiện tại chưa duyệt qua (chưa có trong visitedStates):
        if node[-1] not in visitedStates:
            visitedStates.add(node[-1])  # thêm state hiện tại vào visitedStates
            for action in legalActions(node[-1][0], node[-1][1]): # lấy list các legal action của state này từ hàm legalActions
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # với mỗi item từ list trên, trả về cặp (tọa độ player, tọa độ list_box) tương ứng
                if isFailed(newPosBox): # kiểm tra các tọa độ trong list_box bằng hàm isFailed
                    continue            # trường hợp deadlock thì bỏ qua, nếu không:
                queue.append(node + [(newPosPlayer, newPosBox)])    # thêm state mới này vào queue
                actions.append(node_action + [action[-1]])          # thêm action tương ứng vào list_actions
                
                # maxSize = max(getsize(queue) + getsize(actions) + getsize(visitedStates), maxSize)        # tính toán bộ nhớ tiêu tốn


def heuristic(posBox):
    """Hàm heuristic để tính toán khoảng cách giữa các box còn lại và đích"""
    distance = 0
    completes = set(posGoals) & set(posBox)  # lấy ra những vị trí trùng nhau
    # vd : ((2,4),(4,5),(5,3)) & ((2,3),(4,5),(5,3)) = ((4,5),(5,3))
    # lấy ra những vị trí chưa trùng
    sortposBox = list(set(posBox).difference(completes))
    sortposGoals = list(set(posGoals).difference(completes))   
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + \
            (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance


def cost(actions):
    """Hàm chi phí"""
    # return len([x for x in actions if x.islower()])
    count = 0
    for action in actions:
        if action.islower():
            count += 1

    return count

def aStarSearch():
    """Hiện thực giải thuật tìm kiếm bằng thuật toán A*"""
    count = 0
    beginBox = PosOfBoxes(gameState)  # lấy box list
    beginPlayer = PosOfPlayer(gameState)  # lấy player position

    # 1 state gồm (posplayer, boxlist)
    start_state = (beginPlayer, beginBox)  # ghép 2 tuple thành 1 tuple lớn

    # vì chưa xuất phát nên đặt startstate làm neighbor đầu tiên
    readyqueue = PriorityQueue()
    readyqueue.put((heuristic(beginBox),count, [start_state]))

    visited = [] 

    actions = PriorityQueue()
    actions.put((heuristic(start_state[1]), count, [0])) ## 0, u, d, l, r, U, D,...

    # maxSize = getsize(neighbor) + getsize(visited) + getsize(actions)
    # print(maxSize)

    while readyqueue:
        # xét tại node có cost + heuristic thấp nhất
        node = readyqueue.get()[2]  # [((posplayer) , (posbox))]
        # sử dụng node_action để lưu trạng thái di chuyển của người chơi
        node_action = actions.get()[2]

        # kiểm tra các box đã đến vị trí hay chưa
        if isEndState(node[-1][-1]):  # node[-1][-1] = (posbox)
            # print("maxSize:", maxSize)
            return node_action[1:]
        
        # check visited, nếu node này đã đi qua thì bỏ qua
        # nếu chưa thì duyết các node neighbor của node này
        if node[-1] not in visited:
            visited.append(node[-1])
        
            # đếm những lần di chuyển mà k push
            # ta không nên tính push vì nếu tính thì độ ưu tiên của push và không push là ngang nhau
            Cost = cost(node_action[1:])
            # check các node neighbor hợp lệ
        
            for action in legalActions(node[-1][0], node[-1][1]):
                count += 1
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                Heuristic = heuristic(newPosBox)
                readyqueue.put((Heuristic + Cost, count, node + [(newPosPlayer, newPosBox)]))
                actions.put((Heuristic + Cost, count, node_action + [action[-1]]))

            # maxSize = max(getsize(neighbor) + getsize(actions) + getsize(visited), maxSize) # tính toán bộ nhớ tiêu tốn


"""Đọc input từ terminal"""
def readCommand(argv):
    layout = []
    f = open("sokobanLevels/"+ str(argv[0]) , "r")
    for i in f:
        layout.append(i)
    return layout


def runwithPygame(path):
    """Chuyển đổi cây trạng thái thành giao diện đồ họa mô phỏng các bước di chuyển"""
    pygame.init()

    beginBox = PosOfBoxes(gameState)  # lấy box list
    beginPlayer = PosOfPlayer(gameState)  # lấy player position
    boxList = list(list(i) for i in beginBox)
    Goals = list(list(i) for i in posGoals)
    playerPos = list(beginPlayer)
    maxwidth = 0
    maxheight = 0
    for wall in posWalls:
        if wall[0] > maxheight:
            maxheight = wall[0]
        if wall[1] > maxwidth:
            maxwidth = wall[1]
    game_over = 0
    fixed_size = 32
    maxwidth *= fixed_size
    maxheight *= fixed_size
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((maxwidth + fixed_size, maxheight + fixed_size))
    pygame.display.set_caption("SOKOBAN SOLUTION")

    player = pygame.image.load('img/character.png')
    box = pygame.image.load('img/box-1.png')
    box2 = pygame.image.load('img/box-2.png')
    wall = pygame.image.load('img/wall.png')
    destination = pygame.image.load('img/tracking.png')

    count = 0
    number_step = len(path)
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        push_flag = False
        status = None
        if count < number_step:
            if (path[count] == 'U'):
                playerPos[0] -= 1
                push_flag = True
                status = 'u'
            elif (path[count] == 'u'):
                playerPos[0] -= 1
            elif (path[count] == 'D'):
                playerPos[0] += 1
                push_flag = True
                status = 'd'
            elif (path[count] == 'd'):
                playerPos[0] += 1
            elif (path[count] == 'L'):
                playerPos[1] -= 1
                push_flag = True
                status = 'l'
            elif (path[count] == 'l'):
                playerPos[1] -= 1
            elif (path[count] == 'R'):
                playerPos[1] += 1
                push_flag = True
                status = 'r'
            elif (path[count] == 'r'):
                playerPos[1] += 1
        if push_flag == True:
            for i in boxList:
                if i == playerPos:
                    if status == 'u':
                        i[0] -= 1
                    elif status == 'd':
                        i[0] += 1
                    elif status == 'l':
                        i[1] -= 1
                    else:
                        i[1] += 1

        screen.fill((160,160,160))
        for i in posWalls:
            screen.blit(wall, (i[1]*fixed_size, i[0]*fixed_size))
        for i in Goals:
            screen.blit(destination, (i[1]*fixed_size, i[0]*fixed_size))
        for i in boxList:
            if i not in Goals:
                screen.blit(box, (i[1]*fixed_size, i[0]*fixed_size))
            else:
                screen.blit(box2, (i[1]*fixed_size, i[0]*fixed_size))

        screen.blit(player, (playerPos[1]*fixed_size, playerPos[0]*fixed_size))

        if count < number_step:
            count += 1

        clock.tick(2)
        pygame.display.update()


if __name__ == '__main__':
    layout = readCommand(sys.argv[1:])
    gameState = transferToGameState(layout)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    path = ()
    
    print("1. Astar")
    print("2. BrFS")
    method = int(input("Choose algorithm: "))
    time_start = time.time()
    if method == 1 :
        path = aStarSearch()
    elif method == 2:
        path = breadthFirstSearch()
    time_end = time.time()
    print('Runtime of is: %.2f seconds.' % (time_end-time_start))

    runwithPygame(path)

