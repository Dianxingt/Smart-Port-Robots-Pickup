import sys
import random

n = 200
robot_num = 10
berth_num = 10
N = 210


class Robot:
    def __init__(self, startX=0, startY=0, goods=0, status=0, mbx=0, mby=0):
        self.x = startX
        self.y = startY
        self.goods = goods
        self.status = status
        self.mbx = mbx
        self.mby = mby


robot = [Robot() for _ in range(robot_num + 10)]


class Berth:
    def __init__(self, x=0, y=0, transport_time=0, loading_speed=0):
        self.x = x
        self.y = y
        self.transport_time = transport_time
        self.loading_speed = loading_speed

        self.dist = [[]] # 该港口到每个节点的距离
        self.prev = [[]] # 到每个节点最短路径的前继节点


berth = [Berth() for _ in range(berth_num + 10)]


class Boat:
    def __init__(self, num=0, pos=0, status=0):
        self.num = num
        self.pos = pos
        self.status = status


boat = [Boat() for _ in range(10)]

money = 0
boat_capacity = 0
id = 0
ch = []
gds = [[0 for _ in range(N)] for _ in range(N)]


def Init():
    for i in range(0, n):
        line = input()
        ch.append([c for c in line.split(sep=" ")])
    for i in range(berth_num):
        line = input()
        berth_list = [int(c) for c in line.split(sep=" ")]
        id = berth_list[0]
        berth[id].x = berth_list[1]
        berth[id].y = berth_list[2]
        berth[id].transport_time = berth_list[3]
        berth[id].loading_speed = berth_list[4]
        berth[id].dist, berth[id].prev = BfsShortestPath((berth_list[1], berth_list[2]))

    boat_capacity = int(input())
    okk = input()
    print("OK")
    sys.stdout.flush()


remaining_time = [[0 for _ in range(N)] for _ in range(N)]
def Input():
    id, money = map(int, input().split(" "))
    num = int(input())
    for i in range(num):
        x, y, val = map(int, input().split())
        gds[x][y] = val
        remaining_time[x][y] = 1000
    # 调整货物剩余存活时间
    for i in range(N):
        for j in range(N):
            if remaining_time[i][j] > 0:
                remaining_time[i][j] -= 1
    # 读取机器人状态
    for i in range(robot_num):
        robot[i].goods, robot[i].x, robot[i].y, robot[i].status = map(int, input().split())
    for i in range(5):
        boat[i].status, boat[i].pos = map(int, input().split())
    okk = input()
    return id


def BfsShortestPath(start):
    rows, cols = N, N
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
    distance_matrix = [[-1 for _ in range(cols)] for _ in range(rows)]  # 初始化距离矩阵
    predecessors = [[None for _ in range(cols)] for _ in range(rows)]  # 初始化前继节点矩阵

    # 检查起点是否有效
    if ch[start[0]][start[1]] == '#':
        return distance_matrix

    queue = [(start[0], start[1], 0)]  # 使用列表模拟队列，元素格式为(row, col, distance)
    queue_head = 0  # 队列头部的索引
    distance_matrix[start[0]][start[1]] = 0

    while queue_head < len(queue):
        r, c, d = queue[queue_head]
        queue_head += 1  # 出队操作

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (not (ch[nr][nc] == '#' or ch[nr][nc] == '*')) and distance_matrix[nr][nc] == -1:
                distance_matrix[nr][nc] = d + 1
                predecessors[nr][nc] = (r, c)  # 记录前继节点
                queue.append((nr, nc, d + 1))  # 入队操作

    return distance_matrix, predecessors


def reconstruct_path(predecessors, start, end):
    if predecessors[end[0]][end[1]] is None:
        return []  # 如果终点没有前继节点，说明无法从起点到达终点

    path = [end]
    while path[-1] != start:
        path.append(predecessors[path[-1][0]][path[-1][1]])
    path.reverse()  # 反转路径，因为我们是从终点回溯到起点的
    return path


def SelectNextGoal(r):
    """Select next good to go to.

    Args:
        r (Robot): The robot to find the next goal.
        fromBerth (int): The berth the robot is from. If it's -1, it means its not in berth.

    Returns:
        direction (int): The direction to go to. If it's 4, no move.

    """
    start = (r.x, r.y)
    target_pos = (r.x, r.y)
    dist, pred = BfsShortestPath(start)
    max_value_per_unit_time = 0
    for i in range(N):
        for j in range(N):
            d = dist[i][j]
            # 如果在货物剩余存在时间内无法到达，就直接抛弃
            if d >= remaining_time[i][j]:
                continue
            # 如果当前节点更优秀，则更新节点
            if gds[i][j] / d > max_value_per_unit_time:
                max_value_per_unit_time = gds[i][j] / d
                target_pos = (i, j)
    if target_pos == start:
        return 4
    else:
        path = reconstruct_path(pred, start, target_pos)
        next_node = path[1]
        dx = next_node[0] - start[0]
        dy = next_node[1] - start[1]
        if dx == 1 and dy == 0:
            return 0
        elif dx == -1 and dy == 0:
            return 1
        elif dx == 0 and dy == -1:
            return 2
        elif dx == 0 and dy == 1:
            return 3





def ScheduleNextStep(r):
    # 如果机器人现在没有载货，则选择一个目标并朝那个方向前进
    if r.status == 0:
        pass
    # 如果机器人现在有目标，则直接朝目标前进
    else:
        pass

if __name__ == "__main__":
    Init()
    for zhen in range(1, 15001):
        id = Input()
        for i in range(robot_num):
            print("move", i, random.randint(0, 3))
            sys.stdout.flush()
        print("OK")
        sys.stdout.flush()
