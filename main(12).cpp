#include <iostream>
#include <queue>
#include <fstream>
#include <set>
#include <vector>
#include <algorithm>
#include <limits>
#include <ctime>
#include <random>
// #include "goalSearch.h"
// #include <bits/stdc++.h>
using namespace std;

const int n = 200;
const int robot_num = 10;
const int berth_num = 10;
const int N = 210;
const int INF = 99999;

struct Robot
{
    int x, y, goods;
    int status;
    int mbx, mby;
    int value;
    int target_berth;
    vector<vector<int> > distance_matrix;
    vector<vector<pair<int, int> > > predecessors;

    Robot() {}
    Robot(int startX, int startY) {
        x = startX;
        y = startY;
    }
}robot[robot_num + 10];

struct Berth
{
    int x;
    int y;
    int transport_time;
    int loading_speed;
    vector<vector<int> > dist; // 从这个港口到其他所有点的距离
    vector<vector<pair<int, int> > > prev; // 重建路径所需的前驱节点

    int num_goods;
    int value;
    int occupied = 0;
    int available = 1;

    Berth(){}
    Berth(int x, int y, int transport_time, int loading_speed) {
        this -> x = x;
        this -> y = y;
        this -> transport_time = transport_time;
        this -> loading_speed = loading_speed;
    }
}berth[berth_num + 10];

struct Boat
{
    int num = 0, pos, status;
    int ready2go = 0;
}boat[10];

int money, boat_capacity, id;
char ch[N][N];
int gds[N][N];

int remaining_time[N][N];
int goal_occupied[N][N];
set<pair<int, int>> grid_occupied; // 正在前往的格子集合
int robot_order[robot_num];
vector<pair<short, short> > robot_goals(robot_num);

int Input()
{
    scanf("%d%d", &id, &money);
    int num;
    scanf("%d", &num);
    for(int i = 1; i <= num; i ++)
    {
        int x, y, val;
        scanf("%d%d%d", &x, &y, &val);
        gds[x][y] = val;
        remaining_time[x][y] = 1000;
    }
    // Adjust the remaining time of goods
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            if(remaining_time[i][j] > 0)
                remaining_time[i][j] -= 1;
    for(int i = 0; i < robot_num; i ++)
    {
        int sts;
        scanf("%d%d%d%d", &robot[i].goods, &robot[i].x, &robot[i].y, &robot[i].status);
    }
    for(int i = 0; i < 5; i ++){
        scanf("%d%d\n", &boat[i].status, &boat[i].pos);
    }

    char okk[100];
    scanf("%s", okk);
    return id;
}

void BfsShortestPathRobot(pair<int, int> start, vector<vector<int> > &distance_matrix, vector<vector<pair<int, int> > > &predecessors) {
    int rows = N, cols = N;
    vector<pair<int, int> > directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // 右、下、左、上
    distance_matrix = vector<vector<int> > (rows, vector<int>(cols, INF));
    predecessors = vector<vector<pair<int, int> > > (rows, vector<pair<int, int> >(cols, {-1, -1}));

    if (ch[start.first][start.second] == '#') {
        return ;
    }

    queue<pair<int, int> > queue; // 只需要存储行和列
    distance_matrix[start.first][start.second] = 0;
    queue.push(start);

    while (!queue.empty()) {
        auto [r, c] = queue.front();
        queue.pop();
        int d = distance_matrix[r][c];

        for (auto& [dr, dc] : directions) {
            int nr = r + dr, nc = c + dc;
            if (0 <= nr && nr < rows && 0 <= nc && nc < cols && ch[nr][nc] != '#' && ch[nr][nc] != '*' && ch[nr][nc] != 'O' && distance_matrix[nr][nc] == INF) {
                distance_matrix[nr][nc] = d + 1;
                predecessors[nr][nc] = {r, c};
                queue.push({nr, nc});
            }
        }
    }
}

void BfsShortestPath(pair<int, int> start, pair<int, int> end, vector<vector<int> > &distance_matrix, vector<vector<pair<int, int> > > &predecessors) {
    int rows = N, cols = N;
    vector<pair<int, int> > directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // 右、下、左、上
    distance_matrix = vector<vector<int> > (rows, vector<int>(cols, INF));
    predecessors = vector<vector<pair<int, int> > > (rows, vector<pair<int, int> >(cols, {-1, -1}));

    if (ch[start.first][start.second] == '#') {
        return ;
    }

    queue<pair<int, int> > queue; // 只需要存储行和列
    distance_matrix[start.first][start.second] = 0;
    queue.push(start);

    while (!queue.empty()) {
        auto [r, c] = queue.front();
        queue.pop();
        int d = distance_matrix[r][c];

        for (auto& [dr, dc] : directions) {
            int nr = r + dr, nc = c + dc;
            if (0 <= nr && nr < rows && 0 <= nc && nc < cols && ch[nr][nc] != '#' && ch[nr][nc] != '*' && ch[nr][nc] != 'O' && distance_matrix[nr][nc] == INF) {
                distance_matrix[nr][nc] = d + 1;
                predecessors[nr][nc] = {r, c};
                if (nr == end.first && nc == end.second) {
                    return ;
                }
                queue.push({nr, nc});
            }
        }
    }
}

pair<vector<vector<int> >, vector<vector<pair<int, int> > > > BfsShortestPathBerth(pair<int, int> start) {
    int rows = N, cols = N;
    vector<pair<int, int> > directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // 右、下、左、上
    vector<vector<int> > distance_matrix(rows, vector<int>(cols, INF));
    vector<vector<pair<int, int> > > predecessors(rows, vector<pair<int, int> >(cols, {-1, -1}));

    if (ch[start.first][start.second] == '#') {
        return {distance_matrix, predecessors};
    }

    queue<pair<int, int> > queue; // 只需要存储行和列
    distance_matrix[start.first][start.second] = 0;
    queue.push(start);

    while (!queue.empty()) {
        auto [r, c] = queue.front();
        queue.pop();
        int d = distance_matrix[r][c];

        for (auto& [dr, dc] : directions) {
            int nr = r + dr, nc = c + dc;
            if (0 <= nr && nr < rows && 0 <= nc && nc < cols && ch[nr][nc] != '#' && ch[nr][nc] != '*' && ch[nr][nc] != 'O' && distance_matrix[nr][nc] == INF) {
                distance_matrix[nr][nc] = d + 1;
                predecessors[nr][nc] = {r, c};
                queue.push({nr, nc});
            }
        }
    }
    return {distance_matrix, predecessors};
}

void Init()
{
    for(int i = 0; i < n; i ++)
        scanf("%s", ch[i]);
    for(int i = 0; i < berth_num; i ++)
    {
        int id;
        scanf("%d", &id);
        scanf("%d%d%d%d", &berth[id].x, &berth[id].y, &berth[id].transport_time, &berth[id].loading_speed);
        auto [dist, prev] = BfsShortestPathBerth({berth[id].x, berth[id].y});
        berth[id].dist = dist;
        berth[id].prev = prev;
    }
    scanf("%d", &boat_capacity);
    char okk[100];
    scanf("%s", okk);
    printf("OK\n");
    fflush(stdout);
}

vector<pair<int, int> > ReconstructPath(pair<int, int> start, pair<int, int> end, const vector<vector<pair<int, int> > >& predecessors) {
    // cerr << 3.051 << endl;
    if (predecessors[end.first][end.second] == make_pair(-1, -1)) {
        // cerr << 3.052 << endl;
        return {}; // 如果终点没有前驱节点，说明无法从起点到达终点
    }
    // cerr << 3.055 << endl;
    vector<pair<int, int> > path;
    for (auto at = end; at != start; at = predecessors[at.first][at.second]) {
        path.push_back(at);
    }
    path.push_back(start); // Make sure to add start point
    reverse(path.begin(), path.end()); // 反转路径
    return path;
}

int GetDirectionalCode(pair<int, int> start, pair<int, int> end) {
    if (end == start) {
        return 4; // no move
    }
    int dx = end.first - start.first;
    int dy = end.second - start.second;

    if (dx == 0 && dy == 1) {
        return 0; // right
    } else if (dx == 0 && dy == -1) {
        return 1; // left
    } else if (dx == -1 && dy == 0) {
        return 2; // up
    } else if (dx == 1 && dy == 0) {
        return 3; // down
    } else {
        return -1; // the given dual nodes is not valid
    }
}

float value[N][N];

void GetMapValue(const vector<vector<int> > & dist) {    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            value[i][j] = -1;
            int d = dist[i][j];
            
            if (d >= remaining_time[i][j] || gds[i][j] == 0) {
                continue;
            }
            
            float estimated_time2berth = INF;
            for (int berth_id = 0; berth_id < berth_num; berth_id++) 
                if (berth[berth_id].dist[i][j] < estimated_time2berth)
                    estimated_time2berth = berth[berth_id].dist[i][j];
            
            float t = d + estimated_time2berth;
            // value[i][j] = gds[i][j] / t;
            value[i][j] = gds[i][j] - 1.2 * t;
        }
    }
}

int GetDirection2Go(const pair<int, int> & start, const pair<int, int> &target_pos, const vector<vector<pair<int, int> > > & pred) {
    grid_occupied.insert(start);
    ch[start.first][start.second] = 'O';

    if (target_pos.first == -1 && target_pos.second == -1) {
        return 4; // No move
    }
    goal_occupied[target_pos.first][target_pos.second] = 1;
    auto path = ReconstructPath(start, target_pos, pred);
    if (path.size() == 0) {
        return 4;
    }
    auto next_node = path[1];
    
    if (grid_occupied.find(next_node) != grid_occupied.end()) {
        return 4; // 如果下一个节点已经有机器人准备走过，就等待
    }
    int direction = GetDirectionalCode(start, next_node);
    if (direction >= 0 && direction <= 3) {
        grid_occupied.insert(next_node); // 标记下一个节点正在前往
        ch[next_node.first][next_node.second] = 'O';
    }
    return direction;
}

struct CompareStructs {
    bool operator()(const pair<pair<short, short>, float>& s1, const pair<pair<short, short>, float>& s2) const {
        // 按照优先级从小到大排序
        return s1.second > s2.second;
    }
};

vector<vector<pair<pair<short, short>, float> > > value_matrix(robot_num, vector<pair<pair<short, short>, float> > (robot_num + 1));
set<pair<short, short> > occupied_pos;
priority_queue<pair<pair<short, short>, float> ,
               vector<pair<pair<short, short>, float> >,
               CompareStructs > minHeap;     // 使用最小堆来存储最大的十个值

void get_k_max_value(const int id) {
    for (short i = 0; i < 200; ++i) {
        for (short j = 0; j < 200; ++j) {
            float v = value[i][j]; // 当前值
            if (v == -1) continue;
            if (minHeap.size() < robot_num) {
                minHeap.push({{i, j}, v});
            } else if (v > minHeap.top().second) {
                minHeap.pop();
                minHeap.push({{i, j}, v});
            }
        }
    }

    // 创建最终数据结构并填充前十个最大值
    int cnt = 0;
    while (!minHeap.empty()) {
        auto item = minHeap.top();
        minHeap.pop();
        value_matrix[id][cnt] = item;
        cnt ++;
    }
    reverse(value_matrix[id].begin(), value_matrix[id].begin() + cnt);
    value_matrix[id][cnt] = {{-1, -1}, -1};
}

/**
 * @brief Selects the best goods to go to. Takes one step in the best direction. 
 * Used for a robot without carrying goods (status = 0).
 * 
 * @param r The robot to find the next good for. It is an object of the Robot class.
 * 
 * @return int The direction to go to. Returns 4 if no move is to be made.
 */
int SelectGoods2Go(int robot_id) {
    // return rand() % 4;
    Robot& r = robot[robot_id];
    pair<int, int> start = {r.x, r.y};
    pair<int, int> target_pos = robot_goals[robot_id];

    r.mbx = target_pos.first;
    r.mby = target_pos.second;
    BfsShortestPath(start, target_pos, r.distance_matrix, r.predecessors); // Assuming this returns a pair of distance matrix and predecessor matrix.
    auto& pred = r.predecessors;
    // 设置机器人i的value_matrix
    get_k_max_value(robot_id);

    return GetDirection2Go(start, target_pos, pred);
}

const float BERTH_TIME_WEIGHT = 0.15;
int SelectBerth2Go(Robot& r) {
    int shortest_distance = INF;
    r.target_berth = -1;
    pair<int, int> start = {r.x, r.y};
    BfsShortestPathRobot(start, r.distance_matrix, r.predecessors); // Assuming this returns a pair of distance matrix and predecessor matrix.
    auto& dist = r.distance_matrix;
    auto& pred = r.predecessors;
    for (int berth_id = 0; berth_id < berth_num; berth_id++) {
        if (berth[berth_id].available == 0) continue;
        float distance2berth = berth[berth_id].dist[r.x][r.y] + BERTH_TIME_WEIGHT * berth[berth_id].transport_time;
        if (distance2berth < shortest_distance) {
            shortest_distance = distance2berth;
            r.target_berth = berth_id;
        }
    }
    // 如果没有港口可以去，原地等待
    grid_occupied.insert(start);
    ch[r.x][r.y] = 'O';

    if (r.target_berth == -1) {
        return -1; // 表示无法移动
    }
    // cerr<< r.target_berth << endl;
    // 获得往该港口走的路径
    vector<pair<int, int>> path = ReconstructPath(start, {berth[r.target_berth].x, berth[r.target_berth].y}, pred);
    if (path.size() < 2) { // 路径长度不足以移动
        return 4; // 原地不动
    }
    // 检查下一个节点是否已经被其他机器人占用
    pair<int, int> next_node = path[1]; // 获取路径的下一个节点
    // cerr << r.x << ' ' << r.y << ' ' << next_node.first << ' ' << next_node.second << endl;
    // for(auto i: path)  cerr << i.first << ' ' << i.second << endl;
    int direction = GetDirectionalCode(start, next_node);
    if (direction >= 0 && direction <= 4) {
        grid_occupied.insert(next_node); // 标记下一个节点正在前往
        ch[next_node.first][next_node.second] = 'O';
    }
    return direction;
}

int RobotInBerth(const Robot& r) {
    const Berth& b = berth[r.target_berth];
    // 检查机器人是否在目标港口的范围内
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (r.x == b.x + i && r.y == b.y + j) {
                return 1; // 机器人在港口范围内
            }
        }
    }
    return 0; // 机器人不在港口范围内
}

/**
 * @brief Schedule the next action taken by the robot.
 *
 * @param id The id of the input robot.
 */
void ScheduleRobotAction(int id) {
    // If the robot reaches the port area, drop the goods
    if (RobotInBerth(robot[id]) == 1 && robot[id].goods == 1) {
        // Unload
        printf("pull %d\n", id);
        fflush(stdout);
        // Count the value of the goods stored at the port
        berth[robot[id].target_berth].value += robot[id].value;
        berth[robot[id].target_berth].num_goods += 1;
        robot[id].goods = 0;
        robot[id].value = 0;
    }
    // If the robot is not carrying goods now, choose a target and move towards it
    if (robot[id].goods == 0) {
        int direction = SelectGoods2Go(id);
        if (direction >= 0 && direction <= 3) {
            printf("move %d %d\n", id, direction);
            fflush(stdout);
        }
    }
    // If the robot has reached the target, pick up the goods and update the status
    if (robot[id].x == robot[id].mbx && robot[id].y == robot[id].mby && robot[id].goods == 0) {
        printf("get %d\n", id);
        fflush(stdout);
        robot[id].value = gds[robot[id].x][robot[id].y];
        gds[robot[id].x][robot[id].y] = 0; // The value becomes 0 after picking up
        robot[id].goods = 1;
    }

    // If the robot has a target, move directly towards the target
    if (robot[id].goods == 1) {
        int direction = SelectBerth2Go(robot[id]);
        // cerr << direction << endl;
        if (direction >= 0 && direction <= 3) {
            printf("move %d %d\n", id, direction);
            fflush(stdout);
        }
    }
}

void ScheduleBoatAction(int id, int zhen) {
    // 如果船当前空闲，就先去找一个有货的港口
    if (boat[id].status == 1) {
        // 如果船现在在虚拟点，也就是刚卖完货，那就找一个有货的去装
        if (boat[id].pos == -1) {
            int max_goods = 0;
            int target_berth = -1;
            for (int i = 0; i < berth_num; ++i) {
                if (berth[i].num_goods > max_goods && berth[i].occupied == 0) {
                    // 选择最多货的
                    max_goods = berth[i].num_goods;
                    target_berth = i;
                }
            }
            if (target_berth != -1) {
                berth[target_berth].occupied = 1;
                boat[id].pos = target_berth;
                printf("ship %d %d\n", id, target_berth);
                fflush(stdout);
            }

        } else {
            // 如果船在港口，卸货并出发
            Berth& bt = berth[boat[id].pos];
            // 如果时间不够，直接出发
            if (bt.available == 0){
                printf("go %d\n", id);
                fflush(stdout);
                bt.occupied = 1;
                // boat[id].status = 0;
                boat[id].num = 0;
                boat[id].pos = -1;
                return ;
            }
            if (bt.num_goods <= 0) {
                if (boat[id].num >= int(0.35 * boat_capacity)) {
                    printf("go %d\n", id);
                    bt.occupied = 0;
                    fflush(stdout);
                    // boat[id].status = 0;
                    boat[id].num = 0;
                    boat[id].pos = -1;
                }
                else {
                    int max_goods = 0;
                    int target_berth = -1;
                    for (int i = 0; i < berth_num; ++i) {
                        if (berth[i].num_goods > max_goods && berth[i].occupied == 0) {
                            // 选择最多货的
                            max_goods = berth[i].num_goods;
                            target_berth = i;
                        }
                    }
                    if (target_berth != -1) {
                        bt.occupied = 0;
                        berth[target_berth].occupied = 1;
                        boat[id].pos = target_berth;
                        printf("ship %d %d\n", id, target_berth);
                        fflush(stdout);
                    }
                }
            }
            else {
                if (boat[id].num >= boat_capacity) {
                    printf("go %d\n", id);
                    bt.occupied = 0;
                    fflush(stdout);
                    // boat[id].status = 0;
                    boat[id].num = 0;
                    boat[id].pos = -1;
                }
            }
            // 这里设置的是出发条件，之后还要再改
            if (boat[id].num < boat_capacity) {
                
                // 可以装载的货物不能超过容积，不能多于现有货物
                int goods_can_load = min({bt.loading_speed, boat_capacity - boat[id].num, bt.num_goods});
                // cerr << "dz " << goods_can_load << ' ' << id << ' ' << boat[id].num << " | " << boat[id].pos << ' ' << berth[boat[id].pos].num_goods << endl;
                boat[id].num = boat[id].num + goods_can_load;
                bt.num_goods -= goods_can_load;
                boat[id].ready2go += 1;
            }
        }
    }
}

void get_robot_order() {
    for (int i = 0; i < robot_num; i++) {
        robot_order[i] = i; 
    }
    // int cnt = 0;
    // for (int i = 0; i < robot_num;)
    // return cnt;
}

pair<vector<pair<short, short> >, float> getGoalPoints(const int* robot_order) {
    int id, x, y;
    float res = 0;
    vector<pair<short, short> > goal_pos(robot_num);
    occupied_pos.clear();
    for (int i = 0; i < robot_num; i++) {
        id = robot_order[i];
        if (robot[id].goods == 1) {
            continue;
        }
        // cerr << id << endl;
        for (int j = 0; j < robot_num + 1; j++) {
            auto& item = value_matrix[id][j];
            // cerr << item.second << endl;
            // 如果找到最后还是没有合适的，返回{-1, -1}，表示该机器人不动
            if (item.second == -1) {
                goal_pos[id] = {-1, -1};
                break;
            }
            // 如果该点没有先前的点占用
            if (occupied_pos.find(item.first) == occupied_pos.end()) {
                occupied_pos.insert(item.first);
                goal_pos[id] = item.first;
                res += item.second;
                break;
            }
        }
    }
    return {goal_pos, res};
}

const int ROBOT_COUNT = 10; 
const double T_START = 200; // 初始温度
const double T_END = 1e-3; // 终止温度
const double COOLING_RATE = 0.98; // 冷却率
const int ITERATIONS_PER_TEMPERATURE = 10; // 每个温度的迭代次数
const int SA_INTERVAL = 1;
const float ACCEPT_RATE = 1.1;

float v1 = 0, vr = 0, total_v = 0;
int robot_num0;

void simulatedAnnealing() {
    srand(time(NULL));
    int currentOrder[ROBOT_COUNT], newOrder[ROBOT_COUNT];
    // memcpy(currentOrder, robot_order, sizeof(robot_order));
    for (int i = 0; i < robot_num; i++) currentOrder[i] = i;
    
    int cnt = 0, robot_0_goods;

    for (int i = 0; i < robot_num; i++) {
        if (robot[i].goods == 0) {
            currentOrder[cnt] = i;
            cnt ++;
        }
    }
    robot_0_goods = cnt;
    for (int i = 0; i < robot_num; i++) {
        if (robot[i].goods == 1) {
            currentOrder[cnt] = i;
            cnt ++;
        }
    }
    // cerr << robot_0_goods << endl;
    // robot_0_goods = ROBOT_COUNT;
    if (robot_0_goods > ROBOT_COUNT) {
        robot_0_goods = ROBOT_COUNT;
    }
    if (robot_0_goods == 0) {
        return ;
    }

    auto gp = getGoalPoints(currentOrder);

    double oldValue = total_v, currentValue = gp.second, newValue;
    double temp = T_START;

    while (temp > T_END) {
        for (int i = 0; i < ITERATIONS_PER_TEMPERATURE; ++i) {
            // 复制当前路径到新路径
            memcpy(newOrder, currentOrder, sizeof(currentOrder));

            // 产生新解
            int a = rand() % robot_0_goods;
            int b = rand() % robot_0_goods;
            swap(newOrder[a], newOrder[b]);

            // 计算新解的路径长度
            gp = getGoalPoints(newOrder);
            newValue = gp.second;
            // 接受准则
            double rnd = (double)rand() / RAND_MAX, currentPrb = exp((newValue - currentValue) / temp);

            // cerr << currentPrb << ' ' << rnd << endl;
            if (newValue > currentValue || currentPrb > rnd) {
                memcpy(currentOrder, newOrder, sizeof(currentOrder));
                currentValue = newValue;
            }
        }
        // 降温
        temp *= COOLING_RATE;
    }
    gp = getGoalPoints(currentOrder);
    // cerr << gp.second << ' '; for (int i = 0; i < robot_num; i++) cerr<<currentOrder[i]<<' '; cerr << endl;

    if (gp.second / oldValue >= ACCEPT_RATE) {
        // cerr << gp.second / oldValue << ' ' << gp.second << ' ' << oldValue << endl;
        // for (int i = 0; i < robot_num; i++) cerr<<currentOrder[i]<<' '; cerr << endl;
        // for (int i = 0; i < robot_num; i++) cerr<<robot_order[i]<<' '; cerr << endl;
        
        memcpy(robot_order, currentOrder, sizeof(currentOrder));
        robot_goals = gp.first;
        total_v = gp.second;
    }
    // cerr << total_v << ' '; for (int i = 0; i < robot_num; i++) cerr<<robot_order[i]<<' '; cerr << endl;
}

int main()
{
    Init();
    // random_device rd; 
    // mt19937 gen(rd());
    get_robot_order();
    for(int zhen = 1; zhen <= 15000; zhen ++)
    {
        int id = Input();
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                goal_occupied[i][j] = 0;

        for (int i = 0; i < robot_num; i++){
            grid_occupied.insert({robot[i].x, robot[i].y});
            ch[robot[i].x][robot[i].y] = 'O';
        }

        for (int i = 0; i < berth_num; i++) {
            if (zhen + berth[i].transport_time >= 15000 - 10){
                berth[i].available = 0;
            }
        }
        
        robot_num0 = 0;
        // 初始化目标选择矩阵V
        for (int i = 0; i < robot_num; i++) {
            Robot& r = robot[i];
            if (r.goods == 1) {
                continue;
            }
            robot_num0 ++;
            pair<int, int> start = {r.x, r.y};

            BfsShortestPathRobot(start, r.distance_matrix, r.predecessors); // 得到robot i的距离矩阵
            auto& dist = r.distance_matrix;
            GetMapValue(dist); // 得到robot i的价值矩阵
            get_k_max_value(i); // 将前k大的填充到V矩阵
        }
        
        auto gp = getGoalPoints(robot_order);
        robot_goals = gp.first;
        total_v = gp.second;
        v1 += total_v;
        // cerr << total_v << ' '; for (int i = 0; i < robot_num; i++) cerr<<robot_order[i]<<' '; cerr << endl;
        if (zhen % SA_INTERVAL == 0) {
            simulatedAnnealing();
        }  
        vr += total_v;
        // srand(unsigned(time(0)));
        // shuffle(begin(robot_order), end(robot_order), gen);
        // gp = getGoalPoints(robot_order);
        // robot_goals = gp.first;
        // total_v = gp.second;
        // vr += total_v;
        // cerr << total_v << ' '; for (int i = 0; i < robot_num; i++) cerr<<robot_order[i]<<' '; cerr << endl;
        
        for (int i = 0; i < robot_num; i++) {
            ScheduleRobotAction(robot_order[i]);
        }
        
        // cerr << v1 << endl << vr << endl;

        for(int i = 0; i < 5; i++){
            ScheduleBoatAction(i, zhen);
        }

        for(auto& grid : grid_occupied) {
            ch[grid.first][grid.second] = '.';
        }
        grid_occupied.clear();
        puts("OK");
        fflush(stdout);
    }
    return 0;
}
