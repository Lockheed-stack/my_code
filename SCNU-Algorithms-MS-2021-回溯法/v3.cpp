#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stack>
using namespace std;
vector<vector<int>> chessboard;
int board_size = 0;

struct NODE
{
    int x;
    int y;
    NODE(int loc_x, int loc_y) : x(loc_x), y(loc_y) {}
    NODE()
    {
        x = 0;
        y = 0;
    }
};
//8个方向
int x_axis[8] = {-2, -2, -1, -1, 1, 1, 2, 2};
int y_axis[8] = {-1, 1, -2, 2, -2, 2, -1, 1};

//判断下一步能否走
inline bool check(int x, int y)
{
    //x,y没有越界且下一步没被走过
    if (x >= 0 && y >= 0 && x < board_size && y < board_size && chessboard[x][y] == 0)
    {
        return true;
    }
    return false;
}

//下一步有几种走法
int next_step_ways(int x, int y)
{
    int steps = 0;
    for (int i = 0; i < 8; i++)
    {
        if (check(x + x_axis[i], y + y_axis[i]))
        {
            steps++;
        }
    }
    return steps;
}

//距离中心的距离(平方)
inline int distance_to_center(int x, int y)
{
    int x_mid = board_size / 2 - 1;
    int y_mid = board_size / 2 - 1;

    return (x - x_mid) * (x - x_mid) + (y - y_mid) * (y - y_mid);
}

//排序的比较函数
inline bool cmp_next_step_ways(NODE n1, NODE n2)
{
    if (next_step_ways(n1.x, n1.y) < next_step_ways(n2.x, n2.y))
    {
        return true;
    }
    else if (next_step_ways(n1.x, n1.y) == next_step_ways(n2.x, n2.y))
    {
        if (distance_to_center(n1.x, n1.y) > distance_to_center(n2.x, n2.y))
        {
            return true;
        }
    }
    return false;
}

//判断最后是否回到起点
bool back_to_start(int x, int y)
{

    int x_mid = board_size / 2 - 1;
    int y_mid = board_size / 2 - 1;
    for (int i = 0; i < 8; i++)
    {
        if (x + x_axis[i] == x_mid && y + y_axis[i] == y_mid)
        {
            return true;
        }
    }

    return false;
}

void print_output(int original_x, int original_y)
{

    int all_cells = board_size * board_size;
    int length = all_cells - chessboard[original_x - 1][original_y - 1];

    for (int i = 0; i < board_size; i++)
    {
        for (int j = 0; j < board_size; j++)
        {
            chessboard[i][j] = (chessboard[i][j] + length) % all_cells + 1;
            cout << setw(3) << chessboard[i][j];
            if (j + 1 != board_size)
            {
                cout << " ";
            }
        }
        if (i + 1 != board_size)
        {
            cout << endl;
        }
    }
}

void print()
{
    for (int i = 0; i < board_size; i++)
    {
        for (int j = 0; j < board_size; j++)
        {
            cout << setw(3) << chessboard[i][j];
            if (j + 1 != board_size)
            {
                cout << " ";
            }
        }
        cout << endl;
    }
    cout << endl;
}

void find_next_available_jump(vector<NODE> &prune_branch, int x, int y)
{
    for (int i = 0; i < 8; i++)
    {
        NODE temp;
        if (check(x + x_axis[i], y + y_axis[i]))
        {
            temp.x = x + x_axis[i];
            temp.y = y + y_axis[i];
            prune_branch.emplace_back(temp);
        }
    }
}
void find_way(int x, int y, int original_x, int original_y, int cur_loc)
{

    vector<NODE> prune_branch;
    stack<NODE> recursive_stack;
    stack<NODE> pre_node_loc;

    pre_node_loc.push(NODE(x, y));
    // for (int i = 0; i < 8; i++) //从中心点出发，搜索可用的下一跳
    // {
    //     NODE temp;
    //     if (check(x + x_axis[i], y + y_axis[i]))
    //     {
    //         temp.x = x + x_axis[i];
    //         temp.y = y + y_axis[i];
    //         prune_branch.emplace_back(temp);
    //     }
    // }
    find_next_available_jump(prune_branch, x, y); //从中心点出发，搜索可用的下一跳

    //对搜索顺序进行排序
    sort(prune_branch.begin(), prune_branch.end(), cmp_next_step_ways);
    for (auto iter = prune_branch.rbegin(); iter != prune_branch.rend(); iter++)
    {
        recursive_stack.push(*iter);
    }
    /***********************************************/

    while (!recursive_stack.empty())
    {
        vector<NODE> prune_branch;
        NODE stack_top_node = recursive_stack.top();
        recursive_stack.pop();
        //pre_node_loc.push(stack_top_node);

        // for (int i = 0; i < 8; i++) //寻找可用的下一跳
        // {
        //     NODE temp;
        //     if (check(stack_top_node.x + x_axis[i], stack_top_node.y + y_axis[i]))
        //     {
        //         temp.x = stack_top_node.x + x_axis[i];
        //         temp.y = stack_top_node.y + y_axis[i];
        //         prune_branch.emplace_back(temp);
        //     }
        // }
        //find_next_avaliable_jump(prune_branch, stack_top_node.x, stack_top_node.y); //寻找可用的下一跳
        //对搜索顺序进行排序
        // sort(prune_branch.begin(), prune_branch.end(), cmp_next_step_ways);
        // for (auto iter = prune_branch.rbegin(); iter != prune_branch.rend(); iter++)
        // {
        //     recursive_stack.push(*iter);
        // }

        /******************     进行判断    *************************/

        if (cur_loc == board_size * board_size && back_to_start(x, y))
        {
            //输出结果
            print_output(original_x, original_y);
            exit(0);
        }
        else if (cur_loc < board_size * board_size)
        {

            if (check(stack_top_node.x, stack_top_node.y))
            {
                find_next_available_jump(prune_branch, stack_top_node.x, stack_top_node.y); //寻找可用的下一跳
                sort(prune_branch.begin(), prune_branch.end(), cmp_next_step_ways); //对搜索顺序进行排序
                for (auto iter = prune_branch.rbegin(); iter != prune_branch.rend(); iter++)
                {
                    recursive_stack.push(*iter);
                }

                x = stack_top_node.x;
                y = stack_top_node.y;
                pre_node_loc.push(NODE(x, y));
                chessboard[x][y] = ++cur_loc;
                print();
            }
            else //走错路，回退
            {
                pre_node_loc.pop();
                recursive_stack.pop();
                chessboard[x][y] = 0;
                cur_loc--;
                x = pre_node_loc.top().x;
                y = pre_node_loc.top().y;

                print();
            }
        }
        else //走错路，回退
        {
            pre_node_loc.pop();
            recursive_stack.pop();
            chessboard[x][y] = 0;
            cur_loc--;
            x = pre_node_loc.top().x;
            y = pre_node_loc.top().y;

            print();
        }
    }
}

int main()
{
    int n, a, b;
    cin >> n >> a >> b;
    board_size = n;
    chessboard = vector<vector<int>>(n, vector<int>(n));
    chessboard[n / 2 - 1][n / 2 - 1] = 1;
    find_way(n / 2 - 1, n / 2 - 1, a, b, 1); //从中心点开始
    return 0;
}