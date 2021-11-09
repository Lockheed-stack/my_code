#include <iostream>
#include <vector>
using namespace std;

vector<vector<int>> graph;
vector<int> node_color;
int get_to_terminal = 0;
struct NODE
{
    vector<NODE> children;
    int color;
    int node_id;
};

void node_initialization(int colors_num, NODE &node)
{
    node.children = vector<NODE>(colors_num);
    for (int i = 0; i < colors_num; i++)
    {
        node.children[i].color = i;
    }
}

void node_color_init(vector<int> &node_color, int node_num)
{
    for (int i = 0; i <= node_num; i++)
    {
        node_color[i] = -1;
    }
}

bool is_adjacency(int loc, int numOfnode)
{
    for (int i = 1; i < numOfnode; i++)
    {
        if (graph[loc][i] == 1)
        {
            return true;
        }
    }
    return false;
}

bool is_same_color(NODE node, int loc, int numOfnode)
{
    for (int i = 1; i < numOfnode; i++)
    {
        if (graph[loc][i] == 1 && node_color[i] == node.color)
        {
            return true;
        }
    }
    return false;
}

void backtracking(NODE &root, int colors_num, int all_node_num, int cur_loc)
{
    if (root.node_id == 0) //根节点
    {
        for (int i = 0; i < colors_num; i++)
        {
            root.children[i].node_id = cur_loc + 1;
            backtracking(root.children[i], colors_num, all_node_num, cur_loc + 1); //进入下一层
        }
    }
    else //非根节点
    {
        node_initialization(colors_num, root);

        for (int i = 0; i < colors_num; i++)
        {
            if (is_adjacency(cur_loc, cur_loc)) //存在相邻的点
            {
                if (!is_same_color(root, cur_loc, cur_loc)) //相邻点颜色不同
                {
                    node_color[cur_loc] = root.color;
                    root.children[i].node_id = cur_loc + 1;
                }
                else
                {
                    return;
                }
                if (cur_loc < all_node_num) //还能进入下一层
                {
                    backtracking(root.children[i], colors_num, all_node_num, cur_loc + 1); //进入下一层
                }
                else
                {
                    if (cur_loc == all_node_num)
                    {
                        get_to_terminal += 1;
                    }

                    return;
                }
            }
            else if (all_node_num > cur_loc) //没有(或者暂时没有)相邻的点
            {
                node_color[cur_loc] = root.color;
                root.children[i].node_id = cur_loc + 1;
                backtracking(root.children[i], colors_num, all_node_num, cur_loc + 1); //进入下一层
            }
            else
            {
                if (cur_loc == all_node_num)
                {
                    get_to_terminal += 1;
                }
                return;
            }
        }
    }
}

int main()
{
    int n, m, k; //结点数、边数和可用颜色数
    cin >> n >> m >> k;

    graph = vector<vector<int>>(n + 1, vector<int>(n + 1));
    node_color = vector<int>(n + 1);
    NODE root;

    for (int i = 0; i < m; i++) //create a graph
    {
        int t1, t2;
        cin >> t1 >> t2;
        graph[t2][t1] = graph[t1][t2] = 1;
    }
    node_color_init(node_color, n);
    node_initialization(k, root);
    root.node_id = 0;

    backtracking(root, k, n, 0);

    cout<<get_to_terminal;
    return 0;
}