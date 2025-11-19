#include <bits/stdc++.h>
using namespace std;

// ---------- 抽象游戏状态接口 ----------
struct GameState {
  // 返回当前玩家（1 或 -1）
  virtual int current_player() const = 0;
  // 返回可用动作集合（动作用整数表示）
  virtual vector<int> legal_actions() const = 0;
  // 施加动作并返回新状态
  virtual unique_ptr<GameState> next_state(int action) const = 0;
  // 判断是否结束
  virtual bool is_terminal() const = 0;
  // 返回赢家：1 表示玩家1 胜，-1 表示玩家2 胜，0 表示平局或非终局
  virtual int winner() const = 0;

  virtual std::unique_ptr<GameState> clone() const = 0;
  virtual ~GameState() = default;
};

// ---------- 井字棋实现（3x3） ----------
struct TicTacToeState : public GameState {
  // board: 0 empty, 1 player1 (X), -1 player2 (O)
  array<int, 9> board{};
  int player; // 当前玩家 1 或 -1

  TicTacToeState() {
    board.fill(0);
    player = 1;
  }
  TicTacToeState(const array<int, 9> &b, int p) : board(b), player(p) {}

  int current_player() const override { return player; }

  vector<int> legal_actions() const override {
    vector<int> acts;
    for (int i = 0; i < 9; ++i)
      if (board[i] == 0)
        acts.push_back(i);
    return acts;
  }

  unique_ptr<GameState> next_state(int action) const override {
    auto nb = board;
    nb[action] = player;
    return make_unique<TicTacToeState>(nb, -player);
  }

  std::unique_ptr<GameState> clone() const override {
    return std::make_unique<TicTacToeState>(board, player);
  }

  bool is_terminal() const override {
    if (winner() != 0)
      return true;
    for (int i = 0; i < 9; ++i)
      if (board[i] == 0)
        return false;
    return true; // 平局
  }

  int winner() const override {
    static const int lines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // cols
        {0, 4, 8}, {2, 4, 6}             // diagonals
    };
    for (auto &ln : lines) {
      int s = board[ln[0]] + board[ln[1]] + board[ln[2]];
      if (s == 3)
        return 1;
      if (s == -3)
        return -1;
    }
    return 0;
  }

  // 便于打印
  string str() const {
    string out;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        int v = board[r * 3 + c];
        char ch = v == 1 ? 'X' : (v == -1 ? 'O' : '.');
        out.push_back(ch);
        if (c < 2)
          out.push_back(' ');
      }
      if (r < 2)
        out += '\n';
    }
    return out;
  }
};

// ---------- MCTS 节点 ----------
struct MCTSNode {
  MCTSNode *parent = nullptr;
  vector<unique_ptr<MCTSNode>> children;
  int action_from_parent = -1; // action that led from parent to this node
  int player_to_move;          // 节点对应状态的 current_player()

  double wins = 0.0; // 累计胜利次数（相对于 player_to_move 的父玩家? 下面采用:
                     // 统计对父玩家的胜利数）
  int visits = 0;

  // 保存对应的游戏状态指针（可选），为了快速扩展/模拟我们存储状态副本
  unique_ptr<GameState> state;

  MCTSNode(unique_ptr<GameState> s, MCTSNode *parent_, int action_)
      : parent(parent_), action_from_parent(action_) {
    state = move(s);
    player_to_move = state->current_player();
  }

  bool is_fully_expanded() const {
    auto acts = state->legal_actions();
    return (int)children.size() == (int)acts.size();
  }

  bool is_leaf() const { return children.empty(); }
};

// ---------- 随机数 ----------
static std::mt19937_64 &rng() {
  static thread_local std::mt19937_64 gen(
      (unsigned)chrono::high_resolution_clock::now()
          .time_since_epoch()
          .count());
  return gen;
}

// ---------- 蒙特卡洛树搜索实现 ----------
class MCTS {
public:
  MCTS(int iterations = 1000, double exploration = 1.414)
      : iterations_(iterations), C_(exploration) {}

  // 从根状态运行 MCTS 并返回最佳动作
  int search(const GameState &root_state) {
    auto root = make_unique<MCTSNode>(
        root_state.next_state(-1) /* 复制 trick: we need a unique_ptr state */,
        nullptr, -1);
    // Actually we need a copy of root_state. Let's create via next_state on a
    // fake action: fix by cloning

    // Proper root copy
    root = make_unique<MCTSNode>(clone_state(root_state), nullptr, -1);

    for (int it = 0; it < iterations_; ++it) {
      MCTSNode *node = tree_policy(root.get());
      double reward = default_policy(node->state.get());
      backup(node, reward);
    }

    // 选择访问次数最多的子动作
    MCTSNode *best = nullptr;
    int best_visits = -1;
    for (auto &c : root->children) {
      if (c->visits > best_visits) {
        best_visits = c->visits;
        best = c.get();
      }
    }
    if (!best) {
      // 没有孩子（终局）
      return -1;
    }
    return best->action_from_parent;
  }

private:
  int iterations_;
  double C_;

  unique_ptr<GameState> clone_state(const GameState &s) {
    // 这里我们只知道 TicTacToeState 的实现，实际项目可为 GameState 增加
    // "clone()" 虚函数
    const TicTacToeState *t = dynamic_cast<const TicTacToeState *>(&s);
    if (!t)
      throw runtime_error(
          "clone_state: only TicTacToeState supported in this demo");
    return make_unique<TicTacToeState>(t->board, t->player);
  }

  MCTSNode *tree_policy(MCTSNode *node) {
    // 选择直到可扩展节点
    while (!node->state->is_terminal()) {
      if (!node->is_fully_expanded()) {
        return expand(node);
      } else {
        node = best_uct_child(node);
      }
    }
    return node;
  }

  MCTSNode *expand(MCTSNode *node) {
    // 找到尚未拓展的动作
    auto acts = node->state->legal_actions();
    // 收集已使用动作
    unordered_set<int> used;
    for (auto &c : node->children)
      used.insert(c->action_from_parent);

    vector<int> untried;
    for (int a : acts)
      if (!used.count(a))
        untried.push_back(a);
    if (untried.empty())
      return node; // 已经完全扩展

    uniform_int_distribution<int> dist(0, (int)untried.size() - 1);
    int act = untried[dist(rng())];

    auto child_state = node->state->next_state(act);
    auto child = make_unique<MCTSNode>(move(child_state), node, act);
    MCTSNode *child_ptr = child.get();
    node->children.push_back(move(child));
    return child_ptr;
  }

  MCTSNode *best_uct_child(MCTSNode *node) {
    double best_score = -numeric_limits<double>::infinity();
    MCTSNode *best = nullptr;
    for (auto &c : node->children) {
      // UCT: (w_i / n_i) + C * sqrt(ln N / n_i)
      if (c->visits == 0)
        return c.get(); // 优先探索未访问节点
      double win_rate = c->wins / c->visits;
      double uct = win_rate + C_ * sqrt(log(node->visits) / c->visits);
      if (uct > best_score) {
        best_score = uct;
        best = c.get();
      }
    }
    return best;
  }

  double default_policy(GameState *state) {
    // 模拟（随机游走）直到终局，返回 reward（相对于 root 的当前玩家）
    auto sim = clone_state(*state);
    while (!sim->is_terminal()) {
      auto acts = sim->legal_actions();
      uniform_int_distribution<int> dist(0, (int)acts.size() - 1);
      int a = acts[dist(rng())];
      sim = sim->next_state(a);
    }
    int w = sim->winner();
    // 返回值设计：如果赢家是 root node 的 parent 那么视为 1，否则 0（或者 -1）
    // 常见做法是返回胜者相对于当前玩家的奖励。这里我们返回：
    // 返回 1 表示胜利者是 root 的 current_player() 的父玩家（即上一手玩家），
    // 0.5 平局，0 失败 — 但为了简化，返回 double: 1.0 (player 1 win), 0.0
    // (player -1 win), 0.5 平局
    if (w == 1)
      return 1.0;
    if (w == -1)
      return 0.0;
    return 0.5;
  }

  void backup(MCTSNode *node, double reward) {
    // 向上回溯，更新 visits 和 wins
    // 注意：reward 的定义相对于哪个玩家可能影响更新方式。这里采用简单方式：
    // 假设 reward 为 1.0 表示玩家1 胜利，0.0 表示玩家-1 胜利，0.5 平局。
    // 每个节点记录 wins 为相对于玩家1 的获胜计数。
    while (node) {
      node->visits += 1;
      if (reward == 0.5)
        node->wins += 0.5; // 平局
      else
        node->wins += reward; // 如果 winner==1 则 +1，否则 +0
      node = node->parent;
    }
  }
};

// ---------- 简单演示：AI vs AI 自博弈 ----------
int main() {
  TicTacToeState init;
  MCTS mcts(2000, 1.4); // 迭代次数和探索因子可调

  auto state = make_unique<TicTacToeState>(init);
  while (!state->is_terminal()) {
    cout << "当前玩家: " << (state->current_player() == 1 ? "X" : "O") << "\n";
    cout << state->str() << "\n";

    // MCTS 选动作
    int best_move = mcts.search(*state);
    if (best_move < 0)
      break; // 终局
    cout << "MCTS 选动作: " << best_move << "\n\n";
    auto next = state->next_state(best_move);
    state = std::unique_ptr<TicTacToeState>(
        dynamic_cast<TicTacToeState *>(next.release()));
  }
  cout << "终局局面:\n" << state->str() << "\n";
  int w = state->winner();
  if (w == 1)
    cout << "X 胜\n";
  else if (w == -1)
    cout << "O 胜\n";
  else
    cout << "平局\n";
  return 0;
}
