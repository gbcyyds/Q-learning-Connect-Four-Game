import numpy as np
import random
import pygame
import sys
import matplotlib.pyplot as plt
from collections import deque
import copy


# ==================== Connect Four Game ====================
class ConnectFour:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_move = None

    def get_valid_moves(self):
        moves = []
        for col in range(self.cols):
            if self.board[0][col] == 0:
                moves.append(col)
        return moves

    def drop_piece(self, col):
        if col not in self.get_valid_moves() or self.game_over:
            return False

        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.last_move = (row, col)
                break

        if self.check_win(self.current_player):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = 3 - self.current_player

        return True

    def check_win(self, player):
        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if (self.board[r][c] == player and
                        self.board[r][c + 1] == player and
                        self.board[r][c + 2] == player and
                        self.board[r][c + 3] == player):
                    return True

        # Vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if (self.board[r][c] == player and
                        self.board[r + 1][c] == player and
                        self.board[r + 2][c] == player and
                        self.board[r + 3][c] == player):
                    return True

        # Positive diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if (self.board[r][c] == player and
                        self.board[r + 1][c + 1] == player and
                        self.board[r + 2][c + 2] == player and
                        self.board[r + 3][c + 3] == player):
                    return True

        # Negative diagonal
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if (self.board[r][c] == player and
                        self.board[r - 1][c + 1] == player and
                        self.board[r - 2][c + 2] == player and
                        self.board[r - 3][c + 3] == player):
                    return True
        return False

    def get_state_key(self):
        """Get state representation"""
        state = self.board.copy()
        state = np.where(state == 2, 1, state)
        state = np.where(state == 1, -1, state)
        return str(state.flatten())


# ==================== AI Agent ====================
class QLearningAI:
    def __init__(self):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.training_rewards = []

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def would_win(self, game, col, player):
        """Check if player would win by dropping piece in col"""
        temp_game = ConnectFour()
        temp_game.board = game.board.copy()
        temp_game.current_player = player
        temp_game.drop_piece(col)
        return temp_game.winner == player

    def evaluate_position(self, game, col, player):
        """Evaluate the value of a position after dropping piece"""
        temp_game = ConnectFour()
        temp_game.board = game.board.copy()
        temp_game.current_player = player
        temp_game.drop_piece(col)

        score = 0
        # Center preference
        if col == 3:
            score += 3
        elif col in [2, 4]:
            score += 2
        elif col in [1, 5]:
            score += 1

        # Check for creating winning opportunities (simplified)
        valid_moves = temp_game.get_valid_moves()
        for next_col in valid_moves:
            if self.would_win(temp_game, next_col, player):
                score += 5

        return score

    def choose_action(self, game, training=False):
        """Choose action - hybrid strategy"""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        # Non-training mode (vs player) uses strong heuristics
        if not training:
            # 1. Prioritize winning
            for col in valid_moves:
                if self.would_win(game, col, 2):
                    print(f"AI: I can win! Column {col + 1}")
                    return col

            # 2. Block player win
            for col in valid_moves:
                if self.would_win(game, col, 1):
                    print(f"AI: Blocking you! Column {col + 1}")
                    return col

            # 3. Evaluate each position
            best_score = -1
            best_col = random.choice(valid_moves)
            for col in valid_moves:
                score = self.evaluate_position(game, col, 2)
                if score > best_score:
                    best_score = score
                    best_col = col

            print(f"AI: Choosing column {best_col + 1} (score: {best_score})")
            return best_col

        # Training mode uses Q-learning
        state = game.get_state_key()
        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        q_values = [self.get_q_value(state, a) for a in valid_moves]
        max_q = max(q_values)
        best_moves = [a for a, q in zip(valid_moves, q_values) if q == max_q]
        return random.choice(best_moves)

    def learn(self, state, action, reward, next_state, next_valid_moves, done):
        old_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            next_qs = [self.get_q_value(next_state, a) for a in next_valid_moves]
            target = reward + self.gamma * max(next_qs) if next_qs else reward

        new_q = old_q + self.lr * (target - old_q)
        self.q_table[(state, action)] = new_q

    def strong_opponent(self, game):
        """Strong opponent for training"""
        valid_moves = game.get_valid_moves()

        # 1. Win immediately
        for col in valid_moves:
            if self.would_win(game, col, game.current_player):
                return col

        # 2. Block AI win
        for col in valid_moves:
            if self.would_win(game, col, 3 - game.current_player):
                return col

        # 3. Prefer center
        center = 3
        if center in valid_moves:
            return center

        # 4. Evaluate position
        best_score = -1
        best_col = random.choice(valid_moves)
        for col in valid_moves:
            temp_game = ConnectFour()
            temp_game.board = game.board.copy()
            temp_game.current_player = game.current_player
            temp_game.drop_piece(col)
            score = len(temp_game.get_valid_moves())  # Simple scoring
            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def train(self, episodes=50000):
        """Train AI"""
        print("=" * 60)
        print("Training AI...")
        print(f"Training episodes: {episodes}")
        print("=" * 60)

        game = ConnectFour()
        reward_window = deque(maxlen=1000)

        for episode in range(episodes):
            game.reset()

            # Random first player
            if random.random() < 0.5:
                game.current_player = 2
                ai_player = 2
            else:
                game.current_player = 1
                ai_player = 1

            state = game.get_state_key()
            episode_reward = 0

            while not game.game_over:
                valid_moves = game.get_valid_moves()

                if game.current_player == ai_player:
                    # AI turn
                    action = self.choose_action(game, training=True)
                    if action is None:
                        break
                    game.drop_piece(action)

                    next_state = game.get_state_key()

                    if game.winner == ai_player:
                        reward = 100
                    elif game.winner == 3 - ai_player:
                        reward = -100
                    elif game.winner == 0:
                        reward = 50
                    else:
                        reward = 0.1

                    self.learn(state, action, reward, next_state,
                               game.get_valid_moves(), game.game_over)

                    state = next_state
                    episode_reward += reward

                else:
                    # Opponent turn - use strong opponent
                    action = self.strong_opponent(game)
                    game.drop_piece(action)
                    state = game.get_state_key()

            self.training_rewards.append(episode_reward)
            reward_window.append(episode_reward)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Show progress
            if (episode + 1) % 5000 == 0:
                avg_reward = np.mean(reward_window)
                print(f"Episode {episode + 1}: Average reward = {avg_reward:.1f}")

        print("=" * 60)
        print("Training complete!")
        print(f"Q-table size: {len(self.q_table)}")
        print("=" * 60)

        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(10, 5))

        smooth_rewards = []
        window = 1000
        for i in range(len(self.training_rewards) - window + 1):
            smooth_rewards.append(np.mean(self.training_rewards[i:i + window]))

        plt.plot(range(window, len(self.training_rewards) + 1),
                 smooth_rewards, 'b-', linewidth=2)
        plt.xlabel('Training Episodes')
        plt.ylabel('Average Reward')
        plt.title('AI Training Curve')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150)
        plt.show()


# ==================== Pygame Interface ====================
class GameUI:
    def __init__(self):
        pygame.init()

        self.rows = 6
        self.cols = 7
        self.cell_size = 90
        self.radius = self.cell_size // 2 - 8

        self.width = self.cols * self.cell_size
        self.height = (self.rows + 1) * self.cell_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect Four - Try to Beat AI")

        self.colors = {
            'bg': (0, 0, 150),
            'empty': (220, 220, 220),
            'player': (255, 50, 50),
            'ai': (255, 255, 50),
            'text': (255, 255, 255)
        }

        self.clock = pygame.time.Clock()
        self.game = ConnectFour()
        self.ai = QLearningAI()
        self.hover_col = -1
        self.waiting_for_ai = False
        self.stats = {'wins': 0, 'losses': 0, 'ties': 0}

    def draw_board(self):
        self.screen.fill(self.colors['bg'])

        # Draw grid and pieces
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * self.cell_size + self.cell_size // 2
                y = (r + 1) * self.cell_size + self.cell_size // 2

                pygame.draw.rect(self.screen, (0, 0, 100),
                                 (c * self.cell_size, (r + 1) * self.cell_size,
                                  self.cell_size, self.cell_size))

                if self.game.board[r][c] == 1:
                    pygame.draw.circle(self.screen, self.colors['player'], (x, y), self.radius)
                    pygame.draw.circle(self.screen, (255, 255, 255), (x - 5, y - 5), self.radius // 3)
                elif self.game.board[r][c] == 2:
                    pygame.draw.circle(self.screen, self.colors['ai'], (x, y), self.radius)
                    pygame.draw.circle(self.screen, (255, 255, 255), (x - 5, y - 5), self.radius // 3)
                else:
                    pygame.draw.circle(self.screen, self.colors['empty'], (x, y), self.radius)

        # Draw hover preview
        if not self.waiting_for_ai and self.hover_col != -1 and not self.game.game_over and self.game.current_player == 1:
            if self.hover_col in self.game.get_valid_moves():
                x = self.hover_col * self.cell_size + self.cell_size // 2
                y = self.cell_size // 2
                preview = pygame.Surface((self.radius * 2, self.radius * 2))
                preview.set_alpha(128)
                pygame.draw.circle(preview, self.colors['player'],
                                   (self.radius, self.radius), self.radius)
                self.screen.blit(preview, (x - self.radius, y - self.radius))

        # Draw column numbers
        font = pygame.font.Font(None, 36)
        for c in range(self.cols):
            x = c * self.cell_size + self.cell_size // 2
            y = self.cell_size // 2
            text = font.render(str(c + 1), True, self.colors['text'])
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

        # Display status
        font = pygame.font.Font(None, 24)
        if self.waiting_for_ai:
            text = font.render("AI Thinking...", True, self.colors['ai'])
            self.screen.blit(text, (10, 10))
        elif not self.game.game_over:
            if self.game.current_player == 1:
                text = font.render("Your Turn", True, self.colors['player'])
                self.screen.blit(text, (10, 10))

        # Display stats
        stats_text = f"W: {self.stats['wins']}  L: {self.stats['losses']}  T: {self.stats['ties']}"
        text = font.render(stats_text, True, self.colors['text'])
        self.screen.blit(text, (self.width - 150, 15))

        # Game over display
        if self.game.game_over:
            font = pygame.font.Font(None, 72)
            if self.game.winner == 1:
                text = font.render("YOU WIN!", True, self.colors['player'])
                self.stats['wins'] += 1
                print("Player wins!")
            elif self.game.winner == 2:
                text = font.render("AI WINS!", True, self.colors['ai'])
                self.stats['losses'] += 1
                print("AI wins!")
            elif self.game.winner == 0:
                text = font.render("TIE!", True, self.colors['text'])
                self.stats['ties'] += 1
                print("Tie game!")
            else:
                text = None

            if text:
                text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
                s = pygame.Surface((text_rect.width + 20, text_rect.height + 20))
                s.set_alpha(200)
                s.fill((0, 0, 0))
                self.screen.blit(s, (text_rect.x - 10, text_rect.y - 10))
                self.screen.blit(text, text_rect)

        pygame.display.flip()

    def player_move(self, col):
        """Handle player move"""
        if col in self.game.get_valid_moves():
            self.game.drop_piece(col)
            print(f"Player move: Column {col + 1}")
            self.draw_board()

            if not self.game.game_over:
                self.waiting_for_ai = True
                pygame.time.set_timer(pygame.USEREVENT, 500)

    def ai_move(self):
        """Handle AI move"""
        action = self.ai.choose_action(self.game, training=False)
        if action is not None:
            self.game.drop_piece(action)

        self.waiting_for_ai = False
        pygame.time.set_timer(pygame.USEREVENT, 0)
        self.draw_board()

    def run(self):
        # Train AI
        self.ai.train(episodes=30000)

        # Start game
        self.game.reset()
        print("\n" + "=" * 60)
        print("Game Started!")
        print("Player: Red ●")
        print("AI: Yellow ●")
        print("AI now plays aggressively and defensively. Good luck!")
        print("=" * 60 + "\n")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEMOTION:
                    x, y = pygame.mouse.get_pos()
                    self.hover_col = x // self.cell_size

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.game.game_over and not self.waiting_for_ai:
                        x, y = pygame.mouse.get_pos()
                        col = x // self.cell_size
                        self.player_move(col)

                    elif self.game.game_over:
                        self.game.reset()
                        self.waiting_for_ai = False
                        print("\nNew game started!")

                elif event.type == pygame.USEREVENT:
                    if self.waiting_for_ai and not self.game.game_over:
                        self.ai_move()

            self.draw_board()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    ui = GameUI()
    ui.run()