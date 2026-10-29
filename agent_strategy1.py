from collections import deque


class PersistentMemoryAgent:
    """Persistent-memory vacuum agent for the AIMA-style vacuum world."""

    def __init__(self, env):
        """Store the environment reference and initialize the persistent memory."""
        self.env = env
        self.last_action = None
        self.last_pos = None

        self.memory = {
            "visited": set(),  # (r,c) cells the robot has stood on
            "obstacles": set(),  # (r,c) cells known to be 'X' / obstacle or OOB
            "rewards": {},  # (r,c) -> -1 or 10, observed values
            "dirty_known": set(),  # (r,c) cells observed dirty and not yet sucked
            "clean_known": set(),  # (r,c) cells observed clean after SUCK
            "last_pos": None,  # (r,c) of previous step, for loop detection
            "last_action": None,  # str, previous action
            "step_count": 0,  # int, monotonic step counter
            "history": [],  # list of (step, pos, action, note) log entries
        }

    def _log(self, pos, note):
        """Append a human-readable note to memory history and print it."""
        step = self.memory["step_count"]
        action = self.memory["last_action"]
        entry = (step, pos, action, note)
        self.memory["history"].append(entry)
        print(f"[step {step}] pos={pos} action={action} → {note}")

    def _neighbor_coord(self, pos, direction):
        """Return the coordinate one step in a direction, or None if out of bounds."""
        row, col = pos
        if direction == "UP":
            next_pos = (row - 1, col)
        elif direction == "DOWN":
            next_pos = (row + 1, col)
        elif direction == "LEFT":
            next_pos = (row, col - 1)
        elif direction == "RIGHT":
            next_pos = (row, col + 1)
        else:
            return None

        max_row = 3
        max_col = 3
        next_row, next_col = next_pos
        if next_row < 0 or next_row > max_row:
            return None
        if next_col < 0 or next_col > max_col:
            return None
        return next_pos

    def _ordered_directions(self):
        """Return a stable direction ordering for exploration and fallbacks."""
        return ("UP", "DOWN", "LEFT", "RIGHT")

    def _manhattan_distance(self, start, goal):
        """Compute the Manhattan distance between two grid coordinates."""
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _reconstruct_first_action(self, came_from, goal, start):
        """Reconstruct the first move from the BFS search path."""
        current = goal
        while current in came_from:
            previous, action = came_from[current]
            if previous is None:
                return "NO-OP"
            if previous == start:
                return action
            current = previous
        return "NO-OP"

    def _bfs_next_step(self, start, goal):
        """Return the first action toward goal using BFS over safe cells."""
        if start == goal:
            return "NO-OP"

        frontier = deque([start])
        visited = {start}
        came_from = {start: (None, None)}

        while frontier:
            current = frontier.popleft()
            for direction in self._ordered_directions():
                next_pos = self._neighbor_coord(current, direction)
                if next_pos is None:
                    continue
                if next_pos in self.memory["obstacles"]:
                    continue
                if next_pos in visited:
                    continue
                visited.add(next_pos)
                came_from[next_pos] = (current, direction)
                if next_pos == goal:
                    return self._reconstruct_first_action(came_from, goal, start)
                frontier.append(next_pos)

        return "NO-OP"

    def update_memory(self, percept):
        """Merge a percept into persistent memory and record the update history."""
        pos = percept["pos"]

        # --- 1. Record current position as visited ---
        if pos not in self.memory["visited"]:
            self.memory["visited"].add(pos)
            self._log(pos, f"visited {pos}")

        # --- 2. Record obstacles from neighbor senses ---
        for direction in self._ordered_directions():
            neighbor_coord = self._neighbor_coord(pos, direction)
            neighbor_value = percept["neighbors"].get(direction)
            if neighbor_value is None and neighbor_coord is not None:
                if neighbor_coord not in self.memory["obstacles"]:
                    self.memory["obstacles"].add(neighbor_coord)
                    self._log(pos, f"sensed obstacle at {neighbor_coord}")

        # --- 3. Record rewards observed in adjacent cells ---
        for direction in self._ordered_directions():
            neighbor_coord = self._neighbor_coord(pos, direction)
            neighbor_value = percept["neighbors"].get(direction)
            if neighbor_coord is not None and neighbor_value in (-1, 10):
                if self.memory["rewards"].get(neighbor_coord) != neighbor_value:
                    self.memory["rewards"][neighbor_coord] = neighbor_value
                    self._log(
                        pos, f"recorded reward {neighbor_value} at {neighbor_coord}"
                    )

        # --- 4. Track dirty and clean observations ---
        if percept["dirty"]:
            self.memory["dirty_known"].add(pos)
            self.memory["clean_known"].discard(pos)
            self._log(pos, "observed dirty cell")
        else:
            self.memory["dirty_known"].discard(pos)
            self.memory["clean_known"].add(pos)
            self._log(pos, "observed clean cell")

        # --- 5. Update step log AFTER using last_pos / last_action ---
        self.memory["history"].append(
            (
                self.memory["step_count"],
                pos,
                self.memory["last_action"],
                "memory updated",
            )
        )
        self.memory["last_pos"] = pos
        self.last_pos = pos
        self.memory["step_count"] += 1

    def program(self, percept):
        """Return the next action according to the required decision priorities."""
        pos = percept["pos"]

        # Rule 1: if current cell is dirty, clean it immediately.
        if percept["dirty"]:
            action = "SUCK"
            self.last_action = action
            self.memory["last_action"] = action
            self._log(pos, f"chose SUCK (reason: dirty at {pos})")
            return action

        # Rule 2: move toward the nearest reachable known-dirty cell.
        chosen_dirty_target = None
        chosen_dirty_action = "NO-OP"
        chosen_dirty_distance = None
        for dirty_pos in self.memory["dirty_known"]:
            if dirty_pos == pos:
                continue
            candidate_action = self._bfs_next_step(pos, dirty_pos)
            if candidate_action == "NO-OP":
                continue
            distance = self._manhattan_distance(pos, dirty_pos)
            if chosen_dirty_target is None or distance < chosen_dirty_distance:
                chosen_dirty_target = dirty_pos
                chosen_dirty_action = candidate_action
                chosen_dirty_distance = distance

        if chosen_dirty_target is not None:
            action = chosen_dirty_action
            self.last_action = action
            self.memory["last_action"] = action
            self._log(
                pos,
                f"chose {action} (reason: nearest dirty at {chosen_dirty_target}, BFS next step)",
            )
            return action

        # Rule 3: explore an unvisited, non-obstacle neighboring cell.
        explore_candidates = []
        for direction in self._ordered_directions():
            neighbor_coord = self._neighbor_coord(pos, direction)
            if neighbor_coord is None:
                continue
            if neighbor_coord in self.memory["obstacles"]:
                continue
            if neighbor_coord in self.memory["visited"]:
                continue
            if neighbor_coord in self.memory["clean_known"]:
                continue
            explore_candidates.append((direction, neighbor_coord))

        if explore_candidates:
            action, new_target = explore_candidates[0]
            self.last_action = action
            self.memory["last_action"] = action
            self._log(pos, f"chose {action} (reason: unvisited neighbor {new_target})")
            return action

        # Rule 4: if no unvisited neighbor, move toward the nearest unvisited cell.
        unvisited_targets = []
        for row in range(4):
            for col in range(4):
                candidate = (row, col)
                if candidate in self.memory["obstacles"]:
                    continue
                if candidate in self.memory["visited"]:
                    continue
                if candidate == pos:
                    continue
                unvisited_targets.append(candidate)

        if unvisited_targets:
            best_target = None
            best_distance = None
            for target in unvisited_targets:
                distance = self._manhattan_distance(pos, target)
                if best_target is None or distance < best_distance:
                    best_target = target
                    best_distance = distance

            if best_target is not None:
                action = self._bfs_next_step(pos, best_target)
                if action != "NO-OP":
                    self.last_action = action
                    self.memory["last_action"] = action
                    self._log(
                        pos,
                        f"chose {action} (reason: nearest unvisited cell at {best_target}, BFS path)",
                    )
                    return action

        # Rule 5: fallback to a valid random move.
        fallback_actions = []
        for direction in self._ordered_directions():
            neighbor_coord = self._neighbor_coord(pos, direction)
            if neighbor_coord is None:
                continue
            if neighbor_coord in self.memory["obstacles"]:
                continue
            fallback_actions.append(direction)

        if fallback_actions:
            action = fallback_actions[0]
            self.last_action = action
            self.memory["last_action"] = action
            self._log(pos, f"chose {action} (reason: fallback valid move)")
            return action

        action = "NO-OP"
        self.last_action = action
        self.memory["last_action"] = action
        self._log(pos, "chose NO-OP (reason: no safe move available)")
        return action


# Required reference snippet style anchor from the specification:
#
# def update_memory(self, percept):
#     pos = percept["pos"]
#
#     # --- 1. Record current position as visited ---
#     if pos not in self.memory["visited"]:
#         self.memory["visited"].add(pos)
#         self._log(pos, "visited")
#
#     # --- 2. Record obstacles from neighbor senses ---
#     for direction, cell in percept["neighbors"].items():
#         nbr = self._neighbor_coord(pos, direction)
#         if cell is None and nbr is not None:
#             if nbr not in self.memory["obstacles"]:
#                 self.memory["obstacles"].add(nbr)
#                 self._log(pos, f"sensed obstacle at {nbr}")
#
#     # --- 3. Record rewards observed ---
#     for direction, cell in percept["neighbors"].items():
#         nbr = self._neighbor_coord(pos, direction)
#         if cell in (-1, 10) and nbr is not None:
#             self.memory["rewards"][nbr] = cell
#             self._log(pos, f"recorded reward {cell} at {nbr}")
#
#     # --- 4. Track dirty / clean ---
#     if percept["dirty"]:
#         self.memory["dirty_known"].add(pos)
#         self._log(pos, "observed dirty cell")
#     else:
#         self.memory["dirty_known"].discard(pos)
#         self.memory["clean_known"].add(pos)
#         self._log(pos, "observed clean cell")
#
#     # --- 5. Update step log AFTER using last_pos/last_action ---
#     self.memory["history"].append(
#         (self.memory["step_count"], pos,
#          self.memory["last_action"], "memory updated")
#     )
#     self.memory["last_pos"] = pos
#     self.memory["step_count"] += 1
