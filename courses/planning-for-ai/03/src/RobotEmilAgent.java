import java.util.*;

import com.sun.xml.internal.xsom.impl.scd.Iterators;


public class RobotEmilAgent {
    ArrayList<Action> policy = new ArrayList<>();
    int[] prev = new int[] {-1, -1};
    int policyTraversalIndex = 0;

	public ArrayList<Action> dijkstra(CellContent[][] map, int[] s, int[] t) {
		int cols = map.length; // the number of columns in the x-dimension,
		int rows = map[0].length; // the number of rows in the y-dimesnion

		Queue<int[]> q = new LinkedList<>();
		q.add(s);
		ArrayList<ArrayList<ArrayList<Action>>> plan = new ArrayList<>(); // init should be infinite length
		for (int j=0; j<cols; j++) {
			plan.add(new ArrayList<>());
			for (int i = 0; i < rows; i++) {
				plan.get(j).add(new ArrayList<>());
			}
		}

        while (!q.isEmpty()) {
			int[] currentPos = q.poll();
			int[] newPos = currentPos;

			for (Action a: Action.values()) {
                if (a == Action.SOUTH && currentPos[1] + 1 < rows) {
                    newPos = new int[] {currentPos[0], currentPos[1] + 1};

                    if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && (plan.get(newPos[0]).get(newPos[1]).size() == 0 ||
							plan.get(currentPos[0]).get(currentPos[1]).size() + 1 < plan.get(newPos[0]).get(newPos[1]).size())) {
						q.add(newPos);
						plan.get(newPos[0]).set(newPos[1], new ArrayList<>(plan.get(currentPos[0]).get(currentPos[1])));
                        plan.get(newPos[0]).get(newPos[1]).add(Action.SOUTH);
					}
				} else if (a == Action.EAST && currentPos[0] + 1 < cols) {
                    newPos = new int[] {currentPos[0] + 1, currentPos[1]};

                    if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && (plan.get(newPos[0]).get(newPos[1]).size() == 0 ||
							plan.get(currentPos[0]).get(currentPos[1]).size() + 1 < plan.get(newPos[0]).get(newPos[1]).size())) {
						q.add(newPos);
						plan.get(newPos[0]).set(newPos[1], new ArrayList<>(plan.get(currentPos[0]).get(currentPos[1])));
                        plan.get(newPos[0]).get(newPos[1]).add(Action.EAST);
					}
				} else if (a == Action.NORTH && currentPos[1] - 1 >= 0) {
                    newPos = new int[] {currentPos[0], currentPos[1] - 1};

                    if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && (plan.get(newPos[0]).get(newPos[1]).size() == 0 ||
							plan.get(currentPos[0]).get(currentPos[1]).size() + 1 < plan.get(newPos[0]).get(newPos[1]).size())) {
						q.add(newPos);
						plan.get(newPos[0]).set(newPos[1], new ArrayList<>(plan.get(currentPos[0]).get(currentPos[1])));
                        plan.get(newPos[0]).get(newPos[1]).add(Action.NORTH);
					}
				} else if (a == Action.WEST && currentPos[0] - 1 >= 0) {
                    newPos = new int[] {currentPos[0] - 1, currentPos[1]};

                    if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && (plan.get(currentPos[0] - 1).get(currentPos[1]).size() == 0 ||
							plan.get(currentPos[0]).get(currentPos[1]).size() + 1 < plan.get(currentPos[0] - 1).get(currentPos[1]).size())) {
						q.add(newPos);
						plan.get(newPos[0]).set(newPos[1], new ArrayList<>(plan.get(currentPos[0]).get(currentPos[1])));
                        plan.get(newPos[0]).get(newPos[1]).add(Action.WEST);
					}
				}
			}
		}

		return plan.get(t[0]).get(t[1]);
	}

	public ArrayList<Action> buildPlan(CellContent[][] map, int x, int y) {
		int cols = map.length; // the number of columns in the x-dimension,
		int rows = map[0].length; // the number of rows in the y-dimesnion

		int[] goldPos = new int[2];
		out:
		for (int j=0; j<cols; j++) {
			for (int i=0; i<rows; i++) {
				if (map[j][i] == CellContent.GOLD) {
					goldPos[0] = j;
					goldPos[1] = i;
					break out;
				}
			}
		}

        return dijkstra(map, new int[] {x, y}, goldPos);
	}

	public boolean wasMoveSuccessful(int[] previous, int[] current, Action a) {
	    switch (a) {
            case EAST: return previous[0] + 1 == current[0] && previous[1] == current[1];
            case WEST: return previous[0] - 1 == current[0] && previous[1] == current[1];
            case NORTH: return previous[0] == current[0] && previous[1] - 1 == current[1];
            case SOUTH: return previous[0] == current[0] && previous[1] + 1 == current[1];
        }

        return false;
    }

	/**
	 * This method is called after when a simulation engine request next action.
	 * You are given a position of the robot and the map of the environment.
	 *
	 * The top-left corner has coordinates (0,0). You can check whether
	 * there is an obstacle by querying map[x][y] == CellContent.Obstacle.
	 *
	 * There is one gold on the map. You can query whether a position contains gold by
	 * querying map[x][y] == CellContent.Gold.
	 *
	 * @param x the x-coordinate of the current position of robot
	 * @param y the y-coordinate of the current position of robot
	 * @param map the map of the environment
	 * @return action to perform in the next step
	 */
	public Action nextStep(int x, int y, CellContent[][] map) {
        int[] current = new int[] {x, y};
		if (policy.size() > 0 && policyTraversalIndex < policy.size() && wasMoveSuccessful(prev, current, policy.get(policyTraversalIndex - 1))) {
            prev = current;
            return policy.get(policyTraversalIndex++);
        }
        policy = buildPlan(map, x, y);
        prev = current;
        policyTraversalIndex = 0;
        return policy.get(policyTraversalIndex++);
	}
}
