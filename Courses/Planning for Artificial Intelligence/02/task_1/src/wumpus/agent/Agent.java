package wumpus.agent;

import javafx.scene.control.Cell;
import wumpus.world.*;
import wumpus.world.WorldState;


import javax.vecmath.Point2i;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;


public class Agent implements WorldModel.Agent {

	private Action[][] policy;

	/**
	 * This method is called when the simulation engine requests the next action.
	 * You are given a position of the robot and the map of the environment.
	 *
	 * The top-left corner has coordinates (0,0). 
	 * 
	 * You can check whether there is an obstacle on a particular cell of the map 
	 * by querying map[x][y] == wumpus.world.CellContent.OBSTACLE.
	 *
	 * There is one gold on the map. You can query whether a position contains gold by
	 * querying map[x][y] == wumpus.world.CellContent.GOLD.
	 * 
	 * Further, there are several pits on the map. You can query whether a position contains pit by
	 * querying map[x][y] == wumpus.world.CellContent.PIT.
	 *
	 * @return action to perform in the next step
	 */

	public Action nextStep(WorldState state) {
		//---- value iteration ----
		if (policy == null) {
			DebugVis.initVis();
			policy = computePolicy(state.getMap());
		}

		return policy[state.getAgent().x][state.getAgent().y];
	}

	private int[] getGoldPos(CellContent[][] map) {
        int cols = map.length; // the number of columns in the x-dimension,
        int rows = map[0].length; // the number of rows in the y-dimension

        int[] goldPos = new int[2];
        out:
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                if (map[i][j] == CellContent.GOLD) {
                    goldPos[0] = i;
                    goldPos[1] = j;
                    break out;
                }
            }
        }

        return goldPos;
    }

    private float maxNorm(float[][] oldUtilities, float[][] newUtilities) {
        int cols = oldUtilities.length; // the number of columns in the x-dimension,
        int rows = oldUtilities[0].length; // the number of rows in the y-dimension

        float max = 0;
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                float temp = (float) Math.abs(newUtilities[i][j] - oldUtilities[i][j]);
                if (temp > max) max = temp;
            }
        }

        return max;
    }

    private ArrayList<int[]> getSuccessors(CellContent[][] map, int[] currentPos) {
        int cols = map.length; // the number of columns in the x-dimension,
        int rows = map[0].length; // the number of rows in the y-dimesnion

        ArrayList<int[]> children = new ArrayList<>();
        for (Action a: Action.values()) {
            if (a == Action.SOUTH && currentPos[1] + 1 < rows) {
                int[] newPos = new int[] {currentPos[0], currentPos[1] + 1};

                if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && map[newPos[0]][newPos[1]] != CellContent.PIT)
                    children.add(newPos);
            } else if (a == Action.EAST && currentPos[0] + 1 < cols) {
                int[] newPos = new int[] {currentPos[0] + 1, currentPos[1]};

                if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && map[newPos[0]][newPos[1]] != CellContent.PIT)
                    children.add(newPos);
            } else if (a == Action.NORTH && currentPos[1] - 1 >= 0) {
                int[] newPos = new int[] {currentPos[0], currentPos[1] - 1};

                if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && map[newPos[0]][newPos[1]] != CellContent.PIT)
                    children.add(newPos);
            } else if (a == Action.WEST && currentPos[0] - 1 >= 0) {
                int[] newPos = new int[] {currentPos[0] - 1, currentPos[1]};

                if (map[newPos[0]][newPos[1]] != CellContent.OBSTACLE && map[newPos[0]][newPos[1]] != CellContent.PIT)
                    children.add(newPos);
            }
        }

        return children;
    }

    private float[][] computeValueIteration(CellContent[][] map, int[] goldPos, float[][] oldUtilities) {
        int cols = map.length; // the number of columns in the x-dimension,
        int rows = map[0].length; // the number of rows in the y-dimesnion

        float[][] newUtilities = oldUtilities;
        Queue<int[]> q = new LinkedList<>();
        boolean[][] visited = new boolean[cols][rows];
        visited[goldPos[0]][goldPos[1]] = true;

        q.add(goldPos);
        while (!q.isEmpty()) {
            int[] currentPos = q.poll();

            for (int[] child: getSuccessors(map, currentPos)) {
                if (visited[child[0]][child[1]]) continue;
                visited[child[0]][child[1]] = true;
                q.add(child);

                // Algorithm actually requires to first calculate the potential utilities, and select the action that maximizes the value
                // In this case, not necessarily, the value optimal utility resulting from the previous iteration needs to be the new one
                //  Action best = getBestAction(child, oldUtilities);
                float maxUtility = Float.MIN_VALUE;
                for (Action a: Action.values()) {
                    Set<Transition> transitions = WorldModel.getTransitions(new WorldState(new Point2i(child[0], child[1]), new Point2i[0], map), a);
                    float u = 0;
                    for (Transition transition : transitions) {
                        float normalizer = map[transition.successorState.getX()][transition.successorState.getY()] == CellContent.GOLD ? /*-100*/ 0 : 0;
                        u += transition.probability * (transition.reward + normalizer + oldUtilities[transition.successorState.getX()][transition.successorState.getY()]);
                    }
                    maxUtility = Math.max(maxUtility, u);
                }
                newUtilities[child[0]][child[1]] = maxUtility;
            }
        }

        return newUtilities;
    }

    private Action[][] retrievePolicy(CellContent[][] map, float[][] utilities) {
        int cols = utilities.length; // the number of columns in the x-dimension,
        int rows = utilities[0].length; // the number of rows in the y-dimension
        Action[][] policy = new Action[cols][rows];
        for (int i = 0; i < cols; i++){
            for (int j = 0; j < rows; j++) {
                policy[i][j] = Action.EAST;

                float maxUtility = Float.MIN_VALUE;
                for (Action a: Action.values()) {
                    Set<Transition> transitions = WorldModel.getTransitions(new WorldState(new Point2i(i, j), new Point2i[0], map), a);
                    float u = 0;
                    for (Transition transition : transitions) {
                        u += transition.probability * (transition.reward + utilities[transition.successorState.getX()][transition.successorState.getY()]);
                    }
                    if (maxUtility < u) {
                        maxUtility = u;
                        policy[i][j] = a;
                    }
                }
            }
        }

        return policy;
    }

    private float[][] initializeUtilities(CellContent[][] map) {
        int cols = map.length; // the number of columns in the x-dimension,
        int rows = map[0].length; // the number of rows in the y-dimension

        float[][] utilities = new float[cols][rows];
        for (int i = 0; i < cols; i++){
            for (int j = 0; j < rows; j++) {
                switch(map[i][j]) {
                    case PIT:
                        utilities[i][j] = -100;
                        break;
                    case OBSTACLE:
                    case EMPTY:
                        utilities[i][j] = 0;
                        break;
                    case GOLD:
                        utilities[i][j] = 100;
                        break;
                }
            }
        }

        return utilities;
    }

	/**
	 * Compute an optimal policy for the agent.
	 * @param map map of the environment
	 * @return an array that contains for each cell of the environment one action,
	 * i.e. one of: Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST.
	 */
	private Action[][] computePolicy(CellContent[][] map) {
		// the size of the map can be obtained as
		int cols = map.length; // the number of columns in the x-dimension,
		int rows = map[0].length; // the number of rows in the y-dimension

		/**** YOUR CODE HERE *****/
        float[][] oldUtilities;
        float[][] newUtilities = initializeUtilities(map);
        int[] goldPos = getGoldPos(map);
        float eps = (float) 0.01;
        int h = 200;
        for (int i = 0; i < h; i++) {
            oldUtilities = newUtilities;
            newUtilities = computeValueIteration(map, goldPos, oldUtilities);
//            if (maxNorm(oldUtilities, newUtilities) < eps) break;
        }
        Action[][] policy = retrievePolicy(map, newUtilities);

        // Debug stuff
		DebugVis.setStateValues(newUtilities);
		float[][][] stateActionValues = new float[cols][rows][Action.values().length];
		DebugVis.setStateActionValues(stateActionValues);
		DebugVis.setPolicy(policy);

		return policy;
	}



}
