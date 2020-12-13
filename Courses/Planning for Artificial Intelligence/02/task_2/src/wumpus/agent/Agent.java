package wumpus.agent;

import wumpus.world.*;
import wumpus.world.WorldState;


import javax.vecmath.Point2i;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

public class Agent implements WorldModel.Agent {
	int maxActions = -1;
	/**
	 * This method is called when the simulation engine requests the next action.
	 * You are given a state of the world that consists of position of agent, positions of wumpuses, and the map of world.
	 *
	 * The top-left corner has coordinates (0,0). 
	 *
	 * You can check the current position of agent through state.getAgent(), the positions of all
	 * wumpuses can be obtained via state.getWumpuses() and the map of world through state.getMap().
	 *
	 *
	 * You can check whether there is an obstacle on a particular cell of the map 
	 * by querying state.getMap()[x][y] == CellContent.OBSTACLE.
	 *
	 * There is one gold on the map. You can query whether a position contains gold by
	 * querying state.getMap()[x][y] == CellContent.GOLD.
	 * 
	 * Further, there are several pits on the map. You can query whether a position contains pit by
	 * querying state.getMap()[x][y] == CellContent.PIT.
	 *
	 * @return action to perform in the next step
	 */

	public Action nextStep(WorldState state) {
		// return random action
//		System.out.println(state);
		if(maxActions == -1) maxActions = state.getActionsLeft();
		MonteCarloTreeSearch mcts = new MonteCarloTreeSearch(state, maxActions);
		System.out.println("=====");

//		MonteCarloTreeSearch mcts2 = new MonteCarloTreeSearch(state, maxActions);
//		System.out.println(mcts.getBestAction() + " " + mcts2.getBestAction());
		Action a = mcts.getBestAction();
//		Q = mcts.getQ();
		System.out.println("(" + state.getX() + ", " + state.getY() + "): " + a);
		return a;
	}
}
