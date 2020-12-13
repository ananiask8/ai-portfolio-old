package wumpus.agent;

import wumpus.world.*;

import javax.vecmath.Point2i;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

public class MonteCarloTreeSearch {
    WorldState state;
    double limit;
    int maxActions;
    Node root;
    int horizon;

    public MonteCarloTreeSearch(WorldState state, int maxActions) {
        this.state = state;
        this.maxActions = maxActions;
        this.limit = maxActions > 50 ? 0.85 : 1.15;
        this.horizon = 30;
    }

    public Action getBestAction() {
        root = new Node(state, null, null, 0);
        long init = System.nanoTime();
        int initHorizon = horizon;
        while ((System.nanoTime() - init)/1e9 < limit) {
            horizon = initHorizon;
            Node v = treePolicy(root);
            double r = defaultPolicy(v);
            backup(v.getParent(), r);
        }
        Action action = root.getBestChild(0).getAction();
        return action;
    }

    private Node treePolicy(Node v) {
        Node u = null;
        while (u == null || (!WorldModel.isTerminal(v.getState()) && !isGoldFound(u.getState(), v.getState()) && horizon-- > 0)) {
            u = v;
            if (!v.isExpanded()) return v.expand();
            else v = v.getBestChild(300); // with 30 horizon and 400 Cp I got the 7 points, for seed 120
        }
        return v;
    }

    private Action performRandomMove() {
        int rnd = new Random().nextInt(WorldModel.getActions(state).size());
        int i = 0;
        for (Action a: WorldModel.getActions(state)) if (i++ == rnd) return a;
        return null;
    }

    private void backup(Node v, double r) {
        if (v == null) return;
        v.update(r);
        Node parent = v.getParent();
        backup(parent, r);
    }

    private boolean isGoldFound(WorldState oldState, WorldState newState) {
        return oldState == null || oldState.getMap()[newState.getX()][newState.getY()] == CellContent.GOLD;
    }

    private double defaultPolicy(Node v) {
        WorldState s = v.getState();
        double r = v.getReward();
        while (!WorldModel.isTerminal(s) && horizon-- > 0) {
            Action a = performRandomMove();
            Outcome o = WorldModel.performAction(s, a, new Random());
            r += o.reward;
//            if (isGoldFound(s, o.state)) break;
            s = o.state;
        }
        return r;
    }
}
